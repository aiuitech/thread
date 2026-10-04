/**
 * @experimental Write path, implemented against docs/update-path.md: an upsert encodes one delta segment
 * (`DeltaJob`: primary cell + residual codes against the cell mean, PQ codes with the generation's codebook)
 * and a tombstone set for its ids, a delete just a tombstone set; both commit a new manifest revision with a
 * create-only put of `manifest.{rev+1}.json` (the lock) and advance `idx/current.json` with a compare-and-swap.
 * The Worker's own implementation landed concurrently; the two must stay wire-compatible (see README).
 */
import type { Thread } from "./index.js";
import { asF32 } from "./index.js";
import { safeFree } from "./query.js";
import type { BlobStore } from "./store.js";
import type * as W from "./wasm/thread.js";
import { idUnion, keys, manifestFiles, parseCurrent, parseManifest, ThreadError, type CurrentPointer, type IndexFiles, type Manifest } from "./wire.js";

export interface UpsertInput {
	ids: ArrayLike<number>;
	/** `n * dim` floats (flat) or one array per vector; any norm (normalized here). */
	vectors: Float32Array | number[] | (Float32Array | number[])[];
	/** One u32 per vector (default 0). */
	attrs?: ArrayLike<number>;
	/** One payload per vector (strings are UTF-8 encoded). */
	payloads?: (Uint8Array | string)[];
	/**
	 * Also write a tombstone set for the ids so older copies stop resolving (default true). Pass false when the
	 * ids are known to be new: no tombstones to filter on every later query.
	 */
	replace?: boolean;
}

export interface UpsertResult {
	gen: string;
	rev: number;
	seq: number;
	count: number;
	/** Primary cell of each vector. */
	cells: number[];
	distinctCells: number;
	keys: { delta: string; tombstones: string | null; manifest: string };
	bytes: { delta: number; tombstones: number; manifest: number };
	/** Lost manifest races and current.json CAS retries. */
	conflicts: number;
	casConflicts: number;
	/** Whether current.json now points at this revision (false if another generation is current). */
	currentUpdated: boolean;
	ms: number;
}

export interface DeleteResult {
	gen: string;
	rev: number;
	count: number;
	key: string;
	conflicts: number;
	casConflicts: number;
	currentUpdated: boolean;
	ms: number;
}

export interface WriterContext {
	thread: Thread;
	w: typeof W;
	store: BlobStore;
	gen: string;
	/** The index's loaded PqCodebook for these base files (if any), to avoid re-parsing 1.5 MB. */
	borrowPq: (baseKey: string) => W.PqCodebook | null;
}

export const MAX_UPSERT = 4096;
const COMMIT_ATTEMPTS = 12;
/** Vectors passed to DeltaJob.add per call (bounds the batch copy in wasm memory). */
const DELTA_ADD_CHUNK = 256;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function writable(store: BlobStore): Required<Pick<BlobStore, "put">> & BlobStore {
	if (!store.put) throw new ThreadError("this store is read-only (no put)", 400);
	return store as Required<Pick<BlobStore, "put">> & BlobStore;
}

/** PUT only if the key does not exist yet. */
async function putIfAbsent(store: BlobStore, key: string, data: Uint8Array | string, contentType?: string): Promise<boolean> {
	return (await writable(store).put(key, data, { ifAbsent: true, contentType })) !== null;
}

/** Writes an immutable object at `key`, or at a unique sibling `{stem}.{nonce}{ext}` if the key is taken. */
async function putUnique(store: BlobStore, key: string, data: Uint8Array | string): Promise<string> {
	if (await putIfAbsent(store, key, data)) return key;
	const dot = key.lastIndexOf(".");
	for (let i = 0; i < 4; i++) {
		const alt = `${key.slice(0, dot)}.${crypto.randomUUID().slice(0, 8)}${key.slice(dot)}`;
		if (await putIfAbsent(store, alt, data)) return alt;
	}
	throw new ThreadError(`could not write a unique key for ${key}`);
}

async function readCurrent(store: BlobStore): Promise<{ cur: CurrentPointer; etag: string } | null> {
	// head first: if the pointer changes between the two reads the CAS fails and the writer re-reads.
	const h = await store.head(keys.current());
	if (!h) return null;
	const b = await store.get(keys.current());
	if (!b) return null;
	return { cur: parseCurrent(b), etag: h.etag };
}

/** Moves idx/current.json forward to (gen, rev) with a CAS loop; leaves it alone if it names another generation or a newer rev. */
async function advanceCurrent(store: BlobStore, gen: string, rev: number): Promise<{ updated: boolean; conflicts: number }> {
	let conflicts = 0;
	for (let attempt = 0; attempt < 10; attempt++) {
		const r = await readCurrent(store);
		const body = JSON.stringify({ gen, rev });
		if (!r) {
			if (await putIfAbsent(store, keys.current(), body, "application/json")) return { updated: true, conflicts };
		} else {
			if (r.cur.gen !== gen || r.cur.rev >= rev) return { updated: false, conflicts };
			if (await writable(store).put(keys.current(), body, { ifMatch: r.etag, contentType: "application/json" })) return { updated: true, conflicts };
		}
		conflicts++;
		await sleep(20 + Math.random() * 50);
	}
	throw new ThreadError(`idx/current.json: gave up after ${conflicts} CAS conflicts`, 409);
}

/** The newest manifest revision in the store (a crashed writer's manifest beyond current.json counts). */
async function latestRev(store: BlobStore, gen: string): Promise<number> {
	if (!store.list) throw new ThreadError("this store cannot list objects, which the writer needs after a lost race", 400);
	const prefix = `idx/${gen}/manifest.`;
	let head = -1;
	for (const key of await store.list(prefix)) {
		const m = key.slice(prefix.length).match(/^(?:(\d+)\.)?json$/);
		if (m) head = Math.max(head, m[1] ? Number(m[1]) : 0);
	}
	if (head < 0) throw new ThreadError(`no manifest for generation ${gen}`, 404);
	return head;
}

interface CommitResult {
	rev: number;
	manifest: Manifest;
	manifestKey: string;
	manifestBytes: number;
	conflicts: number;
	casConflicts: number;
	currentUpdated: boolean;
}

/**
 * Commits a new revision: `make` receives the head (rev, manifest) and returns the next manifest (writing the
 * objects it references first). The create-only put of `manifest.{rev+1}.json` serializes writers; a lost
 * race rebuilds on the new head.
 */
async function commitRev(ctx: WriterContext, make: (head: { rev: number; manifest: Manifest; files: IndexFiles }) => Promise<Manifest>): Promise<CommitResult> {
	const { store, gen } = ctx;
	let conflicts = 0;
	let useList = false;
	for (let attempt = 1; attempt <= COMMIT_ATTEMPTS; attempt++) {
		const cur = await readCurrent(store);
		const head = !useList && cur && cur.cur.gen === gen ? cur.cur.rev : await latestRev(store, gen);
		const mb = await store.get(keys.manifest(gen, head));
		if (!mb) throw new ThreadError(`no manifest ${keys.manifest(gen, head)}`, 404);
		const base = parseManifest(mb, gen, head);
		const next = await make({ rev: head, manifest: base, files: manifestFiles(base) });
		const rev = head + 1;
		const manifest: Manifest = { ...next, gen, rev, parentRev: head, updatedAt: new Date().toISOString() };
		const body = JSON.stringify(manifest);
		const key = keys.manifest(gen, rev);
		if (await putIfAbsent(store, key, body, "application/json")) {
			const c = await advanceCurrent(store, gen, rev);
			return { rev, manifest, manifestKey: key, manifestBytes: body.length, conflicts, casConflicts: c.conflicts, currentUpdated: c.updated };
		}
		conflicts++;
		useList = true;
		await sleep(10 + Math.random() * 40 * attempt);
	}
	throw new ThreadError(`${gen}: lost the manifest race ${conflicts} times`, 409);
}

/** The fields a new revision inherits (build notes and the previous change stay with their revision). */
function inherit(m: Manifest): Manifest {
	const { notes: _n, change: _c, rev: _r, parentRev: _p, updatedAt: _u, ...rest } = m;
	return { ...rest, files: manifestFiles(m), deltas: m.deltas ?? [], tombstones: m.tombstones ?? [], nextSeq: m.nextSeq ?? 1 };
}

function u32(a: ArrayLike<number>, name: string): Uint32Array {
	for (let i = 0; i < a.length; i++) if (!Number.isInteger(a[i]) || a[i] < 0 || a[i] > 0xffffffff) throw new ThreadError(`${name}[${i}] is not a u32`, 400);
	return Uint32Array.from(a as ArrayLike<number>);
}

/** Validates the batch and returns ids and unit-normalized vectors (a copy). */
function prepareBatch(input: UpsertInput, dim: number): { ids: Uint32Array; vectors: Float32Array; attrs: Uint32Array | null; payloads: Uint8Array[] | null } {
	const ids = u32(input.ids, "ids");
	const n = ids.length;
	if (n === 0) throw new ThreadError("empty batch", 400);
	if (n > MAX_UPSERT) throw new ThreadError(`at most ${MAX_UPSERT} vectors per upsert`, 400);
	if (new Set(ids).size !== n) throw new ThreadError("duplicate ids in the batch", 400);
	const v = input.vectors;
	let vectors: Float32Array;
	if (Array.isArray(v) && v.length && typeof v[0] !== "number") {
		const rows = v as (Float32Array | number[])[];
		if (rows.length !== n) throw new ThreadError(`${rows.length} vectors for ${n} ids`, 400);
		vectors = new Float32Array(n * dim);
		rows.forEach((r, i) => {
			if (r.length !== dim) throw new ThreadError(`vector ${i} has ${r.length} dims, index dim is ${dim}`, 400);
			vectors.set(r, i * dim);
		});
	} else {
		vectors = Float32Array.from(v as ArrayLike<number>);
		if (vectors.length !== n * dim) throw new ThreadError(`vectors has ${vectors.length} floats, expected ${n} x ${dim}`, 400);
	}
	for (let i = 0; i < n; i++) {
		let ss = 0;
		for (let j = i * dim; j < (i + 1) * dim; j++) ss += vectors[j] * vectors[j];
		if (!(ss > 0) || !Number.isFinite(ss)) throw new ThreadError(`vector ${i} (id ${ids[i]}) has zero or non-finite norm`, 400);
		const inv = 1 / Math.sqrt(ss);
		for (let j = i * dim; j < (i + 1) * dim; j++) vectors[j] *= inv;
	}
	const attrs = input.attrs ? u32(input.attrs, "attrs") : null;
	if (attrs && attrs.length !== n) throw new ThreadError(`${attrs.length} attrs for ${n} vectors`, 400);
	const enc = new TextEncoder();
	const payloads = input.payloads ? input.payloads.map((p) => (typeof p === "string" ? enc.encode(p) : p)) : null;
	if (payloads && payloads.length !== n) throw new ThreadError(`${payloads.length} payloads for ${n} vectors`, 400);
	return { ids, vectors, attrs, payloads };
}

/** The first free delta seq >= `from` (HEAD probes, so a crashed writer's segment is skipped). */
async function allocSeq(store: BlobStore, gen: string, from: number): Promise<number> {
	let seq = Math.max(1, from);
	for (let i = 0; i < 64 && (await store.head(keys.delta(gen, seq))); i++) seq++;
	return seq;
}

interface EncodedDelta {
	seq: number;
	key: string;
	bytes: number;
	cells: Uint32Array;
	nCells: number;
	filesKey: string;
}

/** Encodes and writes one delta segment against the generation's fine means (and PQ codebook). */
async function encodeDelta(ctx: WriterContext, m: Manifest, files: IndexFiles, b: ReturnType<typeof prepareBatch>): Promise<EncodedDelta> {
	const { w, store, gen } = ctx;
	const format = m.format ?? 1;
	if (format === 3 && !files.pq) throw new ThreadError(`format 3 generation ${gen} has no PQ codebook`);
	const baseKey = `${gen}|${files.coarse}|${files.fine}|${files.pq ?? "-"}`;
	const borrowed = ctx.borrowPq(baseKey);
	const [seq, meansBytes, pqBytes] = await Promise.all([allocSeq(store, gen, m.nextSeq ?? 1), store.get(files.fine), format === 3 && !borrowed ? store.get(files.pq!) : Promise.resolve(null)]);
	if (!meansBytes) throw new ThreadError(`missing ${files.fine}`, 404);
	const k = m.k1 * m.k2;
	const means = asF32(meansBytes, files.fine);
	if (means.length !== k * m.dim) throw new ThreadError(`${files.fine} has ${means.length / m.dim} means, expected ${k}`);
	let ownPq: W.PqCodebook | null = null;
	if (format === 3 && !borrowed) {
		if (!pqBytes) throw new ThreadError(`missing ${files.pq}`, 404);
		ownPq = new w.PqCodebook(pqBytes);
	}
	const epoch = ctx.thread.epoch;
	const free = (o: { free(): void }) => epoch === ctx.thread.epoch && safeFree(o);
	const job = new w.DeltaJob(m.dim, k, means, seq); // DeltaJob holds its own copy of the means (~k*dim*4 bytes)
	let finished = false;
	let bytes: Uint8Array;
	const cells = new Uint32Array(b.ids.length);
	try {
		if (format === 3) job.set_pq_codebook((borrowed ?? ownPq)!);
		const dim = m.dim;
		for (let a = 0; a < b.ids.length; a += DELTA_ADD_CHUNK) {
			const e = Math.min(b.ids.length, a + DELTA_ADD_CHUNK);
			const v = b.vectors.subarray(a * dim, e * dim);
			const attrs = b.attrs ? b.attrs.subarray(a, e) : new Uint32Array(0);
			let c: Uint32Array;
			if (b.payloads) {
				const ps = b.payloads.slice(a, e);
				const lens = Uint32Array.from(ps, (p) => p.byteLength);
				const cat = new Uint8Array(lens.reduce((x, y) => x + y, 0));
				let o = 0;
				for (const p of ps) {
					cat.set(p, o);
					o += p.byteLength;
				}
				c = job.add_with_payloads(v, b.ids.subarray(a, e), attrs, cat, lens);
			} else c = job.add(v, b.ids.subarray(a, e), attrs);
			if (c.length !== e - a) throw new ThreadError(`DeltaJob added ${c.length} of ${e - a} vectors`);
			cells.set(c, a);
		}
		finished = true;
		bytes = job.finish(); // consumes the job
	} finally {
		if (!finished) free(job);
		if (ownPq) free(ownPq);
	}
	const nCells = w.delta_cells(bytes).length / 2;
	const key = await putUnique(store, keys.delta(gen, seq), bytes);
	return { seq, key, bytes: bytes.byteLength, cells, nCells, filesKey: baseKey };
}

/** Sorted unique ids as a tombstone buffer (`tombstones_merge` of the empty set and `ids`). */
const tombBytes = (w: typeof W, ids: Uint32Array): Uint8Array => (ids.length ? w.tombstones_merge(new Uint8Array(0), ids) : idUnion([]));

export async function upsert(ctx: WriterContext, input: UpsertInput): Promise<UpsertResult> {
	const t0 = Date.now();
	const { store, gen, w } = ctx;
	writable(store);
	const replace = input.replace ?? true;
	return ctx.thread.withRecovery("upsert", async () => {
		const st: { batch: ReturnType<typeof prepareBatch> | null; delta: EncodedDelta | null; tombKey: string | null; tombBytes: number } = { batch: null, delta: null, tombKey: null, tombBytes: 0 };
		const c = await commitRev(ctx, async ({ rev, manifest, files }) => {
			const batch = (st.batch ??= prepareBatch(input, manifest.dim));
			const next = inherit(manifest);
			const newRev = rev + 1;
			const filesKey = `${gen}|${files.coarse}|${files.fine}|${files.pq ?? "-"}`;
			// The segment is reused across retries unless the means/codebook changed under us; the tombstone set
			// is keyed by the rev, so it is rewritten for every attempt.
			const tb = replace ? tombBytes(w, batch.ids) : null;
			const [delta, tombKey] = await Promise.all([st.delta && st.delta.filesKey === filesKey ? st.delta : encodeDelta(ctx, manifest, files, batch), tb ? putUnique(store, keys.tombstones(gen, newRev), tb) : Promise.resolve(null)]);
			st.delta = delta;
			st.tombKey = tombKey;
			if (tb && tombKey) {
				st.tombBytes = tb.byteLength;
				next.tombstones!.push({ rev: newRev, key: tombKey, count: tb.byteLength / 4 });
			}
			next.deltas!.push({ seq: delta.seq, key: delta.key, count: batch.ids.length, bytes: delta.bytes, addedRev: newRev, cells: delta.nCells });
			next.nextSeq = Math.max(next.nextSeq ?? 1, delta.seq + 1);
			return { ...next, change: { op: "upsert", seq: delta.seq, count: batch.ids.length, replace, writer: "thread-sdk" } };
		});
		const d = st.delta!;
		return {
			gen,
			rev: c.rev,
			seq: d.seq,
			count: st.batch!.ids.length,
			cells: Array.from(d.cells),
			distinctCells: d.nCells,
			keys: { delta: d.key, tombstones: st.tombKey, manifest: c.manifestKey },
			bytes: { delta: d.bytes, tombstones: st.tombBytes, manifest: c.manifestBytes },
			conflicts: c.conflicts,
			casConflicts: c.casConflicts,
			currentUpdated: c.currentUpdated,
			ms: Date.now() - t0,
		};
	});
}

export async function deleteIds(ctx: WriterContext, idsIn: ArrayLike<number>): Promise<DeleteResult> {
	const t0 = Date.now();
	const { store, gen, w } = ctx;
	writable(store);
	const ids = u32(idsIn, "ids");
	if (!ids.length) throw new ThreadError("no ids", 400);
	const tb = tombBytes(w, ids);
	let key = "";
	const c = await commitRev(ctx, async ({ rev, manifest }) => {
		const next = inherit(manifest);
		key = await putUnique(store, keys.tombstones(gen, rev + 1), tb);
		next.tombstones!.push({ rev: rev + 1, key, count: tb.byteLength / 4 });
		return { ...next, change: { op: "delete", count: tb.byteLength / 4, writer: "thread-sdk" } };
	});
	return { gen, rev: c.rev, count: tb.byteLength / 4, key, conflicts: c.conflicts, casConflicts: c.casConflicts, currentUpdated: c.currentUpdated, ms: Date.now() - t0 };
}
