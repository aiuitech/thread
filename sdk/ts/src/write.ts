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
import { defineAttrs, packAttrs, schemaExtends, validateSchema, type AttrFieldSpec, type AttrSchema, type AttrValues } from "./attrs.js";
import { chunkIndexOf, chunkKey, chunkText, resolveChunkOptions, type ChunkOptions } from "./chunk.js";
import { buildTombState, currentBody, hasDense, idUnion, keys, manifestFiles, parseCurrent, parseManifest, resolveLexConfig, tombForSegment, ThreadError, type CurrentPointer, type IndexFiles, type LexConfig, type Manifest, type ResolvedLexConfig, type SegmentRef } from "./wire.js";

export interface UpsertInput {
	/** The documents' u32 ids. Optional with `keys`: the ids of known keys are reused, new keys get fresh ones. */
	ids?: ArrayLike<number>;
	/**
	 * One unique string key per document (an external address, a path, a primary key; at most 4096 bytes), stored
	 * in the segment's `keys` section so `get(key)` / `resolveKeys` find the document. Needs `manifest.lex`. Without
	 * `ids`, each key is resolved through the head revision: a known key keeps its id (so the put replaces the
	 * document), a new key is allocated the next id from `manifest.nextId`.
	 */
	keys?: readonly string[];
	/**
	 * `n * dim` floats (flat) or one array per vector; any norm (normalized here). Optional when the generation has
	 * a lexical configuration: documents without a vector are only found by text.
	 */
	vectors?: Float32Array | number[] | (Float32Array | number[])[];
	/**
	 * One attribute word per document (default 0): field values under the generation's schema (`{tenant: 5, lang:
	 * "en"}`) or a raw u32. Stored with the vector (slab/delta entry) and in the `docs` store; what `filter` tests.
	 */
	attrs?: ArrayLike<AttrValues>;
	/** One payload per vector (strings are UTF-8 encoded). */
	payloads?: (Uint8Array | string)[];
	/**
	 * One text per id, indexed by BM25 (and stored when `manifest.lex.text`) in a lexical segment; needs
	 * `manifest.lex`. `null` indexes an empty document.
	 */
	texts?: (string | null)[];
	/** One metadata string per id for the `docs` store (a title, a JSON object, an external key); default "". */
	metas?: (string | null)[];
	/**
	 * Also write a tombstone set for the ids so older copies stop resolving (default true). Pass false when the
	 * ids are known to be new: no tombstones to filter on every later query.
	 */
	replace?: boolean;
	/**
	 * Split each text into flat word windows (`chunkText`: `words` per chunk, default 200, sharing `overlap` words
	 * with the previous one, default 0; `true` = the defaults) and write one document per chunk: chunk `i` of the
	 * document keyed `k` is stored under the key `k#i` (`chunkKey`) with the parent's meta and attrs, and its
	 * vector comes from `embed`. Needs `keys` (not `ids`) and `texts`; `vectors` cannot be given. Chunk keys of
	 * the parent that this put no longer writes (a shorter text) are tombstoned in the same revision. At most
	 * 4096 chunks per call.
	 */
	chunk?: ChunkOptions | boolean;
	/**
	 * Produces the vectors: called once with the texts to index (the chunks when chunking, else `texts`) and
	 * returns one vector per text, as `vectors` takes them (`n * dim` floats flat or one array per text). The SDK
	 * does no embedding itself; this is where an embedding API call goes.
	 */
	embed?: Embedder;
}

/** `upsert({ embed })`: the texts to embed, in order -> their vectors (`n * dim` flat or one row per text). */
export type Embedder = (texts: string[]) => Promise<Float32Array | number[] | (Float32Array | number[])[]> | Float32Array | number[] | (Float32Array | number[])[];

export interface UpsertResult {
	gen: string;
	rev: number;
	seq: number;
	count: number;
	/** The ids written, in input order (the given ones, or those resolved / allocated for `keys`). */
	ids: number[];
	/** Ids allocated to new keys. */
	allocated: number;
	/** With `chunk`: the input documents, the chunks written (= `count`) and the stale chunk keys tombstoned. */
	chunks?: { documents: number; count: number; stale: number };
	/** Primary cell of each vector (empty without vectors). */
	cells: number[];
	distinctCells: number;
	keys: { delta: string | null; segment: string | null; tombstones: string | null; manifest: string };
	bytes: { delta: number; segment: number; tombstones: number; manifest: number };
	/** Tokens indexed by the lexical segment. */
	tokens: number;
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
	/** `deleteKeys`: the keys that had no live document (nothing was written if every key was missing). */
	missing?: string[];
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
	/** Resolves keys to ids through the `keys` sections of the given (head) revision. */
	resolveKeys: (manifest: Manifest, keys: readonly string[]) => Promise<Map<string, number>>;
	/** Every live key with the prefix in the given (head) revision, in key order. */
	listKeys: (manifest: Manifest, prefix: string) => Promise<string[]>;
}

export interface ConfigureLexicalResult {
	gen: string;
	rev: number;
	config: LexConfig;
	conflicts: number;
	currentUpdated: boolean;
}

export interface ConfigureAttrsResult {
	gen: string;
	rev: number;
	schema: AttrSchema;
	conflicts: number;
	currentUpdated: boolean;
}

export interface CompactSegmentsResult {
	gen: string;
	rev: number;
	/** Segments merged (0 when nothing needed compacting; no revision is written then and `rev` is the head). */
	merged: number;
	key: string | null;
	count: number;
	bytes: number;
	/** Bytes of the segments read. */
	readBytes: number;
	conflicts: number;
	currentUpdated: boolean;
	ms: number;
}

export interface CreateLexicalOptions {
	store: BlobStore;
	gen: string;
	lex: LexConfig;
	/** Point `idx/current.json` at the new generation (default true). */
	setCurrent?: boolean;
	notes?: unknown;
}

export const MAX_UPSERT = 4096;
/** Longest key (UTF-8 bytes). */
export const MAX_KEY_BYTES = 4096;
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
async function advanceCurrent(store: BlobStore, gen: string, rev: number, manifest?: Manifest): Promise<{ updated: boolean; conflicts: number }> {
	let conflicts = 0;
	// The pointer carries a copy of the manifest it names (when small), so an open skips one GET.
	const body = currentBody(gen, rev, manifest);
	for (let attempt = 0; attempt < 10; attempt++) {
		const r = await readCurrent(store);
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
async function commitRev(ctx: WriterContext, make: (head: { rev: number; manifest: Manifest; files: IndexFiles }) => Promise<Manifest | null>): Promise<CommitResult> {
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
		if (next === null) return { rev: head, manifest: base, manifestKey: keys.manifest(gen, head), manifestBytes: mb.byteLength, conflicts, casConflicts: 0, currentUpdated: false };
		const rev = head + 1;
		const manifest: Manifest = { ...next, gen, rev, parentRev: head, updatedAt: new Date().toISOString() };
		const body = JSON.stringify(manifest);
		const key = keys.manifest(gen, rev);
		if (await putIfAbsent(store, key, body, "application/json")) {
			const c = await advanceCurrent(store, gen, rev, manifest);
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
	return { ...rest, files: manifestFiles(m), deltas: m.deltas ?? [], tombstones: m.tombstones ?? [], segments: m.segments ?? [], nextSeq: m.nextSeq ?? 1 };
}

function u32(a: ArrayLike<number>, name: string): Uint32Array {
	for (let i = 0; i < a.length; i++) if (!Number.isInteger(a[i]) || a[i] < 0 || a[i] > 0xffffffff) throw new ThreadError(`${name}[${i}] is not a u32`, 400);
	return Uint32Array.from(a as ArrayLike<number>);
}

interface Batch {
	/** The ids (resolved / allocated per attempt for a keyed batch without ids). */
	ids: Uint32Array;
	/** Whether the caller gave the ids (else they come from the keys). */
	givenIds: boolean;
	keys: string[] | null;
	/** Unit-normalized vectors (a copy), or null for a text-only batch. */
	vectors: Float32Array | null;
	attrs: Uint32Array | null;
	payloads: Uint8Array[] | null;
	texts: (string | null)[] | null;
	metas: (string | null)[] | null;
}

const utf8Len = (s: string): number => new TextEncoder().encode(s).byteLength;

/** Validates the batch and returns ids and unit-normalized vectors (a copy). */
function prepareBatch(input: UpsertInput, dim: number, lex: ResolvedLexConfig | null, schema: AttrSchema | null): Batch {
	const keys = input.keys ? Array.from(input.keys) : null;
	if (keys) {
		if (!lex) throw new ThreadError("keys need a lexical configuration (index.configureLexical)", 400);
		if (keys.length > MAX_UPSERT) throw new ThreadError(`at most ${MAX_UPSERT} documents per upsert`, 400);
		if (new Set(keys).size !== keys.length) throw new ThreadError("duplicate keys in the batch", 400);
		keys.forEach((k, i) => {
			if (typeof k !== "string" || k.length === 0) throw new ThreadError(`keys[${i}] is not a non-empty string`, 400);
			if (k.length > MAX_KEY_BYTES || utf8Len(k) > MAX_KEY_BYTES) throw new ThreadError(`keys[${i}] is longer than ${MAX_KEY_BYTES} bytes`, 400);
		});
	}
	if (input.ids === undefined && !keys) throw new ThreadError("an upsert needs ids or keys", 400);
	const givenIds = input.ids !== undefined;
	const ids = givenIds ? u32(input.ids!, "ids") : new Uint32Array(keys!.length);
	const n = ids.length;
	if (n === 0) throw new ThreadError("empty batch", 400);
	if (n > MAX_UPSERT) throw new ThreadError(`at most ${MAX_UPSERT} documents per upsert`, 400);
	if (givenIds && new Set(ids).size !== n) throw new ThreadError("duplicate ids in the batch", 400);
	if (ids.some((id) => id === 0xffffffff)) throw new ThreadError("id 4294967295 is reserved", 400);
	if (keys && keys.length !== n) throw new ThreadError(`${keys.length} keys for ${n} ids`, 400);
	const texts = input.texts ?? null;
	const metas = input.metas ?? null;
	if (texts && texts.length !== n) throw new ThreadError(`${texts.length} texts for ${n} ids`, 400);
	if (metas && metas.length !== n) throw new ThreadError(`${metas.length} metas for ${n} ids`, 400);
	if ((texts || metas) && !lex) throw new ThreadError("texts/metas need a lexical configuration (index.configureLexical)", 400);
	if (input.attrs && input.attrs.length !== n) throw new ThreadError(`${input.attrs.length} attrs for ${n} ids`, 400);
	const attrs = input.attrs ? Uint32Array.from({ length: n }, (_, i) => packAttrs(schema, input.attrs![i])) : null;
	if (input.vectors === undefined) {
		if (!texts && !metas && !keys) throw new ThreadError("an upsert needs vectors, texts, metas or keys", 400);
		if (input.payloads) throw new ThreadError("payloads belong to vectors", 400);
		return { ids, givenIds, keys, vectors: null, attrs, payloads: null, texts, metas };
	}
	if (dim === 0) throw new ThreadError("this generation has no dense index: upsert texts/metas only", 400);
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
	const enc = new TextEncoder();
	const payloads = input.payloads ? input.payloads.map((p) => (typeof p === "string" ? enc.encode(p) : p)) : null;
	if (payloads && payloads.length !== n) throw new ThreadError(`${payloads.length} payloads for ${n} vectors`, 400);
	return { ids, givenIds, keys, vectors, attrs, payloads, texts, metas };
}

/**
 * The ids of a keyed batch against the head revision: known keys keep their id, new keys take the next ids from
 * `manifest.nextId` (`vectors` when absent). Returns the ids and the number allocated.
 */
async function assignIds(ctx: WriterContext, manifest: Manifest, keys: readonly string[]): Promise<{ ids: Uint32Array; allocated: number }> {
	const found = await ctx.resolveKeys(manifest, keys);
	let nextId = manifest.nextId ?? manifest.vectors ?? 0;
	let allocated = 0;
	const ids = new Uint32Array(keys.length);
	keys.forEach((k, i) => {
		const id = found.get(k);
		if (id !== undefined) ids[i] = id;
		else {
			if (nextId >= 0xffffffff) throw new ThreadError("the id space is exhausted", 400);
			ids[i] = nextId++;
			allocated++;
		}
	});
	return { ids, allocated };
}

/** The first free seq >= `from` for both the delta and the segment key (a crashed writer's objects are skipped). */
async function allocSeq(store: BlobStore, gen: string, from: number): Promise<number> {
	let seq = Math.max(1, from);
	for (let i = 0; i < 64 && ((await store.head(keys.delta(gen, seq))) || (await store.head(keys.segment(gen, seq)))); i++) seq++;
	return seq;
}

interface EncodedSegment {
	key: string;
	bytes: number;
	tokens: number;
	ref: Omit<SegmentRef, "addedRev">;
}

/** Tokenizes and writes one lexical/docs/keys segment for the batch under the generation's `lex` configuration. */
async function encodeSegment(ctx: WriterContext, seq: number, lex: ResolvedLexConfig, b: Batch): Promise<EncodedSegment> {
	const { w, store, gen } = ctx;
	// `{}` is the default tokenizer; a null tokenizer would mean a docs-only segment without a lexical index.
	const job = new w.SegmentJob(JSON.stringify(lex.tokenizer ?? {}), lex.k1, lex.b, lex.lengths, lex.bucketTerms, lex.docs, lex.text, lex.bucketDocs, b.keys !== null, lex.bucketKeys);
	let bytes: Uint8Array;
	let finished = false;
	const epoch = ctx.thread.epoch;
	try {
		for (let i = 0; i < b.ids.length; i++) job.add(b.ids[i], b.texts?.[i] ?? undefined, b.metas?.[i] ?? undefined, b.attrs?.[i] ?? 0, b.keys?.[i]);
		finished = true;
		bytes = job.finish();
	} finally {
		if (!finished && epoch === ctx.thread.epoch) safeFree(job);
	}
	const tokens = 0; // reported from the job before finish would need a second pass; the meta carries the count
	const key = await putUnique(store, keys.segment(gen, seq), bytes);
	return { key, bytes: bytes.byteLength, tokens, ref: { seq, key, count: b.ids.length, bytes: bytes.byteLength, lex: true, docs: lex.docs, text: lex.docs && lex.text, ...(b.keys ? { keys: true } : {}) } };
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
async function encodeDelta(ctx: WriterContext, m: Manifest, files: IndexFiles, b: Batch, seq: number): Promise<EncodedDelta> {
	const { w, store, gen } = ctx;
	const vectors = b.vectors!;
	const format = m.format ?? 1;
	if (format === 3 && !files.pq) throw new ThreadError(`format 3 generation ${gen} has no PQ codebook`);
	const baseKey = `${gen}|${files.coarse}|${files.fine}|${files.pq ?? "-"}`;
	const borrowed = ctx.borrowPq(baseKey);
	const [meansBytes, pqBytes] = await Promise.all([store.get(files.fine), format === 3 && !borrowed ? store.get(files.pq!) : Promise.resolve(null)]);
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
			const v = vectors.subarray(a * dim, e * dim);
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

/** A chunked upsert's expansion: the parent keys and the chunking recorded in the manifest's `change`. */
interface Chunked {
	parents: string[];
	words: number;
	overlap: number;
}

/**
 * Applies `chunk` and `embed`: the input with one document per chunk (keyed `chunkKey(parent, i)`, the parent's meta
 * and attrs) and the vectors `embed` returns for the texts to index. Validation of the result is left to
 * `prepareBatch`, which sees an ordinary batch.
 */
async function expandInput(input: UpsertInput): Promise<{ input: UpsertInput; chunked: Chunked | null }> {
	if (input.embed && input.vectors !== undefined) throw new ThreadError("vectors and embed together: pass one or the other", 400);
	let out = input;
	let chunked: Chunked | null = null;
	if (input.chunk !== undefined && input.chunk !== false) {
		let opts: { words: number; overlap: number };
		try {
			opts = resolveChunkOptions(input.chunk);
		} catch (e) {
			throw new ThreadError(String((e as Error).message ?? e), 400);
		}
		if (input.ids !== undefined || !input.keys) throw new ThreadError("chunked upserts name documents by key: pass keys, not ids", 400);
		if (!input.texts) throw new ThreadError("chunk needs texts (one per key)", 400);
		if (input.vectors !== undefined || input.payloads) throw new ThreadError("with chunk, vectors come from embed (one per chunk)", 400);
		const parents = Array.from(input.keys);
		const n = parents.length;
		if (input.texts.length !== n) throw new ThreadError(`${input.texts.length} texts for ${n} keys`, 400);
		if (input.metas && input.metas.length !== n) throw new ThreadError(`${input.metas.length} metas for ${n} keys`, 400);
		if (input.attrs && input.attrs.length !== n) throw new ThreadError(`${input.attrs.length} attrs for ${n} keys`, 400);
		const keys: string[] = [];
		const texts: string[] = [];
		const metas: (string | null)[] = [];
		const attrs: AttrValues[] = [];
		parents.forEach((parent, i) => {
			for (const c of chunkText(input.texts![i] ?? "", opts)) {
				keys.push(chunkKey(parent, c.index));
				texts.push(c.text);
				if (input.metas) metas.push(input.metas[i]);
				if (input.attrs) attrs.push(input.attrs[i]);
			}
		});
		if (keys.length > MAX_UPSERT) throw new ThreadError(`chunk produced ${keys.length} documents, at most ${MAX_UPSERT} per upsert: pass fewer texts`, 400);
		const { chunk: _c, ids: _i, ...rest } = input;
		out = { ...rest, keys, texts, ...(input.metas ? { metas } : {}), ...(input.attrs ? { attrs } : {}) };
		chunked = { parents, ...opts };
	}
	if (out.embed) {
		const texts = (out.texts ?? []).map((t) => t ?? "");
		if (!texts.length) throw new ThreadError("embed needs texts", 400);
		const vectors = await out.embed(texts);
		const rows = Array.isArray(vectors) && vectors.length && typeof vectors[0] !== "number" ? vectors.length : null;
		if (rows !== null && rows !== texts.length) throw new ThreadError(`embed returned ${rows} vectors for ${texts.length} texts`, 400);
		if (rows === null && vectors.length === 0) throw new ThreadError(`embed returned 0 vectors for ${texts.length} texts`, 400);
		const { embed: _e, ...rest } = out;
		out = { ...rest, vectors };
	}
	return { input: out, chunked };
}

/** The ids of the parents' chunk keys the batch no longer writes (`parent#<i>` for an `i` not in the batch). */
async function staleChunkIds(ctx: WriterContext, manifest: Manifest, chunked: Chunked, written: readonly string[]): Promise<{ keys: string[]; ids: Uint32Array }> {
	const have = new Set(written);
	const stale: string[] = [];
	const lists = await Promise.all(chunked.parents.map((p) => ctx.listKeys(manifest, p + "#")));
	chunked.parents.forEach((p, i) => {
		for (const k of lists[i]) if (!have.has(k) && chunkIndexOf(k, p) !== null) stale.push(k);
	});
	if (!stale.length) return { keys: stale, ids: new Uint32Array(0) };
	const found = await ctx.resolveKeys(manifest, stale);
	return { keys: stale, ids: Uint32Array.from(new Set(found.values())) };
}

export async function upsert(ctx: WriterContext, inputIn: UpsertInput): Promise<UpsertResult> {
	const t0 = Date.now();
	const { store, gen, w } = ctx;
	writable(store);
	const replace = inputIn.replace ?? true;
	return ctx.thread.withRecovery("upsert", async () => {
		const { input, chunked } = await expandInput(inputIn);
		const st: { batch: Batch | null; delta: EncodedDelta | null; seg: EncodedSegment | null; seq: number; tombKey: string | null; tombBytes: number; idsKey: string; idsRev: number; allocated: number; stale: Uint32Array; staleRev: number } = { batch: null, delta: null, seg: null, seq: 0, tombKey: null, tombBytes: 0, idsKey: "", idsRev: -1, allocated: 0, stale: new Uint32Array(0), staleRev: -1 };
		const c = await commitRev(ctx, async ({ rev, manifest, files }) => {
			const lex = resolveLexConfig(manifest);
			const batch = (st.batch ??= prepareBatch(input, hasDense(manifest) ? manifest.dim : 0, lex, manifest.attrs ? validateSchema(manifest.attrs) : null));
			const wantSeg = lex !== null && (batch.texts !== null || batch.metas !== null || batch.keys !== null);
			const next = inherit(manifest);
			const newRev = rev + 1;
			const filesKey = `${gen}|${files.coarse}|${files.fine}|${files.pq ?? "-"}`;
			if (!st.seq) st.seq = await allocSeq(store, gen, manifest.nextSeq ?? 1);
			// Chunk keys of the parents that this put no longer writes are tombstoned with the batch (per head, like
			// the id resolution: another writer may have changed the parents meanwhile).
			if (chunked && st.staleRev !== rev) {
				st.stale = (await staleChunkIds(ctx, manifest, chunked, batch.keys!)).ids;
				st.staleRev = rev;
			}
			// Ids: given, or resolved through the head's keys (re-resolved when a lost race moved the head, since
			// another writer may have allocated ids or written the same keys meanwhile).
			const baseNext = manifest.nextId ?? manifest.vectors ?? 0;
			let nextId = baseNext;
			if (!batch.givenIds) {
				if (st.idsRev !== rev) {
					const a = await assignIds(ctx, manifest, batch.keys!);
					batch.ids = a.ids;
					st.allocated = a.allocated;
					st.idsRev = rev;
				}
				nextId = baseNext + st.allocated; // the allocation took the next ids of this head
			} else for (const id of batch.ids) if (id >= nextId) nextId = id + 1;
			if (nextId !== baseNext || manifest.nextId !== undefined) next.nextId = nextId;
			const idsKey = batch.ids.join(",");
			if (idsKey !== st.idsKey) {
				st.idsKey = idsKey;
				st.delta = null;
				st.seg = null;
			}
			// The delta and the segment are reused across retries (unless the means/codebook or the ids changed
			// under us); the tombstone set is keyed by the rev, so it is rewritten for every attempt.
			const tombIds = replace ? (st.stale.length ? Uint32Array.from(new Set([...batch.ids, ...st.stale])) : batch.ids) : st.stale;
			const tb = tombIds.length ? tombBytes(w, tombIds) : null;
			const [delta, seg, tombKey] = await Promise.all([
				batch.vectors ? (st.delta && st.delta.filesKey === filesKey ? st.delta : encodeDelta(ctx, manifest, files, batch, st.seq)) : Promise.resolve(null),
				wantSeg ? (st.seg ?? encodeSegment(ctx, st.seq, lex!, batch)) : Promise.resolve(null),
				tb ? putUnique(store, keys.tombstones(gen, newRev), tb) : Promise.resolve(null),
			]);
			st.delta = delta;
			st.seg = seg;
			st.tombKey = tombKey;
			if (tb && tombKey) {
				st.tombBytes = tb.byteLength;
				next.tombstones!.push({ rev: newRev, key: tombKey, count: tb.byteLength / 4 });
			}
			if (delta) next.deltas!.push({ seq: delta.seq, key: delta.key, count: batch.ids.length, bytes: delta.bytes, addedRev: newRev, cells: delta.nCells });
			if (seg) next.segments = [...(next.segments ?? []), { ...seg.ref, addedRev: newRev }];
			next.nextSeq = Math.max(next.nextSeq ?? 1, st.seq + 1);
			return { ...next, change: { op: "upsert", seq: st.seq, count: batch.ids.length, replace, dense: !!delta, lexical: !!seg, ...(batch.keys ? { keys: batch.keys.length, allocated: st.allocated } : {}), ...(chunked ? { chunk: { words: chunked.words, overlap: chunked.overlap, documents: chunked.parents.length, stale: st.stale.length } } : {}), writer: "thread-sdk" } };
		});
		const d = st.delta;
		return {
			gen,
			rev: c.rev,
			seq: st.seq,
			count: st.batch!.ids.length,
			ids: Array.from(st.batch!.ids),
			allocated: st.allocated,
			...(chunked ? { chunks: { documents: chunked.parents.length, count: st.batch!.ids.length, stale: st.stale.length } } : {}),
			cells: d ? Array.from(d.cells) : [],
			distinctCells: d?.nCells ?? 0,
			keys: { delta: d?.key ?? null, segment: st.seg?.key ?? null, tombstones: st.tombKey, manifest: c.manifestKey },
			bytes: { delta: d?.bytes ?? 0, segment: st.seg?.bytes ?? 0, tombstones: st.tombBytes, manifest: c.manifestBytes },
			tokens: st.seg?.tokens ?? 0,
			conflicts: c.conflicts,
			casConflicts: c.casConflicts,
			currentUpdated: c.currentUpdated,
			ms: Date.now() - t0,
		};
	});
}

/** Commits a manifest revision carrying a (new or changed) lexical configuration. */
export async function configureLexical(ctx: WriterContext, config: LexConfig): Promise<ConfigureLexicalResult> {
	const { store, gen, w } = ctx;
	writable(store);
	// Validate the tokenizer and the parameters before writing anything.
	w.tokenizer_config_json(JSON.stringify(config.tokenizer ?? {}));
	if (config.lengths !== undefined && config.lengths !== "exact" && config.lengths !== "byte") throw new ThreadError(`lengths must be "exact" or "byte"`, 400);
	if (config.k1 !== undefined && !(config.k1 >= 0)) throw new ThreadError("k1 must be >= 0", 400);
	if (config.b !== undefined && !(config.b >= 0 && config.b <= 1)) throw new ThreadError("b must be in [0, 1]", 400);
	const c = await commitRev(ctx, async ({ manifest }) => {
		if (manifest.lex && (manifest.segments?.length ?? 0) > 0) {
			// Tokenization must stay the same across the segments of a generation.
			const old = JSON.stringify(manifest.lex.tokenizer ?? null);
			if (old !== JSON.stringify(config.tokenizer ?? null)) throw new ThreadError("the tokenizer of a generation with segments cannot change (build a new generation)", 400);
		}
		return { ...inherit(manifest), lex: config, change: { op: "configure-lex", writer: "thread-sdk" } };
	});
	return { gen, rev: c.rev, config, conflicts: c.conflicts, currentUpdated: c.currentUpdated };
}

/**
 * Commits a manifest revision carrying the attribute schema (`manifest.attrs`). A schema may only grow: every
 * existing field keeps its name, type, offset and width (an enum may append values), since stored words are not
 * rewritten.
 */
export async function configureAttrs(ctx: WriterContext, spec: Record<string, AttrFieldSpec> | AttrSchema): Promise<ConfigureAttrsResult> {
	const { store, gen } = ctx;
	writable(store);
	const isSchema = "fields" in spec && Array.isArray((spec as AttrSchema).fields);
	let schema: AttrSchema = isSchema ? validateSchema(spec as AttrSchema) : defineAttrs(spec as Record<string, AttrFieldSpec>);
	const c = await commitRev(ctx, async ({ manifest }) => {
		const prev = manifest.attrs ? validateSchema(manifest.attrs) : null;
		if (prev && !isSchema) schema = defineAttrs(spec as Record<string, AttrFieldSpec>, prev);
		if (prev && !schemaExtends(prev, schema)) throw new ThreadError("the attribute schema can only add fields or enum values (existing fields keep their layout)", 400);
		return { ...inherit(manifest), attrs: schema, change: { op: "configure-attrs", writer: "thread-sdk" } };
	});
	return { gen, rev: c.rev, schema, conflicts: c.conflicts, currentUpdated: c.currentUpdated };
}

/**
 * Merges every segment of the head into one base segment and commits a revision listing only it. Reads the
 * segments whole. Nothing is written when the head has at most one segment and no tombstone set newer than it.
 */
export async function compactSegments(ctx: WriterContext): Promise<CompactSegmentsResult> {
	const t0 = Date.now();
	const { store, gen, w } = ctx;
	writable(store);
	let merged = 0;
	let key: string | null = null;
	let count = 0;
	let bytes = 0;
	let readBytes = 0;
	let built: { forKeys: string; key: string; bytes: number; count: number; keys: boolean } | null = null;
	const c = await commitRev(ctx, async ({ rev, manifest }) => {
		const lex = resolveLexConfig(manifest);
		if (!lex) throw new ThreadError("this generation has no lexical configuration", 400);
		const segs = manifest.segments ?? [];
		const sets = manifest.tombstones ?? [];
		const newest = Math.max(0, ...sets.map((t) => t.rev));
		const needed = segs.length > 1 || (segs.length === 1 && newest > segs[0].addedRev);
		if (!needed) return null;
		const forKeys = segs.map((s) => s.key).join("|");
		if (!built || built.forKeys !== forKeys) {
			const parts = await Promise.all(
				segs.map(async (s) => {
					const b = await store.get(s.key);
					if (!b) throw new ThreadError(`missing segment ${s.key}`, 404);
					return b;
				}),
			);
			readBytes = parts.reduce((a, b) => a + b.byteLength, 0);
			const tombSets = await Promise.all(sets.map(async (t) => ({ rev: t.rev, ids: new Uint32Array((await store.get(t.key))!.slice().buffer) })));
			const tomb = tombSets.length ? buildTombState(tombSets) : null;
			const concat = new Uint8Array(readBytes);
			let o = 0;
			for (const p of parts) {
				concat.set(p, o);
				o += p.byteLength;
			}
			const tombBufs = segs.map((s) => (tomb ? tombForSegment(tomb, s.addedRev) : new Uint8Array(0)));
			const tombs = new Uint8Array(tombBufs.reduce((a, b) => a + b.byteLength, 0));
			o = 0;
			for (const t of tombBufs) {
				tombs.set(t, o);
				o += t.byteLength;
			}
			const out = w.merge_segments(concat, Float64Array.from(parts, (p) => p.byteLength), tombs, Uint32Array.from(tombBufs, (t) => t.byteLength), JSON.stringify(lex.tokenizer ?? {}), lex.k1, lex.b, lex.lengths, lex.bucketTerms, lex.docs, lex.text, lex.bucketDocs, false, lex.bucketKeys);
			const meta = JSON.parse(w.segment_meta(out)) as { lex?: { num_docs: number }; docs?: { n: number }; keys?: { n: number } };
			const k = await putUnique(store, keys.segmentBase(gen, rev + 1), out);
			built = { forKeys, key: k, bytes: out.byteLength, count: meta.docs?.n ?? meta.lex?.num_docs ?? 0, keys: meta.keys !== undefined };
		}
		merged = segs.length;
		key = built.key;
		bytes = built.bytes;
		count = built.count;
		const ref: SegmentRef = { seq: 0, key: built.key, addedRev: rev + 1, count, bytes, lex: true, docs: lex.docs, text: lex.docs && lex.text, ...(built.keys ? { keys: true } : {}), base: true };
		return { ...inherit(manifest), segments: [ref], change: { op: "compact-segments", merged, count, writer: "thread-sdk" } };
	});
	return { gen, rev: c.rev, merged, key, count, bytes, readBytes, conflicts: c.conflicts, currentUpdated: c.currentUpdated, ms: Date.now() - t0 };
}

/**
 * Publishes a lexical-only generation (no dense index: `k2 0, dim 0, vectors 0`) with the given configuration as
 * rev 0, so documents can be upserted with texts and searched by text alone.
 */
export async function createLexicalGeneration(opts: CreateLexicalOptions): Promise<{ gen: string; rev: number; manifest: Manifest }> {
	const { store, gen } = opts;
	writable(store);
	if (!/^[A-Za-z0-9_.-]+$/.test(gen)) throw new ThreadError(`bad generation name ${JSON.stringify(gen)}`, 400);
	const manifest: Manifest = { gen, k1: 1, k2: 0, dim: 0, m: 0, vectors: 0, format: 1, profile: "lexical", createdAt: new Date().toISOString(), lex: opts.lex, segments: [], deltas: [], tombstones: [], nextSeq: 1, ...(opts.notes !== undefined ? { notes: opts.notes } : {}) };
	if (!(await putIfAbsent(store, keys.manifest(gen, 0), JSON.stringify(manifest), "application/json"))) throw new ThreadError(`generation ${gen} already exists`, 409);
	if (opts.setCurrent !== false) {
		const cur = await readCurrent(store);
		const body = currentBody(gen, 0, manifest);
		if (!cur) await putIfAbsent(store, keys.current(), body, "application/json");
		else await writable(store).put(keys.current(), body, { ifMatch: cur.etag, contentType: "application/json" });
	}
	return { gen, rev: 0, manifest };
}

/**
 * Deletes documents by key: the keys are resolved through the head revision's `keys` sections and the ids found
 * are tombstoned (one set, rev + 1). Nothing is written when no key has a live document.
 */
export async function deleteKeys(ctx: WriterContext, keysIn: readonly string[]): Promise<DeleteResult> {
	const t0 = Date.now();
	const { store, gen, w } = ctx;
	writable(store);
	const wanted = [...new Set(keysIn)];
	if (!wanted.length) throw new ThreadError("no keys", 400);
	let key = "";
	let ids = new Uint32Array(0);
	let missing: string[] = [];
	const c = await commitRev(ctx, async ({ rev, manifest }) => {
		if (!resolveLexConfig(manifest)) throw new ThreadError("this generation has no lexical configuration (no keys)", 400);
		const found = await ctx.resolveKeys(manifest, wanted);
		missing = wanted.filter((k) => !found.has(k));
		ids = Uint32Array.from(new Set(found.values()));
		if (!ids.length) return null;
		const tb = tombBytes(w, ids);
		const next = inherit(manifest);
		key = await putUnique(store, keys.tombstones(gen, rev + 1), tb);
		next.tombstones!.push({ rev: rev + 1, key, count: ids.length });
		return { ...next, change: { op: "delete", count: ids.length, keys: wanted.length, writer: "thread-sdk" } };
	});
	return { gen, rev: c.rev, count: ids.length, key, missing, conflicts: c.conflicts, casConflicts: c.casConflicts, currentUpdated: c.currentUpdated, ms: Date.now() - t0 };
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
