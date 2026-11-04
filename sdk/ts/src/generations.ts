/**
 * Generations as a whole: moving `idx/current.json` to another generation with a compare-and-swap, collecting a
 * generation's superseded objects (the Worker's `POST /build/gc`), and deleting a generation that nothing points at
 * any more. No wasm: these take any writable `BlobStore` (with `listObjects` and `delete` for the collectors).
 */
import type { BlobStore, ObjectInfo } from "./store.js";
import { currentBody, decodeJson, keys, manifestFiles, parseCurrent, parseManifest, parseSlabsTable, ThreadError, type CurrentPointer, type Manifest, type SlabRow } from "./wire.js";

function writable(store: BlobStore): Required<Pick<BlobStore, "put">> & BlobStore {
	if (!store.put) throw new ThreadError("this store is read-only (no put)", 400);
	return store as Required<Pick<BlobStore, "put">> & BlobStore;
}

async function readCurrent(store: BlobStore): Promise<{ cur: CurrentPointer; etag: string } | null> {
	const h = await store.head(keys.current());
	if (!h) return null;
	const b = await store.get(keys.current());
	if (!b) return null;
	return { cur: parseCurrent(b), etag: h.etag };
}

/** Manifest revisions of `gen` with their upload times, ascending. */
async function listRevs(store: BlobStore, gen: string): Promise<{ rev: number; uploaded: number }[]> {
	if (!store.listObjects) throw new ThreadError("this store cannot list objects with their upload times (listObjects)", 400);
	const prefix = `idx/${gen}/manifest.`;
	const out: { rev: number; uploaded: number }[] = [];
	for (const o of await store.listObjects(prefix)) {
		const m = o.key.slice(prefix.length).match(/^(?:(\d+)\.)?json$/);
		if (m) out.push({ rev: m[1] ? Number(m[1]) : 0, uploaded: o.uploaded });
	}
	return out.sort((a, b) => a.rev - b.rev);
}

async function headRev(store: BlobStore, gen: string): Promise<number> {
	if (!store.list) throw new ThreadError("this store cannot list objects", 400);
	const prefix = `idx/${gen}/manifest.`;
	let head = -1;
	for (const key of await store.list(prefix)) {
		const m = key.slice(prefix.length).match(/^(?:(\d+)\.)?json$/);
		if (m) head = Math.max(head, m[1] ? Number(m[1]) : 0);
	}
	if (head < 0) throw new ThreadError(`no manifest for generation ${gen}`, 404);
	return head;
}

async function readManifest(store: BlobStore, gen: string, rev: number): Promise<Manifest> {
	const b = await store.get(keys.manifest(gen, rev));
	if (!b) throw new ThreadError(`no manifest ${keys.manifest(gen, rev)}`, 404);
	return parseManifest(b, gen, rev);
}

// ---------- current.json ----------

/** Where `setCurrent` moved the pointer from (`idx/{gen}/superseded.json`), read by `dropGeneration`. */
export const supersededKey = (gen: string): string => `idx/${gen}/superseded.json`;

export interface SetCurrentOptions {
	/** The generation to point at. */
	gen: string;
	/** Its revision (default: the newest manifest of `gen`). */
	rev?: number;
	/**
	 * The pointer the switch expects to replace: `{gen, rev}` must be exactly what `current.json` names (`null`: no
	 * pointer yet). A mismatch is a 409 and nothing is written. Absent: any pointer is replaced (still a CAS on its
	 * etag, retried, so a concurrent change is not lost silently: the result's `previous` is what was replaced).
	 */
	expect?: { gen: string; rev: number } | null;
}

export interface SetCurrentResult {
	gen: string;
	rev: number;
	/** The pointer that was replaced (null: none). */
	previous: { gen: string; rev: number } | null;
	/** False when the pointer already named (gen, rev). */
	updated: boolean;
	/** CAS retries. */
	conflicts: number;
}

/**
 * Points `idx/current.json` at (gen, rev), any generation: the switch from a live generation to one built beside
 * it (`createGeneration({ setCurrent: false })`, filled through `open({ gen })`). A compare-and-swap on the pointer's
 * etag; with `expect` it fails (409) unless the pointer is exactly `expect`. The manifest must exist. Readers that
 * follow `current.json` (`open()` without `gen`) move within their `refreshInterval`; writers of the old generation
 * keep committing to it but no longer move the pointer (they only advance it within their own generation).
 * Records the switch in `idx/{previous}/superseded.json`, which `dropGeneration` waits on.
 */
export async function setCurrent(store: BlobStore, opts: SetCurrentOptions): Promise<SetCurrentResult> {
	writable(store);
	const { gen } = opts;
	if (!/^[A-Za-z0-9_.-]+$/.test(gen)) throw new ThreadError(`bad generation name ${JSON.stringify(gen)}`, 400);
	const rev = opts.rev ?? (await headRev(store, gen));
	if (!Number.isInteger(rev) || rev < 0) throw new ThreadError("rev must be a non-negative integer", 400);
	const manifest = await readManifest(store, gen, rev);
	const body = currentBody(gen, rev, manifest);
	let conflicts = 0;
	for (let attempt = 0; attempt < 10; attempt++) {
		const cur = await readCurrent(store);
		const previous = cur ? { gen: cur.cur.gen, rev: cur.cur.rev } : null;
		if (opts.expect !== undefined) {
			const e = opts.expect;
			const ok = e === null ? previous === null : previous !== null && previous.gen === e.gen && previous.rev === e.rev;
			if (!ok) throw new ThreadError(`idx/current.json is ${previous ? `${previous.gen} rev ${previous.rev}` : "absent"}, expected ${e ? `${e.gen} rev ${e.rev}` : "absent"}`, 409);
		}
		if (previous && previous.gen === gen && previous.rev === rev) return { gen, rev, previous, updated: false, conflicts };
		const put = cur ? await writable(store).put(keys.current(), body, { ifMatch: cur.etag, contentType: "application/json" }) : await writable(store).put(keys.current(), body, { ifAbsent: true, contentType: "application/json" });
		if (put) {
			if (previous && previous.gen !== gen) {
				const at = Date.now();
				await writable(store).put(supersededKey(previous.gen), JSON.stringify({ by: { gen, rev }, rev: previous.rev, ms: at, at: new Date(at).toISOString() }), { contentType: "application/json" });
			}
			return { gen, rev, previous, updated: true, conflicts };
		}
		conflicts++;
		await new Promise((r) => setTimeout(r, 20 + Math.random() * 50));
	}
	throw new ThreadError(`idx/current.json: gave up after ${conflicts} CAS conflicts`, 409);
}

// ---------- GC ----------

export interface GcOptions {
	gen: string;
	/** Newest revisions whose objects are kept (default 2, at least 1). */
	keep?: number;
	/**
	 * Also keep every revision superseded less than this long ago (readers may hold it for their refresh interval),
	 * and never delete an object younger than it (a writer's objects before its commit). Default 300.
	 */
	graceSec?: number;
	/** Default true: report only. */
	dryRun?: boolean;
}

export interface GcResult {
	gen: string;
	dryRun: boolean;
	keptRevs: number[];
	/** Objects under `idx/{gen}/`. */
	objects: number;
	/** Objects the kept revisions reference. */
	referenced: number;
	/** Objects that would be (dryRun) or were deleted. */
	garbage: number;
	kinds: Record<string, { n: number; bytes: number }>;
	deleted: number;
	sample: string[];
}

/** The object kinds revisions create: only these are GC candidates (manifests, base files and the rest stay). */
function kindOf(gen: string, key: string): string | null {
	const rest = key.slice(`idx/${gen}/`.length);
	if (/^fine\.\d+(\.[0-9a-f]+)?\.f32$/.test(rest)) return "fine"; // fine means of a split revision (never fine.f32)
	if (/^tree\.\d+(\.[0-9a-f]+)?\.bin$/.test(rest)) return "tree";
	return rest.startsWith("slab/") ? "slab" : rest.startsWith("delta/") ? "delta" : rest.startsWith("seg/") ? "segment" : rest.startsWith("tombstones.") ? "tombstones" : rest.startsWith("slabs.") ? "slabs-table" : null;
}

function tally(gen: string, objs: ObjectInfo[]): Record<string, { n: number; bytes: number }> {
	const kinds: Record<string, { n: number; bytes: number }> = {};
	for (const o of objs) {
		const kind = kindOf(gen, o.key) ?? "other";
		kinds[kind] ??= { n: 0, bytes: 0 };
		kinds[kind].n++;
		kinds[kind].bytes += o.size;
	}
	return kinds;
}

async function deleteAll(store: BlobStore, keysToDelete: string[]): Promise<number> {
	if (!store.delete) throw new ThreadError("this store cannot delete objects", 400);
	let n = 0;
	let i = 0;
	await Promise.all(
		Array.from({ length: Math.min(6, keysToDelete.length) }, async () => {
			while (i < keysToDelete.length) {
				await store.delete!(keysToDelete[i++]);
				n++;
			}
		}),
	);
	return n;
}

/**
 * The Worker's `POST /build/gc`: deletes the slab, delta, segment, tombstone, slabs-table, `fine.{rev}.f32` and
 * `tree.{rev}.bin` objects under `idx/{gen}/` that no kept revision references (the newest `keep`, plus every
 * revision superseded less than `graceSec` ago), never an object younger than `graceSec`. Manifests stay (old ones
 * then point at deleted objects: a reader pinned to one of them fails, a reader of the generation's head or of
 * `current.json` does not). Works on any generation, the live one or one `current.json` left.
 */
export async function gcGeneration(store: BlobStore, opts: GcOptions): Promise<GcResult> {
	const { gen } = opts;
	if (!gen) throw new ThreadError("gen required", 400);
	if (!store.listObjects) throw new ThreadError("this store cannot list objects with their upload times (listObjects)", 400);
	const keepN = Math.max(1, opts.keep ?? 2);
	const graceMs = (opts.graceSec ?? 300) * 1000;
	const dryRun = opts.dryRun !== false;
	const now = Date.now();
	const revs = await listRevs(store, gen);
	if (!revs.length) throw new ThreadError(`no manifest for ${gen}`, 404);
	const kept = revs.filter((r, i) => i >= revs.length - keepN || (i + 1 < revs.length && now - revs[i + 1].uploaded < graceMs)).map((r) => r.rev);
	const referenced = new Set<string>();
	const tables = new Map<string, number>();
	for (const rev of kept) {
		const m = await readManifest(store, gen, rev);
		const f = manifestFiles(m);
		for (const k of [f.coarse, f.fine, f.pq, f.slabs]) if (k) referenced.add(k);
		if (f.slabs) tables.set(f.slabs, m.k1 * m.k2);
		for (const d of m.deltas ?? []) referenced.add(d.key);
		for (const s of m.segments ?? []) referenced.add(s.key);
		for (const t of m.tombstones ?? []) referenced.add(t.key);
		if (m.tree) referenced.add(m.tree.key);
	}
	for (const [key, k] of tables) {
		const b = await store.get(key);
		if (!b) continue;
		const rows: SlabRow[] = parseSlabsTable(b, k, key);
		rows.forEach((row, cell) => referenced.add(keys.slab(gen, cell, row.length > 4 ? (row[4] as number) : 0)));
	}
	const all = await store.listObjects(`idx/${gen}/`);
	const garbage = all.filter((o) => kindOf(gen, o.key) !== null && !referenced.has(o.key) && now - o.uploaded >= graceMs);
	const deleted = dryRun || !garbage.length ? 0 : await deleteAll(store, garbage.map((o) => o.key));
	return { gen, dryRun, keptRevs: kept, objects: all.length, referenced: referenced.size, garbage: garbage.length, kinds: tally(gen, garbage), deleted, sample: garbage.slice(0, 10).map((o) => o.key) };
}

export interface DropGenerationOptions {
	gen: string;
	/**
	 * How long the generation must have been off `current.json`, and without a write, before it is deleted (default
	 * 300 s; readers that follow the pointer move within their refresh interval, 30 s by default).
	 */
	graceSec?: number;
	/** Default true: report only. */
	dryRun?: boolean;
}

export interface DropGenerationResult {
	gen: string;
	dryRun: boolean;
	/** Objects under `idx/{gen}/` that would be (dryRun) or were deleted. */
	objects: number;
	bytes: number;
	kinds: Record<string, { n: number; bytes: number }>;
	deleted: number;
	/** When the pointer left the generation (ms since the epoch; from `superseded.json`, else the pointer's age). */
	supersededAt: number;
	/** The newest object of the generation (ms since the epoch). */
	lastWriteAt: number;
	current: { gen: string; rev: number } | null;
}

/**
 * Deletes every object under `idx/{gen}/` of a generation `current.json` no longer names: after the switch
 * (`setCurrent`) and the grace period, when nothing has written to it for the grace period either and no compaction
 * holds its lock. Manifests go first, so a reader still opening it fails cleanly (404) instead of meeting missing
 * slabs, and a rerun after a crash finishes the rest. Refuses (409) the current generation, a recent switch or a
 * recent write. Readers pinned to the generation (`open({ gen })`) are not protected: close them first.
 */
export async function dropGeneration(store: BlobStore, opts: DropGenerationOptions): Promise<DropGenerationResult> {
	const { gen } = opts;
	if (!gen || !/^[A-Za-z0-9_.-]+$/.test(gen)) throw new ThreadError(`bad generation name ${JSON.stringify(gen)}`, 400);
	if (!store.listObjects) throw new ThreadError("this store cannot list objects with their upload times (listObjects)", 400);
	const graceMs = (opts.graceSec ?? 300) * 1000;
	const dryRun = opts.dryRun !== false;
	const now = Date.now();
	const cur = await readCurrent(store);
	if (cur?.cur.gen === gen) throw new ThreadError(`${gen} is the current generation (idx/current.json): switch away first (setCurrent)`, 409);
	const marker = await store.get(supersededKey(gen));
	let supersededAt: number;
	if (marker) supersededAt = decodeJson<{ ms: number }>(marker).ms;
	else {
		// No record of the switch (made by another writer): the pointer has not changed since it was last written.
		const p = (await store.listObjects(keys.current())).find((o) => o.key === keys.current());
		supersededAt = p ? p.uploaded : 0;
	}
	const all = await store.listObjects(`idx/${gen}/`);
	const own = all.filter((o) => o.key !== supersededKey(gen));
	const lastWriteAt = own.reduce((a, o) => Math.max(a, o.uploaded), 0);
	const lock = all.find((o) => o.key === `idx/${gen}/compact.lock`);
	if (now - supersededAt < graceMs) throw new ThreadError(`${gen} left idx/current.json ${Math.round((now - supersededAt) / 1000)} s ago, less than the grace period (${graceMs / 1000} s)`, 409);
	if (now - lastWriteAt < graceMs) throw new ThreadError(`${gen} was written ${Math.round((now - lastWriteAt) / 1000)} s ago, less than the grace period (${graceMs / 1000} s): a writer may still use it`, 409);
	if (lock) {
		const b = await store.get(lock.key);
		const l = b ? decodeJson<{ ms?: number; released?: boolean }>(b) : null;
		if (l && !l.released && now - (l.ms ?? 0) < 10 * 60 * 1000) throw new ThreadError(`${gen}: a compaction holds idx/${gen}/compact.lock`, 409);
	}
	const isManifest = (k: string) => /\/manifest(\.\d+)?\.json$/.test(k);
	const ordered = [...all.filter((o) => isManifest(o.key)), ...all.filter((o) => !isManifest(o.key) && o.key !== supersededKey(gen)), ...all.filter((o) => o.key === supersededKey(gen))];
	const deleted = dryRun || !ordered.length ? 0 : await deleteInOrder(store, ordered.map((o) => o.key), (k) => isManifest(k));
	return { gen, dryRun, objects: all.length, bytes: all.reduce((a, o) => a + o.size, 0), kinds: tally(gen, all), deleted, supersededAt, lastWriteAt, current: cur ? { gen: cur.cur.gen, rev: cur.cur.rev } : null };
}

/** Deletes the keys `first` accepts, then the rest (each group in parallel). */
async function deleteInOrder(store: BlobStore, ks: string[], first: (k: string) => boolean): Promise<number> {
	return (await deleteAll(store, ks.filter(first))) + (await deleteAll(store, ks.filter((k) => !first(k))));
}
