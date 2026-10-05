/**
 * vib-bench Worker app: thin I/O shell around the Rust/wasm core.
 *
 * Build stages are stateless HTTP endpoints driven by a local script, so each
 * invocation stays inside the Worker memory/CPU budget and the orchestration
 * (fan-out, retries) lives outside. Query serving caches the routing table per
 * isolate and reads one slab object per probed cell.
 *
 * The deployed entry is src/main.ts: it serves queries with this module and forwards build stages and writes to
 * dynamic "slot" Workers (Worker Loader) that run a second bundle of this same module (vite.config.ts), each in
 * its own isolate. In a slot, env.BUCKET is an R2Bucket-shaped shim over RPC, env.THREAD_SLOT names the slot, and
 * env.SLOTS (also given to ?slot=local requests) lets the automatic compaction spread its cells over other slots.
 *
 * R2 key layout:
 *   ingest/{shard:05}.f32                raw unit vectors, f32 LE
 *   build/{gen}/sample/{i}.f32           strided sample for coarse k-means
 *   build/{gen}/coarse/{iter}.f32        k1*dim means
 *   build/{gen}/fine/{iter}.f32          k1*k2*dim means (grouped by coarse)
 *   build/{gen}/fine/{iter}/{g}.f32      per-group slice written by reduce jobs
 *   build/{gen}/partials/{tag}/{job}.acc partial k-means accumulators
 *   build/{gen}/assign/{iter}/{shard:05}.u32  k-means cell per vector after iteration iter (u32 LE)
 *   build/{gen}/neighbors/{iter}.u32     k x nNeigh cell neighbour table of fine/{iter} (u32 LE)
 *   build/{gen}/neighbors/{iter}/{r}.u32 row ranges written by cell-neighbors jobs
 *   build/{gen}/pq/residuals/{i}.f32     PQ training residuals (sample vectors minus their final cell mean)
 *   build/{gen}/pq/codebook/{it}.bin     PQ codebook after iteration it (0 = pq-init)
 *   build/{gen}/pq/partials/{it}/{j}.acc PQ training map partials of iteration it
 *   build/{gen}/parts/{g}/{job}.part     assign-job output per coarse group
 *   idx/{gen}/manifest.json, coarse.f32, fine.f32, slab/{cell:05}   rev 0 (publish / finalize)
 *   idx/{gen}/slabs.json                 per cell [scanLen, rerankOff, count, bytes] (rerankOff 0 for format 1)
 *   idx/{gen}/pq.bin                     format 3: the PQ codebook the scan tier was encoded with
 *   idx/{gen}/manifest.{rev}.json        revisions >= 1 (publish-rev, upsert, delete, compact); immutable
 *   idx/{gen}/slabs.{rev}.json           slabs table written by a compaction; rows gain a 5th column, slabRev
 *   idx/{gen}/slab/{cell:05}.{slabRev}   slab rewritten by a compaction that read manifest rev slabRev - 1
 *   idx/{gen}/delta/{seq}.vibd           delta segment of one upsert
 *   idx/{gen}/tombstones.{rev}.u32       ids deleted/replaced at rev (sorted u32 LE)
 *   idx/{gen}/tree.{rev}.bin             centroid tree over the fine means (publish / publish-rev with `tree`, or a
 *                                        compaction that split cells)
 *   idx/{gen}/fine.{rev}.f32             fine means of a revision whose compaction split cells (k grows by one per split)
 *   idx/{gen}/compact.lock               automatic compaction in progress (create-only PUT; stale after 10 min)
 *   idx/{gen}/compact.last.json          outcome of the latest automatic compaction runs (GET /index/status)
 *   idx/current.json                     {gen, rev}
 *
 * The manifest records the build profile (profile, m, k1, k2, format, tiers, pq, nNeigh), the query defaults
 * (defaults: {nprobe, rerank, pc, route?}) that /query uses for absent parameters, the centroid tree (tree: {key,
 * groups, beam, ...}) and, from rev 1 on, its files, deltas and tombstone sets. See docs/update-path.md.
 */
import wasmModule from "../../../crates/thread-wasm/pkg/thread_bg.wasm";
import init, * as thread from "../../../crates/thread-wasm/pkg/thread.js";
import { compileFilter, validateSchema, type AttrFilterExpr, type AttrSchema } from "../../../sdk/ts/src/attrs.ts";
// Pure-TS builders shared with the SDK (no wasm import of their own: they take this module's bindings).
import { buildRoutingObjects } from "../../../sdk/ts/src/routing.ts";
import { buildBasePack } from "../../../sdk/ts/src/pack.ts";
import { loadRevState as sdkLoadRevState } from "../../../sdk/ts/src/revision.ts";
import { R2BlobStore } from "../../../sdk/ts/src/store.ts";
import { currentBody, groupedTableOf, keys as sdkKeys, type Manifest as SdkManifest, type RoutingElem, type RoutingOption, type RoutingTables } from "../../../sdk/ts/src/wire.ts";

interface Env {
	BUCKET: R2Bucket;
	BUILD_TOKEN: string;
	/** Runs a stage in another dynamic slot (src/main.ts SlotRpc); absent when this module runs on its own. */
	SLOTS?: SlotsRpc;
	/** The slot this isolate is (absent: the orchestrator / query isolate). */
	THREAD_SLOT?: string;
}

interface SlotsRpc {
	stage(slot: string, stage: string, body: unknown): Promise<{ status: number; data: any }>;
}

let wasmReady: Promise<unknown> | null = null;
/**
 * Bumped on every reset; work started under an older epoch holds stale wasm objects. Rule: every wasm
 * object is freed only in the epoch it was created in (safeFree(o, epoch) skips free() otherwise).
 */
let wasmEpoch = 0;
let wasmResets = 0;
/** Raw wasm exports (for the wasm memory size reported in query stats). */
let wasmRaw: thread.InitOutput | null = null;
function initWasm(): Promise<unknown> {
	return init({ module_or_path: wasmModule }).then((w) => {
		wasmRaw = w;
		thread.init_panic_hook();
	});
}
function ensureWasm(): Promise<unknown> {
	if (!wasmReady) {
		const p = initWasm();
		wasmReady = p;
		p.catch(() => {
			if (wasmReady === p) wasmReady = null;
		});
	}
	return wasmReady;
}

/**
 * After a trap (WebAssembly.RuntimeError) the instance is poisoned: re-instantiate it,
 * reinstall the panic hook and drop every cached wasm object (the Searcher).
 */
function resetWasm(reason: unknown) {
	wasmResets++;
	wasmEpoch++;
	console.error(`wasm trap, resetting instance (reset #${wasmResets}): ${reason instanceof Error ? `${reason.message}\n${reason.stack ?? ""}` : String(reason)}`);
	thread.__wbg_reset_state();
	// init() returns the new instance's exports once the glue holds an instance.
	const p = initWasm();
	wasmReady = p;
	// The cached Searchers/codebooks belong to the dead instance: drop them without free(). Parsed revisions
	// (revCache) and delta segments are plain JS objects and stay valid.
	for (const slot of baseCache.values()) slot.evicted = true;
	baseCache.clear();
	slabCache.clear();
	slabCacheBytes = 0;
}

/**
 * Runs wasm-using work; on a wasm trap resets the instance and retries once. Errors thrown
 * because another request reset the instance mid-flight (stale objects) are retried too.
 */
async function withWasmRecovery<T>(label: string, fn: (attempt: number) => Promise<T>): Promise<T> {
	const epoch = wasmEpoch;
	try {
		return await fn(1);
	} catch (e) {
		const trap = e instanceof WebAssembly.RuntimeError;
		if (!trap && epoch === wasmEpoch) throw e;
		if (trap && epoch === wasmEpoch) resetWasm(e);
		else console.error(`${label}: retrying after concurrent wasm reset: ${e instanceof Error ? e.message : String(e)}`);
		await ensureWasm();
		return await fn(2);
	}
}

// ---------- small utilities ----------

class HttpError extends Error {
	constructor(public status: number, message: string) {
		super(message);
	}
}

function json(data: unknown, status = 200, headers: Record<string, string> = {}): Response {
	return new Response(JSON.stringify(data), {
		status,
		headers: { "content-type": "application/json", ...headers },
	});
}

async function pmap<T, R>(items: T[], limit: number, fn: (item: T, i: number) => Promise<R>): Promise<R[]> {
	const out: R[] = new Array(items.length);
	let next = 0;
	const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
		while (true) {
			const i = next++;
			if (i >= items.length) return;
			out[i] = await fn(items[i], i);
		}
	});
	await Promise.all(workers);
	return out;
}

/** Runs fn over keys sequentially while prefetching `ahead` objects. */
async function streamObjects(
	bucket: R2Bucket,
	keys: string[],
	ahead: number,
	fn: (buf: ArrayBuffer, i: number) => void | Promise<void>,
	range?: (i: number) => R2Range | undefined,
): Promise<number> {
	let bytes = 0;
	const pending: Promise<ArrayBuffer>[] = [];
	const start = (i: number) => getBuffer(bucket, keys[i], range?.(i));
	for (let i = 0; i < Math.min(ahead, keys.length); i++) pending.push(start(i));
	for (let i = 0; i < keys.length; i++) {
		const buf = await pending.shift()!;
		const nextIdx = i + ahead;
		if (nextIdx < keys.length) pending.push(start(nextIdx));
		bytes += buf.byteLength;
		await fn(buf, i);
	}
	return bytes;
}

async function getBuffer(bucket: R2Bucket, key: string, range?: R2Range): Promise<ArrayBuffer> {
	const obj = await bucket.get(key, range ? { range } : undefined);
	if (!obj) throw new HttpError(404, `missing object ${key}`);
	return await obj.arrayBuffer();
}

async function getF32(bucket: R2Bucket, key: string, range?: R2Range): Promise<Float32Array> {
	const buf = await getBuffer(bucket, key, range);
	if (buf.byteLength % 4 !== 0) throw new HttpError(500, `object ${key} is not f32-aligned`);
	return new Float32Array(buf);
}

async function putBytes(bucket: R2Bucket, key: string, data: ArrayBuffer | ArrayBufferView | string) {
	const body = typeof data === "string" ? data : (data as any).buffer ? new Uint8Array((data as any).buffer, (data as any).byteOffset, (data as any).byteLength) : (data as ArrayBuffer);
	await bucket.put(key, body);
}

function pad5(n: number): string {
	return n.toString().padStart(5, "0");
}

/**
 * Frees a wasm object created in `epoch` (capture `wasmEpoch` right after `await ensureWasm()`).
 * Every wasm object is freed only in the epoch it was created in: after a reset its pointer means
 * nothing in the new instance, and free() there could release memory another request now owns.
 * Ignores errors (after a trap the instance is poisoned and free() throws too).
 */
function safeFree(o: { free(): void }, epoch: number) {
	// The object belongs to a dead instance; its pointer means nothing in the new one.
	if (epoch !== wasmEpoch) return;
	try {
		o.free();
	} catch {
		/* the original error is what matters */
	}
}

async function readJson<T>(request: Request): Promise<T> {
	try {
		return (await request.json()) as T;
	} catch {
		throw new HttpError(400, "invalid JSON body");
	}
}

// ---------- build stages ----------

interface SampleReq {
	gen: string;
	shards: string[]; // ingest keys
	perShard: number; // vectors to take from the start of each shard
	dim: number;
	out: string; // sample index
}

/** Reads the first `perShard` vectors of each shard and concatenates them. */
async function stageSample(env: Env, body: SampleReq) {
	const { gen, shards, perShard, dim, out } = body;
	const rowBytes = dim * 4;
	const buf = new Float32Array(shards.length * perShard * dim);
	let n = 0;
	await streamObjects(
		env.BUCKET,
		shards,
		6,
		(b) => {
			const f = new Float32Array(b);
			buf.set(f, n * dim);
			n += f.length / dim;
		},
		() => ({ offset: 0, length: perShard * rowBytes }),
	);
	const key = `build/${gen}/sample/${out}.f32`;
	await putBytes(env.BUCKET, key, buf.subarray(0, n * dim));
	return { key, vectors: n };
}

interface InitCoarseReq {
	gen: string;
	sampleKeys: string[];
	k: number;
	dim: number;
}

/** Initial coarse centroids: the first k sample vectors (sample is strided over shards). */
async function stageInitCoarse(env: Env, body: InitCoarseReq) {
	const { gen, sampleKeys, k, dim } = body;
	const out = new Float32Array(k * dim);
	let n = 0;
	for (const key of sampleKeys) {
		if (n >= k) break;
		const f = await getF32(env.BUCKET, key);
		const take = Math.min(f.length / dim, k - n);
		out.set(f.subarray(0, take * dim), n * dim);
		n += take;
	}
	if (n < k) throw new HttpError(400, `sample has ${n} vectors, need ${k}`);
	const key = `build/${gen}/coarse/0.f32`;
	await putBytes(env.BUCKET, key, out);
	return { key, k, dim };
}

interface KmeansFlatReq {
	gen: string;
	centroidsKey: string;
	inputKeys: string[];
	k: number;
	dim: number;
	outKey: string;
}

/** One full iteration of flat k-means over a small input set (coarse level). */
async function stageKmeansFlat(env: Env, body: KmeansFlatReq) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { centroidsKey, inputKeys, k, dim, outKey } = body;
	const prev = await getF32(env.BUCKET, centroidsKey);
	const job = new thread.FlatKmeansJob(dim, k, prev);
	let vectors = 0;
	const bytes = await streamObjects(env.BUCKET, inputKeys, 2, (b) => {
		vectors += job.add_batch(new Float32Array(b));
	});
	const means = job.means(prev);
	safeFree(job, epoch);
	await putBytes(env.BUCKET, outKey, means);
	return { outKey, vectors, bytes };
}

interface InitFineReq {
	gen: string;
	coarseKey: string;
	sampleKeys: string[];
	k1: number;
	k2: number;
	dim: number;
	seed: number;
	outKey: string;
}

async function stageInitFine(env: Env, body: InitFineReq) {
	await ensureWasm();
	const { coarseKey, sampleKeys, k1, k2, dim, seed, outKey } = body;
	const job = new thread.FineInitJob(dim, k1, k2, await getF32(env.BUCKET, coarseKey), seed >>> 0);
	let vectors = 0;
	await streamObjects(env.BUCKET, sampleKeys, 1, (b) => {
		vectors += job.add_batch(new Float32Array(b));
	});
	const fine = job.finish(); // consumes job
	await putBytes(env.BUCKET, outKey, fine);
	return { outKey, vectors, cells: k1 * k2 };
}

/** A shard of the ingest set: its R2 key and its index (ingest/{index:05}.f32, ids index*shardSize...). */
interface ShardRef {
	key: string;
	index: number;
}

/** Accepts `{key, index}` or a bare ingest key (index parsed from `ingest/{index:05}.f32`). */
function shardRefs(shards: (string | ShardRef)[]): ShardRef[] {
	return shards.map((s) => {
		if (typeof s !== "string") {
			if (!Number.isInteger(s.index) || s.index < 0) throw new HttpError(400, `shard ${s.key}: bad index ${s.index}`);
			return s;
		}
		const m = s.match(/(\d+)\.f32$/);
		if (!m) throw new HttpError(400, `cannot parse a shard index from ${s}`);
		return { key: s, index: Number(m[1]) };
	});
}

/** Per-shard assignment object written by the k-means map stages: one u32 LE cell id per vector. */
const assignKey = (prefix: string, index: number) => `${prefix}/${pad5(index)}.u32`;

async function getU32(bucket: R2Bucket, key: string): Promise<Uint32Array> {
	const buf = await getBuffer(bucket, key);
	if (buf.byteLength % 4 !== 0) throw new HttpError(500, `object ${key} is not u32-aligned`);
	return new Uint32Array(buf);
}

/**
 * Build-time coarse groups searched per vector, clamped to [1, k1]. The build jobs (TreeKmeansJob, AssignJob)
 * panic with probe_coarse 0 (no candidate cells), unlike Searcher.route where 0 means flat; so 0 (or absent)
 * is mapped to k1, which is the exhaustive flat search.
 */
function buildProbeCoarse(pc: number | undefined, k1: number): number {
	if (!Number.isInteger(k1) || k1 < 1) throw new HttpError(400, "k1 must be an integer >= 1");
	if (pc === undefined || pc === null || !Number.isFinite(pc) || pc < 1) return k1;
	return Math.min(Math.floor(pc), k1);
}

interface KmeansTreeMapReq {
	gen: string;
	coarseKey: string;
	fineKey: string;
	shards: (string | ShardRef)[];
	k1: number;
	k2: number;
	dim: number;
	probeCoarse: number;
	outKey: string;
	/** When set, each shard's chosen cells are written to `${assignOutPrefix}/${index:05}.u32` (seeds kmeans-inc-map). */
	assignOutPrefix?: string;
}

async function stageKmeansTreeMap(env: Env, body: KmeansTreeMapReq) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { coarseKey, fineKey, k1, k2, dim, outKey, assignOutPrefix } = body;
	const probeCoarse = buildProbeCoarse(body.probeCoarse, k1);
	const shards = shardRefs(body.shards);
	// Centroid buffers are scoped so the JS copies are collectable once wasm owns them.
	const job = await (async () => {
		const [coarse, fine] = await Promise.all([getF32(env.BUCKET, coarseKey), getF32(env.BUCKET, fineKey)]);
		return new thread.TreeKmeansJob(dim, k1, k2, coarse, fine, probeCoarse);
	})();
	let vectors = 0;
	const writes: Promise<void>[] = [];
	try {
		const bytes = await streamObjects(
			env.BUCKET,
			shards.map((s) => s.key),
			1,
			(b, i) => {
				const v = new Float32Array(b);
				if (assignOutPrefix) {
					const a = job.add_batch_assign(v);
					vectors += a.length;
					writes.push(putBytes(env.BUCKET, assignKey(assignOutPrefix, shards[i].index), a));
				} else vectors += job.add_batch(v);
			},
		);
		await Promise.all(writes);
		const partial = job.partial();
		await putBytes(env.BUCKET, outKey, partial);
		return { outKey, vectors, bytes, partialBytes: partial.byteLength, assignments: assignOutPrefix ? shards.length : 0 };
	} finally {
		safeFree(job, epoch);
	}
}

interface KmeansIncMapReq {
	gen: string;
	fineKey: string;
	/** k x nNeigh u32 table from cell-neighbors for the means in fineKey. */
	neighborsKey: string;
	k: number;
	dim: number;
	nNeigh: number;
	shards: ShardRef[];
	/** Previous iteration's assignments: `${prevAssignPrefix}/${index:05}.u32`. */
	prevAssignPrefix: string;
	assignOutPrefix: string;
	outKey: string;
}

/**
 * Incremental k-means map: each vector is compared only against the nNeigh neighbours of its
 * previous cell. Writes the new assignment per shard and the partial accumulator (same format
 * as kmeans-tree-map, so kmeans-reduce is unchanged).
 */
async function stageKmeansIncMap(env: Env, body: KmeansIncMapReq) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { fineKey, neighborsKey, k, dim, nNeigh, prevAssignPrefix, assignOutPrefix, outKey } = body;
	const shards = shardRefs(body.shards);
	const job = await (async () => {
		const [means, neighbors] = await Promise.all([getF32(env.BUCKET, fineKey), getU32(env.BUCKET, neighborsKey)]);
		if (means.length !== k * dim) throw new HttpError(400, `${fineKey} has ${means.length / dim} means, expected ${k}`);
		if (neighbors.length !== k * nNeigh) throw new HttpError(400, `${neighborsKey} has ${neighbors.length} ids, expected ${k}x${nNeigh}`);
		return new thread.IncKmeansJob(dim, k, means, neighbors, nNeigh);
	})();
	let vectors = 0;
	let changed = 0;
	let bytes = 0;
	const writes: Promise<void>[] = [];
	try {
		// Prefetch one shard (vectors + previous assignment) ahead.
		const fetchShard = (s: ShardRef) => Promise.all([getBuffer(env.BUCKET, s.key), getU32(env.BUCKET, assignKey(prevAssignPrefix, s.index))]);
		let next = shards.length ? fetchShard(shards[0]) : null;
		for (let i = 0; i < shards.length; i++) {
			const [buf, prev] = await next!;
			next = i + 1 < shards.length ? fetchShard(shards[i + 1]) : null;
			bytes += buf.byteLength;
			const v = new Float32Array(buf);
			if (prev.length * dim !== v.length) throw new HttpError(400, `shard ${shards[i].index}: ${v.length / dim} vectors but ${prev.length} previous assignments`);
			const a = job.add_batch(v, prev);
			if (a.length !== prev.length) throw new HttpError(500, `IncKmeansJob returned ${a.length} assignments for ${prev.length} vectors`);
			for (let x = 0; x < a.length; x++) if (a[x] !== prev[x]) changed++;
			vectors += a.length;
			writes.push(putBytes(env.BUCKET, assignKey(assignOutPrefix, shards[i].index), a));
		}
		await Promise.all(writes);
		const partial = job.partial();
		await putBytes(env.BUCKET, outKey, partial);
		return { outKey, vectors, changed, bytes, partialBytes: partial.byteLength };
	} finally {
		safeFree(job, epoch);
	}
}

interface CellNeighborsReq {
	gen: string;
	fineKey: string;
	k: number;
	dim: number;
	first: number;
	count: number;
	n: number;
	outKey: string;
}

/** Rows [first, first+count) of the cell neighbour table of the means in fineKey, as u32 LE (count x n). */
async function stageCellNeighbors(env: Env, body: CellNeighborsReq) {
	await ensureWasm();
	const { fineKey, k, dim, first, count, n, outKey } = body;
	const rows = await (async () => {
		const means = await getF32(env.BUCKET, fineKey);
		if (means.length !== k * dim) throw new HttpError(400, `${fineKey} has ${means.length / dim} means, expected ${k}`);
		return thread.cell_neighbors(means, dim, first, count, n);
	})();
	await putBytes(env.BUCKET, outKey, rows);
	return { outKey, first, count, n, bytes: rows.byteLength };
}

interface KmeansReduceReq {
	gen: string;
	partialKeys: string[];
	prevFineKey: string;
	dim: number;
	firstCell: number;
	nCells: number;
	outKey: string;
	/** Reseed cells with fewer than minCount members next to the densest cells of the range. */
	reseed?: { minCount: number; seed: number };
}

/** Reduces one cell range across all partials and writes the new means for it. */
async function stageKmeansReduce(env: Env, body: KmeansReduceReq) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { partialKeys, prevFineKey, dim, firstCell, nCells, outKey } = body;
	const [off, len] = thread.accum_record_range(dim, firstCell, nCells);
	const reducer = new thread.Reducer(nCells, dim);
	const prevPromise = getF32(env.BUCKET, prevFineKey, { offset: firstCell * dim * 4, length: nCells * dim * 4 });
	const bytes = await streamObjects(
		env.BUCKET,
		partialKeys,
		6,
		(b) => reducer.add_records(new Uint8Array(b)),
		() => ({ offset: off, length: len }),
	);
	const prev = await prevPromise;
	const reseed = body.reseed;
	const means = reseed ? reducer.means_reseed(prev, reseed.minCount >>> 0, reseed.seed >>> 0) : reducer.means(prev);
	const reseeded = reseed ? reducer.last_reseeded() : 0;
	const counts = Array.from(reducer.counts());
	safeFree(reducer, epoch);
	await putBytes(env.BUCKET, outKey, means);
	const empty = counts.filter((c) => c === 0).length;
	return {
		outKey,
		bytes,
		cells: nCells,
		empty,
		reseeded,
		small: reseed ? counts.filter((c) => c < reseed.minCount).length : undefined,
		min: Math.min(...counts),
		max: Math.max(...counts),
		total: counts.reduce((a, b) => a + b, 0),
		counts,
	};
}

interface ConcatReq {
	keys: string[];
	outKey: string;
}

async function stageConcat(env: Env, body: ConcatReq) {
	const parts = await pmap(body.keys, 6, (k) => getBuffer(env.BUCKET, k));
	const total = parts.reduce((a, b) => a + b.byteLength, 0);
	const out = new Uint8Array(total);
	let o = 0;
	for (const p of parts) {
		out.set(new Uint8Array(p), o);
		o += p.byteLength;
	}
	await putBytes(env.BUCKET, body.outKey, out);
	return { outKey: body.outKey, bytes: total };
}

// ---------- PQ codebook training (format 3) ----------

interface PqResidualsReq {
	gen: string;
	shards: ShardRef[];
	/** Rows taken from the start of each shard (the sample stage's perShard). */
	perShard: number;
	/** Final k-means assignment: `${assignPrefix}/${index:05}.u32`. */
	assignPrefix: string;
	/** Final fine means (k x dim). */
	fineKey: string;
	dim: number;
	k: number;
	outKey: string;
}

/**
 * PQ training residuals: for each shard, the first `perShard` vectors minus the mean of their final
 * k-means cell (`thread.residuals_for`), concatenated (f32 LE) into outKey. Also reports the residual
 * energy (mean squared norm), the reference for the PQ mean_sq_err.
 */
async function stagePqResiduals(env: Env, body: PqResidualsReq) {
	await ensureWasm();
	const { perShard, assignPrefix, fineKey, dim, k, outKey } = body;
	const shards = shardRefs(body.shards);
	const means = await getF32(env.BUCKET, fineKey);
	if (means.length !== k * dim) throw new HttpError(400, `${fineKey} has ${means.length / dim} means, expected ${k}`);
	const out = new Float32Array(shards.length * perShard * dim);
	let rows = 0;
	let sumSq = 0;
	let bytes = 0;
	const fetchShard = (s: ShardRef) =>
		Promise.all([getBuffer(env.BUCKET, s.key, { offset: 0, length: perShard * dim * 4 }), getBuffer(env.BUCKET, assignKey(assignPrefix, s.index), { offset: 0, length: perShard * 4 })]);
	let next = shards.length ? fetchShard(shards[0]) : null;
	for (let i = 0; i < shards.length; i++) {
		const [vbuf, abuf] = await next!;
		next = i + 1 < shards.length ? fetchShard(shards[i + 1]) : null;
		bytes += vbuf.byteLength + abuf.byteLength;
		const v = new Float32Array(vbuf);
		const a = new Uint32Array(abuf);
		// The last shard may hold fewer than perShard vectors.
		const n = Math.min(v.length / dim, a.length);
		for (let x = 0; x < n; x++) if (a[x] >= k) throw new HttpError(500, `shard ${shards[i].index}: cell ${a[x]} >= k ${k}`);
		const r = thread.residuals_for(v.subarray(0, n * dim), means, a.subarray(0, n), dim);
		for (let x = 0; x < r.length; x++) sumSq += r[x] * r[x];
		out.set(r, rows * dim);
		rows += n;
	}
	await putBytes(env.BUCKET, outKey, out.subarray(0, rows * dim));
	return { outKey, rows, bytes, residualEnergy: rows ? sumSq / rows : 0 };
}

interface PqInitReq {
	residualKeys: string[];
	dim: number;
	m: number;
	seed: number;
	outKey: string;
}

/** Initial codebook from the concatenated residual objects (~4000 rows). */
async function stagePqInit(env: Env, body: PqInitReq) {
	await ensureWasm();
	const { residualKeys, dim, m, seed, outKey } = body;
	if (dim % m !== 0) throw new HttpError(400, `dim ${dim} is not a multiple of m ${m}`);
	const parts = await pmap(residualKeys, SUBREQUEST_CONCURRENCY, (k) => getBuffer(env.BUCKET, k));
	const res = new Float32Array(parts.reduce((a, b) => a + b.byteLength, 0) / 4);
	let o = 0;
	for (const p of parts) {
		res.set(new Float32Array(p), o);
		o += p.byteLength / 4;
	}
	parts.length = 0;
	const codebook = thread.pq_init_codebook(res, dim, m, seed >>> 0);
	const info = thread.pq_codebook_info(codebook);
	await putBytes(env.BUCKET, outKey, codebook);
	return { outKey, rows: res.length / dim, bytes: codebook.byteLength, info: Array.from(info) };
}

interface PqTrainMapReq {
	codebookKey: string;
	residualKeys: string[];
	dim: number;
	m: number;
	outKey: string;
}

/** One PQ Lloyd map job: streams residual objects through a PqTrainJob and writes its partial. */
async function stagePqTrainMap(env: Env, body: PqTrainMapReq) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { codebookKey, residualKeys, dim, m, outKey } = body;
	const job = new thread.PqTrainJob(dim, m, new Uint8Array(await getBuffer(env.BUCKET, codebookKey)));
	try {
		const bytes = await streamObjects(env.BUCKET, residualKeys, 1, (b) => {
			job.add_residuals(new Float32Array(b));
		});
		const partial = job.partial();
		await putBytes(env.BUCKET, outKey, partial);
		return { outKey, rows: job.rows(), mean_sq_err: job.mean_sq_err(), bytes, partialBytes: partial.byteLength };
	} finally {
		safeFree(job, epoch);
	}
}

interface PqTrainReduceReq {
	partialKeys: string[];
	prevCodebookKey: string;
	dim: number;
	m: number;
	outKey: string;
}

/** Streams the map partials through a PqReducer and writes the next codebook. */
async function stagePqTrainReduce(env: Env, body: PqTrainReduceReq) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { partialKeys, prevCodebookKey, dim, m, outKey } = body;
	const reducer = new thread.PqReducer(dim, m);
	try {
		const prevPromise = getBuffer(env.BUCKET, prevCodebookKey);
		const bytes = await streamObjects(env.BUCKET, partialKeys, 4, (b) => reducer.add_partial(new Uint8Array(b)));
		const rows = reducer.rows();
		const mean_sq_err = reducer.mean_sq_err();
		const empty_centroids = reducer.empty_centroids();
		const next = reducer.finish(new Uint8Array(await prevPromise));
		await putBytes(env.BUCKET, outKey, next);
		return { outKey, rows, mean_sq_err, empty_centroids, bytes, codebookBytes: next.byteLength };
	} finally {
		safeFree(reducer, epoch);
	}
}

interface AssignReq {
	gen: string;
	coarseKey: string;
	fineKey: string;
	/** `index` (the ingest shard index) is needed in candidate mode; it defaults to the index parsed from the key. */
	shards: { key: string; firstId: number; index?: number }[];
	k1: number;
	k2: number;
	dim: number;
	probeCoarse: number;
	m: number;
	job: string;
	/**
	 * Candidate mode (flat, k1 = 1): when neighborsKey/nNeigh/prevAssignPrefix are set, each vector's
	 * top m cells are chosen among the nNeigh neighbours of its last k-means cell
	 * (`${prevAssignPrefix}/${index:05}.u32`) instead of over all cells.
	 */
	neighborsKey?: string;
	nNeigh?: number;
	prevAssignPrefix?: string;
	/**
	 * Split each shard into this many batches and write one part per batch
	 * (`${job}.${s}` with s = shard position * split + batch), bounding the part buffer. Default 1 (`${job}`).
	 */
	split?: number;
	/** PQ codebook (format 3): records also carry the PQ codes of their residual (`AssignJob.set_codebook`). */
	codebookKey?: string;
}

async function stageAssign(env: Env, body: AssignReq) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { gen, coarseKey, fineKey, k1, k2, dim, m, job: jobName, neighborsKey, nNeigh, prevAssignPrefix, codebookKey } = body;
	const probeCoarse = buildProbeCoarse(body.probeCoarse, k1);
	const split = body.split ?? 1;
	const candidates = neighborsKey !== undefined || nNeigh !== undefined || prevAssignPrefix !== undefined;
	if (candidates && (!neighborsKey || !nNeigh || !prevAssignPrefix)) throw new HttpError(400, "candidate mode needs neighborsKey, nNeigh and prevAssignPrefix");
	if (candidates && k1 !== 1) throw new HttpError(400, "candidate mode requires k1 = 1");
	if (!Number.isInteger(split) || split < 1) throw new HttpError(400, "split must be an integer >= 1");
	const shards = body.shards.map((s) => ({ ...s, index: s.index ?? shardRefs([s.key])[0].index }));
	const job = await (async () => {
		if (candidates) {
			const [fine, neighbors] = await Promise.all([getF32(env.BUCKET, fineKey), getU32(env.BUCKET, neighborsKey!)]);
			if (neighbors.length !== k2 * nNeigh!) throw new HttpError(400, `${neighborsKey} has ${neighbors.length} ids, expected ${k2}x${nNeigh}`);
			return thread.AssignJob.with_candidates(dim, k2, fine, neighbors, nNeigh!, m);
		}
		const [coarse, fine] = await Promise.all([getF32(env.BUCKET, coarseKey), getF32(env.BUCKET, fineKey)]);
		return new thread.AssignJob(dim, k1, k2, coarse, fine, probeCoarse, m);
	})();
	if (codebookKey) {
		try {
			job.set_codebook(new Uint8Array(await getBuffer(env.BUCKET, codebookKey)));
		} catch (e) {
			safeFree(job, epoch);
			throw e;
		}
	}
	const pqM = job.pq_m();
	let vectors = 0;
	let partBytes = 0;
	let parts = 0;
	const groups = Array.from({ length: k1 }, (_, g) => g);
	const flush = async (suffix: string) => {
		await pmap(groups, 4, async (g) => {
			const part = job.take_part(g);
			partBytes += part.byteLength;
			parts++;
			await putBytes(env.BUCKET, `build/${gen}/parts/${g}/${jobName}${suffix}.part`, part);
		});
	};
	try {
		const fetchShard = (s: (typeof shards)[number]) =>
			Promise.all([getBuffer(env.BUCKET, s.key), candidates ? getU32(env.BUCKET, assignKey(prevAssignPrefix!, s.index)) : Promise.resolve(null)]);
		let bytes = 0;
		let next = shards.length ? fetchShard(shards[0]) : null;
		for (let i = 0; i < shards.length; i++) {
			const [buf, prev] = await next!;
			next = i + 1 < shards.length ? fetchShard(shards[i + 1]) : null;
			bytes += buf.byteLength;
			const v = new Float32Array(buf);
			const n = v.length / dim;
			if (prev && prev.length !== n) throw new HttpError(400, `shard ${shards[i].index}: ${n} vectors but ${prev.length} previous assignments`);
			const per = Math.ceil(n / split);
			for (let s = 0; s < split; s++) {
				const a = Math.min(n, s * per);
				const b = Math.min(n, a + per);
				if (b > a) {
					const sub = v.subarray(a * dim, b * dim);
					const firstId = shards[i].firstId + a;
					vectors += prev ? job.add_batch_prev(sub, firstId, prev.subarray(a, b)) : job.add_batch(sub, firstId);
				}
				// Without split the job writes one part after all shards (the g2 layout).
				if (split > 1) await flush(`.${i * split + s}`);
			}
		}
		const records = job.assigned_records();
		if (split === 1) await flush("");
		return { vectors, bytes, records, partBytes, parts, pqM, mode: candidates ? "candidates" : "exhaustive" };
	} finally {
		safeFree(job, epoch);
	}
}

interface FinalizeReq {
	gen: string;
	g: number;
	k2: number;
	dim: number;
	partKeys: string[];
	fineKey: string;
	/** Contiguous subset of j in [0,k2) (default: all k2 cells of group g). */
	cells?: number[];
	/**
	 * Slab format: 2 (default) = 4-bit scan tier + int8 rerank tier; 1 = int8 only; 3 = PQ scan tier + int8
	 * rerank tier (parts from an assign with codebookKey).
	 */
	format?: number;
	/** In-flight part reads (directories and spans). Default 6, the per-invocation connection limit. */
	concurrency?: number;
	/**
	 * Record bytes held in SlabJobs at once. If the chunk's cells together exceed it, the cells are
	 * built in consecutive batches and each batch re-reads its own (smaller) spans. Default 48 MB.
	 */
	budgetBytes?: number;
	/**
	 * Write the rev-0 slabs of a generation that already has a manifest (they are cached as immutable). Off by
	 * default; a rebuild that will be republished to the same gen must pass it here and to publish.
	 */
	overwrite?: boolean;
}

type FinalizeResult = { cell: number; count: number; bytes: number; scanLen: number; rerankOff: number; pqM?: number };

/**
 * Builds the slabs of a contiguous chunk of fine cells of coarse group g. Per part it reads the
 * directory, then ONE contiguous span covering all of the chunk's cells (`part_span`), and feeds
 * each cell's records to its SlabJob with `add_records_range` (no per-cell range reads).
 */
async function stageFinalize(env: Env, body: FinalizeReq) {
	await ensureWasm();
	const { gen, g, k2, dim, partKeys, fineKey } = body;
	// idx/{gen}/slab/{cell} of a published gen is served (and edge-cached) as immutable; only a rebuild asks to replace it.
	if (!body.overwrite) {
		const revs = await listRevs(env, gen);
		if (revs.length) throw new HttpError(409, `${gen} already has manifest revs ${revs.map((r) => r.rev).join(",")}; finalize would overwrite its published slabs (overwrite: true for a rebuild of the same gen)`);
	}
	const format = body.format ?? 2;
	if (format !== 1 && format !== 2 && format !== 3) throw new HttpError(400, "format must be 1, 2 or 3");
	const conc = body.concurrency ?? SUBREQUEST_CONCURRENCY;
	const budget = body.budgetBytes ?? 48 * 1024 * 1024;
	const js = body.cells ?? Array.from({ length: k2 }, (_, j) => j);
	if (js.length === 0) throw new HttpError(400, "no cells");
	if (js.some((j, x) => !Number.isInteger(j) || j < 0 || j >= k2 || (x > 0 && j !== js[x - 1] + 1))) {
		throw new HttpError(400, "cells must be a contiguous ascending range in [0, k2)");
	}
	const first = js[0];
	const count = js.length;
	const dirLen = thread.part_directory_len(k2);

	// Pass 1: directories. Keep per part only the chunk's span and its cells' [off, len].
	const tDir = Date.now();
	const [dirs, means] = await Promise.all([
		pmap(partKeys, conc, async (key) => {
			const d = thread.part_directory(new Uint8Array(await getBuffer(env.BUCKET, key, { offset: 0, length: dirLen })));
			if (d[0] !== k2 || d[1] !== dim) throw new HttpError(500, `part ${key}: k2/dim ${d[0]}/${d[1]} != ${k2}/${dim}`);
			const [spanOff, spanLen] = thread.part_span(d, first, count);
			const ent = new Uint32Array(count * 2);
			for (let x = 0; x < count; x++) {
				ent[x * 2] = d[2 + (first + x) * 3 + 1];
				ent[x * 2 + 1] = d[2 + (first + x) * 3 + 2];
			}
			return { key, spanOff, spanLen, ent };
		}),
		getF32(env.BUCKET, fineKey, { offset: (g * k2 + first) * dim * 4, length: count * dim * 4 }),
	]);
	const dirMs = Date.now() - tDir;

	// Record bytes per cell over all parts, to split the chunk into batches within the budget.
	const cellBytes = new Array(count).fill(0);
	for (const p of dirs) for (let x = 0; x < count; x++) cellBytes[x] += p.ent[x * 2 + 1];
	const batches: [number, number][] = [];
	for (let a = 0; a < count; ) {
		let b = a + 1;
		let sum = cellBytes[a];
		while (b < count && sum + cellBytes[b] <= budget) sum += cellBytes[b++];
		batches.push([a, b]);
		a = b;
	}

	const results: FinalizeResult[] = [];
	let spanReads = 0;
	let spanBytes = 0;
	const tSpan = Date.now();
	for (const [a, b] of batches) {
		const epoch = wasmEpoch;
		const slabs: thread.SlabJob[] = Array.from({ length: b - a }, (_, i) => new thread.SlabJob(dim, g * k2 + first + a + i, means.subarray((a + i) * dim, (a + i + 1) * dim), format));
		const live = new Set<thread.SlabJob>(slabs);
		try {
			if (slabs[0].format() !== format) throw new HttpError(500, `SlabJob writes format ${slabs[0].format()}, asked for ${format}`);
			await pmap(dirs, conc, async (p) => {
				// The whole chunk's span when there is one batch; else this batch's span from the entries.
				let off = p.spanOff;
				let len = p.spanLen;
				if (batches.length > 1) {
					let lo = -1;
					let hi = 0;
					for (let x = a; x < b; x++) {
						const [o, l] = [p.ent[x * 2], p.ent[x * 2 + 1]];
						if (l === 0) continue;
						if (lo < 0) lo = o;
						hi = o + l;
					}
					[off, len] = lo < 0 ? [0, 0] : [lo, hi - lo];
				}
				if (len === 0) return;
				const span = new Uint8Array(await getBuffer(env.BUCKET, p.key, { offset: off, length: len }));
				spanReads++;
				spanBytes += len;
				for (let x = a; x < b; x++) {
					const l = p.ent[x * 2 + 1];
					if (l > 0) slabs[x - a].add_records_range(span, off, p.ent[x * 2], l);
				}
			});
			// Finish and write one slab at a time so only one serialized slab is alive.
			for (let i = 0; i < slabs.length; i++) {
				const s = slabs[i];
				const cell = g * k2 + first + a + i;
				const n = s.len();
				live.delete(s);
				const bytes = s.finish(); // consumes
				const info = thread.slab_info(bytes);
				await putBytes(env.BUCKET, `idx/${gen}/slab/${pad5(cell)}`, bytes);
				results.push({ cell, count: n, bytes: bytes.byteLength, scanLen: info[5], rerankOff: info[6], ...(format === 3 ? { pqM: info[8] } : {}) });
			}
		} finally {
			for (const s of live) safeFree(s, epoch);
		}
	}
	return {
		g,
		format,
		slabs: results.length,
		results,
		parts: partKeys.length,
		batches: batches.length,
		spanReads,
		spanBytes,
		dirMs,
		spanMs: Date.now() - tSpan,
	};
}

/** Query defaults recorded in the manifest; a query parameter that is absent takes its value from here. */
interface QueryDefaults {
	nprobe: number;
	rerank: number;
	pc: number;
	/**
	 * Fine-cell routing of flat generations: "flat" ranks every cell (cell_dots + route_from_dots), "beam" ranks
	 * only the cells of the centroid tree's best `tree.beam` groups (route_beam; needs `manifest.tree`). Absent = flat.
	 */
	route?: RouteMode;
	/** Tree groups a route=beam query searches when it gives no `beam` (absent: tree.beam, the delta encoder's beam). */
	beam?: number;
}

type RouteMode = "flat" | "beam";
const ROUTE_MODES: readonly RouteMode[] = ["flat", "beam"];

/**
 * Centroid tree of a revision (`build_centroid_tree` over its fine means, "VIBC" bytes): `groups` group means, each
 * cell a member of its nearest group. Beam routing and the delta encoder score the group means, then only the cells
 * of the best `beam` groups. The recall figures are measured at build time on sample vectors (base set): the
 * fraction whose exact nearest cell the beam search finds, and the mean fraction of their exact top 3 cells it finds.
 */
interface TreeRef {
	key: string;
	groups: number;
	beam: number;
	iters?: number;
	seed?: number;
	bytes?: number;
	/** tree_recall(sample, beam) and tree_recall_top(sample, beam, 3). */
	nearestRecall?: number;
	top3Recall?: number;
	/** The same for other beams (recallBeams), keyed by beam. */
	recallByBeam?: Record<string, { nearest: number; top3: number }>;
	sampleRows?: number;
}

/** `tree` option of publish / publish-rev. */
interface TreeReq {
	/** Groups of the tree (default 128). */
	groups?: number;
	/** Lloyd iterations over the means (default 10). */
	iters?: number;
	seed?: number;
	/** Default beam (groups searched) for beam routing and the delta encoder (default 8). */
	beam?: number;
	/** Sample objects for the recall figures (default: 8 of build/{gen}/sample/*.f32, spread over the sample). */
	sampleKeys?: string[];
	/** Rows read from the start of each sample object (default 250, ~2000 vectors in all). 0 skips the recall figures. */
	sampleRows?: number;
	/** Extra beams to measure recall for (recallByBeam). */
	recallBeams?: number[];
}

/**
 * `routing` option of publish / publish-rev: compact copies of the fine means (`fine.f16`, `fine.i8`) that readers can
 * route with instead of `fine.f32`, and the means in centroid-tree group order (`fine.grouped.{rev}.{elem}`) for cold
 * routing (manifest.routing; SDK `open({routing, coldRouting})`).
 */
interface RoutingReq {
	/** Tables to write (default both). */
	tables?: ("f16" | "i8")[];
	/** Element type of the tree-grouped table, written when the revision has a centroid tree (default "i8"); false = none. */
	grouped?: RoutingElem | false;
}

function checkRoutingReq(r: RoutingReq): Required<RoutingReq> {
	const tables = r.tables ?? ["f16", "i8"];
	if (!Array.isArray(tables) || tables.some((t) => t !== "f16" && t !== "i8") || new Set(tables).size !== tables.length) throw new HttpError(400, 'routing.tables must be a list of distinct "f16" / "i8"');
	const grouped = r.grouped ?? "i8";
	if (grouped !== false && grouped !== "f32" && grouped !== "f16" && grouped !== "i8") throw new HttpError(400, 'routing.grouped must be "f32", "f16", "i8" or false');
	return { tables, grouped };
}

/**
 * Builds the routing objects of `fineKey`'s means (the compact tables, and the tree-grouped table when `tree` is given)
 * and writes them (`unique`: create-only with a unique key, for revisions; else plain puts, for publish).
 */
async function writeRouting(env: Env, gen: string, rev: number, fineKey: string, fine: Float32Array, dim: number, req: RoutingReq, unique: boolean, tree: { key: string; bytes: Uint8Array } | null): Promise<RoutingTables> {
	await ensureWasm();
	const { tables, grouped } = checkRoutingReq(req);
	const built = buildRoutingObjects(thread, { fineKey, fine, dim, tables, grouped: tree && grouped ? { gen, rev, treeKey: tree.key, tree: tree.bytes, elem: grouped } : undefined });
	const put = async (key: string) => {
		const bytes = built.objects.find(([k]) => k === key)![1];
		return unique ? await putUnique(env, key, bytes) : (await putBytes(env.BUCKET, key, bytes), key);
	};
	const routing: RoutingTables = { fine: fineKey };
	for (const elem of tables) routing[elem] = { ...built.routing[elem]!, key: await put(built.routing[elem]!.key) };
	if (built.routing.grouped) routing.grouped = { ...built.routing.grouped, key: await put(built.routing.grouped.key) };
	return routing;
}

/**
 * `basePack` option of publish / publish-rev: one object per revision holding everything a cold open reads
 * (`idx/{gen}/base.{rev}.threadpack`: manifest, slabs table, tombstone sets, coarse, codebook, tree, one routing
 * table), and a cold variant without the routing table (`base.cold.{rev}.threadpack`) when the revision has a
 * tree-grouped table. Readers find them by (gen, rev); the manifest names them in files.base / files.baseCold.
 */
interface BasePackReq {
	/** The routing table in the base pack (default "auto": the smallest the revision has). */
	routing?: RoutingOption;
	/** Also write the cold base pack (default true; only when the revision has a grouped table). */
	cold?: boolean;
}

/** The files.base / files.baseCold names of revision `rev` (set before the commit; the packs are written after it). */
function basePackFiles(gen: string, rev: number, m: { routing?: RoutingTables; tree?: TreeRef; k1: number; files?: IndexFiles }, req: BasePackReq | false | undefined): Pick<IndexFiles, "base" | "baseCold"> {
	if (req === false) return {};
	const cold = req?.cold !== false && groupedTableOf(m as unknown as SdkManifest) !== null;
	return { base: sdkKeys.basePack(gen, rev), ...(cold ? { baseCold: sdkKeys.baseColdPack(gen, rev) } : {}) };
}

/** Writes the base pack(s) of a committed revision (built from the objects it names, read back from R2). */
async function writeBasePacks(env: Env, gen: string, rev: number, req: BasePackReq | false | undefined): Promise<{ key: string; bytes: number; routing: string | null }[]> {
	if (req === false) return [];
	const store = new R2BlobStore(env.BUCKET);
	const rs = await sdkLoadRevState(store, gen, rev);
	const out: { key: string; bytes: number; routing: string | null }[] = [];
	const kinds: ("base" | "cold")[] = rs.files.baseCold ? ["base", "cold"] : ["base"];
	for (const kind of kinds) {
		const p = await buildBasePack(store, rs, kind, req?.routing ?? "auto");
		await putBytes(env.BUCKET, p.key, p.bytes);
		out.push({ key: p.key, bytes: p.bytes.byteLength, routing: p.routing });
	}
	return out;
}

/** The routing request that reproduces a manifest's tables (for a split commit that writes new means). */
const routingReqOf = (r: RoutingTables | undefined): RoutingReq | null => {
	if (!r) return null;
	const tables = (["f16", "i8"] as const).filter((e) => r[e]);
	return tables.length || r.grouped ? { tables, grouped: r.grouped?.elem ?? false } : null;
};

/** Defaults of manifests without `defaults` (generations before g3). */
const LEGACY_DEFAULTS: QueryDefaults = { nprobe: 1, rerank: 32, pc: 0 };

interface PublishReq {
	gen: string;
	coarseKey: string;
	fineKey: string;
	k1: number;
	k2: number;
	dim: number;
	m: number;
	probeCoarse: number;
	vectors: number;
	/** Build profile name (e.g. "edge", "local"). */
	profile?: string;
	/** Slab format of every slab: 3 = PQ scan + sq8 rerank tiers, 2 = sq4 scan + sq8 rerank tiers, 1 = sq8 only. Default 2. */
	format?: number;
	/** "pq{M}+sq8" (format 3), "sq4+sq8" (format 2) or "sq8" (format 1); derived from format when absent. */
	tiers?: string;
	/** Format 3: the trained PQ codebook, copied to idx/{gen}/pq.bin (its m goes to manifest.pq). */
	codebookKey?: string;
	/** Neighbour-table width used by incremental k-means and candidate assignment. */
	nNeigh?: number;
	defaults?: QueryDefaults;
	notes?: unknown;
	/** Finalize results; written to idx/{gen}/slabs.json (array indexed by cell). */
	slabs?: FinalizeResult[];
	/** Point idx/current.json at this generation (default true). */
	setCurrent?: boolean;
	/** Build a centroid tree over the fine means (idx/{gen}/tree.0.bin, manifest.tree). */
	tree?: TreeReq;
	/** Compact routing tables (idx/{gen}/fine.f16, fine.i8; manifest.routing). Default `{}` (both); false = none. */
	routing?: RoutingReq | false;
	/** Base packs of rev 0 (idx/{gen}/base.0.threadpack, base.cold.0.threadpack). Default `{}`; false = none. */
	basePack?: BasePackReq | false;
	/** Automatic compaction thresholds and split cap (manifest.compaction; the driver sends the profile's). */
	compaction?: CompactionPolicy;
	/**
	 * Rewrite rev 0 of a generation that already has a manifest. Off by default: index objects are cached as
	 * immutable, so a changed manifest.json would not be seen by isolates/colos that cached it. Change defaults
	 * with `publish-rev` instead.
	 */
	overwrite?: boolean;
}

const treeKey = (gen: string, rev: number) => `idx/${gen}/tree.${rev}.bin`;

/** Default sample objects for the tree recall figures: up to 8 of the generation's k-means sample objects. */
function defaultTreeSampleKeys(gen: string, sampleGroups = 80): string[] {
	const n = Math.min(8, sampleGroups);
	return Array.from({ length: n }, (_, i) => `build/${gen}/sample/${Math.floor((i * sampleGroups) / n)}.f32`);
}

function checkTreeReq(t: TreeReq, k: number): Required<Pick<TreeReq, "groups" | "iters" | "seed" | "beam">> {
	const groups = t.groups ?? 128;
	const iters = t.iters ?? 10;
	const seed = t.seed ?? 7;
	const beam = t.beam ?? 8;
	if (!Number.isInteger(groups) || groups < 1 || groups > k) throw new HttpError(400, `tree.groups must be an integer in 1..${k}`);
	if (!Number.isInteger(iters) || iters < 0 || iters > 100) throw new HttpError(400, "tree.iters must be an integer in 0..100");
	if (!Number.isInteger(seed) || seed < 0) throw new HttpError(400, "tree.seed must be a non-negative integer");
	if (!Number.isInteger(beam) || beam < 1 || beam > groups) throw new HttpError(400, `tree.beam must be an integer in 1..${groups}`);
	return { groups, iters, seed, beam };
}

/**
 * Builds the centroid tree over `fineKey` (k1 * k2 means) and measures its recall on sample vectors with a Searcher
 * over the same means. Returns the bytes (not yet written) and the manifest entry without its key.
 */
async function buildTree(env: Env, gen: string, m: { dim: number; k1: number; k2: number }, coarseKey: string, fineKey: string, req: TreeReq, sampleGroups?: number, fineData?: Float32Array): Promise<{ bytes: Uint8Array; ref: Omit<TreeRef, "key"> }> {
	await ensureWasm();
	const epoch = wasmEpoch;
	const k = m.k1 * m.k2;
	const { groups, iters, seed, beam } = checkTreeReq(req, k);
	let bytes: Uint8Array;
	{
		// fineData: the means already in memory (a split commit builds the tree before its fine object is cached).
		const fine = fineData ?? (await getF32(env.BUCKET, fineKey));
		if (fine.length !== k * m.dim) throw new HttpError(500, `${fineKey} has ${fine.length / m.dim} means, expected ${k}`);
		bytes = thread.build_centroid_tree(fine, m.dim, groups, iters, seed);
	}
	const [tDim, tK, tGroups] = thread.centroid_tree_info(bytes);
	if (tDim !== m.dim || tK !== k) throw new HttpError(500, `tree is dim ${tDim} k ${tK}, expected ${m.dim} / ${k}`);
	// (No timings: Date.now() does not advance during CPU work in Workers; the caller sees the request time.)
	const ref: Omit<TreeRef, "key"> = { groups: tGroups, beam, iters, seed, bytes: bytes.byteLength };
	const rows = req.sampleRows ?? 250;
	if (rows > 0) {
		const keys = req.sampleKeys ?? defaultTreeSampleKeys(gen, sampleGroups);
		const rowBytes = m.dim * 4;
		const parts = await Promise.all(
			keys.map(async (key) => {
				const o = await env.BUCKET.get(key, { range: { offset: 0, length: rows * rowBytes } });
				return o ? new Float32Array(await o.arrayBuffer()) : null;
			}),
		);
		const got: Float32Array[] = [];
		for (const p of parts) if (p && p.length >= m.dim) got.push(p);
		const n = got.reduce((a, p) => a + Math.floor(p.length / m.dim), 0);
		if (n > 0) {
			const sample = new Float32Array(n * m.dim);
			let o = 0;
			for (const p of got) {
				const len = Math.floor(p.length / m.dim) * m.dim;
				sample.set(p.subarray(0, len), o);
				o += len;
			}
			const [coarse, fine] = await Promise.all([getF32(env.BUCKET, coarseKey), fineData ?? getF32(env.BUCKET, fineKey)]);
			const searcher = new thread.Searcher(m.dim, m.k1, m.k2, coarse, fine);
			try {
				searcher.set_tree(bytes);
				ref.nearestRecall = +searcher.tree_recall(sample, beam).toFixed(5);
				ref.top3Recall = +searcher.tree_recall_top(sample, beam, 3).toFixed(5);
				const extra = [...new Set(req.recallBeams ?? [])].filter((b) => Number.isInteger(b) && b >= 1 && b !== beam).sort((a, b) => a - b);
				if (extra.length) {
					ref.recallByBeam = {};
					for (const b of extra) ref.recallByBeam[b] = { nearest: +searcher.tree_recall(sample, b).toFixed(5), top3: +searcher.tree_recall_top(sample, b, 3).toFixed(5) };
				}
			} finally {
				safeFree(searcher, epoch);
			}
			ref.sampleRows = n;
		}
	}
	return { bytes, ref };
}

async function stagePublish(env: Env, body: PublishReq) {
	const { gen, coarseKey, fineKey, slabs, setCurrent, codebookKey, overwrite, tree: treeReq, routing: routingReq, basePack: basePackReq, ...rest } = body;
	if (!overwrite) {
		const revs = await listRevs(env, gen);
		if (revs.length) throw new HttpError(409, `${gen} already has manifest revs ${revs.map((r) => r.rev).join(",")}; use publish-rev to change defaults (or overwrite: true for a rebuild of the same gen)`);
	}
	const format = body.format ?? 2;
	if (format !== 1 && format !== 2 && format !== 3) throw new HttpError(400, "format must be 1, 2 or 3");
	if ((format === 3) !== !!codebookKey) throw new HttpError(400, format === 3 ? "format 3 needs codebookKey" : `codebookKey given for format ${format}`);
	// Format 3: the codebook's m names the scan tier ("pq192+sq8").
	let codebook: ArrayBuffer | null = null;
	let pqM = 0;
	if (codebookKey) {
		await ensureWasm();
		codebook = await getBuffer(env.BUCKET, codebookKey);
		const [cbDim, cbM] = thread.pq_codebook_info(new Uint8Array(codebook));
		if (cbDim !== body.dim) throw new HttpError(400, `codebook dim ${cbDim} != ${body.dim}`);
		pqM = cbM;
	}
	const expectTiers = format === 3 ? `pq${pqM}+sq8` : format === 2 ? "sq4+sq8" : "sq8";
	const tiers = body.tiers ?? expectTiers;
	if (tiers !== expectTiers) throw new HttpError(400, `tiers ${tiers} does not match format ${format} (expected ${expectTiers})`);
	if (body.defaults) checkDefaults(body.defaults, treeReq ? ({} as TreeRef) : null);
	if (body.compaction) checkCompactionPolicy(body.compaction);
	let slabTable: SlabMeta[] | null = null;
	if (slabs) {
		const cells = body.k1 * body.k2;
		const table: (SlabMeta | null)[] = new Array(cells).fill(null);
		for (const s of slabs) {
			if (!Number.isInteger(s.cell) || s.cell < 0 || s.cell >= cells) throw new HttpError(400, `slab cell ${s.cell} out of range`);
			if (!(s.scanLen > 0)) throw new HttpError(400, `slab ${s.cell} has no scanLen`);
			if (format >= 2 && !(s.rerankOff > 0)) throw new HttpError(400, `slab ${s.cell} has no rerankOff (v1 slab in a format ${format} generation?)`);
			if (format === 3 && s.count > 0 && s.pqM !== undefined && s.pqM !== pqM) throw new HttpError(400, `slab ${s.cell} has pq_m ${s.pqM}, codebook has ${pqM}`);
			if (format === 1 && s.rerankOff !== 0) throw new HttpError(400, `slab ${s.cell} has a rerank tier in a format 1 generation`);
			table[s.cell] = [s.scanLen, s.rerankOff, s.count, s.bytes];
		}
		const missing = table.findIndex((t) => t === null);
		if (missing >= 0) throw new HttpError(400, `slabs is missing cell ${missing}`);
		slabTable = table as SlabMeta[];
	} else if (format >= 2) throw new HttpError(400, `format ${format} needs slabs (finalize results)`);
	const [coarse, fine] = await Promise.all([getBuffer(env.BUCKET, coarseKey), getBuffer(env.BUCKET, fineKey)]);
	await Promise.all([
		putBytes(env.BUCKET, `idx/${gen}/coarse.f32`, coarse),
		putBytes(env.BUCKET, `idx/${gen}/fine.f32`, fine),
		slabTable ? putBytes(env.BUCKET, `idx/${gen}/slabs.json`, JSON.stringify(slabTable)) : Promise.resolve(),
		codebook ? putBytes(env.BUCKET, `idx/${gen}/pq.bin`, codebook) : Promise.resolve(),
	]);
	let tree: TreeRef | undefined;
	let treeBytes: Uint8Array | null = null;
	if (treeReq) {
		const notes = body.notes as { cfg?: { sampleGroups?: number } } | undefined;
		const t = await buildTree(env, gen, body, coarseKey, fineKey, treeReq, notes?.cfg?.sampleGroups);
		tree = { key: treeKey(gen, 0), ...t.ref };
		treeBytes = t.bytes;
		await putBytes(env.BUCKET, tree.key, t.bytes);
	}
	const routing = routingReq === false ? undefined : await writeRouting(env, gen, 0, `idx/${gen}/fine.f32`, new Float32Array(fine), body.dim, routingReq ?? {}, false, tree && treeBytes ? { key: tree.key, bytes: treeBytes } : null);
	// Rev 0 of the generation (manifest.json / slabs.json / slab/{cell:05}); later revisions are manifest.{rev}.json.
	const packFiles = basePackFiles(gen, 0, { routing, tree, k1: body.k1 }, basePackReq);
	// Rev 0 names its files only when it names base packs (older readers use the fixed names either way).
	const files: IndexFiles | undefined = packFiles.base ? { coarse: `idx/${gen}/coarse.f32`, fine: `idx/${gen}/fine.f32`, pq: format === 3 ? `idx/${gen}/pq.bin` : null, slabs: `idx/${gen}/slabs.json`, ...packFiles } : undefined;
	const manifest = { gen, ...rest, format, tiers, ...(format === 3 ? { pq: { m: pqM } } : {}), ...(tree ? { tree } : {}), ...(routing ? { routing } : {}), ...(files ? { files } : {}), rev: 0, createdAt: new Date().toISOString() };
	await putBytes(env.BUCKET, manifestKey(gen, 0), JSON.stringify(manifest));
	const basePacks = slabTable ? await writeBasePacks(env, gen, 0, basePackReq) : [];
	// current.json carries a copy of the manifest when it is small (<= 16 KB): an SDK open skips the manifest GET.
	if (setCurrent !== false) await putBytes(env.BUCKET, CURRENT_KEY, currentBody(gen, 0, manifest as unknown as SdkManifest));
	return { ...manifest, basePacks, current: setCurrent !== false };
}

// ---------- revisions: object names, manifests, current pointer ----------

/**
 * Row of a slabs table (array indexed by cell): [scanLen, rerankOff, count, bytes, slabRev?]. rerankOff is 0 for
 * format 1. slabRev (absent = 0) names the slab object: 0 = `slab/{cell:05}` (written by finalize), r > 0 =
 * `slab/{cell:05}.{r}` (written by a compaction that read manifest rev r - 1).
 */
type SlabMeta = [number, number, number, number] | [number, number, number, number, number];
const slabRevOf = (row: SlabMeta | undefined): number => (row && row.length > 4 ? (row[4] as number) : 0);

/** A delta segment listed in a manifest. */
interface DeltaRef {
	/** Sequence number stored in the segment header (allocated from manifest.nextSeq). */
	seq: number;
	key: string;
	count: number;
	bytes: number;
	/** The manifest rev that first listed this delta: tombstone sets with rev > addedRev hide its entries. */
	addedRev: number;
	/** Distinct primary cells of its entries. */
	cells: number;
}

/** A tombstone set: ids deleted (or replaced by an upsert) at manifest rev `rev`. */
interface TombRef {
	rev: number;
	key: string;
	count: number;
}

/** Objects a revision reads. Manifests without `files` (rev 0) use the generation's fixed names. */
interface IndexFiles {
	coarse: string;
	fine: string;
	pq: string | null;
	slabs: string | null;
	/** The base pack of the revision that wrote it (base.{rev}.threadpack); inherited by later revisions. */
	base?: string;
	/** The cold base pack of that revision (base.cold.{rev}.threadpack). */
	baseCold?: string;
}

interface Manifest {
	gen: string;
	k1: number;
	k2: number;
	dim: number;
	/** Attribute schema (thread-sdk `configureAttrs`): named fields packed into each entry's attrs word. */
	attrs?: AttrSchema;
	m: number;
	probeCoarse: number;
	vectors: number;
	/** 3 = PQ scan tier + int8 rerank tier; 2 = 4-bit scan tier + int8 rerank tier; absent/1 = legacy int8 slabs. */
	format?: number;
	/** g3+: build profile and its query defaults. */
	profile?: string;
	tiers?: string;
	/** Format 3: PQ subspaces of the scan tier (codebook in files.pq). */
	pq?: { m: number };
	nNeigh?: number;
	defaults?: QueryDefaults;
	/** Centroid tree over the fine means (beam routing, delta encoding); inherited by later revisions. */
	tree?: TreeRef;
	createdAt?: string;
	notes?: unknown;
	/** Revision; absent = 0 (the `publish` output, idx/{gen}/manifest.json). */
	rev?: number;
	parentRev?: number;
	updatedAt?: string;
	/** What produced this revision: {op: "publish-rev" | "upsert" | "delete" | "compact", ...}. */
	change?: Record<string, unknown>;
	files?: IndexFiles;
	deltas?: DeltaRef[];
	tombstones?: TombRef[];
	/** Next delta sequence number. */
	nextSeq?: number;
	/** The next id a keyed upsert allocates (SDK writers); absent = `vectors`. */
	nextId?: number;
	/** Automatic compaction thresholds and the cell split cap (absent: no automatic compaction, no splits). */
	compaction?: CompactionPolicy;
	/** Compact routing tables (fine.f16 / fine.i8) made from the means `routing.fine`; ignored by readers once files.fine differs. */
	routing?: RoutingTables;
	/** Lexical / docs configuration written by the SDK (`index.configureLexical`); opaque here. */
	lex?: Record<string, unknown>;
	/** Lexical / docs segments (SDK, `idx/{gen}/seg/*`): filtered by tombstone sets with rev > addedRev, like deltas. */
	segments?: { seq: number; key: string; addedRev: number; count: number; bytes: number }[];
}

/**
 * Compaction policy of a generation (manifest.compaction; profile defaults, changed with publish-rev). After every
 * upsert/delete commit the writer compares the head with the thresholds and, when one is exceeded and no compaction
 * holds idx/{gen}/compact.lock, compacts a bounded slice in ctx.waitUntil (see autoCompact).
 */
interface CompactionPolicy {
	/** Delta segments listed in the head (exceeded when more). */
	maxDeltas: number;
	/** Total delta segment bytes. */
	maxDeltaBytes: number;
	/** Tombstone ids over all sets (with repeats). */
	maxTombstones: number;
	/** A compaction splits every rewritten cell whose entry count would exceed this (0 / absent = never split). */
	maxCellEntries?: number;
	/**
	 * Cells one automatic run may rewrite when the writer rewrites them itself (slots = 1; whole deltas only, the rest
	 * waits for the next trigger).
	 */
	cellsPerRun?: number;
	/**
	 * Dynamic build slots an automatic run spreads its cells over (default AUTO_SLOTS; 1 = the writer rewrites them
	 * itself, as before the Worker Loader). Without env.SLOTS (this module on its own) a run always uses 1.
	 */
	slots?: number;
	/** With slots > 1: cells per slot, i.e. a run takes up to cellsPerSlot x slots cells (default AUTO_CELLS_PER_SLOT). */
	cellsPerSlot?: number;
	/** Wall-clock budget of one automatic run, from the trigger (ms); it aborts before committing when exceeded. */
	deadlineMs?: number;
	/**
	 * An automatic run splits cells only when its slice has at most this many cells (a split commit rewrites the fine
	 * means and rebuilds the tree, ~5 s more); bigger slices defer their splits to a later run or bench/compact.mjs.
	 */
	splitCellsPerRun?: number;
}

/** Profile defaults of manifest.compaction. */
const COMPACTION_PROFILES: Record<string, CompactionPolicy> = {
	edge: { maxDeltas: 16, maxDeltaBytes: 32 * 1024 * 1024, maxTombstones: 50_000, maxCellEntries: 3000, cellsPerRun: 256, deadlineMs: 27_000 },
	local: { maxDeltas: 16, maxDeltaBytes: 32 * 1024 * 1024, maxTombstones: 50_000, maxCellEntries: 1000, cellsPerRun: 256, deadlineMs: 27_000 },
};
const AUTO_CELLS_PER_RUN = 256;
/**
 * Default compaction.slots: build slots (auto-0 ..) an automatic run spreads its cells over, each a dynamic Worker
 * with its own isolate and 6 R2 connections. 4 = the most dynamic Workers one request may have in flight.
 */
const AUTO_SLOTS = 4;
const AUTO_MAX_SLOTS = 4;
/**
 * Default compaction.cellsPerSlot. A slot rewrites 10-14 g4 cells/s (6 R2 connections; g4 trigger runs: 920 cells on
 * 4 slots in 19.4-23.0 s), so 160 cells per slot leave the same margin to the deadline as the single-writer 228-cell
 * slices did (16.5-20 s): 640 cells per run = two to three 250-vector segments instead of one.
 */
const AUTO_CELLS_PER_SLOT = 160;
/** A compact-cells job gets at least this many cells (each job loads the compaction base and the folded deltas). */
const AUTO_MIN_CELLS_PER_SLOT = 24;
/** Chunks per slot on the shared queue of an automatic run (load balancing across slots on faster/slower hosts). */
const AUTO_CHUNKS_PER_SLOT = 4;
/**
 * ctx.waitUntil work may run 30 s after the response; a run gives up before committing past this (from the trigger).
 * Measured: one Worker rewrites ~13 cells/s at concurrency 6 (256 g4 cells, 330 MB each way, in 19.4 s).
 */
const AUTO_DEADLINE_MS = 27_000;
/** Time reserved for the commit when deciding whether to start another cell (more with splits: fine + tree). */
const AUTO_COMMIT_RESERVE_MS = 2_500;
const AUTO_SPLIT_COMMIT_RESERVE_MS = 7_000;
/**
 * Default splitCellsPerRun: (27 s - 7 s) at 11-14 cells/s minus a margin. A first g4 test with 228 cells and splits ran
 * into the deadline before its last cells (no commit); without splits the same slice commits in ~19 s.
 */
const AUTO_SPLIT_CELLS_PER_RUN = 160;
/** A compact.lock older than this is considered abandoned (its writer crashed or was cut off). */
const COMPACT_LOCK_STALE_MS = 10 * 60 * 1000;

function checkCompactionPolicy(c: CompactionPolicy): CompactionPolicy {
	for (const name of ["maxDeltas", "maxDeltaBytes", "maxTombstones"] as const) {
		if (!Number.isInteger(c[name]) || c[name] < 0) throw new HttpError(400, `compaction.${name} must be a non-negative integer`);
	}
	for (const name of ["maxCellEntries", "cellsPerRun", "deadlineMs", "splitCellsPerRun", "slots", "cellsPerSlot"] as const) {
		const v = c[name];
		if (v !== undefined && (!Number.isInteger(v) || v < 0)) throw new HttpError(400, `compaction.${name} must be a non-negative integer`);
	}
	if (c.maxCellEntries !== undefined && c.maxCellEntries > 0 && c.maxCellEntries < 2) throw new HttpError(400, "compaction.maxCellEntries must be 0 (off) or >= 2");
	if (c.cellsPerRun !== undefined && c.cellsPerRun < 1) throw new HttpError(400, "compaction.cellsPerRun must be >= 1");
	if (c.slots !== undefined && (c.slots < 1 || c.slots > AUTO_MAX_SLOTS)) throw new HttpError(400, `compaction.slots must be 1..${AUTO_MAX_SLOTS}`);
	if (c.cellsPerSlot !== undefined && c.cellsPerSlot < 1) throw new HttpError(400, "compaction.cellsPerSlot must be >= 1");
	return c;
}

/** Update pressure of a revision: what the compaction thresholds are compared with. */
function compactionPressure(m: Manifest) {
	const deltas = m.deltas ?? [];
	const tombs = m.tombstones ?? [];
	return {
		deltas: deltas.length,
		deltaBytes: deltas.reduce((a, d) => a + d.bytes, 0),
		deltaEntries: deltas.reduce((a, d) => a + d.count, 0),
		tombstones: tombs.reduce((a, t) => a + t.count, 0),
		tombSets: tombs.length,
	};
}

/** The thresholds a revision exceeds (empty: no compaction needed, or no policy). */
function compactionExceeded(m: Manifest): string[] {
	const c = m.compaction;
	if (!c) return [];
	const p = compactionPressure(m);
	const out: string[] = [];
	if (p.deltas > c.maxDeltas) out.push(`deltas ${p.deltas} > ${c.maxDeltas}`);
	if (p.deltaBytes > c.maxDeltaBytes) out.push(`deltaBytes ${p.deltaBytes} > ${c.maxDeltaBytes}`);
	if (p.tombstones > c.maxTombstones) out.push(`tombstones ${p.tombstones} > ${c.maxTombstones}`);
	return out;
}

/** idx/current.json: the generation and revision queries use by default. */
interface Current {
	gen: string;
	rev: number;
}

const CURRENT_KEY = "idx/current.json";
const manifestKey = (gen: string, rev: number) => (rev === 0 ? `idx/${gen}/manifest.json` : `idx/${gen}/manifest.${rev}.json`);
const slabsTableKey = (gen: string, rev: number) => (rev === 0 ? `idx/${gen}/slabs.json` : `idx/${gen}/slabs.${rev}.json`);
const slabKey = (gen: string, cell: number, slabRev = 0) => (slabRev === 0 ? `idx/${gen}/slab/${pad5(cell)}` : `idx/${gen}/slab/${pad5(cell)}.${slabRev}`);
const tombKey = (gen: string, rev: number) => `idx/${gen}/tombstones.${rev}.u32`;
const deltaKey = (gen: string, seq: number) => `idx/${gen}/delta/${seq}.vibd`;

function manifestFiles(m: Manifest): IndexFiles {
	if (m.files) return m.files;
	const g = m.gen;
	return { coarse: `idx/${g}/coarse.f32`, fine: `idx/${g}/fine.f32`, pq: (m.format ?? 1) === 3 ? `idx/${g}/pq.bin` : null, slabs: `idx/${g}/slabs.json` };
}

const decodeJson = <T>(buf: ArrayBuffer): T => JSON.parse(new TextDecoder().decode(buf)) as T;

/** Every object under `prefix` (paginated R2 list). */
async function listAll(env: Env, prefix: string): Promise<R2Object[]> {
	const out: R2Object[] = [];
	let cursor: string | undefined;
	do {
		const r = await env.BUCKET.list({ prefix, cursor, limit: 1000 });
		out.push(...r.objects);
		cursor = r.truncated ? r.cursor : undefined;
	} while (cursor);
	return out;
}

/** Manifest revisions of `gen` that exist in R2, ascending, with their upload times. */
async function listRevs(env: Env, gen: string): Promise<{ rev: number; uploaded: Date }[]> {
	const prefix = `idx/${gen}/manifest.`;
	const revs: { rev: number; uploaded: Date }[] = [];
	for (const o of await listAll(env, prefix)) {
		const m = o.key.slice(prefix.length).match(/^(?:(\d+)\.)?json$/);
		if (m) revs.push({ rev: m[1] ? Number(m[1]) : 0, uploaded: o.uploaded });
	}
	return revs.sort((a, b) => a.rev - b.rev);
}

/** The newest manifest revision of `gen` (the head writers build on). */
async function latestRev(env: Env, gen: string): Promise<number> {
	const revs = await listRevs(env, gen);
	if (!revs.length) throw new HttpError(404, `no manifest for generation ${gen}`);
	return revs.at(-1)!.rev;
}

/** A manifest read straight from R2 (writers; legacy rev-0 manifests may have been republished, so not through the cache). */
async function readManifestR2(env: Env, gen: string, rev: number): Promise<Manifest> {
	const m = decodeJson<Manifest>(await getBuffer(env.BUCKET, manifestKey(gen, rev)));
	return { ...m, rev: m.rev ?? rev };
}

async function readCurrent(env: Env): Promise<{ cur: Current; etag: string } | null> {
	const o = await env.BUCKET.get(CURRENT_KEY);
	if (!o) return null;
	const j = (await o.json()) as { gen: string; rev?: number };
	return { cur: { gen: j.gen, rev: j.rev ?? 0 }, etag: o.etag };
}

// idx/current.json is the only mutable index object; an isolate re-reads it at most every CURRENT_TTL_MS so a warm
// query does not pay an R2 round trip just to learn the generation and revision.
let currentPtr: { cur: Current; at: number } | null = null;
const CURRENT_TTL_MS = 30_000;
/** Head revision of generations other than the current one (`/query?gen=` without `rev`), same TTL. */
const genHeads = new Map<string, { rev: number; at: number }>();

async function getCurrent(env: Env): Promise<Current> {
	if (currentPtr && Date.now() - currentPtr.at < CURRENT_TTL_MS) return currentPtr.cur;
	const r = await readCurrent(env);
	if (!r) throw new HttpError(404, "no published index");
	currentPtr = { cur: r.cur, at: Date.now() };
	return r.cur;
}

/**
 * The (gen, rev) a query reads: `?gen=&rev=` explicitly, `?rev=` of the current generation, `?gen=` at its head
 * (the current rev if it is the current generation), else idx/current.json.
 */
async function resolveTarget(env: Env, genParam: string | null, revParam: number | null): Promise<Current> {
	if (genParam && revParam !== null) return { gen: genParam, rev: revParam };
	const cur = await getCurrent(env);
	if (!genParam || genParam === cur.gen) return revParam !== null ? { gen: cur.gen, rev: revParam } : cur;
	const h = genHeads.get(genParam);
	if (h && Date.now() - h.at < CURRENT_TTL_MS) return { gen: genParam, rev: h.rev };
	const rev = await latestRev(env, genParam);
	genHeads.set(genParam, { rev, at: Date.now() });
	return { gen: genParam, rev };
}

// ---------- query: index loading and per-isolate caches ----------

/** Routing table and PQ codebook of a generation; shared by every revision with the same files. */
interface LoadedBase {
	key: string;
	gen: string;
	searcher: thread.Searcher;
	/** Format 3: the parsed PQ codebook; each query builds its PqTable from it. */
	pq: thread.PqCodebook | null;
	/**
	 * The centroid tree installed in `searcher` (route_beam), or null. Not part of the base key: revisions that add
	 * or change a tree share the loaded means (no second 25 MB Searcher), and the tree a query's revision names is
	 * installed with `useTree` right before it routes (set_tree replaces the previous one; it is ~1 MB to parse).
	 */
	tree: TreeRef | null;
	/** wasmEpoch the wasm objects were created in (they may be freed only in the same epoch). */
	epoch: number;
	loadMs: number;
	/** Cache API outcome per file when this isolate loaded the base. */
	loadCache: Record<string, CacheOutcome>;
}

/** Tombstone sets of a revision, parsed. */
interface TombState {
	sets: { rev: number; ids: Uint32Array }[];
	/** Ids over all sets (with repeats). */
	total: number;
	/** Union of the sets with rev >= r, sorted unique u32 LE (the scan skip buffer for slabs with slabRev r, and for deltas with addedRev r - 1), by r. */
	unions: Map<number, Uint8Array>;
}

/** One manifest revision: everything a query needs besides the base (JS objects only, no wasm). */
interface RevState {
	gen: string;
	rev: number;
	manifest: Manifest;
	files: IndexFiles;
	baseKey: string;
	/** Per-cell slab layout (format 2/3); null for v1 generations, whose slabs are read whole. */
	slabs: SlabMeta[] | null;
	deltas: DeltaRef[];
	tomb: TombState | null;
	loadMs: number;
	loadCache: Record<string, CacheOutcome>;
}

type CacheMode = "none" | "mem" | "edge";
type CacheOutcome = "mem" | "hit" | "miss" | "none";

/**
 * Loaded bases per isolate. A query holds its slot (users) while it runs; a slot replaced by another base (beyond
 * BASE_CACHE_SIZE) is freed once its last query releases it, so switching generations does not leak the ~25 MB
 * Searcher in wasm memory. Revisions of one generation share the base as long as coarse/fine/pq keys are unchanged.
 */
interface BaseSlot {
	key: string;
	promise: Promise<LoadedBase>;
	users: number;
	evicted: boolean;
}
const baseCache = new Map<string, BaseSlot>();
const BASE_CACHE_SIZE = 1;
/** Parsed revisions per isolate (immutable, keyed `${gen}@${rev}`), most recently used last. */
const revCache = new Map<string, Promise<RevState>>();
const REV_CACHE_SIZE = 8;
/** Parsed slabs tables by key (shared by the revisions between two compactions). */
const slabsTableCache = new Map<string, Promise<SlabMeta[] | null>>();
const SLABS_TABLE_CACHE_SIZE = 3;
// In-isolate slab cache (cache=mem). Kept small: concurrent requests share one isolate's 128 MB.
const slabCache = new Map<string, Uint8Array>();
let slabCacheBytes = 0;
const SLAB_CACHE_LIMIT = 16 * 1024 * 1024;
/**
 * Delta segments by key (immutable). Queries share one copy per isolate; a segment beyond the budget is still
 * shared by the queries that read it concurrently, then dropped.
 */
const deltaCache = new Map<string, Promise<Uint8Array>>();
const deltaCacheSizes = new Map<string, number>();
let deltaCacheBytes = 0;
const DELTA_CACHE_LIMIT = 40 * 1024 * 1024;
/** Distinct cells of each delta segment (from thread.delta_cells), by key: what beam-routed queries compute dots for. */
const deltaCellsCache = new Map<string, Uint32Array>();
const DELTA_CELLS_CACHE_SIZE = 256;

/** Distinct cells (ascending) holding entries of the delta segment `key`, cached per isolate. */
function deltaCellsOf(key: string, bytes: Uint8Array): Uint32Array {
	let cells = deltaCellsCache.get(key);
	if (cells) {
		lruTouch(deltaCellsCache, key, cells);
		return cells;
	}
	const pairs = thread.delta_cells(bytes);
	cells = new Uint32Array(pairs.length / 2);
	for (let i = 0; i < cells.length; i++) cells[i] = pairs[2 * i];
	deltaCellsCache.set(key, cells);
	while (deltaCellsCache.size > DELTA_CELLS_CACHE_SIZE) deltaCellsCache.delete(deltaCellsCache.keys().next().value!);
	return cells;
}

function lruTouch<K, V>(m: Map<K, V>, k: K, v: V) {
	m.delete(k);
	m.set(k, v);
}

// Cache API layer in front of R2. Objects under idx/{gen}/ are immutable (new revisions get new keys), so
// they are cached under a synthetic GET key with a one-year TTL.
const CACHE_ORIGIN = "https://vib-bench.aiuitech.workers.dev/_cache/";
const CACHE_CONTROL = "public, max-age=31536000, immutable";
/** rr=runs: rerank winners of one slab whose records lie within this span are read with one range request. */
const COALESCE_SPAN = 64 * 1024;
/** rr=multi (default): winners within this span share one range of the multi-range request (i.e. adjacent records). */
const MULTI_COALESCE = 4 * 1024;

function cacheRequest(key: string, range?: { offset: number; length: number }): Request {
	const headers: Record<string, string> = {};
	if (range) headers.range = `bytes=${range.offset}-${range.offset + range.length - 1}`;
	return new Request(CACHE_ORIGIN + key.replace(/^idx\//, ""), { method: "GET", headers });
}

/** Stores a full immutable object in caches.default in the background. */
/**
 * caches.default, or null where the Cache API is unavailable (a dynamic slot Worker may not have it): reads then
 * go straight to R2.
 */
function edgeCache(): Cache | null {
	try {
		return typeof caches !== "undefined" ? caches.default : null;
	} catch {
		return null;
	}
}

function putCache(ctx: ExecutionContext, key: string, buf: ArrayBuffer) {
	const cache = edgeCache();
	if (!cache) return;
	const resp = new Response(buf.slice(0), {
		headers: {
			"content-type": "application/octet-stream",
			"content-length": String(buf.byteLength),
			"cache-control": CACHE_CONTROL,
			"accept-ranges": "bytes",
		},
	});
	ctx.waitUntil(cache.put(cacheRequest(key), resp).catch((e) => console.error(`cache.put ${key} failed: ${e}`)));
}

/**
 * Reads an immutable index object (`idx/{gen}/...`) through caches.default.
 * On a miss it reads R2 and stores a copy in the background via ctx.waitUntil.
 */
async function getCached(env: Env, ctx: ExecutionContext, key: string): Promise<{ buf: ArrayBuffer; outcome: CacheOutcome }> {
	const cache = edgeCache();
	const hit = cache ? await cache.match(cacheRequest(key)).catch(() => undefined) : undefined;
	if (hit) return { buf: await hit.arrayBuffer(), outcome: "hit" };
	const buf = await getBuffer(env.BUCKET, key);
	putCache(ctx, key, buf);
	return { buf, outcome: "miss" };
}

async function getCachedOptional(env: Env, ctx: ExecutionContext, key: string): Promise<{ buf: ArrayBuffer; outcome: CacheOutcome } | null> {
	try {
		return await getCached(env, ctx, key);
	} catch (e) {
		if (e instanceof HttpError && e.status === 404) return null;
		throw e;
	}
}

/**
 * Bounds a request's in-flight subrequests (cache.match + body read, R2 GETs). Workers keep at
 * most 6 connections open per invocation and cancel stalled responses beyond that
 * ("Response closed due to connection limit"), so reads are queued here instead.
 */
class Limiter {
	private active = 0;
	private queue: (() => void)[] = [];
	/** Low-priority waiters (speculative reads): started only when no normal read is waiting. */
	private low: (() => void)[] = [];
	/** Reads retried after "connection limit" cancellations. */
	retries = 0;
	constructor(private max: number) {}
	async run<T>(fn: () => Promise<T>, low = false): Promise<T> {
		if (this.active >= this.max) await new Promise<void>((r) => (low ? this.low : this.queue).push(r));
		this.active++;
		try {
			return await fn();
		} finally {
			this.active--;
			(this.queue.shift() ?? this.low.shift())?.();
		}
	}
}
const SUBREQUEST_CONCURRENCY = 6;
/**
 * In-flight Cache API reads per query. Measured on g2 (data/bench-latency.txt): Cache API reads are not held
 * to the 6-connection limit (26 in flight: no "connection limit" cancellations), and 8-12 in flight cut
 * phase 1 by ~10 ms at c=1..50 against 6. R2 reads (cache misses) still retry once on a cancellation.
 */
const QUERY_INFLIGHT = 12;

/** Runs one read; retries once if the runtime closed it for the connection limit, the connection dropped or R2 reported an internal error. */
async function readWithRetry<T>(lim: Limiter, fn: () => Promise<T>, low = false): Promise<T> {
	try {
		return await lim.run(fn, low);
	} catch (e) {
		const transient = e instanceof Error && ["connection limit", "(10001)", "Network connection lost"].some((m) => e.message.includes(m));
		if (!transient) throw e;
		lim.retries++;
		return await lim.run(fn, low);
	}
}

/** How caches.default answered Range requests (206 = honoured, 200 = full body returned). */
interface RangeStats {
	r206: number;
	r200: number;
	miss: number;
}

/**
 * Range read of a cached object. Returns null when the object is not cached. If the cache
 * ignores the Range header (status 200) the full body is returned as `full` as well.
 */
async function cacheMatchRange(key: string, offset: number, length: number, rs: RangeStats): Promise<{ bytes: Uint8Array; full: Uint8Array | null } | null> {
	const r = await caches.default.match(cacheRequest(key, { offset, length }));
	if (!r) {
		rs.miss++;
		return null;
	}
	const body = new Uint8Array(await r.arrayBuffer());
	if (r.status === 206) {
		if (body.byteLength !== length) throw new HttpError(500, `cache range ${key} [${offset}, +${length}) returned ${body.byteLength} bytes`);
		rs.r206++;
		return { bytes: body, full: null };
	}
	rs.r200++;
	if (body.byteLength < offset + length) throw new HttpError(500, `cached ${key} is ${body.byteLength} bytes, need ${offset + length}`);
	return { bytes: body.subarray(offset, offset + length), full: body };
}

/** Frees a loaded base's wasm objects, unless they belong to an instance that was reset since. */
function freeLoadedBase(b: LoadedBase) {
	if (b.epoch !== wasmEpoch) return;
	safeFree(b.searcher, b.epoch);
	if (b.pq) safeFree(b.pq, b.epoch);
}

/** Removes a slot from the cache; its base is freed now, or by the last query still using it. */
function dropBaseSlot(slot: BaseSlot) {
	if (slot.evicted) return;
	slot.evicted = true;
	if (baseCache.get(slot.key) === slot) baseCache.delete(slot.key);
	if (slot.users === 0) slot.promise.then(freeLoadedBase, () => {});
}

const baseKeyOf = (gen: string, f: IndexFiles) => `${gen}|${f.coarse}|${f.fine}|${f.pq ?? "-"}`;

/** Centroid tree bytes by key (immutable, ~1 MB each), fetched through the cache layer before a query needs them. */
const treeBytes = new Map<string, Promise<{ buf: Uint8Array; outcome: CacheOutcome }>>();
const TREE_BYTES_CACHE_SIZE = 2;
const treeBytesReady = new Map<string, Uint8Array>();

function fetchTreeBytes(env: Env, ctx: ExecutionContext, key: string): Promise<{ buf: Uint8Array; outcome: CacheOutcome }> {
	let p = treeBytes.get(key);
	if (p) {
		lruTouch(treeBytes, key, p);
		return p.then((r) => ({ buf: r.buf, outcome: "mem" as CacheOutcome }));
	}
	p = getCached(env, ctx, key).then((r) => {
		const buf = new Uint8Array(r.buf);
		treeBytesReady.set(key, buf);
		return { buf, outcome: r.outcome };
	});
	treeBytes.set(key, p);
	const mine = p;
	p.catch(() => treeBytes.get(key) === mine && treeBytes.delete(key));
	while (treeBytes.size > TREE_BYTES_CACHE_SIZE) {
		const old = treeBytes.keys().next().value!;
		treeBytes.delete(old);
		treeBytesReady.delete(old);
	}
	return p;
}

/**
 * Makes `base.searcher` hold `tree` (synchronous; the bytes must have been fetched with fetchTreeBytes). Called
 * right before routing, with no await in between, so concurrent queries of revisions with different trees cannot
 * route through the wrong one.
 */
function useTree(base: LoadedBase, tree: TreeRef) {
	if (base.tree?.key === tree.key) return;
	const bytes = treeBytesReady.get(tree.key);
	if (!bytes) throw new HttpError(500, `tree ${tree.key} not loaded`);
	base.searcher.set_tree(bytes);
	if (base.searcher.tree_groups() !== tree.groups) throw new HttpError(500, `${tree.key} has ${base.searcher.tree_groups()} groups, manifest says ${tree.groups}`);
	base.tree = tree;
}

/** Parses a PQ codebook and checks it against the manifest. */
function parseCodebook(buf: ArrayBuffer, m: Manifest, key: string): thread.PqCodebook {
	const epoch = wasmEpoch;
	const pq = new thread.PqCodebook(new Uint8Array(buf));
	if (pq.dim() !== m.dim || (m.pq && pq.m() !== m.pq.m)) {
		const msg = `${key} is dim ${pq.dim()} m ${pq.m()}, manifest dim ${m.dim} pq ${JSON.stringify(m.pq)}`;
		safeFree(pq, epoch);
		throw new HttpError(500, msg);
	}
	return pq;
}

async function buildBase(env: Env, ctx: ExecutionContext, rs: RevState): Promise<LoadedBase> {
	basesLoaded++;
	writerMeans = null; // this isolate serves queries now: do not keep a second 25 MB copy of the means
	await ensureWasm();
	const epoch = wasmEpoch;
	const t0 = Date.now();
	const { manifest, files } = rs;
	const tree = manifest.tree ?? null;
	const [c, f, pqObj, treeObj] = await Promise.all([
		getCached(env, ctx, files.coarse),
		getCached(env, ctx, files.fine),
		files.pq ? getCached(env, ctx, files.pq) : Promise.resolve(null),
		tree ? fetchTreeBytes(env, ctx, tree.key) : Promise.resolve(null),
	]);
	for (const [name, r] of [[files.coarse, c], [files.fine, f]] as const) {
		if (r.buf.byteLength % 4 !== 0) throw new HttpError(500, `object ${name} is not f32-aligned`);
	}
	if ((manifest.format ?? 1) === 3 && !pqObj) throw new HttpError(500, `format 3 generation ${rs.gen} has no PQ codebook`);
	const pq = pqObj ? parseCodebook(pqObj.buf, manifest, files.pq!) : null;
	let searcher: thread.Searcher | null = null;
	try {
		searcher = new thread.Searcher(manifest.dim, manifest.k1, manifest.k2, new Float32Array(c.buf), new Float32Array(f.buf));
		if (tree && treeObj) {
			searcher.set_tree(treeObj.buf);
			if (searcher.tree_groups() !== tree.groups) throw new HttpError(500, `${tree.key} has ${searcher.tree_groups()} groups, manifest says ${tree.groups}`);
		}
	} catch (e) {
		if (searcher) safeFree(searcher, epoch);
		if (pq) safeFree(pq, epoch);
		throw e;
	}
	return {
		key: rs.baseKey,
		gen: rs.gen,
		searcher,
		pq,
		tree,
		epoch,
		loadMs: Date.now() - t0,
		loadCache: { coarse: c.outcome, fine: f.outcome, ...(pqObj ? { pq: pqObj.outcome } : {}), ...(treeObj ? { tree: treeObj.outcome } : {}) },
	};
}

/** The loaded base of a revision, held until `release()` (call it once the query is done). */
async function acquireBase(env: Env, ctx: ExecutionContext, rs: RevState): Promise<{ base: LoadedBase; release: () => void }> {
	const key = rs.baseKey;
	let slot = baseCache.get(key);
	if (slot) lruTouch(baseCache, key, slot);
	else {
		const promise = buildBase(env, ctx, rs);
		const fresh: BaseSlot = { key, promise, users: 0, evicted: false };
		slot = fresh;
		baseCache.set(key, fresh);
		promise.catch(() => {
			if (baseCache.get(key) === fresh) baseCache.delete(key);
		});
		while (baseCache.size > BASE_CACHE_SIZE) dropBaseSlot(baseCache.values().next().value!);
	}
	const held = slot;
	held.users++;
	let released = false;
	const release = () => {
		if (released) return;
		released = true;
		held.users--;
		if (held.evicted && held.users === 0) held.promise.then(freeLoadedBase, () => {});
	};
	try {
		return { base: await held.promise, release };
	} catch (e) {
		release();
		throw e;
	}
}

async function loadSlabsTable(env: Env, ctx: ExecutionContext, key: string, optional: boolean): Promise<{ table: SlabMeta[] | null; outcome: CacheOutcome }> {
	const hit = slabsTableCache.get(key);
	if (hit) {
		lruTouch(slabsTableCache, key, hit);
		return { table: await hit, outcome: "mem" };
	}
	let outcome: CacheOutcome = "none";
	const p = (optional ? getCachedOptional(env, ctx, key) : getCached(env, ctx, key)).then((r) => {
		if (!r) return null;
		outcome = r.outcome;
		return decodeJson<SlabMeta[]>(r.buf);
	});
	slabsTableCache.set(key, p);
	p.catch(() => slabsTableCache.get(key) === p && slabsTableCache.delete(key));
	while (slabsTableCache.size > SLABS_TABLE_CACHE_SIZE) slabsTableCache.delete(slabsTableCache.keys().next().value!);
	const table = await p;
	return { table, outcome };
}

/** Sorted unique union of id sets as a tombstone buffer (`u32` LE). */
function idUnion(sets: Uint32Array[]): Uint8Array {
	const all = new Uint32Array(sets.reduce((a, s) => a + s.length, 0));
	let o = 0;
	for (const s of sets) {
		all.set(s, o);
		o += s.length;
	}
	all.sort();
	let n = 0;
	for (let i = 0; i < all.length; i++) if (i === 0 || all[i] !== all[i - 1]) all[n++] = all[i];
	return new Uint8Array(all.buffer, 0, n * 4);
}

function buildTombState(sets: { rev: number; ids: Uint32Array }[]): TombState {
	return { sets, total: sets.reduce((a, s) => a + s.ids.length, 0), unions: new Map() };
}

/**
 * Tombstone buffer for slab hits of a slab with revision `slabRev`: the union of the sets with rev >= slabRev.
 * A slab written by a compaction that read manifest rev R has slabRev R + 1 and already excludes every id of
 * the sets with rev <= R; rev-0 slabs are hidden by every set.
 */
function tombFor(t: TombState, slabRev: number): Uint8Array {
	let u = t.unions.get(slabRev);
	if (!u) {
		u = idUnion(t.sets.filter((s) => s.rev >= slabRev).map((s) => s.ids));
		t.unions.set(slabRev, u);
	}
	return u;
}

async function loadRevState(env: Env, ctx: ExecutionContext, gen: string, rev: number): Promise<RevState> {
	const t0 = Date.now();
	// Rev 0 (manifest.json) is read from R2, not through the edge cache: before revisions existed it was
	// republished in place to change defaults, so a colo may still cache an older copy. Revisions >= 1 are
	// created once (create-only PUT) and are safe to cache.
	const m =
		rev === 0
			? await env.BUCKET.get(manifestKey(gen, 0)).then(async (o) => (o ? { buf: await o.arrayBuffer(), outcome: "none" as CacheOutcome } : null))
			: await getCachedOptional(env, ctx, manifestKey(gen, rev));
	if (!m) throw new HttpError(404, `no manifest ${manifestKey(gen, rev)}`);
	const manifest = decodeJson<Manifest>(m.buf);
	if (manifest.gen !== gen) throw new HttpError(500, `${manifestKey(gen, rev)} is for generation ${manifest.gen}`);
	const files = manifestFiles(manifest);
	const format = manifest.format ?? 1;
	const tombRefs = manifest.tombstones ?? [];
	const [s, tombs] = await Promise.all([
		files.slabs ? loadSlabsTable(env, ctx, files.slabs, format < 2 && !manifest.files) : Promise.resolve({ table: null, outcome: "none" as CacheOutcome }),
		Promise.all(
			tombRefs.map(async (t) => {
				const r = await getCached(env, ctx, t.key);
				if (r.buf.byteLength % 4 !== 0) throw new HttpError(500, `${t.key} is not u32-aligned`);
				return { rev: t.rev, ids: new Uint32Array(r.buf) };
			}),
		),
	]);
	if (format >= 2 && !s.table) throw new HttpError(500, `${files.slabs} missing for format ${format}`);
	if (s.table && s.table.length !== manifest.k1 * manifest.k2) throw new HttpError(500, `${files.slabs} has ${s.table.length} cells, expected ${manifest.k1 * manifest.k2}`);
	return {
		gen,
		rev,
		manifest,
		files,
		baseKey: baseKeyOf(gen, files),
		slabs: s.table,
		deltas: manifest.deltas ?? [],
		tomb: tombs.length ? buildTombState(tombs) : null,
		loadMs: Date.now() - t0,
		loadCache: { manifest: m.outcome, slabs: s.outcome },
	};
}

/** The parsed revision (gen, rev), cached per isolate (revisions are immutable). */
function getRevState(env: Env, ctx: ExecutionContext, gen: string, rev: number): Promise<RevState> {
	const key = `${gen}@${rev}`;
	const hit = revCache.get(key);
	if (hit) {
		lruTouch(revCache, key, hit);
		return hit;
	}
	const p = loadRevState(env, ctx, gen, rev);
	revCache.set(key, p);
	p.catch(() => revCache.get(key) === p && revCache.delete(key));
	while (revCache.size > REV_CACHE_SIZE) revCache.delete(revCache.keys().next().value!);
	return p;
}

/**
 * A delta segment through the isolate cache and the Cache API. Concurrent queries share one in-flight read, so
 * the read is started at once and NOT queued in the first query's Limiter: a request awaiting I/O that another
 * request has not started yet is cancelled by the runtime as hung (seen with 5 MB segments at c=50).
 */
async function getDelta(env: Env, ctx: ExecutionContext, ref: DeltaRef): Promise<{ bytes: Uint8Array; outcome: CacheOutcome }> {
	const hit = deltaCache.get(ref.key);
	if (hit) {
		lruTouch(deltaCache, ref.key, hit);
		return { bytes: await hit, outcome: "mem" };
	}
	let outcome: CacheOutcome = "none";
	const p = getCached(env, ctx, ref.key).then((r) => {
		outcome = r.outcome;
		return new Uint8Array(r.buf);
	});
	deltaCache.set(ref.key, p);
	deltaCacheSizes.set(ref.key, ref.bytes);
	deltaCacheBytes += ref.bytes;
	const drop = (k: string) => {
		deltaCache.delete(k);
		deltaCacheBytes -= deltaCacheSizes.get(k) ?? 0;
		deltaCacheSizes.delete(k);
	};
	p.catch(() => deltaCache.get(ref.key) === p && drop(ref.key));
	// Evict least recently used segments (never the one just added) to stay within the budget.
	for (const k of deltaCache.keys()) {
		if (deltaCacheBytes <= DELTA_CACHE_LIMIT || k === ref.key) break;
		drop(k);
	}
	return { bytes: await p, outcome };
}

function memCachePut(key: string, buf: Uint8Array) {
	if (buf.byteLength >= SLAB_CACHE_LIMIT / 4) return;
	if (slabCacheBytes + buf.byteLength > SLAB_CACHE_LIMIT) {
		slabCache.clear();
		slabCacheBytes = 0;
	}
	slabCache.set(key, buf);
	slabCacheBytes += buf.byteLength;
}

/** v1: the whole slab object. */
async function fetchSlab(env: Env, ctx: ExecutionContext, key: string, mode: CacheMode): Promise<{ buf: Uint8Array; outcome: CacheOutcome }> {
	if (mode === "edge") {
		const r = await getCached(env, ctx, key);
		return { buf: new Uint8Array(r.buf), outcome: r.outcome };
	}
	if (mode === "mem") {
		const hit = slabCache.get(key);
		if (hit) return { buf: hit, outcome: "mem" };
	}
	const buf = new Uint8Array(await getBuffer(env.BUCKET, key));
	if (mode === "mem") memCachePut(key, buf);
	return { buf, outcome: mode === "mem" ? "miss" : "none" };
}

interface ScanSource {
	/** Bytes to scan: the v2 `[0, scanLen)` prefix, or the whole v1 slab. */
	prefix: Uint8Array;
	/** The whole slab when it was read whole (edge miss), so rerank records need no second read. */
	full: Uint8Array | null;
	outcome: CacheOutcome;
	/** Bytes transferred for this slab in phase 1. */
	bytes: number;
}

/**
 * Debug/experiment switches of /query (all optional; the defaults are the production path).
 *   mode=fetchonly   phase 1 reads every prefix and drops it (no scan, no phase 2)
 *   mode=noop        return right after routing (fixed cost: RTT, upload, dispatch, index lookup)
 *   scanRepeat=N     scan each prefix N times (client latency delta / (N-1) = scan CPU per query)
 *   copyRepeat=N     N extra thread.slab_info(prefix) calls per slab: each copies the prefix into wasm memory
 *   copy=1           header + centroid read through wasm (slab_info/slab_centroid copy the whole prefix in);
 *                    copy=0 parses them in JS from the first bytes
 *   inflight=N       per-request in-flight subrequest limit (default 6)
 *   rr=runs|span|multi  rerank reads: runs within `coalesce` bytes (default), one span per slab when it is
 *                    within `spanMax` bytes, or one multi-range request per slab
 *   spec=1           speculative rerank reads: once only `specAt` slabs are left to scan, read the records of the
 *                    running top (T * specMargin / 100) at low priority (idle connection slots during phase 1's
 *                    tail); phase 2 then reads only the winners that were not prefetched
 */
interface DebugOpts {
	mode: "" | "fetchonly" | "noop";
	scanRepeat: number;
	copyRepeat: number;
	copy: number;
	inflight: number;
	rr: "runs" | "span" | "multi";
	coalesce: number;
	spanMax: number;
	/** 1 = speculative rerank reads of the running top-T once only `specAt` slabs remain unscanned. */
	spec: number;
	specAt: number;
	/** Speculate on the running top (T * specMargin / 100). */
	specMargin: number;
}

const DEFAULT_DEBUG: DebugOpts = {
	mode: "",
	scanRepeat: 1,
	copyRepeat: 0,
	copy: 0,
	inflight: QUERY_INFLIGHT,
	rr: "multi",
	coalesce: COALESCE_SPAN,
	spanMax: 256 * 1024,
	spec: 1,
	// Speculating with 6 slabs left beat 8/12 at c=300 (the running top-T is closer to final).
	specAt: 6,
	specMargin: 100,
};

function enumParam<T extends string>(url: URL, name: string, allowed: readonly T[], def: T): T {
	const raw = url.searchParams.get(name);
	if (raw === null || raw === "") return def;
	if (!(allowed as readonly string[]).includes(raw)) throw new HttpError(400, `${name} must be one of ${allowed.join("|")}`);
	return raw as T;
}

function parseDebug(url: URL): DebugOpts {
	const d = DEFAULT_DEBUG;
	return {
		mode: enumParam(url, "mode", ["", "fetchonly", "noop"] as const, d.mode),
		scanRepeat: Math.min(intParam(url, "scanRepeat", d.scanRepeat, 1), 16),
		copyRepeat: Math.min(intParam(url, "copyRepeat", d.copyRepeat, 0), 16),
		copy: Math.min(intParam(url, "copy", d.copy, 0), 1),
		inflight: Math.min(intParam(url, "inflight", d.inflight, 1), 32),
		rr: enumParam(url, "rr", ["runs", "span", "multi"] as const, d.rr),
		// rr=multi asks for each record as its own range (adjacent records merge), so bytes stay minimal.
		coalesce: intParam(url, "coalesce", url.searchParams.get("rr") === "runs" || url.searchParams.get("rr") === "span" ? d.coalesce : MULTI_COALESCE, 0),
		spanMax: intParam(url, "spanMax", d.spanMax, 0),
		spec: Math.min(intParam(url, "spec", d.spec, 0), 1),
		specAt: intParam(url, "specAt", d.specAt, 0),
		specMargin: Math.min(intParam(url, "specMargin", d.specMargin, 100), 400),
	};
}

/** Header fields of a slab / v2 scan prefix (the first 64 bytes), parsed in JS so the prefix is not copied into wasm. */
interface SlabHdr {
	version: number;
	dim: number;
	count: number;
	centroidOff: number;
	scanLen: number;
	rerankOff: number;
	recLen: number;
	/** v3: PQ subspaces of the scan tier (0 otherwise). */
	pqM: number;
}

function parseSlabHeader(b: Uint8Array): SlabHdr {
	if (b.byteLength < 64) throw new HttpError(500, `slab header needs 64 bytes, have ${b.byteLength}`);
	const dv = new DataView(b.buffer, b.byteOffset, 64);
	if (dv.getUint32(0, true) !== 0x53424956) throw new HttpError(500, "bad slab magic"); // "VIBS"
	const version = dv.getUint16(4, true);
	const dim = dv.getUint32(8, true);
	const count = dv.getUint32(12, true);
	const centroidOff = dv.getUint32(20, true);
	// v2 and v3 share the layout of the fields read here (v3: pq_m at 32 instead of scales4_off).
	if (version === 2 || version === 3) {
		return { version, dim, count, centroidOff, scanLen: dv.getUint32(48, true), rerankOff: dv.getUint32(36, true), recLen: 4 + dim, pqM: version === 3 ? dv.getUint32(32, true) : 0 };
	}
	if (version === 1) return { version, dim, count, centroidOff, scanLen: dv.getUint32(36, true), rerankOff: 0, recLen: 0, pqM: 0 };
	throw new HttpError(500, `unknown slab version ${version}`);
}

function slabHdrFromWasm(prefix: Uint8Array): SlabHdr {
	const a = thread.slab_info(prefix);
	return { version: a[4], dim: a[0], count: a[1], centroidOff: 64, scanLen: a[5], rerankOff: a[6], recLen: a[7], pqM: a[8] };
}

/** The cell mean at `centroidOff` (copied out, so it outlives the prefix). */
function centroidFromBytes(b: Uint8Array, h: SlabHdr): Float32Array {
	const s = b.slice(h.centroidOff, h.centroidOff + h.dim * 4);
	return new Float32Array(s.buffer, 0, h.dim);
}

/**
 * Phase 1 read. v2 + edge: Range request for the scan prefix against the cached full object;
 * on a miss GET the whole object from R2, cache it in the background, and keep it in memory.
 */
async function fetchScanSource(env: Env, ctx: ExecutionContext, rv: RevState, cell: number, mode: CacheMode, rs: RangeStats, lim: Limiter): Promise<ScanSource> {
	return readWithRetry(lim, () => fetchScanSourceOnce(env, ctx, rv, cell, mode, rs));
}

async function fetchScanSourceOnce(env: Env, ctx: ExecutionContext, rv: RevState, cell: number, mode: CacheMode, rs: RangeStats): Promise<ScanSource> {
	if (!rv.slabs) {
		const { buf, outcome } = await fetchSlab(env, ctx, slabKey(rv.gen, cell), mode);
		return { prefix: buf, full: null, outcome, bytes: buf.byteLength };
	}
	const meta = rv.slabs[cell];
	if (!meta) throw new HttpError(500, `${rv.files.slabs} has no entry for cell ${cell}`);
	const scanLen = meta[0];
	const key = slabKey(rv.gen, cell, slabRevOf(meta));
	if (mode === "edge") {
		const hit = await cacheMatchRange(key, 0, scanLen, rs);
		if (hit) return { prefix: hit.bytes, full: hit.full, outcome: "hit", bytes: (hit.full ?? hit.bytes).byteLength };
		const buf = await getBuffer(env.BUCKET, key);
		putCache(ctx, key, buf);
		const full = new Uint8Array(buf);
		return { prefix: full.subarray(0, scanLen), full, outcome: "miss", bytes: full.byteLength };
	}
	const memKey = `${key}#scan`;
	if (mode === "mem") {
		const hit = slabCache.get(memKey);
		if (hit) return { prefix: hit, full: null, outcome: "mem", bytes: 0 };
	}
	const prefix = new Uint8Array(await getBuffer(env.BUCKET, key, { offset: 0, length: scanLen }));
	if (mode === "mem") memCachePut(memKey, prefix);
	return { prefix, full: null, outcome: mode === "mem" ? "miss" : "none", bytes: prefix.byteLength };
}

/** A byte range of one object. */
interface ByteRange {
	offset: number;
	length: number;
}

/** Parts of a (possibly multipart/byteranges) 206 body: absolute [start, end] and the bytes. */
interface RangePart {
	start: number;
	end: number;
	bytes: Uint8Array;
}

const latin1 = new TextDecoder("latin1");

/** Splits a multipart/byteranges body into its parts (RFC 9110 §14.6). */
function parseMultipart(body: Uint8Array, boundary: string): RangePart[] {
	const parts: RangePart[] = [];
	const delim = new TextEncoder().encode(`--${boundary}`);
	const find = (pat: Uint8Array, from: number) => {
		outer: for (let i = from; i + pat.length <= body.length; i++) {
			for (let j = 0; j < pat.length; j++) if (body[i + j] !== pat[j]) continue outer;
			return i;
		}
		return -1;
	};
	const crlf2 = new Uint8Array([13, 10, 13, 10]);
	let pos = find(delim, 0);
	while (pos >= 0) {
		const after = pos + delim.length;
		if (body[after] === 45 && body[after + 1] === 45) break; // closing delimiter "--"
		const hdrEnd = find(crlf2, after);
		if (hdrEnd < 0) throw new HttpError(500, "multipart: unterminated part headers");
		const headers = latin1.decode(body.subarray(after, hdrEnd));
		const m = headers.match(/content-range:\s*bytes\s+(\d+)-(\d+)\//i);
		if (!m) throw new HttpError(500, `multipart: part without Content-Range: ${headers.slice(0, 200)}`);
		const start = Number(m[1]);
		const end = Number(m[2]);
		const dataStart = hdrEnd + 4;
		const dataEnd = dataStart + (end - start + 1);
		if (dataEnd > body.length) throw new HttpError(500, "multipart: part runs past the body");
		parts.push({ start, end, bytes: body.subarray(dataStart, dataEnd) });
		pos = find(delim, dataEnd);
	}
	return parts;
}

/**
 * Several ranges of one cached object in ONE Cache API request (`Range: bytes=a-b, c-d`).
 * Handles a multipart/byteranges 206, a single-part 206 (ranges merged by the server) and a 200
 * (Range ignored). Returns one Uint8Array per requested range, or null when the object is not cached.
 */
async function cacheMatchMulti(key: string, ranges: ByteRange[], rs: RangeStats): Promise<{ out: Uint8Array[]; multipart: boolean; wireBytes: number } | null> {
	const hdr = "bytes=" + ranges.map((r) => `${r.offset}-${r.offset + r.length - 1}`).join(", ");
	const r = await caches.default.match(new Request(cacheRequest(key).url, { method: "GET", headers: { range: hdr } }));
	if (!r) {
		rs.miss++;
		return null;
	}
	const body = new Uint8Array(await r.arrayBuffer());
	let parts: RangePart[];
	let multipart = false;
	const ct = r.headers.get("content-type") ?? "";
	if (r.status === 206 && /multipart\/byteranges/i.test(ct)) {
		const b = ct.match(/boundary=("?)([^";]+)\1/i);
		if (!b) throw new HttpError(500, `multipart without boundary: ${ct}`);
		parts = parseMultipart(body, b[2]);
		multipart = true;
		rs.r206++;
	} else if (r.status === 206) {
		const m = (r.headers.get("content-range") ?? "").match(/bytes\s+(\d+)-(\d+)\//);
		if (!m) throw new HttpError(500, `206 without Content-Range for ${key}`);
		parts = [{ start: Number(m[1]), end: Number(m[2]), bytes: body }];
		rs.r206++;
	} else {
		parts = [{ start: 0, end: body.length - 1, bytes: body }];
		rs.r200++;
	}
	const out = ranges.map((rg) => {
		const p = parts.find((p) => p.start <= rg.offset && rg.offset + rg.length - 1 <= p.end);
		if (!p) throw new HttpError(500, `range ${rg.offset}+${rg.length} of ${key} missing from the response (status ${r.status}, ${parts.length} parts)`);
		return p.bytes.subarray(rg.offset - p.start, rg.offset - p.start + rg.length);
	});
	return { out, multipart, wireBytes: body.byteLength };
}

interface Hit {
	id: number;
	score: number;
	idx: number;
}

/** Parses packed hits `[count u32][id u32, score f32, idx u32]*`. */
function parseHits(packed: Uint8Array): Hit[] {
	const dv = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
	const n = dv.getUint32(0, true);
	const out: Hit[] = new Array(n);
	for (let i = 0; i < n; i++) {
		const o = 4 + i * 12;
		out[i] = { id: dv.getUint32(o, true), score: dv.getFloat32(o + 4, true), idx: dv.getUint32(o + 8, true) };
	}
	return out;
}

interface QueryParams {
	k: number;
	/** null = take the value from manifest.defaults (LEGACY_DEFAULTS for older manifests). */
	nprobe: number | null;
	probeCoarse: number | null;
	/** null = manifest.defaults.route (absent: flat). */
	route: RouteMode | null;
	/** Tree groups searched with route=beam; null = manifest.tree.beam. */
	beam: number | null;
	mode: CacheMode;
	rerank: number | null;
	/** Attribute filter (field conditions under manifest.attrs; `{attrs: ...}` on the raw word without a schema). */
	filter: AttrFilterExpr | null;
	/** With a filter: cells to scan at most while widening past nprobe (default 4 x nprobe). */
	maxNprobe: number | null;
	gen: string;
	rev: number;
	tStart: number;
	tGen: number;
	dbg: DebugOpts;
}

/** Query limits: k and rerank size wasm heaps per probed slab; a query body holds at most 65536 f32 values. */
const MAX_K = 1000;
const MAX_RERANK = 4096;
const MAX_QUERY_BYTES = 4 * 65536;

function intParam(url: URL, name: string, def: number, min = 0): number {
	return optIntParam(url, name, min) ?? def;
}

/** An integer query parameter, or null when absent (or empty). */
function optIntParam(url: URL, name: string, min = 0): number | null {
	const raw = url.searchParams.get(name);
	if (raw === null || raw === "") return null;
	const v = Number(raw);
	if (!Number.isInteger(v) || v < min) throw new HttpError(400, `${name} must be an integer >= ${min}`);
	return v;
}

async function handleQuery(request: Request, env: Env, ctx: ExecutionContext, url: URL): Promise<Response> {
	const tStart = Date.now();
	const k = intParam(url, "k", 10, 1);
	// Absent nprobe/pc/rerank take the revision's manifest.defaults (resolved once the manifest is loaded).
	const nprobe = optIntParam(url, "nprobe", 1);
	// pc = coarse groups to probe; 0 = flat routing over all k1*k2 fine cells.
	const probeCoarse = optIntParam(url, "pc", 0);
	// rerank = scan-tier candidates re-scored with the int8 tier (formats 2 and 3); 0 disables phase 2.
	const rerank = optIntParam(url, "rerank", 0);
	// filter = an attribute filter expression as JSON (also accepted in a JSON body as `filter`); maxNprobe bounds
	// the widening of a filtered query's probes.
	let filter: AttrFilterExpr | null = null;
	const filterRaw = url.searchParams.get("filter");
	if (filterRaw) {
		try {
			filter = JSON.parse(filterRaw) as AttrFilterExpr;
		} catch {
			throw new HttpError(400, "filter must be a JSON object");
		}
	}
	const maxNprobe = optIntParam(url, "maxNprobe", 1);
	// route = flat | beam (pruned routing through the centroid tree), beam = tree groups searched.
	const routeRaw = url.searchParams.get("route");
	if (routeRaw !== null && routeRaw !== "" && !ROUTE_MODES.includes(routeRaw as RouteMode)) throw new HttpError(400, `route must be ${ROUTE_MODES.join("|")}`);
	const route = routeRaw ? (routeRaw as RouteMode) : null;
	const beam = optIntParam(url, "beam", 1);
	// k and rerank size per-slab heaps in wasm; nprobe/pc are checked against the manifest in runQueryOn.
	if (k > MAX_K) throw new HttpError(400, `k must be <= ${MAX_K}`);
	if (rerank !== null && rerank > MAX_RERANK) throw new HttpError(400, `rerank must be <= ${MAX_RERANK}`);
	const cacheParam = url.searchParams.get("cache") ?? "edge";
	if (cacheParam !== "none" && cacheParam !== "mem" && cacheParam !== "edge") throw new HttpError(400, "cache must be none|mem|edge");
	const dbg = parseDebug(url);
	const { gen, rev } = await resolveTarget(env, url.searchParams.get("gen"), optIntParam(url, "rev", 0));
	const tGen = Date.now();

	// Read the body once, outside the retry scope.
	let q: Float32Array;
	const ct = request.headers.get("content-type") ?? "";
	if (ct.includes("octet-stream")) {
		const buf = await request.arrayBuffer();
		if (buf.byteLength > MAX_QUERY_BYTES) throw new HttpError(413, `query body must be <= ${MAX_QUERY_BYTES} bytes`);
		if (buf.byteLength % 4) throw new HttpError(400, "query body must be little-endian f32 values");
		q = new Float32Array(buf);
	} else {
		const body = await readJson<{ vector: unknown; filter?: unknown }>(request);
		if (!Array.isArray(body?.vector) || body.vector.length > MAX_QUERY_BYTES / 4 || !body.vector.every((x) => typeof x === "number")) throw new HttpError(400, "vector must be an array of numbers");
		q = Float32Array.from(body.vector as number[]);
		if (body.filter !== undefined && body.filter !== null) filter = body.filter as AttrFilterExpr;
	}
	if (filter !== null && (typeof filter !== "object" || Array.isArray(filter))) throw new HttpError(400, "filter must be a JSON object of field conditions");
	// Finite and nonzero (normalize_vector would otherwise yield NaN scores).
	let norm2 = 0;
	for (const x of q) {
		if (!Number.isFinite(x)) throw new HttpError(400, "query vector must be finite");
		norm2 += x * x;
	}
	if (!(norm2 > 0)) throw new HttpError(400, "query vector must have nonzero norm");
	const p: QueryParams = { k, nprobe, probeCoarse, route, beam, mode: cacheParam, rerank, filter, maxNprobe, gen, rev, tStart, tGen, dbg };
	return withWasmRecovery("query", (attempt) => runQuery(request, env, ctx, q, p, attempt));
}

/** Per-query wasm objects freed when the query ends (only in the epoch they were created in). */
interface QueryScratch {
	table: thread.PqTable | null;
	epoch: number;
}

async function runQuery(request: Request, env: Env, ctx: ExecutionContext, qRaw: Float32Array, p: QueryParams, attempt: number): Promise<Response> {
	const rv = await getRevState(env, ctx, p.gen, p.rev);
	const { base, release } = await acquireBase(env, ctx, rv);
	const scratch: QueryScratch = { table: null, epoch: wasmEpoch };
	try {
		// The revision's tree (if any) is fetched now; runQueryOn installs it synchronously before routing.
		if (rv.manifest.tree && base.tree?.key !== rv.manifest.tree.key) await fetchTreeBytes(env, ctx, rv.manifest.tree.key);
		return await runQueryOn(rv, base, request, env, ctx, qRaw, p, attempt, scratch);
	} finally {
		if (scratch.table) safeFree(scratch.table, scratch.epoch);
		release();
	}
}

async function runQueryOn(rv: RevState, base: LoadedBase, request: Request, env: Env, ctx: ExecutionContext, qRaw: Float32Array, p: QueryParams, attempt: number, scratch: QueryScratch): Promise<Response> {
	const { k, mode, gen, tStart, tGen, dbg } = p;
	const manifest = rv.manifest;
	if (qRaw.length !== manifest.dim) throw new HttpError(400, `query dim ${qRaw.length} != ${manifest.dim}`);
	const defaults = manifest.defaults ?? LEGACY_DEFAULTS;
	const nprobe = p.nprobe ?? defaults.nprobe;
	const probeCoarse = p.probeCoarse ?? defaults.pc;
	const rerankParam = p.rerank ?? defaults.rerank;
	const source = (v: number | null) => (v !== null ? "query" : manifest.defaults ? "manifest" : "legacy");
	// Bounded by the index shape (route_from_dots / route allocate per probe); rejected, not clamped.
	if (nprobe > manifest.k1 * manifest.k2) throw new HttpError(400, `nprobe must be <= ${manifest.k1 * manifest.k2} (k1*k2)`);
	if (probeCoarse > manifest.k1) throw new HttpError(400, `pc must be <= ${manifest.k1} (k1)`);
	// The attribute filter, compiled against the generation's schema (thread-sdk's attrs module); applied inside
	// every scan before top-k insertion. With one, phase 1 may widen past nprobe (see slabPhase).
	let filterWords: Uint32Array;
	try {
		filterWords = compileFilter(manifest.attrs ? validateSchema(manifest.attrs as AttrSchema) : null, p.filter);
	} catch (e) {
		throw new HttpError(400, e instanceof Error ? e.message : String(e));
	}
	const filtered = filterWords.length > 0;
	if (p.maxNprobe !== null && p.maxNprobe < nprobe) throw new HttpError(400, `maxNprobe must be >= nprobe (${nprobe})`);
	const maxNprobe = filtered ? Math.min(manifest.k1 * manifest.k2, p.maxNprobe ?? nprobe * 4) : nprobe;
	const q = thread.normalize_vector(qRaw);
	const dim = manifest.dim;
	const format = manifest.format ?? (rv.slabs ? 2 : 1);
	// Format 1 slabs have no rerank tier: phase 2 is skipped whatever rerank says.
	const doRerank = format >= 2 && rv.slabs !== null && rerankParam > 0 && dbg.mode !== "fetchonly";
	const T = doRerank ? Math.max(k, rerankParam) : k;
	const flat = probeCoarse === 0 || probeCoarse >= manifest.k1;
	// Fine-cell routing of a flat query: every cell (flat) or the cells of the tree's best `beam` groups (beam).
	const routeMode: RouteMode = p.route ?? defaults.route ?? "flat";
	const useBeam = flat && routeMode === "beam";
	const tree = manifest.tree ?? null;
	if (routeMode === "beam" && !tree) throw new HttpError(400, `route=beam needs a centroid tree (${gen} rev ${rv.rev} has none)`);
	const beam = useBeam ? (p.beam ?? defaults.beam ?? tree!.beam) : null;
	if (beam !== null && beam > tree!.groups) throw new HttpError(400, `beam must be <= ${tree!.groups} (tree groups)`);
	const deltas = dbg.mode === "noop" ? [] : rv.deltas;
	const tomb = rv.tomb;
	const lim = new Limiter(dbg.inflight);
	// Delta segments are read first (usually from the isolate cache) and scanned while the slab reads are in flight.
	const deltaReads = deltas.map((d) => getDelta(env, ctx, d));

	const tRoute = Date.now();
	// Flat routing: one q·mean per fine cell (cell_dots), ranked by route_from_dots and reused as scan_delta's
	// per-cell base scores. Beam and two-level routing compute no full table: scan_delta gets `qc`, filled with
	// cell_dots_for for just the cells the delta segments hold entries in (see deltaQc).
	const cellDots = flat && !useBeam ? base.searcher.cell_dots(q) : null;
	if (useBeam) useTree(base, tree!);
	// Routed to the widening bound; an unfiltered query scans exactly its nprobe cells.
	const routeN = Math.max(nprobe, maxNprobe);
	const cells = Array.from(useBeam ? base.searcher.route_beam(q, beam!, routeN) : flat ? base.searcher.route_from_dots(cellDots!, routeN) : base.searcher.route(q, probeCoarse, routeN));
	const nCells = manifest.k1 * manifest.k2;
	let qcPartial: { qc: Float32Array; filled: Uint8Array } | null = null;
	let dotsComputed = cellDots ? nCells : 0;
	/** scan_delta's `qc` for one segment: the full table, or the partial one with this segment's cells filled in. */
	const deltaQc = (d: DeltaRef, bytes: Uint8Array): Float32Array => {
		if (cellDots) return cellDots;
		qcPartial ??= { qc: new Float32Array(nCells), filled: new Uint8Array(nCells) };
		const { qc, filled } = qcPartial;
		const missing = deltaCellsOf(d.key, bytes).filter((c) => !filled[c]);
		if (missing.length) {
			const dots = base.searcher.cell_dots_for(q, missing);
			missing.forEach((c, i) => {
				qc[c] = dots[i];
				filled[c] = 1;
			});
			dotsComputed += missing.length;
		}
		return qc;
	};
	// Format 3: one asymmetric-distance table (m x 256 f32) per query, shared by every probed slab.
	const table = base.pq && dbg.mode !== "noop" ? (scratch.table = thread.PqTable.from_codebook(q, base.pq)) : null;
	const tFetch = Date.now();
	if (dbg.mode === "noop") {
		return json({ gen, rev: rv.rev, ids: [], scores: [], cells, stats: { scanned: 0, slabBytes: 0, cacheHits: 0, cacheMisses: 0, debug: dbg }, timing: { total: Date.now() - tStart, gen: tGen - tStart, load: tRoute - tStart, fetch: 0 } });
	}

	// ---- phase 1: scan-tier scan of each probed slab, processed as it arrives so its
	// buffer can be dropped while the remaining reads are in flight.
	const rs: RangeStats = { r206: 0, r200: 0, miss: 0 };
	const packed: Uint8Array[] = [];
	/** Best scan-tier copy per id (same rule as thread.merge), remembering its cell. */
	const cand = new Map<number, { cell: number; idx: number; score: number }>();
	/** Rerank records of local winners from slabs that were read whole: `${cell}:${idx}`. */
	const local = new Map<string, Uint8Array>();
	const slabInfo = new Map<number, { centroid: Float32Array; rerankOff: number; recLen: number; key: string }>();
	const versions = new Set<number>();
	let scanned = 0;
	let slabBytes = 0;
	let scanBytes = 0;
	let cacheHits = 0;
	let cacheMisses = 0;
	let tLastSlab = tFetch;
	let rerankReads = 0;
	let rerankBytes = 0;
	let rerankMultipart = 0;
	let specReadN = 0;
	let tombDropped = 0;
	let filterDropped = 0;
	let scannedCells = 0;
	// Out-parameter of thread.scan/scan_pq/scan_delta: [scanned, tombstoned entries skipped, filtered out]. Shared:
	// each scan is synchronous and read right after the call.
	const scanStats = new Uint32Array(3);
	const noTomb = new Uint8Array(0);
	/**
	 * Reads the rerank records `idxs` (ascending) of one slab: one multi-range request (rr=multi), else
	 * range reads of runs within `coalesce` bytes (or one span within `spanMax`, rr=span).
	 */
	const fetchRecords = async (cell: number, idxs: number[], low: boolean): Promise<Map<number, Uint8Array>> => {
		const si = slabInfo.get(cell)!;
		const recLen = si.recLen;
		const out = new Map<number, Uint8Array>();
		if (!idxs.length) return out;
		const runs: { a: number; b: number; members: number[] }[] = [];
		const spanBytes = (idxs.at(-1)! - idxs[0] + 1) * recLen;
		const limit = dbg.rr === "span" && spanBytes <= dbg.spanMax ? spanBytes : dbg.coalesce;
		for (const i of idxs) {
			const run = runs.at(-1);
			if (run && (i - run.a + 1) * recLen <= limit) {
				run.b = i;
				run.members.push(i);
			} else runs.push({ a: i, b: i, members: [i] });
		}
		const key = si.key;
		const place = (run: (typeof runs)[number], bytes: Uint8Array) => {
			for (const i of run.members) out.set(i, bytes.subarray((i - run.a) * recLen, (i - run.a + 1) * recLen));
		};
		const rangeOf = (run: (typeof runs)[number]) => ({ offset: si.rerankOff + run.a * recLen, length: (run.b - run.a + 1) * recLen });
		const readRun = async (run: (typeof runs)[number]) => {
			const { offset, length } = rangeOf(run);
			const bytes = await readWithRetry(
				lim,
				async () => {
					const hit = mode === "edge" ? await cacheMatchRange(key, offset, length, rs) : null;
					return hit?.bytes ?? new Uint8Array(await getBuffer(env.BUCKET, key, { offset, length }));
				},
				low,
			);
			rerankReads++;
			if (low) specReadN++;
			rerankBytes += length;
			place(run, bytes);
		};
		if (dbg.rr === "multi" && runs.length > 1 && mode === "edge") {
			const got = await readWithRetry(lim, () => cacheMatchMulti(key, runs.map(rangeOf), rs), low);
			if (got) {
				rerankReads++;
				if (low) specReadN++;
				if (got.multipart) rerankMultipart++;
				rerankBytes += got.wireBytes;
				runs.forEach((run, i) => place(run, got.out[i]));
				return out;
			}
		}
		await Promise.all(runs.map(readRun));
		return out;
	};
	let slabsDone = 0;
	let specFired = false;
	/** Speculative reads per cell (records by entry index). Reads for cells that end up without winners are not awaited. */
	const specReads = new Map<number, Promise<Map<number, Uint8Array> | null>>();
	let specRecords = 0;
	const speculate = () => {
		const top = [...cand.values()].sort((a, b) => b.score - a.score).slice(0, Math.ceil((T * dbg.specMargin) / 100));
		const byCell = new Map<number, number[]>();
		for (const c of top) {
			if (local.has(`${c.cell}:${c.idx}`)) continue;
			let l = byCell.get(c.cell);
			if (!l) byCell.set(c.cell, (l = []));
			l.push(c.idx);
		}
		for (const [cell, idxs] of byCell) {
			idxs.sort((a, b) => a - b);
			specRecords += idxs.length;
			// A failed speculative read only means phase 2 reads those records itself.
			specReads.set(cell, fetchRecords(cell, idxs, true).catch(() => null));
		}
	};
	/** Scans one cell; `batchEnd` is the index past the last cell of the batch it belongs to (the speculation cue). */
	const scanCell = async (cell: number, batchEnd: number): Promise<void> => {
			const src = await fetchScanSource(env, ctx, rv, cell, mode, rs, lim);
			tLastSlab = Date.now();
			if (src.outcome === "hit" || src.outcome === "mem") cacheHits++;
			else if (src.outcome === "miss") cacheMisses++;
			slabBytes += src.bytes;
			if (dbg.mode === "fetchonly") {
				scanBytes += src.prefix.byteLength;
				return;
			}
			const prefix = src.prefix;
			scanBytes += prefix.byteLength;
			// copy=0 (default): header and centroid are read in JS; only thread.scan copies the prefix into wasm.
			const info = dbg.copy ? slabHdrFromWasm(prefix) : parseSlabHeader(prefix);
			for (let r = 0; r < dbg.copyRepeat; r++) thread.slab_info(prefix);
			// Tombstones: ids deleted or replaced at a rev >= this slab's revision (every set for rev-0 slabs) are
			// skipped inside the scan, before top-T insertion, so they never take a candidate slot.
			const slabRev = rv.slabs ? slabRevOf(rv.slabs[cell]) : 0;
			const skip = tomb ? tombFor(tomb, slabRev) : noTomb;
			let hits: Uint8Array;
			if (info.version === 3) {
				if (!table) throw new HttpError(500, `slab ${cell} is v3 but ${gen} has no PQ codebook`);
				hits = thread.scan_pq(prefix, table, q, T, skip, filterWords, scanStats);
				for (let r = 1; r < dbg.scanRepeat; r++) hits = thread.scan_pq(prefix, table, q, T, skip, filterWords, scanStats);
			} else {
				hits = thread.scan(prefix, q, T, skip, filterWords, scanStats);
				for (let r = 1; r < dbg.scanRepeat; r++) hits = thread.scan(prefix, q, T, skip, filterWords, scanStats);
			}
			tombDropped += scanStats[1];
			filterDropped += scanStats[2];
			const centroid = doRerank && info.version >= 2 ? (dbg.copy ? thread.slab_centroid(prefix) : centroidFromBytes(prefix, info)) : null;
			versions.add(info.version);
			scanned += info.count;
			packed.push(hits);
			if (!doRerank || info.version < 2) return;
			const { rerankOff, recLen } = info;
			if (recLen !== 4 + dim) throw new HttpError(500, `slab ${cell}: rerank record length ${recLen} != ${4 + dim}`);
			slabInfo.set(cell, { centroid: centroid!, rerankOff, recLen, key: slabKey(gen, cell, slabRev) });
			for (const h of parseHits(hits)) {
				const prev = cand.get(h.id);
				if (!prev || h.score > prev.score) cand.set(h.id, { cell, idx: h.idx, score: h.score });
				if (src.full) {
					const o = rerankOff + h.idx * recLen;
					local.set(`${cell}:${h.idx}`, src.full.slice(o, o + recLen));
				}
			}
			slabsDone++;
			// Speculate once only a few slabs of the last batch are left: an unfiltered query has one batch; a filtered
			// one may widen, so its batch is known to be last only at the bound or once T candidates match.
			if (dbg.spec && !specFired && batchEnd - slabsDone <= dbg.specAt && (!filtered || batchEnd >= cells.length || cand.size >= T)) {
				specFired = true;
				speculate();
			}
	};
	// Phase 1 in batches: the first nprobe cells, then (filtered queries only) more cells in routing order until T
	// distinct candidates passed the filter or maxNprobe cells were scanned (the same rule as thread-sdk).
	const slabPhase = (async () => {
		let to = Math.min(nprobe, cells.length);
		await Promise.all(cells.slice(0, to).map((cell) => scanCell(cell, to)));
		scannedCells = to;
		const step = Math.max(4, Math.ceil(nprobe / 2));
		while (filtered && cand.size < T && scannedCells < cells.length) {
			to = Math.min(cells.length, scannedCells + step);
			await Promise.all(cells.slice(scannedCells, to).map((cell) => scanCell(cell, to)));
			scannedCells = to;
		}
	})();
	// ---- delta segments: every entry is scored with its int8 record (scores on the reranked scale, no phase 2).
	// A delta entry is skipped in the scan if its id is in a tombstone set newer than the rev that added the delta.
	const deltaHits: { id: number; score: number; seq: number }[] = [];
	let deltaBytes = 0;
	let deltaScanned = 0;
	let deltaDropped = 0;
	let deltaMem = 0;
	let tDeltaEnd = tFetch;
	const deltaPhase = Promise.all(
		deltas.map(async (d, i) => {
			const { bytes, outcome } = await deltaReads[i];
			if (outcome === "mem") deltaMem++;
			deltaBytes += bytes.byteLength;
			deltaScanned += d.count;
			// Sets with rev > addedRev (revs are integers) = tombFor(addedRev + 1); compactCells' deltaTomb is the same union.
			const hits = thread.scan_delta(bytes, q, deltaQc(d, bytes), T, tomb ? tombFor(tomb, d.addedRev + 1) : noTomb, filterWords, scanStats);
			deltaDropped += scanStats[1];
			filterDropped += scanStats[2];
			for (const h of parseHits(hits)) deltaHits.push({ id: h.id, score: h.score, seq: d.seq });
			tDeltaEnd = Date.now();
		}),
	);
	await Promise.all([slabPhase, deltaPhase]);
	const all = new Uint8Array(packed.reduce((a, b) => a + b.byteLength, 0));
	let o = 0;
	for (const b of packed) {
		all.set(b, o);
		o += b.byteLength;
	}
	packed.length = 0;
	const merged = all.byteLength === 0 ? [] : parseHits(thread.merge(all, T));
	const tPhase1 = Date.now();

	// ---- phase 2: re-score the global top-T with the int8 tier.
	let slabResults: { id: number; score: number }[];
	let rerankLocal = 0;
	let rerankSlabs = 0;
	let specUsed = 0;
	const reranked = doRerank && (versions.has(2) || versions.has(3));
	if (reranked) {
		const byCell = new Map<number, { id: number; idx: number }[]>();
		for (const h of merged) {
			const c = cand.get(h.id);
			if (!c) throw new HttpError(500, `merge returned unknown id ${h.id}`);
			let list = byCell.get(c.cell);
			if (!list) byCell.set(c.cell, (list = []));
			list.push({ id: h.id, idx: c.idx });
		}
		rerankSlabs = byCell.size;
		const rescored: { id: number; score: number }[] = [];
		await Promise.all(
			[...byCell].map(async ([cell, ws]) => {
				const si = slabInfo.get(cell)!;
				const recLen = si.recLen;
				ws.sort((a, b) => a.idx - b.idx);
				const records = new Uint8Array(ws.length * recLen);
				const spec = (await specReads.get(cell)) ?? null;
				const remote: number[] = [];
				ws.forEach((w, pos) => {
					const l = local.get(`${cell}:${w.idx}`);
					if (l) {
						records.set(l, pos * recLen);
						rerankLocal++;
						return;
					}
					const sp = spec?.get(w.idx);
					if (sp) {
						records.set(sp, pos * recLen);
						specUsed++;
						return;
					}
					remote.push(pos);
				});
				const got = await fetchRecords(
					cell,
					remote.map((pos) => ws[pos].idx),
					false,
				);
				for (const pos of remote) records.set(got.get(ws[pos].idx)!, pos * recLen);
				const scores = thread.rerank(q, si.centroid, records, ws.length);
				ws.forEach((w, i) => rescored.push({ id: w.id, score: scores[i] }));
			}),
		);
		slabResults = rescored;
	} else {
		slabResults = merged.map((h) => ({ id: h.id, score: h.score }));
	}
	// ---- final top k over slab and delta hits, one copy per id (the best score).
	const best = new Map<number, { id: number; score: number; delta: boolean }>();
	for (const r of slabResults) {
		const prev = best.get(r.id);
		if (!prev || r.score > prev.score) best.set(r.id, { id: r.id, score: r.score, delta: false });
	}
	for (const h of deltaHits) {
		const prev = best.get(h.id);
		if (!prev || h.score > prev.score) best.set(h.id, { id: h.id, score: h.score, delta: true });
	}
	const results = [...best.values()].sort((a, b) => b.score - a.score).slice(0, k);
	const tEnd = Date.now();
	return json({
		gen,
		rev: rv.rev,
		ids: results.map((r) => r.id),
		scores: results.map((r) => r.score),
		cells: cells.slice(0, scannedCells),
		stats: {
			scanned,
			slabBytes,
			scanBytes,
			k,
			nprobe,
			probeCoarse,
			routing: useBeam ? "beam" : flat ? "flat" : "tree",
			beam,
			treeGroups: tree?.groups ?? 0,
			// Cells whose q·mean was computed: all of them on the flat path, else only the delta segments' cells.
			dotsComputed,
			format,
			tiers: manifest.tiers ?? null,
			pqM: base.pq ? base.pq.m() : 0,
			profile: manifest.profile ?? null,
			// The values this request used and where each came from (query string, manifest defaults, legacy constants).
			params: {
				nprobe,
				rerank: rerankParam,
				pc: probeCoarse,
				route: routeMode,
				beam,
				source: {
					nprobe: source(p.nprobe),
					rerank: source(p.rerank),
					pc: source(p.probeCoarse),
					route: p.route !== null ? "query" : manifest.defaults?.route ? "manifest" : "legacy",
					beam: beam === null ? null : p.beam !== null ? "query" : defaults.beam !== undefined ? "manifest" : "tree",
				},
			},
			rerank: reranked ? T : 0,
			rerankCandidates: reranked ? merged.length : 0,
			rerankReads,
			rerankBytes,
			rerankLocal,
			rerankSlabs,
			rerankMultipart,
			specRecords,
			specUsed,
			specReads: specReadN,
			// Update path: delta segments scanned, their entries and bytes, delta hits, tombstoned delta entries skipped
			// in the scan (deltaDropped), hits in the final top k; tombstone ids in this revision and tombstoned slab
			// entries skipped in the scan (tombDropped). Skipped = would have entered a top-T heap when reached.
			deltas: deltas.length,
			deltaBytes,
			deltaScanned,
			deltaHits: deltaHits.length,
			deltaDropped,
			deltaTop: results.filter((r) => r.delta).length,
			deltaMem,
			tombstones: tomb?.total ?? 0,
			tombSets: tomb?.sets.length ?? 0,
			tombDropped,
			// Attribute filter: entries it dropped (among those that would have entered a top-T), distinct scan-tier
			// candidates, and the widening bound (= nprobe without a filter).
			filtered,
			filterDropped,
			candidates: cand.size,
			maxNprobe,
			limiterRetries: lim.retries,
			cache: mode,
			cacheHits,
			cacheMisses,
			range: rs,
			indexCache: { ...base.loadCache, ...rv.loadCache },
			attempt,
			wasmResets,
			wasmMemMB: wasmRaw ? Math.round(wasmRaw.memory.buffer.byteLength / 1048576) : null,
			debug: dbg,
			colo: (request.cf as { colo?: string } | undefined)?.colo,
		},
		// Date.now() only advances across I/O in Workers, so CPU-only phases read as ~0.
		// fetch = until the last scan prefix arrived; phase1 = route end -> merged top-T (slabs and deltas);
		// phase2 = rerank reads + rescoring; delta = until the last delta segment was scanned.
		timing: {
			total: tEnd - tStart,
			gen: tGen - tStart,
			load: tRoute - tStart,
			fetch: tLastSlab - tFetch,
			delta: tDeltaEnd - tFetch,
			phase1: tPhase1 - tFetch,
			phase2: tEnd - tPhase1,
			scanAndMerge: tPhase1 - tLastSlab,
			indexLoadMs: base.loadMs,
			revLoadMs: rv.loadMs,
		},
	});
}

/**
 * Debug: how caches.default answers a Range request for a cached object.
 * `offset`/`length` = one range; `ranges=o:l,o:l,...` = one multi-range request (each part is
 * checked against an R2 range read).
 */
async function handleRangeProbe(env: Env, ctx: ExecutionContext, url: URL): Promise<Response> {
	const key = url.searchParams.get("key") ?? "";
	if (!key.startsWith("idx/")) throw new HttpError(400, "key must be under idx/");
	const offset = intParam(url, "offset", 0);
	const length = intParam(url, "length", 64, 1);
	const prime = url.searchParams.get("prime") === "1";
	let primed = false;
	if (prime && !(await caches.default.match(cacheRequest(key)))) {
		const buf = await getBuffer(env.BUCKET, key);
		const resp = new Response(buf, {
			headers: { "content-type": "application/octet-stream", "content-length": String(buf.byteLength), "cache-control": CACHE_CONTROL, "accept-ranges": "bytes" },
		});
		await caches.default.put(cacheRequest(key), resp);
		primed = true;
	}
	const rangesParam = url.searchParams.get("ranges");
	if (rangesParam) {
		const ranges = rangesParam.split(",").map((s) => {
			const [o, l] = s.split(":").map(Number);
			if (!Number.isInteger(o) || !Number.isInteger(l) || o < 0 || l < 1) throw new HttpError(400, "ranges must be offset:length,...");
			return { offset: o, length: l };
		});
		const hdr = "bytes=" + ranges.map((r) => `${r.offset}-${r.offset + r.length - 1}`).join(", ");
		const r = await caches.default.match(new Request(cacheRequest(key).url, { method: "GET", headers: { range: hdr } }));
		if (!r) return json({ key, cached: false, primed, rangeHeader: hdr });
		const body = new Uint8Array(await r.arrayBuffer());
		const head = { status: r.status, contentType: r.headers.get("content-type"), contentRange: r.headers.get("content-range"), contentLength: r.headers.get("content-length"), bodyBytes: body.byteLength };
		let parts: { offset: number; length: number; matchesR2: boolean }[] | null = null;
		let parseError: string | null = null;
		try {
			const got = await cacheMatchMulti(key, ranges, { r206: 0, r200: 0, miss: 0 });
			if (got) {
				parts = await Promise.all(
					ranges.map(async (rg, i) => {
						const ref = new Uint8Array(await getBuffer(env.BUCKET, key, rg));
						const b = got.out[i];
						return { ...rg, matchesR2: ref.byteLength === b.byteLength && ref.every((v, j) => v === b[j]) };
					}),
				);
			}
		} catch (e) {
			parseError = e instanceof Error ? e.message : String(e);
		}
		return json({ key, cached: true, primed, rangeHeader: hdr, ...head, bodyHead: latin1.decode(body.subarray(0, 300)), parts, parseError });
	}
	const r = await caches.default.match(cacheRequest(key, { offset, length }));
	if (!r) return json({ key, cached: false, primed });
	const body = new Uint8Array(await r.arrayBuffer());
	let matchesR2: boolean | null = null;
	if (r.status === 206) {
		const ref = new Uint8Array(await getBuffer(env.BUCKET, key, { offset, length }));
		matchesR2 = ref.byteLength === body.byteLength && ref.every((v, i) => v === body[i]);
	}
	return json({
		key,
		cached: true,
		primed,
		status: r.status,
		contentRange: r.headers.get("content-range"),
		contentLength: r.headers.get("content-length"),
		bodyBytes: body.byteLength,
		matchesR2,
	});
}

// ---------- write path: revisions, upsert, delete, compaction, gc ----------
//
// Every change is a new manifest revision `idx/{gen}/manifest.{rev+1}.json` written with a create-only conditional
// PUT (R2 `onlyIf: {etagDoesNotMatch: "*"}`): exactly one writer wins each rev number, so the manifest PUT is the
// serialization point. Objects a revision references (delta segments, tombstone sets, slabs tables, slabs) are
// written before it; a writer that loses the race rebuilds its manifest on the new head and retries (its
// already-written delta segment is reused). idx/current.json is then advanced with a compare-and-swap
// (`onlyIf: {etagMatches}`) that only moves it forward; a crashed writer's manifest is picked up by the next
// writer's CAS. Losers leave unreferenced objects behind; `gc` removes them.

/** Body for a PUT: a string or the exact bytes of a view. */
function putBody(data: ArrayBuffer | ArrayBufferView | string): ArrayBuffer | ArrayBufferView | string {
	return typeof data === "string" || data instanceof ArrayBuffer ? data : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

/** PUT only if the key does not exist yet. Returns false if it does. */
async function putIfAbsent(env: Env, key: string, data: ArrayBuffer | ArrayBufferView | string): Promise<boolean> {
	const r = await env.BUCKET.put(key, putBody(data), { onlyIf: { etagDoesNotMatch: "*" } });
	return r !== null;
}

/**
 * Writes an immutable object at `key`, or, if that key exists (a concurrent or crashed writer used the same
 * rev/seq), at a unique sibling `{stem}.{nonce}{ext}`. Returns the key written; manifests store it verbatim.
 */
async function putUnique(env: Env, key: string, data: ArrayBuffer | ArrayBufferView | string): Promise<string> {
	if (await putIfAbsent(env, key, data)) return key;
	const dot = key.lastIndexOf(".");
	for (let i = 0; i < 4; i++) {
		const alt = `${key.slice(0, dot)}.${crypto.randomUUID().slice(0, 8)}${key.slice(dot)}`;
		if (await putIfAbsent(env, alt, data)) return alt;
	}
	throw new HttpError(500, `could not write a unique key for ${key}`);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Moves idx/current.json forward to (gen, rev) with a CAS loop; leaves it alone if it points at another generation or a
 * newer rev. With `manifest` (that revision's), the pointer carries a copy of it when small (the SDKs' `currentBody`),
 * so an SDK open skips the manifest GET.
 */
async function advanceCurrent(env: Env, gen: string, rev: number, setIfOtherGen = false, hint: { cur: Current; etag: string } | null = null, manifest?: Manifest): Promise<{ updated: boolean; conflicts: number; current: Current | null }> {
	let conflicts = 0;
	const pointer = currentBody(gen, rev, manifest as unknown as SdkManifest | undefined);
	for (let attempt = 0; attempt < 10; attempt++) {
		// The first CAS uses the copy read when the commit started (usually still current), saving a round trip.
		const r = attempt === 0 && hint ? hint : await readCurrent(env);
		if (!r) {
			if (await putIfAbsent(env, CURRENT_KEY, pointer)) return { updated: true, conflicts, current: { gen, rev } };
			conflicts++;
			continue;
		}
		if ((r.cur.gen !== gen && !setIfOtherGen) || (r.cur.gen === gen && r.cur.rev >= rev)) return { updated: false, conflicts, current: r.cur };
		const ok = await env.BUCKET.put(CURRENT_KEY, pointer, { onlyIf: { etagMatches: r.etag } });
		if (ok) {
			currentPtr = { cur: { gen, rev }, at: Date.now() };
			return { updated: true, conflicts, current: { gen, rev } };
		}
		conflicts++;
		await sleep(20 + Math.random() * 50);
	}
	throw new HttpError(409, `idx/current.json: gave up after ${conflicts} CAS conflicts`);
}

const COMMIT_ATTEMPTS = 12;

interface CommitResult {
	gen: string;
	rev: number;
	parentRev: number;
	manifest: Manifest;
	/** Lost manifest races (create-only PUT found the rev taken) and current.json CAS retries. */
	conflicts: number;
	casConflicts: number;
	currentUpdated: boolean;
	manifestBytes: number;
}

/**
 * Commits a new revision of `gen` (null = the current generation): `make` receives the head (rev, manifest) and
 * returns the next manifest (it may write the objects that manifest references, keyed by rev + 1, via putUnique).
 * The head is taken from idx/current.json when it points at `gen` (one GET instead of a LIST); a lost race (or a
 * crashed writer's manifest beyond current.json) makes the next attempt list the manifest revisions.
 */
async function commitRev(env: Env, genArg: string | null, make: (head: { gen: string; rev: number; manifest: Manifest; attempt: number }) => Promise<Manifest>): Promise<CommitResult> {
	let conflicts = 0;
	let list = false;
	for (let attempt = 1; attempt <= COMMIT_ATTEMPTS; attempt++) {
		const cur = await readCurrent(env);
		const gen = genArg ?? cur?.cur.gen;
		if (!gen) throw new HttpError(404, "no published index");
		const head = !list && cur && cur.cur.gen === gen ? cur.cur.rev : await latestRev(env, gen);
		const base = await readManifestR2(env, gen, head);
		const next = await make({ gen, rev: head, manifest: base, attempt });
		const rev = head + 1;
		const manifest: Manifest = { ...next, gen, rev, parentRev: head, updatedAt: new Date().toISOString() };
		const body = JSON.stringify(manifest);
		if (await putIfAbsent(env, manifestKey(gen, rev), body)) {
			genHeads.set(gen, { rev, at: Date.now() });
			const c = await advanceCurrent(env, gen, rev, false, cur, manifest);
			return { gen, rev, parentRev: head, manifest, conflicts, casConflicts: c.conflicts, currentUpdated: c.updated, manifestBytes: body.length };
		}
		conflicts++;
		list = true;
		await sleep(10 + Math.random() * 40 * attempt);
	}
	throw new HttpError(409, `${genArg ?? "current generation"}: lost the manifest race ${conflicts} times`);
}

/** The manifest fields a new revision inherits (build notes stay in rev 0). */
function inherit(m: Manifest): Manifest {
	const { notes: _notes, change: _change, rev: _rev, parentRev: _parent, updatedAt: _u, ...rest } = m;
	return { ...rest, files: manifestFiles(m), deltas: m.deltas ?? [], tombstones: m.tombstones ?? [], nextSeq: m.nextSeq ?? 1 };
}

function checkDefaults(d: QueryDefaults, tree?: TreeRef | null) {
	for (const name of ["nprobe", "rerank", "pc"] as const) {
		const v = d[name];
		if (!Number.isInteger(v) || v < (name === "nprobe" ? 1 : 0)) throw new HttpError(400, `defaults.${name} must be a non-negative integer${name === "nprobe" ? " >= 1" : ""}`);
	}
	if (d.route !== undefined && !ROUTE_MODES.includes(d.route)) throw new HttpError(400, `defaults.route must be ${ROUTE_MODES.join("|")}`);
	if (d.route === "beam" && !tree) throw new HttpError(400, "defaults.route beam needs a centroid tree (pass tree)");
	if (d.beam !== undefined && (!Number.isInteger(d.beam) || d.beam < 1 || (tree?.groups !== undefined && d.beam > tree.groups))) throw new HttpError(400, `defaults.beam must be an integer in 1..${tree?.groups ?? "tree groups"}`);
}

interface PublishRevReq {
	gen: string;
	/** Query defaults; omitted fields keep their current value. */
	defaults?: Partial<QueryDefaults>;
	profile?: string;
	/**
	 * Build a centroid tree over the revision's fine means and reference it from the new revision
	 * (idx/{gen}/tree.{rev}.bin); null removes the tree (and a beam route default).
	 */
	tree?: TreeReq | null;
	/**
	 * Write compact routing tables (fine.f16 / fine.i8) from the revision's fine means and name them in the new
	 * revision (manifest.routing); null removes them. Absent: kept (a new tree does not change them).
	 */
	routing?: RoutingReq | null;
	/** Base packs of the new revision (base.{rev}.threadpack, base.cold.{rev}.threadpack). Default `{}`; false = none. */
	basePack?: BasePackReq | false;
	/** Point idx/current.json at this generation even if another one is current (default false). */
	setCurrent?: boolean;
	/**
	 * Compaction policy: fields merged over the head's policy (or the profile's defaults when the head has none);
	 * "profile" = the profile's defaults; null removes it (no automatic compaction, no splits).
	 */
	compaction?: Partial<CompactionPolicy> | "profile" | null;
}

/** The compaction policy a publish-rev asks for, given the head's. */
function resolveCompaction(req: PublishRevReq["compaction"], head: Manifest, profile: string | undefined): CompactionPolicy | undefined {
	if (req === undefined) return head.compaction;
	if (req === null) return undefined;
	const base = COMPACTION_PROFILES[profile ?? head.profile ?? "edge"] ?? COMPACTION_PROFILES.edge;
	if (req === "profile") return checkCompactionPolicy({ ...base });
	return checkCompactionPolicy({ ...(head.compaction ?? base), ...req });
}

/** Manifest rev + 1 with changed query defaults and/or a new centroid tree (no slab, delta or tombstone changes). */
async function stagePublishRev(env: Env, body: PublishRevReq) {
	const { gen } = body;
	if (!gen) throw new HttpError(400, "gen required");
	// The tree is built once (outside the commit loop) over the head's fine means; a retry only re-checks the files.
	let built: { fine: string; bytes: Uint8Array; ref: Omit<TreeRef, "key"> } | null = null;
	let treeWritten: TreeRef | null = null;
	let routingWritten: { for: string; routing: RoutingTables } | null = null;
	const c = await commitRev(env, gen, async ({ rev, manifest }) => {
		const files = manifestFiles(manifest);
		let tree: TreeRef | undefined = manifest.tree;
		if (body.tree === null) tree = undefined;
		else if (body.tree) {
			if (!built || built.fine !== files.fine) {
				const notes = (rev === 0 ? manifest.notes : undefined) as { cfg?: { sampleGroups?: number } } | undefined;
				built = { fine: files.fine, ...(await buildTree(env, gen, manifest, files.coarse, files.fine, body.tree, notes?.cfg?.sampleGroups)) };
				treeWritten = null;
			}
			treeWritten ??= { key: await putUnique(env, treeKey(gen, rev + 1), built.bytes), ...built.ref };
			tree = treeWritten;
		}
		const prev = manifest.defaults ?? LEGACY_DEFAULTS;
		const defaults: QueryDefaults = { ...prev, ...(body.defaults ?? {}) };
		if (!tree && defaults.route === "beam" && body.defaults?.route === undefined) delete defaults.route;
		if (!tree && body.defaults?.beam === undefined) delete defaults.beam;
		checkDefaults(defaults, tree);
		// Routing objects: written on request; otherwise a head's grouped table that follows another tree is rebuilt for
		// the new one (or dropped without a tree), and tables made from other means are dropped.
		let routing: RoutingTables | undefined = manifest.routing?.fine === files.fine ? manifest.routing : undefined;
		const treeArg = async () => (tree ? { key: tree.key, bytes: built && treeWritten?.key === tree.key ? built.bytes : new Uint8Array(await getBuffer(env.BUCKET, tree.key)) } : null);
		const memo = async (req: RoutingReq): Promise<RoutingTables> => {
			const id = `${files.fine}|${tree?.key ?? "-"}|${JSON.stringify(req)}`;
			if (!routingWritten || routingWritten.for !== id) routingWritten = { for: id, routing: await writeRouting(env, gen, rev + 1, files.fine, await getF32(env.BUCKET, files.fine), manifest.dim, req, true, await treeArg()) };
			return routingWritten.routing;
		};
		if (body.routing === null) routing = undefined;
		else if (body.routing) routing = await memo(body.routing);
		else if (routing?.grouped && routing.grouped.tree !== tree?.key) {
			const { grouped, ...rest } = routing;
			const rebuilt = tree ? (await memo({ tables: [], grouped: grouped.elem })).grouped : undefined;
			routing = rebuilt ? { ...rest, grouped: rebuilt } : rest;
		}
		const next: Manifest = { ...inherit(manifest), defaults, ...(body.profile ? { profile: body.profile } : {}) };
		if (tree) next.tree = tree;
		else delete next.tree;
		if (routing) next.routing = routing;
		else delete next.routing;
		const { base: _b, baseCold: _bc, ...plainFiles } = next.files!;
		next.files = { ...plainFiles, ...basePackFiles(gen, rev + 1, { ...next, files: plainFiles }, body.basePack) };
		const compaction = resolveCompaction(body.compaction, manifest, body.profile);
		if (compaction) next.compaction = compaction;
		else delete next.compaction;
		const treeChange = body.tree === undefined ? {} : { tree: tree ? { key: tree.key, groups: tree.groups, beam: tree.beam } : null };
		const routingChange = body.routing === undefined ? {} : { routing: routing ? { f16: routing.f16?.key ?? null, i8: routing.i8?.key ?? null, grouped: routing.grouped?.key ?? null } : null };
		const compactionChange = body.compaction === undefined ? {} : { compaction: { from: manifest.compaction ?? null, to: compaction ?? null } };
		return { ...next, change: { op: "publish-rev", defaults: { from: prev, to: defaults }, ...treeChange, ...routingChange, ...compactionChange } };
	});
	// The packs hold the committed manifest, so they are written after the commit (a reader that sees the revision
	// first finds no pack and reads the separate files).
	const basePacks = c.manifest.files?.base ? await writeBasePacks(env, gen, c.rev, body.basePack) : [];
	const cur = body.setCurrent ? await advanceCurrent(env, gen, c.rev, true, null, c.manifest) : null;
	return { gen, rev: c.rev, parentRev: c.parentRev, defaults: c.manifest.defaults, tree: c.manifest.tree ?? null, routing: c.manifest.routing ?? null, basePacks, compaction: c.manifest.compaction ?? null, conflicts: c.conflicts, currentUpdated: c.currentUpdated || !!cur?.updated };
}

/** Sorted unique ids as a tombstone buffer. */
const tombBytes = (ids: Uint32Array) => idUnion([ids]);

const MAX_UPSERT = 4096;

interface UpsertInput {
	gen: string | null;
	/** Explicit ids, or null with firstId (ids firstId, firstId + 1, ...). */
	ids: Uint32Array | null;
	firstId: number | null;
	/** n * dim, unnormalized. */
	vectors: Float32Array;
	attrs: Uint32Array | null;
	payloads: Uint8Array[] | null;
	/** Write a tombstone set for the ids (replace older copies). Default true. */
	replace: boolean;
}

function b64ToBytes(s: string): Uint8Array {
	const bin = atob(s);
	const out = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
	return out;
}

function parseIdList(raw: string): Uint32Array {
	return u32Array(raw.split(",").filter((x) => x.trim() !== "").map(Number), "ids");
}

function u32Array(a: unknown, name: string): Uint32Array {
	if (!Array.isArray(a)) throw new HttpError(400, `${name} must be an array`);
	for (const v of a) if (!Number.isInteger(v) || v < 0 || v > 0xffffffff) throw new HttpError(400, `${name} must hold u32 integers`);
	return Uint32Array.from(a as number[]);
}

/**
 * Upsert body: JSON {gen?, ids, vectors (flat number[] or base64 f32 LE), attrs?, payloads? (strings, or
 * {b64}), replace?}, or a binary f32 LE body (`content-type: application/octet-stream`) with the ids in the
 * `x-ids` header or `?ids=` (comma-separated) or as `?firstId=` (ids firstId, firstId+1, ...), and `?gen=`,
 * `?replace=0`.
 */
async function parseUpsert(request: Request, url: URL): Promise<UpsertInput> {
	const ct = request.headers.get("content-type") ?? "";
	let gen = url.searchParams.get("gen");
	let replace = url.searchParams.get("replace") !== "0";
	let ids: Uint32Array | null = null;
	let firstId: number | null = null;
	let vectors: Float32Array;
	let attrs: Uint32Array | null = null;
	let payloads: Uint8Array[] | null = null;
	if (ct.includes("octet-stream")) {
		const buf = await request.arrayBuffer();
		if (buf.byteLength % 4 !== 0) throw new HttpError(400, "binary body must be f32 LE");
		vectors = new Float32Array(buf);
		const hdr = request.headers.get("x-ids") ?? url.searchParams.get("ids");
		if (hdr) ids = parseIdList(hdr);
		firstId = optIntParam(url, "firstId", 0);
		if (!ids && firstId === null) throw new HttpError(400, "binary upsert needs ids (x-ids header, ?ids=) or ?firstId=");
	} else {
		const body = await readJson<{ gen?: string; ids: number[]; vectors: number[] | string; attrs?: number[]; payloads?: (string | { b64: string })[]; replace?: boolean }>(request);
		gen = body.gen ?? gen;
		if (body.replace === false) replace = false;
		ids = u32Array(body.ids, "ids");
		if (typeof body.vectors === "string") {
			const b = b64ToBytes(body.vectors);
			if (b.byteLength % 4 !== 0) throw new HttpError(400, "base64 vectors must be f32 LE");
			vectors = new Float32Array(b.buffer, 0, b.byteLength / 4);
		} else if (Array.isArray(body.vectors)) vectors = Float32Array.from(body.vectors);
		else throw new HttpError(400, "vectors must be a flat number[] or base64 f32");
		if (body.attrs) attrs = u32Array(body.attrs, "attrs");
		if (body.payloads) {
			const enc = new TextEncoder();
			payloads = body.payloads.map((p) => (typeof p === "string" ? enc.encode(p) : p && typeof p === "object" && typeof p.b64 === "string" ? b64ToBytes(p.b64) : enc.encode(JSON.stringify(p))));
		}
	}
	return { gen, ids, firstId, vectors, attrs, payloads, replace };
}

/** Validates the batch against the manifest and returns the final ids and the vectors, unit-normalized in place. */
function prepareBatch(input: UpsertInput, dim: number): { ids: Uint32Array; vectors: Float32Array } {
	const { vectors } = input;
	if (vectors.length % dim !== 0) throw new HttpError(400, `vectors has ${vectors.length} floats, not a multiple of dim ${dim}`);
	const n = vectors.length / dim;
	const first = input.firstId;
	const ids = input.ids ?? Uint32Array.from({ length: n }, (_, i) => first! + i);
	if (first !== null && first + n - 1 > 0xffffffff) throw new HttpError(400, "firstId + n exceeds u32");
	if (n === 0) throw new HttpError(400, "empty batch");
	if (n > MAX_UPSERT) throw new HttpError(413, `at most ${MAX_UPSERT} vectors per upsert`);
	if (ids.length !== n) throw new HttpError(400, `${ids.length} ids for ${n} vectors`);
	if (new Set(ids).size !== n) throw new HttpError(400, "duplicate ids in the batch");
	if (input.attrs && input.attrs.length !== n) throw new HttpError(400, `${input.attrs.length} attrs for ${n} vectors`);
	if (input.payloads && input.payloads.length !== n) throw new HttpError(400, `${input.payloads.length} payloads for ${n} vectors`);
	// In place: the buffer is this request's own copy, and a 2000-vector batch is 12 MB.
	for (let i = 0; i < n; i++) {
		let ss = 0;
		for (let j = i * dim; j < (i + 1) * dim; j++) ss += vectors[j] * vectors[j];
		if (!(ss > 0) || !Number.isFinite(ss)) throw new HttpError(400, `vector ${i} (id ${ids[i]}) has zero or non-finite norm`);
		const inv = 1 / Math.sqrt(ss);
		for (let j = i * dim; j < (i + 1) * dim; j++) vectors[j] *= inv;
	}
	return { ids, vectors };
}

/** The first free delta seq >= `from` (HEAD probes, so a crashed writer's segment is skipped). */
async function allocSeq(env: Env, gen: string, from: number): Promise<number> {
	let seq = Math.max(1, from);
	for (let i = 0; i < 64 && (await env.BUCKET.head(deltaKey(gen, seq))); i++) seq++;
	return seq;
}

interface EncodedDelta {
	seq: number;
	key: string;
	count: number;
	bytes: number;
	cells: Uint32Array;
	nCells: number;
	filesKey: string;
	/** Tree groups / beam the primary cells were picked with (null = exhaustive search). */
	tree: { groups: number; beam: number } | null;
	encodeMs: number;
	loadMs: number;
	putMs: number;
}

/** Bases this isolate has loaded (0 = a write-only isolate). */
let basesLoaded = 0;
/** Fine means kept by write-only isolates (no base ever loaded), saving a 25 MB cache read per upsert. */
let writerMeans: { key: string; buf: ArrayBuffer } | null = null;
/** The last centroid tree a write used (~1 MB; kept by every isolate, trees are small). */
let writerTree: { key: string; buf: ArrayBuffer } | null = null;

/** What a delta segment's encoding depends on: a retry re-encodes only if one of these changed under it. */
const deltaFilesKey = (files: IndexFiles, m: Manifest) => `${files.fine}|${files.pq}|${m.tree?.key ?? "-"}|${m.tree?.beam ?? 0}`;

/** Vectors passed to DeltaJob.add per call (bounds the batch copy in wasm memory). */
const DELTA_ADD_CHUNK = 256;

/**
 * Encodes one delta segment (primary cell + residual codes against the generation's fine means; PQ codes with
 * the generation's codebook) and writes it. Writes never load the Searcher. DeltaJob holds a copy of the means
 * (~25 MB at 4096 x 1536, ~50 MB while it is passed in) and wasm memory never shrinks, so in an isolate that
 * also serves queries an idle loaded base is dropped first (its memory is reused; the next query reloads it);
 * a base that queries are using lends its PqCodebook instead. Writes are best sent to a Worker deployment that
 * does not serve queries.
 */
async function encodeDelta(env: Env, ctx: ExecutionContext, m: Manifest, files: IndexFiles, ids: Uint32Array, vectors: Float32Array, input: UpsertInput): Promise<EncodedDelta> {
	await ensureWasm();
	const epoch = wasmEpoch;
	const tLoad = Date.now();
	const gen = m.gen;
	const format = m.format ?? 1;
	// Borrow the codebook of a base that queries are using; drop an idle one (see above).
	let slot = baseCache.get(baseKeyOf(gen, files)) ?? null;
	if (slot && slot.users === 0) {
		dropBaseSlot(slot);
		slot = null;
	}
	for (const other of [...baseCache.values()]) if (other.users === 0) dropBaseSlot(other);
	if (format === 3 && !files.pq) throw new HttpError(500, `format 3 generation ${gen} has no PQ codebook`);
	// Seq probe, codebook and means are fetched in parallel.
	const loads = await Promise.all([
		allocSeq(env, gen, m.nextSeq ?? 1),
		slot ? slot.promise.catch(() => null) : Promise.resolve(null),
		format === 3 && !slot ? getCached(env, ctx, files.pq!) : Promise.resolve(null),
		writerMeans?.key === files.fine ? Promise.resolve({ buf: writerMeans.buf }) : getCached(env, ctx, files.fine),
		!m.tree ? Promise.resolve(null) : writerTree?.key === m.tree.key ? Promise.resolve({ buf: writerTree.buf }) : getCached(env, ctx, m.tree.key),
	]);
	const [seq, loaded] = loads;
	if (basesLoaded === 0) writerMeans = { key: files.fine, buf: loads[3].buf };
	const treeBuf = loads[4];
	if (m.tree && treeBuf) writerTree = { key: m.tree.key, buf: treeBuf.buf };
	let pqBuf: { buf: ArrayBuffer } | null = loads[2];
	let meansBuf: { buf: ArrayBuffer } | null = loads[3];
	(loads as unknown[]).length = 0;
	let pq: thread.PqCodebook | null = loaded && loaded.epoch === wasmEpoch && !slot!.evicted ? loaded.pq : null;
	let ownPq: thread.PqCodebook | null = null;
	if (format === 3 && !pq) pq = ownPq = parseCodebook(pqBuf?.buf ?? (await getCached(env, ctx, files.pq!)).buf, m, files.pq!);
	pqBuf = null;
	if (slot) slot.users++;
	try {
		const job = (() => {
			const means = new Float32Array(meansBuf!.buf);
			meansBuf = null; // the JS copy is collectable once DeltaJob holds its own
			if (means.length !== m.k1 * m.k2 * m.dim) throw new HttpError(500, `${files.fine} has ${means.length / m.dim} means, expected ${m.k1 * m.k2}`);
			return new thread.DeltaJob(m.dim, m.k1 * m.k2, means, seq);
		})();
		const loadMs = Date.now() - tLoad;
		let finished = false;
		let bytes: Uint8Array;
		let cells: Uint32Array;
		const tEnc = Date.now();
		try {
			if (format === 3) job.set_pq_codebook(pq!);
			// Pruned primary-cell search: group means first, then only the cells of the best `beam` groups.
			if (m.tree && treeBuf) job.set_tree(new Uint8Array(treeBuf.buf), m.tree.beam);
			const dim = m.dim;
			cells = new Uint32Array(ids.length);
			for (let a = 0; a < ids.length; a += DELTA_ADD_CHUNK) {
				const b = Math.min(ids.length, a + DELTA_ADD_CHUNK);
				const v = vectors.subarray(a * dim, b * dim);
				const attrs = input.attrs ? input.attrs.subarray(a, b) : new Uint32Array(0);
				let c: Uint32Array;
				if (input.payloads) {
					const ps = input.payloads.slice(a, b);
					const lens = Uint32Array.from(ps, (p) => p.byteLength);
					const cat = new Uint8Array(lens.reduce((x, y) => x + y, 0));
					let o = 0;
					for (const p of ps) {
						cat.set(p, o);
						o += p.byteLength;
					}
					c = job.add_with_payloads(v, ids.subarray(a, b), attrs, cat, lens);
				} else c = job.add(v, ids.subarray(a, b), attrs);
				if (c.length !== b - a) throw new HttpError(500, `DeltaJob added ${c.length} of ${b - a} vectors`);
				cells.set(c, a);
			}
			if (cells.length !== ids.length) throw new HttpError(500, `DeltaJob added ${cells.length} of ${ids.length} vectors`);
			finished = true;
			bytes = job.finish();
		} finally {
			if (!finished) safeFree(job, epoch);
		}
		const encodeMs = Date.now() - tEnc;
		const nCells = thread.delta_cells(bytes).length / 2;
		const tPut = Date.now();
		const key = await putUnique(env, deltaKey(gen, seq), bytes);
		return { seq, key, count: ids.length, bytes: bytes.byteLength, cells, nCells, filesKey: deltaFilesKey(files, m), tree: m.tree ? { groups: m.tree.groups, beam: m.tree.beam } : null, encodeMs, loadMs, putMs: Date.now() - tPut };
	} finally {
		if (ownPq) safeFree(ownPq, epoch);
		if (slot) {
			slot.users--;
			if (slot.evicted && slot.users === 0) slot.promise.then(freeLoadedBase, () => {});
		}
	}
}

async function targetGen(env: Env, gen: string | null): Promise<string> {
	if (gen) return gen;
	const r = await readCurrent(env);
	if (!r) throw new HttpError(404, "no published index");
	return r.cur.gen;
}

/** POST /upsert: one delta segment + (by default) a tombstone set for its ids, in one new manifest revision. */
async function handleUpsert(request: Request, env: Env, ctx: ExecutionContext, url: URL): Promise<Response> {
	const t0 = Date.now();
	const input = await parseUpsert(request, url);
	return withWasmRecovery("upsert", async () => {
		// Held across commit attempts (closure state; a holder object keeps TS from narrowing it to null).
		const st: { delta: EncodedDelta | null; batch: { ids: Uint32Array; vectors: Float32Array } | null; tombKey: string | null; tombBytes: number } = { delta: null, batch: null, tombKey: null, tombBytes: 0 };
		const c = await commitRev(env, input.gen, async ({ gen, rev, manifest }) => {
			const files = manifestFiles(manifest);
			const batch = (st.batch ??= prepareBatch(input, manifest.dim));
			const next = inherit(manifest);
			const newRev = rev + 1;
			const tb = input.replace ? tombBytes(batch.ids) : null;
			// The tombstone set is written while the segment is encoded.
			const tombPut = tb ? putUnique(env, tombKey(gen, newRev), tb) : Promise.resolve(null);
			try {
				// The segment is reused across retries unless the means/codebook changed under us.
				if (!st.delta || st.delta.filesKey !== deltaFilesKey(files, manifest)) st.delta = await encodeDelta(env, ctx, manifest, files, batch.ids, batch.vectors, input);
			} finally {
				st.tombKey = await tombPut.catch(() => null);
			}
			const delta = st.delta;
			if (tb) {
				if (!st.tombKey) st.tombKey = await putUnique(env, tombKey(gen, newRev), tb);
				st.tombBytes = tb.byteLength;
				next.tombstones!.push({ rev: newRev, key: st.tombKey, count: tb.byteLength / 4 });
			}
			next.deltas!.push({ seq: delta.seq, key: delta.key, count: delta.count, bytes: delta.bytes, addedRev: newRev, cells: delta.nCells });
			next.nextSeq = Math.max(next.nextSeq ?? 1, delta.seq + 1);
			return { ...next, change: { op: "upsert", seq: delta.seq, count: delta.count, replace: input.replace } };
		});
		const d = st.delta!;
		const m = c.manifest;
		const compaction = maybeAutoCompact(env, ctx, m, "upsert");
		return json({
			gen: c.gen,
			rev: c.rev,
			seq: d.seq,
			count: d.count,
			cells: Array.from(d.cells),
			distinctCells: d.nCells,
			ms: Date.now() - t0,
			bytes: { delta: d.bytes, tombstones: st.tombBytes, manifest: c.manifestBytes },
			keys: { delta: d.key, tombstones: st.tombKey },
			conflicts: c.conflicts,
			casConflicts: c.casConflicts,
			currentUpdated: c.currentUpdated,
			deltas: m.deltas!.length,
			deltaBytes: m.deltas!.reduce((a, x) => a + x.bytes, 0),
			tombstoneSets: m.tombstones!.length,
			tree: d.tree,
			timing: { load: d.loadMs, encode: d.encodeMs, put: d.putMs },
			// Thresholds this rev exceeds and whether an automatic compaction was started in the background.
			compaction,
		});
	});
}

/** POST /delete {gen?, ids}: one tombstone set in a new manifest revision. */
async function handleDelete(request: Request, env: Env, ctx: ExecutionContext, url: URL): Promise<Response> {
	const t0 = Date.now();
	const body = await readJson<{ gen?: string; ids: number[] }>(request);
	const ids = u32Array(body.ids, "ids");
	if (!ids.length) throw new HttpError(400, "no ids");
	const tb = tombBytes(ids);
	let key = "";
	const c = await commitRev(env, body.gen ?? url.searchParams.get("gen"), async ({ gen, rev, manifest }) => {
		const next = inherit(manifest);
		key = await putUnique(env, tombKey(gen, rev + 1), tb);
		next.tombstones!.push({ rev: rev + 1, key, count: tb.byteLength / 4 });
		return { ...next, change: { op: "delete", count: tb.byteLength / 4 } };
	});
	const compaction = maybeAutoCompact(env, ctx, c.manifest, "delete");
	return json({ gen: c.gen, rev: c.rev, count: tb.byteLength / 4, key, ms: Date.now() - t0, bytes: { tombstones: tb.byteLength, manifest: c.manifestBytes }, conflicts: c.conflicts, casConflicts: c.casConflicts, currentUpdated: c.currentUpdated, tombstoneSets: c.manifest.tombstones!.length, compaction });
}

// ---------- compaction ----------
//
// compact-plan   -> {baseRev, slabRev = baseRev + 1, cells, folded, splits}: cells holding delta entries; with `purge`,
//                   also every cell an earlier compaction rewrote at a rev <= the newest tombstone set (such slabs may
//                   hold ids a set hides, so they block dropping it; rewriting them lets the commit drop the sets);
//                   `splits` = [cell, newCell] for the rewritten cells whose entry count would exceed maxCellEntries
// compact-cells  -> for each cell: slab_to_records(old slab, sets with rev >= its slabRev)
//                   + delta_records_for_cell(delta, cell, sets with rev > delta.addedRev) per folded delta
//                   -> SlabJob(dim, cell, mean, format) -> slab/{cell:05}.{slabRev}; a split cell is 2-means split
//                   (split_slab) and re-encoded (split_slab_records) into slab/{cell:05}.{slabRev} (mean A) and
//                   slab/{newCell:05}.{slabRev} (mean B); returns the slabs-table rows
// compact-commit -> slabs.{rev}.json (rows replaced; split cells appended), manifest rev + 1 without the folded deltas
//                   and without the tombstone sets nothing can reference any more; with splits also fine.{rev}.f32
//                   (k + splits rows), k2 and a rebuilt centroid tree; current.json
// `compact` runs all three in one invocation (small compactions); bench/compact.mjs spreads compact-cells over build slots.
// After an upsert/delete commit, autoCompact runs plan + cells + commit for a bounded slice in ctx.waitUntil when the
// manifest's compaction thresholds are exceeded (idx/{gen}/compact.lock serializes the runs).

/** Inputs of a compaction, loaded from manifest `baseRev`. */
interface CompactBase {
	gen: string;
	baseRev: number;
	manifest: Manifest;
	files: IndexFiles;
	slabs: SlabMeta[];
	deltas: DeltaRef[];
	tombs: { rev: number; key: string; ids: Uint32Array }[];
}

async function loadCompactBase(env: Env, gen: string, baseRev: number): Promise<CompactBase> {
	const manifest = await readManifestR2(env, gen, baseRev);
	const files = manifestFiles(manifest);
	if (!files.slabs) throw new HttpError(400, `${gen} has no slabs table`);
	if ((manifest.format ?? 1) < 2) throw new HttpError(400, "compaction needs a format 2 or 3 generation");
	const [slabs, tombs] = await Promise.all([
		getBuffer(env.BUCKET, files.slabs).then((b) => decodeJson<SlabMeta[]>(b)),
		pmap(manifest.tombstones ?? [], SUBREQUEST_CONCURRENCY, async (t) => ({ rev: t.rev, key: t.key, ids: new Uint32Array(await getBuffer(env.BUCKET, t.key)) })),
	]);
	return { gen, baseRev, manifest, files, slabs, deltas: manifest.deltas ?? [], tombs };
}

/** A delta segment with its entry count per cell (and its bytes, when kept for compactCells). */
interface DeltaCells {
	ref: DeltaRef;
	counts: Map<number, number>;
	bytes?: Uint8Array;
}

async function loadDeltaCells(env: Env, deltas: DeltaRef[], keepBytes: boolean): Promise<Map<string, DeltaCells>> {
	await ensureWasm();
	const out = new Map<string, DeltaCells>();
	await pmap(deltas, SUBREQUEST_CONCURRENCY, async (d) => {
		const bytes = new Uint8Array(await getBuffer(env.BUCKET, d.key));
		const dc = thread.delta_cells(bytes);
		const counts = new Map<number, number>();
		for (let i = 0; i < dc.length; i += 2) if (dc[i + 1] > 0) counts.set(dc[i], dc[i + 1]);
		out.set(d.key, { ref: d, counts, ...(keepBytes ? { bytes } : {}) });
	});
	return out;
}

/** Cells to split ([cell, newCell]) and the candidates that could not be split. */
interface SplitPlan {
	splits: [number, number][];
	/** Estimated entries of every planned split (slab count + folded delta entries; an upper bound). */
	candidates: { cell: number; estimate: number }[];
	blocked: { cell: number; estimate: number; reason: string }[];
}

/**
 * Which of the rewritten `cells` (ascending) are split: those whose estimated entry count (the slab's count plus the
 * folded deltas' entries in the cell; tombstoned entries are not subtracted, so it is an upper bound) exceeds
 * `cap`. New cells are numbered k, k + 1, ... in cell order. A cell is not split when an unfolded delta has entries
 * in it (they were encoded against the cell's current mean, which the split replaces), nor in a two-level generation.
 */
function planSplits(b: CompactBase, cells: number[], folded: Set<string>, dcells: Map<string, DeltaCells>, cap: number): SplitPlan {
	const out: SplitPlan = { splits: [], candidates: [], blocked: [] };
	if (!(cap > 0)) return out;
	let next = b.manifest.k1 * b.manifest.k2;
	const ds = [...dcells.values()];
	for (const cell of cells) {
		let estimate = b.slabs[cell]?.[2] ?? 0;
		for (const d of ds) if (folded.has(d.ref.key)) estimate += d.counts.get(cell) ?? 0;
		if (estimate <= cap) continue;
		const unfolded = ds.find((d) => !folded.has(d.ref.key) && d.counts.has(cell));
		if (b.manifest.k1 !== 1) out.blocked.push({ cell, estimate, reason: "two-level generation (k1 > 1)" });
		else if (unfolded) out.blocked.push({ cell, estimate, reason: `unfolded delta ${unfolded.ref.key} has entries in the cell` });
		else {
			out.splits.push([cell, next++]);
			out.candidates.push({ cell, estimate });
		}
	}
	return out;
}

interface CompactPlanReq {
	gen?: string;
	/** Cells to rewrite (default: see above), or "all". */
	cells?: number[] | "all";
	/** Also rewrite the cells that keep tombstone sets alive (see above). */
	purge?: boolean;
	/** Split cap for this compaction (default manifest.compaction.maxCellEntries; 0 = no splits). */
	maxCellEntries?: number;
}

async function stageCompactPlan(env: Env, body: CompactPlanReq) {
	await ensureWasm();
	const gen = await targetGen(env, body.gen ?? null);
	const baseRev = await latestRev(env, gen);
	const b = await loadCompactBase(env, gen, baseRev);
	const nCells = b.manifest.k1 * b.manifest.k2;
	const dcells = await loadDeltaCells(env, b.deltas, false);
	const deltaCells = new Set<number>();
	for (const d of dcells.values()) for (const c of d.counts.keys()) deltaCells.add(c);
	const maxTomb = b.tombs.reduce((a, t) => Math.max(a, t.rev), 0);
	const tombCells = new Set<number>();
	if (b.tombs.length && body.purge) b.slabs.forEach((row, cell) => slabRevOf(row) > 0 && slabRevOf(row) <= maxTomb && tombCells.add(cell));
	let cells: number[];
	if (body.cells === "all") cells = Array.from({ length: nCells }, (_, i) => i);
	else if (body.cells) {
		cells = [...new Set(body.cells)].sort((x, y) => x - y);
		if (cells.some((c) => !Number.isInteger(c) || c < 0 || c >= nCells)) throw new HttpError(400, "cells out of range");
	} else cells = [...new Set([...deltaCells, ...tombCells])].sort((x, y) => x - y);
	// A delta can leave the manifest only if every cell it touches is rewritten; the others stay (partial compaction).
	const planned = new Set(cells);
	const folded = b.deltas.filter((d) => [...dcells.get(d.key)!.counts.keys()].every((c) => planned.has(c))).map((d) => d.key);
	const cap = body.maxCellEntries ?? b.manifest.compaction?.maxCellEntries ?? 0;
	if (!Number.isInteger(cap) || cap < 0 || cap === 1) throw new HttpError(400, "maxCellEntries must be 0 (no splits) or an integer >= 2");
	const sp = planSplits(b, cells, new Set(folded), dcells, cap);
	return {
		gen,
		baseRev,
		slabRev: baseRev + 1,
		k: nCells,
		cells,
		deltaCells: deltaCells.size,
		tombCells: tombCells.size,
		deltas: b.deltas.map((d) => d.key),
		/** Deltas whose cells are all in `cells`: fold these into compact-cells and compact-commit. */
		folded,
		deltaEntries: b.deltas.reduce((a, d) => a + d.count, 0),
		tombstoneSets: b.tombs.length,
		tombstones: b.tombs.reduce((a, t) => a + t.ids.length, 0),
		maxCellEntries: cap,
		/** [cell, newCell]: pass the pairs of each job's cells to compact-cells and all of them to compact-commit. */
		splits: sp.splits,
		splitCandidates: sp.candidates,
		splitBlocked: sp.blocked,
	};
}

interface CompactCellsReq {
	gen: string;
	baseRev: number;
	cells: number[];
	/** Cells in flight (each holds its old slab, the records and the new slab). Default 3. */
	concurrency?: number;
	/**
	 * Keys of the deltas whose records go into the new slabs (the plan's `folded`; default every delta of baseRev).
	 * A delta left out stays in the manifest after commit and keeps serving its entries.
	 */
	folded?: string[];
	/** The plan's [cell, newCell] pairs for (at least) the cells of this job. */
	splits?: [number, number][];
	/** Abort (503) before starting a cell after this time (epoch ms; set by automatic runs spread over slots). */
	deadline?: number;
}

interface CompactCellsOpts {
	folded?: string[];
	splits?: [number, number][];
	/** Abort (HttpError 503) before starting a cell after this time (Date.now()). */
	deadline?: number;
	/** Deltas already read (bytes kept), by key. */
	preloaded?: Map<string, DeltaCells>;
}

/** One executed split: entries before, and in the two halves. */
interface SplitDone {
	cell: number;
	newCell: number;
	entries: number;
	a: number;
	b: number;
}

async function compactCells(env: Env, b: CompactBase, cells: number[], concurrency: number, opts: CompactCellsOpts = {}) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { gen, manifest, slabs } = b;
	const slabRev = b.baseRev + 1;
	const format = manifest.format ?? 1;
	const dim = manifest.dim;
	const nCells = manifest.k1 * manifest.k2;
	// Delta segments with their cell counts (held for the whole job: they are small).
	const foldSet = opts.folded ? new Set(opts.folded) : null;
	const foldRefs = foldSet ? b.deltas.filter((d) => foldSet.has(d.key)) : b.deltas;
	const deltas = await pmap(foldRefs, SUBREQUEST_CONCURRENCY, async (d) => {
		const pre = opts.preloaded?.get(d.key);
		if (pre?.bytes) return { ref: d, bytes: pre.bytes, counts: pre.counts };
		const bytes = new Uint8Array(await getBuffer(env.BUCKET, d.key));
		const dc = thread.delta_cells(bytes);
		const counts = new Map<number, number>();
		for (let i = 0; i < dc.length; i += 2) counts.set(dc[i], dc[i + 1]);
		return { ref: d, bytes, counts };
	});
	const splitOf = new Map<number, number>();
	for (const [cell, newCell] of opts.splits ?? []) {
		if (!Number.isInteger(newCell) || newCell < nCells) throw new HttpError(400, `split of cell ${cell}: new cell ${newCell} must be >= k (${nCells})`);
		splitOf.set(cell, newCell);
	}
	if (splitOf.size && manifest.k1 !== 1) throw new HttpError(400, "cell splits need a flat generation (k1 = 1)");
	// Format 3 halves are re-encoded with the index's PQ codebook.
	const codebook = splitOf.size && cells.some((c) => splitOf.has(c)) && format === 3 ? new Uint8Array(await getBuffer(env.BUCKET, b.files.pq!)) : new Uint8Array(0);
	const sets = b.tombs.map((t) => ({ rev: t.rev, ids: t.ids }));
	const unionCache = new Map<string, Uint8Array>();
	const union = (pred: (rev: number) => boolean, tag: string) => {
		let u = unionCache.get(tag);
		if (!u) unionCache.set(tag, (u = idUnion(sets.filter((s) => pred(s.rev)).map((s) => s.ids))));
		return u;
	};
	// Old slab copies are hidden by the sets with rev >= their slabRev; delta entries by the sets newer than the delta.
	const slabTomb = (r: number) => union((rev) => rev >= r, `s${r}`);
	const deltaTomb = (addedRev: number) => union((rev) => rev > addedRev, `d${addedRev}`);
	const rows: [number, SlabMeta][] = [];
	const splitsDone: SplitDone[] = [];
	let readBytes = 0;
	let writtenBytes = 0;
	let oldEntries = 0;
	let newEntries = 0;
	let deltaEntries = 0;
	/** A slab of `cell` with `mean` from record chunks; `added[i]` = records taken from chunk i. */
	const buildSlab = (cell: number, mean: Float32Array, records: Uint8Array[]) => {
		const job = new thread.SlabJob(dim, cell, mean, format);
		let finished = false;
		try {
			const added = records.map((r) => job.add_records(r));
			const n = job.len();
			finished = true;
			return { out: job.finish(), n, added };
		} finally {
			if (!finished) safeFree(job, epoch);
		}
	};
	/** Builds a slab from records and writes it as `cell`'s slab revision slabRev. */
	const writeSlab = async (cell: number, mean: Float32Array, records: Uint8Array[]) => {
		const { out, n, added } = buildSlab(cell, mean, records);
		const oi = thread.slab_info(out);
		await putBytes(env.BUCKET, slabKey(gen, cell, slabRev), out);
		writtenBytes += out.byteLength;
		newEntries += n;
		rows.push([cell, [oi[5], oi[6], n, out.byteLength, slabRev]]);
		return { n, added };
	};
	await pmap(cells, concurrency, async (cell) => {
		if (!Number.isInteger(cell) || cell < 0 || cell >= nCells) throw new HttpError(400, `cell ${cell} out of range`);
		if (opts.deadline && Date.now() > opts.deadline) throw new HttpError(503, `compaction deadline exceeded before cell ${cell}`);
		const row = slabs[cell];
		const oldRev = slabRevOf(row);
		if (oldRev >= slabRev) throw new HttpError(409, `cell ${cell} already has slabRev ${oldRev} >= ${slabRev}`);
		const old = new Uint8Array(await getBuffer(env.BUCKET, slabKey(gen, cell, oldRev)));
		readBytes += old.byteLength;
		const info = thread.slab_info(old);
		if (info[2] !== cell) throw new HttpError(500, `slab ${slabKey(gen, cell, oldRev)} is for cell ${info[2]}`);
		oldEntries += info[1];
		const mean = thread.slab_centroid(old);
		const records = [thread.slab_to_records(old, slabTomb(oldRev))];
		for (const d of deltas) {
			if (!d.counts.has(cell)) continue;
			const r = thread.delta_records_for_cell(d.bytes, cell, deltaTomb(d.ref.addedRev));
			records.push(r);
		}
		const newCell = splitOf.get(cell);
		if (newCell === undefined) {
			const { added } = await writeSlab(cell, mean, records);
			deltaEntries += added.slice(1).reduce((x, y) => x + y, 0);
			return;
		}
		// Split: the merged cell (old entries + folded delta entries) -> 2-means over its decoded entries ->
		// every entry re-encoded against the nearer new mean. Cell keeps mean A, newCell gets mean B.
		const { out: merged, n, added } = buildSlab(cell, mean, records);
		deltaEntries += added.slice(1).reduce((x, y) => x + y, 0);
		records.length = 0;
		let means2: Float32Array;
		if (n >= 2) means2 = thread.split_slab(merged, new Float32Array(0), cell + 1);
		else {
			means2 = new Float32Array(2 * dim);
			means2.set(mean);
			means2.set(mean, dim);
		}
		const packed = thread.split_slab_records(merged, new Uint8Array(0), means2, codebook);
		const la = new DataView(packed.buffer, packed.byteOffset, 4).getUint32(0, true);
		const a = (await writeSlab(cell, means2.slice(0, dim), [packed.subarray(4, 4 + la)])).n;
		const bn = (await writeSlab(newCell, means2.slice(dim), [packed.subarray(4 + la)])).n;
		splitsDone.push({ cell, newCell, entries: n, a, b: bn });
		console.log(`compaction ${gen} slabRev ${slabRev}: split cell ${cell} (${n} entries) -> ${cell}: ${a}, ${newCell}: ${bn}`);
	});
	for (const [cell, newCell] of opts.splits ?? []) {
		if (cells.includes(cell) && !splitsDone.some((s) => s.cell === cell)) throw new HttpError(500, `split of cell ${cell} -> ${newCell} was not executed`);
	}
	rows.sort((x, y) => x[0] - y[0]);
	splitsDone.sort((x, y) => x.cell - y.cell);
	return { gen, baseRev: b.baseRev, slabRev, rows, cells: rows.length, readBytes, writtenBytes, oldEntries, deltaEntries, newEntries, splits: splitsDone };
}

async function stageCompactCells(env: Env, body: CompactCellsReq) {
	const b = await loadCompactBase(env, body.gen, body.baseRev);
	return compactCells(env, b, body.cells, Math.max(1, Math.min(body.concurrency ?? 3, 6)), { folded: body.folded, splits: body.splits, deadline: body.deadline });
}

type CompactCellsResult = Awaited<ReturnType<typeof compactCells>>;

/**
 * The cells phase of an automatic run spread over build slots auto-0 .. auto-{n-1} (env.SLOTS): the sorted cells cut
 * into chunks of contiguous cells (about AUTO_CHUNKS_PER_SLOT per slot, at least AUTO_MIN_CELLS_PER_SLOT cells each)
 * on a shared queue; each slot runs one compact-cells job at a time and takes the next chunk when it finishes, so a
 * slot on a slow host takes fewer cells (per-job times differ up to ~2.5x between hosts). Each job reads the
 * compaction base and the folded deltas itself (~0.5 s). Same result shape as compactCells.
 */
async function compactCellsOnSlots(env: Env, b: CompactBase, cells: number[], opts: { folded: string[]; splits: [number, number][]; deadline: number }, maxSlots: number): Promise<CompactCellsResult & { slotMs: number[]; jobs: number }> {
	const n = Math.max(1, Math.min(maxSlots, Math.ceil(cells.length / AUTO_MIN_CELLS_PER_SLOT)));
	const per = Math.max(AUTO_MIN_CELLS_PER_SLOT, Math.ceil(cells.length / (n * AUTO_CHUNKS_PER_SLOT)));
	const chunks: number[][] = [];
	for (let i = 0; i < cells.length; i += per) chunks.push(cells.slice(i, i + per));
	const rs: CompactCellsResult[] = [];
	let next = 0;
	const slotMs = await Promise.all(
		Array.from({ length: n }, async (_, i) => {
			const slot = `auto-${i}`;
			const t = Date.now();
			while (next < chunks.length) {
				const chunk = chunks[next++];
				if (Date.now() > opts.deadline) throw new HttpError(503, `compaction deadline exceeded before chunk at cell ${chunk[0]}`);
				const mine = new Set(chunk);
				const r = await env.SLOTS!.stage(slot, "compact-cells", { gen: b.gen, baseRev: b.baseRev, cells: chunk, concurrency: SUBREQUEST_CONCURRENCY, folded: opts.folded, splits: opts.splits.filter(([c]) => mine.has(c)), deadline: opts.deadline });
				if (r.status !== 200 || !r.data?.ok) throw new HttpError(r.status === 200 ? 500 : r.status, `slot ${slot} (${chunk.length} cells): ${r.data?.error ?? JSON.stringify(r.data).slice(0, 300)}`);
				rs.push(r.data.result as CompactCellsResult);
			}
			return Date.now() - t;
		}),
	);
	const sum = (f: (r: CompactCellsResult) => number) => rs.reduce((a, r) => a + f(r), 0);
	const rows = rs.flatMap((r) => r.rows).sort((x, y) => x[0] - y[0]);
	const splits = rs.flatMap((r) => r.splits).sort((x, y) => x.cell - y.cell);
	return { gen: b.gen, baseRev: b.baseRev, slabRev: b.baseRev + 1, rows, cells: rows.length, readBytes: sum((r) => r.readBytes), writtenBytes: sum((r) => r.writtenBytes), oldEntries: sum((r) => r.oldEntries), deltaEntries: sum((r) => r.deltaEntries), newEntries: sum((r) => r.newEntries), splits, slotMs, jobs: chunks.length };
}

interface CompactCommitReq {
	gen: string;
	baseRev: number;
	rows: [number, SlabMeta][];
	/** Keys of the delta segments folded into the rows (default: every delta of baseRev). */
	folded?: string[];
	/** The plan's [cell, newCell] pairs (every one executed by compact-cells; their rows are in `rows`). */
	splits?: [number, number][];
	/** With splits: measure the rebuilt centroid tree's recall on sample vectors (default true; automatic runs skip it). */
	measureTree?: boolean;
	/** Recorded in the manifest's change (automatic runs: the trigger). */
	auto?: Record<string, unknown>;
}

/**
 * Whether a tombstone set can be dropped from the manifest: no remaining delta or lexical/docs segment predates
 * it, no slab written at a rev in (0, set.rev] remains (such a slab may hold an id the set hides), and none of
 * its ids is a build id (rev-0 slabs hold ids [0, manifest.vectors), the driver's id space).
 */
function tombDroppable(t: { rev: number; ids: Uint32Array }, deltas: DeltaRef[], slabs: SlabMeta[], baseIds: number, segments: { addedRev: number }[] = []): boolean {
	if (deltas.some((d) => d.addedRev < t.rev)) return false;
	if (segments.some((s) => s.addedRev < t.rev)) return false;
	if (slabs.some((row) => slabRevOf(row) > 0 && slabRevOf(row) <= t.rev)) return false;
	for (const id of t.ids) if (id < baseIds) return false;
	return true;
}

/** The cell mean in the header section of a slab object (one range read). */
async function readSlabMean(env: Env, key: string, dim: number): Promise<Float32Array> {
	const o = await env.BUCKET.get(key, { range: { offset: 0, length: 64 + dim * 4 } });
	if (!o) throw new HttpError(500, `missing ${key}`);
	const b = new Uint8Array(await o.arrayBuffer());
	const h = parseSlabHeader(b);
	if (h.dim !== dim) throw new HttpError(500, `${key} has dim ${h.dim}, expected ${dim}`);
	return centroidFromBytes(b, h);
}

/**
 * The routing objects of a revision whose compaction split cells: fine means with k + splits rows (a split cell
 * keeps its index with mean A, each new cell is appended with mean B; the means are read back from the new slabs'
 * headers, which are what the slabs were encoded against) and, if the head has a centroid tree, a tree rebuilt over
 * them with the same groups/iters/seed/beam.
 */
async function splitRouting(env: Env, gen: string, rev: number, m: Manifest, files: IndexFiles, slabRev: number, splits: [number, number][], measureTree: boolean): Promise<{ fine: string; k: number; tree?: TreeRef; routing?: RoutingTables; fineBytes: number }> {
	const dim = m.dim;
	const k = m.k1 * m.k2;
	const old = await getF32(env.BUCKET, files.fine);
	if (old.length !== k * dim) throw new HttpError(500, `${files.fine} has ${old.length / dim} means, expected ${k}`);
	const kk = k + splits.length;
	const means = new Float32Array(kk * dim);
	means.set(old);
	await pmap(splits, SUBREQUEST_CONCURRENCY, async ([cell, newCell]) => {
		const [a, b] = await Promise.all([readSlabMean(env, slabKey(gen, cell, slabRev), dim), readSlabMean(env, slabKey(gen, newCell, slabRev), dim)]);
		means.set(a, cell * dim);
		means.set(b, newCell * dim);
	});
	const fine = await putUnique(env, `idx/${gen}/fine.${rev}.f32`, means);
	let tree: TreeRef | undefined;
	let treeBytes: Uint8Array | null = null;
	if (m.tree) {
		const t = m.tree;
		// The recall check costs ~1.3e10 multiply-adds per call at 2000 rows (exact top 3 over all k cells): measure
		// the tree's own beam on 1000 rows only (publish-rev measures more beams when asked).
		const req: TreeReq = { groups: Math.min(t.groups, kk), iters: t.iters ?? 10, seed: t.seed ?? 7, beam: t.beam, sampleRows: measureTree ? 125 : 0 };
		const built = await buildTree(env, gen, { dim, k1: 1, k2: kk }, files.coarse, fine, req, undefined, means);
		tree = { key: await putUnique(env, treeKey(gen, rev), built.bytes), ...built.ref };
		treeBytes = built.bytes;
	}
	// The head's routing objects describe the old means (and tree): write the same ones for the new means.
	const rr = routingReqOf(m.routing?.fine === files.fine ? m.routing : undefined);
	const routing = rr ? await writeRouting(env, gen, rev, fine, means, dim, rr, true, tree && treeBytes ? { key: tree.key, bytes: treeBytes } : null) : undefined;
	return { fine, k: kk, tree, routing, fineBytes: means.byteLength };
}

async function compactCommit(env: Env, body: CompactCommitReq) {
	const { gen, baseRev } = body;
	const slabRev = baseRev + 1;
	const baseM = await readManifestR2(env, gen, baseRev);
	const baseK = baseM.k1 * baseM.k2;
	const folded = new Set(body.folded ?? (baseM.deltas ?? []).map((d) => d.key));
	const splits = body.splits ?? [];
	const rowCells = new Set(body.rows.map(([cell]) => cell));
	splits.forEach(([cell, newCell], i) => {
		if (baseM.k1 !== 1) throw new HttpError(400, "cell splits need a flat generation (k1 = 1)");
		if (newCell !== baseK + i) throw new HttpError(400, `split ${i}: new cell ${newCell}, expected ${baseK + i} (new cells are numbered from k in order)`);
		if (!rowCells.has(cell) || !rowCells.has(newCell)) throw new HttpError(400, `split of cell ${cell}: rows for ${cell} and ${newCell} are both required`);
	});
	const splitCells = new Set(splits.map(([cell]) => cell));
	// A folded delta leaves the manifest, so every cell it has entries in must be among the rewritten rows
	// (otherwise those entries would vanish). The folded-key check in commitRev pins the delta set to this one.
	await ensureWasm();
	const deltaCellsMemo = new Map<string, Uint32Array>();
	const cellsOfDelta = async (key: string) => {
		let dc = deltaCellsMemo.get(key);
		if (!dc) deltaCellsMemo.set(key, (dc = thread.delta_cells(new Uint8Array(await getBuffer(env.BUCKET, key)))));
		return dc;
	};
	await pmap([...folded], SUBREQUEST_CONCURRENCY, async (key) => {
		const dc = await cellsOfDelta(key);
		for (let i = 0; i < dc.length; i += 2) {
			if (dc[i + 1] > 0 && !rowCells.has(dc[i])) throw new HttpError(409, `folded delta ${key} has entries in cell ${dc[i]}, which is not among the rewritten rows`);
		}
	});
	let dropped: number[] = [];
	let slabsKeyWritten = "";
	let tableBytes = 0;
	let routing: (Awaited<ReturnType<typeof splitRouting>> & { forFine: string }) | null = null;
	const c = await commitRev(env, gen, async ({ rev, manifest }) => {
		const files = manifestFiles(manifest);
		if (files.slabs !== manifestFiles(baseM).slabs) throw new HttpError(409, `slabs table changed since rev ${baseRev} (another compaction committed)`);
		if (manifest.k1 * manifest.k2 !== baseK || files.fine !== manifestFiles(baseM).fine) throw new HttpError(409, `routing table changed since rev ${baseRev}`);
		const deltas = manifest.deltas ?? [];
		for (const k of folded) if (!deltas.some((d) => d.key === k)) throw new HttpError(409, `folded delta ${k} is no longer in rev ${rev}`);
		const remaining = deltas.filter((d) => !folded.has(d.key));
		// Deltas that stay (unfolded, or committed since the plan) were encoded against the old means: none may hold
		// entries in a split cell, whose mean changes.
		if (splitCells.size) {
			for (const d of remaining) {
				const dc = await cellsOfDelta(d.key);
				for (let i = 0; i < dc.length; i += 2) if (dc[i + 1] > 0 && splitCells.has(dc[i])) throw new HttpError(409, `delta ${d.key} (not folded) has entries in split cell ${dc[i]}`);
			}
		}
		const table = decodeJson<SlabMeta[]>(await getBuffer(env.BUCKET, files.slabs!));
		const k = baseK + splits.length;
		for (const [cell, row] of body.rows) {
			if (!Number.isInteger(cell) || cell < 0 || cell >= k) throw new HttpError(400, `row cell ${cell} out of range`);
			if (slabRevOf(row) !== slabRev) throw new HttpError(400, `row for cell ${cell} has slabRev ${slabRevOf(row)}, expected ${slabRev}`);
			table[cell] = row;
		}
		if (table.length !== k || table.some((r) => !r)) throw new HttpError(400, `slabs table would have ${table.length} rows (holes: ${table.findIndex((r) => !r)}), expected ${k}`);
		const tombRefs = manifest.tombstones ?? [];
		const tombs = await pmap(tombRefs, SUBREQUEST_CONCURRENCY, async (t) => ({ ...t, ids: new Uint32Array(await getBuffer(env.BUCKET, t.key)) }));
		const keep = tombs.filter((t) => !tombDroppable(t, remaining, table, manifest.vectors, manifest.segments ?? []));
		dropped = tombs.filter((t) => !keep.includes(t)).map((t) => t.rev);
		const tj = JSON.stringify(table);
		tableBytes = tj.length;
		slabsKeyWritten = await putUnique(env, slabsTableKey(gen, rev + 1), tj);
		const next: Manifest = {
			...inherit(manifest),
			files: { ...files, slabs: slabsKeyWritten },
			deltas: remaining,
			tombstones: keep.map(({ rev: r, key, count }) => ({ rev: r, key, count })),
		};
		if (splits.length) {
			// Built once (a retry on a newer head reuses it: the means only depend on the base and the new slabs).
			if (!routing || routing.forFine !== files.fine) routing = { forFine: files.fine, ...(await splitRouting(env, gen, rev + 1, manifest, files, slabRev, splits, body.measureTree !== false)) };
			next.k2 = routing.k;
			next.files = { ...next.files!, fine: routing.fine };
			if (routing.tree) next.tree = routing.tree;
			if (routing.routing) next.routing = routing.routing;
			else delete next.routing;
		}
		const splitChange = splits.length ? { splits: splits.map(([cell, newCell]) => [cell, newCell, (table[cell] as SlabMeta)[2], (table[newCell] as SlabMeta)[2]]), k: { from: baseK, to: k } } : {};
		return { ...next, change: { op: "compact", baseRev, slabRev, cells: body.rows.length, foldedDeltas: folded.size, droppedTombstoneSets: dropped.length, ...splitChange, ...(body.auto ? { auto: body.auto } : {}) } };
	});
	const r = routing as (Awaited<ReturnType<typeof splitRouting>> & { forFine: string }) | null;
	return {
		gen,
		rev: c.rev,
		baseRev,
		slabRev,
		cells: body.rows.length,
		folded: folded.size,
		droppedTombstoneSets: dropped,
		slabsKey: slabsKeyWritten,
		slabsTableBytes: tableBytes,
		deltas: c.manifest.deltas!.length,
		tombstoneSets: c.manifest.tombstones!.length,
		k: c.manifest.k1 * c.manifest.k2,
		splits: splits.length,
		fine: r ? r.fine : null,
		tree: r?.tree ? { key: r.tree.key, groups: r.tree.groups, beam: r.tree.beam, nearestRecall: r.tree.nearestRecall ?? null, top3Recall: r.tree.top3Recall ?? null } : null,
		conflicts: c.conflicts,
		currentUpdated: c.currentUpdated,
	};
}

interface CompactReq extends CompactPlanReq {
	concurrency?: number;
}

/** Plan + rewrite + commit in one invocation (fine for a few hundred cells; bench/compact.mjs splits larger ones). */
async function stageCompact(env: Env, body: CompactReq) {
	const t0 = Date.now();
	const plan = await stageCompactPlan(env, body);
	if (!plan.cells.length && !plan.deltas.length && !plan.tombstoneSets) return { ...plan, noop: true };
	const b = await loadCompactBase(env, plan.gen, plan.baseRev);
	const tCells = Date.now();
	const r = await compactCells(env, b, plan.cells, Math.max(1, Math.min(body.concurrency ?? 3, 6)), { folded: plan.folded, splits: plan.splits });
	const cellsMs = Date.now() - tCells;
	const commit = await compactCommit(env, { gen: plan.gen, baseRev: plan.baseRev, rows: r.rows, folded: plan.folded, splits: plan.splits });
	return { plan: { ...plan, cells: plan.cells.length }, ...commit, readBytes: r.readBytes, writtenBytes: r.writtenBytes, oldEntries: r.oldEntries, deltaEntries: r.deltaEntries, newEntries: r.newEntries, splitsDone: r.splits, cellsMs, ms: Date.now() - t0 };
}

// ---------- automatic compaction ----------

const compactLockKey = (gen: string) => `idx/${gen}/compact.lock`;
const compactLastKey = (gen: string) => `idx/${gen}/compact.last.json`;

interface CompactLock {
	owner: string;
	/** Epoch ms when taken. */
	ms: number;
	at: string;
	trigger?: AutoTrigger;
}

/** What made a writer start an automatic compaction. */
interface AutoTrigger {
	/** The rev the upsert/delete committed and the thresholds it exceeded. */
	rev: number;
	op: string;
	reasons: string[];
	pressure: ReturnType<typeof compactionPressure>;
	/** Epoch ms of the trigger (the write's commit). */
	ms: number;
}

/** Outcome of one automatic run (idx/{gen}/compact.last.json keeps the latest and the 20 before it). */
interface AutoRun {
	owner: string;
	trigger: AutoTrigger;
	startedAt: string;
	finishedAt?: string;
	/** committed | noop | blocked | conflict | deadline | error */
	outcome: string;
	/** Trigger -> new rev committed (or -> giving up), ms. */
	ms?: number;
	phases?: { lock: number; plan: number; cells: number; commit: number };
	baseRev?: number;
	rev?: number;
	cells?: number;
	folded?: number;
	purgeCells?: number;
	skippedDeltas?: number;
	splits?: SplitDone[];
	/** Cells over maxCellEntries whose split this run deferred (slice larger than splitCellsPerRun). */
	splitsDeferred?: number[];
	k?: number;
	remainingDeltas?: number;
	droppedTombstoneSets?: number;
	readMB?: number;
	writtenMB?: number;
	staleLockTaken?: boolean;
	/** Build slots the cells phase ran on (1 = the writer itself) and each slot's wall time (ms). */
	slots?: number;
	slotMs?: number[];
	/** compact-cells jobs (chunks) the cells were cut into. */
	slotJobs?: number;
	note?: string;
	error?: string;
}

async function readCompactLock(env: Env, gen: string): Promise<{ lock: CompactLock; etag: string } | null> {
	const o = await env.BUCKET.get(compactLockKey(gen));
	if (!o) return null;
	try {
		return { lock: (await o.json()) as CompactLock, etag: o.etag };
	} catch {
		return { lock: { owner: "?", ms: o.uploaded.getTime(), at: o.uploaded.toISOString() }, etag: o.etag };
	}
}

/**
 * Takes idx/{gen}/compact.lock with a create-only PUT. A lock older than COMPACT_LOCK_STALE_MS is taken over with an
 * etag-conditional PUT (exactly one contender wins the swap).
 */
async function acquireCompactLock(env: Env, gen: string, owner: string, trigger: AutoTrigger): Promise<{ acquired: boolean; stale: boolean; holder?: CompactLock }> {
	const now = Date.now();
	const body = JSON.stringify({ owner, ms: now, at: new Date(now).toISOString(), trigger } satisfies CompactLock);
	for (let attempt = 0; attempt < 2; attempt++) {
		if (await putIfAbsent(env, compactLockKey(gen), body)) return { acquired: true, stale: false };
		const cur = await readCompactLock(env, gen);
		if (!cur) continue; // released in between: try the create-only PUT again
		if (now - cur.lock.ms < COMPACT_LOCK_STALE_MS) return { acquired: false, stale: false, holder: cur.lock };
		const ok = await env.BUCKET.put(compactLockKey(gen), body, { onlyIf: { etagMatches: cur.etag } });
		return { acquired: ok !== null, stale: true, holder: cur.lock };
	}
	return { acquired: false, stale: false };
}

async function releaseCompactLock(env: Env, gen: string, owner: string) {
	const cur = await readCompactLock(env, gen);
	if (cur?.lock.owner === owner) await env.BUCKET.delete(compactLockKey(gen));
}

async function recordAutoRun(env: Env, gen: string, run: AutoRun) {
	const o = await env.BUCKET.get(compactLastKey(gen));
	const prev = o ? ((await o.json().catch(() => null)) as { last?: AutoRun; recent?: AutoRun[] } | null) : null;
	const recent = [run, ...(prev?.recent ?? [])].slice(0, 21);
	await env.BUCKET.put(compactLastKey(gen), JSON.stringify({ last: run, recent }));
}

/** Isolate-local guard: the generation whose automatic run this isolate started and has not finished. */
const autoRunning = new Set<string>();

/**
 * Called after an upsert/delete committed `m`: when a compaction threshold is exceeded, starts autoCompact in the
 * background (ctx.waitUntil; the response does not wait). Returns what the response reports.
 */
function maybeAutoCompact(env: Env, ctx: ExecutionContext, m: Manifest, op: string): { reasons: string[]; started: boolean; note?: string } | null {
	const reasons = compactionExceeded(m);
	if (!reasons.length) return null;
	if (autoRunning.has(m.gen)) return { reasons, started: false, note: "a run started by this isolate is in progress" };
	autoRunning.add(m.gen);
	const trigger: AutoTrigger = { rev: m.rev ?? 0, op, reasons, pressure: compactionPressure(m), ms: Date.now() };
	const run = autoCompact(env, m.gen, trigger, m.compaction?.deadlineMs ?? AUTO_DEADLINE_MS)
		.catch((e) => console.error(`auto compaction ${m.gen}: ${e instanceof Error ? (e.stack ?? e.message) : String(e)}`))
		.finally(() => autoRunning.delete(m.gen));
	ctx.waitUntil(run);
	return { reasons, started: true };
}

/**
 * One automatic compaction run: lock, plan a bounded slice on the head, rewrite its cells, commit, record, unlock.
 * The slice takes whole deltas, oldest first, while their cells fit in compaction.cellsPerRun (a delta touching more
 * cells than that is left for bench/compact.mjs; the run reports it); when the tombstone threshold is exceeded, the
 * remaining budget goes to the cells that keep tombstone sets alive (purge cells, oldest slab revision first).
 * Rewritten cells over maxCellEntries are split. The run aborts without committing when it passes the deadline
 * (the cut-off writes become garbage for gc); a commit that loses to another compaction (409) is recorded as a
 * conflict, and the next write that finds a threshold exceeded triggers again.
 */
async function autoCompact(env: Env, gen: string, trigger: AutoTrigger, deadlineMs: number): Promise<AutoRun | null> {
	const owner = crypto.randomUUID();
	const deadline = trigger.ms + deadlineMs;
	const lock = await acquireCompactLock(env, gen, owner, trigger);
	if (!lock.acquired) {
		console.log(`auto compaction ${gen}: lock held by ${lock.holder?.owner ?? "?"} since ${lock.holder?.at ?? "?"}`);
		return null;
	}
	const run: AutoRun = { owner, trigger, startedAt: new Date().toISOString(), outcome: "error", staleLockTaken: lock.stale };
	const tLock = Date.now();
	const phases = { lock: tLock - trigger.ms, plan: 0, cells: 0, commit: 0 };
	run.phases = phases;
	/** Set while the cells phase runs (a failure there still reports how long it ran). */
	let tCellsStart = 0;
	try {
		await ensureWasm();
		const baseRev = await latestRev(env, gen);
		const b = await loadCompactBase(env, gen, baseRev);
		run.baseRev = baseRev;
		const policy = b.manifest.compaction;
		const reasons = compactionExceeded(b.manifest);
		if (!policy || !reasons.length) {
			run.outcome = "noop";
			run.note = policy ? `head rev ${baseRev} is below every threshold` : `head rev ${baseRev} has no compaction policy`;
			return run;
		}
		const slots = env.SLOTS ? Math.max(1, Math.min(policy.slots ?? AUTO_SLOTS, AUTO_MAX_SLOTS)) : 1;
		const slice = await planAutoSlice(env, b, policy, reasons, slots);
		phases.plan = Date.now() - tLock;
		run.skippedDeltas = slice.skipped.length;
		run.purgeCells = slice.purgeCells;
		if (slice.splitsDeferred.length) run.splitsDeferred = slice.splitsDeferred;
		if (!slice.cells.length) {
			run.outcome = slice.skipped.length ? "blocked" : "noop";
			run.note = slice.skipped.length ? `${slice.skipped.length} deltas touch more cells each than one run may rewrite (${slots > 1 ? `cellsPerSlot ${policy.cellsPerSlot ?? AUTO_CELLS_PER_SLOT} x ${slots} slots` : `cellsPerRun ${policy.cellsPerRun ?? AUTO_CELLS_PER_RUN}`}): run bench/compact.mjs` : "nothing to rewrite (tombstone sets held by build ids or no blocking slabs)";
			return run;
		}
		const tCells = Date.now();
		tCellsStart = tCells;
		const cellDeadline = deadline - (slice.splits.length ? AUTO_SPLIT_COMMIT_RESERVE_MS : AUTO_COMMIT_RESERVE_MS);
		let r: CompactCellsResult;
		if (slots > 1) {
			const rr = await compactCellsOnSlots(env, b, slice.cells, { folded: slice.folded, splits: slice.splits, deadline: cellDeadline }, slots);
			run.slots = rr.slotMs.length;
			run.slotMs = rr.slotMs;
			run.slotJobs = rr.jobs;
			r = rr;
		} else {
			r = await compactCells(env, b, slice.cells, SUBREQUEST_CONCURRENCY, { folded: slice.folded, splits: slice.splits, deadline: cellDeadline, preloaded: slice.dcells });
			run.slots = 1;
		}
		phases.cells = Date.now() - tCells;
		tCellsStart = 0;
		run.readMB = +(r.readBytes / 1e6).toFixed(1);
		run.writtenMB = +(r.writtenBytes / 1e6).toFixed(1);
		if (Date.now() > deadline) throw new HttpError(503, `compaction deadline exceeded after the cells (${phases.cells} ms)`);
		const tCommit = Date.now();
		const commit = await compactCommit(env, { gen, baseRev, rows: r.rows, folded: slice.folded, splits: slice.splits, measureTree: false, auto: { trigger: { rev: trigger.rev, op: trigger.op, reasons: trigger.reasons }, owner } });
		phases.commit = Date.now() - tCommit;
		Object.assign(run, { outcome: "committed", rev: commit.rev, cells: commit.cells, folded: commit.folded, splits: r.splits, k: commit.k, remainingDeltas: commit.deltas, droppedTombstoneSets: commit.droppedTombstoneSets.length });
		console.log(`auto compaction ${gen}: rev ${baseRev} -> ${commit.rev}, ${commit.cells} cells on ${run.slots} slots, ${commit.folded} deltas folded, ${r.splits.length} splits, ${Date.now() - trigger.ms} ms after the trigger`);
		return run;
	} catch (e) {
		const status = e instanceof HttpError ? e.status : 500;
		run.outcome = status === 409 ? "conflict" : status === 503 ? "deadline" : "error";
		run.error = e instanceof Error ? e.message : String(e);
		console.error(`auto compaction ${gen}: ${run.outcome}: ${run.error}`);
		return run;
	} finally {
		if (tCellsStart) phases.cells = Date.now() - tCellsStart;
		run.finishedAt = new Date().toISOString();
		run.ms = Date.now() - trigger.ms;
		await recordAutoRun(env, gen, run).catch((e) => console.error(`auto compaction ${gen}: recording the run failed: ${e}`));
		await releaseCompactLock(env, gen, owner).catch((e) => console.error(`auto compaction ${gen}: releasing the lock failed: ${e}`));
	}
}

/** The cells, folded deltas and splits of one automatic run (see autoCompact). */
async function planAutoSlice(env: Env, b: CompactBase, policy: CompactionPolicy, reasons: string[], slots: number) {
	const budget = slots > 1 ? (policy.cellsPerSlot ?? AUTO_CELLS_PER_SLOT) * slots : (policy.cellsPerRun ?? AUTO_CELLS_PER_RUN);
	const dcells = await loadDeltaCells(env, b.deltas, true);
	const chosen = new Set<number>();
	const folded: string[] = [];
	const skipped: string[] = [];
	const order = [...b.deltas].sort((x, y) => x.addedRev - y.addedRev || x.seq - y.seq);
	for (const d of order) {
		const own = [...dcells.get(d.key)!.counts.keys()];
		const add = own.filter((c) => !chosen.has(c));
		if (chosen.size + add.length > budget) {
			skipped.push(d.key);
			continue;
		}
		for (const c of add) chosen.add(c);
		folded.push(d.key);
	}
	let purgeCells = 0;
	if (reasons.some((r) => r.startsWith("tombstones")) && chosen.size < budget) {
		// Only sets without build ids can ever be dropped; rewrite the slabs that block the newest of those.
		const purgeable = b.tombs.filter((t) => !t.ids.some((id) => id < b.manifest.vectors));
		const maxRev = purgeable.reduce((a, t) => Math.max(a, t.rev), 0);
		const blockers = b.slabs
			.map((row, cell) => ({ cell, r: slabRevOf(row) }))
			.filter(({ cell, r }) => r > 0 && r <= maxRev && !chosen.has(cell))
			.sort((x, y) => x.r - y.r || x.cell - y.cell);
		for (const { cell } of blockers.slice(0, budget - chosen.size)) {
			chosen.add(cell);
			purgeCells++;
		}
	}
	const cells = [...chosen].sort((x, y) => x - y);
	const sp = planSplits(b, cells, new Set(folded), dcells, policy.maxCellEntries ?? 0);
	// A split commit costs ~5 s more (fine means + tree): only small slices split; the others defer it.
	const splitBudget = policy.splitCellsPerRun ?? AUTO_SPLIT_CELLS_PER_RUN;
	const deferred = sp.splits.length && cells.length > splitBudget ? sp.candidates.map((c) => c.cell) : [];
	if (deferred.length) console.log(`auto compaction ${b.gen}: ${cells.length} cells > splitCellsPerRun ${splitBudget}, deferring the split of cells ${deferred.join(",")}`);
	else if (sp.splits.length) console.log(`auto compaction ${b.gen}: splitting cells ${sp.candidates.map((c) => `${c.cell} (~${c.estimate})`).join(", ")}`);
	return { cells, folded, skipped, purgeCells, splits: deferred.length ? [] : sp.splits, splitsDeferred: deferred, splitBlocked: sp.blocked, dcells };
}

/** GET /index/status?gen=: update pressure, thresholds, lock and the latest automatic compaction of a generation. */
async function handleStatus(env: Env, url: URL): Promise<Response> {
	const current = (await readCurrent(env))?.cur ?? null;
	const gen = url.searchParams.get("gen") ?? current?.gen;
	if (!gen) throw new HttpError(404, "no published index");
	const rev = await latestRev(env, gen);
	const [m, lock, lastObj] = await Promise.all([readManifestR2(env, gen, rev), readCompactLock(env, gen), env.BUCKET.get(compactLastKey(gen))]);
	const files = manifestFiles(m);
	const last = lastObj ? ((await lastObj.json().catch(() => null)) as { last?: AutoRun; recent?: AutoRun[] } | null) : null;
	let cells: Record<string, unknown> | null = null;
	if (files.slabs && (m.format ?? 1) >= 2) {
		const table = decodeJson<SlabMeta[]>(await getBuffer(env.BUCKET, files.slabs));
		let max = { cell: -1, count: -1 };
		let entries = 0;
		table.forEach((row, cell) => {
			entries += row[2];
			if (row[2] > max.count) max = { cell, count: row[2] };
		});
		const cap = m.compaction?.maxCellEntries ?? 0;
		const top = Math.min(Math.max(0, Number(url.searchParams.get("top") ?? 10) || 0), table.length);
		const largest = table
			.map((row, cell) => [cell, row[2]])
			.sort((x, y) => y[1] - x[1] || x[0] - y[0])
			.slice(0, top);
		cells = { k: table.length, entries, mean: +(entries / table.length).toFixed(1), maxCell: max, overCap: cap > 0 ? table.filter((row) => row[2] > cap).length : 0, compacted: table.filter((row) => slabRevOf(row) > 0).length, largest };
	}
	const age = lock ? Date.now() - lock.lock.ms : 0;
	const recent = last?.recent ?? [];
	return json({
		gen,
		rev,
		current,
		k: m.k1 * m.k2,
		vectors: m.vectors,
		...compactionPressure(m),
		compaction: m.compaction ?? null,
		exceeded: compactionExceeded(m),
		compacting: lock && age < COMPACT_LOCK_STALE_MS ? { owner: lock.lock.owner, since: lock.lock.at, ageMs: age, trigger: lock.lock.trigger ?? null } : false,
		staleLock: lock && age >= COMPACT_LOCK_STALE_MS ? { owner: lock.lock.owner, since: lock.lock.at } : null,
		lastCompaction: last?.last ?? null,
		recentCompactions: recent.map((r) => ({ outcome: r.outcome, baseRev: r.baseRev ?? null, rev: r.rev ?? null, ms: r.ms ?? null, cells: r.cells ?? null, at: r.startedAt })),
		cells,
		files: { fine: files.fine, slabs: files.slabs, tree: m.tree?.key ?? null },
	});
}

// ---------- rollback ----------

/**
 * POST /build/rollback {gen, toRev, force?}: a new head revision with the content of `toRev` (routing table, slabs
 * table, tree, deltas, tombstone sets, defaults, policy), e.g. to undo a compaction or a cell split. Its objects must
 * still exist (run it before gc deletes them). Refused when the head lists deltas or tombstone sets that `toRev` does
 * not (writes after toRev would be lost) unless `force`.
 */
async function stageRollback(env: Env, body: { gen: string; toRev: number; force?: boolean }) {
	const { gen, toRev } = body;
	if (!gen || !Number.isInteger(toRev) || toRev < 0) throw new HttpError(400, "gen and toRev required");
	const target = await readManifestR2(env, gen, toRev);
	const tFiles = manifestFiles(target);
	const missing: string[] = [];
	for (const k of [tFiles.coarse, tFiles.fine, tFiles.pq, tFiles.slabs, target.tree?.key, ...(target.deltas ?? []).map((d) => d.key), ...(target.tombstones ?? []).map((t) => t.key)]) {
		if (k && !(await env.BUCKET.head(k))) missing.push(k);
	}
	if (missing.length) throw new HttpError(409, `rev ${toRev} references deleted objects: ${missing.slice(0, 5).join(", ")}`);
	const c = await commitRev(env, gen, async ({ rev, manifest }) => {
		const tk = new Set([...(target.deltas ?? []).map((d) => d.key), ...(target.tombstones ?? []).map((t) => t.key)]);
		const lost = [...(manifest.deltas ?? []).map((d) => d.key), ...(manifest.tombstones ?? []).map((t) => t.key)].filter((k) => !tk.has(k));
		if (lost.length && !body.force) throw new HttpError(409, `head rev ${rev} lists ${lost.length} deltas/tombstone sets that rev ${toRev} does not (pass force to drop them)`);
		const next = inherit(target);
		next.nextSeq = Math.max(manifest.nextSeq ?? 1, target.nextSeq ?? 1);
		// Ids allocated to keys since toRev must stay allocated: a rollback never reuses an id.
		const nextId = Math.max(manifest.nextId ?? 0, target.nextId ?? 0);
		if (nextId > 0) next.nextId = nextId;
		return { ...next, change: { op: "rollback", toRev, fromRev: rev } };
	});
	return { gen, rev: c.rev, toRev, k: c.manifest.k1 * c.manifest.k2, files: c.manifest.files, currentUpdated: c.currentUpdated };
}

// ---------- gc ----------

interface GcReq {
	gen: string;
	/** Revisions whose objects are kept (newest first). Default 2. */
	keep?: number;
	/** Also keep revisions superseded less than this long ago, and never delete objects younger than it. Default 300. */
	graceSec?: number;
	/** Default true: report only. */
	dryRun?: boolean;
}

/**
 * Deletes objects under idx/{gen}/ that none of the kept revisions references: old slab revisions, folded
 * deltas, dropped tombstone sets, superseded slabs tables, fine.{rev}.f32 / tree.{rev}.bin of older routing tables
 * (cell splits, publish-rev trees; never fine.f32) and the leftovers of writers that lost a race.
 * Manifests are kept (small; they are the history). Kept revisions: the newest `keep`, plus any revision
 * superseded less than graceSec ago (queries may still hold it for CURRENT_TTL_MS).
 */
async function stageGc(env: Env, body: GcReq) {
	const { gen } = body;
	if (!gen) throw new HttpError(400, "gen required");
	const keepN = Math.max(1, body.keep ?? 2);
	const graceMs = (body.graceSec ?? 300) * 1000;
	const dryRun = body.dryRun !== false;
	const now = Date.now();
	const revs = await listRevs(env, gen);
	if (!revs.length) throw new HttpError(404, `no manifest for ${gen}`);
	const kept = revs.filter((r, i) => i >= revs.length - keepN || (i + 1 < revs.length && now - revs[i + 1].uploaded.getTime() < graceMs)).map((r) => r.rev);
	const referenced = new Set<string>();
	const tables = new Set<string>();
	await pmap(kept, SUBREQUEST_CONCURRENCY, async (rev) => {
		const m = await readManifestR2(env, gen, rev);
		const f = manifestFiles(m);
		for (const k of [f.coarse, f.fine, f.pq, f.slabs]) if (k) referenced.add(k);
		if (f.slabs) tables.add(f.slabs);
		for (const d of m.deltas ?? []) referenced.add(d.key);
		for (const s of m.segments ?? []) referenced.add(s.key);
		for (const t of m.tombstones ?? []) referenced.add(t.key);
		if (m.tree) referenced.add(m.tree.key);
	});
	await pmap([...tables], SUBREQUEST_CONCURRENCY, async (key) => {
		const t = decodeJson<SlabMeta[]>(await getBuffer(env.BUCKET, key));
		t.forEach((row, cell) => referenced.add(slabKey(gen, cell, slabRevOf(row))));
	});
	const all = await listAll(env, `idx/${gen}/`);
	// Only the object kinds revisions create are candidates (manifests, base files and anything else stay).
	const kindOf = (key: string) => {
		const rest = key.slice(`idx/${gen}/`.length);
		if (/^fine\.\d+(\.[0-9a-f]+)?\.f32$/.test(rest)) return "fine"; // fine means of a split revision (never fine.f32)
		if (/^tree\.\d+(\.[0-9a-f]+)?\.bin$/.test(rest)) return "tree";
		return rest.startsWith("slab/") ? "slab" : rest.startsWith("delta/") ? "delta" : rest.startsWith("seg/") ? "segment" : rest.startsWith("tombstones.") ? "tombstones" : rest.startsWith("slabs.") ? "slabs-table" : null;
	};
	const garbage = all.filter((o) => kindOf(o.key) !== null && !referenced.has(o.key) && now - o.uploaded.getTime() >= graceMs);
	const kinds: Record<string, { n: number; bytes: number }> = {};
	for (const o of garbage) {
		const kind = kindOf(o.key)!;
		kinds[kind] ??= { n: 0, bytes: 0 };
		kinds[kind].n++;
		kinds[kind].bytes += o.size;
	}
	let deleted = 0;
	if (!dryRun) {
		for (let i = 0; i < garbage.length; i += 1000) {
			const keys = garbage.slice(i, i + 1000).map((o) => o.key);
			await env.BUCKET.delete(keys);
			deleted += keys.length;
		}
	}
	return { gen, dryRun, keptRevs: kept, objects: all.length, referenced: referenced.size, garbage: garbage.length, kinds, deleted, sample: garbage.slice(0, 10).map((o) => o.key) };
}

/**
 * Deletes every object under one of the allowed scratch prefixes (housekeeping of test generations), or under a
 * generation's intermediate build outputs that nothing reads after publish: build/{gen}/partials/ (k-means
 * accumulators), build/{gen}/parts/ (assign output read by finalize) and build/{gen}/pq/partials/ (bench/driver.mjs
 * deletes these after a successful publish unless --gcBuild 0).
 */
const SCRATCH_PREFIXES = ["build/smoke/", "build/g3smoke/", "idx/g3smoke/", "build/pqtest/", "idx/pqtest/", "build/g1/partials/test/", "tmp/probe/"];
const BUILD_SCRATCH_RE = /^build\/[A-Za-z0-9_-]+\/(partials|parts|pq\/partials)\/$/;

async function stageDeletePrefix(env: Env, body: { prefix: string; dryRun?: boolean }) {
	if (!SCRATCH_PREFIXES.includes(body.prefix) && !BUILD_SCRATCH_RE.test(body.prefix)) throw new HttpError(400, `prefix must be one of ${SCRATCH_PREFIXES.join(", ")} or build/{gen}/(partials|parts|pq/partials)/`);
	const objs = await listAll(env, body.prefix);
	const bytes = objs.reduce((a, o) => a + o.size, 0);
	let deleted = 0;
	if (body.dryRun === false) {
		for (let i = 0; i < objs.length; i += 1000) {
			const keys = objs.slice(i, i + 1000).map((o) => o.key);
			await env.BUCKET.delete(keys);
			deleted += keys.length;
		}
	}
	return { prefix: body.prefix, objects: objs.length, bytes, deleted, remaining: body.dryRun === false ? (await listAll(env, body.prefix)).length : objs.length };
}

// ---------- probe ----------

/** This isolate's id (random, made on first use: random values are not allowed at global scope). */
let isolateId = "";
let probesServed = 0;

/**
 * POST /build/probe {waitUntilMs?, tag?}: where a request ran (slot, isolate id, wasm memory, Cache API) for checking
 * slot isolation. With waitUntilMs, writes tmp/probe/{tag} that long after the response from ctx.waitUntil (GET
 * /stat?key=tmp/probe/{tag} shows whether background work survives in this kind of isolate).
 */
async function stageProbe(env: Env, body: { waitUntilMs?: number; tag?: string }, ctx: ExecutionContext) {
	isolateId ||= crypto.randomUUID().slice(0, 8);
	probesServed++;
	await ensureWasm();
	if (body.waitUntilMs) {
		const key = `tmp/probe/${body.tag ?? isolateId}`;
		const t0 = Date.now();
		ctx.waitUntil(sleep(Math.min(body.waitUntilMs, 60_000)).then(() => env.BUCKET.put(key, JSON.stringify({ slot: env.THREAD_SLOT ?? "local", iso: isolateId, delayMs: Date.now() - t0 }))));
	}
	return { slot: env.THREAD_SLOT ?? "local", iso: isolateId, served: probesServed, wasmMemMB: wasmRaw ? Math.round(wasmRaw.memory.buffer.byteLength / 1048576) : null, basesLoaded, cacheApi: edgeCache() !== null, slotsRpc: !!env.SLOTS };
}

// ---------- router ----------

const stages: Record<string, (env: Env, body: any, ctx: ExecutionContext) => Promise<unknown>> = {
	sample: stageSample,
	"init-coarse": stageInitCoarse,
	"kmeans-flat": stageKmeansFlat,
	"init-fine": stageInitFine,
	"kmeans-tree-map": stageKmeansTreeMap,
	"kmeans-inc-map": stageKmeansIncMap,
	"cell-neighbors": stageCellNeighbors,
	"kmeans-reduce": stageKmeansReduce,
	concat: stageConcat,
	"pq-residuals": stagePqResiduals,
	"pq-init": stagePqInit,
	"pq-train-map": stagePqTrainMap,
	"pq-train-reduce": stagePqTrainReduce,
	assign: stageAssign,
	finalize: stageFinalize,
	publish: stagePublish,
	"publish-rev": stagePublishRev,
	compact: stageCompact,
	"compact-plan": stageCompactPlan,
	"compact-cells": stageCompactCells,
	"compact-commit": compactCommit,
	gc: stageGc,
	rollback: stageRollback,
	"delete-prefix": stageDeletePrefix,
	probe: stageProbe,
};

function authorized(request: Request, env: Env): boolean {
	// Fail closed when the secret is not bound (otherwise "Bearer undefined" would pass).
	const token: unknown = env.BUILD_TOKEN;
	if (typeof token !== "string" || token === "") return false;
	const enc = new TextEncoder();
	const got = enc.encode(request.headers.get("authorization") ?? "");
	const want = enc.encode(`Bearer ${token}`);
	return got.byteLength === want.byteLength && crypto.subtle.timingSafeEqual(got, want);
}

export default {
	async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
		const url = new URL(request.url);
		const t0 = Date.now();
		try {
			if (url.pathname === "/health") {
				await ensureWasm();
				return json({ ok: true, wasm: true, wasmResets });
			}
			if (!authorized(request, env)) throw new HttpError(401, "unauthorized");
			if (url.pathname === "/query" && request.method === "POST") {
				return await handleQuery(request, env, ctx, url);
			}
			if (url.pathname.startsWith("/build/") && request.method === "POST") {
				const stage = stages[url.pathname.slice("/build/".length)];
				if (!stage) throw new HttpError(404, "unknown stage");
				const body = await readJson(request);
				// Stages are idempotent, so a wasm trap is retried once on a fresh instance.
				const result = await withWasmRecovery(`stage ${url.pathname}`, () => stage(env, body, ctx));
				return json({ ok: true, ms: Date.now() - t0, result });
			}
			if (url.pathname === "/debug/range" && request.method === "GET") {
				return await handleRangeProbe(env, ctx, url);
			}
			if (url.pathname === "/upsert" && request.method === "POST") {
				return await handleUpsert(request, env, ctx, url);
			}
			if (url.pathname === "/delete" && request.method === "POST") {
				return await handleDelete(request, env, ctx, url);
			}
			if (url.pathname === "/index/status" && request.method === "GET") {
				return await handleStatus(env, url);
			}
			// A manifest revision read fresh from R2 (default: the head of ?gen=, or of the current generation).
			if (url.pathname === "/manifest" && request.method === "GET") {
				const current = (await readCurrent(env))?.cur ?? null;
				const gen = url.searchParams.get("gen") ?? current?.gen;
				if (!gen) throw new HttpError(404, "no published index");
				const head = await latestRev(env, gen);
				const rev = optIntParam(url, "rev", 0) ?? head;
				const m = await readManifestR2(env, gen, rev);
				return json({ gen, rev, head, current, manifest: { ...m, notes: m.notes ? "(omitted)" : undefined } });
			}
			// Drops this isolate's cached current pointer (revisions are immutable, so nothing else can be stale).
			// Since manifest revisions replaced republishing, no edge-cache eviction is needed any more.
			if (url.pathname === "/admin/refresh" && request.method === "POST") {
				currentPtr = null;
				genHeads.clear();
				return json({ ok: true });
			}
			// Self-check of the R2 conditional PUTs the write path relies on (create-only and compare-and-swap).
			if (url.pathname === "/admin/cas-check" && request.method === "POST") {
				const key = `tmp/cas-check/${crypto.randomUUID()}`;
				const first = await putIfAbsent(env, key, "a");
				const second = await putIfAbsent(env, key, "b");
				const head = await env.BUCKET.head(key);
				const casWrong = (await env.BUCKET.put(key, "c", { onlyIf: { etagMatches: "0".repeat(32) } })) !== null;
				const casRight = (await env.BUCKET.put(key, "d", { onlyIf: { etagMatches: head!.etag } })) !== null;
				const final = await (await env.BUCKET.get(key))!.text();
				await env.BUCKET.delete(key);
				const ok = first && !second && !casWrong && casRight && final === "d";
				return json({ ok, createFirst: first, createSecond: second, casWrongEtag: casWrong, casRightEtag: casRight, final }, ok ? 200 : 500);
			}
			if (url.pathname === "/stat" && request.method === "GET") {
				const key = url.searchParams.get("key") ?? "";
				const head = await env.BUCKET.head(key);
				return json(head ? { key, size: head.size, etag: head.etag } : { key, missing: true }, head ? 200 : 404);
			}
			throw new HttpError(404, "not found");
		} catch (e) {
			if (e instanceof HttpError) return json({ error: e.message }, e.status);
			const msg = e instanceof Error ? `${e.name}: ${e.message}\n${e.stack ?? ""}` : String(e);
			return json({ error: msg, ms: Date.now() - t0 }, 500);
		}
	},
} satisfies ExportedHandler<Env>;
