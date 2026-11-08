/**
 * The query path (same semantics as the reference Worker, examples/cloudflare-worker/src/index.ts):
 *  1. route (flat over the cell means by default), build the PQ table (v3),
 *  2. phase 1: read the scan prefix of each probed slab (Range against the cache; a miss reads the whole
 *     object and keeps the local winners' int8 records under the "whole" miss policy, or just the prefix under
 *     "range" / "range+warm"), scan it as it arrives, filter tombstones, drop the buffer; scan every delta
 *     segment in parallel (int8 scores, no phase 2),
 *  3. phase 2: re-score the global top-T with the int8 rerank records, read with one multi-range request
 *     per slab (speculatively once only a few slabs are left to scan),
 *  4. top-k over reranked slab hits and delta hits, one copy per id.
 */
import { compileFilter, validateSchema, type AttrFilterExpr } from "./attrs.js";
import type { BlobRange, BlobStore, BlobTrace, PhaseStat } from "./store.js";
import { newTrace, nowMs, phaseStat, roundMs, snapTrace } from "./store.js";
import { addTrace, envelope, ioOf, Kernels, phaseSpan, type TraceEnvelope, type TraceEvent } from "./trace.js";
import type * as W from "./wasm/thread.js";
import type { ColdRouting, ColdStat } from "./cold.js";
import { PackedBlobStore } from "./pack.js";
import { WarmPrefixStore } from "./warm.js";
import { CachedBlobStore } from "./store.js";
import { concatBytes, deltaEntry, keys, NO_TOMBSTONES, parseDeltaHeader, tombForDelta, parseHits, parseSlabHeader, slabCentroid, slabEntry, slabRevOf, tombFor, ThreadError, type DeltaRef, type Manifest, type SegmentRef, type SlabRow, type TombState, type IndexFiles, type RoutingElem } from "./wire.js";

export interface QueryOptions {
	/** Results to return (default 10). */
	topK?: number;
	/** Slabs (cells) to scan; default `manifest.defaults.nprobe`. */
	nprobe?: number;
	/** Scan-tier candidates re-scored with the int8 tier (formats 2/3); 0 disables phase 2. Default from the manifest. */
	rerank?: number;
	/** Coarse groups to probe (two-level indexes); 0 = flat routing over all cells. Default from the manifest. */
	pc?: number;
	/** Also return each hit's payload bytes (extra range reads for hits whose slab was not read whole). */
	returnPayload?: boolean;
	/**
	 * Reads in flight per query. Default: the store's preference (`store.inflight`): 12 through the Cache API (not
	 * held to the 6-connection limit), 32 against S3 from Node, 6 against S3 inside a Worker; else 12.
	 */
	inflight?: number;
	/** Speculative rerank reads of the running top-T while the last slabs are still scanning (default true). */
	speculate?: boolean;
	/**
	 * Attribute filter: field conditions compiled against `manifest.attrs` (or already compiled words). Applied
	 * inside every scan before top-k insertion, so filtered-out entries never take candidate slots.
	 */
	filter?: AttrFilterExpr | Uint32Array;
	/**
	 * With a filter: keep scanning cells in routing order beyond `nprobe`, in batches, until `max(topK, rerank)`
	 * matching candidates were seen or this many cells were scanned (default 4 x nprobe, at most every cell;
	 * `nprobe` disables widening). Unfiltered queries never widen.
	 */
	maxNprobe?: number;
	/**
	 * Phase 2: rerank records of one slab are read as runs; a run grows while its span stays within this many bytes
	 * (default `store.coalesceBytes`, else 4096: adjacent records only).
	 */
	coalesceBytes?: number;
	/**
	 * Phase 2: when the span from a slab's first to last wanted record is at most this many bytes, read it as ONE
	 * range (default `store.spanBytes`, else 0 = off). Trades bytes for requests on stores without multi-range reads.
	 */
	spanBytes?: number;
	/** Cold-path policy (see `ColdPolicy`); `false` disables an index-wide default. */
	coldPolicy?: ColdPolicy | false;
	/**
	 * Cold routing only (`open({coldRouting: "tree"})`, before the full routing table has loaded): groups of the
	 * centroid tree whose cells are ranked (default `open`'s `coldBeam`).
	 */
	beam?: number;
	/**
	 * Phase 1, `PackedBlobStore` only (also behind a warm-pack layer): merge the scan-prefix reads of probed slabs whose
	 * prefixes lie within this many bytes of each other in the pack into ONE range read (`true` = 64 KiB; default off).
	 * The bytes in between (the earlier slab's rerank tier and payload, unprobed slabs) are read and dropped, so this
	 * pays only in a pack whose entries put a query's slabs next to each other (`bench/cold/make_pack.mjs --order
	 * tree`, measured with `bench/cold/pack_adjacency.mjs`). Not the phase-2 record merging.
	 */
	coalesceAcrossSlabs?: boolean | number;
	/**
	 * Record every store read of this query in `stats.trace.events` (kind, key, start, ms, ranges, bytes, cache
	 * hits/misses). Off by default: the envelope's spans, counters and kernels are always there.
	 */
	traceEvents?: boolean;
}

/**
 * Cheaper parameters for a query whose probed slabs are mostly not cached (each miss is an origin round trip):
 * when more than `threshold` (default 0.5) of the probed slabs would be read from the origin, the query scans
 * only the best `nprobe` of them and reranks `rerank` candidates (each at most the query's own value). A store
 * that cannot tell (no `contains`) counts as all misses when remote, all hits when local. A filtered query that
 * the policy applies to widens from the reduced `nprobe` (to 4 x it, unless `maxNprobe` was given).
 */
export interface ColdPolicy {
	nprobe?: number;
	rerank?: number;
	threshold?: number;
}

/** `coalesceAcrossSlabs: true`. */
export const CROSS_SLAB_GAP = 64 * 1024;

export interface Hit {
	id: number;
	/** Cosine similarity (int8-tier scale for reranked slab hits and delta hits; scan-tier scale when rerank = 0). */
	score: number;
	/** The u32 attribute word stored with the vector. */
	attrs: number;
	/** With `returnPayload`: the payload bytes (empty when none was stored). */
	payload?: Uint8Array;
	/** Where the winning copy came from. */
	source: "slab" | "delta";
}

export type ParamSource = "query" | "manifest" | "legacy";

export interface QueryParams {
	topK: number;
	nprobe: number;
	rerank: number;
	pc: number;
	source: { nprobe: ParamSource; rerank: ParamSource; pc: ParamSource };
	/** Present with a filter: the widening bound in effect. */
	filter?: true;
	maxNprobe?: number;
}

/** Loading a generation's routing table and PQ codebook (`openStats.base`, `timeline.base`). */
export interface BaseStat extends PhaseStat {
	/** ms reading `coarse.f32`, `fine.f32` and `pq.bin` (concurrently). */
	fetch: number;
	/** ms building the wasm `Searcher` and `PqCodebook` from them. */
	build: number;
	/** Bytes of each file. */
	files: { coarse: number; fine: number; pq: number };
}

/**
 * Where a query spent its time and reads: each phase's wall time (ms) and the store work that completed during it
 * (snapshots of the query's trace at the phase boundaries, so a speculative read issued in phase 1 that lands in
 * phase 2 counts in phase 2). Python's `stats["timeline"]` has the same fields.
 */
export interface QueryTimeline {
	/** Normalize, route over the cell means, build the PQ table (CPU; delta reads may start here). */
	route: PhaseStat;
	/** Scan-prefix reads and scans of the probed slabs (plus delta segments and speculative rerank reads). */
	phase1: PhaseStat;
	/** Rerank record reads and the rerank itself. */
	phase2: PhaseStat;
	/** Payload reads (`returnPayload`) and the final top-k. */
	payload: PhaseStat;
	/** Present when this query (re)loaded the routing table (after a wasm reset or a switch of generation). */
	base?: BaseStat;
}

export interface QueryStats {
	gen: string;
	rev: number;
	format: number;
	/** `cold`: ranked over the tree's best groups while the full routing table loads (`coldRouting: "tree"`). */
	routing: "flat" | "tree" | "cold";
	/** The routing table the probed cells were ranked with (`open({routing})`; the grouped table's for `cold`). */
	routingTable: RoutingElem;
	/** Cold routing: the beam and the group / delta-cell means this query had to read. */
	coldRouting?: ColdStat;
	/** Probed cells, in routing order (beyond `nprobe` when a filtered query widened). */
	cells: number[];
	/** Entries the attribute filter dropped (among those that would have entered a top-T) and distinct scan-tier candidates. */
	filterDropped: number;
	candidates: number;
	/** Entries scanned (slabs) and bytes of scan prefixes / of slab objects transferred. */
	scanned: number;
	scanBytes: number;
	slabBytes: number;
	/** Phase 2: T (0 when skipped), candidates, slabs with winners, record reads and bytes, local/speculative records used. */
	rerank: number;
	rerankCandidates: number;
	rerankSlabs: number;
	rerankReads: number;
	rerankBytes: number;
	rerankLocal: number;
	specRecords: number;
	specUsed: number;
	/** Update path: delta segments scanned, their entries, hits kept / dropped by tombstones / in the top k. */
	deltas: number;
	deltaBytes: number;
	deltaScanned: number;
	deltaHits: number;
	deltaDropped: number;
	deltaTop: number;
	tombstones: number;
	tombSets: number;
	tombDropped: number;
	/** Probed slabs whose whole object was in memory (a "whole" miss or a memory-cache hit): their winners' records were local. */
	wholeSlabs: number;
	/** Reads in flight and phase-2 merging the query used. */
	inflight: number;
	coalesceBytes: number;
	spanBytes: number;
	/** Cold-path policy: probed slabs not cached / probed (before the policy), and whether it applied. */
	cold?: { misses: number; probed: number; applied: boolean };
	/** `coalesceAcrossSlabs`: merged prefix reads issued, the slabs they carried, and their bytes (prefixes + gaps). */
	coalesced?: { reads: number; slabs: number; bytes: number };
	/** Store accounting for this query (cache hits/misses, origin reads, bytes, warm fetches queued). */
	io: BlobTrace;
	limiterRetries: number;
	/** 1, or 2 when the query was retried after a wasm reset. */
	attempt: number;
	wasmResets: number;
	wasmMemMB: number;
	/** ms (sub-ms where the runtime's clock allows; CPU-only phases read ~0 in Workers). */
	timing: { total: number; route: number; phase1: number; phase2: number; payload: number };
	/** Per-phase wall time and store work (see `QueryTimeline`). */
	timeline: QueryTimeline;
	/** Entries pushed into a scan's top-T heap (slabs and deltas): the heap work on top of scoring. */
	pushed: number;
	/** The query's trace envelope: phases as spans, counters, kernel CPU (docs/observability.md). */
	trace: TraceEnvelope;
}

export interface QueryResult {
	hits: Hit[];
	params: QueryParams;
	stats: QueryStats;
}

/** One manifest revision, parsed (plain JS, no wasm). */
export interface RevState {
	gen: string;
	rev: number;
	manifest: Manifest;
	files: IndexFiles;
	format: number;
	/** Per-cell slab layout (formats 2/3); null for v1 generations, whose slabs are read whole. */
	slabs: SlabRow[] | null;
	deltas: DeltaRef[];
	/** Lexical / docs segments, oldest first. */
	segments: SegmentRef[];
	tomb: TombState | null;
	baseKey: string;
}

/** Routing table and PQ codebook of a generation, in wasm memory. */
export interface Base {
	key: string;
	/** The full routing table (null for a cold base, which routes with `cold`). */
	searcher: W.Searcher | null;
	/** Cold routing: the centroid tree plus the groups' means loaded so far. */
	cold?: ColdRouting | null;
	pq: W.PqCodebook | null;
	/** The routing table the searcher was built from (f32 = `files.fine`). */
	routing: RoutingElem;
	/** Wasm instance epoch the objects were created in (they may be freed only in that epoch). */
	epoch: number;
}

export interface QueryContext {
	w: typeof W;
	store: BlobStore;
	rev: RevState;
	base: Base;
	/** Current wasm epoch (the query frees its PqTable only if it is unchanged). */
	epoch: () => number;
	wasmResets: number;
	wasmMemBytes: () => number;
	/** A delta segment (whole), shared between concurrent queries. */
	getDelta: (ref: DeltaRef, trace: BlobTrace) => Promise<Uint8Array>;
	/** The index-wide cold-path policy (`OpenOptions.coldPolicy`). */
	coldPolicy?: ColdPolicy;
}

export const QUERY_INFLIGHT = 12;
/** A filtered query's default `maxNprobe`, as a multiple of `nprobe`. */
export const FILTER_WIDEN = 4;
/** Rerank winners of one slab whose records lie within this span share one range (adjacent records). */
export const MULTI_COALESCE = 4 * 1024;
/** Speculate on the running top-T once only this many slabs are left to scan. */
const SPEC_AT = 6;

/** Bounds a query's in-flight reads; low-priority waiters (speculative reads) start only when nothing else waits. */
export class Limiter {
	private active = 0;
	private queue: (() => void)[] = [];
	private low: (() => void)[] = [];
	/** Most reads that were ever in flight at once (tests). */
	peak = 0;
	retries = 0;
	constructor(private max: number) {}
	/**
	 * A free slot is taken synchronously; a released slot is handed straight to the next waiter (it is never
	 * returned to the pool in between), so a newcomer arriving in the same tick cannot slip in: the limit is exact.
	 */
	async run<T>(fn: () => Promise<T>, low = false): Promise<T> {
		if (this.active < this.max) this.active++;
		else await new Promise<void>((r) => (low ? this.low : this.queue).push(r)); // the releaser hands over its slot
		if (this.active > this.peak) this.peak = this.active;
		try {
			return await fn();
		} finally {
			const next = this.queue.shift() ?? this.low.shift();
			if (next) next();
			else this.active--;
		}
	}
	/** Runs one read; retries once if the runtime closed it for the connection limit or R2 reported an internal error. */
	async read<T>(fn: () => Promise<T>, low = false): Promise<T> {
		try {
			return await this.run(fn, low);
		} catch (e) {
			if (!(e instanceof Error) || !(e.message.includes("connection limit") || e.message.includes("(10001)"))) throw e;
			this.retries++;
			return await this.run(fn, low);
		}
	}
}

/** Whether probed `cell`'s slab can be read without an origin request (see `ColdPolicy`). */
export function slabResident(store: BlobStore, rev: RevState, cell: number): boolean {
	const meta = rev.slabs?.[cell];
	const key = keys.slab(rev.gen, cell, meta ? slabRevOf(meta) : 0);
	// Its scan prefix is what phase 1 reads: a held prefix (a warm pack's) counts as resident.
	return store.contains?.(key, meta ? { offset: 0, length: meta[0] } : undefined) ?? !store.remote;
}

export function safeFree(o: { free(): void }): void {
	try {
		o.free();
	} catch {
		/* after a trap the instance is poisoned and free() throws too */
	}
}

interface Cand {
	cell: number;
	idx: number;
	score: number;
	attrs: number;
	payloadOff: number;
	payloadLen: number;
	/** Payload bytes when the slab was in memory whole. */
	payload: Uint8Array | null;
}

interface SlabInfo {
	centroid: Float32Array;
	rerankOff: number;
	recLen: number;
	payloadOff: number;
	key: string;
}

export async function executeQuery(ctx: QueryContext, qRaw: Float32Array, o: QueryOptions, attempt: number): Promise<QueryResult> {
	const { w, store, rev, base } = ctx;
	const { manifest, gen, format, tomb, deltas } = rev;
	const tStart = nowMs();
	const at = Date.now();
	const kern = new Kernels();
	const events: TraceEvent[] | null = o.traceEvents ? [] : null;
	if (qRaw.length !== manifest.dim) throw new ThreadError(`query dim ${qRaw.length} != index dim ${manifest.dim}`, 400);
	const k = o.topK ?? 10;
	if (!Number.isInteger(k) || k < 1) throw new ThreadError("topK must be an integer >= 1", 400);
	const defaults = manifest.defaults ?? { nprobe: 1, rerank: 32, pc: 0 };
	const src = (v: number | undefined): ParamSource => (v !== undefined ? "query" : manifest.defaults ? "manifest" : "legacy");
	let nprobe = o.nprobe ?? defaults.nprobe;
	const pc = o.pc ?? defaults.pc;
	let rerankParam = o.rerank ?? defaults.rerank;
	for (const [name, v, min] of [["nprobe", nprobe, 1], ["pc", pc, 0], ["rerank", rerankParam, 0]] as const) if (!Number.isInteger(v) || v < min) throw new ThreadError(`${name} must be an integer >= ${min}`, 400);
	// The attribute filter, compiled against the generation's schema; with one, probing may widen past nprobe.
	const filterWords = o.filter instanceof Uint32Array ? o.filter : compileFilter(manifest.attrs ? validateSchema(manifest.attrs) : null, o.filter);
	const filtered = filterWords.length > 0;
	const nCells = manifest.k1 * manifest.k2;
	if (o.maxNprobe !== undefined && (!Number.isInteger(o.maxNprobe) || o.maxNprobe < nprobe)) throw new ThreadError(`maxNprobe must be an integer >= nprobe (${nprobe})`, 400);
	let maxNprobe = filtered ? Math.min(nCells, o.maxNprobe ?? nprobe * FILTER_WIDEN) : nprobe;
	const params: QueryParams = { topK: k, nprobe, rerank: rerankParam, pc, source: { nprobe: src(o.nprobe), rerank: src(o.rerank), pc: src(o.pc) }, ...(filtered ? { filter: true as const, maxNprobe } : {}) };
	const speculate = o.speculate ?? true;
	const inflight = o.inflight ?? store.inflight ?? QUERY_INFLIGHT;
	const coalesce = o.coalesceBytes ?? store.coalesceBytes ?? MULTI_COALESCE;
	const spanBytes = o.spanBytes ?? store.spanBytes ?? 0;
	const coldPolicy = o.coldPolicy === false ? undefined : (o.coldPolicy ?? ctx.coldPolicy);
	const dim = manifest.dim;
	const q = kern.time("route", () => w.normalize_vector(qRaw));
	const flat = pc === 0 || pc >= manifest.k1;
	const lim = new Limiter(inflight);
	const trace = newTrace();
	const trStart = snapTrace(trace);
	/**
	 * Runs one store read with the query's trace; with `traceEvents`, with a trace of its own that is added to the
	 * query's afterwards and recorded as an event (started once the read holds a limiter slot).
	 */
	const rd: TracedRead = <T>(kind: string, key: string, ranges: number, fn: (t: BlobTrace) => Promise<T>): Promise<T> => {
		if (!events) return fn(trace);
		const t = newTrace();
		const s0 = nowMs();
		return fn(t).finally(() => {
			addTrace(trace, t);
			events.push({ kind, key, start: roundMs(s0 - tStart), ms: roundMs(nowMs() - s0), ranges, bytes: t.bytes, hits: t.hits, misses: t.misses });
		});
	};
	const epoch = ctx.epoch();
	// Delta segments are read first (usually from memory) and scanned while the slab reads are in flight.
	const deltaReads = deltas.map((d) => ctx.getDelta(d, trace));

	// One q·mean per cell: flat routing ranks them, and scan_delta scores every delta entry against its cell.
	const coldR = base.cold ?? null;
	let cellDots: Float32Array | null = null;
	let cells: number[];
	let coldStat: ColdStat | null = null;
	// Routed to the widening bound; an unfiltered query scans exactly its nprobe cells.
	const routeN = Math.max(nprobe, maxNprobe);
	if (coldR) {
		// Cold routing: the tree's best groups (their means read now unless an earlier query loaded them), then
		// the members of those groups ranked as the full table would rank them.
		const beam = o.beam ?? coldR.beam;
		if (!Number.isInteger(beam) || beam < 1) throw new ThreadError("beam must be an integer >= 1", 400);
		coldStat = coldR.newStat(beam);
		await coldR.ensureGroups(q, beam, trace, coldStat);
		cells = Array.from(kern.time("route", () => coldR.router.route_beam(q, beam, routeN)));
	} else {
		const searcher = base.searcher!;
		cells = kern.time("route", () => {
			cellDots = flat || deltas.length ? searcher.cell_dots(q) : null;
			return Array.from(flat ? searcher.route_from_dots(cellDots!, routeN) : searcher.route(q, pc, routeN));
		});
	}
	// Cold-path policy: fewer probes / a smaller rerank when most probed slabs would come from the origin. Cells are in
	// routing order (best first), so the cold probe set is the best `coldPolicy.nprobe` of them.
	let cold: QueryStats["cold"];
	if (coldPolicy && cells.length) {
		const probed = Math.min(nprobe, cells.length);
		const misses = cells.slice(0, probed).reduce((a, c) => a + (slabResident(store, rev, c) ? 0 : 1), 0);
		const applied = misses / probed > (coldPolicy.threshold ?? 0.5);
		if (applied) {
			if (coldPolicy.nprobe !== undefined) nprobe = Math.max(1, Math.min(nprobe, coldPolicy.nprobe));
			if (coldPolicy.rerank !== undefined) rerankParam = Math.max(0, Math.min(rerankParam, coldPolicy.rerank));
			if (filtered && o.maxNprobe === undefined) maxNprobe = Math.min(nCells, nprobe * FILTER_WIDEN);
			cells = cells.slice(0, Math.max(nprobe, maxNprobe));
			params.nprobe = nprobe;
			params.rerank = rerankParam;
			if (filtered) params.maxNprobe = maxNprobe;
		}
		cold = { misses, probed, applied };
	}
	// Format 1 slabs have no rerank tier: phase 2 is skipped whatever rerank says.
	const doRerank = format >= 2 && rev.slabs !== null && rerankParam > 0;
	const T = doRerank ? Math.max(k, rerankParam) : k;
	// Format 3: one asymmetric-distance table (m x 256 f32) per query, shared by every probed slab.
	const table = base.pq ? kern.time("route", () => w.PqTable.from_codebook(q, base.pq!)) : null;
	const tRoute = nowMs();
	const trRoute = snapTrace(trace);

	const packed: Uint8Array[] = [];
	/** Best scan-tier copy per id (the rule thread.merge applies), remembering where it lives. */
	const cand = new Map<number, Cand>();
	/** Rerank records of local winners from slabs that were read whole: `${cell}:${idx}`. */
	const local = new Map<string, Uint8Array>();
	const slabInfo = new Map<number, SlabInfo>();
	const versions = new Set<number>();
	let scanned = 0;
	let wholeSlabs = 0;
	let slabBytes = 0;
	let scanBytes = 0;
	let rerankReads = 0;
	let rerankBytes = 0;
	let specReadN = 0;
	let tombDropped = 0;
	let filterDropped = 0;
	let pushed = 0;
	let scannedCells = 0;
	const stats = { rerankLocal: 0, specUsed: 0, specRecords: 0 };

	/** Reads the rerank records `idxs` (ascending) of one slab: one multi-range request, else one read per run. */
	const fetchRecords = async (cell: number, idxs: number[], low: boolean): Promise<Map<number, Uint8Array>> => {
		const si = slabInfo.get(cell)!;
		const recLen = si.recLen;
		const out = new Map<number, Uint8Array>();
		if (!idxs.length) return out;
		const runs = coalesceRuns(idxs, recLen, coalesce, spanBytes);
		const place = (run: (typeof runs)[number], bytes: Uint8Array) => {
			for (const i of run.members) out.set(i, bytes.subarray((i - run.a) * recLen, (i - run.a + 1) * recLen));
		};
		const rangeOf = (run: (typeof runs)[number]): BlobRange => ({ offset: si.rerankOff + run.a * recLen, length: (run.b - run.a + 1) * recLen });
		if (runs.length > 1 && store.getRanges) {
			const got = await lim.read(() => rd(low ? "spec" : "records", si.key, runs.length, (t) => store.getRanges!(si.key, runs.map(rangeOf), { trace: t })), low);
			if (got) {
				rerankReads++;
				if (low) specReadN++;
				runs.forEach((run, i) => {
					rerankBytes += got[i].byteLength;
					place(run, got[i]);
				});
				return out;
			}
		}
		await Promise.all(
			runs.map(async (run) => {
				const r = rangeOf(run);
				const bytes = await lim.read(() => rd(low ? "spec" : "records", si.key, 1, (t) => store.get(si.key, r, { trace: t })), low);
				if (!bytes) throw new ThreadError(`missing slab ${si.key}`, 404);
				rerankReads++;
				if (low) specReadN++;
				rerankBytes += r.length;
				place(run, bytes);
			}),
		);
		return out;
	};

	let slabsDone = 0;
	let specFired = false;
	/** Speculative reads per cell (records by entry index); a failed one only means phase 2 reads those records itself. */
	const specReads = new Map<number, Promise<Map<number, Uint8Array> | null>>();
	const fireSpeculation = () => {
		const top = [...cand.values()].sort((a, b) => b.score - a.score).slice(0, T);
		const byCell = new Map<number, number[]>();
		for (const c of top) {
			if (local.has(`${c.cell}:${c.idx}`)) continue;
			let l = byCell.get(c.cell);
			if (!l) byCell.set(c.cell, (l = []));
			l.push(c.idx);
		}
		for (const [cell, idxs] of byCell) {
			idxs.sort((a, b) => a - b);
			stats.specRecords += idxs.length;
			specReads.set(cell, fetchRecords(cell, idxs, true).catch(() => null));
		}
	};

	// Cross-slab coalescing (phase 1, packs): groups of probed slabs whose prefixes lie close together in the pack,
	// each read with one range of the pack the first time one of its members is needed (routing order).
	const gap = o.coalesceAcrossSlabs === true ? CROSS_SLAB_GAP : typeof o.coalesceAcrossSlabs === "number" ? o.coalesceAcrossSlabs : -1;
	const crossSlab = gap >= 0 && rev.slabs ? planCrossSlab(store, rev, gen, cells, gap) : null;
	const coalesced = { reads: 0, slabs: 0, bytes: 0 };
	const mergedRead = (g: CrossSlabGroup): Promise<Uint8Array> => {
		if (!g.read) {
			coalesced.reads++;
			coalesced.slabs += g.members.size;
			coalesced.bytes += g.range.length;
			g.read = lim.read(() => rd("merged", g.pack.key, 1, (t) => g.pack.inner.get(g.pack.key, g.range, { trace: t }))).then((b) => {
				if (!b) throw new ThreadError(`pack ${g.pack.key} disappeared`, 404);
				return b;
			});
		}
		return g.read;
	};

	/** Phase 1 read: the scan prefix (v2/v3) against the cache, whole object on a miss; v1 slabs whole. */
	const readSlab = async (cell: number): Promise<{ prefix: Uint8Array; full: Uint8Array | null; key: string; slabRev: number }> => {
		if (!rev.slabs) {
			const key = keys.slab(gen, cell);
			const buf = await lim.read(() => rd("slab", key, 1, (t) => store.get(key, undefined, { trace: t })));
			if (!buf) throw new ThreadError(`missing slab ${key}`, 404);
			return { prefix: buf, full: buf, key, slabRev: 0 };
		}
		const meta = rev.slabs[cell];
		if (!meta) throw new ThreadError(`${rev.files.slabs} has no entry for cell ${cell}`);
		const slabRev = slabRevOf(meta);
		const key = keys.slab(gen, cell, slabRev);
		const scanLen = meta[0];
		const g = crossSlab?.get(cell);
		if (g) {
			const at = g.members.get(cell)!;
			return { prefix: (await mergedRead(g)).subarray(at, at + scanLen), full: null, key, slabRev };
		}
		const bytes = await lim.read(() => rd("prefix", key, 1, (t) => store.get(key, { offset: 0, length: scanLen }, { trace: t, fill: true })));
		if (!bytes) throw new ThreadError(`missing slab ${key}`, 404);
		// A store that fetched the whole object (cache miss, or a 200 to a Range request) returns a view into it.
		const full = bytes.byteOffset === 0 && bytes.buffer.byteLength === meta[3] && meta[3] > scanLen ? new Uint8Array(bytes.buffer) : null;
		return { prefix: bytes, full, key, slabRev };
	};

	/** Scans one cell; `batchEnd` is the index past the last cell of the batch it belongs to (the speculation cue). */
	const scanCell = async (cell: number, batchEnd: number): Promise<void> => {
			const { prefix, full, key, slabRev } = await readSlab(cell);
			slabBytes += (full ?? prefix).byteLength;
			if (full && full !== prefix) wholeSlabs++;
			scanBytes += prefix.byteLength;
			const hdr = parseSlabHeader(prefix);
			// Tombstones and the attribute filter are applied inside the scan (before top-k insertion): ids deleted or
			// replaced at a rev >= this slab's revision (every set for rev-0 slabs), entries whose attrs fail the filter.
			// scanStats receives [entries scanned, tombstoned skipped, filtered out, pushed into the heap].
			const tombs = tomb ? tombFor(tomb, slabRev) : NO_TOMBSTONES;
			const scanStats = new Uint32Array(4);
			let hits: Uint8Array;
			if (hdr.version === 3) {
				if (!table) throw new ThreadError(`slab ${cell} is v3 but ${gen} has no PQ codebook`);
				hits = kern.time("scan", () => w.scan_pq(prefix, table, q, T, tombs, filterWords, scanStats));
			} else hits = kern.time("scan", () => w.scan(prefix, q, T, tombs, filterWords, scanStats));
			tombDropped += scanStats[1];
			filterDropped += scanStats[2];
			pushed += scanStats[3];
			versions.add(hdr.version);
			scanned += hdr.count;
			packed.push(hits);
			const rerankable = doRerank && hdr.version >= 2;
			if (rerankable) {
				if (hdr.recLen !== 4 + dim) throw new ThreadError(`slab ${cell}: rerank record length ${hdr.recLen} != ${4 + dim}`);
				slabInfo.set(cell, { centroid: slabCentroid(prefix, hdr), rerankOff: hdr.rerankOff, recLen: hdr.recLen, payloadOff: hdr.payloadOff, key });
			}
			for (const h of parseHits(hits)) {
				const prev = cand.get(h.id);
				if (prev && h.score <= prev.score) continue;
				const e = slabEntry(prefix, hdr, h.idx);
				const payload = full && o.returnPayload ? full.slice(hdr.payloadOff + e.payloadOff, hdr.payloadOff + e.payloadOff + e.payloadLen) : null;
				cand.set(h.id, { cell, idx: h.idx, score: h.score, attrs: e.attrs, payloadOff: e.payloadOff, payloadLen: e.payloadLen, payload });
				if (rerankable && full) {
					const off = hdr.rerankOff + h.idx * hdr.recLen;
					local.set(`${cell}:${h.idx}`, full.slice(off, off + hdr.recLen));
				}
			}
			if (!rerankable) return;
			slabsDone++;
			// Speculate once only a few slabs of the last batch are left: an unfiltered query has one batch; a filtered
			// one may widen, so its batch is known to be last only at the bound or once T candidates match.
			if (speculate && !specFired && batchEnd - slabsDone <= SPEC_AT && (!filtered || batchEnd >= cells.length || cand.size >= T)) {
				specFired = true;
				fireSpeculation();
			}
	};
	// Phase 1 in batches: the first nprobe cells, then (filtered queries only) more cells in routing order until
	// T distinct candidates passed the filter or maxNprobe cells were scanned.
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

	// Delta segments: every entry scored with its int8 record (reranked scale, no phase 2). A delta hit is dropped
	// if its id is in a tombstone set newer than the rev that added the segment.
	const deltaHits: { id: number; score: number; attrs: number; payload: Uint8Array | null }[] = [];
	let deltaBytes = 0;
	let deltaScanned = 0;
	let deltaDropped = 0;
	const deltaPhase = Promise.all(
		deltas.map(async (d, i) => {
			const bytes = await deltaReads[i];
			deltaBytes += bytes.byteLength;
			const hdr = parseDeltaHeader(bytes);
			deltaScanned += hdr.count;
			// A delta entry is hidden by the sets newer than the rev that added its segment (applied inside the scan).
			const scanStats = new Uint32Array(4);
			// Cold routing: q·mean for just the cells this segment holds entries in (their rows read if needed).
			const qc = coldR ? await coldR.deltaDots(d.key, bytes, q, trace, coldStat!) : cellDots!;
			const hits = kern.time("deltaScan", () => w.scan_delta(bytes, q, qc, T, tomb ? tombForDelta(tomb, d.addedRev) : NO_TOMBSTONES, filterWords, scanStats));
			deltaDropped += scanStats[1];
			filterDropped += scanStats[2];
			pushed += scanStats[3];
			for (const h of parseHits(hits)) {
				const e = deltaEntry(bytes, hdr, h.idx);
				deltaHits.push({ id: h.id, score: h.score, attrs: e.attrs, payload: o.returnPayload ? bytes.slice(hdr.payloadOff + e.payloadOff, hdr.payloadOff + e.payloadOff + e.payloadLen) : null });
			}
		}),
	);

	let merged: ReturnType<typeof parseHits>;
	let slabResults: { id: number; score: number }[];
	let rerankSlabs = 0;
	let tPhase1: number;
	let trPhase1: BlobTrace;
	try {
		await Promise.all([slabPhase, deltaPhase]);
		const all = concatBytes(packed);
		packed.length = 0;
		merged = kern.time("merge", () => (all.byteLength === 0 ? [] : parseHits(w.merge(all, T))));
		tPhase1 = nowMs();
		trPhase1 = snapTrace(trace);

		// Phase 2: re-score the global top-T with the int8 tier, one multi-range read per slab with winners.
		if (doRerank && (versions.has(2) || versions.has(3))) {
			const byCell = new Map<number, { id: number; idx: number }[]>();
			for (const h of merged) {
				const c = cand.get(h.id);
				if (!c) throw new ThreadError(`merge returned unknown id ${h.id}`);
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
					ws.forEach((wn, pos) => {
						const l = local.get(`${cell}:${wn.idx}`);
						if (l) {
							records.set(l, pos * recLen);
							stats.rerankLocal++;
							return;
						}
						const sp = spec?.get(wn.idx);
						if (sp) {
							records.set(sp, pos * recLen);
							stats.specUsed++;
							return;
						}
						remote.push(pos);
					});
					const got = await fetchRecords(cell, remote.map((pos) => ws[pos].idx), false);
					for (const pos of remote) records.set(got.get(ws[pos].idx)!, pos * recLen);
					const scores = kern.time("rerank", () => w.rerank(q, si.centroid, records, ws.length));
					ws.forEach((wn, i) => rescored.push({ id: wn.id, score: scores[i] }));
				}),
			);
			slabResults = rescored;
		} else slabResults = merged.map((h) => ({ id: h.id, score: h.score }));
	} finally {
		if (table && epoch === ctx.epoch()) safeFree(table);
	}
	const tPhase2 = nowMs();
	const trPhase2 = snapTrace(trace);

	// Final top k over slab and delta hits, one copy per id (the best score).
	const best = new Map<number, Hit & { cand?: Cand }>();
	for (const r of slabResults) {
		const prev = best.get(r.id);
		if (prev && r.score <= prev.score) continue;
		const c = cand.get(r.id)!;
		best.set(r.id, { id: r.id, score: r.score, attrs: c.attrs, source: "slab", cand: c });
	}
	for (const h of deltaHits) {
		const prev = best.get(h.id);
		if (prev && h.score <= prev.score) continue;
		best.set(h.id, { id: h.id, score: h.score, attrs: h.attrs, source: "delta", ...(h.payload ? { payload: h.payload } : {}) });
	}
	const top = [...best.values()].sort((a, b) => b.score - a.score).slice(0, k);
	if (o.returnPayload) await fetchPayloads(top, slabInfo, store, lim, rd);
	const hits: Hit[] = top.map(({ cand: _c, ...h }) => h);
	const tEnd = nowMs();
	const timeline: QueryTimeline = { route: phaseStat(tRoute - tStart, trRoute, trStart), phase1: phaseStat(tPhase1 - tRoute, trPhase1, trRoute), phase2: phaseStat(tPhase2 - tPhase1, trPhase2, trPhase1), payload: phaseStat(tEnd - tPhase2, trace, trPhase2) };
	const deltaTop = hits.filter((h) => h.source === "delta").length;
	const traceEnv = envelope({
		op: "query",
		t0: tStart,
		at,
		ms: tEnd - tStart,
		attrs: { gen, rev: rev.rev, format, routing: coldR ? "cold" : flat ? "flat" : "tree", routingTable: base.routing, topK: k, nprobe: params.nprobe, rerank: params.rerank, pc, filtered, inflight, attempt },
		io: ioOf(trace),
		spans: [phaseSpan("route", 0, timeline.route), phaseSpan("phase1", tRoute - tStart, timeline.phase1), phaseSpan("phase2", tPhase1 - tStart, timeline.phase2), phaseSpan("payload", tPhase2 - tStart, timeline.payload)],
		counters: {
			hits: hits.length,
			cells: scannedCells,
			scanned,
			scanBytes,
			slabBytes,
			pushed,
			candidates: cand.size,
			filterDropped,
			tombDropped,
			wholeSlabs,
			rerankCandidates: rerankSlabs ? merged.length : 0,
			rerankSlabs,
			rerankReads,
			rerankBytes,
			rerankLocal: stats.rerankLocal,
			specRecords: stats.specRecords,
			specUsed: stats.specUsed,
			deltas: deltas.length,
			deltaBytes,
			deltaScanned,
			deltaHits: deltaHits.length,
			deltaDropped,
			deltaTop,
			tombstones: tomb?.total ?? 0,
			tombSets: tomb?.sets.length ?? 0,
			warms: trace.warms ?? 0,
		},
		kernels: kern.done(),
		...(events ? { events: events.sort((a, b) => a.start - b.start) } : {}),
	});
	return {
		hits,
		params,
		stats: {
			gen,
			rev: rev.rev,
			format,
			routing: coldR ? "cold" : flat ? "flat" : "tree",
			routingTable: base.routing,
			...(coldR ? { coldRouting: coldR.finishStat(coldStat!) } : {}),
			cells: cells.slice(0, scannedCells),
			filterDropped,
			candidates: cand.size,
			scanned,
			scanBytes,
			slabBytes,
			rerank: rerankSlabs ? T : 0,
			rerankCandidates: rerankSlabs ? merged.length : 0,
			rerankSlabs,
			rerankReads,
			rerankBytes,
			rerankLocal: stats.rerankLocal,
			specRecords: stats.specRecords,
			specUsed: stats.specUsed,
			deltas: deltas.length,
			deltaBytes,
			deltaScanned,
			deltaHits: deltaHits.length,
			deltaDropped,
			deltaTop,
			tombstones: tomb?.total ?? 0,
			tombSets: tomb?.sets.length ?? 0,
			tombDropped,
			wholeSlabs,
			inflight,
			coalesceBytes: coalesce,
			spanBytes,
			...(cold ? { cold } : {}),
			...(crossSlab ? { coalesced } : {}),
			io: trace,
			limiterRetries: lim.retries,
			attempt,
			wasmResets: ctx.wasmResets,
			wasmMemMB: Math.round(ctx.wasmMemBytes() / 1048576),
			timing: { total: roundMs(tEnd - tStart), route: roundMs(tRoute - tStart), phase1: roundMs(tPhase1 - tRoute), phase2: roundMs(tPhase2 - tPhase1), payload: roundMs(tEnd - tPhase2) },
			timeline,
			pushed,
			trace: traceEnv,
		},
	};
}

/**
 * Groups the ascending record indexes of one slab into runs read as one range each: the whole span when it is at
 * most `spanBytes`, else greedily while a run's span stays within `coalesceBytes`.
 */
export function coalesceRuns(idxs: number[], recLen: number, coalesceBytes: number, spanBytes = 0): { a: number; b: number; members: number[] }[] {
	if (!idxs.length) return [];
	if (idxs.length > 1 && (idxs[idxs.length - 1] - idxs[0] + 1) * recLen <= spanBytes) return [{ a: idxs[0], b: idxs[idxs.length - 1], members: [...idxs] }];
	const runs: { a: number; b: number; members: number[] }[] = [];
	for (const i of idxs) {
		const run = runs.at(-1);
		if (run && (i - run.a + 1) * recLen <= coalesceBytes) {
			run.b = i;
			run.members.push(i);
		} else runs.push({ a: i, b: i, members: [i] });
	}
	return runs;
}

interface CrossSlabGroup {
	pack: PackedBlobStore;
	/** The pack range read for the group. */
	range: BlobRange;
	/** Member cell -> offset of its prefix in the range. */
	members: Map<number, number>;
	read?: Promise<Uint8Array>;
}

/**
 * Groups the probed slabs of a query whose scan prefixes lie within `gap` bytes of each other in the pack behind
 * `store` (a `PackedBlobStore`, possibly behind a warm-pack layer and/or a cache, whose held prefixes are left out).
 * Returns cell -> group for groups of two or more; null when the store is not a pack.
 */
export function planCrossSlab(store: BlobStore, rev: RevState, gen: string, cells: readonly number[], gap: number): Map<number, CrossSlabGroup> | null {
	// Through a warm-pack layer and an in-process cache: prefixes either of them holds are read from there.
	const front = store;
	let base = store;
	if (base instanceof WarmPrefixStore) base = base.inner;
	if (base instanceof CachedBlobStore) base = base.inner;
	if (!(base instanceof PackedBlobStore) || !rev.slabs) return null;
	const pack = base;
	const items: { cell: number; offset: number; length: number }[] = [];
	for (const cell of cells) {
		const meta = rev.slabs[cell];
		if (!meta) continue;
		const key = keys.slab(gen, cell, slabRevOf(meta));
		const r = { offset: 0, length: meta[0] };
		if (front !== pack && front.contains?.(key, r)) continue;
		const at = pack.rangeOf(key, r);
		if (at) items.push({ cell, ...at });
	}
	items.sort((a, b) => a.offset - b.offset);
	const out = new Map<number, CrossSlabGroup>();
	let run: typeof items = [];
	const flush = () => {
		if (run.length > 1) {
			const start = run[0].offset;
			const end = Math.max(...run.map((i) => i.offset + i.length));
			const g: CrossSlabGroup = { pack, range: { offset: start, length: end - start }, members: new Map(run.map((i) => [i.cell, i.offset - start])) };
			for (const i of run) out.set(i.cell, g);
		}
		run = [];
	};
	let end = -Infinity;
	for (const it of items) {
		if (run.length && it.offset - end > gap) flush();
		run.push(it);
		end = run.length === 1 ? it.offset + it.length : Math.max(end, it.offset + it.length);
	}
	flush();
	return out;
}

/** Payloads of slab hits whose slab was not in memory: one (multi-)range read per slab. */
/** One store read of a query, run with the trace it should count in (see `rd` in `executeQuery`). */
type TracedRead = <T>(kind: string, key: string, ranges: number, fn: (t: BlobTrace) => Promise<T>) => Promise<T>;

async function fetchPayloads(top: (Hit & { cand?: Cand })[], slabInfo: Map<number, SlabInfo>, store: BlobStore, lim: Limiter, rd: TracedRead): Promise<void> {
	const bySlab = new Map<string, { hit: Hit & { cand?: Cand }; range: BlobRange }[]>();
	for (const h of top) {
		const c = h.cand;
		if (!c || h.source !== "slab") continue;
		if (c.payload) {
			h.payload = c.payload;
			continue;
		}
		if (c.payloadLen === 0) {
			h.payload = new Uint8Array(0);
			continue;
		}
		const si = slabInfo.get(c.cell);
		if (!si) throw new ThreadError(`payload of id ${h.id}: slab ${c.cell} was read without its layout`);
		let l = bySlab.get(si.key);
		if (!l) bySlab.set(si.key, (l = []));
		l.push({ hit: h, range: { offset: si.payloadOff + c.payloadOff, length: c.payloadLen } });
	}
	await Promise.all(
		[...bySlab].map(async ([key, items]) => {
			const ranges = items.map((i) => i.range);
			const got = (store.getRanges && items.length > 1 ? await lim.read(() => rd("payload", key, ranges.length, (trace) => store.getRanges!(key, ranges, { trace }))) : null) ?? (await Promise.all(ranges.map((r) => lim.read(() => rd("payload", key, 1, (trace) => store.get(key, r, { trace }))))));
			items.forEach((it, i) => {
				const b = got[i];
				if (!b) throw new ThreadError(`missing slab ${key}`, 404);
				it.hit.payload = b.slice();
			});
		}),
	);
}
