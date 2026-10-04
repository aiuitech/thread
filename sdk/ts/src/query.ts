/**
 * The query path (same semantics as the reference Worker, examples/cloudflare-worker/src/index.ts):
 *  1. route (flat over the cell means by default), build the PQ table (v3),
 *  2. phase 1: read the scan prefix of each probed slab (Range against the cache; a miss reads the whole
 *     object and keeps the local winners' int8 records), scan it as it arrives, filter tombstones, drop the
 *     buffer; scan every delta segment in parallel (int8 scores, no phase 2),
 *  3. phase 2: re-score the global top-T with the int8 rerank records, read with one multi-range request
 *     per slab (speculatively once only a few slabs are left to scan),
 *  4. top-k over reranked slab hits and delta hits, one copy per id.
 */
import type { BlobRange, BlobStore, BlobTrace, PhaseStat } from "./store.js";
import { newTrace, nowMs, phaseStat, roundMs, snapTrace } from "./store.js";
import type * as W from "./wasm/thread.js";
import { concatBytes, deltaEntry, keys, NO_TOMBSTONES, parseDeltaHeader, tombForDelta, parseHits, parseSlabHeader, slabCentroid, slabEntry, slabRevOf, tombFor, ThreadError, type DeltaRef, type Manifest, type SegmentRef, type SlabRow, type TombState, type IndexFiles } from "./wire.js";

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
	/** Reads in flight per query (default 12; Cache API reads are not held to the 6-connection limit). */
	inflight?: number;
	/** Speculative rerank reads of the running top-T while the last slabs are still scanning (default true). */
	speculate?: boolean;
}

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
	routing: "flat" | "tree";
	/** Probed cells, in routing order. */
	cells: number[];
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
	/** Store accounting for this query (cache hits/misses, origin reads, bytes). */
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
	searcher: W.Searcher;
	pq: W.PqCodebook | null;
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
}

export const QUERY_INFLIGHT = 12;
/** Rerank winners of one slab whose records lie within this span share one range (adjacent records). */
const MULTI_COALESCE = 4 * 1024;
/** Speculate on the running top-T once only this many slabs are left to scan. */
const SPEC_AT = 6;

/** Bounds a query's in-flight reads; low-priority waiters (speculative reads) start only when nothing else waits. */
export class Limiter {
	private active = 0;
	private queue: (() => void)[] = [];
	private low: (() => void)[] = [];
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
	if (qRaw.length !== manifest.dim) throw new ThreadError(`query dim ${qRaw.length} != index dim ${manifest.dim}`, 400);
	const k = o.topK ?? 10;
	if (!Number.isInteger(k) || k < 1) throw new ThreadError("topK must be an integer >= 1", 400);
	const defaults = manifest.defaults ?? { nprobe: 1, rerank: 32, pc: 0 };
	const src = (v: number | undefined): ParamSource => (v !== undefined ? "query" : manifest.defaults ? "manifest" : "legacy");
	const nprobe = o.nprobe ?? defaults.nprobe;
	const pc = o.pc ?? defaults.pc;
	const rerankParam = o.rerank ?? defaults.rerank;
	for (const [name, v, min] of [["nprobe", nprobe, 1], ["pc", pc, 0], ["rerank", rerankParam, 0]] as const) if (!Number.isInteger(v) || v < min) throw new ThreadError(`${name} must be an integer >= ${min}`, 400);
	const params: QueryParams = { topK: k, nprobe, rerank: rerankParam, pc, source: { nprobe: src(o.nprobe), rerank: src(o.rerank), pc: src(o.pc) } };
	const speculate = o.speculate ?? true;
	const dim = manifest.dim;
	const q = w.normalize_vector(qRaw);
	// Format 1 slabs have no rerank tier: phase 2 is skipped whatever rerank says.
	const doRerank = format >= 2 && rev.slabs !== null && rerankParam > 0;
	const T = doRerank ? Math.max(k, rerankParam) : k;
	const flat = pc === 0 || pc >= manifest.k1;
	const lim = new Limiter(o.inflight ?? QUERY_INFLIGHT);
	const trace = newTrace();
	const trStart = snapTrace(trace);
	const epoch = ctx.epoch();
	// Delta segments are read first (usually from memory) and scanned while the slab reads are in flight.
	const deltaReads = deltas.map((d) => ctx.getDelta(d, trace));

	// One q·mean per cell: flat routing ranks them, and scan_delta scores every delta entry against its cell.
	const cellDots = flat || deltas.length ? base.searcher.cell_dots(q) : null;
	const cells = Array.from(flat ? base.searcher.route_from_dots(cellDots!, nprobe) : base.searcher.route(q, pc, nprobe));
	// Format 3: one asymmetric-distance table (m x 256 f32) per query, shared by every probed slab.
	const table = base.pq ? w.PqTable.from_codebook(q, base.pq) : null;
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
	let slabBytes = 0;
	let scanBytes = 0;
	let rerankReads = 0;
	let rerankBytes = 0;
	let specReadN = 0;
	let tombDropped = 0;
	const stats = { rerankLocal: 0, specUsed: 0, specRecords: 0 };

	/** Reads the rerank records `idxs` (ascending) of one slab: one multi-range request, else one read per run. */
	const fetchRecords = async (cell: number, idxs: number[], low: boolean): Promise<Map<number, Uint8Array>> => {
		const si = slabInfo.get(cell)!;
		const recLen = si.recLen;
		const out = new Map<number, Uint8Array>();
		if (!idxs.length) return out;
		const runs: { a: number; b: number; members: number[] }[] = [];
		for (const i of idxs) {
			const run = runs.at(-1);
			if (run && (i - run.a + 1) * recLen <= MULTI_COALESCE) {
				run.b = i;
				run.members.push(i);
			} else runs.push({ a: i, b: i, members: [i] });
		}
		const place = (run: (typeof runs)[number], bytes: Uint8Array) => {
			for (const i of run.members) out.set(i, bytes.subarray((i - run.a) * recLen, (i - run.a + 1) * recLen));
		};
		const rangeOf = (run: (typeof runs)[number]): BlobRange => ({ offset: si.rerankOff + run.a * recLen, length: (run.b - run.a + 1) * recLen });
		if (runs.length > 1 && store.getRanges) {
			const got = await lim.read(() => store.getRanges!(si.key, runs.map(rangeOf), { trace }), low);
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
				const bytes = await lim.read(() => store.get(si.key, r, { trace }), low);
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

	/** Phase 1 read: the scan prefix (v2/v3) against the cache, whole object on a miss; v1 slabs whole. */
	const readSlab = async (cell: number): Promise<{ prefix: Uint8Array; full: Uint8Array | null; key: string; slabRev: number }> => {
		if (!rev.slabs) {
			const key = keys.slab(gen, cell);
			const buf = await lim.read(() => store.get(key, undefined, { trace }));
			if (!buf) throw new ThreadError(`missing slab ${key}`, 404);
			return { prefix: buf, full: buf, key, slabRev: 0 };
		}
		const meta = rev.slabs[cell];
		if (!meta) throw new ThreadError(`${rev.files.slabs} has no entry for cell ${cell}`);
		const slabRev = slabRevOf(meta);
		const key = keys.slab(gen, cell, slabRev);
		const scanLen = meta[0];
		const bytes = await lim.read(() => store.get(key, { offset: 0, length: scanLen }, { trace, fill: true }));
		if (!bytes) throw new ThreadError(`missing slab ${key}`, 404);
		// A store that fetched the whole object (cache miss, or a 200 to a Range request) returns a view into it.
		const full = bytes.byteOffset === 0 && bytes.buffer.byteLength === meta[3] && meta[3] > scanLen ? new Uint8Array(bytes.buffer) : null;
		return { prefix: bytes, full, key, slabRev };
	};

	const slabPhase = Promise.all(
		cells.map(async (cell) => {
			const { prefix, full, key, slabRev } = await readSlab(cell);
			slabBytes += (full ?? prefix).byteLength;
			scanBytes += prefix.byteLength;
			const hdr = parseSlabHeader(prefix);
			// Tombstones are applied inside the scan (before top-k insertion): ids deleted or replaced at a rev >= this
			// slab's revision (every set for rev-0 slabs). scanStats receives [entries scanned, tombstoned skipped].
			const tombs = tomb ? tombFor(tomb, slabRev) : NO_TOMBSTONES;
			const scanStats = new Uint32Array(2);
			let hits: Uint8Array;
			if (hdr.version === 3) {
				if (!table) throw new ThreadError(`slab ${cell} is v3 but ${gen} has no PQ codebook`);
				hits = w.scan_pq(prefix, table, q, T, tombs, scanStats);
			} else hits = w.scan(prefix, q, T, tombs, scanStats);
			tombDropped += scanStats[1];
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
			if (speculate && !specFired && cells.length - slabsDone <= SPEC_AT) {
				specFired = true;
				fireSpeculation();
			}
		}),
	);

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
			const scanStats = new Uint32Array(2);
			const hits = w.scan_delta(bytes, q, cellDots!, T, tomb ? tombForDelta(tomb, d.addedRev) : NO_TOMBSTONES, scanStats);
			deltaDropped += scanStats[1];
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
		merged = all.byteLength === 0 ? [] : parseHits(w.merge(all, T));
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
					const scores = w.rerank(q, si.centroid, records, ws.length);
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
	if (o.returnPayload) await fetchPayloads(top, slabInfo, store, lim, trace);
	const hits: Hit[] = top.map(({ cand: _c, ...h }) => h);
	const tEnd = nowMs();
	return {
		hits,
		params,
		stats: {
			gen,
			rev: rev.rev,
			format,
			routing: flat ? "flat" : "tree",
			cells,
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
			deltaTop: hits.filter((h) => h.source === "delta").length,
			tombstones: tomb?.total ?? 0,
			tombSets: tomb?.sets.length ?? 0,
			tombDropped,
			io: trace,
			limiterRetries: lim.retries,
			attempt,
			wasmResets: ctx.wasmResets,
			wasmMemMB: Math.round(ctx.wasmMemBytes() / 1048576),
			timing: { total: roundMs(tEnd - tStart), route: roundMs(tRoute - tStart), phase1: roundMs(tPhase1 - tRoute), phase2: roundMs(tPhase2 - tPhase1), payload: roundMs(tEnd - tPhase2) },
			timeline: { route: phaseStat(tRoute - tStart, trRoute, trStart), phase1: phaseStat(tPhase1 - tRoute, trPhase1, trRoute), phase2: phaseStat(tPhase2 - tPhase1, trPhase2, trPhase1), payload: phaseStat(tEnd - tPhase2, trace, trPhase2) },
		},
	};
}

/** Payloads of slab hits whose slab was not in memory: one (multi-)range read per slab. */
async function fetchPayloads(top: (Hit & { cand?: Cand })[], slabInfo: Map<number, SlabInfo>, store: BlobStore, lim: Limiter, trace: BlobTrace): Promise<void> {
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
			const got = (store.getRanges && items.length > 1 ? await lim.read(() => store.getRanges!(key, ranges, { trace })) : null) ?? (await Promise.all(ranges.map((r) => lim.read(() => store.get(key, r, { trace })))));
			items.forEach((it, i) => {
				const b = got[i];
				if (!b) throw new ThreadError(`missing slab ${key}`, 404);
				it.hit.payload = b.slice();
			});
		}),
	);
}
