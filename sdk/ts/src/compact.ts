/**
 * @experimental Dense compaction in one isolate: the SDK's port of the Worker's compaction (plan, cells, commit,
 * `compact.lock`, docs/update-path.md "Compaction" / "Automatic compaction" / "Cell splitting"). One call folds a
 * bounded slice of delta segments into new slab revisions, splits rewritten cells that pass `maxCellEntries`
 * (2-means, new fine means, a rebuilt centroid tree) and drops the tombstone sets nothing references any more.
 *
 * The slice is bounded three ways: cells (`cellsPerRun`), time (`deadlineMs`; at the deadline the run commits the
 * cells it finished whose deltas are complete, instead of throwing the work away) and memory (`maxBytes`: whole
 * deltas, oldest first, while the estimated peak fits). The output is the Worker's, object for object: slab
 * revisions `slab/{cell:05}.{baseRev + 1}`, `slabs.{rev}.json`, `fine.{rev}.f32` and `tree.{rev}.bin` after splits,
 * and the same manifest fields, so the Worker and the SDK can compact one index (both take `compact.lock`).
 */
import { CachedBlobStore, type BlobStore } from "./store.js";
import { WarmPrefixStore } from "./warm.js";
import type * as W from "./wasm/thread.js";
import { buildRoutingObjects } from "./routing.js";
import { hidesDense, hidesLexical, idUnion, NO_TOMBSTONES, type TombScope, decodeJson, DELTA_ENTRY_BYTES, keys, manifestFiles, parseDeltaHeader, parseManifest, parseSlabsTable, ThreadError, type CompactionPolicy, type DeltaRef, type IndexFiles, type Manifest, type RoutingTables, type SlabRow, type TombRef, type TreeRef } from "./wire.js";
import { commitRev, inherit, latestRev, putIfAbsent, putUnique, writable, type WriterContext } from "./write.js";

// ---------- policy and pressure ----------

/** Update pressure of a revision: what the compaction thresholds are compared with. */
export interface CompactionPressure {
	deltas: number;
	deltaBytes: number;
	deltaEntries: number;
	/** Tombstone ids over all sets (with repeats). */
	tombstones: number;
	tombSets: number;
}

export function compactionPressure(m: Manifest): CompactionPressure {
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

/** The thresholds of `manifest.compaction` a revision exceeds (empty: none, or no policy), worded as the Worker words them. */
export function compactionExceeded(m: Manifest, policy: CompactionPolicy | undefined = m.compaction): string[] {
	if (!policy) return [];
	const p = compactionPressure(m);
	const out: string[] = [];
	if (p.deltas > policy.maxDeltas) out.push(`deltas ${p.deltas} > ${policy.maxDeltas}`);
	if (p.deltaBytes > policy.maxDeltaBytes) out.push(`deltaBytes ${p.deltaBytes} > ${policy.maxDeltaBytes}`);
	if (p.tombstones > policy.maxTombstones) out.push(`tombstones ${p.tombstones} > ${policy.maxTombstones}`);
	return out;
}

/** Whether a revision is due for a dense compaction (`ix.compactionDue()`): no I/O, just the manifest. */
export interface CompactionStatus {
	/** A threshold is exceeded: `compact()` would do work. */
	due: boolean;
	reasons: string[];
	pressure: CompactionPressure;
	/** The policy compared with (`manifest.compaction`), or null when the generation has none (never due). */
	policy: CompactionPolicy | null;
}

export function compactionStatus(m: Manifest): CompactionStatus {
	const reasons = compactionExceeded(m);
	return { due: reasons.length > 0, reasons, pressure: compactionPressure(m), policy: m.compaction ?? null };
}

// ---------- options and result ----------

export interface CompactOptions {
	/**
	 * Run even when no threshold of `manifest.compaction` is exceeded (or the generation has no policy): fold every
	 * delta the budgets allow and split oversized cells. Default false: like the Worker's automatic runs, a call on
	 * a head below every threshold is a `noop`.
	 */
	force?: boolean;
	/**
	 * Estimated peak memory of the call (bytes, default 48 MiB): the folded deltas held for the run, the tombstone
	 * sets, and the cells in flight (a fold about 3x the old slab plus 6x the new one; a split 8x plus its decoded
	 * vectors), or, when the call splits, the fine means of the commit (3 copies), whichever is more. Deltas that
	 * would pass it wait for a later call. It also sizes the cells: with a small budget cells are split before
	 * `maxCellEntries` (`CompactionLimits.cellEntries`), so that every cell stays splittable by a call with the
	 * same budget. Use one budget for a generation: a smaller one later may find cells it cannot split (413).
	 */
	maxBytes?: number;
	/** Cells one call may rewrite (default `compaction.cellsPerRun`, else 256). */
	cellsPerRun?: number;
	/**
	 * Wall-clock budget of the call (ms, default `compaction.deadlineMs`, else 27,000), counted from the start of the
	 * call. The read phase reads the deltas' cells oldest first and stops at a third of `deadline - 2.5 s` (the first
	 * 8 deltas are always read); the slice is planned with what was read. No cell starts after `deadline - 2.5 s` (up
	 * to 7 s with splits, by the size of the fine means); the cells finished by then are committed when their deltas
	 * are complete (`outcome: "partial"`), else nothing is (`"deadline"`).
	 *
	 * The oldest delta is not bound by the deadline (the progress guarantee): the cells it needs to commit (its own,
	 * those of the folded deltas that share them, and the splits numbered before any of them) run first, with the
	 * tombstone sets they need, whatever the time, and are committed. So a call that plans always folds at least the
	 * oldest delta (or splits the cell that keeps it out, `presplit`), however large the backlog and however short
	 * the deadline; only the lock, a conflict or a 413 stops it. Such a call can take longer than `deadlineMs` when
	 * the oldest delta alone needs more (a slow store, thousands of tombstone sets); its note then says so.
	 */
	deadlineMs?: number;
	/** Cells in flight at most (default 4; fewer when `maxBytes` does not hold more). */
	concurrency?: number;
	/**
	 * Split cap for this call (default `compaction.maxCellEntries`, else 0 = no splits). Cells split at the smaller
	 * of this and half the largest cell `maxBytes` can split (`CompactionLimits.cellEntries`).
	 */
	maxCellEntries?: number;
	/** A slice of more cells than this defers its splits (default `compaction.splitCellsPerRun`, else 160). */
	splitCellsPerRun?: number;
	/**
	 * Also rewrite the cells that keep tombstone sets alive (slabs written at a rev up to the newest set). Default:
	 * when the tombstone threshold is exceeded, or with `force`.
	 */
	purge?: boolean;
	/** Take `idx/{gen}/compact.lock` (default true; the Worker's automatic runs take it too). */
	lock?: boolean;
	/**
	 * Lock owner (default a random id). A stable owner (a Durable Object's id) lets a call take over its own lock
	 * left by a crashed run at once, instead of after the 10-minute staleness.
	 */
	owner?: string;
	/** Append the run to `idx/{gen}/compact.last.json` (default true; what the Worker's `/index/status` shows). */
	record?: boolean;
}

/**
 * The read phase of a call (`CompactResult.reads`). The deltas' cells are read oldest first only until the slice is
 * full (or the read phase has used its share of the deadline); a tombstone set's ids are read only when a cell the
 * call rewrites or a delta it folds is filtered by it, or a drop decision depends on them.
 */
export interface CompactReads {
	/** Delta segments whose cells were read (one get each when small, which is also the fold's input; else two range gets). */
	deltas: number;
	/** Delta segments the base lists (0 when the call did not plan: locked, or below every threshold). */
	deltasListed: number;
	/** Tombstone sets whose ids were read (each once, in the whole call). */
	tombSets: number;
	/** Tombstone sets the base lists. */
	tombSetsListed: number;
	/** From the start of the call to the plan (lock, manifests, slabs table, deltas' cells), ms. */
	ms: number;
}

/** One executed split: entries before, and in the two halves. */
export interface SplitDone {
	cell: number;
	newCell: number;
	entries: number;
	a: number;
	b: number;
}

export interface CompactResult {
	gen: string;
	/**
	 * `committed`: the slice is in a new revision. `partial`: the deadline cut the slice; the finished part is
	 * committed. `noop`: nothing to do. `blocked`: deltas wait and nothing else could be done in this call (only when
	 * a slice's splits are deferred; `more` is true). Input that no call with these budgets can take is not `blocked`:
	 * the call throws a `ThreadError` with status 413 that names it, before any cell grows past what the budget can
	 * split (see `CompactionLimits`).
	 * `locked`: another run holds `compact.lock`. `deadline`: the deadline passed before anything committable was
	 * done. `conflict`: another writer changed what the slice depends on (a compaction, or a delta in a split cell).
	 */
	outcome: "committed" | "partial" | "noop" | "blocked" | "locked" | "deadline" | "conflict";
	/** The head after the call (the new revision when one was committed). */
	rev: number;
	/** The revision the slice was planned on (null when the call did not plan). */
	baseRev: number | null;
	/** Cells rewritten in the committed revision (new cells of splits included). */
	cells: number;
	/** Delta segments folded into slabs (they left the manifest). */
	folded: number;
	/** Delta segments left in the head. */
	remainingDeltas: number;
	/**
	 * Deltas the slice could not take (cells or memory), of those the call read. The deltas it did not read wait too
	 * (`reads.deltasListed - reads.deltas`).
	 */
	skippedDeltas: number;
	/** What the call read of the base's backlog. */
	reads: CompactReads;
	/** Cells rewritten only to let tombstone sets go. */
	purgeCells: number;
	splits: SplitDone[];
	/** Cells over `maxCellEntries` whose split waits for a later call. */
	splitsDeferred: number[];
	/** Cells of the head after the call. */
	k: number;
	droppedTombstoneSets: number[];
	readBytes: number;
	writtenBytes: number;
	/** The planner's estimate of this call's peak memory (bytes), at most `maxBytes`. */
	estimatedPeakBytes: number;
	/** Cells split on their own (no delta folded into them) because they kept a delta from folding. */
	presplit: number[];
	/** What a call with this budget can process in the generation (absent when the call did not plan). */
	limits?: CompactionLimits;
	/** Cells that ran in parallel. */
	concurrency: number;
	/** Thresholds the base exceeded when the call started. */
	reasons: string[];
	/** Whether another call has work to do (a threshold is still exceeded, a split was deferred, the slice was cut). */
	more: boolean;
	conflicts: number;
	currentUpdated: boolean;
	staleLockTaken: boolean;
	ms: number;
	note?: string;
}

/** The wasm exports compaction uses (the SDK's bindings, or the Worker's own wasm-bindgen module). */
export type CompactWasm = Pick<typeof W, "SlabJob" | "slab_info" | "slab_centroid" | "slab_to_records" | "delta_records_for_cell" | "split_slab" | "split_slab_records" | "build_centroid_tree" | "centroid_tree_info" | "routing_table_encode" | "grouped_table_encode" | "grouped_table_info">;

export const COMPACT_DEFAULT_MAX_BYTES = 48 * 1024 * 1024;
const DEFAULT_CELLS_PER_RUN = 256;
const DEFAULT_DEADLINE_MS = 27_000;
const DEFAULT_SPLIT_CELLS_PER_RUN = 160;
const DEFAULT_CONCURRENCY = 4;
/** Time kept for the commit when deciding whether to start another cell (more with splits: fine means + tree). */
const COMMIT_RESERVE_MS = 2_500;
/**
 * With splits, up to 4.5 s more: what g4's split commits took (a 25 MB fine table and a rebuilt tree; 9.4 s for two
 * splits with the recall check). Scaled by the fine means' bytes when the head has no tree, so a small generation
 * (a one-cell generation from `createGeneration`, a bulk build's few hundred cells) keeps its time for the cells.
 */
const SPLIT_EXTRA_RESERVE_MS = 4_500;
const SPLIT_RESERVE_FINE_BYTES = 25 * 1024 * 1024;
const splitReserveMs = (m: Manifest): number => COMMIT_RESERVE_MS + (m.tree ? SPLIT_EXTRA_RESERVE_MS : Math.ceil(SPLIT_EXTRA_RESERVE_MS * Math.min(1, (m.k1 * m.k2 * m.dim * 4) / SPLIT_RESERVE_FINE_BYTES)));
/**
 * The read phase reads the deltas' cells in windows of this many (concurrently), oldest first, and plans the prefix
 * after each window to see whether the slice is full. The first window is always read: it holds the oldest delta.
 */
const DELTA_READ_WINDOW = 8;
/**
 * The slice is full once this many deltas in a row (in plan order) could not join it: its cells (`cellsPerRun`) or
 * its memory (`maxBytes`) are taken. A later delta that would still fit (all its cells in the slice, small enough)
 * waits for the next call, which starts from the deltas this one left.
 */
const SLICE_FULL_SKIPS = 16;
/** Deltas up to this size are read whole (one get, and the fold needs the bytes) instead of by two range gets. */
const DELTA_WHOLE_BYTES = 64 * 1024;
/** Share of the time before `deadline - 2.5 s` that reading more deltas may take; the cells and their tombstone sets get the rest. */
const READ_SHARE = 1 / 3;
/** Tombstone sets read at once. */
const TOMB_READ_CONCURRENCY = 8;
/** A compact.lock older than this is abandoned (the Worker's rule). */
export const COMPACT_LOCK_STALE_MS = 10 * 60 * 1000;
/**
 * Peak memory of a cell in flight (measured in Node at dim 1024, format 2, with the wasm memory growth counted; see
 * docs/update-path.md "Memory of a call"). A fold holds the old slab and its records, the merged records in the slab
 * job (whose buffers grow by doubling), the new slab in wasm and its copy in JS: `3 x old + 6 x new` covered every
 * measured fold (6.1-6.4 x new from an empty cell, 8.4 x new when the old slab was 95% of it).
 */
const FOLD_OLD_FACTOR = 3;
const FOLD_NEW_FACTOR = 6;
/**
 * A split also holds the merged slab, the re-encoded halves and their two slab jobs (8 x the merged slab, measured
 * 9.4-10.8 x with the folded deltas), plus `split_slab`'s decoded vectors (`entries x dim x 4`), counted apart.
 */
const SPLIT_FACTOR = 8;
/** Manifests, delta entry tables, JSON and small typed arrays: a constant beside the terms that scale. */
const CALL_OVERHEAD = 1024 * 1024;
/** Tombstone ids the limits leave room for when the policy names none (the profiles' `maxTombstones`). */
const LIMIT_TOMBSTONES = 50_000;

/** Estimated peak of folding a cell: `old` bytes of its current slab, `bytes` of the new one. */
const foldPeak = (old: number, bytes: number): number => FOLD_OLD_FACTOR * old + FOLD_NEW_FACTOR * bytes;
/**
 * Estimated peak of splitting a cell of `entries` entries and `bytes` bytes, taken as if the whole cell were already
 * in its slab (a split on its own, the most a later call needs for it): a cell admitted with this cost can always be
 * split again by a call with the same budget.
 */
const splitPeak = (dim: number, bytes: number, entries: number): number => Math.max(foldPeak(bytes, bytes), SPLIT_FACTOR * bytes + entries * dim * 4);

/** Bytes of one entry in a slab of this format (no payload): the entry row, the scan tier and the int8 tier. */
function entryBytesFor(m: Manifest): number {
	const dim = m.dim;
	const format = m.format ?? 1;
	if (format === 1) return 16 + dim + 4;
	if (format === 3) return 16 + Math.max(1, Math.floor(dim / 8)) + 4 + dim;
	return 16 + dim / 2 + 4 + 4 + dim;
}

/** What one `compact({ maxBytes })` call can process in a generation (`compactionLimits`, `CompactResult.limits`). */
export interface CompactionLimits {
	/** The budget the limits are for (bytes). */
	maxBytes: number;
	/** Bytes per entry the limits assume (the largest seen in the slabs and deltas, else the format's). */
	bytesPerEntry: number;
	/**
	 * The largest cell (entries) a call can split on its own. No compaction lets a cell grow past it: a fold that
	 * would is not taken, and the cell is split first. 0 when the generation cannot split (no `maxCellEntries`, or two
	 * levels).
	 */
	splitEntries: number;
	/**
	 * The largest cell a call can rewrite without splitting it. Without splits this is the limit: a delta that would
	 * take a cell past it cannot be folded (an error says so, before the cell grows).
	 */
	foldEntries: number;
	/** Cells are split once they pass this many entries: `min(maxCellEntries, splitEntries / 2)`; 0 = no splits. */
	cellEntries: number;
	/**
	 * Vectors one upsert (one delta segment) may carry and always be foldable with this budget, wherever they land:
	 * into a cell at `cellEntries`, with the split that follows. Larger upserts may still fold, or wait for a split.
	 */
	upsertVectors: number;
	/** The most cells a split commit can hold the fine means of (old, new and the tree build's copy). */
	splitCommitCells: number;
}

/** Largest n with `fixed + cost(n) <= maxBytes` (cost increasing in n), 0 when even one does not fit. */
function largest(maxBytes: number, fixed: number, cost: (n: number) => number): number {
	if (fixed + cost(1) > maxBytes) return 0;
	let lo = 1;
	let hi = 2;
	while (fixed + cost(hi) <= maxBytes && hi < 1 << 30) hi *= 2;
	while (hi - lo > 1) {
		const mid = (lo + hi) >>> 1;
		if (fixed + cost(mid) <= maxBytes) lo = mid;
		else hi = mid;
	}
	return lo;
}

function limitsFor(m: Manifest, maxBytes: number, cap: number, bytesPerEntry: number): CompactionLimits {
	const dim = m.dim;
	const B = bytesPerEntry;
	// What every call holds besides its cells, at the policy's tombstone threshold (so the limits do not shrink as
	// tombstones accumulate up to it): the sets and their unions, the slabs table and its successor, the rest.
	const reserve = CALL_OVERHEAD + 12 * (m.compaction?.maxTombstones ?? LIMIT_TOMBSTONES) + 2 * 64 * (m.k1 * m.k2 + 1);
	const canSplit = cap > 0 && m.k1 === 1;
	const foldEntries = largest(maxBytes, reserve, (n) => foldPeak(n * B, n * B));
	const splitEntries = canSplit ? largest(maxBytes, reserve, (n) => splitPeak(dim, n * B, n)) : 0;
	const cellEntries = canSplit ? Math.max(2, Math.min(cap, Math.floor(splitEntries / 2))) : 0;
	// An upsert of n vectors held whole (n x B) while it folds into a cell at cellEntries that then splits.
	const upsertVectors = canSplit ? largest(maxBytes, reserve, (n) => 2 * n * B + splitPeak(dim, (cellEntries + n) * B, cellEntries + n)) : largest(maxBytes, reserve, (n) => 2 * n * B + foldPeak(n * B, n * B));
	const splitCommitCells = Math.max(0, Math.floor((maxBytes - reserve) / (dim * 4 * 3)));
	return { maxBytes, bytesPerEntry: B, splitEntries, foldEntries, cellEntries, upsertVectors, splitCommitCells };
}

/**
 * What `compact({ maxBytes })` can process in this generation (no I/O): the cell size it splits at, the largest cell
 * it can split, and the largest upsert that is always foldable. Bytes per entry come from the manifest's deltas
 * when it has any, else from the slab format (pass `bytesPerEntry` for payloads). See `CompactionLimits`.
 */
export function compactionLimits(m: Manifest, opts: { maxBytes?: number; maxCellEntries?: number; bytesPerEntry?: number } = {}): CompactionLimits {
	const deltas = (m.deltas ?? []).filter((d) => d.count > 0);
	const seen = deltas.reduce((a, d) => Math.max(a, Math.ceil(d.bytes / d.count)), 0);
	const B = opts.bytesPerEntry ?? Math.max(entryBytesFor(m), seen);
	return limitsFor(m, opts.maxBytes ?? COMPACT_DEFAULT_MAX_BYTES, opts.maxCellEntries ?? m.compaction?.maxCellEntries ?? 0, B);
}

// ---------- the lock ----------

export const compactLockKey = (gen: string): string => `idx/${gen}/compact.lock`;
export const compactLastKey = (gen: string): string => `idx/${gen}/compact.last.json`;

interface CompactLock {
	owner: string;
	ms: number;
	at: string;
	trigger?: Record<string, unknown>;
	released?: boolean;
}

/** This isolate's runs in progress (`{prefix}|{gen}`): a stable owner must not take over a lock it is still using. */
const active = new Set<string>();

async function readLock(store: BlobStore, gen: string): Promise<{ lock: CompactLock; etag: string } | null> {
	const h = await store.head(compactLockKey(gen));
	if (!h) return null;
	const b = await store.get(compactLockKey(gen));
	if (!b) return null;
	try {
		return { lock: decodeJson<CompactLock>(b), etag: h.etag };
	} catch {
		return { lock: { owner: "?", ms: 0, at: "" }, etag: h.etag };
	}
}

/**
 * Takes `compact.lock` with a create-only put; a lock older than 10 minutes (or released without a delete, or
 * held by `owner` itself after a crash) is taken over with an etag compare-and-swap, so exactly one contender wins.
 */
async function acquireLock(store: BlobStore, gen: string, owner: string, trigger: Record<string, unknown>): Promise<{ acquired: boolean; stale: boolean; holder?: CompactLock }> {
	const now = Date.now();
	const body = JSON.stringify({ owner, ms: now, at: new Date(now).toISOString(), trigger } satisfies CompactLock);
	for (let attempt = 0; attempt < 2; attempt++) {
		if (await putIfAbsent(store, compactLockKey(gen), body, "application/json")) return { acquired: true, stale: false };
		const cur = await readLock(store, gen);
		if (!cur) continue;
		const takeover = cur.lock.released === true || cur.lock.owner === owner || now - cur.lock.ms >= COMPACT_LOCK_STALE_MS;
		if (!takeover) return { acquired: false, stale: false, holder: cur.lock };
		const ok = await writable(store).put(compactLockKey(gen), body, { ifMatch: cur.etag, contentType: "application/json" });
		return { acquired: ok !== null, stale: true, holder: cur.lock };
	}
	return { acquired: false, stale: false };
}

/** Deletes the lock if it is still ours; a store without `delete` marks it released (the Worker then sees it as stale). */
async function releaseLock(store: BlobStore, gen: string, owner: string): Promise<void> {
	const cur = await readLock(store, gen);
	if (cur?.lock.owner !== owner || cur.lock.released) return;
	if (store.delete) await store.delete(compactLockKey(gen));
	else await writable(store).put(compactLockKey(gen), JSON.stringify({ owner, ms: 0, at: new Date().toISOString(), released: true } satisfies CompactLock), { ifMatch: cur.etag, contentType: "application/json" });
}

async function recordRun(store: BlobStore, gen: string, run: Record<string, unknown>): Promise<void> {
	const b = await store.get(compactLastKey(gen));
	let prev: { recent?: unknown[] } | null = null;
	try {
		prev = b ? decodeJson<{ recent?: unknown[] }>(b) : null;
	} catch {
		prev = null;
	}
	const recent = [run, ...(prev?.recent ?? [])].slice(0, 21);
	await writable(store).put(compactLastKey(gen), JSON.stringify({ last: run, recent }), { contentType: "application/json" });
}

// ---------- inputs ----------

/** The store under the caches (compaction reads and writes whole slabs: none of them belongs in a cache). */
export function uncachedStore(store: BlobStore): BlobStore {
	let s = store;
	while (s instanceof CachedBlobStore || s instanceof WarmPrefixStore) s = s.inner;
	return s;
}

/** The whole key prefix under the caches (`PrefixedBlobStore` chain), for the in-process run guard. */
function prefixOf(store: BlobStore): string {
	const p = store as { inner?: BlobStore; prefix?: string };
	return p.inner && typeof p.prefix === "string" ? prefixOf(p.inner) + p.prefix : "";
}

interface DeltaCells {
	ref: DeltaRef;
	/** Entries per cell (> 0). */
	counts: Map<number, number>;
}

/**
 * Entries per cell of a delta segment from its entry table: two small range reads, or with `whole` one read of the
 * whole segment, which is then returned too (a small delta costs one get, and its fold needs the bytes anyway).
 */
async function readDeltaCells(store: BlobStore, ref: DeltaRef, whole = false): Promise<DeltaCells & { bytes?: Uint8Array }> {
	const all = whole ? await store.get(ref.key) : null;
	const hb = all ?? (await store.get(ref.key, { offset: 0, length: 64 }));
	if (!hb) throw new ThreadError(`missing ${ref.key}`, 404);
	const h = parseDeltaHeader(hb);
	const counts = new Map<number, number>();
	if (h.count > 0) {
		const len = h.count * DELTA_ENTRY_BYTES;
		const t = all ? all.subarray(h.entriesOff, h.entriesOff + len) : await store.get(ref.key, { offset: h.entriesOff, length: len });
		if (!t || t.byteLength < len) throw new ThreadError(`missing ${ref.key}`, 404);
		const dv = new DataView(t.buffer, t.byteOffset, t.byteLength);
		for (let i = 0; i < h.count; i++) {
			const cell = dv.getUint32(i * DELTA_ENTRY_BYTES + 4, true);
			counts.set(cell, (counts.get(cell) ?? 0) + 1);
		}
	}
	return all ? { ref, counts, bytes: all } : { ref, counts };
}

/**
 * The ids of tombstone sets, read when first asked for and kept for the call (a commit retried on a newer head asks
 * again). `reads.tombSets` counts the sets read.
 */
class TombIds {
	private readonly ids = new Map<string, Promise<Uint32Array>>();
	constructor(
		private readonly store: BlobStore,
		private readonly reads: CompactReads,
	) {}
	of(t: { key: string }): Promise<Uint32Array> {
		let p = this.ids.get(t.key);
		if (!p) {
			this.reads.tombSets++;
			this.ids.set(t.key, (p = getOrThrow(this.store, t.key).then(u32View)));
		}
		return p;
	}
	/** Sorted unique union of the sets' ids (the `filter_hits` buffer `tombFor` / `tombForDelta` build). */
	async union(sets: { key: string }[]): Promise<Uint8Array> {
		return idUnion(await mapLimit(sets, TOMB_READ_CONCURRENCY, (t) => this.of(t)));
	}
}

async function getOrThrow(store: BlobStore, key: string): Promise<Uint8Array> {
	const b = await store.get(key);
	if (!b) throw new ThreadError(`missing ${key}`, 404);
	return b;
}

const slabRevOf = (row: SlabRow | undefined): number => (row && row.length > 4 ? (row[4] as number) : 0);

/**
 * Whether a tombstone set can leave the manifest (the Worker's `tombDroppable`), per side: the dense side holds it
 * while a remaining delta predates it, a slab written at a rev in (0, set.rev] remains, or one of its ids is a
 * build id; the lexical side while a lexical/docs segment predates it. A scoped set needs only its own side clear,
 * an unscoped one both.
 */
export function tombDroppable(t: { rev: number; ids: Uint32Array; scope?: TombScope }, deltas: { addedRev: number }[], slabs: SlabRow[], baseIds: number, segments: { addedRev: number }[] = []): boolean {
	const d = droppableByRefs(t, dropFloors(deltas, slabs, segments), baseIds);
	if (d !== "ids") return d;
	for (const id of t.ids) if (id < baseIds) return false;
	return true;
}

/** What `tombDroppable` compares a set's rev with: the oldest remaining delta and segment, the oldest slab rev > 0. */
interface DropFloors {
	delta: number;
	segment: number;
	slab: number;
}

function dropFloors(deltas: { addedRev: number }[], slabs: SlabRow[], segments: { addedRev: number }[]): DropFloors {
	let slab = Infinity;
	for (const row of slabs) if (slabRevOf(row) > 0) slab = Math.min(slab, slabRevOf(row));
	return { delta: deltas.reduce((a, d) => Math.min(a, d.addedRev), Infinity), segment: segments.reduce((a, s) => Math.min(a, s.addedRev), Infinity), slab };
}

/**
 * `tombDroppable` without the ids: true or false when the rest decides, `"ids"` when only the build-id test is
 * left. A lexical-scoped set never needs its ids, and a dense-side one needs them only when the base has build ids.
 */
function droppableByRefs(t: { rev: number; scope?: TombScope }, f: DropFloors, baseIds: number): boolean | "ids" {
	if (hidesLexical(t) && f.segment < t.rev) return false;
	if (!hidesDense(t)) return true;
	if (f.delta < t.rev) return false;
	if (f.slab <= t.rev) return false;
	return baseIds > 0 ? "ids" : true;
}

/** `tombDroppable` for each set, reading the ids only of the sets whose drop depends on the build-id test. */
async function droppableSets(sets: TombRef[], deltas: { addedRev: number }[], slabs: SlabRow[], baseIds: number, segments: { addedRev: number }[], ids: TombIds): Promise<boolean[]> {
	const f = dropFloors(deltas, slabs, segments);
	const pre = sets.map((t) => droppableByRefs(t, f, baseIds));
	const need = sets.filter((_, i) => pre[i] === "ids");
	const got = new Map(await mapLimit(need, TOMB_READ_CONCURRENCY, async (t) => [t.key, await ids.of(t)] as const));
	return pre.map((p, i) => (p === "ids" ? !got.get(sets[i].key)!.some((id) => id < baseIds) : p));
}

// ---------- plan ----------

interface Plan {
	cells: number[];
	folded: DeltaRef[];
	skipped: DeltaRef[];
	purgeCells: number;
	/** [cell, newCell], new cells numbered k, k + 1, ... in cell order. */
	splits: [number, number][];
	splitsDeferred: number[];
	/**
	 * Split cells that keep their mean (an unfolded delta has entries in them, encoded against it): the new cell
	 * takes the 2-means centre farther from it and the entries nearer that centre.
	 */
	pinned: Set<number>;
	/** Oversized cells that cannot split (a two-level generation, k1 > 1). */
	splitBlocked: number[];
	/** Cells split on their own (no delta folded into them) because they kept a delta from folding. */
	presplit: number[];
	concurrency: number;
	estimatedPeakBytes: number;
	/** The cells each folded delta has entries in. */
	deltaCells: Map<string, number[]>;
	limits: CompactionLimits;
	/** Input these budgets can never process (a delta too large, a cell past what they can split): why. */
	impossible: string[];
	note?: string;
}

interface Base {
	gen: string;
	baseRev: number;
	manifest: Manifest;
	files: IndexFiles;
	slabs: SlabRow[];
	slabsBytes: number;
	/** The deltas whose cells were read: the oldest of the manifest's, in plan order (`addedRev`, then `seq`). */
	dcells: DeltaCells[];
	/** Deltas of the base not read (all after `dcells`): they wait, and may have entries in any cell. */
	unread: number;
	/** The rev of the newest dense-side tombstone set without build ids (purge rewrites the slabs up to it); 0: none. */
	purgeMaxRev: number;
}

const MB = (b: number): string => `${(b / 1048576).toFixed(1)} MiB`;

/**
 * The slice: whole deltas, oldest first, while their cells fit in `cellsPerRun` and the estimated peak fits in
 * `maxBytes`; then the cells that keep a delta from folding, split on their own; then (purge) the cells that keep
 * tombstone sets alive and (force) oversized cells, while the budgets allow; then the splits of the rewritten cells
 * over the cap.
 *
 * The guarantee (docs/update-path.md "Memory of a call"): a cell over `limits.cellEntries` is split in the call that
 * makes it so, and a fold is only taken when the cell it makes could still be split on its own by a call with the
 * same budget (`limits.splitEntries`). So no cell grows past what a later call can process; a delta that would take
 * a cell past it waits while that cell is split first. Input that no call with this budget can take (a delta whose
 * entries in one cell pass `splitEntries`, a cell past it from an older writer) is reported in `impossible`.
 *
 * The plan of the deltas read so far is the plan of the whole backlog cut there: a delta's place in the slice
 * depends only on the deltas before it, and nothing else the planner uses comes from the deltas' cells (the bytes
 * per entry come from the manifest). The deltas not read are treated as waiting with entries anywhere: a split cell
 * keeps its mean, and `force` takes no oversized cell on its own.
 */
function planSlice(b: Base, o: { cellsPerRun: number; maxBytes: number; concurrency: number; cap: number; splitCellsPerRun: number; purge: boolean; force: boolean }): Plan {
	const m = b.manifest;
	const dim = m.dim;
	const k = m.k1 * m.k2;
	const canSplit = o.cap > 0 && m.k1 === 1;
	// The ids over all sets, from the manifest (`TombRef.count`): the sets are read only when a cell needs them.
	const tombIds = (m.tombstones ?? []).reduce((a, t) => a + t.count, 0);
	// Held for the whole run: tombstone sets (and their unions), the slabs table (and the new one), the rest. The fine
	// means of a split commit (old, new, the tree build's copy) are held only once the cells are done: a phase apart.
	const fixed = CALL_OVERHEAD + tombIds * 4 * 3 + b.slabsBytes * 2;
	const commitPeak = (splits: number) => fixed + (splits > 0 ? (k + splits) * dim * 4 * 3 : 0);
	// Bytes per entry: the largest seen in a non-empty slab or a delta (payloads make entries larger), else the format's.
	let B = entryBytesFor(m);
	for (const row of b.slabs) if (row[2] > 0) B = Math.max(B, Math.ceil(row[3] / row[2]));
	for (const d of m.deltas ?? []) if (d.count > 0) B = Math.max(B, Math.ceil(d.bytes / d.count));
	const limits = limitsFor(m, o.maxBytes, o.cap, B);
	const capEff = limits.cellEntries;
	interface Est {
		/** Bytes of the cell's current slab. */
		old: number;
		bytes: number;
		entries: number;
		/** Entries of the cell's current slab. */
		was: number;
	}
	const est = new Map<number, Est>();
	const baseOf = (c: number): Est => ({ old: b.slabs[c]?.[3] ?? 0, bytes: b.slabs[c]?.[3] ?? 0, entries: b.slabs[c]?.[2] ?? 0, was: b.slabs[c]?.[2] ?? 0 });
	const cellOf = (c: number) => est.get(c) ?? baseOf(c);
	const splits = (e: Est) => canSplit && e.entries > capEff;
	// A cell this call would leave past what a call with this budget can always process (splitEntries, or without
	// splits foldEntries) is not taken, even when this call's own (smaller) state would hold it.
	const past = (e: Est) => e.entries > e.was && e.entries > (canSplit ? limits.splitEntries : limits.foldEntries);
	const peak = (e: Est) => (past(e) ? Infinity : splits(e) ? splitPeak(dim, e.bytes, e.entries) : foldPeak(e.old, e.bytes));
	let held = 0;
	let maxDelta = 0;
	let maxPeak = 0;
	const need = (h: number, md: number, mp: number) => fixed + h + md + mp;
	const folded: DeltaRef[] = [];
	const skipped: DeltaRef[] = [];
	const deltaCells = new Map<string, number[]>();
	const impossible: string[] = [];
	/** Cells that alone keep a skipped delta from folding (their entries plus the delta's do not fit). */
	const blockers = new Map<number, number>();
	const order = [...b.dcells].sort((x, y) => x.ref.addedRev - y.ref.addedRev || x.ref.seq - y.ref.seq);
	for (const d of order) {
		const own = [...d.counts.keys()];
		const per = d.ref.bytes / Math.max(1, d.ref.count);
		if (own.length > o.cellsPerRun) {
			impossible.push(`delta ${d.ref.key} has entries in ${own.length} cells, more than cellsPerRun ${o.cellsPerRun} (raise it, or compact with the Worker)`);
			skipped.push(d.ref);
			continue;
		}
		// Can this delta ever fold with this budget? Only if each of its cells could take it starting from nothing.
		const alone = own.find((c) => {
			const n = d.counts.get(c)!;
			const e = { old: 0, bytes: n * per, entries: n, was: 0 };
			return need(d.ref.bytes, d.ref.bytes, peak(e)) > o.maxBytes;
		});
		if (alone !== undefined) {
			const n = d.counts.get(alone)!;
			impossible.push(`delta ${d.ref.key} puts ${n} entries (${MB(n * per)}) into cell ${alone}, more than one call with maxBytes ${MB(o.maxBytes)} can fold (${canSplit ? `cells split at ${capEff} entries; an upsert of up to ${limits.upsertVectors} vectors always folds` : `${limits.foldEntries} entries per cell without splits`}): raise maxBytes, or compact with the Worker`);
			skipped.push(d.ref);
			continue;
		}
		const add = own.filter((c) => !est.has(c));
		if (est.size + add.length > o.cellsPerRun) {
			skipped.push(d.ref);
			continue;
		}
		const next = new Map<number, Est>();
		let mp = maxPeak;
		for (const c of own) {
			const e = cellOf(c);
			const n = d.counts.get(c)!;
			const ne = { old: e.old, bytes: e.bytes + n * per, entries: e.entries + n, was: e.was };
			next.set(c, ne);
			mp = Math.max(mp, peak(ne));
		}
		if (need(held + d.ref.bytes, Math.max(maxDelta, d.ref.bytes), mp) > o.maxBytes) {
			skipped.push(d.ref);
			// A cell that cannot take this delta even in a slice of its own must shrink first.
			for (const [c, ne] of next) if (need(d.ref.bytes, d.ref.bytes, peak(ne)) > o.maxBytes) blockers.set(c, Math.max(blockers.get(c) ?? 0, d.counts.get(c)!));
			continue;
		}
		for (const [c, e] of next) est.set(c, e);
		held += d.ref.bytes;
		maxDelta = Math.max(maxDelta, d.ref.bytes);
		maxPeak = mp;
		folded.push(d.ref);
		deltaCells.set(d.ref.key, own);
	}
	// Extra cells without deltas, while the budgets allow: `cost` is the cell's peak when rewritten.
	const extra = (cells: number[], cost: (e: Est) => number = peak): number[] => {
		const took: number[] = [];
		for (const c of cells) {
			if (est.has(c) || est.size >= o.cellsPerRun) continue;
			const e = cellOf(c);
			const mp = Math.max(maxPeak, cost(e));
			if (need(held, maxDelta, mp) > o.maxBytes) continue;
			est.set(c, e);
			maxPeak = mp;
			took.push(c);
		}
		return took;
	};
	// Cells that keep a delta from folding are split on their own (their mean pinned: the delta was encoded against
	// it), largest first; one the budget cannot split is past what these budgets can process.
	const presplit: number[] = [];
	if (blockers.size) {
		const cand = [...blockers.keys()].filter((c) => !est.has(c)).sort((x, y) => cellOf(y).entries - cellOf(x).entries || x - y);
		for (const c of cand) {
			const e = cellOf(c);
			if (!canSplit || e.entries < 2 || fixed + splitPeak(dim, e.bytes, e.entries) > o.maxBytes) {
				if (!canSplit) impossible.push(`cell ${c} holds ${e.entries} entries (${MB(e.bytes)}): folding ${blockers.get(c)} more needs more than maxBytes ${MB(o.maxBytes)} allows without splits (${limits.foldEntries} entries per cell): raise maxBytes, set maxCellEntries, or rebuild the generation`);
				else if (e.entries >= 2) impossible.push(`cell ${c} holds ${e.entries} entries (${MB(e.bytes)}); splitting it needs about ${MB(fixed + splitPeak(dim, e.bytes, e.entries))}, more than maxBytes ${MB(o.maxBytes)} (this budget splits cells of up to ${limits.splitEntries} entries): run one compaction with a larger maxBytes (or the Worker's), or rebuild the generation`);
				continue;
			}
			presplit.push(...extra([c], (x) => splitPeak(dim, x.bytes, x.entries)));
		}
	}
	let purgeCells = 0;
	if (o.purge && b.purgeMaxRev > 0) {
		// Only dense-side sets wait for slabs; of those, only sets without build ids can ever be dropped (`purgeMaxRev`).
		const maxRev = b.purgeMaxRev;
		const blocking = b.slabs
			.map((row, cell) => ({ cell, r: slabRevOf(row) }))
			.filter(({ r }) => r > 0 && r <= maxRev)
			.sort((x, y) => x.r - y.r || x.cell - y.cell)
			.map((x) => x.cell);
		purgeCells = extra(blocking).length;
	}
	// Not while deltas are unread: any of them may have entries in an oversized cell.
	if (o.force && canSplit && !b.unread) {
		const unfolded = new Set(b.dcells.filter((d) => !deltaCells.has(d.ref.key)).flatMap((d) => [...d.counts.keys()]));
		const over = b.slabs
			.map((row, cell) => ({ cell, n: row[2] }))
			.filter(({ cell, n }) => n > capEff && !unfolded.has(cell))
			.sort((x, y) => y.n - x.n || x.cell - y.cell)
			.map((x) => x.cell);
		extra(over);
	}
	let cells = [...est.keys()].sort((x, y) => x - y);
	// Splits (the Worker's planSplits), except that a cell an unfolded delta has entries in still splits, with its
	// mean pinned (those entries were encoded against it); the Worker leaves such a cell unsplit. Without that, a
	// backlog larger than the budget would keep a cell from ever splitting. No splits in a two-level generation.
	const splitList: [number, number][] = [];
	const pinned = new Set<number>();
	const candidates: number[] = [];
	const splitBlocked: number[] = [];
	const pre = new Set(presplit);
	if (o.cap > 0) {
		const foldedKeys = new Set(folded.map((d) => d.key));
		let nextCell = k;
		for (const cell of cells) {
			const e = est.get(cell)!;
			if (!(e.entries > capEff || (pre.has(cell) && e.entries >= 2))) continue;
			if (m.k1 !== 1) {
				splitBlocked.push(cell);
				continue;
			}
			// An unread delta may have entries in the cell: with any unread, every split keeps its mean.
			if (b.unread > 0 || b.dcells.some((d) => !foldedKeys.has(d.ref.key) && d.counts.has(cell))) pinned.add(cell);
			splitList.push([cell, nextCell++]);
			candidates.push(cell);
		}
	}
	let deferred = splitList.length && cells.length > o.splitCellsPerRun ? candidates : [];
	// The split commit holds the fine means three times: with too many cells for the budget, no split can commit.
	if (splitList.length && !deferred.length && commitPeak(splitList.length) > o.maxBytes) {
		impossible.push(`a split commit at ${k} cells holds about ${MB(commitPeak(splitList.length))} of fine means, more than maxBytes ${MB(o.maxBytes)} (${limits.splitCommitCells} cells at most): raise maxBytes, or rebuild the generation`);
		deferred = candidates;
	}
	if (deferred.length) {
		// Without the splits, the cells taken only to be split would be rewritten unchanged.
		cells = cells.filter((c) => !pre.has(c));
		for (const c of pre) est.delete(c);
	}
	const free = o.maxBytes - need(held, maxDelta, 0);
	const concurrency = cells.length ? Math.max(1, Math.min(o.concurrency, cells.length, maxPeak > 0 ? Math.floor(free / maxPeak) : o.concurrency)) : 0;
	const doneSplits = deferred.length ? [] : splitList;
	const plan: Plan = {
		cells,
		folded,
		skipped,
		purgeCells,
		splits: doneSplits,
		pinned: deferred.length ? new Set() : pinned,
		splitsDeferred: deferred,
		splitBlocked,
		presplit: deferred.length ? [] : presplit,
		concurrency,
		estimatedPeakBytes: cells.length ? Math.max(need(held, maxDelta, concurrency * maxPeak), commitPeak(doneSplits.length)) : fixed,
		deltaCells,
		limits,
		impossible,
	};
	if (impossible.length) plan.note = impossible[0];
	return plan;
}

// ---------- the call ----------

/**
 * One bounded dense compaction of the generation's head (see the module comment and `CompactOptions`). Safe to call
 * again at any time, also after a crash: a run commits through the manifest create-only put, so it either lands as
 * one new revision or not at all; the slab objects it writes are keyed by the base revision and written
 * create-only, so a rerun on the same base reuses identical bytes and never overwrites an object a revision lists.
 */
export async function compactDeltas(ctx: WriterContext, opts: CompactOptions = {}): Promise<CompactResult> {
	const t0 = Date.now();
	const { gen } = ctx;
	const w = ctx.w as CompactWasm;
	writable(ctx.store);
	// Everything below reads and writes under the caches: whole slabs and deltas would only fill them.
	const store = uncachedStore(ctx.store);
	const rctx: WriterContext = { ...ctx, store };
	const owner = opts.owner ?? crypto.randomUUID();
	const guard = `${prefixOf(store)}|${gen}`;
	const result: CompactResult = { gen, outcome: "noop", rev: -1, baseRev: null, cells: 0, folded: 0, remainingDeltas: 0, skippedDeltas: 0, reads: { deltas: 0, deltasListed: 0, tombSets: 0, tombSetsListed: 0, ms: 0 }, purgeCells: 0, splits: [], splitsDeferred: [], presplit: [], k: 0, droppedTombstoneSets: [], readBytes: 0, writtenBytes: 0, estimatedPeakBytes: 0, concurrency: 0, reasons: [], more: false, conflicts: 0, currentUpdated: false, staleLockTaken: false, ms: 0 };
	const finish = () => ((result.ms = Date.now() - t0), result);
	if (active.has(guard)) {
		result.outcome = "locked";
		result.more = true;
		result.note = "a compaction of this generation started by this isolate is in progress";
		result.rev = await latestRev(store, gen);
		return finish();
	}
	active.add(guard);
	let locked = false;
	try {
		const headRev = await latestRev(store, gen);
		const headM = parseManifest(await getOrThrow(store, keys.manifest(gen, headRev)), gen, headRev);
		result.rev = headRev;
		result.k = headM.k1 * headM.k2;
		result.reasons = compactionExceeded(headM);
		result.remainingDeltas = headM.deltas?.length ?? 0;
		if (!opts.force && !result.reasons.length) {
			result.note = headM.compaction ? `head rev ${headRev} is below every threshold` : `head rev ${headRev} has no compaction policy (pass force)`;
			return finish();
		}
		if (opts.lock !== false) {
			const trigger = { rev: headRev, op: "sdk-compact", reasons: result.reasons, pressure: compactionPressure(headM), ms: t0 };
			const l = await acquireLock(store, gen, owner, trigger);
			result.staleLockTaken = l.stale;
			if (!l.acquired) {
				result.outcome = "locked";
				result.more = true;
				result.note = `compact.lock held by ${l.holder?.owner ?? "?"} since ${l.holder?.at ?? "?"}`;
				return finish();
			}
			locked = true;
		}
		let run: Record<string, unknown>;
		try {
			run = await ctx.thread.withRecovery("compact", () => runCompaction(rctx, w, opts, result, t0));
		} catch (e) {
			const conflict = e instanceof ThreadError && e.status === 409;
			run = { trigger: { rev: headRev, op: "sdk-compact", reasons: result.reasons }, outcome: conflict ? "conflict" : "error", ms: Date.now() - t0, baseRev: result.baseRev, error: e instanceof Error ? e.message : String(e) };
			if (!conflict) throw e;
			result.outcome = "conflict";
			result.more = true;
			result.note = e instanceof Error ? e.message : String(e);
		} finally {
			if (opts.record !== false) await recordRun(store, gen, { owner, writer: "thread-sdk", startedAt: new Date(t0).toISOString(), finishedAt: new Date().toISOString(), ...run! }).catch((e) => console.error(`thread-sdk: compact ${gen}: recording the run failed: ${String(e)}`));
		}
		return finish();
	} finally {
		active.delete(guard);
		if (locked) await releaseLock(store, gen, owner).catch((e) => console.error(`thread-sdk: compact ${gen}: releasing the lock failed: ${String(e)}`));
	}
}

/** Plan, cells and commit on the head (inside the lock). Fills `result`; returns what `compact.last.json` records. */
async function runCompaction(ctx: WriterContext, w: CompactWasm, opts: CompactOptions, result: CompactResult, t0: number): Promise<Record<string, unknown>> {
	const { store, gen, thread } = ctx;
	const epoch = thread.epoch;
	const free = (o: { free(): void }) => {
		if (epoch === thread.epoch)
			try {
				o.free();
			} catch {
				/* poisoned instance */
			}
	};
	// ---- base ---- (a retry after a wasm trap starts over)
	Object.assign(result, { readBytes: 0, writtenBytes: 0, splits: [], cells: 0, folded: 0, droppedTombstoneSets: [], reads: { deltas: 0, deltasListed: 0, tombSets: 0, tombSetsListed: 0, ms: 0 } });
	const baseRev = await latestRev(store, gen);
	const manifest = parseManifest(await getOrThrow(store, keys.manifest(gen, baseRev)), gen, baseRev);
	const files = manifestFiles(manifest);
	const policy = manifest.compaction;
	result.baseRev = baseRev;
	result.rev = baseRev;
	result.k = manifest.k1 * manifest.k2;
	result.reasons = compactionExceeded(manifest);
	result.remainingDeltas = manifest.deltas?.length ?? 0;
	const tombRefs = manifest.tombstones ?? [];
	const reads = result.reads;
	const tombIds = new TombIds(store, reads);
	reads.deltasListed = manifest.deltas?.length ?? 0;
	reads.tombSetsListed = tombRefs.length;
	const record = (extra: Record<string, unknown> = {}) => ({ trigger: { rev: baseRev, op: "sdk-compact", reasons: result.reasons }, outcome: result.outcome, ms: Date.now() - t0, baseRev, rev: result.rev, cells: result.cells, folded: result.folded, purgeCells: result.purgeCells, skippedDeltas: result.skippedDeltas, splits: result.splits, splitsDeferred: result.splitsDeferred, k: result.k, remainingDeltas: result.remainingDeltas, droppedTombstoneSets: result.droppedTombstoneSets.length, readMB: +(result.readBytes / 1e6).toFixed(1), writtenMB: +(result.writtenBytes / 1e6).toFixed(1), slots: 1, estimatedPeakMB: +(result.estimatedPeakBytes / 1e6).toFixed(1), reads: { ...reads }, ...(result.note ? { note: result.note } : {}), ...extra });
	if (!opts.force && !result.reasons.length) {
		result.outcome = "noop";
		result.note = `head rev ${baseRev} is below every threshold`;
		return record();
	}
	if (!(manifest.k2 > 0 && manifest.dim > 0)) {
		result.outcome = "noop";
		result.note = "this generation has no dense index";
		return record();
	}
	if (!files.slabs) throw new ThreadError(`${gen} has no slabs table`, 400);
	if ((manifest.format ?? 1) < 2) throw new ThreadError("compaction needs a format 2 or 3 generation", 400);
	const maxBytes = opts.maxBytes ?? COMPACT_DEFAULT_MAX_BYTES;
	if (!(maxBytes > 0)) throw new ThreadError("maxBytes must be positive", 400);
	const cap = opts.maxCellEntries ?? policy?.maxCellEntries ?? 0;
	if (!Number.isInteger(cap) || cap < 0 || cap === 1) throw new ThreadError("maxCellEntries must be 0 (no splits) or an integer >= 2", 400);
	const k = manifest.k1 * manifest.k2;
	const dim = manifest.dim;
	const format = manifest.format ?? 1;
	const slabRev = baseRev + 1;
	const deadlineMs = opts.deadlineMs ?? policy?.deadlineMs ?? DEFAULT_DEADLINE_MS;
	const deadline = t0 + deadlineMs;
	// Reading more deltas stops here; the first window is read whatever the time (it holds the oldest delta).
	const readUntil = t0 + READ_SHARE * Math.max(0, deadlineMs - COMMIT_RESERVE_MS);
	const slabsBytes = await getOrThrow(store, files.slabs);
	const slabs = parseSlabsTable(slabsBytes, k, files.slabs);
	const tombThreshold = result.reasons.some((r) => r.startsWith("tombstones"));
	const purge = opts.purge ?? (tombThreshold || opts.force === true);
	const planOpts = {
		cellsPerRun: opts.cellsPerRun ?? policy?.cellsPerRun ?? DEFAULT_CELLS_PER_RUN,
		maxBytes,
		concurrency: Math.max(1, opts.concurrency ?? DEFAULT_CONCURRENCY),
		cap,
		splitCellsPerRun: opts.splitCellsPerRun ?? policy?.splitCellsPerRun ?? DEFAULT_SPLIT_CELLS_PER_RUN,
		purge,
		force: opts.force === true,
	};

	// ---- read phase ----
	// The deltas' cells, oldest first (the plan's order), a window at a time, until the slice is full or the read phase
	// has used its share of the deadline: the plan of a prefix is the plan of the backlog cut there (see planSlice).
	// Small deltas are read whole; the bytes of those the slice takes are kept for the fold, the others dropped.
	const order = [...(manifest.deltas ?? [])].sort((x, y) => x.addedRev - y.addedRev || x.seq - y.seq);
	const dcells: DeltaCells[] = [];
	const wholeDeltas = new Map<string, Uint8Array>();
	const plan = (purgeMaxRev: number) => planSlice({ gen, baseRev, manifest, files, slabs, slabsBytes: slabsBytes.byteLength, dcells, unread: order.length - dcells.length, purgeMaxRev }, planOpts);
	for (;;) {
		const win = order.slice(dcells.length, dcells.length + DELTA_READ_WINDOW);
		for (const d of await mapLimit(win, DELTA_READ_WINDOW, (d) => readDeltaCells(store, d, d.bytes <= DELTA_WHOLE_BYTES))) {
			dcells.push({ ref: d.ref, counts: d.counts });
			if (d.bytes) {
				wholeDeltas.set(d.ref.key, d.bytes);
				result.readBytes += d.bytes.byteLength;
			}
		}
		if (dcells.length === order.length || Date.now() >= readUntil) break;
		const skipped = new Set(plan(0).skipped.map((d) => d.key));
		for (const key of skipped) wholeDeltas.delete(key);
		let run = 0;
		while (run < dcells.length && skipped.has(dcells[dcells.length - 1 - run].ref.key)) run++;
		if (run >= SLICE_FULL_SKIPS) break;
	}
	reads.deltas = dcells.length;
	// Purge rewrites the slabs up to the newest dense-side set without build ids. With build ids in the base the sets
	// are read newest first until one has none (while the read phase has time; fewer purge cells otherwise).
	let purgeMaxRev = 0;
	if (purge) {
		const dense = tombRefs.filter(hidesDense).sort((x, y) => y.rev - x.rev);
		if (!(manifest.vectors > 0)) purgeMaxRev = dense[0]?.rev ?? 0;
		else
			for (let i = 0; i < dense.length && !purgeMaxRev && Date.now() < readUntil; i += TOMB_READ_CONCURRENCY) {
				const win = dense.slice(i, i + TOMB_READ_CONCURRENCY);
				const ids = await mapLimit(win, TOMB_READ_CONCURRENCY, (t) => tombIds.of(t));
				const j = ids.findIndex((x) => !x.some((id) => id < manifest.vectors));
				if (j >= 0) purgeMaxRev = win[j].rev;
			}
	}
	const slice = plan(purgeMaxRev);
	for (const key of [...wholeDeltas.keys()]) if (!slice.deltaCells.has(key)) wholeDeltas.delete(key);
	reads.ms = Date.now() - t0;
	const unread = order.length - dcells.length;
	const waiting = slice.skipped.length + unread;
	const readSummary = `${reads.ms} ms for ${dcells.length} of ${order.length} deltas and ${reads.tombSets} of ${tombRefs.length} tombstone sets (deadlineMs ${deadlineMs})`;
	Object.assign(result, { skippedDeltas: slice.skipped.length, purgeCells: slice.purgeCells, splitsDeferred: slice.splitsDeferred, presplit: slice.presplit, estimatedPeakBytes: slice.estimatedPeakBytes, concurrency: slice.concurrency, limits: slice.limits });
	if (!slice.cells.length) {
		if (slice.impossible.length) {
			const more = slice.impossible.length > 1 ? ` (and ${slice.impossible.length - 1} more)` : "";
			throw new ThreadError(`compaction of ${gen} rev ${baseRev} cannot proceed with these budgets: ${slice.impossible[0]}${more}`, 413);
		}
		result.outcome = waiting ? "blocked" : "noop";
		result.more = waiting > 0;
		result.note = slice.note ?? (waiting ? `${waiting} deltas wait (${unread ? `${unread} not read; ` : ""}splits deferred: the slice has more than ${planOpts.splitCellsPerRun} cells)` : undefined) ?? (tombRefs.length ? "nothing to rewrite (tombstone sets held by build ids, lexical segments or no blocking slabs)" : "nothing to rewrite");
		// Tombstone sets may still be droppable without rewriting anything (e.g. after a segment compaction).
		if (!waiting && (await droppableSets(tombRefs, manifest.deltas ?? [], slabs, manifest.vectors, manifest.segments ?? [], tombIds)).some(Boolean)) {
			await commitSlice(ctx, w, { baseRev, manifest, files, slabs, k, dim, tombIds, known: dcells, slabRev, rows: [], folded: [], splits: [], means: new Map(), deltaCells: slice.deltaCells, cap, pinned: new Set() }, result);
			result.outcome = "committed";
			result.note = "dropped tombstone sets nothing references";
		}
		return record();
	}

	// ---- cells ----
	const cellDeadline = deadline - (slice.splits.length ? splitReserveMs(manifest) : COMMIT_RESERVE_MS);
	const splitOf = new Map(slice.splits);
	// The folded deltas, in manifest order (the order the Worker adds their records in: same bytes).
	const foldedKeys = new Set(slice.folded.map((d) => d.key));
	const foldedRefs = (manifest.deltas ?? []).filter((d) => foldedKeys.has(d.key));
	const cellDeltas = new Map<number, DeltaRef[]>();
	for (const d of foldedRefs) for (const c of slice.deltaCells.get(d.key)!) (cellDeltas.get(c) ?? cellDeltas.set(c, []).get(c)!).push(d);
	// The progress guarantee: the cells the oldest delta needs to commit run first and whatever the deadline. They are
	// its own cells (or, when a cell keeps it out, that cell's split), with every folded delta in them and that
	// delta's cells, and every split numbered before one of them (new cells are numbered in order), to the fixpoint:
	// exactly what the commit of a cut slice (below) accepts.
	const must = new Set<number>();
	if (order.length) {
		const queue: number[] = [];
		const take = (c: number) => {
			if (!must.has(c) && slice.cells.includes(c)) (must.add(c), queue.push(c));
		};
		(slice.deltaCells.get(order[0].key) ?? slice.presplit.filter((c) => dcells[0].counts.has(c))).forEach(take);
		while (queue.length) {
			const c = queue.pop()!;
			for (const d of cellDeltas.get(c) ?? []) slice.deltaCells.get(d.key)!.forEach(take);
			const i = slice.splits.findIndex(([x]) => x === c);
			for (let j = 0; j < i; j++) take(slice.splits[j][0]);
		}
	}
	// The read phase used the whole time: only the cells the oldest delta needs run; with none, nothing does.
	const readNote = Date.now() > cellDeadline ? `read phase passed the deadline: ${readSummary}` : null;
	if (readNote && !must.size) {
		result.outcome = "deadline";
		result.more = true;
		result.note = `${readNote}; no cell started`;
		return record();
	}
	const codebook = slice.splits.length && format === 3 ? await getOrThrow(store, files.pq!) : new Uint8Array(0);
	// The folded deltas' bytes: the small ones were read whole above, the others are read when a cell needs them.
	const deltaBytes = new Map<string, Promise<Uint8Array>>([...wholeDeltas].map(([key, b]) => [key, Promise.resolve(b)]));
	wholeDeltas.clear();
	const deltaOf = (d: DeltaRef): Promise<Uint8Array> => {
		let p = deltaBytes.get(d.key);
		if (!p) deltaBytes.set(d.key, (p = getOrThrow(store, d.key).then((b) => ((result.readBytes += b.byteLength), b))));
		return p;
	};
	// The filters (tombFor / tombForDelta over the dense view: the lexical-scoped sets are never read here), each
	// union built once and each set read once, when the first cell that needs it runs.
	const unions = new Map<string, Promise<Uint8Array>>();
	const unionOf = (key: string, pick: (t: TombRef) => boolean) => {
		let u = unions.get(key);
		if (!u) unions.set(key, (u = tombIds.union(tombRefs.filter((t) => hidesDense(t) && pick(t)))));
		return u;
	};
	const slabFilter = (rev: number) => unionOf(`slab ${rev}`, (t) => t.rev >= rev);
	const deltaFilter = (addedRev: number) => unionOf(`delta ${addedRev}`, (t) => t.rev > addedRev);
	const rows = new Map<number, SlabRow>();
	const means = new Map<number, Float32Array>();
	const done = new Set<number>();
	const splitsDone: SplitDone[] = [];
	let stopped = false;
	const writeSlab = async (cell: number, mean: Float32Array, records: Uint8Array[]): Promise<number> => {
		const job = new w.SlabJob(dim, cell, mean, format);
		let out: Uint8Array;
		let n: number;
		let finished = false;
		try {
			for (const r of records) job.add_records(r);
			n = job.len();
			finished = true;
			out = job.finish();
		} finally {
			if (!finished) free(job);
		}
		const oi = w.slab_info(out);
		await putSlab(store, gen, baseRev, files, keys.slab(gen, cell, slabRev), out);
		result.writtenBytes += out.byteLength;
		rows.set(cell, [oi[5], oi[6], n, out.byteLength, slabRev]);
		return n;
	};
	const doCell = async (cell: number) => {
		const row = slabs[cell];
		const oldRev = slabRevOf(row);
		if (oldRev >= slabRev) throw new ThreadError(`cell ${cell} already has slabRev ${oldRev} >= ${slabRev}`, 409);
		const oldKey = keys.slab(gen, cell, oldRev);
		let old: Uint8Array | null = await getOrThrow(store, oldKey);
		result.readBytes += old.byteLength;
		const info = w.slab_info(old);
		if (info[2] !== cell) throw new ThreadError(`slab ${oldKey} is for cell ${info[2]}`, 500);
		const mean = w.slab_centroid(old);
		// An empty slab has nothing to filter (a rev-0 cell is hidden by every set, which it then need not read).
		const records = [w.slab_to_records(old, info[1] > 0 ? await slabFilter(oldRev) : NO_TOMBSTONES)];
		old = null;
		const ds = cellDeltas.get(cell) ?? [];
		const inputs = await Promise.all(ds.map(async (d) => [await deltaOf(d), await deltaFilter(d.addedRev)] as const));
		for (const [bytes, filter] of inputs) records.push(w.delta_records_for_cell(bytes, cell, filter));
		inputs.length = 0;
		const newCell = splitOf.get(cell);
		if (newCell === undefined) {
			await writeSlab(cell, mean, records);
			return;
		}
		// Split: the merged cell -> 2-means over its decoded entries -> every entry re-encoded against the nearer
		// new mean; the cell keeps mean A, the new cell gets mean B (the Worker's compactCells).
		const job = new w.SlabJob(dim, cell, mean, format);
		let merged: Uint8Array;
		let n: number;
		let finished = false;
		try {
			for (const r of records) job.add_records(r);
			n = job.len();
			finished = true;
			merged = job.finish();
		} finally {
			if (!finished) free(job);
		}
		records.length = 0;
		let means2: Float32Array;
		if (n >= 2) means2 = w.split_slab(merged, new Float32Array(0), cell + 1);
		else {
			means2 = new Float32Array(2 * dim);
			means2.set(mean);
			means2.set(mean, dim);
		}
		if (slice.pinned.has(cell)) {
			// The cell keeps its mean; the new cell takes the centre farther from it.
			const dotMean = (o: number) => {
				let d = 0;
				for (let i = 0; i < dim; i++) d += means2[o + i] * mean[i];
				return d;
			};
			const far = dotMean(0) < dotMean(dim) ? means2.slice(0, dim) : means2.slice(dim);
			means2 = new Float32Array(2 * dim);
			means2.set(mean);
			means2.set(far, dim);
		}
		const packed = w.split_slab_records(merged, new Uint8Array(0), means2, codebook);
		const la = new DataView(packed.buffer, packed.byteOffset, 4).getUint32(0, true);
		let meanA = means2.slice(0, dim);
		let meanB = means2.slice(dim);
		let recA = packed.subarray(4, 4 + la);
		let recB = packed.subarray(4 + la);
		// 2-means found no two groups (identical vectors): halve the entries under one mean, so the cell still shrinks.
		// Not when a pinned cell gave everything away: it keeps its mean and is empty, which lets its deltas fold.
		if (n >= 2 && (recB.byteLength === 0 || (recA.byteLength === 0 && !slice.pinned.has(cell)))) {
			const all = recB.byteLength === 0 ? recA : recB;
			const mean1 = recB.byteLength === 0 ? meanA : meanB;
			[recA, recB] = halveRecords(all, dim);
			meanA = mean1;
			meanB = mean1.slice();
		}
		const a = await writeSlab(cell, meanA, [recA]);
		const bn = await writeSlab(newCell, meanB, [recB]);
		means.set(cell, meanA);
		means.set(newCell, meanB);
		splitsDone.push({ cell, newCell, entries: n, a, b: bn });
	};
	// The cells the oldest delta needs first (the guarantee), then split cells (their new cells are numbered in order,
	// so a cut keeps a prefix of them), then the cells of each delta, oldest delta first (a cut then completes as many
	// deltas as it can), then the extra cells. The order changes no bytes: each cell's slab is built on its own.
	const ranked: number[] = [];
	const seen = new Set<number>();
	const push = (c: number) => {
		if (!seen.has(c)) {
			seen.add(c);
			ranked.push(c);
		}
	};
	for (const [c] of slice.splits) push(c);
	for (const d of [...foldedRefs].sort((x, y) => x.addedRev - y.addedRev || x.seq - y.seq)) for (const c of slice.deltaCells.get(d.key)!) push(c);
	for (const c of slice.cells) push(c);
	const ordered = [...ranked.filter((c) => must.has(c)), ...ranked.filter((c) => !must.has(c))];
	let next = 0;
	await Promise.all(
		Array.from({ length: slice.concurrency }, async () => {
			while (next < ordered.length && !stopped) {
				// The cells the oldest delta needs start whatever the time; the others not after the cell deadline.
				if (next >= must.size && Date.now() > cellDeadline) {
					stopped = true;
					break;
				}
				const cell = ordered[next++];
				await doCell(cell);
				done.add(cell);
			}
		}),
	);
	deltaBytes.clear();

	// ---- what can be committed ----
	let commitCells = new Set(done);
	let prefix = slice.splits.length;
	let commitFolded = foldedRefs;
	if (stopped) {
		// A split counts only with every split before it (new cells are numbered in order); a delta only when every
		// cell it has entries in is committed; a cell only when every folded delta in it is (else its new slab would
		// hold entries the delta still serves). Iterate to the fixpoint.
		prefix = 0;
		while (prefix < slice.splits.length && done.has(slice.splits[prefix][0])) prefix++;
		for (const [c] of slice.splits.slice(prefix)) commitCells.delete(c);
		for (;;) {
			commitFolded = foldedRefs.filter((d) => slice.deltaCells.get(d.key)!.every((c) => commitCells.has(c)));
			const fk = new Set(commitFolded.map((d) => d.key));
			const cc = new Set([...commitCells].filter((c) => (cellDeltas.get(c) ?? []).every((d) => fk.has(d.key))));
			let p = 0;
			while (p < prefix && cc.has(slice.splits[p][0])) p++;
			for (const [c] of slice.splits.slice(p, prefix)) cc.delete(c);
			prefix = p;
			if (cc.size === commitCells.size) break;
			commitCells = cc;
		}
		if (!commitCells.size) {
			result.outcome = "deadline";
			result.more = true;
			result.note = `deadline: ${done.size} of ${slice.cells.length} cells done, none committable`;
			return record();
		}
	}
	const commitSplits = slice.splits.slice(0, prefix);
	const commitRows: [number, SlabRow][] = [];
	for (const c of [...commitCells].sort((x, y) => x - y)) {
		commitRows.push([c, rows.get(c)!]);
		const nc = commitSplits.find(([x]) => x === c)?.[1];
		if (nc !== undefined) commitRows.push([nc, rows.get(nc)!]);
	}
	commitRows.sort((x, y) => x[0] - y[0]);
	const head = await commitSlice(ctx, w, { baseRev, manifest, files, slabs, k, dim, tombIds, known: dcells, slabRev, rows: commitRows, folded: commitFolded, splits: commitSplits, means, deltaCells: slice.deltaCells, cap: slice.limits.cellEntries, pinned: slice.pinned }, result);
	result.splits = splitsDone.filter((s) => commitSplits.some(([c]) => c === s.cell)).sort((x, y) => x.cell - y.cell);
	result.outcome = stopped ? "partial" : "committed";
	if (readNote) result.note = `${readNote}; the cells the oldest delta needs ran anyway (${commitCells.size} of ${slice.cells.length} cells)`;
	// More splitting only helps while splits move entries (a split with an empty half would repeat forever).
	const moved = result.splits.some((x) => x.a > 0 && x.b > 0);
	result.more = stopped || moreWork(head.manifest, opts.force === true, moved && head.oversized > 0) || result.splitsDeferred.length > 0;
	return record();
}

interface Slice {
	baseRev: number;
	manifest: Manifest;
	files: IndexFiles;
	slabs: SlabRow[];
	k: number;
	dim: number;
	/** The ids of the tombstone sets, read when a drop decision needs them (shared with the cells). */
	tombIds: TombIds;
	/** The deltas whose cells the plan read (the split check needs no second read of them). */
	known: DeltaCells[];
	slabRev: number;
	rows: [number, SlabRow][];
	folded: DeltaRef[];
	splits: [number, number][];
	/** Means of the rewritten cells (split cells: the new means). */
	means: Map<number, Float32Array>;
	deltaCells: Map<string, number[]>;
	/** The split cap (counts the cells still over it). */
	cap: number;
	/** Split cells that kept their mean. */
	pinned: Set<number>;
}

/**
 * Whether another call has work: with `force`, deltas remain, or this call split cells and others are still over the
 * cap; else a threshold is still exceeded.
 */
function moreWork(m: Manifest, force: boolean, splitMore: boolean): boolean {
	if (!force) return compactionExceeded(m).length > 0;
	return (m.deltas?.length ?? 0) > 0 || splitMore;
}

/** The Worker's compactCommit on the head (which may be newer than the base): rows, folded deltas, splits, tombstones. */
async function commitSlice(ctx: WriterContext, w: CompactWasm, s: Slice, result: CompactResult): Promise<{ manifest: Manifest; oversized: number }> {
	const { store, gen } = ctx;
	const folded = new Set(s.folded.map((d) => d.key));
	// Cells whose mean changes (a pinned split keeps it): no delta that stays may have entries in them.
	const splitCells = new Set(s.splits.map(([c]) => c).filter((c) => !s.pinned.has(c)));
	const rowCells = new Set(s.rows.map(([c]) => c));
	s.splits.forEach(([cell, newCell], i) => {
		if (newCell !== s.k + i || !rowCells.has(cell) || !rowCells.has(newCell)) throw new ThreadError(`split ${i} of cell ${cell} -> ${newCell} is inconsistent`, 500);
	});
	for (const d of s.folded) for (const c of s.deltaCells.get(d.key)!) if (!rowCells.has(c)) throw new ThreadError(`folded delta ${d.key} has entries in cell ${c}, which is not among the rewritten rows`, 500);
	// The split check below needs the cells of every delta that stays. While any delta of the base was unread every
	// split is pinned (planSlice), so `splitCells` is empty and nothing is read; otherwise the plan read them all, and
	// only the deltas a writer added after the base are read here.
	const cellsMemo = new Map<string, Promise<DeltaCells>>(s.known.map((d) => [d.ref.key, Promise.resolve(d)]));
	const cellsOf = (d: DeltaRef) => {
		let p = cellsMemo.get(d.key);
		if (!p) cellsMemo.set(d.key, (p = readDeltaCells(store, d, d.bytes <= DELTA_WHOLE_BYTES).then(({ ref, counts }) => ({ ref, counts }))));
		return p;
	};
	let routing: { forFine: string; fine: string; k: number; tree?: TreeRef; routing?: RoutingTables } | null = null;
	let dropped: number[] = [];
	let oversized = 0;
	const c = await commitRev(ctx, async ({ rev, manifest, files }) => {
		if (files.slabs !== s.files.slabs) throw new ThreadError(`slabs table changed since rev ${s.baseRev} (another compaction committed)`, 409);
		if (manifest.k1 * manifest.k2 !== s.k || files.fine !== s.files.fine) throw new ThreadError(`routing table changed since rev ${s.baseRev}`, 409);
		const deltas = manifest.deltas ?? [];
		for (const key of folded) if (!deltas.some((d) => d.key === key)) throw new ThreadError(`folded delta ${key} is no longer in rev ${rev}`, 409);
		const remaining = deltas.filter((d) => !folded.has(d.key));
		if (splitCells.size) {
			for (const d of remaining) {
				const dc = await cellsOf(d);
				for (const cell of dc.counts.keys()) if (splitCells.has(cell)) throw new ThreadError(`delta ${d.key} (not folded) has entries in split cell ${cell}`, 409);
			}
		}
		const kk = s.k + s.splits.length;
		const table: SlabRow[] = s.slabs.slice();
		for (const [cell, row] of s.rows) table[cell] = row;
		if (table.length !== kk || table.some((r) => !r)) throw new ThreadError(`slabs table would have ${table.length} rows, expected ${kk}`, 500);
		oversized = s.cap > 0 ? table.filter((r) => r[2] > s.cap).length : 0;
		// The sets of the head (a writer may have added some since the base): ids are read only for the build-id test.
		const tombRefs: TombRef[] = manifest.tombstones ?? [];
		const drop = await droppableSets(tombRefs, remaining, table, manifest.vectors, manifest.segments ?? [], s.tombIds);
		const keep = tombRefs.filter((_, i) => !drop[i]);
		dropped = tombRefs.filter((_, i) => drop[i]).map((t) => t.rev);
		if (!s.rows.length && !dropped.length) return null;
		const next: Manifest = { ...inherit(manifest), deltas: remaining, tombstones: keep.map(({ rev: r, key, count, scope }) => ({ rev: r, key, count, ...(scope ? { scope } : {}) })) };
		if (s.rows.length) next.files = { ...files, slabs: await putUnique(store, keys.slabsTable(gen, rev + 1), JSON.stringify(table)) };
		if (s.splits.length) {
			if (!routing || routing.forFine !== files.fine) routing = { forFine: files.fine, ...(await splitRouting(ctx, w, rev + 1, manifest, files, s, kk)) };
			next.k2 = routing.k;
			next.files = { ...next.files!, fine: routing.fine };
			if (routing.tree) next.tree = routing.tree;
			if (routing.routing) next.routing = routing.routing;
			else delete next.routing;
		}
		const splitChange = s.splits.length ? { splits: s.splits.map(([cell, newCell]) => [cell, newCell, table[cell][2], table[newCell][2]]), k: { from: s.k, to: kk } } : {};
		return { ...next, change: { op: "compact", baseRev: s.baseRev, slabRev: s.slabRev, cells: s.rows.length, foldedDeltas: folded.size, droppedTombstoneSets: dropped.length, ...splitChange, writer: "thread-sdk" } };
	});
	result.rev = c.rev;
	result.cells = s.rows.length;
	result.folded = folded.size;
	result.remainingDeltas = c.manifest.deltas?.length ?? 0;
	result.droppedTombstoneSets = dropped;
	result.k = c.manifest.k1 * c.manifest.k2;
	result.conflicts = c.conflicts;
	result.currentUpdated = c.currentUpdated;
	return { manifest: c.manifest, oversized };
}

/**
 * The routing objects of a revision whose compaction split cells (the Worker's splitRouting): fine means with
 * k + splits rows (a split cell keeps its index with mean A, each new cell is appended with mean B), a centroid tree
 * rebuilt over them with the same groups/iters/seed/beam when the head has one (no recall measurement, as in the
 * Worker's automatic runs), and the compact routing tables the head had for the old means.
 */
async function splitRouting(ctx: WriterContext, w: CompactWasm, rev: number, m: Manifest, files: IndexFiles, s: Slice, kk: number): Promise<{ fine: string; k: number; tree?: TreeRef; routing?: RoutingTables }> {
	const { store, gen } = ctx;
	const dim = m.dim;
	const old = await getOrThrow(store, files.fine);
	if (old.byteLength !== s.k * dim * 4) throw new ThreadError(`${files.fine} has ${old.byteLength / 4 / dim} means, expected ${s.k}`, 500);
	const means = new Float32Array(kk * dim);
	means.set(new Float32Array(old.slice().buffer));
	for (const [cell, newCell] of s.splits) {
		means.set(s.means.get(cell)!, cell * dim);
		means.set(s.means.get(newCell)!, newCell * dim);
	}
	const fine = await putUnique(store, `idx/${gen}/fine.${rev}.f32`, new Uint8Array(means.buffer));
	let tree: TreeRef | undefined;
	let treeBytes: Uint8Array | null = null;
	if (m.tree) {
		const t = m.tree;
		const groups = Math.min(t.groups, kk);
		const iters = typeof t.iters === "number" ? t.iters : 10;
		const seed = typeof t.seed === "number" ? t.seed : 7;
		const bytes = w.build_centroid_tree(means, dim, groups, iters, seed);
		const [tDim, tK, tGroups] = w.centroid_tree_info(bytes);
		if (tDim !== dim || tK !== kk) throw new ThreadError(`tree is dim ${tDim} k ${tK}, expected ${dim} / ${kk}`, 500);
		tree = { key: await putUnique(store, `idx/${gen}/tree.${rev}.bin`, bytes), groups: tGroups, beam: Math.min(t.beam, tGroups), iters, seed, bytes: bytes.byteLength };
		treeBytes = bytes;
	}
	const r = m.routing && m.routing.fine === files.fine ? m.routing : undefined;
	const tables = r ? (["f16", "i8"] as const).filter((e) => r[e]) : [];
	let routing: RoutingTables | undefined;
	if (r && (tables.length || r.grouped)) {
		const grouped = r.grouped?.elem;
		const built = buildRoutingObjects(w, { fineKey: fine, fine: means, dim, tables: [...tables], grouped: tree && treeBytes && grouped ? { gen, rev, treeKey: tree.key, tree: treeBytes, elem: grouped } : undefined });
		const put = async (key: string) => putUnique(store, key, built.objects.find(([k]) => k === key)![1]);
		routing = { fine };
		for (const elem of tables) routing[elem] = { ...built.routing[elem]!, key: await put(built.routing[elem]!.key) };
		if (built.routing.grouped) routing.grouped = { ...built.routing.grouped, key: await put(built.routing.grouped.key) };
	}
	return { fine, k: kk, tree, routing };
}

/**
 * Writes a slab revision at its conventional key (the slabs table names slabs by cell and slabRev, so the key cannot
 * be made unique). Create-only; an existing object with the same bytes (a rerun on the same base) is reused. One
 * with other bytes is overwritten only while no revision after the base has changed the slabs table, i.e. no
 * revision can list it; otherwise the run stops with a conflict.
 */
async function putSlab(store: BlobStore, gen: string, baseRev: number, files: IndexFiles, key: string, bytes: Uint8Array): Promise<void> {
	if (await putIfAbsent(store, key, bytes)) return;
	const cur = await store.get(key);
	if (cur && equalBytes(cur, bytes)) return;
	if (await slabsChangedAfter(store, gen, baseRev, files)) throw new ThreadError(`${key} exists with other bytes and a revision after ${baseRev} changed the slabs table`, 409);
	await writable(store).put(key, bytes);
}

async function slabsChangedAfter(store: BlobStore, gen: string, baseRev: number, files: IndexFiles): Promise<boolean> {
	const head = await latestRev(store, gen);
	for (let rev = baseRev + 1; rev <= head; rev++) {
		const b = await store.get(keys.manifest(gen, rev));
		if (b && manifestFiles(parseManifest(b, gen, rev)).slabs !== files.slabs) return true;
	}
	return false;
}

/**
 * Splits a run of part records (thread-core part.rs: a 24-byte header with `payload_len` at 16 and `pq_m` at 20,
 * then `dim / 2` 4-bit codes, `dim` int8 codes, the PQ codes and the payload) into its first half and the rest.
 */
export function halveRecords(records: Uint8Array, dim: number): [Uint8Array, Uint8Array] {
	const dv = new DataView(records.buffer, records.byteOffset, records.byteLength);
	const ends: number[] = [];
	for (let pos = 0; pos < records.byteLength; ) {
		if (pos + 24 > records.byteLength) throw new ThreadError("truncated part record", 500);
		pos += 24 + dim / 2 + dim + dv.getUint16(pos + 20, true) + dv.getUint32(pos + 16, true);
		ends.push(pos);
	}
	if (ends[ends.length - 1] !== records.byteLength) throw new ThreadError("part records overrun their buffer", 500);
	const cut = ends.length >= 2 ? ends[Math.ceil(ends.length / 2) - 1] : records.byteLength;
	return [records.subarray(0, cut), records.subarray(cut)];
}

function equalBytes(a: Uint8Array, b: Uint8Array): boolean {
	if (a.byteLength !== b.byteLength) return false;
	for (let i = 0; i < a.byteLength; i++) if (a[i] !== b[i]) return false;
	return true;
}

function u32View(b: Uint8Array): Uint32Array {
	return b.byteOffset % 4 === 0 ? new Uint32Array(b.buffer, b.byteOffset, b.byteLength >> 2) : new Uint32Array(b.slice().buffer);
}

async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
	const out: R[] = new Array(items.length);
	let i = 0;
	await Promise.all(
		Array.from({ length: Math.min(limit, items.length) }, async () => {
			while (i < items.length) {
				const j = i++;
				out[j] = await fn(items[j]);
			}
		}),
	);
	return out;
}
