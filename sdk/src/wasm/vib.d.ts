/* tslint:disable */
/* eslint-disable */

/**
 * Assign job: multi-assigns each vector to its `m` nearest fine cells (within
 * the `probe_coarse` best coarse groups; `0` or `>= k1` = all groups), encodes
 * the residual against each cell's mean in both tiers (4-bit MSE-clipped scan
 * code + int8 rerank code), and buffers records per coarse group. After
 * `set_codebook`, records also carry the residual's PQ codes (needed to
 * build format 3 slabs); without it their `pq_m` is 0.
 */
export class AssignJob {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Vectors have ids `first_id, first_id + 1, ...`. Payloads are empty.
     */
    add_batch(vectors: Float32Array, first_id: number): number;
    /**
     * `with_candidates` jobs only: like `add_batch`, but vector `i`'s top `m`
     * cells are chosen among the neighbours of `prev[i]` (its cell from the
     * last k-means iteration). Returns vectors seen.
     */
    add_batch_prev(vectors: Float32Array, first_id: number, prev: Uint32Array): number;
    assigned_records(): number;
    constructor(dim: number, k1: number, k2: number, coarse_means: Float32Array, fine_means: Float32Array, probe_coarse: number, m: number);
    /**
     * Current serialized size of part `g`.
     */
    part_len(g: number): number;
    /**
     * PQ codes per record (0 without a codebook).
     */
    pq_m(): number;
    /**
     * Makes every following record carry the PQ codes of its residual under
     * this codebook (`pq_init_codebook` / `pq_reduce` bytes; its dim must be the
     * job's dim). Works for jobs from either constructor; call before adding.
     */
    set_codebook(codebook_bytes: Uint8Array): void;
    /**
     * Makes `add_batch` pick each vector's top `m` cells with a pruned search:
     * `tree_bytes` (`build_centroid_tree` over all `k1 * k2` fine means, global
     * cell ids) scores its group means and ranks only the cells of the best
     * `beam` groups; `probe_coarse` is then ignored. With `beam >= groups` the
     * parts equal those of the exhaustive flat search (`probe_coarse = k1`, or
     * `k1 = 1`). `add_batch_prev` is unaffected. Call before adding.
     */
    set_tree(tree_bytes: Uint8Array, beam: number): void;
    /**
     * Serializes part `g` and resets its buffer.
     */
    take_part(g: number): Uint8Array;
    /**
     * Flat (`k1 = 1`: one part group, `k2 = k`) assign job whose
     * `add_batch_prev` picks each vector's top `m` cells among the `n_neigh`
     * neighbours (`neighbors`, a `k x n_neigh` table from `cell_neighbors`)
     * of its previous cell. Requires `n_neigh >= m`. `add_batch` also works
     * (exhaustive flat top-m).
     */
    static with_candidates(dim: number, k: number, means: Float32Array, neighbors: Uint32Array, n_neigh: number, m: number): AssignJob;
}

/**
 * Update job: encodes new vectors into one delta segment ("VIBD"). Holds the
 * `k * dim` cell means (+ inverse norms, + the PQ codebook after
 * `set_codebook`). Each vector goes to its primary cell (highest cosine over
 * all `k` cells) with residual codes against that cell's mean, as an
 * `AssignJob` part record (4-bit + int8, + PQ codes with a codebook), so
 * compaction moves it into the cell's slab without re-encoding. Vectors
 * must be unit-normalized, as for `AssignJob`.
 */
export class DeltaJob {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Adds `n` vectors (`n * dim`) with ids `ids` (`n`) and attrs (`n`, or
     * empty for all 0), no payloads. Returns each vector's primary cell.
     * Nothing is added if the arguments are inconsistent.
     */
    add(vectors: Float32Array, ids: Uint32Array, attrs: Uint32Array): Uint32Array;
    /**
     * Like `add`, with payloads: `payloads` is the concatenation of every
     * vector's payload, `payload_lens[i]` the length of vector `i`'s.
     */
    add_with_payloads(vectors: Float32Array, ids: Uint32Array, attrs: Uint32Array, payloads: Uint8Array, payload_lens: Uint32Array): Uint32Array;
    /**
     * Serialized size of the segment so far.
     */
    byte_len(): number;
    /**
     * The serialized delta segment.
     */
    finish(): Uint8Array;
    /**
     * Entries added so far.
     */
    len(): number;
    /**
     * `means`: the generation's `k * dim` fine means (`fine.f32`); `seq`: the
     * segment's sequence number (stored in its header).
     */
    constructor(dim: number, k: number, means: Float32Array, seq: number);
    /**
     * PQ codes per entry (0 without a codebook).
     */
    pq_m(): number;
    /**
     * Entries also carry PQ codes under this codebook (required to compact
     * into format 3 slabs). Call before adding vectors.
     */
    set_codebook(codebook_bytes: Uint8Array): void;
    /**
     * Same as `set_codebook`, from a parsed `PqCodebook` (copied).
     */
    set_pq_codebook(codebook: PqCodebook): void;
    /**
     * Picks each added vector's primary cell with a pruned search: the tree
     * (`build_centroid_tree` over these `k` means) scores its group means and
     * only the cells of the best `beam` groups (`beam >= groups`: identical to
     * the exhaustive search). Applies to vectors added afterwards.
     */
    set_tree(tree_bytes: Uint8Array, beam: number): void;
}

/**
 * Picks initial fine centroids: a reservoir of up to `k2` sample vectors per
 * coarse group, streamed.
 */
export class FineInitJob {
    free(): void;
    [Symbol.dispose](): void;
    add_batch(vectors: Float32Array): number;
    /**
     * `k1 * k2 * dim` initial fine means, grouped by coarse group.
     */
    finish(): Float32Array;
    constructor(dim: number, k1: number, k2: number, coarse_means: Float32Array, seed: number);
}

/**
 * Flat k-means map job over `k` centroids (used for the coarse level).
 */
export class FlatKmeansJob {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Assigns each vector in the batch and accumulates. Returns vectors seen.
     */
    add_batch(vectors: Float32Array): number;
    /**
     * Convenience for single-job k-means: new means from this job's accumulator alone.
     */
    means(prev: Float32Array): Float32Array;
    constructor(dim: number, k: number, centroid_means: Float32Array);
    /**
     * Serialized partial accumulator.
     */
    partial(): Uint8Array;
}

/**
 * Incremental flat k-means map job: each vector is compared only against the
 * `n_neigh` neighbour cells of its previous cell (`neighbors`, a `k x n_neigh`
 * table from `cell_neighbors`), not all `k` centroids. Accumulates like
 * `TreeKmeansJob`; `partial()` has the same format, so `Reducer` is unchanged.
 */
export class IncKmeansJob {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Assigns vector `i` to its best cell among the neighbours of `prev[i]`,
     * accumulates, and returns the new cell per vector. Nothing is
     * accumulated if the arguments are invalid.
     */
    add_batch(vectors: Float32Array, prev: Uint32Array): Uint32Array;
    constructor(dim: number, k: number, means: Float32Array, neighbors: Uint32Array, n_neigh: number);
    partial(): Uint8Array;
}

/**
 * A parsed codebook to keep across queries (parsing ~1.5 MB per query is
 * avoidable): build each query's table with `PqTable.from_codebook`.
 */
export class PqCodebook {
    free(): void;
    [Symbol.dispose](): void;
    dim(): number;
    m(): number;
    constructor(codebook_bytes: Uint8Array);
}

/**
 * Streaming reduce of PQ training partials: `add_partial` each partial
 * (so only one is held at a time), then `finish(prev_codebook)` returns the
 * next codebook. Empty sub-centroids keep their previous value.
 */
export class PqReducer {
    free(): void;
    [Symbol.dispose](): void;
    add_partial(partial: Uint8Array): void;
    /**
     * Sub-centroids (out of `m * 256`) that received no residual.
     */
    empty_centroids(): number;
    /**
     * Next serialized codebook.
     */
    finish(prev_codebook: Uint8Array): Uint8Array;
    /**
     * Mean squared quantization error over all added partials (under the
     * codebook the map jobs used): a convergence signal per iteration.
     */
    mean_sq_err(): number;
    constructor(dim: number, m: number);
    /**
     * Residuals over all added partials.
     */
    rows(): number;
}

/**
 * Per-query asymmetric-distance table (`m x 256` f32: `q_sub · sub-centroid`).
 * Build once per query, then `scan_pq` every probed v3 slab with it.
 */
export class PqTable {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Same as the constructor, from a cached `PqCodebook`.
     */
    static from_codebook(q: Float32Array, codebook: PqCodebook): PqTable;
    m(): number;
    /**
     * `q`: the unit query; `codebook_bytes`: the index's serialized codebook.
     */
    constructor(q: Float32Array, codebook_bytes: Uint8Array);
}

/**
 * PQ training map job (one Lloyd iteration): assigns each residual's
 * subvectors to the nearest sub-centroids of the current codebook and
 * accumulates sums and counts. `partial()` is ~1.8 MB at dim 1536, m 192.
 */
export class PqTrainJob {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Adds `n * dim` residuals (see `residuals_for`). Returns `n`.
     */
    add_residuals(residuals: Float32Array): number;
    /**
     * Mean squared quantization error of the added residuals under the
     * job's (input) codebook.
     */
    mean_sq_err(): number;
    constructor(dim: number, m: number, codebook_bytes: Uint8Array);
    /**
     * Serialized partial accumulator ("VIBT"), for `pq_reduce` / `PqReducer`.
     */
    partial(): Uint8Array;
    /**
     * Residuals added so far.
     */
    rows(): number;
}

/**
 * Reduce job over a cell range: sums partial records, then emits new means.
 */
export class Reducer {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Adds the raw records for this reducer's cell range from one partial.
     */
    add_records(records: Uint8Array): void;
    /**
     * Per-cell counts after reduction (useful to spot empty cells).
     */
    counts(): Uint32Array;
    /**
     * Cells reseeded by the last `means_reseed` call.
     */
    last_reseeded(): number;
    /**
     * New means; `prev` is the previous means for the same cell range.
     */
    means(prev: Float32Array): Float32Array;
    /**
     * Like `means`, but cells with fewer than `min_count` members are reseeded
     * next to the densest cells of this range (deterministic jitter from `seed`).
     * The number of reseeded cells is available from `last_reseeded()`.
     */
    means_reseed(prev: Float32Array, min_count: number, seed: number): Float32Array;
    constructor(cells: number, dim: number);
}

/**
 * Query-side: routing table plus slab scanning.
 */
export class Searcher {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * `q·mean[c]` (unnormalized means) for all `k1 * k2` fine cells, by global
     * cell id: the `qc` argument of `scan_delta`. Pass the result to
     * `route_from_dots` to route without computing the dots twice.
     */
    cell_dots(q: Float32Array): Float32Array;
    /**
     * `q·mean[c]` for the listed global cell ids only, in order (the same
     * values as `cell_dots(q)[c]`): the delta-scan base scores of just the
     * cells the caller needs.
     */
    cell_dots_for(q: Float32Array, cells: Uint32Array): Float32Array;
    dim(): number;
    constructor(dim: number, k1: number, k2: number, coarse_means: Float32Array, fine_means: Float32Array);
    /**
     * Global ids of the `nprobe` fine cells to scan for unit query `q`.
     * `probe_coarse == 0` (or `>= k1`) ranks all fine cells directly (flat routing);
     * otherwise only the fine cells of the `probe_coarse` best coarse groups are ranked.
     */
    route(q: Float32Array, probe_coarse: number, nprobe: number): Uint32Array;
    /**
     * Pruned flat routing: global ids of the `nprobe` best fine cells (by
     * cosine, descending) among the members of the tree's best `beam` groups.
     * `beam >= groups` equals `route_flat(q, nprobe)`. Error without `set_tree`.
     */
    route_beam(q: Float32Array, beam: number, nprobe: number): Uint32Array;
    /**
     * Flat routing: global ids of the `nprobe` best fine cells over all `k1 * k2`.
     */
    route_flat(q: Float32Array, nprobe: number): Uint32Array;
    /**
     * Flat routing from `cell_dots(q)`: same result as `route_flat(q, nprobe)`.
     */
    route_from_dots(dots: Float32Array, nprobe: number): Uint32Array;
    /**
     * Installs a centroid tree (`build_centroid_tree` over this searcher's
     * `k1 * k2` fine means) for `route_beam`. Replaces any previous tree.
     */
    set_tree(tree_bytes: Uint8Array): void;
    /**
     * Coarse group ids ranked for `q` (for lazy fine-centroid loading).
     */
    top_coarse(q: Float32Array, n: number): Uint32Array;
    /**
     * Groups of the installed tree (0 = none).
     */
    tree_groups(): number;
    /**
     * Fraction of `sample` vectors (`n * dim`) whose exact nearest fine cell
     * the tree finds with this `beam` (for logging; costs an exhaustive
     * search per vector). Error without `set_tree`.
     */
    tree_recall(sample: Float32Array, beam: number): number;
    /**
     * Mean fraction of each sample vector's exact top `m` fine cells that the
     * tree's top `m` with this `beam` contains (`m = 1`: `tree_recall`).
     */
    tree_recall_top(sample: Float32Array, beam: number, m: number): number;
}

/**
 * Finalize job for one fine cell: concatenates part record ranges into a slab.
 */
export class SlabJob {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Adds every record in a records byte range (one cell's directory range,
     * or any run of whole records, e.g. a cell sliced out of a `part_span`).
     */
    add_records(bytes: Uint8Array): number;
    /**
     * Adds the records of one cell from a fetched multi-cell span without
     * copying it in JS: `span` holds part bytes starting at part offset
     * `span_offset`; the cell's directory entry is `[cell_off, cell_len)`
     * (part offsets). Adds `span[cell_off - span_offset, +cell_len)`.
     */
    add_records_range(span: Uint8Array, span_offset: number, cell_off: number, cell_len: number): number;
    byte_len(): number;
    finish(): Uint8Array;
    /**
     * Slab format version this job writes (1, 2 or 3).
     */
    format(): number;
    len(): number;
    /**
     * `format`: 2 (default when omitted) = v2 slab (4-bit scan tier + int8
     * rerank tier); 1 = v1 int8-only slab (from the records' int8 codes);
     * 3 = v3 slab (the records' PQ codes as scan tier + int8 rerank tier; the
     * records must come from an `AssignJob` with `set_codebook`, otherwise
     * `add_records*` errors).
     */
    constructor(dim: number, cell: number, centroid_mean: Float32Array, format?: number | null);
}

/**
 * Two-level k-means map job: assigns each vector to its best fine cell among
 * the `probe_coarse` nearest coarse groups (`0` or `>= k1` = all groups, as
 * `Searcher.route`) and accumulates per fine cell.
 */
export class TreeKmeansJob {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Like `add_batch`, but returns the chosen global fine cell of every
     * vector (with `k1 = 1` a flat assignment), e.g. to seed `IncKmeansJob`.
     */
    add_batch_assign(vectors: Float32Array): Uint32Array;
    add_batch(vectors: Float32Array): number;
    constructor(dim: number, k1: number, k2: number, coarse_means: Float32Array, fine_means: Float32Array, probe_coarse: number);
    partial(): Uint8Array;
}

export function __wbg_reset_state(): void;

/**
 * Byte range `[offset, length]` inside a serialized partial accumulator that
 * covers cells `[first, first + n)`.
 */
export function accum_record_range(dim: number, first: number, n: number): Uint32Array;

/**
 * Builds a centroid tree ("VIBC" bytes) over `k = means.len() / dim` cell
 * means (raw, e.g. `fine.f32`): spherical k-means of the means into
 * `groups` groups (clamped to `1..=k`), k-means++ seeding, up to `iters`
 * Lloyd iterations, deterministic for `seed`. Lookups (`Searcher.set_tree`,
 * `DeltaJob.set_tree`, `AssignJob.set_tree`) score the group means, then only
 * the cells of the best `beam` groups. Cost: about `(iters + 2) * groups * k * dim`
 * multiply-adds (k = 4096, dim = 1536, groups = 64, iters = 10: ~5e9).
 */
export function build_centroid_tree(means: Float32Array, dim: number, groups: number, iters: number, seed: number): Uint8Array;

/**
 * Cell neighbour table rows for cells `[first, first + count)`: the `n`
 * cells with the highest cosine between means, each row starting with the
 * cell itself, flattened row-major (`count * n` ids). `means` is the full
 * `k * dim` table; shard the work over `first/count` ranges.
 */
export function cell_neighbors(means: Float32Array, dim: number, first: number, count: number, n: number): Uint32Array;

/**
 * Header summary of centroid tree bytes: `[dim, k, groups, n_members, byte_len]`.
 */
export function centroid_tree_info(tree_bytes: Uint8Array): Uint32Array;

/**
 * Distinct cells of a delta segment with their entry counts, ascending:
 * `[cell0, count0, cell1, count1, ...]`.
 */
export function delta_cells(delta_bytes: Uint8Array): Uint32Array;

/**
 * Header summary of a delta segment: `[dim, count, pq_m, seq, byte_len]`.
 */
export function delta_info(delta_bytes: Uint8Array): Uint32Array;

/**
 * Payload of delta entry `idx` (a `scan_delta` hit's `idx`).
 */
export function delta_payload(delta_bytes: Uint8Array, idx: number): Uint8Array;

/**
 * Part records (for `SlabJob.add_records`) of every entry of `cell` in a delta
 * segment whose id is not in `tombstones` (`u32` LE ids, may be empty).
 */
export function delta_records_for_cell(delta_bytes: Uint8Array, cell: number, tombstones: Uint8Array): Uint8Array;

/**
 * Drops hits whose id is tombstoned from packed hits (one or more
 * concatenated `scan`/`merge` buffers); returns one packed buffer, order kept.
 * Applied to a finished top-k it can leave fewer than k hits: queries pass
 * the tombstones to `scan`/`scan_pq`/`scan_delta` instead.
 */
export function filter_hits(packed: Uint8Array, tombstones: Uint8Array): Uint8Array;

/**
 * Installs a panic hook that logs the panic message (and location) via
 * `console.error`, so panics show up in Workers logs before the wasm traps.
 * Idempotent; call once after instantiating the module. After a trap
 * (`WebAssembly.RuntimeError`) the instance is poisoned: call
 * `__wbg_reset_state()` and re-init before using it again.
 */
export function init_panic_hook(): void;

/**
 * Merges concatenated packed hit buffers, dedupes by id (keeping the best
 * score and that copy's `idx`), keeps the best `k`.
 */
export function merge(packed: Uint8Array, k: number): Uint8Array;

/**
 * Returns a unit-normalized copy of `q`.
 */
export function normalize_vector(q: Float32Array): Float32Array;

/**
 * Parses a part directory: returns `[k2, dim, count0, off0, len0, count1, ...]`.
 */
export function part_directory(bytes: Uint8Array): Uint32Array;

export function part_directory_len(k2: number): number;

/**
 * Byte span `[offset, length]` of one part covering cells
 * `[first_cell, first_cell + count)`, from `dir` = the output of
 * `part_directory`. Empty cells at the edges are skipped; no records gives
 * `[0, 0]`. Slice each cell out of the fetched span with its directory entry
 * (`off - offset`), or use `SlabJob.add_records_range`.
 */
export function part_span(dir: Uint32Array, first_cell: number, count: number): Uint32Array;

/**
 * `[dim, m, sub]` of a serialized codebook.
 */
export function pq_codebook_info(codebook_bytes: Uint8Array): Uint32Array;

/**
 * Initial PQ codebook (serialized, "VIBQ"): per subspace, the subvectors of
 * 256 random rows of `residuals` (`n * dim`; pass ~2-4k rows). Requires
 * `dim % m == 0`.
 */
export function pq_init_codebook(residuals: Float32Array, dim: number, m: number, seed: number): Uint8Array;

/**
 * One-shot reduce: `partials_concat` is `n_partials` `PqTrainJob.partial()`
 * buffers back to back (all the same size). Returns the next codebook.
 * Prefer `PqReducer` when the partials do not fit in memory together.
 */
export function pq_reduce(partials_concat: Uint8Array, n_partials: number, prev_codebook: Uint8Array): Uint8Array;

/**
 * Re-encodes part records (decoded from their int8 tier against `old_mean`)
 * against `new_mean`: 4-bit, int8 and, with a codebook (empty bytes = none),
 * PQ codes. Ids, attrs and payloads are kept.
 */
export function reencode_records(records: Uint8Array, old_mean: Float32Array, new_mean: Float32Array, codebook_bytes: Uint8Array): Uint8Array;

/**
 * Scores `n` consecutive v2/v3 rerank records (`[scale8 f32][code8 i8 * dim]`,
 * `dim = q.length`) from one slab against `q`: `q·centroid + scale8 * (q·code8)`.
 */
export function rerank(q: Float32Array, centroid: Float32Array, records: Uint8Array, n: number): Float32Array;

/**
 * `vectors[i] - means[assign[i]]` for each row (`n * dim` floats): turns a
 * sample batch and its cell assignment into PQ training residuals.
 */
export function residuals_for(vectors: Float32Array, means: Float32Array, assign: Uint32Array, dim: number): Float32Array;

/**
 * Scans one slab for unit query `q`; returns packed top-k hits
 * (`[count u32][id u32, score f32, idx u32]*`, `idx` = entry index in the slab).
 * For a v2 slab, pass the `[0, scan_len)` prefix (the full slab also works);
 * scores are 4-bit scan-tier scores, to be refined with `rerank`. v1 slabs
 * must be passed whole up to their payload and give int8 scores. v3 slabs are
 * an error: scan them with `scan_pq`.
 *
 * `tombstones` (`u32` LE ids, sorted unique as `tombstones_merge` writes them;
 * empty = none) are skipped inside the scan, before top-k insertion, so the
 * result holds the best `k` live entries. `stats` is an out-parameter
 * (`Uint32Array`, copied back after the call): `stats[0]` = entries scanned,
 * `stats[1]` = tombstoned entries skipped (only those that scored high enough
 * to enter the top-k when reached). Pass `new Uint32Array(0)` to ignore them.
 */
export function scan(slab_scan_bytes: Uint8Array, q: Float32Array, k: number, tombstones: Uint8Array, stats: Uint32Array): Uint8Array;

/**
 * Scans a whole delta segment for unit query `q` with its int8 records:
 * score = `qc[cell] + scale8 * (q·code8)`, where `qc = Searcher.cell_dots(q)`
 * (one dot per fine cell, unnormalized means). Returns the top `k` as packed
 * hits in the `scan` wire format, `idx` = entry index in the segment. Scores
 * are on the reranked (int8) scale, so they merge directly with `rerank`
 * results; no second phase is needed. `tombstones` and `stats` as in `scan`:
 * pass the ids to hide in this segment (for a segment added at rev `a`, the
 * union of the sets with rev > `a`); `stats` receives `[scanned, skipped]`.
 */
export function scan_delta(delta_bytes: Uint8Array, q: Float32Array, qc: Float32Array, k: number, tombstones: Uint8Array, stats: Uint32Array): Uint8Array;

/**
 * Scans one v3 slab (its `[0, scan_len)` prefix, or the whole slab) for unit
 * query `q` with PQ scores `q·centroid + sum_j table[j][code_j]`; returns
 * packed top-k hits in the `scan` wire format (`idx` = entry index, locating
 * the int8 rerank record for `rerank`). `table` must be built from the same
 * `q`. Errors on v1/v2 slabs and on a table whose `m` differs from the slab's.
 * `tombstones` and `stats` as in `scan`: tombstoned ids are skipped before
 * top-k insertion; `stats` receives `[scanned, skipped]`.
 */
export function scan_pq(slab_scan_bytes: Uint8Array, table: PqTable, q: Float32Array, k: number, tombstones: Uint8Array, stats: Uint32Array): Uint8Array;

/**
 * The cell mean stored in a slab (or v2 scan prefix).
 */
export function slab_centroid(slab_scan_bytes: Uint8Array): Float32Array;

/**
 * Header summary of a slab (or v2/v3 scan prefix):
 * `[dim, count, cell, byte_len, version, scan_len, rerank_off, rerank_record_len, pq_m]`.
 * `byte_len` is the full object size from the header (equals `slab.length`
 * for a whole slab). For v1, `scan_len` is the payload offset and
 * `rerank_off`/`rerank_record_len` are 0. `pq_m` is 0 except for v3.
 */
export function slab_info(slab: Uint8Array): Uint32Array;

/**
 * Bytes to range-read for a scan, from the first 64 header bytes.
 */
export function slab_scan_len(header_bytes: Uint8Array): number;

/**
 * Part records (for `SlabJob.add_records`) of every entry of a whole slab
 * (not a scan prefix) whose id is not in `tombstones` (may be empty). A slab
 * rebuilt from them with `SlabJob(dim, cell, mean, sameFormat)` is byte-identical.
 * v3 and v1 slabs have no 4-bit codes: their records are flagged and a
 * format 2 `SlabJob` rejects them (`add_records` errors); v2 records carry no
 * PQ codes, so a format 3 `SlabJob` rejects them.
 */
export function slab_to_records(slab_bytes: Uint8Array, tombstones: Uint8Array): Uint8Array;

/**
 * The same split for part records (e.g. `delta_records_for_cell` of the split
 * cell, or `slab_to_records`): decoded against `old_mean` (the cell's mean the
 * records were encoded with; its length is the dim). Same output layout as
 * `split_slab_records`.
 */
export function split_part_records(records: Uint8Array, old_mean: Float32Array, means2: Float32Array, codebook_bytes: Uint8Array): Uint8Array;

/**
 * Proposes a split of an overfull cell: decodes every entry of a whole slab
 * (any format; v2/v3 need the rerank section) as `mean + scale8 * code8`,
 * runs spherical 2-means (`kmeans::split_cell`, seeded with `seed`) and
 * returns the two new means (`2 * dim` floats). `mean` must be the slab's
 * cell mean; pass an empty array to use the mean stored in the slab.
 * Memory: `count * dim * 4` bytes of decoded vectors.
 */
export function split_slab(slab_bytes: Uint8Array, mean: Float32Array, seed: number): Float32Array;

/**
 * Executes a cell split: every entry of a whole slab (`slab_bytes`, not a scan
 * prefix) whose id is not in `tombstones` (`u32` LE, may be empty) is decoded
 * from its int8 tier against the slab's mean, assigned to the nearer (cosine)
 * of the two new means `means2` (`2 * dim` floats, e.g. from `split_slab`) and
 * re-encoded against it as a part record (4-bit, int8, and PQ codes when
 * `codebook_bytes` is a codebook; pass empty bytes for none, which only
 * format 1/2 slabs accept). Returns ONE buffer: a 4-byte `u32` LE prefix
 * holding the byte length `la` of records_a, then records_a (`[4, 4 + la)`),
 * then records_b (`[4 + la, end)`). Feed them to `SlabJob(dim, cell, meanA,
 * format)` and `SlabJob(dim, newCell, meanB, format)`.
 */
export function split_slab_records(slab_bytes: Uint8Array, tombstones: Uint8Array, means2: Float32Array, codebook_bytes: Uint8Array): Uint8Array;

/**
 * The union of a tombstone buffer (`u32` LE ids, sorted unique; may be
 * empty) and `add`, serialized the same way.
 */
export function tombstones_merge(existing: Uint8Array, add: Uint32Array): Uint8Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_assignjob_free: (a: number, b: number) => void;
    readonly __wbg_deltajob_free: (a: number, b: number) => void;
    readonly __wbg_fineinitjob_free: (a: number, b: number) => void;
    readonly __wbg_flatkmeansjob_free: (a: number, b: number) => void;
    readonly __wbg_inckmeansjob_free: (a: number, b: number) => void;
    readonly __wbg_pqcodebook_free: (a: number, b: number) => void;
    readonly __wbg_pqreducer_free: (a: number, b: number) => void;
    readonly __wbg_pqtable_free: (a: number, b: number) => void;
    readonly __wbg_pqtrainjob_free: (a: number, b: number) => void;
    readonly __wbg_reducer_free: (a: number, b: number) => void;
    readonly __wbg_searcher_free: (a: number, b: number) => void;
    readonly __wbg_slabjob_free: (a: number, b: number) => void;
    readonly __wbg_treekmeansjob_free: (a: number, b: number) => void;
    readonly accum_record_range: (a: number, b: number, c: number) => [number, number];
    readonly assignjob_add_batch: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly assignjob_add_batch_prev: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly assignjob_assigned_records: (a: number) => number;
    readonly assignjob_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number];
    readonly assignjob_part_len: (a: number, b: number) => number;
    readonly assignjob_pq_m: (a: number) => number;
    readonly assignjob_set_codebook: (a: number, b: number, c: number) => [number, number];
    readonly assignjob_set_tree: (a: number, b: number, c: number, d: number) => [number, number];
    readonly assignjob_take_part: (a: number, b: number) => [number, number];
    readonly assignjob_with_candidates: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly build_centroid_tree: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly cell_neighbors: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly centroid_tree_info: (a: number, b: number) => [number, number, number, number];
    readonly delta_cells: (a: number, b: number) => [number, number, number, number];
    readonly delta_info: (a: number, b: number) => [number, number, number, number];
    readonly delta_payload: (a: number, b: number, c: number) => [number, number, number, number];
    readonly delta_records_for_cell: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly deltajob_add: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
    readonly deltajob_add_with_payloads: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number, number, number];
    readonly deltajob_byte_len: (a: number) => number;
    readonly deltajob_finish: (a: number) => [number, number];
    readonly deltajob_len: (a: number) => number;
    readonly deltajob_new: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly deltajob_pq_m: (a: number) => number;
    readonly deltajob_set_codebook: (a: number, b: number, c: number) => [number, number];
    readonly deltajob_set_pq_codebook: (a: number, b: number) => [number, number];
    readonly deltajob_set_tree: (a: number, b: number, c: number, d: number) => [number, number];
    readonly filter_hits: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly fineinitjob_add_batch: (a: number, b: number, c: number) => [number, number, number];
    readonly fineinitjob_finish: (a: number) => [number, number];
    readonly fineinitjob_new: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly flatkmeansjob_add_batch: (a: number, b: number, c: number) => [number, number, number];
    readonly flatkmeansjob_means: (a: number, b: number, c: number) => [number, number];
    readonly flatkmeansjob_new: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly flatkmeansjob_partial: (a: number) => [number, number];
    readonly inckmeansjob_add_batch: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly inckmeansjob_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly inckmeansjob_partial: (a: number) => [number, number];
    readonly init_panic_hook: () => void;
    readonly merge: (a: number, b: number, c: number) => [number, number];
    readonly normalize_vector: (a: number, b: number) => [number, number];
    readonly part_directory: (a: number, b: number) => [number, number, number, number];
    readonly part_directory_len: (a: number) => number;
    readonly part_span: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly pq_codebook_info: (a: number, b: number) => [number, number, number, number];
    readonly pq_init_codebook: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly pq_reduce: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly pqcodebook_dim: (a: number) => number;
    readonly pqcodebook_m: (a: number) => number;
    readonly pqcodebook_new: (a: number, b: number) => [number, number, number];
    readonly pqreducer_add_partial: (a: number, b: number, c: number) => [number, number];
    readonly pqreducer_empty_centroids: (a: number) => number;
    readonly pqreducer_finish: (a: number, b: number, c: number) => [number, number, number, number];
    readonly pqreducer_mean_sq_err: (a: number) => number;
    readonly pqreducer_new: (a: number, b: number) => [number, number, number];
    readonly pqreducer_rows: (a: number) => number;
    readonly pqtable_from_codebook: (a: number, b: number, c: number) => [number, number, number];
    readonly pqtable_m: (a: number) => number;
    readonly pqtable_new: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly pqtrainjob_add_residuals: (a: number, b: number, c: number) => [number, number, number];
    readonly pqtrainjob_mean_sq_err: (a: number) => number;
    readonly pqtrainjob_new: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly pqtrainjob_partial: (a: number) => [number, number];
    readonly pqtrainjob_rows: (a: number) => number;
    readonly reducer_add_records: (a: number, b: number, c: number) => [number, number];
    readonly reducer_counts: (a: number) => [number, number];
    readonly reducer_last_reseeded: (a: number) => number;
    readonly reducer_means: (a: number, b: number, c: number) => [number, number, number, number];
    readonly reducer_means_reseed: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly reducer_new: (a: number, b: number) => number;
    readonly reencode_records: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number, number];
    readonly rerank: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
    readonly residuals_for: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
    readonly scan: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: any) => [number, number, number, number];
    readonly scan_delta: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: any) => [number, number, number, number];
    readonly scan_pq: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: any) => [number, number, number, number];
    readonly searcher_cell_dots: (a: number, b: number, c: number) => [number, number, number, number];
    readonly searcher_cell_dots_for: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly searcher_dim: (a: number) => number;
    readonly searcher_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly searcher_route: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly searcher_route_beam: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly searcher_route_flat: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly searcher_route_from_dots: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly searcher_set_tree: (a: number, b: number, c: number) => [number, number];
    readonly searcher_top_coarse: (a: number, b: number, c: number, d: number) => [number, number];
    readonly searcher_tree_groups: (a: number) => number;
    readonly searcher_tree_recall: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly searcher_tree_recall_top: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly slab_centroid: (a: number, b: number) => [number, number, number, number];
    readonly slab_info: (a: number, b: number) => [number, number, number, number];
    readonly slab_scan_len: (a: number, b: number) => [number, number, number];
    readonly slab_to_records: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly slabjob_add_records: (a: number, b: number, c: number) => [number, number, number];
    readonly slabjob_add_records_range: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly slabjob_byte_len: (a: number) => number;
    readonly slabjob_finish: (a: number) => [number, number];
    readonly slabjob_format: (a: number) => number;
    readonly slabjob_len: (a: number) => number;
    readonly slabjob_new: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly split_part_records: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number, number];
    readonly split_slab: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly split_slab_records: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number, number];
    readonly tombstones_merge: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly treekmeansjob_add_batch: (a: number, b: number, c: number) => [number, number, number];
    readonly treekmeansjob_add_batch_assign: (a: number, b: number, c: number) => [number, number, number, number];
    readonly treekmeansjob_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly treekmeansjob_partial: (a: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
