// Dense-side file written before rustfmt was adopted; kept as is (`cargo fmt` skips it).
#![cfg_attr(rustfmt, rustfmt::skip)]
//! wasm-bindgen surface over `thread-core`. Pure compute; the host (a Cloudflare
//! Worker in TypeScript) does all storage I/O and hands byte buffers in. The dense
//! side is below; the lexical side (tokenizer, segments, paged reader, fusion,
//! excerpts) is in `lex.rs`.
//!
//! Vector batches are `Float32Array`s of `n * dim` unit vectors.

use thread_core::delta::{DeltaEncoder, DeltaView};
use thread_core::kmeans::{
    assign_candidates, assign_candidates_top_m, assign_flat, assign_tree, assign_tree_top_m, check_neighbors,
    effective_probe_coarse, Accum, BucketReservoir,
};
use thread_core::math::{inv_norms, normalize};
use thread_core::part::{span_sub_range, DirEntry, PartBuilder};
use thread_core::pq::{Codebook, PqAccum, QueryTable};
use thread_core::quant::{decode_residual_sq8, encode_residual_sq4, encode_residual_sq8};
use thread_core::route::{CentroidTree, Router};
use thread_core::routing::{decode_rows, encode_grouped, encode_rows, Elem, GroupedTable, PartialRouter};
use thread_core::search::{rerank_records, scan_delta as scan_delta_core, scan_slab, scan_slab_pq, ScanStats};
use thread_core::slab::{SlabHeader, SlabView, SlabWriter, SLAB_HEADER};
use thread_core::tomb::Tombstones;
use thread_core::topk::{merge_dedup, pack_hits, unpack_hits, TopK};
use wasm_bindgen::prelude::*;

mod lex;
pub use lex::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

/// Installs a panic hook that logs the panic message (and location) via
/// `console.error`, so panics show up in Workers logs before the wasm traps.
/// Idempotent; call once after instantiating the module. After a trap
/// (`WebAssembly.RuntimeError`) the instance is poisoned: call
/// `__wbg_reset_state()` and re-init before using it again.
#[wasm_bindgen]
pub fn init_panic_hook() {
    use std::sync::Once;
    static SET: Once = Once::new();
    SET.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            error(&format!("thread-wasm panic: {info}"));
        }));
    });
}

fn js_err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn check_batch(len: usize, dim: usize) -> Result<usize, JsValue> {
    if dim == 0 || len % dim != 0 {
        return Err(js_err(format!("batch length {len} is not a multiple of dim {dim}")));
    }
    Ok(len / dim)
}

/// Parses tree bytes and checks they index `k` cells of dimension `dim`.
fn tree_arg(tree_bytes: &[u8], dim: usize, k: usize) -> Result<CentroidTree, JsValue> {
    let t = CentroidTree::from_bytes(tree_bytes).map_err(js_err)?;
    if t.dim != dim || t.k != k {
        return Err(js_err(format!("tree is for k = {} cells of dim {}, expected k = {k}, dim = {dim}", t.k, t.dim)));
    }
    Ok(t)
}

/// Builds a centroid tree ("VIBC" bytes) over `k = means.len() / dim` cell
/// means (raw, e.g. `fine.f32`): spherical k-means of the means into
/// `groups` groups (clamped to `1..=k`), k-means++ seeding, up to `iters`
/// Lloyd iterations, deterministic for `seed`. Lookups (`Searcher.set_tree`,
/// `DeltaJob.set_tree`, `AssignJob.set_tree`) score the group means, then only
/// the cells of the best `beam` groups. Cost: about `(iters + 2) * groups * k * dim`
/// multiply-adds (k = 4096, dim = 1536, groups = 64, iters = 10: ~5e9).
#[wasm_bindgen]
pub fn build_centroid_tree(means: &[f32], dim: u32, groups: u32, iters: u32, seed: u32) -> Result<Vec<u8>, JsValue> {
    let dim = dim as usize;
    let k = check_batch(means.len(), dim)?;
    if k == 0 {
        return Err(js_err("means is empty"));
    }
    Ok(CentroidTree::build(means, dim, k, groups as usize, iters as usize, seed as u64).to_bytes())
}

/// Header summary of centroid tree bytes: `[dim, k, groups, n_members, byte_len]`.
#[wasm_bindgen]
pub fn centroid_tree_info(tree_bytes: &[u8]) -> Result<Vec<u32>, JsValue> {
    let t = CentroidTree::from_bytes(tree_bytes).map_err(js_err)?;
    Ok(vec![t.dim as u32, t.k as u32, t.groups() as u32, t.n_members() as u32, tree_bytes.len() as u32])
}

/// Returns a unit-normalized copy of `q`.
#[wasm_bindgen]
pub fn normalize_vector(q: &[f32]) -> Vec<f32> {
    let mut v = q.to_vec();
    normalize(&mut v);
    v
}

/// Flat k-means map job over `k` centroids (used for the coarse level).
#[wasm_bindgen]
pub struct FlatKmeansJob {
    dim: usize,
    centroids: Vec<f32>,
    inv: Vec<f32>,
    accum: Accum,
}

#[wasm_bindgen]
impl FlatKmeansJob {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: u32, k: u32, centroid_means: &[f32]) -> Result<FlatKmeansJob, JsValue> {
        let (dim, k) = (dim as usize, k as usize);
        if centroid_means.len() != k * dim {
            return Err(js_err("centroid buffer size mismatch"));
        }
        Ok(FlatKmeansJob {
            dim,
            centroids: centroid_means.to_vec(),
            inv: inv_norms(centroid_means, dim),
            accum: Accum::new(k, dim),
        })
    }

    /// Assigns each vector in the batch and accumulates. Returns vectors seen.
    pub fn add_batch(&mut self, vectors: &[f32]) -> Result<u32, JsValue> {
        let n = check_batch(vectors.len(), self.dim)?;
        for x in vectors.chunks_exact(self.dim) {
            let c = assign_flat(x, &self.centroids, &self.inv, self.dim);
            self.accum.add(c, x);
        }
        Ok(n as u32)
    }

    /// Serialized partial accumulator.
    pub fn partial(&self) -> Vec<u8> {
        self.accum.to_bytes()
    }

    /// Convenience for single-job k-means: new means from this job's accumulator alone.
    pub fn means(&self, prev: &[f32]) -> Vec<f32> {
        self.accum.means(prev)
    }
}

/// Picks initial fine centroids: a reservoir of up to `k2` sample vectors per
/// coarse group, streamed.
#[wasm_bindgen]
pub struct FineInitJob {
    dim: usize,
    coarse_mean: Vec<f32>,
    coarse_inv: Vec<f32>,
    reservoir: BucketReservoir,
}

#[wasm_bindgen]
impl FineInitJob {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: u32, k1: u32, k2: u32, coarse_means: &[f32], seed: u32) -> Result<FineInitJob, JsValue> {
        let (dim, k1, k2) = (dim as usize, k1 as usize, k2 as usize);
        if coarse_means.len() != k1 * dim {
            return Err(js_err("coarse centroid buffer size mismatch"));
        }
        Ok(FineInitJob {
            dim,
            coarse_mean: coarse_means.to_vec(),
            coarse_inv: inv_norms(coarse_means, dim),
            reservoir: BucketReservoir::new(k1, k2, dim, seed as u64),
        })
    }

    pub fn add_batch(&mut self, vectors: &[f32]) -> Result<u32, JsValue> {
        let n = check_batch(vectors.len(), self.dim)?;
        for x in vectors.chunks_exact(self.dim) {
            let g = assign_flat(x, &self.coarse_mean, &self.coarse_inv, self.dim);
            self.reservoir.offer(g, x);
        }
        Ok(n as u32)
    }

    /// `k1 * k2 * dim` initial fine means, grouped by coarse group.
    pub fn finish(self) -> Vec<f32> {
        self.reservoir.finish(&self.coarse_mean)
    }
}

/// Two-level k-means map job: assigns each vector to its best fine cell among
/// the `probe_coarse` nearest coarse groups (`0` or `>= k1` = all groups, as
/// `Searcher.route`) and accumulates per fine cell.
#[wasm_bindgen]
pub struct TreeKmeansJob {
    router: Router,
    probe_coarse: usize,
    accum: Accum,
}

#[wasm_bindgen]
impl TreeKmeansJob {
    #[wasm_bindgen(constructor)]
    pub fn new(
        dim: u32,
        k1: u32,
        k2: u32,
        coarse_means: &[f32],
        fine_means: &[f32],
        probe_coarse: u32,
    ) -> Result<TreeKmeansJob, JsValue> {
        let (dim, k1, k2) = (dim as usize, k1 as usize, k2 as usize);
        if coarse_means.len() != k1 * dim || fine_means.len() != k1 * k2 * dim {
            return Err(js_err("centroid buffer size mismatch"));
        }
        let router = Router::new(dim, k1, k2, coarse_means, fine_means.to_vec());
        // 0 (or >= k1) = every coarse group, as Searcher.route.
        let probe_coarse = effective_probe_coarse(probe_coarse as usize, k1);
        Ok(TreeKmeansJob { router, probe_coarse, accum: Accum::new(k1 * k2, dim) })
    }

    pub fn add_batch(&mut self, vectors: &[f32]) -> Result<u32, JsValue> {
        let n = check_batch(vectors.len(), self.router.dim)?;
        self.assign_accumulate(vectors, |_| {});
        Ok(n as u32)
    }

    /// Like `add_batch`, but returns the chosen global fine cell of every
    /// vector (with `k1 = 1` a flat assignment), e.g. to seed `IncKmeansJob`.
    pub fn add_batch_assign(&mut self, vectors: &[f32]) -> Result<Vec<u32>, JsValue> {
        let n = check_batch(vectors.len(), self.router.dim)?;
        let mut out = Vec::with_capacity(n);
        self.assign_accumulate(vectors, |c| out.push(c as u32));
        Ok(out)
    }

    pub fn partial(&self) -> Vec<u8> {
        self.accum.to_bytes()
    }
}

impl TreeKmeansJob {
    fn assign_accumulate(&mut self, vectors: &[f32], mut emit: impl FnMut(usize)) {
        let dim = self.router.dim;
        for x in vectors.chunks_exact(dim) {
            let cell = assign_tree(
                x,
                self.router.coarse_mean(),
                self.router.coarse_inv(),
                self.router.fine_mean_all(),
                self.router.fine_inv(),
                self.router.k2,
                dim,
                self.probe_coarse,
            );
            self.accum.add(cell, x);
            emit(cell);
        }
    }
}

/// Validates a per-vector previous-assignment array against a batch.
fn check_prev(prev: &[u32], n: usize, k: usize) -> Result<(), JsValue> {
    if prev.len() != n {
        return Err(js_err(format!("prev has {} entries for {n} vectors", prev.len())));
    }
    if let Some(p) = prev.iter().find(|&&p| p as usize >= k) {
        return Err(js_err(format!("prev cell {p} out of range (k = {k})")));
    }
    Ok(())
}

/// Incremental flat k-means map job: each vector is compared only against the
/// `n_neigh` neighbour cells of its previous cell (`neighbors`, a `k x n_neigh`
/// table from `cell_neighbors`), not all `k` centroids. Accumulates like
/// `TreeKmeansJob`; `partial()` has the same format, so `Reducer` is unchanged.
#[wasm_bindgen]
pub struct IncKmeansJob {
    dim: usize,
    k: usize,
    n_neigh: usize,
    means: Vec<f32>,
    inv: Vec<f32>,
    neighbors: Vec<u32>,
    accum: Accum,
}

#[wasm_bindgen]
impl IncKmeansJob {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: u32, k: u32, means: &[f32], neighbors: &[u32], n_neigh: u32) -> Result<IncKmeansJob, JsValue> {
        let (dim, k, n_neigh) = (dim as usize, k as usize, n_neigh as usize);
        if dim == 0 || means.len() != k * dim {
            return Err(js_err("centroid buffer size mismatch"));
        }
        check_neighbors(neighbors, k, n_neigh).map_err(js_err)?;
        Ok(IncKmeansJob {
            dim,
            k,
            n_neigh,
            means: means.to_vec(),
            inv: inv_norms(means, dim),
            neighbors: neighbors.to_vec(),
            accum: Accum::new(k, dim),
        })
    }

    /// Assigns vector `i` to its best cell among the neighbours of `prev[i]`,
    /// accumulates, and returns the new cell per vector. Nothing is
    /// accumulated if the arguments are invalid.
    pub fn add_batch(&mut self, vectors: &[f32], prev: &[u32]) -> Result<Vec<u32>, JsValue> {
        let n = check_batch(vectors.len(), self.dim)?;
        check_prev(prev, n, self.k)?;
        let (dim, nn) = (self.dim, self.n_neigh);
        let mut out = Vec::with_capacity(n);
        for (x, &p) in vectors.chunks_exact(dim).zip(prev) {
            let p = p as usize;
            let cands = &self.neighbors[p * nn..(p + 1) * nn];
            let c = assign_candidates(x, &self.means, &self.inv, dim, cands);
            self.accum.add(c, x);
            out.push(c as u32);
        }
        Ok(out)
    }

    pub fn partial(&self) -> Vec<u8> {
        self.accum.to_bytes()
    }
}

/// Cell neighbour table rows for cells `[first, first + count)`: the `n`
/// cells with the highest cosine between means, each row starting with the
/// cell itself, flattened row-major (`count * n` ids). `means` is the full
/// `k * dim` table; shard the work over `first/count` ranges.
#[wasm_bindgen]
pub fn cell_neighbors(means: &[f32], dim: u32, first: u32, count: u32, n: u32) -> Result<Vec<u32>, JsValue> {
    let (dim, first, count, n) = (dim as usize, first as usize, count as usize, n as usize);
    if dim == 0 || means.len() % dim != 0 {
        return Err(js_err("means length is not a multiple of dim"));
    }
    let k = means.len() / dim;
    if first.checked_add(count).map_or(true, |e| e > k) {
        return Err(js_err(format!("cells [{first}, {first}+{count}) out of range (k = {k})")));
    }
    if n == 0 || n > k {
        return Err(js_err(format!("n = {n} must be in 1..={k}")));
    }
    Ok(thread_core::kmeans::cell_neighbors(means, dim, first, count, n))
}

/// Byte range `[offset, length]` inside a serialized partial accumulator that
/// covers cells `[first, first + n)`.
#[wasm_bindgen]
pub fn accum_record_range(dim: u32, first: u32, n: u32) -> Vec<u32> {
    let (o, l) = Accum::record_range(dim as usize, first as usize, n as usize);
    vec![o as u32, l as u32]
}

/// Reduce job over a cell range: sums partial records, then emits new means.
#[wasm_bindgen]
pub struct Reducer {
    accum: Accum,
    last_reseeded: u32,
}

#[wasm_bindgen]
impl Reducer {
    #[wasm_bindgen(constructor)]
    pub fn new(cells: u32, dim: u32) -> Reducer {
        Reducer { accum: Accum::new(cells as usize, dim as usize), last_reseeded: 0 }
    }

    /// Adds the raw records for this reducer's cell range from one partial.
    pub fn add_records(&mut self, records: &[u8]) -> Result<(), JsValue> {
        self.accum.add_records(records, 0).map_err(js_err)
    }

    /// Per-cell counts after reduction (useful to spot empty cells).
    pub fn counts(&self) -> Vec<u32> {
        self.accum.counts.clone()
    }

    /// New means; `prev` is the previous means for the same cell range.
    pub fn means(&self, prev: &[f32]) -> Result<Vec<f32>, JsValue> {
        if prev.len() != self.accum.k * self.accum.dim {
            return Err(js_err("prev centroid buffer size mismatch"));
        }
        Ok(self.accum.means(prev))
    }

    /// Like `means`, but cells with fewer than `min_count` members are reseeded
    /// next to the densest cells of this range (deterministic jitter from `seed`).
    /// The number of reseeded cells is available from `last_reseeded()`.
    pub fn means_reseed(&mut self, prev: &[f32], min_count: u32, seed: u32) -> Result<Vec<f32>, JsValue> {
        if prev.len() != self.accum.k * self.accum.dim {
            return Err(js_err("prev centroid buffer size mismatch"));
        }
        let (m, n) = self.accum.means_reseed(prev, min_count, seed as u64);
        self.last_reseeded = n;
        Ok(m)
    }

    /// Cells reseeded by the last `means_reseed` call.
    pub fn last_reseeded(&self) -> u32 {
        self.last_reseeded
    }
}

/// Assign job: multi-assigns each vector to its `m` nearest fine cells (within
/// the `probe_coarse` best coarse groups; `0` or `>= k1` = all groups), encodes
/// the residual against each cell's mean in both tiers (4-bit MSE-clipped scan
/// code + int8 rerank code), and buffers records per coarse group. After
/// `set_codebook`, records also carry the residual's PQ codes (needed to
/// build format 3 slabs); without it their `pq_m` is 0.
#[wasm_bindgen]
pub struct AssignJob {
    router: Router,
    probe_coarse: usize,
    m: usize,
    /// `with_candidates` only: `k x n_neigh` neighbour table.
    neighbors: Vec<u32>,
    n_neigh: usize,
    parts: Vec<PartBuilder>,
    code4: Vec<u8>,
    code8: Vec<i8>,
    scratch: Vec<f32>,
    assigned: u64,
    codebook: Option<Codebook>,
    pq: Vec<u8>,
    resid: Vec<f32>,
    /// `set_tree`: pruned top-m search for `add_batch`.
    tree: Option<(CentroidTree, usize)>,
}

#[wasm_bindgen]
impl AssignJob {
    #[wasm_bindgen(constructor)]
    pub fn new(
        dim: u32,
        k1: u32,
        k2: u32,
        coarse_means: &[f32],
        fine_means: &[f32],
        probe_coarse: u32,
        m: u32,
    ) -> Result<AssignJob, JsValue> {
        let (dim, k1, k2) = (dim as usize, k1 as usize, k2 as usize);
        if coarse_means.len() != k1 * dim || fine_means.len() != k1 * k2 * dim {
            return Err(js_err("centroid buffer size mismatch"));
        }
        if dim % 2 != 0 {
            return Err(js_err("dim must be even for 4-bit codes"));
        }
        let router = Router::new(dim, k1, k2, coarse_means, fine_means.to_vec());
        Ok(AssignJob {
            router,
            // 0 (or >= k1) = every coarse group, as Searcher.route.
            probe_coarse: effective_probe_coarse(probe_coarse as usize, k1),
            m: (m as usize).max(1),
            neighbors: Vec::new(),
            n_neigh: 0,
            parts: (0..k1).map(|_| PartBuilder::new(k2, dim)).collect(),
            code4: vec![0; dim / 2],
            code8: vec![0; dim],
            scratch: vec![0.0; dim],
            assigned: 0,
            codebook: None,
            pq: Vec::new(),
            resid: Vec::new(),
            tree: None,
        })
    }

    /// Flat (`k1 = 1`: one part group, `k2 = k`) assign job whose
    /// `add_batch_prev` picks each vector's top `m` cells among the `n_neigh`
    /// neighbours (`neighbors`, a `k x n_neigh` table from `cell_neighbors`)
    /// of its previous cell. Requires `n_neigh >= m`. `add_batch` also works
    /// (exhaustive flat top-m).
    pub fn with_candidates(
        dim: u32,
        k: u32,
        means: &[f32],
        neighbors: &[u32],
        n_neigh: u32,
        m: u32,
    ) -> Result<AssignJob, JsValue> {
        let (kk, nn) = (k as usize, n_neigh as usize);
        check_neighbors(neighbors, kk, nn).map_err(js_err)?;
        if m == 0 || nn < m as usize {
            return Err(js_err(format!("need 1 <= m <= n_neigh (m = {m}, n_neigh = {nn})")));
        }
        // k1 = 1: the single coarse "centroid" is never ranked against anything.
        let coarse = vec![1.0f32; dim as usize];
        let mut job = AssignJob::new(dim, 1, k, &coarse, means, 1, m)?;
        job.neighbors = neighbors.to_vec();
        job.n_neigh = nn;
        Ok(job)
    }

    /// Vectors have ids `first_id, first_id + 1, ...`. Payloads are empty.
    pub fn add_batch(&mut self, vectors: &[f32], first_id: u32) -> Result<u32, JsValue> {
        let dim = self.router.dim;
        let n = check_batch(vectors.len(), dim)?;
        for (i, x) in vectors.chunks_exact(dim).enumerate() {
            if let Some((tree, beam)) = &self.tree {
                let cells = tree.top_m(x, self.router.fine_mean_all(), self.router.fine_inv(), *beam, self.m);
                self.push_cells(x, first_id + i as u32, &cells);
                continue;
            }
            let cells = assign_tree_top_m(
                x,
                self.router.coarse_mean(),
                self.router.coarse_inv(),
                self.router.fine_mean_all(),
                self.router.fine_inv(),
                self.router.k2,
                dim,
                self.probe_coarse,
                self.m,
            );
            self.push_cells(x, first_id + i as u32, &cells);
        }
        Ok(n as u32)
    }

    /// `with_candidates` jobs only: like `add_batch`, but vector `i`'s top `m`
    /// cells are chosen among the neighbours of `prev[i]` (its cell from the
    /// last k-means iteration). Returns vectors seen.
    pub fn add_batch_prev(&mut self, vectors: &[f32], first_id: u32, prev: &[u32]) -> Result<u32, JsValue> {
        if self.n_neigh == 0 {
            return Err(js_err("add_batch_prev needs a job built with AssignJob.with_candidates"));
        }
        let dim = self.router.dim;
        let n = check_batch(vectors.len(), dim)?;
        let k = self.router.cells();
        check_prev(prev, n, k)?;
        let nn = self.n_neigh;
        for (i, (x, &p)) in vectors.chunks_exact(dim).zip(prev).enumerate() {
            let p = p as usize;
            let cells = assign_candidates_top_m(
                x,
                self.router.fine_mean_all(),
                self.router.fine_inv(),
                dim,
                &self.neighbors[p * nn..(p + 1) * nn],
                self.m,
            );
            self.push_cells(x, first_id + i as u32, &cells);
        }
        Ok(n as u32)
    }

    /// Makes every following record carry the PQ codes of its residual under
    /// this codebook (`pq_init_codebook` / `pq_reduce` bytes; its dim must be the
    /// job's dim). Works for jobs from either constructor; call before adding.
    pub fn set_codebook(&mut self, codebook_bytes: &[u8]) -> Result<(), JsValue> {
        let cb = Codebook::from_bytes(codebook_bytes).map_err(js_err)?;
        if cb.dim != self.router.dim {
            return Err(js_err(format!("codebook dim {} != job dim {}", cb.dim, self.router.dim)));
        }
        self.pq = vec![0; cb.m];
        self.resid = vec![0.0; cb.dim];
        self.codebook = Some(cb);
        Ok(())
    }

    /// PQ codes per record (0 without a codebook).
    pub fn pq_m(&self) -> u32 {
        self.pq.len() as u32
    }

    /// Makes `add_batch` pick each vector's top `m` cells with a pruned search:
    /// `tree_bytes` (`build_centroid_tree` over all `k1 * k2` fine means, global
    /// cell ids) scores its group means and ranks only the cells of the best
    /// `beam` groups; `probe_coarse` is then ignored. With `beam >= groups` the
    /// parts equal those of the exhaustive flat search (`probe_coarse = k1`, or
    /// `k1 = 1`). `add_batch_prev` is unaffected. Call before adding.
    pub fn set_tree(&mut self, tree_bytes: &[u8], beam: u32) -> Result<(), JsValue> {
        let t = tree_arg(tree_bytes, self.router.dim, self.router.cells())?;
        self.tree = Some((t, (beam as usize).max(1)));
        Ok(())
    }

    pub fn assigned_records(&self) -> f64 {
        self.assigned as f64
    }

    /// Current serialized size of part `g`.
    pub fn part_len(&self, g: u32) -> u32 {
        self.parts[g as usize].byte_len() as u32
    }

    /// Serializes part `g` and resets its buffer.
    pub fn take_part(&mut self, g: u32) -> Vec<u8> {
        self.parts[g as usize].take()
    }
}

impl AssignJob {
    /// Encodes `x` against each cell's mean in both tiers and appends the records.
    fn push_cells(&mut self, x: &[f32], id: u32, cells: &[(usize, f32)]) {
        for &(cell, _) in cells {
            let mean = self.router.fine_mean(cell);
            let scale4 = encode_residual_sq4(x, mean, &mut self.code4, &mut self.scratch);
            let scale8 = encode_residual_sq8(x, mean, &mut self.code8);
            if let Some(cb) = &self.codebook {
                for i in 0..x.len() {
                    self.resid[i] = x[i] - mean[i];
                }
                thread_core::pq::encode(&self.resid, cb, &mut self.pq);
            }
            let g = cell / self.router.k2;
            let j = cell % self.router.k2;
            self.parts[g].push(j, id, scale4, scale8, 0, &self.code4, &self.code8, &self.pq, &[]);
            self.assigned += 1;
        }
    }
}

/// Finalize job for one fine cell: concatenates part record ranges into a slab.
#[wasm_bindgen]
pub struct SlabJob {
    dim: usize,
    builder: SlabWriter,
}

#[wasm_bindgen]
impl SlabJob {
    /// `format`: 2 (default when omitted) = v2 slab (4-bit scan tier + int8
    /// rerank tier); 1 = v1 int8-only slab (from the records' int8 codes);
    /// 3 = v3 slab (the records' PQ codes as scan tier + int8 rerank tier; the
    /// records must come from an `AssignJob` with `set_codebook`, otherwise
    /// `add_records*` errors).
    #[wasm_bindgen(constructor)]
    pub fn new(dim: u32, cell: u32, centroid_mean: &[f32], format: Option<u8>) -> Result<SlabJob, JsValue> {
        if centroid_mean.len() != dim as usize {
            return Err(js_err("centroid size mismatch"));
        }
        if dim % 2 != 0 {
            return Err(js_err("dim must be even (part records carry 4-bit codes)"));
        }
        let builder = SlabWriter::new(format.unwrap_or(2), dim as usize, cell, centroid_mean).map_err(js_err)?;
        Ok(SlabJob { dim: dim as usize, builder })
    }

    /// Slab format version this job writes (1, 2 or 3).
    pub fn format(&self) -> u32 {
        self.builder.version() as u32
    }

    /// Adds every record in a records byte range (one cell's directory range,
    /// or any run of whole records, e.g. a cell sliced out of a `part_span`).
    pub fn add_records(&mut self, bytes: &[u8]) -> Result<u32, JsValue> {
        self.builder.add_part_records(bytes, self.dim).map_err(js_err)
    }

    /// Adds the records of one cell from a fetched multi-cell span without
    /// copying it in JS: `span` holds part bytes starting at part offset
    /// `span_offset`; the cell's directory entry is `[cell_off, cell_len)`
    /// (part offsets). Adds `span[cell_off - span_offset, +cell_len)`.
    pub fn add_records_range(&mut self, span: &[u8], span_offset: u32, cell_off: u32, cell_len: u32) -> Result<u32, JsValue> {
        let r = span_sub_range(span.len(), span_offset, cell_off, cell_len).map_err(js_err)?;
        self.builder.add_part_records(&span[r], self.dim).map_err(js_err)
    }

    pub fn len(&self) -> u32 {
        self.builder.len() as u32
    }

    pub fn byte_len(&self) -> u32 {
        self.builder.byte_len() as u32
    }

    pub fn finish(self) -> Vec<u8> {
        self.builder.finish()
    }
}

/// Parses a part directory: returns `[k2, dim, count0, off0, len0, count1, ...]`.
#[wasm_bindgen]
pub fn part_directory(bytes: &[u8]) -> Result<Vec<u32>, JsValue> {
    let (k2, dim, dir) = thread_core::part::parse_directory(bytes).map_err(js_err)?;
    let mut out = Vec::with_capacity(2 + dir.len() * 3);
    out.push(k2 as u32);
    out.push(dim as u32);
    for d in dir {
        out.push(d.count);
        out.push(d.off);
        out.push(d.len);
    }
    Ok(out)
}

/// Byte span `[offset, length]` of one part covering cells
/// `[first_cell, first_cell + count)`, from `dir` = the output of
/// `part_directory`. Empty cells at the edges are skipped; no records gives
/// `[0, 0]`. Slice each cell out of the fetched span with its directory entry
/// (`off - offset`), or use `SlabJob.add_records_range`.
#[wasm_bindgen]
pub fn part_span(dir: &[u32], first_cell: u32, count: u32) -> Result<Vec<u32>, JsValue> {
    if dir.len() < 2 || (dir.len() - 2) % 3 != 0 || (dir.len() - 2) / 3 != dir[0] as usize {
        return Err(js_err("dir is not a part_directory() result"));
    }
    let entries: Vec<DirEntry> =
        dir[2..].chunks_exact(3).map(|e| DirEntry { count: e[0], off: e[1], len: e[2] }).collect();
    let (off, len) = thread_core::part::span(&entries, first_cell as usize, count as usize).map_err(js_err)?;
    Ok(vec![off, len])
}

#[wasm_bindgen]
pub fn part_directory_len(k2: u32) -> u32 {
    thread_core::part::directory_len(k2 as usize) as u32
}

fn elem_arg(elem: &str) -> Result<Elem, JsValue> {
    Elem::parse(elem).ok_or_else(|| js_err(format!("unknown routing table element type {elem:?} (f32, f16 or i8)")))
}

/// Encodes cell means (`k * dim` f32, e.g. `fine.f32`) as a compact routing
/// table: `"f16"` (IEEE half, `fine.f16`), `"i8"` (`[scale f32][i8 * dim]` per
/// row, `fine.i8`) or `"f32"` (the raw bytes).
#[wasm_bindgen]
pub fn routing_table_encode(means: &[f32], dim: u32, elem: &str) -> Result<Vec<u8>, JsValue> {
    let elem = elem_arg(elem)?;
    check_batch(means.len(), dim as usize)?;
    Ok(encode_rows(means, dim as usize, elem))
}

/// Decodes a routing table of `elem` rows back to f32 means (what
/// `Searcher.from_table` routes over).
#[wasm_bindgen]
pub fn routing_table_decode(table: &[u8], dim: u32, elem: &str) -> Result<Vec<f32>, JsValue> {
    decode_rows(table, dim as usize, elem_arg(elem)?).map_err(js_err)
}

/// Bytes of one routing table row of `elem` at `dim`.
#[wasm_bindgen]
pub fn routing_row_bytes(dim: u32, elem: &str) -> Result<u32, JsValue> {
    Ok(elem_arg(elem)?.row_bytes(dim as usize) as u32)
}

/// The means (`k * dim` f32) in the group order of a centroid tree built over
/// them ("VIBG": header, directory, rows of `elem`), so a cold reader can
/// range-read the rows of the groups a query needs (`ColdRouter`).
#[wasm_bindgen]
pub fn grouped_table_encode(means: &[f32], dim: u32, tree_bytes: &[u8], elem: &str) -> Result<Vec<u8>, JsValue> {
    let elem = elem_arg(elem)?;
    let k = check_batch(means.len(), dim as usize)?;
    let tree = tree_arg(tree_bytes, dim as usize, k)?;
    Ok(encode_grouped(means, dim as usize, &tree, elem))
}

/// Header of a grouped table (its first `data_off` bytes or the whole object):
/// `[elem code (0 f32, 1 f16, 2 i8), dim, k, groups, n_rows, data_off, row_bytes, byte_len]`.
#[wasm_bindgen]
pub fn grouped_table_info(bytes: &[u8]) -> Result<Vec<u32>, JsValue> {
    let t = GroupedTable::parse(bytes).map_err(js_err)?;
    Ok(vec![t.elem.code(), t.dim as u32, t.k as u32, t.groups as u32, t.n_rows as u32, t.data_off as u32, t.row_bytes() as u32, t.byte_len() as u32])
}

/// Cold-start routing over a subset of the cell means: a centroid tree plus
/// the rows of the groups loaded so far from a tree-grouped table
/// (`fine.grouped.*`). `missing_groups(q, beam)` names the groups to read
/// (`group_rows` gives their rows, so their byte range is `data_off + first *
/// row_bytes`), `add_rows` / `add_group` installs them, `route_beam` then ranks
/// the members of the best `beam` groups as `Searcher.route_beam` does over
/// the full table. Cells not loaded are not candidates. `cell_dots_for` gives
/// delta scans their base scores (load the delta cells' rows first:
/// `missing_cells`, `cell_row`).
#[wasm_bindgen]
pub struct ColdRouter {
    inner: PartialRouter,
}

#[wasm_bindgen]
impl ColdRouter {
    /// `tree_bytes`: the centroid tree the grouped table was written in the
    /// order of; `elem`: its row type.
    #[wasm_bindgen(constructor)]
    pub fn new(tree_bytes: &[u8], elem: &str) -> Result<ColdRouter, JsValue> {
        let elem = elem_arg(elem)?;
        let tree = CentroidTree::from_bytes(tree_bytes).map_err(js_err)?;
        Ok(ColdRouter { inner: PartialRouter::new(tree, elem) })
    }

    pub fn dim(&self) -> u32 {
        self.inner.dim as u32
    }

    /// Cells (rows of the full table).
    pub fn k(&self) -> u32 {
        self.inner.k as u32
    }

    pub fn groups(&self) -> u32 {
        self.inner.groups() as u32
    }

    pub fn row_bytes(&self) -> u32 {
        self.inner.row_bytes() as u32
    }

    /// `[first row, row count]` of group `g` in the grouped table.
    pub fn group_rows(&self, g: u32) -> Result<Vec<u32>, JsValue> {
        if g as usize >= self.inner.groups() {
            return Err(js_err(format!("group {g} >= {} groups", self.inner.groups())));
        }
        let (a, n) = self.inner.group_rows(g as usize);
        Ok(vec![a as u32, n as u32])
    }

    /// Row of cell `c` in the grouped table, -1 if no group lists it.
    pub fn cell_row(&self, c: u32) -> i32 {
        self.inner.cell_row(c as usize).map_or(-1, |r| r as i32)
    }

    pub fn has_cell(&self, c: u32) -> bool {
        self.inner.has_cell(c as usize)
    }

    /// The listed cells whose means are not loaded (in order, duplicates kept).
    pub fn missing_cells(&self, cells: &[u32]) -> Vec<u32> {
        cells.iter().copied().filter(|&c| !self.inner.has_cell(c as usize)).collect()
    }

    pub fn loaded_cells(&self) -> u32 {
        self.inner.loaded_cells() as u32
    }

    pub fn loaded_groups(&self) -> u32 {
        self.inner.loaded_groups() as u32
    }

    /// The `beam` best groups for unit query `q`, best first.
    pub fn top_groups(&self, q: &[f32], beam: u32) -> Result<Vec<u32>, JsValue> {
        self.check_q(q)?;
        Ok(self.inner.top_groups(q, beam as usize).into_iter().map(|g| g as u32).collect())
    }

    /// The groups among the `beam` best for `q` that are not fully loaded, best first.
    pub fn missing_groups(&self, q: &[f32], beam: u32) -> Result<Vec<u32>, JsValue> {
        self.check_q(q)?;
        Ok(self.inner.missing_groups(q, beam as usize).into_iter().map(|g| g as u32).collect())
    }

    /// Installs rows `first_row..` (a whole number of rows); returns the cells newly loaded.
    pub fn add_rows(&mut self, first_row: u32, bytes: &[u8]) -> Result<u32, JsValue> {
        Ok(self.inner.add_rows(first_row as usize, bytes).map_err(js_err)? as u32)
    }

    /// Installs the rows of group `g`; returns the cells newly loaded.
    pub fn add_group(&mut self, g: u32, bytes: &[u8]) -> Result<u32, JsValue> {
        Ok(self.inner.add_group(g as usize, bytes).map_err(js_err)? as u32)
    }

    /// Global ids of the `nprobe` best loaded cells among the members of the
    /// best `beam` groups (cosine, descending).
    pub fn route_beam(&self, q: &[f32], beam: u32, nprobe: u32) -> Result<Vec<u32>, JsValue> {
        self.check_q(q)?;
        Ok(self.inner.route_beam(q, beam as usize, nprobe as usize).into_iter().map(|(c, _)| c as u32).collect())
    }

    /// `q·mean[c]` for the listed cells (all loaded), in order.
    pub fn cell_dots_for(&self, q: &[f32], cells: &[u32]) -> Result<Vec<f32>, JsValue> {
        self.check_q(q)?;
        self.inner.cell_dots_for(q, cells).map_err(js_err)
    }

    /// Bytes of decoded means held in wasm memory.
    pub fn resident_bytes(&self) -> f64 {
        self.inner.resident_bytes() as f64
    }
}

impl ColdRouter {
    fn check_q(&self, q: &[f32]) -> Result<(), JsValue> {
        if q.len() != self.inner.dim {
            return Err(js_err("query dim mismatch"));
        }
        Ok(())
    }
}

/// Query-side: routing table plus slab scanning.
#[wasm_bindgen]
pub struct Searcher {
    router: Router,
    tree: Option<CentroidTree>,
}

#[wasm_bindgen]
impl Searcher {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: u32, k1: u32, k2: u32, coarse_means: &[f32], fine_means: &[f32]) -> Result<Searcher, JsValue> {
        let (dim, k1, k2) = (dim as usize, k1 as usize, k2 as usize);
        if coarse_means.len() != k1 * dim || fine_means.len() != k1 * k2 * dim {
            return Err(js_err("centroid buffer size mismatch"));
        }
        Ok(Searcher { router: Router::new(dim, k1, k2, coarse_means, fine_means.to_vec()), tree: None })
    }

    /// A searcher over a compact routing table (`fine.f16` / `fine.i8`, or
    /// `fine.f32` bytes with `"f32"`): the rows are decoded to f32 once, so
    /// routing costs what it does over `fine.f32`, with the decoded means.
    pub fn from_table(dim: u32, k1: u32, k2: u32, coarse_means: &[f32], table: &[u8], elem: &str) -> Result<Searcher, JsValue> {
        let elem = elem_arg(elem)?;
        let (d, k) = (dim as usize, (k1 * k2) as usize);
        if table.len() != k * elem.row_bytes(d) {
            return Err(js_err(format!("{} routing table is {} bytes, expected {k} rows of {} bytes", elem.name(), table.len(), elem.row_bytes(d))));
        }
        if coarse_means.len() != k1 as usize * d {
            return Err(js_err("centroid buffer size mismatch"));
        }
        let fine = decode_rows(table, d, elem).map_err(js_err)?;
        Ok(Searcher { router: Router::new(d, k1 as usize, k2 as usize, coarse_means, fine), tree: None })
    }

    pub fn dim(&self) -> u32 {
        self.router.dim as u32
    }

    /// Global ids of the `nprobe` fine cells to scan for unit query `q`.
    /// `probe_coarse == 0` (or `>= k1`) ranks all fine cells directly (flat routing);
    /// otherwise only the fine cells of the `probe_coarse` best coarse groups are ranked.
    pub fn route(&self, q: &[f32], probe_coarse: u32, nprobe: u32) -> Result<Vec<u32>, JsValue> {
        if q.len() != self.router.dim {
            return Err(js_err("query dim mismatch"));
        }
        Ok(self.router.route(q, probe_coarse as usize, nprobe as usize).into_iter().map(|(c, _)| c as u32).collect())
    }

    /// Flat routing: global ids of the `nprobe` best fine cells over all `k1 * k2`.
    pub fn route_flat(&self, q: &[f32], nprobe: u32) -> Result<Vec<u32>, JsValue> {
        if q.len() != self.router.dim {
            return Err(js_err("query dim mismatch"));
        }
        Ok(self.router.route_flat(q, nprobe as usize).into_iter().map(|(c, _)| c as u32).collect())
    }

    /// Coarse group ids ranked for `q` (for lazy fine-centroid loading).
    pub fn top_coarse(&self, q: &[f32], n: u32) -> Vec<u32> {
        self.router.top_coarse(q, n as usize).into_iter().map(|(g, _)| g as u32).collect()
    }

    /// `q·mean[c]` (unnormalized means) for all `k1 * k2` fine cells, by global
    /// cell id: the `qc` argument of `scan_delta`. Pass the result to
    /// `route_from_dots` to route without computing the dots twice.
    pub fn cell_dots(&self, q: &[f32]) -> Result<Vec<f32>, JsValue> {
        if q.len() != self.router.dim {
            return Err(js_err("query dim mismatch"));
        }
        Ok(self.router.cell_dots(q))
    }

    /// `q·mean[c]` for the listed global cell ids only, in order (the same
    /// values as `cell_dots(q)[c]`): the delta-scan base scores of just the
    /// cells the caller needs.
    pub fn cell_dots_for(&self, q: &[f32], cells: &[u32]) -> Result<Vec<f32>, JsValue> {
        if q.len() != self.router.dim {
            return Err(js_err("query dim mismatch"));
        }
        let k = self.router.cells();
        if let Some(&c) = cells.iter().find(|&&c| c as usize >= k) {
            return Err(js_err(format!("cell id {c} >= {k} cells")));
        }
        Ok(self.router.cell_dots_for(q, cells))
    }

    /// Installs a centroid tree (`build_centroid_tree` over this searcher's
    /// `k1 * k2` fine means) for `route_beam`. Replaces any previous tree.
    pub fn set_tree(&mut self, tree_bytes: &[u8]) -> Result<(), JsValue> {
        self.tree = Some(tree_arg(tree_bytes, self.router.dim, self.router.cells())?);
        Ok(())
    }

    /// Groups of the installed tree (0 = none).
    pub fn tree_groups(&self) -> u32 {
        self.tree.as_ref().map_or(0, |t| t.groups() as u32)
    }

    /// Pruned flat routing: global ids of the `nprobe` best fine cells (by
    /// cosine, descending) among the members of the tree's best `beam` groups.
    /// `beam >= groups` equals `route_flat(q, nprobe)`. Error without `set_tree`.
    pub fn route_beam(&self, q: &[f32], beam: u32, nprobe: u32) -> Result<Vec<u32>, JsValue> {
        let tree = self.tree.as_ref().ok_or_else(|| js_err("route_beam needs a tree (Searcher.set_tree)"))?;
        if q.len() != self.router.dim {
            return Err(js_err("query dim mismatch"));
        }
        let r = &self.router;
        Ok(tree
            .route(q, r.fine_mean_all(), r.fine_inv(), beam as usize, nprobe as usize)
            .into_iter()
            .map(|(c, _)| c as u32)
            .collect())
    }

    /// Fraction of `sample` vectors (`n * dim`) whose exact nearest fine cell
    /// the tree finds with this `beam` (for logging; costs an exhaustive
    /// search per vector). Error without `set_tree`.
    pub fn tree_recall(&self, sample: &[f32], beam: u32) -> Result<f32, JsValue> {
        self.tree_recall_top(sample, beam, 1)
    }

    /// Mean fraction of each sample vector's exact top `m` fine cells that the
    /// tree's top `m` with this `beam` contains (`m = 1`: `tree_recall`).
    pub fn tree_recall_top(&self, sample: &[f32], beam: u32, m: u32) -> Result<f32, JsValue> {
        let tree = self.tree.as_ref().ok_or_else(|| js_err("tree_recall needs a tree (Searcher.set_tree)"))?;
        check_batch(sample.len(), self.router.dim)?;
        let r = &self.router;
        Ok(if m <= 1 {
            tree.recall_check(sample, r.fine_mean_all(), r.fine_inv(), beam as usize)
        } else {
            tree.recall_check_top_m(sample, r.fine_mean_all(), r.fine_inv(), beam as usize, m as usize)
        })
    }

    /// Flat routing from `cell_dots(q)`: same result as `route_flat(q, nprobe)`.
    pub fn route_from_dots(&self, dots: &[f32], nprobe: u32) -> Result<Vec<u32>, JsValue> {
        if dots.len() != self.router.cells() {
            return Err(js_err(format!("dots has {} entries for {} cells", dots.len(), self.router.cells())));
        }
        Ok(self.router.route_flat_from_dots(dots, nprobe as usize).into_iter().map(|(c, _)| c as u32).collect())
    }
}

/// Writes `[scanned, skipped]` into the first two elements of a scan's `stats`
/// out-parameter (fewer if it is shorter; an empty array ignores them).
fn write_stats(stats: &mut [u32], st: ScanStats) {
    for (o, v) in stats.iter_mut().zip([st.scanned, st.skipped]) {
        *o = v as u32;
    }
}

/// Scans one slab for unit query `q`; returns packed top-k hits
/// (`[count u32][id u32, score f32, idx u32]*`, `idx` = entry index in the slab).
/// For a v2 slab, pass the `[0, scan_len)` prefix (the full slab also works);
/// scores are 4-bit scan-tier scores, to be refined with `rerank`. v1 slabs
/// must be passed whole up to their payload and give int8 scores. v3 slabs are
/// an error: scan them with `scan_pq`.
///
/// `tombstones` (`u32` LE ids, sorted unique as `tombstones_merge` writes them;
/// empty = none) are skipped inside the scan, before top-k insertion, so the
/// result holds the best `k` live entries. `stats` is an out-parameter
/// (`Uint32Array`, copied back after the call): `stats[0]` = entries scanned,
/// `stats[1]` = tombstoned entries skipped (only those that scored high enough
/// to enter the top-k when reached). Pass `new Uint32Array(0)` to ignore them.
#[wasm_bindgen]
pub fn scan(slab_scan_bytes: &[u8], q: &[f32], k: u32, tombstones: &[u8], stats: &mut [u32]) -> Result<Vec<u8>, JsValue> {
    let view = SlabView::parse(slab_scan_bytes).map_err(js_err)?;
    if view.is_v3() {
        return Err(js_err("v3 (PQ) slab: scan it with scan_pq(prefix, PqTable, q, k, tombstones, stats), not scan"));
    }
    if view.dim != q.len() {
        return Err(js_err(format!("slab dim {} != query dim {}", view.dim, q.len())));
    }
    let t = tombstones_arg(tombstones)?;
    let mut top = TopK::new(k as usize);
    write_stats(stats, scan_slab(&view, q, &mut top, t.as_ref()));
    Ok(pack_hits(&top.into_sorted()))
}

/// Merges concatenated packed hit buffers, dedupes by id (keeping the best
/// score and that copy's `idx`), keeps the best `k`.
#[wasm_bindgen]
pub fn merge(packed: &[u8], k: u32) -> Vec<u8> {
    pack_hits(&merge_dedup(unpack_hits(packed), k as usize))
}

/// Header summary of a slab (or v2/v3 scan prefix):
/// `[dim, count, cell, byte_len, version, scan_len, rerank_off, rerank_record_len, pq_m]`.
/// `byte_len` is the full object size from the header (equals `slab.length`
/// for a whole slab). For v1, `scan_len` is the payload offset and
/// `rerank_off`/`rerank_record_len` are 0. `pq_m` is 0 except for v3.
#[wasm_bindgen]
pub fn slab_info(slab: &[u8]) -> Result<Vec<u32>, JsValue> {
    let v = SlabView::parse(slab).map_err(js_err)?;
    let h = v.header;
    Ok(vec![
        h.dim as u32,
        h.count as u32,
        h.cell,
        h.total_len() as u32,
        h.version as u32,
        h.scan_len as u32,
        h.rerank_off as u32,
        h.rerank_record_len as u32,
        h.pq_m as u32,
    ])
}

/// Bytes to range-read for a scan, from the first 64 header bytes.
#[wasm_bindgen]
pub fn slab_scan_len(header_bytes: &[u8]) -> Result<u32, JsValue> {
    if header_bytes.len() < SLAB_HEADER {
        return Err(js_err(format!("need {SLAB_HEADER} header bytes, have {}", header_bytes.len())));
    }
    Ok(SlabHeader::parse(&header_bytes[..SLAB_HEADER]).map_err(js_err)?.scan_len as u32)
}

/// The cell mean stored in a slab (or v2 scan prefix).
#[wasm_bindgen]
pub fn slab_centroid(slab_scan_bytes: &[u8]) -> Result<Vec<f32>, JsValue> {
    Ok(SlabView::parse(slab_scan_bytes).map_err(js_err)?.centroid())
}

/// Scores `n` consecutive v2/v3 rerank records (`[scale8 f32][code8 i8 * dim]`,
/// `dim = q.length`) from one slab against `q`: `q·centroid + scale8 * (q·code8)`.
#[wasm_bindgen]
pub fn rerank(q: &[f32], centroid: &[f32], records: &[u8], n: u32) -> Result<Vec<f32>, JsValue> {
    rerank_records(q, centroid, records, n as usize).map_err(js_err)
}

// ---------------------------------------------------------------------------
// Product quantization (v3 scan tier)
// ---------------------------------------------------------------------------

/// `vectors[i] - means[assign[i]]` for each row (`n * dim` floats): turns a
/// sample batch and its cell assignment into PQ training residuals.
#[wasm_bindgen]
pub fn residuals_for(vectors: &[f32], means: &[f32], assign: &[u32], dim: u32) -> Result<Vec<f32>, JsValue> {
    thread_core::pq::residuals(vectors, means, assign, dim as usize).map_err(js_err)
}

/// Initial PQ codebook (serialized, "VIBQ"): per subspace, the subvectors of
/// 256 random rows of `residuals` (`n * dim`; pass ~2-4k rows). Requires
/// `dim % m == 0`.
#[wasm_bindgen]
pub fn pq_init_codebook(residuals: &[f32], dim: u32, m: u32, seed: u32) -> Result<Vec<u8>, JsValue> {
    Ok(Codebook::init_from_samples(residuals, dim as usize, m as usize, seed as u64).map_err(js_err)?.to_bytes())
}

/// `[dim, m, sub]` of a serialized codebook.
#[wasm_bindgen]
pub fn pq_codebook_info(codebook_bytes: &[u8]) -> Result<Vec<u32>, JsValue> {
    let cb = Codebook::from_bytes(codebook_bytes).map_err(js_err)?;
    Ok(vec![cb.dim as u32, cb.m as u32, cb.sub as u32])
}

/// PQ training map job (one Lloyd iteration): assigns each residual's
/// subvectors to the nearest sub-centroids of the current codebook and
/// accumulates sums and counts. `partial()` is ~1.8 MB at dim 1536, m 192.
#[wasm_bindgen]
pub struct PqTrainJob {
    codebook: Codebook,
    accum: PqAccum,
}

#[wasm_bindgen]
impl PqTrainJob {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: u32, m: u32, codebook_bytes: &[u8]) -> Result<PqTrainJob, JsValue> {
        let codebook = Codebook::from_bytes(codebook_bytes).map_err(js_err)?;
        if codebook.dim != dim as usize || codebook.m != m as usize {
            return Err(js_err(format!("codebook is {}x{}, job is {dim}x{m}", codebook.dim, codebook.m)));
        }
        let accum = PqAccum::new(dim as usize, m as usize).map_err(js_err)?;
        Ok(PqTrainJob { codebook, accum })
    }

    /// Adds `n * dim` residuals (see `residuals_for`). Returns `n`.
    pub fn add_residuals(&mut self, residuals: &[f32]) -> Result<u32, JsValue> {
        let n = check_batch(residuals.len(), self.accum.dim)?;
        for r in residuals.chunks_exact(self.accum.dim) {
            self.accum.add(r, &self.codebook);
        }
        Ok(n as u32)
    }

    /// Residuals added so far.
    pub fn rows(&self) -> f64 {
        self.accum.rows as f64
    }

    /// Mean squared quantization error of the added residuals under the
    /// job's (input) codebook.
    pub fn mean_sq_err(&self) -> f64 {
        self.accum.mean_sq_err()
    }

    /// Serialized partial accumulator ("VIBT"), for `pq_reduce` / `PqReducer`.
    pub fn partial(&self) -> Vec<u8> {
        self.accum.to_bytes()
    }
}

/// Streaming reduce of PQ training partials: `add_partial` each partial
/// (so only one is held at a time), then `finish(prev_codebook)` returns the
/// next codebook. Empty sub-centroids keep their previous value.
#[wasm_bindgen]
pub struct PqReducer {
    accum: PqAccum,
}

#[wasm_bindgen]
impl PqReducer {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: u32, m: u32) -> Result<PqReducer, JsValue> {
        Ok(PqReducer { accum: PqAccum::new(dim as usize, m as usize).map_err(js_err)? })
    }

    pub fn add_partial(&mut self, partial: &[u8]) -> Result<(), JsValue> {
        let a = PqAccum::from_bytes(partial).map_err(js_err)?;
        self.accum.merge(&a).map_err(js_err)
    }

    /// Residuals over all added partials.
    pub fn rows(&self) -> f64 {
        self.accum.rows as f64
    }

    /// Mean squared quantization error over all added partials (under the
    /// codebook the map jobs used): a convergence signal per iteration.
    pub fn mean_sq_err(&self) -> f64 {
        self.accum.mean_sq_err()
    }

    /// Sub-centroids (out of `m * 256`) that received no residual.
    pub fn empty_centroids(&self) -> u32 {
        self.accum.counts.iter().filter(|&&c| c == 0).count() as u32
    }

    /// Next serialized codebook.
    pub fn finish(&self, prev_codebook: &[u8]) -> Result<Vec<u8>, JsValue> {
        let prev = Codebook::from_bytes(prev_codebook).map_err(js_err)?;
        Ok(self.accum.means(&prev).map_err(js_err)?.to_bytes())
    }
}

/// One-shot reduce: `partials_concat` is `n_partials` `PqTrainJob.partial()`
/// buffers back to back (all the same size). Returns the next codebook.
/// Prefer `PqReducer` when the partials do not fit in memory together.
#[wasm_bindgen]
pub fn pq_reduce(partials_concat: &[u8], n_partials: u32, prev_codebook: &[u8]) -> Result<Vec<u8>, JsValue> {
    let prev = Codebook::from_bytes(prev_codebook).map_err(js_err)?;
    let size = PqAccum::serialized_len(prev.dim, prev.m).map_err(js_err)?;
    let n = n_partials as usize;
    if n == 0 || partials_concat.len() != n * size {
        return Err(js_err(format!("expected {n} partials of {size} bytes, got {} bytes", partials_concat.len())));
    }
    let mut acc = PqAccum::new(prev.dim, prev.m).map_err(js_err)?;
    for p in partials_concat.chunks_exact(size) {
        acc.merge(&PqAccum::from_bytes(p).map_err(js_err)?).map_err(js_err)?;
    }
    Ok(acc.means(&prev).map_err(js_err)?.to_bytes())
}

/// A parsed codebook to keep across queries (parsing ~1.5 MB per query is
/// avoidable): build each query's table with `PqTable.from_codebook`.
#[wasm_bindgen]
pub struct PqCodebook {
    cb: Codebook,
}

#[wasm_bindgen]
impl PqCodebook {
    #[wasm_bindgen(constructor)]
    pub fn new(codebook_bytes: &[u8]) -> Result<PqCodebook, JsValue> {
        Ok(PqCodebook { cb: Codebook::from_bytes(codebook_bytes).map_err(js_err)? })
    }

    pub fn dim(&self) -> u32 {
        self.cb.dim as u32
    }

    pub fn m(&self) -> u32 {
        self.cb.m as u32
    }
}

/// Per-query asymmetric-distance table (`m x 256` f32: `q_sub · sub-centroid`).
/// Build once per query, then `scan_pq` every probed v3 slab with it.
#[wasm_bindgen]
pub struct PqTable {
    t: QueryTable,
}

#[wasm_bindgen]
impl PqTable {
    /// `q`: the unit query; `codebook_bytes`: the index's serialized codebook.
    #[wasm_bindgen(constructor)]
    pub fn new(q: &[f32], codebook_bytes: &[u8]) -> Result<PqTable, JsValue> {
        let cb = Codebook::from_bytes(codebook_bytes).map_err(js_err)?;
        Ok(PqTable { t: QueryTable::new(q, &cb).map_err(js_err)? })
    }

    /// Same as the constructor, from a cached `PqCodebook`.
    pub fn from_codebook(q: &[f32], codebook: &PqCodebook) -> Result<PqTable, JsValue> {
        Ok(PqTable { t: QueryTable::new(q, &codebook.cb).map_err(js_err)? })
    }

    pub fn m(&self) -> u32 {
        self.t.m as u32
    }
}

/// Scans one v3 slab (its `[0, scan_len)` prefix, or the whole slab) for unit
/// query `q` with PQ scores `q·centroid + sum_j table[j][code_j]`; returns
/// packed top-k hits in the `scan` wire format (`idx` = entry index, locating
/// the int8 rerank record for `rerank`). `table` must be built from the same
/// `q`. Errors on v1/v2 slabs and on a table whose `m` differs from the slab's.
/// `tombstones` and `stats` as in `scan`: tombstoned ids are skipped before
/// top-k insertion; `stats` receives `[scanned, skipped]`.
#[wasm_bindgen]
pub fn scan_pq(slab_scan_bytes: &[u8], table: &PqTable, q: &[f32], k: u32, tombstones: &[u8], stats: &mut [u32]) -> Result<Vec<u8>, JsValue> {
    let view = SlabView::parse(slab_scan_bytes).map_err(js_err)?;
    let t = tombstones_arg(tombstones)?;
    let mut top = TopK::new(k as usize);
    write_stats(stats, scan_slab_pq(&view, &table.t, q, &mut top, t.as_ref()).map_err(js_err)?);
    Ok(pack_hits(&top.into_sorted()))
}

// ---------------------------------------------------------------------------
// Incremental updates: delta segments, tombstones, compaction
// ---------------------------------------------------------------------------

/// Parses a tombstone buffer (`u32` LE ids); empty means none.
fn tombstones_arg(bytes: &[u8]) -> Result<Option<Tombstones>, JsValue> {
    if bytes.is_empty() {
        return Ok(None);
    }
    Ok(Some(Tombstones::from_bytes(bytes).map_err(js_err)?))
}

/// Update job: encodes new vectors into one delta segment ("VIBD"). Holds the
/// `k * dim` cell means (+ inverse norms, + the PQ codebook after
/// `set_codebook`). Each vector goes to its primary cell (highest cosine over
/// all `k` cells) with residual codes against that cell's mean, as an
/// `AssignJob` part record (4-bit + int8, + PQ codes with a codebook), so
/// compaction moves it into the cell's slab without re-encoding. Vectors
/// must be unit-normalized, as for `AssignJob`.
#[wasm_bindgen]
pub struct DeltaJob {
    enc: DeltaEncoder,
}

#[wasm_bindgen]
impl DeltaJob {
    /// `means`: the generation's `k * dim` fine means (`fine.f32`); `seq`: the
    /// segment's sequence number (stored in its header).
    #[wasm_bindgen(constructor)]
    pub fn new(dim: u32, k: u32, means: &[f32], seq: u32) -> Result<DeltaJob, JsValue> {
        Ok(DeltaJob { enc: DeltaEncoder::new(dim as usize, k as usize, means.to_vec(), seq).map_err(js_err)? })
    }

    /// Entries also carry PQ codes under this codebook (required to compact
    /// into format 3 slabs). Call before adding vectors.
    pub fn set_codebook(&mut self, codebook_bytes: &[u8]) -> Result<(), JsValue> {
        let cb = Codebook::from_bytes(codebook_bytes).map_err(js_err)?;
        self.enc.set_codebook(cb).map_err(js_err)
    }

    /// Picks each added vector's primary cell with a pruned search: the tree
    /// (`build_centroid_tree` over these `k` means) scores its group means and
    /// only the cells of the best `beam` groups (`beam >= groups`: identical to
    /// the exhaustive search). Applies to vectors added afterwards.
    pub fn set_tree(&mut self, tree_bytes: &[u8], beam: u32) -> Result<(), JsValue> {
        let t = tree_arg(tree_bytes, self.enc.dim(), self.enc.k())?;
        self.enc.set_tree(t, beam as usize).map_err(js_err)
    }

    /// Same as `set_codebook`, from a parsed `PqCodebook` (copied).
    pub fn set_pq_codebook(&mut self, codebook: &PqCodebook) -> Result<(), JsValue> {
        self.enc.set_codebook(codebook.cb.clone()).map_err(js_err)
    }

    /// Adds `n` vectors (`n * dim`) with ids `ids` (`n`) and attrs (`n`, or
    /// empty for all 0), no payloads. Returns each vector's primary cell.
    /// Nothing is added if the arguments are inconsistent.
    pub fn add(&mut self, vectors: &[f32], ids: &[u32], attrs: &[u32]) -> Result<Vec<u32>, JsValue> {
        let n = self.check_add(vectors, ids, attrs)?;
        let dim = self.enc.dim();
        Ok((0..n)
            .map(|i| {
                let a = attrs.get(i).copied().unwrap_or(0);
                self.enc.add(&vectors[i * dim..(i + 1) * dim], ids[i], a, &[]) as u32
            })
            .collect())
    }

    /// Like `add`, with payloads: `payloads` is the concatenation of every
    /// vector's payload, `payload_lens[i]` the length of vector `i`'s.
    pub fn add_with_payloads(
        &mut self,
        vectors: &[f32],
        ids: &[u32],
        attrs: &[u32],
        payloads: &[u8],
        payload_lens: &[u32],
    ) -> Result<Vec<u32>, JsValue> {
        let n = self.check_add(vectors, ids, attrs)?;
        if payload_lens.len() != n {
            return Err(js_err(format!("payload_lens has {} entries for {n} vectors", payload_lens.len())));
        }
        let total: u64 = payload_lens.iter().map(|&l| l as u64).sum();
        if total != payloads.len() as u64 {
            return Err(js_err(format!("payload_lens sum to {total}, payloads has {} bytes", payloads.len())));
        }
        let dim = self.enc.dim();
        let mut off = 0usize;
        Ok((0..n)
            .map(|i| {
                let a = attrs.get(i).copied().unwrap_or(0);
                let p = &payloads[off..off + payload_lens[i] as usize];
                off += p.len();
                self.enc.add(&vectors[i * dim..(i + 1) * dim], ids[i], a, p) as u32
            })
            .collect())
    }

    /// Entries added so far.
    pub fn len(&self) -> u32 {
        self.enc.len() as u32
    }

    /// Serialized size of the segment so far.
    pub fn byte_len(&self) -> u32 {
        self.enc.byte_len() as u32
    }

    /// PQ codes per entry (0 without a codebook).
    pub fn pq_m(&self) -> u32 {
        self.enc.pq_m() as u32
    }

    /// The serialized delta segment.
    pub fn finish(self) -> Vec<u8> {
        self.enc.finish()
    }
}

impl DeltaJob {
    fn check_add(&self, vectors: &[f32], ids: &[u32], attrs: &[u32]) -> Result<usize, JsValue> {
        let n = check_batch(vectors.len(), self.enc.dim())?;
        if ids.len() != n {
            return Err(js_err(format!("ids has {} entries for {n} vectors", ids.len())));
        }
        if !attrs.is_empty() && attrs.len() != n {
            return Err(js_err(format!("attrs has {} entries for {n} vectors (pass n or none)", attrs.len())));
        }
        Ok(n)
    }
}

/// Header summary of a delta segment: `[dim, count, pq_m, seq, byte_len]`.
#[wasm_bindgen]
pub fn delta_info(delta_bytes: &[u8]) -> Result<Vec<u32>, JsValue> {
    let v = DeltaView::parse(delta_bytes).map_err(js_err)?;
    Ok(vec![v.dim as u32, v.count as u32, v.pq_m as u32, v.seq, v.byte_len() as u32])
}

/// Scans a whole delta segment for unit query `q` with its int8 records:
/// score = `qc[cell] + scale8 * (q·code8)`, where `qc = Searcher.cell_dots(q)`
/// (one dot per fine cell, unnormalized means). Returns the top `k` as packed
/// hits in the `scan` wire format, `idx` = entry index in the segment. Scores
/// are on the reranked (int8) scale, so they merge directly with `rerank`
/// results; no second phase is needed. `tombstones` and `stats` as in `scan`:
/// pass the ids to hide in this segment (for a segment added at rev `a`, the
/// union of the sets with rev > `a`); `stats` receives `[scanned, skipped]`.
#[wasm_bindgen]
pub fn scan_delta(delta_bytes: &[u8], q: &[f32], qc: &[f32], k: u32, tombstones: &[u8], stats: &mut [u32]) -> Result<Vec<u8>, JsValue> {
    let v = DeltaView::parse(delta_bytes).map_err(js_err)?;
    let t = tombstones_arg(tombstones)?;
    let mut top = TopK::new(k as usize);
    write_stats(stats, scan_delta_core(&v, q, qc, &mut top, t.as_ref()).map_err(js_err)?);
    Ok(pack_hits(&top.into_sorted()))
}

/// Payload of delta entry `idx` (a `scan_delta` hit's `idx`).
#[wasm_bindgen]
pub fn delta_payload(delta_bytes: &[u8], idx: u32) -> Result<Vec<u8>, JsValue> {
    let v = DeltaView::parse(delta_bytes).map_err(js_err)?;
    if idx as usize >= v.count {
        return Err(js_err(format!("entry {idx} out of range (count {})", v.count)));
    }
    Ok(v.payload(idx as usize).to_vec())
}

/// Distinct cells of a delta segment with their entry counts, ascending:
/// `[cell0, count0, cell1, count1, ...]`.
#[wasm_bindgen]
pub fn delta_cells(delta_bytes: &[u8]) -> Result<Vec<u32>, JsValue> {
    let v = DeltaView::parse(delta_bytes).map_err(js_err)?;
    Ok(v.cell_counts().into_iter().flat_map(|(c, n)| [c, n]).collect())
}

/// Part records (for `SlabJob.add_records`) of every entry of `cell` in a delta
/// segment whose id is not in `tombstones` (`u32` LE ids, may be empty).
#[wasm_bindgen]
pub fn delta_records_for_cell(delta_bytes: &[u8], cell: u32, tombstones: &[u8]) -> Result<Vec<u8>, JsValue> {
    let v = DeltaView::parse(delta_bytes).map_err(js_err)?;
    let t = tombstones_arg(tombstones)?;
    let mut out = Vec::new();
    v.records_for_cell(cell, t.as_ref(), &mut out);
    Ok(out)
}

/// Part records (for `SlabJob.add_records`) of every entry of a whole slab
/// (not a scan prefix) whose id is not in `tombstones` (may be empty). A slab
/// rebuilt from them with `SlabJob(dim, cell, mean, sameFormat)` is byte-identical.
/// v3 and v1 slabs have no 4-bit codes: their records are flagged and a
/// format 2 `SlabJob` rejects them (`add_records` errors); v2 records carry no
/// PQ codes, so a format 3 `SlabJob` rejects them.
#[wasm_bindgen]
pub fn slab_to_records(slab_bytes: &[u8], tombstones: &[u8]) -> Result<Vec<u8>, JsValue> {
    let v = SlabView::parse(slab_bytes).map_err(js_err)?;
    let t = tombstones_arg(tombstones)?;
    let mut out = Vec::new();
    v.to_part_records(&mut out, t.as_ref()).map_err(js_err)?;
    Ok(out)
}

/// The union of a tombstone buffer (`u32` LE ids, sorted unique; may be
/// empty) and `add`, serialized the same way.
#[wasm_bindgen]
pub fn tombstones_merge(existing: &[u8], add: &[u32]) -> Result<Vec<u8>, JsValue> {
    Ok(Tombstones::from_bytes(existing).map_err(js_err)?.merge(add).to_bytes())
}

/// Drops hits whose id is tombstoned from packed hits (one or more
/// concatenated `scan`/`merge` buffers); returns one packed buffer, order kept.
/// Applied to a finished top-k it can leave fewer than k hits: queries pass
/// the tombstones to `scan`/`scan_pq`/`scan_delta` instead.
#[wasm_bindgen]
pub fn filter_hits(packed: &[u8], tombstones: &[u8]) -> Result<Vec<u8>, JsValue> {
    let t = Tombstones::from_bytes(tombstones).map_err(js_err)?;
    Ok(pack_hits(&t.filter_hits(unpack_hits(packed))))
}

/// Proposes a split of an overfull cell: decodes every entry of a whole slab
/// (any format; v2/v3 need the rerank section) as `mean + scale8 * code8`,
/// runs spherical 2-means (`kmeans::split_cell`, seeded with `seed`) and
/// returns the two new means (`2 * dim` floats). `mean` must be the slab's
/// cell mean; pass an empty array to use the mean stored in the slab.
/// Memory: `count * dim * 4` bytes of decoded vectors.
#[wasm_bindgen]
pub fn split_slab(slab_bytes: &[u8], mean: &[f32], seed: u32) -> Result<Vec<f32>, JsValue> {
    let v = SlabView::parse(slab_bytes).map_err(js_err)?;
    if v.has_rerank_tier() && !v.has_rerank() {
        return Err(js_err("split_slab needs the slab's rerank section, not a scan prefix"));
    }
    if v.count < 2 {
        return Err(js_err(format!("cannot split a slab with {} entries", v.count)));
    }
    let stored;
    let mean = if mean.is_empty() {
        stored = v.centroid();
        &stored[..]
    } else if mean.len() == v.dim {
        mean
    } else {
        return Err(js_err(format!("mean has {} floats, slab dim is {}", mean.len(), v.dim)));
    };
    let mut xs = vec![0f32; v.count * v.dim];
    for (i, x) in xs.chunks_exact_mut(v.dim).enumerate() {
        decode_residual_sq8(v.code8(i), v.scale8(i), mean, x);
    }
    Ok(thread_core::kmeans::split_cell(&xs, v.dim, seed as u64).0)
}

/// Optional PQ codebook argument: empty bytes = none.
fn codebook_arg(bytes: &[u8], dim: usize) -> Result<Option<Codebook>, JsValue> {
    if bytes.is_empty() {
        return Ok(None);
    }
    let cb = Codebook::from_bytes(bytes).map_err(js_err)?;
    if cb.dim != dim {
        return Err(js_err(format!("codebook dim {} != dim {dim}", cb.dim)));
    }
    Ok(Some(cb))
}

/// `[len_a u32 LE][a][b]`: the output of the split functions below.
fn pack_split(a: Vec<u8>, b: Vec<u8>) -> Result<Vec<u8>, JsValue> {
    let la = u32::try_from(a.len()).map_err(|_| js_err("split half exceeds 4 GB"))?;
    let mut out = Vec::with_capacity(4 + a.len() + b.len());
    out.extend_from_slice(&la.to_le_bytes());
    out.extend_from_slice(&a);
    out.extend_from_slice(&b);
    Ok(out)
}

/// Executes a cell split: every entry of a whole slab (`slab_bytes`, not a scan
/// prefix) whose id is not in `tombstones` (`u32` LE, may be empty) is decoded
/// from its int8 tier against the slab's mean, assigned to the nearer (cosine)
/// of the two new means `means2` (`2 * dim` floats, e.g. from `split_slab`) and
/// re-encoded against it as a part record (4-bit, int8, and PQ codes when
/// `codebook_bytes` is a codebook; pass empty bytes for none, which only
/// format 1/2 slabs accept). Returns ONE buffer: a 4-byte `u32` LE prefix
/// holding the byte length `la` of records_a, then records_a (`[4, 4 + la)`),
/// then records_b (`[4 + la, end)`). Feed them to `SlabJob(dim, cell, meanA,
/// format)` and `SlabJob(dim, newCell, meanB, format)`.
#[wasm_bindgen]
pub fn split_slab_records(slab_bytes: &[u8], tombstones: &[u8], means2: &[f32], codebook_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let dim = SlabView::parse(slab_bytes).map_err(js_err)?.dim;
    let t = tombstones_arg(tombstones)?;
    let cb = codebook_arg(codebook_bytes, dim)?;
    let (a, b) = thread_core::slab::split_records(slab_bytes, t.as_ref(), means2, cb.as_ref()).map_err(js_err)?;
    pack_split(a, b)
}

/// The same split for part records (e.g. `delta_records_for_cell` of the split
/// cell, or `slab_to_records`): decoded against `old_mean` (the cell's mean the
/// records were encoded with; its length is the dim). Same output layout as
/// `split_slab_records`.
#[wasm_bindgen]
pub fn split_part_records(records: &[u8], old_mean: &[f32], means2: &[f32], codebook_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let cb = codebook_arg(codebook_bytes, old_mean.len())?;
    let (a, b) = thread_core::part::split_part_records(records, old_mean, means2, cb.as_ref()).map_err(js_err)?;
    pack_split(a, b)
}

/// Re-encodes part records (decoded from their int8 tier against `old_mean`)
/// against `new_mean`: 4-bit, int8 and, with a codebook (empty bytes = none),
/// PQ codes. Ids, attrs and payloads are kept.
#[wasm_bindgen]
pub fn reencode_records(records: &[u8], old_mean: &[f32], new_mean: &[f32], codebook_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let cb = codebook_arg(codebook_bytes, old_mean.len())?;
    thread_core::part::reencode_records(records, old_mean, new_mean, cb.as_ref()).map_err(js_err)
}

/// Native tests of the job wrappers (success paths only: building a `JsValue`
/// error panics off wasm).
#[cfg(test)]
mod tests {
    use super::*;
    use thread_core::math::{normalized_rows, Rng};

    fn rows(rng: &mut Rng, n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|_| rng.below(20001) as f32 / 10000.0 - 1.0).collect()
    }

    fn setup(k: usize, dim: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut rng = Rng::new(31);
        (rows(&mut rng, k, dim), normalized_rows(&rows(&mut rng, n, dim), dim))
    }

    #[test]
    fn inc_kmeans_with_full_table_equals_flat_tree_job() {
        let (k, dim, n) = (48, 32, 400);
        let (means, xs) = setup(k, dim, n);
        let ones = vec![1.0f32; dim];
        let mut flat = TreeKmeansJob::new(dim as u32, 1, k as u32, &ones, &means, 1).unwrap();
        let assign = flat.add_batch_assign(&xs).unwrap();
        // plain add_batch accumulates the same
        let mut flat2 = TreeKmeansJob::new(dim as u32, 1, k as u32, &ones, &means, 1).unwrap();
        assert_eq!(flat2.add_batch(&xs).unwrap(), n as u32);
        assert_eq!(flat2.partial(), flat.partial());

        let nb = cell_neighbors(&means, dim as u32, 0, k as u32, k as u32).unwrap();
        let mut inc = IncKmeansJob::new(dim as u32, k as u32, &means, &nb, k as u32).unwrap();
        let prev: Vec<u32> = (0..n as u32).map(|i| (i * 7) % k as u32).collect();
        let got = inc.add_batch(&xs, &prev).unwrap();
        assert_eq!(got, assign);
        assert_eq!(inc.partial(), flat.partial());

        // with a small table and prev = the true cell, the true cell is always a
        // candidate (rows start with the cell itself), so assignments can only improve
        let nb8 = cell_neighbors(&means, dim as u32, 0, k as u32, 8).unwrap();
        let mut inc8 = IncKmeansJob::new(dim as u32, k as u32, &means, &nb8, 8).unwrap();
        assert_eq!(inc8.add_batch(&xs, &assign).unwrap(), assign);
    }

    #[test]
    fn two_level_jobs_probe_coarse_zero_means_all_groups() {
        let (k1, k2, dim, n, m) = (4usize, 6usize, 16usize, 80usize, 3u32);
        let mut rng = Rng::new(9);
        let coarse = rows(&mut rng, k1, dim);
        let fine = rows(&mut rng, k1 * k2, dim);
        let xs = normalized_rows(&rows(&mut rng, n, dim), dim);
        let (d, a, b) = (dim as u32, k1 as u32, k2 as u32);
        // probe_coarse 0 used to panic in top_n_dot_scaled (n = 0); now it means all k1 groups
        let mut all = TreeKmeansJob::new(d, a, b, &coarse, &fine, a).unwrap();
        let mut zero = TreeKmeansJob::new(d, a, b, &coarse, &fine, 0).unwrap();
        assert_eq!(zero.add_batch_assign(&xs).unwrap(), all.add_batch_assign(&xs).unwrap());
        assert_eq!(zero.partial(), all.partial());
        let mut aa = AssignJob::new(d, a, b, &coarse, &fine, a, m).unwrap();
        let mut az = AssignJob::new(d, a, b, &coarse, &fine, 0, m).unwrap();
        aa.add_batch(&xs, 0).unwrap();
        az.add_batch(&xs, 0).unwrap();
        assert_eq!(az.assigned_records(), (n * m as usize) as f64);
        for g in 0..a {
            assert_eq!(az.take_part(g), aa.take_part(g));
        }
        // and the Searcher's routing with nprobe 0 is empty, not a panic
        let s = Searcher::new(d, a, b, &coarse, &fine).unwrap();
        assert!(s.route(&xs[..dim], 0, 0).unwrap().is_empty());
        assert!(s.route(&xs[..dim], 2, 0).unwrap().is_empty());
    }

    #[test]
    fn assign_with_candidates_matches_exhaustive_parts() {
        let (k, dim, n, m) = (40, 16, 120, 4);
        let (means, xs) = setup(k, dim, n);
        let ones = vec![1.0f32; dim];
        let mut full = AssignJob::new(dim as u32, 1, k as u32, &ones, &means, 1, m).unwrap();
        full.add_batch(&xs, 1000).unwrap();
        let want = full.take_part(0);

        let nb = cell_neighbors(&means, dim as u32, 0, k as u32, k as u32).unwrap();
        let mut cand = AssignJob::with_candidates(dim as u32, k as u32, &means, &nb, k as u32, m).unwrap();
        let prev: Vec<u32> = (0..n as u32).map(|i| (i * 5) % k as u32).collect();
        assert_eq!(cand.add_batch_prev(&xs, 1000, &prev).unwrap(), n as u32);
        assert_eq!(cand.assigned_records(), (n * m as usize) as f64);
        assert_eq!(cand.take_part(0), want);
        // the exhaustive path of a with_candidates job writes the same part
        cand.add_batch(&xs, 1000).unwrap();
        assert_eq!(cand.take_part(0), want);
    }

    #[test]
    fn slab_job_formats_and_span_ranges() {
        let (k, dim, n) = (6, 16, 60);
        let (means, xs) = setup(k, dim, n);
        let ones = vec![1.0f32; dim];
        let mut a = AssignJob::new(dim as u32, 1, k as u32, &ones, &means, 1, 2).unwrap();
        a.add_batch(&xs, 0).unwrap();
        let part = a.take_part(0);
        let dir = part_directory(&part).unwrap();
        let (first, count) = (1u32, 4u32);
        let sp = part_span(&dir, first, count).unwrap();
        let span = &part[sp[0] as usize..(sp[0] + sp[1]) as usize];
        for cell in first..first + count {
            let (off, len) = (dir[2 + cell as usize * 3 + 1], dir[2 + cell as usize * 3 + 2]);
            let c = &means[cell as usize * dim..(cell as usize + 1) * dim];
            for format in [None, Some(2u8), Some(1)] {
                let mut by_slice = SlabJob::new(dim as u32, cell, c, format).unwrap();
                let mut by_range = SlabJob::new(dim as u32, cell, c, format).unwrap();
                let cnt = by_slice.add_records(&part[off as usize..(off + len) as usize]).unwrap();
                assert_eq!(cnt, dir[2 + cell as usize * 3]);
                assert_eq!(by_range.add_records_range(span, sp[0], off, len).unwrap(), cnt);
                assert_eq!(by_slice.format(), format.unwrap_or(2) as u32);
                let (s1, s2) = (by_slice.finish(), by_range.finish());
                assert_eq!(s1, s2);
                let info = slab_info(&s1).unwrap();
                assert_eq!((info[1], info[2], info[4]), (cnt, cell, format.unwrap_or(2) as u32));
                assert_eq!(info[8], 0);
            }
        }
    }

    #[test]
    fn pq_pipeline_train_assign_finalize_scan() {
        let (k, dim, n, m, pq_m) = (4, 32, 400, 2, 8);
        let (means, xs) = setup(k, dim, n);
        let ones = vec![1.0f32; dim];
        // cell assignment of the sample -> residuals -> init + 3 map-reduce iterations
        let mut tj = TreeKmeansJob::new(dim as u32, 1, k as u32, &ones, &means, 1).unwrap();
        let assign = tj.add_batch_assign(&xs).unwrap();
        let res = residuals_for(&xs, &means, &assign, dim as u32).unwrap();
        let mut cb = pq_init_codebook(&res, dim as u32, pq_m, 5).unwrap();
        assert_eq!(pq_codebook_info(&cb).unwrap(), vec![dim as u32, pq_m, 4]);
        let mut errs = Vec::new();
        for _ in 0..3 {
            let half = (n / 2) * dim;
            let mut a = PqTrainJob::new(dim as u32, pq_m, &cb).unwrap();
            let mut b = PqTrainJob::new(dim as u32, pq_m, &cb).unwrap();
            assert_eq!(a.add_residuals(&res[..half]).unwrap() + b.add_residuals(&res[half..]).unwrap(), n as u32);
            let mut red = PqReducer::new(dim as u32, pq_m).unwrap();
            red.add_partial(&a.partial()).unwrap();
            red.add_partial(&b.partial()).unwrap();
            assert_eq!(red.rows(), n as f64);
            errs.push(red.mean_sq_err());
            let next = red.finish(&cb).unwrap();
            let cat = [a.partial(), b.partial()].concat();
            assert_eq!(pq_reduce(&cat, 2, &cb).unwrap(), next);
            cb = next;
        }
        assert!(errs[2] <= errs[0], "{errs:?}");

        // assign with pq codes (both constructors), finalize v3, scan_pq, rerank
        let mut aj = AssignJob::new(dim as u32, 1, k as u32, &ones, &means, 1, m).unwrap();
        assert_eq!(aj.pq_m(), 0);
        aj.set_codebook(&cb).unwrap();
        assert_eq!(aj.pq_m(), pq_m);
        aj.add_batch(&xs, 0).unwrap();
        let part = aj.take_part(0);
        let nb = cell_neighbors(&means, dim as u32, 0, k as u32, k as u32).unwrap();
        let mut cj = AssignJob::with_candidates(dim as u32, k as u32, &means, &nb, k as u32, m).unwrap();
        cj.set_codebook(&cb).unwrap();
        cj.add_batch_prev(&xs, 0, &assign).unwrap();
        assert_eq!(cj.take_part(0), part);

        let dir = part_directory(&part).unwrap();
        let pcb = PqCodebook::new(&cb).unwrap();
        let q = xs[7 * dim..8 * dim].to_vec();
        let table = PqTable::from_codebook(&q, &pcb).unwrap();
        assert_eq!(PqTable::new(&q, &cb).unwrap().t.table, table.t.table);
        let mut found = false;
        for cell in 0..k {
            let (off, len) = (dir[2 + cell * 3 + 1], dir[2 + cell * 3 + 2]);
            let c = &means[cell * dim..(cell + 1) * dim];
            let mut sj = SlabJob::new(dim as u32, cell as u32, c, Some(3)).unwrap();
            assert_eq!(sj.format(), 3);
            let cnt = sj.add_records(&part[off as usize..(off + len) as usize]).unwrap();
            let slab = sj.finish();
            let info = slab_info(&slab).unwrap();
            assert_eq!((info[1], info[4], info[8]), (cnt, 3, if cnt > 0 { pq_m } else { 0 }));
            let prefix = &slab[..info[5] as usize];
            let mut st = [9u32; 2];
            let hits = unpack_hits(&scan_pq(prefix, &table, &q, 5, &[], &mut st).unwrap());
            assert_eq!(hits.len(), (cnt as usize).min(5));
            assert_eq!(st, [cnt, 0]);
            // a tombstoned best hit does not take a slot: the next live entries fill the top 5
            if let Some(best) = hits.first() {
                let tomb = tombstones_merge(&[], &[best.id]).unwrap();
                let live = unpack_hits(&scan_pq(prefix, &table, &q, 5, &tomb, &mut st).unwrap());
                assert_eq!(live.len(), (cnt as usize - 1).min(5));
                assert_eq!(&live[..hits.len() - 1], &hits[1..]);
                assert_eq!(st[1], 1);
            }
            if let Some(h) = hits.iter().find(|h| h.id == 7) {
                // the query's own vector scores near 1 and its rerank score is ~exact
                let rl = info[7] as usize;
                let o = info[6] as usize + h.idx as usize * rl;
                let s = rerank(&q, c, &slab[o..o + rl], 1).unwrap()[0];
                assert!((s - 1.0).abs() < 1e-2, "rerank {s}");
                found = true;
            }
        }
        assert!(found);
    }

    #[test]
    fn centroid_tree_wrappers_match_exhaustive_at_full_beam() {
        let (k1, k2, dim, n, m) = (2usize, 30usize, 16usize, 150usize, 3u32);
        let k = k1 * k2;
        let (fine, xs) = setup(k, dim, n);
        let mut rng = Rng::new(77);
        let coarse = rows(&mut rng, k1, dim);
        let groups = 7u32;
        let tree = build_centroid_tree(&fine, dim as u32, groups, 5, 9).unwrap();
        assert_eq!(centroid_tree_info(&tree).unwrap(), vec![dim as u32, k as u32, groups, k as u32, tree.len() as u32]);

        // Searcher
        let mut s = Searcher::new(dim as u32, k1 as u32, k2 as u32, &coarse, &fine).unwrap();
        assert_eq!(s.tree_groups(), 0);
        s.set_tree(&tree).unwrap();
        assert_eq!(s.tree_groups(), groups);
        for x in xs.chunks_exact(dim) {
            assert_eq!(s.route_beam(x, groups, 5).unwrap(), s.route_flat(x, 5).unwrap());
            assert_eq!(s.route_beam(x, 100, 1).unwrap(), s.route(x, 0, 1).unwrap());
            assert!(s.route_beam(x, 1, 4).unwrap().len() <= 4);
            let dots = s.cell_dots(x).unwrap();
            let cells = [3u32, 59, 0, 3];
            let want: Vec<f32> = cells.iter().map(|&c| dots[c as usize]).collect();
            assert_eq!(s.cell_dots_for(x, &cells).unwrap(), want);
        }
        assert_eq!(s.tree_recall(&xs, groups).unwrap(), 1.0);
        assert_eq!(s.tree_recall_top(&xs, groups, 3).unwrap(), 1.0);
        let r1 = s.tree_recall(&xs, 1).unwrap();
        assert!((0.0..=1.0).contains(&r1));

        // DeltaJob: full beam writes the identical segment
        let ids: Vec<u32> = (0..n as u32).collect();
        let mut plain = DeltaJob::new(dim as u32, k as u32, &fine, 4).unwrap();
        let c_plain = plain.add(&xs, &ids, &[]).unwrap();
        let mut treed = DeltaJob::new(dim as u32, k as u32, &fine, 4).unwrap();
        treed.set_tree(&tree, groups).unwrap();
        assert_eq!(treed.add(&xs, &ids, &[]).unwrap(), c_plain);
        assert_eq!(treed.finish(), plain.finish());
        // narrow beam: cells agree with Searcher's pruned routing
        let mut narrow = DeltaJob::new(dim as u32, k as u32, &fine, 4).unwrap();
        narrow.set_tree(&tree, 1).unwrap();
        let c_narrow = narrow.add(&xs, &ids, &[]).unwrap();
        for (i, x) in xs.chunks_exact(dim).enumerate() {
            assert_eq!(c_narrow[i], s.route_beam(x, 1, 1).unwrap()[0]);
        }

        // AssignJob: full beam writes the same parts as the exhaustive search
        let mut exh = AssignJob::new(dim as u32, k1 as u32, k2 as u32, &coarse, &fine, k1 as u32, m).unwrap();
        exh.add_batch(&xs, 10).unwrap();
        let mut tj = AssignJob::new(dim as u32, k1 as u32, k2 as u32, &coarse, &fine, 1, m).unwrap();
        tj.set_tree(&tree, groups).unwrap();
        tj.add_batch(&xs, 10).unwrap();
        assert_eq!(tj.assigned_records(), exh.assigned_records());
        for g in 0..k1 as u32 {
            assert_eq!(tj.take_part(g), exh.take_part(g));
        }
        // flat (k1 = 1) with_candidates job: exhaustive add_batch path
        let nb = cell_neighbors(&fine, dim as u32, 0, k as u32, 4).unwrap();
        let mut flat = AssignJob::with_candidates(dim as u32, k as u32, &fine, &nb, 4, m).unwrap();
        flat.add_batch(&xs, 0).unwrap();
        let mut flat_t = AssignJob::with_candidates(dim as u32, k as u32, &fine, &nb, 4, m).unwrap();
        flat_t.set_tree(&tree, groups + 3).unwrap();
        flat_t.add_batch(&xs, 0).unwrap();
        assert_eq!(flat_t.take_part(0), flat.take_part(0));
    }

    #[test]
    fn delta_update_scan_and_compaction() {
        let (k, dim, n, pq_m) = (6, 32, 300, 8);
        let (means, xs) = setup(k, dim, n);
        let ones = vec![1.0f32; dim];
        // build a format 3 generation for the first 240 vectors (m = 1)
        let mut tj = TreeKmeansJob::new(dim as u32, 1, k as u32, &ones, &means, 1).unwrap();
        let assign = tj.add_batch_assign(&xs).unwrap();
        let res = residuals_for(&xs, &means, &assign, dim as u32).unwrap();
        let cb = pq_init_codebook(&res, dim as u32, pq_m, 5).unwrap();
        let base = 240;
        let mut aj = AssignJob::new(dim as u32, 1, k as u32, &ones, &means, 1, 1).unwrap();
        aj.set_codebook(&cb).unwrap();
        aj.add_batch(&xs[..base * dim], 0).unwrap();
        let part = aj.take_part(0);
        let dir = part_directory(&part).unwrap();
        let slab = |fmt: u8, cell: usize| {
            let (off, len) = (dir[2 + cell * 3 + 1], dir[2 + cell * 3 + 2]);
            let mut sj = SlabJob::new(dim as u32, cell as u32, &means[cell * dim..(cell + 1) * dim], Some(fmt)).unwrap();
            sj.add_records(&part[off as usize..(off + len) as usize]).unwrap();
            sj.finish()
        };

        // the update: the last 60 vectors (ids 240..300) go into one delta segment
        let mut dj = DeltaJob::new(dim as u32, k as u32, &means, 7).unwrap();
        dj.set_codebook(&cb).unwrap();
        assert_eq!(dj.pq_m(), pq_m);
        let new = &xs[base * dim..];
        let ids: Vec<u32> = (base as u32..n as u32).collect();
        let half = 30;
        let c1 = dj.add(&new[..half * dim], &ids[..half], &[]).unwrap();
        let payloads: Vec<u8> = (half..60).flat_map(|i| vec![i as u8; i % 3]).collect();
        let lens: Vec<u32> = (half..60).map(|i| (i % 3) as u32).collect();
        let attrs: Vec<u32> = (half..60).map(|i| i as u32).collect();
        let c2 = dj.add_with_payloads(&new[half * dim..], &ids[half..], &attrs, &payloads, &lens).unwrap();
        let cells: Vec<u32> = c1.into_iter().chain(c2).collect();
        // the primary cell is the exhaustive flat assignment
        assert_eq!(cells, assign[base..]);
        assert_eq!(dj.len(), 60);
        let bl = dj.byte_len();
        let delta = dj.finish();
        assert_eq!(delta.len() as u32, bl);
        assert_eq!(delta_info(&delta).unwrap(), vec![dim as u32, 60, pq_m, 7, bl]);
        let dc = delta_cells(&delta).unwrap();
        assert_eq!(dc.chunks_exact(2).map(|p| p[1]).sum::<u32>(), 60);
        assert_eq!(delta_payload(&delta, 35).unwrap(), vec![35u8; 2]);

        // query: scan_delta with Searcher.cell_dots finds a new vector with score ~1
        let searcher = Searcher::new(dim as u32, 1, k as u32, &ones, &means).unwrap();
        let q = xs[(base + 40) * dim..(base + 41) * dim].to_vec();
        let qc = searcher.cell_dots(&q).unwrap();
        assert_eq!(searcher.route_from_dots(&qc, 3).unwrap(), searcher.route_flat(&q, 3).unwrap());
        let hits = unpack_hits(&scan_delta(&delta, &q, &qc, 5, &[], &mut []).unwrap());
        assert_eq!((hits[0].id, hits[0].idx), (base as u32 + 40, 40));
        assert!((hits[0].score - 1.0).abs() < 5e-3);
        // a tombstone removes it
        let tomb = tombstones_merge(&[], &[base as u32 + 40, 3]).unwrap();
        let tomb = tombstones_merge(&tomb, &[3, 1]).unwrap();
        assert_eq!(tomb.len(), 12);
        let kept = unpack_hits(&filter_hits(&scan_delta(&delta, &q, &qc, 5, &[], &mut []).unwrap(), &tomb).unwrap());
        assert_eq!(kept.len(), 4);
        assert!(kept.iter().all(|h| h.id != base as u32 + 40));
        // in the scan it does not take a slot: k = 1 returns the second-best entry, k = 5 still 5 hits
        let mut st = [0u32; 2];
        let one = unpack_hits(&scan_delta(&delta, &q, &qc, 1, &tomb, &mut st).unwrap());
        assert_eq!((one, st), (vec![hits[1]], [60, 1]));
        let five = unpack_hits(&scan_delta(&delta, &q, &qc, 5, &tomb, &mut st).unwrap());
        assert_eq!((five.len(), &five[..4]), (5, &hits[1..]));
        // merge is unchanged
        let packed = scan_delta(&delta, &q, &qc, 5, &[], &mut []).unwrap();
        assert_eq!(unpack_hits(&merge(&[packed.clone(), packed].concat(), 3)), hits[..3].to_vec());

        // compaction of every cell, formats 3 and 2
        let dead = [1u32, 3, base as u32 + 40];
        for fmt in [3u8, 2] {
            let mut total = 0;
            for cell in 0..k {
                let old = slab(fmt, cell);
                let mean = &means[cell * dim..(cell + 1) * dim];
                // rebuilding from its own records is byte-identical
                let mut same = SlabJob::new(dim as u32, cell as u32, mean, Some(fmt)).unwrap();
                same.add_records(&slab_to_records(&old, &[]).unwrap()).unwrap();
                assert_eq!(same.finish(), old);
                let mut sj = SlabJob::new(dim as u32, cell as u32, mean, Some(fmt)).unwrap();
                sj.add_records(&slab_to_records(&old, &tomb).unwrap()).unwrap();
                sj.add_records(&delta_records_for_cell(&delta, cell as u32, &tomb).unwrap()).unwrap();
                let merged = sj.finish();
                let info = slab_info(&merged).unwrap();
                assert_eq!(info[4], fmt as u32);
                let v = SlabView::parse(&merged).unwrap();
                for i in 0..v.count {
                    let id = v.id(i) as usize;
                    assert!(!dead.contains(&(id as u32)));
                    assert_eq!(assign[id] as usize, cell);
                }
                total += v.count;
                // split proposal for the compacted cell, then the split itself
                if v.count >= 2 {
                    let two = split_slab(&merged, mean, 1).unwrap();
                    assert_eq!(two.len(), 2 * dim);
                    assert_eq!(split_slab(&merged, &[], 1).unwrap(), two);
                    let cbb: &[u8] = if fmt == 3 { &cb } else { &[] };
                    let packed = split_slab_records(&merged, &[], &two, cbb).unwrap();
                    let la = u32::from_le_bytes(packed[..4].try_into().unwrap()) as usize;
                    let (ra, rb) = (&packed[4..4 + la], &packed[4 + la..]);
                    let mut ja = SlabJob::new(dim as u32, cell as u32, &two[..dim], Some(fmt)).unwrap();
                    let mut jb = SlabJob::new(dim as u32, k as u32, &two[dim..], Some(fmt)).unwrap();
                    let (na, nb) = (ja.add_records(ra).unwrap(), jb.add_records(rb).unwrap());
                    assert_eq!((na + nb) as usize, v.count);
                    let (sa, sb) = (ja.finish(), jb.finish());
                    assert_eq!(slab_info(&sb).unwrap()[2], k as u32);
                    // every entry is still found with ~its exact score in its new slab (int8 tier)
                    for (s, m) in [(&sa, &two[..dim]), (&sb, &two[dim..])] {
                        let sv = SlabView::parse(s).unwrap();
                        assert_eq!(sv.centroid(), m);
                        for i in 0..sv.count {
                            let id = sv.id(i) as usize;
                            let x = &xs[id * dim..(id + 1) * dim];
                            let got = rerank(x, m, sv.rerank_record(i), 1).unwrap()[0];
                            assert!((got - 1.0).abs() < 0.02, "id {id}: self score {got}");
                        }
                    }
                    // the part-records path gives the same split; reencode_records against mean A keeps ids
                    let recs = slab_to_records(&merged, &[]).unwrap();
                    assert_eq!(split_part_records(&recs, mean, &two, cbb).unwrap(), packed);
                    let re = reencode_records(&recs, mean, &two[..dim], cbb).unwrap();
                    let mut jr = SlabJob::new(dim as u32, cell as u32, &two[..dim], Some(fmt)).unwrap();
                    assert_eq!(jr.add_records(&re).unwrap() as usize, v.count);
                }
            }
            assert_eq!(total, n - dead.len());
            // v3 records cannot build a format 2 slab
            if fmt == 3 {
                let recs = slab_to_records(&slab(3, 0), &[]).unwrap();
                let mut sj = SlabJob::new(dim as u32, 0, &means[..dim], Some(2)).unwrap();
                assert!(sj.builder.add_part_records(&recs, dim).is_err());
            }
        }
    }
}
