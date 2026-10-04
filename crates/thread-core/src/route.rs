// Dense-side file written before rustfmt was adopted; kept as is (`cargo fmt` skips it).
#![cfg_attr(rustfmt, rustfmt::skip)]
//! Two-level routing table: `k1` coarse centroids, each with `k2` fine cells.
//!
//! Stores raw means plus inverse norms, so cosine routing needs no second
//! unit-normalized copy. For 4096 fine cells at 1536 dims that is 25 MB
//! instead of 50 MB, which matters inside a 128 MB Worker.

use crate::error::{f32s_from_le, f32s_to_le, need, rd_u32, FormatError};
use crate::math::{argmax_dot, dot, inv_norms, normalize, normalized_rows, top_n_dot_scaled, Rng};

/// Top `n` of `dots[i] * inv[i]`, descending (same selection as
/// `math::top_n_dot_scaled`, from precomputed dot products). `n = 0` returns
/// an empty list.
pub fn top_n_scaled_dots(dots: &[f32], inv: &[f32], n: usize) -> Vec<(usize, f32)> {
    if n == 0 {
        return Vec::new();
    }
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(n + 1);
    for (i, (&d, &s)) in dots.iter().zip(inv).enumerate() {
        let d = d * s;
        if top.len() < n || d > top[top.len() - 1].1 {
            let pos = top.partition_point(|&(_, s)| s >= d);
            top.insert(pos, (i, d));
            if top.len() > n {
                top.pop();
            }
        }
    }
    top
}

#[derive(Clone)]
pub struct Router {
    pub dim: usize,
    pub k1: usize,
    pub k2: usize,
    coarse_mean: Vec<f32>,
    coarse_inv: Vec<f32>,
    fine_mean: Vec<f32>,
    fine_inv: Vec<f32>,
}

impl Router {
    /// `coarse_mean`: k1*dim, `fine_mean`: k1*k2*dim grouped by coarse.
    pub fn new(dim: usize, k1: usize, k2: usize, coarse_mean: &[f32], fine_mean: Vec<f32>) -> Self {
        assert_eq!(coarse_mean.len(), k1 * dim, "coarse centroid buffer size");
        assert_eq!(fine_mean.len(), k1 * k2 * dim, "fine centroid buffer size");
        let coarse_inv = inv_norms(coarse_mean, dim);
        let fine_inv = inv_norms(&fine_mean, dim);
        Router { dim, k1, k2, coarse_mean: coarse_mean.to_vec(), coarse_inv, fine_mean, fine_inv }
    }

    pub fn cells(&self) -> usize {
        self.k1 * self.k2
    }

    pub fn coarse_mean(&self) -> &[f32] {
        &self.coarse_mean
    }

    pub fn coarse_inv(&self) -> &[f32] {
        &self.coarse_inv
    }

    pub fn fine_mean_all(&self) -> &[f32] {
        &self.fine_mean
    }

    pub fn fine_inv(&self) -> &[f32] {
        &self.fine_inv
    }

    pub fn fine_mean(&self, cell: usize) -> &[f32] {
        &self.fine_mean[cell * self.dim..(cell + 1) * self.dim]
    }

    pub fn top_coarse(&self, q: &[f32], n: usize) -> Vec<(usize, f32)> {
        top_n_dot_scaled(q, &self.coarse_mean, &self.coarse_inv, self.dim, n)
    }

    /// The `nprobe` fine cells (global ids) with the highest cosine to `q`,
    /// ranked over all `k1 * k2` fine cells ("flat routing"). Costs
    /// `k1 * k2 * dim` multiply-adds but has no coarse-stage recall ceiling.
    pub fn route_flat(&self, q: &[f32], nprobe: usize) -> Vec<(usize, f32)> {
        top_n_dot_scaled(q, &self.fine_mean, &self.fine_inv, self.dim, nprobe)
    }

    /// `q·mean[c]` (unnormalized means) for all `k1 * k2` fine cells: the
    /// per-cell base scores a delta segment scan needs (`search::scan_delta`).
    pub fn cell_dots(&self, q: &[f32]) -> Vec<f32> {
        self.fine_mean.chunks_exact(self.dim).map(|r| dot(q, r)).collect()
    }

    /// `q·mean[c]` for the listed cells only, in order: the same values as
    /// `cell_dots(q)[c]`. Panics on a cell id `>= cells()`.
    pub fn cell_dots_for(&self, q: &[f32], cells: &[u32]) -> Vec<f32> {
        cells.iter().map(|&c| dot(q, self.fine_mean(c as usize))).collect()
    }

    /// Flat routing from precomputed [`Router::cell_dots`]: identical to
    /// [`Router::route_flat`] without recomputing the dot products.
    pub fn route_flat_from_dots(&self, dots: &[f32], nprobe: usize) -> Vec<(usize, f32)> {
        assert_eq!(dots.len(), self.cells(), "one dot per fine cell");
        top_n_scaled_dots(dots, &self.fine_inv, nprobe)
    }

    /// The `nprobe` fine cells (global ids) with the highest cosine to `q`,
    /// searched within the `probe_coarse` best coarse groups.
    /// `probe_coarse == 0` or `>= k1` means flat routing over every fine cell.
    pub fn route(&self, q: &[f32], probe_coarse: usize, nprobe: usize) -> Vec<(usize, f32)> {
        if probe_coarse == 0 || probe_coarse >= self.k1 {
            return self.route_flat(q, nprobe);
        }
        if nprobe == 0 {
            return Vec::new();
        }
        let mut top: Vec<(usize, f32)> = Vec::with_capacity(nprobe + 1);
        for (g, _) in self.top_coarse(q, probe_coarse) {
            let rows = &self.fine_mean[g * self.k2 * self.dim..(g + 1) * self.k2 * self.dim];
            for (j, r) in rows.chunks_exact(self.dim).enumerate() {
                let cell = g * self.k2 + j;
                let d = dot(q, r) * self.fine_inv[cell];
                if top.len() < nprobe || d > top[top.len() - 1].1 {
                    let pos = top.partition_point(|&(_, s)| s >= d);
                    top.insert(pos, (cell, d));
                    if top.len() > nprobe {
                        top.pop();
                    }
                }
            }
        }
        top
    }
}

/// Index of the cell-mean tree ("VIBC") format.
pub const TREE_MAGIC: &[u8; 4] = b"VIBC";
pub const TREE_VERSION: u32 = 1;
/// `[magic][version u32][dim u32][k u32][groups u32][n_members u32][reserved u32 * 2]`.
pub const TREE_HEADER: usize = 32;

/// Inserts `(cell, score)` into `top` (score descending, ties: lower cell id
/// first) if it ranks within the first `n` and is not already listed.
#[inline]
fn push_top_unique(top: &mut Vec<(usize, f32)>, n: usize, cell: usize, d: f32) {
    if n == 0 || d.is_nan() {
        return;
    }
    if top.len() >= n {
        let (lc, ls) = top[top.len() - 1];
        if !(d > ls || (d == ls && cell < lc)) {
            return;
        }
    }
    if top.iter().any(|&(c, _)| c == cell) {
        return;
    }
    let pos = top.partition_point(|&(c, s)| s > d || (s == d && c < cell));
    top.insert(pos, (cell, d));
    if top.len() > n {
        top.pop();
    }
}

/// Pruned search structure over a table of `k` cell means: the means are
/// clustered (spherical k-means) into `groups` groups; a lookup scores the
/// group means, keeps the best `beam` groups and scores only their member
/// cells. With `k = 4096`, `groups = 64`, `beam = 8` that is ~64 + ~512 dot
/// products instead of 4096.
///
/// The tree holds no copy of the cell means: every lookup takes the same raw
/// `means` (`k * dim`) and inverse norms (`math::inv_norms`) the exhaustive
/// search uses, and scores a cell exactly as [`crate::kmeans::assign_flat`] /
/// [`Router::route_flat`] do (`dot(x, mean) * inv`), so with `beam >= groups`
/// the results are identical to the exhaustive ones.
///
/// Serialized ("VIBC" v1, little-endian):
/// ```text
/// header (32 bytes): magic "VIBC", version u32 = 1, dim u32, k u32, groups u32,
///                    n_members u32, reserved u32 * 2
/// group means      groups * dim f32 (unit length)
/// group inv norms  groups f32
/// offsets          (groups + 1) u32   CSR: group g's members are
/// members          n_members u32      members[offsets[g]..offsets[g + 1]]
/// ```
/// A built tree lists every cell exactly once; the format (and every lookup)
/// also allows a cell in several groups.
#[derive(Clone, Debug, PartialEq)]
pub struct CentroidTree {
    pub dim: usize,
    pub k: usize,
    group_mean: Vec<f32>,
    group_inv: Vec<f32>,
    offsets: Vec<u32>,
    members: Vec<u32>,
}

impl CentroidTree {
    /// Clusters the `k` cell means (`k * dim`, raw) into `groups` groups
    /// (clamped to `1..=k`) by spherical k-means on their directions:
    /// k-means++ seeding (cosine distance), then `iters` Lloyd iterations
    /// (assign each cell to the group mean with the highest cosine; a group
    /// mean is the normalized sum of its cells' unit directions), stopping
    /// early once no assignment changes. Groups never end up empty (an empty
    /// group takes over the cell farthest from its own group mean).
    /// Deterministic for a given `seed`. Cost: `(iters + 2) * groups * k * dim`
    /// multiply-adds.
    pub fn build(means: &[f32], dim: usize, k: usize, groups: usize, iters: usize, seed: u64) -> CentroidTree {
        assert!(dim > 0 && k > 0, "dim and k must be positive");
        assert_eq!(means.len(), k * dim, "means buffer size != k * dim");
        let g = groups.clamp(1, k);
        let unit = normalized_rows(means, dim);
        let row = |i: usize| &unit[i * dim..(i + 1) * dim];
        let mut rng = Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15);

        // k-means++ seeding with distance 1 - cos (zero rows: distance 1).
        let mut centers: Vec<f32> = Vec::with_capacity(g * dim);
        let mut chosen = vec![false; k];
        let first = rng.below(k as u64) as usize;
        centers.extend_from_slice(row(first));
        chosen[first] = true;
        let mut best_cos = vec![f32::NEG_INFINITY; k];
        for c in 1..g {
            let last = &centers[(c - 1) * dim..c * dim];
            for i in 0..k {
                let d = dot(row(i), last);
                if d > best_cos[i] {
                    best_cos[i] = d;
                }
            }
            let w = |i: usize| if chosen[i] { 0.0 } else { (1.0 - best_cos[i] as f64).max(0.0) };
            let total: f64 = (0..k).map(w).sum();
            let pick = if total > 0.0 {
                let mut t = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
                let mut pick = usize::MAX;
                for i in 0..k {
                    let wi = w(i);
                    if wi > 0.0 {
                        pick = i;
                        if t < wi {
                            break;
                        }
                        t -= wi;
                    }
                }
                pick
            } else {
                // every remaining cell duplicates a center: take any unchosen one
                let start = rng.below(k as u64) as usize;
                (0..k).map(|j| (start + j) % k).find(|&i| !chosen[i]).unwrap()
            };
            chosen[pick] = true;
            centers.extend_from_slice(row(pick));
        }

        let mut assign = vec![u32::MAX; k];
        let mut cos = vec![0f32; k];
        let assign_all = |centers: &[f32], assign: &mut [u32], cos: &mut [f32]| -> bool {
            let mut changed = false;
            for i in 0..k {
                let (c, d) = argmax_dot(row(i), centers, dim);
                changed |= assign[i] != c as u32;
                assign[i] = c as u32;
                cos[i] = d;
            }
            changed
        };
        // Gives every empty group the cell (from a group with >= 2 cells) that is
        // farthest from its group mean; the group mean becomes that cell.
        let fix_empty = |centers: &mut [f32], assign: &mut [u32], cos: &mut [f32]| {
            let mut count = vec![0usize; g];
            for &a in assign.iter() {
                count[a as usize] += 1;
            }
            for e in 0..g {
                if count[e] > 0 {
                    continue;
                }
                let far = (0..k)
                    .filter(|&i| count[assign[i] as usize] >= 2)
                    .min_by(|&a, &b| cos[a].total_cmp(&cos[b]).then(a.cmp(&b)));
                let Some(i) = far else { break };
                count[assign[i] as usize] -= 1;
                count[e] = 1;
                assign[i] = e as u32;
                cos[i] = 1.0;
                centers[e * dim..(e + 1) * dim].copy_from_slice(row(i));
            }
        };

        assign_all(&centers, &mut assign, &mut cos);
        fix_empty(&mut centers, &mut assign, &mut cos);
        for _ in 0..iters {
            // update: normalized sum of member directions
            let mut sums = vec![0f32; g * dim];
            for i in 0..k {
                let s = &mut sums[assign[i] as usize * dim..(assign[i] as usize + 1) * dim];
                for (a, b) in s.iter_mut().zip(row(i)) {
                    *a += b;
                }
            }
            for (c, s) in sums.chunks_exact_mut(dim).enumerate() {
                if s.iter().any(|v| *v != 0.0) {
                    normalize(s);
                    centers[c * dim..(c + 1) * dim].copy_from_slice(s);
                }
            }
            let changed = assign_all(&centers, &mut assign, &mut cos);
            fix_empty(&mut centers, &mut assign, &mut cos);
            if !changed {
                break;
            }
        }

        // CSR membership lists, cells ascending within a group
        let mut offsets = vec![0u32; g + 1];
        for &a in &assign {
            offsets[a as usize + 1] += 1;
        }
        for c in 0..g {
            offsets[c + 1] += offsets[c];
        }
        let mut fill: Vec<u32> = offsets[..g].to_vec();
        let mut members = vec![0u32; k];
        for (i, &a) in assign.iter().enumerate() {
            members[fill[a as usize] as usize] = i as u32;
            fill[a as usize] += 1;
        }
        let group_inv = inv_norms(&centers, dim);
        CentroidTree { dim, k, group_mean: centers, group_inv, offsets, members }
    }

    pub fn groups(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Total membership list length (`k` for a built tree).
    pub fn n_members(&self) -> usize {
        self.members.len()
    }

    pub fn group_means(&self) -> &[f32] {
        &self.group_mean
    }

    pub fn group_inv(&self) -> &[f32] {
        &self.group_inv
    }

    /// Cell ids of group `g`.
    pub fn members(&self, g: usize) -> &[u32] {
        &self.members[self.offsets[g] as usize..self.offsets[g + 1] as usize]
    }

    #[inline]
    fn check_args(&self, x: &[f32], means: &[f32], means_inv: &[f32]) {
        assert_eq!(x.len(), self.dim, "vector dim != tree dim");
        assert_eq!(means.len(), self.k * self.dim, "means buffer size != tree k * dim");
        assert_eq!(means_inv.len(), self.k, "inverse norms != tree k");
    }

    /// The `beam` groups (clamped to `1..=groups`) whose means have the highest cosine to `x`.
    pub fn top_groups(&self, x: &[f32], beam: usize) -> Vec<(usize, f32)> {
        assert_eq!(x.len(), self.dim, "vector dim != tree dim");
        top_n_dot_scaled(x, &self.group_mean, &self.group_inv, self.dim, beam.clamp(1, self.groups()))
    }

    /// Top `m` cells by cosine (`dot(x, means[c]) * means_inv[c]`), descending,
    /// ties to the lower cell id, among the members of the best `beam` groups.
    /// `beam >= groups` gives the exact top `m` over all cells (as
    /// [`Router::route_flat`]). Fewer than `m` only if those groups hold fewer
    /// than `m` distinct cells.
    pub fn top_m(&self, x: &[f32], means: &[f32], means_inv: &[f32], beam: usize, m: usize) -> Vec<(usize, f32)> {
        self.check_args(x, means, means_inv);
        let dim = self.dim;
        let mut top: Vec<(usize, f32)> = Vec::with_capacity(m + 1);
        for (g, _) in self.top_groups(x, beam) {
            for &c in self.members(g) {
                let c = c as usize;
                let d = dot(x, &means[c * dim..(c + 1) * dim]) * means_inv[c];
                push_top_unique(&mut top, m, c, d);
            }
        }
        top
    }

    /// Nearest cell (highest cosine) and its score among the members of the
    /// best `beam` groups; `beam >= groups` equals [`crate::kmeans::assign_flat`].
    /// Falls back to the exhaustive search if those groups are empty (never
    /// the case for a built tree).
    pub fn nearest(&self, x: &[f32], means: &[f32], means_inv: &[f32], beam: usize) -> (usize, f32) {
        self.check_args(x, means, means_inv);
        let dim = self.dim;
        let mut best = (usize::MAX, f32::NEG_INFINITY);
        for (g, _) in self.top_groups(x, beam) {
            for &c in self.members(g) {
                let c = c as usize;
                let d = dot(x, &means[c * dim..(c + 1) * dim]) * means_inv[c];
                if best.0 == usize::MAX || d > best.1 || (d == best.1 && c < best.0) {
                    best = (c, d);
                }
            }
        }
        if best.0 == usize::MAX {
            return crate::math::argmax_dot_scaled(x, means, means_inv, dim);
        }
        best
    }

    /// Query routing: the `nprobe` best cells within the best `beam` groups
    /// (same as [`CentroidTree::top_m`]).
    pub fn route(&self, q: &[f32], means: &[f32], means_inv: &[f32], beam: usize, nprobe: usize) -> Vec<(usize, f32)> {
        self.top_m(q, means, means_inv, beam, nprobe)
    }

    /// Dot products a lookup with this `beam` computes for `x`: `groups` plus
    /// the members of the chosen groups.
    pub fn probe_cost(&self, x: &[f32], beam: usize) -> usize {
        self.groups() + self.top_groups(x, beam).iter().map(|&(g, _)| self.members(g).len()).sum::<usize>()
    }

    /// Fraction of the `sample` vectors (`n * dim`) whose exact nearest cell
    /// (as [`crate::kmeans::assign_flat`]) is what [`CentroidTree::nearest`]
    /// returns with this `beam`. 1.0 for an empty sample.
    pub fn recall_check(&self, sample: &[f32], means: &[f32], means_inv: &[f32], beam: usize) -> f32 {
        assert!(sample.len() % self.dim == 0, "sample length is not a multiple of dim");
        let n = sample.len() / self.dim;
        if n == 0 {
            return 1.0;
        }
        let hit = sample
            .chunks_exact(self.dim)
            .filter(|x| {
                let exact = crate::math::argmax_dot_scaled(x, means, means_inv, self.dim).0;
                self.nearest(x, means, means_inv, beam).0 == exact
            })
            .count();
        hit as f32 / n as f32
    }

    /// Mean fraction of each sample vector's exact top `m` cells that
    /// [`CentroidTree::top_m`] with this `beam` also returns. 1.0 for an empty sample.
    pub fn recall_check_top_m(&self, sample: &[f32], means: &[f32], means_inv: &[f32], beam: usize, m: usize) -> f32 {
        assert!(sample.len() % self.dim == 0, "sample length is not a multiple of dim");
        let n = sample.len() / self.dim;
        if n == 0 || m == 0 {
            return 1.0;
        }
        let mut found = 0usize;
        let mut total = 0usize;
        for x in sample.chunks_exact(self.dim) {
            let exact = top_n_dot_scaled(x, means, means_inv, self.dim, m);
            let got = self.top_m(x, means, means_inv, beam, m);
            total += exact.len();
            found += exact.iter().filter(|e| got.iter().any(|g| g.0 == e.0)).count();
        }
        if total == 0 { 1.0 } else { found as f32 / total as f32 }
    }

    pub fn byte_len(&self) -> usize {
        TREE_HEADER + 4 * (self.group_mean.len() + self.group_inv.len() + self.offsets.len() + self.members.len())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_len());
        out.extend_from_slice(TREE_MAGIC);
        for v in [TREE_VERSION, self.dim as u32, self.k as u32, self.groups() as u32, self.members.len() as u32, 0, 0] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        f32s_to_le(&self.group_mean, &mut out);
        f32s_to_le(&self.group_inv, &mut out);
        for v in self.offsets.iter().chain(&self.members) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        debug_assert_eq!(out.len(), self.byte_len());
        out
    }

    /// Parses and validates a serialized tree (sizes, CSR offsets, cell ids `< k`).
    pub fn from_bytes(buf: &[u8]) -> Result<CentroidTree, FormatError> {
        need(buf, TREE_HEADER)?;
        if &buf[0..4] != TREE_MAGIC {
            return Err(FormatError::BadMagic);
        }
        let version = rd_u32(buf, 4);
        if version != TREE_VERSION {
            return Err(FormatError::BadVersion(version.min(u16::MAX as u32) as u16));
        }
        let (dim, k, groups, n_members) =
            (rd_u32(buf, 8) as usize, rd_u32(buf, 12) as usize, rd_u32(buf, 16) as usize, rd_u32(buf, 20) as usize);
        if dim == 0 || k == 0 || groups == 0 {
            return Err(FormatError::Inconsistent("tree dim, k and groups must be positive"));
        }
        let words = (groups as u64) * (dim as u64) + groups as u64 + (groups as u64 + 1) + n_members as u64;
        let len = TREE_HEADER as u64 + 4 * words;
        if len > usize::MAX as u64 {
            return Err(FormatError::Inconsistent("tree too large"));
        }
        need(buf, len as usize)?;
        if buf.len() as u64 != len {
            return Err(FormatError::Inconsistent("tree byte length != header sizes"));
        }
        let mut off = TREE_HEADER;
        let group_mean = f32s_from_le(&buf[off..off + 4 * groups * dim]);
        off += 4 * groups * dim;
        let group_inv = f32s_from_le(&buf[off..off + 4 * groups]);
        off += 4 * groups;
        let offsets: Vec<u32> = (0..=groups).map(|i| rd_u32(buf, off + 4 * i)).collect();
        off += 4 * (groups + 1);
        let members: Vec<u32> = (0..n_members).map(|i| rd_u32(buf, off + 4 * i)).collect();
        if offsets[0] != 0 || offsets[groups] as usize != n_members || offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(FormatError::Inconsistent("tree offsets are not a CSR index over the members"));
        }
        if members.iter().any(|&c| c as usize >= k) {
            return Err(FormatError::Inconsistent("tree member id >= k"));
        }
        Ok(CentroidTree { dim, k, group_mean, group_inv, offsets, members })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{normalize, Rng};

    fn rand_rows(rng: &mut Rng, n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|_| rng.below(20001) as f32 / 10000.0 - 1.0).collect()
    }

    #[test]
    fn route_flat_matches_brute_force_and_route_zero() {
        let (dim, k1, k2) = (16, 4, 8);
        let mut rng = Rng::new(7);
        let coarse = rand_rows(&mut rng, k1, dim);
        let fine = rand_rows(&mut rng, k1 * k2, dim);
        let router = Router::new(dim, k1, k2, &coarse, fine.clone());
        for _ in 0..20 {
            let mut q = rand_rows(&mut rng, 1, dim);
            normalize(&mut q);
            // brute force cosine over all fine cells
            let mut exact: Vec<(usize, f32)> = fine
                .chunks_exact(dim)
                .enumerate()
                .map(|(i, r)| (i, dot(&q, r) / dot(r, r).sqrt()))
                .collect();
            exact.sort_by(|a, b| b.1.total_cmp(&a.1));
            let nprobe = 5;
            let flat = router.route_flat(&q, nprobe);
            assert_eq!(flat.len(), nprobe);
            for (got, want) in flat.iter().zip(&exact) {
                assert_eq!(got.0, want.0);
                assert!((got.1 - want.1).abs() < 1e-5);
            }
            // descending order
            assert!(flat.windows(2).all(|w| w[0].1 >= w[1].1));
            // nprobe = 0 never panics (flat, two-level, from dots)
            assert!(router.route_flat(&q, 0).is_empty());
            assert!(router.route(&q, 2, 0).is_empty());
            assert!(router.route(&q, 0, 0).is_empty());
            assert!(router.route_flat_from_dots(&router.cell_dots(&q), 0).is_empty());
            assert!(router.top_coarse(&q, 0).is_empty());
            // probe_coarse = 0 and probe_coarse >= k1 both mean flat
            assert_eq!(router.route(&q, 0, nprobe), flat);
            assert_eq!(router.route(&q, k1, nprobe), flat);
            assert_eq!(router.route(&q, k1 + 3, nprobe), flat);
            // routing from precomputed cell dots is identical
            let dots = router.cell_dots(&q);
            assert_eq!(dots.len(), k1 * k2);
            for (c, d) in dots.iter().enumerate() {
                assert_eq!(*d, dot(&q, &fine[c * dim..(c + 1) * dim]));
            }
            assert_eq!(router.route_flat_from_dots(&dots, nprobe), flat);
            assert_eq!(router.route_flat_from_dots(&dots, 1000), router.route_flat(&q, 1000));
            // two-level routing never beats flat on the best cell's score
            let two = router.route(&q, 1, nprobe);
            assert!(two[0].1 <= flat[0].1);
        }
        // nprobe larger than the number of cells returns every cell
        let mut q = rand_rows(&mut rng, 1, dim);
        normalize(&mut q);
        assert_eq!(router.route_flat(&q, 1000).len(), k1 * k2);
    }
}

#[cfg(test)]
mod tree_tests {
    use super::*;
    use crate::kmeans::assign_flat;
    use crate::math::{normalize, Rng};

    fn uni(rng: &mut Rng) -> f32 {
        rng.below(20001) as f32 / 10000.0 - 1.0
    }

    fn rand_unit(rng: &mut Rng, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| uni(rng)).collect();
        normalize(&mut v);
        v
    }

    /// `a + s * noise`, normalized.
    fn jitter(rng: &mut Rng, a: &[f32], s: f32) -> Vec<f32> {
        let mut v: Vec<f32> = a.iter().map(|x| x + s * uni(rng)).collect();
        normalize(&mut v);
        v
    }

    /// Raw (non-unit, varied norm) means of `k` cells around `clusters` planted
    /// directions.
    fn clustered_means(rng: &mut Rng, k: usize, clusters: usize, dim: usize, spread: f32) -> Vec<f32> {
        let centres: Vec<Vec<f32>> = (0..clusters).map(|_| rand_unit(rng, dim)).collect();
        let mut out = Vec::with_capacity(k * dim);
        for c in 0..k {
            let scale = 0.3 + (c % 7) as f32 * 0.1;
            let centre = &centres[rng.below(clusters as u64) as usize];
            out.extend(jitter(rng, centre, spread).into_iter().map(|v| v * scale));
        }
        out
    }

    /// Points near random cells (what an upsert or a query looks like).
    fn sample_near_cells(rng: &mut Rng, means: &[f32], dim: usize, n: usize, s: f32) -> Vec<f32> {
        let k = means.len() / dim;
        let mut out = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let c = rng.below(k as u64) as usize;
            let mut m = means[c * dim..(c + 1) * dim].to_vec();
            normalize(&mut m);
            out.extend(jitter(rng, &m, s));
        }
        out
    }

    fn check_partition(t: &CentroidTree) {
        let mut seen = vec![0u32; t.k];
        for g in 0..t.groups() {
            assert!(!t.members(g).is_empty(), "group {g} empty");
            assert!(t.members(g).windows(2).all(|w| w[0] < w[1]));
            for &c in t.members(g) {
                seen[c as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "every cell in exactly one group");
        assert_eq!(t.n_members(), t.k);
    }

    #[test]
    fn build_partitions_and_roundtrips() {
        let mut rng = Rng::new(3);
        let (k, dim, groups) = (300, 24, 16);
        let means = clustered_means(&mut rng, k, 10, dim, 0.6);
        let t = CentroidTree::build(&means, dim, k, groups, 5, 7);
        assert_eq!((t.dim, t.k, t.groups()), (dim, k, groups));
        check_partition(&t);
        for g in 0..groups {
            let r = &t.group_means()[g * dim..(g + 1) * dim];
            assert!((dot(r, r) - 1.0).abs() < 1e-4);
            assert!((t.group_inv()[g] - 1.0).abs() < 1e-3);
        }
        // deterministic
        assert_eq!(CentroidTree::build(&means, dim, k, groups, 5, 7), t);
        // roundtrip
        let bytes = t.to_bytes();
        assert_eq!(bytes.len(), t.byte_len());
        assert_eq!(&bytes[..4], TREE_MAGIC);
        let back = CentroidTree::from_bytes(&bytes).unwrap();
        assert_eq!(back, t);
        assert_eq!(back.to_bytes(), bytes);
        // corruption is rejected
        assert!(CentroidTree::from_bytes(&bytes[..bytes.len() - 4]).is_err());
        assert!(CentroidTree::from_bytes(&[bytes.clone(), vec![0; 4]].concat()).is_err());
        assert!(CentroidTree::from_bytes(&bytes[..10]).is_err());
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert_eq!(CentroidTree::from_bytes(&bad), Err(FormatError::BadMagic));
        let mut bad = bytes.clone();
        bad[4] = 9;
        assert_eq!(CentroidTree::from_bytes(&bad), Err(FormatError::BadVersion(9)));
        let off0 = TREE_HEADER + 4 * (groups * dim + groups);
        let mut bad = bytes.clone();
        bad[off0 + 4..off0 + 8].copy_from_slice(&(k as u32 + 1).to_le_bytes()); // offsets[1] > n_members
        assert!(CentroidTree::from_bytes(&bad).is_err());
        let mut bad = bytes.clone();
        let last = bad.len() - 4;
        bad[last..].copy_from_slice(&(k as u32).to_le_bytes()); // member id == k
        assert!(CentroidTree::from_bytes(&bad).is_err());
        // a hand-made tree with overlapping and empty groups parses and dedups
        let mut multi = t.clone();
        multi.offsets = vec![0, k as u32, k as u32, 2 * k as u32];
        multi.members = (0..k as u32).chain(0..k as u32).collect();
        multi.group_mean = t.group_mean[..3 * dim].to_vec();
        multi.group_inv = t.group_inv[..3].to_vec();
        let multi = CentroidTree::from_bytes(&multi.to_bytes()).unwrap();
        let inv = inv_norms(&means, dim);
        let x = rand_unit(&mut rng, dim);
        let exact = top_n_dot_scaled(&x, &means, &inv, dim, 10);
        assert_eq!(multi.top_m(&x, &means, &inv, 3, 10), exact);
        assert_eq!(multi.nearest(&x, &means, &inv, 3), exact[0]);
    }

    #[test]
    fn group_count_edge_cases() {
        let mut rng = Rng::new(4);
        let (k, dim) = (20, 8);
        let means: Vec<f32> = (0..k).flat_map(|_| rand_unit(&mut rng, dim)).collect();
        let inv = inv_norms(&means, dim);
        for (groups, want) in [(0usize, 1usize), (1, 1), (7, 7), (20, 20), (50, 20)] {
            for iters in [0usize, 1, 10] {
                let t = CentroidTree::build(&means, dim, k, groups, iters, 1);
                assert_eq!(t.groups(), want);
                check_partition(&t);
                let x = rand_unit(&mut rng, dim);
                assert_eq!(t.nearest(&x, &means, &inv, want).0, assign_flat(&x, &means, &inv, dim));
            }
        }
        // duplicate means (all identical) still give non-empty groups
        let dup: Vec<f32> = (0..k).flat_map(|_| means[..dim].to_vec()).collect();
        let t = CentroidTree::build(&dup, dim, k, 5, 3, 2);
        check_partition(&t);
        // k = 1
        let t = CentroidTree::build(&means[..dim], dim, 1, 64, 3, 2);
        assert_eq!(t.groups(), 1);
        assert_eq!(t.nearest(&means[..dim], &means[..dim], &inv[..1], 4).0, 0);
    }

    #[test]
    fn full_beam_equals_exhaustive() {
        let mut rng = Rng::new(11);
        let (k, dim, groups) = (257, 32, 12);
        let means: Vec<f32> =
            (0..k * dim).map(|i| uni(&mut rng) * (0.5 + (i / dim % 5) as f32 * 0.2)).collect();
        let inv = inv_norms(&means, dim);
        let t = CentroidTree::build(&means, dim, k, groups, 6, 5);
        let router = Router::new(dim, 1, k, &vec![1.0; dim], means.clone());
        let xs: Vec<f32> = (0..200).flat_map(|_| rand_unit(&mut rng, dim)).collect();
        for x in xs.chunks_exact(dim) {
            for beam in [groups, groups + 5, usize::MAX] {
                let (c, s) = t.nearest(x, &means, &inv, beam);
                assert_eq!(c, assign_flat(x, &means, &inv, dim));
                assert_eq!(s, dot(x, &means[c * dim..(c + 1) * dim]) * inv[c]);
                for nprobe in [1usize, 3, 17, 300] {
                    assert_eq!(t.route(x, &means, &inv, beam, nprobe), router.route_flat(x, nprobe));
                }
                assert_eq!(t.top_m(x, &means, &inv, beam, 4), router.route_flat(x, 4));
            }
            // pruned results are a subset in descending order, never better than exact
            let p = t.top_m(x, &means, &inv, 2, 8);
            assert!(p.windows(2).all(|w| w[0].1 >= w[1].1));
            assert!(p[0].1 <= router.route_flat(x, 1)[0].1);
            assert_eq!(t.nearest(x, &means, &inv, 2), p[0]);
            assert_eq!(t.probe_cost(x, groups), groups + k);
            // beam 0 is treated as 1
            assert_eq!(t.top_m(x, &means, &inv, 0, 8), t.top_m(x, &means, &inv, 1, 8));
        }
        assert_eq!(t.recall_check(&xs, &means, &inv, groups), 1.0);
        assert_eq!(t.recall_check_top_m(&xs, &means, &inv, groups, 5), 1.0);
        assert_eq!(t.recall_check(&[], &means, &inv, 1), 1.0);
    }

    #[test]
    fn small_beam_recall_on_clustered_cells() {
        // 2048 cells from 64 planted clusters; sample points near random cells.
        let mut rng = Rng::new(21);
        let (k, dim, groups, clusters) = (2048, 64, 64, 64);
        // cells: cos ~0.8 to their planted direction; points: cos ~0.95 to their cell
        let means = clustered_means(&mut rng, k, clusters, dim, 0.25);
        let inv = inv_norms(&means, dim);
        let t = CentroidTree::build(&means, dim, k, groups, 8, 1);
        check_partition(&t);
        let xs = sample_near_cells(&mut rng, &means, dim, 2000, 0.1);
        let mut prev = 0.0f32;
        for beam in [1usize, 2, 4, 8, 16, 64] {
            let r = t.recall_check(&xs, &means, &inv, beam);
            let r3 = t.recall_check_top_m(&xs, &means, &inv, beam, 3);
            assert!(r >= prev, "recall must not drop with a wider beam");
            prev = r;
            if beam >= 4 {
                assert!(r >= 0.99, "beam {beam}: exact-nearest recall {r}");
                assert!(r3 >= 0.98, "beam {beam}: top-3 recall {r3}");
            }
        }
        assert_eq!(prev, 1.0);
        // the pruned lookup touches a small fraction of the cells
        let cost: usize = xs.chunks_exact(dim).map(|x| t.probe_cost(x, 8)).sum::<usize>() / 2000;
        assert!(cost < k / 2, "beam 8 probes {cost} cells of {k}");
    }

    #[test]
    fn router_cell_dots_for_matches_cell_dots() {
        let mut rng = Rng::new(2);
        let (dim, k1, k2) = (16, 3, 5);
        let fine: Vec<f32> = (0..k1 * k2 * dim).map(|_| uni(&mut rng)).collect();
        let router = Router::new(dim, k1, k2, &vec![0.5; k1 * dim], fine);
        let q = rand_unit(&mut rng, dim);
        let all = router.cell_dots(&q);
        let cells = [14u32, 0, 3, 3, 7];
        let got = router.cell_dots_for(&q, &cells);
        assert_eq!(got, cells.iter().map(|&c| all[c as usize]).collect::<Vec<_>>());
        assert!(router.cell_dots_for(&q, &[]).is_empty());
    }
}
