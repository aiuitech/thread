// Dense-side file written before rustfmt was adopted; kept as is (`cargo fmt` skips it).
#![cfg_attr(rustfmt, rustfmt::skip)]
//! Streaming, map-reduce friendly spherical k-means.
//!
//! A *map* job streams vectors, assigns each to a centroid and adds it into an
//! [`Accum`] (per-cell counts and sums). Partial accumulators are serialized,
//! written to storage and later *reduced* cell-range by cell-range, which
//! keeps every job's memory bounded by `cells_in_range * dim * 4` bytes rather
//! than the dataset size.
//!
//! Serialized accumulator layout (little-endian):
//! `[magic "VIBA"][k: u32][dim: u32][reserved: u32]` then `k` records of
//! `[count: u32][sum: f32 * dim]`. Records are contiguous, so the records for
//! cells `[a, b)` are one byte range.

use crate::error::{need, rd_u32, FormatError};
use crate::math::{dot, normalize, top_n_dot_scaled};

pub const ACCUM_MAGIC: &[u8; 4] = b"VIBA";
pub const ACCUM_HEADER: usize = 16;

#[derive(Clone, Debug)]
pub struct Accum {
    pub dim: usize,
    pub k: usize,
    pub counts: Vec<u32>,
    pub sums: Vec<f32>,
}

impl Accum {
    pub fn new(k: usize, dim: usize) -> Self {
        Accum { dim, k, counts: vec![0; k], sums: vec![0.0; k * dim] }
    }

    #[inline]
    pub fn add(&mut self, cell: usize, x: &[f32]) {
        debug_assert_eq!(x.len(), self.dim);
        self.counts[cell] += 1;
        let s = &mut self.sums[cell * self.dim..(cell + 1) * self.dim];
        for i in 0..self.dim {
            s[i] += x[i];
        }
    }

    pub fn merge(&mut self, other: &Accum) {
        assert_eq!(self.k, other.k);
        assert_eq!(self.dim, other.dim);
        for i in 0..self.k {
            self.counts[i] += other.counts[i];
        }
        for i in 0..self.sums.len() {
            self.sums[i] += other.sums[i];
        }
    }

    pub const fn record_bytes(dim: usize) -> usize {
        4 + dim * 4
    }

    /// Byte range `(offset, len)` within a serialized accumulator that holds the
    /// records for cells `[first, first + n)`.
    pub fn record_range(dim: usize, first: usize, n: usize) -> (usize, usize) {
        let rb = Self::record_bytes(dim);
        (ACCUM_HEADER + first * rb, n * rb)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(ACCUM_HEADER + self.k * Self::record_bytes(self.dim));
        out.extend_from_slice(ACCUM_MAGIC);
        out.extend_from_slice(&(self.k as u32).to_le_bytes());
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        self.append_records(&mut out, 0, self.k);
        out
    }

    /// Appends the raw records for cells `[first, first+n)` (no header).
    pub fn append_records(&self, out: &mut Vec<u8>, first: usize, n: usize) {
        for c in first..first + n {
            out.extend_from_slice(&self.counts[c].to_le_bytes());
            for v in &self.sums[c * self.dim..(c + 1) * self.dim] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Accum, FormatError> {
        need(buf, ACCUM_HEADER)?;
        if &buf[0..4] != ACCUM_MAGIC {
            return Err(FormatError::BadMagic);
        }
        let k = rd_u32(buf, 4) as usize;
        let dim = rd_u32(buf, 8) as usize;
        let mut a = Accum::new(k, dim);
        a.add_records(&buf[ACCUM_HEADER..], 0)?;
        Ok(a)
    }

    /// Adds raw records (as produced by `append_records`) into cells starting
    /// at `first`. Used by reduce jobs that only hold a cell range.
    pub fn add_records(&mut self, records: &[u8], first: usize) -> Result<(), FormatError> {
        let rb = Self::record_bytes(self.dim);
        if records.len() % rb != 0 {
            return Err(FormatError::Inconsistent("record bytes not a multiple of record size"));
        }
        let n = records.len() / rb;
        if first + n > self.k {
            return Err(FormatError::Inconsistent("records exceed accumulator cells"));
        }
        for (i, rec) in records.chunks_exact(rb).enumerate() {
            let c = first + i;
            self.counts[c] += rd_u32(rec, 0);
            let s = &mut self.sums[c * self.dim..(c + 1) * self.dim];
            for j in 0..self.dim {
                s[j] += f32::from_le_bytes([
                    rec[4 + j * 4],
                    rec[5 + j * 4],
                    rec[6 + j * 4],
                    rec[7 + j * 4],
                ]);
            }
        }
        Ok(())
    }

    /// New centroid means. Empty cells keep their previous centroid so they do
    /// not collapse to the origin.
    pub fn means(&self, prev: &[f32]) -> Vec<f32> {
        assert_eq!(prev.len(), self.k * self.dim);
        let mut out = prev.to_vec();
        for c in 0..self.k {
            if self.counts[c] > 0 {
                let inv = 1.0 / self.counts[c] as f32;
                let s = &self.sums[c * self.dim..(c + 1) * self.dim];
                let o = &mut out[c * self.dim..(c + 1) * self.dim];
                for j in 0..self.dim {
                    o[j] = s[j] * inv;
                }
            }
        }
        out
    }

    /// Like [`Accum::means`], but every cell with `count < min_count` (empty
    /// cells included) is reseeded next to a dense cell so that tiny or
    /// duplicate cells get recycled toward dense regions on the next Lloyd
    /// iteration. Returns the new means and how many cells were reseeded.
    ///
    /// The donor is the cell with the largest count in this accumulator. To
    /// avoid stacking every reseed on one donor, each reseed splits the donor's
    /// count between donor and recipient (halving both "effective" counts), so
    /// the next reseed goes to whichever cell is now the largest; with a single
    /// small cell this is exactly "reseed to the largest cell's mean". The new
    /// mean is the donor mean plus deterministic jitter of up to ±1e-3 relative
    /// per dimension (relative to the donor's RMS component, so zero components
    /// are perturbed too). Donors with an effective count below 2 are not split.
    pub fn means_reseed(&self, prev: &[f32], min_count: u32, rng_seed: u64) -> (Vec<f32>, u32) {
        let mut out = self.means(prev);
        let dim = self.dim;
        let mut small: Vec<usize> = (0..self.k).filter(|&c| self.counts[c] < min_count).collect();
        if small.is_empty() {
            return (out, 0);
        }
        // reseed the smallest cells first
        small.sort_by_key(|&c| (self.counts[c], c));
        // small cells never donate unless they were already reseeded (then they
        // stand for half of a dense cell).
        let mut eff: Vec<f64> =
            self.counts.iter().map(|&n| if n < min_count { 0.0 } else { n as f64 }).collect();
        let mut rng = crate::math::Rng::new(rng_seed);
        let mut reseeded = 0u32;
        for c in small {
            let donor = (0..self.k)
                .max_by(|&a, &b| eff[a].total_cmp(&eff[b]).then_with(|| b.cmp(&a)))
                .unwrap();
            if eff[donor] < 2.0 {
                break;
            }
            let src: Vec<f32> = out[donor * dim..(donor + 1) * dim].to_vec();
            let rms = (src.iter().map(|v| v * v).sum::<f32>() / dim.max(1) as f32).sqrt();
            let amp = 1e-3 * if rms > 0.0 { rms } else { 1.0 };
            let o = &mut out[c * dim..(c + 1) * dim];
            for j in 0..dim {
                let u = rng.below(2001) as f32 / 1000.0 - 1.0;
                o[j] = src[j] + amp * u;
            }
            eff[donor] *= 0.5;
            eff[c] = eff[donor];
            reseeded += 1;
        }
        (out, reseeded)
    }
}

/// Flat assignment: index of the centroid with the highest cosine.
/// `centroids` are raw means, `inv` their inverse norms (see `math::inv_norms`).
#[inline]
pub fn assign_flat(x: &[f32], centroids: &[f32], inv: &[f32], dim: usize) -> usize {
    crate::math::argmax_dot_scaled(x, centroids, inv, dim).0
}

/// Coarse groups a two-level assignment searches for `probe_coarse` over `k1`
/// groups: `0` (or anything `>= k1`) means all of them, as in
/// [`crate::route::Router::route`].
#[inline]
pub fn effective_probe_coarse(probe_coarse: usize, k1: usize) -> usize {
    if probe_coarse == 0 || probe_coarse >= k1 {
        k1
    } else {
        probe_coarse
    }
}

/// Two-level assignment. Picks the `probe_coarse` best coarse groups
/// (`0` = all groups), then the best fine cell among their `k2` children.
/// Returns the global fine cell index (`coarse * k2 + j`).
pub fn assign_tree(
    x: &[f32],
    coarse: &[f32],
    coarse_inv: &[f32],
    fine: &[f32],
    fine_inv: &[f32],
    k2: usize,
    dim: usize,
    probe_coarse: usize,
) -> usize {
    assign_tree_top_m(x, coarse, coarse_inv, fine, fine_inv, k2, dim, probe_coarse, 1)[0].0
}

/// Top `m` fine cells for a vector under the two-level scheme, descending by
/// cosine. Used for multi-assignment at build time. `probe_coarse = 0` (or
/// `>= k1`) searches every coarse group; `m = 0` returns an empty list.
pub fn assign_tree_top_m(
    x: &[f32],
    coarse: &[f32],
    coarse_inv: &[f32],
    fine: &[f32],
    fine_inv: &[f32],
    k2: usize,
    dim: usize,
    probe_coarse: usize,
    m: usize,
) -> Vec<(usize, f32)> {
    if m == 0 {
        return Vec::new();
    }
    let probe_coarse = effective_probe_coarse(probe_coarse, coarse.len() / dim);
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(m + 1);
    for (g, _) in top_n_dot_scaled(x, coarse, coarse_inv, dim, probe_coarse) {
        let rows = &fine[g * k2 * dim..(g + 1) * k2 * dim];
        for (j, r) in rows.chunks_exact(dim).enumerate() {
            let cell = g * k2 + j;
            let d = dot(x, r) * fine_inv[cell];
            if top.len() < m || d > top[top.len() - 1].1 {
                let pos = top.partition_point(|&(_, s)| s >= d);
                top.insert(pos, (cell, d));
                if top.len() > m {
                    top.pop();
                }
            }
        }
    }
    top
}

/// Rows processed together by [`cell_neighbors`]: each centroid row `j` is
/// loaded once per block and dotted against every row of the block while it
/// is hot in L1, instead of streaming the whole table once per row.
const NEIGHBOR_BLOCK: usize = 16;

/// Inserts `(cell, score)` into `top` (sorted by score descending, then cell
/// ascending) if it ranks within the first `n`.
#[inline]
fn push_top(top: &mut Vec<(usize, f32)>, n: usize, cell: usize, d: f32) {
    if n == 0 {
        return;
    }
    if top.len() >= n {
        let (lc, ls) = top[top.len() - 1];
        if !(d > ls || (d == ls && cell < lc)) {
            return;
        }
    }
    let pos = top.partition_point(|&(c, s)| s > d || (s == d && c < cell));
    top.insert(pos, (cell, d));
    if top.len() > n {
        top.pop();
    }
}

/// Cell neighbour table: for every cell `i` in `[first, first + count)`, the
/// `n` cells with the highest cosine `dot(mean_i, mean_j) * inv_i * inv_j`,
/// descending (ties: lower cell id first). Each list starts with `i` itself;
/// the other `n - 1` entries never repeat `i`. Returned row-major,
/// `count * n` cell ids. `means` holds all `k = means.len() / dim` raw means.
///
/// Panics unless `first + count <= k` and `1 <= n <= k` (the wasm wrapper
/// validates). Cost: `count * k * dim` multiply-adds.
pub fn cell_neighbors(means: &[f32], dim: usize, first: usize, count: usize, n: usize) -> Vec<u32> {
    assert!(dim > 0 && means.len() % dim == 0, "means buffer is not a multiple of dim");
    let k = means.len() / dim;
    assert!(first + count <= k, "cell range out of bounds");
    assert!(n >= 1 && n <= k, "n must be in 1..=k");
    let inv = crate::math::inv_norms(means, dim);
    let mut out = Vec::with_capacity(count * n);
    let mut tops: Vec<Vec<(usize, f32)>> = Vec::with_capacity(NEIGHBOR_BLOCK);
    let mut b0 = first;
    while b0 < first + count {
        let b1 = (b0 + NEIGHBOR_BLOCK).min(first + count);
        tops.clear();
        tops.extend((b0..b1).map(|_| Vec::with_capacity(n)));
        for (j, rj) in means.chunks_exact(dim).enumerate() {
            let inv_j = inv[j];
            for (t, i) in tops.iter_mut().zip(b0..b1) {
                if i == j {
                    continue;
                }
                let d = dot(&means[i * dim..(i + 1) * dim], rj) * inv[i] * inv_j;
                push_top(t, n - 1, j, d);
            }
        }
        for (t, i) in tops.iter().zip(b0..b1) {
            out.push(i as u32);
            out.extend(t.iter().map(|&(c, _)| c as u32));
        }
        b0 = b1;
    }
    out
}

/// Checks a `k x n_neigh` neighbour table: right size, `n_neigh >= 1`, every id `< k`.
pub fn check_neighbors(neighbors: &[u32], k: usize, n_neigh: usize) -> Result<(), FormatError> {
    if n_neigh == 0 {
        return Err(FormatError::Inconsistent("n_neigh must be at least 1"));
    }
    if neighbors.len() != k * n_neigh {
        return Err(FormatError::Inconsistent("neighbour table size != k * n_neigh"));
    }
    if neighbors.iter().any(|&c| c as usize >= k) {
        return Err(FormatError::Inconsistent("neighbour table holds a cell id >= k"));
    }
    Ok(())
}

/// Best cell by cosine among the candidate cells `cands` (raw `means`, inverse
/// norms `inv`). Ties go to the lower cell id, so when `cands` covers every
/// cell this equals [`assign_flat`]. `cands` must be non-empty.
#[inline]
pub fn assign_candidates(x: &[f32], means: &[f32], inv: &[f32], dim: usize, cands: &[u32]) -> usize {
    let mut best = (usize::MAX, f32::NEG_INFINITY);
    for &c in cands {
        let c = c as usize;
        let d = dot(x, &means[c * dim..(c + 1) * dim]) * inv[c];
        if best.0 == usize::MAX || d > best.1 || (d == best.1 && c < best.0) {
            best = (c, d);
        }
    }
    best.0
}

/// Top `m` distinct cells among `cands`, descending by cosine (ties: lower
/// cell id first). With candidates covering the true top `m`, equals
/// [`assign_tree_top_m`] with `k1 = 1`.
pub fn assign_candidates_top_m(
    x: &[f32],
    means: &[f32],
    inv: &[f32],
    dim: usize,
    cands: &[u32],
    m: usize,
) -> Vec<(usize, f32)> {
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(m + 1);
    for &c in cands {
        let c = c as usize;
        if top.iter().any(|&(t, _)| t == c) {
            continue;
        }
        let d = dot(x, &means[c * dim..(c + 1) * dim]) * inv[c];
        push_top(&mut top, m, c, d);
    }
    top
}

/// Reservoir sampler that keeps up to `cap` vectors per bucket. Used to pick
/// initial fine centroids per coarse group from a streamed sample.
pub struct BucketReservoir {
    dim: usize,
    cap: usize,
    seen: Vec<u64>,
    data: Vec<Vec<f32>>,
    rng: crate::math::Rng,
}

impl BucketReservoir {
    pub fn new(buckets: usize, cap: usize, dim: usize, seed: u64) -> Self {
        BucketReservoir {
            dim,
            cap,
            seen: vec![0; buckets],
            data: (0..buckets).map(|_| Vec::with_capacity(cap * dim)).collect(),
            rng: crate::math::Rng::new(seed),
        }
    }

    pub fn offer(&mut self, bucket: usize, x: &[f32]) {
        let n = self.seen[bucket];
        self.seen[bucket] += 1;
        let have = self.data[bucket].len() / self.dim;
        if have < self.cap {
            self.data[bucket].extend_from_slice(x);
        } else {
            let r = self.rng.below(n + 1) as usize;
            if r < self.cap {
                self.data[bucket][r * self.dim..(r + 1) * self.dim].copy_from_slice(x);
            }
        }
    }

    /// Emits exactly `cap` unit centroids per bucket. Buckets with fewer than
    /// `cap` samples are padded with jittered copies; buckets with none are
    /// seeded from `fallback` (the coarse centroid) with jitter.
    pub fn finish(mut self, fallback: &[f32]) -> Vec<f32> {
        let dim = self.dim;
        let mut out = Vec::with_capacity(self.data.len() * self.cap * dim);
        for (b, bucket) in self.data.iter().enumerate() {
            let have = bucket.len() / dim;
            for i in 0..self.cap {
                let mut row: Vec<f32> = if have > 0 {
                    bucket[(i % have) * dim..(i % have + 1) * dim].to_vec()
                } else {
                    fallback[b * dim..(b + 1) * dim].to_vec()
                };
                if have == 0 || i >= have {
                    for v in row.iter_mut() {
                        let j = (self.rng.below(2001) as f32 - 1000.0) * 1e-5;
                        *v += j;
                    }
                }
                normalize(&mut row);
                out.extend_from_slice(&row);
            }
        }
        out
    }
}

/// Lloyd iterations used by [`split_cell`].
pub const SPLIT_ITERS: usize = 10;

/// Side (0 or 1) of `x` between two means (`means2` = `2 * dim`, `inv` their
/// inverse norms): the mean with the higher cosine, ties to 0. The assignment
/// rule of [`split_cell`]; `part::split_part_records` and
/// `slab::split_records` route entries with it.
#[inline]
pub fn split_side(x: &[f32], means2: &[f32], inv: &[f32; 2]) -> u8 {
    let dim = x.len();
    if dot(x, &means2[dim..2 * dim]) * inv[1] > dot(x, &means2[..dim]) * inv[0] {
        1
    } else {
        0
    }
}

/// Splits one cell's members in two: spherical 2-means over `vectors`
/// (`n * dim`, e.g. entries decoded from a slab's int8 records). Seeds with a
/// random member and a second member drawn with probability proportional to
/// its squared distance from the first (k-means++), then runs up to
/// [`SPLIT_ITERS`] Lloyd iterations (assign by cosine to the means; a mean is
/// the plain average of its members, as [`Accum::means`]), stopping early when
/// no assignment changes. Returns the two means (`2 * dim`) and each member's
/// side (0 or 1). With no members the means are empty; with one member (or
/// identical members) both means equal it.
pub fn split_cell(vectors: &[f32], dim: usize, seed: u64) -> (Vec<f32>, Vec<u8>) {
    assert!(dim > 0 && vectors.len() % dim == 0, "vectors length must be a multiple of dim");
    let n = vectors.len() / dim;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let row = |i: usize| &vectors[i * dim..(i + 1) * dim];
    let mut rng = crate::math::Rng::new(seed);
    let a = rng.below(n as u64) as usize;
    let d2: Vec<f64> = (0..n)
        .map(|i| row(i).iter().zip(row(a)).map(|(x, y)| ((x - y) as f64).powi(2)).sum())
        .collect();
    let total: f64 = d2.iter().sum();
    let b = if total > 0.0 {
        let mut t = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
        let mut pick = n - 1;
        for (i, d) in d2.iter().enumerate() {
            if t < *d {
                pick = i;
                break;
            }
            t -= d;
        }
        pick
    } else {
        a
    };
    let mut means = [row(a).to_vec(), row(b).to_vec()].concat();
    let mut assign = vec![u8::MAX; n];
    for _ in 0..SPLIT_ITERS {
        let inv = crate::math::inv_norms(&means, dim);
        let inv = [inv[0], inv[1]];
        let mut changed = false;
        let mut acc = Accum::new(2, dim);
        for i in 0..n {
            let x = row(i);
            let side = split_side(x, &means, &inv);
            changed |= assign[i] != side;
            assign[i] = side;
            acc.add(side as usize, x);
        }
        means = acc.means(&means);
        if !changed {
            break;
        }
    }
    (means, assign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accum_roundtrip_and_ranges() {
        let dim = 3;
        let mut a = Accum::new(4, dim);
        a.add(1, &[1.0, 2.0, 3.0]);
        a.add(1, &[1.0, 0.0, 1.0]);
        a.add(3, &[5.0, 5.0, 5.0]);
        let bytes = a.to_bytes();
        let b = Accum::from_bytes(&bytes).unwrap();
        assert_eq!(b.counts, vec![0, 2, 0, 1]);

        // Reduce only cells [1,3) from the serialized form.
        let (off, len) = Accum::record_range(dim, 1, 2);
        let mut r = Accum::new(2, dim);
        r.add_records(&bytes[off..off + len], 0).unwrap();
        assert_eq!(r.counts, vec![2, 0]);
        let prev = vec![9.0; 6];
        let m = r.means(&prev);
        assert_eq!(&m[0..3], &[1.0, 1.0, 2.0]);
        assert_eq!(&m[3..6], &[9.0, 9.0, 9.0]);
    }

    #[test]
    fn assign_tree_probe_coarse_zero_means_all_groups() {
        let (dim, k1, k2) = (8, 4, 5);
        let mut rng = crate::math::Rng::new(3);
        let mut rows = |n: usize| -> Vec<f32> { (0..n * dim).map(|_| rng.below(2001) as f32 / 1000.0 - 1.0).collect() };
        let coarse = rows(k1);
        let fine = rows(k1 * k2);
        let xs = rows(30);
        let (ci, fi) = (crate::math::inv_norms(&coarse, dim), crate::math::inv_norms(&fine, dim));
        for x in xs.chunks_exact(dim) {
            let all = assign_tree_top_m(x, &coarse, &ci, &fine, &fi, k2, dim, k1, 3);
            assert_eq!(all.len(), 3);
            // 0 and anything >= k1 search every group (no panic)
            assert_eq!(assign_tree_top_m(x, &coarse, &ci, &fine, &fi, k2, dim, 0, 3), all);
            assert_eq!(assign_tree_top_m(x, &coarse, &ci, &fine, &fi, k2, dim, k1 + 5, 3), all);
            assert_eq!(assign_tree(x, &coarse, &ci, &fine, &fi, k2, dim, 0), all[0].0);
            // the exhaustive top 3 equals flat ranking over all cells
            let flat = crate::math::top_n_dot_scaled(x, &fine, &fi, dim, 3);
            assert_eq!(all.iter().map(|p| p.0).collect::<Vec<_>>(), flat.iter().map(|p| p.0).collect::<Vec<_>>());
            // m = 0 is empty
            assert!(assign_tree_top_m(x, &coarse, &ci, &fine, &fi, k2, dim, 2, 0).is_empty());
        }
        assert_eq!(effective_probe_coarse(0, 4), 4);
        assert_eq!(effective_probe_coarse(2, 4), 2);
        assert_eq!(effective_probe_coarse(9, 4), 4);
    }

    #[test]
    fn means_reseed_recycles_small_cells() {
        let dim = 4;
        let mut a = Accum::new(5, dim);
        for _ in 0..100 {
            a.add(0, &[1.0, 0.0, 0.0, 0.0]);
        }
        for _ in 0..40 {
            a.add(1, &[0.0, 1.0, 0.0, 0.0]);
        }
        a.add(2, &[0.0, 0.0, 1.0, 0.0]); // tiny
        // cell 3 empty, cell 4 has 30
        for _ in 0..30 {
            a.add(4, &[0.0, 0.0, 0.0, 1.0]);
        }
        let prev = vec![0.5f32; 5 * dim];
        let plain = a.means(&prev);
        let (m, n) = a.means_reseed(&prev, 5, 7);
        assert_eq!(n, 2);
        // big cells unchanged
        assert_eq!(&m[0..dim], &plain[0..dim]);
        assert_eq!(&m[dim..2 * dim], &plain[dim..2 * dim]);
        assert_eq!(&m[4 * dim..], &plain[4 * dim..]);
        // the smallest (empty cell 3) is reseeded near the largest cell (0),
        // the next (cell 2, effective count of cell 0 now 50 > 40) near cell 0 too
        for c in [2usize, 3] {
            let row = &m[c * dim..(c + 1) * dim];
            assert!((row[0] - 1.0).abs() <= 1e-3 * 0.5 + 1e-6, "cell {c}: {row:?}");
            assert!(row[1..].iter().all(|v| v.abs() <= 1e-3 * 0.5 + 1e-6), "cell {c}: {row:?}");
            assert_ne!(row, &plain[0..dim]);
        }
        assert_ne!(&m[2 * dim..3 * dim], &m[3 * dim..4 * dim]);
        // deterministic
        assert_eq!(a.means_reseed(&prev, 5, 7), (m.clone(), 2));
        // min_count 61: only cell 0 is dense; all four others are split off it
        // (later donors are earlier reseeds, i.e. halves of cell 0).
        let (m4, n4) = a.means_reseed(&prev, 61, 1);
        assert_eq!(n4, 4);
        for c in 1..5 {
            let row = &m4[c * dim..(c + 1) * dim];
            assert!((row[0] - 1.0).abs() < 1.1e-3 && row[1..].iter().all(|v| v.abs() < 1.1e-3), "{row:?}");
        }
        // a lone dense cell of count 1 cannot be split
        let mut one = Accum::new(2, dim);
        one.add(0, &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one.means_reseed(&vec![0.0; 2 * dim], 1, 1).1, 0);
        // nothing below min_count -> identical to means()
        assert_eq!(a.means_reseed(&prev, 0, 1), (plain, 0));
    }

    fn rand_rows(rng: &mut crate::math::Rng, n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|_| rng.below(20001) as f32 / 10000.0 - 1.0).collect()
    }

    fn unit_rows(rng: &mut crate::math::Rng, n: usize, dim: usize) -> Vec<f32> {
        crate::math::normalized_rows(&rand_rows(rng, n, dim), dim)
    }

    /// Brute-force f64 cosine ranking of every other cell for cell `i`.
    fn brute_neighbors(means: &[f32], dim: usize, i: usize) -> Vec<(usize, f64)> {
        let row = |c: usize| &means[c * dim..(c + 1) * dim];
        let nrm = |c: usize| row(c).iter().map(|v| *v as f64 * *v as f64).sum::<f64>().sqrt();
        let k = means.len() / dim;
        let mut all: Vec<(usize, f64)> = (0..k)
            .filter(|&j| j != i)
            .map(|j| {
                let d: f64 = row(i).iter().zip(row(j)).map(|(a, b)| *a as f64 * *b as f64).sum();
                (j, d / (nrm(i) * nrm(j)))
            })
            .collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all
    }

    #[test]
    fn cell_neighbors_match_brute_force() {
        let mut rng = crate::math::Rng::new(21);
        let (k, dim, n) = (53, 24, 9);
        // raw (non-unit) means with varied norms: cosine must not depend on them
        let mut means = rand_rows(&mut rng, k, dim);
        for (c, r) in means.chunks_exact_mut(dim).enumerate() {
            for v in r.iter_mut() {
                *v *= 0.5 + c as f32 * 0.1;
            }
        }
        let full = cell_neighbors(&means, dim, 0, k, n);
        assert_eq!(full.len(), k * n);
        for i in 0..k {
            let row = &full[i * n..(i + 1) * n];
            assert_eq!(row[0] as usize, i, "self first");
            assert!(row[1..].iter().all(|&c| c as usize != i));
            let want = brute_neighbors(&means, dim, i);
            let all_cos: std::collections::HashMap<usize, f64> = want.iter().copied().collect();
            for (r, &c) in row[1..].iter().enumerate() {
                // same id at each rank, or (near-tie) the same score
                let (wc, ws) = want[r];
                assert!(c as usize == wc || (all_cos[&(c as usize)] - ws).abs() < 1e-5, "cell {i} rank {r}: {c} vs {wc}");
            }
            // distinct
            let mut sorted = row.to_vec();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), n);
        }
        // sharded ranges concatenate to the full table (block boundaries included)
        let mut cat = Vec::new();
        for (first, count) in [(0usize, 7usize), (7, 17), (24, 16), (40, 0), (40, 13)] {
            cat.extend(cell_neighbors(&means, dim, first, count, n));
        }
        assert_eq!(cat, full);
        // n = 1 is just the identity; n = k lists every cell once
        assert_eq!(cell_neighbors(&means, dim, 3, 4, 1), vec![3, 4, 5, 6]);
        let every = cell_neighbors(&means, dim, 5, 1, k);
        let mut s = every.clone();
        s.sort();
        assert_eq!(s, (0..k as u32).collect::<Vec<_>>());
    }

    #[test]
    fn candidate_assignment_with_full_table_equals_flat() {
        let mut rng = crate::math::Rng::new(8);
        let (k, dim) = (40, 32);
        let means = rand_rows(&mut rng, k, dim);
        let inv = crate::math::inv_norms(&means, dim);
        let nb = cell_neighbors(&means, dim, 0, k, k);
        check_neighbors(&nb, k, k).unwrap();
        let xs = unit_rows(&mut rng, 300, dim);
        let (mut a_flat, mut a_inc) = (Accum::new(k, dim), Accum::new(k, dim));
        for x in xs.chunks_exact(dim) {
            let prev = rng.below(k as u64) as usize;
            let want = assign_flat(x, &means, &inv, dim);
            let got = assign_candidates(x, &means, &inv, dim, &nb[prev * k..(prev + 1) * k]);
            assert_eq!(got, want);
            a_flat.add(want, x);
            a_inc.add(got, x);
        }
        assert_eq!(a_flat.to_bytes(), a_inc.to_bytes());
        // ties go to the lower cell id, as in assign_flat
        let dup = [1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let dinv = crate::math::inv_norms(&dup, 2);
        assert_eq!(assign_candidates(&[1.0, 0.0], &dup, &dinv, 2, &[2, 1, 0]), 0);
        assert_eq!(assign_flat(&[1.0, 0.0], &dup, &dinv, 2), 0);
    }

    #[test]
    fn check_neighbors_rejects_bad_tables() {
        assert!(check_neighbors(&[0, 1, 1, 0], 2, 2).is_ok());
        assert!(check_neighbors(&[0, 1, 1], 2, 2).is_err());
        assert!(check_neighbors(&[0, 1, 1, 2], 2, 2).is_err());
        assert!(check_neighbors(&[], 0, 0).is_err());
    }

    #[test]
    fn candidate_top_m_equals_tree_top_m_when_covered() {
        let mut rng = crate::math::Rng::new(13);
        let (k, dim, m) = (64, 16, 8);
        let means = rand_rows(&mut rng, k, dim);
        let inv = crate::math::inv_norms(&means, dim);
        let coarse = vec![1.0f32; dim];
        let cinv = crate::math::inv_norms(&coarse, dim);
        let xs = unit_rows(&mut rng, 200, dim);
        for x in xs.chunks_exact(dim) {
            let want = assign_tree_top_m(x, &coarse, &cinv, &means, &inv, k, dim, 1, m);
            assert_eq!(want.len(), m);
            // candidates: the true top m plus some random cells (and a repeat), shuffled
            let mut cands: Vec<u32> = want.iter().map(|p| p.0 as u32).collect();
            for _ in 0..10 {
                cands.push(rng.below(k as u64) as u32);
            }
            cands.push(cands[0]);
            for i in (1..cands.len()).rev() {
                cands.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let got = assign_candidates_top_m(x, &means, &inv, dim, &cands, m);
            assert_eq!(got, want);
            // full table row also matches
            let all: Vec<u32> = (0..k as u32).rev().collect();
            assert_eq!(assign_candidates_top_m(x, &means, &inv, dim, &all, m), want);
        }
    }

    #[test]
    fn tree_assignment_finds_nearest() {
        let dim = 2;
        let coarse = vec![1.0, 0.0, 0.0, 1.0];
        let fine = vec![1.0, 0.0, 0.8, 0.6, 0.0, 1.0, 0.6, 0.8];
        let ones = vec![1.0f32; 4];
        let cell = assign_tree(&[0.7, 0.7], &coarse, &ones[..2], &fine, &ones, 2, dim, 2);
        // best cosine: (0.8,0.6)->0.98, (0.6,0.8)->0.98 tie; first found wins.
        assert!(cell == 1 || cell == 3);
        let top = assign_tree_top_m(&[0.95, 0.3], &coarse, &ones[..2], &fine, &ones, 2, dim, 2, 2);
        assert_eq!(top[0].0, 0);
        assert_eq!(top[1].0, 1);
    }

    #[test]
    fn split_cell_separates_two_clusters() {
        let dim = 16;
        let mut rng = crate::math::Rng::new(3);
        let centre = |rng: &mut crate::math::Rng| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.below(2001) as f32 / 1000.0 - 1.0).collect();
            normalize(&mut v);
            v
        };
        let (ca, cb) = (centre(&mut rng), centre(&mut rng));
        let mut xs = Vec::new();
        let mut truth = Vec::new();
        for i in 0..120 {
            let c = if i % 3 == 0 { &cb } else { &ca };
            let mut x: Vec<f32> = c.iter().map(|v| v + (rng.below(2001) as f32 / 1000.0 - 1.0) * 0.1).collect();
            normalize(&mut x);
            xs.extend(x);
            truth.push((i % 3 == 0) as u8);
        }
        for seed in [1u64, 2, 3, 99] {
            let (means, side) = split_cell(&xs, dim, seed);
            assert_eq!((means.len(), side.len()), (2 * dim, 120));
            // the split equals the true clusters up to label swap
            let same = side.iter().zip(&truth).filter(|(a, b)| a == b).count();
            assert!(same == 120 || same == 0, "seed {seed}: {same}");
            // each mean is the average of its side
            for s in 0..2u8 {
                let members: Vec<usize> = (0..120).filter(|&i| side[i] == s).collect();
                for d in 0..dim {
                    let avg = members.iter().map(|&i| xs[i * dim + d]).sum::<f32>() / members.len() as f32;
                    assert!((means[s as usize * dim + d] - avg).abs() < 1e-5);
                }
            }
        }
        assert_eq!(split_cell(&[], dim, 1), (vec![], vec![]));
        let (m1, s1) = split_cell(&xs[..dim], dim, 1);
        assert_eq!((&m1[..dim], &m1[dim..], s1.len()), (&xs[..dim], &xs[..dim], 1));
    }
}
