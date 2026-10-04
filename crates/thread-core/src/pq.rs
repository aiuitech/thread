// Dense-side file written before rustfmt was adopted; kept as is (`cargo fmt` skips it).
#![cfg_attr(rustfmt, rustfmt::skip)]
//! Product quantization of residuals (`x - cell mean`) for the v3 scan tier.
//!
//! The residual space of dimension `dim` is split into `m` subspaces of
//! `sub = dim / m` consecutive dimensions; each subspace has its own table of
//! 256 sub-centroids, shared by every cell. A residual is coded as `m` bytes
//! (the nearest sub-centroid per subspace). Scoring a query uses asymmetric
//! distance: per query, [`QueryTable`] holds `q_sub · centroid` for every
//! subspace and sub-centroid (`m x 256` f32), and the score of an entry of cell
//! `c` is `q·c + sum_j table[j][code_j]` (`m` table lookups).
//!
//! Training is streaming Lloyd k-means per subspace in map-reduce form, like
//! `kmeans::Accum`: a map job adds residuals into a [`PqAccum`] (nearest
//! sub-centroid under the previous codebook, per-sub-centroid sums and
//! counts), partials are serialized and merged, and [`PqAccum::means`] gives
//! the next codebook. [`Codebook::init_from_samples`] seeds the first one.
//!
//! Serialized codebook (little-endian): `[magic "VIBQ"][dim u32][m u32]` then
//! `m * 256 * sub` f32, sub-centroid `c` of subspace `j` at index
//! `(j * 256 + c) * sub`.
//!
//! Serialized accumulator (little-endian, 32-byte header):
//! `[magic "VIBT"][dim u32][m u32][reserved u32][rows u64][sq_err f64]`, then
//! `m * 256` u32 counts, then `m * 256 * sub` f32 sums (same order as the
//! codebook). `rows` is the number of residuals added and `sq_err` the sum of
//! their squared quantization errors under the codebook they were assigned with.

use crate::error::{need, rd_u32, FormatError};
use crate::math::Rng;

/// Sub-centroids per subspace (codes are one byte).
pub const PQ_KS: usize = 256;
pub const CODEBOOK_MAGIC: &[u8; 4] = b"VIBQ";
pub const CODEBOOK_HEADER: usize = 12;
pub const PQ_ACCUM_MAGIC: &[u8; 4] = b"VIBT";
pub const PQ_ACCUM_HEADER: usize = 32;

fn check_shape(dim: usize, m: usize) -> Result<usize, FormatError> {
    if m == 0 || dim == 0 || dim % m != 0 {
        return Err(FormatError::Inconsistent("pq: need m > 0 and dim % m == 0"));
    }
    if m > u16::MAX as usize {
        return Err(FormatError::Inconsistent("pq: m must fit in u16"));
    }
    Ok(dim / m)
}

/// A PQ codebook: `m` subspaces x 256 sub-centroids of `sub = dim / m` dims.
#[derive(Clone, Debug)]
pub struct Codebook {
    pub dim: usize,
    pub m: usize,
    pub sub: usize,
    /// `m x 256 x sub`: sub-centroid `c` of subspace `j` is
    /// `centroids[(j * 256 + c) * sub..][..sub]`.
    pub centroids: Vec<f32>,
    /// The same values laid out `m x sub x 256` (one row of 256 per subspace
    /// dimension), so encoding and table building run elementwise over the
    /// 256 sub-centroids (vectorizes without reassociating any sum).
    ct: Vec<f32>,
}

impl PartialEq for Codebook {
    fn eq(&self, o: &Self) -> bool {
        self.dim == o.dim && self.m == o.m && self.centroids == o.centroids
    }
}

impl Codebook {
    pub fn new(dim: usize, m: usize, centroids: Vec<f32>) -> Result<Self, FormatError> {
        let sub = check_shape(dim, m)?;
        if centroids.len() != m * PQ_KS * sub {
            return Err(FormatError::Inconsistent("pq: centroid buffer != m * 256 * sub"));
        }
        let mut ct = vec![0f32; centroids.len()];
        for j in 0..m {
            for c in 0..PQ_KS {
                for d in 0..sub {
                    ct[(j * sub + d) * PQ_KS + c] = centroids[(j * PQ_KS + c) * sub + d];
                }
            }
        }
        Ok(Codebook { dim, m, sub, centroids, ct })
    }

    /// Sub-centroid `c` of subspace `j`.
    #[inline]
    pub fn centroid(&self, j: usize, c: usize) -> &[f32] {
        let o = (j * PQ_KS + c) * self.sub;
        &self.centroids[o..o + self.sub]
    }

    pub fn byte_len(&self) -> usize {
        CODEBOOK_HEADER + self.centroids.len() * 4
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_len());
        out.extend_from_slice(CODEBOOK_MAGIC);
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.m as u32).to_le_bytes());
        crate::error::f32s_to_le(&self.centroids, &mut out);
        out
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, FormatError> {
        need(buf, CODEBOOK_HEADER)?;
        if &buf[0..4] != CODEBOOK_MAGIC {
            return Err(FormatError::BadMagic);
        }
        let dim = rd_u32(buf, 4) as usize;
        let m = rd_u32(buf, 8) as usize;
        let sub = check_shape(dim, m)?;
        let len = CODEBOOK_HEADER + m * PQ_KS * sub * 4;
        need(buf, len)?;
        if buf.len() != len {
            return Err(FormatError::Inconsistent("pq: codebook has trailing bytes"));
        }
        Codebook::new(dim, m, crate::error::f32s_from_le(&buf[CODEBOOK_HEADER..]))
    }

    /// Initial codebook: for every subspace, the subvectors of 256 random rows
    /// of `residuals` (`n x dim`; distinct rows per subspace when `n >= 256`,
    /// otherwise drawn with replacement). Pass a few thousand residuals.
    pub fn init_from_samples(residuals: &[f32], dim: usize, m: usize, seed: u64) -> Result<Self, FormatError> {
        let sub = check_shape(dim, m)?;
        if residuals.is_empty() || residuals.len() % dim != 0 {
            return Err(FormatError::Inconsistent("pq: residuals must be a non-empty multiple of dim"));
        }
        let n = residuals.len() / dim;
        let mut rng = Rng::new(seed);
        let mut idx: Vec<u32> = (0..n as u32).collect();
        let mut centroids = vec![0f32; m * PQ_KS * sub];
        for j in 0..m {
            for c in 0..PQ_KS {
                let row = if n >= PQ_KS {
                    // partial Fisher-Yates: position c gets a row not yet used for this subspace
                    let r = c + rng.below((n - c) as u64) as usize;
                    idx.swap(c, r);
                    idx[c] as usize
                } else {
                    rng.below(n as u64) as usize
                };
                let src = &residuals[row * dim + j * sub..row * dim + (j + 1) * sub];
                centroids[(j * PQ_KS + c) * sub..(j * PQ_KS + c + 1) * sub].copy_from_slice(src);
            }
        }
        Codebook::new(dim, m, centroids)
    }

    /// Squared distances from subvector `x` (`sub` values) of subspace `j` to
    /// all 256 sub-centroids of that subspace.
    #[inline]
    pub fn sub_dists(&self, j: usize, x: &[f32], out: &mut [f32; PQ_KS]) {
        let sub = self.sub;
        debug_assert_eq!(x.len(), sub);
        let ct = &self.ct[j * sub * PQ_KS..(j + 1) * sub * PQ_KS];
        out.fill(0.0);
        for (d, &xd) in x.iter().enumerate() {
            let row: &[f32; PQ_KS] = ct[d * PQ_KS..(d + 1) * PQ_KS].try_into().unwrap();
            for c in 0..PQ_KS {
                let t = xd - row[c];
                out[c] += t * t;
            }
        }
    }

    /// Nearest sub-centroid of subspace `j` to subvector `x`: `(index, squared distance)`.
    #[inline]
    pub fn nearest(&self, j: usize, x: &[f32], scratch: &mut [f32; PQ_KS]) -> (usize, f32) {
        self.sub_dists(j, x, scratch);
        argmin256(scratch)
    }
}

#[inline]
fn argmin256(d: &[f32; PQ_KS]) -> (usize, f32) {
    // min value first (independent lanes), then its first index
    let mut mins = [f32::INFINITY; 8];
    for ch in d.chunks_exact(8) {
        for l in 0..8 {
            mins[l] = if ch[l] < mins[l] { ch[l] } else { mins[l] };
        }
    }
    let best = mins.iter().copied().fold(f32::INFINITY, |a, b| if b < a { b } else { a });
    match d.iter().position(|&v| v == best) {
        Some(i) => (i, best),
        None => (0, d[0]), // all NaN
    }
}

/// Encodes `residual` (`dim` values) as `m` codes: the nearest sub-centroid
/// (squared L2) per subspace. Returns the squared quantization error.
pub fn encode(residual: &[f32], cb: &Codebook, out: &mut [u8]) -> f32 {
    assert_eq!(residual.len(), cb.dim);
    assert_eq!(out.len(), cb.m);
    let mut scratch = [0f32; PQ_KS];
    let mut err = 0.0f32;
    for (j, (o, x)) in out.iter_mut().zip(residual.chunks_exact(cb.sub)).enumerate() {
        let (c, d) = cb.nearest(j, x, &mut scratch);
        *o = c as u8;
        err += d;
    }
    err
}

/// Reconstructs the residual of `code` into `out` (`dim` values).
pub fn decode(code: &[u8], cb: &Codebook, out: &mut [f32]) {
    assert_eq!(code.len(), cb.m);
    assert_eq!(out.len(), cb.dim);
    for (j, (o, &c)) in out.chunks_exact_mut(cb.sub).zip(code).enumerate() {
        o.copy_from_slice(cb.centroid(j, c as usize));
    }
}

/// `vectors[i] - means[assign[i]]` for every row (`n x dim` in, `n x dim` out).
pub fn residuals(vectors: &[f32], means: &[f32], assign: &[u32], dim: usize) -> Result<Vec<f32>, FormatError> {
    if dim == 0 || vectors.len() % dim != 0 || means.len() % dim != 0 {
        return Err(FormatError::Inconsistent("vectors/means length is not a multiple of dim"));
    }
    let n = vectors.len() / dim;
    let k = means.len() / dim;
    if assign.len() != n {
        return Err(FormatError::Inconsistent("assign length != number of vectors"));
    }
    if assign.iter().any(|&a| a as usize >= k) {
        return Err(FormatError::Inconsistent("assign cell out of range"));
    }
    let mut out = vec![0f32; vectors.len()];
    for ((o, x), &a) in out.chunks_exact_mut(dim).zip(vectors.chunks_exact(dim)).zip(assign) {
        let c = &means[a as usize * dim..(a as usize + 1) * dim];
        for i in 0..dim {
            o[i] = x[i] - c[i];
        }
    }
    Ok(out)
}

/// Asymmetric-distance lookup table of one query: `table[j][c] = q_j · centroid(j, c)`.
#[derive(Clone, Debug)]
pub struct QueryTable {
    pub m: usize,
    pub table: Vec<[f32; PQ_KS]>,
}

impl QueryTable {
    pub fn new(q: &[f32], cb: &Codebook) -> Result<Self, FormatError> {
        if q.len() != cb.dim {
            return Err(FormatError::Inconsistent("query dim != codebook dim"));
        }
        let sub = cb.sub;
        let mut table = vec![[0f32; PQ_KS]; cb.m];
        for (j, (t, qs)) in table.iter_mut().zip(q.chunks_exact(sub)).enumerate() {
            let ct = &cb.ct[j * sub * PQ_KS..(j + 1) * sub * PQ_KS];
            for (d, &qd) in qs.iter().enumerate() {
                let row: &[f32; PQ_KS] = ct[d * PQ_KS..(d + 1) * PQ_KS].try_into().unwrap();
                for c in 0..PQ_KS {
                    t[c] += qd * row[c];
                }
            }
        }
        Ok(QueryTable { m: cb.m, table })
    }

    /// `sum_j table[j][code_j]` (the query's dot with the decoded residual).
    #[inline]
    pub fn score(&self, code: &[u8]) -> f32 {
        adc(&self.table, code)
    }
}

/// `sum_j rows[j][code[j]]` with 8 independent partial sums (the lookups are
/// scalar gathers; the partial sums hide the add latency). `code.len() == rows.len()`.
#[inline]
pub fn adc(rows: &[[f32; PQ_KS]], code: &[u8]) -> f32 {
    assert_eq!(rows.len(), code.len());
    let mut a = [0f32; 8];
    let mut cc = code.chunks_exact(8);
    let mut rc = rows.chunks_exact(8);
    for (c, r) in (&mut cc).zip(&mut rc) {
        // indexing a [f32; 256] with a u8 needs no bounds check
        a[0] += r[0][c[0] as usize];
        a[1] += r[1][c[1] as usize];
        a[2] += r[2][c[2] as usize];
        a[3] += r[3][c[3] as usize];
        a[4] += r[4][c[4] as usize];
        a[5] += r[5][c[5] as usize];
        a[6] += r[6][c[6] as usize];
        a[7] += r[7][c[7] as usize];
    }
    for (c, r) in cc.remainder().iter().zip(rc.remainder()) {
        a[0] += r[*c as usize];
    }
    ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))
}

/// Streaming PQ training accumulator (one Lloyd step, map side).
#[derive(Clone, Debug, PartialEq)]
pub struct PqAccum {
    pub dim: usize,
    pub m: usize,
    pub sub: usize,
    /// `m x 256`
    pub counts: Vec<u32>,
    /// `m x 256 x sub`
    pub sums: Vec<f32>,
    /// Residuals added.
    pub rows: u64,
    /// Sum of squared quantization errors of the added residuals.
    pub sq_err: f64,
}

impl PqAccum {
    pub fn new(dim: usize, m: usize) -> Result<Self, FormatError> {
        let sub = check_shape(dim, m)?;
        Ok(PqAccum { dim, m, sub, counts: vec![0; m * PQ_KS], sums: vec![0.0; m * PQ_KS * sub], rows: 0, sq_err: 0.0 })
    }

    /// Assigns every subvector of `residual` to its nearest sub-centroid of
    /// `cb` and accumulates it. Returns the squared quantization error.
    pub fn add(&mut self, residual: &[f32], cb: &Codebook) -> f32 {
        assert_eq!(residual.len(), self.dim);
        assert!(cb.dim == self.dim && cb.m == self.m, "codebook shape != accumulator shape");
        let sub = self.sub;
        let mut scratch = [0f32; PQ_KS];
        let mut err = 0.0f32;
        for (j, x) in residual.chunks_exact(sub).enumerate() {
            let (c, d) = cb.nearest(j, x, &mut scratch);
            err += d;
            let k = j * PQ_KS + c;
            self.counts[k] += 1;
            let s = &mut self.sums[k * sub..(k + 1) * sub];
            for i in 0..sub {
                s[i] += x[i];
            }
        }
        self.rows += 1;
        self.sq_err += err as f64;
        err
    }

    /// Mean squared quantization error per residual added so far (0 if none).
    pub fn mean_sq_err(&self) -> f64 {
        if self.rows == 0 {
            0.0
        } else {
            self.sq_err / self.rows as f64
        }
    }

    pub fn merge(&mut self, o: &PqAccum) -> Result<(), FormatError> {
        if o.dim != self.dim || o.m != self.m {
            return Err(FormatError::Inconsistent("pq accumulator shape mismatch"));
        }
        for (a, b) in self.counts.iter_mut().zip(&o.counts) {
            *a += b;
        }
        for (a, b) in self.sums.iter_mut().zip(&o.sums) {
            *a += b;
        }
        self.rows += o.rows;
        self.sq_err += o.sq_err;
        Ok(())
    }

    pub fn byte_len(&self) -> usize {
        PQ_ACCUM_HEADER + self.counts.len() * 4 + self.sums.len() * 4
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_len());
        out.extend_from_slice(PQ_ACCUM_MAGIC);
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.m as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&self.rows.to_le_bytes());
        out.extend_from_slice(&self.sq_err.to_le_bytes());
        for c in &self.counts {
            out.extend_from_slice(&c.to_le_bytes());
        }
        crate::error::f32s_to_le(&self.sums, &mut out);
        out
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, FormatError> {
        need(buf, PQ_ACCUM_HEADER)?;
        if &buf[0..4] != PQ_ACCUM_MAGIC {
            return Err(FormatError::BadMagic);
        }
        let dim = rd_u32(buf, 4) as usize;
        let m = rd_u32(buf, 8) as usize;
        let mut a = PqAccum::new(dim, m)?;
        if buf.len() != a.byte_len() {
            return Err(FormatError::TooShort { need: a.byte_len(), have: buf.len() });
        }
        a.rows = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        a.sq_err = f64::from_le_bytes(buf[24..32].try_into().unwrap());
        let mut o = PQ_ACCUM_HEADER;
        for c in a.counts.iter_mut() {
            *c = rd_u32(buf, o);
            o += 4;
        }
        a.sums = crate::error::f32s_from_le(&buf[o..]);
        Ok(a)
    }

    /// Serialized size for a shape (to split concatenated partials).
    pub fn serialized_len(dim: usize, m: usize) -> Result<usize, FormatError> {
        let sub = check_shape(dim, m)?;
        Ok(PQ_ACCUM_HEADER + m * PQ_KS * 4 + m * PQ_KS * sub * 4)
    }

    /// New codebook: each sub-centroid becomes the mean of its members; empty
    /// sub-centroids keep their value from `prev`.
    pub fn means(&self, prev: &Codebook) -> Result<Codebook, FormatError> {
        if prev.dim != self.dim || prev.m != self.m {
            return Err(FormatError::Inconsistent("previous codebook shape != accumulator shape"));
        }
        let sub = self.sub;
        let mut c = prev.centroids.clone();
        for k in 0..self.m * PQ_KS {
            let n = self.counts[k];
            if n > 0 {
                let inv = 1.0 / n as f32;
                for i in 0..sub {
                    c[k * sub + i] = self.sums[k * sub + i] * inv;
                }
            }
        }
        Codebook::new(self.dim, self.m, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss(rng: &mut Rng) -> f32 {
        // sum of 4 uniforms, roughly normal with unit variance
        let s: f32 = (0..4).map(|_| rng.below(1_000_001) as f32 / 1_000_000.0).sum();
        (s - 2.0) * 1.732
    }

    /// Residuals drawn around `n_clusters` random centers per subspace (so a PQ
    /// codebook can fit them well), with per-dim noise `noise`.
    fn clustered(rng: &mut Rng, n: usize, dim: usize, m: usize, n_clusters: usize, noise: f32) -> Vec<f32> {
        let sub = dim / m;
        let centers: Vec<f32> = (0..m * n_clusters * sub).map(|_| gauss(rng) * 0.05).collect();
        let mut out = Vec::with_capacity(n * dim);
        for _ in 0..n {
            for j in 0..m {
                let c = rng.below(n_clusters as u64) as usize;
                for d in 0..sub {
                    out.push(centers[(j * n_clusters + c) * sub + d] + gauss(rng) * noise);
                }
            }
        }
        out
    }

    #[test]
    fn codebook_roundtrip_and_validation() {
        let mut rng = Rng::new(3);
        let (dim, m) = (24, 6);
        let res = clustered(&mut rng, 300, dim, m, 10, 0.01);
        let cb = Codebook::init_from_samples(&res, dim, m, 7).unwrap();
        assert_eq!((cb.dim, cb.m, cb.sub, cb.centroids.len()), (24, 6, 4, 6 * 256 * 4));
        let bytes = cb.to_bytes();
        assert_eq!(bytes.len(), cb.byte_len());
        assert_eq!(&bytes[..4], b"VIBQ");
        let back = Codebook::from_bytes(&bytes).unwrap();
        assert_eq!(back, cb);
        assert_eq!(back.ct, cb.ct);
        // init picks distinct rows per subspace when n >= 256: every centroid is a real subvector
        for j in 0..m {
            for c in 0..PQ_KS {
                let s = cb.centroid(j, c);
                assert!(res.chunks_exact(dim).any(|r| &r[j * 4..j * 4 + 4] == s));
            }
        }
        assert!(Codebook::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert_eq!(Codebook::from_bytes(&bad), Err(FormatError::BadMagic));
        assert!(Codebook::new(25, 6, vec![0.0; 6 * 256 * 4]).is_err());
        assert!(Codebook::new(24, 6, vec![0.0; 10]).is_err());
        assert!(Codebook::init_from_samples(&res[..dim - 1], dim, m, 1).is_err());
        // fewer rows than 256 still works (with replacement)
        let small = Codebook::init_from_samples(&res[..10 * dim], dim, m, 1).unwrap();
        assert_eq!(small.centroids.len(), cb.centroids.len());
    }

    #[test]
    fn encode_matches_brute_force_and_decode_error_is_bounded() {
        let mut rng = Rng::new(11);
        let (dim, m) = (32, 8);
        let sub = dim / m;
        let base = clustered(&mut rng, 600, dim, m, 50, 0.0);
        let cb = Codebook::init_from_samples(&base, dim, m, 5).unwrap();
        let mut code = vec![0u8; m];
        let mut back = vec![0f32; dim];
        for t in 0..200 {
            // a residual = codebook entries + small noise: decoding is no farther
            // than the noise in every subspace (the generating entry is a candidate)
            let gen: Vec<u8> = (0..m).map(|_| rng.below(256) as u8).collect();
            let mut r = vec![0f32; dim];
            decode(&gen, &cb, &mut r);
            let noise: Vec<f32> = (0..dim).map(|_| gauss(&mut rng) * 0.002 * (t % 3) as f32).collect();
            for i in 0..dim {
                r[i] += noise[i];
            }
            let err = encode(&r, &cb, &mut code);
            decode(&code, &cb, &mut back);
            let mut tot = 0.0f32;
            for j in 0..m {
                let e: f32 = (0..sub).map(|d| (r[j * sub + d] - back[j * sub + d]).powi(2)).sum();
                let nz: f32 = (0..sub).map(|d| noise[j * sub + d].powi(2)).sum();
                assert!(e <= nz * 1.0001 + 1e-9, "subspace {j}: err {e} > noise {nz}");
                // brute force nearest
                let bf = (0..PQ_KS)
                    .map(|c| cb.centroid(j, c).iter().zip(&r[j * sub..(j + 1) * sub]).map(|(a, b)| (a - b).powi(2)).sum::<f32>())
                    .fold(f32::INFINITY, f32::min);
                assert!((e - bf).abs() <= 1e-6 * (1.0 + bf), "subspace {j}: {e} vs brute force {bf}");
                tot += e;
            }
            assert!((err - tot).abs() <= 1e-5 * (1.0 + tot));
        }
    }

    #[test]
    fn training_reduces_quantization_error() {
        let mut rng = Rng::new(21);
        let (dim, m) = (32, 8);
        // 300 true clusters per subspace > 256 codes, plus noise: k-means has work to do
        let train = clustered(&mut rng, 4000, dim, m, 300, 0.01);
        let mut cb = Codebook::init_from_samples(&train[..2000 * dim], dim, m, 9).unwrap();
        let mut errs = Vec::new();
        for _ in 0..6 {
            // map-reduce: two partial accumulators over halves, serialized and merged
            let mut a = PqAccum::new(dim, m).unwrap();
            let mut b = PqAccum::new(dim, m).unwrap();
            for (i, r) in train.chunks_exact(dim).enumerate() {
                if i % 2 == 0 { a.add(r, &cb) } else { b.add(r, &cb) };
            }
            let mut red = PqAccum::from_bytes(&a.to_bytes()).unwrap();
            assert_eq!(red, a);
            red.merge(&PqAccum::from_bytes(&b.to_bytes()).unwrap()).unwrap();
            assert_eq!(red.rows, 4000);
            assert_eq!(red.counts.iter().map(|&c| c as u64).sum::<u64>(), 4000 * m as u64);
            errs.push(red.mean_sq_err());
            cb = red.means(&cb).unwrap();
        }
        for w in errs.windows(2) {
            assert!(w[1] <= w[0] * 1.0001, "error went up: {errs:?}");
        }
        assert!(errs[5] < errs[0] * 0.8, "training barely helped: {errs:?}");
        // empty sub-centroids keep their previous value
        let empty = PqAccum::new(dim, m).unwrap();
        assert_eq!(empty.means(&cb).unwrap(), cb);
        assert_eq!(PqAccum::serialized_len(dim, m).unwrap(), empty.to_bytes().len());
        assert!(PqAccum::from_bytes(&empty.to_bytes()[..100]).is_err());
    }

    #[test]
    fn query_table_score_equals_dot_with_decoded() {
        let mut rng = Rng::new(4);
        let (dim, m) = (48, 12);
        let res = clustered(&mut rng, 400, dim, m, 40, 0.01);
        let cb = Codebook::init_from_samples(&res, dim, m, 2).unwrap();
        let q: Vec<f32> = (0..dim).map(|_| gauss(&mut rng)).collect();
        let t = QueryTable::new(&q, &cb).unwrap();
        let mut code = vec![0u8; m];
        let mut back = vec![0f32; dim];
        for r in res.chunks_exact(dim).take(50) {
            encode(r, &cb, &mut code);
            decode(&code, &cb, &mut back);
            let want: f32 = q.iter().zip(&back).map(|(a, b)| a * b).sum();
            assert!((t.score(&code) - want).abs() < 1e-5, "{} vs {want}", t.score(&code));
        }
        // m not a multiple of 8 exercises the remainder path
        assert_eq!(m % 8, 4);
        assert!(QueryTable::new(&q[1..], &cb).is_err());
    }

    #[test]
    fn residuals_subtract_assigned_mean() {
        let means = vec![1.0, 2.0, 10.0, 20.0];
        let xs = vec![1.5, 2.5, 11.0, 19.0, 0.0, 0.0];
        let r = residuals(&xs, &means, &[0, 1, 1], 2).unwrap();
        assert_eq!(r, vec![0.5, 0.5, 1.0, -1.0, -10.0, -20.0]);
        assert!(residuals(&xs, &means, &[0, 2, 1], 2).is_err());
        assert!(residuals(&xs, &means, &[0, 1], 2).is_err());
    }
}
