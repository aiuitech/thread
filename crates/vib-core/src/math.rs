//! Numeric kernels.
//!
//! Every hot kernel (`dot`, `dot_f32_i8`, `dot_f32_i4`) has a portable scalar
//! version (`*_scalar`) and, when compiled for `wasm32` with `simd128`
//! enabled, an explicit `core::arch::wasm32` version (`*_simd`). LLVM does not
//! autovectorize these reductions on wasm32 (it will not reassociate f32 adds),
//! so the explicit path is what makes the scan fast in a Worker.
//!
//! Precision: all accumulation is in f32. The scalar kernels keep 8 partial
//! sums, the simd kernels 16 (4 x f32x4); both reduce pairwise at the end. The
//! two paths therefore round differently and can disagree in the last few
//! ulps of the result (|error| is bounded by about `n * eps * sum|a_i b_i|`,
//! typically `sqrt(n) * eps * ...`; for n = 1536 unit vectors that is < 1e-5,
//! two orders of magnitude below the residual-quantization error). The integer
//! codes are converted to f32 exactly (|code| <= 127), so no integer overflow
//! is possible.

/// Dot product of two equal-length slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return dot_simd(a, b);
    }
    #[allow(unreachable_code)]
    dot_scalar(a, b)
}

/// `q · code` where `code` is an int8 vector (residual SQ8 scoring).
#[inline]
pub fn dot_f32_i8(q: &[f32], code: &[i8]) -> f32 {
    assert_eq!(q.len(), code.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return dot_f32_i8_simd(q, code);
    }
    #[allow(unreachable_code)]
    dot_f32_i8_scalar(q, code)
}

/// `q · unpack(code4)` where `code4` holds two signed 4-bit values per byte:
/// the low nibble is dimension `2i`, the high nibble dimension `2i + 1`, each a
/// two's-complement nibble in -8..7 (the encoder only emits -7..7).
/// Requires `q.len() == 2 * code4.len()`.
#[inline]
pub fn dot_f32_i4(q: &[f32], code4: &[u8]) -> f32 {
    assert_eq!(q.len(), code4.len() * 2);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return dot_f32_i4_simd(q, code4);
    }
    #[allow(unreachable_code)]
    dot_f32_i4_scalar(q, code4)
}

/// Sign-extends the low nibble of `b`.
#[inline(always)]
pub fn nib_lo(b: u8) -> i8 {
    ((b << 4) as i8) >> 4
}

/// Sign-extends the high nibble of `b`.
#[inline(always)]
pub fn nib_hi(b: u8) -> i8 {
    (b as i8) >> 4
}

#[inline(always)]
fn hsum8(acc: &[f32; 8]) -> f32 {
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

/// Portable f32 dot product (8 independent partial sums).
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let blocks = n / 8;
    let mut acc = [0.0f32; 8];
    for i in 0..blocks {
        let a8 = &a[i * 8..i * 8 + 8];
        let b8 = &b[i * 8..i * 8 + 8];
        for j in 0..8 {
            acc[j] += a8[j] * b8[j];
        }
    }
    let mut s = hsum8(&acc);
    for i in blocks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// Portable `q · code` for int8 codes.
pub fn dot_f32_i8_scalar(q: &[f32], code: &[i8]) -> f32 {
    assert_eq!(q.len(), code.len());
    let n = q.len();
    let blocks = n / 8;
    let mut acc = [0.0f32; 8];
    for i in 0..blocks {
        let q8 = &q[i * 8..i * 8 + 8];
        let c8 = &code[i * 8..i * 8 + 8];
        for j in 0..8 {
            acc[j] += q8[j] * (c8[j] as f32);
        }
    }
    let mut s = hsum8(&acc);
    for i in blocks * 8..n {
        s += q[i] * (code[i] as f32);
    }
    s
}

/// Portable `q · unpack(code4)` for packed signed nibbles.
pub fn dot_f32_i4_scalar(q: &[f32], code4: &[u8]) -> f32 {
    assert_eq!(q.len(), code4.len() * 2);
    let nb = code4.len();
    let blocks = nb / 4; // 4 bytes = 8 dims per block
    let mut acc = [0.0f32; 8];
    for i in 0..blocks {
        let c4 = &code4[i * 4..i * 4 + 4];
        let q8 = &q[i * 8..i * 8 + 8];
        for j in 0..4 {
            acc[2 * j] += q8[2 * j] * (nib_lo(c4[j]) as f32);
            acc[2 * j + 1] += q8[2 * j + 1] * (nib_hi(c4[j]) as f32);
        }
    }
    let mut s = hsum8(&acc);
    for i in blocks * 4..nb {
        s += q[2 * i] * (nib_lo(code4[i]) as f32) + q[2 * i + 1] * (nib_hi(code4[i]) as f32);
    }
    s
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd {
    use core::arch::wasm32::*;

    #[inline(always)]
    unsafe fn ld(p: *const f32) -> v128 {
        v128_load(p as *const v128)
    }

    #[inline(always)]
    fn hsum(v: v128) -> f32 {
        (f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<2>(v))
            + (f32x4_extract_lane::<1>(v) + f32x4_extract_lane::<3>(v))
    }

    /// Converts 16 i8 lanes into four f32x4 vectors (lanes 0-3, 4-7, 8-11, 12-15).
    #[inline(always)]
    fn i8x16_to_f32x4x4(v: v128) -> [v128; 4] {
        let lo = i16x8_extend_low_i8x16(v);
        let hi = i16x8_extend_high_i8x16(v);
        [
            f32x4_convert_i32x4(i32x4_extend_low_i16x8(lo)),
            f32x4_convert_i32x4(i32x4_extend_high_i16x8(lo)),
            f32x4_convert_i32x4(i32x4_extend_low_i16x8(hi)),
            f32x4_convert_i32x4(i32x4_extend_high_i16x8(hi)),
        ]
    }

    pub fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        let blocks = n / 16;
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [f32x4_splat(0.0); 4];
        for i in 0..blocks {
            let o = i * 16;
            // SAFETY: o + 16 <= n for both slices.
            unsafe {
                for j in 0..4 {
                    acc[j] = f32x4_add(acc[j], f32x4_mul(ld(pa.add(o + 4 * j)), ld(pb.add(o + 4 * j))));
                }
            }
        }
        let mut s = hsum(f32x4_add(f32x4_add(acc[0], acc[2]), f32x4_add(acc[1], acc[3])));
        for i in blocks * 16..n {
            s += a[i] * b[i];
        }
        s
    }

    pub fn dot_f32_i8_simd(q: &[f32], code: &[i8]) -> f32 {
        assert_eq!(q.len(), code.len());
        let n = q.len();
        let blocks = n / 16;
        let (pq, pc) = (q.as_ptr(), code.as_ptr());
        let mut acc = [f32x4_splat(0.0); 4];
        for i in 0..blocks {
            let o = i * 16;
            // SAFETY: o + 16 <= n for both slices; v128_load is unaligned.
            unsafe {
                let c = i8x16_to_f32x4x4(v128_load(pc.add(o) as *const v128));
                for j in 0..4 {
                    acc[j] = f32x4_add(acc[j], f32x4_mul(ld(pq.add(o + 4 * j)), c[j]));
                }
            }
        }
        let mut s = hsum(f32x4_add(f32x4_add(acc[0], acc[2]), f32x4_add(acc[1], acc[3])));
        for i in blocks * 16..n {
            s += q[i] * (code[i] as f32);
        }
        s
    }

    pub fn dot_f32_i4_simd(q: &[f32], code4: &[u8]) -> f32 {
        assert_eq!(q.len(), code4.len() * 2);
        let nb = code4.len();
        let blocks = nb / 16; // 16 bytes = 32 dims per block
        let (pq, pc) = (q.as_ptr(), code4.as_ptr());
        let mut acc = [f32x4_splat(0.0); 4];
        for i in 0..blocks {
            // SAFETY: 16 * (i + 1) <= nb and 32 * (i + 1) <= q.len().
            unsafe {
                let v = v128_load(pc.add(i * 16) as *const v128);
                // even dims: sign-extended low nibbles; odd dims: arithmetic shift of the byte.
                let ev = i8x16_shr(i8x16_shl(v, 4), 4);
                let od = i8x16_shr(v, 4);
                // interleave back into dimension order: dims 0..16 and 16..32.
                let d0 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(ev, od);
                let d1 = i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(ev, od);
                let qo = pq.add(i * 32);
                let c0 = i8x16_to_f32x4x4(d0);
                for j in 0..4 {
                    acc[j] = f32x4_add(acc[j], f32x4_mul(ld(qo.add(4 * j)), c0[j]));
                }
                let c1 = i8x16_to_f32x4x4(d1);
                for j in 0..4 {
                    acc[j] = f32x4_add(acc[j], f32x4_mul(ld(qo.add(16 + 4 * j)), c1[j]));
                }
            }
        }
        let mut s = hsum(f32x4_add(f32x4_add(acc[0], acc[2]), f32x4_add(acc[1], acc[3])));
        for i in blocks * 16..nb {
            s += q[2 * i] * (super::nib_lo(code4[i]) as f32) + q[2 * i + 1] * (super::nib_hi(code4[i]) as f32);
        }
        s
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use simd::{dot_f32_i4_simd, dot_f32_i8_simd, dot_simd};

#[inline]
pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Scales `a` to unit length. Leaves an all-zero vector untouched.
pub fn normalize(a: &mut [f32]) {
    let n = norm(a);
    if n > 0.0 {
        let inv = 1.0 / n;
        for x in a.iter_mut() {
            *x *= inv;
        }
    }
}

/// Returns a normalized copy of every `dim`-sized row in `rows`.
pub fn normalized_rows(rows: &[f32], dim: usize) -> Vec<f32> {
    let mut out = rows.to_vec();
    for r in out.chunks_exact_mut(dim) {
        normalize(r);
    }
    out
}

/// Index and value of the largest dot product between `x` and each `dim`-sized row.
pub fn argmax_dot(x: &[f32], rows: &[f32], dim: usize) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, r) in rows.chunks_exact(dim).enumerate() {
        let d = dot(x, r);
        if d > best.1 {
            best = (i, d);
        }
    }
    best
}

/// Top `n` rows by dot product, descending. `n` is expected to be small;
/// `n = 0` returns an empty list.
pub fn top_n_dot(x: &[f32], rows: &[f32], dim: usize, n: usize) -> Vec<(usize, f32)> {
    if n == 0 {
        return Vec::new();
    }
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(n + 1);
    for (i, r) in rows.chunks_exact(dim).enumerate() {
        let d = dot(x, r);
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

/// Tiny deterministic PRNG (xorshift64*) for sampling without pulling in `rand`.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_naive() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..37).map(|i| (37 - i) as f32).collect();
        let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - naive).abs() < 1e-3);
    }

    #[test]
    fn dot_i8_matches_naive() {
        let q: Vec<f32> = (0..19).map(|i| (i as f32 - 9.0) / 10.0).collect();
        let c: Vec<i8> = (0..19).map(|i| (i * 13 % 255) as i32 as i8).collect();
        let naive: f32 = q.iter().zip(&c).map(|(x, y)| x * (*y as f32)).sum();
        assert!((dot_f32_i8(&q, &c) - naive).abs() < 1e-3);
    }

    fn rand_f32(rng: &mut Rng, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.below(20001) as f32 / 10000.0 - 1.0).collect()
    }

    #[test]
    fn nibble_sign_extension() {
        for v in -8i8..=7 {
            let b = (v as u8) & 0xF;
            assert_eq!(nib_lo(b), v);
            assert_eq!(nib_hi(b << 4), v);
            assert_eq!(nib_lo(b | 0xA0), v);
        }
    }

    /// The kernels on random data of awkward lengths agree with an f64 reference.
    /// On wasm32+simd128 builds `dot*` dispatches to the simd path, so running
    /// this suite under a wasm test runner checks simd against the reference too;
    /// natively it covers the scalar path and the public dispatchers.
    #[test]
    fn kernels_match_f64_reference_on_random_data() {
        let mut rng = Rng::new(99);
        for &n in &[0usize, 1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 64, 100, 1536, 1538] {
            let a = rand_f32(&mut rng, n);
            let b = rand_f32(&mut rng, n);
            let c8: Vec<i8> = (0..n).map(|_| (rng.below(255) as i32 - 127) as i8).collect();
            let n4 = n / 2;
            let c4: Vec<u8> = (0..n4).map(|_| rng.below(256) as u8).collect();

            let r_dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let r_i8: f64 = a.iter().zip(&c8).map(|(x, y)| *x as f64 * *y as f64).sum();
            let r_i4: f64 = (0..n4)
                .map(|i| a[2 * i] as f64 * nib_lo(c4[i]) as f64 + a[2 * i + 1] as f64 * nib_hi(c4[i]) as f64)
                .sum();
            let tol = |mag: f64| 1e-5 * (1.0 + mag) * (n as f64).sqrt().max(1.0);
            let mag = |s: f64| s.abs() + n as f64;
            for (got, want, what) in [
                (dot_scalar(&a, &b), r_dot, "dot_scalar"),
                (dot(&a, &b), r_dot, "dot"),
                (dot_f32_i8_scalar(&a, &c8), r_i8, "dot_f32_i8_scalar"),
                (dot_f32_i8(&a, &c8), r_i8, "dot_f32_i8"),
            ] {
                assert!(((got as f64) - want).abs() < tol(mag(want)), "{what} n={n}: {got} vs {want}");
            }
            let a4 = &a[..n4 * 2];
            for (got, what) in [(dot_f32_i4_scalar(a4, &c4), "dot_f32_i4_scalar"), (dot_f32_i4(a4, &c4), "dot_f32_i4")] {
                assert!(((got as f64) - r_i4).abs() < tol(mag(r_i4)), "{what} n={n}: {got} vs {r_i4}");
            }
        }
    }

    /// Scalar emulation of the simd i4 kernel's lane arrangement (deinterleave
    /// by shifts, re-interleave by shuffle) to pin down the shuffle masks.
    #[test]
    fn i4_simd_lane_layout_emulation() {
        let mut rng = Rng::new(5);
        let bytes: Vec<u8> = (0..16).map(|_| rng.below(256) as u8).collect();
        let ev: Vec<i8> = bytes.iter().map(|b| ((((*b as i8) as i32) << 4) as i8) >> 4).collect();
        let od: Vec<i8> = bytes.iter().map(|b| (*b as i8) >> 4).collect();
        let cat: Vec<i8> = ev.iter().chain(od.iter()).copied().collect();
        let m0 = [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23];
        let m1 = [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31];
        let dims: Vec<i8> = m0.iter().chain(m1.iter()).map(|&i| cat[i]).collect();
        let want: Vec<i8> = bytes.iter().flat_map(|b| [nib_lo(*b), nib_hi(*b)]).collect();
        assert_eq!(dims, want);
    }

    #[test]
    fn top_n_is_sorted() {
        let rows = vec![0.1, 0.9, 0.5, 0.7];
        let x = vec![1.0];
        let t = top_n_dot(&x, &rows, 1, 3);
        assert_eq!(t.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 3, 2]);
    }

    #[test]
    fn top_n_zero_is_empty() {
        let rows = vec![0.1, 0.9, 0.5, 0.7];
        let x = vec![1.0];
        assert!(top_n_dot(&x, &rows, 1, 0).is_empty());
        let inv = inv_norms(&rows, 1);
        assert!(top_n_dot_scaled(&x, &rows, &inv, 1, 0).is_empty());
        // n larger than the row count returns every row
        assert_eq!(top_n_dot_scaled(&x, &rows, &inv, 1, 9).len(), 4);
    }
}

/// `1 / ||row||` for every `dim`-sized row (0 for a zero row). Lets callers
/// route by cosine against raw means without materializing unit copies.
pub fn inv_norms(rows: &[f32], dim: usize) -> Vec<f32> {
    rows.chunks_exact(dim)
        .map(|r| {
            let n = norm(r);
            if n > 0.0 { 1.0 / n } else { 0.0 }
        })
        .collect()
}

/// Like `argmax_dot` but scales each dot by `inv[i]` (cosine against means).
pub fn argmax_dot_scaled(x: &[f32], rows: &[f32], inv: &[f32], dim: usize) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, r) in rows.chunks_exact(dim).enumerate() {
        let d = dot(x, r) * inv[i];
        if d > best.1 {
            best = (i, d);
        }
    }
    best
}

/// Like `top_n_dot` but scales each dot by `inv[i]`. `n = 0` returns an empty list.
pub fn top_n_dot_scaled(x: &[f32], rows: &[f32], inv: &[f32], dim: usize, n: usize) -> Vec<(usize, f32)> {
    if n == 0 {
        return Vec::new();
    }
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(n + 1);
    for (i, r) in rows.chunks_exact(dim).enumerate() {
        let d = dot(x, r) * inv[i];
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
