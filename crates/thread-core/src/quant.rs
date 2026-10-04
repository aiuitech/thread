// Dense-side file written before rustfmt was adopted; kept as is (`cargo fmt` skips it).
#![cfg_attr(rustfmt, rustfmt::skip)]
//! Residual scalar quantization with a per-vector scale.
//!
//! A vector `x` stored in the slab of cell `c` is written as `code` such that
//! `x ≈ c + scale * code`. Scoring a unit query `q` is then
//! `q·x ≈ q·c + scale * (q·code)`, where `q·c` is computed once per slab.
//!
//! Two code widths are used:
//! - SQ4 (scan tier): signed 4-bit codes in -7..7, two per byte (low nibble =
//!   even dimension), with an MSE-optimal clipped scale.
//! - SQ8 (rerank tier): int8 codes in -127..127 with a max-abs scale.

use crate::math::{dot_f32_i4, dot_f32_i8, nib_hi, nib_lo};

pub const SQ8_MAX: f32 = 127.0;
pub const SQ4_MAX: f32 = 7.0;

/// Clip fractions of `max|r|` tried by the SQ4 encoder: 0.30, 0.35, ..., 1.00.
pub const SQ4_CLIP_FRACTIONS: usize = 15;

/// Encodes `x - c` into `out` and returns the scale. `out.len() == x.len()`.
pub fn encode_residual_sq8(x: &[f32], c: &[f32], out: &mut [i8]) -> f32 {
    assert_eq!(x.len(), c.len());
    assert_eq!(x.len(), out.len());
    let mut maxabs = 0.0f32;
    for i in 0..x.len() {
        maxabs = maxabs.max((x[i] - c[i]).abs());
    }
    if maxabs == 0.0 {
        out.fill(0);
        return 0.0;
    }
    let scale = maxabs / SQ8_MAX;
    let inv = 1.0 / scale;
    for i in 0..x.len() {
        let v = ((x[i] - c[i]) * inv).round().clamp(-SQ8_MAX, SQ8_MAX);
        out[i] = v as i8;
    }
    scale
}

/// Reconstructs `c + scale * code` into `out`.
pub fn decode_residual_sq8(code: &[i8], scale: f32, c: &[f32], out: &mut [f32]) {
    for i in 0..code.len() {
        out[i] = c[i] + scale * (code[i] as f32);
    }
}

/// Approximate `q·x` given `qc = q·c`.
#[inline]
pub fn score_sq8(q: &[f32], code: &[i8], scale: f32, qc: f32) -> f32 {
    qc + scale * dot_f32_i8(q, code)
}

#[inline(always)]
fn q4(v: f32, inv: f32) -> f32 {
    // round-half-even is a single instruction on wasm (f32.nearest); the tie
    // rule is irrelevant for quantization quality.
    (v * inv).round_ties_even().clamp(-SQ4_MAX, SQ4_MAX)
}

/// Encodes `x - c` as packed signed 4-bit codes and returns the scale.
///
/// The scale is chosen among `SQ4_CLIP_FRACTIONS` clip points `f * max|r| / 7`
/// (f = 0.30, 0.35, ..., 1.00) to minimise `||r - scale * code||²`; clipping
/// the few large components buys finer resolution for the many small ones.
///
/// `out_nibbles.len() == x.len() / 2` (dim must be even); `scratch.len() >= x.len()`.
/// Nibble `2i` is the low half of byte `i`, values stored as `(v & 0xF)`.
pub fn encode_residual_sq4(x: &[f32], c: &[f32], out_nibbles: &mut [u8], scratch: &mut [f32]) -> f32 {
    let dim = x.len();
    assert_eq!(dim, c.len());
    assert!(dim % 2 == 0, "sq4 requires an even dimension");
    assert_eq!(out_nibbles.len(), dim / 2);
    assert!(scratch.len() >= dim);
    let r = &mut scratch[..dim];
    let mut maxabs = 0.0f32;
    for i in 0..dim {
        r[i] = x[i] - c[i];
        maxabs = maxabs.max(r[i].abs());
    }
    if maxabs == 0.0 || !maxabs.is_finite() {
        out_nibbles.fill(0);
        return 0.0;
    }
    let mut best = (f32::INFINITY, maxabs / SQ4_MAX);
    for k in 0..SQ4_CLIP_FRACTIONS {
        let f = 0.30 + 0.05 * k as f32;
        let scale = f * maxabs / SQ4_MAX;
        let inv = 1.0 / scale;
        let mut err = 0.0f32;
        for &v in r.iter() {
            let d = v - scale * q4(v, inv);
            err += d * d;
        }
        if err < best.0 {
            best = (err, scale);
        }
    }
    let scale = best.1;
    let inv = 1.0 / scale;
    for (o, pair) in out_nibbles.iter_mut().zip(r.chunks_exact(2)) {
        let lo = q4(pair[0], inv) as i8 as u8 & 0xF;
        let hi = q4(pair[1], inv) as i8 as u8 & 0xF;
        *o = lo | (hi << 4);
    }
    scale
}

/// Reconstructs `c + scale * unpack(code4)` into `out` (`out.len() == 2 * code4.len()`).
pub fn decode_residual_sq4(code4: &[u8], scale: f32, c: &[f32], out: &mut [f32]) {
    assert_eq!(out.len(), code4.len() * 2);
    for (i, b) in code4.iter().enumerate() {
        out[2 * i] = c[2 * i] + scale * (nib_lo(*b) as f32);
        out[2 * i + 1] = c[2 * i + 1] + scale * (nib_hi(*b) as f32);
    }
}

/// Approximate `q·x` from a 4-bit code given `qc = q·c`.
#[inline]
pub fn score_sq4(q: &[f32], code4: &[u8], scale: f32, qc: f32) -> f32 {
    qc + scale * dot_f32_i4(q, code4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{dot, normalize, Rng};

    fn setup(rng: &mut Rng, dim: usize, spread: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut c: Vec<f32> = (0..dim).map(|_| (rng.below(1000) as f32 / 1000.0) - 0.5).collect();
        normalize(&mut c);
        let mut x: Vec<f32> = c.iter().map(|v| v + (rng.below(1000) as f32 / 1000.0 - 0.5) * spread).collect();
        normalize(&mut x);
        let mut q: Vec<f32> = (0..dim).map(|_| (rng.below(1000) as f32 / 1000.0) - 0.5).collect();
        normalize(&mut q);
        (c, x, q)
    }

    #[test]
    fn residual_roundtrip_is_close() {
        let dim = 64;
        let mut rng = Rng::new(42);
        let (c, x, q) = setup(&mut rng, dim, 0.1);

        let mut code = vec![0i8; dim];
        let scale = encode_residual_sq8(&x, &c, &mut code);
        let exact = dot(&q, &x);
        let approx = score_sq8(&q, &code, scale, dot(&q, &c));
        assert!((exact - approx).abs() < 2e-3, "exact {exact} approx {approx}");

        let mut back = vec![0f32; dim];
        decode_residual_sq8(&code, scale, &c, &mut back);
        let err: f32 = x.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 1e-3);
    }

    #[test]
    fn sq4_roundtrip_error_bound() {
        let mut rng = Rng::new(17);
        for &dim in &[2usize, 64, 1536] {
            for _ in 0..20 {
                let (c, x, q) = setup(&mut rng, dim, 0.2);
                let mut code4 = vec![0u8; dim / 2];
                let mut scratch = vec![0f32; dim];
                let scale = encode_residual_sq4(&x, &c, &mut code4, &mut scratch);
                let maxabs = x.iter().zip(&c).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                assert!(scale > 0.0 && scale <= maxabs / 7.0 * 1.0001);
                assert!(scale >= 0.30 * maxabs / 7.0 * 0.9999);
                let mut back = vec![0f32; dim];
                decode_residual_sq4(&code4, scale, &c, &mut back);
                // every in-range component is within half a step; clipped ones within the clip gap.
                let clip = 7.0 * scale;
                let mut sq = 0.0f32;
                for i in 0..dim {
                    let r = x[i] - c[i];
                    let e = (x[i] - back[i]).abs();
                    let bound = if r.abs() <= clip { 0.5 * scale } else { r.abs() - clip };
                    assert!(e <= bound * 1.001 + 1e-7, "dim {dim} i {i}: err {e} bound {bound}");
                    sq += e * e;
                }
                // the MSE-chosen scale is never worse than the plain max-abs scale.
                let s_max = maxabs / 7.0;
                let sq_max: f32 = (0..dim)
                    .map(|i| {
                        let r = x[i] - c[i];
                        let d = r - s_max * (r / s_max).round().clamp(-7.0, 7.0);
                        d * d
                    })
                    .sum();
                assert!(sq <= sq_max * 1.0001 + 1e-12, "clipped {sq} > maxabs {sq_max}");
                // scoring matches the decoded vector.
                let approx = score_sq4(&q, &code4, scale, dot(&q, &c));
                assert!((approx - dot(&q, &back)).abs() < 1e-4);
                if dim == 1536 {
                    assert!((approx - dot(&q, &x)).abs() < 2e-2);
                }
            }
        }
    }

    #[test]
    fn sq4_nibbles_are_signed_and_in_range() {
        let x = [1.0f32, -1.0, 0.5, -0.5, 0.0, 0.25];
        let c = [0.0f32; 6];
        let mut code4 = [0u8; 3];
        let mut scratch = [0f32; 6];
        let s = encode_residual_sq4(&x, &c, &mut code4, &mut scratch);
        let vals: Vec<i8> = code4.iter().flat_map(|b| [nib_lo(*b), nib_hi(*b)]).collect();
        assert!(vals.iter().all(|v| (-7..=7).contains(v)));
        assert!(vals[0] > 0 && vals[1] < 0 && vals[0] == -vals[1]);
        assert_eq!(vals[4], 0);
        assert!(s > 0.0);
        // zero residual
        let s0 = encode_residual_sq4(&c, &c, &mut code4, &mut scratch);
        assert_eq!(s0, 0.0);
        assert_eq!(code4, [0, 0, 0]);
    }
}
