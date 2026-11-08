// Dense-side file written before rustfmt was adopted; kept as is (`cargo fmt` skips it).
#![cfg_attr(rustfmt, rustfmt::skip)]
//! Scanning a slab for a query, and reranking int8 records.

use crate::error::{rd_f32, FormatError};
use crate::filter::AttrFilter;
use crate::math::{dot, dot_f32_i4, dot_f32_i8};
use crate::pq::{adc, QueryTable};
use crate::quant::score_sq8;
use crate::slab::{rerank_record_len, SlabView};
use crate::tomb::Tombstones;
use crate::topk::TopK;

/// What a scan did: entries scored, entries left out of the top-k because their
/// id is in the `skip` set or their attrs failed the filter, and heap insertions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScanStats {
    pub scanned: usize,
    /// Tombstoned entries that scored above the heap threshold when they were
    /// reached (entries that would have entered the top-k). Entries scoring
    /// below it are never looked up, so this is not the number of tombstoned
    /// entries in the input.
    pub skipped: usize,
    /// Entries that scored above the heap threshold but failed the attribute
    /// filter (same caveat as `skipped`: only entries that would have entered).
    pub filtered: usize,
    /// Entries pushed into the top-k heap (they beat its threshold, matched the
    /// filter and were not tombstoned). `pushed / scanned` is how selective the
    /// running threshold was: the heap work a scan did on top of scoring.
    pub pushed: usize,
}

/// Offers `(id, s, idx)` to `topk`, applying the attribute `filter` and the
/// tombstone `skip` set before insertion. Both run only for a score that would
/// enter the heap; the entry's `attrs` word is read only when there is a filter.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn offer(topk: &mut TopK, skip: Option<&Tombstones>, filter: Option<&AttrFilter>, st: &mut ScanStats, id: u32, attrs: impl FnOnce() -> u32, s: f32, idx: u32) {
    if s > topk.threshold() {
        if let Some(f) = filter {
            if !f.matches(attrs()) {
                st.filtered += 1;
                return;
            }
        }
        if let Some(t) = skip {
            if t.contains(id) {
                st.skipped += 1;
                return;
            }
        }
        st.pushed += 1;
        topk.push(id, s, idx);
    }
}

/// Scores every entry in `slab` against unit query `q` and pushes
/// `(id, score, entry index)` into `topk`, leaving out ids in `skip` and
/// entries whose `attrs` word fails `filter` (neither takes a heap slot, so
/// live, matching entries ranked below them are kept).
///
/// - v1 slabs: int8 residual score `q·c + scale * (q·code8)`.
/// - v2 slabs: 4-bit scan tier `q·c + scale4 * (q·unpack(code4))`; works on a
///   `[0, scan_len)` prefix. Rerank the winners with [`rerank_records`].
///
/// Panics on a v3 slab (its scan tier needs the PQ table: use [`scan_slab_pq`]).
pub fn scan_slab(slab: &SlabView<'_>, q: &[f32], topk: &mut TopK, skip: Option<&Tombstones>, filter: Option<&AttrFilter>) -> ScanStats {
    assert_eq!(q.len(), slab.dim);
    assert!(!slab.is_v3(), "v3 slabs are scanned with scan_slab_pq");
    let centroid = slab.centroid();
    let qc = dot(q, &centroid);
    let mut st = ScanStats { scanned: slab.count, ..ScanStats::default() };
    if slab.is_v2() {
        let nb = slab.dim / 2;
        let codes = slab.codes4_section();
        for i in 0..slab.count {
            let s = qc + slab.scale4(i) * dot_f32_i4(q, &codes[i * nb..(i + 1) * nb]);
            offer(topk, skip, filter, &mut st, slab.id(i), || slab.attrs(i), s, i as u32);
        }
    } else {
        for i in 0..slab.count {
            let s = score_sq8(q, slab.code8(i), slab.scale8(i), qc);
            offer(topk, skip, filter, &mut st, slab.id(i), || slab.attrs(i), s, i as u32);
        }
    }
    st
}

/// Scans a v3 slab (or its `[0, scan_len)` prefix) with asymmetric PQ
/// distances: `q·c + sum_j table[j][code_j]`, pushing `(id, score, entry
/// index)` into `topk` and leaving out ids in `skip` and entries failing
/// `filter`. `table` is [`QueryTable::new`] of the same unit query `q` and the
/// index's codebook. Rerank the winners with [`rerank_records`] (v3 rerank
/// records are v2's).
pub fn scan_slab_pq(slab: &SlabView<'_>, table: &QueryTable, q: &[f32], topk: &mut TopK, skip: Option<&Tombstones>, filter: Option<&AttrFilter>) -> Result<ScanStats, FormatError> {
    if !slab.is_v3() {
        return Err(FormatError::Inconsistent("scan_pq needs a v3 slab (use scan for v1/v2)"));
    }
    if q.len() != slab.dim {
        return Err(FormatError::Inconsistent("query dim != slab dim"));
    }
    if slab.count == 0 {
        return Ok(ScanStats::default());
    }
    let m = slab.pq_m();
    if m != table.m {
        return Err(FormatError::Inconsistent("slab pq_m != PQ table m"));
    }
    let qc = dot(q, &slab.centroid());
    let codes = slab.pq_section();
    let rows = &table.table[..];
    let mut st = ScanStats { scanned: slab.count, ..ScanStats::default() };
    for (i, code) in codes.chunks_exact(m).enumerate() {
        let s = qc + adc(rows, code);
        offer(topk, skip, filter, &mut st, slab.id(i), || slab.attrs(i), s, i as u32);
    }
    Ok(st)
}

/// Scans a delta segment for unit query `q` with its int8 records:
/// `qc[cell] + scale8 * (q·code8)`, where `qc[c] = q·mean[c]` for every cell
/// (unnormalized means, e.g. `Router::cell_dots`). Pushes `(id, score, entry
/// index)` into `topk`, leaving out ids in `skip` and entries failing `filter`.
/// Scores are on the same scale as reranked slab scores. Errors if an entry's
/// cell is not covered by `qc` or the dims differ.
pub fn scan_delta(delta: &crate::delta::DeltaView<'_>, q: &[f32], qc: &[f32], topk: &mut TopK, skip: Option<&Tombstones>, filter: Option<&AttrFilter>) -> Result<ScanStats, FormatError> {
    if q.len() != delta.dim {
        return Err(FormatError::Inconsistent("query dim != delta dim"));
    }
    let mut st = ScanStats { scanned: delta.count, ..ScanStats::default() };
    for i in 0..delta.count {
        let cell = delta.cell(i) as usize;
        let Some(&qcc) = qc.get(cell) else {
            return Err(FormatError::Inconsistent("delta entry cell >= qc length"));
        };
        let s = score_sq8(q, delta.code8(i), delta.scale8(i), qcc);
        offer(topk, skip, filter, &mut st, delta.id(i), || delta.attrs(i), s, i as u32);
    }
    Ok(st)
}

/// Scores `n` consecutive rerank records (`[scale8 f32][code8 i8 * dim]`, as
/// stored in a v2 or v3 slab's rerank section) against `q` and that slab's centroid:
/// `q·c + scale8 * (q·code8)`. `dim = q.len()`.
pub fn rerank_records(q: &[f32], centroid: &[f32], records: &[u8], n: usize) -> Result<Vec<f32>, FormatError> {
    let dim = q.len();
    if centroid.len() != dim {
        return Err(FormatError::Inconsistent("centroid dim != query dim"));
    }
    let rl = rerank_record_len(dim);
    let need_len = n.checked_mul(rl).ok_or(FormatError::Inconsistent("record count overflow"))?;
    if records.len() < need_len {
        return Err(FormatError::TooShort { need: need_len, have: records.len() });
    }
    let qc = dot(q, centroid);
    Ok(records[..need_len]
        .chunks_exact(rl)
        .map(|r| {
            let code = &r[4..];
            // i8 and u8 have identical layout and alignment.
            let code = unsafe { core::slice::from_raw_parts(code.as_ptr() as *const i8, dim) };
            qc + rd_f32(r, 0) * dot_f32_i8(q, code)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{normalize, Rng};
    use crate::pq::Codebook;
    use crate::quant::{encode_residual_sq4, encode_residual_sq8, score_sq8};
    use crate::slab::{SlabBuilder, SlabBuilderV1, SlabHeader};

    fn cluster(dim: usize, n: usize, seed: u64) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut rng = Rng::new(seed);
        let rand_unit = |rng: &mut Rng| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.below(2000) as f32 / 1000.0 - 1.0).collect();
            normalize(&mut v);
            v
        };
        let c = rand_unit(&mut rng);
        let mut xs = Vec::new();
        for _ in 0..n {
            let mut x: Vec<f32> = c.iter().map(|v| v + (rng.below(2000) as f32 / 1000.0 - 1.0) * 0.2).collect();
            normalize(&mut x);
            xs.push(x);
        }
        (c, xs)
    }

    fn build_v2(c: &[f32], xs: &[Vec<f32>]) -> Vec<u8> {
        let dim = c.len();
        let mut b = SlabBuilder::new(dim, 0, c);
        let (mut c4, mut c8, mut scratch) = (vec![0u8; dim / 2], vec![0i8; dim], vec![0f32; dim]);
        for (i, x) in xs.iter().enumerate() {
            let s4 = encode_residual_sq4(x, c, &mut c4, &mut scratch);
            let s8 = encode_residual_sq8(x, c, &mut c8);
            b.push(i as u32, &c4, s4, &c8, s8, &[], 0);
        }
        b.finish()
    }

    fn exact_top(xs: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
        let mut exact: Vec<(usize, f32)> = xs.iter().enumerate().map(|(i, x)| (i, dot(q, x))).collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1));
        exact.iter().take(k).map(|p| p.0).collect()
    }

    #[test]
    fn scan_recovers_exact_neighbor_order() {
        let dim = 32;
        let (c, xs) = cluster(dim, 50, 7);
        let mut b = SlabBuilderV1::new(dim, 0, &c);
        let mut code = vec![0i8; dim];
        for (i, x) in xs.iter().enumerate() {
            let s = encode_residual_sq8(x, &c, &mut code);
            b.push(i as u32, &code, s, &[], 0);
        }
        let bytes = b.finish();
        let view = SlabView::parse(&bytes).unwrap();

        let q = xs[17].clone();
        let exact_ids = exact_top(&xs, &q, 5);
        let mut top = TopK::new(5);
        assert_eq!(scan_slab(&view, &q, &mut top, None, None).scanned, 50);
        let got = top.into_sorted();
        assert_eq!(got[0].id, 17);
        assert!(got.iter().all(|h| h.idx == h.id));
        let got_ids: Vec<usize> = got.iter().map(|h| h.id as usize).collect();
        let overlap = got_ids.iter().filter(|i| exact_ids.contains(i)).count();
        assert!(overlap >= 4, "exact {exact_ids:?} got {got_ids:?}");
    }

    #[test]
    fn scan_v2_recovers_exact_neighbor_order() {
        let dim = 32;
        let (c, xs) = cluster(dim, 50, 7);
        let bytes = build_v2(&c, &xs);
        let h = SlabHeader::parse(&bytes).unwrap();
        // scan only the prefix, as the host does
        let view = SlabView::parse(&bytes[..h.scan_len]).unwrap();
        let full = SlabView::parse(&bytes).unwrap();

        for qi in [3usize, 17, 40] {
            let q = xs[qi].clone();
            let exact_ids = exact_top(&xs, &q, 5);
            // 4-bit tier: top-T candidates
            let mut top = TopK::new(15);
            assert_eq!(scan_slab(&view, &q, &mut top, None, None).scanned, 50);
            let cands = top.into_sorted();
            assert_eq!(cands[0].id as usize, qi);
            // rerank candidates with int8 records located via idx
            let mut recs = Vec::new();
            for h in &cands {
                recs.extend_from_slice(full.rerank_record(h.idx as usize));
            }
            let scores = rerank_records(&q, &view.centroid(), &recs, cands.len()).unwrap();
            let mut rr: Vec<(u32, f32)> = cands.iter().zip(&scores).map(|(h, s)| (h.id, *s)).collect();
            rr.sort_by(|a, b| b.1.total_cmp(&a.1));
            let got_ids: Vec<usize> = rr.iter().take(5).map(|p| p.0 as usize).collect();
            let overlap = got_ids.iter().filter(|i| exact_ids.contains(i)).count();
            assert!(overlap >= 4, "exact {exact_ids:?} got {got_ids:?}");
        }
    }

    #[test]
    fn slab_writer_v1_scan_equals_v2_rerank() {
        use crate::part::{parse_directory, PartBuilder};
        use crate::slab::SlabWriter;
        let dim = 64;
        let (c, xs) = cluster(dim, 40, 23);
        // a part with this cell (j = 1) between two other cells, as an assign job writes it
        let mut p = PartBuilder::new(3, dim);
        let (mut c4, mut c8, mut scratch) = (vec![0u8; dim / 2], vec![0i8; dim], vec![0f32; dim]);
        p.push(0, 999, 1.0, 1.0, 0, &vec![0x11; dim / 2], &vec![1; dim], &[], b"x");
        for (i, x) in xs.iter().enumerate() {
            let s4 = encode_residual_sq4(x, &c, &mut c4, &mut scratch);
            let s8 = encode_residual_sq8(x, &c, &mut c8);
            let payload = vec![i as u8; i % 4];
            p.push(1, 100 + i as u32, s4, s8, i as u32 * 3, &c4, &c8, &[], &payload);
        }
        p.push(2, 998, 1.0, 1.0, 0, &vec![0x22; dim / 2], &vec![2; dim], &[], b"");
        let part = p.take();
        let (_, _, dir) = parse_directory(&part).unwrap();
        let recs = &part[dir[1].off as usize..(dir[1].off + dir[1].len) as usize];

        let mut w1 = SlabWriter::new(1, dim, 5, &c).unwrap();
        let mut w2 = SlabWriter::new(2, dim, 5, &c).unwrap();
        // feed v1 in two chunks (records are self-delimiting, chunks split at record bounds)
        let (a, b) = {
            let mut it = crate::part::records(recs, dim);
            let mut cut = 0;
            for _ in 0..17 {
                let r = it.next().unwrap().unwrap();
                cut += crate::part::record_len(dim) + r.payload.len();
            }
            (&recs[..cut], &recs[cut..])
        };
        assert_eq!(w1.add_part_records(a, dim).unwrap() + w1.add_part_records(b, dim).unwrap(), 40);
        assert_eq!(w2.add_part_records(recs, dim).unwrap(), 40);
        assert_eq!((w1.version(), w2.version()), (1, 2));
        let bl1 = w1.byte_len();
        let (s1, s2) = (w1.finish(), w2.finish());
        assert_eq!(s1.len(), bl1);

        let v1 = SlabView::parse(&s1).unwrap();
        let v2 = SlabView::parse(&s2).unwrap();
        assert_eq!((v1.version(), v1.count, v1.cell, v1.dim), (1, 40, 5, dim));
        assert_eq!(v1.centroid(), c);
        assert!(s1.len() < s2.len());
        for i in 0..40 {
            assert_eq!(v1.id(i), v2.id(i));
            assert_eq!(v1.entry(i).attrs, i as u32 * 3);
            assert_eq!(v1.payload(i), v2.payload(i));
            assert_eq!(v1.code8(i), v2.code8(i));
            assert_eq!(v1.scale8(i), v2.scale8(i));
        }
        for qi in [0usize, 13, 39] {
            let q = xs[qi].clone();
            let mut top = TopK::new(40);
            assert_eq!(scan_slab(&v1, &q, &mut top, None, None).scanned, 40);
            let hits = top.into_sorted();
            let h = v2.header;
            let all = &s2[h.rerank_off..h.rerank_off + 40 * h.rerank_record_len];
            let rr = rerank_records(&q, &v2.centroid(), all, 40).unwrap();
            assert_eq!(hits.len(), 40);
            for hit in &hits {
                assert_eq!(hit.id, v2.id(hit.idx as usize));
                assert_eq!(hit.score, rr[hit.idx as usize], "q {qi} entry {}", hit.idx);
            }
            assert_eq!(hits[0].id as usize, 100 + qi);
        }
        // records without pq codes cannot build a v3 slab
        let mut w3 = SlabWriter::new(3, dim, 5, &c).unwrap();
        assert!(w3.add_part_records(recs, dim).is_err());
        assert_eq!(w3.len(), 0);
        // unknown formats and odd dims for v2 are rejected
        assert!(SlabWriter::new(4, dim, 0, &c).is_err());
        assert!(SlabWriter::new(2, 3, 0, &[0.0; 3]).is_err());
        assert!(SlabWriter::new(1, dim, 0, &c[..3]).is_err());
    }

    /// Clustered vectors around one cell mean whose residuals have per-subspace
    /// structure, a PQ codebook trained on them, and a v3 slab built through a
    /// part (as assign + finalize do).
    fn pq_fixture(dim: usize, m: usize, n: usize, seed: u64) -> (Vec<f32>, Vec<Vec<f32>>, Codebook, Vec<u8>) {
        use crate::part::{parse_directory, PartBuilder};
        use crate::pq::{encode, PqAccum};
        use crate::slab::SlabWriter;
        let (c, xs) = cluster(dim, n, seed);
        let res: Vec<f32> = xs.iter().flat_map(|x| x.iter().zip(&c).map(|(a, b)| a - b)).collect();
        let mut cb = Codebook::init_from_samples(&res, dim, m, seed).unwrap();
        for _ in 0..8 {
            let mut acc = PqAccum::new(dim, m).unwrap();
            for r in res.chunks_exact(dim) {
                acc.add(r, &cb);
            }
            cb = acc.means(&cb).unwrap();
        }
        let mut p = PartBuilder::new(1, dim);
        let (mut c4, mut c8, mut scratch, mut pq) = (vec![0u8; dim / 2], vec![0i8; dim], vec![0f32; dim], vec![0u8; m]);
        for (i, x) in xs.iter().enumerate() {
            let s4 = encode_residual_sq4(x, &c, &mut c4, &mut scratch);
            let s8 = encode_residual_sq8(x, &c, &mut c8);
            encode(&res[i * dim..(i + 1) * dim], &cb, &mut pq);
            p.push(0, i as u32, s4, s8, 0, &c4, &c8, &pq, &[]);
        }
        let part = p.take();
        let (_, _, dir) = parse_directory(&part).unwrap();
        let mut w = SlabWriter::new(3, dim, 9, &c).unwrap();
        let recs = &part[dir[0].off as usize..(dir[0].off + dir[0].len) as usize];
        assert_eq!(w.add_part_records(recs, dim).unwrap() as usize, n);
        assert_eq!(w.version(), 3);
        let bl = w.byte_len();
        let slab = w.finish();
        assert_eq!(slab.len(), bl);
        (c, xs, cb, slab)
    }

    #[test]
    fn scan_pq_then_rerank_matches_int8_ranking() {
        let (dim, m, n, k, t) = (64, 16, 1000, 10, 50);
        let (c, xs, cb, slab) = pq_fixture(dim, m, n, 41);
        let h = SlabHeader::parse(&slab).unwrap();
        let prefix = SlabView::parse(&slab[..h.scan_len]).unwrap();
        let full = SlabView::parse(&slab).unwrap();
        assert_eq!((prefix.pq_m(), prefix.count), (m, n));
        let all_rr = &slab[h.rerank_off..h.rerank_off + n * h.rerank_record_len];
        let mut rng = Rng::new(77);
        for qi in 0..10 {
            // queries near the cluster but not equal to any member
            let mut q: Vec<f32> = xs[qi * 37].iter().map(|v| v + (rng.below(2000) as f32 / 1000.0 - 1.0) * 0.05).collect();
            normalize(&mut q);
            let table = QueryTable::new(&q, &cb).unwrap();
            let mut top = TopK::new(t);
            assert_eq!(scan_slab_pq(&prefix, &table, &q, &mut top, None, None).unwrap().scanned, n);
            let cands = top.into_sorted();
            assert_eq!(cands.len(), t);
            // scores equal q·(c + decoded residual)
            let mut back = vec![0f32; dim];
            for hit in cands.iter().take(3) {
                crate::pq::decode(prefix.pq_code(hit.idx as usize), &cb, &mut back);
                let want = dot(&q, &c) + dot(&q, &back);
                assert!((hit.score - want).abs() < 1e-4, "{} vs {want}", hit.score);
                assert_eq!(hit.id, hit.idx);
            }
            // rerank the PQ top-t with their int8 records
            let mut recs = Vec::new();
            for hit in &cands {
                recs.extend_from_slice(full.rerank_record(hit.idx as usize));
            }
            let scores = rerank_records(&q, &prefix.centroid(), &recs, t).unwrap();
            let mut rr: Vec<(u32, f32)> = cands.iter().zip(&scores).map(|(h, s)| (h.id, *s)).collect();
            rr.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let got: Vec<u32> = rr.iter().take(k).map(|p| p.0).collect();
            // int8 ranking over every entry
            let all = rerank_records(&q, &c, all_rr, n).unwrap();
            let mut int8: Vec<(u32, f32)> = all.iter().enumerate().map(|(i, s)| (i as u32, *s)).collect();
            int8.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let want: Vec<u32> = int8.iter().take(k).map(|p| p.0).collect();
            assert_eq!(got, want, "query {qi}");
            // PQ alone is a much rougher ranking; report overlap with exact for context
            let exact = exact_top(&xs, &q, k);
            let pq_only = cands.iter().take(k).filter(|h| exact.contains(&(h.id as usize))).count();
            assert!(pq_only >= 3, "query {qi}: PQ-only top-{k} overlap {pq_only}");
        }
        // misuse is an error, not a panic
        let q = xs[0].clone();
        let table = QueryTable::new(&q, &cb).unwrap();
        let v2 = build_v2(&c, &xs[..10]);
        let v2v = SlabView::parse(&v2).unwrap();
        assert!(scan_slab_pq(&v2v, &table, &q, &mut TopK::new(5), None, None).is_err());
        let other = Codebook::init_from_samples(&vec![0.5f32; 300 * dim], dim, 8, 1).unwrap();
        let t8 = QueryTable::new(&q, &other).unwrap();
        assert!(scan_slab_pq(&prefix, &t8, &q, &mut TopK::new(5), None, None).is_err());
        assert!(scan_slab_pq(&prefix, &table, &q[..dim - 2], &mut TopK::new(5), None, None).is_err());
    }

    #[test]
    #[should_panic(expected = "scan_slab_pq")]
    fn scan_slab_rejects_v3() {
        let (_, xs, _, slab) = pq_fixture(16, 4, 300, 3);
        let v = SlabView::parse(&slab).unwrap();
        scan_slab(&v, &xs[0], &mut TopK::new(3), None, None);
    }

    #[test]
    fn scan_delta_matches_brute_force_ranking() {
        use crate::delta::{DeltaEncoder, DeltaView};
        use crate::route::Router;
        let (dim, k, n, top) = (64, 8, 400, 20);
        let mut rng = Rng::new(13);
        // k cluster centres; vectors spread around them
        let mut means = Vec::new();
        let mut xs = Vec::new();
        for c in 0..k {
            let (m, cx) = cluster(dim, n / k, 100 + c as u64);
            means.extend(m.iter().map(|v| v * 0.9));
            xs.extend(cx);
        }
        let mut enc = DeltaEncoder::new(dim, k, means.clone(), 1).unwrap();
        for (i, x) in xs.iter().enumerate() {
            enc.add(x, i as u32, 0, &[]);
        }
        let bytes = enc.finish();
        let v = DeltaView::parse(&bytes).unwrap();
        let router = Router::new(dim, 1, k, &vec![1.0; dim], means.clone());
        for qi in 0..10 {
            let mut q: Vec<f32> = xs[qi * 37].iter().map(|v| v + (rng.below(2000) as f32 / 1000.0 - 1.0) * 0.05).collect();
            normalize(&mut q);
            let qc = router.cell_dots(&q);
            let mut t = TopK::new(top);
            let st = scan_delta(&v, &q, &qc, &mut t, None, None).unwrap();
            assert_eq!((st.scanned, st.skipped, st.filtered), (n, 0, 0));
            assert!(st.pushed >= top.min(n) && st.pushed <= n, "pushed {}", st.pushed);
            let hits = t.into_sorted();
            let exact = exact_top(&xs, &q, top);
            // every hit's score is the cosine within int8 error
            for h in &hits {
                assert!((h.score - dot(&q, &xs[h.id as usize])).abs() < 5e-3, "q {qi} id {}", h.id);
                assert_eq!(h.id, h.idx);
            }
            // same ranking as brute force: rank j holds a vector whose exact
            // cosine equals the j-th best exact cosine up to twice the int8 error
            let got: Vec<usize> = hits.iter().map(|h| h.id as usize).collect();
            assert_eq!(got.len(), top);
            assert_eq!(got[0], exact[0], "q {qi}");
            for (j, (g, e)) in got.iter().zip(&exact).enumerate() {
                let d = (dot(&q, &xs[*g]) - dot(&q, &xs[*e])).abs();
                assert!(d < 1e-2, "q {qi} rank {j}: {g} vs {e} ({d})");
            }
            let overlap = got.iter().filter(|i| exact.contains(i)).count();
            assert!(overlap >= top - 2, "q {qi}: overlap {overlap}");
        }
        // a qc table that does not cover the cells is an error
        let q = xs[0].clone();
        assert!(scan_delta(&v, &q, &router.cell_dots(&q)[..k - 1], &mut TopK::new(3), None, None).is_err());
        assert!(scan_delta(&v, &q[..dim - 2], &router.cell_dots(&q), &mut TopK::new(3), None, None).is_err());
    }

    fn record_ids(recs: &[u8], dim: usize) -> Vec<u32> {
        crate::part::records(recs, dim).map(|r| r.unwrap().id).collect()
    }

    #[test]
    fn slab_records_rebuild_identical_slabs() {
        use crate::slab::SlabWriter;
        use crate::tomb::Tombstones;
        let dim = 64;
        // v2 with payloads and attrs
        let (c, xs) = cluster(dim, 50, 19);
        let mut b = SlabBuilder::new(dim, 33, &c);
        let (mut c4, mut c8, mut scratch) = (vec![0u8; dim / 2], vec![0i8; dim], vec![0f32; dim]);
        for (i, x) in xs.iter().enumerate() {
            let s4 = encode_residual_sq4(x, &c, &mut c4, &mut scratch);
            let s8 = encode_residual_sq8(x, &c, &mut c8);
            b.push(500 + i as u32, &c4, s4, &c8, s8, &vec![i as u8; i % 5], i as u32 * 11);
        }
        let v2 = b.finish();
        // v3 from a part with PQ codes; v1 from the v2 records
        let (c3, _, _, v3) = pq_fixture(dim, 16, 300, 5);
        let mut recs = Vec::new();
        let v2v = SlabView::parse(&v2).unwrap();
        assert_eq!(v2v.to_part_records(&mut recs, None).unwrap(), 50);
        let mut w1 = SlabWriter::new(1, dim, 33, &c).unwrap();
        w1.add_part_records(&recs, dim).unwrap();
        let v1 = w1.finish();
        for (fmt, slab, cent, cell) in [(2u8, &v2, &c, 33u32), (3, &v3, &c3, 9), (1, &v1, &c, 33)] {
            let v = SlabView::parse(slab).unwrap();
            let mut recs = Vec::new();
            assert_eq!(v.to_part_records(&mut recs, None).unwrap() as usize, v.count);
            let mut w = SlabWriter::new(fmt, dim, cell, cent).unwrap();
            assert_eq!(w.add_part_records(&recs, dim).unwrap() as usize, v.count);
            assert_eq!(&w.finish(), slab, "format {fmt} rebuild differs");
            // tombstoned ids are skipped
            let gone: Vec<u32> = (0..v.count).step_by(3).map(|i| v.id(i)).collect();
            let t = Tombstones::from_ids(&gone);
            let mut kept = Vec::new();
            let n = v.to_part_records(&mut kept, Some(&t)).unwrap() as usize;
            assert_eq!(n, v.count - gone.len());
            let ids = record_ids(&kept, dim);
            assert!(ids.iter().all(|id| !gone.contains(id)) && ids.len() == n);
            // v3/v1 records carry no 4-bit codes: a format 2 writer rejects them
            let mut w2 = SlabWriter::new(2, dim, cell, cent).unwrap();
            let r = w2.add_part_records(&recs, dim);
            if fmt == 2 {
                assert!(r.is_ok());
            } else {
                assert!(r.is_err(), "format {fmt} records into a format 2 writer");
                assert_eq!(w2.len(), 0);
            }
            // a scan prefix is not enough
            if fmt != 1 {
                assert!(SlabView::parse(&slab[..v.scan_len()]).unwrap().to_part_records(&mut Vec::new(), None).is_err());
            }
        }
        // v2 records carry no PQ codes: a format 3 writer rejects them
        let mut w3 = SlabWriter::new(3, dim, 33, &c).unwrap();
        assert!(w3.add_part_records(&recs, dim).is_err());
    }

    #[test]
    fn compaction_merges_slab_and_delta_records() {
        use crate::delta::{DeltaEncoder, DeltaView};
        use crate::slab::SlabWriter;
        use crate::tomb::Tombstones;
        let (dim, m, n, cell) = (64, 16, 300, 9usize);
        let (c, xs, cb, v3) = pq_fixture(dim, m, n, 29);
        // 10 cells; cell 9 is the slab's mean, the others random directions
        let mut rng = Rng::new(4);
        let mut means = Vec::new();
        for j in 0..10 {
            if j == cell {
                means.extend_from_slice(&c);
            } else {
                let mut r: Vec<f32> = (0..dim).map(|_| rng.below(2001) as f32 / 1000.0 - 1.0).collect();
                normalize(&mut r);
                means.extend(r);
            }
        }
        // new vectors: 40 near the slab's cluster, 20 elsewhere
        let (_, near) = cluster(dim, 40, 29);
        let far: Vec<Vec<f32>> = (0..20).map(|j| {
            let mut x = means[(j % 9) * dim..(j % 9 + 1) * dim].to_vec();
            x[j] += 0.3;
            normalize(&mut x);
            x
        }).collect();
        let news: Vec<Vec<f32>> = near.into_iter().chain(far).collect();
        let build_delta = |with_pq: bool| {
            let mut enc = DeltaEncoder::new(dim, 10, means.clone(), 1).unwrap();
            if with_pq {
                enc.set_codebook(cb.clone()).unwrap();
            }
            let cells: Vec<usize> = news.iter().enumerate().map(|(i, x)| enc.add(x, 10_000 + i as u32, 1, b"new")).collect();
            (enc.finish(), cells)
        };
        let (delta, cells) = build_delta(true);
        let dv = DeltaView::parse(&delta).unwrap();
        let in_cell: Vec<u32> = (0..news.len()).filter(|&i| cells[i] == cell).map(|i| 10_000 + i as u32).collect();
        assert!(in_cell.len() >= 30, "{}", in_cell.len());
        assert_eq!(dv.cell_counts().iter().find(|p| p.0 == cell as u32).unwrap().1 as usize, in_cell.len());
        // delete 10 old ids and 3 new ones (one outside the cell)
        let mut dead: Vec<u32> = (0..10).map(|i| i * 7).collect();
        dead.extend_from_slice(&in_cell[..2]);
        dead.push(10_059);
        let t = Tombstones::from_ids(&dead);

        let old = SlabView::parse(&v3).unwrap();
        let mut w = SlabWriter::new(3, dim, cell as u32, &c).unwrap();
        let mut recs = Vec::new();
        old.to_part_records(&mut recs, Some(&t)).unwrap();
        dv.records_for_cell(cell as u32, Some(&t), &mut recs);
        w.add_part_records(&recs, dim).unwrap();
        let merged = w.finish();
        let mv = SlabView::parse(&merged).unwrap();
        assert_eq!(mv.count, n - 10 + in_cell.len() - 2);
        let ids: Vec<u32> = (0..mv.count).map(|i| mv.id(i)).collect();
        assert!(ids.iter().all(|id| !t.contains(*id)));
        for id in (0..n as u32).filter(|i| !t.contains(*i)).chain(in_cell[2..].iter().copied()) {
            assert!(ids.contains(&id), "missing {id}");
        }
        // a new vector is found by the PQ scan of the compacted slab and reranks to ~1
        let qi = (in_cell[5] - 10_000) as usize;
        let q = news[qi].clone();
        let table = QueryTable::new(&q, &cb).unwrap();
        let mut top = TopK::new(10);
        scan_slab_pq(&mv, &table, &q, &mut top, None, None).unwrap();
        let hit = top.into_sorted().into_iter().find(|h| h.id == in_cell[5]).expect("new vector not in top 10");
        let s = rerank_records(&q, &c, mv.rerank_record(hit.idx as usize), 1).unwrap()[0];
        assert!((s - 1.0).abs() < 5e-3, "{s}");
        assert_eq!(mv.payload(hit.idx as usize), b"new");

        // format 2 compaction: v2 slab + delta records with or without PQ codes
        let v2 = build_v2(&c, &xs);
        for with_pq in [false, true] {
            let (d, _) = build_delta(with_pq);
            let dv = DeltaView::parse(&d).unwrap();
            let mut w = SlabWriter::new(2, dim, cell as u32, &c).unwrap();
            let mut recs = Vec::new();
            SlabView::parse(&v2).unwrap().to_part_records(&mut recs, Some(&t)).unwrap();
            dv.records_for_cell(cell as u32, Some(&t), &mut recs);
            assert_eq!(w.add_part_records(&recs, dim).unwrap() as usize, n - 10 + in_cell.len() - 2);
            // the delta entries' scan-tier codes are kept
            let s = w.finish();
            let sv = SlabView::parse(&s).unwrap();
            let j = (0..sv.count).find(|&i| sv.id(i) == in_cell[5]).unwrap();
            let k = (0..dv.count).find(|&i| dv.id(i) == in_cell[5]).unwrap();
            assert_eq!((sv.code4(j), sv.scale4(j), sv.code8(j)), (dv.code4(k), dv.scale4(k), dv.code8(k)));
            // a v3 compaction needs delta records with PQ codes
            if !with_pq {
                let mut recs = Vec::new();
                dv.records_for_cell(cell as u32, None, &mut recs);
                assert!(SlabWriter::new(3, dim, cell as u32, &c).unwrap().add_part_records(&recs, dim).is_err());
            }
        }
    }

    /// Tombstones are applied before heap insertion: with `k` = 1 and the best
    /// entry tombstoned, the second-best live entry is returned (filtering the
    /// top-k afterwards would return nothing). `run(topk, skip)` scans one
    /// input of `n` entries.
    fn check_skip_before_heap(n: usize, run: &dyn Fn(&mut TopK, Option<&Tombstones>) -> ScanStats) {
        let mut all = TopK::new(n);
        assert_eq!(run(&mut all, None), ScanStats { scanned: n, skipped: 0, filtered: 0, pushed: n });
        let full = all.into_sorted();
        assert_eq!(full.len(), n);
        assert!(full[0].score > full[1].score);
        let best = Tombstones::from_ids(&[full[0].id]);
        // the bug: top-1 then filter leaves nothing although live entries remain
        let mut t = TopK::new(1);
        run(&mut t, None);
        assert!(best.filter_hits(t.into_sorted()).is_empty());
        // the fix
        let mut t = TopK::new(1);
        let st = run(&mut t, Some(&best));
        assert_eq!(st.scanned, n);
        assert!(st.skipped >= 1);
        assert_eq!(t.into_sorted(), vec![full[1]]);
        // several dead entries around the top: the result is the live ranking's top k
        let dead = Tombstones::from_ids(&[full[0].id, full[2].id, full[3].id, full[n - 1].id]);
        let mut t = TopK::new(3);
        let st = run(&mut t, Some(&dead));
        assert!((3..=4).contains(&st.skipped), "{st:?}");
        let want: Vec<_> = full.iter().filter(|h| !dead.contains(h.id)).take(3).copied().collect();
        assert_eq!(t.into_sorted(), want);
        // an empty set changes nothing
        let mut t = TopK::new(5);
        assert_eq!(run(&mut t, Some(&Tombstones::new())).skipped, 0);
        assert_eq!(t.into_sorted(), full[..5].to_vec());
    }

    #[test]
    fn scan_slab_skips_tombstones_before_heap() {
        let dim = 32;
        let (c, xs) = cluster(dim, 60, 31);
        // v2 prefix
        let v2 = build_v2(&c, &xs);
        let h = SlabHeader::parse(&v2).unwrap();
        let view = SlabView::parse(&v2[..h.scan_len]).unwrap();
        let q = xs[11].clone();
        check_skip_before_heap(60, &|t, skip| scan_slab(&view, &q, t, skip, None));
        // v1
        let mut b = SlabBuilderV1::new(dim, 0, &c);
        let mut code = vec![0i8; dim];
        for (i, x) in xs.iter().enumerate() {
            let s = encode_residual_sq8(x, &c, &mut code);
            b.push(i as u32, &code, s, &[], 0);
        }
        let v1 = b.finish();
        let view = SlabView::parse(&v1).unwrap();
        check_skip_before_heap(60, &|t, skip| scan_slab(&view, &q, t, skip, None));
    }

    #[test]
    fn scan_slab_pq_skips_tombstones_before_heap() {
        let (dim, m, n) = (64, 16, 300);
        let (_, xs, cb, slab) = pq_fixture(dim, m, n, 17);
        let h = SlabHeader::parse(&slab).unwrap();
        let prefix = SlabView::parse(&slab[..h.scan_len]).unwrap();
        let q = xs[42].clone();
        let table = QueryTable::new(&q, &cb).unwrap();
        check_skip_before_heap(n, &|t, skip| scan_slab_pq(&prefix, &table, &q, t, skip, None).unwrap());
    }

    #[test]
    fn scan_delta_skips_tombstones_before_heap() {
        use crate::delta::{DeltaEncoder, DeltaView};
        use crate::route::Router;
        let (dim, k, n) = (32, 4, 120);
        let mut means = Vec::new();
        let mut xs = Vec::new();
        for c in 0..k {
            let (m, cx) = cluster(dim, n / k, 200 + c as u64);
            means.extend(m.iter().map(|v| v * 0.9));
            xs.extend(cx);
        }
        let mut enc = DeltaEncoder::new(dim, k, means.clone(), 1).unwrap();
        for (i, x) in xs.iter().enumerate() {
            enc.add(x, 1000 + i as u32, 0, &[]);
        }
        let bytes = enc.finish();
        let v = DeltaView::parse(&bytes).unwrap();
        let router = Router::new(dim, 1, k, &vec![1.0; dim], means);
        let q = xs[7].clone();
        let qc = router.cell_dots(&q);
        check_skip_before_heap(n, &|t, skip| scan_delta(&v, &q, &qc, t, skip, None).unwrap());
    }

    /// Attribute filters are applied before heap insertion, like tombstones: with `k` = 1 and the
    /// best entry failing the filter, the best matching entry is returned. `attrs_of(id)` is the
    /// word each entry was written with.
    fn check_filter_before_heap(n: usize, run: &dyn Fn(&mut TopK, Option<&AttrFilter>) -> ScanStats, attrs_of: &dyn Fn(u32) -> u32) {
        use crate::filter::{Clause, Op};
        let mut all = TopK::new(n);
        assert_eq!(run(&mut all, None), ScanStats { scanned: n, skipped: 0, filtered: 0, pushed: n });
        let full = all.into_sorted();
        assert_eq!(full.len(), n);
        // pick the class the best entry is NOT in, so a post-filter of the top-1 would be empty
        let want_class = (attrs_of(full[0].id) + 1) % 3;
        let eq = AttrFilter::new(vec![Clause { mask: u32::MAX, op: Op::Eq, a: want_class, b: 0, values: vec![] }]);
        let want: Vec<_> = full.iter().filter(|h| attrs_of(h.id) == want_class).copied().collect();
        assert!(want.len() >= 5);
        let mut t = TopK::new(1);
        let st = run(&mut t, Some(&eq));
        assert_eq!(st.scanned, n);
        assert!(st.filtered >= 1);
        assert_eq!(t.into_sorted(), vec![want[0]]);
        let mut t = TopK::new(5);
        run(&mut t, Some(&eq));
        assert_eq!(t.into_sorted(), want[..5].to_vec());
        // a set and a bit test; conjunction with a tombstone set
        let in_set = AttrFilter::new(vec![Clause { mask: u32::MAX, op: Op::In, a: 0, b: 0, values: vec![want_class, (want_class + 1) % 3] }]);
        let want2: Vec<_> = full.iter().filter(|h| attrs_of(h.id) != (want_class + 2) % 3).copied().collect();
        let mut t = TopK::new(4);
        run(&mut t, Some(&in_set));
        assert_eq!(t.into_sorted(), want2[..4].to_vec());
        // a filter nothing matches: every entry is tested (the heap never fills) and dropped
        let mut t = TopK::new(3);
        let st = run(&mut t, Some(&AttrFilter::nothing()));
        assert_eq!((st.filtered, t.into_sorted().len()), (n, 0));
    }

    #[test]
    fn scan_slab_filters_attrs_before_heap() {
        let dim = 32;
        let (c, xs) = cluster(dim, 60, 33);
        let mut b = SlabBuilder::new(dim, 0, &c);
        let (mut c4, mut c8, mut scratch) = (vec![0u8; dim / 2], vec![0i8; dim], vec![0f32; dim]);
        for (i, x) in xs.iter().enumerate() {
            let s4 = encode_residual_sq4(x, &c, &mut c4, &mut scratch);
            let s8 = encode_residual_sq8(x, &c, &mut c8);
            b.push(i as u32, &c4, s4, &c8, s8, &[], i as u32 % 3);
        }
        let v2 = b.finish();
        let h = SlabHeader::parse(&v2).unwrap();
        let view = SlabView::parse(&v2[..h.scan_len]).unwrap();
        let q = xs[11].clone();
        assert_eq!(view.attrs(4), 1);
        check_filter_before_heap(60, &|t, f| scan_slab(&view, &q, t, None, f), &|id| id % 3);
        // tombstones and the filter compose: the best matching entry tombstoned -> the next one
        use crate::filter::{Clause, Op};
        let mut all = TopK::new(60);
        scan_slab(&view, &q, &mut all, None, None);
        let full = all.into_sorted();
        let eq = AttrFilter::new(vec![Clause { mask: u32::MAX, op: Op::Eq, a: 2, b: 0, values: vec![] }]);
        let twos: Vec<_> = full.iter().filter(|h| h.id % 3 == 2).copied().collect();
        let dead = Tombstones::from_ids(&[twos[0].id]);
        let mut t = TopK::new(1);
        let st = scan_slab(&view, &q, &mut t, Some(&dead), Some(&eq));
        assert_eq!(t.into_sorted(), vec![twos[1]]);
        assert_eq!(st.skipped, 1);
    }

    #[test]
    fn scan_delta_filters_attrs_before_heap() {
        use crate::delta::{DeltaEncoder, DeltaView};
        use crate::route::Router;
        let (dim, k, n) = (32, 4, 120);
        let mut means = Vec::new();
        let mut xs = Vec::new();
        for c in 0..k {
            let (m, cx) = cluster(dim, n / k, 300 + c as u64);
            means.extend(m.iter().map(|v| v * 0.9));
            xs.extend(cx);
        }
        let mut enc = DeltaEncoder::new(dim, k, means.clone(), 1).unwrap();
        for (i, x) in xs.iter().enumerate() {
            enc.add(x, 1000 + i as u32, i as u32 % 3, &[]);
        }
        let bytes = enc.finish();
        let v = DeltaView::parse(&bytes).unwrap();
        assert_eq!(v.attrs(5), 2);
        let router = Router::new(dim, 1, k, &vec![1.0; dim], means);
        let q = xs[7].clone();
        let qc = router.cell_dots(&q);
        check_filter_before_heap(n, &|t, f| scan_delta(&v, &q, &qc, t, None, f).unwrap(), &|id| (id - 1000) % 3);
    }

    #[test]
    fn rerank_records_equals_score_sq8() {
        let dim = 48;
        let (c, xs) = cluster(dim, 20, 11);
        let bytes = build_v2(&c, &xs);
        let v = SlabView::parse(&bytes).unwrap();
        let q = xs[5].clone();
        let qc = dot(&q, &c);
        let h = v.header;
        let all = &bytes[h.rerank_off..h.rerank_off + v.count * h.rerank_record_len];
        let got = rerank_records(&q, &c, all, v.count).unwrap();
        let mut code = vec![0i8; dim];
        for i in 0..v.count {
            let s8 = encode_residual_sq8(&xs[i], &c, &mut code);
            let want = score_sq8(&q, &code, s8, qc);
            assert_eq!(got[i], want, "entry {i}");
            assert_eq!(got[i], score_sq8(&q, v.code8(i), v.scale8(i), qc));
        }
        // a sub-range
        let rl = h.rerank_record_len;
        let sub = rerank_records(&q, &c, &all[3 * rl..6 * rl], 3).unwrap();
        assert_eq!(&sub[..], &got[3..6]);
        assert!(rerank_records(&q, &c, &all[..rl * 2 - 1], 2).is_err());
    }
}
