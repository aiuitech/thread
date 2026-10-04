// Dense-side file written before rustfmt was adopted; kept as is (`cargo fmt` skips it).
#![cfg_attr(rustfmt, rustfmt::skip)]
//! Delta segments: small immutable objects holding vectors added since the
//! last compaction.
//!
//! An update job assigns each new vector to its primary (nearest) cell and
//! encodes its residual against that cell's mean exactly as an assign job
//! encodes a part record (4-bit + int8, + PQ with a codebook). Queries scan
//! every delta segment with the int8 records ([`crate::search::scan_delta`]),
//! next to the probed slabs; compaction re-emits a cell's delta entries as
//! part records ([`DeltaView::records_for_cell`]) and feeds them, together
//! with the old slab's records, to an ordinary slab writer.
//!
//! Layout "VIBD" v1 (little-endian, offsets from the segment start, sections
//! 8-byte aligned, in this order):
//! ```text
//! header (64 bytes)
//!   0  magic "VIBD"   4 version u16 = 1   6 flags u8 = 0   7 reserved u8
//!   8  dim u32 (even)   12 count u32   16 pq_m u32 (0 = no PQ codes)
//!   20 seq u32          segment sequence number
//!   24 entries_off u32  count * 24: id u32, cell u32, attrs u32,
//!                       payload_off u32 (relative to payload section), payload_len u32, reserved u32
//!   28 rerank_off u32   count * (4 + dim): [scale8 f32][code8 i8 * dim] (residual vs mean[cell])
//!   32 pq_off u32       count * pq_m (0 when pq_m = 0)
//!   36 code4_off u32    count * dim/2 packed signed nibbles (low nibble = even dim)
//!   40 scales4_off u32  count * f32
//!   44 payload_off u32  48 payload_len u32
//!   52..64 reserved
//! ```

use crate::error::{need, rd_f32, rd_u16, rd_u32, FormatError};
use crate::math::{argmax_dot_scaled, inv_norms};
use crate::part::write_record;
use crate::pq::Codebook;
use crate::quant::{encode_residual_sq4, encode_residual_sq8};
use crate::route::CentroidTree;
use crate::tomb::Tombstones;

pub const DELTA_MAGIC: &[u8; 4] = b"VIBD";
pub const DELTA_VERSION: u16 = 1;
pub const DELTA_HEADER: usize = 64;
pub const DELTA_ENTRY_BYTES: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaEntry {
    pub id: u32,
    pub cell: u32,
    pub attrs: u32,
    pub payload_off: u32,
    pub payload_len: u32,
}

struct Layout {
    entries_off: usize,
    rerank_off: usize,
    pq_off: usize,
    code4_off: usize,
    scales4_off: usize,
    payload_off: usize,
}

impl Layout {
    fn new(dim: usize, pq_m: usize, n: usize) -> Self {
        let entries_off = DELTA_HEADER;
        let rerank_off = align8(entries_off + n * DELTA_ENTRY_BYTES);
        let pq_start = align8(rerank_off + n * (4 + dim));
        let code4_off = align8(pq_start + n * pq_m);
        let scales4_off = align8(code4_off + n * (dim / 2));
        let payload_off = align8(scales4_off + n * 4);
        let pq_off = if pq_m == 0 { 0 } else { pq_start };
        Layout { entries_off, rerank_off, pq_off, code4_off, scales4_off, payload_off }
    }
}

/// Assembles a delta segment in memory.
pub struct DeltaBuilder {
    dim: usize,
    pq_m: usize,
    seq: u32,
    entries: Vec<DeltaEntry>,
    rerank: Vec<u8>,
    pq: Vec<u8>,
    code4: Vec<u8>,
    scales4: Vec<f32>,
    payload: Vec<u8>,
}

impl DeltaBuilder {
    /// `dim` must be even; `pq_m` is 0 (no PQ codes) or divides `dim`.
    pub fn new(dim: usize, pq_m: usize, seq: u32) -> Self {
        assert!(dim > 0 && dim % 2 == 0, "delta segments require an even dimension");
        assert!(pq_m == 0 || dim % pq_m == 0, "pq_m must divide dim");
        DeltaBuilder {
            dim,
            pq_m,
            seq,
            entries: Vec::new(),
            rerank: Vec::new(),
            pq: Vec::new(),
            code4: Vec::new(),
            scales4: Vec::new(),
            payload: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn pq_m(&self) -> usize {
        self.pq_m
    }

    pub fn seq(&self) -> u32 {
        self.seq
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Appends one entry: its primary `cell`, the residual codes against that
    /// cell's mean (`code4`: `dim/2` bytes, `code8`: `dim`, `pq`: `pq_m`).
    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        id: u32,
        cell: u32,
        attrs: u32,
        scale4: f32,
        code4: &[u8],
        scale8: f32,
        code8: &[i8],
        pq: &[u8],
        payload: &[u8],
    ) {
        assert_eq!(code4.len(), self.dim / 2);
        assert_eq!(code8.len(), self.dim);
        assert_eq!(pq.len(), self.pq_m, "pq code count != segment pq_m");
        let payload_off = self.payload.len() as u32;
        self.payload.extend_from_slice(payload);
        self.entries.push(DeltaEntry { id, cell, attrs, payload_off, payload_len: payload.len() as u32 });
        self.rerank.extend_from_slice(&scale8.to_le_bytes());
        self.rerank.extend(code8.iter().map(|c| *c as u8));
        self.pq.extend_from_slice(pq);
        self.code4.extend_from_slice(code4);
        self.scales4.push(scale4);
    }

    pub fn byte_len(&self) -> usize {
        Layout::new(self.dim, self.pq_m, self.entries.len()).payload_off + self.payload.len()
    }

    pub fn finish(self) -> Vec<u8> {
        let n = self.entries.len();
        let l = Layout::new(self.dim, self.pq_m, n);
        let mut out = vec![0u8; l.payload_off + self.payload.len()];
        out[0..4].copy_from_slice(DELTA_MAGIC);
        out[4..6].copy_from_slice(&DELTA_VERSION.to_le_bytes());
        put_u32(&mut out, 8, self.dim as u32);
        put_u32(&mut out, 12, n as u32);
        put_u32(&mut out, 16, self.pq_m as u32);
        put_u32(&mut out, 20, self.seq);
        put_u32(&mut out, 24, l.entries_off as u32);
        put_u32(&mut out, 28, l.rerank_off as u32);
        put_u32(&mut out, 32, l.pq_off as u32);
        put_u32(&mut out, 36, l.code4_off as u32);
        put_u32(&mut out, 40, l.scales4_off as u32);
        put_u32(&mut out, 44, l.payload_off as u32);
        put_u32(&mut out, 48, self.payload.len() as u32);
        for (i, e) in self.entries.iter().enumerate() {
            let o = l.entries_off + i * DELTA_ENTRY_BYTES;
            put_u32(&mut out, o, e.id);
            put_u32(&mut out, o + 4, e.cell);
            put_u32(&mut out, o + 8, e.attrs);
            put_u32(&mut out, o + 12, e.payload_off);
            put_u32(&mut out, o + 16, e.payload_len);
        }
        out[l.rerank_off..l.rerank_off + self.rerank.len()].copy_from_slice(&self.rerank);
        if self.pq_m > 0 {
            out[l.pq_off..l.pq_off + self.pq.len()].copy_from_slice(&self.pq);
        }
        out[l.code4_off..l.code4_off + self.code4.len()].copy_from_slice(&self.code4);
        for (i, s) in self.scales4.iter().enumerate() {
            put_u32(&mut out, l.scales4_off + i * 4, s.to_bits());
        }
        out[l.payload_off..].copy_from_slice(&self.payload);
        out
    }
}

/// Zero-copy view over a whole delta segment. Every section (and every
/// entry's payload range) is bounds-checked by [`DeltaView::parse`], so the
/// accessors do not fail for `i < count`.
pub struct DeltaView<'a> {
    buf: &'a [u8],
    pub dim: usize,
    pub count: usize,
    pub pq_m: usize,
    pub seq: u32,
    entries_off: usize,
    rerank_off: usize,
    pq_off: usize,
    code4_off: usize,
    scales4_off: usize,
    payload_off: usize,
    payload_len: usize,
}

impl<'a> DeltaView<'a> {
    pub fn parse(buf: &'a [u8]) -> Result<Self, FormatError> {
        need(buf, DELTA_HEADER)?;
        if &buf[0..4] != DELTA_MAGIC {
            return Err(FormatError::BadMagic);
        }
        let version = rd_u16(buf, 4);
        if version != DELTA_VERSION {
            return Err(FormatError::BadVersion(version));
        }
        let dim = rd_u32(buf, 8) as usize;
        let count = rd_u32(buf, 12) as usize;
        let pq_m = rd_u32(buf, 16) as usize;
        if dim == 0 || dim % 2 != 0 {
            return Err(FormatError::Inconsistent("delta dim must be even and non-zero"));
        }
        if pq_m > 0 && dim % pq_m != 0 {
            return Err(FormatError::Inconsistent("delta pq_m does not divide dim"));
        }
        let v = DeltaView {
            buf,
            dim,
            count,
            pq_m,
            seq: rd_u32(buf, 20),
            entries_off: rd_u32(buf, 24) as usize,
            rerank_off: rd_u32(buf, 28) as usize,
            pq_off: rd_u32(buf, 32) as usize,
            code4_off: rd_u32(buf, 36) as usize,
            scales4_off: rd_u32(buf, 40) as usize,
            payload_off: rd_u32(buf, 44) as usize,
            payload_len: rd_u32(buf, 48) as usize,
        };
        let end = |off: usize, per: usize| {
            count
                .checked_mul(per)
                .and_then(|l| off.checked_add(l))
                .ok_or(FormatError::Inconsistent("delta section overflow"))
        };
        let mut sections = vec![
            end(v.entries_off, DELTA_ENTRY_BYTES)?,
            end(v.rerank_off, 4 + dim)?,
            end(v.code4_off, dim / 2)?,
            end(v.scales4_off, 4)?,
            v.payload_off.checked_add(v.payload_len).ok_or(FormatError::Inconsistent("delta section overflow"))?,
        ];
        if pq_m > 0 {
            sections.push(end(v.pq_off, pq_m)?);
        }
        need(buf, sections.into_iter().max().unwrap())?;
        for i in 0..count {
            let e = v.entry(i);
            if e.payload_off as usize + e.payload_len as usize > v.payload_len {
                return Err(FormatError::Inconsistent("delta entry payload out of range"));
            }
        }
        Ok(v)
    }

    /// Serialized size from the header (the payload section ends the segment).
    pub fn byte_len(&self) -> usize {
        self.payload_off + self.payload_len
    }

    #[inline]
    pub fn entry(&self, i: usize) -> DeltaEntry {
        let o = self.entries_off + i * DELTA_ENTRY_BYTES;
        DeltaEntry {
            id: rd_u32(self.buf, o),
            cell: rd_u32(self.buf, o + 4),
            attrs: rd_u32(self.buf, o + 8),
            payload_off: rd_u32(self.buf, o + 12),
            payload_len: rd_u32(self.buf, o + 16),
        }
    }

    #[inline]
    pub fn id(&self, i: usize) -> u32 {
        rd_u32(self.buf, self.entries_off + i * DELTA_ENTRY_BYTES)
    }

    #[inline]
    pub fn cell(&self, i: usize) -> u32 {
        rd_u32(self.buf, self.entries_off + i * DELTA_ENTRY_BYTES + 4)
    }

    /// Rerank record `i` (`[scale8 f32][code8 i8 * dim]`, same layout as a slab's).
    #[inline]
    pub fn rerank_record(&self, i: usize) -> &'a [u8] {
        let rl = 4 + self.dim;
        &self.buf[self.rerank_off + i * rl..self.rerank_off + (i + 1) * rl]
    }

    #[inline]
    pub fn scale8(&self, i: usize) -> f32 {
        rd_f32(self.buf, self.rerank_off + i * (4 + self.dim))
    }

    #[inline]
    pub fn code8(&self, i: usize) -> &'a [i8] {
        let s = &self.rerank_record(i)[4..];
        // i8 and u8 have identical layout and alignment.
        unsafe { core::slice::from_raw_parts(s.as_ptr() as *const i8, s.len()) }
    }

    /// PQ codes of entry `i` (empty when `pq_m == 0`).
    #[inline]
    pub fn pq_code(&self, i: usize) -> &'a [u8] {
        let m = self.pq_m;
        if m == 0 {
            return &[];
        }
        &self.buf[self.pq_off + i * m..self.pq_off + (i + 1) * m]
    }

    #[inline]
    pub fn code4(&self, i: usize) -> &'a [u8] {
        let n = self.dim / 2;
        &self.buf[self.code4_off + i * n..self.code4_off + (i + 1) * n]
    }

    #[inline]
    pub fn scale4(&self, i: usize) -> f32 {
        rd_f32(self.buf, self.scales4_off + i * 4)
    }

    pub fn payload(&self, i: usize) -> &'a [u8] {
        let e = self.entry(i);
        let s = self.payload_off + e.payload_off as usize;
        &self.buf[s..s + e.payload_len as usize]
    }

    /// Appends entry `i` as a part record (part format version 2; residual
    /// against its cell's mean, so it can go straight into that cell's slab).
    pub fn to_part_record(&self, i: usize, out: &mut Vec<u8>) {
        let e = self.entry(i);
        write_record(
            out,
            e.id,
            self.scale4(i),
            self.scale8(i),
            e.attrs,
            0,
            self.code4(i),
            self.code8(i),
            self.pq_code(i),
            self.payload(i),
        );
    }

    /// Appends the part records of every entry of `cell` not in `skip`, in
    /// entry order. Returns the number appended.
    pub fn records_for_cell(&self, cell: u32, skip: Option<&Tombstones>, out: &mut Vec<u8>) -> u32 {
        let mut n = 0;
        for i in 0..self.count {
            if self.cell(i) == cell && !skip.is_some_and(|t| t.contains(self.id(i))) {
                self.to_part_record(i, out);
                n += 1;
            }
        }
        n
    }

    /// Distinct cells with their entry counts, ascending by cell.
    pub fn cell_counts(&self) -> Vec<(u32, u32)> {
        let mut cells: Vec<u32> = (0..self.count).map(|i| self.cell(i)).collect();
        cells.sort_unstable();
        let mut out: Vec<(u32, u32)> = Vec::new();
        for c in cells {
            match out.last_mut() {
                Some((lc, n)) if *lc == c => *n += 1,
                _ => out.push((c, 1)),
            }
        }
        out
    }
}

/// Update-path encoder: holds the cell means (+ inverse norms, + optional PQ
/// codebook) and a [`DeltaBuilder`]; each added vector goes to its primary
/// cell (highest cosine over all `k` means) with residual codes against that
/// mean, exactly as an assign job with `m = 1` and exhaustive routing encodes it.
/// With [`DeltaEncoder::set_tree`] the primary cell comes from a pruned
/// [`CentroidTree`] search instead (identical when `beam >= groups`).
pub struct DeltaEncoder {
    dim: usize,
    k: usize,
    means: Vec<f32>,
    inv: Vec<f32>,
    codebook: Option<Codebook>,
    builder: DeltaBuilder,
    code4: Vec<u8>,
    code8: Vec<i8>,
    scratch: Vec<f32>,
    pq: Vec<u8>,
    tree: Option<(CentroidTree, usize)>,
}

impl DeltaEncoder {
    /// `means`: `k * dim` raw (unnormalized) cell means, as in `fine.f32`.
    pub fn new(dim: usize, k: usize, means: Vec<f32>, seq: u32) -> Result<Self, FormatError> {
        if dim == 0 || dim % 2 != 0 {
            return Err(FormatError::Inconsistent("dim must be even and non-zero"));
        }
        if k == 0 || means.len() != k * dim {
            return Err(FormatError::Inconsistent("means buffer size != k * dim"));
        }
        let inv = inv_norms(&means, dim);
        Ok(DeltaEncoder {
            dim,
            k,
            means,
            inv,
            codebook: None,
            builder: DeltaBuilder::new(dim, 0, seq),
            code4: vec![0; dim / 2],
            code8: vec![0; dim],
            scratch: vec![0.0; dim],
            pq: Vec::new(),
            tree: None,
        })
    }

    /// Picks primary cells with `tree.nearest(.., beam)` (a [`CentroidTree`]
    /// built over these same `k` means) instead of the exhaustive search.
    /// Applies to vectors added afterwards.
    pub fn set_tree(&mut self, tree: CentroidTree, beam: usize) -> Result<(), FormatError> {
        if tree.dim != self.dim || tree.k != self.k {
            return Err(FormatError::Inconsistent("tree dim / k != encoder dim / k"));
        }
        self.tree = Some((tree, beam.max(1)));
        Ok(())
    }

    /// The tree and beam set by [`DeltaEncoder::set_tree`], if any.
    pub fn tree(&self) -> Option<(&CentroidTree, usize)> {
        self.tree.as_ref().map(|(t, b)| (t, *b))
    }

    /// Entries also carry PQ codes under `cb` (needed to compact into v3
    /// slabs). Only before the first `add`.
    pub fn set_codebook(&mut self, cb: Codebook) -> Result<(), FormatError> {
        if !self.builder.is_empty() {
            return Err(FormatError::Inconsistent("set the codebook before adding vectors"));
        }
        if cb.dim != self.dim {
            return Err(FormatError::Inconsistent("codebook dim != job dim"));
        }
        self.builder = DeltaBuilder::new(self.dim, cb.m, self.builder.seq());
        self.pq = vec![0; cb.m];
        self.codebook = Some(cb);
        Ok(())
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn pq_m(&self) -> usize {
        self.builder.pq_m()
    }

    pub fn len(&self) -> usize {
        self.builder.len()
    }

    pub fn is_empty(&self) -> bool {
        self.builder.is_empty()
    }

    pub fn byte_len(&self) -> usize {
        self.builder.byte_len()
    }

    /// Primary cell of unit vector `x` (highest cosine to the raw means;
    /// within the tree's beam when a tree is set).
    pub fn primary_cell(&self, x: &[f32]) -> usize {
        if let Some((tree, beam)) = &self.tree {
            return tree.nearest(x, &self.means, &self.inv, *beam).0;
        }
        argmax_dot_scaled(x, &self.means, &self.inv, self.dim).0
    }

    /// Encodes `x` (a unit vector, `dim` floats) into its primary cell and
    /// appends it. Returns the cell.
    pub fn add(&mut self, x: &[f32], id: u32, attrs: u32, payload: &[u8]) -> usize {
        assert_eq!(x.len(), self.dim);
        let cell = self.primary_cell(x);
        let mean = &self.means[cell * self.dim..(cell + 1) * self.dim];
        let scale4 = encode_residual_sq4(x, mean, &mut self.code4, &mut self.scratch);
        let scale8 = encode_residual_sq8(x, mean, &mut self.code8);
        if let Some(cb) = &self.codebook {
            for i in 0..self.dim {
                self.scratch[i] = x[i] - mean[i];
            }
            crate::pq::encode(&self.scratch, cb, &mut self.pq);
        }
        self.builder.push(id, cell as u32, attrs, scale4, &self.code4, scale8, &self.code8, &self.pq, payload);
        cell
    }

    pub fn finish(self) -> Vec<u8> {
        self.builder.finish()
    }
}

#[inline]
fn align8(x: usize) -> usize {
    (x + 7) & !7
}

#[inline]
fn put_u32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{dot, normalize, Rng};
    use crate::part::records;
    use crate::quant::decode_residual_sq8;

    fn sample(pq_m: usize) -> (Vec<u8>, Vec<(u32, u32, u32, Vec<u8>, Vec<i8>, Vec<u8>, Vec<u8>)>) {
        let dim = 8;
        let mut b = DeltaBuilder::new(dim, pq_m, 17);
        let mut want = Vec::new();
        for i in 0..5u32 {
            let c4: Vec<u8> = (0..dim / 2).map(|d| (i as usize * 16 + d) as u8).collect();
            let c8: Vec<i8> = (0..dim).map(|d| (i as i32 * 10 - d as i32) as i8).collect();
            let pq: Vec<u8> = (0..pq_m).map(|d| (i as usize * 3 + d) as u8).collect();
            let payload = vec![b'a' + i as u8; i as usize];
            let cell = [4u32, 1, 4, 9, 1][i as usize];
            b.push(100 + i, cell, i * 7, 0.5 + i as f32, &c4, 0.01 * (i + 1) as f32, &c8, &pq, &payload);
            want.push((100 + i, cell, i * 7, c4, c8, pq, payload));
        }
        let bl = b.byte_len();
        let bytes = b.finish();
        assert_eq!(bytes.len(), bl);
        (bytes, want)
    }

    #[test]
    fn delta_roundtrip_with_payloads() {
        for pq_m in [0usize, 4] {
            let (bytes, want) = sample(pq_m);
            let v = DeltaView::parse(&bytes).unwrap();
            assert_eq!((v.dim, v.count, v.pq_m, v.seq), (8, 5, pq_m, 17));
            assert_eq!(v.byte_len(), bytes.len());
            for (i, (id, cell, attrs, c4, c8, pq, payload)) in want.iter().enumerate() {
                let e = v.entry(i);
                assert_eq!((e.id, e.cell, e.attrs), (*id, *cell, *attrs));
                assert_eq!(v.code4(i), &c4[..]);
                assert_eq!(v.code8(i), &c8[..]);
                assert_eq!(v.pq_code(i), &pq[..]);
                assert_eq!(v.payload(i), &payload[..]);
                assert_eq!(v.scale4(i), 0.5 + i as f32);
                assert_eq!(v.scale8(i), 0.01 * (i + 1) as f32);
                // the part record carries the same fields
                let mut rec = Vec::new();
                v.to_part_record(i, &mut rec);
                let r: Vec<_> = records(&rec, 8).map(|r| r.unwrap()).collect();
                assert_eq!(r.len(), 1);
                assert_eq!((r[0].id, r[0].attrs, r[0].flags), (*id, *attrs, 0));
                assert_eq!((r[0].scale4, r[0].scale8), (v.scale4(i), v.scale8(i)));
                assert_eq!((r[0].code4, r[0].code8, r[0].pq, r[0].payload), (&c4[..], &c8[..], &pq[..], &payload[..]));
            }
            for off in [24usize, 28, 36, 40, 44] {
                assert_eq!(rd_u32(&bytes, off) % 8, 0);
            }
            assert_eq!(rd_u32(&bytes, 32) == 0, pq_m == 0);
            assert_eq!(v.cell_counts(), vec![(1, 2), (4, 2), (9, 1)]);
            let mut out = Vec::new();
            assert_eq!(v.records_for_cell(4, None, &mut out), 2);
            let ids: Vec<u32> = records(&out, 8).map(|r| r.unwrap().id).collect();
            assert_eq!(ids, vec![100, 102]);
            let t = Tombstones::from_ids(&[102]);
            out.clear();
            assert_eq!(v.records_for_cell(4, Some(&t), &mut out), 1);
            assert_eq!(v.records_for_cell(5, None, &mut out), 0);
            // truncation and corruption are errors
            assert!(DeltaView::parse(&bytes[..bytes.len() - 1]).is_err());
            let mut bad = bytes.clone();
            bad[0] = b'X';
            assert_eq!(DeltaView::parse(&bad).map(|_| ()), Err(FormatError::BadMagic));
            let mut bad = bytes.clone();
            bad[4] = 2;
            assert_eq!(DeltaView::parse(&bad).map(|_| ()), Err(FormatError::BadVersion(2)));
            // entry 4's payload_len past the payload section
            let mut bad = bytes.clone();
            put_u32(&mut bad, 64 + 4 * 24 + 16, 1000);
            assert!(DeltaView::parse(&bad).is_err());
        }
        // empty segment
        let e = DeltaBuilder::new(4, 0, 3).finish();
        let v = DeltaView::parse(&e).unwrap();
        assert_eq!((v.count, v.seq), (0, 3));
        assert!(v.cell_counts().is_empty());
    }

    fn rand_unit(rng: &mut Rng, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.below(20001) as f32 / 10000.0 - 1.0).collect();
        normalize(&mut v);
        v
    }

    #[test]
    fn encoder_assigns_primary_cell_and_decodes() {
        let (dim, k, n) = (32, 12, 200);
        let mut rng = Rng::new(5);
        // unnormalized means of different lengths: assignment must be by cosine
        let means: Vec<f32> = (0..k)
            .flat_map(|c| rand_unit(&mut rng, dim).into_iter().map(move |v| v * (0.5 + c as f32 * 0.1)))
            .collect();
        let mut enc = DeltaEncoder::new(dim, k, means.clone(), 2).unwrap();
        let mut xs = Vec::new();
        let mut cells = Vec::new();
        for i in 0..n {
            let x = rand_unit(&mut rng, dim);
            cells.push(enc.add(&x, 1000 + i as u32, i as u32, &[i as u8]));
            xs.push(x);
        }
        assert_eq!(enc.len(), n);
        let bytes = enc.finish();
        let v = DeltaView::parse(&bytes).unwrap();
        assert_eq!((v.count, v.pq_m, v.seq), (n, 0, 2));
        let mut back = vec![0f32; dim];
        for (i, x) in xs.iter().enumerate() {
            // brute-force cosine argmax
            let want = (0..k)
                .max_by(|&a, &b| {
                    let ma = &means[a * dim..(a + 1) * dim];
                    let mb = &means[b * dim..(b + 1) * dim];
                    (dot(x, ma) / dot(ma, ma).sqrt()).total_cmp(&(dot(x, mb) / dot(mb, mb).sqrt()))
                })
                .unwrap();
            assert_eq!(cells[i], want);
            assert_eq!(v.cell(i) as usize, want);
            assert_eq!((v.id(i), v.entry(i).attrs, v.payload(i)), (1000 + i as u32, i as u32, &[i as u8][..]));
            let mean = &means[want * dim..(want + 1) * dim];
            decode_residual_sq8(v.code8(i), v.scale8(i), mean, &mut back);
            // within half an int8 step of the residual's max-abs scale
            let maxabs = x.iter().zip(mean).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let err = x.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(err <= 0.5 * maxabs / 127.0 + 1e-6, "entry {i}: err {err}");
        }
        // bad shapes are errors
        assert!(DeltaEncoder::new(3, 1, vec![0.0; 3], 0).is_err());
        assert!(DeltaEncoder::new(4, 2, vec![0.0; 4], 0).is_err());
    }

    #[test]
    fn encoder_with_tree() {
        let (dim, k, n) = (32, 40, 150);
        let mut rng = Rng::new(17);
        let means: Vec<f32> = (0..k).flat_map(|c| rand_unit(&mut rng, dim).into_iter().map(move |v| v * (1.0 + c as f32 * 0.05))).collect();
        let xs: Vec<Vec<f32>> = (0..n).map(|_| rand_unit(&mut rng, dim)).collect();
        let tree = CentroidTree::build(&means, dim, k, 6, 5, 3);
        let inv = inv_norms(&means, dim);
        let run = |tree: Option<(CentroidTree, usize)>| {
            let mut enc = DeltaEncoder::new(dim, k, means.clone(), 1).unwrap();
            if let Some((t, b)) = tree {
                enc.set_tree(t, b).unwrap();
            }
            let cells: Vec<usize> = xs.iter().enumerate().map(|(i, x)| enc.add(x, i as u32, 0, &[])).collect();
            (cells, enc.finish())
        };
        let exact = run(None);
        // full beam: byte-identical segment
        assert_eq!(run(Some((tree.clone(), 6))), exact);
        // narrow beam: the tree's choice, encoded against that cell's mean
        let (cells, bytes) = run(Some((tree.clone(), 1)));
        let v = DeltaView::parse(&bytes).unwrap();
        for (i, x) in xs.iter().enumerate() {
            assert_eq!(cells[i], tree.nearest(x, &means, &inv, 1).0);
            assert_eq!(v.cell(i) as usize, cells[i]);
        }
        // mismatched tree
        let other = CentroidTree::build(&means[..10 * dim], dim, 10, 3, 2, 1);
        let mut enc = DeltaEncoder::new(dim, k, means.clone(), 1).unwrap();
        assert!(enc.set_tree(other, 2).is_err());
        assert!(enc.tree().is_none());
    }

    #[test]
    fn encoder_with_codebook_writes_pq_codes() {
        let (dim, k, m) = (16, 3, 4);
        let mut rng = Rng::new(9);
        let means: Vec<f32> = (0..k).flat_map(|_| rand_unit(&mut rng, dim)).collect();
        let res: Vec<f32> = (0..300 * dim).map(|_| rng.below(2001) as f32 / 10000.0 - 0.1).collect();
        let cb = Codebook::init_from_samples(&res, dim, m, 1).unwrap();
        let mut enc = DeltaEncoder::new(dim, k, means.clone(), 0).unwrap();
        enc.set_codebook(cb.clone()).unwrap();
        let x = rand_unit(&mut rng, dim);
        let cell = enc.add(&x, 1, 0, &[]);
        assert!(enc.set_codebook(cb.clone()).is_err());
        let v_bytes = enc.finish();
        let v = DeltaView::parse(&v_bytes).unwrap();
        assert_eq!(v.pq_m, m);
        let r: Vec<f32> = x.iter().zip(&means[cell * dim..(cell + 1) * dim]).map(|(a, b)| a - b).collect();
        let mut want = vec![0u8; m];
        crate::pq::encode(&r, &cb, &mut want);
        assert_eq!(v.pq_code(0), &want[..]);
    }
}
