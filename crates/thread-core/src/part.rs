// Dense-side file written before rustfmt was adopted; kept as is (`cargo fmt` skips it).
#![cfg_attr(rustfmt, rustfmt::skip)]
//! Build-time "parts": the intermediate output of an assign job.
//!
//! An assign job streams a batch of vectors and, for each coarse group `g`,
//! emits one part blob holding the records of every vector assigned to any
//! fine cell under `g`, grouped by fine cell. A finalize job for cell
//! `(g, j)` reads the directory of every part for `g`, then range-reads just
//! the records for `j` and concatenates them into a slab.
//!
//! Layout (little-endian):
//! ```text
//! header 16: magic "VIBP", k2 u32, dim u32, version u32 (= 2)
//! directory: k2 * { count u32, off u32, len u32 }   (off relative to part start)
//! records:   per fine cell, `count` * { id u32, scale4 f32, scale8 f32, attrs u32,
//!                                       payload_len u32, pq_m u16, flags u16,
//!                                       code4 u8*(dim/2), code8 i8*dim,
//!                                       pq u8*pq_m, payload bytes }
//! ```
//! `dim` must be even (two 4-bit codes per byte). `pq_m` is 0 when the assign
//! job had no PQ codebook (such records only build format 1/2 slabs).
//! `flags` (formerly reserved, 0 in assign-job parts): bit 0
//! [`RECORD_FLAG_NO_CODE4`] marks a record whose 4-bit codes are absent
//! (zeros), e.g. one re-emitted from a v3 or v1 slab by
//! `SlabView::to_part_records`; such records cannot build a format 2 slab.
//! Version 1 parts (20-byte record header, no PQ codes; header word 0) are
//! rejected by [`parse_directory`].

use crate::error::{need, rd_f32, rd_u32, FormatError};
use crate::pq::Codebook;
use crate::quant::{decode_residual_sq8, encode_residual_sq4, encode_residual_sq8};

pub const PART_MAGIC: &[u8; 4] = b"VIBP";
pub const PART_HEADER: usize = 16;
pub const DIR_ENTRY: usize = 12;
/// Part format version (header word at offset 12).
pub const PART_VERSION: u32 = 2;
/// Fixed bytes before the codes in each record.
pub const RECORD_HEADER: usize = 24;
/// Record flag: the record has no 4-bit codes (`code4` is zeros, `scale4` 0).
pub const RECORD_FLAG_NO_CODE4: u16 = 1;

/// Appends one part record (the layout `PartBuilder` writes per cell).
/// `code4.len() == dim/2`, `code8.len() == dim`, `pq.len() <= u16::MAX`.
#[allow(clippy::too_many_arguments)]
pub fn write_record(
    out: &mut Vec<u8>,
    id: u32,
    scale4: f32,
    scale8: f32,
    attrs: u32,
    flags: u16,
    code4: &[u8],
    code8: &[i8],
    pq: &[u8],
    payload: &[u8],
) {
    assert_eq!(code4.len() * 2, code8.len(), "code4 must hold dim/2 bytes");
    assert!(pq.len() <= u16::MAX as usize, "pq_m must fit in u16");
    out.reserve(record_len_pq(code8.len(), pq.len()) + payload.len());
    out.extend_from_slice(&id.to_le_bytes());
    out.extend_from_slice(&scale4.to_le_bytes());
    out.extend_from_slice(&scale8.to_le_bytes());
    out.extend_from_slice(&attrs.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&(pq.len() as u16).to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(code4);
    out.extend(code8.iter().map(|c| *c as u8));
    out.extend_from_slice(pq);
    out.extend_from_slice(payload);
}

/// Bytes of one record without PQ codes and with an empty payload.
pub const fn record_len(dim: usize) -> usize {
    RECORD_HEADER + dim / 2 + dim
}

/// Bytes of one record with `pq_m` PQ codes and an empty payload.
pub const fn record_len_pq(dim: usize, pq_m: usize) -> usize {
    record_len(dim) + pq_m
}

pub fn directory_len(k2: usize) -> usize {
    PART_HEADER + k2 * DIR_ENTRY
}

pub struct PartBuilder {
    dim: usize,
    k2: usize,
    cells: Vec<Vec<u8>>,
    counts: Vec<u32>,
}

impl PartBuilder {
    pub fn new(k2: usize, dim: usize) -> Self {
        assert!(dim % 2 == 0, "parts require an even dimension");
        PartBuilder { dim, k2, cells: (0..k2).map(|_| Vec::new()).collect(), counts: vec![0; k2] }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        j: usize,
        id: u32,
        scale4: f32,
        scale8: f32,
        attrs: u32,
        code4: &[u8],
        code8: &[i8],
        pq: &[u8],
        payload: &[u8],
    ) {
        assert_eq!(code4.len(), self.dim / 2);
        assert_eq!(code8.len(), self.dim);
        write_record(&mut self.cells[j], id, scale4, scale8, attrs, 0, code4, code8, pq, payload);
        self.counts[j] += 1;
    }

    pub fn total_records(&self) -> u32 {
        self.counts.iter().sum()
    }

    pub fn byte_len(&self) -> usize {
        directory_len(self.k2) + self.cells.iter().map(|c| c.len()).sum::<usize>()
    }

    /// Serializes and resets the builder so it can be reused.
    pub fn take(&mut self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_len());
        out.extend_from_slice(PART_MAGIC);
        out.extend_from_slice(&(self.k2 as u32).to_le_bytes());
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&PART_VERSION.to_le_bytes());
        let mut off = directory_len(self.k2);
        for j in 0..self.k2 {
            out.extend_from_slice(&self.counts[j].to_le_bytes());
            out.extend_from_slice(&(off as u32).to_le_bytes());
            out.extend_from_slice(&(self.cells[j].len() as u32).to_le_bytes());
            off += self.cells[j].len();
        }
        for j in 0..self.k2 {
            out.extend_from_slice(&self.cells[j]);
            self.cells[j].clear();
            self.counts[j] = 0;
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirEntry {
    pub count: u32,
    pub off: u32,
    pub len: u32,
}

/// Parses the directory prefix of a part (the first `directory_len(k2)` bytes
/// are enough; extra bytes are ignored).
pub fn parse_directory(buf: &[u8]) -> Result<(usize, usize, Vec<DirEntry>), FormatError> {
    need(buf, PART_HEADER)?;
    if &buf[0..4] != PART_MAGIC {
        return Err(FormatError::BadMagic);
    }
    let k2 = rd_u32(buf, 4) as usize;
    let dim = rd_u32(buf, 8) as usize;
    let version = rd_u32(buf, 12);
    if version != PART_VERSION {
        return Err(FormatError::BadVersion(version.min(u16::MAX as u32) as u16));
    }
    need(buf, directory_len(k2))?;
    let dir = (0..k2)
        .map(|j| {
            let o = PART_HEADER + j * DIR_ENTRY;
            DirEntry { count: rd_u32(buf, o), off: rd_u32(buf, o + 4), len: rd_u32(buf, o + 8) }
        })
        .collect();
    Ok((k2, dim, dir))
}

/// Byte span `(offset, length)` of a part covering the records of cells
/// `[first, first + count)` of `dir`. Parts write cells in increasing order,
/// so these records are one contiguous range; empty cells at either edge are
/// skipped. A range with no records gives `(0, 0)`. Errors if the range is out
/// of bounds or the non-empty cells are not laid out back to back.
pub fn span(dir: &[DirEntry], first: usize, count: usize) -> Result<(u32, u32), FormatError> {
    let end = first.checked_add(count).ok_or(FormatError::Inconsistent("cell range overflow"))?;
    if end > dir.len() {
        return Err(FormatError::Inconsistent("cell range exceeds the part directory"));
    }
    let cells = &dir[first..end];
    let Some(a) = cells.iter().position(|d| d.len > 0) else {
        return Ok((0, 0));
    };
    let b = cells.iter().rposition(|d| d.len > 0).unwrap();
    let mut pos = cells[a].off as u64;
    for d in &cells[a..=b] {
        if d.len > 0 {
            if d.off as u64 != pos {
                return Err(FormatError::Inconsistent("part cells are not contiguous"));
            }
            pos += d.len as u64;
        }
    }
    let start = cells[a].off;
    let len = u32::try_from(pos - start as u64).map_err(|_| FormatError::Inconsistent("span length overflow"))?;
    Ok((start, len))
}

/// Range within a fetched span (that starts at part offset `span_offset` and
/// is `span_len` bytes long) holding the cell range `[cell_off, cell_off + cell_len)`
/// given in part offsets. Errors if the cell range is not inside the span.
pub fn span_sub_range(
    span_len: usize,
    span_offset: u32,
    cell_off: u32,
    cell_len: u32,
) -> Result<core::ops::Range<usize>, FormatError> {
    if cell_len == 0 {
        return Ok(0..0);
    }
    let start = (cell_off as u64)
        .checked_sub(span_offset as u64)
        .ok_or(FormatError::Inconsistent("cell starts before the span"))? as usize;
    let end = start + cell_len as usize;
    if end > span_len {
        return Err(FormatError::TooShort { need: end, have: span_len });
    }
    Ok(start..end)
}

/// One decoded record borrowed from a records byte range.
pub struct Record<'a> {
    pub id: u32,
    pub scale4: f32,
    pub scale8: f32,
    pub attrs: u32,
    /// Record flags (see [`RECORD_FLAG_NO_CODE4`]); 0 in assign-job parts.
    pub flags: u16,
    pub code4: &'a [u8],
    pub code8: &'a [i8],
    /// PQ codes (`pq_m` bytes; empty when the record has none).
    pub pq: &'a [u8],
    pub payload: &'a [u8],
}

/// Iterates the records in a byte range produced by `PartBuilder` for one cell.
pub fn records<'a>(buf: &'a [u8], dim: usize) -> impl Iterator<Item = Result<Record<'a>, FormatError>> + 'a {
    let mut pos = 0usize;
    core::iter::from_fn(move || {
        if pos >= buf.len() {
            return None;
        }
        let fixed = record_len(dim);
        if buf.len() < pos + fixed {
            let need = pos + fixed;
            pos = buf.len();
            return Some(Err(FormatError::TooShort { need, have: buf.len() }));
        }
        let id = rd_u32(buf, pos);
        let scale4 = rd_f32(buf, pos + 4);
        let scale8 = rd_f32(buf, pos + 8);
        let attrs = rd_u32(buf, pos + 12);
        let plen = rd_u32(buf, pos + 16) as usize;
        let pq_m = crate::error::rd_u16(buf, pos + 20) as usize;
        let flags = crate::error::rd_u16(buf, pos + 22);
        let c4 = pos + RECORD_HEADER;
        let c8 = c4 + dim / 2;
        let code4 = &buf[c4..c8];
        let code8_u8 = &buf[c8..c8 + dim];
        // i8 and u8 have identical layout and alignment.
        let code8 = unsafe { core::slice::from_raw_parts(code8_u8.as_ptr() as *const i8, dim) };
        let pqs = pos + fixed;
        if buf.len() - pqs < pq_m {
            pos = buf.len();
            return Some(Err(FormatError::TooShort { need: pqs + pq_m, have: buf.len() }));
        }
        let pq = &buf[pqs..pqs + pq_m];
        let pstart = pqs + pq_m;
        if buf.len() - pstart < plen {
            pos = buf.len();
            return Some(Err(FormatError::TooShort { need: pstart.saturating_add(plen), have: buf.len() }));
        }
        let payload = &buf[pstart..pstart + plen];
        pos = pstart + plen;
        Some(Ok(Record { id, scale4, scale8, attrs, flags, code4, code8, pq, payload }))
    })
}

/// Encodes vectors as part records against a given mean, exactly as an assign
/// job does: 4-bit MSE-clipped codes, int8 codes, and PQ codes when a codebook
/// is set (flags 0, so every record can build a format 1, 2 or 3 slab). Holds
/// the scratch buffers, so encoding many records allocates nothing.
pub struct RecordEncoder<'c> {
    dim: usize,
    codebook: Option<&'c Codebook>,
    code4: Vec<u8>,
    code8: Vec<i8>,
    scratch: Vec<f32>,
    pq: Vec<u8>,
}

impl<'c> RecordEncoder<'c> {
    /// Errors if `dim` is odd or zero, or the codebook is for another dim.
    pub fn new(dim: usize, codebook: Option<&'c Codebook>) -> Result<Self, FormatError> {
        if dim == 0 || dim % 2 != 0 {
            return Err(FormatError::Inconsistent("part records require an even, non-zero dimension"));
        }
        if codebook.is_some_and(|cb| cb.dim != dim) {
            return Err(FormatError::Inconsistent("codebook dim != record dim"));
        }
        Ok(RecordEncoder {
            dim,
            codebook,
            code4: vec![0; dim / 2],
            code8: vec![0; dim],
            scratch: vec![0.0; dim],
            pq: vec![0; codebook.map_or(0, |cb| cb.m)],
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Appends the record of vector `x` (`dim` floats) encoded against `mean`.
    pub fn encode(&mut self, out: &mut Vec<u8>, x: &[f32], mean: &[f32], id: u32, attrs: u32, payload: &[u8]) {
        assert_eq!(x.len(), self.dim);
        assert_eq!(mean.len(), self.dim);
        let scale4 = encode_residual_sq4(x, mean, &mut self.code4, &mut self.scratch);
        let scale8 = encode_residual_sq8(x, mean, &mut self.code8);
        if let Some(cb) = self.codebook {
            for i in 0..self.dim {
                self.scratch[i] = x[i] - mean[i];
            }
            crate::pq::encode(&self.scratch, cb, &mut self.pq);
        }
        write_record(out, id, scale4, scale8, attrs, 0, &self.code4, &self.code8, &self.pq, payload);
    }
}

fn check_mean(mean: &[f32], dim: usize, what: &'static str) -> Result<(), FormatError> {
    if mean.len() != dim {
        return Err(FormatError::Inconsistent(what));
    }
    Ok(())
}

/// Re-encodes part records (e.g. a delta segment's records of one cell, from
/// `DeltaView::records_for_cell`) against a new mean: each record is decoded
/// from its int8 tier (`old_mean + scale8 * code8`, the most precise copy) and
/// encoded again against `new_mean` by a [`RecordEncoder`] (4-bit, int8 and,
/// with `codebook`, PQ codes). Ids, attrs and payloads are kept. `dim` is
/// `old_mean.len()`.
pub fn reencode_records(
    records_bytes: &[u8],
    old_mean: &[f32],
    new_mean: &[f32],
    codebook: Option<&Codebook>,
) -> Result<Vec<u8>, FormatError> {
    let dim = old_mean.len();
    check_mean(new_mean, dim, "new_mean length != old_mean length")?;
    let mut enc = RecordEncoder::new(dim, codebook)?;
    let mut x = vec![0f32; dim];
    let mut out = Vec::with_capacity(records_bytes.len());
    for r in records(records_bytes, dim) {
        let r = r?;
        decode_residual_sq8(r.code8, r.scale8, old_mean, &mut x);
        enc.encode(&mut out, &x, new_mean, r.id, r.attrs, r.payload);
    }
    Ok(out)
}

/// Splits part records between two new means (`means2` = `2 * dim`, e.g. from
/// `kmeans::split_cell`): each record is decoded from its int8 tier against
/// `old_mean`, goes to the mean with the higher cosine
/// ([`crate::kmeans::split_side`], the rule `split_cell` assigns with) and is
/// re-encoded against it. Returns `(records_a, records_b)`.
pub fn split_part_records(
    records_bytes: &[u8],
    old_mean: &[f32],
    means2: &[f32],
    codebook: Option<&Codebook>,
) -> Result<(Vec<u8>, Vec<u8>), FormatError> {
    let dim = old_mean.len();
    let mut sp = Splitter::new(dim, means2, codebook)?;
    let mut x = vec![0f32; dim];
    for r in records(records_bytes, dim) {
        let r = r?;
        decode_residual_sq8(r.code8, r.scale8, old_mean, &mut x);
        sp.push(&x, r.id, r.attrs, r.payload);
    }
    Ok(sp.finish())
}

/// Shared by [`split_part_records`] and `slab::split_records`: routes decoded
/// vectors to one of two means and re-encodes them there.
pub(crate) struct Splitter<'c> {
    enc: RecordEncoder<'c>,
    means2: Vec<f32>,
    inv: [f32; 2],
    out: [Vec<u8>; 2],
}

impl<'c> Splitter<'c> {
    pub(crate) fn new(dim: usize, means2: &[f32], codebook: Option<&'c Codebook>) -> Result<Self, FormatError> {
        check_mean(means2, 2 * dim, "means2 must hold 2 * dim floats")?;
        let enc = RecordEncoder::new(dim, codebook)?;
        let inv = crate::math::inv_norms(means2, dim);
        Ok(Splitter { enc, means2: means2.to_vec(), inv: [inv[0], inv[1]], out: [Vec::new(), Vec::new()] })
    }

    pub(crate) fn push(&mut self, x: &[f32], id: u32, attrs: u32, payload: &[u8]) {
        let dim = self.enc.dim();
        let side = crate::kmeans::split_side(x, &self.means2, &self.inv) as usize;
        let mean = &self.means2[side * dim..(side + 1) * dim];
        self.enc.encode(&mut self.out[side], x, mean, id, attrs, payload);
    }

    pub(crate) fn finish(self) -> (Vec<u8>, Vec<u8>) {
        let [a, b] = self.out;
        (a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_matches_directory() {
        let dim = 4;
        let mut p = PartBuilder::new(8, dim);
        // cells 0, 3 and 7 empty; varying payloads so records differ in length
        let cells = [1usize, 2, 2, 4, 5, 5, 5, 6, 1];
        for (i, &j) in cells.iter().enumerate() {
            let payload = vec![i as u8; i % 3];
            p.push(j, i as u32, 0.5, 0.05, 0, &[0x12, 0x34], &[1, 2, 3, 4], &[], &payload);
        }
        let bytes = p.take();
        let (_, _, dir) = parse_directory(&bytes).unwrap();
        for first in 0..8 {
            for count in 0..=(8 - first) {
                let (off, len) = span(&dir, first, count).unwrap();
                let nonempty: Vec<&DirEntry> = dir[first..first + count].iter().filter(|d| d.len > 0).collect();
                if nonempty.is_empty() {
                    assert_eq!((off, len), (0, 0));
                    continue;
                }
                assert_eq!(off, nonempty[0].off);
                let last = nonempty[nonempty.len() - 1];
                assert_eq!(off + len, last.off + last.len);
                // the span is exactly the concatenation of the per-cell ranges
                let sp = &bytes[off as usize..(off + len) as usize];
                let mut cat = Vec::new();
                for d in &dir[first..first + count] {
                    let r = span_sub_range(sp.len(), off, d.off, d.len).unwrap();
                    assert_eq!(&sp[r.clone()], &bytes[d.off as usize..(d.off + d.len) as usize]);
                    cat.extend_from_slice(&sp[r]);
                }
                assert_eq!(cat, sp);
                let n = records(sp, dim).map(|r| r.unwrap()).count();
                assert_eq!(n as u32, dir[first..first + count].iter().map(|d| d.count).sum::<u32>());
            }
        }
        assert!(span(&dir, 5, 4).is_err());
        assert!(span(&dir, usize::MAX, 2).is_err());
        // a cell outside the span
        let (off, len) = span(&dir, 4, 2).unwrap();
        assert!(span_sub_range(len as usize, off, dir[2].off, dir[2].len).is_err());
        assert!(span_sub_range(len as usize, off, dir[6].off, dir[6].len).is_err());
        // a gap between non-empty cells is rejected
        let mut bad = dir.clone();
        bad[4].off += 1;
        assert!(span(&bad, 2, 4).is_err());
    }

    #[test]
    fn part_roundtrip() {
        let mut p = PartBuilder::new(3, 2);
        p.push(2, 10, 0.5, 0.05, 0, &[0x1F], &[1, 2], &[], b"ab");
        p.push(0, 11, 0.25, 0.025, 7, &[0x00], &[3, 4], &[], b"");
        p.push(2, 12, 0.125, 0.0125, 0, &[0x9A], &[5, -6], &[], b"c");
        assert_eq!(p.byte_len(), directory_len(3) + 3 * record_len(2) + 3);
        let bytes = p.take();
        assert_eq!(bytes.len(), directory_len(3) + 3 * record_len(2) + 3);
        let (k2, dim, dir) = parse_directory(&bytes).unwrap();
        assert_eq!((k2, dim), (3, 2));
        assert_eq!(dir[0].count, 1);
        assert_eq!(dir[1].count, 0);
        assert_eq!(dir[2].count, 2);
        let r2 = &bytes[dir[2].off as usize..(dir[2].off + dir[2].len) as usize];
        let recs: Vec<_> = records(r2, dim).map(|r| r.unwrap()).collect();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].id, 10);
        assert_eq!(recs[0].payload, b"ab");
        assert_eq!((recs[0].scale4, recs[0].scale8), (0.5, 0.05));
        assert_eq!(recs[0].code4, &[0x1F]);
        assert_eq!(recs[1].id, 12);
        assert_eq!(recs[1].code4, &[0x9A]);
        assert_eq!(recs[1].code8, &[5, -6]);
        assert_eq!(recs[1].payload, b"c");
        let r0 = &bytes[dir[0].off as usize..(dir[0].off + dir[0].len) as usize];
        let rec0: Vec<_> = records(r0, dim).map(|r| r.unwrap()).collect();
        assert_eq!((rec0[0].id, rec0[0].attrs, rec0[0].scale8), (11, 7, 0.025));
        assert!(rec0[0].payload.is_empty());
        // truncated range is an error, not a panic
        assert!(records(&r2[..r2.len() - 1], dim).any(|r| r.is_err()));
        // builder was reset
        assert_eq!(p.total_records(), 0);
        assert!(rec0[0].pq.is_empty());
        // a version 1 part (header word 0) is rejected
        let mut old = bytes.clone();
        old[12..16].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(parse_directory(&old).map(|_| ()), Err(FormatError::BadVersion(0)));
    }

    #[test]
    fn part_roundtrip_with_pq_codes() {
        let dim = 16;
        for pq_m in [0usize, 192] {
            let mut p = PartBuilder::new(2, dim);
            let pq_a: Vec<u8> = (0..pq_m).map(|i| (i * 7 % 256) as u8).collect();
            let pq_b: Vec<u8> = (0..pq_m).map(|i| (255 - i % 256) as u8).collect();
            let c4: Vec<u8> = (0..dim / 2).map(|i| i as u8).collect();
            let c8: Vec<i8> = (0..dim).map(|i| i as i8 - 8).collect();
            p.push(1, 5, 0.5, 0.05, 3, &c4, &c8, &pq_a, b"payload");
            p.push(1, 6, 0.25, 0.025, 4, &c4, &c8, &pq_b, b"");
            p.push(0, 7, 0.125, 0.0125, 5, &c4, &c8, &pq_a, b"z");
            let want = directory_len(2) + 3 * record_len_pq(dim, pq_m) + 8;
            assert_eq!(p.byte_len(), want);
            let bytes = p.take();
            assert_eq!(bytes.len(), want);
            let (_, _, dir) = parse_directory(&bytes).unwrap();
            let r1 = &bytes[dir[1].off as usize..(dir[1].off + dir[1].len) as usize];
            let recs: Vec<_> = records(r1, dim).map(|r| r.unwrap()).collect();
            assert_eq!(recs.len(), 2);
            assert_eq!((recs[0].id, recs[0].attrs, recs[0].scale4, recs[0].scale8), (5, 3, 0.5, 0.05));
            assert_eq!(recs[0].code4, &c4[..]);
            assert_eq!(recs[0].code8, &c8[..]);
            assert_eq!(recs[0].pq, &pq_a[..]);
            assert_eq!(recs[0].payload, b"payload");
            assert_eq!(recs[1].pq, &pq_b[..]);
            assert_eq!(recs[1].payload, b"");
            let (_, len) = span(&dir, 0, 2).unwrap();
            assert_eq!(len as usize, want - directory_len(2));
            // truncation inside the pq codes or the payload is an error
            assert!(records(&r1[..record_len(dim) + pq_m / 2 + 1], dim).any(|r| r.is_err()));
            assert!(records(&r1[..r1.len() - 1], dim).any(|r| r.is_err()));
        }
    }
}
