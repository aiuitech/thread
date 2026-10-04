// Dense-side file written before rustfmt was adopted; kept as is (`cargo fmt` skips it).
#![cfg_attr(rustfmt, rustfmt::skip)]
//! The slab: one immutable blob per fine cell.
//!
//! Version 2 (written by [`SlabBuilder`]; little-endian, offsets relative to
//! the slab start, every section 8-byte aligned, in this order):
//! ```text
//! header (64 bytes)
//!   0  magic        "VIBS"
//!   4  version      u16 = 2
//!   6  code_type    u8  = 2 (4-bit scan tier + int8 rerank tier)
//!   7  flags        u8
//!   8  dim          u32   (even)
//!   12 count        u32
//!   16 cell         u32   global fine cell id
//!   20 centroid_off u32   dim * f32 (the cell mean; residual base)
//!   24 entries_off  u32   count * 16
//!   28 codes4_off   u32   count * dim/2: signed 4-bit residual codes, low nibble = even dim
//!   32 scales4_off  u32   count * f32
//!   36 rerank_off   u32   count * (4 + dim): [scale8 f32][code8 i8 * dim]
//!   40 payload_off  u32
//!   44 payload_len  u32
//!   48 scan_len     u32   = rerank_off: one range read of [0, scan_len) is enough to scan
//!   52..64 reserved
//! entry (16 bytes): id u32, payload_off u32 (relative to payload section),
//!                   payload_len u32, attrs u32
//! ```
//! Rerank record `i` is the range `[rerank_off + i*(4+dim), +4+dim)`.
//!
//! Version 3 (written by [`SlabBuilderV3`]): PQ scan tier + the v2 int8 rerank tier.
//! ```text
//!   4  version u16 = 3, 6 code_type u8 = 3 (residual PQ + residual SQ8)
//!   20 centroid_off u32   dim * f32
//!   24 entries_off  u32   count * 16 (as v2)
//!   28 pq_off       u32   count * pq_m: one u8 code per subspace (see `pq`)
//!   32 pq_m         u32   subspaces (dim % pq_m == 0; 0 only in an empty slab)
//!   36 rerank_off   u32   count * (4 + dim), as v2
//!   40 payload_off  u32   44 payload_len u32
//!   48 scan_len     u32   = rerank_off
//!   52..64 reserved
//! ```
//! Scanning a v3 slab needs the index's PQ codebook (as a per-query
//! `pq::QueryTable`); see `search::scan_slab_pq`.
//!
//! Version 1 (legacy, still parsed; written by [`SlabBuilderV1`]):
//! ```text
//!   4 version u16 = 1, 6 code_type u8 = 1 (residual SQ8)
//!   28 codes_off  (count * dim i8)   32 scales_off (count * f32)
//!   36 payload_off  40 payload_len   44..64 reserved
//! ```
//! The centroid is stored inside the slab so a slab is self-describing: a
//! scanner needs only the slab bytes and the query.

use crate::error::{need, rd_f32, rd_u16, rd_u32, FormatError};

pub const SLAB_MAGIC: &[u8; 4] = b"VIBS";
pub const SLAB_VERSION: u16 = 2;
pub const SLAB_VERSION_V1: u16 = 1;
pub const SLAB_VERSION_V3: u16 = 3;
pub const SLAB_HEADER: usize = 64;
pub const ENTRY_BYTES: usize = 16;
/// v1 code type: residual SQ8, per-vector scale.
pub const CODE_SQ8_RESIDUAL: u8 = 1;
/// v2 code type: residual SQ4 scan tier + residual SQ8 rerank tier.
pub const CODE_SQ4_SQ8_RESIDUAL: u8 = 2;
/// v3 code type: residual PQ scan tier + residual SQ8 rerank tier.
pub const CODE_PQ_SQ8_RESIDUAL: u8 = 3;

/// Bytes per rerank record for `dim`: `[scale8 f32][code8 i8 * dim]`.
#[inline]
pub const fn rerank_record_len(dim: usize) -> usize {
    4 + dim
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub id: u32,
    pub payload_off: u32,
    pub payload_len: u32,
    pub attrs: u32,
}

/// Section offsets of a v2 slab with `n` entries.
struct LayoutV2 {
    centroid_off: usize,
    entries_off: usize,
    codes4_off: usize,
    scales4_off: usize,
    rerank_off: usize,
    payload_off: usize,
}

impl LayoutV2 {
    fn new(dim: usize, n: usize) -> Self {
        let centroid_off = SLAB_HEADER;
        let entries_off = align8(centroid_off + dim * 4);
        let codes4_off = align8(entries_off + n * ENTRY_BYTES);
        let scales4_off = align8(codes4_off + n * (dim / 2));
        let rerank_off = align8(scales4_off + n * 4);
        let payload_off = align8(rerank_off + n * rerank_record_len(dim));
        LayoutV2 { centroid_off, entries_off, codes4_off, scales4_off, rerank_off, payload_off }
    }
}

/// Incrementally assembles a v2 slab in memory.
pub struct SlabBuilder {
    dim: usize,
    cell: u32,
    centroid: Vec<f32>,
    entries: Vec<Entry>,
    codes4: Vec<u8>,
    scales4: Vec<f32>,
    rerank: Vec<u8>,
    payload: Vec<u8>,
}

impl SlabBuilder {
    pub fn new(dim: usize, cell: u32, centroid: &[f32]) -> Self {
        assert_eq!(centroid.len(), dim);
        assert!(dim % 2 == 0, "v2 slabs require an even dimension");
        SlabBuilder {
            dim,
            cell,
            centroid: centroid.to_vec(),
            entries: Vec::new(),
            codes4: Vec::new(),
            scales4: Vec::new(),
            rerank: Vec::new(),
            payload: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Appends one entry: `code4` is `dim/2` packed nibbles with scale `scale4`,
    /// `code8` the `dim` int8 rerank codes with scale `scale8`.
    #[allow(clippy::too_many_arguments)]
    pub fn push(&mut self, id: u32, code4: &[u8], scale4: f32, code8: &[i8], scale8: f32, payload: &[u8], attrs: u32) {
        assert_eq!(code4.len(), self.dim / 2);
        assert_eq!(code8.len(), self.dim);
        let payload_off = self.payload.len() as u32;
        self.payload.extend_from_slice(payload);
        self.entries.push(Entry { id, payload_off, payload_len: payload.len() as u32, attrs });
        self.codes4.extend_from_slice(code4);
        self.scales4.push(scale4);
        self.rerank.extend_from_slice(&scale8.to_le_bytes());
        self.rerank.extend(code8.iter().map(|c| *c as u8));
    }

    /// Serialized size the slab will have.
    pub fn byte_len(&self) -> usize {
        LayoutV2::new(self.dim, self.entries.len()).payload_off + self.payload.len()
    }

    /// Bytes a scanner needs (`[0, scan_len)`).
    pub fn scan_len(&self) -> usize {
        LayoutV2::new(self.dim, self.entries.len()).rerank_off
    }

    pub fn finish(self) -> Vec<u8> {
        let n = self.entries.len();
        let l = LayoutV2::new(self.dim, n);
        let total = l.payload_off + self.payload.len();

        let mut out = vec![0u8; total];
        out[0..4].copy_from_slice(SLAB_MAGIC);
        out[4..6].copy_from_slice(&SLAB_VERSION.to_le_bytes());
        out[6] = CODE_SQ4_SQ8_RESIDUAL;
        out[7] = 0;
        put_u32(&mut out, 8, self.dim as u32);
        put_u32(&mut out, 12, n as u32);
        put_u32(&mut out, 16, self.cell);
        put_u32(&mut out, 20, l.centroid_off as u32);
        put_u32(&mut out, 24, l.entries_off as u32);
        put_u32(&mut out, 28, l.codes4_off as u32);
        put_u32(&mut out, 32, l.scales4_off as u32);
        put_u32(&mut out, 36, l.rerank_off as u32);
        put_u32(&mut out, 40, l.payload_off as u32);
        put_u32(&mut out, 44, self.payload.len() as u32);
        put_u32(&mut out, 48, l.rerank_off as u32);

        put_f32s(&mut out, l.centroid_off, &self.centroid);
        put_entries(&mut out, l.entries_off, &self.entries);
        out[l.codes4_off..l.codes4_off + self.codes4.len()].copy_from_slice(&self.codes4);
        put_f32s(&mut out, l.scales4_off, &self.scales4);
        out[l.rerank_off..l.rerank_off + self.rerank.len()].copy_from_slice(&self.rerank);
        out[l.payload_off..].copy_from_slice(&self.payload);
        out
    }
}

/// Legacy v1 (int8-only) slab writer. Kept for compatibility tests and
/// benchmarks; production builds write v2 via [`SlabBuilder`].
pub struct SlabBuilderV1 {
    dim: usize,
    cell: u32,
    centroid: Vec<f32>,
    entries: Vec<Entry>,
    codes: Vec<i8>,
    scales: Vec<f32>,
    payload: Vec<u8>,
}

impl SlabBuilderV1 {
    pub fn new(dim: usize, cell: u32, centroid: &[f32]) -> Self {
        assert_eq!(centroid.len(), dim);
        SlabBuilderV1 {
            dim,
            cell,
            centroid: centroid.to_vec(),
            entries: Vec::new(),
            codes: Vec::new(),
            scales: Vec::new(),
            payload: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn push(&mut self, id: u32, code: &[i8], scale: f32, payload: &[u8], attrs: u32) {
        assert_eq!(code.len(), self.dim);
        let payload_off = self.payload.len() as u32;
        self.payload.extend_from_slice(payload);
        self.entries.push(Entry { id, payload_off, payload_len: payload.len() as u32, attrs });
        self.codes.extend_from_slice(code);
        self.scales.push(scale);
    }

    pub fn byte_len(&self) -> usize {
        let n = self.entries.len();
        align8(SLAB_HEADER + self.dim * 4)
            + align8(n * ENTRY_BYTES)
            + align8(n * self.dim)
            + align8(n * 4)
            + self.payload.len()
    }

    pub fn finish(self) -> Vec<u8> {
        let n = self.entries.len();
        let centroid_off = SLAB_HEADER;
        let entries_off = align8(centroid_off + self.dim * 4);
        let codes_off = align8(entries_off + n * ENTRY_BYTES);
        let scales_off = align8(codes_off + n * self.dim);
        let payload_off = align8(scales_off + n * 4);
        let total = payload_off + self.payload.len();

        let mut out = vec![0u8; total];
        out[0..4].copy_from_slice(SLAB_MAGIC);
        out[4..6].copy_from_slice(&SLAB_VERSION_V1.to_le_bytes());
        out[6] = CODE_SQ8_RESIDUAL;
        out[7] = 0;
        put_u32(&mut out, 8, self.dim as u32);
        put_u32(&mut out, 12, n as u32);
        put_u32(&mut out, 16, self.cell);
        put_u32(&mut out, 20, centroid_off as u32);
        put_u32(&mut out, 24, entries_off as u32);
        put_u32(&mut out, 28, codes_off as u32);
        put_u32(&mut out, 32, scales_off as u32);
        put_u32(&mut out, 36, payload_off as u32);
        put_u32(&mut out, 40, self.payload.len() as u32);

        put_f32s(&mut out, centroid_off, &self.centroid);
        put_entries(&mut out, entries_off, &self.entries);
        for (i, c) in self.codes.iter().enumerate() {
            out[codes_off + i] = *c as u8;
        }
        put_f32s(&mut out, scales_off, &self.scales);
        out[payload_off..].copy_from_slice(&self.payload);
        out
    }
}

/// Section offsets of a v3 slab with `n` entries of `pq_m` codes.
struct LayoutV3 {
    centroid_off: usize,
    entries_off: usize,
    pq_off: usize,
    rerank_off: usize,
    payload_off: usize,
}

impl LayoutV3 {
    fn new(dim: usize, pq_m: usize, n: usize) -> Self {
        let centroid_off = SLAB_HEADER;
        let entries_off = align8(centroid_off + dim * 4);
        let pq_off = align8(entries_off + n * ENTRY_BYTES);
        let rerank_off = align8(pq_off + n * pq_m);
        let payload_off = align8(rerank_off + n * rerank_record_len(dim));
        LayoutV3 { centroid_off, entries_off, pq_off, rerank_off, payload_off }
    }
}

/// Incrementally assembles a v3 slab (PQ scan tier + int8 rerank tier).
pub struct SlabBuilderV3 {
    dim: usize,
    cell: u32,
    centroid: Vec<f32>,
    /// 0 until the first entry fixes it (when built with `pq_m = 0`).
    pq_m: usize,
    entries: Vec<Entry>,
    pq: Vec<u8>,
    rerank: Vec<u8>,
    payload: Vec<u8>,
}

impl SlabBuilderV3 {
    /// `pq_m = 0` takes the code count from the first pushed entry.
    pub fn new(dim: usize, cell: u32, centroid: &[f32], pq_m: usize) -> Self {
        assert_eq!(centroid.len(), dim);
        SlabBuilderV3 {
            dim,
            cell,
            centroid: centroid.to_vec(),
            pq_m,
            entries: Vec::new(),
            pq: Vec::new(),
            rerank: Vec::new(),
            payload: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn pq_m(&self) -> usize {
        self.pq_m
    }

    /// Appends one entry: `pq` its PQ codes, `code8`/`scale8` its rerank record.
    /// Errors (nothing added) if `pq` is empty, does not divide `dim`, or differs
    /// in length from earlier entries.
    pub fn push(&mut self, id: u32, pq: &[u8], code8: &[i8], scale8: f32, payload: &[u8], attrs: u32) -> Result<(), FormatError> {
        if pq.is_empty() {
            return Err(FormatError::Inconsistent("v3 slab entry without PQ codes (assign job had no codebook)"));
        }
        if self.pq_m == 0 {
            if self.dim % pq.len() != 0 {
                return Err(FormatError::Inconsistent("pq_m does not divide dim"));
            }
            self.pq_m = pq.len();
        } else if pq.len() != self.pq_m {
            return Err(FormatError::Inconsistent("entry pq_m differs from the slab's pq_m"));
        }
        if code8.len() != self.dim {
            return Err(FormatError::Inconsistent("code8 length != dim"));
        }
        let payload_off = self.payload.len() as u32;
        self.payload.extend_from_slice(payload);
        self.entries.push(Entry { id, payload_off, payload_len: payload.len() as u32, attrs });
        self.pq.extend_from_slice(pq);
        self.rerank.extend_from_slice(&scale8.to_le_bytes());
        self.rerank.extend(code8.iter().map(|c| *c as u8));
        Ok(())
    }

    pub fn byte_len(&self) -> usize {
        LayoutV3::new(self.dim, self.pq_m, self.entries.len()).payload_off + self.payload.len()
    }

    pub fn scan_len(&self) -> usize {
        LayoutV3::new(self.dim, self.pq_m, self.entries.len()).rerank_off
    }

    pub fn finish(self) -> Vec<u8> {
        let n = self.entries.len();
        let l = LayoutV3::new(self.dim, self.pq_m, n);
        let total = l.payload_off + self.payload.len();
        let mut out = vec![0u8; total];
        out[0..4].copy_from_slice(SLAB_MAGIC);
        out[4..6].copy_from_slice(&SLAB_VERSION_V3.to_le_bytes());
        out[6] = CODE_PQ_SQ8_RESIDUAL;
        out[7] = 0;
        put_u32(&mut out, 8, self.dim as u32);
        put_u32(&mut out, 12, n as u32);
        put_u32(&mut out, 16, self.cell);
        put_u32(&mut out, 20, l.centroid_off as u32);
        put_u32(&mut out, 24, l.entries_off as u32);
        put_u32(&mut out, 28, l.pq_off as u32);
        put_u32(&mut out, 32, self.pq_m as u32);
        put_u32(&mut out, 36, l.rerank_off as u32);
        put_u32(&mut out, 40, l.payload_off as u32);
        put_u32(&mut out, 44, self.payload.len() as u32);
        put_u32(&mut out, 48, l.rerank_off as u32);
        put_f32s(&mut out, l.centroid_off, &self.centroid);
        put_entries(&mut out, l.entries_off, &self.entries);
        out[l.pq_off..l.pq_off + self.pq.len()].copy_from_slice(&self.pq);
        out[l.rerank_off..l.rerank_off + self.rerank.len()].copy_from_slice(&self.rerank);
        out[l.payload_off..].copy_from_slice(&self.payload);
        out
    }
}

/// Slab writer for an index-time code tier choice: format 2 (v2: 4-bit scan
/// tier + int8 rerank tier), format 1 (v1: int8 only, built from the
/// records' `code8`/`scale8`; their 4-bit codes are dropped) or format 3 (v3:
/// the records' PQ codes as scan tier + int8 rerank tier; records must carry
/// PQ codes). Formats 1 and 2 ignore PQ codes.
pub enum SlabWriter {
    V1(SlabBuilderV1),
    V2(SlabBuilder),
    V3(SlabBuilderV3),
}

impl SlabWriter {
    pub fn new(format: u8, dim: usize, cell: u32, centroid: &[f32]) -> Result<Self, FormatError> {
        if centroid.len() != dim {
            return Err(FormatError::Inconsistent("centroid size != dim"));
        }
        match format {
            1 => Ok(SlabWriter::V1(SlabBuilderV1::new(dim, cell, centroid))),
            2 => {
                if dim % 2 != 0 {
                    return Err(FormatError::Inconsistent("v2 slabs require an even dimension"));
                }
                Ok(SlabWriter::V2(SlabBuilder::new(dim, cell, centroid)))
            }
            3 => Ok(SlabWriter::V3(SlabBuilderV3::new(dim, cell, centroid, 0))),
            v => Err(FormatError::BadVersion(v as u16)),
        }
    }

    pub fn version(&self) -> u16 {
        match self {
            SlabWriter::V1(_) => SLAB_VERSION_V1,
            SlabWriter::V2(_) => SLAB_VERSION,
            SlabWriter::V3(_) => SLAB_VERSION_V3,
        }
    }

    /// Adds one part record. Errors (nothing added) for format 3 on a record
    /// without PQ codes or with a different `pq_m` than earlier records, and for
    /// format 2 on a record flagged [`crate::part::RECORD_FLAG_NO_CODE4`] (one
    /// re-emitted from a v3 or v1 slab: it has no 4-bit codes to scan with, and
    /// silently writing zero codes would make the entry unfindable). Format 2
    /// ignores PQ codes, so records with any `pq_m` (0 or e.g. 192) are accepted.
    pub fn push_record(&mut self, r: &crate::part::Record<'_>) -> Result<(), FormatError> {
        match self {
            SlabWriter::V1(b) => b.push(r.id, r.code8, r.scale8, r.payload, r.attrs),
            SlabWriter::V2(b) => {
                if r.flags & crate::part::RECORD_FLAG_NO_CODE4 != 0 {
                    return Err(FormatError::Inconsistent(
                        "record has no 4-bit codes (re-emitted from a v3/v1 slab): cannot build a format 2 slab",
                    ));
                }
                b.push(r.id, r.code4, r.scale4, r.code8, r.scale8, r.payload, r.attrs)
            }
            SlabWriter::V3(b) => return b.push(r.id, r.pq, r.code8, r.scale8, r.payload, r.attrs),
        }
        Ok(())
    }

    /// Adds every record of a part records byte range (one or more whole
    /// cells). Returns the number added. On a malformed range, the records
    /// before the bad one have already been added.
    pub fn add_part_records(&mut self, bytes: &[u8], dim: usize) -> Result<u32, FormatError> {
        let mut n = 0;
        for r in crate::part::records(bytes, dim) {
            self.push_record(&r?)?;
            n += 1;
        }
        Ok(n)
    }

    pub fn len(&self) -> usize {
        match self {
            SlabWriter::V1(b) => b.len(),
            SlabWriter::V2(b) => b.len(),
            SlabWriter::V3(b) => b.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn byte_len(&self) -> usize {
        match self {
            SlabWriter::V1(b) => b.byte_len(),
            SlabWriter::V2(b) => b.byte_len(),
            SlabWriter::V3(b) => b.byte_len(),
        }
    }

    pub fn finish(self) -> Vec<u8> {
        match self {
            SlabWriter::V1(b) => b.finish(),
            SlabWriter::V2(b) => b.finish(),
            SlabWriter::V3(b) => b.finish(),
        }
    }
}

/// Header fields common to all versions, readable from the first 64 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabHeader {
    pub version: u16,
    pub code_type: u8,
    pub dim: usize,
    pub count: usize,
    pub cell: u32,
    /// Bytes needed to scan: v2/v3 `scan_len`; v1 `payload_off` (codes + scales end there).
    pub scan_len: usize,
    /// v2/v3 rerank section offset; 0 for v1.
    pub rerank_off: usize,
    /// v2/v3 `4 + dim`; 0 for v1.
    pub rerank_record_len: usize,
    /// v3 PQ codes per entry; 0 for v1/v2.
    pub pq_m: usize,
    pub payload_off: usize,
    pub payload_len: usize,
}

impl SlabHeader {
    /// Parses and sanity-checks the 64-byte header (extra bytes are ignored).
    pub fn parse(buf: &[u8]) -> Result<Self, FormatError> {
        need(buf, SLAB_HEADER)?;
        if &buf[0..4] != SLAB_MAGIC {
            return Err(FormatError::BadMagic);
        }
        let version = rd_u16(buf, 4);
        let code_type = buf[6];
        let dim = rd_u32(buf, 8) as usize;
        let count = rd_u32(buf, 12) as usize;
        let cell = rd_u32(buf, 16);
        match (version, code_type) {
            (SLAB_VERSION_V1, CODE_SQ8_RESIDUAL) => Ok(SlabHeader {
                version,
                code_type,
                dim,
                count,
                cell,
                scan_len: rd_u32(buf, 36) as usize,
                rerank_off: 0,
                rerank_record_len: 0,
                pq_m: 0,
                payload_off: rd_u32(buf, 36) as usize,
                payload_len: rd_u32(buf, 40) as usize,
            }),
            (SLAB_VERSION, CODE_SQ4_SQ8_RESIDUAL) => {
                if dim % 2 != 0 {
                    return Err(FormatError::Inconsistent("odd dim in v2 slab"));
                }
                let rerank_off = rd_u32(buf, 36) as usize;
                let scan_len = rd_u32(buf, 48) as usize;
                if scan_len != rerank_off {
                    return Err(FormatError::Inconsistent("scan_len != rerank_off"));
                }
                Ok(SlabHeader {
                    version,
                    code_type,
                    dim,
                    count,
                    cell,
                    scan_len,
                    rerank_off,
                    rerank_record_len: rerank_record_len(dim),
                    pq_m: 0,
                    payload_off: rd_u32(buf, 40) as usize,
                    payload_len: rd_u32(buf, 44) as usize,
                })
            }
            (SLAB_VERSION_V3, CODE_PQ_SQ8_RESIDUAL) => {
                let pq_m = rd_u32(buf, 32) as usize;
                if pq_m == 0 && count > 0 {
                    return Err(FormatError::Inconsistent("v3 slab with entries but pq_m = 0"));
                }
                if pq_m > 0 && dim % pq_m != 0 {
                    return Err(FormatError::Inconsistent("v3 pq_m does not divide dim"));
                }
                let rerank_off = rd_u32(buf, 36) as usize;
                let scan_len = rd_u32(buf, 48) as usize;
                if scan_len != rerank_off {
                    return Err(FormatError::Inconsistent("scan_len != rerank_off"));
                }
                Ok(SlabHeader {
                    version,
                    code_type,
                    dim,
                    count,
                    cell,
                    scan_len,
                    rerank_off,
                    rerank_record_len: rerank_record_len(dim),
                    pq_m,
                    payload_off: rd_u32(buf, 40) as usize,
                    payload_len: rd_u32(buf, 44) as usize,
                })
            }
            (SLAB_VERSION_V1, _) | (SLAB_VERSION, _) | (SLAB_VERSION_V3, _) => {
                Err(FormatError::Inconsistent("unknown code type"))
            }
            (v, _) => Err(FormatError::BadVersion(v)),
        }
    }

    /// Full serialized size of the slab object.
    pub fn total_len(&self) -> usize {
        self.payload_off + self.payload_len
    }
}

/// Zero-copy view over a serialized slab (v1, or v2/v3 — possibly only its
/// `[0, scan_len)` prefix).
pub struct SlabView<'a> {
    buf: &'a [u8],
    pub header: SlabHeader,
    pub dim: usize,
    pub count: usize,
    pub cell: u32,
    centroid_off: usize,
    entries_off: usize,
    /// v1: int8 codes. v2: packed 4-bit codes. v3: PQ codes.
    codes_off: usize,
    /// v1: int8 scales. v2: 4-bit scales. v3: unused (0).
    scales_off: usize,
    /// v3: PQ codes per entry; 0 otherwise.
    pq_m: usize,
    has_rerank: bool,
    has_payload: bool,
}

impl<'a> SlabView<'a> {
    /// Parses a v1 slab (must be complete up to `payload_off`) or a v2/v3 slab
    /// or scan prefix (must cover `[0, scan_len)`; the rerank and payload
    /// sections are validated and exposed only when the buffer reaches them).
    pub fn parse(buf: &'a [u8]) -> Result<Self, FormatError> {
        let h = SlabHeader::parse(buf)?;
        let (dim, count) = (h.dim, h.count);
        let end = |off: usize, len: Option<usize>| {
            len.and_then(|l| off.checked_add(l)).ok_or(FormatError::Inconsistent("offset overflow"))
        };
        let centroid_off = rd_u32(buf, 20) as usize;
        let entries_off = rd_u32(buf, 24) as usize;
        let codes_off = rd_u32(buf, 28) as usize;
        let v3 = h.version == SLAB_VERSION_V3;
        let scales_off = if v3 { 0 } else { rd_u32(buf, 32) as usize };
        let code_bytes = match h.version {
            SLAB_VERSION => dim / 2,
            SLAB_VERSION_V3 => h.pq_m,
            _ => dim,
        };
        let scales_end = if v3 { 0 } else { end(scales_off, count.checked_mul(4))? };
        let scan_sections = [
            end(centroid_off, dim.checked_mul(4))?,
            end(entries_off, count.checked_mul(ENTRY_BYTES))?,
            end(codes_off, count.checked_mul(code_bytes))?,
            scales_end,
        ];
        let scan_end = scan_sections.iter().copied().max().unwrap();
        if scan_end > h.scan_len {
            return Err(FormatError::Inconsistent("scan sections extend past scan_len"));
        }
        // the host reads exactly [0, scan_len); require all of it (incl. padding).
        need(buf, h.scan_len)?;

        let (has_rerank, has_payload) = if h.rerank_record_len > 0 {
            let rerank_end = end(h.rerank_off, count.checked_mul(h.rerank_record_len))?;
            let payload_end = end(h.payload_off, Some(h.payload_len))?;
            if rerank_end > h.payload_off {
                return Err(FormatError::Inconsistent("rerank section overlaps payload"));
            }
            (buf.len() >= rerank_end, buf.len() >= payload_end)
        } else {
            need(buf, end(h.payload_off, Some(h.payload_len))?)?;
            (false, true)
        };

        Ok(SlabView {
            buf,
            header: h,
            dim,
            count,
            cell: h.cell,
            centroid_off,
            entries_off,
            codes_off,
            scales_off,
            pq_m: h.pq_m,
            has_rerank,
            has_payload,
        })
    }

    pub fn version(&self) -> u16 {
        self.header.version
    }

    pub fn is_v2(&self) -> bool {
        self.header.version == SLAB_VERSION
    }

    pub fn is_v3(&self) -> bool {
        self.header.version == SLAB_VERSION_V3
    }

    /// v2/v3: the slab has an int8 rerank section (v1 keeps int8 codes in its scan sections).
    pub fn has_rerank_tier(&self) -> bool {
        self.header.rerank_record_len > 0
    }

    /// v3: PQ codes per entry (0 for v1/v2).
    pub fn pq_m(&self) -> usize {
        self.pq_m
    }

    /// v3: PQ codes of entry `i` (`pq_m` bytes).
    #[inline]
    pub fn pq_code(&self, i: usize) -> &'a [u8] {
        debug_assert!(self.is_v3());
        let m = self.pq_m;
        &self.buf[self.codes_off + i * m..self.codes_off + (i + 1) * m]
    }

    /// v3: all PQ codes, `count * pq_m` bytes.
    pub fn pq_section(&self) -> &'a [u8] {
        debug_assert!(self.is_v3());
        &self.buf[self.codes_off..self.codes_off + self.count * self.pq_m]
    }

    /// v2/v3 only: the buffer holds the whole rerank section.
    pub fn has_rerank(&self) -> bool {
        self.has_rerank
    }

    /// The buffer holds the whole payload section.
    pub fn has_payload(&self) -> bool {
        self.has_payload
    }

    pub fn scan_len(&self) -> usize {
        self.header.scan_len
    }

    pub fn centroid(&self) -> Vec<f32> {
        (0..self.dim).map(|i| rd_f32(self.buf, self.centroid_off + i * 4)).collect()
    }

    #[inline]
    pub fn entry(&self, i: usize) -> Entry {
        let o = self.entries_off + i * ENTRY_BYTES;
        Entry {
            id: rd_u32(self.buf, o),
            payload_off: rd_u32(self.buf, o + 4),
            payload_len: rd_u32(self.buf, o + 8),
            attrs: rd_u32(self.buf, o + 12),
        }
    }

    #[inline]
    pub fn id(&self, i: usize) -> u32 {
        rd_u32(self.buf, self.entries_off + i * ENTRY_BYTES)
    }

    /// The `attrs` word of entry `i` (the attribute filter's input).
    #[inline]
    pub fn attrs(&self, i: usize) -> u32 {
        rd_u32(self.buf, self.entries_off + i * ENTRY_BYTES + 12)
    }

    /// v2: packed 4-bit code of entry `i` (`dim/2` bytes).
    #[inline]
    pub fn code4(&self, i: usize) -> &'a [u8] {
        debug_assert!(self.is_v2());
        let n = self.dim / 2;
        &self.buf[self.codes_off + i * n..self.codes_off + (i + 1) * n]
    }

    /// v2: 4-bit scale of entry `i`.
    #[inline]
    pub fn scale4(&self, i: usize) -> f32 {
        debug_assert!(self.is_v2());
        rd_f32(self.buf, self.scales_off + i * 4)
    }

    /// All packed 4-bit codes (v2), `count * dim/2` bytes.
    pub fn codes4_section(&self) -> &'a [u8] {
        &self.buf[self.codes_off..self.codes_off + self.count * (self.dim / 2)]
    }

    /// v2/v3: rerank record `i` (`[scale8 f32][code8 i8 * dim]`). Requires `has_rerank()`.
    pub fn rerank_record(&self, i: usize) -> &'a [u8] {
        assert!(self.has_rerank, "slab buffer does not include the rerank section");
        let len = self.header.rerank_record_len;
        let o = self.header.rerank_off + i * len;
        &self.buf[o..o + len]
    }

    /// int8 code of entry `i`: v1 codes section, or v2/v3 rerank record (requires `has_rerank()`).
    #[inline]
    pub fn code8(&self, i: usize) -> &'a [i8] {
        let s = if self.has_rerank_tier() {
            &self.rerank_record(i)[4..]
        } else {
            &self.buf[self.codes_off + i * self.dim..self.codes_off + (i + 1) * self.dim]
        };
        // i8 and u8 have identical layout and alignment.
        unsafe { core::slice::from_raw_parts(s.as_ptr() as *const i8, s.len()) }
    }

    /// int8 scale of entry `i` (v1 scales section, or v2/v3 rerank record).
    #[inline]
    pub fn scale8(&self, i: usize) -> f32 {
        if self.has_rerank_tier() {
            rd_f32(self.rerank_record(i), 0)
        } else {
            rd_f32(self.buf, self.scales_off + i * 4)
        }
    }

    /// Payload of entry `i`. Requires `has_payload()`.
    pub fn payload(&self, i: usize) -> &'a [u8] {
        assert!(self.has_payload, "slab buffer does not include the payload section");
        let e = self.entry(i);
        let start = self.header.payload_off + e.payload_off as usize;
        &self.buf[start..start + e.payload_len as usize]
    }

    pub fn payload_section(&self) -> &'a [u8] {
        assert!(self.has_payload, "slab buffer does not include the payload section");
        &self.buf[self.header.payload_off..self.header.payload_off + self.header.payload_len]
    }

    /// Appends one part-format record (see `part`) per entry, in entry order,
    /// skipping ids in `skip`. Returns the number of records appended. Needs the
    /// whole slab (rerank and payload sections). Feeding the records to a
    /// `SlabWriter` of the slab's own format, cell and centroid rebuilds the
    /// slab byte for byte (minus skipped entries).
    ///
    /// - v2: `code4`/`scale4` from the scan tier, no PQ codes (`pq_m` 0).
    /// - v3: PQ codes from the scan tier; `code4` zeros, `scale4` 0 and the
    ///   record flagged [`crate::part::RECORD_FLAG_NO_CODE4`] (a format 2 writer
    ///   rejects it).
    /// - v1: `code8`/`scale8` from the codes section, no `code4` (flagged as v3).
    ///
    /// `code8`/`scale8` always come from the int8 tier, payload and attrs from
    /// the entry. Errors (nothing appended) if the buffer is only a scan prefix
    /// or the dimension is odd.
    pub fn to_part_records(&self, out: &mut Vec<u8>, skip: Option<&crate::tomb::Tombstones>) -> Result<u32, FormatError> {
        use crate::part::{write_record, RECORD_FLAG_NO_CODE4};
        if !self.has_payload || (self.has_rerank_tier() && !self.has_rerank) {
            return Err(FormatError::Inconsistent("to_part_records needs the whole slab, not a scan prefix"));
        }
        if self.dim % 2 != 0 {
            return Err(FormatError::Inconsistent("part records require an even dimension"));
        }
        let zeros4 = vec![0u8; self.dim / 2];
        let mut n = 0u32;
        for i in 0..self.count {
            let e = self.entry(i);
            if skip.is_some_and(|t| t.contains(e.id)) {
                continue;
            }
            let (code4, scale4, flags, pq) = if self.is_v2() {
                (self.code4(i), self.scale4(i), 0, &[][..])
            } else if self.is_v3() {
                (&zeros4[..], 0.0, RECORD_FLAG_NO_CODE4, self.pq_code(i))
            } else {
                (&zeros4[..], 0.0, RECORD_FLAG_NO_CODE4, &[][..])
            };
            write_record(out, e.id, scale4, self.scale8(i), e.attrs, flags, code4, self.code8(i), pq, self.payload(i));
            n += 1;
        }
        Ok(n)
    }
}

/// Splits a cell's slab between two new means (`means2` = `2 * dim`, e.g. from
/// `kmeans::split_cell` over the slab's decoded entries): every entry not in
/// `tombstones` is decoded from its int8 tier against the slab's own mean
/// (`mean + scale8 * code8`), goes to the new mean with the higher cosine
/// (`kmeans::split_side`) and is RE-ENCODED against that mean as a part record
/// (4-bit MSE-clipped, int8 and, with `codebook`, PQ codes; flags 0; id, attrs
/// and payload kept). Returns `(records_a, records_b)`: fed to slab writers
/// with means A and B they give two correct residual slabs. Pass the index's
/// codebook for format 3 (records without PQ codes cannot build a v3 slab).
/// Needs the whole slab (rerank and payload sections). Errors on a scan
/// prefix, an odd dim, a `means2` of the wrong length or a codebook of
/// another dim.
pub fn split_records(
    slab_bytes: &[u8],
    tombstones: Option<&crate::tomb::Tombstones>,
    means2: &[f32],
    codebook: Option<&crate::pq::Codebook>,
) -> Result<(Vec<u8>, Vec<u8>), FormatError> {
    let v = SlabView::parse(slab_bytes)?;
    if !v.has_payload() || (v.has_rerank_tier() && !v.has_rerank()) {
        return Err(FormatError::Inconsistent("split_records needs the whole slab, not a scan prefix"));
    }
    let mean = v.centroid();
    let mut sp = crate::part::Splitter::new(v.dim, means2, codebook)?;
    let mut x = vec![0f32; v.dim];
    for i in 0..v.count {
        let e = v.entry(i);
        if tombstones.is_some_and(|t| t.contains(e.id)) {
            continue;
        }
        crate::quant::decode_residual_sq8(v.code8(i), v.scale8(i), &mean, &mut x);
        sp.push(&x, e.id, e.attrs, v.payload(i));
    }
    Ok(sp.finish())
}

#[inline]
fn align8(x: usize) -> usize {
    (x + 7) & !7
}

#[inline]
fn put_u32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_f32s(buf: &mut [u8], off: usize, v: &[f32]) {
    for (i, x) in v.iter().enumerate() {
        buf[off + i * 4..off + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
    }
}

fn put_entries(buf: &mut [u8], off: usize, entries: &[Entry]) {
    for (i, e) in entries.iter().enumerate() {
        let o = off + i * ENTRY_BYTES;
        put_u32(buf, o, e.id);
        put_u32(buf, o + 4, e.payload_off);
        put_u32(buf, o + 8, e.payload_len);
        put_u32(buf, o + 12, e.attrs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_parse_v1() {
        let dim = 5;
        let c = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mut b = SlabBuilderV1::new(dim, 42, &c);
        b.push(7, &[1, -2, 3, -4, 5], 0.01, b"hello", 9);
        b.push(8, &[0, 0, 0, 0, 0], 0.0, b"", 0);
        b.push(9, &[127, -127, 1, 1, 1], 0.5, b"xyz", 1);
        let n = 3;
        assert_eq!(b.byte_len(), align8(64 + dim * 4) + align8(n * 16) + align8(n * dim) + align8(n * 4) + 8);
        let bytes = b.finish();
        let v = SlabView::parse(&bytes).unwrap();
        assert_eq!(v.version(), 1);
        assert!(!v.has_rerank());
        assert_eq!(v.count, 3);
        assert_eq!(v.cell, 42);
        assert_eq!(v.dim, dim);
        assert_eq!(v.centroid(), c);
        assert_eq!(v.id(0), 7);
        assert_eq!(v.code8(0), &[1, -2, 3, -4, 5]);
        assert_eq!(v.code8(2), &[127, -127, 1, 1, 1]);
        assert!((v.scale8(2) - 0.5).abs() < 1e-9);
        assert_eq!(v.payload(0), b"hello");
        assert_eq!(v.payload(1), b"");
        assert_eq!(v.payload(2), b"xyz");
        assert_eq!(v.entry(0).attrs, 9);
        assert_eq!(v.scan_len(), v.header.payload_off);
        assert_eq!(v.header.total_len(), bytes.len());
    }

    fn sample_v2() -> (Vec<f32>, Vec<u8>) {
        let dim = 6;
        let c = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let mut b = SlabBuilder::new(dim, 42, &c);
        b.push(7, &[0x1F, 0x23, 0x9A], 0.25, &[1, -2, 3, -4, 5, -6], 0.01, b"hello", 9);
        b.push(8, &[0, 0, 0], 0.0, &[0; 6], 0.0, b"", 0);
        b.push(9, &[0x77, 0x99, 0x01], 0.5, &[127, -127, 1, 1, 1, 0], 0.03, b"xyz", 1);
        let n = 3;
        let want = align8(64 + dim * 4) + align8(n * 16) + align8(n * dim / 2) + align8(n * 4) + align8(n * (4 + dim)) + 8;
        assert_eq!(b.byte_len(), want);
        let scan_len = b.scan_len();
        let bytes = b.finish();
        assert_eq!(bytes.len(), want);
        assert_eq!(rd_u32(&bytes, 48) as usize, scan_len);
        (c, bytes)
    }

    #[test]
    fn build_and_parse_v2() {
        let (c, bytes) = sample_v2();
        let v = SlabView::parse(&bytes).unwrap();
        assert_eq!(v.version(), 2);
        assert!(v.has_rerank() && v.has_payload());
        assert_eq!((v.count, v.cell, v.dim), (3, 42, 6));
        assert_eq!(v.centroid(), c);
        assert_eq!(v.id(2), 9);
        assert_eq!(v.code4(0), &[0x1F, 0x23, 0x9A]);
        assert_eq!(v.code4(2), &[0x77, 0x99, 0x01]);
        assert_eq!(v.scale4(0), 0.25);
        assert_eq!(v.scale4(2), 0.5);
        assert_eq!(v.code8(0), &[1, -2, 3, -4, 5, -6]);
        assert_eq!(v.code8(2), &[127, -127, 1, 1, 1, 0]);
        assert_eq!(v.scale8(2), 0.03);
        assert_eq!(v.rerank_record(1).len(), 10);
        assert_eq!(&v.rerank_record(0)[..4], &0.01f32.to_le_bytes());
        assert_eq!(v.payload(0), b"hello");
        assert_eq!(v.payload(1), b"");
        assert_eq!(v.payload(2), b"xyz");
        assert_eq!(v.entry(2).attrs, 1);
        // section order and alignment
        let h = v.header;
        for off in [20usize, 24, 28, 32, 36, 40] {
            assert_eq!(rd_u32(&bytes, off) % 8, 0);
        }
        assert!(h.rerank_off < h.payload_off);
        assert_eq!(h.total_len(), bytes.len());
        assert_eq!(h.rerank_record_len, 10);
    }

    #[test]
    fn parse_v2_scan_prefix() {
        let (c, bytes) = sample_v2();
        let h = SlabHeader::parse(&bytes[..64]).unwrap();
        let prefix = &bytes[..h.scan_len];
        let v = SlabView::parse(prefix).unwrap();
        assert!(!v.has_rerank());
        assert!(!v.has_payload());
        assert_eq!(v.centroid(), c);
        assert_eq!(v.code4(1), &[0, 0, 0]);
        assert_eq!(v.scale4(2), 0.5);
        // rerank present but payload cut off
        let v = SlabView::parse(&bytes[..h.payload_off]).unwrap();
        assert!(v.has_rerank() && !v.has_payload());
        assert_eq!(v.code8(0), &[1, -2, 3, -4, 5, -6]);
        // one byte short of the scan prefix is an error
        assert!(matches!(SlabView::parse(&bytes[..h.scan_len - 1]), Err(FormatError::TooShort { .. })));
    }

    fn sample_v3() -> (Vec<f32>, Vec<Vec<u8>>, Vec<u8>) {
        let (dim, m) = (8, 4);
        let c: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let codes: Vec<Vec<u8>> = (0..5u8).map(|i| vec![i, 255 - i, i * 3, 7]).collect();
        let mut b = SlabBuilderV3::new(dim, 77, &c, 0);
        for (i, code) in codes.iter().enumerate() {
            let c8: Vec<i8> = (0..dim).map(|d| (i * 10 + d) as i8).collect();
            let payload = vec![b'a' + i as u8; i];
            b.push(100 + i as u32, code, &c8, 0.01 * i as f32, &payload, i as u32).unwrap();
        }
        assert_eq!(b.pq_m(), m);
        // wrong code count or missing codes are rejected without adding anything
        assert!(b.push(1, &[1, 2, 3], &[0; 8], 0.0, b"", 0).is_err());
        assert!(b.push(1, &[], &[0; 8], 0.0, b"", 0).is_err());
        assert_eq!(b.len(), 5);
        let n = 5;
        let want = align8(64 + dim * 4) + align8(n * 16) + align8(n * m) + align8(n * (4 + dim)) + (0 + 1 + 2 + 3 + 4);
        assert_eq!(b.byte_len(), want);
        let scan_len = b.scan_len();
        let bytes = b.finish();
        assert_eq!(bytes.len(), want);
        assert_eq!(rd_u32(&bytes, 48) as usize, scan_len);
        (c, codes, bytes)
    }

    #[test]
    fn build_and_parse_v3() {
        let (c, codes, bytes) = sample_v3();
        let v = SlabView::parse(&bytes).unwrap();
        assert_eq!(v.version(), 3);
        assert!(v.is_v3() && !v.is_v2() && v.has_rerank_tier());
        assert!(v.has_rerank() && v.has_payload());
        assert_eq!((v.count, v.cell, v.dim, v.pq_m()), (5, 77, 8, 4));
        assert_eq!(v.header.pq_m, 4);
        assert_eq!(v.centroid(), c);
        for i in 0..5 {
            assert_eq!(v.id(i), 100 + i as u32);
            assert_eq!(v.pq_code(i), &codes[i][..]);
            assert_eq!(v.code8(i)[3], (i * 10 + 3) as i8);
            assert_eq!(v.scale8(i), 0.01 * i as f32);
            assert_eq!(v.payload(i), &vec![b'a' + i as u8; i][..]);
            assert_eq!(v.entry(i).attrs, i as u32);
        }
        assert_eq!(v.pq_section(), codes.concat());
        for off in [20usize, 24, 28, 36, 40] {
            assert_eq!(rd_u32(&bytes, off) % 8, 0);
        }
        assert_eq!(rd_u32(&bytes, 32), 4);
        assert_eq!(v.header.total_len(), bytes.len());
        // the scan prefix parses on its own and holds the pq codes
        let h = SlabHeader::parse(&bytes[..64]).unwrap();
        let p = SlabView::parse(&bytes[..h.scan_len]).unwrap();
        assert!(!p.has_rerank() && !p.has_payload());
        assert_eq!(p.pq_code(4), &codes[4][..]);
        assert_eq!(p.centroid(), c);
        assert!(matches!(SlabView::parse(&bytes[..h.scan_len - 1]), Err(FormatError::TooShort { .. })));
        // pq_m that does not divide dim is rejected
        let mut bad = bytes.clone();
        put_u32(&mut bad, 32, 3);
        assert!(SlabView::parse(&bad).is_err());
        // an empty v3 slab (no records, pq_m unknown) is valid
        let e = SlabBuilderV3::new(8, 1, &c, 0).finish();
        let ev = SlabView::parse(&e).unwrap();
        assert_eq!((ev.count, ev.pq_m(), ev.version()), (0, 0, 3));
    }

    /// Two well separated clusters of unit vectors, encoded against their common
    /// mean into a v3 slab (with a trained-from-samples PQ codebook).
    fn split_fixture() -> (usize, Vec<f32>, Vec<f32>, crate::pq::Codebook, Vec<u8>) {
        use crate::math::{normalize, Rng};
        let (dim, n) = (16usize, 120usize);
        let mut rng = Rng::new(11);
        let mut xs = Vec::with_capacity(n * dim);
        for i in 0..n {
            let mut x: Vec<f32> = (0..dim).map(|_| (rng.below(2001) as f32 / 1000.0 - 1.0) * 0.2).collect();
            x[if i % 2 == 0 { 0 } else { 1 }] += 1.0;
            normalize(&mut x);
            xs.extend_from_slice(&x);
        }
        let mut mean = vec![0f32; dim];
        for x in xs.chunks_exact(dim) {
            for d in 0..dim {
                mean[d] += x[d] / n as f32;
            }
        }
        let resid: Vec<f32> = xs.chunks_exact(dim).flat_map(|x| x.iter().zip(&mean).map(|(a, b)| a - b).collect::<Vec<_>>()).collect();
        let cb = crate::pq::Codebook::init_from_samples(&resid, dim, 4, 3).unwrap();
        let mut enc = crate::part::RecordEncoder::new(dim, Some(&cb)).unwrap();
        let mut recs = Vec::new();
        for (i, x) in xs.chunks_exact(dim).enumerate() {
            enc.encode(&mut recs, x, &mean, 1000 + i as u32, i as u32 % 5, &vec![b'p'; i % 3]);
        }
        let mut w = SlabWriter::new(3, dim, 9, &mean).unwrap();
        assert_eq!(w.add_part_records(&recs, dim).unwrap(), n as u32);
        (dim, xs, mean, cb, w.finish())
    }

    #[test]
    fn split_records_reencodes_against_the_new_means() {
        use crate::part::records;
        use crate::quant::{decode_residual_sq8, encode_residual_sq4};
        let (dim, _xs, mean, cb, slab) = split_fixture();
        let v = SlabView::parse(&slab).unwrap();
        let n = v.count;
        let decoded: Vec<f32> = (0..n)
            .flat_map(|i| {
                let mut x = vec![0f32; dim];
                decode_residual_sq8(v.code8(i), v.scale8(i), &mean, &mut x);
                x
            })
            .collect();
        let (means2, side) = crate::kmeans::split_cell(&decoded, dim, 1);
        // the two clusters (even / odd entries) end up on different sides
        assert!((0..n).all(|i| side[i] == side[i % 2]) && side[0] != side[1]);
        let dead = crate::tomb::Tombstones::from_ids(&[1000, 1007]);
        let (a, b) = split_records(&slab, Some(&dead), &means2, Some(&cb)).unwrap();
        let mut seen = 0;
        for (half, recs) in [(0usize, &a), (1, &b)] {
            let m = &means2[half * dim..(half + 1) * dim];
            let mut w3 = SlabWriter::new(3, dim, 9, m).unwrap();
            let mut w2 = SlabWriter::new(2, dim, 9, m).unwrap();
            for r in records(recs, dim) {
                let r = r.unwrap();
                let i = (r.id - 1000) as usize;
                assert!(!dead.contains(r.id));
                assert_eq!(side[i] as usize, half, "entry {i} on the wrong side");
                assert_eq!((r.attrs, r.payload.len(), r.flags), (i as u32 % 5, i % 3, 0));
                let x = &decoded[i * dim..(i + 1) * dim];
                // int8 tier: residual against the NEW mean, reconstructs the decoded vector
                let mut y = vec![0f32; dim];
                decode_residual_sq8(r.code8, r.scale8, m, &mut y);
                for d in 0..dim {
                    assert!((y[d] - x[d]).abs() <= r.scale8 * 0.5 + 1e-6, "dim {d}: {} vs {}", y[d], x[d]);
                }
                // 4-bit and PQ tiers are what an assign job would write for x against m
                let (mut c4, mut scratch) = (vec![0u8; dim / 2], vec![0f32; dim]);
                assert_eq!(encode_residual_sq4(x, m, &mut c4, &mut scratch), r.scale4);
                assert_eq!(r.code4, &c4[..]);
                let resid: Vec<f32> = x.iter().zip(m).map(|(a, b)| a - b).collect();
                let mut pq = vec![0u8; cb.m];
                crate::pq::encode(&resid, &cb, &mut pq);
                assert_eq!(r.pq, &pq[..]);
                seen += 1;
            }
            // both halves build v3 and v2 slabs
            assert_eq!(w3.add_part_records(recs, dim).unwrap() as usize, records(recs, dim).count());
            assert_eq!(w2.add_part_records(recs, dim).unwrap() as usize, records(recs, dim).count());
            let s3 = w3.finish();
            assert_eq!(SlabView::parse(&s3).unwrap().centroid(), m);
        }
        assert_eq!(seen, n - 2);
        // the part-records path (old slab records, e.g. delta records) splits identically
        let mut old = Vec::new();
        v.to_part_records(&mut old, Some(&dead)).unwrap();
        assert_eq!(crate::part::split_part_records(&old, &mean, &means2, Some(&cb)).unwrap(), (a.clone(), b.clone()));
        // reencode_records against one mean equals the split half for that side's entries
        let ra = crate::part::reencode_records(&old, &mean, &means2[..dim], Some(&cb)).unwrap();
        let ids_a: Vec<u32> = records(&a, dim).map(|r| r.unwrap().id).collect();
        let ra_a: Vec<u8> = records(&ra, dim)
            .map(|r| r.unwrap())
            .filter(|r| ids_a.contains(&r.id))
            .flat_map(|r| {
                let mut o = Vec::new();
                crate::part::write_record(&mut o, r.id, r.scale4, r.scale8, r.attrs, r.flags, r.code4, r.code8, r.pq, r.payload);
                o
            })
            .collect();
        assert_eq!(ra_a, a);
        // re-encoding against the same mean keeps the int8 tier (within one step) and without a codebook drops PQ codes
        let same = crate::part::reencode_records(&old, &mean, &mean, None).unwrap();
        for (r, o) in records(&same, dim).zip(records(&old, dim)) {
            let (r, o) = (r.unwrap(), o.unwrap());
            assert!(r.pq.is_empty() && r.id == o.id);
            assert!((r.scale8 - o.scale8).abs() <= o.scale8 * 1e-5);
            assert!(r.code8.iter().zip(o.code8).all(|(x, y)| (*x as i32 - *y as i32).abs() <= 1));
        }
        // errors: scan prefix, wrong means2 length, codebook of another dim
        assert!(split_records(&slab[..v.scan_len()], None, &means2, Some(&cb)).is_err());
        assert!(split_records(&slab, None, &means2[..dim], Some(&cb)).is_err());
        let other = crate::pq::Codebook::init_from_samples(&vec![0.5f32; 8 * 300], 8, 2, 1).unwrap();
        assert!(split_records(&slab, None, &means2, Some(&other)).is_err());
        assert!(crate::part::reencode_records(&old, &mean, &mean[..4], None).is_err());
    }

    #[test]
    fn rejects_garbage() {
        assert!(SlabView::parse(b"nope").is_err());
        let mut bad = SlabBuilder::new(2, 0, &[0.0, 0.0]).finish();
        bad[0] = b'X';
        assert_eq!(SlabView::parse(&bad), Err(FormatError::BadMagic));
        let mut bad = SlabBuilder::new(2, 0, &[0.0, 0.0]).finish();
        bad[4] = 4;
        assert_eq!(SlabView::parse(&bad), Err(FormatError::BadVersion(4)));
        let mut bad = SlabBuilder::new(2, 0, &[0.0, 0.0]).finish();
        bad[4] = 3; // v3 with v2 code type
        assert!(SlabView::parse(&bad).is_err());
        let mut bad = SlabBuilder::new(2, 0, &[0.0, 0.0]).finish();
        bad[6] = 1; // v2 with v1 code type
        assert!(SlabView::parse(&bad).is_err());
        let mut bad = SlabBuilderV1::new(2, 0, &[0.0, 0.0]).finish();
        bad[6] = 2; // v1 with v2 code type
        assert!(SlabView::parse(&bad).is_err());
    }
}

impl<'a> PartialEq for SlabView<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.buf == other.buf
    }
}
impl<'a> core::fmt::Debug for SlabView<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "SlabView(cell={}, dim={}, count={})", self.cell, self.dim, self.count)
    }
}
