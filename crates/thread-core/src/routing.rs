//! Compact routing tables: the fine cell means stored as IEEE half floats
//! (`fine.f16`) or int8 with one f32 scale per row (`fine.i8`), the same means
//! in centroid-tree group order (`fine.grouped.*`, "VIBG") so a cold reader can
//! range-read just the groups a query needs, and [`PartialRouter`], which
//! routes over whatever subset of cell means it has been given.
//!
//! The global means table only routes (picks the `nprobe` nearest cells) and
//! gives delta scans their per-cell base score `q·mean`; every slab carries its
//! own f32 mean for scoring. So a coarser copy changes which cells a query
//! probes at the margin, never how a probed slab is scored. Decoding happens
//! once, on load: a [`crate::route::Router`] over decoded means routes exactly
//! as it does over `fine.f32`, only with slightly different means.
//!
//! Row layouts (little-endian, `k` rows of `dim` values, no header):
//! ```text
//! f32  dim * 4 bytes      the raw means (fine.f32)
//! f16  dim * 2 bytes      IEEE 754 binary16, round to nearest even
//! i8   4 + dim bytes      [scale f32][code i8 * dim], value = code * scale,
//!                         scale = max |x| / 127 (0 for a zero row)
//! ```
//!
//! Tree-grouped table ("VIBG" v1, little-endian):
//! ```text
//! header (32 bytes): magic "VIBG", version u32 = 1, elem u32 (0 f32, 1 f16, 2 i8),
//!                    dim u32, k u32, groups u32, n_rows u32, data_off u32
//! offsets   (groups + 1) u32    CSR: group g is rows offsets[g]..offsets[g + 1]
//! cells     n_rows u32          the cell id of every row
//! zero padding to data_off (a multiple of 8)
//! rows      n_rows * row_bytes  in the order above
//! ```
//! The directory is the tree's own membership (`CentroidTree::member_offsets`
//! / `all_members`), so a reader holding the tree can compute the byte range of
//! any group without reading the directory: `data_off + offsets[g] * row_bytes`.

use crate::error::{need, rd_f32, rd_u32, FormatError};
use crate::math::{dot, inv_norms};
use crate::route::{push_top_unique, CentroidTree};

/// Element type of a routing table row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    F32,
    F16,
    I8,
}

impl Elem {
    pub fn code(self) -> u32 {
        match self {
            Elem::F32 => 0,
            Elem::F16 => 1,
            Elem::I8 => 2,
        }
    }

    pub fn from_code(c: u32) -> Option<Elem> {
        match c {
            0 => Some(Elem::F32),
            1 => Some(Elem::F16),
            2 => Some(Elem::I8),
            _ => None,
        }
    }

    /// `"f32"`, `"f16"` or `"i8"` (the manifest's names).
    pub fn name(self) -> &'static str {
        match self {
            Elem::F32 => "f32",
            Elem::F16 => "f16",
            Elem::I8 => "i8",
        }
    }

    pub fn parse(s: &str) -> Option<Elem> {
        match s {
            "f32" => Some(Elem::F32),
            "f16" => Some(Elem::F16),
            "i8" => Some(Elem::I8),
            _ => None,
        }
    }

    /// Bytes of one row of `dim` values.
    pub fn row_bytes(self, dim: usize) -> usize {
        match self {
            Elem::F32 => 4 * dim,
            Elem::F16 => 2 * dim,
            Elem::I8 => 4 + dim,
        }
    }
}

/// `x` as IEEE 754 binary16 bits, rounded to nearest even (overflow gives
/// infinity, NaN stays NaN).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign
            | 0x7c00
            | if man != 0 {
                0x200 | (man >> 13) as u16
            } else {
                0
            };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // subnormal half (or zero): value = r * 2^-24
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1; // a carry into the exponent is the correct rounding (up to infinity)
    }
    sign | h as u16
}

/// The f32 value of binary16 bits `h` (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e: u32 = 127 - 14;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

fn encode_row(row: &[f32], elem: Elem, out: &mut Vec<u8>) {
    match elem {
        Elem::F32 => {
            for x in row {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Elem::F16 => {
            for &x in row {
                out.extend_from_slice(&f32_to_f16(x).to_le_bytes());
            }
        }
        Elem::I8 => {
            let max = row.iter().fold(0f32, |m, x| m.max(x.abs()));
            let scale = if max > 0.0 && max.is_finite() {
                max / 127.0
            } else {
                0.0
            };
            out.extend_from_slice(&scale.to_le_bytes());
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for &x in row {
                out.push((x * inv).round().clamp(-127.0, 127.0) as i8 as u8);
            }
        }
    }
}

/// Encodes `means` (`k * dim` f32) as `k` rows of `elem`.
pub fn encode_rows(means: &[f32], dim: usize, elem: Elem) -> Vec<u8> {
    assert!(
        dim > 0 && means.len() % dim == 0,
        "means is not a whole number of rows"
    );
    let mut out = Vec::with_capacity(means.len() / dim * elem.row_bytes(dim));
    for row in means.chunks_exact(dim) {
        encode_row(row, elem, &mut out);
    }
    out
}

/// Decodes rows of `elem` into `out` (`rows * dim` f32).
pub fn decode_rows_into(
    bytes: &[u8],
    dim: usize,
    elem: Elem,
    out: &mut [f32],
) -> Result<(), FormatError> {
    let rb = elem.row_bytes(dim);
    if dim == 0 || bytes.len() % rb != 0 {
        return Err(FormatError::Inconsistent(
            "routing table length is not a whole number of rows",
        ));
    }
    if out.len() != bytes.len() / rb * dim {
        return Err(FormatError::Inconsistent(
            "routing table rows != output rows",
        ));
    }
    match elem {
        Elem::F32 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        Elem::F16 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        Elem::I8 => {
            for (row, o) in bytes.chunks_exact(rb).zip(out.chunks_exact_mut(dim)) {
                let scale = rd_f32(row, 0);
                for (v, &c) in o.iter_mut().zip(&row[4..]) {
                    *v = c as i8 as f32 * scale;
                }
            }
        }
    }
    Ok(())
}

/// Decodes a whole routing table of `elem` rows into f32 means (`rows * dim`).
pub fn decode_rows(bytes: &[u8], dim: usize, elem: Elem) -> Result<Vec<f32>, FormatError> {
    let rb = elem.row_bytes(dim);
    if dim == 0 || bytes.len() % rb != 0 {
        return Err(FormatError::Inconsistent(
            "routing table length is not a whole number of rows",
        ));
    }
    let mut out = vec![0f32; bytes.len() / rb * dim];
    decode_rows_into(bytes, dim, elem, &mut out)?;
    Ok(out)
}

pub const GROUPED_MAGIC: &[u8; 4] = b"VIBG";
pub const GROUPED_VERSION: u32 = 1;
pub const GROUPED_HEADER: usize = 32;

/// Byte offset of the first row of a grouped table with this many groups and rows.
pub fn grouped_data_off(groups: usize, n_rows: usize) -> usize {
    (GROUPED_HEADER + 4 * (groups + 1) + 4 * n_rows).div_ceil(8) * 8
}

/// Header and directory of a tree-grouped routing table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupedTable {
    pub elem: Elem,
    pub dim: usize,
    pub k: usize,
    pub groups: usize,
    pub n_rows: usize,
    pub data_off: usize,
    pub offsets: Vec<u32>,
    pub cells: Vec<u32>,
}

impl GroupedTable {
    pub fn row_bytes(&self) -> usize {
        self.elem.row_bytes(self.dim)
    }

    /// `(byte offset, byte length)` of group `g`'s rows in the object.
    pub fn group_range(&self, g: usize) -> (usize, usize) {
        let (a, b) = (self.offsets[g] as usize, self.offsets[g + 1] as usize);
        (
            self.data_off + a * self.row_bytes(),
            (b - a) * self.row_bytes(),
        )
    }

    /// Total object size.
    pub fn byte_len(&self) -> usize {
        self.data_off + self.n_rows * self.row_bytes()
    }

    /// Whether the rows are in `tree`'s membership order (the directory is the tree's).
    pub fn matches_tree(&self, tree: &CentroidTree) -> bool {
        self.dim == tree.dim
            && self.k == tree.k
            && self.offsets == tree.member_offsets()
            && self.cells == tree.all_members()
    }

    /// Parses the header and directory (`buf` may be just the first `data_off` bytes or the whole object).
    pub fn parse(buf: &[u8]) -> Result<GroupedTable, FormatError> {
        need(buf, GROUPED_HEADER)?;
        if &buf[0..4] != GROUPED_MAGIC {
            return Err(FormatError::BadMagic);
        }
        let version = rd_u32(buf, 4);
        if version != GROUPED_VERSION {
            return Err(FormatError::BadVersion(version.min(u16::MAX as u32) as u16));
        }
        let elem = Elem::from_code(rd_u32(buf, 8)).ok_or(FormatError::Inconsistent(
            "grouped table: unknown element type",
        ))?;
        let (dim, k, groups, n_rows, data_off) = (
            rd_u32(buf, 12) as usize,
            rd_u32(buf, 16) as usize,
            rd_u32(buf, 20) as usize,
            rd_u32(buf, 24) as usize,
            rd_u32(buf, 28) as usize,
        );
        if dim == 0 || k == 0 || groups == 0 {
            return Err(FormatError::Inconsistent(
                "grouped table: dim, k and groups must be positive",
            ));
        }
        if data_off != grouped_data_off(groups, n_rows) {
            return Err(FormatError::Inconsistent(
                "grouped table: data_off does not follow the directory",
            ));
        }
        need(buf, GROUPED_HEADER + 4 * (groups + 1 + n_rows))?;
        let mut off = GROUPED_HEADER;
        let offsets: Vec<u32> = (0..=groups).map(|i| rd_u32(buf, off + 4 * i)).collect();
        off += 4 * (groups + 1);
        let cells: Vec<u32> = (0..n_rows).map(|i| rd_u32(buf, off + 4 * i)).collect();
        if offsets[0] != 0
            || offsets[groups] as usize != n_rows
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(FormatError::Inconsistent(
                "grouped table: offsets are not a CSR index over the rows",
            ));
        }
        if cells.iter().any(|&c| c as usize >= k) {
            return Err(FormatError::Inconsistent("grouped table: cell id >= k"));
        }
        let t = GroupedTable {
            elem,
            dim,
            k,
            groups,
            n_rows,
            data_off,
            offsets,
            cells,
        };
        if buf.len() > t.data_off && buf.len() != t.byte_len() {
            return Err(FormatError::Inconsistent(
                "grouped table: byte length != header sizes",
            ));
        }
        Ok(t)
    }
}

/// Writes `means` (`k * dim`, raw) as a tree-grouped table of `elem` rows in `tree`'s group order.
pub fn encode_grouped(means: &[f32], dim: usize, tree: &CentroidTree, elem: Elem) -> Vec<u8> {
    assert_eq!(tree.dim, dim, "tree dim != means dim");
    assert_eq!(
        means.len(),
        tree.k * dim,
        "means buffer size != tree k * dim"
    );
    let groups = tree.groups();
    let cells = tree.all_members();
    let data_off = grouped_data_off(groups, cells.len());
    let mut out = Vec::with_capacity(data_off + cells.len() * elem.row_bytes(dim));
    out.extend_from_slice(GROUPED_MAGIC);
    for v in [
        GROUPED_VERSION,
        elem.code(),
        dim as u32,
        tree.k as u32,
        groups as u32,
        cells.len() as u32,
        data_off as u32,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in tree.member_offsets().iter().chain(cells) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.resize(data_off, 0);
    for &c in cells {
        let c = c as usize;
        encode_row(&means[c * dim..(c + 1) * dim], elem, &mut out);
    }
    out
}

const NONE: u32 = u32::MAX;

/// Routing over a subset of the cell means: the centroid tree plus the rows
/// of the groups loaded so far (from a tree-grouped table, any element type).
/// A beam lookup ranks the members of the best `beam` groups exactly as
/// [`CentroidTree::route`] does over the full table (`dot(q, mean) * inv`,
/// ties to the lower cell id); cells whose means are not loaded are not
/// candidates. Load the groups [`PartialRouter::missing_groups`] names first
/// to get the full-table result.
#[derive(Clone, Debug)]
pub struct PartialRouter {
    pub dim: usize,
    pub k: usize,
    elem: Elem,
    tree: CentroidTree,
    /// Slot of each cell in `means` / `inv` (`NONE` = not loaded).
    slot: Vec<u32>,
    means: Vec<f32>,
    inv: Vec<f32>,
    /// Row of each cell in the grouped table (its first appearance).
    row_of: Vec<u32>,
    group_loaded: Vec<bool>,
}

impl PartialRouter {
    /// A router with no means loaded yet, for rows of `elem` in `tree`'s group order.
    pub fn new(tree: CentroidTree, elem: Elem) -> PartialRouter {
        let mut row_of = vec![NONE; tree.k];
        for (r, &c) in tree.all_members().iter().enumerate() {
            if row_of[c as usize] == NONE {
                row_of[c as usize] = r as u32;
            }
        }
        let groups = tree.groups();
        PartialRouter {
            dim: tree.dim,
            k: tree.k,
            elem,
            slot: vec![NONE; tree.k],
            means: Vec::new(),
            inv: Vec::new(),
            row_of,
            group_loaded: vec![false; groups],
            tree,
        }
    }

    pub fn elem(&self) -> Elem {
        self.elem
    }

    pub fn tree(&self) -> &CentroidTree {
        &self.tree
    }

    pub fn groups(&self) -> usize {
        self.tree.groups()
    }

    /// Rows of the grouped table (`n_members` of the tree).
    pub fn n_rows(&self) -> usize {
        self.tree.n_members()
    }

    pub fn row_bytes(&self) -> usize {
        self.elem.row_bytes(self.dim)
    }

    /// `(first row, row count)` of group `g` in the grouped table.
    pub fn group_rows(&self, g: usize) -> (usize, usize) {
        let o = self.tree.member_offsets();
        (o[g] as usize, (o[g + 1] - o[g]) as usize)
    }

    /// Row of cell `c` in the grouped table (`None` if no group lists it).
    pub fn cell_row(&self, c: usize) -> Option<usize> {
        self.row_of
            .get(c)
            .filter(|&&r| r != NONE)
            .map(|&r| r as usize)
    }

    pub fn has_cell(&self, c: usize) -> bool {
        self.slot.get(c).is_some_and(|&s| s != NONE)
    }

    pub fn has_group(&self, g: usize) -> bool {
        self.group_loaded.get(g).copied().unwrap_or(false)
    }

    pub fn loaded_cells(&self) -> usize {
        self.inv.len()
    }

    pub fn loaded_groups(&self) -> usize {
        self.group_loaded.iter().filter(|&&l| l).count()
    }

    /// The `beam` best groups for `q` (clamped to `1..=groups`), best first.
    pub fn top_groups(&self, q: &[f32], beam: usize) -> Vec<usize> {
        self.tree
            .top_groups(q, beam)
            .into_iter()
            .map(|(g, _)| g)
            .collect()
    }

    /// The groups among the `beam` best for `q` whose rows are not all loaded, best first.
    pub fn missing_groups(&self, q: &[f32], beam: usize) -> Vec<usize> {
        self.top_groups(q, beam)
            .into_iter()
            .filter(|&g| !self.group_loaded[g])
            .collect()
    }

    /// Adds consecutive rows `first_row..` of the grouped table (`bytes` = a
    /// whole number of rows). Cells already loaded are left as they are.
    /// Returns how many cells were newly loaded.
    pub fn add_rows(&mut self, first_row: usize, bytes: &[u8]) -> Result<usize, FormatError> {
        let rb = self.row_bytes();
        if bytes.len() % rb != 0 {
            return Err(FormatError::Inconsistent(
                "grouped rows: not a whole number of rows",
            ));
        }
        let n = bytes.len() / rb;
        let members = self.tree.all_members();
        if first_row + n > members.len() {
            return Err(FormatError::Inconsistent("grouped rows: past the last row"));
        }
        let dim = self.dim;
        let mut added = 0;
        let mut row = vec![0f32; dim];
        for i in 0..n {
            let c = members[first_row + i] as usize;
            if self.slot[c] != NONE {
                continue;
            }
            decode_rows_into(&bytes[i * rb..(i + 1) * rb], dim, self.elem, &mut row)?;
            self.slot[c] = self.inv.len() as u32;
            self.inv.push(inv_norms(&row, dim)[0]);
            self.means.extend_from_slice(&row);
            added += 1;
        }
        if added > 0 {
            for g in 0..self.groups() {
                if !self.group_loaded[g] {
                    self.group_loaded[g] = self
                        .tree
                        .members(g)
                        .iter()
                        .all(|&c| self.slot[c as usize] != NONE);
                }
            }
        }
        Ok(added)
    }

    /// Adds the rows of group `g` (`bytes` = exactly its rows).
    pub fn add_group(&mut self, g: usize, bytes: &[u8]) -> Result<usize, FormatError> {
        if g >= self.groups() {
            return Err(FormatError::Inconsistent(
                "grouped rows: group id >= groups",
            ));
        }
        let (first, n) = self.group_rows(g);
        if bytes.len() != n * self.row_bytes() {
            return Err(FormatError::Inconsistent(
                "grouped rows: byte length != the group's rows",
            ));
        }
        self.add_rows(first, bytes)
    }

    #[inline]
    fn mean(&self, c: usize) -> Option<(&[f32], f32)> {
        let s = *self.slot.get(c)?;
        if s == NONE {
            return None;
        }
        let s = s as usize;
        Some((&self.means[s * self.dim..(s + 1) * self.dim], self.inv[s]))
    }

    /// The `nprobe` best loaded cells (by cosine, descending, ties to the lower
    /// cell id) among the members of the best `beam` groups. With every one of
    /// those groups loaded this equals [`CentroidTree::route`] over the full table.
    pub fn route_beam(&self, q: &[f32], beam: usize, nprobe: usize) -> Vec<(usize, f32)> {
        assert_eq!(q.len(), self.dim, "query dim != router dim");
        let mut top: Vec<(usize, f32)> = Vec::with_capacity(nprobe + 1);
        for g in self.top_groups(q, beam) {
            for &c in self.tree.members(g) {
                if let Some((m, inv)) = self.mean(c as usize) {
                    push_top_unique(&mut top, nprobe, c as usize, dot(q, m) * inv);
                }
            }
        }
        top
    }

    /// `q·mean[c]` for the listed cells, in order (the delta-scan base scores);
    /// an error if one of them is not loaded.
    pub fn cell_dots_for(&self, q: &[f32], cells: &[u32]) -> Result<Vec<f32>, FormatError> {
        cells
            .iter()
            .map(|&c| {
                self.mean(c as usize)
                    .map(|(m, _)| dot(q, m))
                    .ok_or(FormatError::Inconsistent("cell mean not loaded"))
            })
            .collect()
    }

    /// Bytes of decoded means held.
    pub fn resident_bytes(&self) -> usize {
        4 * (self.means.len() + self.inv.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{normalize, Rng};
    use crate::route::Router;

    fn uni(rng: &mut Rng) -> f32 {
        rng.below(20001) as f32 / 10000.0 - 1.0
    }

    fn means(rng: &mut Rng, k: usize, dim: usize) -> Vec<f32> {
        (0..k * dim)
            .map(|i| uni(rng) * (0.05 + (i / dim % 7) as f32 * 0.02))
            .collect()
    }

    fn unit(rng: &mut Rng, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| uni(rng)).collect();
        normalize(&mut v);
        v
    }

    #[test]
    fn f16_roundtrips_every_half_and_rounds_to_nearest_even() {
        for h in 0..=u16::MAX {
            let x = f16_to_f32(h);
            if x.is_nan() {
                assert!(f16_to_f32(f32_to_f16(x)).is_nan());
                continue;
            }
            assert_eq!(f32_to_f16(x), h, "half {h:#06x} -> {x}");
        }
        // nearest (ties to even) against an exhaustive search over the halves around x
        let mut rng = Rng::new(5);
        for i in 0..20000 {
            let x = match i % 4 {
                0 => uni(&mut rng) * 1e-3,
                1 => uni(&mut rng) * 1e-6, // subnormal halves
                2 => uni(&mut rng) * 70000.0,
                _ => uni(&mut rng),
            };
            let h = f32_to_f16(x);
            let y = f16_to_f32(h);
            if y.is_infinite() {
                assert!(x.abs() >= 65504.0);
                continue;
            }
            let err = (y as f64 - x as f64).abs();
            for d in [-1i32, 1] {
                let n = (h as i32 + d) as u16;
                if n & 0x7c00 == 0x7c00 || (h & 0x7fff == 0 && d < 0) {
                    continue;
                }
                let other = f16_to_f32(n);
                if other.signum() != y.signum() && y != 0.0 {
                    continue;
                }
                let e2 = (other as f64 - x as f64).abs();
                assert!(
                    err < e2 || (err == e2 && h & 1 == 0),
                    "x {x}: {y} (h {h:#06x}) vs {other}"
                );
            }
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(1e-9), 0);
    }

    #[test]
    fn rows_encode_and_decode() {
        let mut rng = Rng::new(1);
        let (k, dim) = (37, 24);
        let m = means(&mut rng, k, dim);
        for elem in [Elem::F32, Elem::F16, Elem::I8] {
            let b = encode_rows(&m, dim, elem);
            assert_eq!(b.len(), k * elem.row_bytes(dim));
            let d = decode_rows(&b, dim, elem).unwrap();
            assert_eq!(d.len(), m.len());
            for (row, drow) in m.chunks_exact(dim).zip(d.chunks_exact(dim)) {
                let max = row.iter().fold(0f32, |a, x| a.max(x.abs()));
                for (x, y) in row.iter().zip(drow) {
                    let tol = match elem {
                        Elem::F32 => 0.0,
                        Elem::F16 => x.abs() / 1024.0,
                        Elem::I8 => max / 254.0 * 1.0001,
                    };
                    assert!((x - y).abs() <= tol, "{elem:?}: {x} vs {y}");
                }
            }
            assert_eq!(Elem::parse(elem.name()), Some(elem));
            assert_eq!(Elem::from_code(elem.code()), Some(elem));
            assert!(decode_rows(&b[..b.len() - 1], dim, elem).is_err());
        }
        assert_eq!(
            decode_rows(&encode_rows(&m, dim, Elem::F32), dim, Elem::F32).unwrap(),
            m
        );
        // a zero row encodes with scale 0
        let z = encode_rows(&vec![0.0; dim], dim, Elem::I8);
        assert_eq!(decode_rows(&z, dim, Elem::I8).unwrap(), vec![0.0; dim]);
    }

    #[test]
    fn quantized_router_routes_like_the_f32_one_on_separated_means() {
        let mut rng = Rng::new(9);
        let (k, dim) = (200, 64);
        let m = means(&mut rng, k, dim);
        let full = Router::new(dim, 1, k, &vec![1.0; dim], m.clone());
        let f16 = Router::new(
            dim,
            1,
            k,
            &vec![1.0; dim],
            decode_rows(&encode_rows(&m, dim, Elem::F16), dim, Elem::F16).unwrap(),
        );
        let mut same = 0;
        for _ in 0..200 {
            let q = unit(&mut rng, dim);
            let a: Vec<usize> = full.route_flat(&q, 10).into_iter().map(|x| x.0).collect();
            let b: Vec<usize> = f16.route_flat(&q, 10).into_iter().map(|x| x.0).collect();
            same += a.iter().filter(|c| b.contains(c)).count();
        }
        assert!(same as f32 / 2000.0 > 0.99, "f16 probe overlap {same}/2000");
    }

    #[test]
    fn grouped_table_layout_and_partial_routing() {
        let mut rng = Rng::new(3);
        let (k, dim, groups) = (300, 32, 12);
        let m = means(&mut rng, k, dim);
        let tree = CentroidTree::build(&m, dim, k, groups, 5, 1);
        let inv = inv_norms(&m, dim);
        for elem in [Elem::F32, Elem::F16, Elem::I8] {
            let bytes = encode_grouped(&m, dim, &tree, elem);
            let t = GroupedTable::parse(&bytes).unwrap();
            assert!(t.matches_tree(&tree));
            assert_eq!(t.byte_len(), bytes.len());
            assert_eq!(
                (t.elem, t.dim, t.k, t.groups, t.n_rows),
                (elem, dim, k, groups, k)
            );
            assert_eq!(t.data_off % 8, 0);
            // the directory alone parses too; a truncated object does not
            assert_eq!(GroupedTable::parse(&bytes[..t.data_off]).unwrap(), t);
            assert!(GroupedTable::parse(&bytes[..bytes.len() - 1]).is_err());
            let mut bad = bytes.clone();
            bad[0] = b'X';
            assert_eq!(GroupedTable::parse(&bad), Err(FormatError::BadMagic));
            // every row decodes to the cell's (quantized) mean
            let flat = decode_rows(&encode_rows(&m, dim, elem), dim, elem).unwrap();
            for g in 0..groups {
                let (off, len) = t.group_range(g);
                let rows = decode_rows(&bytes[off..off + len], dim, elem).unwrap();
                for (i, &c) in tree.members(g).iter().enumerate() {
                    let c = c as usize;
                    assert_eq!(&rows[i * dim..(i + 1) * dim], &flat[c * dim..(c + 1) * dim]);
                }
            }

            let mut pr = PartialRouter::new(tree.clone(), elem);
            assert_eq!(pr.loaded_cells(), 0);
            let q = unit(&mut rng, dim);
            // nothing loaded: no candidates, every top group missing
            assert!(pr.route_beam(&q, 4, 10).is_empty());
            assert_eq!(pr.missing_groups(&q, 4), pr.top_groups(&q, 4));
            for g in pr.missing_groups(&q, 4) {
                let (off, len) = t.group_range(g);
                assert!(pr.add_group(g, &bytes[off..off + len]).unwrap() > 0);
                assert!(pr.has_group(g));
                // idempotent
                assert_eq!(pr.add_group(g, &bytes[off..off + len]).unwrap(), 0);
            }
            assert!(pr.missing_groups(&q, 4).is_empty());
            let finv = inv_norms(&flat, dim);
            assert_eq!(
                pr.route_beam(&q, 4, 10),
                tree.route(&q, &flat, &finv, 4, 10)
            );
            if elem == Elem::F32 {
                assert_eq!(pr.route_beam(&q, 4, 10), tree.route(&q, &m, &inv, 4, 10));
            }
            // delta base scores: loaded cells only
            let c0 = tree.members(pr.top_groups(&q, 1)[0])[0];
            assert_eq!(
                pr.cell_dots_for(&q, &[c0]).unwrap(),
                vec![dot(&q, &flat[c0 as usize * dim..(c0 as usize + 1) * dim])]
            );
            let far = (0..k as u32).find(|&c| !pr.has_cell(c as usize)).unwrap();
            assert!(pr.cell_dots_for(&q, &[far]).is_err());
            // a single row (a delta's cell) by its row number
            let r = pr.cell_row(far as usize).unwrap();
            let rb = elem.row_bytes(dim);
            assert_eq!(
                pr.add_rows(r, &bytes[t.data_off + r * rb..t.data_off + (r + 1) * rb])
                    .unwrap(),
                1
            );
            assert!(pr.has_cell(far as usize));
            assert!(pr.cell_dots_for(&q, &[far]).is_ok());
            // everything loaded: full beam equals flat routing over the decoded table
            let all = &bytes[t.data_off..];
            pr.add_rows(0, all).unwrap();
            assert_eq!(pr.loaded_groups(), groups);
            let router = Router::new(dim, 1, k, &vec![1.0; dim], flat.clone());
            for _ in 0..20 {
                let q = unit(&mut rng, dim);
                assert_eq!(pr.route_beam(&q, groups, 17), router.route_flat(&q, 17));
            }
            assert!(pr.add_rows(k, &bytes[t.data_off..t.data_off + rb]).is_err());
            assert!(pr.add_group(0, &bytes[t.data_off..t.data_off + 1]).is_err());
        }
    }
}
