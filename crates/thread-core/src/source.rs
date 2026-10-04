//! Where index bytes come from (ported from Wiper's thread engine).
//!
//! Nothing in the query code assumes the whole blob is one contiguous slice. Every read goes
//! through [`ByteSource::read`], which hands back a `Cow<[u8]>`: a borrowed slice for in-memory
//! and memory-mapped sources (zero copy), and for a [`Prefetched`] source either a borrowed slice
//! of a piece the host already fetched or [`SourceError::Missing`] naming the ranges it has to
//! fetch first (the demand-paging protocol, see `DESIGN.md`).
//!
//! Coordinates: a source's own offsets run from 0 to [`ByteSource::len`]. A [`SubSource`] is a
//! window over another source; `read` takes window-relative offsets but every `Missing` error it
//! propagates, and every [`Plan`] an index computes through [`ByteSource::to_absolute`], is in the
//! coordinates of the outermost source, which is the one the host feeds. So plans and `Missing`
//! ranges are always "blob-absolute" and can be fetched as they are.

use std::borrow::Cow;
use std::fmt;

/// A byte range `[offset, offset + len)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Range {
    pub offset: u64,
    pub len: u64,
}

impl Range {
    #[inline]
    pub const fn new(offset: u64, len: u64) -> Range {
        Range { offset, len }
    }
    /// `[start, end)`.
    #[inline]
    pub const fn span(start: u64, end: u64) -> Range {
        Range {
            offset: start,
            len: end.saturating_sub(start),
        }
    }
    #[inline]
    pub const fn end(&self) -> u64 {
        self.offset + self.len
    }
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Whether `other` lies entirely inside `self`.
    #[inline]
    pub const fn contains(&self, other: &Range) -> bool {
        other.offset >= self.offset && other.end() <= self.end()
    }
    #[inline]
    pub const fn intersects(&self, other: &Range) -> bool {
        self.offset < other.end() && other.offset < self.end()
    }
    /// The same range moved by `delta` bytes.
    #[inline]
    pub const fn shifted(&self, delta: u64) -> Range {
        Range {
            offset: self.offset + delta,
            len: self.len,
        }
    }
}

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {})", self.offset, self.end())
    }
}

/// Why a read could not be served.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceError {
    /// The host has to fetch these (blob-absolute, coalesced) ranges and insert them into the
    /// [`Prefetched`] source, then call again.
    Missing(Vec<Range>),
    /// The range lies outside the source. On a validated index this means the index is corrupt.
    OutOfBounds(Range),
    /// An I/O failure (mmap, file read).
    Io(String),
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SourceError::Missing(ranges) => {
                let total: u64 = ranges.iter().map(|r| r.len).sum();
                write!(
                    f,
                    "{} byte range(s) ({total} bytes) not fetched yet",
                    ranges.len()
                )
            }
            SourceError::OutOfBounds(r) => write!(f, "byte range {r} is outside the source"),
            SourceError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for SourceError {}

impl From<std::io::Error> for SourceError {
    fn from(e: std::io::Error) -> Self {
        SourceError::Io(e.to_string())
    }
}

/// Random-access bytes. Implemented by slices, vectors, [`Prefetched`], [`SubSource`] and
/// [`Window`]. `read` must return exactly `range.len` bytes or an error.
pub trait ByteSource {
    /// Total size in bytes (for a [`Prefetched`] source: the size the host declared).
    fn len(&self) -> u64;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bytes of `range`. Borrowed whenever the source holds them contiguously.
    fn read(&self, range: Range) -> Result<Cow<'_, [u8]>, SourceError>;

    /// Translates a range in this source's coordinates to the outermost source's coordinates
    /// (identity except for [`SubSource`] windows). Plans are built through this so that the
    /// host can fetch them as they are.
    #[inline]
    fn to_absolute(&self, range: Range) -> Range {
        range
    }

    /// `[0, len)`.
    fn whole(&self) -> Range {
        Range::new(0, self.len())
    }
}

#[inline]
fn check_bounds(len: u64, range: Range) -> Result<(), SourceError> {
    if range
        .offset
        .checked_add(range.len)
        .is_some_and(|end| end <= len)
    {
        Ok(())
    } else {
        Err(SourceError::OutOfBounds(range))
    }
}

impl ByteSource for [u8] {
    #[inline]
    fn len(&self) -> u64 {
        <[u8]>::len(self) as u64
    }
    #[inline]
    fn read(&self, range: Range) -> Result<Cow<'_, [u8]>, SourceError> {
        check_bounds(<[u8]>::len(self) as u64, range)?;
        Ok(Cow::Borrowed(
            &self[range.offset as usize..range.end() as usize],
        ))
    }
}

impl ByteSource for Vec<u8> {
    #[inline]
    fn len(&self) -> u64 {
        Vec::len(self) as u64
    }
    #[inline]
    fn read(&self, range: Range) -> Result<Cow<'_, [u8]>, SourceError> {
        self.as_slice().read(range)
    }
}

impl<S: ByteSource + ?Sized> ByteSource for &S {
    #[inline]
    fn len(&self) -> u64 {
        (**self).len()
    }
    #[inline]
    fn read(&self, range: Range) -> Result<Cow<'_, [u8]>, SourceError> {
        (**self).read(range)
    }
    #[inline]
    fn to_absolute(&self, range: Range) -> Range {
        (**self).to_absolute(range)
    }
}

impl<S: ByteSource + ?Sized> ByteSource for Box<S> {
    #[inline]
    fn len(&self) -> u64 {
        (**self).len()
    }
    #[inline]
    fn read(&self, range: Range) -> Result<Cow<'_, [u8]>, SourceError> {
        (**self).read(range)
    }
    #[inline]
    fn to_absolute(&self, range: Range) -> Range {
        (**self).to_absolute(range)
    }
}

impl<S: ByteSource + ?Sized> ByteSource for std::sync::Arc<S> {
    #[inline]
    fn len(&self) -> u64 {
        (**self).len()
    }
    #[inline]
    fn read(&self, range: Range) -> Result<Cow<'_, [u8]>, SourceError> {
        (**self).read(range)
    }
    #[inline]
    fn to_absolute(&self, range: Range) -> Range {
        (**self).to_absolute(range)
    }
}

/// A window `[base, base + len)` over another source, so a container section can be addressed
/// with offsets relative to itself. `Missing` errors from the inner source and
/// [`ByteSource::to_absolute`] stay in the inner source's coordinates; `OutOfBounds` raised by
/// the window itself names the window-relative range.
#[derive(Clone, Copy, Debug)]
pub struct SubSource<'a, S: ByteSource + ?Sized> {
    inner: &'a S,
    base: u64,
    len: u64,
}

impl<'a, S: ByteSource + ?Sized> SubSource<'a, S> {
    /// Fails with `OutOfBounds` when the window does not fit inside `inner`.
    pub fn new(inner: &'a S, window: Range) -> Result<SubSource<'a, S>, SourceError> {
        check_bounds(inner.len(), window)?;
        Ok(SubSource {
            inner,
            base: window.offset,
            len: window.len,
        })
    }
    #[inline]
    pub fn base(&self) -> u64 {
        self.base
    }
    #[inline]
    pub fn inner(&self) -> &'a S {
        self.inner
    }
    /// The window as a range of the inner source.
    #[inline]
    pub fn window(&self) -> Range {
        Range::new(self.base, self.len)
    }
}

impl<S: ByteSource + ?Sized> ByteSource for SubSource<'_, S> {
    #[inline]
    fn len(&self) -> u64 {
        self.len
    }
    #[inline]
    fn read(&self, range: Range) -> Result<Cow<'_, [u8]>, SourceError> {
        check_bounds(self.len, range)?;
        self.inner.read(range.shifted(self.base))
    }
    #[inline]
    fn to_absolute(&self, range: Range) -> Range {
        self.inner.to_absolute(range.shifted(self.base))
    }
}

/// An owned window `[base, base + len)` over a source it holds (typically an `Arc<Vec<u8>>` shared
/// by every section reader of one blob), for callers that cannot borrow. Same coordinate rules
/// as [`SubSource`].
#[derive(Clone, Debug)]
pub struct Window<S: ByteSource> {
    inner: S,
    base: u64,
    len: u64,
}

impl<S: ByteSource> Window<S> {
    /// Fails with `OutOfBounds` when the window does not fit inside `inner`.
    pub fn new(inner: S, window: Range) -> Result<Window<S>, SourceError> {
        check_bounds(inner.len(), window)?;
        Ok(Window {
            inner,
            base: window.offset,
            len: window.len,
        })
    }
    #[inline]
    pub fn base(&self) -> u64 {
        self.base
    }
    #[inline]
    pub fn inner(&self) -> &S {
        &self.inner
    }
    /// The window as a range of the inner source.
    #[inline]
    pub fn window(&self) -> Range {
        Range::new(self.base, self.len)
    }
}

impl<S: ByteSource> ByteSource for Window<S> {
    #[inline]
    fn len(&self) -> u64 {
        self.len
    }
    #[inline]
    fn read(&self, range: Range) -> Result<Cow<'_, [u8]>, SourceError> {
        check_bounds(self.len, range)?;
        self.inner.read(range.shifted(self.base))
    }
    #[inline]
    fn to_absolute(&self, range: Range) -> Range {
        self.inner.to_absolute(range.shifted(self.base))
    }
}

/// A set of byte ranges a query needs. Built by the `plan_*` functions of each index, handed to
/// the host, which fetches them and feeds a [`Prefetched`] source.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    ranges: Vec<Range>,
}

impl Plan {
    pub fn new() -> Plan {
        Plan::default()
    }
    pub fn push(&mut self, range: Range) {
        if range.len > 0 {
            self.ranges.push(range);
        }
    }
    pub fn extend<I: IntoIterator<Item = Range>>(&mut self, ranges: I) {
        for r in ranges {
            self.push(r);
        }
    }
    pub fn append(&mut self, other: Plan) {
        self.ranges.extend(other.ranges);
    }
    pub fn ranges(&self) -> &[Range] {
        &self.ranges
    }
    pub fn into_ranges(self) -> Vec<Range> {
        self.ranges
    }
    pub fn iter(&self) -> std::slice::Iter<'_, Range> {
        self.ranges.iter()
    }
    pub fn len(&self) -> usize {
        self.ranges.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
    /// Bytes the plan covers (after `coalesce`, bytes the host will fetch).
    pub fn total_bytes(&self) -> u64 {
        self.ranges.iter().map(|r| r.len).sum()
    }
    /// Sorts, drops empty and duplicate ranges and ranges fully inside another one.
    pub fn dedup(&mut self) -> &mut Plan {
        self.ranges.retain(|r| r.len > 0);
        self.ranges.sort_unstable();
        let mut out: Vec<Range> = Vec::with_capacity(self.ranges.len());
        for r in self.ranges.drain(..) {
            if let Some(last) = out.last() {
                if last.contains(&r) {
                    continue;
                }
            }
            out.push(r);
        }
        self.ranges = out;
        self
    }
    /// Sorts and merges ranges whose gap is at most `max_gap` bytes (0 merges only overlapping
    /// or adjacent ranges). One fetch per remaining range is the host's cost model.
    pub fn coalesce(&mut self, max_gap: u64) -> &mut Plan {
        self.dedup();
        let mut out: Vec<Range> = Vec::with_capacity(self.ranges.len());
        for r in self.ranges.drain(..) {
            if let Some(last) = out.last_mut() {
                if r.offset <= last.end().saturating_add(max_gap) {
                    let end = last.end().max(r.end());
                    last.len = end - last.offset;
                    continue;
                }
            }
            out.push(r);
        }
        self.ranges = out;
        self
    }
    /// Moves every range by `delta` bytes (a section's base offset).
    pub fn shift(&mut self, delta: u64) -> &mut Plan {
        for r in &mut self.ranges {
            r.offset += delta;
        }
        self
    }
    /// Maps every range through `source.to_absolute`.
    pub fn absolute<S: ByteSource + ?Sized>(mut self, source: &S) -> Plan {
        for r in &mut self.ranges {
            *r = source.to_absolute(*r);
        }
        self
    }
}

impl From<Vec<Range>> for Plan {
    fn from(ranges: Vec<Range>) -> Plan {
        Plan { ranges }
    }
}

impl IntoIterator for Plan {
    type Item = Range;
    type IntoIter = std::vec::IntoIter<Range>;
    fn into_iter(self) -> Self::IntoIter {
        self.ranges.into_iter()
    }
}

impl<'a> IntoIterator for &'a Plan {
    type Item = &'a Range;
    type IntoIter = std::slice::Iter<'a, Range>;
    fn into_iter(self) -> Self::IntoIter {
        self.ranges.iter()
    }
}

/// One fetched piece of a [`Prefetched`] source.
#[derive(Clone, Debug)]
struct Piece {
    offset: u64,
    data: Vec<u8>,
}

impl Piece {
    #[inline]
    fn end(&self) -> u64 {
        self.offset + self.data.len() as u64
    }
    #[inline]
    fn range(&self) -> Range {
        Range::new(self.offset, self.data.len() as u64)
    }
}

/// A source made of the byte ranges a host has fetched so far (sorted, non-overlapping pieces;
/// touching or overlapping inserts are merged into one piece). `read` serves any range that
/// lies inside one piece and answers `Missing(vec![range])` otherwise, so an index query over it
/// either succeeds or tells the host what to fetch.
#[derive(Clone, Debug)]
pub struct Prefetched {
    len: u64,
    pieces: Vec<Piece>,
}

impl Prefetched {
    /// An empty source of `len` bytes (the blob's size, e.g. from an HTTP HEAD).
    pub fn new(len: u64) -> Prefetched {
        Prefetched {
            len,
            pieces: Vec::new(),
        }
    }

    /// Inserts `bytes` at `offset`, merging with pieces it touches or overlaps (new bytes win).
    /// Fails with `OutOfBounds` when the piece ends past the declared length.
    pub fn insert(&mut self, offset: u64, bytes: Vec<u8>) -> Result<(), SourceError> {
        let new = Range::new(offset, bytes.len() as u64);
        check_bounds(self.len, new)?;
        if new.is_empty() {
            return Ok(());
        }
        // Pieces touching or overlapping [offset, end): the first with end >= offset, up to the
        // last with offset <= end.
        let first = self.pieces.partition_point(|p| p.end() < new.offset);
        let last = self.pieces.partition_point(|p| p.offset <= new.end());
        if first >= last {
            self.pieces.insert(
                first,
                Piece {
                    offset,
                    data: bytes,
                },
            );
            return Ok(());
        }
        let start = self.pieces[first].offset.min(new.offset);
        let end = self.pieces[last - 1].end().max(new.end());
        if first + 1 == last && self.pieces[first].range().contains(&new) {
            let p = &mut self.pieces[first];
            let at = (new.offset - p.offset) as usize;
            p.data[at..at + bytes.len()].copy_from_slice(&bytes);
            return Ok(());
        }
        let mut data = vec![0u8; (end - start) as usize];
        for p in &self.pieces[first..last] {
            let at = (p.offset - start) as usize;
            data[at..at + p.data.len()].copy_from_slice(&p.data);
        }
        let at = (new.offset - start) as usize;
        data[at..at + bytes.len()].copy_from_slice(&bytes);
        self.pieces.splice(
            first..last,
            [Piece {
                offset: start,
                data,
            }],
        );
        Ok(())
    }

    /// Inserts every `(offset, bytes)` pair.
    pub fn insert_all<I: IntoIterator<Item = (u64, Vec<u8>)>>(
        &mut self,
        pieces: I,
    ) -> Result<(), SourceError> {
        for (offset, bytes) in pieces {
            self.insert(offset, bytes)?;
        }
        Ok(())
    }

    /// Forgets the bytes inside `range` (pieces are split as needed), to bound memory.
    pub fn evict(&mut self, range: Range) {
        if range.is_empty() {
            return;
        }
        let mut out: Vec<Piece> = Vec::with_capacity(self.pieces.len() + 1);
        for p in self.pieces.drain(..) {
            if !p.range().intersects(&range) {
                out.push(p);
                continue;
            }
            if p.offset < range.offset {
                let keep = (range.offset - p.offset) as usize;
                out.push(Piece {
                    offset: p.offset,
                    data: p.data[..keep].to_vec(),
                });
            }
            if p.end() > range.end() {
                let from = (range.end() - p.offset) as usize;
                out.push(Piece {
                    offset: range.end(),
                    data: p.data[from..].to_vec(),
                });
            }
        }
        self.pieces = out;
    }

    /// Forgets everything.
    pub fn clear(&mut self) {
        self.pieces.clear();
    }

    /// Bytes currently held.
    pub fn held_bytes(&self) -> u64 {
        self.pieces.iter().map(|p| p.data.len() as u64).sum()
    }

    /// The held pieces, ascending.
    pub fn pieces(&self) -> impl Iterator<Item = (Range, &[u8])> + '_ {
        self.pieces.iter().map(|p| (p.range(), p.data.as_slice()))
    }

    /// Whether `range` is fully held.
    pub fn holds(&self, range: Range) -> bool {
        self.missing(&[range]).is_empty()
    }

    /// The parts of `ranges` not held yet (sorted, merged), i.e. what the host still has to
    /// fetch before a query planned with these ranges can run. Ranges outside the source are
    /// clipped to it.
    pub fn missing(&self, ranges: &[Range]) -> Vec<Range> {
        let mut gaps = Plan::new();
        for r in ranges {
            let end = r.end().min(self.len);
            let mut cursor = r.offset.min(end);
            let mut i = self.pieces.partition_point(|p| p.end() <= cursor);
            while cursor < end {
                match self.pieces.get(i) {
                    Some(p) if p.offset < end => {
                        if p.offset > cursor {
                            gaps.push(Range::span(cursor, p.offset));
                        }
                        cursor = cursor.max(p.end());
                        i += 1;
                    }
                    _ => {
                        gaps.push(Range::span(cursor, end));
                        break;
                    }
                }
            }
        }
        gaps.coalesce(0);
        gaps.into_ranges()
    }
}

impl ByteSource for Prefetched {
    #[inline]
    fn len(&self) -> u64 {
        self.len
    }
    fn read(&self, range: Range) -> Result<Cow<'_, [u8]>, SourceError> {
        check_bounds(self.len, range)?;
        if range.is_empty() {
            return Ok(Cow::Borrowed(&[]));
        }
        let i = self.pieces.partition_point(|p| p.offset <= range.offset);
        if i > 0 {
            let p = &self.pieces[i - 1];
            if p.end() >= range.end() {
                let at = (range.offset - p.offset) as usize;
                return Ok(Cow::Borrowed(&p.data[at..at + range.len as usize]));
            }
        }
        Err(SourceError::Missing(vec![range]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_reads_are_borrowed_and_bounded() {
        let v: Vec<u8> = (0..32).collect();
        let got = v.read(Range::new(4, 4)).unwrap();
        assert!(matches!(got, Cow::Borrowed(_)));
        assert_eq!(&*got, &[4, 5, 6, 7]);
        assert_eq!(
            v.read(Range::new(30, 4)),
            Err(SourceError::OutOfBounds(Range::new(30, 4)))
        );
        assert_eq!(
            v.read(Range::new(u64::MAX, 4)),
            Err(SourceError::OutOfBounds(Range::new(u64::MAX, 4)))
        );
        assert_eq!(&*v.read(Range::new(32, 0)).unwrap(), &[] as &[u8]);
        let s: &[u8] = &v;
        assert_eq!(s.len(), 32);
        assert_eq!(&*s.read(Range::new(0, 2)).unwrap(), &[0, 1]);
    }

    #[test]
    fn sub_source_windows_and_absolute() {
        let v: Vec<u8> = (0..64).collect();
        let sub = SubSource::new(&v, Range::new(16, 32)).unwrap();
        assert_eq!(sub.len(), 32);
        assert_eq!(&*sub.read(Range::new(0, 2)).unwrap(), &[16, 17]);
        assert_eq!(
            sub.read(Range::new(31, 2)),
            Err(SourceError::OutOfBounds(Range::new(31, 2)))
        );
        assert_eq!(sub.to_absolute(Range::new(3, 5)), Range::new(19, 5));
        let nested = SubSource::new(&sub, Range::new(8, 8)).unwrap();
        assert_eq!(&*nested.read(Range::new(0, 1)).unwrap(), &[24]);
        assert_eq!(nested.to_absolute(Range::new(1, 1)), Range::new(25, 1));
        assert!(SubSource::new(&v, Range::new(60, 8)).is_err());
    }

    #[test]
    fn plan_coalesce_and_dedup() {
        let mut p = Plan::new();
        p.extend([
            Range::new(100, 10),
            Range::new(0, 10),
            Range::new(8, 4),
            Range::new(50, 0),
            Range::new(0, 10),
            Range::new(2, 2),
        ]);
        p.dedup();
        assert_eq!(
            p.ranges(),
            &[Range::new(0, 10), Range::new(8, 4), Range::new(100, 10)]
        );
        let mut c = p.clone();
        c.coalesce(0);
        assert_eq!(c.ranges(), &[Range::new(0, 12), Range::new(100, 10)]);
        assert_eq!(c.total_bytes(), 22);
        let mut c = p.clone();
        c.coalesce(88);
        assert_eq!(c.ranges(), &[Range::new(0, 110)]);
        c.shift(5);
        assert_eq!(c.ranges(), &[Range::new(5, 110)]);
    }

    #[test]
    fn prefetched_serves_merges_and_reports_missing() {
        let blob: Vec<u8> = (0..200u32).map(|i| (i % 251) as u8).collect();
        let mut p = Prefetched::new(200);
        assert_eq!(
            p.read(Range::new(0, 8)),
            Err(SourceError::Missing(vec![Range::new(0, 8)]))
        );
        assert_eq!(
            p.missing(&[Range::new(0, 8), Range::new(4, 20)]),
            vec![Range::new(0, 24)]
        );
        p.insert(10, blob[10..20].to_vec()).unwrap();
        p.insert(30, blob[30..40].to_vec()).unwrap();
        assert_eq!(p.pieces().count(), 2);
        assert_eq!(&*p.read(Range::new(12, 4)).unwrap(), &blob[12..16]);
        assert_eq!(
            p.read(Range::new(18, 4)),
            Err(SourceError::Missing(vec![Range::new(18, 4)]))
        );
        assert_eq!(
            p.missing(&[Range::new(0, 50)]),
            vec![Range::new(0, 10), Range::new(20, 10), Range::new(40, 10)]
        );
        assert_eq!(p.missing(&[Range::new(12, 4)]), vec![]);
        // Touching + overlapping inserts merge into one piece with the new bytes winning.
        p.insert(20, blob[20..30].to_vec()).unwrap();
        assert_eq!(p.pieces().count(), 1);
        assert_eq!(&*p.read(Range::new(10, 30)).unwrap(), &blob[10..40]);
        p.insert(5, blob[5..15].to_vec()).unwrap();
        p.insert(35, blob[35..45].to_vec()).unwrap();
        assert_eq!(p.pieces().count(), 1);
        assert_eq!(&*p.read(Range::new(5, 40)).unwrap(), &blob[5..45]);
        assert_eq!(p.held_bytes(), 40);
        // Out of bounds is not Missing.
        assert_eq!(
            p.read(Range::new(195, 10)),
            Err(SourceError::OutOfBounds(Range::new(195, 10)))
        );
        assert!(p.insert(195, vec![0; 10]).is_err());
        assert_eq!(
            p.missing(&[Range::new(190, 100)]),
            vec![Range::new(190, 10)]
        );
        // Evicting the middle splits the piece.
        p.evict(Range::new(20, 10));
        assert_eq!(
            p.pieces().map(|(r, _)| r).collect::<Vec<_>>(),
            vec![Range::new(5, 15), Range::new(30, 15)]
        );
        assert_eq!(&*p.read(Range::new(30, 15)).unwrap(), &blob[30..45]);
        assert_eq!(p.missing(&[Range::new(5, 40)]), vec![Range::new(20, 10)]);
        p.clear();
        assert_eq!(p.held_bytes(), 0);
        p.insert_all([(0, blob[0..100].to_vec()), (100, blob[100..200].to_vec())])
            .unwrap();
        assert_eq!(p.pieces().count(), 1);
        assert_eq!(&*p.read(Range::new(0, 200)).unwrap(), &blob[..]);
    }
}
