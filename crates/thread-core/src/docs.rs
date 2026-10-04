//! The `docs` section: document id (the index's u32 id) -> a metadata string (a title, a JSON
//! object, an external key: whatever the application wants back with a hit) and, optionally, the
//! document's text (for excerpts and reading). This is the "unified docs" store of an index: the
//! dense slabs and the lexical postings both carry the same ids, and this section names them.
//!
//! Ids are sparse and caller-assigned, so the section is bucketed by id behind a small directory
//! (one 16-byte entry per bucket, resident on hosts): resolving the hits of a query is a binary
//! search over the directory and one bucket read per hit (coalesced), then one range read per
//! text actually shown. Texts live in their own region after the buckets, so looking a title up
//! never pays for the text.
//!
//! ```text
//! Header, 64 bytes
//! 0   magic "THRDDOCS"
//! 8   version u32 = 2 (1 still read: no attrs)
//! 12  n u32              documents
//! 16  bucket count B u32 = ceil(n / bucket_docs)
//! 20  bucket_docs u32
//! 24  flags u32          bit 0: texts stored; bit 1: attrs stored (version 2)
//! 28  reserved u32
//! 32  dir_off u64 (= 64)   40 dir_len u64 (= 16 B)   48 buckets_off u64   56 buckets_len u64
//! directory: B entries (first id u32, bucket len u32, bucket offset u64), ascending by first id
//! bucket entry: id varint, [attrs varint, when bit 1], meta len varint, meta (UTF-8), text len varint,
//!   text offset varint (only when text len > 0; relative to the text region, which starts at
//!   align8(buckets_off + buckets_len) and runs to the end of the section)
//! ```
//!
//! `attrs` is the document's 32-bit attribute word, the same one its dense entry carries (see
//! `crate::filter`): a lexical query resolves it here to apply an attribute filter to its hits.
//! A writer sets bit 1 only when some document has a non-zero word, so the common case costs
//! nothing; readers of version 1 sections see every attrs as 0.

use std::collections::BTreeMap;

use crate::error::Error;
use crate::source::{ByteSource, Plan, Range, SourceError};
use crate::tomb::Tombstones;
use crate::varint;

pub const MAGIC: &[u8; 8] = b"THRDDOCS";
pub const VERSION: u32 = 2;
pub const HEADER_LEN: u64 = 64;
pub const DIR_ENTRY_LEN: u64 = 16;
/// Documents per bucket unless the builder is told otherwise: a bucket of titles is ~5 KB, one
/// request's worth; a million documents have a 125 KB directory.
pub const DEFAULT_BUCKET_DOCS: u32 = 128;
pub const PLAN_GAP: u64 = 4096;
const FLAG_TEXT: u32 = 1;
const FLAG_ATTRS: u32 = 2;

#[inline]
fn align8(x: u64) -> u64 {
    (x + 7) & !7
}
#[inline]
fn u32_at(buf: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(buf[o..o + 4].try_into().unwrap())
}
#[inline]
fn u64_at(buf: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(buf[o..o + 8].try_into().unwrap())
}

/// Builds a `docs` section from `(id, meta, text, attrs)` in any id order.
pub struct DocsBuilder {
    bucket_docs: u32,
    with_text: bool,
    docs: BTreeMap<u32, (String, String, u32)>,
}

impl Default for DocsBuilder {
    fn default() -> Self {
        DocsBuilder::new()
    }
}

impl DocsBuilder {
    pub fn new() -> DocsBuilder {
        DocsBuilder {
            bucket_docs: DEFAULT_BUCKET_DOCS,
            with_text: false,
            docs: BTreeMap::new(),
        }
    }
    /// Documents per bucket (default [`DEFAULT_BUCKET_DOCS`]).
    pub fn bucket_docs(mut self, n: u32) -> DocsBuilder {
        self.bucket_docs = n;
        self
    }
    /// Store the texts (default: metadata only; texts passed to `add` are then dropped).
    pub fn with_text(mut self, yes: bool) -> DocsBuilder {
        self.with_text = yes;
        self
    }
    pub fn stores_text(&self) -> bool {
        self.with_text
    }
    pub fn len(&self) -> u32 {
        self.docs.len() as u32
    }
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }
    pub fn contains(&self, id: u32) -> bool {
        self.docs.contains_key(&id)
    }
    /// Adds a document with attrs 0. Fails on a duplicate id or `id == u32::MAX`.
    pub fn add(&mut self, id: u32, meta: &str, text: Option<&str>) -> Result<(), Error> {
        self.add_attrs(id, meta, text, 0)
    }

    /// Adds a document with its attribute word. Fails on a duplicate id or `id == u32::MAX`.
    pub fn add_attrs(
        &mut self,
        id: u32,
        meta: &str,
        text: Option<&str>,
        attrs: u32,
    ) -> Result<(), Error> {
        if id == u32::MAX {
            return Err(Error::Format("document id u32::MAX is reserved".into()));
        }
        if self.docs.contains_key(&id) {
            return Err(Error::Format(format!("duplicate document id {id}")));
        }
        let text = if self.with_text {
            text.unwrap_or("").to_string()
        } else {
            String::new()
        };
        self.docs.insert(id, (meta.to_string(), text, attrs));
        Ok(())
    }

    /// Whether any document has a non-zero attribute word (the section then stores them).
    pub fn has_attrs(&self) -> bool {
        self.docs.values().any(|d| d.2 != 0)
    }

    pub fn write_to(&self, out: &mut Vec<u8>) -> Result<(), Error> {
        if self.bucket_docs == 0 {
            return Err(Error::Format("docs: bucket_docs must be > 0".into()));
        }
        let n = self.docs.len() as u64;
        let bd = self.bucket_docs as usize;
        let b_count = n.div_ceil(bd as u64);
        let dir_len = DIR_ENTRY_LEN * b_count;
        let buckets_off = align8(HEADER_LEN + dir_len);
        let start = out.len();
        out.resize(start + buckets_off as usize, 0);
        let with_attrs = self.has_attrs();
        let entries: Vec<(&u32, &(String, String, u32))> = self.docs.iter().collect();
        let mut dir: Vec<u8> = Vec::with_capacity(dir_len as usize);
        let mut texts: Vec<u8> = Vec::new();
        for chunk in entries.chunks(bd) {
            let b_start = out.len() as u64 - start as u64;
            for (id, (meta, text, attrs)) in chunk {
                varint::put(out, **id as u64);
                if with_attrs {
                    varint::put(out, *attrs as u64);
                }
                varint::put(out, meta.len() as u64);
                out.extend_from_slice(meta.as_bytes());
                varint::put(out, text.len() as u64);
                if !text.is_empty() {
                    varint::put(out, texts.len() as u64);
                    texts.extend_from_slice(text.as_bytes());
                }
            }
            let b_len = out.len() as u64 - start as u64 - b_start;
            if b_len > u32::MAX as u64 {
                return Err(Error::Format("docs: bucket longer than 4 GiB".into()));
            }
            dir.extend_from_slice(&chunk[0].0.to_le_bytes());
            dir.extend_from_slice(&(b_len as u32).to_le_bytes());
            dir.extend_from_slice(&b_start.to_le_bytes());
        }
        let buckets_len = out.len() as u64 - start as u64 - buckets_off;
        let text_off = align8(buckets_off + buckets_len);
        out.resize(start + text_off as usize, 0);
        out.extend_from_slice(&texts);
        let h = &mut out[start..start + HEADER_LEN as usize];
        h[0..8].copy_from_slice(MAGIC);
        h[8..12].copy_from_slice(&VERSION.to_le_bytes());
        h[12..16].copy_from_slice(&(n as u32).to_le_bytes());
        h[16..20].copy_from_slice(&(b_count as u32).to_le_bytes());
        h[20..24].copy_from_slice(&self.bucket_docs.to_le_bytes());
        let flags =
            if self.with_text { FLAG_TEXT } else { 0 } | if with_attrs { FLAG_ATTRS } else { 0 };
        h[24..28].copy_from_slice(&flags.to_le_bytes());
        h[32..40].copy_from_slice(&HEADER_LEN.to_le_bytes());
        h[40..48].copy_from_slice(&dir_len.to_le_bytes());
        h[48..56].copy_from_slice(&buckets_off.to_le_bytes());
        h[56..64].copy_from_slice(&buckets_len.to_le_bytes());
        let d = start + HEADER_LEN as usize;
        out[d..d + dir.len()].copy_from_slice(&dir);
        Ok(())
    }

    pub fn finish(&self) -> Result<Vec<u8>, Error> {
        let mut v = Vec::new();
        self.write_to(&mut v)?;
        Ok(v)
    }
}

/// `meta.docs` (`"attrs":true` only when the section stores attribute words, so the common
/// meta stays short enough for a container's open prefix).
pub fn meta_json(
    n: u32,
    with_text: bool,
    with_attrs: bool,
    bucket_docs: u32,
    nbytes: u64,
) -> String {
    let attrs = if with_attrs { r#","attrs":true"# } else { "" };
    format!(
        r#"{{"version":{VERSION},"n":{n},"text":{with_text}{attrs},"bucket_docs":{bucket_docs},"nbytes":{nbytes}}}"#
    )
}

pub fn is_docs(head: &[u8]) -> bool {
    head.len() >= 8 && &head[..8] == MAGIC
}

/// One document's entry: its metadata and, when the section stores texts, where its text is
/// (section-relative; empty for an empty text).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocEntry {
    pub id: u32,
    pub meta: String,
    pub text: Range,
    /// The document's attribute word (0 in sections without attrs).
    pub attrs: u32,
}

/// The `docs` section over any [`ByteSource`]. `open` reads the 64-byte header only.
pub struct Docs<S: ByteSource> {
    source: S,
    n: u32,
    buckets: u32,
    bucket_docs: u32,
    with_text: bool,
    with_attrs: bool,
    directory: Range,
    bucket_bytes: Range,
    text_bytes: Range,
}

impl<S: ByteSource> Docs<S> {
    pub fn open(source: S) -> Result<Docs<S>, Error> {
        let head = source.read(Range::new(0, HEADER_LEN))?;
        if !is_docs(&head) {
            return Err(Error::Format(
                "not a thread docs section (bad magic)".into(),
            ));
        }
        let version = u32_at(&head, 8);
        if version != 1 && version != VERSION {
            return Err(Error::Format(format!(
                "unsupported docs version {version} (expected 1 or {VERSION})"
            )));
        }
        let n = u32_at(&head, 12);
        let buckets = u32_at(&head, 16);
        let bucket_docs = u32_at(&head, 20);
        let flags = u32_at(&head, 24);
        let with_text = flags & FLAG_TEXT != 0;
        let with_attrs = version >= 2 && flags & FLAG_ATTRS != 0;
        let directory = Range::new(u64_at(&head, 32), u64_at(&head, 40));
        let bucket_bytes = Range::new(u64_at(&head, 48), u64_at(&head, 56));
        drop(head);
        let total = source.len();
        let expect = if bucket_docs == 0 {
            0
        } else {
            (n as u64).div_ceil(bucket_docs as u64)
        };
        let text_off = align8(bucket_bytes.end());
        if (n > 0 && bucket_docs == 0)
            || buckets as u64 != expect
            || directory.offset != HEADER_LEN
            || directory.len != DIR_ENTRY_LEN * buckets as u64
            || bucket_bytes.offset < directory.end()
            || bucket_bytes.offset.checked_add(bucket_bytes.len).is_none()
            || text_off > total
        {
            return Err(Error::Format("corrupt docs header".into()));
        }
        Ok(Docs {
            source,
            n,
            buckets,
            bucket_docs,
            with_text,
            with_attrs,
            directory,
            bucket_bytes,
            text_bytes: Range::new(text_off, total - text_off),
        })
    }

    pub fn source(&self) -> &S {
        &self.source
    }
    pub fn into_source(self) -> S {
        self.source
    }
    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn has_text(&self) -> bool {
        self.with_text
    }
    /// Whether the section stores attribute words (else every document's attrs is 0).
    pub fn has_attrs(&self) -> bool {
        self.with_attrs
    }
    pub fn bucket_count(&self) -> u32 {
        self.buckets
    }
    pub fn bucket_docs(&self) -> u32 {
        self.bucket_docs
    }
    pub fn nbytes(&self) -> u64 {
        self.source.len()
    }
    pub fn directory_nbytes(&self) -> u64 {
        self.directory.len
    }
    pub fn text_nbytes(&self) -> u64 {
        self.text_bytes.len
    }
    pub fn meta_json(&self) -> String {
        meta_json(
            self.n,
            self.with_text,
            self.with_attrs,
            self.bucket_docs,
            self.nbytes(),
        )
    }

    /// The directory: one range (blob-absolute) a host fetches once and keeps.
    pub fn plan_directory(&self) -> Plan {
        let mut p = Plan::new();
        p.push(self.directory);
        p.absolute(&self.source)
    }

    #[inline]
    fn dir_entry(dir: &[u8], i: usize) -> (u32, Range) {
        let o = i * DIR_ENTRY_LEN as usize;
        (
            u32_at(dir, o),
            Range::new(u64_at(dir, o + 8), u32_at(dir, o + 4) as u64),
        )
    }

    /// The bucket that may hold `id`: the last one whose first id is `<= id`.
    fn candidate(&self, dir: &[u8], id: u32) -> Option<usize> {
        let (mut lo, mut hi) = (0usize, self.buckets as usize);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if Self::dir_entry(dir, mid).0 <= id {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.checked_sub(1)
    }

    fn check_bucket(&self, r: Range) -> Result<Range, SourceError> {
        if r.offset < self.bucket_bytes.offset || r.end() > self.bucket_bytes.end() {
            return Err(SourceError::OutOfBounds(self.source.to_absolute(r)));
        }
        Ok(r)
    }

    /// Parses the entry at `pos` (advancing it); `None` on corrupt bytes.
    fn parse_entry(&self, buf: &[u8], pos: &mut usize) -> Option<DocEntry> {
        let id = u32::try_from(varint::read(buf, pos)).ok()?;
        let attrs = if self.with_attrs {
            u32::try_from(varint::read(buf, pos)).ok()?
        } else {
            0
        };
        let ml = varint::read(buf, pos) as usize;
        let meta = buf.get(*pos..pos.checked_add(ml)?)?;
        *pos += ml;
        let tl = varint::read(buf, pos);
        let text = if tl > 0 {
            let to = varint::read(buf, pos);
            if to.checked_add(tl)? > self.text_bytes.len {
                return None;
            }
            Range::new(self.text_bytes.offset + to, tl)
        } else {
            Range::new(self.text_bytes.offset, 0)
        };
        Some(DocEntry {
            id,
            meta: String::from_utf8_lossy(meta).into_owned(),
            text,
            attrs,
        })
    }

    fn find_in_bucket(&self, buf: &[u8], id: u32) -> Result<Option<DocEntry>, SourceError> {
        let corrupt = || SourceError::OutOfBounds(self.source.to_absolute(self.bucket_bytes));
        let mut pos = 0usize;
        while pos < buf.len() {
            let e = self.parse_entry(buf, &mut pos).ok_or_else(corrupt)?;
            match e.id.cmp(&id) {
                std::cmp::Ordering::Less => continue,
                std::cmp::Ordering::Equal => return Ok(Some(e)),
                std::cmp::Ordering::Greater => return Ok(None),
            }
        }
        Ok(None)
    }

    /// The bucket ranges `lookup(ids)` reads (`Missing` the directory until it is held).
    pub fn plan_lookup(&self, ids: &[u32]) -> Result<Plan, SourceError> {
        let dir = self.source.read(self.directory)?;
        let mut plan = Plan::new();
        for &id in ids {
            if let Some(i) = self.candidate(&dir, id) {
                plan.push(self.check_bucket(Self::dir_entry(&dir, i).1)?);
            }
        }
        plan.coalesce(PLAN_GAP);
        Ok(plan.absolute(&self.source))
    }

    /// The entries of `ids` (`None` when the section lacks the id). One round on a cold source
    /// once the directory is held: every bucket is attempted before failing.
    pub fn lookup(&self, ids: &[u32]) -> Result<Vec<Option<DocEntry>>, SourceError> {
        let dir = self.source.read(self.directory)?;
        let mut out = vec![None; ids.len()];
        let mut wants: Vec<(usize, usize)> = Vec::with_capacity(ids.len());
        for (k, &id) in ids.iter().enumerate() {
            if let Some(i) = self.candidate(&dir, id) {
                wants.push((i, k));
            }
        }
        wants.sort_unstable();
        let mut acc = Plan::new();
        let mut j = 0;
        while j < wants.len() {
            let bucket = wants[j].0;
            let r = self.check_bucket(Self::dir_entry(&dir, bucket).1)?;
            match self.source.read(r) {
                Ok(buf) => {
                    while j < wants.len() && wants[j].0 == bucket {
                        let k = wants[j].1;
                        out[k] = self.find_in_bucket(&buf, ids[k])?;
                        j += 1;
                    }
                }
                Err(SourceError::Missing(rs)) => {
                    acc.extend(rs);
                    while j < wants.len() && wants[j].0 == bucket {
                        j += 1;
                    }
                }
                Err(e) => return Err(e),
            }
        }
        if !acc.is_empty() {
            acc.coalesce(PLAN_GAP);
            return Err(SourceError::Missing(acc.into_ranges()));
        }
        Ok(out)
    }

    /// The text ranges of `entries`, blob-absolute and coalesced.
    pub fn plan_texts(&self, entries: &[DocEntry]) -> Plan {
        let mut p = Plan::new();
        for e in entries {
            p.push(e.text);
        }
        p.coalesce(PLAN_GAP);
        p.absolute(&self.source)
    }

    /// The texts of `entries` (empty strings when the section stores none). One round; every
    /// read is attempted before failing.
    pub fn texts(&self, entries: &[DocEntry]) -> Result<Vec<String>, SourceError> {
        let mut out = Vec::with_capacity(entries.len());
        let mut acc = Plan::new();
        for e in entries {
            if e.text.is_empty() {
                out.push(String::new());
                continue;
            }
            match self.source.read(e.text) {
                Ok(b) => out.push(String::from_utf8_lossy(&b).into_owned()),
                Err(SourceError::Missing(rs)) => {
                    acc.extend(rs);
                    out.push(String::new());
                }
                Err(e) => return Err(e),
            }
        }
        if !acc.is_empty() {
            acc.coalesce(PLAN_GAP);
            return Err(SourceError::Missing(acc.into_ranges()));
        }
        Ok(out)
    }

    /// `lookup` + `texts`: `(meta, text)` per id (`None` for unknown ids). Two rounds cold.
    pub fn get(&self, ids: &[u32]) -> Result<Vec<Option<(String, String)>>, SourceError> {
        let entries = self.lookup(ids)?;
        let present: Vec<DocEntry> = entries.iter().flatten().cloned().collect();
        let texts = self.texts(&present)?;
        let mut it = texts.into_iter();
        Ok(entries
            .into_iter()
            .map(|e| e.map(|e| (e.meta, it.next().unwrap_or_default())))
            .collect())
    }

    /// Every entry in id order (reads the whole bucket region): for merges and tools.
    pub fn entries(&self) -> Result<Vec<DocEntry>, SourceError> {
        let buf = self.source.read(self.bucket_bytes)?;
        let corrupt = || SourceError::OutOfBounds(self.source.to_absolute(self.bucket_bytes));
        let mut out = Vec::with_capacity(self.n as usize);
        let mut pos = 0usize;
        while pos < buf.len() {
            out.push(self.parse_entry(&buf, &mut pos).ok_or_else(corrupt)?);
        }
        Ok(out)
    }
}

/// Rebuilds one `docs` section from several segments given **oldest first**: for an id present
/// in several, the newest copy wins; ids in a segment's `skip` set are left out of that segment.
pub fn merge<S: ByteSource>(
    inputs: &[(&Docs<S>, Option<&Tombstones>)],
    bucket_docs: u32,
    with_text: bool,
) -> Result<DocsBuilder, Error> {
    let mut b = DocsBuilder::new()
        .bucket_docs(bucket_docs)
        .with_text(with_text);
    for (d, skip) in inputs.iter().rev() {
        let entries = d.entries()?;
        let live: Vec<DocEntry> = entries
            .into_iter()
            .filter(|e| !b.contains(e.id) && !skip.is_some_and(|t| t.contains(e.id)))
            .collect();
        let texts = if with_text && d.has_text() {
            d.texts(&live)?
        } else {
            vec![String::new(); live.len()]
        };
        for (e, t) in live.into_iter().zip(texts) {
            b.add_attrs(e.id, &e.meta, Some(&t), e.attrs)?;
        }
    }
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Prefetched;

    fn sample(with_text: bool, bucket_docs: u32) -> (Vec<u8>, Vec<(u32, String, String)>) {
        let mut rows: Vec<(u32, String, String)> = (0..300u32)
            .map(|i| {
                (
                    i * 7 + 3,
                    format!(r#"{{"title":"Doc {i} — ünïcode"}}"#),
                    if i % 10 == 0 {
                        String::new()
                    } else {
                        format!(
                            "Text of document {i}\nline two {}",
                            "pad ".repeat(i as usize % 40)
                        )
                    },
                )
            })
            .collect();
        rows.reverse(); // any order in
        let mut b = DocsBuilder::new()
            .bucket_docs(bucket_docs)
            .with_text(with_text);
        for (id, m, t) in &rows {
            b.add(*id, m, Some(t)).unwrap();
        }
        rows.sort();
        (b.finish().unwrap(), rows)
    }

    #[test]
    fn roundtrip_lookup_texts_entries() {
        for with_text in [true, false] {
            for bd in [1u32, 7, DEFAULT_BUCKET_DOCS, 10_000] {
                let (bytes, rows) = sample(with_text, bd);
                assert!(is_docs(&bytes));
                let d = Docs::open(bytes.as_slice()).unwrap();
                assert_eq!(d.n(), 300);
                assert_eq!(d.has_text(), with_text);
                assert_eq!(d.bucket_count() as u64, 300u64.div_ceil(bd as u64));
                assert_eq!(d.nbytes() as usize, bytes.len());
                if !with_text {
                    assert_eq!(d.text_nbytes(), 0);
                }
                let ids: Vec<u32> = rows.iter().map(|r| r.0).collect();
                let got = d.lookup(&ids).unwrap();
                for (r, g) in rows.iter().zip(&got) {
                    let g = g.as_ref().unwrap();
                    assert_eq!((g.id, &g.meta), (r.0, &r.1));
                    assert_eq!(g.text.len as usize, if with_text { r.2.len() } else { 0 });
                }
                let texts = d
                    .texts(&got.iter().flatten().cloned().collect::<Vec<_>>())
                    .unwrap();
                for (r, t) in rows.iter().zip(&texts) {
                    assert_eq!(
                        *t,
                        if with_text {
                            r.2.clone()
                        } else {
                            String::new()
                        }
                    );
                }
                let misses = [0u32, 1, 2, 4, 2095, 2097, 5000, u32::MAX - 1];
                assert_eq!(d.lookup(&misses).unwrap(), vec![None; misses.len()]);
                let mixed = d.get(&[3, 1, 2096, 10]).unwrap();
                assert_eq!(mixed[0].as_ref().unwrap().0, rows[0].1);
                assert!(mixed[1].is_none());
                assert_eq!(
                    mixed[2].as_ref().map(|x| x.1.clone()),
                    Some(if with_text {
                        rows[299].2.clone()
                    } else {
                        String::new()
                    })
                );
                assert_eq!(mixed[3].as_ref().unwrap().0, rows[1].1);
                let entries = d.entries().unwrap();
                assert_eq!(entries, got.into_iter().flatten().collect::<Vec<_>>());
                assert_eq!(
                    d.meta_json(),
                    meta_json(300, with_text, false, bd, bytes.len() as u64)
                );
                assert!(!d.has_attrs());
            }
        }
        let empty = DocsBuilder::new().finish().unwrap();
        let d = Docs::open(empty.as_slice()).unwrap();
        assert_eq!(d.n(), 0);
        assert_eq!(d.lookup(&[0]).unwrap(), vec![None]);
        assert!(d.entries().unwrap().is_empty());
    }

    #[test]
    fn attrs_roundtrip_merge_and_version_1() {
        let mut b = DocsBuilder::new().bucket_docs(3).with_text(true);
        for i in 0..20u32 {
            b.add_attrs(
                i * 5,
                &format!("m{i}"),
                Some(if i % 2 == 0 { "t" } else { "" }),
                i % 7,
            )
            .unwrap();
        }
        assert!(b.has_attrs());
        let bytes = b.finish().unwrap();
        let d = Docs::open(bytes.as_slice()).unwrap();
        assert!(d.has_attrs());
        assert_eq!(u32_at(&bytes, 8), VERSION);
        let ids: Vec<u32> = (0..20).map(|i| i * 5).collect();
        for (i, e) in d.lookup(&ids).unwrap().into_iter().enumerate() {
            let e = e.unwrap();
            assert_eq!(
                (e.id, e.attrs, e.meta.as_str()),
                (i as u32 * 5, i as u32 % 7, format!("m{i}").as_str())
            );
        }
        assert_eq!(d.get(&[10]).unwrap()[0], Some(("m2".into(), "t".into())));
        assert!(d.meta_json().contains(r#""attrs":true"#));
        // merge keeps the newest copy's attrs
        let mut nb = DocsBuilder::new();
        nb.add_attrs(10, "m2-new", None, 99).unwrap();
        let newer = nb.finish().unwrap();
        let dn = Docs::open(newer.as_slice()).unwrap();
        let m = merge(&[(&d, None), (&dn, None)], 4, false)
            .unwrap()
            .finish()
            .unwrap();
        let dm = Docs::open(m.as_slice()).unwrap();
        let got = dm.lookup(&[10, 15]).unwrap();
        assert_eq!(
            (
                got[0].as_ref().unwrap().attrs,
                got[0].as_ref().unwrap().meta.as_str()
            ),
            (99, "m2-new")
        );
        assert_eq!(got[1].as_ref().unwrap().attrs, 3);
        // a section whose documents all have attrs 0 is written without them (flag clear)
        let mut z = DocsBuilder::new();
        z.add_attrs(1, "a", None, 0).unwrap();
        let zb = z.finish().unwrap();
        assert!(!Docs::open(zb.as_slice()).unwrap().has_attrs());
        // a version-1 section (no flag bit, no attrs varint) reads with attrs 0
        let mut v1 = zb.clone();
        v1[8..12].copy_from_slice(&1u32.to_le_bytes());
        let d1 = Docs::open(v1.as_slice()).unwrap();
        assert!(!d1.has_attrs());
        assert_eq!(d1.lookup(&[1]).unwrap()[0].as_ref().unwrap().attrs, 0);
        // version 1 with the attrs flag set is still read as having none
        v1[24..28].copy_from_slice(&FLAG_ATTRS.to_le_bytes());
        assert!(!Docs::open(v1.as_slice()).unwrap().has_attrs());
    }

    #[test]
    fn errors_and_corruption() {
        let mut b = DocsBuilder::new();
        b.add(5, "a", None).unwrap();
        assert!(b.add(5, "b", None).is_err());
        assert!(b.add(u32::MAX, "b", None).is_err());
        assert!(DocsBuilder::new().bucket_docs(0).finish().is_err());
        let (bytes, _) = sample(true, 16);
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(matches!(Docs::open(bad.as_slice()), Err(Error::Format(_))));
        let mut bad = bytes.clone();
        bad[8] = 7;
        assert!(matches!(Docs::open(bad.as_slice()), Err(Error::Format(_))));
        let mut bad = bytes.clone();
        bad[16] = 1; // bucket count disagrees
        assert!(matches!(Docs::open(bad.as_slice()), Err(Error::Format(_))));
        assert!(matches!(Docs::open(&bytes[..40]), Err(Error::Source(_))));
        // a directory entry pointing outside the buckets is OutOfBounds, never a panic
        let mut bad = bytes.clone();
        bad[HEADER_LEN as usize + 15] = 255; // bucket 0's offset, top byte
        let d = Docs::open(bad.as_slice()).unwrap();
        assert!(matches!(d.lookup(&[3]), Err(SourceError::OutOfBounds(_))));
    }

    #[test]
    fn paged_lookup_and_texts_are_one_round_each_after_the_directory() {
        let (bytes, rows) = sample(true, 8);
        let base = 4096u64;
        let mut blob = vec![0u8; base as usize];
        blob.extend_from_slice(&bytes);
        let win = Range::new(base, bytes.len() as u64);
        let mut p = Prefetched::new(blob.len() as u64);
        let feed = |p: &mut Prefetched, rs: &[Range]| {
            for r in rs {
                p.insert(r.offset, blob[r.offset as usize..r.end() as usize].to_vec())
                    .unwrap();
            }
        };
        feed(&mut p, &[Range::new(base, HEADER_LEN)]);
        let ids = [rows[5].0, rows[6].0, rows[250].0, 1];
        let d = Docs::open(crate::source::SubSource::new(&p, win).unwrap()).unwrap();
        let dir = d.plan_directory();
        assert_eq!(
            d.lookup(&ids),
            Err(SourceError::Missing(dir.ranges().to_vec()))
        );
        feed(&mut p, dir.ranges());
        let d = Docs::open(crate::source::SubSource::new(&p, win).unwrap()).unwrap();
        let plan = d.plan_lookup(&ids).unwrap();
        assert_eq!(plan.len(), 2); // rows 5 and 6 share a bucket, 250 is another; id 1 sorts before everything
        assert_eq!(
            d.lookup(&ids),
            Err(SourceError::Missing(plan.ranges().to_vec()))
        );
        feed(&mut p, plan.ranges());
        let d = Docs::open(crate::source::SubSource::new(&p, win).unwrap()).unwrap();
        let got = d.lookup(&ids).unwrap();
        assert_eq!(got[0].as_ref().unwrap().meta, rows[5].1);
        assert!(got[3].is_none());
        let present: Vec<DocEntry> = got.iter().flatten().cloned().collect();
        let tplan = d.plan_texts(&present);
        assert_eq!(
            d.texts(&present),
            Err(SourceError::Missing(tplan.ranges().to_vec()))
        );
        feed(&mut p, tplan.ranges());
        let d = Docs::open(crate::source::SubSource::new(&p, win).unwrap()).unwrap();
        assert_eq!(
            d.texts(&present).unwrap(),
            vec![rows[5].2.clone(), rows[6].2.clone(), rows[250].2.clone()]
        );
        assert!(p.held_bytes() < bytes.len() as u64 / 3);
    }

    #[test]
    fn merge_newest_wins_and_tombstones_drop() {
        let mut a = DocsBuilder::new().with_text(true);
        a.add(1, "a1", Some("text a1")).unwrap();
        a.add(2, "a2", Some("text a2")).unwrap();
        a.add(3, "a3", Some("")).unwrap();
        let mut b = DocsBuilder::new().with_text(true);
        b.add(2, "b2", Some("text b2")).unwrap();
        b.add(4, "b4", Some("text b4")).unwrap();
        let (ab, bb) = (a.finish().unwrap(), b.finish().unwrap());
        let (da, db) = (
            Docs::open(ab.as_slice()).unwrap(),
            Docs::open(bb.as_slice()).unwrap(),
        );
        let dead = Tombstones::from_ids(&[3]);
        let m = merge(&[(&da, Some(&dead)), (&db, None)], 2, true)
            .unwrap()
            .finish()
            .unwrap();
        let d = Docs::open(m.as_slice()).unwrap();
        let got = d.get(&[1, 2, 3, 4]).unwrap();
        assert_eq!(got[0], Some(("a1".into(), "text a1".into())));
        assert_eq!(got[1], Some(("b2".into(), "text b2".into())));
        assert_eq!(got[2], None);
        assert_eq!(got[3], Some(("b4".into(), "text b4".into())));
        // merged without texts
        let m2 = merge(&[(&da, None), (&db, None)], 2, false)
            .unwrap()
            .finish()
            .unwrap();
        let d2 = Docs::open(m2.as_slice()).unwrap();
        assert!(!d2.has_text());
        assert_eq!(d2.get(&[2]).unwrap()[0], Some(("b2".into(), String::new())));
    }
}
