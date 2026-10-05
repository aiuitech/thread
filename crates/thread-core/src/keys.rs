//! The `keys` section: a unique string key (an external address, a path, a URL, a primary key)
//! -> the index's u32 id. It is the directory that makes `get(key)` a point lookup on an index
//! whose ids are u32: one bucket read per key once the small directory is resident, then the
//! `docs` bucket of the id it names. The section is sorted by key bytes, so a prefix listing is
//! a binary search to the first bucket and a forward scan.
//!
//! The layout is the `terms` directory of `crate::lex` with an id in place of a posting range:
//!
//! ```text
//! Header, 64 bytes
//! 0   magic "THRDKEYS"
//! 8   version u32 = 1
//! 12  n u32              keys
//! 16  bucket count B u32 = ceil(n / bucket_keys)
//! 20  bucket_keys u32
//! 24  flags u32 = 0
//! 28  reserved u32
//! 32  dir_off u64 (= 64)   40 dir_len u64   48 buckets_off u64   56 buckets_len u64
//! directory: B x 16 bytes (bucket offset u64, bucket len u32, first-key offset u32 relative to the
//!   directory), then every bucket's first key back to back (full keys, so a binary search over the
//!   directory names exactly one bucket)
//! bucket entry: key len varint, key (UTF-8), id varint
//! ```
//!
//! Keys are unique within a section. Across the segments of an index the newest live entry of a
//! key wins (a replaced key is written again in a newer segment under the same id; a deleted id
//! hides its key through the tombstone rule), exactly like `docs` entries.

use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::error::Error;
use crate::source::{ByteSource, Plan, Range, SourceError};
use crate::tomb::Tombstones;
use crate::varint;

pub const MAGIC: &[u8; 8] = b"THRDKEYS";
pub const VERSION: u32 = 1;
pub const HEADER_LEN: u64 = 64;
pub const DIR_ENTRY_LEN: u64 = 16;
/// Keys per bucket unless the builder is told otherwise: a bucket of 60-byte keys is ~8 KB, one
/// request's worth; a million keys have a ~600 KB directory (16 B + one key per bucket).
pub const DEFAULT_BUCKET_KEYS: u32 = 128;
pub const PLAN_GAP: u64 = 4096;
/// Longest key accepted (bytes).
pub const MAX_KEY_LEN: usize = 4096;

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

/// Builds a `keys` section from `(key, id)` pairs in any order.
pub struct KeysBuilder {
    bucket_keys: u32,
    keys: BTreeMap<Vec<u8>, u32>,
}

impl Default for KeysBuilder {
    fn default() -> Self {
        KeysBuilder::new()
    }
}

impl KeysBuilder {
    pub fn new() -> KeysBuilder {
        KeysBuilder {
            bucket_keys: DEFAULT_BUCKET_KEYS,
            keys: BTreeMap::new(),
        }
    }
    /// Keys per bucket (default [`DEFAULT_BUCKET_KEYS`]).
    pub fn bucket_keys(mut self, n: u32) -> KeysBuilder {
        self.bucket_keys = n;
        self
    }
    pub fn len(&self) -> u32 {
        self.keys.len() as u32
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    pub fn contains(&self, key: &str) -> bool {
        self.keys.contains_key(key.as_bytes())
    }
    /// Adds a key. Fails on a duplicate key, an empty or over-long key, or `id == u32::MAX`.
    pub fn add(&mut self, key: &str, id: u32) -> Result<(), Error> {
        if id == u32::MAX {
            return Err(Error::Format("document id u32::MAX is reserved".into()));
        }
        if key.is_empty() {
            return Err(Error::Format("keys: an empty key".into()));
        }
        if key.len() > MAX_KEY_LEN {
            return Err(Error::Format(format!(
                "keys: a key of {} bytes (at most {MAX_KEY_LEN})",
                key.len()
            )));
        }
        if self.keys.contains_key(key.as_bytes()) {
            return Err(Error::Format(format!("duplicate key {key:?}")));
        }
        self.keys.insert(key.as_bytes().to_vec(), id);
        Ok(())
    }

    pub fn write_to(&self, out: &mut Vec<u8>) -> Result<(), Error> {
        if self.bucket_keys == 0 {
            return Err(Error::Format("keys: bucket_keys must be > 0".into()));
        }
        let n = self.keys.len() as u64;
        let bk = self.bucket_keys as usize;
        let b_count = n.div_ceil(bk as u64);
        let entries: Vec<(&Vec<u8>, &u32)> = self.keys.iter().collect();
        let chunks: Vec<&[(&Vec<u8>, &u32)]> = entries.chunks(bk).collect();
        let first_keys: u64 = chunks.iter().map(|c| c[0].0.len() as u64).sum();
        let dir_len = DIR_ENTRY_LEN * b_count + first_keys;
        if dir_len > u32::MAX as u64 {
            return Err(Error::Format("keys: directory longer than 4 GiB".into()));
        }
        let buckets_off = align8(HEADER_LEN + dir_len);
        let start = out.len();
        out.resize(start + buckets_off as usize, 0);
        let mut dir: Vec<u8> = Vec::with_capacity(dir_len as usize);
        let mut firsts: Vec<u8> = Vec::with_capacity(first_keys as usize);
        for chunk in &chunks {
            let b_start = out.len() as u64 - start as u64;
            for (key, id) in chunk.iter() {
                varint::put(out, key.len() as u64);
                out.extend_from_slice(key);
                varint::put(out, **id as u64);
            }
            let b_len = out.len() as u64 - start as u64 - b_start;
            if b_len > u32::MAX as u64 {
                return Err(Error::Format("keys: bucket longer than 4 GiB".into()));
            }
            dir.extend_from_slice(&b_start.to_le_bytes());
            dir.extend_from_slice(&(b_len as u32).to_le_bytes());
            let fk_off = DIR_ENTRY_LEN * b_count + firsts.len() as u64;
            dir.extend_from_slice(&(fk_off as u32).to_le_bytes());
            firsts.extend_from_slice(chunk[0].0);
        }
        dir.extend_from_slice(&firsts);
        let buckets_len = out.len() as u64 - start as u64 - buckets_off;
        let h = &mut out[start..start + HEADER_LEN as usize];
        h[0..8].copy_from_slice(MAGIC);
        h[8..12].copy_from_slice(&VERSION.to_le_bytes());
        h[12..16].copy_from_slice(&(n as u32).to_le_bytes());
        h[16..20].copy_from_slice(&(b_count as u32).to_le_bytes());
        h[20..24].copy_from_slice(&self.bucket_keys.to_le_bytes());
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

/// `meta.keys`.
pub fn meta_json(n: u32, bucket_keys: u32, nbytes: u64) -> String {
    format!(r#"{{"version":{VERSION},"n":{n},"bucket_keys":{bucket_keys},"nbytes":{nbytes}}}"#)
}

pub fn is_keys(head: &[u8]) -> bool {
    head.len() >= 8 && &head[..8] == MAGIC
}

/// A page of a prefix listing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeyPage {
    /// `(key, id)` in key order.
    pub entries: Vec<(String, u32)>,
    /// Whether more keys with the prefix follow the last entry.
    pub more: bool,
}

/// The `keys` section over any [`ByteSource`]. `open` reads the 64-byte header only.
pub struct Keys<S: ByteSource> {
    source: S,
    n: u32,
    buckets: u32,
    bucket_keys: u32,
    directory: Range,
    bucket_bytes: Range,
}

impl<S: ByteSource> Keys<S> {
    pub fn open(source: S) -> Result<Keys<S>, Error> {
        let head = source.read(Range::new(0, HEADER_LEN))?;
        if !is_keys(&head) {
            return Err(Error::Format(
                "not a thread keys section (bad magic)".into(),
            ));
        }
        let version = u32_at(&head, 8);
        if version != VERSION {
            return Err(Error::Format(format!(
                "unsupported keys version {version} (expected {VERSION})"
            )));
        }
        let n = u32_at(&head, 12);
        let buckets = u32_at(&head, 16);
        let bucket_keys = u32_at(&head, 20);
        let directory = Range::new(u64_at(&head, 32), u64_at(&head, 40));
        let bucket_bytes = Range::new(u64_at(&head, 48), u64_at(&head, 56));
        drop(head);
        let total = source.len();
        let expect = if bucket_keys == 0 {
            0
        } else {
            (n as u64).div_ceil(bucket_keys as u64)
        };
        if (n > 0 && bucket_keys == 0)
            || buckets as u64 != expect
            || directory.offset != HEADER_LEN
            || directory.len < DIR_ENTRY_LEN * buckets as u64
            || bucket_bytes.offset < directory.end()
            || bucket_bytes.offset.checked_add(bucket_bytes.len).is_none()
            || bucket_bytes.end() > total
        {
            return Err(Error::Format("corrupt keys header".into()));
        }
        Ok(Keys {
            source,
            n,
            buckets,
            bucket_keys,
            directory,
            bucket_bytes,
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
    pub fn bucket_count(&self) -> u32 {
        self.buckets
    }
    pub fn bucket_keys(&self) -> u32 {
        self.bucket_keys
    }
    pub fn nbytes(&self) -> u64 {
        self.source.len()
    }
    pub fn directory_nbytes(&self) -> u64 {
        self.directory.len
    }
    pub fn meta_json(&self) -> String {
        meta_json(self.n, self.bucket_keys, self.nbytes())
    }

    /// The directory: one range (blob-absolute) a host fetches once and keeps.
    pub fn plan_directory(&self) -> Plan {
        let mut p = Plan::new();
        p.push(self.directory);
        p.absolute(&self.source)
    }

    fn directory(&self) -> Result<Cow<'_, [u8]>, SourceError> {
        self.source.read(self.directory)
    }

    #[inline]
    fn bucket_range(dir: &[u8], i: usize) -> Range {
        let o = i * DIR_ENTRY_LEN as usize;
        Range::new(u64_at(dir, o), u32_at(dir, o + 8) as u64)
    }

    #[inline]
    fn first_key<'d>(&self, dir: &'d [u8], i: usize) -> Option<&'d [u8]> {
        let o = i * DIR_ENTRY_LEN as usize;
        let s = u32_at(dir, o + 12) as usize;
        let e = if i + 1 < self.buckets as usize {
            u32_at(dir, o + DIR_ENTRY_LEN as usize + 12) as usize
        } else {
            dir.len()
        };
        if s < DIR_ENTRY_LEN as usize * self.buckets as usize || e < s {
            return None;
        }
        dir.get(s..e)
    }

    /// The bucket holding `k` if any does: the last one whose first key is `<= k`.
    fn candidate(&self, dir: &[u8], k: &[u8]) -> Result<Option<usize>, SourceError> {
        let corrupt = || SourceError::OutOfBounds(self.source.to_absolute(self.directory));
        let (mut lo, mut hi) = (0usize, self.buckets as usize);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.first_key(dir, mid).ok_or_else(corrupt)? <= k {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        Ok(lo.checked_sub(1))
    }

    fn check_bucket(&self, r: Range) -> Result<Range, SourceError> {
        if r.offset < self.bucket_bytes.offset || r.end() > self.bucket_bytes.end() {
            return Err(SourceError::OutOfBounds(self.source.to_absolute(r)));
        }
        Ok(r)
    }

    /// Parses the entry at `pos`, advancing it. `None` on corrupt bytes.
    fn parse_entry<'b>(buf: &'b [u8], pos: &mut usize) -> Option<(&'b [u8], u32)> {
        let len = varint::read(buf, pos) as usize;
        let key = buf.get(*pos..pos.checked_add(len)?)?;
        *pos += len;
        let id = u32::try_from(varint::read(buf, pos)).ok()?;
        Some((key, id))
    }

    fn find_in_bucket(buf: &[u8], k: &[u8]) -> Option<u32> {
        let mut pos = 0usize;
        while pos < buf.len() {
            let (key, id) = Self::parse_entry(buf, &mut pos)?;
            match key.cmp(k) {
                std::cmp::Ordering::Less => continue,
                std::cmp::Ordering::Equal => return Some(id),
                std::cmp::Ordering::Greater => return None,
            }
        }
        None
    }

    /// The bucket ranges `lookup(keys)` reads, blob-absolute and coalesced (`Missing` with the
    /// directory's range until it is held).
    pub fn plan_lookup(&self, keys: &[&str]) -> Result<Plan, SourceError> {
        let dir = self.directory()?;
        let mut plan = Plan::new();
        for k in keys {
            if let Some(i) = self.candidate(&dir, k.as_bytes())? {
                plan.push(self.check_bucket(Self::bucket_range(&dir, i))?);
            }
        }
        plan.coalesce(PLAN_GAP);
        Ok(plan.absolute(&self.source))
    }

    /// Every key's id (`None` when the section lacks the key). One round on a cold source after
    /// the directory: every bucket is attempted before failing.
    pub fn lookup(&self, keys: &[&str]) -> Result<Vec<Option<u32>>, SourceError> {
        let dir = self.directory()?;
        let mut out = vec![None; keys.len()];
        let mut wants: Vec<(usize, usize)> = Vec::with_capacity(keys.len());
        for (k, key) in keys.iter().enumerate() {
            if let Some(i) = self.candidate(&dir, key.as_bytes())? {
                wants.push((i, k));
            }
        }
        wants.sort_unstable();
        let mut acc = Plan::new();
        let mut j = 0;
        while j < wants.len() {
            let bucket = wants[j].0;
            let r = self.check_bucket(Self::bucket_range(&dir, bucket))?;
            match self.source.read(r) {
                Ok(buf) => {
                    while j < wants.len() && wants[j].0 == bucket {
                        let k = wants[j].1;
                        out[k] = Self::find_in_bucket(&buf, keys[k].as_bytes());
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

    /// The first bucket a listing from `start` reads (the one holding `start`, else bucket 0) and
    /// how many buckets `limit` entries can span from there.
    fn list_window(
        &self,
        dir: &[u8],
        start: &[u8],
        limit: usize,
    ) -> Result<(usize, usize), SourceError> {
        let first = self.candidate(dir, start)?.unwrap_or(0);
        // `limit + 1` matching entries decide `more`; the first bucket may be partly before `start`.
        let span = (limit as u64 + 1).div_ceil(self.bucket_keys.max(1) as u64) as usize + 1;
        Ok((
            first,
            span.min(self.buckets as usize - first.min(self.buckets as usize)),
        ))
    }

    /// The bucket ranges `list(prefix, after, limit)` reads first (contiguous buckets, so usually
    /// one range).
    pub fn plan_list(
        &self,
        prefix: &str,
        after: Option<&str>,
        limit: usize,
    ) -> Result<Plan, SourceError> {
        let dir = self.directory()?;
        let mut plan = Plan::new();
        if self.buckets == 0 {
            return Ok(plan);
        }
        let start = after
            .map_or(prefix.as_bytes(), str::as_bytes)
            .max(prefix.as_bytes());
        let (first, span) = self.list_window(&dir, start, limit)?;
        for i in first..first + span {
            plan.push(self.check_bucket(Self::bucket_range(&dir, i))?);
        }
        plan.coalesce(PLAN_GAP);
        Ok(plan.absolute(&self.source))
    }

    /// Up to `limit` keys that start with `prefix`, in key order, those `> after` only when
    /// `after` is given (paging: pass the last key returned). `more` says whether keys with the
    /// prefix follow. One round on a cold source after the directory for a page that fits the
    /// planned window; a longer scan over sparse matches may need more.
    pub fn list(
        &self,
        prefix: &str,
        after: Option<&str>,
        limit: usize,
    ) -> Result<KeyPage, SourceError> {
        let mut page = KeyPage::default();
        if self.buckets == 0 || limit == 0 {
            return Ok(page);
        }
        let dir = self.directory()?;
        let p = prefix.as_bytes();
        let start = after.map_or(p, str::as_bytes).max(p);
        // `after` itself is excluded (it was the last key of the previous page); the prefix is not.
        let skip_start = after.is_some_and(|a| a.as_bytes() >= p);
        let (first, span) = self.list_window(&dir, start, limit)?;
        // Read the whole window up front so a cold source reports every missing bucket at once.
        let mut acc = Plan::new();
        for i in first..first + span {
            if let Err(SourceError::Missing(rs)) = self
                .source
                .read(self.check_bucket(Self::bucket_range(&dir, i))?)
            {
                acc.extend(rs);
            }
        }
        if !acc.is_empty() {
            acc.coalesce(PLAN_GAP);
            return Err(SourceError::Missing(acc.into_ranges()));
        }
        let corrupt = || SourceError::OutOfBounds(self.source.to_absolute(self.bucket_bytes));
        for i in first..self.buckets as usize {
            let buf = self
                .source
                .read(self.check_bucket(Self::bucket_range(&dir, i))?)?;
            let mut pos = 0usize;
            while pos < buf.len() {
                let (key, id) = Self::parse_entry(&buf, &mut pos).ok_or_else(corrupt)?;
                if key < start || (skip_start && key == start) {
                    continue;
                }
                if !key.starts_with(p) {
                    // sorted: nothing after this key can match
                    return Ok(page);
                }
                if page.entries.len() == limit {
                    page.more = true;
                    return Ok(page);
                }
                page.entries
                    .push((String::from_utf8_lossy(key).into_owned(), id));
            }
        }
        Ok(page)
    }

    /// Every `(key, id)` in key order (reads the whole bucket region): for merges and tools.
    pub fn entries(&self) -> Result<Vec<(String, u32)>, SourceError> {
        let buf = self.source.read(self.bucket_bytes)?;
        let corrupt = || SourceError::OutOfBounds(self.source.to_absolute(self.bucket_bytes));
        let mut out = Vec::with_capacity(self.n as usize);
        let mut pos = 0usize;
        while pos < buf.len() {
            let (key, id) = Self::parse_entry(&buf, &mut pos).ok_or_else(corrupt)?;
            out.push((String::from_utf8_lossy(key).into_owned(), id));
        }
        Ok(out)
    }
}

/// Rebuilds one `keys` section from several segments given **oldest first**: for a key present in
/// several, the newest copy wins; entries whose id is in a segment's `skip` set are left out of
/// that segment.
pub fn merge<S: ByteSource>(
    inputs: &[(&Keys<S>, Option<&Tombstones>)],
    bucket_keys: u32,
) -> Result<KeysBuilder, Error> {
    let mut b = KeysBuilder::new().bucket_keys(bucket_keys);
    for (k, skip) in inputs.iter().rev() {
        for (key, id) in k.entries()? {
            if b.contains(&key) || skip.is_some_and(|t| t.contains(id)) {
                continue;
            }
            b.add(&key, id)?;
        }
    }
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Prefetched;

    fn sample(bucket_keys: u32) -> (Vec<u8>, Vec<(String, u32)>) {
        let mut rows: Vec<(String, u32)> = (0..300u32)
            .map(|i| (format!("s3://bucket/{:02}/obj-{i}.bin", i % 7), i * 3 + 1))
            .collect();
        rows.reverse();
        let mut b = KeysBuilder::new().bucket_keys(bucket_keys);
        for (k, id) in &rows {
            b.add(k, *id).unwrap();
        }
        rows.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        (b.finish().unwrap(), rows)
    }

    #[test]
    fn roundtrip_lookup_list_entries() {
        for bk in [1u32, 7, DEFAULT_BUCKET_KEYS, 10_000] {
            let (bytes, rows) = sample(bk);
            assert!(is_keys(&bytes));
            let k = Keys::open(bytes.as_slice()).unwrap();
            assert_eq!(k.n(), 300);
            assert_eq!(k.bucket_count() as u64, 300u64.div_ceil(bk as u64));
            assert_eq!(k.nbytes() as usize, bytes.len());
            let names: Vec<&str> = rows.iter().map(|r| r.0.as_str()).collect();
            let got = k.lookup(&names).unwrap();
            for (r, g) in rows.iter().zip(&got) {
                assert_eq!(*g, Some(r.1));
            }
            let misses = [
                "",
                "a",
                "s3://bucket/00/obj-1.bi",
                "s3://bucket/00/obj-1.binx",
                "zzz",
            ];
            assert_eq!(k.lookup(&misses).unwrap(), vec![None; misses.len()]);
            assert_eq!(k.entries().unwrap(), rows);
            // prefix listing with paging
            let want: Vec<(String, u32)> = rows
                .iter()
                .filter(|r| r.0.starts_with("s3://bucket/03/"))
                .cloned()
                .collect();
            let mut all = Vec::new();
            let mut after: Option<String> = None;
            loop {
                let page = k.list("s3://bucket/03/", after.as_deref(), 10).unwrap();
                assert!(page.entries.len() <= 10);
                let done = !page.more;
                after = page.entries.last().map(|e| e.0.clone());
                all.extend(page.entries);
                if done {
                    break;
                }
            }
            assert_eq!(all, want);
            assert_eq!(k.list("nope", None, 5).unwrap(), KeyPage::default());
            assert_eq!(k.list("", None, 0).unwrap(), KeyPage::default());
            let whole = k.list("", None, 1000).unwrap();
            assert_eq!(whole.entries, rows);
            assert!(!whole.more);
            let first = k.list("", None, 1).unwrap();
            assert_eq!(first.entries, rows[..1].to_vec());
            assert!(first.more);
            assert_eq!(k.meta_json(), meta_json(300, bk, bytes.len() as u64));
        }
        let empty = KeysBuilder::new().finish().unwrap();
        let k = Keys::open(empty.as_slice()).unwrap();
        assert_eq!(k.n(), 0);
        assert_eq!(k.lookup(&["x"]).unwrap(), vec![None]);
        assert_eq!(k.list("", None, 10).unwrap(), KeyPage::default());
        assert!(k.entries().unwrap().is_empty());
    }

    #[test]
    fn errors_and_corruption() {
        let mut b = KeysBuilder::new();
        b.add("a", 5).unwrap();
        assert!(b.add("a", 6).is_err());
        assert!(b.add("", 6).is_err());
        assert!(b.add("b", u32::MAX).is_err());
        assert!(b.add(&"k".repeat(MAX_KEY_LEN + 1), 1).is_err());
        assert!(KeysBuilder::new().bucket_keys(0).finish().is_err());
        let (bytes, _) = sample(16);
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(matches!(Keys::open(bad.as_slice()), Err(Error::Format(_))));
        let mut bad = bytes.clone();
        bad[8] = 7;
        assert!(matches!(Keys::open(bad.as_slice()), Err(Error::Format(_))));
        let mut bad = bytes.clone();
        bad[16] = 1;
        assert!(matches!(Keys::open(bad.as_slice()), Err(Error::Format(_))));
        assert!(matches!(Keys::open(&bytes[..40]), Err(Error::Source(_))));
        let mut bad = bytes.clone();
        bad[HEADER_LEN as usize + 7] = 255; // bucket 0's offset, top byte
        let k = Keys::open(bad.as_slice()).unwrap();
        assert!(matches!(
            k.lookup(&["s3://bucket/00/obj-0.bin"]),
            Err(SourceError::OutOfBounds(_))
        ));
    }

    #[test]
    fn paged_lookup_and_list_are_one_round_after_the_directory() {
        let (bytes, rows) = sample(8);
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
        let names = [
            rows[5].0.as_str(),
            rows[6].0.as_str(),
            rows[250].0.as_str(),
            "zzz",
        ];
        let k = Keys::open(crate::source::SubSource::new(&p, win).unwrap()).unwrap();
        let dir = k.plan_directory();
        assert_eq!(
            k.lookup(&names),
            Err(SourceError::Missing(dir.ranges().to_vec()))
        );
        feed(&mut p, dir.ranges());
        let k = Keys::open(crate::source::SubSource::new(&p, win).unwrap()).unwrap();
        let plan = k.plan_lookup(&names).unwrap();
        assert_eq!(plan.len(), 2); // 5 and 6 share a bucket, 250 is another; "zzz" sorts after everything
        assert_eq!(
            k.lookup(&names),
            Err(SourceError::Missing(plan.ranges().to_vec()))
        );
        feed(&mut p, plan.ranges());
        let k = Keys::open(crate::source::SubSource::new(&p, win).unwrap()).unwrap();
        let got = k.lookup(&names).unwrap();
        assert_eq!(
            got,
            vec![Some(rows[5].1), Some(rows[6].1), Some(rows[250].1), None]
        );
        // a listing: the planned window is one contiguous range, then the page comes back
        let lp = k.plan_list("s3://bucket/02/", None, 10).unwrap();
        assert_eq!(lp.len(), 1);
        let err = k.list("s3://bucket/02/", None, 10).unwrap_err();
        assert!(matches!(err, SourceError::Missing(_)));
        feed(&mut p, lp.ranges());
        let k = Keys::open(crate::source::SubSource::new(&p, win).unwrap()).unwrap();
        let page = k.list("s3://bucket/02/", None, 10).unwrap();
        assert_eq!(page.entries.len(), 10);
        assert!(page.more);
        assert!(page
            .entries
            .iter()
            .all(|e| e.0.starts_with("s3://bucket/02/")));
        assert!(p.held_bytes() < bytes.len() as u64 / 2);
    }

    #[test]
    fn merge_newest_wins_and_tombstones_drop() {
        let mut a = KeysBuilder::new();
        a.add("k1", 1).unwrap();
        a.add("k2", 2).unwrap();
        a.add("k3", 3).unwrap();
        let mut b = KeysBuilder::new();
        b.add("k2", 2).unwrap(); // re-put under the same id
        b.add("k4", 4).unwrap();
        b.add("k5", 55).unwrap(); // a key moved to another id
        let mut c = KeysBuilder::new();
        c.add("k5", 5).unwrap();
        let (ab, bb, cb) = (
            a.finish().unwrap(),
            b.finish().unwrap(),
            c.finish().unwrap(),
        );
        let (ka, kb, kc) = (
            Keys::open(ab.as_slice()).unwrap(),
            Keys::open(bb.as_slice()).unwrap(),
            Keys::open(cb.as_slice()).unwrap(),
        );
        let dead_a = Tombstones::from_ids(&[3]);
        let dead_b = Tombstones::from_ids(&[55]);
        let m = merge(
            &[(&ka, Some(&dead_a)), (&kb, Some(&dead_b)), (&kc, None)],
            2,
        )
        .unwrap()
        .finish()
        .unwrap();
        let k = Keys::open(m.as_slice()).unwrap();
        assert_eq!(
            k.entries().unwrap(),
            vec![
                ("k1".into(), 1),
                ("k2".into(), 2),
                ("k4".into(), 4),
                ("k5".into(), 5)
            ]
        );
    }
}
