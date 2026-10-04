//! The lexical (BM25) index of a segment: the `terms` and `postings` sections.
//!
//! Shaped for object storage, where a query's cost is the number of range requests and a few
//! kilobytes of extra bytes per request are free (`PLAN_GAP`):
//!
//! * `terms`: the segment's terms sorted by bytes, in buckets of `bucket_terms` terms behind a
//!   small directory (16 bytes + the first term per bucket, kept resident by hosts). Each bucket
//!   entry carries the term's document frequency and the byte range of its posting list, so the
//!   whole "vocabulary + per-term metadata" round of a classic index is **one bucket read per
//!   query term**. There are no term ids and no idf array: idf is recomputed at query time from
//!   `df` and the corpus size, which also lets several segments score like one index.
//! * `postings`: per term, its postings in ascending document id as LEB128 varints:
//!   `gap << 1 | (tf != 1)`, then `tf` when the flag is set, then the **document's length** (an
//!   exact varint, or one Lucene-style SmallFloat byte with [`Lengths::Byte`]). Carrying the
//!   length inline is what makes a query need no corpus-sized length array: it reads exactly the
//!   posting lists of its terms, nothing else. Document ids are the index's u32 ids, sparse and
//!   caller-assigned, the same ids the dense slabs hold.
//!
//! A query over a cold demand-paged source is therefore two rounds after the directory is held:
//! the terms' buckets, then their posting lists (both as one coalesced plan). Tombstoned ids are
//! skipped before top-k insertion like in the dense scans.
//!
//! Layouts (little-endian; offsets relative to the section start):
//!
//! ```text
//! terms                                              postings
//! 0   magic "THRDTERM"                                0   magic "THRDPOST"
//! 8   version u32 = 1                                 8   version u32 = 1
//! 12  term count T u32                                12  flags u32 (bit 0: lengths as SmallFloat bytes)
//! 16  bucket count B u32 = ceil(T / bucket_terms)     16  posting lists back to back
//! 20  bucket_terms u32
//! 24  num_docs u32        documents the segment indexed (incl. empty ones)
//! 28  flags u32           bit 0: lengths as SmallFloat bytes
//! 32  total_len u64       sum of the documents' lengths (tokens)
//! 40  num_postings u64
//! 48  k1 f64              the BM25 parameters the segment was built for
//! 56  b f64
//! 64  dir_off u64 (= 96)  72 dir_len u64   80 buckets_off u64   88 buckets_len u64
//! directory, at dir_off: B entries of 16 bytes (bucket offset u64, bucket len u32, first-term
//!   offset u32 relative to the directory start), then every bucket's first term back to back
//! bucket entry: len varint, term (UTF-8), df varint, post_off varint, post_len varint
//!   (`post_off`/`post_len` locate the term's list inside the postings section, header included)
//! ```
//!
//! The score of a document is the bm25s/Lucene expression used by Wiper's engine:
//! `Σ_terms idf * tf / (k1 * ((1 - b) + b * len / avg_len) + tf)` with `idf = ln(1 + (N - df +
//! 0.5) / (df + 0.5))` as float32, the tf component in float64, each product cast to float32 and
//! summed in float32. `N`, `avg_len` and `df` are **corpus-wide** ([`Stats`]): a host sums them
//! over the index's segments before scoring, so a corpus split over segments scores exactly like
//! the same corpus in one segment (tombstoned documents keep counting until a compaction, as
//! Lucene's deleted documents do).

use std::borrow::Cow;
use std::collections::HashMap;

use crate::error::Error;
use crate::source::{ByteSource, Plan, Range, SourceError};
use crate::tomb::Tombstones;
use crate::topk::{Hit, TopK};
use crate::varint;

pub const TERMS_MAGIC: &[u8; 8] = b"THRDTERM";
pub const POSTINGS_MAGIC: &[u8; 8] = b"THRDPOST";
pub const VERSION: u32 = 1;
pub const TERMS_HEADER_LEN: u64 = 96;
pub const POSTINGS_HEADER_LEN: u64 = 16;
pub const DIR_ENTRY_LEN: u64 = 16;
/// Terms per bucket unless the builder is told otherwise: with ~8-byte terms and ~4 bytes of
/// df/offset/len a bucket is 3-5 KB, one request's worth; a 250k-term segment has a ~25 KB
/// directory. Larger buckets shrink the directory, smaller ones the bytes read per query term.
pub const DEFAULT_BUCKET_TERMS: u32 = 256;
/// Ranges closer than this are merged in plans and `Missing` errors.
pub const PLAN_GAP: u64 = 4096;
const FLAG_BYTE_LENGTHS: u32 = 1;

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
#[inline]
fn f64_at(buf: &[u8], o: usize) -> f64 {
    f64::from_le_bytes(buf[o..o + 8].try_into().unwrap())
}

// ---- document lengths -------------------------------------------------------------------------

/// How a posting carries its document's length (a build parameter: size vs exact scores).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lengths {
    /// An exact LEB128 varint (1 byte below 128 tokens, 2 below 16384): exact BM25 scores.
    Exact,
    /// One byte, Lucene's `SmallFloat.intToByte4` encoding: exact up to 40 tokens, then a 3-bit
    /// mantissa (at most ~12% relative error on the length, a few percent on the score). Saves
    /// about one byte per posting on corpora of long documents.
    Byte,
}

impl Lengths {
    pub fn name(self) -> &'static str {
        match self {
            Lengths::Exact => "exact",
            Lengths::Byte => "byte",
        }
    }
    pub fn from_name(s: &str) -> Option<Lengths> {
        match s {
            "exact" => Some(Lengths::Exact),
            "byte" => Some(Lengths::Byte),
            _ => None,
        }
    }
    fn from_flags(flags: u32) -> Lengths {
        if flags & FLAG_BYTE_LENGTHS != 0 {
            Lengths::Byte
        } else {
            Lengths::Exact
        }
    }
    fn flags(self) -> u32 {
        match self {
            Lengths::Exact => 0,
            Lengths::Byte => FLAG_BYTE_LENGTHS,
        }
    }
}

const NUM_FREE_VALUES: u32 = 24;

fn long_to_int4(i: u64) -> u32 {
    let num_bits = 64 - i.leading_zeros();
    if num_bits < 4 {
        return i as u32;
    }
    let shift = num_bits - 4;
    // keep the 4 most significant bits, drop the implicit leading one, store shift + 1
    let mut encoded = (i >> shift) as u32 & 0x07;
    encoded |= (shift + 1) << 3;
    encoded
}

fn int4_to_long(i: u32) -> u64 {
    let bits = (i & 7) as u64;
    let shift = (i >> 3) as i32 - 1;
    if shift < 0 {
        bits
    } else {
        (bits | 8) << shift
    }
}

/// Lucene's `SmallFloat.intToByte4`: lossless below 40, then a 3-bit mantissa with an exponent.
pub fn len_to_byte(len: u32) -> u8 {
    if len < NUM_FREE_VALUES {
        len as u8
    } else {
        (NUM_FREE_VALUES + long_to_int4((len - NUM_FREE_VALUES) as u64)) as u8
    }
}

/// Lucene's `SmallFloat.byte4ToInt`.
pub fn byte_to_len(b: u8) -> u32 {
    let i = b as u32;
    if i < NUM_FREE_VALUES {
        i
    } else {
        NUM_FREE_VALUES + int4_to_long(i - NUM_FREE_VALUES) as u32
    }
}

// ---- scoring ------------------------------------------------------------------------------------

/// Corpus-wide counts a query scores with: summed over every segment of an index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub num_docs: u64,
    pub total_len: u64,
}

impl Stats {
    pub fn add(&mut self, other: Stats) {
        self.num_docs += other.num_docs;
        self.total_len += other.total_len;
    }
    /// Mean document length (0 for an empty corpus).
    pub fn avg_len(&self) -> f64 {
        if self.num_docs == 0 {
            0.0
        } else {
            self.total_len as f64 / self.num_docs as f64
        }
    }
    /// Lucene idf of a term seen in `df` documents of this corpus, as float32 (0 for df = 0).
    pub fn idf(&self, df: u64) -> f32 {
        lucene_idf(df, self.num_docs)
    }
    pub fn scoring(&self, k1: f64, b: f64) -> Scoring {
        Scoring {
            k1,
            b,
            avg_len: self.avg_len(),
        }
    }
}

/// `ln(1 + (N - df + 0.5) / (df + 0.5))` as float32, like bm25s (method `lucene`).
#[inline]
pub fn lucene_idf(df: u64, n: u64) -> f32 {
    if df == 0 {
        return 0.0;
    }
    let (df, n) = (df as f64, n as f64);
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln() as f32
}

/// The BM25 parameters and the corpus mean length a query scores with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scoring {
    pub k1: f64,
    pub b: f64,
    pub avg_len: f64,
}

/// `k1 * ((1 - b) + b * len / avg_len)` in float64.
#[inline(always)]
fn length_norm(s: &Scoring, len: u32) -> f64 {
    s.k1 * ((1.0 - s.b) + s.b * len as f64 / s.avg_len)
}

/// `idf * (tf / (norm + tf))`: float32 idf, float64 tf component, float32 product (bm25s).
#[inline(always)]
fn contribution(idf: f32, tf: f32, norm: f64) -> f32 {
    let tf = tf as f64;
    (idf as f64 * (tf / (norm + tf))) as f32
}

// ---- builder ------------------------------------------------------------------------------------

/// Build parameters of a lexical segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LexParams {
    pub k1: f64,
    pub b: f64,
    pub lengths: Lengths,
    pub bucket_terms: u32,
}

impl Default for LexParams {
    fn default() -> Self {
        LexParams {
            k1: 1.2,
            b: 0.75,
            lengths: Lengths::Exact,
            bucket_terms: DEFAULT_BUCKET_TERMS,
        }
    }
}

/// The two sections a build produces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LexBytes {
    pub terms: Vec<u8>,
    pub postings: Vec<u8>,
}

/// Builds the `terms` and `postings` sections from tokenized documents (any id order).
pub struct LexBuilder {
    params: LexParams,
    /// term -> (doc, tf), in insertion order (sorted at `finish`).
    terms: HashMap<String, Vec<(u32, u32)>>,
    /// doc -> length (also rejects duplicate ids).
    docs: HashMap<u32, u32>,
    num_postings: u64,
}

impl LexBuilder {
    pub fn new(params: LexParams) -> LexBuilder {
        LexBuilder {
            params,
            terms: HashMap::new(),
            docs: HashMap::new(),
            num_postings: 0,
        }
    }

    pub fn params(&self) -> &LexParams {
        &self.params
    }
    pub fn num_docs(&self) -> u32 {
        self.docs.len() as u32
    }
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }
    pub fn num_postings(&self) -> u64 {
        self.num_postings
    }
    pub fn total_len(&self) -> u64 {
        self.docs.values().map(|&l| l as u64).sum()
    }
    pub fn contains(&self, id: u32) -> bool {
        self.docs.contains_key(&id)
    }

    /// Adds document `id` with its tokens (repeats count as term frequency). A document with no
    /// tokens still counts towards `num_docs`. Fails on a duplicate id or `id == u32::MAX`.
    pub fn add(&mut self, id: u32, tokens: &[String]) -> Result<(), Error> {
        let mut sorted: Vec<&str> = tokens.iter().map(String::as_str).collect();
        sorted.sort_unstable();
        let mut counts: Vec<(&str, u32)> = Vec::new();
        for t in sorted {
            match counts.last_mut() {
                Some((lt, n)) if *lt == t => *n += 1,
                _ => counts.push((t, 1)),
            }
        }
        self.add_counts(id, tokens.len() as u32, counts)
    }

    /// Adds document `id` of `len` tokens from its distinct terms with their frequencies.
    pub fn add_counts<'t, I>(&mut self, id: u32, len: u32, counts: I) -> Result<(), Error>
    where
        I: IntoIterator<Item = (&'t str, u32)>,
    {
        if id == u32::MAX {
            return Err(Error::Format("document id u32::MAX is reserved".into()));
        }
        if self.docs.insert(id, len).is_some() {
            return Err(Error::Format(format!("duplicate document id {id}")));
        }
        for (t, tf) in counts {
            if tf == 0 {
                continue;
            }
            self.terms.entry(t.to_string()).or_default().push((id, tf));
            self.num_postings += 1;
        }
        Ok(())
    }

    /// Serializes both sections.
    pub fn finish(&self) -> Result<LexBytes, Error> {
        let p = &self.params;
        if p.bucket_terms == 0 {
            return Err(Error::Format("lex: bucket_terms must be > 0".into()));
        }
        if self.terms.len() > u32::MAX as usize {
            return Err(Error::Format("lex: too many terms".into()));
        }
        let mut names: Vec<&String> = self.terms.keys().collect();
        names.sort_unstable_by(|a, b| a.as_bytes().cmp(b.as_bytes()));

        // Postings: header, then every list in term order.
        let mut postings = vec![0u8; POSTINGS_HEADER_LEN as usize];
        postings[0..8].copy_from_slice(POSTINGS_MAGIC);
        postings[8..12].copy_from_slice(&VERSION.to_le_bytes());
        postings[12..16].copy_from_slice(&p.lengths.flags().to_le_bytes());
        let mut infos: Vec<(u32, u64, u64)> = Vec::with_capacity(names.len()); // df, off, len
        let mut list: Vec<(u32, u32)> = Vec::new();
        for name in &names {
            list.clear();
            list.extend_from_slice(&self.terms[*name]);
            list.sort_unstable();
            let start = postings.len() as u64;
            let mut prev: u32 = u32::MAX;
            for &(doc, tf) in &list {
                let gap = doc.wrapping_sub(prev).wrapping_sub(1) as u64;
                varint::put(&mut postings, (gap << 1) | (tf != 1) as u64);
                if tf != 1 {
                    varint::put(&mut postings, tf as u64);
                }
                let len = self.docs[&doc];
                match p.lengths {
                    Lengths::Exact => varint::put(&mut postings, len as u64),
                    Lengths::Byte => postings.push(len_to_byte(len)),
                }
                prev = doc;
            }
            infos.push((list.len() as u32, start, postings.len() as u64 - start));
        }

        // Terms: directory + buckets.
        let t_count = names.len() as u64;
        let bt = p.bucket_terms as usize;
        let b_count = t_count.div_ceil(bt as u64);
        let firsts: u64 = names.chunks(bt).map(|c| c[0].len() as u64).sum();
        let dir_len = DIR_ENTRY_LEN * b_count + firsts;
        if dir_len > u32::MAX as u64 {
            return Err(Error::Format("lex: directory longer than 4 GiB".into()));
        }
        let buckets_off = align8(TERMS_HEADER_LEN + dir_len);
        let mut out = vec![0u8; buckets_off as usize];
        let mut entries: Vec<u8> = Vec::with_capacity((DIR_ENTRY_LEN * b_count) as usize);
        let mut first_names: Vec<u8> = Vec::with_capacity(firsts as usize);
        for (ci, chunk) in names.chunks(bt).enumerate() {
            let b_start = out.len() as u64;
            for (j, name) in chunk.iter().enumerate() {
                let (df, off, len) = infos[ci * bt + j];
                varint::put(&mut out, name.len() as u64);
                out.extend_from_slice(name.as_bytes());
                varint::put(&mut out, df as u64);
                varint::put(&mut out, off);
                varint::put(&mut out, len);
            }
            let b_len = out.len() as u64 - b_start;
            if b_len > u32::MAX as u64 {
                return Err(Error::Format("lex: bucket longer than 4 GiB".into()));
            }
            entries.extend_from_slice(&b_start.to_le_bytes());
            entries.extend_from_slice(&(b_len as u32).to_le_bytes());
            let first_off = DIR_ENTRY_LEN * b_count + first_names.len() as u64;
            entries.extend_from_slice(&(first_off as u32).to_le_bytes());
            first_names.extend_from_slice(chunk[0].as_bytes());
        }
        let buckets_len = out.len() as u64 - buckets_off;
        let h = &mut out[..TERMS_HEADER_LEN as usize];
        h[0..8].copy_from_slice(TERMS_MAGIC);
        h[8..12].copy_from_slice(&VERSION.to_le_bytes());
        h[12..16].copy_from_slice(&(t_count as u32).to_le_bytes());
        h[16..20].copy_from_slice(&(b_count as u32).to_le_bytes());
        h[20..24].copy_from_slice(&p.bucket_terms.to_le_bytes());
        h[24..28].copy_from_slice(&(self.docs.len() as u32).to_le_bytes());
        h[28..32].copy_from_slice(&p.lengths.flags().to_le_bytes());
        h[32..40].copy_from_slice(&self.total_len().to_le_bytes());
        h[40..48].copy_from_slice(&self.num_postings.to_le_bytes());
        h[48..56].copy_from_slice(&p.k1.to_le_bytes());
        h[56..64].copy_from_slice(&p.b.to_le_bytes());
        h[64..72].copy_from_slice(&TERMS_HEADER_LEN.to_le_bytes());
        h[72..80].copy_from_slice(&dir_len.to_le_bytes());
        h[80..88].copy_from_slice(&buckets_off.to_le_bytes());
        h[88..96].copy_from_slice(&buckets_len.to_le_bytes());
        let d = TERMS_HEADER_LEN as usize;
        out[d..d + entries.len()].copy_from_slice(&entries);
        out[d + entries.len()..d + dir_len as usize].copy_from_slice(&first_names);
        Ok(LexBytes {
            terms: out,
            postings,
        })
    }
}

// ---- terms section --------------------------------------------------------------------------------

/// What a bucket entry says about a term: its document frequency and its posting list's range
/// inside the `postings` section (section-relative).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TermInfo {
    pub df: u32,
    pub postings: Range,
}

/// The `terms` section over any [`ByteSource`].
pub struct Terms<S: ByteSource> {
    source: S,
    terms: u32,
    buckets: u32,
    bucket_terms: u32,
    num_docs: u32,
    lengths: Lengths,
    total_len: u64,
    num_postings: u64,
    k1: f64,
    b: f64,
    directory: Range,
    bucket_bytes: Range,
}

impl<S: ByteSource> Terms<S> {
    /// Opens a section: reads and validates the 96-byte header only.
    pub fn open(source: S) -> Result<Terms<S>, Error> {
        let head = source.read(Range::new(0, TERMS_HEADER_LEN))?;
        if &head[0..8] != TERMS_MAGIC {
            return Err(Error::Format(
                "not a thread terms section (bad magic)".into(),
            ));
        }
        let version = u32_at(&head, 8);
        if version != VERSION {
            return Err(Error::Format(format!(
                "unsupported terms version {version} (expected {VERSION})"
            )));
        }
        let terms = u32_at(&head, 12);
        let buckets = u32_at(&head, 16);
        let bucket_terms = u32_at(&head, 20);
        let num_docs = u32_at(&head, 24);
        let lengths = Lengths::from_flags(u32_at(&head, 28));
        let total_len = u64_at(&head, 32);
        let num_postings = u64_at(&head, 40);
        let k1 = f64_at(&head, 48);
        let b = f64_at(&head, 56);
        let directory = Range::new(u64_at(&head, 64), u64_at(&head, 72));
        let bucket_bytes = Range::new(u64_at(&head, 80), u64_at(&head, 88));
        drop(head);
        let expect_buckets = if bucket_terms == 0 {
            0
        } else {
            (terms as u64).div_ceil(bucket_terms as u64)
        };
        if (terms > 0 && bucket_terms == 0)
            || buckets as u64 != expect_buckets
            || directory.offset < TERMS_HEADER_LEN
            || directory.len < DIR_ENTRY_LEN * buckets as u64
            || directory.len > u32::MAX as u64
            || directory.end() > bucket_bytes.offset
            || bucket_bytes.offset.checked_add(bucket_bytes.len) != Some(source.len())
            || !(k1.is_finite() && b.is_finite())
        {
            return Err(Error::Format("corrupt terms header".into()));
        }
        Ok(Terms {
            source,
            terms,
            buckets,
            bucket_terms,
            num_docs,
            lengths,
            total_len,
            num_postings,
            k1,
            b,
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
    /// Distinct terms.
    pub fn len(&self) -> u32 {
        self.terms
    }
    pub fn is_empty(&self) -> bool {
        self.terms == 0
    }
    pub fn bucket_count(&self) -> u32 {
        self.buckets
    }
    pub fn bucket_terms(&self) -> u32 {
        self.bucket_terms
    }
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }
    pub fn total_len(&self) -> u64 {
        self.total_len
    }
    pub fn num_postings(&self) -> u64 {
        self.num_postings
    }
    pub fn lengths(&self) -> Lengths {
        self.lengths
    }
    pub fn k1(&self) -> f64 {
        self.k1
    }
    pub fn b(&self) -> f64 {
        self.b
    }
    /// The segment's own counts (sum them over segments into the query's [`Stats`]).
    pub fn stats(&self) -> Stats {
        Stats {
            num_docs: self.num_docs as u64,
            total_len: self.total_len,
        }
    }
    pub fn params(&self) -> LexParams {
        LexParams {
            k1: self.k1,
            b: self.b,
            lengths: self.lengths,
            bucket_terms: self.bucket_terms,
        }
    }
    pub fn nbytes(&self) -> u64 {
        self.bucket_bytes.end()
    }
    pub fn directory_nbytes(&self) -> u64 {
        self.directory.len
    }
    /// The `meta.lex` object.
    pub fn meta_json(&self, postings_nbytes: u64) -> String {
        format!(
            r#"{{"version":{VERSION},"num_docs":{},"terms":{},"num_postings":{},"total_len":{},"k1":{},"b":{},"lengths":"{}","bucket_terms":{},"terms_nbytes":{},"postings_nbytes":{}}}"#,
            self.num_docs,
            self.terms,
            self.num_postings,
            self.total_len,
            self.k1,
            self.b,
            self.lengths.name(),
            self.bucket_terms,
            self.nbytes(),
            postings_nbytes
        )
    }

    /// The directory: one range (blob-absolute) a host fetches once and keeps.
    pub fn plan_directory(&self) -> Plan {
        let mut p = Plan::new();
        p.push(self.directory);
        p.absolute(&self.source)
    }

    #[inline]
    fn bucket_range(dir: &[u8], i: usize) -> Range {
        let o = i * DIR_ENTRY_LEN as usize;
        Range::new(u64_at(dir, o), u32_at(dir, o + 8) as u64)
    }

    #[inline]
    fn first_term<'d>(&self, dir: &'d [u8], i: usize) -> Option<&'d [u8]> {
        let o = i * DIR_ENTRY_LEN as usize;
        let s = u32_at(dir, o + 12) as usize;
        let e = if i + 1 < self.buckets as usize {
            u32_at(dir, o + DIR_ENTRY_LEN as usize + 12) as usize
        } else {
            dir.len()
        };
        if s < DIR_ENTRY_LEN as usize * self.buckets as usize {
            return None;
        }
        dir.get(s..e)
    }

    /// The bucket holding `t` if any does: the last one whose first term is `<= t`.
    fn candidate(&self, dir: &[u8], t: &[u8]) -> Result<Option<usize>, SourceError> {
        let corrupt = || SourceError::OutOfBounds(self.source.to_absolute(self.directory));
        let (mut lo, mut hi) = (0usize, self.buckets as usize);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.first_term(dir, mid).ok_or_else(corrupt)? <= t {
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
    fn parse_entry<'b>(buf: &'b [u8], pos: &mut usize) -> Option<(&'b [u8], TermInfo)> {
        let len = varint::read(buf, pos) as usize;
        let term = buf.get(*pos..pos.checked_add(len)?)?;
        *pos += len;
        let df = u32::try_from(varint::read(buf, pos)).ok()?;
        let off = varint::read(buf, pos);
        let plen = varint::read(buf, pos);
        Some((
            term,
            TermInfo {
                df,
                postings: Range::new(off, plen),
            },
        ))
    }

    fn find_in_bucket(buf: &[u8], t: &[u8]) -> Option<TermInfo> {
        let mut pos = 0usize;
        while pos < buf.len() {
            let (term, info) = Self::parse_entry(buf, &mut pos)?;
            match term.cmp(t) {
                std::cmp::Ordering::Less => continue,
                std::cmp::Ordering::Equal => return Some(info),
                std::cmp::Ordering::Greater => return None,
            }
        }
        None
    }

    fn directory(&self) -> Result<Cow<'_, [u8]>, SourceError> {
        self.source.read(self.directory)
    }

    /// The bucket ranges `lookup(terms)` reads, blob-absolute and coalesced (`Missing` with the
    /// directory's range until it is held).
    pub fn plan_lookup(&self, terms: &[&str]) -> Result<Plan, SourceError> {
        let dir = self.directory()?;
        let mut plan = Plan::new();
        for t in terms {
            if let Some(i) = self.candidate(&dir, t.as_bytes())? {
                plan.push(self.check_bucket(Self::bucket_range(&dir, i))?);
            }
        }
        plan.coalesce(PLAN_GAP);
        Ok(plan.absolute(&self.source))
    }

    /// Every term's df and posting range (`None` when the segment lacks the term). One round on
    /// a cold source after the directory: every bucket is attempted before failing.
    pub fn lookup(&self, terms: &[&str]) -> Result<Vec<Option<TermInfo>>, SourceError> {
        let dir = self.directory()?;
        let mut out = vec![None; terms.len()];
        let mut wants: Vec<(usize, usize)> = Vec::with_capacity(terms.len());
        for (k, t) in terms.iter().enumerate() {
            if let Some(i) = self.candidate(&dir, t.as_bytes())? {
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
                        out[k] = Self::find_in_bucket(&buf, terms[k].as_bytes());
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

    /// Every `(term, info)` in sorted order (reads the whole section): for merges and tools.
    pub fn entries(&self) -> Result<Vec<(String, TermInfo)>, SourceError> {
        let buf = self.source.read(self.bucket_bytes)?;
        let corrupt = || SourceError::OutOfBounds(self.source.to_absolute(self.bucket_bytes));
        let mut out = Vec::with_capacity(self.terms as usize);
        let mut pos = 0usize;
        while pos < buf.len() {
            let (term, info) = Self::parse_entry(&buf, &mut pos).ok_or_else(corrupt)?;
            out.push((String::from_utf8_lossy(term).into_owned(), info));
        }
        Ok(out)
    }
}

// ---- postings section -----------------------------------------------------------------------------

/// One decoded posting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Posting {
    pub doc: u32,
    pub tf: u32,
    pub len: u32,
}

/// A query term ready for scoring: where its list is, its corpus-wide idf and its weight (the
/// number of times it occurs in the query).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryTerm {
    pub info: TermInfo,
    pub idf: f32,
    pub weight: f32,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    remaining: usize,
    lengths: Lengths,
    doc: u32,
    tf: f32,
    len: u32,
}

impl Cursor<'_> {
    #[inline(always)]
    fn advance(&mut self) {
        if self.remaining == 0 || self.pos >= self.buf.len() {
            self.doc = u32::MAX;
            return;
        }
        self.remaining -= 1;
        let v = varint::read(self.buf, &mut self.pos);
        self.doc = self.doc.wrapping_add((v >> 1) as u32).wrapping_add(1);
        self.tf = if v & 1 == 1 {
            varint::read(self.buf, &mut self.pos) as f32
        } else {
            1.0
        };
        self.len = match self.lengths {
            Lengths::Exact => varint::read(self.buf, &mut self.pos) as u32,
            Lengths::Byte => {
                let b = self.buf.get(self.pos).copied().unwrap_or(0);
                self.pos += 1;
                byte_to_len(b)
            }
        };
    }
}

/// The `postings` section over any [`ByteSource`].
pub struct Postings<S: ByteSource> {
    source: S,
    lengths: Lengths,
}

impl<S: ByteSource> Postings<S> {
    /// Opens the section: reads and validates the 16-byte header only.
    pub fn open(source: S) -> Result<Postings<S>, Error> {
        let head = source.read(Range::new(0, POSTINGS_HEADER_LEN))?;
        if &head[0..8] != POSTINGS_MAGIC {
            return Err(Error::Format(
                "not a thread postings section (bad magic)".into(),
            ));
        }
        let version = u32_at(&head, 8);
        if version != VERSION {
            return Err(Error::Format(format!(
                "unsupported postings version {version} (expected {VERSION})"
            )));
        }
        let lengths = Lengths::from_flags(u32_at(&head, 12));
        drop(head);
        Ok(Postings { source, lengths })
    }

    pub fn source(&self) -> &S {
        &self.source
    }
    pub fn into_source(self) -> S {
        self.source
    }
    pub fn lengths(&self) -> Lengths {
        self.lengths
    }
    pub fn nbytes(&self) -> u64 {
        self.source.len()
    }

    fn check(&self, info: &TermInfo) -> Result<Range, SourceError> {
        let r = info.postings;
        if r.offset < POSTINGS_HEADER_LEN
            || r.offset
                .checked_add(r.len)
                .is_none_or(|e| e > self.source.len())
        {
            return Err(SourceError::OutOfBounds(self.source.to_absolute(r)));
        }
        Ok(r)
    }

    /// The posting lists of `terms`, blob-absolute and coalesced: the second round of a query.
    pub fn plan(&self, terms: &[TermInfo]) -> Result<Plan, SourceError> {
        let mut plan = Plan::new();
        for t in terms {
            if t.df > 0 {
                plan.push(self.check(t)?);
            }
        }
        plan.coalesce(PLAN_GAP);
        Ok(plan.absolute(&self.source))
    }

    fn cursor<'b>(&self, buf: &'b [u8], df: u32) -> Cursor<'b> {
        let mut c = Cursor {
            buf,
            pos: 0,
            remaining: df as usize,
            lengths: self.lengths,
            doc: u32::MAX,
            tf: 0.0,
            len: 0,
        };
        c.advance();
        c
    }

    /// Decodes one list (tools, merges).
    pub fn decode(&self, info: &TermInfo) -> Result<Vec<Posting>, SourceError> {
        if info.df == 0 {
            return Ok(Vec::new());
        }
        let buf = self.source.read(self.check(info)?)?;
        let mut c = self.cursor(&buf, info.df);
        let mut out = Vec::with_capacity(info.df as usize);
        while c.doc != u32::MAX {
            out.push(Posting {
                doc: c.doc,
                tf: c.tf as u32,
                len: c.len,
            });
            c.advance();
        }
        Ok(out)
    }

    /// Scores the documents of this segment for `terms` (document-at-a-time over the lists) and
    /// returns the top `k` with a positive score, best first, ties by ascending id; ids in `skip`
    /// never take a slot. One round on a cold source: every list is attempted before failing.
    /// Results of several segments scored with the same [`Scoring`] and idfs merge with
    /// [`merge_hits`] into exactly the top `k` of the union.
    pub fn search(
        &self,
        terms: &[QueryTerm],
        scoring: &Scoring,
        k: usize,
        skip: Option<&Tombstones>,
    ) -> Result<Vec<Hit>, SourceError> {
        let live: Vec<&QueryTerm> = terms
            .iter()
            .filter(|t| t.info.df > 0 && t.weight != 0.0)
            .collect();
        if k == 0 || live.is_empty() {
            return Ok(Vec::new());
        }
        let mut acc = Plan::new();
        let mut bufs: Vec<Option<Cow<'_, [u8]>>> = Vec::with_capacity(live.len());
        for t in &live {
            match self.source.read(self.check(&t.info)?) {
                Ok(b) => bufs.push(Some(b)),
                Err(SourceError::Missing(rs)) => {
                    acc.extend(rs);
                    bufs.push(None);
                }
                Err(e) => return Err(e),
            }
        }
        if !acc.is_empty() {
            acc.coalesce(PLAN_GAP);
            return Err(SourceError::Missing(acc.into_ranges()));
        }
        let mut cursors: Vec<(Cursor<'_>, f32, f32)> = live
            .iter()
            .zip(&bufs)
            .map(|(t, b)| {
                (
                    self.cursor(b.as_ref().expect("read"), t.info.df),
                    t.idf,
                    t.weight,
                )
            })
            .collect();
        let mut top = TopK::new(k);
        loop {
            let mut d = u32::MAX;
            let mut len = 0u32;
            for (c, _, _) in &cursors {
                if c.doc < d {
                    d = c.doc;
                    len = c.len;
                }
            }
            if d == u32::MAX {
                break;
            }
            let norm = length_norm(scoring, len);
            let mut score = 0f32;
            for (c, idf, w) in cursors.iter_mut() {
                if c.doc == d {
                    score += *w * contribution(*idf, c.tf, norm);
                    c.advance();
                }
            }
            if score > 0.0 && score > top.threshold() && !skip.is_some_and(|t| t.contains(d)) {
                top.push(d, score, 0);
            }
        }
        Ok(top.into_sorted())
    }
}

/// Merges per-segment results (each best first) into one top-`k` list: score descending, ties by
/// ascending id. Ids are unique across live segments (a replaced document's older copies are
/// tombstoned), so no deduplication is needed; a duplicate keeps its best score anyway.
pub fn merge_hits(parts: &[Vec<Hit>], k: usize) -> Vec<Hit> {
    let all: Vec<Hit> = parts.iter().flatten().copied().collect();
    crate::topk::merge_dedup(all, k)
}

// ---- merge (compaction) -----------------------------------------------------------------------------

/// One segment's lexical sections for [`merge`], with the ids to leave out.
pub struct MergeInput<'a, S: ByteSource> {
    pub terms: &'a Terms<S>,
    pub postings: &'a Postings<S>,
    pub skip: Option<&'a Tombstones>,
}

/// Rebuilds one lexical index from several segments (the compaction): every term's lists are
/// concatenated (documents re-sorted, tombstoned ones dropped) and re-encoded with `params`. A
/// document with no tokens leaves no trace in the postings and is not counted in the result.
pub fn merge<S: ByteSource>(
    inputs: &[MergeInput<'_, S>],
    params: LexParams,
) -> Result<LexBuilder, Error> {
    let mut builder = LexBuilder::new(params);
    // Gather every (term, input, info); terms sort by bytes so equal terms are adjacent.
    let mut all: Vec<(String, usize, TermInfo)> = Vec::new();
    for (i, inp) in inputs.iter().enumerate() {
        for (t, info) in inp.terms.entries()? {
            all.push((t, i, info));
        }
    }
    all.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()).then(a.1.cmp(&b.1)));
    // Documents first: the builder wants each document's length before its terms.
    let mut lens: HashMap<u32, u32> = HashMap::new();
    let mut per_term: Vec<(String, Vec<(u32, u32)>)> = Vec::new();
    let mut i = 0;
    while i < all.len() {
        let term = &all[i].0;
        let mut list: Vec<(u32, u32)> = Vec::new();
        let mut j = i;
        while j < all.len() && all[j].0 == *term {
            let inp = &inputs[all[j].1];
            for p in inp.postings.decode(&all[j].2)? {
                if inp.skip.is_some_and(|t| t.contains(p.doc)) {
                    continue;
                }
                lens.entry(p.doc).or_insert(p.len);
                list.push((p.doc, p.tf));
            }
            j += 1;
        }
        if !list.is_empty() {
            list.sort_unstable();
            per_term.push((term.clone(), list));
        }
        i = j;
    }
    // Invert back to documents so the builder's `add_counts` sees each document once.
    let mut by_doc: HashMap<u32, Vec<(usize, u32)>> = HashMap::new();
    for (ti, (_, list)) in per_term.iter().enumerate() {
        for &(doc, tf) in list {
            by_doc.entry(doc).or_default().push((ti, tf));
        }
    }
    let mut docs: Vec<u32> = by_doc.keys().copied().collect();
    docs.sort_unstable();
    for doc in docs {
        let counts = &by_doc[&doc];
        builder.add_counts(
            doc,
            lens[&doc],
            counts.iter().map(|&(ti, tf)| (per_term[ti].0.as_str(), tf)),
        )?;
    }
    Ok(builder)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Prefetched;

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    /// Reference BM25 over documents `(id, tokens)` for `query`, bm25s arithmetic.
    fn reference(
        docs: &[(u32, Vec<String>)],
        query: &[&str],
        params: &LexParams,
    ) -> Vec<(u32, f32)> {
        let n = docs.len() as u64;
        let total: u64 = docs.iter().map(|d| d.1.len() as u64).sum();
        let avg = total as f64 / n as f64;
        let scoring = Scoring {
            k1: params.k1,
            b: params.b,
            avg_len: avg,
        };
        let mut out = Vec::new();
        for (id, t) in docs {
            let mut s = 0f32;
            for q in query {
                let df = docs.iter().filter(|d| d.1.iter().any(|x| x == q)).count() as u64;
                let tf = t.iter().filter(|x| *x == q).count() as f32;
                if tf > 0.0 {
                    s += contribution(lucene_idf(df, n), tf, length_norm(&scoring, t.len() as u32));
                }
            }
            if s > 0.0 {
                out.push((*id, s));
            }
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    fn corpus() -> Vec<(u32, Vec<String>)> {
        let mut docs = vec![
            (10, toks("the deploy failed on tuesday")),
            (3, toks("retry the deploy deploy deploy")),
            (7, toks("")),
            (1000, toks("budget review for the galaxy budget")),
            (42, toks("galaxy galaxy budget")),
        ];
        for i in 0..300u32 {
            let mut t = toks("filler line");
            if i % 7 == 0 {
                t.push("deploy".into());
            }
            if i % 13 == 0 {
                t.extend(toks("galaxy budget"));
            }
            for _ in 0..(i % 5) {
                t.push(format!("w{i}"));
            }
            docs.push((2000 + i * 3, t));
        }
        docs
    }

    fn build(docs: &[(u32, Vec<String>)], params: LexParams) -> LexBytes {
        let mut b = LexBuilder::new(params);
        for (id, t) in docs {
            b.add(*id, t).unwrap();
        }
        assert_eq!(b.num_docs() as usize, docs.len());
        b.finish().unwrap()
    }

    fn query_terms<S: ByteSource>(terms: &Terms<S>, stats: &Stats, q: &[&str]) -> Vec<QueryTerm> {
        let infos = terms.lookup(q).unwrap();
        let mut out: Vec<QueryTerm> = Vec::new();
        for (i, info) in infos.into_iter().enumerate() {
            let Some(info) = info else { continue };
            if let Some(e) = out.iter_mut().find(|e| e.info == info) {
                e.weight += 1.0;
            } else {
                let _ = i;
                out.push(QueryTerm {
                    info,
                    idf: stats.idf(info.df as u64),
                    weight: 1.0,
                });
            }
        }
        out
    }

    #[test]
    fn smallfloat_roundtrip_is_monotone_and_exact_below_40() {
        for len in 0..40u32 {
            assert_eq!(byte_to_len(len_to_byte(len)), len, "{len}");
        }
        let mut prev = 0;
        for len in [40u32, 41, 64, 100, 200, 205, 1000, 5000, 65536, 1 << 20] {
            let d = byte_to_len(len_to_byte(len));
            assert!(d <= len && d >= prev, "{len} -> {d}");
            assert!((len - d) as f64 / (len as f64) < 0.13, "{len} -> {d}");
            // re-encoding the representative is stable
            assert_eq!(len_to_byte(d), len_to_byte(len));
            prev = d;
        }
    }

    #[test]
    fn build_lookup_search_matches_reference() {
        let docs = corpus();
        for lengths in [Lengths::Exact, Lengths::Byte] {
            for bucket_terms in [1u32, 5, 64, DEFAULT_BUCKET_TERMS] {
                let params = LexParams {
                    lengths,
                    bucket_terms,
                    ..LexParams::default()
                };
                let bytes = build(&docs, params);
                let terms = Terms::open(bytes.terms.as_slice()).unwrap();
                let postings = Postings::open(bytes.postings.as_slice()).unwrap();
                assert_eq!(terms.num_docs() as usize, docs.len());
                assert_eq!(terms.lengths(), lengths);
                assert_eq!(terms.params(), params);
                assert_eq!(terms.nbytes() as usize, bytes.terms.len());
                let total: u64 = docs.iter().map(|d| d.1.len() as u64).sum();
                assert_eq!(
                    terms.stats(),
                    Stats {
                        num_docs: docs.len() as u64,
                        total_len: total
                    }
                );
                let stats = terms.stats();
                let scoring = stats.scoring(terms.k1(), terms.b());
                // every term resolves and decodes back to its postings
                let entries = terms.entries().unwrap();
                assert_eq!(entries.len() as u32, terms.len());
                assert!(entries.windows(2).all(|w| w[0].0 < w[1].0));
                let mut postings_total = 0u64;
                for (t, info) in &entries {
                    let got = terms.lookup(&[t.as_str()]).unwrap()[0].unwrap();
                    assert_eq!(got, *info);
                    let dec = postings.decode(info).unwrap();
                    assert_eq!(dec.len() as u32, info.df);
                    assert!(dec.windows(2).all(|w| w[0].doc < w[1].doc));
                    for p in &dec {
                        let d = docs.iter().find(|d| d.0 == p.doc).unwrap();
                        assert_eq!(p.tf as usize, d.1.iter().filter(|x| *x == t).count());
                        if lengths == Lengths::Exact {
                            assert_eq!(p.len as usize, d.1.len());
                        } else {
                            assert_eq!(p.len, byte_to_len(len_to_byte(d.1.len() as u32)));
                        }
                    }
                    postings_total += info.df as u64;
                }
                assert_eq!(postings_total, terms.num_postings());
                assert_eq!(
                    terms.lookup(&["nope", "", "zzzz", "deplo"]).unwrap(),
                    vec![None; 4]
                );
                for q in [
                    vec!["deploy"],
                    vec!["galaxy", "budget"],
                    vec!["deploy", "deploy", "w3"],
                    vec!["nope"],
                    vec!["filler"],
                ] {
                    let qt = query_terms(&terms, &stats, &q);
                    let got = postings.search(&qt, &scoring, 20, None).unwrap();
                    let want = reference(&docs, &q, &params);
                    let want: Vec<(u32, f32)> = want.into_iter().take(20).collect();
                    if lengths == Lengths::Exact {
                        assert_eq!(
                            got.iter().map(|h| (h.id, h.score)).collect::<Vec<_>>(),
                            want,
                            "{q:?} bt {bucket_terms}"
                        );
                    } else {
                        // Byte lengths: same candidates, scores within a few percent.
                        let gi: Vec<u32> = got.iter().map(|h| h.id).collect();
                        let wi: Vec<u32> = want.iter().map(|w| w.0).collect();
                        assert_eq!(gi.len(), wi.len());
                        for h in &got {
                            let w = want.iter().find(|w| w.0 == h.id).unwrap().1;
                            assert!((h.score - w).abs() / w < 0.08, "{q:?}: {} vs {w}", h.score);
                        }
                    }
                }
                // tombstones never take a slot
                let qt = query_terms(&terms, &stats, &["deploy"]);
                let all = postings.search(&qt, &scoring, 100, None).unwrap();
                let dead = Tombstones::from_ids(&[all[0].id, all[1].id]);
                let live = postings.search(&qt, &scoring, 3, Some(&dead)).unwrap();
                assert_eq!(
                    live.iter().map(|h| h.id).collect::<Vec<_>>(),
                    all[2..5].iter().map(|h| h.id).collect::<Vec<_>>()
                );
                assert!(postings.search(&qt, &scoring, 0, None).unwrap().is_empty());
            }
        }
    }

    #[test]
    fn builder_rejects_bad_input() {
        let mut b = LexBuilder::new(LexParams::default());
        b.add(1, &toks("a b")).unwrap();
        assert!(b.add(1, &toks("c")).is_err());
        assert!(b.add(u32::MAX, &toks("c")).is_err());
        assert!(LexBuilder::new(LexParams {
            bucket_terms: 0,
            ..LexParams::default()
        })
        .finish()
        .is_err());
        let empty = LexBuilder::new(LexParams::default()).finish().unwrap();
        let t = Terms::open(empty.terms.as_slice()).unwrap();
        assert_eq!((t.len(), t.num_docs(), t.bucket_count()), (0, 0, 0));
        assert_eq!(t.lookup(&["x"]).unwrap(), vec![None]);
        let p = Postings::open(empty.postings.as_slice()).unwrap();
        assert!(p
            .search(&[], &Stats::default().scoring(1.2, 0.75), 5, None)
            .unwrap()
            .is_empty());
        // corrupt headers are format errors, a corrupt entry an OutOfBounds, never a panic
        let good = build(&corpus(), LexParams::default());
        let mut bad = good.terms.clone();
        bad[0] = b'X';
        assert!(matches!(Terms::open(bad.as_slice()), Err(Error::Format(_))));
        let mut bad = good.terms.clone();
        bad[16] = 200;
        assert!(matches!(Terms::open(bad.as_slice()), Err(Error::Format(_))));
        assert!(matches!(
            Terms::open(&good.terms[..good.terms.len() - 1]),
            Err(Error::Format(_))
        ));
        assert!(matches!(
            Postings::open(&good.terms[..]),
            Err(Error::Format(_))
        ));
        let mut bad = good.terms.clone();
        bad[TERMS_HEADER_LEN as usize + 8] = 255; // bucket length past the section
        let t = Terms::open(bad.as_slice()).unwrap();
        assert!(matches!(
            t.lookup(&["deploy"]),
            Err(SourceError::OutOfBounds(_))
        ));
        // a posting range outside the postings section
        let p = Postings::open(good.postings.as_slice()).unwrap();
        let outside = TermInfo {
            df: 3,
            postings: Range::new(good.postings.len() as u64 - 2, 10),
        };
        assert!(matches!(
            p.decode(&outside),
            Err(SourceError::OutOfBounds(_))
        ));
        let _ = p.decode(&TermInfo {
            df: 1000,
            postings: Range::new(POSTINGS_HEADER_LEN, 3),
        }); // truncated list decodes to zeros, no panic
    }

    #[test]
    fn segments_score_like_one_index_and_merge_back() {
        let docs = corpus();
        let params = LexParams {
            bucket_terms: 16,
            ..LexParams::default()
        };
        let whole = build(&docs, params);
        let wt = Terms::open(whole.terms.as_slice()).unwrap();
        let wp = Postings::open(whole.postings.as_slice()).unwrap();
        // three segments by id ranges (interleaved ids)
        let parts: Vec<Vec<(u32, Vec<String>)>> = (0..3)
            .map(|s| {
                docs.iter()
                    .filter(|d| (d.0 as usize) % 3 == s)
                    .cloned()
                    .collect()
            })
            .collect();
        let segs: Vec<LexBytes> = parts.iter().map(|p| build(p, params)).collect();
        type Pair<'a> = (Terms<&'a [u8]>, Postings<&'a [u8]>);
        let readers: Vec<Pair<'_>> = segs
            .iter()
            .map(|s| {
                (
                    Terms::open(s.terms.as_slice()).unwrap(),
                    Postings::open(s.postings.as_slice()).unwrap(),
                )
            })
            .collect();
        let mut stats = Stats::default();
        for (t, _) in &readers {
            stats.add(t.stats());
        }
        assert_eq!(stats, wt.stats());
        let scoring = stats.scoring(params.k1, params.b);
        for q in [
            vec!["deploy"],
            vec!["galaxy", "budget", "deploy"],
            vec!["w3", "w4", "filler"],
        ] {
            let want = wp
                .search(&query_terms(&wt, &stats, &q), &scoring, 15, None)
                .unwrap();
            // global df: sum of the segments' df per term
            let per_seg: Vec<Vec<Option<TermInfo>>> =
                readers.iter().map(|(t, _)| t.lookup(&q).unwrap()).collect();
            let dfs: Vec<u64> = (0..q.len())
                .map(|i| {
                    per_seg
                        .iter()
                        .map(|s| s[i].map_or(0, |x| x.df as u64))
                        .sum()
                })
                .collect();
            let mut hits = Vec::new();
            for (si, (_, p)) in readers.iter().enumerate() {
                let qt: Vec<QueryTerm> = (0..q.len())
                    .filter_map(|i| {
                        per_seg[si][i].map(|info| QueryTerm {
                            info,
                            idf: stats.idf(dfs[i]),
                            weight: 1.0,
                        })
                    })
                    .collect();
                hits.push(p.search(&qt, &scoring, 15, None).unwrap());
            }
            let got = merge_hits(&hits, 15);
            assert_eq!(got, want, "{q:?}");
        }
        // merging the segments gives the whole index back (byte-identical, since every document
        // has at least... no: id 7 is empty and drops out, so compare on the non-empty corpus)
        let non_empty: Vec<(u32, Vec<String>)> =
            docs.iter().filter(|d| !d.1.is_empty()).cloned().collect();
        let want = build(&non_empty, params);
        let inputs: Vec<MergeInput<'_, &[u8]>> = readers
            .iter()
            .map(|(t, p)| MergeInput {
                terms: t,
                postings: p,
                skip: None,
            })
            .collect();
        let merged = merge(&inputs, params).unwrap().finish().unwrap();
        assert_eq!(merged, want);
        // with tombstones the dead documents vanish
        let dead = Tombstones::from_ids(&[10, 42, 2000]);
        let inputs: Vec<MergeInput<'_, &[u8]>> = readers
            .iter()
            .map(|(t, p)| MergeInput {
                terms: t,
                postings: p,
                skip: Some(&dead),
            })
            .collect();
        let alive: Vec<(u32, Vec<String>)> = non_empty
            .iter()
            .filter(|d| !dead.contains(d.0))
            .cloned()
            .collect();
        assert_eq!(
            merge(&inputs, params).unwrap().finish().unwrap(),
            build(&alive, params)
        );
    }

    /// 3000 documents of 40 words from a 2000-word vocabulary (plus a few from `corpus`).
    fn big_corpus() -> Vec<(u32, Vec<String>)> {
        let mut seed = 7u64;
        let mut rnd = |m: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % m
        };
        let mut docs = corpus();
        for i in 0..3000u32 {
            let n = 20 + rnd(40) as usize;
            let t: Vec<String> = (0..n).map(|_| format!("v{}", rnd(2000))).collect();
            docs.push((100_000 + i, t));
        }
        docs
    }

    #[test]
    fn demand_paging_is_two_rounds_after_the_directory() {
        let docs = big_corpus();
        let params = LexParams {
            bucket_terms: 8,
            ..LexParams::default()
        };
        let bytes = build(&docs, params);
        // one blob: terms then postings, at offsets like a container would place them
        let t_base = 1024u64;
        let p_base = align8(t_base + bytes.terms.len() as u64) + 512;
        let mut blob = vec![0u8; p_base as usize + bytes.postings.len()];
        blob[t_base as usize..t_base as usize + bytes.terms.len()].copy_from_slice(&bytes.terms);
        blob[p_base as usize..].copy_from_slice(&bytes.postings);
        let t_win = Range::new(t_base, bytes.terms.len() as u64);
        let p_win = Range::new(p_base, bytes.postings.len() as u64);
        let mut pf = Prefetched::new(blob.len() as u64);
        let feed = |pf: &mut Prefetched, rs: &[Range]| {
            for r in rs {
                pf.insert(r.offset, blob[r.offset as usize..r.end() as usize].to_vec())
                    .unwrap();
            }
        };
        // headers
        let e = Terms::open(crate::source::SubSource::new(&pf, t_win).unwrap())
            .err()
            .unwrap();
        assert_eq!(
            e.missing(),
            Some(&[Range::new(t_base, TERMS_HEADER_LEN)][..])
        );
        feed(
            &mut pf,
            &[
                Range::new(t_base, TERMS_HEADER_LEN),
                Range::new(p_base, POSTINGS_HEADER_LEN),
            ],
        );
        let direct_t = Terms::open(bytes.terms.as_slice()).unwrap();
        let direct_p = Postings::open(bytes.postings.as_slice()).unwrap();
        let stats = direct_t.stats();
        let scoring = stats.scoring(params.k1, params.b);
        let q = ["galaxy", "budget", "v17", "v1999", "nope"];
        let want = direct_p
            .search(&query_terms(&direct_t, &stats, &q), &scoring, 10, None)
            .unwrap();
        assert!(!want.is_empty());
        // round 0: the directory
        {
            let t = Terms::open(crate::source::SubSource::new(&pf, t_win).unwrap()).unwrap();
            let dir = t.plan_directory();
            assert_eq!(
                dir.ranges(),
                &[Range::new(t_base + TERMS_HEADER_LEN, t.directory_nbytes())]
            );
            assert_eq!(
                t.lookup(&q),
                Err(SourceError::Missing(dir.ranges().to_vec()))
            );
            feed(&mut pf, dir.ranges());
        }
        // round 1: the buckets (Missing == plan_lookup)
        let infos = {
            let t = Terms::open(crate::source::SubSource::new(&pf, t_win).unwrap()).unwrap();
            let plan = t.plan_lookup(&q).unwrap();
            assert!(!plan.is_empty() && plan.total_bytes() < bytes.terms.len() as u64 / 4);
            assert_eq!(
                t.lookup(&q),
                Err(SourceError::Missing(plan.ranges().to_vec()))
            );
            feed(&mut pf, plan.ranges());
            let t = Terms::open(crate::source::SubSource::new(&pf, t_win).unwrap()).unwrap();
            t.lookup(&q).unwrap()
        };
        assert_eq!(infos, direct_t.lookup(&q).unwrap());
        // round 2: the posting lists (Missing == plan)
        let qt: Vec<QueryTerm> = infos
            .iter()
            .flatten()
            .map(|i| QueryTerm {
                info: *i,
                idf: stats.idf(i.df as u64),
                weight: 1.0,
            })
            .collect();
        let p = Postings::open(crate::source::SubSource::new(&pf, p_win).unwrap()).unwrap();
        let plan = p
            .plan(&infos.iter().flatten().copied().collect::<Vec<_>>())
            .unwrap();
        assert!(plan.ranges().iter().all(|r| p_win.contains(r)));
        assert_eq!(
            p.search(&qt, &scoring, 10, None),
            Err(SourceError::Missing(plan.ranges().to_vec()))
        );
        feed(&mut pf, plan.ranges());
        let p = Postings::open(crate::source::SubSource::new(&pf, p_win).unwrap()).unwrap();
        assert_eq!(p.search(&qt, &scoring, 10, None).unwrap(), want);
        assert!(plan.total_bytes() < bytes.postings.len() as u64 / 4);
        assert!(pf.held_bytes() < blob.len() as u64 / 4);
    }
}
