//! wasm-bindgen surface of the lexical side: the tokenizer, the segment builder, a demand-paged
//! segment reader, segment merging, rank fusion and excerpts. The host (the TypeScript SDK) does
//! every read and write; the reader follows the demand-paging protocol of `thread_core::source`:
//! a call either succeeds or throws a JS `Error` whose `name` is `"MissingRanges"` and whose
//! `ranges` property lists `[{offset, len}]`, every blob-absolute range its current round still
//! needs. The host fetches them (one multi-range request), `feed`s them and calls again.
//!
//! Other errors are `Error`s named `IndexFormat` (bad magic/version/header/config),
//! `IndexCorrupt` (an out-of-bounds read on a validated section), `IndexError` (not opened, no
//! such section) and `RangeError` (bad arguments). Numbers cross as `f64` (offsets, lengths),
//! `u32` (ids, counts) and typed arrays; structured results are built as JSON and parsed with
//! `JSON.parse` (no `js-sys` dependency).

use thread_core::container::{self, Container};
use thread_core::docs::DocEntry;
use thread_core::error::Error;
use thread_core::excerpts::{self, DocLines, Term};
use thread_core::fusion::{rrf, RrfList};
use thread_core::lex::{Lengths, LexParams, QueryTerm, Stats, TermInfo};
use thread_core::rerank;
use thread_core::segment::{self, Segment, SegmentBuilder, SegmentOptions};
use thread_core::source::{ByteSource, Plan, Prefetched, Range, SourceError};
use thread_core::tokenize::{Tokenizer, TokenizerConfig};
use thread_core::tomb::Tombstones;
use thread_core::topk::{pack_hits, TopK};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = JSON, js_name = parse)]
    fn json_parse(text: &str) -> JsValue;
    #[wasm_bindgen(js_namespace = Reflect, js_name = set)]
    fn reflect_set(target: &JsValue, key: &JsValue, value: &JsValue) -> bool;
}

const MAX_SAFE: f64 = 9007199254740992.0;

fn to_u64(x: f64, what: &str) -> Result<u64, String> {
    if !x.is_finite() || x < 0.0 || x.fract() != 0.0 || x > MAX_SAFE {
        return Err(format!(
            "{what} must be a non-negative integer <= 2^53, got {x}"
        ));
    }
    Ok(x as u64)
}

fn flat_to_ranges(flat: &[f64]) -> Result<Vec<Range>, String> {
    if !flat.len().is_multiple_of(2) {
        return Err(format!(
            "ranges must be a flat [offset, len, ..] array of even length, got {}",
            flat.len()
        ));
    }
    flat.chunks(2)
        .map(|p| Ok(Range::new(to_u64(p[0], "offset")?, to_u64(p[1], "len")?)))
        .collect()
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn str_list_json(items: &[String]) -> String {
    let mut s = String::from("[");
    for (i, it) in items.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&json_str(it));
    }
    s.push(']');
    s
}

fn ranges_json(ranges: &[Range]) -> String {
    let mut s = String::with_capacity(ranges.len() * 32 + 2);
    s.push('[');
    for (i, r) in ranges.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(r#"{{"offset":{},"len":{}}}"#, r.offset, r.len));
    }
    s.push(']');
    s
}

fn ranges_value(ranges: &[Range]) -> JsValue {
    json_parse(&ranges_json(ranges))
}

fn js_error(name: &str, message: &str) -> JsValue {
    let e = JsValue::from(JsError::new(message));
    reflect_set(&e, &JsValue::from_str("name"), &JsValue::from_str(name));
    e
}

fn missing_error(ranges: &[Range]) -> JsValue {
    let total: u64 = ranges.iter().map(|r| r.len).sum();
    let e = js_error(
        "MissingRanges",
        &format!(
            "{} byte range(s) ({total} bytes) not fetched yet",
            ranges.len()
        ),
    );
    reflect_set(&e, &JsValue::from_str("ranges"), &ranges_value(ranges));
    e
}

fn source_error(e: &SourceError) -> JsValue {
    match e {
        SourceError::Missing(r) => missing_error(r),
        SourceError::OutOfBounds(_) => js_error("IndexCorrupt", &e.to_string()),
        SourceError::Io(m) => js_error("IndexError", m),
    }
}

fn index_error(e: &Error) -> JsValue {
    match e {
        Error::Source(s) => source_error(s),
        Error::Format(m) => js_error("IndexFormat", m),
    }
}

fn arg_error(m: String) -> JsValue {
    js_error("RangeError", &m)
}

fn tokenizer_config(json: Option<String>) -> Result<Option<TokenizerConfig>, JsValue> {
    match json.as_deref().map(str::trim) {
        None | Some("") | Some("null") => Ok(None),
        Some(j) => TokenizerConfig::from_json(j)
            .map(Some)
            .map_err(|e| index_error(&e)),
    }
}

#[allow(clippy::too_many_arguments)]
fn segment_options(
    tokenizer_json: Option<String>,
    k1: f64,
    b: f64,
    lengths: &str,
    bucket_terms: u32,
    docs: bool,
    text: bool,
    bucket_docs: u32,
) -> Result<SegmentOptions, JsValue> {
    let lengths = Lengths::from_name(lengths).ok_or_else(|| {
        arg_error(format!(
            "lengths must be \"exact\" or \"byte\", got {lengths:?}"
        ))
    })?;
    if !(k1.is_finite() && k1 >= 0.0 && b.is_finite() && (0.0..=1.0).contains(&b)) {
        return Err(arg_error(format!(
            "k1 must be >= 0 and b in [0, 1], got k1 {k1} b {b}"
        )));
    }
    Ok(SegmentOptions {
        tokenizer: tokenizer_config(tokenizer_json)?,
        lex: LexParams {
            k1,
            b,
            lengths,
            bucket_terms: if bucket_terms == 0 {
                LexParams::default().bucket_terms
            } else {
                bucket_terms
            },
        },
        docs,
        text,
        bucket_docs: if bucket_docs == 0 {
            SegmentOptions::default().bucket_docs
        } else {
            bucket_docs
        },
    })
}

// ---- tokenizer ----------------------------------------------------------------------------------

/// The index's tokenizer (bm25s-compatible: lowercase, `\w\w+`, English stopwords, Snowball
/// English by default; see `thread_core::tokenize`). Keep one per isolate per config: it caches
/// stems. `config_json` is the `tokenizer` object of the manifest / a segment's `meta`; empty or
/// null is the default config.
#[wasm_bindgen]
pub struct LexTokenizer {
    cfg: TokenizerConfig,
    tok: Tokenizer,
}

#[wasm_bindgen]
impl LexTokenizer {
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: Option<String>) -> Result<LexTokenizer, JsValue> {
        let cfg = tokenizer_config(config_json)?.unwrap_or_default();
        let tok = Tokenizer::new(&cfg).map_err(|e| index_error(&e))?;
        Ok(LexTokenizer { cfg, tok })
    }

    /// Tokens of `text`, in order.
    pub fn tokenize(&mut self, text: &str) -> Vec<String> {
        self.tok.tokenize(text)
    }

    /// The config as the canonical JSON object.
    pub fn config(&self) -> String {
        self.cfg.to_json()
    }
}

/// The canonical JSON of a tokenizer config (defaults filled in), or an error for an invalid one.
#[wasm_bindgen]
pub fn tokenizer_config_json(config_json: Option<String>) -> Result<String, JsValue> {
    let cfg = tokenizer_config(config_json)?.unwrap_or_default();
    Tokenizer::new(&cfg).map_err(|e| index_error(&e))?;
    Ok(cfg.to_json())
}

// ---- segment builder ------------------------------------------------------------------------------

/// Builds one segment (the lexical index and/or the docs entries of a batch of documents).
/// `tokenizer_json` is the tokenizer config (`"{}"` = the default config); null or empty = **no
/// lexical index** (a docs-only segment). `lengths` `"exact"` | `"byte"`; `bucket_terms` /
/// `bucket_docs` 0 = defaults.
#[wasm_bindgen]
pub struct SegmentJob {
    b: SegmentBuilder,
}

#[wasm_bindgen]
impl SegmentJob {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenizer_json: Option<String>,
        k1: f64,
        b: f64,
        lengths: &str,
        bucket_terms: u32,
        docs: bool,
        text: bool,
        bucket_docs: u32,
    ) -> Result<SegmentJob, JsValue> {
        let opts = segment_options(
            tokenizer_json,
            k1,
            b,
            lengths,
            bucket_terms,
            docs,
            text,
            bucket_docs,
        )?;
        Ok(SegmentJob {
            b: SegmentBuilder::new(&opts).map_err(|e| index_error(&e))?,
        })
    }

    /// Adds a document (`text` indexed and, when the segment stores texts, kept; `meta` and
    /// `attrs`, the document's attribute word (default 0), stored in `docs`). Returns the tokens
    /// indexed. Errors on a duplicate id (nothing added).
    pub fn add(
        &mut self,
        id: u32,
        text: Option<String>,
        meta: Option<String>,
        attrs: Option<u32>,
    ) -> Result<u32, JsValue> {
        self.b
            .add(id, text.as_deref(), meta.as_deref(), attrs.unwrap_or(0))
            .map(|n| n as u32)
            .map_err(|e| index_error(&e))
    }

    /// Documents added so far.
    pub fn len(&self) -> u32 {
        self.b.len()
    }

    pub fn is_empty(&self) -> bool {
        self.b.is_empty()
    }

    /// Tokens indexed so far.
    pub fn tokens(&self) -> f64 {
        self.b.tokens() as f64
    }

    /// The segment bytes (a `THREADIX` container).
    pub fn finish(self) -> Result<Vec<u8>, JsValue> {
        self.b.finish().map_err(|e| index_error(&e))
    }
}

/// The `meta` JSON of a whole segment held in memory (tools and tests).
#[wasm_bindgen]
pub fn segment_meta(bytes: &[u8]) -> Result<String, JsValue> {
    Ok(Segment::open(bytes)
        .map_err(|e| index_error(&e))?
        .meta()
        .to_string())
}

/// Merges whole segments (`concat` = their bytes back to back, `lens` their lengths, **oldest
/// first**) into one, leaving out each segment's tombstoned ids (`tombs` = the sets' `u32` LE
/// buffers back to back, `tomb_lens` their byte lengths, one per segment; 0 = none). The lexical
/// index is rebuilt with the given parameters (`tokenizer_json` null = keep the inputs'),
/// `docs` / `text` say what the merged segment keeps.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn merge_segments(
    concat: &[u8],
    lens: &[f64],
    tombs: &[u8],
    tomb_lens: &[u32],
    tokenizer_json: Option<String>,
    k1: f64,
    b: f64,
    lengths: &str,
    bucket_terms: u32,
    docs: bool,
    text: bool,
    bucket_docs: u32,
) -> Result<Vec<u8>, JsValue> {
    let opts = segment_options(
        tokenizer_json,
        k1,
        b,
        lengths,
        bucket_terms,
        docs,
        text,
        bucket_docs,
    )?;
    if tomb_lens.len() != lens.len() {
        return Err(arg_error(format!(
            "{} tombstone sets for {} segments",
            tomb_lens.len(),
            lens.len()
        )));
    }
    let mut parts: Vec<&[u8]> = Vec::with_capacity(lens.len());
    let mut at = 0usize;
    for &l in lens {
        let l = to_u64(l, "segment length").map_err(arg_error)? as usize;
        let end = at
            .checked_add(l)
            .filter(|&e| e <= concat.len())
            .ok_or_else(|| arg_error("segment lengths exceed the buffer".into()))?;
        parts.push(&concat[at..end]);
        at = end;
    }
    if at != concat.len() {
        return Err(arg_error(format!(
            "segment lengths sum to {at}, buffer has {}",
            concat.len()
        )));
    }
    let mut sets: Vec<Option<Tombstones>> = Vec::with_capacity(lens.len());
    let mut at = 0usize;
    for &l in tomb_lens {
        let end = at
            .checked_add(l as usize)
            .filter(|&e| e <= tombs.len())
            .ok_or_else(|| arg_error("tombstone lengths exceed the buffer".into()))?;
        sets.push(if l == 0 {
            None
        } else {
            Some(Tombstones::from_bytes(&tombs[at..end]).map_err(|e| arg_error(e.to_string()))?)
        });
        at = end;
    }
    let inputs: Vec<(&[u8], Option<&Tombstones>)> = parts
        .iter()
        .zip(&sets)
        .map(|(p, s)| (*p, s.as_ref()))
        .collect();
    segment::merge(&inputs, &opts).map_err(|e| index_error(&e))
}

// ---- paged segment reader -------------------------------------------------------------------------

/// One segment object under demand paging. `feed` the ranges a call reports as `MissingRanges`
/// and call again. Opening: `plan_open()` (the container header + table, one range), `open()`,
/// then `plan_headers()` and `plan_directories()` are the small ranges a host keeps resident so
/// that a search is two rounds (`lookup`: the terms' buckets; `search`: their posting lists) and
/// a docs lookup one or two (`docs_get`: buckets, then texts).
#[wasm_bindgen]
pub struct SegmentReader {
    source: Prefetched,
    seg: Option<Segment>,
    last_missing: Vec<Range>,
}

type Sub<'a> = thread_core::source::SubSource<'a, Prefetched>;

impl SegmentReader {
    fn fail(&mut self, e: SourceError) -> JsValue {
        if let SourceError::Missing(r) = &e {
            self.last_missing = r.clone();
        }
        source_error(&e)
    }

    fn fail_index(&mut self, e: Error) -> JsValue {
        match e {
            Error::Source(s) => self.fail(s),
            e => index_error(&e),
        }
    }

    fn seg(&self) -> Result<&Segment, JsValue> {
        self.seg
            .as_ref()
            .ok_or_else(|| js_error("IndexError", "open() has not succeeded yet"))
    }

    fn section(&self, name: &str) -> Result<Range, JsValue> {
        self.seg()?.section(name).ok_or_else(|| {
            js_error(
                "IndexError",
                &format!("this segment has no {name:?} section"),
            )
        })
    }

    /// Runs `f` over a section window; errors become JS exceptions, `Missing` is remembered.
    fn with_section<T>(
        &mut self,
        name: &str,
        f: impl FnOnce(Sub<'_>) -> Result<T, Error>,
    ) -> Result<T, JsValue> {
        let r = self.section(name)?;
        let sub = match thread_core::source::SubSource::new(&self.source, r) {
            Ok(s) => s,
            Err(e) => return Err(self.fail(e)),
        };
        match f(sub) {
            Ok(v) => Ok(v),
            Err(e) => Err(self.fail_index(e)),
        }
    }

    fn with_terms<T>(
        &mut self,
        f: impl FnOnce(&thread_core::lex::Terms<Sub<'_>>) -> Result<T, SourceError>,
    ) -> Result<T, JsValue> {
        self.with_section(container::SEC_TERMS, |sub| {
            let t = thread_core::lex::Terms::open(sub)?;
            f(&t).map_err(Error::Source)
        })
    }

    fn with_postings<T>(
        &mut self,
        f: impl FnOnce(&thread_core::lex::Postings<Sub<'_>>) -> Result<T, SourceError>,
    ) -> Result<T, JsValue> {
        self.with_section(container::SEC_POSTINGS, |sub| {
            let p = thread_core::lex::Postings::open(sub)?;
            f(&p).map_err(Error::Source)
        })
    }

    fn with_docs<T>(
        &mut self,
        f: impl FnOnce(&thread_core::docs::Docs<Sub<'_>>) -> Result<T, SourceError>,
    ) -> Result<T, JsValue> {
        self.with_section(container::SEC_DOCS, |sub| {
            let d = thread_core::docs::Docs::open(sub)?;
            f(&d).map_err(Error::Source)
        })
    }
}

fn infos_from_flat(flat: &[f64]) -> Result<Vec<Option<TermInfo>>, JsValue> {
    if !flat.len().is_multiple_of(3) {
        return Err(arg_error(format!(
            "term infos must be a flat [df, offset, len, ..] array, got {} numbers",
            flat.len()
        )));
    }
    flat.chunks(3)
        .map(|c| {
            if c[0] < 0.0 {
                return Ok(None);
            }
            let df = to_u64(c[0], "df").map_err(arg_error)?;
            if df > u32::MAX as u64 {
                return Err(arg_error(format!("df {df} does not fit u32")));
            }
            Ok(Some(TermInfo {
                df: df as u32,
                postings: Range::new(
                    to_u64(c[1], "offset").map_err(arg_error)?,
                    to_u64(c[2], "len").map_err(arg_error)?,
                ),
            }))
        })
        .collect()
}

#[wasm_bindgen]
impl SegmentReader {
    /// A reader over a segment object of `total_len` bytes that holds nothing yet.
    #[wasm_bindgen(constructor)]
    pub fn new(total_len: f64) -> Result<SegmentReader, JsValue> {
        let len = to_u64(total_len, "totalLen").map_err(arg_error)?;
        Ok(SegmentReader {
            source: Prefetched::new(len),
            seg: None,
            last_missing: Vec::new(),
        })
    }

    pub fn total_len(&self) -> f64 {
        self.source.len() as f64
    }

    /// Inserts fetched bytes (touching or overlapping pieces merge; new bytes win). Feeding more
    /// than asked for is fine.
    pub fn feed(&mut self, offset: f64, bytes: &[u8]) -> Result<(), JsValue> {
        let offset = to_u64(offset, "offset").map_err(arg_error)?;
        self.source
            .insert(offset, bytes.to_vec())
            .map_err(|e| self.fail(e))
    }

    /// Forgets `[offset, offset + len)`.
    pub fn evict(&mut self, offset: f64, len: f64) -> Result<(), JsValue> {
        let r = Range::new(
            to_u64(offset, "offset").map_err(arg_error)?,
            to_u64(len, "len").map_err(arg_error)?,
        );
        self.source.evict(r);
        Ok(())
    }

    /// Forgets the given ranges (flat `[offset, len, ..]`): what a query fetched and no later
    /// query is likely to need (posting lists, buckets), keeping the headers and directories.
    pub fn evict_ranges(&mut self, flat: &[f64]) -> Result<(), JsValue> {
        for r in flat_to_ranges(flat).map_err(arg_error)? {
            self.source.evict(r);
        }
        Ok(())
    }

    pub fn evict_all(&mut self) {
        self.source.clear();
    }

    pub fn held_bytes(&self) -> f64 {
        self.source.held_bytes() as f64
    }

    pub fn holds(&self, offset: f64, len: f64) -> Result<bool, JsValue> {
        let r = Range::new(
            to_u64(offset, "offset").map_err(arg_error)?,
            to_u64(len, "len").map_err(arg_error)?,
        );
        Ok(self.source.holds(r))
    }

    /// The held pieces, `[{offset, len}]` ascending.
    pub fn pieces(&self) -> JsValue {
        let pieces: Vec<Range> = self.source.pieces().map(|(r, _)| r).collect();
        ranges_value(&pieces)
    }

    /// The parts of the given ranges (flat `[offset, len, ..]`) not held yet, merged.
    pub fn missing(&self, flat: &[f64]) -> Result<JsValue, JsValue> {
        let ranges = flat_to_ranges(flat).map_err(arg_error)?;
        Ok(ranges_value(&self.source.missing(&ranges)))
    }

    /// The ranges of the last call that threw `MissingRanges`.
    pub fn last_missing(&self) -> JsValue {
        ranges_value(&self.last_missing)
    }

    /// What to fetch before `open()`: the container header + a 16-entry table (one range,
    /// clipped to the object), which usually covers the small `meta` section too.
    pub fn plan_open(&self) -> JsValue {
        let len = self.source.len();
        let clipped: Vec<Range> = Container::plan_open()
            .into_iter()
            .filter_map(|r| {
                let end = r.end().min(len);
                (r.offset < end).then(|| Range::span(r.offset, end))
            })
            .collect();
        ranges_value(&clipped)
    }

    /// Opens the container and reads `meta`. Returns `{hasLex, hasDocs, tokenizer: json|null,
    /// meta: json, sections: [{name, offset, len}]}`.
    pub fn open(&mut self) -> Result<JsValue, JsValue> {
        let seg = match Segment::open(&self.source) {
            Ok(s) => s,
            Err(e) => return Err(self.fail_index(e)),
        };
        let mut s = format!(
            r#"{{"hasLex":{},"hasDocs":{},"tokenizer":{},"meta":{},"sections":["#,
            seg.has_lex(),
            seg.has_docs(),
            seg.tokenizer()
                .map(|c| c.to_json())
                .unwrap_or_else(|| "null".into()),
            seg.meta()
        );
        for (i, (name, r)) in seg.container().sections().iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                r#"{{"name":{},"offset":{},"len":{}}}"#,
                json_str(name),
                r.offset,
                r.len
            ));
        }
        s.push_str("]}");
        self.seg = Some(seg);
        Ok(json_parse(&s))
    }

    /// The section headers (first bytes of `terms`, `postings`, `docs`): fetch once, keep.
    pub fn plan_headers(&self) -> Result<JsValue, JsValue> {
        let seg = self.seg()?;
        let mut plan = Plan::new();
        for (name, head) in [
            (container::SEC_TERMS, thread_core::lex::TERMS_HEADER_LEN),
            (
                container::SEC_POSTINGS,
                thread_core::lex::POSTINGS_HEADER_LEN,
            ),
            (container::SEC_DOCS, thread_core::docs::HEADER_LEN),
        ] {
            if let Some(r) = seg.section(name) {
                plan.push(Range::new(r.offset, head.min(r.len)));
            }
        }
        plan.coalesce(thread_core::lex::PLAN_GAP);
        Ok(ranges_value(plan.ranges()))
    }

    /// The segment's counts, from the section headers (`MissingRanges` until `plan_headers` is
    /// held): `{numDocs, totalLen, numTerms, numPostings, k1, b, lengths, bucketTerms,
    /// termsDirectoryBytes, termsBytes, postingsBytes, docs: {n, hasText, bucketDocs,
    /// directoryBytes, textBytes} | null}`; `numDocs` etc. are 0 without a lexical index.
    pub fn info(&mut self) -> Result<JsValue, JsValue> {
        let has_lex = self.seg()?.has_lex();
        let has_docs = self.seg()?.has_docs();
        let mut acc: Vec<Range> = Vec::new();
        let mut lex_json = String::from(
            r#""numDocs":0,"totalLen":0,"numTerms":0,"numPostings":0,"k1":0,"b":0,"lengths":null,"bucketTerms":0,"termsDirectoryBytes":0,"termsBytes":0,"postingsBytes":0"#,
        );
        if has_lex {
            let postings_len = self.section(container::SEC_POSTINGS)?.len;
            match self.with_terms(|t| {
                Ok(format!(
                    r#""numDocs":{},"totalLen":{},"numTerms":{},"numPostings":{},"k1":{},"b":{},"lengths":{},"bucketTerms":{},"termsDirectoryBytes":{},"termsBytes":{},"postingsBytes":{}"#,
                    t.num_docs(),
                    t.total_len(),
                    t.len(),
                    t.num_postings(),
                    t.k1(),
                    t.b(),
                    json_str(t.lengths().name()),
                    t.bucket_terms(),
                    t.directory_nbytes(),
                    t.nbytes(),
                    postings_len
                ))
            }) {
                Ok(j) => lex_json = j,
                Err(e) => {
                    if self.last_missing.is_empty() {
                        return Err(e);
                    }
                    acc.append(&mut self.last_missing);
                }
            }
        }
        let mut docs_json = String::from("null");
        if has_docs {
            match self.with_docs(|d| {
                Ok(format!(
                    r#"{{"n":{},"hasText":{},"bucketDocs":{},"directoryBytes":{},"textBytes":{},"nbytes":{}}}"#,
                    d.n(),
                    d.has_text(),
                    d.bucket_docs(),
                    d.directory_nbytes(),
                    d.text_nbytes(),
                    d.nbytes()
                ))
            }) {
                Ok(j) => docs_json = j,
                Err(e) => {
                    if self.last_missing.is_empty() {
                        return Err(e);
                    }
                    acc.append(&mut self.last_missing);
                }
            }
        }
        if !acc.is_empty() {
            let mut p = Plan::from(acc);
            p.coalesce(thread_core::lex::PLAN_GAP);
            return Err(self.fail(SourceError::Missing(p.into_ranges())));
        }
        Ok(json_parse(&format!("{{{lex_json},\"docs\":{docs_json}}}")))
    }

    /// The terms and docs directories (needs the headers): fetch once, keep.
    pub fn plan_directories(&mut self) -> Result<JsValue, JsValue> {
        let has_lex = self.seg()?.has_lex();
        let has_docs = self.seg()?.has_docs();
        let mut plan = Plan::new();
        if has_lex {
            plan.append(self.with_terms(|t| Ok(t.plan_directory()))?);
        }
        if has_docs {
            plan.append(self.with_docs(|d| Ok(d.plan_directory()))?);
        }
        plan.coalesce(thread_core::lex::PLAN_GAP);
        Ok(ranges_value(plan.ranges()))
    }

    /// The segment's counts for the corpus-wide statistics: `[numDocs, totalLen]` (needs the
    /// terms header).
    pub fn stats(&mut self) -> Result<Vec<f64>, JsValue> {
        if !self.seg()?.has_lex() {
            return Ok(vec![0.0, 0.0]);
        }
        self.with_terms(|t| Ok(vec![t.num_docs() as f64, t.total_len() as f64]))
    }

    /// The bucket ranges `lookup(terms)` will read.
    pub fn plan_lookup(&mut self, terms: Vec<String>) -> Result<JsValue, JsValue> {
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        let plan = self.with_terms(|t| t.plan_lookup(&refs))?;
        Ok(ranges_value(plan.ranges()))
    }

    /// Looks the terms up: three numbers per term, `[df, offset, len]` of its posting list
    /// (`df = -1` when the segment lacks the term). One round cold after the directory.
    pub fn lookup(&mut self, terms: Vec<String>) -> Result<Vec<f64>, JsValue> {
        if !self.seg()?.has_lex() {
            return Ok(vec![-1.0; terms.len() * 3]
                .chunks(3)
                .flat_map(|c| [c[0], 0.0, 0.0])
                .collect());
        }
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        let infos = self.with_terms(|t| t.lookup(&refs))?;
        Ok(infos
            .into_iter()
            .flat_map(|i| match i {
                Some(i) => [i.df as f64, i.postings.offset as f64, i.postings.len as f64],
                None => [-1.0, 0.0, 0.0],
            })
            .collect())
    }

    /// The posting-list ranges `search(infos, ..)` will read.
    pub fn plan_search(&mut self, infos: &[f64]) -> Result<JsValue, JsValue> {
        let infos: Vec<TermInfo> = infos_from_flat(infos)?.into_iter().flatten().collect();
        let plan = self.with_postings(|p| p.plan(&infos))?;
        Ok(ranges_value(plan.ranges()))
    }

    /// Scores this segment's documents: `infos` as `lookup` returned them (3 per term, -1 df =
    /// skip), `dfs` the **corpus-wide** document frequency of each term and `weights` its query
    /// weight (occurrences in the query), `num_docs` / `total_len` the corpus-wide counts
    /// (`stats()` summed over the segments), `k1` / `b` the index's parameters. Returns the top
    /// `k` as packed hits (`[count u32][id u32, score f32, idx u32 = 0]*`), best first, ids in
    /// `tombstones` (`u32` LE, sorted unique; empty = none) never taking a slot. One round cold.
    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &mut self,
        infos: &[f64],
        dfs: &[f64],
        weights: &[f32],
        num_docs: f64,
        total_len: f64,
        k1: f64,
        b: f64,
        k: u32,
        tombstones: &[u8],
    ) -> Result<Vec<u8>, JsValue> {
        let infos = infos_from_flat(infos)?;
        if dfs.len() != infos.len() || weights.len() != infos.len() {
            return Err(arg_error(format!(
                "{} infos, {} dfs, {} weights",
                infos.len(),
                dfs.len(),
                weights.len()
            )));
        }
        let stats = Stats {
            num_docs: to_u64(num_docs, "numDocs").map_err(arg_error)?,
            total_len: to_u64(total_len, "totalLen").map_err(arg_error)?,
        };
        let scoring = stats.scoring(k1, b);
        let terms: Vec<QueryTerm> = infos
            .iter()
            .zip(dfs)
            .zip(weights)
            .filter_map(|((i, &df), &w)| {
                i.map(|info| QueryTerm {
                    info,
                    idf: stats.idf(df.max(0.0) as u64),
                    weight: w,
                })
            })
            .collect();
        let skip = if tombstones.is_empty() {
            None
        } else {
            Some(Tombstones::from_bytes(tombstones).map_err(|e| arg_error(e.to_string()))?)
        };
        if !self.seg()?.has_lex() || terms.is_empty() {
            return Ok(pack_hits(&TopK::new(0).into_sorted()));
        }
        let hits = self.with_postings(|p| p.search(&terms, &scoring, k as usize, skip.as_ref()))?;
        Ok(pack_hits(&hits))
    }

    /// The bucket ranges `docs_get(ids)` reads first.
    pub fn plan_docs_lookup(&mut self, ids: &[u32]) -> Result<JsValue, JsValue> {
        let plan = self.with_docs(|d| d.plan_lookup(ids))?;
        Ok(ranges_value(plan.ranges()))
    }

    /// `[{id, meta, text, attrs} | null]` for `ids` (`text` is `null` unless `with_text` and the
    /// segment stores texts; `attrs` is the document's attribute word, 0 when the section stores
    /// none). One round cold after the directory for the entries, one more for the texts.
    pub fn docs_get(&mut self, ids: &[u32], with_text: bool) -> Result<JsValue, JsValue> {
        if !self.seg()?.has_docs() {
            return Ok(json_parse(&format!(
                "[{}]",
                vec!["null"; ids.len()].join(",")
            )));
        }
        let (entries, texts): (Vec<Option<DocEntry>>, Option<Vec<String>>) =
            self.with_docs(|d| {
                let entries = d.lookup(ids)?;
                let texts = if with_text && d.has_text() {
                    let present: Vec<DocEntry> = entries.iter().flatten().cloned().collect();
                    Some(d.texts(&present)?)
                } else {
                    None
                };
                Ok((entries, texts))
            })?;
        let mut tx = texts.map(|t| t.into_iter());
        let mut s = String::from("[");
        for (i, e) in entries.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            match e {
                None => s.push_str("null"),
                Some(e) => {
                    let text = tx
                        .as_mut()
                        .and_then(|it| it.next())
                        .map(|t| json_str(&t))
                        .unwrap_or_else(|| "null".into());
                    s.push_str(&format!(
                        r#"{{"id":{},"meta":{},"text":{},"attrs":{}}}"#,
                        e.id,
                        json_str(&e.meta),
                        text,
                        e.attrs
                    ));
                }
            }
        }
        s.push(']');
        Ok(json_parse(&s))
    }
}

// ---- fusion ---------------------------------------------------------------------------------------

/// A fused result list: `ids` (`Uint32Array`) and float64 RRF `scores` (`Float64Array`), best
/// first, ties by ascending id.
#[wasm_bindgen]
pub struct FusedHits {
    ids: Vec<u32>,
    scores: Vec<f64>,
}

#[wasm_bindgen]
impl FusedHits {
    #[wasm_bindgen(getter)]
    pub fn ids(&self) -> Vec<u32> {
        self.ids.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn scores(&self) -> Vec<f64> {
        self.scores.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn len(&self) -> u32 {
        self.ids.len() as u32
    }
    #[wasm_bindgen(getter)]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

fn split_lists<'a>(flat: &'a [u32], lens: &[u32]) -> Result<Vec<&'a [u32]>, JsValue> {
    let total: usize = lens.iter().map(|&l| l as usize).sum();
    if total != flat.len() {
        return Err(arg_error(format!(
            "list lengths sum to {total} but {} ids were given",
            flat.len()
        )));
    }
    let mut out = Vec::with_capacity(lens.len());
    let mut at = 0;
    for &l in lens {
        out.push(&flat[at..at + l as usize]);
        at += l as usize;
    }
    Ok(out)
}

/// Reciprocal Rank Fusion of any number of ranked lists (`flat_ids` = the lists' ids back to
/// back, `lens` their lengths, `weights` one per list), in list order: `score = Σ weight /
/// (rrf_k + rank)` with rank from 1, float64, ties by ascending id, top `top_n`.
#[wasm_bindgen]
pub fn rrf_fuse(
    flat_ids: &[u32],
    lens: &[u32],
    weights: &[f64],
    rrf_k: f64,
    top_n: u32,
) -> Result<FusedHits, JsValue> {
    let lists = split_lists(flat_ids, lens)?;
    if weights.len() != lists.len() {
        return Err(arg_error(format!(
            "{} weights for {} lists",
            weights.len(),
            lists.len()
        )));
    }
    let rrf_lists: Vec<RrfList<'_>> = lists
        .iter()
        .zip(weights)
        .map(|(ids, &w)| RrfList::new(w, ids))
        .collect();
    let (ids, scores) = rrf(&rrf_lists, rrf_k, top_n as usize).into_iter().unzip();
    Ok(FusedHits { ids, scores })
}

/// Each window document's best rank over several candidate lists of one family (`flat_ids` /
/// `lens` as in `rrf_fuse`): distinct documents are numbered by first appearance per list and
/// the minimum kept; 0 when no list has the document.
#[wasm_bindgen]
pub fn best_doc_ranks(flat_ids: &[u32], lens: &[u32], window: &[u32]) -> Result<Vec<u32>, JsValue> {
    let lists = split_lists(flat_ids, lens)?;
    let best = rerank::best_doc_ranks(&lists, |id| id);
    Ok(window
        .iter()
        .map(|d| best.get(d).copied().unwrap_or(0))
        .collect())
}

fn opt_u32_json(v: Option<u32>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "null".into(),
    }
}

/// Rank fusion around a host-side judge (`thread_core::rerank::fuse`): `window_len` documents
/// (the first of the fused list, in order); `owners[p]` the window index of passage `p`,
/// `scores[p]` the judge's score for it (higher = better); `lexical[d]` / `dense[d]` the window
/// documents' best list ranks from `best_doc_ranks` (0 = none). Returns `{order, signals:
/// [{jev, fused, lexical, dense, score}], passage_rank}`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn rerank_fuse(
    window_len: u32,
    owners: &[u32],
    scores: &[f64],
    lexical: &[u32],
    dense: &[u32],
    k: f64,
    w_jev: f64,
    w_fused: f64,
    w_lexical: f64,
    w_dense: f64,
) -> Result<JsValue, JsValue> {
    if owners.len() != scores.len() {
        return Err(arg_error(format!(
            "owners ({}) and scores ({}) differ in length",
            owners.len(),
            scores.len()
        )));
    }
    let n = window_len as usize;
    if owners.iter().any(|&o| o as usize >= n) {
        return Err(arg_error("a passage owner is outside the window".into()));
    }
    let to_opt = |v: &[u32]| -> Vec<Option<u32>> {
        (0..n)
            .map(|d| v.get(d).copied().filter(|&r| r > 0))
            .collect()
    };
    let owners: Vec<usize> = owners.iter().map(|&o| o as usize).collect();
    let w = rerank::Weights {
        k,
        jev: w_jev,
        fused: w_fused,
        lexical: w_lexical,
        dense: w_dense,
    };
    let f = rerank::fuse(n, &owners, scores, &to_opt(lexical), &to_opt(dense), &w);
    let order: Vec<String> = f.order.iter().map(|d| d.to_string()).collect();
    let signals: Vec<String> = f
        .signals
        .iter()
        .map(|sg| {
            format!(
                r#"{{"jev":{},"fused":{},"lexical":{},"dense":{},"score":{}}}"#,
                opt_u32_json(sg.jev),
                sg.fused,
                opt_u32_json(sg.lexical),
                opt_u32_json(sg.dense),
                sg.score
            )
        })
        .collect();
    let ranks: Vec<String> = f.passage_rank.iter().map(|r| r.to_string()).collect();
    Ok(json_parse(&format!(
        r#"{{"order":[{}],"signals":[{}],"passage_rank":[{}]}}"#,
        order.join(","),
        signals.join(","),
        ranks.join(",")
    )))
}

// ---- excerpts -------------------------------------------------------------------------------------

fn ranges_pairs_json(ranges: &[(usize, usize)]) -> String {
    let parts: Vec<String> = ranges.iter().map(|(a, b)| format!("[{a},{b}]")).collect();
    format!("[{}]", parts.join(","))
}

fn terms_json(terms: &[Term]) -> String {
    let parts: Vec<String> = terms
        .iter()
        .map(|t| {
            format!(
                r#"{{"term":{},"tokens":{},"source":{}}}"#,
                json_str(&t.term),
                str_list_json(&t.tokens),
                json_str(t.source.name())
            )
        })
        .collect();
    format!("[{}]", parts.join(","))
}

fn excerpt_json(e: &excerpts::Excerpt, text: Option<&str>) -> String {
    let mut s = format!(
        r#"{{"a":{},"b":{},"score":{},"terms":{}"#,
        e.a,
        e.b,
        e.score,
        str_list_json(&e.terms)
    );
    if let Some(t) = text {
        s.push_str(&format!(r#","text":{}"#, json_str(t)));
    }
    s.push('}');
    s
}

/// `"12-30, 45, L50-L60"` as merged `[[a, b], ..]` clamped to `1..=n`.
#[wasm_bindgen]
pub fn parse_ranges(spec: &str, n: u32) -> JsValue {
    json_parse(&ranges_pairs_json(&excerpts::parse_ranges(
        spec, n as usize,
    )))
}

/// Whitespace collapsed and cut to `n` characters with an ellipsis.
#[wasm_bindgen]
pub fn cut_text(text: &str, n: u32) -> String {
    excerpts::cut(text, n as usize)
}

/// A passage label: `"<title or fallback> · lines a-b of n"`.
#[wasm_bindgen]
pub fn passage_label(title: &str, fallback: &str, a: u32, b: u32, n: u32) -> String {
    excerpts::passage_label(title, fallback, a as usize, b as usize, n as usize)
}

/// A document on its line grid (`excerpts::DocLines`); the host keeps one per fetched text.
#[wasm_bindgen]
pub struct Lines {
    dl: DocLines,
}

#[wasm_bindgen]
impl Lines {
    /// The line grid of `text` (physical lines, blank ones dropped, long ones wrapped at 200).
    #[wasm_bindgen(constructor)]
    pub fn new(text: &str) -> Lines {
        Lines {
            dl: DocLines::new(text),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn n(&self) -> u32 {
        self.dl.n() as u32
    }

    pub fn lines(&self) -> Vec<String> {
        self.dl.lines.clone()
    }

    /// Lines `a..=b` (1-based) joined with spaces.
    pub fn plain(&self, a: u32, b: u32) -> String {
        self.dl.plain(a as usize, b as usize)
    }

    /// Lines `a..=b`, one per row, each prefixed with its right-aligned number.
    pub fn numbered(&self, a: u32, b: u32, indent: &str) -> String {
        self.dl.numbered(a as usize, b as usize, indent)
    }

    pub fn chars(&self, a: u32, b: u32) -> u32 {
        self.dl.chars(a as usize, b as usize) as u32
    }

    /// `"<label> — <plain(a, b) cut to max_chars>"`: the passage text shown to a judge.
    pub fn passage_text(&self, label: &str, a: u32, b: u32, max_chars: u32) -> String {
        excerpts::passage_text(label, &self.dl, a as usize, b as usize, max_chars as usize)
    }

    /// The lines of a range spec (`"12-30, 45"`), plain or numbered, joined by newlines.
    pub fn read(&self, spec: &str, numbered: bool) -> String {
        let rs = excerpts::parse_ranges(spec, self.dl.n());
        rs.iter()
            .map(|&(a, b)| {
                if numbered {
                    self.dl.numbered(a, b, "")
                } else {
                    self.dl.plain(a, b)
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// The query terms of one question, applied to any number of documents.
#[wasm_bindgen]
pub struct Excerpter {
    terms: Vec<Term>,
}

#[wasm_bindgen]
impl Excerpter {
    #[wasm_bindgen(constructor)]
    pub fn new(
        query: &str,
        hints: Vec<String>,
        variants: Vec<String>,
        hint_terms: u32,
    ) -> Excerpter {
        let h: Vec<&str> = hints.iter().map(String::as_str).collect();
        let v: Vec<&str> = variants.iter().map(String::as_str).collect();
        Excerpter {
            terms: excerpts::query_terms(query, &h, &v, hint_terms as usize),
        }
    }

    /// `[{term, tokens, source}]`.
    pub fn terms(&self) -> JsValue {
        json_parse(&terms_json(&self.terms))
    }

    /// The excerpts of a document: `{lines, matched, windows: [{a, b, score, terms, text}], more:
    /// [{a, b, score, terms}]}`. `budget` in characters, `max_windows` windows, `before`/`after`
    /// lines of context, `more_cap` matching lines reported but not shown.
    pub fn find(
        &self,
        lines: &Lines,
        budget: u32,
        max_windows: u32,
        before: u32,
        after: u32,
        more_cap: u32,
    ) -> JsValue {
        let ex = excerpts::find(
            &lines.dl,
            &self.terms,
            budget as usize,
            max_windows as usize,
            before as usize,
            after as usize,
            more_cap as usize,
        );
        let windows: Vec<String> = ex
            .windows
            .iter()
            .map(|w| excerpt_json(w, Some(&lines.dl.plain(w.a, w.b))))
            .collect();
        let more: Vec<String> = ex.more.iter().map(|w| excerpt_json(w, None)).collect();
        json_parse(&format!(
            r#"{{"lines":{},"matched":{},"windows":[{}],"more":[{}]}}"#,
            lines.dl.n(),
            ex.matched,
            windows.join(","),
            more.join(",")
        ))
    }
}
