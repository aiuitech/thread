//! The lexical side: the tokenizer, the segment builder, segment merging and a demand-paged
//! segment reader. The reader follows the protocol of `thread_core::source` (as `thread-wasm`
//! does): a call either succeeds or raises `MissingRanges`, whose `ranges` attribute lists every
//! blob-absolute `(offset, len)` its current round still needs; the host fetches them, `feed`s
//! them and calls again.
//!
//! Readers and tokenizers are safe to share between Python threads: their state sits behind a
//! mutex that is only taken with the GIL released.

use crate::{
    opt_slice, u32_vec, value_err, Buf, IndexCorruptError, IndexFormatError, IndexStateError,
    MissingRanges,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::sync::{Mutex, MutexGuard};
use thread_core::container::{self, Container};
use thread_core::docs::DocEntry;
use thread_core::error::Error;
use thread_core::lex::{Lengths, LexParams, QueryTerm, Stats, TermInfo};
use thread_core::segment::{self, Segment, SegmentBuilder, SegmentOptions};
use thread_core::source::{ByteSource, Plan, Prefetched, Range, SourceError};
use thread_core::tokenize::{Tokenizer, TokenizerConfig};
use thread_core::tomb::Tombstones;
use thread_core::topk::{pack_hits, TopK};

/// An error built without the GIL, turned into a Python exception once it is held again.
enum LexErr {
    Missing(Vec<Range>),
    Corrupt(String),
    State(String),
    Format(String),
    Arg(String),
}

impl LexErr {
    fn into_py(self, py: Python<'_>) -> PyErr {
        match self {
            LexErr::Missing(ranges) => {
                let total: u64 = ranges.iter().map(|r| r.len).sum();
                let e = MissingRanges::new_err(format!(
                    "{} byte range(s) ({total} bytes) not fetched yet",
                    ranges.len()
                ));
                let pairs: Vec<(u64, u64)> = ranges.iter().map(|r| (r.offset, r.len)).collect();
                if let Err(set) = e.value(py).setattr("ranges", pairs) {
                    return set;
                }
                e
            }
            LexErr::Corrupt(m) => IndexCorruptError::new_err(m),
            LexErr::State(m) => IndexStateError::new_err(m),
            LexErr::Format(m) => IndexFormatError::new_err(m),
            LexErr::Arg(m) => value_err(m),
        }
    }
}

impl From<SourceError> for LexErr {
    fn from(e: SourceError) -> Self {
        match e {
            SourceError::Missing(r) => LexErr::Missing(r),
            e @ SourceError::OutOfBounds(_) => LexErr::Corrupt(e.to_string()),
            SourceError::Io(m) => LexErr::State(m),
        }
    }
}

impl From<Error> for LexErr {
    fn from(e: Error) -> Self {
        match e {
            Error::Source(s) => s.into(),
            Error::Format(m) => LexErr::Format(m),
        }
    }
}

fn ranges_out(ranges: &[Range]) -> Vec<(u64, u64)> {
    ranges.iter().map(|r| (r.offset, r.len)).collect()
}

fn ranges_in(pairs: &[(u64, u64)]) -> Vec<Range> {
    pairs.iter().map(|&(o, l)| Range::new(o, l)).collect()
}

fn tokenizer_config(json: Option<&str>) -> Result<Option<TokenizerConfig>, LexErr> {
    match json.map(str::trim) {
        None | Some("") | Some("null") => Ok(None),
        Some(j) => TokenizerConfig::from_json(j)
            .map(Some)
            .map_err(LexErr::from),
    }
}

#[allow(clippy::too_many_arguments)]
fn segment_options(
    tokenizer_json: Option<&str>,
    k1: f64,
    b: f64,
    lengths: &str,
    bucket_terms: u32,
    docs: bool,
    text: bool,
    bucket_docs: u32,
) -> Result<SegmentOptions, LexErr> {
    let lengths = Lengths::from_name(lengths).ok_or_else(|| {
        LexErr::Arg(format!(
            "lengths must be \"exact\" or \"byte\", got {lengths:?}"
        ))
    })?;
    if !(k1.is_finite() && k1 >= 0.0 && b.is_finite() && (0.0..=1.0).contains(&b)) {
        return Err(LexErr::Arg(format!(
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

// ---- tokenizer --------------------------------------------------------------------------------

/// The index's tokenizer (bm25s-compatible: lowercase, `\w\w+`, English stopwords, Snowball
/// English by default). `config_json` is the `tokenizer` object of the manifest; empty or None is
/// the default config. Caches stems; keep one per config.
#[pyclass(frozen, module = "thread_sdk._native")]
pub struct LexTokenizer {
    config: String,
    tok: Mutex<Tokenizer>,
}

#[pymethods]
impl LexTokenizer {
    #[new]
    #[pyo3(signature = (config_json = None))]
    fn new(py: Python<'_>, config_json: Option<&str>) -> PyResult<Self> {
        let cfg = tokenizer_config(config_json)
            .map_err(|e| e.into_py(py))?
            .unwrap_or_default();
        let tok = Tokenizer::new(&cfg).map_err(|e| LexErr::from(e).into_py(py))?;
        Ok(LexTokenizer {
            config: cfg.to_json(),
            tok: Mutex::new(tok),
        })
    }

    /// Tokens of `text`, in order.
    fn tokenize(&self, py: Python<'_>, text: &str) -> Vec<String> {
        py.detach(|| {
            self.tok
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .tokenize(text)
        })
    }

    /// The config as canonical JSON (defaults filled in).
    fn config(&self) -> String {
        self.config.clone()
    }
}

/// The canonical JSON of a tokenizer config, or an error for an invalid one.
#[pyfunction]
#[pyo3(signature = (config_json = None))]
fn tokenizer_config_json(py: Python<'_>, config_json: Option<&str>) -> PyResult<String> {
    let cfg = tokenizer_config(config_json)
        .map_err(|e| e.into_py(py))?
        .unwrap_or_default();
    Tokenizer::new(&cfg).map_err(|e| LexErr::from(e).into_py(py))?;
    Ok(cfg.to_json())
}

// ---- segment builder ----------------------------------------------------------------------------

/// Builds one segment (the BM25 index and/or the docs entries of a batch of documents).
/// `tokenizer_json` `"{}"` is the default tokenizer; None or empty = no lexical index (docs only).
#[pyclass(module = "thread_sdk._native")]
pub struct SegmentJob {
    b: Option<SegmentBuilder>,
}

impl SegmentJob {
    fn builder(&mut self) -> PyResult<&mut SegmentBuilder> {
        self.b
            .as_mut()
            .ok_or_else(|| value_err("SegmentJob is finished"))
    }
}

#[pymethods]
impl SegmentJob {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        tokenizer_json: Option<&str>,
        k1: f64,
        b: f64,
        lengths: &str,
        bucket_terms: u32,
        docs: bool,
        text: bool,
        bucket_docs: u32,
    ) -> PyResult<Self> {
        let opts = segment_options(
            tokenizer_json,
            k1,
            b,
            lengths,
            bucket_terms,
            docs,
            text,
            bucket_docs,
        )
        .map_err(|e| e.into_py(py))?;
        Ok(SegmentJob {
            b: Some(SegmentBuilder::new(&opts).map_err(|e| LexErr::from(e).into_py(py))?),
        })
    }

    /// Adds a document. Returns the tokens indexed. Errors on a duplicate id (nothing added).
    #[pyo3(signature = (id, text = None, meta = None))]
    fn add(
        &mut self,
        py: Python<'_>,
        id: u32,
        text: Option<&str>,
        meta: Option<&str>,
    ) -> PyResult<u32> {
        let r = self.builder()?.add(id, text, meta);
        r.map(|n| n as u32).map_err(|e| LexErr::from(e).into_py(py))
    }

    fn __len__(&self) -> usize {
        self.b.as_ref().map_or(0, |b| b.len() as usize)
    }

    /// Tokens indexed so far.
    #[getter]
    fn tokens(&self) -> u64 {
        self.b.as_ref().map_or(0, |b| b.tokens())
    }

    /// The segment bytes (a `THREADIX` container); the job cannot be used afterwards.
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let b = self
            .b
            .take()
            .ok_or_else(|| value_err("SegmentJob is finished"))?;
        let bytes = py
            .detach(|| b.finish())
            .map_err(|e| LexErr::from(e).into_py(py))?;
        Ok(PyBytes::new(py, &bytes))
    }
}

/// The `meta` JSON of a whole segment held in memory.
#[pyfunction]
fn segment_meta(py: Python<'_>, segment: Buf) -> PyResult<String> {
    Segment::open(segment.as_slice())
        .map(|s| s.meta().to_string())
        .map_err(|e| LexErr::from(e).into_py(py))
}

/// Merges whole segments (**oldest first**) into one, leaving out each segment's tombstoned ids
/// (`tombstones[i]`: `u32` LE ids hidden in segment `i`, or None). The lexical index is rebuilt with
/// the given parameters.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn merge_segments<'py>(
    py: Python<'py>,
    segments: Vec<Buf>,
    tombstones: Vec<Option<Buf>>,
    tokenizer_json: Option<&str>,
    k1: f64,
    b: f64,
    lengths: &str,
    bucket_terms: u32,
    docs: bool,
    text: bool,
    bucket_docs: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let opts = segment_options(
        tokenizer_json,
        k1,
        b,
        lengths,
        bucket_terms,
        docs,
        text,
        bucket_docs,
    )
    .map_err(|e| e.into_py(py))?;
    if tombstones.len() != segments.len() {
        return Err(value_err(format!(
            "{} tombstone sets for {} segments",
            tombstones.len(),
            segments.len()
        )));
    }
    let sets: Vec<Option<Tombstones>> = tombstones
        .iter()
        .map(|t| {
            let s = opt_slice(t);
            if s.is_empty() {
                Ok(None)
            } else {
                Tombstones::from_bytes(s).map(Some).map_err(value_err)
            }
        })
        .collect::<PyResult<_>>()?;
    let inputs: Vec<(&[u8], Option<&Tombstones>)> = segments
        .iter()
        .zip(&sets)
        .map(|(p, s)| (p.as_slice(), s.as_ref()))
        .collect();
    let out = py
        .detach(|| segment::merge(&inputs, &opts))
        .map_err(|e| LexErr::from(e).into_py(py))?;
    Ok(PyBytes::new(py, &out))
}

// ---- paged segment reader -------------------------------------------------------------------------

struct ReaderState {
    source: Prefetched,
    seg: Option<Segment>,
}

type Sub<'a> = thread_core::source::SubSource<'a, Prefetched>;

impl ReaderState {
    fn seg(&self) -> Result<&Segment, LexErr> {
        self.seg
            .as_ref()
            .ok_or_else(|| LexErr::State("open() has not succeeded yet".into()))
    }

    fn section(&self, name: &str) -> Result<Range, LexErr> {
        self.seg()?
            .section(name)
            .ok_or_else(|| LexErr::State(format!("this segment has no {name:?} section")))
    }

    fn with_section<T>(
        &self,
        name: &str,
        f: impl FnOnce(Sub<'_>) -> Result<T, Error>,
    ) -> Result<T, LexErr> {
        let r = self.section(name)?;
        let sub = thread_core::source::SubSource::new(&self.source, r)?;
        Ok(f(sub)?)
    }

    fn with_terms<T>(
        &self,
        f: impl FnOnce(&thread_core::lex::Terms<Sub<'_>>) -> Result<T, SourceError>,
    ) -> Result<T, LexErr> {
        self.with_section(container::SEC_TERMS, |sub| {
            let t = thread_core::lex::Terms::open(sub)?;
            f(&t).map_err(Error::Source)
        })
    }

    fn with_postings<T>(
        &self,
        f: impl FnOnce(&thread_core::lex::Postings<Sub<'_>>) -> Result<T, SourceError>,
    ) -> Result<T, LexErr> {
        self.with_section(container::SEC_POSTINGS, |sub| {
            let p = thread_core::lex::Postings::open(sub)?;
            f(&p).map_err(Error::Source)
        })
    }

    fn with_docs<T>(
        &self,
        f: impl FnOnce(&thread_core::docs::Docs<Sub<'_>>) -> Result<T, SourceError>,
    ) -> Result<T, LexErr> {
        self.with_section(container::SEC_DOCS, |sub| {
            let d = thread_core::docs::Docs::open(sub)?;
            f(&d).map_err(Error::Source)
        })
    }
}

/// Segment counts from the section headers (see `SegmentReader.info`).
struct Info {
    lex: Option<[f64; 11]>,
    lengths: Option<&'static str>,
    docs: Option<(u64, bool, u64, u64, u64, u64)>,
}

/// One segment object under demand paging. Opening: fetch `plan_open()`, call `open()`, then fetch
/// `plan_headers()` and `plan_directories()` and keep them resident, so that a search is two rounds
/// (`lookup`: the terms' buckets; `search`: their posting lists) and a docs lookup one or two.
#[pyclass(frozen, module = "thread_sdk._native")]
pub struct SegmentReader {
    st: Mutex<ReaderState>,
}

impl SegmentReader {
    fn lock(&self) -> MutexGuard<'_, ReaderState> {
        self.st.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Runs `f` on the state with the GIL released (the lock is taken after releasing it, so a
    /// thread waiting for the lock never holds the GIL another thread needs).
    fn run<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut ReaderState) -> Result<T, LexErr> + Send,
    ) -> PyResult<T> {
        py.detach(|| f(&mut self.lock())).map_err(|e| e.into_py(py))
    }
}

#[pymethods]
impl SegmentReader {
    /// A reader over a segment object of `total_len` bytes that holds nothing yet.
    #[new]
    fn new(total_len: u64) -> Self {
        SegmentReader {
            st: Mutex::new(ReaderState {
                source: Prefetched::new(total_len),
                seg: None,
            }),
        }
    }

    #[getter]
    fn total_len(&self) -> u64 {
        self.lock().source.len()
    }

    /// Inserts fetched bytes (overlapping pieces merge; new bytes win).
    fn feed(&self, py: Python<'_>, offset: u64, data: Buf) -> PyResult<()> {
        let bytes = data.as_slice().to_vec();
        self.run(py, |st| Ok(st.source.insert(offset, bytes)?))
    }

    /// Forgets the given `(offset, len)` ranges.
    fn evict_ranges(&self, py: Python<'_>, ranges: Vec<(u64, u64)>) -> PyResult<()> {
        self.run(py, |st| {
            for r in ranges_in(&ranges) {
                st.source.evict(r);
            }
            Ok(())
        })
    }

    fn evict_all(&self, py: Python<'_>) -> PyResult<()> {
        self.run(py, |st| {
            st.source.clear();
            Ok(())
        })
    }

    #[getter]
    fn held_bytes(&self) -> u64 {
        self.lock().source.held_bytes()
    }

    /// The parts of the given ranges not held yet, merged.
    fn missing(&self, py: Python<'_>, ranges: Vec<(u64, u64)>) -> PyResult<Vec<(u64, u64)>> {
        self.run(py, |st| {
            Ok(ranges_out(&st.source.missing(&ranges_in(&ranges))))
        })
    }

    /// What to fetch before `open()`: the container header + table (one range, clipped).
    fn plan_open(&self) -> Vec<(u64, u64)> {
        let len = self.lock().source.len();
        Container::plan_open()
            .into_iter()
            .filter_map(|r| {
                let end = r.end().min(len);
                (r.offset < end).then(|| (r.offset, end - r.offset))
            })
            .collect()
    }

    /// Opens the container and reads `meta`. Returns `{has_lex, has_docs, tokenizer: json | None,
    /// meta: json, sections: [(name, offset, len)]}`.
    fn open<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        type Opened = (bool, bool, Option<String>, String, Vec<(String, u64, u64)>);
        let (has_lex, has_docs, tok, meta, sections): Opened = self.run(py, |st| {
            let seg = Segment::open(&st.source)?;
            let out = (
                seg.has_lex(),
                seg.has_docs(),
                seg.tokenizer().map(|c| c.to_json()),
                seg.meta().to_string(),
                seg.container()
                    .sections()
                    .iter()
                    .map(|(n, r)| (n.to_string(), r.offset, r.len))
                    .collect(),
            );
            st.seg = Some(seg);
            Ok(out)
        })?;
        let d = PyDict::new(py);
        d.set_item("has_lex", has_lex)?;
        d.set_item("has_docs", has_docs)?;
        d.set_item("tokenizer", tok)?;
        d.set_item("meta", meta)?;
        d.set_item("sections", sections)?;
        Ok(d)
    }

    /// The section headers (first bytes of `terms`, `postings`, `docs`): fetch once, keep.
    fn plan_headers(&self, py: Python<'_>) -> PyResult<Vec<(u64, u64)>> {
        self.run(py, |st| {
            let seg = st.seg()?;
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
            Ok(ranges_out(plan.ranges()))
        })
    }

    /// The segment's counts from the section headers (`MissingRanges` until `plan_headers` is held):
    /// `{num_docs, total_len, num_terms, num_postings, k1, b, lengths, bucket_terms,
    /// terms_directory_bytes, terms_bytes, postings_bytes, docs: {n, has_text, bucket_docs,
    /// directory_bytes, text_bytes, nbytes} | None}`; the counts are 0 without a lexical index.
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = self.run(py, |st| {
            let (has_lex, has_docs) = (st.seg()?.has_lex(), st.seg()?.has_docs());
            let mut acc: Vec<Range> = Vec::new();
            let mut out = Info {
                lex: None,
                lengths: None,
                docs: None,
            };
            if has_lex {
                let postings_len = st.section(container::SEC_POSTINGS)?.len;
                match st.with_terms(|t| {
                    Ok((
                        [
                            t.num_docs() as f64,
                            t.total_len() as f64,
                            t.len() as f64,
                            t.num_postings() as f64,
                            t.k1(),
                            t.b(),
                            t.bucket_terms() as f64,
                            t.directory_nbytes() as f64,
                            t.nbytes() as f64,
                            postings_len as f64,
                            0.0,
                        ],
                        t.lengths().name(),
                    ))
                }) {
                    Ok((v, lengths)) => {
                        out.lex = Some(v);
                        out.lengths = Some(lengths);
                    }
                    Err(LexErr::Missing(mut r)) => acc.append(&mut r),
                    Err(e) => return Err(e),
                }
            }
            if has_docs {
                match st.with_docs(|d| {
                    Ok((
                        d.n() as u64,
                        d.has_text(),
                        d.bucket_docs() as u64,
                        d.directory_nbytes(),
                        d.text_nbytes(),
                        d.nbytes(),
                    ))
                }) {
                    Ok(v) => out.docs = Some(v),
                    Err(LexErr::Missing(mut r)) => acc.append(&mut r),
                    Err(e) => return Err(e),
                }
            }
            if !acc.is_empty() {
                let mut p = Plan::from(acc);
                p.coalesce(thread_core::lex::PLAN_GAP);
                return Err(LexErr::Missing(p.into_ranges()));
            }
            Ok(out)
        })?;
        let d = PyDict::new(py);
        let lex = info.lex.unwrap_or([0.0; 11]);
        for (i, name) in ["num_docs", "total_len", "num_terms", "num_postings"]
            .iter()
            .enumerate()
        {
            d.set_item(*name, lex[i] as u64)?;
        }
        d.set_item("k1", lex[4])?;
        d.set_item("b", lex[5])?;
        d.set_item("lengths", info.lengths)?;
        for (i, name) in [
            "bucket_terms",
            "terms_directory_bytes",
            "terms_bytes",
            "postings_bytes",
        ]
        .iter()
        .enumerate()
        {
            d.set_item(*name, lex[6 + i] as u64)?;
        }
        match info.docs {
            None => d.set_item("docs", py.None())?,
            Some((n, has_text, bucket_docs, dir, text, nbytes)) => {
                let dd = PyDict::new(py);
                dd.set_item("n", n)?;
                dd.set_item("has_text", has_text)?;
                dd.set_item("bucket_docs", bucket_docs)?;
                dd.set_item("directory_bytes", dir)?;
                dd.set_item("text_bytes", text)?;
                dd.set_item("nbytes", nbytes)?;
                d.set_item("docs", dd)?;
            }
        }
        Ok(d)
    }

    /// The terms and docs directories (needs the headers): fetch once, keep.
    fn plan_directories(&self, py: Python<'_>) -> PyResult<Vec<(u64, u64)>> {
        self.run(py, |st| {
            let (has_lex, has_docs) = (st.seg()?.has_lex(), st.seg()?.has_docs());
            let mut plan = Plan::new();
            if has_lex {
                plan.append(st.with_terms(|t| Ok(t.plan_directory()))?);
            }
            if has_docs {
                plan.append(st.with_docs(|d| Ok(d.plan_directory()))?);
            }
            plan.coalesce(thread_core::lex::PLAN_GAP);
            Ok(ranges_out(plan.ranges()))
        })
    }

    /// Looks the terms up: `(df, offset, len)` of each term's posting list, `df = -1` when the
    /// segment lacks the term. One round cold after the directory.
    fn lookup(&self, py: Python<'_>, terms: Vec<String>) -> PyResult<Vec<(i64, u64, u64)>> {
        self.run(py, |st| {
            if !st.seg()?.has_lex() {
                return Ok(vec![(-1, 0, 0); terms.len()]);
            }
            let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
            let infos = st.with_terms(|t| t.lookup(&refs))?;
            Ok(infos
                .into_iter()
                .map(|i| match i {
                    Some(i) => (i.df as i64, i.postings.offset, i.postings.len),
                    None => (-1, 0, 0),
                })
                .collect())
        })
    }

    /// Scores this segment's documents: `infos` as `lookup` returned them, `dfs` the corpus-wide
    /// document frequency of each term, `weights` its query weight, `num_docs` / `total_len` the
    /// corpus-wide counts, `k1` / `b` the index's parameters. Returns the top `k` as packed hits,
    /// best first; ids in `tombstones` never take a slot. One round cold.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (infos, dfs, weights, num_docs, total_len, k1, b, k, tombstones = None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        infos: Vec<(i64, u64, u64)>,
        dfs: Vec<f64>,
        weights: Vec<f32>,
        num_docs: u64,
        total_len: u64,
        k1: f64,
        b: f64,
        k: usize,
        tombstones: Option<Buf>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if dfs.len() != infos.len() || weights.len() != infos.len() {
            return Err(value_err(format!(
                "{} infos, {} dfs, {} weights",
                infos.len(),
                dfs.len(),
                weights.len()
            )));
        }
        let tomb = opt_slice(&tombstones);
        let skip = if tomb.is_empty() {
            None
        } else {
            Some(Tombstones::from_bytes(tomb).map_err(value_err)?)
        };
        let stats = Stats {
            num_docs,
            total_len,
        };
        let terms: Vec<QueryTerm> = infos
            .iter()
            .zip(&dfs)
            .zip(&weights)
            .filter(|((&(df, _, _), _), _)| df >= 0)
            .map(|((&(df, off, len), &cdf), &w)| QueryTerm {
                info: TermInfo {
                    df: df as u32,
                    postings: Range::new(off, len),
                },
                idf: stats.idf(cdf.max(0.0) as u64),
                weight: w,
            })
            .collect();
        let packed = self.run(py, |st| {
            if !st.seg()?.has_lex() || terms.is_empty() {
                return Ok(pack_hits(&TopK::new(0).into_sorted()));
            }
            let scoring = stats.scoring(k1, b);
            let hits = st.with_postings(|p| p.search(&terms, &scoring, k, skip.as_ref()))?;
            Ok(pack_hits(&hits))
        })?;
        Ok(PyBytes::new(py, &packed))
    }

    /// `[(id, meta, text | None) | None]` for `ids` (`text` only with `with_text` and a segment that
    /// stores texts). One round cold after the directory for the entries, one more for the texts.
    fn docs_get<'py>(
        &self,
        py: Python<'py>,
        ids: &Bound<'py, PyAny>,
        with_text: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let ids = u32_vec(ids)?;
        type Got = (Vec<Option<DocEntry>>, Option<Vec<String>>);
        let (entries, texts): Got = self.run(py, |st| {
            if !st.seg()?.has_docs() {
                return Ok((vec![None; ids.len()], None));
            }
            st.with_docs(|d| {
                let entries = d.lookup(&ids)?;
                let texts = if with_text && d.has_text() {
                    let present: Vec<DocEntry> = entries.iter().flatten().cloned().collect();
                    Some(d.texts(&present)?)
                } else {
                    None
                };
                Ok((entries, texts))
            })
        })?;
        let mut tx = texts.map(|t| t.into_iter());
        let out = PyList::empty(py);
        for e in entries {
            match e {
                None => out.append(py.None())?,
                Some(e) => {
                    let text = tx.as_mut().and_then(|it| it.next());
                    out.append((e.id, e.meta, text))?;
                }
            }
        }
        Ok(out)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LexTokenizer>()?;
    m.add_class::<SegmentJob>()?;
    m.add_class::<SegmentReader>()?;
    m.add_function(wrap_pyfunction!(tokenizer_config_json, m)?)?;
    m.add_function(wrap_pyfunction!(segment_meta, m)?)?;
    m.add_function(wrap_pyfunction!(merge_segments, m)?)?;
    Ok(())
}
