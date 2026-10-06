//! `thread seg ...` and `thread tokenize`: the lexical side of the index on local files. A segment file is the
//! same `THREADIX` container an upsert writes to the bucket (`idx/{gen}/seg/{seq}.thread`), so a corpus can be
//! indexed here and uploaded, or a downloaded segment inspected and searched. Every command prints JSON.
//!
//! ```text
//! thread seg build --out seg.thread --docs docs.jsonl [--tokenizer <json|file>] [--k1 1.2] [--b 0.75]
//!                  [--lengths exact|byte] [--bucket-terms N] [--bucket-docs N] [--no-docs] [--text]
//!                  [--id-field id] [--text-field text] [--meta-field meta]
//! thread seg search --text "query" [--k 10] [--with-docs] [--tombstones ids.u32] seg.thread [more.thread ...]
//! thread seg inspect seg.thread
//! thread seg merge --out base.thread [--tombstones a.u32,b.u32,...] [build options] seg.thread [more.thread ...]
//! thread tokenize [--tokenizer <json|file>] "text" ["text" ...]
//! thread chunk [--words 200] [--overlap 0] [--file path] ["text" ...]   (chunk.rs)
//! ```
//!
//! `docs.jsonl`: one JSON object per line with an integer `id` (u32), a `text`, optionally an `attrs` (u32, the
//! document's attribute word for filtering) and optionally a `meta` (a string, or
//! any JSON value stored as its text). Field names change with `--id-field` etc. `thread <command> --help` prints
//! each command's options; `USAGE` the overview.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use serde_json::{json, Map, Value};
use thread_core::lex::{Lengths, LexParams, QueryTerm, Stats};
use thread_core::segment::{self, Segment, SegmentBuilder, SegmentOptions};
use thread_core::tokenize::{Tokenizer, TokenizerConfig};
use thread_core::tomb::Tombstones;
use thread_core::topk::Hit;
use thread_core::Error;

pub struct Args {
    flags: HashMap<String, String>,
    pub positional: Vec<String>,
}

impl Args {
    pub fn parse(args: &[String]) -> Result<Args, String> {
        let mut flags = HashMap::new();
        let mut positional = Vec::new();
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            if let Some(name) = a.strip_prefix("--") {
                if let Some((k, v)) = name.split_once('=') {
                    flags.insert(k.to_string(), v.to_string());
                } else if matches!(name, "no-docs" | "with-docs" | "help")
                    || args.get(i + 1).is_none_or(|n| n.starts_with("--"))
                {
                    // boolean flags, and value flags given without a value (`--text` at the end
                    // of `seg build`: store the texts)
                    flags.insert(name.to_string(), "true".to_string());
                } else {
                    flags.insert(name.to_string(), args[i + 1].clone());
                    i += 1;
                }
            } else {
                positional.push(a.clone());
            }
            i += 1;
        }
        Ok(Args { flags, positional })
    }
    pub fn get(&self, k: &str) -> Option<&str> {
        self.flags.get(k).map(String::as_str)
    }
    pub fn flag(&self, k: &str) -> bool {
        self.get(k).is_some()
    }
    pub fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> Result<T, String> {
        match self.get(k) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("--{k}: bad value {v:?}")),
        }
    }
}

fn err<T>(e: impl std::fmt::Display) -> Result<T, String> {
    Err(e.to_string())
}

/// `--tokenizer`: inline JSON or a file holding it; absent = the default config.
fn tokenizer_config(spec: Option<&str>) -> Result<TokenizerConfig, String> {
    let Some(spec) = spec else {
        return Ok(TokenizerConfig::bm25s_default());
    };
    let text = if spec.trim_start().starts_with('{') {
        spec.to_string()
    } else {
        std::fs::read_to_string(spec).map_err(|e| format!("{spec}: {e}"))?
    };
    let cfg = TokenizerConfig::from_json(&text).map_err(|e| e.to_string())?;
    Tokenizer::new(&cfg).map_err(|e| e.to_string())?;
    Ok(cfg)
}

fn segment_options(a: &Args) -> Result<SegmentOptions, String> {
    let s = a.get("lengths").unwrap_or("exact");
    let lengths = Lengths::from_name(s)
        .ok_or_else(|| format!("--lengths must be exact or byte, got {s:?}"))?;
    let docs = !a.flag("no-docs");
    if a.get("text").is_some_and(|v| v != "true") {
        return err("--text takes no value here (it stores the texts); the query goes to `seg search --text`");
    }
    Ok(SegmentOptions {
        tokenizer: Some(tokenizer_config(a.get("tokenizer"))?),
        lex: LexParams {
            k1: a.num("k1", 1.2)?,
            b: a.num("b", 0.75)?,
            lengths,
            bucket_terms: a.num("bucket-terms", LexParams::default().bucket_terms)?,
        },
        docs,
        text: docs && a.flag("text"),
        bucket_docs: a.num("bucket-docs", SegmentOptions::default().bucket_docs)?,
        keys: a.flag("keys"),
        bucket_keys: a.num("bucket-keys", SegmentOptions::default().bucket_keys)?,
    })
}

fn read_tombstones(spec: Option<&str>) -> Result<Vec<Option<Tombstones>>, String> {
    let Some(spec) = spec else {
        return Ok(Vec::new());
    };
    spec.split(',')
        .map(|p| {
            if p.is_empty() || p == "-" {
                return Ok(None);
            }
            let bytes = std::fs::read(p).map_err(|e| format!("{p}: {e}"))?;
            Tombstones::from_bytes(&bytes)
                .map(Some)
                .map_err(|e| format!("{p}: {e}"))
        })
        .collect()
}

fn meta_string(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub fn build(a: &Args) -> Result<Value, String> {
    let t0 = Instant::now();
    let out = a.get("out").ok_or("--out <file.thread> is required")?;
    let docs_path = a.get("docs").ok_or("--docs <docs.jsonl> is required")?;
    let (id_f, text_f, meta_f, key_f) = (
        a.get("id-field").unwrap_or("id"),
        a.get("text-field").unwrap_or("text"),
        a.get("meta-field").unwrap_or("meta"),
        a.get("key-field").unwrap_or("key"),
    );
    let opts = segment_options(a)?;
    // --chunk-words / --chunk-overlap: every document's text is split into flat word windows (chunk.rs) and each
    // chunk is a document of its own, keyed `key#i`, with ids running from --first-id (the id field is not used).
    let chunking = a.get("chunk-words").is_some() || a.get("chunk-overlap").is_some();
    let (chunk_words, chunk_overlap) = (
        a.num("chunk-words", crate::chunk::DEFAULT_WORDS)?,
        a.num("chunk-overlap", crate::chunk::DEFAULT_OVERLAP)?,
    );
    if chunking {
        crate::chunk::check(chunk_words, chunk_overlap)?;
    }
    let mut next_id: u32 = a.num("first-id", 0u32)?;
    let mut b = SegmentBuilder::new(&opts).map_err(|e| e.to_string())?;
    let file = std::fs::File::open(docs_path).map_err(|e| format!("{docs_path}: {e}"))?;
    let mut rows = 0usize;
    let mut chunks = 0usize;
    for (i, line) in BufReader::with_capacity(1 << 20, file).lines().enumerate() {
        let line = line.map_err(|e| format!("{docs_path}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value =
            serde_json::from_str(&line).map_err(|e| format!("{docs_path}:{}: {e}", i + 1))?;
        let id = if chunking {
            0
        } else {
            v.get(id_f)
                .and_then(Value::as_u64)
                .filter(|&x| x < u32::MAX as u64)
                .ok_or_else(|| {
                    format!(
                        "{docs_path}:{}: {id_f:?} must be an integer in [0, 2^32 - 1)",
                        i + 1
                    )
                })? as u32
        };
        let text = v.get(text_f).map(meta_string);
        let meta = v.get(meta_f).map(meta_string);
        let attrs = match v.get("attrs") {
            None | Some(Value::Null) => 0,
            Some(a) => a
                .as_u64()
                .filter(|&x| x <= u32::MAX as u64)
                .ok_or_else(|| {
                    format!(
                        "{docs_path}:{}: `attrs` must be an integer in [0, 2^32)",
                        i + 1
                    )
                })? as u32,
        };
        let key = match v.get(key_f) {
            None | Some(Value::Null) => None,
            Some(Value::String(k)) => Some(k.clone()),
            Some(_) => return err(format!("{docs_path}:{}: {key_f:?} must be a string", i + 1)),
        };
        if chunking {
            let whole = text.unwrap_or_default();
            for c in crate::chunk::chunk_text(&whole, chunk_words, chunk_overlap)? {
                if next_id == u32::MAX {
                    return err(format!("{docs_path}:{}: the id space is exhausted", i + 1));
                }
                let ck = key.as_deref().map(|k| crate::chunk::chunk_key(k, c.index));
                b.add_keyed(next_id, ck.as_deref(), Some(c.text), meta.as_deref(), attrs)
                    .map_err(|e| format!("{docs_path}:{}: {e}", i + 1))?;
                next_id += 1;
                chunks += 1;
            }
        } else {
            b.add_keyed(id, key.as_deref(), text.as_deref(), meta.as_deref(), attrs)
                .map_err(|e| format!("{docs_path}:{}: {e}", i + 1))?;
        }
        rows += 1;
    }
    let bytes = b.finish().map_err(|e| e.to_string())?;
    std::fs::write(out, &bytes).map_err(|e| format!("{out}: {e}"))?;
    let seg = Segment::open(&bytes).map_err(|e| e.to_string())?;
    let mut o = json!({"path": out, "bytes": bytes.len(), "docs": if chunking { chunks } else { rows }, "tokens": b.tokens(), "meta": serde_json::from_str::<Value>(seg.meta()).unwrap_or(Value::Null), "elapsed_ms": t0.elapsed().as_secs_f64() * 1000.0});
    if chunking {
        o["documents"] = json!(rows);
        o["chunk"] = json!({"words": chunk_words, "overlap": chunk_overlap, "firstId": a.num("first-id", 0u32)?, "nextId": next_id});
    }
    Ok(o)
}

pub fn inspect(a: &Args) -> Result<Value, String> {
    let path = a.positional.first().ok_or("a segment file is required")?;
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let seg = Segment::open(&bytes).map_err(|e| e.to_string())?;
    let sections: Vec<Value> = seg
        .container()
        .sections()
        .iter()
        .map(|(n, r)| json!({"name": n, "offset": r.offset, "len": r.len}))
        .collect();
    let mut o = Map::new();
    o.insert("path".into(), json!(path));
    o.insert("bytes".into(), json!(bytes.len()));
    o.insert("sections".into(), Value::Array(sections));
    o.insert(
        "meta".into(),
        serde_json::from_str::<Value>(seg.meta()).unwrap_or(Value::Null),
    );
    if let Some(t) = seg.terms(&bytes) {
        let t = t.map_err(|e| e.to_string())?;
        o.insert("terms".into(), json!({"count": t.len(), "num_docs": t.num_docs(), "total_len": t.total_len(), "num_postings": t.num_postings(), "buckets": t.bucket_count(), "bucket_terms": t.bucket_terms(), "directory_bytes": t.directory_nbytes(), "lengths": t.lengths().name(), "k1": t.k1(), "b": t.b()}));
    }
    if let Some(d) = seg.docs(&bytes) {
        let d = d.map_err(|e| e.to_string())?;
        o.insert("docs".into(), json!({"n": d.n(), "text": d.has_text(), "buckets": d.bucket_count(), "bucket_docs": d.bucket_docs(), "directory_bytes": d.directory_nbytes(), "text_bytes": d.text_nbytes()}));
    }
    if let Some(k) = seg.keys(&bytes) {
        let k = k.map_err(|e| e.to_string())?;
        o.insert("keys".into(), json!({"n": k.n(), "buckets": k.bucket_count(), "bucket_keys": k.bucket_keys(), "directory_bytes": k.directory_nbytes()}));
    }
    Ok(Value::Object(o))
}

/// Resolves keys (`--get a,b`) and/or lists a prefix (`--prefix p [--after k] [--limit n]`) over one
/// or more segments given oldest first, newest copy of a key winning, `--tombstones` applying per
/// segment as in `seg search`.
pub fn keys(a: &Args) -> Result<Value, String> {
    let t0 = Instant::now();
    let segs = open_all(&a.positional)?;
    let tombs = read_tombstones(a.get("tombstones"))?;
    if !tombs.is_empty() && tombs.len() != segs.len() {
        return err(format!(
            "{} tombstone sets for {} segments",
            tombs.len(),
            segs.len()
        ));
    }
    let hidden = |i: usize, id: u32| {
        tombs
            .get(i)
            .and_then(|t| t.as_ref())
            .is_some_and(|t| t.contains(id))
    };
    let mut readers = Vec::new();
    for (i, s) in segs.iter().enumerate() {
        if let Some(k) = s.seg.keys(&s.bytes) {
            readers.push((i, k.map_err(|e| format!("{}: {e}", s.path))?));
        }
    }
    if readers.is_empty() {
        return err("none of the segments has a keys section");
    }
    let mut o = Map::new();
    if let Some(list) = a.get("get") {
        let wanted: Vec<&str> = list.split(',').filter(|k| !k.is_empty()).collect();
        let mut found: Vec<Option<u32>> = vec![None; wanted.len()];
        for (i, k) in readers.iter().rev() {
            let pending: Vec<usize> = (0..wanted.len()).filter(|&j| found[j].is_none()).collect();
            if pending.is_empty() {
                break;
            }
            let ask: Vec<&str> = pending.iter().map(|&j| wanted[j]).collect();
            let ids = k.lookup(&ask).map_err(|e| e.to_string())?;
            for (j, id) in pending.into_iter().zip(ids) {
                if let Some(id) = id.filter(|&id| !hidden(*i, id)) {
                    found[j] = Some(id);
                }
            }
        }
        let got: Map<String, Value> = wanted
            .iter()
            .zip(found)
            .map(|(k, id)| (k.to_string(), id.map_or(Value::Null, Value::from)))
            .collect();
        o.insert("ids".into(), Value::Object(got));
    }
    if let Some(prefix) = a.get("prefix") {
        let limit: usize = a.num("limit", 100)?;
        let after = a.get("after");
        // newest segment first: the first copy of a key seen decides (live, or hidden = dead)
        let mut decided: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut live: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
        let mut more = false;
        for (i, k) in readers.iter().rev() {
            let page = k.list(prefix, after, limit).map_err(|e| e.to_string())?;
            more |= page.more;
            for (key, id) in page.entries {
                if decided.insert(key.clone()) && !hidden(*i, id) {
                    live.insert(key, id);
                }
            }
        }
        let mut entries: Vec<Value> = Vec::new();
        for (key, id) in live {
            if entries.len() == limit {
                more = true;
                break;
            }
            entries.push(json!({"key": key, "id": id}));
        }
        o.insert("entries".into(), Value::Array(entries));
        o.insert("more".into(), json!(more));
    }
    if !o.contains_key("ids") && !o.contains_key("entries") {
        return err("--get <k1,k2,..> and/or --prefix <p> is required");
    }
    o.insert(
        "elapsed_ms".into(),
        json!(t0.elapsed().as_secs_f64() * 1000.0),
    );
    Ok(Value::Object(o))
}

struct Opened {
    path: String,
    bytes: Vec<u8>,
    seg: Segment,
}

fn open_all(paths: &[String]) -> Result<Vec<Opened>, String> {
    if paths.is_empty() {
        return err("at least one segment file is required");
    }
    paths
        .iter()
        .map(|p| {
            let bytes = std::fs::read(p).map_err(|e| format!("{p}: {e}"))?;
            let seg = Segment::open(&bytes).map_err(|e| format!("{p}: {e}"))?;
            Ok(Opened {
                path: p.clone(),
                bytes,
                seg,
            })
        })
        .collect()
}

/// Searches one or more segments as one index (corpus-wide statistics), oldest first with
/// `--tombstones a.u32,b.u32` applying per segment (`-` = none).
pub fn search(a: &Args) -> Result<Value, String> {
    let t0 = Instant::now();
    let text = a.get("text").ok_or("--text is required")?;
    let k: usize = a.num("k", 10)?;
    let segs = open_all(&a.positional)?;
    let tombs = read_tombstones(a.get("tombstones"))?;
    if !tombs.is_empty() && tombs.len() != segs.len() {
        return err(format!(
            "{} tombstone sets for {} segments",
            tombs.len(),
            segs.len()
        ));
    }
    let cfg = segs
        .iter()
        .find_map(|s| s.seg.tokenizer().cloned())
        .ok_or("none of the segments has a lexical index")?;
    let mut tok = Tokenizer::new(&cfg).map_err(|e| e.to_string())?;
    let tokens = tok.tokenize(text);
    let mut weights: Vec<(String, f32)> = Vec::new();
    for t in &tokens {
        match weights.iter_mut().find(|(x, _)| x == t) {
            Some(w) => w.1 += 1.0,
            None => weights.push((t.clone(), 1.0)),
        }
    }
    let distinct: Vec<&str> = weights.iter().map(|(t, _)| t.as_str()).collect();
    // readers
    let mut readers = Vec::new();
    for (i, s) in segs.iter().enumerate() {
        if let (Some(t), Some(p)) = (s.seg.terms(&s.bytes), s.seg.postings(&s.bytes)) {
            readers.push((
                i,
                t.map_err(|e| format!("{}: {e}", s.path))?,
                p.map_err(|e| format!("{}: {e}", s.path))?,
            ));
        }
    }
    let mut stats = Stats::default();
    for (_, t, _) in &readers {
        stats.add(t.stats());
    }
    let infos: Vec<Vec<Option<thread_core::lex::TermInfo>>> = readers
        .iter()
        .map(|(_, t, _)| t.lookup(&distinct).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let dfs: Vec<u64> = (0..distinct.len())
        .map(|j| {
            infos
                .iter()
                .map(|inf| inf[j].map_or(0, |x| x.df as u64))
                .sum()
        })
        .collect();
    let (k1, b) = readers
        .first()
        .map(|(_, t, _)| (t.k1(), t.b()))
        .unwrap_or((1.2, 0.75));
    let scoring = stats.scoring(k1, b);
    let mut all: Vec<Vec<Hit>> = Vec::new();
    for (ri, (si, _, p)) in readers.iter().enumerate() {
        let qt: Vec<QueryTerm> = (0..distinct.len())
            .filter_map(|j| {
                infos[ri][j].map(|info| QueryTerm {
                    info,
                    idf: stats.idf(dfs[j]),
                    weight: weights[j].1,
                })
            })
            .collect();
        let skip = tombs.get(*si).and_then(|t| t.as_ref());
        all.push(
            p.search(&qt, &scoring, k, skip)
                .map_err(|e| e.to_string())?,
        );
    }
    let hits = thread_core::lex::merge_hits(&all, k);
    let mut docs_by_id: HashMap<u32, Value> = HashMap::new();
    if a.flag("with-docs") {
        let ids: Vec<u32> = hits.iter().map(|h| h.id).collect();
        for (si, s) in segs.iter().enumerate().rev() {
            let Some(d) = s.seg.docs(&s.bytes) else {
                continue;
            };
            let d = d.map_err(|e| e.to_string())?;
            let want: Vec<u32> = ids
                .iter()
                .copied()
                .filter(|id| {
                    !docs_by_id.contains_key(id)
                        && !tombs
                            .get(si)
                            .and_then(|t| t.as_ref())
                            .is_some_and(|t| t.contains(*id))
                })
                .collect();
            for (id, got) in want.iter().zip(d.get(&want).map_err(|e| e.to_string())?) {
                if let Some((meta, text)) = got {
                    docs_by_id.insert(*id, json!({"meta": meta, "text": if d.has_text() { Value::String(text) } else { Value::Null }}));
                }
            }
        }
    }
    let hits_json: Vec<Value> = hits
        .iter()
        .map(|h| {
            let mut o = json!({"id": h.id, "score": h.score});
            if a.flag("with-docs") {
                o["doc"] = docs_by_id.get(&h.id).cloned().unwrap_or(Value::Null);
            }
            o
        })
        .collect();
    let df_json: Map<String, Value> = distinct
        .iter()
        .zip(&dfs)
        .map(|(t, d)| (t.to_string(), json!(d)))
        .collect();
    Ok(
        json!({"hits": hits_json, "tokens": tokens, "df": df_json, "num_docs": stats.num_docs, "avg_len": stats.avg_len(), "segments": readers.len(), "elapsed_ms": t0.elapsed().as_secs_f64() * 1000.0}),
    )
}

pub fn merge(a: &Args) -> Result<Value, String> {
    let t0 = Instant::now();
    let out = a.get("out").ok_or("--out <file.thread> is required")?;
    let segs = open_all(&a.positional)?;
    let tombs = read_tombstones(a.get("tombstones"))?;
    if !tombs.is_empty() && tombs.len() != segs.len() {
        return err(format!(
            "{} tombstone sets for {} segments",
            tombs.len(),
            segs.len()
        ));
    }
    let mut opts = segment_options(a)?;
    if a.get("tokenizer").is_none() {
        opts.tokenizer = None; // keep the inputs' tokenizer
    }
    if a.get("text").is_none() {
        opts.text = segs.iter().any(|s| s.seg.meta().contains(r#""text":true"#));
    }
    let inputs: Vec<(&[u8], Option<&Tombstones>)> = segs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.bytes.as_slice(), tombs.get(i).and_then(|t| t.as_ref())))
        .collect();
    let bytes = segment::merge(&inputs, &opts).map_err(|e: Error| e.to_string())?;
    std::fs::write(out, &bytes).map_err(|e| format!("{out}: {e}"))?;
    let seg = Segment::open(&bytes).map_err(|e| e.to_string())?;
    Ok(
        json!({"path": out, "bytes": bytes.len(), "merged": segs.len(), "meta": serde_json::from_str::<Value>(seg.meta()).unwrap_or(Value::Null), "elapsed_ms": t0.elapsed().as_secs_f64() * 1000.0}),
    )
}

pub fn tokenize(a: &Args) -> Result<Value, String> {
    let cfg = tokenizer_config(a.get("tokenizer"))?;
    let mut tok = Tokenizer::new(&cfg).map_err(|e| e.to_string())?;
    let texts: Vec<Vec<String>> = a.positional.iter().map(|t| tok.tokenize(t)).collect();
    Ok(
        json!({"tokenizer": serde_json::from_str::<Value>(&cfg.to_json()).unwrap_or(Value::Null), "tokens": texts}),
    )
}

pub const USAGE: &str = "  thread seg build   --out seg.thread --docs docs.jsonl [options]   build a segment from a JSONL corpus
  thread seg inspect seg.thread                                       sections, meta and per-section counts
  thread seg search  --text \"query\" [--k 10] [--with-docs] [--tombstones a.u32,-] seg.thread [more ...]
  thread seg merge   --out base.thread [--tombstones ...] seg.thread [more ...]   compact segments into one
  thread seg keys    [--get k1,k2] [--prefix p --after k --limit n] [--tombstones ...] seg.thread [more ...]
  thread tokenize    [--tokenizer json|file] \"text\" [\"text\" ...]      the index's tokenizer
  thread chunk       [--words 200] [--overlap 0] [--file path] [\"text\" ...]  flat word-window chunks of a text
  (every command prints one JSON object on stdout; `thread <command> --help` lists its options)";

const BUILD_OPTIONS: &str = "options shared by `seg build` and `seg merge`:
  --tokenizer <json|file>  the tokenizer object: {\"lowercase\": true, \"token_pattern\": \"\\\\w\\\\w+\", \"stopwords\": \"en\"|null,
                           \"stemmer\": \"english\"|\"english-light\"|null, \"segmenter\": \"words\"|\"cjk-bigram\",
                           \"identifiers\": \"off\"|\"split\"|\"keep\"|\"both\"}; default: bm25s's (lowercase, \\w\\w+, en, english)
                           (merge: default keeps the inputs' tokenizer)
  --k1 <f>                 BM25 k1 (default 1.2)
  --b <f>                  BM25 b (default 0.75)
  --lengths exact|byte     how postings carry document lengths: exact varint (default, exact scores) or one
                           SmallFloat byte (about one byte less per posting, scores within a few percent)
  --bucket-terms <n>       terms per `terms` bucket (default 256): smaller = fewer bytes read per query term,
                           larger = smaller resident directory (16 B + the first term per bucket)
  --bucket-docs <n>        documents per `docs` bucket (default 128): the same trade for resolving hits
  --no-docs                write no `docs` section (ids only)
  --text                   store the documents' texts in `docs` (for excerpts and reading)
  --keys                   write a `keys` section: each document's unique string key -> its id (build: from the
                           key field; merge: written anyway when an input has one)
  --bucket-keys <n>        keys per `keys` bucket (default 128)";

const HELP_BUILD: &str = "thread seg build --out <seg.thread> --docs <docs.jsonl> [options]

Builds one segment: a BM25 index over the documents (`terms` + `postings`) and their `docs` entries, as the
THREADIX container an upsert writes to idx/{gen}/seg/{seq}.thread. Prints {path, bytes, docs, tokens, meta}.

  --out <file>             the segment to write (required)
  --docs <file.jsonl>      one JSON object per line (required): an integer `id` in [0, 2^32 - 1), a `text`
                           (string; absent = an empty document) and an optional `meta` (a string, or any JSON
                           value stored as its text); blank lines are skipped
  --id-field <name>        the id field (default `id`)
  --text-field <name>      the text field (default `text`)
  --meta-field <name>      the metadata field (default `meta`)
  --key-field <name>       the key field (default `key`; a string, stored with --keys)
  --chunk-words <n>        split every text into flat windows of n words (default 200 when only --chunk-overlap is
                           given) and index each chunk as a document of its own, as the SDKs' `upsert({chunk})`
                           does: chunk i of the document keyed k is keyed `k#i` and carries the document's meta
                           and attrs; ids run from --first-id in file order (the id field is not used). Prints
                           `documents` (input rows) next to `docs` (chunks written). See `thread chunk --help`.
  --chunk-overlap <n>      words shared with the previous chunk (default 0; < --chunk-words)
  --first-id <n>           the first chunk id (default 0)
";

const HELP_INSPECT: &str = "thread seg inspect <seg.thread>

Prints {path, bytes, sections: [{name, offset, len}], meta, terms: {count, num_docs, total_len, num_postings,
buckets, bucket_terms, directory_bytes, lengths, k1, b}, docs: {n, text, buckets, bucket_docs, directory_bytes,
text_bytes}} (terms/docs only when the segment has them).
";

const HELP_SEARCH: &str = "thread seg search --text <query> [--k 10] [--with-docs] [--tombstones a.u32,-,c.u32] <seg.thread> [more.thread ...]

BM25 over one or more segments as one index: the query is tokenized with the segments' tokenizer, every term's
document frequency and the document counts are summed over the segments (corpus-wide scoring), each segment is
scored and the lists are merged. Give the segments oldest first. Prints {hits: [{id, score, doc?}], tokens, df,
num_docs, avg_len, segments, elapsed_ms}.

  --text <query>           the query (required)
  --k <n>                  results (default 10)
  --with-docs              resolve each hit through the `docs` sections (newest segment first): {meta, text|null}
  --tombstones <list>      comma-separated u32 LE id files, one per segment in order (`-` = none): ids hidden in
                           that segment (an upsert's or delete's tombstone set)
";

const HELP_MERGE: &str = "thread seg merge --out <base.thread> [--tombstones a.u32,-,c.u32] [options] <seg.thread> [more.thread ...]

Compacts segments (oldest first) into one: posting lists are concatenated per term with tombstoned documents
dropped, the newest copy of each document id wins in `docs`, the result equals a fresh build over the live
documents. Prints {path, bytes, merged, meta}. Texts are kept when any input stores them unless --no-docs.

  --out <file>             the merged segment to write (required)
  --tombstones <list>      as in `seg search`
";

const HELP_KEYS: &str = "thread seg keys [--get k1,k2,..] [--prefix p] [--after k] [--limit 100] [--tombstones a.u32,-] <seg.thread> [more.thread ...]

Point lookups and prefix listings over the `keys` sections of one or more segments (oldest first; the newest
copy of a key wins, --tombstones hide ids per segment as in `seg search`). Prints {ids: {key: id|null}} for
--get and {entries: [{key, id}], more} for --prefix.
";

const HELP_TOKENIZE: &str = "thread tokenize [--tokenizer <json|file>] <text> [<text> ...]

Prints {tokenizer, tokens: [[...], ...]}: the tokens of each text under the config (default: bm25s's).
See `thread seg build --help` for the tokenizer keys.
";

fn help_for(cmd: &str, sub: Option<&str>) -> Option<String> {
    Some(match (cmd, sub) {
        ("tokenize", _) => HELP_TOKENIZE.to_string(),
        ("chunk", _) => crate::chunk::HELP.to_string(),
        ("seg", Some("build")) => format!("{HELP_BUILD}\n{BUILD_OPTIONS}\n"),
        ("seg", Some("inspect")) => HELP_INSPECT.to_string(),
        ("seg", Some("search")) => HELP_SEARCH.to_string(),
        ("seg", Some("merge")) => format!("{HELP_MERGE}\n{BUILD_OPTIONS}\n"),
        ("seg", Some("keys")) => HELP_KEYS.to_string(),
        ("seg", _) => format!("thread seg <build|inspect|search|merge|keys> ...\n\n{USAGE}\n"),
        _ => return None,
    })
}

/// What `run` produced: JSON to print, or a help text.
pub enum Output {
    Json(Value),
    Help(String),
}

/// Runs `thread seg <cmd> ...` / `thread tokenize ...`; `--help` / `-h` / `help` anywhere prints the
/// command's help instead.
pub fn run(cmd: &str, rest: &[String]) -> Result<Output, String> {
    let wants_help = rest
        .iter()
        .any(|a| a == "--help" || a == "-h" || a == "help");
    let sub = if cmd == "seg" {
        rest.first()
            .map(String::as_str)
            .filter(|s| !s.starts_with('-') && *s != "help")
    } else {
        None
    };
    if wants_help {
        return help_for(cmd, sub)
            .map(Output::Help)
            .ok_or_else(|| format!("unknown command {cmd:?}"));
    }
    run_json(cmd, rest).map(Output::Json)
}

fn run_json(cmd: &str, rest: &[String]) -> Result<Value, String> {
    match cmd {
        "tokenize" => tokenize(&Args::parse(rest)?),
        "chunk" => crate::chunk::run(&Args::parse(rest)?),
        "seg" => {
            let sub = rest.first().map(String::as_str).ok_or(
                "seg needs a subcommand: build | search | inspect | merge | keys (try `thread seg --help`)",
            )?;
            let a = Args::parse(&rest[1..])?;
            match sub {
                "build" => build(&a),
                "search" => search(&a),
                "inspect" => inspect(&a),
                "merge" => merge(&a),
                "keys" => keys(&a),
                other => err(format!(
                    "unknown seg subcommand {other:?} (try `thread seg --help`)"
                )),
            }
        }
        other => err(format!("unknown command {other:?}")),
    }
}
