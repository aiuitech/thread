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
//! ```
//!
//! `docs.jsonl`: one JSON object per line with an integer `id` (u32), a `text` and optionally a `meta` (a string, or
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
    positional: Vec<String>,
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
    fn get(&self, k: &str) -> Option<&str> {
        self.flags.get(k).map(String::as_str)
    }
    fn flag(&self, k: &str) -> bool {
        self.get(k).is_some()
    }
    fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> Result<T, String> {
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
    let (id_f, text_f, meta_f) = (
        a.get("id-field").unwrap_or("id"),
        a.get("text-field").unwrap_or("text"),
        a.get("meta-field").unwrap_or("meta"),
    );
    let opts = segment_options(a)?;
    let mut b = SegmentBuilder::new(&opts).map_err(|e| e.to_string())?;
    let file = std::fs::File::open(docs_path).map_err(|e| format!("{docs_path}: {e}"))?;
    let mut rows = 0usize;
    for (i, line) in BufReader::with_capacity(1 << 20, file).lines().enumerate() {
        let line = line.map_err(|e| format!("{docs_path}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value =
            serde_json::from_str(&line).map_err(|e| format!("{docs_path}:{}: {e}", i + 1))?;
        let id = v
            .get(id_f)
            .and_then(Value::as_u64)
            .filter(|&x| x < u32::MAX as u64)
            .ok_or_else(|| {
                format!(
                    "{docs_path}:{}: {id_f:?} must be an integer in [0, 2^32 - 1)",
                    i + 1
                )
            })? as u32;
        let text = v.get(text_f).map(meta_string);
        let meta = v.get(meta_f).map(meta_string);
        b.add(id, text.as_deref(), meta.as_deref())
            .map_err(|e| format!("{docs_path}:{}: {e}", i + 1))?;
        rows += 1;
    }
    let bytes = b.finish().map_err(|e| e.to_string())?;
    std::fs::write(out, &bytes).map_err(|e| format!("{out}: {e}"))?;
    let seg = Segment::open(&bytes).map_err(|e| e.to_string())?;
    Ok(
        json!({"path": out, "bytes": bytes.len(), "docs": rows, "tokens": b.tokens(), "meta": serde_json::from_str::<Value>(seg.meta()).unwrap_or(Value::Null), "elapsed_ms": t0.elapsed().as_secs_f64() * 1000.0}),
    )
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
  thread tokenize    [--tokenizer json|file] \"text\" [\"text\" ...]      the index's tokenizer
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
  --text                   store the documents' texts in `docs` (for excerpts and reading)";

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

const HELP_TOKENIZE: &str = "thread tokenize [--tokenizer <json|file>] <text> [<text> ...]

Prints {tokenizer, tokens: [[...], ...]}: the tokens of each text under the config (default: bm25s's).
See `thread seg build --help` for the tokenizer keys.
";

fn help_for(cmd: &str, sub: Option<&str>) -> Option<String> {
    Some(match (cmd, sub) {
        ("tokenize", _) => HELP_TOKENIZE.to_string(),
        ("seg", Some("build")) => format!("{HELP_BUILD}\n{BUILD_OPTIONS}\n"),
        ("seg", Some("inspect")) => HELP_INSPECT.to_string(),
        ("seg", Some("search")) => HELP_SEARCH.to_string(),
        ("seg", Some("merge")) => format!("{HELP_MERGE}\n{BUILD_OPTIONS}\n"),
        ("seg", _) => format!("thread seg <build|inspect|search|merge> ...\n\n{USAGE}\n"),
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
        "seg" => {
            let sub = rest.first().map(String::as_str).ok_or(
                "seg needs a subcommand: build | search | inspect | merge (try `thread seg --help`)",
            )?;
            let a = Args::parse(&rest[1..])?;
            match sub {
                "build" => build(&a),
                "search" => search(&a),
                "inspect" => inspect(&a),
                "merge" => merge(&a),
                other => err(format!(
                    "unknown seg subcommand {other:?} (try `thread seg --help`)"
                )),
            }
        }
        other => err(format!("unknown command {other:?}")),
    }
}
