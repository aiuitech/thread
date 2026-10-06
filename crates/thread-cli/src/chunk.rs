//! `thread chunk`: flat word-window chunking of a text before it is embedded, the same rule as the SDKs'
//! `chunkText` / `chunk_text` (checked against sdk/ts/test/fixtures/chunks.json). Chunking is a client-side
//! transform, so it lives here and not in thread-core: the index only ever sees the chunks.
//!
//! A word is a maximal run of characters that are not Unicode White_Space (`char::is_whitespace`). Chunk `i`
//! holds words `[i * (words - overlap), i * (words - overlap) + words)`; the last chunk is whatever remains and
//! no chunk starts once the previous one reached the end. A text with no words is one empty chunk. `start` /
//! `end` are code-point offsets of the chunk in the text (whitespace inside a chunk is kept, around it not).
//! Chunk `i` of the document keyed `k` is stored under the key `k#i`.

use serde_json::{json, Value};

use crate::lex_cli::Args;

pub const DEFAULT_WORDS: usize = 200;
pub const DEFAULT_OVERLAP: usize = 0;
pub const KEY_SEPARATOR: char = '#';

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk<'a> {
    pub index: usize,
    pub text: &'a str,
    /// Code-point offsets in the text.
    pub start: usize,
    pub end: usize,
    pub words: usize,
}

impl Chunk<'_> {
    pub fn to_json(&self) -> Value {
        json!({"index": self.index, "text": self.text, "start": self.start, "end": self.end, "words": self.words})
    }
}

/// Validates the parameters: `words >= 1`, `0 <= overlap < words`.
pub fn check(words: usize, overlap: usize) -> Result<(), String> {
    if words < 1 {
        return Err(format!("chunk words must be >= 1, got {words}"));
    }
    if overlap >= words {
        return Err(format!(
            "chunk overlap must be in [0, words), got {overlap} for {words} words"
        ));
    }
    Ok(())
}

/// The chunks of `text`: windows of `words` words sharing `overlap` words with the previous one.
pub fn chunk_text(text: &str, words: usize, overlap: usize) -> Result<Vec<Chunk<'_>>, String> {
    check(words, overlap)?;
    // Word spans as (byte start, byte end, code-point start, code-point end).
    let mut spans: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut open: Option<(usize, usize)> = None;
    let mut cp = 0usize;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some((bs, cs)) = open.take() {
                spans.push((bs, i, cs, cp));
            }
        } else if open.is_none() {
            open = Some((i, cp));
        }
        cp += 1;
    }
    if let Some((bs, cs)) = open {
        spans.push((bs, text.len(), cs, cp));
    }
    let n = spans.len();
    if n == 0 {
        return Ok(vec![Chunk {
            index: 0,
            text: "",
            start: 0,
            end: 0,
            words: 0,
        }]);
    }
    let stride = words - overlap;
    let mut out = Vec::new();
    let mut first = 0usize;
    loop {
        let last = (first + words).min(n);
        let (bs, _, cs, _) = spans[first];
        let (_, be, _, ce) = spans[last - 1];
        out.push(Chunk {
            index: out.len(),
            text: &text[bs..be],
            start: cs,
            end: ce,
            words: last - first,
        });
        if last >= n {
            return Ok(out);
        }
        first += stride;
    }
}

/// The key chunk `index` of the document keyed `key` is stored under.
pub fn chunk_key(key: &str, index: usize) -> String {
    format!("{key}{KEY_SEPARATOR}{index}")
}

pub const HELP: &str = "thread chunk [--words 200] [--overlap 0] [--file <path>] [<text> ...]

Prints {words, overlap, chunks: [[{index, text, start, end, words}, ...], ...]}: the flat word-window chunks of
each text (the positional texts, then --file's contents as one text), the same rule as the SDKs' `chunkText` /
`chunk_text` and `seg build --chunk-words`. A word is a maximal run of non-whitespace (Unicode White_Space)
characters; chunk i holds words [i * (words - overlap), i * (words - overlap) + words); the last chunk is what
remains, and a text with no words is one empty chunk. `start` / `end` are code-point offsets of the chunk in its
text (whitespace inside a chunk is kept, around it not).

  --words <n>              words per chunk (default 200, >= 1)
  --overlap <n>            words shared with the previous chunk (default 0, < --words)
  --file <path>            a UTF-8 file to chunk as one text
";

/// `thread chunk`: the chunks of each positional text and of `--file`.
pub fn run(a: &Args) -> Result<Value, String> {
    let words: usize = a.num("words", DEFAULT_WORDS)?;
    let overlap: usize = a.num("overlap", DEFAULT_OVERLAP)?;
    check(words, overlap)?;
    let mut texts: Vec<String> = a.positional.clone();
    if let Some(path) = a.get("file") {
        texts.push(std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?);
    }
    if texts.is_empty() {
        return Err("a text or --file <path> is required".to_string());
    }
    let chunks: Vec<Value> = texts
        .iter()
        .map(|t| {
            Ok(Value::Array(
                chunk_text(t, words, overlap)?
                    .iter()
                    .map(Chunk::to_json)
                    .collect(),
            ))
        })
        .collect::<Result<_, String>>()?;
    Ok(json!({"words": words, "overlap": overlap, "chunks": chunks}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_shared_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sdk/ts/test/fixtures/chunks.json"
        );
        let fx: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for case in fx["cases"].as_array().unwrap() {
            let words = case["words"].as_u64().map_or(DEFAULT_WORDS, |w| w as usize);
            let overlap = case["overlap"]
                .as_u64()
                .map_or(DEFAULT_OVERLAP, |o| o as usize);
            let text = case["text"].as_str().unwrap();
            let got: Vec<Value> = chunk_text(text, words, overlap)
                .unwrap()
                .iter()
                .map(Chunk::to_json)
                .collect();
            assert_eq!(Value::Array(got), case["chunks"], "{}", case["name"]);
            for (c, j) in chunk_text(text, words, overlap)
                .unwrap()
                .iter()
                .zip(case["chunks"].as_array().unwrap())
            {
                let by_cp: String = text.chars().skip(c.start).take(c.end - c.start).collect();
                assert_eq!(
                    by_cp,
                    j["text"].as_str().unwrap(),
                    "{}: code-point offsets",
                    case["name"]
                );
            }
        }
        for e in fx["errors"].as_array().unwrap() {
            let (w, o) = (e["words"].as_i64().unwrap(), e["overlap"].as_i64().unwrap());
            if w >= 0 && o >= 0 {
                assert!(chunk_text("a b c", w as usize, o as usize).is_err(), "{e}");
            }
        }
        for k in fx["keys"].as_array().unwrap() {
            assert_eq!(
                chunk_key(
                    k["key"].as_str().unwrap(),
                    k["index"].as_u64().unwrap() as usize
                ),
                k["chunk"].as_str().unwrap()
            );
        }
    }
}
