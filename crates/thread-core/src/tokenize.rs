//! Tokenizer reproducing `bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"))`
//! (bm25s 0.3.11) token for token (ported from Wiper's thread engine), so a query is tokenized
//! exactly like the corpus the segment was built from. The config travels in the manifest and in
//! each segment's `meta` section.
//!
//! The algorithm bm25s runs, in order:
//!
//! 1. `text.lower()` (Python's full Unicode lowercase; Rust's `str::to_lowercase` applies the same
//!    `SpecialCasing`/`UnicodeData` mappings, including the final-sigma rule);
//! 2. `re.findall(r"(?u)\b\w\w+\b", text)`: maximal runs of word characters, keeping runs of two
//!    or more characters; a word character is Python's Unicode `\w`, i.e. `str.isalnum()` or `_`;
//! 3. tokens equal to one of the English stopwords ([`STOPWORDS_EN`], bm25s's `STOPWORDS_EN`)
//!    are dropped *before* stemming, by exact match on the lowercased token;
//! 4. each surviving token is stemmed with Snowball English (bm25s stems the unique tokens and
//!    maps every occurrence through that table, which is the same as stemming each token).
//!
//! Pure Rust, no `regex`, no `std::fs`: builds for `wasm32-unknown-unknown`.
//!
//! # Known divergences from Python's `\w`
//!
//! # Options beyond bm25s
//!
//! The `meta.tokenizer` object can select other tokenizations; every key defaults to the bm25s
//! behaviour above and is serialized only when it differs from it (so a default config's JSON,
//! and every existing index, is unchanged):
//!
//! * `token_pattern`: `\w\w+` (default), `\w+`, `\w{N,}` (N >= 1), optionally in bm25s's
//!   `(?u)\b..\b` spelling: only the minimum token length (in chars) changes;
//! * `segmenter`: `"words"` (default, the scanner above), `"cjk-bigram"` ([`segment`]),
//!   (`"icu"` and `"charabia"`, Wiper's feature-gated segmenters, are not carried over: they
//!   need large dependencies; the names are rejected with a config error);
//! * `identifiers`: `"off"` (default), `"split"`, `"keep"`, `"both"` ([`identifiers`]);
//! * `stemmer`: `"english"` (default), `"english-light"` ([`light`], Harman's S-stemmer), null;
//! * `stopwords`: `"en"` (default) or null; `lowercase`: true (default) or false.
//!
//! Order per text: technical runs are found on the original text (they need its case), each is
//! emitted by the `identifiers` mode; the text between them is lowercased, segmented, and every
//! segment goes through the minimum length, the stopword list and the stemmer, as in bm25s.
//!
//! # Known divergences from Python's `\w`
//!
//! The scanner uses `char::is_alphanumeric() || c == '_'`. Rust's `is_alphanumeric` is the
//! Unicode `Alphabetic` or `Numeric` derived property; Python's `\w` is `isalnum()`, which is
//! general category `L*` or a `Numeric_Type` (`Nd`, `Nl`, `No`, plus a few `Lo` ideographs that
//! are letters anyway). The two agree on every letter, digit and number form; they differ only on
//! characters that are `Other_Alphabetic` without being letters: combining vowel signs of Indic,
//! Arabic, Hebrew and similar scripts (category `Mn`/`Mc`, e.g. U+093E DEVANAGARI VOWEL SIGN AA)
//! and circled letters (U+24B6..U+24E9, category `So`). Rust treats those as word characters,
//! Python does not, so a Devanagari word with vowel signs is one token here and several in
//! Python. ASCII, Latin with precomposed or combining accents (U+0300..U+036F are not
//! `Other_Alphabetic`), Greek, Cyrillic, CJK ideographs, digits and emoji behave identically.
//! The Unicode versions of `std` and of the running CPython may also differ for characters
//! assigned after the older of the two.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::Error;

pub mod identifiers;
pub mod light;
pub mod segment;
mod snowball;
pub use identifiers::Identifiers;
pub use light::stem_s as stem_english_light;
pub use segment::{is_cjk, split_words_min};
pub use snowball::stem_english;

/// `segmenter` when the key is absent: the `\w`-run scanner.
pub const DEFAULT_SEGMENTER: &str = "words";
/// `identifiers` when the key is absent: no identifier handling.
pub const DEFAULT_IDENTIFIERS: &str = "off";

/// bm25s's `STOPWORDS_EN` (its default and Wiper's `stopwords: "en"`), verbatim.
pub const STOPWORDS_EN: [&str; 33] = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// The token pattern `TokenizerConfig::token_pattern` holds: bm25s's `(?u)\b\w\w+\b` without the
/// boundary assertions, which the maximal-run scanner makes implicit.
pub const TOKEN_PATTERN: &str = r"\w\w+";

/// How a text becomes tokens. Serialized as the `meta.tokenizer` JSON object of a container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizerConfig {
    /// `str.lower()` the text first (bm25s `lower=True`).
    pub lowercase: bool,
    /// [`TOKEN_PATTERN`] (`\w\w+`, bm25s's `(?u)\b\w\w+\b` is an alias), `\w+` or `\w{N,}`:
    /// only the minimum token length differs (see [`token_pattern_min_len`]).
    pub token_pattern: String,
    /// `None` keeps every token; `"en"` / `"english"` drops [`STOPWORDS_EN`].
    pub stopwords: Option<String>,
    /// `None` leaves tokens unstemmed; `"english"` applies the vendored Snowball 3.1 English
    /// stemmer, `"english-light"` Harman's S-stemmer ([`light`]).
    pub stemmer: Option<String>,
    /// `"words"` ([`DEFAULT_SEGMENTER`]) or `"cjk-bigram"`.
    pub segmenter: String,
    /// `"off"` ([`DEFAULT_IDENTIFIERS`]), `"split"`, `"keep"` or `"both"` ([`Identifiers`]).
    pub identifiers: String,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        TokenizerConfig::bm25s_default()
    }
}

impl TokenizerConfig {
    /// What Wiper's `bm25/v1` and `bm25/v2` use: lowercase, `\w\w+`, English stopwords, Snowball
    /// English.
    pub fn bm25s_default() -> TokenizerConfig {
        TokenizerConfig {
            lowercase: true,
            token_pattern: TOKEN_PATTERN.to_string(),
            stopwords: Some("en".to_string()),
            stemmer: Some("english".to_string()),
            segmenter: DEFAULT_SEGMENTER.to_string(),
            identifiers: DEFAULT_IDENTIFIERS.to_string(),
        }
    }

    /// Alias of [`TokenizerConfig::bm25s_default`] (the name `DESIGN.md` first used).
    pub fn bm25s_english() -> TokenizerConfig {
        TokenizerConfig::bm25s_default()
    }

    /// The `meta.tokenizer` object, e.g.
    /// `{"lowercase":true,"token_pattern":"\\w\\w+","stopwords":"en","stemmer":"english"}`.
    /// The four bm25s keys are always written; `segmenter` and `identifiers` follow only when
    /// they differ from their defaults, so the default config's JSON never changes.
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(96);
        s.push_str("{\"lowercase\":");
        s.push_str(if self.lowercase { "true" } else { "false" });
        s.push_str(",\"token_pattern\":");
        json_string(&mut s, &self.token_pattern);
        s.push_str(",\"stopwords\":");
        match &self.stopwords {
            Some(v) => json_string(&mut s, v),
            None => s.push_str("null"),
        }
        s.push_str(",\"stemmer\":");
        match &self.stemmer {
            Some(v) => json_string(&mut s, v),
            None => s.push_str("null"),
        }
        if self.segmenter != DEFAULT_SEGMENTER {
            s.push_str(",\"segmenter\":");
            json_string(&mut s, &self.segmenter);
        }
        if self.identifiers != DEFAULT_IDENTIFIERS {
            s.push_str(",\"identifiers\":");
            json_string(&mut s, &self.identifiers);
        }
        s.push('}');
        s
    }

    /// Parses a `meta.tokenizer` object. Missing keys take the bm25s defaults; unknown keys are
    /// ignored; a value of the wrong type is an error. Does not validate the values themselves
    /// (that happens when a [`Tokenizer`] is created), so a config round-trips unchanged.
    pub fn from_json(text: &str) -> Result<TokenizerConfig, Error> {
        let mut cfg = TokenizerConfig::bm25s_default();
        let mut p = JsonParser::new(text);
        p.expect('{')?;
        p.skip_ws();
        if p.peek() == Some('}') {
            p.pos += 1;
        } else {
            loop {
                p.skip_ws();
                let key = p.string()?;
                p.skip_ws();
                p.expect(':')?;
                p.skip_ws();
                match key.as_str() {
                    "lowercase" => cfg.lowercase = p.bool()?,
                    "token_pattern" => cfg.token_pattern = p.string()?,
                    "stopwords" => cfg.stopwords = p.string_or_null()?,
                    "stemmer" => cfg.stemmer = p.string_or_null()?,
                    // null means the default, like a missing key
                    "segmenter" => {
                        cfg.segmenter = p
                            .string_or_null()?
                            .unwrap_or_else(|| DEFAULT_SEGMENTER.to_string())
                    }
                    "identifiers" => {
                        cfg.identifiers = p
                            .string_or_null()?
                            .unwrap_or_else(|| DEFAULT_IDENTIFIERS.to_string())
                    }
                    _ => p.skip_value()?,
                }
                p.skip_ws();
                match p.next() {
                    Some(',') => continue,
                    Some('}') => break,
                    _ => return Err(Error::Format("tokenizer json: expected ',' or '}'".into())),
                }
            }
        }
        p.skip_ws();
        if p.pos != text.len() {
            return Err(Error::Format(
                "tokenizer json: trailing characters after the object".into(),
            ));
        }
        Ok(cfg)
    }
}

impl TokenizerConfig {
    /// The `tokenizer` object of a whole `meta` section (`Ok(None)` when the key is absent or
    /// null). Other keys are skipped without being interpreted.
    pub fn from_meta_json(meta: &str) -> Result<Option<TokenizerConfig>, Error> {
        let mut p = JsonParser::new(meta);
        p.expect('{')?;
        p.skip_ws();
        if p.peek() == Some('}') {
            return Ok(None);
        }
        loop {
            p.skip_ws();
            let key = p.string()?;
            p.skip_ws();
            p.expect(':')?;
            p.skip_ws();
            if key == "tokenizer" {
                if p.literal("null") {
                    return Ok(None);
                }
                let start = p.pos;
                p.skip_value()?;
                return TokenizerConfig::from_json(&meta[start..p.pos]).map(Some);
            }
            p.skip_value()?;
            p.skip_ws();
            match p.next() {
                Some(',') => continue,
                Some('}') => return Ok(None),
                _ => return Err(Error::Format("meta json: expected ',' or '}'".into())),
            }
        }
    }
}

/// Writes `v` as a JSON string literal.
fn json_string(out: &mut String, v: &str) {
    out.push('"');
    for c in v.chars() {
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
}

/// A tiny JSON reader: enough for a flat object of strings, booleans and null, skipping
/// anything else (numbers, nested values) under unknown keys.
struct JsonParser<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(text: &'a str) -> Self {
        JsonParser { text, pos: 0 }
    }
    fn peek(&self) -> Option<char> {
        self.text[self.pos..].chars().next()
    }
    fn next(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }
    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\n' | '\r')) {
            self.pos += 1;
        }
    }
    fn expect(&mut self, want: char) -> Result<(), Error> {
        self.skip_ws();
        if self.next() == Some(want) {
            Ok(())
        } else {
            Err(Error::Format(format!("tokenizer json: expected {want:?}")))
        }
    }
    fn literal(&mut self, word: &str) -> bool {
        if self.text[self.pos..].starts_with(word) {
            self.pos += word.len();
            true
        } else {
            false
        }
    }
    fn bool(&mut self) -> Result<bool, Error> {
        if self.literal("true") {
            Ok(true)
        } else if self.literal("false") {
            Ok(false)
        } else {
            Err(Error::Format("tokenizer json: expected a boolean".into()))
        }
    }
    fn string_or_null(&mut self) -> Result<Option<String>, Error> {
        if self.literal("null") {
            Ok(None)
        } else {
            self.string().map(Some)
        }
    }
    fn string(&mut self) -> Result<String, Error> {
        if self.next() != Some('"') {
            return Err(Error::Format("tokenizer json: expected a string".into()));
        }
        let mut out = String::new();
        loop {
            let c = self
                .next()
                .ok_or_else(|| Error::Format("tokenizer json: unterminated string".into()))?;
            match c {
                '"' => return Ok(out),
                '\\' => {
                    let e = self
                        .next()
                        .ok_or_else(|| Error::Format("tokenizer json: bad escape".into()))?;
                    match e {
                        '"' | '\\' | '/' => out.push(e),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let mut code = self.hex4()?;
                            if (0xD800..0xDC00).contains(&code) {
                                // A surrogate pair: `😀`.
                                if !(self.literal("\\u")) {
                                    return Err(Error::Format(
                                        "tokenizer json: lone surrogate".into(),
                                    ));
                                }
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return Err(Error::Format(
                                        "tokenizer json: lone surrogate".into(),
                                    ));
                                }
                                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                            }
                            out.push(char::from_u32(code).ok_or_else(|| {
                                Error::Format("tokenizer json: bad \\u escape".into())
                            })?);
                        }
                        _ => return Err(Error::Format("tokenizer json: bad escape".into())),
                    }
                }
                c => out.push(c),
            }
        }
    }
    fn hex4(&mut self) -> Result<u32, Error> {
        let s = self
            .text
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| Error::Format("tokenizer json: bad \\u escape".into()))?;
        let v = u32::from_str_radix(s, 16)
            .map_err(|_| Error::Format("tokenizer json: bad \\u escape".into()))?;
        self.pos += 4;
        Ok(v)
    }
    /// Skips one value of any kind (for keys this crate does not know).
    fn skip_value(&mut self) -> Result<(), Error> {
        match self.peek() {
            Some('"') => self.string().map(|_| ()),
            Some('{') | Some('[') => {
                let mut depth = 0i32;
                loop {
                    match self.next() {
                        Some('"') => {
                            self.pos -= 1;
                            self.string()?;
                        }
                        Some('{') | Some('[') => depth += 1,
                        Some('}') | Some(']') => {
                            depth -= 1;
                            if depth == 0 {
                                return Ok(());
                            }
                        }
                        Some(_) => {}
                        None => {
                            return Err(Error::Format("tokenizer json: unterminated value".into()))
                        }
                    }
                }
            }
            Some(_) => {
                while let Some(c) = self.peek() {
                    if c == ',' || c == '}' || c == ']' || c.is_whitespace() {
                        break;
                    }
                    self.pos += c.len_utf8();
                }
                Ok(())
            }
            None => Err(Error::Format("tokenizer json: missing value".into())),
        }
    }
}

/// Python's `\w` for the `re` module in Unicode mode (see the module docs for the divergences).
#[inline]
pub fn is_word_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Maximal runs of word characters of length >= 2 (`(?u)\b\w\w+\b`), in order, borrowed.
pub fn split_words(text: &str) -> impl Iterator<Item = &str> {
    split_words_min(text, 2)
}

/// The minimum token length (in chars) a `token_pattern` stands for: `\w\w+` -> 2, `\w+` -> 1,
/// `\w\w\w+` -> 3, `\w{N,}` -> N (N >= 1), each optionally as `(?u)\b<pattern>\b` (bm25s's
/// spelling) or with only the `(?u)` flag. `None` for anything else.
pub fn token_pattern_min_len(pattern: &str) -> Option<usize> {
    let mut p = pattern.strip_prefix("(?u)").unwrap_or(pattern);
    if let Some(inner) = p.strip_prefix(r"\b").and_then(|q| q.strip_suffix(r"\b")) {
        p = inner;
    }
    if let Some(n) = p.strip_prefix(r"\w{").and_then(|q| q.strip_suffix(",}")) {
        if n.is_empty() || n.len() > 6 || !n.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let n: usize = n.parse().ok()?;
        return (n >= 1).then_some(n);
    }
    let mut k = 0usize;
    while let Some(rest) = p.strip_prefix(r"\w") {
        if rest == "+" {
            return Some(k + 1);
        }
        p = rest;
        k += 1;
    }
    None
}

/// The stemmers `stemmer` names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stemmer {
    None,
    /// The vendored Snowball 3.1.0 English stemmer (`snowball/`): what PyStemmer 3.1 computes,
    /// verified word for word against it.
    English,
    /// Harman's S-stemmer ([`light`]).
    EnglishLight,
}

/// The segmenters `segmenter` names (the feature-gated ones only when built in).
enum Segmenter {
    Words,
    CjkBigram,
}

/// A validated [`TokenizerConfig`] with its stemmer, reused across texts.
pub struct Tokenizer {
    lowercase: bool,
    /// Minimum token length in chars, from `token_pattern`.
    min_len: usize,
    segmenter: Segmenter,
    identifiers: Identifiers,
    stopwords: &'static [&'static str],
    stemmer: Stemmer,
    cache: HashMap<String, String>,
}

impl Tokenizer {
    /// Validates `cfg`: the pattern must be one [`token_pattern_min_len`] understands, the
    /// stopword list `en`/`english` or absent, the stemmer `"english"`, `"english-light"` or
    /// absent, the segmenter `words` or `cjk-bigram`, `identifiers` one of `off`/`split`/`keep`/`both`.
    pub fn new(cfg: &TokenizerConfig) -> Result<Tokenizer, Error> {
        let min_len = token_pattern_min_len(&cfg.token_pattern).ok_or_else(|| {
            Error::Format(format!(
                "unsupported token_pattern {:?} (one of \"\\\\w\\\\w+\", \"\\\\w+\", \"\\\\w{{N,}}\")",
                cfg.token_pattern
            ))
        })?;
        let segmenter = match cfg.segmenter.as_str() {
            "words" => Segmenter::Words,
            "cjk-bigram" => Segmenter::CjkBigram,
            "icu" | "charabia" => {
                return Err(Error::Format(format!(
                "segmenter {:?} is not available in this build (only \"words\" and \"cjk-bigram\")",
                cfg.segmenter
            )))
            }
            other => {
                return Err(Error::Format(format!(
                    "unsupported segmenter {other:?} (\"words\" or \"cjk-bigram\")"
                )))
            }
        };
        let identifiers = Identifiers::parse(&cfg.identifiers).ok_or_else(|| {
            Error::Format(format!(
                "unsupported identifiers {:?} (\"off\", \"split\", \"keep\" or \"both\")",
                cfg.identifiers
            ))
        })?;
        let stopwords: &'static [&'static str] = match cfg.stopwords.as_deref() {
            None => &[],
            Some("en") | Some("english") => &STOPWORDS_EN,
            Some(other) => {
                return Err(Error::Format(format!(
                    "unsupported stopwords {other:?} (only \"en\" or null)"
                )))
            }
        };
        let stemmer = match cfg.stemmer.as_deref() {
            None => Stemmer::None,
            Some("english") => Stemmer::English,
            Some("english-light") => Stemmer::EnglishLight,
            Some(other) => {
                return Err(Error::Format(format!(
                    "unsupported stemmer {other:?} (\"english\", \"english-light\" or null)"
                )))
            }
        };
        Ok(Tokenizer {
            lowercase: cfg.lowercase,
            min_len,
            segmenter,
            identifiers,
            stopwords,
            stemmer,
            cache: HashMap::new(),
        })
    }

    #[inline]
    fn is_stopword(&self, word: &str) -> bool {
        // 33 short strings: a linear scan beats hashing, and it is what bm25s's set does
        // semantically (exact match on the lowercased token).
        self.stopwords.contains(&word)
    }

    /// Stems `word`, memoizing stems across calls (bm25s stems each unique token once).
    #[inline]
    fn stem(&mut self, word: &str) -> String {
        match self.stemmer {
            Stemmer::None => word.to_string(),
            Stemmer::EnglishLight => light::stem_s(word).into_owned(),
            Stemmer::English => {
                if let Some(s) = self.cache.get(word) {
                    return s.clone();
                }
                let stem = stem_english(word);
                if self.cache.len() < 1 << 20 {
                    self.cache.insert(word.to_string(), stem.clone());
                }
                stem
            }
        }
    }

    /// `text` lowercased when the config says so.
    #[inline]
    fn lower<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if self.lowercase {
            Cow::Owned(text.to_lowercase())
        } else {
            Cow::Borrowed(text)
        }
    }

    /// A segment that passed the length rule: dropped if a stopword, else stemmed and pushed.
    #[inline]
    fn push_word(&mut self, word: &str, out: &mut Vec<String>) {
        if !self.is_stopword(word) {
            let stem = self.stem(word);
            out.push(stem);
        }
    }

    /// Tokens of one text, in order.
    pub fn tokenize(&mut self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        if self.identifiers == Identifiers::Off {
            self.segment_into(text, &mut out);
            return out;
        }
        let mut last = 0;
        for (a, b) in identifiers::technical_runs(text) {
            self.segment_into(&text[last..a], &mut out);
            self.technical_into(&text[a..b], &mut out);
            last = b;
        }
        self.segment_into(&text[last..], &mut out);
        out
    }

    /// Lowercase, segment, then length / stopword / stem every segment of `text`.
    fn segment_into(&mut self, text: &str, out: &mut Vec<String>) {
        if text.is_empty() {
            return;
        }
        let lowered = self.lower(text);
        let mut words: Vec<&str> = Vec::new();
        match &self.segmenter {
            Segmenter::Words => {
                // The bm25s path: no intermediate vector.
                for word in split_words_min(&lowered, self.min_len) {
                    self.push_word(word, out);
                }
                return;
            }
            Segmenter::CjkBigram => segment::cjk_bigrams(&lowered, self.min_len, &mut words),
        }
        for word in words {
            self.push_word(word, out);
        }
    }

    /// One technical run (original case) by the `identifiers` mode.
    fn technical_into(&mut self, run: &str, out: &mut Vec<String>) {
        let exact = self.lower(run).into_owned();
        let keep = matches!(self.identifiers, Identifiers::Keep | Identifiers::Both)
            && exact.chars().nth(self.min_len.saturating_sub(1)).is_some();
        if keep {
            out.push(exact.clone());
        }
        if matches!(self.identifiers, Identifiers::Split | Identifiers::Both) {
            for part in identifiers::parts(run) {
                let part = self.lower(part);
                if part.chars().nth(self.min_len.saturating_sub(1)).is_none()
                    || self.is_stopword(&part)
                {
                    continue;
                }
                let stem = self.stem(&part);
                if keep && stem == exact {
                    continue;
                }
                out.push(stem);
            }
        }
    }

    /// Tokens of many texts.
    pub fn tokenize_many(&mut self, texts: &[&str]) -> Vec<Vec<String>> {
        texts.iter().map(|t| self.tokenize(t)).collect()
    }
}

/// `bm25s.tokenize([text], ...)[0]` for `cfg`. Panics on an invalid config (validate it with
/// [`Tokenizer::new`] when it comes from the outside).
pub fn tokenize(text: &str, cfg: &TokenizerConfig) -> Vec<String> {
    Tokenizer::new(cfg)
        .expect("invalid tokenizer config")
        .tokenize(text)
}

/// `bm25s.tokenize(texts, ...)` for `cfg`, stems memoized across the texts.
pub fn tokenize_many(texts: &[&str], cfg: &TokenizerConfig) -> Vec<Vec<String>> {
    Tokenizer::new(cfg)
        .expect("invalid tokenizer config")
        .tokenize_many(texts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(text: &str) -> Vec<String> {
        tokenize(text, &TokenizerConfig::bm25s_default())
    }

    // Expected outputs below were produced by bm25s 0.3.11 + PyStemmer 3.1.0:
    // `bm25s.tokenize([s], stopwords="en", stemmer=Stemmer.Stemmer("english"), return_ids=False)`.

    #[test]
    fn plain_english() {
        assert_eq!(
            toks("The deployments failed. Retry the deployment!"),
            ["deploy", "fail", "retri", "deploy"]
        );
        assert_eq!(
            toks("galaxy galaxy galaxy budget"),
            ["galaxi", "galaxi", "galaxi", "budget"]
        );
        assert_eq!(
            toks("Running runs ran RUNNING"),
            ["run", "run", "ran", "run"]
        );
        assert_eq!(
            toks("what is the secret code in zeta42 astronomy"),
            ["what", "secret", "code", "zeta42", "astronomi"]
        );
        // Words Snowball 3.x stems differently from 2.x: PyStemmer 3.1 answers.
        assert_eq!(
            toks("added internal organization university interval biologist pasted generically"),
            [
                "add",
                "internal",
                "organiz",
                "universiti",
                "interval",
                "biolog",
                "paste",
                "generic"
            ]
        );
        assert_eq!(stem_english("skies"), "sky");
        assert_eq!(stem_english("ab"), "ab");
        assert_eq!(stem_english(""), "");
    }

    #[test]
    fn stopwords_go_before_stemming_and_only_whole_tokens() {
        // "the and of" are stopwords; "these"/"this"/"there" too; "thesis" is not.
        assert_eq!(toks("the and of"), Vec::<String>::new());
        assert_eq!(toks("these this there thesis"), ["thesi"]);
        // Stopword check happens on the raw lowercased token: "is" is dropped, "was" is dropped,
        // "wasps" is kept and stemmed.
        assert_eq!(toks("Is it? It was. wasps"), ["wasp"]);
        for w in STOPWORDS_EN {
            assert!(toks(w).is_empty(), "{w}");
        }
        assert_eq!(STOPWORDS_EN.len(), 33);
    }

    #[test]
    fn short_runs_digits_underscores_punctuation() {
        assert_eq!(toks("a b c I"), Vec::<String>::new());
        assert_eq!(
            toks("x1 y_2 _ __ 3 42 a1b"),
            ["x1", "y_2", "__", "42", "a1b"]
        );
        assert_eq!(
            toks("don't can't rock'n'roll"),
            ["don", "can", "rock", "roll"]
        );
        assert_eq!(toks("state-of-the-art e-mail"), ["state", "art", "mail"]);
        assert_eq!(
            toks("foo.bar/baz@qux.com"),
            ["foo", "bar", "baz", "qux", "com"]
        );
        assert_eq!(
            toks("snake_case_name __init__"),
            ["snake_case_nam", "__init__"]
        );
    }

    #[test]
    fn unicode() {
        assert_eq!(toks("Café cafe CAFÉ"), ["café", "cafe", "café"]);
        assert_eq!(toks("naïve façade Ångström"), ["naïv", "façad", "ångström"]);
        // Combining accents are not word characters: "cafe" + U+0301 splits.
        assert_eq!(toks("cafe\u{301}s"), ["cafe"]);
        assert_eq!(toks("ΟΔΟΣ Straße İstanbul"), ["οδος", "straße", "stanbul"]);
        assert_eq!(toks("日本語のテキスト 中文"), ["日本語のテキスト", "中文"]);
        assert_eq!(toks("emoji 😀 and ½ ² ²²"), ["emoji", "²²"]);
        assert_eq!(toks("Привет мир"), ["привет", "мир"]);
    }

    #[test]
    fn empty_and_whitespace() {
        assert_eq!(toks(""), Vec::<String>::new());
        assert_eq!(toks("   \n\t  "), Vec::<String>::new());
        assert_eq!(toks("!!! ... ---"), Vec::<String>::new());
        assert_eq!(
            tokenize_many(&["", "ab", "a"], &TokenizerConfig::bm25s_default()),
            vec![vec![], vec!["ab".to_string()], vec![]]
        );
    }

    #[test]
    fn config_variants() {
        let mut cfg = TokenizerConfig::bm25s_default();
        cfg.stemmer = None;
        assert_eq!(
            tokenize("The deployments failed", &cfg),
            ["deployments", "failed"]
        );
        cfg.stopwords = None;
        assert_eq!(
            tokenize("The deployments failed", &cfg),
            ["the", "deployments", "failed"]
        );
        cfg.lowercase = false;
        assert_eq!(
            tokenize("The deployments failed", &cfg),
            ["The", "deployments", "failed"]
        );
        // `\w+` is accepted since the short-token option exists; other regexes are not.
        cfg.token_pattern = r"[a-z]+".into();
        assert!(Tokenizer::new(&cfg).is_err());
        cfg.token_pattern = r"(?u)\b\w\w+\b".into();
        assert!(Tokenizer::new(&cfg).is_ok());
        cfg.stemmer = Some("klingon".into());
        assert!(Tokenizer::new(&cfg).is_err());
        // Only the vendored English stemmer exists: other Snowball languages are config errors.
        cfg.stemmer = Some("german".into());
        assert!(Tokenizer::new(&cfg).is_err());
        cfg.stemmer = Some("english".into());
        assert!(Tokenizer::new(&cfg).is_ok());
        cfg.stopwords = Some("de".into());
        assert!(Tokenizer::new(&cfg).is_err());
    }

    #[test]
    fn json_roundtrip() {
        let cfg = TokenizerConfig::bm25s_default();
        let json = cfg.to_json();
        assert_eq!(
            json,
            r#"{"lowercase":true,"token_pattern":"\\w\\w+","stopwords":"en","stemmer":"english"}"#
        );
        assert_eq!(TokenizerConfig::from_json(&json).unwrap(), cfg);
        let none = TokenizerConfig {
            lowercase: false,
            token_pattern: TOKEN_PATTERN.into(),
            stopwords: None,
            stemmer: None,
            ..TokenizerConfig::bm25s_default()
        };
        assert_eq!(TokenizerConfig::from_json(&none.to_json()).unwrap(), none);
        // Whitespace, unknown keys (nested, numeric, string), escapes, defaults for missing keys.
        let loose = r#" { "extra": {"a": [1, 2, {"b": "}"}]}, "n": 1.5e3, "s": "x\"y",
                         "stemmer": null, "lowercase" : true , "token_pattern": "\\w+" } "#;
        let loose = loose.replace("\\\\w+", "\\u005cw\\\\w+"); // JSON `\u005cw\\w+`
        let parsed = TokenizerConfig::from_json(&loose).unwrap();
        assert_eq!(parsed.stemmer, None);
        assert_eq!(parsed.stopwords, Some("en".into()));
        assert_eq!(parsed.token_pattern, TOKEN_PATTERN);
        assert_eq!(TokenizerConfig::from_json("{}").unwrap(), cfg);
        // A whole meta section: the `tokenizer` key is found behind other keys and skipped
        // values; absent or null means "no config".
        let meta = format!(
            r#"{{"format":"thread","version":1,"bm25":{{"num_docs":3,"x":[1,{{"y":"}}"}}]}},"tokenizer":{},"vocab":{{"terms":9}}}}"#,
            none.to_json()
        );
        assert_eq!(
            TokenizerConfig::from_meta_json(&meta).unwrap(),
            Some(none.clone())
        );
        assert_eq!(
            TokenizerConfig::from_meta_json(r#"{"format":"thread"}"#).unwrap(),
            None
        );
        assert_eq!(
            TokenizerConfig::from_meta_json(r#"{"tokenizer":null}"#).unwrap(),
            None
        );
        assert_eq!(TokenizerConfig::from_meta_json("{}").unwrap(), None);
        assert!(TokenizerConfig::from_meta_json(r#"{"tokenizer":{"stemmer":1}}"#).is_err());
        assert!(TokenizerConfig::from_meta_json("[]").is_err());
        for bad in [
            "",
            "{",
            r#"{"lowercase": "yes"}"#,
            r#"{"stemmer": 1}"#,
            "{} x",
            "[]",
        ] {
            assert!(TokenizerConfig::from_json(bad).is_err(), "{bad:?}");
        }
    }

    // ---- options beyond bm25s --------------------------------------------------------------

    /// Tokens of `text` under the config `json` (missing keys = bm25s defaults).
    fn with(json: &str, text: &str) -> Vec<String> {
        let cfg = TokenizerConfig::from_json(json).unwrap();
        Tokenizer::new(&cfg).unwrap().tokenize(text)
    }

    #[test]
    fn token_pattern_minimum_length() {
        for (p, want) in [
            (r"\w\w+", Some(2)),
            (r"(?u)\b\w\w+\b", Some(2)),
            (r"\w+", Some(1)),
            (r"(?u)\b\w+\b", Some(1)),
            (r"(?u)\w+", Some(1)),
            (r"\w\w\w+", Some(3)),
            (r"\w{1,}", Some(1)),
            (r"\w{3,}", Some(3)),
            (r"(?u)\b\w{4,}\b", Some(4)),
            (r"\w{0,}", None),
            (r"\w{+3,}", None),
            (r"\w{2,5}", None),
            (r"\w{,}", None),
            (r"\w*", None),
            (r"\w", None),
            (r"\w\w", None),
            (r"\b\w+", None),
            (r"[a-z]+", None),
            ("", None),
        ] {
            assert_eq!(token_pattern_min_len(p), want, "{p}");
        }
        // `\w+` keeps one-char tokens (stopwords still apply: "a" goes)
        assert_eq!(
            with(
                r#"{"token_pattern":"\\w+"}"#,
                "C and R are languages; x = 7 a b"
            ),
            ["c", "r", "languag", "x", "7", "b"]
        );
        assert_eq!(
            with(r#"{"token_pattern":"(?u)\\b\\w+\\b"}"#, "C R x"),
            ["c", "r", "x"]
        );
        assert_eq!(
            with(
                r#"{"token_pattern":"\\w{3,}"}"#,
                "the cat sat on a mat by me"
            ),
            ["cat", "sat", "mat"]
        );
        // `\w{2,}` is `\w\w+` under another spelling
        assert_eq!(
            with(r#"{"token_pattern":"\\w{2,}"}"#, "x1 y_2 _ __ 3 42 a1b"),
            toks("x1 y_2 _ __ 3 42 a1b")
        );
    }

    const TECH: &str =
        "Ticket INC-12345 blocks v2.3.1 on k8s.io; see getUserName, HTTPServer and snake_case_name";

    #[test]
    fn identifiers_modes() {
        let off = with(r#"{"stemmer":null}"#, TECH);
        assert_eq!(
            off,
            [
                "ticket",
                "inc",
                "12345",
                "blocks",
                "v2",
                "k8s",
                "io",
                "see",
                "getusername",
                "httpserver",
                "snake_case_name"
            ]
        );
        assert_eq!(off, with(r#"{"stemmer":null,"identifiers":"off"}"#, TECH));
        assert_eq!(
            with(r#"{"stemmer":null,"identifiers":"split"}"#, TECH),
            [
                "ticket", "inc", "12345", "blocks", "v2", "k8s", "io", "see", "get", "user",
                "name", "http", "server", "snake", "case", "name"
            ]
        );
        assert_eq!(
            with(r#"{"stemmer":null,"identifiers":"keep"}"#, TECH),
            [
                "ticket",
                "inc-12345",
                "blocks",
                "v2.3.1",
                "k8s.io",
                "see",
                "getusername",
                "httpserver",
                "snake_case_name"
            ]
        );
        assert_eq!(
            with(r#"{"stemmer":null,"identifiers":"both"}"#, TECH),
            [
                "ticket",
                "inc-12345",
                "inc",
                "12345",
                "blocks",
                "v2.3.1",
                "v2",
                "k8s.io",
                "k8s",
                "io",
                "see",
                "getusername",
                "get",
                "user",
                "name",
                "httpserver",
                "http",
                "server",
                "snake_case_name",
                "snake",
                "case",
                "name"
            ]
        );
    }

    #[test]
    fn identifiers_exact_token_is_unstemmed_and_kept_parts_are_filtered() {
        let both = r#"{"identifiers":"both"}"#;
        // the exact token: lowercased, joiners kept, not stemmed, not a stopword; the parts go
        // through stopwords, minimum length and stemming
        assert_eq!(
            with(both, "INC-12345 deployments"),
            ["inc-12345", "inc", "12345", "deploy"]
        );
        assert_eq!(
            with(both, "Deployments_v2 IT-1"),
            ["deployments_v2", "deploy", "v2", "it-1"]
        );
        assert_eq!(
            with(r#"{"identifiers":"split"}"#, "Deployments_v2 IT-1"),
            ["deploy", "v2"]
        );
        assert_eq!(
            with(r#"{"identifiers":"keep"}"#, "Deployments_v2 IT-1"),
            ["deployments_v2", "it-1"]
        );
        // a part equal to the exact token is not repeated; letters and digits stay together
        assert_eq!(with(both, "zeta42 k8s md5"), ["zeta42", "k8s", "md5"]);
        // the exact token follows the minimum length too
        assert_eq!(with(both, "7 v 8"), Vec::<String>::new());
        assert_eq!(
            with(r#"{"identifiers":"both","token_pattern":"\\w+"}"#, "7 x"),
            ["7", "x"]
        );
        // `lowercase: false` keeps the case of exact tokens and parts
        assert_eq!(
            with(
                r#"{"identifiers":"both","lowercase":false,"stemmer":null}"#,
                "getUserName"
            ),
            ["getUserName", "get", "User", "Name"]
        );
    }

    #[test]
    fn identifiers_on_the_tricky_strings() {
        let both = r#"{"identifiers":"both"}"#;
        // unchanged: no technical runs, or runs whose exact token is their only part
        for s in [
            "The deployments failed. Retry the deployment!",
            "x1 y_2 _ __ 3 42 a1b",
            "don't can't rock'n'roll",
            "state-of-the-art e-mail",
            "Café cafe CAFÉ naïve ΟΔΟΣ Straße",
            "日本語のテキスト 中文",
            "emoji 😀 and ½ ² ²²",
            "NASA APIs URLs",
        ] {
            assert_eq!(with(both, s), toks(s), "{s}");
        }
        assert_eq!(
            with(both, "foo.bar/baz@qux.com"),
            ["foo.bar/baz", "foo", "bar", "baz", "qux.com", "qux", "com"]
        );
        assert_eq!(
            with(r#"{"identifiers":"split"}"#, "foo.bar/baz@qux.com"),
            toks("foo.bar/baz@qux.com")
        );
        assert_eq!(
            with(r#"{"identifiers":"keep"}"#, "foo.bar/baz@qux.com"),
            ["foo.bar/baz", "qux.com"]
        );
        assert_eq!(
            with(both, "snake_case_name __init__"),
            // Snowball keeps the final e of the short parts (it drops it from `snake_case_name`)
            [
                "snake_case_name",
                "snake",
                "case",
                "name",
                "__init__",
                "init"
            ]
        );
        // the query side goes through the same path: INC-12345 in a question
        assert_eq!(
            with(both, "what broke in INC-12345?"),
            ["what", "broke", "inc-12345", "inc", "12345"]
        );
    }

    #[test]
    fn cjk_bigram_segmenter() {
        let cfg = r#"{"segmenter":"cjk-bigram"}"#;
        assert_eq!(
            with(cfg, "日本語のテキスト 中文 mixed文字abc"),
            [
                "日本", "本語", "語の", "のテ", "テキ", "キス", "スト", "中文", "mix", "文字",
                "abc"
            ]
        );
        assert_eq!(with(cfg, "한국어 텍스트"), ["한국", "국어", "텍스", "스트"]);
        assert_eq!(with(cfg, "中 a中b"), ["中", "中"]);
        // English is untouched
        let s = "The deployments failed. Retry the deployment!";
        assert_eq!(with(cfg, s), toks(s));
        // with identifiers: technical runs first (CJK never belongs to one), the rest bigrammed
        assert_eq!(
            with(
                r#"{"segmenter":"cjk-bigram","identifiers":"both","stemmer":null}"#,
                "버전 v2.3.1 日本語v2"
            ),
            ["버전", "v2.3.1", "v2", "日本", "本語", "v2"]
        );
    }

    #[test]
    fn english_light_stemmer() {
        assert_eq!(
            with(
                r#"{"stemmer":"english-light"}"#,
                "The queries and boxes of status classes, deployments failed"
            ),
            ["query", "boxe", "status", "classe", "deployment", "failed"]
        );
        assert_eq!(stem_english_light("studies"), "study");
    }

    #[test]
    fn options_json_roundtrip() {
        // defaults are never written
        let json = TokenizerConfig::bm25s_default().to_json();
        assert!(!json.contains("segmenter") && !json.contains("identifiers"));
        let explicit = TokenizerConfig::from_json(
            r#"{"segmenter":"words","identifiers":"off","stemmer":"english"}"#,
        )
        .unwrap();
        assert_eq!(explicit, TokenizerConfig::bm25s_default());
        assert_eq!(explicit.to_json(), json);
        let nulls = TokenizerConfig::from_json(r#"{"segmenter":null,"identifiers":null}"#).unwrap();
        assert_eq!(nulls, TokenizerConfig::bm25s_default());
        // non-default keys round-trip
        let cfg = TokenizerConfig {
            token_pattern: r"\w+".into(),
            stemmer: Some("english-light".into()),
            segmenter: "cjk-bigram".into(),
            identifiers: "both".into(),
            ..TokenizerConfig::bm25s_default()
        };
        let json = cfg.to_json();
        assert_eq!(
            json,
            r#"{"lowercase":true,"token_pattern":"\\w+","stopwords":"en","stemmer":"english-light","segmenter":"cjk-bigram","identifiers":"both"}"#
        );
        assert_eq!(TokenizerConfig::from_json(&json).unwrap(), cfg);
        let only_ids = TokenizerConfig {
            identifiers: "keep".into(),
            ..TokenizerConfig::bm25s_default()
        };
        assert!(only_ids
            .to_json()
            .ends_with(r#""stemmer":"english","identifiers":"keep"}"#));
        let meta = format!(r#"{{"format":"thread","tokenizer":{json},"vocab":{{}}}}"#);
        assert_eq!(TokenizerConfig::from_meta_json(&meta).unwrap(), Some(cfg));
        assert!(TokenizerConfig::from_json(r#"{"segmenter":1}"#).is_err());
        assert!(TokenizerConfig::from_json(r#"{"identifiers":true}"#).is_err());
    }

    #[test]
    fn option_config_errors() {
        let err = |json: &str| match Tokenizer::new(&TokenizerConfig::from_json(json).unwrap()) {
            Ok(_) => panic!("{json} should be rejected"),
            Err(e) => format!("{e:?}"),
        };
        assert!(err(r#"{"segmenter":"klingon"}"#).contains("segmenter"));
        assert!(err(r#"{"segmenter":"Words"}"#).contains("segmenter"));
        assert!(err(r#"{"identifiers":"maybe"}"#).contains("identifiers"));
        assert!(err(r#"{"stemmer":"light"}"#).contains("stemmer"));
        assert!(err(r#"{"token_pattern":"\\w{0,}"}"#).contains("token_pattern"));
        assert!(err(r#"{"token_pattern":"\\S+"}"#).contains("token_pattern"));
        for ok in [
            r#"{"segmenter":"words"}"#,
            r#"{"segmenter":"cjk-bigram"}"#,
            r#"{"identifiers":"split"}"#,
            r#"{"identifiers":"keep"}"#,
            r#"{"identifiers":"both"}"#,
            r#"{"stemmer":"english-light"}"#,
            r#"{"token_pattern":"\\w{5,}"}"#,
        ] {
            assert!(
                Tokenizer::new(&TokenizerConfig::from_json(ok).unwrap()).is_ok(),
                "{ok}"
            );
        }
        // segmenters this build does not carry are config errors naming the ones it has
        assert!(err(r#"{"segmenter":"icu"}"#).contains("not available"));
        assert!(err(r#"{"segmenter":"charabia"}"#).contains("not available"));
    }
}
