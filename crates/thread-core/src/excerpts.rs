//! Query-time excerpts: documents addressed by line, passages cut around the lines holding the
//! most query terms. A port of Wiper's `excerpts_v1` (and the term extraction of its
//! `evidence_view`), token for token and line for line, so a line number means the same thing
//! to the Python reranker, a Worker's page and a later `read(doc, "40-55")`.
//!
//! **Line grid.** A document's lines are its text's physical lines (a literal `\n` sequence
//! counts as a line break: some sources store escaped newlines), blank lines dropped, whitespace
//! collapsed, and every line longer than [`WRAP`] characters soft-wrapped (at a sentence end in
//! the second half of the window, else at the last space). Lines are numbered from 1.
//!
//! **Terms.** [`query_terms`] gives the question's content unigrams and adjacent bigrams
//! (`query`), the best vocabulary hints (`vocab`, whole phrases) and the content terms of searched
//! variants (`variant`). [`find`] scores every line by the distinct terms it contains, each
//! weighted by its source, a phrase bonus and its rarity inside the document
//! (`1 / sqrt(lines containing it)`); windows of `before`/`after` lines grow around matching
//! lines and are picked greedily by the weight of the terms they add (terms already shown count
//! [`REPEAT`]), merging windows that overlap or touch, until `budget` characters or
//! `max_windows`. Matching lines left out are reported as `more`. A document with no matching
//! line gets its opening lines.
//!
//! Pure string code, no dependencies; the numbers (`score`) are rounded to four decimals like
//! the Python original.

use crate::tokenize::STOPWORDS_EN;

/// Max characters per line after wrapping.
pub const WRAP: usize = 200;
/// Weight of a term an earlier window already shows.
pub const REPEAT: f64 = 0.25;
pub const PHRASE_BONUS: f64 = 1.5;
/// Content terms shorter than this are not query terms (as `vocab_clusters/v1`'s `min_len`).
pub const MIN_TERM_LEN: usize = 3;
/// Characters shown when no line matches, at most.
pub const OPENING_CHARS: usize = 1200;

/// Function words the stopword list keeps; a term made only of these never seeds an excerpt.
pub const FILLER: [&str; 37] = [
    "can",
    "could",
    "would",
    "should",
    "may",
    "might",
    "must",
    "will",
    "shall",
    "any",
    "some",
    "all",
    "each",
    "every",
    "also",
    "there",
    "here",
    "including",
    "include",
    "includes",
    "such",
    "other",
    "another",
    "long",
    "many",
    "much",
    "more",
    "most",
    "very",
    "just",
    "only",
    "specify",
    "specified",
    "describe",
    "described",
    "mention",
    "mentioned",
];

/// Question words bm25s keeps but that say nothing about where an answer is.
pub const QUESTION_WORDS: [&str; 13] = [
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "does", "did", "many",
    "much",
];

fn is_filler(t: &str) -> bool {
    FILLER.contains(&t)
}

/// Where a term came from; sets its weight (`query` 1.0, `vocab` 0.6, `variant` 0.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Query,
    Vocab,
    Variant,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Query => "query",
            Source::Vocab => "vocab",
            Source::Variant => "variant",
        }
    }
    pub fn from_name(s: &str) -> Option<Source> {
        match s {
            "query" => Some(Source::Query),
            "vocab" => Some(Source::Vocab),
            "variant" => Some(Source::Variant),
            _ => None,
        }
    }
    pub fn weight(self) -> f64 {
        match self {
            Source::Query => 1.0,
            Source::Vocab => 0.6,
            Source::Variant => 0.5,
        }
    }
}

/// A term to look for: its display form (`tokens` joined by spaces), its tokens, its source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Term {
    pub term: String,
    pub tokens: Vec<String>,
    pub source: Source,
}

impl Term {
    pub fn new(tokens: Vec<String>, source: Source) -> Term {
        Term {
            term: tokens.join(" "),
            tokens,
            source,
        }
    }
}

// ---- tokens ---------------------------------------------------------------------------------

/// Python's `str.isspace` for the characters that matter here: Unicode `White_Space` plus the
/// ASCII separators 0x1c..0x1f.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// The tokens of a line for matching: `[A-Za-z][A-Za-z0-9_\-]*[A-Za-z0-9]`, lowercased (the
/// corpus vocabulary's tokenizer). ASCII only, by design.
pub fn tokens(text: &str) -> Vec<String> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_alphabetic() {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_' || b[j] == b'-') {
            j += 1;
        }
        // The last character must be alphanumeric: back off over trailing `_`/`-`.
        let mut end = j;
        while end > i + 1 && !b[end - 1].is_ascii_alphanumeric() {
            end -= 1;
        }
        if end >= i + 2 {
            out.push(text[i..end].to_ascii_lowercase());
            i = end;
        } else {
            i += 1;
        }
    }
    out
}

/// Unigrams and adjacent bigrams of the kept tokens (lowercase, no stopwords, >= 3 chars,
/// question words dropped); a dropped token breaks the bigram chain.
pub fn content_terms(text: &str) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    let mut prev: Option<String> = None;
    for t in tokens(text) {
        if t.len() >= MIN_TERM_LEN
            && !STOPWORDS_EN.contains(&t.as_str())
            && !QUESTION_WORDS.contains(&t.as_str())
        {
            out.push(vec![t.clone()]);
            if let Some(p) = &prev {
                if *p != t {
                    out.push(vec![p.clone(), t.clone()]);
                }
            }
            prev = Some(t);
        } else {
            prev = None;
        }
    }
    out
}

/// The terms to look for: the question's content unigrams and bigrams (`query`), the first
/// `hint_terms` vocabulary hints as whole phrases (`vocab`, stopwords kept), then the content
/// terms of the searched variants (`variant`). The first source wins on duplicates.
pub fn query_terms(text: &str, hints: &[&str], variants: &[&str], hint_terms: usize) -> Vec<Term> {
    let mut out: Vec<Term> = Vec::new();
    let mut seen: Vec<Vec<String>> = Vec::new();
    let mut add = |toks: Vec<String>, source: Source| {
        if !toks.is_empty() && !seen.contains(&toks) {
            seen.push(toks.clone());
            out.push(Term::new(toks, source));
        }
    };
    for t in content_terms(text) {
        add(t, Source::Query);
    }
    for h in hints.iter().take(hint_terms) {
        add(tokens(h), Source::Vocab);
    }
    for v in variants {
        for t in content_terms(v) {
            add(t, Source::Variant);
        }
    }
    out
}

// ---- lines ----------------------------------------------------------------------------------

/// Python's `str.splitlines` boundaries.
fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | '\x0b'
            | '\x0c'
            | '\x1c'
            | '\x1d'
            | '\x1e'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

fn split_physical(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = text.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if is_line_break(c) {
            out.push(&text[start..i]);
            let mut next = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(j, '\n')) = it.peek() {
                    it.next();
                    next = j + 1;
                }
            }
            start = next;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

fn collapse(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut first = true;
    for w in raw.split(is_space).filter(|w| !w.is_empty()) {
        if !first {
            out.push(' ');
        }
        first = false;
        out.push_str(w);
    }
    out
}

/// The line grid of a text (see the module docs). `wrap` is [`WRAP`] by default.
pub fn split_lines(text: &str, wrap: usize) -> Vec<String> {
    let wrap = wrap.max(1);
    let unescaped = text.replace("\\n", "\n");
    let mut out = Vec::new();
    for raw in split_physical(&unescaped) {
        // After collapsing, the only whitespace left is single ASCII spaces.
        let mut line: Vec<char> = collapse(raw).chars().collect();
        while line.len() > wrap {
            let head = &line[..wrap];
            // The last sentence end followed by a space in the second half of the window.
            let mut cut: isize = -1;
            for i in 0..head.len() - 1 {
                if matches!(head[i], '.' | '!' | '?' | ';' | ':') && head[i + 1] == ' ' {
                    let end = i + 1;
                    if end > wrap / 2 && end as isize > cut {
                        cut = end as isize;
                    }
                }
            }
            if cut < 0 {
                cut = head
                    .iter()
                    .rposition(|&c| c == ' ')
                    .map(|p| p as isize)
                    .unwrap_or(-1);
            }
            if cut <= 0 {
                cut = wrap as isize;
            }
            let cut = cut as usize;
            let piece: String = line[..cut].iter().collect();
            out.push(piece.trim_end_matches(' ').to_string());
            let mut rest = cut;
            while rest < line.len() && line[rest] == ' ' {
                rest += 1;
            }
            line.drain(..rest);
        }
        if !line.is_empty() {
            out.push(line.into_iter().collect());
        }
    }
    out
}

/// A document on its line grid: lines numbered from 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocLines {
    pub lines: Vec<String>,
    toks: Vec<Vec<String>>,
}

impl DocLines {
    /// From a text, with the default wrap.
    pub fn new(text: &str) -> DocLines {
        DocLines::from_lines(split_lines(text, WRAP))
    }

    pub fn from_lines(lines: Vec<String>) -> DocLines {
        let toks = lines.iter().map(|l| tokens(l)).collect();
        DocLines { lines, toks }
    }

    pub fn n(&self) -> usize {
        self.lines.len()
    }

    pub fn tokens(&self) -> &[Vec<String>] {
        &self.toks
    }

    /// Characters of lines `a..=b` (1-based), one more per line for the join.
    pub fn chars(&self, a: usize, b: usize) -> usize {
        let a = a.max(1);
        let b = b.min(self.lines.len());
        if a > b {
            return 0;
        }
        self.lines[a - 1..b]
            .iter()
            .map(|l| l.chars().count() + 1)
            .sum()
    }

    /// Lines `a..=b` joined with spaces.
    pub fn plain(&self, a: usize, b: usize) -> String {
        let a = a.max(1);
        let b = b.min(self.lines.len());
        if a > b {
            return String::new();
        }
        self.lines[a - 1..b].join(" ")
    }

    /// Lines `a..=b`, one per row, each prefixed with its right-aligned number.
    pub fn numbered(&self, a: usize, b: usize, indent: &str) -> String {
        let a = a.max(1);
        let b = b.min(self.lines.len());
        if a > b {
            return String::new();
        }
        let w = b.to_string().len();
        (a..=b)
            .map(|i| format!("{indent}{i:>w$}  {}", self.lines[i - 1]))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ---- ranges ---------------------------------------------------------------------------------

/// Sorted ranges with overlapping or nearby ones (at most `gap` lines apart) joined.
pub fn merge(ranges: &[(usize, usize)], gap: usize) -> Vec<(usize, usize)> {
    let mut sorted = ranges.to_vec();
    sorted.sort();
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (a, b) in sorted {
        if let Some(last) = out.last_mut() {
            if a <= last.1 + gap {
                last.1 = last.1.max(b);
                continue;
            }
        }
        out.push((a, b));
    }
    out
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `"12-30, 45, L50-L60"` as merged ranges clamped to `1..=n`. Tolerates `L12-L30`, en/em
/// dashes, `to`, `..`, the words `line`/`lines` and spaces; two bare numbers in a row are a
/// range. Empty for `n == 0`.
pub fn parse_ranges(spec: &str, n: usize) -> Vec<(usize, usize)> {
    if n == 0 {
        return Vec::new();
    }
    // `\blines?\b` and `\bl(?=\d)` (case-insensitive) become spaces.
    let chars: Vec<char> = spec.chars().collect();
    let mut cleaned: Vec<char> = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let at_boundary = i == 0 || !is_word_char(chars[i - 1]);
        let lower = chars[i].to_ascii_lowercase();
        if at_boundary && lower == 'l' {
            let rest: String = chars[i..]
                .iter()
                .take(5)
                .map(|c| c.to_ascii_lowercase())
                .collect();
            let word_end = |len: usize| i + len >= chars.len() || !is_word_char(chars[i + len]);
            if rest.starts_with("lines") && word_end(5) {
                cleaned.push(' ');
                i += 5;
                continue;
            }
            if rest.starts_with("line") && word_end(4) {
                cleaned.push(' ');
                i += 4;
                continue;
            }
            if i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
                cleaned.push(' ');
                i += 1;
                continue;
            }
        }
        cleaned.push(chars[i]);
        i += 1;
    }
    let c = cleaned;
    let digits = |from: usize| -> Option<(usize, usize)> {
        let mut j = from;
        while j < c.len() && c[j].is_ascii_digit() {
            j += 1;
        }
        if j == from {
            return None;
        }
        let v: usize = c[from..j]
            .iter()
            .collect::<String>()
            .parse()
            .unwrap_or(usize::MAX);
        Some((v, j))
    };
    let skip_ws = |mut j: usize| -> usize {
        while j < c.len() && is_space(c[j]) {
            j += 1;
        }
        j
    };
    let mut found = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let Some((a, mut j)) = digits(i) else {
            i += 1;
            continue;
        };
        j = skip_ws(j);
        // Optional separator.
        if j < c.len() && matches!(c[j], '-' | '–' | '—') {
            j += 1;
        } else if j + 1 < c.len()
            && ((c[j].eq_ignore_ascii_case(&'t') && c[j + 1].eq_ignore_ascii_case(&'o'))
                || (c[j] == '.' && c[j + 1] == '.'))
        {
            j += 2;
        }
        j = skip_ws(j);
        // Optional `l`, `line`, `lines` then spaces.
        if j < c.len() && c[j].eq_ignore_ascii_case(&'l') {
            let mut k = j + 1;
            let rest: String = c[k..]
                .iter()
                .take(4)
                .map(|x| x.to_ascii_lowercase())
                .collect();
            if rest.starts_with("ines") {
                k += 4;
            } else if rest.starts_with("ine") {
                k += 3;
            }
            j = skip_ws(k);
        }
        let b = match digits(j) {
            Some((b, k)) => {
                j = k;
                b
            }
            None => a,
        };
        let (mut lo, mut hi) = (a.min(b), a.max(b));
        lo = lo.max(1);
        hi = hi.min(n);
        if lo <= hi {
            found.push((lo, hi));
        }
        i = j.max(i + 1);
    }
    merge(&found, 1)
}

/// `"12-30, 45"`.
pub fn fmt_ranges(ranges: &[(usize, usize)]) -> String {
    ranges
        .iter()
        .map(|&(a, b)| {
            if a == b {
                a.to_string()
            } else {
                format!("{a}-{b}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

// ---- excerpts -------------------------------------------------------------------------------

/// One window of lines `a..=b` (1-based, inclusive).
#[derive(Debug, Clone, PartialEq)]
pub struct Excerpt {
    pub a: usize,
    pub b: usize,
    /// Sum of the weights of the distinct terms in the window, rounded to four decimals.
    pub score: f64,
    /// The terms in the window, heaviest first (ties by term).
    pub terms: Vec<String>,
}

/// The excerpts of one document.
#[derive(Debug, Clone, PartialEq)]
pub struct Excerpts {
    /// In pick order (best first).
    pub windows: Vec<Excerpt>,
    /// Matching lines not shown, merged, best first.
    pub more: Vec<Excerpt>,
    /// False: no line matched; `windows` holds the opening lines.
    pub matched: bool,
}

/// Python's `sum()` over floats (CPython >= 3.12): Neumaier's compensated summation, so the
/// result does not depend on the order of nearly equal terms the way a plain left-to-right sum
/// does. Used wherever the Python original calls `sum(...)` on weights or scores.
pub fn py_sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut hi = 0.0f64;
    let mut c = 0.0f64;
    for x in values {
        let t = hi + x;
        if hi.abs() >= x.abs() {
            c += (hi - t) + x;
        } else {
            c += (x - t) + hi;
        }
        hi = t;
    }
    if c != 0.0 && c.is_finite() {
        hi += c;
    }
    hi
}

/// Python's `round(x, 4)`: the float nearest to the correctly rounded 4-decimal value.
fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap_or(x)
}

/// Per line: the `(term index, weight)` pairs of the query terms it contains, in term order.
fn line_terms(dl: &DocLines, terms: &[Term]) -> Vec<Vec<(usize, f64)>> {
    let toks = dl.tokens();
    let mut hits: Vec<(usize, Vec<usize>, f64)> = Vec::new(); // (term idx, lines, weight)
    let mut seen_terms: Vec<&str> = Vec::new();
    for (ti, t) in terms.iter().enumerate() {
        let tt = &t.tokens;
        if tt.is_empty() || tt.iter().all(|x| is_filler(x)) {
            continue;
        }
        let mut lines = Vec::new();
        for (i, tl) in toks.iter().enumerate() {
            if !tl.contains(&tt[0]) {
                continue;
            }
            if tt.len() == 1 || tl.windows(tt.len()).any(|w| w == tt.as_slice()) {
                lines.push(i);
            }
        }
        if !lines.is_empty() && !seen_terms.contains(&t.term.as_str()) {
            seen_terms.push(&t.term);
            let phrase = if tt.len() > 1 { PHRASE_BONUS } else { 1.0 };
            let w = t.source.weight() * phrase / (lines.len() as f64).sqrt();
            hits.push((ti, lines, w));
        }
    }
    let mut out: Vec<Vec<(usize, f64)>> = vec![Vec::new(); dl.n()];
    for (ti, lines, w) in hits {
        for i in lines {
            out[i].push((ti, w));
        }
    }
    out
}

/// Distinct terms of lines `a..=b` in first-appearance order (a dict update in the original).
fn terms_in(per_line: &[Vec<(usize, f64)>], a: usize, b: usize) -> Vec<(usize, f64)> {
    let mut out: Vec<(usize, f64)> = Vec::new();
    for line in &per_line[a - 1..b] {
        for &(ti, w) in line {
            if !out.iter().any(|(t, _)| *t == ti) {
                out.push((ti, w));
            }
        }
    }
    out
}

/// The excerpts of `dl` for `terms`: see the module docs. `budget` is in characters,
/// `more_cap` caps the `more` list.
pub fn find(
    dl: &DocLines,
    terms: &[Term],
    budget: usize,
    max_windows: usize,
    before: usize,
    after: usize,
    more_cap: usize,
) -> Excerpts {
    find_traced(
        dl,
        terms,
        budget,
        max_windows,
        before,
        after,
        more_cap,
        &mut |_, _, _, _| {},
    )
}

/// `find` reporting every candidate it pops, `(a, b, gain, accepted)`, to `on_pick` (parity
/// debugging).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn find_traced(
    dl: &DocLines,
    terms: &[Term],
    budget: usize,
    max_windows: usize,
    before: usize,
    after: usize,
    more_cap: usize,
    on_pick: &mut dyn FnMut(usize, usize, f64, bool),
) -> Excerpts {
    let n = dl.n();
    if n == 0 {
        return Excerpts {
            windows: Vec::new(),
            more: Vec::new(),
            matched: false,
        };
    }
    let per_line = line_terms(dl, terms);
    let seeds: Vec<usize> = per_line
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.is_empty())
        .map(|(i, _)| i + 1)
        .collect();
    if seeds.is_empty() {
        let cap = budget.min(OPENING_CHARS);
        let mut b = 1;
        while b < n && dl.chars(1, b + 1) <= cap {
            b += 1;
        }
        return Excerpts {
            windows: vec![Excerpt {
                a: 1,
                b,
                score: 0.0,
                terms: Vec::new(),
            }],
            more: Vec::new(),
            matched: false,
        };
    }
    let excerpt = |a: usize, b: usize| -> Excerpt {
        let t = terms_in(&per_line, a, b);
        let sum = py_sum(t.iter().map(|&(_, w)| w));
        let mut names: Vec<(f64, &str)> = t
            .iter()
            .map(|&(ti, w)| (w, terms[ti].term.as_str()))
            .collect();
        names.sort_by(|x, y| {
            y.0.partial_cmp(&x.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(x.1.cmp(y.1))
        });
        Excerpt {
            a,
            b,
            score: round4(sum),
            terms: names.into_iter().map(|(_, s)| s.to_string()).collect(),
        }
    };
    let mut pool: Vec<(usize, usize)> = seeds
        .iter()
        .map(|&s| (s.saturating_sub(before).max(1), (s + after).min(n)))
        .collect();
    let mut picks: Vec<(usize, usize)> = Vec::new();
    let mut picked: Vec<(usize, usize)> = Vec::new();
    let mut shown: Vec<usize> = Vec::new();
    while !pool.is_empty() {
        // The candidate adding the most term weight; ties by the lower start (first wins).
        let mut best_j = 0;
        let mut best_gain = f64::NEG_INFINITY;
        for (j, &(a, b)) in pool.iter().enumerate() {
            let gain = py_sum(
                terms_in(&per_line, a, b)
                    .into_iter()
                    .map(|(ti, w)| w * if shown.contains(&ti) { REPEAT } else { 1.0 }),
            );
            if gain > best_gain || (gain == best_gain && a < pool[best_j].0) {
                best_gain = gain;
                best_j = j;
            }
        }
        let best = pool.remove(best_j);
        let mut with = picked.clone();
        with.push(best);
        let union = merge(&with, 1);
        if !picked.is_empty()
            && (union.len() > max_windows
                || union.iter().map(|&(a, b)| dl.chars(a, b)).sum::<usize>() > budget)
        {
            on_pick(best.0, best.1, best_gain, false);
            continue;
        }
        picks.push(best);
        picked = union;
        on_pick(best.0, best.1, best_gain, true);
        for (ti, _) in terms_in(&per_line, best.0, best.1) {
            if !shown.contains(&ti) {
                shown.push(ti);
            }
        }
    }
    // Union windows in the order of the first pick each one contains.
    let mut ordered: Vec<(usize, (usize, usize))> = picked
        .iter()
        .map(|&w| {
            let first = picks
                .iter()
                .position(|&(a, b)| w.0 <= a && b <= w.1)
                .unwrap_or(usize::MAX);
            (first, w)
        })
        .collect();
    ordered.sort_by_key(|&(first, w)| (first, w));
    let windows: Vec<Excerpt> = ordered.iter().map(|&(_, (a, b))| excerpt(a, b)).collect();
    let covered = |line: usize| picked.iter().any(|&(a, b)| a <= line && line <= b);
    let left: Vec<(usize, usize)> = seeds
        .iter()
        .filter(|&&s| !covered(s))
        .map(|&s| (s, s))
        .collect();
    let mut more: Vec<Excerpt> = merge(&left, 1)
        .into_iter()
        .map(|(a, b)| excerpt(a, b))
        .collect();
    more.sort_by(|x, y| {
        y.score
            .partial_cmp(&x.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.a.cmp(&y.a))
    });
    more.truncate(more_cap);
    Excerpts {
        windows,
        more,
        matched: true,
    }
}

/// Whitespace collapsed and cut to `n` characters with an ellipsis (`jev_docs/v1`'s `_cut`).
pub fn cut(text: &str, n: usize) -> String {
    let t = collapse(text);
    let count = t.chars().count();
    if count <= n {
        return t;
    }
    let keep = n.saturating_sub(1).max(1);
    let head: String = t.chars().take(keep).collect();
    format!("{}…", head.trim_end_matches(is_space))
}

/// Whitespace-separated words of a text (`len(text.split())`).
pub fn word_count(text: &str) -> usize {
    text.split(is_space).filter(|w| !w.is_empty()).count()
}

/// A passage's label and text as the Jev reranker shows them: `"<title> · lines a-b of n"` and
/// `"<label> — <plain text cut to max_chars>"`.
pub fn passage_label(title: &str, fallback: &str, a: usize, b: usize, n: usize) -> String {
    let t = if title.is_empty() { fallback } else { title };
    format!("{} · lines {a}-{b} of {n}", cut(t, 100))
}

pub fn passage_text(label: &str, dl: &DocLines, a: usize, b: usize, max_chars: usize) -> String {
    format!("{label} — {}", cut(&dl.plain(a, b), max_chars))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_match_the_regex() {
        assert_eq!(
            tokens("Deploy-failed, re_try x a9 _x 9a A"),
            ["deploy-failed", "re_try", "a9"]
        );
        assert_eq!(tokens("x_ x- -x aB"), ["ab"]);
        assert!(tokens("é ü 42").is_empty());
        assert_eq!(tokens("end-"), ["end"]);
    }

    #[test]
    fn content_terms_and_query_terms() {
        let ct = content_terms("What does the deployment pipeline do when retries fail?");
        let flat: Vec<String> = ct.iter().map(|t| t.join(" ")).collect();
        assert_eq!(
            flat,
            [
                "deployment",
                "pipeline",
                "deployment pipeline",
                "retries",
                "fail",
                "retries fail"
            ]
        );
        let qt = query_terms(
            "deploy failed",
            &["Deploy Pipeline", "failed", "zz"],
            &["deploy failed retry"],
            2,
        );
        let names: Vec<(&str, Source)> = qt.iter().map(|t| (t.term.as_str(), t.source)).collect();
        assert_eq!(
            names,
            [
                ("deploy", Source::Query),
                ("failed", Source::Query),
                ("deploy failed", Source::Query),
                ("deploy pipeline", Source::Vocab),
                ("retry", Source::Variant),
                ("failed retry", Source::Variant),
            ]
        );
    }

    #[test]
    fn lines_wrap_and_collapse() {
        assert_eq!(
            split_lines("a\n\n  b   c \r\nd\\ne", WRAP),
            ["a", "b c", "d", "e"]
        );
        let long = "word ".repeat(100); // 499 chars
        let lines = split_lines(&long, 50);
        assert!(lines.iter().all(|l| l.chars().count() <= 50));
        assert_eq!(lines.join(" "), long.trim_end());
        // Sentence end in the second half wins over the last space.
        let s = format!("{}. {}", "x".repeat(30), "y".repeat(30));
        assert_eq!(split_lines(&s, 40)[0], format!("{}.", "x".repeat(30)));
        // No space at all: a hard cut at `wrap`.
        assert_eq!(split_lines(&"z".repeat(10), 4), ["zzzz", "zzzz", "zz"]);
        let dl = DocLines::new("one two\nthree\nfour");
        assert_eq!(dl.chars(1, 2), 8 + 6);
        assert_eq!(dl.plain(2, 3), "three four");
        assert_eq!(dl.numbered(9, 10, ""), "");
        assert_eq!(
            DocLines::new("a\nb\nc\nd\ne\nf\ng\nh\ni\nj").numbered(9, 10, "> "),
            ">  9  i\n> 10  j"
        );
    }

    #[test]
    fn ranges_parse_and_merge() {
        assert_eq!(
            parse_ranges("12-30, 45, L50-L60", 100),
            [(12, 30), (45, 45), (50, 60)]
        );
        assert_eq!(
            parse_ranges("lines 3 to 5; 6–7 .. ; 9..11", 10),
            [(3, 7), (9, 10)]
        );
        assert_eq!(parse_ranges("30-12", 20), [(12, 20)]);
        assert_eq!(parse_ranges("0-2", 20), [(1, 2)]);
        assert_eq!(parse_ranges("12 later 30", 50), [(12, 12), (30, 30)]);
        assert_eq!(parse_ranges("12 30", 50), [(12, 30)]);
        assert!(parse_ranges("none", 5).is_empty());
        assert!(parse_ranges("1-2", 0).is_empty());
        assert_eq!(fmt_ranges(&[(1, 1), (3, 9)]), "1, 3-9");
        assert_eq!(merge(&[(5, 6), (1, 2), (3, 3)], 1), [(1, 3), (5, 6)]);
    }

    #[test]
    fn find_picks_windows_and_more() {
        let text = (1..=30)
            .map(|i| match i {
                5 => "the deploy failed on tuesday".to_string(),
                12 => "retry the deploy".to_string(),
                25 => "budget review".to_string(),
                _ => format!("filler line {i}"),
            })
            .collect::<Vec<_>>()
            .join("\n");
        let dl = DocLines::new(&text);
        let terms = query_terms("deploy failed budget", &[], &[], 0);
        let ex = find(&dl, &terms, 3000, 3, 1, 2, 6);
        assert!(ex.matched);
        assert_eq!(ex.windows.len(), 3);
        assert_eq!((ex.windows[0].a, ex.windows[0].b), (4, 7));
        assert_eq!(ex.windows[0].terms, ["deploy failed", "failed", "deploy"]);
        assert_eq!((ex.windows[1].a, ex.windows[1].b), (24, 27));
        assert_eq!((ex.windows[2].a, ex.windows[2].b), (11, 14));
        assert!(ex.more.is_empty());
        // A tight budget keeps the best window and reports the rest as `more`.
        let tight = find(&dl, &terms, 120, 1, 1, 2, 6);
        assert_eq!(tight.windows.len(), 1);
        assert_eq!(tight.more.len(), 2);
        assert_eq!(tight.more[0].a, 25);
        // No match: the opening lines within min(budget, 1200) characters.
        let none = find(&dl, &query_terms("zebra", &[], &[], 0), 3000, 3, 1, 2, 6);
        assert!(!none.matched);
        assert_eq!(none.windows[0].a, 1);
        assert!(none.windows[0].b > 1);
        assert_eq!(
            find(&DocLines::new(""), &terms, 100, 1, 1, 1, 1)
                .windows
                .len(),
            0
        );
    }

    #[test]
    fn py_sum_is_compensated() {
        let w = [
            0.16666666666666666,
            0.17677669529663687,
            0.08980265101338746,
            0.4008918628686366,
        ];
        // Plain left-to-right sums of these four terms differ by an ulp between orders; CPython's
        // compensated `sum()` gives the same value for every order.
        let a = py_sum(w.iter().map(|x| x * 0.25));
        let b = py_sum([w[0], w[1], w[3], w[2]].iter().map(|x| x * 0.25));
        assert_eq!(a, b);
        assert_eq!(a, 0.2085344689613319);
        assert_eq!(py_sum([]), 0.0);
        assert_eq!(py_sum([1.0, 1e100, 1.0, -1e100]), 2.0);
    }

    #[test]
    fn cut_and_labels() {
        assert_eq!(cut("  a  b ", 10), "a b");
        assert_eq!(cut("abcdefghij", 5), "abcd…");
        assert_eq!(cut("abc d", 5), "abc d");
        assert_eq!(round4(0.123456), 0.1235);
        assert_eq!(
            passage_label("", "doc_1", 3, 5, 9),
            "doc_1 · lines 3-5 of 9"
        );
        let dl = DocLines::new("a b\nc d");
        assert_eq!(passage_text("L", &dl, 1, 2, 100), "L — a b c d");
        assert_eq!(word_count(" a  b\tc "), 3);
    }
}
