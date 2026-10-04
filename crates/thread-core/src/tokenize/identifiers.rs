//! `identifiers`: technical terms (ticket ids, versions, hostnames, code identifiers) found on the
//! original, not yet lowercased text.
//!
//! A *technical run* is a maximal run of word characters (`\w`, CJK excluded: see below) and the
//! joiners `.` `-` `/` `:` `+` `#`, where each joiner has a word character immediately on both
//! sides, that contains at least one of: a digit (`char::is_numeric`), an underscore, one of the
//! joiners `.` `/` `:`, or an internal case change. A case change is lower -> Upper (`getUser`)
//! or Upper-Upper-lower (`HTTPServer`, the boundary goes before the last capital) when at least
//! two lowercase letters follow, so plural acronyms (`APIs`, `URLs`, `IDs`) and `IPv6` are not
//! split there. `state-of-the-art`, `e-mail`, `Deployments`, `NASA`, `APIs` are not technical.
//!
//! CJK characters ([`super::segment::is_cjk`]) never belong to a technical run: they end it, and
//! they are left to the segmenter, so `日本語v2` is `日本語` (segmented) plus the run `v2`.
//!
//! The parts of a technical run are its pieces between joiners and underscores, each further cut
//! at its case changes; letters and digits are never split apart (`zeta42`, `k8s`, `md5`).

use super::is_word_char;
use super::segment::is_cjk;

/// How technical runs are tokenized (`meta.tokenizer.identifiers`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Identifiers {
    /// No identifier handling: the segmenter sees the whole text (the default, bm25s parity).
    Off,
    /// Technical runs become their parts (each through stopwords, minimum length, stemming).
    Split,
    /// Technical runs become one exact token: the run lowercased, joiners kept, unstemmed.
    Keep,
    /// The exact token of `Keep`, then the parts of `Split` (a part equal to it is not repeated).
    Both,
}

impl Identifiers {
    /// Parses the config value (`None` = the default).
    pub fn parse(v: &str) -> Option<Identifiers> {
        Some(match v {
            "off" => Identifiers::Off,
            "split" => Identifiers::Split,
            "keep" => Identifiers::Keep,
            "both" => Identifiers::Both,
            _ => return None,
        })
    }
}

#[inline]
fn is_run_char(c: char) -> bool {
    is_word_char(c) && !is_cjk(c)
}

#[inline]
fn is_joiner(c: char) -> bool {
    matches!(c, '.' | '-' | '/' | ':' | '+' | '#')
}

/// Byte spans `(start, end)` of the technical runs of `text`, in order.
pub fn technical_runs(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let byte_at = |i: usize| chars.get(i).map_or(text.len(), |&(b, _)| b);
    let mut spans = Vec::new();
    let mut i = 0;
    while i < n {
        if !is_run_char(chars[i].1) {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i + 1; // one past the last char of the run so far (always a word char)
        loop {
            if j < n && is_run_char(chars[j].1) {
                j += 1;
            } else if j + 1 < n && is_joiner(chars[j].1) && is_run_char(chars[j + 1].1) {
                j += 2;
            } else {
                break;
            }
        }
        let run = &text[byte_at(start)..byte_at(j)];
        if is_technical(run) {
            spans.push((byte_at(start), byte_at(j)));
        }
        i = j;
    }
    spans
}

/// Whether a run (word chars and inner joiners only) is technical.
fn is_technical(run: &str) -> bool {
    if run
        .chars()
        .any(|c| c.is_numeric() || matches!(c, '_' | '.' | '/' | ':'))
    {
        return true;
    }
    run.split(|c: char| is_joiner(c) || c == '_')
        .any(|piece| case_cut(piece).is_some())
}

/// Byte offset of the first case change in `piece` (start of the next part), if any.
fn case_cut(piece: &str) -> Option<usize> {
    let cs: Vec<(usize, char)> = piece.char_indices().collect();
    for k in 1..cs.len() {
        let (a, b) = (cs[k - 1].1, cs[k].1);
        if a.is_lowercase() && b.is_uppercase() {
            return Some(cs[k].0);
        }
        if a.is_uppercase()
            && b.is_uppercase()
            && cs.get(k + 1).is_some_and(|x| x.1.is_lowercase())
            && cs.get(k + 2).is_some_and(|x| x.1.is_lowercase())
        {
            return Some(cs[k].0);
        }
    }
    None
}

/// The parts of a technical run, in order, in their original case (borrowed from `run`).
pub fn parts(run: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for piece in run.split(|c: char| is_joiner(c) || c == '_') {
        let mut rest = piece;
        while !rest.is_empty() {
            match case_cut(rest) {
                Some(cut) => {
                    out.push(&rest[..cut]);
                    rest = &rest[cut..];
                }
                None => {
                    out.push(rest);
                    break;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(text: &str) -> Vec<&str> {
        technical_runs(text)
            .into_iter()
            .map(|(a, b)| &text[a..b])
            .collect()
    }

    #[test]
    fn which_runs_are_technical() {
        assert_eq!(
            runs("See INC-12345, v2.3.1 and k8s.io/foo.bar.baz on aws:iam."),
            ["INC-12345", "v2.3.1", "k8s.io/foo.bar.baz", "aws:iam"]
        );
        assert_eq!(
            runs("getUserName HTTPServer snake_case_name __init__ zeta42 md5 x86_64"),
            [
                "getUserName",
                "HTTPServer",
                "snake_case_name",
                "__init__",
                "zeta42",
                "md5",
                "x86_64"
            ]
        );
        // not technical: hyphenated words, plain words, all caps, plural acronyms, Title Case
        assert!(runs(
            "state-of-the-art e-mail Deployments NASA APIs URLs IDs The end. c++ C# rock'n'roll"
        )
        .is_empty());
        // a joiner needs a word character on both sides
        assert_eq!(runs("end. v2. -x1 a.b"), ["v2", "x1", "a.b"]);
        // CJK ends a run and never belongs to one
        assert_eq!(runs("日本語v2の版 中文"), ["v2"]);
        assert_eq!(runs("IPv6 iPhone"), ["IPv6", "iPhone"]);
        assert!(runs("").is_empty());
    }

    #[test]
    fn parts_of_runs() {
        for (run, want) in [
            ("getUserName", vec!["get", "User", "Name"]),
            ("HTTPServer", vec!["HTTP", "Server"]),
            ("XMLHttpRequest", vec!["XML", "Http", "Request"]),
            ("snake_case_name", vec!["snake", "case", "name"]),
            ("__init__", vec!["init"]),
            ("INC-12345", vec!["INC", "12345"]),
            ("v2.3.1", vec!["v2", "3", "1"]),
            ("k8s.io", vec!["k8s", "io"]),
            ("foo.bar.baz", vec!["foo", "bar", "baz"]),
            ("aws:iam", vec!["aws", "iam"]),
            ("zeta42", vec!["zeta42"]),
            ("md5", vec!["md5"]),
            ("IPv6", vec!["IPv6"]),
            ("URLsToFetch", vec!["URLs", "To", "Fetch"]),
            ("x86_64", vec!["x86", "64"]),
        ] {
            assert_eq!(parts(run), want, "{run}");
        }
    }
}
