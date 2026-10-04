//! `stemmer: "english-light"`: Harman's S-stemmer (Harman 1991, "How effective is suffixing?"),
//! the three plural rules Lucene's `EnglishMinimalStemmer` implements, applied to words of at
//! least 4 characters:
//!
//! 1. `-ies` -> `-y`, except after `e` or `a` (`eies`, `aies` stay);
//! 2. `-es` -> `-e`, except `aes`, `ees`, `oes` (and `ies` words rule 1 left alone) stay;
//! 3. a final `-s` is dropped, except `us` and `ss` stay.
//!
//! The first rule whose suffix matches decides (an excepted word is left as is, it does not fall
//! through to the next rule). Zero dependencies.

use std::borrow::Cow;

/// Words shorter than this (in chars) are never touched.
pub const MIN_LEN: usize = 4;

/// Harman's S-stemmer on one (lowercased) word.
pub fn stem_s(word: &str) -> Cow<'_, str> {
    if !word.ends_with('s') || word.chars().count() < MIN_LEN {
        return Cow::Borrowed(word);
    }
    let b = word.as_bytes();
    let n = b.len();
    // `word` ends in ASCII `s` and has >= 4 chars, so n >= 4 bytes and the last byte is `s`.
    match b[n - 2] {
        b'u' | b's' => Cow::Borrowed(word),
        b'e' => match b[n - 3] {
            b'i' => {
                if matches!(b[n - 4], b'e' | b'a') {
                    Cow::Borrowed(word)
                } else {
                    Cow::Owned(format!("{}y", &word[..n - 3]))
                }
            }
            b'a' | b'e' | b'o' => Cow::Borrowed(word),
            _ => Cow::Borrowed(&word[..n - 1]),
        },
        _ => Cow::Borrowed(&word[..n - 1]),
    }
}

#[cfg(test)]
mod tests {
    use super::stem_s;

    #[test]
    fn harman_rules_and_exceptions() {
        for (w, want) in [
            // rule 1: ies -> y, except eies / aies
            ("queries", "query"),
            ("studies", "study"),
            ("series", "sery"),
            ("zombies", "zomby"),
            ("aies", "aies"),
            ("plaies", "plaies"),
            ("eies", "eies"),
            // rule 2: es -> e, except aes / ees / oes
            ("boxes", "boxe"),
            ("horses", "horse"),
            ("caches", "cache"),
            ("toes", "toes"),
            ("shoes", "shoes"),
            ("trees", "trees"),
            ("algaes", "algaes"),
            // rule 3: drop a final s, except us / ss
            ("cats", "cat"),
            ("deployments", "deployment"),
            ("news", "new"),
            ("status", "status"),
            ("corpus", "corpus"),
            ("class", "class"),
            ("process", "process"),
            // shorter than 4 chars, or no final s: untouched
            ("its", "its"),
            ("gas", "gas"),
            ("bus", "bus"),
            ("ies", "ies"),
            ("running", "running"),
            ("data", "data"),
            ("", ""),
            // non-ASCII before the suffix
            ("cafés", "café"),
            ("ñus", "ñus"),
            ("über", "über"),
        ] {
            assert_eq!(stem_s(w), want, "{w}");
        }
    }
}
