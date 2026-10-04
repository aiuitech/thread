//! The zero-dependency segmenters: `words` (maximal runs of word characters, the bm25s scanner,
//! with a configurable minimum length) and `cjk-bigram` (the same runs, with every maximal
//! sub-run of CJK characters turned into overlapping character bigrams, Lucene `CJKAnalyzer`
//! style).

use super::is_word_char;

/// CJK characters for `cjk-bigram` (and the CJK exemptions of the minimum length and of
/// identifier detection): Han ideographs (URO, extensions A to I, compatibility ideographs and
/// their supplement, the iteration/closing marks U+3005..U+3007), Hiragana, Katakana (incl. the
/// phonetic extensions, halfwidth forms and the Kana supplements), Hangul syllables and jamo
/// (incl. compatibility jamo, extensions A/B and halfwidth forms). Explicit ranges, no tables.
#[inline]
pub fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x1100..=0x11FF       // Hangul Jamo
        | 0x3005..=0x3007     // 々 〆 〇
        | 0x3040..=0x309F     // Hiragana
        | 0x30A0..=0x30FF     // Katakana
        | 0x3130..=0x318F     // Hangul Compatibility Jamo
        | 0x31F0..=0x31FF     // Katakana Phonetic Extensions
        | 0x3400..=0x4DBF     // CJK Extension A
        | 0x4E00..=0x9FFF     // CJK Unified Ideographs
        | 0xA960..=0xA97F     // Hangul Jamo Extended-A
        | 0xAC00..=0xD7AF     // Hangul Syllables
        | 0xD7B0..=0xD7FF     // Hangul Jamo Extended-B
        | 0xF900..=0xFAFF     // CJK Compatibility Ideographs
        | 0xFF66..=0xFF9F     // Halfwidth Katakana
        | 0xFFA0..=0xFFDC     // Halfwidth Hangul
        | 0x1AFF0..=0x1B16F   // Kana Extended-B, Kana Supplement, Kana Extended-A, Small Kana Extension
        | 0x20000..=0x2A6DF   // CJK Extension B
        | 0x2A700..=0x2EE5F   // CJK Extensions C, D, E, F, I
        | 0x2F800..=0x2FA1F   // CJK Compatibility Ideographs Supplement
        | 0x30000..=0x323AF // CJK Extensions G, H
    )
}

/// True when every char of `s` is CJK (and `s` is not empty): such tokens are exempt from the
/// minimum length under the CJK-aware segmenters, since one ideograph is a word.
#[inline]
pub fn all_cjk(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_cjk)
}

/// Maximal runs of word characters at least `min_len` chars long, in order, borrowed.
/// `min_len = 2` is bm25s's `(?u)\b\w\w+\b`.
pub fn split_words_min(text: &str, min_len: usize) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || loop {
        let start = rest.find(is_word_char)?;
        let after = &rest[start..];
        let end = after
            .find(|c: char| !is_word_char(c))
            .unwrap_or(after.len());
        let word = &after[..end];
        rest = &after[end..];
        if min_len <= 1 || word.chars().nth(min_len - 1).is_some() {
            return Some(word);
        }
    })
}

/// `cjk-bigram`: every maximal word run of `text`; inside a run, each maximal sub-run of CJK
/// characters becomes its overlapping character bigrams (a one-char sub-run becomes that
/// unigram, whatever `min_len`), and each non-CJK sub-run is one token when it has at least
/// `min_len` chars. Tokens are pushed in text order.
pub fn cjk_bigrams<'t>(text: &'t str, min_len: usize, out: &mut Vec<&'t str>) {
    for run in split_words_min(text, 1) {
        let mut sub_start = 0usize;
        let mut sub_cjk: Option<bool> = None;
        let flush = |from: usize, to: usize, cjk: bool, out: &mut Vec<&'t str>| {
            let sub = &run[from..to];
            if cjk {
                let idx: Vec<usize> = sub.char_indices().map(|(i, _)| i).collect();
                if idx.len() == 1 {
                    out.push(sub);
                } else {
                    for w in 0..idx.len() - 1 {
                        let end = idx.get(w + 2).copied().unwrap_or(sub.len());
                        out.push(&sub[idx[w]..end]);
                    }
                }
            } else if min_len <= 1 || sub.chars().nth(min_len - 1).is_some() {
                out.push(sub);
            }
        };
        for (i, c) in run.char_indices() {
            let cjk = is_cjk(c);
            match sub_cjk {
                Some(prev) if prev != cjk => {
                    flush(sub_start, i, prev, out);
                    sub_start = i;
                }
                _ => {}
            }
            sub_cjk = Some(cjk);
        }
        if let Some(prev) = sub_cjk {
            flush(sub_start, run.len(), prev, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bigrams(text: &str, min_len: usize) -> Vec<&str> {
        let mut out = Vec::new();
        cjk_bigrams(text, min_len, &mut out);
        out
    }

    #[test]
    fn min_length_scanner() {
        let words = |t, n| split_words_min(t, n).collect::<Vec<_>>();
        assert_eq!(
            words("C R x 7 ab abc", 1),
            ["C", "R", "x", "7", "ab", "abc"]
        );
        assert_eq!(words("C R x 7 ab abc", 2), ["ab", "abc"]);
        assert_eq!(words("C R x 7 ab abc", 3), ["abc"]);
        assert_eq!(words("é ñu", 2), ["ñu"]);
        assert_eq!(
            words("x1 y_2 _ __ 3 42 a1b", 2),
            super::super::split_words("x1 y_2 _ __ 3 42 a1b").collect::<Vec<_>>()
        );
    }

    #[test]
    fn cjk_ranges() {
        for c in [
            '日', '本', '語', 'の', 'テ', 'ー', 'ｶ', '한', 'ᄀ', 'ㄱ', '々', '〇', '㐀', '𠀀', '豈',
        ] {
            assert!(is_cjk(c), "{c} U+{:04X}", c as u32);
        }
        for c in ['a', 'é', '7', '_', 'Я', 'ก', '。', '、', ' ', '😀'] {
            assert!(!is_cjk(c), "{c}");
        }
    }

    #[test]
    fn bigrams_of_mixed_runs() {
        assert_eq!(
            bigrams("日本語のテキスト 中文 mixed文字abc", 2),
            [
                "日本", "本語", "語の", "のテ", "テキ", "キス", "スト", "中文", "mixed", "文字",
                "abc"
            ]
        );
        // a lone CJK char is a unigram; short non-CJK parts follow the minimum length
        assert_eq!(bigrams("中 a中b 中x", 2), ["中", "中", "中"]);
        assert_eq!(bigrams("中 a中b 中x", 1), ["中", "a", "中", "b", "中", "x"]);
        // Hangul syllables are bigrammed too (Lucene CJKBigramFilter does the same)
        assert_eq!(
            bigrams("한국어 텍스트 검색", 2),
            ["한국", "국어", "텍스", "스트", "검색"]
        );
        // punctuation breaks runs (it is not a word character)
        assert_eq!(bigrams("東京、大阪。", 2), ["東京", "大阪"]);
        assert_eq!(
            bigrams("plain english words", 2),
            ["plain", "english", "words"]
        );
        assert!(bigrams("", 2).is_empty());
    }
}
