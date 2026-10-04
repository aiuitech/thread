//! Rank fusion around a judge: the arithmetic of Wiper's `jev_excerpts/v1` (= `jev_chunks/v2`),
//! with the judge itself left to the host. The host picks the first `docs` documents of the fused
//! list (the *window*), cuts passages from them (`excerpts`), asks any model to score the
//! passages (Jev, a cross-encoder, an LLM: a network call that does not belong in this crate),
//! and hands the scores back; this module turns them into the final document order:
//!
//! ```text
//! score(d) = Σ_s w_s / (k + rank_s(d))     s ∈ {jev, fused, lexical, dense}, where d has a rank
//! ```
//!
//! * `jev`: competition rank of the document's best passage rank among all judged passages
//!   (passages ranked by score descending, ties by passage index);
//! * `fused`: the document's position in the window (the pipeline's own fusion);
//! * `lexical` / `dense`: competition rank of the document's best document-level rank over the
//!   lists of that index family ([`best_doc_ranks`]);
//! * documents with no value for a signal get nothing from it; ties by window position.
//!
//! With no judge scores (an empty or failed judgement) the host keeps the fused order, as the
//! Python component does.

use std::collections::HashMap;
use std::hash::Hash;

use crate::excerpts::py_sum;

/// Fusion weights (`jev_excerpts/v1` defaults; the lead strategy uses `jev = 6`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weights {
    pub k: f64,
    pub jev: f64,
    pub fused: f64,
    pub lexical: f64,
    pub dense: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Weights {
            k: 10.0,
            jev: 3.0,
            fused: 1.0,
            lexical: 0.5,
            dense: 0.5,
        }
    }
}

/// The signals a window document was fused from, plus its score.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Signals {
    pub jev: Option<u32>,
    pub fused: u32,
    pub lexical: Option<u32>,
    pub dense: Option<u32>,
    pub score: f64,
}

/// The fused window.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Fused {
    /// Window indices, best first.
    pub order: Vec<usize>,
    /// Per window index.
    pub signals: Vec<Signals>,
    /// Per passage: its rank among the judged passages (1 = best).
    pub passage_rank: Vec<u32>,
}

/// Ranks of passages by score descending, ties by index (1-based).
pub fn passage_ranks(scores: &[f64]) -> Vec<u32> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut out = vec![0u32; scores.len()];
    for (r, i) in idx.into_iter().enumerate() {
        out[i] = r as u32 + 1;
    }
    out
}

/// Competition ranks (1, 2, 2, 4) of the entries that have a value, higher values first; equal
/// values share the better rank; ties keep the given order.
pub fn competition_ranks(values: &[Option<f64>]) -> Vec<Option<u32>> {
    let mut have: Vec<usize> = (0..values.len()).filter(|&i| values[i].is_some()).collect();
    have.sort_by(|&a, &b| {
        values[b]
            .unwrap()
            .partial_cmp(&values[a].unwrap())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = vec![None; values.len()];
    let mut prev: Option<usize> = None;
    for (n, &i) in have.iter().enumerate() {
        let rank = match prev {
            Some(p) if values[p] == values[i] => out[p],
            _ => Some(n as u32 + 1),
        };
        out[i] = rank;
        prev = Some(i);
    }
    out
}

/// A document's best document-level rank over several lists: in each list, distinct documents
/// are numbered in order of first appearance; the minimum over the lists is kept.
pub fn best_doc_ranks<K: Hash + Eq + Clone>(
    lists: &[&[u32]],
    mut doc_of: impl FnMut(u32) -> K,
) -> HashMap<K, u32> {
    let mut best: HashMap<K, u32> = HashMap::new();
    for list in lists {
        let mut seen: HashMap<K, u32> = HashMap::new();
        for &id in *list {
            let d = doc_of(id);
            if !seen.contains_key(&d) {
                let r = seen.len() as u32 + 1;
                seen.insert(d.clone(), r);
                let e = best.entry(d).or_insert(u32::MAX);
                *e = (*e).min(r);
            }
        }
    }
    best
}

/// Fuses a window of `window_len` documents. `passage_owner[p]` is the window index of passage
/// `p`, `passage_score[p]` the judge's score for it (higher = better); `lexical[d]` / `dense[d]`
/// are the window documents' best list ranks (`None` when the family did not return them).
pub fn fuse(
    window_len: usize,
    passage_owner: &[usize],
    passage_score: &[f64],
    lexical: &[Option<u32>],
    dense: &[Option<u32>],
    w: &Weights,
) -> Fused {
    let passage_rank = passage_ranks(passage_score);
    // Best passage rank per document, as a "higher is better" value for competition ranking.
    let mut best_prank: Vec<Option<f64>> = vec![None; window_len];
    for (p, &d) in passage_owner.iter().enumerate() {
        if d < window_len {
            let r = passage_rank[p] as f64;
            let e = &mut best_prank[d];
            if e.is_none_or(|v| -r > v) {
                *e = Some(-r);
            }
        }
    }
    let jev = competition_ranks(&best_prank);
    let as_values = |ranks: &[Option<u32>]| -> Vec<Option<f64>> {
        (0..window_len)
            .map(|d| ranks.get(d).copied().flatten().map(|r| -(r as f64)))
            .collect()
    };
    let lex = competition_ranks(&as_values(lexical));
    let den = competition_ranks(&as_values(dense));
    let mut signals = Vec::with_capacity(window_len);
    for d in 0..window_len {
        let s = Signals {
            jev: jev[d],
            fused: d as u32 + 1,
            lexical: lex[d],
            dense: den[d],
            score: 0.0,
        };
        // Python's `sum(...)` over the signals the document has, in the order jev, fused,
        // lexical, dense (compensated summation, as CPython's `sum` does).
        let terms = [
            s.jev.map(|r| w.jev / (w.k + r as f64)),
            Some(w.fused / (w.k + s.fused as f64)),
            s.lexical.map(|r| w.lexical / (w.k + r as f64)),
            s.dense.map(|r| w.dense / (w.k + r as f64)),
        ];
        let score = py_sum(terms.into_iter().flatten());
        signals.push(Signals { score, ..s });
    }
    let mut order: Vec<usize> = (0..window_len).collect();
    order.sort_by(|&a, &b| {
        signals[b]
            .score
            .partial_cmp(&signals[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    Fused {
        order,
        signals,
        passage_rank,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks() {
        assert_eq!(passage_ranks(&[0.1, 0.9, 0.9, 0.0]), [3, 1, 2, 4]);
        assert_eq!(
            competition_ranks(&[Some(1.0), None, Some(3.0), Some(1.0), Some(0.5)]),
            [Some(2), None, Some(1), Some(2), Some(4)]
        );
        let best = best_doc_ranks(&[&[5, 5, 7, 9], &[9, 1]], |id| id / 2);
        assert_eq!(best[&2], 1); // 5 -> doc 2, first in list 1
        assert_eq!(best[&3], 2); // 7
        assert_eq!(best[&4], 1); // 9 is third in list 1 but first in list 2
        assert_eq!(best[&0], 2);
    }

    #[test]
    fn fuse_matches_the_python_arithmetic() {
        // Three documents; passages p0,p1 of doc 0, p2 of doc 1, p3 of doc 2.
        let owner = [0, 0, 1, 2];
        let scores = [0.02, 0.1, 0.98, 0.05];
        let lexical = [Some(1), Some(3), None];
        let dense = [Some(2), Some(1), Some(5)];
        let w = Weights {
            k: 10.0,
            jev: 6.0,
            fused: 1.0,
            lexical: 0.5,
            dense: 0.5,
        };
        let f = fuse(3, &owner, &scores, &lexical, &dense, &w);
        assert_eq!(f.passage_rank, [4, 2, 1, 3]);
        assert_eq!(f.signals[0].jev, Some(2));
        assert_eq!(f.signals[1].jev, Some(1));
        assert_eq!(f.signals[2].jev, Some(3));
        assert_eq!(f.signals[2].lexical, None);
        let s1 = 6.0 / 11.0 + 1.0 / 12.0 + 0.5 / (10.0 + 2.0) + 0.5 / 11.0;
        assert_eq!(f.signals[1].score, s1);
        assert_eq!(f.order, [1, 0, 2]);
        // No passages at all: fused order with the lexical/dense terms only.
        let g = fuse(3, &[], &[], &lexical, &dense, &w);
        assert!(g.signals.iter().all(|s| s.jev.is_none()));
        assert_eq!(g.order, [0, 1, 2]);
        // Ties fall back to window position.
        let h = fuse(2, &[], &[], &[None, None], &[None, None], &w);
        assert_eq!(h.order, [0, 1]);
        assert_eq!(h.signals[0].score, 1.0 / 11.0);
    }
}
