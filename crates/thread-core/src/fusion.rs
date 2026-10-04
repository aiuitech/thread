//! Reciprocal Rank Fusion, the rule `hybrid` search uses in the CLI and the wasm `Session`.
//!
//! It is exactly what Wiper's `fusion/rrf_v1` (`src/wiper/components/fusion/rrf_v1.py`)
//! computes, so a caller can check parity list for list:
//!
//! ```text
//! score(d) = Σ_lists weight(list) / (k + rank(d, list))      rank starts at 1, float64
//! ```
//!
//! accumulated in list order (then in rank order inside a list), ranked by score descending
//! and ties by ascending document id, truncated to `top_n`. Lists are the per-index result
//! lists (ids best first); their own scores do not matter, only the ranks. No dependencies.

/// One ranked list to fuse: its weight and its ids, best first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RrfList<'a> {
    pub weight: f64,
    pub ids: &'a [u32],
}

impl<'a> RrfList<'a> {
    pub fn new(weight: f64, ids: &'a [u32]) -> RrfList<'a> {
        RrfList { weight, ids }
    }
}

/// Fused `(id, score)` pairs, best first, at most `top_n`. `k` is RRF's constant (60 in
/// `rrf/v1`). An id appearing twice in one list counts twice, like the Python loop.
pub fn rrf(lists: &[RrfList<'_>], k: f64, top_n: usize) -> Vec<(u32, f64)> {
    // Insertion-ordered accumulation so that the float64 sums happen in the same order as the
    // Python dict (`scores[idx] += w / (k + rank)` in list order, then rank order).
    let mut order: Vec<u32> = Vec::new();
    let mut scores: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    for list in lists {
        for (i, &id) in list.ids.iter().enumerate() {
            let contribution = list.weight / (k + (i + 1) as f64);
            match scores.get_mut(&id) {
                Some(s) => *s += contribution,
                None => {
                    scores.insert(id, contribution);
                    order.push(id);
                }
            }
        }
    }
    let mut ranked: Vec<(u32, f64)> = order.into_iter().map(|id| (id, scores[&id])).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    ranked.truncate(top_n);
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_rrf_v1() {
        // Two lists, k = 60: doc 7 is first in both, doc 3 second in one and absent in the other,
        // doc 9 third in the first and second in the other.
        let a = [7, 3, 9];
        let b = [7, 9, 11];
        let fused = rrf(&[RrfList::new(1.0, &a), RrfList::new(1.0, &b)], 60.0, 10);
        let ids: Vec<u32> = fused.iter().map(|p| p.0).collect();
        assert_eq!(ids, [7, 9, 3, 11]);
        let s7 = 1.0 / 61.0 + 1.0 / 61.0;
        let s9 = 1.0 / 63.0 + 1.0 / 62.0;
        assert_eq!(fused[0].1, s7);
        assert_eq!(fused[1].1, s9);
        assert_eq!(fused[2].1, 1.0 / 62.0);
        assert_eq!(fused[3].1, 1.0 / 63.0);
        // Ties (same score) go by ascending id; weights scale a list; top_n truncates.
        let x = [5];
        let y = [2];
        let tie = rrf(&[RrfList::new(1.0, &x), RrfList::new(1.0, &y)], 60.0, 10);
        assert_eq!(tie.iter().map(|p| p.0).collect::<Vec<_>>(), [2, 5]);
        let weighted = rrf(&[RrfList::new(0.5, &x), RrfList::new(1.0, &y)], 60.0, 1);
        assert_eq!(weighted, vec![(2, 1.0 / 61.0)]);
        assert!(rrf(&[], 60.0, 5).is_empty());
    }
}
