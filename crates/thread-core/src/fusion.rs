//! Reciprocal Rank Fusion, and the rest of a unified search that is not I/O: [`plan_search`]
//! resolves a request against the manifest's `lex.defaults` (mode, list depth, RRF constant,
//! weights) and [`fuse_search`] turns the dense and lexical lists into the final hits. The SDKs
//! only fetch the two lists in between.
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

/// RRF constant when neither the request nor `manifest.lex.defaults.rrfK` sets one.
pub const DEFAULT_RRF_K: f64 = 60.0;
/// Per-list depth floor of a hybrid search when no `recallK` is set: `max(top_k, 50)`.
pub const DEFAULT_RECALL_K: u32 = 50;
/// Results when the request sets no `top_k`.
pub const DEFAULT_TOP_K: u32 = 10;

/// The fusion knobs a request or `manifest.lex.defaults` may set; `None` falls through to the
/// next source (request, then manifest, then the constants above).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FusionParams {
    pub rrf_k: Option<f64>,
    pub recall_k: Option<u32>,
    pub dense_weight: Option<f64>,
    pub lexical_weight: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMode {
    Dense,
    Lexical,
    Hybrid,
}

impl SearchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SearchMode::Dense => "dense",
            SearchMode::Lexical => "lexical",
            SearchMode::Hybrid => "hybrid",
        }
    }
}

/// What the request asks for and what the generation has.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SearchInput {
    /// `None`: [`DEFAULT_TOP_K`]. A float so a host can pass its number through and have it
    /// checked here (an integer >= 1).
    pub top_k: Option<f64>,
    pub has_text: bool,
    pub has_vector: bool,
    /// The generation has a lexical configuration (`manifest.lex`).
    pub has_lexical: bool,
    /// The generation has a dense index (`dim > 0`).
    pub has_dense: bool,
}

/// A resolved search: which lists to run, how deep, and how to fuse them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchPlan {
    pub mode: SearchMode,
    pub top_k: u32,
    /// Depth each list is run at: `max(recall_k, top_k)` for a hybrid search, `top_k` otherwise.
    pub depth: u32,
    pub rrf_k: f64,
    pub dense_weight: f64,
    pub lexical_weight: f64,
}

/// Resolves a search request against the manifest defaults, or says why it cannot run.
pub fn plan_search(
    input: &SearchInput,
    request: &FusionParams,
    manifest: &FusionParams,
) -> Result<SearchPlan, &'static str> {
    let top_k = match input.top_k {
        None => DEFAULT_TOP_K,
        Some(t) if t.is_finite() && t.fract() == 0.0 && t >= 1.0 && t <= u32::MAX as f64 => {
            t as u32
        }
        Some(_) => return Err("topK must be an integer >= 1"),
    };
    if !input.has_text && !input.has_vector {
        return Err("search needs a text, a vector or both");
    }
    if input.has_text && !input.has_lexical {
        return Err("this generation has no lexical index (configure lexical first)");
    }
    if input.has_vector && !input.has_dense {
        return Err("this generation has no dense index");
    }
    let mode = match (input.has_vector, input.has_text) {
        (true, true) => SearchMode::Hybrid,
        (true, false) => SearchMode::Dense,
        _ => SearchMode::Lexical,
    };
    let recall_k = request
        .recall_k
        .or(manifest.recall_k)
        .unwrap_or(top_k.max(DEFAULT_RECALL_K));
    Ok(SearchPlan {
        mode,
        top_k,
        depth: if mode == SearchMode::Hybrid {
            recall_k.max(top_k)
        } else {
            top_k
        },
        rrf_k: request.rrf_k.or(manifest.rrf_k).unwrap_or(DEFAULT_RRF_K),
        dense_weight: request
            .dense_weight
            .or(manifest.dense_weight)
            .unwrap_or(1.0),
        lexical_weight: request
            .lexical_weight
            .or(manifest.lexical_weight)
            .unwrap_or(1.0),
    })
}

/// A hit's position in one list (1-based) and its score there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ListRank {
    pub rank: u32,
    pub score: f64,
}

/// One result of a unified search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchHit {
    pub id: u32,
    /// The fused RRF score (hybrid), else the list's own score (cosine or BM25).
    pub score: f64,
    pub dense: Option<ListRank>,
    pub lexical: Option<ListRank>,
}

/// One list as a host got it back: ids best first and their scores (same length).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankedList<'a> {
    pub ids: &'a [u32],
    pub scores: &'a [f64],
}

fn ranks(list: Option<RankedList<'_>>) -> std::collections::HashMap<u32, ListRank> {
    let mut out = std::collections::HashMap::new();
    if let Some(l) = list {
        for (i, (&id, &score)) in l.ids.iter().zip(l.scores).enumerate() {
            out.entry(id).or_insert(ListRank {
                rank: i as u32 + 1,
                score,
            });
        }
    }
    out
}

/// The final hits of a search planned by [`plan_search`]: the two lists fused with [`rrf`]
/// (hybrid), or the one list cut to `top_k`, each hit carrying its rank and score in the lists
/// it came from. The list for a mode that does not use it is ignored.
pub fn fuse_search(
    plan: &SearchPlan,
    dense: Option<RankedList<'_>>,
    lexical: Option<RankedList<'_>>,
) -> Result<Vec<SearchHit>, &'static str> {
    for l in [dense, lexical].into_iter().flatten() {
        if l.ids.len() != l.scores.len() {
            return Err("a list's ids and scores differ in length");
        }
    }
    let top = plan.top_k as usize;
    let single = |l: RankedList<'_>, is_dense: bool| {
        l.ids
            .iter()
            .zip(l.scores)
            .take(top)
            .enumerate()
            .map(|(i, (&id, &score))| {
                let r = Some(ListRank {
                    rank: i as u32 + 1,
                    score,
                });
                SearchHit {
                    id,
                    score,
                    dense: if is_dense { r } else { None },
                    lexical: if is_dense { None } else { r },
                }
            })
            .collect()
    };
    match plan.mode {
        SearchMode::Dense => dense
            .map(|l| single(l, true))
            .ok_or("a dense search needs the dense list"),
        SearchMode::Lexical => lexical
            .map(|l| single(l, false))
            .ok_or("a lexical search needs the lexical list"),
        SearchMode::Hybrid => {
            let (Some(d), Some(l)) = (dense, lexical) else {
                return Err("a hybrid search needs both lists");
            };
            let fused = rrf(
                &[
                    RrfList::new(plan.dense_weight, d.ids),
                    RrfList::new(plan.lexical_weight, l.ids),
                ],
                plan.rrf_k,
                top,
            );
            let (dr, lr) = (ranks(dense), ranks(lexical));
            Ok(fused
                .into_iter()
                .map(|(id, score)| SearchHit {
                    id,
                    score,
                    dense: dr.get(&id).copied(),
                    lexical: lr.get(&id).copied(),
                })
                .collect())
        }
    }
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

    fn input(text: bool, vector: bool) -> SearchInput {
        SearchInput {
            top_k: None,
            has_text: text,
            has_vector: vector,
            has_lexical: true,
            has_dense: true,
        }
    }

    #[test]
    fn plan_defaults_and_precedence() {
        let none = FusionParams::default();
        let p = plan_search(&input(true, true), &none, &none).unwrap();
        assert_eq!(
            p,
            SearchPlan {
                mode: SearchMode::Hybrid,
                top_k: 10,
                depth: 50,
                rrf_k: 60.0,
                dense_weight: 1.0,
                lexical_weight: 1.0
            }
        );
        // top_k above the recall floor raises the depth; single lists run at top_k.
        let big = SearchInput {
            top_k: Some(80.0),
            ..input(true, true)
        };
        assert_eq!(plan_search(&big, &none, &none).unwrap().depth, 80);
        assert_eq!(
            plan_search(&input(false, true), &none, &none)
                .unwrap()
                .depth,
            10
        );
        assert_eq!(
            plan_search(&input(true, false), &none, &none).unwrap().mode,
            SearchMode::Lexical
        );
        // request beats manifest beats the constants, field by field
        let manifest = FusionParams {
            rrf_k: Some(10.0),
            recall_k: Some(20),
            dense_weight: None,
            lexical_weight: Some(0.5),
        };
        let request = FusionParams {
            rrf_k: None,
            recall_k: Some(5),
            dense_weight: Some(2.0),
            lexical_weight: None,
        };
        let p = plan_search(&input(true, true), &request, &manifest).unwrap();
        assert_eq!(
            (p.rrf_k, p.depth, p.dense_weight, p.lexical_weight),
            (10.0, 10, 2.0, 0.5)
        );
    }

    #[test]
    fn plan_refusals() {
        let none = FusionParams::default();
        for t in [0.0, 1.5, -1.0, f64::NAN] {
            let i = SearchInput {
                top_k: Some(t),
                ..input(true, true)
            };
            assert_eq!(
                plan_search(&i, &none, &none),
                Err("topK must be an integer >= 1")
            );
        }
        assert!(plan_search(&input(false, false), &none, &none)
            .unwrap_err()
            .contains("text, a vector"));
        let no_lex = SearchInput {
            has_lexical: false,
            ..input(true, true)
        };
        assert!(plan_search(&no_lex, &none, &none)
            .unwrap_err()
            .contains("no lexical index"));
        let no_dense = SearchInput {
            has_dense: false,
            ..input(true, true)
        };
        assert!(plan_search(&no_dense, &none, &none)
            .unwrap_err()
            .contains("no dense index"));
    }

    #[test]
    fn fuse_search_hits() {
        let none = FusionParams::default();
        let (d_ids, d_scores) = ([7, 3, 9], [0.9, 0.8, 0.7]);
        let (l_ids, l_scores) = ([7, 9, 11], [12.0, 11.0, 10.0]);
        let d = RankedList {
            ids: &d_ids,
            scores: &d_scores,
        };
        let l = RankedList {
            ids: &l_ids,
            scores: &l_scores,
        };
        let mut plan = plan_search(&input(true, true), &none, &none).unwrap();
        plan.top_k = 3;
        let hits = fuse_search(&plan, Some(d), Some(l)).unwrap();
        let fused = rrf(
            &[RrfList::new(1.0, &d_ids), RrfList::new(1.0, &l_ids)],
            60.0,
            3,
        );
        assert_eq!(
            hits.iter().map(|h| (h.id, h.score)).collect::<Vec<_>>(),
            fused
        );
        assert_eq!(
            hits[0].dense,
            Some(ListRank {
                rank: 1,
                score: 0.9
            })
        );
        assert_eq!(
            hits[1].lexical,
            Some(ListRank {
                rank: 2,
                score: 11.0
            })
        );
        assert_eq!(hits[2].id, 3);
        assert_eq!(hits[2].lexical, None);
        // one list: its own scores, cut to top_k
        plan.mode = SearchMode::Lexical;
        plan.top_k = 2;
        let hits = fuse_search(&plan, None, Some(l)).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!((hits[1].id, hits[1].score, hits[1].dense), (9, 11.0, None));
        assert!(fuse_search(&plan, Some(d), None).is_err());
        let bad = RankedList {
            ids: &d_ids,
            scores: &[1.0],
        };
        assert!(fuse_search(&plan, None, Some(bad)).is_err());
    }
}
