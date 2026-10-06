//! A unified search's planning and fusion (`thread_core::fusion`): `plan_search` resolves a
//! request against `manifest.lex.defaults`, the host runs the lists it asks for, and
//! `fuse_search` turns them into the final hits.

use crate::value_err;
use pyo3::prelude::*;
use thread_core::fusion::{self, FusionParams, ListRank, RankedList, SearchInput};

/// `(rrf_k, recall_k, dense_weight, lexical_weight)`, each optional.
type Params = (Option<f64>, Option<u32>, Option<f64>, Option<f64>);

fn params(p: Option<Params>) -> FusionParams {
    let (rrf_k, recall_k, dense_weight, lexical_weight) = p.unwrap_or_default();
    FusionParams {
        rrf_k,
        recall_k,
        dense_weight,
        lexical_weight,
    }
}

/// A resolved search: `mode` ("dense", "lexical" or "hybrid"), `top_k`, `depth` (each list's
/// depth), `rrf_k`, `dense_weight` and `lexical_weight`.
#[pyclass(frozen, module = "thread_sdk._native")]
pub struct SearchPlan(fusion::SearchPlan);

#[pymethods]
impl SearchPlan {
    #[getter]
    fn mode(&self) -> &'static str {
        self.0.mode.as_str()
    }
    #[getter]
    fn top_k(&self) -> u32 {
        self.0.top_k
    }
    #[getter]
    fn depth(&self) -> u32 {
        self.0.depth
    }
    #[getter]
    fn rrf_k(&self) -> f64 {
        self.0.rrf_k
    }
    #[getter]
    fn dense_weight(&self) -> f64 {
        self.0.dense_weight
    }
    #[getter]
    fn lexical_weight(&self) -> f64 {
        self.0.lexical_weight
    }
}

/// Resolves a search request (`request`) against `manifest.lex.defaults` (`manifest`), each a
/// `(rrf_k, recall_k, dense_weight, lexical_weight)` tuple of optional values; an unset value
/// falls through to the next. Raises `ValueError` with the reason when the search cannot run.
#[pyfunction]
#[pyo3(signature = (top_k, has_text, has_vector, has_lexical, has_dense, request = None, manifest = None))]
fn plan_search(
    top_k: Option<f64>,
    has_text: bool,
    has_vector: bool,
    has_lexical: bool,
    has_dense: bool,
    request: Option<Params>,
    manifest: Option<Params>,
) -> PyResult<SearchPlan> {
    let input = SearchInput {
        top_k,
        has_text,
        has_vector,
        has_lexical,
        has_dense,
    };
    fusion::plan_search(&input, &params(request), &params(manifest))
        .map(SearchPlan)
        .map_err(value_err)
}

/// `(id, score, dense, lexical)` per hit; `dense` / `lexical` are `(rank, score)` or None.
type Hit = (u32, f64, Option<(u32, f64)>, Option<(u32, f64)>);

/// The final hits of a planned search from the lists the host fetched (`(ids, scores)`, best
/// first; None for a list the mode does not run): fused with RRF (hybrid) or the one list cut
/// to `top_k`.
#[pyfunction]
#[pyo3(signature = (plan, dense = None, lexical = None))]
fn fuse_search(
    plan: &SearchPlan,
    dense: Option<(Vec<u32>, Vec<f64>)>,
    lexical: Option<(Vec<u32>, Vec<f64>)>,
) -> PyResult<Vec<Hit>> {
    let d = dense
        .as_ref()
        .map(|(ids, scores)| RankedList { ids, scores });
    let l = lexical
        .as_ref()
        .map(|(ids, scores)| RankedList { ids, scores });
    let rank = |r: Option<ListRank>| r.map(|r| (r.rank, r.score));
    let hits = fusion::fuse_search(&plan.0, d, l).map_err(value_err)?;
    Ok(hits
        .into_iter()
        .map(|h| (h.id, h.score, rank(h.dense), rank(h.lexical)))
        .collect())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchPlan>()?;
    m.add_function(wrap_pyfunction!(plan_search, m)?)?;
    m.add_function(wrap_pyfunction!(fuse_search, m)?)?;
    Ok(())
}
