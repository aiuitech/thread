//! The CPU half of one dense query (`QueryRun`), the same steps as the TypeScript SDK's `executeQuery`: the
//! Python side does every read and hands each buffer in as it arrives; this keeps the scan-tier candidates (the
//! best copy per id), the rerank records of slabs that were read whole, the phase-2 plan and the counters, so no
//! per-hit work happens in Python.
//!
//!   run = QueryRun(query, t, do_rerank, return_payload)
//!   run.add_slab(cell, prefix, full, tombstones)   # per probed slab, any thread, scans with the GIL released
//!   run.speculation()                              # optional: the running top-T's records worth reading early
//!   run.add_delta(delta, tombstones)               # per delta segment
//!   run.plan()                                     # global top-T, grouped by slab: records still to read
//!   run.add_records(cell, first_idx, bytes, idxs)  # the records read (speculatively or for the plan)
//!   run.finish(k)                                  # rerank, merge with the delta hits, the top k

use crate::{value_err, Buf, Query};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, MutexGuard};
use thread_core::delta::DeltaView;
use thread_core::search::{rerank_records, scan_delta, scan_slab, scan_slab_pq};
use thread_core::slab::SlabView;
use thread_core::tomb::Tombstones;
use thread_core::topk::{merge_dedup, Hit, TopK};

struct Cand {
    cell: u32,
    idx: u32,
    score: f32,
    attrs: u32,
    payload_off: u32,
    payload_len: u32,
    /// Payload bytes when the slab was read whole.
    payload: Option<Vec<u8>>,
}

struct SlabInfo {
    centroid: Vec<f32>,
    rerank_off: u64,
    rec_len: u64,
    payload_off: u64,
}

/// Records to read from one slab: `(cell, rerank_off, rec_len, [idx...])`, idx ascending.
type RecordReads = (u32, u64, u64, Vec<u32>);

struct DeltaHit {
    id: u32,
    score: f32,
    attrs: u32,
    payload: Option<Vec<u8>>,
}

#[derive(Default)]
struct RunState {
    packed: Vec<Hit>,
    cand: HashMap<u32, Cand>,
    /// Rerank records of local winners from slabs read whole, by (cell, idx).
    local: HashMap<(u32, u32), Vec<u8>>,
    /// Records the host read, by (cell, idx), with whether a speculative read brought them.
    fetched: HashMap<(u32, u32), (Vec<u8>, bool)>,
    slab_info: HashMap<u32, SlabInfo>,
    versions: HashSet<u16>,
    scanned: u64,
    scan_bytes: u64,
    slab_bytes: u64,
    tomb_dropped: u64,
    slabs_done: usize,
    delta_hits: Vec<DeltaHit>,
    delta_bytes: u64,
    delta_scanned: u64,
    delta_dropped: u64,
    merged: Vec<Hit>,
    /// Phase 2: per slab with winners, its winners `(id, idx)` by ascending idx.
    plan: Vec<(u32, Vec<(u32, u32)>)>,
    rerank_local: u64,
    spec_used: u64,
    spec_records: u64,
}

fn tombs(bytes: &[u8]) -> Result<Option<Tombstones>, String> {
    if bytes.is_empty() {
        return Ok(None);
    }
    Tombstones::from_bytes(bytes)
        .map(Some)
        .map_err(|e| e.to_string())
}

fn slice(buf: &[u8], off: u64, len: u64, what: &str) -> Result<Vec<u8>, String> {
    let end = off.checked_add(len).filter(|&e| e <= buf.len() as u64);
    match end {
        Some(e) => Ok(buf[off as usize..e as usize].to_vec()),
        None => Err(format!(
            "{what} [{off}, +{len}) is outside the buffer ({} bytes)",
            buf.len()
        )),
    }
}

/// One dense query's candidates and counters (see the module docs). Safe to feed from several threads.
#[pyclass(frozen, module = "thread_sdk._native")]
pub struct QueryRun {
    query: Py<Query>,
    t: usize,
    do_rerank: bool,
    return_payload: bool,
    st: Mutex<RunState>,
}

impl QueryRun {
    fn lock(&self) -> MutexGuard<'_, RunState> {
        self.st.lock().unwrap_or_else(|p| p.into_inner())
    }
}

#[pymethods]
impl QueryRun {
    /// `t`: candidates kept per slab and overall (`max(k, rerank)` with a rerank, else `k`); `do_rerank`: phase 2
    /// runs (formats 2/3); `return_payload`: keep the payloads of slabs read whole.
    #[new]
    fn new(query: Py<Query>, t: usize, do_rerank: bool, return_payload: bool) -> Self {
        QueryRun {
            query,
            t,
            do_rerank,
            return_payload,
            st: Mutex::new(RunState::default()),
        }
    }

    /// Scans one slab's scan prefix (`full`: the whole object, when the store read it whole) with the tombstones
    /// that apply to its revision. Returns `(rerankable, slabs_done)`: whether the slab joins phase 2 and how many
    /// rerankable slabs are done so far (the host's cue for speculation).
    #[pyo3(signature = (cell, prefix, full = None, tombstones = None))]
    fn add_slab(
        &self,
        py: Python<'_>,
        cell: u32,
        prefix: Buf,
        full: Option<Buf>,
        tombstones: Option<Buf>,
    ) -> PyResult<(bool, usize)> {
        let q = self.query.get();
        let (prefix, full) = (prefix.as_slice(), full.as_ref().map(Buf::as_slice));
        let tomb = crate::opt_slice(&tombstones);
        let (t, do_rerank, return_payload) = (self.t, self.do_rerank, self.return_payload);
        py.detach(|| {
            let view = SlabView::parse(prefix).map_err(|e| format!("slab {cell}: {e}"))?;
            let dim = q.q.len();
            if view.dim != dim {
                return Err(format!("slab {cell}: dim {} != query dim {dim}", view.dim));
            }
            let skip = tombs(tomb)?;
            let mut top = TopK::new(t);
            let st = if view.is_v3() {
                let table = q
                    .table
                    .as_ref()
                    .ok_or_else(|| format!("slab {cell} is v3 but the query has no PQ table"))?;
                scan_slab_pq(&view, table, &q.q, &mut top, skip.as_ref())
                    .map_err(|e| e.to_string())?
            } else {
                scan_slab(&view, &q.q, &mut top, skip.as_ref())
            };
            let hits = top.into_sorted();
            let h = view.header;
            let rerankable = do_rerank && h.version >= 2;
            if rerankable && h.rerank_record_len != 4 + dim {
                return Err(format!(
                    "slab {cell}: rerank record length {} != {}",
                    h.rerank_record_len,
                    4 + dim
                ));
            }
            let info = rerankable.then(|| SlabInfo {
                centroid: view.centroid(),
                rerank_off: h.rerank_off as u64,
                rec_len: h.rerank_record_len as u64,
                payload_off: h.payload_off as u64,
            });
            let mut found = Vec::with_capacity(hits.len());
            for hit in &hits {
                let e = view.entry(hit.idx as usize);
                let payload = match (full, return_payload) {
                    (Some(f), true) => Some(slice(
                        f,
                        h.payload_off as u64 + e.payload_off as u64,
                        e.payload_len as u64,
                        "payload",
                    )?),
                    _ => None,
                };
                let rec = match (full, &info) {
                    (Some(f), Some(i)) => Some(slice(
                        f,
                        i.rerank_off + hit.idx as u64 * i.rec_len,
                        i.rec_len,
                        "rerank record",
                    )?),
                    _ => None,
                };
                found.push((*hit, e, payload, rec));
            }
            let mut s = self.lock();
            s.slab_bytes += full.map_or(prefix.len(), <[u8]>::len) as u64;
            s.scan_bytes += prefix.len() as u64;
            s.tomb_dropped += st.skipped as u64;
            s.scanned += h.count as u64;
            s.versions.insert(h.version);
            s.packed.extend_from_slice(&hits);
            if let Some(i) = info {
                s.slab_info.insert(cell, i);
            }
            for (hit, e, payload, rec) in found {
                if let Some(prev) = s.cand.get(&hit.id) {
                    if hit.score <= prev.score {
                        continue;
                    }
                }
                s.cand.insert(
                    hit.id,
                    Cand {
                        cell,
                        idx: hit.idx,
                        score: hit.score,
                        attrs: e.attrs,
                        payload_off: e.payload_off,
                        payload_len: e.payload_len,
                        payload,
                    },
                );
                if let Some(r) = rec {
                    s.local.insert((cell, hit.idx), r);
                }
            }
            if rerankable {
                s.slabs_done += 1;
            }
            Ok((rerankable, s.slabs_done))
        })
        .map_err(value_err)
    }

    /// The running top-T's records not held locally, by slab: `[(cell, rerank_off, rec_len, [idx...])]`, idx
    /// ascending. Read them and hand them in with `add_records(..., spec=True)`.
    fn speculation(&self) -> Vec<RecordReads> {
        let mut s = self.lock();
        let mut top: Vec<&Cand> = s.cand.values().collect();
        top.sort_by(|a, b| b.score.total_cmp(&a.score));
        top.truncate(self.t);
        let mut by_cell: HashMap<u32, Vec<u32>> = HashMap::new();
        for c in top {
            if !s.local.contains_key(&(c.cell, c.idx)) && s.slab_info.contains_key(&c.cell) {
                by_cell.entry(c.cell).or_default().push(c.idx);
            }
        }
        let mut out = Vec::with_capacity(by_cell.len());
        let mut n = 0u64;
        for (cell, mut idxs) in by_cell {
            idxs.sort_unstable();
            n += idxs.len() as u64;
            let i = &s.slab_info[&cell];
            out.push((cell, i.rerank_off, i.rec_len, idxs));
        }
        s.spec_records += n;
        out.sort_by_key(|o| o.0);
        out
    }

    /// Scans a whole delta segment (int8 scores, reranked scale) with the tombstones newer than the revision that
    /// added it.
    #[pyo3(signature = (delta, tombstones = None))]
    fn add_delta(&self, py: Python<'_>, delta: Buf, tombstones: Option<Buf>) -> PyResult<()> {
        let q = self.query.get();
        let qc =
            q.qc.as_ref()
                .ok_or_else(|| value_err("delta scans need a query prepared with dots=True"))?;
        let delta = delta.as_slice();
        let tomb = crate::opt_slice(&tombstones);
        let (t, return_payload) = (self.t, self.return_payload);
        py.detach(|| {
            let v = DeltaView::parse(delta).map_err(|e| e.to_string())?;
            let skip = tombs(tomb)?;
            let mut top = TopK::new(t);
            let st =
                scan_delta(&v, &q.q, qc, &mut top, skip.as_ref()).map_err(|e| e.to_string())?;
            let hits: Vec<DeltaHit> = top
                .into_sorted()
                .into_iter()
                .map(|h| DeltaHit {
                    id: h.id,
                    score: h.score,
                    attrs: v.entry(h.idx as usize).attrs,
                    payload: return_payload.then(|| v.payload(h.idx as usize).to_vec()),
                })
                .collect();
            let mut s = self.lock();
            s.delta_bytes += delta.len() as u64;
            s.delta_scanned += v.count as u64;
            s.delta_dropped += st.skipped as u64;
            s.delta_hits.extend(hits);
            Ok::<_, String>(())
        })
        .map_err(value_err)
    }

    /// Hands in rerank records of slab `cell`: `data` holds the records of entries `first_idx, first_idx + 1, ...`
    /// back to back; the ones listed in `idxs` are kept.
    #[pyo3(signature = (cell, first_idx, data, idxs, spec = false))]
    fn add_records(
        &self,
        cell: u32,
        first_idx: u32,
        data: Buf,
        idxs: Vec<u32>,
        spec: bool,
    ) -> PyResult<()> {
        let data = data.as_slice();
        let mut s = self.lock();
        let rec_len = s
            .slab_info
            .get(&cell)
            .ok_or_else(|| {
                value_err(format!(
                    "records for slab {cell}, which was not scanned for phase 2"
                ))
            })?
            .rec_len;
        for i in idxs {
            let off = (i
                .checked_sub(first_idx)
                .ok_or_else(|| value_err("idx before first_idx"))? as u64)
                * rec_len;
            let rec = slice(data, off, rec_len, "record").map_err(value_err)?;
            s.fetched.insert((cell, i), (rec, spec));
        }
        Ok(())
    }

    /// Merges the slab candidates into the global top-T and, with phase 2, groups its winners by slab. Returns the
    /// records still to read: `[(cell, rerank_off, rec_len, [idx...])]`, idx ascending (empty without phase 2).
    fn plan(&self) -> PyResult<Vec<RecordReads>> {
        let mut s = self.lock();
        let merged = merge_dedup(std::mem::take(&mut s.packed), self.t);
        let mut out = Vec::new();
        let rerank = self.do_rerank && (s.versions.contains(&2) || s.versions.contains(&3));
        let mut plan: Vec<(u32, Vec<(u32, u32)>)> = Vec::new();
        if rerank {
            let mut by_cell: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
            for h in &merged {
                let c = s
                    .cand
                    .get(&h.id)
                    .ok_or_else(|| value_err(format!("merge returned unknown id {}", h.id)))?;
                by_cell.entry(c.cell).or_default().push((h.id, c.idx));
            }
            plan = by_cell.into_iter().collect();
            plan.sort_by_key(|p| p.0);
            for (cell, ws) in plan.iter_mut() {
                ws.sort_by_key(|w| w.1);
                let need: Vec<u32> = ws
                    .iter()
                    .map(|w| w.1)
                    .filter(|&i| {
                        !s.local.contains_key(&(*cell, i)) && !s.fetched.contains_key(&(*cell, i))
                    })
                    .collect();
                if !need.is_empty() {
                    let i = s
                        .slab_info
                        .get(cell)
                        .ok_or_else(|| value_err(format!("slab {cell} has no rerank layout")))?;
                    out.push((*cell, i.rerank_off, i.rec_len, need));
                }
            }
        }
        s.merged = merged;
        s.plan = plan;
        Ok(out)
    }

    /// Re-scores the planned winners with their int8 records, merges them with the delta hits (one copy per id, the
    /// best score) and returns the top `k`: `(hits, payload_reads)`, hits as `(id, score, attrs, source, payload)`
    /// and payload reads as `(hit position, cell, offset, len)` for slab hits whose payload is still to read
    /// (`return_payload` only).
    #[allow(clippy::type_complexity)]
    fn finish<'py>(
        &self,
        py: Python<'py>,
        k: usize,
    ) -> PyResult<(
        Vec<(u32, f32, u32, &'static str, Option<Bound<'py, PyBytes>>)>,
        Vec<(usize, u32, u64, u64)>,
    )> {
        let q = self.query.get();
        let return_payload = self.return_payload;
        type Top = Vec<(
            u32,
            f32,
            u32,
            bool,
            Option<Vec<u8>>,
            Option<(u32, u64, u64)>,
        )>;
        let top: Top = py
            .detach(|| {
                let mut guard = self.lock();
                let s = &mut *guard;
                let mut slab_results: Vec<(u32, f32)> = Vec::with_capacity(s.merged.len());
                if s.plan.is_empty() {
                    slab_results.extend(s.merged.iter().map(|h| (h.id, h.score)));
                } else {
                    for (cell, ws) in &s.plan {
                        let info = &s.slab_info[cell];
                        let mut recs = Vec::with_capacity(ws.len() * info.rec_len as usize);
                        for &(_, idx) in ws {
                            if let Some(r) = s.local.get(&(*cell, idx)) {
                                recs.extend_from_slice(r);
                                s.rerank_local += 1;
                            } else if let Some((r, spec)) = s.fetched.get(&(*cell, idx)) {
                                recs.extend_from_slice(r);
                                s.spec_used += *spec as u64;
                            } else {
                                return Err(format!(
                                    "slab {cell}: rerank record {idx} was not read"
                                ));
                            }
                        }
                        let scores = rerank_records(&q.q, &info.centroid, &recs, ws.len())
                            .map_err(|e| e.to_string())?;
                        slab_results.extend(ws.iter().zip(scores).map(|(&(id, _), sc)| (id, sc)));
                    }
                }
                // Final top k over slab and delta hits, one copy per id (the best score); equal scores keep the
                // order ids were first seen in (slab results, then delta hits), as the TypeScript SDK does.
                let mut best: HashMap<u32, (f32, bool, usize, usize)> = HashMap::new();
                let candidates = slab_results
                    .iter()
                    .enumerate()
                    .map(|(i, &(id, score))| (id, score, true, i))
                    .chain(
                        s.delta_hits
                            .iter()
                            .enumerate()
                            .map(|(i, d)| (d.id, d.score, false, i)),
                    );
                for (id, score, slab, i) in candidates {
                    let order = best.len();
                    match best.get_mut(&id) {
                        Some(b) if score > b.0 => *b = (score, slab, i, b.3),
                        Some(_) => {}
                        None => {
                            best.insert(id, (score, slab, i, order));
                        }
                    }
                }
                let mut ranked: Vec<(u32, (f32, bool, usize, usize))> = best.into_iter().collect();
                ranked.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0).then(a.1 .3.cmp(&b.1 .3)));
                ranked.truncate(k);
                Ok(ranked
                    .into_iter()
                    .map(|(id, (score, slab, i, _))| {
                        if slab {
                            let c = &s.cand[&id];
                            let (payload, read) = match (return_payload, &c.payload) {
                                (false, _) => (None, None),
                                (true, Some(p)) => (Some(p.clone()), None),
                                (true, None) if c.payload_len == 0 => (Some(Vec::new()), None),
                                (true, None) => {
                                    let off = s.slab_info.get(&c.cell).map(|i| i.payload_off);
                                    (
                                        None,
                                        off.map(|o| {
                                            (c.cell, o + c.payload_off as u64, c.payload_len as u64)
                                        }),
                                    )
                                }
                            };
                            (id, score, c.attrs, true, payload, read)
                        } else {
                            let d = &s.delta_hits[i];
                            (id, score, d.attrs, false, d.payload.clone(), None)
                        }
                    })
                    .collect())
            })
            .map_err(value_err)?;
        let mut hits = Vec::with_capacity(top.len());
        let mut reads = Vec::new();
        for (pos, (id, score, attrs, slab, payload, read)) in top.into_iter().enumerate() {
            if let Some((cell, off, len)) = read {
                reads.push((pos, cell, off, len));
            } else if return_payload && slab && payload.is_none() {
                return Err(value_err(format!(
                    "payload of id {id}: its slab was read without its layout"
                )));
            }
            let source = if slab { "slab" } else { "delta" };
            hits.push((
                id,
                score,
                attrs,
                source,
                payload.map(|p| PyBytes::new(py, &p)),
            ));
        }
        Ok((hits, reads))
    }

    /// The query's counters: scanned entries and bytes, phase 2, deltas, tombstoned entries skipped.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.lock();
        let d = PyDict::new(py);
        let rerank_slabs = s.plan.len();
        d.set_item("scanned", s.scanned)?;
        d.set_item("scan_bytes", s.scan_bytes)?;
        d.set_item("slab_bytes", s.slab_bytes)?;
        d.set_item("rerank", if rerank_slabs > 0 { self.t } else { 0 })?;
        d.set_item(
            "rerank_candidates",
            if rerank_slabs > 0 { s.merged.len() } else { 0 },
        )?;
        d.set_item("rerank_slabs", rerank_slabs)?;
        d.set_item("rerank_local", s.rerank_local)?;
        d.set_item("spec_records", s.spec_records)?;
        d.set_item("spec_used", s.spec_used)?;
        d.set_item("delta_bytes", s.delta_bytes)?;
        d.set_item("delta_scanned", s.delta_scanned)?;
        d.set_item("delta_hits", s.delta_hits.len())?;
        d.set_item("delta_dropped", s.delta_dropped)?;
        d.set_item("tomb_dropped", s.tomb_dropped)?;
        Ok(d)
    }
}
