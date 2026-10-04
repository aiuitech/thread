//! PyO3 surface over `thread-core` for the `thread_sdk` Python package (`thread_sdk._native`).
//! Pure compute, like `thread-wasm`: the Python side does every read and write and hands byte
//! buffers in. The dense side is below; the lexical side (tokenizer, segments, paged reader) is in
//! `lex.rs`.
//!
//! Byte arguments accept any C-contiguous buffer (`bytes`, `bytearray`, `memoryview`, `array`,
//! numpy arrays) and are borrowed without copying. Float arguments accept a float32/float64
//! buffer, raw little-endian f32 bytes (an `idx/*.f32` object as read) or a sequence of floats.
//! Heavy calls release the GIL, so queries on several Python threads scan in parallel.

use pyo3::buffer::PyUntypedBuffer;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::fmt::Display;
use std::sync::RwLock;
use thread_core::delta::{DeltaEncoder, DeltaView};
use thread_core::kmeans::{assign_tree_top_m, effective_probe_coarse};
use thread_core::math::normalize;
use thread_core::part::PartBuilder;
use thread_core::pq::{Codebook, QueryTable};
use thread_core::quant::{encode_residual_sq4, encode_residual_sq8};
use thread_core::route::{CentroidTree, Router};
use thread_core::routing::{
    decode_rows, encode_grouped, encode_rows, Elem, GroupedTable, PartialRouter,
};
use thread_core::search::{
    rerank_records, scan_delta as scan_delta_core, scan_slab, scan_slab_pq, ScanStats,
};
use thread_core::slab::{SlabView, SlabWriter};
use thread_core::tomb::Tombstones;
use thread_core::topk::{merge_dedup, pack_hits, unpack_hits, TopK};

mod lex;
mod run;

#[cfg(target_endian = "big")]
compile_error!(
    "thread-py reads little-endian index objects in place; big-endian hosts are not supported"
);

create_exception!(
    _native,
    MissingRanges,
    PyException,
    "A paged reader needs bytes it does not hold yet: `ranges` lists the blob-absolute (offset, len) ranges to fetch and feed."
);
create_exception!(
    _native,
    IndexFormatError,
    PyValueError,
    "Bad magic, version, header or configuration."
);
create_exception!(
    _native,
    IndexCorruptError,
    PyValueError,
    "An out-of-bounds read inside a validated section."
);
create_exception!(
    _native,
    IndexStateError,
    PyRuntimeError,
    "The reader is not opened, or has no such section."
);

pub(crate) fn value_err<E: Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn err_s<E: Display>(e: E) -> String {
    e.to_string()
}

// ---- argument conversion ----------------------------------------------------------------------

/// A bytes-like argument, borrowed (the buffer export pins it while we hold it).
pub(crate) struct Buf(PyUntypedBuffer);

impl Buf {
    pub(crate) fn as_slice(&self) -> &[u8] {
        let len = self.0.len_bytes();
        if len == 0 {
            return &[];
        }
        // SAFETY: the buffer is C-contiguous (checked in `extract`) and stays exported while
        // `self` lives; index objects are never mutated while a call reads them.
        unsafe { std::slice::from_raw_parts(self.0.buf_ptr() as *const u8, len) }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Buf {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let b = PyUntypedBuffer::get(&obj)?;
        if !b.is_c_contiguous() {
            return Err(PyValueError::new_err("buffer must be C-contiguous"));
        }
        Ok(Buf(b))
    }
}

pub(crate) fn opt_slice(b: &Option<Buf>) -> &[u8] {
    b.as_ref().map_or(&[], Buf::as_slice)
}

/// The buffer's struct format without its byte-order prefix (`@`, `=`, `<` are little-endian here).
fn plain_format(b: &PyUntypedBuffer) -> PyResult<Vec<u8>> {
    let f = b.format().to_bytes();
    match f.first() {
        Some(b'>') | Some(b'!') => Err(PyValueError::new_err(
            "big-endian buffers are not supported",
        )),
        Some(b'@') | Some(b'=') | Some(b'<') => Ok(f[1..].to_vec()),
        _ => Ok(f.to_vec()),
    }
}

fn raw(b: &PyUntypedBuffer) -> &[u8] {
    let len = b.len_bytes();
    if len == 0 {
        return &[];
    }
    // SAFETY: as in `Buf::as_slice`; the caller checked C-contiguity.
    unsafe { std::slice::from_raw_parts(b.buf_ptr() as *const u8, len) }
}

fn le_f32s(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// Floats from a float32/float64 buffer, raw little-endian f32 bytes, or a sequence.
pub(crate) fn f32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(b) = PyUntypedBuffer::get(obj) {
        if b.is_c_contiguous() {
            let fmt = plain_format(&b)?;
            let bytes = raw(&b);
            return match (fmt.as_slice(), b.item_size()) {
                (b"f", 4) => Ok(le_f32s(bytes)),
                (b"d", 8) => {
                    let (chunks, _) = bytes.as_chunks::<8>();
                    Ok(chunks.iter().map(|c| f64::from_le_bytes(*c) as f32).collect())
                }
                (b"B", 1) | (b"b", 1) | (b"c", 1) => {
                    if !bytes.len().is_multiple_of(4) {
                        return Err(PyValueError::new_err(format!(
                            "{} bytes are not a whole number of f32 values",
                            bytes.len()
                        )));
                    }
                    Ok(le_f32s(bytes))
                }
                (f, n) => Err(PyValueError::new_err(format!(
                    "expected float32/float64 values or raw f32 bytes, got buffer format {:?} (item size {n})",
                    String::from_utf8_lossy(f)
                ))),
            };
        }
    }
    obj.extract::<Vec<f32>>()
}

/// u32 values from an integer buffer (range-checked) or a sequence of ints.
pub(crate) fn u32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(b) = PyUntypedBuffer::get(obj) {
        if b.is_c_contiguous() {
            let fmt = plain_format(&b)?;
            let bytes = raw(&b);
            let size = b.item_size();
            let signed = matches!(fmt.as_slice(), b"b" | b"h" | b"i" | b"l" | b"q" | b"n");
            let unsigned = matches!(fmt.as_slice(), b"B" | b"H" | b"I" | b"L" | b"Q" | b"N");
            if !(signed || unsigned) || ![1, 2, 4, 8].contains(&size) {
                return Err(PyValueError::new_err(format!(
                    "expected integer values, got buffer format {:?}",
                    String::from_utf8_lossy(&fmt)
                )));
            }
            return bytes
                .chunks_exact(size)
                .map(|c| {
                    let mut w = [0u8; 8];
                    w[..size].copy_from_slice(c);
                    let v: i128 = if signed {
                        let shift = 64 - 8 * size as u32;
                        ((i64::from_le_bytes(w) << shift) >> shift) as i128
                    } else {
                        u64::from_le_bytes(w) as i128
                    };
                    u32::try_from(v).map_err(|_| PyValueError::new_err(format!("{v} is not a u32")))
                })
                .collect();
        }
    }
    obj.extract::<Vec<u32>>()
}

fn tomb_arg(bytes: &[u8]) -> Result<Option<Tombstones>, String> {
    if bytes.is_empty() {
        return Ok(None);
    }
    Tombstones::from_bytes(bytes).map(Some).map_err(err_s)
}

fn check_batch(len: usize, dim: usize) -> PyResult<usize> {
    if dim == 0 || !len.is_multiple_of(dim) {
        return Err(value_err(format!(
            "batch length {len} is not a multiple of dim {dim}"
        )));
    }
    Ok(len / dim)
}

fn f32_bytes<'py>(py: Python<'py>, v: &[f32]) -> Bound<'py, PyBytes> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    PyBytes::new(py, &out)
}

// ---- query side ---------------------------------------------------------------------------------

/// A parsed PQ codebook (`pq.bin`, "VIBQ"), kept across queries.
#[pyclass(frozen, module = "thread_sdk._native")]
pub struct PqCodebook {
    cb: Codebook,
}

#[pymethods]
impl PqCodebook {
    #[new]
    fn new(codebook: Buf) -> PyResult<Self> {
        Ok(PqCodebook {
            cb: Codebook::from_bytes(codebook.as_slice()).map_err(value_err)?,
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.cb.dim
    }

    #[getter]
    fn m(&self) -> usize {
        self.cb.m
    }
}

/// One prepared query: the unit vector, optionally `q·mean` for every fine cell (flat routing and
/// delta scans) and the PQ asymmetric-distance table (format 3 scans). Immutable; shared by every
/// scan of the query, on any thread.
#[pyclass(frozen, module = "thread_sdk._native")]
pub struct Query {
    q: Vec<f32>,
    qc: Option<Vec<f32>>,
    table: Option<QueryTable>,
}

#[pymethods]
impl Query {
    #[getter]
    fn dim(&self) -> usize {
        self.q.len()
    }

    #[getter]
    fn has_dots(&self) -> bool {
        self.qc.is_some()
    }

    #[getter]
    fn has_table(&self) -> bool {
        self.table.is_some()
    }

    /// The unit-normalized query vector.
    fn vector(&self) -> Vec<f32> {
        self.q.clone()
    }
}

/// The routing table of a generation (`coarse.f32` + `fine.f32`).
#[pyclass(frozen, module = "thread_sdk._native")]
pub struct Searcher {
    router: Router,
}

#[pymethods]
impl Searcher {
    #[new]
    fn new(
        py: Python<'_>,
        dim: usize,
        k1: usize,
        k2: usize,
        coarse_means: &Bound<'_, PyAny>,
        fine_means: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let coarse = f32_vec(coarse_means)?;
        let fine = f32_vec(fine_means)?;
        if dim == 0 || coarse.len() != k1 * dim || fine.len() != k1 * k2 * dim {
            return Err(value_err(format!(
                "centroid buffer size mismatch: {} coarse and {} fine floats for k1 {k1}, k2 {k2}, dim {dim}",
                coarse.len(),
                fine.len()
            )));
        }
        Ok(py.detach(|| Searcher {
            router: Router::new(dim, k1, k2, &coarse, fine),
        }))
    }

    /// A searcher over a compact routing table (`fine.f16` / `fine.i8`, or `fine.f32` bytes with
    /// `"f32"`): the rows are decoded to f32 once, so routing costs what it does over `fine.f32`.
    #[staticmethod]
    fn from_table(
        py: Python<'_>,
        dim: usize,
        k1: usize,
        k2: usize,
        coarse_means: &Bound<'_, PyAny>,
        table: Buf,
        elem: &str,
    ) -> PyResult<Self> {
        let elem = elem_arg(elem)?;
        let coarse = f32_vec(coarse_means)?;
        let table = table.as_slice();
        let k = k1 * k2;
        if dim == 0 || coarse.len() != k1 * dim {
            return Err(value_err(format!(
                "centroid buffer size mismatch: {} coarse floats for k1 {k1}, dim {dim}",
                coarse.len()
            )));
        }
        if table.len() != k * elem.row_bytes(dim) {
            return Err(value_err(format!(
                "{} routing table is {} bytes, expected {k} rows of {} bytes",
                elem.name(),
                table.len(),
                elem.row_bytes(dim)
            )));
        }
        py.detach(|| {
            let fine = decode_rows(table, dim, elem).map_err(err_s)?;
            Ok::<_, String>(Searcher {
                router: Router::new(dim, k1, k2, &coarse, fine),
            })
        })
        .map_err(value_err)
    }

    #[getter]
    fn dim(&self) -> usize {
        self.router.dim
    }

    #[getter]
    fn cells(&self) -> usize {
        self.router.cells()
    }

    /// Prepares a query: normalizes `q`, computes the cell dots when `dots` and the PQ table when a
    /// `codebook` is given.
    #[pyo3(signature = (q, dots = false, codebook = None))]
    fn query(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        dots: bool,
        codebook: Option<&Bound<'_, PqCodebook>>,
    ) -> PyResult<Query> {
        let mut q = f32_vec(q)?;
        if q.len() != self.router.dim {
            return Err(value_err(format!(
                "query dim {} != index dim {}",
                q.len(),
                self.router.dim
            )));
        }
        normalize(&mut q);
        let cb = codebook.map(|c| &c.get().cb);
        py.detach(|| {
            let qc = dots.then(|| self.router.cell_dots(&q));
            let table = cb
                .map(|cb| QueryTable::new(&q, cb))
                .transpose()
                .map_err(err_s)?;
            Ok::<_, String>(Query { q, qc, table })
        })
        .map_err(value_err)
    }

    /// Global ids of the `nprobe` cells to scan: flat over every cell when `probe_coarse` is 0 (or
    /// >= k1), using the query's cell dots when it has them; else two-level routing.
    fn route(
        &self,
        py: Python<'_>,
        query: &Bound<'_, Query>,
        probe_coarse: usize,
        nprobe: usize,
    ) -> PyResult<Vec<u32>> {
        let q = query.get();
        if q.q.len() != self.router.dim {
            return Err(value_err("query dim mismatch"));
        }
        let flat = probe_coarse == 0 || probe_coarse >= self.router.k1;
        let r = &self.router;
        Ok(py.detach(|| {
            let cells = match (&q.qc, flat) {
                (Some(dots), true) if dots.len() == r.cells() => {
                    r.route_flat_from_dots(dots, nprobe)
                }
                (_, true) => r.route_flat(&q.q, nprobe),
                (_, false) => r.route(&q.q, probe_coarse, nprobe),
            };
            cells.into_iter().map(|(c, _)| c as u32).collect()
        }))
    }
}

fn elem_arg(elem: &str) -> PyResult<Elem> {
    Elem::parse(elem).ok_or_else(|| {
        value_err(format!(
            "unknown routing table element type {elem:?} (f32, f16 or i8)"
        ))
    })
}

/// Encodes cell means (`k * dim` floats, e.g. `fine.f32`) as a routing table of `elem` rows:
/// `"f16"` (IEEE half), `"i8"` (`[scale f32][i8 * dim]` per row) or `"f32"`.
#[pyfunction]
fn routing_table_encode<'py>(
    py: Python<'py>,
    means: &Bound<'_, PyAny>,
    dim: usize,
    elem: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let elem = elem_arg(elem)?;
    let means = f32_vec(means)?;
    check_batch(means.len(), dim)?;
    let out = py.detach(|| encode_rows(&means, dim, elem));
    Ok(PyBytes::new(py, &out))
}

/// Decodes a routing table of `elem` rows to f32 means (raw little-endian f32 bytes).
#[pyfunction]
fn routing_table_decode<'py>(
    py: Python<'py>,
    table: Buf,
    dim: usize,
    elem: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let elem = elem_arg(elem)?;
    let v = decode_rows(table.as_slice(), dim, elem).map_err(value_err)?;
    Ok(f32_bytes(py, &v))
}

/// Builds a centroid tree ("VIBC") over `k = len(means) / dim` cell means: spherical k-means of the
/// means into `groups` groups, deterministic for `seed` (the Worker's `build_centroid_tree`).
#[pyfunction]
fn build_centroid_tree<'py>(
    py: Python<'py>,
    means: &Bound<'_, PyAny>,
    dim: usize,
    groups: usize,
    iters: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let means = f32_vec(means)?;
    let k = check_batch(means.len(), dim)?;
    if k == 0 {
        return Err(value_err("means is empty"));
    }
    let out = py.detach(|| CentroidTree::build(&means, dim, k, groups, iters, seed).to_bytes());
    Ok(PyBytes::new(py, &out))
}

/// The means (`k * dim`) in the group order of a centroid tree built over them ("VIBG").
#[pyfunction]
fn grouped_table_encode<'py>(
    py: Python<'py>,
    means: &Bound<'_, PyAny>,
    dim: usize,
    tree: Buf,
    elem: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let elem = elem_arg(elem)?;
    let means = f32_vec(means)?;
    let k = check_batch(means.len(), dim)?;
    let tree = CentroidTree::from_bytes(tree.as_slice()).map_err(value_err)?;
    if tree.dim != dim || tree.k != k {
        return Err(value_err(format!(
            "tree is for k = {} cells of dim {}, expected k = {k}, dim = {dim}",
            tree.k, tree.dim
        )));
    }
    let out = py.detach(|| encode_grouped(&means, dim, &tree, elem));
    Ok(PyBytes::new(py, &out))
}

/// Header of a grouped table: `[elem code, dim, k, groups, n_rows, data_off, row_bytes, byte_len]`.
#[pyfunction]
fn grouped_table_info(table: Buf) -> PyResult<Vec<u64>> {
    let t = GroupedTable::parse(table.as_slice()).map_err(value_err)?;
    Ok(vec![
        t.elem.code() as u64,
        t.dim as u64,
        t.k as u64,
        t.groups as u64,
        t.n_rows as u64,
        t.data_off as u64,
        t.row_bytes() as u64,
        t.byte_len() as u64,
    ])
}

/// Cold-start routing over a subset of the cell means: a centroid tree plus the rows of the
/// groups loaded so far from a tree-grouped table (`fine.grouped.*`). Thread-safe: loads take a
/// write lock, lookups a read lock.
#[pyclass(frozen, module = "thread_sdk._native")]
pub struct ColdRouter {
    inner: RwLock<PartialRouter>,
    dim: usize,
    k: usize,
}

impl ColdRouter {
    fn read(&self) -> std::sync::RwLockReadGuard<'_, PartialRouter> {
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }
    fn write(&self) -> std::sync::RwLockWriteGuard<'_, PartialRouter> {
        self.inner.write().unwrap_or_else(|e| e.into_inner())
    }
    fn check_q(&self, q: &[f32]) -> PyResult<()> {
        if q.len() != self.dim {
            return Err(value_err(format!(
                "query dim {} != index dim {}",
                q.len(),
                self.dim
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl ColdRouter {
    #[new]
    fn new(tree: Buf, elem: &str) -> PyResult<Self> {
        let elem = elem_arg(elem)?;
        let tree = CentroidTree::from_bytes(tree.as_slice()).map_err(value_err)?;
        let (dim, k) = (tree.dim, tree.k);
        Ok(ColdRouter {
            inner: RwLock::new(PartialRouter::new(tree, elem)),
            dim,
            k,
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    #[getter]
    fn k(&self) -> usize {
        self.k
    }

    #[getter]
    fn groups(&self) -> usize {
        self.read().groups()
    }

    #[getter]
    fn row_bytes(&self) -> usize {
        self.read().row_bytes()
    }

    #[getter]
    fn loaded_cells(&self) -> usize {
        self.read().loaded_cells()
    }

    #[getter]
    fn loaded_groups(&self) -> usize {
        self.read().loaded_groups()
    }

    /// `(first row, row count)` of group `g` in the grouped table.
    fn group_rows(&self, g: usize) -> PyResult<(usize, usize)> {
        let r = self.read();
        if g >= r.groups() {
            return Err(value_err(format!("group {g} >= {} groups", r.groups())));
        }
        Ok(r.group_rows(g))
    }

    /// Row of cell `c` in the grouped table (None if no group lists it).
    fn cell_row(&self, c: usize) -> Option<usize> {
        self.read().cell_row(c)
    }

    /// The listed cells whose means are not loaded, in order.
    fn missing_cells(&self, cells: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
        let cells = u32_vec(cells)?;
        let r = self.read();
        Ok(cells
            .into_iter()
            .filter(|&c| !r.has_cell(c as usize))
            .collect())
    }

    /// The groups among the `beam` best for the (unit) query that are not fully loaded, best first.
    fn missing_groups(&self, query: &Bound<'_, Query>, beam: usize) -> PyResult<Vec<usize>> {
        let q = &query.get().q;
        self.check_q(q)?;
        Ok(self.read().missing_groups(q, beam))
    }

    /// Installs rows `first_row..` (a whole number of rows); returns the cells newly loaded.
    fn add_rows(&self, py: Python<'_>, first_row: usize, rows: Buf) -> PyResult<usize> {
        let rows = rows.as_slice();
        py.detach(|| self.write().add_rows(first_row, rows).map_err(err_s))
            .map_err(value_err)
    }

    /// Prepares a query: normalizes `q`, fills `q·mean` for `dot_cells` (which must be loaded; the
    /// delta scans' base scores) and builds the PQ table when a `codebook` is given.
    #[pyo3(signature = (q, dot_cells = None, codebook = None))]
    fn query(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        dot_cells: Option<&Bound<'_, PyAny>>,
        codebook: Option<&Bound<'_, PqCodebook>>,
    ) -> PyResult<Query> {
        let mut q = f32_vec(q)?;
        self.check_q(&q)?;
        normalize(&mut q);
        let cells = dot_cells.map(u32_vec).transpose()?;
        let cb = codebook.map(|c| &c.get().cb);
        let k = self.k;
        py.detach(|| {
            let qc = match cells {
                Some(cells) => {
                    if let Some(&c) = cells.iter().find(|&&c| c as usize >= k) {
                        return Err(format!("cell id {c} >= {k} cells"));
                    }
                    let dots = self.read().cell_dots_for(&q, &cells).map_err(err_s)?;
                    let mut qc = vec![0f32; k];
                    for (&c, d) in cells.iter().zip(dots) {
                        qc[c as usize] = d;
                    }
                    Some(qc)
                }
                None => None,
            };
            let table = cb
                .map(|cb| QueryTable::new(&q, cb))
                .transpose()
                .map_err(err_s)?;
            Ok(Query { q, qc, table })
        })
        .map_err(value_err)
    }

    /// Global ids of the `nprobe` best loaded cells among the members of the best `beam` groups.
    fn route_beam(
        &self,
        py: Python<'_>,
        query: &Bound<'_, Query>,
        beam: usize,
        nprobe: usize,
    ) -> PyResult<Vec<u32>> {
        let q = &query.get().q;
        self.check_q(q)?;
        Ok(py.detach(|| {
            self.read()
                .route_beam(q, beam, nprobe)
                .into_iter()
                .map(|(c, _)| c as u32)
                .collect()
        }))
    }
}

fn scan_result<'py>(
    py: Python<'py>,
    r: Result<(Vec<u8>, ScanStats), String>,
) -> PyResult<(Bound<'py, PyBytes>, usize, usize)> {
    let (packed, st) = r.map_err(value_err)?;
    Ok((PyBytes::new(py, &packed), st.scanned, st.skipped))
}

/// Scans one slab (its `[0, scan_len)` prefix, or the whole object) for a prepared query; v3 slabs
/// need a query built with the PQ codebook. Tombstoned ids (`u32` LE, sorted unique) are skipped
/// before top-k insertion. Returns `(packed hits, scanned, skipped)`; packed hits are
/// `[count u32][id u32, score f32, idx u32]*`.
#[pyfunction]
#[pyo3(signature = (slab, query, k, tombstones = None))]
fn scan<'py>(
    py: Python<'py>,
    slab: Buf,
    query: &Bound<'py, Query>,
    k: usize,
    tombstones: Option<Buf>,
) -> PyResult<(Bound<'py, PyBytes>, usize, usize)> {
    let q = query.get();
    let (slab, tomb) = (slab.as_slice(), opt_slice(&tombstones));
    let r = py.detach(|| {
        let view = SlabView::parse(slab).map_err(err_s)?;
        if view.dim != q.q.len() {
            return Err(format!("slab dim {} != query dim {}", view.dim, q.q.len()));
        }
        let t = tomb_arg(tomb)?;
        let mut top = TopK::new(k);
        let st = if view.is_v3() {
            let table = q
                .table
                .as_ref()
                .ok_or("v3 (PQ) slab: prepare the query with the PQ codebook")?;
            scan_slab_pq(&view, table, &q.q, &mut top, t.as_ref()).map_err(err_s)?
        } else {
            scan_slab(&view, &q.q, &mut top, t.as_ref())
        };
        Ok((pack_hits(&top.into_sorted()), st))
    });
    scan_result(py, r)
}

/// Scans a whole delta segment with its int8 records (`q·mean[cell] + scale8 * (q·code8)`); the
/// query must carry its cell dots. Scores are on the reranked scale. Returns
/// `(packed hits, scanned, skipped)`.
#[pyfunction]
#[pyo3(signature = (delta, query, k, tombstones = None))]
fn scan_delta<'py>(
    py: Python<'py>,
    delta: Buf,
    query: &Bound<'py, Query>,
    k: usize,
    tombstones: Option<Buf>,
) -> PyResult<(Bound<'py, PyBytes>, usize, usize)> {
    let q = query.get();
    let qc =
        q.qc.as_ref()
            .ok_or_else(|| value_err("scan_delta needs a query prepared with dots=True"))?;
    let (delta, tomb) = (delta.as_slice(), opt_slice(&tombstones));
    let r = py.detach(|| {
        let v = DeltaView::parse(delta).map_err(err_s)?;
        let t = tomb_arg(tomb)?;
        let mut top = TopK::new(k);
        let st = scan_delta_core(&v, &q.q, qc, &mut top, t.as_ref()).map_err(err_s)?;
        Ok((pack_hits(&top.into_sorted()), st))
    });
    scan_result(py, r)
}

/// Merges concatenated packed hit buffers, dedupes by id (best score and that copy's `idx`), keeps
/// the best `k`.
#[pyfunction]
fn merge<'py>(py: Python<'py>, packed: Buf, k: usize) -> Bound<'py, PyBytes> {
    PyBytes::new(
        py,
        &pack_hits(&merge_dedup(unpack_hits(packed.as_slice()), k)),
    )
}

/// Scores `n` consecutive rerank records (`[scale8 f32][code8 i8 * dim]`) of one slab against the
/// query: `q·centroid + scale8 * (q·code8)`.
#[pyfunction]
fn rerank(
    py: Python<'_>,
    query: &Bound<'_, Query>,
    centroid: &Bound<'_, PyAny>,
    records: Buf,
    n: usize,
) -> PyResult<Vec<f32>> {
    let q = query.get();
    let centroid = f32_vec(centroid)?;
    let records = records.as_slice();
    py.detach(|| rerank_records(&q.q, &centroid, records, n))
        .map_err(value_err)
}

/// Unit-normalizes `n * dim` floats row by row with float64 arithmetic (the TypeScript SDK's
/// rule, so both write the same bytes). Returns `(f32 LE bytes, first bad row or -1)`; a row with a
/// zero or non-finite norm stops the conversion.
#[pyfunction]
fn normalize_rows<'py>(
    py: Python<'py>,
    vectors: &Bound<'py, PyAny>,
    dim: usize,
) -> PyResult<(Bound<'py, PyBytes>, i64)> {
    let mut v = f32_vec(vectors)?;
    check_batch(v.len(), dim)?;
    for (i, row) in v.chunks_exact_mut(dim).enumerate() {
        let ss: f64 = row.iter().map(|&x| x as f64 * x as f64).sum();
        if ss.is_nan() || ss <= 0.0 || !ss.is_finite() {
            return Ok((PyBytes::new(py, &[]), i as i64));
        }
        let inv = 1.0 / ss.sqrt();
        for x in row.iter_mut() {
            *x = (*x as f64 * inv) as f32;
        }
    }
    Ok((f32_bytes(py, &v), -1))
}

// ---- write side: delta segments, tombstones -------------------------------------------------------

/// Encodes new unit vectors into one delta segment ("VIBD"): each goes to its primary cell with
/// residual codes against that cell's mean (+ PQ codes with a codebook).
#[pyclass(module = "thread_sdk._native")]
pub struct DeltaJob {
    enc: Option<DeltaEncoder>,
}

impl DeltaJob {
    fn enc(&mut self) -> PyResult<&mut DeltaEncoder> {
        self.enc
            .as_mut()
            .ok_or_else(|| value_err("DeltaJob is finished"))
    }

    fn check_add(&mut self, vectors: &[f32], ids: &[u32], attrs: &[u32]) -> PyResult<usize> {
        let n = check_batch(vectors.len(), self.enc()?.dim())?;
        if ids.len() != n {
            return Err(value_err(format!(
                "ids has {} entries for {n} vectors",
                ids.len()
            )));
        }
        if !attrs.is_empty() && attrs.len() != n {
            return Err(value_err(format!(
                "attrs has {} entries for {n} vectors (pass n or none)",
                attrs.len()
            )));
        }
        Ok(n)
    }
}

#[pymethods]
impl DeltaJob {
    #[new]
    fn new(dim: usize, k: usize, means: &Bound<'_, PyAny>, seq: u32) -> PyResult<Self> {
        let means = f32_vec(means)?;
        Ok(DeltaJob {
            enc: Some(DeltaEncoder::new(dim, k, means, seq).map_err(value_err)?),
        })
    }

    /// Entries also carry PQ codes under this codebook (required for format 3). Call before adding.
    fn set_pq_codebook(&mut self, codebook: &Bound<'_, PqCodebook>) -> PyResult<()> {
        let cb = codebook.get().cb.clone();
        self.enc()?.set_codebook(cb).map_err(value_err)
    }

    /// Adds `n` unit vectors with their ids and attrs (`n`, or none for all 0) and optional payloads
    /// (`n` bytes-like). Returns each vector's primary cell.
    #[pyo3(signature = (vectors, ids, attrs = None, payloads = None))]
    fn add(
        &mut self,
        py: Python<'_>,
        vectors: &Bound<'_, PyAny>,
        ids: &Bound<'_, PyAny>,
        attrs: Option<&Bound<'_, PyAny>>,
        payloads: Option<Vec<Buf>>,
    ) -> PyResult<Vec<u32>> {
        let vectors = f32_vec(vectors)?;
        let ids = u32_vec(ids)?;
        let attrs = attrs.map(u32_vec).transpose()?.unwrap_or_default();
        let n = self.check_add(&vectors, &ids, &attrs)?;
        if let Some(p) = &payloads {
            if p.len() != n {
                return Err(value_err(format!("{} payloads for {n} vectors", p.len())));
            }
        }
        let slices: Vec<&[u8]> = payloads
            .as_ref()
            .map_or_else(Vec::new, |p| p.iter().map(Buf::as_slice).collect());
        let enc = self.enc()?;
        let dim = enc.dim();
        Ok(py.detach(|| {
            (0..n)
                .map(|i| {
                    let a = attrs.get(i).copied().unwrap_or(0);
                    let p = slices.get(i).copied().unwrap_or(&[]);
                    enc.add(&vectors[i * dim..(i + 1) * dim], ids[i], a, p) as u32
                })
                .collect()
        }))
    }

    fn __len__(&self) -> usize {
        self.enc.as_ref().map_or(0, |e| e.len())
    }

    #[getter]
    fn byte_len(&self) -> usize {
        self.enc.as_ref().map_or(0, |e| e.byte_len())
    }

    #[getter]
    fn pq_m(&self) -> usize {
        self.enc.as_ref().map_or(0, |e| e.pq_m())
    }

    /// The serialized delta segment (the job cannot be used afterwards).
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let enc = self
            .enc
            .take()
            .ok_or_else(|| value_err("DeltaJob is finished"))?;
        Ok(PyBytes::new(py, &enc.finish()))
    }
}

/// `[dim, count, pq_m, seq, byte_len]` of a delta segment.
#[pyfunction]
fn delta_info(delta: Buf) -> PyResult<Vec<u64>> {
    let v = DeltaView::parse(delta.as_slice()).map_err(value_err)?;
    Ok(vec![
        v.dim as u64,
        v.count as u64,
        v.pq_m as u64,
        v.seq as u64,
        v.byte_len() as u64,
    ])
}

/// Distinct cells of a delta segment with their entry counts, ascending: `[(cell, count)]`.
#[pyfunction]
fn delta_cells(delta: Buf) -> PyResult<Vec<(u32, u32)>> {
    Ok(DeltaView::parse(delta.as_slice())
        .map_err(value_err)?
        .cell_counts())
}

/// The union of a tombstone buffer (`u32` LE ids; may be empty) and `add`, sorted unique.
#[pyfunction]
fn tombstones_merge<'py>(
    py: Python<'py>,
    existing: Buf,
    add: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let add = u32_vec(add)?;
    let t = Tombstones::from_bytes(existing.as_slice()).map_err(value_err)?;
    Ok(PyBytes::new(py, &t.merge(&add).to_bytes()))
}

// ---- build side (tests, tools): assign, slabs, PQ ------------------------------------------------

/// Assign map job (the build pipeline's first stage, without the tree and candidate options of the
/// wasm job): each vector goes to its top `m` cells, encoded against each cell's mean in both tiers
/// (+ PQ codes after `set_codebook`), and buffered per coarse group.
#[pyclass(module = "thread_sdk._native")]
pub struct AssignJob {
    router: Router,
    probe_coarse: usize,
    m: usize,
    parts: Vec<PartBuilder>,
    codebook: Option<Codebook>,
}

#[pymethods]
impl AssignJob {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        k1: usize,
        k2: usize,
        coarse_means: &Bound<'_, PyAny>,
        fine_means: &Bound<'_, PyAny>,
        probe_coarse: usize,
        m: usize,
    ) -> PyResult<Self> {
        let coarse = f32_vec(coarse_means)?;
        let fine = f32_vec(fine_means)?;
        if dim == 0 || coarse.len() != k1 * dim || fine.len() != k1 * k2 * dim {
            return Err(value_err("centroid buffer size mismatch"));
        }
        if !dim.is_multiple_of(2) {
            return Err(value_err("dim must be even for 4-bit codes"));
        }
        Ok(AssignJob {
            router: Router::new(dim, k1, k2, &coarse, fine),
            probe_coarse: effective_probe_coarse(probe_coarse, k1),
            m: m.max(1),
            parts: (0..k1).map(|_| PartBuilder::new(k2, dim)).collect(),
            codebook: None,
        })
    }

    /// Records also carry PQ codes under this codebook (needed for format 3 slabs). Call before adding.
    fn set_codebook(&mut self, codebook: Buf) -> PyResult<()> {
        let cb = Codebook::from_bytes(codebook.as_slice()).map_err(value_err)?;
        if cb.dim != self.router.dim {
            return Err(value_err(format!(
                "codebook dim {} != job dim {}",
                cb.dim, self.router.dim
            )));
        }
        self.codebook = Some(cb);
        Ok(())
    }

    /// Adds unit vectors with ids `first_id, first_id + 1, ...`. Returns the number added.
    fn add_batch(
        &mut self,
        py: Python<'_>,
        vectors: &Bound<'_, PyAny>,
        first_id: u32,
    ) -> PyResult<usize> {
        let vectors = f32_vec(vectors)?;
        let dim = self.router.dim;
        let n = check_batch(vectors.len(), dim)?;
        let AssignJob {
            router,
            probe_coarse,
            m,
            parts,
            codebook,
        } = self;
        py.detach(|| {
            let (mut code4, mut code8, mut scratch) =
                (vec![0u8; dim / 2], vec![0i8; dim], vec![0f32; dim]);
            let mut pq = vec![0u8; codebook.as_ref().map_or(0, |c| c.m)];
            let mut resid = vec![0f32; dim];
            for (i, x) in vectors.chunks_exact(dim).enumerate() {
                let cells = assign_tree_top_m(
                    x,
                    router.coarse_mean(),
                    router.coarse_inv(),
                    router.fine_mean_all(),
                    router.fine_inv(),
                    router.k2,
                    dim,
                    *probe_coarse,
                    *m,
                );
                for (cell, _) in cells {
                    let mean = router.fine_mean(cell);
                    let scale4 = encode_residual_sq4(x, mean, &mut code4, &mut scratch);
                    let scale8 = encode_residual_sq8(x, mean, &mut code8);
                    if let Some(cb) = codebook.as_ref() {
                        for d in 0..dim {
                            resid[d] = x[d] - mean[d];
                        }
                        thread_core::pq::encode(&resid, cb, &mut pq);
                    }
                    let (g, j) = (cell / router.k2, cell % router.k2);
                    parts[g].push(
                        j,
                        first_id + i as u32,
                        scale4,
                        scale8,
                        0,
                        &code4,
                        &code8,
                        &pq,
                        &[],
                    );
                }
            }
        });
        Ok(n)
    }

    /// Serializes part `g` (one coarse group) and resets its buffer.
    fn take_part<'py>(&mut self, py: Python<'py>, g: usize) -> PyResult<Bound<'py, PyBytes>> {
        let part = self
            .parts
            .get_mut(g)
            .ok_or_else(|| value_err(format!("no part {g}")))?;
        Ok(PyBytes::new(py, &part.take()))
    }
}

/// Finalize job for one cell: part records in, one slab out (format 1, 2 or 3).
#[pyclass(module = "thread_sdk._native")]
pub struct SlabJob {
    dim: usize,
    writer: Option<SlabWriter>,
}

#[pymethods]
impl SlabJob {
    #[new]
    #[pyo3(signature = (dim, cell, centroid_mean, format = 2))]
    fn new(dim: usize, cell: u32, centroid_mean: &Bound<'_, PyAny>, format: u8) -> PyResult<Self> {
        let mean = f32_vec(centroid_mean)?;
        if mean.len() != dim {
            return Err(value_err("centroid size mismatch"));
        }
        if !dim.is_multiple_of(2) {
            return Err(value_err(
                "dim must be even (part records carry 4-bit codes)",
            ));
        }
        Ok(SlabJob {
            dim,
            writer: Some(SlabWriter::new(format, dim, cell, &mean).map_err(value_err)?),
        })
    }

    /// Adds every record of a records byte range. Returns the number added.
    fn add_records(&mut self, records: Buf) -> PyResult<u32> {
        let dim = self.dim;
        let w = self
            .writer
            .as_mut()
            .ok_or_else(|| value_err("SlabJob is finished"))?;
        w.add_part_records(records.as_slice(), dim)
            .map_err(value_err)
    }

    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let w = self
            .writer
            .take()
            .ok_or_else(|| value_err("SlabJob is finished"))?;
        Ok(PyBytes::new(py, &w.finish()))
    }
}

/// `(k2, dim, [(count, off, len)] per cell)`.
type PartDirectory = (usize, usize, Vec<(u32, u32, u32)>);

/// A part's directory: `(k2, dim, [(count, off, len)] per cell)`.
#[pyfunction]
fn part_directory(part: Buf) -> PyResult<PartDirectory> {
    let (k2, dim, dir) = thread_core::part::parse_directory(part.as_slice()).map_err(value_err)?;
    Ok((
        k2,
        dim,
        dir.into_iter().map(|d| (d.count, d.off, d.len)).collect(),
    ))
}

/// `[dim, count, cell, byte_len, version, scan_len, rerank_off, rerank_record_len, pq_m]` of a slab
/// (or scan prefix).
#[pyfunction]
fn slab_info(slab: Buf) -> PyResult<Vec<u64>> {
    let h = SlabView::parse(slab.as_slice()).map_err(value_err)?.header;
    Ok(vec![
        h.dim as u64,
        h.count as u64,
        h.cell as u64,
        h.total_len() as u64,
        h.version as u64,
        h.scan_len as u64,
        h.rerank_off as u64,
        h.rerank_record_len as u64,
        h.pq_m as u64,
    ])
}

/// `vectors[i] - means[assign[i]]` per row, as f32 LE bytes (PQ training residuals).
#[pyfunction]
fn residuals_for<'py>(
    py: Python<'py>,
    vectors: &Bound<'py, PyAny>,
    means: &Bound<'py, PyAny>,
    assign: &Bound<'py, PyAny>,
    dim: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let r =
        thread_core::pq::residuals(&f32_vec(vectors)?, &f32_vec(means)?, &u32_vec(assign)?, dim)
            .map_err(value_err)?;
    Ok(f32_bytes(py, &r))
}

/// An initial PQ codebook ("VIBQ" bytes) from residual samples (`dim % m == 0`).
#[pyfunction]
fn pq_init_codebook<'py>(
    py: Python<'py>,
    residuals: &Bound<'py, PyAny>,
    dim: usize,
    m: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let r = f32_vec(residuals)?;
    let cb = Codebook::init_from_samples(&r, dim, m, seed).map_err(value_err)?;
    Ok(PyBytes::new(py, &cb.to_bytes()))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("MissingRanges", py.get_type::<MissingRanges>())?;
    m.add("IndexFormatError", py.get_type::<IndexFormatError>())?;
    m.add("IndexCorruptError", py.get_type::<IndexCorruptError>())?;
    m.add("IndexStateError", py.get_type::<IndexStateError>())?;
    m.add_class::<PqCodebook>()?;
    m.add_class::<Query>()?;
    m.add_class::<Searcher>()?;
    m.add_class::<ColdRouter>()?;
    m.add_class::<DeltaJob>()?;
    m.add_class::<AssignJob>()?;
    m.add_class::<SlabJob>()?;
    m.add_class::<run::QueryRun>()?;
    m.add_function(wrap_pyfunction!(scan, m)?)?;
    m.add_function(wrap_pyfunction!(scan_delta, m)?)?;
    m.add_function(wrap_pyfunction!(merge, m)?)?;
    m.add_function(wrap_pyfunction!(rerank, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_rows, m)?)?;
    m.add_function(wrap_pyfunction!(delta_info, m)?)?;
    m.add_function(wrap_pyfunction!(delta_cells, m)?)?;
    m.add_function(wrap_pyfunction!(tombstones_merge, m)?)?;
    m.add_function(wrap_pyfunction!(part_directory, m)?)?;
    m.add_function(wrap_pyfunction!(slab_info, m)?)?;
    m.add_function(wrap_pyfunction!(residuals_for, m)?)?;
    m.add_function(wrap_pyfunction!(pq_init_codebook, m)?)?;
    m.add_function(wrap_pyfunction!(routing_table_encode, m)?)?;
    m.add_function(wrap_pyfunction!(routing_table_decode, m)?)?;
    m.add_function(wrap_pyfunction!(grouped_table_encode, m)?)?;
    m.add_function(wrap_pyfunction!(build_centroid_tree, m)?)?;
    m.add_function(wrap_pyfunction!(grouped_table_info, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    lex::register(m)
}
