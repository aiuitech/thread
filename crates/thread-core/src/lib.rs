//! thread-core: a blob-native IVF vector index.
//!
//! Everything here operates on byte slices and `f32` slices. There is no I/O:
//! the caller (a Cloudflare Worker, a CLI, a test) fetches bytes from wherever
//! they live and hands them to these kernels. That keeps the crate compilable
//! for `wasm32-unknown-unknown` and natively from the same source.
//!
//! Index shape (see `slab` and `route`):
//! - two-level spherical k-means: `k1` coarse groups, each with `k2` fine cells
//! - every vector is written into the slabs of its `m` nearest fine cells
//! - a slab (v2) is one immutable blob holding ids, residual 4-bit scan codes,
//!   residual int8 rerank records (+ payload); v1 slabs (int8 only) still parse;
//!   v3 slabs replace the 4-bit tier with product-quantization codes (`pq`)
//! - `route::CentroidTree`: optional pruned (beam) search over the fine cell
//!   means for upserts, flat assignment and query routing
//! - a query routes to `nprobe` fine cells, scans the slabs' scan prefixes with
//!   the 4-bit codes, merges top-T, then reranks the winners with int8 records
//! - incremental updates: new vectors go into small immutable delta segments
//!   (`delta`, scanned with int8 records next to the probed slabs), deletes
//!   are tombstones (`tomb`); compaction re-emits a slab's and the deltas'
//!   records for a cell (`SlabView::to_part_records`,
//!   `DeltaView::records_for_cell`) into an ordinary slab writer
//! - cell splits: `kmeans::split_cell` (2-means over a cell's decoded
//!   entries), then `slab::split_records` / `part::split_part_records`
//!   re-encode every entry against the nearer new mean (`part::RecordEncoder`)

pub mod delta;
pub mod error;
pub mod kmeans;
pub mod math;
pub mod part;
pub mod pq;
pub mod quant;
pub mod route;
pub mod search;
pub mod slab;
pub mod tomb;
pub mod topk;

pub use error::FormatError;
pub use topk::Hit;
