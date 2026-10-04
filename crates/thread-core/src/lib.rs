// Dense-side file written before rustfmt was adopted; kept as is (`cargo fmt` skips it).
#![cfg_attr(rustfmt, rustfmt::skip)]
//! thread-core: a blob-native search index, dense (IVF vectors) and lexical (BM25), with one
//! document id space and rank fusion between the two.
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
//! - attribute filters (`filter`): a predicate over each entry's 32-bit `attrs` word, applied
//!   inside the scans before top-k insertion, like tombstones
//! - incremental updates: new vectors go into small immutable delta segments
//!   (`delta`, scanned with int8 records next to the probed slabs), deletes
//!   are tombstones (`tomb`); compaction re-emits a slab's and the deltas'
//!   records for a cell (`SlabView::to_part_records`,
//!   `DeltaView::records_for_cell`) into an ordinary slab writer
//! - cell splits: `kmeans::split_cell` (2-means over a cell's decoded
//!   entries), then `slab::split_records` / `part::split_part_records`
//!   re-encode every entry against the nearer new mean (`part::RecordEncoder`)
//!
//! Lexical side (ported and reshaped from Wiper's thread engine; see `docs/lexical.md`):
//! - `source`: `ByteSource`, `Range`, `Plan`, `Prefetched` (demand paging over range reads),
//!   `SubSource`
//! - `container`: the `THREADIX` blob of named sections a segment is stored as
//! - `tokenize`: the bm25s-compatible tokenizer (Snowball English, stopwords, CJK bigrams,
//!   identifiers), configured per index
//! - `lex`: the `terms` (bucketed term table with df and posting ranges) and `postings`
//!   (gap/tf/length varints) sections, BM25 scoring with corpus-wide statistics so several
//!   segments score like one index, and their merge (compaction)
//! - `docs`: the `docs` section, document id -> metadata (+ optional text), bucketed by id
//! - `segment`: one object holding `meta`, `terms`, `postings` and `docs` for a batch of
//!   documents (an upsert) or for the whole corpus (a compaction)
//! - `fusion`: Reciprocal Rank Fusion of the dense and lexical lists
//! - `excerpts`, `rerank`: query-time passages and judge-based rank fusion

pub mod container;
pub mod delta;
pub mod docs;
pub mod error;
pub mod excerpts;
pub mod filter;
pub mod fusion;
pub mod lex;
pub mod kmeans;
pub mod math;
pub mod part;
pub mod pq;
pub mod quant;
pub mod rerank;
pub mod route;
pub mod routing;
pub mod search;
pub mod segment;
pub mod slab;
pub mod source;
pub mod tokenize;
pub mod tomb;
pub mod topk;
pub mod varint;

pub use error::{Error, FormatError};
pub use source::{ByteSource, Plan, Prefetched, Range, SourceError, SubSource, Window};
pub use topk::Hit;
