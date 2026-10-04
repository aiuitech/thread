# Architecture

thread is a search index that lives in object storage and is queried by range reads. This page is the design in
one sitting: what the objects are, how a query reads them, how updates avoid rewriting anything, and where the
code lives. The other pages go deeper: [query-path.md](query-path.md) (slab formats, the two-phase query, caching,
routing), [build-pipeline.md](build-pipeline.md) (how an index is built), [profiles.md](profiles.md) (the layout
knobs), [update-path.md](update-path.md) (revisions, deltas, tombstones, compaction) and [lexical.md](lexical.md)
(BM25 segments, the document store, fusion).

## The dense index in one paragraph

Two-level spherical k-means (`k1` coarse groups x `k2` fine cells; `k1=1` is a flat k-means, the default since g2).
Every vector is written into the slabs of its `m` nearest fine cells (multi-assignment), so a single probed slab
already contains the query's neighbourhood. A slab is one immutable object holding the residuals of its members
(`x - centroid`) twice: a compact **scan tier** (PQ-192 codes since g4, 4-bit codes in g2/g3) and an int8
**rerank tier**, plus ids and an optional payload section. A query routes against the cached centroid table,
range-reads the scan tier of `nprobe` slabs, scans them with a bounded top-k heap, merges and dedupes, then
re-scores the best `rerank` candidates (150 by default) with their int8 records. Storage is traded for fewer bytes
read per query everywhere. Updates never rewrite an object: an upsert or delete adds a small delta segment and/or
tombstone set plus a new manifest *revision*; compaction folds them into new slab revisions, automatically once a
revision exceeds the generation's compaction thresholds; cells that outgrow a cap are split in two.

## The lexical index in one paragraph

Text lives in **segments** next to the slabs: immutable `THREADIX` containers holding BM25 postings (a bucketed
term table carrying each term's `df` and posting range, lists of `gap/tf/length` varints) and a `docs` store
(id -> metadata, optionally the text), read by range requests in two rounds per query. Segments share the slabs'
u32 ids and tombstone sets, so one upsert adds a vector, its text and its metadata, and one delete hides all of
them; a compaction merges segments into one base. A query can be a vector, a text or both (Reciprocal Rank
Fusion), and hits resolve through the one `docs` store. The lexical side was ported from Wiper's `thread` engine
(demand-paged byte sources, the `THREADIX` container, a bm25s-compatible tokenizer with Snowball English, CJK
bigrams and technical identifiers, RRF, excerpts) and reshaped for blob storage with sparse ids: no term ids or
vocabulary table, document lengths inline in the postings, a `docs` store bucketed by id, segments + tombstones
instead of whole-file appends. Details and tuning: [lexical.md](lexical.md).

## Why it is cheap to run

- **Immutable objects.** Nothing under `idx/{gen}/` is ever overwritten, so every object can sit in an edge cache
  (the Workers Cache API, a local disk cache) for as long as you like, and a query always reads a consistent
  snapshot. Writes are new objects plus a new manifest revision.
- **Range reads, not whole objects.** A query reads only the scan prefix of each probed slab, then the handful of
  int8 records it wants to rerank, with multi-range requests. On g4 that is about 6 MB per query at 95% recall,
  down from 41 MB in g1.
- **Small cold starts.** The centroid table is also published as `fine.f16` / `fine.i8` (12.6 / 6.3 MB), as
  tree-grouped rows a cold query range-reads by group, and in a per-revision base pack, so a fresh process routes
  its first query after `current.json` + one GET (see [query-path.md](query-path.md#cold-opens-compact-routing-cold-tree-routing-base-packs)).
- **Bounded memory.** The only resident state is the centroid table (25 MB at 4096 x 1536), a PQ codebook (1.5 MB)
  and a few KB per lexical segment. Everything else streams through. That is what lets the query path run in a
  128 MB Worker isolate, and the build path too.
- **Compute is just arithmetic.** Routing is one matrix-vector product, scanning is table lookups and adds. All of
  it is Rust compiled to wasm (Workers, Node) or to a native extension (Python).

## What the generations taught us

The recall gap was routing, not quantization (rank all cells, never prune through coarse averages). At fixed
recall the number of scanned entries is nearly constant, so replication (`m`) only trades hops for bytes and PQ is
the real bytes lever. The Cache API honours multi-range requests and is not bound by the 6-connection limit, which
makes a 150-record rerank cheap. The numbers behind this are in [benchmarks.md](benchmarks.md).

## Code layout

Everything numeric is in Rust (`crates/thread-core`), exposed to JS through wasm-bindgen (`crates/thread-wasm`) and
to Python through PyO3 (`crates/thread-py`). The SDKs do the I/O and hand bytes to the kernels; the Worker is one
such host that also runs the build.

- `crates/thread-core`   format, quantization, k-means accumulators, router, scan; lexical side: `source`
                      (demand paging), `container`, `tokenize`, `lex` (terms/postings), `docs`, `segment`,
                      `fusion`, `excerpts`, `rerank`
- `crates/thread-wasm`   wasm-bindgen jobs: FlatKmeansJob, TreeKmeansJob, IncKmeansJob, Reducer, AssignJob, SlabJob,
                      Searcher, cell_neighbors; PQ: residuals_for, pq_init_codebook, PqTrainJob, PqReducer,
                      PqCodebook, PqTable, scan_pq; updates: DeltaJob, scan_delta,
                      delta_info/cells/records_for_cell/payload, slab_to_records, tombstones_merge, filter_hits;
                      cell splits: split_slab, split_slab_records, split_part_records, reencode_records;
                      centroid tree: build_centroid_tree, centroid_tree_info,
                      Searcher.set_tree/route_beam/cell_dots_for/tree_recall/tree_recall_top, DeltaJob.set_tree,
                      AssignJob.set_tree (not used by the driver yet); compact routing: routing_table_encode/
                      decode, Searcher.from_table, grouped_table_encode/info, ColdRouter; lexical (`src/lex.rs`): LexTokenizer,
                      SegmentJob, SegmentReader (paged), merge_segments, rrf_fuse, best_doc_ranks, rerank_fuse,
                      Lines, Excerpter
- `crates/thread-cli`    native CLI: `seg build|inspect|search|merge`, `tokenize`, slab/scan benchmarks
- `crates/thread-py`     PyO3 extension (`thread_sdk._native`): Searcher, PqCodebook, the per-query `QueryRun`
                      (scan, merge, rerank), DeltaJob, tombstones, AssignJob/SlabJob, LexTokenizer, SegmentJob,
                      SegmentReader (paged), merge_segments
- `sdk/ts/`           `thread-sdk` (npm): the query path as a library (R2/S3/memory stores, Cache API; directory
                      stores and packs via `thread-sdk/node`), the lexical index, `search` (dense, lexical, hybrid),
                      the write path (upsert with texts, delete, segment compaction), snapshots and packs
                      (`pack.ts`, see [portability.md](portability.md)). See [sdk/ts/README.md](../sdk/ts/README.md)
- `sdk/python/`       `thread-sdk` (PyPI): the same library for Python (S3/R2, directory, pack and memory stores,
                      memory/disk caches), wire-compatible with `sdk/ts`. See
                      [sdk/python/README.md](../sdk/python/README.md)
- `examples/`         deployment targets. `cloudflare-worker/`: the `cf`-managed `vib-bench` Worker: build-stage
                      endpoints, `/query`, `/upsert`, `/delete`, compaction (manual and automatic), cell splits,
                      `/index/status`, rollback, gc (`src/index.ts`); `src/main.ts` runs build stages and writes in
                      build slots (Worker Loader). See
                      [examples/cloudflare-worker/README.md](../examples/cloudflare-worker/README.md)
- `bench/`            clients of the deployed Worker (build driver, compaction, query/update/split benchmarks);
                      `bench/prep/` dataset prep and shard upload; `bench/offline/` local numpy experiments. See
                      [bench/README.md](../bench/README.md)
- `docs/`             this directory, see [README.md](README.md)
- `data/`             (gitignored) dataset and prep output

The Worker (`examples/cloudflare-worker/`) is an I/O shell that streams R2 objects through the wasm kernels so no
stage holds more than a few tens of MB. The SDKs are the same query path as a library, plus the lexical side, the
write path and the fused `search`: `sdk/ts` for TypeScript over the wasm, `sdk/python` for Python over the PyO3
extension, both published as `thread-sdk` (npm, PyPI) and sharing one wire format.
