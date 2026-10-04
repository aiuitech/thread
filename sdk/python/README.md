# thread-sdk (Python)

Query and update a [thread](../../README.md) blob-native search index from Python: dense vectors (the IVF slabs),
BM25 text (lexical segments), or both fused by rank, with one document store for the hits. The index lives as
immutable objects in R2, any S3-compatible store or a local directory. It is the same index and wire format the
Worker and the [TypeScript SDK](../ts/README.md) read and write, and the numeric kernels are the same Rust code
(`crates/thread-core`), compiled as a native extension.

```python
from thread_sdk import S3BlobStore, open_index

store = S3BlobStore.r2(account_id, "vib-bench", access_key_id, secret_access_key)
with open_index(store) as index:
    hits = index.query(embedding, top_k=10).hits            # dense: [Hit(id, score, attrs, source), ...]
    r = index.search(text="deploy failed", vector=embedding, top_k=10, docs=True)
    # hybrid: r.hits = [SearchHit(id, score, dense=Rank(rank, score), lexical=Rank(...), doc=DocRecord(id, meta)), ...]
```

## Install

```sh
pip install thread-sdk            # once published: one abi3 wheel per platform, CPython >= 3.11
```

No runtime dependencies (the S3 client and SigV4 signing use the standard library). From a checkout:

```sh
cd sdk/python
uv venv .venv && . .venv/bin/activate && uv pip install maturin pytest
maturin develop --uv              # builds crates/thread-py into thread_sdk/_native (profile `python`)
maturin build                     # a wheel in ../../target/wheels
```

The extension is built with the workspace's `python` cargo profile: `release` with `panic = "unwind"`, so a Rust
panic surfaces as a Python exception instead of aborting the interpreter (`release` aborts, which suits the wasm
build).

### Why a native extension (and not the wasm or the CLI)

- The wasm module's bindings are wasm-bindgen glue for JavaScript (JS strings, typed arrays, `JsValue` errors);
  calling it from Python would mean reimplementing that glue over a wasm runtime, for a slower result.
- The `thread` CLI works on local files (segment tools, benchmarks); it cannot read an index from R2, and a process
  per query would reload the routing table (25 MB at 4096 x 1536) every time.
- PyO3 binds `thread-core` directly (`crates/thread-py`), like `thread-wasm` does for JS: Python does the I/O,
  Rust the numbers. The kernels release the GIL, so queries on several threads scan in parallel.

## Pointing at the index

```python
from thread_sdk import DiskCache, FileBlobStore, MemoryBlobStore, MemoryCache, PackedBlobStore, S3BlobStore, open_index

# 1. Any S3-compatible endpoint: R2's S3 API, AWS S3, MinIO, ...
store = S3BlobStore("https://<account>.r2.cloudflarestorage.com", "vib-bench", key_id, secret)  # region "auto"
store = S3BlobStore("https://s3.us-east-1.amazonaws.com", "bucket", key_id, secret, region="us-east-1",
                    path_style=False, session_token=token)

# 2. A local directory holding the bucket's layout (a synced copy, or a local index you write to)
store = FileBlobStore("/data/vib-bench")
index = open_index("/data/vib-bench")                        # a path does the same

# 3. One file: a pack of one revision (see "Snapshots and packs"), on disk or as an object in a bucket
index = open_index("/data/vib-bench-g4-rev12.threadpack")
store = PackedBlobStore(S3BlobStore(...), "packs/g4-rev12.threadpack")

# 4. Memory, or your own BlobStore subclass
store = MemoryBlobStore()

index = open_index(store)                                    # remote stores get a 256 MB MemoryCache by default
index = open_index(store, cache=DiskCache("/var/cache/thread", max_bytes=8 << 30))   # survives restarts
index = open_index(store, cache=False)
```

The layout under the store is the one thread's build pipeline writes (`idx/current.json`,
`idx/{gen}/manifest[.rev].json`, `slabs[.rev].json`, `slab/{cell:05}[.rev]`, `delta/{seq}.vibd`, `seg/*.thread`,
`tombstones.{rev}.u32`, `coarse.f32`, `fine.f32`, `pq.bin`).

A cache plays the part the Workers Cache API plays for the TypeScript SDK: objects under `idx/` are immutable, so
whole objects are cached and ranges are served from them (`current.json` is never cached). A phase-1 miss reads the
whole slab once (concurrent misses of the same object share one read) and keeps the local winners' rerank records,
so phase 2 costs nothing for that slab. `MemoryCache` is an LRU in process memory; `DiskCache` keeps files in a
directory shared by every process that uses it and reads only the requested ranges.

## API

### `open_index(store, **options) -> ThreadIndex`

`store` is a `BlobStore`, or a path: a directory holding the layout (`FileBlobStore`) or a pack file
(`PackedBlobStore`); `store_at(path)` is the function behind that.

| option | default | meaning |
|---|---|---|
| `prefix` | `""` | key prefix in front of `idx/` (several indexes in one bucket) |
| `gen`, `rev` | `idx/current.json` | read a specific generation / revision; with both, the index is pinned and never refreshes |
| `cache` | `MemoryCache()` for remote stores, none for local ones | a `MemoryCache` / `DiskCache`; `True` = default cache, `False` = none |
| `refresh_interval` | 30.0 s | a query older than this re-reads `current.json` first (0 = only `refresh()`) |
| `delta_cache_bytes` | 40 MB | delta segments kept in memory across queries |
| `lex_transient_bytes` | 8 MB | transient lexical bytes (buckets, posting lists, texts) each segment reader may hold |
| `io_workers` | 32 | threads for reads, shared by concurrent queries |
| `job_workers` | 4-16 (CPUs) | threads for `query_many` and the lexical half of a hybrid search |
| `warm_pack` | - | key of a warm pack (hot slab prefixes in one object): fetched with one GET on a background thread at open, its prefixes then served from memory in front of the cache (`index.warm_ready`, `index.warm_stats`; see docs/cold-start.md, "Warm packs") |

`gen` without `rev` reads the generation's head (the current rev when `current.json` names that generation, else
the newest `manifest.*.json` in the store). `ThreadIndex` is a context manager; `close()` releases its threads.

`index.open_stats` is the cold-start timeline of `open_index`: `{total, current, manifest, slabs, base, pack?}`, each
step a phase stat (`current`: current.json; `manifest`; `slabs`: slabs table + tombstone sets; `base`: routing table
and codebook, plus `fetch` / `build` ms and `files: {coarse, fine, pq}` bytes; `pack`, for a `PackedBlobStore`, its
own head + tail + directory reads, `PackedBlobStore.open_stats`, not part of `total`). A "read" is a request that
reached the origin store; with a cache in front, `hits` / `misses` count cache lookups. Same fields as the
TypeScript SDK's `index.openStats` (which adds `wasmInit`). `bench/cold/` measures these against S3.

Properties: `manifest` (the stored dict, camelCase keys), `gen`, `rev`, `dim`, `cells` (`k1 * k2`), `format`
(1 int8 only, 2 sq4 + sq8, 3 pq + sq8), `defaults` (`{nprobe, rerank, pc}`), `has_dense`, `lex_config`
(`ResolvedLexConfig` or None), `segments`, `revision` (the parsed `RevState`).

### `index.query(vector, **options) -> QueryResult`

`vector`: `dim` floats of any norm (normalized by the SDK): a list, an `array('f')`/`array('d')`, a numpy array
(float32 or float64) or raw little-endian f32 bytes.

| option | default | meaning |
|---|---|---|
| `top_k` | 10 | results |
| `nprobe` | manifest default | slabs (cells) to scan |
| `rerank` | manifest default | scan-tier candidates re-scored with the int8 tier (`T = max(top_k, rerank)`); 0 = scan-tier scores only |
| `pc` | manifest default | coarse groups to probe (two-level indexes); 0 = flat routing over all cells |
| `return_payload` | False | also return each hit's payload bytes |
| `inflight` | 12 | reads in flight per query (remote stores) |
| `speculate` | True | read the running top-T's rerank records while the last 6 slabs are still scanning (remote stores) |

```python
@dataclass
class QueryResult:
    hits: list[Hit]          # Hit(id, score, attrs, source: "slab" | "delta", payload: bytes | None)
    params: dict             # {top_k, nprobe, rerank, pc, source: {nprobe, rerank, pc}} ("query" | "manifest" | "legacy")
    stats: dict              # gen, rev, format, routing, cells, scanned, scan_bytes, slab_bytes,
                             # rerank, rerank_candidates, rerank_slabs, rerank_reads, rerank_bytes, rerank_local,
                             # spec_records, spec_used, deltas, delta_bytes, delta_scanned, delta_hits, delta_dropped,
                             # delta_top, tombstones, tomb_sets, tomb_dropped,
                             # io: {reads, hits, misses, r206, r200, bytes}, timing: {total, route, phase1, phase2, payload} (ms),
                             # timeline: {route, phase1, phase2, payload, base?} (phase stats, below)
```

`stats["timeline"]` splits the query into phases, each a phase stat `{ms, reads, hits, misses, r206, r200, bytes}`:
its wall time and the store work that completed during it (trace snapshots at the phase boundaries). `base` is
present when this query (re)loaded the routing table (a switch to another generation). Same fields as the
TypeScript SDK's `stats.timeline`.

`index.query_many(vectors, **options)` runs the queries concurrently against the same snapshot.

### `index.search(text=None, vector=None, **options) -> SearchResult`

Text, vector or both (see [docs/lexical.md](../../docs/lexical.md)): the dense list and the BM25 list run
concurrently at depth `recall_k` and are fused with Reciprocal Rank Fusion; a single list is returned as is.

| option | default | meaning |
|---|---|---|
| `top_k` | 10 | results |
| `recall_k` | `lex.defaults.recallK`, else max(top_k, 50) | depth of each list before fusion |
| `rrf_k` | `lex.defaults.rrfK`, else 60 | RRF constant |
| `weights` | `lex.defaults.weights`, else 1 / 1 | `{"dense": .., "lexical": ..}` list weights |
| `dense` | - | `query` options of the dense list (`{"nprobe": 8, ...}`) |
| `docs` | False | `True`: resolve each hit through the `docs` store; `"text"`: the stored text too |

`SearchResult(hits, mode, params, dense, lexical, stats)`: `mode` is `"dense"`, `"lexical"` or `"hybrid"`; each
`SearchHit(id, score, dense, lexical, doc)` carries its rank and score in each list (`Rank(rank, score)`) and, with
`docs`, its `DocRecord(id, meta, text)` (None when the id has no docs entry). `dense` / `lexical` hold the full
list results (`QueryResult` / `LexSearchResult(hits, tokens, found, stats)`).

`index.lexical_search(text, k)` is the BM25 list alone, `index.tokenize(text)` the query's tokens,
`index.docs(ids, text=False)` the document store (`{id: DocRecord}`, newest live copy of each id).

### `index.refresh() -> bool`

Re-reads `current.json` (or the generation's head) and switches to the new revision if it changed. Revisions are
immutable, so everything else is cached per index; new data is visible within `refresh_interval`, at once after
`refresh()`, or at once on the handle that wrote it.

### Write path

```python
index.configure_lexical(text=True, lengths="exact", bucket_terms=256)    # once: manifest.lex (a revision)
r = index.upsert([1000, 1001], [emb_a, emb_b], texts=["...", "..."], metas=["Title A", "Title B"], attrs=[5, 6])
# UpsertResult(gen, rev, seq, count, cells, distinct_cells, keys, bytes, conflicts, cas_conflicts, current_updated, ms)
index.upsert([1002], texts=["text-only document"], metas=["C"])        # no vector: found by text alone
index.delete([1000])                    # hides the vector, the text and the doc
index.compact_segments()                # merge the lexical/docs segments into one base segment
create_lexical_generation(store, "docs1", {"text": True})               # a text-only generation (no dense index)
```

`vectors` may be `n * dim` floats, one row per id, an `(n, dim)` numpy array or raw f32 bytes. Up to 4096 ids per
call. `replace=False` skips the tombstone set when the ids are known to be new. `configure_lexical` takes the
wire-format dict (`{"bucketTerms": 64}`) or snake_case keyword arguments; the tokenizer of a generation with
segments is frozen.

Commits follow docs/update-path.md, as the TypeScript SDK does: a delta segment and/or a lexical segment and a
tombstone set, then a create-only put of `manifest.{rev+1}.json` (the writer lock; a lost race rebuilds on the new
head) and a compare-and-swap of `idx/current.json`. Needs a store with conditional puts (`S3BlobStore`,
`FileBlobStore`, `MemoryBlobStore`); revisions written from Python record `change.writer: "thread-sdk-python"`.
Compaction of slabs and GC stay Worker-only.

### `BlobStore`

```python
class BlobStore:
    supports_put = supports_list = supports_ranges = remote = False
    def get(self, key, range: BlobRange | None = None, *, trace=None, fill=False) -> bytes | memoryview | None: ...
    def get_ranges(self, key, ranges, *, trace=None, fill=False) -> list | None: ...   # None: read ranges one by one
    def head(self, key) -> BlobHead | None: ...
    def put(self, key, data, *, if_absent=False, if_match=None, content_type=None) -> BlobHead | None: ...
    def list(self, prefix) -> list[str]: ...
```

`get` may return a `memoryview` into a larger buffer (the whole object fetched on a cache miss). Stores must be
thread-safe. `remote = True` sends reads through the thread pool (in parallel, `inflight` per query, with
speculative phase-2 reads); local stores are read on the calling thread.

Stores: `S3BlobStore`, `FileBlobStore(root)` (atomic puts, create-only via hard link, CAS under an advisory lock),
`MemoryBlobStore`, `PrefixedBlobStore(inner, prefix)`, `PackedBlobStore(inner, key)` (the objects of a pack under
their original keys, each read as a range of the pack; read-only; `PackedBlobStore.from_bytes(data)` for a pack in
memory) and `CachedBlobStore(inner, cache)`. Put a cache in front of a pack, not behind it (`open_index` does).

### Snapshots and packs

```python
from thread_sdk import copy_objects, pack_index, pack_to_bytes, snapshot_keys, store_at

keys = snapshot_keys(index.revision)                 # every object the open revision reads, sorted
copy_objects(src, dst)                               # everything src lists -> dst (any two stores): (copied, bytes, skipped)
copy_objects(src, dst, ["idx/current.json", *keys])  # one consistent revision
copy_objects(src, dst, skip_existing=True, workers=16)

r = pack_index(src, "idx.threadpack", gen="g4", rev=12)   # PackResult(gen, rev, count, bytes, keys); atomic rename
pack_index(src, sys.stdout.buffer)                        # any binary file object: streams, never seeks back
data, r = pack_to_bytes(src)                              # the same pack in memory (small indexes)
index = open_index("idx.threadpack")                      # queried in place; store_at(path) gives the store alone
```

`pack_index` writes `idx/current.json` (pointing at the packed revision) and then `snapshot_keys` in key order, so
the two SDKs produce identical files for the same revision (the interop test checks). A pack is read-only:
`copy_objects(pack, dst)` into a directory or a bucket to write on top of it. Format and rationale:
[docs/portability.md](../../docs/portability.md).

## Threads, asyncio

A `ThreadIndex` is safe to share between threads: queries, searches and writes may run concurrently. The calls
are blocking; from asyncio, run them in a thread (`await asyncio.to_thread(index.search, text=q)`). Internally a
query never waits on its own pool from inside the pool, so any number of concurrent queries share `io_workers`
threads without deadlock.

## Compatibility with the TypeScript SDK

`tests/test_interop.py` has the TypeScript SDK build and update an index on disk (deltas, lexical segments,
replacements, deletes), checks that Python returns the same answers, then writes on top from Python (upserts,
deletes, a segment compaction) and checks that the TypeScript SDK returns what Python returns; finally both SDKs
pack that revision and the files must be byte-identical, each SDK answering from the other's pack. Ids, their order,
attrs, payloads, docs, tokens, BM25 scores and fused scores agree exactly. Cosine scores agree to about 1e-7: the
wasm build accumulates dot products in 16 SIMD lanes and the native build in 8 scalar partial sums (see
`thread-core/src/math.rs`), so a score can differ in its last float32 ulp or two.

Differences from the TypeScript API: snake_case names and keyword options; results are dataclasses (stats stay
dicts with snake_case keys); `docs()` returns a dict; there is no wasm instance to reset (`Thread.init`,
`withRecovery`); the Cache API is replaced by `MemoryCache` / `DiskCache`; `FileBlobStore` and `store_at` live in the
package itself rather than a `thread-sdk/node` entry.

## Performance

A local index of 100 000 vectors (dim 384, 256 cells, PQ-48 + int8, m 2), `nprobe 16, rerank 100`, 300 queries
(12 700 entries scanned per query), on one laptop:

| | p50 | p95 | 8 threads |
|---|---:|---:|---:|
| Python, `FileBlobStore` | 1.04 ms | 1.24 ms | 1 044 queries/s |
| Python, `FileBlobStore` + `MemoryCache` | 0.80 ms | 1.05 ms | 1 933 queries/s |
| Python, the remote (thread-pool) path over a warm `MemoryCache` | 1.29 ms | 2.49 ms | 431 queries/s |
| TypeScript SDK in Node (wasm), files in memory | 1.64 ms | 3.46 ms | - |

Against R2 the network dominates (see the TypeScript SDK's VERIFICATION.md for edge numbers); the remote path
keeps `inflight` reads per query in flight and scans each slab as it arrives.

## Development

```sh
cd sdk/python && . .venv/bin/activate
maturin develop --uv && pytest            # end to end against memory/file/pack/S3 (local server) stores
(cd ../ts && npm run build)               # test_interop.py needs node and sdk/ts/dist, else it is skipped
mypy thread_sdk
```

Layout: `crates/thread-py` (the PyO3 bindings: dense kernels and the per-query `QueryRun` in `lib.rs`/`run.rs`, the
lexical reader and builders in `lex.rs`), `thread_sdk/` (`index.py` the `ThreadIndex`, `revision.py` resolving and loading a
revision and `snapshot_keys`, `query.py` the dense query I/O, `lex.py` the segment reader loop and BM25 across
segments, `write.py` the commit protocol, `store.py` / `s3.py` the stores and caches, `pack.py` packs and
`copy_objects`, `wire.py` the manifest and object formats, `_native.pyi` the extension's types).
