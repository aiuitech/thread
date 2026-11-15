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
    # hybrid: r.hits = [SearchHit(id, score, dense=Rank(rank, score), lexical=Rank(...), doc=DocRecord(id, meta, text, attrs)), ...]
    f = index.search(text="deploy failed", vector=embedding, top_k=10, filter={"tenant": 42, "lang": {"$in": ["en", "fr"]}})
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
whole objects are cached and ranges are served from them (`current.json` is never cached). `MemoryCache` is an LRU
in process memory; `DiskCache` keeps files in a directory shared by every process that uses it and reads only the
requested ranges. What a cache miss reads is the **miss policy** (`miss_policy`):

| `miss_policy` | a slab-prefix miss reads | phase 2 for that slab | right for |
|---|---|---|---|
| `"range+warm"` (default) | the prefix now (~200 KB on g4); the whole slab into the cache in the background | one range read per record run | an in-process cache: the first query does not wait for the whole slabs (~57 MB on g4), later queries hit |
| `"whole"` | the whole slab once (concurrent misses share one read), cached | free: the winners' records are in memory | a cache many processes share |
| `"range"` | exactly the prefix, nothing cached | one range read per record run | no caching |

Background warms run on a small pool of their own (`warm_concurrency`, default 2, only 1 while the query's own
reads are in flight; at most 1024 queued; deduplicated); `index.store.drain()` waits for them,
`index.store.warm_stats` counts them. Before this, a cold query read every probed slab whole. Reads without `fill`
(rerank records, postings) never fetch whole objects, and a `PackedBlobStore` turns every read, warms included,
into a range of the pack.

**Disk cache.** `DiskCache(dir, max_bytes=4 << 30)` keeps whole objects as files in the bucket layout
(`{dir}/idx/g4/slab/00123`), the layout the TypeScript SDK's `DiskCache` uses (they can share a directory),
evicting least recently used objects beyond `max_bytes` (order in memory, and in the files' mtimes across
restarts). A restart is then not a cold start. A `DiskCache` is also a `FileBlobStore`, so warming a laptop from a
bucket is a copy: `copy_objects(s3, disk, snapshot_keys(index.revision))`.

**Connections.** `S3BlobStore` uses `http.client` with a pool of keep-alive connections shared by every thread (at
most `max_connections`, default `inflight` = 32; a thread takes an idle connection or opens one, and waits when all
are busy; a connection the server closed is reopened and the request retried), so there is no third-party HTTP
client and no extra dependency. Before, each I/O thread kept its own connection, so a cold process opened one per
thread (33). `open_index` warms `inflight` of them with HEADs of `idx/current.json` while it loads the index
(`warmup`), so the first query does not pay the TLS handshakes. Signing costs one HMAC per request: the derived SigV4 key is cached per (date, region, service)
(`SigningKeyCache`; ~5 us instead of ~9 us per request). `S3BlobStore(..., inflight=32, coalesce_bytes=128 << 10,
chunk_bytes=8 << 20, chunk_concurrency=4, max_connections=inflight, warmup=0)`: rerank records are read as runs of
up to 128 KB (half the requests of a cold query for ~45% more bytes), and a whole-object GET larger than
`chunk_bytes` becomes parallel range GETs (one S3 stream is bandwidth-bound; the routing table is 25 MB on g4); the
rest starts as soon as the first response's headers give the size, so the split costs no extra round trip.

## API

### `open_index(store, **options) -> ThreadIndex`

`store` is a `BlobStore`, or a path: a directory holding the layout (`FileBlobStore`) or a pack file
(`PackedBlobStore`); `store_at(path)` is the function behind that.

| option | default | meaning |
|---|---|---|
| `prefix` | `""` | key prefix in front of `idx/` (several indexes in one bucket) |
| `gen`, `rev` | `idx/current.json` | read a specific generation / revision; with both, the index is pinned and never refreshes |
| `cache` | `MemoryCache()` for remote stores, none for local ones | a `MemoryCache` / `DiskCache`; `True` = default cache, `False` = none |
| `refresh_interval` | 30.0 s | how often a query checks `current.json` for a new revision, in the background (0 = only `refresh()`) |
| `delta_cache_bytes` | 40 MB | delta segments kept in memory across queries |
| `lex_transient_bytes` | 8 MB | transient lexical bytes (buckets, posting lists, texts) each segment reader may hold |
| `io_workers` | 32 | threads for reads, shared by concurrent queries |
| `routing` | `"f32"` | routing table to load: `"f32"` (`fine.f32`, 25 MB at 4096 x 1536), `"f16"` (half), `"i8"` (a quarter) or `"auto"` (the smallest the revision has); decoded to f32 on load, a revision without the table falls back to f32 |
| `cold_routing` | `"off"` | `"tree"`: the open waits only for the centroid tree and the codebook; until the full table has loaded (on a background thread) queries route over the tree's best `cold_beam` groups, range-reading just those groups' means |
| `embed` | - | how texts are embedded when the generation names its model (`manifest.embedding`): `{"account_id", "api_token"}` (Workers AI REST), `{"api_key"}` (OpenAI-compatible; also `headers`, `post`), or a function `(texts, kind) -> vectors`; without it `search` needs a `vector` and `upsert` needs `vectors` or `embed` (see [docs/embedding.md](../../docs/embedding.md)) |
| `cold_beam` | `defaults.beam`, else 16 | groups ranked per cold-routed query |
| `base_pack` | True | GET the revision's base pack (one object with everything the open reads) and, without one, start every read of the open as soon as the manifest is in; False: one read after another |
| `job_workers` | 4-16 (CPUs) | threads for `query_many`, the lexical half of a hybrid search and background refreshes |
| `miss_policy` | the cache's: `"range+warm"` for `MemoryCache` / `DiskCache` | what a slab-prefix cache miss reads (see "Pointing at the index") |
| `warm_concurrency` | 2 | `"range+warm"`: background whole-slab fetches at once |
| `warmup` | on for remote stores | open connections while opening: `True` = the store's `inflight` (32 for S3), an int, `False` |
| `cold_policy` | None | `{"nprobe", "rerank", "threshold"}` for queries whose probed slabs are mostly not cached (see `query`) |
| `prefetch_bytes` | 64 MB | slab prefixes / delta and lexical segments a revision switch prefetches into the cache |
| `warm_pack` | - | key of a warm pack (hot slab prefixes in one object): fetched with one GET on a background thread at open; its prefixes go into the `MemoryCache` (else a prefix layer in front of the store) and are served from memory (`index.warm_ready`, `index.warm_stats`). Opt-in; measured no gain on our workload (docs/cold-start.md, "Warm packs") |

`gen` without `rev` reads the generation's head (the current rev when `current.json` names that generation, else
the newest `manifest.*.json` in the store). `ThreadIndex` is a context manager; `close()` releases its threads.

With a base pack (`idx/{gen}/base.{rev}.threadpack`, written by publish) a cold open is two round trips:
`current.json`, then the pack (the manifest is read alongside, in case there is none, unless `current.json` carries
an inline copy, which the SDK writers and the Worker's publish add when it is at most 16 KB); without one, three
(two with an inline manifest: the base files start right after `current.json`). With
`routing="i8"` (or `"auto"`, the table a base pack holds by default) that is `current.json` + one 8.8 MB GET on g4;
`cold_routing="tree"` reads the 2.5 MB cold base pack and ~1 MB of group means with the first query. The trade-offs
and measured numbers: [docs/query-path.md](../../docs/query-path.md#cold-opens-compact-routing-cold-tree-routing-base-packs).

`index.open_stats` is the cold-start timeline of `open_index`: `{total, prefetch, current, manifest, slabs, base, pack?}`, each
step a phase stat (`current`: current.json; `manifest`; `slabs`: slabs table + tombstone sets; `base`: routing table
and codebook, plus `fetch` / `build` ms and `files: {coarse, fine, pq}` bytes; `pack`, for a `PackedBlobStore`, its
own head + tail + directory reads, `PackedBlobStore.open_stats`, not part of `total`; `prefetch`: how the open's
objects were fetched up front, `source` `"base-pack"` / `"index-pack"` / `"files"` / `"none"` and `key`, with the
pack read's work; objects read early are counted in the step that uses them). A "read" is a request that
reached the origin store; with a cache in front, `hits` / `misses` count cache lookups. Same fields as the
TypeScript SDK's `index.openStats` (which adds `wasmInit`). `bench/cold/` measures these against S3.

Properties: `manifest` (the stored dict, camelCase keys), `gen`, `rev`, `dim`, `cells` (`k1 * k2`), `format`
(1 int8 only, 2 sq4 + sq8, 3 pq + sq8), `defaults` (`{nprobe, rerank, pc}`), `has_dense`, `lex_config`
(`ResolvedLexConfig` or None), `segments`, `revision` (the parsed `RevState`), `routing_state` (`"full"` once the
routing table is loaded, `"cold"` while queries route cold); `when_full_routing()` waits for the full table.

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
| `inflight` | the store's: 32 for `S3BlobStore`, else 12 | reads in flight per query (remote stores) |
| `speculate` | True | read the running top-T's rerank records while the last 6 slabs are still scanning (remote stores) |
| `beam` | `cold_beam` | cold routing only: tree groups whose cells are ranked (until the full routing table is in) |
| `filter` | None | an attribute filter (see [Filtering](#filtering)): field conditions, or the compiled words |
| `max_nprobe` | 4 x nprobe | with a filter: how many cells a selective filter may scan in all, in routing order, until `max(top_k, rerank)` candidates match (`nprobe` = no widening) |
| `coalesce_bytes` | the store's: 128 KB for `S3BlobStore`, else 4096 | phase 2: a slab's wanted rerank records are merged into one range while it stays within this many bytes |
| `span_bytes` | the store's, else 0 (off) | phase 2: read a slab's records as ONE range from the first to the last wanted one when that span is at most this |
| `cold_policy` | `open_index(cold_policy=)` | `{"nprobe", "rerank", "threshold"}`: when more than `threshold` (default 0.5) of the probed slabs are not cached, scan only the best `nprobe` and rerank `rerank` (each capped at the query's value); `False` disables it |

```python
@dataclass
class QueryResult:
    hits: list[Hit]          # Hit(id, score, attrs, source: "slab" | "delta", payload: bytes | None)
    params: dict             # {top_k, nprobe, rerank, pc, source: {nprobe, rerank, pc}} ("query" | "manifest" | "legacy");
                             # with a filter also filter: True and the max_nprobe in effect
    stats: dict              # gen, rev, format, routing ("flat" | "tree" | "cold"), routing_table ("f32" | "f16" | "i8"),
                             # cold_routing? ({beam, groups, cells, reads, bytes, loaded_groups, loaded_cells}),
                             # cells (scanned; beyond nprobe when a filter widened), filter_dropped,
                             # candidates, scanned, scan_bytes, slab_bytes,
                             # rerank, rerank_candidates, rerank_slabs, rerank_reads, rerank_bytes, rerank_local,
                             # spec_records, spec_used, deltas, delta_bytes, delta_scanned, delta_hits, delta_dropped,
                             # delta_top, tombstones, tomb_sets, tomb_dropped,
                             # io: {reads, hits, misses, r206, r200, bytes}, timing: {total, route, phase1, phase2, payload} (ms),
                             # timeline: {route, phase1, phase2, payload, base?} (phase stats, below)
                             # whole_slabs (slabs whose whole object was in memory), inflight, coalesce_bytes,
                             # span_bytes, cold: {misses, probed, applied} (with a cold policy); io also has warms
```

`stats["timeline"]` splits the query into phases, each a phase stat `{ms, reads, hits, misses, r206, r200, bytes}`:
its wall time and the store work that completed during it (trace snapshots at the phase boundaries). `base` is
present when this query (re)loaded the routing table (a switch to another generation). Same fields as the
TypeScript SDK's `stats.timeline`.

`stats["trace"]` is the query's trace envelope, the same JSON (camelCase keys) as the TypeScript SDK's and the
Worker's: spans (the phases above), store work, counters (`pushed`: entries that entered a scan's heap) and the CPU
of each kernel (`route`, `scan`, `deltaScan`, `merge`, `rerank`, timed inside the extension with the GIL released).
`trace_events=True` adds one event per store read. `search` and `lexical_search` carry one in `stats["trace"]`,
`list_keys` pages, `upsert` and `delete` results in `.trace`, `index.open_trace` is the open's, and
`open_index(on_trace=fn, trace_sample=0.1)` hands a sampled share of them (`get`'s too) to `fn`. See
[docs/observability.md](../../docs/observability.md).

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
| `dense` | - | `query` options of the dense list (`{"nprobe": 8, "max_nprobe": 32, ...}`) |
| `docs` | False | `True`: resolve each hit through the `docs` store; `"text"`: the stored text too |
| `filter` | None | an attribute filter applied to both lists (see [Filtering](#filtering)) |

`SearchResult(hits, mode, params, dense, lexical, stats)`: `mode` is `"dense"`, `"lexical"` or `"hybrid"`; each
`SearchHit(id, score, dense, lexical, doc)` carries its rank and score in each list (`Rank(rank, score)`) and, with
`docs`, its `DocRecord(id, meta, text, attrs)` (None when the id has no docs entry). `dense` / `lexical` hold the
full list results (`QueryResult` / `LexSearchResult(hits, tokens, found, stats)`).

`index.lexical_search(text, k, filter=None)` is the BM25 list alone, `index.tokenize(text)` the query's tokens,
`index.docs(ids, text=False)` the document store (`{id: DocRecord}`, newest live copy of each id).

### Filtering

Every document carries a 32-bit attribute word, stored with its vector (inside the scan prefix every query reads)
and in its `docs` entry; a filter is a condition on that word, applied inside the scan before an entry can take a
candidate slot. A schema names fields inside the word (design and knobs: [docs/filtering.md](../../docs/filtering.md)):

```python
index.configure_attrs(tenant={"bits": 12}, lang={"values": ["en", "fr", "de"]}, public="bool", year={"bits": 7, "min": 1970})
index.upsert(ids, vectors, texts=texts, metas=metas, attrs=[{"tenant": 42, "lang": "en", "public": True, "year": 2024}, {"tenant": 7}])
r = index.query(embedding, top_k=10, filter={"tenant": 42, "lang": {"$in": ["en", "fr"]}, "year": {"$gte": 2020}})
h = index.search(text="deploy failed", vector=embedding, filter={"public": True}, docs=True)
index.unpack_attrs(r.hits[0].attrs)      # {"tenant": 42, "lang": "en", "public": True, "year": 2024}
```

| in a filter | fields | meaning |
|---|---|---|
| `field: value`, `{"$eq"}`, `{"$ne"}` | all | equal / not equal |
| `{"$in": [...]}`, `{"$nin": [...]}` | all | in / not in a list |
| `{"$lt", "$lte", "$gt", "$gte"}` | number | a range (combined into one test) |
| `{"$any", "$all", "$none"}` (a bit mask) | number | some / every / no bit of the mask set |

Conditions on different fields are ANDed; there is no `$or`. Field specs: `{"bits": n}` (an unsigned integer, with
an optional `"min"`), `{"values": [...]}` (an enum) or `"bool"`; the fields are laid out in order and must fit 32
bits. A later `configure_attrs` may add fields or enum values but never changes an existing field's layout. Without
a schema the one field is `attrs`, the raw word (`filter={"attrs": {"$any": 1}}`); `attrs` in an upsert may always
be given as raw ints. `index.compile_filter(expr)` returns the compiled words (an `array('I')`), the same words the
TypeScript SDK compiles.

What it costs: the dense scan reads nothing extra and tests the word only for candidates that would have entered
the top-T. A selective filter makes the query scan more cells in routing order, in batches, until
`max(top_k, rerank)` candidates match or `max_nprobe` cells were scanned (default 4 x nprobe): `stats["cells"]`
lists what was scanned. The BM25 list is filtered after scoring: it is taken deeper (4 x k, then 4x more up to
65 536) and each candidate's word is resolved through the `docs` store.

### `index.refresh() -> bool`

Re-reads `current.json` (or the generation's head) and switches to the new revision if it changed. Before switching
it loads the new revision's manifest, slabs table and tombstones, its routing table when the base files changed and
its new delta segments; then it warms the rest of what the revision adds (`snapshot_keys(new)` minus
`snapshot_keys(old)`, slabs as scan prefixes) in the background within `prefetch_bytes`. The periodic check
(`refresh_interval`) runs in the background on the job pool: the query that notices a new revision answers from the
current one instead of waiting for the switch (~130 ms on g4 through a 25 ms-RTT S3). New data is visible within
`refresh_interval` (plus the load), at once after `refresh()`, or at once on the handle that wrote it.

`index.warm(keys=None, budget=None)` reads `keys` (default: every object of the current revision) through the
store's cache: slabs as a `fill` read of their scan prefix, the rest whole, skipping cached keys, slab prefixes and
segments within `budget`. Returns `{objects, bytes, skipped}`.

### Write path

```python
index.configure_lexical(text=True, lengths="exact", bucket_terms=256)    # once: manifest.lex (a revision)
index.configure_attrs(tenant={"bits": 12}, lang={"values": ["en", "fr"]})          # once: manifest.attrs (a revision)
r = index.upsert([1000, 1001], [emb_a, emb_b], texts=["...", "..."], metas=["Title A", "Title B"], attrs=[{"tenant": 5, "lang": "en"}, 6])
# UpsertResult(gen, rev, seq, count, cells, distinct_cells, keys, bytes, conflicts, cas_conflicts, current_updated, ms)
index.upsert([1002], texts=["text-only document"], metas=["C"])        # no vector: found by text alone
index.upsert([1003, 1004], [emb_d, None], texts=["...", "..."])          # a None row: that document has no vector (one revision)
index.delete([1000])                    # hides the vector, the text and the doc
k = index.upsert(keys=["s3://b/a.bin", "s3://b/c.bin"], metas=["A", "C"], texts=["host:cdn-1", "host:cdn-2"])
# by key (needs manifest.lex): known keys keep their id, new keys are allocated from manifest.nextId; k.ids lists them
index.get(["s3://b/a.bin"], text=True)  # {key: KeyRecord(id, meta, text, attrs, key)}
index.resolve_keys(["s3://b/a.bin"])    # {key: id}
page = index.list_keys("s3://b/", limit=100)   # KeyPage(entries=[(key, id)], more, next); continue with after=page.next
index.delete_keys(["s3://b/c.bin"])     # DeleteResult with .missing
# long texts: chunked into windows of `words` words (default 200) sharing `overlap` (default 0), one document per chunk keyed `k#i`
# with the parent's meta/attrs; `embed` is called once with the chunk texts and returns their vectors (r.chunks = {documents, count, stale})
index.upsert(keys=["docs/a.md"], texts=[long_text], metas=["A"], chunk={"words": 200, "overlap": 20}, embed=embed_all)
chunk_text(long_text, words=200, overlap=20)   # [Chunk(index, text, start, end, words), ...], the same windows
index.upsert(keys=["s3://b/a.bin"], vectors=[emb], scope="dense")       # a vector for an existing key, text/meta untouched
index.upsert(keys=["s3://b/a.bin"], texts=["host:cdn-2"], scope="lexical")   # new text, the vector untouched
index.compact_segments()                # merge the lexical/docs segments into one base segment
index.compact_segments(max_input_bytes=4 << 20)   # or one contiguous run that fits 4 MB (memory-limited writers)
create_lexical_generation(store, "docs1", {"text": True})               # a text-only generation (no dense index)
# a dense generation from nothing, for an embedding model (one empty cell; grows by the Worker's compaction and cell splits)
create_generation(store, "docs2", embedding={"provider": "openai", "model": "text-embedding-3-small", "dim": 1536}, lex={"text": True})
ix = open_index(store, embed={"api_key": api_key})   # ix.can_embed; ix.embedding; ix.embed(texts, "query" | "document")
ix.upsert(keys=["docs/a.md"], texts=[long_text], chunk=True)   # no vectors: embedded as documents (embed=False = text-only)
ix.search(text="deploy failed", top_k=10, docs=True)           # embedded as a query + BM25, fused (embed=False = BM25 only)
```

`vectors` may be `n * dim` floats, one row per id, an `(n, dim)` numpy array or raw f32 bytes. Up to 4096 ids per
call. `replace=False` skips the tombstone set when the ids are known to be new. Keys (non-empty strings of at most
4096 bytes, unique per batch) go to the segment's `keys` section and resolve newest segment first; see
[docs/keys.md](../../docs/keys.md). `chunk` (`True`, a dict or `ChunkOptions`) splits each text into flat word
windows before writing (needs `keys` and `texts`; vectors come from `embed`; chunk keys the parent no longer
produces are tombstoned in the same revision); see [docs/chunking.md](../../docs/chunking.md). `configure_lexical` takes the
wire-format dict (`{"bucketTerms": 64}`) or snake_case keyword arguments; the tokenizer of a generation with
segments is frozen. `create_generation` publishes rev 0 with one empty cell for `embedding` (or a bare `dim`),
`lex` (default `{}`; `False` for ids only), `attrs`, a `profile` (`edge` splits cells at 3000 entries, `local` at
1000), `defaults` and the automatic `compaction` policy (`False` for none); what that generation lacks compared
with a k-means build is in [docs/embedding.md](../../docs/embedding.md).

Commits follow docs/update-path.md, as the TypeScript SDK does: a delta segment and/or a lexical segment and a
tombstone set, then a create-only put of `manifest.{rev+1}.json` (the writer lock; a lost race rebuilds on the new
head) and a compare-and-swap of `idx/current.json`. Needs a store with conditional puts (`S3BlobStore`,
`FileBlobStore`, `MemoryBlobStore`); revisions written from Python record `change.writer: "thread-sdk-python"`.
Each `upsert` or `delete` is one revision with its own delta, segment and tombstone set, so batch at least 64
documents per call, or what arrived in the last second ([docs/update-path.md](../../docs/update-path.md#small-writes)).
Compaction of slabs and GC stay with the Worker and the TypeScript SDK; the sets the TypeScript compaction merges
read like any others ([docs/update-path.md](../../docs/update-path.md#merging-tombstone-sets)).

### `BlobStore`

```python
class BlobStore:
    supports_put = supports_list = supports_ranges = remote = False
    def get(self, key, range: BlobRange | None = None, *, trace=None, fill=False) -> bytes | memoryview | None: ...
    def get_ranges(self, key, ranges, *, trace=None, fill=False) -> list | None: ...   # None: read ranges one by one
    def get_tail(self, key, length, *, trace=None) -> tuple[bytes, BlobHead] | None: ...   # bytes=-N; NotImplementedError: head + range
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

`pack_index` writes `idx/current.json` (pointing at the packed revision), then the objects a cold open reads
(`base_keys`: manifest, slabs table, tombstone sets, coarse, codebook, tree, the smallest routing table), then the
rest of `snapshot_keys` in key order, and records that leading range as `base: [offset, length]`: `open_index` on a
`PackedBlobStore` loads it with one range read. The two SDKs produce identical files for the same revision (the
interop test checks). `thread_sdk.pack.build_base_pack(store, rev_state, "base" | "cold", routing)` builds a
revision's base pack (`idx/{gen}/base.{rev}.threadpack`, what publish writes), byte-identical to the TypeScript one. A pack is read-only:
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
attrs, payloads, docs, tokens, BM25 scores, fused scores, compiled filters and the cells a filtered query scans
agree exactly. Cosine scores agree to about 1e-7: the
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

Releases: a `v<version>` tag on the repo builds abi3 wheels and an sdist and publishes them to PyPI through trusted
publishing (see [docs/operations.md](../../docs/operations.md#releasing)); bump `pyproject.toml` together with the
TypeScript and Cargo versions, and add a [CHANGELOG](../../CHANGELOG.md) entry first.

Layout: `crates/thread-py` (the PyO3 bindings: dense kernels and the per-query `QueryRun` in `lib.rs`/`run.rs`, the
lexical reader and builders in `lex.rs`), `thread_sdk/` (`index.py` the `ThreadIndex`, `revision.py` resolving and loading a
revision and `snapshot_keys`, `query.py` the dense query I/O, `lex.py` the segment reader loop and BM25 across
segments, `write.py` the commit protocol, `store.py` / `s3.py` the stores and caches, `pack.py` packs and
`copy_objects`, `wire.py` the manifest and object formats, `_native.pyi` the extension's types).
