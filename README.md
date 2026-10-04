# thread

**A search index that lives in your bucket.**

thread is a vector + full-text search index stored as plain, immutable files in object storage (Cloudflare R2,
S3, or a directory on disk) and queried straight from a Cloudflare Worker, a Node process or Python. There is no
database to run and nothing to keep in RAM beyond a small routing table. On the standard 1M-vector benchmark it
matches Cloudflare Vectorize's recall, beats its tail latency and throughput, and costs 10-20x less.

```ts
const index = await thread.open({ bucket: env.BUCKET });
const { hits } = await index.search({ text: "deploy failed", vector: embedding, topK: 10, docs: true });
```

## Why you might want it

- **Your index is just files.** A bucket prefix holds everything: centroids, slabs, text segments, a manifest.
  Copy it, version it, sync it to a laptop, serve it from a CDN. Nothing is ever rewritten in place, so every
  object caches forever and every query reads a consistent snapshot.
- **It runs where you already are.** The same index is read by a Worker (wasm, inside a 128 MB isolate), by Node,
  and by Python (a native extension). The TypeScript and Python SDKs share one wire format and give identical
  answers, down to the id order.
- **It is cheap.** A query reads about 6 MB of ranges from cache and does a little arithmetic. At 1M vectors and
  1M queries a month that is about $1-2 on Workers + R2, against about $21 for Vectorize. At 10M queries it is about
  $10 against $111.
- **It is fast enough to serve from.** 95% recall@10 with p50 around 80 ms and over 2000 requests per second from
  a single Worker deployment at concurrency 300. From a laptop against a local copy, a query takes about a
  millisecond.
- **Vectors and text, one id space.** BM25 search over the same documents, fused with the vector results by rank.
  One upsert adds a vector, its text and its metadata; one delete hides all three.
- **Writes without rebuilds.** Upserts and deletes append small segments and a new manifest revision. Compaction
  folds them back in automatically, in the background, without a failed query.

## Install

```sh
npm install thread-sdk        # TypeScript, Cloudflare Workers and Node
pip install thread-sdk        # Python 3.11+
```

## Quick start

### From a Cloudflare Worker

```ts
import wasmModule from "thread-sdk/wasm/thread_bg.wasm";
import { Thread } from "thread-sdk";

const thread = await Thread.init({ wasmModule });                // once per isolate
const index = await thread.open({ bucket: env.BUCKET });         // an R2 binding; S3 works too

// vectors only
const { hits } = await index.query(embedding, { topK: 10 });     // [{ id, score, attrs }, ...]

// text, vectors, or both (fused by Reciprocal Rank Fusion), with the stored document
const r = await index.search({ text: "deploy failed", vector: embedding, topK: 10, docs: true });
// r.hits[0] = { id, score, dense?: { rank, score }, lexical?: { rank, score }, doc?: { id, meta } }

// write
await index.upsert({ ids: [1000, 1001], vectors: [embA, embB], texts: ["...", "..."], metas: ["A", "B"] });
await index.delete([1000]);
```

The full API, the stores (R2 binding, S3, memory) and a mapping from Vectorize's calls are in
[sdk/ts/README.md](sdk/ts/README.md).

### From Python

```python
from thread_sdk import FileBlobStore, S3BlobStore, open_index

store = S3BlobStore.r2(account_id, "my-bucket", access_key_id, secret_access_key)   # or FileBlobStore("/data/idx")
with open_index(store) as index:
    hits = index.query(embedding, top_k=10).hits
    r = index.search(text="deploy failed", vector=embedding, top_k=10, docs=True)
    index.upsert([1000, 1001], [emb_a, emb_b], texts=["...", "..."], metas=["A", "B"])
```

Python reads the same bucket the Worker serves from, or a local directory holding a copy of it, with an optional
disk cache. See [sdk/python/README.md](sdk/python/README.md).

### Building an index

A text-only index can be created from either SDK (`createLexicalGeneration` / `create_lexical_generation`) and
filled with upserts. Building the dense base for a large corpus is a k-means pipeline that runs inside Workers
today, driven by a script on your laptop; a 975k-vector index takes about 30 minutes. See
[docs/operations.md](docs/operations.md) and [docs/build-pipeline.md](docs/build-pipeline.md). Once a generation
exists, vectors are added with ordinary upserts from any SDK.

## How it compares

Measured on dbpedia-openai-1M (975k vectors, 1536 dims, cosine, top-10) against Cloudflare's published Vectorize
numbers for the same dataset. thread was served by one Worker reading R2 through the edge cache. Full tables and
methodology: [docs/benchmarks.md](docs/benchmarks.md).

| | Vectorize | thread (concurrency 300) | thread (concurrency 50) |
|---|---:|---:|---:|
| recall@10 | 95.4% | 95.1% | 95.1% |
| p50 latency | 31 ms | 83 ms | 71 ms |
| p95 latency | 380 ms | 176-223 ms | 116-128 ms |
| throughput | 343 RPS | 2000-2650 RPS | ~600 RPS |
| cost, 1M vectors + 1M queries/month | ~$21 | ~$1-2 | |
| cost, 1M vectors + 10M queries/month | ~$111 | ~$10 | |
| index size | managed | 5.1 GB in R2 | |
| upsert of 2000 vectors | | 3.3 s, visible within 30 s | |

Where it is a near drop-in for Vectorize: `query`, `upsert`, `delete`, `topK`, metadata returned per hit. Where it
is not: ids are u32, distance is cosine only, and there is no server-side metadata filtering yet (hits carry a
small `attrs` field for client-side filtering). The [mapping table](sdk/ts/README.md#vectorize-mapping) lists
every call.

## How it works, briefly

Vectors are clustered into a few thousand cells. Each cell is one immutable object, a **slab**, holding its
members' residuals twice: compact product-quantized codes for scanning and int8 codes for re-scoring. A query
scores the cell centroids (the only thing kept in memory), range-reads the scan prefix of the best few dozen
slabs, keeps a bounded top-k, then re-scores the best 150 candidates with a handful of multi-range reads. Text
lives in **segments** beside the slabs: BM25 postings and a document store, read in two range-request rounds per
query. Updates append delta segments and tombstone sets under a new manifest revision; compaction folds them into
new slab revisions and splits cells that grow too large.

All the arithmetic is Rust (`crates/thread-core`), compiled to wasm for Workers and Node and to a native extension
for Python. The SDKs and the Worker only do I/O. [docs/architecture.md](docs/architecture.md) has the whole
picture; [docs/query-path.md](docs/query-path.md), [docs/update-path.md](docs/update-path.md) and
[docs/lexical.md](docs/lexical.md) have the details.

## What is in the repo

| path | what |
|---|---|
| `crates/thread-core` | the engine: formats, quantization, k-means, routing, scan, BM25, fusion |
| `crates/thread-wasm`, `crates/thread-py` | wasm-bindgen and PyO3 bindings over the engine |
| `crates/thread-cli` | native CLI for lexical segments (`thread seg build\|inspect\|search\|merge`, `thread tokenize`) |
| `sdk/ts`, `sdk/python` | the `thread-sdk` packages (npm, PyPI) |
| `examples/cloudflare-worker` | the reference Worker: queries, writes, and the whole build pipeline in build slots |
| `bench` | build driver, benchmarks and dataset prep for the deployed Worker |
| `docs` | design and operations docs, indexed in [docs/README.md](docs/README.md) |

## Status

Version 0.1.0. The query path, the lexical index and hybrid search are tested end to end from both SDKs and the
Worker, including a cross-check that Python and TypeScript return the same results from the same bucket. The SDK
write path is marked experimental: it writes the same objects the Worker does, but the two have not yet been
cross-verified against each other in production. Compaction and GC of slabs run in the Worker only.

## Development

```sh
cargo test --workspace
cd sdk/ts && npm run typecheck && npm test
cd sdk/python && maturin develop --uv && pytest
```

The full list of checks, the wasm build and the deploy steps are in [docs/operations.md](docs/operations.md).

MIT licensed.
