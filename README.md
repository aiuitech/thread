# thread

**A search index that lives in your bucket.**

thread is a vector + full-text search index stored as plain, immutable files in object storage (Cloudflare R2,
S3, or a directory on disk) and queried straight from a Cloudflare Worker, a Node process or Python. There is no
database to run and nothing to keep in RAM beyond a small routing table. On the standard 1M-vector benchmark it
comes within 0.3 points of Cloudflare Vectorize's recall, halves its p95 latency at 7-8x its throughput (its median
latency is about 3x Vectorize's), and costs a few percent as much.

```ts
const index = await thread.open({ bucket: env.BUCKET });
const { hits } = await index.search({ text: "deploy failed", vector: embedding, topK: 10, docs: true });
```

## Why you might want it

- **Your index is just files.** A bucket prefix holds everything: centroids, slabs, text segments, a manifest.
  Copy it, version it, sync it to a laptop, serve it from a CDN. Nothing is ever rewritten in place, so every
  object caches forever and every query reads a consistent snapshot. Or pack one revision into a single file and
  query that file anywhere, in place: `open_index("idx.threadpack")`.
- **It runs where you already are.** The same index is read by a Worker (wasm, inside a 128 MB isolate), by Node,
  and by Python (a native extension). The TypeScript and Python SDKs share one wire format and give identical
  answers, down to the id order.
- **It is cheap.** A query reads about 6 MB of ranges from cache and spends about 33 ms of CPU. At 1M vectors and
  1M queries a month that is about $0.10 on Workers + R2 beyond the Workers Paid plan both need, against about $31
  for Vectorize. At 10M queries it is about $6 against $169. Both figures assume a warm edge cache: every cache miss is an
  R2 read ([costs](docs/benchmarks.md#cost)).
- **It is fast enough to serve from.** 95% recall@10 with p50 around 90 ms and about 2500 requests per second from
  a single Worker deployment at concurrency 300 (p50 around 75 ms at concurrency 50). From a laptop against a local
  copy, a query takes about a millisecond.
- **Vectors and text, one id space.** BM25 search over the same documents, fused with the vector results by rank.
  One upsert adds a vector, its text and its metadata; one delete hides all three.
- **Lookups by key.** Documents can be named by a unique string key (an address, a path, a primary key): `get(key)`
  is two range reads, keyed upserts need no ids, and keys list by prefix. A text-only generation with keys, a JSON
  meta and facet tokens is a small key-value index with equality filters ([docs/keys.md](docs/keys.md)).
- **Filters that cost nothing extra.** Every document carries a 32-bit attribute word (named fields: a tenant, a
  language, flags, a date) that lives inside the bytes a query reads anyway. A filter is applied inside the scan,
  before candidates are chosen, and a selective filter widens its probes instead of returning less.
- **Writes without rebuilds.** Upserts and deletes append small segments and a new manifest revision. Compaction
  folds them back in automatically, in the background, without a failed query.

## Install

```sh
npm install @aiuitech/thread-sdk        # TypeScript, Cloudflare Workers and Node
pip install thread-sdk        # Python 3.11+
```

## Quick start

### From a Cloudflare Worker

```ts
import wasmModule from "@aiuitech/thread-sdk/wasm/thread_bg.wasm";
import { Thread } from "@aiuitech/thread-sdk";

const thread = await Thread.init({ wasmModule });                // once per isolate
const index = await thread.open({ bucket: env.BUCKET });         // an R2 binding; S3 works too

// vectors only
const { hits } = await index.query(embedding, { topK: 10 });     // [{ id, score, attrs }, ...]

// text, vectors, or both (fused by Reciprocal Rank Fusion), with the stored document
const r = await index.search({ text: "deploy failed", vector: embedding, topK: 10, docs: true });
// r.hits[0] = { id, score, dense?: { rank, score }, lexical?: { rank, score }, doc?: { id, meta, attrs } }

// filter by attributes (named fields packed into each document's 32-bit word; see docs/filtering.md)
await index.configureAttrs({ tenant: { bits: 12 }, lang: { values: ["en", "fr", "de"] }, public: "bool" });
const f = await index.search({ text: "deploy failed", vector: embedding, topK: 10, filter: { tenant: 42, lang: { $in: ["en", "fr"] } } });

// write
await index.upsert({ ids: [1000, 1001], vectors: [embA, embB], texts: ["...", "..."], metas: ["A", "B"], attrs: [{ tenant: 42, lang: "en" }, { tenant: 7 }] });
await index.delete([1000]);

// by key: ids are allocated for new keys and reused for known ones (a re-put replaces the document)
await index.upsert({ keys: ["s3://b/a.bin"], metas: [JSON.stringify({ host: "cdn-1", size: 4096 })], texts: ["host:cdn-1"] });
const { records } = await index.get(["s3://b/a.bin"]);                 // Map<key, { key, id, meta, attrs }>
const page = await index.listKeys({ prefix: "s3://b/", limit: 100 });  // { entries: [{ key, id }], more, next }
await index.upsert({ keys: ["s3://b/a.bin"], vectors: [emb], scope: "dense" });   // one side only: the vector, text and meta stay

// long texts: one document per window of 200 words (overlap 0), keyed "docs/a.md#0", "docs/a.md#1", ...; vectors from your embedder
await index.upsert({ keys: ["docs/a.md"], texts: [longText], metas: ["A"], chunk: { words: 200, overlap: 20 }, embed: async (texts) => embedAll(texts) });

// an index for an embedding model: created from nothing, filled from texts, searched by text (docs/embedding.md)
await thread.createGeneration({ store: new R2BlobStore(env.BUCKET), gen: "docs", embedding: { provider: "workers-ai", model: "@cf/baai/bge-base-en-v1.5", dim: 768, options: { pooling: "cls" } } });
const ix = await thread.open({ bucket: env.BUCKET, embed: { ai: env.AI } });      // the index calls the model; it never runs one
await ix.upsert({ keys: ["docs/a.md"], texts: [longText], chunk: true });           // embedded as documents, one per chunk
const hits = await ix.search({ text: "deploy failed", topK: 10, docs: true });     // embedded as a query + BM25, fused
```

The full API, the stores (R2 binding, S3, memory) and a mapping from Vectorize's calls are in
[sdk/ts/README.md](sdk/ts/README.md).

### From Python

```python
from thread_sdk import FileBlobStore, S3BlobStore, open_index

store = S3BlobStore.r2(account_id, "my-bucket", access_key_id, secret_access_key)   # or FileBlobStore("/data/idx")
with open_index(store) as index:
    hits = index.query(embedding, top_k=10).hits
    index.configure_attrs(tenant={"bits": 12}, lang={"values": ["en", "fr", "de"]}, public="bool")   # once
    r = index.search(text="deploy failed", vector=embedding, top_k=10, docs=True, filter={"tenant": 42})
    index.upsert([1000, 1001], [emb_a, emb_b], texts=["...", "..."], metas=["A", "B"], attrs=[{"tenant": 42}, {"tenant": 7}])
    index.upsert(keys=["s3://b/a.bin"], metas=['{"host": "cdn-1"}'])          # by key; r.ids holds the allocated ids
    index.get(["s3://b/a.bin"])                                                # {key: KeyRecord(id, meta, text, attrs, key)}
    index.upsert(keys=["docs/a.md"], texts=[long_text], chunk={"words": 200, "overlap": 20}, embed=embed_all)   # one doc per chunk, keyed "docs/a.md#i"

create_generation(store, "docs", embedding={"provider": "openai", "model": "text-embedding-3-small", "dim": 1536})   # an index for a model, from nothing
with open_index(store, embed={"api_key": os.environ["OPENAI_API_KEY"]}) as index:
    index.upsert(keys=["docs/a.md"], texts=[long_text], chunk=True)        # embedded as documents through the model
    index.search(text="deploy failed", top_k=10, docs=True)                 # embedded as a query + BM25, fused
```

Python reads the same bucket the Worker serves from, a local directory holding a copy of it, or a pack
(`open_index("/data/idx.threadpack")`), with an optional disk cache. See [sdk/python/README.md](sdk/python/README.md)
and [docs/portability.md](docs/portability.md) for moving an index between all of these.

### Building an index

An index for an embedding model is created from nothing with `createGeneration` / `create_generation` / the
Worker's `POST /create` (one empty cell, `manifest.embedding` naming the model) and filled with upserts: vectors go
to delta segments, the Worker's automatic compaction folds them into cells and splits cells as they pass 3000
entries, so the cell count grows with the data. Texts are embedded through the model on upsert and search (Workers
AI, any OpenAI-compatible endpoint, or your own function); thread never runs a model itself. See
[docs/embedding.md](docs/embedding.md). A text-only index is `createLexicalGeneration` / `create_lexical_generation`.
Building the dense base for a large pre-embedded corpus with a global k-means and a PQ codebook is a pipeline that
runs inside Workers, driven by a script on your laptop; a 975k-vector index takes about 30 minutes. See
[docs/operations.md](docs/operations.md) and [docs/build-pipeline.md](docs/build-pipeline.md). Once a generation
exists, vectors are added with ordinary upserts from any SDK.

## How it compares

Measured on dbpedia-openai-1M (975k vectors, 1536 dims, cosine, top-10) against Cloudflare's published Vectorize
numbers for the same dataset. thread was served by one Worker (main `610501f`, g4 rev 179) reading R2 through a
warm edge cache, measured 2026-10-05 from near Montreal, three runs each (recall over all 5000 test queries at
concurrency 300, over the first 1000 at 50). Full tables, the runs, a cold-cache figure and methodology:
[docs/benchmarks.md](docs/benchmarks.md).

| | Vectorize | thread (concurrency 300) | thread (concurrency 50) |
|---|---:|---:|---:|
| recall@10 | 95.4% | 95.1% | 94.9% |
| p50 latency | 31 ms | 88-90 ms | 69-78 ms |
| p95 latency | 380 ms | 176-193 ms | 120-133 ms |
| throughput | 343 RPS | 2495-2785 RPS | 578-608 RPS |
| cost, 1M vectors + 1M queries/month | ~$31 | ~$0.10 | |
| cost, 1M vectors + 10M queries/month | ~$169 | ~$6 | |
| index size | managed | 5.1 GB in R2 | |
| upsert of 2000 vectors | | 3.3 s, visible within 30 s | |
| metadata filtering | indexed properties | fields packed into a 32-bit word, tested inside the scan; selective filters widen the probes | |

Where it is a near drop-in for Vectorize: `query`, `upsert`, `delete`, `topK`, metadata returned per hit, and
metadata filters (`$eq`, `$ne`, `$in`, `$nin`, `$lt`, `$lte`, `$gt`, `$gte`) over fields packed into a 32-bit
attribute word. Where it is not: ids are u32, distance is cosine only, and filterable metadata is bounded by that
word (a tenant, a language, flags, a date: see [docs/filtering.md](docs/filtering.md)). The
[mapping table](sdk/ts/README.md#vectorize-mapping) lists every call.

## How it works, briefly

Vectors are clustered into a few thousand cells. Each cell is one immutable object, a **slab**, holding its
members' residuals twice: compact product-quantized codes for scanning and int8 codes for re-scoring. A query
scores the cell centroids (the only thing kept in memory), range-reads the scan prefix of the best few dozen
slabs, keeps a bounded top-k, then re-scores the best 150 candidates with a handful of multi-range reads. Each
entry's attribute word sits in that prefix, so a filter is tested before an entry can take a candidate slot, and
a selective filter scans more cells in routing order until it has enough matches. Text lives in **segments**
beside the slabs: BM25 postings and a document store, read in two range-request rounds per query. Updates append
delta segments and tombstone sets under a new manifest revision; compaction folds them into new slab revisions
and splits cells that grow too large.

All the arithmetic is Rust (`crates/thread-core`), compiled to wasm for Workers and Node and to a native extension
for Python. The SDKs and the Worker only do I/O. [docs/architecture.md](docs/architecture.md) has the whole
picture; [docs/query-path.md](docs/query-path.md), [docs/update-path.md](docs/update-path.md) and
[docs/lexical.md](docs/lexical.md) have the details; [docs/filtering.md](docs/filtering.md) covers attribute
filters and [docs/keys.md](docs/keys.md) string keys.

## What is in the repo

| path | what |
|---|---|
| `crates/thread-core` | the engine: formats, quantization, k-means, routing, scan, attribute filters, BM25, fusion |
| `crates/thread-wasm`, `crates/thread-py` | wasm-bindgen and PyO3 bindings over the engine |
| `crates/thread-cli` | native CLI for lexical segments (`thread seg build\|inspect\|search\|merge\|keys`, `thread tokenize`, `thread chunk`) |
| `sdk/ts`, `sdk/python` | the `thread-sdk` packages (npm, PyPI) |
| `examples/cloudflare-worker` | the reference Worker: queries, writes, and the whole build pipeline in build slots |
| `bench` | build driver, benchmarks and dataset prep for the deployed Worker |
| `docs` | design and operations docs, indexed in [docs/README.md](docs/README.md) |

## Status

Version 0.1.0. The query path, the lexical index, hybrid search, attribute filtering and packs are tested end to
end from both SDKs and the Worker, including a cross-check that Python and TypeScript return the same results
(ids, scores, documents, compiled filters and the cells a filtered query scans) from the same bucket or pack. The
SDK write path is marked experimental: it writes the same objects the Worker does, but the two have not yet been
cross-verified against each other in production. Compaction and GC of slabs run in the Worker only. Filtering is
new in this version: the attribute word is bounded to 32 bits, text queries post-filter through the document
store, and there is no `$or` ([docs/filtering.md](docs/filtering.md#not-done)).

## Development

```sh
cargo test --workspace
cd sdk/ts && npm run typecheck && npm test
cd sdk/python && maturin develop --uv && pytest
```

The full list of checks, the wasm build, the deploy steps and how a version is released (a `v*` tag publishes
`@aiuitech/thread-sdk` and `thread-sdk`) are in [docs/operations.md](docs/operations.md); versions and what changed in
each are in [CHANGELOG.md](CHANGELOG.md).

thread is developed inside [inseam](https://inseam.io)'s repository, and this repository is a mirror of it. Issues
and pull requests are welcome here: an accepted pull request is applied there and comes back through the mirror.

MIT licensed.
