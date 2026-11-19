# thread

**A search index that lives in your bucket.**

thread gives your app vector search, full-text search, or both together. The index is a set of plain files in
object storage (Cloudflare R2, S3, or a folder on disk). Your code reads those files directly, from a Cloudflare
Worker, from Node or from Python. There is no database to run and no search service to pay for.

```ts
const index = await thread.open({ bucket: env.BUCKET, embed: { ai: env.AI } });

await index.upsert({ keys: ["notes/deploy.md"], texts: ["The deploy failed because the migration timed out."] });

const { hits } = await index.search({ text: "why did the deployment fail?", topK: 10, docs: true });
```

**[Quickstart](docs/quickstart.md)** · **[Why thread](docs/why-thread.md)** · **[How it works](docs/how-it-works.md)** · **[All docs](docs/README.md)**

## Install

```sh
npm install @aiuitech/thread-sdk      # TypeScript: Cloudflare Workers and Node 20+
pip install thread-sdk                # Python 3.11+
```

Both packages have no runtime dependencies. They read and write the same index and return the same answers.

## Why use it

- **You own the index.** It is files under a prefix of your bucket. You can copy it, back it up, sync it to a
  laptop, or pack it into one file and query that file in place.
- **Nothing to operate.** A query is a few range reads and some arithmetic. It runs inside your Worker or your
  process. No cluster, no replicas, no capacity planning.
- **It is cheap.** At 1 million vectors and 1 million queries a month, a Worker with a warm edge cache costs about
  $0.10 beyond the Workers Paid plan. Cloudflare Vectorize costs about $31 for the same load
  ([how we measured](docs/benchmarks.md#cost)).
- **Vectors and text in one index.** One `search` call runs semantic search, BM25 keyword search, or both, and
  merges the two result lists by rank.
- **It works everywhere.** The same index opens in a Worker (wasm), in Node, and in Python (a native extension).
  It also opens offline, from a folder or from a single pack file.

## What you can do with it

| You want to | Use | Guide |
|---|---|---|
| Search by meaning | `search({ text })` with an embedding model, or `query(vector)` | [Search](docs/search.md) |
| Search by keywords | `search({ text })` on a text index (BM25) | [Search](docs/search.md) |
| Do both at once (hybrid) | `search({ text })` on an index that has both | [Search](docs/search.md) |
| Add, replace and delete documents | `upsert`, `delete`, `deleteKeys` | [Write documents](docs/writing.md) |
| Limit results to a tenant, a language, a date | `filter: { tenant: 42 }` | [Filter](docs/filtering.md) |
| Get a document by its name | `get(["notes/deploy.md"])`, `listKeys({ prefix })` | [Keys](docs/keys.md) |
| Index long documents | `upsert({ chunk: true })` | [Chunking](docs/chunking.md) |
| Let the index call your embedding model | `open({ embed })` | [Embedding models](docs/embedding.md) |
| Ship an index as one file | `packIndex`, then open the pack anywhere | [Move an index](docs/portability.md) |

## How it compares

thread is a library over files, not a service. That is the main difference from the other options.

| | thread | A managed vector database (Vectorize, Pinecone, turbopuffer) | A database extension (pgvector) |
|---|---|---|---|
| Where the index is | Your bucket or disk | The vendor's storage | Your database server |
| What you run | Nothing | Nothing | A database |
| What you pay for | Object storage and your own compute | Stored vectors, queries, often a monthly minimum | A server that is always on |
| Keyword and hybrid search | Built in | Depends on the vendor | Postgres full-text search |
| Works offline, or from one file | Yes | No | No |
| Rich metadata filters | Small: 32 bits of fields per document | Yes | Full SQL |
| Lowest latency for one query | Good, not the best | Often better | Good on a warm server |

Measured against Cloudflare Vectorize on the standard 1M-vector benchmark (dbpedia-openai-1M, 1536 dimensions,
2026-10-05): the same recall (95.1% against 95.4%), about 3x the median latency (88 ms against 31 ms), half the
p95 latency, 7 to 8x the throughput, and a few percent of the cost.

The full comparison, with the cases where another tool is the better choice, is in
[Why thread](docs/why-thread.md). The numbers are in [Benchmarks](docs/benchmarks.md).

## How it works, in short

1. Vectors are grouped into **cells** of similar vectors. Each cell is one immutable file.
2. A query finds the few cells nearest to it, reads a small part of each file, and re-scores the best 150
   candidates. At 1 million vectors that is about 6 MB of cached reads and 33 ms of CPU.
3. Text is stored beside the cells as BM25 **segments**. A text query reads them in two rounds.
4. A write never changes a file. It adds small new files and a new **revision**. A background step, compaction,
   merges them later.

Because no file ever changes, every file can stay in a cache for ever, and every query reads a consistent
snapshot. [How it works](docs/how-it-works.md) explains this with pictures. The engine is Rust, compiled to wasm
for Workers and Node, and to a native extension for Python.

## Status

thread is young. It runs in production at [inseam](https://inseam.io), where it was built. Queries, text search,
hybrid search, filters and packs are tested end to end in both SDKs, and a cross-check confirms that Python and
TypeScript return the same results from the same index. The write path in the SDKs is marked experimental.
[Limits and status](docs/limits.md) lists what thread does not do.

## Repository

| Path | What is there |
|---|---|
| `crates/thread-core` | The engine: file formats, quantization, k-means, routing, scan, filters, BM25, fusion |
| `crates/thread-wasm`, `crates/thread-py` | wasm-bindgen and PyO3 bindings over the engine |
| `crates/thread-cli` | A native CLI for text segments (`thread seg build`, `thread tokenize`, `thread chunk`) |
| `sdk/ts`, `sdk/python` | The SDK packages ([TypeScript reference](sdk/ts/README.md), [Python reference](sdk/python/README.md)) |
| `examples/cloudflare-worker` | The reference Worker: queries, writes, and the build pipeline for large indexes |
| `bench` | The build driver, benchmarks and dataset preparation |
| `docs` | The documentation, listed in [docs/README.md](docs/README.md) |

## Development

```sh
cargo test --workspace
cd sdk/ts && npm run typecheck && npm test
cd sdk/python && maturin develop --uv && pytest
```

[Operations](docs/operations.md) has every check, the wasm build, the deploy steps and the release process.
[CHANGELOG.md](CHANGELOG.md) has the versions.

thread is developed inside [inseam](https://inseam.io)'s repository, and this repository is a mirror of it. Issues
and pull requests are welcome here: an accepted pull request is applied there and comes back through the mirror.

MIT licensed.
