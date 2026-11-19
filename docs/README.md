# thread docs

thread is a search index that lives in your bucket: vector search, full-text search, or both, read directly from
a Cloudflare Worker, Node or Python.

New here? Read these three, in order: [Quickstart](quickstart.md), [Why thread](why-thread.md),
[How it works](how-it-works.md).

## Start here

| Page | What you get |
|---|---|
| [Quickstart](quickstart.md) | A working index in five minutes: TypeScript, Python, then a Cloudflare Worker |
| [Why thread](why-thread.md) | The reasons for the design, when to use it, when not to, and a comparison with Vectorize, Pinecone, turbopuffer, S3 Vectors, pgvector and LanceDB |
| [How it works](how-it-works.md) | The ideas, with pictures: files, cells, the query in three steps, hybrid search, writes |
| [Limits and status](limits.md) | What thread does not do, the hard limits, and how mature each part is |

## Guides

Each guide is a task. It starts with the code and ends with the limits.

| Page | Task |
|---|---|
| [Create an index](create-an-index.md) | Choose the kind of index, create it, load a lot of data, change the model |
| [Connect to storage](storage.md) | R2, S3, a folder, a pack file, memory, and the caches |
| [Write documents](writing.md) | Add, replace and delete documents, and the rules for cheap writes |
| [Search](search.md) | Keyword, semantic and hybrid search, the hits, the tokenizer, tuning |
| [Filter](filtering.md) | Limit a search to a tenant, a language, a date |
| [Keys](keys.md) | Name documents by a string, fetch and list them, use an index as a key-value store |
| [Embedding models](embedding.md) | Let the index call Workers AI, OpenAI or your own model |
| [Chunking](chunking.md) | Split long documents into passages |
| [Move an index](portability.md) | Copy an index, snapshot one revision, pack it into one file |
| [Migrate from Vectorize](migrate-from-vectorize.md) | Map every call, and a migration plan |
| [Run it in production](production.md) | Batching, compaction, garbage collection, freshness, monitoring |

## Reference

| Page | What it covers |
|---|---|
| [TypeScript SDK](../sdk/ts/README.md) | Every call, option and result field, the stores and the caches |
| [Python SDK](../sdk/python/README.md) | The same for Python, and the differences from TypeScript |
| [The reference Worker](../examples/cloudflare-worker/README.md) | The Worker that serves queries and runs the build pipeline |
| [Changelog](../CHANGELOG.md) | Every version and what changed |

## Under the hood

These pages are the detail: formats, algorithms and measurements. You do not need them to use thread. Read them
when you want to know exactly what happens, or when you work on thread itself.

| Page | What it covers |
|---|---|
| [Architecture](architecture.md) | The design on one page: slabs, two tiers, routing, segments, and where the code is |
| [Query path](query-path.md) | Slab formats, the two-phase query, speculative reads, caches, the centroid tree, cold opens |
| [Update path](update-path.md) | Manifest revisions, delta segments, tombstones, compaction, cell splitting, garbage collection |
| [Text index](lexical.md) | BM25 segments, the document store, the key section, the tokenizer, fusion, the native CLI |
| [Profiles](profiles.md) | The `edge` and `local` layouts, what each setting trades, and the query defaults |
| [Benchmarks](benchmarks.md) | thread against Cloudflare Vectorize on dbpedia-openai-1M, the cost model, update and compaction numbers |
| [Cold start](cold-start.md) | What the first query of a new process costs against S3, and every option that changes it |
| [Observability](observability.md) | Trace envelopes, sinks and sampling, kernel benchmarks, how to find a bottleneck |

## Build and operate thread itself

For people who build large indexes with the pipeline, deploy the reference Worker, or release thread.

| Page | What it covers |
|---|---|
| [Operations](operations.md) | Dataset preparation, the wasm build, deploy, building a generation, the checks before a commit, releasing |
| [Build pipeline](build-pipeline.md) | The build stages, build slots, codebook training, incremental k-means |
| [Build flags](build-flags.md) | Every flag of the build driver |

## Research

| Page | What it covers |
|---|---|
| [Worker Loader spike](worker-loader.md) | The experiment that showed dynamic Workers give each build slot its own isolate |
| [Research across search hops](research-hops-experiment.md) | A proposed experiment on search and reranker objectives across hops |

Raw measurements are in `data/bench-*.txt` (not in the repository) and in the TypeScript SDK's
[VERIFICATION.md](../sdk/ts/VERIFICATION.md).
