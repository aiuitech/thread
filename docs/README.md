# thread docs

Start with [architecture.md](architecture.md) if you want to know how it works, [operations.md](operations.md)
if you want to run it, [portability.md](portability.md) if you want to move or ship one, and the SDK READMEs ([TypeScript](../sdk/ts/README.md), [Python](../sdk/python/README.md))
if you want to use it.

| page | what it covers |
|---|---|
| [architecture.md](architecture.md) | the design in one sitting: slabs, two tiers, routing, segments, why it is cheap, where the code lives |
| [cold-start.md](cold-start.md) | cold-start cost against S3 for both SDKs: the `bench/cold/` harness (fake S3 proxy with a network model, benches, packs), the open/query timeline in the SDK stats, the baseline and where the time goes |
| [benchmarks.md](benchmarks.md) | thread vs Cloudflare Vectorize on dbpedia-openai-1M, generation history, update and compaction numbers, offline diagnostics |
| [query-path.md](query-path.md) | slab formats, the two-phase query (scan, rerank, speculative reads), caching of revisions and bases, wasm trap recovery, the centroid tree and beam routing |
| [profiles.md](profiles.md) | the `edge` and `local` layouts, what each knob trades, manifest query defaults and how to change them |
| [update-path.md](update-path.md) | manifest revisions, delta segments, tombstones, compaction (manual and automatic), cell splitting, GC |
| [portability.md](portability.md) | moving an index: copying between stores, snapshotting one revision, packs (one file, queried in place) and their format |
| [lexical.md](lexical.md) | BM25 segments, the `docs` store, the tokenizer, fusion, size-vs-speed tuning, the native CLI |
| [build-pipeline.md](build-pipeline.md) | the build stages, build slots (Worker Loader), PQ codebook training, incremental k-means, finalize |
| [build-flags.md](build-flags.md) | every `bench/driver.mjs` flag, and the commands that built g1 and g2 |
| [operations.md](operations.md) | dataset prep, wasm build, deploy, build a generation, benchmarks, compaction, checks before a commit |
| [worker-loader.md](worker-loader.md) | the spike that showed dynamic Workers give each build slot its own isolate |

Raw measurements live in `data/bench-*.txt` (gitignored) and the SDK's [VERIFICATION.md](../sdk/ts/VERIFICATION.md).
