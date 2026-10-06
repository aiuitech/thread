# thread docs

Start with [architecture.md](architecture.md) if you want to know how it works, [operations.md](operations.md)
if you want to run it, [portability.md](portability.md) if you want to move or ship one, and the SDK READMEs ([TypeScript](../sdk/ts/README.md), [Python](../sdk/python/README.md))
if you want to use it.

| page | what it covers |
|---|---|
| [architecture.md](architecture.md) | the design in one sitting: slabs, two tiers, routing, segments, why it is cheap, where the code lives |
| [cold-start.md](cold-start.md) | cold-start cost against S3 for both SDKs: a summary of what changed (baseline -> merged defaults, opt-in modes, rejected ideas), the `bench/cold/` harness (fake S3 proxy with a network model, benches, packs), the open/query timeline in the SDK stats, the baseline and where the time goes, store-level options, compact routing / cold tree routing / base packs, warm packs, the inline manifest, options outside the SDK |
| [benchmarks.md](benchmarks.md) | thread vs Cloudflare Vectorize on dbpedia-openai-1M, generation history, update and compaction numbers, offline diagnostics |
| [query-path.md](query-path.md) | slab formats, the two-phase query (scan, rerank, speculative reads), caching of revisions and bases, the SDKs' cache and miss policies, wasm trap recovery, the centroid tree and beam routing, cold opens (compact routing tables, cold tree routing, base packs) |
| [profiles.md](profiles.md) | the `edge` and `local` layouts, what each knob trades, manifest query defaults and how to change them |
| [update-path.md](update-path.md) | manifest revisions, delta segments, tombstones, compaction (manual and automatic), cell splitting, GC |
| [portability.md](portability.md) | moving an index: copying between stores, snapshotting one revision, packs (one file, queried in place) and their format, base packs, warm packs |
| [lexical.md](lexical.md) | BM25 segments, the `docs` store, the `keys` section, the tokenizer, fusion, size-vs-speed tuning, the native CLI |
| [chunking.md](chunking.md) | flat word-window chunking before embedding (`words` 200, `overlap` 0), `upsert({chunk, embed})` in both SDKs, chunk keys `k#i`, stale chunks on re-put, `thread chunk` / `seg build --chunk-words` |
| [keys.md](keys.md) | string keys: `get(key)` in two reads, keyed upserts and id allocation (`manifest.nextId`), prefix listings, delete by key, the key-value index case (keys + JSON meta + facet tokens) |
| [filtering.md](filtering.md) | attribute filters: the 32-bit word and its schema, filter expressions and their compiled clauses, pre-filtering inside the scan with probe widening, post-filtering of text queries |
| [build-pipeline.md](build-pipeline.md) | the build stages, build slots (Worker Loader), PQ codebook training, incremental k-means, finalize |
| [build-flags.md](build-flags.md) | every `bench/driver.mjs` flag, and the commands that built g1 and g2 |
| [operations.md](operations.md) | dataset prep, wasm build, deploy, build a generation, benchmarks, compaction, checks before a commit |
| [worker-loader.md](worker-loader.md) | the spike that showed dynamic Workers give each build slot its own isolate |

Raw measurements live in `data/bench-*.txt` (gitignored) and the SDK's [VERIFICATION.md](../sdk/ts/VERIFICATION.md).
