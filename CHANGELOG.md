# Changelog

Versions are shared by the npm package `@aiuitech/thread-sdk`, the PyPI package `thread-sdk` and the Rust crates
(`Cargo.toml` workspace version). A release is a `v<version>` tag; `.github/workflows/release.yml` builds the wasm
from source, runs every check and publishes.

## 0.2.1 (unreleased)

- **Bounded lexical compaction**: `compactSegments({ maxInputBytes })` / `compact_segments(max_input_bytes=)`
  merges one contiguous run of segments whose bytes (plus 4 bytes per tombstoned id) fit the budget, for writers
  with little memory (a Durable Object). The run keeps its place and its newest `addedRev`; `merged` 0 when no two
  adjacent segments fit.
- **Cache keys**: the in-process cache (`MemoryCache`, any `BlobCache`) is scoped by the index's key prefix, as
  the Cache API was in 0.2.0, and so is a `store` passed as a `PrefixedBlobStore` without `prefix`.
- **Dense compaction in the TypeScript SDK**: `index.compact(opts)` / `compactDeltas(ctx, opts)` folds delta
  segments into slab revisions, splits cells over `maxCellEntries` and drops unreferenced tombstone sets in one
  isolate, bounded by `maxBytes` (estimated peak memory, default 48 MiB), `cellsPerRun` and `deadlineMs` (a cut slice
  commits its finished part); `r.more` tells whether to call again. Same objects as the Worker's compaction, byte for
  byte, and the same `compact.lock`. `index.compactionDue()` / `compactionStatus(manifest)` check the thresholds
  without I/O. Not in the Python SDK.
- **Switching generations from the SDK**: `setCurrent(store, { gen, rev?, expect? })` / `index.setCurrent` moves
  `idx/current.json` to another generation with a compare-and-swap (409 unless the pointer is `expect`);
  `gcGeneration` / `index.gc` is the Worker's `/build/gc` (keep, graceSec, dry run); `dropGeneration` deletes a
  generation the pointer left once the grace period has passed with no write to it. TypeScript only.
- `BlobStore.delete` and `BlobStore.listObjects` (key, size, upload time), optional; implemented by the R2, S3,
  memory, file, prefixed and cached stores.
- `sdk/ts/package-lock.json` names the package `@aiuitech/thread-sdk`.

## 0.2.0 (2026-11-03)

First published version.

- **Embedding models** ([docs/embedding.md](docs/embedding.md)): `manifest.embedding` names the model a dense
  generation is for; `open({ embed })` / `open_index(embed=...)` takes the Workers AI binding, Workers AI REST
  credentials, an OpenAI-compatible key or a function; `search({ text })` embeds the query (and still fuses with
  BM25), `upsert({ texts })` embeds documents; `index.embed(texts, kind)`, `index.canEmbed`, `index.embedding`.
- **A dense generation from nothing**: `createGeneration` / `create_generation` / the Worker's `POST /create`
  publish one empty cell for a model (or a bare `dim`); it grows through upserts, the Worker's automatic compaction
  and cell splits. Readers clamp a manifest default `nprobe` to the cell count.
- **Worker**: `AI` and `EMBED_API_KEY` bindings; JSON `/query {text}` and `/upsert {texts}` are embedded in the
  orchestrator before forwarding; `POST /embed`.
- The TypeScript package is now `@aiuitech/thread-sdk` (was the unpublished `thread-sdk`); the import paths
  `@aiuitech/thread-sdk`, `@aiuitech/thread-sdk/node` and `@aiuitech/thread-sdk/wasm/thread_bg.wasm`.

Before 0.2.0 (unreleased, see the git history): blob-native dense index in R2/S3 (two-tier slabs, PQ scan tier,
centroid tree routing), BM25 segments and fusion in thread-core, keys and the docs store, attribute filters,
chunking, incremental updates with compaction and cell splitting, packs, cold-start work, the Python SDK.
