# Changelog

Versions are shared by the npm package `@aiuitech/thread-sdk`, the PyPI package `thread-sdk` and the Rust crates
(`Cargo.toml` workspace version). A release is a `v<version>` tag; `.github/workflows/release.yml` builds the wasm
from source, runs every check and publishes.

## 0.2.3 (unreleased)

- **Dense compaction keeps every cell processable by its budget** (TypeScript `index.compact` / `compactDeltas`).
  Before, at dim 1024 a 32 MiB call folded a one-cell generation up to just under `maxCellEntries` (3000, profile
  `edge`) and then could never split it (a split of 3000 entries needs about 46 MiB), so every later call was
  `blocked` and the deltas piled up (inseam's `e1`, 2026-11-04). Now `maxBytes` also sizes the cells: they split at
  `min(maxCellEntries, splitEntries / 2)`, where `splitEntries` is the largest cell the budget can split on its
  own; no fold takes a cell past `splitEntries`, and a cell that keeps a delta out is split first (`r.presplit`).
  `compactionLimits(manifest, { maxBytes })` / `index.compactionLimits(maxBytes)` give `cellEntries`,
  `splitEntries`, `foldEntries`, `upsertVectors` (an upsert of that many vectors always folds) and
  `splitCommitCells` without I/O; `r.limits` has them too.
- **Input a budget cannot process is an error, not `blocked`**: a delta with more entries in one cell than one call
  can fold, a cell an older writer let grow past `splitEntries`, a delta over `cellsPerRun` cells, or a split commit
  whose fine means pass the budget throw a `ThreadError` with status 413 that names it and what it needs. `blocked`
  is left only for a slice whose splits were deferred (`more` is true).
- **`estimatedPeakBytes` is an upper bound again**: measured with the wasm memory growth counted
  (`bench/compact_memory.mjs`), a fold holds about 3x the old slab plus 6x the new one (the model said 4x the new
  one: inseam's 16:57 fold was planned at 18.2 MB and measured 38.9 MB), a split 8x plus the decoded vectors, and
  1 MiB is fixed; the fine means of a split commit are a phase of their own after the cells. Details in
  docs/update-path.md "Memory of a call".
- A split whose 2-means leaves one half empty (identical vectors) halves the entries under the one mean, so such a
  cell still shrinks.
- With splits, a compaction keeps up to 4.5 s more than the 2.5 s commit reserve, scaled by the fine means' bytes
  (all of it with a tree): before, it always kept 7 s, so under a deadline under about 8 s a call with splits
  started no cell.
- **Bulk build** (TypeScript, `@experimental`): `thread.bulkBuild({ store, gen, embedding | dim, lex, attrs,
  maxBytes })` stages documents with vectors (`add`) under `build/{gen}/`, then `seal()` and `step({ deadlineMs })`
  run sample, k-means (wasm), create, load (each stage replayed as an upsert into `k` cells, compacting when due)
  and compaction, in steps bounded by `maxBytes` and resumable from `build/{gen}/state.json`; `cleanup()` removes
  the staging objects. `bulkBuildPlan(n, dim, { maxBytes })` gives `k`, the cell and sample sizes and `maxCells`
  (about 1000 cells at dim 1024 and 32 MiB; larger builds belong in the Worker's build slots). See
  docs/embedding.md "A bulk build in bounded steps".
- `createGeneration({ means })` (TypeScript): rev 0 with `k` empty cells with those means.

## 0.2.2 (2026-11-04)

- **Committing a segment merged elsewhere**: `index.commitMergedSegment({ run, readRev, key, bytes, count, keys })` /
  `commitMergedSegment(ctx, input)`. A process with the memory to read a large run whole (thread-core
  `segment::merge`, which the wasm `merge_segments` wraps) writes the merged object under `idx/{gen}/seg/`, and the
  index's writer commits it in place of the run. The object is opened first (container, section headers,
  directories) and its size checked, so a truncated or foreign object is refused before a revision names it.
  `merged` 0 and no revision when the head no longer lists the run contiguously. The segment gets
  `addedRev = readRev`: sets newer than the plan still hide its ids, and the lexical-only sets the merge applied
  drop once no segment predates them. For writers with little memory (a Durable Object) whose bounded
  `compactSegments({ maxInputBytes })` cannot keep a large index in shape. TypeScript only.

## 0.2.1 (2026-11-03)

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
- **Scoped tombstones**: `upsert({ scope: "dense" | "lexical" })` (Python `scope=`) replaces one side of the
  documents: a vector for existing keys without touching their docs, texts and keys, or new texts and metas without
  the vector (attrs cannot drift between the sides). `TombRef.scope`; dense reads and the dense compaction use the
  dense view, segments, docs and keys the lexical view; sets drop per side, and segment compaction now drops the
  lexical-only ones. Manifests with scoped sets carry `features: ["scoped-tombstones"]`; readers refuse features
  they do not know (readers before 0.2.1 ignore `features` and would over-hide). The Worker reads the dense view,
  compacts per scope and takes `scope: "dense"` on `/upsert`.
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
