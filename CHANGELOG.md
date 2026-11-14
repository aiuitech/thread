# Changelog

Versions are shared by the npm package `@aiuitech/thread-sdk`, the PyPI package `thread-sdk` and the Rust crates
(`Cargo.toml` workspace version). A release is a `v<version>` tag; `.github/workflows/release.yml` builds the wasm
from source, runs every check and publishes.

## Unreleased

- **Dense compaction reads what its slice needs, not the whole backlog, and the oldest delta always folds**
  (TypeScript, `compact()` / `compactDeltas`). Before it planned, a call read every tombstone set (one get each)
  and every delta's entry table (two range gets each), and the deadline counted from the start of the call. When
  that read phase took longer than the deadline, no cell started, the run ended `deadline` with 0 cells, and the
  backlog only grew (inseam, 2026-11-14, SDK 0.2.5: 767 deltas of about 3 KB and 2,531 tombstone sets, about 4,000
  gets taking 20 to 30 s against a 10 s deadline; every run ended `deadline` with 0 cells, and one Durable Object
  alarm passed its limit of 10,000 R2 reads). Now:
  - The deltas' cells are read oldest first in windows of 8, and the call plans after each window. Reading stops
    when the slice is full (16 deltas in a row could not join it), or when reading has used a third of
    `deadline - 2.5 s`; the first window is always read. A delta up to 64 KiB is read whole, one get that is also
    the fold's input (before: two range gets, then the whole delta again). While some deltas are unread, a split
    cell keeps its mean and `force` takes no oversized cell on its own.
  - A tombstone set's ids are read once, when a cell being rewritten or a delta being folded is filtered by it (a
    slab of slabRev r by the dense-view sets with rev >= r, an empty slab by none, a delta added at rev a by those
    with rev > a), or when its drop depends on the build-id test alone. Lexical-scoped sets are never read. The
    planner's memory estimate uses the counts in the manifest.
  - The oldest delta folds in every call that plans: the cells it needs to commit run first, whatever the deadline,
    and are committed. Only the lock, a conflict or a 413 stops it. Such a call can take longer than `deadlineMs`
    when the oldest delta alone needs more; its note then starts `read phase passed the deadline:` with what was
    read. A purge- or `force`-only call whose read phase passed the deadline ends `deadline` with that note.
  - `CompactResult.reads` (also in `compact.last.json`): deltas and tombstone sets read, of those the base lists,
    and the ms to the plan.
  - The objects a compaction writes are unchanged (`test/compact-worker.test.ts` still matches the Worker byte for
    byte). `test/compact-backlog.test.ts` builds 300 deltas and 1,240 tombstone sets from small upserts,
    dense-scoped and lexical-scoped re-puts and deletes, then calls `compact({ deadlineMs: 1 })` until nothing is
    due: on 0.2.6 each call made 2,145 gets and ended `deadline` with 0 cells; now 56 calls take the backlog from
    300 deltas to 3, each folds at least one, reads 8 deltas, never reads a lexical-scoped set, and makes at most
    1,157 gets (615 on average). Every live document, deleted id and replaced vector answers as before. The
    tombstone sets are most of those gets: a cell whose slab predates a set needs it. Unioning the sets into fewer
    objects is the next step.
- Python: no change (it has no dense compaction). The Worker's own compaction is unchanged.

## 0.2.6 (2026-11-09)

- **The block cache keeps alive only the bytes a read asked for** (TypeScript, `CachedBlobStore` over the Workers
  Cache API). 0.2.5's block reads returned `subarray` views of the blocks they filled, so every range a caller kept
  held its whole block after the block's `blockBuffers` slot was released, and `blockBuffers` bounded nothing: 160
  ranges of 16 bytes across a large segment, read cold, kept 160 MiB alive with `blockBuffers: 2` (inseam's review,
  2026-11-09; 0.2.4 kept 2,560 bytes). In a Worker (128 MB isolate) a cold search over sparse ranges of large
  segments could run out of memory. Now each range is copied out of its block (or out of the cache response it came
  in) before the block's slot is given back, on every path of `get` and `getRanges`: a block filled from the origin,
  a block being filled by another reader, a cached block (206 or 200), and an object cached whole read for a range
  without `fill`. The memory block reads keep alive is the bytes returned plus at most `blockBuffers` blocks in
  flight. `test/blocks.test.ts` counts the block buffers alive after a forced GC: 160 cold reads of 16 bytes over
  64 KiB blocks kept 160 blocks (10 MiB) on 0.2.5 and keep none (2,560 bytes) now, and a block read starts with at
  most one other block alive at `blockBuffers: 2` (159 before). The 0.2.5 workaround `open({ blockBytes: 64 * 1024 })`
  can go back to the default.
- Unchanged, on purpose: a `fill` read (a slab's scan prefix) still returns a view into the whole object, so the
  query takes the slab's rerank records from the buffer (`stats.wholeSlabs`); hits of an in-process cache
  (`MemoryCache`) are still views into the object it holds (bounded by its per-object cap, a quarter of `maxBytes`;
  a copy would hold the bytes twice while it is cached); packs and warm packs in memory still serve views into
  themselves.
- `S3BlobStore`: a range answered with the whole object (200, Range ignored) returns the range copied out, not a
  view that holds the whole object.
- Python: no change. Its `CachedBlobStore` has no block cache (`fills` is always 0); its `memoryview`s are into
  whole objects held by its in-process caches or read for a `fill`, as in TypeScript.

## 0.2.5 (2026-11-09)

- **Large lexical segments are cached in blocks** (TypeScript, `CachedBlobStore` over the Workers Cache API).
  A segment over 2 MiB (`LexSegmentOptions.fillBytes`) was read by range and never cached: a range read without
  `fill` that missed went to R2 and was not stored, so every search read every large segment from R2 again, and
  merging segments made search slower (inseam, 2026-11-09: 39 segments of 4 MiB, 293 R2 reads and 4.2 s in the
  lexical list on a warm cache). Now such reads pass the segment's size (`GetOptions.size`) and the store serves
  them from aligned blocks of `blockBytes` (default 1 MiB), each cached under its own key in the store's scope; a
  block that misses is read from R2 (that block only, never the whole object), serves the read and is put in the
  cache through `waitUntil`. Concurrent readers of a block share one R2 read, at most `blockBuffers` blocks (16 MiB
  worth) are held from their read until their put settles, and block reads in flight stay within `inflight`. An
  object already cached whole is still served from that entry; slabs, deltas, manifests and segments up to 2 MiB
  keep whole-object caching. Measured against R2 (`bench/blocks/`, docs/lexical.md "Block cache"): a warm hybrid
  search over 8 segments of 4.1 MB went from 4.3 s and 125 R2 reads to 0.2 s and none; over 2 segments of 46 MB,
  from 1.3 s and 54 reads to 0.1 s and none. Dense-only searches are unchanged.
- `open({ blockBytes, blockBuffers })` and `CachedBlobStoreOptions.blockBytes` / `blockBuffers`; `blockBytes: 0`
  turns blocks off. `CachedBlobStore.blockStats` counts fills; `drain()` also waits for block fills.
- **`fills` in the trace envelope's `io`** (`TraceIo.fills`, `BlobTrace.fills`, `PhaseStat.fills`): the blocks a
  span read from R2 into the cache. Each fill is also a `read` and a `miss`, and its whole block is in `bytes`; a
  block lookup is one `hit` or `miss`. Python's envelopes carry `fills: 0`, so both SDKs keep one schema.

## 0.2.4 (2026-11-08)

- **A bulk build step ends by progress, not only by the clock** (TypeScript `BulkBuild.step`). A `load` pass
  starts no replay within 4 s of the deadline; when opening the build's index left less than that, `step()` called
  the phase again and again, and in Workers `Date.now()` does not move without I/O, so the deadline never came and
  the invocation ran until its CPU limit (inseam, 2026-11-08: a rebuild alarm killed at 32 s of CPU in the load of
  a 23,594-document, 40-cell build). Now `step()` returns as soon as a pass over its phase commits nothing (no
  piece done, no state written), with `done: false` and the phase it stopped in; every pass it repeats has
  written something. The same holds in the other phases: a `sample` pass with no time left writes nothing (it
  used to write the state unchanged), and a held `compact.lock` in `load` or `compact` ends the step instead of
  being retried until the deadline.
- `BulkStepResult.progressed`: false when the call committed nothing (call again; keep the handle, so that the
  next call does not open the index again). `worked` now lists only the phases the call made progress in.
- The `load` reserve is at most 40% of `deadlineMs`: with a deadline of 4 s or less, `load` could never start a
  replay.
- A `compact` phase with no delta left (the last compaction of `load` folded them all) finishes the build: the
  forced compaction is a `noop`, and the build used to stay in `compact`. A `noop` of a due compaction in `load`
  drops the handle's index, so the next pass opens the head.
- **`BulkBuild.close()`**: releases the index the handle opened for `load` and `compact`, and writes or deletes
  nothing; `thread.bulkBuild` on the same `gen` resumes the build, and a later `step` on the handle opens the
  index again. For a handle dropped before `done`, e.g. after a failed step whose state ETag may be stale.

## 0.2.3 (2026-11-08)

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
- `upsert` (TypeScript) refuses `metas` and `texts` that are not strings or null with a 400 naming the entry, before
  the chunker, the embedder or the wasm: an object passed as a meta used to trap the wasm instance and reset it
  for every index in the isolate. `bulkBuild.add` checks the same when staging.
- **Trace envelopes** ([docs/observability.md](docs/observability.md)): every SDK operation (query, search,
  lexical search, open, get, listKeys, upsert, delete) builds one JSON envelope, the same schema in TypeScript,
  Python and the reference Worker: spans with their start, wall time and store work, counters, the CPU of each
  kernel (wasm calls timed in Node, the native `QueryRun` timing its scans, merge and rerank with the GIL released)
  and, opt-in, one event per store read (`traceEvents` / `trace_events`). `stats.trace` / `stats["trace"]`,
  `r.trace`, `index.openTrace` / `index.open_trace`; `open({ onTrace, traceSample })` /
  `open_index(on_trace=, trace_sample=)` hand a sampled share to a function. The interop test compares both SDKs'
  envelopes. Python's `KeyPage`, `UpsertResult` and `DeleteResult` gain a `trace` field (excluded from equality).
- **`pushed`**: the scan kernels count entries pushed into the top-T heap (`ScanStats.pushed`, the wasm scans'
  fourth stats slot, `stats.pushed`). Callers passing a three-slot stats array keep working.
- **Reference Worker**: `/query` returns its envelope (`trace`, `sdk: "worker"`); `TRACE_SAMPLE` (default `"0"`) logs
  a share of them as JSON lines and writes them to the `QUERY_METRICS` Analytics Engine dataset (`thread_queries`).
- **Profiling**: Criterion benches of the query kernels at g4's shape (`cargo bench -p thread-core`, real slab and
  codebook fixtures via `THREAD_BENCH_SLAB` / `THREAD_BENCH_PQ`), a `profiling` cargo profile (release with line
  tables) for flamegraphs, `bench/scenarios.mjs` (workload scenarios over the SDK), and `bench/query_bench.mjs`
  prints the Worker's per-span breakdown.

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
