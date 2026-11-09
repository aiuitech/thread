# thread-sdk

Query and update a [thread](../../README.md) blob-native search index from a Cloudflare Worker: dense vectors (the
IVF slabs), BM25 text (lexical segments), or both fused by rank, with one document store for the hits. The index
lives as immutable objects in R2 (or any S3-compatible store); the SDK streams the probed slabs and the queried
posting lists through the Rust/wasm kernels, like a Vectorize binding but with the index in your own bucket,
readable through the Workers Cache API at the edge.

```ts
import wasmModule from "@aiuitech/thread-sdk/wasm/thread_bg.wasm";
import { Thread } from "@aiuitech/thread-sdk";

const thread = await Thread.init({ wasmModule });                   // once per isolate
const index = await thread.open({ bucket: env.BUCKET, waitUntil: (p) => ctx.waitUntil(p) });
const { hits } = await index.query(embedding, { topK: 10 });  // dense: [{ id, score, attrs, source }, ...]
const r = await index.search({ text: "deploy failed", vector: embedding, topK: 10, docs: true });
// hybrid: r.hits = [{ id, score, dense?: {rank, score}, lexical?: {rank, score}, doc?: {id, meta, attrs} }, ...]
const f = await index.search({ text: "deploy failed", vector: embedding, topK: 10, filter: { tenant: 42, lang: { $in: ["en", "fr"] } } });
```

## Install

```sh
npm install @aiuitech/thread-sdk
```

No runtime dependencies: the package ships its TypeScript build (`dist/`) and the wasm-bindgen output of
`crates/thread-wasm` (`dist/wasm/thread_bg.wasm`, ~240 KB). Node >= 20 for the tests; the SDK itself targets the
Workers runtime (WebCrypto, `fetch`, Cache API) and runs in Node for tests and tooling.

### Importing the wasm

The SDK is bundler-agnostic: you import the `.wasm` through your bundler, which turns it into a
`WebAssembly.Module`, and hand it to `Thread.init`. The SDK handles instantiation, the panic hook and recovery
from traps.

**Cloudflare Vite plugin** (`cf init workers`, `@cloudflare/vite-plugin`): import the file by a specifier ending
in `.wasm` so the plugin's `CompiledWasm` rule applies (a bare `@aiuitech/thread-sdk/wasm` is taken over by Vite's own wasm
handling and fails):

```ts
import wasmModule from "@aiuitech/thread-sdk/wasm/thread_bg.wasm";
```

```ts
// src/env.d.ts
declare module "*.wasm" {
	const module: WebAssembly.Module;
	export default module;
}
```

**Wrangler** bundles `*.wasm` imports as `CompiledWasm` modules by default; the same import works. With a
custom `rules` list keep `{ type: "CompiledWasm", globs: ["**/*.wasm"] }`.

**Node** (tests, scripts): `new WebAssembly.Module(fs.readFileSync("node_modules/thread-sdk/dist/wasm/thread_bg.wasm"))`.

### Pointing at the index

The index layout under the bucket is the one `thread`'s build pipeline writes (`idx/current.json`,
`idx/{gen}/manifest[.rev].json`, `slabs[.rev].json`, `slab/{cell:05}[.rev]`, `delta/{seq}.vibd`,
`tombstones.{rev}.u32`, `coarse.f32`, `fine.f32`, `pq.bin`). Three ways to reach it:

```ts
// 1. R2 binding (fastest: no auth, no TLS, same-colo reads)
const index = await thread.open({ bucket: env.BUCKET });
// equivalent, explicit:
const index = await thread.open({ store: new R2BlobStore(env.BUCKET) });

// 2. Any S3-compatible endpoint, e.g. R2's S3 API from a Worker (or from anywhere)
const index = await thread.open({
	store: new S3BlobStore({
		endpoint: "https://<account-id>.r2.cloudflarestorage.com",
		bucket: "vib-bench",
		region: "auto",                       // AWS S3: the bucket's region
		accessKeyId: env.S3_KEY,
		secretAccessKey: env.S3_SECRET,
		// pathStyle: true (default) -> https://endpoint/bucket/key; false -> https://bucket.endpoint/key
	}),
});

// 3. In Node: a directory holding the layout, or a single pack file (see "BlobStore" and docs/portability.md)
import { storeAt } from "@aiuitech/thread-sdk/node";
const index = await thread.open({ store: await storeAt("/data/idx") });
const index = await thread.open({ store: await storeAt("/data/idx-g4-rev12.threadpack") });

// 4. A pack kept in a bucket, read by range requests, from a Worker or anywhere
const index = await thread.open({ store: await PackedBlobStore.open(new R2BlobStore(env.BUCKET), "packs/g4.threadpack") });

// 5. Your own BlobStore (a KV, a test fixture) - see "BlobStore" below
const index = await thread.open({ store: new MemoryBlobStore() });
```

Whatever the store, `open` puts a `CachedBlobStore` (Workers Cache API, `caches.default`) in front of it unless
`cache: false`. Pass `waitUntil` so background cache fills outlive the response. Outside Workers (Node) there is no
Cache API: a remote store (`S3BlobStore`) gets an in-process `MemoryCache` (256 MB) instead, and a `DiskCache`
from `thread-sdk/node` survives restarts (see "Caches and the cold path").

## API

### `Thread.init({ wasmModule }): Promise<Thread>`

Instantiates the wasm once per isolate (wasm-bindgen keeps a single instance per module scope; a second `init`
with another module is a no-op). `thread.wasm` exposes the raw bindings for advanced use (build jobs, slab tools).

On a wasm trap (`WebAssembly.RuntimeError`) the instance is poisoned: `thread.withRecovery` (used by `query`,
`upsert`, `delete`) calls `__wbg_reset_state()`, reinstalls the panic hook, bumps `thread.epoch`, invalidates every
index's wasm objects (they are rebuilt from the cached base files on the next query) and retries the operation
once. `thread.resets` counts resets; `stats.attempt` is 2 on a retried query.

### `thread.open(options): Promise<ThreadIndex>`

| option | default | meaning |
|---|---|---|
| `bucket` / `store` | required | an R2 binding, or any `BlobStore` |
| `prefix` | `""` | key prefix in front of `idx/` (several indexes in one bucket) |
| `gen`, `rev` | `idx/current.json` | read a specific generation / revision; with both, the index is pinned and never refreshes |
| `cache` | `caches.default`; in Node a `MemoryCache` for remote stores | the Cache API (`CacheLike`) or a `BlobCache` (`MemoryCache`, `DiskCache`) in front of the store; `false` disables it |
| `cacheOrigin` | `https://vib-sdk.cache/` | synthetic origin of the cache keys (`${origin}idx/...`); use one per store |
| `missPolicy` | `"whole"` (Cache API), `"range+warm"` (`MemoryCache` / `DiskCache`) | what a slab-prefix cache miss reads: the whole slab, the range, or the range now and the whole slab in the background |
| `warmConcurrency` | 2 | `"range+warm"`: background whole-slab fetches at once (1 while the query's own reads are in flight) |
| `blockBytes` | 1 MiB | Cache API: block size of the block cache that serves range reads of lexical segments over 2 MiB ([docs/lexical.md](../../docs/lexical.md#block-cache)); `0` = off (those reads go to the store uncached) |
| `blockBuffers` | 16 MiB worth (16 at 1 MiB) | Cache API: blocks held in memory at once, from their store read until their cache put settles (ranges are copied out of them, so a range kept does not hold its block) |
| `waitUntil` | - | `ctx.waitUntil`, keeps background cache writes and warm fetches alive |
| `warmup` | on for remote stores outside Workers | open connections during `open` (`true` = the store's `inflight`, 32 against S3 from Node; a number; `false`) |
| `coldPolicy` | off | `{ nprobe?, rerank?, threshold? }` for queries whose probed slabs are mostly not cached (see `query`) |
| `embed` | - | how texts are embedded when the generation names its model (`manifest.embedding`): `{ ai: env.AI }` (Workers AI binding), `{ accountId, apiToken }` (Workers AI REST), `{ apiKey, headers?, fetch? }` (OpenAI-compatible), or a function `(texts, kind) => vectors`; without it `search` needs a `vector` and `upsert` needs `vectors` or `embed` (see [docs/embedding.md](../../docs/embedding.md)) |
| `refreshInterval` | 30 000 ms | how often a query checks `current.json` for a new revision, in the background (0 = only `refresh()`) |
| `prefetchBytes` | 64 MB | slab prefixes / delta and lexical segments a revision switch prefetches into the cache |
| `deltaCacheBytes` | 40 MB | delta segments kept in memory across queries (per index) |
| `routing` | `"f32"` | routing table to load: `"f32"` (`fine.f32`, 25 MB at 4096 x 1536), `"f16"` (`fine.f16`, half), `"i8"` (`fine.i8`, a quarter) or `"auto"` (the smallest the revision has); decoded to f32 on load, a revision without the table falls back to f32 |
| `coldRouting` | `"off"` | `"tree"`: the open waits only for the centroid tree and the codebook; until the full table has loaded (in the background) queries route over the tree's best `coldBeam` groups, range-reading just those groups' means from the tree-grouped table |
| `coldBeam` | `defaults.beam`, else 16 | groups ranked per cold-routed query |
| `basePack` | true | GET the revision's base pack (one object with everything the open reads) and, without one, start every read of the open as soon as the manifest is in; `false`: one read after another |
| `warmPack` | - | key of a warm pack (hot slab prefixes in one object): fetched with one GET in the background at open; its prefixes go into the `MemoryCache` (else a prefix layer in front of the store, `index.warmLayer`) and are served from memory (`index.warmReady`, `index.warmStats`). Opt-in; measured no gain on our workload (docs/cold-start.md, "Warm packs") |

`open` reads `current.json` (a pointer without `rev` means rev 0 = the legacy `idx/{gen}/manifest.json`), the
manifest (unless `current.json` carries an inline copy, which the SDK writers add when it is at most 16 KB), the slabs table and the tombstone sets, and loads the routing table (`coarse.f32` + `fine.f32`, ~25 MB
in wasm at 4096 x 1536) and the PQ codebook (format 3). With a base pack (`idx/{gen}/base.{rev}.threadpack`,
written by publish) that is two round trips: `current.json`, then the pack (the manifest is read alongside, in case
there is none, unless `current.json` inlines it); without one, three, or two with an inline manifest: the base files
are all requested as soon as the manifest is in. `routing: "i8"`
(or `"auto"`) cuts the routing table to 6.3 MB and is the table a base pack holds by default, so the whole open is
`current.json` + one 8.8 MB GET on g4; `coldRouting: "tree"` cuts it to the 2.5 MB cold base pack plus ~1 MB of
group means read by the first query. Which mode suits which deployment, and the probe overlap and recall each
costs: [docs/query-path.md](../../docs/query-path.md#cold-opens-compact-routing-cold-tree-routing-base-packs). `gen` without `rev` reads the generation's head: the
current rev when `current.json` names that generation, else the newest `manifest.*.json` listed in the store
(stores without `list` fall back to rev 0).

`index.openStats` is the cold-start timeline of `open`: `{ total, wasmInit, current, prefetch, manifest, slabs, base, pack? }`
where `current` (current.json), `manifest`, `slabs` (slabs table + tombstone sets) and `base` (routing table and
codebook) are `PhaseStat`s (`base` adds `fetch` and `build` ms and `files: {coarse, fine, pq}` bytes), `wasmInit`
is how long `Thread.init` took (`thread.initMs`, not part of `total`) and `pack`, for a `PackedBlobStore`, is its
own open (`head` + tail + directory reads, `PackedBlobStore.openStats`; it happened before `open`). `prefetch` is
how the open's objects were fetched up front: `source` `"base-pack"` (the revision's base pack, `key`),
`"index-pack"` (a `PackedBlobStore`'s base range, one range read), `"files"` (no pack: all requested once the
manifest was in) or `"none"` (`basePack: false`), with the pack read's work; objects read early are counted in the
step that uses them. A "read" is a
request that reached the origin store; with a cache in front, `hits`/`misses` count cache lookups. Python's
`index.open_stats` has the same fields except `wasmInit`. `bench/cold/` measures these against S3.

Properties: `index.manifest`, `index.gen`, `index.rev`, `index.dim`, `index.cells` (`k1 * k2`), `index.format`
(1 int8 only, 2 sq4 + sq8, 3 pq + sq8), `index.defaults` (`{nprobe, rerank, pc}`), `index.revision` (parsed
manifest, slabs table, deltas, tombstone state), `index.routingState` (`"full"` once the routing table is
loaded, `"cold"` while queries route cold). `await index.whenFullRouting()` waits for the full table.

### `index.query(vector, options): Promise<QueryResult>`

`vector`: `Float32Array | number[]` of `index.dim` floats, any norm (normalized by the SDK).

| option | default | meaning |
|---|---|---|
| `topK` | 10 | results |
| `nprobe` | manifest default | slabs (cells) to scan |
| `rerank` | manifest default | scan-tier candidates re-scored with the int8 tier (`T = max(topK, rerank)`); 0 = scan-tier scores only. Formats 2/3; format 1 never reranks |
| `pc` | manifest default | coarse groups to probe (two-level indexes); 0 = flat routing over all cells |
| `returnPayload` | false | also return each hit's payload bytes (one extra range read per slab with payloads not already in memory) |
| `inflight` | the store's: 12 (Cache API), 32 (S3 from Node), 6 (S3 inside a Worker) | reads in flight per query |
| `speculate` | true | read the running top-T's rerank records while the last 6 slabs are still scanning |
| `beam` | `coldBeam` | cold routing only: tree groups whose cells are ranked (until the full routing table is in) |
| `filter` | - | an attribute filter (see [Filtering](#filtering)): field conditions, or the compiled words |
| `maxNprobe` | 4 x nprobe | with a filter: how many cells a selective filter may scan in all, in routing order, until `max(topK, rerank)` candidates match (`nprobe` = no widening) |
| `coalesceBytes` | the store's: 128 KB for `S3BlobStore`, else 4096 | phase 2: a slab's wanted rerank records are merged into one range while the range stays within this many bytes |
| `spanBytes` | the store's, else 0 (off) | phase 2: read a slab's records as ONE range from the first to the last wanted record when that span is at most this |
| `coldPolicy` | `OpenOptions.coldPolicy` | `{ nprobe?, rerank?, threshold? }`: when more than `threshold` (default 0.5) of the probed slabs are not cached, scan only the best `nprobe` of them and rerank `rerank` (each capped at the query's value); `false` disables it for this query |
| `coalesceAcrossSlabs` | off | `PackedBlobStore` only: merge phase-1 prefix reads of slabs within this many bytes of each other in the pack into one range (`true` = 64 KiB). Measured: no gain on today's packs (docs/cold-start.md) |

Manifests without `defaults` (g1, g2) use `nprobe 1, rerank 32, pc 0`; `params.source` says where each value
came from (`query`, `manifest`, `legacy`); with a filter `params` also carries `filter: true` and the `maxNprobe` in
effect.

```ts
interface QueryResult {
	hits: { id: number; score: number; attrs: number; payload?: Uint8Array; source: "slab" | "delta" }[];
	params: { topK; nprobe; rerank; pc; source: { nprobe; rerank; pc } };
	stats: {
		gen; rev; format; routing: "flat" | "tree" | "cold"; routingTable: "f32" | "f16" | "i8";
		cells: number[];                                                 // cells scanned (beyond nprobe when a filter widened)
		coldRouting?: { beam; groups; cells; reads; bytes; loadedGroups; loadedCells };  // cold routing: means this query read
		filterDropped; candidates;                                       // entries the filter dropped, distinct scan-tier candidates
		scanned; scanBytes; slabBytes;                                   // phase 1
		rerank; rerankCandidates; rerankSlabs; rerankReads; rerankBytes; rerankLocal; specRecords; specUsed;  // phase 2
		deltas; deltaBytes; deltaScanned; deltaHits; deltaDropped; deltaTop; tombstones; tombSets; tombDropped;
		io: { reads; hits; misses; r206; r200; bytes; warms };            // store accounting (cache hits/misses, origin reads, background warms queued)
		wholeSlabs; inflight; coalesceBytes; spanBytes;                   // slabs read whole (records local); the I/O knobs used
		cold?: { misses; probed; applied };                              // with a cold policy: probed slabs not cached, and whether it applied
		limiterRetries; attempt; wasmResets; wasmMemMB;
		timing: { total; route; phase1; phase2; payload };               // ms; CPU-only phases read ~0 in Workers
		timeline: { route; phase1; phase2; payload; base? };             // PhaseStat per phase (below)
	};
}
```

`stats.timeline` splits the query into phases, each a `PhaseStat` `{ ms, reads, hits, misses, r206, r200, bytes }`:
its wall time and the store work that completed during it (the query's trace snapshotted at the phase
boundaries, so a speculative read issued in phase 1 that lands in phase 2 counts in phase 2). `base` is present
when this query (re)loaded the routing table (after a wasm reset or a switch of generation).

Scores are cosine similarities: reranked slab hits and delta hits are on the int8 scale and merge directly;
with `rerank: 0` they are scan-tier (4-bit or PQ) estimates.

`stats.trace` is the query's trace envelope, the same JSON in this SDK, the Python SDK and the Worker: spans (the
phases above), store work, counters (`pushed`: entries that entered a scan's heap), and the CPU of each wasm kernel
(`route`, `scan`, `deltaScan`, `merge`, `rerank`; 0 inside Workers, whose clock stops during CPU work).
`traceEvents: true` adds one event per store read. `search`, `lexicalSearch`, `get`, `listKeys`, `upsert` and
`delete` return one too, `index.openTrace` is the open's, and `open({ onTrace, traceSample })` hands a sampled share
of them to a function. See [docs/observability.md](../../docs/observability.md).

`index.queryMany(vectors, options)` runs the queries concurrently against the same snapshot and routing table.

### `index.search(request): Promise<SearchResult>`

Text, vector or both (see [docs/lexical.md](../../docs/lexical.md)). A text needs `manifest.lex` (`configureLexical`),
a vector a dense generation.

| field | default | meaning |
|---|---|---|
| `text` | - | the query text, tokenized with the index's tokenizer, BM25 over the lexical segments; also embedded for the dense list when the index can embed (below) and no `vector` is given |
| `vector` | - | the query embedding (`query` with `dense` options) |
| `embed` | when `index.canEmbed` | whether to embed `text` for the dense list; `true` fails when the index cannot embed, `false` keeps the text BM25-only |
| `topK` | 10 | results |
| `recallK` | `lex.defaults.recallK`, else max(topK, 50) | depth of each list before fusion |
| `rrfK` | `lex.defaults.rrfK`, else 60 | RRF constant |
| `weights` | `lex.defaults.weights`, else 1 / 1 | `{dense, lexical}` list weights |
| `dense` | - | `QueryOptions` of the dense list (nprobe, rerank, maxNprobe, ...) |
| `docs` | false | `true`: resolve each hit's metadata through the `docs` store; `"text"`: the stored text too |
| `filter` | - | an attribute filter applied to both lists (see [Filtering](#filtering)) |

```ts
interface SearchResult {
	hits: { id; score; dense?: { rank; score }; lexical?: { rank; score }; doc?: { id; meta; attrs; text? } | null }[];
	mode: "dense" | "lexical" | "hybrid";        // one list is returned as is; two are fused with RRF
	params: { topK; recallK; rrfK; weights };
	dense?: QueryResult;                         // the dense query's full result
	lexical?: { hits; tokens; found; stats: { segments; numDocs; totalLen; df; io: { rounds; ranges; bytes }; ms } };
	stats: { docsIo?; timing: { total; dense; lexical; docs } };
}
```

`index.lexicalSearch(text, k, {filter?})` is the BM25 list alone, `index.tokenize(text)` the query's tokens,
`index.docs(ids, {text})` the document store (`Map<id, {id, meta, attrs, text?}>`, newest live copy of each id,
tombstone-aware). `index.lexConfig`, `index.segments`, `index.attrSchema` and `index.hasDense` describe the
generation.

### Embedding (`manifest.embedding`)

A generation created for a model ([`createGeneration`](#write-path-experimental), the Worker's `POST /create`)
names it in `manifest.embedding` (`EmbeddingConfig`: `provider`, `model`, `dim`, `queryPrefix`, `docPrefix`,
`baseUrl`, `options`, `maxBatch`). The SDK never runs a model; `open({ embed })` says how to call it:

```ts
const index = await thread.open({ bucket: env.BUCKET, embed: { ai: env.AI } });            // provider "workers-ai", the binding
const index = await thread.open({ store, embed: { accountId, apiToken } });                 // "workers-ai" outside Workers (REST API)
const index = await thread.open({ store, embed: { apiKey, headers?, fetch? } });            // "openai": any OpenAI-compatible /embeddings
const index = await thread.open({ store, embed: (texts, kind) => myModel(texts, kind) });   // "custom", or to override any provider
index.embedding;                                  // EmbeddingConfig | null: the generation's model
index.canEmbed;                                   // true when a model is named AND open gave a way to call it
await index.embed(["a query"], "query");          // Float32Array[]: unit-normalized, with the kind's prefix (ThreadError 400 if it cannot)
```

With `canEmbed`, `search({ text })` without a `vector` embeds the text as a `"query"` (and still runs BM25 when the
generation has a lexical index), and `upsert({ texts })` without `vectors` or `embed` embeds them as `"document"`s,
one per chunk with `chunk`. Both take `embed: false` to skip that. Without `canEmbed` the index behaves as before:
`search` needs a `vector`, `upsert` needs `vectors` or an `embed` function. Provider failures are `ThreadError`
502 with the HTTP status. `resolveEmbedder(config, options)` is the function behind this, usable on its own; the
design, the providers' details (Workers AI `pooling`) and what a created generation lacks compared with a k-means
build are in [docs/embedding.md](../../docs/embedding.md).

### Filtering

Every document carries a 32-bit attribute word, stored with its vector (inside the scan prefix every query reads)
and in its `docs` entry; a filter is a condition on that word, applied inside the scan before an entry can take a
candidate slot. A schema names fields inside the word (design and knobs: [docs/filtering.md](../../docs/filtering.md)):

```ts
await index.configureAttrs({ tenant: { bits: 12 }, lang: { values: ["en", "fr", "de"] }, public: "bool", year: { bits: 7, min: 1970 } });
await index.upsert({ ids, vectors, texts, metas, attrs: [{ tenant: 42, lang: "en", public: true, year: 2024 }, { tenant: 7 }] });
const r = await index.query(embedding, { topK: 10, filter: { tenant: 42, lang: { $in: ["en", "fr"] }, year: { $gte: 2020 } } });
const h = await index.search({ text: "deploy failed", vector: embedding, filter: { public: true }, docs: true });
index.unpackAttrs(r.hits[0].attrs);      // { tenant: 42, lang: "en", public: true, year: 2024 }
```

| in a filter | fields | meaning |
|---|---|---|
| `field: value`, `{ $eq }`, `{ $ne }` | all | equal / not equal |
| `{ $in: [...] }`, `{ $nin: [...] }` | all | in / not in a list |
| `{ $lt, $lte, $gt, $gte }` | number | a range (combined into one test) |
| `{ $any, $all, $none }` (a bit mask) | number | some / every / no bit of the mask set |

Conditions on different fields are ANDed; there is no `$or`. Field specs: `{ bits }` (an unsigned integer, with
an optional `min`), `{ values: [...] }` (an enum) or `"bool"`; the fields are laid out in order and must fit 32
bits. A later `configureAttrs` may add fields or enum values but never changes an existing field's layout. Without a
schema the one field is `attrs`, the raw word (`filter: { attrs: { $any: 1 } }`); `attrs` in an upsert may always be
given as raw numbers.

What it costs: the dense scan reads nothing extra and tests the word only for candidates that would have entered
the top-T. A selective filter (few matches among the `nprobe` cells) makes the query scan more cells in routing
order, in batches, until `max(topK, rerank)` candidates match or `maxNprobe` cells were scanned (default 4 x nprobe):
`stats.cells` lists what was scanned. The BM25 list is filtered after scoring: it is taken deeper (4 x k, then 4x
more up to 65 536) and each candidate's word is resolved through the `docs` store, one bucket read per candidate.
`index.compileFilter(expr)` returns the compiled words a filter turns into (reusable across queries).

### `index.refresh(): Promise<boolean>`

Re-reads `current.json` (or the generation's head) and switches to the new revision if it changed (true when it
did). Before switching it loads what the first query on the new revision would otherwise wait for: the manifest,
slabs table and tombstone sets, the routing table/codebook when the revision names other base files, and the new
delta segments; then it warms the rest of what the new revision adds (`snapshotKeys(new)` minus `snapshotKeys(old)`:
new slab revisions as scan prefixes, lexical segments) into the cache in the background, within `prefetchBytes`.
Revisions are immutable, so everything is cached per index. The periodic check (`refreshInterval`) runs this in the
background: the query that notices a new revision answers from the current one instead of waiting (on g4 through a
25 ms-RTT S3, ~120 ms saved on that query), and the next queries read the new one. New data is visible within
`refreshInterval` (plus the load), at once after `await refresh()`, or at once on the handle that wrote it.

### `index.warm(keys?, { budget?, concurrency? })`

Reads `keys` (default: every object of the current revision) through the store so its cache holds them: slabs as a
`fill` read of their scan prefix (the miss policy decides whether the whole slab is cached now or in the
background), everything else whole, skipping cached keys and stopping slab prefixes / segments at `budget` bytes.
Returns `{ objects, bytes, skipped }`. To fill a `DiskCache` with a whole revision ahead of time, copy instead:
`copyObjects(s3, diskCache, snapshotKeys(index.revision))`.

### `index.close()`

Frees the wasm objects (Searcher, PqCodebook). Queries in flight finish first; the index cannot be used
afterwards.

### Write path (`@experimental`)

```ts
await index.configureLexical({ text: true, lengths: "exact", bucketTerms: 256 });   // once: manifest.lex (a revision)
await index.configureAttrs({ tenant: { bits: 12 }, lang: { values: ["en", "fr"] } });                // once: manifest.attrs (a revision)
const r = await index.upsert({ ids: [1000, 1001], vectors: [embA, embB], texts: ["...", "..."], metas: ["Title A", "Title B"], attrs: [{ tenant: 5, lang: "en" }, 6] });
// { gen, rev, seq, count, cells, distinctCells, keys: { delta, segment, tombstones, manifest }, bytes, conflicts, casConflicts, currentUpdated, ms }
await index.upsert({ ids: [1002], texts: ["text-only document"], metas: ["C"] });   // no vector: found by text alone
await index.delete([1000]);           // { gen, rev, count, key, conflicts, ... } (hides the vector, the text and the doc)
// by key (needs manifest.lex): known keys keep their id, new keys are allocated from manifest.nextId; r.ids lists them
const k = await index.upsert({ keys: ["s3://b/a.bin", "s3://b/c.bin"], metas: ["A", "C"], texts: ["host:cdn-1", "host:cdn-2"] });
const { records } = await index.get(["s3://b/a.bin"], { text: true });   // Map<key, { key, id, meta, attrs, text? }>
const { ids } = await index.resolveKeys(["s3://b/a.bin"]);              // Map<key, id>
const page = await index.listKeys({ prefix: "s3://b/", limit: 100 });   // { entries: [{ key, id }], more, next }; continue with { after: page.next }
await index.deleteKeys(["s3://b/c.bin"]);                                // { count, missing, ... }
// one side only: a vector for an existing key (text, meta, key untouched), or new text without the vector
await index.upsert({ keys: ["s3://b/a.bin"], vectors: [emb], scope: "dense" });
await index.upsert({ keys: ["s3://b/a.bin"], texts: ["host:cdn-2"], metas: ["A2"], scope: "lexical" });
// long texts: chunked into windows of `words` words (default 200) sharing `overlap` (default 0), one document per chunk keyed `k#i`
// with the parent's meta/attrs; `embed` is called once with the chunk texts and returns their vectors (r.chunks = { documents, count, stale })
await index.upsert({ keys: ["docs/a.md"], texts: [longText], metas: ["A"], chunk: { words: 200, overlap: 20 }, embed: async (texts) => embedAll(texts) });
chunkText(longText, { words: 200, overlap: 20 });                       // [{ index, text, start, end, words }, ...], the same windows
await index.compactSegments();        // merge the lexical/docs segments into one base segment (a revision)
await index.compactSegments({ maxInputBytes: 4 << 20 });   // or merge one contiguous run that fits 4 MB (memory-limited writers)
await index.commitMergedSegment({ run, readRev, key, bytes, count, keys }); // commit a run merged by another process (merged 0: the run changed)
// dense compaction in this isolate: fold delta segments into slabs, split cells over maxCellEntries, drop tombstone sets
// a generation built beside the live one, then swapped in with a compare-and-swap of idx/current.json
await thread.createGeneration({ store, gen: "dense", embedding, lex: { text: true }, setCurrent: false });
const next = await thread.open({ store, gen: "dense", embed: { ai: env.AI } });     // writes commit to "dense", the pointer stays
await setCurrent(store, { gen: "dense", expect: { gen: "lex", rev: liveRev } });    // 409 unless the pointer is exactly that
await gcGeneration(store, { gen: "dense", keep: 2, graceSec: 300, dryRun: false }); // superseded objects of a generation (/build/gc)
await dropGeneration(store, { gen: "lex", graceSec: 300, dryRun: false });          // a whole generation the pointer left
if (index.compactionDue().due) {                            // no I/O: the open revision against manifest.compaction
	let r;
	do r = await index.compact({ maxBytes: 48 << 20 });      // { outcome, rev, cells, folded, splits, k, more, estimatedPeakBytes, limits, ... }
	while (r.more);
}
index.compactionLimits(32 << 20);                           // { cellEntries, splitEntries, upsertVectors, ... }: what that budget can process
await createLexicalGeneration({ store, gen: "docs1", lex: { text: true } });   // a text-only generation (no dense index)
// a dense generation from nothing, for an embedding model (one empty cell; grows by compaction and cell splits)
await thread.createGeneration({ store, gen: "docs2", embedding: { provider: "openai", model: "text-embedding-3-small", dim: 1536 }, lex: { text: true } });
const ix = await thread.open({ store, embed: { apiKey } });   // ix.canEmbed; ix.embedding; await ix.embed(texts, "query" | "document")
await ix.upsert({ keys: ["docs/a.md"], texts: [longText], chunk: true });     // no vectors: embedded as documents (embed: false = text-only)
await ix.search({ text: "deploy failed", topK: 10, docs: true });            // embedded as a query + BM25, fused (embed: false = BM25 only)
```

`manifest.lex` (`LexConfig`): `tokenizer` (frozen once a segment exists), `k1`, `b`, `lengths: "exact" | "byte"`,
`bucketTerms`, `docs`, `text`, `bucketDocs`, `bucketKeys`, `defaults: {rrfK, recallK, weights}`; see docs/lexical.md
for what each one trades. Keys (non-empty strings of at most 4096 bytes, unique per batch) are stored in the
segment's `keys` section and resolved newest segment first; the design and the key-value index case are in
[docs/keys.md](../../docs/keys.md). `chunk` splits each text into flat word windows before writing (needs `keys`
and `texts`; vectors come from `embed`; chunk keys the parent no longer produces are tombstoned in the same
revision); the rule, the key scheme and the limits are in [docs/chunking.md](../../docs/chunking.md).
`scope` (default `"both"`) replaces one side of the documents: `"dense"` writes the delta and a tombstone set that
hides only older dense copies (keys must exist; no segment, so docs, texts and keys stay; `texts` only feed the
embedder; no `metas`/`chunk`), `"lexical"` writes the segment and a set that hides only older lexical copies (no
`vectors`/`payloads`/`embed`/`chunk`; absent `attrs` keep the live word, a changed word is refused in a dense
generation). Deletes stay unscoped. Scoped sets mark the manifest `features: ["scoped-tombstones"]`, which readers
before 0.2.1 ignore ([docs/update-path.md](../../docs/update-path.md#scoped-tombstone-sets-one-sided-updates)).
`createGeneration` (also `createGeneration(thread.wasm, opts)` and the Worker's `POST /create`) publishes rev 0
with one empty cell for `embedding` (or a bare `dim`), `lex` (default `{}`; `null` for ids only), `attrs`, a
`profile` (`edge` splits cells at 3000 entries, `local` at 1000), `defaults` and the automatic `compaction` policy;
what that generation lacks compared with a k-means build is in [docs/embedding.md](../../docs/embedding.md).
`means` (`k x dim` floats) gives rev 0 `k` empty cells instead of one. For a large load, `thread.bulkBuild({ store,
gen, embedding, lex, attrs, maxBytes })` (`@experimental`) stages documents with `b.add({ keys, metas, texts, attrs,
vectors })`, and after `b.seal()` each `b.step({ deadlineMs })` advances sample, k-means, create, load and
compaction within `maxBytes`, resuming from `build/{gen}/state.json`, until `done` (a step that could commit nothing
returns at once with `progressed: false`); then `setCurrent` and `b.cleanup()`. `b.close()` releases the handle's
index and keeps the build, which `thread.bulkBuild` resumes ([docs/embedding.md](../../docs/embedding.md#a-bulk-build-in-bounded-steps)).

`index.compact(opts)` (also `compactDeltas(writerContext, opts)`) is the Worker's dense compaction in one isolate:
one call folds a slice of delta segments (whole segments, oldest first) into new slab revisions, splits the
rewritten cells over `maxCellEntries` (2-means; new fine means, `k2` + splits, the centroid tree rebuilt when the
head has one) and drops the tombstone sets nothing references. A call is bounded by `maxBytes` (estimated peak
memory, default 48 MiB; `r.estimatedPeakBytes` is the plan's figure, an upper bound of what was measured),
`cellsPerRun` (default the policy's, 256) and `deadlineMs` (default the policy's, 27 s: no cell starts after
`deadline - 2.5 s`, up to 7 s with splits by the size of the fine means; the finished cells whose deltas are
complete are committed, `outcome: "partial"`). `maxBytes` also sizes the cells: they split at `min(maxCellEntries, splitEntries / 2)`, where
`splitEntries` is the largest cell the budget can split on its own, and no fold takes a cell past `splitEntries` (a
cell in the way of a delta is split first, `r.presplit`). So every cell stays processable by a later call with the
same budget; `index.compactionLimits(maxBytes)` / `compactionLimits(manifest, { maxBytes })` tells the numbers
without I/O (at dim 1024 and 32 MiB: cells split at ~960 entries, an upsert of up to ~800 vectors always folds).
Keep one budget per generation. Input no call with the budget can take (a delta with more entries in one cell than
it can fold, a cell an older writer let grow past it, a delta over `cellsPerRun` cells) is a `ThreadError` with
status 413 that names it and what it needs, not an outcome. `r.more` says whether another call has work;
`r.outcome` is `committed`, `partial`, `noop`, `blocked` (only when a slice's splits were deferred and nothing
else could be done; `more` is true), `locked` (another run holds `idx/{gen}/compact.lock`), `deadline` or
`conflict` (another writer changed what the slice depends on; call again). Without `force` a head below every threshold is a `noop`, as in the Worker's
automatic runs; `force: true` folds everything the budgets allow and also splits oversized cells. Other options:
`concurrency` (cells in flight, default 4), `maxCellEntries`, `splitCellsPerRun`, `purge` (rewrite the cells that
keep tombstone sets alive; default when the tombstone threshold is exceeded), `lock` (default true), `owner` (a
stable lock owner, e.g. a Durable Object id, takes over its own lock after a crash at once instead of after 10
minutes), `record` (append the run to `compact.last.json`, default true). The objects are the Worker's, byte for
byte (`test/compact-worker.test.ts` compacts one index both ways), so the Worker and the SDK can compact one index;
a run commits through the manifest create-only put and is safe to repeat after a crash. Details:
[docs/update-path.md](../../docs/update-path.md#compaction-in-the-sdk).

`setCurrent(store, { gen, rev?, expect? })` (or `index.setCurrent`) moves `idx/current.json` to any generation with
an etag compare-and-swap; `expect: {gen, rev}` (or `null`: no pointer) makes it fail with a 409, writing nothing,
unless the pointer is exactly that. `gcGeneration(store, { gen, keep = 2, graceSec = 300, dryRun = true })` (or
`index.gc`) is the Worker's `/build/gc`, and `dropGeneration(store, { gen, graceSec = 300, dryRun = true })` deletes
a whole generation once the pointer left it and nothing wrote to it for `graceSec`. Both need a store with
`listObjects` and `delete` (R2 binding, `S3BlobStore`, `MemoryBlobStore`, `FileBlobStore`). Details:
[docs/update-path.md](../../docs/update-path.md#switching-generations).

Implemented against [docs/update-path.md](../../docs/update-path.md): an upsert encodes one delta segment
(`DeltaJob`: primary cell + residual codes, PQ codes with the generation's codebook) and, unless
`replace: false`, a tombstone set of its ids (`tombstones_merge`), then commits manifest `rev + 1` with a
create-only put (`manifest.{rev+1}.json`, the writer lock; a lost race rebuilds on the new head and reuses the
segment) and advances `idx/current.json` with an etag compare-and-swap. The handle then reads the new revision.
Needs a store with `put` and `head` (R2 binding, `S3BlobStore`, `MemoryBlobStore`); `list` is used after a lost
race. Up to 4096 vectors per call; an upsert holds the fine means twice (JS + wasm, ~25 MB each at 4096 x 1536),
so write from a Worker that does not also serve many queries. The Worker's `/upsert` and `/delete` landed
concurrently: the two write the same objects but have not been cross-verified yet (see "Experimental" below).

### `BlobStore`

```ts
interface BlobStore {
	get(key, range?: { offset; length }, opts?: { trace?; fill? }): Promise<Uint8Array | null>;   // null = missing
	getRanges?(key, ranges, opts?): Promise<Uint8Array[] | null>;  // one request, several ranges; null = cannot (cache miss)
	getTail?(key, length, opts?): Promise<{ bytes; size; etag? } | null>;  // last `length` bytes + size in one request (bytes=-N); packs open with it
	head(key): Promise<{ size; etag } | null>;
	put?(key, data, opts?: { ifAbsent?; ifMatch?; contentType? }): Promise<{ size; etag } | null>;  // null = condition failed
	list?(prefix): Promise<string[]>;
}
```

- `R2BlobStore(bucket)`: the binding (`R2BucketLike`, structural). Conditional puts use `onlyIf`.
- `S3BlobStore({ endpoint, bucket, region, accessKeyId, secretAccessKey, sessionToken?, pathStyle?, fetch?, inflight?,
  coalesceBytes?, spanBytes?, chunkBytes?, chunkConcurrency?, warmup? })`: plain `fetch` + SigV4 (WebCrypto), single-range GETs, HEAD, PUT with
  `If-None-Match: *` / `If-Match`, ListObjectsV2. S3 has no multi-range GET, so a rerank read costs one request per
  record run. The derived signing key is cached per (date, region, service) (`SigningKeyCache`): a request costs
  one SHA-256 and one HMAC (26 us in Node instead of 130 us). `inflight` defaults to 32 (Node: no connection cap)
  or 6 inside a Worker; `coalesceBytes` to 128 KB (rerank records read as runs of up to 128 KB: half the requests
  of a cold query for ~45% more bytes). A whole-object GET of an object larger than `chunkBytes` (8 MiB) is split
  into parallel range GETs (`chunkConcurrency` 4 beside the first, started as soon as the first response's headers
  give the size), since one S3 stream is bandwidth-bound and the routing table is 25 MB on g4. `store.warmup(n)` opens `n` connections with one-byte GETs of `idx/current.json` (not HEADs: Node's `fetch` closes a connection after a HEAD answer with a Content-Length); `new S3BlobStore({ ..., warmup: 32 })` starts that at construction, so creating the store before `Thread.init` overlaps the handshakes (and Node's lazy `fetch` globals) with the wasm compile.
- `CachedBlobStore(inner, cache, { origin, waitUntil, cacheable, missPolicy, warmConcurrency, warmQueue, blockBytes, blockBuffers })`: a cache
  in front of any store, either the Workers Cache API or an in-process `BlobCache`. Objects under `idx/` are
  immutable, so they are cached whole (Cache API: for a year under `${origin}${key}`; `current.json` is never
  cached) and ranges are served from them (the Cache API answers a 206, `multipart/byteranges` for several ranges,
  parsed by the SDK). A phase-1 miss follows `missPolicy` (next section). Over the Cache API, a range read that
  passes the object's `size` and is not a `fill` (a lexical segment over 2 MiB) is served from cached aligned blocks
  of `blockBytes` instead (a miss reads and caches that block only; `blockBuffers` bounds the blocks in memory;
  `store.blockStats` counts the fills). `store.drain()` resolves once background warms and block fills are done;
  `store.warmStats` counts the warms.
- `MemoryCache(maxBytes = 256 MB)`: an LRU of whole objects in memory (objects over a quarter of it are not kept).
  A hit is a view into the cached slab, so the query also has that slab's rerank records.
- `MemoryBlobStore()`: a Map, with etags and conditions (tests, tiny local indexes).
- `PrefixedBlobStore(inner, prefix)`: what `open({ prefix })` wraps.
- `PackedBlobStore.open(inner, key)`: the objects of a pack (one file holding a whole snapshot) under their
  original keys, each read as a range of the pack. Read-only. `PackedBlobStore.fromBytes(bytes)` for a pack in
  memory. Put a cache in front of it, not behind it (`open` does).
- From `thread-sdk/node` (Node only): `FileBlobStore(root)`, an index in a directory (the layout Python's
  `FileBlobStore` reads and writes; atomic puts, create-only via hard link, CAS under a lock directory);
  `storeAt(path)`, a `FileBlobStore` for a directory or a `PackedBlobStore` for a pack file;
  `packIndexToFile(store, file, { gen?, rev? })`; `DiskCache(dir, { maxBytes = 4 GB, missPolicy })`, a size-bounded
  cache of whole objects on disk (next section); `nodeFetch({ keepAliveTimeout, allowH2, connections })`, a `fetch`
  for `S3BlobStore` with a longer keep-alive.

### Caches and the cold path

A cold query is one whose slabs are not cached. What its misses read is the cache's **miss policy**:

| `missPolicy` | a slab-prefix miss reads | phase 2 for that slab | right for |
|---|---|---|---|
| `"whole"` (Cache API default) | the whole slab (~1.7 MB on g4), cached | free: the winners' records are in memory | a cache many requests share: the edge cache pays the transfer once for everybody |
| `"range"` | exactly the scan prefix (~200 KB), nothing cached | one range read per record run | no caching at all (measurement, memory-starved processes) |
| `"range+warm"` (`MemoryCache`/`DiskCache` default) | the prefix now; the whole slab into the cache in the background | one range read per record run | an in-process cache: the first query does not wait for 57 MB, later queries hit |

Background warms are deduplicated, bounded (`warmConcurrency`, default 2, only 1 while foreground reads are in
flight; `warmQueue` 1024) and handed to `waitUntil`. Reads without `fill` (rerank records, postings) never fetch whole
objects; `PackedBlobStore` translates every read, whole-object warms included, into a range of the pack (it never
asks its own store for the pack).

What each runtime gets by default:

| | cache | miss policy | `inflight` | warm-up |
|---|---|---|---|---|
| Worker, R2 binding or S3 | Cache API | `"whole"` | 12 | off |
| Worker, `cache: false`, S3 directly | none | - | 6 | off |
| Node, `S3BlobStore` | `MemoryCache` 256 MB | `"range+warm"` | 32 | `inflight` connections |
| Node, `FileBlobStore` / pack file | none | - | 12 | - |

Before this, Node put no cache in front of S3 (`caches` does not exist there), so every query was cold. Everywhere,
`open` loads the f32 routing table by default (`routing: "i8"` is the opt-in that cuts it 4x and fetches the base
pack), starts every base read as soon as the manifest is known, and skips the manifest GET when `current.json`
carries an inline copy. Measured end to end: [docs/cold-start.md](../../docs/cold-start.md#summary).

**Disk cache.** `new DiskCache(dir)` keeps whole objects as files in the bucket layout (`{dir}/idx/g4/slab/00123`),
the same layout Python's `DiskCache` uses (the two SDKs can share a directory), evicting least recently used
objects beyond `maxBytes` (order in memory, and in the files' mtimes across restarts). A process restart is then not
a cold start. A `DiskCache` is also a `FileBlobStore`, so warming a laptop from a bucket is a copy:

```ts
import { DiskCache, FileBlobStore } from "@aiuitech/thread-sdk/node";
const disk = new DiskCache("/var/cache/thread", { maxBytes: 8 << 30 });
await copyObjects(s3, disk, snapshotKeys(index.revision));            // one revision, ~7 GB on g4
const index = await thread.open({ store: s3, cache: disk });           // or offline: { store: new FileBlobStore(dir), gen, rev }
```

**Connections.** Node's `fetch` (undici) speaks HTTP/1.1 (no HTTP/2 unless an `Agent` with `allowH2` is used), one
connection per request in flight, and keeps idle connections for 4 s. A cold query with `inflight` 32 opens up to 32
connections; `open` warms them while it loads the index (`warmup`) so the first query does not pay 32 TLS
handshakes. Queries more than 4 s apart reconnect: `new S3BlobStore({ ..., fetch: await nodeFetch({ keepAliveTimeout:
60_000 }) })` keeps connections for a minute.

**Cold-path policy.** `coldPolicy: { nprobe: 20, rerank: 100 }` (on `open` or per query) scans fewer slabs when most
probed slabs are not cached (`threshold`, default more than half), trading recall for latency only on cold
queries; `stats.cold` says whether it applied. Off by default. Residency is known for in-process caches; for the
Cache API only for keys this isolate has seen, and a store without a cache counts as cold when remote.

A store may return views into a larger buffer (the whole object on a miss or a memory-cache hit); the SDK detects
that for slab prefixes and uses the rest of the object instead of re-reading it (`stats.wholeSlabs`). Over the
Cache API, only `fill` reads return such views: other range reads (blocks, ranges of an object cached whole) return
the range in a buffer of its own.

### Snapshots and packs

```ts
import { copyObjects, packIndex, packToBytes, snapshotKeys, MemorySink } from "@aiuitech/thread-sdk";
import { FileBlobStore, packIndexToFile, storeAt } from "@aiuitech/thread-sdk/node";

const keys = snapshotKeys(index.revision);                       // every object the open revision reads, sorted
await copyObjects(src, dst);                                      // everything src lists -> dst (any two stores)
await copyObjects(src, dst, ["idx/current.json", ...keys]);       // one consistent revision
await copyObjects(src, dst, undefined, { skipExisting: true, concurrency: 16 });

const r = await packIndexToFile(src, "idx.threadpack", { gen: "g4", rev: 12 });   // { gen, rev, count, bytes, keys }
const { bytes } = await packToBytes(src);                         // the same pack in memory (small indexes)
await packIndex(src, sink);                                       // any PackSink: { write(bytes) }
```

`packIndex` writes `idx/current.json` (pointing at the packed revision), then the objects a cold open reads
(`baseKeys`: manifest, slabs table, tombstone sets, coarse, codebook, tree, the smallest routing table) and then the
rest of `snapshotKeys` in key order, and records that leading range as `base: [offset, length]` in the directory:
`thread.open` over a `PackedBlobStore` loads it with one range read. The two SDKs produce identical files for the
same revision. `buildBasePack(store, revState, "base" | "cold", routing)` builds a revision's base pack (what publish
writes as `idx/{gen}/base.{rev}.threadpack` / `base.cold.{rev}.threadpack`; `bench/cold/make_base_pack.mjs` offline). A pack is read-only: `copyObjects(pack, dst)` into a
directory or a bucket to write on top of it. Format and rationale: [docs/portability.md](../../docs/portability.md).

## Memory and limits

- The wasm linear memory never shrinks. Routing table 25 MB (4096 x 1536; f16 / i8 tables are decoded to the same
  25 MB: they save download, not memory) + codebook 1.5 MB per generation, one
  PqTable (192 x 256 x 4 B) per query, one slab prefix (~200 KB at nprobe 34 on g4) copied into wasm per `scan`
  call and freed right after. A query holds its prefixes only until scanned; on a cold Cache API (`"whole"`) every
  slab is read whole (~1.7 MB each on g4), so a cold query transfers ~57 MB and keeps ~T x 1.5 KB of rerank
  records; with `"range+warm"` it transfers the prefixes and records (~6.5 MB) and the slabs follow in the
  background into the `MemoryCache` (256 MB holds ~150 g4 slabs).
- `inflight` 12 Cache API reads per query is safe; origin (R2/S3) reads beyond 6 in flight inside a Worker may be
  cancelled by the runtime ("connection limit") and are retried once, hence the default of 6 for S3 there.
- An isolate serving several indexes keeps one base per index; `close()` what you no longer need, or keep one
  `ThreadIndex` per store at module level (the example does).
- Lexical segments: one wasm `SegmentReader` per segment per isolate holds the container table, section headers
  and directories (a few KB per segment) plus up to `lexTransientBytes` (8 MB) of recently read buckets and posting
  lists before they are evicted. The wasm module is ~750 KB with the tokenizer, stemmer and lexical kernels.
- Payloads: `returnPayload` reads them with one (multi-)range request per slab; keep payloads small or store
  them elsewhere keyed by id.
- `index.compact()`: the planner keeps its estimate under `maxBytes` (48 MiB by default): 1 MiB fixed, the folded
  deltas held for the run, the tombstone sets (x3), the slabs table (x2), the largest folded delta again (its copy
  in wasm) and, per cell in flight, 3x its old slab plus 6x the new one (a split: 8x plus 4 B x entries x dim of
  decoded vectors); or, when it splits, the fine means of the commit (x3), whichever phase is larger. The factors
  are measured, wasm memory growth included ([docs/update-path.md](../../docs/update-path.md#memory-of-a-call)).
  Reads and writes go under the caches. The wasm memory keeps its high-water
  mark afterwards, so run it in the isolate that writes (a Durable Object), not one that serves many queries.

## Vectorize mapping

| Vectorize | thread-sdk |
|---|---|
| `env.VECTORIZE.query(vector, { topK, returnMetadata })` | `index.query(vector, { topK, returnPayload })`: `hits[].id/score`; `attrs` (u32, `unpackAttrs` for the fields) and `payload` (bytes) are the metadata |
| `matches[].id` (string) | `hits[].id` (u32) |
| `upsert([{ id, values, metadata }])` | `index.upsert({ ids, vectors, attrs?, payloads? })` (replace semantics by default; `attrs` as field values under `configureAttrs`) |
| `insert(...)` | `index.upsert({ ..., replace: false })` |
| `deleteByIds(ids)` | `index.delete(ids)` |
| `getByIds` | not available (no id -> slab map; keep ids -> data in your own store) |
| `query(vector, { filter: { field: { $eq | $ne | $in | $nin | $lt | $lte | $gt | $gte } } })` on indexed metadata | `index.query(vector, { filter })` with the same operators (plus `$any`/`$all`/`$none` bit tests) over fields packed into the 32-bit word (`configureAttrs`); no `$or` |
| `returnMetadata` | `search({ docs: true })`: the `docs` store's metadata string (and text) per hit |
| (no equivalent) | `search({ text })`: BM25 and hybrid search over the same ids |
| `describe()` | `index.manifest`, `index.dim`, `index.cells`, `index.defaults` |
| namespaces | one index per `prefix` (or per bucket) |
| cosine / euclidean / dot | cosine only (vectors are normalized) |
| consistency | a query reads one immutable revision; writes are visible after `refresh()` / within `refreshInterval` |

## Experimental

- `upsert` / `delete`: the SDK writes delta segments, tombstone sets and manifest revisions exactly as
  docs/update-path.md specifies and as its own tests check (Node, MemoryBlobStore, wasm), but it has not yet
  been run against a revision written by the Worker and vice versa. Points to reconcile: the Worker writes
  `change.op` fields the SDK copies (`upsert`/`delete` + `writer: "@aiuitech/thread-sdk"`); the Worker's `/upsert` also
  accepts `firstId`, base64 vectors and `{b64}` payloads, which the SDK does not need; compaction and GC are
  Worker-only (`POST /build/compact`, `/build/gc`).
- Spec notes from the implementation: (1) `scan`/`scan_pq`/`scan_delta` now take the tombstone buffer and apply
  it before top-k insertion (the earlier `filter_hits` after the scan could leave fewer than k hits); the SDK
  passes `tombFor(slabRev)` for slabs and the union of sets with `rev > addedRev` for deltas. (2) The spec says
  the slabs table "may" gain a fifth column after compaction; the SDK treats a 4-column row as slabRev 0.
  (3) `current.json` has no version field; the CAS relies on the store's etag (R2 binding `etag`, S3 `ETag`).
  (4) The S3 API path serializes writers with `If-None-Match: *` / `If-Match`; R2 and AWS S3 document support,
  but it is unverified here (no S3 credentials were available, see VERIFICATION.md), so prefer the R2 binding
  for writers until checked.
- `S3BlobStore` against a live R2 S3 endpoint: signing is verified against AWS's published test vectors and a
  fake server; a live run needs an R2 API token (dashboard) in `examples/basic/.env.secrets`.

## Development

```sh
npm install
npm run sync-wasm      # copy crates/thread-wasm/pkg/{thread.js,thread.d.ts,thread_bg.wasm,...} into src/wasm
npm test               # vitest: wire formats, stores, packs, the Node file store, SigV4, end-to-end with the wasm in Node
npm run build          # dist/ (tsc + wasm assets)
cd examples/basic && npm install && cf deploy --secrets-file .env.secrets
node scripts/verify.mjs --n 50 --nprobe 34 --rerank 150    # compare the example with the main Worker
```

Releases: a `v<version>` tag on the repo publishes this package to npm (see
[docs/operations.md](../../docs/operations.md#releasing)); bump `package.json` (and the lock) together with the
Python and Cargo versions, and add a [CHANGELOG](../../CHANGELOG.md) entry first.

`examples/basic/` is a `cf init workers` project bound to the `vib-bench` bucket exposing `POST /search`
(`{vector, topK?, nprobe?, rerank?, store?: "r2" | "s3"}`), `GET /manifest`, `POST /refresh` and `GET /health`.
