# thread-sdk

Query and update a [thread](../../README.md) blob-native search index from a Cloudflare Worker: dense vectors (the
IVF slabs), BM25 text (lexical segments), or both fused by rank, with one document store for the hits. The index
lives as immutable objects in R2 (or any S3-compatible store); the SDK streams the probed slabs and the queried
posting lists through the Rust/wasm kernels, like a Vectorize binding but with the index in your own bucket,
readable through the Workers Cache API at the edge.

```ts
import wasmModule from "thread-sdk/wasm/thread_bg.wasm";
import { Thread } from "thread-sdk";

const thread = await Thread.init({ wasmModule });                   // once per isolate
const index = await thread.open({ bucket: env.BUCKET, waitUntil: (p) => ctx.waitUntil(p) });
const { hits } = await index.query(embedding, { topK: 10 });  // dense: [{ id, score, attrs, source }, ...]
const r = await index.search({ text: "deploy failed", vector: embedding, topK: 10, docs: true });
// hybrid: r.hits = [{ id, score, dense?: {rank, score}, lexical?: {rank, score}, doc?: {id, meta, attrs} }, ...]
const f = await index.search({ text: "deploy failed", vector: embedding, topK: 10, filter: { tenant: 42, lang: { $in: ["en", "fr"] } } });
```

## Install

```sh
npm install thread-sdk
```

No runtime dependencies: the package ships its TypeScript build (`dist/`) and the wasm-bindgen output of
`crates/thread-wasm` (`dist/wasm/thread_bg.wasm`, ~240 KB). Node >= 20 for the tests; the SDK itself targets the
Workers runtime (WebCrypto, `fetch`, Cache API) and runs in Node for tests and tooling.

### Importing the wasm

The SDK is bundler-agnostic: you import the `.wasm` through your bundler, which turns it into a
`WebAssembly.Module`, and hand it to `Thread.init`. The SDK handles instantiation, the panic hook and recovery
from traps.

**Cloudflare Vite plugin** (`cf init workers`, `@cloudflare/vite-plugin`): import the file by a specifier ending
in `.wasm` so the plugin's `CompiledWasm` rule applies (a bare `thread-sdk/wasm` is taken over by Vite's own wasm
handling and fails):

```ts
import wasmModule from "thread-sdk/wasm/thread_bg.wasm";
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
import { storeAt } from "thread-sdk/node";
const index = await thread.open({ store: await storeAt("/data/idx") });
const index = await thread.open({ store: await storeAt("/data/idx-g4-rev12.threadpack") });

// 4. A pack kept in a bucket, read by range requests, from a Worker or anywhere
const index = await thread.open({ store: await PackedBlobStore.open(new R2BlobStore(env.BUCKET), "packs/g4.threadpack") });

// 5. Your own BlobStore (a KV, a test fixture) - see "BlobStore" below
const index = await thread.open({ store: new MemoryBlobStore() });
```

Whatever the store, `open` puts a `CachedBlobStore` (Workers Cache API, `caches.default`) in front of it unless
`cache: false`. Pass `waitUntil` so background cache fills outlive the response.

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
| `cache` | `caches.default` | a `Cache` to put in front of the store; `false` disables it |
| `cacheOrigin` | `https://vib-sdk.cache/` | synthetic origin of the cache keys (`${origin}idx/...`); use one per store |
| `waitUntil` | - | `ctx.waitUntil`, keeps background cache writes alive |
| `refreshInterval` | 30 000 ms | a query older than this re-reads `current.json` first (0 = only `refresh()`) |
| `deltaCacheBytes` | 40 MB | delta segments kept in memory across queries (per index) |

`open` reads `current.json` (a pointer without `rev` means rev 0 = the legacy `idx/{gen}/manifest.json`), the
manifest, the slabs table and the tombstone sets, and loads the routing table (`coarse.f32` + `fine.f32`, ~25 MB
in wasm at 4096 x 1536) and the PQ codebook (format 3). `gen` without `rev` reads the generation's head: the
current rev when `current.json` names that generation, else the newest `manifest.*.json` listed in the store
(stores without `list` fall back to rev 0).

`index.openStats` is the cold-start timeline of `open`: `{ total, wasmInit, current, manifest, slabs, base, pack? }`
where `current` (current.json), `manifest`, `slabs` (slabs table + tombstone sets) and `base` (routing table and
codebook) are `PhaseStat`s (`base` adds `fetch` and `build` ms and `files: {coarse, fine, pq}` bytes), `wasmInit`
is how long `Thread.init` took (`thread.initMs`, not part of `total`) and `pack`, for a `PackedBlobStore`, is its
own open (`head` + tail + directory reads, `PackedBlobStore.openStats`; it happened before `open`). A "read" is a
request that reached the origin store; with a cache in front, `hits`/`misses` count cache lookups. Python's
`index.open_stats` has the same fields except `wasmInit`. `bench/cold/` measures these against S3.

Properties: `index.manifest`, `index.gen`, `index.rev`, `index.dim`, `index.cells` (`k1 * k2`), `index.format`
(1 int8 only, 2 sq4 + sq8, 3 pq + sq8), `index.defaults` (`{nprobe, rerank, pc}`), `index.revision` (parsed
manifest, slabs table, deltas, tombstone state).

### `index.query(vector, options): Promise<QueryResult>`

`vector`: `Float32Array | number[]` of `index.dim` floats, any norm (normalized by the SDK).

| option | default | meaning |
|---|---|---|
| `topK` | 10 | results |
| `nprobe` | manifest default | slabs (cells) to scan |
| `rerank` | manifest default | scan-tier candidates re-scored with the int8 tier (`T = max(topK, rerank)`); 0 = scan-tier scores only. Formats 2/3; format 1 never reranks |
| `pc` | manifest default | coarse groups to probe (two-level indexes); 0 = flat routing over all cells |
| `returnPayload` | false | also return each hit's payload bytes (one extra range read per slab with payloads not already in memory) |
| `inflight` | 12 | reads in flight per query |
| `speculate` | true | read the running top-T's rerank records while the last 6 slabs are still scanning |
| `filter` | - | an attribute filter (see [Filtering](#filtering)): field conditions, or the compiled words |
| `maxNprobe` | 4 x nprobe | with a filter: how many cells a selective filter may scan in all, in routing order, until `max(topK, rerank)` candidates match (`nprobe` = no widening) |

Manifests without `defaults` (g1, g2) use `nprobe 1, rerank 32, pc 0`; `params.source` says where each value
came from (`query`, `manifest`, `legacy`); with a filter `params` also carries `filter: true` and the `maxNprobe` in
effect.

```ts
interface QueryResult {
	hits: { id: number; score: number; attrs: number; payload?: Uint8Array; source: "slab" | "delta" }[];
	params: { topK; nprobe; rerank; pc; source: { nprobe; rerank; pc } };
	stats: {
		gen; rev; format; routing: "flat" | "tree"; cells: number[];     // cells scanned (beyond nprobe when a filter widened)
		filterDropped; candidates;                                       // entries the filter dropped, distinct scan-tier candidates
		scanned; scanBytes; slabBytes;                                   // phase 1
		rerank; rerankCandidates; rerankSlabs; rerankReads; rerankBytes; rerankLocal; specRecords; specUsed;  // phase 2
		deltas; deltaBytes; deltaScanned; deltaHits; deltaDropped; deltaTop; tombstones; tombSets; tombDropped;
		io: { reads; hits; misses; r206; r200; bytes };                   // store accounting (cache hits/misses, origin reads)
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

`index.queryMany(vectors, options)` runs the queries concurrently against the same snapshot and routing table.

### `index.search(request): Promise<SearchResult>`

Text, vector or both (see [docs/lexical.md](../../docs/lexical.md)). A text needs `manifest.lex` (`configureLexical`),
a vector a dense generation.

| field | default | meaning |
|---|---|---|
| `text` | - | the query text, tokenized with the index's tokenizer, BM25 over the lexical segments |
| `vector` | - | the query embedding (`query` with `dense` options) |
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
did). Revisions are immutable, so everything else is cached per index: the parsed manifest, slabs table and
tombstone sets per revision; the routing table/codebook per generation (reloaded only when the revision names
other base files); delta segments by key. New data is visible within `refreshInterval`, or at once after
`refresh()`, or at once on the handle that wrote it.

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
await index.compactSegments();        // merge the lexical/docs segments into one base segment (a revision)
await createLexicalGeneration({ store, gen: "docs1", lex: { text: true } });   // a text-only generation (no dense index)
```

`manifest.lex` (`LexConfig`): `tokenizer` (frozen once a segment exists), `k1`, `b`, `lengths: "exact" | "byte"`,
`bucketTerms`, `docs`, `text`, `bucketDocs`, `defaults: {rrfK, recallK, weights}`; see docs/lexical.md for what
each one trades.

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
	head(key): Promise<{ size; etag } | null>;
	put?(key, data, opts?: { ifAbsent?; ifMatch?; contentType? }): Promise<{ size; etag } | null>;  // null = condition failed
	list?(prefix): Promise<string[]>;
}
```

- `R2BlobStore(bucket)`: the binding (`R2BucketLike`, structural). Conditional puts use `onlyIf`.
- `S3BlobStore({ endpoint, bucket, region, accessKeyId, secretAccessKey, sessionToken?, pathStyle? })`: plain
  `fetch` + SigV4 (WebCrypto), single-range GETs, HEAD, PUT with `If-None-Match: *` / `If-Match`, ListObjectsV2.
  S3 has no multi-range GET, so without a cache in front the rerank reads one range per record run.
- `CachedBlobStore(inner, cache, { origin, waitUntil, cacheable })`: Cache API in front of any store. Objects
  under `idx/` are immutable, so they are stored for a year under `${origin}${key}` (`current.json` is never
  cached). Range requests are answered from the cached full object with a 206 (`multipart/byteranges` for several
  ranges, parsed by the SDK). A phase-1 miss reads the whole slab (`fill`), caches it in the background and keeps
  the local winners' rerank records, so phase 2 costs nothing for that slab.
- `MemoryBlobStore()`: a Map, with etags and conditions (tests, tiny local indexes).
- `PrefixedBlobStore(inner, prefix)`: what `open({ prefix })` wraps.
- `PackedBlobStore.open(inner, key)`: the objects of a pack (one file holding a whole snapshot) under their
  original keys, each read as a range of the pack. Read-only. `PackedBlobStore.fromBytes(bytes)` for a pack in
  memory. Put a cache in front of it, not behind it (`open` does).
- From `thread-sdk/node` (Node only): `FileBlobStore(root)`, an index in a directory (the layout Python's
  `FileBlobStore` reads and writes; atomic puts, create-only via hard link, CAS under a lock directory);
  `storeAt(path)`, a `FileBlobStore` for a directory or a `PackedBlobStore` for a pack file;
  `packIndexToFile(store, file, { gen?, rev? })`.

A store may return views into a larger buffer (the whole object on a miss); the SDK detects that for slab
prefixes and uses the rest of the object instead of re-reading it.

### Snapshots and packs

```ts
import { copyObjects, packIndex, packToBytes, snapshotKeys, MemorySink } from "thread-sdk";
import { FileBlobStore, packIndexToFile, storeAt } from "thread-sdk/node";

const keys = snapshotKeys(index.revision);                       // every object the open revision reads, sorted
await copyObjects(src, dst);                                      // everything src lists -> dst (any two stores)
await copyObjects(src, dst, ["idx/current.json", ...keys]);       // one consistent revision
await copyObjects(src, dst, undefined, { skipExisting: true, concurrency: 16 });

const r = await packIndexToFile(src, "idx.threadpack", { gen: "g4", rev: 12 });   // { gen, rev, count, bytes, keys }
const { bytes } = await packToBytes(src);                         // the same pack in memory (small indexes)
await packIndex(src, sink);                                       // any PackSink: { write(bytes) }
```

`packIndex` writes `idx/current.json` (pointing at the packed revision) and then `snapshotKeys` in key order, so
the two SDKs produce identical files for the same revision. A pack is read-only: `copyObjects(pack, dst)` into a
directory or a bucket to write on top of it. Format and rationale: [docs/portability.md](../../docs/portability.md).

## Memory and limits

- The wasm linear memory never shrinks. Routing table 25 MB (4096 x 1536) + codebook 1.5 MB per generation, one
  PqTable (192 x 256 x 4 B) per query, one slab prefix (~200 KB at nprobe 34 on g4) copied into wasm per `scan`
  call and freed right after. A query holds its prefixes only until scanned; on a cold cache every slab is read
  whole (~1.7 MB each on g4), so a cold query transfers ~57 MB and keeps ~T x 1.5 KB of rerank records.
- `inflight` 12 Cache API reads per query is safe; origin (R2/S3) reads beyond 6 in flight may be cancelled by
  the runtime ("connection limit") and are retried once.
- An isolate serving several indexes keeps one base per index; `close()` what you no longer need, or keep one
  `ThreadIndex` per store at module level (the example does).
- Lexical segments: one wasm `SegmentReader` per segment per isolate holds the container table, section headers
  and directories (a few KB per segment) plus up to `lexTransientBytes` (8 MB) of recently read buckets and posting
  lists before they are evicted. The wasm module is ~750 KB with the tokenizer, stemmer and lexical kernels.
- Payloads: `returnPayload` reads them with one (multi-)range request per slab; keep payloads small or store
  them elsewhere keyed by id.

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
  `change.op` fields the SDK copies (`upsert`/`delete` + `writer: "thread-sdk"`); the Worker's `/upsert` also
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

`examples/basic/` is a `cf init workers` project bound to the `vib-bench` bucket exposing `POST /search`
(`{vector, topK?, nprobe?, rerank?, store?: "r2" | "s3"}`), `GET /manifest`, `POST /refresh` and `GET /health`.
