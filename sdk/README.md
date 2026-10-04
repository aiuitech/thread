# thread-sdk

Query (and, experimentally, update) a [thread](../README.md) blob-native vector index from a Cloudflare
Worker. The index lives as immutable objects in R2 (or any S3-compatible store); the SDK streams the probed
slabs through the Rust/wasm kernels and returns the top-k, like a Vectorize binding but with the index in
your own bucket, readable through the Workers Cache API at the edge.

```ts
import wasmModule from "thread-sdk/wasm/thread_bg.wasm";
import { Thread } from "thread-sdk";

const thread = await Thread.init({ wasmModule });                   // once per isolate
const index = await thread.open({ bucket: env.BUCKET, waitUntil: (p) => ctx.waitUntil(p) });
const { hits } = await index.query(embedding, { topK: 10 });  // [{ id, score, attrs, source }, ...]
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

// 3. Your own BlobStore (the "local profile": a disk, a KV, a test fixture) - see "BlobStore" below
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

Manifests without `defaults` (g1, g2) use `nprobe 1, rerank 32, pc 0`; `params.source` says where each value
came from (`query`, `manifest`, `legacy`).

```ts
interface QueryResult {
	hits: { id: number; score: number; attrs: number; payload?: Uint8Array; source: "slab" | "delta" }[];
	params: { topK; nprobe; rerank; pc; source: { nprobe; rerank; pc } };
	stats: {
		gen; rev; format; routing: "flat" | "tree"; cells: number[];     // probed cells
		scanned; scanBytes; slabBytes;                                   // phase 1
		rerank; rerankCandidates; rerankSlabs; rerankReads; rerankBytes; rerankLocal; specRecords; specUsed;  // phase 2
		deltas; deltaBytes; deltaScanned; deltaHits; deltaDropped; deltaTop; tombstones; tombSets; tombDropped;
		io: { reads; hits; misses; r206; r200; bytes };                   // store accounting (cache hits/misses, origin reads)
		limiterRetries; attempt; wasmResets; wasmMemMB;
		timing: { total; route; phase1; phase2; payload };               // ms; CPU-only phases read ~0 in Workers
	};
}
```

Scores are cosine similarities: reranked slab hits and delta hits are on the int8 scale and merge directly;
with `rerank: 0` they are scan-tier (4-bit or PQ) estimates.

`index.queryMany(vectors, options)` runs the queries concurrently against the same snapshot and routing table.

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
const r = await index.upsert({ ids: [1000, 1001], vectors: [embA, embB], attrs: [5, 6], payloads: ["a", "b"] });
// { gen, rev, seq, count, cells, distinctCells, keys: { delta, tombstones, manifest }, bytes, conflicts, casConflicts, currentUpdated, ms }
await index.delete([1000]);   // { gen, rev, count, key, conflicts, ... }
```

Implemented against [docs/update-path.md](../docs/update-path.md): an upsert encodes one delta segment
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

A store may return views into a larger buffer (the whole object on a miss); the SDK detects that for slab
prefixes and uses the rest of the object instead of re-reading it.

## Memory and limits

- The wasm linear memory never shrinks. Routing table 25 MB (4096 x 1536) + codebook 1.5 MB per generation, one
  PqTable (192 x 256 x 4 B) per query, one slab prefix (~200 KB at nprobe 34 on g4) copied into wasm per `scan`
  call and freed right after. A query holds its prefixes only until scanned; on a cold cache every slab is read
  whole (~1.7 MB each on g4), so a cold query transfers ~57 MB and keeps ~T x 1.5 KB of rerank records.
- `inflight` 12 Cache API reads per query is safe; origin (R2/S3) reads beyond 6 in flight may be cancelled by
  the runtime ("connection limit") and are retried once.
- An isolate serving several indexes keeps one base per index; `close()` what you no longer need, or keep one
  `ThreadIndex` per store at module level (the example does).
- Payloads: `returnPayload` reads them with one (multi-)range request per slab; keep payloads small or store
  them elsewhere keyed by id.

## Vectorize mapping

| Vectorize | thread-sdk |
|---|---|
| `env.VECTORIZE.query(vector, { topK, returnMetadata })` | `index.query(vector, { topK, returnPayload })`: `hits[].id/score`; `attrs` (u32) and `payload` (bytes) are the metadata |
| `matches[].id` (string) | `hits[].id` (u32) |
| `upsert([{ id, values, metadata }])` | `index.upsert({ ids, vectors, attrs?, payloads? })` (replace semantics by default) |
| `insert(...)` | `index.upsert({ ..., replace: false })` |
| `deleteByIds(ids)` | `index.delete(ids)` |
| `getByIds` | not available (no id -> slab map; keep ids -> data in your own store) |
| metadata filters | not available; `attrs` is returned per hit for client-side filtering |
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
npm run sync-wasm      # copy worker/pkg/{thread.js,thread.d.ts,thread_bg.wasm,...} into src/wasm
npm test               # vitest: wire formats, stores, SigV4, end-to-end with the wasm in Node
npm run build          # dist/ (tsc + wasm assets)
cd examples/basic && npm install && cf deploy --secrets-file .env.secrets
node scripts/verify.mjs --n 50 --nprobe 34 --rerank 150    # compare the example with the main Worker
```

`examples/basic/` is a `cf init workers` project bound to the `vib-bench` bucket exposing `POST /search`
(`{vector, topK?, nprobe?, rerank?, store?: "r2" | "s3"}`), `GET /manifest`, `POST /refresh` and `GET /health`.
