# vib-bench Worker

One deployment (`vib-bench`) serves queries and runs the whole build: build stages, writes and compaction
slices run in dynamic Workers ("build slots") created with the Worker Loader binding (`env.LOADER`). This
replaced the vib-bench-0..7 pool of identical scripts.

```
src/main.ts    entry point: routing, BucketRpc (R2 for slots), SlotRpc (slot -> slot stages), the slot shim
src/index.ts   the app: queries, upsert/delete, every build stage, compaction (also the slots' code)
vite.config.ts plugin `thread-stage-bundle`: bundles src/index.ts a second time (rolldown) into the virtual
               module `virtual:thread-stage-bundle` = {source, hash, wasmPath}; the wasm import stays external
```

## Deploy

```sh
cargo build --release --target wasm32-unknown-unknown -p thread-wasm      # from the repo root
wasm-bindgen --target web --experimental-reset-state-function --out-dir crates/thread-wasm/pkg --out-name thread target/wasm32-unknown-unknown/release/thread_wasm.wasm
cd examples/cloudflare-worker && cf deploy --secrets-file .env.secrets  # BUILD_TOKEN=...
```

`cf deploy` runs `vite build`, which builds the stage bundle as part of the same build (no separate step).
Bindings: `BUCKET` (R2 `vib-bench`), `BUILD_TOKEN` (secret), `LOADER` (`bindings.workerLoader()`; Workers Paid).
`THREAD_WORKER_NAME=<name> cf deploy ...` deploys a copy under another name (e.g. a staging Worker).

## Routing

| request | runs in |
|---|---|
| `POST /build/<stage>?slot=<id>` | dynamic Worker `thread-build:<hash>:slot-<id>` (default slot `admin`) |
| `POST /upsert`, `POST /delete` | slot `writer` (or `?slot=<id>`) |
| `?slot=local` on any of the above | this isolate (debugging; a write here grows a query isolate's wasm memory) |
| everything else (`/query`, `/manifest`, `/index/status`, `/stat`, ...) | this isolate |

A slot id is one isolate with its own 128 MB and `limits.cpuMs` 300,000 (dynamic Workers default to ~30 s of CPU,
and isolate startup counts). Concurrent requests to the same slot id share that isolate: send one job per slot at a
time (`bench/driver.mjs` keeps a free list of slot ids `0..THREAD_SLOTS-1`; `bench/compact.mjs` uses `cmp-0..`; automatic
compaction uses `auto-0..3`). The bundle hash is part of the id, so a deploy never reuses an isolate holding old
code. Unique dynamic Workers are billed per (id, code) per day after 1,000 per month; with stable ids that is
at most ~22 per day (8 driver slots, 8 compaction slots, 4 automatic-compaction slots, writer, admin).

Slots cannot take an R2 binding in their env, so they get `BUCKET_RPC` (a loopback `WorkerEntrypoint`) and the shim
gives the bundle an R2Bucket-shaped `env.BUCKET` (get with range/onlyIf, head, put with onlyIf, list, delete).
GET bodies are R2's own streams; PUT bodies of 256 KB and more are sent as a byte stream with their length and
written through a `FixedLengthStream`, so the orchestrator isolate never holds an object.

`POST /build/probe {waitUntilMs?, tag?}` reports where a request ran (slot, isolate id, wasm memory, Cache API) and
optionally writes `tmp/probe/{tag}` from `ctx.waitUntil` after a delay (`delete-prefix` `tmp/probe/` removes it).

Background on the design: [docs/worker-loader.md](../../docs/worker-loader.md).
