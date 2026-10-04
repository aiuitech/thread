# Worker Loader spike: can dynamic Workers replace the vib-bench-0..7 build pool?

Orchestrator `vib-loader-spike` (https://vib-loader-spike.aiuitech.workers.dev) with bindings
`BUCKET` (R2 `vib-bench`), `BUILD_TOKEN` (secret) and `LOADER` (`bindings.workerLoader()`).
Experiments: `node test.mjs <mem|local|r2|build|cold|cleanup>` (usage in the header of test.mjs).

The spike's code (`spikes/worker-loader/`: `src/dynamic.ts`, `src/index.ts`, `test.mjs`) is no longer in the tree;
`git show 32b3027:spikes/worker-loader/<file>` has it. What it proved is built into the Cloudflare Worker example
(`examples/cloudflare-worker/src/main.ts`).

## Short answer
Yes. Every dynamic Worker id runs in its own isolate with its own 128 MB. Requirements:
1. R2 cannot be passed into a dynamic Worker's `env` (`DataCloneError`); pass an RPC stub
   (`class BucketRpc extends WorkerEntrypoint` exposing r2get/r2head/r2put/r2list/r2delete via
   `ctx.exports.BucketRpc({props:{}})`) and wrap the bundle with a small shim that presents an
   R2-shaped `env.BUCKET` (see `src/dynamic.ts`, `SHIM_MODULE`). Bodies stream over RPC.
2. Set `limits: { cpuMs: 300_000 }` explicitly: dynamic Workers default to ~30 s CPU and CPU
   includes isolate startup (a 22 s job failed at the default limit).
3. One in-flight job per slot id: concurrent requests to the same id share its isolate (same
   failure as the pool). The driver's free list already enforces this.

## The binding
- `env: { LOADER: bindings.workerLoader() }`; runtime type `WorkerLoader` with `get(id, cb)`
  (cached by id; include the bundle version in the id) and `load(code)` (fresh each call, billed
  per call). No beta flag needed; Workers Paid plan required per docs.
- `WorkerCode`: `{compatibilityDate, compatibilityFlags?, mainModule, modules, env?,
  globalOutbound?, limits?: {cpuMs, subRequests}, tails?}`.
- Wasm: the bundle's `.wasm` is imported in the orchestrator as a compiled `WebAssembly.Module`
  and passed in `modules` under the bundle's relative path (`assets/vib_bg-*.wasm`); the bundle
  JS is embedded via `import src from "../build-bundle/index.js?raw"`.

## Memory isolation
| experiment | result |
|---|---|
| 8 concurrent slots, 90 MB JS buffers each, held 4 s | 8/8 ok, 8 distinct isolates |
| same with 90 MB `WebAssembly.Memory` each | 8/8 ok |
| 4 concurrent requests to ONE slot id, 90 MB | 1/4 ok, 3 x "Worker exceeded memory limit" |
| control: 8 x 90 MB inside the orchestrator itself | 4/8 ok, 4 x 503 (the pool problem) |
| single dynamic Worker, JS 260 MB / 300+ MB | 260 ok (soft), 300+ killed; same as a plain Worker |
| single `WebAssembly.Memory` 128 MB / 140 MB | 128 ok, 140 RangeError |

## Real build bundle (`kmeans-tree-map`, 10 shards = 126 MB read, 25 MB partial written)
| run | result |
|---|---|
| 1 job, default limits | failed: CPU time limit at ~35 s wall (also with cpuMs 30000/15000) |
| 1 job, cpuMs 300000 | ok, 21-23 s |
| 8 jobs on 8 slots | 8/8 ok, wall 64 s, per job 22-64 s (mean 45 s) |
| 16 jobs over 8 slots | 16/16 ok, wall 122 s = 7.6 s/job throughput |
| pool baseline (build-g4 fine iter 1, 48 jobs, 8-wide) | 307 s = 6.4 s/job, ~51 s per job |
Throughput is within noise of the pool. Per-job times split into ~22 s and ~55-65 s groups
regardless of orchestrator isolate: host CPU contention, also visible in the pool.

## Limits and cost
- At most 4 distinct dynamic Workers with in-flight requests per orchestrator request (irrelevant:
  one forward per request).
- 1,000 unique dynamic Workers/month included, then $0.002 per unique (id, code) per day; requests
  and CPU at Workers Standard rates. With stable ids: ~8 per day.
- Cold load of the 223 KB JS + 273 KB wasm bundle: +25-45 ms; isolates are often not kept warm.
- Each dynamic isolate: 128 MB, wasm memory capped at 128 MB. Orchestrator isolates are shared, so
  the orchestrator must not buffer large bodies. `r2put` currently buffers the body (25-38 MB) in
  the orchestrator; fix before production by streaming (`ReadableStream` + `FixedLengthStream`).

## Recommendation
Replace the 8-script pool with one orchestrator Worker plus dynamic slots:
1. Add the loader to `vib-bench` (or a sibling `thread-build`); `/build/*` forwards to
   `env.LOADER.get(\`thread-build:${BUNDLE_HASH}:slot-${slot}\`, ...)`. Produce the stage bundle with a
   second vite build, import it `?raw`, hash it at build time.
2. Always set `limits.cpuMs = 300000`.
3. Use the BucketRpc + shim pattern with streamed puts.
4. Keep one in-flight job per slot (driver free list).

Driver change: `POOL = 0..SLOTS-1`, `fetch(\`${ORCH}/build/${stage}?slot=${slot}\`, ...)`; bodies,
auth and retries unchanged. Optionally `&gen=<runId>` to force fresh isolates per run.
