# Observability and profiling

How to see where an operation spends its time, in every host thread runs in, and how to turn that into an
optimization. Three layers, each answering a different question:

| layer | question | where |
|---|---|---|
| 1. trace envelopes | which step of which operation is slow, how much it read, how much CPU its kernels used | `sdk/ts/src/trace.ts`, `sdk/python/thread_sdk/trace.py`, the Worker's `/query` |
| 2. kernel CPU | how fast route, scan, rerank, merge and BM25 are, and where inside them the time goes | envelope `kernels`; `crates/thread-core/benches/kernels.rs`; the `profiling` cargo profile |
| 3. workloads and aggregation | how each use case behaves, before and after a change, and in production | `bench/scenarios.mjs`, `bench/query_bench.mjs`, `TRACE_SAMPLE` + Analytics Engine in the Worker |

## 1. Trace envelopes

Every SDK operation builds one envelope, a plain JSON object with the same shape in the TypeScript SDK, the Python
SDK and the reference Worker. The schema is versioned (`v`); keys are camelCase in every implementation, like the
manifest.

```jsonc
{
  "v": 1,
  "op": "query",                 // query | search | lexical | open | get | listKeys | upsert | delete
  "sdk": "ts",                   // ts | py | worker
  "at": 1794146206821,           // start, ms since the epoch
  "ms": 3.12,                    // wall time of the operation
  "clock": "real",               // "io-only" inside Cloudflare Workers (below)
  "attrs": { "gen": "g4", "rev": 179, "format": 3, "routing": "flat", "topK": 10, "nprobe": 34, "rerank": 150, "filtered": false, ... },
  "io": { "reads": 41, "hits": 41, "misses": 0, "r206": 41, "r200": 0, "bytes": 6744000 },
  "spans": [
    { "name": "route",  "start": 0,    "ms": 0.9,  "io": { ... } },
    { "name": "phase1", "start": 0.9,  "ms": 1.8,  "io": { ... } },
    { "name": "phase2", "start": 2.7,  "ms": 0.35, "io": { ... } },
    { "name": "payload","start": 3.05, "ms": 0.07, "io": { ... } }
  ],
  "counters": { "scanned": 24276, "scanBytes": 6489000, "pushed": 3120, "candidates": 2900, "rerankReads": 9, ... },
  "kernels": { "route": { "calls": 3, "ms": 0.85 }, "scan": { "calls": 34, "ms": 1.6 }, "merge": { "calls": 1, "ms": 0.1 }, "rerank": { "calls": 9, "ms": 0.08 } },
  "events": [ ... ]              // only with traceEvents / trace_events
}
```

- **spans**: the operation's steps. `start` is ms from the operation's start; a span whose start is not known (a
  pack opened before `open`) has none. `io` is the store work that completed during the span. Nested operations
  are children: a search's `dense` span holds the dense query's phases.
- **counters**: work done. The query counters are the same numbers `stats` already had, plus `pushed`, the entries
  that entered a scan's top-T heap (a new counter from the Rust kernels; `pushed / scanned` is how selective the
  running threshold was).
- **kernels**: CPU spent in the wasm (TypeScript) or native (Python) kernels, `{calls, ms}` per kernel. The rest
  of a span's time is I/O wait and host-language work around the kernels.
- **clock**: in Cloudflare Workers the clock only advances across I/O, so CPU-only spans and every kernel `ms`
  read 0 (`clock: "io-only"`). Use Workers Logs' `cpuTimeMs` or a CPU profile for Worker CPU (layer 2).
- **events** (opt-in): one entry per store read of a dense query: `{kind, key, start, ms, ranges, bytes, hits,
  misses}`, kind `prefix` (phase-1 scan prefix), `slab` (a v1 slab whole), `merged` (cross-slab pack range),
  `records` (phase-2 rerank records), `spec` (speculative records), `payload`. Delta segment reads are not events
  (they are usually served from the delta cache).

### Spans by operation

| op | spans | kernels |
|---|---|---|
| `query` | `route`, `phase1`, `phase2`, `payload`; `base` (children `fetch`, `build`) first when the query loaded the routing table | `route` (normalize, cell dots, routing, PQ table), `scan`, `deltaScan`, `merge`, `rerank` |
| `search` | `embed` (when the index embedded the text), `dense` (the query's spans), `lexical` (the lexical search's spans), `fuse`, `docs` | the dense and lexical kernels |
| `lexical` | `lookup` (term lookups, every segment), `score` (BM25 over the postings); a filtered search has one `pass` (children `lookup`, `score`) and one `attrs` per depth | `tokenize`, `bm25Lookup`, `bm25Score` (calls count attempts: a reader that needs pages it does not hold is run again after they arrive) |
| `open` | `pack` (no start), `current`, `prefetch`, `manifest`, `slabs`, `base` (children `fetch`, `build`) | none |
| `get` | `keys`, `docs` | none |
| `listKeys` | `keys` | none |
| `upsert`, `delete` | `commit` (objects and the manifest CAS), `useRevision` (switching the index to the new revision) | none; counters are the result's numeric fields |
| Worker `query` | `gen`, `load`, `route`, `phase1` (children `fetch`, `delta`, `scanAndMerge`), `phase2` | none (io-only clock) |

Spans need not tile the operation: gaps are host work between steps (parsing, building the result).

### Where to find them

| | TypeScript | Python | Worker |
|---|---|---|---|
| query | `r.stats.trace` | `r.stats["trace"]` | `trace` in the `/query` response |
| search | `r.stats.trace` | `r.stats["trace"]` | |
| lexical | `r.stats.trace` | `r.stats["trace"]` | |
| open | `index.openTrace` | `index.open_trace` | |
| get | `r.trace` | the sink only (the result is a plain dict) | |
| listKeys | `page.trace` | `page.trace` | |
| upsert / delete | `r.trace` | `r.trace` | |
| per-read events | `query(v, { traceEvents: true })`, `search({ dense: { traceEvents: true } })` | `query(v, trace_events=True)` | |

### Sinks

Hand every envelope to a function, sampled, without changing call sites. One call per operation: a search's dense
and lexical lists are its spans, not envelopes of their own. A sink that throws is ignored; it runs inline, so it
must not block.

```ts
const index = await thread.open({ bucket: env.BUCKET, onTrace: (t) => console.log(JSON.stringify({ trace: t })), traceSample: 0.05 });
```

```python
index = open_index(store, on_trace=lambda t: log.info("trace", extra={"trace": t}), trace_sample=0.05)
```

An OpenTelemetry exporter is a sink that maps spans to OTel spans (`start` + `at` gives absolute times); none
ships yet. The envelope is the data model and OTel would be one target.

### The Worker

`/query` responses carry `trace` (`sdk: "worker"`). The Worker runs its own dense query implementation, not the
SDK's, so its spans are its own timing points mapped onto the same schema. The `QueryBody` / `queryEnvelope`
adapter is in `examples/cloudflare-worker/src/index.ts`.

Two bindings export a sample of them (`examples/cloudflare-worker/cloudflare.config.ts`):

- `TRACE_SAMPLE` (text, default `"0"` = off): the share of `/query` requests whose envelope is logged as one JSON
  line, `{"trace": {...}, "colo": "YYZ"}`. Workers Logs indexes the fields, so `trace.spans`, `trace.counters.*`
  and `trace.attrs.*` can be filtered and grouped in the dashboard or with the observability query API.
- `QUERY_METRICS` (Analytics Engine dataset `thread_queries`): one data point per sampled query, cheap to write and
  queryable with SQL. The layout is fixed, because queries depend on it:

| field | value |
|---|---|
| `index1` | gen |
| `blob1`..`blob5` | gen, colo, routing, cache mode, filtered (`"1"` / `"0"`) |
| `double1`..`double12` | total ms, gen ms, load ms, phase1 ms, fetch ms, phase2 ms, scanBytes, rerankBytes, cache misses, scanned, nprobe, rerank |

```sh
cf analytics_engine sql query --help   # then, for example:
# SELECT blob2 AS colo, quantileWeighted(0.5)(double1, _sample_interval) AS p50, quantileWeighted(0.95)(double1, _sample_interval) AS p95,
#        avg(double5) AS fetch_ms, avg(double9) AS misses
# FROM thread_queries WHERE timestamp > NOW() - INTERVAL '1' HOUR GROUP BY colo ORDER BY p95 DESC
```

Turn it on by changing `TRACE_SAMPLE` in the config and deploying (`cf deploy` from the example directory). At
2500 requests per second a sample of 0.01 is 25 log lines and data points a second.

## 2. Kernel CPU

**Per operation.** On Node and Python the envelope's `kernels` say how much of an operation was kernel CPU.
TypeScript times each wasm call; Python's `QueryRun` times its scans, merge and rerank natively (`Instant`, with the
GIL released, so concurrent scans on the I/O pool are measured exactly), and Python times routing around the native
calls. Inside Workers they read 0. There the request's `cpuTimeMs` in Workers Logs is the truth for cost, and a CPU
profile says where it went.

**Benchmarks.** `cargo bench -p thread-core` runs Criterion benches of the query kernels at g4's shape (1536 dims,
4096 cells, 714-entry slabs, PQ-192, T = 150): `route/flat`, `route/cell_dots`, `scan/pq`, `scan/pq_table`,
`scan/sq4`, `rerank/int8`, `merge/dedup`, `tokenize/english`. Environment variables change the shape
(`THREAD_BENCH_DIM`, `THREAD_BENCH_CELLS`, `THREAD_BENCH_SLAB_N`, `THREAD_BENCH_PQ_M`). Real fixtures replace
the synthetic slab with a production one:

```sh
mkdir -p data/fixtures
cf r2 objects get idx/g4/pq.bin --bucket-name vib-bench > data/fixtures/pq.bin
cf r2 objects get idx/g4/slab/00042.<slabRev> --bucket-name vib-bench > data/fixtures/slab.bin   # a key from the slabs table
THREAD_BENCH_SLAB=data/fixtures/slab.bin THREAD_BENCH_PQ=data/fixtures/pq.bin cargo bench -p thread-core -- scan/pq_fixture
```

Criterion keeps the previous run in `target/criterion`, so a second run after a change prints the difference.

**Native is not wasm.** The dot products in `math.rs` have a SIMD path only for wasm32 + simd128; native builds (the
benches, the CLI, the Python extension) run scalar fallbacks. Use the benches to compare a kernel with its own
previous version, and Node's envelope `kernels` for wasm costs. Measured on an Apple M2 Pro:

| kernel (g4 shape) | native (Criterion) | wasm in Node |
|---|---:|---:|
| `cell_dots`, 4096 x 1536 | 0.94 ms | 0.50 ms |
| PQ-192 scan, one 714-entry slab | 60 µs | 79 µs |
| 4-bit scan, 1536 dims | 0.44 µs/entry | 0.19 µs/entry |

**Flamegraphs.** The `profiling` cargo profile is release code with line tables and symbols:

```sh
cargo bench -p thread-core --profile profiling --bench kernels --no-run      # prints the bench executable
samply record target/profiling/deps/kernels-<hash> --bench --profile-time 10 scan/pq
cargo build -p thread-cli --profile profiling && samply record target/profiling/thread bench-scan 2000 1536 192
```

For the Python extension, build it with debug info (`RUSTFLAGS="-C debuginfo=line-tables-only" maturin develop
--uv --profile python`) and record the Python process with `samply` or `py-spy --native`.

**Production CPU profiles.** Cloudflare's [profiling of deployed Workers](https://developers.cloudflare.com/workers/observability/profiling-in-production/)
returns pprof. The shipped `thread_bg.wasm` keeps its `name` section, so wasm frames
are named. The JS around them (`sdk/ts/src/query.ts`, the Worker) maps to source only with source maps enabled.

## 3. Workloads

**Scenarios.** `bench/scenarios.mjs` runs each use case many times in this process, through the built TypeScript
SDK, and summarizes the envelopes: wall percentiles, per-span p50/p95/mean with store work, mean kernel CPU and
counters, and ops per second. It builds a synthetic index (clustered vectors, Zipf-ish texts, keys, a `tenant`
attribute) or opens one on disk:

```sh
(cd sdk/ts && npm run build)
node bench/scenarios.mjs                                   # 20k docs x 256 dims, every scenario
node bench/scenarios.mjs --docs 50000 --dim 768 --only dense,hybrid --n 500
node bench/scenarios.mjs --store data/scn-768              # build once into a directory, reuse on later runs
node bench/scenarios.mjs --index data/idx-g4               # a local copy of a real index
node bench/scenarios.mjs --latency 15                      # every read waits 15 ms: the I/O-bound picture
node bench/scenarios.mjs --only dense --events --n 20      # the slowest query's reads, one line each
node bench/scenarios.mjs --json data/scenarios.jsonl       # one row per scenario, to diff before/after
```

Scenarios: `dense`, `dense-filtered`, `lexical`, `lexical-filtered`, `hybrid`, `hybrid-docs`, `get`, `listKeys`,
`open`, `upsert`. A row looks like this (8,000 docs x 128 dims, M2 Pro, in memory):

```
[dense] 100 ops at concurrency 1, 1310 ops/s
  query n=100 (ts, clock real): p50 0.71 ms  p95 0.9  p99 1.85  mean 0.73  | 7.94 reads 807.5 KB 0 misses
    span                         start      p50      p95     mean       KB  reads
    route                         0.00     0.01     0.02     0.01      0.0    0.0
    phase1                        0.01     0.56     0.69     0.58    777.2    5.0
    phase2                        0.59     0.09     0.15     0.10     30.4    2.9
    payload                       0.69     0.03     0.06     0.04      0.0    0.0
    kernels (CPU ms per op, mean): route 0.005 (2 calls)  scan 0.306 (4 calls)  merge 0.034 (1 calls)  rerank 0.008 (3.94 calls)  = 0.353 of 0.73 ms
```

Read it as: phase 1 is 80% of the query, and the scan kernel is about half of phase 1. The other half is JS
around the kernel (parsing hits, the candidate map, slab entry lookups), which is where to look next on this shape.

**Against the deployed Worker.** `bench/query_bench.mjs` prints the same per-span summary for the Worker's
envelopes after its usual output, and adds it to `--json` rows as `trace`. `--trace 0` hides it.

**In production.** Sample with `TRACE_SAMPLE`, then group the Analytics Engine points by colo, gen, routing and
filtered (above).

## Finding a bottleneck

1. **Pick the use case** and reproduce it with a scenario (or `query_bench.mjs` against the Worker). Note the
   p50/p95 and ops per second.
2. **Find the span.** The span table says which step holds the time. Big `misses` or `reads` on it means I/O. A
   span much longer than its kernels with little I/O means host work.
3. **I/O-bound** (Worker, S3, cold cache): run with `--events` (or `traceEvents`) and look at the slowest query's
   reads. You will see whether it waits on one slow read, on the limiter (reads starting late), or on many small
   reads that could be one multi-range request.
4. **CPU-bound in a kernel**: confirm with the Criterion bench, make the change, run the bench again (Criterion
   prints the difference), then the scenario.
5. **CPU-bound outside the kernels**: profile the host. Use `node --cpu-prof bench/scenarios.mjs --only dense`, or
   a Workers production profile for the Worker.
6. **Check the change everywhere**: `--json` rows before and after, `query_bench.mjs` against a deployment, and
   the cross-SDK test (below).

### First readings

These are measurements from building this, not conclusions:

- **The Worker's CPU is mostly outside the kernels.** As wasm on an M2 Pro, one g4 query's kernels add up to
  about 3.4 ms: `cell_dots` 0.5 ms, 34 PQ scans of 79 µs, plus rerank and merge (the table above). Workers Logs
  report 31 to 35 ms of CPU per warm query ([benchmarks.md](benchmarks.md#cost)). An edge CPU is slower than an M2
  Pro, but not by 10x. The rest is likely the JS around the kernels, response building and Cache API body
  handling. A Workers CPU profile of `vib-bench` under load would show where, and it is the cheapest next
  measurement: CPU is the query's whole marginal cost.
- **The Python SDK runs scalar dot products** where wasm uses SIMD. Its 4-bit scan is 2.3x slower per entry and its
  routing dots 1.9x slower than the TypeScript SDK's. A NEON / SSE path in `math.rs` would close that.
- **On a small in-memory index, phase 1 is half kernel and half JS**: the scenario row above.

## Rules for keeping it cheap and consistent

- Nothing is recorded inside the scan loops: the kernels count (`ScanStats`), the hosts time whole calls.
  `pushed` is one increment on the branch that already pushes into the heap.
- The default level costs a few small objects per operation. Per-read events are opt-in.
- Sinks never fail an operation, and envelopes never change results. The tests check both.
- One schema. `sdk/python/tests/test_interop.py` compares the TypeScript and Python envelopes of every interop
  request: span names, the counters that depend on neither caching nor timing (including `pushed`, which checks
  the wasm and native kernels against each other) and the kernel and attrs keys. A new counter, span or kernel
  goes into both SDKs, and the Worker adapter where it applies, in the same change. Breaking changes bump `v`.
