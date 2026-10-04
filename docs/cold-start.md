# Cold-start performance against S3

What the first query costs when nothing is warm: a new process, a new index handle, empty caches, against a plain
S3 endpoint (R2's S3 API here, AWS S3 in spirit), for both SDKs. This page describes the measurement harness in
`bench/cold/`, the SDK timeline it reads, the baseline it recorded (g4 rev 179, 2026-10-04) and where the time
goes. Later optimizations re-run the same harness on the same query set and network profiles and compare medians
(`bench/cold/summarize.mjs`).

## What is measured

Each trial is a fresh child process (fresh TLS/connection pool, fresh wasm or native module, fresh index, empty
caches) that runs three scenarios on query vectors no earlier trial used:

| scenario | what runs | query vector |
|---|---|---|
| `process-cold` | spawn, import the SDK, compile + init the wasm (TS), `open`, one query | `2t` |
| `slab-cold` | a second query in the same process: index open, routing table loaded, its slabs not read yet | `2t + 1` |
| `warm` | the same vector again: whatever the SDK caches is warm | `2t + 1` |

`--n 10` therefore uses queries 0..19 of `data/prep/queries.f32`. Every result is checked against
`data/prep/groundtruth.u32` (recall@10), so an optimization that changes what is read shows up as a recall change.
The fixed set warmed into the proxy and packed is queries **0..29** (30 x 34 probes = 820 distinct slabs, 1.2 GB);
none of the 30 has a near-duplicate twin inside the set (`data/prep/query-twins.json`), so no trial pre-warms
another's slabs beyond ordinary cell overlap. Parameters are the manifest defaults: nprobe 34, rerank 150, top 10.

Defaults matter here, because the two SDKs differ in what sits in front of the store:

- **TypeScript in Node: no cache at all.** `thread.open` puts a `CachedBlobStore` in front of the store only when
  the Workers Cache API exists (`caches.default`); Node has none, and the SDK has no in-memory equivalent of Python's
  `MemoryCache`. So in Node every query, including `warm`, reads everything again: the 34 scan prefixes as range
  GETs, then the rerank records as **one range GET per record run** (S3 has no multi-range GET, `S3BlobStore` has no
  `getRanges`), ~140 GETs per query. The `fill` hint (read the whole slab on a miss) is ignored by `S3BlobStore`,
  so Node never reads whole slabs.
- **Python: a 256 MB `MemoryCache` by default** for remote stores. A phase-1 miss reads the WHOLE slab (scan
  prefix + rerank tier, 8x the prefix bytes), keeps it, and phase 2 then needs no request. `warm` costs 0 requests.

What a "request" is in the tables:

- `req ...` are the SDK's own counters (`openStats` / `stats.timeline` `reads`): requests that reached the store,
  i.e. HTTP GETs (and a pack's HEAD). Cache hits (Python) are not requests; they are counted in `cache hits`.
- `MB ...` from the SDK counts bytes delivered to the query, including bytes served from Python's memory cache
  (so Python's `warm` shows 6 MB with 0 requests). The network truth is `proxy MB`.
- `proxy requests` / `proxy MB` / `proxy new connections` are what the fake S3 saw for that scenario (exact; reset
  per scenario). For `process-cold` they include the open. They are absent for `r2s3`.

## The harness

| piece | role |
|---|---|
| `bench/cold/fake_s3.mjs` | a read-only caching proxy in front of R2's S3 API: GET with one Range, HEAD, ListObjectsV2, R2's error answers verbatim, any SigV4 signature accepted. First touch fetches the whole object from R2 into `data/cold/objects/`; `--overlay DIR` is checked first. Network model per request: `--rtt-ms` before the first byte (+ uniform `--jitter-ms`), `--conn-ms` extra on the first request of a new TCP connection (TCP + TLS handshakes, which plain HTTP to localhost would hide; default 2 x rtt), body streamed at `--mbps` per response, optionally all bodies sharing one `--link-mbps` link. JSONL request log; `GET /__stats`, `POST /__reset`, `GET/POST /__config`. It refuses PUT/POST/DELETE and never writes upstream. |
| `bench/cold/verify_proxy.mjs` | proxy vs R2 on 35 cases (whole objects, closed / open-ended / suffix / clamped / unsatisfiable / reversed / multi ranges, HEAD, 404, a listing): status, Content-Range, length and body sha256 all match (`data/cold/verify-proxy.txt`). R2 answers a multi-range GET with 400 InvalidArgument (AWS ignores it and sends 200); the proxy does what R2 does. |
| `bench/cold/warm_proxy.mjs` | fills the proxy cache for a query set; the second run must report 0 upstream fetches. Every bench row carries `proxy upstream fetches`, which must be 0 (it was in every baseline run). |
| `bench/cold/make_pack.mjs` | the pinned revision as a pack (`data/cold/overlay/packs/g4-cold.threadpack`) with exactly `packIndex`'s layout, built from the proxy cache only: slabs outside the warmed set are holes in a sparse file (5.17 GB logical, 1.3 GB on disk). Offsets, directory and open cost are the full pack's; a query outside queries 0..29 reads zeros and fails on the slab header. |
| `bench/cold/cold_bench.mjs`, `cold_bench.py` | the bench, TS and Python SDK: same CLI (`--store r2s3\|proxy\|pack --n --queries --nprobe --rerank --out --label`), same JSONL schema `thread-cold/1` (a `meta` row, then a `trial` row per trial and scenario with `ms`, `reqs`, `bytes`, `stats`, `proxy`, `recall`, `ids`), same table. |
| `bench/cold/summarize.mjs` | the table of saved runs; several files side by side (`--only-compare`) or as markdown (`--markdown`, what this page shows). |
| `bench/cold/slab_bytes.mjs` | per-slab byte accounting (below). |
| `bench/cold/run_baseline.sh` | everything below in one go; `TAG=after-x` for a comparison run. |

### SDK timeline

Both SDKs now expose where a cold start goes, with the same field names (TS camelCase where both differ already:
`openStats` vs `open_stats`). A phase stat is `{ms, reads, hits, misses, r206, r200, bytes}`.

- `index.openStats` / `index.open_stats`: `total`, `current` (current.json), `manifest`, `slabs` (slabs table and
  tombstone sets), `base` (`coarse.f32`, `fine.f32`, `pq.bin` read and loaded; plus `fetch` / `build` ms and
  per-file bytes), `pack` (a `PackedBlobStore`'s own HEAD + tail + directory reads; it happens when the pack store is
  created, so it is not in `total`), and in TS `wasmInit` (`thread.initMs`).
- `stats.timeline` per query: `route`, `phase1`, `phase2`, `payload` (trace snapshots at the phase boundaries: a
  speculative rerank read issued in phase 1 that lands in phase 2 counts in phase 2), and `base` when that query
  reloaded the routing table.

The instrumentation lives in `revision.ts` / `revision.py` (`resolveTarget` / `loadRevState` take an optional
timeline), `loadBase` and `PackedBlobStore`.

## Commands

```sh
(cd sdk/ts && npm run build)                         # the proxy and the TS bench import sdk/ts/dist
cd sdk/python && uv venv .venv && uv pip install --python .venv/bin/python maturin pytest \
  && .venv/bin/maturin develop --release && cd ../..
bench/cold/run_baseline.sh                           # TAG=baseline N=10: warm-up, pack, 3 profiles x {proxy, pack} x {ts, py}, r2s3
# what it runs, by hand:
node bench/cold/fake_s3.mjs --rtt-ms 0 --mbps 0 --conn-ms 0 --overlay data/cold/overlay &   # warm-up proxy
node bench/cold/warm_proxy.mjs --from 0 --count 30 --keys-out data/cold/warm-keys.json      # twice: 2nd = 0 upstream
node bench/cold/make_pack.mjs
node bench/cold/fake_s3.mjs --rtt-ms 25 --mbps 200 --overlay data/cold/overlay --log data/cold/baseline-proxy-rtt25.jsonl &
node bench/cold/cold_bench.mjs --store proxy --n 10 --out data/cold/baseline-ts-proxy-rtt25.jsonl
sdk/python/.venv/bin/python bench/cold/cold_bench.py --store pack --n 10 --out data/cold/baseline-py-pack-rtt25.jsonl
node bench/cold/cold_bench.mjs --store r2s3 --n 10 --out data/cold/baseline-ts-r2s3.jsonl   # real network, no proxy
# compare an optimization with the baseline
TAG=after-x R2S3=0 WARM=0 bench/cold/run_baseline.sh
node bench/cold/summarize.mjs --only-compare data/cold/baseline-ts-proxy-rtt25.jsonl data/cold/after-x-ts-proxy-rtt25.jsonl
node bench/cold/summarize.mjs --markdown data/cold/baseline-ts-*.jsonl
```

Profiles (`conn` = extra delay on a new connection, 2 x rtt by default):

| profile | rtt | bandwidth | conn | meant to look like |
|---|---:|---|---:|---|
| `rtt25` | 25 ms | 200 Mbps per response | +50 ms | a nearby S3 region / R2 from a good network |
| `rtt60` | 60 ms | 100 Mbps per response | +120 ms | cross-region AWS S3 |
| `rtt25link500` | 25 ms | 200 Mbps per response, all responses share 500 Mbps | +50 ms | `rtt25` on a client whose link is the limit |
| `r2s3` | real | real | real | this host to R2's S3 endpoint over HTTPS (answered from Cloudflare's YUL colo) |

Measured against R2 from this host: a serial request takes 75-105 ms (manifest 87 ms, slabs table 104 ms, median),
the first one 130 ms with DNS + TCP + TLS, and `fine.f32` (25 MB) arrives in 534 ms median (~380 Mbps on one
stream, 1.1 s at p90). Real R2 is thus closer to `rtt60` in latency and faster than both profiles in per-stream
bandwidth.

Files: `data/cold/baseline-{ts,py}-{proxy,pack}-{rtt25,rtt60,rtt25link500}.jsonl`, `data/cold/baseline-{ts,py}-r2s3.jsonl`,
each with its printed table in `.txt`; proxy request logs `data/cold/baseline-proxy-*.jsonl`; `data/cold/pack-info.json`,
`data/cold/warm-keys.json`, `data/cold/verify-proxy.txt` (all gitignored).

## Baseline

g4 rev 179 (4103 cells, 1536 dims, PQ-192 scan tier + int8 rerank tier, no deltas or tombstones), nprobe 34,
rerank 150, n = 10 trials per scenario, Node 24.21 / CPython 3.14, git `e169ffe` + the bench fixes in this commit
(the TS runs were repeated after the `wasmInit` attribution fix described below; the Python runs did not change).
Cells are median / p90 (recall@10: mean / min over the 10 queries). `pack-*` columns read the pack through the same
proxy profile.

### TypeScript SDK (Node, no cache)

**process-cold** (median / p90; recall@10: mean / min)

| metric | proxy-rtt25 | pack-rtt25 | proxy-rtt60 | pack-rtt60 | proxy-rtt25link500 | pack-rtt25link500 | r2s3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ms boot (spawn -> main) | 23.0 / 31.0 | 24.0 / 30.0 | 27.0 / 38.0 | 26.5 / 32.0 | 24.0 / 39.0 | 25.0 / 33.0 | 30.5 / 39.0 |
| ms import SDK | 6.5 / 9.6 | 6.3 / 9.7 | 6.4 / 8.9 | 6.5 / 7.4 | 6.3 / 6.8 | 6.5 / 9.7 | 6.4 / 9.3 |
| ms wasm compile | 1.4 / 1.9 | 3.3 / 4.8 | 1.6 / 4.7 | 1.4 / 1.6 | 1.4 / 1.5 | 1.6 / 4.8 | 1.5 / 1.8 |
| ms wasm init | 13.8 / 16.5 | 13.4 / 16.9 | 13.5 / 16.7 | 13.3 / 13.9 | 14.8 / 16.5 | 13.6 / 14.3 | 13.9 / 16.4 |
| ms open.pack | - | 218.4 / 223.7 | - | 476.8 / 486.8 | - | 215.3 / 224.0 | - |
| ms open.current | 96.1 / 96.7 | 27.2 / 27.7 | 195.8 / 202.3 | 63.8 / 65.2 | 94.8 / 96.7 | 27.5 / 28.2 | 129.2 / 171.1 |
| ms open.manifest | 30.9 / 31.3 | 29.4 / 30.3 | 64.3 / 66.7 | 66.0 / 67.7 | 32.4 / 32.7 | 30.3 / 30.7 | 86.5 / 104.5 |
| ms open.slabs | 34.5 / 36.1 | 34.0 / 36.2 | 75.8 / 77.3 | 76.4 / 77.8 | 33.9 / 35.1 | 34.0 / 37.3 | 104.1 / 134.0 |
| ms open.base | 1132.7 / 1141.2 | 1140.8 / 1145.0 | 2248.3 / 2255.7 | 2255.9 / 2258.1 | 1135.3 / 1143.2 | 1138.8 / 1147.0 | 557.3 / 1112.5 |
| ms   base fetch | 1109.8 / 1119.6 | 1115.7 / 1123.3 | 2225.8 / 2231.3 | 2232.8 / 2236.5 | 1113.4 / 1120.1 | 1114.3 / 1120.8 | 534.1 / 1091.4 |
| ms   base build | 22.4 / 23.8 | 24.1 / 26.9 | 23.0 / 24.4 | 22.3 / 24.1 | 23.7 / 25.0 | 23.0 / 26.0 | 22.0 / 24.5 |
| ms open total | 1292.3 / 1302.9 | 1447.0 / 1459.9 | 2587.2 / 2594.8 | 2941.9 / 2946.1 | 1293.7 / 1304.6 | 1449.8 / 1458.6 | 896.9 / 1595.0 |
| ms route | 3.0 / 3.4 | 3.1 / 3.7 | 3.0 / 3.5 | 3.0 / 3.3 | 3.0 / 3.6 | 3.0 / 3.9 | 3.0 / 3.5 |
| ms phase1 | 183.2 / 195.7 | 179.2 / 193.6 | 375.4 / 401.0 | 373.5 / 383.2 | 190.3 / 207.8 | 184.4 / 192.8 | 364.4 / 389.9 |
| ms phase2 | 392.6 / 419.6 | 320.5 / 362.6 | 756.5 / 834.7 | 632.1 / 754.5 | 409.9 / 434.7 | 332.5 / 377.5 | 932.3 / 1076.8 |
| ms query total | 576.0 / 613.8 | 508.1 / 534.7 | 1141.3 / 1228.6 | 1005.3 / 1131.8 | 602.5 / 635.9 | 522.4 / 559.6 | 1305.1 / 1489.1 |
| ms process e2e (spawn -> hits) | 1924.5 / 1954.0 | 2011.5 / 2061.0 | 3785.5 / 3878.0 | 4011.0 / 4138.0 | 1954.5 / 1997.0 | 2033.0 / 2080.0 | 2319.0 / 2976.0 |
| req open (pack+cur+man+slabs+base) | 6 | 9 | 6 | 9 | 6 | 9 | 6 |
| req phase1 | 41.5 / 42 | 41.5 / 42 | 34 / 41 | 34 / 40 | 34 | 34 | 35 / 41 |
| req phase2 | 135.5 / 143 | 136 / 142 | 137 / 144 | 138 / 150 | 140.5 / 153 | 140.5 / 154 | 136.5 / 151 |
| req query | 174.5 / 184 | 174.5 / 184 | 174.5 / 184 | 174.5 / 184 | 174.5 / 188 | 174.5 / 188 | 173.5 / 185 |
| MB open | 25.66 | 25.89 | 25.66 | 25.89 | 25.66 | 25.89 | 25.66 |
| MB phase1 | 6.24 / 6.97 | 6.24 / 6.95 | 6.23 / 6.95 | 6.23 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 | 6.23 / 6.95 |
| MB phase2 | 0.21 / 0.22 | 0.21 / 0.22 | 0.22 | 0.22 | 0.22 / 0.23 | 0.22 / 0.23 | 0.22 |
| MB query | 6.45 / 7.17 | 6.45 / 7.17 | 6.45 / 7.17 | 6.45 / 7.17 | 6.45 / 7.17 | 6.45 / 7.17 | 6.45 / 7.17 |
| cache hits (query) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scan prefix MB | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 |
| slab MB transferred | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 |
| rerank reads | 140.5 / 149 | 140.5 / 150 | 140.5 / 149 | 140.5 / 150 | 140.5 / 153 | 140.5 / 154 | 139.5 / 151 |
| rerank local / spec used | 150 | 150 | 150 | 150 | 150 | 150 | 150 |
| proxy requests | 180.5 / 190 | 183.5 / 193 | 180.5 / 190 | 183.5 / 193 | 180.5 / 194 | 183.5 / 197 | - |
| proxy MB | 32.11 / 32.84 | 32.34 / 33.07 | 32.11 / 32.84 | 32.34 / 33.07 | 32.12 / 32.84 | 32.35 / 33.07 | - |
| proxy new connections | 12 | 17 / 18 | 12 | 17 / 18 | 12 | 17 / 18 | - |
| proxy upstream fetches (want 0) | 0 | 0 | 0 | 0 | 0 | 0 | - |
| recall@10 | 88.0% / 60.0% | 88.0% / 60.0% | 88.0% / 60.0% | 88.0% / 60.0% | 88.0% / 60.0% | 88.0% / 60.0% | 88.0% / 60.0% |

**slab-cold** (median / p90; recall@10: mean / min)

| metric | proxy-rtt25 | pack-rtt25 | proxy-rtt60 | pack-rtt60 | proxy-rtt25link500 | pack-rtt25link500 | r2s3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ms route | 2.2 / 2.5 | 2.5 / 3.2 | 2.3 / 2.6 | 2.3 / 2.5 | 2.2 / 2.4 | 2.0 / 3.6 | 3.2 / 4.1 |
| ms phase1 | 134.5 / 143.9 | 131.3 / 136.1 | 261.4 / 275.6 | 259.8 / 267.6 | 157.2 / 170.9 | 164.7 / 177.7 | 301.5 / 318.9 |
| ms phase2 | 414.7 / 456.3 | 353.1 / 376.2 | 834.8 / 918.3 | 693.4 / 774.3 | 404.0 / 427.4 | 305.2 / 343.5 | 960.0 / 1049.2 |
| ms query total | 564.2 / 591.5 | 481.6 / 515.8 | 1090.5 / 1174.7 | 956.8 / 1041.5 | 564.3 / 601.2 | 456.1 / 497.7 | 1293.0 / 1357.9 |
| req phase1 | 34 | 34 | 34 | 34 | 34 | 34 | 34 / 40 |
| req phase2 | 143 / 149 | 143.5 / 149 | 143.5 / 148 | 145 / 150 | 144.5 / 150 | 144.5 / 150 | 143.5 / 148 |
| req query | 177 / 183 | 177.5 / 183 | 177.5 / 182 | 179 / 184 | 178.5 / 184 | 178.5 / 184 | 179.5 / 186 |
| MB phase1 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 6.00 / 7.33 |
| MB phase2 | 0.22 | 0.22 | 0.22 | 0.22 / 0.23 | 0.22 | 0.22 | 0.22 / 0.23 |
| MB query | 6.22 / 7.55 | 6.22 / 7.55 | 6.22 / 7.54 | 6.22 / 7.55 | 6.22 / 7.54 | 6.22 / 7.54 | 6.22 / 7.54 |
| cache hits (query) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scan prefix MB | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 |
| slab MB transferred | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 |
| rerank reads | 143 / 149 | 143.5 / 149 | 143.5 / 148 | 145 / 150 | 144.5 / 150 | 144.5 / 150 | 145.5 / 152 |
| rerank local / spec used | 149 | 149 / 150 | 149 / 150 | 149 | 149.5 / 150 | 149.5 / 150 | 149 / 150 |
| proxy requests | 177 / 183 | 177.5 / 183 | 177.5 / 182 | 179 / 184 | 178.5 / 184 | 178.5 / 184 | - |
| proxy MB | 6.22 / 7.55 | 6.22 / 7.55 | 6.22 / 7.54 | 6.22 / 7.55 | 6.22 / 7.54 | 6.22 / 7.54 | - |
| proxy new connections | 0 | 0 / 4 | 0 | 0 / 4 | 0 | 0 / 4 | - |
| proxy upstream fetches (want 0) | 0 | 0 | 0 | 0 | 0 | 0 | - |
| recall@10 | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% |

**warm** (median / p90; recall@10: mean / min)

| metric | proxy-rtt25 | pack-rtt25 | proxy-rtt60 | pack-rtt60 | proxy-rtt25link500 | pack-rtt25link500 | r2s3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ms route | 3.5 / 3.8 | 2.7 / 3.4 | 3.5 / 4.0 | 2.9 / 3.9 | 3.4 / 3.8 | 2.6 / 3.2 | 3.5 / 4.0 |
| ms phase1 | 132.7 / 138.4 | 131.2 / 142.4 | 263.1 / 267.2 | 254.4 / 266.5 | 159.9 / 183.5 | 155.1 / 173.1 | 299.9 / 342.9 |
| ms phase2 | 396.7 / 424.0 | 343.1 / 389.4 | 847.0 / 897.6 | 719.0 / 761.6 | 379.2 / 401.3 | 310.5 / 350.5 | 876.2 / 1045.4 |
| ms query total | 536.4 / 561.4 | 470.9 / 518.2 | 1121.5 / 1166.9 | 978.2 / 1011.8 | 547.9 / 575.2 | 469.6 / 524.0 | 1175.5 / 1443.3 |
| req phase1 | 34 | 34 | 34 | 34 | 34 | 34 | 35 / 44 |
| req phase2 | 143 / 149 | 143 / 148 | 142.5 / 149 | 143 / 149 | 144.5 / 150 | 144.5 / 150 | 140.5 / 149 |
| req query | 177 / 183 | 177 / 182 | 176.5 / 183 | 177 / 183 | 178.5 / 184 | 178.5 / 184 | 177.5 / 183 |
| MB phase1 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 6.00 / 7.34 |
| MB phase2 | 0.22 | 0.22 | 0.22 | 0.22 | 0.22 | 0.22 | 0.22 |
| MB query | 6.22 / 7.55 | 6.22 / 7.54 | 6.22 / 7.55 | 6.22 / 7.55 | 6.22 / 7.54 | 6.22 / 7.54 | 6.22 / 7.55 |
| cache hits (query) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scan prefix MB | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 |
| slab MB transferred | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 |
| rerank reads | 143 / 149 | 143 / 148 | 142.5 / 149 | 143 / 149 | 144.5 / 150 | 144.5 / 150 | 143.5 / 149 |
| rerank local / spec used | 149 | 149 / 150 | 149 / 150 | 149 | 149.5 / 150 | 149.5 / 150 | 149.5 / 150 |
| proxy requests | 177 / 183 | 177 / 182 | 176.5 / 183 | 177 / 183 | 178.5 / 184 | 178.5 / 184 | - |
| proxy MB | 6.22 / 7.55 | 6.22 / 7.54 | 6.22 / 7.55 | 6.22 / 7.55 | 6.22 / 7.54 | 6.22 / 7.54 | - |
| proxy new connections | 0 | 0 | 0 | 0 | 0 | 0 | - |
| proxy upstream fetches (want 0) | 0 | 0 | 0 | 0 | 0 | 0 | - |
| recall@10 | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% |

### Python SDK (default 256 MB MemoryCache)

**process-cold** (median / p90; recall@10: mean / min)

| metric | proxy-rtt25 | pack-rtt25 | proxy-rtt60 | pack-rtt60 | proxy-rtt25link500 | pack-rtt25link500 | r2s3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ms boot (spawn -> main) | 34.5 / 38.0 | 36.5 / 47.0 | 35.0 / 39.0 | 37.0 / 41.0 | 34.0 / 38.0 | 36.0 / 41.0 | 33.0 / 39.0 |
| ms import SDK | 25.6 / 26.1 | 25.9 / 27.6 | 24.3 / 25.8 | 25.6 / 28.6 | 24.4 / 26.3 | 28.3 / 31.3 | 22.7 / 23.4 |
| ms open.pack | - | 144.9 / 148.1 | - | 321.5 / 328.4 | - | 136.3 / 144.9 | - |
| ms open.current | 76.3 / 76.9 | 25.8 / 26.7 | 180.6 / 180.9 | 60.0 / 60.5 | 75.5 / 76.1 | 25.1 / 25.8 | 92.3 / 113.8 |
| ms open.manifest | 26.6 / 27.4 | 26.6 / 27.0 | 61.1 / 61.4 | 60.5 / 61.6 | 26.9 / 27.5 | 26.4 / 27.1 | 77.9 / 85.5 |
| ms open.slabs | 36.2 / 37.3 | 34.4 / 36.5 | 70.8 / 70.9 | 70.7 / 73.2 | 34.8 / 36.5 | 31.4 / 35.0 | 87.7 / 96.1 |
| ms open.base | 1113.3 / 1121.0 | 1112.5 / 1118.1 | 2212.6 / 2213.6 | 2213.3 / 2225.5 | 1114.2 / 1122.8 | 1103.8 / 1115.5 | 513.0 / 1213.2 |
| ms   base fetch | 1096.1 / 1097.2 | 1095.0 / 1096.6 | 2200.6 / 2200.9 | 2201.6 / 2209.1 | 1094.8 / 1097.0 | 1089.5 / 1096.6 | 490.7 / 1194.9 |
| ms   base build | 17.9 / 24.0 | 16.9 / 22.5 | 11.7 / 12.9 | 11.7 / 20.2 | 19.2 / 25.2 | 14.4 / 21.3 | 15.3 / 26.6 |
| ms open total | 1254.6 / 1259.2 | 1345.2 / 1352.6 | 2525.2 / 2526.5 | 2725.4 / 2752.0 | 1251.8 / 1263.2 | 1329.7 / 1346.3 | 778.5 / 1450.8 |
| ms route | 2.0 / 2.5 | 2.0 / 2.3 | 2.0 / 2.5 | 1.9 / 2.5 | 2.2 / 2.6 | 2.3 / 2.6 | 2.0 / 2.3 |
| ms phase1 | 470.4 / 515.1 | 467.0 / 516.1 | 1006.8 / 1082.8 | 1027.1 / 1098.2 | 885.4 / 985.1 | 881.6 / 984.9 | 673.0 / 795.6 |
| ms phase2 | 1.9 / 2.1 | 1.6 / 2.0 | 0.5 / 0.6 | 0.4 / 1.8 | 1.8 / 2.1 | 1.1 / 1.7 | 1.8 / 2.0 |
| ms query total | 476.1 / 519.7 | 471.4 / 520.8 | 1009.5 / 1085.9 | 1031.5 / 1101.5 | 890.5 / 990.4 | 885.8 / 988.7 | 677.8 / 800.3 |
| ms process e2e (spawn -> hits) | 1812.5 / 1848.0 | 1892.0 / 1928.0 | 3605.0 / 3685.0 | 3863.0 / 3908.0 | 2215.0 / 2316.0 | 2301.5 / 2394.0 | 1609.0 / 2187.0 |
| req open (pack+cur+man+slabs+base) | 6 | 9 | 6 | 9 | 6 | 9 | 6 |
| req phase1 | 34 | 34 | 34 | 34 | 34 | 34 | 34 |
| req phase2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| req query | 34 | 34 | 34 | 34 | 34 | 34 | 34 |
| MB open | 25.66 | 25.89 | 25.66 | 25.89 | 25.66 | 25.89 | 25.66 |
| MB phase1 | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 |
| MB phase2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MB query | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 |
| cache hits (query) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scan prefix MB | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 | 6.22 / 6.95 |
| slab MB transferred | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 | 50.82 / 56.95 |
| rerank reads | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rerank local / spec used | 150 | 150 | 150 | 150 | 150 | 150 | 150 |
| proxy requests | 40 | 43 | 40 | 43 | 40 | 43 | - |
| proxy MB | 76.48 / 82.61 | 76.71 / 82.84 | 76.48 / 82.61 | 76.71 / 82.84 | 76.48 / 82.61 | 76.71 / 82.84 | - |
| proxy new connections | 33 | 33 | 33 | 33 | 33 | 33 | - |
| proxy upstream fetches (want 0) | 0 | 0 | 0 | 0 | 0 | 0 | - |
| recall@10 | 88.0% / 60.0% | 88.0% / 60.0% | 88.0% / 60.0% | 88.0% / 60.0% | 88.0% / 60.0% | 88.0% / 60.0% | 88.0% / 60.0% |

**slab-cold** (median / p90; recall@10: mean / min)

| metric | proxy-rtt25 | pack-rtt25 | proxy-rtt60 | pack-rtt60 | proxy-rtt25link500 | pack-rtt25link500 | r2s3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ms route | 3.5 / 4.1 | 3.5 / 3.7 | 1.9 / 2.5 | 2.1 / 3.6 | 3.5 / 3.9 | 3.1 / 3.6 | 3.5 / 3.9 |
| ms phase1 | 300.9 / 354.8 | 299.9 / 348.9 | 609.5 / 779.2 | 629.8 / 729.7 | 822.9 / 1033.8 | 849.8 / 1034.2 | 592.4 / 697.8 |
| ms phase2 | 1.7 / 1.8 | 1.6 / 1.8 | 0.4 / 0.5 | 0.4 / 0.8 | 1.5 / 1.7 | 1.1 / 1.7 | 1.5 / 1.7 |
| ms query total | 307.1 / 361.0 | 306.2 / 355.1 | 612.3 / 781.8 | 635.4 / 732.3 | 829.3 / 1039.1 | 855.3 / 1038.7 | 598.6 / 705.2 |
| req phase1 | 34 | 34 | 34 | 34 | 34 | 34 | 34 |
| req phase2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| req query | 34 | 34 | 34 | 34 | 34 | 34 | 34 |
| MB phase1 | 47.85 / 60.03 | 47.85 / 60.03 | 47.85 / 60.03 | 47.85 / 60.03 | 47.85 / 60.03 | 47.85 / 60.03 | 47.85 / 60.03 |
| MB phase2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MB query | 47.85 / 60.03 | 47.85 / 60.03 | 47.85 / 60.03 | 47.85 / 60.03 | 47.85 / 60.03 | 47.85 / 60.03 | 47.85 / 60.03 |
| cache hits (query) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scan prefix MB | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 |
| slab MB transferred | 48.88 / 60.03 | 48.88 / 60.03 | 48.88 / 60.03 | 48.88 / 60.03 | 48.88 / 60.03 | 48.88 / 60.03 | 48.88 / 60.03 |
| rerank reads | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rerank local / spec used | 150 | 150 | 150 | 150 | 150 | 150 | 150 |
| proxy requests | 34 | 34 | 34 | 34 | 34 | 34 | - |
| proxy MB | 47.32 / 60.03 | 47.32 / 60.03 | 47.32 / 60.03 | 47.32 / 60.03 | 47.32 / 60.03 | 47.32 / 60.03 | - |
| proxy new connections | 0 | 0 | 0 | 0 | 0 | 0 | - |
| proxy upstream fetches (want 0) | 0 | 0 | 0 | 0 | 0 | 0 | - |
| recall@10 | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% |

**warm** (median / p90; recall@10: mean / min)

| metric | proxy-rtt25 | pack-rtt25 | proxy-rtt60 | pack-rtt60 | proxy-rtt25link500 | pack-rtt25link500 | r2s3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ms route | 3.6 / 3.9 | 3.2 / 3.7 | 2.0 / 2.7 | 2.2 / 2.3 | 3.3 / 4.0 | 3.3 / 3.5 | 3.5 / 4.0 |
| ms phase1 | 7.4 / 8.4 | 6.2 / 6.8 | 2.5 / 2.9 | 2.9 / 5.1 | 6.0 / 6.6 | 4.4 / 7.5 | 6.1 / 7.0 |
| ms phase2 | 1.6 / 1.8 | 1.6 / 1.8 | 0.4 | 0.4 / 1.0 | 1.5 / 1.7 | 1.1 / 1.7 | 1.6 / 1.7 |
| ms query total | 13.0 / 14.8 | 11.8 / 13.1 | 5.2 / 5.5 | 5.4 / 9.1 | 11.1 / 13.0 | 9.4 / 13.4 | 12.3 / 13.4 |
| req phase1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| req phase2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| req query | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| MB phase1 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 |
| MB phase2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MB query | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 |
| cache hits (query) | 34 | 34 | 34 | 34 | 34 | 34 | 34 |
| scan prefix MB | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 | 5.99 / 7.32 |
| slab MB transferred | 48.88 / 60.03 | 48.88 / 60.03 | 48.88 / 60.03 | 48.88 / 60.03 | 48.88 / 60.03 | 48.88 / 60.03 | 48.88 / 60.03 |
| rerank reads | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rerank local / spec used | 150 | 150 | 150 | 150 | 150 | 150 | 150 |
| proxy requests | 0 | 0 | 0 | 0 | 0 | 0 | - |
| proxy MB | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - |
| proxy new connections | 0 | 0 | 0 | 0 | 0 | 0 | - |
| proxy upstream fetches (want 0) | 0 | 0 | 0 | 0 | 0 | 0 | - |
| recall@10 | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% | 95.0% / 60.0% |

## Where cold time goes

Process-cold, median, TS SDK through the proxy at `rtt25` (e2e 1925 ms) and against real R2 (e2e 2319 ms):

| step | rtt25 ms | share | r2s3 ms | share | what it is |
|---|---:|---:|---:|---:|---|
| process, import, wasm compile + init | 45 | 2% | 52 | 2% | Node boot 23-30 ms; `wasmInit` 14 ms of which ~13 ms is Node loading its lazy `fetch`/`Response` globals (see below) |
| open chain: current -> manifest -> slabs | 161 | 8% | 320 | 14% | 3 serial GETs, the first also paying TCP + TLS (+50 ms in the model) and Node's first-`fetch` setup |
| open.base | 1133 | 59% | 557 | 24% | ONE 25.2 MB GET (`fine.105.f32`, 4103 x 1536 f32) + 1.6 MB `pq.bin` + 6 KB `coarse.f32`, then 22 ms building the wasm `Searcher` |
| phase 1 | 183 | 10% | 364 | 16% | 34 scan-prefix GETs (6.2 MB), 12 in flight, 12 new connections |
| phase 2 | 393 | 20% | 932 | 40% | ~140 single-record GETs (1540 B each, 0.22 MB in all), 12 in flight: ~12 serial round trips |

Python at `rtt25` (e2e 1813 ms): process + import 60 ms, open chain 139 ms, base 1113 ms (61%), phase 1 470 ms
(26%: 34 whole-slab GETs, 51 MB, plus 33 new connections), phase 2 2 ms. Against R2 (e2e 1609 ms): base 513 ms,
phase 1 673 ms.

1. **Base load is the largest single cost, and it is one object.** `fine.f32` is 98% of the 25.7 MB an open reads
   and arrives over one connection, so it is bound by single-stream bandwidth: 1.1 s at 200 Mbps, 2.2 s at
   100 Mbps, 0.53 s (p90 1.1 s) from R2 here. It is ~60% of process-cold time in every proxy profile, for both
   SDKs, and nothing else in the open can start the query before it is complete (routing needs all 4103 means).
   The packed open does not change this (the same 25.2 MB range).
2. **Round trips, not bytes, dominate the TypeScript query.** With no cache in Node and no multi-range GET on S3,
   phase 2 is ~140 one-record GETs; at 12 in flight that is ~12 sequential round trips: 393 ms at 25 ms, 757 ms at
   60 ms, 932 ms against R2, for 0.22 MB. Phase 2 is 70% of the TS query and does not move when bandwidth is
   capped (`rtt25link500`: 564 ms slab-cold either way). Because nothing is cached, `warm` costs the same 177
   requests and ~540 ms as `slab-cold`: every Node query is a cold query. Phase 2 is also very sensitive to the
   in-flight limit (an A/B with `inflight` 14 instead of 12 cut it from ~390 to ~270 ms through the proxy).
3. **Python trades round trips for 8x the bytes, and many connections.** Its default `MemoryCache` turns every
   phase-1 miss into a whole-slab read (51 MB per cold query, 8.2x the 6.2 MB of prefixes), after which phase 2 is
   free and `warm` costs 0 requests and 13 ms. With per-response bandwidth that wins (307 ms slab-cold vs TS 564 ms
   at rtt25; 599 vs 1293 ms against R2), but on a shared 500 Mbps link the same query takes 829 ms (TS: 564 ms):
   it is bandwidth-bound. Its `S3BlobStore` keeps one connection per I/O thread, so a process-cold query opens
   33 connections (TS: 12), which costs ~170 ms of the 470 ms process-cold phase 1 at rtt25 (slab-cold: 301 ms).
4. **The open chain is serial.** current.json -> manifest -> slabs table -> base files are four dependent steps:
   161 ms at rtt25, 336 ms at rtt60, 320 ms against R2 (TS), before the 25 MB read even starts.
5. **Signing is not a bottleneck.** TS `signV4` (WebCrypto: 5 HMAC key imports and signs plus a digest, all async)
   costs 125-190 us per request (measured by each run, `signUsPerRequest` in the meta row), ~25-30 ms of event-loop time over a 180-request query, mostly overlapped with
   waiting; Python's `sign_v4` costs 9 us.

### Per-slab byte accounting

`node bench/cold/slab_bytes.mjs data/cold/baseline-py-proxy-rtt25.jsonl` (rerank record = 4 + 1536 = 1540 B):

| | slabs | entries median / p90 | scan prefix median / p90 | rerank tier median / p90 | whole slab median / p90 | total prefix / whole |
|---|---:|---:|---:|---:|---:|---:|
| all cells | 4103 | 657 / 1250 | 139.5 / 260.0 KB | 988.1 / 1879.9 KB | 1127.6 / 2139.9 KB | 604.5 MB / 4900.3 MB (12.3%) |
| probed by queries 0..29 | 820 | 821 / 1423 | 172.8 / 295.1 KB | 1234.7 / 2140.1 KB | 1407.5 / 2435.2 KB | 148.1 MB / 1208.9 MB (12.3%) |

Probed slabs are larger than the median cell (big cells are probed more often). Per cold query (median): scan
prefixes 6.2 MB; TS adds the 150 rerank records it needs, 0.22 MB in ~140 GETs; Python reads the whole slabs,
50.3 MB in 34 GETs, of which the rerank records it uses are 0.23 MB (0.5%). Reading slabs whole moves 8.1x the
bytes of reading prefixes.

### Packs

The pack (`--store pack`) is the same objects behind one key, so the query reads are identical (same request
count and bytes; each slab read is a range of the pack). What changes is the open: `PackedBlobStore.open` does a
HEAD, a 64 KB tail read and, because the directory of 4110 entries is 175 KB and does not fit in the tail, a
directory read: 3 serial requests before the usual chain, which then runs as ranges of the pack. That adds
~155 ms (TS) / ~90 ms (Python) to the open at rtt25 and ~355 / ~200 ms at rtt60; the pack's own open is 218 /
145 ms at rtt25 and includes the connection handshake that the per-object store pays in `open.current`.

Entry order is key order, as `packIndex` writes it: `idx/current.json` @16, `coarse.f32` @40, `fine.105.f32` @6184
(25.2 MB), `manifest.179.json`, `pq.bin`, then the 4103 slabs from 26.8 MB to 5.16 GB, then `slabs.179.json`
(121 KB, sorted after `slab/` because `/` < `s`), `tree.105.bin` (804 KB, never read by the SDKs), the directory
(175 KB at 5.166 GB) and the footer. So the objects an open needs sit at both ends: current, coarse, fine, manifest
and pq are one contiguous 26.8 MB range at the head; the slabs table is ~1 MB before the end, behind the tree.

The TS pack runs show a faster phase 2 (321 vs 393 ms at rtt25) with the same requests. That is not the pack: it is
a bug in the SDK's `Limiter` (below), which the pack path triggers. Compare pack and per-object TS phase-2 numbers
only after that is fixed.

## Surprises and caveats

- **`Limiter` oversubscribes (TS SDK bug).** `Limiter.run` increments `active` only after a woken waiter resumes,
  so when a slot is released and a new caller arrives in the same tick, both get in. With a slot taken and released
  without I/O, which `fetchRecords` does on a `PackedBlobStore` over S3 (`getRanges` answers null at once, then the
  single-range reads follow), a standalone reproduction reaches 24 reads in flight under a limit of 12; through the
  proxy the pack path averaged 13.5 in flight vs 10.9 for per-object keys. Not fixed here (it changes query
  behaviour under the baseline); the harness will show the effect of the fix.
- **Node's lazy `fetch` globals cost 13-18 ms** in a cold process, paid by whatever touches `Response`/`fetch`
  first. wasm-bindgen's init does `typeof Response`, so it lands in `wasmInit`. The bench takes care not to touch
  them earlier (an early `fetch` to the proxy had moved that cost out of the measurement; the TS runs were redone).
- **TS `open.current` is ~96 ms at rtt25** (25 ms rtt + 50 ms handshake + ~20 ms of Node's first-`fetch` setup).
- **The network model is simple.** rtt is time to first byte with zero server time; bandwidth is per response
  unless `--link-mbps` is set, so in `rtt25` / `rtt60` twelve parallel responses get 12x the bandwidth, which
  flatters Python's whole-slab reads (see `rtt25link500`). TLS is modelled as `--conn-ms` on a connection's first
  request; the proxy speaks HTTP/1.1 with keep-alive like R2's endpoint. R2 itself measured ~75-105 ms per serial
  request and ~380 Mbps per stream from this host.
- **The proxy pins a revision.** It caches `idx/current.json` like everything else, so it serves rev 179 until
  `data/cold/objects/idx/current.json` is deleted; `--store r2s3` reads whatever is live.
- **Small samples, and R2 is not deterministic.** n = 10 per scenario. Proxy runs repeat closely: the TS rerun
  moved every proxy median by under 20 ms apart from the `wasmInit` shift. r2s3 does not: the TS r2s3 query median
  was 1163 ms in the first run and 1305 ms an hour later, and its base fetch p90 is 1.1 s (median 0.53 s). Judge
  optimizations on the proxy profiles; use r2s3 as a sanity check.
- The Python extension was built with `maturin develop --release` (cargo `release` profile, not the `python` profile
  the README uses; only panic behaviour differs). Python's `load_rev_state` reads the slabs table and tombstone sets
  one after the other (TS reads them concurrently); g4 rev 179 has no tombstone sets, so it does not show here.

## Hot sets, pack order and the open chain

Three ideas measured on the harness above (branch `cold/open`): seed the hottest slab prefixes from one object at
open (**warm packs**), order a pack so a query's slabs sit together and merge their reads (**tree order +
`coalesceAcrossSlabs`**), and drop the manifest GET from the open (**inline manifest**). All runs use
`bench/cold/run_open.sh`: the same benches, queries and profiles as the baseline, through an `--offline` proxy on its
own port and overlay (`data/cold/open-overlay`), n = 10, files `data/cold/open/*.jsonl`. Cells are medians.

**Phase-2 caveat.** The TS `Limiter` bug (above) makes any path that takes a read slot and releases it without I/O
look faster in phase 2. A warm-pack layer exposing an always-null `getRanges` did exactly that in a first run (phase 2
418 -> 321 ms with no warm hit at all); the layer now has `getRanges` only when its inner store does, and an empty
warm pack measures like the baseline. Warm prefix hits still release slots without I/O, so the warm-pack and pack
numbers below (`lf-*` files) were taken with the store agent's `Limiter` fix applied locally and never committed;
the controls (`lf-base`, `lf-pack-key`) ran the same way. Python has no such bug.

### Warm packs

**How skewed is probe frequency?** `bench/cold/hot_cells.mjs` routes queries 1000..2999 (68,000 probes at nprobe 34)
with the SDK's own routing (only the base is read, never a slab): 3,986 of the 4,103 cells are probed at least once;
the most-probed cell by 8.9% of the queries; the median probed cell 14 times (0.7%). The skew is real but mild,
because these queries follow the data distribution (dbpedia queries); a production query log with repeated or
topical queries may well be more skewed, and `hot_cells.mjs` is the tool to check it.

| top N cells | share of probes (sample) | uniform would be | scan-prefix MiB (= warm pack size) | warm pack bytes |
|---:|---:|---:|---:|---:|
| 64 | 7.4% | 1.6% | 18.4 | - |
| 128 | 12.3% | 3.1% | 33.8 | 35,476,831 |
| 256 | 20.3% | 6.2% | 62.3 | 65,336,909 |
| 512 | 33.2% | 12.5% | 116.2 | 121,871,483 |
| 1024 | 53.2% | 25.0% | 207.6 | - |
| 305 (by probes per byte, 32 MiB budget) | 15.2% | 7.4% | 32.0 | 33,572,105 |

Hot cells are big cells (a cell is probed often because it is dense), so the pack grows faster than the coverage;
ranking by probes per prefix byte (`make_warm_pack.mjs --rank density --budget-mb 32`) covers 15.2% of probes in
32 MiB where the top-count ranking covers 11.6%.

**Hot-set hit rate on queries 0..29** (held out: not in the sample that chose the cells), share of each query's 34
probes inside the top N:

| N | mean | median | min | max | probes in the set per query (queries 0..29) |
|---:|---:|---:|---:|---:|---|
| 128 | 8.4% | 4.4% | 0% | 64.7% | 0 0 8 7 1 2 1 1 1 4 7 0 4 0 0 2 5 3 0 2 0 0 2 22 0 4 4 0 0 6 |
| 256 | 15.5% | 11.8% | 0% | 67.6% | 7 4 10 9 2 8 2 5 4 6 11 0 6 1 0 4 12 4 4 3 0 0 3 23 0 8 10 2 0 10 |
| 512 | 28.7% | 26.5% | 0% | 79.4% | 13 16 15 11 8 18 3 12 8 12 17 4 11 4 1 6 17 8 10 4 2 0 7 27 0 14 20 8 0 17 |
| 1024 | 50.7% | 51.5% | 8.8% | 94.1% | 15 28 21 20 22 32 6 26 18 22 29 8 17 15 4 9 23 13 22 12 6 7 14 28 3 22 29 15 5 26 |

(byte share is a little higher: 11.1 / 19.8 / 35.6 / 57.2% of the prefix bytes.) Close to the sample coverage, so the
hot set generalizes; it is simply not very hot.

**The format.** A warm pack is a pack (`portability.md`) with `meta.kind: "warm"` whose entries are slab scan
prefixes under the slabs' own keys, each marked `[key, offset, length, {"partial": true, "size": n}]` (pack version
stays 1; partial entries appear only in warm packs). `open({ warmPack: key })` / `open_index(..., warm_pack=key)`
issues ONE GET for it at the very start of the open, never waits for it, and when it lands seeds a `WarmPrefixStore`
in front of the cache with the prefixes of slabs the open revision reads (`index.warmReady` / `index.warm_ready`,
`index.warmStats`). A read of a range inside a held prefix is served from memory (a cache hit in the stats); phase-2
records and whole-object reads go to the store. Behind a cache (Python's default `MemoryCache`, the Workers Cache
API) a served prefix read that asked for `fill` also fetches the whole slab into that cache in the background, so the
warm pack does not keep the cache cold. `PrefixSeed` (`seedPrefix(key, bytes)`) is the hook an in-process cache can
implement to take the prefixes instead (it must keep them prefix-only).

**Measured** (slab-cold: the warm pack was seeded before the query; `--warm-wait slab`):

| TS, proxy | rtt25 phase1 | rtt25 query | phase-1 GETs | warm hits | seeded after open start |
|---|---:|---:|---:|---:|---:|
| baseline (`lf-base`) | 127.1 | 551.8 | 34 | 0 | - |
| N = 128 (35 MB) | 124.8 | 520.5 | 33.5 | 2 | 1.5 s |
| N = 256 (65 MB) | 119.7 | 516.5 | 36 * | 4 | 2.8 s |
| N = 512 (122 MB) | 105.3 | 500.5 | 30 | 9.5 | 5.1 s |
| density, 32 MiB | 130.2 | 541.0 | 33.5 | 2 | 1.5 s |

(* phase-1 GETs include speculative rerank reads that land in phase 1.) N = 512 at the other profiles: slab-cold phase
1 257 -> 240 ms (rtt60), 162 -> 120 ms (rtt25link500). The query totals move by less than phase 2's run-to-run noise
(+-30 ms at rtt25, more at rtt60).

| Python, proxy, slab-cold query ms | rtt25 | rtt60 | rtt25link500 |
|---|---:|---:|---:|
| baseline | 305.3 | 630.0 | 1037.0 |
| N = 256 | 294.6 | - | - |
| N = 512 | 247.6 | 545.6 | 745.3 |

Python gains more (a warm hit skips a whole-slab GET of 8x the prefix bytes, and on a shared link those bytes are the
bottleneck), but it pays elsewhere: process-cold at rtt25link500 got slower (895 -> 1028 ms) because the 122 MB warm
GET shares the link with the query's 51 MB, and the `warm` scenario (the same query again) went from 11 ms to 36-74 ms
because some hot slabs' background fills were still running and their rerank records came from S3.

**Verdict: not worth it for this workload.** A warm pack never helps the first query of a process (35-122 MB arrives
1.5-5 s after the open starts, at 200 Mbps per response, while the 25 MB base takes 1.1 s), and for later queries it
removes 2-10 of 34 prefix reads, which shortens phase 1 by 0-22 ms (TS) of a 500-550 ms query whose cost is phase 2's
~140 record GETs. Per process it costs a download 6-20x the bytes of a whole query. It would pay with a much more
skewed query log (a hot set of a few dozen cells covering half the probes), with a small pack that arrives with the
base (<= ~25 MB at 200 Mbps), or in Python on a bandwidth-limited link (-28% slab-cold at rtt25link500).

**Publishing one.** `bench/cold/make_warm_pack.mjs` builds one offline from `hot_cells.mjs`'s counts (from the proxy
cache; prefixes of unwarmed slabs are holes, like `make_pack.mjs`). In production the counts would come from query
logs: every query result carries its probed cells (`stats.cells`), a Worker can aggregate them per generation (a
counter per cell in a Durable Object or Analytics Engine), and `publish` / `publish-rev` would write
`idx/{gen}/warm.{rev}.threadpack` with `writeWarmPack(store, rev, topCells, sink)` right after the slab revisions it
names, and could record its key in the manifest so `open` finds it without an option. The Worker's publish is not
changed here.

### Pack order by centroid-tree group, and cross-slab coalescing

**Adjacency in the baseline pack** (`bench/cold/pack_adjacency.mjs`, queries 0..29, routing only): the pack lists
slabs in key order, i.e. cell id order, which k-means leaves unrelated to geometry. Pairs of a query's probed slabs
that are adjacent entries: median 0, mean 0.53 per query, 13 of 30 queries have any. A query's 34 probes fall in a
median of 7.5 of the tree's 128 groups (2..18).

**Tree order** (`make_pack.mjs --order tree`, `data/cold/open-overlay/packs/g4-cold-tree.threadpack`): slabs grouped by
centroid-tree group (`idx/g4/tree.105.bin`; the wasm exposes no group-of-cell helper, so `tree_order.mjs` reads the
tree's CSR member lists, all 4,103 cells are listed), groups chained by group-mean similarity, cells within a group
chained by nearest neighbour. Adjacent probed pairs: median 16.5, mean 15.9 per query, all 30 queries. Half the
probed slabs now have a probed neighbour in the pack.

**But a slab entry is [scan prefix | rerank tier | payload]**, so between two adjacent prefixes lies the first
slab's rerank tier, ~8x its prefix (~1 MB). `coalesceAcrossSlabs: GAP` (TS query option; `PackedBlobStore` only,
also behind a warm-pack layer) merges phase-1 prefix reads whose gap is at most GAP into one range of the pack:

| layout, gap (planned, queries 0..29) | phase-1 requests | MB read | x prefix bytes |
|---|---:|---:|---:|
| key order, any gap up to 1 MiB | 34 | 6.16 | 1.00 |
| tree order, gap <= 256 KiB | 34 | 6.16 | 1.00 |
| tree order, 1 MiB | 30 | 9.27 | 1.50 |
| tree order, 4 MiB | 13.5 | 39.3 | 6.47 |
| tree-split (not expressible today), gap 0 | 17.5 | 6.16 | 1.00 |
| tree-split, 256 KiB | 15 | 6.57 | 1.06 |

Measured (TS, `--store pack`, slab-cold phase 1 ms / requests / MB):

| pack, `coalesceAcrossSlabs` | rtt25 | rtt60 |
|---|---|---|
| key order, off | 133.7 / 34 / 5.99 | 258.1 / 34 / 5.99 |
| key order, 1 MiB | 144.1 / 34 / 6.33 | - |
| tree order, off | 137.1 / 34 / 5.99 | 264.0 / 34 / 5.99 |
| tree order, 64 KiB (rtt60: 256 KiB) | 138.7 / 34 / 5.99 (no merge) | 260.1 / 34 / 5.99 (no merge) |
| tree order, 1 MiB | 144.2 / 31 / 8.87 | 259.3 / 31 / 8.87 |
| tree order, 4 MiB | 868.3 / 17.5 / 36.7 | - |

**No gain.** With whole slabs as pack entries the order puts slabs together but not their prefixes, and merging
across a rerank tier trades a round trip for ~1 MB at single-stream bandwidth. A **split layout** (each group's
prefixes contiguous, then its tails) would halve the requests at no extra bytes, but it needs a pack entry with two
extents (a format change readers before it could not read), and the latency model in `pack_adjacency.mjs` (reads in
routing order, `inflight` at a time, rtt + bytes at the per-response rate; it is ~20-30 ms optimistic against the
measured key-order rows) says that pays only when the in-flight limit is small: phase 1 104 -> 94 ms (rtt25) and
237 -> 210 ms (rtt60) at 12 in flight, 197 -> 147 / 454 -> 322 ms at 6 in flight (a Worker reading S3 over its 6
connections), and nothing at 32 in flight (61 vs 90 ms: merged reads are fewer but serialize more bytes). Not
implemented; worth revisiting only for a 6-connection client.

### Inline manifest in current.json

`current.json -> manifest -> slabs table -> base` is the serial chain. The writers in both SDKs now write
`current.json` as `{"gen", "rev", "manifest": {...}}` when the manifest serializes to at most 16 KB
(`currentBody` / `current_body`, the same bytes in both SDKs; g4 rev 179's pointer grows from 22 B to 1,058 B).
Readers use a copy that checks out for (gen, rev), at open and on refresh, and otherwise fetch the manifest as before;
older readers ignore the field. The Worker's publish is unchanged, so pointers it writes carry no copy.

Measured with the copy in an overlay `idx/current.json` (`data/cold/open-overlay-inline`), process-cold:

| | TS rtt25 | TS rtt60 | Python rtt25 | Python rtt60 |
|---|---:|---:|---:|---:|
| open.manifest ms, before -> after | 29.8 -> 0.1 | 66.0 -> 0.1 | 26.2 -> 0 | 61.6 -> 0 |
| open total ms, before -> after | 1292.8 -> 1271.5 | 2595.2 -> 2532.7 | 1250.2 -> 1229.2 | 2549.8 -> 2486.4 |
| open requests | 6 -> 5 | 6 -> 5 | 6 -> 5 | 6 -> 5 |

One round trip (the larger pointer costs nothing measurable): ~25 ms at rtt25, ~63 ms at rtt60, ~87 ms against R2
from this host (the manifest step's median there). **Does it still pay on top of base packs?** A base-pack open
(`cold/base`) is `current.json -> base pack`, with the manifest GET issued in parallel only as a fallback, so the
copy saves no round trip there (it saves that fallback request). It does pay for every open that reads the files
instead of a base pack (`cold/base` fetches all of them as soon as the manifest is in: `current -> manifest -> files`
becomes `current -> files`), and for every refresh that finds a new revision (`current -> manifest -> slabs table`
becomes `current -> slabs table`), which is what a long-lived index does every time a writer commits. The cost is
~1 KB more per `current.json` poll (every 30 s per index). Kept: it is cheap and never slower.

## Options outside the SDK

What deployment can do, reasoned from the baseline (TS, Node, process-cold at rtt25: open chain 161 ms in 3 serial
GETs, base 1,133 ms for one 25 MB GET, phase 1 183 ms for 34 prefix GETs, phase 2 393 ms for ~140 record GETs).
**None of these were tested against the owner's AWS or Cloudflare account, deliberately**; the S3 Express estimate is
the proxy with a shorter first byte, which is a model, not a measurement of the service.

### A CDN in front of the bucket

The objects under `idx/` are immutable, so any CDN can cache them forever; only `idx/current.json` must stay
uncached (or cached for seconds). What a CDN changes is who pays the origin round trip: the first cold isolate or
process in a region fills the edge, every later one reads from the edge (first byte ~5-15 ms from a nearby edge
instead of 25-60 ms from S3, ~75-105 ms from R2 here), and the edge collapses concurrent misses of one object into
one origin request. For a fleet of cold isolates that is the base (25 MB, 59% of process-cold) and the hot prefixes
served at edge latency and edge bandwidth; for a single long-lived process it changes little.

- **CloudFront with origin access control (OAC) in front of S3.** A range GET is served from cache when the edge holds
  the whole object or that range; on a miss CloudFront forwards the range to S3 (possibly widened) and caches the
  partial response, so the second request for the same slab prefix hits. Multi-range requests are honoured in
  ascending order. Price: data out to the internet ~$0.085/GB in North America for the first 10 TB (S3 -> CloudFront
  origin fetches are free) plus ~$0.01 per 10,000 HTTPS requests, versus S3 direct at ~$0.09/GB and $0.0004 per 1,000
  GETs. A cold TS process reads ~32 MB in ~180 requests (25.7 MB open + 6.5 MB query): ~$0.003 per cold start
  through CloudFront; a query (~177 requests, 6.2 MB) ~$0.0007, hit or miss, since the edge bills bytes out. Inside the same AWS region S3 -> EC2/Lambda transfer is free, so a CDN there buys latency,
  not cost.
- **Cloudflare CDN on an R2 custom domain.** R2 egress is free and the CDN bandwidth is not billed; what you pay is R2
  Class B reads ($0.36 per million) on cache misses only. On a cold miss the cache fetches from the origin without the
  client's Range (or, with Origin Range Requests, in 1 MiB-aligned chunks) and answers the range from what it
  stored; repeated ranges then hit. Caching is per data center unless Tiered Cache is on. Inside Workers this is
  the same effect the Cache API already gives `CachedBlobStore` (per colo), so it matters mostly for clients outside
  Workers (Node/Python services) that would otherwise pay R2's 75-105 ms first byte from far away.

Estimated effect on the baseline for a cold process near an edge that already holds the objects (first byte ~10 ms,
~500 Mbps): open chain 161 -> ~60 ms (3 serial GETs plus the handshake), base 1,133 -> ~430 ms, phase 1 183 -> ~80 ms,
phase 2 393 -> ~150 ms: process-cold ~1.9 s -> ~0.8 s. The first process per edge pays the full origin cost.

### S3 Express One Zone

Directory buckets in one Availability Zone with single-digit-ms first byte (vs 20-60 ms for S3 Standard). Modelled with
the proxy at 5 ms first byte (+10 ms on a new connection) and the baseline's 200 Mbps per response, then 1 Gbps per
response for a client in the same AZ (`run_open.sh` profiles `rtt5`, `rtt5g1`; `data/cold/open/open-base-*-rtt5*`):

| process-cold, median ms | TS rtt25 | TS rtt5 | TS rtt5g1 | Python rtt25 | Python rtt5 | Python rtt5g1 |
|---|---:|---:|---:|---:|---:|---:|
| open (current + manifest + slabs) | 157 | 44 | 41 | 135 | 33 | 27 |
| open.base | 1,139 | 1,076 | 268 | 1,116 | 1,055 | 235 |
| phase 1 | 188 | 68 | 60 | 448 | 290 | 128 |
| phase 2 | 394 | 98 | 87 | 2 | 2 | 1 |
| query | 582 | 170 | 154 | 453 | 296 | 132 |
| process e2e | 1,966 | 1,356 | 536 | 1,786 | 1,482 | 479 |

The first byte fixes the TS query (582 -> 170 ms: ~175 requests at 12 in flight are ~15 round trips) and the open
chain; the base load needs bandwidth, which in-AZ placement gives (1,139 -> 268 ms at 1 Gbps). Together a TS cold
start drops from ~2.0 s to ~0.54 s. The catch is placement: the client must run in that AZ (one AZ, no regional
redundancy) and the bucket holds a copy of the index (a directory bucket; copy the snapshot with `copyObjects`).
Price (after the April 2025 cut, us-east-1): storage $0.11/GB-month (g4 rev 179 is ~5.2 GB: ~$0.57/month; S3 Standard
$0.023/GB), GETs $0.00003 per 1,000 (S3 Standard $0.0004), data retrieval $0.0006/GB on every byte. A cold TS start
(~180 requests, 32 MB) costs ~$0.000025; a warm query (~177 requests, 6.2 MB) ~$0.000009. Cheaper per request than S3
Standard, dearer to store.

### Offloading routing to a warm process

The routing table is 25 MB of f32 that every cold isolate downloads and keeps (1.1 s at rtt25, 0.56 s median / 1.1 s
p90 from R2 here; 25 MB of a 128 MB Workers isolate), to compute one 4,103 x 1,536 product per query (3 ms). A warm
process holding it (a Durable Object, or a Worker reached through a service binding) answers `route(vector, nprobe)`:
6 KB in, 34 cell ids out. The PQ codebook (1.6 MB) can stay local or travel with the answer as the per-query table
(192 x 256 f32, 192 KB).

- **Hop cost h per query:** a service binding to a Worker on the same machine is sub-millisecond to ~1 ms; a Durable
  Object in the same colo a few ms; a DO placed elsewhere (a DO lives in one location) 20-100 ms of network.
- **Saving B per cold isolate:** the base load (0.5-1.1 s) and its 25 MB of memory. The routing process loads it once
  (from R2 in-colo, a few hundred ms) and keeps it while it receives traffic; DOs are evicted when idle, so a quiet
  index pays that load again, once, for all isolates.
- **Break-even:** offloading wins while an isolate serves fewer than B / h queries in its lifetime. With B = 560 ms
  (R2 median): h = 5 ms -> 112 queries, h = 20 ms -> 28, h = 60 ms -> 9; with B = 1.1 s (p90, or rtt25) twice that.
  Spiky or low-traffic deployments, where most isolates are cold and serve a handful of queries, are on the
  offloading side; a busy isolate serving thousands of queries is not, unless memory is the constraint.
- **Cost:** DO requests ($0.15 per million) plus duration while active; a service binding call is billed as a request
  to the routing Worker. Pair it with `cold/base`'s smaller tables (f16/int8) and tree-grouped cold routing, which
  shrink B instead of moving it.
