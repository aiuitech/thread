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

## Routing base and base packs

The levers the baseline points at (the routing table is ~60% of process-cold; a pack open costs 3 serial requests
before the open starts), implemented on `cold/base` and described in
[query-path.md](query-path.md#cold-opens-compact-routing-cold-tree-routing-base-packs):

- **Compact routing tables** `fine.f16` (12.6 MB) / `fine.i8` (6.3 MB, int8 residuals from the means' centre),
  `open({routing})` / `open_index(routing=)`, decoded to f32 on load.
- **Cold tree routing** `coldRouting: "tree"` / `cold_routing="tree"`: the open loads the tree (0.8 MB) and the
  codebook; first queries range-read only their beam's groups of the tree-grouped table `fine.grouped.{rev}.i8`;
  the full table loads in the background.
- **Base packs** `idx/{gen}/base.{rev}.threadpack` (8.8 MB: manifest, slabs table, tombstones, coarse, codebook,
  tree, `fine.i8`) and `base.cold.{rev}.threadpack` (2.5 MB, no routing table): `current.json` + one GET; files
  requested together once the manifest is in when there is no pack (or the open's table is not in it).
- **Index packs**: `PackedBlobStore` opens with ONE suffix-range read (`getTail`, `Range: bytes=-524288`: footer +
  175 KB directory, no HEAD); `packIndex` writes `idx/current.json` and the base objects first and records
  `base: [offset, length]` and `baseRouting` (the table ending the range), so the open after that is one range read
  (minus the routing table when the open loads another one).

All objects for g4 were generated offline (`make_routing.mjs`, `make_base_pack.mjs`, `make_pack.mjs --overlay`) as
rev 180 = rev 179 + routing objects, served from `data/cold/base-overlay` by a second proxy on port 9761; nothing was
written to R2. Same queries (0..19), same proxy cache, n = 10, medians. Bytes are what the SDK read before its first
slab read (open + routing).

### TypeScript SDK, per-object store (`--store proxy`)

| profile | mode | MB before 1st slab read | open ms | process e2e ms | vs baseline e2e |
|---|---|---:|---:|---:|---:|
| rtt25 | baseline (f32, as before) | 25.66 | 1292 | 1925 | |
| rtt25 | f16 | 13.64 | 744 | 1363 | -29% |
| rtt25 | i8 + base pack | 8.42 | 556 | 1200 | -38% |
| rtt25 | cold beam 16 + cold pack | 3.42 | 275 | 1033 | -46% |
| rtt60 | baseline | 25.66 | 2587 | 3786 | |
| rtt60 | i8 + base pack | 8.42 | 1123 | 2373 | -37% |
| rtt60 | cold beam 16 + cold pack | 3.42 | 591 | 2043 | -46% |
| rtt25link500 | baseline | 25.66 | 1294 | 1955 | |
| rtt25link500 | i8 + base pack | 8.42 | 554 | 1233 | -37% |
| rtt25link500 | cold beam 16 + cold pack | 3.42 | 282 | 1070 | -45% |

The f32 rows of this branch (default routing, `basePack` on, which for f32 means the files prefetch) match the
baseline within 30-80 ms (rtt25 open 1251 ms, e2e 1881 ms): the routing table's bytes are the cost, and only a
smaller table or no table removes it. Full per-mode tables (f32 / f16 / i8 with and without base pack, cold beams
8 / 16 / 32, an rtt90 / 400 Mbps profile) are in query-path.md.

### Python SDK, per-object store, rtt25

| mode | open ms | process e2e ms | vs baseline e2e |
|---|---:|---:|---:|
| baseline (f32) | 1255 | 1813 | |
| f32 (this branch, files prefetch) | 1264 | 1818 | 0% |
| i8 + base pack | 479 | 1025 | -43% |
| cold beam 16 + cold pack | 213 | 810 | -55% |

(Python's cold groups are read in parallel on its I/O pool; S3 has no multi-range GET, so a beam of 16 is ~16
range requests in one round trip.)

### Packs (`--store pack`), rtt25

The old pack (`packs/g4-cold.threadpack`, rev 179, key order) and a base-first pack of rev 180
(`packs/g4-base.threadpack`, `make_pack.mjs --overlay data/cold/base-overlay`), both sparse, both through the proxy.
`open.pack` is the `PackedBlobStore` open; `open total` includes it.

| SDK | pack, mode | open.pack ms | open total ms | process e2e ms |
|---|---|---:|---:|---:|
| TS | baseline: old pack, old SDK (HEAD + 64 KB tail + directory) | 218 | 1447 | 2012 |
| TS | old pack, new SDK (one suffix read) | 110 | 1308 | 1897 |
| TS | base-first pack, f32 (base range up to `fine.i8`, then `fine.f32`) | 111 | 1339 | 1947 |
| TS | base-first pack, i8 (one range: 8.8 MB) | 111 | 536 | 1177 |
| TS | base-first pack, cold beam 16 | 111 | 516 | 1187 |
| Python | baseline: old pack, old SDK | 145 | 1345 | 1892 |
| Python | old pack, new SDK | 105 | 1327 | 1855 |
| Python | base-first pack, f32 | 105 | 1348 | 1899 |
| Python | base-first pack, i8 | 105 | 507 | 1061 |
| Python | base-first pack, cold beam 16 | 105 | 493 | 1107 |

The pack open went from three serial requests to one (-108 ms TS, -40 ms Python); a pack open is now
`tail read` + `one base range` (+ the routing table when the open wants another one than the pack's smallest).
With an f32 open, a pack is as fast as the per-object store again; with i8 it is 1.18 s process e2e instead of 2.0 s.
In a pack, cold routing gains nothing over i8: the base range is one read either way and the cold query pays a
round trip for its groups.

### Recall

Probe agreement over 2000 queries (no slab reads, `routing_agreement.mjs`): f16 99.98%, i8 99.82%, cold beam
8 / 16 / 32 with i8 rows 90.6% / 95.9% / 98.7% of flat-f32 probes (full table in query-path.md). End-to-end recall@10
over 2000 queries against R2 (`routing_recall.mjs`, every query routed by the mode): f32 95.11%, f16 95.11%, i8
95.11% (same top 10 as f32 for 99.6% of queries), cold beam 8 / 16 / 32 94.36% / 94.93% / 95.10%. On the 10
process-cold queries of the bench every mode returns the baseline's recall except cold beam 8 / 16 (one / two
queries lose a neighbour).
