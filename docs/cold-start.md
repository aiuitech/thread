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

## Store-level optimizations

What the SDKs now do between a query and S3, both SDKs with the same option names (TS camelCase / Python
snake_case). Every item was measured with the harness above (same pinned rev 179, same queries 0..19, n = 10 per
scenario); the option matrix is `bench/cold/run_store.sh` (one run per config, proxy on its own port, configs set
through `cold_bench` flags), the tables come from `node bench/cold/store_table.mjs`, and `TAG=store
bench/cold/run_baseline.sh` repeats the baseline runs (proxy and pack, three profiles) with the new defaults.

| # | option | default | what it does |
|---|---|---|---|
| 1 | `missPolicy` / `miss_policy` on `CachedBlobStore` and `open` | `whole` for the Workers Cache API, `range+warm` for `MemoryCache` / `DiskCache` | what a slab-prefix cache miss reads: the whole slab, exactly the range, or the range now and the whole slab into the cache in the background (2 at a time, 1 while the query's own reads are in flight, deduplicated, bounded queue, `waitUntil`) |
| 1 | Node's default cache | `MemoryCache` (256 MB) in front of a remote store outside Workers | before: none, so every Node query was cold |
| 2 | `inflight` (store hint, query option) | 32 for `S3BlobStore` in Node / Python, 6 for S3 inside a Worker, 12 for the Cache API and other stores | reads in flight per query; the TS `Limiter` is now exact (it oversubscribed, see below) |
| 2 | `coalesceBytes` / `coalesce_bytes` | 128 KB for `S3BlobStore`, 4 KB elsewhere | a slab's wanted rerank records are read as runs of up to this many bytes |
| 2 | `spanBytes` / `span_bytes` | off | one range from a slab's first to last wanted record when the span is at most this |
| 3 | SigV4 signing-key cache | on | the derived key per (date, region, service), imported once as an HMAC `CryptoKey` (TS) |
| 4 | `warmup` (open option; `S3BlobStore({ warmup })` at construction) | `inflight` connections for remote stores outside Workers | opens connections while `open` loads the index; TS sends one-byte GETs (a HEAD's connection is not reused by Node's `fetch`) and no longer cancels small bodies (which closed the socket); Python's `S3BlobStore` now has one bounded keep-alive pool shared by all threads instead of a connection per thread |
| 4 | `chunkBytes` / `chunk_bytes` on `S3BlobStore` | 8 MiB, 4 at a time | a whole-object GET above this becomes parallel range GETs (the 25 MB routing table) |
| 5 | `DiskCache` (`thread-sdk/node`, Python) | - | whole objects on disk in the bucket layout (shared by both SDKs, LRU by mtime, size-bounded); a `FileBlobStore`, so `copyObjects(s3, disk, snapshotKeys(rev))` pre-warms it |
| 6 | `coldPolicy` / `cold_policy` | off | `{nprobe, rerank, threshold}` for queries whose probed slabs are mostly (> 50%) not cached |
| 7 | prefetch on refresh, `warm(keys?)` | on (`prefetchBytes` 64 MB) | a new revision's manifest, slabs table, tombstones, base files (if changed) and deltas are loaded before the switch, the rest of `snapshotKeys(new) - snapshotKeys(old)` warmed afterwards; the periodic `current.json` check runs in the background |

How to read the tables: `process-cold` includes the open (e2e = spawn to hits); `slab-cold` is a new query in the
same process; `warm` repeats the `slab-cold` query right away, so under `range+warm` it usually runs while the
background warms of the previous query are still in flight (`default-drain` waits for them first: that is the warm
cost once the cache has filled). Requests and SDK MB are the query's own foreground reads; proxy MB is everything
the proxy sent during the scenario, background warms included. `legacy` is the new code configured like the old
defaults (TS: no cache, inflight 12, coalesce 4 KB, no warm-up, no chunking; Python: `MemoryCache` with `whole`
misses, inflight 12; Python's connection pool is shared in every config).

### The new defaults against the baseline

`TAG=store WARM=0 R2S3=0 bench/cold/run_baseline.sh` (both SDKs with no options, proxy and pack, three profiles),
medians, baseline -> now (`node bench/cold/store_table.mjs --tag store`):

| profile | store | sdk | process-cold e2e | process-cold query | slab-cold query | warm query | slab-cold requests | slab-cold proxy MB | process-cold proxy conns | recall@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rtt25 | proxy | ts | 1925 -> 1190 | 576 -> 155 | 564 -> 141 | 536 -> 136 | 177 -> 87 | 6.2 -> 11.2 | 12 -> 33 | 91.5 -> 91.5% |
| rtt25 | proxy | py | 1813 -> 1188 | 476 -> 139 | 307 -> 151 | 13 -> 148 | 34 -> 86 | 47.3 -> 11.9 | 33 -> 32 | 91.5 -> 91.5% |
| rtt25 | pack | ts | 2012 -> 1019 | 508 -> 141 | 482 -> 132 | 471 -> 143 | 178 -> 88 | 6.2 -> 11.8 | 17 -> 34 | 91.5 -> 91.5% |
| rtt25 | pack | py | 1892 -> 914 | 471 -> 149 | 306 -> 147 | 12 -> 151 | 34 -> 87 | 47.3 -> 11.2 | 33 -> 32 | 91.5 -> 91.5% |
| rtt25link500 | proxy | ts | 1955 -> 1251 | 603 -> 222 | 564 -> 220 | 548 -> 219 | 179 -> 86 | 6.2 -> 10.5 | 12 -> 33 | 91.5 -> 91.5% |
| rtt25link500 | proxy | py | 2215 -> 1286 | 891 -> 230 | 829 -> 243 | 11 -> 240 | 34 -> 88 | 47.3 -> 11.4 | 33 -> 32 | 91.5 -> 91.5% |
| rtt25link500 | pack | ts | 2033 -> 1190 | 522 -> 225 | 456 -> 216 | 470 -> 227 | 179 -> 85 | 6.2 -> 10.5 | 17 -> 34 | 91.5 -> 91.5% |
| rtt25link500 | pack | py | 2302 -> 1102 | 886 -> 230 | 855 -> 244 | 9 -> 239 | 34 -> 91 | 47.3 -> 11.7 | 33 -> 32 | 91.5 -> 91.5% |
| rtt60 | proxy | ts | 3786 -> 2234 | 1141 -> 277 | 1091 -> 280 | 1122 -> 254 | 178 -> 87 | 6.2 -> 11.2 | 12 -> 33 | 91.5 -> 91.5% |
| rtt60 | proxy | py | 3605 -> 2300 | 1010 -> 310 | 612 -> 308 | 5 -> 312 | 34 -> 89 | 47.3 -> 10.9 | 33 -> 32 | 91.5 -> 91.5% |
| rtt60 | pack | ts | 4011 -> 1942 | 1005 -> 261 | 957 -> 274 | 978 -> 257 | 179 -> 88 | 6.2 -> 10.8 | 17 -> 34 | 91.5 -> 91.5% |
| rtt60 | pack | py | 3863 -> 1773 | 1031 -> 305 | 635 -> 305 | 5 -> 309 | 34 -> 85 | 47.3 -> 11.3 | 33 -> 32 | 91.5 -> 91.5% |

Against the real R2 S3 endpoint (option matrix runs `legacy` -> `default`, one run each, R2 is +-12% between runs):
TS process-cold e2e 2142 -> 1445 ms, slab-cold query 1198 -> 440 ms (phase 2 869 -> 220 ms, 142 -> 48 requests);
Python e2e 1622 -> 1385 ms, slab-cold 610 -> 465 ms (47.8 -> 9.1 MB per query).

- **TypeScript**: cold queries are 2.6-4x faster (phase 2 was ~140 single-record GETs at 12 in flight; now ~35 runs
  of up to 128 KB at 32 in flight), and repeat queries are cached at all now (Node had no cache). The open is
  ~320 ms shorter at rtt25 (1285 -> 961 ms; `open.base` 1137 -> 794 ms: the 25 MB routing table as parallel range
  GETs).
- **Python**: cold queries 2-3.4x faster, 4-5x less transfer (the prefixes and records instead of whole slabs, which
  matters most on a shared link: 829 -> 243 ms slab-cold at `rtt25link500`). The price: a query repeated right
  after a cold one is no longer ~10 ms but ~140-310 ms until the background warms land (`warm` column); once they
  have (`default-drain` below), warm queries are 10-14 ms again. `miss_policy="whole"` restores the old trade.
- **Packs** gain the most at open (2012 -> 1019 ms TS, 1892 -> 914 ms Python e2e at rtt25; TS `open.base` 1141 ->
  422 ms): the routing table is one 25 MB range of the pack, now read as 4 parallel ranges in one round (a per-object whole GET needs a first chunk to
  learn the size, then the rest: two rounds).
- New connections per cold process: TS 12 -> 33 (inflight 32: warmed during the open, then reused by the query);
  Python 33 -> 32, from one shared pool.

### Option by option

1. **Miss policy.** On per-response bandwidth (`rtt25`) a whole-slab miss looks affordable at 32 in flight
   (slab-cold 222 ms TS / 171 ms Python with `whole-i32` vs 141 / 146 ms with `range+warm`), but on a shared link it
   is not: at `rtt25link500` `whole` costs 855 ms per slab-cold query (48 MB through 500 Mbps) against 226 / 234 ms
   for the prefix-and-records reads, and 351 vs 273 ms (TS) at `rtt60`. `range` and `range+warm` cost the same in
   the query (the warm traffic is background: proxy MB 11 vs 9); what `range+warm` adds is that a later query hits:
   `default-drain`'s warm query is 14 ms TS / 10 ms Python, where `range` stays at ~145 ms. The price is the query
   right after a cold one: until the background warms land (34 slabs at 2 at a time take ~1-2 s through the proxy)
   a repeat is still mostly cold (`warm` column of `default`), where Python's old `whole` default answered it in
   13 ms.
2. **In flight and coalescing.** These two are what moved the query. `inflight` 12 -> 32 cut the slab-cold query
   from 289 to 141 ms (TS) and 286 to 146 ms (Python) at `rtt25`; 6 (the Worker-direct-to-S3 default) would roughly
   double it again (v1 run: 910 ms TS). Coalescing a slab's records into runs of up to 4 KB / 32 KB / 128 KB /
   512 KB takes phase 2 from 148 to 71 / 35 / 18 requests (TS, slab-cold, `rtt25`) and the query from 261 to 196 /
   141 / 139 ms, for 6.2 / 6.8 / 9.1 / 12.4 MB per query. On the shared link 512 KB is already worse than 128 KB
   (277 vs 226 ms TS, 293 vs 234 ms Python): the extra bytes cost more than the saved round trips, so 128 KB is the
   default for `S3BlobStore`. At `rtt60`, 4 KB vs 128 KB is 472 vs 273 ms (TS) and 486 vs 305 ms (Python).
   `spanBytes` 1 MB (one range per winner slab when its records lie within 1 MB) is no better than 128 KB runs
   (142 vs 141 ms TS, 168 vs 146 ms Python) and reads more; it stays off.
3. **SigV4 key cache.** `signV4` per request: 135 us -> 24 us in Node (median of 32 runs; 115-238 -> 21-44), Python
   `sign_v4` 9 -> 5 us. Not visible in query latency (the signing overlapped the waits before too).
4. **Connections.** With `inflight` 32 a cold query needs 32 connections. Warming them during `open` saves 44 ms of
   process-cold query in TS (192 -> 148 ms) and 37 ms in Python (183 -> 146 ms) at `rtt25`; creating the store with
   `warmup` before `Thread.init` (`early-warmup`) also takes Node's lazy `fetch`/`Response` load out of `wasmInit`
   (13.0 -> 0.5 ms) and saves another ~35 ms of e2e (1209 -> 1175 ms). Two bugs made the first version of the TS
   warm-up useless (65 instead of 33 new connections per process): `S3BlobStore` cancelled small response bodies
   (HEAD, 404, PUT), which makes Node's `fetch` close the socket, and undici does not reuse a connection after a
   HEAD answer with a Content-Length; small bodies are now read to the end and the warm-up sends one-byte GETs.
   Python's store now keeps one bounded pool shared by all threads (32 connections per cold process, all reused by
   the query), instead of one connection per I/O thread. Splitting whole-object GETs above 8 MiB into 4 parallel
   ranges (`chunkBytes`) cuts the routing-table read from 1088 to 799 ms (TS) and 1458 -> 1172 ms of Python e2e at
   `rtt25`, and 2821 -> 2229 ms TS e2e at `rtt60`: it is the largest single open-time win here. Node's `fetch` is
   HTTP/1.1 (undici; HTTP/2 only with an `Agent` using `allowH2`); against R2, `nodeFetch({ allowH2: true })` was not
   faster for 32 parallel range GETs (hot rounds ~180 ms vs ~140 ms on HTTP/1.1), and a 60 s keep-alive
   (`nodeFetch({ keepAliveTimeout: 60000 })`) made a query after 6 s idle ~175 ms instead of ~260 ms (undici's 4 s
   default had closed the connections).
5. **Disk cache.** A restarted process over a filled `DiskCache` (`disk-restart`; the cache filled by
   `disk-fill`'s process, which drained its warms) answers process-cold in 472 ms e2e TS / 396 ms Python (79 / 36 ms
   query, 0 requests) instead of ~1200 ms. `DiskCache` hits are file range reads, so its slabs are not "whole" in
   memory and phase 2 reads records from disk (17 / 6 ms).
6. **Cold-path policy.** `coldPolicy {nprobe: 20, rerank: 100}` on the cold queries: recall@10 91.5% -> 86.5%
   (over the 20 cold queries), slab-cold 141 -> 128 ms (TS) / 146 -> 131 ms (Python) at `rtt25`, 226 -> 176 /
   234 -> 192 ms on the shared link, 273 -> 244 / 305 -> 241 ms at `rtt60`. It only applies while most probed
   slabs miss the cache, so warm queries keep full recall. Off by default: 5 points of recall for 10-25% of a cold
   query.
7. **Prefetch on refresh.** A revision switch (rev 179 -> 180 in a proxy overlay: new manifest, a new 121 KB slabs
   table, a tombstone set) used to stall the query that noticed it: `await refresh()` + query took 117 ms (TS) /
   137 ms (Python) median at `rtt25`. Now that query answers from the current revision in 19 ms / 12 ms while the
   switch completes 126 / 118 ms later in the background, and the next query (on the new revision) takes 12 / 16 ms
   (`node bench/cold/refresh_bench.mjs --sdk ts|py`, n = 10).

**Packs.** With a `PackedBlobStore`, every read (whole-object warms included) is a range of the pack: a cache in
front sees the objects' own keys, and `whole` / `range+warm` fetch the object's range of the pack, never the pack
(tested in both SDKs). Coalescing phase-1 prefix reads of different slabs would need them adjacent in the pack: in
key order only 3.1% of probed cells have a probed neighbour (0.53 adjacent pairs per query over queries 0..29),
and the prefixes of neighbouring cells are ~1 MB apart (the first slab's rerank tier lies between them), so
merging them would read 1 MB to save one request. Not worth it with the current entry order.

**Limiter.** The TS `Limiter` is exact now (a released slot is handed to the next waiter): the hammering test
reproduced the old oversubscription (7 in flight under a limit of 3); Python's limiter took its slot under a lock
and was already exact. With the fix, the pack and per-object paths issue the same requests at the same
concurrency.

**What did not change.** The open is still dominated by the routing table (25 MB, now ~800 ms at `rtt25` instead of
~1.1 s) and its serial chain (current -> manifest -> slabs); `warmup` connections start with the open, so
`open.current` pays ~20 ms more while 32 handshakes run beside it (starting them after `current.json` would avoid
that; not done). A per-object whole GET learns the object's size from its first 8 MiB range and fetches the rest
after it (two rounds); a pack range or a size hint from the manifest (`fine.f32` is `k1 * k2 * dim * 4` bytes) would
allow one round, as the pack's 422 ms `open.base` shows (not done). `range+warm` makes the query right after a cold query slower than Python's old whole-slab default
(137 vs 13 ms) for the ~1-2 s the background warms take; `missPolicy: "whole"` brings it back where bandwidth is
plentiful.

### The option matrix

`bench/cold/run_store.sh` (n = 10 per scenario, one run per config). `slab-cold req` and `MB SDK` are the
query's own reads (speculative reads that land during phase 1 count there); `MB proxy` includes background warms.
`r2s3` has no proxy counters.

#### rtt25, TypeScript (Node)

| config | process-cold e2e | process-cold query | slab-cold query med / p90 | slab-cold phase1 / phase2 ms | slab-cold req phase1 / phase2 | slab-cold MB SDK / proxy | warm query | recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (docs) | 1925 | 576 | 564 / 592 | 135 / 415 | 34 / 143 | 6.2 / 6.2 | 536.4 | 91.5% |
| legacy | 1895 | 528 | 507 / 540 | 120 / 389 | 34 / 142 | 6.2 / 6.2 | 509.4 | 91.5% |
| default | 1209 | 148 | 141 / 164 | 99 / 38 | 60 / 35 | 9.1 / 11.2 | 141.9 | 91.5% |
| default-drain | 1164 | 136 | 147 / 161 | 95 / 42 | 60 / 41 | 9.2 / 11.6 | 14.3 | 91.5% |
| whole-i32 | 1281 | 241 | 222 / 266 | 218 / 1 | 34 / 0 | 47.8 / 47.3 | 16.0 | 91.5% |
| range-i32 | 1188 | 149 | 142 / 165 | 95 / 43 | 56 / 38 | 9.2 / 9.2 | 145.5 | 91.5% |
| inflight12 | 1287 | 265 | 289 / 326 | 132 / 154 | 34 / 52 | 8.9 / 13.5 | 274.3 | 91.5% |
| coalesce4k | 1275 | 235 | 261 / 311 | 100 / 156 | 63 / 148 | 6.2 / 10.8 | 238.0 | 91.5% |
| coalesce32k | 1235 | 187 | 196 / 212 | 95 / 84 | 63 / 71 | 6.8 / 10.0 | 180.3 | 91.5% |
| coalesce512k | 1186 | 143 | 139 / 168 | 96 / 38 | 45 / 18 | 12.4 / 14.3 | 149.5 | 91.5% |
| span1m | 1184 | 138 | 142 / 172 | 100 / 42 | 56 / 18 | 11.6 / 13.8 | 158.0 | 91.5% |
| chunk0 | 1478 | 134 | 141 / 159 | 97 / 40 | 62 / 35 | 9.0 / 11.2 | 136.9 | 91.5% |
| warmup0 | 1267 | 192 | 152 / 178 | 104 / 40 | 60 / 35 | 9.3 / 12.7 | 146.3 | 91.5% |
| early-warmup | 1175 | 141 | 141 / 171 | 102 / 37 | 62 / 34 | 9.2 / 11.0 | 133.9 | 91.5% |
| cold-np20-rr100 | 1153 | 123 | 128 / 135 | 52 / 73 | 20 / 44 | 5.7 / 7.6 | 128.1 | 86.5% |
| disk-fill | 1195 | 137 | 138 / 179 | 96 / 38 | 60 / 32 | 9.2 / 11.2 | 30.5 | 91.5% |
| disk-restart | 472 | 79 | 70 / 88 | 50 / 17 | 0 / 0 | 8.9 / 0.0 | 18.8 | 91.5% |

#### rtt25, Python

| config | process-cold e2e | process-cold query | slab-cold query med / p90 | slab-cold phase1 / phase2 ms | slab-cold req phase1 / phase2 | slab-cold MB SDK / proxy | warm query | recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (docs) | 1813 | 476 | 307 / 361 | 301 / 2 | 34 / 0 | 47.8 / 47.3 | 13.0 | 91.5% |
| legacy | 1699 | 368 | 288 / 332 | 282 / 2 | 34 / 0 | 47.8 / 47.3 | 11.6 | 91.5% |
| default | 1172 | 146 | 146 / 161 | 113 / 30 | 83 / 4 | 8.9 / 11.4 | 137.1 | 91.5% |
| default-drain | 1170 | 130 | 152 / 158 | 115 / 31 | 83 / 3 | 9.0 / 10.9 | 9.5 | 91.5% |
| whole-i32 | 1220 | 170 | 171 / 201 | 166 / 1 | 34 / 0 | 47.8 / 47.3 | 11.9 | 91.5% |
| range-i32 | 1178 | 136 | 146 / 166 | 113 / 29 | 84 / 3 | 9.4 / 9.4 | 146.0 | 91.5% |
| inflight12 | 1311 | 271 | 286 / 316 | 251 / 31 | 84 / 4 | 8.9 / 13.7 | 294.2 | 91.5% |
| coalesce4k | 1256 | 223 | 224 / 248 | 183 / 30 | 176 / 4 | 6.2 / 9.6 | 226.4 | 91.5% |
| coalesce32k | 1200 | 167 | 165 / 199 | 138 / 28 | 124 / 2 | 6.8 / 9.9 | 177.4 | 91.5% |
| coalesce512k | 1192 | 145 | 141 / 150 | 99 / 39 | 59 / 3 | 12.2 / 14.4 | 134.4 | 91.5% |
| span1m | 1196 | 149 | 168 / 193 | 120 / 34 | 68 / 4 | 11.4 / 14.3 | 174.2 | 91.5% |
| chunk0 | 1458 | 149 | 145 / 169 | 116 / 32 | 83 / 6 | 8.9 / 10.8 | 145.9 | 91.5% |
| warmup0 | 1226 | 183 | 135 / 145 | 103 / 30 | 85 / 7 | 9.5 / 12.5 | 130.4 | 91.5% |
| cold-np20-rr100 | 1173 | 126 | 131 / 138 | 95 / 32 | 54 / 8 | 5.5 / 7.9 | 120.8 | 86.5% |
| disk-fill | 1246 | 158 | 148 / 178 | 120 / 29 | 83 / 3 | 8.9 / 11.1 | 23.8 | 91.5% |
| disk-restart | 396 | 36 | 39 / 52 | 29 / 6 | 0 / 0 | 9.0 / 0.0 | 20.0 | 91.5% |

#### rtt25link500, TypeScript (Node)

| config | process-cold e2e | process-cold query | slab-cold query med / p90 | slab-cold phase1 / phase2 ms | slab-cold req phase1 / phase2 | slab-cold MB SDK / proxy | warm query | recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (docs) | 1955 | 603 | 564 / 601 | 157 / 404 | 34 / 145 | 6.2 / 6.2 | 547.9 | 91.5% |
| legacy | 1923 | 560 | 555 / 575 | 155 / 385 | 34 / 145 | 6.2 / 6.2 | 536.4 | 91.5% |
| default | 1273 | 228 | 226 / 267 | 141 / 83 | 34 / 51 | 8.9 / 10.5 | 220.7 | 91.5% |
| default-drain | 1262 | 233 | 227 / 261 | 143 / 80 | 34 / 52 | 8.9 / 10.6 | 14.4 | 91.5% |
| whole-i32 | 1930 | 898 | 855 / 1086 | 852 / 1 | 34 / 0 | 47.8 / 47.3 | 12.6 | 91.5% |
| range-i32 | 1252 | 218 | 218 / 260 | 131 / 83 | 34 / 51 | 8.9 / 8.9 | 216.7 | 91.5% |
| coalesce4k | 1371 | 336 | 310 / 375 | 139 / 170 | 34 / 145 | 6.2 / 9.7 | 332.9 | 91.5% |
| coalesce512k | 1288 | 266 | 277 / 319 | 139 / 127 | 34 / 28 | 12.2 / 13.5 | 272.5 | 91.5% |
| chunk0 | 1561 | 230 | 228 / 263 | 141 / 84 | 34 / 51 | 8.9 / 10.7 | 225.4 | 91.5% |
| cold-np20-rr100 | 1207 | 167 | 176 / 197 | 99 / 71 | 20 / 39 | 5.4 / 7.5 | 164.1 | 86.5% |

#### rtt25link500, Python

| config | process-cold e2e | process-cold query | slab-cold query med / p90 | slab-cold phase1 / phase2 ms | slab-cold req phase1 / phase2 | slab-cold MB SDK / proxy | warm query | recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (docs) | 2215 | 891 | 829 / 1039 | 823 / 2 | 34 / 0 | 47.8 / 47.3 | 11.1 | 91.5% |
| legacy | 2238 | 917 | 831 / 1085 | 826 / 2 | 34 / 0 | 47.8 / 47.3 | 12.4 | 91.5% |
| default | 1260 | 229 | 234 / 277 | 200 / 31 | 82 / 4 | 9.0 / 11.4 | 232.8 | 91.5% |
| default-drain | 1286 | 244 | 236 / 286 | 202 / 31 | 82 / 5 | 8.9 / 11.4 | 12.1 | 91.5% |
| whole-i32 | 1933 | 889 | 854 / 1072 | 848 / 2 | 34 / 0 | 47.8 / 47.3 | 11.3 | 91.5% |
| range-i32 | 1255 | 212 | 223 / 267 | 189 / 32 | 82 / 8 | 9.0 / 9.0 | 223.6 | 91.5% |
| coalesce4k | 1357 | 307 | 320 / 339 | 272 / 33 | 176 / 9 | 6.2 / 9.5 | 305.3 | 91.5% |
| coalesce512k | 1303 | 288 | 293 / 349 | 258 / 29 | 58 / 3 | 12.2 / 15.1 | 304.1 | 91.5% |
| chunk0 | 1540 | 234 | 239 / 282 | 203 / 31 | 83 / 4 | 9.0 / 11.1 | 237.3 | 91.5% |
| cold-np20-rr100 | 1216 | 185 | 192 / 207 | 148 / 37 | 55 / 8 | 6.1 / 8.4 | 180.1 | 86.5% |

#### rtt60, TypeScript (Node)

| config | process-cold e2e | process-cold query | slab-cold query med / p90 | slab-cold phase1 / phase2 ms | slab-cold req phase1 / phase2 | slab-cold MB SDK / proxy | warm query | recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (docs) | 3786 | 1141 | 1091 / 1175 | 261 / 835 | 34 / 144 | 6.2 / 6.2 | 1121.5 | 91.5% |
| legacy | 3773 | 1115 | 1089 / 1146 | 259 / 829 | 34 / 145 | 6.2 / 6.2 | 1068.8 | 91.5% |
| default | 2229 | 287 | 273 / 295 | 200 / 66 | 63 / 22 | 9.0 / 10.9 | 259.3 | 91.5% |
| whole-i32 | 2312 | 354 | 351 / 410 | 347 / 1 | 34 / 0 | 47.8 / 47.3 | 12.6 | 91.5% |
| coalesce4k | 2443 | 503 | 472 / 605 | 204 / 267 | 63 / 116 | 6.2 / 10.1 | 500.3 | 91.5% |
| chunk0 | 2821 | 262 | 278 / 327 | 205 / 71 | 63 / 25 | 8.9 / 11.1 | 260.3 | 91.5% |
| cold-np20-rr100 | 2208 | 241 | 244 / 287 | 94 / 146 | 20 / 46 | 5.7 / 7.4 | 243.2 | 86.5% |

#### rtt60, Python

| config | process-cold e2e | process-cold query | slab-cold query med / p90 | slab-cold phase1 / phase2 ms | slab-cold req phase1 / phase2 | slab-cold MB SDK / proxy | warm query | recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (docs) | 3605 | 1010 | 612 / 782 | 610 / 0 | 34 / 0 | 47.8 / 47.3 | 5.2 | 91.5% |
| legacy | 3361 | 734 | 582 / 775 | 575 / 2 | 34 / 0 | 47.8 / 47.3 | 12.4 | 91.5% |
| default | 2307 | 302 | 305 / 324 | 233 / 65 | 82 / 3 | 8.9 / 11.5 | 301.1 | 91.5% |
| whole-i32 | 2359 | 343 | 345 / 436 | 338 / 2 | 34 / 0 | 47.8 / 47.3 | 12.8 | 91.5% |
| coalesce4k | 2478 | 483 | 486 / 547 | 414 / 65 | 174 / 5 | 6.2 / 9.4 | 482.1 | 91.5% |
| chunk0 | 2864 | 295 | 302 / 340 | 234 / 65 | 83 / 3 | 8.9 / 11.1 | 302.6 | 91.5% |
| cold-np20-rr100 | 2255 | 260 | 241 / 297 | 163 / 71 | 50 / 10 | 5.8 / 7.7 | 243.5 | 86.5% |

#### r2s3, TypeScript (Node)

| config | process-cold e2e | process-cold query | slab-cold query med / p90 | slab-cold phase1 / phase2 ms | slab-cold req phase1 / phase2 | slab-cold MB SDK / proxy | warm query | recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (docs) | 2319 | 1305 | 1293 / 1358 | 302 / 960 | 34 / 144 | 6.2 / - | 1175.5 | 91.5% |
| legacy | 2142 | 1276 | 1198 / 1431 | 297 / 869 | 34 / 142 | 6.2 / - | 1177.4 | 91.5% |
| default | 1445 | 503 | 440 / 704 | 221 / 220 | 41 / 48 | 9.1 / - | 431.0 | 91.5% |
| whole-i32 | 1747 | 719 | 686 / 823 | 681 / 1 | 34 / 0 | 47.8 / - | 14.8 | 91.5% |
| coalesce4k | 1603 | 560 | 562 / 709 | 170 / 387 | 34 / 148 | 6.2 / - | 641.7 | 91.5% |
| chunk0 | 1265 | 376 | 362 / 433 | 174 / 181 | 34 / 50 | 9.0 / - | 400.6 | 91.5% |
| cold-np20-rr100 | 1335 | 312 | 376 / 469 | 159 / 174 | 22 / 34 | 5.7 / - | 359.4 | 86.5% |
| disk-fill | 1359 | 420 | 384 / 550 | 240 / 164 | 56 / 24 | 8.9 / - | 21.9 | 91.5% |
| disk-restart | 440 | 50 | 34 / 73 | 23 / 4 | 0 / 0 | 8.9 / - | 13.5 | 91.5% |

#### r2s3, Python

| config | process-cold e2e | process-cold query | slab-cold query med / p90 | slab-cold phase1 / phase2 ms | slab-cold req phase1 / phase2 | slab-cold MB SDK / proxy | warm query | recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (docs) | 1609 | 678 | 599 / 705 | 592 / 2 | 34 / 0 | 47.8 / - | 12.3 | 91.5% |
| legacy | 1622 | 687 | 610 / 893 | 604 / 2 | 34 / 0 | 47.8 / - | 8.7 | 91.5% |
| default | 1385 | 474 | 465 / 736 | 349 / 112 | 82 / 8 | 9.1 / - | 445.6 | 91.5% |
| whole-i32 | 1867 | 678 | 658 / 769 | 651 / 2 | 34 / 0 | 47.8 / - | 11.6 | 91.5% |
| coalesce4k | 1721 | 626 | 660 / 968 | 534 / 150 | 178 / 38 | 6.3 / - | 668.5 | 91.5% |
| chunk0 | 1647 | 556 | 531 / 600 | 360 / 140 | 85 / 8 | 9.3 / - | 552.5 | 91.5% |
| cold-np20-rr100 | 1864 | 388 | 407 / 592 | 297 / 117 | 52 / 12 | 6.3 / - | 435.8 | 86.5% |
| disk-fill | 1820 | 553 | 504 / 653 | 366 / 123 | 86 / 6 | 9.2 / - | 16.9 | 91.5% |
| disk-restart | 305 | 27 | 25 / 37 | 19 / 3 | 0 / 0 | 8.9 / - | 15.4 | 91.5% |

## Surprises and caveats

- **`Limiter` oversubscribes (TS SDK bug).** `Limiter.run` increments `active` only after a woken waiter resumes,
  so when a slot is released and a new caller arrives in the same tick, both get in. With a slot taken and released
  without I/O, which `fetchRecords` does on a `PackedBlobStore` over S3 (`getRanges` answers null at once, then the
  single-range reads follow), a standalone reproduction reaches 24 reads in flight under a limit of 12; through the
  proxy the pack path averaged 13.5 in flight vs 10.9 for per-object keys. Fixed in "Store-level
  optimizations" above (a released slot is handed to the next waiter).
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
