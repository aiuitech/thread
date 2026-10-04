# bench/cold: cold-start measurement against S3

How long the first query takes when nothing is warm (a new process, a new index handle, empty caches), where that
time goes, and how many requests and bytes it costs, for both SDKs, against real R2 S3 or a deterministic fake S3.
Results, method and analysis: [docs/cold-start.md](../../docs/cold-start.md).

| file | what it does |
|---|---|
| `fake_s3.mjs` | read-only caching proxy in front of R2's S3 API with a network model (`--rtt-ms`, `--mbps`, `--jitter-ms`, `--conn-ms`, `--link-mbps`), `--overlay DIR`, JSONL request log, `/__stats` `/__reset` `/__config` |
| `verify_proxy.mjs` | proxy vs R2: status, Content-Range, length and body hash on 35 range/HEAD/404/list cases |
| `warm_proxy.mjs` | fills the proxy cache for a query set (run twice: the second must fetch nothing upstream) |
| `make_pack.mjs` | the pinned revision as a pack in `data/cold/overlay/packs/` (sparse: unwarmed slabs are holes) |
| `cold_bench.mjs` / `cold_bench.py` | the bench, TypeScript / Python SDK: same CLI, same JSONL (`thread-cold/1`), same table |
| `summarize.mjs` | summary of saved runs; with several files, medians side by side (before/after) |
| `slab_bytes.mjs` | per-slab byte accounting: scan prefix vs rerank tier vs whole slab |
| `run_baseline.sh` | the whole baseline: warm-up, pack, 3 network profiles x {proxy, pack} x {ts, py}, then r2s3 (`TAG`, `N`, `PROFILES`, `SDKS`, `WARM`, `R2S3`) |
| `lib.mjs` | dataset, recall, the summary table |

```sh
(cd sdk/ts && npm run build)                         # the benches and the proxy use sdk/ts/dist
bench/cold/run_baseline.sh                           # TAG=baseline N=10 -> data/cold/baseline-*.jsonl (+ .txt)
TAG=after-x R2S3=0 bench/cold/run_baseline.sh        # an optimization, same query set and profiles
node bench/cold/summarize.mjs data/cold/baseline-ts-proxy-rtt25.jsonl data/cold/after-x-ts-proxy-rtt25.jsonl --only-compare

# by hand
node bench/cold/fake_s3.mjs --rtt-ms 25 --mbps 200 --overlay data/cold/overlay --log data/cold/proxy.jsonl &
node bench/cold/cold_bench.mjs --store proxy --n 10 --out data/cold/x-ts.jsonl
sdk/python/.venv/bin/python bench/cold/cold_bench.py --store pack --n 10 --out data/cold/x-py-pack.jsonl
```

Scenarios per trial (a fresh child process each): `process-cold` (import, wasm, open, query `2t`), `slab-cold`
(query `2t + 1` in the same process), `warm` (query `2t + 1` again). Queries never repeat across trials, so a run of
`--n 10` uses queries 0..19; the warm-up covers 0..29. A proxy run that reports `upstream fetches` > 0 was not warmed
up and measured R2 as well; re-run `warm_proxy.mjs`. Never point anything here at the bucket for writing: the proxy
refuses PUT/POST/DELETE and only GETs, HEADs and lists upstream.

To try a new object format without touching R2: write the objects under `data/cold/overlay/` with their bucket keys
(e.g. `data/cold/overlay/idx/g4/manifest.180.json`, plus an `idx/current.json` naming it), start the proxy with
`--overlay data/cold/overlay` and run the benches as usual. Everything under `data/` is gitignored.
