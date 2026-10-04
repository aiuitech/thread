# Operations: building, deploying, benchmarking

The commands behind the `vib-bench` deployment and the dbpedia-openai-1M benchmark. What each stage does:
[build-pipeline.md](build-pipeline.md); every driver flag: [build-flags.md](build-flags.md); the scripts
themselves: [bench/README.md](../bench/README.md).

## Prepare the dataset

```sh
# prep dataset (dbpedia-openai-1M-1536-angular) and upload shards
.venv/bin/python bench/prep/prep_dataset.py
bench/prep/upload_shards.sh 6
```

## Build the wasm and deploy the Worker

```sh
cargo build --release --target wasm32-unknown-unknown -p thread-wasm
wasm-bindgen --target web --experimental-reset-state-function --out-dir crates/thread-wasm/pkg --out-name thread target/wasm32-unknown-unknown/release/thread_wasm.wasm
(cd examples/cloudflare-worker && cf deploy --secrets-file .env.secrets)   # run from the example dir, not the repo root
```

That is the only deployment: build stages and writes run in build slots (dynamic Workers of `vib-bench`, one
isolate per slot id, see [build-pipeline.md](build-pipeline.md#build-slots-worker-loader) and
[examples/cloudflare-worker/README.md](../examples/cloudflare-worker/README.md)). The stage bundle the slots run is
built by the same `cf deploy` (vite plugin in `examples/cloudflare-worker/vite.config.ts`). `THREAD_ORCH` /
`THREAD_SLOTS` point the drivers at another deployment / change the build width (default 8).

`--experimental-reset-state-function` makes wasm-bindgen emit `__wbg_reset_state()`, which the Worker uses to
recover from a trap. After rebuilding the wasm, `sdk/ts/scripts/sync-wasm.sh` copies it into the SDK
(`cd sdk/ts && npm test`).

`examples/cloudflare-worker/.env.secrets` holds `BUILD_TOKEN=<random>`; every endpoint except `/health` requires
`Authorization: Bearer <token>`.

## Build a generation and benchmark it

```sh
# build an index generation (edge profile), then benchmark queries
node bench/driver.mjs --gen g4 --profile edge
node bench/query_bench.mjs --gen g4 --n 1000 --concurrency 50          # manifest defaults
node bench/query_bench.mjs --gen g4 --n 1000 --concurrency 50 --nprobe 22 --rerank 100
# filtered queries (g4's words are all 0, so only {"attrs":0} or {"attrs":{"$none":...}} match; a schema'd index takes field conditions)
node bench/query_bench.mjs --gen g4 --n 1000 --concurrency 50 --filter '{"attrs":{"$lt":1}}' --maxNprobe 68

# change a generation's query defaults (new manifest revision; isolates pick it up within 30 s)
curl -X POST -H "authorization: Bearer $TOKEN" $URL/build/publish-rev -d '{"gen":"g4","defaults":{"nprobe":34,"rerank":150}}'
# add a centroid tree to an existing generation (new revision; records its recall), compare routings
node bench/driver.mjs --gen g4 --publish-rev --tree 128,8 --treeRecallBeams 4,16,32
node bench/query_bench.mjs --gen g4 --n 6000 --concurrency 50 --nprobe 34 --rerank 150 --extras "route=flat|route=beam&beam=8|route=beam&beam=16"
```

`query_bench.mjs` sends `nprobe`/`rerank`/`pc` only when the flag is given and prints the values the server used
(`server params: nprobe=10 (manifest) ...`).

## Updates, compaction, splits

Details: [update-path.md](update-path.md).

```sh
node bench/compact.mjs --gen g4 [--purge 1]           # fold deltas/tombstones into new slab revisions (build slots)
node bench/update_bench.mjs --gen g4                  # upsert/delete/compaction benchmark -> data/bench-updates.txt
node bench/update_bench.mjs --gen g4 --only trigger --out data/bench-trigger.txt   # automatic compaction benchmark
node bench/split_bench.mjs --gen g4 --maxCellEntries 2400   # split the largest cells, recall/latency before/after
curl -H "authorization: Bearer $TOKEN" "$URL/index/status?gen=g4"   # deltas, tombstones, thresholds, lock, last runs
```

Set or change the automatic compaction policy of a generation:

```sh
node bench/driver.mjs --gen g4 --publish-rev --compaction profile \
  [--compactMaxDeltas N --compactMaxDeltaMB N --compactMaxTombstones N --compactMaxCellEntries N \
   --compactCellsPerRun N --compactDeadlineMs N --compactSplitCellsPerRun N]      # --compaction 0 removes it
```

## Lexical segments on local files

`thread --help` and `thread <command> --help` describe every command and option:

```sh
cargo build --release -p thread-cli
target/release/thread seg build --out seg.thread --docs docs.jsonl --text        # {"id", "text", "meta"} per line
target/release/thread seg search --text "deploy failed" --k 10 --with-docs seg.thread
```

A segment built locally is byte-for-byte what an upsert writes (see [lexical.md](lexical.md#api-summary)).

## Checks before a commit

```sh
cargo test --workspace
cargo check -p thread-core -p thread-wasm --target wasm32-unknown-unknown
cargo fmt --all --check
cargo clippy --all-targets
cd sdk/ts && npm run typecheck && npm test
cd sdk/python && maturin develop --uv && pytest && mypy thread_sdk
```

The dense-side files predate rustfmt and carry `#![cfg_attr(rustfmt, rustfmt::skip)]`, so the formatter touches
only the lexical modules and new code. The Python tests include a cross-check against the built TypeScript SDK in
`sdk/ts/dist`.
