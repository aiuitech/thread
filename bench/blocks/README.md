# bench/blocks

The block cache of `CachedBlobStore` (0.2.5, [docs/lexical.md](../../docs/lexical.md#block-cache)) against a real
store: a Worker reads an index under `tmp/blockbench/` of the `vib-bench` bucket the way inseam does (one
`thread.open` per request, the Cache API in front, `waitUntil`), cold and then warm. The same Worker code is
deployed once per SDK build, so two builds run side by side in one data center.

```
build_index.mjs   a local index shaped like an inseam host: dense slabs + lexical/docs segments of ~4 MiB
upload.mjs        a local index -> tmp/blockbench/<name>/ through the Worker's PUT /put
run.mjs           cold / warm / warm again / new terms, per Worker and block size; medians per phase
worker/           the bench Worker (POST /search, PUT /put, POST /delete; all under tmp/blockbench/ only)
```

```sh
cd sdk/ts && npm run build && cd ../..
node bench/blocks/build_index.mjs --out data/blockbench/hybrid8 --words 140                      # 8 x 4.1 MB
node bench/blocks/build_index.mjs --out data/blockbench/big2 --docs 8000 --segments 2 --words 1100  # 2 x 46 MB

cd bench/blocks/worker && npm install
echo "BENCH_TOKEN=$(openssl rand -hex 24)" > .env.secrets
BENCH_NAME=thread-blockbench-new SDK_LABEL=new cf deploy --secrets-file .env.secrets
mkdir -p /tmp/v024 && git -C ../../.. archive v0.2.4 sdk/ts/src | tar -x -C /tmp/v024 && cp ../../../sdk/ts/src/wasm/* /tmp/v024/sdk/ts/src/wasm/
THREAD_SDK=/tmp/v024/sdk/ts BENCH_NAME=thread-blockbench-old SDK_LABEL=0.2.4 cf deploy --secrets-file .env.secrets
cd ../../..

export BENCH_TOKEN=... BENCH_URL=https://thread-blockbench-new.<subdomain>.workers.dev
node bench/blocks/upload.mjs data/blockbench/hybrid8 hybrid8
UPLOAD_CONCURRENCY=1 node bench/blocks/upload.mjs data/blockbench/big2 big2
node bench/blocks/run.mjs --url old=https://thread-blockbench-old.<subdomain>.workers.dev \
  --url new=$BENCH_URL --index hybrid8 --blocks 262144,1048576,4194304 --reps 3
node bench/blocks/run.mjs --mode dense --docs none --url old=... --url new=... --blocks 1048576 --reps 10
```

Each repetition uses a fresh `cacheOrigin` (a cold cache), then the same one: `warm` and `warm2` repeat the query
after the background puts (`--pause`, default 4 s), `newq` searches other terms. The Worker counts every call that
reaches the R2 binding (`R2 gets`, `R2 MB`), whatever the SDK's trace says. Segment opens (three rounds per segment
per `thread.open`) happen inside the search but are not in any span's `io`, so the R2 counts include them and the
lexical span does not. Results of 2026-11-09 are in docs/lexical.md "Block cache".

Clean up with `POST /delete {"prefix":"tmp/blockbench/"}` and `cf workers delete` (or the dashboard) for the two
Workers.
