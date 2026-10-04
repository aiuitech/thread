# Query path: slab formats, the two-phase query, caching and routing

How a dense query turns into bytes read and scores returned. The Worker (`examples/cloudflare-worker/src/index.ts`)
and the SDKs (`sdk/ts`, `sdk/python`) implement the same path; the endpoint names below are the Worker's, the
SDK equivalents are in their READMEs. Build-time knobs: [profiles.md](profiles.md). Deltas and tombstones on the
read path: [update-path.md](update-path.md#read-path).

## Slab formats

A v2 or v3 slab stores every entry twice. The **scan tier**, bytes `[0, scan_len)`, holds the header, the cell
mean, the entry table (16 B per entry) and the scan codes:

- **v3 (format 3, `tiers: "pq192+sq8"`, g4+):** one byte per PQ subspace, `pq_m` = 192 subspaces of 8 dims with
  256 sub-centroids each, one codebook for the whole index (`idx/{gen}/pq.bin`, 1.5 MB). 208 B/entry at dim 1536.
- **v2 (format 2, `tiers: "sq4+sq8"`, g2/g3):** 4-bit residual codes with a per-vector MSE-clipped scale.
  ~790 B/entry.

The **rerank tier**, at `rerank_off`, is the same in both: one `4 + dim` byte record per entry
(`[scale8 f32][code8 i8 * dim]`), so record `i` is the range `[rerank_off + i*(4+dim), +4+dim)`. `publish` writes
`idx/{gen}/slabs.json` (array indexed by cell: `[scanLen, rerankOff, count, bytes]`; compactions add a fifth
column, the slab revision) and records `format` (and for format 3 `pq: {m}`) in the manifest. v1 generations (int8
only, e.g. `g1`) are still served: the Worker reads their slabs whole and skips phase 2. One Worker serves every
format; the format is read from the manifest and from each slab header.

Offline (`data/quant-experiment.md`), a 4-bit scan + int8 rerank of the top 32 equals int8 recall at half the scan
bytes; the PQ-192 scan tier needs a rerank of the top 100-200 (93.3% / 94.0% vs 94.2% for int8 at nprobe 22) but
scans ~3.8x fewer bytes than 4-bit.

## The two-phase query

A query (`POST /query?nprobe=N&rerank=R&pc=P&route=flat|beam&beam=B`) runs in two phases. Every parameter is
optional: an absent one takes the revision's `manifest.defaults` (see [profiles.md](profiles.md)); manifests
without `defaults` (g1, g2) fall back to `nprobe=1, rerank=32, pc=0`, and an absent `route` means flat. The
response's `stats.params` shows the values used and where each came from (`query`, `manifest`, `tree` or `legacy`).

1. **Scan.** Route (flat by default, `route=beam` through the [centroid tree](#centroid-tree-and-beam-routing)),
   then read only the scan prefix of each probed slab: a `Range: bytes=0-(scanLen-1)` match against the Cache API
   copy (it answers 206 from a cached full object). On a cache miss the Worker GETs the whole object from R2,
   caches it in the background, and keeps the rerank records of that slab's local winners in memory. Each prefix
   is scanned for the top `T = max(k, rerank)`: format 3 with `thread.scan_pq(prefix, table, q, T)`, where
   `table = PqTable.from_codebook(q, codebook)` is built once per query (192 x 256 dot products) and freed when
   the query ends; format 2 with `thread.scan`. Reads are issued as a sliding window of 12 in flight per query
   (`QUERY_INFLIGHT`). Cache API reads are not held to the Workers limit of 6 open connections (measured in
   `data/bench-latency.txt`); R2 reads are retried once if the runtime cancels them for the connection limit.
2. **Rerank.** The global top-T (after `merge`) is grouped by slab, and the winners of each slab are read with ONE
   multi-range Cache API request (`Range: bytes=a-b, c-d, ...`, one range per record, adjacent records within 4 KB
   merged; the 206 multipart/byteranges answer is parsed in `cacheMatchMulti`), then re-scored with
   `thread.rerank`. The top k by reranked score are returned. **Speculative reads:** once only 6 slabs remain to be
   scanned, the running top-T is read the same way at low priority (only when no prefix read is waiting), so
   phase 2 usually finds every record already in memory and costs about one request per winner slab at most.
   `rerank=0` disables phase 2 (scan-tier scores only). Format 1 generations (int8-only slabs) always skip phase 2.

With multi-range and speculative reads, phase 2 costs one request per winner slab (~16 at T=100, ~20 at T=200 on
g2), not one per record. Stats in the response include `scanBytes`, `rerankReads`, `rerankBytes`, `specUsed`,
`tiers`, `pqM` and the timings `phase1`/`phase2`. `GET /debug/range?key=idx/...&offset=&length=` shows how the
Cache API answers a Range request for a cached object (status, Content-Range, and whether the bytes match R2).

## Revisions, bases and caches

A query reads the (gen, rev) that `idx/current.json` names (re-read at most every 30 s per isolate; `?gen=&rev=`
overrides). Parsed revisions (manifest, slabs table, tombstone sets) are cached per isolate and never revalidated,
since revisions are immutable. The base (Searcher over the centroid tables, PQ codebook) is keyed by its object
keys and shared by every revision of the generation that uses the same files; one base is cached per isolate
(`BASE_CACHE_SIZE`). Queries hold it while they run. When a query for another generation replaces it, the old one
is freed once its last query finishes, so alternating generations does not leak wasm memory. Wasm objects are
freed only in the instance (epoch) that created them. The centroid tree is not part of the base key: a revision
that adds or changes a tree reuses the loaded means, and the tree it names is installed (`Searcher.set_tree`)
right before the query routes.

A cold open (`thread.open` / `open_index`) does not keep a base around: see
[Cold opens](#cold-opens-compact-routing-cold-tree-routing-base-packs) for the compact routing tables, cold tree
routing and base packs. None of them changes the base key: it stays `gen|coarse|fine|pq`, where `fine` is the f32
means the loaded table was made from (`files.fine`), so a revision that adds f16 / i8 tables or a base pack reuses the
loaded base, and the table variant (`base.routing`) is a loading detail. A cold base (tree + codebook, cold routing)
is keyed by the base key plus the grouped table's key and is dropped once the full base of the same key is in.
The open reads through a prefetching view (`PrefetchStore`): with a base pack its objects are served from the pack,
without one every base object is requested as soon as the manifest is in; the view is dropped after the open.

## Recovering from a wasm trap

The wasm instance installs a panic hook after init. If a request hits a wasm trap (`WebAssembly.RuntimeError`),
the Worker calls `__wbg_reset_state()`, re-inits, drops the cached index (its `Searcher` and `PqCodebook`
pointers are invalid) and retries the request once. `--experimental-reset-state-function` makes wasm-bindgen emit
`__wbg_reset_state()` (see [operations.md](operations.md#build-the-wasm-and-deploy-the-worker)).

## Centroid tree and beam routing

Flat routing scores the query against all 4096 cell means (`cell_dots`, 4096 x 1536 multiply-adds), and an upsert
used to do the same per vector to find its primary cell. A **centroid tree** prunes that search:
`thread.build_centroid_tree(fineMeans, dim, groups, iters, seed)` clusters the cell means into `groups` group
means (spherical k-means over the means), and a lookup scores the group means first, then only the cells of the
best `beam` groups (~128 + beam x 32 dot products at 128 groups).

- **Object.** `publish` and `publish-rev` take `tree: {groups: 128, iters: 10, seed, beam: 8}` and write
  `idx/{gen}/tree.{rev}.bin` (~0.8 MB at 4096 x 1536, 128 groups), referenced from the manifest as
  `tree: {key, groups, beam, nearestRecall, top3Recall, recallByBeam?}`. The recall figures come from
  `Searcher.tree_recall` / `tree_recall_top(sample, beam, 3)` on ~2000 base vectors (250 rows from each of 8
  k-means sample objects). Later revisions inherit the tree; `publish-rev {tree: null}` removes it. The driver
  builds it with `--tree groups,beam` (edge profile: 128,8); for an existing generation:
  `node bench/driver.mjs --gen g4 --publish-rev --tree 128,8`.
- **Upserts** call `DeltaJob.set_tree(tree, tree.beam)` before encoding, so the primary cell comes from the pruned
  search. On g4 (beam 8 finds the exact nearest cell for 99.35% of vectors) a 2000-vector upsert went from 7.5 s
  to 3.3 s, 500 from 2.5 s to 1.4 s in a same-session A/B; self-retrieval stayed 100% before and after folding the
  vectors into slabs (`data/bench-tree.txt`).
- **Queries** take `route=beam` (and `beam=N`, default `defaults.beam`, else `tree.beam`): `route_beam(q, beam,
  nprobe)` replaces `cell_dots` + `route_from_dots`. Delta segments then need `q·mean` only for the cells they
  hold entries in: `cell_dots_for(q, cells)` over the union of each segment's `delta_cells` (cached per segment),
  scattered into the `qc` table `scan_delta` reads (scores identical to the flat path).
- **Measured on g4** (nprobe 34, rerank 150, interleaved, `data/bench-tree.txt`):

| c | routing | recall@10 | p50 ms | p95 ms | RPS (median of 3 separate runs) |
|---:|---|---:|---:|---:|---:|
| 50 | flat | 95.12% | 70.5 | 115.7 | 655 |
| 50 | beam 8 | 94.49% | 68.7 | 119.3 | 591 |
| 50 | beam 16 | 95.00% | 69.6 | 115.2 | 652 |
| 300 | flat | 95.12% | 83.7 | 175.9 | 2529 |
| 300 | beam 8 | 94.35% | 81.9 | 171.1 | 2599 |
| 300 | beam 16 | 94.93% | 82.4 | 170.3 | 2630 |
| 300 | beam 32 | 95.11% | 83.3 | 174.3 | 2617 |

Beam 8 costs 0.6-0.8 points of recall; beam 16/32 keep recall within 0.2 points but no beam improves p50 or RPS
measurably (routing is ~1-2 ms of CPU in a 26-34 ms server time spent mostly on Cache API reads, and the 34 probes
stay the same). So **flat stays the default** (g4 rev 103: `defaults.route: "flat"`, `defaults.beam: 32` for
queries that ask for `route=beam`); the tree pays on the write path.

## Cold opens: compact routing, cold tree routing, base packs

A fresh process or isolate must get the routing table before its first query can pick cells: `fine.f32` is
4103 x 1536 f32 = 25.2 MB on g4, and with `pq.bin` (1.5 MB) it was ~60% of a cold first query's time against S3
([cold-start.md](cold-start.md)). The global table only routes (picks the `nprobe` nearest cells) and gives delta
segments their base score `q·mean`; every slab carries its own f32 mean, so slab scores never depend on it. Three
independent levers cut what a cold open must read, all opt-in on the reader and written by publish:

1. **Compact routing tables** (`open({routing})`, `open_index(routing=)`): `fine.f16` (IEEE half, 12.6 MB) or
   `fine.i8` (6.3 MB: the f32 centre of all means, then per row `[scale f32][i8 * dim]` residuals from it), decoded
   to f32 once on load (`Searcher.from_table`), so the hot path is unchanged. `"auto"` = the smallest the revision
   has. Centring matters: cell means share a large common component (|centre| = 0.83 of a mean norm of 0.90 on g4,
   one dimension at 0.66), and a per-row scale over the raw means spends its 8 bits on it (96.9% probe overlap with
   f32 routing instead of 99.8%).
2. **Cold tree routing** (`coldRouting: "tree"`, `coldBeam`): the open loads the centroid tree (0.8 MB) and the
   codebook only. A query scores the tree's 128 group means, range-reads the rows of its best `beam` groups from the
   tree-grouped table `fine.grouped.{rev}.i8` (rows in tree member order, so group g is `[dataOff + offsets[g] *
   rowBytes, +n * rowBytes)`; ~32 cells x 1540 B per group; the first read also brings the table's 23 KB head with
   the centre) and ranks those groups' cells exactly as `route_beam` does over the full table (`ColdRouter` /
   `PartialRouter`). Loaded groups stay loaded; delta segments get exact `q·mean` for their cells (their rows read
   by row). The full table (`routing`) loads in the background and takes over (`routingState`, `whenFullRouting`).
3. **Base packs** (`basePack`, default on): `idx/{gen}/base.{rev}.threadpack` is a pack of everything a cold open
   reads (manifest, slabs table, tombstone sets, coarse, codebook, tree, `fine.i8`; 8.8 MB on g4), found by its
   conventional key from `current.json`'s (gen, rev): the open is `current.json` + one GET (the manifest is read
   alongside in case there is no pack). `base.cold.{rev}.threadpack` (2.5 MB) leaves the routing table out, for cold
   routing. An open with `routing` f32 / f16 does not fetch a pack (it would download `fine.i8` for nothing); every
   open without a pack requests all base objects as soon as the manifest is in (3 round trips instead of 4). An index
   pack (`packIndex`) now stores these base objects first and records their range, so an open on a `PackedBlobStore`
   reads it with one range read, after a single suffix-range read for the footer and directory (`getTail`).

### Probe agreement (2000 queries, g4 rev 180, nprobe 34, no slab reads)

`bench/cold/routing_agreement.mjs`: each mode's 34 probed cells against flat routing over `fine.f32`. Overlap =
mean |cells ∩ f32 cells| / 34; identical = queries with the same set; first cell = the top cell agrees; `q·mean`
error = what the table changes in a delta entry's base score (over the f32 probes).

| mode | table / bytes read before routing | probe overlap | identical | first cell | `q·mean` error mean / max |
|---|---|---:|---:|---:|---|
| flat f32 | `fine.f32` 25.2 MB | 100% | 100% | 100% | 0 |
| flat f16 | `fine.f16` 12.6 MB | 99.98% | 99.2% | 100% | 7.8e-5 / 1.8e-4 |
| flat i8 | `fine.i8` 6.3 MB | 99.82% | 93.8% | 99.9% | 6.0e-5 / 6.0e-4 |
| tree beam 8, f32 means | tree 0.8 MB + 2.0 MB of f32 rows | 90.60% | 22.5% | 99.3% | exact for loaded cells |
| tree beam 16, f32 means | + 3.9 MB | 95.99% | 44.5% | 99.9% | |
| tree beam 32, f32 means | + 7.4 MB | 98.81% | 74.8% | 100% | |
| cold beam 8, i8 rows | tree 0.8 MB + 0.50 MB | 90.57% | 21.3% | 99.2% | as flat i8 |
| cold beam 16, i8 rows | tree 0.8 MB + 0.97 MB | 95.90% | 41.7% | 99.8% | |
| cold beam 32, i8 rows | tree 0.8 MB + 1.86 MB | 98.67% | 70.3% | 99.9% | |

(f16 rows route like f32 rows within 0.02 points at every beam, at twice the bytes of i8 rows.) The beam, not the
row precision, decides cold routing's agreement: beam 8 misses ~3 of 34 cells, beam 32 fewer than 1.

### Bytes and time before the first query (g4 rev 180, fake S3 proxy)

`bench/cold/run_base.sh` (TypeScript SDK, Node, no cache; `cold_bench.mjs` process-cold scenario: spawn, wasm, open,
one query; n = 10, queries 0..19, medians). The objects of `make_routing.mjs` / `make_base_pack.mjs` are served from
a local overlay over the proxy's R2 cache; nothing was written to R2. Modes: `-seq` = `basePack: false` (the open's
reads one after another, as before this change); without the suffix the default `basePack: true` (a base pack for i8
and cold opens, the files requested together for f32 / f16).

**rtt 25 ms, 200 Mbps per response** (baseline f32 process e2e: 1925 ms, [cold-start.md](cold-start.md))

| mode | requests / MB before the first slab read | open ms | first query ms | process e2e ms | recall@10 (10 queries) |
|---|---|---:|---:|---:|---:|
| f32 (`-seq`, as before) | 6 / 25.66 | 1278 | 564 | 1900 | 88% |
| f32 | 6 / 25.66 | 1251 | 558 | 1881 | 88% |
| f16 | 6 / 13.64 | 744 | 548 | 1363 | 88% |
| i8 `-seq` | 6 / 7.65 | 514 | 581 | 1176 | 88% |
| i8 (base pack) | 2 / 8.42 | 556 | 579 | 1200 | 88% |
| cold beam 8 (cold pack) | 11 / 2.92 | 276 | 691 | 1030 | 84% |
| cold beam 16 | 17 / 3.42 | 275 | 697 | 1033 | 87% |
| cold beam 32 | 28 / 4.27 | 278 | 672 | 1021 | 88% |
| cold beam 16 `-seq` | 20 / 3.41 | 300 | 677 | 1037 | 87% |

Process e2e, the other profiles (f32-seq / f16 / i8-seq / i8 / cold 8 / 16 / 32, ms): **rtt 60 ms, 100 Mbps**:
3802 / 2714 / 2281 / 2373 / 2040 / 2043 / 2071; **rtt 25 ms with all responses sharing a 500 Mbps link**:
1949 / 1413 / 1178 / 1233 / 1077 / 1070 / 1093. **rtt 90 ms, 400 Mbps** (closer to R2 from this host: serial
requests 75-105 ms, ~380 Mbps per stream), f32-seq / f32 / i8-seq / i8 / cold16 / cold16-seq: open 1313 / 1222 /
928 / 783 / 626 / 799 ms, process e2e 3051 / 2919 / 2680 / 2538 / 2538 / 2753 ms.

Python SDK (`cold_bench.py`, 256 MB `MemoryCache` default), rtt 25 ms / 200 Mbps: f32-seq open 1247 ms, e2e 1802;
i8-seq 485 / 1035; i8 (base pack) 479 / 1025; cold16 (cold pack) 213 / 810; cold16-seq 282 / 856.

Reading the numbers:

- The routing table is the cold open: f32 -> i8 cuts 18 MB and ~0.75 s at 200 Mbps (1.5 s at 100 Mbps) off every
  cold start, for 0.18 points of probe overlap and no measurable recall change (below).
- Cold tree routing makes the open itself 2.5-3.5 MB and ~0.28 s, but the first query pays one round trip more for
  its groups (route ~100 ms at rtt 25, ~300 ms at rtt 90) and, at beam 8 / 16, routes slightly worse; the second
  query already routes with the full table (it was in after ~0.5 s). Best when bandwidth is the limit (rtt 25: e2e
  1030 vs 1200 ms for i8), no better than i8 + base pack when latency is (rtt 90: 2538 ms both).
- Base packs save round trips, not bytes: on the per-response bandwidth model at 25-60 ms RTT one 8.4 MB response is
  ~40 ms slower than six parallel ones, at 90 ms RTT the pack is 145 ms faster (783 vs 928 ms open), and for cold
  routing it saves 25-170 ms. Against R2 (75-105 ms per serial request) every saved round trip is ~0.1 s.
- f32 opens do not fetch a base pack: they would download `fine.i8` for nothing (1.61 s instead of 1.28 s open at
  rtt 25 when they did); they get the "files" prefetch instead (all base objects once the manifest is in, -30 ms).
