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
   With an attribute filter (`filter=` JSON, [filtering.md](filtering.md)) the scan tests each candidate's attrs
   word (in the entry table, inside the prefix) before it can enter the heap, and once the first `nprobe` slabs
   are scanned the query keeps scanning cells in routing order, in batches, until `T` candidates match or
   `maxNprobe` (default 4 x nprobe) cells were scanned.
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
