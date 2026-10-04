# thread: a blob-native search index (dense + lexical)

Cosine-similarity search over vector embeddings and BM25 search over text, over one
document id space and fused by rank, where the index lives in object storage
(Cloudflare R2) and queries run in RAM-constrained environments (Cloudflare Workers,
128 MB). The build also runs inside Workers. The dense side is described below; the
lexical side (BM25 segments, the unified document store, fusion) is in
[docs/lexical.md](docs/lexical.md) and summarized in
[Lexical search and unified docs](#lexical-search-and-unified-docs).

## Design in one paragraph

Two-level spherical k-means (`k1` coarse groups x `k2` fine cells; `k1=1` is a flat
k-means, the default since g2). Every vector
is written into the slabs of its `m` nearest fine cells (multi-assignment), so a
single probed slab already contains the query's neighbourhood. A slab is one
immutable R2 object holding the residuals of its members (`x - centroid`) twice: a
compact scan tier (PQ-192 codes since g4, 4-bit codes in g2/g3) and an int8 rerank
tier, plus ids and an optional payload section. A query routes against the cached
centroid table, range-reads the scan tier of `nprobe` slabs, scans them with a bounded
top-k heap, merges and dedupes, then re-scores the best `rerank` candidates (150 by
default) with their int8 records. Storage is traded for fewer bytes read per query
everywhere. Updates never rewrite an object: an upsert or delete adds a small delta segment
and/or tombstone set plus a new manifest *revision*; compaction folds them into new slab
revisions (see [Updates](#updates-manifest-revisions-deltas-tombstones-compaction)), automatically once a revision
exceeds the generation's compaction thresholds; cells that outgrow a cap are split in two.

Everything numeric is in Rust (`crates/thread-core`), exposed to JS through
wasm-bindgen (`crates/thread-wasm`). The Worker (`worker/`) is an I/O shell that
streams R2 objects through the wasm kernels so no stage holds more than a few
tens of MB; the SDKs (`sdk/`) are the same query path as a library, plus the lexical
side, the write path and the fused `search`: `sdk/ts` for TypeScript over the wasm, and
`sdk/python` for Python over a PyO3 extension of `thread-core` (`crates/thread-py`), both
published as `thread-sdk` (npm, PyPI).

Text lives in **segments** next to the slabs: immutable `THREADIX` containers holding
BM25 postings (a bucketed term table carrying each term's `df` and posting range, lists
of `gap/tf/length` varints) and a `docs` store (id -> metadata, optionally the text),
read by range requests in two rounds per query. Segments share the slabs' u32 ids and
tombstone sets, so one upsert adds a vector, its text and its metadata, and one delete
hides all of them; a compaction merges segments into one base. A query can be a vector,
a text or both (Reciprocal Rank Fusion), and hits resolve through the one `docs` store.

## Results so far (dbpedia-openai-1M-1536-angular, 975k vectors, top-10, warm edge cache)

Cloudflare's published Vectorize numbers on this dataset are the reference: p50 31 ms,
p95 380 ms, 343 RPS at concurrency 300, 95.4% precision. Ours, measured from a client in
Montreal against `vib-bench` with `bench/query_bench.mjs` (warm pass, then measured pass):

| generation | layout | recall@10 | p50 ms (c=300) | p95 ms | RPS | scan MB/query | build |
|---|---|---:|---:|---:|---:|---:|---|
| g1 | 64x64 tree, int8, m=3, tree routing | 88% at nprobe 8 | 389 (95% at nprobe 22, flat routing) | 827 | 560 | 41 | 22 min |
| g2 | flat 4096, 4-bit + int8 rerank, m=3 | 95.4% at nprobe 26 | 132-192 | 330-453 | 1263-1766 | 16.5 | 85 min |
| g3 | as g2 with m=8, incremental k-means | 95.3% at nprobe 16 | 273 | 670 | 780 | 32 | 42 min |
| g4 | flat 4096, PQ-192 + int8 rerank 150, m=3 | 95.1% at nprobe 34 | 83-158 | 204-354 | 1480-2650 | 6.2 | 30 min |

Ranges reflect network variance between sessions; rows within one session are comparable
(see `data/bench-*.txt`). Cost at 1M vectors and 1M queries/month is about $1-2 against
roughly $21 for Vectorize. Writes: upsert of 500 vectors 1.4 s (2000: 3.3 s) with the centroid tree, delete 0.8 s,
visible within 30 s; 20 pending delta segments add ~6 ms to p50; past 16 segments the writer compacts one on its own
(~19 s in the background, no failed query). g4 now has 4103 cells (its 7 largest were split, recall unchanged).

What the generations taught us: the recall gap was routing, not quantization (rank all cells,
never prune through coarse averages); at fixed recall the number of scanned entries is nearly
constant, so replication (m) only trades hops for bytes and PQ is the real bytes lever; the
Cache API honours multi-range requests and is not bound by the 6-connection limit, which makes a
150-record rerank cheap.

## Slab formats and the two-phase query

A v2 or v3 slab stores every entry twice. The **scan tier**, bytes `[0, scan_len)`, holds
the header, the cell mean, the entry table (16 B per entry) and the scan codes:

- **v3 (format 3, `tiers: "pq192+sq8"`, g4+):** one byte per PQ subspace, `pq_m` = 192
  subspaces of 8 dims with 256 sub-centroids each, one codebook for the whole index
  (`idx/{gen}/pq.bin`, 1.5 MB). 208 B/entry at dim 1536.
- **v2 (format 2, `tiers: "sq4+sq8"`, g2/g3):** 4-bit residual codes with a per-vector
  MSE-clipped scale. ~790 B/entry.

The **rerank tier**, at `rerank_off`, is the same in both: one `4 + dim` byte record per
entry (`[scale8 f32][code8 i8 * dim]`), so record `i` is the range
`[rerank_off + i*(4+dim), +4+dim)`. `publish` writes `idx/{gen}/slabs.json` (array indexed
by cell: `[scanLen, rerankOff, count, bytes]`; compactions add a fifth column, the slab revision)
and records `format` (and for format 3 `pq: {m}`) in the manifest. v1 generations (int8 only, e.g. `g1`) are still served: the
Worker reads their slabs whole and skips phase 2. One Worker serves every format; the
format is read from the manifest and from each slab header.

A query (`POST /query?nprobe=N&rerank=R&pc=P&route=flat|beam&beam=B`) runs in two phases. Every
parameter is optional: an absent one takes the revision's `manifest.defaults` (see
[Profiles](#index-time-profiles)); manifests without `defaults` (g1, g2) fall back to
`nprobe=1, rerank=32, pc=0`, and an absent `route` means flat. The response's `stats.params` shows
the values used and where each came from (`query`, `manifest`, `tree` or `legacy`).

1. **Scan.** Route (flat by default, `route=beam` through the [centroid tree](#centroid-tree-and-beam-routing)),
   then read only the scan prefix of each probed
   slab: a `Range: bytes=0-(scanLen-1)` match against the Cache API copy (it answers
   206 from a cached full object). On a cache miss the Worker GETs the whole object
   from R2, caches it in the background, and keeps the rerank records of that slab's
   local winners in memory. Each prefix is scanned for the top `T = max(k, rerank)`:
   format 3 with `thread.scan_pq(prefix, table, q, T)`, where `table = PqTable.from_codebook(q, codebook)`
   is built once per query (192 x 256 dot products) and freed when the query ends;
   format 2 with `thread.scan`. Reads are issued as a sliding window of 12 in flight per
   query (`QUERY_INFLIGHT`). Cache API reads are not held to the Workers limit of 6
   open connections (measured in `data/bench-latency.txt`); R2 reads are retried once
   if the runtime cancels them for the connection limit.
2. **Rerank.** The global top-T (after `merge`) is grouped by slab, and the winners of
   each slab are read with ONE multi-range Cache API request (`Range: bytes=a-b, c-d, ...`,
   one range per record, adjacent records within 4 KB merged; the 206 multipart/byteranges
   answer is parsed in `cacheMatchMulti`), then re-scored with `thread.rerank`. The top k by
   reranked score are returned. **Speculative reads:** once only 6 slabs remain to be
   scanned, the running top-T is read the same way at low priority (only when no prefix
   read is waiting), so phase 2 usually finds every record already in memory and costs
   about one request per winner slab at most. `rerank=0` disables phase 2 (scan-tier
   scores only). Format 1 generations (int8-only slabs) always skip phase 2.

Offline (`data/quant-experiment.md`) 4-bit scan + int8 rerank of the top 32 equals
int8 recall at half the scan bytes; the PQ-192 scan tier needs a rerank of the top
100-200 (93.3% / 94.0% vs 94.2% for int8 at nprobe 22) but scans ~3.8x fewer bytes than
4-bit. With multi-range and speculative reads, phase 2 costs one request per winner slab
(~16 at T=100, ~20 at T=200 on g2), not one per record. Stats in the response include
`scanBytes`, `rerankReads`, `rerankBytes`, `specUsed`, `tiers`, `pqM` and the timings
`phase1`/`phase2`.
`GET /debug/range?key=idx/...&offset=&length=` shows how the Cache API answers a Range
request for a cached object (status, Content-Range, and whether the bytes match R2).

The wasm instance installs a panic hook after init. If a request hits a wasm trap
(`WebAssembly.RuntimeError`), the Worker calls `__wbg_reset_state()`, re-inits, drops
the cached index (its `Searcher` and `PqCodebook` pointers are invalid) and retries the
request once.

A query reads the (gen, rev) that `idx/current.json` names (re-read at most every 30 s per
isolate; `?gen=&rev=` overrides). Parsed revisions (manifest, slabs table, tombstone sets) are
cached per isolate and never revalidated, since revisions are immutable. The base (Searcher over
the centroid tables, PQ codebook) is keyed by its object keys and shared by every revision of the
generation that uses the same files; one base is cached per isolate (`BASE_CACHE_SIZE`).
Queries hold it while they run. When a query for another generation replaces it, the old one is
freed once its last query finishes, so alternating generations does not leak wasm memory.
Wasm objects are freed only in the instance (epoch) that created them. The centroid tree is not part
of the base key: a revision that adds or changes a tree reuses the loaded means, and the tree it names
is installed (`Searcher.set_tree`) right before the query routes.

### Centroid tree and beam routing

Flat routing scores the query against all 4096 cell means (`cell_dots`, 4096 x 1536 multiply-adds),
and an upsert used to do the same per vector to find its primary cell. A **centroid tree** prunes that
search: `thread.build_centroid_tree(fineMeans, dim, groups, iters, seed)` clusters the cell means into
`groups` group means (spherical k-means over the means), and a lookup scores the group means first, then
only the cells of the best `beam` groups (~128 + beam x 32 dot products at 128 groups).

- **Object.** `publish` and `publish-rev` take `tree: {groups: 128, iters: 10, seed, beam: 8}` and write
  `idx/{gen}/tree.{rev}.bin` (~0.8 MB at 4096 x 1536, 128 groups), referenced from the manifest as
  `tree: {key, groups, beam, nearestRecall, top3Recall, recallByBeam?}`. The recall figures come from
  `Searcher.tree_recall` / `tree_recall_top(sample, beam, 3)` on ~2000 base vectors (250 rows from each of 8
  k-means sample objects). Later revisions inherit the tree; `publish-rev {tree: null}` removes it.
  The driver builds it with `--tree groups,beam` (edge profile: 128,8); for an existing generation:
  `node bench/driver.mjs --gen g4 --publish-rev --tree 128,8`.
- **Upserts** call `DeltaJob.set_tree(tree, tree.beam)` before encoding, so the primary cell comes from the
  pruned search. On g4 (beam 8 finds the exact nearest cell for 99.35% of vectors) a 2000-vector upsert
  went from 7.5 s to 3.3 s, 500 from 2.5 s to 1.4 s in a same-session A/B; self-retrieval stayed 100% before
  and after folding the vectors into slabs (`data/bench-tree.txt`).
- **Queries** take `route=beam` (and `beam=N`, default `defaults.beam`, else `tree.beam`): `route_beam(q, beam,
  nprobe)` replaces `cell_dots` + `route_from_dots`. Delta segments then need `q·mean` only for the cells
  they hold entries in: `cell_dots_for(q, cells)` over the union of each segment's `delta_cells` (cached per
  segment), scattered into the `qc` table `scan_delta` reads (scores identical to the flat path).
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

  Beam 8 costs 0.6-0.8 points of recall; beam 16/32 keep recall within 0.2 points but no beam improves p50 or
  RPS measurably (routing is ~1-2 ms of CPU in a 26-34 ms server time spent mostly on Cache API reads, and the
  34 probes stay the same). So **flat stays the default** (g4 rev 103: `defaults.route: "flat"`, `defaults.beam: 32`
  for queries that ask for `route=beam`); the tree pays on the write path.

## Index-time profiles

Every layout knob is a build parameter recorded in the manifest (`idx/{gen}/manifest.json` = rev 0),
together with the query defaults the Worker applies when a request omits them. Defaults are changed
with a new revision (`POST /build/publish-rev {gen, defaults}`), never by republishing rev 0:

```json
{"gen":"g4","profile":"edge","k1":1,"k2":4096,"m":3,"format":3,"tiers":"pq192+sq8","pq":{"m":192},
 "nNeigh":64,"defaults":{"nprobe":34,"rerank":150,"pc":0,"route":"flat","beam":32}, "probeCoarse":1,"vectors":975000,
 "tree":{"key":"idx/g4/tree.88.bin","groups":128,"beam":8,"nearestRecall":0.9935,"top3Recall":0.98067,...}, "rev":103, ...}
```

| profile | k1 x k2 | m | format / tiers | PQ (pqM, pqIters, pqSampleObjects) | nNeigh | tree (groups, beam) | defaults (nprobe, rerank, pc) | intent |
|---|---|---:|---|---|---:|---|---|---|
| `edge` (default) | 1 x 4096 | 3 | 3 / `pq192+sq8` | 192, 12, 30 | 64 | 128, 8 | 34, 150, 0 | g2's layout (m=3) with a PQ scan tier: ~3.7x fewer scan bytes per probe; the coarser PQ scores are fixed by an int8 rerank of the top 150, and nprobe 34 (instead of g2's 26) brings recall back to 95% |
| `local` | 1 x 4096 | 1 | 3 / `pq192+sq8` | 192, 12, 30 | 64 | none | 32, 150, 0 | no duplication, smallest index, more probes |

The g3 edge profile (m=8, format 2, nprobe 10 / rerank 20) aimed at few probes because the
Worker was then assumed to be limited to 6 reads in flight. Cache API reads turned out not to
be (`data/bench-latency.txt`), so the edge profile went back to m=3 and spends the bytes on
probes instead of duplication.

g4 (edge, `data/bench-g4.txt`, same session as a g2 reference): at nprobe 26 a query scans
4.8 MB instead of 16.5 MB and reaches 93.8% recall@10 (n=1000) / 94.1% (n=5000), about 1.1 points
below g2 at the same nprobe. In exchange it has p50 76 ms and 526 RPS at c=50 (g2: 108 ms, 363 RPS)
and p50 126 ms and 1788 RPS at c=300 (g2: 192 ms, 1224 RPS). At nprobe 34 it reaches 95.1%
(c=300 p50 158 ms and 1480 RPS in that session, 6.2 MB scanned), still faster than g2's 95.4%.
nprobe 34 / rerank 150 are now g4's defaults (manifest rev 1, via `publish-rev`). Official operating
point (rev 79, a later session with a faster network split over the YUL and YYZ colos): recall 94.89%
(n=1000) / 95.12% (n=5000), c=50 p50 73 ms, p95 122-128 ms, ~600 RPS; c=300 p50 83-85 ms, p95
204-223 ms, 2000-2650 RPS. A same-session nprobe 26 reference: c=50 p50 66 ms, c=300 p50 78 ms.
Rerank T is the recall knob for the PQ tier: T=50/100/150/200 give 92.0/93.5/93.8/94.0% at nprobe 26,
at ~1 multi-range request per winner slab (7 to 13) and 1.6 KB per record.

Explicit flags override the profile (`--m`, `--k2`, `--format`, `--tiers`, `--pqM`, `--pqIters`,
`--nNeigh`, `--tree`, `--defNprobe`, `--defRerank`, `--defPc`, `--defRoute`, `--defBeam`, ...). `--tiers` implies the format and vice
versa (`pq192+sq8` = 3, `sq4+sq8` = 2, `sq8` = 1). `--format 2` (or 1) skips the PQ stage and
defaults `rerank` to 32 unless `--defRerank` is given.

## Updates (manifest revisions, deltas, tombstones, compaction)

Details: [docs/update-path.md](docs/update-path.md). Measurements: `data/bench-updates.txt`.

Every change is a new manifest revision `idx/{gen}/manifest.{rev}.json` (rev 0 = `manifest.json` from
`publish`); `idx/current.json` is `{gen, rev}`. A revision lists its files (`coarse`, `fine`, `pq`, slabs
table), its delta segments (`delta/{seq}.vibd`, each with the rev that added it) and its tombstone sets
(`tombstones.{rev}.u32`, the ids deleted or replaced at that rev). Nothing is overwritten, so every object
can stay in the edge cache for a year and a query always reads a consistent snapshot.

- `POST /upsert` (JSON `{ids, vectors, attrs?, payloads?}` or a binary f32 body with `x-ids`/`?firstId=`, up to
  4096 vectors): one delta segment (`DeltaJob`: primary cell, found through the centroid tree when the manifest has
  one, + 4-bit/int8/PQ codes against the cell mean) and a
  tombstone set for the ids (replace semantics: older copies stop resolving), committed as rev + 1. Responds
  `{gen, rev, seq, cells, ms, bytes, conflicts}`.
- `POST /delete {ids}`: one tombstone set, rev + 1.
- Writers serialize on a create-only PUT of `manifest.{rev+1}.json` (R2 `onlyIf: {etagDoesNotMatch: "*"}`),
  retry on a lost race, and advance `current.json` with an etag compare-and-swap.
- Queries scan every delta segment with `scan_delta` (int8 scores, no phase 2) next to the probed slabs and
  apply the tombstone rule: a slab hit is dropped if its id is in a set with rev >= the slab's revision
  (every set for rev-0 slabs); a delta hit if its id is in a set newer than the rev that added the delta.
  New data is visible within 30 s (current.json TTL), or immediately with `?rev=` from the write response.
- **Automatic compaction** (`manifest.compaction`, profile defaults: more than 16 segments, 32 MB of segments or
  50,000 tombstone ids): after an upsert/delete commit that exceeds a threshold, the writer takes
  `idx/{gen}/compact.lock` (create-only PUT, stale after 10 min) and compacts a bounded slice in `ctx.waitUntil`: whole
  segments, oldest first, up to `cellsPerRun` (256) cells, purge cells when the tombstone threshold is exceeded, nothing
  committed after `deadlineMs` (27 s) from the trigger; the commit is the usual create-only manifest PUT + current.json
  CAS (a lost race is recorded as `conflict`, the next write triggers again). `GET /index/status` shows the pressure,
  thresholds, lock and the latest runs. Set or change it with `node bench/driver.mjs --gen g4 --publish-rev
  --compaction profile [--compactMaxDeltas N --compactMaxDeltaMB N --compactMaxTombstones N --compactMaxCellEntries N
  --compactCellsPerRun N --compactDeadlineMs N --compactSplitCellsPerRun N]` (`--compaction 0` removes it).
- **Cell splitting:** a compaction splits every rewritten cell whose entry count would exceed `maxCellEntries` (edge
  3000, local 1000): 2-means over the cell's decoded entries (`split_slab`), every entry re-encoded against its new
  mean (`split_slab_records`), the cell keeps mean A and a new cell k is appended with mean B; the revision gets
  `fine.{rev}.f32` (k + 1 means), `k2` + 1 and a rebuilt centroid tree, and query isolates load the new base on their
  first query of it. Automatic runs split only in slices of at most `splitCellsPerRun` (160) cells. Details:
  [docs/update-path.md](docs/update-path.md#cell-splitting).
- `node bench/compact.mjs --gen g4 [--purge 1] [--maxCellEntries N]` (or `POST /build/compact`): rewrites every cell holding delta
  entries from its old slab's records minus tombstones plus the delta records (`slab/{cell}.{rev}`), commits a
  revision without the folded deltas, and drops the tombstone sets nothing can reference any more. A delta is
  folded only when every cell it touches is rewritten; a partial compaction (`--cells`) keeps the rest. `--purge`
  also rewrites cells earlier compactions wrote, so the replace/delete sets can be dropped.
  `POST /build/gc {gen, dryRun: false}` deletes objects no recent revision references.

Measured on g4 (`bench/update_bench.mjs`, the 5000 test queries inserted as ids 1,000,000+i):

| | |
|---|---|
| upsert latency (writer Worker, client) | with the centroid tree (128 groups, beam 8): 100: 1.16 s, 500: 1.43 s, 2000: 3.27 s; same session without it (exhaustive primary-cell search, ~2.6 ms/vector): 1.42 / 2.49 / 7.48 s (`data/bench-tree.txt`). An earlier session measured 0.86 / 1.86 / 5.07 s exhaustive (250: 1.13 s); ~0.5-0.9 s is sequential R2 round trips |
| bytes written per upsert | 2528 B per vector (delta segment) + 4 B per id (tombstones) + ~1-3 KB manifest |
| delete latency | ~0.75 s (1000 or 5000 ids) |
| visibility | own id at rank 1 for 100% of the inserted queries (counting the 894 queries that have an identical twin), in the top 10 for 100%; deleted ids never returned |
| recall@10 vs baseline (95.12%, nprobe 34) | within 0.02 points with 20 deltas, after deletes, after compaction and after replaces |
| query cost of deltas (c=50, interleaved A/B) | +2.4 ms p50 with 5 segments (3 MB), +5.6 ms with 20 (12 MB); ~0.45 ms per MB |
| compaction of 5000 vectors | 2151 cells, 3.0 GB read + 3.0 GB written, 43-51 s over the former 8-Worker pool; latency back at baseline; all tombstone sets dropped |
| cleanup | delete everything + `compact --purge` -> the original slabs again (byte-identical, checked by MD5 on a sample cell), recall = baseline; `gc` then deleted 12.2 GB of superseded objects |
| automatic compaction over build slots (`--only trigger`, `data/bench-trigger-slots.txt`) | 17th upsert of 250 vectors triggers it; the cells are spread over 4 slots: 606 cells (3 segments folded) in 16.7 s, trigger -> new rev 18.9 s; 0 failed out of 7,779 background queries; query isolates stay at 54 MB of wasm memory |
| automatic compaction, single writer (before the build slots, `data/bench-trigger.txt`) | 17th upsert of 250 vectors triggers it; trigger -> new rev 18.8 s (228 cells = 1 segment in 16.5 s, commit 1.1 s) on vib-bench-0; 0 failed queries out of 9,555 background queries; p50 (interleaved A/B, c=50) 76.8 ms clean, 81.9 ms at 17 segments, 79.3 ms after the run; cleanup back to a clean head at 95.13% |
| cell split (`bench/split_bench.mjs`, `data/bench-split.txt`) | 7 cells over 2400 entries split (k 4096 -> 4103): largest cell 3109 -> 2257 entries, recall@10 95.12% -> 95.13%, query isolates' wasm memory unchanged (54 MB) |

Writes run in the `writer` build slot (the Worker forwards `/upsert` and `/delete` there unless `?slot=local`):
`DeltaJob` holds a ~25 MB copy of the means (~50 MB while it is passed in), and wasm memory never shrinks, so a
write must not run in an isolate that serves queries. (Until g4 the benchmarks sent writes to vib-bench-0 of the
build pool for the same reason.)

## Lexical search and unified docs

Details: [docs/lexical.md](docs/lexical.md). The lexical side was ported from Wiper's `thread` engine
(`../wiper/rust/thread`: demand-paged `ByteSource`/`Prefetched`, `THREADIX` container, bm25s-compatible
tokenizer with Snowball 3.1 English, CJK bigrams and technical identifiers, RRF, excerpts, judge-side rank
fusion) and reshaped for blob storage with sparse ids: no term ids or vocabulary table (a `terms` bucket entry
carries `df` and the posting range), document lengths inline in the postings (no corpus-sized array per query),
a `docs` store bucketed by id, segments + tombstones instead of whole-file appends.

- **Objects.** `idx/{gen}/seg/{seq}.thread` per upsert with texts/metadata (the `seq` of its dense delta),
  `idx/{gen}/seg/base.{rev}.thread` after a compaction; listed in the manifest as `segments` with the
  configuration in `manifest.lex` (`index.configureLexical` on a dense generation, or `createLexicalGeneration`
  for a text-only index without base files).
- **Query.** Tokenize with the index's config; per segment, one round for the query terms' buckets and one for
  their posting lists (one multi-range Cache API request each), all segments concurrently; scored with
  corpus-wide `N`, `avg_len` and `df` summed over the segments, so segments score exactly like one index; merged,
  and fused with the dense list by RRF when both are given. Docs resolve newest segment first, tombstone-aware.
- **Tuning** (per generation, `manifest.lex`): `lengths: "exact" | "byte"` (exact scores vs ~1 byte less per
  posting), `bucketTerms` / `bucketDocs` (resident directory size vs bytes per lookup), `docs` / `text` (store
  metadata, store texts), the tokenizer (stemming, stopwords, CJK bigrams, identifiers), RRF defaults.
  Measured on a 20k-document synthetic corpus: 3.63 B/posting exact, 2.69 B/posting byte lengths, 2.5 KB
  resident directory per 30k terms, a 4-term query in 4.6 ms natively.
- **Updates.** `index.upsert({ids, vectors?, texts?, metas?})`, `index.delete(ids)` (shared tombstone sets:
  a segment added at rev `a` is filtered by the sets with `rev > a`, the delta rule), `index.compactSegments()`
  (merge to one base; byte-identical to a fresh build over the live documents). The Worker's compaction keeps a
  tombstone set while any segment predates it, and its GC keeps every listed segment.
- **CLI.** `thread seg build|inspect|search|merge`, `thread tokenize` on local files (`crates/thread-cli`):
  a segment built locally is what an upsert writes.

```ts
await index.configureLexical({ text: true });
await index.upsert({ ids, vectors, texts, metas });
const { hits } = await index.search({ text: "deploy failed", vector: embedding, topK: 10, docs: "text" });
// hits[i]: { id, score, dense?: {rank, score}, lexical?: {rank, score}, doc?: {id, meta, text} }
```

## Layout

- `crates/thread-core`   format, quantization, k-means accumulators, router, scan; lexical side: `source`
                      (demand paging), `container`, `tokenize`, `lex` (terms/postings), `docs`, `segment`,
                      `fusion`, `excerpts`, `rerank`
- `crates/thread-wasm`   wasm-bindgen jobs: FlatKmeansJob, TreeKmeansJob, IncKmeansJob, Reducer, AssignJob, SlabJob, Searcher, cell_neighbors,
                      PQ: residuals_for, pq_init_codebook, PqTrainJob, PqReducer, PqCodebook, PqTable, scan_pq,
                      updates: DeltaJob, scan_delta, delta_info/cells/records_for_cell/payload, slab_to_records,
                      tombstones_merge, filter_hits, cell splits: split_slab, split_slab_records, split_part_records,
                      reencode_records,
                      centroid tree: build_centroid_tree, centroid_tree_info, Searcher.set_tree/route_beam/cell_dots_for/
                      tree_recall/tree_recall_top, DeltaJob.set_tree, AssignJob.set_tree (not used by the driver yet),
                      lexical (`src/lex.rs`): LexTokenizer, SegmentJob, SegmentReader (paged), merge_segments, rrf_fuse,
                      best_doc_ranks, rerank_fuse, Lines, Excerpter
- `crates/thread-cli`    native CLI: `seg build|inspect|search|merge`, `tokenize`, slab/scan benchmarks
- `crates/thread-py`     PyO3 extension (`thread_sdk._native`): Searcher, PqCodebook, the per-query `QueryRun`
                      (scan, merge, rerank), DeltaJob, tombstones, AssignJob/SlabJob, LexTokenizer, SegmentJob,
                      SegmentReader (paged), merge_segments
- `sdk/ts/`           `thread-sdk` (npm): the query path as a library (R2/S3/memory stores, Cache API), the lexical
                      index, `search` (dense, lexical, hybrid), the write path (upsert with texts, delete, segment
                      compaction)
- `sdk/python/`       `thread-sdk` (PyPI): the same library for Python (S3/R2, directory and memory stores,
                      memory/disk caches), wire-compatible with `sdk/ts` (see `sdk/python/README.md`)
- `worker/`           `cf`-managed Worker: build-stage endpoints, `/query`, `/upsert`, `/delete`, compaction (manual and
                      automatic), cell splits, `/index/status`, rollback, gc (`src/index.ts`); `src/main.ts` runs build
                      stages and writes in build slots (Worker Loader); see `worker/README.md`
- `spikes/worker-loader/` the spike that showed dynamic Workers give each slot its own isolate (FINDINGS.md)
- `bench/`            dataset prep, shard upload, build driver, query benchmark, compaction driver, update benchmark
- `docs/`             `update-path.md`: the update path (revisions, deltas, tombstones, compaction) in detail;
                      `lexical.md`: the lexical index, the docs store and fusion
- `data/`             (gitignored) dataset and prep output

## Build pipeline (all stages run in the Worker, driven from a laptop)

```
sample          -> build/{gen}/sample/{i}.f32        strided sample (160k vectors)
init-coarse     -> build/{gen}/coarse/0.f32           (k1=1: the only coarse table)
kmeans-flat     -> build/{gen}/coarse/{it}.f32        Lloyd iterations on the sample (skipped if coarseIters=0)
init-fine       -> build/{gen}/fine/0.f32             reservoir per coarse group
fine k-means, iteration 1 (exhaustive):
  kmeans-tree-map -> partials/fine1/{j}.acc + assign/1/{shard}.u32   every vector vs all cells
fine k-means, iterations 2..N (incremental):
  cell-neighbors  -> neighbors/{it-1}/{r}.u32, concat -> neighbors/{it-1}.u32   k x nNeigh table of fine/{it-1}
  kmeans-inc-map  -> partials/fine{it}/{j}.acc + assign/{it}/{shard}.u32  vector vs nNeigh neighbours of its last cell
kmeans-reduce / concat -> build/{gen}/fine/{it}.f32   (every iteration; small-cell reseeding on 1..N-2)
PQ codebook (format 3 only, stage `pq`):
  pq-residuals    -> pq/residuals/{i}.f32             sample group i minus its final cell means (~2000 rows, 12 MB)
  pq-init         -> pq/codebook/0.bin                256 random residual rows per subspace (from 2 objects)
  pq-train-map    -> pq/partials/{it}/{j}.acc         8 jobs: assign subvectors, accumulate (1.8 MB partial)
  pq-train-reduce -> pq/codebook/{it}.bin             PqReducer.finish(prev); reports mean_sq_err
assign          -> build/{gen}/parts/{g}/{job}.part   top-m among the neighbours of the final cell, 4-bit + int8 (+ PQ codes)
finalize        -> idx/{gen}/slab/{cell}              one span read per part per cell chunk, v3 (or v2/v1) slabs
publish         -> idx/{gen}/manifest.json (rev 0), coarse.f32, fine.f32, slabs.json, pq.bin, tree.0.bin (with --tree), idx/current.json {gen, rev: 0}
publish-rev     -> idx/{gen}/manifest.{rev+1}.json    changed query defaults and/or a new centroid tree (tree.{rev+1}.bin);
                                                      publish refuses to rewrite rev 0
```

Memory per stage is bounded by the centroid table (k1*k2*dim*4 bytes, 25 MB at
4096 x 1536) plus one in-flight shard (12.6 MB) plus the job's output buffers
(k-means: a 25 MB accumulator; assign: ~2.3 KB (+ pqM bytes with a codebook) x m per
vector of parts; finalize: the SlabJobs of one cell chunk, capped by `budgetBytes`;
PQ: one 12 MB residual object, a 1.5 MB codebook and a 1.8 MB partial).

### Build slots (Worker Loader)

The driver needs many heavy jobs in parallel (k-means map and assign use 60-100 MB each), but concurrent requests
to one Worker can share a single 128 MB isolate. Until g4 that meant a pool of eight identical scripts
(vib-bench-0..7). Now there is one deployment: `worker/src/main.ts` forwards `POST /build/<stage>?slot=<id>` to
the dynamic Worker `thread-build:<bundle hash>:slot-<id>` (`env.LOADER.get`), which runs a second bundle of the
same `worker/src/index.ts` in its own isolate with its own 128 MB and `limits.cpuMs` 300,000. The driver keeps a
free list of slot ids `0..THREAD_SLOTS-1` (default 8) and sends one job per slot at a time; retries, bodies and auth
are unchanged. R2 reaches the slots through a loopback RPC entrypoint (`BucketRpc`; an R2 binding cannot be put in a
dynamic Worker's env) and a shim that gives the bundle an R2Bucket-shaped `env.BUCKET`; object bodies stream
through the orchestrator in both directions. Writes (`/upsert`, `/delete`) run in the `writer` slot and the
automatic compaction spreads its cells over slots `auto-0..3`. `?slot=local` runs a request in the query isolate
(debugging). Design and measurements of the isolation: `spikes/worker-loader/FINDINGS.md`; deploy notes and
routing: `worker/README.md`.

g5 was built this way with the same seed and profile as g4 (`node bench/driver.mjs --gen g5 --profile edge
--setCurrent 0`, `data/build-g5.log`): identical k-means, PQ, assignment and slab statistics, recall@10 95.12% at
nprobe 34 (c=50, n=2000), and no slot errors (one client-side `fetch failed`, retried):

| stage (s) | sample | init-fine | fine | pq | assign | finalize | publish | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| g4, 8-Worker pool | 11.1 | 44.2 | 1017.2 | 70.2 | 174.4 | 466.6 | 2.0 | 1786 |
| g5, 8 build slots | 11.7 | 33.3 | 914.7 | 59.6 | 154.7 | 450.9 | 15.2 | 1645 |

(g5's publish also builds the centroid tree and measures its recall; g4 got its tree later with publish-rev. The
driver then deletes the intermediates nothing reads any more, `build/{gen}/partials/`, `parts/`, `pq/partials/`:
22 GB for g5.) Per-job times are bimodal (~21 s or ~53 s for the same 10-shard k-means map job) on either setup:
host contention, not the slot mechanism.

The query Worker was chosen over a sibling `thread-build` Worker: one deployment instead of two, the slots' code is
guaranteed to be the version that serves queries (same build, same wasm module), and writes keep working on the
query URL. The orchestrator isolates only forward requests and pump RPC streams (no object is buffered there).

### PQ codebook training (format 3)

The scan tier encodes each entry's residual against its slab's cell mean with one
index-wide product quantizer: 192 subspaces of 8 dims, 256 sub-centroids each. Training
runs on residuals of the k-means sample against the **final** k-means cells:

1. `pq-residuals` (one job per sample group, the first `pqSampleObjects` = 30 groups of
   the sample stage): reads the first `perShard` (336) vectors of each of the group's ~6
   shards and the first `perShard` entries of their final assignment
   (`assign/{fineIters}/{shard}.u32`), and writes `thread.residuals_for(vectors, means, assign)`
   (~60k rows, 370 MB in total). It also reports the mean residual energy `|r|^2`.
2. `pq-init`: `thread.pq_init_codebook` over the first two residual objects (~4000 rows).
3. `pqIters` (12) Lloyd iterations: 8 `pq-train-map` jobs (`PqTrainJob`, ~4 residual
   objects each) write partial sums, one `pq-train-reduce` (`PqReducer`, partials streamed
   one at a time) writes the next codebook and reports `mean_sq_err` (the error under the
   codebook the iteration started from) and the empty sub-centroids, which keep their
   previous value. The driver logs `mean_sq_err` as a fraction of the residual energy.

`assign` then calls `AssignJob.set_codebook(pq/codebook/{pqIters}.bin)`, so every part
record also carries its PQ codes (part format version 2), `finalize` builds v3 slabs
(`SlabJob(dim, cell, mean, 3)`), and `publish` copies the codebook to `idx/{gen}/pq.bin`
and adds `pq: {m}` to the manifest. The Worker parses the codebook once per isolate
(`PqCodebook`) and builds one `PqTable` per query.

### Incremental k-means

An exhaustive flat iteration compares 975k vectors with 4096 means (~6 TFLOP at dim 1536)
and took ~7 min per iteration in g2. After the first iteration most vectors stay in or near
their cell, so iterations 2..N compare each vector only with the `nNeigh` (64) cells closest to
its previous cell (`IncKmeansJob`), 64x less arithmetic; the map stage becomes I/O bound
(reading the 6 GB of shards). The neighbour table (`cell_neighbors`: per cell, the `n` cells with
the highest cosine between means, the cell itself first) is recomputed from the new means after
every iteration by 8 `cell-neighbors` jobs of 512 rows (~1 s each) and a `concat`. Each map job
writes the new cell of every vector to `build/{gen}/assign/{it}/{shard:05}.u32` and reports
how many vectors changed cell. Cells reseeded by the reducer move next to a dense cell and
appear in that cell's neighbour rows, so its vectors can move into them on the next pass.
`--exactLast` makes the last iteration exhaustive again.

`assign` then uses the same idea (`AssignJob.with_candidates`): each vector's top `m` cells are
chosen among the `nNeigh` neighbours of its final k-means cell, from a neighbour table of the
final means. The per-job part buffer is ~2.3 KB (+ 192 B of PQ codes in format 3) x m per
vector: with m=3 a job takes two shards (~31 MB of parts), with m=8 one shard (~38 MB;
`--assignSplit 2` halves it further).

`finalize` reads each part's directory, then ONE contiguous span per part covering the job's
whole cell chunk (`part_span`), and hands each cell its slice with
`SlabJob.add_records_range` (no per-cell range reads). R2 reads are capped at 6 in flight per
job (`SUBREQUEST_CONCURRENCY`, the per-invocation connection limit for R2). If the
chunk's cells hold more than `budgetBytes` (default 48 MB) of records, the Worker builds them in
consecutive batches, each with its own smaller spans.

## Running

```sh
# prep dataset (dbpedia-openai-1M-1536-angular) and upload shards
.venv/bin/python bench/prep_dataset.py
bench/upload_shards.sh 6

# build wasm and deploy the worker
cargo build --release --target wasm32-unknown-unknown -p thread-wasm
wasm-bindgen --target web --experimental-reset-state-function --out-dir worker/pkg --out-name thread target/wasm32-unknown-unknown/release/thread_wasm.wasm
(cd worker && cf deploy --secrets-file .env.secrets)   # run from worker/, not the repo root
# That is the only deployment: build stages and writes run in build slots (dynamic Workers of vib-bench,
# one isolate per slot id, see "Build slots" above and worker/README.md). The stage bundle the slots run is
# built by the same `cf deploy` (vite plugin in worker/vite.config.ts). THREAD_ORCH / THREAD_SLOTS point the
# drivers at another deployment / change the build width (default 8).

# build an index generation (edge profile), then benchmark queries
node bench/driver.mjs --gen g4 --profile edge
node bench/query_bench.mjs --gen g4 --n 1000 --concurrency 50          # manifest defaults
node bench/query_bench.mjs --gen g4 --n 1000 --concurrency 50 --nprobe 22 --rerank 100

# change a generation's query defaults (new manifest revision; isolates pick it up within 30 s)
curl -X POST -H "authorization: Bearer $TOKEN" $URL/build/publish-rev -d '{"gen":"g4","defaults":{"nprobe":34,"rerank":150}}'
# add a centroid tree to an existing generation (new revision; records its recall), compare routings
node bench/driver.mjs --gen g4 --publish-rev --tree 128,8 --treeRecallBeams 4,16,32
node bench/query_bench.mjs --gen g4 --n 6000 --concurrency 50 --nprobe 34 --rerank 150 --extras "route=flat|route=beam&beam=8|route=beam&beam=16"
# updates: see "Updates" below and docs/update-path.md
node bench/compact.mjs --gen g4 [--purge 1]           # fold deltas/tombstones into new slab revisions (build slots)
node bench/update_bench.mjs --gen g4                  # upsert/delete/compaction benchmark -> data/bench-updates.txt
node bench/update_bench.mjs --gen g4 --only trigger --out data/bench-trigger.txt   # automatic compaction benchmark
node bench/split_bench.mjs --gen g4 --maxCellEntries 2400   # split the largest cells, recall/latency before/after
curl -H "authorization: Bearer $TOKEN" "$URL/index/status?gen=g4"   # deltas, tombstones, thresholds, lock, last runs
```

`query_bench.mjs` sends `nprobe`/`rerank`/`pc` only when the flag is given and prints the
values the server used (`server params: nprobe=10 (manifest) ...`).

`--experimental-reset-state-function` makes wasm-bindgen emit `__wbg_reset_state()`,
which the Worker uses to recover from a trap. After rebuilding the wasm, `sdk/ts/scripts/sync-wasm.sh` copies it into
the SDK (`cd sdk/ts && npm test`). Lexical segments on local files (`thread --help` and `thread <command> --help`
describe every command and option):

```sh
cargo build --release -p thread-cli
target/release/thread seg build --out seg.thread --docs docs.jsonl --text        # {"id", "text", "meta"} per line
target/release/thread seg search --text "deploy failed" --k 10 --with-docs seg.thread
```

Checks before a commit: `cargo test --workspace`, `cargo check -p thread-core -p thread-wasm --target
wasm32-unknown-unknown`, `cargo fmt --all --check` (the dense-side files predate rustfmt and carry
`#![cfg_attr(rustfmt, rustfmt::skip)]`, so the formatter touches only the lexical modules and new code),
`cargo clippy --all-targets`, `cd sdk/ts && npm run typecheck && npm test`, and
`cd sdk/python && maturin develop --uv && pytest && mypy thread_sdk` (the Python tests include a cross-check against
the built TypeScript SDK in `sdk/ts/dist`).

### Build flags (`bench/driver.mjs`)

The defaults build the `edge` profile: a flat 4096-cell index, m=3, PQ-192 + int8 slabs (g4).

| flag | default | meaning |
|---|---:|---|
| `--profile` | edge | `edge` or `local`: sets k1, k2, m, format, tiers, PQ settings, nNeigh and the query defaults (table above) |
| `--k1` / `--k2` | profile (1 / 4096) | coarse groups x fine cells per group. `k1=1` makes routing and assignment flat with the two-level code |
| `--m` | profile (edge 3, local 1) | cells each vector is written to |
| `--format` / `--tiers` | profile (3 / `pq192+sq8`) | slab format: 3 = PQ scan + int8 rerank tier, 2 = 4-bit scan + int8 rerank tier, 1 = `sq8` int8 only |
| `--pqM` | profile (192) | PQ subspaces (format 3; must divide dim) |
| `--pqIters` | profile (12) | PQ Lloyd iterations (map + reduce) |
| `--pqSampleObjects` | profile (30) | sample groups used for PQ training residuals (~2000 rows each) |
| `--pqMapJobs` | 8 | `pq-train-map` jobs per iteration |
| `--pqStart` | 1 | with `--start pq`: resume PQ training at this iteration (needs `pq/codebook/{it-1}.bin`; residuals are not rebuilt) |
| `--nNeigh` | profile (64) | neighbour-table width for incremental k-means and candidate assignment |
| `--defNprobe` / `--defRerank` / `--defPc` | profile | query defaults written to `manifest.defaults` |
| `--defRoute` / `--defBeam` | absent (flat; tree beam) | `manifest.defaults.route` (`flat` or `beam`, which needs a tree) and the beam of `route=beam` queries that give none |
| `--tree` | profile (edge `128,8`, local none) | `groups,beam` of the centroid tree publish builds over the final fine means (`0` = none) |
| `--treeIters` / `--treeSeed` / `--treeRecallBeams` | 10 / 7 / - | tree k-means iterations and seed; extra beams to record recall for (`4,16,32`) |
| `--publish-rev` | - | instead of a build: one `publish-rev` of `--gen` with the given `--tree` (bare `--tree` = the profile's), `--def*` and `--compact*` flags |
| `--compaction` | profile | `profile` (publish-rev: the profile's policy) or `0` (no policy: no automatic compaction, no splits) |
| `--compactMaxDeltas` / `--compactMaxDeltaMB` / `--compactMaxTombstones` | 16 / 32 / 50000 | automatic compaction thresholds (`manifest.compaction`) |
| `--compactMaxCellEntries` | profile (edge 3000, local 1000) | compactions split cells above this (0 = never) |
| `--compactCellsPerRun` / `--compactDeadlineMs` / `--compactSplitCellsPerRun` | 256 / 27000 / 160 | bounds of one automatic run (cells, ms from the trigger, largest slice that may split) |
| `--pc` | 2 (clamped to [1, k1]; 0 = k1) | coarse groups searched per vector at build time (0 = all groups; `AssignJob`/`TreeKmeansJob` now treat 0 like `Searcher.route` does) |
| `--perShard` / `--sampleGroups` | 336 / 80 | k-means sample: 336 vectors from each of 477 shards (160k) in 80 objects of ~2000 vectors |
| `--coarseIters` | 0 | coarse Lloyd iterations; 0 skips `kmeans-flat` and uses `coarse/0.f32` from `init-coarse` |
| `--fineIters` | 12 | map-reduce Lloyd iterations over all vectors (1 exhaustive + 11 incremental) |
| `--incremental` | 1 | iterations 2..N use `kmeans-inc-map`; 0 = every iteration exhaustive (g2 behaviour) |
| `--exactLast` | 0 | make the last fine iteration exhaustive |
| `--candidates` | 1 | `assign` picks top m among the neighbours of the final cell; 0 = exhaustive top m |
| `--minCount` | 20 | on fine iterations 1..fineIters-2, reseed cells with fewer members next to the densest cells (0 = off) |
| `--reduceCells` | 64 | cells per `kmeans-reduce` job (fixed chunks over all k1*k2 cells) |
| `--mapShards` / `--incShards` | 10 / 10 | shards per exhaustive `kmeans-tree-map` / incremental `kmeans-inc-map` job |
| `--neighborJobs` | 8 | `cell-neighbors` jobs per neighbour table (row ranges of k/8) |
| `--assignShards` | 2 if m <= 4, else 1 | shards per `assign` job (m=3: ~15 MB of parts per shard, m=8: ~38 MB) |
| `--assignSplit` | 1 | split each assign shard into this many parts (`{job}.{s}`) to bound the part buffer |
| `--finalizeCells` | 8 | cells per `finalize` job (one span read per part per job) |
| `--finalizeBudgetMB` | 48 | record bytes a finalize job holds at once; larger chunks are built in batches |
| `--setCurrent` | 1 | point `idx/current.json` at the new generation on publish |
| `--overwrite` | 0 | let `finalize` and `publish` rewrite the slabs and rev 0 of a generation that already has a manifest (they refuse otherwise: index objects are cached as immutable; use `publish-rev` to change defaults) |
| `--start` | sample | resume from a stage (`sample, init-coarse, coarse, init-fine, fine, pq, assign, finalize, publish`) |
| `--fineStart` | 1 | with `--start fine`: resume k-means at this iteration (needs `fine/{it-1}` and `assign/{it-1}`) |

With `k1=1` part keys are `build/{gen}/parts/0/{job}.part` and finalize jobs use
`g=0` with contiguous `cells` ranges. `finalize` results (with `scanLen`/`rerankOff`) are saved
to `data/slabs-{gen}.json` and passed to `publish`. The g2 build was
`--profile edge --m 3 --fineIters 8 --incremental 0 --candidates 0 --assignShards 2 --finalizeCells 16 --defNprobe 22 --defRerank 32`
(g2's manifest predates profiles). The g1 two-level build was
`--k1 64 --k2 64 --pc 2 --m 3 --perShard 42 --sampleGroups 10 --coarseIters 8 --fineIters 5 --minCount 0 --assignShards 4 --incremental 0 --candidates 0`.

`worker/.env.secrets` holds `BUILD_TOKEN=<random>`; every endpoint except
`/health` requires `Authorization: Bearer <token>`.
