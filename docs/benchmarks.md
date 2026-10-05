# thread vs Cloudflare Vectorize: benchmark summary

Dataset: dbpedia-openai-1M-1536-angular (975,000 vectors, 1536 dims, cosine, top-10 against the
dataset's exact ground truth). Vectorize numbers are Cloudflare's published figures for this dataset
(warm cache, concurrency 300 via a Worker binding; [source](https://blog.cloudflare.com/workers-ai-bigger-better-faster/#how-fast-is-vectorize),
checked 2026-10-05, unchanged). thread numbers are measured against the deployed Worker `vib-bench` reading R2
bucket `vib-bench`, warm edge cache, client near Montreal, using `bench/query_bench.mjs` (warm pass, then
measured pass over the same queries, manifest defaults nprobe 34 / rerank 150). Concurrency 300 runs use
`--n 5000`, concurrency 50 runs `--n 1000` (as for rev 103; see [why](#concurrency-300-with---n-1000)). Ranges are
across separate runs; network variance between sessions was up to ~30%.

## Headline: serving configuration vs Vectorize

The 2026-10-05 columns are the deployment described in [Measured 2026-10-05](#measured-2026-10-05); the rev 103
columns are the previous figures (2026-10-04, same generation before the update benchmarks), kept for comparison.

| metric | Cloudflare Vectorize (published) | thread g4 rev 179, c300, measured 2026-10-05 | thread g4 rev 179, c50, measured 2026-10-05 | thread g4 rev 103, c300 | thread g4 rev 103, c50 |
|---|---:|---:|---:|---:|---:|
| recall@10 ("precision") | 95.4% | 95.13% | 94.91% | 95.1% | 94.9-95.1% |
| p50 latency | 31 ms | 88-90 ms | 69-78 ms | 83-85 ms | 70-73 ms |
| p75 latency | 56 ms | 112-115 ms | 88-94 ms | about 110 ms | about 85 ms |
| p90 latency | 159 ms | 147-154 ms | 106-112 ms | about 150 ms | about 105 ms |
| p95 latency | 380 ms | 176-193 ms | 120-133 ms | 176-223 ms | 116-128 ms |
| throughput | 343 RPS | 2495-2785 RPS | 578-608 RPS | 2000-2650 RPS | 600-655 RPS |
| bytes read per query | n/a | 6.2 MB scan + ~0.24 MB rerank | same | same | same |
| index size (975k vectors) | n/a (managed) | 5.1 GB on R2 (3x replication, two code tiers) | | | |
| monthly cost, 1M vectors + 1M queries | about $31 | about $0.10, warm cache ([cost](#cost)) | | | |
| monthly cost, 1M vectors + 10M queries | about $169 | about $6, warm cache | | | |
| vector limit per index | 5M | bounded by centroid table in RAM (~8k cells at 1536 dims today) | | | |
| metadata filtering | yes | fields packed into a 32-bit word, applied inside the scan; selective filters widen the probes (docs/filtering.md) | | | |
| payload returned with hits | metadata, size-limited | arbitrary bytes co-located in the slab | | | |
| freshness | seconds (WAL + executors) | upsert visible within 30 s; 2000 vectors in 3.3 s, delete 0.8 s | | | |
| build | managed, async | 27-30 min for 975k vectors, entirely inside Workers (one deployment, 8 build slots via the Worker Loader; g4 used an 8-Worker pool) | | | |

Against rev 103: at concurrency 300, p50 is about 5 ms slower (88-90 against 83-85 ms), p75 and p90 are about the
same, and p95 and throughput are the same or better. At concurrency 50 the runs spread wider and are a few ms
worse at most percentiles (p50 69-78 against 70-73, p95 120-133 against 116-128), with 578-608 RPS against
600-655. Server time per query is unchanged at concurrency 300 (35-36 ms, 27-28 ms of it waiting for scan
prefixes) and 2-3 ms higher at concurrency 50 (28.3-28.6 ms against 25-27 ms, all of it in the Cache API reads).
The rest is most likely the network: every query on 2026-10-05 ran in Toronto (YYZ), where the rev 103 sessions
ran mostly in Montreal (YUL), next to the client. Recall did not change: rev 179 gives 95.13% over all 5000
queries (95.12% at rev 103) and 94.91% over the first 1000 (94.89% at rev 103).

## Measured 2026-10-05

**Deployment.** Worker `vib-bench`, version `09c46b6d-e20e-4925-909f-33ae3ca5ccf8`, deployed 2026-10-05 01:14:33
UTC by `cf deploy` from main `610501f` (the attribute-filtering and cold-start merges; the query responses carry
`stats.filtered`, `filterDropped`, `candidates` and `maxNprobe`, which the previous version did not). Workers Logs
attribute every request of the runs below to that version. It served g4 rev 179 throughout: `/index/status` showed 0
delta segments, 0 tombstones, k = 4103 cells (rev 103 had 4096; the split benchmark added 7), 975,000 vectors, no
compaction thresholds exceeded and no compaction running, and `idx/current.json` read `{"gen":"g4","rev":179}`
(written 2026-10-04 02:34 UTC by the previous version, so without the inline manifest) before every run. Rev 179 is
the clean head left by the split, update and compaction benchmarks: every vector they inserted was deleted and
purged (`change.op: compact`, 2305 slabs rewritten), so it holds the same 975,000 vectors as rev 103.

**Client.** A Linux workstation near Montreal (12-15 ms TCP round trip to the edge), Node 24 `fetch`, load average
0.3-4.3 during the runs from other work on the machine (20 cores). Every
request reached the Worker in YYZ (Toronto).

**Commands** (three rounds about four minutes apart, 01:16, 01:20 and 01:24 UTC, each round c300 n1000, c50 n1000,
c300 n5000; outputs in `data/bench-g4-2026-10-05.new.*`):

```sh
node bench/query_bench.mjs --gen g4 --n 5000 --concurrency 300
node bench/query_bench.mjs --gen g4 --n 1000 --concurrency 50
node bench/query_bench.mjs --gen g4 --n 1000 --concurrency 300
node bench/query_bench.mjs --gen g4 --n 2000 --concurrency 300 --extras "route=flat|route=beam&beam=32"
node bench/query_bench.mjs --gen g4 --n 1000 --concurrency 50 --filter '{"attrs":{"$lt":1}}' --maxNprobe 68
node bench/edge_cold.mjs --gen g4 --queries 1500,1800,2100,2400,2700,3000,3300,3600,3900,4200,4500,4800 --gap 20
```

**Runs** (measured passes; client latency in ms; server = mean `timing.total`; every pass 0 errors, 100% edge-cache
hits, 6.2 MB scanned and ~244 KB reranked per query):

| run | recall@10 | p50 | p75 | p90 | p95 | p99 | RPS | server ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c300 n5000 #1 | 95.13% | 90.0 | 114.9 | 153.7 | 192.6 | 374.3 | 2495 | 35.7 |
| c300 n5000 #2 | 95.13% | 89.1 | 112.0 | 150.3 | 180.9 | 370.6 | 2530 | 35.2 |
| c300 n5000 #3 | 95.13% | 87.9 | 112.9 | 147.0 | 175.5 | 354.0 | 2785 | 35.9 |
| c50 n1000 #1 | 94.91% | 69.1 | 87.6 | 112.2 | 132.8 | 182.1 | 597 | 28.6 |
| c50 n1000 #2 | 94.91% | 77.9 | 93.7 | 109.0 | 122.4 | 187.9 | 578 | 28.3 |
| c50 n1000 #3 | 94.91% | 75.0 | 90.9 | 106.1 | 119.8 | 152.1 | 608 | 28.4 |
| c300 n1000 #1 | 94.91% | 102.6 | 144.9 | 258.1 | 402.8 | 562.1 | 1186 | 41.7 |
| c300 n1000 #2 | 94.91% | 98.8 | 143.2 | 352.9 | 611.3 | 852.4 | 930 | 67.4 |
| c300 n1000 #3 | 94.91% | 97.4 | 141.0 | 217.1 | 369.3 | 436.9 | 1274 | 37.9 |

One more c300 n5000 run started 8 s after the deploy (01:14:41, already all on the new version): 95.13%, p50 87.0,
p95 182.6, 2385 RPS; its warm pass, the first load on new isolates, had p95 1718 ms and p99 5864 ms
(`data/bench-g4-2026-10-05.new-first-n5000.jsonl`; its text output is at the end of the `prev-c382c9c0` file).

### Concurrency 300 with `--n 1000`

At concurrency 300 a 1000-query measured pass lasts under a second, so it is dominated by the Worker scaling out:
each burst after a few idle minutes starts new isolates that load the routing table and codebook before their
first query (the warm passes before these runs averaged 80-120 ms of `timing.load`, against under 1 ms once warm),
and some of that is still under way in the measured pass. That gives 930-1274 RPS and a p95 of 369-611 ms. The
rev 103 concurrency-300 figures were all `--n 5000` runs, which is why the headline uses n 5000; c300 n1000 is
reported here so it is not mistaken for the steady state.

### Routing and filters

Interleaved flat vs beam routing (c300, n 2000, 1000 queries per config, same queries): flat 94.91%, p50 90.0,
p95 238 ms; beam 32 94.92%, p50 88.5, p95 335 ms; 2377 RPS for the pass. The same as at rev 103: no measurable
difference at this sample size, so flat stays the default.

Filtered queries, c50 n1000, `{"attrs":{"$lt":1}}` with `--maxNprobe 68` (g4's attribute words are all 0, so the
filter keeps every entry and measures the cost of evaluating it; 34 cells scanned, nothing widened, 0 entries
dropped): 94.91%, p50 73.8, p75 89.2, p90 103.1, p95 111.4 ms, 622 RPS, server 25.2 ms. Within the run-to-run
spread of the unfiltered c50 runs. A selective filter would widen its probes and read up to 2x the bytes at this
bound; g4 has no selective attributes to measure that with.

### Cold edge cache

| measurement | client ms p50 (range) | server ms p50 | slab cache misses p50 | bytes read p50 | cold isolates |
|---|---:|---:|---:|---:|---:|
| first query of a fresh query vector, edge cache partly cold (12 queries, 20 s apart) | 826 (381-1325) | 668 | 17.5 of 34 | 29 MB | 2 of 12 |

Measured with `bench/edge_cold.mjs` at 00:59-01:03 UTC, before the warm runs, on the previous version
(`c382c9c0`, below). A miss read the whole slab from R2 (median slab 1.15 MB) and cached it, as
[query-path.md](query-path.md#the-two-phase-query) describes for the current code: phase 2 took 0 ms whenever a
query had misses, since the winners' records came with the whole slabs. After about a day without traffic about half of
each query's slabs had left the YYZ cache. The two queries that also landed on a cold isolate took 1124 and 1325 ms
(the isolate's index load was 70-90 ms of that; the R2 reads are the rest). Recall over these 12 queries was 90%.
This is not comparable to Vectorize's warm-cache figures and is not in the headline. It could not be repeated on
the new version: the warm runs had by then pulled every slab into the YYZ cache.

### Previous deployment (version `c382c9c0`, 2026-10-04 02:42 UTC)

The runs first taken on 2026-10-05 (00:59-01:14 UTC) hit the version deployed on 2026-10-04, from before this
repository's history (it ignores `filter=` and returns no filter stats), serving the same rev 179. c50 n1000, three
runs: 94.91%, p50 74.2-80.0, p95 119.6-137.7 ms, 532-585 RPS. c300 n1000: 94.86-94.91%, p50 97.0-102.1, p95
416-428 ms and 1061-1091 RPS in the two later runs; the first, the first concurrency-300 burst of the day, had 34
client errors (`fetch failed`), p95 1926 ms and 398 RPS. Routing comparison (c300 n2000): flat p50 92.2 / p95 307,
beam 32 p50 88.3 / p95 211, 2192 RPS. Outputs: `data/bench-g4-2026-10-05.prev-c382c9c0.*`.

### Cost

List prices checked 2026-10-05: [Workers Standard](https://developers.cloudflare.com/workers/platform/pricing/)
($5 a month, 10M requests and 30M CPU ms included, then $0.30 per million requests and $0.02 per million CPU ms;
subrequests, such as the Worker's Cache API reads, are not billed),
[R2](https://developers.cloudflare.com/r2/pricing/) ($0.015 per GB-month, Class B reads $0.36 per million,
10 GB-month and 10M Class B reads free) and
[Vectorize](https://developers.cloudflare.com/vectorize/platform/pricing/) (`(queried + stored vectors) x dimensions`
at $0.01 per million, plus stored dimensions at $0.05 per 100 million; 50M queried and 10M stored dimensions
included). Both need Workers Paid, so the $5 subscription is left out of both columns.

- **thread CPU per query.** Workers Logs report a mean `cpuTimeMs` of 31-35 ms per warm query (median 27-33) in
  the c50 and c300 n5000 passes, filtered or not, and up to 40 ms in the short c300 n1000 passes. 1M queries = 33M CPU ms, 3M over the allowance: about $0.06. 10M
  queries = 330M CPU ms: about $6.00; 10M requests are within the allowance.
- **R2.** 5.1 GB fits the 10 GB free tier (else $0.08 a month). With a warm edge cache a query reads nothing from
  R2. Each edge-cache miss is one Class B read, $0.36 per million beyond the first 10M: at the cold-cache sample's
  17 misses per query, 1M queries a month would add about $2.50 if every query were that cold, 10M about $58.
- **Vectorize**, 975,000 stored 1536-dim vectors: 1M queries = 3.03 billion queried dimensions, about $30.6;
  10M queries = 16.9 billion, about $168.8. The figures previously in this table ($21 and $111) do not follow
  from the published formula at 1536 dimensions.

So at 1M queries a month thread costs about $0.10 (up to ~$2.60 if every query found the cache that cold) against
about $31 for Vectorize, and at 10M queries about $6 (up to ~$64) against about $169. The edge-cache hit rate a
real workload sees is the unknown here; the benchmark's is 100% after the warm pass. This counts list prices only and assumes the account
has no other Workers usage drawing on the allowances.

## Generation history (95% recall operating point, concurrency 300)

| gen | layout | nprobe for 95% | p50 ms | p95 ms | RPS | scan MB/query | build time | index GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| g1 | 64x64 tree k-means, int8, m=3, flat routing | 22 | 389 | 827 | 560 | 41 | 22 min | 4.6 |
| g2 | flat 4096, 4-bit scan + int8 rerank 20-32, m=3 | 26 | 132-192 | 330-453 | 1263-1766 | 16.5 | 85 min | 6.8 |
| g3 | g2 layout with m=8, incremental k-means | 16 | 273 | 670 | 780 | 32 | 42 min | 18.2 |
| g4 | flat 4096, PQ-192 scan + int8 rerank 150, m=3 | 34 | 83-158 | 176-354 | 1480-2650 | 6.2 | 30 min | 5.1 |

g1 with its original tree routing (pc=2) capped at 91% recall regardless of nprobe; the 389 ms row
is g1 with flat routing. g2's spread reflects Worker-side improvements made while it was serving
(edge cache, multi-range rerank, speculative reads) and session variance.

## Query-path contributions (g2, concurrency 50, same session)

| change | p50 ms | p95 ms | RPS |
|---|---:|---:|---:|
| direct R2 reads, no cache | 140 | 372 | 241 |
| + Cache API (edge) in front of R2 | 72 | 187 | 480 |
| + flat routing (recall 82.6% -> 94.6% at nprobe 22; bytes up) | 203 | 431 | 195 |
| + multi-range rerank, speculative reads, fewer copies, 12 in flight | 97.5 | 225 | 391 |

## Updates (g4, writer on a pool Worker (now the `writer` build slot), same session A/B)

| operation | without centroid tree | with centroid tree (128 groups, beam 8) |
|---|---:|---:|
| upsert 100 vectors | 1.42 s | 1.16 s |
| upsert 500 vectors | 2.49 s | 1.43 s |
| upsert 2000 vectors | 7.48 s | 3.27 s |
| delete 1000-5000 ids | 0.8-0.9 s | same |
| query p50 overhead, 5 delta segments (3 MB) | +2.4 ms | same |
| query p50 overhead, 20 delta segments (12 MB, 5k tombstones) | +5.6 ms | same |
| compaction of 5000 inserted vectors (2151 cells, 3 GB rewritten) | 43-51 s | same |
| recall drift through inserts, deletes, compaction | within 0.02 points | same |

## Automatic compaction and cell splitting (g4, concurrency 50 background load)

| measurement | value |
|---|---:|
| trigger fired after | 17 upserts of 250 vectors (maxDeltas 16) |
| trigger to new revision | 18.8 s (228 cells folded in 16.5 s) |
| queries failed during compaction | 0 of 9,555 |
| query p50: clean / 17 deltas / after auto-compaction | 76.8 / 81.9 / 79.3 ms |
| recall after the cycle | 95.11% (baseline 95.12%) |
| split of the 7 largest cells (cap 2400) | k 4096 -> 4103, largest cell 3109 -> 2257 entries |
| recall before / after split | 95.12% / 95.13% |
| single-Worker compaction throughput | 11-14 cells/s (one 250-vector delta per 27 s run) |

## Offline diagnostics that shaped the design

| question | answer |
|---|---:|
| int8 residual quantization loss (exact top-50 candidates) | 99.4-99.5% recall@10 vs exact |
| 4-bit alone / 4-bit + int8 rerank of top 30 | 90.4% / 94.2% (= int8) end to end at nprobe 22 |
| PQ-192 alone / + int8 rerank of top 200 | 65.6% / 94.0% end to end at nprobe 22 |
| routing recall ceiling with 2 coarse probes (m=3) | ~91% at any nprobe |
| entries scanned for 95% recall, any layout tried | 20k-28k |
| centroid tree 128 groups, beam 8: exact nearest cell found | 99.35% at ~11% of the exhaustive cost |

SDK: `thread-sdk` returns identical top-10 to the Worker (300/300 via the R2 binding, 50/50 via the S3-compatible store against R2's S3 endpoint); cold S3-store query 457 ms with 15 signed origin reads, warm 93 ms.

Sources: `data/bench-*.txt`, `data/diagnosis-g1.md`, `data/quant-experiment.md`,
`sdk/ts/VERIFICATION.md`.
