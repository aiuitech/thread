# thread vs Cloudflare Vectorize: benchmark summary

Dataset: dbpedia-openai-1M-1536-angular (975,000 vectors, 1536 dims, cosine, top-10 against the
dataset's exact ground truth). Vectorize numbers are Cloudflare's published figures for this dataset
(warm cache, concurrency 300 via a Worker binding). thread numbers are measured against the deployed
Worker `vib-bench` reading R2 bucket `vib-bench`, warm edge cache, client in Montreal, using
`bench/query_bench.mjs` (warm pass, then measured pass). Ranges are across sessions; network
variance between sessions was up to ~30%.

## Headline: serving configuration (g4 rev 103) vs Vectorize

| metric | Cloudflare Vectorize (published) | thread g4, concurrency 300 | thread g4, concurrency 50 |
|---|---:|---:|---:|
| recall@10 ("precision") | 95.4% | 95.1% | 94.9-95.1% |
| p50 latency | 31 ms | 83-85 ms | 70-73 ms |
| p75 latency | 56 ms | about 110 ms | about 85 ms |
| p90 latency | 159 ms | about 150 ms | about 105 ms |
| p95 latency | 380 ms | 176-223 ms | 116-128 ms |
| throughput | 343 RPS | 2000-2650 RPS | 600-655 RPS |
| bytes read per query | n/a | 6.2 MB scan + ~0.25 MB rerank | same |
| index size (975k vectors) | n/a (managed) | 5.1 GB on R2 (3x replication, two code tiers) | |
| monthly cost, 1M vectors + 1M queries | about $21 | about $1-2 (R2 ~$0.10, Workers CPU/requests ~$1) | |
| monthly cost, 1M vectors + 10M queries | about $111 | about $10 | |
| vector limit per index | 5M | bounded by centroid table in RAM (~8k cells at 1536 dims today) | |
| metadata filtering | yes | attrs field only (not wired) | |
| payload returned with hits | metadata, size-limited | arbitrary bytes co-located in the slab | |
| freshness | seconds (WAL + executors) | upsert visible within 30 s; 2000 vectors in 3.3 s, delete 0.8 s | |
| build | managed, async | 27-30 min for 975k vectors, entirely inside Workers (one deployment, 8 build slots via the Worker Loader; g4 used an 8-Worker pool) | |

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
