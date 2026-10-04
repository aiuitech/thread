# Build flags (`bench/driver.mjs`)

The defaults build the `edge` profile: a flat 4096-cell index, m=3, PQ-192 + int8 slabs (g4). Profiles:
[profiles.md](profiles.md); what the stages do: [build-pipeline.md](build-pipeline.md).

| flag | default | meaning |
|---|---:|---|
| `--profile` | edge | `edge` or `local`: sets k1, k2, m, format, tiers, PQ settings, nNeigh and the query defaults |
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

## Historical builds

The g2 build was `--profile edge --m 3 --fineIters 8 --incremental 0 --candidates 0 --assignShards 2
--finalizeCells 16 --defNprobe 22 --defRerank 32` (g2's manifest predates profiles). The g1 two-level build was
`--k1 64 --k2 64 --pc 2 --m 3 --perShard 42 --sampleGroups 10 --coarseIters 8 --fineIters 5 --minCount 0
--assignShards 4 --incremental 0 --candidates 0`.
