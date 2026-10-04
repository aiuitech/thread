# Build pipeline

How a generation is built: every stage runs inside the Worker (in build slots, one isolate each), driven from a
laptop by `bench/driver.mjs`. The driver's flags are in [build-flags.md](build-flags.md), the layout knobs in
[profiles.md](profiles.md), the commands to run it in [operations.md](operations.md).

## Stages

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

Memory per stage is bounded by the centroid table (k1*k2*dim*4 bytes, 25 MB at 4096 x 1536) plus one in-flight
shard (12.6 MB) plus the job's output buffers (k-means: a 25 MB accumulator; assign: ~2.3 KB (+ pqM bytes with a
codebook) x m per vector of parts; finalize: the SlabJobs of one cell chunk, capped by `budgetBytes`; PQ: one
12 MB residual object, a 1.5 MB codebook and a 1.8 MB partial).

With `k1=1` part keys are `build/{gen}/parts/0/{job}.part` and finalize jobs use `g=0` with contiguous `cells`
ranges. `finalize` results (with `scanLen`/`rerankOff`) are saved to `data/slabs-{gen}.json` and passed to
`publish`.

## Build slots (Worker Loader)

The driver needs many heavy jobs in parallel (k-means map and assign use 60-100 MB each), but concurrent requests
to one Worker can share a single 128 MB isolate. Until g4 that meant a pool of eight identical scripts
(vib-bench-0..7). Now there is one deployment: `examples/cloudflare-worker/src/main.ts` forwards
`POST /build/<stage>?slot=<id>` to the dynamic Worker `thread-build:<bundle hash>:slot-<id>` (`env.LOADER.get`),
which runs a second bundle of the same `examples/cloudflare-worker/src/index.ts` in its own isolate with its own
128 MB and `limits.cpuMs` 300,000. The driver keeps a free list of slot ids `0..THREAD_SLOTS-1` (default 8) and
sends one job per slot at a time; retries, bodies and auth are unchanged. R2 reaches the slots through a loopback
RPC entrypoint (`BucketRpc`; an R2 binding cannot be put in a dynamic Worker's env) and a shim that gives the
bundle an R2Bucket-shaped `env.BUCKET`; object bodies stream through the orchestrator in both directions. Writes
(`/upsert`, `/delete`) run in the `writer` slot and the automatic compaction spreads its cells over slots
`auto-0..3`. `?slot=local` runs a request in the query isolate (debugging). Design and measurements of the
isolation: [worker-loader.md](worker-loader.md); deploy notes and routing:
[examples/cloudflare-worker/README.md](../examples/cloudflare-worker/README.md).

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

The query Worker was chosen over a sibling `thread-build` Worker: one deployment instead of two, the slots' code
is guaranteed to be the version that serves queries (same build, same wasm module), and writes keep working on the
query URL. The orchestrator isolates only forward requests and pump RPC streams (no object is buffered there).

## PQ codebook training (format 3)

The scan tier encodes each entry's residual against its slab's cell mean with one index-wide product quantizer:
192 subspaces of 8 dims, 256 sub-centroids each. Training runs on residuals of the k-means sample against the
**final** k-means cells:

1. `pq-residuals` (one job per sample group, the first `pqSampleObjects` = 30 groups of the sample stage): reads
   the first `perShard` (336) vectors of each of the group's ~6 shards and the first `perShard` entries of their
   final assignment (`assign/{fineIters}/{shard}.u32`), and writes `thread.residuals_for(vectors, means, assign)`
   (~60k rows, 370 MB in total). It also reports the mean residual energy `|r|^2`.
2. `pq-init`: `thread.pq_init_codebook` over the first two residual objects (~4000 rows).
3. `pqIters` (12) Lloyd iterations: 8 `pq-train-map` jobs (`PqTrainJob`, ~4 residual objects each) write partial
   sums, one `pq-train-reduce` (`PqReducer`, partials streamed one at a time) writes the next codebook and reports
   `mean_sq_err` (the error under the codebook the iteration started from) and the empty sub-centroids, which keep
   their previous value. The driver logs `mean_sq_err` as a fraction of the residual energy.

`assign` then calls `AssignJob.set_codebook(pq/codebook/{pqIters}.bin)`, so every part record also carries its PQ
codes (part format version 2), `finalize` builds v3 slabs (`SlabJob(dim, cell, mean, 3)`), and `publish` copies
the codebook to `idx/{gen}/pq.bin` and adds `pq: {m}` to the manifest. The Worker parses the codebook once per
isolate (`PqCodebook`) and builds one `PqTable` per query.

## Incremental k-means

An exhaustive flat iteration compares 975k vectors with 4096 means (~6 TFLOP at dim 1536) and took ~7 min per
iteration in g2. After the first iteration most vectors stay in or near their cell, so iterations 2..N compare
each vector only with the `nNeigh` (64) cells closest to its previous cell (`IncKmeansJob`), 64x less arithmetic;
the map stage becomes I/O bound (reading the 6 GB of shards). The neighbour table (`cell_neighbors`: per cell, the
`n` cells with the highest cosine between means, the cell itself first) is recomputed from the new means after
every iteration by 8 `cell-neighbors` jobs of 512 rows (~1 s each) and a `concat`. Each map job writes the new
cell of every vector to `build/{gen}/assign/{it}/{shard:05}.u32` and reports how many vectors changed cell. Cells
reseeded by the reducer move next to a dense cell and appear in that cell's neighbour rows, so its vectors can
move into them on the next pass. `--exactLast` makes the last iteration exhaustive again.

`assign` then uses the same idea (`AssignJob.with_candidates`): each vector's top `m` cells are chosen among the
`nNeigh` neighbours of its final k-means cell, from a neighbour table of the final means. The per-job part buffer
is ~2.3 KB (+ 192 B of PQ codes in format 3) x m per vector: with m=3 a job takes two shards (~31 MB of parts),
with m=8 one shard (~38 MB; `--assignSplit 2` halves it further).

## Finalize

`finalize` reads each part's directory, then ONE contiguous span per part covering the job's whole cell chunk
(`part_span`), and hands each cell its slice with `SlabJob.add_records_range` (no per-cell range reads). R2 reads
are capped at 6 in flight per job (`SUBREQUEST_CONCURRENCY`, the per-invocation connection limit for R2). If the
chunk's cells hold more than `budgetBytes` (default 48 MB) of records, the Worker builds them in consecutive
batches, each with its own smaller spans.
