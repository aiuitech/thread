# Index-time profiles and query defaults

Every layout knob is a build parameter recorded in the manifest (`idx/{gen}/manifest.json` = rev 0), together
with the query defaults the Worker and the SDKs apply when a request omits them. Defaults are changed with a new
revision (`POST /build/publish-rev {gen, defaults}`), never by republishing rev 0:

```json
{"gen":"g4","profile":"edge","k1":1,"k2":4096,"m":3,"format":3,"tiers":"pq192+sq8","pq":{"m":192},
 "nNeigh":64,"defaults":{"nprobe":34,"rerank":150,"pc":0,"route":"flat","beam":32}, "probeCoarse":1,"vectors":975000,
 "tree":{"key":"idx/g4/tree.88.bin","groups":128,"beam":8,"nearestRecall":0.9935,"top3Recall":0.98067,...}, "rev":103, ...}
```

## Profiles

| profile | k1 x k2 | m | format / tiers | PQ (pqM, pqIters, pqSampleObjects) | nNeigh | tree (groups, beam) | defaults (nprobe, rerank, pc) | intent |
|---|---|---:|---|---|---:|---|---|---|
| `edge` (default) | 1 x 4096 | 3 | 3 / `pq192+sq8` | 192, 12, 30 | 64 | 128, 8 | 34, 150, 0 | g2's layout (m=3) with a PQ scan tier: ~3.7x fewer scan bytes per probe; the coarser PQ scores are fixed by an int8 rerank of the top 150, and nprobe 34 (instead of g2's 26) brings recall back to 95% |
| `local` | 1 x 4096 | 1 | 3 / `pq192+sq8` | 192, 12, 30 | 64 | none | 32, 150, 0 | no duplication, smallest index, more probes |

The g3 edge profile (m=8, format 2, nprobe 10 / rerank 20) aimed at few probes because the Worker was then
assumed to be limited to 6 reads in flight. Cache API reads turned out not to be (`data/bench-latency.txt`), so
the edge profile went back to m=3 and spends the bytes on probes instead of duplication.

## What the edge profile measures (g4)

g4 (edge, `data/bench-g4.txt`, same session as a g2 reference): at nprobe 26 a query scans 4.8 MB instead of
16.5 MB and reaches 93.8% recall@10 (n=1000) / 94.1% (n=5000), about 1.1 points below g2 at the same nprobe. In
exchange it has p50 76 ms and 526 RPS at c=50 (g2: 108 ms, 363 RPS) and p50 126 ms and 1788 RPS at c=300 (g2:
192 ms, 1224 RPS). At nprobe 34 it reaches 95.1% (c=300 p50 158 ms and 1480 RPS in that session, 6.2 MB scanned),
still faster than g2's 95.4%. nprobe 34 / rerank 150 are now g4's defaults (manifest rev 1, via `publish-rev`).
Official operating point (rev 79, a later session with a faster network split over the YUL and YYZ colos): recall
94.89% (n=1000) / 95.12% (n=5000), c=50 p50 73 ms, p95 122-128 ms, ~600 RPS; c=300 p50 83-85 ms, p95 204-223 ms,
2000-2650 RPS. A same-session nprobe 26 reference: c=50 p50 66 ms, c=300 p50 78 ms. Rerank T is the recall knob
for the PQ tier: T=50/100/150/200 give 92.0/93.5/93.8/94.0% at nprobe 26, at ~1 multi-range request per winner
slab (7 to 13) and 1.6 KB per record.

The full comparison against Vectorize and the generation history are in [benchmarks.md](benchmarks.md).

## Overriding a profile

Explicit driver flags override the profile (`--m`, `--k2`, `--format`, `--tiers`, `--pqM`, `--pqIters`,
`--nNeigh`, `--tree`, `--defNprobe`, `--defRerank`, `--defPc`, `--defRoute`, `--defBeam`, ...). `--tiers` implies
the format and vice versa (`pq192+sq8` = 3, `sq4+sq8` = 2, `sq8` = 1). `--format 2` (or 1) skips the PQ stage and
defaults `rerank` to 32 unless `--defRerank` is given. Every flag: [build-flags.md](build-flags.md).

Changing the defaults of a published generation:

```sh
curl -X POST -H "authorization: Bearer $TOKEN" $URL/build/publish-rev -d '{"gen":"g4","defaults":{"nprobe":34,"rerank":150}}'
```

Query isolates pick the new revision up within 30 s (`idx/current.json` TTL).
