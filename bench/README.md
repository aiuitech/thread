# bench

Scripts around the dbpedia-openai-1M dataset and the `vib-bench` deployment (`examples/cloudflare-worker/`). Grouped by what they
talk to; results go to `data/` (gitignored).

```
prep/             dataset -> data/prep/ -> R2 ingest/ (run once)
  prep_dataset.py   data/vectors.npy + tests.jsonl -> ingest shards, queries.f32, groundtruth.u32, manifest.json
  upload_shards.sh  data/prep/ingest/*.f32 -> R2 ingest/ (skips ones already present)

*.mjs             clients of the deployed Worker (THREAD_ORCH, token from examples/cloudflare-worker/.env.secrets)
  driver.mjs        builds and publishes a generation over build slots; --publish-rev changes a manifest
  compact.mjs       folds deltas and tombstones into new slab revisions (also imported by the two below)
  query_bench.mjs   latency, throughput and recall@10 against data/prep ground truth
  update_bench.mjs  upsert / delete / compaction latency and self-retrieval
  split_bench.mjs   splits the largest cells, recall and latency before and after

cold/             cold-start measurements against S3 (fake_s3.mjs proxy with a network model, or R2's S3 API;
                  read-only: new object formats are served from a local overlay, never uploaded)
  fake_s3.mjs        caching read-through S3 proxy with --overlay DIR and --rtt-ms/--mbps
  cold_bench.mjs/.py process-cold / slab-cold / warm open + query timings of either SDK (--routing, --cold-routing,
                     --cold-beam, --base-pack select the open mode)
  make_routing.mjs   offline publish-rev {routing}: fine.f16/.i8, tree-grouped tables and a manifest revision
                     for an existing generation (data/cold/g4-base -> data/cold/base-overlay)
  make_base_pack.mjs offline base packs (base.{rev}.threadpack, base.cold.{rev}.threadpack) into the overlay
  routing_agreement.mjs  probe-set agreement of every routing mode with flat f32 routing (no slab reads)
  routing_recall.mjs     end-to-end recall@10 of the routing modes over R2 (range reads, local range cache)
  run_base.sh        the cold_bench runs of docs/query-path.md "Cold opens"

offline/          numpy experiments on local files only (g1 era; data/idx-g1, data/vectors.npy)
  diagnose_recall.py  routing vs quantization recall loss
  quant_experiment.py residual encodings for the scan tier
```

Order for a fresh setup: `prep/prep_dataset.py`, `prep/upload_shards.sh`, deploy `examples/cloudflare-worker/`, `driver.mjs`, then
any benchmark. The full commands are in [docs/operations.md](../docs/operations.md).
