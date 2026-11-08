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
  edge_cold.mjs     first-query latency against a cold edge cache: one fresh query vector at a time, spaced apart
  compact_memory.mjs  (local, no Worker) measured peak memory of one SDK dense compaction against its estimate

scenarios.mjs     workload scenarios in this process over the built TS SDK (dense, filtered, lexical, hybrid, get,
                  listKeys, open, upsert) on a synthetic index or one on disk: per-span breakdown, store work,
                  kernel CPU and counters from the trace envelopes (docs/observability.md)
lib/trace.mjs     summaries of trace envelopes, shared by scenarios.mjs and query_bench.mjs

cold/             cold-start benches of both SDKs against S3 (no Worker): fake_s3.mjs proxy with a network
                  model, cold_bench.mjs / cold_bench.py, warm-up, packs, run_baseline.sh, the routing / base-pack
                  tools (make_routing, make_base_pack, routing_agreement, routing_recall, run_base.sh); see cold/README.md
                  and docs/cold-start.md

offline/          numpy experiments on local files only (g1 era; data/idx-g1, data/vectors.npy)
  diagnose_recall.py  routing vs quantization recall loss
  quant_experiment.py residual encodings for the scan tier
```

Order for a fresh setup: `prep/prep_dataset.py`, `prep/upload_shards.sh`, deploy `examples/cloudflare-worker/`, `driver.mjs`, then
any benchmark. The full commands are in [docs/operations.md](../docs/operations.md).
