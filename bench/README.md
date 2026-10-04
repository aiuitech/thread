# bench

Scripts around the dbpedia-openai-1M dataset and the `vib-bench` deployment (`worker/`). Grouped by what they
talk to; results go to `data/` (gitignored).

```
prep/             dataset -> data/prep/ -> R2 ingest/ (run once)
  prep_dataset.py   data/vectors.npy + tests.jsonl -> ingest shards, queries.f32, groundtruth.u32, manifest.json
  upload_shards.sh  data/prep/ingest/*.f32 -> R2 ingest/ (skips ones already present)

*.mjs             clients of the deployed Worker (THREAD_ORCH, token from worker/.env.secrets)
  driver.mjs        builds and publishes a generation over build slots; --publish-rev changes a manifest
  compact.mjs       folds deltas and tombstones into new slab revisions (also imported by the two below)
  query_bench.mjs   latency, throughput and recall@10 against data/prep ground truth
  update_bench.mjs  upsert / delete / compaction latency and self-retrieval
  split_bench.mjs   splits the largest cells, recall and latency before and after

offline/          numpy experiments on local files only (g1 era; data/idx-g1, data/vectors.npy)
  diagnose_recall.py  routing vs quantization recall loss
  quant_experiment.py residual encodings for the scan tier
```

Order for a fresh setup: `prep/prep_dataset.py`, `prep/upload_shards.sh`, deploy `worker/`, `driver.mjs`, then
any benchmark. The full commands are in the top-level README ("Running").
