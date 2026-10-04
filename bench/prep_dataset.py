"""Converts the dbpedia-openai-1M dataset into raw little-endian shards for ingest.

Outputs under data/prep/:
  ingest/XXXXX.f32   SHARD vectors each (last one shorter), f32 LE, row-major
  queries.f32        5000 x dim f32 LE (unit-normalized)
  groundtruth.u32    5000 x 10 u32 LE (ids of exact top-10 by cosine)
  manifest.json      counts/dims so the builder never guesses
"""
import json, os, sys
import numpy as np

SHARD = 2048
root = os.path.join(os.path.dirname(__file__), "..", "data")
out = os.path.join(root, "prep")
os.makedirs(os.path.join(out, "ingest"), exist_ok=True)

v = np.load(os.path.join(root, "vectors.npy"), mmap_mode="r")
n, dim = v.shape
print("vectors", n, dim, flush=True)

shards = (n + SHARD - 1) // SHARD
for s in range(shards):
    path = os.path.join(out, "ingest", f"{s:05d}.f32")
    if os.path.exists(path) and os.path.getsize(path) == min(SHARD, n - s * SHARD) * dim * 4:
        continue
    block = np.ascontiguousarray(v[s * SHARD:(s + 1) * SHARD], dtype="<f4")
    # Normalize defensively; the dataset is already unit length.
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    block = (block / norms).astype("<f4")
    block.tofile(path)
    if s % 50 == 0:
        print("shard", s, "/", shards, flush=True)

queries, gt = [], []
with open(os.path.join(root, "tests.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        q = np.asarray(d["query"], dtype="<f4")
        q /= max(np.linalg.norm(q), 1e-12)
        queries.append(q)
        gt.append(np.asarray(d["closest_ids"][:10], dtype="<u4"))
queries = np.stack(queries).astype("<f4")
gt = np.stack(gt).astype("<u4")
queries.tofile(os.path.join(out, "queries.f32"))
gt.tofile(os.path.join(out, "groundtruth.u32"))

manifest = {
    "dataset": "dbpedia-openai-1M-1536-angular",
    "vectors": int(n),
    "dim": int(dim),
    "shard": SHARD,
    "shards": int(shards),
    "queries": int(queries.shape[0]),
    "k": int(gt.shape[1]),
}
json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)
print(json.dumps(manifest))
