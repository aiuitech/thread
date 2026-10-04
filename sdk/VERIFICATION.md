# vib-sdk verification

Date: 2026-10-03. Index: `vib-bench` bucket, generation **g4** revision **79** (edge profile, format 3
`pq192+sq8`, k2 4096, m 3, 975 000 vectors of dim 1536, 0 delta segments, 0 tombstone sets after the last
compaction). Queries: the first n vectors of `data/prep/queries.f32`; ground truth `data/prep/groundtruth.u32`
(exact top-10). Parameters: `k=10, nprobe=34, rerank=150` on both sides.

## SDK example (R2 binding) vs the reference Worker

Example Worker `vib-sdk-example` (`sdk/examples/basic`, `POST /search`, JSON body, store `r2`) against the
main Worker `vib-bench` (`POST /query?nprobe=34&rerank=150`, octet-stream body). `node sdk/scripts/verify.mjs`
sends each query to both, after one warm-up pass so both have the slabs in their edge cache; latencies are
client-observed from a laptop, sequential, one request to each Worker at a time.

| run | identical top-10 (ids and order) | recall@10 SDK | recall@10 main | client p50 / p90 / mean ms SDK | main | server total ms SDK / main |
|---|---|---|---|---|---|---|
| n = 50 | **50 / 50** | 93.40 % | 93.40 % | 77.8 / 122.8 / 88.0 | 118.9 / 163.9 / 127.7 | 35.4 / 50.0 |
| n = 300 | **300 / 300** | **95.17 %** | 95.17 % | 67.1 / 96.2 / 74.3 | 87.7 / 132.6 / 99.7 | 27.0 / 36.9 |

- Every query returned the same ten ids in the same order as the reference Worker (no float-tie differences
  showed up at n = 300).
- Recall@10 at nprobe 34 / rerank 150 is 95.17 % over 300 queries, matching the README's 95.1 % for g4 (the
  first 50 queries alone are a harder sample: 93.4 % for both implementations).
- SDK stats per query (mean, warm): 6.22 MB scanned over 34 slab prefixes, 12.1 rerank reads for 12.0 slabs
  with winners (one multi-range Cache API request per slab), 147.2 of 150 rerank records already in memory from
  speculative reads, 46.1 cache hits / 0 misses / 0 origin reads, all Range requests honoured (206), no wasm
  resets or retries. A cold isolate's first query read the 34 slabs whole from R2 (56.8 MB, 986 ms server
  time, wasm memory 54 MB afterwards); the second query of the same vector took 31 ms with 39 cache hits.
- The example is faster end to end than the main Worker here mostly because its response is smaller and it
  skips the main Worker's debug accounting; the query path itself is the same algorithm.

Commands:

```sh
cd sdk && npm test && npm run build
cd examples/basic && npm install && cf deploy --secrets-file .env.secrets
node sdk/scripts/verify.mjs --n 50 --nprobe 34 --rerank 150
node sdk/scripts/verify.mjs --n 300 --nprobe 34 --rerank 150
```

## S3 path (`S3BlobStore`)

Not verified against the live bucket: no S3 credentials for `vib-bench` could be obtained from this
environment. The `cf` CLI's token endpoints (`cf accounts tokens create/list`, `cf user tokens list`,
`cf accounts tokens permission-groups list`) answer 403 with the configured credentials, and
`cf r2 temporary-credentials create` needs the access key id of an existing R2 API token as its parent. Deriving
S3 credentials from the CLI's own API token was not attempted. Creating an R2 API token therefore needs the
dashboard (R2 > Manage R2 API Tokens, "Object Read & Write" on `vib-bench`).

What was verified instead:

- `signV4` reproduces AWS's published Signature Version 4 examples byte for byte (GET Object with a `Range`
  header: `f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41`; GET Bucket with query
  parameters: `34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7`), see `test/s3.test.ts`.
- `S3BlobStore` against a fake S3 server over `fetch`: path-style and virtual-hosted URLs, session tokens, GET,
  range GET (206), HEAD, 404 -> null, PUT, conditional PUT (`If-None-Match: *`, `If-Match`) -> null on 412,
  ListObjectsV2 parsing, etag normalization.
- The example Worker already has the S3 path wired (`?store=s3` or `{"store":"s3"}`), configured by the
  secrets `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` (currently `-` = unset; the
  Worker answers 400 "the S3 store is not configured").

To run the S3 verification once a token exists: put `S3_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com`,
`S3_BUCKET=vib-bench`, `S3_ACCESS_KEY_ID=...`, `S3_SECRET_ACCESS_KEY=...` into
`sdk/examples/basic/.env.secrets`, redeploy (`cf deploy --secrets-file .env.secrets`), then
`node sdk/scripts/verify.mjs --n 50 --store s3` and compare with the `--store r2` run above. Expected
differences: the same ids (the store does not affect scoring); cold queries slower (TLS + SigV4 + 34 single
GETs instead of binding reads); warm queries identical (everything is served by the Cache API in both cases).
Conditional PUT behaviour of R2's S3 API (`If-Match` on PutObject) is documented by Cloudflare but was not
exercised here; until it is, use the R2 binding for writers.

## Unit and end-to-end tests (Node, `npm test`)

30 tests in 4 files, all passing (vitest 4, Node 24):

- `test/wire.test.ts`: packed hits, slab headers v1/v2/v3 (synthetic slabs), delta header, manifest parsing
  with the rev-0 fallback and `files`, current.json without `rev`, slabs table / slabRev, key layout, tombstone
  unions and the (id, rev) rule for slabs and deltas.
- `test/store.test.ts`: MemoryBlobStore conditions, multipart/byteranges parsing, CachedBlobStore (miss fills
  the whole object and returns a view into it, 206 hits, multi-range hits, 200 fallback, miss delegation,
  current.json never cached), PrefixedBlobStore.
- `test/s3.test.ts`: SigV4 vectors and the fake S3 server (above).
- `test/wasm.test.ts`: the real wasm in Node. Builds tiny v1, v2 and v3 generations with the wasm build jobs
  (`AssignJob`, `pq_init_codebook`, `SlabJob`) into a MemoryBlobStore and checks: manifest defaults and
  `params.source`, recall against brute force (>= 0.9 / 0.95 with every cell probed), rerank 0, payload
  return, queryMany, a legacy v1 generation without a slabs table and without `rev` in current.json, upsert
  (delta hit with attrs and payload, current.json CAS, a second handle seeing the change only after
  `refresh()`), replace semantics (the slab copy tombstoned), delete (delta entry hidden by a newer set),
  a pinned open, a lost manifest race retried on the new head, and recovery after a wasm reset.

## S3 path: live verification (2026-10-04)

Credentials: a dedicated R2 API token (object read/write on `vib-bench`) supplied by the owner in
`examples/basic/.env.secrets`; `S3_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com`, region `auto`.

- `node sdk/scripts/verify.mjs --n 50 --store s3 --nprobe 34 --rerank 150` (g4 rev 103): 50/50 identical
  top-10 vs the main Worker, recall@10 93.40% = main, client p50 74 ms (warm pass; all reads served from the
  per-store edge cache, which the warm pass filled through the S3 store).
- Cold query through the S3 store (fresh query vector, empty per-store cache): HTTP 200, 15 origin reads
  via SigV4 GETs to the R2 S3 endpoint (29.4 MB, whole slabs on a miss), 457 ms; the same query warm:
  0 origin reads, 49 cache hits (all 206 range responses), 93 ms. A cold query through the R2 binding for
  comparison: 26 origin reads, 49.6 MB, 842 ms. Results were identical between stores.
- Conclusion: `S3BlobStore` works against R2's S3 API for reads from a Worker; the write path (conditional
  PUT semantics) has not yet been exercised over S3.
