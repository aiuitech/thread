# Embedding models: an index for a model, filled from texts

A dense index only makes sense for one embedding model: every vector in it, and every query against it, must come
from the same model with the same instructions. Until now thread took vectors and did not care where they came from,
and a dense generation could only be born through the k-means build pipeline over a large pre-embedded corpus
([build-pipeline.md](build-pipeline.md)). This page describes the three pieces that make an index *for a model*:
the manifest names the model, the SDKs and the Worker call it for texts (never running a model themselves), and a
dense generation can be created from nothing and grown by ordinary upserts.

## `manifest.embedding`

```json
"embedding": { "provider": "workers-ai", "model": "@cf/baai/bge-base-en-v1.5", "dim": 768, "options": { "pooling": "cls" } }
"embedding": { "provider": "openai", "model": "text-embedding-3-small", "dim": 256, "options": { "dimensions": 256 } }
"embedding": { "provider": "openai", "model": "nomic-embed-text", "dim": 768, "baseUrl": "https://my-host/v1", "queryPrefix": "search_query: ", "docPrefix": "search_document: " }
"embedding": { "provider": "custom", "model": "my-model", "dim": 1024 }
```

| field | meaning |
|---|---|
| `provider` | `workers-ai` (the Workers AI binding, or its REST API outside Workers), `openai` (any OpenAI-compatible `POST {baseUrl}/embeddings`), `custom` (a function the caller supplies) |
| `model` | the model name as the provider takes it |
| `dim` | the vector size the model produces (with `options`); must equal `manifest.dim` |
| `queryPrefix`, `docPrefix` | prepended to query / document texts (e5- and nomic-style instructions); default none |
| `baseUrl` | the API base (`openai`: default `https://api.openai.com/v1`; `workers-ai` REST: Cloudflare's) |
| `options` | request fields sent verbatim with every call: `{pooling: "cls"}` for the Workers AI bge models (cls and mean embeddings are not compatible, so this belongs in the index, not the caller), `{dimensions}` for OpenAI's text-embedding-3 |
| `maxBatch` | texts per provider call (default: workers-ai 100, openai 2048, custom 256) |

The block is inherited by every revision (like `lex` and `attrs`). It is set when the generation is created
(`createGeneration({ embedding })` / `create_generation(embedding=...)` / `POST /create`); changing the model of an
existing generation means a new generation (its vectors would not be comparable). The validation, the prefixing
rule and the batching are pinned by one fixture both SDKs test against, `sdk/ts/test/fixtures/embedding.json`.

## Calling the model

Nothing in thread runs a model. The index is given a way to *call* one, and uses it in two places:

- **`search({ text })` without a `vector`** embeds the text as a `query` (with `queryPrefix`) for the dense list,
  and still runs BM25 over it when the generation has a lexical index; the lists are fused by thread-core as for any
  hybrid search. `embed: false` keeps the text lexical-only; `embed: true` fails rather than falling back when the
  index cannot embed. A `vector` given with the text is used as is.
- **`upsert({ texts })` without `vectors` or `embed`** embeds the texts as `document`s (with `docPrefix`), one per
  chunk when chunking ([chunking.md](chunking.md)). `embed: false` writes text-only documents; an explicit `embed`
  function overrides the index's.

The index gets its embedder from `open`:

```ts
const index = await thread.open({ bucket: env.BUCKET, embed: { ai: env.AI } });            // Workers AI binding
const index = await thread.open({ store, embed: { accountId, apiToken } });                 // Workers AI REST, outside Workers
const index = await thread.open({ store, embed: { apiKey: process.env.OPENAI_API_KEY } });  // OpenAI-compatible
const index = await thread.open({ store, embed: (texts, kind) => myModel(texts, kind) });   // any provider, your function
index.canEmbed;                                   // false without one of the above (the index still reads, and takes vectors)
await index.embed(["a query"], "query");          // one unit-normalized Float32Array per text, prefixed
```

```python
index = open_index(store, embed={"account_id": ..., "api_token": ...})   # Workers AI REST
index = open_index(store, embed={"api_key": ...})                        # OpenAI-compatible
index = open_index(store, embed=lambda texts, kind: model.encode(texts)) # your function
index.search(text="deploy failed", top_k=10, docs=True)                  # embedded + BM25, fused
index.upsert(keys=["docs/a.md"], texts=[text], chunk=True)               # one embedded document per chunk
```

Vectors are unit-normalized by the SDK whatever the provider returns (bge outputs are not unit length). Provider
errors surface as `ThreadError` 502 with the HTTP status; a model that returns the wrong number or size of vectors
is a 502 too. Missing credentials for the provider make `canEmbed` false rather than failing `open`.

`resolveEmbedder(config, options)` (TS) / `resolve_embedder(config, ...)` (Python) is the piece behind this, usable
on its own: it prefixes, batches by `maxBatch`, and returns rows.

## In the Worker

The bench Worker (`examples/cloudflare-worker`) has the Workers AI binding (`AI`) and an optional OpenAI-compatible
key (`EMBED_API_KEY`), and embeds in the orchestrator isolate, which is the only place that holds the bindings (a
dynamic Worker's env cannot carry them):

| request | effect |
|---|---|
| `POST /create {gen, embedding?, dim?, lex?, attrs?, profile?, defaults?, compaction?}` | a dense generation from nothing (below) |
| `POST /query {text, ...}` (JSON, no `vector`) | the text is embedded as a query, then the usual dense query |
| `POST /upsert {ids, texts, ...}` (JSON, no `vectors`) | the texts are embedded as documents, then the usual vector upsert (the Worker's own upsert is vector-only: keys, texts and metas go through the SDKs) |
| `POST /embed {gen?, texts, kind?}` | the texts' unit vectors under the generation's model, for clients that embed through the Worker |

The generation's `embedding` block is read through `/manifest` and cached for 30 s per isolate. The SDK example
Worker (`sdk/ts/examples/basic`) does the same through the SDK: `POST /search {text}` opens the index with
`embed: { ai: env.AI, apiKey: env.EMBED_API_KEY }` and calls `index.search`.

## A dense generation from nothing

`createGeneration` (TS: `thread.createGeneration(opts)` or `createGeneration(wasm, opts)`; Python:
`create_generation(store, gen, ...)`; Worker: `POST /create`) publishes rev 0 of a dense generation with **one
cell**: an empty format-2 slab (4-bit scan tier + int8 rerank tier, no PQ codebook), `fine.f32` and `coarse.f32`
of one unit row, a one-row slabs table, and a manifest with

```json
{"k1": 1, "k2": 1, "dim": 768, "m": 1, "vectors": 0, "format": 2, "tiers": "sq4+sq8", "profile": "edge",
 "defaults": {"nprobe": 32, "rerank": 150, "pc": 0}, "lex": {}, "embedding": {...},
 "compaction": {"maxDeltas": 16, "maxDeltaBytes": 33554432, "maxTombstones": 50000, "maxCellEntries": 3000, "cellsPerRun": 256, "deadlineMs": 27000},
 "deltas": [], "tombstones": [], "segments": [], "nextSeq": 1, "nextId": 0}
```

Options: `embedding` (sizes the index by its `dim`) or `dim` alone (bring your own vectors); `lex` (default `{}`,
so keys and texts work; `null` / `False` for ids-only), `attrs` (a schema), `profile` (`edge`: cells split at 3000
entries, `local`: 1000), `defaults`, `compaction` (overrides, or `null` / `False` for no automatic compaction),
`setCurrent`, `notes`. The manifest is written first with a create-only put, so two creators of one name cannot
both succeed.

From there the generation grows through the ordinary update path ([update-path.md](update-path.md)):

1. **Upserts** write delta segments (every vector's primary cell is cell 0, residuals against the seed mean) and
   lexical segments; queries scan the deltas next to the one slab. Nothing else is needed for a small index.
2. **Compaction** folds the deltas into the cell's slab: the Worker's automatic runs (after a write that passes a
   threshold: 16 segments, 32 MB, 50k tombstones), or the TypeScript SDK's `index.compact()` in the writer's own
   isolate (a Durable Object), bounded by memory, cells and time ([update-path.md](update-path.md#compaction-in-the-sdk)).
   `index.compactionDue()` checks the thresholds without I/O.
3. **Cell splitting** (same run): a rewritten cell over `maxCellEntries` is split by 2-means over its decoded
   entries, every entry re-encoded against its new mean, `fine.{rev}.f32` gets one more row and `k2` grows. So a
   generation of N vectors converges to about N / `maxCellEntries` cells, each with a data-derived mean. (When a
   delta that stays has entries in the cell, the SDK splits it with the cell's mean kept; the Worker waits.)

Readers clamp a manifest default `nprobe` to the cell count (the SDKs always did; the Worker now does for a
default, and still rejects an explicit `nprobe` above `k1 * k2`), so `nprobe: 32` is right from the first cell on.

What this path does not give, and when to care:

- **The seed mean is arbitrary** (`1/sqrt(dim)` everywhere). Residuals against it are large until the first split
  replaces it, so the 4-bit scan tier is coarse for the first `maxCellEntries` vectors; the int8 rerank of 150
  candidates over at most 3000 entries makes that moot in practice.
- **No PQ codebook** (format 2, not 3): slabs are ~4x larger per entry in the scan tier than g4's PQ-192, and a
  query reads more bytes per probed cell. Training a codebook needs residuals against stable means: a later
  `publish-rev` with `pq` after a rebuild is the place for it.
- **Means from successive 2-means splits** route worse than one global k-means over the same vectors. For an index
  that has grown large (hundreds of thousands of vectors), a rebuild stage that runs the existing pipeline in build
  slots over the generation's own int8 records (sample, k-means, PQ, assign, finalize, publish as a new generation
  with the same `embedding` block) is the planned follow-up; the SDKs need nothing new for it.
- Only the TypeScript SDK and the Worker compact and split. The Python SDK writes deltas but does not compact
  them: a generation written only from Python accumulates deltas, which still query correctly (every delta is
  scanned) but slower past a few dozen segments; compact it from TypeScript or the Worker.

## Workers AI specifics

- `@cf/baai/bge-small-en-v1.5` (384), `@cf/baai/bge-base-en-v1.5` (768), `@cf/baai/bge-large-en-v1.5` (1024),
  `@cf/baai/bge-m3` (1024): 100 texts per call; put `{"pooling": "cls"}` in `options` for the bge-en models (the
  default `mean` pooling exists for compatibility; `cls` embeddings are better and not interchangeable with them).
- The binding (`env.AI.run(model, {text: [...], ...options})` -> `{data: number[][]}`) is used inside Workers; outside,
  the REST API `POST https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}` with an API token that
  has Workers AI access (`embed: { accountId, apiToken }` / `embed={"account_id", "api_token"}`).
- Running a model inside the wasm module was considered and rejected: weights do not fit the script size limit,
  would have to come from R2 at isolate start, and compete with the routing tables for the 128 MB; the quality of a
  model small enough to try is below bge-base through the binding.

## Files

| | |
|---|---|
| `sdk/ts/src/embed.ts`, `sdk/python/thread_sdk/embed.py` | config validation, prefixing, batching, the provider embedders |
| `sdk/ts/src/create.ts`, `sdk/python/thread_sdk/write.py` (`create_generation`) | the from-zero generation |
| `sdk/ts/src/index.ts` (`open({embed})`, `search`, `embed`), `sdk/python/thread_sdk/index.py` | the index side |
| `sdk/ts/src/write.ts` (`expandInput`), `sdk/python/thread_sdk/write.py` (`_expand`) | upserts without vectors |
| `examples/cloudflare-worker/src/main.ts` | `/embed`, the embedding pre-pass of `/query` and `/upsert` |
| `examples/cloudflare-worker/src/index.ts` | `/create`, the default-nprobe clamp |
| `sdk/ts/test/fixtures/embedding.json`, `sdk/ts/test/embed.test.ts`, `sdk/python/tests/test_embed.py` | the shared fixture and the tests |
