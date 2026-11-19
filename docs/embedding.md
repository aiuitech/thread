# Embedding models

thread does not run an embedding model. You tell an index which model it is for, and how to call that model. The
index then embeds your documents when you write and your query when you search.

```ts
// Create: name the model
await thread.createGeneration({
  store, gen: "docs",
  embedding: { provider: "openai", model: "text-embedding-3-small", dim: 1536 },
  lex: { text: true },
});

// Open: say how to call it
const index = await thread.open({ store, embed: { apiKey: process.env.OPENAI_API_KEY } });

await index.upsert({ keys: ["a.md"], texts: [text] });        // the text is embedded
await index.search({ text: "deploy failed", topK: 10 });      // the query is embedded
```

## Why the model belongs to the index

Every vector in an index, and every query, must come from the same model with the same settings. Vectors from two
models cannot be compared. So the model is written in the index itself, and every reader and writer uses it. A
wrong model in one service cannot damage your results without anyone noticing.

Your credentials are not in the index. You give them when you open it.

## Choose a provider

| Provider | Create with | Open with |
|---|---|---|
| **Workers AI**, in a Worker | `provider: "workers-ai"` | `embed: { ai: env.AI }` |
| **Workers AI**, from anywhere | `provider: "workers-ai"` | `embed: { accountId, apiToken }` |
| **OpenAI**, or any OpenAI-compatible API | `provider: "openai"`, with `baseUrl` for other hosts | `embed: { apiKey }` |
| **Your own function** | `provider: "custom"` | `embed: (texts, kind) => vectors` |

Examples of the `embedding` block:

```ts
// Workers AI. Set pooling to "cls" for the bge models.
{ provider: "workers-ai", model: "@cf/baai/bge-base-en-v1.5", dim: 768, options: { pooling: "cls" } }

// OpenAI, with shorter vectors
{ provider: "openai", model: "text-embedding-3-small", dim: 256, options: { dimensions: 256 } }

// A self-hosted OpenAI-compatible server, with the prefixes that the model expects
{ provider: "openai", model: "nomic-embed-text", dim: 768, baseUrl: "https://my-host/v1",
  queryPrefix: "search_query: ", docPrefix: "search_document: " }

// Your own function
{ provider: "custom", model: "my-model", dim: 1024 }
```

In Python:

```python
create_generation(store, "docs", embedding={"provider": "openai", "model": "text-embedding-3-small", "dim": 1536})

index = open_index(store, embed={"api_key": os.environ["OPENAI_API_KEY"]})          # OpenAI-compatible
index = open_index(store, embed={"account_id": account, "api_token": token})        # Workers AI
index = open_index(store, embed=lambda texts, kind: model.encode(texts))            # your own function
```

A custom function receives the texts and a `kind`, which is `"query"` or `"document"`, and returns one vector for
each text. thread normalizes the vectors.

## Bring your own vectors

You can always pass vectors yourself. Given vectors are stored as they are, and the model is not called.

```ts
await index.upsert({ keys, vectors, texts });           // your vectors, and the text for keyword search
await index.search({ vector: queryVector, text: "deploy failed" });
```

An index created with `dim` and no `embedding` block takes vectors only.

## Control when the model is called

| Call | Effect |
|---|---|
| `search({ text })` | The query is embedded. Keyword search also runs if the index has text. |
| `search({ text, embed: false })` | Keyword search only. The model is not called. |
| `upsert({ texts })` | Each text is embedded |
| `upsert({ texts, embed: false })` | Text-only documents. No vectors. |
| `upsert({ texts, embed: myFunction })` | Your function is used for this call |
| `index.canEmbed` | `true` when the index names a model and `open` got a way to call it |
| `await index.embed(["a query"], "query")` | The vectors, if you want them yourself |

If the provider fails, the call fails with a `ThreadError` of status 502. Nothing is written.

## Change the model

A change of model, or of the vector size, means a new index. Build the new generation beside the live one, fill
it, then switch: [Change the model, or rebuild](create-an-index.md#change-the-model-or-rebuild).

## Under the hood

The rest of this page is the detail: every field of the `embedding` block, what the reference Worker does, how a
generation that starts empty grows, and the bulk build.

### `manifest.embedding`

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

### Calling the model

Nothing in thread runs a model. The index is given a way to *call* one, and uses it in two places:

- **`search({ text })` without a `vector`** embeds the text as a `query` (with `queryPrefix`) for the dense list,
  and still runs BM25 over it when the generation has a lexical index; the lists are fused by thread-core as for any
  hybrid search. `embed: false` keeps the text lexical-only; `embed: true` fails rather than falling back when the
  index cannot embed. A `vector` given with the text is used as is.
- **`upsert({ texts })` without `vectors` or `embed`** embeds the texts as `document`s (with `docPrefix`), one per
  chunk when chunking ([chunking.md](chunking.md)). `embed: false` writes text-only documents; an explicit `embed`
  function overrides the index's. Given `vectors` are never embedded over, and a `null` row among them writes that
  document text-only (`vectors: [embA, null]`).

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

### In the Worker

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

### A dense generation from nothing

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
`setCurrent`, `notes`, and in TypeScript `means` (`k x dim` floats: rev 0 then has `k` empty cells with those means
instead of one; the bulk build below passes its k-means result). The manifest is written first with a create-only
put, so two creators of one name cannot both succeed.

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

To add a model to a live index, build the dense generation beside it and swap it in: `createGeneration({ setCurrent:
false })`, fill it through `open({ gen })`, then `setCurrent(store, { gen, expect })` (a compare-and-swap of
`idx/current.json`) and later `dropGeneration` of the old one ([update-path.md](update-path.md#switching-generations)).

Readers clamp a manifest default `nprobe` to the cell count (the SDKs always did; the Worker now does for a
default, and still rejects an explicit `nprobe` above `k1 * k2`), so `nprobe: 32` is right from the first cell on.

What this path does not give, and when to care:

- **The seed mean is arbitrary** (`1/sqrt(dim)` everywhere). Residuals against it are large until the first split
  replaces it, so the 4-bit scan tier is coarse for the first `maxCellEntries` vectors; the int8 rerank of 150
  candidates over at most 3000 entries makes that moot in practice.
- **No PQ codebook** (format 2, not 3): slabs are ~4x larger per entry in the scan tier than g4's PQ-192, and a
  query reads more bytes per probed cell. Training a codebook needs residuals against stable means: a later
  `publish-rev` with `pq` after a rebuild is the place for it.
- **Means from successive 2-means splits** route worse than one global k-means over the same vectors. A
  generation loaded in bulk should be built with [the bulk build](#a-bulk-build-in-bounded-steps) instead. For an
  index that has grown past what one isolate clusters (millions of vectors), a rebuild stage that runs the existing
  pipeline in build slots over the generation's own int8 records (sample, k-means, PQ, assign, finalize, publish as
  a new generation with the same `embedding` block) is still the planned follow-up.
- Only the TypeScript SDK and the Worker compact and split. The Python SDK writes deltas but does not compact
  them: a generation written only from Python accumulates deltas, which still query correctly (every delta is
  scanned) but slower past a few dozen segments; compact it from TypeScript or the Worker.

### A bulk build in bounded steps

Loading tens of thousands of vectors into a one-cell generation through upserts makes every vector pass through
cell 0 and a chain of 2-means splits. `thread.bulkBuild(opts)` (TypeScript, `sdk/ts/src/build.ts`) builds the
generation with a layout from k-means instead, in steps small enough for a Durable Object alarm:

```ts
const b = await thread.bulkBuild({ store, gen: "e2", embedding, lex, attrs, maxBytes: 32 << 20, owner: state.id.toString() });
// while copying: stage what you would have upserted (vectors may be null: text-only documents)
await b.add({ keys, metas, texts, attrs, vectors });       // any number of calls, at most ~800 vectors each at dim 1024
await b.seal();                                            // no more adds; k, the cell size and the sample size are fixed
// one per alarm:
const r = await b.step({ deadlineMs: 10_000 });            // { phase, done, worked, progressed, estimatedPeakBytes, progress, ms }
if (r.done) { await setCurrent(store, { gen: "e2", expect }); await b.cleanup(); }
// dropping the handle without finishing (a failed step, an evicted object): b.close(); reopen with thread.bulkBuild
```

`thread.bulkBuild` resumes the build in `build/{gen}/state.json` when there is one (the options then only matter
for a new build), so a new handle after a restart continues where the last step left. The phases:

| phase | what a step does | holds at most |
|---|---|---|
| `stage` (`add`) | one JSON object (keys, metas, texts, attrs) and one f32 object per call under `build/{gen}/stage/`; the generation does not exist yet | the call's documents |
| `sample` | reads the f32 objects in order, one at a time, and keeps `S` evenly spaced rows (`S` = 256 per cell, at most a quarter of `maxBytes` of vectors); a chunk per step under `build/{gen}/sample/` | `S x dim x 4` + one f32 object x2 |
| `kmeans` | Lloyd iterations over the sample in wasm (`FlatKmeansJob`, `Reducer.means_reseed` for empty cells), initialised with `k` sample rows from a seeded shuffle; the means saved after each iteration (default 10) | the sample + 6 copies of the means |
| `create` | `createGeneration({ means, setCurrent: false })`: rev 0 with `k` empty cells | the means x3 |
| `load` | replays each stage in order as one ordinary upsert (texts and vectors, a null row for a document staged without one; the vectors go to deltas of their nearest cell), then `compact({ maxBytes })` when it is due | a stage x5, then the compaction's `maxBytes` |
| `compact` | `compact({ force: true, maxBytes })` until no delta is left | `maxBytes` |

Sizes from `maxBytes` (`bulkBuildPlan(n, dim, { maxBytes })` computes them): cells of half what the budget splits at
(`compactionLimits(...).cellEntries / 2`, 481 entries at dim 1024 and 32 MiB), so `k = ceil(n / 481)` and every cell
can grow to twice its size before it splits; an `add` may carry at most `upsertVectors` vectors (810 at dim 1024
and 32 MiB) and an eighth of `maxBytes`, so every replay folds within the budget (413 otherwise: split the add). A
step keeps its estimate under `maxBytes` (`r.estimatedPeakBytes`) and stops starting work at `deadlineMs` (a
replay is not started within 4 s of it, or 40% of `deadlineMs` when that is less, which the compaction after it
uses).

A step also ends as soon as a pass over its phase commits nothing, and reports `progressed: false` when its
first pass did not: the index opened too close to the deadline to start a replay, another run holds
`compact.lock`, or the deadline had passed. It never runs a phase again without progress, so it does not depend on
the clock moving: in Workers `Date.now()` stands still between I/O, and in 0.2.3 a `load` step whose index open
left less than the reserve called the phase again and again until the CPU limit (inseam, 2026-11-08). The next
step continues. Keep the handle between steps where you can, since a new handle opens the index again; if that
open alone takes more than `deadlineMs` minus the reserve on every call, raise `deadlineMs`.

`b.close()` releases the index the handle opened (its caches and wasm objects) and writes or deletes nothing: the
staging objects, the state and the generation stay, and `thread.bulkBuild` on the same `gen` resumes the build. A
later `step` on the same handle opens the index again. Close a handle you drop before `done`, for example after a
failed step, whose state's ETag may be stale (open the build again). `cleanup()` closes it too.

Each piece commits its own progress through a compare-and-swap of the state, so a step that crashed is run again:
a sample chunk or a means iteration is written again; a replayed stage is an upsert under the same keys (or ids)
and replaces what the lost attempt wrote; the create phase recognises its own generation. Two steps at once: the
second fails with 409. Deletes and updates during a build are later `add`s of the same keys (an update) or go to
the generation after `done`.

Measured in Node against an in-memory store (`sdk/ts/test/build.test.ts`): 20,000 documents of dim 1024 (2%
text-only, 19,600 vectors), 32 MiB: k-means made 41 cells, 45 after the compactions' splits (7 to 640 entries,
none past what the budget splits); 5 steps (three cut at 0.3 s, one of 5 s that replayed 68 stages and kept the
deltas at the trigger, one of 0.4 s), 7.3 s in all; the largest step estimate 30.5 MB, the wasm memory grew by
6.7 MB. Against R2 every read and write costs a round trip, so a Durable Object will need more steps; that was not
measured.

What does not fit one isolate: the means are held about six times during k-means and three times in a split
commit, so one isolate of 32 MiB clusters at most about 1000 cells at dim 1024 (`bulkBuildPlan(...).maxCells`,
about 480,000 vectors at 481 per cell); a larger build fails at `seal` with a 413. Those belong in the Worker's
build slots (`bench/driver.mjs`: sample, k-means, PQ, assign and finalize spread over isolates), or on the
caller's device, as inseam already runs large lexical merges (`commitMergedSegment`): the device runs the same SDK
calls with a larger `maxBytes` against the same store.

### Workers AI specifics

- `@cf/baai/bge-small-en-v1.5` (384), `@cf/baai/bge-base-en-v1.5` (768), `@cf/baai/bge-large-en-v1.5` (1024),
  `@cf/baai/bge-m3` (1024): 100 texts per call; put `{"pooling": "cls"}` in `options` for the bge-en models (the
  default `mean` pooling exists for compatibility; `cls` embeddings are better and not interchangeable with them).
- The binding (`env.AI.run(model, {text: [...], ...options})` -> `{data: number[][]}`) is used inside Workers; outside,
  the REST API `POST https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}` with an API token that
  has Workers AI access (`embed: { accountId, apiToken }` / `embed={"account_id", "api_token"}`).
- Running a model inside the wasm module was considered and rejected: weights do not fit the script size limit,
  would have to come from R2 at isolate start, and compete with the routing tables for the 128 MB; the quality of a
  model small enough to try is below bge-base through the binding.

### Files

| | |
|---|---|
| `sdk/ts/src/embed.ts`, `sdk/python/thread_sdk/embed.py` | config validation, prefixing, batching, the provider embedders |
| `sdk/ts/src/create.ts`, `sdk/python/thread_sdk/write.py` (`create_generation`) | the from-zero generation |
| `sdk/ts/src/index.ts` (`open({embed})`, `search`, `embed`), `sdk/python/thread_sdk/index.py` | the index side |
| `sdk/ts/src/write.ts` (`expandInput`), `sdk/python/thread_sdk/write.py` (`_expand`) | upserts without vectors |
| `examples/cloudflare-worker/src/main.ts` | `/embed`, the embedding pre-pass of `/query` and `/upsert` |
| `examples/cloudflare-worker/src/index.ts` | `/create`, the default-nprobe clamp |
| `sdk/ts/test/fixtures/embedding.json`, `sdk/ts/test/embed.test.ts`, `sdk/python/tests/test_embed.py` | the shared fixture and the tests |
