# Create an index

You create an index once. After that you open it, write to it and search it. This page helps you choose the kind
of index and shows how to create it.

## Choose the kind of index

| Your situation | Create | Search with |
|---|---|---|
| You have text and want semantic search. You want thread to call the embedding model. | [An index for a model](#an-index-for-an-embedding-model) | `search({ text })` |
| You have text and want keyword search only. No model, no vectors. | [A text index](#a-text-index) | `search({ text })` |
| You already have vectors from your own pipeline. | [An index for your own vectors](#an-index-for-your-own-vectors) | `query(vector)` or `search({ vector, text })` |
| You have records with a unique name and a few attributes, and want lookups by name. | [A text index](#a-text-index), used as a [key-value index](keys.md#the-key-value-case) | `get([key])` |
| You have hundreds of thousands of vectors or more to load at once. | [A bulk build](#load-a-lot-of-data) | Any of the above |

One index (one **generation**) is for one embedding model. Vectors from two models cannot be compared, so a
change of model means a new generation. [Change the model](#change-the-model-or-rebuild) shows how to do that
without downtime.

## An index for an embedding model

This is the usual choice. You name the model when you create the index. From then on, the index embeds your
documents and your queries with that model.

```ts
import { R2BlobStore, Thread } from "@aiuitech/thread-sdk";

const thread = await Thread.init({ wasmModule });

await thread.createGeneration({
  store: new R2BlobStore(env.BUCKET),
  gen: "docs",                                      // the name of the index
  embedding: { provider: "workers-ai", model: "@cf/baai/bge-base-en-v1.5", dim: 768, options: { pooling: "cls" } },
  lex: { text: true },                              // also index the text for keyword search, and store it
});
```

```python
from thread_sdk import create_generation

create_generation(
    store, "docs",
    embedding={"provider": "openai", "model": "text-embedding-3-small", "dim": 1536},
    lex={"text": True},
)
```

The options that matter:

| Option | Meaning |
|---|---|
| `gen` | The name of the generation. Any short string. |
| `embedding` | The model: `provider` (`workers-ai`, `openai` or `custom`), `model`, and `dim`, the size of its vectors. [Embedding models](embedding.md) lists every field. |
| `lex` | The text index. `{}` (the default) indexes text for keyword search. `{ text: true }` also stores the text, so a hit can return it. `null` (Python: `False`) means no text index: ids and vectors only. |
| `attrs` | A [filter schema](filtering.md), if you know it now. You can add it later. |
| `setCurrent` | `true` (the default) makes this generation the live one. `false` builds it beside the live one. |

`createGeneration` fails with a 409 error if a generation with that name exists.

## A text index

A text index has no vectors and needs no model. It gives you BM25 keyword search, lookups by key, and stored
documents.

```ts
import { createLexicalGeneration } from "@aiuitech/thread-sdk";

await createLexicalGeneration({ store, gen: "notes", lex: { text: true } });
```

```python
from thread_sdk import create_lexical_generation

create_lexical_generation(store, "notes", lex={"text": True})
```

You can add vectors later, with a new generation built beside this one
([Change the model](#change-the-model-or-rebuild)).

## An index for your own vectors

If you make the vectors yourself, give the size of the vectors in place of a model.

```ts
await thread.createGeneration({ store, gen: "images", dim: 512 });

const index = await thread.open({ store });
await index.upsert({ keys: ["cat.jpg", "dog.jpg"], vectors: [catVector, dogVector], metas: ["A cat", "A dog"] });
const { hits } = await index.query(queryVector, { topK: 5 });
```

```python
create_generation(store, "images", dim=512)

with open_index(store) as index:
    index.upsert(keys=["cat.jpg", "dog.jpg"], vectors=[cat_vector, dog_vector], metas=["A cat", "A dog"])
    hits = index.query(query_vector, top_k=5).hits
```

thread normalizes every vector and compares with cosine similarity.

## How an index grows

A new index has one empty cell. That is correct for an index of any size at the start:

1. Each upsert adds a small file. Queries read those files beside the cell.
2. **Compaction** merges the small files into the cell.
3. When a cell passes 3,000 vectors, compaction splits it into two.

So the number of cells follows the data, and you do not choose it. What you must do is make sure that compaction
runs. [Run it in production](production.md#run-compaction) shows the few lines that it takes.

## Load a lot of data

Upserts are the right way to add data to a live index. For a large first load, there are better ways.

| Size of the load | Use | Why |
|---|---|---|
| Up to a few tens of thousands of vectors | Ordinary upserts, in batches of up to 4,096 | Simple. Compaction arranges the cells. |
| Tens of thousands to a few hundred thousand | `thread.bulkBuild` (TypeScript) | It clusters the vectors first, so the cells are good from the start. It runs in small steps that fit a Durable Object alarm. |
| Around a million and more | The build pipeline in the reference Worker | It spreads the work over many isolates and trains a product-quantization codebook, which makes queries read about 4 times fewer bytes. |

A bulk build, in outline:

```ts
const build = await thread.bulkBuild({ store, gen: "docs-v2", embedding, lex: { text: true }, maxBytes: 32 << 20 });

await build.add({ keys, metas, texts, vectors });    // call as many times as you need
await build.seal();                                  // no more documents

let step;
do step = await build.step({ deadlineMs: 10_000 });  // sample, cluster, load, compact
while (!step.done);

await setCurrent(store, { gen: "docs-v2" });         // make it live
await build.cleanup();
```

A build is saved in the bucket after every step. If the process stops, `thread.bulkBuild` with the same `gen`
continues it. Details: [a bulk build in bounded steps](embedding.md#a-bulk-build-in-bounded-steps) and, for the
pipeline, [Build pipeline](build-pipeline.md) and [Operations](operations.md).

## Change the model, or rebuild

Build the new generation beside the live one, then move the pointer. Readers move to the new generation within
30 seconds and never see a half-built index.

```ts
// 1. Create it without making it live
await thread.createGeneration({ store, gen: "docs-v2", embedding: newModel, lex: { text: true }, setCurrent: false });

// 2. Fill it. Writes to a named generation do not move the pointer.
const next = await thread.open({ store, gen: "docs-v2", embed: { ai: env.AI } });
await next.upsert({ keys, texts, metas });

// 3. Switch. `expect` makes the switch fail if someone else moved the pointer first.
await setCurrent(store, { gen: "docs-v2", expect: { gen: "docs", rev: liveRev } });

// 4. Later, delete the old generation
await dropGeneration(store, { gen: "docs", dryRun: false });
```

Details: [switching generations](update-path.md#switching-generations).

## Several indexes in one bucket

Give each index a prefix. Every key of the index is then under that prefix.

```ts
const index = await thread.open({ bucket: env.BUCKET, prefix: "customers/acme/" });
```

To create an index under a prefix, wrap the store: `new PrefixedBlobStore(new R2BlobStore(env.BUCKET),
"customers/acme/")` in TypeScript, `PrefixedBlobStore(store, "customers/acme/")` in Python.

One index for each customer is a good pattern: an idle index costs only its storage, and one customer's data
cannot appear in another customer's results.

## Next

- [Connect to storage](storage.md): R2, S3, a folder, a pack file.
- [Write documents](writing.md): add, replace and delete.
