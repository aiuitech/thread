# Write documents

You add and change documents with `upsert`, and remove them with `delete` or `deleteKeys`. This page shows each
case, and the three rules that keep writes cheap.

## A document

A document can have these parts. All are optional except a name (a key or an id).

| Part | What it is | Example |
|---|---|---|
| `keys` | A unique string name. thread gives the document an id for you. | `"notes/deploy.md"` |
| `ids` | A numeric id (an unsigned 32-bit integer), if you prefer to choose it yourself. | `1042` |
| `texts` | The text. It is indexed for keyword search, and embedded if the index has a model. | `"The deploy failed..."` |
| `vectors` | The embedding, if you make it yourself. | `Float32Array(768)` |
| `metas` | A string that comes back with a hit: a title, or a JSON object. | `'{"title":"Runbook"}'` |
| `attrs` | Small fields that you can [filter](filtering.md) on. Define the fields first. | `{ tenant: 42, lang: "en" }` |

Every part is a list, with one entry for each document in the batch.

## Add or replace documents

```ts
const result = await index.upsert({
  keys: ["notes/deploy.md", "notes/billing.md"],
  texts: ["The deploy failed because the migration timed out.", "Invoices go out on the first of the month."],
  metas: ["Deploy runbook", "Billing FAQ"],
});
// result.ids = [0, 1]    the ids of the documents
// result.rev = 4         the revision that this write made
```

```python
result = index.upsert(
    keys=["notes/deploy.md", "notes/billing.md"],
    texts=["The deploy failed because the migration timed out.", "Invoices go out on the first of the month."],
    metas=["Deploy runbook", "Billing FAQ"],
)
```

`upsert` means "insert or replace". If the key exists, the new document replaces the old one: its vector, text,
meta and attributes. If the key is new, the document is added.

Where the vectors come from:

| The index | You pass | Result |
|---|---|---|
| Has an embedding model, opened with `embed` | `texts` | Each text is embedded for you |
| Any index with vectors | `vectors` | Your vectors are stored as they are (after normalization) |
| Any index with vectors | `texts` and `embed: async (texts) => vectors` | Your function is called once for the batch |
| A text index, or `embed: false` | `texts` | Text only. The document is found by keywords. |

In one batch, some documents can have a vector and others not: `vectors: [vectorA, null]`.

## Long documents

A long text gives a poor vector, and a hit on a 50-page file does not tell you where to look. Add `chunk: true`
and each text is split into passages of 200 words. Each passage becomes its own document.

```ts
await index.upsert({ keys: ["docs/handbook.md"], texts: [handbook], metas: ["Handbook"], chunk: true });
// stored as docs/handbook.md#0, docs/handbook.md#1, ...
```

[Chunking](chunking.md) has the options and the rules.

## Change one part of a document

To add or change only the vector, or only the text, use `scope`. The other part stays as it is.

```ts
// add a vector to a document that has only text
await index.upsert({ keys: ["notes/deploy.md"], vectors: [vector], attrs: [{ tenant: 42 }], scope: "dense" });

// change the text and the meta, keep the vector
await index.upsert({ keys: ["notes/deploy.md"], texts: [newText], metas: ["Deploy runbook v2"], scope: "lexical" });
```

**With `scope: "dense"`, pass the document's `attrs` again.** The vector keeps its own copy of the attributes,
and a filter on a vector search tests that copy. Without `attrs`, the new vector has none, and a filter such as
`{ tenant: 42 }` no longer finds the document by its vector.

## Delete documents

```ts
await index.deleteKeys(["notes/billing.md"]);     // by key
await index.delete([1042, 1043]);                 // by id
```

```python
index.delete_keys(["notes/billing.md"])
index.delete([1042, 1043])
```

A delete hides the whole document at once: its vector, its text, its meta and its key.

To delete every chunk of a long document, see [Chunking](chunking.md#update-or-delete-a-chunked-document).

## When a write is visible

| Reader | Sees the write |
|---|---|
| The handle that made the write | At once |
| Another open index that runs searches | After its next background refresh: a search checks for a new revision when `refreshInterval` (30 seconds) has passed, and the searches after that one use it |
| Another open index that only calls `get` or `listKeys` | After `await index.refresh()`. Lookups do not start a refresh. |

When a lookup must see a write that has completed, call `await index.refresh()` first.
[Know your freshness](production.md#know-your-freshness) has the details.

## Three rules for cheap writes

**1. Batch.** Every `upsert` or `delete` makes one revision and about ten requests to the store, whether it holds
1 document or 1,000. Send at least 64 documents in a call, or everything that arrived in the last second. Never
make one call for each document. One call can hold up to 4,096 documents.

**2. Compact.** Each write adds small files that every later query must read. Compaction merges them. If writes
go through your own code, run compaction after a write when it is due:

```ts
if (index.compactionDue().due) {
  for (let i = 0; i < 8; i++) {                           // a few bounded slices, then stop
    const run = await index.compact();
    if (!run.more || run.outcome === "locked") break;     // locked: another run has the work
  }
}
await index.compactSegments();     // the same for the text side, from time to time
```

[Run it in production](production.md#run-compaction) explains where to run this.

**3. Write from one place.** Two writers at the same time are safe on a bucket: the store accepts one revision
and the other writer tries again. (A folder on disk that Node and Python both write to is the exception: see
[Write from one place](production.md#write-from-one-place).) But each retry costs time. One writer for each index (a queue consumer, a Durable Object, a
cron job) is simpler and faster. In a Worker, also keep writes out of the isolates that serve queries: a write
uses memory that the isolate does not give back.

## What a write returns

| Field | Meaning |
|---|---|
| `rev` | The revision that the write made |
| `ids` | The id of each document, in order |
| `allocated` | How many keys were new |
| `count` | How many documents were written |
| `chunks` | With `chunk`: `{ documents, count, stale }` |
| `conflicts` | How many times the write lost a race with another writer and tried again |

## Errors

Errors are `ThreadError` with an HTTP-style `status`.

| Status | Meaning | What to do |
|---|---|---|
| 400 | The request is wrong: a missing part, a vector of the wrong size, too many documents in one call, an index with no text configuration | Fix the call |
| 404 | An object is missing: no index at this store or prefix | Check the store, the prefix and the generation |
| 409 | A conflict: the generation exists, or the pointer was not where you expected | Read the state again, then decide |
| 413 | Compaction or a bulk build cannot process an input within its memory budget | Send smaller batches, or raise the budget |
| 502 | The embedding provider failed | Try again later |

## Next

- [Search](search.md)
- [Filter](filtering.md), [Keys](keys.md), [Chunking](chunking.md)
- [Update path](update-path.md): what a write puts in the bucket, byte by byte.
