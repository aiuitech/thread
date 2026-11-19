# Migrate from Cloudflare Vectorize

thread's vector calls are close to Vectorize's, so most code changes are small. This page maps each call, lists
the differences that need a decision, and gives a migration plan.

If you have not chosen yet, [Why thread](why-thread.md#cloudflare-vectorize) compares the two, with measurements.

## Before you start: check these four

| Vectorize | thread | What to do |
|---|---|---|
| Ids are strings | Ids are unsigned 32-bit integers | Use your string ids as **keys**: `upsert({ keys })`. thread gives each key a numeric id. |
| Metadata is a JSON object of up to 10 KiB. You filter on indexed properties. | `metas` is a string that is returned but not filtered. Filters use `attrs`: 32 bits of fields for each document. | Put what you display in `metas` (as JSON). Put what you filter on in `attrs`. If your filters need more than 32 bits, or `$or`, thread is not a fit. |
| Cosine, euclidean or dot product | Cosine only | If your index is not cosine, thread is not a fit. |
| A write is visible after a short delay that Vectorize manages | A write is visible to other readers after their next refresh (30 seconds by default), and compaction is your task | Read [Run it in production](production.md). |

## The calls

| Vectorize | thread |
|---|---|
| `env.VECTORIZE.query(vector, { topK })` | `index.query(vector, { topK })` |
| `matches[].id`, `matches[].score` | `hits[].id`, `hits[].score` |
| `query(vector, { returnMetadata: "all" })` | `index.search({ vector, topK, docs: true })`, then `hit.doc.meta` |
| `query(vector, { filter: { field: { $eq: value } } })` | `index.query(vector, { filter: { field: value } })` with the same operators: `$eq`, `$ne`, `$in`, `$nin`, `$lt`, `$lte`, `$gt`, `$gte` |
| `upsert([{ id, values, metadata }])` | `index.upsert({ keys, vectors, metas, attrs })` |
| `insert([...])`, which keeps the first vector of an id | No equivalent. `upsert` always replaces. |
| `deleteByIds(ids)` | `index.deleteKeys(keys)`, or `index.delete(ids)` for numeric ids |
| `getByIds(ids)` | `index.get(keys)` returns the meta, the attributes and the text. The stored vector is not returned. |
| `describe()` | `index.manifest`, `index.dim`, `index.cells` |
| Namespaces | One index for each `prefix`, or an `attrs` field such as `tenant` |
| No equivalent | `index.search({ text })`: keyword and hybrid search |

## Before and after

```ts
// Vectorize
await env.VECTORIZE.upsert([
  { id: "doc-1", values: embedding, metadata: { title: "Deploy runbook", tenant: 42 } },
]);
const { matches } = await env.VECTORIZE.query(queryVector, {
  topK: 10,
  returnMetadata: "all",
  filter: { tenant: 42 },
});
```

```ts
// thread
await index.upsert({
  keys: ["doc-1"],
  vectors: [embedding],
  metas: [JSON.stringify({ key: "doc-1", title: "Deploy runbook" })],
  attrs: [{ tenant: 42 }],
});
const { hits } = await index.search({
  vector: queryVector,
  topK: 10,
  docs: true,
  filter: { tenant: 42 },
});
const results = hits.map((hit) => ({ score: hit.score, ...JSON.parse(hit.doc!.meta) }));
```

A hit does not carry its key, so the example stores the key in `metas`.

## A migration plan

1. **Create the index** with the dimension of your vectors, and the filter fields you need.

   ```ts
   await thread.createGeneration({
     store: new R2BlobStore(env.BUCKET),
     gen: "main",
     dim: 768,
     attrs: { tenant: { bits: 16 }, lang: { values: ["en", "fr", "de"] } },
   });
   ```

   If you want thread to embed texts for you from now on, give `embedding` in place of `dim`
   ([Create an index](create-an-index.md)).

2. **Stop writes, or record them.** A copy that runs while documents change can put an old version over a new
   one, or bring back a deleted document. So either pause writes until step 4 is done, or record every write and
   delete in order (a queue, a log table), from before the copy starts.

3. **Copy the data.** Read your documents from your source of truth, with their vectors, and upsert them in
   batches of a few hundred to 4,096.

   For more than a few tens of thousands of vectors, use a
   [bulk build](create-an-index.md#load-a-lot-of-data) in place of steps 1 and 3. It creates the generation
   itself, so give it a name that does not exist yet.

4. **Replay, then write to both.** When the copy is complete, apply the recorded changes to thread in the order
   they happened. From then on, send each write and delete to Vectorize and to thread.

5. **Compare.** Run the same queries against both and compare the top results. They will not be identical: both
   indexes are approximate. Expect most of the top 10 to agree.

6. **Switch reads** to thread. Then stop the writes to Vectorize.

## What you gain and what you give up

You gain keyword and hybrid search, an index in your own bucket, a much lower cost at the same recall, and an
index that also opens in Node, in Python and offline.

You give up some median latency (about 90 ms against about 30 ms on the 1M benchmark), rich metadata filters, and
a service that runs compaction for you.
