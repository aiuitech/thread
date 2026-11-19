# Search

`search` is the one call for every kind of search. You give it a text, a vector, or both, and it runs what the
index supports.

```ts
const { hits } = await index.search({ text: "why did the deployment fail?", topK: 10, docs: true });
```

```python
hits = index.search(text="why did the deployment fail?", top_k=10, docs=True).hits
```

## What runs

| You pass | The index has | What runs | `mode` in the result |
|---|---|---|---|
| `text` | A text index only | Keyword search (BM25) | `"lexical"` |
| `text` | A model and a text index | The text is embedded. Semantic and keyword search both run, and the lists are merged. | `"hybrid"` |
| `text` | A model, no text index | Semantic search | `"dense"` |
| `vector` | Vectors | Semantic search with your vector | `"dense"` |
| `text` and `vector` | Vectors and a text index | Keyword search on the text, semantic search with your vector, merged | `"hybrid"` |

"Has a model" means that the index was [created for an embedding model](create-an-index.md) and opened with
`embed`. To force keyword search only on such an index, pass `embed: false`.

## Read the hits

```ts
const { hits, mode } = await index.search({ text: "deploy failed", topK: 3, docs: "text" });

hits[0];
// {
//   id: 2,
//   score: 0.0328,                          the fused score. Higher is better.
//   dense:   { rank: 1, score: 0.83 },      the position and cosine similarity in the semantic list
//   lexical: { rank: 1, score: 0.46 },      the position and BM25 score in the keyword list
//   doc: { id: 2, meta: "On-call guide", attrs: 0, text: "When a deployment fails..." },
// }
```

- `docs: true` adds `doc.meta`, the string that you stored with the document. `docs: "text"` also adds the stored
  text. Without `docs`, a hit is an id and a score, and the query is a little faster.
- `dense` or `lexical` is absent when the document was found by one list only.
- **A hit does not include the document's key.** If you need the key or a URL in your results, put it in `metas`
  when you write, for example `metas: [JSON.stringify({ key, title })]`.
- The score of a hybrid search is a rank score, not a similarity. Use it to order results, not as a threshold.
  For a threshold, use `hit.dense.score`, which is a cosine similarity from -1 to 1.

## Keyword search

Keyword search uses BM25, the standard ranking for text. The query and the documents go through the same steps:

1. Lowercase, and split into words.
2. Remove common English words (`the`, `and`, ...).
3. Reduce each word to its stem, so `deployments`, `deployed` and `deploy` all match.

This suits English prose. For other content, set the tokenizer when you create the index:

| Content | Tokenizer option |
|---|---|
| Code, ticket numbers, host names (`INC-12345`, `getUserName`, `k8s.io`) | `identifiers: "both"` keeps each one whole and also indexes its parts |
| Chinese, Japanese, Korean | `segmenter: "cjk-bigram"` |
| Languages other than English | `stemmer: null, stopwords: null` turns off the English rules |
| Exact tags such as `host:cdn-1` | `identifiers: "keep"` |

```ts
await createLexicalGeneration({ store, gen: "tickets", lex: { text: true, tokenizer: { identifiers: "both" } } });
```

The tokenizer of an index cannot change after the first write. `await index.tokenize("some text")` shows the
tokens that a text produces. All options: [the tokenizer](lexical.md#tokenizer).

## Semantic search with your own vector

`query` is the vector search alone. Use it when you make the query vector yourself and do not need text.

```ts
const { hits } = await index.query(queryVector, { topK: 10 });
// hits[0] = { id, score, attrs, source }      score is the cosine similarity
```

```python
hits = index.query(query_vector, top_k=10).hits
```

`index.queryMany(vectors, options)` runs several queries together on the same revision.

## Filter the results

Add `filter` to limit a search to documents whose attributes match. It applies to both lists.

```ts
await index.search({ text: "deploy failed", topK: 10, filter: { tenant: 42, lang: { $in: ["en", "fr"] } } });
```

[Filter](filtering.md) shows how to define the fields.

To limit a search to a known set of documents, pass their ids (TypeScript): `search({ text, ids: [3, 7, 12] })`.

## Tune the results

Most applications need only `topK`. These options exist for when the results are not what you want.

| Option | Default | Change it when |
|---|---|---|
| `topK` | 10 | You want more or fewer results |
| `weights` | `{ dense: 1, lexical: 1 }` | One list should count more. `{ dense: 1, lexical: 2 }` favours exact words. |
| `recallK` | 50, or `topK` if larger | Good results are missing from a hybrid search: take each list deeper before the merge |
| `dense: { nprobe }` | Set by the index | Semantic results miss documents that you know are close: scan more cells. More bytes for each query. |
| `dense: { rerank }` | 150 | The order of semantic results is poor: score more candidates precisely |
| `rrfK` | 60 | Rarely. A lower value gives the first ranks more weight. |

Python uses the same names in snake case: `top_k`, `recall_k`, `rrf_k`.

To see what a query did, read `result.stats`: the time for each phase, the bytes read, the cache hits and misses.
[Observability](observability.md) explains every field.

## Fetch documents without a search

```ts
const { records } = await index.get(["notes/deploy.md"], { text: true });   // by key
const docs = await index.docs([2, 7], { text: true });                      // by id
```

[Keys](keys.md) covers lookups and listings.

## Next

- [Filter](filtering.md)
- [How hybrid search merges two lists](how-it-works.md#how-hybrid-search-merges-two-lists)
- [Query path](query-path.md): what a query reads, in detail.
