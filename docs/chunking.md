# Chunking: word windows before embedding

Embedding models take a bounded passage, and a hit is more useful when it points at a passage than at a whole
file. Both SDKs and the CLI therefore split a text into **flat word windows** before it is embedded and upserted.
This is a client-side transform: the index never sees the parent text, each chunk is an ordinary document, and
nothing in the wire format changes. The implementations are `sdk/ts/src/chunk.ts`, `sdk/python/thread_sdk/chunk.py`
and `crates/thread-cli/src/chunk.rs`; all three are checked against one fixture, `sdk/ts/test/fixtures/chunks.json`,
so they produce the same chunks character for character.

## The rule

- A **word** is a maximal run of characters that are not Unicode White_Space (what Rust's `char::is_whitespace`
  tests; spelled out in the TypeScript and Python sources since JS's `\s` and Python's `str.split` each include a
  few characters the other does not).
- `words` (default **200**, at least 1) is the window; `overlap` (default **0**, less than `words`) is how many
  words a chunk shares with the previous one. Chunk `i` holds words `[i * (words - overlap), i * (words - overlap)
  + words)`; the last chunk is whatever remains, and no chunk starts once the previous one reached the end.
- A chunk's text is a verbatim slice of the original: whitespace inside a chunk is kept, whitespace around it is
  not. `start` / `end` are the chunk's code-point offsets in the text.
- A text with no words is one empty chunk, so a keyed document always writes at least one chunk.

```ts
import { chunkText } from "@aiuitech/thread-sdk";
chunkText("w1 w2 w3 w4 w5 w6 w7", { words: 3, overlap: 1 });
// [{ index: 0, text: "w1 w2 w3", start: 0, end: 8, words: 3 }, { index: 1, text: "w3 w4 w5", start: 6, end: 14, words: 3 }, { index: 2, text: "w5 w6 w7", start: 12, end: 20, words: 3 }]
```

```python
from thread_sdk import chunk_text
chunk_text("w1 w2 w3 w4 w5 w6 w7", words=3, overlap=1)      # [Chunk(index=0, text="w1 w2 w3", start=0, end=8, words=3), ...]
```

```sh
thread chunk --words 3 --overlap 1 "w1 w2 w3 w4 w5 w6 w7"   # {"words": 3, "overlap": 1, "chunks": [[{index, text, start, end, words}, ...]]}
thread chunk --file report.txt                               # the defaults
```

## Chunked upserts

`upsert` takes the same parameters and writes one document per chunk:

```ts
await index.upsert({
  keys: ["docs/a.md", "docs/b.md"], texts: [textA, textB], metas: ["A", "B"], attrs: [{ tenant: 1 }, { tenant: 2 }],
  chunk: { words: 200, overlap: 20 },                  // or `chunk: true` for the defaults
  embed: async (texts) => (await openai.embeddings.create({ model, input: texts })).data.map((d) => d.embedding),
});
// r.chunks = { documents: 2, count: 11, stale: 0 }; r.ids lists the chunk ids in order
```

```python
r = index.upsert(keys=["docs/a.md", "docs/b.md"], texts=[text_a, text_b], metas=["A", "B"], chunk={"words": 200, "overlap": 20},
                 embed=lambda texts: model.encode(texts))
```

- **Keys.** Chunk `i` of the document keyed `k` is stored under the key `k#i` (`chunkKey` / `chunk_key`), so
  `get("docs/a.md#3")`, `listKeys({ prefix: "docs/a.md#" })` and `deleteKeys` work as for any key, and a hit's key
  names the passage. Chunked upserts need `keys` (one id cannot hold several chunks) and `texts`.
- **Meta and attrs** are the parent's, copied to every chunk, so a filter on `tenant` or a `meta` title behaves
  as before.
- **Vectors** come from `embed`, called once with every chunk's text in order; it returns one vector per text, in
  the shapes `vectors` accepts. The SDKs do no embedding themselves. Without `embed` the chunks are text-only
  documents (found by BM25 alone, as for any upsert without vectors). `embed` also works without `chunk`: it then
  embeds `texts` as given.
- **Re-puts.** A known chunk key keeps its id, so re-putting a document replaces its chunks in place. Chunk keys
  of the parent that the new text no longer produces (it got shorter) are tombstoned in the same revision, with
  `replace: false` too; `r.chunks.stale` counts them and the manifest's `change.chunk` records `{words, overlap,
  documents, stale}`. Finding them is one prefix listing per parent against the head revision.
- **Limits.** At most 4096 chunks per call (the ordinary batch limit): pass fewer texts per call for long
  documents.

The CLI mirrors this for local segments: `thread seg build --chunk-words 200 [--chunk-overlap 0] [--first-id 0]`
splits every row's text, keys the chunks `k#i` when rows have keys, copies `meta` and `attrs`, and numbers the
chunk ids from `--first-id` in file order (the id field is not used). Its output reports `documents` (rows) next
to `docs` (chunks written).

## What is not here yet

Only the flat strategy exists. Sentence- or token-aware windows, per-chunk metadata (the chunk's offsets in a
JSON meta), and deleting a parent's chunks in one call (`listKeys` by prefix, then `deleteKeys`, today) are
possible follow-ups; the fixture is where a new strategy's rule would be pinned first.
