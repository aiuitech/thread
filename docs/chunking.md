# Chunking

A long document makes a poor search result. Its one vector is an average of everything in it, and a hit on a
50-page file does not say where to look. Chunking splits each text into passages, and each passage becomes its own
document with its own vector.

```ts
await index.upsert({ keys: ["docs/handbook.md"], texts: [handbook], metas: ["Handbook"], chunk: true });
```

```python
index.upsert(keys=["docs/handbook.md"], texts=[handbook], metas=["Handbook"], chunk=True)
```

## What happens

1. The text is split into windows of 200 words.
2. Each window is stored as a document with the key `docs/handbook.md#0`, `docs/handbook.md#1`, and so on.
3. Each window gets the `metas` and `attrs` of the parent, so titles and filters work as before.
4. Each window is embedded: by the index's model, or by the `embed` function that you pass.

A search then returns passages. The key of a passage tells you the document and the position.

## Options

```ts
await index.upsert({ keys, texts, chunk: { words: 200, overlap: 20 } });
```

| Option | Default | Meaning |
|---|---|---|
| `words` | 200 | The size of a window, in words |
| `overlap` | 0 | How many words a window shares with the one before it |

Which values to choose:

- **200 words** suits most embedding models and most prose. It is about one long paragraph.
- **Smaller windows** (50 to 100) give more precise hits and more documents. Use them for FAQs and short notes.
- **An overlap** of 10% to 20% helps when a sentence that matters can fall on a border between two windows.
- Check the input limit of your embedding model. A window must fit in it.

## Update or delete a chunked document

Upsert the document again with the same key. Its chunks are replaced. If the new text is shorter, the chunks that
no longer exist are removed in the same write.

To delete a chunked document, list its chunk keys and delete them. Read every page of the listing, and keep
only the keys that are exactly a chunk key of this document: another document can have a key that starts the
same way, such as `docs/handbook.md#notes`.

```ts
import { chunkIndexOf, chunkKey } from "@aiuitech/thread-sdk";

const parent = "docs/handbook.md";
const chunkKeys: string[] = [];
let after: string | undefined;
do {
  const page = await index.listKeys({ prefix: parent + "#", limit: 1000, after });
  for (const { key } of page.entries) {
    const i = chunkIndexOf(key, parent);
    if (i !== null && chunkKey(parent, i) === key) chunkKeys.push(key);   // "#3" is a chunk, "#03" is not
  }
  after = page.more ? page.next : undefined;
} while (after !== undefined);

if (chunkKeys.length > 0) await index.deleteKeys(chunkKeys);
```

Python has the same two functions: `chunk_index_of(key, parent)` and `chunk_key(parent, i)`.

## Chunk without writing

`chunkText` gives you the same windows, if you want to see them or embed them yourself.

```ts
import { chunkText } from "@aiuitech/thread-sdk";

chunkText("w1 w2 w3 w4 w5 w6 w7", { words: 3, overlap: 1 });
// [{ index: 0, text: "w1 w2 w3", start: 0, end: 8, words: 3 }, { index: 1, text: "w3 w4 w5", ... }, ...]
```

```python
from thread_sdk import chunk_text

chunk_text("w1 w2 w3 w4 w5 w6 w7", words=3, overlap=1)
```

## Limits

- A chunked upsert needs `keys` and `texts`. It does not work with numeric ids alone.
- One call writes at most 4,096 chunks. Send fewer documents in a call when they are long.
- Windows are counted in words. There is no sentence-aware or token-aware splitting.
- TypeScript, Python and the CLI produce the same chunks, character for character.

## Under the hood

The rest of this page is the exact rule and the behaviour of chunked writes.

### The rule

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

### Chunked upserts

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

### What is not here yet

Only the flat strategy exists. Sentence- or token-aware windows, per-chunk metadata (the chunk's offsets in a
JSON meta), and deleting a parent's chunks in one call (`listKeys` by prefix, then `deleteKeys`, today) are
possible follow-ups; the fixture is where a new strategy's rule would be pinned first.
