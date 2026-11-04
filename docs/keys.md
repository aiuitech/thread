# Keys: `get(key)`, unique string keys and a key-value index

thread's ids are u32. Many corpora are named by strings instead: an object address, a path, a URL, a primary key.
This page is the design of the **key store**: how a string key resolves to an id and its document, how keyed
writes allocate ids, what a prefix listing costs, and how a "simple lookup of unique keys with a few attributes"
(a key-value index, with or without vectors and text) fits the blob-native design. The format is in
[lexical.md](lexical.md#keys-keysrs-magic-thrdkeys-version-1), the code in `crates/thread-core/src/keys.rs`, the
SDK methods in [sdk/ts/README.md](../sdk/ts/README.md) and [sdk/python/README.md](../sdk/python/README.md).

## In one paragraph

A segment (the immutable `THREADIX` object an upsert writes, see [lexical.md](lexical.md)) can carry a `keys`
section: the batch's unique string keys sorted by bytes, each with its u32 id, bucketed behind a small resident
directory, exactly like the `terms` directory with an id in place of a posting range. `get(key)` is then two
range reads against cached objects: the key's bucket (-> id), then the id's `docs` bucket (-> meta, attrs, text).
Segments are walked newest first and the usual tombstone rule applies, so a re-put of a key replaces its document
and a delete hides it; a segment compaction merges the sections (newest copy of a key wins). Keyed upserts need no
ids: a known key keeps its id, a new key takes the next one from a counter in the manifest (`nextId`). Because the
section is sorted, a prefix listing is a binary search to the first bucket and a forward scan. Nothing in the
dense query path changes.

## API

```ts
await index.upsert({ keys: ["s3://b/a.bin", "s3://b/c.bin"], metas: [metaA, metaC], texts: ["host:cdn-1 size:large", ...] });
// -> { ids: [1000, 1001], allocated: 2, ... }   known keys keep their id, new keys get the next ones
const { records } = await index.get(["s3://b/a.bin"], { text: true });   // Map<key, {key, id, meta, attrs, text?}>
const { ids } = await index.resolveKeys(["s3://b/a.bin"]);              // Map<key, id>
const page = await index.listKeys({ prefix: "s3://b/", limit: 100 });   // {entries: [{key, id}], more, next}
await index.listKeys({ prefix: "s3://b/", after: page.next });           // the next page
await index.deleteKeys(["s3://b/c.bin"]);                                // {count, missing, ...}
await index.upsert({ ids: [7], keys: ["seven"], metas: ["m"] });         // ids and keys together: keys name the ids
```

```python
r = index.upsert(keys=["s3://b/a.bin", "s3://b/c.bin"], metas=[meta_a, meta_c], texts=[...])   # r.ids, r.allocated
index.get(["s3://b/a.bin"], text=True)              # {key: KeyRecord(id, meta, text, attrs, key)}
index.resolve_keys(["s3://b/a.bin"])                # {key: id}
page = index.list_keys("s3://b/", limit=100)        # KeyPage(entries=[(key, id)], more, next)
index.list_keys("s3://b/", after=page.next)
index.delete_keys(["s3://b/c.bin"])
```

Keys are non-empty UTF-8 strings of at most 4096 bytes, unique within a batch. A generation needs a lexical
configuration (`configureLexical` / `createLexicalGeneration`) to store them; `manifest.lex.bucketKeys` (default
128) is the only knob. A segment carries a `keys` section when its upsert passed keys (`segments[].keys: true`);
other segments are unaffected.

## The key-value case

The case that motivated this: records named by a unique string address with a few attributes (a host, a size,
...) that are looked up by key and sometimes listed or filtered, without vectors. This is a **lexical-only
generation** with keys:

| need | mechanism | cost |
|---|---|---|
| the record and its attributes | `metas` (a JSON object) in the `docs` store | one bucket read per lookup |
| `get(key)` | the `keys` section | one bucket read, then the docs read |
| `host = X` (any cardinality) | a facet token in `texts`, e.g. `host:cdn-1.example.com`, with `tokenizer: {identifiers: "keep"}`; `search({text: "host:cdn-1.example.com"})` is the exact set of ids with that host | one term lookup + one posting list |
| a bucketed range such as size | a few bits of the 32-bit attrs word ([filtering.md](filtering.md)), tested inside the scan or through `docs` | nothing extra |
| `keys under a prefix` | `listKeys({prefix})` | one contiguous range per segment |

Facet tokens are plain terms, so the BM25 score of a facet-only query is meaningless but the hit set is exact, and
they combine with real text terms and with vector search in `search`. With `identifiers: "keep"` a token like
`host:cdn-1.example.com` stays one term (it is a technical run with joiners); with the default tokenizer it would
be split into words.

Approximate size per record with 60-byte keys, a 100-byte meta and three facets: keys ~65 B, docs ~105 B,
postings ~6 B; about 180 MB per million records, with resident directories under 1 MB (16 B + one key per
bucket of 128 keys, 16 B per bucket of 128 docs).

## Ids: allocation and the counter

`manifest.nextId` is the next id a keyed upsert allocates; absent, it is `manifest.vectors` (a built dense base
holds ids `[0, vectors)`; a lexical-only generation starts at 0). A keyed upsert resolves its keys against the
head revision's segments, reuses the ids of known keys and takes `nextId, nextId + 1, ...` for the rest, then
commits `nextId + allocated`. The manifest commit is the serialization point (a create-only PUT), so two writers
never allocate the same id: a lost race re-resolves the keys against the new head and re-encodes with the new ids.
An upsert with explicit ids moves the counter past its largest id, so explicit and allocated ids never collide
once both have been used. A rollback keeps the larger counter: an id is never reused.

Reusing the id is what makes a re-put a replacement: the new segment carries the key again under the same id and
the upsert's tombstone set hides the older copies (slab, delta, docs, postings), exactly as an upsert by id does.
Giving `ids` and `keys` together is allowed: the keys then name those ids (a key moved to another id resolves to
the newest copy; the old id's document stays until it is deleted or replaced).

## Resolution and listing across segments

`resolveKeys` walks the segments newest first. In each, the pending keys are looked up (one coalesced bucket read
per segment); a key found there is decided by that copy: live if no tombstone set with `rev > addedRev` holds its
id, dead otherwise (a delete by key tombstones the id, which hides the key's newest entry; a later put of the same
key writes a newer entry under a new id). After a compaction every live key sits in the base segment, so a lookup
is one keys read and one docs read; before it, a key reads at most one bucket per segment, bounded by the
segment-count threshold that triggers compaction. A missing key costs a bucket read in every segment that has
keys.

`listKeys` asks every segment for a page (`prefix`, `after`, `limit`) concurrently and merges them newest first,
the first copy of a key deciding whether it is live. A segment page that stopped early (more keys follow) bounds
what is known, since keys past its last entry could have a newer copy there: the merged page ends at the smallest
such cursor and returns it as `next`; the caller continues with `after: next` until `more` is false. Within one
segment a page is one contiguous range read (`ceil((limit + 1) / bucketKeys) + 1` buckets, usually a few KB).

## Delete, compaction, GC, packs

`deleteKeys` resolves the keys through the head revision and writes one tombstone set for the ids found (nothing
when none was). `compactSegments` merges the `keys` sections like the `docs` sections: inputs oldest first, the
newest copy of a key wins, entries whose id a tombstone set hides are dropped; the base carries `keys: true`. GC,
snapshots and packs treat segments as opaque objects, so keys travel with them, and the interop test checks that
an index keyed from one SDK resolves identically from the other.

## What it costs on the hot path

Nothing. The dense and lexical query paths do not read the `keys` section; a segment's resident parts grow by its
keys directory (one entry plus one key per bucket). `get` is two rounds (keys bucket, docs bucket) against
immutable cached objects: a few milliseconds from a local copy, tens of milliseconds from a Worker through a warm
edge cache, the same as resolving a hit's document today.

## Adding vectors to a text-first index

A keyed index often gets its texts and metas first and its embeddings later. `upsert({ keys, vectors, scope:
"dense" })` (Python `scope="dense"`) attaches the vectors to the existing documents: the keys resolve through the
head (a key without a document is an error), no segment is written, and the tombstone set hides only older dense
copies, so `get(key)`, `listKeys`, text search and the stored meta and text stay as they were. With an embedding
model, `texts` (not stored) can stand in for `vectors`. The other way round, `scope: "lexical"` changes texts or
metas without the vector. See [update-path.md](update-path.md#scoped-tombstone-sets-one-sided-updates).

## Not done

- No reverse lookup (id -> key): put the key in `meta` if a search hit must show it.
- No Bloom filter for negative lookups: a missing key reads one bucket per segment with keys.
- Facet tokens count in BM25 document lengths; a flag that excludes them from the length would keep text scores
  unchanged in a mixed corpus.
- A `raw` tokenizer option (the whole text as one token) would make facet tokens independent of the
  `identifiers` mode.
