# Attribute filtering

How a query is restricted to the documents that match a condition on their metadata, without the index reading
anything it does not read already. The API is in the SDK READMEs ([TypeScript](../sdk/ts/README.md#filtering),
[Python](../sdk/python/README.md#filtering)); this page is the design and the knobs.

## The attribute word

Every document carries one 32-bit **attrs** word. It is stored three times, always next to something a query
reads anyway:

- in the entry table of every slab the vector was written to (16 bytes per entry, inside the scan prefix: see
  [query-path.md](query-path.md#slab-formats)),
- in the delta segment entry of a vector added by an upsert ([update-path.md](update-path.md)),
- in the document's `docs` entry ([lexical.md](lexical.md#docs-docsrs-magic-thrddocs-version-2)), where a text query
  resolves it.

So filtering costs no extra bytes on the dense path: the scan already has the word in the prefix it range-reads.
That is why the feature was built on the word rather than on the free-form `meta` string, which lives in buckets the
scan never touches.

A generation may declare a **schema** (`manifest.attrs`) that packs named fields into the word:

```json
"attrs": {"fields": [
  {"name": "tenant", "type": "number", "offset": 0,  "bits": 12},
  {"name": "lang",   "type": "enum",   "offset": 12, "bits": 2, "values": ["en", "fr", "de"]},
  {"name": "public", "type": "bool",   "offset": 14, "bits": 1},
  {"name": "year",   "type": "number", "offset": 15, "bits": 7, "min": 1970}
]}
```

- `number`: an unsigned integer of `bits` bits, stored as `value - min` (a year fits 7 bits from 1970).
- `enum`: one of `values`, stored as its 1-based index (0 = unset); `ceil(log2(n + 1))` bits.
- `bool`: one bit.

`configureAttrs` lays the fields out in declaration order and writes the schema as a manifest revision. A schema
only **grows**: a later call keeps every existing field's offset and width (an enum may append values that fit its
width), keeps fields it does not mention, and appends new fields after them, so the words already stored stay
valid. Without a schema the whole word is one numeric field named `attrs`, which is how a generation built before
the schema existed (g4) is filtered. 32 bits is the budget: a tenant id, a language, a few flags and a coarse date
fit; anything larger belongs in the `meta` string or in the application's own store, keyed by id.

## Filter expressions and their compiled form

A filter is a conjunction of conditions on fields, in a Vectorize-like syntax:

```ts
{ tenant: 5, lang: { $in: ["en", "fr"] }, public: true, year: { $gte: 2020, $lt: 2024 } }
```

| operator | fields | meaning |
|---|---|---|
| bare value, `$eq`, `$ne` | all | equal / not equal |
| `$in`, `$nin` | all | in / not in a list |
| `$lt`, `$lte`, `$gt`, `$gte` | number | range (the four combine into one clause) |
| `$any`, `$all`, `$none` | number | bit tests on the field's bits: some / every / no bit of the mask set |

There is no `$or`; conditions on different fields are ANDed. A value a field cannot hold (an unknown enum value,
a number out of the field's range) makes `$eq` match nothing and `$ne` match everything, and is dropped from
`$in` / `$nin`; a range that excludes every value matches nothing. These rules are what lets a query ask for
`year: {$gte: 1900}` on a 1970-based field.

The SDK compiles the expression against the schema into the form the kernel evaluates, a list of **clauses** over
the masked word (`x = attrs & mask`):

```text
[clause count n] then n x [mask, op, a, b, value count v, v sorted values]
ops: 0 eq (x == a)  1 ne  2 in (values)  3 nin  4 range (a <= x <= b)  5 any (x != 0)  6 all (x == mask)  7 none (x == 0)
```

The kernel knows nothing about fields: a field condition is a clause whose mask selects the field's bits and whose
values are pre-shifted. Both SDKs compile to exactly the same words (`sdk/ts/test/fixtures/filters.json` is
checked by both test suites, and the interop test compares them live); a host that already has the words (the
Worker, a cached query) passes them as they are. The compiled words are the argument every scan kernel takes
(`scan`, `scan_pq`, `scan_delta`), next to the tombstone set.

## Dense queries: pre-filtering inside the scan

The scan kernels apply the filter the way they apply tombstones: an entry's attrs word is tested **before top-k
insertion**, and only for entries whose score beats the running heap threshold, so the cost is a masked compare
per candidate that would have entered the heap. A filtered-out entry never takes a candidate slot, so the top-T
of a slab is the top-T of its *matching* entries, and the rerank re-scores matching candidates only. Without a
filter the kernels take the exact same path as before (one `Option` check per candidate; the scan
micro-benchmark, `thread bench-scan 3000 1536 192`, measures the same vectors/ms with and without the argument).

A selective filter leaves fewer matching entries in the `nprobe` cells a query would normally probe. Rather than
accept a shorter or worse result, a filtered query **widens its probes**: the routing step ranks cells to a bound
`maxNprobe` (default 4 x `nprobe`, at most every cell), phase 1 scans the first `nprobe` cells as usual, then keeps
scanning more cells in routing order, in batches of `max(4, nprobe / 2)`, until `T = max(topK, rerank)` distinct
candidates have passed the filter or the bound is reached. Delta segments are always scanned whole, so they need
no widening. The response reports the cells actually scanned (`stats.cells`), the candidates found and the
entries the filter dropped (`stats.candidates`, `stats.filterDropped`), and the bound in effect
(`params.maxNprobe`).

The knobs, and what they trade:

| knob | default | effect |
|---|---|---|
| `nprobe` | manifest default | the cells always scanned; recall for unselective filters |
| `maxNprobe` | 4 x nprobe | how far a selective filter may widen: bytes and latency against recall; `nprobe` disables widening |
| `rerank` | manifest default | with a filter, also the number of matching candidates the widening aims for |

What this means for cost: a filter that keeps a large fraction of the index (a language, a public flag) finds `T`
matches within the usual `nprobe` and costs nothing extra. A filter that keeps 1% of the index scans up to
`maxNprobe` cells, i.e. up to 4 x the bytes of an unfiltered query at the default bound, for a result whose
recall is measured against the matching documents. A filter that keeps a handful of documents is better served by
the application's own index of those ids. Unfiltered queries never widen and are unchanged.

Speculative rerank reads (see [query-path.md](query-path.md#the-two-phase-query)) fire once only a few cells of
the *last* batch remain; a filtered query knows a batch is last when it reaches the bound or already holds `T`
matching candidates.

## Text queries: post-filtering through the docs store

Postings carry no attrs (that would cost bytes on every posting of every term), so a text query filters after
scoring: the BM25 list is taken deeper than `k` (first `max(4 k, 32)`), each candidate's word is resolved through
the `docs` store (one bucket read per hit, coalesced, newest live copy, the same lookup `docs: true` does), the
non-matching candidates are dropped, and the list deepens by 4x until `k` match, the postings are exhausted or
`maxDepth` (65 536) is reached. Deeper lists cost heap size and docs lookups only: the posting lists of the query's
terms are read whole in any case. A hybrid search compiles the filter once and applies it to both lists before
fusion, so the fused result holds matching ids only.

`DocRecord` carries the word (`attrs`), so a hit resolved with `docs: true` can be decoded with `unpackAttrs`
without another read.

## Formats

- Slab and delta entries have carried the word since the first format; no index needs rebuilding. The index built
  before schemas existed has every word at 0.
- The `docs` section is now version 2: a bucket entry gains an `attrs` varint after the id when the section's
  flag bit 1 is set. A writer sets the bit only when some document has a non-zero word, so sections without attrs
  are byte-identical to version 1, and version 1 sections still read (every attrs 0). The container's open plan
  grew from 576 bytes to 1 KiB so a `meta` with the extra summary field still arrives in the first fetch.
- `manifest.attrs` holds the schema; `configureAttrs` writes it (`change.op: "configure-attrs"`).

## Not done

- No `$or` between fields; the clause list is a conjunction. A disjunction over one field is `$in`.
- A text query's filter is a post-filter: a filter that keeps very few of the documents matching the query terms
  deepens up to `maxDepth` and may still return fewer than `k`. Storing the word beside each posting would let
  the BM25 heap pre-filter at the cost of 1-5 bytes per posting; it was not worth it for the first version.
- Ranges on `number` fields are exact; there is no float or string-order comparison.
- The Worker takes `filter` (JSON in the query string or body) and `maxNprobe` and applies the same widening, but
  its `/upsert` takes raw `attrs` words only; pack them with the SDK's `packAttrs` or write through the SDK.
