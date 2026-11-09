# Lexical search and unified docs

thread searches by vector (the IVF index of `docs/architecture.md`) and by text (BM25), over one id space, and fuses the two
lists. This document is the design of the text side and of the document store: what the objects hold, how a query
reads them from blob storage, how updates and compaction work, and which knobs trade index size for query speed.
The code is `crates/thread-core/src/{source,container,tokenize,lex,docs,segment,fusion,excerpts,rerank}.rs`, the
bindings `crates/thread-wasm/src/lex.rs`, the host side `sdk/ts/src/{lex,write,index}.ts`, and the native CLI
`crates/thread-cli/src/lex_cli.rs`.

It is a port and reshaping of Wiper's `thread` engine (`../wiper/rust/thread`), which held BM25 postings, a
vocabulary, int8 vectors and documents in one `.thread` file read by range requests. What moved over unchanged: the
demand-paging `ByteSource` / `Prefetched` / `Plan` model (a query either succeeds or names every byte range its
current round still needs, coalesced at 4 KiB), the `THREADIX` container of named sections, the bm25s-compatible
tokenizer with the vendored Snowball 3.1 English stemmer, CJK bigrams, technical identifiers and Harman's light
stemmer, Reciprocal Rank Fusion, the excerpt cutter and the judge-side rank fusion. What was redesigned for a
blob store with sparse caller-assigned ids: the inverted index layout (below), the document store (keyed by id, not
by row), segments with tombstones instead of whole-file appends, and the fusion of the two indexes (planned and fused in the
core, the two lists fetched by the SDK).
What was left behind: the flat int8 vector sections (the IVF index covers that), document-range sharding over
Durable Objects and its skip lists, the SQLite/D1 byte sources, and the feature-gated ICU and charabia segmenters
(large dependencies; the names are rejected with a config error).

## Objects

Everything lexical lives in **segments**: immutable `THREADIX` containers next to the dense objects of a generation.

```
idx/{gen}/seg/{seq}.thread        one per upsert that carried texts or metadata (seq shared with delta/{seq}.vibd)
idx/{gen}/seg/base.{rev}.thread   written by a segment compaction that read manifest rev - 1
```

A segment holds up to five sections (`container.rs`; 64-byte header, 32-byte table entries, 8-byte aligned):

| section | content |
|---|---|
| `meta` | JSON: `format: "thread-seg"`, `version`, the `tokenizer` object, a `lex` summary, a `docs` summary, a `keys` summary |
| `terms` | the segment's terms sorted by bytes, in buckets behind a directory; each entry carries the term's `df` and the byte range of its posting list |
| `postings` | the posting lists, back to back |
| `docs` | document id -> metadata string (+ optional text), bucketed by id behind a directory |
| `keys` | unique string key -> document id, sorted by key bytes in buckets behind a directory ([keys.md](keys.md)) |

A segment may hold `terms` + `postings` only (a lexical index for documents named elsewhere), `docs` only (a document
store for a dense-only index), `keys` only, or any combination; an upsert that passed keys writes a `keys` section
(`segments[].keys: true`). The manifest lists them oldest first:

```json
"lex": {"tokenizer": {...}, "k1": 1.2, "b": 0.75, "lengths": "exact", "bucketTerms": 256, "docs": true, "text": false,
        "bucketDocs": 128, "bucketKeys": 128, "defaults": {"rrfK": 60, "recallK": 50, "weights": {"dense": 1, "lexical": 1}}},
"segments": [{"seq": 0, "key": "idx/g4/seg/base.12.thread", "addedRev": 12, "count": 975000, "bytes": 61234567,
              "lex": true, "docs": true, "text": false, "base": true},
             {"seq": 7, "key": "idx/g4/seg/7.thread", "addedRev": 14, "count": 250, "bytes": 31744, "lex": true, "docs": true, "text": false, "keys": true}],
"nextId": 975250
```

`manifest.lex` is the configuration every segment of the generation is built with (`index.configureLexical` writes
it as a manifest revision; the tokenizer is frozen once a segment exists, the other knobs apply to later segments). A
generation can be dense + lexical (`configureLexical` on a published vector index) or lexical only
(`createLexicalGeneration`: `k2 0, dim 0, vectors 0`, no base files; `query()` is refused, `search({text})` works).

## Formats

### `terms` (`lex.rs`, magic `THRDTERM`, version 1)

```
0   magic "THRDTERM"      8  version u32 = 1       12 term count T u32    16 bucket count B u32 = ceil(T / bucket_terms)
20  bucket_terms u32      24 num_docs u32          28 flags u32 (bit 0: lengths as SmallFloat bytes)
32  total_len u64         40 num_postings u64      48 k1 f64              56 b f64
64  dir_off u64 (= 96)    72 dir_len u64           80 buckets_off u64     88 buckets_len u64
directory, at dir_off: B x 16 bytes (bucket offset u64, bucket len u32, first-term offset u32 relative to the directory),
  then every bucket's first term back to back (full terms, so a binary search over the raw entries names exactly one bucket)
bucket entry: len varint, term (UTF-8), df varint, post_off varint, post_len varint   (offsets inside the postings section)
```

There are no term ids, no idf array and no per-term offset arrays: a bucket entry *is* the term's metadata. Looking
every query term up is one bucket read per term (coalesced) once the directory is held. `idf` is recomputed at
query time from `df` and the corpus size, which is what lets several segments score like one index.

### `postings` (magic `THRDPOST`, version 1)

```
0  magic "THRDPOST"   8 version u32 = 1   12 flags u32 (bit 0: lengths as bytes)   16 lists back to back
posting: gap << 1 | (tf != 1) as varint, then tf as varint when the flag is set, then the document's length:
         an exact varint (`lengths: "exact"`) or one byte (`lengths: "byte"`, Lucene's SmallFloat.intToByte4:
         exact below 40, then a 3-bit mantissa)
```

Document ids are the index's u32 ids (sparse, caller-assigned; `u32::MAX` is reserved), gaps start from `u32::MAX`
wrapping so the first gap is the id itself. The document length rides in every posting: scoring then reads
exactly the posting lists of the query's terms and nothing else. A length array indexed by sparse ids would be an
id -> length table of 8 bytes per document read whole per query, or resident per isolate; inline lengths cost
about one to two bytes per posting instead (the `byte` option one).

### `docs` (`docs.rs`, magic `THRDDOCS`, version 2)

```
0   magic "THRDDOCS"   8 version u32 = 2   12 n u32   16 bucket count B u32   20 bucket_docs u32   24 flags u32 (bit 0: texts stored, bit 1: attrs stored)
32  dir_off u64 (= 64)   40 dir_len u64 (= 16 B)   48 buckets_off u64   56 buckets_len u64
directory: B x (first id u32, bucket len u32, bucket offset u64), ascending
bucket entry: id varint, [attrs varint, with bit 1], meta len varint, meta (UTF-8), text len varint, [text offset varint]  (offset in the text region)
text region: align8(buckets end) .. section end
```

`meta` is whatever the application wants back with a hit (a title, a JSON object, an external key). Texts sit in
their own region so resolving a title never pays for the text; a hit's text is one more range read, only when asked
for. `attrs` is the document's 32-bit attribute word, the same one its dense entry carries ([filtering.md](filtering.md));
a text query resolves it here to apply a filter, and a writer sets bit 1 only when some document's word is non-zero
(version 1 sections, without the bit, still read with every word 0). This is the one document store of the index:
dense hits and lexical hits resolve through the same section. (The dense slabs still carry their per-entry `attrs`
and `payload`; `docs` is the recommended place for anything larger than a few bytes, since slab payloads are
duplicated `m` times.)

### `keys` (`keys.rs`, magic `THRDKEYS`, version 1)

```
0   magic "THRDKEYS"   8 version u32 = 1   12 n u32   16 bucket count B u32   20 bucket_keys u32   24 flags u32 = 0
32  dir_off u64 (= 64)   40 dir_len u64   48 buckets_off u64   56 buckets_len u64
directory: B x 16 bytes (bucket offset u64, bucket len u32, first-key offset u32 relative to the directory), then every
  bucket's first key back to back (full keys: a binary search over the directory names exactly one bucket)
bucket entry: key len varint, key (UTF-8), id varint
```

The `terms` directory with an id in place of a posting range. Keys are unique within a section (non-empty, at most
4096 bytes); across segments the newest live entry of a key wins, so a re-put of a key under the same id replaces its
document and a deleted id hides its key, through the one tombstone rule. A lookup is one bucket read per key
(coalesced) after the directory; a prefix listing is a binary search to the first bucket and a forward scan over
contiguous buckets (one range). The design, the id counter (`manifest.nextId`) and the key-value use case:
[keys.md](keys.md).

### Container and `meta`

Sections are written `meta`, `terms`, `postings`, `docs`, `keys`. `meta` records the tokenizer config (the core
parses only that key back) and a summary of each section:

```json
{"format":"thread-seg","version":1,
 "tokenizer":{"lowercase":true,"token_pattern":"\\w\\w+","stopwords":"en","stemmer":"english"},
 "lex":{"version":1,"num_docs":20000,"terms":29999,"num_postings":2353244,"total_len":3994184,"k1":1.2,"b":0.75,"lengths":"exact","bucket_terms":256,"terms_nbytes":383492,"postings_nbytes":8545861},
 "docs":{"version":1,"n":20000,"text":false,"bucket_docs":128,"nbytes":368168},
 "keys":{"version":1,"n":20000,"bucket_keys":128,"nbytes":1364120}}
```

## Query

Scoring is bm25s's (Lucene's) expression: `Σ_terms idf · tf / (k1 · ((1 - b) + b · len / avg_len) + tf)` with
`idf = ln(1 + (N - df + 0.5) / (df + 0.5))` as float32, the tf component in float64, each product cast to float32
and summed in float32; results are the top k with a positive score, ties by ascending id. `N`, `avg_len` and `df`
are **corpus-wide**: the host sums `num_docs` / `total_len` over the segments and each term's `df` over the
segments' lookups, then scores every segment with those numbers (`Postings::search` takes the idfs), and merges.
A corpus split over segments therefore scores exactly like the same corpus in one segment (checked by
`lex::tests::segments_score_like_one_index_and_merge_back`). Tombstoned documents keep counting in `N` and
`avg_len` until a compaction, as deleted documents do in Lucene.

A query over one segment on a cold isolate, in rounds of range reads (each round one multi-range request against
the Cache API, or one request per range on a miss):

| round | reads | kept |
|---|---|---|
| open | container header + table + `meta` (the first 1 KB) | per isolate |
| headers | `terms` (96 B), `postings` (16 B), `docs` (64 B) headers | per isolate |
| directories | terms directory (~22 B per bucket) + docs directory (16 B per bucket) | per isolate |
| lookup | one bucket per distinct query term (~3-5 KB each at 256 terms per bucket) | transient |
| search | one posting list per query term that exists | transient |
| docs | one bucket per hit id (coalesced), then one text range per text shown | transient |
| keys | one bucket per key looked up (coalesced); a listing one contiguous range | transient |

So a warm isolate answers a text query in **two rounds per segment**, all segments concurrently, resolves docs
in one or two more, and a `get(key)` in two (keys bucket, docs bucket). Transient bytes are evicted from the wasm reader once they exceed a budget (8 MB by default);
the resident parts are a few KB per segment (20k documents: 2.5 KB terms directory + 2.5 KB docs directory).

### Block cache

Behind the Workers Cache API (`CachedBlobStore`, what `thread.open` puts in front of an R2 bucket), a segment of up
to 2 MiB (`LexSegmentOptions.fillBytes`) is read whole on its first miss and cached whole, like a slab. A larger
segment is read by range with its size (`GetOptions.size`), and the store serves those ranges from **blocks**:
block `i` is bytes `[i * B, min((i + 1) * B, size))` of the segment, cached under its own key
(`{origin}{scope}{key}/~block/{B}/{i}`, so indexes under other prefixes never share one). A read looks up each block
it covers once (its ranges in that block as one range request); a block that misses is read from R2 (that block
only, never the whole segment), serves the read, and is put in the cache through `waitUntil`. Concurrent readers of
a block share one R2 read; at most `blockBuffers` blocks (16 MiB worth) are held from their R2 read until their put
settles, and block reads from R2 in flight are also at most the store's `inflight`. Each range is copied out of its
block before the block's slot is given back, so the memory a read keeps alive is the bytes it asked for plus at most
`blockBuffers` blocks in flight (0.2.5 returned views into the blocks: 160 ranges of 16 bytes held 160 MiB, whatever
`blockBuffers` was). An object already cached whole
(before 0.2.5, or by `index.warm`) is still served from that entry. Slabs, deltas, manifests and the routing files
keep whole-object caching: blocks only serve range reads that pass a size and are not a `fill`.

Before 0.2.5 these reads went to R2 on every search: a segment over 2 MiB was never cached (a range read without
`fill` that missed was read from the origin and not cached), so merging segments made search slower (inseam,
2026-11-09: 39 segments of 4 MiB, 293 R2 reads and 4.2 s in the lexical list on a warm cache).

Measured with `bench/blocks/` (Workers on workers.dev, R2 `vib-bench`, one `thread.open` per request as inseam does,
hybrid search with `docs: "text"`, medians of 3; "new terms" is another query on the same warm cache):

| index | SDK, B | cold search | warm search | warm R2 reads | new terms | new terms R2 reads |
|---|---|---|---|---|---|---|
| 8 segments of 4.1 MB, 20k docs | 0.2.4 | 4.6 s | 4.3 s | 125 | 3.2 s | 127 |
| | 256 KiB | 2.2 s | 0.27 s | 0 | 1.5 s | 23 |
| | 1 MiB | 2.0 s | 0.20 s | 0 | 0.41 s | 1 |
| | 4 MiB | 1.5 s | 0.18 s | 0 | 0.19 s | 0 |
| 2 segments of 46 MB, 8k docs | 0.2.4 | 1.9 s | 1.3 s | 54 | 1.9 s | 51 |
| | 256 KiB | 1.0 s | 0.14 s | 0 | 0.96 s | 23 |
| | 1 MiB | 1.9 s | 0.10 s | 0 | 0.60 s | 11 |
| | 2 MiB | 2.4 s | 0.11 s | 0 | 1.1 s | 8 |
| | 4 MiB | 2.8 s | 0.12 s | 0 | 1.5 s | 5 |

`B` defaults to 1 MiB (`blockBytes`). Large blocks win when a segment is one or two blocks; on large segments each
miss reads a whole block for a few KB of postings, so 4 MiB read 62 MB from R2 for a cold search of the 46 MB
segments (1 MiB: 26 MB) and was the slowest cold. 1 MiB was best or close on both shapes for a new query on a
warm cache, the common case once an index has been searched. A dense-only search is unchanged (the same 16 R2
reads cold, 18 cache hits and ~20 ms warm, with 0.2.4 and 0.2.5). In the trace envelope a block lookup is a `hit`
or a `miss`, a fill is also a `read`, its whole block is in `bytes`, and `io.fills` counts the fills.

`search` is planned and fused in the core (`fusion.rs`); the SDKs only fetch the lists in between.
`plan_search` resolves the request against `lex.defaults` (request value, else manifest value, else `rrfK` 60,
`recallK` max(topK, 50), weights 1 / 1, `topK` 10), picks the mode and the depth of each list. The dense query
and the lexical query then run concurrently at that depth, and `fuse_search` fuses the two id lists with
Reciprocal Rank Fusion (`score = Σ weight / (rrfK + rank)`, float64, ties by id), or cuts the one list to
`topK`; the top k is resolved through `docs` when asked. Each hit reports its rank and score in
each list it appeared in. With an attribute filter the lexical list is taken deeper, each candidate's word is
resolved through `docs` and non-matching candidates are dropped before fusion ([filtering.md](filtering.md#text-queries-post-filtering-through-the-docs-store)).

Tombstones use the dense index's rule for deltas: a segment added at manifest rev `a` is filtered by the union of
the tombstone sets with `rev > a` (`tombForSegment`), inside the scan before top-k insertion. Replace semantics
come for free: an upsert of an existing id writes the new copy in a new segment and tombstones the id at that rev,
which hides the older copies (in older segments and in the dense slabs/deltas) but not the new one. Docs resolve
newest segment first, skipping hidden copies.

## Updates and compaction

`index.upsert({ids?, keys?, vectors?, texts?, metas?, attrs?})` writes, in one manifest revision: a dense delta segment for
the vectors (as before), a lexical/docs/keys segment for the texts, metadata, attribute words and keys (`SegmentJob`:
tokenize, build `terms` + `postings` + `docs` + `keys`), and the tombstone set. With `keys` and no `ids`, the keys are
resolved through the head revision: known keys keep their ids, new keys take the next ones from `manifest.nextId`
([keys.md](keys.md#ids-allocation-and-the-counter)). Vectors are optional when the generation has a lexical configuration
(text-only documents are found by text alone), texts/metas are optional (dense-only upserts are unchanged). The
delta and the segment share the `seq`. `index.delete(ids)` is one tombstone set, as before, and hides both sides.

`index.compactSegments()` reads every segment of the head whole, merges them (`segment::merge`: posting lists
concatenated per term with tombstoned documents dropped and documents re-sorted, docs with the newest copy of each
id, texts kept when configured, keys with the newest copy of each key) into `seg/base.{rev}.thread` and commits a
revision listing only it. The merged
`terms`/`postings` are byte-identical to a fresh build over the live documents (checked); a document with no tokens
leaves the lexical count then (it stays in `docs`). Segment compaction drops the tombstone sets only the lexical
side needed once no segment predates them: sets scoped `lexical` ([scoped tombstone sets](update-path.md#scoped-tombstone-sets-one-sided-updates)),
and in a generation without a dense index every set; the others wait for the dense compaction, whose rule keeps a
set while any segment predates it (`tombDroppable`). Segments apply the lexical view of the sets (unscoped and
`lexical`). The Worker's GC keeps every segment a kept revision lists. Compaction reads the segments whole, so it runs where memory allows (a build
slot or a laptop with the CLI, `thread seg merge`), not in a query isolate, for large bases.

`compactSegments({ maxInputBytes })` / `compact_segments(max_input_bytes=)` bounds one call for a memory-limited
writer (a Durable Object): it merges the first contiguous run of at least two segments whose bytes, plus 4 bytes per
tombstoned id (every set is read), fit the budget, and leaves the segments before and after it in place. The merged
segment takes the run's newest `addedRev`, so tombstone sets newer than the run still hide it and the copies in later
segments still win. No pair fits: nothing is written (`merged` 0). Peak memory is about three times the budget (the
parts, their concatenation, its copy in wasm memory) plus the merged output; repeated calls with the same budget fold
small segments together until no two adjacent ones fit.

`index.commitMergedSegment({ run, readRev, key, bytes, count, keys })` (TypeScript) splits a compaction between
two processes: one with memory reads a run of manifest `readRev` whole, applies to each segment the lexical view of
the sets newer than its `addedRev` (`tombForSegment`), merges with thread-core `segment::merge` (what wasm
`merge_segments` wraps) and writes the result under `idx/{gen}/seg/`; the index's writer then commits it in place of
the run. The writer opens the object (container, section headers, directories) and checks its size first. Nothing
is written (`merged` 0) when the head no longer lists the run contiguously. The merged segment takes `addedRev =
readRev`: sets newer than the plan still hide its ids, and the sets it applied drop by the rule above.

## Tuning: size or query speed

Every knob is a build parameter in `manifest.lex` (or a flag of `thread seg build`); segments record what they were
built with in their headers.

| knob | default | smaller index | faster / cheaper queries |
|---|---|---|---|
| `lengths` | `exact` | `byte`: one length byte per posting (-0.9 B/posting below; scores within a few percent) | `exact`: exact BM25 scores |
| `bucketTerms` | 256 | larger buckets: a smaller resident directory (22 B per bucket) | smaller buckets: fewer bytes per query term (an entry is ~12 B) |
| `bucketDocs` | 128 | larger buckets: a smaller resident docs directory (16 B per bucket) | smaller buckets: fewer bytes per resolved hit |
| `bucketKeys` | 128 | larger buckets: a smaller resident keys directory (16 B + one key per bucket) | smaller buckets: fewer bytes per key lookup |
| `docs` | true | false: no document store (ids only) | |
| `text` | false | false: texts stay outside the index | true: texts available for excerpts and reading, one range read each |
| `tokenizer` | bm25s defaults | `stopwords: "en"` drops 33 words; `stemmer: "english"` merges inflections (fewer terms) | `identifiers: "both"` indexes `INC-12345` as the exact token and its parts |
| `defaults.recallK` | max(topK, 50) | | smaller: fewer bytes per list (but less overlap for fusion) |
| segments | | compaction: one base, one lookup + one list per term | many small segments cost a round of reads each (concurrent) |

Measured on a synthetic corpus (`thread seg build`, 20,000 documents of ~200 tokens, a 30k-word Zipfian
vocabulary, ids spaced by 5, no stemming or stopwords; `data/` is not needed):

| | exact | byte |
|---|---:|---:|
| postings (2,353,244) | 8.55 MB, 3.63 B/posting | 6.34 MB, 2.69 B/posting |
| terms (29,999 terms, 118 buckets) | 383 KB, directory 2.5 KB | 378 KB |
| docs (20,000, metadata only, 157 buckets) | 368 KB, directory 2.5 KB | same |
| build (tokenize + write, one core) | 0.54 s | 0.52 s |
| query of 4 terms (df 19,778 / 13,379 / 1,245 / 28), native release | 4.6 ms | |

The posting bytes split into about 1.6 B of gap + tf flag, 0.3 B of tf and 1.7 B (exact) or 1 B (byte) of length
at this document length; denser ids (a corpus numbered 0..N) and shorter documents bring the gap and the length
down to one byte each. Wiper's row-numbered index with a separate length array measured 1.39 B/posting on a real
192k-chunk corpus; the difference is the inline length and the sparse ids, which is the price of needing no
corpus-sized array per query.

## Tokenizer

`tokenize.rs` reproduces `bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"))` token for
token by default (lowercase, maximal `\w` runs of two or more chars, 33 English stopwords dropped before stemming,
Snowball 3.1 English), and takes the options Wiper added, as keys of the `tokenizer` object:

| key | values (default first) |
|---|---|
| `lowercase` | `true`, `false` |
| `token_pattern` | `"\\w\\w+"`, `"\\w+"`, `"\\w{N,}"` (the minimum token length) |
| `stopwords` | `"en"`, `null` |
| `stemmer` | `"english"` (Snowball 3.1), `"english-light"` (Harman's S-stemmer), `null` |
| `segmenter` | `"words"`, `"cjk-bigram"` (Han/Kana/Hangul runs as overlapping bigrams) |
| `identifiers` | `"off"`, `"split"`, `"keep"`, `"both"` (technical runs like `INC-12345`, `getUserName`, `k8s.io`) |

The query goes through the same tokenizer as the documents (the config travels in the manifest and in every
segment's `meta`), so `deployments` and `deployment` meet at `deploy`.

## Excerpts and judge-side reranking

`excerpts.rs` (the line grid, `query_terms`, `find`) and `rerank.rs` (`fuse`, `best_doc_ranks`) came over as pure
functions, exposed to JS as `Lines`, `Excerpter`, `parse_ranges`, `best_doc_ranks`, `rerank_fuse`: a host that
stores texts (`text: true`) can cut the passages of the top documents around the query terms and fuse a judge's
passage scores (a cross-encoder, an LLM) with the fused rank and the per-list ranks. The SDK does not call a judge
itself; the arithmetic is there for the host that does.

## API summary

```ts
const index = await thread.open({ bucket: env.BUCKET });
await index.configureLexical({ text: true });                         // once per generation (a manifest revision)
await index.upsert({ ids, vectors, texts, metas });                   // dense delta + lexical/docs segment + tombstones
await index.upsert({ ids, texts, metas });                            // text-only documents
const r = await index.search({ text: "deploy failed", vector: q, topK: 10, docs: "text" });
r.hits[0]                       // { id, score, dense?: {rank, score}, lexical?: {rank, score}, doc?: {id, meta, text} }
await index.lexicalSearch("deploy failed", 20);                       // BM25 only: { hits, tokens, found, stats }
await index.docs([1, 2, 3], { text: true });                          // Map<id, {id, meta, text}>
await index.upsert({ keys: ["s3://b/a", "s3://b/c"], metas, texts });  // keyed documents: ids resolved or allocated
await index.get(["s3://b/a"]);                                        // Map<key, {key, id, meta, attrs, text?}>
await index.listKeys({ prefix: "s3://b/", limit: 100 });              // {entries: [{key, id}], more, next}
await index.deleteKeys(["s3://b/c"]);
await index.compactSegments();                                        // merge the segments into one base
await createLexicalGeneration({ store, gen: "docs1", lex: { text: true } });   // a text-only index
```

CLI (`cargo build --release -p thread-cli`, `target/release/thread`):

```sh
thread seg build --out seg.thread --docs docs.jsonl [--tokenizer '{"stemmer":null}'] [--lengths byte] [--bucket-terms 256] [--text] [--keys]
thread seg inspect seg.thread
thread seg search --text "deploy failed" --k 10 --with-docs [--tombstones dead.u32,-] seg.thread older.thread
thread seg keys --get "s3://b/a,s3://b/c" --prefix "s3://b/" [--after k] [--limit 100] seg1.thread seg2.thread
thread seg merge --out base.thread [--tombstones a.u32,b.u32] seg1.thread seg2.thread
thread tokenize [--tokenizer '{"identifiers":"both"}'] "INC-12345 failed"
thread chunk [--words 200] [--overlap 0] [--file report.txt] "a long text ..."    # flat word windows, see chunking.md
thread seg build --out seg.thread --docs docs.jsonl --chunk-words 200 --chunk-overlap 20 --keys --text   # one doc per chunk, keyed k#i
```

`docs.jsonl` is one object per line with an integer `id`, a `text`, an optional `attrs` (the u32 attribute word),
an optional `meta` (a string, or any JSON value stored as its text) and, with `--keys`, a string `key`. A segment file built here is byte-for-byte what an upsert writes, so a corpus can be
indexed locally and uploaded as `idx/{gen}/seg/base.0.thread` with a matching `segments` entry in rev 0.

## Not done

- The lexical base of a large corpus is one object; a query range-reads it, but a compaction rewrites it whole.
  Partitioning a base into several objects (by term range) would bound compaction memory.
- No block-max pruning: a query reads the whole posting list of each term. A per-block maximum score would let a
  query skip blocks (and their reads) that cannot enter the top k; it is the natural next size-vs-speed knob.
- Wiper's `search_range` sharding over Durable Objects, its SQLite/D1 byte sources and the ICU/charabia
  segmenters were not carried over.
- Fusion is RRF only; the judge-based reranker is exposed as arithmetic, not wired to a model.
- Keys: no reverse lookup (id -> key), no Bloom filter for negative lookups ([keys.md](keys.md#not-done)).
