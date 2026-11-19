# Limits and status

This page lists what thread does not do, the hard limits, and how mature each part is. Read it before you commit
to thread for a project.

## Status

thread is at version 0.2. It runs in production at [inseam](https://inseam.io). The format and the SDK calls can
still change between minor versions. [The changelog](../CHANGELOG.md) records every change.

| Part | Status |
|---|---|
| Vector queries, text search, hybrid search, filters | Tested end to end in both SDKs and in the reference Worker |
| Results from TypeScript and Python | Checked to be the same: ids, order, documents, compiled filters. Scores agree to about 1e-7. |
| Packs, snapshots, copies | Tested. The two SDKs write byte-identical packs. |
| Writes from the SDKs (`upsert`, `delete`) | Marked experimental. Used in production at inseam. |
| Writes through the S3 API | The conditional requests are documented by R2 and AWS S3 but not yet verified by thread against a live endpoint. Prefer the R2 binding or a file store for writers. |
| Compaction of vectors | TypeScript SDK and the reference Worker. Not in the Python SDK. |
| Bulk build (`thread.bulkBuild`) | Experimental. TypeScript only. |

## Hard limits

| Limit | Value |
|---|---|
| Document ids | Unsigned 32-bit integers, 0 to 4,294,967,294 (the last value is reserved) |
| Key length | 4,096 bytes, UTF-8 |
| Documents in one `upsert` | 4,096 |
| Chunks in one chunked `upsert` | 4,096 |
| Filter fields for each document | 32 bits in total |
| Distance | Cosine similarity only. Vectors are normalized. |
| A bulk build in one isolate | About 480,000 vectors at 1,024 dimensions with a 32 MiB budget |

Memory in a reader: the routing table (25 MB for 4,096 cells of 1,536 dimensions), a 1.5 MB codebook if the index
has one, and a few KB for each text segment. Everything else streams through.

## What thread does not do

**Search**

- No distance other than cosine.
- No `$or` in a filter. Conditions on different fields are joined with "and". Use `$in` for several values of one
  field.
- No filter on strings or large numbers. Filters test small fields: integers of a few bits, enums, booleans.
- A filter on a text search is applied after scoring. A filter that keeps very few of the matching documents can
  return fewer than `topK` results.
- A hit does not include the document's key. Put the key in `metas`.
- No stored vector in results. Keep your vectors in your own store if you need them back.
- No built-in reranking model. The arithmetic to merge a reranker's scores is there, the model call is yours.
- Text search has no phrase queries and no fuzzy matching.
- Chunking is by word count only. There is no sentence-aware or token-aware chunking.

**Writes**

- No transactions across documents or across indexes. One `upsert` call is atomic: all of it becomes visible, or
  none of it.
- A write is not visible to other readers at once. They see it after their next refresh: a search starts one
  every 30 seconds by default, and a lookup (`get`, `listKeys`) never does.
- A search is not a strict snapshot. If the handle refreshes while a search runs, the ranked lists and the
  documents can come from two revisions. Pin a handle to a revision when you need one.
- No insert that fails when the key exists. `upsert` always replaces.
- Many small writes are expensive. Batch them.
- Compaction is not automatic when you write through the SDKs. You call it.
- No call deletes every chunk of a document. List the keys by prefix, then delete them.

**Platform**

- thread runs no embedding model. It calls yours.
- There is no hosted service, no dashboard and no HTTP API, except the one you build. The reference Worker is an
  example and a benchmark tool, not a product.
- No access control. Whoever can read the bucket can read the index. Put your own authentication in front of
  your Worker.
- No encryption at rest beyond what your storage provides.

## Things that surprise people

| Surprise | Explanation |
|---|---|
| The first query of a new process is slow | The cache is empty and the routing table must load. See [Cold start](cold-start.md). |
| The bucket grows after deletes | Files are never changed in place. [Garbage collection](production.md#delete-old-files) removes old ones. |
| A hybrid score looks tiny, such as 0.03 | It is a rank score, `1 / (60 + rank)` for each list. Use it for order only. |
| Results differ a little from an exact search | The index is approximate: 95% recall on the 1M benchmark at the default settings. Raise `nprobe` for more. |
| `createGeneration` fails with 409 | The generation exists. Create is not idempotent by design, so that two creators cannot both succeed. |
| A tokenizer option cannot be changed | The tokenizer is fixed after the first write, because old and new documents must be cut the same way. Build a new generation. |

## Where to ask

thread is open source under the MIT licence. Open an issue or a pull request on
[GitHub](https://github.com/aiuitech/thread).
