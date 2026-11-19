# Run it in production

thread has no server, so there is little to operate. This page is the short list of things to get right before
you serve real traffic. Each item says what to do and why.

## Checklist

| Do this | Why |
|---|---|
| [Open the index once](#open-the-index-once) | Opening loads a routing table. Do it once for each isolate or process, not for each request. |
| [Batch your writes](#batch-your-writes) | A write of 1 document costs the same requests as a write of 1,000. |
| [Write from one place](#write-from-one-place) | Fewer retries, and write memory stays out of your query path. |
| [Run compaction](#run-compaction) | Without it, every write makes every later query a little slower. |
| [Delete old files](#delete-old-files) | Compaction leaves the files it replaced in the bucket. |
| [Know your freshness](#know-your-freshness) | Other readers see a write after their next refresh, not at once. |
| [Watch the numbers](#watch-the-numbers) | Every result carries its own timings and I/O counts. |
| [Keep a copy](#keep-a-copy) | The index is files. A backup is a copy. |

## Open the index once

`Thread.init` compiles the engine, and `open` reads the manifest and the routing table. Keep both for the life of
the isolate or process.

```ts
let index: Promise<ThreadIndex> | undefined;

function getIndex(env: Env, ctx: ExecutionContext) {
  if (!index) {
    index = Thread.init({ wasmModule }).then((thread) =>
      thread.open({ bucket: env.BUCKET, embed: { ai: env.AI }, waitUntil: (p) => ctx.waitUntil(p) }),
    );
    index.catch(() => (index = undefined));   // do not keep a failed open
  }
  return index;
}
```

In Python, open one index at start-up and share it. A `ThreadIndex` is safe to use from several threads. Its
calls block, so from asyncio run them in a thread: `await asyncio.to_thread(index.search, text=q)`.

An open index follows new revisions by itself ([Know your freshness](#know-your-freshness)). You do not need to
open it again after a write.

## Batch your writes

Every `upsert` or `delete` is one revision: about ten requests to the store, a new manifest, and new small files
that queries must read until compaction merges them.

- Send at least 64 documents in a call, or everything that arrived in the last second.
- Send one call for one logical unit (a crawled page, an imported file), not one call for each row.
- One call holds at most 4,096 documents.

If documents arrive one at a time, put a queue in front of the writer (a Cloudflare Queue, a buffer in a Durable
Object) and write each batch.

## Write from one place

Concurrent writers are safe on a bucket. The store accepts one new revision, and the other writer builds on it
and tries again. But a writer that loses the race does its work twice.

One exception: do not write to the same **folder on disk** from Node and from Python at the same time. The two
SDKs lock the pointer file in different ways, and a write can be lost. Writers of one language, or writers that
take turns, are fine.

A good design is one writer for each index:

```mermaid
flowchart LR
    A[Your app] -->|documents| Q[Queue]
    Q -->|batches| W[Writer<br/>a Durable Object or a queue consumer]
    W -->|upsert, compact| B[(Bucket)]
    B -->|cached range reads| R[Query Workers]
    U[Users] -->|search| R
```

In Workers, keep writes out of the isolates that answer queries. A write holds the routing table twice, and wasm
memory never shrinks, so a query isolate that also writes keeps that memory for the rest of its life. A Durable
Object or a separate Worker for writes avoids this.

For writers, prefer the R2 binding or a file store. The S3 path uses conditional requests (`If-None-Match`,
`If-Match`), which R2 and AWS S3 document, but thread has not verified them against a live endpoint yet.

## Run compaction

Compaction merges the small files that writes leave behind, and splits cells that grew too large. Who runs it
depends on how you write:

| You write through | Compaction |
|---|---|
| The reference Worker (`/upsert`, `/delete`) | Automatic. The Worker starts it after a write that passes a threshold. |
| The TypeScript SDK | You call it. A few lines, below. |
| The Python SDK | Python does not compact vectors. Run the TypeScript lines from a small script or a Worker, on a schedule. |

In the writer, after a write:

```ts
// Vectors: merge delta files into the cells, split large cells
if (index.compactionDue().due) {
  for (let i = 0; i < 8; i++) {                           // a few bounded slices, then stop
    const run = await index.compact();
    if (!run.more || run.outcome === "locked") break;     // locked: another run has the work
  }
}
```

Do not loop until `more` is false without a limit. When another writer holds the lock, every call returns
`outcome: "locked"` with `more: true` at once, and a lock that a crashed writer left stays for ten minutes. Stop,
and let the next write or the next alarm try again.

`compactionDue()` reads nothing from the store. It compares the open revision with the thresholds of the index:
more than 16 delta files, more than 32 MB of them, or more than 50,000 replaced or deleted ids.

Each `compact()` call is bounded in memory (48 MiB by default), in cells and in time (27 seconds), so it fits a
Worker or a Durable Object alarm. It takes a lock in the bucket, so two compactions never run at once. If a call
is interrupted, a later one continues. Queries during a compaction are not affected.

The text side has its own merge. Run it from time to time, for example after every 16 writes or once a day:

```ts
await index.compactSegments();           // TypeScript
```

```python
index.compact_segments()                 # Python can do this one
```

Details and every option: [compaction in the SDK](update-path.md#compaction-in-the-sdk).

## Delete old files

Nothing is deleted when you write, compact or delete documents. Old revisions stay readable, which is what makes
reads consistent and rollbacks possible. Garbage collection removes files that no recent revision uses.

```ts
const report = await index.gc();                                    // a dry run: what would be deleted
await index.gc({ keep: 2, graceSec: 300, dryRun: false });          // keep the 2 newest revisions
```

Run it on a schedule, for example once a day. `graceSec` protects files that a writer made in the last few
minutes and has not committed yet.

## Know your freshness

| Event | Visible to |
|---|---|
| A write | The handle that wrote: at once. Other open indexes: after their next refresh. |
| `await index.refresh()` | That index: at once |
| A switch to a new generation | Readers that follow the pointer: after their next refresh |

How an open index refreshes:

- A `search` or a `query` checks the pointer in the background when `refreshInterval` (30 seconds by default) has
  passed. That call still answers from the revision it has. Later calls use the new one.
- `get`, `listKeys` and `docs` do not start a refresh. A handle that only does lookups stays on its revision until
  you call `refresh()`.

If a user must see their own write at once (read-your-writes), run their next call on the handle that wrote. On
another handle, refresh until it has reached the revision that the write returned:

```ts
const written = await writer.upsert({ keys, texts });
while (reader.rev < written.rev) await reader.refresh();     // one refresh is usually enough
```

A shorter `refreshInterval` gives faster visibility for one more small read per interval.

A search is not a strict snapshot when the handle refreshes while it runs: the ranked lists can come from one
revision and the documents (`docs: true`) from the next. If every part of a result must come from one revision,
open a handle with `gen` and `rev`, which never refreshes.

## Watch the numbers

Every result has a `stats` object: the time of each phase, the bytes read, the cache hits and misses. Every
operation also produces a **trace**, one JSON object with the same shape in both SDKs.

```ts
const index = await thread.open({
  bucket: env.BUCKET,
  onTrace: (trace) => console.log(JSON.stringify(trace)),   // one line for each sampled operation
  traceSample: 0.01,                                         // 1% of operations
});
```

What to watch:

| Signal | Healthy | If not |
|---|---|---|
| Cache misses of a search (`result.stats.trace.io.misses`) | Near 0 after warm-up | Traffic is too low to keep the edge cache warm, or the revision changes very often |
| Delta files in the revision (`compactionDue().pressure.deltas`) | Under 16 | Compaction is not running, or writes are too small |
| Write conflicts (`result.conflicts`) | 0 | More than one writer. Use one writer. |
| Time of a search (`result.stats.timing.total`) | Stable | Read the phases to find which step grew |

[Observability](observability.md) explains traces, sinks and how to find a bottleneck.

## Keep a copy

```ts
import { copyObjects, setCurrent, snapshotKeys } from "@aiuitech/thread-sdk";

// One consistent revision, to another bucket or a folder
const revision = index.revision;                                          // take it once: the handle can move on
await copyObjects(source, backup, snapshotKeys(revision));                // the files of that revision
await setCurrent(backup, { gen: revision.gen, rev: revision.rev });       // then a pointer to exactly it
```

Do not copy `idx/current.json` from the source. It can point to a newer revision than the files you copied.

Or pack one revision into a single file: [Move an index](portability.md).

An earlier revision stays readable while garbage collection has not removed its files:
`thread.open({ store, gen: "docs", rev: 41 })` opens revision 41 and stays on it.

To make an earlier state the live one again, restore the backup into an **empty** prefix and point your readers
and writers at that prefix. Do not copy a backup over the live prefix: the newer manifests are still there, and
the next write builds on them, so the newer documents come back. The reference Worker's
[`/build/rollback`](update-path.md#write-path) does a rollback in place, as a new revision.

## Cold starts

The first query of a new isolate or process reads the routing table and finds an empty cache, so it is slower
than the others. With steady traffic on Workers you will rarely see it, because the edge cache is shared by all
isolates.

If cold starts matter (low traffic, or a process that starts for each job), these help, in this order:

1. A disk cache, for Node and Python ([Connect to storage](storage.md#caches)).
2. `routing: "i8"`, which loads a routing table 4 times smaller.
3. A pack file on local disk.

[Cold start](cold-start.md) has the measurements for each option.

## Limits to plan for

- Document ids are unsigned 32-bit integers: about 4.2 billion for each index.
- One call writes at most 4,096 documents.
- A filter can use 32 bits of fields for each document.
- A Worker isolate has 128 MB. The routing table of a 1-million-vector index uses 25 MB of it.

[Limits and status](limits.md) has the full list.
