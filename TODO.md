# TODO

Nothing in thread is open. Finished work is in `CHANGELOG.md` (scoped tombstones: 0.2.1; the compaction read phase
and tombstone set merging: 0.2.7) and in `docs/update-path.md`.

## Open, outside this repo

- [ ] **inseam: write once per crawl page, not once per read batch.** The crawl of a cloud host indexes each page
      (200 entries) in batches of up to 64 reads, and each batch is one upsert. On 2026-11-14 that made 767 deltas of
      about 3 KB and 2,531 tombstone sets at index revision 2,803. Keep the extracted text (not the bytes) across the
      page's batches and apply it in one write. `docs/update-path.md` "Small writes" gives the cost of a write and
      the minimum batch (64 documents, or what arrived in the last second). Track this in inseam.
- [ ] **inseam: take the next SDK release and drop the workaround** (inseam 73588f0: a deadline that grows with the
      read phase, at most one compaction a minute). `compact()` now folds at least the oldest delta on every call.

## Check after the release

- The read phase's constants are reasoned, not measured against R2: delta windows of 8, the slice full after 16
  skipped in a row, a third of `deadline - 2.5 s` for reads, deltas up to 64 KiB read whole. The same goes for the
  merge: runs of 8 or more sets, at most 256 extra reads a call, none in the last second. Look at
  `CompactResult.reads` and `mergedTombstoneSets` in inseam's `compact.last.json` and tune if they are off.
- A call whose reads pass the deadline still runs the oldest delta's cells, so it can run past `deadlineMs`. That
  is the progress guarantee. Callers with a hard limit (a Durable Object alarm) should leave room for it.

## Not worth doing now

- **A delta-coalescing step in compaction.** Folding deltas into slabs already coalesces them, and one call folds
  many tiny deltas per cell (83 one-vector deltas into 60 cells in one call). The cost of small writes is on the
  writer's side: batch them.
- **Streaming segment merge.** It would help a large base whose single segments pass `maxInputBytes`. Nobody has
  hit that. Segment compaction already takes bounded partial merges.
- **Merging tombstone sets in segment compaction.** It already removes the segment barrier, and the next dense
  compaction then merges the sets.
- **Tiering merged tombstone sets.** A large merged set is rewritten each time 7 or more new sets join its run. The
  sets are small (ids, 4 bytes each) and a purge drops them at `maxTombstones`. Revisit if a merged set grows past
  a few MB.
- **Attrs overlay** (cheap attrs changes on dense documents under `scope: "lexical"`). 0.2.1 rejects such a change;
  the caller re-sends the vector with `scope: "both"`. Revisit if a caller needs it.
- **Unforced Worker rollback across a merge.** `/build/rollback` to a revision before a tombstone merge needs
  `force`, because the head lists the merged key and the old revision does not. It works with `force`.
