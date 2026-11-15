# TODO

Nothing in thread is open. Finished work is in `CHANGELOG.md` (scoped tombstones: 0.2.1; the compaction read phase
and tombstone set merging: 0.2.7; optional vectors per document: 0.2.8) and in `docs/update-path.md`.

## Check after the release

- The read phase's constants are reasoned, not measured against R2 under a backlog: delta windows of 8, the slice
  full after 16 skipped in a row, a third of `deadline - 2.5 s` for reads, deltas up to 64 KiB read whole. The same
  goes for the merge: runs of 8 or more sets, at most 256 extra reads a call, none in the last second. Look at
  `CompactResult.reads` and `mergedTombstoneSets` in inseam's "dense compaction" log lines and tune if they are off.
  - inseam, 2026-11-15, the first figures from production (0.2.7, one call, on a new Google account host; there is
    no backlog now, because the host of 2026-11-14 was deleted): 17 deltas read and 17 folded, 52 tombstone sets
    read, 32 merged, 22 left, 888 ms to the plan (the read budget is 2,500 ms), 2 cells, committed. Nothing to
    tune from this. A backlog is still unmeasured.
- inseam took both of its open items: one write per crawl page (inseam 7807db2) and 0.2.7 without the deadline
  workaround (inseam 67f3c51). It keeps one dense compaction a minute per index after writes, because a call that
  folds the oldest delta of a backlog still reads every tombstone set newer than it.
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
