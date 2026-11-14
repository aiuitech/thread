# Incremental updates: manifest revisions, delta segments, tombstones, compaction

Status: implemented in the Worker (`examples/cloudflare-worker/src/index.ts`) on top of the Rust delta kernels (`DeltaJob`,
`scan_delta`, `delta_records_for_cell`, `slab_to_records`, `filter_hits`, `split_slab`, `split_slab_records`, ...),
including automatic compaction after writes and cell splitting; the TypeScript SDK runs the same compaction in one
isolate ([Compaction in the SDK](#compaction-in-the-sdk)). Benchmarks: `bench/update_bench.mjs` ->
`data/bench-updates.txt` (and `--only trigger` -> `data/bench-trigger.txt`), `bench/split_bench.mjs` ->
`data/bench-split.txt`.

## Summary

Every change is a new manifest revision `idx/{gen}/manifest.{rev}.json` (rev 0 = `manifest.json` from `publish`);
`idx/current.json` is `{gen, rev}`. A revision lists its files (`coarse`, `fine`, `pq`, slabs table), its delta
segments (`delta/{seq}.vibd`, each with the rev that added it) and its tombstone sets (`tombstones.{rev}.u32`, the
ids deleted or replaced at that rev, or a set the SDK's compaction merged from several, see
[Merging tombstone sets](#merging-tombstone-sets)). Nothing is overwritten, so every object can stay in the edge
cache for a year and a query always reads a consistent snapshot.

- `POST /upsert` (JSON `{ids, vectors, attrs?, payloads?}` or a binary f32 body with `x-ids`/`?firstId=`, up to
  4096 vectors): one delta segment (`DeltaJob`: primary cell, found through the centroid tree when the manifest has
  one, + 4-bit/int8/PQ codes against the cell mean, each entry's 32-bit `attrs` word for filtering) and a tombstone
  set for the ids (replace semantics: older copies stop resolving), committed as rev + 1. Responds
  `{gen, rev, seq, cells, ms, bytes, conflicts}`. The SDKs' `upsert` also packs named fields into the word under
  `manifest.attrs` ([filtering.md](filtering.md)).
- `POST /delete {ids}`: one tombstone set, rev + 1.
- Writers serialize on a create-only PUT of `manifest.{rev+1}.json` (R2 `onlyIf: {etagDoesNotMatch: "*"}`),
  retry on a lost race, and advance `current.json` with an etag compare-and-swap.
- Queries scan every delta segment with `scan_delta` (int8 scores, no phase 2) next to the probed slabs and apply
  the tombstone rule: a slab hit is dropped if its id is in a set with rev >= the slab's revision (every set for
  rev-0 slabs); a delta hit if its id is in a set newer than the rev that added the delta. New data is visible
  within 30 s (current.json TTL), or immediately with `?rev=` from the write response.
- **Automatic compaction** (`manifest.compaction`, profile defaults: more than 16 segments, 32 MB of segments or
  50,000 tombstone ids): after an upsert/delete commit that exceeds a threshold, the writer takes
  `idx/{gen}/compact.lock` (create-only PUT, stale after 10 min) and compacts a bounded slice in `ctx.waitUntil`:
  whole segments, oldest first, up to `cellsPerRun` (256) cells, purge cells when the tombstone threshold is
  exceeded, nothing committed after `deadlineMs` (27 s) from the trigger; the commit is the usual create-only
  manifest PUT + current.json CAS (a lost race is recorded as `conflict`, the next write triggers again).
  `GET /index/status` shows the pressure, thresholds, lock and the latest runs.
- **Cell splitting:** a compaction splits every rewritten cell whose entry count would exceed `maxCellEntries`
  (edge 3000, local 1000): 2-means over the cell's decoded entries (`split_slab`), every entry re-encoded against
  its new mean (`split_slab_records`), the cell keeps mean A and a new cell k is appended with mean B; the revision
  gets `fine.{rev}.f32` (k + 1 means), `k2` + 1 and a rebuilt centroid tree, and query isolates load the new base
  on their first query of it. Automatic runs split only in slices of at most `splitCellsPerRun` (160) cells. See
  [Cell splitting](#cell-splitting).
- `node bench/compact.mjs --gen g4 [--purge 1] [--maxCellEntries N]` (or `POST /build/compact`): rewrites every
  cell holding delta entries from its old slab's records minus tombstones plus the delta records
  (`slab/{cell}.{rev}`), commits a revision without the folded deltas, and drops the tombstone sets nothing can
  reference any more. A delta is folded only when every cell it touches is rewritten; a partial compaction
  (`--cells`) keeps the rest. `--purge` also rewrites cells earlier compactions wrote, so the replace/delete sets
  can be dropped. `POST /build/gc {gen, dryRun: false}` deletes objects no recent revision references.

Writes run in the `writer` build slot (the Worker forwards `/upsert` and `/delete` there unless `?slot=local`):
`DeltaJob` holds a ~25 MB copy of the means (~50 MB while it is passed in), and wasm memory never shrinks, so a
write must not run in an isolate that serves queries. (Until g4 the benchmarks sent writes to vib-bench-0 of the
build pool for the same reason.)

## Several indexes in one bucket

`thread.open({ prefix })`, or a `store` that is a `PrefixedBlobStore`, puts the index under a key prefix. Cache
keys (the Cache API URL and the in-process `BlobCache` key) carry the whole prefix as a `scope`, so two indexes
that share one cache, which every index in one Worker does, never see each other's `idx/current.json`, manifests
or segments. `CachedBlobStore` takes `scope` directly when it is built by hand. Before 0.2.0 a second prefixed
index read the first one's objects from the shared cache (seen as `section "docs": bad range` on the second
index); 0.2.0 scoped the Cache API keys only, 0.2.1 the in-process cache and `PrefixedBlobStore` stores too.

## Objects and revisions

All index objects stay immutable (the edge cache depends on it). Every change is new objects plus a new
manifest *revision*:

```
idx/{gen}/manifest.json                 rev 0, written by `publish` (with slabs.json and slab/{cell:05})
idx/{gen}/manifest.{rev}.json           rev >= 1
idx/{gen}/slabs.json | slabs.{rev}.json per-cell [scanLen, rerankOff, count, bytes, slabRev?] (slabRev absent = 0)
idx/{gen}/slab/{cell:05}                slab revision 0 (finalize)
idx/{gen}/slab/{cell:05}.{slabRev}      slab written by a compaction that read manifest rev slabRev - 1
idx/{gen}/delta/{seq}.vibd              one delta segment per upsert (vectors)
idx/{gen}/seg/{seq}.thread              one lexical/docs segment per upsert with texts or metadata (docs/lexical.md)
idx/{gen}/seg/base.{rev}.thread         the merged segment a segment compaction wrote
idx/{gen}/tombstones.{rev}.u32          sorted unique u32 LE ids deleted/replaced AT rev (one set per rev, not cumulative;
                                        `tombstones.{rev}.{8 hex}.u32` when a lost race left the canonical key taken)
idx/{gen}/tombstones.{rev}.m{from}.u32  a merged set (the SDK's dense compaction): the union of sets of revs from..rev
                                        that no data object tells apart, listed with rev = its newest member's
idx/{gen}/tree.{rev}.bin                centroid tree over the fine means (publish / publish-rev with `tree`, or a
                                        compaction that split cells); inherited
idx/{gen}/fine.{rev}.f32                fine means of a revision whose compaction split cells (k + splits rows)
idx/{gen}/fine{.rev}.f16, fine{.rev}.i8 compact copies of those means (publish, publish-rev {routing}, a split
                                        commit): manifest.routing, docs/query-path.md#cold-opens-compact-routing-cold-tree-routing-base-packs
idx/{gen}/fine.grouped.{rev}.{elem}     the means in centroid-tree group order (cold routing); rebuilt with the tree
idx/{gen}/base.{rev}.threadpack         base pack of rev (publish / publish-rev): everything a cold open reads
idx/{gen}/base.cold.{rev}.threadpack    the same without the routing table (coldRouting "tree")
idx/{gen}/compact.lock                  automatic compaction in progress (create-only PUT, stale after 10 min); mutable
idx/{gen}/compact.last.json             the latest automatic compaction runs (GET /index/status); mutable
idx/current.json                        {gen, rev} (rev absent = 0 for older pointers)
```

Existing generations (g1..g4) are rev 0 as published: `manifest.json`, `slabs.json` and `slab/{cell:05}` are
read unchanged. A manifest of rev >= 1 carries everything a query needs:

```json
{"gen":"g4", "rev":12, "parentRev":11, "updatedAt":"...", "change":{"op":"upsert","seq":3,"count":250,"replace":true},
 "defaults":{"nprobe":34,"rerank":150,"pc":0,"route":"flat","beam":32}, "...build fields of rev 0 (k1, k2, dim, m, format, tiers, pq, vectors)...",
 "tree":{"key":"idx/g4/tree.88.bin","groups":128,"beam":8,"nearestRecall":0.9935,"top3Recall":0.98067},
 "compaction":{"maxDeltas":16,"maxDeltaBytes":33554432,"maxTombstones":50000,"maxCellEntries":3000,"cellsPerRun":256,"deadlineMs":27000},
 "files":{"coarse":"idx/g4/coarse.f32","fine":"idx/g4/fine.f32","pq":"idx/g4/pq.bin","slabs":"idx/g4/slabs.11.json"},
 "deltas":[{"seq":3,"key":"idx/g4/delta/3.vibd","count":250,"bytes":632064,"addedRev":12,"cells":228}],
 "tombstones":[{"rev":12,"key":"idx/g4/tombstones.12.u32","count":250}],
 "nextSeq":4}
```

`files` names the routing table, codebook and slabs table (rev 0 manifests have none and use the fixed names, unless
publish wrote base packs; a revision whose compaction split cells names `fine.{rev}.f32` and has `k2` = k + splits),
and `files.base` / `files.baseCold` the newest base packs (inherited, informational: readers fetch the pack of the
revision they open by its conventional key, and an upsert or compaction revision has none);
`routing` (optional, inherited) names the compact routing tables made from the means `routing.fine` (readers ignore
them once `files.fine` differs; a split commit writes new ones);
`tree` (optional) names the centroid tree the delta encoder and `route=beam` queries use; `compaction` (optional,
inherited) is the automatic compaction policy and split cap (see [Automatic compaction](#automatic-compaction));
`deltas[].addedRev` is the rev that first listed the segment. Build `notes` stay in rev 0.

## Read path

1. **Target.** `/query` resolves (gen, rev) from `?gen=&rev=`, else `idx/current.json` (re-read at most every
   30 s per isolate), else the head of `?gen=` (an R2 LIST, also cached 30 s). A change is visible within 30 s,
   immediately with an explicit `rev` (writers return it).
2. **Caches per isolate.** Parsed revisions (`revCache`, 8 revs: manifest, slabs table, tombstone sets) are
   immutable, so there is nothing to revalidate. Manifests of rev >= 1 are read through the edge cache; rev 0
   (`manifest.json`) is read from R2 once per isolate, because older deployments republished it in place and a
   colo can still cache a stale copy (seen for g2 in YYZ: legacy defaults instead of 26/20). The base (Searcher over coarse/fine, PqCodebook) is keyed by
   `gen|coarse|fine|pq` and shared by every revision with the same files; slabs tables are shared by key.
   Delta segments live in an isolate cache (40 MB budget) in front of the Cache API. Concurrent queries
   share one in-flight read of a new segment, and that read starts at once instead of queuing in the
   first query's read limiter: a request that awaits I/O another request has not started yet gets cancelled
   by the runtime as hung (5 errors in 5000 queries at the first burst after two 5 MB segments; 0 since).
3. **Routing.** `cellDots = searcher.cell_dots(q)` once; flat routing uses `route_from_dots(cellDots, nprobe)`
   (two-level generations with `pc > 0` still use `route`). With `route=beam` (needs `manifest.tree`; not the
   default, see `data/bench-tree.txt`) `route_beam(q, beam, nprobe)` ranks only the cells of the tree's best
   `beam` groups and no full `cellDots` table is computed: each delta segment gets `q·mean` for just the cells
   it has entries in (`delta_cells`, cached per segment, via `cell_dots_for`), filled into a per-query `qc`
   table that `scan_delta` reads (stats `dotsComputed`: 4096 on the flat path, e.g. 228 with one 250-vector
   segment on the beam path; the delta scores are identical).
4. **Phase 1.** Probed slabs are read and scanned as before; the slab key comes from the row's slabRev. Each
   slab is scanned with `scan_pq(prefix, table, q, T, tombFor(slabRev), stats)` (or `scan`): tombstoned
   entries are skipped *inside* the scan, before they can take a top-T slot, so a deleted entry never hides a
   live one ranked below it and a replaced id never reranks its stale copy. In parallel every delta segment is
   scanned with `scan_delta(bytes, q, cellDots, T, tombFor(addedRev + 1), stats)`; its hits are already
   int8-scale scores and skip phase 2.
5. **Phase 2** reranks the slab top-T as before. The final top-k is taken over reranked slab hits and delta
   hits, one copy per id (best score).

Stats: `deltas`, `deltaBytes`, `deltaScanned`, `deltaHits` (kept), `deltaDropped`, `deltaTop` (in the final
top k), `tombstones` (ids over all sets), `tombSets`, `tombDropped` (slab hits removed); `timing.delta`.

### Tombstone rule (id, rev)

A tombstone set at rev `r` hides every copy of its ids that was written *before* `r`:

- **Slab hit** from a slab with `slabRev = s` is dropped iff its id is in a set with `rev >= s`.
  Rev-0 slabs (`s = 0`) are hidden by every set. A compaction that read manifest rev `R` writes slabRev
  `R + 1` and has already applied every set with rev <= R, so only newer sets apply to its slabs (this is
  what lets a deleted-then-reinserted id survive compaction). `tombFor(s)` is the union of the sets with
  rev >= s, built in JS once per (revision, s) and passed to the scan as its skip set.
- **Delta hit** from a segment with `addedRev = a` is dropped iff its id is in a set with `rev > a`
  (the skip set `tombFor(a + 1)`, the same union the slab path uses). The upsert's own set has
  rev = a, so it hides older copies but not the new one.

An upsert is a delete + insert: by default (`replace`, opt out with `replace: false` / `?replace=0` when the
ids are known to be new) it writes a tombstone set of its ids at the same rev as the delta, so older copies in
slabs and older deltas stop resolving immediately, whatever cells they live in.

### Scoped tombstone sets (one-sided updates)

A set may hide one side of its documents only: `{"rev": 14, "key": "...", "count": 3, "scope": "dense"}` hides older
dense copies (slab and delta entries), `"scope": "lexical"` older lexical copies (postings, docs entries, keys in
segments); no `scope` hides both, as before. The SDKs' `upsert({ scope })` writes them: `"dense"` attaches a vector
to documents that already exist (keys resolve through the head, a new key is an error; no segment is written, so
the docs, text and key stay), `"lexical"` re-puts texts, metas or keys and leaves the vectors (an absent attrs word
keeps the live one; in a dense generation a changed word is refused, because vector filters test the copy in the
dense entry). Deletes stay unscoped. The Worker's `/upsert` takes `scope: "dense"` (it writes no segments).

The rules above apply per view: slabs and deltas use the sets without a scope or scoped `dense` (`tombFor`,
`tombForDelta`); segments, docs, `get`, `resolveKeys` and `listKeys` the sets without a scope or scoped `lexical`
(`tombForSegment`, `tombHides`). Dropping is per side too (`tombDroppable`): a lexical set goes once no segment
predates it (segment compaction drops those, and in a generation without a dense index the unscoped ones as well),
a dense set once no delta or written slab predates it and it holds no build id (the dense compaction), an unscoped
set when both hold. A revision with a scoped set lists `"features": ["scoped-tombstones"]` (inherited); readers
refuse a manifest naming a feature they do not know (TS and Python SDKs from 0.2.1, the Worker). Readers before
0.2.1 do not check `features`: they would apply a scoped set to both sides, hiding too much (a document's text after
a vector-only upsert, its vector after a text-only one), never too little. Upgrade readers before writing scoped
upserts.

### Merging tombstone sets

Every upsert or delete writes a set, however few ids it holds, so a writer of small batches leaves thousands of tiny
sets (inseam, 2026-11-14: 2,531 sets at rev 2,803), and every cold open and every compaction reads them one get
each. The SDK's dense compaction (`index.compact`, TypeScript) therefore replaces, at its commit, runs of sets that
no reader can tell apart by one set with the union of their ids (`tombMergeRuns` in `sdk/ts/src/compact.ts`).

**The rule.** Sets s1 < s2 < ... < sn of one scope (absent, `dense` or `lexical`) can be replaced by
`{rev: sn, key, count: |union|, scope}` exactly when the revision being committed has no data object of a side they
hide whose filter picks some of them and not the others:

- dense view (no scope, or `dense`): no remaining delta with addedRev in [s1, sn) (a delta added at a is filtered by
  the sets with rev > a), and no slab with slabRev in (s1, sn] (a slab of slabRev r by the sets with rev >= r;
  rev-0 slabs by every set, so they never tell sets apart). Empty slabs count too, since `tombDroppable` waits for
  them.
- lexical view (no scope, or `lexical`): no lexical/docs segment with addedRev in [s1, sn).

Then every filter a reader builds (`tombFor`, `tombForDelta`, `tombForSegment`, and `tombHides`, whose `maxRev` per
id only ever is compared with segment addedRevs) is the same union before and after, so every query, `get`,
`resolveKeys` and `listKeys` answers the same. Objects written later have revs past the commit, so they never lie
between members either. `tombDroppable` gives the merged set the answer its members have together: the oldest
remaining delta, segment and slab rev > 0 cannot fall between s1 and sn, so each member compares with them alike,
except for the build-id test. A set with build ids (`id < manifest.vectors`) is never dropped, so sets with build ids
are merged only with each other (a merge with the others would keep their ids alive for ever), which also leaves
the purge target (the newest dense-view set without build ids) where it was. Sets of different scopes hide
different sides and never merge, and a set with a scope the SDK does not know never merges. A deleted id that comes
back in a later delta is the case a wrong merge would break: the delta lies between the delete's set and any later
set, so they stay apart until it is folded, and the slab it is folded into (slabRev past both) is filtered by
neither.

In practice: an unscoped upsert with texts writes a segment at its own set's rev, so its set merges with earlier
sets but not with later ones until a segment compaction has merged that segment (`compactSegments`: the merged
segment's addedRev is past them, or the newest addedRev of its run). A dense-scoped set (and an unscoped one with
no segment between) merges with its neighbours once the deltas between them are folded and no slab rev lies
between them; a compaction of a backlog writes its slabs at a rev past every set it read, so the sets behind the
oldest remaining delta form one run. Lexical-scoped sets wait for the segments of their own upserts the same way.

**Cost and convergence.** A run merges once it has 8 sets (fewer wait, so a large merged set is not rewritten for
every set added after it), the longest runs first. The ids the call read for its cells and drops cost nothing; the
call reads at most 256 other sets for merging, in windows of 8, and none within 1 s of its deadline (a call past its
deadline, such as the progress guarantee's, merges only what it read anyway). With `force`, `r.more` stays true
while runs are left after a call that merged some, so `while (r.more)` merges a backlog of thousands over a few
calls; without `force`, merges ride along with the calls that are due. A call that has nothing to fold or rewrite
still commits a revision when it can drop or merge sets (`note: "merged n tombstone sets"`). `mergeTombstoneSets:
false` turns it off.

**The object.** `idx/{gen}/tombstones.{rev}.m{from}.u32` (rev and from: the newest and oldest member's revs), sorted
unique u32 LE like every set, written create-only through `putUnique` (a taken key gets a random sibling); a writer's
own set is `tombstones.{rev}.u32` or `tombstones.{rev}.{8 hex}.u32`, so the two never collide. `count` is the
union's size, so `compactionPressure().tombstones` (the ids over all sets, compared with `maxTombstones`) drops by
the ids that were in several members: the threshold then counts what a filter holds (`idUnion` removes repeats
anyway), and a purge triggered by it comes later than with the repeats. The compaction revision's `change` records
`mergedTombstoneSets` (sets replaced) and `mergedInto` (sets written), `r.mergedTombstoneSets` and `r.tombSets` the
same for the call, and so does `compact.last.json`.

**Readers, packs, GC, rollback.** Every reader takes a set's key from `TombRef.key` (TS `revision.ts`, Python
`revision.py`, the Worker's `loadRevState` and compaction), never from its rev, and the writers already wrote
`{key}.{nonce}` siblings, so a merged manifest is an ordinary manifest with fewer, larger sets: no `features` entry,
readers of every version read it. Packs and base packs list the sets by key; GC (`gcGeneration`, `/build/gc`) keeps
what kept revisions list by key and classifies any `tombstones.*` object. Nothing else deletes objects, so a revision
before the merge still finds its own sets (a pinned reader, or `/build/rollback`, which needs `force` across a merge:
the head lists the merged set, which the older revision does not). The Worker's compaction does not merge; it reads,
filters and drops merged sets like any other.

**Measured** (`sdk/ts/test/compact-merge.test.ts`: 78 cells, 805 sets from 150 rounds of small writes: vector
upserts, dense- and lexical-scoped re-puts, keyed upserts with texts, deletes, re-inserts of deleted ids; the same
bytes compacted with merging and without, every answer compared with an uncompacted copy):

| step | sets left, merging | sets left, not merging |
|---|---:|---:|
| 218 calls of `deadlineMs: 1` while due (fold a backlog of about 250 deltas) | 500 | 805 |
| `compactSegments`, then `force` calls without purge | 2 | 730 |
| 50 more rounds (+270 sets), and their calls | 94 | 998 |

The purge that followed read 69 sets with merging and 973 without. A call folding the oldest delta of a backlog
still needs every dense-view set newer than that delta, and the deltas between those sets keep them apart, so the
short calls read about as many sets either way (89,430 against 95,852 over the 218 calls), and so do the calls
after the later writes (243 each). Merging shortens the reads of cold opens and of the calls that rewrite cells
whose slabs predate the sets (a purge, the first rewrite of a cell in a while), not those of the delta filters while
the backlog lasts.

## Write path

Serialization: the manifest PUT is the lock. `manifest.{rev+1}.json` is written with a create-only conditional
PUT (`onlyIf: {etagDoesNotMatch: "*"}`, verified by `POST /admin/cas-check`), so exactly one writer gets each rev.
Referenced objects are written first, with `putUnique` (create-only at the canonical key, else a
`{key}.{nonce}` sibling, so a crashed or concurrent writer can never overwrite an object a manifest
references; manifests store keys verbatim). The writer takes the head from `current.json` (one GET), loses
the race if the rev is taken, and then rebuilds on the listed head (it keeps its already-written delta
segment). After winning it advances `current.json` with a compare-and-swap (`onlyIf: {etagMatches}`) that only
moves forward and never switches generation; a crashed writer's manifest is picked up by the next CAS.
Responses report `conflicts` (lost manifest races) and `casConflicts`.

- `POST /upsert` — JSON `{gen?, ids, vectors (flat number[] or base64 f32), attrs?, payloads? (strings or {b64}), replace?}`
  or a binary f32 LE body with `x-ids: 1,2,3` / `?ids=` / `?firstId=`. Up to 4096 vectors (rows are normalized).
  `DeltaJob(dim, k, fine means, seq)` + `set_pq_codebook` (the isolate's cached PqCodebook, else pq.bin parsed
  for the job) + `set_tree(tree, manifest.tree.beam)` when the manifest has a centroid tree (the tree bytes are
  read through the cache layer and kept by the isolate) encodes each vector against its primary cell, found by
  scoring the tree's group means and then only the cells of the best `beam` groups (exhaustive over all 4096
  cells without a tree); the segment goes to `delta/{seq}.vibd` (seq from
  `nextSeq`, HEAD-probed), the tombstone set to `tombstones.{rev}.u32` (written while encoding). Response:
  `{gen, rev, seq, count, cells: primary cell per vector, distinctCells, ms, bytes: {delta, tombstones, manifest}, conflicts, tree: {groups, beam} | null, ...}`.
  A retry after a lost manifest race re-encodes only if the fine means, codebook or tree changed under it.
  Writes never load the Searcher. DeltaJob holds a copy of the means (~25 MB, ~50 MB while it is passed
  in), and wasm memory never shrinks, so a write in an isolate that also serves queries first drops an
  idle base (queries reload it); batches are normalized in place and added 256 vectors at a time. Writes
  therefore run in the `writer` build slot: the Worker forwards `/upsert` and `/delete` to that dynamic Worker
  (same code, its own isolate, never serves queries; `?slot=<id>` picks another slot, `?slot=local` runs the write
  in the query isolate). Until g4 the benchmark sent writes to vib-bench-0 of the build pool for the same reason.
  A slot that has never loaded a base keeps the means in JS between writes. With the query Worker
  taking 2000-vector upserts, wasm memory reached 78 MB (54 MB otherwise) and 2 of 5000 queries failed.
- `POST /delete` `{gen?, ids}` — one tombstone set + manifest rev + 1.
- `POST /build/publish-rev` `{gen, defaults?, profile?, tree?, setCurrent?}` — manifest rev + 1 with changed query
  defaults and/or a centroid tree built over the head's fine means (`tree: {groups, iters, seed, beam}` ->
  `tree.{rev}.bin` + `manifest.tree` with its measured recall; `tree: null` removes it), nothing else. `publish` still writes rev 0 and now refuses to overwrite an existing generation
  (`overwrite: true` to force), so defaults are never changed by republishing. `/admin/refresh` only drops the
  isolate's current pointer; no edge-cache eviction is needed any more.
- `POST /build/publish-rev` also takes `compaction: {maxDeltas?, maxDeltaBytes?, maxTombstones?, maxCellEntries?,
  cellsPerRun?, deadlineMs?, splitCellsPerRun?}` (merged over the head's policy, or over the profile's when the head
  has none), `"profile"` (the profile's defaults) or `null` (no policy: no automatic compaction, no splits).
  Driver: `node bench/driver.mjs --gen g4 --publish-rev --compaction profile [--compactMaxDeltas N --compactMaxDeltaMB N
  --compactMaxTombstones N --compactMaxCellEntries N --compactCellsPerRun N --compactDeadlineMs N
  --compactSplitCellsPerRun N]`; a build's `publish` writes the profile's policy (with the flags) into rev 0.
- `POST /build/rollback {gen, toRev, force?}` — a new head with the content of `toRev` (files, k2, tree, deltas,
  tombstone sets, defaults, policy), e.g. to undo a split; refused if `toRev` references deleted objects, or if the
  head lists deltas/tombstone sets `toRev` does not (unless `force`).
- `GET /manifest?gen=&rev=` — a revision read fresh from R2 (default: the head).
- `GET /index/status?gen=&top=10` — the head's update pressure and compaction state: `{gen, rev, current, k, vectors,
  deltas, deltaBytes, deltaEntries, tombstones, tombSets, compaction (policy), exceeded (thresholds the head
  exceeds), compacting (false, or {owner, since, ageMs, trigger} while compact.lock is held), staleLock,
  lastCompaction (the latest automatic run, below), recentCompactions, cells: {k, entries, mean, maxCell, overCap,
  compacted, largest: [[cell, count] x top]}, files}`. Upsert and delete responses carry `compaction: {reasons,
  started}` when the new revision exceeds a threshold.

Upsert latency (writer Worker, client-observed, `data/bench-tree.txt`, same-session A/B on g4): with the centroid
tree (128 groups, beam 8) 100 vectors 1.16 s, 500 1.43 s, 2000 3.27 s (server 0.99 / 1.05 / 2.04 s); without it
(exhaustive primary-cell search, 4096 x 1536 multiply-adds per vector, ~2.6 ms per vector in that session) 1.42 /
2.49 / 7.48 s (server 1.13 / 1.54 / 6.67 s). An earlier session measured 0.86 / 1.86 / 5.07 s exhaustive, so
absolute numbers move between sessions; the A/B is the comparison. Encoding is now ~0.15-0.25 ms per vector, and
small batches are dominated by the five sequential R2 round trips (~0.5-0.9 s). The beam search misses the exact
nearest cell for 0.65% of vectors (tree_recall at beam 8 on 2000 base vectors; those go to the best cell the beam
reaches): self-retrieval after folding the upserted vectors into slabs stayed 100% and recall@10 95.10%, the
same as with exhaustive cells. A delete is ~0.75 s (four round trips). Segments cost 2528 B per vector.

## Compaction

```
POST /build/compact-plan   {gen?, cells?: number[] | "all", purge?, maxCellEntries?}
                           -> {baseRev, slabRev = baseRev + 1, cells, deltas, folded, splits: [[cell, newCell]], splitCandidates, splitBlocked}
POST /build/compact-cells  {gen, baseRev, cells, folded?, splits?, concurrency?}
                           -> rows [cell, [scanLen, rerankOff, count, bytes, slabRev]], splits: [{cell, newCell, entries, a, b}]
POST /build/compact-commit {gen, baseRev, rows, folded, splits?, measureTree?} -> manifest rev (k, fine, tree with splits)
POST /build/compact        {gen?, cells?, purge?, maxCellEntries?, concurrency?} -> all three in one invocation (small compactions)
node bench/compact.mjs [--gen g4] [--purge 1] [--cells a,b] [--maxCellEntries N] [--chunk 48]
                           -> plan and commit in build slot admin, cells over slots cmp-0..7 (one job per slot)
```

- **Cells:** every cell holding a delta entry (`delta_cells`), or the given cells. With `purge`, also every
  cell an earlier compaction rewrote at a rev <= the newest tombstone set (they may hold ids a set hides).
- **Per cell** (one at a time per slot, 3 slots per job): `slab_to_records(oldSlab, tombFor(oldSlabRev))` +
  `delta_records_for_cell(delta, cell, sets with rev > delta.addedRev)` for each folded delta touching the cell ->
  `new SlabJob(dim, cell, slab_centroid(oldSlab), format)` -> PUT `slab/{cell:05}.{baseRev+1}`. Rebuilding a slab
  from its own records is byte-identical, so a cell whose additions were all deleted again comes back with the
  original bytes (checked: same MD5 as `slab/{cell:05}`).
- **Folded deltas:** a delta is folded (its records go into the new slabs and it leaves the manifest) only when
  every cell it has entries in is rewritten (`folded` in the plan); a partial compaction (`cells` subset) keeps
  the other deltas, which go on serving their entries. The commit rejects (409) a folded delta with entries in a
  cell that is not among the rows.
- **Commit** (commitRev on the head, which may be newer than baseRev): fails with 409 if another compaction
  changed the slabs table or a folded delta disappeared; writes `slabs.{rev}.json` with the new rows, drops the
  folded deltas (deltas added since baseRev stay), and drops a tombstone set only when nothing can reference it:
  no remaining delta predates it, no lexical/docs segment (`manifest.segments`, see docs/lexical.md) predates it,
  no slab with slabRev in (0, set.rev] remains, and none of its ids is a build id (rev-0 slabs hold ids
  [0, manifest.vectors), the driver's id space). Otherwise the set is kept.
- **Splits:** see [Cell splitting](#cell-splitting).
- Concurrency: compactions must not run concurrently on one generation (the second one's commit fails with 409;
  their slab objects would have identical bytes, since both read the same base). Automatic runs serialize on
  `compact.lock`; `bench/compact.mjs` does not take the lock, so a manual compaction that races an automatic one
  loses its commit with 409 (nothing is lost; check `GET /index/status` for `compacting` and rerun).

Query cost grows with total delta bytes (every query scans every segment), and replace/delete tombstone sets
accumulate (~4 B per id per query isolate; every slab hit is filtered through them) until a compaction can drop
them, so compactions are now triggered automatically from thresholds:

## Compaction in the SDK

`index.compact(opts)` / `compactDeltas(ctx, opts)` (TypeScript, `sdk/ts/src/compact.ts`) runs the same compaction in
one isolate, for a writer that owns its index and has no build slots (a Durable Object with 128 MB). It writes the
Worker's objects (`slab/{cell:05}.{baseRev + 1}`, `slabs.{rev}.json`, `fine.{rev}.f32`, `tree.{rev}.bin`, the
routing tables of the new means) and the same manifest fields; `sdk/ts/test/compact-worker.test.ts` compacts one
index with the Worker's `POST /build/compact` and with the SDK over three rounds (folds, splits from k 1 to 8, a
rebuilt tree, dropped tombstone sets) and compares every object byte for byte. One thing the Worker does not do:
the SDK merges runs of tombstone sets at its commit ([Merging tombstone sets](#merging-tombstone-sets)); the
cross-check's runs stay below the 8 sets a merge needs, so its `tombstones` compare too.

1. **Lock** `compact.lock` exactly as the automatic runs take it (create-only, stale after 10 minutes, etag
   takeover), so the Worker and the SDK never compact one generation at once. A stable `owner` takes over its own
   lock (a crash left it); an isolate never runs two compactions of one generation. A store without `delete` marks
   the lock released (`ms: 0`), which the Worker treats as stale.
2. **Read phase and plan** on the newest manifest (listed): unless `force`, nothing happens below every threshold.
   The call reads only what the slice needs, not the whole backlog:
   - **Deltas**: their cells come from their entry tables, read oldest first (`addedRev`, then `seq`) in windows of
     8. A delta up to 64 KiB is read whole (one get; the fold uses the bytes), a larger one by two range reads.
     After each window the call plans the deltas read so far. That plan is the plan of the whole backlog cut there,
     since a delta's place in the slice depends only on the deltas before it. Reading stops when every delta is
     read, when the last 16 deltas could not join the slice (its cells or its memory are taken: the slice is full),
     or when reading has used a third of `deadline - 2.5 s`. The first window is always read. The deltas not read
     wait for a later call, which starts from them.
   - **Tombstone sets**: the planner's memory term uses the id counts in the manifest (`TombRef.count`), so no set
     is read to plan. A set's ids are read, once per call, when a cell being rewritten or a delta being folded is
     filtered by it: a slab of slabRev r by the dense-view sets with rev >= r (an empty slab by none), a delta added
     at rev a by the dense-view sets with rev > a. Lexical-scoped sets are not read for this: dense compaction does
     not apply them, and they can be dropped or kept without their ids. A dense-view set's ids are also read when its
     drop depends on the build-id test only (the base has build ids, `vectors > 0`, and no delta, slab or segment
     holds the set), and for purge in a base with build ids, newest first until a set without build ids is found.
     The commit may read up to 256 more sets, of any scope, to merge them (step 5).

   Whole deltas, oldest first, join the slice while their cells fit `cellsPerRun` and the estimated peak fits
   `maxBytes` (see [Memory of a call](#memory-of-a-call)); the cells in flight are as many as the rest of the budget
   holds (at most `concurrency`). While some deltas are unread, every split keeps its mean (an unread delta may have
   entries in the cell), and `force` takes no oversized cell on its own. `r.reads` gives the deltas and tombstone
   sets read, of those the base lists, and the ms to the plan. Cells split once they pass
   `cellEntries = min(maxCellEntries, splitEntries / 2)`, and a delta is only taken when no cell it lands in passes
   `splitEntries`, the largest cell the budget splits on its own. A delta that does not fit waits; a cell that alone keeps a delta out is split first, on its own and with its mean
   kept (`r.presplit`), and the delta folds in a later call. Input that no call with the budget can take (a delta
   with more entries in one cell than it can fold, a cell already past `splitEntries`, a delta over `cellsPerRun`
   cells) is a `ThreadError` with status 413 naming it. With the tombstone threshold exceeded (or `purge`, or
   `force`) purge cells follow, and with `force` the cells over the cap, while the budgets allow. Splits as in the
   Worker, except that a cell an unfolded delta has entries in is split with its mean kept (the new cell takes the
   2-means centre farther from it, and the entries nearer that centre): those entries stay valid, and a backlog
   larger than the budget cannot keep a cell from splitting. When 2-means leaves one half empty (identical
   vectors), the entries are halved under the one mean, so such a cell still shrinks.
3. **Cells** under the caches (whole slabs would only fill them), split cells first, then each delta's cells, oldest
   delta first. Slab objects are written create-only: the same bytes already there (a rerun on the same base after a
   crash) are reused; other bytes are overwritten only while no revision after the base changed the slabs table (so
   none can list them), else the run ends as a `conflict`.
4. **Deadline**: counted from the start of the call. No cell starts after `deadline - 2.5 s`; with splits up to
   4.5 s more, scaled by the fine means' bytes (the 4.5 s were g4's 25 MB table and tree; a head with a tree always
   keeps them), so a generation with few cells keeps its time for the cells. When the deadline cuts the slice, the
   run commits the finished cells whose deltas are all finished (a cell only with every folded delta in it, splits
   only as a prefix, since new cells are numbered in order; to the fixpoint) as `partial`, so a slice too large for
   the time still makes progress. Nothing committable: `deadline`.

   **The oldest delta always folds.** The cells it needs to commit (its own, the cells of the folded deltas that
   share them, and every split numbered before one of them, to the same fixpoint) run first, and they start
   whatever the time. When a cell keeps the oldest delta out, that cell's split takes its place. So a call that
   plans folds at least the oldest delta, however large the backlog and however short the deadline; only the lock,
   a conflict or a 413 stops it. Such a call can take longer than `deadlineMs` when the oldest delta alone needs
   more (a slow store, many tombstone sets); when the read phase alone passed the cell deadline, the note says
   `read phase passed the deadline: ...` with what was read. A call with no delta to fold (purge or `force` only)
   whose read phase passed the deadline starts no cell and ends `deadline` with that note.
5. **Commit**: `compactCommit`'s rules on the head (409 if the slabs table or the fine means changed, a folded delta
   left, or a delta that stays has entries in a cell whose mean changed), then `compact.last.json` and the lock
   release. That last check reads only the deltas a writer added after the base: the plan read the others, and
   while any was unread every split kept its mean, so there is nothing to check. On the state the revision commits
   (the deltas that remain, the new slabs table, the head's segments, the sets not dropped) the commit then merges
   the runs of tombstone sets no data object tells apart ([Merging tombstone sets](#merging-tombstone-sets)); a
   retry on a newer head merges again on that head and reuses the objects it wrote. `r.more` is true while a
   threshold is exceeded (with `force`: while deltas remain, cells stay over the cap after a split that moved
   entries, or runs are left to merge after a call that merged), a split was deferred or the slice was cut.

A delta fold and cell split in a Durable Object then looks like:

```ts
const r = await ix.upsert({ keys, texts, chunk: true });
if (ix.compactionDue().due) for (let c = await ix.compact({ owner: state.id.toString() }); c.more && c.outcome !== "locked"; c = await ix.compact({ owner: state.id.toString() }));
```

Tombstone sets also wait for the lexical segments that predate them: run `compactSegments` (a full merge, or
bounded runs) too, or the sets stay, and the unscoped ones cannot merge across those segments. The Python SDK does
not compact (it reads merged sets like any others).

### Small writes

What one `upsert` or `delete` costs, whatever its size: a manifest revision (a create-only put of a manifest that
lists every delta, segment and set of the head, 50 to 120 bytes each, so it grows with the backlog and is rewritten
whole by every write; a head of 767 deltas and 2,531 sets is a manifest of about 200 KB), the `current.json`
compare-and-swap (a head, a get and a conditional put; it carries a copy of the manifest only while that is under
16 KB), one delta segment when it has vectors (1,640 bytes for one vector of dim 1024), one lexical/docs segment
when it has texts, metas or keys, and one tombstone set unless `replace: false`: about ten requests in all, before
the writer's own index loads the new revision. Readers then pay per object: every query scans every delta (a get
each until cached), every text query reads every segment, and a cold open reads every set.

What compaction does about it:

- **Deltas** need no step of their own: folding is the coalescing. A call folds whole deltas oldest first, as many as
  the slice holds, and many tiny deltas in one cell fold into that cell's one new slab in the same call (in
  `test/compact-merge.test.ts` one call with the default deadline folds 83 deltas of one vector into 60 cells). But
  a compaction is due after `maxDeltas` (16) deltas, so the documents a compaction folds are 16 x the batch size,
  and every cell they land in is rewritten whole (a slab of `cellEntries` entries, 1.5 MB at 1000 entries of dim
  1024): with writes of one vector, a compaction writes about one slab per vector.
- **Tombstone sets** are merged at the commit where no data object tells them apart (above), and dropped once nothing
  holds them. The deltas and segments of the writes themselves keep sets apart until they are folded or merged.
- **Segments** are merged by `compactSegments` only.

**Batch at least 64 documents per write, or what arrived in the last second when fewer come: never one write per
document, and one write per logical unit (a crawl page), not per sub-batch.** At 64 documents the requests and
the manifest of a write cost less than the documents' own bytes (a delta entry is 1,560 bytes at dim
1024, 64 of them about 100 KB, against ten requests and a manifest of a few KB); the 16 deltas a compaction waits
for hold about 1,000 documents, so a generation of up to a few hundred cells gets several new entries per rewritten
slab instead of one; and at one write per second a writer makes at most 3,600 revisions an hour, which a
compaction every 16 of them keeps to 16 deltas. Merging then keeps about one set per compaction whose slabs are
still live, instead of one per write (unscoped writes with texts: once their segments are merged). inseam's crawl
wrote one revision per one or two vectors (767 deltas of about 3 KB).

### Memory of a call

What a call holds at its peak, as the planner estimates it (`r.estimatedPeakBytes`, at most `maxBytes`):

- **always**: 1 MiB (manifests, delta entry tables, JSON), the tombstone sets x3 (the ids, their dense and lexical
  unions; counted from the manifest's `count`s, an upper bound since a call reads only the sets it needs), the
  slabs table x2;
- **cells phase**: the folded deltas (held for the run) plus the largest of them again (its copy in wasm), and per
  cell in flight `3 x old slab + 6 x new slab` for a fold, `max(9, 8 + 4 x dim / bytesPerEntry) x new slab` for a
  split (the merged slab, the re-encoded halves, their two slab jobs, `split_slab`'s decoded vectors);
- **commit phase** (only with splits, after the cells are released): the fine means x3 (old, new, the tree build's
  copy), `k x dim x 12` bytes.

The factors come from measurements in Node with the wasm memory growth counted (the wasm memory never shrinks, so
its growth over a call is part of the isolate's cost for good): `bench/compact_memory.mjs` samples the JS
ArrayBuffers and the wasm memory after every wasm call and store read or write, in a fresh process per call (`all`
runs the table below; `measure` takes any generation directory, such as a copy of an R2 index). Dim 1024, format 2
(1560 B per entry):

| call | estimate | measured | before 0.2.3: estimate |
|---|---:|---:|---:|
| fold of 1000 entries into an empty cell | 11.8 MB | 9.4 MB | 7.8 MB |
| fold of 3000 entries into an empty cell | 32.8 MB | 28.6 MB | 22.8 MB |
| fold of 6000 entries into an empty cell | 64.2 MB | 54.8 MB | 45.3 MB |
| split of 1000 / 3000 / 6000 entries | 18.7 / 53.5 / 105.6 MB | 16.3 / 47.6 / 96.2 MB | 14.7 / 43.5 / 86.7 MB |
| fold of 111 entries into a cell of 2869 (inseam's e1, rev 206) | - (now a 413) | 38.9 MB | 18.2 MB |

The old model (4x the new slab for a fold) undercounted the slab job's buffer growth and the old slab's records:
the e1 fold above was planned at 18.2 MB under a 32 MiB budget and measured 38.9 MB.

**Cell sizes follow the budget.** A split costs about `8 x bytes + 4 x dim x entries`, so at dim 1024 one call of
32 MiB splits cells of up to about 1900 entries, fewer than the `edge` profile's 3000. Before 0.2.3 the planner
folded deltas into a cell up to `maxCellEntries` without splitting (a cell under the cap needs none), and then no
delta could join it: the fold would pass the cap, the split that then follows did not fit the budget, so every
call was `blocked` and the deltas piled up. That is what happened to inseam's `e1` on 2026-11-04 (one cell, 2980
entries, 32 MiB calls, 47 deltas and growing). Now `compactionLimits(manifest, { maxBytes })` derives
`splitEntries` (the largest cell a call splits on its own, with room for the policy's `maxTombstones`),
`cellEntries = min(maxCellEntries, splitEntries / 2)` where cells split, and `upsertVectors` (an upsert of that many
vectors always folds, even into a cell at `cellEntries`); at dim 1024 and 32 MiB they are about 1920, 960 and 810.
The invariant is that no compaction leaves a cell past `splitEntries`, so every later call with the same budget can
fold into it after splitting it. A smaller budget later may find cells it cannot split: the call then fails with a
413 that names the cell and the budget it needs (one call with that budget, or the Worker, repairs it).
`sdk/ts/test/compact-bulk.test.ts` replays the bulk copy: 20,000 vectors of dim 1024 in upserts of 200, a 32 MiB
compaction after each one that is due; k grows to 32, the deltas never pass 17, no cell passes `splitEntries`, no
estimate passes the budget, and the wasm memory grows by 15 MB over the whole run.

## Automatic compaction

Policy: `manifest.compaction` (inherited by every revision; profile defaults for `edge` and `local`:
`maxDeltas` 16, `maxDeltaBytes` 32 MB, `maxTombstones` 50,000 ids, `maxCellEntries` 3000 (edge) / 1000 (local),
`cellsPerRun` 256, `deadlineMs` 27,000, `splitCellsPerRun` 160; Worker defaults when absent: `slots` 4, `cellsPerSlot`
160). g4 has the edge policy since rev 104 (publish-rev; current values since rev 139).

1. **Trigger.** After every upsert/delete commit the writer compares the new revision with the thresholds (more
   than `maxDeltas` segments, more than `maxDeltaBytes`, more than `maxTombstones` ids over all sets). If one is
   exceeded and this isolate is not already running one, it starts `autoCompact` with `ctx.waitUntil` and answers
   at once (`compaction: {reasons, started: true}` in the response). The writer is the `writer` build slot (a
   dynamic Worker; `waitUntil` work there was checked to survive 28 s after the response).
2. **Lock.** `idx/{gen}/compact.lock` is written with a create-only PUT (`onlyIf: {etagDoesNotMatch: "*"}`) carrying
   `{owner, ms, at, trigger}`; if it exists and is younger than 10 minutes the run gives up (another run is in
   progress); an older lock is taken over with an etag-conditional PUT (exactly one contender wins). The lock is
   deleted (if still ours) when the run ends, however it ends.
3. **Slice** (bounded: waitUntil work may run 30 s after the response; one isolate rewrites ~10-14 g4 cells/s at 6 R2
   connections, measured 256 cells / 330 MB each way in 19.4 s): whole deltas, oldest first, while the union of
   their cells fits in the run's budget, `cellsPerSlot` x `slots` (640) when the cells are spread over build slots,
   `cellsPerRun` (256) when the writer rewrites them itself (`slots: 1`, or the module running without the loader);
   a delta that does not fit is left for the next run (or, if it alone touches more cells than the budget, e.g. a
   1000+ vector upsert, for `bench/compact.mjs`: the run reports `blocked`). When the
   tombstone threshold is exceeded the remaining budget goes to purge cells (slabs with slabRev in (0, newest set
   without build ids], oldest slab revision first), so the sets can eventually be dropped. Rewritten cells over
   `maxCellEntries` are split, but only when the slice has at most `splitCellsPerRun` cells (a split commit also
   rewrites the 25 MB fine table and rebuilds the tree: 9.4 s measured for 2 splits without the recall check): a larger slice defers them (`splitsDeferred`), so a
   run never misses its deadline because of splits. (The first test, before this rule, split inside a 228-cell slice
   and stopped at the deadline before its last cells, without committing.)
4. **Rewrite + commit:** the cells phase runs in build slots `auto-0..3` (`compactCellsOnSlots`): the sorted cells are
   cut into ~16 chunks of contiguous cells (4 per slot, at least 24 cells each) on a shared queue, and each slot runs
   one `compact-cells` job at a time and takes the next chunk when it is done, so a slot on a slow host (per-job times
   differ by up to ~2.5x between hosts) does less of the work; each job reads the compaction base and the folded
   deltas itself. With `slots: 1` the writer runs `compactCells` (6 cells in flight) itself as before. Lock, plan,
   commit and record stay in the writer; `compactCommit` is exactly as in a manual compaction. No cell (and no
   chunk) is started after `deadline - 2.5 s` (`- 7 s` with splits) and nothing is committed after the deadline (`deadlineMs`
   from the trigger), so a run ends inside the waitUntil window; a cut-off run's slab objects are garbage for `gc`.
   The commit is the usual create-only manifest PUT + `current.json` CAS; if another compaction committed first
   (slabs table, routing table or folded deltas changed) it fails with 409 and the run records `conflict`: the next
   write that finds a threshold exceeded triggers again.
5. **Record:** `idx/{gen}/compact.last.json` = `{last, recent}` (21 runs) with `{trigger: {rev, op, reasons, pressure,
   ms}, outcome: committed | noop | blocked | conflict | deadline | error, ms (trigger -> new rev), phases: {lock, plan,
   cells, commit}, baseRev, rev, cells, folded, splits, splitsDeferred, readMB, writtenMB, slots, slotJobs, slotMs
   (each slot's wall time), ...}`; the manifest's
   `change.auto` names the trigger. `GET /index/status` shows it.

Each run folds only what fits. With 250-vector upserts (~228 cells per segment, ~200 more per further segment
because they overlap) a run over 4 slots folds three segments (606 cells) in about the time a single Worker took for
one; a single-writer run folds one. For bulk loads, use bigger batches and `bench/compact.mjs` (build slots).

Slot-parallel runs on g4 (`node bench/update_bench.mjs --gen g4 --only trigger`, writes through the query URL to the
`writer` slot, 10 background queries on the current pointer throughout):

| budget | cells (segments folded) | cells phase (slowest slot; per-slot range) | trigger -> new rev | file |
|---|---|---|---:|---|
| single writer on vib-bench-0 (before) | 228 (1) | 16.5 s | 18.8 s | `data/bench-trigger.txt` |
| 4 slots x 160 (default) | 606 (3) | 16.7 s (14.7-16.7 s, 16 jobs) | 18.9 s | `data/bench-trigger-slots.txt` |
| 4 slots x 256 | 920 (5) | 23.0 s (20.6-23.0 s, 16 jobs) | 25.3 s | `data/bench-trigger-slots-1024.txt` |

The 920-cell runs committed (also on g5 with fixed per-slot ranges: 19.4 s slowest slot, 13.5 s fastest, 21.4 s to
the new rev), but the g4 one finished its cells 0.4 s before the cell deadline (24.5 s after the trigger), so the
default is 160 cells per slot: the same margin as the single-Worker 228-cell slices had. Default run: 0 errors in
7,779 background queries, p50 86 / 87 / 89 ms before / during / after the run (all three slot runs: 0 background
errors); self-retrieval 100%, recall@10 95.11%; the query isolates' wasm memory stayed at 54 MB through every pass. Cleanup (`compact --purge` over slots
cmp-0..7, 2305 cells): 53-61 s (45 s over the old pool).

Single-writer measurement on g4 (before the build slots; `data/bench-trigger.txt`, writes to vib-bench-0):

| | |
|---|---|
| trigger | upsert 17 of 250 vectors (rev 124): `deltas 17 > 16` |
| run | lock 0.2 s, plan 1.0 s, 228 cells (1 segment folded) 16.5 s, commit 1.1 s: trigger -> rev 125 in 18.8 s (client: upsert response -> status shows it 19.5 s) |
| background queries | 0 errors in 9,555 (before / during / after the run); p50 83 / 82 / 89 ms, p95 306 / 174 / 260 ms |
| interleaved A/B (c=50, explicit rev) | clean 76.8 ms p50, 17 deltas 81.9 ms, 16 deltas after the run 79.3 ms |
| quality after the run | self-retrieval 100% (rank 1, twins ok), recall@10 95.11% |
| cleanup | delete 4250 ids + `compact --purge` (2305 cells, 45 s, 18 tombstone sets dropped): clean, recall@10 95.13% |

Follow-up tests with a temporary policy (`maxDeltas 0`, `maxCellEntries 1900`; rolled back afterwards with
`/build/rollback`, details in `data/bench-split.txt`): the same 228-cell slice needed 16.5-20.0 s of cell work in
three committed runs (trigger -> rev 18.8-21.7 s) and once more than 21 s, which hit the then 24 s deadline (nothing
committed, lock released, the next trigger succeeded): hence the 27 s default. A 228-cell slice with 3 splits hit the
deadline too, hence `splitCellsPerRun`; a 2-vector upsert into two oversized cells gave a 2-cell slice that split
both (k 4103 -> 4105, trigger -> rev 11.8 s, commit 9.4 s with the new fine table and tree). A single-writer run
folds about one 250-vector segment; the compaction throughput of one isolate (~11-14 cells/s) is the limit, which
the build slots now multiply (above).

## Measured (g4, `data/bench-updates.txt`)

`bench/update_bench.mjs` inserts the 5000 test queries as ids 1,000,000+i, deletes and replaces them, and compacts.

| | |
|---|---|
| upsert latency (writer Worker, client) | with the centroid tree (128 groups, beam 8): 100: 1.16 s, 500: 1.43 s, 2000: 3.27 s; same session without it (exhaustive primary-cell search, ~2.6 ms/vector): 1.42 / 2.49 / 7.48 s (`data/bench-tree.txt`). An earlier session measured 0.86 / 1.86 / 5.07 s exhaustive (250: 1.13 s); ~0.5-0.9 s is sequential R2 round trips |
| bytes written per upsert | 2528 B per vector (delta segment) + 4 B per id (tombstones) + ~1-3 KB manifest |
| delete latency | ~0.75 s (1000 or 5000 ids) |
| visibility | own id at rank 1 for 100% of the inserted queries (counting the 894 queries that have an identical twin), in the top 10 for 100%, right after the upsert (with `?rev=`); deleted ids never returned |
| recall@10 vs baseline (95.12%, nprobe 34) | within 0.02 points with 20 deltas, after deletes, after compaction and after 2000-vector replaces |
| query cost of deltas (c=50, interleaved A/B) | +2.4 ms p50 with 5 segments (3 MB), +5.6 ms p50 / +5.4 ms mean with 20 (12 MB, 5000 tombstones); ~0.45 ms per MB, all CPU in `scan_delta` |
| compaction of 5000 vectors | 2151 cells, 3.0 GB read + 3.0 GB written, 43-51 s over the former 8-Worker pool; latency back at the no-delta baseline; all tombstone sets dropped |
| cleanup | delete everything + `compact --purge` -> the original slabs again (byte-identical, checked by MD5 on a sample cell), recall = baseline; `gc` then deleted 12.2 GB of superseded objects |
| automatic compaction over build slots (`--only trigger`, `data/bench-trigger-slots.txt`) | 17th upsert of 250 vectors triggers it; the cells are spread over 4 slots: 606 cells (3 segments folded) in 16.7 s, trigger -> new rev 18.9 s; 0 failed out of 7,779 background queries; query isolates stay at 54 MB of wasm memory |
| automatic compaction, single writer (before the build slots, `data/bench-trigger.txt`) | 17th upsert of 250 vectors triggers it; trigger -> new rev 18.8 s (228 cells = 1 segment in 16.5 s, commit 1.1 s) on vib-bench-0; 0 failed queries out of 9,555 background queries; p50 (interleaved A/B, c=50) 76.8 ms clean, 81.9 ms at 17 segments, 79.3 ms after the run; cleanup back to a clean head at 95.13% |
| cell split (`bench/split_bench.mjs`, `data/bench-split.txt`) | 7 cells over 2400 entries split (k 4096 -> 4103): largest cell 3109 -> 2257 entries, recall@10 95.12% -> 95.13%, query isolates' wasm memory unchanged (54 MB) |

## GC

`POST /build/gc {gen, keep = 2, graceSec = 300, dryRun = true}` deletes slab, delta, tombstone, slabs-table,
`fine.{rev}.f32` and `tree.{rev}.bin` objects under `idx/{gen}/` that none of the kept revisions references: the newest `keep` revisions plus any
revision superseded less than `graceSec` ago, and never an object younger than `graceSec` (an in-flight writer's
uncommitted objects). Manifests are kept as history (old ones then point at deleted objects). Base files and
anything not created by revisions are never touched. After the benchmark, `gc` on g4 kept revs 78-79 and
deleted 8748 objects (12.2 GB: superseded slab revisions including the rev-0 slabs of the 2151 rewritten
cells, 58 deltas, 66 tombstone sets, 8 slabs tables). So g4's rev 0 can no longer be served, and republishing it
from `data/slabs-g4.json` would point at deleted slabs; rebuild instead.

From the TypeScript SDK the same rule is `gcGeneration(store, {gen, keep, graceSec, dryRun})` (or `index.gc(opts)`
for the index's own generation; dry run unless `dryRun: false`), over any store with `listObjects` (upload times)
and `delete`. It returns the same fields as `/build/gc` (`keptRevs`, `garbage`, `kinds`, `deleted`, `sample`).

## Switching generations

A generation can be built beside the live one and swapped in: `createGeneration({ setCurrent: false })`, writes
through `open({ gen })` (they commit to that generation and never move `idx/current.json`, which names another
one), then one compare-and-swap of the pointer:

```ts
const r = await setCurrent(store, { gen: "dense", rev, expect: { gen: "lex", rev: liveRev } });   // or index.setCurrent(...)
// r = { gen, rev, previous: { gen: "lex", rev: liveRev }, updated: true, conflicts }
```

`rev` defaults to the generation's newest manifest, which must exist. With `expect` (`{gen, rev}`, or `null` for no
pointer) the switch fails with a 409 and writes nothing unless the pointer is exactly that; without it any pointer
is replaced, still through an etag CAS (`previous` says what was replaced). The pointer carries the new manifest
inline when it is small, as every writer's does. Readers that follow the pointer (`open()` without `gen`) move
within their refresh interval (30 s); writers still on the old generation keep committing to it but no longer
advance the pointer, since a writer only moves it within its own generation. The switch is recorded in
`idx/{old}/superseded.json` (`{by: {gen, rev}, rev, ms, at}`).

The old generation can then go, as a whole:

```ts
await dropGeneration(store, { gen: "lex", graceSec: 300 });                    // dry run: { objects, bytes, kinds, ... }
await dropGeneration(store, { gen: "lex", graceSec: 300, dryRun: false });     // deletes every object under idx/lex/
```

It refuses (409) the current generation, a generation the pointer left less than `graceSec` ago (from
`superseded.json`, else the pointer's own age when another writer made the switch), one written less than
`graceSec` ago (a writer may still use it) and one whose `compact.lock` is held. Manifests are deleted first, so a
reader that still opens it gets a clean 404 instead of missing slabs, and a rerun after a crash finishes the rest.
Readers pinned to the old generation (`open({ gen: "lex" })`) are not protected: close them before the grace
period ends. To keep the old generation readable but smaller instead, `gcGeneration` it with `keep: 1`.

## Cell splitting

A compaction splits every rewritten cell whose entry count would exceed `maxCellEntries` (`manifest.compaction`,
or `maxCellEntries` in the compact-plan request / `bench/compact.mjs --maxCellEntries`; 0 = never).

- **Plan:** the estimate is the slab's count plus the folded deltas' entries in the cell (tombstoned entries are not
  subtracted, so it is an upper bound). New cells are numbered k, k + 1, ... in cell order (`splits: [[cell,
  newCell]]`). A cell is not split (`splitBlocked`) when an unfolded delta has entries in it (those were encoded
  against the cell's current mean), nor in a two-level generation (k1 > 1).
- **Cells:** the merged cell (old records minus tombstones + folded delta records, one `SlabJob` with the old mean)
  -> `split_slab(merged, [], cell + 1)` (spherical 2-means over the int8-decoded entries, k-means++ seed, 10 Lloyd
  iterations) -> `split_slab_records(merged, [], means2, codebook)`: every entry is decoded from its int8 tier
  (`old mean + scale8 * code8`), goes to the new mean with the higher cosine (`kmeans::split_side`, the rule 2-means
  assigns with) and is RE-ENCODED against it (4-bit MSE-clipped, int8, PQ with the index codebook), so both halves are
  correct residual slabs. The result is one buffer: a `u32` LE byte length `la` of records_a, then records_a, then
  records_b. `SlabJob(dim, cell, meanA)` -> `slab/{cell:05}.{slabRev}`, `SlabJob(dim, newCell, meanB)` ->
  `slab/{newCell:05}.{slabRev}`. (Also exposed: `split_part_records(records, oldMean, means2, codebook)` with the same
  output for part records, e.g. `delta_records_for_cell`, and `reencode_records(records, oldMean, newMean, codebook)`.)
  Every split is logged (`console.log`) and returned (`splits: [{cell, newCell, entries, a, b}]`).
- **Commit:** `fine.{rev}.f32` = the old means with row `cell` = mean A and rows k.. = the B means (read back from the
  new slabs' headers, which the slabs were encoded against), `k2` = k + splits, the slabs table gains the rows, and
  the centroid tree is rebuilt over the new means with the same groups/iters/seed/beam (`tree.{rev}.bin`; recall
  measured on 1000 sample vectors by manual compactions, skipped by automatic ones). The commit refuses (409) when
  the routing table changed since the plan or a delta that stays (unfolded, or written since the plan) has entries
  in a split cell. Concurrent upserts that read the old means re-encode on their retry (their files key changed).
- **Readers:** the base key (`gen|coarse|fine|pq`) changes, so every query isolate builds a new Searcher over k + 1
  means on its first query of the new rev (a 25 MB `fine.{rev}.f32` read, Cache API miss once per colo); the old base
  is freed when its last query finishes. Measured: query isolates stayed at 54 MB of wasm memory (max over 10,000
  queries before and after the switch; the old Searcher's pages are reused), median base load 418 ms on the first
  queries after the switch (92 ms before, from cache). Writers reload the means (keyed by the fine key) and the tree.
- `gc` deletes the split cells' old slab revisions as usual, and `fine.{rev}.f32` / `tree.{rev}.bin` objects no kept
  revision references (never `fine.f32`).

g4 (`node bench/split_bench.mjs --gen g4 --maxCellEntries 2400` -> `data/bench-split.txt`): the 7 cells over 2400
entries (3109, 3069, 3054, 2569, 2493, 2453, 2402) were split in a 7-cell compaction (rev 104 -> 105, k 4096 ->
4103; cells 3.7 s, commit 64 s with the old full recall check over 4 beams, now ~1000 rows / one beam). The largest
cell went from 3109 to 2257 entries, recall@10 at nprobe 34 from 95.12% to 95.13% (beam 32: 95.11% both), scanned
entries per query 30,265 -> 30,079. Kept: g4 serves k = 4103 since rev 105.

Centroids otherwise stay fixed between full rebuilds. Schedule a full rebuild when assignment quality degrades
(e.g. the fraction of new vectors whose primary-cell cosine falls below a threshold).

## Build slots (Worker Loader) instead of the pool

The spike ([worker-loader.md](worker-loader.md)) showed that every dynamic Worker id runs in its own isolate with its
own 128 MB, so the vib-bench-0..7 pool was replaced by one deployment: `vib-bench` forwards `/build/<stage>?slot=N`,
`/upsert` and `/delete` to dynamic slot Workers that run the same code (`examples/cloudflare-worker/README.md`). For the update path
this means: writes run in the `writer` slot by default (no query isolate ever holds a DeltaJob; checked with two
2000-vector upserts through the query URL: query isolates stayed at 54 MB, where the query Worker taking such writes
reached 78 MB), manual compactions use slots `admin` (plan, commit) and `cmp-0..7` (cells), and automatic runs spread
their cells over `auto-0..3` (above). A 2000-vector upsert touches ~1500 cells, more than one automatic run may
rewrite: such deltas report `blocked` and need `bench/compact.mjs`.

Seen while checking the writer slot (query path, unchanged by this work): in the first query burst (c=50) right
after a new 5 MB (2000-vector) segment, 3-13 of 1500 queries failed with the runtime's "code had hung and would
never generate a response" (cross-request wait on the shared read of the new segment); later bursts on the same
revision had none, and 250-vector segments showed none.
