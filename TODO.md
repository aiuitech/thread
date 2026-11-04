# TODO

## Updates that touch one side of a document

### Problem

An upsert replaces a whole document. Its tombstone set (`tombstones.{rev}.u32`) hides every older copy of its ids
on both sides: the dense entries in slabs and deltas, and the docs, postings and keys in lexical segments
(`tombFor`, `tombForDelta`, `tombForSegment` and `tombHides` in `sdk/ts/src/wire.ts` all read one `TombState`).
So:

- **Vector-only upsert under a known key or id** (backfilling embeddings after a text-first index): the new delta
  carries the vector, no segment is written, and the tombstone hides the old docs, postings and keys. The
  document drops out of text search, `get(key)` stops resolving it, and its meta is lost.
- **Metadata, tag or text change** (a mail label, a rename, an enrichment pass): the caller has to re-send the
  vector, or the tombstone hides the old one. The caller usually doesn't have the vector: thread can't return it
  (slabs hold PQ and int8 residuals, not the f32 input), so it means re-embedding.

inseam keeps its catalog and its search index in one thread index per host. It needs both kinds of update:
text and metadata first, vectors backfilled later, and frequent metadata-only churn. For it this is a
correctness bug, not an optimisation.

### Design

A tombstone set gets a **scope**: which side it hides.

- `TombRef` in the manifest gets `scope?: "dense" | "lexical"`. Absent means both: today's behaviour, and every
  existing manifest stays valid.
- Dense reads (`tombFor` for slabs, `tombForDelta` for deltas) apply the sets with no scope or `scope: "dense"`.
  Lexical reads (`tombForSegment`, `tombHides`, key resolution, `listKeys`, docs resolution) apply the sets with no
  scope or `scope: "lexical"`. In practice that's two `TombState`s per revision, or one with two views.
- `upsert` takes `scope?: "both" | "dense" | "lexical"` (default `"both"`):
  - `"lexical"`: write the segment (keys, docs, postings, attrs in docs) and a lexical-scoped tombstone set.
    Reject `vectors` and `payloads`. Dense entries are untouched.
  - `"dense"`: write the delta and a dense-scoped tombstone set. Reject `texts` and `metas`. With `keys`, resolve
    them to ids through the head revision but **write no segment**, so the docs and keys stay where they are.
    Unknown keys are an error: there is no document to attach a vector to, and allocating an id without storing
    the key would orphan it.
  - `"both"`: unchanged.
- `delete` and `deleteKeys` stay unscoped: a delete hides everything.
- `replace: false` keeps its meaning (no tombstone set at all) and composes with `scope`.

### Attrs: the hard part

The attribute word is stored twice: in the dense entry (inside the slab scan prefix, where the filter is tested)
and in the `docs` store (where text queries post-filter). A lexical-scoped upsert that changes attrs on an id
with a dense entry leaves the dense copy stale, so a filtered vector query would test the old word. Options, to
decide before building:

1. **Reject it.** In a lexical-scoped upsert, an id whose attrs differ from its live docs copy, and that has a
   dense entry, is an error. The caller has to re-send the vector with `scope: "both"`. Comparing costs a docs
   read per id, which keyed upserts already do for id resolution. Simplest. It puts the constraint on the caller:
   keep frequently changing fields out of attrs in dense indexes.
2. **Re-encode in place.** Find the id's current dense entry and copy its codes into a new delta with the new
   attrs. Needs id → cell, which thread doesn't keep. Without it, this is a scan of every slab: too expensive
   unless an id → cell map is added.
3. **Attrs overlay.** A small per-revision `attrs.{rev}` map (id → word) that dense scans consult for overlaid ids
   before the filter. Correct and cheap to write, but it adds a lookup to the hot scan path and needs folding at
   compaction.

Start with (1). Revisit (3) if callers need cheap attrs changes on dense documents.

### Work

Done for 0.2.1 (TS SDK, Python SDK, reference Worker): the gate is `manifest.features: ["scoped-tombstones"]`; attrs
use option 1, with every id of a dense generation treated as having a dense entry and absent attrs keeping the live
word; `chunk` needs scope "both"; `texts` under "dense" feed the embedder only. Rust needed no change (`tomb.rs`
only sees the buffers it is handed). Readers before 0.2.1 cannot be made to refuse: they ignore `features` and
over-hide.

- [x] **Format.** Add `scope` to `TombRef` (`sdk/ts/src/wire.ts`, the Worker's copy in
      `examples/cloudflare-worker/src/index.ts`, `sdk/python/thread_sdk/wire.py`). Gate it: an older reader that
      ignores `scope` would hide too much. Bump `manifest.format`, or add a `features: ["scoped-tombstones"]` that
      readers must understand, and have older SDKs refuse such a manifest rather than misread it. Document it in
      `docs/update-path.md` (manifest example and tombstone rule) and `docs/lexical.md`.
- [x] **Tombstone state.** Split the views in `buildTombState`: dense unions for `tombFor` and `tombForDelta`,
      lexical unions and `maxRev` for `tombForSegment` and `tombHides`. Same in Python. Rust (`tomb.rs`) only sees
      the buffers it's handed, so it shouldn't need changes. Confirm.
- [x] **Write path, TS SDK** (`sdk/ts/src/write.ts`). Add `scope` to `UpsertInput`. In `prepareBatch`, validate
      which fields each scope allows. In `upsert`, skip `encodeSegment` for `"dense"` even when `keys` are given
      (keys only resolve), skip `encodeDelta` for `"lexical"`, write the tombstone set with its scope, and record
      `change.scope` in the manifest. Error on unknown keys under `"dense"`.
- [x] **Attrs check** for `scope: "lexical"` on dense generations: option 1 above. Resolve the live docs copy
      (already read for keyed resolution; read it for id-only batches too) and reject changed attrs on ids that
      have a dense entry. Knowing whether an id has a dense entry also needs deciding: ids below
      `manifest.vectors` (the built base) plus ids in live deltas, or treat every id in a dense generation as dense.
- [x] **Write path, Python SDK** (`sdk/python/thread_sdk/write.py`): the same, keeping wire parity.
- [x] **Write path, Worker** (`POST /upsert` in the reference Worker): accept `scope` (only `"dense"` and
      `"both"` apply there; it writes no segments).
- [x] **Query path.** Worker and both SDKs: slab and delta scans use the dense view. Lexical search, docs
      resolution, `get`, `resolveKeys` and `listKeys` use the lexical view. `search` fusion needs no change.
- [x] **Compaction.** Segment compaction (`compactSegments`, `merge_segments`) applies only the lexical view.
      Dense compaction (folding deltas into slabs, purges) applies only the dense view. The rule for dropping
      tombstone sets (`tombDroppable` in the Worker) has to be per scope: a lexical set can be dropped once no
      segment predates it, a dense set once no slab or delta predates it, and an unscoped set only when both hold.
- [x] **GC and packs.** Tombstone objects stay opaque to both, so check that packs carry `scope` through the
      manifest and that the pack reader builds both views.
- [x] **Tests.**
      - Vector-only upsert by key after a text-only upsert: text search, `get` and `listKeys` still find the
        document, and dense search finds the vector.
      - Lexical-only re-put after a dense upsert: the old vector still ranks, the old text no longer matches, and
        the new text does.
      - Unscoped delete after both: hidden everywhere.
      - Attrs change under `"lexical"` on a dense id is rejected.
      - Compaction of each side keeps the other side's documents.
      - Rollback across a scoped revision.
      - Cross-SDK interop: TS writes a scoped upsert and Python reads it identically, and the reverse.
      - Older-reader refusal on a gated manifest.
- [x] **Docs.** `docs/keys.md` (keyed vector backfill becomes the documented way to add embeddings to a
      text-first index), `docs/update-path.md`, both SDK READMEs (the `scope` option and its rules), and the
      README's write example.

### Related, already known

- Done in 0.2.1: dense compaction runs in the TS SDK (`index.compact`, bounded by memory, cells and time), and
  segment compaction takes `maxInputBytes` (bounded partial merges). A streaming segment merge would still help a
  large base whose single segments pass the budget.
