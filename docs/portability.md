# Moving an index: copies, snapshots and packs

An index is a set of immutable objects under a prefix, so it moves by copying. This page covers the three ways to do
that from either SDK, when each one fits, and the format of a **pack**: one file holding a whole snapshot, which can
be copied anywhere and queried in place without unpacking it.

| you want to | do |
|---|---|
| mirror a bucket to a laptop, or a laptop to a bucket | `copyObjects(src, dst)` / `copy_objects(src, dst)` |
| ship exactly one revision, nothing stale | `copyObjects(src, dst, ["idx/current.json", ...snapshotKeys(index.revision)])` |
| move it as **one file** and query it in place | `packIndex` / `pack_index`, then `storeAt(file)` / `open_index(file)` |
| write on top of a pack | copy it into a directory or bucket first (`copyObjects(pack, dst)`); packs are read-only |

## Stores you can move between

Every store speaks the same `BlobStore` interface (`get` with ranges, `head`, optional `put` with create-only and
compare-and-swap conditions, optional `list`), so any pair can be copied between:

| store | TypeScript | Python |
|---|---|---|
| R2 binding | `R2BlobStore` (Workers) | - |
| S3-compatible endpoint (R2's S3 API, AWS, MinIO) | `S3BlobStore` | `S3BlobStore` |
| a directory on disk | `FileBlobStore` (`thread-sdk/node`) | `FileBlobStore` |
| one pack file | `PackedBlobStore` (over any store) | `PackedBlobStore` (over any store) |
| memory | `MemoryBlobStore` | `MemoryBlobStore` |

`storeAt(path)` (Node) and `store_at(path)` / `open_index(path)` (Python) open a directory or a pack from a path,
sniffing the pack header.

The directory layout is the bucket layout, one file per object (`{root}/idx/current.json`, `{root}/idx/{gen}/...`).
The two file stores write the same bytes and read each other's directories; the interop test builds an index with
one SDK and queries and extends it with the other. One caveat for *concurrent* writers from both SDKs into the same
directory: the manifest create-only put (a hard link) serializes them correctly, but the compare-and-swap of
`current.json` is guarded by a `.thread-lock.d` directory in Node and an advisory `flock` in Python, which do not
exclude each other. Writers of one language, or writers taking turns, are fine.

## Snapshots

A live prefix holds history: every manifest revision ever committed, slab revisions a compaction has superseded,
delta segments a compaction has folded in. A query only needs what its revision lists. `snapshotKeys(revState)` /
`snapshot_keys(rev_state)` returns exactly that, sorted:

- the manifest `idx/{gen}/manifest[.rev].json`,
- the base files it names (`coarse.f32`, `fine.f32`, `pq.bin`, `slabs[.rev].json`, and a centroid tree if one
  was published),
- one slab per cell at the revision the slabs table names,
- the delta segments, tombstone sets and lexical/docs segments the manifest lists.

`idx/current.json` is not in the list because a snapshot writes its own, pointing at the packed revision. An index
opened from a snapshot is pinned to it: `refresh()` finds nothing newer.

## Packs

```ts
// Node
import { packIndexToFile, storeAt } from "thread-sdk/node";
await packIndexToFile(store, "/backups/idx-2026-10-04.threadpack");           // the current revision
await packIndexToFile(store, "g4-rev12.threadpack", { gen: "g4", rev: 12 });  // a specific one
const index = await thread.open({ store: await storeAt("g4-rev12.threadpack") });

// anywhere (Workers included): a pack as an object in a bucket, read by ranges
const packed = await PackedBlobStore.open(new R2BlobStore(env.BUCKET), "packs/g4-rev12.threadpack");
const index = await thread.open({ store: packed });                           // the Cache API caches its objects
```

```python
from thread_sdk import PackedBlobStore, S3BlobStore, open_index, pack_index

pack_index(store, "/backups/idx-2026-10-04.threadpack")                       # the current revision
pack_index(store, "g4-rev12.threadpack", gen="g4", rev=12)
index = open_index("g4-rev12.threadpack")                                     # one file, zero setup
index = open_index(PackedBlobStore(S3BlobStore(...), "packs/g4-rev12.threadpack"))
```

What a pack gives you:

- **One object to move.** `scp` it, attach it to a release, put it in a bucket, keep it in a Git LFS repo. No
  directory of thousands of slabs to sync, no half-copied index.
- **A consistent snapshot.** It is the revision you asked for and nothing else; stale objects are left behind.
- **Queried in place.** `PackedBlobStore` translates every read into a range read of the pack, so a query costs what
  it costs against a prefix: the scan prefixes of `nprobe` slabs, a few rerank records, two rounds for text.
- **Deterministic.** The two SDKs write byte-identical packs for the same revision (the interop test checks).

What it does not give you: writes. A pack is immutable. To continue an index from a pack, `copyObjects(pack, dst)`
into a directory or bucket and open that; the first upsert commits the next revision there.

**Caches go in front of a pack, never behind it.** `open` already does this: it wraps whatever store you pass in
the Cache API (TypeScript) or a `MemoryCache` for remote stores (Python), so the cache sees the objects' own keys
(`idx/g4/slab/00123`) and a miss fetches one slab. A cache *below* the pack would see one 5 GB key and try to fill
it on a miss; `PackedBlobStore` never forwards `fill` to its inner store for that reason.

### The file format

All integers are little-endian. One file, four regions:

```
offset        size    field
0             8       magic   "THRDPACK"
8             2       version u16 = 1
10            2       flags   u16 = 0
12            4       reserved u32 = 0
16            ...     objects: each begins at an offset that is a multiple of 8; zero padding between them
D             L       directory: UTF-8 JSON (below)
end - 32      8       directory offset D   u64
end - 24      8       directory length L   u64
end - 16      4       entry count          u32
end - 12      2       version              u16 = 1
end - 10      2       flags                u16 = 0
end - 8       8       magic   "THRDPACK"
```

The directory is `{"entries":[[key, offset, length], ...], ...meta}`, entries in the order written, offsets
absolute. `packIndex` writes `idx/current.json` first, then the objects a cold open reads (`baseKeys`: the
manifest, slabs table, tombstone sets, `coarse.f32`, `pq.bin`, the centroid tree and the smallest routing table),
then the rest of `snapshotKeys` in key order, and records `"gen"`, `"rev"` and `"base": [offset, length]`, the
range from `idx/current.json` to the end of the base objects: opening an index on a `PackedBlobStore` reads that
range with ONE range read (`PackedBlobStore.readBase` / `read_base`) and serves the open from it, so a cold open of a
pack in a bucket is head + tail + one range, then the first query; `PackWriter` takes any JSON-serializable `meta`. A reader does one `head` for the
size, one range read of the last 64 KB (footer and, usually, the whole directory), and a second range read only if
the directory is larger than that.

Why the footer is at the end: the writer streams objects as it reads them and never seeks back, so a pack can be
written to a pipe or an upload stream as well as to a file. Why entries are 8-byte aligned: a pack mapped or held in
memory can hand out f32 views without copying. Why JSON: the manifests already are, both SDKs parse it, and a
4096-slab directory is about 200 KB.

Flags and the reserved field are zero in version 1 and are the place to put a checksum or compression scheme later;
a reader must reject a version it does not know.

**Partial entries (warm packs).** An entry may carry a fourth element, `[key, offset, length, {"partial": true,
"size": n}]`: it holds only bytes `[0, length)` of the object `key`, which is `n` bytes in all. The format stays
version 1: a reader that predates the marker ignores extra elements, which is why partial entries are only written to
**warm packs** (`meta.kind: "warm"`), never to index snapshots. A warm pack holds the scan prefixes of an index's
most-probed slabs, keyed by the slabs' own (immutable) keys; `open({ warmPack })` / `open_index(..., warm_pack=)`
fetches it with one GET in the background and serves those prefixes from memory (see
[cold-start.md](cold-start.md#warm-packs)). `PackedBlobStore` serves ranges inside a partial entry, reports the whole
size from `head`, and refuses a read past the prefix rather than return a short object. Unknown fourth elements
(anything without `"partial": true`) are ignored.

## Copying by hand

A pack and a directory are plain files, so the usual tools work: `rsync -a laptop/idx/ bucket-mirror/`, `aws s3
sync`, `rclone`. Copy `idx/{gen}/` before `idx/current.json` if readers may open the destination while the copy is
running, since `current.json` is the only pointer they follow.

## Base packs

A **base pack** is a pack of just what a cold open reads, written per revision by publish / publish-rev next to the
index: `idx/{gen}/base.{rev}.threadpack` holds the manifest, slabs table, tombstone sets, `coarse.f32`, `pq.bin`,
the centroid tree and one routing table (default the smallest, `fine.i8`); `idx/{gen}/base.cold.{rev}.threadpack`
leaves the routing table out, for `coldRouting: "tree"`. Directory metadata `{gen, rev, kind: "base" | "base-cold",
routing}`. The SDKs know its key from `idx/current.json`, so a cold open is two round trips: `current.json`, then
one whole GET of the pack (with the manifest read alongside in case there is none). Its objects are byte-identical
copies of the separate files and keyed by them, so loading from a pack gives the same state as loading the files.
`buildBasePack` / `build_base_pack` write the same bytes in both SDKs (the interop test checks). Details and
numbers: [query-path.md](query-path.md#cold-opens-compact-routing-cold-tree-routing-base-packs).
