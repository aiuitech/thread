# Connect to storage

An index is files, and a **store** is where the files are. You give a store to `open`, and everything else is the
same: the same index opens from a bucket, a folder, one file, or memory.

## Choose a store

| The index is in | TypeScript | Python |
|---|---|---|
| Cloudflare R2, from a Worker | `thread.open({ bucket: env.BUCKET })` | - |
| R2, AWS S3, MinIO or any S3-compatible endpoint | `S3BlobStore` | `S3BlobStore` |
| A folder on disk | `FileBlobStore` (Node) | `FileBlobStore`, or a path |
| One pack file | `storeAt(path)` (Node), `PackedBlobStore` | a path, or `PackedBlobStore` |
| Memory (tests) | `MemoryBlobStore` | `MemoryBlobStore` |

## Cloudflare R2, from a Worker

Use the R2 binding. It is the fastest way: no credentials and no TLS, and reads stay inside Cloudflare.

```ts
const index = await thread.open({
  bucket: env.BUCKET,
  waitUntil: (p) => ctx.waitUntil(p),      // lets cache writes finish after the response
});
```

## Any S3-compatible endpoint

```ts
import { S3BlobStore } from "@aiuitech/thread-sdk";

const store = new S3BlobStore({
  endpoint: "https://<account-id>.r2.cloudflarestorage.com",   // or https://s3.us-east-1.amazonaws.com
  bucket: "my-bucket",
  region: "auto",                                              // AWS: the region of the bucket
  accessKeyId: process.env.S3_KEY,
  secretAccessKey: process.env.S3_SECRET,
});
const index = await thread.open({ store });
```

```python
from thread_sdk import S3BlobStore, open_index

store = S3BlobStore.r2(account_id, "my-bucket", access_key_id, secret_access_key)
# or any endpoint:
store = S3BlobStore("https://s3.us-east-1.amazonaws.com", "my-bucket", key_id, secret, region="us-east-1", path_style=False)

index = open_index(store)
```

The S3 client is part of the SDK. You do not need the AWS SDK or boto3.

## A folder on disk

A folder holds the same layout as a bucket. Use it for development, for tests, and to serve an index from a
machine that has a synced copy.

```ts
import { FileBlobStore } from "@aiuitech/thread-sdk/node";

const index = await thread.open({ store: new FileBlobStore("/data/my-index") });
```

```python
index = open_index("/data/my-index")        # the same as open_index(FileBlobStore("/data/my-index"))
```

## One pack file

A **pack** is one file that holds one revision of an index. You can query it in place, from disk or from a
bucket. A pack is read-only. [Move an index](portability.md) shows how to make one.

```ts
import { storeAt } from "@aiuitech/thread-sdk/node";

const index = await thread.open({ store: await storeAt("/data/docs.threadpack") });

// or a pack that is an object in a bucket, read by range requests (this works in a Worker too)
import { PackedBlobStore, R2BlobStore } from "@aiuitech/thread-sdk";
const packed = await PackedBlobStore.open(new R2BlobStore(env.BUCKET), "packs/docs.threadpack");
const index2 = await thread.open({ store: packed });
```

```python
index = open_index("/data/docs.threadpack")
```

## Caches

Index files never change, so a cache can keep them for ever. `open` adds the right cache for where you are. You
only need to know what it is.

| Where you run | The default cache | What to do |
|---|---|---|
| A Cloudflare Worker | The Workers Cache API (the edge cache) | Pass `waitUntil`. Nothing else. |
| Node or Python, with a remote store | 256 MB in the memory of the process | Add a disk cache if the process restarts often |
| Node or Python, with a folder or a pack file | None. The operating system caches the files. | Nothing |

A disk cache survives restarts, so a new process does not start cold:

```ts
import { DiskCache } from "@aiuitech/thread-sdk/node";

const index = await thread.open({ store, cache: new DiskCache("/var/cache/thread", { maxBytes: 8 * 1024 ** 3 }) });
```

```python
from thread_sdk import DiskCache

index = open_index(store, cache=DiskCache("/var/cache/thread", max_bytes=8 << 30))
```

The TypeScript and Python disk caches use the same layout and can share a folder.

The first query of a new process is slower than the others, because nothing is in the cache yet.
[Cold start](cold-start.md) measures that and lists the options that reduce it. Most applications need none of
them.

## Which revision you read

By default, `open` follows the pointer `idx/current.json`. A search checks the pointer in the background every
30 seconds, and the index moves to a new revision when a writer has committed one. Three options change that:

| Option | Effect |
|---|---|
| `prefix` | Opens the index under a key prefix. Use it for several indexes in one bucket. |
| `gen` | Opens a named generation, not the live one. |
| `gen` and `rev` together | Pins the index to one revision. It never moves. Use it for reproducible results. |
| `refreshInterval` (Python: `refresh_interval`) | How often a query checks for a new revision. The default is 30 seconds. `0` means only when you call `refresh()`. |

## Your own store

A store is a small interface: `get` with a byte range, `head`, and, for writes, `put` with two conditions (create
only, and compare-and-swap). You can put an index in anything that can do that. The interface is in the
[TypeScript reference](../sdk/ts/README.md#blobstore) and the [Python reference](../sdk/python/README.md#blobstore).

## Next

- [Write documents](writing.md)
- [Move an index](portability.md): copy between stores, and make packs.
