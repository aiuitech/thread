"""Blob stores: the SDK reads index objects through the `BlobStore` interface, so the same query code runs
against any S3-compatible endpoint (`S3BlobStore`, see s3.py), a local directory (`FileBlobStore`, e.g. a synced
copy of the bucket), memory (`MemoryBlobStore`, tests and small indexes) and, in front of any of them, a cache of
the immutable index objects (`CachedBlobStore` over a `MemoryCache` or a `DiskCache`), which plays the part the
Workers Cache API plays for the TypeScript SDK. What a cache miss reads is the cache store's `miss_policy` (see
`CachedBlobStore`).

A store may return a `memoryview` into a larger buffer (the whole object fetched on a cache miss); callers must not
assume they got exactly the bytes asked for in their own buffer. Every method is safe to call from several threads.
"""

from __future__ import annotations

import errno
import hashlib
import os
import logging
import tempfile
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Sequence

from .wire import ThreadError

try:  # POSIX advisory locks serialize conditional puts across processes sharing a FileBlobStore
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "BlobRange",
    "BlobHead",
    "BlobTrace",
    "BlobStore",
    "MemoryBlobStore",
    "FileBlobStore",
    "PrefixedBlobStore",
    "BlobCache",
    "MemoryCache",
    "DiskCache",
    "CachedBlobStore",
    "MISS_POLICIES",
    "phase_stat",
]

log = logging.getLogger("thread_sdk")

#: What a cache store does on a miss of a `fill` range read (a slab's scan prefix); see `CachedBlobStore`.
MISS_POLICIES = ("whole", "range", "range+warm")

Bytes = bytes | bytearray | memoryview


class BlobRange(NamedTuple):
    offset: int
    length: int


@dataclass(frozen=True)
class BlobHead:
    size: int
    #: Opaque version tag, used for compare-and-swap puts.
    etag: str


@dataclass
class BlobTrace:
    """Per-call read counters a caller may pass to have the store account its work."""

    #: Reads that reached the origin store (S3/disk/memory).
    reads: int = 0
    #: Cache outcomes (CachedBlobStore only).
    hits: int = 0
    misses: int = 0
    #: Range reads the cache served from a cached object (always, for the local caches) or as the whole object.
    r206: int = 0
    r200: int = 0
    #: Bytes transferred into the process.
    bytes: int = 0
    #: Cache stores with the "range+warm" miss policy: whole-object fetches this call's misses queued in the
    #: background (not part of `reads` / `bytes`; `CachedBlobStore.warm_stats` counts what they transferred).
    warms: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, **counts: int) -> None:
        with self._lock:
            for k, v in counts.items():
                setattr(self, k, getattr(self, k) + v)

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return {k: getattr(self, k) for k in ("reads", "hits", "misses", "r206", "r200", "bytes", "warms")}


TRACE_FIELDS = ("reads", "hits", "misses", "r206", "r200", "bytes")


def phase_stat(ms: float, after: dict[str, int] | BlobTrace, before: dict[str, int] | BlobTrace | None = None
               ) -> dict[str, float | int]:
    """One step of a cold-start timeline (`index.open_stats`, `stats["timeline"]`): its wall time in ms and the store
    work done during it (the difference of two trace snapshots; `before` defaults to an empty trace). The TypeScript
    SDK's `PhaseStat` has the same fields."""
    a = after.as_dict() if isinstance(after, BlobTrace) else after
    b = (before.as_dict() if isinstance(before, BlobTrace) else before) if before is not None else None
    return {"ms": round(ms, 3), **{k: a[k] - (b[k] if b else 0) for k in TRACE_FIELDS}}


def _trace_read(trace: BlobTrace | None, n: int) -> None:
    if trace is not None:
        trace.add(reads=1, bytes=n)


class BlobStore:
    """Read (and optionally write) access to the objects of an index. Subclasses implement `get` and `head`, and
    set `supports_put` / `supports_list` when they implement `put` / `list`."""

    supports_put = False
    supports_list = False
    #: Whether `get_ranges` can serve several ranges in one call (else the SDK reads ranges in parallel).
    supports_ranges = False
    #: Whether reads cross a network (`open` puts a memory cache in front of remote stores by default).
    remote = False
    #: Preferred reads in flight per query (a query's `inflight` default; 12 when None).
    inflight: int | None = None
    #: Preferred phase-2 range merging (a query's `coalesce_bytes` / `span_bytes` defaults when not None).
    coalesce_bytes: int | None = None
    span_bytes: int | None = None

    def contains(self, key: str) -> bool | None:
        """Whether `key` can be read without an origin request (in a local cache); None when the store cannot tell.
        Used by the cold-path query policy."""
        return None

    def warmup(self, n: int, executor: ThreadPoolExecutor | None = None) -> list[Future]:
        """Opens up to `n` connections ahead of the first query (remote stores; a no-op here). With `executor`, the
        work runs on its threads (whose connections the queries then reuse)."""
        return []

    def get(self, key: str, range: BlobRange | None = None, *, trace: BlobTrace | None = None,
            fill: bool = False) -> Bytes | None:
        """The object, or `range` of it; None if it does not exist. `fill` (cache stores only): on a miss of a range
        read, fetch the WHOLE object, cache it and return the range as a view into it."""
        raise NotImplementedError

    def get_ranges(self, key: str, ranges: Sequence[BlobRange], *, trace: BlobTrace | None = None,
                   fill: bool = False) -> list[Bytes] | None:
        """Several ranges of one object in one request, one buffer per range in order; None when the store cannot
        (the caller then falls back to single-range `get`s)."""
        return None

    def head(self, key: str) -> BlobHead | None:
        raise NotImplementedError

    def get_tail(self, key: str, length: int, *, trace: BlobTrace | None = None) -> tuple[Bytes, BlobHead] | None:
        """The last `length` bytes of an object (all of it when shorter) and its size / etag, in ONE request
        (`Range: bytes=-length`), so a reader of a trailer (a pack's footer and directory) needs no `head` first.
        Raises NotImplementedError when the store cannot (callers fall back to `head` + a range); None if the object
        does not exist."""
        raise NotImplementedError

    def put(self, key: str, data: Bytes | str, *, if_absent: bool = False, if_match: str | None = None,
            content_type: str | None = None) -> BlobHead | None:
        """Writes an object; None when a put condition failed (`if_absent`: create-only; `if_match`: compare the
        current etag)."""
        raise ThreadError("this store is read-only (no put)", 400)

    def list(self, prefix: str) -> list[str]:
        """Keys under `prefix`."""
        raise ThreadError("this store cannot list objects", 400)


def _encode(data: Bytes | str) -> bytes:
    return data.encode("utf-8") if isinstance(data, str) else bytes(data)


def _check_range(key: str, size: int, r: BlobRange) -> None:
    if r.offset < 0 or r.length < 0 or r.offset + r.length > size:
        raise ThreadError(f"{key} [{r.offset}, +{r.length}) is outside the object ({size} bytes)")


# ---------- in-memory store (tests, small local indexes) ----------


class MemoryBlobStore(BlobStore):
    supports_put = True
    supports_list = True
    supports_ranges = True

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def get(self, key, range=None, *, trace=None, fill=False):
        o = self.objects.get(key)
        if o is None:
            return None
        b = o[0]
        if range is not None:
            _check_range(key, len(b), range)
            b = b[range.offset : range.offset + range.length]
        _trace_read(trace, len(b))
        return b

    def get_ranges(self, key, ranges, *, trace=None, fill=False):
        o = self.objects.get(key)
        if o is None:
            return None
        out = []
        for r in ranges:
            _check_range(key, len(o[0]), r)
            out.append(o[0][r.offset : r.offset + r.length])
        _trace_read(trace, sum(len(b) for b in out))
        return out

    def head(self, key):
        o = self.objects.get(key)
        return BlobHead(len(o[0]), o[1]) if o else None

    def get_tail(self, key, length, *, trace=None):
        o = self.objects.get(key)
        if o is None:
            return None
        b = o[0][max(0, len(o[0]) - length):]
        _trace_read(trace, len(b))
        return b, BlobHead(len(o[0]), o[1])

    def put(self, key, data, *, if_absent=False, if_match=None, content_type=None):
        b = _encode(data)
        with self._lock:
            cur = self.objects.get(key)
            if if_absent and cur is not None:
                return None
            if if_match is not None and (cur is None or cur[1] != if_match):
                return None
            self._seq += 1
            etag = f"m{self._seq}"
            self.objects[key] = (b, etag)
        return BlobHead(len(b), etag)

    def list(self, prefix):
        return sorted(k for k in list(self.objects) if k.startswith(prefix))


# ---------- local directory ----------


class FileBlobStore(BlobStore):
    """An index in a local directory (`{root}/idx/current.json`, ...): a synced copy of a bucket, or a writable
    local index. Puts are atomic (temp file + rename); create-only and compare-and-swap puts are serialized across
    processes with an advisory lock on `{root}/.thread-lock` (POSIX)."""

    supports_put = True
    supports_list = True
    supports_ranges = True

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = os.path.abspath(os.fspath(root))
        self._lock = threading.Lock()

    def path(self, key: str) -> str:
        parts = key.split("/")
        if not key or key.startswith("/") or any(p in ("", ".", "..") for p in parts[:-1]) or parts[-1] in (".", ".."):
            raise ThreadError(f"bad object key {key!r}", 400)
        return os.path.join(self.root, *parts)

    @staticmethod
    def _etag(st: os.stat_result) -> str:
        # Every put renames a fresh temp file into place, so the inode changes with each write.
        return f"{st.st_ino:x}-{st.st_mtime_ns:x}-{st.st_size:x}"

    def get(self, key, range=None, *, trace=None, fill=False):
        try:
            with open(self.path(key), "rb") as f:
                if range is None:
                    b = f.read()
                else:
                    size = os.fstat(f.fileno()).st_size
                    _check_range(key, size, range)
                    f.seek(range.offset)
                    b = f.read(range.length)
        except FileNotFoundError:
            return None
        _trace_read(trace, len(b))
        return b

    def get_ranges(self, key, ranges, *, trace=None, fill=False):
        try:
            with open(self.path(key), "rb") as f:
                size = os.fstat(f.fileno()).st_size
                out = []
                for r in ranges:
                    _check_range(key, size, r)
                    f.seek(r.offset)
                    out.append(f.read(r.length))
        except FileNotFoundError:
            return None
        _trace_read(trace, sum(len(b) for b in out))
        return out

    def head(self, key):
        try:
            st = os.stat(self.path(key))
        except FileNotFoundError:
            return None
        return BlobHead(st.st_size, self._etag(st))

    def get_tail(self, key, length, *, trace=None):
        try:
            with open(self.path(key), "rb") as f:
                st = os.fstat(f.fileno())
                n = min(st.st_size, length)
                f.seek(st.st_size - n)
                b = f.read(n)
        except FileNotFoundError:
            return None
        _trace_read(trace, len(b))
        return b, BlobHead(st.st_size, self._etag(st))

    def put(self, key, data, *, if_absent=False, if_match=None, content_type=None):
        path = self.path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(_encode(data))
            if if_absent:
                try:
                    os.link(tmp, path)  # atomic create-only
                except FileExistsError:
                    return None
                except OSError as e:  # no hard links on this filesystem: fall back to the lock
                    if e.errno not in (errno.EPERM, errno.ENOTSUP, errno.EXDEV):
                        raise
                    with self._exclusive():
                        if os.path.exists(path):
                            return None
                        os.replace(tmp, path)
                        tmp = ""
            elif if_match is not None:
                with self._exclusive():
                    cur = self.head(key)
                    if cur is None or cur.etag != if_match:
                        return None
                    os.replace(tmp, path)
                    tmp = ""
            else:
                os.replace(tmp, path)
                tmp = ""
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        h = self.head(key)
        assert h is not None
        return h

    def _exclusive(self):
        store = self

        class _Lock:
            def __enter__(self):
                store._lock.acquire()
                self.f = None
                if fcntl is not None:
                    os.makedirs(store.root, exist_ok=True)
                    self.f = open(os.path.join(store.root, ".thread-lock"), "a")
                    fcntl.flock(self.f, fcntl.LOCK_EX)

            def __exit__(self, *exc):
                if self.f is not None:
                    fcntl.flock(self.f, fcntl.LOCK_UN)
                    self.f.close()
                store._lock.release()

        return _Lock()

    def list(self, prefix):
        # Walk the deepest directory the prefix names and filter by the full prefix.
        head, _, _ = prefix.rpartition("/")
        base = os.path.join(self.root, *head.split("/")) if head else self.root
        out = []
        for dirpath, dirnames, filenames in os.walk(base):
            rel = os.path.relpath(dirpath, self.root)
            rel = "" if rel == "." else rel.replace(os.sep, "/") + "/"
            dirnames[:] = [d for d in dirnames if (rel + d + "/").startswith(prefix) or prefix.startswith(rel + d + "/")]
            for name in filenames:
                k = rel + name
                if k.startswith(prefix) and not name.startswith(".tmp-") and k != ".thread-lock":
                    out.append(k)
        return sorted(out)


# ---------- prefix ----------


class PrefixedBlobStore(BlobStore):
    """A store with every key prefixed (one bucket holding several indexes, e.g. `tenants/a/`)."""

    def __init__(self, inner: BlobStore, prefix: str) -> None:
        self.inner = inner
        self.prefix = prefix
        self.supports_put = inner.supports_put
        self.supports_list = inner.supports_list
        self.supports_ranges = inner.supports_ranges
        self.remote = inner.remote
        self.inflight = inner.inflight
        self.coalesce_bytes = inner.coalesce_bytes
        self.span_bytes = inner.span_bytes

    def contains(self, key):
        return self.inner.contains(self.prefix + key)

    def warmup(self, n, executor=None):
        return self.inner.warmup(n, executor)

    def get(self, key, range=None, *, trace=None, fill=False):
        return self.inner.get(self.prefix + key, range, trace=trace, fill=fill)

    def get_ranges(self, key, ranges, *, trace=None, fill=False):
        return self.inner.get_ranges(self.prefix + key, ranges, trace=trace, fill=fill)

    def head(self, key):
        return self.inner.head(self.prefix + key)

    def get_tail(self, key, length, *, trace=None):
        return self.inner.get_tail(self.prefix + key, length, trace=trace)

    def put(self, key, data, *, if_absent=False, if_match=None, content_type=None):
        return self.inner.put(self.prefix + key, data, if_absent=if_absent, if_match=if_match,
                              content_type=content_type)

    def list(self, prefix):
        n = len(self.prefix)
        return [k[n:] for k in self.inner.list(self.prefix + prefix)]


# ---------- caches of immutable objects ----------


class BlobCache:
    """Whole immutable objects by key. `get_range` lets a cache serve part of an object without reading it all."""

    #: The miss policy a `CachedBlobStore` over this cache uses by default. In-process caches warm in the background;
    #: a cache shared by many processes (an edge cache) may prefer "whole".
    default_miss_policy = "range+warm"

    def get(self, key: str) -> Bytes | None:
        raise NotImplementedError

    def get_range(self, key: str, offset: int, length: int) -> Bytes | None:
        b = self.get(key)
        if b is None:
            return None
        if offset + length > len(b):
            raise ThreadError(f"cached {key} is {len(b)} bytes, need {offset + length}")
        return memoryview(b)[offset : offset + length]

    def get_ranges(self, key: str, ranges: Sequence[BlobRange]) -> list[Bytes] | None:
        """Several ranges of one cached object, or None if it is not cached."""
        b = self.get(key)
        if b is None:
            return None
        for r in ranges:
            _check_range(key, len(b), r)
        mv = memoryview(b)
        return [mv[r.offset : r.offset + r.length] for r in ranges]

    def contains(self, key: str) -> bool:
        """Whether the object is cached (no I/O beyond a lookup)."""
        return self.get(key) is not None

    def put(self, key: str, data: bytes) -> None:
        raise NotImplementedError


class MemoryCache(BlobCache):
    """An LRU of whole objects in process memory (default 256 MB). Objects larger than `max_bytes / 4` are not
    kept."""

    def __init__(self, max_bytes: int = 256 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes
        self.held = 0
        self._items: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def contains(self, key):
        return key in self._items

    def get(self, key):
        with self._lock:
            b = self._items.get(key)
            if b is not None:
                self._items.move_to_end(key)
            return b

    def put(self, key, data):
        if len(data) > self.max_bytes // 4:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self.held -= len(old)
            self._items[key] = data
            self.held += len(data)
            while self.held > self.max_bytes and self._items:
                _, ev = self._items.popitem(last=False)
                self.held -= len(ev)


#: A hit refreshes the file's mtime at most this often (seconds), so the LRU order survives a restart.
_TOUCH_EVERY = 60.0


class DiskCache(FileBlobStore, BlobCache):
    """Whole immutable objects as files in a directory (default limit 4 GB, least recently used evicted first), shared
    by every process that points at it. The directory has the bucket layout (`{dir}/idx/{gen}/slab/00123`), the
    same one the TypeScript SDK's `DiskCache` writes, so the two SDKs can share a cache directory, and the cache is
    also a directory store (`FileBlobStore`): `copy_objects(s3, disk_cache, snapshot_keys(rev))` pre-warms it with one
    revision. Range reads read only the range. The LRU order is kept in memory and, coarsely, in the files' mtimes,
    which the startup scan sorts by; a file another process evicted reads as a miss."""

    def __init__(self, directory: str | os.PathLike[str], max_bytes: int = 4 * 1024**3, *,
                 miss_policy: str = "range+warm") -> None:
        super().__init__(directory)
        self.dir = self.root
        self.max_bytes = max_bytes
        self.default_miss_policy = miss_policy
        os.makedirs(self.dir, exist_ok=True)
        self._lru_lock = threading.Lock()
        #: key -> [size, last touch (s)], least recently used first
        self._sizes: OrderedDict[str, list[float]] = OrderedDict()
        entries = []
        for dirpath, dirnames, filenames in os.walk(self.dir):
            rel = os.path.relpath(dirpath, self.dir)
            rel = "" if rel == "." else rel.replace(os.sep, "/") + "/"
            dirnames[:] = [d for d in dirnames if not (rel == "" and d == ".thread-lock.d")]
            for name in filenames:
                if name.startswith(".tmp-") or name == ".thread-lock":
                    continue
                try:
                    st = os.stat(os.path.join(dirpath, name))
                except FileNotFoundError:
                    continue
                entries.append((st.st_mtime, rel + name, st.st_size))
        for mtime, key, size in sorted(entries):
            self._sizes[key] = [size, mtime]
        self.held = sum(int(v[0]) for v in self._sizes.values())

    def contains(self, key):
        return key in self._sizes

    def _touch(self, key: str, size: int | None) -> None:
        now = time.time()
        touch = False
        with self._lru_lock:
            e = self._sizes.get(key)
            if e is None:
                if size is None:
                    return
                self._sizes[key] = [size, now]
                self.held += size
                evict = self._evict_locked()
            else:
                self._sizes.move_to_end(key)
                evict = []
                if now - e[1] > _TOUCH_EVERY:
                    e[1] = now
                    touch = True
        if touch:
            try:
                os.utime(self.path(key))
            except OSError:
                pass
        self._unlink(evict)

    def _forget(self, key: str) -> None:
        with self._lru_lock:
            e = self._sizes.pop(key, None)
            if e is not None:
                self.held -= int(e[0])

    def _evict_locked(self) -> list[str]:
        out = []
        while self.held > self.max_bytes and len(self._sizes) > 1:
            old, e = self._sizes.popitem(last=False)
            self.held -= int(e[0])
            out.append(old)
        return out

    def _unlink(self, keys_: list[str]) -> None:
        for k in keys_:
            try:
                os.unlink(self.path(k))
            except FileNotFoundError:
                pass

    def get(self, key, range=None, *, trace=None, fill=False):
        b = super().get(key, range, trace=trace)
        if b is None:
            self._forget(key)
        else:
            self._touch(key, len(b) if range is None else None)
        return b

    def get_range(self, key, offset, length):
        try:
            return self.get(key, BlobRange(offset, length))
        except ThreadError as e:
            raise ThreadError(f"cached {key}: {e}") from e

    def get_ranges(self, key, ranges, *, trace=None, fill=False):
        out = super().get_ranges(key, ranges, trace=trace)
        if out is None:
            self._forget(key)
        else:
            self._touch(key, None)
        return out

    def put(self, key, data, *, if_absent=False, if_match=None, content_type=None):
        b = _encode(data)
        if len(b) > self.max_bytes:
            return None
        h = super().put(key, b, if_absent=if_absent, if_match=if_match, content_type=content_type)
        if h is not None:
            with self._lru_lock:
                old = self._sizes.pop(key, None)
                if old is not None:
                    self.held -= int(old[0])
                self._sizes[key] = [h.size, time.time()]
                self.held += h.size
                evict = self._evict_locked()
            self._unlink(evict)
        return h


def _not_current(key: str) -> bool:
    return not key.endswith("current.json")


class CachedBlobStore(BlobStore):
    """A cache in front of another store. Index objects under `idx/` are immutable (every revision writes new keys),
    so whole objects are cached and ranges served from them; `current.json` is never cached.

    `miss_policy` decides what a miss of a `fill` range read (a slab's scan prefix) does:

    - "whole": read the whole object once (concurrent misses of the same key share one read), cache it and return
      the range as a view into it, so the query also has that slab's rerank records in memory. Right for a cache
      many processes share, where the first query pays the transfer for everybody.
    - "range": read exactly the range from the origin and cache nothing (whole-object reads are not cached either).
    - "range+warm" (default for `MemoryCache` / `DiskCache`): read exactly the range now and fetch the whole object
      into the cache in the background: `warm_concurrency` fetches at once (one while foreground reads are in
      flight), at most `warm_queue` queued, deduplicated, never on the query's critical path.

    Reads without `fill` (rerank records, lexical postings) never fetch whole objects."""

    def __init__(self, inner: BlobStore, cache: BlobCache, *, cacheable: Callable[[str], bool] = _not_current,
                 miss_policy: str | None = None, warm_concurrency: int = 2, warm_queue: int = 1024,
                 inflight: int | None = None, coalesce_bytes: int | None = None,
                 span_bytes: int | None = None) -> None:
        self.inner = inner
        self.cache = cache
        self.cacheable = cacheable
        self.miss_policy = miss_policy or getattr(cache, "default_miss_policy", "range+warm")
        if self.miss_policy not in MISS_POLICIES:
            raise ThreadError(f"miss_policy must be one of {', '.join(MISS_POLICIES)}", 400)
        self.supports_put = inner.supports_put
        self.supports_list = inner.supports_list
        self.supports_ranges = True
        self.remote = inner.remote
        self.inflight = inflight if inflight is not None else inner.inflight
        self.coalesce_bytes = coalesce_bytes if coalesce_bytes is not None else inner.coalesce_bytes
        self.span_bytes = span_bytes if span_bytes is not None else inner.span_bytes
        self.warm_concurrency = warm_concurrency
        self.warm_queue = warm_queue
        #: Background warm fetches: queued, fetched, bytes, failed, dropped (queue full).
        self.warm_stats = {"queued": 0, "fetched": 0, "bytes": 0, "failed": 0, "dropped": 0}
        self._fills: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._warm_q: deque[str] = deque()
        self._warming: set[str] = set()
        self._warm_active = 0
        self._foreground = 0
        self._idle = threading.Condition(self._lock)
        self._warm_pool: ThreadPoolExecutor | None = None

    def contains(self, key):
        if not self.cacheable(key):
            return self.inner.contains(key)
        return self.cache.contains(key) or bool(self.inner.contains(key))

    def warmup(self, n, executor=None):
        return self.inner.warmup(n, executor)

    def _origin(self, fn: Callable[[], Any]) -> Any:
        """Runs a foreground origin read (counted, so background warms yield to it)."""
        with self._lock:
            self._foreground += 1
        try:
            return fn()
        finally:
            with self._lock:
                self._foreground -= 1
                idle = self._foreground == 0
            if idle:
                self._pump()

    def _fill(self, key: str, trace: BlobTrace | None, cache: bool = True) -> bytes | None:
        fut: Future = Future()
        with self._lock:
            pending = self._fills.get(key)
            if pending is None:
                self._fills[key] = fut
        if pending is not None:
            return pending.result()
        try:
            full = self._origin(lambda: self.inner.get(key, None, trace=trace))
            full = None if full is None else bytes(full)
            if full is not None and cache:
                self.cache.put(key, full)
            fut.set_result(full)
            return full
        except BaseException as e:
            fut.set_exception(e)
            raise
        finally:
            with self._lock:
                self._fills.pop(key, None)
                self._idle.notify_all()

    def _warm(self, key: str, trace: BlobTrace | None) -> None:
        """Queues a background whole-object fetch of `key` ("range+warm")."""
        if self.warm_concurrency <= 0 or self.cache.contains(key):
            return
        with self._lock:
            if key in self._warming or key in self._fills:
                return
            if len(self._warm_q) >= self.warm_queue:
                self.warm_stats["dropped"] += 1
                return
            self._warming.add(key)
            self._warm_q.append(key)
            self.warm_stats["queued"] += 1
        if trace is not None:
            trace.add(warms=1)
        self._pump()

    def _pump(self) -> None:
        starts = []
        with self._lock:
            cap = self.warm_concurrency if self._foreground == 0 else min(1, self.warm_concurrency)
            while self._warm_active < cap and self._warm_q:
                self._warm_active += 1
                starts.append(self._warm_q.popleft())
            if starts and self._warm_pool is None:
                self._warm_pool = ThreadPoolExecutor(max_workers=max(1, self.warm_concurrency),
                                                     thread_name_prefix="thread-warm")
        for key in starts:
            try:
                self._warm_pool.submit(self._warm_one, key)  # type: ignore[union-attr]
            except RuntimeError:  # interpreter shutting down
                self._warm_done(key)

    def _warm_one(self, key: str) -> None:
        try:
            if not self.cache.contains(key):
                full = self.inner.get(key)
                if full is not None:
                    full = bytes(full)
                    self.cache.put(key, full)
                    with self._lock:
                        self.warm_stats["fetched"] += 1
                        self.warm_stats["bytes"] += len(full)
        except Exception as e:  # noqa: BLE001 - a failed warm only means a later miss reads the origin
            with self._lock:
                self.warm_stats["failed"] += 1
            log.error("thread_sdk: warm fetch of %s failed: %s", key, e)
        finally:
            self._warm_done(key)

    def _warm_done(self, key: str) -> None:
        with self._lock:
            self._warm_active -= 1
            self._warming.discard(key)
            self._idle.notify_all()
        self._pump()

    def drain(self, timeout: float | None = None) -> bool:
        """Waits until no background warm fetch or whole-object fill is queued or running (tests, benchmarks).
        Returns False on timeout."""
        with self._lock:
            return self._idle.wait_for(lambda: not (self._warm_active or self._warm_q or self._fills), timeout)

    def close(self) -> None:
        """Stops the warm thread (queued warms are dropped)."""
        with self._lock:
            self._warm_q.clear()
            pool, self._warm_pool = self._warm_pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _miss_fill(self, key: str, ranges: Sequence[BlobRange], single: bool, trace: BlobTrace | None) -> Any:
        if self.miss_policy == "whole":
            full = self._fill(key, trace)
            if full is None:
                return None
            for r in ranges:
                _check_range(key, len(full), r)
            mv = memoryview(full)
            return [mv[r.offset : r.offset + r.length] for r in ranges]
        if single:
            b = self._origin(lambda: self.inner.get(key, ranges[0], trace=trace))
            got = None if b is None else [b]
        else:
            got = self._origin(lambda: self.inner.get_ranges(key, ranges, trace=trace))
        if got is not None and self.miss_policy == "range+warm":
            self._warm(key, trace)
        return got

    def get(self, key, range=None, *, trace=None, fill=False):
        if not self.cacheable(key):
            return self.inner.get(key, range, trace=trace, fill=fill)
        b = self.cache.get(key) if range is None else self.cache.get_range(key, range.offset, range.length)
        if b is not None:
            if trace is not None:
                trace.add(hits=1, bytes=len(b), r206=1 if range is not None else 0)
            return b
        if trace is not None:
            trace.add(misses=1)
        if range is None:
            return self._fill(key, trace, cache=self.miss_policy != "range")
        if fill:
            got = self._miss_fill(key, [range], True, trace)
            return None if got is None else got[0]
        return self._origin(lambda: self.inner.get(key, range, trace=trace))

    def get_ranges(self, key, ranges, *, trace=None, fill=False):
        if not ranges:
            return []
        if not self.cacheable(key):
            return self.inner.get_ranges(key, ranges, trace=trace, fill=fill)
        parts = self.cache.get_ranges(key, ranges)
        if parts is not None:
            if trace is not None:
                trace.add(hits=1, r206=1, bytes=sum(len(p) for p in parts))
            return parts
        if trace is not None:
            trace.add(misses=1)
        if fill:
            return self._miss_fill(key, ranges, False, trace)
        return self._origin(lambda: self.inner.get_ranges(key, ranges, trace=trace))

    def get_tail(self, key, length, *, trace=None):
        """The last `length` bytes and the size: from the cache when it holds the whole object, else ONE origin
        request (a tail is not cached: a pack reader reads it once per open)."""
        full = self.cache.get(key) if self.cacheable(key) and self.cache.contains(key) else None
        if full is not None:
            b = memoryview(full)[max(0, len(full) - length):]
            if trace is not None:
                trace.add(hits=1, bytes=len(b))
            return b, BlobHead(len(full), "")
        return self._origin(lambda: self.inner.get_tail(key, length, trace=trace))

    def head(self, key):
        return self.inner.head(key)

    def put(self, key, data, *, if_absent=False, if_match=None, content_type=None):
        return self.inner.put(key, data, if_absent=if_absent, if_match=if_match, content_type=content_type)

    def list(self, prefix):
        return self.inner.list(prefix)


def unique_suffix() -> str:
    return uuid.uuid4().hex[:8]
