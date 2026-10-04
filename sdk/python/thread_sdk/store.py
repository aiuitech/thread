"""Blob stores: the SDK reads index objects through the `BlobStore` interface, so the same query code runs
against any S3-compatible endpoint (`S3BlobStore`, see s3.py), a local directory (`FileBlobStore`, e.g. a synced
copy of the bucket), memory (`MemoryBlobStore`, tests and small indexes) and, in front of any of them, a cache of
the immutable index objects (`CachedBlobStore` over a `MemoryCache` or a `DiskCache`), which plays the part the
Workers Cache API plays for the TypeScript SDK.

A store may return a `memoryview` into a larger buffer (the whole object fetched on a cache miss); callers must not
assume they got exactly the bytes asked for in their own buffer. Every method is safe to call from several threads.
"""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Sequence

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
    "phase_stat",
]

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
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, **counts: int) -> None:
        with self._lock:
            for k, v in counts.items():
                setattr(self, k, getattr(self, k) + v)

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return {k: getattr(self, k) for k in ("reads", "hits", "misses", "r206", "r200", "bytes")}


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

    def get(self, key, range=None, *, trace=None, fill=False):
        return self.inner.get(self.prefix + key, range, trace=trace, fill=fill)

    def get_ranges(self, key, ranges, *, trace=None, fill=False):
        return self.inner.get_ranges(self.prefix + key, ranges, trace=trace, fill=fill)

    def head(self, key):
        return self.inner.head(self.prefix + key)

    def put(self, key, data, *, if_absent=False, if_match=None, content_type=None):
        return self.inner.put(self.prefix + key, data, if_absent=if_absent, if_match=if_match,
                              content_type=content_type)

    def list(self, prefix):
        n = len(self.prefix)
        return [k[n:] for k in self.inner.list(self.prefix + prefix)]


# ---------- caches of immutable objects ----------


class BlobCache:
    """Whole immutable objects by key. `get_range` lets a cache serve part of an object without reading it all."""

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


class DiskCache(BlobCache):
    """Whole immutable objects as files in a directory (default limit 4 GB, least recently used evicted first),
    shared by every process that points at it. Range reads read only the range."""

    def __init__(self, directory: str | os.PathLike[str], max_bytes: int = 4 * 1024**3) -> None:
        self.dir = os.path.abspath(os.fspath(directory))
        self.max_bytes = max_bytes
        os.makedirs(self.dir, exist_ok=True)
        self._lock = threading.Lock()
        self._sizes: OrderedDict[str, int] = OrderedDict()
        entries = []
        for name in os.listdir(self.dir):
            if name.startswith(".tmp-"):
                continue
            try:
                st = os.stat(os.path.join(self.dir, name))
            except FileNotFoundError:
                continue
            entries.append((st.st_mtime_ns, name, st.st_size))
        for _, name, size in sorted(entries):
            self._sizes[name] = size
        self.held = sum(self._sizes.values())

    @staticmethod
    def _name(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _touch(self, name: str) -> None:
        with self._lock:
            if name in self._sizes:
                self._sizes.move_to_end(name)

    def get(self, key):
        name = self._name(key)
        try:
            with open(os.path.join(self.dir, name), "rb") as f:
                b = f.read()
        except FileNotFoundError:
            return None
        self._touch(name)
        return b

    def get_range(self, key, offset, length):
        name = self._name(key)
        try:
            with open(os.path.join(self.dir, name), "rb") as f:
                size = os.fstat(f.fileno()).st_size
                if offset + length > size:
                    raise ThreadError(f"cached {key} is {size} bytes, need {offset + length}")
                f.seek(offset)
                b = f.read(length)
        except FileNotFoundError:
            return None
        self._touch(name)
        return b

    def get_ranges(self, key, ranges):
        name = self._name(key)
        try:
            with open(os.path.join(self.dir, name), "rb") as f:
                size = os.fstat(f.fileno()).st_size
                out = []
                for r in ranges:
                    _check_range(key, size, r)
                    f.seek(r.offset)
                    out.append(f.read(r.length))
        except FileNotFoundError:
            return None
        self._touch(name)
        return out

    def put(self, key, data):
        if len(data) > self.max_bytes:
            return
        name = self._name(key)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, os.path.join(self.dir, name))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        evict = []
        with self._lock:
            self.held += len(data) - self._sizes.pop(name, 0)
            self._sizes[name] = len(data)
            while self.held > self.max_bytes and len(self._sizes) > 1:
                old, size = self._sizes.popitem(last=False)
                self.held -= size
                evict.append(old)
        for old in evict:
            try:
                os.unlink(os.path.join(self.dir, old))
            except FileNotFoundError:
                pass


def _not_current(key: str) -> bool:
    return not key.endswith("current.json")


class CachedBlobStore(BlobStore):
    """A cache in front of another store. Index objects under `idx/` are immutable (every revision writes new keys),
    so whole objects are cached and ranges served from them; `current.json` is never cached. A miss of a `fill`
    read fetches the whole object once (concurrent misses of the same key share one fetch) and returns the range
    as a view into it."""

    def __init__(self, inner: BlobStore, cache: BlobCache, *, cacheable: Callable[[str], bool] = _not_current) -> None:
        self.inner = inner
        self.cache = cache
        self.cacheable = cacheable
        self.supports_put = inner.supports_put
        self.supports_list = inner.supports_list
        self.supports_ranges = True
        self.remote = inner.remote
        self._fills: dict[str, Future] = {}
        self._lock = threading.Lock()

    def _fill(self, key: str, trace: BlobTrace | None) -> bytes | None:
        fut: Future = Future()
        with self._lock:
            pending = self._fills.get(key)
            if pending is None:
                self._fills[key] = fut
        if pending is not None:
            return pending.result()
        try:
            full = self.inner.get(key, None, trace=trace)
            full = None if full is None else bytes(full)
            if full is not None:
                self.cache.put(key, full)
            fut.set_result(full)
            return full
        except BaseException as e:
            fut.set_exception(e)
            raise
        finally:
            with self._lock:
                self._fills.pop(key, None)

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
            return self._fill(key, trace)
        if fill:
            full = self._fill(key, trace)
            if full is None:
                return None
            if range.offset + range.length > len(full):
                raise ThreadError(f"{key} is {len(full)} bytes, need {range.offset + range.length}")
            return memoryview(full)[range.offset : range.offset + range.length]
        return self.inner.get(key, range, trace=trace)

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
            full = self._fill(key, trace)
            if full is None:
                return None
            for r in ranges:
                _check_range(key, len(full), r)
            mv = memoryview(full)
            return [mv[r.offset : r.offset + r.length] for r in ranges]
        return self.inner.get_ranges(key, ranges, trace=trace)

    def head(self, key):
        return self.inner.head(key)

    def put(self, key, data, *, if_absent=False, if_match=None, content_type=None):
        return self.inner.put(key, data, if_absent=if_absent, if_match=if_match, content_type=content_type)

    def list(self, prefix):
        return self.inner.list(prefix)


def unique_suffix() -> str:
    return uuid.uuid4().hex[:8]
