"""Packs: one immutable file holding a whole index snapshot, so an index can be moved as a single object and queried
anywhere a `BlobStore` can range-read it (a directory, R2, S3, memory). `pack_index` writes the objects one revision
reads (`snapshot_keys`) plus an `idx/current.json` naming it; `PackedBlobStore` serves them back under their original
keys by translating reads into ranges of the pack. A pack is read-only: to write on top, `copy_objects` it into a
directory or a bucket.

Layout (all integers little-endian; docs/portability.md has the specification):

    [0, 16)      header: magic "THRDPACK", version u16 (1), flags u16 (0), reserved u32 (0)
    [16, D)      the objects, each at an 8-byte-aligned offset, zero padding between
    [D, D + L)   directory: UTF-8 JSON {"entries":[[key, offset, length], ...], ...meta}; an entry may carry a
                 fourth element {"partial": true, "size": n}: it holds only bytes [0, length) of an object of n bytes
                 (warm packs, see warm.py)
    [end - 32)   footer: directory offset u64, directory length u64, entry count u32, version u16, flags u16,
                 magic "THRDPACK"

The footer comes last so a pack streams out without seeking back; a reader does one `head` and one or two range
reads to open it. The TypeScript SDK writes and reads the same bytes (`sdk/ts/src/pack.ts`).
"""

from __future__ import annotations

import io
import os
import struct
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Iterable, Sequence

from .query import RevState
from .revision import base_keys, load_rev_state, resolve_target, snapshot_keys
from .store import BlobHead, BlobRange, BlobStore, BlobTrace, Bytes, FileBlobStore, MemoryBlobStore, phase_stat
from .wire import ThreadError, choose_routing, decode_json, dumps, has_dense, keys as index_keys

__all__ = [
    "PACK_MAGIC",
    "PACK_VERSION",
    "PACK_HEADER_BYTES",
    "PACK_FOOTER_BYTES",
    "PackedBlobStore",
    "PackWriter",
    "PackResult",
    "pack_index",
    "pack_to_bytes",
    "copy_objects",
    "store_at",
    "is_pack_header",
    "pack_entries",
    "build_base_pack",
    "BasePackResult",
    "PrefetchStore",
]

PACK_MAGIC = b"THRDPACK"
PACK_VERSION = 1
PACK_HEADER_BYTES = 16
PACK_FOOTER_BYTES = 32
#: Bytes a reader fetches from the end of a pack to find the footer and, in the same request, the directory (~43 bytes
#: per entry: 512 KB covers ~12,000 objects).
PACK_TAIL_READ = 512 * 1024
_ALIGN = 8
_HEADER = struct.Struct("<8sHHI")
_FOOTER = struct.Struct("<QQIHH8s")


def is_pack_header(b: Bytes) -> bool:
    """Whether `b` starts with a pack header (the first 16 bytes of a file are enough)."""
    if len(b) < PACK_HEADER_BYTES:
        return False
    magic, version, _flags, _res = _HEADER.unpack_from(b, 0)
    return magic == PACK_MAGIC and version == PACK_VERSION


def pack_header() -> bytes:
    return _HEADER.pack(PACK_MAGIC, PACK_VERSION, 0, 0)


def pack_footer(dir_off: int, dir_len: int, count: int) -> bytes:
    return _FOOTER.pack(dir_off, dir_len, count, PACK_VERSION, 0, PACK_MAGIC)


def parse_pack_footer(f: Bytes) -> tuple[int, int, int]:
    """(directory offset, directory length, entry count) from the last 32 bytes of a pack."""
    if len(f) != PACK_FOOTER_BYTES:
        raise ThreadError(f"pack footer needs {PACK_FOOTER_BYTES} bytes, have {len(f)}")
    dir_off, dir_len, count, version, _flags, magic = _FOOTER.unpack(bytes(f))
    if magic != PACK_MAGIC:
        raise ThreadError("not a thread pack (bad footer magic)")
    if version != PACK_VERSION:
        raise ThreadError(f"unknown pack version {version}")
    return dir_off, dir_len, count


def parse_pack_directory(b: Bytes, dir_off: int, count: int) -> tuple[dict[str, tuple[int, int]], dict[str, Any]]:
    """Entries by key and the writer's metadata; checks every entry lies inside `[PACK_HEADER_BYTES, dir_off)`."""
    entries, meta, _partial = parse_pack_directory_full(b, dir_off, count)
    return entries, meta


def parse_pack_directory_full(b: Bytes, dir_off: int, count: int
                              ) -> tuple[dict[str, tuple[int, int]], dict[str, Any], dict[str, int]]:
    """`parse_pack_directory` plus the partial entries: key -> size of the whole object, for entries marked
    `{"partial": true, "size": n}` (they hold only the object's first `length` bytes)."""
    j = decode_json(b)
    if not isinstance(j, dict) or not isinstance(j.get("entries"), list):
        raise ThreadError("pack directory has no entries")
    if len(j["entries"]) != count:
        raise ThreadError(f"pack directory lists {len(j['entries'])} entries, footer says {count}")
    entries: dict[str, tuple[int, int]] = {}
    partial: dict[str, int] = {}
    for e in j["entries"]:
        if (not isinstance(e, list) or len(e) < 3 or not isinstance(e[0], str) or isinstance(e[1], bool)
                or not isinstance(e[1], int) or isinstance(e[2], bool) or not isinstance(e[2], int)):
            raise ThreadError("pack directory: bad entry")
        key, off, ln = e[0], e[1], e[2]
        if off < PACK_HEADER_BYTES or ln < 0 or off + ln > dir_off:
            raise ThreadError(f"pack entry {key} [{off}, +{ln}) is outside the data area")
        if key in entries:
            raise ThreadError(f"pack directory lists {key} twice")
        entries[key] = (off, ln)
        x = e[3] if len(e) > 3 else None
        if isinstance(x, dict) and x.get("partial") is True:
            size = x.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < ln:
                raise ThreadError(f"pack entry {key}: partial entry needs a size >= its length {ln}")
            partial[key] = size
    meta = {k: v for k, v in j.items() if k != "entries"}
    return entries, meta, partial


def pack_entries(data: Bytes) -> tuple[dict[str, memoryview], dict[str, Any]]:
    """The objects of a pack held in memory, by key (views into `data`), and its metadata."""
    mv = memoryview(data)
    if len(mv) < PACK_HEADER_BYTES + PACK_FOOTER_BYTES or not is_pack_header(mv):
        raise ThreadError("not a thread pack (bad header)")
    dir_off, dir_len, count = parse_pack_footer(mv[len(mv) - PACK_FOOTER_BYTES:])
    if dir_off < PACK_HEADER_BYTES or dir_off + dir_len > len(mv) - PACK_FOOTER_BYTES:
        raise ThreadError(f"pack directory [{dir_off}, +{dir_len}) is outside the pack")
    entries, meta = parse_pack_directory(mv[dir_off: dir_off + dir_len], dir_off, count)
    return {k: mv[o: o + n] for k, (o, n) in entries.items()}, meta


# ---------- reading ----------


class PackedBlobStore(BlobStore):
    """The objects of a pack under their original keys, read as ranges of one object `key` of `inner`. Read-only.
    Put a cache in front of this store (`open_index` does), never behind it: a cache behind it would fetch the whole
    pack on a miss, which is why `fill` is never forwarded."""

    supports_list = True

    def __init__(self, inner: BlobStore, key: str, tail_bytes: int = PACK_TAIL_READ) -> None:
        self.inner = inner
        self.key = key
        self.supports_ranges = inner.supports_ranges
        self.remote = inner.remote
        self.inflight = inner.inflight
        self.coalesce_bytes = inner.coalesce_bytes
        self.span_bytes = inner.span_bytes
        t0 = time.perf_counter()
        trace = BlobTrace()
        # One suffix-range request when the store can (footer + directory, no head), else a head and a range.
        try:
            got = inner.get_tail(key, tail_bytes, trace=trace)
            if got is None:
                raise ThreadError(f"no pack at {key}", 404)
            tail, h = got
        except NotImplementedError:
            hh = inner.head(key)
            trace.add(reads=1)
            if hh is None:
                raise ThreadError(f"no pack at {key}", 404) from None
            h, tail = hh, None
        if h.size < PACK_HEADER_BYTES + PACK_FOOTER_BYTES:
            raise ThreadError(f"{key} is too small to be a pack ({h.size} bytes)")
        #: Size and etag of the pack object.
        self.pack = h
        self.warmup = inner.warmup  # type: ignore[method-assign]
        tail_len = min(h.size, len(tail) if tail is not None else tail_bytes)
        if tail is None:
            tail = inner.get(key, BlobRange(h.size - tail_len, tail_len), trace=trace)
        if tail is None:
            raise ThreadError(f"no pack at {key}", 404)
        tail = memoryview(tail)
        dir_off, dir_len, count = parse_pack_footer(tail[len(tail) - PACK_FOOTER_BYTES:])
        if dir_off < PACK_HEADER_BYTES or dir_off + dir_len > h.size - PACK_FOOTER_BYTES:
            raise ThreadError(f"{key}: directory [{dir_off}, +{dir_len}) is outside the pack")
        tail_start = h.size - tail_len
        if dir_off >= tail_start:
            d: Bytes = tail[dir_off - tail_start: dir_off - tail_start + dir_len]
        else:
            got = inner.get(key, BlobRange(dir_off, dir_len), trace=trace)
            if got is None:
                raise ThreadError(f"no pack at {key}", 404)
            d = got
        #: key -> (offset in the pack, length); `partial`: key -> whole object size, for entries that hold only the
        #: object's first `length` bytes (warm packs).
        self.entries, self.meta, self.partial = parse_pack_directory_full(d, dir_off, count)
        #: What opening the pack did (`phase_stat`): the head (counted as a read of 0 bytes), the tail read and, when
        #: the directory does not fit in the tail, the directory read. `open_index` reports it as `open_stats["pack"]`.
        self.open_stats = phase_stat((time.perf_counter() - t0) * 1000, trace)

    @classmethod
    def from_bytes(cls, data: Bytes, key: str = "index.threadpack") -> "PackedBlobStore":
        """A pack held in memory (tests, small indexes)."""
        m = MemoryBlobStore()
        m.objects[key] = (bytes(data), "pack")
        return cls(m, key)

    @property
    def gen(self) -> str | None:
        """The generation the pack records (None for packs written with other metadata)."""
        g = self.meta.get("gen")
        return g if isinstance(g, str) else None

    @property
    def rev(self) -> int | None:
        r = self.meta.get("rev")
        return r if isinstance(r, int) and not isinstance(r, bool) else None

    @property
    def base_range(self) -> BlobRange | None:
        """The `base: [offset, length]` range `pack_index` records: the objects a cold open reads, contiguous."""
        b = self.meta.get("base")
        if (not isinstance(b, list) or len(b) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in b)):
            return None
        return BlobRange(b[0], b[1])

    def read_base(self, trace: BlobTrace | None = None, with_routing: bool = True) -> dict[str, memoryview] | None:
        """Reads the pack's base range with ONE range read and returns the objects inside it (views into the read),
        for a cold open (`open_index` does this for a `PackedBlobStore`). None when the pack records no range.
        `with_routing=False` stops before the routing table that ends the range (`meta["baseRouting"]`), for an open
        that loads another table."""
        r = self.base_range
        if r is None or r.length == 0:
            return None
        rk = self.meta.get("baseRouting")
        re_ = self.entries.get(rk) if isinstance(rk, str) else None
        if not with_routing and re_ is not None and re_[0] > r.offset and re_[0] + re_[1] == r.offset + r.length:
            r = BlobRange(r.offset, re_[0] - r.offset)
        b = self.inner.get(self.key, r, trace=trace)
        if b is None:
            raise ThreadError(f"pack {self.key} disappeared", 404)
        mv = memoryview(b)
        return {k: mv[o - r.offset: o - r.offset + n] for k, (o, n) in self.entries.items()
                if o >= r.offset and o + n <= r.offset + r.length}

    def _locate(self, key: str, range: BlobRange | None) -> BlobRange | None:
        e = self.entries.get(key)
        if e is None:
            return None
        off, ln = e
        size = self.partial.get(key)
        short = f"the pack holds only its first {ln} of {size} bytes" if size is not None and size > ln else None
        if range is None:
            if short:
                raise ThreadError(f"{key}: {short}")
            return BlobRange(off, ln)
        if range.offset < 0 or range.length < 0 or range.offset + range.length > ln:
            raise ThreadError(f"{key} [{range.offset}, +{range.length}): {short}" if short else
                              f"{key} [{range.offset}, +{range.length}) is outside the object ({ln} bytes)")
        return BlobRange(off + range.offset, range.length)

    def range_of(self, key: str, range: BlobRange | None = None) -> BlobRange | None:
        """Where `range` of `key` (default: all of it) lies in the pack, or None when the pack does not hold `key`."""
        return self._locate(key, range)

    def get(self, key, range=None, *, trace=None, fill=False):
        r = self._locate(key, range)
        if r is None:
            return None
        if r.length == 0:
            return b""
        b = self.inner.get(self.key, r, trace=trace)
        if b is None:
            raise ThreadError(f"pack {self.key} disappeared", 404)
        return b

    def get_ranges(self, key, ranges, *, trace=None, fill=False):
        if not self.inner.supports_ranges or key not in self.entries:
            return None
        rs = [self._locate(key, r) for r in ranges]
        return self.inner.get_ranges(self.key, rs, trace=trace)  # type: ignore[arg-type]

    def head(self, key):
        e = self.entries.get(key)
        return BlobHead(self.partial.get(key, e[1]), f"{self.pack.etag}@{e[0]}") if e else None

    def list(self, prefix):
        return sorted(k for k in self.entries if k.startswith(prefix))

    def put(self, key, data, *, if_absent=False, if_match=None, content_type=None):
        raise ThreadError("this store is read-only (a pack); copy_objects it into a directory or bucket to write", 400)


# ---------- writing ----------


class PackWriter:
    """Streams a pack to a binary file object: `add` each object (any order, keys unique), then `finish`. Entries are
    8-byte aligned; the directory lists them in the order added and carries `meta` (`pack_index` records
    `{"gen", "rev"}`)."""

    def __init__(self, out: BinaryIO, meta: dict[str, Any] | None = None) -> None:
        self.out = out
        self.meta = dict(meta or {})
        self.entries: list[list[Any]] = []
        self._seen: set[str] = set()
        self.written = 0
        self._finished = False

    def _emit(self, b: Bytes) -> None:
        mv = memoryview(b)
        while len(mv):
            n = self.out.write(mv)
            if n is None:  # a non-blocking file returned nothing
                raise ThreadError("pack: write returned no count")
            mv = mv[n:]
        self.written += len(b)

    def add(self, key: str, data: Bytes, *, partial_of: int | None = None) -> None:
        """Adds an object. With `partial_of`, `data` is only the first bytes of an object of `partial_of` bytes (a warm
        pack's slab prefix): the entry is marked `{"partial": true, "size": partial_of}`."""
        if self._finished:
            raise ThreadError("pack writer is finished", 400)
        if not key:
            raise ThreadError("pack: empty key", 400)
        if key in self._seen:
            raise ThreadError(f"pack: {key} added twice", 400)
        if partial_of is not None and (isinstance(partial_of, bool) or not isinstance(partial_of, int)
                                       or partial_of < len(data)):
            raise ThreadError(f"pack: {key} is {len(data)} bytes, more than its object (partial_of {partial_of})", 400)
        if self.written == 0:
            self._emit(pack_header())
        pad = (_ALIGN - self.written % _ALIGN) % _ALIGN
        if pad:
            self._emit(bytes(pad))
        self._seen.add(key)
        self.entries.append([key, self.written, len(data)] if partial_of is None
                            else [key, self.written, len(data), {"partial": True, "size": partial_of}])
        self._emit(data)

    def finish(self) -> tuple[int, int]:
        """Writes the directory and footer; returns (pack size, entry count)."""
        if self._finished:
            raise ThreadError("pack writer is finished", 400)
        self._finished = True
        if self.written == 0:
            self._emit(pack_header())
        directory = dumps({"entries": self.entries, **self.meta}).encode("utf-8")
        dir_off = self.written
        self._emit(directory)
        self._emit(pack_footer(dir_off, len(directory), len(self.entries)))
        return self.written, len(self.entries)


@dataclass
class PackResult:
    gen: str
    rev: int
    #: Objects in the pack, including the synthesized `idx/current.json`.
    count: int
    bytes: int
    keys: list[str] = field(default_factory=list)


def _base_meta(w: PackWriter, state: RevState, last: str) -> None:
    """`base: [offset, length]` (current.json to the end of the base objects) and `baseRouting`, the routing table
    that ends the range (an open with another routing table stops before it)."""
    w.meta["base"] = [PACK_HEADER_BYTES, w.written - PACK_HEADER_BYTES]
    if has_dense(state.manifest) and last == choose_routing(state.manifest, state.files, "auto")[1]:
        w.meta["baseRouting"] = last


def _pack_into(store: BlobStore, out: BinaryIO, gen: str | None, rev: int | None, workers: int) -> PackResult:
    g, r = resolve_target(store, gen, rev)
    state = load_rev_state(store, g, r)
    snap = snapshot_keys(state)
    in_snap = set(snap)
    base = [k for k in base_keys(state, "base", "auto") if k in in_snap]
    base_set = set(base)
    ks = base + [k for k in snap if k not in base_set]
    w = PackWriter(out, {"gen": g, "rev": r})
    w.add("idx/current.json", dumps({"gen": g, "rev": r}).encode("utf-8"))
    trace = BlobTrace()

    def fetch(k: str) -> Bytes:
        b = store.get(k, None, trace=trace)
        if b is None:
            raise ThreadError(f"snapshot of {g} rev {r} needs {k}, which is missing", 404)
        return b

    if workers > 1 and store.remote:
        # Read ahead `workers` objects, write them in order.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="thread-pack") as pool:
            futs = [pool.submit(fetch, k) for k in ks[:workers]]
            for i, k in enumerate(ks):
                if i + workers < len(ks):
                    futs.append(pool.submit(fetch, ks[i + workers]))
                w.add(k, futs[i].result())
                futs[i] = None  # type: ignore[call-overload]
                if i == len(base) - 1:
                    _base_meta(w, state, k)
    else:
        for i, k in enumerate(ks):
            w.add(k, fetch(k))
            if i == len(base) - 1:
                _base_meta(w, state, k)
    size, count = w.finish()
    return PackResult(g, r, count, size, ["idx/current.json", *ks])


def pack_index(store: BlobStore, dest: str | os.PathLike[str] | BinaryIO, *, gen: str | None = None,
               rev: int | None = None, workers: int = 8) -> PackResult:
    """Writes a snapshot of one revision to `dest` (a path, written to a temp sibling and renamed into place, or an
    open binary file): `idx/current.json` pointing at it, then the objects a cold open reads (`base_keys` with the
    smallest routing table), contiguous, then every other object the revision reads (`snapshot_keys`) in key order;
    the directory records `{gen, rev, base: [offset, length], baseRouting}` (`PackedBlobStore.read_base`). `gen` / `rev` pick
    the revision as `open_index` does (default: what `idx/current.json` names).
    Two SDKs packing the same revision produce the same bytes."""
    if hasattr(dest, "write"):
        return _pack_into(store, dest, gen, rev, workers)  # type: ignore[arg-type]
    path = os.path.abspath(os.fspath(dest))  # type: ignore[arg-type]
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".tmp-{os.path.basename(path)}-")
    try:
        with os.fdopen(fd, "wb") as f:
            result = _pack_into(store, f, gen, rev, workers)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
        tmp = ""
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
    return result


def pack_to_bytes(store: BlobStore, *, gen: str | None = None, rev: int | None = None) -> tuple[bytes, PackResult]:
    """`pack_index` into memory."""
    buf = io.BytesIO()
    r = _pack_into(store, buf, gen, rev, 1)
    return buf.getvalue(), r


def copy_objects(src: BlobStore, dst: BlobStore, keys: Iterable[str] | None = None, *, skip_existing: bool = False,
                 workers: int = 8) -> tuple[int, int, int]:
    """Copies objects from one store to another: a pack into a directory (to write on top of it), a bucket prefix
    to a laptop, a laptop index to a bucket. `keys` defaults to everything `src` lists; pass `snapshot_keys(rev)`
    plus `idx/current.json` to copy one consistent revision out of a live index. With `skip_existing`, keys already
    in `dst` with the same size are left alone. Returns (copied, bytes, skipped)."""
    if not dst.supports_put:
        raise ThreadError("the destination store is read-only (no put)", 400)
    if keys is None:
        if not src.supports_list:
            raise ThreadError("the source store cannot list objects: pass the keys to copy", 400)
        keys = src.list("")
    ks: Sequence[str] = list(keys)

    def one(k: str) -> tuple[int, int]:
        if skip_existing:
            sh, dh = src.head(k), dst.head(k)
            if sh is not None and dh is not None and sh.size == dh.size:
                return 0, 0
        b = src.get(k)
        if b is None:
            raise ThreadError(f"missing object {k}", 404)
        if dst.put(k, b) is None:
            raise ThreadError(f"put of {k} was refused by the destination", 409)
        return 1, len(b)

    results: Iterable[tuple[int, int]]
    if workers > 1 and (src.remote or dst.remote) and len(ks) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(ks)), thread_name_prefix="thread-copy") as pool:
            results = list(pool.map(one, ks))
    else:
        results = [one(k) for k in ks]
    copied = sum(c for c, _ in results)
    return copied, sum(n for _, n in results), len(ks) - copied


def store_at(path: str | os.PathLike[str]) -> BlobStore:
    """The store at a path: a directory holding an index layout (`FileBlobStore`), or a pack file (`PackedBlobStore`
    over the file's directory). `open_index` accepts a path and calls this."""
    p = os.path.abspath(os.fspath(path))
    if os.path.isdir(p):
        return FileBlobStore(p)
    if not os.path.isfile(p):
        raise ThreadError(f"no such file or directory: {p}", 404)
    with open(p, "rb") as f:
        head = f.read(PACK_HEADER_BYTES)
    if not is_pack_header(head):
        raise ThreadError(f"{p} is not a thread pack", 400)
    return PackedBlobStore(FileBlobStore(os.path.dirname(p)), os.path.basename(p))


# ---------- base packs ----------


@dataclass
class BasePackResult:
    key: str
    data: bytes
    #: The objects in it, in order.
    keys: list[str]
    #: The routing table it holds (kind "base").
    routing: str | None


def build_base_pack(store: BlobStore, rev: RevState, kind: str = "base", routing: str = "auto") -> BasePackResult:
    """The base pack of a revision (`idx/{gen}/base.{rev}.threadpack`): every object a cold open reads
    (`base_keys`), so a reader that knows (gen, rev) from `idx/current.json` loads the revision and its routing table
    with ONE GET; `kind="cold"` (`base.cold.{rev}.threadpack`) leaves the routing table out, for
    `cold_routing="tree"`. The directory records `{gen, rev, kind: "base" | "base-cold", routing}`; the TypeScript
    SDK's `buildBasePack` writes the same bytes."""
    ks = base_keys(rev, kind, routing)
    elem = choose_routing(rev.manifest, rev.files, routing)[0] if kind == "base" and has_dense(rev.manifest) else None
    buf = io.BytesIO()
    w = PackWriter(buf, {"gen": rev.gen, "rev": rev.rev, "kind": "base-cold" if kind == "cold" else "base",
                         "routing": elem})
    for k in ks:
        b = store.get(k)
        if b is None:
            raise ThreadError(f"base pack of {rev.gen} rev {rev.rev} needs {k}, which is missing", 404)
        w.add(k, b)
    w.finish()
    key = index_keys.base_cold_pack(rev.gen, rev.rev) if kind == "cold" else index_keys.base_pack(rev.gen, rev.rev)
    return BasePackResult(key, buf.getvalue(), ks, elem)


class PrefetchStore(BlobStore):
    """A read-through view of a store with some objects already in memory (a base pack's, or reads started early):
    reads of such a key are served from memory (byte-identical: the same object), everything else goes to the inner
    store. A read started early (`prefetch`) keeps its own trace, which the first `get` of the key adds to the
    caller's, so per-step accounting (`open_stats`) still sees the read where its bytes are used. Used by
    `open_index` for the reads of the open only; read-only."""

    def __init__(self, inner: BlobStore) -> None:
        self.inner = inner
        self.supports_list = inner.supports_list
        self.supports_ranges = True
        self.remote = inner.remote
        # key -> [bytes-like | Future, BlobTrace | None (not yet credited)]
        self._entries: dict[str, list[Any]] = {}
        self._lock = threading.Lock()

    def set(self, key: str, data: Any) -> None:
        """Serves `key` from `data` (accounted elsewhere, e.g. as part of a pack)."""
        with self._lock:
            self._entries[key] = [data, None]

    def prefetch(self, key: str, io: ThreadPoolExecutor) -> None:
        """Starts reading `key` now on `io` (once); later reads of it wait for this one."""
        with self._lock:
            if key in self._entries:
                return
            trace = BlobTrace()
            self._entries[key] = [io.submit(self.inner.get, key, None, trace=trace), trace]

    def add_pack(self, data: Bytes) -> dict[str, Any]:
        """Serves every object of a pack held in memory; returns its metadata."""
        entries, meta = pack_entries(data)
        with self._lock:
            for k, v in entries.items():
                self._entries[k] = [v, None]
        return meta

    def has(self, key: str) -> bool:
        return key in self._entries

    def peek(self, key: str) -> Any:
        """The bytes of a prefetched key without crediting its read (None when absent)."""
        with self._lock:
            e = self._entries.get(key)
        if e is None:
            return None
        v = e[0]
        return v.result() if isinstance(v, Future) else v

    def _take(self, key: str, trace: BlobTrace | None) -> tuple[bool, Any]:
        with self._lock:
            e = self._entries.get(key)
        if e is None:
            return False, None
        v = e[0].result() if isinstance(e[0], Future) else e[0]
        with self._lock:
            credit, e[1] = e[1], None
        if credit is not None and trace is not None:
            trace.add(**credit.as_dict())
        return True, v

    def get(self, key, range=None, *, trace=None, fill=False):
        found, b = self._take(key, trace)
        if not found:
            return self.inner.get(key, range, trace=trace, fill=fill)
        if b is None or range is None:
            return b
        if range.offset < 0 or range.length < 0 or range.offset + range.length > len(b):
            raise ThreadError(f"{key} [{range.offset}, +{range.length}) is outside the object ({len(b)} bytes)")
        return memoryview(b)[range.offset: range.offset + range.length]

    def get_ranges(self, key, ranges, *, trace=None, fill=False):
        if key not in self._entries:
            return self.inner.get_ranges(key, ranges, trace=trace, fill=fill)
        return [self.get(key, r, trace=trace) for r in ranges]

    def head(self, key):
        if key not in self._entries:
            return self.inner.head(key)
        b = self.peek(key)
        return BlobHead(len(b), "prefetched") if b is not None else None

    def list(self, prefix):
        return self.inner.list(prefix)
