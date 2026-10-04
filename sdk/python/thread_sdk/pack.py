"""Packs: one immutable file holding a whole index snapshot, so an index can be moved as a single object and queried
anywhere a `BlobStore` can range-read it (a directory, R2, S3, memory). `pack_index` writes the objects one revision
reads (`snapshot_keys`) plus an `idx/current.json` naming it; `PackedBlobStore` serves them back under their original
keys by translating reads into ranges of the pack. A pack is read-only: to write on top, `copy_objects` it into a
directory or a bucket.

Layout (all integers little-endian; docs/portability.md has the specification):

    [0, 16)      header: magic "THRDPACK", version u16 (1), flags u16 (0), reserved u32 (0)
    [16, D)      the objects, each at an 8-byte-aligned offset, zero padding between
    [D, D + L)   directory: UTF-8 JSON {"entries":[[key, offset, length], ...], ...meta}
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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Iterable, Sequence

from .revision import load_rev_state, resolve_target, snapshot_keys
from .store import BlobHead, BlobRange, BlobStore, BlobTrace, Bytes, FileBlobStore, MemoryBlobStore, phase_stat
from .wire import ThreadError, decode_json, dumps

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
]

PACK_MAGIC = b"THRDPACK"
PACK_VERSION = 1
PACK_HEADER_BYTES = 16
PACK_FOOTER_BYTES = 32
#: Bytes a reader fetches from the end of a pack to find the footer (the directory often fits in them).
PACK_TAIL_READ = 64 * 1024
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
    j = decode_json(b)
    if not isinstance(j, dict) or not isinstance(j.get("entries"), list):
        raise ThreadError("pack directory has no entries")
    if len(j["entries"]) != count:
        raise ThreadError(f"pack directory lists {len(j['entries'])} entries, footer says {count}")
    entries: dict[str, tuple[int, int]] = {}
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
    meta = {k: v for k, v in j.items() if k != "entries"}
    return entries, meta


# ---------- reading ----------


class PackedBlobStore(BlobStore):
    """The objects of a pack under their original keys, read as ranges of one object `key` of `inner`. Read-only.
    Put a cache in front of this store (`open_index` does), never behind it: a cache behind it would fetch the whole
    pack on a miss, which is why `fill` is never forwarded."""

    supports_list = True

    def __init__(self, inner: BlobStore, key: str) -> None:
        self.inner = inner
        self.key = key
        self.supports_ranges = inner.supports_ranges
        self.remote = inner.remote
        self.inflight = inner.inflight
        self.coalesce_bytes = inner.coalesce_bytes
        self.span_bytes = inner.span_bytes
        t0 = time.perf_counter()
        trace = BlobTrace()
        h = inner.head(key)
        trace.add(reads=1)
        if h is None:
            raise ThreadError(f"no pack at {key}", 404)
        if h.size < PACK_HEADER_BYTES + PACK_FOOTER_BYTES:
            raise ThreadError(f"{key} is too small to be a pack ({h.size} bytes)")
        #: Size and etag of the pack object.
        self.pack = h
        self.warmup = inner.warmup  # type: ignore[method-assign]
        tail_len = min(h.size, PACK_TAIL_READ)
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
        #: key -> (offset in the pack, length).
        self.entries, self.meta = parse_pack_directory(d, dir_off, count)
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

    def _locate(self, key: str, range: BlobRange | None) -> BlobRange | None:
        e = self.entries.get(key)
        if e is None:
            return None
        off, ln = e
        if range is None:
            return BlobRange(off, ln)
        if range.offset < 0 or range.length < 0 or range.offset + range.length > ln:
            raise ThreadError(f"{key} [{range.offset}, +{range.length}) is outside the object ({ln} bytes)")
        return BlobRange(off + range.offset, range.length)

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
        return BlobHead(e[1], f"{self.pack.etag}@{e[0]}") if e else None

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

    def add(self, key: str, data: Bytes) -> None:
        if self._finished:
            raise ThreadError("pack writer is finished", 400)
        if not key:
            raise ThreadError("pack: empty key", 400)
        if key in self._seen:
            raise ThreadError(f"pack: {key} added twice", 400)
        if self.written == 0:
            self._emit(pack_header())
        pad = (_ALIGN - self.written % _ALIGN) % _ALIGN
        if pad:
            self._emit(bytes(pad))
        self._seen.add(key)
        self.entries.append([key, self.written, len(data)])
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


def _pack_into(store: BlobStore, out: BinaryIO, gen: str | None, rev: int | None, workers: int) -> PackResult:
    g, r = resolve_target(store, gen, rev)
    state = load_rev_state(store, g, r)
    ks = snapshot_keys(state)
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
    else:
        for k in ks:
            w.add(k, fetch(k))
    size, count = w.finish()
    return PackResult(g, r, count, size, ["idx/current.json", *ks])


def pack_index(store: BlobStore, dest: str | os.PathLike[str] | BinaryIO, *, gen: str | None = None,
               rev: int | None = None, workers: int = 8) -> PackResult:
    """Writes a snapshot of one revision to `dest` (a path, written to a temp sibling and renamed into place, or an
    open binary file): `idx/current.json` pointing at it, then every object the revision reads (`snapshot_keys`),
    in key order. `gen` / `rev` pick the revision as `open_index` does (default: what `idx/current.json` names).
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
