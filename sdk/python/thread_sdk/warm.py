"""Warm packs: the scan prefixes (bytes `[0, scan_len)`) of an index's most-probed slabs in ONE object, fetched with one
GET when an index opens (`open_index(..., warm_pack=key)`, in the background) and used to seed an in-process layer, so
the first queries of a cold process read those prefixes from memory instead of one range GET each. Same format and
names as the TypeScript SDK's `warm.ts`.

Format: a pack (pack.py) whose directory records `{"kind": "warm", "gen", "rev", ...}` and whose entries are the slabs'
own keys (`idx/{gen}/slab/{cell}.{slab_rev}`, so a prefix is only ever used for the immutable object it was cut
from), each marked `{"partial": true, "size": <whole slab bytes>}`. A seeded prefix serves ranges inside it; anything
else (the rerank records phase 2 reads, a whole-object read) goes to the store as before.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable, Protocol, Sequence

from .pack import PACK_FOOTER_BYTES, PACK_HEADER_BYTES, PackWriter, is_pack_header, parse_pack_directory_full, \
    parse_pack_footer
from .query import RevState
from .store import BlobRange, BlobStore, BlobTrace, Bytes, CachedBlobStore, phase_stat
from .wire import ThreadError, keys, slab_rev_of

__all__ = ["WARM_PACK_KIND", "WarmEntry", "parse_warm_pack", "write_warm_pack", "build_warm_pack", "WarmPackResult",
           "PrefixSeed", "WarmPrefixStore", "seed_warm_pack", "revision_slab_keys"]

WARM_PACK_KIND = "warm"
log = logging.getLogger("thread_sdk")


@dataclass
class WarmEntry:
    """The first `len(data)` bytes of the object `key` (`size` bytes in all)."""

    key: str
    data: memoryview
    size: int


def parse_warm_pack(data: Bytes) -> tuple[dict[str, Any], list[WarmEntry]]:
    """A warm pack held in memory: its metadata and its prefixes (views into `data`)."""
    mv = memoryview(data)
    if len(mv) < PACK_HEADER_BYTES + PACK_FOOTER_BYTES or not is_pack_header(mv):
        raise ThreadError("not a thread pack (bad header)")
    dir_off, dir_len, count = parse_pack_footer(mv[len(mv) - PACK_FOOTER_BYTES:])
    if dir_off < PACK_HEADER_BYTES or dir_off + dir_len > len(mv) - PACK_FOOTER_BYTES:
        raise ThreadError(f"pack directory [{dir_off}, +{dir_len}) is outside the pack")
    entries, meta, partial = parse_pack_directory_full(mv[dir_off:dir_off + dir_len], dir_off, count)
    if meta.get("kind") != WARM_PACK_KIND:
        raise ThreadError(f"not a warm pack (kind {meta.get('kind')!r})")
    return meta, [WarmEntry(k, mv[off:off + ln], partial.get(k, ln)) for k, (off, ln) in entries.items()]


@dataclass
class WarmPackResult:
    gen: str
    rev: int
    #: The slab keys in the pack, in the order given (hottest first, by convention).
    keys: list[str]
    #: Pack size and the prefix bytes in it.
    bytes: int
    prefix_bytes: int


def write_warm_pack(store: BlobStore, rev: RevState, cells: Sequence[int], out: BinaryIO, *,
                    meta: dict[str, Any] | None = None) -> WarmPackResult:
    """Writes the warm pack of `cells` (format 2/3 slabs: their scan prefixes at the revision's slab revisions) to
    `out`, entries in the order given. Reads only the prefixes from `store`. The TypeScript SDK's `writeWarmPack`
    writes the same bytes."""
    if rev.slabs is None:
        raise ThreadError(f"{rev.gen} rev {rev.rev} has no slabs table (format 1 slabs are read whole: no prefixes to "
                          "warm)", 400)
    items = []
    for cell in cells:
        row = rev.slabs[cell] if 0 <= cell < len(rev.slabs) else None
        if not row:
            raise ThreadError(f"{rev.files['slabs']} has no entry for cell {cell}", 400)
        items.append((keys.slab(rev.gen, cell, slab_rev_of(row)), row[0], row[3]))
    if len({k for k, _, _ in items}) != len(items):
        raise ThreadError("warm pack: a cell is listed twice", 400)
    w = PackWriter(out, {**(meta or {}), "kind": WARM_PACK_KIND, "gen": rev.gen, "rev": rev.rev, "cells": len(items)})
    prefix_bytes = 0
    for key, scan_len, size in items:
        b = store.get(key, BlobRange(0, scan_len))
        if b is None:
            raise ThreadError(f"warm pack needs {key}, which is missing", 404)
        prefix_bytes += len(b)
        w.add(key, b, partial_of=size)
    total, _ = w.finish()
    return WarmPackResult(rev.gen, rev.rev, [k for k, _, _ in items], total, prefix_bytes)


def build_warm_pack(store: BlobStore, rev: RevState, cells: Sequence[int], *, meta: dict[str, Any] | None = None
                    ) -> tuple[bytes, WarmPackResult]:
    """`write_warm_pack` into memory."""
    buf = io.BytesIO()
    r = write_warm_pack(store, rev, cells, buf, meta=meta)
    return buf.getvalue(), r


# ---------- seeding ----------


class PrefixSeed(Protocol):
    """Where a warm pack's prefixes go: an in-process layer that serves ranges inside an object's first bytes and sends
    everything else to the store. `WarmPrefixStore` is the one `open_index` uses; an in-process cache can implement
    this to take the prefixes instead (keeping them prefix-only: a read past the prefix goes to the store)."""

    def seed_prefix(self, key: str, data: Bytes) -> None: ...


class WarmPrefixStore(BlobStore):
    """A store in front of another one that holds object PREFIXES in memory: a read of a range inside a held prefix is
    served from memory (a cache hit in the trace), every other read goes to the inner store unchanged.

    When the inner store is a cache (`CachedBlobStore`, `open_index`'s default for a remote store) and a served read
    asked for `fill` (a slab's scan prefix, which on a cache miss would have fetched the whole slab into the cache),
    the whole object is fetched into the cache in the background instead (`fill_workers` at once, once per key), so
    later queries find the slab's rerank records there: the warm pack takes the first touch off the critical path
    without keeping the cache cold."""

    def __init__(self, inner: BlobStore, *, fill_workers: int = 2) -> None:
        self.inner = inner
        self._fill_workers = fill_workers
        self._fill_pool: ThreadPoolExecutor | None = None
        self._filling: set[str] = set()
        #: Background whole-object fills behind a cache: queued, done, failed.
        self.fills = {"queued": 0, "done": 0, "failed": 0}
        self.supports_put = inner.supports_put
        self.supports_list = inner.supports_list
        self.supports_ranges = inner.supports_ranges
        self.remote = inner.remote
        self._prefixes: dict[str, bytes] = {}
        self._lock = threading.Lock()
        #: Reads served from a prefix, and their bytes.
        self.hits = 0
        self.hit_bytes = 0

    def seed_prefix(self, key: str, data: Bytes) -> None:
        # Own bytes per prefix: a view into the pack would let a store-level "is this the whole slab?" check see the
        # pack's buffer.
        b = bytes(data)
        with self._lock:
            cur = self._prefixes.get(key)
            if cur is None or len(cur) < len(b):
                self._prefixes[key] = b

    def retain(self, keep: Callable[[str], bool]) -> None:
        """Keeps only the prefixes `keep` accepts (after a switch to a revision with other slab revisions)."""
        with self._lock:
            for k in [k for k in self._prefixes if not keep(k)]:
                del self._prefixes[k]

    def _held(self, key: str, r: BlobRange) -> bytes | None:
        p = self._prefixes.get(key)
        return p if p is not None and r.offset >= 0 and r.length >= 0 and r.offset + r.length <= len(p) else None

    def holds(self, key: str, r: BlobRange) -> bool:
        return self._held(key, r) is not None

    @property
    def size(self) -> tuple[int, int]:
        """(prefixes held, their bytes)."""
        with self._lock:
            return len(self._prefixes), sum(len(b) for b in self._prefixes.values())

    def _count(self, n: int, trace: BlobTrace | None) -> None:
        with self._lock:
            self.hits += 1
            self.hit_bytes += n
        if trace is not None:
            trace.add(hits=1, bytes=n)

    def _fill_behind(self, key: str) -> None:
        if not isinstance(self.inner, CachedBlobStore) or self._fill_workers <= 0:
            return
        with self._lock:
            if key in self._filling:
                return
            self._filling.add(key)
            self.fills["queued"] += 1
            if self._fill_pool is None:
                self._fill_pool = ThreadPoolExecutor(max_workers=self._fill_workers,
                                                     thread_name_prefix="thread-warm-fill")
            pool = self._fill_pool

        def run() -> None:
            try:
                self.inner.get(key)  # a whole-object read through the cache: a miss fills it
                with self._lock:
                    self.fills["done"] += 1
            except Exception as e:  # noqa: BLE001 - a later hit may retry
                with self._lock:
                    self.fills["failed"] += 1
                    self._filling.discard(key)
                log.error("thread_sdk: background fill of %s failed: %s", key, e)

        try:
            pool.submit(run)
        except RuntimeError:  # closed
            pass

    def drain(self) -> None:
        """Waits for the queued background fills (tests, benchmarks)."""
        with self._lock:
            pool = self._fill_pool
            self._fill_pool = None
        if pool is not None:
            pool.shutdown(wait=True)

    def close(self) -> None:
        """Drops queued background fills (running ones finish)."""
        with self._lock:
            pool = self._fill_pool
            self._fill_pool = None
            self._fill_workers = 0
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def get(self, key, range=None, *, trace=None, fill=False):
        if range is not None:
            p = self._held(key, range)
            if p is not None:
                self._count(range.length, trace)
                if fill:
                    self._fill_behind(key)
                return memoryview(p)[range.offset:range.offset + range.length]
        return self.inner.get(key, range, trace=trace, fill=fill)

    def get_ranges(self, key, ranges, *, trace=None, fill=False):
        held = [self._held(key, r) for r in ranges]
        if ranges and all(p is not None for p in held):
            self._count(sum(r.length for r in ranges), trace)
            return [memoryview(p)[r.offset:r.offset + r.length] for p, r in zip(held, ranges)]  # type: ignore[index]
        return self.inner.get_ranges(key, ranges, trace=trace, fill=fill)

    def head(self, key):
        return self.inner.head(key)

    def put(self, key, data, *, if_absent=False, if_match=None, content_type=None):
        return self.inner.put(key, data, if_absent=if_absent, if_match=if_match, content_type=content_type)

    def list(self, prefix):
        return self.inner.list(prefix)


def seed_warm_pack(data: Bytes, target: PrefixSeed, accept: Callable[[str], bool] = lambda k: True) -> dict[str, Any]:
    """Seeds `target` with the prefixes of a warm pack that `accept` takes (default: all). Returns
    `{entries, seeded, skipped, seeded_bytes, meta}`."""
    meta, entries = parse_warm_pack(data)
    seeded = seeded_bytes = 0
    for e in entries:
        if not accept(e.key):
            continue
        target.seed_prefix(e.key, e.data)
        seeded += 1
        seeded_bytes += len(e.data)
    return {"entries": len(entries), "seeded": seeded, "skipped": len(entries) - seeded, "seeded_bytes": seeded_bytes,
            "meta": meta}


def revision_slab_keys(rev: RevState) -> set[str]:
    """The slab keys a revision reads (what a warm pack's entries must match to be used)."""
    if rev.slabs is None:
        return set()
    return {keys.slab(rev.gen, cell, slab_rev_of(row)) for cell, row in enumerate(rev.slabs)}


@dataclass
class WarmStats:
    """What loading a warm pack did (`index.warm_stats`, a dict view via `as_dict`)."""

    key: str
    state: str = "loading"
    fetch: dict[str, Any] | None = None
    entries: int = 0
    seeded: int = 0
    skipped: int = 0
    seeded_bytes: int = 0
    ready_ms: float | None = None
    error: str | None = None
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "state": self.state, "fetch": self.fetch, "entries": self.entries,
                "seeded": self.seeded, "skipped": self.skipped, "seeded_bytes": self.seeded_bytes,
                "ready_ms": self.ready_ms, **({"error": self.error} if self.error else {})}


def fetch_warm_pack(source: BlobStore, key: str, t0: float) -> tuple[Bytes, dict[str, Any]]:
    """The ONE GET of a warm pack; returns its bytes and the read's `phase_stat` (ms from `t0`)."""
    trace = BlobTrace()
    b = source.get(key, None, trace=trace)
    if b is None:
        raise ThreadError(f"no warm pack at {key}", 404)
    return b, phase_stat((time.perf_counter() - t0) * 1000, trace)
