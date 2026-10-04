"""The lexical side of an index: its segments (`manifest["segments"]`, each a `THREADIX` container object with a
BM25 index over a batch of documents and/or their `docs` entries) read on demand through the native
`SegmentReader`.

Every native call either succeeds or raises `MissingRanges` naming every blob-absolute range its current round still
needs; `LexSegment.run` fetches them from the store (one multi-range request where the store can, else one range per
request), feeds them and calls again. The resident parts (container table, `meta`, section headers, the terms and
docs directories) are fetched once per segment; after that a search is two rounds per segment (the query terms'
buckets, then their posting lists), run for every segment concurrently, and a docs lookup one or two (buckets, then
texts). Scores use corpus-wide statistics (N, average length, per-term df summed over the segments), so the
segments together score exactly like one index over all documents.
"""

from __future__ import annotations

import functools
import json
import threading
import time
from array import array
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar

from . import _native
from ._native import MissingRanges
from .store import BlobRange, BlobStore, BlobTrace
from .wire import ResolvedLexConfig, TombState, ThreadError, parse_hits, tomb_for_segment, tomb_hides

__all__ = ["LexIo", "LexSegment", "LexIndex", "LexHit", "LexSearchResult", "DocRecord", "coalesce", "PLAN_GAP"]

T = TypeVar("T")

#: Ranges closer than this are fetched as one (the core's `PLAN_GAP`).
PLAN_GAP = 4096


def coalesce(ranges: Iterable[tuple[int, int]], max_gap: int = PLAN_GAP) -> list[tuple[int, int]]:
    """Sorts and merges `(offset, len)` ranges whose gap is at most `max_gap` bytes."""
    out: list[list[int]] = []
    for off, n in sorted((o, n) for o, n in ranges if n > 0):
        if out and off <= out[-1][0] + out[-1][1] + max_gap:
            out[-1][1] = max(out[-1][0] + out[-1][1], off + n) - out[-1][0]
        else:
            out.append([off, n])
    return [(o, n) for o, n in out]


@dataclass
class LexIo:
    """What a lexical call cost (fetch rounds, ranges and bytes over every segment)."""

    rounds: int = 0
    ranges: int = 0
    bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, rounds: int, ranges: int, nbytes: int) -> None:
        with self._lock:
            self.rounds += rounds
            self.ranges += ranges
            self.bytes += nbytes

    def as_dict(self) -> dict[str, int]:
        return {"rounds": self.rounds, "ranges": self.ranges, "bytes": self.bytes}


class LexSegment:
    """One segment object under demand paging. `open` fetches the resident parts; `run` executes a native call
    through the `MissingRanges` loop. One per segment key, shared by every revision that lists it and safe to use
    from several threads."""

    def __init__(self, store: BlobStore, ref: dict[str, Any], size: int, *, transient_budget: int = 8 * 1024 * 1024,
                 max_rounds: int = 8, fill_bytes: int = 2 * 1024 * 1024) -> None:
        self.store = store
        self.ref = ref
        self.key: str = ref["key"]
        self.size = size
        self.reader = _native.SegmentReader(size)
        self.info: dict[str, Any] | None = None
        self.has_lex = False
        self.has_docs = False
        #: The tokenizer config recorded in the segment's `meta` (JSON) or None.
        self.tokenizer: str | None = None
        self.transient_budget = transient_budget
        self.max_rounds = max_rounds
        self.fill_bytes = fill_bytes
        self._transient: list[tuple[int, int]] = []
        self._transient_bytes = 0
        self._lock = threading.Lock()

    @classmethod
    def open(cls, store: BlobStore, ref: dict[str, Any], io: LexIo | None = None, trace: BlobTrace | None = None,
             **opts: Any) -> "LexSegment":
        """Opens the segment: container table + meta, section headers, directories (three fetches cold, once)."""
        io = io if io is not None else LexIo()
        size = ref.get("bytes") or 0
        if not size:
            h = store.head(ref["key"])
            if h is None:
                raise ThreadError(f"missing segment {ref['key']}", 404)
            size = h.size
        seg = cls(store, ref, size, **opts)
        seg._fetch(seg.reader.plan_open(), io, trace, True)
        o = seg.run(seg.reader.open, io, trace, True)
        seg.has_lex = o["has_lex"]
        seg.has_docs = o["has_docs"]
        seg.tokenizer = o["tokenizer"]
        seg._fetch(seg.run(seg.reader.plan_headers, io, trace, True), io, trace, True)
        seg.info = seg.run(seg.reader.info, io, trace, True)
        seg._fetch(seg.run(seg.reader.plan_directories, io, trace, True), io, trace, True)
        return seg

    @property
    def num_docs(self) -> int:
        return self.info["num_docs"] if self.info else 0

    @property
    def total_len(self) -> int:
        return self.info["total_len"] if self.info else 0

    def _fetch(self, ranges: list[tuple[int, int]], io: LexIo, trace: BlobTrace | None, resident: bool) -> None:
        """Fetches `ranges` (coalesced) and feeds them; resident ranges are never evicted."""
        wanted = [(o, min(n, self.size - o)) for o, n in coalesce(ranges) if o < self.size]
        missing = self.reader.missing(wanted)
        if not missing:
            return
        blob_ranges = [BlobRange(o, n) for o, n in missing]
        fill = self.size <= self.fill_bytes
        parts = self.store.get_ranges(self.key, blob_ranges, trace=trace, fill=fill) if len(blob_ranges) > 1 else None
        if parts is None:
            parts = []
            for r in blob_ranges:
                b = self.store.get(self.key, r, trace=trace, fill=fill)
                if b is None:
                    raise ThreadError(f"missing segment {self.key}", 404)
                parts.append(b)
        for (off, _), b in zip(missing, parts):
            self.reader.feed(off, b)
        if not resident:
            with self._lock:
                self._transient.extend(missing)
                self._transient_bytes += sum(n for _, n in missing)
        io.add(1, len(missing), sum(len(b) for b in parts))

    def run(self, fn: Callable[[], T], io: LexIo, trace: BlobTrace | None = None, resident: bool = False) -> T:
        """Calls `fn` until it stops raising `MissingRanges`, fetching what it asks for."""
        round = 0
        while True:
            try:
                v = fn()
            except MissingRanges as e:
                if round >= self.max_rounds:
                    raise ThreadError(f"{self.key}: still missing {len(e.ranges)} range(s) after {self.max_rounds} rounds") from e
                self._fetch(e.ranges, io, trace, resident)
                round += 1
                continue
            self._trim()
            return v

    def _trim(self) -> None:
        """Evicts the transient bytes once they exceed the budget (the resident parts stay). A coalesced transient
        fetch may have overlapped a resident range; the next call simply re-fetches that sliver."""
        with self._lock:
            if self._transient_bytes <= self.transient_budget or not self._transient:
                return
            ranges, self._transient, self._transient_bytes = self._transient, [], 0
        self.reader.evict_ranges(ranges)

    @property
    def held_bytes(self) -> int:
        """Bytes held in native memory for this segment."""
        return self.reader.held_bytes


@dataclass
class LexHit:
    id: int
    #: BM25 score (float32), corpus-wide statistics.
    score: float


@dataclass
class LexSearchResult:
    hits: list[LexHit]
    #: The query's tokens (after the index's tokenizer).
    tokens: list[str]
    #: Distinct tokens found in at least one segment.
    found: int
    #: `{segments, num_docs, total_len, df: {token: corpus df}, io: {rounds, ranges, bytes}, ms}`.
    stats: dict[str, Any]


@dataclass
class DocRecord:
    """A resolved document: its metadata string and, when stored and requested, its text."""

    id: int
    meta: str
    text: str | None = None


class LexIndex:
    """The lexical index of one revision: its segments (oldest first), tokenizer and tombstones."""

    def __init__(self, config: ResolvedLexConfig, segments: list[LexSegment], tomb: TombState | None,
                 io_pool: Executor | None = None) -> None:
        self.config = config
        self.segments = segments
        self.tomb = tomb
        self.io_pool = io_pool
        self.tokenizer = _native.LexTokenizer(json.dumps(config.tokenizer or {}, separators=(",", ":")))

    @property
    def has_lex(self) -> bool:
        return any(s.has_lex for s in self.segments)

    @property
    def has_docs(self) -> bool:
        return any(s.has_docs for s in self.segments)

    def stats(self) -> tuple[int, int]:
        """Corpus-wide `(num_docs, total_len)` over the lexical segments."""
        lex = [s for s in self.segments if s.has_lex]
        return sum(s.num_docs for s in lex), sum(s.total_len for s in lex)

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)

    def _each(self, segs: list[LexSegment], fn: Callable[[LexSegment], T]) -> list[T]:
        """`fn` over the segments, concurrently on the I/O pool when there are several (each call is a leaf: it
        fetches and feeds, never waits for another task)."""
        if len(segs) <= 1 or self.io_pool is None:
            return [fn(s) for s in segs]
        futs = [self.io_pool.submit(fn, s) for s in segs]
        return [f.result() for f in futs]

    def search(self, text: str, k: int, trace: BlobTrace | None = None) -> LexSearchResult:
        """BM25 top `k` over every segment: tokenize, look the distinct tokens up in every lexical segment (one round
        each, concurrently), sum their df, score every segment with the corpus-wide statistics (one round each),
        merge. Tombstoned ids never take a slot."""
        t0 = time.perf_counter()
        io = LexIo()
        tokens = self.tokenize(text)
        weights: dict[str, int] = {}
        for t in tokens:
            weights[t] = weights.get(t, 0) + 1
        distinct = list(weights)
        lex_segs = [s for s in self.segments if s.has_lex]
        num_docs, total_len = self.stats()

        def result(hits: list[LexHit], found: int, df: dict[str, float]) -> LexSearchResult:
            return LexSearchResult(hits, tokens, found, {"segments": len(lex_segs), "num_docs": num_docs,
                                                         "total_len": total_len, "df": df, "io": io.as_dict(),
                                                         "ms": round((time.perf_counter() - t0) * 1000, 3)})

        if not distinct or not lex_segs or k < 1:
            return result([], 0, {})
        infos = self._each(lex_segs, lambda s: s.run(lambda: s.reader.lookup(distinct), io, trace))
        dfs = [0.0] * len(distinct)
        for inf in infos:
            for i, (df, _, _) in enumerate(inf):
                if df >= 0:
                    dfs[i] += df
        df_record = dict(zip(distinct, dfs))
        found = sum(1 for d in dfs if d > 0)
        if not found:
            return result([], 0, df_record)
        w = [float(weights[t]) for t in distinct]
        cfg = self.config

        def score(i: int) -> bytes | None:
            seg, inf = lex_segs[i], infos[i]
            if not any(df >= 0 for df, _, _ in inf):
                return None
            tomb = tomb_for_segment(self.tomb, seg.ref["addedRev"]) if self.tomb else b""
            return seg.run(lambda: seg.reader.search(inf, dfs, w, num_docs, total_len, cfg.k1, cfg.b, k, tomb), io, trace)

        packed = self._each(list(range(len(lex_segs))), score)  # type: ignore[arg-type]
        # Ids are unique across live segments (older copies of a replaced id are tombstoned); keep the best anyway.
        best: dict[int, float] = {}
        for p in packed:
            if p:
                for h in parse_hits(p):
                    if best.get(h.id, float("-inf")) < h.score:
                        best[h.id] = h.score
        hits = sorted((LexHit(i, s) for i, s in best.items()), key=lambda h: (-h.score, h.id))[:k]
        return result(hits, found, df_record)

    def docs(self, ids: Iterable[int], *, text: bool = False, trace: BlobTrace | None = None) -> tuple[dict[int, DocRecord], LexIo]:
        """Resolves ids through the `docs` sections, newest segment first (the live copy of a replaced id is the
        newest one not hidden by a later tombstone set). Ids with no live entry are absent."""
        io = LexIo()
        out: dict[int, DocRecord] = {}
        pending = list(dict.fromkeys(ids))
        for seg in reversed(self.segments):
            if not pending:
                break
            if not seg.has_docs:
                continue
            added = seg.ref["addedRev"]
            want = [i for i in pending if not tomb_hides(self.tomb, i, added)]
            if not want:
                continue
            got = seg.run(functools.partial(seg.reader.docs_get, array("I", want), text), io, trace)
            resolved = set()
            for wid, g in zip(want, got):
                if g is None:
                    continue
                resolved.add(wid)
                out[wid] = DocRecord(wid, g[1], g[2] if text and g[2] is not None else None)
            pending = [i for i in pending if i not in resolved]
        return out, io
