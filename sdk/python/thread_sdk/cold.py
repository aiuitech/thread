"""Cold routing (`open_index(..., cold_routing="tree")`), the Python counterpart of the TypeScript SDK's `cold.ts`:
until the full routing table has loaded, a query routes with the centroid tree and the means of just the groups it
needs, range-read from the tree-grouped table (`fine.grouped.{rev}.{elem}`, "VIBG"). Group g's rows are
`[dataOff + offsets[g] * rowBytes, +n * rowBytes)` with the tree's CSR offsets. Loaded groups stay loaded for later
queries; concurrent queries share in-flight reads. Delta segments need `q·mean` for the cells they hold entries in:
their rows are read the same way, so delta scores match the full table's."""

from __future__ import annotations

import threading
from concurrent.futures import Executor, Future
from typing import Any, Callable

from . import _native
from .store import BlobRange, BlobStore, BlobTrace
from .wire import BytesLike, ThreadError

__all__ = ["ColdRouting", "new_cold_stat"]


def new_cold_stat(beam: int) -> dict[str, int]:
    """What a cold query had to read (`stats["cold_routing"]`): the beam, the groups and delta cells whose means it
    read, the requests and bytes of those reads, and the groups / cells loaded so far."""
    return {"beam": beam, "groups": 0, "cells": 0, "reads": 0, "bytes": 0, "loaded_groups": 0, "loaded_cells": 0}


def _coalesce(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for first, n in sorted(runs):
        if out and out[-1][0] + out[-1][1] == first:
            out[-1][1] += n
        else:
            out.append([first, n])
    return [(a, n) for a, n in out]


class ColdRouting:
    """The cold router of one revision's base and the reads that feed it. Thread-safe."""

    def __init__(self, store: BlobStore, router: _native.ColdRouter, ref: dict[str, Any], beam: int,
                 io: Executor | None = None) -> None:
        self.store = store
        #: Reads of several runs go in parallel on this pool (a store without multi-range requests); the caller is
        #: never a pool thread itself (queries call in from their own thread).
        self.io = io
        self.router = router
        self.ref = ref
        self.beam = beam
        self.k = router.k
        self._lock = threading.Lock()
        self._pending: dict[str, Future] = {}
        self._delta_cells: dict[str, list[int]] = {}

    def _read_runs(self, runs: list[tuple[int, int]], trace: BlobTrace, stat: dict[str, int]) -> None:
        rb, off, key = self.ref["rowBytes"], self.ref["dataOff"], self.ref["key"]
        ranges = [BlobRange(off + first * rb, n * rb) for first, n in runs]
        # i8 rows are residuals from the centre in the table's head: the first read brings it along.
        head = self.router.needs_head
        if head:
            ranges.insert(0, BlobRange(0, off))
        multi = len(ranges) > 1 and self.store.supports_ranges
        got = self.store.get_ranges(key, ranges, trace=trace) if multi else None
        reads = 1
        if got is None:
            if self.io is not None and len(ranges) > 1 and self.store.remote:
                got = list(self.io.map(lambda r: self.store.get(key, r, trace=trace), ranges))
            else:
                got = [self.store.get(key, r, trace=trace) for r in ranges]
            reads = len(ranges)
        if head:
            h = got[0]
            if h is None:
                raise ThreadError(f"missing {key}", 404)
            if self.router.needs_head:
                self.router.set_head(h)
        for (first, n), b in zip(runs, got[1:] if head else got):
            if b is None:
                raise ThreadError(f"missing {key}", 404)
            if len(b) != n * rb:
                raise ThreadError(f"{key}: rows {first}..{first + n} came back as {len(b)} bytes")
            self.router.add_rows(first, b)
        with self._lock:
            stat["reads"] += reads
            stat["bytes"] += sum(len(b) for b in got)  # type: ignore[arg-type]

    def _load(self, ids: list[str], runs_of: Callable[[list[str]], list[tuple[int, int]]], trace: BlobTrace,
              stat: dict[str, int]) -> int:
        """Starts (or joins) the loads of `ids` (`g`/`c` prefixed); returns how many this call read."""
        waits: list[Future] = []
        fresh: list[str] = []
        fut: Future = Future()
        with self._lock:
            for i in ids:
                p = self._pending.get(i)
                if p is not None:
                    waits.append(p)
                else:
                    fresh.append(i)
                    self._pending[i] = fut
        if fresh:
            try:
                self._read_runs(_coalesce(runs_of(fresh)), trace, stat)
            except BaseException as e:
                with self._lock:
                    for i in fresh:
                        if self._pending.get(i) is fut:
                            del self._pending[i]
                fut.set_exception(e)
                raise
            fut.set_result(None)
        for w in waits:
            w.result()
        return len(fresh)

    def ensure_groups(self, query: _native.Query, beam: int, trace: BlobTrace, stat: dict[str, int]) -> None:
        """Loads the groups among the `beam` best for the (prepared) query that are not loaded yet."""
        missing = self.router.missing_groups(query, beam)
        if missing:
            stat["groups"] += self._load([f"g{g}" for g in missing],
                                         lambda fresh: [self.router.group_rows(int(i[1:])) for i in fresh], trace, stat)

    def ensure_cells(self, cells: list[int], trace: BlobTrace, stat: dict[str, int]) -> None:
        """Loads the means of `cells` that are not loaded yet (by row of the grouped table)."""
        missing = sorted(set(self.router.missing_cells(cells)))
        if not missing:
            return

        def runs(fresh: list[str]) -> list[tuple[int, int]]:
            out = []
            for i in fresh:
                row = self.router.cell_row(int(i[1:]))
                if row is None:
                    raise ThreadError(f"{self.ref['key']}: cell {i[1:]} is in no group of the tree")
                out.append((row, 1))
            return out

        stat["cells"] += self._load([f"c{c}" for c in missing], runs, trace, stat)

    def cells_of(self, key: str, delta: BytesLike) -> list[int]:
        """The cells a delta segment holds entries in (cached by key)."""
        with self._lock:
            hit = self._delta_cells.get(key)
        if hit is None:
            hit = [c for c, _ in _native.delta_cells(delta)]
            with self._lock:
                self._delta_cells[key] = hit
        return hit

    def finish_stat(self, stat: dict[str, int]) -> dict[str, int]:
        stat["loaded_groups"] = self.router.loaded_groups
        stat["loaded_cells"] = self.router.loaded_cells
        return stat
