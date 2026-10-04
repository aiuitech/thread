"""The dense query path, the same algorithm as the reference Worker and the TypeScript SDK:

1. route (flat over the cell means by default), build the PQ table (format 3);
2. phase 1: read the scan prefix of each probed slab (a cache miss reads the whole object and keeps the local
   winners' int8 records), scan it as it arrives with the tombstones applied inside the scan; scan every delta
   segment (int8 scores, no phase 2);
3. phase 2: re-score the global top-T with the int8 rerank records, read with one multi-range request per slab
   (speculatively, once only a few slabs are left to scan);
4. top-k over reranked slab hits and delta hits, one copy per id.

The per-hit work (scans, the best copy per id, the local rerank records, the phase-2 plan, the rerank and the final
merge) happens in the native `QueryRun`; this module does the reads. For a remote store they run on the index's
I/O thread pool, at most `inflight` per query, and each slab is scanned (with the GIL released) as soon as it
arrives; pool tasks never wait for other pool tasks (only the calling thread waits), so concurrent queries cannot
deadlock the pool. For a local store (memory, disk) reads are cheap and everything runs on the calling thread.
"""

from __future__ import annotations

import functools
import threading
import time
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from . import _native
from .cold import ColdRouting, new_cold_stat
from .store import BlobRange, BlobStore, BlobTrace, Bytes, phase_stat
from .wire import LEGACY_DEFAULTS, TombState, ThreadError, keys, slab_rev_of, tomb_for, tomb_for_delta

__all__ = ["Hit", "QueryResult", "RevState", "Base", "QUERY_INFLIGHT", "execute_query"]

QUERY_INFLIGHT = 12
#: Rerank winners of one slab whose records lie within this span share one range (adjacent records).
MULTI_COALESCE = 4 * 1024
#: Speculate on the running top-T once only this many slabs are left to scan.
SPEC_AT = 6


@dataclass
class Hit:
    id: int
    #: Cosine similarity (int8-tier scale for reranked slab hits and delta hits; scan-tier scale when rerank = 0).
    score: float
    #: The u32 attribute word stored with the vector.
    attrs: int
    #: Where the winning copy came from: "slab" or "delta".
    source: Literal["slab", "delta"]
    #: With `return_payload`: the payload bytes (empty when none was stored).
    payload: bytes | None = None


@dataclass
class QueryResult:
    hits: list[Hit]
    #: `{top_k, nprobe, rerank, pc, source: {nprobe, rerank, pc}}`; source is "query", "manifest" or "legacy".
    params: dict[str, Any]
    #: Routing, phase 1/2, delta and tombstone counters, store I/O and timings (ms); see the README.
    stats: dict[str, Any]


@dataclass
class RevState:
    """One manifest revision, parsed."""

    gen: str
    rev: int
    manifest: dict[str, Any]
    files: dict[str, str | None]
    format: int
    #: Per-cell slab layout (formats 2/3); None for v1 generations, whose slabs are read whole.
    slabs: list[list[int]] | None
    deltas: list[dict[str, Any]]
    #: Lexical / docs segments, oldest first.
    segments: list[dict[str, Any]]
    tomb: TombState | None
    base_key: str


@dataclass
class Base:
    """Routing table and PQ codebook of a generation, in native memory."""

    key: str
    #: The full routing table (None for a cold base, which routes with `cold`).
    searcher: _native.Searcher | None
    pq: _native.PqCodebook | None
    #: The routing table the searcher was built from ("f32" = `files.fine`; the grouped table's for a cold base).
    routing: str = "f32"
    #: Cold routing (`cold_routing="tree"`): the centroid tree plus the groups' means loaded so far.
    cold: ColdRouting | None = None


@dataclass
class QueryContext:
    store: BlobStore
    rev: RevState
    base: Base
    io: Executor
    #: A delta segment (whole), shared between concurrent queries.
    get_delta: Callable[[dict[str, Any], BlobTrace], Future]


class Limiter:
    """Bounds a query's in-flight reads; low-priority acquirers (speculative reads) start only when no normal one
    waits."""

    def __init__(self, max_inflight: int) -> None:
        self.max = max(1, max_inflight)
        self.active = 0
        self.waiting = 0
        self.cond = threading.Condition()

    def run(self, fn: Callable[[], Any], low: bool = False) -> Any:
        with self.cond:
            if not low:
                self.waiting += 1
            try:
                while self.active >= self.max or (low and self.waiting > 0):
                    self.cond.wait()
            finally:
                if not low:
                    self.waiting -= 1
            self.active += 1
        try:
            return fn()
        finally:
            with self.cond:
                self.active -= 1
                self.cond.notify_all()


@dataclass
class _Run:
    """Records `a..b` of one slab read as one range; `members` are the ones wanted."""

    a: int
    b: int
    members: list[int] = field(default_factory=list)


def _coalesce_runs(idxs: list[int], rec_len: int) -> list[_Run]:
    runs: list[_Run] = []
    for i in idxs:
        if runs and (i - runs[-1].a + 1) * rec_len <= MULTI_COALESCE:
            runs[-1].b = i
            runs[-1].members.append(i)
        else:
            runs.append(_Run(i, i, [i]))
    return runs


def _int_param(name: str, v: Any, lo: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ThreadError(f"{name} must be an integer >= {lo}", 400)
    return v


def _wait_all(futs: list[Future]) -> None:
    """Waits for every future, then raises the first error (so no task is still feeding the query)."""
    err: BaseException | None = None
    for f in futs:
        try:
            f.result()
        except BaseException as e:  # noqa: BLE001
            err = err or e
    if err is not None:
        raise err


def execute_query(
    ctx: QueryContext,
    q_raw: Any,
    *,
    top_k: int = 10,
    nprobe: int | None = None,
    rerank: int | None = None,
    pc: int | None = None,
    return_payload: bool = False,
    inflight: int = QUERY_INFLIGHT,
    speculate: bool = True,
    beam: int | None = None,
) -> QueryResult:
    store, rev, base, io = ctx.store, ctx.rev, ctx.base, ctx.io
    manifest, gen, fmt, tomb, deltas = rev.manifest, rev.gen, rev.format, rev.tomb, rev.deltas
    t_start = time.perf_counter()
    k = _int_param("top_k", top_k, 1)
    defaults = manifest.get("defaults") or LEGACY_DEFAULTS

    def src(v: int | None) -> str:
        return "query" if v is not None else "manifest" if manifest.get("defaults") else "legacy"

    nprobe_v = _int_param("nprobe", nprobe if nprobe is not None else defaults["nprobe"], 1)
    pc_v = _int_param("pc", pc if pc is not None else defaults["pc"], 0)
    rerank_v = _int_param("rerank", rerank if rerank is not None else defaults["rerank"], 0)
    params = {"top_k": k, "nprobe": nprobe_v, "rerank": rerank_v, "pc": pc_v,
              "source": {"nprobe": src(nprobe), "rerank": src(rerank), "pc": src(pc)}}
    # Format 1 slabs have no rerank tier: phase 2 is skipped whatever rerank says.
    do_rerank = fmt >= 2 and rev.slabs is not None and rerank_v > 0
    T = max(k, rerank_v) if do_rerank else k
    flat = pc_v == 0 or pc_v >= manifest["k1"]
    parallel = store.remote
    lim = Limiter(inflight)
    trace = BlobTrace()
    tr_start = trace.as_dict()
    # Delta segments are read first (usually from memory) and scanned while the slab reads are in flight.
    delta_reads = [ctx.get_delta(d, trace) for d in deltas]
    cold = base.cold
    cold_stat: dict[str, int] | None = None
    try:
        if cold is not None:
            # Cold routing: the tree's best groups (their means read now unless an earlier query loaded them), then
            # the members of those groups ranked as the full table would rank them. Delta segments get q·mean for
            # the cells they hold entries in (those rows read too).
            beam_v = _int_param("beam", beam if beam is not None else cold.beam, 1)
            cold_stat = new_cold_stat(beam_v)
            q0 = cold.router.query(q_raw)
            cold.ensure_groups(q0, beam_v, trace, cold_stat)
            dot_cells = sorted({c for d, f in zip(deltas, delta_reads) for c in cold.cells_of(d["key"], f.result())})
            cold.ensure_cells(dot_cells, trace, cold_stat)
            query = cold.router.query(q_raw, dot_cells=dot_cells or None, codebook=base.pq)
            cells = cold.router.route_beam(query, beam_v, nprobe_v)
        else:
            assert base.searcher is not None
            query = base.searcher.query(q_raw, dots=flat or bool(deltas), codebook=base.pq)
            cells = base.searcher.route(query, 0 if flat else pc_v, nprobe_v)
    except ValueError as e:
        raise ThreadError(str(e), 400) from e
    t_route = time.perf_counter()
    tr_route = trace.as_dict()

    run = _native.QueryRun(query, T, do_rerank, return_payload)
    slab_keys: dict[int, str] = {}
    lock = threading.Lock()
    st = {"rerank_reads": 0, "rerank_bytes": 0}
    spec_futs: list[Future] = []
    spec_fired = False

    def read(fn: Callable[[], Any], low: bool = False) -> Any:
        return lim.run(fn, low) if parallel else fn()

    def read_records(cell: int, rerank_off: int, rec_len: int, runs: list[_Run], low: bool, multi: bool) -> None:
        """Reads `runs` of one slab's rerank records (one multi-range request when `multi`, else one read per run)
        and hands them to the run. A failed speculative read is dropped: phase 2 reads those records itself."""
        try:
            key = slab_keys[cell]
            ranges = [BlobRange(rerank_off + r.a * rec_len, (r.b - r.a + 1) * rec_len) for r in runs]
            got = read(functools.partial(store.get_ranges, key, ranges, trace=trace), low) if multi else None
            n_reads = 1
            if got is None:
                got = [read(functools.partial(store.get, key, r, trace=trace), low) for r in ranges]
                n_reads = len(ranges)
            for r, b in zip(runs, got):
                if b is None:
                    raise ThreadError(f"missing slab {key}", 404)
                run.add_records(cell, r.a, b, r.members, low)
            with lock:
                st["rerank_reads"] += n_reads
                st["rerank_bytes"] += sum(len(b) for b in got)
        except Exception:
            if not low:
                raise

    def records(cell: int, rerank_off: int, rec_len: int, idxs: list[int], low: bool) -> list[Future]:
        """Reads records `idxs` of one slab: inline for a local store; on the pool as one task trying a multi-range
        request, or, for a store without them, one task per run (in parallel)."""
        runs = _coalesce_runs(idxs, rec_len)
        multi = store.supports_ranges and len(runs) > 1
        if not parallel:
            read_records(cell, rerank_off, rec_len, runs, low, multi)
            return []
        if multi:
            return [io.submit(read_records, cell, rerank_off, rec_len, runs, low, True)]
        return [io.submit(read_records, cell, rerank_off, rec_len, [r], low, False) for r in runs]

    def read_slab(cell: int) -> tuple[Bytes, Bytes | None, int]:
        """Phase 1 read: the scan prefix (v2/v3), the whole object on a cache miss; v1 slabs whole."""
        if rev.slabs is None:
            key = slab_keys[cell] = keys.slab(gen, cell)
            buf = read(lambda: store.get(key, None, trace=trace))
            if buf is None:
                raise ThreadError(f"missing slab {key}", 404)
            return buf, buf, 0
        meta = rev.slabs[cell] if cell < len(rev.slabs) else None
        if not meta:
            raise ThreadError(f"{rev.files['slabs']} has no entry for cell {cell}")
        slab_rev = slab_rev_of(meta)
        key = slab_keys[cell] = keys.slab(gen, cell, slab_rev)
        scan_len = meta[0]
        b = read(lambda: store.get(key, BlobRange(0, scan_len), trace=trace, fill=True))
        if b is None:
            raise ThreadError(f"missing slab {key}", 404)
        # A cache store that fetched the whole object returns a view into it.
        full = None
        if isinstance(b, memoryview) and isinstance(b.obj, bytes) and len(b.obj) == meta[3] and meta[3] > scan_len:
            full = b.obj
        return b, full, slab_rev

    def slab_task(cell: int) -> None:
        nonlocal spec_fired
        prefix, full, slab_rev = read_slab(cell)
        # Tombstones are applied inside the scan (before top-k insertion): ids deleted or replaced at a rev >= this
        # slab's revision (every set for rev-0 slabs).
        rerankable, done = run.add_slab(cell, prefix, full, tomb_for(tomb, slab_rev) if tomb else None)
        if not (parallel and speculate and rerankable and len(cells) - done <= SPEC_AT):
            return
        with lock:
            if spec_fired:
                return
            spec_fired = True
        # Once only a few slabs are left to scan, read the running top-T's records at low priority.
        for c, off, rec_len, idxs in run.speculation():
            futs = records(c, off, rec_len, idxs, True)
            with lock:
                spec_futs.extend(futs)

    def scan_deltas() -> None:
        # Every delta entry is scored with its int8 record; it is hidden by the sets newer than the rev that added
        # its segment.
        for d, fut in zip(deltas, delta_reads):
            run.add_delta(fut.result(), tomb_for_delta(tomb, d["addedRev"]) if tomb else None)

    if parallel:
        slab_futs = [io.submit(slab_task, c) for c in cells]
        try:
            scan_deltas()
        finally:
            _wait_all(slab_futs)
        for f in spec_futs:
            f.exception()  # wait; a failed speculative read only means phase 2 reads those records itself
    else:
        for c in cells:
            slab_task(c)
        scan_deltas()
    t_phase1 = time.perf_counter()
    tr_phase1 = trace.as_dict()

    # Phase 2: the global top-T's records that are neither local nor read speculatively, one request per slab.
    futs = []
    for c, off, rec_len, idxs in run.plan():
        futs.extend(records(c, off, rec_len, idxs, False))
    _wait_all(futs)
    raw, payload_reads = run.finish(k)
    hits = [Hit(i, s, a, src_, p) for i, s, a, src_, p in raw]
    t_phase2 = time.perf_counter()
    tr_phase2 = trace.as_dict()
    if payload_reads:
        _fetch_payloads(hits, payload_reads, slab_keys, store, io if parallel else None, read, trace)
    t_end = time.perf_counter()
    tr_end = trace.as_dict()
    rs = run.stats()
    ms = lambda a, b: round((b - a) * 1000, 3)  # noqa: E731
    return QueryResult(
        hits=hits,
        params=params,
        stats={
            "gen": gen,
            "rev": rev.rev,
            "format": fmt,
            "routing": "cold" if cold is not None else "flat" if flat else "tree",
            "routing_table": base.routing,
            **({"cold": cold.finish_stat(cold_stat)} if cold is not None and cold_stat is not None else {}),
            "cells": cells,
            "scanned": rs["scanned"],
            "scan_bytes": rs["scan_bytes"],
            "slab_bytes": rs["slab_bytes"],
            "rerank": rs["rerank"],
            "rerank_candidates": rs["rerank_candidates"],
            "rerank_slabs": rs["rerank_slabs"],
            "rerank_reads": st["rerank_reads"],
            "rerank_bytes": st["rerank_bytes"],
            "rerank_local": rs["rerank_local"],
            "spec_records": rs["spec_records"],
            "spec_used": rs["spec_used"],
            "deltas": len(deltas),
            "delta_bytes": rs["delta_bytes"],
            "delta_scanned": rs["delta_scanned"],
            "delta_hits": rs["delta_hits"],
            "delta_dropped": rs["delta_dropped"],
            "delta_top": sum(1 for h in hits if h.source == "delta"),
            "tombstones": tomb.total if tomb else 0,
            "tomb_sets": len(tomb.sets) if tomb else 0,
            "tomb_dropped": rs["tomb_dropped"],
            "io": trace.as_dict(),
            "timing": {"total": ms(t_start, t_end), "route": ms(t_start, t_route), "phase1": ms(t_route, t_phase1),
                       "phase2": ms(t_phase1, t_phase2), "payload": ms(t_phase2, t_end)},
            # Per-phase wall time and the store work that completed during it (trace snapshots at the boundaries).
            "timeline": {"route": phase_stat((t_route - t_start) * 1000, tr_route, tr_start),
                         "phase1": phase_stat((t_phase1 - t_route) * 1000, tr_phase1, tr_route),
                         "phase2": phase_stat((t_phase2 - t_phase1) * 1000, tr_phase2, tr_phase1),
                         "payload": phase_stat((t_end - t_phase2) * 1000, tr_end, tr_phase2)},
        },
    )


def _fetch_payloads(hits: list[Hit], reads: list[tuple[int, int, int, int]], slab_keys: dict[int, str],
                    store: BlobStore, io: Executor | None, read: Callable[..., Any], trace: BlobTrace) -> None:
    """Payloads of slab hits whose slab was not in memory: one (multi-)range read per slab."""
    by_key: dict[str, list[tuple[int, BlobRange]]] = {}
    for pos, cell, off, n in reads:
        by_key.setdefault(slab_keys[cell], []).append((pos, BlobRange(off, n)))

    def fetch(key: str, items: list[tuple[int, BlobRange]]) -> None:
        ranges = [r for _, r in items]
        got = read(functools.partial(store.get_ranges, key, ranges, trace=trace)) if len(items) > 1 else None
        if got is None:
            got = [read(functools.partial(store.get, key, r, trace=trace)) for r in ranges]
        for (pos, _), b in zip(items, got):
            if b is None:
                raise ThreadError(f"missing slab {key}", 404)
            hits[pos].payload = bytes(b)

    if io is None:
        for key, items in by_key.items():
            fetch(key, items)
    else:
        _wait_all([io.submit(fetch, key, items) for key, items in by_key.items()])
