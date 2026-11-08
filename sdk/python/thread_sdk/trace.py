"""Trace envelopes: one JSON-shaped dict per SDK operation (query, search, lexical, open, get, listKeys, upsert,
delete), the same shape as the TypeScript SDK's `TraceEnvelope` and the reference Worker's, so one bench script, log
query or dashboard reads every host the same way. Keys are camelCase, as in the manifest. See docs/observability.md.

- `spans`: the operation's steps, `{name, start?, ms, io?, children?}` (start: ms from the operation's start);
- `counters`: work done (entries scanned, bytes, candidates, ...);
- `kernels`: CPU of the native kernels, `{name: {calls, ms}}`, measured inside the extension with the GIL released;
- `events` (opt-in, `trace_events=True` on a query): one entry per store read.

`open_index(on_trace=fn, trace_sample=0.1)` hands a sampled share of them to `fn`; a sink that raises never fails
the operation.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable, Iterable, Mapping

from .store import TRACE_FIELDS, BlobTrace

__all__ = ["TRACE_VERSION", "TraceSink", "envelope", "phase_span", "shift_spans", "emit_trace", "Kernels",
           "merge_kernels", "prefix_counters", "io_of", "zero_io", "add_io", "numeric_fields"]

TRACE_VERSION = 1

#: `on_trace` receives one envelope per operation.
TraceSink = Callable[[dict[str, Any]], None]


def zero_io() -> dict[str, int]:
    return {k: 0 for k in TRACE_FIELDS}


def io_of(t: BlobTrace | Mapping[str, Any]) -> dict[str, int]:
    """The `io` part of a store trace or a phase stat."""
    d = t.as_dict() if isinstance(t, BlobTrace) else t
    return {k: int(d[k]) for k in TRACE_FIELDS}


def add_io(a: Mapping[str, int], b: Mapping[str, int] | None) -> dict[str, int]:
    return {k: a[k] + (b[k] if b else 0) for k in TRACE_FIELDS}


def phase_span(name: str, start: float | None, p: Mapping[str, Any],
               children: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A span from a phase stat (`timeline[...]`, `open_stats[...]`)."""
    s: dict[str, Any] = {"name": name}
    if start is not None:
        s["start"] = round(start, 3)
    s["ms"] = p["ms"]
    s["io"] = io_of(p)
    if children:
        s["children"] = children
    return s


def shift_spans(spans: list[dict[str, Any]], by: float) -> list[dict[str, Any]]:
    """The same spans `by` ms later (a nested operation placed inside its parent)."""
    out = []
    for s in spans:
        c = dict(s)
        if "start" in c:
            c["start"] = round(c["start"] + by, 3)
        if "children" in c:
            c["children"] = shift_spans(c["children"], by)
        out.append(c)
    return out


class Kernels:
    """Kernel timers: `k.time("route", fn)` adds one call and its wall time; `k.merge(native_stats)` adds the
    extension's own measurements."""

    def __init__(self) -> None:
        self.stats: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def add(self, name: str, calls: int, ms: float) -> None:
        with self._lock:
            k = self.stats.setdefault(name, [0, 0.0])
            k[0] += calls
            k[1] += ms

    def time(self, name: str, fn: Callable[[], Any]) -> Any:
        t = time.perf_counter()
        try:
            return fn()
        finally:
            self.add(name, 1, (time.perf_counter() - t) * 1000)

    def merge(self, native: Mapping[str, Mapping[str, Any]]) -> None:
        for name, v in native.items():
            self.add(name, int(v["calls"]), float(v["ms"]))

    def done(self) -> dict[str, dict[str, Any]]:
        return {k: {"calls": int(v[0]), "ms": round(v[1], 3)} for k, v in self.stats.items()}


def merge_kernels(*all_: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Kernel stats of several operations, summed by name (a search's dense and lexical lists)."""
    k = Kernels()
    for ks in all_:
        if ks:
            k.merge(ks)
    return k.done()


def prefix_counters(prefix: str, c: Mapping[str, int] | None) -> dict[str, int]:
    """Counters of a nested operation under a prefix (`dense.scanned`, `lexical.rounds`)."""
    return {f"{prefix}.{k}": v for k, v in (c or {}).items()}


def numeric_fields(o: Any, skip: Iterable[str] = ()) -> dict[str, int | float]:
    """Every int / float among a result's public fields (a write result's counts and bytes)."""
    skip_ = set(skip)
    fields = vars(o) if hasattr(o, "__dict__") else {}
    return {k: v for k, v in fields.items()
            if not k.startswith("_") and k not in skip_ and isinstance(v, (int, float)) and not isinstance(v, bool)}


def envelope(op: str, t0: float, at: float, *, attrs: Mapping[str, Any] | None = None,
             io: Mapping[str, int] | None = None, spans: list[dict[str, Any]] | None = None,
             counters: Mapping[str, int | float] | None = None, kernels: Mapping[str, Any] | None = None,
             events: list[dict[str, Any]] | None = None, ms: float | None = None) -> dict[str, Any]:
    """A Python SDK envelope. `t0` is `time.perf_counter()` at the start, `at` the epoch seconds at the start; `io`
    defaults to the sum of the top-level spans' io."""
    spans = spans or []
    total = zero_io()
    for s in spans:
        total = add_io(total, s.get("io"))
    e: dict[str, Any] = {
        "v": TRACE_VERSION,
        "op": op,
        "sdk": "py",
        "at": int(at * 1000),
        "ms": round(ms if ms is not None else (time.perf_counter() - t0) * 1000, 3),
        "clock": "real",
        "attrs": dict(attrs or {}),
        "io": dict(io) if io is not None else total,
        "spans": spans,
        "counters": dict(counters or {}),
        "kernels": dict(kernels or {}),
    }
    if events is not None:
        e["events"] = events
    return e


def emit_trace(sink: TraceSink | None, sample: float, t: dict[str, Any] | None) -> None:
    """Hands an envelope to the sink, sampled; a sink that raises never fails the operation."""
    if sink is None or t is None:
        return
    if sample < 1 and random.random() >= sample:
        return
    try:
        sink(t)
    except Exception:  # noqa: BLE001 - observability must not break a query
        pass
