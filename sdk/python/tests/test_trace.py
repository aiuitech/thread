"""Trace envelopes (docs/observability.md): every operation produces one, its spans, io, counters and kernels agree
with the operation's own stats, `trace_events` records each read, and `on_trace` receives them (sampled, once per
operation, never for the nested lists of a search). The cross-SDK shape check is in test_interop.py."""

from __future__ import annotations

import json

from thread_sdk import MemoryBlobStore, create_generation, open_index
from thread_sdk.trace import TRACE_VERSION

from helpers import build_gen, rng, row, unit_rows


def names(spans):
    return [s["name"] for s in spans]


def well_formed(t: dict) -> None:
    assert t["v"] == TRACE_VERSION and t["sdk"] == "py" and t["clock"] == "real"
    assert t["ms"] >= 0 and t["at"] > 1.7e12
    assert all(isinstance(v, (int, float)) for v in t["counters"].values())
    for k in t["kernels"].values():
        assert k["calls"] > 0 and k["ms"] >= 0

    def walk(spans):
        for s in spans:
            assert s["ms"] >= 0 and s.get("start", 0) >= 0
            walk(s.get("children", []))

    walk(t["spans"])
    assert json.loads(json.dumps(t)) == t  # plain JSON data


def test_query_envelope_matches_its_stats():
    b = build_gen("t1", 3, n=600, k2=8)
    with open_index(b.store, cache=False, refresh_interval=0) as ix:
        r = ix.query(row(b.vectors, 5, b.dim), top_k=5, nprobe=4, rerank=30)
        t = r.stats["trace"]
        well_formed(t)
        assert t["op"] == "query"
        assert names(t["spans"]) == ["route", "phase1", "phase2", "payload"]
        assert {k: t["attrs"][k] for k in ("gen", "rev", "format", "routing", "topK", "nprobe", "rerank", "filtered")} == \
            {"gen": "t1", "rev": 0, "format": 3, "routing": "flat", "topK": 5, "nprobe": 4, "rerank": 30, "filtered": False}
        c = t["counters"]
        assert c["scanned"] == r.stats["scanned"] and c["cells"] == len(r.stats["cells"]) and c["hits"] == len(r.hits)
        assert c["pushed"] == r.stats["pushed"] and r.stats["candidates"] <= c["pushed"] <= c["scanned"]
        assert t["io"]["bytes"] == r.stats["io"]["bytes"]
        assert sum(s["io"]["bytes"] for s in t["spans"]) == t["io"]["bytes"]
        # The kernels the extension timed natively, plus routing timed around the native calls.
        assert sorted(t["kernels"]) == ["merge", "rerank", "route", "scan"]
        assert t["kernels"]["scan"]["calls"] == len(r.stats["cells"])
        assert t["kernels"]["scan"]["ms"] > 0
        assert "events" not in t


def test_trace_events_record_every_read():
    b = build_gen("t2", 3, n=600, k2=8)
    with open_index(b.store, cache=False, refresh_interval=0) as ix:
        r = ix.query(row(b.vectors, 0, b.dim), top_k=5, nprobe=4, rerank=30, trace_events=True, return_payload=True)
        ev = r.stats["trace"]["events"]
        assert sum(1 for e in ev if e["kind"] == "prefix") == len(r.stats["cells"])
        assert sum(e["bytes"] for e in ev) == r.stats["trace"]["io"]["bytes"]
        assert all(e["key"].startswith("idx/t2/") and e["ranges"] >= 1 for e in ev)
        assert [e["start"] for e in ev] == sorted(e["start"] for e in ev)
        plain = ix.query(row(b.vectors, 0, b.dim), top_k=5, nprobe=4, rerank=30, return_payload=True)
        assert [h.id for h in plain.hits] == [h.id for h in r.hits]


def test_search_nests_lists_and_every_operation_has_an_envelope():
    store = MemoryBlobStore()
    dim = 16
    create_generation(store, "h", dim=dim, compaction=False)
    seen: list[dict] = []
    with open_index(store, cache=False, refresh_interval=0, on_trace=seen.append) as ix:
        assert [t["op"] for t in seen] == ["open"]
        well_formed(ix.open_trace)
        assert names(ix.open_trace["spans"]) == ["current", "prefetch", "manifest", "slabs", "base"]
        vecs = unit_rows(40, dim, rng(3))
        texts = [f"deploy failed on host {i}" if i % 3 == 0 else f"budget review {i}" for i in range(40)]
        up = ix.upsert(keys=[f"doc/{i}" for i in range(40)], vectors=vecs, texts=texts, metas=texts)
        well_formed(up.trace)
        assert up.trace["op"] == "upsert" and names(up.trace["spans"]) == ["commit", "useRevision"]
        assert up.trace["counters"]["count"] == 40

        seen.clear()
        s = ix.search(text="deploy failed", vector=row(vecs, 0, dim), top_k=5, docs=True)
        t = s.stats["trace"]
        well_formed(t)
        assert [x["op"] for x in seen] == ["search"]  # the dense and lexical lists are spans, not envelopes
        assert t["attrs"]["mode"] == "hybrid"
        assert sorted(names(t["spans"])) == ["dense", "docs", "fuse", "lexical"]
        by = {x["name"]: x for x in t["spans"]}
        assert names(by["dense"]["children"]) == ["route", "phase1", "phase2", "payload"]
        assert names(by["lexical"]["children"]) == ["lookup", "score"]
        assert t["counters"]["dense.scanned"] == s.dense.stats["scanned"]
        assert t["counters"]["lexical.hits"] == len(s.lexical.hits)
        assert t["kernels"]["bm25Score"]["calls"] > 0 and t["kernels"]["deltaScan"]["calls"] == 1

        got = ix.get(["doc/0", "doc/1", "nope"])
        assert len(got) == 2
        assert seen[-1]["op"] == "get" and seen[-1]["counters"]["keys"] == 3 and seen[-1]["counters"]["found"] == 2
        assert names(seen[-1]["spans"]) == ["keys", "docs"]
        page = ix.list_keys("doc/", limit=10)
        assert page.trace["counters"] == {**page.trace["counters"], "entries": 10, "more": 1}
        d = ix.delete([0, 1])
        assert d.trace["op"] == "delete"
        assert [x["op"] for x in seen] == ["search", "get", "listKeys", "delete"]


def test_sampling_and_a_failing_sink():
    b = build_gen("t3", 2, n=200)
    calls = []
    with open_index(b.store, cache=False, refresh_interval=0, on_trace=calls.append, trace_sample=0.0) as ix:
        r = ix.query(row(b.vectors, 0, b.dim), top_k=3)
        assert r.stats["trace"]["op"] == "query"
    assert calls == []

    def boom(_t):
        raise RuntimeError("sink down")

    with open_index(b.store, cache=False, refresh_interval=0, on_trace=boom) as ix:
        assert len(ix.query(row(b.vectors, 0, b.dim), top_k=3).hits) == 3
