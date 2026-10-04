"""Compact routing tables (fine.f16 / fine.i8): opening with `routing`, fallbacks, snapshots."""

from __future__ import annotations

import pytest

from helpers import Gated, add_routing_rev, brute_force, build_gen, rng, row, unit_rows
from thread_sdk import _native, keys, open_index
from thread_sdk.revision import load_rev_state, snapshot_keys
from thread_sdk.wire import choose_routing


def test_encode_decode_and_wire_compat():
    dim = 16
    fine = unit_rows(20, dim, rng(3))
    for elem, step in (("f16", 1e-3), ("i8", 1 / 127)):
        b = _native.routing_table_encode(fine, dim, elem)
        assert len(b) == (20 * 2 * dim if elem == "f16" else 4 * dim + 20 * (4 + dim))
        d = memoryview(_native.routing_table_decode(b, dim, elem)).cast("f")
        assert all(abs(d[i] - fine[i]) < step for i in range(len(fine)))
    with pytest.raises(ValueError, match="f32, f16 or i8"):
        _native.routing_table_encode(fine, dim, "f8")
    with pytest.raises(ValueError, match="20 rows of"):
        _native.Searcher.from_table(dim, 1, 20, [0.25] * dim, _native.routing_table_encode(fine, dim, "f16")[2:],
                                    "f16")
    assert keys.routing_table("idx/g/fine.105.f32", "i8") == "idx/g/fine.105.i8"


def test_open_with_routing_and_fallbacks():
    b = build_gen("gr", 3, k2=16, n=600)
    q0 = row(unit_rows(1, b.dim, rng(9)), 0, b.dim)
    with open_index(b.store, cache=False, routing="auto") as ix:
        assert ix.query(q0).stats["routing_table"] == "f32"  # no tables yet
    add_routing_rev(b.store, "gr")
    qs = unit_rows(30, b.dim, rng(5))
    cells = {}
    for routing in ("f32", "f16", "i8", "auto"):
        with open_index(b.store, cache=False, routing=routing) as ix:
            cells[routing] = []
            for i in range(30):
                r = ix.query(row(qs, i, b.dim), nprobe=4, top_k=5)
                assert r.stats["routing_table"] == ("i8" if routing == "auto" else routing)
                assert len(r.hits) == 5
                cells[routing].append(r.stats["cells"])
    assert cells["f16"] == cells["f32"]
    same = sum(len(set(a) & set(b_)) for a, b_ in zip(cells["i8"], cells["f32"]))
    assert same / 120 > 0.95
    with pytest.raises(Exception, match="routing must be"):
        open_index(b.store, cache=False, routing="f8")


def test_stale_tables_ignored_and_snapshots():
    b = build_gen("gs", 3)
    add_routing_rev(b.store, "gs", tables=("f16",))
    ok = load_rev_state(b.store, "gs", 1)
    assert "idx/gs/fine.f16" in snapshot_keys(ok)
    assert choose_routing(ok.manifest, ok.files, "i8")[0] == "f32"
    add_routing_rev(b.store, "gs", stale_fine=True)
    stale = load_rev_state(b.store, "gs", 2)
    assert "idx/gs/fine.f16" not in snapshot_keys(stale)
    assert choose_routing(stale.manifest, stale.files, "f16") == ("f32", "idx/gs/fine.f32")
    with open_index(b.store, cache=False, routing="f16") as ix:
        assert ix.query(row(unit_rows(1, b.dim, rng(2)), 0, b.dim)).stats["routing_table"] == "f32"


def test_cold_routing_full_beam_equals_flat_then_hands_over():
    b = build_gen("gt", 3, k2=48, n=1500)
    add_routing_rev(b.store, "gt", tree_groups=6, grouped_elem="f32")
    gated = Gated(b.store, "idx/gt/fine.f32")
    with open_index(b.store, cache=False) as full, open_index(gated, cache=False, cold_routing="tree",
                                                             cold_beam=6) as cold:
        assert cold.routing_state == "cold"
        qs = unit_rows(20, b.dim, rng(4))
        for i in range(20):
            q = row(qs, i, b.dim)
            a, c = full.query(q, nprobe=5), cold.query(q, nprobe=5)
            assert c.stats["routing"] == "cold"
            assert c.stats["cells"] == a.stats["cells"]
            assert [(h.id, h.score) for h in c.hits] == [(h.id, h.score) for h in a.hits]
        gated.release.set()
        cold.when_full_routing()
        assert cold.routing_state == "full"
        r = cold.query(row(qs, 0, b.dim), nprobe=4)
        assert r.stats["routing"] == "flat" and "cold_routing" not in r.stats


def test_cold_routing_reads_beam_groups_once_and_deltas_exact():
    b = build_gen("gb", 3, k2=32, n=800)
    add_routing_rev(b.store, "gb", tree_groups=8, grouped_elem="f32")
    fresh = unit_rows(40, b.dim, rng(77))
    with open_index(b.store, cache=False) as w:
        w.upsert(list(range(100000, 100040)), fresh)
    gated = Gated(b.store, "idx/gb/fine.f32")
    with open_index(gated, cache=False, cold_routing="tree", cold_beam=1) as cold, \
            open_index(b.store, cache=False) as full:
        assert len(cold.revision.deltas) == 1
        q = row(fresh, 0, b.dim)
        first = cold.query(q, top_k=3)
        st = first.stats["cold_routing"]
        assert st["beam"] == 1 and st["groups"] == 1 and st["cells"] > 0
        assert st["bytes"] == st["loaded_cells"] * 4 * b.dim
        again = cold.query(q, top_k=3)
        assert again.stats["cold_routing"]["groups"] == 0 and again.stats["cold_routing"]["bytes"] == 0
        for i in range(5):
            q = row(fresh, i, b.dim)
            c, f = cold.query(q, top_k=3), full.query(q, top_k=3)
            assert c.hits[0].id == 100000 + i and c.hits[0].source == "delta"
            assert c.hits[0].score == next(h.score for h in f.hits if h.id == 100000 + i)
        gated.release.set()
    plain = build_gen("gp", 3)
    with open_index(plain.store, cache=False, cold_routing="tree") as ix:
        assert ix.routing_state == "full"
        assert ix.query(row(unit_rows(1, plain.dim, rng(3)), 0, plain.dim)).stats["routing"] == "flat"


def test_cold_routing_i8_rows_fetch_the_head_with_the_first_groups():
    b = build_gen("gi", 3, k2=64, n=2000)
    add_routing_rev(b.store, "gi", tree_groups=8, grouped_elem="i8")
    gated = Gated(b.store, "idx/gi/fine.f32")
    with open_index(gated, cache=False, cold_routing="tree", cold_beam=8) as cold, \
            open_index(b.store, cache=False) as full:
        g = cold.manifest["routing"]["grouped"]
        q = row(unit_rows(1, b.dim, rng(21)), 0, b.dim)
        first = cold.query(q, nprobe=4)
        st = first.stats["cold_routing"]
        assert first.stats["routing_table"] == "i8"
        assert st["reads"] == 1 and st["groups"] == 8
        assert st["bytes"] == g["dataOff"] + st["loaded_cells"] * (4 + b.dim)
        # every group loaded: the ranking of the centred i8 means, almost always the f32 one on this index
        same = 0
        qs = unit_rows(30, b.dim, rng(22))
        for i in range(30):
            c, f = cold.query(row(qs, i, b.dim), nprobe=4), full.query(row(qs, i, b.dim), nprobe=4)
            same += len(set(c.stats["cells"]) & set(f.stats["cells"]))
        assert same / 120 > 0.95
        gated.release.set()


def test_attribute_filters_with_every_routing_mode():
    """Filters apply inside the scans and widen the same way whatever table routed the query (f32, f16, i8, cold)."""
    b = build_gen("gfr", 3, k2=32, n=1200, attrs=lambda i: i % 4)
    add_routing_rev(b.store, "gfr", tree_groups=8, grouped_elem="i8")
    qs = unit_rows(20, b.dim, rng(31))
    gated = Gated(b.store, "idx/gfr/fine.f32")
    opened = {m: open_index(b.store, cache=False, routing=m) for m in ("f32", "f16", "i8")}
    opened["cold"] = open_index(gated, cache=False, cold_routing="tree", cold_beam=8)
    got: dict[str, list[list[int]]] = {}
    hits: dict[str, int] = {}
    try:
        for mode, ix in opened.items():
            got[mode], hits[mode] = [], 0
            for i in range(20):
                q = row(qs, i, b.dim)
                r = ix.query(q, top_k=5, nprobe=8, rerank=20, filter={"attrs": 1})
                assert r.stats["routing"] == ("cold" if mode == "cold" else "flat")
                assert r.stats["routing_table"] == ("i8" if mode == "cold" else mode)
                assert r.params["filter"] is True and r.params["max_nprobe"] == 32
                assert len(r.hits) == 5 and all(h.attrs == 1 and h.id % 4 == 1 for h in r.hits)
                assert r.stats["filter_dropped"] > 0
                truth = [j for j in brute_force(b.vectors, b.dim, q, len(b.vectors) // b.dim) if j % 4 == 1][:5]
                hits[mode] += sum(1 for h in r.hits if h.id in truth)
                got[mode].append([h.id for h in r.hits])
    finally:
        gated.release.set()
        for ix in opened.values():
            ix.close()
    assert got["f16"] == got["f32"]
    for mode in ("i8", "cold"):
        assert hits[mode] >= 0.9 * hits["f32"], (mode, hits)
