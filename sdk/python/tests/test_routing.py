"""Compact routing tables (fine.f16 / fine.i8): opening with `routing`, fallbacks, snapshots."""

from __future__ import annotations

import json

import pytest

from helpers import build_gen, rng, row, unit_rows
from thread_sdk import _native, keys, open_index
from thread_sdk.revision import load_rev_state, snapshot_keys
from thread_sdk.wire import choose_routing, manifest_files


def add_routing_rev(store, gen: str, *, tables=("f16", "i8"), stale_fine: bool = False) -> int:
    """A revision naming fine.f16 / fine.i8 made from the generation's means (what publish-rev writes)."""
    head = json.loads(bytes(store.get(keys.current()))).get("rev", 0)
    m = json.loads(bytes(store.get(keys.manifest(gen, head))))
    files = manifest_files(m)
    fine = bytes(store.get(files["fine"]))
    routing = {"fine": f"idx/{gen}/fine.0.f32" if stale_fine else files["fine"]}
    for elem in tables:
        b = _native.routing_table_encode(fine, m["dim"], elem)
        key = keys.routing_table(files["fine"], elem)
        store.put(key, b)
        routing[elem] = {"key": key, "bytes": len(b)}
    rev = head + 1
    store.put(keys.manifest(gen, rev), json.dumps({**m, "rev": rev, "parentRev": head, "files": files,
                                                   "routing": routing}))
    store.put(keys.current(), json.dumps({"gen": gen, "rev": rev}))
    return rev


def test_encode_decode_and_wire_compat():
    dim = 16
    fine = unit_rows(20, dim, rng(3))
    for elem, step in (("f16", 1e-3), ("i8", 1 / 127)):
        b = _native.routing_table_encode(fine, dim, elem)
        assert len(b) == 20 * (2 * dim if elem == "f16" else 4 + dim)
        d = memoryview(_native.routing_table_decode(b, dim, elem)).cast("f")
        assert all(abs(d[i] - fine[i]) < step for i in range(len(fine)))
    with pytest.raises(ValueError, match="f32, f16 or i8"):
        _native.routing_table_encode(fine, dim, "f8")
    with pytest.raises(ValueError, match="expected 20 rows"):
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
