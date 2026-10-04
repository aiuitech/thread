"""Compact routing tables (fine.f16 / fine.i8): opening with `routing`, fallbacks, snapshots."""

from __future__ import annotations

import json
import threading

import pytest

from helpers import build_gen, rng, row, unit_rows
from thread_sdk import MemoryBlobStore, _native, keys, open_index
from thread_sdk.revision import load_rev_state, snapshot_keys
from thread_sdk.wire import choose_routing, manifest_files


def add_routing_rev(store, gen: str, *, tables=("f16", "i8"), stale_fine: bool = False, tree_groups: int = 0,
                    grouped_elem: str = "i8") -> int:
    """A revision naming fine.f16 / fine.i8 (and, with `tree_groups`, a centroid tree and the tree-grouped table)
    made from the generation's means (what publish-rev writes)."""
    head = json.loads(bytes(store.get(keys.current()))).get("rev", 0)
    m = json.loads(bytes(store.get(keys.manifest(gen, head))))
    files = manifest_files(m)
    fine = bytes(store.get(files["fine"]))
    rev = head + 1
    routing = {"fine": f"idx/{gen}/fine.0.f32" if stale_fine else files["fine"]}
    for elem in tables:
        b = _native.routing_table_encode(fine, m["dim"], elem)
        key = keys.routing_table(files["fine"], elem)
        store.put(key, b)
        routing[elem] = {"key": key, "bytes": len(b)}
    extra = {}
    if tree_groups:
        tree = _native.build_centroid_tree(fine, m["dim"], tree_groups, 5, 7)
        tkey = f"idx/{gen}/tree.{rev}.bin"
        store.put(tkey, tree)
        g = _native.grouped_table_encode(fine, m["dim"], tree, grouped_elem)
        _, _, _, groups, rows, data_off, row_bytes, size = _native.grouped_table_info(g)
        gkey = keys.grouped_table(gen, rev, grouped_elem)
        store.put(gkey, g)
        routing["grouped"] = {"key": gkey, "tree": tkey, "elem": grouped_elem, "groups": groups, "rows": rows,
                              "dataOff": data_off, "rowBytes": row_bytes, "bytes": size}
        extra["tree"] = {"key": tkey, "groups": tree_groups, "beam": 2}
    store.put(keys.manifest(gen, rev), json.dumps({**m, "rev": rev, "parentRev": head, "files": files,
                                                   "routing": routing, **extra}))
    store.put(keys.current(), json.dumps({"gen": gen, "rev": rev}))
    return rev


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


class Gated(MemoryBlobStore):
    """A memory store whose reads of `key` wait until `release` is set (to hold the full routing table back)."""

    def __init__(self, inner: MemoryBlobStore, key: str) -> None:
        super().__init__()
        self.objects = inner.objects
        self.key = key
        self.release = threading.Event()
        self.reads: list[str] = []

    def get(self, key, range=None, *, trace=None, fill=False):
        self.reads.append(key)
        if key == self.key:
            self.release.wait(10)
        return super().get(key, range, trace=trace, fill=fill)


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
        assert r.stats["routing"] == "flat" and "cold" not in r.stats


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
        st = first.stats["cold"]
        assert st["beam"] == 1 and st["groups"] == 1 and st["cells"] > 0
        assert st["bytes"] == st["loaded_cells"] * 4 * b.dim
        again = cold.query(q, top_k=3)
        assert again.stats["cold"]["groups"] == 0 and again.stats["cold"]["bytes"] == 0
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
        st = first.stats["cold"]
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
