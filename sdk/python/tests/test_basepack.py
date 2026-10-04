"""Base packs: one GET for a cold open, byte-identical state, index packs with a base range."""

from __future__ import annotations

from helpers import Gated, add_routing_rev, build_gen, rng, row, unit_rows
from thread_sdk import MemoryBlobStore, PackedBlobStore, keys, open_index
from thread_sdk.pack import build_base_pack, pack_entries, pack_to_bytes
from thread_sdk.revision import load_rev_state


class Counting(MemoryBlobStore):
    """A view of a memory store that records the reads reaching it."""

    def __init__(self, inner: MemoryBlobStore) -> None:
        super().__init__()
        self.objects = inner.objects
        self.reads: list[str] = []

    def get(self, key, range=None, *, trace=None, fill=False):
        self.reads.append(key if range is None else f"{key}@{range.offset}+{range.length}")
        return super().get(key, range, trace=trace, fill=fill)

    def get_ranges(self, key, ranges, *, trace=None, fill=False):
        self.reads.append(f"{key}#{len(ranges)}")
        return super().get_ranges(key, ranges, trace=trace, fill=fill)


def write_pack(store, gen: str, rev: int, kind: str) -> str:
    p = build_base_pack(store, load_rev_state(store, gen, rev), kind)
    store.put(p.key, p.data)
    return p.key


def same_answers(a, b, dim: int) -> None:
    assert b.revision.manifest == a.revision.manifest
    assert b.revision.slabs == a.revision.slabs
    assert (b.revision.tomb.sets if b.revision.tomb else None) == (a.revision.tomb.sets if a.revision.tomb else None)
    qs = unit_rows(10, dim, rng(31))
    for i in range(10):
        x, y = a.query(row(qs, i, dim), nprobe=4), b.query(row(qs, i, dim), nprobe=4)
        assert [(h.id, h.score, h.source) for h in y.hits] == [(h.id, h.score, h.source) for h in x.hits]
        assert y.stats["cells"] == x.stats["cells"]


def test_cold_open_is_current_plus_base_pack_and_state_is_identical():
    b = build_gen("bp", 3, k2=32, n=900)
    add_routing_rev(b.store, "bp", tree_groups=4)
    with open_index(b.store, cache=False) as w:
        w.delete([3, 4, 5])  # a tombstone set: rev 2
    key = write_pack(b.store, "bp", 2, "base")
    assert key == keys.base_pack("bp", 2)
    entries, meta = pack_entries(b.store.get(key))
    assert meta == {"gen": "bp", "rev": 2, "kind": "base", "routing": "i8"}
    assert list(entries) == ["idx/bp/manifest.2.json", "idx/bp/slabs.json", "idx/bp/tombstones.2.u32",
                             "idx/bp/coarse.f32", "idx/bp/pq.bin", "idx/bp/tree.1.bin", "idx/bp/fine.i8"]
    for k, v in entries.items():
        assert bytes(v) == bytes(b.store.get(k))
    c = Counting(b.store)
    with open_index(b.store, cache=False, routing="i8", base_pack=False) as plain, \
            open_index(c, cache=False, routing="i8") as fast:
        assert plain.open_stats["prefetch"]["source"] == "none"
        assert fast.open_stats["prefetch"]["source"] == "base-pack"
        assert sorted(c.reads) == ["idx/bp/base.2.threadpack", "idx/bp/manifest.2.json", "idx/current.json"]
        same_answers(plain, fast, b.dim)


def test_without_a_pack_reads_start_after_the_manifest_and_are_accounted():
    b = build_gen("bf", 3, k2=16)
    add_routing_rev(b.store, "bf")
    c = Counting(b.store)
    with open_index(c, cache=False, routing="f16") as ix, \
            open_index(b.store, cache=False, routing="f16", base_pack=False) as plain:
        assert ix.open_stats["prefetch"]["source"] == "files"
        assert set(c.reads) == {"idx/current.json", "idx/bf/base.1.threadpack", "idx/bf/manifest.1.json",
                                "idx/bf/slabs.json", "idx/bf/coarse.f32", "idx/bf/pq.bin", "idx/bf/fine.f16"}
        assert ix.open_stats["manifest"]["reads"] == 1
        assert ix.open_stats["slabs"]["reads"] == 1
        assert ix.open_stats["base"]["reads"] == 3
        same_answers(plain, ix, b.dim)


def test_cold_base_pack():
    b = build_gen("bc", 3, k2=48, n=1500)
    add_routing_rev(b.store, "bc", tree_groups=6)
    key = write_pack(b.store, "bc", 1, "cold")
    assert key == keys.base_cold_pack("bc", 1)
    entries, meta = pack_entries(b.store.get(key))
    assert meta["kind"] == "base-cold" and meta["routing"] is None
    assert list(entries) == ["idx/bc/manifest.1.json", "idx/bc/slabs.json", "idx/bc/coarse.f32", "idx/bc/pq.bin",
                             "idx/bc/tree.1.bin"]
    gated = Gated(b.store, "idx/bc/fine.f32")
    with open_index(gated, cache=False, cold_routing="tree", cold_beam=6) as cold, \
            open_index(b.store, cache=False, base_pack=False) as full:
        assert cold.open_stats["prefetch"]["source"] == "base-pack"
        assert cold.routing_state == "cold"
        assert sorted(k for k in gated.reads if k != "idx/bc/fine.f32") == [
            "idx/bc/base.cold.1.threadpack", "idx/bc/manifest.1.json", "idx/current.json"]
        r = cold.query(row(unit_rows(1, b.dim, rng(31)), 0, b.dim), nprobe=5)
        assert r.stats["routing"] == "cold" and r.stats["cold"]["reads"] == 1 and len(r.hits) == 10
        gated.release.set()
        cold.when_full_routing()
        same_answers(full, cold, b.dim)


def test_index_pack_base_range_is_one_read():
    b = build_gen("bi", 3, k2=16)
    add_routing_rev(b.store, "bi", tree_groups=4)
    data, res = pack_to_bytes(b.store)
    assert res.keys[:8] == ["idx/current.json", "idx/bi/manifest.1.json", "idx/bi/slabs.json", "idx/bi/coarse.f32",
                            "idx/bi/pq.bin", "idx/bi/tree.1.bin", "idx/bi/fine.i8", "idx/bi/fine.f16"]
    p = PackedBlobStore.from_bytes(data)
    r = p.base_range
    off, ln = p.entries["idx/bi/fine.i8"]
    assert (r.offset, r.length) == (16, off + ln - 16)
    inner = Counting(p.inner)
    pc = PackedBlobStore(inner, p.key)
    inner.reads.clear()
    with open_index(pc, cache=False, routing="auto") as ix, \
            open_index(b.store, cache=False, routing="auto", base_pack=False) as plain:
        assert ix.open_stats["prefetch"]["source"] == "index-pack"
        assert inner.reads == [f"index.threadpack@16+{r.length}"]
        same_answers(plain, ix, b.dim)
