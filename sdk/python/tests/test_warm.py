"""Warm packs: the partial-entry directory format (writer, parser, PackedBlobStore's view of a partial entry), building
one from a generation, the prefix layer, and `open_index(..., warm_pack=)` seeding it in the background so queries read
the hot prefixes from memory with the same answers (also behind the default memory cache)."""

from __future__ import annotations

import io
import json
import threading

import pytest

from thread_sdk import (
    BlobRange,
    CachedBlobStore,
    MemoryCache,
    BlobTrace,
    MemoryBlobStore,
    PackedBlobStore,
    PackWriter,
    ThreadError,
    WarmPrefixStore,
    build_warm_pack,
    keys,
    open_index,
    parse_warm_pack,
    seed_warm_pack,
)
from thread_sdk.pack import PACK_FOOTER_BYTES, parse_pack_directory, parse_pack_directory_full, parse_pack_footer
from thread_sdk.wire import slab_rev_of

from helpers import build_gen, rng, unit_rows


def data(n: int, seed: int = 0) -> bytes:
    return bytes((i * 31 + seed) & 0xFF for i in range(n))


class Counting(MemoryBlobStore):
    """Records every read reaching it; `remote` so `open_index` puts its default memory cache in front."""

    def __init__(self, remote: bool = False) -> None:
        super().__init__()
        self.remote = remote
        self.reads: list[tuple[str, object]] = []
        self.lock = threading.Lock()

    def get(self, key, range=None, *, trace=None, fill=False):
        with self.lock:
            self.reads.append((key, range))
        return super().get(key, range, trace=trace, fill=fill)

    def get_ranges(self, key, ranges, *, trace=None, fill=False):
        with self.lock:
            self.reads.append((key, list(ranges)))
        return super().get_ranges(key, ranges, trace=trace, fill=fill)


def test_partial_entries_in_the_directory():
    buf = io.BytesIO()
    w = PackWriter(buf, {"kind": "warm", "gen": "g", "rev": 1})
    w.add("idx/g/slab/00003.1", data(100, 3), partial_of=1000)
    w.add("whole", data(10, 1))
    with pytest.raises(ThreadError, match="more than its object"):
        w.add("bad", data(10), partial_of=5)
    w.finish()
    pack = buf.getvalue()
    dir_off, dir_len, n = parse_pack_footer(pack[-PACK_FOOTER_BYTES:])
    j = json.loads(pack[dir_off:dir_off + dir_len])
    assert j["entries"][0] == ["idx/g/slab/00003.1", 16, 100, {"partial": True, "size": 1000}]
    assert len(j["entries"][1]) == 3
    # byte-identical to what the TypeScript writer emits (JSON.stringify, no spaces)
    assert b'["idx/g/slab/00003.1",16,100,{"partial":true,"size":1000}]' in pack
    entries, meta, partial = parse_pack_directory_full(pack[dir_off:dir_off + dir_len], dir_off, n)
    assert entries["idx/g/slab/00003.1"] == (16, 100) and partial == {"idx/g/slab/00003.1": 1000}
    assert parse_pack_directory(pack[dir_off:dir_off + dir_len], dir_off, n)[0] == entries

    s = PackedBlobStore.from_bytes(pack)
    assert bytes(s.get("idx/g/slab/00003.1", BlobRange(10, 90))) == data(100, 3)[10:]
    with pytest.raises(ThreadError, match="only its first 100 of 1000 bytes"):
        s.get("idx/g/slab/00003.1")
    with pytest.raises(ThreadError, match="only its first 100"):
        s.get("idx/g/slab/00003.1", BlobRange(90, 20))
    assert s.head("idx/g/slab/00003.1").size == 1000
    assert bytes(s.get("whole")) == data(10, 1)

    meta, ents = parse_warm_pack(pack)
    assert meta["kind"] == "warm"
    assert [(e.key, len(e.data), e.size) for e in ents] == [("idx/g/slab/00003.1", 100, 1000), ("whole", 10, 10)]


def test_bad_markers_and_non_warm_packs():
    d = json.dumps({"entries": [["a", 16, 8, {"partial": True, "size": 4}]]}).encode()
    with pytest.raises(ThreadError, match="size >= its length"):
        parse_pack_directory_full(d, 24, 1)
    e, _, p = parse_pack_directory_full(json.dumps({"entries": [["a", 16, 8, {"x": 1}]]}).encode(), 24, 1)
    assert e == {"a": (16, 8)} and p == {}
    buf = io.BytesIO()
    w = PackWriter(buf, {"gen": "g", "rev": 1})
    w.add("a", data(8))
    w.finish()
    with pytest.raises(ThreadError, match="not a warm pack"):
        parse_warm_pack(buf.getvalue())
    with pytest.raises(ThreadError, match="not a thread pack"):
        parse_warm_pack(data(64))


def test_prefix_layer():
    m = Counting()
    m.put("k", data(1000, 7))
    layer = WarmPrefixStore(m)
    layer.seed_prefix("k", memoryview(data(1000, 7))[:100])
    trace = BlobTrace()
    assert bytes(layer.get("k", BlobRange(0, 100), trace=trace)) == data(1000, 7)[:100]
    got = layer.get_ranges("k", [BlobRange(0, 4), BlobRange(50, 4)])
    assert [bytes(b) for b in got] == [data(1000, 7)[:4], data(1000, 7)[50:54]]
    assert m.reads == []
    assert trace.as_dict()["hits"] == 1 and trace.as_dict()["bytes"] == 100 and trace.as_dict()["reads"] == 0
    assert bytes(layer.get("k", BlobRange(90, 20))) == data(1000, 7)[90:110]
    assert bytes(layer.get("k")) == data(1000, 7)
    layer.get_ranges("k", [BlobRange(0, 4), BlobRange(500, 4)])
    assert [k for k, _ in m.reads] == ["k", "k", "k"]
    assert layer.hits == 2
    layer.retain(lambda k: False)
    assert layer.size == (0, 0)


def test_prefix_layer_fills_the_cache_behind_it():
    m = Counting()
    m.put("idx/g/slab/00001", data(1000, 5))
    cache = MemoryCache()
    layer = WarmPrefixStore(CachedBlobStore(m, cache))
    layer.seed_prefix("idx/g/slab/00001", data(1000, 5)[:100])
    assert bytes(layer.get("idx/g/slab/00001", BlobRange(0, 100), fill=True)) == data(1000, 5)[:100]
    layer.get("idx/g/slab/00001", BlobRange(0, 100), fill=True)
    layer.get("idx/g/slab/00001", BlobRange(0, 50))  # no fill asked
    layer.drain()
    assert layer.fills == {"queued": 1, "done": 1, "failed": 0}
    assert m.reads == [("idx/g/slab/00001", None)]
    trace = BlobTrace()
    assert bytes(layer.get("idx/g/slab/00001", BlobRange(500, 10), trace=trace)) == data(1000, 5)[500:510]
    assert trace.as_dict()["reads"] == 0 and trace.as_dict()["hits"] == 1


@pytest.mark.parametrize("remote", [False, True])
def test_warm_pack_end_to_end(remote):
    store = Counting(remote)
    b = build_gen("w3", 3, k2=8, n=600, store=store)
    with open_index(b.store, cache=False) as plain:
        rev = plain.revision
        hot = [5, 1, 6, 2]
        wp, result = build_warm_pack(b.store, rev, hot, meta={"source": "test"})
        assert result.keys == [keys.slab("w3", c, slab_rev_of(rev.slabs[c])) for c in hot]
        meta, ents = parse_warm_pack(wp)
        assert {k: meta[k] for k in ("kind", "gen", "rev", "cells", "source")} == \
            {"kind": "warm", "gen": "w3", "rev": 0, "cells": 4, "source": "test"}
        for c, e in zip(hot, ents):
            assert len(e.data) == rev.slabs[c][0] and e.size == rev.slabs[c][3]
            assert bytes(e.data) == bytes(b.store.get(e.key))[:rev.slabs[c][0]]
        # plus a stale entry (another slab revision of a cell): not seeded
        buf = io.BytesIO()
        w = PackWriter(buf, {"kind": "warm", "gen": "w3", "rev": 0})
        for e in ents:
            w.add(e.key, e.data, partial_of=e.size)
        w.add(keys.slab("w3", 0, 99), data(64), partial_of=1000)
        w.finish()
        b.store.put("warm/w3.threadpack", buf.getvalue())

        # remote: the default memory cache sits behind the warm layer (cache=None)
        with open_index(b.store, cache=None if remote else False, warm_pack="warm/w3.threadpack") as index:
            st = index.warm_ready.result(timeout=10)
            assert st["state"] == "ready" and (st["entries"], st["seeded"], st["skipped"]) == (5, 4, 1)
            assert st["seeded_bytes"] == sum(rev.slabs[c][0] for c in hot)
            assert index.warm.size[0] == 4
            hot_keys = {keys.slab("w3", c, slab_rev_of(rev.slabs[c])) for c in hot}
            qs = unit_rows(6, b.dim, rng(9))
            warm_hits = 0
            for i in range(6):
                q = qs[i * b.dim:(i + 1) * b.dim]
                a = plain.query(q, nprobe=4, rerank=30)
                store.reads = []
                r = index.query(q, nprobe=4, rerank=30)
                assert [h.id for h in r.hits] == [h.id for h in a.hits]
                assert [h.score for h in r.hits] == pytest.approx([h.score for h in a.hits])
                hot_probed = [c for c in r.stats["cells"] if c in hot]
                warm_hits += len(hot_probed)
                assert r.stats["timeline"]["phase1"]["hits"] >= len(hot_probed)
                # no read of a hot cell's prefix reached the store
                assert not [k for k, rg in store.reads if k in hot_keys and isinstance(rg, BlobRange) and rg.offset == 0]
            assert warm_hits > 0


def test_missing_or_broken_warm_pack_never_fails():
    b = build_gen("w4", 3, k2=8, n=300)
    q = unit_rows(1, b.dim, rng(3))
    with open_index(b.store, cache=False, warm_pack="warm/nope.threadpack") as index:
        assert index.query(q).hits
        st = index.warm_ready.result(timeout=10)
        assert st["state"] == "failed" and "no warm pack" in st["error"]
    b.store.put("warm/bad.threadpack", data(200))
    with open_index(b.store, cache=False, warm_pack="warm/bad.threadpack") as index:
        assert index.warm_ready.result(timeout=10)["state"] == "failed"
        assert index.query(q).hits
    with open_index(b.store, cache=False) as index:
        assert index.warm is None and index.warm_ready is None and index.warm_stats is None


def test_seed_warm_pack_filters():
    buf = io.BytesIO()
    w = PackWriter(buf, {"kind": "warm", "gen": "g", "rev": 1})
    w.add("a", data(8, 1), partial_of=80)
    w.add("b", data(8, 2), partial_of=80)
    w.finish()
    seen = []

    class Seed:
        def seed_prefix(self, key, d):
            seen.append((key, len(d)))

    r = seed_warm_pack(buf.getvalue(), Seed(), lambda k: k == "b")
    assert seen == [("b", 8)]
    assert (r["entries"], r["seeded"], r["skipped"], r["seeded_bytes"]) == (2, 1, 1, 8)
