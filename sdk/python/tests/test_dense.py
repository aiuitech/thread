"""End to end: builds tiny v1/v2/v3 generations with the native build jobs, then queries them through the SDK
(brute-force recall), upserts/deletes with tombstones, refreshes, and checks concurrency, caches and stores."""

from __future__ import annotations

import json
import threading
from array import array
from concurrent.futures import ThreadPoolExecutor

import pytest

from thread_sdk import (
    DiskCache,
    FileBlobStore,
    MemoryBlobStore,
    MemoryCache,
    ThreadError,
    keys,
    open_index,
)

from helpers import Built, brute_force, build_gen, rng, row, unit_rows


def recall(index, b: Built, nq: int, k: int, **qopts) -> float:
    r = rng(7)
    hit = 0
    for _ in range(nq):
        q = unit_rows(1, b.dim, r)
        truth = set(brute_force(b.vectors, b.dim, q, k))
        res = index.query(q, top_k=k, **qopts)
        assert len(res.hits) == k
        assert len({h.id for h in res.hits}) == k
        assert all(res.hits[j].score <= res.hits[j - 1].score for j in range(1, k))
        hit += sum(1 for h in res.hits if h.id in truth)
    return hit / (nq * k)


def test_v3_defaults_params_stats_recall():
    b = build_gen("g3", 3)
    index = open_index(b.store, cache=False)
    assert (index.gen, index.rev, index.dim, index.cells, index.format) == ("g3", 0, 16, 8, 3)
    res = index.query(row(b.vectors, 0, 16))
    assert res.hits[0].id == 0
    assert res.hits[0].source == "slab"
    assert res.params == {"top_k": 10, "nprobe": 3, "rerank": 20, "pc": 0,
                          "source": {"nprobe": "manifest", "rerank": "manifest", "pc": "manifest"}}
    assert len(res.stats["cells"]) == 3
    assert res.stats["rerank"] == 20
    assert res.stats["rerank_slabs"] > 0
    assert res.stats["scanned"] > 0
    assert res.stats["io"]["reads"] > 0
    assert recall(index, b, 20, 10, nprobe=8, rerank=40) >= 0.9
    # rerank = 0: scan-tier scores only (PQ), lower but sane
    r0 = index.query(row(b.vectors, 1, 16), rerank=0, nprobe=8)
    assert r0.stats["rerank"] == 0
    assert r0.params["source"]["rerank"] == "query"
    assert 1 in [h.id for h in r0.hits]
    # return_payload on slab hits without payloads yields empty bytes
    rp = index.query(row(b.vectors, 0, 16), return_payload=True, top_k=3)
    assert all(h.payload == b"" for h in rp.hits)
    with pytest.raises(ThreadError, match="dim 5"):
        index.query([0.0] * 5)
    with pytest.raises(ThreadError, match="top_k"):
        index.query(row(b.vectors, 0, 16), top_k=0)
    index.close()
    with pytest.raises(ThreadError, match="closed"):
        index.query(row(b.vectors, 0, 16))


def test_v2_and_query_many():
    b = build_gen("g2", 2)
    with open_index(b.store, cache=False) as index:
        assert index.format == 2
        assert recall(index, b, 20, 10, nprobe=8) >= 0.95
        many = index.query_many([row(b.vectors, 0, 16), row(b.vectors, 2, 16)], top_k=1, nprobe=8)
        assert [r.hits[0].id for r in many] == [0, 2]


def test_v1_legacy_generation():
    """No slabs table, no rev in current.json: slabs read whole, no phase 2."""
    b = build_gen("g1", 1, legacy=True)
    with open_index(b.store, cache=False) as index:
        assert index.format == 1
        assert index.revision.slabs is None
        res = index.query(row(b.vectors, 3, 16), nprobe=8)
        assert res.hits[0].id == 3
        assert res.stats["rerank"] == 0
        assert res.stats["slab_bytes"] >= res.stats["scan_bytes"]
        assert recall(index, b, 20, 10, nprobe=8) >= 0.95


def test_vector_input_types():
    b = build_gen("gt", 3)
    with open_index(b.store, cache=False) as index:
        q = row(b.vectors, 5, 16)
        want = [h.id for h in index.query(q, nprobe=8).hits]
        assert [h.id for h in index.query(list(q), nprobe=8).hits] == want
        assert [h.id for h in index.query(q.tobytes(), nprobe=8).hits] == want
        assert [h.id for h in index.query(memoryview(q), nprobe=8).hits] == want
        assert [h.id for h in index.query(array("d", q), nprobe=8).hits] == want
        np = pytest.importorskip("numpy")
        assert [h.id for h in index.query(np.asarray(q, dtype=np.float32), nprobe=8).hits] == want
        assert [h.id for h in index.query(np.asarray(q, dtype=np.float64) * 3, nprobe=8).hits] == want


def test_upsert_delete_replace_refresh():
    b = build_gen("gu", 3)
    a = open_index(b.store, cache=False)
    other = open_index(b.store, cache=False, refresh_interval=0)
    fresh = unit_rows(2, b.dim, rng(99))
    up = a.upsert([1000, 1001], fresh, attrs=[5, 6], payloads=["p-1000", b"p-1001"])
    assert (up.rev, up.seq, up.count, len(up.cells)) == (1, 1, 2, 2)
    assert up.keys["delta"] == "idx/gu/delta/1.vibd"
    assert up.keys["tombstones"] == "idx/gu/tombstones.1.u32"
    assert up.current_updated
    assert a.rev == 1
    assert len(a.manifest["deltas"]) == 1 and len(a.manifest["tombstones"]) == 1
    assert json.loads(b.store.get(keys.current())) == {"gen": "gu", "rev": 1}
    q1 = a.query(row(fresh, 0, 16), top_k=3, return_payload=True)
    h0 = q1.hits[0]
    assert (h0.id, h0.source, h0.attrs, h0.payload) == (1000, "delta", 5, b"p-1000")
    assert q1.stats["deltas"] == 1 and q1.stats["delta_scanned"] == 2 and q1.stats["delta_top"] >= 1
    # the other handle still reads rev 0 until refreshed
    assert other.rev == 0
    assert other.query(row(fresh, 0, 16), top_k=3).hits[0].id != 1000
    assert other.refresh() is True
    assert other.rev == 1
    assert other.query(row(fresh, 0, 16), top_k=3).hits[0].id == 1000
    assert other.refresh() is False

    # replace id 3 (in rev-0 slabs) with a vector equal to fresh[1]: the slab copy is tombstoned
    old3 = row(b.vectors, 3, 16)
    assert a.query(old3, top_k=1, nprobe=8).hits[0].id == 3
    up2 = a.upsert([3], [list(row(fresh, 1, 16))])
    assert (up2.rev, up2.seq) == (2, 2)
    q3 = a.query(old3, top_k=5, nprobe=8)
    assert 3 not in [h.id for h in q3.hits]
    assert q3.stats["tomb_dropped"] >= 1
    q4 = a.query(row(fresh, 1, 16), top_k=3)
    assert sorted(h.id for h in q4.hits[:2]) == [3, 1001]
    assert q4.hits[0].source == "delta"

    # delete 1000: a tombstone set newer than the delta hides its entry
    d = a.delete([1000])
    assert (d.rev, d.count) == (3, 1)
    q5 = a.query(row(fresh, 0, 16), top_k=3)
    assert 1000 not in [h.id for h in q5.hits]
    assert q5.stats["delta_dropped"] == 1
    assert q5.stats["tomb_sets"] == 3
    c = open_index(b.store, cache=False)
    assert c.rev == 3
    assert c.manifest["nextSeq"] == 3
    assert c.manifest["change"]["op"] == "delete" and c.manifest["change"]["count"] == 1
    # pinned open of rev 0 ignores everything
    pinned = open_index(b.store, cache=False, gen="gu", rev=0)
    assert pinned.query(old3, top_k=1, nprobe=8).hits[0].id == 3
    assert pinned.refresh() is False
    for ix in (a, other, c, pinned):
        ix.close()


def test_upsert_input_validation():
    b = build_gen("gv", 2)
    with open_index(b.store, cache=False) as ix:
        v = row(b.vectors, 0, 16)
        with pytest.raises(ThreadError, match="duplicate"):
            ix.upsert([1, 1], [v, v])
        with pytest.raises(ThreadError, match="zero or non-finite"):
            ix.upsert([1], [[0.0] * 16])
        with pytest.raises(ThreadError, match="dims"):
            ix.upsert([1], [[1.0] * 3])
        with pytest.raises(ThreadError, match="u32"):
            ix.upsert([-1], [v])
        with pytest.raises(ThreadError, match="lexical"):
            ix.upsert([1], [v], texts=["x"])
        with pytest.raises(ThreadError, match="floats"):
            ix.upsert([1, 2], v)
        # flat floats, rows, raw f32 bytes and numpy are all accepted
        assert ix.upsert([7], v.tobytes()).count == 1
        np = pytest.importorskip("numpy")
        r = ix.upsert([8, 9], np.stack([np.asarray(v), np.asarray(v) * 2]))
        assert r.count == 2 and len(set(r.cells)) == 1


def test_write_conflict_retried_on_new_head():
    b = build_gen("gc", 2)
    a = open_index(b.store, cache=False)
    # Another writer took rev 1 behind our back (current.json still says rev 0, as after a crash before the CAS).
    head = json.loads(b.store.get(keys.manifest("gc", 0)))
    b.store.put(keys.manifest("gc", 1), json.dumps({**head, "rev": 1, "parentRev": 0, "deltas": [], "tombstones": [],
                                                    "nextSeq": 5}))
    d = a.delete([1, 2])
    assert d.conflicts == 1
    assert d.rev == 2
    assert a.manifest["nextSeq"] == 5
    assert a.manifest["parentRev"] == 1
    a.close()


def test_concurrent_queries_and_writes():
    """Many threads query while another upserts; every result is well formed and the final state sees the upserts."""
    b = build_gen("gx", 3, n=600)
    index = open_index(b.store, cache=False, io_workers=8)
    want = {i: index.query(row(b.vectors, i, 16), top_k=1, nprobe=8).hits[0].id for i in range(40)}
    fresh = unit_rows(20, b.dim, rng(5))

    def q(i: int) -> int:
        return index.query(row(b.vectors, i % 40, 16), top_k=5, nprobe=8).hits[0].id

    with ThreadPoolExecutor(16) as ex:
        futs = [ex.submit(q, i) for i in range(400)]
        for j in range(5):
            index.upsert([5000 + 4 * j + t for t in range(4)], fresh[j * 64 : (j + 1) * 64])
        got = [f.result() for f in futs]
    assert all(g == want[i % 40] for i, g in enumerate(got))
    assert index.rev == 5
    assert index.query(row(fresh, 19, 16), top_k=1).hits[0].id == 5019
    index.close()


@pytest.mark.parametrize("policy", ["whole", "range+warm"])
@pytest.mark.parametrize("kind", ["memory", "disk"])
def test_cache_in_front(tmp_path, kind, policy):
    """A cache serves the second query; a "whole" fill read keeps the local rerank records (no phase-2 reads)."""
    b = build_gen("gk", 3)
    cache = MemoryCache() if kind == "memory" else DiskCache(tmp_path / "cache")
    with open_index(b.store, cache=cache, miss_policy=policy) as ix:
        q = row(b.vectors, 9, 16)
        r1 = ix.query(q, nprobe=8)
        assert r1.stats["io"]["misses"] > 0
        assert ix.store.drain(5)
        r2 = ix.query(q, nprobe=8)
        assert [h.id for h in r2.hits] == [h.id for h in r1.hits]
        assert [h.score for h in r2.hits] == [h.score for h in r1.hits]
        assert r2.stats["io"]["misses"] == 0 and r2.stats["io"]["reads"] == 0
        if kind == "memory" and policy == "whole":
            assert r1.stats["rerank_local"] == r1.stats["rerank_candidates"]
            assert r1.stats["rerank_reads"] == 0
        if policy == "range+warm":
            assert r1.stats["whole_slabs"] == 0 and r1.stats["rerank_local"] == 0
            assert r1.stats["io"]["warms"] == 8
            assert r2.stats["whole_slabs"] == (8 if kind == "memory" else 0)
    with open_index(b.store, cache=False) as plain:
        r3 = plain.query(q, nprobe=8)
        assert [(h.id, h.score) for h in r3.hits] == [(h.id, h.score) for h in r1.hits]


def test_file_store_round_trip(tmp_path):
    """The same generation on disk: identical answers, and writes land as files."""
    mem = build_gen("gf", 3)
    fs = FileBlobStore(tmp_path / "bucket")
    for k in mem.store.list(""):
        fs.put(k, mem.store.get(k))
    with open_index(mem.store, cache=False) as a, open_index(fs) as f:
        for i in (0, 11, 77):
            q = row(mem.vectors, i, 16)
            assert [(h.id, h.score) for h in a.query(q).hits] == [(h.id, h.score) for h in f.query(q).hits]
        up = f.upsert([4242], row(mem.vectors, 11, 16))
        assert (tmp_path / "bucket" / "idx" / "gf" / "delta" / "1.vibd").exists()
        assert json.loads((tmp_path / "bucket" / "idx" / "current.json").read_text()) == {"gen": "gf", "rev": up.rev}
        assert {h.id for h in f.query(row(mem.vectors, 11, 16), top_k=2).hits} == {11, 4242}
    assert fs.list("idx/gf/manifest.") == ["idx/gf/manifest.1.json", "idx/gf/manifest.json"]


def test_store_without_list_and_missing_index():
    with pytest.raises(ThreadError, match="current.json missing"):
        open_index(MemoryBlobStore())


class RemoteLike(MemoryBlobStore):
    """A memory store that claims to be remote (the parallel read path, speculation) with a little latency, with or
    without multi-range reads."""

    remote = True

    def __init__(self, inner: MemoryBlobStore, ranges: bool) -> None:
        super().__init__()
        self.objects = inner.objects
        self.supports_ranges = ranges

    def get(self, key, range=None, *, trace=None, fill=False):
        threading.Event().wait(0.001)
        return super().get(key, range, trace=trace, fill=fill)

    def get_ranges(self, key, ranges, *, trace=None, fill=False):
        return super().get_ranges(key, ranges, trace=trace, fill=fill) if self.supports_ranges else None


@pytest.mark.parametrize("ranges", [True, False])
@pytest.mark.parametrize("cache", [False, True])
def test_parallel_path_matches_inline(ranges, cache):
    """The remote (parallel, speculative) path returns exactly what the inline path returns, under concurrency, with
    deltas and tombstones in play."""
    b = build_gen("gp", 3, n=800)
    with open_index(b.store, cache=False) as w:
        fresh = unit_rows(6, b.dim, rng(3))
        w.upsert([9000 + i for i in range(6)], fresh, payloads=[f"x{i}" for i in range(6)], attrs=list(range(6)))
        w.delete([1, 2, 3])
    remote = RemoteLike(b.store, ranges)
    with open_index(b.store, cache=False) as inline, \
            open_index(remote, cache=MemoryCache() if cache else False, miss_policy="whole") as par:
        qs = [row(b.vectors, i, 16) for i in range(0, 60, 3)] + [row(fresh, i, 16) for i in range(6)]
        opts = dict(top_k=10, nprobe=8, rerank=40, return_payload=True)
        want = [[(h.id, h.score, h.attrs, h.source, h.payload) for h in inline.query(q, **opts).hits] for q in qs]
        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(lambda q: par.query(q, **opts), qs * 2))
        for r, w_ in zip(got, want * 2):
            assert [(h.id, h.score, h.attrs, h.source, h.payload) for h in r.hits] == w_
        if not cache:  # speculative reads of the running top-T while the last slabs scan
            assert any(r.stats["spec_used"] > 0 for r in got)
        else:  # fill reads keep every winner's record in memory: no phase-2 reads
            assert all(r.stats["rerank_local"] == r.stats["rerank_candidates"] for r in got[len(qs):])
