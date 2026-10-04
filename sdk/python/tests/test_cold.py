"""The cold path: miss policies (whole / range / range+warm), per-store in-flight and coalescing defaults, the cold-path
query policy, the SigV4 signing-key cache, connection warm-up, the disk cache layout and prefetch on refresh."""

from __future__ import annotations

import os
import threading
import time

import pytest

from thread_sdk import (
    BlobRange,
    BlobTrace,
    CachedBlobStore,
    DiskCache,
    FileBlobStore,
    MemoryBlobStore,
    MemoryCache,
    PackedBlobStore,
    S3BlobStore,
    SigningKeyCache,
    ThreadError,
    coalesce_runs,
    copy_objects,
    open_index,
    pack_to_bytes,
    sign_v4,
    snapshot_keys,
)
from thread_sdk.s3 import EMPTY_SHA256

from helpers import build_gen, rng, row, unit_rows

BODY = bytes((i * 7) & 0xFF for i in range(5000))


class RemoteStore(MemoryBlobStore):
    """A memory store that says it is remote (like S3; no multi-range reads) and records every read. Shares the
    objects of `src`, so writes through another handle are visible."""

    remote = True
    supports_ranges = False

    def __init__(self, src: MemoryBlobStore | None = None, delay: float = 0.0) -> None:
        super().__init__()
        if src is not None:
            self.objects = src.objects
        self.reads: list[tuple[str, BlobRange | None]] = []
        self.heads = 0
        self.delay = delay
        self._rl = threading.Lock()

    def get(self, key, range=None, *, trace=None, fill=False):
        with self._rl:
            self.reads.append((key, range))
        if self.delay:
            time.sleep(self.delay)
        return super().get(key, range, trace=trace, fill=fill)

    def get_ranges(self, key, ranges, *, trace=None, fill=False):
        return None

    def head(self, key):
        with self._rl:
            self.heads += 1
        return super().head(key)

    def warmup(self, n, executor=None):
        return [executor.submit(self.head, "idx/current.json") if executor else None for _ in range(n)]


def cached(policy: str | None = None, **kw):
    inner = RemoteStore()
    inner.put("idx/g/slab/00001", BODY)
    cache = MemoryCache()
    return inner, cache, CachedBlobStore(inner, cache, miss_policy=policy, **kw)


def test_policy_defaults_and_validation():
    assert CachedBlobStore(MemoryBlobStore(), MemoryCache()).miss_policy == "range+warm"
    assert CachedBlobStore(MemoryBlobStore(), MemoryCache(), miss_policy="whole").miss_policy == "whole"
    with pytest.raises(ThreadError, match="miss_policy"):
        CachedBlobStore(MemoryBlobStore(), MemoryCache(), miss_policy="bogus")


def test_whole_policy():
    inner, cache, s = cached("whole")
    t = BlobTrace()
    v = s.get("idx/g/slab/00001", BlobRange(100, 50), trace=t, fill=True)
    assert bytes(v) == BODY[100:150] and isinstance(v, memoryview) and len(v.obj) == 5000
    assert inner.reads == [("idx/g/slab/00001", None)]
    assert cache.contains("idx/g/slab/00001") and s.contains("idx/g/slab/00001")


def test_range_policy_caches_nothing():
    inner, cache, s = cached("range")
    t = BlobTrace()
    assert bytes(s.get("idx/g/slab/00001", BlobRange(100, 50), trace=t, fill=True)) == BODY[100:150]
    assert bytes(s.get("idx/g/slab/00001", trace=t)) == BODY
    assert s.drain(5)
    assert inner.reads == [("idx/g/slab/00001", BlobRange(100, 50)), ("idx/g/slab/00001", None)]
    assert not cache.contains("idx/g/slab/00001") and t.warms == 0


def test_range_warm_policy_warms_once_in_background():
    inner, cache, s = cached("range+warm")
    inner.delay = 0.01
    t = BlobTrace()
    out = []
    ths = [threading.Thread(target=lambda i=i: out.append(s.get("idx/g/slab/00001", BlobRange(i, 10), trace=t,
                                                                 fill=True))) for i in range(3)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    assert [len(b) for b in out] == [10, 10, 10] and t.reads == 3
    assert t.warms == 1  # deduplicated
    assert s.drain(5)
    assert cache.contains("idx/g/slab/00001")
    assert sum(1 for _, r in inner.reads if r is None) == 1
    assert s.warm_stats["fetched"] == 1 and s.warm_stats["bytes"] == 5000
    t2 = BlobTrace()
    assert bytes(s.get("idx/g/slab/00001", BlobRange(0, 4), trace=t2, fill=True)) == BODY[:4]
    assert (t2.hits, t2.reads, t2.warms) == (1, 0, 0)
    s.close()


def test_range_warm_queue_is_bounded():
    inner = RemoteStore(delay=0.01)
    for i in range(6):
        inner.put(f"idx/g/slab/0000{i}", BODY)
    s = CachedBlobStore(inner, MemoryCache(), warm_concurrency=1, warm_queue=1)
    ths = [threading.Thread(target=s.get, args=(f"idx/g/slab/0000{i}", BlobRange(0, 8)), kwargs={"fill": True})
           for i in range(6)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    assert s.drain(5)
    assert s.warm_stats["queued"] + s.warm_stats["dropped"] == 6 and s.warm_stats["dropped"] > 0
    s.close()


@pytest.mark.parametrize("policy", ["whole", "range+warm"])
def test_policies_over_a_pack_fetch_objects_not_the_pack(policy):
    b = build_gen("gpk", 3)
    pack, _ = pack_to_bytes(b.store)
    inner = RemoteStore()
    inner.put("packs/g.threadpack", pack)
    ps = PackedBlobStore(inner, "packs/g.threadpack")
    inner.reads.clear()
    cache = MemoryCache()
    store = CachedBlobStore(ps, cache, miss_policy=policy)
    with open_index(store, cache=False) as ix:
        q = row(b.vectors, 0, 16)
        r1 = ix.query(q, nprobe=8)
        assert store.drain(5)
        assert all(k == "packs/g.threadpack" and r is not None and r.length < len(pack) for k, r in inner.reads)
        assert all(k.startswith("idx/gpk/") for k in cache._items)
        assert cache.contains("idx/gpk/slab/00000")
        r2 = ix.query(q, nprobe=8)
        assert r2.hits == r1.hits and r2.stats["io"]["reads"] == 0


@pytest.mark.parametrize("policy", ["whole", "range", "range+warm"])
def test_queries_under_each_policy(policy):
    b = build_gen("gq", 3, n=800)
    qs = [row(b.vectors, i, 16) for i in (0, 5, 9, 33)]
    with open_index(b.store, cache=False) as plain:
        want = [[(h.id, h.score) for h in plain.query(q, nprobe=8, rerank=40).hits] for q in qs]
    remote = RemoteStore(b.store)
    with open_index(remote, miss_policy=policy, warmup=False) as ix:
        assert isinstance(ix.store, CachedBlobStore) and ix.store.miss_policy == policy
        r1 = ix.query(qs[0], nprobe=8, rerank=40)
        assert [(h.id, h.score) for h in r1.hits] == want[0]
        if policy == "whole":
            assert r1.stats["whole_slabs"] == 8 and r1.stats["rerank_reads"] == 0
            assert r1.stats["rerank_local"] == r1.stats["rerank_candidates"]
        else:
            assert r1.stats["whole_slabs"] == 0 and r1.stats["rerank_local"] == 0
            assert r1.stats["rerank_reads"] > 0 and r1.stats["slab_bytes"] == r1.stats["scan_bytes"]
            assert r1.stats["io"]["warms"] == (8 if policy == "range+warm" else 0)
        assert ix.store.drain(5)
        rest = [ix.query(q, nprobe=8, rerank=40) for q in qs]
        assert [[(h.id, h.score) for h in r.hits] for r in rest] == want
        if policy == "range":
            assert rest[0].stats["io"]["reads"] > 0
        else:
            assert rest[0].stats["io"]["reads"] == 0 and rest[0].stats["whole_slabs"] == 8


def test_inflight_and_coalescing_defaults():
    s3 = S3BlobStore("https://x.example", "b", "a", "s")
    assert (s3.remote, s3.inflight) == (True, 32)
    assert S3BlobStore("https://x.example", "b", "a", "s", inflight=6, coalesce_bytes=65536).coalesce_bytes == 65536
    b = build_gen("gi", 3)
    remote = RemoteStore(b.store)
    remote.inflight = 32
    remote.coalesce_bytes = 1 << 20
    with open_index(remote, warmup=False) as ix:
        r = ix.query(row(b.vectors, 0, 16), nprobe=8)
        assert (r.stats["inflight"], r.stats["coalesce_bytes"], r.stats["span_bytes"]) == (32, 1 << 20, 0)
        r2 = ix.query(row(b.vectors, 0, 16), nprobe=8, inflight=3, coalesce_bytes=4096, span_bytes=9999)
        assert (r2.stats["inflight"], r2.stats["coalesce_bytes"], r2.stats["span_bytes"]) == (3, 4096, 9999)


def test_coalesce_runs():
    runs = lambda *a: [(r.a, r.b) for r in coalesce_runs(*a)]  # noqa: E731
    assert runs([1, 2, 5, 9], 100, 400) == [(1, 2), (5, 5), (9, 9)]
    assert runs([1, 2, 5, 9], 100, 500) == [(1, 5), (9, 9)]
    assert runs([1, 2, 5, 9], 100, 100, 900) == [(1, 9)]
    assert len(coalesce_runs([1, 2, 5, 9], 100, 100, 800)) == 4
    assert coalesce_runs([], 100, 100, 800) == []


def test_larger_coalescing_fewer_requests_same_answers():
    b = build_gen("gco", 3, n=1200)
    qs = [row(b.vectors, i, 16) for i in (3, 17, 40)]
    res = {}
    for name, o in {"c4k": {}, "c128k": {"coalesce_bytes": 128 << 10}, "span": {"span_bytes": 1 << 20}}.items():
        with open_index(RemoteStore(b.store), cache=False, warmup=False) as ix:
            rs = [ix.query(q, nprobe=8, rerank=60, speculate=False, **o) for q in qs]
            res[name] = (sum(r.stats["rerank_reads"] for r in rs), sum(r.stats["rerank_bytes"] for r in rs),
                         [[h.id for h in r.hits] for r in rs])
    assert res["c128k"][2] == res["c4k"][2] == res["span"][2]
    assert res["c128k"][0] < res["c4k"][0]
    assert res["span"][0] <= res["c128k"][0] and res["span"][1] >= res["c4k"][1]


def test_get_tail_through_a_cache():
    inner = RemoteStore()
    body = bytes((i * 7) & 0xFF for i in range(5000))
    inner.put("idx/g/obj", body)
    cs = CachedBlobStore(inner, MemoryCache())
    t1 = BlobTrace()
    tail, head = cs.get_tail("idx/g/obj", 100, trace=t1)
    assert bytes(tail) == body[4900:] and head.size == 5000 and (t1.reads, t1.hits) == (1, 0)
    assert not cs.contains("idx/g/obj")  # a tail is not cached
    cs.get("idx/g/obj")
    t2 = BlobTrace()
    tail, head = cs.get_tail("idx/g/obj", 100, trace=t2)
    assert bytes(tail) == body[4900:] and head.size == 5000 and (t2.reads, t2.hits) == (0, 1)
    assert cs.get_tail("idx/g/none", 10) is None


def test_cold_policy():
    b = build_gen("gcp", 3, n=800)
    q = row(b.vectors, 0, 16)
    with open_index(RemoteStore(b.store), warmup=False, cold_policy={"nprobe": 3, "rerank": 10}) as ix:
        cold = ix.query(q, nprobe=8, rerank=40)
        assert cold.stats["cold"] == {"misses": 8, "probed": 8, "applied": True}
        assert len(cold.stats["cells"]) == 3 and cold.params["nprobe"] == 3 and cold.params["rerank"] == 10
        off = ix.query(q, nprobe=8, rerank=40, cold_policy=False)
        assert "cold" not in off.stats and len(off.stats["cells"]) == 8
        assert ix.store.drain(5)
        warm = ix.query(q, nprobe=8, rerank=40)
        assert warm.stats["cold"] == {"misses": 0, "probed": 8, "applied": False}
        assert len(warm.stats["cells"]) == 8 and warm.params["rerank"] == 40
        never = ix.query(row(b.vectors, 5, 16), nprobe=8, cold_policy={"nprobe": 2, "threshold": 1})
        assert never.stats["cold"]["applied"] is False
    with open_index(b.store, cache=False, cold_policy={"nprobe": 2}) as local:
        assert local.query(q, nprobe=8).stats["cold"] == {"misses": 0, "probed": 8, "applied": False}


def test_cold_policy_with_filter_widens_from_the_reduced_nprobe():
    b = build_gen("gcpf", 3, n=800, attrs=lambda i: i % 4)
    q = row(b.vectors, 0, 16)
    with open_index(RemoteStore(b.store), warmup=False, cold_policy={"nprobe": 2, "rerank": 10}) as ix:
        r = ix.query(q, nprobe=6, rerank=40, filter={"attrs": 1})
        assert r.stats["cold"] == {"misses": 6, "probed": 6, "applied": True}
        assert (r.params["nprobe"], r.params["rerank"], r.params["max_nprobe"]) == (2, 10, 8)
        assert len(r.stats["cells"]) <= 8 and all(h.attrs == 1 for h in r.hits)
    with open_index(RemoteStore(b.store), warmup=False, cold_policy={"nprobe": 2}) as fresh:
        capped = fresh.query(q, nprobe=6, rerank=40, filter={"attrs": 1}, max_nprobe=6)
        assert (capped.params["nprobe"], capped.params["max_nprobe"]) == (2, 6) and len(capped.stats["cells"]) <= 6


CREDS = {"access_key_id": "AKIAIOSFODNN7EXAMPLE", "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"}


def test_signing_key_cache_reproduces_aws_vectors():
    keys = SigningKeyCache(CREDS["secret_access_key"])
    for _ in range(3):
        h = sign_v4(method="GET", path="/test.txt", headers={"range": "bytes=0-9"}, payload_hash=EMPTY_SHA256,
                    host="examplebucket.s3.amazonaws.com", region="us-east-1", amz_date="20130524T000000Z", keys=keys,
                    **CREDS)
        assert "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41" in h["authorization"]
    h = sign_v4(method="GET", path="/", query={"max-keys": "2", "prefix": "J"}, headers={}, payload_hash=EMPTY_SHA256,
                host="examplebucket.s3.amazonaws.com", region="us-east-1", amz_date="20130524T000000Z", keys=keys,
                **CREDS)
    assert "Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7" in h["authorization"]
    assert keys.derived == 1
    sign_v4(method="GET", path="/x", headers={}, payload_hash=EMPTY_SHA256, host="h", region="us-east-1",
            amz_date="20130525T000000Z", keys=keys, **CREDS)
    assert keys.derived == 2


def test_warmup_on_open():
    b = build_gen("gw", 3)
    r1 = RemoteStore(b.store)
    r1.inflight = 5
    with open_index(r1):
        pass
    assert r1.heads == 5
    r2 = RemoteStore(b.store)
    with open_index(r2, warmup=False):
        pass
    assert r2.heads == 0
    r3 = RemoteStore(b.store)
    with open_index(r3, warmup=2):
        pass
    assert r3.heads == 2


def test_disk_cache_layout_lru_restart(tmp_path):
    d = tmp_path / "dc"
    dc = DiskCache(d, max_bytes=12000)
    dc.put("idx/g/slab/00001", BODY)
    assert (d / "idx" / "g" / "slab" / "00001").exists() and dc.contains("idx/g/slab/00001")
    assert [bytes(p) for p in dc.get_ranges("idx/g/slab/00001", [BlobRange(10, 5)])] == [BODY[10:15]]
    dc.put("idx/g/slab/00002", BODY)
    dc.get("idx/g/slab/00001")
    dc.put("idx/g/slab/00003", BODY)
    assert dc.held <= 12000 and not dc.contains("idx/g/slab/00002")
    assert not (d / "idx" / "g" / "slab" / "00002").exists()
    again = DiskCache(d, max_bytes=12000)
    assert again.held == dc.held and again.contains("idx/g/slab/00003")
    assert FileBlobStore(d).get("idx/g/slab/00003") == BODY
    os.unlink(d / "idx" / "g" / "slab" / "00003")
    assert again.get("idx/g/slab/00003") is None and not again.contains("idx/g/slab/00003")


def test_disk_cache_restart_and_prewarm(tmp_path):
    b = build_gen("gdc", 3, n=800)
    remote = RemoteStore(b.store)
    q = row(b.vectors, 0, 16)
    with open_index(remote, cache=DiskCache(tmp_path / "a"), warmup=False) as ix1:
        r1 = ix1.query(q, nprobe=8)
        assert ix1.store.drain(5)
    with open_index(remote, cache=DiskCache(tmp_path / "a"), warmup=False) as ix2:
        r2 = ix2.query(q, nprobe=8)
        assert r2.hits == r1.hits and r2.stats["io"]["reads"] == 0
        snap = snapshot_keys(ix2.revision)
    dc = DiskCache(tmp_path / "b")
    copied, _, _ = copy_objects(remote, dc, snap)
    assert copied == len(snap)
    before = len(remote.reads)
    with open_index(remote, cache=DiskCache(tmp_path / "b"), warmup=False) as ix3:
        assert ix3.query(q, nprobe=8).hits == r1.hits
    assert [k for k, _ in remote.reads[before:]] == ["idx/current.json"]
    with open_index(FileBlobStore(tmp_path / "b"), cache=False, gen="gdc", rev=0) as from_dir:
        assert from_dir.query(q, nprobe=8).hits == r1.hits


def test_background_refresh_prefetches_before_switching():
    b = build_gen("grf", 3, n=800)
    remote = RemoteStore(b.store)
    with open_index(b.store, cache=False) as writer, open_index(remote, refresh_interval=0.001, warmup=False) as ix:
        q = row(b.vectors, 0, 16)
        ix.query(q)
        fresh = unit_rows(2, b.dim, rng(5))
        up = writer.upsert([7000, 7001], fresh)
        time.sleep(0.01)
        n0 = len(remote.reads)
        during = ix.query(row(fresh, 0, 16), top_k=1)  # triggers the check, does not wait for it
        assert during.stats["rev"] == 0
        for _ in range(200):
            if ix.rev == up.rev:
                break
            time.sleep(0.005)
        assert ix.rev == up.rev
        new = [k for k, _ in remote.reads[n0:]]
        assert up.keys["delta"] in new
        assert not any(k.endswith("coarse.f32") for k in new)
        assert ix.store.drain(5)
        assert ix.query(row(fresh, 0, 16), top_k=1).hits[0].id == 7000
        w = ix.warm(budget=1 << 30)
        assert w["objects"] + w["skipped"] == len(snapshot_keys(ix.revision))
        assert ix.store.drain(5)
        assert all(ix.store.contains(k) for k in snapshot_keys(ix.revision))
