"""Packs: the file format (writer, footer, directory), PackedBlobStore over memory and over a file, a snapshot of a
built generation with deltas, tombstones and lexical segments queried through the pack with the same answers as the
source store, a cache in front of a pack, read-only writes, copying a pack back out, and `open_index(path)`."""

from __future__ import annotations

import io
import json
import os

import pytest

from thread_sdk import (
    BlobRange,
    CachedBlobStore,
    FileBlobStore,
    MemoryBlobStore,
    MemoryCache,
    PackedBlobStore,
    PackWriter,
    ThreadError,
    copy_objects,
    open_index,
    pack_index,
    pack_to_bytes,
    snapshot_keys,
    store_at,
)
from thread_sdk.pack import PACK_FOOTER_BYTES, PACK_HEADER_BYTES, pack_footer, parse_pack_footer

from helpers import build_gen, rng, unit_rows


def data(n: int, seed: int = 0) -> bytes:
    return bytes((i * 31 + seed) & 0xFF for i in range(n))


def test_pack_format_roundtrip():
    buf = io.BytesIO()
    w = PackWriter(buf, {"gen": "g", "rev": 3})
    w.add("idx/current.json", b'{"gen":"g","rev":3}')
    w.add("idx/g/slab/00001", data(1001, 1))
    w.add("empty", b"")
    w.add("idx/g/slab/00000", data(17, 2))
    with pytest.raises(ThreadError, match="added twice"):
        w.add("empty", b"x")
    size, count = w.finish()
    pack = buf.getvalue()
    assert (len(pack), count) == (size, 4)
    assert pack[:8] == b"THRDPACK" and pack[-8:] == b"THRDPACK"
    dir_off, dir_len, n = parse_pack_footer(pack[-PACK_FOOTER_BYTES:])
    assert n == 4
    assert json.loads(pack[dir_off : dir_off + dir_len])["gen"] == "g"

    s = PackedBlobStore.from_bytes(pack)
    assert (s.gen, s.rev) == ("g", 3)
    assert s.list("") == ["empty", "idx/current.json", "idx/g/slab/00000", "idx/g/slab/00001"]
    assert s.list("idx/g/") == ["idx/g/slab/00000", "idx/g/slab/00001"]
    assert all(off % 8 == 0 for off, _ in s.entries.values())
    assert s.entries["idx/current.json"][0] == PACK_HEADER_BYTES
    assert bytes(s.get("idx/g/slab/00001")) == data(1001, 1)
    assert bytes(s.get("idx/g/slab/00001", BlobRange(1000, 1))) == data(1001, 1)[1000:]
    assert [bytes(p) for p in s.get_ranges("idx/g/slab/00000", [BlobRange(0, 2), BlobRange(15, 2)])] == [data(17, 2)[:2], data(17, 2)[15:]]
    assert s.get("empty") == b""
    assert s.get("nope") is None and s.head("nope") is None
    assert s.head("idx/g/slab/00001").size == 1001
    with pytest.raises(ThreadError, match="outside the object"):
        s.get("idx/g/slab/00000", BlobRange(10, 10))
    assert not s.supports_put
    with pytest.raises(ThreadError, match="read-only"):
        s.put("x", b"1")


def test_pack_rejects_non_packs():
    with pytest.raises(ThreadError, match="too small"):
        PackedBlobStore.from_bytes(data(10))
    with pytest.raises(ThreadError, match="bad footer magic"):
        PackedBlobStore.from_bytes(data(100))
    bad = bytearray(pack_footer(16, 10, 1))
    bad[20] = 9  # version
    with pytest.raises(ThreadError, match="unknown pack version 9"):
        PackedBlobStore.from_bytes(data(40) + bytes(bad))
    buf = io.BytesIO()
    w = PackWriter(buf)
    w.add("a", data(5))
    w.finish()
    p = bytearray(buf.getvalue())
    p[-PACK_FOOTER_BYTES:] = pack_footer(len(p), 10, 1)  # directory outside the pack
    with pytest.raises(ThreadError, match="outside the pack"):
        PackedBlobStore.from_bytes(bytes(p))
    with pytest.raises(ThreadError, match="no pack at missing"):
        PackedBlobStore(MemoryBlobStore(), "missing")


def test_snapshot_keys():
    b = build_gen("gs", 3)
    with open_index(b.store, cache=False) as index:
        k0 = snapshot_keys(index.revision)
        assert k0 == sorted(["idx/gs/coarse.f32", "idx/gs/fine.f32", "idx/gs/manifest.json", "idx/gs/pq.bin",
                             "idx/gs/slabs.json", *(f"idx/gs/slab/0000{c}" for c in range(8))])
        index.configure_lexical(text=True)
        index.upsert([900, 901], unit_rows(2, b.dim, rng(5)), texts=["nine hundred", "nine oh one"], metas=["a", "b"])
        index.delete([3])
        k = snapshot_keys(index.revision)
        assert "idx/gs/manifest.3.json" in k
        assert sum(x.startswith("idx/gs/delta/") for x in k) == 1
        assert sum(x.startswith("idx/gs/seg/") for x in k) == 1
        assert sum(x.startswith("idx/gs/tombstones.") for x in k) == 2
        assert not {"idx/gs/manifest.json", "idx/gs/manifest.1.json", "idx/current.json"} & set(k)
    legacy = build_gen("gl", 1, legacy=True)
    with open_index(legacy.store, cache=False) as li:
        assert snapshot_keys(li.revision) == sorted(["idx/gl/coarse.f32", "idx/gl/fine.f32", "idx/gl/manifest.json",
                                                      *(f"idx/gl/slab/0000{c}" for c in range(8))])


def _same_search(a, c) -> None:
    assert [(h.id, h.score, h.doc.meta if h.doc else None) for h in a.hits] == \
        [(h.id, h.score, h.doc.meta if h.doc else None) for h in c.hits]


def test_pack_answers_like_its_source_and_is_read_only():
    b = build_gen("gp", 3, n=300)
    src = open_index(b.store, cache=False)
    src.configure_lexical(text=True, bucket_terms=8)
    fresh = unit_rows(20, b.dim, rng(3))
    ids = [5000 + i for i in range(20)]
    src.upsert(ids, fresh, texts=[f"fresh vector {i} {'odd' if i % 2 else 'even'}" for i in ids],
               metas=[f"m{i}" for i in ids], attrs=[i % 5 for i in ids])
    src.delete([1, 2, 5003])
    pack, result = pack_to_bytes(b.store)
    assert (result.gen, result.rev, result.count, result.bytes) == ("gp", src.rev, len(result.keys), len(pack))
    assert result.keys[0] == "idx/current.json"
    assert len(b.store.list("")) > result.count  # old manifests are not in the snapshot

    ps = PackedBlobStore.from_bytes(pack)
    assert ps.list("") == sorted(result.keys)
    idx = open_index(ps, cache=False)
    assert (idx.gen, idx.rev) == ("gp", src.rev)
    qr = rng(21)
    for _ in range(6):
        q = unit_rows(1, b.dim, qr)
        a = src.query(q, nprobe=5, rerank=30, return_payload=True)
        c = idx.query(q, nprobe=5, rerank=30, return_payload=True)
        assert c.hits == a.hits
        _same_search(src.search(text="fresh odd", vector=q, top_k=8, docs=True),
                     idx.search(text="fresh odd", vector=q, top_k=8, docs=True))
    _same_search(src.search(text="even vector", top_k=5, docs=True), idx.search(text="even vector", top_k=5, docs=True))
    with pytest.raises(ThreadError, match="read-only"):
        idx.upsert([1], unit_rows(1, b.dim, qr))
    with pytest.raises(ThreadError, match="read-only"):
        idx.delete([1])
    # the source moves on; the pack stays at its revision
    src.delete([5004])
    assert idx.refresh() is False
    assert idx.rev == src.rev - 1

    # copied out into a writable store, the snapshot continues from where it was packed
    dst = MemoryBlobStore()
    copied, nbytes, skipped = copy_objects(ps, dst)
    assert (copied, skipped) == (result.count, 0)
    assert nbytes == sum(ln for _, ln in ps.entries.values())
    assert copy_objects(ps, dst, skip_existing=True) == (0, 0, result.count)
    with open_index(dst, cache=False) as w:
        assert w.rev == idx.rev
        w.upsert([7000], unit_rows(1, b.dim, qr), texts=["copied then written"], metas=["c"])
        assert w.rev == idx.rev + 1
        assert w.search(text="copied", top_k=1).hits[0].id == 7000
    with pytest.raises(ThreadError, match="read-only"):
        copy_objects(MemoryBlobStore(), ps)
    src.close()
    idx.close()


def test_cache_in_front_of_a_pack_caches_objects_not_the_pack():
    b = build_gen("gc", 3)
    pack, _ = pack_to_bytes(b.store)
    reads: list[int] = []

    class Counting(MemoryBlobStore):
        def get(self, key, range=None, *, trace=None, fill=False):
            r = super().get(key, range, trace=trace, fill=fill)
            reads.append(len(r) if r is not None else 0)
            return r

    inner = Counting()
    inner.put("packs/gc.threadpack", pack)
    ps = PackedBlobStore(inner, "packs/gc.threadpack")
    reads.clear()  # opening read the tail (the whole pack here: it is smaller than the tail read)
    cache = MemoryCache()
    store = CachedBlobStore(ps, cache)
    idx = open_index(store, cache=False)
    q = b.vectors[: b.dim]
    r1 = idx.query(q, nprobe=8)
    assert r1.stats["io"]["misses"] > 0
    assert all(k.startswith("idx/gc/") for k in cache._items)
    assert max(reads) < len(pack)  # every read is one object or a range of it, never the pack
    before = len(reads)
    r2 = idx.query(q, nprobe=8)
    assert r2.hits == r1.hits
    assert r2.stats["io"]["reads"] == 0
    assert len(reads) == before
    idx.close()


def test_pack_index_picks_a_revision_and_fails_on_missing_objects():
    b = build_gen("gr", 2)
    with open_index(b.store, cache=False) as src:
        src.upsert([1], unit_rows(1, b.dim, rng(1)))
    _, r0 = pack_to_bytes(b.store, gen="gr", rev=0)
    assert r0.rev == 0 and not any(k.startswith("idx/gr/delta/") for k in r0.keys)
    _, r1 = pack_to_bytes(b.store, gen="gr")
    assert r1.rev == 1
    with pytest.raises(ThreadError, match="no manifest"):
        pack_to_bytes(b.store, gen="zz")
    del b.store.objects["idx/gr/slab/00003"]
    with pytest.raises(ThreadError, match="needs idx/gr/slab/00003"):
        pack_to_bytes(b.store)


def test_threaded_read_ahead_writes_the_same_bytes(tmp_path):
    b = build_gen("gt", 3)
    sequential, _ = pack_to_bytes(b.store)

    class Remote(MemoryBlobStore):
        remote = True

    remote = Remote()
    copy_objects(b.store, remote)
    r = pack_index(remote, tmp_path / "gt.threadpack", workers=3)
    assert (tmp_path / "gt.threadpack").read_bytes() == sequential
    assert r.bytes == len(sequential)
    assert not [n for n in os.listdir(tmp_path) if n.startswith(".tmp-")]


def test_packs_on_disk_and_open_index_from_a_path(tmp_path):
    b = build_gen("gd", 3, n=300)
    d = tmp_path / "idx"
    copy_objects(b.store, FileBlobStore(d))
    assert isinstance(store_at(d), FileBlobStore)
    on_disk = open_index(str(d), cache=False)
    assert isinstance(on_disk.store, FileBlobStore)
    on_disk.configure_lexical(text=True)
    fresh = unit_rows(10, b.dim, rng(8))
    ids = [800 + i for i in range(10)]
    on_disk.upsert(ids, fresh, texts=[f"disk doc {i}" for i in ids], metas=[f"d{i}" for i in ids])
    on_disk.delete([0, 801])
    file = tmp_path / "packs" / "gd.threadpack"
    r = pack_index(FileBlobStore(d), file)
    assert (r.gen, r.rev, file.stat().st_size) == ("gd", on_disk.rev, r.bytes)
    assert sorted(r.keys) == sorted(["idx/current.json", *snapshot_keys(on_disk.revision)])

    ps = store_at(file)
    assert isinstance(ps, PackedBlobStore) and ps.supports_ranges and not ps.remote
    from_pack = open_index(file, cache=False)
    assert (from_pack.gen, from_pack.rev) == ("gd", on_disk.rev)
    qr = rng(31)
    for _ in range(5):
        q = unit_rows(1, b.dim, qr)
        assert from_pack.query(q, nprobe=6).hits == on_disk.query(q, nprobe=6).hits
        _same_search(on_disk.search(text="disk doc", vector=q, top_k=6, docs=True),
                     from_pack.search(text="disk doc", vector=q, top_k=6, docs=True))
    with pytest.raises(ThreadError, match="read-only"):
        from_pack.upsert([1], unit_rows(1, b.dim, qr))
    on_disk.close()
    from_pack.close()
    (tmp_path / "plain.bin").write_bytes(data(100))
    with pytest.raises(ThreadError, match="not a thread pack"):
        store_at(tmp_path / "plain.bin")
    with pytest.raises(ThreadError, match="no such file"):
        open_index(tmp_path / "nothing")


def test_pack_open_stats_show_up_in_open_index():
    b = build_gen("go", 3, n=200)
    pack, _ = pack_to_bytes(b.store)
    ps = PackedBlobStore.from_bytes(pack)
    st = ps.open_stats
    # head (counted as a read of 0 bytes) + one tail read: a small directory fits in the tail
    assert st["reads"] == 2 and st["bytes"] == min(len(pack), 64 * 1024)
    idx = open_index(ps, cache=False)
    assert idx.open_stats["pack"] == st
    # the pack's base range (idx/current.json first, then the base objects) is one range read, before anything else
    assert idx.open_stats["prefetch"]["source"] == "index-pack" and idx.open_stats["prefetch"]["reads"] == 1
    assert idx.open_stats["current"]["reads"] == 0 and idx.open_stats["base"]["reads"] == 0
    idx.close()
