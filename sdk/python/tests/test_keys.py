"""The key store: keyed upserts allocate ids, `get` / `resolve_keys` / `list_keys` resolve them, a re-put keeps the id
(so it replaces the document), `delete_keys` hides them, compaction keeps the section, facet tokens list by attribute,
and a dense generation allocates past its build ids."""

from __future__ import annotations

import json

import pytest

from thread_sdk import KeyPage, KeyRecord, MemoryBlobStore, ThreadError, _native, create_lexical_generation, keys, open_index

from helpers import build_gen, rng, row, unit_rows


def key(i: int) -> str:
    return f"s3://bucket/{i % 3}/obj-{i}.bin"


def meta(i: int, host: str | None = None) -> str:
    return json.dumps({"host": host or f"h{i % 2}.example.com", "size": 100 * i})


def facets(i: int) -> str:
    return f"host:h{i % 2}.example.com size:{'small' if i < 2 else 'large'}"


def manifest_of(store, gen: str, rev: int) -> dict:
    return json.loads(store.get(keys.manifest(gen, rev)))


def test_keyed_upserts_get_list_replace_delete_compact():
    store = MemoryBlobStore()
    create_lexical_generation(store, "kv", {"tokenizer": {"identifiers": "keep"}, "bucketKeys": 2, "bucketDocs": 2,
                                            "bucketTerms": 4})
    with open_index(store, cache=False) as index:
        assert "nextId" not in index.manifest
        ks = [key(i) for i in range(4)]
        up = index.upsert(keys=ks, metas=[meta(i) for i in range(4)], texts=[facets(i) for i in range(4)])
        assert up.ids == [0, 1, 2, 3] and up.allocated == 4 and up.rev == 1
        m = manifest_of(store, "kv", 1)
        assert m["nextId"] == 4 and m["segments"][0]["keys"] is True
        assert m["change"]["keys"] == 4 and m["change"]["allocated"] == 4

        got = index.get([key(1), key(3), "nope"])
        assert list(got) == [key(1), key(3)]
        assert got[key(1)] == KeyRecord(1, meta(1), None, 0, key(1))
        assert got[key(3)].id == 3
        assert index.resolve_keys([key(2), "nope"]) == {key(2): 2}

        everything = index.list_keys("s3://bucket/")
        assert everything == KeyPage(sorted((k, i) for i, k in enumerate(ks)), False)
        assert [i for _, i in index.list_keys("s3://bucket/0/").entries] == [0, 3]
        page1 = index.list_keys("s3://bucket/", limit=1)
        assert len(page1.entries) == 1 and page1.more and page1.next == page1.entries[0][0]
        page2 = index.list_keys("s3://bucket/", after=page1.next, limit=10)
        assert page1.entries + page2.entries == everything.entries and not page2.more
        assert index.list_keys("nope") == KeyPage([], False)

        # facet tokens: an equality filter over an attribute is a term lookup
        assert sorted(h.id for h in index.search(text="host:h1.example.com", top_k=10).hits) == [1, 3]
        small = index.search(text="size:small", top_k=10, docs=True)
        assert sorted(h.id for h in small.hits) == [0, 1]
        assert small.hits[0].doc is not None and small.hits[0].doc.meta == meta(small.hits[0].id)

        # a re-put of a known key keeps its id and replaces the document
        re = index.upsert(keys=[key(1), key(4)], metas=[meta(1, "moved.example.com"), meta(4)], texts=[facets(1), facets(4)])
        assert re.ids == [1, 4] and re.allocated == 1
        assert manifest_of(store, "kv", 2)["nextId"] == 5
        assert len(index.segments) == 2
        assert index.get([key(1)])[key(1)].meta == meta(1, "moved.example.com")
        assert index.list_keys("s3://bucket/1/").entries == [(key(1), 1), (key(4), 4)]

        # explicit ids may come with keys; the counter moves past them
        ex = index.upsert([100], keys=["explicit/100"], metas=["e"])
        assert ex.ids == [100] and ex.allocated == 0
        assert manifest_of(store, "kv", 3)["nextId"] == 101
        assert index.upsert(keys=["after/explicit"], metas=["n"]).ids == [101]
        assert index.get(["explicit/100"])["explicit/100"].id == 100

        # delete by key
        d = index.delete_keys([key(2), "nope"])
        assert d.count == 1 and d.missing == ["nope"] and d.rev == 5
        assert index.get([key(2)]) == {} and index.resolve_keys([key(2)]) == {}
        assert index.list_keys("s3://bucket/2/").entries == []
        assert sorted(h.id for h in index.search(text="size:large", top_k=10).hits) == [3, 4]
        none = index.delete_keys(["nope"])
        assert none.count == 0 and none.rev == 5  # nothing written

        # compaction keeps the keys
        c = index.compact_segments()
        assert c.merged == 4 and len(index.segments) == 1 and index.segments[0]["keys"] is True
        assert index.list_keys().entries == [("after/explicit", 101), ("explicit/100", 100), (key(0), 0), (key(3), 3),
                                             (key(1), 1), (key(4), 4)]
        assert index.get([key(1)])[key(1)].meta == meta(1, "moved.example.com")
        assert index.get([key(2)]) == {}
        again = index.upsert(keys=[key(3)], metas=["v3"])
        assert again.ids == [3] and again.allocated == 0
        assert manifest_of(store, "kv", again.rev)["nextId"] == 102

    with open_index(store, cache=False) as other:
        assert other.get([key(3), key(0)])[key(3)].meta == "v3"
        assert [i for _, i in other.list_keys("s3://bucket/0/").entries] == [0, 3]


def test_keyed_batch_validation():
    store = MemoryBlobStore()
    create_lexical_generation(store, "kv2", {})
    with open_index(store, cache=False) as index:
        with pytest.raises(ThreadError, match="duplicate keys"):
            index.upsert(keys=["a", "a"], metas=["x", "y"])
        with pytest.raises(ThreadError, match=r"keys\[1\]"):
            index.upsert(keys=["a", ""], metas=["x", "y"])
        with pytest.raises(ThreadError, match="1 keys for 2 ids"):
            index.upsert([1, 2], keys=["a"], metas=["x", "y"])
        with pytest.raises(ThreadError, match="longer than 4096"):
            index.upsert(keys=["k" * 5000], metas=["x"])
        with pytest.raises(ThreadError, match="needs ids or keys"):
            index.upsert(metas=["x"])
        with pytest.raises(ThreadError, match="no keys"):
            index.delete_keys([])
        only = index.upsert(keys=["just-a-key"])
        assert only.ids == [0]
        assert index.get(["just-a-key"]) == {"just-a-key": KeyRecord(0, "", None, 0, "just-a-key")}
    b = build_gen("kd0", 2)
    with open_index(b.store, cache=False) as dense:
        with pytest.raises(ThreadError, match="keys need a lexical configuration"):
            dense.upsert(keys=["a"], vectors=unit_rows(1, b.dim, rng(1)))
        assert dense.get(["a"]) == {} and dense.list_keys() == KeyPage([], False)


def test_dense_generation_allocates_past_build_ids():
    b = build_gen("kd", 3, n=100)
    with open_index(b.store, cache=False) as index:
        index.configure_lexical(bucket_keys=4)
        r = rng(7)
        v = unit_rows(3, b.dim, r)
        up = index.upsert(keys=["doc/a", "doc/b", "doc/c"], vectors=v, metas=["A", "B", "C"], attrs=[1, 2, 3])
        assert up.ids == [100, 101, 102] and up.allocated == 3 and len(up.cells) == 3
        assert index.manifest["nextId"] == 103
        assert index.get(["doc/b"]) == {"doc/b": KeyRecord(101, "B", None, 2, "doc/b")}
        assert index.query(row(v, 1, b.dim), top_k=1).hits[0].id == 101
        index.upsert([500], unit_rows(1, b.dim, r))
        assert index.manifest["nextId"] == 501
        assert index.upsert(keys=["doc/d"], vectors=unit_rows(1, b.dim, r)).ids == [501]
        s = index.search(vector=row(v, 0, b.dim), text="A", top_k=3, docs=True)
        hit = next(h for h in s.hits if h.id == 100)
        assert hit.doc is not None and hit.doc.meta == "A"


def test_native_keys_reader_protocol():
    """`keys_get` / `keys_list` on the paged reader: one round after the directory, structured MissingRanges."""
    job = _native.SegmentJob(None, 1.2, 0.75, "exact", 0, True, False, 4, True, 3)
    for i in range(50):
        job.add(i * 2, None, f"m{i}", 0, f"k/{i:03}")
    with pytest.raises(_native.IndexFormatError, match="duplicate key"):
        job.add(999, None, "dup", 0, "k/007")
    data = job.finish()
    meta = json.loads(_native.segment_meta(data))
    assert meta["keys"] == {"version": 1, "n": 50, "bucket_keys": 3, "nbytes": meta["keys"]["nbytes"]}
    assert "lex" not in meta
    reader = _native.SegmentReader(len(data))
    for off, n in reader.plan_open():
        reader.feed(off, data[off:off + n])
    o = reader.open()
    assert o["has_keys"] and o["has_docs"] and not o["has_lex"]
    for off, n in reader.plan_headers():
        reader.feed(off, data[off:off + n])
    assert reader.info()["keys"]["n"] == 50
    for off, n in reader.plan_directories():
        reader.feed(off, data[off:off + n])
    with pytest.raises(_native.MissingRanges) as e:
        reader.keys_get(["k/010", "k/011", "zzz"])
    assert len(e.value.ranges) == 1  # 010 and 011 share a bucket; zzz has none
    for off, n in e.value.ranges:
        reader.feed(off, data[off:off + n])
    assert reader.keys_get(["k/010", "k/011", "zzz", "k/0"]) == [20, 22, None, None]
    # (the tiny sections sit within one coalesced fetch, so the listing's buckets are usually held already)
    for _ in range(4):
        try:
            reader.keys_list("k/02", None, 5)
            break
        except _native.MissingRanges as e:
            for off, n in e.ranges:
                reader.feed(off, data[off:off + n])
    page = reader.keys_list("k/02", None, 5)
    assert page["entries"] == [(f"k/{i:03}", i * 2) for i in range(20, 25)] and page["more"] is True
    page = reader.keys_list("k/02", "k/024", 10)
    assert page["entries"] == [(f"k/{i:03}", i * 2) for i in range(25, 30)] and page["more"] is False
