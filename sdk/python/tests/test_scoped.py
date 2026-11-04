"""Scoped tombstones (`upsert(scope=...)`): a vector-only upsert by key keeps the text, docs and key; a text-only
re-put keeps the vector; deletes hide both; attrs cannot drift; segment compaction drops the lexical sets; readers
refuse unknown manifest features. The same rules as the TypeScript SDK (sdk/ts/test/scoped.test.ts)."""

from __future__ import annotations

import json

import pytest

from thread_sdk import MemoryBlobStore, ThreadError, create_generation, keys, open_index

from helpers import rng, unit_rows

DIM = 16


def setup():
    store = MemoryBlobStore()
    create_generation(store, "s", dim=DIM, lex={"text": True}, attrs={"tag": {"bits": 4}},
                      compaction={"maxDeltas": 1, "maxCellEntries": 0})
    return store, open_index(store, cache=False, refresh_interval=0), unit_rows(8, DIM, rng(4))


def row(v, i):
    return list(v[i * DIM:(i + 1) * DIM])


def top(ix, q):
    hits = ix.query(q, top_k=1, nprobe=8).hits
    return hits[0].id if hits else None


def text_ids(ix, text):
    return [h.id for h in ix.search(text=text, top_k=5, embed=False).hits]


def test_dense_scope_keeps_the_lexical_side():
    store, ix, v = setup()
    with ix:
        a = ix.upsert(keys=["mail/1", "mail/2"], texts=["invoice from the plumber", "team lunch friday"],
                      metas=["m1", "m2"], attrs=[{"tag": 3}, {"tag": 1}])
        r = ix.upsert(keys=["mail/1"], vectors=[row(v, 0)], attrs=[{"tag": 3}], scope="dense")
        assert r.ids == [a.ids[0]] and r.keys["segment"] is None
        assert ix.manifest["features"] == ["scoped-tombstones"] and ix.manifest["tombstones"][-1]["scope"] == "dense"
        assert text_ids(ix, "plumber") == [a.ids[0]]
        got = ix.get(["mail/1"], text=True)["mail/1"]
        assert (got.id, got.meta, got.text) == (a.ids[0], "m1", "invoice from the plumber")
        assert top(ix, row(v, 0)) == a.ids[0]
        with pytest.raises(ThreadError) as e:
            ix.upsert(keys=["mail/9"], vectors=[row(v, 2)], scope="dense")
        assert e.value.status == 404


def test_lexical_scope_keeps_the_vector_and_the_attrs():
    store, ix, v = setup()
    with ix:
        i = ix.upsert(keys=["doc"], texts=["old words here"], vectors=[row(v, 0)], attrs=[{"tag": 5}]).ids[0]
        ix.upsert(keys=["doc"], texts=["fresh new text"], metas=["renamed"], scope="lexical")
        assert top(ix, row(v, 0)) == i
        assert text_ids(ix, "old words") == [] and text_ids(ix, "fresh") == [i]
        assert ix.get(["doc"])["doc"].attrs == 5
        with pytest.raises(ThreadError, match="cannot change the attrs"):
            ix.upsert(keys=["doc"], texts=["x"], attrs=[{"tag": 6}], scope="lexical")
        with pytest.raises(ThreadError):
            ix.upsert(keys=["doc"], vectors=[row(v, 1)], scope="lexical")
        ix.delete([i])
        assert top(ix, row(v, 0)) != i and text_ids(ix, "fresh") == [] and ix.get(["doc"]) == {}


def test_segment_compaction_drops_lexical_sets_and_keeps_vectors():
    store, ix, v = setup()
    with ix:
        a = ix.upsert(keys=["a", "b"], texts=["alpha text", "bravo text"], vectors=[row(v, 0), row(v, 1)])
        ix.upsert(keys=["b"], texts=["bravo renamed"], scope="lexical")
        assert [t.get("scope") for t in ix.manifest["tombstones"]] == [None, "lexical"]
        c = ix.compact_segments()
        assert c.merged == 2
        assert [t.get("scope") for t in ix.manifest["tombstones"]] == [None]
        assert text_ids(ix, "renamed") == [a.ids[1]] and top(ix, row(v, 1)) == a.ids[1]


def test_unknown_features_are_refused():
    store, ix, v = setup()
    with ix:
        ix.upsert(keys=["a"], texts=["alpha"], vectors=[row(v, 0)])
        rev = ix.rev
        m = json.loads(store.get(keys.manifest("s", rev)))
        store.put(keys.manifest("s", rev + 1), json.dumps({**m, "rev": rev + 1, "features": ["from-the-future"]}))
        store.put(keys.current(), json.dumps({"gen": "s", "rev": rev + 1}))
    with pytest.raises(ThreadError, match="from-the-future"):
        open_index(store, cache=False)
