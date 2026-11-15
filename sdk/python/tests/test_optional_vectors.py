"""Optional vectors (`upsert(vectors=[row, None, ...])`): a None row writes that document without a vector (text, meta
and key only) and the others with theirs, in one revision; the delta holds only the rows given, with their ids,
attrs and payloads. The same rules as the TypeScript SDK (sdk/ts/test/optional-vectors.test.ts)."""

from __future__ import annotations

import pytest

from thread_sdk import MemoryBlobStore, ThreadError, create_generation, open_index

from helpers import rng, unit_rows

DIM = 16


def setup(lex=True):
    store = MemoryBlobStore()
    create_generation(store, "o", dim=DIM, lex={"text": True} if lex else False, attrs={"tag": {"bits": 4}},
                      compaction={"maxDeltas": 1, "maxCellEntries": 0})
    return store, open_index(store, cache=False, refresh_interval=0), unit_rows(8, DIM, rng(9))


def row(v, i):
    return list(v[i * DIM:(i + 1) * DIM])


def top(ix, q, filter=None):
    hits = ix.query(q, top_k=1, nprobe=8, **({"filter": filter} if filter else {})).hits
    return hits[0].id if hits else None


def text_ids(ix, text):
    return [h.id for h in ix.search(text=text, top_k=5, embed=False).hits]


def test_a_mixed_batch_writes_the_given_rows_and_every_document():
    store, ix, v = setup()
    with ix:
        before = ix.rev
        r = ix.upsert(keys=["a", "b", "c"], texts=["alpha words", "bravo words", "charlie words"], metas=["A", "B", "C"],
                      attrs=[{"tag": 1}, {"tag": 2}, {"tag": 3}], vectors=[row(v, 0), None, row(v, 2)],
                      payloads=["pa", "", "pc"])
        assert r.rev == before + 1 and r.count == 3 and len(r.cells) == 2
        assert ix.manifest["deltas"][-1]["count"] == 2 and ix.manifest["segments"][-1]["count"] == 3
        a, b, c = r.ids
        assert top(ix, row(v, 0)) == a and top(ix, row(v, 2)) == c
        assert top(ix, row(v, 2), {"tag": 3}) == c and top(ix, row(v, 0), {"tag": 2}) is None
        assert text_ids(ix, "bravo") == [b]
        got = ix.get(["b"], text=True)["b"]
        assert (got.id, got.meta, got.attrs, got.text) == (b, "B", 2, "bravo words")
        assert bytes(ix.query(row(v, 2), top_k=1, nprobe=8, return_payload=True).hits[0].payload) == b"pc"


def test_a_none_row_hides_the_older_vector():
    store, ix, v = setup()
    with ix:
        a = ix.upsert(keys=["doc", "other"], texts=["first text", "other text"], vectors=[row(v, 0), row(v, 1)])
        ix.upsert(keys=["doc", "other"], texts=["second text", "other text"], vectors=[None, row(v, 1)])
        assert top(ix, row(v, 0)) == a.ids[1]
        assert text_ids(ix, "second") == [a.ids[0]] and text_ids(ix, "first") == []


def test_every_row_none_is_text_only():
    store, ix, v = setup()
    with ix:
        r = ix.upsert(keys=["x", "y"], texts=["xray", "yankee"], vectors=[None, None])
        assert r.keys["delta"] is None and r.cells == []
        assert text_ids(ix, "yankee") == [r.ids[1]]


@pytest.mark.parametrize("kw, msg", [
    ({"ids": [1, 2]}, "needs texts, metas or keys"),
    ({"ids": [1, 2], "texts": ["a", "b"], "payloads": ["p", "q"]}, r"payloads\[1\] belongs to no vector"),
    ({"ids": [1, 2], "texts": ["a", "b"], "payloads": ["p"]}, "1 payloads for 2 documents"),
])
def test_refused_none_rows(kw, msg):
    store, ix, v = setup()
    with ix:
        before = ix.rev
        with pytest.raises(ThreadError, match=msg):
            ix.upsert(vectors=[row(v, 0), None], **kw)
        assert ix.rev == before


def test_refused_without_lex_or_under_dense_scope():
    store, ix, v = setup()
    with ix:
        ix.upsert(keys=["k"], texts=["kilo"])
        with pytest.raises(ThreadError, match="needs a vector for every document"):
            ix.upsert(keys=["k"], vectors=[None], scope="dense")
        with pytest.raises(ThreadError, match=r"vector 2 \(id 9\)"):
            ix.upsert(ids=[7, 8, 9], texts=["a", "b", "c"], vectors=[None, row(v, 0), [0.0] * DIM])
    store, ix, v = setup(lex=False)
    with ix:
        with pytest.raises(ThreadError, match="needs a lexical configuration"):
            ix.upsert(ids=[1, 2], vectors=[row(v, 0), None])
