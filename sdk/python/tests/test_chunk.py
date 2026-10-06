"""Flat word-window chunking against the fixtures shared with the TypeScript SDK and the CLI
(sdk/ts/test/fixtures/chunks.json), and chunked upserts."""

from __future__ import annotations

import json
from array import array
from pathlib import Path

import pytest

from thread_sdk import (DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_WORDS, Chunk, ChunkOptions, MemoryBlobStore, ThreadError,
                        chunk_index_of, chunk_key, chunk_text, create_lexical_generation, keys, open_index,
                        resolve_chunk_options)

from helpers import build_gen, rng, row, unit_rows

FX = json.loads((Path(__file__).parent.parent.parent / "ts" / "test" / "fixtures" / "chunks.json").read_text())


def test_defaults():
    assert {"words": DEFAULT_CHUNK_WORDS, "overlap": DEFAULT_CHUNK_OVERLAP} == FX["defaults"]
    assert resolve_chunk_options(None) == resolve_chunk_options(True) == ChunkOptions(200, 0)
    assert resolve_chunk_options({"overlap": 20}) == ChunkOptions(200, 20)
    with pytest.raises(ValueError):
        resolve_chunk_options({"size": 3})


@pytest.mark.parametrize("case", FX["cases"], ids=[c["name"] for c in FX["cases"]])
def test_fixture(case):
    kw = {k: case[k] for k in ("words", "overlap") if case[k] is not None}
    got = chunk_text(case["text"], **kw)
    assert [vars(c) for c in got] == [{"index": c["index"], "text": c["text"], "start": c["start"], "end": c["end"],
                                      "words": c["words"]} for c in case["chunks"]]
    for c in got:
        assert case["text"][c.start:c.end] == c.text


def test_errors():
    for e in FX["errors"]:
        with pytest.raises(ValueError):
            chunk_text("a b c", e["words"], e["overlap"])
    with pytest.raises(ValueError):
        chunk_text("a", 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        chunk_text("a", True)  # type: ignore[arg-type]


def test_keys():
    for k in FX["keys"]:
        assert chunk_key(k["key"], k["index"]) == k["chunk"]
    for p in FX["key_parse"]:
        assert chunk_index_of(p["key"], p["parent"]) == p["index"]


def words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def test_chunked_upsert_text_only():
    store = MemoryBlobStore()
    create_lexical_generation(store, "docs", {"text": True, "tokenizer": {"stopwords": None, "stemmer": None}})
    with open_index(store, cache=False) as index:
        index.configure_attrs(tenant={"bits": 4})
        r = index.upsert(keys=["a", "b"], texts=[words(5, "a"), words(2, "b")], metas=["A", "B"],
                         attrs=[{"tenant": 3}, {"tenant": 4}], chunk={"words": 2, "overlap": 1})
        # a: a0..a4 in windows of 2 with overlap 1 -> [a0 a1] [a1 a2] [a2 a3] [a3 a4]; b: one chunk
        assert r.count == 5 and r.ids == [0, 1, 2, 3, 4]
        assert r.chunks == {"documents": 2, "count": 5, "stale": 0}
        got = index.get(["a#0", "a#3", "b#0", "a"], text=True)
        assert list(got) == ["a#0", "a#3", "b#0"]
        assert (got["a#0"].id, got["a#0"].meta, got["a#0"].text) == (0, "A", "a0 a1")
        assert (got["a#3"].id, got["a#3"].meta, got["a#3"].text) == (3, "A", "a3 a4")
        assert (got["b#0"].id, got["b#0"].meta, got["b#0"].text) == (4, "B", "b0 b1")
        assert index.unpack_attrs(got["a#3"].attrs) == {"tenant": 3}
        assert index.unpack_attrs(got["b#0"].attrs) == {"tenant": 4}
        assert [k for k, _ in index.list_keys("a#").entries] == ["a#0", "a#1", "a#2", "a#3"]
        assert sorted(h.id for h in index.search(text="a2", top_k=10).hits) == [1, 2]
        m = json.loads(store.get(keys.manifest("docs", r.rev)))
        assert m["change"]["keys"] == 5 and m["change"]["chunk"] == {"words": 2, "overlap": 1, "documents": 2, "stale": 0}

        # a re-put with fewer chunks keeps the ids of the chunks it still has and tombstones the leftover ones
        r2 = index.upsert(keys=["a"], texts=[words(3, "a")], metas=["A2"], chunk=ChunkOptions(2, 1))
        assert r2.ids == [0, 1] and r2.allocated == 0
        assert r2.chunks == {"documents": 1, "count": 2, "stale": 2}
        assert [k for k, _ in index.list_keys("a#").entries] == ["a#0", "a#1"]
        assert index.get(["a#0"])["a#0"].meta == "A2"
        assert index.search(text="a4", top_k=10).hits == [] and index.search(text="a3", top_k=10).hits == []
        m2 = json.loads(store.get(keys.manifest("docs", r2.rev)))
        assert m2["tombstones"][-1]["count"] == 4  # ids 0, 1 (replaced) + 2, 3 (stale)

        # replace=False still tombstones stale chunks
        r3 = index.upsert(keys=["a"], texts=["a0"], chunk=True, replace=False)
        assert r3.chunks == {"documents": 1, "count": 1, "stale": 1}
        assert [k for k, _ in index.list_keys("a#").entries] == ["a#0"]

        # the defaults: 200 words, no overlap; a text with no words is one empty chunk
        r4 = index.upsert(keys=["c", "d"], texts=[words(450, "c"), "   "], chunk=True)
        assert r4.chunks == {"documents": 2, "count": 4, "stale": 0}
        assert len(index.list_keys("c#").entries) == 3
        assert index.get(["d#0"], text=True)["d#0"].text == ""


def test_chunked_upsert_embeds_and_validates():
    g = build_gen("cg", 3, n=100, dim=8)
    with open_index(g.store, cache=False) as index:
        index.configure_lexical(text=True)
        seen: list[list[str]] = []
        r9 = rng(9)

        def embed(texts: list[str]):
            seen.append(list(texts))
            return unit_rows(len(texts), g.dim, r9)  # flat n x dim, as `vectors` takes it

        r = index.upsert(keys=["p"], texts=[words(7, "p")], chunk={"words": 3}, embed=embed)
        assert r.count == 3 and seen == [["p0 p1 p2", "p3 p4 p5", "p6"]] and r.distinct_cells > 0
        assert len(index.query(row(embed(["p6"]), 0, g.dim), top_k=3).hits) == 3
        assert r.ids == [100, 101, 102]
        # embed without chunk: the texts as given
        r2 = index.upsert([900], texts=["whole document"], embed=embed)
        assert r2.count == 1 and seen[-1] == ["whole document"]

        with pytest.raises(ThreadError, match="by key"):
            index.upsert([1], texts=["x"], chunk=True)
        with pytest.raises(ThreadError, match="texts"):
            index.upsert(keys=["k"], chunk=True)
        with pytest.raises(ThreadError, match="embed"):
            index.upsert(keys=["k"], vectors=unit_rows(1, g.dim, rng(1)), texts=["x"], chunk=True)
        with pytest.raises(ThreadError, match="embed"):
            index.upsert(keys=["k"], vectors=unit_rows(1, g.dim, rng(1)), texts=["x"], embed=embed)
        with pytest.raises(ThreadError):
            index.upsert(keys=["k"], texts=["x"], chunk={"words": 0})
        with pytest.raises(ThreadError, match="embed returned 0 vectors for 2"):
            index.upsert(keys=["k"], texts=["x y"], chunk={"words": 1}, embed=lambda t: [])
        with pytest.raises(ThreadError, match="expected 2 x 8"):
            index.upsert(keys=["k"], texts=["x y"], chunk={"words": 1}, embed=lambda t: array("f", [1.0] * 8))
        with pytest.raises(ThreadError, match="4096"):
            index.upsert(keys=["k"], texts=[words(5000, "z")], chunk={"words": 1})
