"""Embedding models: the manifest `embedding` block against the fixture shared with the TypeScript SDK
(sdk/ts/test/fixtures/embedding.json), the provider embedders over a fake transport, and a dense generation created
from nothing for a model (`create_generation`) that embeds texts on upsert and search."""

from __future__ import annotations

import json
import math
from array import array
from pathlib import Path

import pytest

from thread_sdk import (COMPACTION_PROFILES, CREATE_DEFAULTS, EMBED_BATCH_DEFAULTS, EMBEDDING_PROVIDERS, MemoryBlobStore,
                        ThreadError, create_generation, embedding_batches, embedding_inputs, keys, normalize_rows,
                        open_index, resolve_embedder, split_rows, validate_embedding_config)

from helpers import brute_force, dot, rng, unit_rows

FX = json.loads((Path(__file__).parent.parent.parent / "ts" / "test" / "fixtures" / "embedding.json").read_text())
DIM = 16


def fake_model(dim: int = DIM):
    """A deterministic stand-in for a model: a text's vector is the sum of a seeded random unit vector per word (so
    texts sharing words are close). The same construction as the TypeScript test's."""
    words: dict[str, array] = {}

    def word_vec(w: str) -> array:
        v = words.get(w)
        if v is None:
            h = 2166136261
            for ch in w:
                h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
            v = unit_rows(1, dim, rng(h))
            words[w] = v
        return v

    def model(texts, kind=None):
        out = []
        for t in texts:
            v = array("f", [0.0] * dim)
            ws = [w for w in t.split() if w]
            if not ws:
                v[0] = 1.0
            for w in ws:
                wv = word_vec(w)
                for d in range(dim):
                    v[d] += wv[d]
            out.append(v)
        return out

    return model


def test_fixture_constants():
    assert list(EMBEDDING_PROVIDERS) == FX["providers"]
    assert EMBED_BATCH_DEFAULTS == FX["batch_defaults"]
    assert CREATE_DEFAULTS == FX["create_defaults"]


@pytest.mark.parametrize("case", FX["configs"], ids=[c["name"] for c in FX["configs"]])
def test_config(case):
    if case.get("error"):
        with pytest.raises(ThreadError):
            validate_embedding_config(case["config"], case["dim"])
    else:
        assert validate_embedding_config(case["config"], case["dim"]) == case["resolved"]


def test_inputs_and_batches():
    for c in FX["inputs"]:
        assert embedding_inputs(c["config"], c["texts"], c["kind"]) == c["inputs"]
    for c in FX["batches"]:
        assert embedding_batches(c["config"], c["n"]) == c["batches"]


def test_rows():
    assert [list(r) for r in split_rows([1, 0, 0, 2], 2, 2, "x")] == [[1, 0], [0, 2]]
    assert [list(r) for r in split_rows([[3, 4]], 1, 2, "x")] == [[3, 4]]
    assert [list(r) for r in split_rows(array("f", [1, 2]).tobytes(), 1, 2, "x")] == [[1, 2]]
    with pytest.raises(ThreadError, match="3 floats"):
        split_rows([1, 2, 3], 2, 2, "x")
    with pytest.raises(ThreadError, match="3 dims"):
        split_rows([[1, 2, 3]], 1, 2, "x")
    r = normalize_rows([array("f", [3, 4])], "x")[0]
    assert abs(r[0] - 0.6) < 1e-6 and abs(r[1] - 0.8) < 1e-6
    with pytest.raises(ThreadError, match="zero"):
        normalize_rows([array("f", [0, 0])], "x")
    np = pytest.importorskip("numpy")
    assert [list(r) for r in split_rows(np.array([[1, 2], [3, 4]], dtype="f4"), 2, 2, "x")] == [[1, 2], [3, 4]]
    with pytest.raises(ThreadError, match="shape"):
        split_rows(np.zeros((2, 3)), 2, 2, "x")


def test_workers_ai_rest():
    seen = []

    def post(url, headers, body):
        seen.append((url, headers, json.loads(body)))
        texts = json.loads(body)["text"]
        return 200, json.dumps({"success": True, "result": {"shape": [len(texts), 4], "data": [[len(t), i, 1, 0] for i, t in enumerate(texts)]}}).encode()

    cfg = {"provider": "workers-ai", "model": "@cf/baai/bge-small-en-v1.5", "dim": 4, "options": {"pooling": "cls"}, "queryPrefix": "Q: "}
    embed = resolve_embedder(cfg, account_id="acct", api_token="tok", post=post)
    rows = embed([f"t{i}" for i in range(150)], "document")
    assert len(rows) == 150
    assert [len(s[2]["text"]) for s in seen] == [100, 50]
    assert seen[0][0] == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/baai/bge-small-en-v1.5"
    assert seen[0][1]["authorization"] == "Bearer tok"
    assert seen[0][2]["pooling"] == "cls"
    assert list(rows[7]) == [2, 7, 1, 0] and list(rows[107]) == [4, 7, 1, 0]
    q = embed(["hello"], "query")
    assert seen[2][2]["text"] == ["Q: hello"] and q[0][0] == 8
    with pytest.raises(ThreadError, match="account_id and api_token"):
        resolve_embedder(cfg)
    bad = resolve_embedder(cfg, account_id="a", api_token="t", post=lambda u, h, b: (200, b'{"result": {"data": [[1, 2]]}}'))
    with pytest.raises(ThreadError, match="2 dims, expected 4"):
        bad(["x"], "query")
    err = resolve_embedder(cfg, account_id="a", api_token="t", post=lambda u, h, b: (401, b"nope"))
    with pytest.raises(ThreadError, match="HTTP 401"):
        err(["x"], "query")


def test_openai_compatible():
    reqs = []

    def post(url, headers, body):
        b = json.loads(body)
        reqs.append((url, headers, b))
        data = [{"object": "embedding", "index": i, "embedding": [i, len(t)]} for i, t in enumerate(b["input"])][::-1]
        return 200, json.dumps({"object": "list", "data": data}).encode()

    cfg = {"provider": "openai", "model": "text-embedding-3-small", "dim": 2, "options": {"dimensions": 2}, "baseUrl": "https://example.test/v1/", "maxBatch": 2}
    rows = resolve_embedder(cfg, api_key="sk-test", post=post)(["a", "bb", "ccc"], "document")
    assert [list(r) for r in rows] == [[0, 1], [1, 2], [0, 3]]
    assert len(reqs) == 2
    assert reqs[0][0] == "https://example.test/v1/embeddings"
    assert reqs[0][1]["authorization"] == "Bearer sk-test"
    assert reqs[0][2] == {"dimensions": 2, "model": "text-embedding-3-small", "input": ["a", "bb"]}


def test_custom_provider():
    cfg = {"provider": "custom", "model": "m", "dim": 2, "maxBatch": 2, "docPrefix": "D "}
    with pytest.raises(ThreadError, match="needs an embedder"):
        resolve_embedder(cfg)
    seen = []

    def emb(texts, kind):
        seen.append((texts, kind))
        return [[1, 1] for _ in texts]

    assert len(resolve_embedder(cfg, embedder=emb)(["a", "b", "c"], "document")) == 3
    assert seen == [(["D a", "D b"], "document"), (["D c"], "document")]


def test_create_generation_for_a_model():
    store = MemoryBlobStore()
    model = fake_model()
    embedding = {"provider": "custom", "model": "fake-16", "dim": DIM}
    m = create_generation(store, "e1", embedding=embedding, attrs={"tenant": {"bits": 8}})
    for k, v in FX["create_manifest"].items():
        assert m[k] == v
    assert m["gen"] == "e1" and m["dim"] == DIM and m["profile"] == "edge" and m["defaults"] == FX["create_defaults"]
    assert m["embedding"] == embedding and m["lex"] == {} and m["compaction"] == COMPACTION_PROFILES["edge"]
    table = json.loads(store.get(keys.slabs_table("e1", 0)))
    assert len(table) == 1 and table[0][2] == 0 and table[0][3] == len(store.get(keys.slab("e1", 0)))
    assert len(store.get("idx/e1/fine.f32")) == DIM * 4
    with pytest.raises(ThreadError, match="already exists"):
        create_generation(store, "e1", dim=DIM)

    # Without a way to call the model, the index reads and takes vectors but cannot embed.
    plain = open_index(store, cache=False)
    assert plain.has_dense and not plain.can_embed and plain.embedding == embedding
    with pytest.raises(ThreadError, match="cannot embed"):
        plain.embed(["x"], "query")
    assert plain.query(unit_rows(1, DIM, rng(1))).hits == []  # the empty cell
    with pytest.raises(ThreadError, match="cannot embed"):
        plain.search(text="x", embed=True)
    plain.close()

    calls = []

    def embed(texts, kind):
        calls.append(kind)
        return model(texts)

    index = open_index(store, cache=False, embed=embed)
    assert index.can_embed
    docs = ["red apples and pears", "green pears", "blue sky over the sea", "the sea at night", "apples in the sky",
            "night market with pears and apples", "an empty sky", "red sea"]
    up = index.upsert(keys=[f"d/{i}" for i in range(len(docs))], texts=docs, metas=docs, attrs=[{"tenant": i % 2} for i in range(len(docs))])
    assert calls == ["document"]
    assert up.ids == list(range(len(docs))) and up.keys["delta"] is not None and all(c == 0 for c in up.cells)
    assert index.rev == 1

    # search(text=) embeds the text (as a query) and fuses it with BM25.
    hybrid = index.search(text="pears", top_k=4, docs=True)
    assert calls == ["document", "query"]
    assert hybrid.mode == "hybrid" and len(hybrid.hits) == 4
    assert all(h.dense or h.lexical for h in hybrid.hits) and any(h.dense and h.lexical for h in hybrid.hits)
    assert "pears" in hybrid.hits[0].doc.meta
    # embed=False keeps the text lexical-only; a given vector is used as is.
    assert index.search(text="pears", embed=False, top_k=3).mode == "lexical"
    assert len(calls) == 2
    q = normalize_rows(model(["apples"]), "q")[0]
    dense = index.search(vector=q, top_k=3)
    assert dense.mode == "dense"
    rows = normalize_rows(model(docs), "d")
    flat = array("f")
    for r in rows:
        flat.extend(r)
    want = brute_force(flat, DIM, q, 3)
    assert [h.id for h in dense.hits] == want
    via_text = index.search(text="apples", top_k=3, weights={"lexical": 0})
    assert [h.id for h in via_text.hits if h.dense] == want
    filtered = index.search(text="apples", top_k=3, filter={"tenant": 1})
    assert all(h.id % 2 == 1 for h in filtered.hits)
    assert len(calls) == 4
    # Keys and docs work as for any generation; a text-only upsert (embed=False) adds no vector.
    assert index.get(["d/3"])["d/3"].meta == docs[3]
    text_only = index.upsert(keys=["d/8"], texts=["pears pears pears"], embed=False)
    assert text_only.keys["delta"] is None and len(calls) == 4
    own = index.upsert(keys=["d/9"], texts=["x"], embed=lambda texts: [unit_rows(1, DIM, rng(9)) for _ in texts])
    assert own.keys["delta"] is not None and len(calls) == 4
    with pytest.raises(ThreadError, match="dims"):
        index.upsert(ids=[100], vectors=[array("f", [0.0] * (DIM + 1))])
    # Chunked documents embed one vector per chunk.
    r = index.upsert(keys=["doc"], texts=[" ".join(f"w{i}" for i in range(9))], chunk={"words": 3})
    assert r.chunks == {"documents": 1, "count": 3, "stale": 0} and r.keys["delta"] is not None
    assert calls[-1] == "document"
    hit = index.search(text="w4", top_k=1)
    assert hit.hits[0].lexical.rank == 1
    index.close()


def test_create_generation_dense_only():
    store = MemoryBlobStore()
    m = create_generation(store, "v1", dim=8, lex=False, profile="local", defaults={"nprobe": 2}, compaction=False, set_current=False)
    assert "lex" not in m and "embedding" not in m and "compaction" not in m
    assert m["defaults"] == {"nprobe": 2, "rerank": 150, "pc": 0}
    assert store.head(keys.current()) is None
    index = open_index(store, gen="v1", cache=False)
    assert index.embedding is None
    with pytest.raises(ThreadError, match="names no embedding model"):
        index.embed(["x"], "query")
    vectors = unit_rows(300, 8, rng(3))
    for a in range(0, 300, 100):
        index.upsert(ids=list(range(a, a + 100)), vectors=vectors[a * 8:(a + 100) * 8], replace=False)
    assert index.rev == 3
    q = unit_rows(1, 8, rng(4))
    assert [h.id for h in index.query(q, top_k=5).hits] == brute_force(vectors, 8, q, 5)
    with pytest.raises(ThreadError, match="lexical configuration"):
        index.upsert(keys=["k"], texts=["t"])
    with pytest.raises(ThreadError, match="no lexical index"):
        index.search(text="t")
    index.close()


def test_create_generation_errors():
    store = MemoryBlobStore()
    with pytest.raises(ThreadError, match="embedding model .* or a vector size"):
        create_generation(store, "x")
    with pytest.raises(ThreadError, match="bad generation name"):
        create_generation(store, "bad name", dim=4)
    with pytest.raises(ThreadError, match="does not match"):
        create_generation(store, "x", dim=4, embedding={"provider": "custom", "model": "m", "dim": 8})
    with pytest.raises(ThreadError, match="profile"):
        create_generation(store, "x", dim=4, profile="fast")
    with pytest.raises(ThreadError, match="nprobe"):
        create_generation(store, "x", dim=4, defaults={"nprobe": 0})
    assert store.head(keys.manifest("x", 0)) is None


def test_ts_manifest_opens_here(tmp_path):
    """A generation the TypeScript SDK created (its manifest shape) opens and embeds here: both writers produce the
    fixture's manifest fields, so the interop is in the shared fixture rather than a cross-process run."""
    store = MemoryBlobStore()
    m = create_generation(store, "g", embedding={"provider": "custom", "model": "m", "dim": 4})
    for k, v in FX["create_manifest"].items():
        assert m[k] == v
    ix = open_index(store, cache=False, embed=lambda texts, kind: [[1.0, 0.0, 0.0, 0.0] for _ in texts])
    assert [list(r) for r in ix.embed(["a"], "query")] == [[1.0, 0.0, 0.0, 0.0]]
    ix.close()
