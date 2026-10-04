"""End-to-end lexical and hybrid search: a lexical-only generation (texts in, BM25 out, docs resolved), then a dense
generation given a lexical configuration (hybrid RRF), replace/delete through tombstones, segment compaction,
refresh on another handle, and the paged segment reader's protocol."""

from __future__ import annotations

import json
import math
import struct

import pytest

from thread_sdk import (
    DocRecord,
    MemoryBlobStore,
    MissingRanges,
    ThreadError,
    _native,
    create_lexical_generation,
    keys,
    open_index,
    rrf,
)
from thread_sdk.lex import LexIo, LexSegment
from thread_sdk.store import BlobTrace

from helpers import build_gen, rng, row, unit_rows

DOCS = [
    (10, "The deployments failed on Tuesday after the budget review", json.dumps({"title": "Deploy failure"})),
    (3, "Retry the deployment, then retry the deployment again", "Retry notes"),
    (1000, "Budget review for the galaxy budget", "Budget"),
    (42, "galaxy galaxy budget", "Galaxy"),
    (7, "", "empty"),
    (99, "Nothing to see here, move along", "Nothing"),
]


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def reference_bm25(docs: list[tuple[int, list[str]]], query: list[str], k1: float = 1.2, b: float = 0.75):
    """bm25s/Lucene arithmetic over the live documents for a tokenized query."""
    n = len(docs)
    avg = sum(len(t) for _, t in docs) / n
    out = []
    for id, toks in docs:
        s = 0.0
        for q in query:
            df = sum(1 for _, t in docs if q in t)
            tf = toks.count(q)
            if not tf:
                continue
            idf = f32(math.log(1 + (n - df + 0.5) / (df + 0.5)))
            norm = k1 * (1 - b + b * len(toks) / avg)
            s = f32(s + f32(idf * (tf / (norm + tf))))
        if s > 0:
            out.append((id, s))
    return sorted(out, key=lambda p: (-p[1], p[0]))


def test_lexical_only_generation():
    store = MemoryBlobStore()
    created = create_lexical_generation(store, "lx", {"text": True, "bucketTerms": 4, "bucketDocs": 2})
    assert created["gen"] == "lx"
    index = open_index(store)
    assert index.gen == "lx"
    assert not index.has_dense
    assert index.lex_config.k1 == 1.2
    with pytest.raises(ThreadError, match="no dense index"):
        index.query([0.0] * 4)
    with pytest.raises(ThreadError, match="no dense index"):
        index.search(text="x", vector=[0.0] * 4)

    # two upserts = two segments; query tokens stem like the documents ("deployment" -> "deploy")
    first, second = DOCS[:3], DOCS[3:]
    up1 = index.upsert([d[0] for d in first], texts=[d[1] for d in first], metas=[d[2] for d in first])
    assert up1.rev == 1
    assert up1.keys["segment"] == "idx/lx/seg/1.thread"
    assert up1.keys["delta"] is None
    assert up1.keys["tombstones"] == "idx/lx/tombstones.1.u32"
    up2 = index.upsert([d[0] for d in second], texts=[d[1] for d in second], metas=[d[2] for d in second],
                       replace=False)
    assert up2.rev == 2
    assert up2.keys["segment"] == "idx/lx/seg/2.thread"
    assert up2.keys["tombstones"] is None
    assert [s["addedRev"] for s in index.segments] == [1, 2]

    assert index.tokenize("Deployment retries") == ["deploy", "retri"]
    res = index.search(text="deployment retries", top_k=5, docs="text")
    assert res.mode == "lexical"
    assert [h.id for h in res.hits] == [3, 10]
    assert res.hits[0].lexical.rank == 1
    assert res.hits[0].doc == DocRecord(3, "Retry notes", DOCS[1][1])
    assert res.hits[1].doc.meta == json.dumps({"title": "Deploy failure"})
    assert res.lexical.tokens == ["deploy", "retri"]
    assert res.lexical.found == 2
    assert res.lexical.stats["segments"] == 2
    assert res.lexical.stats["num_docs"] == 6

    # scores equal the reference over the whole corpus although the documents sit in two segments
    tok = _native.LexTokenizer("{}")
    corpus = [(d[0], tok.tokenize(d[1])) for d in DOCS]
    for q in ["galaxy budget", "deployment", "budget review galaxy nothing", "zebra"]:
        want = reference_bm25(corpus, tok.tokenize(q))[:10]
        got = index.lexical_search(q, 10)
        assert [h.id for h in got.hits] == [w[0] for w in want]
        for h, w in zip(got.hits, want):
            assert h.score == pytest.approx(w[1], abs=1e-6)

    # docs: unknown ids are absent, the empty document has an entry
    d = index.docs([7, 12345, 42], text=True)
    assert d[7] == DocRecord(7, "empty", "")
    assert 12345 not in d
    assert d[42].meta == "Galaxy"

    other = open_index(store, refresh_interval=0)
    assert other.rev == 2
    assert [h.id for h in other.search(text="galaxy").hits] == [42, 1000]

    # replace id 3 with new text and metadata: the old copy is hidden by the tombstone set
    assert index.upsert([3], texts=["Completely different content about zebras"], metas=["Zebra"]).rev == 3
    assert [h.id for h in index.search(text="retry deployment", docs=True).hits] == [10]
    z = index.search(text="zebras", docs=True)
    assert [h.id for h in z.hits] == [3]
    assert z.hits[0].doc.meta == "Zebra"
    index.delete([1000])
    assert [h.id for h in index.search(text="budget").hits] == [42, 10]
    assert index.docs([1000]) == {}
    assert len(index.segments) == 3
    assert len(index.manifest["tombstones"]) == 3

    # compaction: one base segment, same answers, replaced/deleted documents gone for good
    before = index.search(text="budget galaxy deployment zebras", top_k=10, docs="text")
    c = index.compact_segments()
    assert c.merged == 3
    assert c.key == "idx/lx/seg/base.5.thread"
    assert c.count == 5  # 10, 3, 42, 7, 99
    assert index.rev == 5
    assert len(index.segments) == 1 and index.segments[0]["base"] is True
    after = index.search(text="budget galaxy deployment zebras", top_k=10, docs="text")
    assert [(h.id, h.doc.meta, h.doc.text) for h in after.hits] == [(h.id, h.doc.meta, h.doc.text) for h in before.hits]
    # Before the compaction the replaced and deleted copies still counted in N; afterwards the scores equal a fresh
    # index over the live documents (the empty document 7 has no postings and leaves the lexical count).
    assert before.lexical.stats["num_docs"] == 7
    assert after.lexical.stats["num_docs"] == 4
    live = [(10, tok.tokenize(DOCS[0][1])), (3, tok.tokenize("Completely different content about zebras")),
            (42, tok.tokenize(DOCS[3][1])), (99, tok.tokenize(DOCS[5][1]))]
    want = reference_bm25(live, tok.tokenize("budget galaxy deployment zebras"))
    assert [h.id for h in after.hits] == [w[0] for w in want]
    for h, w in zip(after.hits, want):
        assert h.score == pytest.approx(w[1], abs=1e-6)
    # a second compaction is a no-op (one segment, no newer tombstone set)
    c2 = index.compact_segments()
    assert (c2.merged, c2.rev) == (0, 5)
    assert other.refresh() is True
    assert [h.id for h in other.search(text="zebras").hits] == [3]
    # the tokenizer of a generation with segments is frozen; other knobs may change
    with pytest.raises(ThreadError, match="tokenizer"):
        index.configure_lexical({"tokenizer": {"stemmer": None}, "text": True})
    index.configure_lexical(text=True, bucket_terms=64)
    assert index.lex_config.bucket_terms == 64
    assert index.manifest["lex"] == {"text": True, "bucketTerms": 64}
    index.close()
    other.close()
    with pytest.raises(ThreadError, match="closed"):
        index.search(text="x")


def test_rejects_bad_input():
    store = MemoryBlobStore()
    create_lexical_generation(store, "lb", {})
    with pytest.raises(ThreadError, match="exists"):
        create_lexical_generation(store, "lb", {})
    with open_index(store) as index:
        with pytest.raises(ThreadError, match="texts"):
            index.upsert([1], texts=["a", "b"])
        with pytest.raises(ThreadError, match="duplicate"):
            index.upsert([1, 1], texts=["a", "b"])
        with pytest.raises(ThreadError, match="needs"):
            index.upsert([1])
        with pytest.raises(ThreadError, match="no dense index"):
            index.upsert([1], [[1, 0]])
        with pytest.raises(ThreadError, match="text, a vector"):
            index.search()
        with pytest.raises(ThreadError, match="lengths"):
            index.configure_lexical(lengths="fuzzy")
        with pytest.raises(ThreadError, match="stemmer"):
            index.configure_lexical(tokenizer={"stemmer": "klingon"})
        # a query with only unknown terms is empty, not an error; an empty text is rejected
        index.upsert([1], texts=["hello world"])
        assert index.search(text="zzz").hits == []
        with pytest.raises(ThreadError, match="text, a vector"):
            index.search(text="   ")


def test_hybrid_on_dense_generation():
    b = build_gen("gh", 3)
    index = open_index(b.store)
    assert index.lex_config is None
    with pytest.raises(ThreadError, match="configure_lexical"):
        index.search(text="x")
    cfg = index.configure_lexical(lengths="byte", docs=True, text=False,
                                  defaults={"rrf_k": 10, "weights": {"dense": 1, "lexical": 0.5}})
    assert cfg.rev == 1
    assert index.manifest["lex"]["defaults"] == {"rrfK": 10, "weights": {"dense": 1, "lexical": 0.5}}
    assert index.lex_config.lengths == "byte"
    assert index.query(row(b.vectors, 0, 16), nprobe=8).hits[0].id == 0

    # 1000..1003 get vectors and texts; 2000 text only (a lexical-only document in a dense generation)
    fresh = unit_rows(4, b.dim, rng(5))
    up = index.upsert([1000, 1001, 1002, 1003], fresh,
                      texts=["alpha bravo charlie", "alpha bravo", "charlie delta", "echo foxtrot"],
                      metas=["A", "B", "C", "D"], attrs=[1, 2, 3, 4])
    assert up.keys["delta"] == "idx/gh/delta/1.vibd"
    assert up.keys["segment"] == "idx/gh/seg/1.thread"
    assert len(up.cells) == 4
    text_only = index.upsert([2000], texts=["alpha alpha alpha"], metas=["text only"])
    assert text_only.keys["delta"] is None
    assert text_only.keys["segment"] == "idx/gh/seg/2.thread"

    lex = index.search(text="alpha", docs=True)
    assert lex.mode == "lexical"
    assert [h.id for h in lex.hits] == [2000, 1001, 1000]
    assert lex.hits[0].doc.meta == "text only"
    assert lex.hits[0].doc.text is None
    den = index.search(vector=row(fresh, 0, 16), top_k=3, dense={"nprobe": 8})
    assert den.mode == "dense"
    assert den.hits[0].id == 1000 and den.hits[0].dense.rank == 1
    hy = index.search(text="alpha bravo", vector=row(fresh, 1, 16), top_k=5, dense={"nprobe": 8}, docs=True)
    assert hy.mode == "hybrid"
    assert hy.params == {"top_k": 5, "recall_k": 50, "rrf_k": 10, "weights": {"dense": 1, "lexical": 0.5}}
    fused = rrf([(1, [h.id for h in hy.dense.hits]), (0.5, [h.id for h in hy.lexical.hits])], 10, 5)
    assert [(h.id, h.score) for h in hy.hits] == fused
    assert hy.hits[0].id == 1001  # first in both lists
    assert hy.hits[0].dense.rank == 1 and hy.hits[0].lexical.rank == 1
    assert hy.hits[0].doc.meta == "B"
    base_hit = next(h for h in hy.hits if h.id < 1000)
    assert base_hit.doc is None and base_hit.lexical is None
    hy2 = index.search(text="alpha bravo", vector=row(fresh, 1, 16), top_k=3, rrf_k=60, weights={"lexical": 2},
                       recall_k=100, dense={"nprobe": 8})
    assert hy2.params == {"top_k": 3, "recall_k": 100, "rrf_k": 60, "weights": {"dense": 1, "lexical": 2}}

    index.delete([1001])
    assert [h.id for h in index.search(text="alpha bravo").hits] == [1000, 2000]
    assert index.query(row(fresh, 1, 16), top_k=1, nprobe=8).hits[0].id != 1001
    index.upsert([1000], row(fresh, 0, 16), texts=["golf hotel"], metas=["A2"])
    assert [h.id for h in index.search(text="alpha").hits] == [2000]
    golf = index.search(text="golf", vector=row(fresh, 0, 16), docs=True, dense={"nprobe": 8})
    assert golf.hits[0].id == 1000 and golf.hits[0].doc.meta == "A2"

    # compaction of the segments keeps the dense deltas and tombstone sets as they are
    c = index.compact_segments()
    assert c.merged == 3
    assert len(index.manifest["deltas"]) == 2  # the text-only upsert wrote no delta
    assert len(index.manifest["tombstones"]) == 4
    assert [(h.id, h.doc.meta) for h in index.search(text="golf hotel", docs=True).hits] == [(1000, "A2")]
    assert len(index.docs([1001, 2000])) == 1
    index.close()


def test_attribute_filters_lexical_and_hybrid():
    """Filters on the lexical list (resolved through the docs store), hybrid search and docs; a replace rewrites
    the word and a compaction keeps it."""
    b = build_gen("gl", 3)
    with open_index(b.store, cache=False) as index:
        index.configure_lexical(text=True)
        index.configure_attrs(lang={"values": ["en", "fr"]}, public="bool")
        fresh = unit_rows(4, b.dim, rng(8))
        index.upsert([1000, 1001, 1002, 1003], fresh,
                     texts=["alpha bravo", "alpha bravo charlie", "alpha", "alpha bravo delta"],
                     metas=["A", "B", "C", "D"],
                     attrs=[{"lang": "en", "public": True}, {"lang": "fr", "public": True}, {"lang": "en"}, {"lang": "fr"}])
        # a text-only document carries attrs too (they live in the docs store)
        index.upsert([2000], texts=["alpha alpha alpha bravo"], metas=["T"], attrs=[{"lang": "en", "public": True}])
        assert [h.id for h in index.search(text="alpha bravo", top_k=10).hits] == [1000, 2000, 1001, 1003, 1002]
        en = index.search(text="alpha bravo", top_k=10, filter={"lang": "en"}, docs=True)
        assert [h.id for h in en.hits] == [1000, 2000, 1002]
        assert [index.unpack_attrs(h.doc.attrs)["lang"] for h in en.hits] == ["en", "en", "en"]
        assert en.lexical.stats["filter_dropped"] == 2 and en.lexical.stats["depth"] == 40
        assert [h.id for h in index.search(text="alpha", top_k=2, filter={"public": True, "lang": "fr"}).hits] == [1001]
        # the list deepens until k match: with a depth of 1 the first round sees one candidate (1000), which fails
        fr = index.compile_filter({"lang": "fr"})
        deep = index._lex_index().search("alpha bravo", 2, filter=fr, filter_depth=1)
        assert [h.id for h in deep.hits] == [1001, 1003] and deep.stats["depth"] == 8
        assert index._lex_index().search("alpha", 1, filter=fr, filter_depth=1, max_depth=1).hits == []
        assert [h.id for h in index.lexical_search("alpha bravo", 5, filter={"lang": "fr"}).hits] == [1001, 1003]
        # hybrid: both lists filtered, so the fused result holds only matching ids; docs carry the word
        hy = index.search(text="alpha bravo", vector=row(fresh, 0, 16), top_k=5, dense={"nprobe": 8},
                          filter={"lang": "fr"}, docs=True)
        assert hy.mode == "hybrid"
        assert sorted(h.id for h in hy.hits) == [1001, 1003]
        assert all(index.unpack_attrs(h.doc.attrs)["lang"] == "fr" for h in hy.hits)
        assert all(h.id in (1001, 1003) for h in hy.dense.hits)
        assert [h.id for h in hy.lexical.hits] == [1001, 1003]
        pub = index.search(vector=row(fresh, 0, 16), top_k=5, dense={"nprobe": 8}, filter={"public": True})
        assert sorted(h.id for h in pub.hits) == [1000, 1001]
        assert index.docs([1002, 2000])[2000].attrs == index.pack_attrs({"lang": "en", "public": True})
        index.upsert([1002], texts=["alpha"], metas=["C2"], attrs=[{"lang": "fr", "public": True}])
        assert sorted(h.id for h in index.search(text="alpha", top_k=10, filter={"lang": "fr", "public": True}).hits) == [1001, 1002]
        index.compact_segments()
        assert sorted(h.id for h in index.search(text="alpha", top_k=10, filter={"lang": "fr", "public": True}).hits) == [1001, 1002]
        d = index.docs([1002])[1002]
        assert (d.meta, d.attrs) == ("C2", index.pack_attrs({"lang": "fr", "public": True}))


def test_manifest_round_trip():
    b = build_gen("gm", 2)
    with open_index(b.store) as index:
        index.configure_lexical()
        index.upsert([5000], texts=["some text"], metas=["m"])
        m = json.loads(b.store.get(keys.manifest("gm", 2)))
        assert m["lex"] == {}
        seg = m["segments"]
        assert len(seg) == 1 and isinstance(seg[0].pop("bytes"), int)
        assert seg == [{"seq": 1, "key": "idx/gm/seg/1.thread", "count": 1, "lex": True, "docs": True, "text": False,
                        "addedRev": 2}]
        assert m["change"]["op"] == "upsert" and m["change"]["dense"] is False and m["change"]["lexical"] is True
        assert m["nextSeq"] == 2
        meta = json.loads(_native.segment_meta(b.store.get("idx/gm/seg/1.thread")))
        assert meta["format"] == "thread-seg"
        assert meta["tokenizer"]["stemmer"] == "english"
        assert meta["lex"]["num_docs"] == 1 and meta["docs"]["n"] == 1


def test_segment_paging():
    """Opens with a handful of fetches, searches in two rounds per call; MissingRanges is structured."""
    store = MemoryBlobStore()
    job = _native.SegmentJob("{}", 1.2, 0.75, "exact", 4, True, True, 4)
    rows = []
    for i in range(3000):
        rows.append(f"document number {i} talks about topic{i % 170} and theme{i % 50} with word{i} and "
                    f"item{(i * 7) % 1000} filler")
        job.add(i * 3, rows[i], f"meta {i}")
    data = job.finish()
    store.put("seg", data)
    trace = BlobTrace()
    io = LexIo()
    ref = {"seq": 1, "key": "seg", "addedRev": 1, "count": 3000, "bytes": len(data), "lex": True, "docs": True,
           "text": True}
    seg = LexSegment.open(store, ref, io, trace, fill_bytes=0)
    assert seg.has_lex and seg.has_docs
    assert seg.info["num_docs"] == 3000
    assert seg.info["lengths"] == "exact"
    assert seg.info["docs"]["has_text"] is True
    assert io.rounds <= 4
    assert io.bytes < len(data) / 4
    opened = io.bytes
    q = LexIo()
    infos = seg.run(lambda: seg.reader.lookup(["topic3", "theme1", "nope"]), q, trace)
    assert len(infos) == 3 and infos[0][0] > 0 and infos[2][0] == -1
    assert q.rounds == 1
    hits = seg.run(lambda: seg.reader.search(infos, [infos[0][0], infos[1][0], 0], [1, 1, 1], 3000,
                                             seg.info["total_len"], 1.2, 0.75, 5), q, trace)
    assert q.rounds == 2
    assert struct.unpack_from("<I", hits)[0] == 5
    assert q.bytes < len(data) / 4
    q2 = LexIo()
    seg.run(lambda: seg.reader.lookup(["topic3"]), q2, trace)
    assert q2.rounds == 0
    cold = _native.SegmentReader(len(data))
    with pytest.raises(MissingRanges) as e:
        cold.open()
    assert e.value.ranges[0][0] == 0 and e.value.ranges[0][1] > 0
    d = LexIo()
    got = seg.run(lambda: seg.reader.docs_get([3, 4, 8997], True), d, trace)
    assert got[0] == (3, "meta 1", rows[1], 0)
    assert got[1] is None
    assert got[2][1] == "meta 2999"
    assert d.rounds <= 2
    assert seg.held_bytes > opened
