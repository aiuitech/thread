"""Cross-implementation check against the TypeScript SDK (needs `node` and a built `sdk/ts/dist`, else skipped):

1. the TS SDK builds a generation on disk and writes to it (deltas, lexical segments, replacements, deletes); Python
   opens the same directory and must return the TS results for every recorded query;
2. Python writes on top (upserts with vectors and texts, a delete, a segment compaction); the TS SDK opens the
   result and must return what Python returns;
3. both SDKs pack that revision into one file: the bytes must be identical, and each SDK must answer the same
   queries from the other's pack.

Ids, their order, attrs, payloads, docs, tokens and BM25 scores must match exactly. Cosine scores may differ in the
last float32 ulps: the wasm build accumulates dot products in 16 SIMD lanes, the native build in 8 scalar partial
sums (see thread-core `math.rs`), so they are compared within 1e-5.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from thread_sdk import FileBlobStore, PackedBlobStore, open_index, pack_index

from helpers import rng, unit_rows

HERE = Path(__file__).parent
SCRIPT = HERE / "interop" / "ts_side.mjs"
DIST = HERE.parent.parent / "ts" / "dist" / "index.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None or not DIST.exists(),
                                reason="needs node and a built sdk/ts/dist (cd sdk/ts && npm run build)")


def node(cmd: str, d: Path, *extra: str) -> str:
    r = subprocess.run(["node", str(SCRIPT), cmd, str(d), *extra], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return r.stdout


def answer(index, req: dict):
    """The Python side of ts_side.mjs `answer`, in the same JSON shape."""
    if req["kind"] == "query":
        r = index.query(req["vector"], top_k=req["topK"], nprobe=req["nprobe"], rerank=req["rerank"],
                        return_payload=True)
        return [[h.id, h.score, h.attrs, h.source, h.payload.decode()] for h in r.hits]
    if req["kind"] == "search":
        r = index.search(text=req.get("text"), vector=req.get("vector"), top_k=req["topK"],
                         dense={"nprobe": req["nprobe"]}, docs="text")
        return {"mode": r.mode, "hits": [[h.id, h.score, [h.doc.meta, h.doc.text] if h.doc else None] for h in r.hits]}
    if req["kind"] == "lexical":
        r = index.lexical_search(req["text"], req["topK"])
        return {"tokens": r.tokens, "hits": [[h.id, h.score] for h in r.hits], "numDocs": r.stats["num_docs"]}
    raise AssertionError(req["kind"])


def same(a, b, path="") -> None:
    """Structural equality, floats within 1e-5."""
    if isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-5, abs=1e-6), path
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], f"{path}.{k}")
    else:
        assert a == b, path


def test_typescript_and_python_agree(tmp_path):
    node("build", tmp_path)
    ts = json.loads((tmp_path / "ts-results.json").read_text())
    store = FileBlobStore(tmp_path)

    # 1. Python reads what TypeScript wrote and answers exactly the same.
    with open_index(store) as index:
        assert (index.gen, index.rev) == (ts["gen"], ts["rev"])
        assert len(index.manifest["deltas"]) == 3 and len(index.segments) == 4 and len(index.manifest["tombstones"]) == 5
        for req, want in zip(ts["requests"], ts["results"]):
            got = json.loads(json.dumps(answer(index, req)))
            same(got, want, json.dumps(req)[:80])

        # 2. Python writes on top; TypeScript reads it back.
        fresh = unit_rows(30, index.dim, rng(123))
        ids = list(range(30000, 30030))
        index.upsert(ids, fresh, texts=[f"python wrote {i} galaxy item{i % 5}" for i in ids],
                     metas=[f"py{i}" for i in ids], attrs=[i % 11 for i in ids], payloads=[f"pp{i}" for i in ids])
        index.upsert([10003, 20000], [list(fresh[0:32]), list(fresh[32:64])], texts=["replaced by python", None],
                     metas=["py-r1", "py-r2"])
        index.delete([30001, 10004])
        c = index.compact_segments()
        assert c.merged == 6
        index.upsert([40000], texts=["text only from python galaxy"], metas=["py-text"])
        requests = []
        qr = rng(77)
        for i in range(8):
            requests.append({"kind": "query", "vector": list(unit_rows(1, index.dim, qr)), "topK": 10, "nprobe": 3 + i,
                             "rerank": 40})
        for i in (0, 1, 2, 29):
            requests.append({"kind": "query", "vector": list(fresh[i * 32 : (i + 1) * 32]), "topK": 5, "nprobe": 4,
                             "rerank": 40})
        for t in ("python galaxy", "replaced", "galaxy budget item3", "text only"):
            requests.append({"kind": "lexical", "text": t, "topK": 10})
        requests.append({"kind": "search", "text": "python item2", "vector": list(fresh[64:96]), "topK": 8,
                         "nprobe": 5})
        requests.append({"kind": "search", "text": "galaxy", "vector": None, "topK": 6, "nprobe": 4})
        mine = [json.loads(json.dumps(answer(index, r))) for r in requests]
        rev = index.rev
    (tmp_path / "requests.json").write_text(json.dumps({"requests": requests}))
    theirs = json.loads(node("run", tmp_path))
    assert theirs["rev"] == rev
    for req, a, b in zip(requests, mine, theirs["results"]):
        same(a, b, json.dumps(req)[:80])

    # 3. One file. Both SDKs pack the revision: same bytes; each answers from the other's pack.
    py_pack, ts_pack = tmp_path / "py.threadpack", tmp_path / "ts.threadpack"
    r = pack_index(store, py_pack)
    assert (r.gen, r.rev) == (theirs["gen"], rev)
    t = json.loads(node("pack", tmp_path, str(ts_pack)))
    assert (t["gen"], t["rev"], t["count"], t["bytes"]) == (r.gen, r.rev, r.count, r.bytes)
    assert py_pack.read_bytes() == ts_pack.read_bytes()
    with open_index(ts_pack) as packed:
        assert isinstance(packed.store, PackedBlobStore) and packed.rev == rev
        for req, a in zip(requests, mine):
            same(json.loads(json.dumps(answer(packed, req))), a, json.dumps(req)[:80])
    from_py_pack = json.loads(node("run", tmp_path, str(py_pack)))
    assert from_py_pack["rev"] == rev
    for req, a, b in zip(requests, mine, from_py_pack["results"]):
        same(a, b, json.dumps(req)[:80])
