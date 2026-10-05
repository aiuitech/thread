"""An inline manifest in `idx/current.json`: the writers add a copy of the manifest the pointer names (when small),
readers use it instead of the manifest GET at open and on refresh, and fall back to the GET when it is absent or does
not check out. Byte-compatible with the TypeScript SDK's `currentBody`."""

from __future__ import annotations

import json

from thread_sdk import MemoryBlobStore, keys, open_index
from thread_sdk.wire import INLINE_MANIFEST_MAX, current_body, parse_current, parse_current_full

from helpers import build_gen, rng, unit_rows


class Counting(MemoryBlobStore):
    def __init__(self) -> None:
        super().__init__()
        self.reads: list[str] = []

    def get(self, key, range=None, *, trace=None, fill=False):
        self.reads.append(key)
        return super().get(key, range, trace=trace, fill=fill)


def test_body_and_parse():
    m = {"gen": "g", "rev": 3, "k1": 1, "k2": 4, "dim": 8, "vectors": 10}
    body = current_body("g", 3, m)
    assert body == '{"gen":"g","rev":3,"manifest":{"gen":"g","rev":3,"k1":1,"k2":4,"dim":8,"vectors":10}}'
    assert parse_current_full(body.encode()) == ("g", 3, m)
    assert parse_current(body.encode()) == ("g", 3)
    big = {**m, "notes": "x" * INLINE_MANIFEST_MAX}
    assert json.loads(current_body("g", 3, big)) == {"gen": "g", "rev": 3}
    assert current_body("g", 3) == '{"gen":"g","rev":3}'
    for bad in ({**m, "rev": 2}, {**m, "gen": "h"}, {"gen": "g", "rev": 3}, 7):
        assert parse_current_full(json.dumps({"gen": "g", "rev": 3, "manifest": bad}).encode()) == ("g", 3, None)
    m0 = {"gen": "g", "k1": 1, "k2": 4, "dim": 8, "vectors": 10}
    assert parse_current_full(current_body("g", 0, dict(m0)).encode())[2] == {**m0, "rev": 0}


def test_open_uses_the_inline_copy():
    store = Counting()
    b = build_gen("gi", 3, k2=8, n=300, store=store)
    manifest = json.loads(b.store.get(keys.manifest("gi", 0)))
    store.reads = []
    with open_index(store, cache=False) as plain:
        assert keys.manifest("gi", 0) in store.reads
        assert plain.open_stats["manifest"]["reads"] == 1
        store.put(keys.current(), current_body("gi", 0, manifest))
        store.reads = []
        with open_index(store, cache=False) as inl:
            assert keys.manifest("gi", 0) not in store.reads
            assert inl.open_stats["manifest"]["reads"] == 0
            assert inl.manifest == plain.manifest
            q = unit_rows(1, b.dim, rng(2))
            assert [h.id for h in inl.query(q).hits] == [h.id for h in plain.query(q).hits]


def test_writers_inline_and_refresh_skips_the_manifest_get():
    store = Counting()
    b = build_gen("gw", 3, k2=8, n=300, store=store)
    with open_index(store, cache=False) as writer, open_index(store, cache=False, refresh_interval=0) as reader:
        fresh = unit_rows(2, b.dim, rng(4))
        up = writer.upsert([9000, 9001], fresh)
        g, r, m = parse_current_full(store.get(keys.current()))
        assert (g, r) == ("gw", up.rev) and m == writer.manifest
        store.reads = []
        assert reader.refresh()
        assert reader.rev == up.rev
        assert keys.manifest("gw", up.rev) not in store.reads
        assert reader.query(fresh[:b.dim], top_k=1).hits[0].id == 9000
        # a pointer from an older writer (no copy) still works
        store.put(keys.current(), json.dumps({"gen": "gw", "rev": up.rev}))
        with open_index(store, cache=False) as old:
            assert old.open_stats["manifest"]["reads"] == 1 and old.manifest == writer.manifest
