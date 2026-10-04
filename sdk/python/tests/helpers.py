"""Shared by the end-to-end tests: a seeded RNG, synthetic unit vectors and a tiny dense generation built with the
native build jobs (the same stages as the Worker's assign + finalize + publish)."""

from __future__ import annotations

import json
import math
import threading
from array import array
from dataclasses import dataclass

from thread_sdk import MemoryBlobStore, _native, keys
from thread_sdk.wire import manifest_files
from thread_sdk.store import BlobStore


def rng(seed: int):
    """mulberry32, as the TypeScript tests use."""
    s = seed & 0xFFFFFFFF

    def nxt() -> float:
        nonlocal s
        s = (s + 0x6D2B79F5) & 0xFFFFFFFF
        t = s
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return nxt


def gaussian(r) -> float:
    return math.sqrt(-2 * math.log(r() + 1e-12)) * math.cos(2 * math.pi * r())


def unit_rows(n: int, dim: int, r, centers: array | None = None, spread: float = 0.5) -> array:
    out = array("f", bytes(4 * n * dim))
    for i in range(n):
        c = int(r() * (len(centers) // dim)) if centers is not None else -1
        ss = 0.0
        for d in range(dim):
            v = gaussian(r) * (spread if c >= 0 else 1) + (centers[c * dim + d] if c >= 0 else 0)
            out[i * dim + d] = v
            ss += v * v
        inv = 1 / math.sqrt(ss)
        for d in range(dim):
            out[i * dim + d] *= inv
    return out


def row(a: array, i: int, dim: int) -> array:
    return a[i * dim : (i + 1) * dim]


def dot(a: array, b: array, ao: int, bo: int, dim: int) -> float:
    return sum(a[ao + d] * b[bo + d] for d in range(dim))


def brute_force(vectors: array, dim: int, q: array, k: int) -> list[int]:
    n = len(vectors) // dim
    scored = sorted(range(n), key=lambda i: -dot(vectors, q, i * dim, 0, dim))
    return scored[:k]


@dataclass
class Built:
    store: BlobStore
    vectors: array
    dim: int


def build_gen(gen: str, fmt: int, *, dim: int = 16, k2: int = 8, m: int = 2, n: int = 400, legacy: bool = False,
              store: BlobStore | None = None) -> Built:
    """A generation like the Worker's publish, with a slabs table unless `legacy`."""
    r = rng(42)
    fine = unit_rows(k2, dim, r)
    vectors = unit_rows(n, dim, r, fine, 0.6)
    coarse = array("f", [1 / math.sqrt(dim)] * dim)
    store = store if store is not None else MemoryBlobStore()
    job = _native.AssignJob(dim, 1, k2, coarse, fine, 1, m)
    codebook = None
    pq_m = 4
    if fmt == 3:
        assign = array("I", [max(range(k2), key=lambda c: dot(vectors, fine, i * dim, c * dim, dim)) for i in range(n)])
        codebook = _native.pq_init_codebook(_native.residuals_for(vectors, fine, assign, dim), dim, pq_m, 7)
        job.set_codebook(codebook)
    assert job.add_batch(vectors, 0) == n
    part = job.take_part(0)
    pk2, pdim, directory = _native.part_directory(part)
    assert (pk2, pdim) == (k2, dim)
    rows = []
    for cell, (count, off, ln) in enumerate(directory):
        sj = _native.SlabJob(dim, cell, fine[cell * dim : (cell + 1) * dim], fmt)
        if ln:
            assert sj.add_records(part[off : off + ln]) == count
        b = sj.finish()
        info = _native.slab_info(b)
        assert info[4] == fmt
        rows.append([info[5], info[6], info[1], len(b)])
        store.put(keys.slab(gen, cell), b)
    store.put(f"idx/{gen}/coarse.f32", coarse.tobytes())
    store.put(f"idx/{gen}/fine.f32", fine.tobytes())
    if codebook is not None:
        store.put(f"idx/{gen}/pq.bin", codebook)
    if not legacy:
        store.put(f"idx/{gen}/slabs.json", json.dumps(rows))
    manifest = {"gen": gen, "k1": 1, "k2": k2, "dim": dim, "m": m, "probeCoarse": 1, "vectors": n, "format": fmt,
                "tiers": f"pq{pq_m}+sq8" if fmt == 3 else "sq4+sq8" if fmt == 2 else "sq8",
                **({"pq": {"m": pq_m}} if fmt == 3 else {}), "defaults": {"nprobe": 3, "rerank": 20, "pc": 0}}
    store.put(keys.manifest(gen, 0), json.dumps(manifest))
    store.put(keys.current(), json.dumps({"gen": gen} if legacy else {"gen": gen, "rev": 0}))
    return Built(store, vectors, dim)


def add_routing_rev(store, gen: str, *, tables=("f16", "i8"), stale_fine: bool = False, tree_groups: int = 0,
                    grouped_elem: str = "i8") -> int:
    """A revision naming fine.f16 / fine.i8 (and, with `tree_groups`, a centroid tree and the tree-grouped table)
    made from the generation's means (what publish-rev writes)."""
    head = json.loads(bytes(store.get(keys.current()))).get("rev", 0)
    m = json.loads(bytes(store.get(keys.manifest(gen, head))))
    files = manifest_files(m)
    fine = bytes(store.get(files["fine"]))
    rev = head + 1
    routing = {"fine": f"idx/{gen}/fine.0.f32" if stale_fine else files["fine"]}
    for elem in tables:
        b = _native.routing_table_encode(fine, m["dim"], elem)
        key = keys.routing_table(files["fine"], elem)
        store.put(key, b)
        routing[elem] = {"key": key, "bytes": len(b)}
    extra = {}
    if tree_groups:
        tree = _native.build_centroid_tree(fine, m["dim"], tree_groups, 5, 7)
        tkey = f"idx/{gen}/tree.{rev}.bin"
        store.put(tkey, tree)
        g = _native.grouped_table_encode(fine, m["dim"], tree, grouped_elem)
        _, _, _, groups, rows, data_off, row_bytes, size = _native.grouped_table_info(g)
        gkey = keys.grouped_table(gen, rev, grouped_elem)
        store.put(gkey, g)
        routing["grouped"] = {"key": gkey, "tree": tkey, "elem": grouped_elem, "groups": groups, "rows": rows,
                              "dataOff": data_off, "rowBytes": row_bytes, "bytes": size}
        extra["tree"] = {"key": tkey, "groups": tree_groups, "beam": 2}
    store.put(keys.manifest(gen, rev), json.dumps({**m, "rev": rev, "parentRev": head, "files": files,
                                                   "routing": routing, **extra}))
    store.put(keys.current(), json.dumps({"gen": gen, "rev": rev}))
    return rev


class Gated(MemoryBlobStore):
    """A memory store whose reads of `key` wait until `release` is set (to hold the full routing table back)."""

    def __init__(self, inner: MemoryBlobStore, key: str) -> None:
        super().__init__()
        self.objects = inner.objects
        self.key = key
        self.release = threading.Event()
        self.reads: list[str] = []

    def get(self, key, range=None, *, trace=None, fill=False):
        self.reads.append(key)
        if key == self.key:
            self.release.wait(10)
        return super().get(key, range, trace=trace, fill=fill)
