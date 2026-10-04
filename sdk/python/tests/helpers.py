"""Shared by the end-to-end tests: a seeded RNG, synthetic unit vectors and a tiny dense generation built with the
native build jobs (the same stages as the Worker's assign + finalize + publish)."""

from __future__ import annotations

import json
import math
from array import array
from dataclasses import dataclass

from thread_sdk import MemoryBlobStore, _native, keys
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
              store: BlobStore | None = None, attrs=None) -> Built:
    """A generation like the Worker's publish, with a slabs table unless `legacy`. With `attrs` (a word per vector
    index), the slabs are built the way a compaction builds them, from a delta segment's records (one primary cell
    per vector, so `m` is 1), since the assign job writes no attrs."""
    if attrs is not None:
        m = 1
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
    if attrs is not None:
        dj = _native.DeltaJob(dim, k2, fine, 1)
        if codebook is not None:
            dj.set_pq_codebook(_native.PqCodebook(codebook))
        assert len(dj.add(vectors, array("I", range(n)), array("I", [attrs(i) for i in range(n)]))) == n
        delta = dj.finish()
        records = [_native.delta_records_for_cell(delta, cell, None) for cell in range(k2)]
    else:
        assert job.add_batch(vectors, 0) == n
        part = job.take_part(0)
        pk2, pdim, directory = _native.part_directory(part)
        assert (pk2, pdim) == (k2, dim)
        records = [part[off : off + ln] for _, off, ln in directory]
    rows = []
    for cell, recs in enumerate(records):
        sj = _native.SlabJob(dim, cell, fine[cell * dim : (cell + 1) * dim], fmt)
        if len(recs):
            assert sj.add_records(recs) > 0
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
