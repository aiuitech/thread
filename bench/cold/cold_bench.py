#!/usr/bin/env python3
"""Cold-start bench of the Python SDK against S3: real R2 S3, the fake_s3.mjs proxy, or a pack served by the proxy.
The same bench as bench/cold/cold_bench.mjs (TypeScript SDK): same CLI, same scenarios, same JSONL schema, same table.

    sdk/python/.venv/bin/python bench/cold/cold_bench.py --store r2s3|proxy|pack [--n 10] [--queries 0]
        [--nprobe N] [--rerank R] [--proxy http://127.0.0.1:9555] [--pack-key packs/g4-cold.threadpack]
        [--label TEXT] [--out data/cold/NAME.jsonl] [--env sdk/ts/examples/basic/.env.secrets]
        store / cache options (absent = the SDK's defaults), the same as cold_bench.mjs:
        [--legacy] (the pre-cold-store defaults: MemoryCache with "whole" misses, inflight 12, no warm-up, no base pack)
        [--cache none|memory|disk] [--disk-dir data/cold-store/dc-{t}] [--miss-policy whole|range|range+warm]
        [--inflight N] [--coalesce BYTES] [--span BYTES] [--chunk BYTES] [--warmup N|false] [--cold-nprobe N] [--cold-rerank R]
        [--drain] (wait for background cache warms before the `warm` scenario)
        open options: [--routing f32|f16|i8|auto] [--cold-routing tree] [--cold-beam N] [--base-pack 0|1] [--gen G --rev R]

The open options are open_index's (`routing`, `cold_routing`, `cold_beam`, `base_pack`, `gen`/`rev`), as in
cold_bench.mjs.

Each trial t spawns a fresh interpreter (fresh connections, fresh native module import, fresh index, empty caches):
  process-cold  query (offset + 2t): import thread_sdk, open the index, one query
  slab-cold     query (offset + 2t + 1), same process: index open, base loaded, slabs not read yet
  warm          query (offset + 2t + 1) again
The index is opened with `open_index(store)` and no cache argument: what a Python user gets by default, a 256 MB
MemoryCache in front of a remote store (so a phase-1 miss reads the whole slab once, and `warm` reads nothing).
"""

from __future__ import annotations

import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
import urllib.request
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCHEMA = "thread-cold/1"
SCENARIOS = ["process-cold", "slab-cold", "warm"]


def parse_args(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            v = argv[i + 1] if i + 1 < len(argv) else None
            if v is None or v.startswith("--"):
                out[a[2:]] = "true"
            else:
                out[a[2:]] = v
                i += 1
        i += 1
    return out


ARGS = parse_args(sys.argv[1:])
STORE = ARGS.get("store", "proxy")
if STORE not in ("r2s3", "proxy", "pack"):
    raise SystemExit(f"--store must be r2s3, proxy or pack, not {STORE}")
N = int(ARGS.get("n", 10))
OFFSET = int(ARGS.get("queries", 0))
PROXY = ARGS.get("proxy", "http://127.0.0.1:9555")
PACK_KEY = ARGS.get("pack-key", "packs/g4-cold.threadpack")
ENV_FILE = ARGS.get("env", "sdk/ts/examples/basic/.env.secrets")
NPROBE = int(ARGS["nprobe"]) if "nprobe" in ARGS else None
RERANK = int(ARGS["rerank"]) if "rerank" in ARGS else None


def _num(k: str) -> int | None:
    return int(ARGS[k]) if k in ARGS else None


LEGACY = ARGS.get("legacy") == "true"
#: The store / cache / query configuration under test (recorded in every row).
CONFIG: dict[str, Any] = {
    "legacy": LEGACY,
    "cache": ARGS.get("cache", "memory" if LEGACY else "default"),
    "diskDir": ARGS.get("disk-dir"),
    "missPolicy": ARGS.get("miss-policy", "whole" if LEGACY else None),
    "inflight": _num("inflight") if "inflight" in ARGS else (12 if LEGACY else None),
    "coalesceBytes": _num("coalesce") if "coalesce" in ARGS else (4096 if LEGACY else None),
    "spanBytes": _num("span"),
    "chunkBytes": _num("chunk") if "chunk" in ARGS else (0 if LEGACY else None),
    "warmup": 0 if ARGS.get("warmup") == "false" else (_num("warmup") if "warmup" in ARGS else (0 if LEGACY else None)),
    "coldPolicy": ({**({"nprobe": _num("cold-nprobe")} if "cold-nprobe" in ARGS else {}),
                    **({"rerank": _num("cold-rerank")} if "cold-rerank" in ARGS else {})}
                   if "cold-nprobe" in ARGS or "cold-rerank" in ARGS else None),
    "drain": ARGS.get("drain") == "true",
}


def load_env(file: str) -> dict[str, str]:
    out = {}
    with open(os.path.join(ROOT, file)) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


class Dataset:
    """Query vectors and top-k ground truth from data/prep (see bench/prep/prep_dataset.py)."""

    def __init__(self) -> None:
        with open(os.path.join(ROOT, "data/prep/manifest.json")) as f:
            self.dim = json.load(f)["dim"]
        self.q = open(os.path.join(ROOT, "data/prep/queries.f32"), "rb").read()
        self.gt = open(os.path.join(ROOT, "data/prep/groundtruth.u32"), "rb").read()
        self.total = len(self.q) // (self.dim * 4)
        self.gtk = len(self.gt) // (self.total * 4)

    def vector(self, i: int) -> bytes:
        return self.q[i * self.dim * 4 : (i + 1) * self.dim * 4]  # raw little-endian f32: the SDK takes bytes

    def truth(self, i: int, k: int = 10) -> list[int]:
        n = min(k, self.gtk)
        return list(struct.unpack_from(f"<{n}I", self.gt, i * self.gtk * 4))


def recall_at(ids: list[int], truth: list[int]) -> float:
    t = set(truth)
    return sum(1 for i in ids[: len(truth)] if i in t) / len(truth)


def proxy_call(method: str, path: str) -> dict[str, Any]:
    req = urllib.request.Request(PROXY + path, method=method, data=b"" if method == "POST" else None)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def proxy_stats(s: dict[str, Any] | None) -> dict[str, Any] | None:
    if not s:
        return None
    return {k: s[k] for k in ("requests", "bytes", "newConnections", "upstreamFetches", "byStatus")}


def r1(x: float | None) -> float | None:
    return None if x is None else round(x, 1)


# ---------- child: one trial ----------


def open_opts() -> dict[str, Any]:
    """open_index options from the command line."""
    o: dict[str, Any] = {}
    if "routing" in ARGS:
        o["routing"] = ARGS["routing"]
    if "cold-routing" in ARGS:
        o["cold_routing"] = ARGS["cold-routing"]
    if "cold-beam" in ARGS:
        o["cold_beam"] = int(ARGS["cold-beam"])
    if "base-pack" in ARGS:
        o["base_pack"] = ARGS["base-pack"] != "0"
    elif LEGACY:
        o["base_pack"] = False
    if "gen" in ARGS:
        o["gen"] = ARGS["gen"]
    if "rev" in ARGS:
        o["rev"] = int(ARGS["rev"])
    return o


def child() -> None:
    t_entry = time.time() * 1000
    spawn_at = float(ARGS["spawn-at"])
    trial = int(ARGS["trial"])
    use_proxy = STORE != "r2s3"
    if use_proxy:
        proxy_call("POST", "/__reset")
    ms: dict[str, Any] = {"boot": round(t_entry - spawn_at)}
    t = time.perf_counter()
    import thread_sdk  # noqa: PLC0415 - timed on purpose
    from thread_sdk import DiskCache, MemoryCache, PackedBlobStore, S3BlobStore, open_index  # noqa: PLC0415

    ms["import"] = (time.perf_counter() - t) * 1000
    env = load_env(ENV_FILE)
    ds = Dataset()
    kw = {k: v for k, v in (("inflight", CONFIG["inflight"]), ("coalesce_bytes", CONFIG["coalesceBytes"]),
                              ("span_bytes", CONFIG["spanBytes"]),
                              ("chunk_bytes", CONFIG["chunkBytes"])) if v is not None}
    if use_proxy:
        s3 = S3BlobStore(PROXY, env.get("S3_BUCKET", "vib-bench"), "fake", "fake", **kw)
    else:
        s3 = S3BlobStore(env["S3_ENDPOINT"], env["S3_BUCKET"], env["S3_ACCESS_KEY_ID"], env["S3_SECRET_ACCESS_KEY"], **kw)
    store = PackedBlobStore(s3, PACK_KEY) if STORE == "pack" else s3
    # No options: the default for a remote store (a 256 MB MemoryCache with "range+warm" misses).
    okw: dict[str, Any] = {}
    c = CONFIG["cache"]
    if c == "none":
        okw["cache"] = False
    elif c == "memory":
        okw["cache"] = MemoryCache()
    elif c == "disk":
        okw["cache"] = DiskCache(os.path.join(ROOT, (CONFIG["diskDir"] or "data/cold-store/dc-{t}").replace("{t}", str(trial))))
    if CONFIG["missPolicy"]:
        okw["miss_policy"] = CONFIG["missPolicy"]
    if CONFIG["warmup"] is not None:
        okw["warmup"] = CONFIG["warmup"]
    if CONFIG["coldPolicy"]:
        okw["cold_policy"] = CONFIG["coldPolicy"]
    index = open_index(store, **okw, **open_opts())
    os_ = index.open_stats
    qopts: dict[str, Any] = {"top_k": 10}
    if NPROBE is not None:
        qopts["nprobe"] = NPROBE
    if RERANK is not None:
        qopts["rerank"] = RERANK

    def emit(row: dict[str, Any]) -> None:
        sys.stdout.write("@@ROW " + json.dumps(row) + "\n")
        sys.stdout.flush()

    def run(scenario: str, qi: int, with_open: bool) -> None:
        t0 = time.perf_counter()
        res = index.query(ds.vector(qi), **qopts)
        wall = (time.perf_counter() - t0) * 1000
        e2e = round(time.time() * 1000 - spawn_at) if with_open else None
        proxy = proxy_stats(proxy_call("POST", "/__reset")) if use_proxy else None
        st = res.stats
        tl = st["timeline"]
        ids = [h.id for h in res.hits]
        pack = os_.get("pack")
        open_phases = [pack, os_.get("prefetch"), os_["current"], os_["manifest"], os_["slabs"], os_["base"]] \
            if with_open else []

        def osum(f: str) -> int:
            return sum(p[f] for p in open_phases if p)

        def o(v: Any) -> Any:
            return v if with_open else None

        emit({
            "kind": "trial",
            "schema": SCHEMA,
            "sdk": "py",
            "store": STORE,
            "label": ARGS.get("label"),
            "trial": trial,
            "scenario": scenario,
            "query": qi,
            "gen": st["gen"],
            "rev": st["rev"],
            "nprobe": res.params["nprobe"],
            "rerank": res.params["rerank"],
            "recall": recall_at(ids, ds.truth(qi)),
            "ids": ids,
            "routing": st["routing"],
            "routingTable": st.get("routing_table"),
            "cells": st["cells"],
            "coldRouting": st.get("cold_routing"),
            "prefetchSource": os_["prefetch"]["source"] if with_open and os_.get("prefetch") else None,
            "ms": {
                "boot": o(ms["boot"]),
                "import": o(r1(ms["import"])),
                "wasmCompile": None,
                "wasmInit": None,
                "pack": r1(pack["ms"]) if with_open and pack else None,
                "prefetch": r1(os_["prefetch"]["ms"]) if with_open and os_.get("prefetch") else None,
                "current": o(r1(os_["current"]["ms"])),
                "manifest": o(r1(os_["manifest"]["ms"])),
                "slabs": o(r1(os_["slabs"]["ms"])),
                "base": o(r1(os_["base"]["ms"])),
                "baseFetch": o(r1(os_["base"]["fetch"])),
                "baseBuild": o(r1(os_["base"]["build"])),
                "open": o(r1(os_["total"] + (pack["ms"] if pack else 0))),
                "route": r1(tl["route"]["ms"]),
                "phase1": r1(tl["phase1"]["ms"]),
                "phase2": r1(tl["phase2"]["ms"]),
                "query": r1(wall),
                "e2e": e2e,
            },
            "reqs": {
                "pack": pack["reads"] if with_open and pack else None,
                "prefetch": os_["prefetch"]["reads"] if with_open and os_.get("prefetch") else None,
                "current": o(os_["current"]["reads"]),
                "manifest": o(os_["manifest"]["reads"]),
                "slabs": o(os_["slabs"]["reads"]),
                "base": o(os_["base"]["reads"]),
                "open": osum("reads") if with_open else None,
                "route": tl["route"]["reads"],
                "phase1": tl["phase1"]["reads"],
                "phase2": tl["phase2"]["reads"],
                "query": st["io"]["reads"],
            },
            "bytes": {
                "pack": pack["bytes"] if with_open and pack else None,
                "prefetch": os_["prefetch"]["bytes"] if with_open and os_.get("prefetch") else None,
                "current": o(os_["current"]["bytes"]),
                "manifest": o(os_["manifest"]["bytes"]),
                "slabs": o(os_["slabs"]["bytes"]),
                "base": o(os_["base"]["bytes"]),
                "open": osum("bytes") if with_open else None,
                "route": tl["route"]["bytes"],
                "phase1": tl["phase1"]["bytes"],
                "phase2": tl["phase2"]["bytes"],
                "query": st["io"]["bytes"],
            },
            "config": CONFIG,
            "stats": {"wholeSlabs": st["whole_slabs"], "warms": st["io"].get("warms", 0), "inflight": st["inflight"],
                      "coalesceBytes": st["coalesce_bytes"], "spanBytes": st["span_bytes"], "cold": st.get("cold"),
                      "scanned": st["scanned"], "scanBytes": st["scan_bytes"], "slabBytes": st["slab_bytes"],
                      "rerankSlabs": st["rerank_slabs"], "rerankReads": st["rerank_reads"],
                      "rerankBytes": st["rerank_bytes"], "rerankLocal": st["rerank_local"],
                      "specRecords": st["spec_records"], "specUsed": st["spec_used"], "cacheHits": st["io"]["hits"],
                      "cacheMisses": st["io"]["misses"], "baseFiles": o(os_["base"]["files"])},
            "proxy": proxy,
        })

    qa = OFFSET + 2 * trial
    run("process-cold", qa, True)
    run("slab-cold", qa + 1, False)
    if CONFIG["drain"] and hasattr(index.store, "drain"):
        index.store.drain(60)
        if use_proxy:
            proxy_call("POST", "/__reset")  # background warm traffic is not the warm query's
    run("warm", qa + 1, False)
    index.close()
    _ = thread_sdk


# ---------- summary (same table as lib.mjs) ----------

MB = 1048576


def median(xs: list[float]) -> float | None:
    s = sorted(xs)
    n = len(s)
    if not n:
        return None
    return s[(n - 1) // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def pct(xs: list[float], q: float) -> float | None:
    s = sorted(xs)
    if not s:
        return None
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def _g(*path: str):
    def get(r: dict[str, Any]) -> Any:
        v: Any = r
        for p in path:
            if v is None:
                return None
            v = v.get(p)
        return v
    return get


METRICS = [
    ("ms boot (spawn -> main)", _g("ms", "boot"), "ms"),
    ("ms import SDK", _g("ms", "import"), "ms"),
    ("ms wasm compile", _g("ms", "wasmCompile"), "ms"),
    ("ms wasm init", _g("ms", "wasmInit"), "ms"),
    ("ms open.pack", _g("ms", "pack"), "ms"),
    ("ms open.current", _g("ms", "current"), "ms"),
    ("ms open.manifest", _g("ms", "manifest"), "ms"),
    ("ms open.slabs", _g("ms", "slabs"), "ms"),
    ("ms open.base", _g("ms", "base"), "ms"),
    ("ms   base fetch", _g("ms", "baseFetch"), "ms"),
    ("ms   base build", _g("ms", "baseBuild"), "ms"),
    ("ms open total", _g("ms", "open"), "ms"),
    ("ms route", _g("ms", "route"), "ms"),
    ("ms phase1", _g("ms", "phase1"), "ms"),
    ("ms phase2", _g("ms", "phase2"), "ms"),
    ("ms query total", _g("ms", "query"), "ms"),
    ("ms process e2e (spawn -> hits)", _g("ms", "e2e"), "ms"),
    ("req open (pack+cur+man+slabs+base)", _g("reqs", "open"), "n"),
    ("req phase1", _g("reqs", "phase1"), "n"),
    ("req phase2", _g("reqs", "phase2"), "n"),
    ("req query", _g("reqs", "query"), "n"),
    ("MB open", _g("bytes", "open"), "mb"),
    ("MB phase1", _g("bytes", "phase1"), "mb"),
    ("MB phase2", _g("bytes", "phase2"), "mb"),
    ("MB query", _g("bytes", "query"), "mb"),
    ("cache hits (query)", _g("stats", "cacheHits"), "n"),
    ("scan prefix MB", _g("stats", "scanBytes"), "mb"),
    ("slab MB transferred", _g("stats", "slabBytes"), "mb"),
    ("rerank reads", _g("stats", "rerankReads"), "n"),
    ("rerank local / spec used", lambda r: (r["stats"].get("rerankLocal") or 0) + (r["stats"].get("specUsed") or 0), "n"),
    ("proxy requests", _g("proxy", "requests"), "n"),
    ("proxy MB", _g("proxy", "bytes"), "mb"),
    ("proxy new connections", _g("proxy", "newConnections"), "n"),
    ("proxy upstream fetches (want 0)", _g("proxy", "upstreamFetches"), "n"),
    ("recall@10", _g("recall"), "pct"),
]


def fmt_val(v: Any, kind: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if kind == "ms":
        return f"{v:.1f}"
    if kind == "mb":
        return f"{v / MB:.2f}"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(header)]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(cells))

    return "\n".join([line(header), line(["-" * n for n in w]), *(line(r) for r in rows)])


def summary(rows: list[dict[str, Any]]) -> str:
    trials = [r for r in rows if r.get("kind") == "trial"]
    by = {s: [r for r in trials if r["scenario"] == s] for s in SCENARIOS}
    out = []
    for label, get, kind in METRICS:
        cells: list[str] = []
        any_ = False
        for s in SCENARIOS:
            xs = [v for v in (get(r) for r in by[s]) if v is not None]
            any_ = any_ or bool(xs)
            if kind == "pct":
                cells += [fmt_val(sum(xs) / len(xs) if xs else None, kind), fmt_val(min(xs) if xs else None, kind)]
            else:
                cells += [fmt_val(median(xs), kind), fmt_val(pct(xs, 0.9), kind)]
        if any_:
            out.append([label, *cells])
    n = "/".join(str(len(by[s])) for s in SCENARIOS)
    header = [f"metric (n={n})"] + [c for s in SCENARIOS for c in (f"{s} med", "p90")]
    return _table(header, out) + "\n(recall@10 columns: mean, min)"


# ---------- parent ----------


def run_child(trial: int) -> list[dict[str, Any]]:
    argv = []
    skip = False
    for i, a in enumerate(sys.argv[1:]):
        if skip:
            skip = False
            continue
        if a == "--out":
            skip = True
            continue
        argv.append(a)
    cmd = [sys.executable, os.path.abspath(__file__), *argv, "--child", "--trial", str(trial),
           "--spawn-at", str(time.time() * 1000)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise SystemExit(f"trial {trial}: child exited with {p.returncode}")
    return [json.loads(line[6:]) for line in p.stdout.splitlines() if line.startswith("@@ROW ")]


def sign_microbench(cached: bool = False) -> int:
    """SigV4 cost per request (us), deriving the signing key every time or with a `SigningKeyCache`."""
    sys.path.insert(0, os.path.join(ROOT, "sdk/python"))
    from thread_sdk import s3 as s3mod  # noqa: PLC0415
    from thread_sdk.s3 import EMPTY_SHA256, amz_date_of, sign_v4  # noqa: PLC0415

    extra: dict[str, Any] = {}
    if cached and hasattr(s3mod, "SigningKeyCache"):
        extra["keys"] = s3mod.SigningKeyCache("secret")
    n = 300
    t0 = time.perf_counter()
    for i in range(n):
        sign_v4(method="GET", path="/vib-bench/idx/g4/slab/01234.179", headers={"range": f"bytes=0-{1000 + i}"},
                payload_hash=EMPTY_SHA256, host="x.r2.cloudflarestorage.com", region="auto",
                access_key_id="AKIDEXAMPLE", secret_access_key="secret", amz_date=amz_date_of(time.time()), **extra)
    return round((time.perf_counter() - t0) / n * 1e6)


def parent() -> None:
    proxy_config = proxy_call("GET", "/__config") if STORE != "r2s3" else None
    env = load_env(ENV_FILE)
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        git = None
    from urllib.parse import urlsplit  # noqa: PLC0415

    meta = {
        "kind": "meta",
        "schema": SCHEMA,
        "sdk": "py",
        "runtime": f"python {platform.python_version()}",
        "store": STORE,
        "endpoint": urlsplit(env["S3_ENDPOINT"]).netloc if STORE == "r2s3" else PROXY,
        "packKey": PACK_KEY if STORE == "pack" else None,
        "proxyConfig": proxy_config,
        "cache": CONFIG["cache"],
        "config": CONFIG,
        "n": N,
        "queries": {"offset": OFFSET, "used": f"{OFFSET}..{OFFSET + 2 * N - 1}"},
        "nprobe": NPROBE if NPROBE is not None else "manifest",
        "rerank": RERANK if RERANK is not None else "manifest",
        "signUsPerRequest": sign_microbench(),
        "signUsPerRequestCached": sign_microbench(True),
        "label": ARGS.get("label"),
        "open": open_opts(),
        "git": git,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rows: list[dict[str, Any]] = [meta]
    out = os.path.join(ROOT, ARGS["out"]) if "out" in ARGS else None
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(json.dumps(meta) + "\n")
    pc = f" (rtt {proxy_config['rttMs']} ms, {proxy_config['mbps']} Mbps, conn +{proxy_config['connMs']} ms, jitter {proxy_config['jitterMs']} ms" + (f", link {proxy_config['linkMbps']} Mbps shared" if proxy_config.get('linkMbps') else "") + ")" if proxy_config else ""
    print(f"cold_bench py store={STORE} {meta['endpoint']}{pc} n={N} queries {meta['queries']['used']}  "
          f"sign_v4 {meta['signUsPerRequest']} us/request ({meta['signUsPerRequestCached']} with the key cache)  "
          f"config {json.dumps(CONFIG)}", flush=True)
    for t in range(N):
        got = run_child(t)
        rows += got
        if out:
            with open(out, "a") as f:
                for r in got:
                    f.write(json.dumps(r) + "\n")
        p, s, w = got
        reqs = p["proxy"]["requests"] if p["proxy"] else p["reqs"]["open"] + p["reqs"]["query"]
        warn = "  WARNING: proxy fetched from upstream (not warmed up)" if any((r["proxy"] or {}).get("upstreamFetches") for r in got) else ""
        print(f"  trial {t}: process-cold e2e {p['ms']['e2e']} ms (open {p['ms']['open']}, query {p['ms']['query']}, "
              f"{reqs} req)  slab-cold {s['ms']['query']} ms  warm {w['ms']['query']} ms  "
              f"recall {p['recall']}/{s['recall']}{warn}", flush=True)
    print("\n" + summary(rows))
    if out:
        print(f"\nwrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    if ARGS.get("child") == "true":
        child()
    else:
        parent()
