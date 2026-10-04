#!/usr/bin/env python3
"""The Python child of bench/cold/refresh_bench.mjs (one trial: open, switch current.json, measure)."""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from cold_bench import ARGS, Dataset  # noqa: E402

from thread_sdk import S3BlobStore, open_index  # noqa: E402


def main() -> None:
    proxy, overlay, mode, trial = ARGS["proxy"], ARGS["overlay"], ARGS["mode"], int(ARGS["trial"])
    a, b = (int(x) for x in ARGS.get("revs", "179,180").split(","))
    cur = os.path.join(overlay, "idx/current.json")
    ds = Dataset()
    ix = open_index(S3BlobStore(proxy, "vib-bench", "f", "f"), refresh_interval=1.0)
    q = ds.vector(20 + 2 * trial)
    ix.query(q)
    ix.store.drain(30)
    with open(cur, "w") as f:
        json.dump({"gen": "g4", "rev": b}, f)
    time.sleep(1.1)
    out: dict = {"mode": mode, "trial": trial}
    t = time.perf_counter()
    if mode == "blocking":
        ix.refresh()
        r = ix.query(q)
    else:
        r = ix.query(q)
    out["trigger"] = (time.perf_counter() - t) * 1000
    out["triggerRev"] = r.stats["rev"]
    if mode != "blocking":
        while ix.rev != b:
            time.sleep(0.001)
        out["switchMs"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    r2 = ix.query(q)
    out["after"] = (time.perf_counter() - t) * 1000
    out["afterRev"] = r2.stats["rev"]
    ix.close()
    print("@@ROW " + json.dumps(out), flush=True)
    _ = a


if __name__ == "__main__":
    main()
