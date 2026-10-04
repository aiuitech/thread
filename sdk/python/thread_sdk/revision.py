"""Resolving and loading a revision from a store: `idx/current.json` (or a generation's head) to (gen, rev), then
the manifest, slabs table and tombstone sets that revision reads; and `snapshot_keys`, the objects it reads. No
native code here, so the pack writer and other tools can use it without an index."""

from __future__ import annotations

from typing import Any

from .query import RevState
from .store import BlobStore
from .wire import (
    BytesLike,
    ThreadError,
    build_tomb_state,
    grouped_table_of,
    has_dense,
    keys,
    manifest_files,
    manifest_rev_of_key,
    parse_current,
    parse_manifest,
    parse_slabs_table,
    routing_tables_of,
    slab_rev_of,
    u32_array,
)

__all__ = ["head_revision", "resolve_target", "load_rev_state", "snapshot_keys"]


def get_required(store: BlobStore, key: str | None) -> BytesLike:
    assert key is not None
    b = store.get(key)
    if b is None:
        raise ThreadError(f"missing object {key}", 404)
    return b


def head_revision(store: BlobStore, gen: str) -> int:
    """The newest manifest revision of `gen` present in the store (needs `list`), else 0 (legacy `manifest.json`)."""
    if not store.supports_list:
        return 0
    prefix = f"idx/{gen}/manifest."
    head = -1
    for key in store.list(prefix):
        r = manifest_rev_of_key(key[len(prefix):])
        if r is not None:
            head = max(head, r)
    if head < 0:
        raise ThreadError(f"no manifest for generation {gen}", 404)
    return head


def resolve_target(store: BlobStore, gen: str | None = None, rev: int | None = None) -> tuple[str, int]:
    """The (gen, rev) to read: both given; `gen` alone at its head (current.json's rev when it names that
    generation, else the newest manifest in the store); else `idx/current.json` (rev absent = 0)."""
    if gen is not None and rev is not None:
        return gen, rev
    cur = store.get(keys.current())
    if cur is None:
        if gen is None:
            raise ThreadError("no published index (idx/current.json missing)", 404)
        return gen, head_revision(store, gen)
    cgen, crev = parse_current(cur)
    if gen is None or gen == cgen:
        return cgen, rev if rev is not None else crev
    return gen, head_revision(store, gen)


def load_rev_state(store: BlobStore, gen: str, rev: int) -> RevState:
    """Loads a revision: manifest, slabs table, tombstone sets."""
    mb = store.get(keys.manifest(gen, rev))
    if mb is None:
        raise ThreadError(f"no manifest {keys.manifest(gen, rev)}", 404)
    manifest = parse_manifest(mb, gen, rev)
    files = manifest_files(manifest)
    fmt = manifest.get("format") or 1
    cells = manifest["k1"] * manifest["k2"]
    slabs = None
    slabs_key = files.get("slabs")
    if slabs_key and has_dense(manifest):
        b = store.get(slabs_key)
        if b is not None:
            slabs = parse_slabs_table(b, cells, slabs_key)
        elif fmt >= 2 or manifest.get("files"):
            # v1 generations may predate slabs tables: their slabs are read whole.
            raise ThreadError(f"missing {files['slabs']}", 404)
    sets = [(t["rev"], u32_array(get_required(store, t["key"]), t["key"])) for t in manifest.get("tombstones") or []]
    if fmt >= 2 and slabs is None and has_dense(manifest):
        raise ThreadError(f"{files['slabs']} missing for format {fmt}")
    return RevState(gen=gen, rev=rev, manifest=manifest, files=files, format=fmt, slabs=slabs,
                    deltas=list(manifest.get("deltas") or []), segments=list(manifest.get("segments") or []),
                    tomb=build_tomb_state(sets) if sets else None,
                    base_key=f"{gen}|{files['coarse']}|{files['fine']}|{files.get('pq') or '-'}")


def snapshot_keys(rev: RevState) -> list[str]:
    """Every object a revision reads, sorted: its manifest, the base files (`coarse`, `fine`, `pq`, `slabs` and any
    other file the manifest lists), one slab per cell at the revision the slabs table names, the delta segments,
    tombstone sets and lexical/docs segments. `idx/current.json` is not included: a snapshot writes its own pointer."""
    m: dict[str, Any] = rev.manifest
    out = {keys.manifest(rev.gen, rev.rev)}
    if has_dense(m):
        for name, v in rev.files.items():
            if name == "slabs" and rev.slabs is None:
                continue  # v1 generations without a slabs table
            if name in ("base", "baseCold"):
                continue  # base packs repeat objects listed here
            if isinstance(v, str) and v:
                out.add(v)
        # Compact routing tables made from the current means (fine.f16, fine.i8, the tree-grouped table).
        r = routing_tables_of(m, rev.files) or {}
        for t in (r.get("f16"), r.get("i8"), grouped_table_of(m, rev.files)):
            if isinstance(t, dict) and isinstance(t.get("key"), str):
                out.add(t["key"])
        for cell in range(m["k1"] * m["k2"]):
            out.add(keys.slab(rev.gen, cell, slab_rev_of(rev.slabs[cell]) if rev.slabs is not None else 0))
    # A centroid tree (written by the Worker's publish-rev --tree) is read by the Worker's beam routing.
    tree = m.get("tree")
    if isinstance(tree, dict) and isinstance(tree.get("key"), str):
        out.add(tree["key"])
    for d in rev.deltas:
        out.add(d["key"])
    for t in m.get("tombstones") or []:
        out.add(t["key"])
    for s in rev.segments:
        out.add(s["key"])
    return sorted(out)
