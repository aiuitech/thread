"""Wire formats shared with the thread Worker and the TypeScript SDK: packed hits, slab and delta-segment
headers, the manifest (revisions, deltas, tombstone sets, lexical segments), the slabs table and the object key
layout. Plain Python over bytes (no native code), so headers can be parsed without copying a slab.

Manifests stay plain dicts with their stored (camelCase) keys: they are the wire format, shared with the other
writers, and every field a newer writer adds survives a revision written from Python.
"""

from __future__ import annotations

import json
import re
import struct
from array import array
from dataclasses import dataclass, field
from typing import Any, Iterable, NamedTuple, Union

from . import _native

#: What stores return: bytes, or a view into a larger buffer.
BytesLike = Union[bytes, bytearray, memoryview]

__all__ = [
    "ThreadError",
    "PackedHit",
    "parse_hits",
    "packed_count",
    "SlabHeader",
    "parse_slab_header",
    "slab_entry",
    "DeltaHeader",
    "parse_delta_header",
    "delta_entry",
    "LEGACY_DEFAULTS",
    "ResolvedLexConfig",
    "resolve_lex_config",
    "keys",
    "manifest_files",
    "ROUTING_ELEMS",
    "routing_row_bytes",
    "routing_tables_of",
    "choose_routing",
    "grouped_table_of",
    "parse_current",
    "parse_manifest",
    "parse_slabs_table",
    "has_dense",
    "slab_rev_of",
    "TombState",
    "build_tomb_state",
    "id_union",
    "tomb_for",
    "tomb_for_delta",
    "tomb_for_segment",
    "tomb_hides",
    "NO_TOMBSTONES",
    "dumps",
]


class ThreadError(Exception):
    """An SDK error with an HTTP-like status: 400 bad input, 404 missing object, 409 write conflict, 500 corrupt
    index."""

    def __init__(self, message: str, status: int = 500) -> None:
        super().__init__(message)
        self.status = status


def dumps(obj: Any) -> str:
    """JSON the way the other writers serialize it (`JSON.stringify`: no spaces)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _is_int(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    return isinstance(v, int) or (isinstance(v, float) and v.is_integer())


# ---------- packed hits ----------


class PackedHit(NamedTuple):
    """One hit of `scan`/`scan_delta`/`merge`: `idx` is the entry index in its slab or delta segment."""

    id: int
    score: float
    idx: int


_HIT = struct.Struct("<IfI")


def parse_hits(packed: BytesLike) -> list[PackedHit]:
    """Parses packed hits `[count u32][id u32, score f32, idx u32]*` (little-endian)."""
    if len(packed) < 4:
        return []
    (n,) = struct.unpack_from("<I", packed, 0)
    if len(packed) < 4 + n * 12:
        raise ThreadError(f"packed hits: {n} hits need {4 + n * 12} bytes, have {len(packed)}")
    return [PackedHit(*t) for t in _HIT.iter_unpack(memoryview(packed)[4 : 4 + n * 12])]


def packed_count(b: BytesLike) -> int:
    return 0 if len(b) < 4 else struct.unpack_from("<I", b, 0)[0]


# ---------- slab header (64 bytes, "VIBS") ----------

SLAB_MAGIC = 0x53424956  # "VIBS" LE
SLAB_ENTRY_BYTES = 16


@dataclass(frozen=True)
class SlabHeader:
    #: 1 = int8 only (read whole, no phase 2); 2 = 4-bit scan + int8 rerank; 3 = PQ scan + int8 rerank.
    version: int
    dim: int
    count: int
    cell: int
    centroid_off: int
    #: Entry table: `count` x 16 B `[id u32, payloadOff u32, payloadLen u32, attrs u32]`.
    entries_off: int
    #: Bytes to range-read for a scan (`[0, scan_len)`); for v1 the payload offset.
    scan_len: int
    #: v2/v3: rerank record `i` is `[rerank_off + i * rec_len, +rec_len)`; 0 for v1.
    rerank_off: int
    rec_len: int
    payload_off: int
    payload_len: int
    #: v3: PQ subspaces of the scan tier (0 otherwise).
    pq_m: int


def parse_slab_header(b: BytesLike) -> SlabHeader:
    if len(b) < 64:
        raise ThreadError(f"slab header needs 64 bytes, have {len(b)}")
    magic, version = struct.unpack_from("<IH", b, 0)
    if magic != SLAB_MAGIC:
        raise ThreadError("bad slab magic")
    dim, count, cell, centroid_off, entries_off = struct.unpack_from("<5I", b, 8)
    if version in (2, 3):
        # v2 and v3 share every field read here (v3 stores pq_m at 32, where v2 has scales4_off).
        pq_m, rerank_off, payload_off, payload_len, scan_len = struct.unpack_from("<5I", b, 32)
        return SlabHeader(version, dim, count, cell, centroid_off, entries_off, scan_len, rerank_off, 4 + dim,
                          payload_off, payload_len, pq_m if version == 3 else 0)
    if version == 1:
        payload_off, payload_len = struct.unpack_from("<2I", b, 36)
        return SlabHeader(version, dim, count, cell, centroid_off, entries_off, payload_off, 0, 0, payload_off,
                          payload_len, 0)
    raise ThreadError(f"unknown slab version {version}")


class SlabEntry(NamedTuple):
    id: int
    payload_off: int
    payload_len: int
    attrs: int


def slab_entry(b: BytesLike, h: SlabHeader, i: int) -> SlabEntry:
    """Entry `i` of the slab entry table (inside the scan prefix)."""
    o = h.entries_off + i * SLAB_ENTRY_BYTES
    if i >= h.count or len(b) < o + SLAB_ENTRY_BYTES:
        raise ThreadError(f"slab entry {i} out of range")
    return SlabEntry(*struct.unpack_from("<4I", b, o))


def slab_centroid(b: BytesLike, h: SlabHeader) -> bytes:
    """The cell mean stored in a slab prefix, as f32 LE bytes (copied, so it outlives the prefix)."""
    end = h.centroid_off + h.dim * 4
    if len(b) < end:
        raise ThreadError(f"slab prefix too short for the centroid ({len(b)} < {end})")
    return bytes(b[h.centroid_off : end])


# ---------- delta segment header (64 bytes, "VIBD") ----------

DELTA_MAGIC = 0x44424956  # "VIBD" LE
DELTA_ENTRY_BYTES = 24


@dataclass(frozen=True)
class DeltaHeader:
    version: int
    dim: int
    count: int
    pq_m: int
    seq: int
    #: Entry table: `count` x 24 B `[id, cell, attrs, payloadOff, payloadLen, reserved]` (u32 each).
    entries_off: int
    rerank_off: int
    payload_off: int
    payload_len: int


def parse_delta_header(b: BytesLike) -> DeltaHeader:
    if len(b) < 64:
        raise ThreadError(f"delta header needs 64 bytes, have {len(b)}")
    magic, version = struct.unpack_from("<IH", b, 0)
    if magic != DELTA_MAGIC:
        raise ThreadError("bad delta magic")
    if version != 1:
        raise ThreadError(f"unknown delta version {version}")
    dim, count, pq_m, seq, entries_off, rerank_off = struct.unpack_from("<6I", b, 8)
    payload_off, payload_len = struct.unpack_from("<2I", b, 44)
    return DeltaHeader(version, dim, count, pq_m, seq, entries_off, rerank_off, payload_off, payload_len)


class DeltaEntry(NamedTuple):
    id: int
    cell: int
    attrs: int
    payload_off: int
    payload_len: int


def delta_entry(b: BytesLike, h: DeltaHeader, i: int) -> DeltaEntry:
    o = h.entries_off + i * DELTA_ENTRY_BYTES
    if i >= h.count or len(b) < o + DELTA_ENTRY_BYTES:
        raise ThreadError(f"delta entry {i} out of range")
    return DeltaEntry(*struct.unpack_from("<5I", b, o))


# ---------- manifest, slabs table, current pointer ----------

#: Query defaults of manifests without `defaults` (generations before g3).
LEGACY_DEFAULTS: dict[str, int] = {"nprobe": 1, "rerank": 32, "pc": 0}


def slab_rev_of(row: list[int] | None) -> int:
    """A slabs-table row is `[scanLen, rerankOff, count, bytes, slabRev?]`; an absent slabRev is 0."""
    return row[4] if row is not None and len(row) > 4 else 0


@dataclass(frozen=True)
class ResolvedLexConfig:
    """`manifest.lex` with its defaults filled in."""

    tokenizer: dict[str, Any] | None
    k1: float
    b: float
    lengths: str
    bucket_terms: int
    docs: bool
    text: bool
    bucket_docs: int
    rrf_k: float
    recall_k: int | None
    weights: dict[str, float]


def resolve_lex_config(m: dict[str, Any]) -> ResolvedLexConfig | None:
    """`manifest.lex` with its defaults filled in (None when the generation has no lexical configuration)."""
    c = m.get("lex")
    if c is None:
        return None
    d = c.get("defaults") or {}
    w = d.get("weights") or {}
    return ResolvedLexConfig(
        tokenizer=c.get("tokenizer"),
        k1=c.get("k1", 1.2),
        b=c.get("b", 0.75),
        lengths=c.get("lengths", "exact"),
        bucket_terms=c.get("bucketTerms", 256),
        docs=c.get("docs", True),
        text=c.get("text", False),
        bucket_docs=c.get("bucketDocs", 128),
        rrf_k=d.get("rrfK", 60),
        recall_k=d.get("recallK"),
        weights={"dense": w.get("dense", 1), "lexical": w.get("lexical", 1)},
    )


def _pad5(n: int) -> str:
    return str(n).rjust(5, "0")


class keys:
    """Object keys of the index layout, relative to the store prefix."""

    @staticmethod
    def current() -> str:
        return "idx/current.json"

    @staticmethod
    def manifest(gen: str, rev: int) -> str:
        return f"idx/{gen}/manifest.json" if rev == 0 else f"idx/{gen}/manifest.{rev}.json"

    @staticmethod
    def slabs_table(gen: str, rev: int) -> str:
        return f"idx/{gen}/slabs.json" if rev == 0 else f"idx/{gen}/slabs.{rev}.json"

    @staticmethod
    def slab(gen: str, cell: int, slab_rev: int = 0) -> str:
        return f"idx/{gen}/slab/{_pad5(cell)}" if slab_rev == 0 else f"idx/{gen}/slab/{_pad5(cell)}.{slab_rev}"

    @staticmethod
    def tombstones(gen: str, rev: int) -> str:
        return f"idx/{gen}/tombstones.{rev}.u32"

    @staticmethod
    def delta(gen: str, seq: int) -> str:
        return f"idx/{gen}/delta/{seq}.vibd"

    @staticmethod
    def segment(gen: str, seq: int) -> str:
        return f"idx/{gen}/seg/{seq}.thread"

    @staticmethod
    def segment_base(gen: str, rev: int) -> str:
        return f"idx/{gen}/seg/base.{rev}.thread"

    @staticmethod
    def base_pack(gen: str, rev: int) -> str:
        """The base pack of a revision: every object a cold open reads, in one file."""
        return f"idx/{gen}/base.{rev}.threadpack"

    @staticmethod
    def base_cold_pack(gen: str, rev: int) -> str:
        """The cold base pack: the base pack without the full routing table (for `cold_routing="tree"`)."""
        return f"idx/{gen}/base.cold.{rev}.threadpack"

    @staticmethod
    def routing_table(fine: str, elem: str) -> str:
        """Compact routing tables derived from the f32 means at `fine` (`fine.105.f32` -> `fine.105.f16`)."""
        return fine if elem == "f32" else re.sub(r"\.f32$", "", fine) + f".{elem}"

    @staticmethod
    def grouped_table(gen: str, rev: int, elem: str) -> str:
        return f"idx/{gen}/fine.grouped.{rev}.{elem}"


#: Element types of a routing table: the f32 means, IEEE half floats, int8 with one f32 scale per row.
ROUTING_ELEMS = ("f32", "f16", "i8")


def routing_row_bytes(dim: int, elem: str) -> int:
    """Bytes of one routing table row."""
    return 4 * dim if elem == "f32" else 2 * dim if elem == "f16" else 4 + dim


def routing_tables_of(m: dict[str, Any], files: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """`manifest.routing` if it was made from the means `files.fine` names (else None: a compaction that split cells
    wrote new means, and readers fall back to them)."""
    files = files if files is not None else manifest_files(m)
    r = m.get("routing")
    if not isinstance(r, dict) or r.get("fine") != files.get("fine"):
        return None
    return r


def choose_routing(m: dict[str, Any], files: dict[str, Any], opt: str = "f32") -> tuple[str, str]:
    """(elem, key) of the routing table to load for `opt` (f32, f16, i8 or auto = the smallest available); a
    revision without the requested table falls back to f32."""
    if opt not in ("auto", *ROUTING_ELEMS):
        raise ThreadError(f"routing must be one of f32, f16, i8, auto (got {opt!r})", 400)
    r = routing_tables_of(m, files) or {}

    def pick(e: str) -> tuple[str, str] | None:
        if e == "f32":
            return "f32", files["fine"]
        t = r.get(e)
        return (e, t["key"]) if isinstance(t, dict) and t.get("key") else None

    if opt == "auto":
        return pick("i8") or pick("f16") or pick("f32")  # type: ignore[return-value]
    return pick(opt) or pick("f32")  # type: ignore[return-value]


def grouped_table_of(m: dict[str, Any], files: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The tree-grouped routing table of a revision, if it follows its current tree and means."""
    files = files if files is not None else manifest_files(m)
    g = (routing_tables_of(m, files) or {}).get("grouped")
    tree = m.get("tree")
    if not isinstance(g, dict) or not isinstance(tree, dict) or g.get("tree") != tree.get("key") or m.get("k1") != 1:
        return None
    return g


def manifest_files(m: dict[str, Any]) -> dict[str, str | None]:
    """Objects a revision reads (`coarse`, `fine`, `pq`, `slabs`). Manifests without `files` (rev 0) use the
    generation's fixed names."""
    if m.get("files"):
        return m["files"]
    g = m["gen"]
    return {
        "coarse": f"idx/{g}/coarse.f32",
        "fine": f"idx/{g}/fine.f32",
        "pq": f"idx/{g}/pq.bin" if (m.get("format") or 1) == 3 else None,
        "slabs": f"idx/{g}/slabs.json",
    }


def decode_json(b: BytesLike) -> Any:
    return json.loads(bytes(b).decode("utf-8"))


def parse_current(b: BytesLike) -> tuple[str, int]:
    """Parses `idx/current.json` into `(gen, rev)`; a pointer without `rev` (older publishes) means rev 0."""
    j = decode_json(b)
    gen = j.get("gen") if isinstance(j, dict) else None
    if not isinstance(gen, str) or not gen:
        raise ThreadError("current.json has no gen")
    rev = j.get("rev", 0)
    if not _is_int(rev) or rev < 0:
        raise ThreadError(f"current.json has a bad rev {rev!r}")
    return gen, int(rev)


def has_dense(m: dict[str, Any]) -> bool:
    """Whether a generation has a dense index (a lexical-only generation is published with `k2: 0, dim: 0`)."""
    return m.get("k2", 0) > 0 and m.get("dim", 0) > 0


def parse_manifest(b: BytesLike, gen: str, rev: int) -> dict[str, Any]:
    """Parses a manifest and checks it is for `gen`; fills `rev` (absent in rev-0 manifests)."""
    m = decode_json(b)
    if not isinstance(m, dict) or m.get("gen") != gen:
        raise ThreadError(f"manifest for {gen} rev {rev} is for generation {m.get('gen') if isinstance(m, dict) else None}")
    for f in ("k1", "k2", "dim", "vectors"):
        if not _is_int(m.get(f)):
            raise ThreadError(f"manifest {gen} rev {rev}: missing {f}")
    if m.get("rev", 0) != rev:
        raise ThreadError(f"manifest {gen} rev {rev} says rev {m.get('rev')}")
    m["rev"] = rev
    return m


def parse_slabs_table(b: BytesLike, cells: int, key: str) -> list[list[int]]:
    t = decode_json(b)
    if not isinstance(t, list) or len(t) != cells:
        raise ThreadError(f"{key} has {len(t) if isinstance(t, list) else 'no'} rows, expected {cells}")
    return t


_MANIFEST_KEY = re.compile(r"^(?:(\d+)\.)?json$")


def manifest_rev_of_key(suffix: str) -> int | None:
    """The rev of a `manifest.{rev}.json` / `manifest.json` key suffix (after `idx/{gen}/manifest.`), else None."""
    m = _MANIFEST_KEY.match(suffix)
    if not m:
        return None
    return int(m.group(1)) if m.group(1) else 0


# ---------- tombstones ----------

NO_TOMBSTONES = b""


def id_union(sets: Iterable[array]) -> bytes:
    """Sorted unique union of id sets as a tombstone buffer (`u32` LE)."""
    all_ids = array("I")
    for s in sets:
        all_ids.extend(s)
    if not all_ids:
        return b""
    return _native.tombstones_merge(b"", all_ids)


@dataclass
class TombState:
    """The tombstone sets of a revision, parsed once; the filter buffers are cached per slab revision / added
    rev."""

    sets: list[tuple[int, array]]
    #: Ids over all sets (with repeats).
    total: int
    unions: dict[int, bytes] = field(default_factory=dict)
    delta_unions: dict[int, bytes] = field(default_factory=dict)
    _max_rev: dict[int, int] | None = None

    @property
    def max_rev(self) -> dict[int, int]:
        """id -> highest rev of a set containing it (the delta rule), built on first use."""
        if self._max_rev is None:
            mr: dict[int, int] = {}
            for rev, ids in self.sets:
                for i in ids:
                    if mr.get(i, -1) < rev:
                        mr[i] = rev
            self._max_rev = mr
        return self._max_rev


def build_tomb_state(sets: list[tuple[int, array]]) -> TombState:
    return TombState(sets=sets, total=sum(len(ids) for _, ids in sets))


def tomb_for(t: TombState, slab_rev: int) -> bytes:
    """Filter buffer for hits of a slab with revision `slab_rev`: the union of the sets with rev >= slab_rev. A slab
    written by a compaction that read manifest rev R has slabRev R + 1 and already excludes the sets with rev <= R;
    rev-0 slabs are hidden by every set."""
    u = t.unions.get(slab_rev)
    if u is None:
        u = id_union(ids for rev, ids in t.sets if rev >= slab_rev)
        t.unions[slab_rev] = u
    return u


def tomb_for_delta(t: TombState, added_rev: int) -> bytes:
    """Filter buffer for the entries of a delta segment added at manifest rev `added_rev`: the union of the sets
    with rev > added_rev (the upsert's own set has rev = added_rev and hides older copies, not the new one)."""
    u = t.delta_unions.get(added_rev)
    if u is None:
        u = id_union(ids for rev, ids in t.sets if rev > added_rev)
        t.delta_unions[added_rev] = u
    return u


#: Filter buffer for the entries of a lexical/docs segment added at rev `added_rev`: the delta rule.
tomb_for_segment = tomb_for_delta


def tomb_hides(t: TombState | None, id: int, added_rev: int) -> bool:
    """Whether `id` is hidden in a segment added at `added_rev` (a set with rev > added_rev holds it)."""
    if t is None:
        return False
    r = t.max_rev.get(id)
    return r is not None and r > added_rev


def u32_array(b: BytesLike, what: str) -> array:
    if len(b) % 4 != 0:
        raise ThreadError(f"{what} is not u32-aligned")
    a = array("I")
    a.frombytes(bytes(b))
    return a
