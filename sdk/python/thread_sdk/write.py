"""Write path, implemented against docs/update-path.md and wire-compatible with the TypeScript SDK and the Worker:
an upsert encodes one delta segment (`DeltaJob`: primary cell + residual codes against the cell mean, PQ codes with
the generation's codebook) and/or one lexical/docs segment, plus a tombstone set for its ids; a delete just a
tombstone set; both commit a new manifest revision with a create-only put of `manifest.{rev+1}.json` (the lock) and
advance `idx/current.json` with a compare-and-swap.
"""

from __future__ import annotations

import json
import math
import operator
import random
import re
import time
from array import array
from concurrent.futures import Executor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from . import _native
from .attrs import AttrSchema, define_attrs, pack_attrs, schema_extends, validate_schema
from .chunk import ChunkOptions, chunk_index_of, chunk_key, chunk_text, resolve_chunk_options
from .embed import Rows, validate_embedding_config
from .store import BlobStore, unique_suffix
from .wire import (
    COMPACTION_PROFILES,
    CREATE_DEFAULTS,
    ThreadError,
    current_body,
    build_tomb_state,
    dumps,
    has_dense,
    keys,
    manifest_files,
    manifest_rev_of_key,
    parse_current,
    parse_current_full,
    parse_manifest,
    resolve_lex_config,
    tomb_for_segment,
    u32_array,
    ResolvedLexConfig,
)

__all__ = [
    "MAX_UPSERT",
    "UpsertResult",
    "DeleteResult",
    "ConfigureLexicalResult",
    "ConfigureAttrsResult",
    "CompactSegmentsResult",
    "create_lexical_generation",
    "create_generation",
]

MAX_UPSERT = 4096
COMMIT_ATTEMPTS = 12
#: Vectors passed to DeltaJob.add per call.
DELTA_ADD_CHUNK = 256


@dataclass
class UpsertResult:
    gen: str
    rev: int
    seq: int
    count: int
    #: Primary cell of each vector (empty without vectors).
    cells: list[int]
    distinct_cells: int
    #: `{delta, segment, tombstones, manifest}` object keys (None where nothing was written).
    keys: dict[str, str | None]
    bytes: dict[str, int]
    #: Lost manifest races and current.json CAS retries.
    conflicts: int
    cas_conflicts: int
    #: Whether current.json now points at this revision (False if another generation is current).
    current_updated: bool
    ms: float
    #: The ids written, in input order (the given ones, or those resolved / allocated for `keys`).
    ids: list[int] = field(default_factory=list)
    #: Ids allocated to new keys.
    allocated: int = 0
    #: With `chunk`: `{documents, count, stale}`, the input documents, the chunks written (= `count`) and the stale
    #: chunk keys tombstoned.
    chunks: dict[str, int] | None = None


@dataclass
class DeleteResult:
    gen: str
    rev: int
    count: int
    key: str
    conflicts: int
    cas_conflicts: int
    current_updated: bool
    ms: float
    #: `delete_keys`: the keys that had no live document (nothing was written if every key was missing).
    missing: list[str] | None = None


@dataclass
class ConfigureLexicalResult:
    gen: str
    rev: int
    config: dict[str, Any]
    conflicts: int
    current_updated: bool


@dataclass
class ConfigureAttrsResult:
    gen: str
    rev: int
    schema: AttrSchema
    conflicts: int
    current_updated: bool


@dataclass
class CompactSegmentsResult:
    gen: str
    rev: int
    #: Segments merged (0 when nothing needed compacting; no revision is written then and `rev` is the head).
    merged: int
    key: str | None
    count: int
    bytes: int
    #: Bytes of the segments read.
    read_bytes: int
    conflicts: int
    current_updated: bool
    ms: float


@dataclass
class WriterContext:
    store: BlobStore
    gen: str
    io: Executor
    #: The index's loaded PqCodebook for these base files (if any), to avoid re-parsing 1.5 MB.
    borrow_pq: Callable[[str], Any]
    #: Resolves keys to ids through the `keys` sections of the given (head) revision's manifest.
    resolve_keys: Callable[[dict[str, Any], Sequence[str]], dict[str, int]] = lambda m, k: {}
    #: Every live key with the prefix in the given (head) revision, in key order.
    list_keys: Callable[[dict[str, Any], str], list[str]] = lambda m, p: []
    #: The index's embedder for `manifest.embedding` (`(texts, kind) -> rows`), or None when it cannot embed.
    embedder: Callable[[Sequence[str], str], Rows] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _writable(store: BlobStore) -> BlobStore:
    if not store.supports_put:
        raise ThreadError("this store is read-only (no put)", 400)
    return store


def _put_if_absent(store: BlobStore, key: str, data: bytes | str, content_type: str | None = None) -> bool:
    return _writable(store).put(key, data, if_absent=True, content_type=content_type) is not None


def _put_unique(store: BlobStore, key: str, data: bytes | str) -> str:
    """Writes an immutable object at `key`, or at a unique sibling `{stem}.{nonce}{ext}` if the key is taken."""
    if _put_if_absent(store, key, data):
        return key
    dot = key.rfind(".")
    for _ in range(4):
        alt = f"{key[:dot]}.{unique_suffix()}{key[dot:]}"
        if _put_if_absent(store, alt, data):
            return alt
    raise ThreadError(f"could not write a unique key for {key}")


def _read_current(store: BlobStore) -> tuple[tuple[str, int], str] | None:
    # head first: if the pointer changes between the two reads the CAS fails and the writer re-reads.
    h = store.head(keys.current())
    if h is None:
        return None
    b = store.get(keys.current())
    if b is None:
        return None
    return parse_current(b), h.etag


def _advance_current(store: BlobStore, gen: str, rev: int, manifest: dict[str, Any] | None = None) -> tuple[bool, int]:
    """Moves idx/current.json forward to (gen, rev) with a CAS loop; leaves it alone if it names another generation
    or a newer rev. Returns (updated, conflicts). The pointer carries a copy of `manifest` (when small), so an open
    skips one GET."""
    conflicts = 0
    body = current_body(gen, rev, manifest)
    for _ in range(10):
        r = _read_current(store)
        if r is None:
            if _put_if_absent(store, keys.current(), body, "application/json"):
                return True, conflicts
        else:
            (cgen, crev), etag = r
            if cgen != gen or crev >= rev:
                return False, conflicts
            if _writable(store).put(keys.current(), body, if_match=etag, content_type="application/json") is not None:
                return True, conflicts
        conflicts += 1
        time.sleep((20 + random.random() * 50) / 1000)
    raise ThreadError(f"idx/current.json: gave up after {conflicts} CAS conflicts", 409)


def latest_rev(store: BlobStore, gen: str) -> int:
    """The newest manifest revision in the store (a crashed writer's manifest beyond current.json counts)."""
    if not store.supports_list:
        raise ThreadError("this store cannot list objects, which the writer needs after a lost race", 400)
    prefix = f"idx/{gen}/manifest."
    head = -1
    for key in store.list(prefix):
        r = manifest_rev_of_key(key[len(prefix):])
        if r is not None:
            head = max(head, r)
    if head < 0:
        raise ThreadError(f"no manifest for generation {gen}", 404)
    return head


@dataclass
class _Commit:
    rev: int
    manifest: dict[str, Any]
    manifest_key: str
    manifest_bytes: int
    conflicts: int
    cas_conflicts: int
    current_updated: bool


def _commit_rev(ctx: WriterContext, make: Callable[[int, dict[str, Any], dict[str, Any]], dict[str, Any] | None]) -> _Commit:
    """Commits a new revision: `make(head_rev, head_manifest, files)` returns the next manifest (writing the objects
    it references first), or None for nothing to do. The create-only put of `manifest.{rev+1}.json` serializes
    writers; a lost race rebuilds on the new head."""
    store, gen = ctx.store, ctx.gen
    conflicts = 0
    use_list = False
    for attempt in range(1, COMMIT_ATTEMPTS + 1):
        cur = _read_current(store)
        head = cur[0][1] if not use_list and cur and cur[0][0] == gen else latest_rev(store, gen)
        mb = store.get(keys.manifest(gen, head))
        if mb is None:
            raise ThreadError(f"no manifest {keys.manifest(gen, head)}", 404)
        base = parse_manifest(mb, gen, head)
        nxt = make(head, base, manifest_files(base))
        if nxt is None:
            return _Commit(head, base, keys.manifest(gen, head), len(mb), conflicts, 0, False)
        rev = head + 1
        manifest = {**nxt, "gen": gen, "rev": rev, "parentRev": head, "updatedAt": _now_iso()}
        body = dumps(manifest)
        key = keys.manifest(gen, rev)
        if _put_if_absent(store, key, body, "application/json"):
            updated, cas = _advance_current(store, gen, rev, manifest)
            return _Commit(rev, manifest, key, len(body.encode("utf-8")), conflicts, cas, updated)
        conflicts += 1
        use_list = True
        time.sleep((10 + random.random() * 40 * attempt) / 1000)
    raise ThreadError(f"{gen}: lost the manifest race {conflicts} times", 409)


def _inherit(m: dict[str, Any]) -> dict[str, Any]:
    """The fields a new revision inherits (build notes and the previous change stay with their revision)."""
    rest = {k: v for k, v in m.items() if k not in ("notes", "change", "rev", "parentRev", "updatedAt")}
    return {**rest, "files": manifest_files(m), "deltas": list(m.get("deltas") or []),
            "tombstones": list(m.get("tombstones") or []), "segments": list(m.get("segments") or []),
            "nextSeq": m.get("nextSeq") or 1}


def _u32(values: Iterable[Any], name: str) -> array:
    out = array("I")
    for i, v in enumerate(values):
        try:
            x = operator.index(v)
        except TypeError:
            raise ThreadError(f"{name}[{i}] is not a u32", 400) from None
        if not 0 <= x <= 0xFFFFFFFF:
            raise ThreadError(f"{name}[{i}] is not a u32", 400)
        out.append(x)
    return out


def _is_buffer(v: Any) -> bool:
    try:
        memoryview(v)
        return True
    except TypeError:
        return False


#: Longest key (UTF-8 bytes).
MAX_KEY_BYTES = 4096


@dataclass
class _Batch:
    #: The ids (resolved / allocated per attempt for a keyed batch without ids).
    ids: array
    #: Unit-normalized vectors (f32 LE bytes), or None for a text-only batch.
    vectors: bytes | None
    attrs: array | None
    payloads: list[bytes] | None
    texts: list[str | None] | None
    metas: list[str | None] | None
    keys: list[str] | None = None
    #: Whether the caller gave the ids (else they come from the keys).
    given_ids: bool = True


def _prepare_batch(ids_in: Iterable[int] | None, vectors: Any, attrs: Iterable[Any] | None, payloads: Sequence[bytes | str] | None,
                   texts: Sequence[str | None] | None, metas: Sequence[str | None] | None, dim: int,
                   lex: ResolvedLexConfig | None, schema: AttrSchema | None,
                   keys_in: Sequence[str] | None = None) -> _Batch:
    keys_l = list(keys_in) if keys_in is not None else None
    if keys_l is not None:
        if lex is None:
            raise ThreadError("keys need a lexical configuration (index.configure_lexical)", 400)
        if len(keys_l) > MAX_UPSERT:
            raise ThreadError(f"at most {MAX_UPSERT} documents per upsert", 400)
        if len(set(keys_l)) != len(keys_l):
            raise ThreadError("duplicate keys in the batch", 400)
        for i, k in enumerate(keys_l):
            if not isinstance(k, str) or not k:
                raise ThreadError(f"keys[{i}] is not a non-empty string", 400)
            if len(k.encode("utf-8")) > MAX_KEY_BYTES:
                raise ThreadError(f"keys[{i}] is longer than {MAX_KEY_BYTES} bytes", 400)
    if ids_in is None and keys_l is None:
        raise ThreadError("an upsert needs ids or keys", 400)
    given_ids = ids_in is not None
    ids = _u32(ids_in, "ids") if ids_in is not None else array("I", [0] * len(keys_l or []))
    n = len(ids)
    if n == 0:
        raise ThreadError("empty batch", 400)
    if n > MAX_UPSERT:
        raise ThreadError(f"at most {MAX_UPSERT} documents per upsert", 400)
    if given_ids and len(set(ids)) != n:
        raise ThreadError("duplicate ids in the batch", 400)
    if 0xFFFFFFFF in ids:
        raise ThreadError("id 4294967295 is reserved", 400)
    if keys_l is not None and len(keys_l) != n:
        raise ThreadError(f"{len(keys_l)} keys for {n} ids", 400)
    texts_l = list(texts) if texts is not None else None
    metas_l = list(metas) if metas is not None else None
    if texts_l is not None and len(texts_l) != n:
        raise ThreadError(f"{len(texts_l)} texts for {n} ids", 400)
    if metas_l is not None and len(metas_l) != n:
        raise ThreadError(f"{len(metas_l)} metas for {n} ids", 400)
    if (texts_l is not None or metas_l is not None) and lex is None:
        raise ThreadError("texts/metas need a lexical configuration (index.configure_lexical)", 400)
    # One attribute word per document: field values under the schema or a raw u32; stored with the vector and in docs.
    attrs_a = array("I", [pack_attrs(schema, a) for a in attrs]) if attrs is not None else None
    if attrs_a is not None and len(attrs_a) != n:
        raise ThreadError(f"{len(attrs_a)} attrs for {n} ids", 400)
    if vectors is None:
        if texts_l is None and metas_l is None and keys_l is None:
            raise ThreadError("an upsert needs vectors, texts, metas or keys", 400)
        if payloads is not None:
            raise ThreadError("payloads belong to vectors", 400)
        return _Batch(ids, None, attrs_a, None, texts_l, metas_l, keys_l, given_ids)
    if dim == 0:
        raise ThreadError("this generation has no dense index: upsert texts/metas only", 400)
    # A buffer (numpy array of shape (n, dim) or (n * dim,), array('f'), raw f32 bytes) goes to the native side as
    # is; a list is either flat floats or one row per vector.
    if _is_buffer(vectors):
        flat: Any = vectors
    else:
        rows = list(vectors)
        if rows and not isinstance(rows[0], (int, float)) and not hasattr(rows[0], "__float__"):
            if len(rows) != n:
                raise ThreadError(f"{len(rows)} vectors for {n} ids", 400)
            flat = array("f")
            for i, r in enumerate(rows):
                r = list(r)
                if len(r) != dim:
                    raise ThreadError(f"vector {i} has {len(r)} dims, index dim is {dim}", 400)
                flat.extend(r)
        else:
            flat = rows
    try:
        unit, bad = _native.normalize_rows(flat, dim)
    except (ValueError, TypeError) as e:
        raise ThreadError(f"vectors: {e}", 400) from e
    if bad >= 0:
        raise ThreadError(f"vector {bad} (id {ids[bad]}) has zero or non-finite norm", 400)
    if len(unit) != n * dim * 4:
        raise ThreadError(f"vectors has {len(unit) // 4} floats, expected {n} x {dim}", 400)
    payloads_l = [p.encode("utf-8") if isinstance(p, str) else bytes(p) for p in payloads] if payloads is not None else None
    if payloads_l is not None and len(payloads_l) != n:
        raise ThreadError(f"{len(payloads_l)} payloads for {n} vectors", 400)
    return _Batch(ids, unit, attrs_a, payloads_l, texts_l, metas_l, keys_l, given_ids)


def _assign_ids(ctx: WriterContext, manifest: dict[str, Any], keys_l: Sequence[str]) -> tuple[array, int]:
    """The ids of a keyed batch against the head revision: known keys keep their id, new keys take the next ids from
    `manifest.nextId` (`vectors` when absent). Returns the ids and the number allocated."""
    found = ctx.resolve_keys(manifest, keys_l)
    next_id = manifest.get("nextId")
    if next_id is None:
        next_id = manifest.get("vectors") or 0
    allocated = 0
    ids = array("I")
    for k in keys_l:
        id = found.get(k)
        if id is None:
            if next_id >= 0xFFFFFFFF:
                raise ThreadError("the id space is exhausted", 400)
            id, next_id, allocated = next_id, next_id + 1, allocated + 1
        ids.append(id)
    return ids, allocated


def _alloc_seq(store: BlobStore, gen: str, start: int) -> int:
    """The first free seq >= `start` for both the delta and the segment key (a crashed writer's objects are
    skipped)."""
    seq = max(1, start)
    for _ in range(64):
        if store.head(keys.delta(gen, seq)) is None and store.head(keys.segment(gen, seq)) is None:
            break
        seq += 1
    return seq


@dataclass
class _EncodedSegment:
    key: str
    bytes: int
    ref: dict[str, Any]


def _encode_segment(ctx: WriterContext, seq: int, lex: ResolvedLexConfig, b: _Batch) -> _EncodedSegment:
    """Tokenizes and writes one lexical/docs segment for the batch under the generation's `lex` configuration."""
    # `{}` is the default tokenizer; a None tokenizer would mean a docs-only segment without a lexical index.
    job = _native.SegmentJob(dumps(lex.tokenizer or {}), lex.k1, lex.b, lex.lengths, lex.bucket_terms, lex.docs,
                             lex.text, lex.bucket_docs, b.keys is not None, lex.bucket_keys)
    for i, id in enumerate(b.ids):
        job.add(id, b.texts[i] if b.texts else None, b.metas[i] if b.metas else None,
                b.attrs[i] if b.attrs is not None else 0, b.keys[i] if b.keys is not None else None)
    data = job.finish()
    key = _put_unique(ctx.store, keys.segment(ctx.gen, seq), data)
    ref = {"seq": seq, "key": key, "count": len(b.ids), "bytes": len(data), "lex": True, "docs": lex.docs,
           "text": lex.docs and lex.text}
    if b.keys is not None:
        ref["keys"] = True
    return _EncodedSegment(key, len(data), ref)


@dataclass
class _EncodedDelta:
    seq: int
    key: str
    bytes: int
    cells: list[int]
    n_cells: int
    files_key: str


def _base_key(gen: str, files: dict[str, Any]) -> str:
    return f"{gen}|{files['coarse']}|{files['fine']}|{files.get('pq') or '-'}"


def _encode_delta(ctx: WriterContext, m: dict[str, Any], files: dict[str, Any], b: _Batch, seq: int) -> _EncodedDelta:
    """Encodes and writes one delta segment against the generation's fine means (and PQ codebook)."""
    store, gen = ctx.store, ctx.gen
    fmt = m.get("format") or 1
    if fmt == 3 and not files.get("pq"):
        raise ThreadError(f"format 3 generation {gen} has no PQ codebook")
    base_key = _base_key(gen, files)
    pq = ctx.borrow_pq(base_key) if fmt == 3 else None
    means = store.get(files["fine"])
    if means is None:
        raise ThreadError(f"missing {files['fine']}", 404)
    k, dim = m["k1"] * m["k2"], m["dim"]
    if len(means) != k * dim * 4:
        raise ThreadError(f"{files['fine']} has {len(means) // (4 * dim)} means, expected {k}")
    if fmt == 3 and pq is None:
        pqb = store.get(files["pq"])
        if pqb is None:
            raise ThreadError(f"missing {files['pq']}", 404)
        pq = _native.PqCodebook(pqb)
    job = _native.DeltaJob(dim, k, means, seq)
    if pq is not None:
        job.set_pq_codebook(pq)
    assert b.vectors is not None
    vec = memoryview(b.vectors)
    cells: list[int] = []
    for a in range(0, len(b.ids), DELTA_ADD_CHUNK):
        e = min(len(b.ids), a + DELTA_ADD_CHUNK)
        c = job.add(vec[a * dim * 4 : e * dim * 4], b.ids[a:e], b.attrs[a:e] if b.attrs is not None else None,
                    b.payloads[a:e] if b.payloads is not None else None)
        if len(c) != e - a:
            raise ThreadError(f"DeltaJob added {len(c)} of {e - a} vectors")
        cells.extend(c)
    data = job.finish()
    n_cells = len(_native.delta_cells(data))
    key = _put_unique(store, keys.delta(gen, seq), data)
    return _EncodedDelta(seq, key, len(data), cells, n_cells, base_key)


def _tomb_bytes(ids: array) -> bytes:
    """Sorted unique ids as a tombstone buffer."""
    return _native.tombstones_merge(b"", ids) if len(ids) else b""


@dataclass
class _Chunked:
    """A chunked upsert's expansion: the parent keys and the chunking recorded in the manifest's `change`."""

    parents: list[str]
    words: int
    overlap: int


def _expand(ids: Iterable[int] | None, vectors: Any, attrs: Iterable[Any] | None, payloads: Sequence[bytes | str] | None,
            texts: Sequence[str | None] | None, metas: Sequence[str | None] | None, keys_: Sequence[str] | None,
            chunk: ChunkOptions | Mapping[str, Any] | bool | None, embed: Callable[[list[str]], Any] | Literal[False] | None,
            default_embed: Callable[[Sequence[str], str], Rows] | None = None, dense: bool = False
            ) -> tuple[tuple[Any, ...], _Chunked | None]:
    """Applies `chunk` and `embed`: the arguments with one document per chunk (keyed `chunk_key(parent, i)`, the
    parent's meta and attrs) and the vectors `embed` returns for the texts to index. Validation of the result is
    left to `_prepare_batch`, which sees an ordinary batch. With `embed` absent, texts and no vectors, a dense
    generation that names its model embeds the texts as documents through `default_embed`; `embed=False` skips that."""
    if embed and vectors is not None:
        raise ThreadError("vectors and embed together: pass one or the other", 400)
    if embed is None and vectors is None and texts is not None and dense and default_embed is not None:
        de = default_embed
        embed = lambda t: de(t, "document")  # noqa: E731
    if embed is False:
        embed = None
    chunked = None
    if chunk is not None and chunk is not False:
        try:
            opts = resolve_chunk_options(chunk)
        except ValueError as e:
            raise ThreadError(str(e), 400) from None
        if ids is not None or keys_ is None:
            raise ThreadError("chunked upserts name documents by key: pass keys, not ids", 400)
        if texts is None:
            raise ThreadError("chunk needs texts (one per key)", 400)
        if vectors is not None or payloads is not None:
            raise ThreadError("with chunk, vectors come from embed (one per chunk)", 400)
        parents = list(keys_)
        n = len(parents)
        if len(texts) != n:
            raise ThreadError(f"{len(texts)} texts for {n} keys", 400)
        metas_l = list(metas) if metas is not None else None
        attrs_l = list(attrs) if attrs is not None else None
        if metas_l is not None and len(metas_l) != n:
            raise ThreadError(f"{len(metas_l)} metas for {n} keys", 400)
        if attrs_l is not None and len(attrs_l) != n:
            raise ThreadError(f"{len(attrs_l)} attrs for {n} keys", 400)
        ck: list[str] = []
        ct: list[str | None] = []
        cm: list[str | None] = []
        ca: list[Any] = []
        for i, parent in enumerate(parents):
            for c in chunk_text(texts[i] or "", opts.words, opts.overlap):
                ck.append(chunk_key(parent, c.index))
                ct.append(c.text)
                if metas_l is not None:
                    cm.append(metas_l[i])
                if attrs_l is not None:
                    ca.append(attrs_l[i])
        if len(ck) > MAX_UPSERT:
            raise ThreadError(f"chunk produced {len(ck)} documents, at most {MAX_UPSERT} per upsert: pass fewer texts", 400)
        keys_, texts = ck, ct
        metas = cm if metas_l is not None else None
        attrs = ca if attrs_l is not None else None
        chunked = _Chunked(parents, opts.words, opts.overlap)
    if embed is not None:
        to_embed = [t or "" for t in (texts or [])]
        if not to_embed:
            raise ThreadError("embed needs texts", 400)
        vectors = embed(to_embed)
        rows = len(vectors) if isinstance(vectors, (list, tuple)) and vectors and not isinstance(vectors[0], (int, float)) else None
        if rows is not None and rows != len(to_embed):
            raise ThreadError(f"embed returned {rows} vectors for {len(to_embed)} texts", 400)
        if rows is None and len(vectors) == 0:
            raise ThreadError(f"embed returned 0 vectors for {len(to_embed)} texts", 400)
    return (ids, vectors, attrs, payloads, texts, metas, keys_), chunked


def _stale_chunk_ids(ctx: WriterContext, manifest: dict[str, Any], chunked: _Chunked, written: Sequence[str]) -> array:
    """The ids of the parents' chunk keys the batch no longer writes (`parent#<i>` for an `i` not in the batch)."""
    have = set(written)
    stale = [k for p in chunked.parents for k in ctx.list_keys(manifest, p + "#")
             if k not in have and chunk_index_of(k, p) is not None]
    if not stale:
        return array("I")
    return array("I", sorted(set(ctx.resolve_keys(manifest, stale).values())))


def upsert(ctx: WriterContext, ids: Iterable[int] | None, vectors: Any = None, *, attrs: Iterable[Any] | None = None,
           payloads: Sequence[bytes | str] | None = None, texts: Sequence[str | None] | None = None,
           metas: Sequence[str | None] | None = None, keys_: Sequence[str] | None = None,
           replace: bool = True, chunk: ChunkOptions | Mapping[str, Any] | bool | None = None,
           embed: Callable[[list[str]], Any] | Literal[False] | None = None) -> UpsertResult:
    t0 = time.perf_counter()
    store, gen = ctx.store, ctx.gen
    _writable(store)
    (ids, vectors, attrs, payloads, texts, metas, keys_), chunked = _expand(ids, vectors, attrs, payloads, texts, metas,
                                                                           keys_, chunk, embed, ctx.embedder,
                                                                           _head_has_dense(ctx))
    st: dict[str, Any] = {"batch": None, "delta": None, "seg": None, "seq": 0, "tomb_key": None, "tomb_bytes": 0,
                          "ids_key": None, "ids_rev": -1, "allocated": 0, "stale": array("I"), "stale_rev": -1}

    def make(rev: int, manifest: dict[str, Any], files: dict[str, Any]) -> dict[str, Any]:
        lex = resolve_lex_config(manifest)
        if st["batch"] is None:
            schema = validate_schema(manifest["attrs"]) if manifest.get("attrs") else None
            st["batch"] = _prepare_batch(ids, vectors, attrs, payloads, texts, metas,
                                         manifest["dim"] if has_dense(manifest) else 0, lex, schema, keys_)
        batch: _Batch = st["batch"]
        want_seg = lex is not None and (batch.texts is not None or batch.metas is not None or batch.keys is not None)
        nxt = _inherit(manifest)
        new_rev = rev + 1
        if not st["seq"]:
            st["seq"] = _alloc_seq(store, gen, manifest.get("nextSeq") or 1)
        seq = st["seq"]
        # Chunk keys of the parents that this put no longer writes are tombstoned with the batch (per head, like the
        # id resolution: another writer may have changed the parents meanwhile).
        if chunked is not None and st["stale_rev"] != rev:
            assert batch.keys is not None
            st["stale"] = _stale_chunk_ids(ctx, manifest, chunked, batch.keys)
            st["stale_rev"] = rev
        # Ids: given, or resolved through the head's keys (re-resolved when a lost race moved the head, since another
        # writer may have allocated ids or written the same keys meanwhile).
        base_next = manifest.get("nextId")
        if base_next is None:
            base_next = manifest.get("vectors") or 0
        next_id = base_next
        if not batch.given_ids:
            if st["ids_rev"] != rev:
                assert batch.keys is not None
                batch.ids, st["allocated"] = _assign_ids(ctx, manifest, batch.keys)
                st["ids_rev"] = rev
            next_id = base_next + st["allocated"]  # the allocation took the next ids of this head
        else:
            next_id = max(next_id, max(batch.ids) + 1)
        if next_id != base_next or manifest.get("nextId") is not None:
            nxt["nextId"] = next_id
        ids_key = batch.ids.tobytes()
        if ids_key != st["ids_key"]:
            st["ids_key"], st["delta"], st["seg"] = ids_key, None, None
        # The delta and the segment are reused across retries (unless the means/codebook or the ids changed under
        # us); the tombstone set is keyed by the rev, so it is rewritten for every attempt.
        stale: array = st["stale"]
        tomb_ids = (array("I", sorted(set(batch.ids) | set(stale))) if len(stale) else batch.ids) if replace else stale
        tb = _tomb_bytes(tomb_ids) if len(tomb_ids) else None
        delta_f = seg_f = tomb_f = None
        if batch.vectors is not None:
            prev = st["delta"]
            if prev is None or prev.files_key != _base_key(gen, files):
                delta_f = ctx.io.submit(_encode_delta, ctx, manifest, files, batch, seq)
        if want_seg and st["seg"] is None:
            assert lex is not None
            seg_f = ctx.io.submit(_encode_segment, ctx, seq, lex, batch)
        if tb is not None:
            tomb_f = ctx.io.submit(_put_unique, store, keys.tombstones(gen, new_rev), tb)
        if delta_f is not None:
            st["delta"] = delta_f.result()
        if seg_f is not None:
            st["seg"] = seg_f.result()
        st["tomb_key"] = tomb_f.result() if tomb_f is not None else None
        delta: _EncodedDelta | None = st["delta"] if batch.vectors is not None else None
        seg: _EncodedSegment | None = st["seg"] if want_seg else None
        if tb is not None:
            st["tomb_bytes"] = len(tb)
            nxt["tombstones"].append({"rev": new_rev, "key": st["tomb_key"], "count": len(tb) // 4})
        if delta is not None:
            nxt["deltas"].append({"seq": delta.seq, "key": delta.key, "count": len(batch.ids), "bytes": delta.bytes,
                                  "addedRev": new_rev, "cells": delta.n_cells})
        if seg is not None:
            nxt["segments"] = [*nxt["segments"], {**seg.ref, "addedRev": new_rev}]
        nxt["nextSeq"] = max(nxt.get("nextSeq") or 1, seq + 1)
        nxt["change"] = {"op": "upsert", "seq": seq, "count": len(batch.ids), "replace": replace,
                         "dense": delta is not None, "lexical": seg is not None, "writer": "thread-sdk-python"}
        if batch.keys is not None:
            nxt["change"].update(keys=len(batch.keys), allocated=st["allocated"])
        if chunked is not None:
            nxt["change"]["chunk"] = {"words": chunked.words, "overlap": chunked.overlap,
                                      "documents": len(chunked.parents), "stale": len(stale)}
        return nxt

    c = _commit_rev(ctx, make)
    batch, d, seg = st["batch"], st["delta"], st["seg"]
    return UpsertResult(
        gen=gen,
        rev=c.rev,
        seq=st["seq"],
        count=len(batch.ids),
        cells=d.cells if d else [],
        distinct_cells=d.n_cells if d else 0,
        keys={"delta": d.key if d else None, "segment": seg.key if seg else None, "tombstones": st["tomb_key"],
              "manifest": c.manifest_key},
        bytes={"delta": d.bytes if d else 0, "segment": seg.bytes if seg else 0, "tombstones": st["tomb_bytes"],
               "manifest": c.manifest_bytes},
        conflicts=c.conflicts,
        cas_conflicts=c.cas_conflicts,
        current_updated=c.current_updated,
        ms=round((time.perf_counter() - t0) * 1000, 3),
        ids=list(batch.ids),
        allocated=st["allocated"],
        chunks={"documents": len(chunked.parents), "count": len(batch.ids), "stale": len(st["stale"])} if chunked else None,
    )


def delete_keys(ctx: WriterContext, keys_in: Iterable[str]) -> DeleteResult:
    """Deletes documents by key: the keys are resolved through the head revision's `keys` sections and the ids found
    are tombstoned (one set, rev + 1). Nothing is written when no key has a live document."""
    t0 = time.perf_counter()
    store, gen = ctx.store, ctx.gen
    _writable(store)
    wanted = list(dict.fromkeys(keys_in))
    if not wanted:
        raise ThreadError("no keys", 400)
    st: dict[str, Any] = {"key": "", "ids": array("I"), "missing": []}

    def make(rev: int, manifest: dict[str, Any], files: dict[str, Any]) -> dict[str, Any] | None:
        if resolve_lex_config(manifest) is None:
            raise ThreadError("this generation has no lexical configuration (no keys)", 400)
        found = ctx.resolve_keys(manifest, wanted)
        st["missing"] = [k for k in wanted if k not in found]
        st["ids"] = array("I", sorted(set(found.values())))
        if not st["ids"]:
            return None
        tb = _tomb_bytes(st["ids"])
        nxt = _inherit(manifest)
        st["key"] = _put_unique(store, keys.tombstones(gen, rev + 1), tb)
        nxt["tombstones"].append({"rev": rev + 1, "key": st["key"], "count": len(st["ids"])})
        nxt["change"] = {"op": "delete", "count": len(st["ids"]), "keys": len(wanted), "writer": "thread-sdk-python"}
        return nxt

    c = _commit_rev(ctx, make)
    return DeleteResult(gen, c.rev, len(st["ids"]), st["key"], c.conflicts, c.cas_conflicts, c.current_updated,
                        round((time.perf_counter() - t0) * 1000, 3), st["missing"])


def delete_ids(ctx: WriterContext, ids_in: Iterable[int]) -> DeleteResult:
    t0 = time.perf_counter()
    store, gen = ctx.store, ctx.gen
    _writable(store)
    ids = _u32(ids_in, "ids")
    if not len(ids):
        raise ThreadError("no ids", 400)
    tb = _tomb_bytes(ids)
    out = {"key": ""}

    def make(rev: int, manifest: dict[str, Any], files: dict[str, Any]) -> dict[str, Any]:
        nxt = _inherit(manifest)
        out["key"] = _put_unique(store, keys.tombstones(gen, rev + 1), tb)
        nxt["tombstones"].append({"rev": rev + 1, "key": out["key"], "count": len(tb) // 4})
        nxt["change"] = {"op": "delete", "count": len(tb) // 4, "writer": "thread-sdk-python"}
        return nxt

    c = _commit_rev(ctx, make)
    return DeleteResult(gen, c.rev, len(tb) // 4, out["key"], c.conflicts, c.cas_conflicts, c.current_updated,
                        round((time.perf_counter() - t0) * 1000, 3))


def configure_lexical(ctx: WriterContext, config: dict[str, Any]) -> ConfigureLexicalResult:
    """Commits a manifest revision carrying a (new or changed) lexical configuration (`manifest.lex`)."""
    store, gen = ctx.store, ctx.gen
    _writable(store)
    # Validate the tokenizer and the parameters before writing anything.
    try:
        _native.tokenizer_config_json(dumps(config.get("tokenizer") or {}))
    except ValueError as e:
        raise ThreadError(f"tokenizer: {e}", 400) from e
    lengths = config.get("lengths")
    if lengths is not None and lengths not in ("exact", "byte"):
        raise ThreadError('lengths must be "exact" or "byte"', 400)
    k1, b = config.get("k1"), config.get("b")
    if k1 is not None and not k1 >= 0:
        raise ThreadError("k1 must be >= 0", 400)
    if b is not None and not 0 <= b <= 1:
        raise ThreadError("b must be in [0, 1]", 400)

    def make(rev: int, manifest: dict[str, Any], files: dict[str, Any]) -> dict[str, Any]:
        if manifest.get("lex") is not None and manifest.get("segments"):
            # Tokenization must stay the same across the segments of a generation.
            if dumps(manifest["lex"].get("tokenizer")) != dumps(config.get("tokenizer")):
                raise ThreadError("the tokenizer of a generation with segments cannot change (build a new generation)", 400)
        return {**_inherit(manifest), "lex": config, "change": {"op": "configure-lex", "writer": "thread-sdk-python"}}

    c = _commit_rev(ctx, make)
    return ConfigureLexicalResult(gen, c.rev, config, c.conflicts, c.current_updated)


def configure_attrs(ctx: WriterContext, spec: Mapping[str, Any]) -> ConfigureAttrsResult:
    """Commits a manifest revision carrying the attribute schema (`manifest.attrs`): `spec` is either a field spec
    (`{"tenant": {"bits": 12}, "lang": {"values": ["en", "fr"]}, "public": "bool"}`, laid out in order) or a schema
    dict. A schema may only grow: every existing field keeps its name, type, offset and width (an enum may append
    values), since stored words are not rewritten."""
    store, gen = ctx.store, ctx.gen
    _writable(store)
    is_schema = isinstance(spec.get("fields"), list)
    out: dict[str, Any] = {"schema": validate_schema(spec) if is_schema else define_attrs(spec)}

    def make(rev: int, manifest: dict[str, Any], files: dict[str, Any]) -> dict[str, Any]:
        prev = validate_schema(manifest["attrs"]) if manifest.get("attrs") else None
        if prev is not None and not is_schema:
            out["schema"] = define_attrs(spec, prev)
        if prev is not None and not schema_extends(prev, out["schema"]):
            raise ThreadError("the attribute schema can only add fields or enum values (existing fields keep their layout)", 400)
        return {**_inherit(manifest), "attrs": out["schema"], "change": {"op": "configure-attrs", "writer": "thread-sdk-python"}}

    c = _commit_rev(ctx, make)
    return ConfigureAttrsResult(gen, c.rev, out["schema"], c.conflicts, c.current_updated)


def compact_segments(ctx: WriterContext, max_input_bytes: int | None = None) -> CompactSegmentsResult:
    """Merges every segment of the head into one base segment and commits a revision listing only it. Reads the
    segments whole. Nothing is written when the head has at most one segment and no tombstone set newer than it.

    `max_input_bytes` bounds what one call reads (segment bytes plus 4 bytes per tombstoned id): it merges the
    first contiguous run of at least two segments that fits, in place, so unmerged newer copies keep their
    precedence, and merges nothing (`merged` 0) when no two adjacent segments fit."""
    if max_input_bytes is not None and (not math.isfinite(max_input_bytes) or max_input_bytes <= 0):
        raise ThreadError("max_input_bytes must be positive and finite", 400)
    t0 = time.perf_counter()
    store, gen = ctx.store, ctx.gen
    _writable(store)
    st: dict[str, Any] = {"merged": 0, "key": None, "count": 0, "bytes": 0, "read_bytes": 0, "built": None}

    def make(rev: int, manifest: dict[str, Any], files: dict[str, Any]) -> dict[str, Any] | None:
        lex = resolve_lex_config(manifest)
        if lex is None:
            raise ThreadError("this generation has no lexical configuration", 400)
        all_segs = manifest.get("segments") or []
        sets = manifest.get("tombstones") or []
        start, segs = 0, all_segs
        if max_input_bytes is not None:
            budget = max_input_bytes - sum(t["count"] * 4 for t in sets)
            segs = []
            for start in range(len(all_segs)):
                used, end = 0, start
                while end < len(all_segs) and used + all_segs[end]["bytes"] <= budget:
                    used += all_segs[end]["bytes"]
                    end += 1
                if end - start >= 2:
                    segs = all_segs[start:end]
                    break
            if len(segs) < 2:
                return None
        newest = max([0, *(t["rev"] for t in sets)])
        if not (len(segs) > 1 or (len(segs) == 1 and newest > segs[0]["addedRev"])):
            return None
        for_keys = "|".join(s["key"] for s in segs)
        built = st["built"]
        if built is None or built["for_keys"] != for_keys:
            parts = []
            for s, p in zip(segs, ctx.io.map(lambda s: store.get(s["key"]), segs)):
                if p is None:
                    raise ThreadError(f"missing segment {s['key']}", 404)
                parts.append(p)
            st["read_bytes"] = sum(len(p) for p in parts)
            tomb_sets = []
            for t in sets:
                tb = store.get(t["key"])
                if tb is None:
                    raise ThreadError(f"missing {t['key']}", 404)
                tomb_sets.append((t["rev"], u32_array(tb, t["key"])))
            tomb = build_tomb_state(tomb_sets) if tomb_sets else None
            tombs = [tomb_for_segment(tomb, s["addedRev"]) if tomb else None for s in segs]
            out = _native.merge_segments(parts, tombs, dumps(lex.tokenizer or {}), lex.k1, lex.b, lex.lengths,
                                         lex.bucket_terms, lex.docs, lex.text, lex.bucket_docs, False, lex.bucket_keys)
            meta = json.loads(_native.segment_meta(out))
            key = _put_unique(store, keys.segment_base(gen, rev + 1), out)
            count = (meta.get("docs") or {}).get("n") or (meta.get("lex") or {}).get("num_docs") or 0
            built = st["built"] = {"for_keys": for_keys, "key": key, "bytes": len(out), "count": count,
                                   "keys": meta.get("keys") is not None}
        st.update(merged=len(segs), key=built["key"], bytes=built["bytes"], count=built["count"])
        # A partial merge keeps the newest addedRev of its run: tombstone sets newer than that still hide its ids.
        added = rev + 1 if max_input_bytes is None else max(s["addedRev"] for s in segs)
        ref = {"seq": 0, "key": built["key"], "addedRev": added, "count": built["count"], "bytes": built["bytes"],
               "lex": True, "docs": lex.docs, "text": lex.docs and lex.text, "base": True}
        if built["keys"]:
            ref["keys"] = True
        return {**_inherit(manifest), "segments": [*all_segs[:start], ref, *all_segs[start + len(segs):]],
                "change": {"op": "compact-segments", "merged": len(segs), "count": built["count"],
                           "writer": "thread-sdk-python"}}

    c = _commit_rev(ctx, make)
    return CompactSegmentsResult(gen, c.rev, st["merged"], st["key"], st["count"], st["bytes"], st["read_bytes"],
                                 c.conflicts, c.current_updated, round((time.perf_counter() - t0) * 1000, 3))


_GEN_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _head_has_dense(ctx: WriterContext) -> bool:
    """Whether the generation's head revision has a dense index (decides whether texts without vectors are embedded)."""
    if ctx.embedder is None:
        return False
    store, gen = ctx.store, ctx.gen
    inline: dict[str, Any] | None = None
    head: int | None = None
    b = store.get(keys.current())
    if b is not None:
        g, r, inline = parse_current_full(b)
        if g == gen:
            head = r
        else:
            inline = None
    if head is None:
        head = latest_rev(store, gen)
    if inline is None:
        mb = store.get(keys.manifest(gen, head))
        if mb is None:
            raise ThreadError(f"no manifest {keys.manifest(gen, head)}", 404)
        inline = parse_manifest(mb, gen, head)
    return has_dense(inline)


def create_generation(store: BlobStore, gen: str, *, embedding: dict[str, Any] | None = None, dim: int | None = None,
                      lex: dict[str, Any] | None | Literal[False] = None, attrs: Mapping[str, Any] | None = None,
                      profile: str = "edge", defaults: Mapping[str, int] | None = None,
                      compaction: Mapping[str, int] | None | Literal[False] = None, set_current: bool = True,
                      notes: Any = None) -> dict[str, Any]:
    """Publishes a dense generation from nothing, for an embedding model (`embedding`, a manifest `embedding` block)
    or a vector size (`dim`): rev 0 with ONE cell (an empty format-2 slab, `fine.f32` and `coarse.f32` of one unit
    row, a slabs table) plus the lexical configuration (`lex`, default `{}`; `False` for a dense-only generation)
    and the compaction policy of `profile` (`"edge"`: cells split at 3000 entries, `"local"`: 1000; `compaction`
    overrides fields, `False` = no automatic compaction). Vectors (or texts, with `embedding` and an embedder) are
    then added with ordinary upserts: they go to delta segments, the Worker's automatic compaction folds them into
    the cell and splits it in two when it passes `maxCellEntries`, so the cell count grows with the data. No k-means
    build, no PQ codebook: the scan tier is 4-bit (`sq4+sq8`). Returns the manifest."""
    _writable(store)
    if not _GEN_NAME.match(gen):
        raise ThreadError(f"bad generation name {json.dumps(gen)}", 400)
    emb = validate_embedding_config(embedding, dim) if embedding is not None else None
    d = emb["dim"] if emb is not None else dim
    if d is None:
        raise ThreadError("create_generation needs an embedding model (embedding) or a vector size (dim)", 400)
    if isinstance(d, bool) or not isinstance(d, int) or d < 1 or d > 65_536:
        raise ThreadError("dim must be an integer in [1, 65536]", 400)
    if profile not in COMPACTION_PROFILES:
        raise ThreadError('profile must be "edge" or "local"', 400)
    dfl = {**CREATE_DEFAULTS, **(defaults or {})}
    for name, lo in (("nprobe", 1), ("rerank", 0), ("pc", 0)):
        v = dfl.get(name)
        if isinstance(v, bool) or not isinstance(v, int) or v < lo:
            raise ThreadError(f"defaults.{name} must be an integer >= {lo}", 400)
    policy = None if compaction is False else {**COMPACTION_PROFILES[profile], **(compaction or {})}
    lex_cfg: dict[str, Any] | None = {} if lex is None else (None if lex is False else dict(lex))
    schema = None
    if attrs is not None:
        schema = validate_schema(attrs) if isinstance(attrs.get("fields"), list) else define_attrs(attrs)
    # One cell whose mean is an arbitrary unit vector (the data decides the means once compaction splits the cell).
    mean = array("f", [1 / math.sqrt(d)] * d)
    job = _native.SlabJob(d, 0, mean, 2)
    slab = bytes(job.finish())
    info = _native.slab_info(slab)
    manifest: dict[str, Any] = {"gen": gen, "k1": 1, "k2": 1, "dim": d, "m": 1, "probeCoarse": 1, "vectors": 0,
                                "format": 2, "tiers": "sq4+sq8", "profile": profile, "defaults": dfl,
                                "createdAt": _now_iso()}
    if notes is not None:
        manifest["notes"] = notes
    if lex_cfg is not None:
        manifest["lex"] = lex_cfg
    if schema is not None:
        manifest["attrs"] = schema
    if emb is not None:
        manifest["embedding"] = emb
    if policy is not None:
        manifest["compaction"] = policy
    manifest.update({"deltas": [], "tombstones": [], "segments": [], "nextSeq": 1, "nextId": 0})
    # The manifest is the create-only lock: written first, so two creators of the same name cannot both succeed.
    if not _put_if_absent(store, keys.manifest(gen, 0), dumps(manifest), "application/json"):
        raise ThreadError(f"generation {gen} already exists", 409)
    mean_bytes = mean.tobytes()
    store.put(f"idx/{gen}/coarse.f32", mean_bytes)
    store.put(f"idx/{gen}/fine.f32", mean_bytes)
    store.put(keys.slab(gen, 0), slab)
    store.put(keys.slabs_table(gen, 0), dumps([[int(info[5]), int(info[6]), 0, len(slab)]]), content_type="application/json")
    if set_current:
        cur = _read_current(store)
        body = current_body(gen, 0, manifest)
        if cur is None:
            _put_if_absent(store, keys.current(), body, "application/json")
        else:
            store.put(keys.current(), body, if_match=cur[1], content_type="application/json")
    return manifest


def create_lexical_generation(store: BlobStore, gen: str, lex: dict[str, Any] | None = None, *,
                              set_current: bool = True, notes: Any = None) -> dict[str, Any]:
    """Publishes a lexical-only generation (no dense index: `k2 0, dim 0, vectors 0`) with the given `manifest.lex`
    configuration as rev 0, so documents can be upserted with texts and searched by text alone. Returns the
    manifest."""
    _writable(store)
    if not _GEN_NAME.match(gen):
        raise ThreadError(f"bad generation name {json.dumps(gen)}", 400)
    manifest: dict[str, Any] = {"gen": gen, "k1": 1, "k2": 0, "dim": 0, "m": 0, "vectors": 0, "format": 1,
                                "profile": "lexical", "createdAt": _now_iso(), "lex": lex or {}, "segments": [],
                                "deltas": [], "tombstones": [], "nextSeq": 1}
    if notes is not None:
        manifest["notes"] = notes
    if not _put_if_absent(store, keys.manifest(gen, 0), dumps(manifest), "application/json"):
        raise ThreadError(f"generation {gen} already exists", 409)
    if set_current:
        cur = _read_current(store)
        body = current_body(gen, 0, manifest)
        if cur is None:
            _put_if_absent(store, keys.current(), body, "application/json")
        else:
            store.put(keys.current(), body, if_match=cur[1], content_type="application/json")
    return manifest


def lex_config_from_kwargs(config: dict[str, Any] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    """`manifest.lex` from a wire-format dict and/or snake_case keyword arguments (`bucket_terms=64`,
    `defaults={"rrf_k": 10}`)."""
    names = {"bucket_terms": "bucketTerms", "bucket_docs": "bucketDocs", "bucket_keys": "bucketKeys", "rrf_k": "rrfK",
             "recall_k": "recallK"}
    out = dict(config or {})
    for k, v in kwargs.items():
        if k == "defaults" and isinstance(v, dict):
            v = {names.get(dk, dk): dv for dk, dv in v.items()}
        out[names.get(k, k)] = v
    return out
