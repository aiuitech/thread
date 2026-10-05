"""`ThreadIndex`: one opened index, the Python counterpart of the TypeScript SDK's `ThreadIndex`."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

from . import _native
from .attrs import AttrFilterExpr, AttrSchema, compile_filter, pack_attrs, unpack_attrs, validate_schema
from .cold import ColdRouting
from .fusion import DEFAULT_RRF_K, rrf
from .lex import DocRecord, KeyPage, LexIndex, LexIo, LexSearchResult, LexSegment
from .query import Base, QueryContext, QueryResult, RevState, execute_query
from .store import BlobCache, BlobStore, BlobTrace, CachedBlobStore, MemoryCache, PrefixedBlobStore, phase_stat
from .pack import PackedBlobStore, PrefetchStore, is_pack_header, store_at
from .revision import base_keys_of, get_required, head_revision, load_rev_state, resolve_target, snapshot_keys
from .warm import WarmPrefixStore, WarmPackStats, fetch_warm_pack, revision_slab_keys, seed_warm_pack
from .wire import (
    LEGACY_DEFAULTS,
    BytesLike,
    ResolvedLexConfig,
    ThreadError,
    choose_routing,
    grouped_table_of,
    has_dense,
    keys,
    manifest_files,
    parse_manifest,
    resolve_lex_config,
    routing_table_bytes,
    slab_rev_of,
)
from .write import (
    CompactSegmentsResult,
    ConfigureAttrsResult,
    ConfigureLexicalResult,
    DeleteResult,
    UpsertResult,
    WriterContext,
    compact_segments,
    configure_attrs,
    configure_lexical,
    delete_ids,
    delete_keys,
    lex_config_from_kwargs,
    upsert,
)

__all__ = ["ThreadIndex", "open_index", "SearchHit", "SearchResult", "Rank", "resolve_target", "load_rev_state",
           "head_revision"]
_ = (resolve_target, load_rev_state, head_revision)  # re-exported from here for compatibility

log = logging.getLogger("thread_sdk")

#: Bytes a revision switch prefetches by default (`prefetch_bytes`).
DEFAULT_PREFETCH_BYTES = 64 * 1024 * 1024


@dataclass
class Rank:
    """A hit's position in one list (1-based) and its score there."""

    rank: int
    score: float


@dataclass
class SearchHit:
    id: int
    #: The fused RRF score (hybrid), the cosine (dense only) or the BM25 score (lexical only).
    score: float
    dense: Rank | None = None
    lexical: Rank | None = None
    #: With `docs`: the document's metadata (and text); None when the id has no docs entry or docs were not asked.
    doc: DocRecord | None = None


@dataclass
class KeyRecord(DocRecord):
    """A document fetched by key: its key, id and `docs` entry."""

    key: str = ""


@dataclass
class SearchResult:
    hits: list[SearchHit]
    mode: Literal["dense", "lexical", "hybrid"]
    #: `{top_k, recall_k, rrf_k, weights: {dense, lexical}}`.
    params: dict[str, Any]
    #: The dense query's full result (when a vector was given).
    dense: QueryResult | None
    #: The lexical query's full result (when a text was given).
    lexical: LexSearchResult | None
    #: `{docs_io?, timing: {total, dense, lexical, docs}}` (ms).
    stats: dict[str, Any]


def load_base(store: BlobStore, rev: RevState, io: ThreadPoolExecutor, loaded: dict[str, Any] | None = None,
              routing: str = "f32") -> Base:
    """Loads the routing table (and PQ codebook) of a revision's generation into native memory. `routing` picks the
    table: `fine.f32` (default), the compact `fine.f16` / `fine.i8` copies (decoded to f32 on load) or "auto" (the
    smallest the revision has); a revision without the requested table falls back to `fine.f32`. With `loaded`,
    records the load as `loaded["stat"]`: `phase_stat` fields plus `fetch` (ms reading the files, concurrently),
    `build` (ms building the native objects) and `files` (`{coarse, fine, pq}` bytes)."""
    m, files = rev.manifest, rev.files
    elem, table_key = choose_routing(m, files, routing)
    t0 = time.perf_counter()
    trace = BlobTrace()
    fc = io.submit(get_required, store, files["coarse"], trace)
    ff = io.submit(get_required, store, table_key, trace)
    fp = io.submit(get_required, store, files["pq"], trace) if files.get("pq") else None
    c, f, p = fc.result(), ff.result(), fp.result() if fp else None
    t1 = time.perf_counter()
    if rev.format == 3 and p is None:
        raise ThreadError(f"format 3 generation {rev.gen} has no PQ codebook")
    pq = None
    if p is not None:
        pq = _native.PqCodebook(p)
        if pq.dim != m["dim"] or (m.get("pq") and pq.m != m["pq"]["m"]):
            raise ThreadError(f"{files['pq']} is dim {pq.dim} m {pq.m}, manifest dim {m['dim']} pq {m.get('pq')}")
    k = m["k1"] * m["k2"]
    if len(c) != m["k1"] * m["dim"] * 4:
        raise ThreadError(f"{files['coarse']} is {len(c)} bytes, expected {m['k1']} x {m['dim']} f32")
    if len(f) != routing_table_bytes(m["dim"], k, elem):
        raise ThreadError(f"{table_key} is {len(f)} bytes, expected {k} {elem} rows of dim {m['dim']}")
    if elem == "f32":
        searcher = _native.Searcher(m["dim"], m["k1"], m["k2"], c, f)
    else:
        searcher = _native.Searcher.from_table(m["dim"], m["k1"], m["k2"], c, f, elem)
    t2 = time.perf_counter()
    if loaded is not None:
        loaded["stat"] = {**phase_stat((t2 - t0) * 1000, trace), "fetch": round((t1 - t0) * 1000, 3),
                          "build": round((t2 - t1) * 1000, 3),
                          "files": {"coarse": len(c), "fine": len(f), "pq": len(p) if p is not None else 0}}
    return Base(rev.base_key, searcher, pq, elem)


def _prefetch_index_pack(pre: PrefetchStore, packed: PackedBlobStore | None, with_routing: bool = True
                         ) -> dict[str, Any] | None:
    """A `PackedBlobStore`'s recorded base range, read with one range read into `pre` (None when there is none)."""
    if packed is None:
        return None
    t0 = time.perf_counter()
    trace = BlobTrace()
    got = packed.read_base(trace, with_routing)
    if got is None:
        return None
    for k, b in got.items():
        pre.set(k, b)
    return {**phase_stat((time.perf_counter() - t0) * 1000, trace), "source": "index-pack", "key": packed.key}


def _prefetch_open(store: BlobStore, pre: PrefetchStore, gen: str, rev: int, packed: bool,
                   base_pack: bool, cold: bool, routing: str, io: ThreadPoolExecutor,
                   inline: dict[str, Any] | None = None) -> dict[str, Any]:
    """Starts the reads of an open before the loaders ask for them: the revision's base pack (one GET, in parallel
    with the manifest in case there is none), a `PackedBlobStore`'s base range, or, without a pack, every base object
    as soon as the manifest is in. The loaders then read `pre`, which serves those objects from memory. `inline`: the
    manifest when `current.json` carried it (nothing waits for the manifest GET, and none is issued). Returns
    `open_stats["prefetch"]`."""
    t0 = time.perf_counter()
    trace = BlobTrace()

    def done(source: str, key: str | None) -> dict[str, Any]:
        return {**phase_stat((time.perf_counter() - t0) * 1000, trace), "source": source, "key": key}

    if not base_pack:
        return done("none", None)
    mkey = keys.manifest(gen, rev)
    m0 = inline if inline is not None and inline.get("gen") == gen and inline.get("rev", 0) == rev else None

    def manifest_of() -> dict[str, Any] | None:
        if m0 is not None:
            return m0
        try:
            mb = pre.peek(mkey)
            return parse_manifest(mb, gen, rev) if mb is not None else None
        except Exception:  # noqa: BLE001 - load_rev_state reports it
            return None

    # A base pack holds the smallest routing table (publish's default): worth its GET for cold routing and for
    # routing "auto" / "i8"; an f32 / f16 open would download that table for nothing and reads the files instead.
    # (A pack is looked up by its conventional key whatever `files.base` says: it may have been written after the
    # manifest.)
    pack_key = keys.base_cold_pack(gen, rev) if cold else keys.base_pack(gen, rev)
    if not packed and (cold or routing in ("auto", "i8")):
        if m0 is None:
            pre.prefetch(mkey, io)  # in case there is no pack
        if not cold:
            # Should the pack hold another table than the one asked for (a generation without fine.i8), start that
            # one as soon as the manifest is in rather than after the pack.
            def other_table() -> None:
                try:
                    m = manifest_of()
                    if m is None or not has_dense(m):
                        return
                    files = manifest_files(m)
                    want = choose_routing(m, files, routing)[1]
                    if want != choose_routing(m, files, "auto")[1]:
                        pre.prefetch(want, io)
                except Exception:  # noqa: BLE001 - the loaders report it
                    pass

            threading.Thread(target=other_table, daemon=True).start()
        try:
            b = store.get(pack_key, None, trace=trace)
        except Exception:  # noqa: BLE001 - an unreadable pack only means the separate files
            b = None
        if b is not None and is_pack_header(b):
            try:
                meta = pre.add_pack(b)
                if meta.get("gen") == gen and meta.get("rev") == rev:
                    return done("base-pack", pack_key)
            except ThreadError as e:
                log.error("thread_sdk: ignoring the base pack %s: %s", pack_key, e)
    # No pack (or not worth one): every base object as soon as the manifest is in (each read is accounted where its
    # bytes are used).
    if m0 is None:
        pre.prefetch(mkey, io)
    m = manifest_of()
    if m is not None:
        for k in base_keys_of(m, gen, rev, "cold" if cold else "base", "f32" if cold else routing):
            if not (m0 is not None and k == mkey):
                pre.prefetch(k, io)
    return done("files", None)


def _cold_key(rev: RevState) -> str:
    g = grouped_table_of(rev.manifest, rev.files)
    return f"{rev.base_key}|{g['key'] if g else '-'}"


def load_cold_base(store: BlobStore, rev: RevState, io: ThreadPoolExecutor, beam: int | None = None,
                   loaded: dict[str, Any] | None = None) -> Base:
    """Loads a cold base (`cold_routing="tree"`): the centroid tree and the PQ codebook, plus an empty cold router over
    the tree-grouped table that queries fill group by group. `beam` defaults to `manifest.defaults.beam`, else 16."""
    m, files = rev.manifest, rev.files
    g = grouped_table_of(m, files)
    if g is None:
        raise ThreadError(f"{rev.gen} rev {rev.rev} has no tree-grouped routing table for its tree")
    b = beam if beam is not None else (m.get("defaults") or {}).get("beam") or 16
    if isinstance(b, bool) or not isinstance(b, int) or b < 1:
        raise ThreadError("cold_beam must be an integer >= 1", 400)
    t0 = time.perf_counter()
    trace = BlobTrace()
    ft = io.submit(get_required, store, g["tree"], trace)
    fp = io.submit(get_required, store, files["pq"], trace) if files.get("pq") else None
    # (i8 rows are residuals from a centre in the grouped table's head: the first group read fetches it too.)
    t, p = ft.result(), fp.result() if fp else None
    t1 = time.perf_counter()
    if rev.format == 3 and p is None:
        raise ThreadError(f"format 3 generation {rev.gen} has no PQ codebook")
    pq = None
    if p is not None:
        pq = _native.PqCodebook(p)
        if pq.dim != m["dim"] or (m.get("pq") and pq.m != m["pq"]["m"]):
            raise ThreadError(f"{files['pq']} is dim {pq.dim} m {pq.m}, manifest dim {m['dim']} pq {m.get('pq')}")
    router = _native.ColdRouter(t, g["elem"])
    k = m["k1"] * m["k2"]
    if router.k != k or router.dim != m["dim"] or router.groups != g["groups"] or router.row_bytes != g["rowBytes"]:
        raise ThreadError(f"{g['tree']} (k {router.k}, dim {router.dim}, {router.groups} groups) does not match "
                          f"{g['key']} / the manifest (k {k}, dim {m['dim']}, {g['groups']} groups)")
    cold = ColdRouting(store, router, g, b, io)
    t2 = time.perf_counter()
    if loaded is not None:
        loaded["stat"] = {**phase_stat((t2 - t0) * 1000, trace), "fetch": round((t1 - t0) * 1000, 3),
                          "build": round((t2 - t1) * 1000, 3),
                          "files": {"coarse": 0, "fine": len(t), "pq": len(p) if p is not None else 0}}
    return Base(_cold_key(rev), None, pq, g["elem"], cold)


class ThreadIndex:
    """An opened index: one (gen, rev) snapshot plus the generation's routing table in native memory. Safe to share
    between threads; `query`, `search` and the write methods may run concurrently."""

    def __init__(self, store: BlobStore, rev: RevState, *, gen: str | None, pinned: bool, refresh_interval: float,
                 delta_cache_bytes: int, lex_transient_bytes: int, io_workers: int, job_workers: int,
                 cold_policy: dict[str, Any] | None = None, prefetch_bytes: int = DEFAULT_PREFETCH_BYTES,
                 routing: str = "f32", cold_routing: str = "off", cold_beam: int | None = None) -> None:
        self.store = store
        self._rev = rev
        self._gen_opt = gen
        self._pinned = pinned
        self._refresh_interval = refresh_interval
        self._delta_budget = delta_cache_bytes
        self._lex_transient = lex_transient_bytes
        self._routing = routing
        if cold_routing not in ("tree", "off"):
            raise ThreadError(f'cold_routing must be "tree" or "off" (got {cold_routing!r})', 400)
        self._cold_routing = cold_routing
        self._cold_beam = cold_beam
        #: (cold key, future of the cold Base) while queries route cold.
        self._cold: tuple[str, Future] | None = None
        self._bg_load: threading.Thread | None = None
        #: The prefetching view `open` reads through (None after the open).
        self._open_store: BlobStore | None = None
        self._io = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="thread-io")
        self._jobs = ThreadPoolExecutor(max_workers=job_workers, thread_name_prefix="thread-job")
        self._lock = threading.RLock()
        #: (base key, future of the Base) of the routing table queries use.
        self._base: tuple[str, Future] | None = None
        #: The base a revision switch in progress loaded for the new revision.
        self._pre_base: tuple[str, Future] | None = None
        self._delta_cache: dict[str, Future] = {}
        self._seg_cache: dict[str, Future] = {}
        self._lex: tuple[str, int, Future] | None = None
        self._last_check = time.monotonic()
        self._refresh_lock = threading.Lock()
        self._refreshing = False
        self._cold_policy = cold_policy
        self._prefetch_bytes = prefetch_bytes
        self._closed = False
        #: Where `open` spent its time and reads: `{total, current, manifest, slabs, base}` (see `open_index`).
        self.open_stats: dict[str, Any] = {}
        #: The warm-pack layer (`warm_pack=`), None without one.
        self.warm_layer: WarmPrefixStore | None = None
        #: Where the warm pack's prefixes went (the cache store or `warm_layer`), None without one.
        self._warm_target: Any = None
        self._warm_stats: WarmPackStats | None = None
        #: Resolves (with `warm_stats`) once the warm pack is seeded or failed; None without one. Never raises.
        self.warm_ready: Future | None = None

    @classmethod
    def open(
        cls,
        store: BlobStore | str | os.PathLike[str],
        *,
        prefix: str = "",
        gen: str | None = None,
        rev: int | None = None,
        cache: BlobCache | bool | None = None,
        refresh_interval: float = 30.0,
        delta_cache_bytes: int = 40 * 1024 * 1024,
        lex_transient_bytes: int = 8 * 1024 * 1024,
        io_workers: int = 32,
        job_workers: int | None = None,
        miss_policy: str | None = None,
        warm_concurrency: int = 2,
        warmup: bool | int | None = None,
        cold_policy: dict[str, Any] | None = None,
        prefetch_bytes: int = DEFAULT_PREFETCH_BYTES,
        routing: str = "f32",
        cold_routing: str = "off",
        cold_beam: int | None = None,
        base_pack: bool = True,
        warm_pack: str | None = None,
    ) -> "ThreadIndex":
        """Opens the index under `store` (see `open_index`)."""
        if not isinstance(store, BlobStore):
            store = store_at(store)
        pack_store = store if isinstance(store, PackedBlobStore) else None
        pack = pack_store.open_stats if pack_store is not None else None
        packed = pack_store if not prefix else None
        if prefix:
            store = PrefixedBlobStore(store, prefix)
        if cache is None:
            cache = MemoryCache() if store.remote else False
        elif cache is True:
            cache = MemoryCache()
        if cache:
            store = CachedBlobStore(store, cache, miss_policy=miss_policy,  # type: ignore[arg-type]
                                    warm_concurrency=warm_concurrency)
        t0 = time.perf_counter()
        warm: tuple[Any, WarmPrefixStore | None, Future] | None = None
        if warm_pack:
            # The warm pack's one GET starts first and runs alongside the whole open; nothing waits for it. Its
            # prefixes go into the in-process cache when it holds prefixes (`MemoryCache`), else into a prefix layer
            # in front of the store. It is read around an in-process cache (it is not an index object).
            source = pack_store.inner if pack_store is not None else \
                store.inner if isinstance(store, CachedBlobStore) else store
            got: Future = Future()

            def fetch() -> None:
                try:
                    got.set_result(fetch_warm_pack(source, warm_pack, t0))
                except BaseException as e:  # noqa: BLE001 - reported through warm_stats
                    got.set_exception(e)

            threading.Thread(target=fetch, name="thread-warm", daemon=True).start()
            if isinstance(store, CachedBlobStore) and store.seeds_prefixes:
                warm = (store, None, got)
            else:
                layer = WarmPrefixStore(store)
                warm = (layer, layer, got)
                store = layer
        timeline: dict[str, Any] = {}
        ix = cls(store, None, gen=gen, pinned=gen is not None and rev is not None,  # type: ignore[arg-type]
                 refresh_interval=refresh_interval, delta_cache_bytes=delta_cache_bytes,
                 lex_transient_bytes=lex_transient_bytes, io_workers=io_workers,
                 job_workers=job_workers or max(4, min(16, os.cpu_count() or 4)), cold_policy=cold_policy,
                 prefetch_bytes=prefetch_bytes, routing=routing, cold_routing=cold_routing, cold_beam=cold_beam)
        try:
            # Connection warm-up: HEADs on the I/O pool's threads (each keeps its own connection), overlapping the
            # open's own sequential reads (current.json, manifest, base files).
            n = (store.inflight or 12) if warmup is None or warmup is True else int(warmup or 0)
            if warmup is None and not store.remote:
                n = 0
            if n > 0:
                store.warmup(n, ix._io)
            # The open's reads go through a prefetching view: a pack's base range, the base pack, or the files
            # started together. A PackedBlobStore's base range (it starts with idx/current.json) is read before
            # anything else. The range ends with the smallest routing table (`fine{.r}.{elem}`): an open that loads
            # another one stops before it.
            pre = PrefetchStore(store)
            rk = packed.meta.get("baseRouting") if packed is not None else None
            in_range = None if not isinstance(rk, str) else "i8" if rk.endswith(".i8") else "f16" if rk.endswith(
                ".f16") else "f32"
            from_pack = _prefetch_index_pack(pre, packed, cold_routing == "tree" or routing in ("auto", in_range)) \
                if base_pack else None
            # A manifest inlined in current.json saves its GET (and the base files start right after current.json).
            inline: dict[str, Any] = {}
            g, r = resolve_target(pre if from_pack else store, gen, rev, timeline, inline)
            m0 = inline.get("manifest")
            if from_pack is not None and (m0 is not None or pre.has(keys.manifest(g, r))):
                prefetch = from_pack
            else:
                prefetch = _prefetch_open(store, pre, g, r, packed is not None, base_pack, cold_routing == "tree",
                                          routing, ix._io, m0)
            state = load_rev_state(pre, g, r, timeline, m0)
        except BaseException:
            ix.close()
            raise
        ix._rev = state
        ix._open_store = pre
        if warm is not None:
            ix._start_warm(warm[0], warm[1], warm[2], warm_pack, t0)  # type: ignore[arg-type]
        loaded: dict[str, Any] = {}
        try:
            if has_dense(state.manifest):
                if ix._cold_wanted(state):
                    # Cold routing: wait for the tree and the codebook only; the full table loads in the background.
                    ix._acquire_cold(state, loaded)
                    ix._load_full_in_background()
                else:
                    ix._acquire_base(state, loaded)
        except BaseException:
            ix.close()
            raise
        ix._open_store = None  # the prefetched bytes are not kept
        none = phase_stat(0, BlobTrace())
        ix.open_stats = {"total": round((time.perf_counter() - t0) * 1000, 3), "prefetch": prefetch,
                         **{k: timeline.get(k, none) for k in ("current", "manifest", "slabs")},
                         "base": loaded.get("stat")}
        if pack is not None:
            ix.open_stats["pack"] = pack
        return ix

    # ---- properties ----

    @property
    def manifest(self) -> dict[str, Any]:
        return self._rev.manifest

    @property
    def gen(self) -> str:
        return self._rev.gen

    @property
    def rev(self) -> int:
        return self._rev.rev

    @property
    def dim(self) -> int:
        return self._rev.manifest["dim"]

    @property
    def cells(self) -> int:
        """Fine cells (`k1 * k2`)."""
        return self._rev.manifest["k1"] * self._rev.manifest["k2"]

    @property
    def format(self) -> int:
        """Slab format: 1 int8 only, 2 sq4 + sq8, 3 pq + sq8."""
        return self._rev.format

    @property
    def defaults(self) -> dict[str, int]:
        """The query defaults a query without explicit parameters uses."""
        return self._rev.manifest.get("defaults") or dict(LEGACY_DEFAULTS)

    @property
    def revision(self) -> RevState:
        """The revision state (manifest, slabs table, deltas, tombstones, segments) queries read."""
        return self._rev

    @property
    def has_dense(self) -> bool:
        """Whether the generation has a dense (vector) index. A lexical-only generation has none."""
        return has_dense(self._rev.manifest)

    @property
    def lex_config(self) -> ResolvedLexConfig | None:
        """The lexical / docs configuration (`manifest.lex`, defaults filled in), or None."""
        return resolve_lex_config(self._rev.manifest)

    @property
    def segments(self) -> list[dict[str, Any]]:
        """The lexical / docs segments of the revision, oldest first."""
        return self._rev.segments

    @property
    def attr_schema(self) -> AttrSchema | None:
        """The attribute schema (`manifest["attrs"]`), or None: the raw 32-bit word is then the one field, `attrs`."""
        a = self._rev.manifest.get("attrs")
        return validate_schema(a) if a else None

    def pack_attrs(self, values: Any) -> int:
        """Packs field values into a document's attrs word under the schema (`{"tenant": 5, "lang": "en"}`)."""
        return pack_attrs(self.attr_schema, values)

    def unpack_attrs(self, word: int) -> dict[str, Any]:
        """The field values of a hit's `attrs` word under the schema."""
        return unpack_attrs(self.attr_schema, word)

    def compile_filter(self, filter: AttrFilterExpr | None) -> Any:
        """Compiles a filter expression against the schema into the words `query(filter=...)` takes."""
        return compile_filter(self.attr_schema, filter)

    # ---- warm pack ----

    @property
    def warm_stats(self) -> dict[str, Any] | None:
        """How loading the warm pack went (`{key, state: loading|ready|failed, target: cache|layer, fetch, entries,
        seeded, skipped, seeded_bytes, ready_ms, error?}`), None without one."""
        return self._warm_stats.as_dict() if self._warm_stats is not None else None

    def _start_warm(self, target: Any, layer: WarmPrefixStore | None, got: Future, key: str, t0: float) -> None:
        """Seeds the warm-pack target (the cache store or `warm_layer`) once the warm pack's GET lands (only prefixes
        of slabs the current revision reads)."""
        self.warm_layer = layer
        self._warm_target = target
        st = self._warm_stats = WarmPackStats(key, target="layer" if layer is not None else "cache")
        ready: Future = Future()
        self.warm_ready = ready

        def done(f: Future) -> None:
            try:
                data, stat = f.result()
                st.fetch = stat
                if self._closed:
                    raise ThreadError("index closed while the warm pack loaded")
                with self._lock:
                    current = revision_slab_keys(self._rev)
                r = seed_warm_pack(data, target, current.__contains__)
                st.entries, st.seeded, st.skipped, st.seeded_bytes = r["entries"], r["seeded"], r["skipped"], \
                    r["seeded_bytes"]
                st.ready_ms = round((time.perf_counter() - t0) * 1000, 3)
                st.state = "ready"
            except BaseException as e:  # noqa: BLE001 - a warm pack never fails the index
                st.state = "failed"
                st.error = str(e)
                log.error("thread_sdk: warm pack %s not used: %s", key, e)
            ready.set_result(st.as_dict())

        got.add_done_callback(done)

    # ---- revisions ----

    def _check_open(self) -> None:
        if self._closed:
            raise ThreadError("index is closed", 400)

    def refresh(self) -> bool:
        """Re-reads `idx/current.json` (or the generation's head) and switches to the new revision if it changed
        (after loading what its first query would wait for, see `use_revision`). Returns True when it did. A pinned
        index (`gen` and `rev` given) never changes."""
        self._check_open()
        if self._pinned:
            return False
        with self._refresh_lock:
            inline: dict[str, Any] = {}
            g, r = resolve_target(self.store, self._gen_opt, None, None, inline)
            self._last_check = time.monotonic()
            if g == self._rev.gen and r == self._rev.rev:
                return False
            self.use_revision(g, r, inline.get("manifest"))
            return True

    def _maybe_refresh(self) -> None:
        """Starts a background refresh when `refresh_interval` has elapsed (queries never wait for it)."""
        if self._refresh_interval <= 0 or self._pinned or time.monotonic() - self._last_check <= self._refresh_interval:
            return
        with self._lock:
            if self._refreshing:
                return
            self._refreshing = True
            self._last_check = time.monotonic()

        def run() -> None:
            try:
                self.refresh()
            except Exception as e:  # noqa: BLE001 - a failed refresh keeps the current revision
                log.error("thread_sdk: refresh failed: %s", e)
            finally:
                with self._lock:
                    self._refreshing = False

        try:
            self._jobs.submit(run)
        except RuntimeError:  # closed
            with self._lock:
                self._refreshing = False

    def use_revision(self, gen: str, rev: int, manifest: dict[str, Any] | None = None) -> None:
        """Switches to (gen, rev); drops delta and lexical segments the new revision no longer lists. Before switching
        it loads what the first query on the new revision would otherwise wait for: the manifest, slabs table and
        tombstones, the base files when they changed (routing table, PQ codebook) and the new delta segments; then it
        warms the rest of the difference (`warm`) in the background. `manifest`: its manifest when already known (an
        inline copy in `idx/current.json`), saving the GET."""
        prev = self._rev
        nxt = load_rev_state(self.store, gen, rev, None, manifest)
        if has_dense(nxt.manifest):
            self._acquire_base(nxt, install=False)
        if sum(d.get("bytes", 0) for d in nxt.deltas) <= self._delta_budget:
            futs = [self._get_delta(d, BlobTrace(), nxt) for d in nxt.deltas]
            for f in futs:
                f.exception()
        with self._lock:
            self._rev = nxt
            if self._warm_target is not None:
                current = revision_slab_keys(nxt)
                self._warm_target.retain(current.__contains__)
            listed = {d["key"] for d in nxt.deltas}
            for k in [k for k in self._delta_cache if k not in listed]:
                del self._delta_cache[k]
            segs = {s["key"] for s in nxt.segments}
            for k in [k for k in self._seg_cache if k not in segs]:
                del self._seg_cache[k]
            self._lex = None
            if self._base is not None and self._base[0] != nxt.base_key:
                self._base = None
            pre = self._pre_base
            if pre is not None and pre[0] == nxt.base_key and self._base is None:
                self._base = pre
                self._cold = None  # the full table takes over from cold routing
            self._pre_base = None
            if self._cold is not None and self._cold[0] != _cold_key(nxt):
                self._cold = None
        before = set(snapshot_keys(prev))
        fresh = [k for k in snapshot_keys(nxt) if k not in before]
        if fresh and self._prefetch_bytes > 0:
            def bg() -> None:
                try:
                    self.warm(fresh)
                except Exception as e:  # noqa: BLE001 - prefetching is best effort
                    log.error("thread_sdk: prefetch failed: %s", e)

            try:
                self._jobs.submit(bg)
            except RuntimeError:
                pass

    def warm(self, keys_: Sequence[str] | None = None, *, budget: int | None = None,
             workers: int = 4) -> dict[str, int]:
        """Reads `keys_` (default: every object of the current revision, `snapshot_keys(index.revision)`) through the
        store so its cache holds them: slabs as a `fill` read of their scan prefix (the cache's miss policy decides
        whether the whole slab is cached now or in the background), every other object whole. Keys already cached are
        skipped; slab prefixes and delta / lexical segments stop at `budget` bytes (default `prefetch_bytes`, 64 MB).
        A store without a cache in front (a local directory) is not read. A revision switch runs this on the keys the
        new revision adds. Returns `{objects, bytes, skipped}`."""
        rev = self._rev
        out = {"objects": 0, "bytes": 0, "skipped": 0}
        if not self.store.remote and not isinstance(self.store, CachedBlobStore):
            return out
        left = [self._prefetch_bytes if budget is None else budget]
        # A slab costs its scan prefix, or the whole slab when a miss reads it whole (the "whole" policy).
        whole = isinstance(self.store, CachedBlobStore) and self.store.miss_policy == "whole"
        prefix_of: dict[str, int] = {}
        cost_of: dict[str, int] = {}
        if rev.slabs is not None:
            for cell, row in enumerate(rev.slabs):
                if row:
                    k = keys.slab(rev.gen, cell, slab_rev_of(row))
                    prefix_of[k] = row[0]
                    cost_of[k] = row[3] if whole else row[0]
        budgeted = {d["key"] for d in rev.deltas} | {s["key"] for s in rev.segments}
        todo = list(keys_ if keys_ is not None else snapshot_keys(rev))
        lock = threading.Lock()
        trace = BlobTrace()

        def worker() -> None:
            from .store import BlobRange
            while True:
                with lock:
                    if not todo:
                        return
                    k = todo.pop(0)
                if self.store.contains(k):
                    with lock:
                        out["skipped"] += 1
                    continue
                scan_len = prefix_of.get(k)
                is_budgeted = scan_len is not None or k in budgeted
                with lock:
                    if is_budgeted and left[0] <= 0:
                        out["skipped"] += 1
                        continue
                if scan_len is not None:
                    b = self.store.get(k, BlobRange(0, scan_len), trace=trace, fill=True)
                else:
                    b = self.store.get(k, trace=trace)
                if b is None:
                    continue
                with lock:
                    out["objects"] += 1
                    out["bytes"] += len(b)
                    if is_budgeted:
                        left[0] -= cost_of[k] if scan_len is not None else len(b)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, workers))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return out

    def _acquire_base(self, rev: RevState, loaded: dict[str, Any] | None = None, install: bool = True,
                      store: BlobStore | None = None) -> Base:
        """The routing table of `rev`'s base files, loaded once (concurrent callers share the load); `loaded["stat"]`
        is set when this call did the load. With `install=False` (a revision switch preparing its base) a new base is
        kept aside until the switch."""
        with self._lock:
            slot = self._base
            pre = self._pre_base
            if slot is not None and slot[0] == rev.base_key:
                fut, owner = slot[1], False
            elif pre is not None and pre[0] == rev.base_key:
                fut, owner = pre[1], False
            else:
                fut, owner = Future(), True
                slot = (rev.base_key, fut)
                if install:
                    self._base = slot
                else:
                    self._pre_base = slot
        if not owner:
            return fut.result()
        try:
            base = load_base(store or self._open_store or self.store, rev, self._io, loaded, self._routing)
        except BaseException as e:
            with self._lock:
                if self._base is slot:
                    self._base = None
                if self._pre_base is slot:
                    self._pre_base = None
            fut.set_exception(e)
            raise
        fut.set_result(base)
        return base

    # ---- cold routing ----

    def _cold_wanted(self, rev: RevState) -> bool:
        """Whether queries of `rev` may route cold (the option is on and the revision has a usable grouped table)."""
        return self._cold_routing == "tree" and has_dense(rev.manifest) and grouped_table_of(rev.manifest,
                                                                                           rev.files) is not None

    def _full_ready(self, rev: RevState) -> bool:
        with self._lock:
            slot = self._base
        return slot is not None and slot[0] == rev.base_key and slot[1].done() and slot[1].exception() is None

    def _acquire_query_base(self, rev: RevState, loaded: dict[str, Any]) -> Base:
        """The base a query uses: the full routing table once it is loaded; before that, with cold routing, the cold
        base (and the full table keeps loading in the background)."""
        if self._full_ready(rev) or not self._cold_wanted(rev):
            return self._acquire_base(rev, loaded)
        self._load_full_in_background()
        return self._acquire_cold(rev, loaded)

    def _load_full_in_background(self) -> None:
        """Starts loading the full routing table on a background thread (once per revision's base)."""
        with self._lock:
            if self._bg_load is not None and self._bg_load.is_alive():
                return
            rev = self._rev
            if self._base is not None and self._base[0] == rev.base_key:
                return

            # the open's prefetching view, while the open is still running (its codebook and coarse are in it)
            store = self._open_store or self.store

            def run() -> None:
                try:
                    self._acquire_base(rev, store=store)
                    with self._lock:
                        self._cold = None  # the full table takes over from cold routing
                except BaseException as e:  # noqa: BLE001 - cold routing continues
                    if not self._closed:
                        log.error("thread_sdk: loading the routing table failed (cold routing continues): %s", e)

            self._bg_load = threading.Thread(target=run, name="thread-routing-load", daemon=True)
            self._bg_load.start()

    def _acquire_cold(self, rev: RevState, loaded: dict[str, Any] | None = None) -> Base:
        key = _cold_key(rev)
        with self._lock:
            slot = self._cold
            if slot is not None and slot[0] == key:
                fut, owner = slot[1], False
            else:
                fut, owner = Future(), True
                slot = self._cold = (key, fut)
        if not owner:
            return fut.result()
        try:
            base = load_cold_base(self._open_store or self.store, rev, self._io, self._cold_beam, loaded)
        except BaseException as e:
            with self._lock:
                if self._cold is slot:
                    self._cold = None
            fut.set_exception(e)
            raise
        fut.set_result(base)
        return base

    @property
    def routing_state(self) -> str:
        """"full" once the routing table is loaded, "cold" while queries route cold, "none" before either."""
        if self._full_ready(self._rev):
            return "full"
        with self._lock:
            c = self._cold
        return "cold" if c is not None and c[1].done() and c[1].exception() is None else "none"

    def when_full_routing(self) -> None:
        """Waits until the full routing table is loaded (starting the load if needed)."""
        if self.has_dense:
            self._acquire_base(self._rev)
            with self._lock:
                self._cold = None

    def _get_delta(self, ref: dict[str, Any], trace: BlobTrace, rev: RevState | None = None) -> Future:
        key = ref["key"]
        with self._lock:
            hit = self._delta_cache.get(key)
            if hit is not None:
                return hit

        def read() -> BytesLike:
            b = self.store.get(key, None, trace=trace)
            if b is None:
                raise ThreadError(f"missing delta segment {key}", 404)
            return b

        fut = self._io.submit(read)
        if sum(d.get("bytes", 0) for d in (rev or self._rev).deltas) <= self._delta_budget:
            with self._lock:
                self._delta_cache.setdefault(key, fut)

            def forget(f: Future) -> None:
                if f.exception() is not None:
                    with self._lock:
                        if self._delta_cache.get(key) is f:
                            del self._delta_cache[key]

            fut.add_done_callback(forget)
        return fut

    # ---- dense ----

    def query(
        self,
        vector: Any,
        *,
        top_k: int = 10,
        nprobe: int | None = None,
        rerank: int | None = None,
        pc: int | None = None,
        return_payload: bool = False,
        inflight: int | None = None,
        speculate: bool = True,
        filter: Any = None,
        max_nprobe: int | None = None,
        coalesce_bytes: int | None = None,
        span_bytes: int | None = None,
        cold_policy: dict[str, Any] | Literal[False] | None = None,
        beam: int | None = None,
    ) -> QueryResult:
        """Nearest neighbours of `vector` (any norm; normalized): a list of floats, an `array('f')`, a numpy array or
        raw f32 bytes of `dim` values. `nprobe` slabs are scanned, the best `rerank` scan-tier candidates re-scored
        with the int8 tier (0 = scan-tier scores only), `pc` coarse groups probed (0 = flat routing); absent ones take
        the manifest defaults. `filter`: an attribute filter (field conditions under the schema, e.g. `{"tenant": 5,
        "lang": {"$in": ["en", "fr"]}}`, or compiled words) applied inside every scan before top-k insertion; with
        one, probing widens past `nprobe` in routing order until `max(top_k, rerank)` candidates match or
        `max_nprobe` cells were scanned (default 4 x nprobe). `inflight` (default: the store's, 32 against S3, else
        12), `coalesce_bytes` / `span_bytes` (phase-2 range merging) and `cold_policy` (`{"nprobe", "rerank",
        "threshold"}`, applied when more than `threshold` of the probed slabs are not cached; `False` disables an
        index-wide one) tune the I/O. `beam`: groups ranked by a cold-routed query (`cold_routing="tree"`, before the
        full routing table has loaded; default `cold_beam`)."""
        self._check_open()
        if not self.has_dense:
            raise ThreadError("this generation has no dense index (lexical only): use search(text=...)", 400)
        self._maybe_refresh()
        rev = self._rev
        loaded: dict[str, Any] = {}
        base = self._acquire_query_base(rev, loaded)
        ctx = QueryContext(self.store, rev, base, self._io, self._get_delta, self._cold_policy)
        words = filter if filter is None or not isinstance(filter, Mapping) else compile_filter(
            validate_schema(rev.manifest["attrs"]) if rev.manifest.get("attrs") else None, filter)
        r = execute_query(ctx, vector, top_k=top_k, nprobe=nprobe, rerank=rerank, pc=pc,
                          return_payload=return_payload, inflight=inflight, speculate=speculate, filter=words,
                          max_nprobe=max_nprobe, coalesce_bytes=coalesce_bytes, span_bytes=span_bytes,
                          cold_policy=cold_policy, beam=beam)
        if "stat" in loaded:  # this query (re)loaded the routing table (a switch to another generation)
            r.stats["timeline"]["base"] = loaded["stat"]
        return r

    def query_many(self, vectors: Iterable[Any], **opts: Any) -> list[QueryResult]:
        """Several queries against the same snapshot and routing table, concurrently."""
        futs = [self._jobs.submit(self.query, v, **opts) for v in vectors]
        return [f.result() for f in futs]

    # ---- lexical ----

    def _lex_index(self) -> LexIndex:
        """The lexical index of the current revision (segments opened on first use and cached by key)."""
        cfg = self.lex_config
        if cfg is None:
            raise ThreadError("this generation has no lexical index (index.configure_lexical first)", 400)
        with self._lock:
            rev = self._rev
            if self._lex is not None and self._lex[0] == rev.gen and self._lex[1] == rev.rev:
                fut = self._lex[2]
                owner = False
            else:
                fut = Future()
                self._lex = (rev.gen, rev.rev, fut)
                owner = True
        if not owner:
            return fut.result()
        try:
            io = LexIo()
            seg_futs = []
            with self._lock:
                for ref in rev.segments:
                    f = self._seg_cache.get(ref["key"])
                    if f is None:
                        f = self._io.submit(LexSegment.open, self.store, ref, io, None,
                                            transient_budget=self._lex_transient)
                        self._seg_cache[ref["key"]] = f
                    seg_futs.append(f)
            segs = []
            for ref, f in zip(rev.segments, seg_futs):
                try:
                    segs.append(f.result())
                except BaseException:
                    with self._lock:
                        if self._seg_cache.get(ref["key"]) is f:
                            del self._seg_cache[ref["key"]]
                    raise
            index = LexIndex(cfg, segs, rev.tomb, self._io)
        except BaseException as e:
            with self._lock:
                if self._lex is not None and self._lex[2] is fut:
                    self._lex = None
            fut.set_exception(e)
            raise
        fut.set_result(index)
        return index

    def lexical_search(self, text: str, k: int = 10, *, filter: Any = None) -> LexSearchResult:
        """BM25 top `k` for `text` over the revision's lexical segments; `filter` is an attribute filter expression
        or its compiled words (see `LexIndex.search`)."""
        self._check_open()
        words = compile_filter(self.attr_schema, filter) if isinstance(filter, Mapping) else filter
        return self._lex_index().search(text, k, filter=words)

    def tokenize(self, text: str) -> list[str]:
        """The query's tokens under the index's tokenizer."""
        return self._lex_index().tokenize(text)

    def docs(self, ids: Iterable[int], *, text: bool = False) -> dict[int, DocRecord]:
        """Resolves ids through the `docs` store (newest live copy of each id); ids without an entry are absent."""
        return self._docs(ids, text=text)[0]

    def _docs(self, ids: Iterable[int], *, text: bool) -> tuple[dict[int, DocRecord], LexIo]:
        self._check_open()
        if self.lex_config is None:
            return {}, LexIo()
        return self._lex_index().docs(ids, text=text)

    # ---- keys ----

    def resolve_keys(self, keys_: Iterable[str]) -> dict[str, int]:
        """Resolves string keys (`upsert(keys=...)`) to ids through the `keys` sections, newest segment first; keys
        with no live document are absent."""
        self._check_open()
        if self.lex_config is None:
            return {}
        return self._lex_index().resolve_keys(keys_)[0]

    def get(self, keys_: Iterable[str], *, text: bool = False) -> dict[str, KeyRecord]:
        """Fetches documents by key: each key's id (`resolve_keys`) and its `docs` entry (`docs`), two rounds against
        cached objects. Keys with no live document are absent; with `text`, the stored text comes along."""
        self._check_open()
        if self.lex_config is None:
            return {}
        lex = self._lex_index()
        ids, _ = lex.resolve_keys(keys_)
        if not ids:
            return {}
        docs, _ = lex.docs(ids.values(), text=text)
        out: dict[str, KeyRecord] = {}
        for key, id in ids.items():
            d = docs.get(id)
            out[key] = KeyRecord(id, d.meta, d.text, d.attrs, key) if d is not None else KeyRecord(id, "", None, 0, key)
        return out

    def list_keys(self, prefix: str = "", *, after: str | None = None, limit: int = 100) -> KeyPage:
        """Lists keys with a prefix in key order, paged: continue with `after=page.next` while `page.more`."""
        self._check_open()
        if self.lex_config is None:
            return KeyPage([], False)
        return self._lex_index().list_keys(prefix, after, limit)[0]

    def delete_keys(self, keys_: Iterable[str]) -> DeleteResult:
        """Deletes documents by key (their vectors, texts, docs entries and keys): one tombstone set in a new manifest
        revision (none when no key has a live document)."""
        r = delete_keys(self._writer(), keys_)
        if r.key:
            self.use_revision(r.gen, r.rev)
        return r

    def _lex_index_for(self, manifest: dict[str, Any]) -> LexIndex:
        """A lexical index over another revision's segments (the head a writer reads), sharing the segment cache."""
        rev = self._rev
        if manifest.get("gen") == rev.gen and (manifest.get("rev") or 0) == rev.rev:
            return self._lex_index()
        cfg = resolve_lex_config(manifest)
        if cfg is None:
            raise ThreadError("this generation has no lexical index (index.configure_lexical first)", 400)
        state = load_rev_state(self.store, manifest["gen"], manifest.get("rev") or 0, None, manifest)
        io = LexIo()
        seg_futs = []
        with self._lock:
            for ref in state.segments:
                f = self._seg_cache.get(ref["key"])
                if f is None:
                    f = self._io.submit(LexSegment.open, self.store, ref, io, None, transient_budget=self._lex_transient)
                    self._seg_cache[ref["key"]] = f
                seg_futs.append(f)
        segs = []
        for ref, f in zip(state.segments, seg_futs):
            try:
                segs.append(f.result())
            except BaseException:
                with self._lock:
                    if self._seg_cache.get(ref["key"]) is f:
                        del self._seg_cache[ref["key"]]
                raise
        return LexIndex(cfg, segs, state.tomb, self._io)

    # ---- unified search ----

    def search(
        self,
        text: str | None = None,
        vector: Any = None,
        *,
        top_k: int = 10,
        recall_k: int | None = None,
        rrf_k: float | None = None,
        weights: dict[str, float] | None = None,
        dense: dict[str, Any] | None = None,
        docs: bool | Literal["text"] = False,
        filter: AttrFilterExpr | None = None,
    ) -> SearchResult:
        """Searches by text, by vector, or by both: the dense list (`query`, options in `dense`) and the lexical list
        (BM25 over the segments) run concurrently at depth `recall_k` and are fused with Reciprocal Rank Fusion
        (`rrf_k`, list `weights` `{dense, lexical}`; defaults from `manifest.lex.defaults`); a single list is returned
        as is. With `docs`, hits are resolved through the `docs` store (`"text"`: the stored text too). `filter`
        (field conditions under the attribute schema) applies to both lists: the dense scan drops non-matching
        entries before top-k insertion (widening its probes up to `dense["max_nprobe"]`), the BM25 list is taken
        deeper and resolved through the `docs` store."""
        self._check_open()
        t0 = time.perf_counter()
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ThreadError("top_k must be an integer >= 1", 400)
        has_text = isinstance(text, str) and text.strip() != ""
        has_vector = vector is not None
        if not has_text and not has_vector:
            raise ThreadError("search needs a text, a vector or both", 400)
        self._maybe_refresh()
        cfg = self.lex_config
        if has_text and cfg is None:
            raise ThreadError("this generation has no lexical index (index.configure_lexical first)", 400)
        if has_vector and not self.has_dense:
            raise ThreadError("this generation has no dense index", 400)
        if recall_k is None:
            recall_k = cfg.recall_k if cfg and cfg.recall_k is not None else max(top_k, 50)
        if rrf_k is None:
            rrf_k = cfg.rrf_k if cfg else DEFAULT_RRF_K
        w = weights or {}
        wts = {"dense": w.get("dense", cfg.weights["dense"] if cfg else 1),
               "lexical": w.get("lexical", cfg.weights["lexical"] if cfg else 1)}
        both = has_text and has_vector
        depth = max(recall_k, top_k) if both else top_k
        timing = {"dense": 0.0, "lexical": 0.0, "docs": 0.0}
        # The filter is compiled once and applied to both lists.
        words = compile_filter(self.attr_schema, filter) if filter else None

        def run_lex() -> LexSearchResult:
            t = time.perf_counter()
            r = self.lexical_search(text, depth, filter=words)  # type: ignore[arg-type]
            timing["lexical"] = round((time.perf_counter() - t) * 1000, 3)
            return r

        lex_fut = self._jobs.submit(run_lex) if both else None
        dense_r: QueryResult | None = None
        lex_r: LexSearchResult | None = None
        if has_vector:
            t = time.perf_counter()
            dense_r = self.query(vector, **{**(dense or {}), "top_k": depth, **({"filter": words} if words else {})})
            timing["dense"] = round((time.perf_counter() - t) * 1000, 3)
        if lex_fut is not None:
            lex_r = lex_fut.result()
        elif has_text:
            lex_r = run_lex()

        hits: list[SearchHit]
        mode: Literal["dense", "lexical", "hybrid"]
        if both:
            mode = "hybrid"
            assert dense_r is not None and lex_r is not None
            fused = rrf([(wts["dense"], [h.id for h in dense_r.hits]), (wts["lexical"], [h.id for h in lex_r.hits])],
                        rrf_k, top_k)
            d_rank = {h.id: Rank(i + 1, h.score) for i, h in enumerate(dense_r.hits)}
            l_rank = {h.id: Rank(i + 1, h.score) for i, h in enumerate(lex_r.hits)}
            hits = [SearchHit(i, s, d_rank.get(i), l_rank.get(i)) for i, s in fused]
        elif dense_r is not None:
            mode = "dense"
            hits = [SearchHit(h.id, h.score, dense=Rank(i + 1, h.score)) for i, h in enumerate(dense_r.hits[:top_k])]
        else:
            mode = "lexical"
            assert lex_r is not None
            hits = [SearchHit(h.id, h.score, lexical=Rank(i + 1, h.score)) for i, h in enumerate(lex_r.hits[:top_k])]
        stats: dict[str, Any] = {}
        if docs and hits:
            t = time.perf_counter()
            found, io = self._docs([h.id for h in hits], text=docs == "text")
            for h in hits:
                h.doc = found.get(h.id)
            stats["docs_io"] = io.as_dict()
            timing["docs"] = round((time.perf_counter() - t) * 1000, 3)
        stats["timing"] = {"total": round((time.perf_counter() - t0) * 1000, 3), **timing}
        return SearchResult(hits, mode, {"top_k": top_k, "recall_k": depth, "rrf_k": rrf_k, "weights": wts},
                            dense_r, lex_r, stats)

    # ---- writes ----

    def _writer(self) -> WriterContext:
        self._check_open()

        def borrow_pq(base_key: str) -> Any:
            with self._lock:
                slot = self._base
            if slot is not None and slot[0] == base_key and slot[1].done() and slot[1].exception() is None:
                return slot[1].result().pq
            return None

        def resolve_keys(manifest: dict[str, Any], keys_: Sequence[str]) -> dict[str, int]:
            return self._lex_index_for(manifest).resolve_keys(keys_)[0]

        return WriterContext(self.store, self._rev.gen, self._io, borrow_pq, resolve_keys)

    def upsert(
        self,
        ids: Iterable[int] | None = None,
        vectors: Any = None,
        *,
        attrs: Iterable[Any] | None = None,
        payloads: Sequence[bytes | str] | None = None,
        texts: Sequence[str | None] | None = None,
        metas: Sequence[str | None] | None = None,
        keys: Sequence[str] | None = None,
        replace: bool = True,
    ) -> UpsertResult:
        """Adds or replaces documents: one delta segment (vectors: `n * dim` floats flat, one row per id, or an
        `(n, dim)` numpy array; any norm), one lexical/docs/keys segment (`texts` indexed by BM25, `metas` stored in
        the docs store, `keys` unique string keys; needs `configure_lexical`) and, with `replace` (default), a
        tombstone set hiding older copies of the ids, committed as a new manifest revision; the index then reads
        that revision. With `keys` and no `ids`, a known key keeps its id and a new one is allocated the next id
        (`manifest.nextId`); the result lists the ids. `attrs`: one attribute word per document, as field values
        under the schema (`{"tenant": 5, "lang": "en"}`) or a raw u32; stored with the vector and in the docs store,
        and what `filter` tests. Up to 4096 per call."""
        r = upsert(self._writer(), ids, vectors, attrs=attrs, payloads=payloads, texts=texts, metas=metas,
                   keys_=keys, replace=replace)
        self.use_revision(r.gen, r.rev)
        return r

    def delete(self, ids: Iterable[int]) -> DeleteResult:
        """Deletes ids (their vectors, texts and docs entries): one tombstone set in a new manifest revision."""
        r = delete_ids(self._writer(), ids)
        self.use_revision(r.gen, r.rev)
        return r

    def configure_lexical(self, config: dict[str, Any] | None = None, **kwargs: Any) -> ConfigureLexicalResult:
        """Gives the generation a lexical index / docs store, or changes its configuration for future segments: a
        manifest revision with `lex`. `config` is the wire-format dict (`{"bucketTerms": 64}`); keyword arguments
        may use snake_case (`bucket_terms=64, text=True, lengths="byte", defaults={"rrf_k": 10}`). Existing
        documents are not indexed retroactively: upsert them again with their texts."""
        r = configure_lexical(self._writer(), lex_config_from_kwargs(config, kwargs))
        self.use_revision(r.gen, r.rev)
        return r

    def configure_attrs(self, spec: Mapping[str, Any] | None = None, **fields: Any) -> ConfigureAttrsResult:
        """Declares (or extends) the attribute schema: named fields packed into each document's 32-bit attrs word, in
        declaration order (`configure_attrs(tenant={"bits": 12}, lang={"values": ["en", "fr"]}, public="bool")`, or a
        schema dict), as a manifest revision. Existing fields keep their layout (an enum may append values);
        documents written before keep their words."""
        r = configure_attrs(self._writer(), {**(spec or {}), **fields})
        self.use_revision(r.gen, r.rev)
        return r

    def compact_segments(self) -> CompactSegmentsResult:
        """Merges every lexical/docs segment of the head revision into one base segment (tombstoned documents
        dropped, replaced ids resolved to their newest copy) and commits a revision listing only it."""
        r = compact_segments(self._writer())
        self.use_revision(r.gen, r.rev)
        return r

    # ---- lifecycle ----

    def close(self) -> None:
        """Releases the thread pools and the native objects. The index cannot be used afterwards."""
        if self._closed:
            return
        self._closed = True
        self._io.shutdown(wait=False, cancel_futures=True)
        self._jobs.shutdown(wait=False, cancel_futures=True)
        if isinstance(self.store, CachedBlobStore):
            self.store.close()
        with self._lock:
            self._base = None
            self._cold = None
            self._delta_cache.clear()
            self._seg_cache.clear()
            self._lex = None

    def __enter__(self) -> "ThreadIndex":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<ThreadIndex gen={self.gen!r} rev={self.rev} dim={self.dim} cells={self.cells} format={self.format}>"


def open_index(
    store: BlobStore | str | os.PathLike[str],
    *,
    prefix: str = "",
    gen: str | None = None,
    rev: int | None = None,
    cache: BlobCache | bool | None = None,
    refresh_interval: float = 30.0,
    delta_cache_bytes: int = 40 * 1024 * 1024,
    lex_transient_bytes: int = 8 * 1024 * 1024,
    io_workers: int = 32,
    job_workers: int | None = None,
    miss_policy: str | None = None,
    warm_concurrency: int = 2,
    warmup: bool | int | None = None,
    cold_policy: dict[str, Any] | None = None,
    prefetch_bytes: int = DEFAULT_PREFETCH_BYTES,
    routing: str = "f32",
    cold_routing: str = "off",
    cold_beam: int | None = None,
    base_pack: bool = True,
    warm_pack: str | None = None,
) -> ThreadIndex:
    """Opens the index under `store`: a `BlobStore`, or a path to a directory holding an index layout or to a pack
    file (`store_at`).

    - `prefix`: key prefix in front of `idx/` (several indexes in one bucket);
    - `gen` / `rev`: read a specific generation / revision (default: `idx/current.json`); with both, the index is
      pinned and never refreshes;
    - `cache`: a `MemoryCache` / `DiskCache` in front of the store (default: a 256 MB `MemoryCache` for remote
      stores, none for local ones; `True` for the default cache, `False` for none);
    - `refresh_interval`: seconds after which a query re-reads `current.json` first (0 = only `refresh()`);
    - `delta_cache_bytes`: delta segments kept in memory across queries;
    - `lex_transient_bytes`: transient lexical bytes (buckets, posting lists, texts) each segment reader may hold;
    - `io_workers`: threads for reads (shared by concurrent queries; each query keeps at most `inflight` in flight);
    - `job_workers`: threads for `query_many` and the lexical half of a hybrid search;
    - `miss_policy`: what a slab-prefix cache miss reads: "whole", "range" or "range+warm" (default: the cache's,
      "range+warm" for `MemoryCache` / `DiskCache`); `warm_concurrency`: background whole-object fetches at once;
    - `warmup`: open connections to a remote store while opening (True = the store's `inflight`, an int = that many,
      False = none; default on for remote stores);
    - `cold_policy`: `{"nprobe", "rerank", "threshold"}` applied to queries whose probed slabs are mostly not cached
      (off by default);
    - `prefetch_bytes`: slab prefixes / delta and lexical segments a revision switch prefetches into the cache;
    - `routing`: the routing table to load: "f32" (default, `fine.f32`), "f16" / "i8" (the compact copies, half and a
      quarter of the bytes, decoded to f32 on load) or "auto" (the smallest the revision has); a revision without the
      table falls back to f32. It only picks which cells to probe: slabs are scored with their own f32 means;
    - `cold_routing`: "tree" makes a cold open wait only for the centroid tree and the PQ codebook; until the full
      routing table has loaded (in the background), queries rank the cells of the tree's best `cold_beam` groups,
      range-reading just those groups' means from the tree-grouped table (needs `manifest.routing.grouped`; else the
      open loads the full table as usual). `cold_beam` defaults to `manifest.defaults.beam`, else 16;
    - `base_pack`: (default True) a cold open first GETs the revision's base pack (`idx/{gen}/base.{rev}.threadpack`,
      `base.cold.{rev}` with `cold_routing="tree"`), every object the open reads in one object, in parallel with the
      manifest; from a `PackedBlobStore` it reads the pack's recorded base range. Without a pack, the open starts
      every read as soon as the manifest is in. False: the reads one after another, as before;
    - `warm_pack`: key of a warm pack (`meta.kind: "warm"`, see warm.py): the scan prefixes of the most-probed slabs in
      one object. `open` starts ONE GET of it on a background thread and does not wait for it; once it is in, its
      prefixes (those of slabs the open revision reads) are served from memory, so a query probing those cells skips
      their prefix reads (phase-2 reads of those slabs still go to the store). They go into the in-process cache when
      it holds prefixes (`MemoryCache`: a prefix hit of a fill read then warms the whole slab in the background under
      the cache's miss policy, and counts as resident for `cold_policy`), else into a prefix layer in front of the
      store (`index.warm_layer`). Read from `store` (with `prefix`, around an in-process cache), or from the inner
      store of a `PackedBlobStore`. `index.warm_ready` (a Future) resolves once it is seeded; `index.warm_stats` says
      how it went. Measured: no gain on the dbpedia workload (docs/cold-start.md, "Warm packs"); opt-in.
    """
    return ThreadIndex.open(store, prefix=prefix, gen=gen, rev=rev, cache=cache, refresh_interval=refresh_interval,
                            delta_cache_bytes=delta_cache_bytes, lex_transient_bytes=lex_transient_bytes,
                            io_workers=io_workers, job_workers=job_workers, miss_policy=miss_policy,
                            warm_concurrency=warm_concurrency, warmup=warmup, cold_policy=cold_policy,
                            prefetch_bytes=prefetch_bytes, routing=routing, cold_routing=cold_routing,
                            cold_beam=cold_beam, base_pack=base_pack, warm_pack=warm_pack)
