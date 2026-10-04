"""`ThreadIndex`: one opened index, the Python counterpart of the TypeScript SDK's `ThreadIndex`."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

from . import _native
from .fusion import DEFAULT_RRF_K, rrf
from .lex import DocRecord, LexIndex, LexIo, LexSearchResult, LexSegment
from .query import QUERY_INFLIGHT, Base, QueryContext, QueryResult, RevState, execute_query
from .store import BlobCache, BlobStore, BlobTrace, CachedBlobStore, MemoryCache, PrefixedBlobStore, phase_stat
from .pack import PackedBlobStore, store_at
from .revision import get_required, head_revision, load_rev_state, resolve_target
from .warm import WarmPrefixStore, WarmStats, fetch_warm_pack, revision_slab_keys, seed_warm_pack
from .wire import LEGACY_DEFAULTS, BytesLike, ResolvedLexConfig, ThreadError, has_dense, resolve_lex_config
from .write import (
    CompactSegmentsResult,
    ConfigureLexicalResult,
    DeleteResult,
    UpsertResult,
    WriterContext,
    compact_segments,
    configure_lexical,
    delete_ids,
    lex_config_from_kwargs,
    upsert,
)

__all__ = ["ThreadIndex", "open_index", "SearchHit", "SearchResult", "Rank", "resolve_target", "load_rev_state",
           "head_revision"]
_ = (resolve_target, load_rev_state, head_revision)  # re-exported from here for compatibility

log = logging.getLogger("thread_sdk")


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


def load_base(store: BlobStore, rev: RevState, io: ThreadPoolExecutor, loaded: dict[str, Any] | None = None) -> Base:
    """Loads the routing table (and PQ codebook) of a revision's generation into native memory. With `loaded`, records
    the load as `loaded["stat"]`: `phase_stat` fields plus `fetch` (ms reading the files, concurrently), `build` (ms
    building the native objects) and `files` (`{coarse, fine, pq}` bytes)."""
    m, files = rev.manifest, rev.files
    t0 = time.perf_counter()
    trace = BlobTrace()
    fc = io.submit(get_required, store, files["coarse"], trace)
    ff = io.submit(get_required, store, files["fine"], trace)
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
    for name, b, n in (("coarse", c, m["k1"]), ("fine", f, m["k1"] * m["k2"])):
        if len(b) != n * m["dim"] * 4:
            raise ThreadError(f"{files[name]} is {len(b)} bytes, expected {n} x {m['dim']} f32")
    searcher = _native.Searcher(m["dim"], m["k1"], m["k2"], c, f)
    t2 = time.perf_counter()
    if loaded is not None:
        loaded["stat"] = {**phase_stat((t2 - t0) * 1000, trace), "fetch": round((t1 - t0) * 1000, 3),
                          "build": round((t2 - t1) * 1000, 3),
                          "files": {"coarse": len(c), "fine": len(f), "pq": len(p) if p is not None else 0}}
    return Base(rev.base_key, searcher, pq)


class ThreadIndex:
    """An opened index: one (gen, rev) snapshot plus the generation's routing table in native memory. Safe to share
    between threads; `query`, `search` and the write methods may run concurrently."""

    def __init__(self, store: BlobStore, rev: RevState, *, gen: str | None, pinned: bool, refresh_interval: float,
                 delta_cache_bytes: int, lex_transient_bytes: int, io_workers: int, job_workers: int) -> None:
        self.store = store
        self._rev = rev
        self._gen_opt = gen
        self._pinned = pinned
        self._refresh_interval = refresh_interval
        self._delta_budget = delta_cache_bytes
        self._lex_transient = lex_transient_bytes
        self._io = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="thread-io")
        self._jobs = ThreadPoolExecutor(max_workers=job_workers, thread_name_prefix="thread-job")
        self._lock = threading.RLock()
        #: (base key, future of the Base) of the routing table queries use.
        self._base: tuple[str, Future] | None = None
        self._delta_cache: dict[str, Future] = {}
        self._seg_cache: dict[str, Future] = {}
        self._lex: tuple[str, int, Future] | None = None
        self._last_check = time.monotonic()
        self._refresh_lock = threading.Lock()
        self._closed = False
        #: Where `open` spent its time and reads: `{total, current, manifest, slabs, base}` (see `open_index`).
        self.open_stats: dict[str, Any] = {}
        #: The warm-pack layer (`warm_pack=`), None without one.
        self.warm: WarmPrefixStore | None = None
        self._warm_stats: WarmStats | None = None
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
        warm_pack: str | None = None,
    ) -> "ThreadIndex":
        """Opens the index under `store` (see `open_index`)."""
        if not isinstance(store, BlobStore):
            store = store_at(store)
        packed = store if isinstance(store, PackedBlobStore) else None
        pack = packed.open_stats if packed is not None else None
        if prefix:
            store = PrefixedBlobStore(store, prefix)
        if cache is None:
            cache = MemoryCache() if store.remote else False
        elif cache is True:
            cache = MemoryCache()
        if cache:
            store = CachedBlobStore(store, cache)  # type: ignore[arg-type]
        t0 = time.perf_counter()
        warm: tuple[WarmPrefixStore, Future] | None = None
        if warm_pack:
            # The warm pack's one GET starts first and runs alongside the whole open; nothing waits for it.
            source = packed.inner if packed is not None else store
            got: Future = Future()

            def fetch() -> None:
                try:
                    got.set_result(fetch_warm_pack(source, warm_pack, t0))
                except BaseException as e:  # noqa: BLE001 - reported through warm_stats
                    got.set_exception(e)

            threading.Thread(target=fetch, name="thread-warm", daemon=True).start()
            warm = (WarmPrefixStore(store), got)
            store = warm[0]
        timeline: dict[str, Any] = {}
        inline: dict[str, Any] = {}
        g, r = resolve_target(store, gen, rev, timeline, inline)
        state = load_rev_state(store, g, r, timeline, inline.get("manifest"))
        ix = cls(store, state, gen=gen, pinned=gen is not None and rev is not None, refresh_interval=refresh_interval,
                 delta_cache_bytes=delta_cache_bytes, lex_transient_bytes=lex_transient_bytes, io_workers=io_workers,
                 job_workers=job_workers or max(4, min(16, os.cpu_count() or 4)))
        if warm is not None:
            ix._start_warm(warm[0], warm[1], warm_pack, t0)  # type: ignore[arg-type]
        loaded: dict[str, Any] = {}
        try:
            if has_dense(state.manifest):
                ix._acquire_base(state, loaded)
        except BaseException:
            ix.close()
            raise
        none = phase_stat(0, BlobTrace())
        ix.open_stats = {"total": round((time.perf_counter() - t0) * 1000, 3),
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

    # ---- warm pack ----

    @property
    def warm_stats(self) -> dict[str, Any] | None:
        """How loading the warm pack went (`{key, state: loading|ready|failed, fetch, entries, seeded, skipped,
        seeded_bytes, ready_ms, error?}`), None without one."""
        return self._warm_stats.as_dict() if self._warm_stats is not None else None

    def _start_warm(self, layer: WarmPrefixStore, got: Future, key: str, t0: float) -> None:
        """Seeds the warm layer once the warm pack's GET lands (only prefixes of slabs the current revision reads)."""
        self.warm = layer
        st = self._warm_stats = WarmStats(key)
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
                r = seed_warm_pack(data, layer, current.__contains__)
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
        """Re-reads `idx/current.json` (or the generation's head) and switches to the new revision if it changed.
        Returns True when it did. A pinned index (`gen` and `rev` given) never changes."""
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
        if self._refresh_interval > 0 and not self._pinned and time.monotonic() - self._last_check > self._refresh_interval:
            try:
                self.refresh()
            except Exception as e:  # noqa: BLE001 - a failed refresh keeps the current revision
                log.error("thread_sdk: refresh failed: %s", e)

    def use_revision(self, gen: str, rev: int, manifest: dict[str, Any] | None = None) -> None:
        """Switches to (gen, rev); drops delta and lexical segments the new revision no longer lists. `manifest`: its
        manifest when already known (an inline copy in `idx/current.json`), saving the GET."""
        nxt = load_rev_state(self.store, gen, rev, None, manifest)
        with self._lock:
            self._rev = nxt
            if self.warm is not None:
                current = revision_slab_keys(nxt)
                self.warm.retain(current.__contains__)
            listed = {d["key"] for d in nxt.deltas}
            for k in [k for k in self._delta_cache if k not in listed]:
                del self._delta_cache[k]
            segs = {s["key"] for s in nxt.segments}
            for k in [k for k in self._seg_cache if k not in segs]:
                del self._seg_cache[k]
            self._lex = None
            if self._base is not None and self._base[0] != nxt.base_key:
                self._base = None

    def _acquire_base(self, rev: RevState, loaded: dict[str, Any] | None = None) -> Base:
        """The routing table of `rev`'s base files, loaded once (concurrent callers share the load); `loaded["stat"]`
        is set when this call did the load."""
        with self._lock:
            slot = self._base
            if slot is not None and slot[0] == rev.base_key:
                fut, owner = slot[1], False
            else:
                fut, owner = Future(), True
                slot = self._base = (rev.base_key, fut)
        if not owner:
            return fut.result()
        try:
            base = load_base(self.store, rev, self._io, loaded)
        except BaseException as e:
            with self._lock:
                if self._base is slot:
                    self._base = None
            fut.set_exception(e)
            raise
        fut.set_result(base)
        return base

    def _get_delta(self, ref: dict[str, Any], trace: BlobTrace) -> Future:
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
        if sum(d.get("bytes", 0) for d in self._rev.deltas) <= self._delta_budget:
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
        inflight: int = QUERY_INFLIGHT,
        speculate: bool = True,
    ) -> QueryResult:
        """Nearest neighbours of `vector` (any norm; normalized): a list of floats, an `array('f')`, a numpy array or
        raw f32 bytes of `dim` values. `nprobe` slabs are scanned, the best `rerank` scan-tier candidates re-scored
        with the int8 tier (0 = scan-tier scores only), `pc` coarse groups probed (0 = flat routing); absent ones take
        the manifest defaults."""
        self._check_open()
        if not self.has_dense:
            raise ThreadError("this generation has no dense index (lexical only): use search(text=...)", 400)
        self._maybe_refresh()
        rev = self._rev
        loaded: dict[str, Any] = {}
        base = self._acquire_base(rev, loaded)
        ctx = QueryContext(self.store, rev, base, self._io, self._get_delta)
        r = execute_query(ctx, vector, top_k=top_k, nprobe=nprobe, rerank=rerank, pc=pc,
                          return_payload=return_payload, inflight=inflight, speculate=speculate)
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

    def lexical_search(self, text: str, k: int = 10) -> LexSearchResult:
        """BM25 top `k` for `text` over the revision's lexical segments."""
        self._check_open()
        return self._lex_index().search(text, k)

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
    ) -> SearchResult:
        """Searches by text, by vector, or by both: the dense list (`query`, options in `dense`) and the lexical list
        (BM25 over the segments) run concurrently at depth `recall_k` and are fused with Reciprocal Rank Fusion
        (`rrf_k`, list `weights` `{dense, lexical}`; defaults from `manifest.lex.defaults`); a single list is returned
        as is. With `docs`, hits are resolved through the `docs` store (`"text"`: the stored text too)."""
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

        def run_lex() -> LexSearchResult:
            t = time.perf_counter()
            r = self.lexical_search(text, depth)  # type: ignore[arg-type]
            timing["lexical"] = round((time.perf_counter() - t) * 1000, 3)
            return r

        lex_fut = self._jobs.submit(run_lex) if both else None
        dense_r: QueryResult | None = None
        lex_r: LexSearchResult | None = None
        if has_vector:
            t = time.perf_counter()
            dense_r = self.query(vector, **{**(dense or {}), "top_k": depth})
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

        return WriterContext(self.store, self._rev.gen, self._io, borrow_pq)

    def upsert(
        self,
        ids: Iterable[int],
        vectors: Any = None,
        *,
        attrs: Iterable[int] | None = None,
        payloads: Sequence[bytes | str] | None = None,
        texts: Sequence[str | None] | None = None,
        metas: Sequence[str | None] | None = None,
        replace: bool = True,
    ) -> UpsertResult:
        """Adds or replaces documents: one delta segment (vectors: `n * dim` floats flat, one row per id, or an
        `(n, dim)` numpy array; any norm), one lexical/docs segment (`texts` indexed by BM25, `metas` stored in the
        docs store; needs `configure_lexical`) and, with `replace` (default), a tombstone set hiding older copies of
        the ids, committed as a new manifest revision; the index then reads that revision. Up to 4096 per call."""
        r = upsert(self._writer(), ids, vectors, attrs=attrs, payloads=payloads, texts=texts, metas=metas,
                   replace=replace)
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
        if self.warm is not None:
            self.warm.close()
        self._io.shutdown(wait=False, cancel_futures=True)
        self._jobs.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            self._base = None
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
    - `warm_pack`: key of a warm pack (`meta.kind: "warm"`, see warm.py): the scan prefixes of the most-probed slabs in
      one object. `open` starts ONE GET of it on a background thread and does not wait for it; once it is in, its
      prefixes (those of slabs the open revision reads) are served from memory in front of the cache, so a query
      probing those cells skips their prefix reads (phase-2 reads of those slabs still go to the store). Read from
      `store` (with `prefix`), or from the inner store of a `PackedBlobStore`. `index.warm_ready` (a Future) resolves
      once it is seeded; `index.warm_stats` says how it went.
    """
    return ThreadIndex.open(store, prefix=prefix, gen=gen, rev=rev, cache=cache, refresh_interval=refresh_interval,
                            delta_cache_bytes=delta_cache_bytes, lex_transient_bytes=lex_transient_bytes,
                            io_workers=io_workers, job_workers=job_workers, warm_pack=warm_pack)
