/**
 * thread-sdk: query and update a thread blob-native search index (dense vectors, BM25 text, or both fused) from a
 * Cloudflare Worker.
 *
 *   import wasmModule from "thread-sdk/wasm/thread_bg.wasm";
 *   import { Thread } from "thread-sdk";
 *   const thread = await Thread.init({ wasmModule });                 // once per isolate
 *   const index = await thread.open({ bucket: env.BUCKET, waitUntil: ctx.waitUntil.bind(ctx) });
 *   const { hits } = await index.search({ text: "deploy failed", vector: embedding, topK: 10, docs: true });
 *   const { hits } = await index.query(vector, { topK: 10 });        // dense only
 */
import init, * as W from "./wasm/thread.js";
import { CachedBlobStore, inWorkers, MemoryCache, newTrace, nowMs, phaseStat, PrefixedBlobStore, R2BlobStore, roundMs, type BlobCache, type BlobStore, type BlobTrace, type CacheLike, type MissPolicy, type PhaseStat, type R2BucketLike } from "./store.js";
import { executeQuery, safeFree, type Base, type BaseStat, type ColdPolicy, type QueryOptions, type QueryResult, type RevState } from "./query.js";
import { compactSegments, configureAttrs, configureLexical, createLexicalGeneration, deleteIds, upsert, type CompactSegmentsResult, type ConfigureAttrsResult, type ConfigureLexicalResult, type CreateLexicalOptions, type DeleteResult, type UpsertInput, type UpsertResult, type WriterContext } from "./write.js";
import { compileFilter, packAttrs, unpackAttrs, validateSchema, type AttrFieldSpec, type AttrFilterExpr, type AttrSchema, type AttrValues } from "./attrs.js";
import { chooseRouting, groupedTableOf, hasDense, keys, manifestFiles, parseManifest, resolveLexConfig, slabRevOf, ThreadError, type DeltaRef, type LexConfig, type Manifest, type QueryDefaults, type ResolvedLexConfig, type RoutingOption, type SegmentRef } from "./wire.js";
import { asF32, baseKeysOf, getRequired, loadRevState, resolveTarget, snapshotKeys, type RevTimeline } from "./revision.js";
import { LexIndex, LexSegment, type DocRecord, type LexIo, type LexSearchOptions, type LexSearchResult } from "./lex.js";
import { DEFAULT_RRF_K, rrf } from "./fusion.js";
import { ColdRouting } from "./cold.js";
import { isPackHeader, PackedBlobStore, PrefetchStore } from "./pack.js";
import { fetchWarmPack, revisionSlabKeys, seedWarmPack, WarmPrefixStore, type PrefixSeed, type WarmPackStats } from "./warm.js";

export * from "./wire.js";
export * from "./attrs.js";
export * from "./store.js";
export * from "./s3.js";
export * from "./lex.js";
export * from "./fusion.js";
export * from "./routing.js";
export type { ColdStat } from "./cold.js";
export * from "./pack.js";
export * from "./warm.js";
export type { Base, BaseStat, ColdPolicy, Hit, ParamSource, QueryOptions, QueryParams, QueryResult, QueryStats, QueryTimeline, RevState } from "./query.js";
export type { RevTimeline } from "./revision.js";
export { CROSS_SLAB_GAP, FILTER_WIDEN, Limiter, MULTI_COALESCE, QUERY_INFLIGHT, coalesceRuns, executeQuery, planCrossSlab, slabResident } from "./query.js";
export type { CompactSegmentsResult, ConfigureAttrsResult, ConfigureLexicalResult, CreateLexicalOptions, DeleteResult, UpsertInput, UpsertResult } from "./write.js";
export { createLexicalGeneration } from "./write.js";

/** A unified search: text, vector or both. */
export interface SearchRequest {
	/** The query text (BM25 over the lexical segments, tokenized with the index's tokenizer). */
	text?: string;
	/** The query embedding (dense index). */
	vector?: Float32Array | number[];
	/** Results to return (default 10). */
	topK?: number;
	/** Depth of each list before fusion (default `manifest.lex.defaults.recallK`, else max(topK, 50)). */
	recallK?: number;
	/** RRF constant (default `manifest.lex.defaults.rrfK`, else 60). */
	rrfK?: number;
	/** RRF list weights (default from the manifest, else 1 / 1). */
	weights?: { dense?: number; lexical?: number };
	/** Dense query options (nprobe, rerank, ...). */
	dense?: QueryOptions;
	/** Resolve each hit through the `docs` store: `true` for the metadata, `"text"` for the stored text too. */
	docs?: boolean | "text";
	/**
	 * Attribute filter on both lists: field conditions under the generation's schema (`{tenant: 5, lang: {$in:
	 * ["en", "fr"]}}`; `{attrs: {$any: 1}}` on the raw word without one). The dense scan applies it before top-k
	 * insertion (widening its probes up to `dense.maxNprobe`); the BM25 list is taken deeper and resolved through
	 * the `docs` store (see `LexSearchOptions`).
	 */
	filter?: AttrFilterExpr;
}

export interface SearchHit {
	id: number;
	/** The fused RRF score (hybrid), the cosine (dense only) or the BM25 score (lexical only). */
	score: number;
	/** Position in each list this hit appeared in (1-based) and its list score. */
	dense?: { rank: number; score: number };
	lexical?: { rank: number; score: number };
	/** With `docs`: the document's metadata (and text); `null` when the id has no docs entry. */
	doc?: DocRecord | null;
}

export interface SearchResult {
	hits: SearchHit[];
	mode: "dense" | "lexical" | "hybrid";
	params: { topK: number; recallK: number; rrfK: number; weights: { dense: number; lexical: number } };
	/** The dense query's result (when a vector was given). */
	dense?: QueryResult;
	/** The lexical query's result (when a text was given). */
	lexical?: LexSearchResult;
	stats: { docsIo?: LexIo; timing: { total: number; dense: number; lexical: number; docs: number } };
}
/** The raw wasm-bindgen bindings (advanced use: build jobs, slab tools). */
export type WasmBindings = typeof W;

/**
 * The cold-start timeline of `thread.open` (`index.openStats`): each step's wall time (ms) and store work (origin
 * reads, cache hits/misses, bytes). Python's `index.open_stats` has the same fields, without `wasmInit`.
 */
export interface OpenStats {
	/** Wall time of `open` (ms): current + manifest + slabs + base, plus parsing. */
	total: number;
	/** How long `Thread.init` took (wasm instantiation + panic hook; once per isolate, not part of `total`). */
	wasmInit: number;
	/** `idx/current.json` (or listing the generation's manifests); zero work for a pinned (gen, rev). */
	current: PhaseStat;
	/** The manifest revision. */
	manifest: PhaseStat;
	/** The slabs table and the tombstone sets (read concurrently). */
	slabs: PhaseStat;
	/** The routing table and PQ codebook (`coarse.f32`, `fine.f32`, `pq.bin`) read and loaded into wasm; null without a dense index. */
	base: BaseStat | null;
	/**
	 * How the open's objects were fetched up front (`basePack`): from the revision's base pack (`"base-pack"`), from
	 * a `PackedBlobStore`'s base range (`"index-pack"`), started all at once after the manifest (`"files"`), or not
	 * at all (`"none"`, `basePack: false`); the wall time until then and the work of the pack read (objects served
	 * from a pack count here only; reads started early count in the step that uses them), and the pack key tried.
	 */
	prefetch: PhaseStat & { source: "base-pack" | "index-pack" | "files" | "none"; key: string | null };
	/**
	 * When the store is a `PackedBlobStore`: its own open (head, tail and directory reads, `PackedBlobStore.openStats`).
	 * It happened before `open` (when the pack store was created), so it is not part of `total`.
	 */
	pack?: PhaseStat;
}

export interface ThreadInitOptions {
	/** The compiled `thread_bg.wasm` (`import wasmModule from "thread-sdk/wasm"` with the Cloudflare Vite plugin or Wrangler). */
	wasmModule: WebAssembly.Module;
}

export interface OpenOptions {
	/** R2 binding (sugar for `store: new R2BlobStore(bucket)`). */
	bucket?: R2BucketLike;
	/** Any BlobStore (`R2BlobStore`, `S3BlobStore`, `MemoryBlobStore`, your own). One of `bucket`/`store` is required. */
	store?: BlobStore;
	/** Key prefix in front of `idx/` when the bucket holds several indexes (e.g. `"tenants/a/"`). */
	prefix?: string;
	/** Read this generation (default: the one `idx/current.json` names). */
	gen?: string;
	/** Read this manifest revision (default: current.json's rev, or the generation's head). With `gen`, pins the index. */
	rev?: number;
	/**
	 * A cache in front of the store: the Workers Cache API (`caches.default` by default when available), or an
	 * in-process `BlobCache` (`MemoryCache`, `DiskCache` from `thread-sdk/node`). Outside Workers a remote store
	 * (S3) gets a 256 MB `MemoryCache` by default. `false` disables caching.
	 */
	cache?: CacheLike | BlobCache | false;
	/** Synthetic origin of the cache keys (default `https://vib-sdk.cache/`). */
	cacheOrigin?: string;
	/** What a slab-prefix cache miss reads (see `MissPolicy`): default `"whole"` for the Cache API, `"range+warm"` for an in-process cache. */
	missPolicy?: MissPolicy;
	/** `"range+warm"`: background whole-object fetches at once (default 2). */
	warmConcurrency?: number;
	/** `ctx.waitUntil`, so background cache fills outlive the response. */
	waitUntil?: (p: Promise<unknown>) => void;
	/**
	 * Opens connections to a remote store while `open` loads the index, so the first query does not pay the TLS
	 * handshakes: `true` = as many as the store's `inflight` (32 against S3 from Node), a number = that many, `false`
	 * = none. Default on for remote stores that support it (`S3BlobStore`) outside Workers.
	 */
	warmup?: boolean | number;
	/** Cold-path policy applied to every query (see `ColdPolicy`); off by default. */
	coldPolicy?: ColdPolicy;
	/**
	 * How often a query checks `current.json` for a new revision (ms; default 30 000; 0 = only `refresh()`). The check
	 * runs in the background: the query that triggers it reads the current revision, and the switch happens once the
	 * new revision's manifest, slabs table, tombstones, new delta segments and (if they changed) base files are loaded.
	 */
	refreshInterval?: number;
	/** Bytes of slab prefixes / delta and lexical segments a revision switch prefetches into the cache (default 64 MB; 0 = none). */
	prefetchBytes?: number;
	/** Delta segments kept in memory across queries (bytes; default 40 MB). */
	deltaCacheBytes?: number;
	/** Transient lexical bytes (buckets, posting lists, texts) each segment reader may hold (default 8 MB). */
	lexTransientBytes?: number;
	/**
	 * Routing table to load (default `"f32"`, the 25 MB `fine.f32` at 4096 x 1536): `"f16"` (half the bytes) or
	 * `"i8"` (a quarter), decoded to f32 on load, or `"auto"` for the smallest the revision has. A revision without
	 * the table falls back to f32. The table only picks which cells to probe; slabs are scored with their own f32
	 * means. See docs/query-path.md for the measured probe overlap and recall.
	 */
	routing?: RoutingOption;
	/**
	 * `"tree"`: a cold open does not wait for the routing table. It loads the centroid tree (0.8 MB at 4096 cells) and
	 * the PQ codebook, and the first queries rank the cells of the tree's best `coldBeam` groups, range-reading just
	 * those groups' means from the tree-grouped table; the full table (`routing`) loads in the background and takes
	 * over once it is in. Needs a revision with `manifest.routing.grouped` for its tree, else the open waits for the
	 * full table as usual. Default `"off"`.
	 */
	coldRouting?: "tree" | "off";
	/** Groups ranked per cold-routed query (default: `manifest.defaults.beam`, else 16). See docs/query-path.md. */
	coldBeam?: number;
	/**
	 * Base packs (default true): a cold open first GETs the revision's base pack
	 * (`idx/{gen}/base.{rev}.threadpack`, `base.cold.{rev}` with `coldRouting: "tree"`), every object the open reads in
	 * one object, in parallel with the manifest; from a `PackedBlobStore` it reads the pack's recorded base range. Without
	 * a pack, the open starts every read as soon as the manifest is in. `false`: the reads one after another, as before.
	 */
	basePack?: boolean;
	/**
	 * Key of a warm pack (`meta.kind: "warm"`, see warm.ts): the scan prefixes of the most-probed slabs in one object.
	 * `open` starts ONE GET of it right away and does not wait for it; once it is in, its prefixes (those of slabs the
	 * open revision reads) are served from memory, so a query probing those cells skips their prefix reads. They go
	 * into the in-process cache when it holds prefixes (`MemoryCache`: a prefix hit that asked for `fill` then warms
	 * the whole slab in the background under the cache's miss policy, and counts as resident for `coldPolicy`), else
	 * into a prefix layer in front of the store (`index.warmLayer`). Phase-2 reads of those slabs still go to the
	 * store. The key is read from `store`/`bucket` (with `prefix`, around an in-process cache), or from the inner store
	 * of a `PackedBlobStore`. `index.warmReady` resolves when it is seeded; `index.warmStats` says how it went.
	 * Measured: no gain on the dbpedia workload (docs/cold-start.md, "Warm packs"); opt-in.
	 */
	warmPack?: string;
}

/** One wasm instance per isolate. `Thread.init` once, then `open` indexes. */
export class Thread {
	/** Bumped on every reset; wasm objects created under an older epoch are invalid and must not be freed. */
	epoch = 0;
	resets = 0;
	/** How long `Thread.init` took (ms): wasm instantiation and the panic hook. */
	initMs = 0;
	private raw: W.InitOutput | null = null;
	private readonly resetListeners = new Set<() => void>();
	private constructor(readonly module: WebAssembly.Module) {}

	static async init(opts: ThreadInitOptions): Promise<Thread> {
		const t0 = nowMs();
		const v = new Thread(opts.wasmModule);
		// wasm-bindgen keeps one instance per module scope: a second init() with another module is a no-op.
		v.raw = await init({ module_or_path: opts.wasmModule });
		W.init_panic_hook();
		v.initMs = roundMs(nowMs() - t0);
		return v;
	}

	/** The raw bindings (every wasm export). */
	get wasm(): WasmBindings {
		return W;
	}

	/** Current size of the wasm linear memory (never shrinks). */
	memoryBytes(): number {
		return this.raw?.memory.buffer.byteLength ?? 0;
	}

	/** Called after a reset; indexes drop their wasm objects (Searcher, PqCodebook) without freeing them. */
	onReset(fn: () => void): () => void {
		this.resetListeners.add(fn);
		return () => this.resetListeners.delete(fn);
	}

	/**
	 * After a trap (`WebAssembly.RuntimeError`) the instance is poisoned: re-instantiate it, reinstall the panic
	 * hook and invalidate every wasm object created before.
	 */
	reset(reason: unknown): void {
		this.resets++;
		this.epoch++;
		console.error(`thread-sdk: wasm trap, resetting instance (reset #${this.resets}): ${reason instanceof Error ? `${reason.message}\n${reason.stack ?? ""}` : String(reason)}`);
		W.__wbg_reset_state();
		W.init_panic_hook();
		for (const fn of this.resetListeners) fn();
	}

	/** Runs wasm-using work; on a trap resets the instance and retries once (also when another request reset it mid-flight). */
	async withRecovery<T>(label: string, fn: (attempt: number) => Promise<T>): Promise<T> {
		const epoch = this.epoch;
		try {
			return await fn(1);
		} catch (e) {
			const trap = e instanceof WebAssembly.RuntimeError;
			if (!trap && epoch === this.epoch) throw e;
			if (trap && epoch === this.epoch) this.reset(e);
			else console.error(`thread-sdk: ${label}: retrying after a concurrent wasm reset: ${e instanceof Error ? e.message : String(e)}`);
			return await fn(2);
		}
	}

	open(opts: OpenOptions): Promise<ThreadIndex> {
		return ThreadIndex.open(this, opts);
	}
}

/** Bytes a revision switch prefetches by default (`OpenOptions.prefetchBytes`). */
export const DEFAULT_PREFETCH_BYTES = 64 * 1024 * 1024;

interface BaseSlot extends Base {
	users: number;
	dropped: boolean;
}

/** An opened index: one (gen, rev) snapshot plus the generation's routing table in wasm memory. */
export class ThreadIndex {
	readonly store: BlobStore;
	/** Where `open` spent its time and reads (see `OpenStats`). */
	openStats!: OpenStats;
	private revState: RevState;
	private base: BaseSlot | null = null;
	private basePromise: Promise<BaseSlot> | null = null;
	/** The prefetching view the open reads through (null after the open). */
	private openStore: BlobStore | null = null;
	/** Cold routing (`coldRouting: "tree"`) while the full routing table loads. */
	private cold: BaseSlot | null = null;
	private coldPromise: Promise<BaseSlot> | null = null;
	private readonly pinned: boolean;
	private readonly refreshInterval: number;
	private lastCheck = 0;
	private refreshing: Promise<boolean> | null = null;
	private refreshStarted = 0;
	private closed = false;
	private readonly deltaCache = new Map<string, Promise<Uint8Array>>();
	private readonly deltaBudget: number;
	private readonly unsubscribe: () => void;
	/** Lexical segment readers by key (wasm objects of `epoch`), shared across revisions. */
	private segCache = new Map<string, { seg: LexSegment; epoch: number }>();
	private segLoading = new Map<string, Promise<LexSegment>>();
	private lex: { rev: number; gen: string; index: LexIndex; epoch: number } | null = null;
	private lexLoading: Promise<LexIndex> | null = null;
	/**
	 * The warm-pack prefix layer in front of the store (`OpenOptions.warmPack` without an in-process cache that holds
	 * the prefixes itself), else null.
	 */
	readonly warmLayer: WarmPrefixStore | null = null;
	/** Where the warm pack's prefixes went (the cache store or `warmLayer`). */
	private warmTarget: (PrefixSeed & { retain(keep: (key: string) => boolean): void }) | null = null;
	/** How loading the warm pack went (null without one); updated in place as it loads. */
	warmStats: WarmPackStats | null = null;
	/** Resolves once the warm pack is seeded (or failed: see `warmStats.error`); null without one. Never rejects. */
	warmReady: Promise<WarmPackStats | null> = Promise.resolve(null);

	private constructor(
		readonly thread: Thread,
		store: BlobStore,
		rev: RevState,
		private readonly opts: OpenOptions,
	) {
		this.store = store;
		this.revState = rev;
		this.pinned = opts.gen !== undefined && opts.rev !== undefined;
		this.refreshInterval = opts.refreshInterval ?? 30_000;
		this.deltaBudget = opts.deltaCacheBytes ?? 40 * 1024 * 1024;
		this.lastCheck = Date.now();
		this.unsubscribe = thread.onReset(() => {
			// The Searcher/PqCodebook (and the lexical readers) belong to the dead instance: drop them without free().
			if (this.base) this.base.dropped = true;
			this.base = null;
			this.basePromise = null;
			if (this.cold) this.cold.dropped = true;
			this.cold = null;
			this.coldPromise = null;
			this.segCache.clear();
			this.segLoading.clear();
			this.lex = null;
			this.lexLoading = null;
		});
	}

	static async open(thread: Thread, opts: OpenOptions): Promise<ThreadIndex> {
		let store: BlobStore | undefined = opts.store ?? (opts.bucket ? new R2BlobStore(opts.bucket) : undefined);
		if (!store) throw new ThreadError("open: pass a `bucket` (R2 binding) or a `store`", 400);
		if (opts.prefix) store = new PrefixedBlobStore(store, opts.prefix);
		let cache: CacheLike | BlobCache | null = opts.cache === false ? null : (opts.cache ?? ((globalThis as { caches?: { default?: CacheLike } }).caches?.default ?? null));
		// Outside Workers there is no Cache API: a remote store gets an in-process cache, like the Python SDK.
		if (!cache && opts.cache === undefined && store.remote && !inWorkers()) cache = new MemoryCache();
		if (cache) store = new CachedBlobStore(store, cache, { origin: opts.cacheOrigin, waitUntil: opts.waitUntil, missPolicy: opts.missPolicy, warmConcurrency: opts.warmConcurrency });
		const t0 = nowMs();
		// The warm pack's one GET starts first and runs alongside the whole open; nothing waits for it. Its prefixes
		// go into the in-process cache when it can hold prefixes (`MemoryCache`), else into a prefix layer in front of
		// the store.
		let warm: { target: PrefixSeed & { retain(keep: (key: string) => boolean): void }; layer: WarmPrefixStore | null; got: Promise<{ bytes: Uint8Array; stat: PhaseStat }> } | null = null;
		if (opts.warmPack) {
			// Read around an in-process cache (the pack is not an index object), through the Cache API (shared, worth it).
			const source = opts.store instanceof PackedBlobStore ? opts.store.inner : store instanceof CachedBlobStore && store.blobCache ? store.inner : store;
			const got = fetchWarmPack(source, opts.warmPack, t0);
			got.catch(() => {});
			if (store instanceof CachedBlobStore && store.seedsPrefixes) warm = { target: store, layer: null, got };
			else {
				const layer = new WarmPrefixStore(store);
				warm = { target: layer, layer, got };
				store = layer;
			}
		}
		// Connection warm-up overlaps the open's own sequential reads (current.json, manifest, base files).
		const warmN = opts.warmup ?? (store.remote && !inWorkers());
		const n = typeof warmN === "number" ? warmN : warmN ? (store.inflight ?? 12) : 0;
		if (n > 0 && store.warmup) {
			const p = store.warmup(n).catch(() => {});
			opts.waitUntil?.(p);
		}
		const tl: RevTimeline = {};
		// The open's reads go through a prefetching view: a pack's base range, the base pack, or the files started
		// together. A PackedBlobStore's base range (it starts with idx/current.json) is read before anything else.
		// A manifest inlined in current.json saves its GET (and the base files start right after current.json).
		const pre = new PrefetchStore(store);
		const fromPack = await prefetchIndexPack(pre, opts);
		const target = await resolveTarget(fromPack ? pre : store, opts.gen, opts.rev, tl);
		const prefetch = fromPack && (target.manifest || pre.has(keys.manifest(target.gen, target.rev))) ? fromPack : await prefetchOpen(store, pre, target.gen, target.rev, opts, target.manifest);
		const rev = await loadRevState(pre, target.gen, target.rev, tl, target.manifest);
		const ix = new ThreadIndex(thread, store, rev, opts);
		ix.openStore = pre;
		if (warm) ix.startWarm(warm.target, warm.layer, warm.got, opts.warmPack!, t0);
		const loaded: { stat?: BaseStat } = {};
		if (hasDense(rev.manifest)) {
			if (ix.coldWanted(rev)) {
				// Cold routing: wait for the tree and the codebook only; the full table loads in the background.
				await ix.acquireCold(loaded).then((s) => ix.releaseBase(s));
				ix.loadFullInBackground();
			} else await ix.acquireBase(loaded).then((s) => ix.releaseBase(s));
		}
		ix.openStore = null; // the prefetched bytes are not kept
		const none = phaseStat(0, newTrace());
		ix.openStats = { total: roundMs(nowMs() - t0), wasmInit: thread.initMs, current: tl.current ?? none, prefetch, manifest: tl.manifest ?? none, slabs: tl.slabs ?? none, base: loaded.stat ?? null };
		if (opts.store instanceof PackedBlobStore) ix.openStats.pack = opts.store.openStats;
		return ix;
	}

	get manifest(): Manifest {
		return this.revState.manifest;
	}
	get gen(): string {
		return this.revState.gen;
	}
	get rev(): number {
		return this.revState.rev;
	}
	get dim(): number {
		return this.revState.manifest.dim;
	}
	/** Fine cells (`k1 * k2`). */
	get cells(): number {
		return this.revState.manifest.k1 * this.revState.manifest.k2;
	}
	/** Slab format: 1 int8 only, 2 sq4 + sq8, 3 pq + sq8. */
	get format(): number {
		return this.revState.format;
	}
	/** The query defaults a query without explicit parameters uses. */
	get defaults(): QueryDefaults {
		return this.revState.manifest.defaults ?? { nprobe: 1, rerank: 32, pc: 0 };
	}
	/** The revision state (manifest, slabs table, deltas, tombstones, segments) queries read. */
	get revision(): RevState {
		return this.revState;
	}
	/** Whether the generation has a dense (vector) index. A lexical-only generation has none. */
	get hasDense(): boolean {
		return hasDense(this.revState.manifest);
	}
	/** The lexical / docs configuration (`manifest.lex`, defaults filled in), or null. */
	get lexConfig(): ResolvedLexConfig | null {
		return resolveLexConfig(this.revState.manifest);
	}
	/** The lexical / docs segments of the revision, oldest first. */
	get segments(): SegmentRef[] {
		return this.revState.segments;
	}
	/** The attribute schema (`manifest.attrs`), or null: the raw 32-bit word is then the one field, `attrs`. */
	get attrSchema(): AttrSchema | null {
		return this.revState.manifest.attrs ? validateSchema(this.revState.manifest.attrs) : null;
	}
	/** Packs field values into a document's attrs word under the schema (`{tenant: 5, lang: "en"}`). */
	packAttrs(values: AttrValues): number {
		return packAttrs(this.attrSchema, values);
	}
	/** The field values of a hit's `attrs` word under the schema. */
	unpackAttrs(word: number): Record<string, number | string | boolean> {
		return unpackAttrs(this.attrSchema, word);
	}
	/** Compiles a filter expression against the schema into the words `query({filter})` and `search({filter})` take. */
	compileFilter(filter: AttrFilterExpr | null | undefined): Uint32Array {
		return compileFilter(this.attrSchema, filter);
	}

	/**
	 * Re-reads `idx/current.json` (or the generation's head) and switches to the new revision if it changed.
	 * Returns true when the revision changed. A pinned index (`gen` and `rev` given) never changes.
	 */
	refresh(): Promise<boolean> {
		if (this.closed) throw new ThreadError("index is closed", 400);
		if (this.pinned) return Promise.resolve(false);
		// A refresh started by a request that ended may never settle (Workers): start over after a minute.
		if (this.refreshing && Date.now() - this.refreshStarted < 60_000) return this.refreshing;
		this.refreshStarted = Date.now();
		const p = (async () => {
			const t = await resolveTarget(this.store, this.opts.gen, undefined);
			this.lastCheck = Date.now();
			if (t.gen === this.revState.gen && t.rev === this.revState.rev) return false;
			await this.useRevision(t.gen, t.rev, t.manifest);
			return true;
		})();
		this.refreshing = p;
		p.finally(() => {
			if (this.refreshing === p) this.refreshing = null;
		}).catch(() => {});
		return p;
	}

	/** Seeds the warm-pack target once the warm pack's GET lands (only prefixes of slabs the current revision reads). */
	private startWarm(target: PrefixSeed & { retain(keep: (key: string) => boolean): void }, layer: WarmPrefixStore | null, got: Promise<{ bytes: Uint8Array; stat: PhaseStat }>, key: string, t0: number): void {
		(this as { warmLayer: WarmPrefixStore | null }).warmLayer = layer;
		this.warmTarget = target;
		const st: WarmPackStats = { key, state: "loading", target: layer ? "layer" : "cache", fetch: null, entries: 0, seeded: 0, skipped: 0, seededBytes: 0, readyMs: null };
		this.warmStats = st;
		this.warmReady = got
			.then(({ bytes, stat }) => {
				st.fetch = stat;
				if (this.closed) throw new ThreadError("index closed while the warm pack loaded");
				const current = revisionSlabKeys(this.revState);
				const r = seedWarmPack(bytes, target, (k) => current.has(k));
				Object.assign(st, { entries: r.entries, seeded: r.seeded, skipped: r.skipped, seededBytes: r.seededBytes, state: "ready", readyMs: roundMs(nowMs() - t0) });
				return st;
			})
			.catch((e: unknown) => {
				st.state = "failed";
				st.error = e instanceof Error ? e.message : String(e);
				console.error(`thread-sdk: warm pack ${key} not used: ${st.error}`);
				return st;
			});
		this.opts.waitUntil?.(this.warmReady);
	}

	/** Starts a background refresh when `refreshInterval` has elapsed (queries never wait for it). */
	private maybeRefresh(): void {
		if (this.refreshInterval <= 0 || this.pinned || Date.now() - this.lastCheck <= this.refreshInterval) return;
		this.lastCheck = Date.now();
		const p = this.refresh().catch((e: unknown) => console.error(`thread-sdk: refresh failed: ${e instanceof Error ? e.message : String(e)}`));
		this.opts.waitUntil?.(p);
	}

	/**
	 * Switches to (gen, rev); drops delta and lexical segments the new revision no longer lists. Before switching it
	 * loads what the first query on the new revision would otherwise wait for: the manifest, slabs table and
	 * tombstones, the base files when they changed (routing table, PQ codebook) and the new delta segments; then it
	 * warms the rest of the difference (`warm`) in the background. `manifest`: its manifest when already known (an
	 * inline copy in `idx/current.json`), saving the GET.
	 */
	async useRevision(gen: string, rev: number, manifest?: Manifest): Promise<void> {
		const prev = this.revState;
		const next = await loadRevState(this.store, gen, rev, undefined, manifest);
		const epoch = this.thread.epoch;
		let pre: BaseSlot | null = null;
		const haveBase = (this.base && this.base.key === next.baseKey && !this.base.dropped) || (this.basePromise && !this.base && prev.baseKey === next.baseKey);
		if (hasDense(next.manifest) && !haveBase) {
			const b = await loadBase(this.store, next, epoch, undefined, this.opts.routing);
			pre = { ...b, users: 0, dropped: false };
		}
		const total = next.deltas.reduce((a, d) => a + d.bytes, 0);
		if (total <= this.deltaBudget) await Promise.all(next.deltas.filter((d) => !this.deltaCache.has(d.key)).map((d) => this.getDelta(d, newTrace(), next).catch(() => {})));
		if (this.closed) {
			if (pre && pre.epoch === this.thread.epoch) freeBase(pre);
			return;
		}
		this.revState = next;
		if (this.warmTarget) {
			const current = revisionSlabKeys(next);
			this.warmTarget.retain((k) => current.has(k));
		}
		const listed = new Set(next.deltas.map((d) => d.key));
		for (const k of this.deltaCache.keys()) if (!listed.has(k)) this.deltaCache.delete(k);
		const segs = new Set(next.segments.map((s) => s.key));
		for (const [k, v] of this.segCache) {
			if (segs.has(k)) continue;
			this.segCache.delete(k);
			if (v.epoch === this.thread.epoch) v.seg.free();
		}
		if (this.lex) {
			if (this.lex.epoch === this.thread.epoch) this.lex.index.free();
			this.lex = null;
		}
		this.lexLoading = null;
		if (this.base && this.base.key !== next.baseKey) this.dropBase();
		if (this.cold && this.cold.key !== coldKeyOf(next)) this.dropCold();
		if (pre) {
			if (pre.epoch !== this.thread.epoch) pre.dropped = true; // a reset happened while loading: rebuilt on demand
			else if (this.base && this.base.key === pre.key && !this.base.dropped) freeBase(pre); // loaded concurrently by a query
			else {
				this.dropBase();
				this.base = pre;
				this.basePromise = Promise.resolve(pre);
				this.dropCold(); // the full table takes over from cold routing
			}
		}
		const before = new Set(snapshotKeys(prev));
		const fresh = snapshotKeys(next).filter((k) => !before.has(k));
		if (fresh.length && (this.opts.prefetchBytes ?? DEFAULT_PREFETCH_BYTES) > 0) {
			const p = this.warm(fresh).catch((e: unknown) => console.error(`thread-sdk: prefetch failed: ${e instanceof Error ? e.message : String(e)}`));
			this.opts.waitUntil?.(p);
		}
	}

	/**
	 * Reads `keys` (default: every object of the current revision, `snapshotKeys(index.revision)`) through the
	 * store so its cache holds them: slabs as a `fill` read of their scan prefix (the cache's miss policy decides
	 * whether the whole slab is cached now or in the background), every other object whole. Keys already cached are
	 * skipped; slab prefixes and delta / lexical segments stop at `budget` bytes (default `prefetchBytes`, 64 MB).
	 * A store without a cache in front (a local directory) is not read. A revision switch runs this on the keys the
	 * new revision adds. Returns what was read.
	 */
	async warm(want?: readonly string[], opts: { budget?: number; concurrency?: number } = {}): Promise<{ objects: number; bytes: number; skipped: number }> {
		const rev = this.revState;
		const out = { objects: 0, bytes: 0, skipped: 0 };
		if (!this.store.remote && !(this.store instanceof CachedBlobStore)) return out;
		let budget = opts.budget ?? this.opts.prefetchBytes ?? DEFAULT_PREFETCH_BYTES;
		// A slab costs its scan prefix, or the whole slab when a miss reads it whole (the "whole" policy).
		const whole = this.store instanceof CachedBlobStore && this.store.missPolicy === "whole";
		const slabPrefix = new Map<string, [number, number]>();
		if (rev.slabs) rev.slabs.forEach((row, cell) => row && slabPrefix.set(keys.slab(rev.gen, cell, slabRevOf(row)), [row[0], whole ? row[3] : row[0]]));
		const budgeted = new Set([...rev.deltas.map((d) => d.key), ...rev.segments.map((s) => s.key)]);
		const todo = [...(want ?? snapshotKeys(rev))];
		const trace = newTrace();
		const worker = async () => {
			for (let k = todo.shift(); k !== undefined; k = todo.shift()) {
				if (this.store.contains?.(k)) {
					out.skipped++;
					continue;
				}
				const [scanLen, cost] = slabPrefix.get(k) ?? [undefined, undefined];
				const isBudgeted = scanLen !== undefined || budgeted.has(k);
				if (isBudgeted && budget <= 0) {
					out.skipped++;
					continue;
				}
				const b = scanLen !== undefined ? await this.store.get(k, { offset: 0, length: scanLen }, { trace, fill: true }) : await this.store.get(k, undefined, { trace });
				if (!b) continue;
				out.objects++;
				out.bytes += b.byteLength;
				if (isBudgeted) budget -= cost ?? b.byteLength;
			}
		};
		await Promise.all(Array.from({ length: Math.max(1, opts.concurrency ?? 4) }, worker));
		return out;
	}

	// ---- unified search ----

	/**
	 * Searches by text, by vector, or by both: the dense list (`query`) and the lexical list (BM25 over the
	 * segments) are run concurrently at depth `recallK` and fused with Reciprocal Rank Fusion; a single list is
	 * returned as is. With `docs`, hits are resolved through the `docs` store.
	 */
	async search(req: SearchRequest): Promise<SearchResult> {
		if (this.closed) throw new ThreadError("index is closed", 400);
		const t0 = Date.now();
		const topK = req.topK ?? 10;
		if (!Number.isInteger(topK) || topK < 1) throw new ThreadError("topK must be an integer >= 1", 400);
		const hasText = typeof req.text === "string" && req.text.trim().length > 0;
		const hasVector = req.vector !== undefined;
		if (!hasText && !hasVector) throw new ThreadError("search needs a text, a vector or both", 400);
		this.maybeRefresh();
		const cfg = this.lexConfig;
		if (hasText && !cfg) throw new ThreadError("this generation has no lexical index (index.configureLexical first)", 400);
		if (hasVector && !this.hasDense) throw new ThreadError("this generation has no dense index", 400);
		const recallK = req.recallK ?? cfg?.recallK ?? Math.max(topK, 50);
		const rrfK = req.rrfK ?? cfg?.rrfK ?? DEFAULT_RRF_K;
		const weights = { dense: req.weights?.dense ?? cfg?.weights.dense ?? 1, lexical: req.weights?.lexical ?? cfg?.weights.lexical ?? 1 };
		const both = hasText && hasVector;
		const depth = both ? Math.max(recallK, topK) : topK;
		// The filter is compiled once and applied to both lists.
		const filter = req.filter ? this.compileFilter(req.filter) : undefined;
		let tDense = 0;
		let tLex = 0;
		const [dense, lexical] = await Promise.all([
			hasVector
				? (async () => {
						const t = Date.now();
						const r = await this.query(req.vector!, { ...req.dense, topK: depth, ...(filter ? { filter } : {}) });
						tDense = Date.now() - t;
						return r;
					})()
				: Promise.resolve(undefined),
			hasText
				? (async () => {
						const t = Date.now();
						const r = await this.lexicalSearch(req.text!, depth, filter ? { filter } : {});
						tLex = Date.now() - t;
						return r;
					})()
				: Promise.resolve(undefined),
		]);
		let hits: SearchHit[];
		let mode: SearchResult["mode"];
		if (both) {
			mode = "hybrid";
			const dIds = dense!.hits.map((h) => h.id);
			const lIds = lexical!.hits.map((h) => h.id);
			const fused = rrf(
				[
					[weights.dense, dIds],
					[weights.lexical, lIds],
				],
				rrfK,
				topK,
			);
			const dRank = new Map(dense!.hits.map((h, i) => [h.id, { rank: i + 1, score: h.score }]));
			const lRank = new Map(lexical!.hits.map((h, i) => [h.id, { rank: i + 1, score: h.score }]));
			hits = Array.from(fused.ids, (id, i) => {
				const h: SearchHit = { id, score: fused.scores[i] };
				const d = dRank.get(id);
				const l = lRank.get(id);
				if (d) h.dense = d;
				if (l) h.lexical = l;
				return h;
			});
		} else if (dense) {
			mode = "dense";
			hits = dense.hits.slice(0, topK).map((h, i) => ({ id: h.id, score: h.score, dense: { rank: i + 1, score: h.score } }));
		} else {
			mode = "lexical";
			hits = lexical!.hits.slice(0, topK).map((h, i) => ({ id: h.id, score: h.score, lexical: { rank: i + 1, score: h.score } }));
		}
		let docsIo: LexIo | undefined;
		let tDocs = 0;
		if (req.docs && hits.length) {
			const t = Date.now();
			const r = await this.docs(
				hits.map((h) => h.id),
				{ text: req.docs === "text" },
			);
			docsIo = r.io;
			for (const h of hits) h.doc = r.docs.get(h.id) ?? null;
			tDocs = Date.now() - t;
		}
		return {
			hits,
			mode,
			params: { topK, recallK: depth, rrfK, weights },
			...(dense ? { dense } : {}),
			...(lexical ? { lexical } : {}),
			stats: { ...(docsIo ? { docsIo } : {}), timing: { total: Date.now() - t0, dense: tDense, lexical: tLex, docs: tDocs } },
		};
	}

	/** BM25 top `k` for `text` over the revision's lexical segments (see `LexIndex.search`); `opts.filter` takes compiled words. */
	async lexicalSearch(text: string, k: number, opts: LexSearchOptions = {}): Promise<LexSearchResult> {
		if (this.closed) throw new ThreadError("index is closed", 400);
		return this.thread.withRecovery("lexicalSearch", async () => (await this.lexIndex()).search(text, k, opts));
	}

	/** The query's tokens under the index's tokenizer. */
	async tokenize(text: string): Promise<string[]> {
		return (await this.lexIndex()).tokenize(text);
	}

	/** Resolves ids through the `docs` store (newest live copy of each id); ids without an entry are absent. */
	async docs(ids: readonly number[], opts: { text?: boolean } = {}): Promise<{ docs: Map<number, DocRecord>; io: LexIo }> {
		if (this.closed) throw new ThreadError("index is closed", 400);
		if (!this.lexConfig) return { docs: new Map(), io: { rounds: 0, ranges: 0, bytes: 0 } };
		return this.thread.withRecovery("docs", async () => (await this.lexIndex()).docs(ids, opts));
	}

	/**
	 * @experimental Gives the generation a lexical index / docs store (or changes its configuration for future
	 * segments): a manifest revision with `lex`. Existing documents are not indexed retroactively: upsert them
	 * again with their texts.
	 */
	async configureLexical(config: LexConfig): Promise<ConfigureLexicalResult> {
		const r = await configureLexical(this.writer(), config);
		await this.useRevision(r.gen, r.rev);
		return r;
	}

	/**
	 * @experimental Declares (or extends) the attribute schema: named fields packed into each document's 32-bit
	 * attrs word, in declaration order (`{tenant: {bits: 12}, lang: {values: ["en", "fr"]}, public: "bool"}`), as a
	 * manifest revision. Existing fields keep their layout (an enum may append values); documents written before
	 * keep their words.
	 */
	async configureAttrs(spec: Record<string, AttrFieldSpec> | AttrSchema): Promise<ConfigureAttrsResult> {
		const r = await configureAttrs(this.writer(), spec);
		await this.useRevision(r.gen, r.rev);
		return r;
	}

	/**
	 * @experimental Merges every lexical/docs segment of the head revision into one base segment (tombstoned
	 * documents dropped, replaced ids resolved to their newest copy) and commits a revision listing only it.
	 * Reads every segment whole: run it where memory allows (segments' total size, roughly twice).
	 */
	async compactSegments(): Promise<CompactSegmentsResult> {
		const r = await compactSegments(this.writer());
		await this.useRevision(r.gen, r.rev);
		return r;
	}

	/** The lexical index of the current revision (segments opened on first use and cached by key). */
	private lexIndex(): Promise<LexIndex> {
		const rev = this.revState;
		const cfg = this.lexConfig;
		if (!cfg) throw new ThreadError("this generation has no lexical index (index.configureLexical first)", 400);
		if (this.lex && this.lex.rev === rev.rev && this.lex.gen === rev.gen && this.lex.epoch === this.thread.epoch) return Promise.resolve(this.lex.index);
		if (this.lexLoading) return this.lexLoading;
		const epoch = this.thread.epoch;
		const p = (async () => {
			const io: LexIo = { rounds: 0, ranges: 0, bytes: 0 };
			const segs = await Promise.all(
				rev.segments.map(async (ref) => {
					const cached = this.segCache.get(ref.key);
					if (cached && cached.epoch === epoch) return { ref, seg: cached.seg };
					let loading = this.segLoading.get(ref.key);
					if (!loading) {
						loading = LexSegment.open(W, this.store, ref, epoch, io, undefined, { transientBudget: this.opts.lexTransientBytes }).then((seg) => {
							if (epoch === this.thread.epoch && !this.closed) this.segCache.set(ref.key, { seg, epoch });
							return seg;
						});
						this.segLoading.set(ref.key, loading);
						loading.finally(() => this.segLoading.get(ref.key) === loading && this.segLoading.delete(ref.key)).catch(() => {});
					}
					return { ref, seg: await loading };
				}),
			);
			const index = new LexIndex(W, cfg, segs, rev.tomb);
			if (epoch === this.thread.epoch && !this.closed && this.revState === rev) this.lex = { rev: rev.rev, gen: rev.gen, index, epoch };
			return index;
		})();
		this.lexLoading = p;
		p.finally(() => {
			if (this.lexLoading === p) this.lexLoading = null;
		}).catch(() => {});
		return p;
	}

	/** Searches for the nearest neighbours of `vector` (any norm; it is normalized). */
	async query(vector: Float32Array | number[], opts: QueryOptions = {}): Promise<QueryResult> {
		if (this.closed) throw new ThreadError("index is closed", 400);
		if (!this.hasDense) throw new ThreadError("this generation has no dense index (lexical only): use search({text})", 400);
		const q = vector instanceof Float32Array ? vector : Float32Array.from(vector);
		this.maybeRefresh();
		return this.thread.withRecovery("query", async (attempt) => {
			const loaded: { stat?: BaseStat } = {};
			const base = await this.acquireQueryBase(loaded);
			try {
				const rev = this.revState;
				const r = await executeQuery(
					{
						w: W,
						store: this.store,
						rev,
						base,
						epoch: () => this.thread.epoch,
						wasmResets: this.thread.resets,
						wasmMemBytes: () => this.thread.memoryBytes(),
						getDelta: (ref, trace) => this.getDelta(ref, trace),
						coldPolicy: this.opts.coldPolicy,
					},
					q,
					opts,
					attempt,
				);
				// This query (re)loaded the routing table (after a reset or a switch to another generation).
				if (loaded.stat) r.stats.timeline.base = loaded.stat;
				return r;
			} finally {
				this.releaseBase(base);
			}
		});
	}

	/** Several queries against the same snapshot and routing table, concurrently. */
	queryMany(vectors: (Float32Array | number[])[], opts: QueryOptions = {}): Promise<QueryResult[]> {
		return Promise.all(vectors.map((v) => this.query(v, opts)));
	}

	/**
	 * @experimental Adds or replaces vectors: one delta segment plus (with `replace`, the default) a tombstone set
	 * for the ids, committed as a new manifest revision; the index then reads that revision. Needs a store with
	 * `put`/`head` (R2 binding or S3). See README "Write path".
	 */
	async upsert(input: UpsertInput): Promise<UpsertResult> {
		const r = await upsert(this.writer(), input);
		await this.useRevision(r.gen, r.rev);
		return r;
	}

	/** @experimental Deletes ids: one tombstone set in a new manifest revision; the index then reads that revision. */
	async delete(ids: ArrayLike<number>): Promise<DeleteResult> {
		const r = await deleteIds(this.writer(), ids);
		await this.useRevision(r.gen, r.rev);
		return r;
	}

	/** Frees the wasm objects (routing table, PQ codebook, lexical readers). The index cannot be used afterwards. */
	close(): void {
		if (this.closed) return;
		this.closed = true;
		this.unsubscribe();
		this.dropBase();
		this.dropCold();
		this.deltaCache.clear();
		for (const v of this.segCache.values()) if (v.epoch === this.thread.epoch) v.seg.free();
		this.segCache.clear();
		if (this.lex && this.lex.epoch === this.thread.epoch) this.lex.index.free();
		this.lex = null;
	}

	// ---- internals ----

	private writer(): WriterContext {
		if (this.closed) throw new ThreadError("index is closed", 400);
		return { thread: this.thread, w: W, store: this.store, gen: this.revState.gen, borrowPq: (key) => (this.base && !this.base.dropped && this.base.key === key && this.base.epoch === this.thread.epoch ? this.base.pq : null) };
	}

	private async getDelta(ref: DeltaRef, trace: BlobTrace, rev: RevState = this.revState): Promise<Uint8Array> {
		const hit = this.deltaCache.get(ref.key);
		if (hit) return hit;
		const p = this.store.get(ref.key, undefined, { trace }).then((b) => {
			if (!b) throw new ThreadError(`missing delta segment ${ref.key}`, 404);
			return b;
		});
		const total = rev.deltas.reduce((a, d) => a + d.bytes, 0);
		if (total <= this.deltaBudget) {
			this.deltaCache.set(ref.key, p);
			p.catch(() => this.deltaCache.get(ref.key) === p && this.deltaCache.delete(ref.key));
		}
		return p;
	}

	/** The routing table of the current revision; `loaded.stat` is set when this call started the load. */
	private acquireBase(loaded?: { stat?: BaseStat }): Promise<BaseSlot> {
		if (this.base && this.base.epoch === this.thread.epoch && !this.base.dropped) {
			this.base.users++;
			return Promise.resolve(this.base);
		}
		if (!this.basePromise) {
			const rev = this.revState;
			const p = loadBase(this.openStore ?? this.store, rev, this.thread.epoch, loaded, this.opts.routing).then((b) => {
				const slot: BaseSlot = { ...b, users: 0, dropped: false };
				if (this.basePromise === p && !this.closed) {
					this.base = slot;
					this.dropCold(); // the full table takes over from cold routing
				} else slot.dropped = true; // superseded while loading
				return slot;
			});
			this.basePromise = p;
			p.catch(() => this.basePromise === p && (this.basePromise = null));
		}
		return this.basePromise.then((slot) => {
			slot.users++;
			return slot;
		});
	}

	private releaseBase(slot: BaseSlot): void {
		slot.users--;
		if (slot.dropped && slot.users === 0 && slot.epoch === this.thread.epoch) freeBase(slot);
	}

	private dropBase(): void {
		const b = this.base;
		this.base = null;
		this.basePromise = null;
		if (!b || b.dropped) return;
		b.dropped = true;
		if (b.users === 0 && b.epoch === this.thread.epoch) freeBase(b);
	}

	// ---- cold routing ----

	/** Whether queries of `rev` may route cold (the option is on and the revision has a usable grouped table). */
	private coldWanted(rev: RevState): boolean {
		return this.opts.coldRouting === "tree" && hasDense(rev.manifest) && groupedTableOf(rev.manifest, rev.files) !== null;
	}

	/**
	 * The base a query uses: the full routing table once it is loaded; before that, with cold routing, the cold base
	 * (and the full table keeps loading in the background).
	 */
	private acquireQueryBase(loaded?: { stat?: BaseStat }): Promise<BaseSlot> {
		if (this.base && this.base.epoch === this.thread.epoch && !this.base.dropped) return this.acquireBase(loaded);
		const rev = this.revState;
		if (!this.coldWanted(rev)) return this.acquireBase(loaded);
		this.loadFullInBackground();
		return this.acquireCold(loaded);
	}

	/** Starts loading the full routing table (once) without waiting for it. */
	private loadFullInBackground(): void {
		if (this.basePromise || (this.base && !this.base.dropped && this.base.epoch === this.thread.epoch)) return;
		const p = this.acquireBase()
			.then((s) => this.releaseBase(s))
			.catch((e: unknown) => console.error(`thread-sdk: loading the routing table failed (cold routing continues): ${e instanceof Error ? e.message : String(e)}`));
		this.opts.waitUntil?.(p);
	}

	private acquireCold(loaded?: { stat?: BaseStat }): Promise<BaseSlot> {
		const rev = this.revState;
		const key = coldKeyOf(rev);
		if (this.cold && this.cold.epoch === this.thread.epoch && !this.cold.dropped && this.cold.key === key) {
			this.cold.users++;
			return Promise.resolve(this.cold);
		}
		if (!this.coldPromise) {
			const p = loadColdBase(this.openStore ?? this.store, rev, this.thread.epoch, this.opts.coldBeam, loaded).then((b) => {
				const slot: BaseSlot = { ...b, users: 0, dropped: false };
				if (this.coldPromise === p && !this.closed && !this.base) this.cold = slot;
				else slot.dropped = true; // superseded while loading (or the full table won the race)
				return slot;
			});
			this.coldPromise = p;
			p.catch(() => this.coldPromise === p && (this.coldPromise = null));
		}
		return this.coldPromise.then((slot) => {
			slot.users++;
			return slot;
		});
	}

	private dropCold(): void {
		const b = this.cold;
		this.cold = null;
		this.coldPromise = null;
		if (!b || b.dropped) return;
		b.dropped = true;
		if (b.users === 0 && b.epoch === this.thread.epoch) freeBase(b);
	}

	/** `"full"` once the routing table is loaded, `"cold"` while queries route cold, `"none"` before either. */
	get routingState(): "full" | "cold" | "none" {
		if (this.base && !this.base.dropped) return "full";
		return this.cold && !this.cold.dropped ? "cold" : "none";
	}

	/** Waits until the full routing table is loaded (starting the load if needed). */
	async whenFullRouting(): Promise<void> {
		if (!this.hasDense) return;
		await this.acquireBase().then((s) => this.releaseBase(s));
	}
}

// ---------- loading ----------

/** A `PackedBlobStore`'s recorded base range, read with one range read into `pre` (null when there is none). */
async function prefetchIndexPack(pre: PrefetchStore, opts: OpenOptions): Promise<OpenStats["prefetch"] | null> {
	if (opts.basePack === false || !(opts.store instanceof PackedBlobStore) || opts.prefix) return null;
	const t0 = nowMs();
	const trace = newTrace();
	// The range ends with the smallest routing table (`fine{.r}.{elem}`): an open that loads another one stops before it.
	const routing = opts.routing ?? "f32";
	const rk = opts.store.directory.meta.baseRouting;
	const inRange = typeof rk !== "string" ? null : rk.endsWith(".i8") ? "i8" : rk.endsWith(".f16") ? "f16" : "f32";
	const got = await opts.store.readBase(trace, opts.coldRouting === "tree" || routing === "auto" || routing === inRange);
	if (!got) return null;
	for (const [k, b] of got) pre.set(k, b);
	return { ...phaseStat(nowMs() - t0, trace), source: "index-pack", key: opts.store.key };
}

/**
 * Starts the reads of an open before the loaders ask for them: the revision's base pack (one GET, in parallel with
 * the manifest in case there is none), a `PackedBlobStore`'s base range, or, without a pack, every base object as
 * soon as the manifest is in (slabs table, tombstone sets, coarse, codebook, routing table or, for cold routing, the
 * tree). The loaders then read `pre`, which serves those objects from memory.
 */
async function prefetchOpen(store: BlobStore, pre: PrefetchStore, gen: string, rev: number, opts: OpenOptions, inline?: Manifest): Promise<OpenStats["prefetch"]> {
	const t0 = nowMs();
	const trace = newTrace();
	const done = (source: OpenStats["prefetch"]["source"], key: string | null) => ({ ...phaseStat(nowMs() - t0, trace), source, key });
	if (opts.basePack === false) return done("none", null);
	const cold = opts.coldRouting === "tree";
	const routing = opts.routing ?? "f32";
	const mkey = keys.manifest(gen, rev);
	// A manifest inlined in current.json: nothing waits for the manifest GET (and none is issued).
	const m0 = inline && inline.gen === gen && (inline.rev ?? 0) === rev ? inline : null;
	const manifestOf = async (): Promise<Manifest | null> => {
		if (m0) return m0;
		const mb = await pre.peek(mkey).catch(() => null);
		try {
			return mb ? parseManifest(mb, gen, rev) : null;
		} catch {
			return null; // loadRevState reports it
		}
	};
	// A base pack holds the smallest routing table (publish's default): worth its GET for cold routing and for
	// routing "auto" / "i8"; an f32 / f16 open would download that table for nothing and reads the files instead.
	// (A pack is looked up by its conventional key whatever `files.base` says: it may have been written after the
	// manifest.)
	const packKey = cold ? keys.baseColdPack(gen, rev) : keys.basePack(gen, rev);
	if (!(opts.store instanceof PackedBlobStore) && (cold || routing === "auto" || routing === "i8")) {
		if (!m0) pre.prefetch(mkey); // in case there is no pack
		if (!cold) {
			// Should the pack hold another table than the one asked for (a generation without fine.i8), start that one
			// as soon as the manifest is in rather than after the pack.
			manifestOf()
				.then((m) => {
					if (!m || !hasDense(m)) return;
					const files = manifestFiles(m);
					const want = chooseRouting(m, files, routing).key;
					if (want !== chooseRouting(m, files, "auto").key) pre.prefetch(want);
				})
				.catch(() => {});
		}
		const b = await store.get(packKey, undefined, { trace }).catch(() => null);
		if (b && isPackHeader(b)) {
			try {
				const meta = pre.addPack(b);
				if (meta.gen === gen && meta.rev === rev) return done("base-pack", packKey);
			} catch (e) {
				console.error(`thread-sdk: ignoring the base pack ${packKey}: ${e instanceof Error ? e.message : String(e)}`);
			}
		}
	}
	// No pack (or not worth one): every base object as soon as the manifest is in (each read is accounted where its
	// bytes are used).
	if (!m0) pre.prefetch(mkey);
	const m = await manifestOf();
	if (m) for (const k of baseKeysOf(m, gen, rev, cold ? "cold" : "base", routing)) if (!(m0 && k === mkey)) pre.prefetch(k);
	return done("files", null);
}

function freeBase(b: Base): void {
	if (b.searcher) safeFree(b.searcher);
	if (b.cold) safeFree(b.cold.router);
	if (b.pq) safeFree(b.pq);
}

/** Key of a cold base: the base files plus the grouped table (a revision with another tree or means has another). */
const coldKeyOf = (rev: RevState): string => `${rev.baseKey}|${groupedTableOf(rev.manifest, rev.files)?.key ?? "-"}`;

function checkedCodebook(rev: RevState, p: Uint8Array): W.PqCodebook {
	const { manifest, files } = rev;
	const pq = new W.PqCodebook(p);
	if (pq.dim() !== manifest.dim || (manifest.pq && pq.m() !== manifest.pq.m)) {
		const msg = `${files.pq} is dim ${pq.dim()} m ${pq.m()}, manifest dim ${manifest.dim} pq ${JSON.stringify(manifest.pq)}`;
		safeFree(pq);
		throw new ThreadError(msg);
	}
	return pq;
}

/**
 * Loads a cold base (`coldRouting: "tree"`): the centroid tree and the PQ codebook, plus an empty cold router over the
 * tree-grouped table that queries fill group by group. `beam` defaults to `manifest.defaults.beam`, else 16.
 */
export async function loadColdBase(store: BlobStore, rev: RevState, epoch: number, beam?: number, loaded?: { stat?: BaseStat }): Promise<Base> {
	const { manifest, files } = rev;
	const g = groupedTableOf(manifest, files);
	if (!g) throw new ThreadError(`${rev.gen} rev ${rev.rev} has no tree-grouped routing table for its tree`);
	const b = beam ?? (manifest.defaults as { beam?: number } | undefined)?.beam ?? 16;
	if (!Number.isInteger(b) || b < 1) throw new ThreadError("coldBeam must be an integer >= 1", 400);
	const t0 = nowMs();
	const trace = newTrace();
	// (i8 rows are residuals from a centre in the grouped table's head: the first group read fetches it too.)
	const [t, p] = await Promise.all([getRequired(store, g.tree, trace), files.pq ? getRequired(store, files.pq, trace) : Promise.resolve(null)]);
	const t1 = nowMs();
	if (rev.format === 3 && !p) throw new ThreadError(`format 3 generation ${rev.gen} has no PQ codebook`);
	const pq = p ? checkedCodebook(rev, p) : null;
	let router: W.ColdRouter | null = null;
	try {
		router = new W.ColdRouter(t, g.elem, new Uint8Array(0));
		const k = manifest.k1 * manifest.k2;
		if (router.k() !== k || router.dim() !== manifest.dim || router.groups() !== g.groups || router.row_bytes() !== g.rowBytes) throw new ThreadError(`${g.tree} (k ${router.k()}, dim ${router.dim()}, ${router.groups()} groups) does not match ${g.key} / the manifest (k ${k}, dim ${manifest.dim}, ${g.groups} groups)`);
		const cold = new ColdRouting(W, store, router, g, b);
		const t2 = nowMs();
		if (loaded) loaded.stat = { ...phaseStat(t2 - t0, trace), fetch: roundMs(t1 - t0), build: roundMs(t2 - t1), files: { coarse: 0, fine: t.byteLength, pq: p?.byteLength ?? 0 } };
		return { key: coldKeyOf(rev), searcher: null, cold, pq, epoch, routing: g.elem };
	} catch (e) {
		if (router) safeFree(router);
		if (pq) safeFree(pq);
		throw e;
	}
}

export { asF32, baseKeys, baseKeysOf, headRevision, loadRevState, resolveTarget, snapshotKeys } from "./revision.js";

/**
 * Loads the routing table (and PQ codebook) of a revision's generation into wasm memory. `routing` picks the table:
 * `fine.f32` (default), the compact `fine.f16` / `fine.i8` copies (decoded to f32 on load) or `"auto"` (the smallest
 * the revision has); a revision without the requested table falls back to `fine.f32`. With `loaded`, records the
 * load's wall time, reads and bytes in `loaded.stat` (`fetch`: reading the files, `build`: the wasm objects).
 */
export async function loadBase(store: BlobStore, rev: RevState, epoch: number, loaded?: { stat?: BaseStat }, routing: RoutingOption = "f32"): Promise<Base> {
	const { manifest, files } = rev;
	const table = chooseRouting(manifest, files, routing);
	const t0 = nowMs();
	const trace = newTrace();
	const [c, f, p] = await Promise.all([getRequired(store, files.coarse, trace), getRequired(store, table.key, trace), files.pq ? getRequired(store, files.pq, trace) : Promise.resolve(null)]);
	const t1 = nowMs();
	if (rev.format === 3 && !p) throw new ThreadError(`format 3 generation ${rev.gen} has no PQ codebook`);
	const pq = p ? checkedCodebook(rev, p) : null;
	try {
		const searcher = table.elem === "f32" ? new W.Searcher(manifest.dim, manifest.k1, manifest.k2, asF32(c, files.coarse), asF32(f, files.fine)) : W.Searcher.from_table(manifest.dim, manifest.k1, manifest.k2, asF32(c, files.coarse), f, table.elem);
		const t2 = nowMs();
		if (loaded) loaded.stat = { ...phaseStat(t2 - t0, trace), fetch: roundMs(t1 - t0), build: roundMs(t2 - t1), files: { coarse: c.byteLength, fine: f.byteLength, pq: p?.byteLength ?? 0 } };
		return { key: rev.baseKey, searcher, pq, epoch, routing: table.elem };
	} catch (e) {
		if (pq) safeFree(pq);
		throw e;
	}
}
