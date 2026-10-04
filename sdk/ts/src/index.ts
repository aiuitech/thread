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
import { CachedBlobStore, PrefixedBlobStore, R2BlobStore, type BlobStore, type BlobTrace, type CacheLike, type R2BucketLike } from "./store.js";
import { executeQuery, safeFree, type Base, type QueryOptions, type QueryResult, type RevState } from "./query.js";
import { compactSegments, configureLexical, createLexicalGeneration, deleteIds, upsert, type CompactSegmentsResult, type ConfigureLexicalResult, type CreateLexicalOptions, type DeleteResult, type UpsertInput, type UpsertResult, type WriterContext } from "./write.js";
import { hasDense, resolveLexConfig, ThreadError, type DeltaRef, type LexConfig, type Manifest, type QueryDefaults, type ResolvedLexConfig, type SegmentRef } from "./wire.js";
import { asF32, getRequired, loadRevState, resolveTarget } from "./revision.js";
import { LexIndex, LexSegment, type DocRecord, type LexIo, type LexSearchResult } from "./lex.js";
import { DEFAULT_RRF_K, rrf } from "./fusion.js";

export * from "./wire.js";
export * from "./store.js";
export * from "./s3.js";
export * from "./lex.js";
export * from "./fusion.js";
export * from "./pack.js";
export type { Base, Hit, ParamSource, QueryOptions, QueryParams, QueryResult, QueryStats, RevState } from "./query.js";
export { Limiter, QUERY_INFLIGHT, executeQuery } from "./query.js";
export type { CompactSegmentsResult, ConfigureLexicalResult, CreateLexicalOptions, DeleteResult, UpsertInput, UpsertResult } from "./write.js";
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
	/** Cache API in front of the store (default `caches.default` when available); `false` disables it. */
	cache?: CacheLike | false;
	/** Synthetic origin of the cache keys (default `https://vib-sdk.cache/`). */
	cacheOrigin?: string;
	/** `ctx.waitUntil`, so background cache fills outlive the response. */
	waitUntil?: (p: Promise<unknown>) => void;
	/** How often a query re-reads `current.json` to pick up new revisions (ms; default 30 000; 0 = only `refresh()`). */
	refreshInterval?: number;
	/** Delta segments kept in memory across queries (bytes; default 40 MB). */
	deltaCacheBytes?: number;
	/** Transient lexical bytes (buckets, posting lists, texts) each segment reader may hold (default 8 MB). */
	lexTransientBytes?: number;
}

/** One wasm instance per isolate. `Thread.init` once, then `open` indexes. */
export class Thread {
	/** Bumped on every reset; wasm objects created under an older epoch are invalid and must not be freed. */
	epoch = 0;
	resets = 0;
	private raw: W.InitOutput | null = null;
	private readonly resetListeners = new Set<() => void>();
	private constructor(readonly module: WebAssembly.Module) {}

	static async init(opts: ThreadInitOptions): Promise<Thread> {
		const v = new Thread(opts.wasmModule);
		// wasm-bindgen keeps one instance per module scope: a second init() with another module is a no-op.
		v.raw = await init({ module_or_path: opts.wasmModule });
		W.init_panic_hook();
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

interface BaseSlot extends Base {
	users: number;
	dropped: boolean;
}

/** An opened index: one (gen, rev) snapshot plus the generation's routing table in wasm memory. */
export class ThreadIndex {
	readonly store: BlobStore;
	private revState: RevState;
	private base: BaseSlot | null = null;
	private basePromise: Promise<BaseSlot> | null = null;
	private readonly pinned: boolean;
	private readonly refreshInterval: number;
	private lastCheck = 0;
	private refreshing: Promise<boolean> | null = null;
	private closed = false;
	private readonly deltaCache = new Map<string, Promise<Uint8Array>>();
	private readonly deltaBudget: number;
	private readonly unsubscribe: () => void;
	/** Lexical segment readers by key (wasm objects of `epoch`), shared across revisions. */
	private segCache = new Map<string, { seg: LexSegment; epoch: number }>();
	private segLoading = new Map<string, Promise<LexSegment>>();
	private lex: { rev: number; gen: string; index: LexIndex; epoch: number } | null = null;
	private lexLoading: Promise<LexIndex> | null = null;

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
		const cache = opts.cache === false ? null : (opts.cache ?? ((globalThis as { caches?: { default?: CacheLike } }).caches?.default ?? null));
		if (cache) store = new CachedBlobStore(store, cache, { origin: opts.cacheOrigin, waitUntil: opts.waitUntil });
		const target = await resolveTarget(store, opts.gen, opts.rev);
		const rev = await loadRevState(store, target.gen, target.rev);
		const ix = new ThreadIndex(thread, store, rev, opts);
		if (hasDense(rev.manifest)) await ix.acquireBase().then((s) => ix.releaseBase(s));
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

	/**
	 * Re-reads `idx/current.json` (or the generation's head) and switches to the new revision if it changed.
	 * Returns true when the revision changed. A pinned index (`gen` and `rev` given) never changes.
	 */
	refresh(): Promise<boolean> {
		if (this.closed) throw new ThreadError("index is closed", 400);
		if (this.pinned) return Promise.resolve(false);
		if (this.refreshing) return this.refreshing;
		const p = (async () => {
			const t = await resolveTarget(this.store, this.opts.gen, undefined);
			this.lastCheck = Date.now();
			if (t.gen === this.revState.gen && t.rev === this.revState.rev) return false;
			await this.useRevision(t.gen, t.rev);
			return true;
		})();
		this.refreshing = p;
		p.finally(() => {
			if (this.refreshing === p) this.refreshing = null;
		}).catch(() => {});
		return p;
	}

	/** Switches to (gen, rev); drops delta and lexical segments the new revision no longer lists. */
	async useRevision(gen: string, rev: number): Promise<void> {
		const next = await loadRevState(this.store, gen, rev);
		this.revState = next;
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
		if (this.refreshInterval > 0 && !this.pinned && Date.now() - this.lastCheck > this.refreshInterval) await this.refresh().catch((e: unknown) => console.error(`thread-sdk: refresh failed: ${e instanceof Error ? e.message : String(e)}`));
		const cfg = this.lexConfig;
		if (hasText && !cfg) throw new ThreadError("this generation has no lexical index (index.configureLexical first)", 400);
		if (hasVector && !this.hasDense) throw new ThreadError("this generation has no dense index", 400);
		const recallK = req.recallK ?? cfg?.recallK ?? Math.max(topK, 50);
		const rrfK = req.rrfK ?? cfg?.rrfK ?? DEFAULT_RRF_K;
		const weights = { dense: req.weights?.dense ?? cfg?.weights.dense ?? 1, lexical: req.weights?.lexical ?? cfg?.weights.lexical ?? 1 };
		const both = hasText && hasVector;
		const depth = both ? Math.max(recallK, topK) : topK;
		let tDense = 0;
		let tLex = 0;
		const [dense, lexical] = await Promise.all([
			hasVector
				? (async () => {
						const t = Date.now();
						const r = await this.query(req.vector!, { ...req.dense, topK: depth });
						tDense = Date.now() - t;
						return r;
					})()
				: Promise.resolve(undefined),
			hasText
				? (async () => {
						const t = Date.now();
						const r = await this.lexicalSearch(req.text!, depth);
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

	/** BM25 top `k` for `text` over the revision's lexical segments (see `LexIndex.search`). */
	async lexicalSearch(text: string, k: number): Promise<LexSearchResult> {
		if (this.closed) throw new ThreadError("index is closed", 400);
		return this.thread.withRecovery("lexicalSearch", async () => (await this.lexIndex()).search(text, k));
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
		if (this.refreshInterval > 0 && !this.pinned && Date.now() - this.lastCheck > this.refreshInterval) await this.refresh().catch((e: unknown) => console.error(`thread-sdk: refresh failed: ${e instanceof Error ? e.message : String(e)}`));
		return this.thread.withRecovery("query", async (attempt) => {
			const base = await this.acquireBase();
			try {
				const rev = this.revState;
				return await executeQuery(
					{
						w: W,
						store: this.store,
						rev,
						base,
						epoch: () => this.thread.epoch,
						wasmResets: this.thread.resets,
						wasmMemBytes: () => this.thread.memoryBytes(),
						getDelta: (ref, trace) => this.getDelta(ref, trace),
					},
					q,
					opts,
					attempt,
				);
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

	private async getDelta(ref: DeltaRef, trace: BlobTrace): Promise<Uint8Array> {
		const hit = this.deltaCache.get(ref.key);
		if (hit) return hit;
		const p = this.store.get(ref.key, undefined, { trace }).then((b) => {
			if (!b) throw new ThreadError(`missing delta segment ${ref.key}`, 404);
			return b;
		});
		const total = this.revState.deltas.reduce((a, d) => a + d.bytes, 0);
		if (total <= this.deltaBudget) {
			this.deltaCache.set(ref.key, p);
			p.catch(() => this.deltaCache.get(ref.key) === p && this.deltaCache.delete(ref.key));
		}
		return p;
	}

	private acquireBase(): Promise<BaseSlot> {
		if (this.base && this.base.epoch === this.thread.epoch && !this.base.dropped) {
			this.base.users++;
			return Promise.resolve(this.base);
		}
		if (!this.basePromise) {
			const rev = this.revState;
			const p = loadBase(this.store, rev, this.thread.epoch).then((b) => {
				const slot: BaseSlot = { ...b, users: 0, dropped: false };
				if (this.basePromise === p && !this.closed) this.base = slot;
				else slot.dropped = true; // superseded while loading
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
		if (slot.dropped && slot.users === 0 && slot.epoch === this.thread.epoch) {
			safeFree(slot.searcher);
			if (slot.pq) safeFree(slot.pq);
		}
	}

	private dropBase(): void {
		const b = this.base;
		this.base = null;
		this.basePromise = null;
		if (!b || b.dropped) return;
		b.dropped = true;
		if (b.users === 0 && b.epoch === this.thread.epoch) {
			safeFree(b.searcher);
			if (b.pq) safeFree(b.pq);
		}
	}
}

// ---------- loading ----------

export { asF32, headRevision, loadRevState, resolveTarget, snapshotKeys } from "./revision.js";

/** Loads the routing table (and PQ codebook) of a revision's generation into wasm memory. */
export async function loadBase(store: BlobStore, rev: RevState, epoch: number): Promise<Base> {
	const { manifest, files } = rev;
	const [c, f, p] = await Promise.all([getRequired(store, files.coarse), getRequired(store, files.fine), files.pq ? getRequired(store, files.pq) : Promise.resolve(null)]);
	if (rev.format === 3 && !p) throw new ThreadError(`format 3 generation ${rev.gen} has no PQ codebook`);
	let pq: W.PqCodebook | null = null;
	if (p) {
		pq = new W.PqCodebook(p);
		if (pq.dim() !== manifest.dim || (manifest.pq && pq.m() !== manifest.pq.m)) {
			const msg = `${files.pq} is dim ${pq.dim()} m ${pq.m()}, manifest dim ${manifest.dim} pq ${JSON.stringify(manifest.pq)}`;
			safeFree(pq);
			throw new ThreadError(msg);
		}
	}
	try {
		const searcher = new W.Searcher(manifest.dim, manifest.k1, manifest.k2, asF32(c, files.coarse), asF32(f, files.fine));
		return { key: rev.baseKey, searcher, pq, epoch };
	} catch (e) {
		if (pq) safeFree(pq);
		throw e;
	}
}
