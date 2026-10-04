/**
 * vib-sdk: query (and, experimentally, update) a vib blob-native vector index from a Cloudflare Worker.
 *
 *   import wasmModule from "vib-sdk/wasm/vib_bg.wasm";
 *   import { Vib } from "vib-sdk";
 *   const vib = await Vib.init({ wasmModule });                 // once per isolate
 *   const index = await vib.open({ bucket: env.BUCKET, waitUntil: ctx.waitUntil.bind(ctx) });
 *   const { hits } = await index.query(vector, { topK: 10 });
 */
import init, * as W from "./wasm/vib.js";
import { CachedBlobStore, PrefixedBlobStore, R2BlobStore, type BlobStore, type BlobTrace, type CacheLike, type R2BucketLike } from "./store.js";
import { executeQuery, safeFree, type Base, type QueryOptions, type QueryResult, type RevState } from "./query.js";
import { deleteIds, upsert, type DeleteResult, type UpsertInput, type UpsertResult, type WriterContext } from "./write.js";
import { buildTombState, keys, manifestFiles, parseCurrent, parseManifest, parseSlabsTable, VibError, type CurrentPointer, type DeltaRef, type Manifest, type QueryDefaults } from "./wire.js";

export * from "./wire.js";
export * from "./store.js";
export * from "./s3.js";
export type { Base, Hit, ParamSource, QueryOptions, QueryParams, QueryResult, QueryStats, RevState } from "./query.js";
export { Limiter, QUERY_INFLIGHT, executeQuery } from "./query.js";
export type { DeleteResult, UpsertInput, UpsertResult } from "./write.js";
/** The raw wasm-bindgen bindings (advanced use: build jobs, slab tools). */
export type WasmBindings = typeof W;

export interface VibInitOptions {
	/** The compiled `vib_bg.wasm` (`import wasmModule from "vib-sdk/wasm"` with the Cloudflare Vite plugin or Wrangler). */
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
}

/** One wasm instance per isolate. `Vib.init` once, then `open` indexes. */
export class Vib {
	/** Bumped on every reset; wasm objects created under an older epoch are invalid and must not be freed. */
	epoch = 0;
	resets = 0;
	private raw: W.InitOutput | null = null;
	private readonly resetListeners = new Set<() => void>();
	private constructor(readonly module: WebAssembly.Module) {}

	static async init(opts: VibInitOptions): Promise<Vib> {
		const v = new Vib(opts.wasmModule);
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
		console.error(`vib-sdk: wasm trap, resetting instance (reset #${this.resets}): ${reason instanceof Error ? `${reason.message}\n${reason.stack ?? ""}` : String(reason)}`);
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
			else console.error(`vib-sdk: ${label}: retrying after a concurrent wasm reset: ${e instanceof Error ? e.message : String(e)}`);
			return await fn(2);
		}
	}

	open(opts: OpenOptions): Promise<VibIndex> {
		return VibIndex.open(this, opts);
	}
}

interface BaseSlot extends Base {
	users: number;
	dropped: boolean;
}

/** An opened index: one (gen, rev) snapshot plus the generation's routing table in wasm memory. */
export class VibIndex {
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

	private constructor(
		readonly vib: Vib,
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
		this.unsubscribe = vib.onReset(() => {
			// The Searcher/PqCodebook belong to the dead instance: drop them without free().
			if (this.base) this.base.dropped = true;
			this.base = null;
			this.basePromise = null;
		});
	}

	static async open(vib: Vib, opts: OpenOptions): Promise<VibIndex> {
		let store: BlobStore | undefined = opts.store ?? (opts.bucket ? new R2BlobStore(opts.bucket) : undefined);
		if (!store) throw new VibError("open: pass a `bucket` (R2 binding) or a `store`", 400);
		if (opts.prefix) store = new PrefixedBlobStore(store, opts.prefix);
		const cache = opts.cache === false ? null : (opts.cache ?? ((globalThis as { caches?: { default?: CacheLike } }).caches?.default ?? null));
		if (cache) store = new CachedBlobStore(store, cache, { origin: opts.cacheOrigin, waitUntil: opts.waitUntil });
		const target = await resolveTarget(store, opts.gen, opts.rev);
		const rev = await loadRevState(store, target.gen, target.rev);
		const ix = new VibIndex(vib, store, rev, opts);
		await ix.acquireBase().then((s) => ix.releaseBase(s));
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
	/** The revision state (manifest, slabs table, deltas, tombstones) queries read. */
	get revision(): RevState {
		return this.revState;
	}

	/**
	 * Re-reads `idx/current.json` (or the generation's head) and switches to the new revision if it changed.
	 * Returns true when the revision changed. A pinned index (`gen` and `rev` given) never changes.
	 */
	refresh(): Promise<boolean> {
		if (this.closed) throw new VibError("index is closed", 400);
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

	/** Switches to (gen, rev); drops delta segments the new revision no longer lists. */
	async useRevision(gen: string, rev: number): Promise<void> {
		const next = await loadRevState(this.store, gen, rev);
		this.revState = next;
		const listed = new Set(next.deltas.map((d) => d.key));
		for (const k of this.deltaCache.keys()) if (!listed.has(k)) this.deltaCache.delete(k);
		if (this.base && this.base.key !== next.baseKey) this.dropBase();
	}

	/** Searches for the nearest neighbours of `vector` (any norm; it is normalized). */
	async query(vector: Float32Array | number[], opts: QueryOptions = {}): Promise<QueryResult> {
		if (this.closed) throw new VibError("index is closed", 400);
		const q = vector instanceof Float32Array ? vector : Float32Array.from(vector);
		if (this.refreshInterval > 0 && !this.pinned && Date.now() - this.lastCheck > this.refreshInterval) await this.refresh().catch((e: unknown) => console.error(`vib-sdk: refresh failed: ${e instanceof Error ? e.message : String(e)}`));
		return this.vib.withRecovery("query", async (attempt) => {
			const base = await this.acquireBase();
			try {
				const rev = this.revState;
				return await executeQuery(
					{
						w: W,
						store: this.store,
						rev,
						base,
						epoch: () => this.vib.epoch,
						wasmResets: this.vib.resets,
						wasmMemBytes: () => this.vib.memoryBytes(),
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

	/** Frees the wasm objects (routing table, PQ codebook). The index cannot be used afterwards. */
	close(): void {
		if (this.closed) return;
		this.closed = true;
		this.unsubscribe();
		this.dropBase();
		this.deltaCache.clear();
	}

	// ---- internals ----

	private writer(): WriterContext {
		if (this.closed) throw new VibError("index is closed", 400);
		return { vib: this.vib, w: W, store: this.store, gen: this.revState.gen, borrowPq: (key) => (this.base && !this.base.dropped && this.base.key === key && this.base.epoch === this.vib.epoch ? this.base.pq : null) };
	}

	private async getDelta(ref: DeltaRef, trace: BlobTrace): Promise<Uint8Array> {
		const hit = this.deltaCache.get(ref.key);
		if (hit) return hit;
		const p = this.store.get(ref.key, undefined, { trace }).then((b) => {
			if (!b) throw new VibError(`missing delta segment ${ref.key}`, 404);
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
		if (this.base && this.base.epoch === this.vib.epoch && !this.base.dropped) {
			this.base.users++;
			return Promise.resolve(this.base);
		}
		if (!this.basePromise) {
			const rev = this.revState;
			const p = loadBase(this.store, rev, this.vib.epoch).then((b) => {
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
		if (slot.dropped && slot.users === 0 && slot.epoch === this.vib.epoch) {
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
		if (b.users === 0 && b.epoch === this.vib.epoch) {
			safeFree(b.searcher);
			if (b.pq) safeFree(b.pq);
		}
	}
}

// ---------- loading ----------

async function getRequired(store: BlobStore, key: string): Promise<Uint8Array> {
	const b = await store.get(key);
	if (!b) throw new VibError(`missing object ${key}`, 404);
	return b;
}

/** The newest manifest revision of `gen` present in the store (needs `list`), else 0 (legacy `manifest.json`). */
export async function headRevision(store: BlobStore, gen: string): Promise<number> {
	if (!store.list) return 0;
	const prefix = `idx/${gen}/manifest.`;
	let head = -1;
	for (const key of await store.list(prefix)) {
		const m = key.slice(prefix.length).match(/^(?:(\d+)\.)?json$/);
		if (m) head = Math.max(head, m[1] ? Number(m[1]) : 0);
	}
	if (head < 0) throw new VibError(`no manifest for generation ${gen}`, 404);
	return head;
}

/**
 * The (gen, rev) to read: both given; `gen` alone at its head (current.json's rev when it names that
 * generation, else the newest manifest in the store); else `idx/current.json` (rev absent = 0, the legacy
 * `idx/{gen}/manifest.json`).
 */
export async function resolveTarget(store: BlobStore, gen?: string, rev?: number): Promise<CurrentPointer> {
	if (gen !== undefined && rev !== undefined) return { gen, rev };
	const cur = await store.get(keys.current());
	if (!cur) {
		if (gen === undefined) throw new VibError("no published index (idx/current.json missing)", 404);
		return { gen, rev: await headRevision(store, gen) };
	}
	const c = parseCurrent(cur);
	if (gen === undefined || gen === c.gen) return rev !== undefined ? { gen: c.gen, rev } : c;
	return { gen, rev: await headRevision(store, gen) };
}

/** Loads a revision: manifest (falling back to nothing else: rev 0 is `manifest.json`), slabs table, tombstone sets. */
export async function loadRevState(store: BlobStore, gen: string, rev: number): Promise<RevState> {
	const mb = await store.get(keys.manifest(gen, rev));
	if (!mb) throw new VibError(`no manifest ${keys.manifest(gen, rev)}`, 404);
	const manifest = parseManifest(mb, gen, rev);
	const files = manifestFiles(manifest);
	const format = manifest.format ?? 1;
	const cells = manifest.k1 * manifest.k2;
	const [slabs, sets] = await Promise.all([
		(async () => {
			if (!files.slabs) return null;
			const b = await store.get(files.slabs);
			// v1 generations may predate slabs tables: their slabs are read whole.
			if (!b && format < 2 && !manifest.files) return null;
			if (!b) throw new VibError(`missing ${files.slabs}`, 404);
			return parseSlabsTable(b, cells, files.slabs);
		})(),
		Promise.all(
			(manifest.tombstones ?? []).map(async (t) => {
				const b = await getRequired(store, t.key);
				if (b.byteLength % 4 !== 0) throw new VibError(`${t.key} is not u32-aligned`);
				return { rev: t.rev, ids: new Uint32Array(b.slice().buffer) };
			}),
		),
	]);
	if (format >= 2 && !slabs) throw new VibError(`${files.slabs} missing for format ${format}`);
	return { gen, rev, manifest, files, format, slabs, deltas: manifest.deltas ?? [], tomb: sets.length ? buildTombState(sets) : null, baseKey: `${gen}|${files.coarse}|${files.fine}|${files.pq ?? "-"}` };
}

/** A view of `b` as f32 (copying when the view is not 4-byte aligned). */
export function asF32(b: Uint8Array, what: string): Float32Array {
	if (b.byteLength % 4 !== 0) throw new VibError(`${what} is not f32-aligned`);
	return b.byteOffset % 4 === 0 ? new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4) : new Float32Array(b.slice().buffer);
}

/** Loads the routing table (and PQ codebook) of a revision's generation into wasm memory. */
export async function loadBase(store: BlobStore, rev: RevState, epoch: number): Promise<Base> {
	const { manifest, files } = rev;
	const [c, f, p] = await Promise.all([getRequired(store, files.coarse), getRequired(store, files.fine), files.pq ? getRequired(store, files.pq) : Promise.resolve(null)]);
	if (rev.format === 3 && !p) throw new VibError(`format 3 generation ${rev.gen} has no PQ codebook`);
	let pq: W.PqCodebook | null = null;
	if (p) {
		pq = new W.PqCodebook(p);
		if (pq.dim() !== manifest.dim || (manifest.pq && pq.m() !== manifest.pq.m)) {
			const msg = `${files.pq} is dim ${pq.dim()} m ${pq.m()}, manifest dim ${manifest.dim} pq ${JSON.stringify(manifest.pq)}`;
			safeFree(pq);
			throw new VibError(msg);
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
