/**
 * Blob stores: the SDK reads index objects through the `BlobStore` interface, so the same query code runs
 * against an R2 binding (`R2BlobStore`), any S3-compatible endpoint (`S3BlobStore`, see s3.ts), memory
 * (`MemoryBlobStore`, tests and small local indexes) and, in front of any of them, the Workers Cache API
 * (`CachedBlobStore`: immutable objects cached for a year, served with single- and multi-range requests) or an
 * in-process cache (`MemoryCache`, or `DiskCache` from `thread-sdk/node`).
 */
import { ThreadError } from "./wire.js";

export interface BlobRange {
	offset: number;
	length: number;
}

export interface BlobHead {
	size: number;
	/** Opaque version tag (R2: the hex etag without quotes). Used for compare-and-swap puts. */
	etag: string;
}

/** Per-call read counters a caller may pass to have the store account its work. */
export interface BlobTrace {
	/** Reads that reached the origin store (R2/S3/memory). */
	reads: number;
	/** Cache API outcomes (CachedBlobStore only). */
	hits: number;
	misses: number;
	/** Range requests the cache honoured (206) or answered with the full object (200). */
	r206: number;
	r200: number;
	/** Bytes transferred into the Worker. */
	bytes: number;
	/**
	 * Cache stores with the `"range+warm"` miss policy: whole-object fetches this call's misses queued in the
	 * background (not part of `reads`/`bytes`; `CachedBlobStore.warmStats` counts what they transferred).
	 */
	warms?: number;
	/**
	 * Block fills (`CachedBlobStore` over the Cache API, `blockBytes`): blocks this call read from the origin and put
	 * in the cache. Each is also one of `reads`, one of `misses`, and its whole block is in `bytes`.
	 */
	fills?: number;
}

export const newTrace = (): BlobTrace => ({ reads: 0, hits: 0, misses: 0, r206: 0, r200: 0, bytes: 0, warms: 0, fills: 0 });

/** Whether this runs inside the Cloudflare Workers runtime (6 simultaneous open connections per request). */
export const inWorkers = (): boolean => (globalThis as { navigator?: { userAgent?: string } }).navigator?.userAgent === "Cloudflare-Workers";

/**
 * One step of a cold-start timeline (`index.openStats`, `stats.timeline`): its wall time and the store work done
 * during it (the `BlobTrace` counters: origin reads, cache hits/misses, bytes). Python's SDK uses the same names.
 */
export interface PhaseStat extends BlobTrace {
	/** Wall time in ms. */
	ms: number;
}

/** Wall-clock ms (sub-ms where the runtime allows; in Workers it only advances across I/O, like `Date.now`). */
export const nowMs = (): number => (typeof performance !== "undefined" ? performance.now() : Date.now());
export const roundMs = (ms: number): number => Math.round(ms * 1000) / 1000;

/** A copy of a trace's counters (to diff later with `phaseStat`). */
export const snapTrace = (t: BlobTrace): BlobTrace => ({ reads: t.reads, hits: t.hits, misses: t.misses, r206: t.r206, r200: t.r200, bytes: t.bytes, fills: t.fills ?? 0 });

/** The phase between two snapshots of a trace (`before` defaults to an empty trace). */
export function phaseStat(ms: number, after: BlobTrace, before: BlobTrace = newTrace()): PhaseStat {
	return { ms: roundMs(ms), reads: after.reads - before.reads, hits: after.hits - before.hits, misses: after.misses - before.misses, r206: after.r206 - before.r206, r200: after.r200 - before.r200, bytes: after.bytes - before.bytes, fills: (after.fills ?? 0) - (before.fills ?? 0) };
}

export interface GetOptions {
	trace?: BlobTrace;
	/**
	 * Cache stores only: on a miss of a range read, fetch the WHOLE object, cache it and return the range as a
	 * view into it (the caller can recover the full object from the view's buffer). Default false: a miss reads
	 * just the range from the origin and caches nothing.
	 */
	fill?: boolean;
	/**
	 * The object's size, when the caller already knows it (a segment ref, a `head`). A `CachedBlobStore` over the
	 * Cache API then serves a range read that is not a `fill` from cached aligned blocks of the object
	 * (`CachedBlobStoreOptions.blockBytes`); without it, such a read that misses goes to the origin and caches nothing.
	 */
	size?: number;
}

export interface PutOptions {
	/** Create-only: fail (return null) if the key exists. */
	ifAbsent?: boolean;
	/** Compare-and-swap: fail (return null) unless the current etag equals this one. */
	ifMatch?: string;
	contentType?: string;
}

/**
 * Read (and optionally write) access to the objects of an index. Implementations may return views into a
 * larger buffer (for example the whole object fetched on a cache miss); callers must not assume
 * `byteOffset === 0` or `buffer.byteLength === byteLength`. A view keeps its whole buffer alive, so the SDK's stores
 * return one only where that buffer is held anyway or wanted: a `fill` read, an in-process cache's object, a pack
 * or warm pack in memory. Other range reads return the range in a buffer of its own.
 */
export interface BlobStore {
	/** The object, or `range` of it; `null` if it does not exist. */
	get(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null>;
	/**
	 * Several ranges of one object in ONE request (`Range: bytes=a-b, c-d`), one buffer per range in order.
	 * `null` when the store cannot serve them in one request (cache miss, no multi-range support); the caller
	 * then falls back to single-range `get`s. Optional.
	 */
	getRanges?(key: string, ranges: BlobRange[], opts?: GetOptions): Promise<Uint8Array[] | null>;
	head(key: string): Promise<BlobHead | null>;
	/**
	 * The last `length` bytes of an object (the whole object when it is shorter) and its size, in ONE request
	 * (`Range: bytes=-length`), so a reader of a trailer (a pack's footer and directory) needs no `head` first.
	 * `null` if the object does not exist. Optional: callers fall back to `head` + a range.
	 */
	getTail?(key: string, length: number, opts?: GetOptions): Promise<{ bytes: Uint8Array; size: number; etag?: string } | null>;
	/** Writes an object; `null` when a put condition failed. Optional (read-only stores). */
	put?(key: string, data: Uint8Array | string, opts?: PutOptions): Promise<BlobHead | null>;
	/** Keys under `prefix`. Optional (used to find a generation's head revision and by writers). */
	list?(prefix: string): Promise<string[]>;
	/**
	 * Objects under `prefix` with their size and upload time (ms since the epoch). Optional: GC needs it for its grace
	 * period (an object younger than it may belong to a writer that has not committed yet).
	 */
	listObjects?(prefix: string): Promise<ObjectInfo[]>;
	/** Deletes an object (a missing one is not an error). Optional: GC, and releasing `compact.lock`. */
	delete?(key: string): Promise<void>;
	/** Whether reads cross a network (`open` puts an in-process cache in front of remote stores outside Workers). */
	readonly remote?: boolean;
	/** Preferred reads in flight per query (a query's `inflight` default; 12 when absent). */
	readonly inflight?: number;
	/** Preferred phase-2 range merging (a query's `coalesceBytes` / `spanBytes` defaults). */
	readonly coalesceBytes?: number;
	readonly spanBytes?: number;
	/**
	 * Whether `key` (with `range`: that range of it) can be read without an origin request (in a local cache, a
	 * seeded prefix, or a local store): `undefined` when the store cannot tell. Without a range it means the whole
	 * object. Used by the cold-path query policy and by `index.warm`. Optional.
	 */
	contains?(key: string, range?: BlobRange): boolean | undefined;
	/** Opens `n` connections ahead of the first query (cheap HEADs); remote stores only. Optional. */
	warmup?(n: number): Promise<void>;
}

/** One object of a `listObjects` listing. */
export interface ObjectInfo {
	key: string;
	size: number;
	/** Upload (last modification) time, ms since the epoch. */
	uploaded: number;
}

export const rangeHeader = (ranges: BlobRange[]): string => "bytes=" + ranges.map((r) => `${r.offset}-${r.offset + r.length - 1}`).join(", ");

const toView = (data: Uint8Array | string): Uint8Array | string => (typeof data === "string" ? data : new Uint8Array(data.buffer, data.byteOffset, data.byteLength));

// ---------- R2 binding ----------

/** The subset of the Workers `R2Bucket` binding the SDK uses (structural, so no type dependency). */
export interface R2BucketLike {
	get(key: string, options?: { range?: BlobRange | { suffix: number } }): Promise<{ arrayBuffer(): Promise<ArrayBuffer>; etag: string; size: number } | null>;
	head(key: string): Promise<{ etag: string; size: number } | null>;
	put(key: string, value: ArrayBuffer | ArrayBufferView | string, options?: { onlyIf?: { etagMatches?: string; etagDoesNotMatch?: string }; httpMetadata?: { contentType?: string } }): Promise<{ etag: string; size: number } | null>;
	list(options?: { prefix?: string; cursor?: string; limit?: number }): Promise<{ objects: { key: string; size?: number; uploaded?: Date }[]; truncated: boolean; cursor?: string }>;
	delete?(keys: string | string[]): Promise<void>;
}

export class R2BlobStore implements BlobStore {
	readonly remote = true;
	constructor(public readonly bucket: R2BucketLike) {}

	async get(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		const obj = await this.bucket.get(key, range ? { range } : undefined);
		if (!obj) return null;
		const b = new Uint8Array(await obj.arrayBuffer());
		if (range && b.byteLength !== range.length) throw new ThreadError(`${key} [${range.offset}, +${range.length}): got ${b.byteLength} bytes`);
		if (opts?.trace) {
			opts.trace.reads++;
			opts.trace.bytes += b.byteLength;
		}
		return b;
	}

	async getTail(key: string, length: number, opts?: GetOptions): Promise<{ bytes: Uint8Array; size: number; etag?: string } | null> {
		const obj = await this.bucket.get(key, { range: { suffix: length } });
		if (!obj) return null;
		const bytes = new Uint8Array(await obj.arrayBuffer());
		if (opts?.trace) {
			opts.trace.reads++;
			opts.trace.bytes += bytes.byteLength;
		}
		return { bytes, size: obj.size, etag: obj.etag };
	}

	async head(key: string): Promise<BlobHead | null> {
		const o = await this.bucket.head(key);
		return o ? { size: o.size, etag: o.etag } : null;
	}

	async put(key: string, data: Uint8Array | string, opts: PutOptions = {}): Promise<BlobHead | null> {
		const onlyIf = opts.ifAbsent ? { etagDoesNotMatch: "*" } : opts.ifMatch !== undefined ? { etagMatches: opts.ifMatch } : undefined;
		const o = await this.bucket.put(key, toView(data), { ...(onlyIf ? { onlyIf } : {}), ...(opts.contentType ? { httpMetadata: { contentType: opts.contentType } } : {}) });
		return o ? { size: o.size, etag: o.etag } : null;
	}

	async list(prefix: string): Promise<string[]> {
		const out: string[] = [];
		let cursor: string | undefined;
		do {
			const r = await this.bucket.list({ prefix, cursor, limit: 1000 });
			for (const o of r.objects) out.push(o.key);
			cursor = r.truncated ? r.cursor : undefined;
		} while (cursor);
		return out;
	}

	async listObjects(prefix: string): Promise<ObjectInfo[]> {
		const out: ObjectInfo[] = [];
		let cursor: string | undefined;
		do {
			const r = await this.bucket.list({ prefix, cursor, limit: 1000 });
			for (const o of r.objects) {
				if (!o.uploaded || o.size === undefined) throw new ThreadError("this R2 binding's list() gives no upload times", 400);
				out.push({ key: o.key, size: o.size, uploaded: o.uploaded.getTime() });
			}
			cursor = r.truncated ? r.cursor : undefined;
		} while (cursor);
		return out;
	}

	async delete(key: string): Promise<void> {
		if (!this.bucket.delete) throw new ThreadError("this R2 binding cannot delete", 400);
		await this.bucket.delete(key);
	}
}

// ---------- in-memory store (tests, tiny local indexes) ----------

export class MemoryBlobStore implements BlobStore {
	readonly objects = new Map<string, { bytes: Uint8Array; etag: string; uploaded?: number }>();
	/** The clock upload times come from (tests move it to age objects). */
	now: () => number = Date.now;
	private seq = 0;

	async get(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		const o = this.objects.get(key);
		if (!o) return null;
		const b = range ? o.bytes.slice(range.offset, range.offset + range.length) : o.bytes;
		if (range && b.byteLength !== range.length) throw new ThreadError(`${key} [${range.offset}, +${range.length}) is outside the object (${o.bytes.byteLength} bytes)`);
		if (opts?.trace) {
			opts.trace.reads++;
			opts.trace.bytes += b.byteLength;
		}
		return b;
	}

	async getRanges(key: string, ranges: BlobRange[], opts?: GetOptions): Promise<Uint8Array[] | null> {
		const o = this.objects.get(key);
		if (!o) return null;
		const out = ranges.map((r) => {
			if (r.offset < 0 || r.length < 0 || r.offset + r.length > o.bytes.byteLength) throw new ThreadError(`${key} [${r.offset}, +${r.length}) is outside the object (${o.bytes.byteLength} bytes)`);
			return o.bytes.slice(r.offset, r.offset + r.length);
		});
		if (opts?.trace) {
			opts.trace.reads++;
			opts.trace.bytes += out.reduce((a, b) => a + b.byteLength, 0);
		}
		return out;
	}

	async head(key: string): Promise<BlobHead | null> {
		const o = this.objects.get(key);
		return o ? { size: o.bytes.byteLength, etag: o.etag } : null;
	}

	async getTail(key: string, length: number, opts?: GetOptions): Promise<{ bytes: Uint8Array; size: number; etag?: string } | null> {
		const o = this.objects.get(key);
		if (!o) return null;
		const bytes = o.bytes.slice(Math.max(0, o.bytes.byteLength - length));
		if (opts?.trace) {
			opts.trace.reads++;
			opts.trace.bytes += bytes.byteLength;
		}
		return { bytes, size: o.bytes.byteLength, etag: o.etag };
	}

	async put(key: string, data: Uint8Array | string, opts: PutOptions = {}): Promise<BlobHead | null> {
		const cur = this.objects.get(key);
		if (opts.ifAbsent && cur) return null;
		if (opts.ifMatch !== undefined && cur?.etag !== opts.ifMatch) return null;
		const bytes = typeof data === "string" ? new TextEncoder().encode(data) : data.slice();
		const o = { bytes, etag: `m${++this.seq}`, uploaded: this.now() };
		this.objects.set(key, o);
		return { size: bytes.byteLength, etag: o.etag };
	}

	async list(prefix: string): Promise<string[]> {
		return [...this.objects.keys()].filter((k) => k.startsWith(prefix)).sort();
	}

	async listObjects(prefix: string): Promise<ObjectInfo[]> {
		return (await this.list(prefix)).map((key) => {
			const o = this.objects.get(key)!;
			return { key, size: o.bytes.byteLength, uploaded: o.uploaded ?? 0 };
		});
	}

	async delete(key: string): Promise<void> {
		this.objects.delete(key);
	}
}

// ---------- Cache API layer ----------

/** The part of the Workers `Cache` interface the SDK uses (`caches.default` or a named cache). */
export interface CacheLike {
	match(request: Request): Promise<Response | undefined>;
	put(request: Request, response: Response): Promise<void>;
	/** Optional: dropped entries of deleted objects (Workers' `Cache.delete`). */
	delete?(request: Request): Promise<boolean>;
}

/**
 * What a cache store does when a range read that asked for `fill` (a slab's scan prefix) misses:
 *
 * - `"whole"`: GET the whole object from the origin, cache it, return the range as a view into it (the query then
 *   also has the slab's rerank records in memory). Right when a shared cache serves later queries (the Workers
 *   Cache API, an edge cache): the first query pays the transfer once for everybody.
 * - `"range"`: read exactly the range from the origin and cache nothing.
 * - `"range+warm"`: read exactly the range now, and fetch the whole object into the cache in the background
 *   (bounded concurrency, deduplicated, `waitUntil` when given, never on the query's critical path). Right for an
 *   in-process cache, where the whole-object transfer would otherwise sit inside the first (cold) query.
 *
 * Reads without `fill` (rerank records, lexical postings) never fetch whole objects; whole-object reads (no range)
 * are cached under every policy but `"range"`. Over the Cache API, a range read without `fill` that passes the
 * object's `size` is served from cached blocks of it instead (`CachedBlobStoreOptions.blockBytes`), under every policy.
 */
export type MissPolicy = "whole" | "range" | "range+warm";
export const MISS_POLICIES: readonly MissPolicy[] = ["whole", "range", "range+warm"];

/**
 * Whole immutable objects by key, held by this process (`MemoryCache`; `DiskCache` from `thread-sdk/node`):
 * the in-process counterpart of the Workers Cache API, used through `CachedBlobStore`.
 */
export interface BlobCache {
	/** The whole object, or null when it is not cached. */
	get(key: string): Promise<Uint8Array | null>;
	/** Ranges of a cached object (views are fine), or null when it is not cached. */
	getRanges(key: string, ranges: BlobRange[]): Promise<Uint8Array[] | null>;
	put(key: string, bytes: Uint8Array): Promise<unknown>;
	/** Whether the whole object is cached (no I/O). */
	has(key: string): boolean;
	/** The miss policy a `CachedBlobStore` over this cache uses by default (`"range+warm"` when absent). */
	readonly defaultMissPolicy?: MissPolicy;
	/**
	 * Optional prefix seeding (`PrefixSeed`, warm packs): hold the first `bytes.byteLength` bytes of `key` only.
	 * `getRanges` then serves ranges inside a held prefix; `get`/`has` still mean the whole object.
	 */
	seedPrefix?(key: string, bytes: Uint8Array): void;
	/** Whether `range` of `key` can be served (whole object or a held prefix). */
	holds?(key: string, range: BlobRange): boolean;
	/** Drops the held prefixes `keep` rejects (a revision switch). */
	retainPrefixes?(keep: (key: string) => boolean): void;
	/** Drops an object (a store delete). Optional. */
	delete?(key: string): void;
}

const ownBuffer = (b: Uint8Array): Uint8Array => (b.byteOffset === 0 && b.buffer.byteLength === b.byteLength ? b : b.slice());

/**
 * `[a, e)` of `b` in a buffer of its own size (`b` itself when it is all of its buffer). A range read that is not a
 * `fill` returns this, so a range the caller keeps never holds the block, response body or object it was cut from.
 */
const copyOut = (b: Uint8Array, a = 0, e = b.byteLength): Uint8Array => (a === 0 && e === b.byteLength ? ownBuffer(b) : b.slice(a, e));

/**
 * An LRU of whole objects in memory (default 256 MB; objects larger than a quarter of it are not kept). Hits are
 * views into the cached object, so a slab prefix served from it carries the slab's rerank records too.
 *
 * It also holds seeded PREFIXES (`seedPrefix`, a warm pack's slab scan prefixes): ranges inside one are served
 * from it, the whole object is not (`get` / `has` miss), so a `CachedBlobStore` still fetches the whole slab in the
 * background on a prefix hit (`"range+warm"`). Prefixes are views into the warm pack, are not counted in `held` and
 * are not evicted by the LRU (they are bounded by the warm pack); one is dropped once its whole object is cached.
 */
export class MemoryCache implements BlobCache {
	held = 0;
	private readonly items = new Map<string, Uint8Array>();
	private readonly prefixes = new Map<string, Uint8Array>();
	constructor(readonly maxBytes = 256 * 1024 * 1024) {}

	has(key: string): boolean {
		return this.items.has(key);
	}

	seedPrefix(key: string, bytes: Uint8Array): void {
		if (this.items.has(key)) return;
		const cur = this.prefixes.get(key);
		if (!cur || cur.byteLength < bytes.byteLength) this.prefixes.set(key, bytes);
	}

	holds(key: string, range: BlobRange): boolean {
		const b = this.items.get(key) ?? this.prefixes.get(key);
		return !!b && range.offset >= 0 && range.length >= 0 && range.offset + range.length <= b.byteLength;
	}

	retainPrefixes(keep: (key: string) => boolean): void {
		for (const k of [...this.prefixes.keys()]) if (!keep(k)) this.prefixes.delete(k);
	}

	/** Seeded prefixes held and their bytes. */
	get prefixStats(): { prefixes: number; bytes: number } {
		let bytes = 0;
		for (const b of this.prefixes.values()) bytes += b.byteLength;
		return { prefixes: this.prefixes.size, bytes };
	}

	private touch(key: string): Uint8Array | null {
		const b = this.items.get(key);
		if (!b) return null;
		this.items.delete(key);
		this.items.set(key, b);
		return b;
	}

	async get(key: string): Promise<Uint8Array | null> {
		return this.touch(key);
	}

	async getRanges(key: string, ranges: BlobRange[]): Promise<Uint8Array[] | null> {
		const b = this.touch(key);
		if (!b) {
			const p = this.prefixes.get(key);
			if (!p || !ranges.length || !ranges.every((r) => r.offset >= 0 && r.length >= 0 && r.offset + r.length <= p.byteLength)) return null;
			return ranges.map((r) => p.subarray(r.offset, r.offset + r.length));
		}
		return ranges.map((r) => {
			if (r.offset < 0 || r.length < 0 || r.offset + r.length > b.byteLength) throw new ThreadError(`cached ${key} is ${b.byteLength} bytes, need [${r.offset}, +${r.length})`);
			return b.subarray(r.offset, r.offset + r.length);
		});
	}

	delete(key: string): void {
		const old = this.items.get(key);
		if (old) this.held -= old.byteLength;
		this.items.delete(key);
		this.prefixes.delete(key);
	}

	async put(key: string, bytes: Uint8Array): Promise<void> {
		if (bytes.byteLength > this.maxBytes / 4) return;
		const own = ownBuffer(bytes);
		const old = this.items.get(key);
		if (old) {
			this.held -= old.byteLength;
			this.items.delete(key);
		}
		this.items.set(key, own);
		this.prefixes.delete(key);
		this.held += own.byteLength;
		for (const [k, v] of this.items) {
			if (this.held <= this.maxBytes) break;
			this.items.delete(k);
			this.held -= v.byteLength;
		}
	}
}

/** Cache API reads in flight per query (they are not held to the Workers limit of 6 connections). */
export const CACHE_API_INFLIGHT = 12;

/** Default `blockBytes` of a `CachedBlobStore` over the Cache API (docs/lexical.md "Block cache" has the measurements). */
export const DEFAULT_BLOCK_BYTES = 1024 * 1024;
/** Default memory of the blocks a store buffers at once (`blockBuffers` = this / `blockBytes`, at least 2). */
export const BLOCK_BUFFER_BYTES = 16 * 1024 * 1024;
/** Smallest and largest `blockBytes` (the Cache API holds at most 512 MB per entry). */
export const MIN_BLOCK_BYTES = 64 * 1024;
export const MAX_BLOCK_BYTES = 256 * 1024 * 1024;
/**
 * A block slot held this long is given back even if its cache put never settled: a put cancelled with the request
 * that started it (no `waitUntil`, or past its 30 s) would otherwise hold the slot for the life of the isolate.
 */
const BLOCK_SLOT_STALE_MS = 30_000;

/**
 * A counting semaphore. A holder older than `staleMs` is reclaimed by the next `take` that finds no free slot, and a
 * waiter looks again every `staleMs`, so a holder whose promise never settles cannot block it forever.
 */
class Slots {
	private readonly holders = new Set<{ since: number }>();
	private readonly waiters: (() => void)[] = [];
	constructor(
		readonly max: number,
		readonly staleMs: number,
	) {}

	get used(): number {
		return this.holders.size;
	}

	/** Waits for a slot; resolves to its release (idempotent). */
	async take(): Promise<() => void> {
		for (;;) {
			if (this.holders.size >= this.max) {
				const now = Date.now();
				for (const h of this.holders) if (now - h.since > this.staleMs) this.holders.delete(h);
			}
			if (this.holders.size < this.max) {
				const h = { since: Date.now() };
				this.holders.add(h);
				return () => {
					if (this.holders.delete(h)) this.waiters.shift()?.();
				};
			}
			await new Promise<void>((resolve) => {
				const t = setTimeout(resolve, this.staleMs);
				(t as { unref?: () => void }).unref?.();
				this.waiters.push(() => {
					clearTimeout(t);
					resolve();
				});
			});
		}
	}
}

export interface CachedBlobStoreOptions {
	/** Synthetic origin the cache keys live under (any https URL; objects are keyed `${origin}${scope}${key}`). Cache API only. */
	origin?: string;
	/**
	 * A prefix for every cache key (Cache API and in-process), so two indexes that share one cache but live under
	 * different prefixes of one bucket never see each other's objects. `ThreadIndex.open` sets it to `prefix`.
	 */
	scope?: string;
	/** Keeps background cache writes and warm fetches alive after the response (pass `ctx.waitUntil`). */
	waitUntil?: (p: Promise<unknown>) => void;
	/** Which keys are immutable and may be cached. Default: everything except `current.json`. */
	cacheable?: (key: string) => boolean;
	cacheControl?: string;
	/** See `MissPolicy`. Default `"whole"` for the Cache API, the cache's `defaultMissPolicy` (else `"range+warm"`) for a `BlobCache`. */
	missPolicy?: MissPolicy;
	/** `"range+warm"`: whole-object fetches running at once in the background (default 2; at most 1 while foreground reads are in flight; 0 disables warming). */
	warmConcurrency?: number;
	/** `"range+warm"`: most warm fetches queued (default 1024); later misses are not warmed until the queue drains. */
	warmQueue?: number;
	/** Reads in flight per query (default: 12 for the Cache API, else the inner store's). */
	inflight?: number;
	/** Phase-2 range merging defaults for queries (default: the inner store's). */
	coalesceBytes?: number;
	spanBytes?: number;
	/**
	 * Cache API only: the block size for range reads of objects that are not cached whole. A range read without
	 * `fill` that passes the object's `size` (lexical segments over `LexSegmentOptions.fillBytes`) is served from the
	 * aligned blocks it covers, `[i * blockBytes, min((i + 1) * blockBytes, size))`, each cached under its own key; a
	 * block that misses is read from the origin (that block only), serves the read and is put in the cache through
	 * `waitUntil`. An object already cached whole is still served from that entry. Default `DEFAULT_BLOCK_BYTES`
	 * (1 MiB); 0 turns blocks off (such reads go to the origin and are not cached, as before 0.2.5).
	 */
	blockBytes?: number;
	/**
	 * Blocks held in memory at once, from the start of their origin read until their cache put settles; a block miss
	 * past it waits. Default `BLOCK_BUFFER_BYTES / blockBytes` (16 MiB worth), at least 2. Block reads from the
	 * origin in flight are also at most `inflight`. Ranges are copied out of their blocks before a block's slot is
	 * given back, so the memory block reads keep alive is the bytes returned plus at most this many blocks.
	 */
	blockBuffers?: number;
}

/** What the block cache of a `CachedBlobStore` did since it was created. */
export interface BlockStats {
	/** Blocks read from the origin and put in the cache, and their bytes. */
	fills: number;
	bytes: number;
	/** Cache puts of blocks that failed. */
	failed: number;
}

/** Background warm fetches of a `CachedBlobStore` (`"range+warm"`). */
export interface WarmStats {
	queued: number;
	fetched: number;
	bytes: number;
	failed: number;
	/** Queued keys dropped because the queue was full. */
	dropped: number;
}

/** A part of a 206 body: absolute `[start, end]` and its bytes. */
export interface RangePart {
	start: number;
	end: number;
	bytes: Uint8Array;
}

const latin1 = new TextDecoder("latin1");

/** Splits a `multipart/byteranges` body into its parts (RFC 9110 §14.6). */
export function parseMultipart(body: Uint8Array, boundary: string): RangePart[] {
	const parts: RangePart[] = [];
	const delim = new TextEncoder().encode(`--${boundary}`);
	const find = (pat: Uint8Array, from: number) => {
		outer: for (let i = from; i + pat.length <= body.length; i++) {
			for (let j = 0; j < pat.length; j++) if (body[i + j] !== pat[j]) continue outer;
			return i;
		}
		return -1;
	};
	const crlf2 = new Uint8Array([13, 10, 13, 10]);
	let pos = find(delim, 0);
	while (pos >= 0) {
		const after = pos + delim.length;
		if (body[after] === 45 && body[after + 1] === 45) break; // closing "--"
		const hdrEnd = find(crlf2, after);
		if (hdrEnd < 0) throw new ThreadError("multipart: unterminated part headers");
		const headers = latin1.decode(body.subarray(after, hdrEnd));
		const m = headers.match(/content-range:\s*bytes\s+(\d+)-(\d+)\//i);
		if (!m) throw new ThreadError(`multipart: part without Content-Range: ${headers.slice(0, 200)}`);
		const start = Number(m[1]);
		const end = Number(m[2]);
		const dataStart = hdrEnd + 4;
		const dataEnd = dataStart + (end - start + 1);
		if (dataEnd > body.length) throw new ThreadError("multipart: part runs past the body");
		parts.push({ start, end, bytes: body.subarray(dataStart, dataEnd) });
		pos = find(delim, dataEnd);
	}
	return parts;
}

/**
 * Parts of a Range response: a multipart/byteranges 206, a single-part 206 (ranges merged by the server)
 * or a 200 (Range ignored, whole object).
 */
export function responseParts(status: number, contentType: string | null, contentRange: string | null, body: Uint8Array): { parts: RangePart[]; kind: "multipart" | "single" | "full" } {
	if (status === 206 && /multipart\/byteranges/i.test(contentType ?? "")) {
		const b = (contentType ?? "").match(/boundary=("?)([^";]+)\1/i);
		if (!b) throw new ThreadError(`multipart without boundary: ${contentType}`);
		return { parts: parseMultipart(body, b[2]), kind: "multipart" };
	}
	if (status === 206) {
		const m = (contentRange ?? "").match(/bytes\s+(\d+)-(\d+)\//);
		if (!m) throw new ThreadError("206 without Content-Range");
		return { parts: [{ start: Number(m[1]), end: Number(m[2]), bytes: body }], kind: "single" };
	}
	return { parts: [{ start: 0, end: body.length - 1, bytes: body }], kind: "full" };
}

/** Cuts the requested ranges out of the parts of a Range response. */
export function sliceRanges(parts: RangePart[], ranges: BlobRange[], key: string): Uint8Array[] {
	return ranges.map((rg) => {
		const p = parts.find((p) => p.start <= rg.offset && rg.offset + rg.length - 1 <= p.end);
		if (!p) throw new ThreadError(`range ${rg.offset}+${rg.length} of ${key} missing from the response (${parts.length} parts)`);
		return p.bytes.subarray(rg.offset - p.start, rg.offset - p.start + rg.length);
	});
}

/**
 * A cache in front of another store: the Workers Cache API (`CacheLike`, e.g. `caches.default`) or an in-process
 * `BlobCache` (`MemoryCache`, `DiskCache`). Index objects under `idx/` are immutable (every revision writes new
 * keys), so whole objects are cached (Cache API: under a synthetic URL for a year) and ranges are served from them:
 * the Cache API answers a `Range` against a cached full object with a 206 (multipart for several ranges). What a
 * miss of a `fill` read does is the `missPolicy`.
 */
export class CachedBlobStore implements BlobStore {
	readonly origin: string;
	readonly scope: string;
	readonly missPolicy: MissPolicy;
	readonly remote: boolean;
	readonly inflight: number | undefined;
	readonly coalesceBytes: number | undefined;
	readonly spanBytes: number | undefined;
	/** The Cache API, or null for a `BlobCache`. */
	readonly cacheApi: CacheLike | null;
	/** The in-process cache, or null for the Cache API. */
	readonly blobCache: BlobCache | null;
	readonly warmStats: WarmStats = { queued: 0, fetched: 0, bytes: 0, failed: 0, dropped: 0 };
	/** Block size of the block cache (0: off, always for an in-process `BlobCache`). */
	readonly blockBytes: number;
	readonly blockStats: BlockStats = { fills: 0, bytes: 0, failed: 0 };
	/** Blocks buffered (origin read until cache put) and block reads from the origin in flight. */
	private readonly blockBuffers: Slots;
	private readonly blockReads: Slots;
	/** Block fills in flight by block id, until their cache put settles (concurrent readers of a block share one). */
	private readonly blockFills = new Map<string, Promise<Uint8Array | null>>();
	private readonly waitUntil: ((p: Promise<unknown>) => void) | undefined;
	private readonly cacheable: (key: string) => boolean;
	private readonly cacheControl: string;
	private readonly warmConcurrency: number;
	private readonly warmQueueMax: number;
	/** Whole-object fetches in flight (concurrent misses of one key share one). */
	private readonly fills = new Map<string, Promise<Uint8Array | null>>();
	private readonly warmQ: string[] = [];
	private readonly warming = new Set<string>();
	private warmActive = 0;
	/** Foreground origin reads in flight (warm fetches run at full concurrency only when there are none). */
	private foreground = 0;
	private idleWaiters: (() => void)[] = [];
	/** Cache API: keys this instance stored or saw hit (residency is otherwise unknown without a lookup). */
	private readonly known = new Set<string>();

	constructor(
		public readonly inner: BlobStore,
		public readonly cache: CacheLike | BlobCache,
		opts: CachedBlobStoreOptions = {},
	) {
		const isApi = typeof (cache as CacheLike).match === "function";
		this.cacheApi = isApi ? (cache as CacheLike) : null;
		this.blobCache = isApi ? null : (cache as BlobCache);
		this.origin = (opts.origin ?? "https://vib-sdk.cache/").replace(/\/?$/, "/");
		this.scope = opts.scope ?? "";
		this.waitUntil = opts.waitUntil;
		this.cacheable = opts.cacheable ?? ((k) => !k.endsWith("current.json"));
		this.cacheControl = opts.cacheControl ?? "public, max-age=31536000, immutable";
		this.missPolicy = opts.missPolicy ?? (isApi ? "whole" : (this.blobCache!.defaultMissPolicy ?? "range+warm"));
		if (!MISS_POLICIES.includes(this.missPolicy)) throw new ThreadError(`missPolicy must be one of ${MISS_POLICIES.join(", ")}`, 400);
		this.warmConcurrency = opts.warmConcurrency ?? 2;
		this.warmQueueMax = opts.warmQueue ?? 1024;
		this.remote = inner.remote ?? false;
		this.inflight = opts.inflight ?? (isApi ? CACHE_API_INFLIGHT : inner.inflight);
		this.coalesceBytes = opts.coalesceBytes ?? inner.coalesceBytes;
		this.spanBytes = opts.spanBytes ?? inner.spanBytes;
		const bb = opts.blockBytes ?? DEFAULT_BLOCK_BYTES;
		if (!Number.isInteger(bb) || (bb !== 0 && (bb < MIN_BLOCK_BYTES || bb > MAX_BLOCK_BYTES))) throw new ThreadError(`blockBytes must be 0 or an integer in [${MIN_BLOCK_BYTES}, ${MAX_BLOCK_BYTES}]`, 400);
		this.blockBytes = isApi ? bb : 0;
		const nb = opts.blockBuffers ?? Math.max(2, Math.floor(BLOCK_BUFFER_BYTES / (bb || DEFAULT_BLOCK_BYTES)));
		if (!Number.isInteger(nb) || nb < 1) throw new ThreadError("blockBuffers must be an integer >= 1", 400);
		this.blockBuffers = new Slots(nb, BLOCK_SLOT_STALE_MS);
		this.blockReads = new Slots(Math.max(1, Math.min(nb, this.inflight ?? CACHE_API_INFLIGHT)), BLOCK_SLOT_STALE_MS);
	}

	request(key: string, range?: string): Request {
		return new Request(this.origin + this.scope + key.replace(/^\/+/, ""), { method: "GET", headers: range ? { range } : {} });
	}

	/** The cache request of block `i` of `key` (under the scope, so indexes sharing a cache never share blocks). */
	blockRequest(key: string, i: number, range?: string): Request {
		return new Request(`${this.origin}${this.scope}${key.replace(/^\/+/, "")}/~block/${this.blockBytes}/${i}`, { method: "GET", headers: range ? { range } : {} });
	}

	private note(key: string): void {
		if (!this.cacheApi) return;
		if (this.known.size >= 65536) this.known.clear();
		this.known.add(key);
	}

	/** Stores a full immutable object in the cache; the returned promise is also handed to `waitUntil`. */
	store(key: string, bytes: Uint8Array): Promise<void> {
		this.note(key);
		let p: Promise<void>;
		if (this.cacheApi) {
			const resp = new Response(bytes.slice(), {
				headers: { "content-type": "application/octet-stream", "content-length": String(bytes.byteLength), "cache-control": this.cacheControl, "accept-ranges": "bytes" },
			});
			p = this.cacheApi.put(this.request(key), resp);
		} else p = this.blobCache!.put(this.scope + key, bytes).then(() => {});
		p = p.catch((e: unknown) => console.error(`thread-sdk: cache put ${key} failed: ${e instanceof Error ? e.message : String(e)}`));
		this.waitUntil?.(p);
		return p;
	}

	/** Runs a foreground origin read (counted, so background warms yield to it). */
	private async origin_<T>(fn: () => Promise<T>): Promise<T> {
		this.foreground++;
		try {
			return await fn();
		} finally {
			if (--this.foreground === 0) this.pumpWarm();
		}
	}

	/** Reads the whole object from the origin once (concurrent callers share the read) and caches it unless `cache` is false. */
	private fetchWhole(key: string, opts: GetOptions | undefined, cache: boolean): Promise<Uint8Array | null> {
		const pending = this.fills.get(key);
		if (pending) return pending;
		const done = () => {
			if (this.fills.get(key) === p) this.fills.delete(key);
			this.wakeIdle();
		};
		const p: Promise<Uint8Array | null> = (async () => {
			const full = await this.origin_(() => this.inner.get(key, undefined, opts?.trace ? { trace: opts.trace } : undefined));
			// The caller gets the object at once; the fill stays registered until the cache holds it, so a concurrent
			// miss shares this read instead of starting another.
			if (full && cache) this.store(key, full).then(done, done);
			else queueMicrotask(done);
			return full;
		})();
		p.catch(done);
		this.fills.set(key, p);
		return p;
	}

	private static slice(key: string, full: Uint8Array, ranges: BlobRange[]): Uint8Array[] {
		for (const r of ranges) if (r.offset + r.length > full.byteLength) throw new ThreadError(`${key} is ${full.byteLength} bytes, need ${r.offset + r.length}`);
		return ranges.map((r) => full.subarray(r.offset, r.offset + r.length));
	}

	/**
	 * Queues a background whole-object fetch of `key` into the cache (deduplicated, bounded, `waitUntil`), as the
	 * `"range+warm"` policy does after a range miss. A warm-pack prefix layer in front of this store calls it.
	 */
	warmObject(key: string, trace?: BlobTrace): void {
		if (this.cacheable(key)) this.warm(key, trace);
	}

	/** Whether the cache takes seeded prefixes (`MemoryCache`): `open({warmPack})` then seeds it instead of a layer. */
	get seedsPrefixes(): boolean {
		return typeof this.blobCache?.seedPrefix === "function";
	}

	seedPrefix(key: string, bytes: Uint8Array): void {
		if (!this.blobCache?.seedPrefix) throw new ThreadError("this cache does not hold prefixes", 400);
		this.blobCache.seedPrefix(this.scope + key, bytes);
	}

	/** Drops seeded prefixes `keep` rejects. */
	retain(keep: (key: string) => boolean): void {
		this.blobCache?.retainPrefixes?.(keep);
	}

	/** Queues a background whole-object fetch of `key` (`"range+warm"`). */
	private warm(key: string, trace?: BlobTrace): void {
		if (this.warmConcurrency <= 0 || this.warming.has(key) || this.fills.has(key) || this.blobCache?.has(this.scope + key)) return;
		if (this.warmQ.length >= this.warmQueueMax) {
			this.warmStats.dropped++;
			return;
		}
		this.warming.add(key);
		this.warmQ.push(key);
		this.warmStats.queued++;
		if (trace) trace.warms = (trace.warms ?? 0) + 1;
		this.pumpWarm();
	}

	private pumpWarm(): void {
		const cap = this.foreground === 0 ? this.warmConcurrency : Math.min(1, this.warmConcurrency);
		while (this.warmActive < cap && this.warmQ.length) {
			const key = this.warmQ.shift()!;
			this.warmActive++;
			const p = (async () => {
				if (this.blobCache?.has(this.scope + key)) return;
				const full = await this.inner.get(key);
				if (!full) return;
				this.warmStats.fetched++;
				this.warmStats.bytes += full.byteLength;
				await this.store(key, full);
			})()
				.catch((e: unknown) => {
					this.warmStats.failed++;
					console.error(`thread-sdk: warm fetch of ${key} failed: ${e instanceof Error ? e.message : String(e)}`);
				})
				.finally(() => {
					this.warmActive--;
					this.warming.delete(key);
					this.pumpWarm();
					this.wakeIdle();
				});
			this.waitUntil?.(p);
		}
	}

	private wakeIdle(): void {
		if (this.warmActive || this.warmQ.length || this.fills.size || this.blockFills.size) return;
		const w = this.idleWaiters;
		this.idleWaiters = [];
		for (const fn of w) fn();
	}

	/** Resolves once no background warm fetch, whole-object fill or block fill is queued or running (tests, benchmarks). */
	async drain(): Promise<void> {
		while (this.warmActive || this.warmQ.length || this.fills.size || this.blockFills.size) await new Promise<void>((r) => this.idleWaiters.push(r));
	}

	contains(key: string, range?: BlobRange): boolean | undefined {
		if (!this.cacheable(key)) return this.inner.contains?.(key, range);
		if (this.blobCache) return this.blobCache.has(this.scope + key) || (range !== undefined && !!this.blobCache.holds?.(this.scope + key, range)) || this.inner.contains?.(key, range) || false;
		return this.known.has(key) ? true : this.inner.contains?.(key, range);
	}

	warmup(n: number): Promise<void> {
		return this.inner.warmup?.(n) ?? Promise.resolve();
	}

	/** A miss of a `fill` range read, per the miss policy. */
	private async missFill(key: string, ranges: BlobRange[], single: boolean, opts: GetOptions): Promise<Uint8Array[] | null> {
		if (this.missPolicy === "whole") {
			const full = await this.fetchWhole(key, opts, true);
			return full ? CachedBlobStore.slice(key, full, ranges) : null;
		}
		const trace = opts.trace ? { trace: opts.trace } : undefined;
		const got = single ? await this.origin_(() => this.inner.get(key, ranges[0], trace)).then((b) => (b ? [b] : null)) : await this.origin_(async () => (await this.inner.getRanges?.(key, ranges, trace)) ?? null);
		if (got && this.missPolicy === "range+warm") this.warm(key, opts.trace);
		return got;
	}

	/** Whether a range read goes through the block cache: Cache API, blocks on, not a `fill`, the size known. */
	private blocked(opts: GetOptions | undefined): opts is GetOptions & { size: number } {
		return this.blockBytes > 0 && !opts?.fill && typeof opts?.size === "number" && Number.isSafeInteger(opts.size) && opts.size >= 0;
	}

	private blockId(key: string, i: number): string {
		return `${key}\n${i}`;
	}

	/** The Cache API entry of the whole object, answering `ranges`: its parts, or null when it is not cached (not counted). */
	private async matchWhole(key: string, ranges: BlobRange[], trace?: BlobTrace): Promise<RangePart[] | null> {
		const r = await this.cacheApi!.match(this.request(key, rangeHeader(ranges)));
		if (!r) return null;
		this.note(key);
		const body = new Uint8Array(await r.arrayBuffer());
		const { parts, kind } = responseParts(r.status, r.headers.get("content-type"), r.headers.get("content-range"), body);
		if (trace) {
			trace.hits++;
			trace.bytes += body.byteLength;
			if (kind === "full") trace.r200++;
			else trace.r206++;
		}
		return parts;
	}

	/**
	 * `pieces` (absolute, all inside block `i`) from a block fill in flight or the block's cache entry (one request,
	 * the pieces as its ranges), or null when the block has to be filled (counted as a miss).
	 */
	private async lookBlock(key: string, i: number, size: number, pieces: BlobRange[], trace?: BlobTrace): Promise<Uint8Array[] | null> {
		const start = i * this.blockBytes;
		const len = Math.min(this.blockBytes, size - start);
		const rel = pieces.map((p) => ({ offset: p.offset - start, length: p.length }));
		const pending = this.blockFills.get(this.blockId(key, i));
		if (pending) {
			const b = await pending;
			if (b) return rel.map((p) => copyOut(b, p.offset, p.offset + p.length));
		}
		const want = mergeRanges(rel);
		const whole = want.length === 1 && want[0].offset === 0 && want[0].length === len;
		const r = await this.cacheApi!.match(this.blockRequest(key, i, whole ? undefined : rangeHeader(want)));
		const body = r ? new Uint8Array(await r.arrayBuffer()) : null;
		const got = r && body ? responseParts(r.status, r.headers.get("content-type"), r.headers.get("content-range"), body) : null;
		// A block entry of another length is not this object's block (an entry written under another size): refill it.
		if (!got || !body || (got.kind === "full" && body.byteLength !== len)) {
			if (trace) trace.misses++;
			return null;
		}
		if (trace) {
			trace.hits++;
			trace.bytes += body.byteLength;
			if (!whole) {
				if (got.kind === "full") trace.r200++;
				else trace.r206++;
			}
		}
		return sliceRanges(got.parts, rel, key).map((b) => copyOut(b));
	}

	/**
	 * Block `i` of `key` from the origin (that block's range only), put in the cache through `waitUntil`. Concurrent
	 * callers share one read until the put settles; at most `blockBuffers` blocks are held at once and `blockReads`
	 * read at once. Null when the object does not exist.
	 */
	private fillBlock(key: string, i: number, size: number, trace?: BlobTrace): Promise<Uint8Array | null> {
		const id = this.blockId(key, i);
		const pending = this.blockFills.get(id);
		if (pending) return pending;
		const start = i * this.blockBytes;
		const len = Math.min(this.blockBytes, size - start);
		let release = () => {};
		let put: Promise<void> = Promise.resolve();
		const p: Promise<Uint8Array | null> = (async () => {
			release = await this.blockBuffers.take();
			const read = await this.blockReads.take();
			let block: Uint8Array | null;
			try {
				block = await this.origin_(() => this.inner.get(key, { offset: start, length: len }, trace ? { trace } : undefined));
			} finally {
				read();
			}
			if (block) {
				if (trace) trace.fills = (trace.fills ?? 0) + 1;
				this.blockStats.fills++;
				this.blockStats.bytes += block.byteLength;
				put = this.putBlock(key, i, block);
			}
			return block;
		})();
		// The fill stays registered and its slot held until the cache holds the block, so a concurrent reader shares
		// these bytes instead of reading the block again. Each reader copies its pieces out as soon as the fill
		// resolves: the readers' reactions to `p` are queued with this one and `await put` resumes after them, so once
		// the slot is given back, nothing the store returned holds the block.
		const done = () => {
			release();
			if (this.blockFills.get(id) === p) this.blockFills.delete(id);
			this.wakeIdle();
		};
		p.then(async () => {
			await put;
			done();
		}, done);
		this.blockFills.set(id, p);
		return p;
	}

	/** Puts a block in the cache (handed to `waitUntil`); never rejects. */
	private putBlock(key: string, i: number, block: Uint8Array): Promise<void> {
		// The block is this store's own read; readers copy their pieces out of it, so the body need not be a copy.
		const resp = new Response(block, {
			headers: { "content-type": "application/octet-stream", "content-length": String(block.byteLength), "cache-control": this.cacheControl, "accept-ranges": "bytes" },
		});
		const p = this.cacheApi!.put(this.blockRequest(key, i), resp).catch((e: unknown) => {
			this.blockStats.failed++;
			console.error(`thread-sdk: cache put of block ${i} of ${key} failed: ${e instanceof Error ? e.message : String(e)}`);
		});
		this.waitUntil?.(p);
		return p;
	}

	/**
	 * `ranges` of `key` (`size` bytes) through the block cache. Each range is cut into one piece per block it covers;
	 * each block is looked up once (its pieces as one range request), a block being filled is shared, and the blocks
	 * that missed come from the object's whole entry when it has one (an object cached whole stays served from it),
	 * else are filled from the origin. Null when the object does not exist.
	 */
	private async readBlocks(key: string, ranges: BlobRange[], size: number, trace?: BlobTrace): Promise<Uint8Array[] | null> {
		for (const r of ranges) if (r.offset < 0 || r.length < 0 || r.offset + r.length > size) throw new ThreadError(`${key} [${r.offset}, +${r.length}) is outside the object (${size} bytes)`);
		const nonEmpty = ranges.filter((r) => r.length > 0);
		const cut = (parts: RangePart[], rs: BlobRange[]) => rs.map((r) => (r.length === 0 ? new Uint8Array(0) : copyOut(sliceRanges(parts, [r], key)[0])));
		if (!nonEmpty.length) return ranges.map(() => new Uint8Array(0));
		let wholeTried = false;
		if (this.known.has(key)) {
			wholeTried = true;
			const parts = await this.matchWhole(key, mergeRanges(nonEmpty), trace);
			if (parts) return cut(parts, ranges);
		}
		const B = this.blockBytes;
		const pieces = new Map<number, BlobRange[]>();
		const plan = ranges.map((r) => {
			const refs: [block: number, piece: number][] = [];
			if (r.length === 0) return refs;
			const last = Math.floor((r.offset + r.length - 1) / B);
			for (let i = Math.floor(r.offset / B); i <= last; i++) {
				const a = Math.max(r.offset, i * B);
				const e = Math.min(r.offset + r.length, (i + 1) * B);
				let ps = pieces.get(i);
				if (!ps) pieces.set(i, (ps = []));
				refs.push([i, ps.length]);
				ps.push({ offset: a, length: e - a });
			}
			return refs;
		});
		const got = new Map<number, Uint8Array[]>();
		const missed: number[] = [];
		await Promise.all(
			[...pieces].map(async ([i, ps]) => {
				const parts = await this.lookBlock(key, i, size, ps, trace);
				if (parts) got.set(i, parts);
				else missed.push(i);
			}),
		);
		if (missed.length && !wholeTried) {
			const want = missed.flatMap((i) => pieces.get(i)!);
			const parts = await this.matchWhole(key, mergeRanges(want), trace);
			if (parts) {
				for (const i of missed) got.set(i, sliceRanges(parts, pieces.get(i)!, key).map((b) => copyOut(b)));
				missed.length = 0;
			}
		}
		let absent = false;
		await Promise.all(
			missed.map(async (i) => {
				const b = await this.fillBlock(key, i, size, trace);
				if (!b) {
					absent = true;
					return;
				}
				got.set(
					i,
					pieces.get(i)!.map((p) => copyOut(b, p.offset - i * B, p.offset - i * B + p.length)),
				);
			}),
		);
		if (absent) return null;
		return plan.map((refs, k) => {
			if (refs.length === 0) return new Uint8Array(0);
			if (refs.length === 1) return got.get(refs[0][0])![refs[0][1]];
			const out = new Uint8Array(ranges[k].length);
			let o = 0;
			for (const [i, j] of refs) {
				const b = got.get(i)![j];
				out.set(b, o);
				o += b.byteLength;
			}
			return out;
		});
	}

	async get(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		if (!this.cacheable(key)) return this.inner.get(key, range, opts);
		if (range && this.blocked(opts)) return (await this.readBlocks(key, [range], opts.size, opts.trace))?.[0] ?? null;
		const trace = opts?.trace;
		if (this.blobCache) {
			const got = range ? await this.blobCache.getRanges(this.scope + key, [range]) : await this.blobCache.get(this.scope + key).then((b) => (b ? [b] : null));
			if (got) {
				if (trace) {
					trace.hits++;
					trace.bytes += got[0].byteLength;
					if (range) trace.r206++;
				}
				// A seeded prefix served a fill read: fetch the whole slab in the background, as a miss would have.
				if (range && opts?.fill && this.missPolicy !== "range" && !this.blobCache.has(this.scope + key)) this.warm(key, trace);
				return got[0];
			}
		} else {
			const r = await this.cacheApi!.match(this.request(key, range && rangeHeader([range])));
			if (r) {
				this.note(key);
				const body = new Uint8Array(await r.arrayBuffer());
				if (trace) {
					trace.hits++;
					trace.bytes += body.byteLength;
				}
				if (!range) return body;
				if (r.status === 206) {
					if (trace) trace.r206++;
					if (body.byteLength !== range.length) throw new ThreadError(`cache range ${key} [${range.offset}, +${range.length}) returned ${body.byteLength} bytes`);
					return body;
				}
				if (trace) trace.r200++;
				if (body.byteLength < range.offset + range.length) throw new ThreadError(`cached ${key} is ${body.byteLength} bytes, need ${range.offset + range.length}`);
				// A fill read keeps the view (its reader takes the rest of the object from the buffer); any other read
				// copies its range out, so that it does not hold the whole object.
				return opts?.fill ? body.subarray(range.offset, range.offset + range.length) : copyOut(body, range.offset, range.offset + range.length);
			}
		}
		if (trace) trace.misses++;
		if (!range) return this.fetchWhole(key, opts, this.missPolicy !== "range");
		if (opts?.fill) return (await this.missFill(key, [range], true, opts))?.[0] ?? null;
		return this.origin_(() => this.inner.get(key, range, opts));
	}

	async getRanges(key: string, ranges: BlobRange[], opts?: GetOptions): Promise<Uint8Array[] | null> {
		if (!ranges.length) return [];
		if (!this.cacheable(key)) return this.inner.getRanges?.(key, ranges, opts) ?? null;
		if (this.blocked(opts)) return this.readBlocks(key, ranges, opts.size, opts.trace);
		const trace = opts?.trace;
		if (this.blobCache) {
			const got = await this.blobCache.getRanges(this.scope + key, ranges);
			if (got) {
				if (trace) {
					trace.hits++;
					trace.r206++;
					trace.bytes += got.reduce((a, b) => a + b.byteLength, 0);
				}
				if (opts?.fill && this.missPolicy !== "range" && !this.blobCache.has(this.scope + key)) this.warm(key, trace);
				return got;
			}
		} else {
			const parts = await this.matchWhole(key, ranges, trace);
			if (parts) {
				const out = sliceRanges(parts, ranges, key);
				return opts?.fill ? out : out.map((b) => copyOut(b));
			}
		}
		if (trace) trace.misses++;
		if (opts?.fill) return this.missFill(key, ranges, false, opts);
		return this.origin_(async () => (await this.inner.getRanges?.(key, ranges, opts)) ?? null);
	}

	/**
	 * The last `length` bytes and the object's size: served from an in-process cache that holds the whole object,
	 * else ONE origin request (a tail is not cached: a pack reader reads it once per open).
	 */
	get getTail(): BlobStore["getTail"] {
		const inner = this.inner;
		if (!inner.getTail) return undefined;
		return async (key, length, opts) => {
			const full = this.blobCache && this.cacheable(key) ? await this.blobCache.get(this.scope + key) : null;
			if (full) {
				const bytes = full.subarray(Math.max(0, full.byteLength - length));
				if (opts?.trace) {
					opts.trace.hits++;
					opts.trace.bytes += bytes.byteLength;
				}
				return { bytes, size: full.byteLength };
			}
			return this.origin_(() => inner.getTail!(key, length, opts));
		};
	}

	head(key: string): Promise<BlobHead | null> {
		return this.inner.head(key);
	}

	async put(key: string, data: Uint8Array | string, opts?: PutOptions): Promise<BlobHead | null> {
		if (!this.inner.put) throw new ThreadError("the underlying store is read-only", 400);
		return this.inner.put(key, data, opts);
	}

	async list(prefix: string): Promise<string[]> {
		if (!this.inner.list) throw new ThreadError("the underlying store cannot list", 400);
		return this.inner.list(prefix);
	}

	get listObjects(): BlobStore["listObjects"] {
		const inner = this.inner;
		return inner.listObjects ? (prefix) => inner.listObjects!(prefix) : undefined;
	}

	/** Deletes from the store and drops the cached copy (in-process; the Cache API entry when it can delete). */
	get delete(): BlobStore["delete"] {
		const inner = this.inner;
		if (!inner.delete) return undefined;
		return async (key) => {
			await inner.delete!(key);
			this.known.delete(key);
			this.blobCache?.delete?.(this.scope + key);
			if (this.cacheApi?.delete) await this.cacheApi.delete(this.request(key)).catch(() => false);
		};
	}
}

/** Sorts ranges and merges the ones that overlap or touch. */
function mergeRanges(ranges: readonly BlobRange[]): BlobRange[] {
	const out: BlobRange[] = [];
	for (const r of [...ranges].sort((a, b) => a.offset - b.offset)) {
		const last = out[out.length - 1];
		if (last && r.offset <= last.offset + last.length) last.length = Math.max(last.length, r.offset + r.length - last.offset);
		else out.push({ offset: r.offset, length: r.length });
	}
	return out;
}

/** A store with every key prefixed (one bucket holding several indexes, e.g. `tenants/a/`). */
export class PrefixedBlobStore implements BlobStore {
	readonly remote: boolean;
	readonly inflight: number | undefined;
	readonly coalesceBytes: number | undefined;
	readonly spanBytes: number | undefined;
	constructor(
		public readonly inner: BlobStore,
		public readonly prefix: string,
	) {
		this.remote = inner.remote ?? false;
		this.inflight = inner.inflight;
		this.coalesceBytes = inner.coalesceBytes;
		this.spanBytes = inner.spanBytes;
	}
	get(key: string, range?: BlobRange, opts?: GetOptions) {
		return this.inner.get(this.prefix + key, range, opts);
	}
	contains(key: string, range?: BlobRange) {
		return this.inner.contains?.(this.prefix + key, range);
	}
	warmup(n: number) {
		return this.inner.warmup?.(n) ?? Promise.resolve();
	}
	getRanges(key: string, ranges: BlobRange[], opts?: GetOptions) {
		return this.inner.getRanges?.(this.prefix + key, ranges, opts) ?? Promise.resolve(null);
	}
	head(key: string) {
		return this.inner.head(this.prefix + key);
	}
	get getTail(): BlobStore["getTail"] {
		const inner = this.inner;
		return inner.getTail ? (key, length, opts) => inner.getTail!(this.prefix + key, length, opts) : undefined;
	}
	put(key: string, data: Uint8Array | string, opts?: PutOptions) {
		if (!this.inner.put) throw new ThreadError("the underlying store is read-only", 400);
		return this.inner.put(this.prefix + key, data, opts);
	}
	async list(prefix: string) {
		if (!this.inner.list) throw new ThreadError("the underlying store cannot list", 400);
		return (await this.inner.list(this.prefix + prefix)).map((k) => k.slice(this.prefix.length));
	}
	get listObjects(): BlobStore["listObjects"] {
		const inner = this.inner;
		return inner.listObjects ? async (prefix) => (await inner.listObjects!(this.prefix + prefix)).map((o) => ({ ...o, key: o.key.slice(this.prefix.length) })) : undefined;
	}
	get delete(): BlobStore["delete"] {
		const inner = this.inner;
		return inner.delete ? (key) => inner.delete!(this.prefix + key) : undefined;
	}
}
