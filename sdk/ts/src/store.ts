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
}

export const newTrace = (): BlobTrace => ({ reads: 0, hits: 0, misses: 0, r206: 0, r200: 0, bytes: 0, warms: 0 });

/** Whether this runs inside the Cloudflare Workers runtime (6 simultaneous open connections per request). */
export const inWorkers = (): boolean => (globalThis as { navigator?: { userAgent?: string } }).navigator?.userAgent === "Cloudflare-Workers";

export interface GetOptions {
	trace?: BlobTrace;
	/**
	 * Cache stores only: on a miss of a range read, fetch the WHOLE object, cache it and return the range as a
	 * view into it (the caller can recover the full object from the view's buffer). Default false: a miss reads
	 * just the range from the origin and caches nothing.
	 */
	fill?: boolean;
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
 * `byteOffset === 0` or `buffer.byteLength === byteLength`.
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
	/** Writes an object; `null` when a put condition failed. Optional (read-only stores). */
	put?(key: string, data: Uint8Array | string, opts?: PutOptions): Promise<BlobHead | null>;
	/** Keys under `prefix`. Optional (used to find a generation's head revision and by writers). */
	list?(prefix: string): Promise<string[]>;
	/** Whether reads cross a network (`open` puts an in-process cache in front of remote stores outside Workers). */
	readonly remote?: boolean;
	/** Preferred reads in flight per query (a query's `inflight` default; 12 when absent). */
	readonly inflight?: number;
	/** Preferred phase-2 range merging (a query's `coalesceBytes` / `spanBytes` defaults). */
	readonly coalesceBytes?: number;
	readonly spanBytes?: number;
	/**
	 * Whether `key` can be read without an origin request (in a local cache, or a local store): `undefined` when the
	 * store cannot tell. Used by the cold-path query policy. Optional.
	 */
	contains?(key: string): boolean | undefined;
	/** Opens `n` connections ahead of the first query (cheap HEADs); remote stores only. Optional. */
	warmup?(n: number): Promise<void>;
}

export const rangeHeader = (ranges: BlobRange[]): string => "bytes=" + ranges.map((r) => `${r.offset}-${r.offset + r.length - 1}`).join(", ");

const toView = (data: Uint8Array | string): Uint8Array | string => (typeof data === "string" ? data : new Uint8Array(data.buffer, data.byteOffset, data.byteLength));

// ---------- R2 binding ----------

/** The subset of the Workers `R2Bucket` binding the SDK uses (structural, so no type dependency). */
export interface R2BucketLike {
	get(key: string, options?: { range?: BlobRange }): Promise<{ arrayBuffer(): Promise<ArrayBuffer>; etag: string; size: number } | null>;
	head(key: string): Promise<{ etag: string; size: number } | null>;
	put(key: string, value: ArrayBuffer | ArrayBufferView | string, options?: { onlyIf?: { etagMatches?: string; etagDoesNotMatch?: string }; httpMetadata?: { contentType?: string } }): Promise<{ etag: string; size: number } | null>;
	list(options?: { prefix?: string; cursor?: string; limit?: number }): Promise<{ objects: { key: string }[]; truncated: boolean; cursor?: string }>;
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
}

// ---------- in-memory store (tests, tiny local indexes) ----------

export class MemoryBlobStore implements BlobStore {
	readonly objects = new Map<string, { bytes: Uint8Array; etag: string }>();
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

	async put(key: string, data: Uint8Array | string, opts: PutOptions = {}): Promise<BlobHead | null> {
		const cur = this.objects.get(key);
		if (opts.ifAbsent && cur) return null;
		if (opts.ifMatch !== undefined && cur?.etag !== opts.ifMatch) return null;
		const bytes = typeof data === "string" ? new TextEncoder().encode(data) : data.slice();
		const o = { bytes, etag: `m${++this.seq}` };
		this.objects.set(key, o);
		return { size: bytes.byteLength, etag: o.etag };
	}

	async list(prefix: string): Promise<string[]> {
		return [...this.objects.keys()].filter((k) => k.startsWith(prefix)).sort();
	}
}

// ---------- Cache API layer ----------

/** The part of the Workers `Cache` interface the SDK uses (`caches.default` or a named cache). */
export interface CacheLike {
	match(request: Request): Promise<Response | undefined>;
	put(request: Request, response: Response): Promise<void>;
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
 * are cached under every policy but `"range"`.
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
	/** Whether the object is cached (no I/O). */
	has(key: string): boolean;
	/** The miss policy a `CachedBlobStore` over this cache uses by default (`"range+warm"` when absent). */
	readonly defaultMissPolicy?: MissPolicy;
}

const ownBuffer = (b: Uint8Array): Uint8Array => (b.byteOffset === 0 && b.buffer.byteLength === b.byteLength ? b : b.slice());

/**
 * An LRU of whole objects in memory (default 256 MB; objects larger than a quarter of it are not kept). Hits are
 * views into the cached object, so a slab prefix served from it carries the slab's rerank records too.
 */
export class MemoryCache implements BlobCache {
	held = 0;
	private readonly items = new Map<string, Uint8Array>();
	constructor(readonly maxBytes = 256 * 1024 * 1024) {}

	has(key: string): boolean {
		return this.items.has(key);
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
		if (!b) return null;
		return ranges.map((r) => {
			if (r.offset < 0 || r.length < 0 || r.offset + r.length > b.byteLength) throw new ThreadError(`cached ${key} is ${b.byteLength} bytes, need [${r.offset}, +${r.length})`);
			return b.subarray(r.offset, r.offset + r.length);
		});
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

export interface CachedBlobStoreOptions {
	/** Synthetic origin the cache keys live under (any https URL; objects are keyed `${origin}${key}`). Cache API only. */
	origin?: string;
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
	}

	request(key: string, range?: string): Request {
		return new Request(this.origin + key.replace(/^\/+/, ""), { method: "GET", headers: range ? { range } : {} });
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
		} else p = this.blobCache!.put(key, bytes).then(() => {});
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
		const p = (async () => {
			const full = await this.origin_(() => this.inner.get(key, undefined, opts?.trace ? { trace: opts.trace } : undefined));
			// The fill stays registered until the cache holds the object, so a concurrent miss shares it.
			if (full && cache) await this.store(key, full);
			return full;
		})();
		this.fills.set(key, p);
		const done = () => {
			if (this.fills.get(key) === p) this.fills.delete(key);
			this.wakeIdle();
		};
		p.then(done, done);
		return p;
	}

	private static slice(key: string, full: Uint8Array, ranges: BlobRange[]): Uint8Array[] {
		for (const r of ranges) if (r.offset + r.length > full.byteLength) throw new ThreadError(`${key} is ${full.byteLength} bytes, need ${r.offset + r.length}`);
		return ranges.map((r) => full.subarray(r.offset, r.offset + r.length));
	}

	/** Queues a background whole-object fetch of `key` (`"range+warm"`). */
	private warm(key: string, trace?: BlobTrace): void {
		if (this.warmConcurrency <= 0 || this.warming.has(key) || this.fills.has(key) || this.blobCache?.has(key)) return;
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
				if (this.blobCache?.has(key)) return;
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
		if (this.warmActive || this.warmQ.length || this.fills.size) return;
		const w = this.idleWaiters;
		this.idleWaiters = [];
		for (const fn of w) fn();
	}

	/** Resolves once no background warm fetch or whole-object fill is queued or running (tests, benchmarks). */
	async drain(): Promise<void> {
		while (this.warmActive || this.warmQ.length || this.fills.size) await new Promise<void>((r) => this.idleWaiters.push(r));
	}

	contains(key: string): boolean | undefined {
		if (!this.cacheable(key)) return this.inner.contains?.(key);
		if (this.blobCache) return this.blobCache.has(key) || this.inner.contains?.(key) || false;
		return this.known.has(key) ? true : this.inner.contains?.(key);
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

	async get(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		if (!this.cacheable(key)) return this.inner.get(key, range, opts);
		const trace = opts?.trace;
		if (this.blobCache) {
			const got = range ? await this.blobCache.getRanges(key, [range]) : await this.blobCache.get(key).then((b) => (b ? [b] : null));
			if (got) {
				if (trace) {
					trace.hits++;
					trace.bytes += got[0].byteLength;
					if (range) trace.r206++;
				}
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
				return body.subarray(range.offset, range.offset + range.length);
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
		const trace = opts?.trace;
		if (this.blobCache) {
			const got = await this.blobCache.getRanges(key, ranges);
			if (got) {
				if (trace) {
					trace.hits++;
					trace.r206++;
					trace.bytes += got.reduce((a, b) => a + b.byteLength, 0);
				}
				return got;
			}
		} else {
			const r = await this.cacheApi!.match(this.request(key, rangeHeader(ranges)));
			if (r) {
				this.note(key);
				const body = new Uint8Array(await r.arrayBuffer());
				const { parts, kind } = responseParts(r.status, r.headers.get("content-type"), r.headers.get("content-range"), body);
				if (trace) {
					trace.hits++;
					trace.bytes += body.byteLength;
					if (kind === "full") trace.r200++;
					else trace.r206++;
				}
				return sliceRanges(parts, ranges, key);
			}
		}
		if (trace) trace.misses++;
		if (opts?.fill) return this.missFill(key, ranges, false, opts);
		return this.origin_(async () => (await this.inner.getRanges?.(key, ranges, opts)) ?? null);
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
	contains(key: string) {
		return this.inner.contains?.(this.prefix + key);
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
	put(key: string, data: Uint8Array | string, opts?: PutOptions) {
		if (!this.inner.put) throw new ThreadError("the underlying store is read-only", 400);
		return this.inner.put(this.prefix + key, data, opts);
	}
	async list(prefix: string) {
		if (!this.inner.list) throw new ThreadError("the underlying store cannot list", 400);
		return (await this.inner.list(this.prefix + prefix)).map((k) => k.slice(this.prefix.length));
	}
}
