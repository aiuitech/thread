/**
 * Blob stores: the SDK reads index objects through the `BlobStore` interface, so the same query code runs
 * against an R2 binding (`R2BlobStore`), any S3-compatible endpoint (`S3BlobStore`, see s3.ts), memory
 * (`MemoryBlobStore`, tests and small local indexes) and, in front of any of them, the Workers Cache API
 * (`CachedBlobStore`: immutable objects cached for a year, served with single- and multi-range requests).
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
}

export const newTrace = (): BlobTrace => ({ reads: 0, hits: 0, misses: 0, r206: 0, r200: 0, bytes: 0 });

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
export const snapTrace = (t: BlobTrace): BlobTrace => ({ reads: t.reads, hits: t.hits, misses: t.misses, r206: t.r206, r200: t.r200, bytes: t.bytes });

/** The phase between two snapshots of a trace (`before` defaults to an empty trace). */
export function phaseStat(ms: number, after: BlobTrace, before: BlobTrace = newTrace()): PhaseStat {
	return { ms: roundMs(ms), reads: after.reads - before.reads, hits: after.hits - before.hits, misses: after.misses - before.misses, r206: after.r206 - before.r206, r200: after.r200 - before.r200, bytes: after.bytes - before.bytes };
}

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
}

export const rangeHeader = (ranges: BlobRange[]): string => "bytes=" + ranges.map((r) => `${r.offset}-${r.offset + r.length - 1}`).join(", ");

const toView = (data: Uint8Array | string): Uint8Array | string => (typeof data === "string" ? data : new Uint8Array(data.buffer, data.byteOffset, data.byteLength));

// ---------- R2 binding ----------

/** The subset of the Workers `R2Bucket` binding the SDK uses (structural, so no type dependency). */
export interface R2BucketLike {
	get(key: string, options?: { range?: BlobRange | { suffix: number } }): Promise<{ arrayBuffer(): Promise<ArrayBuffer>; etag: string; size: number } | null>;
	head(key: string): Promise<{ etag: string; size: number } | null>;
	put(key: string, value: ArrayBuffer | ArrayBufferView | string, options?: { onlyIf?: { etagMatches?: string; etagDoesNotMatch?: string }; httpMetadata?: { contentType?: string } }): Promise<{ etag: string; size: number } | null>;
	list(options?: { prefix?: string; cursor?: string; limit?: number }): Promise<{ objects: { key: string }[]; truncated: boolean; cursor?: string }>;
}

export class R2BlobStore implements BlobStore {
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

export interface CachedBlobStoreOptions {
	/** Synthetic origin the cache keys live under (any https URL; objects are keyed `${origin}${key}`). */
	origin?: string;
	/** Keeps background cache writes alive after the response (pass `ctx.waitUntil`). */
	waitUntil?: (p: Promise<unknown>) => void;
	/** Which keys are immutable and may be cached. Default: everything except `current.json`. */
	cacheable?: (key: string) => boolean;
	cacheControl?: string;
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
 * Cache API in front of another store. Index objects under `idx/` are immutable (every revision writes new
 * keys), so they are cached under a synthetic URL for a year and served with Range requests: the cache
 * answers a `Range` against a cached full object with a 206 (multipart for several ranges).
 */
export class CachedBlobStore implements BlobStore {
	readonly origin: string;
	private readonly waitUntil: ((p: Promise<unknown>) => void) | undefined;
	private readonly cacheable: (key: string) => boolean;
	private readonly cacheControl: string;

	constructor(
		public readonly inner: BlobStore,
		public readonly cache: CacheLike,
		opts: CachedBlobStoreOptions = {},
	) {
		this.origin = (opts.origin ?? "https://vib-sdk.cache/").replace(/\/?$/, "/");
		this.waitUntil = opts.waitUntil;
		this.cacheable = opts.cacheable ?? ((k) => !k.endsWith("current.json"));
		this.cacheControl = opts.cacheControl ?? "public, max-age=31536000, immutable";
	}

	request(key: string, range?: string): Request {
		return new Request(this.origin + key.replace(/^\/+/, ""), { method: "GET", headers: range ? { range } : {} });
	}

	/** Stores a full immutable object in the cache in the background. */
	store(key: string, bytes: Uint8Array): void {
		const resp = new Response(bytes.slice(), {
			headers: { "content-type": "application/octet-stream", "content-length": String(bytes.byteLength), "cache-control": this.cacheControl, "accept-ranges": "bytes" },
		});
		const p = this.cache.put(this.request(key), resp).catch((e: unknown) => console.error(`thread-sdk: cache.put ${key} failed: ${e instanceof Error ? e.message : String(e)}`));
		this.waitUntil?.(p);
	}

	private async fill(key: string, ranges: BlobRange[] | null, opts?: GetOptions): Promise<Uint8Array[] | null> {
		const full = await this.inner.get(key, undefined, opts);
		if (!full) return null;
		this.store(key, full);
		if (!ranges) return [full];
		for (const r of ranges) if (r.offset + r.length > full.byteLength) throw new ThreadError(`${key} is ${full.byteLength} bytes, need ${r.offset + r.length}`);
		return ranges.map((r) => full.subarray(r.offset, r.offset + r.length));
	}

	async get(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		if (!this.cacheable(key)) return this.inner.get(key, range, opts);
		const r = await this.cache.match(this.request(key, range && rangeHeader([range])));
		const trace = opts?.trace;
		if (r) {
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
		if (trace) trace.misses++;
		if (!range) return (await this.fill(key, null, opts))?.[0] ?? null;
		if (opts?.fill) return (await this.fill(key, [range], opts))?.[0] ?? null;
		return this.inner.get(key, range, opts);
	}

	async getRanges(key: string, ranges: BlobRange[], opts?: GetOptions): Promise<Uint8Array[] | null> {
		if (!ranges.length) return [];
		if (!this.cacheable(key)) return this.inner.getRanges?.(key, ranges, opts) ?? null;
		const r = await this.cache.match(this.request(key, rangeHeader(ranges)));
		const trace = opts?.trace;
		if (!r) {
			if (trace) trace.misses++;
			if (opts?.fill) return this.fill(key, ranges, opts);
			return this.inner.getRanges?.(key, ranges, opts) ?? null;
		}
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
	constructor(
		public readonly inner: BlobStore,
		public readonly prefix: string,
	) {}
	get(key: string, range?: BlobRange, opts?: GetOptions) {
		return this.inner.get(this.prefix + key, range, opts);
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
}
