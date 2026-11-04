/**
 * `S3BlobStore`: the BlobStore interface over any S3-compatible endpoint (R2's S3 API, AWS S3, MinIO, ...)
 * with plain `fetch` and AWS Signature V4 (WebCrypto HMAC-SHA256; no dependencies). Single-range GETs, HEAD,
 * PUT and conditional PUT (`If-None-Match: *` for create-only, `If-Match` for compare-and-swap) and
 * ListObjectsV2. S3 has no multi-range GET, so `getRanges` is not implemented: the SDK issues one request
 * per rerank range (put a `CachedBlobStore` in front for multi-range reads from the edge cache).
 *
 * Signing costs one SHA-256 and one HMAC per request: the derived signing key of each (date, region, service) is
 * imported once as an HMAC `CryptoKey` and reused (`SigningKeyCache`).
 */
import { inWorkers, type BlobHead, type BlobRange, type BlobStore, type GetOptions, type ObjectInfo, type PutOptions } from "./store.js";
import { ThreadError } from "./wire.js";

export interface S3BlobStoreOptions {
	/** `https://<account>.r2.cloudflarestorage.com`, `https://s3.us-east-1.amazonaws.com`, `http://localhost:9000`, ... */
	endpoint: string;
	bucket: string;
	/** Signing region: R2 uses `"auto"` (default). */
	region?: string;
	accessKeyId: string;
	secretAccessKey: string;
	/** STS/temporary credentials (R2 temporary access credentials). */
	sessionToken?: string;
	/** `https://endpoint/bucket/key` (default, works everywhere) vs `https://bucket.endpoint/key`. */
	pathStyle?: boolean;
	/** `fetch` to use (default: the global one). */
	fetch?: typeof fetch;
	/** Retries of a GET/HEAD after a network error or 5xx (default 1). */
	retries?: number;
	/** Clock override for tests (ms since epoch). */
	now?: () => number;
	/**
	 * Reads in flight per query (a query's `inflight` default): 32 in Node and other runtimes without a connection
	 * cap, 6 inside a Worker (the Workers limit of simultaneous open connections per request).
	 */
	inflight?: number;
	/** Phase-2 range merging defaults for queries (`QueryOptions.coalesceBytes`, default 128 KB / `spanBytes`). */
	coalesceBytes?: number;
	spanBytes?: number;
	/**
	 * Whole-object GETs of objects larger than this are split into parallel range GETs of this size (default 8 MiB;
	 * 0 = one GET). Speeds up the routing table (25 MB on g4) on a cold open.
	 */
	chunkBytes?: number;
	/** Parallel range GETs of one chunked object (default 4). */
	chunkConcurrency?: number;
	/**
	 * Open this many connections right away (`warmup(n)`), e.g. while the wasm compiles: create the store before
	 * `Thread.init`. In Node the first `fetch` also loads the lazy `fetch`/`Response` globals (13-18 ms) here
	 * instead of inside `Thread.init`. `open` then skips its own warm-up. Default 0.
	 */
	warmup?: number;
}

/** Reads in flight per query against S3 directly: no connection cap outside Workers, 6 inside. */
export const S3_INFLIGHT = 32;
export const S3_INFLIGHT_WORKERS = 6;
/**
 * Phase-2 range merging against S3 (no multi-range GET): a slab's wanted rerank records are read as runs of up to
 * 128 KB, which halves the requests of a cold query for ~45% more bytes (docs/cold-start.md).
 */
export const S3_COALESCE = 128 * 1024;

const enc = new TextEncoder();
const HEX = Array.from({ length: 256 }, (_, i) => i.toString(16).padStart(2, "0"));
const hex = (b: ArrayBuffer | Uint8Array): string => {
	const u = b instanceof Uint8Array ? b : new Uint8Array(b);
	let s = "";
	for (let i = 0; i < u.length; i++) s += HEX[u[i]];
	return s;
};
const sha256 = async (data: Uint8Array | string): Promise<string> => hex(await crypto.subtle.digest("SHA-256", typeof data === "string" ? enc.encode(data) : (data as Uint8Array<ArrayBuffer>)));
const importHmac = (key: Uint8Array | string): Promise<CryptoKey> => crypto.subtle.importKey("raw", typeof key === "string" ? enc.encode(key) : (key as Uint8Array<ArrayBuffer>), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
async function hmac(key: Uint8Array | string | CryptoKey, data: string): Promise<Uint8Array> {
	const k = typeof key === "string" || key instanceof Uint8Array ? await importHmac(key) : key;
	return new Uint8Array(await crypto.subtle.sign("HMAC", k, enc.encode(data)));
}

/** Derives the SigV4 signing key `kSigning` for (date, region, service) from a secret (4 HMACs). */
export async function deriveSigningKey(secretAccessKey: string, date: string, region: string, service: string): Promise<Uint8Array> {
	const kDate = await hmac("AWS4" + secretAccessKey, date);
	const kRegion = await hmac(kDate, region);
	const kService = await hmac(kRegion, service);
	return hmac(kService, "aws4_request");
}

/**
 * Derived signing keys of one secret by (date, region, service), imported once as HMAC `CryptoKey`s: a request then
 * costs one HMAC instead of five HMACs and five key imports. Holds the last few scopes (the date rolls daily).
 */
export class SigningKeyCache {
	private readonly keys = new Map<string, Promise<CryptoKey>>();
	/** Key derivations performed (tests, benchmarks). */
	derived = 0;
	constructor(private readonly secretAccessKey: string) {}
	get(date: string, region: string, service: string): Promise<CryptoKey> {
		const scope = `${date}/${region}/${service}`;
		let k = this.keys.get(scope);
		if (!k) {
			this.derived++;
			k = deriveSigningKey(this.secretAccessKey, date, region, service).then(importHmac);
			k.catch(() => this.keys.get(scope) === k && this.keys.delete(scope));
			if (this.keys.size >= 4) this.keys.delete(this.keys.keys().next().value!);
			this.keys.set(scope, k);
		}
		return k;
	}
}
/** RFC 3986 encoding (AWS style: `*` and `~` handling, `/` kept in paths). */
export const awsEncode = (s: string, keepSlash: boolean): string =>
	encodeURIComponent(s)
		.replace(/[!'()*]/g, (c) => "%" + c.charCodeAt(0).toString(16).toUpperCase())
		.replace(/%2F/g, keepSlash ? "/" : "%2F")
		.replace(/%7E/g, "~");

export const EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

export interface SigV4Input {
	method: string;
	/** Already-encoded canonical path (`/bucket/key` or `/key`). */
	path: string;
	/** Query parameters (unencoded). */
	query?: Record<string, string>;
	/** Extra headers to sign (lower-case names); `host`, `x-amz-date`, `x-amz-content-sha256` are added. */
	headers: Record<string, string>;
	payloadHash: string;
	host: string;
	region: string;
	service?: string;
	accessKeyId: string;
	secretAccessKey: string;
	sessionToken?: string;
	/** `YYYYMMDDTHHMMSSZ`. */
	amzDate: string;
}

export const amzDateOf = (ms: number): string => new Date(ms).toISOString().replace(/[:-]|\.\d{3}/g, "");

/**
 * Signs a request (AWS Signature Version 4, header auth). Returns the headers to send. With `keys` (a cache for
 * `i.secretAccessKey`) the derived signing key is reused across requests.
 */
export async function signV4(i: SigV4Input, keys?: SigningKeyCache): Promise<Record<string, string>> {
	const service = i.service ?? "s3";
	const date = i.amzDate.slice(0, 8);
	const headers: Record<string, string> = { ...i.headers, host: i.host, "x-amz-date": i.amzDate, "x-amz-content-sha256": i.payloadHash };
	if (i.sessionToken) headers["x-amz-security-token"] = i.sessionToken;
	const names = Object.keys(headers).sort();
	const canonicalHeaders = names.map((n) => `${n}:${headers[n].trim().replace(/\s+/g, " ")}\n`).join("");
	const signed = names.join(";");
	const query = Object.entries(i.query ?? {})
		.map(([k, v]) => [awsEncode(k, false), awsEncode(v, false)] as const)
		.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] < b[1] ? -1 : 1))
		.map(([k, v]) => `${k}=${v}`)
		.join("&");
	const canonical = [i.method, i.path, query, canonicalHeaders, signed, i.payloadHash].join("\n");
	const scope = `${date}/${i.region}/${service}/aws4_request`;
	const toSign = ["AWS4-HMAC-SHA256", i.amzDate, scope, await sha256(canonical)].join("\n");
	const kSigning = keys ? await keys.get(date, i.region, service) : await deriveSigningKey(i.secretAccessKey, date, i.region, service);
	const signature = hex(await hmac(kSigning, toSign));
	const out = { ...headers };
	delete out.host; // fetch sets it
	out.authorization = `AWS4-HMAC-SHA256 Credential=${i.accessKeyId}/${scope}, SignedHeaders=${signed}, Signature=${signature}`;
	return out;
}

/**
 * Finishes a response whose body we do not need (HEAD, 404, PUT, a 5xx before a retry) by reading it: these bodies
 * are empty or a few hundred bytes, and reading them keeps the keep-alive connection (cancelling the stream makes
 * Node's fetch close the socket, so the next request pays a new TCP + TLS handshake).
 */
async function settle(r: Response): Promise<void> {
	await r.arrayBuffer().catch(() => {});
}

const stripQuotes = (etag: string | null): string => (etag ?? "").replace(/^W\//, "").replace(/^"|"$/g, "");

export class S3BlobStore implements BlobStore {
	readonly endpoint: URL;
	readonly bucket: string;
	readonly region: string;
	readonly pathStyle: boolean;
	readonly remote = true;
	readonly inflight: number;
	readonly coalesceBytes: number;
	readonly spanBytes: number | undefined;
	readonly signingKeys: SigningKeyCache;
	readonly chunkBytes: number;
	readonly chunkConcurrency: number;
	private readonly fetchFn: typeof fetch;
	private readonly retries: number;
	private readonly now: () => number;
	constructor(private readonly opts: S3BlobStoreOptions) {
		this.inflight = opts.inflight ?? (inWorkers() ? S3_INFLIGHT_WORKERS : S3_INFLIGHT);
		this.coalesceBytes = opts.coalesceBytes ?? S3_COALESCE;
		this.spanBytes = opts.spanBytes;
		this.signingKeys = new SigningKeyCache(opts.secretAccessKey);
		this.chunkBytes = opts.chunkBytes ?? 8 * 1024 * 1024;
		this.chunkConcurrency = Math.max(1, opts.chunkConcurrency ?? 4);
		this.endpoint = new URL(opts.endpoint);
		this.bucket = opts.bucket;
		this.region = opts.region ?? "auto";
		this.pathStyle = opts.pathStyle ?? true;
		this.fetchFn = opts.fetch ?? ((input, init) => fetch(input, init));
		this.retries = opts.retries ?? 1;
		this.now = opts.now ?? Date.now;
		if (!opts.accessKeyId || !opts.secretAccessKey) throw new ThreadError("S3BlobStore needs accessKeyId and secretAccessKey", 400);
		if (opts.warmup) void this.warmup(opts.warmup);
	}

	private warming: { n: number; at: number; p: Promise<void> } | null = null;

	/** Host and canonical path of an object (or of the bucket when `key` is empty). */
	private target(key: string): { host: string; path: string } {
		const host = this.pathStyle ? this.endpoint.host : `${this.bucket}.${this.endpoint.host}`;
		const base = this.endpoint.pathname.replace(/\/$/, "");
		const path = this.pathStyle ? `${base}/${awsEncode(this.bucket, false)}${key ? "/" + awsEncode(key, true) : ""}` : `${base}/${key ? awsEncode(key, true) : ""}`;
		return { host, path: path || "/" };
	}

	private async send(method: string, key: string, headers: Record<string, string>, body?: Uint8Array, query?: Record<string, string>): Promise<Response> {
		const { host, path } = this.target(key);
		const payloadHash = body ? await sha256(body) : EMPTY_SHA256;
		const signedHeaders = await signV4({ method, path, query, headers, payloadHash, host, region: this.region, accessKeyId: this.opts.accessKeyId, secretAccessKey: this.opts.secretAccessKey, sessionToken: this.opts.sessionToken, amzDate: amzDateOf(this.now()) }, this.signingKeys);
		const qs = query ? "?" + Object.entries(query).map(([k, v]) => `${awsEncode(k, false)}=${awsEncode(v, false)}`).join("&") : "";
		const url = `${this.endpoint.protocol}//${host}${path}${qs}`;
		let lastErr: unknown;
		for (let attempt = 0; attempt <= (method === "PUT" ? 0 : this.retries); attempt++) {
			try {
				const r = await this.fetchFn(url, { method, headers: signedHeaders, body: body ? (body as Uint8Array<ArrayBuffer>) : undefined });
				if (r.status >= 500 && attempt < this.retries && method !== "PUT") {
					await settle(r);
					lastErr = new ThreadError(`S3 ${method} ${key}: HTTP ${r.status}`, r.status);
					continue;
				}
				return r;
			} catch (e) {
				lastErr = e;
			}
		}
		throw lastErr instanceof Error ? lastErr : new ThreadError(`S3 ${method} ${key} failed: ${String(lastErr)}`);
	}

	private async fail(r: Response, what: string): Promise<never> {
		const text = (await r.text().catch(() => "")).slice(0, 300);
		throw new ThreadError(`S3 ${what}: HTTP ${r.status} ${text}`, r.status === 403 ? 403 : 502);
	}

	get(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		if (this.chunkBytes > 0 && !range) return this.getChunked(key, opts);
		if (this.chunkBytes > 0 && range && range.length > 2 * this.chunkBytes) return this.getRangeChunked(key, range, opts);
		return this.getOne(key, range, opts);
	}

	/** A large range (a 25 MB routing table inside a pack) as parallel range GETs of `chunkBytes`. */
	private async getRangeChunked(key: string, range: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		const c = this.chunkBytes;
		const out = new Uint8Array(range.length);
		const todo: number[] = [];
		for (let off = 0; off < range.length; off += c) todo.push(off);
		let missing = false;
		const worker = async () => {
			for (let off = todo.shift(); off !== undefined && !missing; off = todo.shift()) {
				const b = await this.getOne(key, { offset: range.offset + off, length: Math.min(c, range.length - off) }, opts);
				if (!b) missing = true;
				else out.set(b, off);
			}
		};
		await Promise.all(Array.from({ length: Math.min(this.chunkConcurrency, todo.length) }, worker));
		return missing ? null : out;
	}

	/** One GET (of `range`, else of the whole object). */
	private async getOne(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		const headers: Record<string, string> = range ? { range: `bytes=${range.offset}-${range.offset + range.length - 1}` } : {};
		const r = await this.send("GET", key, headers);
		if (r.status === 404) {
			await settle(r);
			return null;
		}
		if (!r.ok) return this.fail(r, `GET ${key}`);
		let b = new Uint8Array(await r.arrayBuffer());
		if (opts?.trace) {
			opts.trace.reads++;
			opts.trace.bytes += b.byteLength;
		}
		if (range) {
			if (r.status === 200) b = b.subarray(range.offset, range.offset + range.length); // Range ignored
			if (b.byteLength !== range.length) throw new ThreadError(`${key} [${range.offset}, +${range.length}): got ${b.byteLength} bytes`);
		}
		return b;
	}

	/**
	 * Opens up to `n` connections ahead of the first query: `n` concurrent one-byte GETs of `key` (default
	 * `idx/current.json`; a 404 is fine), so the TLS handshakes are not on the query's critical path. Errors are
	 * swallowed. Idle connections are kept by the HTTP client for its keep-alive timeout (undici: 4 s by default;
	 * see `nodeFetch` in `thread-sdk/node` for a longer one).
	 */
	warmup(n: number, key = "idx/current.json"): Promise<void> {
		// A warm-up of at least n connections started in the last 3 s (e.g. at construction) covers this one.
		if (this.warming && this.warming.n >= n && Date.now() - this.warming.at < 3000) return this.warming.p;
		// A one-byte ranged GET, not a HEAD: Node's fetch does not return a connection to its pool after a HEAD answer
		// that carries a Content-Length, so a HEAD would warm nothing.
		const p = Promise.all(Array.from({ length: Math.max(0, n) }, () => this.get(key, { offset: 0, length: 1 }).catch(() => null))).then(() => {});
		this.warming = { n, at: Date.now(), p };
		return p;
	}

	/**
	 * A whole object as parallel range GETs: the first `chunkBytes`, whose Content-Range gives the size, and the rest
	 * in chunks, `chunkConcurrency` at a time beside the first. The rest starts as soon as the first response's HEADERS arrive,
	 * while its body still streams, so splitting costs no extra round trip in the critical path: an object just above
	 * `chunkBytes` (an 8.8 MB base pack) takes as long as one GET, a 25 MB routing table ~1/3 of it. One S3 stream is
	 * bandwidth-bound (~380 Mbps from R2 to one client, 200 Mbps per response in the proxy model).
	 */
	private async getChunked(key: string, opts?: GetOptions): Promise<Uint8Array | null> {
		const c = this.chunkBytes;
		const r = await this.send("GET", key, { range: `bytes=0-${c - 1}` });
		if (r.status === 404) {
			await settle(r);
			return null;
		}
		if (r.status === 416) {
			await settle(r); // an empty object: R2 refuses any range of it
			return this.getOne(key, undefined, opts);
		}
		if (!r.ok) return this.fail(r, `GET ${key}`);
		const count = (b: number) => {
			if (opts?.trace) {
				opts.trace.reads++;
				opts.trace.bytes += b;
			}
		};
		if (r.status !== 206) {
			// Range ignored: the whole object in one response
			const whole = new Uint8Array(await r.arrayBuffer());
			count(whole.byteLength);
			return whole;
		}
		const total = Number((r.headers.get("content-range") ?? "").match(/\/(\d+)$/)?.[1] ?? NaN);
		if (!Number.isFinite(total)) {
			await settle(r);
			throw new ThreadError(`S3 GET ${key}: 206 without a usable Content-Range`);
		}
		const firstLen = Math.min(c, total);
		const out = total > firstLen ? new Uint8Array(total) : null;
		let rest: Promise<void> = Promise.resolve();
		if (out) {
			const todo: number[] = [];
			for (let off = firstLen; off < total; off += c) todo.push(off);
			const worker = async () => {
				for (let off = todo.shift(); off !== undefined; off = todo.shift()) {
					const len = Math.min(c, total - off);
					const b = await this.getOne(key, { offset: off, length: len });
					if (!b) throw new ThreadError(`${key} disappeared while being read`, 404);
					count(b.byteLength);
					out.set(b, off);
				}
			};
			rest = Promise.all(Array.from({ length: Math.min(this.chunkConcurrency, todo.length) }, worker)).then(() => {});
			rest.catch(() => {}); // awaited below; never an unhandled rejection meanwhile
		}
		const first = new Uint8Array(await r.arrayBuffer());
		count(first.byteLength);
		if (first.byteLength !== firstLen) throw new ThreadError(`${key} [0, +${firstLen}): got ${first.byteLength} bytes`);
		if (!out) return first;
		out.set(first, 0);
		await rest;
		return out;
	}

	/** One GET with `Range: bytes=-length`; the size comes from `Content-Range` (or the body of a 200). */
	async getTail(key: string, length: number, opts?: GetOptions): Promise<{ bytes: Uint8Array; size: number; etag?: string } | null> {
		const r = await this.send("GET", key, { range: `bytes=-${Math.max(1, length)}` });
		if (r.status === 404) {
			await settle(r);
			return null;
		}
		if (r.status === 416) {
			// an empty object has no last byte
			await settle(r);
			return { bytes: new Uint8Array(0), size: 0 };
		}
		if (!r.ok) return this.fail(r, `GET ${key} (tail)`);
		let bytes = new Uint8Array(await r.arrayBuffer());
		if (opts?.trace) {
			opts.trace.reads++;
			opts.trace.bytes += bytes.byteLength;
		}
		if (r.status === 200) {
			// Range ignored: the whole object
			const size = bytes.byteLength;
			bytes = bytes.subarray(Math.max(0, size - length));
			return { bytes, size, etag: stripQuotes(r.headers.get("etag")) };
		}
		const m = /\/(\d+)\s*$/.exec(r.headers.get("content-range") ?? "");
		if (!m) throw new ThreadError(`S3 GET ${key} (tail): 206 without a Content-Range size`, 502);
		return { bytes, size: Number(m[1]), etag: stripQuotes(r.headers.get("etag")) };
	}

	async head(key: string): Promise<BlobHead | null> {
		const r = await this.send("HEAD", key, {});
		await settle(r);
		if (r.status === 404) return null;
		if (!r.ok) throw new ThreadError(`S3 HEAD ${key}: HTTP ${r.status}`, 502);
		return { size: Number(r.headers.get("content-length") ?? 0), etag: stripQuotes(r.headers.get("etag")) };
	}

	/**
	 * Conditional puts use `If-None-Match: *` (create-only) and `If-Match: "<etag>"`; a 412 (or 409) means the
	 * condition failed. R2's S3 API and AWS S3 support both on PutObject; a store that ignores them cannot
	 * serialize writers (use the R2 binding or a Durable Object for that).
	 */
	async put(key: string, data: Uint8Array | string, opts: PutOptions = {}): Promise<BlobHead | null> {
		const body = typeof data === "string" ? enc.encode(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		const headers: Record<string, string> = { "content-type": opts.contentType ?? "application/octet-stream", "content-length": String(body.byteLength) };
		if (opts.ifAbsent) headers["if-none-match"] = "*";
		if (opts.ifMatch !== undefined) headers["if-match"] = `"${stripQuotes(opts.ifMatch)}"`;
		const r = await this.send("PUT", key, headers, body);
		await settle(r);
		if (r.status === 412 || r.status === 409) return null;
		if (!r.ok) throw new ThreadError(`S3 PUT ${key}: HTTP ${r.status}`, 502);
		return { size: body.byteLength, etag: stripQuotes(r.headers.get("etag")) };
	}

	async list(prefix: string): Promise<string[]> {
		return (await this.listObjects(prefix)).map((o) => o.key);
	}

	/** ListObjectsV2 pages: key, size and LastModified of each object. */
	async listObjects(prefix: string): Promise<ObjectInfo[]> {
		const out: ObjectInfo[] = [];
		let token: string | undefined;
		do {
			const query: Record<string, string> = { "list-type": "2", prefix, "max-keys": "1000" };
			if (token) query["continuation-token"] = token;
			const r = await this.send("GET", "", {}, undefined, query);
			if (!r.ok) return this.fail(r, `LIST ${prefix}`);
			const xml = await r.text();
			for (const c of xml.matchAll(/<Contents>([\s\S]*?)<\/Contents>/g)) {
				const key = c[1].match(/<Key>([^<]*)<\/Key>/);
				if (!key) continue;
				const size = c[1].match(/<Size>(\d+)<\/Size>/);
				const mod = c[1].match(/<LastModified>([^<]*)<\/LastModified>/);
				out.push({ key: decodeXml(key[1]), size: size ? Number(size[1]) : 0, uploaded: mod ? Date.parse(mod[1]) : 0 });
			}
			const next = xml.match(/<NextContinuationToken>([^<]*)<\/NextContinuationToken>/);
			token = /<IsTruncated>true<\/IsTruncated>/.test(xml) && next ? decodeXml(next[1]) : undefined;
		} while (token);
		return out;
	}

	async delete(key: string): Promise<void> {
		const r = await this.send("DELETE", key, {});
		await settle(r);
		if (!r.ok && r.status !== 404) throw new ThreadError(`S3 DELETE ${key}: HTTP ${r.status}`, 502);
	}
}

const decodeXml = (s: string): string => s.replace(/&(lt|gt|amp|quot|apos|#(\d+));/g, (_, n: string, d?: string) => (d ? String.fromCodePoint(Number(d)) : ({ lt: "<", gt: ">", amp: "&", quot: '"', apos: "'" } as Record<string, string>)[n]));
