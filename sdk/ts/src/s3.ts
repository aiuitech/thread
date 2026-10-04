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
import { inWorkers, type BlobHead, type BlobRange, type BlobStore, type GetOptions, type PutOptions } from "./store.js";
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
	/** Phase-2 range merging defaults for queries (`QueryOptions.coalesceBytes` / `spanBytes`). */
	coalesceBytes?: number;
	spanBytes?: number;
}

/** Reads in flight per query against S3 directly: no connection cap outside Workers, 6 inside. */
export const S3_INFLIGHT = 32;
export const S3_INFLIGHT_WORKERS = 6;

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

const stripQuotes = (etag: string | null): string => (etag ?? "").replace(/^W\//, "").replace(/^"|"$/g, "");

export class S3BlobStore implements BlobStore {
	readonly endpoint: URL;
	readonly bucket: string;
	readonly region: string;
	readonly pathStyle: boolean;
	readonly remote = true;
	readonly inflight: number;
	readonly coalesceBytes: number | undefined;
	readonly spanBytes: number | undefined;
	readonly signingKeys: SigningKeyCache;
	private readonly fetchFn: typeof fetch;
	private readonly retries: number;
	private readonly now: () => number;
	constructor(private readonly opts: S3BlobStoreOptions) {
		this.inflight = opts.inflight ?? (inWorkers() ? S3_INFLIGHT_WORKERS : S3_INFLIGHT);
		this.coalesceBytes = opts.coalesceBytes;
		this.spanBytes = opts.spanBytes;
		this.signingKeys = new SigningKeyCache(opts.secretAccessKey);
		this.endpoint = new URL(opts.endpoint);
		this.bucket = opts.bucket;
		this.region = opts.region ?? "auto";
		this.pathStyle = opts.pathStyle ?? true;
		this.fetchFn = opts.fetch ?? ((input, init) => fetch(input, init));
		this.retries = opts.retries ?? 1;
		this.now = opts.now ?? Date.now;
		if (!opts.accessKeyId || !opts.secretAccessKey) throw new ThreadError("S3BlobStore needs accessKeyId and secretAccessKey", 400);
	}

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
					await r.body?.cancel();
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

	async get(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		const headers: Record<string, string> = range ? { range: `bytes=${range.offset}-${range.offset + range.length - 1}` } : {};
		const r = await this.send("GET", key, headers);
		if (r.status === 404) {
			await r.body?.cancel();
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
	 * Opens up to `n` connections ahead of the first query: `n` concurrent HEADs of `key` (default
	 * `idx/current.json`; a 404 is fine), so the TLS handshakes are not on the query's critical path. Errors are
	 * swallowed. Idle connections are kept by the HTTP client for its keep-alive timeout (undici: 4 s by default;
	 * see `nodeFetch` in `thread-sdk/node` for a longer one).
	 */
	async warmup(n: number, key = "idx/current.json"): Promise<void> {
		await Promise.all(Array.from({ length: Math.max(0, n) }, () => this.head(key).catch(() => null)));
	}

	async head(key: string): Promise<BlobHead | null> {
		const r = await this.send("HEAD", key, {});
		await r.body?.cancel();
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
		await r.body?.cancel();
		if (r.status === 412 || r.status === 409) return null;
		if (!r.ok) throw new ThreadError(`S3 PUT ${key}: HTTP ${r.status}`, 502);
		return { size: body.byteLength, etag: stripQuotes(r.headers.get("etag")) };
	}

	async list(prefix: string): Promise<string[]> {
		const out: string[] = [];
		let token: string | undefined;
		do {
			const query: Record<string, string> = { "list-type": "2", prefix, "max-keys": "1000" };
			if (token) query["continuation-token"] = token;
			const r = await this.send("GET", "", {}, undefined, query);
			if (!r.ok) return this.fail(r, `LIST ${prefix}`);
			const xml = await r.text();
			for (const m of xml.matchAll(/<Key>([^<]*)<\/Key>/g)) out.push(decodeXml(m[1]));
			const next = xml.match(/<NextContinuationToken>([^<]*)<\/NextContinuationToken>/);
			token = /<IsTruncated>true<\/IsTruncated>/.test(xml) && next ? decodeXml(next[1]) : undefined;
		} while (token);
		return out;
	}
}

const decodeXml = (s: string): string => s.replace(/&(lt|gt|amp|quot|apos|#(\d+));/g, (_, n: string, d?: string) => (d ? String.fromCodePoint(Number(d)) : ({ lt: "<", gt: ">", amp: "&", quot: '"', apos: "'" } as Record<string, string>)[n]));
