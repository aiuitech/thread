/**
 * vib-bench entry point: query serving plus the build orchestrator.
 *
 * Everything except the forwarding below is src/index.ts (`app`): queries, upserts/deletes and every build stage.
 * Heavy work does not run in the isolates that serve queries. It runs in dynamic Workers (Worker Loader binding),
 * one per *slot*: `/build/<stage>?slot=N` and the write routes are forwarded to the dynamic Worker
 * `thread-build:<bundle hash>:slot-<N>`, which runs the same code (a second bundle of src/index.ts, see vite.config.ts)
 * in its own isolate with its own 128 MB and `limits.cpuMs` 300,000. Each slot id is one isolate: send at most one
 * job at a time per slot (bench/driver.mjs keeps a free list of slot ids). Concurrent requests to one slot share its
 * 128 MB, which is exactly the failure the old vib-bench-0..7 pool existed to avoid.
 *
 *   /build/<stage>?slot=<id>   forwarded to slot <id> (default `admin`); `?slot=local` runs it in this isolate
 *   /upsert, /delete           forwarded to slot `writer` (or `?slot=<id>`) so DeltaJob's ~50 MB of wasm memory
 *                              never lands in a query isolate (wasm memory never shrinks); `?slot=local` runs here
 *   /embed                     embeds texts through the generation's model (manifest.embedding): this isolate
 *   everything else            this isolate (app.fetch)
 *
 * Embedding: a generation created for a model (`POST /create {embedding}`, thread-sdk's `createGeneration`) names it
 * in `manifest.embedding`. A JSON `/query {text}` without `vector` and a JSON `/upsert {texts}` without `vectors`
 * are embedded HERE, before forwarding (bindings cannot be passed into a dynamic Worker's env, and a query isolate
 * has the AI binding): Workers AI through `env.AI`, an OpenAI-compatible endpoint with `env.EMBED_API_KEY`.
 *
 * Slot ids in use: driver and bench/compact.mjs jobs `0`..`N-1` (THREAD_SLOTS), compact.mjs commit/plan `admin`,
 * writes `writer`, automatic compaction slices `auto-0`..`auto-3` (dispatched from the writer through SlotRpc).
 *
 * R2 cannot be passed into a dynamic Worker's env (DataCloneError), so slots get `BUCKET_RPC` (BucketRpc below,
 * a loopback WorkerEntrypoint of this Worker) and the shim (SLOT_SHIM) presents an R2Bucket-shaped `env.BUCKET` on
 * top of it. Bodies stream in both directions: GET bodies are R2's own streams, PUT bodies arrive as a byte stream
 * plus length and go to R2 through a FixedLengthStream, so this isolate never holds a whole object.
 */
import { WorkerEntrypoint } from "cloudflare:workers";
import wasmModule from "../../../crates/thread-wasm/pkg/thread_bg.wasm";
import stageBundle from "virtual:thread-stage-bundle";
import app from "./index.ts";
import { normalizeRows, resolveEmbedder, validateEmbeddingConfig, type EmbedKind, type EmbeddingConfig } from "../../../sdk/ts/src/embed.ts";
import { ThreadError } from "../../../sdk/ts/src/wire.ts";

interface Env {
	BUCKET: R2Bucket;
	BUILD_TOKEN: string;
	LOADER: WorkerLoader;
	/** Workers AI (for generations whose `embedding.provider` is `workers-ai`). */
	AI?: Ai;
	/** API key for `embedding.provider: "openai"` endpoints ("-" or unset = none). */
	EMBED_API_KEY?: string;
}

const COMPAT_DATE = "2026-10-01";
/** Dynamic Workers default to ~30 s of CPU, and isolate startup counts: always set the limit explicitly. */
const SLOT_CPU_MS = 300_000;
const SLOT_RE = /^[a-z0-9][a-z0-9-]{0,31}$/;

/** Exports of this module as loopback bindings (ctx.exports; typed loosely: the generated types do not list them). */
type Loopback = { BucketRpc(o: { props: object }): unknown; SlotRpc(o: { props: object }): unknown };
const loopback = (ctx: ExecutionContext): Loopback => (ctx as unknown as { exports: Loopback }).exports;

function meta(o: R2Object) {
	return {
		key: o.key,
		version: o.version,
		size: o.size,
		etag: o.etag,
		httpEtag: o.httpEtag,
		uploaded: o.uploaded,
		httpMetadata: o.httpMetadata,
		customMetadata: o.customMetadata,
		range: o.range,
		storageClass: o.storageClass,
	};
}

/**
 * R2 for the slots: every R2Bucket method the stages use (get with range, head, put incl. onlyIf, list, delete).
 * put takes a byte stream and its length (FixedLengthStream gives R2 the length up front), a string, or null.
 */
export class BucketRpc extends WorkerEntrypoint<Env> {
	async get(key: string, opts?: R2GetOptions) {
		const o = await this.env.BUCKET.get(key, opts);
		if (!o) return null;
		// With onlyIf a failed precondition returns an R2Object without a body.
		return { meta: meta(o), body: "body" in o ? (o as R2ObjectBody).body : null };
	}
	async head(key: string) {
		const o = await this.env.BUCKET.head(key);
		return o ? meta(o) : null;
	}
	async put(key: string, value: ReadableStream | string | null, length: number | null, opts?: R2PutOptions) {
		if (!(value instanceof ReadableStream)) {
			const o = await this.env.BUCKET.put(key, value, opts);
			return o ? meta(o) : null;
		}
		if (length === null || !Number.isInteger(length) || length < 0) throw new Error(`put ${key}: a stream needs its byte length`);
		const fls = new FixedLengthStream(length);
		const abort = new AbortController();
		// A conditional put whose precondition fails may never read the body: stop the pipe once R2 has answered.
		const pipe = value.pipeTo(fls.writable, { signal: abort.signal }).catch((e) => e);
		try {
			const o = await this.env.BUCKET.put(key, fls.readable, opts);
			return o ? meta(o) : null;
		} finally {
			abort.abort();
			await pipe;
		}
	}
	async list(opts?: R2ListOptions) {
		const r = await this.env.BUCKET.list(opts);
		return { objects: r.objects.map(meta), truncated: r.truncated, cursor: r.truncated ? r.cursor : undefined, delimitedPrefixes: r.delimitedPrefixes };
	}
	async delete(keys: string | string[]) {
		await this.env.BUCKET.delete(keys);
	}
}

/**
 * Entry module of every slot: the stage bundle's handler with an R2Bucket-shaped env.BUCKET over BUCKET_RPC.
 * Responses carry the slot's isolate id and its in-flight count (x-slot-iso, x-slot-inflight: >1 means two jobs
 * shared the slot's 128 MB).
 */
const SLOT_SHIM = `
import inner from "./bundle.js";
let ISO = "";
let inflight = 0, served = 0;

function wrapObject(meta, body) {
	if (!meta) return null;
	let used = false;
	const take = () => {
		if (used) throw new TypeError("body already used");
		used = true;
		return new Response(body);
	};
	return {
		...meta,
		body,
		get bodyUsed() { return used; },
		arrayBuffer: () => take().arrayBuffer(),
		bytes: async () => new Uint8Array(await take().arrayBuffer()),
		text: () => take().text(),
		json: () => take().json(),
		blob: () => take().blob(),
		writeHttpMetadata(h) {
			for (const [k, v] of Object.entries(meta.httpMetadata ?? {})) if (typeof v === "string") h.set(k.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase()), v);
		},
	};
}

/** Small bodies go over RPC as a copy; larger ones as a byte stream (the orchestrator never buffers them). */
const STREAM_MIN = 256 * 1024;

function wrapBucket(rpc) {
	return {
		async get(key, opts) {
			const r = await rpc.get(key, opts);
			return r ? wrapObject(r.meta, r.body) : null;
		},
		async head(key) {
			return wrapObject(await rpc.head(key), null);
		},
		async put(key, value, opts) {
			if (value === null || value === undefined || typeof value === "string") return wrapObject(await rpc.put(key, value ?? null, null, opts), null);
			let bytes;
			if (value instanceof ArrayBuffer) bytes = new Uint8Array(value);
			else if (ArrayBuffer.isView(value)) bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
			else if (value instanceof Blob) bytes = new Uint8Array(await value.arrayBuffer());
			else if (value instanceof ReadableStream) bytes = new Uint8Array(await new Response(value).arrayBuffer());
			else throw new TypeError("unsupported R2 put body");
			if (bytes.byteLength < STREAM_MIN) return wrapObject(await rpc.put(key, bytes.slice().buffer, null, opts), null);
			return wrapObject(await rpc.put(key, new Response(bytes).body, bytes.byteLength, opts), null);
		},
		async list(opts) {
			return await rpc.list(opts);
		},
		async delete(keys) {
			await rpc.delete(keys);
		},
	};
}

export default {
	async fetch(req, env, ctx) {
		ISO ||= crypto.randomUUID().slice(0, 8);
		inflight++; served++;
		const inflightAtStart = inflight;
		try {
			const resp = await inner.fetch(req, { ...env, BUCKET: wrapBucket(env.BUCKET_RPC) }, ctx);
			const out = new Response(resp.body, resp);
			out.headers.set("x-slot-iso", ISO);
			out.headers.set("x-slot-inflight", String(inflightAtStart));
			out.headers.set("x-slot-served", String(served));
			return out;
		} finally {
			inflight--;
		}
	},
};
`;

/** The dynamic Worker of `slot` (created on first use; the loader caches it by id, the id includes the bundle hash). */
function slotStub(env: Env, ctx: ExecutionContext, slot: string, loaded?: { cold: boolean }): WorkerStub {
	return env.LOADER.get(`thread-build:${stageBundle.hash}:slot-${slot}`, () => {
		if (loaded) loaded.cold = true;
		const exp = loopback(ctx);
		return {
			compatibilityDate: COMPAT_DATE,
			mainModule: "shim.js",
			modules: { "shim.js": SLOT_SHIM, "bundle.js": stageBundle.source, [stageBundle.wasmPath]: wasmModule },
			env: { BUILD_TOKEN: env.BUILD_TOKEN, THREAD_SLOT: slot, BUCKET_RPC: exp.BucketRpc({ props: {} }), SLOTS: exp.SlotRpc({ props: {} }) },
			limits: { cpuMs: SLOT_CPU_MS },
		} as WorkerLoaderWorkerCode;
	});
}

const json = (o: unknown, status = 200) => new Response(JSON.stringify(o), { status, headers: { "content-type": "application/json" } });
const errText = (e: unknown) => (e instanceof Error ? `${e.name}: ${e.message}` : String(e));

/**
 * Lets code running in a slot (the writer's automatic compaction) run stages in other slots:
 * `env.SLOTS.stage(slot, stage, body)` -> {status, data} (data = the stage's JSON response).
 */
export class SlotRpc extends WorkerEntrypoint<Env> {
	async stage(slot: string, stage: string, body: unknown): Promise<{ status: number; data: any }> {
		if (!SLOT_RE.test(slot) || slot === "local") throw new Error(`bad slot ${slot}`);
		try {
			const req = new Request(`https://thread-slot/build/${stage}`, {
				method: "POST",
				headers: { authorization: `Bearer ${this.env.BUILD_TOKEN}`, "content-type": "application/json" },
				body: JSON.stringify(body),
			});
			const resp = await slotStub(this.env, this.ctx, slot).getEntrypoint().fetch(req);
			const text = await resp.text();
			try {
				return { status: resp.status, data: JSON.parse(text) };
			} catch {
				return { status: resp.status === 200 ? 502 : resp.status, data: { error: `slot ${slot}: non-JSON response: ${text.slice(0, 300)}` } };
			}
		} catch (e) {
			return { status: 502, data: { error: `slot ${slot} failed: ${errText(e)}` } };
		}
	}
}

// ---------- embedding (manifest.embedding) ----------

/** How long the (gen -> embedding config) answer of /manifest is reused. */
const EMBEDDING_TTL_MS = 30_000;
const embeddingCache = new Map<string, { at: number; gen: string; dim: number; embedding: EmbeddingConfig | null }>();
/** Texts per /embed or embedded /upsert call (the upsert batch limit). */
const MAX_EMBED_TEXTS = 4096;

/** The embedding block of the head revision of `gen` (default: the current generation), through the app's /manifest. */
async function embeddingOf(env: Env, ctx: ExecutionContext, gen: string | null): Promise<{ gen: string; dim: number; embedding: EmbeddingConfig | null }> {
	const key = gen ?? "";
	const hit = embeddingCache.get(key);
	if (hit && Date.now() - hit.at < EMBEDDING_TTL_MS) return hit;
	const url = new URL("https://thread-local/manifest");
	if (gen) url.searchParams.set("gen", gen);
	const resp = await app.fetch(new Request(url, { headers: { authorization: `Bearer ${env.BUILD_TOKEN}` } }), { ...env, SLOTS: loopback(ctx).SlotRpc({ props: {} }) } as any, ctx);
	const j = (await resp.json()) as { gen?: string; manifest?: { dim?: number; embedding?: unknown }; error?: string };
	if (!resp.ok || !j.gen || !j.manifest) throw new HttpError(resp.status === 200 ? 500 : resp.status, j.error ?? "no manifest");
	const embedding = j.manifest.embedding ? validateEmbeddingConfig(j.manifest.embedding, j.manifest.dim) : null;
	const out = { at: Date.now(), gen: j.gen, dim: j.manifest.dim ?? 0, embedding };
	embeddingCache.set(key, out);
	return out;
}

/** Embeds `texts` as `kind` through the model of `gen`'s head revision: one unit-normalized row per text. */
async function embedTexts(env: Env, ctx: ExecutionContext, gen: string | null, texts: string[], kind: EmbedKind): Promise<{ gen: string; embedding: EmbeddingConfig; rows: Float32Array[] }> {
	if (!Array.isArray(texts) || !texts.length || !texts.every((t) => typeof t === "string")) throw new HttpError(400, "texts must be a non-empty array of strings");
	if (texts.length > MAX_EMBED_TEXTS) throw new HttpError(413, `at most ${MAX_EMBED_TEXTS} texts per call`);
	const m = await embeddingOf(env, ctx, gen);
	if (!m.embedding) throw new HttpError(400, `generation ${m.gen} names no embedding model: pass vectors`);
	const apiKey = env.EMBED_API_KEY && env.EMBED_API_KEY !== "-" ? env.EMBED_API_KEY : undefined;
	let embed: ReturnType<typeof resolveEmbedder>;
	try {
		embed = resolveEmbedder(m.embedding, { ai: env.AI, apiKey });
	} catch (e) {
		throw new HttpError(400, e instanceof Error ? e.message : String(e));
	}
	try {
		const rows = normalizeRows(await embed(texts, kind), `${m.embedding.provider} ${m.embedding.model}`);
		return { gen: m.gen, embedding: m.embedding, rows };
	} catch (e) {
		if (e instanceof ThreadError) throw new HttpError(e.status, e.message);
		throw e;
	}
}

class HttpError extends Error {
	constructor(
		public readonly status: number,
		message: string,
	) {
		super(message);
	}
}

/**
 * A JSON /query with `text` and no `vector`, or a JSON /upsert with `texts` and no `vectors`, with the texts embedded
 * through the generation's model; any other request as is. The body is only parsed when it looks like one of those
 * (a 25 MB vectors upsert is not parsed here).
 */
async function withEmbeddedBody(request: Request, env: Env, ctx: ExecutionContext, url: URL): Promise<Request> {
	if (request.method !== "POST" || (url.pathname !== "/query" && url.pathname !== "/upsert")) return request;
	if (!(request.headers.get("content-type") ?? "").includes("json")) return request;
	// The body is read from a clone, so the original request is returned untouched when nothing is embedded.
	const text = await request.clone().text();
	const query = url.pathname === "/query";
	const field = query ? '"text"' : '"texts"';
	const have = query ? '"vector"' : '"vectors"';
	if (!text.includes(field) || text.includes(have)) return request;
	let body: Record<string, unknown>;
	try {
		body = JSON.parse(text) as Record<string, unknown>;
	} catch {
		return request;
	}
	if (!body || typeof body !== "object") return request;
	const gen = (typeof body.gen === "string" ? body.gen : url.searchParams.get("gen")) || null;
	const rewritten = (b: Record<string, unknown>) => {
		const headers = new Headers(request.headers);
		headers.delete("content-length");
		return new Request(request.url, { method: "POST", headers, body: JSON.stringify(b) });
	};
	if (query) {
		if (typeof body.text !== "string" || body.vector !== undefined) return request;
		const { rows } = await embedTexts(env, ctx, gen, [body.text], "query");
		const { text: _t, ...rest } = body;
		return rewritten({ ...rest, vector: Array.from(rows[0]) });
	}
	if (!Array.isArray(body.texts) || body.vectors !== undefined) return request;
	const { rows } = await embedTexts(env, ctx, gen, body.texts as string[], "document");
	const flat = new Float32Array(rows.length * (rows[0]?.length ?? 0));
	rows.forEach((r, i) => flat.set(r, i * r.length));
	const { texts: _x, ...rest } = body;
	return rewritten({ ...rest, vectors: base64Of(new Uint8Array(flat.buffer)) });
}

/** Base64 of bytes without spreading them into one call (a 4096 x 1536 f32 batch is 25 MB). */
function base64Of(bytes: Uint8Array): string {
	let s = "";
	for (let i = 0; i < bytes.length; i += 8192) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192) as unknown as number[]);
	return btoa(s);
}

/** POST /embed {gen?, texts, kind?: "query" | "document"}: the texts' unit vectors under the generation's model. */
async function handleEmbed(request: Request, env: Env, ctx: ExecutionContext, url: URL): Promise<Response> {
	const t0 = Date.now();
	let body: { gen?: string; texts?: unknown; kind?: unknown };
	try {
		body = (await request.json()) as typeof body;
	} catch {
		throw new HttpError(400, "invalid JSON body");
	}
	const kind = body.kind ?? "query";
	if (kind !== "query" && kind !== "document") throw new HttpError(400, 'kind must be "query" or "document"');
	const gen = body.gen ?? url.searchParams.get("gen");
	const { gen: g, embedding, rows } = await embedTexts(env, ctx, gen, body.texts as string[], kind);
	return json({ gen: g, model: embedding.model, provider: embedding.provider, dim: embedding.dim, kind, vectors: rows.map((r) => Array.from(r)), ms: Date.now() - t0 });
}

function authorized(request: Request, env: Env): boolean {
	const token: unknown = env.BUILD_TOKEN;
	if (typeof token !== "string" || token === "") return false;
	const enc = new TextEncoder();
	const got = enc.encode(request.headers.get("authorization") ?? "");
	const want = enc.encode(`Bearer ${token}`);
	return got.byteLength === want.byteLength && crypto.subtle.timingSafeEqual(got, want);
}

/** Which slot a request runs in: "local" = this isolate. */
function slotOf(url: URL, method: string): string {
	const p = url.pathname;
	const def = p.startsWith("/build/") && method === "POST" ? "admin" : (p === "/upsert" || p === "/delete") && method === "POST" ? "writer" : null;
	if (def === null) return "local";
	const slot = url.searchParams.get("slot") ?? def;
	if (slot !== "local" && !SLOT_RE.test(slot)) throw new Error(`bad slot ${slot} (lowercase letters, digits and -, or local)`);
	return slot;
}

export default {
	async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
		const url = new URL(request.url);
		let slot: string;
		try {
			slot = slotOf(url, request.method);
		} catch (e) {
			return json({ error: errText(e) }, 400);
		}
		// Embedding and local execution: the app gets SLOTS too, so a ?slot=local write can still spread its
		// compaction over slots. Texts are embedded in this isolate before a request is forwarded anywhere.
		try {
			if (url.pathname === "/embed" && request.method === "POST") {
				if (!authorized(request, env)) return json({ error: "unauthorized" }, 401);
				return await handleEmbed(request, env, ctx, url);
			}
			// /query runs here, /upsert in the writer slot: both get their texts embedded first (never unauthenticated).
			if (request.method === "POST" && (url.pathname === "/query" || url.pathname === "/upsert")) {
				if (!authorized(request, env)) return json({ error: "unauthorized" }, 401);
				request = await withEmbeddedBody(request, env, ctx, url);
			}
			if (slot === "local") return app.fetch(request, { ...env, SLOTS: loopback(ctx).SlotRpc({ props: {} }) } as any, ctx);
			// Do not start dynamic Workers for unauthenticated requests.
			if (!authorized(request, env)) return json({ error: "unauthorized" }, 401);
		} catch (e) {
			if (e instanceof HttpError) return json({ error: e.message }, e.status);
			return json({ error: errText(e) }, 500);
		}
		const t0 = Date.now();
		const loaded = { cold: false };
		try {
			const stub = slotStub(env, ctx, slot, loaded);
			url.searchParams.delete("slot");
			const resp = await stub.getEntrypoint().fetch(new Request(url, request));
			const out = new Response(resp.body, resp);
			out.headers.set("x-slot", slot);
			out.headers.set("x-slot-cold", String(loaded.cold));
			out.headers.set("x-orch-ms", String(Date.now() - t0));
			out.headers.set("x-bundle", stageBundle.hash);
			return out;
		} catch (e) {
			// e.g. "Worker exceeded memory limit" or the CPU limit: the driver retries.
			return json({ error: `slot ${slot} failed: ${errText(e)}`, slot, ms: Date.now() - t0, cold: loaded.cold }, 502);
		}
	},
} satisfies ExportedHandler<Env>;
