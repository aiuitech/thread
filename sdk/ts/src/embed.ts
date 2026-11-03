/**
 * Embedding models: the `embedding` block of a manifest names the model a dense generation was built for, and
 * `resolveEmbedder` turns it into a function the index calls for query texts (`search({text})` without a vector)
 * and document texts (`upsert({texts})` without vectors). The index never runs a model itself: `workers-ai` calls
 * the Workers AI binding (or its REST API outside Workers), `openai` any OpenAI-compatible `/embeddings` endpoint,
 * and `custom` whatever function the caller passes to `open({ embed })`. The config shape, the prefixing rule and
 * the batching are shared with the Python SDK through `sdk/ts/test/fixtures/embedding.json`.
 */
import { ThreadError } from "./wire.js";

/** Providers the SDKs call themselves; any other `provider` needs a caller-supplied embedder. */
export const EMBEDDING_PROVIDERS = ["workers-ai", "openai", "custom"] as const;
export type EmbeddingProvider = (typeof EMBEDDING_PROVIDERS)[number];

/** Texts per provider call when the config does not say (Workers AI accepts 100 strings per request). */
export const EMBED_BATCH_DEFAULTS: Record<string, number> = { "workers-ai": 100, openai: 2048, custom: 256 };

/**
 * `manifest.embedding`: the embedding model of a dense generation. Every vector in the generation must come from
 * this model with these prefixes and options; `dim` must equal `manifest.dim`.
 */
export interface EmbeddingConfig {
	/** `"workers-ai"`, `"openai"` (any OpenAI-compatible endpoint) or `"custom"` (an embedder given to `open`). */
	provider: EmbeddingProvider;
	/** Model name as the provider takes it: `@cf/baai/bge-base-en-v1.5`, `text-embedding-3-small`, ... */
	model: string;
	/** Vector size the model produces (with `options`, e.g. OpenAI's `dimensions`); equals `manifest.dim`. */
	dim: number;
	/** Prepended to query texts (e5-style instructions: `"query: "`, `"Represent this sentence ..."`); default none. */
	queryPrefix?: string;
	/** Prepended to document texts; default none. */
	docPrefix?: string;
	/** `openai`: the API base (default `https://api.openai.com/v1`); `workers-ai` REST: the API base (default Cloudflare's). */
	baseUrl?: string;
	/**
	 * Provider request fields sent with every call, verbatim: `{pooling: "cls"}` for the Workers AI bge models (cls
	 * and mean embeddings are not compatible), `{dimensions: 256}` for OpenAI's text-embedding-3 models.
	 */
	options?: Record<string, unknown>;
	/** Texts per provider call (default per provider: workers-ai 100, openai 2048, custom 256). */
	maxBatch?: number;
}

/** What a text is embedded as: a query (search) or a document (upsert). Picks the prefix. */
export type EmbedKind = "query" | "document";

/**
 * Embeds texts of one kind: returns one vector per text, as `upsert({vectors})` takes them (`n * dim` floats flat or
 * one row per text). The SDK normalizes.
 */
export type TextEmbedder = (texts: string[], kind: EmbedKind) => Promise<Float32Array | number[] | (Float32Array | number[])[]> | Float32Array | number[] | (Float32Array | number[])[];

/** The Workers AI binding (`env.AI`), as much of it as the embedder uses. */
export interface AiLike {
	run(model: string, inputs: Record<string, unknown>): Promise<unknown>;
}

/** Credentials and transports for `resolveEmbedder`; what `open({ embed })` takes besides a function. */
export interface EmbedOptions {
	/** The Workers AI binding (inside a Worker). */
	ai?: AiLike;
	/** Workers AI through its REST API (outside Workers): the account and an API token with Workers AI access. */
	accountId?: string;
	apiToken?: string;
	/** `openai`: the API key (`Authorization: Bearer`). */
	apiKey?: string;
	/** Extra request headers for HTTP providers. */
	headers?: Record<string, string>;
	/** `fetch` to use (default the global one). */
	fetch?: typeof fetch;
	/** The embedder for `provider: "custom"` (or to override any provider). */
	embedder?: TextEmbedder;
}

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

/**
 * Checks an `embedding` block and returns it with unknown fields dropped. `dim` (the generation's) must match
 * `config.dim` when given.
 */
export function validateEmbeddingConfig(c: unknown, dim?: number): EmbeddingConfig {
	if (!isObj(c)) throw new ThreadError("embedding must be an object {provider, model, dim, ...}", 400);
	const provider = c.provider;
	if (typeof provider !== "string" || !provider) throw new ThreadError("embedding.provider must be a non-empty string", 400);
	if (!(EMBEDDING_PROVIDERS as readonly string[]).includes(provider)) throw new ThreadError(`embedding.provider must be one of ${EMBEDDING_PROVIDERS.join(", ")}`, 400);
	if (typeof c.model !== "string" || !c.model) throw new ThreadError("embedding.model must be a non-empty string", 400);
	if (!Number.isInteger(c.dim) || (c.dim as number) < 1) throw new ThreadError("embedding.dim must be an integer >= 1", 400);
	if (dim !== undefined && c.dim !== dim) throw new ThreadError(`embedding.dim ${String(c.dim)} does not match the generation's dim ${dim}`, 400);
	for (const f of ["queryPrefix", "docPrefix", "baseUrl"] as const) if (c[f] !== undefined && typeof c[f] !== "string") throw new ThreadError(`embedding.${f} must be a string`, 400);
	if (c.options !== undefined && !isObj(c.options)) throw new ThreadError("embedding.options must be an object", 400);
	if (c.maxBatch !== undefined && (!Number.isInteger(c.maxBatch) || (c.maxBatch as number) < 1)) throw new ThreadError("embedding.maxBatch must be an integer >= 1", 400);
	const out: EmbeddingConfig = { provider: provider as EmbeddingProvider, model: c.model, dim: c.dim as number };
	if (c.queryPrefix) out.queryPrefix = c.queryPrefix as string;
	if (c.docPrefix) out.docPrefix = c.docPrefix as string;
	if (c.baseUrl) out.baseUrl = c.baseUrl as string;
	if (c.options && Object.keys(c.options).length) out.options = { ...(c.options as Record<string, unknown>) };
	if (c.maxBatch !== undefined) out.maxBatch = c.maxBatch as number;
	return out;
}

/** The strings sent to the model for `texts` of `kind`: the kind's prefix in front of each text. */
export function embeddingInputs(c: EmbeddingConfig, texts: readonly string[], kind: EmbedKind): string[] {
	const prefix = (kind === "query" ? c.queryPrefix : c.docPrefix) ?? "";
	return texts.map((t) => prefix + t);
}

/** Texts per provider call for a config. */
export const embeddingBatchSize = (c: EmbeddingConfig): number => c.maxBatch ?? EMBED_BATCH_DEFAULTS[c.provider] ?? 256;

/** Sizes of the provider calls that embed `n` texts under `c`. */
export function embeddingBatches(c: EmbeddingConfig, n: number): number[] {
	const size = embeddingBatchSize(c);
	const out: number[] = [];
	for (let i = 0; i < n; i += size) out.push(Math.min(size, n - i));
	return out;
}

/** Rows of a provider response (`data: number[][]`, or OpenAI's `data: [{embedding, index}]`) checked against `dim`. */
function rowsOf(data: unknown, n: number, dim: number, what: string): Float32Array[] {
	if (!Array.isArray(data) || data.length !== n) throw new ThreadError(`${what} returned ${Array.isArray(data) ? data.length : "no"} embeddings for ${n} texts`, 502);
	const rows: Float32Array[] = new Array(n);
	data.forEach((row, i) => {
		let v: unknown = row;
		let at = i;
		if (isObj(row)) {
			v = row.embedding;
			if (Number.isInteger(row.index)) at = row.index as number;
		}
		if (!Array.isArray(v) || v.length !== dim) throw new ThreadError(`${what} returned a vector of ${Array.isArray(v) ? v.length : "no"} dims, expected ${dim}`, 502);
		if (at < 0 || at >= n || rows[at]) throw new ThreadError(`${what} returned a bad embedding index ${at}`, 502);
		rows[at] = Float32Array.from(v as number[]);
	});
	for (let i = 0; i < n; i++) if (!rows[i]) throw new ThreadError(`${what} returned no embedding for text ${i}`, 502);
	return rows;
}

async function readError(resp: Response): Promise<string> {
	const text = await resp.text().catch(() => "");
	return text.slice(0, 300);
}

/**
 * The embedder for a config: `provider: "workers-ai"` runs `opts.ai` (the binding) or, with `accountId` and
 * `apiToken`, the REST API; `"openai"` POSTs to `{baseUrl}/embeddings` with `apiKey`; `"custom"` (or any provider
 * with `opts.embedder`) calls `opts.embedder`. The returned function takes the texts of one kind, prefixes them,
 * calls the provider in batches of `maxBatch` and returns one `Float32Array` row per text (not yet normalized).
 */
export function resolveEmbedder(c: EmbeddingConfig, opts: EmbedOptions = {}): (texts: readonly string[], kind: EmbedKind) => Promise<Float32Array[]> {
	const custom = opts.embedder;
	let call: (inputs: string[]) => Promise<Float32Array[]>;
	if (custom) {
		call = async (inputs) => {
			const v = await custom(inputs, "document");
			return splitRows(v, inputs.length, c.dim, "the custom embedder");
		};
	} else if (c.provider === "workers-ai") {
		const ai = opts.ai;
		if (ai) {
			call = async (inputs) => {
				const r = (await ai.run(c.model, { ...(c.options ?? {}), text: inputs })) as { data?: unknown } | null;
				return rowsOf(r?.data, inputs.length, c.dim, `Workers AI ${c.model}`);
			};
		} else {
			if (!opts.accountId || !opts.apiToken) throw new ThreadError(`embedding model ${c.model}: pass the Workers AI binding (embed: {ai: env.AI}) or an accountId and apiToken`, 400);
			const base = (c.baseUrl ?? "https://api.cloudflare.com/client/v4").replace(/\/+$/, "");
			const url = `${base}/accounts/${opts.accountId}/ai/run/${c.model}`;
			const f = opts.fetch ?? fetch;
			call = async (inputs) => {
				const resp = await f(url, { method: "POST", headers: { "content-type": "application/json", authorization: `Bearer ${opts.apiToken}`, ...(opts.headers ?? {}) }, body: JSON.stringify({ ...(c.options ?? {}), text: inputs }) });
				if (!resp.ok) throw new ThreadError(`Workers AI ${c.model}: HTTP ${resp.status}: ${await readError(resp)}`, 502);
				const j = (await resp.json()) as { result?: { data?: unknown }; errors?: unknown };
				return rowsOf(j.result?.data, inputs.length, c.dim, `Workers AI ${c.model}`);
			};
		}
	} else if (c.provider === "openai") {
		const base = (c.baseUrl ?? "https://api.openai.com/v1").replace(/\/+$/, "");
		const f = opts.fetch ?? fetch;
		call = async (inputs) => {
			const headers: Record<string, string> = { "content-type": "application/json", ...(opts.headers ?? {}) };
			if (opts.apiKey) headers.authorization = `Bearer ${opts.apiKey}`;
			const resp = await f(`${base}/embeddings`, { method: "POST", headers, body: JSON.stringify({ ...(c.options ?? {}), model: c.model, input: inputs }) });
			if (!resp.ok) throw new ThreadError(`${c.model} at ${base}: HTTP ${resp.status}: ${await readError(resp)}`, 502);
			const j = (await resp.json()) as { data?: unknown };
			return rowsOf(j.data, inputs.length, c.dim, `${c.model} at ${base}`);
		};
	} else throw new ThreadError(`embedding provider ${c.provider} needs an embedder: open({ embed: (texts, kind) => ... })`, 400);
	const size = embeddingBatchSize(c);
	return async (texts, kind) => {
		const inputs = embeddingInputs(c, texts, kind);
		if (custom) {
			// A custom embedder sees the kind itself (it may apply its own instructions): one call per batch, kind passed through.
			const out: Float32Array[] = [];
			for (let i = 0; i < inputs.length; i += size) {
				const batch = inputs.slice(i, i + size);
				out.push(...splitRows(await custom(batch, kind), batch.length, c.dim, "the custom embedder"));
			}
			return out;
		}
		const out: Float32Array[] = [];
		for (let i = 0; i < inputs.length; i += size) out.push(...(await call(inputs.slice(i, i + size))));
		return out;
	};
}

/** One `Float32Array` row per text from what an embedder returned (flat `n * dim` or one row per text). */
export function splitRows(v: Float32Array | number[] | (Float32Array | number[])[], n: number, dim: number, what: string): Float32Array[] {
	if (Array.isArray(v) && v.length && typeof v[0] !== "number") {
		const rows = v as (Float32Array | number[])[];
		if (rows.length !== n) throw new ThreadError(`${what} returned ${rows.length} vectors for ${n} texts`, 400);
		return rows.map((r, i) => {
			if (r.length !== dim) throw new ThreadError(`${what} returned vector ${i} with ${r.length} dims, expected ${dim}`, 400);
			return r instanceof Float32Array ? r : Float32Array.from(r);
		});
	}
	const flat = v instanceof Float32Array ? v : Float32Array.from(v as number[]);
	if (flat.length !== n * dim) throw new ThreadError(`${what} returned ${flat.length} floats for ${n} texts of ${dim} dims`, 400);
	const out: Float32Array[] = new Array(n);
	for (let i = 0; i < n; i++) out[i] = flat.slice(i * dim, (i + 1) * dim);
	return out;
}

/** Unit-normalizes rows in place (a zero or non-finite row is an error). */
export function normalizeRows(rows: Float32Array[], what: string): Float32Array[] {
	rows.forEach((r, i) => {
		let ss = 0;
		for (const x of r) ss += x * x;
		if (!(ss > 0) || !Number.isFinite(ss)) throw new ThreadError(`${what}: vector ${i} has zero or non-finite norm`, 400);
		const inv = 1 / Math.sqrt(ss);
		for (let j = 0; j < r.length; j++) r[j] *= inv;
	});
	return rows;
}
