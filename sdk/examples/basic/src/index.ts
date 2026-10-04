/**
 * Example Worker for vib-sdk: a Vectorize-like `POST /search` over a vib index in R2.
 *
 *   POST /search            JSON {vector: number[], topK?, nprobe?, rerank?, pc?, returnPayload?, store?: "r2" | "s3"}
 *                           or an f32 LE body (content-type application/octet-stream) with the options as query
 *                           parameters (?topK=10&nprobe=34&rerank=150&store=r2)
 *   GET  /health            {ok, gen, rev}
 *   GET  /manifest          the manifest revision the index reads (?store=s3 for the S3 handle)
 *   POST /refresh           re-read current.json now (?store=)
 *
 * The wasm module is imported through the bundler (the Cloudflare Vite plugin turns the .wasm into a
 * WebAssembly.Module) and handed to the SDK once per isolate; one VibIndex per store is kept across requests.
 */
import wasmModule from "vib-sdk/wasm/vib_bg.wasm";
import { R2BlobStore, S3BlobStore, Vib, VibError, type BlobStore, type QueryOptions, type VibIndex } from "vib-sdk";

interface Env {
	BUCKET: R2Bucket;
	AUTH_TOKEN: string;
	S3_ENDPOINT: string;
	S3_BUCKET: string;
	S3_ACCESS_KEY_ID: string;
	S3_SECRET_ACCESS_KEY: string;
}

type StoreKind = "r2" | "s3";

let vibPromise: Promise<Vib> | null = null;
const indexes = new Map<StoreKind, Promise<VibIndex>>();

function getVib(): Promise<Vib> {
	if (!vibPromise) {
		vibPromise = Vib.init({ wasmModule });
		vibPromise.catch(() => (vibPromise = null));
	}
	return vibPromise;
}

const configured = (v: string | undefined): v is string => !!v && v !== "-";

function makeStore(env: Env, kind: StoreKind): BlobStore {
	if (kind === "r2") return new R2BlobStore(env.BUCKET);
	if (!configured(env.S3_ENDPOINT) || !configured(env.S3_ACCESS_KEY_ID) || !configured(env.S3_SECRET_ACCESS_KEY)) throw new VibError("the S3 store is not configured (S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY)", 400);
	return new S3BlobStore({ endpoint: env.S3_ENDPOINT, bucket: configured(env.S3_BUCKET) ? env.S3_BUCKET : "vib-bench", region: "auto", accessKeyId: env.S3_ACCESS_KEY_ID, secretAccessKey: env.S3_SECRET_ACCESS_KEY });
}

/** The index for a store kind, opened once per isolate (its routing table lives in wasm memory). */
function getIndex(env: Env, ctx: ExecutionContext, kind: StoreKind): Promise<VibIndex> {
	let p = indexes.get(kind);
	if (!p) {
		p = getVib().then((vib) =>
			vib.open({
				store: makeStore(env, kind),
				// Cache API in front of the store (default caches.default); keep each store's cache keys apart.
				cacheOrigin: `https://vib-sdk-example.cache/${kind}/`,
				waitUntil: (pr) => ctx.waitUntil(pr),
			}),
		);
		indexes.set(kind, p);
		p.catch(() => indexes.get(kind) === p && indexes.delete(kind));
	}
	return p;
}

const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });

function intParam(url: URL, name: string): number | undefined {
	const v = url.searchParams.get(name);
	if (v === null || v === "") return undefined;
	const n = Number(v);
	if (!Number.isInteger(n) || n < 0) throw new VibError(`${name} must be a non-negative integer`, 400);
	return n;
}

interface SearchBody {
	vector?: number[];
	topK?: number;
	nprobe?: number;
	rerank?: number;
	pc?: number;
	returnPayload?: boolean;
	store?: StoreKind;
}

async function handleSearch(request: Request, env: Env, ctx: ExecutionContext, url: URL): Promise<Response> {
	let vector: Float32Array;
	let opts: QueryOptions;
	let kind: StoreKind = (url.searchParams.get("store") as StoreKind | null) ?? "r2";
	if ((request.headers.get("content-type") ?? "").includes("octet-stream")) {
		vector = new Float32Array(await request.arrayBuffer());
		opts = { topK: intParam(url, "topK") ?? intParam(url, "k"), nprobe: intParam(url, "nprobe"), rerank: intParam(url, "rerank"), pc: intParam(url, "pc"), returnPayload: url.searchParams.get("returnPayload") === "1" };
	} else {
		const body = (await request.json().catch(() => null)) as SearchBody | null;
		if (!body || !Array.isArray(body.vector)) throw new VibError("body must be JSON {vector: number[], ...}", 400);
		vector = Float32Array.from(body.vector);
		opts = { topK: body.topK, nprobe: body.nprobe, rerank: body.rerank, pc: body.pc, returnPayload: body.returnPayload };
		if (body.store) kind = body.store;
	}
	if (kind !== "r2" && kind !== "s3") throw new VibError("store must be r2 or s3", 400);
	const t0 = Date.now();
	const index = await getIndex(env, ctx, kind);
	const tOpen = Date.now();
	const res = await index.query(vector, opts);
	const dec = new TextDecoder();
	return json({
		store: kind,
		gen: index.gen,
		rev: index.rev,
		ids: res.hits.map((h) => h.id),
		scores: res.hits.map((h) => h.score),
		hits: res.hits.map((h) => ({ id: h.id, score: h.score, attrs: h.attrs, source: h.source, ...(h.payload ? { payload: dec.decode(h.payload) } : {}) })),
		params: res.params,
		stats: res.stats,
		timing: { request: Date.now() - t0, open: tOpen - t0, ...res.stats.timing },
	});
}

export default {
	async fetch(request, env, ctx): Promise<Response> {
		const url = new URL(request.url);
		try {
			if (url.pathname === "/health") {
				const ix = await indexes.get("r2");
				return json({ ok: true, gen: ix?.gen ?? null, rev: ix?.rev ?? null });
			}
			if (configured(env.AUTH_TOKEN) && request.headers.get("authorization") !== `Bearer ${env.AUTH_TOKEN}`) return json({ error: "unauthorized" }, 401);
			const kind = (url.searchParams.get("store") as StoreKind | null) ?? "r2";
			if (url.pathname === "/search" && request.method === "POST") return await handleSearch(request, env, ctx, url);
			if (url.pathname === "/manifest" && request.method === "GET") {
				const ix = await getIndex(env, ctx, kind);
				return json({ store: kind, gen: ix.gen, rev: ix.rev, dim: ix.dim, cells: ix.cells, format: ix.format, defaults: ix.defaults, manifest: { ...ix.manifest, notes: undefined } });
			}
			if (url.pathname === "/refresh" && request.method === "POST") {
				const ix = await getIndex(env, ctx, kind);
				return json({ store: kind, changed: await ix.refresh(), gen: ix.gen, rev: ix.rev });
			}
			return json({ error: "not found" }, 404);
		} catch (e) {
			if (e instanceof VibError) return json({ error: e.message }, e.status >= 400 && e.status < 600 ? e.status : 500);
			console.error(e instanceof Error ? e.stack : String(e));
			return json({ error: e instanceof Error ? e.message : String(e) }, 500);
		}
	},
} satisfies ExportedHandler<Env>;
