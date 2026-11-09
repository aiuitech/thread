/**
 * Block-cache bench Worker (bench/blocks/README.md). Every request needs `Authorization: Bearer $BENCH_TOKEN`.
 *
 *   POST /search  {prefix, text?, vector?, topK?, recallK?, cacheOrigin, blockBytes?, docs?}
 *                 opens the index under `prefix` (it must be under tmp/blockbench/) the way inseam does, once per
 *                 request with the Cache API in front and `waitUntil`, runs one `search` and closes it. Returns the
 *                 search's spans (ms and store work), the open's, what reached R2 (counted at the binding) and the
 *                 block cache's stats. A fresh `cacheOrigin` is a cold cache; the same one again is warm.
 *   PUT  /put?key=tmp/blockbench/...     writes the body (the index upload)
 *   POST /delete  {prefix}               deletes every object under a tmp/blockbench/ prefix
 */
import wasmModule from "thread-sdk/wasm";
import * as sdk from "thread-sdk";

interface Env {
	BUCKET: R2Bucket;
	BENCH_TOKEN: string;
	SDK_LABEL: string;
}

const ROOT = "tmp/blockbench/";

let threadP: Promise<sdk.Thread> | null = null;
const getThread = () => (threadP ??= sdk.Thread.init({ wasmModule }));

/** The R2 binding with every call counted (what reached the origin, whatever the SDK's trace says). */
function counted(b: R2Bucket) {
	const c = { gets: 0, ranged: 0, heads: 0, bytes: 0 };
	const bucket = {
		async get(key: string, options?: R2GetOptions) {
			c.gets++;
			if (options?.range) c.ranged++;
			const o = await b.get(key, options);
			if (!o) return null;
			return {
				etag: o.etag,
				size: o.size,
				async arrayBuffer() {
					const ab = await o.arrayBuffer();
					c.bytes += ab.byteLength;
					return ab;
				},
			};
		},
		async head(key: string) {
			c.heads++;
			return b.head(key);
		},
		put: () => {
			throw new Error("read-only");
		},
		list: (o?: R2ListOptions) => b.list(o),
	};
	return { bucket: bucket as unknown as sdk.R2BucketLike, c };
}

const json = (v: unknown, status = 200) => new Response(JSON.stringify(v), { status, headers: { "content-type": "application/json" } });

const io = (x: sdk.TraceIo | undefined) => (x ? { reads: x.reads, hits: x.hits, misses: x.misses, fills: (x as { fills?: number }).fills ?? 0, bytes: x.bytes } : null);

async function search(env: Env, ctx: ExecutionContext, body: Record<string, unknown>) {
	const prefix = String(body.prefix ?? "");
	if (!prefix.startsWith(ROOT)) throw new Error(`prefix must be under ${ROOT}`);
	const thread = await getThread();
	const { bucket, c } = counted(env.BUCKET);
	const t0 = Date.now();
	const opts: Record<string, unknown> = { bucket, prefix, waitUntil: (p: Promise<unknown>) => ctx.waitUntil(p), cacheOrigin: String(body.cacheOrigin), refreshInterval: 0 };
	if (typeof body.blockBytes === "number") opts.blockBytes = body.blockBytes;
	const ix = await thread.open(opts as sdk.OpenOptions);
	const tOpen = Date.now() - t0;
	const r0 = { ...c };
	try {
		const vector = Array.isArray(body.vector) ? Float32Array.from(body.vector as number[]) : undefined;
		const req = { text: typeof body.text === "string" ? body.text : undefined, vector, topK: Number(body.topK ?? 20), recallK: Number(body.recallK ?? 100), docs: (body.docs === false ? undefined : (body.docs ?? "text")) as "text" | undefined, embed: false };
		const t1 = Date.now();
		const res = await ix.search(req as sdk.SearchRequest);
		const tSearch = Date.now() - t1;
		const store = (ix as unknown as { store: unknown }).store;
		let s: unknown = store;
		while (s && !(s instanceof sdk.CachedBlobStore)) s = (s as { inner?: unknown }).inner;
		const blocks = s instanceof sdk.CachedBlobStore ? { blockBytes: (s as { blockBytes?: number }).blockBytes ?? 0, ...((s as { blockStats?: object }).blockStats ?? {}) } : null;
		return {
			sdk: env.SDK_LABEL,
			gen: ix.gen,
			rev: ix.rev,
			segments: ix.segments.length,
			segmentBytes: ix.segments.map((x) => x.bytes),
			open: { ms: tOpen, io: io(ix.openTrace?.io), r2: r0 },
			search: { ms: tSearch, io: io(res.stats.trace.io) },
			spans: res.stats.trace.spans.map((x) => ({ name: x.name, ms: x.ms, io: io(x.io) })),
			r2: { gets: c.gets - r0.gets, ranged: c.ranged - r0.ranged, heads: c.heads - r0.heads, bytes: c.bytes - r0.bytes },
			blocks,
			mode: res.mode,
			hits: res.hits.slice(0, 5).map((h) => h.id),
		};
	} finally {
		ix.close();
	}
}

async function deletePrefix(env: Env, prefix: string) {
	if (!prefix.startsWith(ROOT)) throw new Error(`prefix must be under ${ROOT}`);
	let n = 0;
	let cursor: string | undefined;
	do {
		const l = await env.BUCKET.list({ prefix, cursor, limit: 1000 });
		if (l.objects.length) await env.BUCKET.delete(l.objects.map((o) => o.key));
		n += l.objects.length;
		cursor = l.truncated ? l.cursor : undefined;
	} while (cursor);
	return { deleted: n };
}

export default {
	async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
		if (!env.BENCH_TOKEN || req.headers.get("authorization") !== `Bearer ${env.BENCH_TOKEN}`) return json({ error: "unauthorized" }, 401);
		const url = new URL(req.url);
		try {
			if (req.method === "POST" && url.pathname === "/search") return json(await search(env, ctx, (await req.json()) as Record<string, unknown>));
			if (req.method === "PUT" && url.pathname === "/put") {
				const key = url.searchParams.get("key") ?? "";
				if (!key.startsWith(ROOT)) return json({ error: `key must be under ${ROOT}` }, 400);
				const o = await env.BUCKET.put(key, await req.arrayBuffer());
				return json({ key, size: o?.size });
			}
			if (req.method === "POST" && url.pathname === "/delete") return json(await deletePrefix(env, String(((await req.json()) as { prefix?: string }).prefix ?? "")));
			return json({ error: "not found" }, 404);
		} catch (e) {
			return json({ error: e instanceof Error ? `${e.message}\n${e.stack}` : String(e) }, 500);
		}
	},
};
