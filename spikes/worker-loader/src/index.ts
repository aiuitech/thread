/**
 * vib-loader-spike: can the Worker Loader binding replace the vib-bench-0..7 build pool?
 *
 * Routes (all except /health need `authorization: Bearer $BUILD_TOKEN`):
 *   POST /run?slot=N            body {mb, holdMs, kind: "js"|"wasm"}: memory probe in dynamic Worker `mem:slot-N`
 *   POST /run-local             same probe inside this (orchestrator) isolate, as a control
 *   GET  /r2test?slot=N&key=K   reads K from a dynamic Worker, once with the R2 binding passed directly
 *                               in env and once through the BucketRpc entrypoint stub
 *   POST /cleanup               deletes build/spike/*
 *   POST /build/<stage>?slot=N  runs a real build stage in dynamic Worker `vib-build:<bundle>:slot-N`,
 *                               which runs the vib-bench bundle (build-bundle/, copied from worker/)
 * Query `gen=<string>` is mixed into the dynamic Worker id, to force fresh isolates between runs.
 */
import { WorkerEntrypoint } from "cloudflare:workers";
import bundleSource from "../build-bundle/index.js?raw";
import vibWasm from "../build-bundle/assets/vib_bg-Ddd4raaf.wasm";
import { MEM_MODULE, SHIM_MODULE } from "./dynamic.ts";

interface Env {
	BUCKET: R2Bucket;
	BUILD_TOKEN: string;
	LOADER: WorkerLoader;
}

const COMPAT_DATE = "2026-10-01";
// Bump when build-bundle/ changes (the loader caches code by id).
const BUNDLE_VERSION = "43cfba85-d3ff1cbc";
// Random values are not allowed at global scope, so the isolate id is made on first use.
let orchIso = "";
const ORCH_ISO_FN = () => (orchIso ||= crypto.randomUUID().slice(0, 8));
let orchInflight = 0;

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

/** R2 access for dynamic Workers (an R2 binding cannot be passed into a dynamic Worker's env). */
export class BucketRpc extends WorkerEntrypoint<Env> {
	async r2get(key: string, opts?: R2GetOptions) {
		const o = await this.env.BUCKET.get(key, opts);
		if (!o) return null;
		// With onlyIf a failed precondition returns an R2Object without a body.
		return { meta: meta(o), body: "body" in o ? (o as R2ObjectBody).body : null };
	}
	async r2head(key: string) {
		const o = await this.env.BUCKET.head(key);
		return o ? meta(o) : null;
	}
	async r2put(key: string, value: ArrayBuffer | ArrayBufferView | string | null, opts?: R2PutOptions) {
		const o = await this.env.BUCKET.put(key, value, opts);
		return o ? meta(o) : null;
	}
	async r2list(opts?: R2ListOptions) {
		const r = await this.env.BUCKET.list(opts);
		return { objects: r.objects.map(meta), truncated: r.truncated, cursor: r.truncated ? r.cursor : undefined, delimitedPrefixes: r.delimitedPrefixes };
	}
	async r2delete(keys: string | string[]) {
		await this.env.BUCKET.delete(keys);
	}
}

const json = (o: unknown, status = 200, headers: Record<string, string> = {}) =>
	new Response(JSON.stringify(o), { status, headers: { "content-type": "application/json", ...headers } });

function authorized(request: Request, env: Env): boolean {
	const token: unknown = env.BUILD_TOKEN;
	if (typeof token !== "string" || token === "") return false;
	const enc = new TextEncoder();
	const got = enc.encode(request.headers.get("authorization") ?? "");
	const want = enc.encode(`Bearer ${token}`);
	return got.byteLength === want.byteLength && crypto.subtle.timingSafeEqual(got, want);
}

/** Forwards to a dynamic Worker and adds orchestrator-side timing; failures become a JSON 502. */
async function forward(stub: WorkerStub, request: Request, loaded: { cold: boolean; codeMs: number }, t0: number, opts?: WorkerStubEntrypointOptions) {
	try {
		const resp = await stub.getEntrypoint(undefined, opts).fetch(request);
		const out = new Response(resp.body, resp);
		out.headers.set("x-orch-ms", String(Date.now() - t0));
		out.headers.set("x-orch-cold", String(loaded.cold));
		out.headers.set("x-orch-code-ms", String(loaded.codeMs));
		out.headers.set("x-orch-iso", ORCH_ISO_FN());
		out.headers.set("x-orch-inflight", String(orchInflight));
		return out;
	} catch (e) {
		return json({ ok: false, error: `dynamic worker failed: ${e instanceof Error ? `${e.name}: ${e.message}` : String(e)}`, ms: Date.now() - t0, cold: loaded.cold, orchIso: ORCH_ISO_FN() }, 502);
	}
}

function slotOf(url: URL): number {
	const slot = Number(url.searchParams.get("slot") ?? "0");
	if (!Number.isInteger(slot) || slot < 0 || slot > 63) throw new Error("bad slot");
	return slot;
}

export default {
	async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
		const url = new URL(request.url);
		const t0 = Date.now();
		if (url.pathname === "/health") return json({ ok: true, orchIso: ORCH_ISO_FN() });
		if (!authorized(request, env)) return json({ error: "unauthorized" }, 401);
		const exportsAny = (ctx as any).exports;
		const gen = url.searchParams.get("gen") ?? "0";
		orchInflight++;
		try {
			if (url.pathname === "/run" && request.method === "POST") {
				const slot = slotOf(url);
				const loaded = { cold: false, codeMs: 0 };
				const stub = env.LOADER.get(`mem:${gen}:slot-${slot}`, () => {
					loaded.cold = true;
					return { compatibilityDate: COMPAT_DATE, mainModule: "mem.js", modules: { "mem.js": MEM_MODULE }, env: {}, globalOutbound: null };
				});
				return await forward(stub, request, loaded, t0);
			}

			if (url.pathname === "/run-local" && request.method === "POST") {
				const { mb = 90, holdMs = 3000 } = (await request.json()) as { mb?: number; holdMs?: number };
				const chunks: Uint8Array[] = [];
				for (let i = 0; i < mb; i++) chunks.push(new Uint8Array(1 << 20).fill((i % 250) + 1));
				await new Promise((r) => setTimeout(r, holdMs));
				return json({ ok: true, orchIso: ORCH_ISO_FN(), orchInflight, mbAllocated: chunks.length, ms: Date.now() - t0 });
			}

			if (url.pathname === "/r2test" && request.method === "GET") {
				const slot = slotOf(url);
				const key = url.searchParams.get("key") ?? "idx/current.json";
				const rpc = exportsAny.BucketRpc({ props: {} });
				const out: Record<string, unknown> = {};
				// 1. The R2 binding itself in env (expected to fail: env must be structured-cloneable or service stubs).
				try {
					const stub = env.LOADER.get(`r2direct:${gen}:slot-${slot}`, () => ({
						compatibilityDate: COMPAT_DATE,
						mainModule: "mem.js",
						modules: { "mem.js": MEM_MODULE },
						env: { BUCKET: env.BUCKET },
					}));
					const r = await stub.getEntrypoint().fetch(`https://dyn/r2?key=${encodeURIComponent(key)}`);
					out.direct = { status: r.status, body: await r.json() };
				} catch (e) {
					out.direct = { error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) };
				}
				// 2. A WorkerEntrypoint stub (ctx.exports loopback) wrapping the bucket.
				try {
					const stub = env.LOADER.get(`r2rpc:${gen}:slot-${slot}`, () => ({
						compatibilityDate: COMPAT_DATE,
						mainModule: "mem.js",
						modules: { "mem.js": MEM_MODULE },
						env: { BUCKET_RPC: rpc },
					}));
					const r = await stub.getEntrypoint().fetch(`https://dyn/r2?key=${encodeURIComponent(key)}`);
					out.rpc = { status: r.status, body: await r.json() };
				} catch (e) {
					out.rpc = { error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) };
				}
				out.ms = Date.now() - t0;
				return json(out);
			}

			// Deletes this spike's scratch output (build/spike/) directly through the orchestrator's binding.
			if (url.pathname === "/cleanup" && request.method === "POST") {
				let deleted = 0;
				let cursor: string | undefined;
				do {
					const r = await env.BUCKET.list({ prefix: "build/spike/", cursor, limit: 1000 });
					if (r.objects.length) await env.BUCKET.delete(r.objects.map((o) => o.key));
					deleted += r.objects.length;
					cursor = r.truncated ? r.cursor : undefined;
				} while (cursor);
				return json({ ok: true, deleted });
			}

			if (url.pathname.startsWith("/build/") && request.method === "POST") {
				const slot = slotOf(url);
				const loaded = { cold: false, codeMs: 0 };
				const stub = env.LOADER.get(`vib-build:${BUNDLE_VERSION}:${gen}:slot-${slot}`, () => {
					const tc = Date.now();
					loaded.cold = true;
					const code: WorkerLoaderWorkerCode = {
						compatibilityDate: COMPAT_DATE,
						mainModule: "shim.js",
						modules: {
							"shim.js": SHIM_MODULE,
							"index.js": bundleSource,
							"assets/vib_bg-Ddd4raaf.wasm": vibWasm,
						},
						env: { BUILD_TOKEN: env.BUILD_TOKEN, BUCKET_RPC: exportsAny.BucketRpc({ props: {} }) },
					};
					const cpuMs = Number(url.searchParams.get("cpuMs") ?? "0");
					if (cpuMs > 0) code.limits = { cpuMs };
					loaded.codeMs = Date.now() - tc;
					return code;
				});
				// Strip orchestrator-only query params before the bundle sees the request.
				const fwdUrl = new URL(url.pathname, "https://vib-dyn");
				const fwd = new Request(fwdUrl, request);
				return await forward(stub, fwd, loaded, t0);
			}
			return json({ error: "not found" }, 404);
		} catch (e) {
			return json({ error: e instanceof Error ? `${e.name}: ${e.message}` : String(e), ms: Date.now() - t0 }, 500);
		} finally {
			orchInflight--;
		}
	},
} satisfies ExportedHandler<Env>;
