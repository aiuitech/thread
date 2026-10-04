// Source of the modules that run inside the dynamic Workers (passed to the loader as strings).

/**
 * Memory probe. POST / {mb, holdMs, kind: "js" | "wasm"} allocates `mb` MB, touches every page,
 * holds it for holdMs, then reports which isolate it ran in (a random id per isolate) and how many
 * requests were in flight in that isolate at the same time (>1 means requests share an isolate).
 * GET /r2?key=... reads an object through env.BUCKET (direct binding) or env.BUCKET_RPC (RPC stub).
 */
export const MEM_MODULE = `
let ISO = "";
let BORN = 0;
let inflight = 0, maxInflight = 0, served = 0;
const json = (o, s = 200) => new Response(JSON.stringify(o), { status: s, headers: { "content-type": "application/json" } });

export default {
	async fetch(req, env) {
		const url = new URL(req.url);
		ISO ||= crypto.randomUUID().slice(0, 8);
		BORN ||= Date.now();
		inflight++; served++;
		maxInflight = Math.max(maxInflight, inflight);
		const inflightAtStart = inflight;
		try {
			if (url.pathname === "/r2") {
				const key = url.searchParams.get("key") ?? "idx/current.json";
				const out = { iso: ISO, envKeys: Object.keys(env ?? {}) };
				for (const name of ["BUCKET", "BUCKET_RPC"]) {
					const b = env?.[name];
					if (!b) { out[name] = "absent"; continue; }
					try {
						const o = name === "BUCKET" ? await b.get(key) : await b.r2get(key);
						if (!o) { out[name] = "null"; continue; }
						// Direct R2 binding returns R2ObjectBody; the RPC stub returns {meta, body}.
						const text = o.text ? await o.text() : await new Response(o.body).text();
						out[name] = { ok: true, size: o.size ?? o.meta?.size, text: text.slice(0, 200) };
					} catch (e) {
						out[name] = { ok: false, error: String(e) };
					}
				}
				return json(out);
			}
			const { mb = 90, holdMs = 3000, kind = "js" } = await req.json();
			const t0 = Date.now();
			let bytes = 0, sum = 0;
			if (kind === "wasm") {
				// 1 MB = 16 wasm pages of 64 KB.
				const mem = new WebAssembly.Memory({ initial: mb * 16, maximum: mb * 16 });
				const u8 = new Uint8Array(mem.buffer);
				for (let i = 0; i < u8.length; i += 4096) u8[i] = (i >> 12) & 255;
				u8.fill(7, 0, u8.length);
				bytes = u8.length;
				await new Promise((r) => setTimeout(r, holdMs));
				for (let i = 0; i < u8.length; i += 1 << 20) sum += u8[i];
			} else {
				const chunks = [];
				for (let i = 0; i < mb; i++) {
					const c = new Uint8Array(1 << 20);
					c.fill((i % 250) + 1);
					chunks.push(c);
					bytes += c.length;
				}
				await new Promise((r) => setTimeout(r, holdMs));
				for (const c of chunks) sum += c[c.length - 1];
			}
			return json({ ok: true, iso: ISO, isoAgeMs: Date.now() - BORN, served, inflightAtStart, maxInflight, kind, mbAllocated: bytes / 1048576, checksum: sum, allocAndHoldMs: Date.now() - t0 });
		} catch (e) {
			return json({ ok: false, iso: ISO, error: String(e) }, 500);
		} finally {
			inflight--;
		}
	},
};
`;

/**
 * Entry shim for the real build bundle. Imports the bundle's default handler and gives it an
 * R2Bucket-shaped env.BUCKET built on the BUCKET_RPC stub (R2 bindings cannot be put in a dynamic
 * Worker's env directly), and tags each response with the isolate id and in-flight count.
 */
export const SHIM_MODULE = `
import inner from "./index.js";
let ISO = "";
let inflight = 0, maxInflight = 0, served = 0;

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

function wrapBucket(rpc) {
	return {
		async get(key, opts) {
			const r = await rpc.r2get(key, opts);
			return r ? wrapObject(r.meta, r.body) : null;
		},
		async head(key) {
			return wrapObject(await rpc.r2head(key), null);
		},
		async put(key, value, opts) {
			if (value instanceof ReadableStream) value = await new Response(value).arrayBuffer();
			else if (value instanceof Blob) value = await value.arrayBuffer();
			return wrapObject(await rpc.r2put(key, value, opts), null);
		},
		async list(opts) {
			return await rpc.r2list(opts);
		},
		async delete(keys) {
			await rpc.r2delete(keys);
		},
	};
}

export default {
	async fetch(req, env, ctx) {
		ISO ||= crypto.randomUUID().slice(0, 8);
		inflight++; served++;
		maxInflight = Math.max(maxInflight, inflight);
		const inflightAtStart = inflight;
		try {
			const resp = await inner.fetch(req, { ...env, BUCKET: wrapBucket(env.BUCKET_RPC) }, ctx);
			const out = new Response(resp.body, resp);
			out.headers.set("x-dyn-iso", ISO);
			out.headers.set("x-dyn-inflight", String(inflightAtStart));
			out.headers.set("x-dyn-max-inflight", String(maxInflight));
			out.headers.set("x-dyn-served", String(served));
			return out;
		} finally {
			inflight--;
		}
	},
};
`;
