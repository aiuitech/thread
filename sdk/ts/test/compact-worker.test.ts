/**
 * Cross-check of the SDK's dense compaction against the Worker's own: the same index (created and filled by the
 * SDK) is compacted by `examples/cloudflare-worker`'s `POST /build/compact` over an in-memory R2 bucket and by
 * `ix.compact()` over a copy, and the results must be the same objects: every slab revision byte for byte, the
 * slabs table, the fine means after splits, and the manifest fields readers use. A second round (more upserts, a
 * delete, several cells) repeats it on the grown index.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { MemoryBlobStore, keys, type Manifest } from "../src/index.js";
import { rng, thread, unitRows } from "./helpers.js";

const here = path.dirname(fileURLToPath(import.meta.url));
// The Worker imports its wasm as a module (a bundler feature): give it the compiled module.
vi.mock("../../../crates/thread-wasm/pkg/thread_bg.wasm", async () => {
	const nodeFs = await import("node:fs");
	const nodePath = await import("node:path");
	const url = await import("node:url");
	const dir = nodePath.dirname(url.fileURLToPath(import.meta.url));
	const bytes = nodeFs.readFileSync(nodePath.join(dir, "../../../crates/thread-wasm/pkg/thread_bg.wasm"));
	return { default: new (WebAssembly as unknown as { Module: new (b: BufferSource) => WebAssembly.Module }).Module(bytes) };
});

const WORKER = path.join(here, "../../../examples/cloudflare-worker/src/index.ts");
const haveWorkerWasm = fs.existsSync(path.join(here, "../../../crates/thread-wasm/pkg/thread_bg.wasm"));

/** The R2 binding surface the Worker uses, over a Map (conditional puts, ranges, listing with upload times). */
class FakeR2 {
	readonly objects = new Map<string, { bytes: Uint8Array; etag: string; uploaded: Date }>();
	private seq = 0;
	private obj(key: string, o: { bytes: Uint8Array; etag: string; uploaded: Date }, bytes = o.bytes) {
		return {
			key,
			size: o.bytes.byteLength,
			etag: o.etag,
			httpEtag: `"${o.etag}"`,
			uploaded: o.uploaded,
			arrayBuffer: async () => bytes.slice().buffer,
			text: async () => new TextDecoder().decode(bytes),
			json: async () => JSON.parse(new TextDecoder().decode(bytes)),
		};
	}
	async get(key: string, opts?: { range?: { offset?: number; length?: number; suffix?: number } }) {
		const o = this.objects.get(key);
		if (!o) return null;
		const r = opts?.range;
		if (!r) return this.obj(key, o);
		if (r.suffix !== undefined) return this.obj(key, o, o.bytes.subarray(Math.max(0, o.bytes.byteLength - r.suffix)));
		const off = r.offset ?? 0;
		return this.obj(key, o, o.bytes.subarray(off, r.length !== undefined ? off + r.length : undefined));
	}
	async head(key: string) {
		const o = this.objects.get(key);
		return o ? this.obj(key, o) : null;
	}
	async put(key: string, value: ArrayBuffer | ArrayBufferView | string, opts?: { onlyIf?: { etagMatches?: string; etagDoesNotMatch?: string } }) {
		const cur = this.objects.get(key);
		if (opts?.onlyIf?.etagDoesNotMatch === "*" && cur) return null;
		if (opts?.onlyIf?.etagMatches !== undefined && cur?.etag !== opts.onlyIf.etagMatches) return null;
		const bytes = typeof value === "string" ? new TextEncoder().encode(value) : value instanceof ArrayBuffer ? new Uint8Array(value.slice(0)) : new Uint8Array(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength));
		const o = { bytes, etag: `r${++this.seq}`, uploaded: new Date() };
		this.objects.set(key, o);
		return this.obj(key, o);
	}
	async list(opts: { prefix?: string; cursor?: string; limit?: number } = {}) {
		const all = [...this.objects.keys()].filter((k) => k.startsWith(opts.prefix ?? "")).sort();
		const start = opts.cursor ? Number(opts.cursor) : 0;
		const limit = opts.limit ?? 1000;
		const page = all.slice(start, start + limit);
		return { objects: page.map((k) => this.obj(k, this.objects.get(k)!)), truncated: start + limit < all.length, cursor: String(start + limit), delimitedPrefixes: [] };
	}
	async delete(keys: string | string[]) {
		for (const k of Array.isArray(keys) ? keys : [keys]) this.objects.delete(k);
	}
}

type WorkerModule = { default: { fetch(req: Request, env: unknown, ctx: unknown): Promise<Response> } };

async function workerCompact(bucket: FakeR2, body: Record<string, unknown>): Promise<Record<string, unknown>> {
	const subtle = crypto.subtle as unknown as { timingSafeEqual?: (a: Uint8Array, b: Uint8Array) => boolean };
	subtle.timingSafeEqual ??= (a, b) => a.byteLength === b.byteLength && a.every((x, i) => x === b[i]);
	const worker = (await import(/* @vite-ignore */ WORKER)) as WorkerModule;
	const res = await worker.default.fetch(new Request("https://w/build/compact", { method: "POST", headers: { authorization: "Bearer t" }, body: JSON.stringify(body) }), { BUCKET: bucket, BUILD_TOKEN: "t" }, { waitUntil() {}, passThroughOnException() {} });
	const j = (await res.json()) as { ok?: boolean; result?: Record<string, unknown>; error?: string };
	if (!j.ok) throw new Error(`worker /build/compact: HTTP ${res.status}: ${j.error}`);
	return j.result!;
}

const DIM = 16;
const dec = new TextDecoder();

function copyInto(from: MemoryBlobStore, to: FakeR2 | MemoryBlobStore) {
	for (const [k, o] of from.objects) {
		if (to instanceof FakeR2) to.objects.set(k, { bytes: o.bytes.slice(), etag: o.etag, uploaded: new Date() });
		else to.objects.set(k, { ...o, bytes: o.bytes.slice() });
	}
}

async function headOf(get: (key: string) => Promise<Uint8Array | null>, list: () => Promise<string[]>, gen: string): Promise<Manifest> {
	const revs = (await list()).map((k) => k.match(/manifest\.(\d+)\.json$/)?.[1]).filter(Boolean).map(Number);
	const rev = Math.max(0, ...revs);
	return JSON.parse(dec.decode((await get(keys.manifest(gen, rev)))!)) as Manifest;
}

describe.skipIf(!haveWorkerWasm)("SDK compaction against the Worker's", () => {
	it("writes the same slabs, slabs table, fine means and manifest", async () => {
		const gen = "x1";
		const src = new MemoryBlobStore();
		await thread.createGeneration({ store: src, gen, dim: DIM, lex: null, compaction: { maxDeltas: 2, maxCellEntries: 50 } });
		const r = rng(5);
		const centers = unitRows(6, DIM, r);
		const v = unitRows(400, DIM, r, centers, 0.3);
		const writer = await thread.open({ store: src, cache: false, refreshInterval: 0 });
		const upsert = async (a: number, e: number) => writer.upsert({ ids: Array.from({ length: e - a }, (_, i) => a + i), vectors: v.subarray(a * DIM, e * DIM) });
		for (let a = 0; a < 120; a += 40) await upsert(a, a + 40);

		for (let round = 0; round < 3; round++) {
			// From round 1 on the head has a centroid tree, which a split commit rebuilds over the new means.
			if (round === 1) await addTree(src, gen);
			const bucket = new FakeR2();
			const sdkStore = new MemoryBlobStore();
			copyInto(src, bucket);
			copyInto(src, sdkStore);
			const wr = await workerCompact(bucket, { gen });
			const ix = await thread.open({ store: sdkStore, gen, cache: false, refreshInterval: 0 });
			const sr = await ix.compact({ force: true, purge: false, maxBytes: 1 << 30, cellsPerRun: 1 << 20 });
			ix.close();
			expect(sr.outcome).toBe("committed");
			expect(sr.rev).toBe(wr.rev);
			expect(sr.k).toBe(wr.k);
			expect(sr.folded).toBe(wr.folded);
			expect(sr.splits.length).toBe(wr.splits);
			const wm = await headOf((k) => bucket.get(k).then(async (o) => (o ? new Uint8Array(await o.arrayBuffer()) : null)), async () => [...bucket.objects.keys()], gen);
			const sm = await headOf((k) => sdkStore.get(k), () => sdkStore.list(""), gen);
			for (const f of ["k1", "k2", "dim", "vectors", "deltas", "tombstones", "files", "tree", "routing", "nextSeq"] as const) expect(sm[f], f).toEqual(wm[f]);
			// Every object the two compactions wrote, byte for byte (slab revisions, slabs table, fine means).
			const written = (keysOf: string[]) => keysOf.filter((k) => !src.objects.has(k) && !/manifest|current\.json|compact\./.test(k)).sort();
			const wk = written([...bucket.objects.keys()]);
			const sk = written([...sdkStore.objects.keys()]);
			expect(sk).toEqual(wk);
			expect(wk.some((k) => k.includes("/slab/"))).toBe(true);
			expect(wk.some((k) => k.includes("/fine."))).toBe(true);
			if (round >= 1) expect(wk.some((k) => k.includes("/tree."))).toBe(true);
			for (const k of wk) expect(Buffer.from(sdkStore.objects.get(k)!.bytes).equals(Buffer.from(bucket.objects.get(k)!.bytes)), k).toBe(true);
			// The SDK continues from its own result: the next round's input is identical for both.
			for (const k of src.objects.keys()) if (!sdkStore.objects.has(k)) src.objects.delete(k);
			copyInto(sdkStore, src);
			const w2 = await thread.open({ store: src, gen, cache: false, refreshInterval: 0 });
			const base = 120 + round * 90;
			for (let a = base; a < base + 90; a += 30) await w2.upsert({ ids: Array.from({ length: 30 }, (_, i) => a + i), vectors: v.subarray(a * DIM, (a + 30) * DIM) });
			await w2.delete([round, 10 + round]);
			await w2.upsert({ ids: [20 + round], vectors: v.subarray((399 - round) * DIM, (400 - round) * DIM) });
			w2.close();
		}
		writer.close();
	});
});

/** A revision of the head with a centroid tree over its means (what the Worker's publish-rev `tree` writes). */
async function addTree(store: MemoryBlobStore, gen: string) {
	const revs = (await store.list(`idx/${gen}/manifest.`)).map((k) => Number(k.match(/manifest\.(\d+)\.json$/)?.[1] ?? 0));
	const rev = Math.max(...revs);
	const m = JSON.parse(dec.decode((await store.get(keys.manifest(gen, rev)))!)) as Manifest;
	const fine = new Float32Array((await store.get(m.files!.fine))!.slice().buffer);
	const k = m.k1 * m.k2;
	const tree = thread.wasm.build_centroid_tree(fine, m.dim, k, 10, 7);
	const key = `idx/${gen}/tree.${rev + 1}.bin`;
	await store.put(key, tree);
	const next = { ...m, rev: rev + 1, parentRev: rev, tree: { key, groups: k, beam: k, iters: 10, seed: 7, bytes: tree.byteLength }, change: { op: "publish-rev" } };
	await store.put(keys.manifest(gen, rev + 1), JSON.stringify(next));
	await store.put(keys.current(), JSON.stringify({ gen, rev: rev + 1 }));
}
