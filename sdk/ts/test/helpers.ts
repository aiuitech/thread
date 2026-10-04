/** Shared by the end-to-end tests: the wasm in Node, a seeded RNG, synthetic unit vectors and a tiny dense generation built with the wasm build jobs. */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect } from "vitest";
import { MemoryBlobStore, Thread, keys, type CacheLike, type Manifest, type SlabRow } from "../src/index.js";

// Node has a constructible WebAssembly.Module (the workers-types declaration is abstract, hence the cast).
const wasmBytes = fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), "../src/wasm/thread_bg.wasm"));
export const wasmModule = new (WebAssembly as unknown as { Module: new (b: BufferSource) => WebAssembly.Module }).Module(wasmBytes);
export const thread = await Thread.init({ wasmModule });
export const w = thread.wasm;

export function rng(seed: number): () => number {
	let s = seed >>> 0;
	return () => {
		s = (s + 0x6d2b79f5) >>> 0;
		let t = s;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
export function gaussian(r: () => number): number {
	return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
}
export function unitRows(n: number, dim: number, r: () => number, centers?: Float32Array, spread = 0.5): Float32Array {
	const out = new Float32Array(n * dim);
	for (let i = 0; i < n; i++) {
		const c = centers ? Math.floor(r() * (centers.length / dim)) : -1;
		let ss = 0;
		for (let d = 0; d < dim; d++) {
			const v = gaussian(r) * (c >= 0 ? spread : 1) + (c >= 0 ? centers![c * dim + d] : 0);
			out[i * dim + d] = v;
			ss += v * v;
		}
		const inv = 1 / Math.sqrt(ss);
		for (let d = 0; d < dim; d++) out[i * dim + d] *= inv;
	}
	return out;
}
export const f32bytes = (f: Float32Array) => new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
export const dot = (a: Float32Array, b: Float32Array, ao: number, bo: number, dim: number) => {
	let s = 0;
	for (let d = 0; d < dim; d++) s += a[ao + d] * b[bo + d];
	return s;
};
export function bruteForce(vectors: Float32Array, dim: number, q: Float32Array, k: number): number[] {
	const n = vectors.length / dim;
	const scored = Array.from({ length: n }, (_, i) => ({ i, s: dot(vectors, q, i * dim, 0, dim) }));
	scored.sort((a, b) => b.s - a.s);
	return scored.slice(0, k).map((x) => x.i);
}

export interface Built {
	store: MemoryBlobStore;
	vectors: Float32Array;
	dim: number;
}

/** Builds a generation like the Worker's assign + finalize + publish stages, with a slabs table unless `legacy`. */
export async function buildGen(gen: string, format: 1 | 2 | 3, opts: { dim?: number; k2?: number; m?: number; n?: number; legacy?: boolean; store?: MemoryBlobStore } = {}): Promise<Built> {
	const dim = opts.dim ?? 16;
	const k2 = opts.k2 ?? 8;
	const m = opts.m ?? 2;
	const n = opts.n ?? 400;
	const r = rng(42);
	const fine = unitRows(k2, dim, r);
	const vectors = unitRows(n, dim, r, fine, 0.6);
	const coarse = new Float32Array(dim).fill(1 / Math.sqrt(dim));
	const store = opts.store ?? new MemoryBlobStore();
	const job = new w.AssignJob(dim, 1, k2, coarse, fine, 1, m);
	let codebook: Uint8Array | null = null;
	const pqM = 4;
	if (format === 3) {
		const assign = new Uint32Array(n);
		for (let i = 0; i < n; i++) {
			let best = -1;
			let bs = -Infinity;
			for (let c = 0; c < k2; c++) {
				const s = dot(vectors, fine, i * dim, c * dim, dim);
				if (s > bs) (bs = s), (best = c);
			}
			assign[i] = best;
		}
		codebook = w.pq_init_codebook(w.residuals_for(vectors, fine, assign, dim), dim, pqM, 7);
		job.set_codebook(codebook);
	}
	expect(job.add_batch(vectors, 0)).toBe(n);
	const part = job.take_part(0);
	job.free();
	const dir = w.part_directory(part);
	expect(dir[0]).toBe(k2);
	const rows: SlabRow[] = [];
	for (let cell = 0; cell < k2; cell++) {
		const [count, off, len] = [dir[2 + cell * 3], dir[3 + cell * 3], dir[4 + cell * 3]];
		const sj = new w.SlabJob(dim, cell, fine.subarray(cell * dim, (cell + 1) * dim), format);
		if (len) expect(sj.add_records(part.subarray(off, off + len))).toBe(count);
		const bytes = sj.finish();
		const info = w.slab_info(bytes);
		expect(info[4]).toBe(format);
		rows.push([info[5], info[6], info[1], bytes.byteLength]);
		await store.put(keys.slab(gen, cell), bytes);
	}
	await store.put(`idx/${gen}/coarse.f32`, f32bytes(coarse));
	await store.put(`idx/${gen}/fine.f32`, f32bytes(fine));
	if (codebook) await store.put(`idx/${gen}/pq.bin`, codebook);
	if (!opts.legacy) await store.put(`idx/${gen}/slabs.json`, JSON.stringify(rows));
	const manifest: Manifest = { gen, k1: 1, k2, dim, m, probeCoarse: 1, vectors: n, format, tiers: format === 3 ? `pq${pqM}+sq8` : format === 2 ? "sq4+sq8" : "sq8", ...(format === 3 ? { pq: { m: pqM } } : {}), defaults: { nprobe: 3, rerank: 20, pc: 0 } };
	await store.put(keys.manifest(gen, 0), JSON.stringify(manifest));
	await store.put(keys.current(), JSON.stringify(opts.legacy ? { gen } : { gen, rev: 0 }));
	return { store, vectors, dim };
}

/** A Cache API stand-in that honours single and multi-range requests like Cloudflare's cache does. */
export class FakeCache implements CacheLike {
	objects = new Map<string, Uint8Array>();
	matches = 0;
	puts = 0;
	/** 0 = honour Range (206), 1 = ignore Range (200 full body). */
	ignoreRange = false;
	async match(req: Request): Promise<Response | undefined> {
		this.matches++;
		const body = this.objects.get(req.url);
		if (!body) return undefined;
		const range = req.headers.get("range");
		if (!range || this.ignoreRange) return new Response(body.slice(), { status: 200, headers: { "content-length": String(body.byteLength) } });
		const ranges = range
			.replace(/^bytes=/, "")
			.split(",")
			.map((r) => r.trim().split("-").map(Number) as [number, number]);
		if (ranges.length === 1) {
			const [a, b] = ranges[0];
			return new Response(body.slice(a, b + 1), { status: 206, headers: { "content-range": `bytes ${a}-${b}/${body.byteLength}` } });
		}
		const boundary = "fakeboundary42";
		const enc = new TextEncoder();
		const parts: Uint8Array[] = [];
		for (const [a, b] of ranges) {
			parts.push(enc.encode(`--${boundary}\r\nContent-Type: application/octet-stream\r\nContent-Range: bytes ${a}-${b}/${body.byteLength}\r\n\r\n`));
			parts.push(body.slice(a, b + 1));
			parts.push(enc.encode("\r\n"));
		}
		parts.push(enc.encode(`--${boundary}--\r\n`));
		const total = parts.reduce((x, p) => x + p.byteLength, 0);
		const out = new Uint8Array(total);
		let o = 0;
		for (const p of parts) {
			out.set(p, o);
			o += p.byteLength;
		}
		return new Response(out, { status: 206, headers: { "content-type": `multipart/byteranges; boundary=${boundary}` } });
	}
	async put(req: Request, res: Response): Promise<void> {
		this.puts++;
		this.objects.set(req.url, new Uint8Array(await res.arrayBuffer()));
	}
}
