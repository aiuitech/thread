/**
 * End-to-end in Node: builds tiny v1/v2/v3 generations with the wasm build jobs into a MemoryBlobStore, then
 * queries them through the SDK (brute-force recall), upserts/deletes with tombstones, refreshes and resets.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { MemoryBlobStore, Vib, VibIndex, keys, type Manifest, type SlabRow } from "../src/index.js";

// Node has a constructible WebAssembly.Module (the workers-types declaration is abstract, hence the cast).
const wasmBytes = fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), "../src/wasm/vib_bg.wasm"));
const wasmModule = new (WebAssembly as unknown as { Module: new (b: BufferSource) => WebAssembly.Module }).Module(wasmBytes);
const vib = await Vib.init({ wasmModule });
const w = vib.wasm;

function rng(seed: number): () => number {
	let s = seed >>> 0;
	return () => {
		s = (s + 0x6d2b79f5) >>> 0;
		let t = s;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
function gaussian(r: () => number): number {
	return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
}
function unitRows(n: number, dim: number, r: () => number, centers?: Float32Array, spread = 0.5): Float32Array {
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
const f32bytes = (f: Float32Array) => new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
const dot = (a: Float32Array, b: Float32Array, ao: number, bo: number, dim: number) => {
	let s = 0;
	for (let d = 0; d < dim; d++) s += a[ao + d] * b[bo + d];
	return s;
};
function bruteForce(vectors: Float32Array, dim: number, q: Float32Array, k: number): number[] {
	const n = vectors.length / dim;
	const scored = Array.from({ length: n }, (_, i) => ({ i, s: dot(vectors, q, i * dim, 0, dim) }));
	scored.sort((a, b) => b.s - a.s);
	return scored.slice(0, k).map((x) => x.i);
}

interface Built {
	store: MemoryBlobStore;
	vectors: Float32Array;
	dim: number;
}

/** Builds a generation like the Worker's assign + finalize + publish stages, with a slabs table unless `legacy`. */
async function buildGen(gen: string, format: 1 | 2 | 3, opts: { dim?: number; k2?: number; m?: number; n?: number; legacy?: boolean; store?: MemoryBlobStore } = {}): Promise<Built> {
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

async function recall(index: VibIndex, b: Built, nq: number, k: number, qopts: Parameters<VibIndex["query"]>[1] = {}): Promise<number> {
	const r = rng(7);
	let hit = 0;
	for (let i = 0; i < nq; i++) {
		const q = unitRows(1, b.dim, r);
		const truth = new Set(bruteForce(b.vectors, b.dim, q, k));
		const res = await index.query(q, { topK: k, ...qopts });
		expect(res.hits.length).toBe(k);
		expect(new Set(res.hits.map((h) => h.id)).size).toBe(k);
		for (let j = 1; j < res.hits.length; j++) expect(res.hits[j].score).toBeLessThanOrEqual(res.hits[j - 1].score);
		hit += res.hits.filter((h) => truth.has(h.id)).length;
	}
	return hit / (nq * k);
}

describe("end to end (wasm in node, MemoryBlobStore)", () => {
	it("v3 (PQ scan + int8 rerank): manifest defaults, params, stats, recall", async () => {
		const b = await buildGen("g3", 3);
		const index = await vib.open({ store: b.store, cache: false });
		expect(index.gen).toBe("g3");
		expect(index.rev).toBe(0);
		expect(index.dim).toBe(16);
		expect(index.cells).toBe(8);
		expect(index.format).toBe(3);
		const res = await index.query(b.vectors.subarray(0, 16));
		expect(res.hits[0].id).toBe(0);
		expect(res.hits[0].source).toBe("slab");
		expect(res.params).toEqual({ topK: 10, nprobe: 3, rerank: 20, pc: 0, source: { nprobe: "manifest", rerank: "manifest", pc: "manifest" } });
		expect(res.stats.cells.length).toBe(3);
		expect(res.stats.rerank).toBe(20);
		expect(res.stats.rerankSlabs).toBeGreaterThan(0);
		expect(res.stats.scanned).toBeGreaterThan(0);
		expect(res.stats.io.reads).toBeGreaterThan(0);
		// All cells probed: recall against brute force is high even with a toy codebook thanks to the int8 rerank.
		expect(await recall(index, b, 20, 10, { nprobe: 8, rerank: 40 })).toBeGreaterThanOrEqual(0.9);
		// rerank = 0: scan-tier scores only (PQ), lower but sane
		const r0 = await index.query(b.vectors.subarray(16, 32), { rerank: 0, nprobe: 8 });
		expect(r0.stats.rerank).toBe(0);
		expect(r0.params.source.rerank).toBe("query");
		expect(r0.hits.map((h) => h.id)).toContain(1);
		// returnPayload on slab hits without payloads yields empty buffers
		const rp = await index.query(b.vectors.subarray(0, 16), { returnPayload: true, topK: 3 });
		expect(rp.hits.every((h) => h.payload instanceof Uint8Array && h.payload.byteLength === 0)).toBe(true);
		await expect(index.query(new Float32Array(5))).rejects.toThrow(/dim 5/);
		index.close();
		await expect(index.query(b.vectors.subarray(0, 16))).rejects.toThrow(/closed/);
	});

	it("v2 (4-bit scan + int8 rerank) and queryMany", async () => {
		const b = await buildGen("g2", 2);
		const index = await vib.open({ store: b.store, cache: false });
		expect(index.format).toBe(2);
		expect(await recall(index, b, 20, 10, { nprobe: 8 })).toBeGreaterThanOrEqual(0.95);
		const many = await index.queryMany([b.vectors.subarray(0, 16), b.vectors.subarray(32, 48)], { topK: 1, nprobe: 8 });
		expect(many.map((r) => r.hits[0].id)).toEqual([0, 2]);
		index.close();
	});

	it("v1 legacy generation (no slabs table, no rev in current.json): slabs read whole, no phase 2", async () => {
		const b = await buildGen("g1", 1, { legacy: true });
		const index = await vib.open({ store: b.store, cache: false });
		expect(index.format).toBe(1);
		expect(index.revision.slabs).toBeNull();
		const res = await index.query(b.vectors.subarray(48, 64), { nprobe: 8 });
		expect(res.hits[0].id).toBe(3);
		expect(res.stats.rerank).toBe(0);
		expect(res.stats.slabBytes).toBeGreaterThan(res.stats.scanBytes - 1);
		expect(await recall(index, b, 20, 10, { nprobe: 8 })).toBeGreaterThanOrEqual(0.95);
		index.close();
	});

	it("upsert / delete: delta segments, tombstones, replace semantics, refresh on another handle", async () => {
		const b = await buildGen("gu", 3);
		const a = await vib.open({ store: b.store, cache: false });
		const other = await vib.open({ store: b.store, cache: false, refreshInterval: 0 });
		const r = rng(99);
		const fresh = unitRows(2, b.dim, r);
		const up = await a.upsert({ ids: [1000, 1001], vectors: fresh, attrs: [5, 6], payloads: ["p-1000", new TextEncoder().encode("p-1001")] });
		expect(up.rev).toBe(1);
		expect(up.seq).toBe(1);
		expect(up.count).toBe(2);
		expect(up.cells.length).toBe(2);
		expect(up.keys.delta).toBe("idx/gu/delta/1.vibd");
		expect(up.keys.tombstones).toBe("idx/gu/tombstones.1.u32");
		expect(up.currentUpdated).toBe(true);
		expect(a.rev).toBe(1);
		expect(a.manifest.deltas?.length).toBe(1);
		expect(a.manifest.tombstones?.length).toBe(1);
		expect(JSON.parse(new TextDecoder().decode((await b.store.get(keys.current()))!))).toEqual({ gen: "gu", rev: 1 });
		const q1 = await a.query(fresh.subarray(0, 16), { topK: 3, returnPayload: true });
		expect(q1.hits[0]).toMatchObject({ id: 1000, source: "delta", attrs: 5 });
		expect(new TextDecoder().decode(q1.hits[0].payload)).toBe("p-1000");
		expect(q1.stats.deltas).toBe(1);
		expect(q1.stats.deltaScanned).toBe(2);
		expect(q1.stats.deltaTop).toBeGreaterThanOrEqual(1);
		// the other handle still reads rev 0 until refreshed
		expect(other.rev).toBe(0);
		expect((await other.query(fresh.subarray(0, 16), { topK: 3 })).hits[0].id).not.toBe(1000);
		expect(await other.refresh()).toBe(true);
		expect(other.rev).toBe(1);
		expect((await other.query(fresh.subarray(0, 16), { topK: 3 })).hits[0].id).toBe(1000);
		expect(await other.refresh()).toBe(false);

		// replace id 3 (lives in rev-0 slabs) with a vector equal to fresh[1]: the slab copy is tombstoned
		const old3 = b.vectors.slice(48, 64);
		expect((await a.query(old3, { topK: 1, nprobe: 8 })).hits[0].id).toBe(3);
		const up2 = await a.upsert({ ids: [3], vectors: [Array.from(fresh.subarray(16, 32))] });
		expect(up2.rev).toBe(2);
		expect(up2.seq).toBe(2);
		const q3 = await a.query(old3, { topK: 5, nprobe: 8 });
		expect(q3.hits.map((h) => h.id)).not.toContain(3);
		expect(q3.stats.tombDropped).toBeGreaterThanOrEqual(1);
		const q4 = await a.query(fresh.subarray(16, 32), { topK: 3 });
		expect(q4.hits.slice(0, 2).map((h) => h.id).sort((x, y) => x - y)).toEqual([3, 1001]);
		expect(q4.hits[0].source).toBe("delta");

		// delete 1000: a tombstone set newer than the delta hides its entry
		const del = await a.delete([1000]);
		expect(del.rev).toBe(3);
		expect(del.count).toBe(1);
		const q5 = await a.query(fresh.subarray(0, 16), { topK: 3 });
		expect(q5.hits.map((h) => h.id)).not.toContain(1000);
		expect(q5.stats.deltaDropped).toBe(1);
		expect(q5.stats.tombSets).toBe(3);
		// a fresh open sees the head revision through current.json
		const c = await vib.open({ store: b.store, cache: false });
		expect(c.rev).toBe(3);
		expect(c.manifest.nextSeq).toBe(3);
		expect(c.manifest.change).toMatchObject({ op: "delete", count: 1 });
		// pinned open of rev 0 ignores everything
		const pinned = await vib.open({ store: b.store, cache: false, gen: "gu", rev: 0 });
		expect((await pinned.query(old3, { topK: 1, nprobe: 8 })).hits[0].id).toBe(3);
		expect(await pinned.refresh()).toBe(false);
		for (const ix of [a, other, c, pinned]) ix.close();
	});

	it("write conflicts: a lost manifest race is retried on the new head", async () => {
		const b = await buildGen("gc", 2);
		const a = await vib.open({ store: b.store, cache: false });
		// Another writer took rev 1 behind our back (current.json still says rev 0, as after a crash before the CAS).
		const head = JSON.parse(new TextDecoder().decode((await b.store.get(keys.manifest("gc", 0)))!)) as Manifest;
		await b.store.put(keys.manifest("gc", 1), JSON.stringify({ ...head, rev: 1, parentRev: 0, deltas: [], tombstones: [], nextSeq: 5 }));
		const del = await a.delete([1, 2]);
		expect(del.conflicts).toBe(1);
		expect(del.rev).toBe(2);
		expect(a.manifest.nextSeq).toBe(5);
		expect(a.manifest.parentRev).toBe(1);
		a.close();
	});

	it("wasm reset: the routing table is rebuilt and the query retried", async () => {
		const b = await buildGen("gr", 3);
		const index = await vib.open({ store: b.store, cache: false });
		const before = vib.resets;
		vib.reset("test");
		const res = await index.query(b.vectors.subarray(0, 16), { nprobe: 8 });
		expect(res.hits[0].id).toBe(0);
		expect(vib.resets).toBe(before + 1);
		let calls = 0;
		const out = await vib.withRecovery("t", async (attempt) => {
			calls++;
			if (attempt === 1) throw new WebAssembly.RuntimeError("unreachable");
			return (await index.query(b.vectors.subarray(16, 32), { nprobe: 8 })).hits[0].id;
		});
		expect(out).toBe(1);
		expect(calls).toBe(2);
		expect(vib.resets).toBe(before + 2);
		index.close();
	});
});
