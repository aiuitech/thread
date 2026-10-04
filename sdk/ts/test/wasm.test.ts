/**
 * End-to-end in Node: builds tiny v1/v2/v3 generations with the wasm build jobs into a MemoryBlobStore, then
 * queries them through the SDK (brute-force recall), upserts/deletes with tombstones, refreshes and resets.
 */
import { describe, expect, it } from "vitest";
import { keys, ThreadIndex, type Manifest } from "../src/index.js";
import { bruteForce, buildGen, FakeCache, rng, thread, unitRows, type Built } from "./helpers.js";

async function recall(index: ThreadIndex, b: Built, nq: number, k: number, qopts: Parameters<ThreadIndex["query"]>[1] = {}): Promise<number> {
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
		const index = await thread.open({ store: b.store, cache: false });
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
		const index = await thread.open({ store: b.store, cache: false });
		expect(index.format).toBe(2);
		expect(await recall(index, b, 20, 10, { nprobe: 8 })).toBeGreaterThanOrEqual(0.95);
		const many = await index.queryMany([b.vectors.subarray(0, 16), b.vectors.subarray(32, 48)], { topK: 1, nprobe: 8 });
		expect(many.map((r) => r.hits[0].id)).toEqual([0, 2]);
		index.close();
	});

	it("v1 legacy generation (no slabs table, no rev in current.json): slabs read whole, no phase 2", async () => {
		const b = await buildGen("g1", 1, { legacy: true });
		const index = await thread.open({ store: b.store, cache: false });
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
		const a = await thread.open({ store: b.store, cache: false });
		const other = await thread.open({ store: b.store, cache: false, refreshInterval: 0 });
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
		// the pointer carries a copy of the manifest it names
		const cur = JSON.parse(new TextDecoder().decode((await b.store.get(keys.current()))!));
		expect(cur).toMatchObject({ gen: "gu", rev: 1 });
		expect(cur.manifest).toEqual(JSON.parse(new TextDecoder().decode((await b.store.get(keys.manifest("gu", 1)))!)));
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
		const c = await thread.open({ store: b.store, cache: false });
		expect(c.rev).toBe(3);
		expect(c.manifest.nextSeq).toBe(3);
		expect(c.manifest.change).toMatchObject({ op: "delete", count: 1 });
		// pinned open of rev 0 ignores everything
		const pinned = await thread.open({ store: b.store, cache: false, gen: "gu", rev: 0 });
		expect((await pinned.query(old3, { topK: 1, nprobe: 8 })).hits[0].id).toBe(3);
		expect(await pinned.refresh()).toBe(false);
		for (const ix of [a, other, c, pinned]) ix.close();
	});

	it("write conflicts: a lost manifest race is retried on the new head", async () => {
		const b = await buildGen("gc", 2);
		const a = await thread.open({ store: b.store, cache: false });
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

	it("cold-start timeline: openStats and stats.timeline account every read", async () => {
		const b = await buildGen("gt", 3);
		const index = await thread.open({ store: b.store, cache: false });
		const os = index.openStats;
		expect(os.wasmInit).toBe(thread.initMs);
		expect(os.current).toMatchObject({ reads: 1, hits: 0, misses: 0 });
		expect(os.current.bytes).toBe(b.store.objects.get(keys.current())!.bytes.byteLength);
		expect(os.manifest).toMatchObject({ reads: 1, bytes: b.store.objects.get(keys.manifest("gt", 0))!.bytes.byteLength });
		expect(os.slabs.reads).toBe(1); // slabs table, no tombstone sets yet
		const base = os.base!;
		expect(base.reads).toBe(3); // coarse, fine, pq
		expect(base.files.pq).toBeGreaterThan(0);
		expect(base.bytes).toBe(base.files.coarse + base.files.fine + base.files.pq);
		expect(base.fetch + base.build).toBeLessThanOrEqual(base.ms + 0.01);
		for (const p of [os.current, os.manifest, os.slabs, base]) expect(p.ms).toBeGreaterThanOrEqual(0);
		expect(os.total).toBeGreaterThanOrEqual(os.current.ms + os.manifest.ms + os.slabs.ms + base.ms - 0.01);

		const res = await index.query(b.vectors.subarray(0, 16), { nprobe: 4 });
		const tl = res.stats.timeline;
		expect(tl.base).toBeUndefined();
		expect(tl.route.reads).toBe(0);
		expect(tl.phase1.reads).toBeGreaterThanOrEqual(4); // one scan-prefix read per probed slab (+ speculative reads)
		const sum = (f: "reads" | "bytes") => tl.route[f] + tl.phase1[f] + tl.phase2[f] + tl.payload[f];
		expect(sum("reads")).toBe(res.stats.io.reads);
		expect(sum("bytes")).toBe(res.stats.io.bytes);
		expect(tl.phase1.ms).toBe(res.stats.timing.phase1);
		expect(tl.phase2.ms).toBe(res.stats.timing.phase2);

		// A delete adds a tombstone set: a fresh open reads it in the `slabs` step.
		await index.delete([5]);
		const again = await thread.open({ store: b.store, cache: false });
		expect(again.openStats.slabs.reads).toBe(2);
		// After a wasm reset the next query reloads the routing table and reports it.
		thread.reset("test");
		const r2 = await again.query(b.vectors.subarray(0, 16), { nprobe: 4 });
		expect(r2.stats.timeline.base?.reads).toBe(3);
		index.close();
		again.close();

		// Behind a cache: a cold open misses and fills, a second open is all cache hits.
		const cache = new FakeCache();
		const c1 = await thread.open({ store: b.store, cache });
		expect(c1.openStats.base!.misses).toBe(3);
		expect(c1.openStats.base!.reads).toBe(3);
		await new Promise((r) => setTimeout(r, 0));
		const c2 = await thread.open({ store: b.store, cache });
		expect(c2.openStats.current.reads).toBe(1); // current.json is never cached
		expect(c2.openStats.base).toMatchObject({ reads: 0, misses: 0, hits: 3 });
		c1.close();
		c2.close();
	});

	it("wasm reset: the routing table is rebuilt and the query retried", async () => {
		const b = await buildGen("gr", 3);
		const index = await thread.open({ store: b.store, cache: false });
		const before = thread.resets;
		thread.reset("test");
		const res = await index.query(b.vectors.subarray(0, 16), { nprobe: 8 });
		expect(res.hits[0].id).toBe(0);
		expect(thread.resets).toBe(before + 1);
		let calls = 0;
		const out = await thread.withRecovery("t", async (attempt) => {
			calls++;
			if (attempt === 1) throw new WebAssembly.RuntimeError("unreachable");
			return (await index.query(b.vectors.subarray(16, 32), { nprobe: 8 })).hits[0].id;
		});
		expect(out).toBe(1);
		expect(calls).toBe(2);
		expect(thread.resets).toBe(before + 2);
		index.close();
	});
});
