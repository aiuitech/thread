/**
 * End-to-end in Node: builds tiny v1/v2/v3 generations with the wasm build jobs into a MemoryBlobStore, then
 * queries them through the SDK (brute-force recall), upserts/deletes with tombstones, refreshes and resets.
 */
import { describe, expect, it } from "vitest";
import { keys, ThreadIndex, type Manifest } from "../src/index.js";
import { bruteForce, buildGen, rng, thread, unitRows, type Built } from "./helpers.js";

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

describe("attribute filters (dense)", () => {
	/** Brute-force top-k among the vectors whose index passes `keep`. */
	const truthOf = (b: Built, q: Float32Array, k: number, keep: (i: number) => boolean) => bruteForce(b.vectors, b.dim, q, b.vectors.length / b.dim).filter(keep).slice(0, k);

	it("filters inside the slab and delta scans, widens the probes, and keeps recall among the matching entries", async () => {
		const b = await buildGen("gf", 3, { attrs: (i) => i % 4 });
		const index = await thread.open({ store: b.store, cache: false });
		expect(index.attrSchema).toBeNull();
		// unfiltered: params and stats keep their shape, no widening
		const plain = await index.query(b.vectors.subarray(0, 16), { nprobe: 3 });
		expect(plain.params).toEqual({ topK: 10, nprobe: 3, rerank: 20, pc: 0, source: { nprobe: "query", rerank: "manifest", pc: "manifest" } });
		expect(plain.stats.cells.length).toBe(3);
		expect(plain.stats.filterDropped).toBe(0);
		expect(plain.hits.map((h) => h.attrs)).toEqual(plain.hits.map((h) => h.id % 4));
		// every cell probed: the filtered result is the brute-force top-k among the matching class
		const r = rng(3);
		let hit = 0;
		for (let i = 0; i < 20; i++) {
			const q = unitRows(1, b.dim, r);
			const cls = i % 4;
			const res = await index.query(q, { topK: 10, nprobe: 8, rerank: 40, filter: { attrs: cls } });
			expect(res.hits.length).toBe(10);
			expect(res.hits.every((h) => h.attrs === cls && h.id % 4 === cls)).toBe(true);
			expect(res.params).toMatchObject({ filter: true, maxNprobe: 8 });
			expect(res.stats.filterDropped).toBeGreaterThan(0);
			const truth = new Set(truthOf(b, q, 10, (j) => j % 4 === cls));
			hit += res.hits.filter((h) => truth.has(h.id)).length;
		}
		expect(hit / 200).toBeGreaterThanOrEqual(0.9);
		// widening: nprobe 2 is not enough for T = 40 matching candidates of one class, so more cells are scanned in
		// routing order (step 4) until it is, bounded by maxNprobe (default 4 x nprobe, here every cell)
		const q = unitRows(1, b.dim, rng(5));
		const wide = await index.query(q, { topK: 5, nprobe: 2, rerank: 40, filter: { attrs: { $in: [1] } } });
		expect(wide.params.maxNprobe).toBe(8);
		expect(wide.stats.cells.length).toBeGreaterThan(2);
		expect(wide.stats.cells.length === 8 || wide.stats.candidates >= 40).toBe(true);
		expect(wide.hits.every((h) => h.attrs === 1)).toBe(true);
		const capped = await index.query(q, { topK: 5, nprobe: 2, rerank: 40, filter: { attrs: { $in: [1] } }, maxNprobe: 2 });
		expect(capped.stats.cells.length).toBe(2);
		expect(capped.stats.cells).toEqual(wide.stats.cells.slice(0, 2));
		await expect(index.query(q, { nprobe: 4, filter: { attrs: 1 }, maxNprobe: 3 })).rejects.toThrow(/maxNprobe/);
		// a filter nothing matches scans up to the bound and returns nothing
		const none = await index.query(q, { nprobe: 2, filter: { attrs: 9 } });
		expect(none.hits).toEqual([]);
		expect(none.stats.cells.length).toBe(8);
		// compiled words are accepted as is; an unknown field is rejected
		const words = index.compileFilter({ attrs: { $gte: 2 } });
		const hi = await index.query(q, { nprobe: 8, filter: words });
		expect(hi.hits.every((h) => h.attrs >= 2)).toBe(true);
		await expect(index.query(q, { filter: { tenant: 1 } })).rejects.toThrow(/no attribute schema/);
		// delta entries are filtered too: an upsert with attrs, then the filter decides which copy wins
		const fresh = unitRows(2, b.dim, rng(9));
		await index.upsert({ ids: [5000, 5001], vectors: fresh, attrs: [7, 2] });
		const d7 = await index.query(fresh.subarray(0, 16), { topK: 3, nprobe: 8, filter: { attrs: 7 } });
		expect(d7.hits.map((h) => [h.id, h.source])).toEqual([[5000, "delta"]]);
		const d2 = await index.query(fresh.subarray(16, 32), { topK: 3, nprobe: 8, filter: { attrs: { $in: [2] } } });
		expect(d2.hits[0]).toMatchObject({ id: 5001, source: "delta", attrs: 2 });
		expect(d2.hits.slice(1).every((h) => h.source === "slab" && h.attrs === 2)).toBe(true);
		const d0 = await index.query(fresh.subarray(0, 16), { topK: 3, nprobe: 8, filter: { attrs: { $ne: 7 } } });
		expect(d0.hits.map((h) => h.id)).not.toContain(5000);
		expect(d0.stats.filterDropped).toBeGreaterThanOrEqual(1);
		index.close();
	});

	it("schema: named fields packed into the word, filtered by field, unpacked from hits", async () => {
		const b = await buildGen("gs", 2);
		const index = await thread.open({ store: b.store, cache: false });
		const cfg = await index.configureAttrs({ tenant: { bits: 12 }, lang: { values: ["en", "fr"] }, public: "bool" });
		expect(cfg.rev).toBe(1);
		expect(index.attrSchema?.fields.map((f) => [f.name, f.offset, f.bits])).toEqual([
			["tenant", 0, 12],
			["lang", 12, 2],
			["public", 14, 1],
		]);
		expect(index.packAttrs({ tenant: 5, lang: "fr", public: true })).toBe(5 + (2 << 12) + (1 << 14));
		const fresh = unitRows(4, b.dim, rng(21));
		await index.upsert({ ids: [100, 101, 102, 103], vectors: fresh, attrs: [{ tenant: 5, lang: "en", public: true }, { tenant: 5, lang: "fr" }, { tenant: 6, lang: "en", public: true }, 0] });
		const q = fresh.subarray(0, 16);
		const t5 = await index.query(q, { topK: 10, nprobe: 8, filter: { tenant: 5 } });
		expect(t5.hits.map((h) => h.id).sort()).toEqual([100, 101]);
		expect(index.unpackAttrs(t5.hits[0].attrs)).toEqual({ tenant: 5, lang: "en", public: true });
		const pub = await index.query(q, { topK: 10, nprobe: 8, filter: { public: true, lang: { $in: ["en", "de"] } } });
		expect(pub.hits.map((h) => h.id).sort()).toEqual([100, 102]);
		const unset = await index.query(q, { topK: 10, nprobe: 8, filter: { tenant: { $lt: 1 } } });
		expect(unset.hits.every((h) => h.attrs === 0)).toBe(true); // the base vectors and id 103
		await expect(index.upsert({ ids: [104], vectors: fresh.subarray(0, 16), attrs: [{ lang: "it" }] })).rejects.toThrow(/not a valid enum/);
		await expect(index.query(q, { filter: { lang: { $gt: 1 } } })).rejects.toThrow(/number field/);
		// the schema may grow but not change: existing fields keep their layout, an enum appends within its width
		await expect(index.configureAttrs({ tenant: { bits: 8 } })).rejects.toThrow(/layout/);
		await expect(index.configureAttrs({ lang: { values: ["en", "fr", "de", "it"] } })).rejects.toThrow(/fit/);
		await expect(index.configureAttrs({ fields: [{ name: "tenant", type: "number", offset: 0, bits: 8 }] })).rejects.toThrow(/only add/);
		const grown = await index.configureAttrs({ lang: { values: ["en", "fr", "de"] }, year: { bits: 7, min: 1970 } });
		expect(grown.rev).toBe(3);
		expect(index.attrSchema?.fields.map((f) => [f.name, f.offset, f.bits])).toEqual([
			["tenant", 0, 12],
			["lang", 12, 2],
			["public", 14, 1],
			["year", 15, 7],
		]);
		expect(index.packAttrs({ lang: "de", year: 2001 })).toBe((3 << 12) + (31 << 15));
		expect((await index.query(q, { topK: 10, nprobe: 8, filter: { tenant: 5 } })).hits.map((h) => h.id).sort()).toEqual([100, 101]);
		index.close();
	});
});
