/**
 * Cross-slab coalescing (`coalesceAcrossSlabs`): phase-1 prefix reads of slabs close together in a pack become one
 * range read of the pack, with the same answers; a warm-pack layer in front keeps its prefixes out of the plan; other
 * stores are untouched.
 */
import { describe, expect, it } from "vitest";
import { buildWarmPack, keys, MemoryBlobStore, PackedBlobStore, packToBytes, planCrossSlab, slabRevOf, type BlobRange, type BlobStore, type GetOptions } from "../src/index.js";
import { buildGen, rng, thread, unitRows } from "./helpers.js";

/** Counts reads of the pack object. */
class Counting implements BlobStore {
	reads: BlobRange[] = [];
	constructor(readonly inner: MemoryBlobStore) {}
	get(key: string, range?: BlobRange, opts?: GetOptions) {
		if (range) this.reads.push(range);
		return this.inner.get(key, range, opts);
	}
	head(key: string) {
		return this.inner.head(key);
	}
}

describe("coalesceAcrossSlabs", () => {
	it("merges prefix reads of nearby slabs in a pack into one range, same answers", async () => {
		const built = await buildGen("c3", 3, { k2: 8, n: 600 });
		const { bytes } = await packToBytes(built.store);
		const m = new MemoryBlobStore();
		await m.put("p.threadpack", bytes);
		const counting = new Counting(m);
		const pack = await PackedBlobStore.open(counting, "p.threadpack");
		const index = await thread.open({ store: pack, cache: false });
		const qs = unitRows(4, built.dim, rng(5));
		for (let i = 0; i < 4; i++) {
			const q = qs.subarray(i * built.dim, (i + 1) * built.dim);
			counting.reads = [];
			const a = await index.query(q, { nprobe: 5, rerank: 30 });
			const plainReads = counting.reads.length;
			expect(a.stats.coalesced).toBeUndefined();
			counting.reads = [];
			// a gap larger than the pack: all five prefixes in one read
			const b = await index.query(q, { nprobe: 5, rerank: 30, coalesceAcrossSlabs: 1 << 30 });
			expect(b.hits.map((h) => [h.id, h.score])).toEqual(a.hits.map((h) => [h.id, h.score]));
			expect(b.stats.coalesced).toMatchObject({ reads: 1, slabs: 5 });
			expect(b.stats.timeline.phase1.reads + b.stats.timeline.phase2.reads + b.stats.timeline.payload.reads).toBe(b.stats.io.reads);
			expect(counting.reads.length).toBe(plainReads - 4);
			// gap 0 in key order: only slabs that touch merge (8-byte alignment padding is within any gap >= 7)
			const c = await index.query(q, { nprobe: 5, rerank: 30, coalesceAcrossSlabs: 0 });
			expect(c.hits.map((h) => h.id)).toEqual(a.hits.map((h) => h.id));
			expect(c.stats.coalesced!.reads).toBeLessThanOrEqual(2);
		}
		index.close();
	});

	it("leaves warm-pack prefixes out of the plan and ignores stores that are not packs", async () => {
		const built = await buildGen("c4", 3, { k2: 8, n: 400 });
		const { bytes } = await packToBytes(built.store);
		const m = new MemoryBlobStore();
		await m.put("p.threadpack", bytes);
		const plain = await thread.open({ store: built.store, cache: false });
		const rev = plain.revision;
		const q = unitRows(1, built.dim, rng(7));
		const ref = await plain.query(q, { nprobe: 6, rerank: 30 });
		const hot = ref.stats.cells.slice(0, 2);
		const { bytes: wp } = await buildWarmPack(built.store, rev, hot);
		await m.put("warm.threadpack", wp);
		const pack = await PackedBlobStore.open(m, "p.threadpack");
		const index = await thread.open({ store: pack, cache: false, warmPack: "warm.threadpack" });
		await index.warmReady;
		expect(index.warmStats!.seeded).toBe(2);
		const plan = planCrossSlab(index.store, rev, "c4", ref.stats.cells, 1 << 30)!;
		const key = (c: number) => keys.slab("c4", c, slabRevOf(rev.slabs![c]));
		for (const c of hot) expect(plan.has(c)).toBe(false);
		expect(new Set(plan.values()).size).toBe(1);
		expect(plan.size).toBe(4);
		expect(key(hot[0])).toContain("/slab/");
		const r = await index.query(q, { nprobe: 6, rerank: 30, coalesceAcrossSlabs: true });
		expect(r.hits.map((h) => [h.id, h.score])).toEqual(ref.hits.map((h) => [h.id, h.score]));
		expect(r.stats.timeline.phase1.hits).toBe(2);
		// not a pack: no plan, no stats
		expect(planCrossSlab(built.store, rev, "c4", ref.stats.cells, 1 << 30)).toBeNull();
		expect((await plain.query(q, { nprobe: 6, rerank: 30, coalesceAcrossSlabs: true })).stats.coalesced).toBeUndefined();
	});
});
