/**
 * Warm packs: the partial-entry directory format (writer, parser, PackedBlobStore's view of a partial entry), building
 * one from a generation, the prefix layer, and `open({ warmPack })` seeding it in the background so queries read the
 * hot prefixes from memory with the same answers.
 */
import { describe, expect, it } from "vitest";
import { buildWarmPack, CachedBlobStore, keys, MemoryBlobStore, MemorySink, PackedBlobStore, PackWriter, parsePackDirectory, parsePackFooter, parseWarmPack, PACK_FOOTER_BYTES, seedWarmPack, slabRevOf, WarmPrefixStore, type BlobRange, type BlobStore, type GetOptions } from "../src/index.js";
import { buildGen, FakeCache, thread, unitRows, rng } from "./helpers.js";

const bytes = (n: number, seed = 0) => Uint8Array.from({ length: n }, (_, i) => (i * 31 + seed) & 0xff);

/** A store that records every read reaching it. */
class CountingStore implements BlobStore {
	reads: { key: string; range?: BlobRange; ranges?: BlobRange[] }[] = [];
	constructor(readonly inner: MemoryBlobStore) {}
	get(key: string, range?: BlobRange, opts?: GetOptions) {
		this.reads.push({ key, range });
		return this.inner.get(key, range, opts);
	}
	getRanges(key: string, ranges: BlobRange[], opts?: GetOptions) {
		this.reads.push({ key, ranges });
		return this.inner.getRanges(key, ranges, opts);
	}
	head(key: string) {
		return this.inner.head(key);
	}
	list(prefix: string) {
		return this.inner.list(prefix);
	}
}

describe("warm pack format", () => {
	it("marks partial entries with a fourth directory element; readers see the prefix and the whole size", async () => {
		const sink = new MemorySink();
		const w = new PackWriter(sink, { kind: "warm", gen: "g", rev: 1 });
		await w.add("idx/g/slab/00003.1", bytes(100, 3), { partialOf: 1000 });
		await w.add("whole", bytes(10, 1));
		await expect(w.add("bad", bytes(10), { partialOf: 5 })).rejects.toThrow(/more than its object/);
		await w.finish();
		const pack = sink.bytes();
		const f = parsePackFooter(pack.subarray(pack.byteLength - PACK_FOOTER_BYTES));
		const dirJson = JSON.parse(new TextDecoder().decode(pack.subarray(f.dirOff, f.dirOff + f.dirLen)));
		expect(dirJson.entries[0]).toEqual(["idx/g/slab/00003.1", 16, 100, { partial: true, size: 1000 }]);
		expect(dirJson.entries[1]).toHaveLength(3);
		expect(dirJson).toMatchObject({ kind: "warm", gen: "g", rev: 1 });
		const d = parsePackDirectory(pack.subarray(f.dirOff, f.dirOff + f.dirLen), f.dirOff, f.count);
		expect(d.entries.get("idx/g/slab/00003.1")).toEqual({ offset: 16, length: 100, size: 1000 });
		expect(d.entries.get("whole")!.size).toBeUndefined();

		// A PackedBlobStore serves ranges inside the prefix and refuses the rest (never a short object).
		const s = await PackedBlobStore.fromBytes(pack);
		expect(await s.get("idx/g/slab/00003.1", { offset: 10, length: 90 })).toEqual(bytes(100, 3).subarray(10));
		await expect(s.get("idx/g/slab/00003.1")).rejects.toThrow(/only its first 100 of 1000 bytes/);
		await expect(s.get("idx/g/slab/00003.1", { offset: 90, length: 20 })).rejects.toThrow(/only its first 100/);
		expect((await s.head("idx/g/slab/00003.1"))!.size).toBe(1000);
		expect(await s.get("whole")).toEqual(bytes(10, 1));

		const parsed = parseWarmPack(pack);
		expect(parsed.meta.kind).toBe("warm");
		expect(parsed.entries.map((e) => [e.key, e.bytes.byteLength, e.size])).toEqual([
			["idx/g/slab/00003.1", 100, 1000],
			["whole", 10, 10],
		]);
	});

	it("rejects a bad partial marker and a pack that is not a warm pack", async () => {
		const enc = new TextEncoder();
		const dir = enc.encode(JSON.stringify({ entries: [["a", 16, 8, { partial: true, size: 4 }]] }));
		expect(() => parsePackDirectory(dir, 24, 1)).toThrow(/size >= its length/);
		// an unknown fourth element is ignored (forward compatible)
		const other = parsePackDirectory(enc.encode(JSON.stringify({ entries: [["a", 16, 8, { something: 1 }]] })), 24, 1);
		expect(other.entries.get("a")).toEqual({ offset: 16, length: 8 });
		const sink = new MemorySink();
		const w = new PackWriter(sink, { gen: "g", rev: 1 });
		await w.add("a", bytes(8));
		await w.finish();
		expect(() => parseWarmPack(sink.bytes())).toThrow(/not a warm pack/);
		expect(() => parseWarmPack(bytes(64))).toThrow(/not a thread pack/);
	});
});

describe("prefix layer", () => {
	it("serves ranges inside a seeded prefix and passes everything else through", async () => {
		const m = new MemoryBlobStore();
		await m.put("k", bytes(1000, 7));
		const inner = new CountingStore(m);
		const layer = new WarmPrefixStore(inner);
		layer.seedPrefix("k", bytes(1000, 7).subarray(0, 100));
		const trace = { reads: 0, hits: 0, misses: 0, r206: 0, r200: 0, bytes: 0 };
		expect(await layer.get("k", { offset: 0, length: 100 }, { trace })).toEqual(bytes(1000, 7).subarray(0, 100));
		expect(await layer.getRanges!("k", [{ offset: 0, length: 4 }, { offset: 50, length: 4 }])).toEqual([bytes(1000, 7).subarray(0, 4), bytes(1000, 7).subarray(50, 54)]);
		expect(inner.reads).toHaveLength(0);
		expect(trace).toMatchObject({ reads: 0, hits: 1, bytes: 100 });
		// past the prefix, the whole object, a range list crossing the prefix: the store
		expect(await layer.get("k", { offset: 90, length: 20 })).toEqual(bytes(1000, 7).subarray(90, 110));
		expect(await layer.get("k")).toEqual(bytes(1000, 7));
		await layer.getRanges!("k", [{ offset: 0, length: 4 }, { offset: 500, length: 4 }]);
		expect(inner.reads.map((r) => r.key)).toEqual(["k", "k", "k"]);
		expect(layer.hits).toBe(2);
		layer.retain(() => false);
		expect(layer.size).toEqual({ prefixes: 0, bytes: 0 });
		expect(layer.put).toBeUndefined();
		// no getRanges on the inner store: none on the layer either
		expect(new WarmPrefixStore({ get: m.get.bind(m), head: m.head.bind(m) }).getRanges).toBeUndefined();
	});
});

describe("prefix layer in front of a cache", () => {
	it("a hit of a fill read fetches the whole object into the cache in the background, once", async () => {
		const m = new MemoryBlobStore();
		await m.put("idx/g/slab/00001", bytes(1000, 5));
		const inner = new CountingStore(m);
		const cache = new FakeCache();
		const pending: Promise<unknown>[] = [];
		const layer = new WarmPrefixStore(new CachedBlobStore(inner, cache), { waitUntil: (p) => pending.push(p) });
		layer.seedPrefix("idx/g/slab/00001", bytes(1000, 5).subarray(0, 100));
		expect(await layer.get("idx/g/slab/00001", { offset: 0, length: 100 }, { fill: true })).toEqual(bytes(1000, 5).subarray(0, 100));
		await layer.get("idx/g/slab/00001", { offset: 0, length: 100 }, { fill: true });
		await layer.get("idx/g/slab/00001", { offset: 0, length: 50 }); // no fill asked: nothing queued
		await Promise.all(pending);
		await Promise.all(pending);
		expect(layer.fills).toEqual({ queued: 1, done: 1, failed: 0 });
		expect(inner.reads).toEqual([{ key: "idx/g/slab/00001", range: undefined }]);
		expect(cache.puts).toBe(1);
		// later phase-2 style reads of that slab are cache hits
		const trace = { reads: 0, hits: 0, misses: 0, r206: 0, r200: 0, bytes: 0 };
		expect(await layer.get("idx/g/slab/00001", { offset: 500, length: 10 }, { trace })).toEqual(bytes(1000, 5).subarray(500, 510));
		expect(trace).toMatchObject({ reads: 0, hits: 1 });
	});
});

describe("warm packs end to end", () => {
	it("builds the prefixes of chosen cells; open({ warmPack }) seeds them and queries skip those prefix reads", async () => {
		const built = await buildGen("w3", 3, { k2: 8, n: 600 });
		const rev = (await thread.open({ store: built.store, cache: false })).revision;
		const hot = [5, 1, 6, 2];
		const { bytes: wp, result } = await buildWarmPack(built.store, rev, hot, { meta: { source: "test" } });
		expect(result.keys).toEqual(hot.map((c) => keys.slab("w3", c, slabRevOf(rev.slabs![c]))));
		const parsed = parseWarmPack(wp);
		expect(parsed.meta).toMatchObject({ kind: "warm", gen: "w3", rev: 0, cells: 4, source: "test" });
		for (const [i, e] of parsed.entries.entries()) {
			const c = hot[i];
			expect(e.bytes.byteLength).toBe(rev.slabs![c][0]);
			expect(e.size).toBe(rev.slabs![c][3]);
			expect(e.bytes).toEqual((await built.store.get(e.key))!.subarray(0, rev.slabs![c][0]));
		}
		// a stale entry (another slab revision of a cell) is not seeded
		const stale = new MemorySink();
		const sw = new PackWriter(stale, { kind: "warm", gen: "w3", rev: 0 });
		for (const e of parsed.entries) await sw.add(e.key, e.bytes, { partialOf: e.size });
		await sw.add(keys.slab("w3", 0, 99), bytes(64), { partialOf: 1000 });
		await sw.finish();
		await built.store.put("warm/w3.threadpack", stale.bytes());

		const counting = new CountingStore(built.store);
		const plain = await thread.open({ store: built.store, cache: false });
		const index = await thread.open({ store: counting, cache: false, warmPack: "warm/w3.threadpack" });
		expect(index.warmStats).toMatchObject({ key: "warm/w3.threadpack" });
		const st = await index.warmReady;
		expect(st).toMatchObject({ state: "ready", entries: 5, seeded: 4, skipped: 1 });
		expect(st!.seededBytes).toBe(hot.reduce((a, c) => a + rev.slabs![c][0], 0));
		expect(index.warm!.size.prefixes).toBe(4);

		const qs = unitRows(6, built.dim, rng(9));
		let warmHits = 0;
		let rerankHot = 0;
		for (let i = 0; i < 6; i++) {
			const q = qs.subarray(i * built.dim, (i + 1) * built.dim);
			counting.reads = [];
			const a = await plain.query(q, { nprobe: 4, rerank: 30 });
			const b = await index.query(q, { nprobe: 4, rerank: 30 });
			expect(b.hits.map((h) => h.id)).toEqual(a.hits.map((h) => h.id));
			expect(b.hits.map((h) => h.score)).toEqual(a.hits.map((h) => h.score));
			const hotProbed = b.stats.cells.filter((c) => hot.includes(c));
			warmHits += hotProbed.length;
			expect(b.stats.timeline.phase1.hits).toBe(hotProbed.length);
			// no prefix read of a hot cell reached the store; every cold probed cell's prefix did
			const prefixReads = counting.reads.filter((r) => r.range && r.range.offset === 0);
			for (const c of b.stats.cells) {
				const key = keys.slab("w3", c, slabRevOf(rev.slabs![c]));
				expect(prefixReads.filter((r) => r.key === key).length).toBe(hot.includes(c) ? 0 : 1);
			}
			// phase 2 of a hot cell's winners still reads the store (the pack holds prefixes only)
			const hotKeys = new Set(hot.map((c) => keys.slab("w3", c, slabRevOf(rev.slabs![c]))));
			rerankHot += counting.reads.filter((r) => hotKeys.has(r.key) && (r.ranges || (r.range && r.range.offset > 0))).length;
		}
		expect(warmHits).toBeGreaterThan(0);
		expect(rerankHot).toBeGreaterThan(0);
		index.close();
		plain.close();
	});

	it("a missing or broken warm pack never fails the open or a query", async () => {
		const built = await buildGen("w4", 3, { k2: 8, n: 300 });
		const index = await thread.open({ store: built.store, cache: false, warmPack: "warm/nope.threadpack" });
		const q = unitRows(1, built.dim, rng(3));
		expect((await index.query(q)).hits.length).toBeGreaterThan(0);
		expect(await index.warmReady).toMatchObject({ state: "failed" });
		expect(index.warmStats!.error).toMatch(/no warm pack/);
		await built.store.put("warm/bad.threadpack", bytes(200));
		const bad = await thread.open({ store: built.store, cache: false, warmPack: "warm/bad.threadpack" });
		expect(await bad.warmReady).toMatchObject({ state: "failed" });
		expect((await bad.query(q)).hits.length).toBeGreaterThan(0);
		// no warm pack: nothing to wait for
		const none = await thread.open({ store: built.store, cache: false });
		expect(none.warm).toBeNull();
		expect(await none.warmReady).toBeNull();
	});

	it("seedWarmPack seeds any PrefixSeed and filters with accept", async () => {
		const sink = new MemorySink();
		const w = new PackWriter(sink, { kind: "warm", gen: "g", rev: 1 });
		await w.add("a", bytes(8, 1), { partialOf: 80 });
		await w.add("b", bytes(8, 2), { partialOf: 80 });
		await w.finish();
		const seen: string[] = [];
		const r = seedWarmPack(sink.bytes(), { seedPrefix: (k, b) => seen.push(`${k}:${b.byteLength}`) }, (k) => k === "b");
		expect(seen).toEqual(["b:8"]);
		expect(r).toMatchObject({ entries: 2, seeded: 1, skipped: 1, seededBytes: 8 });
	});
});
