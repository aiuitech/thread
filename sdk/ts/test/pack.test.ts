/**
 * Packs: the file format (writer, footer, directory), PackedBlobStore over memory, a snapshot of a built generation
 * with deltas, tombstones and lexical segments queried through the pack with the same answers as the source store,
 * a cache in front of a pack, read-only writes, and copying a pack back out into a writable store.
 */
import { describe, expect, it } from "vitest";
import { CachedBlobStore, copyObjects, MemoryBlobStore, MemorySink, PACK_FOOTER_BYTES, PACK_HEADER_BYTES, PackedBlobStore, packFooter, packIndex, packToBytes, PackWriter, parsePackFooter, snapshotKeys, ThreadError, type BlobStore, type SearchResult } from "../src/index.js";
import { buildGen, FakeCache, rng, thread, unitRows } from "./helpers.js";

const enc = new TextEncoder();
const bytes = (n: number, seed = 0) => Uint8Array.from({ length: n }, (_, i) => (i * 31 + seed) & 0xff);

describe("pack format", () => {
	it("writes aligned entries, a JSON directory and a footer; reads them back", async () => {
		const sink = new MemorySink();
		const w = new PackWriter(sink, { gen: "g", rev: 3 });
		await w.add("idx/current.json", enc.encode('{"gen":"g","rev":3}'));
		await w.add("idx/g/slab/00001", bytes(1001, 1));
		await w.add("empty", new Uint8Array(0));
		await w.add("idx/g/slab/00000", bytes(17, 2));
		await expect(w.add("empty", bytes(1))).rejects.toThrow(/added twice/);
		const { bytes: size, count } = await w.finish();
		expect(count).toBe(4);
		const pack = sink.bytes();
		expect(pack.byteLength).toBe(size);
		expect(new TextDecoder().decode(pack.subarray(0, 8))).toBe("THRDPACK");
		expect(new TextDecoder().decode(pack.subarray(size - 8))).toBe("THRDPACK");
		const footer = parsePackFooter(pack.subarray(size - PACK_FOOTER_BYTES));
		expect(footer.count).toBe(4);
		expect(JSON.parse(new TextDecoder().decode(pack.subarray(footer.dirOff, footer.dirOff + footer.dirLen)))).toMatchObject({ gen: "g", rev: 3 });

		const s = await PackedBlobStore.fromBytes(pack);
		expect(s.gen).toBe("g");
		expect(s.rev).toBe(3);
		expect(await s.list("")).toEqual(["empty", "idx/current.json", "idx/g/slab/00000", "idx/g/slab/00001"]);
		expect(await s.list("idx/g/")).toEqual(["idx/g/slab/00000", "idx/g/slab/00001"]);
		for (const e of s.directory.entries.values()) expect(e.offset % 8).toBe(0);
		expect(s.directory.entries.get("idx/current.json")!.offset).toBe(PACK_HEADER_BYTES);
		expect(await s.get("idx/g/slab/00001")).toEqual(bytes(1001, 1));
		expect(await s.get("idx/g/slab/00001", { offset: 1000, length: 1 })).toEqual(bytes(1001, 1).subarray(1000));
		expect(await s.getRanges("idx/g/slab/00000", [{ offset: 0, length: 2 }, { offset: 15, length: 2 }])).toEqual([bytes(17, 2).subarray(0, 2), bytes(17, 2).subarray(15)]);
		expect((await s.get("empty"))!.byteLength).toBe(0);
		expect(await s.get("nope")).toBeNull();
		expect(await s.head("nope")).toBeNull();
		expect((await s.head("idx/g/slab/00001"))!.size).toBe(1001);
		await expect(s.get("idx/g/slab/00000", { offset: 10, length: 10 })).rejects.toThrow(/outside the object/);
		expect((s as BlobStore).put).toBeUndefined();
	});

	it("rejects things that are not packs", async () => {
		await expect(PackedBlobStore.fromBytes(bytes(10))).rejects.toThrow(/too small/);
		await expect(PackedBlobStore.fromBytes(bytes(100))).rejects.toThrow(/bad footer magic/);
		const bad = packFooter(16, 10, 1);
		bad[20] = 9; // version
		await expect(PackedBlobStore.fromBytes(new Uint8Array([...bytes(40), ...bad]))).rejects.toThrow(/unknown pack version 9/);
		// a footer whose directory lies outside the pack
		const sink = new MemorySink();
		const w = new PackWriter(sink);
		await w.add("a", bytes(5));
		await w.finish();
		const p = sink.bytes();
		const f = packFooter(p.byteLength, 10, 1);
		p.set(f, p.byteLength - PACK_FOOTER_BYTES);
		await expect(PackedBlobStore.fromBytes(p)).rejects.toThrow(/outside the pack/);
		const m = new MemoryBlobStore();
		await expect(PackedBlobStore.open(m, "missing")).rejects.toThrow(/no pack at missing/);
	});
});

describe("snapshots", () => {
	it("snapshotKeys lists what a revision reads, and nothing from other revisions", async () => {
		const b = await buildGen("gs", 3);
		const index = await thread.open({ store: b.store, cache: false });
		const k0 = snapshotKeys(index.revision);
		expect(k0).toEqual(["idx/gs/coarse.f32", "idx/gs/fine.f32", "idx/gs/manifest.json", "idx/gs/pq.bin", ...Array.from({ length: 8 }, (_, c) => `idx/gs/slab/0000${c}`), "idx/gs/slabs.json"].sort());
		await index.configureLexical({ text: true });
		await index.upsert({ ids: [900, 901], vectors: unitRows(2, b.dim, rng(5)), texts: ["nine hundred", "nine oh one"], metas: ["a", "b"] });
		await index.delete([3]);
		const k = snapshotKeys(index.revision);
		expect(k).toContain("idx/gs/manifest.3.json");
		expect(k.filter((x) => x.startsWith("idx/gs/delta/")).length).toBe(1);
		expect(k.filter((x) => x.startsWith("idx/gs/seg/")).length).toBe(1);
		expect(k.filter((x) => x.startsWith("idx/gs/tombstones.")).length).toBe(2);
		expect(k).not.toContain("idx/gs/manifest.json");
		expect(k).not.toContain("idx/gs/manifest.1.json");
		expect(k).not.toContain("idx/current.json");
		// the legacy v1 layout: no slabs table, rev-0 slabs
		const l = await buildGen("gl", 1, { legacy: true });
		const li = await thread.open({ store: l.store, cache: false });
		expect(snapshotKeys(li.revision)).toEqual(["idx/gl/coarse.f32", "idx/gl/fine.f32", "idx/gl/manifest.json", ...Array.from({ length: 8 }, (_, c) => `idx/gl/slab/0000${c}`)].sort());
		index.close();
		li.close();
	});

	it("PackedBlobStore.open records its reads, and thread.open reports them as openStats.pack", async () => {
		const b = await buildGen("go", 3, { n: 200 });
		const { bytes: pack } = await packToBytes(b.store);
		const ps = await PackedBlobStore.fromBytes(pack);
		// one suffix-range read (getTail): footer and directory in one request, no head
		expect(ps.openStats).toMatchObject({ reads: 1, bytes: Math.min(pack.byteLength, 512 * 1024) });
		// a store without getTail: head (counted as a read of 0 bytes) + one tail read
		const m = new MemoryBlobStore();
		m.objects.set("p", { bytes: pack, etag: "e" });
		const noTail = await PackedBlobStore.open({ get: (k, r, o) => m.get(k, r, o), head: (k) => m.head(k) }, "p");
		expect(noTail.openStats).toMatchObject({ reads: 2, bytes: Math.min(pack.byteLength, 512 * 1024) });
		expect([...noTail.directory.entries]).toEqual([...ps.directory.entries]);
		// a tail read too short for the directory takes one more range read
		const short = await PackedBlobStore.open(m, "p", 64);
		expect(short.openStats.reads).toBe(2);
		expect([...short.directory.entries]).toEqual([...ps.directory.entries]);
		const idx = await thread.open({ store: ps, cache: false });
		expect(idx.openStats.pack).toEqual(ps.openStats);
		// the pack's base range (idx/current.json first, then the base objects) is one range read, before anything else
		expect(idx.openStats.prefetch).toMatchObject({ source: "index-pack", reads: 1 });
		expect(idx.openStats.current.reads).toBe(0);
		expect(idx.openStats.base!.reads).toBe(0);
		idx.close();
	});

	it("a pack answers like the store it was taken from, is a pinned snapshot, and is read-only", async () => {
		const b = await buildGen("gp", 3, { n: 300 });
		const src = await thread.open({ store: b.store, cache: false });
		await src.configureLexical({ text: true, bucketTerms: 8 });
		const fresh = unitRows(20, b.dim, rng(3));
		const ids = Array.from({ length: 20 }, (_, i) => 5000 + i);
		await src.upsert({ ids, vectors: fresh, texts: ids.map((i) => `fresh vector ${i} ${i % 2 ? "odd" : "even"}`), metas: ids.map((i) => `m${i}`), attrs: ids.map((i) => i % 5) });
		await src.delete([1, 2, 5003]);
		const { bytes: pack, result } = await packToBytes(b.store);
		expect(result).toMatchObject({ gen: "gp", rev: src.rev, count: result.keys.length });
		expect(result.keys[0]).toBe("idx/current.json");
		expect(result.bytes).toBe(pack.byteLength);
		// the source has more objects than the snapshot (old manifests)
		expect((await b.store.list("")).length).toBeGreaterThan(result.count);

		const ps = await PackedBlobStore.fromBytes(pack);
		expect(await ps.list("")).toEqual(result.keys.slice().sort());
		const idx = await thread.open({ store: ps, cache: false });
		expect([idx.gen, idx.rev]).toEqual(["gp", src.rev]);
		const qr = rng(21);
		const same = (a: SearchResult, c: SearchResult) => expect(a.hits.map((h) => [h.id, h.score, h.doc?.meta ?? null])).toEqual(c.hits.map((h) => [h.id, h.score, h.doc?.meta ?? null]));
		for (let i = 0; i < 6; i++) {
			const q = unitRows(1, b.dim, qr);
			const a = await src.query(q, { nprobe: 5, rerank: 30, returnPayload: true });
			const c = await idx.query(q, { nprobe: 5, rerank: 30, returnPayload: true });
			expect(c.hits).toEqual(a.hits);
			same(await src.search({ text: "fresh odd", vector: q, topK: 8, docs: true }), await idx.search({ text: "fresh odd", vector: q, topK: 8, docs: true }));
		}
		same(await src.search({ text: "even vector", topK: 5, docs: true }), await idx.search({ text: "even vector", topK: 5, docs: true }));
		// writes to the pack are refused
		await expect(idx.upsert({ ids: [1], vectors: unitRows(1, b.dim, qr) })).rejects.toThrow(/read-only/);
		await expect(idx.delete([1])).rejects.toThrow(/read-only/);
		// the source moves on; the pack stays at its revision even after a refresh
		await src.delete([5004]);
		expect(await idx.refresh()).toBe(false);
		expect(idx.rev).toBe(src.rev - 1);

		// copied out into a writable store, the snapshot continues from where it was packed
		const dst = new MemoryBlobStore();
		const copied = await copyObjects(ps, dst);
		expect(copied).toMatchObject({ count: result.count, skipped: 0 });
		expect(copied.bytes).toBe(result.keys.length ? [...ps.directory.entries.values()].reduce((a, e) => a + e.length, 0) : 0);
		const again = await copyObjects(ps, dst, undefined, { skipExisting: true });
		expect(again).toMatchObject({ count: 0, skipped: result.count });
		const w = await thread.open({ store: dst, cache: false });
		expect(w.rev).toBe(idx.rev);
		await w.upsert({ ids: [7000], vectors: unitRows(1, b.dim, qr), texts: ["copied then written"], metas: ["c"] });
		expect(w.rev).toBe(idx.rev + 1);
		expect((await w.search({ text: "copied", topK: 1 })).hits[0].id).toBe(7000);
		expect(await copyObjects(new MemoryBlobStore(), ps).catch((e: ThreadError) => e.message)).toMatch(/read-only/);
		src.close();
		idx.close();
		w.close();
	});

	it("a cache in front of a pack caches the objects, never the pack", async () => {
		const b = await buildGen("gc", 3);
		const { bytes: pack } = await packToBytes(b.store);
		const inner = new MemoryBlobStore();
		await inner.put("packs/gc.threadpack", pack);
		const reads: number[] = [];
		const counting = {
			get: async (k: string, r?: { offset: number; length: number }, o?: object) => {
				const x = await inner.get(k, r, o);
				reads.push(x?.byteLength ?? 0);
				return x;
			},
			head: (k: string) => inner.head(k),
		};
		const ps = await PackedBlobStore.open(counting, "packs/gc.threadpack");
		reads.length = 0; // opening read the tail (the whole pack here: it is smaller than the tail read)
		const cache = new FakeCache();
		const pending: Promise<unknown>[] = [];
		const store = new CachedBlobStore(ps, cache, { origin: "https://t.cache/", waitUntil: (p) => pending.push(p) });
		const idx = await thread.open({ store, cache: false });
		const q = b.vectors.subarray(0, b.dim);
		const r1 = await idx.query(q, { nprobe: 8 });
		await Promise.all(pending);
		expect(r1.stats.io.misses).toBeGreaterThan(0);
		expect([...cache.objects.keys()].every((u) => u.startsWith("https://t.cache/idx/gc/"))).toBe(true);
		expect(cache.objects.has("https://t.cache/packs/gc.threadpack")).toBe(false);
		expect(Math.max(...reads)).toBeLessThan(pack.byteLength); // every read is one object or a range of it, never the pack
		const before = reads.length;
		const r2 = await idx.query(q, { nprobe: 8 });
		expect(r2.hits).toEqual(r1.hits);
		expect(r2.stats.io.reads).toBe(0);
		expect(reads.length).toBe(before);
		idx.close();
	});

	it("packIndex with gen/rev picks a revision; missing objects fail loudly", async () => {
		const b = await buildGen("gr", 2);
		const src = await thread.open({ store: b.store, cache: false });
		await src.upsert({ ids: [1], vectors: unitRows(1, b.dim, rng(1)) });
		const r0 = await packIndex(b.store, new MemorySink(), { gen: "gr", rev: 0 });
		expect(r0.rev).toBe(0);
		expect(r0.keys.some((k) => k.startsWith("idx/gr/delta/"))).toBe(false);
		const r1 = await packIndex(b.store, new MemorySink(), { gen: "gr" });
		expect(r1.rev).toBe(1);
		await expect(packIndex(b.store, new MemorySink(), { gen: "zz" })).rejects.toThrow(/no manifest/);
		b.store.objects.delete("idx/gr/slab/00003");
		await expect(packIndex(b.store, new MemorySink())).rejects.toThrow(/needs idx\/gr\/slab\/00003/);
		src.close();
	});
});
