/** Base packs: one GET for a cold open, byte-identical state, index packs with a base range. */
import { describe, expect, it } from "vitest";
import { buildBasePack, keys, loadRevState, MemorySink, packEntries, packIndex, PackedBlobStore, type BlobRange, type BlobStore, type GetOptions, type MemoryBlobStore, type ThreadIndex } from "../src/index.js";
import { addRoutingRev, buildGen, rng, thread, unitRows } from "./helpers.js";

/** Counts the reads that reach `inner` (key, or key@offset+length for a range). */
function counting(inner: BlobStore): { store: BlobStore; reads: string[] } {
	const reads: string[] = [];
	const store: BlobStore = {
		get: (k: string, r?: BlobRange, o?: GetOptions) => (reads.push(r ? `${k}@${r.offset}+${r.length}` : k), inner.get(k, r, o)),
		getRanges: inner.getRanges ? (k: string, rs: BlobRange[], o?: GetOptions) => (reads.push(`${k}#${rs.length}`), inner.getRanges!(k, rs, o)) : undefined,
		head: (k: string) => (reads.push(`head ${k}`), inner.head(k)),
		list: inner.list ? (p: string) => inner.list!(p) : undefined,
		put: inner.put ? (k, d, o) => inner.put!(k, d, o) : undefined,
	};
	return { store, reads };
}

const qs = (n: number, dim: number) => {
	const v = unitRows(n, dim, rng(31));
	return Array.from({ length: n }, (_, i) => v.subarray(i * dim, (i + 1) * dim));
};

async function sameAnswers(a: ThreadIndex, b: ThreadIndex, dim: number): Promise<void> {
	expect(b.revision.manifest).toEqual(a.revision.manifest);
	expect(b.revision.slabs).toEqual(a.revision.slabs);
	expect(b.revision.tomb?.sets).toEqual(a.revision.tomb?.sets);
	for (const q of qs(10, dim)) {
		const [x, y] = [await a.query(q, { nprobe: 4 }), await b.query(q, { nprobe: 4 })];
		expect(y.hits).toEqual(x.hits);
		expect(y.stats.cells).toEqual(x.stats.cells);
	}
}

async function packed(store: MemoryBlobStore, gen: string, rev: number, kind: "base" | "cold"): Promise<string> {
	const p = await buildBasePack(store, await loadRevState(store, gen, rev), kind);
	await store.put(p.key, p.bytes);
	return p.key;
}

describe("base packs", () => {
	it("a cold open is current.json + the base pack (+ the manifest, read alongside) and loads identical state", async () => {
		const b = await buildGen("bp", 3, { k2: 32, n: 900 });
		await addRoutingRev(b.store, "bp", { tree: { groups: 4 } });
		const writer = await thread.open({ store: b.store, cache: false });
		await writer.delete([3, 4, 5]); // a tombstone set: rev 2
		writer.close();
		const key = await packed(b.store, "bp", 2, "base");
		expect(key).toBe(keys.basePack("bp", 2));
		const { entries, meta } = packEntries((await b.store.get(key))!);
		expect(meta).toMatchObject({ gen: "bp", rev: 2, kind: "base", routing: "i8" });
		expect([...entries.keys()]).toEqual(["idx/bp/manifest.2.json", "idx/bp/slabs.json", "idx/bp/tombstones.2.u32", "idx/bp/coarse.f32", "idx/bp/pq.bin", "idx/bp/tree.1.bin", "idx/bp/fine.i8"]);
		for (const [k, v] of entries) expect(v).toEqual(await b.store.get(k)); // byte-identical objects

		const plain = await thread.open({ store: b.store, cache: false, routing: "i8", basePack: false });
		expect(plain.openStats.prefetch.source).toBe("none");
		const c = counting(b.store);
		const fast = await thread.open({ store: c.store, cache: false, routing: "i8" });
		expect(fast.openStats.prefetch.source).toBe("base-pack");
		expect(fast.openStats.prefetch.key).toBe(key);
		expect(c.reads.sort()).toEqual(["idx/bp/base.2.threadpack", "idx/bp/manifest.2.json", "idx/current.json"]);
		await sameAnswers(plain, fast, b.dim);
		// the default routing (f32) is not in the pack: that table is read on its own, the rest still comes from the pack
		const c2 = counting(b.store);
		const f32 = await thread.open({ store: c2.store, cache: false });
		expect(c2.reads.filter((k) => !k.startsWith("idx/bp/slab/")).sort()).toEqual(["idx/bp/base.2.threadpack", "idx/bp/fine.f32", "idx/bp/manifest.2.json", "idx/current.json"]);
		for (const ix of [plain, fast, f32]) ix.close();
	});

	it("without a pack every base read starts once the manifest is in; state is the same", async () => {
		const b = await buildGen("bf", 3, { k2: 16 });
		await addRoutingRev(b.store, "bf");
		const c = counting(b.store);
		const ix = await thread.open({ store: c.store, cache: false, routing: "f16" });
		expect(ix.openStats.prefetch.source).toBe("files");
		expect(new Set(c.reads)).toEqual(new Set(["idx/current.json", "idx/bf/base.1.threadpack", "idx/bf/manifest.1.json", "idx/bf/slabs.json", "idx/bf/coarse.f32", "idx/bf/pq.bin", "idx/bf/fine.f16"]));
		// every read is accounted in the step that used it
		expect(ix.openStats.manifest.reads).toBe(1);
		expect(ix.openStats.slabs.reads).toBe(1);
		expect(ix.openStats.base!.reads).toBe(3);
		const plain = await thread.open({ store: b.store, cache: false, routing: "f16", basePack: false });
		await sameAnswers(plain, ix, b.dim);
		ix.close();
		plain.close();
	});

	it("cold base pack: tree + codebook in one GET, the first query reads the beam's groups", async () => {
		const b = await buildGen("bc", 3, { k2: 48, n: 1500 });
		await addRoutingRev(b.store, "bc", { tree: { groups: 6 } });
		const key = await packed(b.store, "bc", 1, "cold");
		expect(key).toBe(keys.baseColdPack("bc", 1));
		const { entries, meta } = packEntries((await b.store.get(key))!);
		expect(meta).toMatchObject({ kind: "base-cold", routing: null });
		expect([...entries.keys()]).toEqual(["idx/bc/manifest.1.json", "idx/bc/slabs.json", "idx/bc/coarse.f32", "idx/bc/pq.bin", "idx/bc/tree.1.bin"]);
		let release!: () => void;
		const gate = new Promise<void>((r) => (release = r));
		const c = counting({ ...b.store, get: async (k, r, o) => (k === "idx/bc/fine.f32" && (await gate), b.store.get(k, r, o)), head: (k) => b.store.head(k), getRanges: (k, r, o) => b.store.getRanges(k, r, o) });
		const cold = await thread.open({ store: c.store, cache: false, coldRouting: "tree", coldBeam: 6 });
		expect(cold.openStats.prefetch.source).toBe("base-pack");
		expect(cold.routingState).toBe("cold");
		expect(c.reads.filter((k) => k !== "idx/bc/fine.f32").sort()).toEqual(["idx/bc/base.cold.1.threadpack", "idx/bc/manifest.1.json", "idx/current.json"]);
		const full = await thread.open({ store: b.store, cache: false, basePack: false });
		const q = qs(1, b.dim)[0];
		const r = await cold.query(q, { nprobe: 5 });
		expect(r.stats.routing).toBe("cold");
		expect(r.stats.cold!.reads).toBe(1);
		expect(r.hits.length).toBe(10);
		release();
		await cold.whenFullRouting();
		await sameAnswers(full, cold, b.dim);
		cold.close();
		full.close();
	});

	it("index packs record a base range: a PackedBlobStore open reads it with one range read", async () => {
		const b = await buildGen("bi", 3, { k2: 16 });
		await addRoutingRev(b.store, "bi", { tree: { groups: 4 } });
		const sink = new MemorySink();
		const res = await packIndex(b.store, sink);
		const bytes = sink.bytes();
		// current.json, then the base objects (smallest routing table), then the rest in key order
		expect(res.keys.slice(0, 8)).toEqual(["idx/current.json", "idx/bi/manifest.1.json", "idx/bi/slabs.json", "idx/bi/coarse.f32", "idx/bi/pq.bin", "idx/bi/tree.1.bin", "idx/bi/fine.i8", "idx/bi/fine.f16"]);
		const p = await PackedBlobStore.fromBytes(bytes);
		const range = p.baseRange!;
		const fineI8 = p.directory.entries.get("idx/bi/fine.i8")!;
		expect(range).toEqual({ offset: 16, length: fineI8.offset + fineI8.length - 16 });
		const c = counting(p.inner);
		const viaRange = await PackedBlobStore.open(c.store, p.key);
		c.reads.length = 0;
		const ix = await thread.open({ store: viaRange, cache: false, routing: "auto" });
		expect(ix.openStats.prefetch.source).toBe("index-pack");
		expect(c.reads).toEqual([`index.threadpack@16+${range.length}`]);
		const plain = await thread.open({ store: b.store, cache: false, routing: "auto", basePack: false });
		await sameAnswers(plain, ix, b.dim);
		ix.close();
		plain.close();
	});
});
