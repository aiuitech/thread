/**
 * Switching `idx/current.json` to a generation built beside the live one (`setCurrent`, a compare-and-swap),
 * collecting a generation's superseded objects (`gcGeneration`, the Worker's `/build/gc`) and deleting a generation
 * the pointer left (`dropGeneration`), with the grace periods that protect readers and writers.
 */
import { describe, expect, it } from "vitest";
import { createLexicalGeneration, dropGeneration, gcGeneration, keys, MemoryBlobStore, setCurrent, supersededKey, ThreadError } from "../src/index.js";
import { rng, thread, unitRows } from "./helpers.js";

const DIM = 16;
const current = async (store: MemoryBlobStore) => JSON.parse(new TextDecoder().decode((await store.get(keys.current()))!)) as { gen: string; rev: number };
/** Makes every object (or those `pick` accepts) look `ageMs` old. */
const age = (store: MemoryBlobStore, ageMs: number, pick: (k: string) => boolean = () => true) => {
	for (const [k, o] of store.objects) if (pick(k)) o.uploaded = Date.now() - ageMs;
};

/** A live lexical generation `lex` (current) and a dense one `dense` built beside it, filled through `open({gen})`. */
async function twoGenerations() {
	const store = new MemoryBlobStore();
	await createLexicalGeneration({ store, gen: "lex", lex: { text: true } });
	const live = await thread.open({ store, cache: false, refreshInterval: 0 });
	await live.upsert({ keys: ["a", "b"], texts: ["the plumber fixed the tap", "coffee grinder"] });
	await thread.createGeneration({ store, gen: "dense", dim: DIM, lex: { text: true }, setCurrent: false });
	const dense = await thread.open({ store, gen: "dense", cache: false, refreshInterval: 0 });
	const v = unitRows(2, DIM, rng(3));
	await dense.upsert({ keys: ["a", "b"], texts: ["the plumber fixed the tap", "coffee grinder"], vectors: v });
	return { store, live, dense, v };
}

describe("setCurrent", () => {
	it("moves current.json to another generation with a compare-and-swap", async () => {
		const { store, live, dense } = await twoGenerations();
		// Writes to the dense generation never moved the pointer.
		expect(await current(store)).toMatchObject({ gen: "lex", rev: live.rev });
		const follower = await thread.open({ store, cache: false, refreshInterval: 0 });
		expect(follower.gen).toBe("lex");
		// A wrong expectation writes nothing.
		await expect(setCurrent(store, { gen: "dense", expect: { gen: "lex", rev: live.rev + 5 } })).rejects.toMatchObject({ status: 409 });
		await expect(setCurrent(store, { gen: "dense", expect: null })).rejects.toThrow(ThreadError);
		expect((await current(store)).gen).toBe("lex");
		await expect(setCurrent(store, { gen: "nope" })).rejects.toMatchObject({ status: 404 });
		await expect(setCurrent(store, { gen: "dense", rev: dense.rev + 1 })).rejects.toMatchObject({ status: 404 });
		// The switch: rev defaults to the head of the generation.
		const r = await live.setCurrent({ gen: "dense", expect: { gen: "lex", rev: live.rev } });
		expect(r).toMatchObject({ gen: "dense", rev: dense.rev, previous: { gen: "lex", rev: live.rev }, updated: true });
		const ptr = await current(store);
		expect(ptr).toMatchObject({ gen: "dense", rev: dense.rev });
		expect((ptr as { manifest?: { gen: string } }).manifest?.gen).toBe("dense"); // the inline copy is the new one
		expect(JSON.parse(new TextDecoder().decode((await store.get(supersededKey("lex")))!))).toMatchObject({ by: { gen: "dense", rev: dense.rev }, rev: live.rev });
		// The same switch again is a no-op; a follower moves on refresh; the old writer no longer moves the pointer.
		expect((await setCurrent(store, { gen: "dense", expect: { gen: "dense", rev: dense.rev } })).updated).toBe(false);
		await follower.refresh();
		expect(follower.gen).toBe("dense");
		expect((await follower.search({ text: "plumber", topK: 3 })).hits.length).toBe(1);
		const w = await live.upsert({ keys: ["c"], texts: ["late write to the old generation"] });
		expect(w.currentUpdated).toBe(false);
		expect((await current(store)).gen).toBe("dense");
		const d = await dense.upsert({ keys: ["c"], texts: ["first write after the switch"] });
		expect(d.currentUpdated).toBe(true);
		expect(await current(store)).toMatchObject({ gen: "dense", rev: d.rev });
		// No pointer at all: expect null creates it.
		const empty = new MemoryBlobStore();
		await createLexicalGeneration({ store: empty, gen: "g", lex: {}, setCurrent: false });
		expect((await setCurrent(empty, { gen: "g", expect: null })).previous).toBeNull();
		follower.close();
		live.close();
		dense.close();
	});
});

describe("gcGeneration", () => {
	it("deletes what no kept revision references, never what is young", async () => {
		const store = new MemoryBlobStore();
		await thread.createGeneration({ store, gen: "d", dim: DIM, lex: null, compaction: { maxDeltas: 1, maxCellEntries: 30 } });
		const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
		const v = unitRows(120, DIM, rng(9));
		for (let a = 0; a < 120; a += 30) await ix.upsert({ ids: Array.from({ length: 30 }, (_, i) => a + i), vectors: v.subarray(a * DIM, (a + 30) * DIM) });
		while ((await ix.compact({ force: true })).more);
		await ix.delete([7]);
		// Everything young: nothing goes.
		const young = await ix.gc({ keep: 1 });
		expect(young.garbage).toBe(0);
		age(store, 3_600_000);
		const dry = await ix.gc({ keep: 1 });
		expect(dry.dryRun).toBe(true);
		expect(dry.keptRevs).toEqual([ix.rev]);
		expect(dry.garbage).toBeGreaterThan(0);
		expect(dry.kinds.delta?.n).toBe(4);
		expect(dry.kinds.slab?.n).toBeGreaterThan(0);
		expect(dry.deleted).toBe(0);
		const objects = store.objects.size;
		const real = await gcGeneration(store, { gen: "d", keep: 1, dryRun: false });
		expect(real.deleted).toBe(dry.garbage);
		expect(store.objects.size).toBe(objects - dry.garbage);
		// The head still answers, from a cold open; manifests stay as history.
		const cold = await thread.open({ store, cache: false });
		for (const i of [0, 50, 119]) expect((await cold.query(v.subarray(i * DIM, (i + 1) * DIM), { topK: 1, nprobe: 64 })).hits[0].id).toBe(i);
		expect((await cold.query(v.subarray(7 * DIM, 8 * DIM), { topK: 1, nprobe: 64 })).hits[0].id).not.toBe(7);
		expect(await store.head(keys.manifest("d", 1))).not.toBeNull();
		// An unreferenced object younger than the grace period stays.
		await store.put("idx/d/slab/00000.999", new Uint8Array(8));
		expect((await gcGeneration(store, { gen: "d", keep: 1 })).sample).not.toContain("idx/d/slab/00000.999");
		// A revision superseded within the grace period is kept with its objects.
		await ix.delete([8]);
		const recent = await gcGeneration(store, { gen: "d", keep: 1 });
		expect(recent.keptRevs).toEqual([ix.rev - 1, ix.rev]);
		cold.close();
		ix.close();
	});
});

describe("dropGeneration", () => {
	it("deletes a generation the pointer left, after the grace period", async () => {
		const { store, live, dense } = await twoGenerations();
		await expect(dropGeneration(store, { gen: "lex" })).rejects.toThrow(/current generation/);
		await setCurrent(store, { gen: "dense", expect: { gen: "lex", rev: live.rev } });
		await expect(dropGeneration(store, { gen: "lex" })).rejects.toThrow(/left idx\/current.json/);
		// The switch is old, but the old generation was written recently: still refused.
		age(store, 3_600_000, (k) => k === supersededKey("lex"));
		const m = JSON.parse(new TextDecoder().decode((await store.get(supersededKey("lex")))!));
		await store.put(supersededKey("lex"), JSON.stringify({ ...m, ms: Date.now() - 3_600_000 }));
		await expect(dropGeneration(store, { gen: "lex" })).rejects.toThrow(/written .* ago/);
		age(store, 3_600_000, (k) => k.startsWith("idx/lex/"));
		await expect(dense.dropGeneration({ gen: "dense" })).rejects.toMatchObject({ status: 409 });
		const dry = await dense.dropGeneration({ gen: "lex" });
		expect(dry.dryRun).toBe(true);
		const lexObjects = [...store.objects.keys()].filter((k) => k.startsWith("idx/lex/"));
		expect(dry.objects).toBe(lexObjects.length);
		expect(dry.deleted).toBe(0);
		const real = await dropGeneration(store, { gen: "lex", dryRun: false });
		expect(real.deleted).toBe(lexObjects.length);
		expect([...store.objects.keys()].filter((k) => k.startsWith("idx/lex/"))).toEqual([]);
		// The live generation is untouched and still current.
		expect((await current(store)).gen).toBe("dense");
		const r = await thread.open({ store, cache: false });
		expect((await r.search({ text: "coffee", topK: 3 })).hits.length).toBe(1);
		r.close();
		live.close();
		dense.close();
	});
});
