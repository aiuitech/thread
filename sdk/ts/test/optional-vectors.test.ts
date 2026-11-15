/**
 * Optional vectors (`upsert({ vectors: [row, null, ...] })`): a null row writes that document without a vector (text,
 * meta and key only) and the others with theirs, in one revision; the delta holds only the rows given, with their
 * ids, attrs and payloads; an older vector of a null-row document is hidden; compaction folds the delta as usual.
 */
import { describe, expect, it } from "vitest";
import { MemoryBlobStore } from "../src/index.js";
import { rng, thread, unitRows } from "./helpers.js";

const DIM = 16;

async function setup(lex = true) {
	const store = new MemoryBlobStore();
	await thread.createGeneration({ store, gen: "o", dim: DIM, ...(lex ? { lex: { text: true } } : { lex: null }), attrs: { tag: { bits: 4 } }, compaction: { maxDeltas: 1, maxCellEntries: 0 } });
	const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
	return { store, ix, v: unitRows(8, DIM, rng(9)) };
}
const row = (v: Float32Array, i: number) => v.subarray(i * DIM, (i + 1) * DIM);
type Ix = Awaited<ReturnType<typeof setup>>["ix"];
const top = async (ix: Ix, q: Float32Array, filter?: Record<string, number>) => (await ix.query(q, { topK: 1, nprobe: 8, ...(filter ? { filter } : {}) })).hits[0]?.id;
const textIds = async (ix: Ix, text: string) => (await ix.search({ text, topK: 5, embed: false })).hits.map((h) => h.id);

describe("optional vectors", () => {
	it("a mixed batch writes one delta for the rows given and one segment for every document", async () => {
		const { ix, v } = await setup();
		const before = ix.rev;
		const r = await ix.upsert({ keys: ["a", "b", "c"], texts: ["alpha words", "bravo words", "charlie words"], metas: ["A", "B", "C"], attrs: [{ tag: 1 }, { tag: 2 }, { tag: 3 }], vectors: [row(v, 0), null, row(v, 2)], payloads: ["pa", "", "pc"] });
		expect(r.rev).toBe(before + 1);
		expect(r.count).toBe(3);
		expect(r.cells).toHaveLength(2);
		expect(ix.manifest.deltas!.at(-1)!.count).toBe(2);
		expect(ix.manifest.segments!.at(-1)!.count).toBe(3);
		const [a, b, c] = r.ids;
		// the vector rows are found with their own ids and attrs; b only by text
		expect(await top(ix, row(v, 0))).toBe(a);
		expect(await top(ix, row(v, 2))).toBe(c);
		expect(await top(ix, row(v, 2), { tag: 3 })).toBe(c);
		expect(await top(ix, row(v, 0), { tag: 2 })).toBeUndefined();
		expect(await textIds(ix, "bravo")).toEqual([b]);
		expect((await ix.get(["b"], { text: true })).records.get("b")).toMatchObject({ id: b, meta: "B", attrs: 2, text: "bravo words" });
		// the payloads stay with their vectors
		const hits = (await ix.query(row(v, 2), { topK: 1, nprobe: 8, returnPayload: true })).hits;
		expect(new TextDecoder().decode(hits[0].payload!)).toBe("pc");
		// compaction folds the two vectors; nothing changes for the reader
		expect((await ix.compact({ force: true })).outcome).toBe("committed");
		expect(await top(ix, row(v, 0))).toBe(a);
		expect(await top(ix, row(v, 2), { tag: 3 })).toBe(c);
		expect(await textIds(ix, "bravo")).toEqual([b]);
		ix.close();
	});

	it("a null row re-put hides the document's older vector and keeps it findable by text", async () => {
		const { ix, v } = await setup();
		const a = await ix.upsert({ keys: ["doc", "other"], texts: ["first text", "other text"], vectors: [row(v, 0), row(v, 1)] });
		await ix.upsert({ keys: ["doc", "other"], texts: ["second text", "other text"], vectors: [null, row(v, 1)] });
		expect(await top(ix, row(v, 0))).toBe(a.ids[1]);
		expect(await textIds(ix, "second")).toEqual([a.ids[0]]);
		expect(await textIds(ix, "first")).toEqual([]);
		ix.close();
	});

	it("every row null is a text-only batch: no delta", async () => {
		const { ix } = await setup();
		const r = await ix.upsert({ keys: ["x", "y"], texts: ["xray", "yankee"], vectors: [null, null] });
		expect(r.keys.delta).toBeNull();
		expect(r.cells).toEqual([]);
		expect(await textIds(ix, "yankee")).toEqual([r.ids[1]]);
		ix.close();
	});

	it("refuses null rows where the document would be lost or the payload has no vector", async () => {
		const { ix, v } = await setup();
		const before = ix.rev;
		await expect(ix.upsert({ ids: [1, 2], vectors: [row(v, 0), null] })).rejects.toThrow(/needs texts, metas or keys/);
		await expect(ix.upsert({ ids: [1, 2], texts: ["a", "b"], vectors: [row(v, 0), null], payloads: ["p", "q"] })).rejects.toThrow(/payloads\[1\] belongs to no vector/);
		await expect(ix.upsert({ ids: [1, 2], texts: ["a", "b"], vectors: [row(v, 0), null], payloads: ["p"] })).rejects.toThrow(/1 payloads for 2 documents/);
		await expect(ix.upsert({ ids: [1], texts: ["a"], vectors: [new Float32Array(DIM)] })).rejects.toThrow(/zero or non-finite/);
		await ix.upsert({ keys: ["k"], texts: ["kilo"] });
		await expect(ix.upsert({ keys: ["k"], vectors: [null], scope: "dense" })).rejects.toThrow(/needs a vector for every document/);
		expect(ix.rev).toBe(before + 1);
		ix.close();
		const { ix: dense } = await setup(false);
		await expect(dense.upsert({ ids: [1, 2], vectors: [row(v, 0), null] })).rejects.toThrow(/needs a lexical configuration/);
		dense.close();
	});

	it("the zero-norm error names the document, not the row", async () => {
		const { ix, v } = await setup();
		await expect(ix.upsert({ ids: [7, 8, 9], texts: ["a", "b", "c"], vectors: [null, row(v, 0), new Float32Array(DIM)] })).rejects.toThrow(/vector 2 \(id 9\)/);
		ix.close();
	});
});
