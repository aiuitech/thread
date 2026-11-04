/**
 * Scoped tombstones (`upsert({ scope })`): a vector-only upsert by key keeps the document's text, docs and key; a
 * text-only re-put keeps its vector; deletes hide both; attrs cannot drift between the sides; each compaction keeps
 * the other side and drops its own sets; manifests with features a reader does not know are refused.
 */
import { describe, expect, it } from "vitest";
import { FEATURE_SCOPED_TOMBSTONES, MemoryBlobStore, PackedBlobStore, keys, packToBytes, type Manifest } from "../src/index.js";
import { rng, thread, unitRows } from "./helpers.js";

const DIM = 16;

async function setup() {
	const store = new MemoryBlobStore();
	await thread.createGeneration({ store, gen: "s", dim: DIM, lex: { text: true }, attrs: { tag: { bits: 4 } }, compaction: { maxDeltas: 1, maxCellEntries: 0 } });
	const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
	return { store, ix, v: unitRows(8, DIM, rng(4)) };
}
const row = (v: Float32Array, i: number) => v.subarray(i * DIM, (i + 1) * DIM);
const top = async (ix: Awaited<ReturnType<typeof setup>>["ix"], q: Float32Array) => (await ix.query(q, { topK: 1, nprobe: 8 })).hits[0]?.id;
const textIds = async (ix: Awaited<ReturnType<typeof setup>>["ix"], text: string) => (await ix.search({ text, topK: 5, embed: false })).hits.map((h) => h.id);

describe("scoped upserts", () => {
	it("a dense-scoped upsert by key attaches a vector and keeps the text, docs and key", async () => {
		const { ix, v } = await setup();
		const a = await ix.upsert({ keys: ["mail/1", "mail/2"], texts: ["invoice from the plumber", "team lunch friday"], metas: ["m1", "m2"], attrs: [{ tag: 3 }, { tag: 1 }] });
		const r = await ix.upsert({ keys: ["mail/1"], vectors: [row(v, 0)], attrs: [{ tag: 3 }], scope: "dense" });
		expect(r.ids).toEqual([a.ids[0]]);
		expect(r.keys.segment).toBeNull();
		expect(ix.manifest.features).toEqual([FEATURE_SCOPED_TOMBSTONES]);
		expect(ix.manifest.tombstones!.at(-1)!.scope).toBe("dense");
		expect(await textIds(ix, "plumber")).toEqual([a.ids[0]]);
		const got = (await ix.get(["mail/1"], { text: true })).records.get("mail/1");
		expect(got).toMatchObject({ id: a.ids[0], meta: "m1", text: "invoice from the plumber" });
		expect((await ix.listKeys({ prefix: "mail/" })).entries.map((e) => e.key)).toEqual(["mail/1", "mail/2"]);
		expect(await top(ix, row(v, 0))).toBe(a.ids[0]);
		// texts can be the embedder's input (not stored); a key with no document is an error, nothing is written
		const e = await ix.upsert({ keys: ["mail/2"], texts: ["ignored as text"], embed: () => [row(v, 1)], scope: "dense" });
		expect(e.keys.segment).toBeNull();
		expect(await textIds(ix, "lunch")).toEqual([a.ids[1]]);
		expect(await textIds(ix, "ignored")).toEqual([]);
		const before = ix.rev;
		await expect(ix.upsert({ keys: ["mail/9"], vectors: [row(v, 2)], scope: "dense" })).rejects.toMatchObject({ status: 404 });
		await expect(ix.upsert({ keys: ["mail/1"], vectors: [row(v, 2)], metas: ["x"], scope: "dense" })).rejects.toMatchObject({ status: 400 });
		expect(ix.rev).toBe(before);
		ix.close();
	});

	it("a lexical-scoped re-put changes the text and keeps the vector", async () => {
		const { ix, v } = await setup();
		const a = await ix.upsert({ keys: ["doc"], texts: ["old words here"], vectors: [row(v, 0)], attrs: [{ tag: 5 }] });
		const id = a.ids[0];
		await ix.upsert({ keys: ["doc"], texts: ["fresh new text"], metas: ["renamed"], scope: "lexical" });
		expect(await top(ix, row(v, 0))).toBe(id);
		expect(await textIds(ix, "old words")).toEqual([]);
		expect(await textIds(ix, "fresh")).toEqual([id]);
		// absent attrs kept the live word; a different word is refused while the dense entry has the old one
		expect((await ix.get(["doc"])).records.get("doc")!.attrs).toBe(5);
		expect((await ix.query(row(v, 0), { topK: 1, nprobe: 8, filter: { tag: 5 } })).hits[0].id).toBe(id);
		await expect(ix.upsert({ keys: ["doc"], texts: ["x"], attrs: [{ tag: 6 }], scope: "lexical" })).rejects.toThrow(/cannot change the attrs/);
		await ix.upsert({ keys: ["doc"], texts: ["same word ok"], attrs: [{ tag: 5 }], scope: "lexical" });
		await expect(ix.upsert({ keys: ["doc"], vectors: [row(v, 1)], scope: "lexical" })).rejects.toMatchObject({ status: 400 });
		// an unscoped delete hides both sides
		await ix.delete([id]);
		expect(await top(ix, row(v, 0))).not.toBe(id);
		expect(await textIds(ix, "same word")).toEqual([]);
		expect((await ix.get(["doc"])).records.size).toBe(0);
		ix.close();
	});

	it("each compaction keeps the other side and drops only the sets it no longer needs", async () => {
		const { store, ix, v } = await setup();
		const a = await ix.upsert({ keys: ["a", "b"], texts: ["alpha text", "bravo text"], vectors: [row(v, 0), row(v, 1)] });
		await ix.upsert({ keys: ["a"], vectors: [row(v, 2)], scope: "dense" }); // a's vector replaced, text kept
		await ix.upsert({ keys: ["b"], texts: ["bravo renamed"], scope: "lexical" }); // b's text replaced, vector kept
		const scopes = () => (ix.manifest.tombstones ?? []).map((t) => t.scope ?? "both");
		expect(scopes()).toEqual(["both", "dense", "lexical"]);
		// Dense compaction: folds the deltas; the dense set goes, and the first upsert's own set (nothing predates it);
		// the lexical one waits for the segment it hides.
		const c = await ix.compact({ force: true });
		expect(c.outcome).toBe("committed");
		expect(scopes()).toEqual(["lexical"]);
		expect(await top(ix, row(v, 2))).toBe(a.ids[0]);
		expect(await top(ix, row(v, 1))).toBe(a.ids[1]);
		expect(await textIds(ix, "alpha")).toEqual([a.ids[0]]);
		expect(await textIds(ix, "renamed")).toEqual([a.ids[1]]);
		// Segment compaction: merges the segments, and the lexical set goes with the copy it hid.
		await ix.compactSegments();
		expect(ix.manifest.tombstones).toEqual([]);
		expect(await textIds(ix, "bravo")).toEqual([a.ids[1]]);
		expect(await top(ix, row(v, 1))).toBe(a.ids[1]);
		// A cold reader of the head sees the same.
		const cold = await thread.open({ store, cache: false });
		expect(await top(cold, row(v, 2))).toBe(a.ids[0]);
		expect(await textIds(cold, "renamed")).toEqual([a.ids[1]]);
		cold.close();
		ix.close();
	});

	it("a pack of a scoped revision carries the scopes", async () => {
		const { store, ix, v } = await setup();
		const a = await ix.upsert({ keys: ["a", "b"], texts: ["alpha text", "bravo text"], vectors: [row(v, 0), row(v, 1)] });
		await ix.upsert({ keys: ["a"], vectors: [row(v, 2)], scope: "dense" });
		await ix.upsert({ keys: ["b"], texts: ["bravo renamed"], scope: "lexical" });
		const packed = await thread.open({ store: await PackedBlobStore.fromBytes((await packToBytes(store)).bytes), cache: false });
		expect(packed.manifest.tombstones!.map((t) => t.scope ?? "both")).toEqual(["both", "dense", "lexical"]);
		expect(await top(packed, row(v, 2))).toBe(a.ids[0]);
		expect(await top(packed, row(v, 1))).toBe(a.ids[1]);
		expect(await textIds(packed, "alpha")).toEqual([a.ids[0]]);
		expect(await textIds(packed, "renamed")).toEqual([a.ids[1]]);
		packed.close();
		ix.close();
	});

	it("readers refuse a manifest with a feature they do not know", async () => {
		const { store, ix, v } = await setup();
		await ix.upsert({ keys: ["a"], texts: ["alpha"], vectors: [row(v, 0)] });
		const m = JSON.parse(new TextDecoder().decode((await store.get(keys.manifest("s", ix.rev)))!)) as Manifest;
		await store.put(keys.manifest("s", ix.rev + 1), JSON.stringify({ ...m, rev: ix.rev + 1, parentRev: ix.rev, features: [FEATURE_SCOPED_TOMBSTONES, "from-the-future"] }));
		await store.put(keys.current(), JSON.stringify({ gen: "s", rev: ix.rev + 1 }));
		await expect(thread.open({ store, cache: false })).rejects.toThrow(/from-the-future/);
		ix.close();
	});
});
