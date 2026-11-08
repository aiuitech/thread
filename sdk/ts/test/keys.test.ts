/**
 * The key store: keyed upserts allocate ids, `get` / `resolveKeys` / `listKeys` resolve them, a re-put keeps the id
 * (so it replaces the document), `deleteKeys` hides them, compaction keeps the section, facet tokens list by
 * attribute, and a dense generation allocates past its build ids.
 */
import { describe, expect, it } from "vitest";
import { createLexicalGeneration, keys, MemoryBlobStore, type Manifest } from "../src/index.js";
import { buildGen, rng, thread, unitRows } from "./helpers.js";

const key = (i: number) => `s3://bucket/${i % 3}/obj-${i}.bin`;
const meta = (i: number, host = `h${i % 2}.example.com`) => JSON.stringify({ host, size: 100 * i });
const facets = (i: number) => `host:h${i % 2}.example.com size:${i < 2 ? "small" : "large"}`;

async function manifestOf(store: MemoryBlobStore, gen: string, rev: number): Promise<Manifest> {
	return JSON.parse(new TextDecoder().decode((await store.get(keys.manifest(gen, rev)))!)) as Manifest;
}

describe("keys", () => {
	it("allocates ids for keys, resolves and lists them, replaces by key, deletes by key, survives compaction", async () => {
		const store = new MemoryBlobStore();
		await createLexicalGeneration({ store, gen: "kv", lex: { tokenizer: { identifiers: "keep" }, bucketKeys: 2, bucketDocs: 2, bucketTerms: 4 } });
		const index = await thread.open({ store, cache: false });
		expect(index.manifest.nextId).toBeUndefined();

		// four keyed documents: ids come from the manifest's counter (0 for a lexical-only generation)
		const ks = [0, 1, 2, 3].map(key);
		const up = await index.upsert({ keys: ks, metas: ks.map((_, i) => meta(i)), texts: ks.map((_, i) => facets(i)) });
		expect(up.ids).toEqual([0, 1, 2, 3]);
		expect(up.allocated).toBe(4);
		expect(up.rev).toBe(1);
		let m = await manifestOf(store, "kv", 1);
		expect(m.nextId).toBe(4);
		expect(m.segments![0].keys).toBe(true);
		expect(m.change).toMatchObject({ op: "upsert", keys: 4, allocated: 4 });

		// point lookups: two rounds (keys bucket, docs bucket)
		const got = await index.get([key(1), key(3), "nope"]);
		expect([...got.records.keys()]).toEqual([key(1), key(3)]);
		expect(got.records.get(key(1))).toEqual({ key: key(1), id: 1, meta: meta(1), attrs: 0 });
		expect(got.records.get(key(3))!.id).toBe(3);
		expect((await index.resolveKeys([key(2), "nope"])).ids).toEqual(new Map([[key(2), 2]]));

		// prefix listing in key order, paged
		const all = await index.listKeys({ prefix: "s3://bucket/" });
		expect(all.entries).toEqual(ks.map((k, i) => ({ key: k, id: i })).sort((a, b) => (a.key < b.key ? -1 : 1)));
		expect(all.more).toBe(false);
		const zero = await index.listKeys({ prefix: "s3://bucket/0/" });
		expect(zero.entries.map((e) => e.id)).toEqual([0, 3]);
		const page1 = await index.listKeys({ prefix: "s3://bucket/", limit: 1 });
		expect(page1.entries.length).toBe(1);
		expect(page1.more).toBe(true);
		const page2 = await index.listKeys({ prefix: "s3://bucket/", after: page1.next, limit: 10 });
		expect([...page1.entries, ...page2.entries]).toEqual(all.entries);
		expect(page2.more).toBe(false);
		expect((await index.listKeys({ prefix: "nope" })).entries).toEqual([]);

		// facet tokens: an equality filter over an attribute is a term lookup
		const hosts = await index.search({ text: "host:h1.example.com", topK: 10 });
		expect(hosts.hits.map((h) => h.id).sort()).toEqual([1, 3]);
		const small = await index.search({ text: "size:small", topK: 10, docs: true });
		expect(small.hits.map((h) => h.id).sort()).toEqual([0, 1]);
		expect(small.hits[0].doc?.meta).toBe(meta(small.hits[0].id));

		// a re-put of a known key keeps its id and replaces the document; a second segment holds the new copy
		const re = await index.upsert({ keys: [key(1), key(4)], metas: [meta(1, "moved.example.com"), meta(4)], texts: [facets(1), facets(4)] });
		expect(re.ids).toEqual([1, 4]);
		expect(re.allocated).toBe(1);
		m = await manifestOf(store, "kv", 2);
		expect(m.nextId).toBe(5);
		expect(index.segments.length).toBe(2);
		expect((await index.get([key(1)])).records.get(key(1))!.meta).toBe(meta(1, "moved.example.com"));
		expect((await index.listKeys({ prefix: "s3://bucket/1/" })).entries).toEqual([
			{ key: key(1), id: 1 },
			{ key: key(4), id: 4 },
		]);

		// explicit ids may come with keys; the counter moves past them and the next allocation follows
		const ex = await index.upsert({ ids: [100], keys: ["explicit/100"], metas: ["e"] });
		expect(ex.ids).toEqual([100]);
		expect(ex.allocated).toBe(0);
		expect((await manifestOf(store, "kv", 3)).nextId).toBe(101);
		const next = await index.upsert({ keys: ["after/explicit"], metas: ["n"] });
		expect(next.ids).toEqual([101]);
		expect((await index.get(["explicit/100", "after/explicit"])).records.get("explicit/100")!.id).toBe(100);

		// delete by key: the id is tombstoned, so the key, the doc and the postings disappear
		const del = await index.deleteKeys([key(2), "nope"]);
		expect(del.count).toBe(1);
		expect(del.missing).toEqual(["nope"]);
		expect(del.rev).toBe(5);
		expect((await index.get([key(2)])).records.size).toBe(0);
		expect((await index.resolveKeys([key(2)])).ids.size).toBe(0);
		expect((await index.listKeys({ prefix: "s3://bucket/2/" })).entries).toEqual([]);
		expect((await index.search({ text: "size:large", topK: 10 })).hits.map((h) => h.id).sort()).toEqual([3, 4]);
		const none = await index.deleteKeys(["nope"]);
		expect(none.count).toBe(0);
		expect(none.rev).toBe(5); // nothing written

		// compaction: one base segment with a keys section; everything resolves as before
		const c = await index.compactSegments();
		expect(c.merged).toBe(4);
		expect(index.segments.length).toBe(1);
		expect(index.segments[0].keys).toBe(true);
		expect((await index.listKeys({})).entries).toEqual([
			{ key: "after/explicit", id: 101 },
			{ key: "explicit/100", id: 100 },
			{ key: key(0), id: 0 },
			{ key: key(3), id: 3 },
			{ key: key(1), id: 1 },
			{ key: key(4), id: 4 },
		]);
		expect((await index.get([key(1)])).records.get(key(1))!.meta).toBe(meta(1, "moved.example.com"));
		expect((await index.get([key(2)])).records.size).toBe(0);
		// a re-put after compaction still finds the id in the base
		const again = await index.upsert({ keys: [key(3)], metas: ["v3"] });
		expect(again.ids).toEqual([3]);
		expect(again.allocated).toBe(0);
		expect((await manifestOf(store, "kv", again.rev)).nextId).toBe(102);

		// another handle sees the same state
		const other = await thread.open({ store, cache: false });
		expect((await other.get([key(3), key(0)])).records.get(key(3))!.meta).toBe("v3");
		expect((await other.listKeys({ prefix: "s3://bucket/0/" })).entries.map((e) => e.id)).toEqual([0, 3]);
		other.close();
		index.close();
	});

	it("validates keyed batches", async () => {
		const store = new MemoryBlobStore();
		await createLexicalGeneration({ store, gen: "kv2", lex: {} });
		const index = await thread.open({ store, cache: false });
		await expect(index.upsert({ keys: ["a", "a"], metas: ["x", "y"] })).rejects.toThrow(/duplicate keys/);
		await expect(index.upsert({ keys: ["a", ""], metas: ["x", "y"] })).rejects.toThrow(/keys\[1\]/);
		await expect(index.upsert({ keys: ["a"], ids: [1, 2], metas: ["x", "y"] })).rejects.toThrow(/1 keys for 2 ids/);
		await expect(index.upsert({ keys: ["k".repeat(5000)], metas: ["x"] })).rejects.toThrow(/longer than 4096/);
		await expect(index.upsert({ metas: ["x"] })).rejects.toThrow(/needs ids or keys/);
		await expect(index.deleteKeys([])).rejects.toThrow(/no keys/);
		// metas and texts are strings: anything else is refused before the wasm sees it (it used to trap and reset
		// the instance), and the instance is not reset
		const resets = thread.resets;
		await expect(index.upsert({ keys: ["a", "b"], metas: ["x", { k: 1 } as never] })).rejects.toMatchObject({ status: 400, message: expect.stringMatching(/metas\[1\] is object, not a string or null/) });
		await expect(index.upsert({ keys: ["a"], texts: [42 as never] })).rejects.toThrow(/texts\[0\] is number/);
		await expect(index.upsert({ keys: ["a"], texts: [["x"] as never], chunk: true })).rejects.toThrow(/texts\[0\] is an array/);
		expect(thread.resets).toBe(resets);
		// a key alone is a document (an empty docs entry): get returns it with an empty meta
		const only = await index.upsert({ keys: ["just-a-key"] });
		expect(only.ids).toEqual([0]);
		expect((await index.get(["just-a-key"])).records.get("just-a-key")).toEqual({ key: "just-a-key", id: 0, meta: "", attrs: 0 });
		index.close();
		// a generation without a lexical configuration has no keys
		const b = await buildGen("kd0", 2, { store: new MemoryBlobStore() });
		const dense = await thread.open({ store: b.store, cache: false });
		await expect(dense.upsert({ keys: ["a"], vectors: unitRows(1, b.dim, rng(1)) })).rejects.toThrow(/keys need a lexical configuration/);
		expect((await dense.get(["a"])).records.size).toBe(0);
		expect((await dense.listKeys({})).entries).toEqual([]);
		dense.close();
	});

	it("dense generation: ids are allocated past the build ids and the vectors are found", async () => {
		const b = await buildGen("kd", 3, { n: 100 });
		const index = await thread.open({ store: b.store, cache: false });
		await index.configureLexical({ bucketKeys: 4 });
		const r = rng(7);
		const v = unitRows(3, b.dim, r);
		const up = await index.upsert({ keys: ["doc/a", "doc/b", "doc/c"], vectors: v, metas: ["A", "B", "C"], attrs: [1, 2, 3] });
		expect(up.ids).toEqual([100, 101, 102]); // the base holds ids [0, 100)
		expect(up.allocated).toBe(3);
		expect(up.cells.length).toBe(3);
		expect(index.manifest.nextId).toBe(103);
		const got = await index.get(["doc/b"]);
		expect(got.records.get("doc/b")).toEqual({ key: "doc/b", id: 101, meta: "B", attrs: 2 });
		const q = await index.query(v.subarray(b.dim, 2 * b.dim), { topK: 1 });
		expect(q.hits[0].id).toBe(101);
		// an explicit id above the counter moves it
		await index.upsert({ ids: [500], vectors: unitRows(1, b.dim, r) });
		expect(index.manifest.nextId).toBe(501);
		const more = await index.upsert({ keys: ["doc/d"], vectors: unitRows(1, b.dim, r) });
		expect(more.ids).toEqual([501]);
		// a hybrid search resolves the doc of a keyed document
		const s = await index.search({ vector: v.subarray(0, b.dim), text: "A", topK: 3, docs: true });
		expect(s.hits.find((h) => h.id === 100)?.doc?.meta).toBe("A");
		index.close();
	});
});
