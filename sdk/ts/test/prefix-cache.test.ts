/**
 * Two indexes under different prefixes of one bucket share one cache (the edge cache, or one in-process cache per
 * isolate). Their objects have the same keys after the prefix (`idx/current.json`, `idx/g1/seg/1.thread`), so the
 * cache keys must carry the prefix, or one index reads the other's pointer, manifest or segment.
 */
import { describe, expect, it } from "vitest";
import { createLexicalGeneration, MemoryBlobStore, MemoryCache, PrefixedBlobStore } from "../src/index.js";
import { FakeCache, thread } from "./helpers.js";

describe("open with a prefix and a shared cache", () => {
	it("keeps each index's cached objects apart (Cache API)", async () => {
		const bucket = new MemoryBlobStore();
		const texts = { "a/": "the plumber fixed the kitchen tap", "b/": "coffee beans and a grinder for the office, a long note that makes this segment bigger" };
		for (const [prefix, text] of Object.entries(texts)) {
			const store = new PrefixedBlobStore(bucket, prefix);
			await createLexicalGeneration({ store, gen: "g1", lex: { docs: true, text: true } });
			const ix = await thread.open({ store, cache: false });
			await ix.upsert({ keys: [prefix + "doc"], texts: [text], metas: [prefix] });
			ix.close();
		}
		const cache = new FakeCache();
		for (const prefix of ["b/", "a/"]) {
			const ix = await thread.open({ store: bucket, prefix, cache, refreshInterval: 0 });
			const q = prefix === "a/" ? "plumber kitchen" : "coffee grinder";
			const r = await ix.search({ text: q, topK: 5, docs: "text" });
			expect(r.hits.map((h) => h.doc?.meta)).toEqual([prefix]);
			ix.close();
		}
		expect([...cache.objects.keys()].some((k) => k.includes("/a/idx/"))).toBe(true);
	});

	it("keeps each index's cached objects apart (in-process cache, prefix or PrefixedBlobStore)", async () => {
		const store = new MemoryBlobStore();
		const cache = new MemoryCache();
		const lex = { tokenizer: { identifiers: "keep" }, docs: true, text: true };
		await createLexicalGeneration({ store: new PrefixedBlobStore(store, "a/"), gen: "g1", lex });
		await createLexicalGeneration({ store: new PrefixedBlobStore(store, "b/"), gen: "g1", lex });

		const a = await thread.open({ store, prefix: "a/", cache });
		await a.upsert({ keys: ["k"], metas: ["A"], texts: ["alpha only"] });
		a.close();
		// The second index is opened on a PrefixedBlobStore instead of `prefix`: its cache keys must carry it too.
		const b = await thread.open({ store: new PrefixedBlobStore(store, "b/"), cache });
		await b.upsert({ keys: ["k", "k2"], metas: ["B", "B2"], texts: ["bravo only", "bravo two words"] });
		b.close();

		// Fresh opens read through the shared cache: each must see its own current.json, manifest and segments.
		const a2 = await thread.open({ store, prefix: "a/", cache });
		const b2 = await thread.open({ store: new PrefixedBlobStore(store, "b/"), cache });
		expect((await a2.get(["k"])).records.get("k")?.meta).toBe("A");
		expect((await b2.get(["k"])).records.get("k")?.meta).toBe("B");
		expect((await a2.search({ text: "bravo", topK: 5 })).hits.length).toBe(0);
		expect((await b2.search({ text: "bravo", topK: 5, docs: true })).hits.length).toBe(2);
		expect((await a2.search({ text: "alpha", topK: 5 })).hits.length).toBe(1);
		a2.close();
		b2.close();
	});
});
