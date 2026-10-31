/**
 * Two indexes under different prefixes of one bucket share one cache. Their objects have the same keys after the
 * prefix (`idx/g1/seg/1.thread`), so the cache keys must carry the prefix, or one index reads the other's segment.
 */
import { describe, expect, it } from "vitest";
import { createLexicalGeneration, MemoryBlobStore, PrefixedBlobStore } from "../src/index.js";
import { FakeCache, thread } from "./helpers.js";

describe("open with a prefix and a shared cache", () => {
	it("keeps each index's cached objects apart", async () => {
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
});
