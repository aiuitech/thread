/**
 * The must-clause (`ids` on `search` and `lexicalSearch`): only the given ids may be returned, and the BM25 list keeps
 * to them inside its scan, so the top k is the top k of the set however many other documents score higher. Across
 * two segments, with a tombstoned member, and with no ids.
 */
import { describe, expect, it } from "vitest";
import { createLexicalGeneration, MemoryBlobStore } from "../src/index.js";
import { thread } from "./helpers.js";

describe("the must-clause", () => {
	it("returns the best of the set, not of the index", async () => {
		const store = new MemoryBlobStore();
		await createLexicalGeneration({ store, gen: "m", lex: { tokenizer: { identifiers: "keep" } } });
		const index = await thread.open({ store, cache: false });
		// 600 documents with "report"; the ones with more of it rank first in the whole index. The set's members rank last.
		const n = 600;
		const text = (i: number) => (i % 200 === 0 ? "report owner:bob" : `report report report filler${i % 13}`);
		await index.upsert({ ids: Array.from({ length: 300 }, (_, i) => i), texts: Array.from({ length: 300 }, (_, i) => text(i)) });
		await index.upsert({ ids: Array.from({ length: 300 }, (_, i) => i + 300), texts: Array.from({ length: 300 }, (_, i) => text(i + 300)) });
		const set = [0, 200, 400];

		// Without the clause, the set is below the depth of a top 50.
		const whole = await index.search({ text: "report", topK: 50 });
		expect(whole.hits.some((h) => set.includes(h.id))).toBe(false);
		// With it: the set, from both segments, and nothing else.
		const kept = await index.search({ text: "report", topK: 50, ids: set });
		expect(kept.hits.map((h) => h.id).sort((a, b) => a - b)).toEqual(set);
		expect((await index.lexicalSearch("report", 2, { ids: set })).hits).toHaveLength(2);
		// A member that was deleted is not returned; an empty set returns nothing.
		await index.delete([200]);
		const after = await thread.open({ store, cache: false });
		expect((await after.search({ text: "report", topK: 50, ids: set })).hits.map((h) => h.id).sort((a, b) => a - b)).toEqual([0, 400]);
		expect((await after.search({ text: "report", topK: 50, ids: [] })).hits).toEqual([]);
	});
});
