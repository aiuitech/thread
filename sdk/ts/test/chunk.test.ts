/** Flat word-window chunking against the fixtures shared with the Python SDK and the CLI, and chunked upserts. */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { chunkIndexOf, chunkKey, chunkText, createLexicalGeneration, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_WORDS, MemoryBlobStore, resolveChunkOptions, ThreadError, type Chunk, type ChunkOptions } from "../src/index.js";
import { buildGen, rng, thread, unitRows } from "./helpers.js";

const fx = JSON.parse(fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), "fixtures/chunks.json"), "utf8")) as {
	defaults: { words: number; overlap: number };
	cases: { name: string; text: string; words: number | null; overlap: number | null; chunks: Chunk[] }[];
	errors: { words: number; overlap: number }[];
	keys: { key: string; index: number; chunk: string }[];
	key_parse: { key: string; parent: string; index: number | null }[];
};

describe("chunkText", () => {
	it("has the fixture's defaults", () => {
		expect({ words: DEFAULT_CHUNK_WORDS, overlap: DEFAULT_CHUNK_OVERLAP }).toEqual(fx.defaults);
		expect(resolveChunkOptions()).toEqual(fx.defaults);
		expect(resolveChunkOptions(true)).toEqual(fx.defaults);
		expect(resolveChunkOptions({ overlap: 20 })).toEqual({ words: 200, overlap: 20 });
	});
	for (const c of fx.cases) {
		it(c.name, () => {
			const opts = { ...(c.words !== null ? { words: c.words } : {}), ...(c.overlap !== null ? { overlap: c.overlap } : {}) };
			const got = chunkText(c.text, opts);
			expect(got).toEqual(c.chunks);
			// start/end are code-point offsets into the text
			const cps = Array.from(c.text);
			for (const k of got) expect(cps.slice(k.start, k.end).join("")).toBe(k.text);
		});
	}
	it("rejects bad parameters", () => {
		for (const e of fx.errors) expect(() => chunkText("a b c", e)).toThrow(RangeError);
		expect(() => chunkText("a", { words: 1.5 })).toThrow(RangeError);
		expect(() => chunkText("a", false as unknown as ChunkOptions)).toThrow(RangeError);
	});
	it("names chunks by key", () => {
		for (const k of fx.keys) expect(chunkKey(k.key, k.index)).toBe(k.chunk);
		for (const p of fx.key_parse) expect(chunkIndexOf(p.key, p.parent)).toBe(p.index);
	});
});

const text = (n: number, tag: string) => Array.from({ length: n }, (_, i) => `${tag}${i}`).join(" ");

describe("chunked upserts", () => {
	it("writes one keyed document per chunk with the parent's meta and attrs, embeds the chunks, and drops stale chunks on a re-put", async () => {
		const store = new MemoryBlobStore();
		await createLexicalGeneration({ store, gen: "docs", lex: { text: true, tokenizer: { stopwords: null, stemmer: null } } });
		const index = await thread.open({ store, cache: false });
		await index.configureAttrs({ tenant: { bits: 4 } });
		const r = await index.upsert({ keys: ["a", "b"], texts: [text(5, "a"), text(2, "b")], metas: ["A", "B"], attrs: [{ tenant: 3 }, { tenant: 4 }], chunk: { words: 2, overlap: 1 } });
		// a: a0..a4 in windows of 2 with overlap 1 -> [a0 a1] [a1 a2] [a2 a3] [a3 a4]; b: one chunk
		expect(r.count).toBe(5);
		expect(r.ids).toEqual([0, 1, 2, 3, 4]);
		expect(r.chunks).toEqual({ documents: 2, count: 5, stale: 0 });
		const got = await index.get(["a#0", "a#3", "b#0", "a"], { text: true });
		expect([...got.records.keys()]).toEqual(["a#0", "a#3", "b#0"]);
		expect(got.records.get("a#0")).toMatchObject({ id: 0, meta: "A", text: "a0 a1" });
		expect(got.records.get("a#3")).toMatchObject({ id: 3, meta: "A", text: "a3 a4" });
		expect(got.records.get("b#0")).toMatchObject({ id: 4, meta: "B", text: "b0 b1" });
		expect(index.unpackAttrs(got.records.get("a#3")!.attrs)).toEqual({ tenant: 3 });
		expect(index.unpackAttrs(got.records.get("b#0")!.attrs)).toEqual({ tenant: 4 });
		expect((await index.listKeys({ prefix: "a#" })).entries.map((e) => e.key)).toEqual(["a#0", "a#1", "a#2", "a#3"]);
		const hit = await index.search({ text: "a2", topK: 10, docs: true });
		expect(hit.hits.map((h) => h.id).sort()).toEqual([1, 2]);
		const m = JSON.parse(new TextDecoder().decode((await store.get(`idx/docs/manifest.${r.rev}.json`))!)) as { change: Record<string, unknown> };
		expect(m.change).toMatchObject({ op: "upsert", keys: 5, chunk: { words: 2, overlap: 1, documents: 2, stale: 0 } });

		// a re-put with fewer chunks keeps the ids of the chunks it still has and tombstones the leftover ones
		const r2 = await index.upsert({ keys: ["a"], texts: [text(3, "a")], metas: ["A2"], chunk: { words: 2, overlap: 1 } });
		expect(r2.ids).toEqual([0, 1]);
		expect(r2.allocated).toBe(0);
		expect(r2.chunks).toEqual({ documents: 1, count: 2, stale: 2 });
		expect((await index.listKeys({ prefix: "a#" })).entries.map((e) => e.key)).toEqual(["a#0", "a#1"]);
		expect((await index.get(["a#0"])).records.get("a#0")!.meta).toBe("A2");
		expect((await index.search({ text: "a4", topK: 10 })).hits).toEqual([]);
		expect((await index.search({ text: "a3", topK: 10 })).hits).toEqual([]);
		const m2 = JSON.parse(new TextDecoder().decode((await store.get(`idx/docs/manifest.${r2.rev}.json`))!)) as { tombstones: { count: number }[] };
		expect(m2.tombstones.at(-1)!.count).toBe(4); // ids 0, 1 (replaced) + 2, 3 (stale)

		// `replace: false` still tombstones stale chunks
		const r3 = await index.upsert({ keys: ["a"], texts: ["a0"], chunk: true, replace: false });
		expect(r3.chunks).toEqual({ documents: 1, count: 1, stale: 1 });
		expect((await index.listKeys({ prefix: "a#" })).entries.map((e) => e.key)).toEqual(["a#0"]);

		// the defaults: 200 words, no overlap; a text with no words is one empty chunk
		const r4 = await index.upsert({ keys: ["c", "d"], texts: [text(450, "c"), "   "], chunk: true });
		expect(r4.chunks).toEqual({ documents: 2, count: 4, stale: 0 });
		expect((await index.listKeys({ prefix: "c#" })).entries.length).toBe(3);
		expect((await index.get(["d#0"], { text: true })).records.get("d#0")!.text).toBe("");
		index.close();
	});

	it("embeds the chunks for a dense generation and validates the input", async () => {
		const { store, dim } = await buildGen("cg", 3, { n: 100, dim: 8 });
		const index = await thread.open({ store, cache: false });
		await index.configureLexical({ text: true });
		const seen: string[][] = [];
		const r9 = rng(9);
		const embed = async (texts: string[]) => {
			seen.push(texts);
			return unitRows(texts.length, dim, r9); // flat n x dim, as `vectors` takes it
		};
		const r = await index.upsert({ keys: ["p"], texts: [text(7, "p")], chunk: { words: 3 }, embed });
		expect(r.count).toBe(3);
		expect(seen).toEqual([["p0 p1 p2", "p3 p4 p5", "p6"]]);
		expect(r.distinctCells).toBeGreaterThan(0);
		const q = await index.query((await embed(["p6"])).subarray(0, dim), { topK: 3 });
		expect(q.hits.length).toBe(3);
		expect(r.ids).toEqual([100, 101, 102]);
		// embed without chunk: the texts as given
		const r2 = await index.upsert({ ids: [900], texts: ["whole document"], embed });
		expect(r2.count).toBe(1);
		expect(seen.at(-1)).toEqual(["whole document"]);

		await expect(index.upsert({ ids: [1], texts: ["x"], chunk: true })).rejects.toThrow(/by key/);
		await expect(index.upsert({ keys: ["k"], chunk: true })).rejects.toThrow(/texts/);
		await expect(index.upsert({ keys: ["k"], texts: ["x"], vectors: unitRows(1, dim, rng(1)), chunk: true })).rejects.toThrow(/embed/);
		await expect(index.upsert({ keys: ["k"], texts: ["x"], vectors: unitRows(1, dim, rng(1)), embed })).rejects.toThrow(/embed/);
		await expect(index.upsert({ keys: ["k"], texts: ["x"], chunk: { words: 0 } })).rejects.toThrow(ThreadError);
		await expect(index.upsert({ keys: ["k"], texts: ["x y"], chunk: { words: 1 }, embed: async () => [] })).rejects.toThrow(/embed returned 0 vectors for 2/);
		await expect(index.upsert({ keys: ["k"], texts: ["x y"], chunk: { words: 1 }, embed: async () => new Float32Array(3) })).rejects.toThrow(/expected 2 x 8/);
		await expect(index.upsert({ keys: ["k"], texts: [text(5000, "z")], chunk: { words: 1 } })).rejects.toThrow(/4096/);
		index.close();
	});
});
