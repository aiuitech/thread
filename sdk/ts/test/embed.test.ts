/**
 * Embedding models: the manifest `embedding` block against the fixture shared with the Python SDK, the provider
 * embedders over fake transports, and a dense generation created from nothing for a model (`createGeneration`) that
 * embeds texts on upsert and search.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { CREATE_DEFAULTS, EMBED_BATCH_DEFAULTS, EMBEDDING_PROVIDERS, MemoryBlobStore, ThreadError, createGeneration, embeddingBatches, embeddingInputs, keys, normalizeRows, resolveEmbedder, splitRows, validateEmbeddingConfig, type BlobStore, type EmbedKind, type EmbeddingConfig } from "../src/index.js";
import { bruteForce, dot, rng, thread, unitRows } from "./helpers.js";

const fx = JSON.parse(fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), "fixtures/embedding.json"), "utf8")) as {
	providers: string[];
	batch_defaults: Record<string, number>;
	configs: { name: string; config: unknown; dim: number | null; resolved?: EmbeddingConfig; error?: boolean }[];
	inputs: { config: EmbeddingConfig; kind: EmbedKind; texts: string[]; inputs: string[] }[];
	batches: { config: EmbeddingConfig; n: number; batches: number[] }[];
	create_defaults: { nprobe: number; rerank: number; pc: number };
	create_manifest: Record<string, unknown>;
};

const DIM = 16;

/**
 * A deterministic stand-in for a model: a text's vector is the sum of a seeded random unit vector per word (so texts
 * sharing words are close), with a kind-dependent twist when `kinds` is set (to check which kind was asked for).
 */
function fakeModel(dim = DIM): (texts: readonly string[], kind?: EmbedKind) => Float32Array[] {
	const words = new Map<string, Float32Array>();
	const wordVec = (w: string) => {
		let v = words.get(w);
		if (!v) {
			let h = 2166136261;
			for (const ch of w) h = Math.imul(h ^ ch.charCodeAt(0), 16777619) >>> 0;
			v = unitRows(1, dim, rng(h));
			words.set(w, v);
		}
		return v;
	};
	return (texts) =>
		texts.map((t) => {
			const out = new Float32Array(dim);
			const ws = t.split(/\s+/).filter(Boolean);
			if (!ws.length) out[0] = 1;
			for (const w of ws) {
				const v = wordVec(w);
				for (let d = 0; d < dim; d++) out[d] += v[d];
			}
			return out;
		});
}

describe("embedding config (fixture)", () => {
	it("has the fixture's providers and batch defaults", () => {
		expect([...EMBEDDING_PROVIDERS]).toEqual(fx.providers);
		expect(EMBED_BATCH_DEFAULTS).toEqual(fx.batch_defaults);
		expect(CREATE_DEFAULTS).toEqual(fx.create_defaults);
	});
	for (const c of fx.configs) {
		it(c.name, () => {
			const dim = c.dim ?? undefined;
			if (c.error) expect(() => validateEmbeddingConfig(c.config, dim)).toThrow(ThreadError);
			else expect(validateEmbeddingConfig(c.config, dim)).toEqual(c.resolved);
		});
	}
	it("prefixes inputs by kind", () => {
		for (const c of fx.inputs) expect(embeddingInputs(c.config, c.texts, c.kind)).toEqual(c.inputs);
	});
	it("batches provider calls", () => {
		for (const c of fx.batches) expect(embeddingBatches(c.config, c.n)).toEqual(c.batches);
	});
	it("splits and normalizes rows", () => {
		expect(splitRows([1, 0, 0, 2], 2, 2, "x").map((r) => Array.from(r))).toEqual([
			[1, 0],
			[0, 2],
		]);
		expect(splitRows([[3, 4]], 1, 2, "x").map((r) => Array.from(r))).toEqual([[3, 4]]);
		expect(() => splitRows([1, 2, 3], 2, 2, "x")).toThrow(/3 floats/);
		expect(() => splitRows([[1, 2, 3]], 1, 2, "x")).toThrow(/3 dims/);
		expect(Array.from(normalizeRows([Float32Array.from([3, 4])], "x")[0])).toEqual([0.6000000238418579, 0.800000011920929]);
		expect(() => normalizeRows([new Float32Array(2)], "x")).toThrow(/zero/);
	});
});

describe("resolveEmbedder", () => {
	it("calls the Workers AI binding in batches of 100 with the options and the kind's prefix", async () => {
		const calls: { model: string; inputs: Record<string, unknown> }[] = [];
		const ai: { run(model: string, inputs: Record<string, unknown>): Promise<unknown> } = {
			run: async (model: string, inputs: Record<string, unknown>) => {
				calls.push({ model, inputs });
				const text = inputs.text as string[];
				return { shape: [text.length, 4], data: text.map((t, i) => [t.length, i, 1, 0]) };
			},
		};
		const cfg: EmbeddingConfig = { provider: "workers-ai", model: "@cf/baai/bge-small-en-v1.5", dim: 4, options: { pooling: "cls" }, queryPrefix: "Q: " };
		const embed = resolveEmbedder(cfg, { ai });
		const texts = Array.from({ length: 150 }, (_, i) => `t${i}`);
		const rows = await embed(texts, "document");
		expect(rows.length).toBe(150);
		expect(calls.map((c) => (c.inputs.text as string[]).length)).toEqual([100, 50]);
		expect(calls[0].model).toBe(cfg.model);
		expect(calls[0].inputs.pooling).toBe("cls");
		expect(Array.from(rows[7])).toEqual([2, 7, 1, 0]);
		expect(Array.from(rows[107])).toEqual([4, 7, 1, 0]);
		const q = await embed(["hello"], "query");
		expect((calls[2].inputs.text as string[])[0]).toBe("Q: hello");
		expect(q[0][0]).toBe(8);
		// A response of the wrong shape is a 502.
		ai.run = async () => ({ data: [[1, 2]] });
		await expect(embed(["x"], "query")).rejects.toThrow(/2 dims, expected 4/);
	});
	it("posts to an OpenAI-compatible endpoint and orders the rows by index", async () => {
		const reqs: { url: string; headers: Record<string, string>; body: Record<string, unknown> }[] = [];
		const f = (async (url: string, init: { headers: Record<string, string>; body: string }) => {
			const body = JSON.parse(init.body) as { input: string[] };
			reqs.push({ url, headers: init.headers, body });
			const data = body.input.map((t, i) => ({ object: "embedding", index: i, embedding: [i, t.length] })).reverse();
			return new Response(JSON.stringify({ object: "list", data }), { status: 200 });
		}) as unknown as typeof fetch;
		const cfg: EmbeddingConfig = { provider: "openai", model: "text-embedding-3-small", dim: 2, options: { dimensions: 2 }, baseUrl: "https://example.test/v1/", maxBatch: 2 };
		const embed = resolveEmbedder(cfg, { apiKey: "sk-test", fetch: f });
		const rows = await embed(["a", "bb", "ccc"], "document");
		expect(rows.map((r) => Array.from(r))).toEqual([
			[0, 1],
			[1, 2],
			[0, 3],
		]);
		expect(reqs.length).toBe(2);
		expect(reqs[0].url).toBe("https://example.test/v1/embeddings");
		expect(reqs[0].headers.authorization).toBe("Bearer sk-test");
		expect(reqs[0].body).toEqual({ dimensions: 2, model: "text-embedding-3-small", input: ["a", "bb"] });
		// HTTP errors carry the status.
		const bad = resolveEmbedder(cfg, { fetch: (async () => new Response("nope", { status: 401 })) as unknown as typeof fetch });
		await expect(bad(["a"], "query")).rejects.toThrow(/HTTP 401/);
	});
	it("uses the Workers AI REST API outside Workers", async () => {
		let seen: { url: string; auth: string; body: Record<string, unknown> } | null = null;
		const f = (async (url: string, init: { headers: Record<string, string>; body: string }) => {
			seen = { url, auth: init.headers.authorization, body: JSON.parse(init.body) as Record<string, unknown> };
			return new Response(JSON.stringify({ success: true, result: { shape: [1, 2], data: [[1, 2]] } }), { status: 200 });
		}) as unknown as typeof fetch;
		const cfg: EmbeddingConfig = { provider: "workers-ai", model: "@cf/baai/bge-m3", dim: 2 };
		const rows = await resolveEmbedder(cfg, { accountId: "acct", apiToken: "tok", fetch: f })(["x"], "query");
		expect(Array.from(rows[0])).toEqual([1, 2]);
		expect(seen!.url).toBe("https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/baai/bge-m3");
		expect(seen!.auth).toBe("Bearer tok");
		expect(seen!.body).toEqual({ text: ["x"] });
		expect(() => resolveEmbedder(cfg, {})).toThrow(/accountId and apiToken/);
	});
	it("needs an embedder for a custom provider and passes the kind through", async () => {
		const cfg: EmbeddingConfig = { provider: "custom", model: "m", dim: 2, maxBatch: 2, docPrefix: "D " };
		expect(() => resolveEmbedder(cfg)).toThrow(/needs an embedder/);
		const seen: [string[], EmbedKind][] = [];
		const embed = resolveEmbedder(cfg, {
			embedder: (texts, kind) => {
				seen.push([texts, kind]);
				return texts.map(() => [1, 1]);
			},
		});
		expect((await embed(["a", "b", "c"], "document")).length).toBe(3);
		expect(seen).toEqual([
			[["D a", "D b"], "document"],
			[["D c"], "document"],
		]);
	});
});

describe("createGeneration", () => {
	it("publishes one empty cell for a model, then embeds upserts and searches", async () => {
		const store = new MemoryBlobStore();
		const model = fakeModel();
		const embedding: EmbeddingConfig = { provider: "custom", model: "fake-16", dim: DIM };
		const r = await thread.createGeneration({ store, gen: "e1", embedding, attrs: { tenant: { bits: 8 } } });
		expect(r.rev).toBe(0);
		expect(r.currentUpdated).toBe(true);
		expect(r.manifest).toMatchObject({ ...fx.create_manifest, gen: "e1", dim: DIM, profile: "edge", defaults: fx.create_defaults, embedding: { provider: "custom", model: "fake-16", dim: DIM }, lex: {} });
		expect(r.manifest.compaction).toMatchObject({ maxDeltas: 16, maxCellEntries: 3000 });
		for (const k of r.keys) expect(await store.head(k)).not.toBeNull();
		expect(JSON.parse(new TextDecoder().decode((await store.get(keys.slabsTable("e1", 0)))!)).length).toBe(1);
		expect(new Float32Array((await store.get(`idx/e1/fine.f32`))!.slice().buffer).length).toBe(DIM);
		// A second creation of the name is refused.
		await expect(thread.createGeneration({ store, gen: "e1", dim: DIM })).rejects.toThrow(/already exists/);

		// Without a way to call the model, the index reads and takes vectors but cannot embed.
		const plain = await thread.open({ store, cache: false });
		expect(plain.hasDense).toBe(true);
		expect(plain.canEmbed).toBe(false);
		expect(plain.embedding).toEqual(embedding);
		await expect(plain.embed(["x"], "query")).rejects.toThrow(/cannot embed/);
		expect((await plain.query(unitRows(1, DIM, rng(1)))).hits).toEqual([]); // the empty cell
		await expect(plain.search({ text: "x", embed: true })).rejects.toThrow(/cannot embed/);
		plain.close();

		const calls: EmbedKind[] = [];
		const index = await thread.open({
			store,
			cache: false,
			embed: (texts, kind) => {
				calls.push(kind);
				return model(texts);
			},
		});
		expect(index.canEmbed).toBe(true);
		const docs = ["red apples and pears", "green pears", "blue sky over the sea", "the sea at night", "apples in the sky", "night market with pears and apples", "an empty sky", "red sea"];
		const ids = docs.map((_, i) => i);
		const up = await index.upsert({ keys: docs.map((_, i) => `d/${i}`), texts: docs, metas: docs, attrs: docs.map((_, i) => ({ tenant: i % 2 })) });
		expect(calls).toEqual(["document"]);
		expect(up.ids).toEqual(ids);
		expect(up.keys.delta).not.toBeNull();
		expect(up.cells.every((c) => c === 0)).toBe(true);
		expect(index.rev).toBe(1);

		// search({text}) embeds the text (as a query) and fuses it with BM25.
		const hybrid = await index.search({ text: "pears", topK: 4, docs: true });
		expect(calls).toEqual(["document", "query"]);
		expect(hybrid.mode).toBe("hybrid");
		expect(hybrid.hits.length).toBe(4);
		expect(hybrid.hits.every((h) => h.dense || h.lexical)).toBe(true);
		expect(hybrid.hits.some((h) => h.dense && h.lexical)).toBe(true);
		expect(hybrid.hits[0].doc?.meta).toContain("pears");
		// embed: false keeps the text lexical-only; a given vector is used as is.
		const lexOnly = await index.search({ text: "pears", embed: false, topK: 3 });
		expect(lexOnly.mode).toBe("lexical");
		expect(calls.length).toBe(2);
		const q = normalizeRows(model(["apples"]), "q")[0];
		const dense = await index.search({ vector: q, topK: 3 });
		expect(dense.mode).toBe("dense");
		const want = bruteForce(normalizeRows(model(docs), "d").reduce((all, r, i) => (all.set(r, i * DIM), all), new Float32Array(docs.length * DIM)), DIM, q, 3);
		expect(dense.hits.map((h) => h.id)).toEqual(want);
		// A query the index embeds ranks the same as the vector given by hand.
		const viaText = await index.search({ text: "apples", topK: 3, weights: { lexical: 0 } });
		expect(viaText.hits.map((h) => h.id).slice(0, 1)).toEqual(want.slice(0, 1));
		expect(viaText.hits.filter((h) => h.dense).map((h) => h.id)).toEqual(want);
		// Filters apply to the embedded dense list too.
		const filtered = await index.search({ text: "apples", topK: 3, filter: { tenant: 1 } });
		expect(filtered.hits.every((h) => h.id % 2 === 1)).toBe(true);
		// Keys and docs work as for any generation; a text-only upsert (embed: false) adds no vector.
		expect((await index.get(["d/3"])).records.get("d/3")?.meta).toBe(docs[3]);
		const textOnly = await index.upsert({ keys: ["d/8"], texts: ["pears pears pears"], embed: false });
		expect(textOnly.keys.delta).toBeNull();
		expect(calls.length).toBe(4); // the two searches by text above embedded their queries; nothing here
		// An explicit embed function overrides the index's.
		const own = await index.upsert({ keys: ["d/9"], texts: ["x"], embed: (texts) => texts.map(() => unitRows(1, DIM, rng(9))) });
		expect(own.keys.delta).not.toBeNull();
		expect(calls.length).toBe(4);
		// Vectors of the wrong size are rejected like any upsert's.
		await expect(index.upsert({ ids: [100], vectors: [new Float32Array(DIM + 1)] })).rejects.toThrow(/dims/);
		index.close();
	});

	it("is dense-only with lex: null and takes vectors by id without a model", async () => {
		const store = new MemoryBlobStore();
		const r = await createGeneration(thread.wasm, { store, gen: "v1", dim: 8, lex: null, profile: "local", defaults: { nprobe: 2 }, compaction: null, setCurrent: false });
		expect(r.manifest.lex).toBeUndefined();
		expect(r.manifest.embedding).toBeUndefined();
		expect(r.manifest.compaction).toBeUndefined();
		expect(r.manifest.defaults).toEqual({ nprobe: 2, rerank: 150, pc: 0 });
		expect(r.currentUpdated).toBe(false);
		expect(await store.head(keys.current())).toBeNull();
		const index = await thread.open({ store, gen: "v1", cache: false });
		expect(index.embedding).toBeNull();
		await expect(index.embed(["x"], "query")).rejects.toThrow(/names no embedding model/);
		const vectors = unitRows(300, 8, rng(3));
		for (let a = 0; a < 300; a += 100) await index.upsert({ ids: Array.from({ length: 100 }, (_, i) => a + i), vectors: vectors.subarray(a * 8, (a + 100) * 8), replace: false });
		expect(index.rev).toBe(3);
		const q = unitRows(1, 8, rng(4));
		const got = (await index.query(q, { topK: 5 })).hits.map((h) => h.id);
		expect(got).toEqual(bruteForce(vectors, 8, q, 5));
		await expect(index.upsert({ keys: ["k"], texts: ["t"] })).rejects.toThrow(/lexical configuration/);
		await expect(index.search({ text: "t" })).rejects.toThrow(/no lexical index/);
		index.close();
	});

	it("rejects bad options", async () => {
		const store = new MemoryBlobStore();
		await expect(thread.createGeneration({ store, gen: "x" })).rejects.toThrow(/embedding model .* or a vector size/);
		await expect(thread.createGeneration({ store, gen: "bad name", dim: 4 })).rejects.toThrow(/bad generation name/);
		await expect(thread.createGeneration({ store, gen: "x", dim: 4, embedding: { provider: "custom", model: "m", dim: 8 } })).rejects.toThrow(/does not match/);
		await expect(thread.createGeneration({ store, gen: "x", dim: 4, profile: "fast" as "edge" })).rejects.toThrow(/profile/);
		await expect(thread.createGeneration({ store, gen: "x", dim: 4, defaults: { nprobe: 0 } })).rejects.toThrow(/nprobe/);
		const readOnly = { get: store.get.bind(store), head: store.head.bind(store) } as unknown as BlobStore;
		await expect(thread.createGeneration({ store: readOnly, gen: "x", dim: 4 })).rejects.toThrow(/read-only/);
		expect(await store.head(keys.manifest("x", 0))).toBeNull();
	});

	it("searches texts of a chunked document embedded through the index's model", async () => {
		const store = new MemoryBlobStore();
		const model = fakeModel();
		await thread.createGeneration({ store, gen: "c1", embedding: { provider: "custom", model: "fake-16", dim: DIM } });
		const index = await thread.open({ store, cache: false, embed: (texts) => model(texts) });
		const text = Array.from({ length: 9 }, (_, i) => `w${i}`).join(" ");
		const r = await index.upsert({ keys: ["doc"], texts: [text], chunk: { words: 3 } });
		expect(r.chunks).toEqual({ documents: 1, count: 3, stale: 0 });
		expect(r.keys.delta).not.toBeNull();
		const hit = await index.search({ text: "w4", topK: 1, docs: true });
		expect(hit.hits[0].id).toBe(1);
		expect(hit.hits[0].lexical?.rank).toBe(1);
		expect((await index.get(["doc#1"])).records.get("doc#1")?.id).toBe(1);
		// The dense list ranks the chunks by their embeddings (the fake model is noisy, so compare with brute force).
		const q = normalizeRows(model(["w4"]), "q")[0];
		const rows = normalizeRows(model(["w0 w1 w2", "w3 w4 w5", "w6 w7 w8"]), "d");
		const scores = rows.map((v) => dot(v, q, 0, 0, DIM));
		const denseOnly = await index.search({ text: "w4", topK: 3, weights: { lexical: 0 } });
		expect(denseOnly.hits.filter((h) => h.dense).map((h) => h.id)).toEqual([0, 1, 2].sort((a, b) => scores[b] - scores[a]));
		index.close();
	});
});
