/**
 * End-to-end lexical and hybrid search in Node: a lexical-only generation (texts in, BM25 out, docs resolved),
 * then a dense generation given a lexical configuration (hybrid RRF), replace/delete through tombstones, segment
 * compaction, refresh on another handle and a wasm reset.
 */
import { describe, expect, it } from "vitest";
import { createLexicalGeneration, isMissingRanges, keys, LexSegment, MemoryBlobStore, rrf, ThreadError, type Manifest } from "../src/index.js";
import { buildGen, rng, thread, unitRows, w } from "./helpers.js";

const DOCS: { id: number; text: string; meta: string }[] = [
	{ id: 10, text: "The deployments failed on Tuesday after the budget review", meta: JSON.stringify({ title: "Deploy failure" }) },
	{ id: 3, text: "Retry the deployment, then retry the deployment again", meta: "Retry notes" },
	{ id: 1000, text: "Budget review for the galaxy budget", meta: "Budget" },
	{ id: 42, text: "galaxy galaxy budget", meta: "Galaxy" },
	{ id: 7, text: "", meta: "empty" },
	{ id: 99, text: "Nothing to see here, move along", meta: "Nothing" },
];

/** Reference BM25 (bm25s/Lucene arithmetic) over the live documents for a tokenized query. */
function referenceBm25(docs: { id: number; tokens: string[] }[], query: string[], k1 = 1.2, b = 0.75): { id: number; score: number }[] {
	const n = docs.length;
	const avg = docs.reduce((a, d) => a + d.tokens.length, 0) / n;
	const out: { id: number; score: number }[] = [];
	for (const d of docs) {
		let s = 0;
		for (const q of query) {
			const df = docs.filter((x) => x.tokens.includes(q)).length;
			const tf = d.tokens.filter((t) => t === q).length;
			if (!tf) continue;
			const idf = Math.fround(Math.log(1 + (n - df + 0.5) / (df + 0.5)));
			const norm = k1 * (1 - b + (b * d.tokens.length) / avg);
			s = Math.fround(s + Math.fround(idf * (tf / (norm + tf))));
		}
		if (s > 0) out.push({ id: d.id, score: s });
	}
	return out.sort((a, b) => b.score - a.score || a.id - b.id);
}

describe("lexical-only generation", () => {
	it("creates, upserts texts, searches by text with corpus-wide scores across segments, resolves docs", async () => {
		const store = new MemoryBlobStore();
		const created = await createLexicalGeneration({ store, gen: "lx", lex: { text: true, bucketTerms: 4, bucketDocs: 2 } });
		expect(created.rev).toBe(0);
		const index = await thread.open({ store, cache: false });
		expect(index.gen).toBe("lx");
		expect(index.hasDense).toBe(false);
		expect(index.lexConfig?.k1).toBe(1.2);
		await expect(index.query(new Float32Array(4))).rejects.toThrow(/no dense index/);
		await expect(index.search({ text: "x", vector: new Float32Array(4) })).rejects.toThrow(/no dense index/);

		// two upserts = two segments; tokens of the query stem like the documents ("deployment" -> "deploy")
		const first = DOCS.slice(0, 3);
		const second = DOCS.slice(3);
		const up1 = await index.upsert({ ids: first.map((d) => d.id), texts: first.map((d) => d.text), metas: first.map((d) => d.meta) });
		expect(up1.rev).toBe(1);
		expect(up1.keys.segment).toBe("idx/lx/seg/1.thread");
		expect(up1.keys.delta).toBeNull();
		expect(up1.keys.tombstones).toBe("idx/lx/tombstones.1.u32");
		const up2 = await index.upsert({ ids: second.map((d) => d.id), texts: second.map((d) => d.text), metas: second.map((d) => d.meta), replace: false });
		expect(up2.rev).toBe(2);
		expect(up2.keys.segment).toBe("idx/lx/seg/2.thread");
		expect(up2.keys.tombstones).toBeNull();
		expect(index.segments.length).toBe(2);
		expect(index.segments.map((s) => s.addedRev)).toEqual([1, 2]);

		const tokens = await index.tokenize("Deployment retries");
		expect(tokens).toEqual(["deploy", "retri"]);
		const res = await index.search({ text: "deployment retries", topK: 5, docs: "text" });
		expect(res.mode).toBe("lexical");
		expect(res.hits.map((h) => h.id)).toEqual([3, 10]);
		expect(res.hits[0].lexical?.rank).toBe(1);
		expect(res.hits[0].doc).toEqual({ id: 3, meta: "Retry notes", text: DOCS[1].text });
		expect(res.hits[1].doc?.meta).toBe(JSON.stringify({ title: "Deploy failure" }));
		expect(res.lexical?.tokens).toEqual(["deploy", "retri"]);
		expect(res.lexical?.found).toBe(2);
		expect(res.lexical?.stats.segments).toBe(2);
		expect(res.lexical?.stats.numDocs).toBe(6);
		expect(res.lexical?.stats.io.rounds).toBeGreaterThanOrEqual(0); // tiny segments are resident after open

		// scores equal the reference over the whole corpus although the documents sit in two segments
		const tok = new w.LexTokenizer("{}");
		const corpus = DOCS.map((d) => ({ id: d.id, tokens: tok.tokenize(d.text) }));
		for (const q of ["galaxy budget", "deployment", "budget review galaxy nothing", "zebra"]) {
			const want = referenceBm25(corpus, tok.tokenize(q)).slice(0, 10);
			const got = await index.lexicalSearch(q, 10);
			expect(got.hits.map((h) => h.id)).toEqual(want.map((x) => x.id));
			got.hits.forEach((h, i) => expect(h.score).toBeCloseTo(want[i].score, 6));
		}
		tok.free();

		// docs: unknown ids are absent, the empty document has an entry
		const d = await index.docs([7, 12345, 42], { text: true });
		expect(d.docs.get(7)).toEqual({ id: 7, meta: "empty", text: "" });
		expect(d.docs.has(12345)).toBe(false);
		expect(d.docs.get(42)?.meta).toBe("Galaxy");

		// a second handle sees the segments after refresh
		const other = await thread.open({ store, cache: false, refreshInterval: 0 });
		expect(other.rev).toBe(2);
		expect((await other.search({ text: "galaxy" })).hits.map((h) => h.id)).toEqual([42, 1000]);

		// replace id 3 with new text and metadata: the old copy is hidden by the tombstone set
		const up3 = await index.upsert({ ids: [3], texts: ["Completely different content about zebras"], metas: ["Zebra"] });
		expect(up3.rev).toBe(3);
		const r3 = await index.search({ text: "retry deployment", docs: true });
		expect(r3.hits.map((h) => h.id)).toEqual([10]);
		const z = await index.search({ text: "zebras", docs: true });
		expect(z.hits.map((h) => h.id)).toEqual([3]);
		expect(z.hits[0].doc?.meta).toBe("Zebra");
		// delete 1000: gone from search and docs
		await index.delete([1000]);
		expect((await index.search({ text: "budget" })).hits.map((h) => h.id)).toEqual([42, 10]);
		expect((await index.docs([1000])).docs.size).toBe(0);
		expect(index.segments.length).toBe(3);
		expect(index.manifest.tombstones?.length).toBe(3);

		// compaction: one base segment, same answers, replaced/deleted documents gone for good
		const before = await index.search({ text: "budget galaxy deployment zebras", topK: 10, docs: "text" });
		const c = await index.compactSegments();
		expect(c.merged).toBe(3);
		expect(c.key).toBe("idx/lx/seg/base.5.thread");
		expect(c.count).toBe(5); // 10, 3, 42, 7, 99
		expect(index.rev).toBe(5);
		expect(index.segments.length).toBe(1);
		expect(index.segments[0].base).toBe(true);
		const after = await index.search({ text: "budget galaxy deployment zebras", topK: 10, docs: "text" });
		expect(after.hits.map((h) => [h.id, h.doc?.meta, h.doc?.text])).toEqual(before.hits.map((h) => [h.id, h.doc?.meta, h.doc?.text]));
		// Before the compaction the replaced and deleted copies still counted in N and avg_len (as Lucene's deleted
		// documents do); afterwards the scores equal a fresh index over the live documents.
		expect(before.lexical?.stats.numDocs).toBe(7); // 3 + 3 + 1 (the replacement), dead copies included
		expect(after.lexical?.stats.numDocs).toBe(4); // the empty document 7 has no postings and leaves the lexical count
		const tok2 = new w.LexTokenizer("{}");
		const live = [
			{ id: 10, tokens: tok2.tokenize(DOCS[0].text) },
			{ id: 3, tokens: tok2.tokenize("Completely different content about zebras") },
			{ id: 42, tokens: tok2.tokenize(DOCS[3].text) },
			{ id: 99, tokens: tok2.tokenize(DOCS[5].text) },
		];
		const want = referenceBm25(live, tok2.tokenize("budget galaxy deployment zebras"));
		tok2.free();
		expect(after.hits.map((h) => h.id)).toEqual(want.map((x) => x.id));
		after.hits.forEach((h, i) => expect(h.score).toBeCloseTo(want[i].score, 6));
		// a second compaction is a no-op (one segment, no newer tombstone set)
		const c2 = await index.compactSegments();
		expect(c2.merged).toBe(0);
		expect(c2.rev).toBe(5);
		// the other handle follows
		expect(await other.refresh()).toBe(true);
		expect((await other.search({ text: "zebras" })).hits.map((h) => h.id)).toEqual([3]);
		// the tokenizer of a generation with segments is frozen; other knobs may change
		await expect(index.configureLexical({ tokenizer: { stemmer: null }, text: true })).rejects.toThrow(/tokenizer/);
		await index.configureLexical({ text: true, bucketTerms: 64 });
		expect(index.lexConfig?.bucketTerms).toBe(64);
		index.close();
		other.close();
		await expect(index.search({ text: "x" })).rejects.toThrow(/closed/);
	});

	it("rejects bad input", async () => {
		const store = new MemoryBlobStore();
		await createLexicalGeneration({ store, gen: "lb", lex: {} });
		await expect(createLexicalGeneration({ store, gen: "lb", lex: {} })).rejects.toThrow(/exists/);
		const index = await thread.open({ store, cache: false });
		await expect(index.upsert({ ids: [1], texts: ["a", "b"] })).rejects.toThrow(/texts/);
		await expect(index.upsert({ ids: [1, 1], texts: ["a", "b"] })).rejects.toThrow(/duplicate/);
		await expect(index.upsert({ ids: [1] })).rejects.toThrow(/needs/);
		await expect(index.upsert({ ids: [1], vectors: [[1, 0]] })).rejects.toThrow(/no dense index/);
		await expect(index.search({})).rejects.toThrow(/text, a vector/);
		await expect(index.configureLexical({ lengths: "fuzzy" as "exact" })).rejects.toThrow(/lengths/);
		await expect(index.configureLexical({ tokenizer: { stemmer: "klingon" } })).rejects.toThrow(/stemmer/);
		// a query with only unknown terms is empty, not an error; an empty text is rejected
		await index.upsert({ ids: [1], texts: ["hello world"] });
		expect((await index.search({ text: "zzz" })).hits).toEqual([]);
		await expect(index.search({ text: "   " })).rejects.toThrow(/text, a vector/);
		index.close();
	});
});

describe("hybrid search on a dense generation", () => {
	it("configures lexical, upserts vectors + texts, fuses with RRF, survives a reset", async () => {
		const b = await buildGen("gh", 3);
		const index = await thread.open({ store: b.store, cache: false });
		expect(index.lexConfig).toBeNull();
		await expect(index.search({ text: "x" })).rejects.toThrow(/configureLexical/);
		const cfg = await index.configureLexical({ lengths: "byte", docs: true, text: false, defaults: { rrfK: 10, weights: { dense: 1, lexical: 0.5 } } });
		expect(cfg.rev).toBe(1);
		expect(index.lexConfig?.lengths).toBe("byte");
		// dense queries are unaffected by the configuration
		expect((await index.query(b.vectors.subarray(0, 16), { nprobe: 8 })).hits[0].id).toBe(0);

		// 1000..1003 get vectors and texts; 2000 text only (lexical-only document in a dense generation)
		const r = rng(5);
		const fresh = unitRows(4, b.dim, r);
		const up = await index.upsert({
			ids: [1000, 1001, 1002, 1003],
			vectors: fresh,
			texts: ["alpha bravo charlie", "alpha bravo", "charlie delta", "echo foxtrot"],
			metas: ["A", "B", "C", "D"],
			attrs: [1, 2, 3, 4],
		});
		expect(up.keys.delta).toBe("idx/gh/delta/1.vibd");
		expect(up.keys.segment).toBe("idx/gh/seg/1.thread");
		expect(up.cells.length).toBe(4);
		const textOnly = await index.upsert({ ids: [2000], texts: ["alpha alpha alpha"], metas: ["text only"] });
		expect(textOnly.keys.delta).toBeNull();
		expect(textOnly.keys.segment).toBe("idx/gh/seg/2.thread");

		// lexical alone
		const lex = await index.search({ text: "alpha", docs: true });
		expect(lex.mode).toBe("lexical");
		expect(lex.hits.map((h) => h.id)).toEqual([2000, 1001, 1000]);
		expect(lex.hits[0].doc?.meta).toBe("text only");
		expect(lex.hits[0].doc?.text).toBeUndefined();
		// dense alone through search()
		const den = await index.search({ vector: fresh.subarray(0, 16), topK: 3, dense: { nprobe: 8 } });
		expect(den.mode).toBe("dense");
		expect(den.hits[0].id).toBe(1000);
		expect(den.hits[0].dense?.rank).toBe(1);
		// hybrid: the fused order is RRF over the two lists with the manifest's rrfK and weights
		const hy = await index.search({ text: "alpha bravo", vector: fresh.subarray(16, 32), topK: 5, dense: { nprobe: 8 }, docs: true });
		expect(hy.mode).toBe("hybrid");
		expect(hy.params).toEqual({ topK: 5, recallK: 50, rrfK: 10, weights: { dense: 1, lexical: 0.5 } });
		const fused = rrf(
			[
				[1, hy.dense!.hits.map((h) => h.id)],
				[0.5, hy.lexical!.hits.map((h) => h.id)],
			],
			10,
			5,
		);
		expect(hy.hits.map((h) => h.id)).toEqual(Array.from(fused.ids));
		hy.hits.forEach((h, i) => expect(h.score).toBe(fused.scores[i]));
		expect(hy.hits[0].id).toBe(1001); // first in both lists
		expect(hy.hits[0].dense?.rank).toBe(1);
		expect(hy.hits[0].lexical?.rank).toBe(1);
		expect(hy.hits[0].doc?.meta).toBe("B");
		// a dense-only hit (a base vector) has no docs entry: doc is null
		const baseHit = hy.hits.find((h) => h.id < 1000);
		expect(baseHit?.doc).toBeNull();
		expect(baseHit?.lexical).toBeUndefined();
		// explicit weights / rrfK override the manifest defaults
		const hy2 = await index.search({ text: "alpha bravo", vector: fresh.subarray(16, 32), topK: 3, rrfK: 60, weights: { lexical: 2 }, recallK: 100, dense: { nprobe: 8 } });
		expect(hy2.params).toEqual({ topK: 3, recallK: 100, rrfK: 60, weights: { dense: 1, lexical: 2 } });

		// delete 1001 everywhere; replace 1000's text only (its vector copy stays... no: replace tombstones every copy)
		await index.delete([1001]);
		// 1000 holds both terms; 2000 only "alpha" (saturated tf)
		expect((await index.search({ text: "alpha bravo" })).hits.map((h) => h.id)).toEqual([1000, 2000]);
		expect((await index.query(fresh.subarray(16, 32), { topK: 1, nprobe: 8 })).hits[0].id).not.toBe(1001);
		await index.upsert({ ids: [1000], vectors: fresh.subarray(0, 16), texts: ["golf hotel"], metas: ["A2"] });
		expect((await index.search({ text: "alpha" })).hits.map((h) => h.id)).toEqual([2000]);
		const golf = await index.search({ text: "golf", vector: fresh.subarray(0, 16), docs: true, dense: { nprobe: 8 } });
		expect(golf.hits[0].id).toBe(1000);
		expect(golf.hits[0].doc?.meta).toBe("A2");

		// compaction of the segments keeps the dense deltas and tombstone sets as they are
		const c = await index.compactSegments();
		expect(c.merged).toBe(3);
		expect(index.manifest.deltas?.length).toBe(2); // the text-only upsert wrote no delta
		expect(index.manifest.tombstones?.length).toBe(4);
		expect((await index.search({ text: "golf hotel", docs: true })).hits.map((h) => [h.id, h.doc?.meta])).toEqual([[1000, "A2"]]);
		expect((await index.docs([1001, 2000])).docs.size).toBe(1);

		// wasm reset: the lexical readers are rebuilt
		thread.reset("test");
		const again = await index.search({ text: "alpha", vector: fresh.subarray(0, 16), dense: { nprobe: 8 }, docs: true, weights: { lexical: 1 } });
		expect(again.hits.map((h) => h.id)).toContain(2000);
		expect(again.hits.map((h) => h.id)).toContain(1000);
		index.close();
	});

	it("manifest round trip: segments, lex config and change records", async () => {
		const b = await buildGen("gm", 2);
		const index = await thread.open({ store: b.store, cache: false });
		await index.configureLexical({});
		await index.upsert({ ids: [5000], texts: ["some text"], metas: ["m"] });
		const m = JSON.parse(new TextDecoder().decode((await b.store.get(keys.manifest("gm", 2)))!)) as Manifest;
		expect(m.lex).toEqual({});
		expect(m.segments).toEqual([{ seq: 1, key: "idx/gm/seg/1.thread", addedRev: 2, count: 1, bytes: expect.any(Number), lex: true, docs: true, text: false }]);
		expect(m.change).toMatchObject({ op: "upsert", dense: false, lexical: true });
		expect(m.nextSeq).toBe(2);
		// the segment object is a container whose meta records the tokenizer and counts
		const meta = JSON.parse(w.segment_meta((await b.store.get("idx/gm/seg/1.thread"))!)) as { format: string; tokenizer: { stemmer: string }; lex: { num_docs: number }; docs: { n: number } };
		expect(meta.format).toBe("thread-seg");
		expect(meta.tokenizer.stemmer).toBe("english");
		expect(meta.lex.num_docs).toBe(1);
		expect(meta.docs.n).toBe(1);
		index.close();
	});
});

describe("LexSegment paging", () => {
	it("opens with a handful of fetches and searches in two rounds per call; MissingRanges is structured", async () => {
		const store = new MemoryBlobStore();
		const job = new w.SegmentJob("{}", 1.2, 0.75, "exact", 4, true, true, 4);
		const rows: string[] = [];
		// a wide vocabulary, so the buckets and lists of one query are a small part of the sections
		for (let i = 0; i < 3000; i++) {
			rows.push(`document number ${i} talks about topic${i % 170} and theme${i % 50} with word${i} and item${(i * 7) % 1000} filler`);
			job.add(i * 3, rows[i], `meta ${i}`);
		}
		const bytes = job.finish();
		await store.put("seg", bytes);
		const trace = { reads: 0, hits: 0, misses: 0, r206: 0, r200: 0, bytes: 0 };
		const io = { rounds: 0, ranges: 0, bytes: 0 };
		const seg = await LexSegment.open(w, store, { seq: 1, key: "seg", addedRev: 1, count: 200, bytes: bytes.byteLength, lex: true, docs: true, text: true }, thread.epoch, io, trace, { fillBytes: 0 });
		expect(seg.hasLex && seg.hasDocs).toBe(true);
		expect(seg.info?.numDocs).toBe(3000);
		expect(seg.info?.lengths).toBe("exact");
		expect(seg.info?.docs?.hasText).toBe(true);
		expect(io.rounds).toBeLessThanOrEqual(4);
		expect(io.bytes).toBeLessThan(bytes.byteLength / 4);
		const opened = io.bytes;
		// a search: lookup (one round) then search (one round)
		const q = { rounds: 0, ranges: 0, bytes: 0 };
		const infos = await seg.run(() => seg.reader.lookup(["topic3", "theme1", "nope"]), q, trace);
		expect(infos.length).toBe(9);
		expect(infos[0]).toBeGreaterThan(0);
		expect(infos[6]).toBe(-1);
		expect(q.rounds).toBe(1);
		const hits = await seg.run(() => seg.reader.search(infos, Float64Array.from([infos[0], infos[3], 0]), Float32Array.from([1, 1, 1]), 3000, seg.info!.totalLen, 1.2, 0.75, 5, new Uint8Array(0)), q, trace);
		expect(q.rounds).toBe(2);
		expect(new DataView(hits.buffer, hits.byteOffset).getUint32(0, true)).toBe(5);
		expect(q.bytes).toBeLessThan(bytes.byteLength / 4);
		// warm: the same call needs no fetch
		const q2 = { rounds: 0, ranges: 0, bytes: 0 };
		await seg.run(() => seg.reader.lookup(["topic3"]), q2, trace);
		expect(q2.rounds).toBe(0);
		// a raw MissingRanges error carries the ranges
		const cold = new w.SegmentReader(bytes.byteLength);
		try {
			cold.open();
			expect.unreachable();
		} catch (e) {
			expect(isMissingRanges(e)).toBe(true);
			if (isMissingRanges(e)) expect(e.ranges[0]).toEqual({ offset: 0, len: expect.any(Number) });
		}
		cold.free();
		// docs: entries then texts
		const d = { rounds: 0, ranges: 0, bytes: 0 };
		const got = (await seg.run(() => seg.reader.docs_get(Uint32Array.from([3, 4, 8997]), true), d, trace)) as ({ id: number; meta: string; text: string } | null)[];
		expect(got[0]).toEqual({ id: 3, meta: "meta 1", text: rows[1] });
		expect(got[1]).toBeNull();
		expect(got[2]?.meta).toBe("meta 2999");
		expect(d.rounds).toBeLessThanOrEqual(2);
		expect(seg.heldBytes()).toBeGreaterThan(opened);
		seg.free();
		expect(() => new ThreadError("x")).not.toThrow();
	});
});
