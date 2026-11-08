/**
 * Trace envelopes (docs/observability.md): every operation returns one, its spans, io, counters and kernels agree
 * with the operation's own stats, `traceEvents` records each read, and `onTrace` receives them (sampled, once per
 * operation, never for the nested lists of a search).
 */
import { describe, expect, it } from "vitest";
import { MemoryBlobStore, TRACE_VERSION, type TraceEnvelope, type TraceSpan } from "../src/index.js";
import { buildGen, rng, thread, unitRows } from "./helpers.js";

const names = (spans: TraceSpan[]) => spans.map((s) => s.name);

/** The invariants every envelope keeps. */
function wellFormed(t: TraceEnvelope): void {
	expect(t.v).toBe(TRACE_VERSION);
	expect(t.sdk).toBe("ts");
	expect(t.clock).toBe("real");
	expect(t.ms).toBeGreaterThanOrEqual(0);
	expect(t.at).toBeGreaterThan(1.7e12);
	for (const v of Object.values(t.counters)) expect(Number.isFinite(v)).toBe(true);
	for (const k of Object.values(t.kernels)) {
		expect(k.calls).toBeGreaterThan(0);
		expect(k.ms).toBeGreaterThanOrEqual(0);
	}
	const walk = (spans: TraceSpan[]) => {
		for (const s of spans) {
			expect(s.ms).toBeGreaterThanOrEqual(0);
			if (s.start !== undefined) expect(s.start).toBeGreaterThanOrEqual(0);
			if (s.children) walk(s.children);
		}
	};
	walk(t.spans);
	// JSON round trip: an envelope is plain data.
	expect(JSON.parse(JSON.stringify(t))).toEqual(t);
}

describe("trace envelopes", () => {
	it("a query's envelope matches its stats", async () => {
		const { store, vectors, dim } = await buildGen("t1", 3, { n: 600, k2: 8 });
		const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
		const r = await ix.query(vectors.subarray(5 * dim, 6 * dim), { topK: 5, nprobe: 4, rerank: 30 });
		const t = r.stats.trace;
		wellFormed(t);
		expect(t.op).toBe("query");
		expect(names(t.spans)).toEqual(["route", "phase1", "phase2", "payload"]);
		expect(t.attrs).toMatchObject({ gen: "t1", rev: 0, format: 3, routing: "flat", topK: 5, nprobe: 4, rerank: 30, filtered: false });
		expect(t.counters.scanned).toBe(r.stats.scanned);
		expect(t.counters.cells).toBe(r.stats.cells.length);
		expect(t.counters.hits).toBe(r.hits.length);
		expect(t.counters.rerankCandidates).toBe(r.stats.rerankCandidates);
		expect(t.counters.pushed).toBe(r.stats.pushed);
		// The heap took at least the candidates it kept, and at most every entry scanned.
		expect(r.stats.pushed).toBeGreaterThanOrEqual(r.stats.candidates);
		expect(r.stats.pushed).toBeLessThanOrEqual(r.stats.scanned);
		expect(t.io.bytes).toBe(r.stats.io.bytes);
		expect(t.io.reads).toBe(r.stats.io.reads);
		// The phases' io adds up to the query's, and the phases follow each other.
		expect(t.spans.reduce((a, s) => a + s.io!.bytes, 0)).toBe(t.io.bytes);
		for (let i = 1; i < t.spans.length; i++) expect(t.spans[i].start!).toBeGreaterThanOrEqual(t.spans[i - 1].start!);
		// One scan kernel call per probed slab, and the CPU kernels of a PQ query.
		expect(t.kernels.scan.calls).toBe(r.stats.cells.length);
		expect(Object.keys(t.kernels).sort()).toEqual(["merge", "rerank", "route", "scan"]);
		expect(t.events).toBeUndefined();
		ix.close();
	});

	it("traceEvents records every read of a query", async () => {
		const { store, vectors, dim } = await buildGen("t2", 3, { n: 600, k2: 8 });
		const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
		const r = await ix.query(vectors.subarray(0, dim), { topK: 5, nprobe: 4, rerank: 30, traceEvents: true, returnPayload: true });
		const ev = r.stats.trace.events!;
		expect(ev.filter((e) => e.kind === "prefix").length).toBe(r.stats.cells.length);
		expect(ev.reduce((a, e) => a + e.bytes, 0)).toBe(r.stats.trace.io.bytes);
		expect(ev.every((e) => e.key.startsWith("idx/t2/") && e.ranges >= 1 && e.start >= 0)).toBe(true);
		expect(ev.map((e) => e.start)).toEqual([...ev.map((e) => e.start)].sort((a, b) => a - b));
		// Same answer with or without events.
		const plain = await ix.query(vectors.subarray(0, dim), { topK: 5, nprobe: 4, rerank: 30, returnPayload: true });
		expect(plain.hits.map((h) => h.id)).toEqual(r.hits.map((h) => h.id));
		ix.close();
	});

	it("a hybrid search nests the dense and lexical lists; get, listKeys and writes have envelopes", async () => {
		const store = new MemoryBlobStore();
		const dim = 16;
		await thread.createGeneration({ store, gen: "h", dim, compaction: null });
		const seen: TraceEnvelope[] = [];
		const ix = await thread.open({ store, cache: false, refreshInterval: 0, onTrace: (t) => seen.push(t) });
		expect(seen.map((t) => t.op)).toEqual(["open"]);
		wellFormed(ix.openTrace);
		expect(names(ix.openTrace.spans)).toEqual(["current", "prefetch", "manifest", "slabs", "base"]);
		const r = rng(3);
		const vecs = unitRows(40, dim, r);
		const texts = Array.from({ length: 40 }, (_, i) => (i % 3 === 0 ? `deploy failed on host ${i}` : `budget review ${i}`));
		const up = await ix.upsert({ keys: texts.map((_, i) => `doc/${i}`), vectors: Array.from({ length: 40 }, (_, i) => vecs.subarray(i * dim, (i + 1) * dim)), texts, metas: texts });
		wellFormed(up.trace);
		expect(up.trace.op).toBe("upsert");
		expect(names(up.trace.spans)).toEqual(["commit", "useRevision"]);
		expect(up.trace.counters.count).toBe(40);

		seen.length = 0;
		const s = await ix.search({ text: "deploy failed", vector: vecs.subarray(0, dim), topK: 5, docs: true });
		const t = s.stats.trace;
		wellFormed(t);
		// One envelope for the search: its dense and lexical lists are spans, not envelopes of their own.
		expect(seen.map((x) => x.op)).toEqual(["search"]);
		expect(t.op).toBe("search");
		expect(t.attrs.mode).toBe("hybrid");
		expect(names(t.spans).sort()).toEqual(["dense", "docs", "fuse", "lexical"]);
		const dense = t.spans.find((x) => x.name === "dense")!;
		expect(names(dense.children!)).toEqual(["route", "phase1", "phase2", "payload"]);
		expect(names(t.spans.find((x) => x.name === "lexical")!.children!)).toEqual(["lookup", "score"]);
		expect(t.counters["dense.scanned"]).toBe(s.dense!.stats.scanned);
		expect(t.counters["lexical.hits"]).toBe(s.lexical!.hits.length);
		expect(t.kernels.bm25Score.calls).toBeGreaterThan(0);
		expect(t.kernels.deltaScan.calls).toBe(1);
		expect(t.io.bytes).toBe(t.spans.reduce((a, x) => a + (x.io?.bytes ?? 0), 0));

		const g = await ix.get(["doc/0", "doc/1", "nope"]);
		wellFormed(g.trace);
		expect(g.trace.counters).toMatchObject({ keys: 3, found: 2 });
		expect(names(g.trace.spans)).toEqual(["keys", "docs"]);
		const page = await ix.listKeys({ prefix: "doc/", limit: 10 });
		expect(page.trace.counters).toMatchObject({ entries: 10, more: 1 });
		const del = await ix.delete([0, 1]);
		expect(del.trace.op).toBe("delete");
		expect(seen.map((x) => x.op)).toEqual(["search", "get", "listKeys", "delete"]);
		ix.close();
	});

	it("traceSample 0 builds envelopes but hands none to the sink; a throwing sink fails nothing", async () => {
		const { store, vectors, dim } = await buildGen("t3", 2, { n: 200 });
		let calls = 0;
		const ix = await thread.open({ store, cache: false, refreshInterval: 0, onTrace: () => calls++, traceSample: 0 });
		const r = await ix.query(vectors.subarray(0, dim), { topK: 3 });
		expect(r.stats.trace.op).toBe("query");
		expect(calls).toBe(0);
		ix.close();
		const bad = await thread.open({
			store,
			cache: false,
			refreshInterval: 0,
			onTrace: () => {
				throw new Error("sink down");
			},
		});
		expect((await bad.query(vectors.subarray(0, dim), { topK: 3 })).hits.length).toBe(3);
		bad.close();
	});
});
