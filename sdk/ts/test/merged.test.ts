/**
 * commitMergedSegment: a segment merged outside the writer (here with the same wasm merge a native process runs
 * through thread-core) replaces the run it read. Newer replacements stay hidden, the sets the merge applied drop,
 * a run that changed is not committed, and an object that does not open is refused.
 */
import { describe, expect, it } from "vitest";
import { buildTombState, createLexicalGeneration, MemoryBlobStore, resolveLexConfig, tombForSegment, type BlobStore, type ThreadIndex } from "../src/index.js";
import { thread, w } from "./helpers.js";

/** What a native merger does with a plan: read the run whole, apply the plan's sets, merge, write. */
async function mergeOutside(store: BlobStore, ix: ThreadIndex, run: readonly string[], name: string) {
	const manifest = ix.manifest;
	const lex = resolveLexConfig(manifest)!;
	const segs = (manifest.segments ?? []).filter((s) => run.includes(s.key));
	const parts = await Promise.all(segs.map(async (s) => (await store.get(s.key))!));
	const sets = await Promise.all((manifest.tombstones ?? []).map(async (t) => ({ rev: t.rev, ids: new Uint32Array((await store.get(t.key))!.slice().buffer), ...(t.scope ? { scope: t.scope } : {}) })));
	const tomb = sets.length ? buildTombState(sets) : null;
	const tombs = segs.map((s) => (tomb ? tombForSegment(tomb, s.addedRev) : new Uint8Array(0)));
	const cat = (xs: Uint8Array[]) => {
		const out = new Uint8Array(xs.reduce((n, x) => n + x.byteLength, 0));
		let o = 0;
		for (const x of xs) (out.set(x, o), (o += x.byteLength));
		return out;
	};
	const out = w.merge_segments(cat(parts), Float64Array.from(parts, (p) => p.byteLength), cat(tombs), Uint32Array.from(tombs, (t) => t.byteLength), JSON.stringify(lex.tokenizer ?? {}), lex.k1, lex.b, lex.lengths, lex.bucketTerms, lex.docs, lex.text, lex.bucketDocs, false, lex.bucketKeys);
	const meta = JSON.parse(w.segment_meta(out)) as { lex?: { num_docs: number }; docs?: { n: number }; keys?: { n: number } };
	const key = `idx/${manifest.gen}/seg/${name}.thread`;
	await store.put!(key, out);
	return { run, readRev: manifest.rev ?? ix.rev, key, bytes: out.byteLength, count: meta.docs?.n ?? meta.lex?.num_docs ?? 0, keys: meta.keys !== undefined };
}

describe("commitMergedSegment", () => {
	it("replaces the run, keeps newer replacements hidden and drops the sets it applied", async () => {
		const store = new MemoryBlobStore();
		await createLexicalGeneration({ store, gen: "m", lex: { docs: true, text: true } });
		const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
		await ix.upsert({ keys: ["a", "b"], texts: ["apple", "banana"], metas: ["a1", "b1"] });
		await ix.upsert({ keys: ["a"], texts: ["apricot"], metas: ["a2"] }); // tombstones the first a
		await ix.upsert({ keys: ["c"], texts: ["cherry"], metas: ["c1"] });
		expect(ix.segments.length).toBe(3);
		expect((ix.manifest.tombstones ?? []).map((t) => t.rev)).toEqual([1, 2, 3]); // each keyed upsert writes a set
		const job = await mergeOutside(store, ix, ix.segments.map((s) => s.key), "merged.3");
		// While the merge ran: a replaces b, which the merge has not seen.
		await ix.upsert({ keys: ["b"], texts: ["blueberry"], metas: ["b2"] });
		const r = await ix.commitMergedSegment(job);
		expect(r.merged).toBe(3);
		expect(ix.segments.map((s) => s.key)).toEqual([job.key, ix.segments[1].key]);
		expect(ix.segments[0].addedRev).toBe(job.readRev);
		// The sets up to rev 3 were applied by the merge and dropped; b's set (rev 4) still hides the old b.
		expect((ix.manifest.tombstones ?? []).map((t) => t.rev)).toEqual([4]);
		ix.close();
		const cold = await thread.open({ store, cache: false, refreshInterval: 0 });
		const got = (await cold.get(["a", "b", "c"], { text: true })).records;
		expect([got.get("a")?.text, got.get("b")?.text, got.get("c")?.text]).toEqual(["apricot", "blueberry", "cherry"]);
		expect((await cold.search({ text: "banana" })).hits).toEqual([]);
		expect((await cold.search({ text: "apple" })).hits).toEqual([]);
		expect((await cold.search({ text: "apricot", docs: true })).hits.map((h) => h.doc?.meta)).toEqual(["a2"]);
		cold.close();
	});

	it("does not commit a run that changed, and refuses an object that does not open", async () => {
		const store = new MemoryBlobStore();
		await createLexicalGeneration({ store, gen: "s", lex: { docs: true, text: true } });
		const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
		for (const k of ["a", "b", "c"]) await ix.upsert({ keys: [k], texts: [k + " text"] });
		const job = await mergeOutside(store, ix, ix.segments.map((s) => s.key), "merged.stale");
		await ix.compactSegments();
		const head = ix.rev;
		const stale = await ix.commitMergedSegment(job);
		expect(stale.merged).toBe(0);
		expect(stale.rev).toBe(head);

		const fresh = await mergeOutside(store, ix, ix.segments.map((s) => s.key), "merged.bad");
		const bytes = (await store.get(fresh.key))!;
		await store.put!(fresh.key, bytes.slice(0, 16));
		await expect(ix.commitMergedSegment(fresh)).rejects.toThrow(/bytes/);
		await expect(ix.commitMergedSegment({ ...fresh, bytes: 16 })).rejects.toThrow();
		await expect(ix.commitMergedSegment({ ...fresh, key: "idx/other/seg/x.thread" })).rejects.toThrow();
		expect(ix.rev).toBe(head);
		ix.close();
	});
});
