/**
 * Bulk build (`thread.bulkBuild`): staged documents with vectors become a dense generation with k cells from
 * k-means, in bounded steps that resume from `build/{gen}/state.json`, every step within `maxBytes`.
 */
import { describe, expect, it } from "vitest";
import { bulkBuildPlan, MemoryBlobStore, parseSlabsTable, type BulkBuild, type BulkStepResult } from "../src/index.js";
import { rng, thread, unitRows } from "./helpers.js";

const MiB = 1024 * 1024;

async function run(b: BulkBuild, maxBytes: number, deadlineMs: number, maxSteps = 500): Promise<BulkStepResult[]> {
	const steps: BulkStepResult[] = [];
	for (let i = 0; i < maxSteps; i++) {
		const r = await b.step({ deadlineMs });
		expect(r.estimatedPeakBytes).toBeLessThanOrEqual(maxBytes);
		steps.push(r);
		if (r.done) return steps;
	}
	throw new Error(`not done after ${maxSteps} steps: ${JSON.stringify(steps[steps.length - 1])}`);
}

describe("bulk build of a dense generation", () => {
	it(
		"builds 20,000 vectors of dim 1024 within 32 MiB per step, resuming from its state",
		async () => {
			const dim = 1024;
			const n = 20_000;
			const maxBytes = 32 * MiB;
			const store = new MemoryBlobStore();
			const r = rng(11);
			const centers = unitRows(64, dim, r);
			const wasm0 = thread.memoryBytes();
			const t0 = Date.now();
			const b = await thread.bulkBuild({ store, gen: "e2", dim, lex: { text: true }, maxBytes, owner: "do-1" });
			const sample: { key: string; v: Float32Array }[] = [];
			for (let a = 0; a < n; a += 200) {
				const v = unitRows(200, dim, r, centers, 0.05);
				const keys = Array.from({ length: 200 }, (_, i) => `doc/${a + i}`);
				// Every 50th document has no vector: staged and loaded text-only.
				const vectors = keys.map((_, i) => ((a + i) % 50 === 49 ? null : v.subarray(i * dim, (i + 1) * dim)));
				if (a % 2000 === 0) sample.push({ key: keys[3], v: v.slice(3 * dim, 4 * dim) });
				await b.add({ keys, metas: keys.map((k) => `meta ${k}`), texts: keys.map((_, i) => `document number${a + i} about topic${(a + i) % 37}`), vectors });
			}
			const withVectors = n - n / 50;
			expect(b.state.vectors).toBe(withVectors);
			await b.seal();
			const plan = bulkBuildPlan(withVectors, dim, { maxBytes });
			expect(b.state.k).toBe(plan.k);
			expect(plan.k).toBeGreaterThan(20);
			// A few steps, then a new handle on the same state (a Durable Object restarted between alarms).
			const first = [];
			for (let i = 0; i < 3; i++) first.push(await b.step({ deadlineMs: 300 }));
			expect(first[0].worked).toContain("sample");
			const b2 = await thread.bulkBuild({ store, gen: "e2", dim, maxBytes, owner: "do-1" });
			expect(b2.state.phase).toBe(b.state.phase);
			const steps = await run(b2, maxBytes, 5_000);
			const all = [...first, ...steps];
			const buildMs = Date.now() - t0;
			// The wasm memory (which never shrinks) grew by less than the budget over staging and every step.
			expect(thread.memoryBytes() - wasm0).toBeLessThanOrEqual(maxBytes);
			for (const p of ["sample", "kmeans", "create", "load", "compact"]) expect(all.some((s) => s.worked.includes(p as never))).toBe(true);
			const ix = await thread.open({ store, gen: "e2", cache: false, refreshInterval: 0 });
			const m = ix.manifest;
			expect(m.deltas).toEqual([]);
			expect(m.k2).toBeGreaterThanOrEqual(plan.k);
			const table = parseSlabsTable((await store.get(m.files!.slabs!))!, m.k1 * m.k2, "slabs");
			expect(table.reduce((acc, row) => acc + row[2], 0)).toBe(withVectors);
			// Cells from k-means: none empty-handed past the budget, none past what it splits.
			const sizes = table.map((row) => row[2]);
			expect(Math.max(...sizes)).toBeLessThanOrEqual(plan.limits.splitEntries);
			console.log(`bulk build: ${JSON.stringify({ k: plan.k, k2: m.k2, cells: { min: Math.min(...sizes), max: Math.max(...sizes) }, steps: all.length, stepMs: all.map((x) => x.ms), peakMB: +(Math.max(...all.map((x) => x.estimatedPeakBytes)) / MiB).toFixed(1), wasmGrowthMB: +((thread.memoryBytes() - wasm0) / MiB).toFixed(1), buildMs })}`);
			const got = await ix.get(sample.map((s) => s.key));
			for (const s of sample) {
				const id = got.records.get(s.key)!.id;
				expect((await ix.query(s.v, { topK: 1, nprobe: 8 })).hits[0]?.id).toBe(id);
			}
			const text = await ix.get(["doc/49"], { text: true });
			expect(text.records.get("doc/49")?.text).toBe("document number49 about topic12");
			expect((await ix.lexicalSearch("number12345", 1)).hits[0]?.id).toBe((await ix.get(["doc/12345"])).records.get("doc/12345")!.id);
			ix.close();
			expect(await b2.cleanup()).toBeGreaterThan(200);
			expect(await store.list("build/e2/")).toEqual([]);
		},
		300_000,
	);

	it("replays a stage again after a crash without duplicating documents", async () => {
		const dim = 16;
		const maxBytes = 4 * MiB;
		const store = new MemoryBlobStore();
		const r = rng(5);
		const centers = unitRows(8, dim, r);
		const b = await thread.bulkBuild({ store, gen: "s1", dim, lex: {}, maxBytes, cellEntries: 100 });
		for (let a = 0; a < 2000; a += 100) await b.add({ keys: Array.from({ length: 100 }, (_, i) => `k${a + i}`), metas: Array.from({ length: 100 }, () => "m"), vectors: unitRows(100, dim, r, centers, 0.3) });
		await b.seal();
		await run(b, maxBytes, 5_000);
		// The state as if the last three replays committed their upserts but not their progress (crashes), and the
		// final compaction had not run: a new handle replays them and compacts again.
		const st = { ...b.state, phase: "load", load: { pos: b.state.staged - 3 } };
		await store.put("build/s1/state.json", JSON.stringify(st));
		const again = await thread.bulkBuild({ store, gen: "s1", dim, maxBytes });
		expect(again.state.phase).toBe("load");
		const steps = await run(again, maxBytes, 5_000);
		expect(steps.some((x) => x.worked.includes("load"))).toBe(true);
		const ix = await thread.open({ store, gen: "s1", cache: false, refreshInterval: 0 });
		const table = parseSlabsTable((await store.get(ix.manifest.files!.slabs!))!, ix.manifest.k1 * ix.manifest.k2, "slabs");
		expect(table.reduce((acc, row) => acc + row[2], 0)).toBe(2000);
		expect((await ix.get(["k0", "k1999"])).records.size).toBe(2);
		ix.close();
	});

	it("refuses what it could not finish within the budget, before staging it", async () => {
		const dim = 1024;
		const store = new MemoryBlobStore();
		const b = await thread.bulkBuild({ store, gen: "x1", dim, maxBytes: 32 * MiB });
		const r = rng(3);
		// More vectors in one add than a 32 MiB compaction always folds (about 800 at dim 1024).
		await expect(b.add({ ids: Array.from({ length: 1000 }, (_, i) => i), vectors: unitRows(1000, dim, r) })).rejects.toMatchObject({ status: 413 });
		// A meta that is not a string is refused when it is added, not when it is replayed after seal.
		await expect(b.add({ keys: ["m"], metas: [{ path: "m" } as never], vectors: unitRows(1, dim, r) })).rejects.toMatchObject({ status: 400 });
		expect(b.state.staged).toBe(0);
		await b.add({ ids: [1, 2], vectors: unitRows(2, dim, r) });
		await b.seal();
		await expect(b.add({ ids: [3], vectors: unitRows(1, dim, r) })).rejects.toMatchObject({ status: 409 });
		// A generation that exists already is not built over.
		await thread.createGeneration({ store, gen: "x2", dim, setCurrent: false });
		await expect(thread.bulkBuild({ store, gen: "x2", dim })).rejects.toMatchObject({ status: 409 });
		// More cells than one isolate can cluster at this budget: the Worker's build slots.
		expect(() => bulkBuildPlan(5_000_000, dim, { maxBytes: 32 * MiB })).not.toThrow();
		const p = bulkBuildPlan(5_000_000, dim, { maxBytes: 32 * MiB });
		expect(p.k).toBeGreaterThan(p.maxCells);
	});
});
