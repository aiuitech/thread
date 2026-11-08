/**
 * The bulk copy that left inseam's e1 generation blocked (2026-11-04): a generation from nothing (one cell, profile
 * `edge`, cells split at 3000 entries), dim 1024, filled by upserts of 200 vectors, with `compact({ maxBytes: 32 MiB,
 * deadlineMs: 10 s })` after each one once it is due. Before 0.2.3 the planner folded the cell up to just under
 * 3000 entries, which a 32 MiB call cannot split (about 46 MiB), and every later call was `blocked`. Now cells split
 * at what the budget can split (`compactionLimits`), so k2 grows, the deltas stay few, and no call's estimated peak
 * passes the budget. Its own file: the wasm memory growth it measures is this worker's alone.
 */
import { describe, expect, it } from "vitest";
import { compactionLimits, MemoryBlobStore, parseSlabsTable, type CompactResult, type Manifest } from "../src/index.js";
import { rng, thread, unitRows } from "./helpers.js";

const DIM = 1024;
const N = 20_000;
const BATCH = 200;
const MAX_BYTES = 32 * 1024 * 1024;

/** Drops slab revisions and delta segments the head no longer lists (a MemoryBlobStore would hold every revision). */
async function prune(store: MemoryBlobStore, m: Manifest): Promise<void> {
	const table = parseSlabsTable((await store.get(m.files!.slabs!))!, m.k1 * m.k2, "slabs");
	const live = new Set(table.map((row, cell) => `idx/${m.gen}/slab/${String(cell).padStart(5, "0")}${row.length > 4 && row[4] ? `.${row[4]}` : ""}`));
	const deltas = new Set((m.deltas ?? []).map((d) => d.key));
	for (const key of [...store.objects.keys()]) {
		if (key.includes("/slab/") && !live.has(key)) store.objects.delete(key);
		if (key.includes("/delta/") && !deltas.has(key)) store.objects.delete(key);
	}
}

describe("bulk copy into a generation from nothing, dim 1024, 32 MiB compactions", () => {
	it(
		"splits cells as they grow: k2 grows, deltas stay bounded, every call within maxBytes",
		async () => {
			const store = new MemoryBlobStore();
			await thread.createGeneration({ store, gen: "e1", dim: DIM, lex: null, profile: "edge" });
			const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
			const r = rng(2026);
			const centers = unitRows(64, DIM, r);
			const wasm0 = thread.memoryBytes();
			const limits = compactionLimits(ix.manifest, { maxBytes: MAX_BYTES });
			// At dim 1024 a 32 MiB call splits cells of up to ~1900 entries: below the profile's 3000.
			expect(limits.splitEntries).toBeLessThan(3000);
			expect(limits.cellEntries).toBe(Math.floor(limits.splitEntries / 2));
			expect(limits.upsertVectors).toBeGreaterThanOrEqual(BATCH);
			const outcomes: Record<string, number> = {};
			let maxDeltas = 0;
			let maxPeak = 0;
			let maxCell = 0;
			const ks: number[] = [];
			const sample: { id: number; v: Float32Array }[] = [];
			for (let a = 0; a < N; a += BATCH) {
				const v = unitRows(BATCH, DIM, r, centers, 0.04);
				if (a % 2000 === 0) sample.push({ id: a + 7, v: v.slice(7 * DIM, 8 * DIM) });
				await ix.upsert({ ids: Array.from({ length: BATCH }, (_, i) => a + i), vectors: v });
				maxDeltas = Math.max(maxDeltas, ix.manifest.deltas?.length ?? 0);
				if (!ix.compactionDue().due) continue;
				const c: CompactResult = await ix.compact({ maxBytes: MAX_BYTES, deadlineMs: 10_000, owner: "bulk" });
				outcomes[c.outcome] = (outcomes[c.outcome] ?? 0) + 1;
				expect(["committed", "partial"]).toContain(c.outcome);
				expect(c.estimatedPeakBytes).toBeLessThanOrEqual(MAX_BYTES);
				maxPeak = Math.max(maxPeak, c.estimatedPeakBytes);
				const m = ix.manifest;
				const table = parseSlabsTable((await store.get(m.files!.slabs!))!, m.k1 * m.k2, "slabs");
				maxCell = Math.max(maxCell, ...table.map((row) => row[2]));
				ks.push(m.k2);
				await prune(store, m);
			}
			// k2 grew with the data, and kept growing to the end (no stall at one cell).
			expect(ks[ks.length - 1]).toBeGreaterThanOrEqual(Math.floor(N / limits.splitEntries));
			expect(ks[Math.floor(ks.length / 2)]).toBeLessThan(ks[ks.length - 1]);
			// The backlog stays near the trigger (16): never the 47+ of the incident.
			expect(maxDeltas).toBeLessThanOrEqual(24);
			// No cell ever passed what a call with this budget can split on its own.
			expect(maxCell).toBeLessThanOrEqual(limits.splitEntries);
			// The wasm memory (which never shrinks) grew by less than the budget over all the calls and upserts.
			expect(thread.memoryBytes() - wasm0).toBeLessThanOrEqual(MAX_BYTES);
			// Fold the rest, then every vector is there and the sampled ones are found.
			for (let c = await ix.compact({ force: true, maxBytes: MAX_BYTES }); c.more; c = await ix.compact({ force: true, maxBytes: MAX_BYTES })) expect(c.estimatedPeakBytes).toBeLessThanOrEqual(MAX_BYTES);
			const m = ix.manifest;
			expect(m.deltas).toEqual([]);
			const table = parseSlabsTable((await store.get(m.files!.slabs!))!, m.k1 * m.k2, "slabs");
			expect(table.reduce((acc, row) => acc + row[2], 0)).toBe(N);
			for (const s of sample) expect((await ix.query(s.v, { topK: 1, nprobe: 1 << 20 })).hits[0]?.id).toBe(s.id);
			console.log(`bulk copy: ${JSON.stringify({ outcomes, k: m.k1 * m.k2, maxDeltas, maxCell, limits, maxPeakMB: +(maxPeak / 1048576).toFixed(1), wasmGrowthMB: +((thread.memoryBytes() - wasm0) / 1048576).toFixed(1) })}`);
			ix.close();
		},
		600_000,
	);
});
