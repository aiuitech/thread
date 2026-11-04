/**
 * Dense compaction in the SDK (`ix.compact` / `compactDeltas`): a generation created from nothing grows cells
 * through folds and splits; results, deletes and replaces survive; the memory, cell and time budgets bound a call
 * (more calls finish the work); the lock, a rerun after a crash and tombstone dropping follow the Worker's rules.
 */
import { describe, expect, it } from "vitest";
import { compactionStatus, keys, MemoryBlobStore, parseSlabsTable, type BlobStore, type Manifest } from "../src/index.js";
import { rng, thread, unitRows } from "./helpers.js";

const DIM = 16;

/** A generation from nothing with a policy that triggers early and splits small cells. */
async function fresh(store: BlobStore, gen = "d1", cap = 60) {
	await thread.createGeneration({ store, gen, dim: DIM, lex: { text: true }, compaction: { maxDeltas: 2, maxCellEntries: cap } });
	return thread.open({ store, cache: false, refreshInterval: 0 });
}

/** Clustered unit vectors (8 clusters), so 2-means splits have structure to find. */
function data(n: number, seed = 7): Float32Array {
	const r = rng(seed);
	const centers = unitRows(8, DIM, r);
	return unitRows(n, DIM, r, centers, 0.3);
}

const row = (v: Float32Array, i: number) => v.subarray(i * DIM, (i + 1) * DIM);

async function upsertAll(ix: Awaited<ReturnType<typeof fresh>>, v: Float32Array, n: number, batch: number, first = 0) {
	for (let a = 0; a < n; a += batch) {
		const e = Math.min(n, a + batch);
		await ix.upsert({ ids: Array.from({ length: e - a }, (_, i) => first + a + i), vectors: v.subarray(a * DIM, e * DIM) });
	}
}

/** Top-1 id of each row, probing every cell. */
async function selfTop(ix: Awaited<ReturnType<typeof fresh>>, v: Float32Array, ids: number[]) {
	const out: number[] = [];
	for (const i of ids) out.push((await ix.query(row(v, i), { topK: 1, nprobe: 1 << 20 })).hits[0]?.id ?? -1);
	return out;
}

async function head(store: BlobStore, gen: string): Promise<Manifest> {
	const ix = await thread.open({ store, gen, cache: false, refreshInterval: 0 });
	const m = ix.manifest;
	ix.close();
	return m;
}

describe("dense compaction in the SDK", () => {
	it("folds deltas into cells, splits them as they grow, keeps every result", async () => {
		const store = new MemoryBlobStore();
		const ix = await fresh(store);
		const n = 300;
		const v = data(n);
		await upsertAll(ix, v, n, 50);
		expect(ix.compactionDue().due).toBe(true);
		const ids = Array.from({ length: n }, (_, i) => i);
		const before = await selfTop(ix, v, ids);
		let calls = 0;
		let k = 1;
		for (;;) {
			const r = await ix.compact({ force: true });
			calls++;
			expect(["committed", "noop"]).toContain(r.outcome);
			expect(r.estimatedPeakBytes).toBeLessThanOrEqual(48 * 1024 * 1024);
			k = r.k;
			if (!r.more) break;
			expect(calls).toBeLessThan(20);
		}
		expect(k).toBeGreaterThan(1);
		expect(ix.manifest.deltas).toEqual([]);
		// Every upsert wrote a replace set (replace defaults to true): all dropped now.
		expect(ix.manifest.tombstones).toEqual([]);
		const table = parseSlabsTable((await store.get(ix.manifest.files!.slabs!))!, k, "slabs");
		expect(table.reduce((a, r) => a + r[2], 0)).toBe(n);
		expect(await selfTop(ix, v, ids)).toEqual(before);
		expect(before).toEqual(ids);
		// A cold reader of the head (fine.{rev}.f32, k2, the slabs table) answers the same.
		const cold = await thread.open({ store, cache: false });
		expect(await selfTop(cold, v, ids.slice(0, 40))).toEqual(ids.slice(0, 40));
		cold.close();
		// Nothing left: a further call is a noop, with or without force.
		expect((await ix.compact()).outcome).toBe("noop");
		expect((await ix.compact({ force: true })).outcome).toBe("noop");
		ix.close();
	});

	it("keeps deletes and replaces through a fold, and drops the sets once nothing needs them", async () => {
		const store = new MemoryBlobStore();
		await thread.createGeneration({ store, gen: "d2", dim: DIM, lex: null, compaction: { maxDeltas: 1, maxCellEntries: 0 } });
		const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
		const v = data(120, 3);
		await upsertAll(ix, v, 100, 25);
		await ix.compact({ force: true });
		await ix.delete([3, 4]);
		const w2 = data(1, 99);
		await ix.upsert({ ids: [5], vectors: w2 }); // replace id 5 with another vector
		await ix.upsert({ ids: [3], vectors: row(v, 3) }); // re-insert a deleted id
		expect(compactionStatus(ix.manifest).reasons.length).toBeGreaterThan(0);
		const r = await ix.compact();
		expect(r.outcome).toBe("committed");
		expect(r.folded).toBe(2);
		const hit = async (q: Float32Array) => (await ix.query(q, { topK: 1, nprobe: 64 })).hits[0];
		expect((await hit(row(v, 4))).id).not.toBe(4);
		expect((await hit(row(v, 3))).id).toBe(3);
		expect((await hit(w2)).id).toBe(5);
		expect((await hit(row(v, 5))).id).not.toBe(5);
		expect(ix.manifest.tombstones).toEqual([]);
		expect(r.droppedTombstoneSets.length).toBeGreaterThan(0);
		ix.close();
	});

	it("bounds a call by memory and cells, and finishes in several calls", async () => {
		const store = new MemoryBlobStore();
		// Cells split at 40 entries, so they stay small; the budget holds about four 20-vector deltas and their cell.
		const ix = await fresh(store, "d3", 40);
		const n = 200;
		const v = data(n, 11);
		await upsertAll(ix, v, n, 20); // 10 deltas
		const maxBytes = 40_000;
		const outcomes: string[] = [];
		for (let i = 0; i < 20; i++) {
			const r = await ix.compact({ force: true, maxBytes });
			outcomes.push(r.outcome);
			expect(r.estimatedPeakBytes).toBeLessThanOrEqual(maxBytes);
			if (!r.more) break;
		}
		expect(outcomes.length).toBeGreaterThan(2);
		expect(ix.manifest.deltas).toEqual([]);
		expect(await selfTop(ix, v, [0, 50, 199])).toEqual([0, 50, 199]);
		// A budget too small for the first delta blocks (and says why) instead of passing it.
		await upsertAll(ix, data(30, 12), 30, 30, 1000);
		const b = await ix.compact({ force: true, maxBytes: 1000 });
		expect(b.outcome).toBe("blocked");
		expect(b.more).toBe(false);
		expect(b.note).toMatch(/does not fit/);
		ix.close();
	});

	it("commits the finished part when the deadline cuts a slice", async () => {
		const store = new MemoryBlobStore();
		await thread.createGeneration({ store, gen: "d4", dim: DIM, lex: null, compaction: { maxDeltas: 1, maxCellEntries: 40 } });
		const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
		const v = data(400, 5);
		await upsertAll(ix, v, 400, 400);
		while ((await ix.compact({ force: true })).more);
		const k = ix.manifest.k1 * ix.manifest.k2;
		expect(k).toBeGreaterThan(4);
		// Small deltas touching one or two cells each, then a deadline that only lets the first cells start.
		const extra = data(6, 77);
		for (let i = 0; i < 6; i++) await ix.upsert({ ids: [2000 + i], vectors: row(extra, i) });
		const slow = new SlowStore(store, 80);
		const ix2 = await thread.open({ store: slow, cache: false, refreshInterval: 0 });
		const r = await ix2.compact({ force: true, deadlineMs: 2_600, concurrency: 1, maxCellEntries: 0, purge: false });
		expect(["partial", "deadline", "committed"]).toContain(r.outcome);
		if (r.outcome === "partial") {
			expect(r.folded).toBeGreaterThan(0);
			expect(r.remainingDeltas).toBeGreaterThan(0);
			expect(r.more).toBe(true);
		}
		while ((await ix2.compact({ force: true, maxCellEntries: 0 })).more);
		expect(ix2.manifest.deltas).toEqual([]);
		expect(await selfTop(ix2, extra, [0, 1, 2, 3, 4, 5])).toEqual([2000, 2001, 2002, 2003, 2004, 2005]);
		expect(await selfTop(ix2, v, [0, 60, 120])).toEqual([0, 60, 120]);
		ix2.close();
		ix.close();
	});

	it("takes compact.lock: a fresh foreign lock blocks, the owner's own lock is taken over, a stale one too", async () => {
		const store = new MemoryBlobStore();
		const ix = await fresh(store, "d5");
		await upsertAll(ix, data(60), 60, 20);
		const lockKey = "idx/d5/compact.lock";
		await store.put(lockKey, JSON.stringify({ owner: "other", ms: Date.now(), at: new Date().toISOString() }));
		const r1 = await ix.compact({ owner: "me" });
		expect(r1.outcome).toBe("locked");
		expect(r1.more).toBe(true);
		await store.put(lockKey, JSON.stringify({ owner: "me", ms: Date.now(), at: new Date().toISOString() }));
		const r2 = await ix.compact({ owner: "me" });
		expect(r2.outcome).toBe("committed");
		expect(r2.staleLockTaken).toBe(true);
		expect(await store.head(lockKey)).toBeNull(); // released
		await upsertAll(ix, data(60, 8), 60, 20, 100);
		await store.put(lockKey, JSON.stringify({ owner: "other", ms: Date.now() - 11 * 60_000, at: "" }));
		expect((await ix.compact()).outcome).toBe("committed");
		// The run is recorded where the Worker's /index/status reads it.
		const last = JSON.parse(new TextDecoder().decode((await store.get("idx/d5/compact.last.json"))!));
		expect(last.last.outcome).toBe("committed");
		expect(last.recent.length).toBe(2);
		ix.close();
	});

	it("overwrites a crashed run's slab only while no revision can list it", async () => {
		const store = new MemoryBlobStore();
		const ix = await fresh(store, "d7", 50);
		const v = data(150, 31);
		await upsertAll(ix, v, 150, 50);
		const base = ix.manifest.rev!;
		// The crashed run did not split (cap 0): its slab of cell 0 differs from the one a splitting rerun builds.
		const crashing = new CrashStore(store, (key) => /manifest\.\d+\.json$/.test(key));
		const ixc = await thread.open({ store: crashing, cache: false, refreshInterval: 0 });
		await expect(ixc.compact({ force: true, maxCellEntries: 0, lock: false })).rejects.toThrow(/crash/);
		ixc.close();
		const key = keys.slab("d7", 0, base + 1);
		const crashed = store.objects.get(key)!.bytes.slice();
		const r = await ix.compact({ force: true });
		expect(r.outcome).toBe("committed");
		expect(r.splits.length).toBe(1);
		expect(Buffer.from(store.objects.get(key)!.bytes).equals(Buffer.from(crashed))).toBe(false);
		expect(await selfTop(ix, v, [0, 75, 149])).toEqual([0, 75, 149]);
		ix.close();
	});

	it("is safe to rerun after a crash: the same base reuses identical slab bytes, other bytes are never served", async () => {
		const store = new MemoryBlobStore();
		const ix = await fresh(store, "d6");
		const v = data(150, 21);
		await upsertAll(ix, v, 150, 50);
		const base = ix.manifest.rev!;
		// A crash between the slabs and the commit: every manifest put fails once.
		const crashing = new CrashStore(store, (key) => /manifest\.\d+\.json$/.test(key));
		const ixc = await thread.open({ store: crashing, cache: false, refreshInterval: 0 });
		await expect(ixc.compact({ force: true, owner: "do-1" })).rejects.toThrow(/crash/);
		ixc.close();
		const written = [...store.objects.keys()].filter((k) => k.includes("/slab/") && k.endsWith(`.${base + 1}`));
		expect(written.length).toBeGreaterThan(0);
		expect((await head(store, "d6")).rev).toBe(base); // nothing committed
		// The lock was left behind by the crash; the same owner takes it over at once and the rerun commits.
		crashing.armed = false;
		const r = await ix.compact({ force: true, owner: "do-1" });
		expect(r.outcome).toBe("committed");
		expect(r.baseRev).toBe(base);
		expect(await selfTop(ix, v, [0, 75, 149])).toEqual([0, 75, 149]);
		ix.close();
	});
});

/** A store whose puts of matching keys throw while `armed` (a crash in the middle of a run). */
class CrashStore implements BlobStore {
	armed = true;
	constructor(
		private readonly inner: MemoryBlobStore,
		private readonly fails: (key: string) => boolean,
	) {}
	get = (key: string, range?: { offset: number; length: number }) => this.inner.get(key, range);
	getRanges = (key: string, ranges: { offset: number; length: number }[]) => this.inner.getRanges(key, ranges);
	head = (key: string) => this.inner.head(key);
	list = (prefix: string) => this.inner.list(prefix);
	delete = (key: string) => this.inner.delete(key);
	put = async (key: string, data: Uint8Array | string, opts?: { ifAbsent?: boolean; ifMatch?: string }) => {
		if (this.armed && this.fails(key)) throw new Error(`crash before ${key}`);
		return this.inner.put(key, data, opts);
	};
}

/** A store that takes `ms` per slab read (so a deadline falls inside the cells phase). */
class SlowStore implements BlobStore {
	constructor(
		private readonly inner: MemoryBlobStore,
		private readonly ms: number,
	) {}
	get = async (key: string, range?: { offset: number; length: number }) => {
		if (key.includes("/slab/")) await new Promise((r) => setTimeout(r, this.ms));
		return this.inner.get(key, range);
	};
	getRanges = (key: string, ranges: { offset: number; length: number }[]) => this.inner.getRanges(key, ranges);
	head = (key: string) => this.inner.head(key);
	list = (prefix: string) => this.inner.list(prefix);
	delete = (key: string) => this.inner.delete(key);
	put = (key: string, data: Uint8Array | string, opts?: { ifAbsent?: boolean; ifMatch?: string }) => this.inner.put(key, data, opts);
}

