/**
 * Dense compaction of a large backlog (inseam, 2026-11-14: 767 deltas and 2,531 tombstone sets, about 4,000 gets
 * before the plan, every call `deadline` with 0 cells). The read phase reads the deltas' cells oldest first until
 * the slice is full or its share of the deadline is used, a tombstone set only when a rewritten cell or a folded
 * delta is filtered by it, and the oldest delta folds in every call whatever the deadline.
 */
import { describe, expect, it } from "vitest";
import { MemoryBlobStore, parseSlabsTable, type BlobStore, type Manifest } from "../src/index.js";
import { rng, thread, unitRows } from "./helpers.js";

const DIM = 16;

/**
 * A store that counts the gets of one compaction (and which keys they were for): from `reset()` until the run
 * releases `compact.lock`, its last step. After it the index adopts the new revision and loads the deltas it serves.
 */
class CountingStore implements BlobStore {
	gets = 0;
	keys: string[] = [];
	private counting = false;
	constructor(private readonly base: MemoryBlobStore) {}
	reset() {
		this.gets = 0;
		this.keys = [];
		this.counting = true;
	}
	get = (key: string, range?: { offset: number; length: number }) => {
		if (this.counting) {
			this.gets++;
			this.keys.push(key);
		}
		return this.base.get(key, range);
	};
	getRanges = (key: string, ranges: { offset: number; length: number }[]) => this.base.getRanges(key, ranges);
	head = (key: string) => this.base.head(key);
	list = (prefix: string) => this.base.list(prefix);
	delete = (key: string) => {
		if (key.endsWith("/compact.lock")) this.counting = false;
		return this.base.delete(key);
	};
	put = (key: string, data: Uint8Array | string, opts?: { ifAbsent?: boolean; ifMatch?: string }) => this.base.put(key, data, opts);
}

const row = (v: Float32Array, i: number) => v.subarray(i * DIM, (i + 1) * DIM);
type Ix = Awaited<ReturnType<typeof thread.open>>;

/** The live dense ids: every entry a full scan returns. */
async function liveIds(ix: Ix, q: Float32Array): Promise<number[]> {
	return (await ix.query(q, { topK: 1000, rerank: 1000, nprobe: 1 << 20 })).hits.map((h) => h.id).sort((a, b) => a - b);
}

async function slabsOf(store: BlobStore, m: Manifest) {
	return parseSlabsTable((await store.get(m.files!.slabs!))!, m.k1 * m.k2, "slabs");
}

describe("dense compaction of a large backlog", () => {
	it(
		"folds at least the oldest delta in every call, reads only what the slice needs, and keeps every result",
		async () => {
			const store = new MemoryBlobStore();
			await thread.createGeneration({ store, gen: "bl", dim: DIM, lex: { text: true }, compaction: { maxDeltas: 8, maxCellEntries: 40, maxTombstones: 1_000_000 } });
			const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
			const r = rng(17);
			const centers = unitRows(6, DIM, r);
			// 60 documents with text and vectors, compacted into a few cells.
			const docs = unitRows(60, DIM, r, centers, 0.3);
			await ix.upsert({ keys: Array.from({ length: 60 }, (_, i) => `doc/${i}`), texts: Array.from({ length: 60 }, (_, i) => `document ${i} word${i % 7}`), vectors: docs });
			while ((await ix.compact({ force: true })).more);
			expect(ix.manifest.k1 * ix.manifest.k2).toBeGreaterThan(1);
			// The backlog: many small upserts, each a delta and a tombstone set, dense-scoped re-puts, lexical-scoped
			// re-puts (a set only the lexical side applies), deletes of live ids and of unknown ones.
			const N = 200;
			const v = unitRows(N, DIM, r, centers, 0.3);
			const w2 = unitRows(N, DIM, r, centers, 0.3);
			const live = new Set(Array.from({ length: 60 }, (_, i) => i));
			for (let i = 0; i < N; i++) {
				await ix.upsert({ ids: [1000 + i], vectors: row(v, i) });
				live.add(1000 + i);
				if (i % 2 === 1) await ix.upsert({ ids: [1000 + i - 1], vectors: row(w2, i - 1), scope: "dense" });
				else await ix.upsert({ ids: [i % 60], texts: [`retitled ${i} word${i % 7}`], scope: "lexical" });
				if (i % 5 === 4) {
					await ix.delete([1000 + i - 2]);
					live.delete(1000 + i - 2);
				}
				for (let j = 0; j < 4; j++) await ix.delete([90_000 + 4 * i + j]);
			}
			const m0 = ix.manifest;
			const sets0 = m0.tombstones ?? [];
			expect(m0.deltas!.length).toBe(300);
			expect(sets0.length).toBeGreaterThan(1200);
			const lexicalKeys = new Set(sets0.filter((t) => t.scope === "lexical").map((t) => t.key));
			expect(lexicalKeys.size).toBe(100);
			const probe = row(v, 3);
			const before = await liveIds(ix, probe);
			expect(before).toEqual([...live].sort((a, b) => a - b));
			const texts = async (i: Ix) => Promise.all(["word3", "retitled", "document"].map(async (t) => (await i.search({ text: t, topK: 100, embed: false })).hits.map((h) => h.id).sort((a, b) => a - b)));
			const textBefore = await texts(ix);
			const selfBefore = await Promise.all([1000, 1001, 1003, 1050, 1199].map(async (id) => (await ix.query(row(v, id - 1000), { topK: 1, nprobe: 1 << 20 })).hits[0].id));

			// Successive calls with a deadline too short for anything but the oldest delta.
			const counting = new CountingStore(store);
			const cx = await thread.open({ store: counting, cache: false, refreshInterval: 0 });
			const calls: { deltas: number; folded: number; gets: number; deltasRead: number; setsRead: number; setsListed: number }[] = [];
			let deltas = cx.manifest.deltas!.length;
			while (cx.compactionDue().due) {
				const base = cx.manifest;
				// What the slice can need: the dense-view sets at least as new as the oldest non-empty slab or newer than the
				// oldest delta (a rewritten cell and a folded delta are filtered by no others).
				const slabs = await slabsOf(store, base);
				const oldestSlab = Math.min(...slabs.filter((s) => s[2] > 0).map((s) => (s.length > 4 ? (s[4] as number) : 0)));
				const oldestDelta = Math.min(...base.deltas!.map((d) => d.addedRev));
				const mayNeed = (base.tombstones ?? []).filter((t) => t.scope !== "lexical" && (t.rev >= oldestSlab || t.rev > oldestDelta)).length;
				counting.reset();
				const c = await cx.compact({ deadlineMs: 1, owner: "backlog" });
				expect(["committed", "partial"]).toContain(c.outcome);
				expect(c.folded).toBeGreaterThanOrEqual(1);
				expect(c.remainingDeltas).toBeLessThan(deltas);
				expect(c.note).toMatch(/^read phase passed the deadline: \d+ ms for 8 of \d+ deltas and \d+ of \d+ tombstone sets \(deadlineMs 1\); the cells the oldest delta needs ran anyway/);
				// Only the first window of deltas is read; no lexical-scoped set is ever read.
				expect(c.reads.deltas).toBeLessThanOrEqual(8);
				expect(c.reads.deltasListed).toBe(deltas);
				expect(c.reads.tombSets).toBeLessThanOrEqual(mayNeed);
				expect(counting.keys.filter((k) => lexicalKeys.has(k))).toEqual([]);
				// Every get is one the slice needs: the deltas read (whole, one get each), the sets read, a slab per cell
				// rewritten, and a constant (lock, manifests, slabs table, the commit, compact.last.json).
				expect(counting.gets).toBeLessThanOrEqual(c.reads.deltas + c.reads.tombSets + c.cells + 16);
				calls.push({ deltas, folded: c.folded, gets: counting.gets, deltasRead: c.reads.deltas, setsRead: c.reads.tombSets, setsListed: c.reads.tombSetsListed });
				deltas = c.remainingDeltas;
				expect(calls.length).toBeLessThan(300);
			}
			expect(deltas).toBeLessThanOrEqual(8);
			// Before, each call read every set and two ranges of every delta before it planned (and found no time left).
			const firstOld = 2 * calls[0].deltas + calls[0].setsListed;
			console.log(`backlog: ${calls.length} calls, deltas ${calls[0].deltas} -> ${deltas}; gets per call: first ${calls[0].gets} (read-everything: ${firstOld}+), max ${Math.max(...calls.map((x) => x.gets))}, mean ${Math.round(calls.reduce((a, x) => a + x.gets, 0) / calls.length)}; sets read per call: max ${Math.max(...calls.map((x) => x.setsRead))} of ${calls[0].setsListed}`);
			for (const x of calls) expect(x.gets).toBeLessThan(2 * x.deltas + x.setsListed);

			// The same live documents on both sides, after the short calls and after a full drain.
			expect(await liveIds(cx, probe)).toEqual(before);
			expect(await texts(cx)).toEqual(textBefore);
			while ((await cx.compact({ force: true })).more);
			expect(cx.manifest.deltas).toEqual([]);
			expect(await liveIds(cx, probe)).toEqual(before);
			expect(await texts(cx)).toEqual(textBefore);
			expect(await Promise.all([1000, 1001, 1003, 1050, 1199].map(async (id) => (await cx.query(row(v, id - 1000), { topK: 1, nprobe: 1 << 20 })).hits[0].id))).toEqual(selfBefore);
			// A replaced vector finds its id; the deleted and replaced ones are gone.
			expect((await cx.query(row(w2, 10), { topK: 1, nprobe: 1 << 20 })).hits[0].id).toBe(1010);
			expect(before).not.toContain(1002);
			cx.close();
			ix.close();
		},
		60_000,
	);

	it("says when the read phase passed the deadline and nothing could start", async () => {
		const store = new MemoryBlobStore();
		await thread.createGeneration({ store, gen: "rd", dim: DIM, lex: null, compaction: { maxDeltas: 4, maxCellEntries: 0 } });
		const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
		const v = unitRows(100, DIM, rng(3));
		await ix.upsert({ ids: Array.from({ length: 100 }, (_, i) => i), vectors: v });
		while ((await ix.compact({ force: true })).more);
		expect(ix.manifest.deltas).toEqual([]);
		// No delta, one cell over the cap: only force's split is left, and no time for it.
		const rev = ix.rev;
		const r = await ix.compact({ force: true, maxCellEntries: 40, deadlineMs: 1 });
		expect(r.outcome).toBe("deadline");
		expect(r.more).toBe(true);
		expect(r.cells).toBe(0);
		expect(r.note).toMatch(/^read phase passed the deadline: \d+ ms for 0 of 0 deltas and 0 of 0 tombstone sets \(deadlineMs 1\); no cell started$/);
		expect(r.reads).toMatchObject({ deltas: 0, deltasListed: 0, tombSets: 0, tombSetsListed: 0 });
		expect(ix.rev).toBe(rev);
		const last = JSON.parse(new TextDecoder().decode((await store.get("idx/rd/compact.last.json"))!));
		expect(last.last).toMatchObject({ outcome: "deadline", note: r.note, reads: { deltas: 0, deltasListed: 0, tombSets: 0, tombSetsListed: 0 } });
		// With time, the split happens.
		const r2 = await ix.compact({ force: true, maxCellEntries: 40 });
		expect(r2.outcome).toBe("committed");
		expect(r2.splits.length).toBe(1);
		ix.close();
	});

	it("keeps the mean of a split cell while deltas are unread, and reads none of them to commit", async () => {
		const store = new MemoryBlobStore();
		await thread.createGeneration({ store, gen: "sp", dim: DIM, lex: null, compaction: { maxDeltas: 4, maxCellEntries: 40 } });
		const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
		const r = rng(9);
		const v = unitRows(68, DIM, r, unitRows(4, DIM, r), 0.3);
		await ix.upsert({ ids: Array.from({ length: 38 }, (_, i) => i), vectors: v.subarray(0, 38 * DIM) });
		while ((await ix.compact({ force: true })).more);
		expect(ix.manifest.k2).toBe(1);
		expect((await slabsOf(store, ix.manifest))[0][2]).toBe(38);
		const fine0 = new Float32Array((await store.get(ix.manifest.files!.fine))!.slice().buffer);
		for (let i = 38; i < 68; i++) await ix.upsert({ ids: [i], vectors: row(v, i) });
		const unread = ix.manifest.deltas!.slice(8).map((d) => d.key);
		const counting = new CountingStore(store);
		const cx = await thread.open({ store: counting, cache: false, refreshInterval: 0 });
		counting.reset();
		// The first window (8 deltas) takes cell 0 to 46 entries, over the cap: it splits, with 22 deltas unread.
		const c = await cx.compact({ deadlineMs: 1 });
		expect(c.outcome).toBe("committed");
		expect(c.reads).toMatchObject({ deltas: 8, deltasListed: 30 });
		expect(c.folded).toBe(8);
		expect(c.splits).toHaveLength(1);
		expect(c.splits[0]).toMatchObject({ cell: 0, newCell: 1, entries: 46 });
		// The cell kept its mean (the unread deltas' entries in it were encoded against it); the new cell has another.
		const fine1 = new Float32Array((await store.get(cx.manifest.files!.fine))!.slice().buffer);
		expect(cx.manifest.k2).toBe(2);
		expect(Array.from(fine1.subarray(0, DIM))).toEqual(Array.from(fine0.subarray(0, DIM)));
		expect(Array.from(fine1.subarray(DIM, 2 * DIM))).not.toEqual(Array.from(fine0.subarray(0, DIM)));
		// Pinned, the split needs no check of the deltas that stay: none of the unread ones was read.
		expect(counting.keys.filter((k) => unread.includes(k))).toEqual([]);
		while ((await cx.compact({ force: true })).more);
		expect(cx.manifest.deltas).toEqual([]);
		const ids = Array.from({ length: 68 }, (_, i) => i);
		expect(await Promise.all(ids.map(async (i) => (await cx.query(row(v, i), { topK: 1, nprobe: 1 << 20 })).hits[0].id))).toEqual(ids);
		cx.close();
		ix.close();
	});
});
