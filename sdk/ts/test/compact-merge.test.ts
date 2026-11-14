/**
 * Merging tombstone sets at the dense compaction's commit (`tombMergeRuns`, docs/update-path.md "Merging tombstone
 * sets"), end to end: a backlog of small writers' revisions (vector upserts, dense- and lexical-scoped re-puts,
 * keyed upserts with texts, deletes, re-inserts of deleted ids) on a generation of many cells is compacted from the
 * same bytes with merging and without, then more small writes come and are compacted too. Both copies answer every
 * request as an uncompacted copy of the same writes does; the merging one keeps far fewer sets, so its readers read
 * fewer, and so do its later calls (a cell whose slab predates the sets needs all of them). A revision before the
 * merges still reads as it did.
 */
import { describe, expect, it } from "vitest";
import { keys, MemoryBlobStore, type CompactResult, type Manifest } from "../src/index.js";
import { rng, thread, unitRows } from "./helpers.js";

const DIM = 16;
const BASE = 600;
const N = 200;
const row = (v: Float32Array, i: number) => v.subarray(i * DIM, (i + 1) * DIM);
type Ix = Awaited<ReturnType<typeof thread.open>>;

function clone(from: MemoryBlobStore): MemoryBlobStore {
	const to = new MemoryBlobStore();
	for (const [k, o] of from.objects) to.objects.set(k, { ...o, bytes: o.bytes.slice() });
	return to;
}

const r = rng(23);
const centers = unitRows(8, DIM, r);
const docs = unitRows(BASE, DIM, r, centers, 0.3);
const v = unitRows(N, DIM, r, centers, 0.3);
const w2 = unitRows(N, DIM, r, centers, 0.3);
const w3 = unitRows(N, DIM, r, centers, 0.3);
const nv = unitRows(N, DIM, r, centers, 0.3);

/** Small writes, one revision each (iterations `from` to `to`); returns the deleted and re-inserted ids. */
async function writes(ix: Ix, from: number, to: number, deleted: number[], reinserted: Map<number, number>): Promise<void> {
	for (let i = from; i < to; i++) {
		await ix.upsert({ ids: [1000 + i], vectors: row(v, i) });
		if (i % 2 === 1) await ix.upsert({ ids: [1000 + i - 1], vectors: row(w2, i - 1), scope: "dense" });
		else await ix.upsert({ ids: [i % BASE], texts: [`retitled ${i} word${i % 7}`], scope: "lexical" });
		if (i % 10 === 3) await ix.upsert({ keys: [`new/${i}`], texts: [`fresh ${i} word${i % 7}`], vectors: row(nv, i) });
		if (i % 5 === 4) {
			await ix.delete([1000 + i - 2]);
			deleted.push(1000 + i - 2);
		}
		// A deleted id comes back with another vector: the case a wrong merge (across its delta) would hide.
		if (i % 15 === 14 && deleted.length > 2) {
			const id = deleted.shift()!;
			await ix.upsert({ ids: [id], vectors: row(w3, i) });
			reinserted.set(id, i);
		}
		if (i % 25 === 24) await ix.deleteKeys([`new/${i - 21}`]);
		for (let j = 0; j < 3; j++) await ix.delete([90_000 + 3 * i + j]);
	}
}

/** Everything a reader can ask. */
async function answers(x: Ix, deleted: number[], reinserted: Map<number, number>) {
	const probe = row(v, 3);
	return {
		live: (await x.query(probe, { topK: 5000, rerank: 5000, nprobe: 1 << 20 })).hits.map((h) => h.id).sort((a, b) => a - b),
		texts: await Promise.all(["word3", "retitled", "document", "fresh", "word5"].map(async (t) => (await x.search({ text: t, topK: 1000, embed: false })).hits.map((h) => h.id).sort((a, b) => a - b))),
		get: [...(await x.get([...Array.from({ length: 80 }, (_, i) => `doc/${i}`), ...Array.from({ length: N }, (_, i) => `new/${i}`)], { text: true })).records].map(([k, d]) => [k, d.id, d.text]),
		keys: (await x.listKeys({ limit: 5000 })).entries.map((e) => [e.key, e.id]),
		self: await Promise.all([...reinserted].map(async ([id, i]) => (await x.query(row(w3, i), { topK: 1, nprobe: 1 << 20 })).hits[0]?.id === id)),
		replaced: await Promise.all([1000, 1010, 1100].map(async (id) => (await x.query(row(w2, id - 1000), { topK: 1, nprobe: 1 << 20 })).hits[0]?.id === id)),
		deleted: deleted.filter((id) => !reinserted.has(id)),
	};
}

describe("merging tombstone sets", () => {
	it(
		"keeps far fewer sets, answers exactly as without, and later calls read fewer sets",
		async () => {
			const store = new MemoryBlobStore();
			await thread.createGeneration({ store, gen: "mg", dim: DIM, lex: { text: true }, compaction: { maxDeltas: 8, maxCellEntries: 12, maxTombstones: 1_000_000 } });
			const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
			await ix.upsert({ keys: Array.from({ length: BASE }, (_, i) => `doc/${i}`), texts: Array.from({ length: BASE }, (_, i) => `document ${i} word${i % 7}`), vectors: docs });
			while ((await ix.compact({ force: true })).more);
			const k = ix.manifest.k1 * ix.manifest.k2;
			expect(k).toBeGreaterThan(50);
			const deleted: number[] = [];
			const reinserted = new Map<number, number>();
			await writes(ix, 0, 150, deleted, reinserted);
			expect(reinserted.size).toBeGreaterThan(5);
			const rev0 = ix.rev;
			const m0 = ix.manifest;
			const written = m0.tombstones!.length;
			expect(m0.deltas!.length).toBeGreaterThan(240);
			expect(written).toBeGreaterThan(800);
			const before = await answers(ix, deleted, reinserted);
			expect(before.self.every(Boolean)).toBe(true);
			expect(before.replaced.every(Boolean)).toBe(true);
			for (const id of before.deleted) expect(before.live).not.toContain(id);
			for (const id of reinserted.keys()) expect(before.live).toContain(id);
			expect(before.keys.length).toBeGreaterThan(BASE / 2);

			// The same bytes compacted with and without merging (and a copy never compacted, for the later writes):
			// 1. calls with a deadline too short for anything but the oldest delta, while compaction is due;
			// 2. a segment compaction, then calls without purge (the sets stay, held by the cells not rewritten);
			// 3. more small writes, and calls while due with the default deadline, then without purge again.
			const ref = clone(store);
			const run = async (merge: boolean) => {
				const s = clone(store);
				const x = await thread.open({ store: s, cache: false, refreshInterval: 0 });
				const calls: CompactResult[] = [];
				const go = (o: Parameters<Ix["compact"]>[0]) => x.compact({ ...o, mergeTombstoneSets: merge, owner: "t" }).then((c) => (calls.push(c), c));
				while (x.compactionDue().due) {
					await go({ deadlineMs: 1 });
					expect(calls.length).toBeLessThan(400);
				}
				const phase1 = { calls: calls.length, sets: x.manifest.tombstones!.length, manifest: x.manifest };
				expect(await answers(x, deleted, reinserted)).toEqual(before);
				await x.compactSegments();
				while ((await go({ force: true, purge: false })).more) expect(calls.length).toBeLessThan(500);
				const phase2 = { calls: calls.length, sets: x.manifest.tombstones!.length };
				expect(await answers(x, deleted, reinserted)).toEqual(before);
				const d2 = [...deleted];
				const r2 = new Map(reinserted);
				await writes(x, 150, N, d2, r2);
				const written2 = x.manifest.tombstones!.length;
				const deltas3 = x.manifest.deltas!.length;
				while (x.compactionDue().due) {
					await go({});
					expect(calls.length).toBeLessThan(600);
				}
				while ((await go({ force: true, purge: false })).more) expect(calls.length).toBeLessThan(700);
				const phase3 = { calls: calls.length, sets: x.manifest.tombstones!.length };
				// 4. A purge: it rewrites the cells whose slabs predate the sets, so it reads every set they hold.
				while ((await go({ force: true })).more) expect(calls.length).toBeLessThan(800);
				// Tiny deltas need no step of their own: with the default deadline one call folds them all, several per cell.
				const first3 = calls[phase2.calls];
				expect(deltas3).toBeGreaterThan(60);
				expect(first3.folded).toBe(deltas3);
				expect(first3.cells).toBeLessThan(deltas3);
				return { s, x, calls, phase1, phase2, phase3, written2, sets: x.manifest.tombstones!.length, d2, r2 };
			};
			const on = await run(true);
			const off = await run(false);
			const rx = await thread.open({ store: ref, cache: false, refreshInterval: 0 });
			await writes(rx, 150, N, [...deleted], new Map(reinserted));
			const after = await answers(rx, on.d2, on.r2);
			expect(after.self.every(Boolean)).toBe(true);
			expect(await answers(on.x, on.d2, on.r2)).toEqual(after);
			expect(await answers(off.x, off.d2, off.r2)).toEqual(after);

			const merged = on.calls.reduce((a, c) => a + c.mergedTombstoneSets, 0);
			const setReads = (x: typeof on, a: number, b = x.calls.length) => x.calls.slice(a, b).reduce((n, c) => n + c.reads.tombSets, 0);
			console.log(
				`merge (k ${k} cells): ${written} sets written, ${merged} merged; left after the short calls ${on.phase1.sets} (without merging ${off.phase1.sets}), after a segment compaction and calls without purge ${on.phase2.sets} (${off.phase2.sets}), after ${N - 150} more iterations (${on.written2} sets, ${off.written2}) and their calls ${on.phase3.sets} (${off.phase3.sets}), after a purge ${on.sets} (${off.sets}); sets read: ${setReads(on, 0, on.phase1.calls)} in ${on.phase1.calls} short calls (${setReads(off, 0, off.phase1.calls)} in ${off.phase1.calls}), ${setReads(on, on.phase2.calls, on.phase3.calls)} in ${on.phase3.calls - on.phase2.calls} calls after the later writes (${setReads(off, off.phase2.calls, off.phase3.calls)} in ${off.phase3.calls - off.phase2.calls}), ${setReads(on, on.phase3.calls)} in ${on.calls.length - on.phase3.calls} purge calls (${setReads(off, off.phase3.calls)} in ${off.calls.length - off.phase3.calls})`,
			);
			expect(off.calls.every((c) => c.mergedTombstoneSets === 0)).toBe(true);
			expect(merged).toBeGreaterThan(written / 2);
			// The merged sets have keys of their own, and each one's count is its ids.
			for (const t of on.phase1.manifest.tombstones!) expect((await on.s.get(t.key))!.byteLength).toBe(4 * t.count);
			expect(on.phase1.manifest.tombstones!.filter((t) => /\/tombstones\.\d+\.m\d+\.u32$/.test(t.key)).length).toBeGreaterThan(10);
			const c1 = on.calls.find((c) => c.mergedTombstoneSets > 0)!;
			const m1 = JSON.parse(new TextDecoder().decode((await on.s.get(keys.manifest("mg", c1.rev)))!)) as Manifest;
			expect(m1.change).toMatchObject({ op: "compact", mergedTombstoneSets: c1.mergedTombstoneSets });
			const last = JSON.parse(new TextDecoder().decode((await on.s.get("idx/mg/compact.last.json"))!));
			expect(last.recent.some((x: { mergedTombstoneSets?: number }) => (x.mergedTombstoneSets ?? 0) > 0)).toBe(true);
			// Far fewer sets at every step: what a cold open reads.
			expect(on.phase1.sets).toBeLessThan(off.phase1.sets / 1.5);
			expect(on.phase2.sets).toBeLessThan(off.phase2.sets / 10);
			expect(on.phase3.sets).toBeLessThan(off.phase3.sets / 4);
			expect(on.phase2.sets).toBeLessThan(written / 20);
			// Calls read fewer sets: in the short calls (the cells rewritten first need the merged sets behind the oldest
			// delta), not after the later writes (those calls need the new sets, which deltas between them keep apart), and
			// far fewer in the purge (the old sets of the cells it rewrites are merged).
			expect(setReads(on, 0, on.phase1.calls)).toBeLessThan(setReads(off, 0, off.phase1.calls));
			expect(setReads(on, on.phase2.calls, on.phase3.calls)).toBeLessThanOrEqual(setReads(off, off.phase2.calls, off.phase3.calls));
			expect(setReads(on, on.phase3.calls)).toBeLessThan(setReads(off, off.phase3.calls) / 4);

			// A revision before the merges still reads as it did: pinned, and as the head again (a rollback writes a
			// revision with its content), from which compaction goes on.
			const pinned = await thread.open({ store: on.s, gen: "mg", rev: rev0, cache: false });
			expect(await answers(pinned, deleted, reinserted)).toEqual(before);
			pinned.close();
			const back = clone(on.s);
			const p1 = await thread.open({ store: back, cache: false, refreshInterval: 0 });
			const head = p1.rev;
			p1.close();
			await back.put(keys.manifest("mg", head + 1), JSON.stringify({ ...m0, rev: head + 1, parentRev: head, nextSeq: on.x.manifest.nextSeq, nextId: on.x.manifest.nextId, change: { op: "rollback", toRev: rev0, fromRev: head } }));
			await back.put(keys.current(), JSON.stringify({ gen: "mg", rev: head + 1 }));
			const rolled = await thread.open({ store: back, cache: false, refreshInterval: 0 });
			expect(rolled.rev).toBe(head + 1);
			expect(await answers(rolled, deleted, reinserted)).toEqual(before);
			await rolled.compactSegments();
			while ((await rolled.compact({ force: true, purge: false })).more);
			expect(rolled.manifest.tombstones!.length).toBeLessThan(written / 20);
			expect(await answers(rolled, deleted, reinserted)).toEqual(before);
			rolled.close();
			rx.close();
			on.x.close();
			off.x.close();
			ix.close();
		},
		180_000,
	);
});
