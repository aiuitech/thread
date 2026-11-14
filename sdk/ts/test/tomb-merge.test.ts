/**
 * The merge rule for tombstone sets (`tombMergeRuns`), checked on random histories: replacing each run by one set
 * (its newest rev, the union of its ids, its scope) leaves every filter a reader builds unchanged (`tombFor` per slab,
 * `tombForDelta` per delta, `tombForSegment` and `tombHides` per segment), and `tombDroppable` of the merged set is
 * "every member droppable", now and after the history moves on (deltas folded, segments merged, slabs rewritten).
 * Every cut it makes is needed: merging the two sets on either side of one changes a filter.
 */
import { describe, expect, it } from "vitest";
import { buildTombState, tombDroppable, tombFor, tombForDelta, tombForSegment, tombHides, tombMergeRuns, type SlabRow, type TombRef, type TombScope, type TombSet } from "../src/index.js";
import { rng } from "./helpers.js";

interface History {
	sets: (TombRef & { ids: Uint32Array })[];
	deltas: { addedRev: number }[];
	slabs: SlabRow[];
	segments: { addedRev: number }[];
	baseIds: number;
}

/**
 * A random revision: each rev 1..n writes some of a set (scope absent, dense or lexical), a delta, a segment; slabs
 * at rev 0 or at random revs (some empty). Every set holds an id of its own (so any two sets differ) and some shared
 * ones (so unions overlap), some of them build ids.
 */
function history(r: () => number): History {
	const n = 4 + Math.floor(r() * 40);
	const sets: History["sets"] = [];
	const deltas: { addedRev: number }[] = [];
	const segments: { addedRev: number }[] = [];
	const baseIds = r() < 0.5 ? 0 : 50;
	const scopes: (TombScope | undefined)[] = [undefined, "dense", "lexical"];
	for (let rev = 1; rev <= n; rev++) {
		if (r() < 0.8) {
			const ids = new Set<number>([1000 + rev]);
			const shared = Math.floor(r() * 4);
			for (let i = 0; i < shared; i++) ids.add(baseIds > 0 && r() < 0.15 ? Math.floor(r() * baseIds) : 100 + Math.floor(r() * 30));
			const u = Uint32Array.from([...ids].sort((a, b) => a - b));
			const scope = scopes[Math.floor(r() * 3)];
			sets.push({ rev, key: `t${rev}`, count: u.length, ids: u, ...(scope ? { scope } : {}) });
		}
		if (r() < 0.15) deltas.push({ addedRev: rev });
		if (r() < 0.15) segments.push({ addedRev: rev });
	}
	const k = 1 + Math.floor(r() * 6);
	const slabs: SlabRow[] = Array.from({ length: k }, () => {
		const rev = r() < 0.3 ? 0 : 1 + Math.floor(r() * (n + 2));
		const count = r() < 0.2 ? 0 : 5;
		return rev > 0 ? [0, 0, count, 100, rev] : [0, 0, count, 100];
	});
	return { sets, deltas, slabs, segments, baseIds };
}

const toSet = (t: History["sets"][number]): TombSet => ({ rev: t.rev, ids: t.ids, ...(t.scope ? { scope: t.scope } : {}) });

/** The sets after merging `groups` (each replaced by one set at its newest member's rev). */
function merged(sets: History["sets"], groups: History["sets"][]): History["sets"] {
	const out = sets.filter((t) => !groups.some((g) => g.includes(t)));
	for (const g of groups) {
		const ids = Uint32Array.from(new Set(g.flatMap((t) => [...t.ids]))).sort();
		const last = g[g.length - 1];
		out.push({ rev: last.rev, key: `m${g[0].rev}-${last.rev}`, count: ids.length, ids, ...(last.scope ? { scope: last.scope } : {}) });
	}
	return out.sort((x, y) => x.rev - y.rev);
}

/** Every filter a reader of the revision builds, and the lexical hides of every id, as comparable strings. */
function filters(h: History, sets: History["sets"]): string[] {
	const t = buildTombState(sets.map(toSet));
	const hex = (b: Uint8Array) => Array.from(new Uint32Array(b.slice().buffer)).join(",");
	const out: string[] = [];
	for (const row of h.slabs) {
		const rev = row.length > 4 ? (row[4] as number) : 0;
		out.push(`slab ${rev}: ${hex(tombFor(t, rev))}`);
	}
	for (const d of h.deltas) out.push(`delta ${d.addedRev}: ${hex(tombForDelta(t, d.addedRev))}`);
	const ids = [...new Set(h.sets.flatMap((s) => [...s.ids]))].sort((a, b) => a - b);
	for (const s of h.segments) {
		out.push(`segment ${s.addedRev}: ${hex(tombForSegment(t, s.addedRev))}`);
		out.push(`hides ${s.addedRev}: ${ids.filter((id) => tombHides(t, id, s.addedRev)).join(",")}`);
	}
	return out;
}

/** Keeps sets with build ids apart from the others, as the compaction does. */
function byBuild(run: History["sets"], baseIds: number): History["sets"][] {
	if (run[0].scope === "lexical" || !baseIds) return [run];
	const build = run.filter((t) => t.ids.some((id) => id < baseIds));
	const rest = run.filter((t) => !build.includes(t));
	return [build, rest].filter((g) => g.length >= 2);
}

/** The history moved on: some deltas folded, some segments merged into a newer one, some slabs rewritten later. */
function later(h: History, r: () => number): History {
	const top = Math.max(0, ...h.sets.map((t) => t.rev)) + 1;
	return {
		...h,
		deltas: h.deltas.filter(() => r() < 0.5),
		segments: [...h.segments.filter(() => r() < 0.5), ...(r() < 0.5 ? [{ addedRev: top }] : [])],
		slabs: h.slabs.map((row) => (r() < 0.5 ? row : [row[0], row[1], row[2], row[3], top + Math.floor(r() * 3)])),
	};
}

describe("tombMergeRuns", () => {
	it("merges only sets no data object tells apart: every filter and every drop stays the same", () => {
		const r = rng(2026);
		let runs = 0;
		let members = 0;
		let cuts = 0;
		for (let trial = 0; trial < 3000; trial++) {
			const h = history(r);
			const found = tombMergeRuns(h.sets, h.deltas, h.slabs, h.segments) as History["sets"][];
			for (const run of found) {
				expect(run.length).toBeGreaterThanOrEqual(2);
				expect(new Set(run.map((t) => t.scope)).size).toBe(1);
				for (let i = 1; i < run.length; i++) expect(run[i].rev).toBeGreaterThan(run[i - 1].rev);
			}
			// Each run whole, and a random subset of each (also mergeable), with build-id sets kept apart.
			const whole = found.flatMap((run) => byBuild(run, h.baseIds));
			const some = found.map((run) => run.filter(() => r() < 0.6)).filter((g) => g.length >= 2);
			for (const groups of [whole, some]) {
				const after = merged(h.sets, groups);
				expect(filters(h, after)).toEqual(filters(h, h.sets));
				// Droppable: the merged set exactly when all its members are, now and as the history moves on.
				for (const state of [h, later(h, r), later(later(h, r), r)]) {
					expect(filters(state, merged(h.sets, groups))).toEqual(filters(state, h.sets));
					if (groups !== whole) continue;
					for (const g of groups) {
						const m = after.find((t) => t.rev === g[g.length - 1].rev)!;
						const all = g.every((t) => tombDroppable(t, state.deltas, state.slabs, h.baseIds, state.segments));
						expect(tombDroppable(m, state.deltas, state.slabs, h.baseIds, state.segments)).toBe(all);
					}
				}
			}
			runs += found.length;
			members += found.reduce((a, run) => a + run.length, 0);
			// Every cut is needed: two consecutive sets of a scope that are not in one run differ for some data object.
			for (const scope of [undefined, "dense", "lexical"] as const) {
				const of = h.sets.filter((t) => t.scope === scope);
				for (let i = 1; i < of.length; i++) {
					const together = found.some((run) => run.includes(of[i - 1]) && run.includes(of[i]));
					if (together) continue;
					cuts++;
					expect(filters(h, merged(h.sets, [[of[i - 1], of[i]]]))).not.toEqual(filters(h, h.sets));
				}
			}
		}
		// The histories exercise both outcomes.
		expect(runs).toBeGreaterThan(1000);
		expect(members).toBeGreaterThan(3000);
		expect(cuts).toBeGreaterThan(1000);
	});

	it("keeps the edges: a delta or segment at the older set's rev cuts, at the newer one's does not; a slab the other way", () => {
		const sets: TombRef[] = [
			{ rev: 3, key: "a", count: 1 },
			{ rev: 5, key: "b", count: 1 },
		];
		const slab0: SlabRow[] = [[0, 0, 1, 1]];
		const runs = (deltas: number[], slabRevs: number[], segments: number[], s: TombRef[] = sets) =>
			tombMergeRuns(
				s,
				deltas.map((addedRev) => ({ addedRev })),
				slabRevs.length ? slabRevs.map((rev) => [0, 0, 1, 1, rev] as SlabRow) : slab0,
				segments.map((addedRev) => ({ addedRev })),
			).length;
		expect(runs([], [], [])).toBe(1);
		expect(runs([3], [], [])).toBe(0); // a delta added with the older set: the newer one hides its entries
		expect(runs([5], [], [])).toBe(1);
		expect(runs([2], [], [])).toBe(1);
		expect(runs([], [3], [])).toBe(1); // a slab at 3 is filtered by both (rev >= 3)
		expect(runs([], [4], [])).toBe(0);
		expect(runs([], [5], [])).toBe(0); // a slab at 5 is filtered by the newer one only
		expect(runs([], [6], [])).toBe(1);
		expect(runs([], [], [3])).toBe(0);
		expect(runs([], [], [5])).toBe(1);
		// Scoped sets see only their side's objects; sets of different scopes never merge.
		const dense = sets.map((t) => ({ ...t, scope: "dense" as const }));
		const lexical = sets.map((t) => ({ ...t, scope: "lexical" as const }));
		expect(runs([], [], [3], dense)).toBe(1);
		expect(runs([3], [4], [], lexical)).toBe(1);
		expect(runs([], [], [3], lexical)).toBe(0);
		expect(runs([], [], [], [sets[0], { ...sets[1], scope: "dense" }])).toBe(0);
	});
});
