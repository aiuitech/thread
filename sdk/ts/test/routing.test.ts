/** Compact routing tables (fine.f16 / fine.i8): building them, opening with `routing`, fallbacks, snapshots. */
import { describe, expect, it } from "vitest";
import { asF32, chooseRouting, groupedTableOf, manifestFiles, snapshotKeys, loadRevState, type BlobStore } from "../src/index.js";
import { addRoutingRev, bruteForce, buildGen, rng, thread, unitRows, w } from "./helpers.js";

const queries = (n: number, dim: number, seed: number) => {
	const qs = unitRows(n, dim, rng(seed));
	return Array.from({ length: n }, (_, i) => qs.subarray(i * dim, (i + 1) * dim));
};

describe("compact routing tables", () => {
	it("encode/decode: f16 within half precision, i8 within a step of its row scale", () => {
		const dim = 16;
		const fine = unitRows(20, dim, rng(3));
		for (const elem of ["f16", "i8"] as const) {
			const b = w.routing_table_encode(fine, dim, elem);
			expect(b.byteLength).toBe(elem === "f16" ? 20 * 2 * dim : 4 * dim + 20 * (4 + dim));
			expect(w.routing_table_bytes(dim, 20, elem)).toBe(b.byteLength);
			expect(w.routing_row_bytes(dim, elem)).toBe(elem === "f16" ? 2 * dim : 4 + dim);
			const d = w.routing_table_decode(b, dim, elem);
			for (let i = 0; i < fine.length; i++) expect(Math.abs(d[i] - fine[i])).toBeLessThan(elem === "f16" ? 1e-3 : 1 / 127);
		}
		expect(Array.from(w.routing_table_decode(w.routing_table_encode(fine, dim, "f32"), dim, "f32"))).toEqual(Array.from(fine));
		expect(() => w.routing_table_encode(fine, dim, "f8")).toThrow(/f32, f16 or i8/);
		const coarse = new Float32Array(dim).fill(0.25);
		expect(() => w.Searcher.from_table(dim, 1, 20, coarse, w.routing_table_encode(fine, dim, "f16").subarray(2), "f16")).toThrow(/20 rows of/);
	});

	it("open({routing}) loads the requested table, routes like f32 and falls back when the revision has none", async () => {
		const b = await buildGen("gr", 3, { k2: 16, n: 600 });
		// a revision without routing tables: every option loads fine.f32
		const legacy = await thread.open({ store: b.store, cache: false, routing: "auto" });
		const q0 = queries(1, b.dim, 9)[0];
		expect((await legacy.query(q0)).stats.routingTable).toBe("f32");
		legacy.close();

		const { manifest } = await addRoutingRev(b.store, "gr");
		expect(manifest.routing!.f16!.key).toBe("idx/gr/fine.f16");
		expect(manifest.routing!.i8!.bytes).toBe(4 * b.dim + 16 * (4 + b.dim)); // centre + rows
		expect(chooseRouting(manifest, manifestFiles(manifest), "auto")).toEqual({ elem: "i8", key: "idx/gr/fine.i8" });
		const f32 = await thread.open({ store: b.store, cache: false });
		const results: Record<string, number[][]> = {};
		for (const routing of ["f32", "f16", "i8", "auto"] as const) {
			const ix = routing === "f32" ? f32 : await thread.open({ store: b.store, cache: false, routing });
			results[routing] = [];
			for (const q of queries(30, b.dim, 5)) {
				const r = await ix.query(q, { nprobe: 4, topK: 5 });
				expect(r.stats.routingTable).toBe(routing === "auto" ? "i8" : routing);
				expect(r.hits.length).toBe(5);
				results[routing].push(r.stats.cells);
			}
			if (ix !== f32) ix.close();
		}
		// f16 decodes to the f32 means within 2^-11: the same probe sets here; i8 almost always.
		expect(results.f16).toEqual(results.f32);
		let same = 0;
		results.i8.forEach((cells, i) => (same += cells.filter((c) => results.f32[i].includes(c)).length));
		expect(same / (30 * 4)).toBeGreaterThan(0.95);
		f32.close();
	});

	it("tables made from other means are ignored (a split wrote new ones); snapshots list the usable tables only", async () => {
		const b = await buildGen("gs", 3, { k2: 8 });
		await addRoutingRev(b.store, "gs", { tables: ["f16"] });
		const ok = await loadRevState(b.store, "gs", 1);
		expect(snapshotKeys(ok)).toContain("idx/gs/fine.f16");
		expect(chooseRouting(ok.manifest, ok.files, "i8").elem).toBe("f32"); // no i8 table: f32
		await addRoutingRev(b.store, "gs", { staleFine: true });
		const stale = await loadRevState(b.store, "gs", 2);
		expect(snapshotKeys(stale)).not.toContain("idx/gs/fine.f16");
		expect(chooseRouting(stale.manifest, stale.files, "f16")).toEqual({ elem: "f32", key: "idx/gs/fine.f32" });
		const ix = await thread.open({ store: b.store, cache: false, routing: "f16" });
		expect((await ix.query(queries(1, b.dim, 2)[0])).stats.routingTable).toBe("f32");
		ix.close();
	});
});

/** A store whose reads of `key` wait until `release()` (to hold the full routing table back). */
function gated(inner: BlobStore, key: string): { store: BlobStore; release: () => void; reads: string[] } {
	let release!: () => void;
	const gate = new Promise<void>((r) => (release = r));
	const reads: string[] = [];
	const store: BlobStore = {
		get: async (k, range, o) => {
			reads.push(range ? `${k}@${range.offset}+${range.length}` : k);
			if (k === key) await gate;
			return inner.get(k, range, o);
		},
		getRanges: inner.getRanges ? async (k, ranges, o) => (reads.push(`${k}#${ranges.length}`), inner.getRanges!(k, ranges, o)) : undefined,
		head: (k) => inner.head(k),
		list: inner.list ? (p) => inner.list!(p) : undefined,
		put: inner.put ? (k, d, o) => inner.put!(k, d, o) : undefined,
	};
	return { store, release, reads };
}

describe("cold routing (coldRouting: tree)", () => {
	it("grouped table: layout from the tree, partial routing equals full beam / flat routing", async () => {
		const b = await buildGen("gt", 3, { k2: 48, n: 1500 });
		const { manifest } = await addRoutingRev(b.store, "gt", { tree: { groups: 6, elem: "f32" } });
		const g = groupedTableOf(manifest, manifestFiles(manifest))!;
		expect(g.key).toBe("idx/gt/fine.grouped.1.f32");
		expect(g.rows).toBe(48);
		const bytes = (await b.store.get(g.key))!;
		expect(Array.from(w.grouped_table_info(bytes))).toEqual([0, b.dim, 48, 6, 48, g.dataOff, 4 * b.dim, bytes.byteLength]);
		const tree = (await b.store.get(manifest.tree!.key))!;
		const full = await thread.open({ store: b.store, cache: false });
		const gate = gated(b.store, "idx/gt/fine.f32");
		const cold = await thread.open({ store: gate.store, cache: false, coldRouting: "tree", coldBeam: 6 });
		expect(cold.routingState).toBe("cold");
		expect(gate.reads).toContain(manifest.tree!.key);
		for (const q of queries(20, b.dim, 4)) {
			const a = await full.query(q, { nprobe: 5 });
			const c = await cold.query(q, { nprobe: 5 });
			// beam = every group: the exact flat ranking over the (f32) means, so the same probes and hits
			expect(c.stats.routing).toBe("cold");
			expect(c.stats.cells).toEqual(a.stats.cells);
			expect(c.hits).toEqual(a.hits);
		}
		// the searcher a ColdRouter matches: Searcher.route_beam over the full table with the same tree
		const s = new w.Searcher(b.dim, 1, 48, new Float32Array(b.dim).fill(0.25), asF32((await b.store.get("idx/gt/fine.f32"))!, "fine"));
		s.set_tree(tree);
		const q = queries(1, b.dim, 8)[0];
		const qn = w.normalize_vector(q);
		const r = await cold.query(q, { nprobe: 4, beam: 2 });
		expect(r.stats.cells).toEqual(Array.from(s.route_beam(qn, 2, 4)));
		s.free();
		gate.release();
		await cold.whenFullRouting();
		expect(cold.routingState).toBe("full");
		const after = await cold.query(q, { nprobe: 4 });
		expect(after.stats.routing).toBe("flat");
		expect(after.stats.coldRouting).toBeUndefined();
		full.close();
		cold.close();
	});

	it("reads only the beam's groups, once; later queries reuse them; i8 rows route almost like f32", async () => {
		const b = await buildGen("gb", 3, { k2: 64, n: 2000 });
		await addRoutingRev(b.store, "gb", { tree: { groups: 8 } }); // i8 rows
		const gate = gated(b.store, "idx/gb/fine.f32");
		const cold = await thread.open({ store: gate.store, cache: false, coldRouting: "tree", coldBeam: 2 });
		const full = await thread.open({ store: b.store, cache: false });
		const q = queries(1, b.dim, 21)[0];
		const first = await cold.query(q, { nprobe: 4 });
		expect(first.stats.routingTable).toBe("i8");
		expect(first.stats.coldRouting!.beam).toBe(2);
		expect(first.stats.coldRouting!.groups).toBeGreaterThanOrEqual(1);
		expect(first.stats.coldRouting!.groups).toBeLessThanOrEqual(2);
		expect(first.stats.coldRouting!.reads).toBe(1); // one (multi-)range request for the groups
		// i8 rows: the first read also brings the table's head (directory + the centre the rows are residuals from)
		const g = groupedTableOf(cold.manifest, cold.revision.files)!;
		expect(first.stats.coldRouting!.bytes).toBe(g.dataOff + first.stats.coldRouting!.loadedCells * (4 + b.dim));
		const again = await cold.query(q, { nprobe: 4 });
		expect(again.stats.coldRouting!.groups).toBe(0);
		expect(again.stats.coldRouting!.bytes).toBe(0);
		expect(again.stats.cells).toEqual(first.stats.cells);
		// recall of cold routing at a wide beam stays close to the full table's on this tiny index
		let hitsCold = 0;
		let hitsFull = 0;
		for (const qq of queries(30, b.dim, 22)) {
			const truth = bruteForce(b.vectors, b.dim, qq, 5);
			hitsCold += (await cold.query(qq, { nprobe: 8, topK: 5, beam: 4 })).hits.filter((h) => truth.includes(h.id)).length;
			hitsFull += (await full.query(qq, { nprobe: 8, topK: 5 })).hits.filter((h) => truth.includes(h.id)).length;
		}
		expect(hitsCold).toBeGreaterThanOrEqual(hitsFull * 0.9);
		gate.release();
		full.close();
		cold.close();
	});

	it("delta segments get exact q·mean for their cells (rows read on demand); falls back without a grouped table", async () => {
		const b = await buildGen("gd", 3, { k2: 32, n: 800 });
		await addRoutingRev(b.store, "gd", { tree: { groups: 8, elem: "f32" } });
		const writer = await thread.open({ store: b.store, cache: false });
		const fresh = unitRows(40, b.dim, rng(77));
		await writer.upsert({ ids: Array.from({ length: 40 }, (_, i) => 100000 + i), vectors: fresh });
		writer.close();
		const gate = gated(b.store, "idx/gd/fine.f32");
		const cold = await thread.open({ store: gate.store, cache: false, coldRouting: "tree", coldBeam: 1 });
		expect(cold.revision.deltas.length).toBe(1);
		expect(groupedTableOf(cold.manifest, cold.revision.files)).not.toBeNull(); // inherited by the upsert's revision
		const full = await thread.open({ store: b.store, cache: false });
		for (let i = 0; i < 5; i++) {
			const q = fresh.subarray(i * b.dim, (i + 1) * b.dim);
			const c = await cold.query(q, { topK: 3 });
			const f = await full.query(q, { topK: 3 });
			expect(c.hits[0]).toEqual({ id: 100000 + i, score: c.hits[0].score, attrs: 0, source: "delta" });
			const fd = f.hits.find((h) => h.id === 100000 + i)!;
			expect(c.hits[0].score).toBe(fd.score); // same delta score as with the full table
			if (i === 0) expect(c.stats.coldRouting!.cells).toBeGreaterThan(0);
		}
		gate.release();
		cold.close();
		full.close();
		// no grouped table: coldRouting is ignored
		const plain = await buildGen("gp", 3, { k2: 8 });
		const ix = await thread.open({ store: plain.store, cache: false, coldRouting: "tree" });
		expect(ix.routingState).toBe("full");
		expect((await ix.query(queries(1, plain.dim, 3)[0])).stats.routing).toBe("flat");
		ix.close();
	});
});

describe("attribute filters with every routing mode", () => {
	it("filter inside the scans and widen the same way whatever table routed the query (f32, f16, i8, cold)", async () => {
		const b = await buildGen("gfr", 3, { k2: 32, n: 1200, attrs: (i) => i % 4 });
		await addRoutingRev(b.store, "gfr", { tables: ["f16", "i8"], tree: { groups: 8 } });
		const gate = gated(b.store, "idx/gfr/fine.f32");
		const opened: Record<string, Awaited<ReturnType<typeof thread.open>>> = {};
		for (const routing of ["f32", "f16", "i8"] as const) opened[routing] = await thread.open({ store: b.store, cache: false, routing });
		opened.cold = await thread.open({ store: gate.store, cache: false, coldRouting: "tree", coldBeam: 8 });
		const got: Record<string, number[][]> = {};
		const hits: Record<string, number> = {};
		for (const [mode, ix] of Object.entries(opened)) {
			got[mode] = [];
			hits[mode] = 0;
			for (const q of queries(20, b.dim, 31)) {
				const r = await ix.query(q, { topK: 5, nprobe: 8, rerank: 20, filter: { attrs: 1 } });
				expect(r.stats.routing).toBe(mode === "cold" ? "cold" : "flat");
				expect(r.stats.routingTable).toBe(mode === "cold" ? "i8" : mode);
				expect(r.params).toMatchObject({ filter: true, maxNprobe: 32 });
				expect(r.hits.length).toBe(5);
				expect(r.hits.every((h) => h.attrs === 1 && h.id % 4 === 1)).toBe(true);
				expect(r.stats.filterDropped).toBeGreaterThan(0);
				const truth = bruteForce(b.vectors, b.dim, q, b.vectors.length / b.dim).filter((j) => j % 4 === 1).slice(0, 5);
				hits[mode] += r.hits.filter((h) => truth.includes(h.id)).length;
				got[mode].push(r.hits.map((h) => h.id));
			}
		}
		gate.release();
		for (const ix of Object.values(opened)) ix.close();
		expect(got.f16).toEqual(got.f32);
		expect(hits.i8).toBeGreaterThanOrEqual(0.9 * hits.f32);
		expect(hits.cold).toBeGreaterThanOrEqual(0.9 * hits.f32);
	});
});
