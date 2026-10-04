/** Compact routing tables (fine.f16 / fine.i8): building them, opening with `routing`, fallbacks, snapshots. */
import { describe, expect, it } from "vitest";
import { asF32, buildRoutingObjects, chooseRouting, keys, manifestFiles, snapshotKeys, loadRevState, type Manifest, type MemoryBlobStore } from "../src/index.js";
import { buildGen, rng, thread, unitRows, w } from "./helpers.js";

/** Adds a revision naming `fine.f16` / `fine.i8` made from the generation's means (what publish-rev writes). */
export async function addRoutingRev(store: MemoryBlobStore, gen: string, opts: { tables?: ("f16" | "i8")[]; staleFine?: boolean } = {}): Promise<{ rev: number; manifest: Manifest }> {
	const cur = JSON.parse(new TextDecoder().decode((await store.get(keys.current()))!)) as { gen: string; rev?: number };
	const head = cur.rev ?? 0;
	const m = JSON.parse(new TextDecoder().decode((await store.get(keys.manifest(gen, head)))!)) as Manifest;
	const files = manifestFiles(m);
	const fine = asF32((await store.get(files.fine))!, files.fine);
	const built = buildRoutingObjects(w, { fineKey: files.fine, fine, dim: m.dim, tables: opts.tables });
	for (const [k, b] of built.objects) await store.put(k, b);
	const rev = head + 1;
	const routing = opts.staleFine ? { ...built.routing, fine: `idx/${gen}/fine.0.f32` } : built.routing;
	const manifest: Manifest = { ...m, rev, parentRev: head, files, routing };
	await store.put(keys.manifest(gen, rev), JSON.stringify(manifest));
	await store.put(keys.current(), JSON.stringify({ gen, rev }));
	return { rev, manifest };
}

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
			expect(b.byteLength).toBe(20 * (elem === "f16" ? 2 * dim : 4 + dim));
			expect(w.routing_row_bytes(dim, elem)).toBe(b.byteLength / 20);
			const d = w.routing_table_decode(b, dim, elem);
			for (let i = 0; i < fine.length; i++) expect(Math.abs(d[i] - fine[i])).toBeLessThan(elem === "f16" ? 1e-3 : 1 / 127);
		}
		expect(Array.from(w.routing_table_decode(w.routing_table_encode(fine, dim, "f32"), dim, "f32"))).toEqual(Array.from(fine));
		expect(() => w.routing_table_encode(fine, dim, "f8")).toThrow(/f32, f16 or i8/);
		const coarse = new Float32Array(dim).fill(0.25);
		expect(() => w.Searcher.from_table(dim, 1, 20, coarse, w.routing_table_encode(fine, dim, "f16").subarray(2), "f16")).toThrow(/expected 20 rows/);
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
		expect(manifest.routing!.i8!.bytes).toBe(16 * (4 + b.dim));
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
