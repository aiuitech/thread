#!/usr/bin/env node
/**
 * Routing agreement of the compact / cold routing modes with flat f32 routing, offline (no slab reads): for each
 * query, the nprobe cells each mode picks against the cells flat routing over fine.f32 picks.
 *
 *   node bench/cold/routing_agreement.mjs [--base data/cold/g4-base] [--overlay data/cold/base-overlay] [--gen g4]
 *        [--rev 180] [--queries 2000] [--nprobe 34] [--beams 8,16,32] [--json data/cold/routing-agreement.json]
 *
 * Modes: flat over fine.f16 / fine.i8 (decoded, `Searcher.from_table`); beam routing through the centroid tree over
 * the f32 means (`Searcher.route_beam`) and over each tree-grouped table found (`ColdRouter` with every group loaded,
 * which is what a cold query computes once its beam's groups are in). Reports the mean probe-set overlap
 * |cells ∩ flat-f32 cells| / nprobe, the share of queries with the identical set, the first cell agreeing, and for
 * the compact tables the error they put on a delta entry's base score (q·mean, f32 vs decoded). For the cold modes
 * it also reports the means bytes a first query reads (its beam's groups) on top of the tree.
 */
import fs from "node:fs";
import path from "node:path";
import { loadDataset, parseArgs, root } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const BASE = path.resolve(root, args.base ?? "data/cold/g4-base");
const OVERLAY = path.resolve(root, args.overlay ?? "data/cold/base-overlay");
const GEN = args.gen ?? "g4";
const NQ = Number(args.queries ?? 2000);
const NPROBE = Number(args.nprobe ?? 34);
const BEAMS = (args.beams ?? "8,16,32").split(",").map(Number);
const dist = path.join(root, "sdk/ts/dist");
const sdk = await import(path.join(dist, "index.js"));
const thread = await sdk.Thread.init({ wasmModule: new WebAssembly.Module(fs.readFileSync(path.join(dist, "wasm/thread_bg.wasm"))) });
const w = thread.wasm;
const read = (key) => {
	for (const d of [OVERLAY, BASE]) if (fs.existsSync(path.join(d, key))) return new Uint8Array(fs.readFileSync(path.join(d, key)));
	throw new Error(`missing ${key} in ${OVERLAY} and ${BASE}`);
};
const rev = Number(args.rev ?? JSON.parse(new TextDecoder().decode(read("idx/current.json"))).rev);
const m = JSON.parse(new TextDecoder().decode(read(sdk.keys.manifest(GEN, rev))));
const files = sdk.manifestFiles(m);
const k = m.k1 * m.k2;
const coarse = sdk.asF32(read(files.coarse), files.coarse);
const fine = sdk.asF32(read(files.fine), files.fine);
const ds = loadDataset();
const queries = Array.from({ length: NQ }, (_, i) => w.normalize_vector(ds.vector(i)));

const ref = new w.Searcher(m.dim, m.k1, m.k2, coarse, fine);
const tree = read(m.tree.key);
ref.set_tree(tree);
const refCells = queries.map((q) => Array.from(ref.route_flat(q, NPROBE)));

const modes = [];
for (const elem of ["f16", "i8"]) {
	const t = m.routing?.[elem];
	if (!t) continue;
	const bytes = read(t.key);
	const s = w.Searcher.from_table(m.dim, m.k1, m.k2, coarse, bytes, elem);
	const decoded = w.routing_table_decode(bytes, m.dim, elem);
	modes.push({ name: `flat ${elem}`, bytes: bytes.byteLength, route: (q) => Array.from(s.route_flat(q, NPROBE)), decoded });
}
for (const beam of BEAMS) modes.push({ name: `tree beam ${beam} (f32 means)`, route: (q) => Array.from(ref.route_beam(q, beam, NPROBE)), beam });
const grouped = fs
	.readdirSync(path.join(OVERLAY, "idx", GEN))
	.filter((f) => f.startsWith(`fine.grouped.${rev}.`))
	.map((f) => `idx/${GEN}/${f}`);
for (const key of grouped) {
	const bytes = read(key);
	const [code, , , groups, , dataOff, rowBytes] = w.grouped_table_info(bytes);
	const elem = ["f32", "f16", "i8"][code];
	const cr = new w.ColdRouter(tree, elem, bytes.subarray(0, dataOff));
	cr.add_rows(0, bytes.subarray(dataOff));
	for (const beam of BEAMS)
		modes.push({
			name: `cold beam ${beam} (${elem} rows)`,
			route: (q) => Array.from(cr.route_beam(q, beam, NPROBE)),
			beam,
			coldBytes: (q) => Array.from(cr.top_groups(q, beam)).reduce((a, g) => a + cr.group_rows(g)[1] * rowBytes, 0),
			groups,
		});
}

const out = { gen: GEN, rev, queries: NQ, nprobe: NPROBE, k, dim: m.dim, treeBytes: tree.byteLength, modes: [] };
console.log(`${GEN} rev ${rev}: ${k} cells x ${m.dim}, ${NQ} queries, nprobe ${NPROBE}; reference = flat over ${files.fine}`);
for (const mode of modes) {
	let overlap = 0;
	let identical = 0;
	let first = 0;
	let coldBytes = 0;
	let maxErr = 0;
	let sumErr = 0;
	let nErr = 0;
	queries.forEach((q, i) => {
		const got = mode.route(q);
		const r = refCells[i];
		const set = new Set(r);
		const inter = got.filter((c) => set.has(c)).length;
		overlap += inter / NPROBE;
		if (inter === NPROBE) identical++;
		if (got[0] === r[0]) first++;
		if (mode.coldBytes) coldBytes += mode.coldBytes(q);
		if (mode.decoded) {
			// delta base score error q·mean over the reference cells
			for (const c of r) {
				let a = 0;
				let b = 0;
				for (let d = 0; d < m.dim; d++) {
					a += q[d] * fine[c * m.dim + d];
					b += q[d] * mode.decoded[c * m.dim + d];
				}
				const e = Math.abs(a - b);
				maxErr = Math.max(maxErr, e);
				sumErr += e;
				nErr++;
			}
		}
	});
	const row = { mode: mode.name, overlap: overlap / NQ, identical: identical / NQ, firstCell: first / NQ, ...(mode.bytes ? { tableBytes: mode.bytes } : {}), ...(mode.coldBytes ? { firstQueryMeansBytes: Math.round(coldBytes / NQ) } : {}), ...(nErr ? { deltaScoreErrMean: sumErr / nErr, deltaScoreErrMax: maxErr } : {}) };
	out.modes.push(row);
	console.log(`${mode.name.padEnd(28)} overlap ${(row.overlap * 100).toFixed(2)}%  identical ${(row.identical * 100).toFixed(1)}%  first cell ${(row.firstCell * 100).toFixed(1)}%` + (row.firstQueryMeansBytes ? `  first-query means ${(row.firstQueryMeansBytes / 1e6).toFixed(2)} MB` : "") + (nErr ? `  q·mean err mean ${row.deltaScoreErrMean.toExponential(2)} max ${row.deltaScoreErrMax.toExponential(2)}` : ""));
}
if (args.json) fs.writeFileSync(path.resolve(root, args.json), JSON.stringify(out, null, 1));
