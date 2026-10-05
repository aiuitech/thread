#!/usr/bin/env node
/**
 * How skewed is probe frequency? Routes a large query sample with the SDK's own routing (the wasm `Searcher` over the
 * real routing table, exactly what `executeQuery` does: normalize, `cell_dots`, `route_from_dots`) and counts how
 * often each cell is probed. Routing reads only the base (`coarse`, `fine`, `pq`, manifest, slabs table), never a
 * slab, so a sample of thousands of queries costs nothing beyond what the proxy already holds.
 *
 *   node bench/cold/hot_cells.mjs [--proxy http://127.0.0.1:9575] [--from 1000] [--count 2000] [--nprobe 34]
 *        [--eval-from 0] [--eval-count 30] [--ns 64,128,256,512,1024] [--out data/cold/open/hot-cells.json]
 *
 * Prints, for the sample: distinct cells probed, the coverage curve (share of all probes that the N most-probed
 * cells account for) and the scan-prefix bytes of those N cells (what a warm pack of them holds). Then, for the
 * held-out evaluation queries (default 0..29, the cold-start bench's set, disjoint from the sample), the hot-set hit
 * rate per query: the share of each query's probes (and of its prefix bytes) that fall inside the top N. Writes the
 * per-cell counts and the ranking to `--out` (make_warm_pack.mjs reads it).
 */
import fs from "node:fs";
import path from "node:path";
import { loadDataset, median, parseArgs, pct, root } from "./lib.mjs";
import { openRouting } from "./route.mjs";

const args = parseArgs(process.argv.slice(2));
const PROXY = args.proxy ?? "http://127.0.0.1:9575";
const FROM = Number(args.from ?? 1000);
const COUNT = Number(args.count ?? 2000);
const NPROBE = args.nprobe !== undefined ? Number(args.nprobe) : undefined;
const EVAL_FROM = Number(args["eval-from"] ?? 0);
const EVAL_COUNT = Number(args["eval-count"] ?? 30);
const NS = (args.ns ?? "64,128,256,512,1024").split(",").map(Number);
const OUT = path.resolve(root, args.out ?? "data/cold/open/hot-cells.json");

const ds = loadDataset();
if (FROM + COUNT > ds.total) throw new Error(`the dataset has ${ds.total} queries, the sample needs ${FROM + COUNT}`);
const overlap = Math.max(0, Math.min(FROM + COUNT, EVAL_FROM + EVAL_COUNT) - Math.max(FROM, EVAL_FROM));
if (overlap) throw new Error(`the sample ${FROM}..${FROM + COUNT - 1} overlaps the evaluation set ${EVAL_FROM}..${EVAL_FROM + EVAL_COUNT - 1}`);

const R = await openRouting(PROXY);
const { rev } = R;
const t = { gen: R.gen, rev: R.revNo };
const nprobe = NPROBE ?? R.defaultNprobe;
const cellsN = rev.manifest.k1 * rev.manifest.k2;
const scanLen = (c) => rev.slabs[c][0];

/** The cells `executeQuery` probes for query `i` (flat routing, the manifest default). */
const route = (i) => R.route(ds.vector(i), nprobe);

const t0 = performance.now();
const counts = new Uint32Array(cellsN);
for (let i = FROM; i < FROM + COUNT; i++) for (const c of route(i)) counts[c]++;
const routeMs = performance.now() - t0;
const ranking = Array.from(counts.keys())
	.filter((c) => counts[c] > 0)
	.sort((a, b) => counts[b] - counts[a] || a - b);
const totalProbes = COUNT * nprobe;
const ALIGN = 8;
/** Pack bytes of the scan prefixes of `cells` (8-byte aligned entries + header/footer; the directory is extra). */
const prefixBytes = (cells) => cells.reduce((a, c) => a + Math.ceil(scanLen(c) / ALIGN) * ALIGN, 16 + 32);

const curve = NS.map((n) => {
	const top = ranking.slice(0, n);
	const probes = top.reduce((a, c) => a + counts[c], 0);
	return { n, probes, coverage: probes / totalProbes, prefixBytes: prefixBytes(top), minCount: top.length ? counts[top.at(-1)] : 0 };
});

// Held-out evaluation: the bench's own queries.
const evalRows = [];
for (let i = EVAL_FROM; i < EVAL_FROM + EVAL_COUNT; i++) {
	const cells = route(i);
	const row = { query: i, cells, hits: {} };
	const bytes = cells.reduce((a, c) => a + scanLen(c), 0);
	for (const n of NS) {
		const hot = new Set(ranking.slice(0, n));
		const inHot = cells.filter((c) => hot.has(c));
		row.hits[n] = { cells: inHot.length, rate: inHot.length / cells.length, byteRate: inHot.reduce((a, c) => a + scanLen(c), 0) / bytes };
	}
	evalRows.push(row);
}
const evalSummary = NS.map((n) => {
	const rates = evalRows.map((r) => r.hits[n].rate);
	const byteRates = evalRows.map((r) => r.hits[n].byteRate);
	const cells = evalRows.map((r) => r.hits[n].cells);
	return { n, meanRate: rates.reduce((a, b) => a + b, 0) / rates.length, medianRate: median(rates), minRate: Math.min(...rates), maxRate: Math.max(...rates), p10Rate: pct(rates, 0.1), meanByteRate: byteRates.reduce((a, b) => a + b, 0) / byteRates.length, medianCells: median(cells), minCells: Math.min(...cells), maxCells: Math.max(...cells) };
});

const MB = 1048576;
const pctS = (x) => (100 * x).toFixed(1) + "%";
console.log(`${t.gen} rev ${t.rev}: ${cellsN} cells, nprobe ${nprobe}; sample queries ${FROM}..${FROM + COUNT - 1} (${totalProbes} probes, routed in ${routeMs.toFixed(0)} ms)`);
console.log(`distinct cells probed: ${ranking.length} of ${cellsN} (${pctS(ranking.length / cellsN)}); most-probed cell ${counts[ranking[0]]} times (${pctS(counts[ranking[0]] / COUNT)} of queries); median probed cell ${median(ranking.map((c) => counts[c]))}`);
console.log(`uniform reference: N cells would cover N/${cellsN} of probes (128: ${pctS(128 / cellsN)}, 512: ${pctS(512 / cellsN)})`);
console.log("\ncoverage curve (sample)");
console.log("     N   probes  coverage  min count  prefix MB");
for (const r of curve) console.log(`${String(r.n).padStart(6)}  ${String(r.probes).padStart(7)}  ${pctS(r.coverage).padStart(8)}  ${String(r.minCount).padStart(9)}  ${(r.prefixBytes / MB).toFixed(1).padStart(9)}`);
console.log(`\nhot-set hit rate on held-out queries ${EVAL_FROM}..${EVAL_FROM + EVAL_COUNT - 1} (share of each query's ${nprobe} probes inside the sample's top N)`);
console.log("     N   mean  median    p10    min    max  probes in set med [min, max]  byte share mean");
for (const r of evalSummary) console.log(`${String(r.n).padStart(6)}  ${pctS(r.meanRate).padStart(5)}  ${pctS(r.medianRate).padStart(6)}  ${pctS(r.p10Rate).padStart(5)}  ${pctS(r.minRate).padStart(5)}  ${pctS(r.maxRate).padStart(5)}  ${String(r.medianCells).padStart(17)} [${r.minCells}, ${r.maxCells}]  ${pctS(r.meanByteRate).padStart(15)}`);

fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(
	OUT,
	JSON.stringify({ gen: t.gen, rev: t.rev, nprobe, sample: { from: FROM, count: COUNT, probes: totalProbes }, cells: cellsN, distinct: ranking.length, curve, eval: { from: EVAL_FROM, count: EVAL_COUNT, summary: evalSummary, perQuery: evalRows.map((r) => ({ query: r.query, hits: r.hits })) }, ranking, counts: ranking.map((c) => counts[c]) }) + "\n",
);
console.log(`\nwrote ${path.relative(root, OUT)}`);
