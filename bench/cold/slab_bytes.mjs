#!/usr/bin/env node
/**
 * Per-slab byte accounting of the pinned revision (from the proxy's disk cache): scan prefix (header, mean, entry
 * table, PQ codes) vs rerank tier vs whole slab, for every cell and for the slabs a bench run probed, and what one
 * query moves when it reads prefixes (+ rerank records) vs whole slabs.
 *
 *   node bench/cold/slab_bytes.mjs [--cache-dir data/cold/objects] [data/cold/baseline-ts-proxy-rtt25.jsonl]
 */
import path from "node:path";
import { median, parseArgs, pct, readJsonl, root } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const dist = path.join(root, "sdk/ts/dist");
const sdk = await import(path.join(dist, "index.js"));
const { FileBlobStore } = await import(path.join(dist, "node.js"));
const cache = new FileBlobStore(path.resolve(root, args["cache-dir"] ?? "data/cold/objects"));
const t = await sdk.resolveTarget(cache);
const rev = await sdk.loadRevState(cache, t.gen, t.rev);
const rec = 4 + rev.manifest.dim;
const rows = rev.slabs.map(([scanLen, rerankOff, count, bytes]) => ({ scanLen, rerankOff, count, bytes, rerank: count * rec }));
const KB = (x) => (x / 1024).toFixed(1);
const MB = (x) => (x / 1048576).toFixed(2);
const describe = (name, rs) => {
	const f = (get) => {
		const xs = rs.map(get);
		return `median ${KB(median(xs))} KB  p90 ${KB(pct(xs, 0.9))} KB  max ${KB(Math.max(...xs))} KB  total ${MB(xs.reduce((a, b) => a + b, 0))} MB`;
	};
	const scan = rs.reduce((a, r) => a + r.scanLen, 0);
	const whole = rs.reduce((a, r) => a + r.bytes, 0);
	console.log(`${name}: ${rs.length} slabs, entries median ${median(rs.map((r) => r.count))} p90 ${pct(rs.map((r) => r.count), 0.9)} max ${Math.max(...rs.map((r) => r.count))}`);
	console.log(`  scan prefix  ${f((r) => r.scanLen)}`);
	console.log(`  rerank tier  ${f((r) => r.rerank)}`);
	console.log(`  whole slab   ${f((r) => r.bytes)}`);
	console.log(`  prefix / whole = ${((100 * scan) / whole).toFixed(1)}% (${(whole / scan).toFixed(2)}x more bytes to read slabs whole)`);
};
console.log(`${t.gen} rev ${t.rev}: ${rows.length} cells, dim ${rev.manifest.dim}, rerank record ${rec} B, PQ m ${rev.manifest.pq?.m}`);
describe("all cells", rows);
const run = args._?.[0];
if (run) {
	// The bench rows carry the query's own counters: scan-prefix bytes, slab bytes transferred (whole slabs when a
	// cache sits in front, else = prefixes) and rerank-record bytes.
	const cold = readJsonl(path.resolve(root, run)).filter((r) => r.kind === "trial" && r.scenario !== "warm");
	const m = (get) => MB(median(cold.map(get)));
	console.log(`\nper cold query in ${path.basename(run)} (${cold.length} queries, medians): scan prefixes ${m((r) => r.stats.scanBytes)} MB, slab bytes transferred ${m((r) => r.stats.slabBytes)} MB, rerank records ${m((r) => r.stats.rerankBytes)} MB`);
}
try {
	const warm = (await import("node:fs")).readFileSync(path.join(root, "data/cold/warm-keys.json"), "utf8");
	const { slabs, from, count } = JSON.parse(warm);
	const set = new Set(slabs);
	const touched = rows.filter((_, cell) => set.has(sdk.keys.slab(rev.gen, cell, sdk.slabRevOf(rev.slabs[cell]))));
	console.log("");
	describe(`slabs probed by queries ${from}..${from + count - 1}`, touched);
} catch {}
