#!/usr/bin/env node
/**
 * How close together are a query's probed slabs in a pack, and what would merging their phase-1 prefix reads into
 * fewer ranges cost? Routes the held-out queries (routing only: no slab is read), maps each probed cell to its slab's
 * place in a pack layout and plans the merged reads `coalesceAcrossSlabs` would issue: prefixes sorted by pack
 * offset, a range grows while the gap from its end to the next prefix is at most `gap` bytes (the bytes in between,
 * the previous slab's rerank tier and payload and any unprobed slab, are read and thrown away).
 *
 *   node bench/cold/pack_adjacency.mjs [--proxy http://127.0.0.1:9575] [--from 0] [--count 30]
 *        [--pack data/cold/overlay/packs/g4-cold.threadpack,data/cold/open-overlay/packs/g4-cold-tree.threadpack]
 *        [--gaps 0,65536,262144,1048576,4194304] [--out data/cold/open/pack-adjacency.json]
 *        [--model rtt25:25:200,rtt60:60:100] [--inflight 12]
 *
 * Layouts: each `--pack` (its real directory), plus virtual ones computed from the slabs table: `key` (cell order,
 * what packIndex writes), `tree` (make_pack.mjs --order tree) and `tree-split` (a layout the pack format cannot
 * express today: within each tree group, all scan prefixes first, then all rerank tiers + payloads).
 *
 * `--model name:rtt:mbps` adds a modelled phase-1 time per layout and gap (median over the queries): the merged reads
 * issued in routing order of their first member, `--inflight` at a time, each taking rtt + bytes at mbps (the proxy's
 * per-response model, warm connections, no scan CPU). It is a model, for layouts that cannot be measured yet; check
 * it against a measured layout before trusting it.
 */
import fs from "node:fs";
import path from "node:path";
import { loadDataset, median, parseArgs, root } from "./lib.mjs";
import { openRouting } from "./route.mjs";
import { parseTree, treeOrder } from "./tree_order.mjs";

const args = parseArgs(process.argv.slice(2));
const PROXY = args.proxy ?? "http://127.0.0.1:9575";
const FROM = Number(args.from ?? 0);
const COUNT = Number(args.count ?? 30);
const PACKS = (args.pack ?? "data/cold/overlay/packs/g4-cold.threadpack").split(",").filter(Boolean);
const GAPS = (args.gaps ?? "0,65536,262144,1048576,4194304").split(",").map(Number);
const OUT = path.resolve(root, args.out ?? "data/cold/open/pack-adjacency.json");
const MODELS = (args.model ?? "rtt25:25:200,rtt60:60:100").split(",").filter(Boolean).map((m) => {
	const [name, rtt, mbps] = m.split(":");
	return { name, rtt: Number(rtt), mbps: Number(mbps) };
});
const INFLIGHT = Number(args.inflight ?? 12);

const R = await openRouting(PROXY);
const { sdk, rev, store } = R;
const { FileBlobStore } = await import(path.join(root, "sdk/ts/dist/node.js"));
const k = rev.manifest.k1 * rev.manifest.k2;
const slabKey = (c) => sdk.keys.slab(rev.gen, c, sdk.slabRevOf(rev.slabs[c]));
const scanLen = (c) => rev.slabs[c][0];
const size = (c) => rev.slabs[c][3];
const ds = loadDataset();
const queries = Array.from({ length: COUNT }, (_, i) => ({ q: FROM + i, cells: R.route(ds.vector(FROM + i)) }));

// Layouts: cell -> {off: prefix offset, idx: position in the order}.
const align = (x) => Math.ceil(x / 8) * 8;
function wholeLayout(order) {
	const at = new Map();
	let pos = 0;
	order.forEach((c, idx) => {
		pos = align(pos);
		at.set(c, { off: pos, idx });
		pos += size(c);
	});
	return at;
}
const layouts = [];
for (const p of PACKS) {
	const file = path.resolve(root, p);
	if (!fs.existsSync(file)) {
		console.log(`(skipping ${p}: not found)`);
		continue;
	}
	const ps = await sdk.PackedBlobStore.open(new FileBlobStore(path.dirname(file)), path.basename(file));
	const slabEntries = [...ps.directory.entries].filter(([key]) => key.includes("/slab/"));
	const idxOf = new Map(slabEntries.map(([key], i) => [key, i]));
	const at = new Map();
	for (let c = 0; c < k; c++) {
		const e = ps.directory.entries.get(slabKey(c));
		if (!e) throw new Error(`${p} has no ${slabKey(c)}`);
		at.set(c, { off: e.offset, idx: idxOf.get(slabKey(c)) });
	}
	layouts.push({ name: `pack:${path.basename(p)}`, at });
}
const cellOrder = Array.from({ length: k }, (_, c) => c);
layouts.push({ name: "key", at: wholeLayout(cellOrder) });
const tree = parseTree(await store.get(rev.manifest.tree.key));
const fine = sdk.asF32(await store.get(rev.files.fine), "fine");
const { order: tOrder, group } = treeOrder(tree, fine, k);
layouts.push({ name: "tree", at: wholeLayout(tOrder) });
{
	// tree-split: per group, prefixes then tails.
	const at = new Map();
	let pos = 0;
	let i = 0;
	while (i < tOrder.length) {
		let j = i;
		while (j < tOrder.length && group[tOrder[j]] === group[tOrder[i]]) j++;
		const g = tOrder.slice(i, j);
		for (const c of g) {
			pos = align(pos);
			at.set(c, { off: pos, idx: at.size });
			pos += scanLen(c);
		}
		for (const c of g) pos = align(pos) + size(c) - scanLen(c);
		i = j;
	}
	layouts.push({ name: "tree-split", at });
}

/** Merged phase-1 reads of one query in one layout: count, bytes, and each read's size in issue order. */
function plan(at, cells, gap) {
	const rank = new Map(cells.map((c, i) => [c, i]));
	const items = cells.map((c) => ({ off: at.get(c).off, len: scanLen(c), rank: rank.get(c) })).sort((a, b) => a.off - b.off);
	const reads = [];
	let cur = null;
	for (const it of items) {
		if (cur && it.off - cur.end <= gap) {
			cur.end = Math.max(cur.end, it.off + it.len);
			cur.first = Math.min(cur.first, it.rank);
		} else {
			cur = { start: it.off, end: it.off + it.len, first: it.rank };
			reads.push(cur);
		}
	}
	reads.sort((a, b) => a.first - b.first);
	return { reqs: reads.length, bytes: reads.reduce((a, r) => a + r.end - r.start, 0), sizes: reads.map((r) => r.end - r.start) };
}

/** Makespan (ms) of reads issued in order, `INFLIGHT` at a time, each rtt + size at mbps. */
function makespan(sizes, m) {
	const slots = new Array(INFLIGHT).fill(0);
	let end = 0;
	for (const b of sizes) {
		let i = 0;
		for (let j = 1; j < slots.length; j++) if (slots[j] < slots[i]) i = j;
		slots[i] += m.rtt + (b * 8) / (m.mbps * 1000);
		end = Math.max(end, slots[i]);
	}
	return end;
}

const MB = 1048576;
const out = { from: FROM, count: COUNT, nprobe: R.defaultNprobe, gaps: GAPS, layouts: [] };
const groupsPerQuery = queries.map((q) => new Set(q.cells.map((c) => group[c])).size);
const prefixBytes = queries.map((q) => q.cells.reduce((a, c) => a + scanLen(c), 0));
console.log(`queries ${FROM}..${FROM + COUNT - 1}, nprobe ${R.defaultNprobe}: tree groups per query median ${median(groupsPerQuery)} [${Math.min(...groupsPerQuery)}, ${Math.max(...groupsPerQuery)}] of ${tree.groups}; prefix MB per query median ${(median(prefixBytes) / MB).toFixed(2)}`);
for (const L of layouts) {
	const adj = queries.map((q) => {
		const idx = q.cells.map((c) => L.at.get(c).idx).sort((a, b) => a - b);
		let n = 0;
		for (let i = 1; i < idx.length; i++) if (idx[i] === idx[i - 1] + 1) n++;
		return n;
	});
	const row = { name: L.name, adjacentPairs: { median: median(adj), mean: adj.reduce((a, b) => a + b, 0) / adj.length, max: Math.max(...adj), queriesWithAny: adj.filter((x) => x > 0).length }, byGap: [] };
	console.log(`\n${L.name}: probed slabs adjacent in the pack (pairs per query) median ${row.adjacentPairs.median}, mean ${row.adjacentPairs.mean.toFixed(2)}, max ${row.adjacentPairs.max}; queries with any: ${row.adjacentPairs.queriesWithAny}/${COUNT}`);
	console.log(`        gap   requests med (mean)   MB read med (mean)   x prefix bytes${MODELS.map((m) => `   model ${m.name} ms`).join("")}`);
	for (const gap of GAPS) {
		const ps = queries.map((q) => plan(L.at, q.cells, gap));
		const reqs = ps.map((p) => p.reqs);
		const bytes = ps.map((p) => p.bytes);
		const ratio = ps.map((p, i) => p.bytes / prefixBytes[i]);
		const model = Object.fromEntries(MODELS.map((m) => [m.name, median(ps.map((p) => makespan(p.sizes, m)))]));
		const r = { gap, reqsMedian: median(reqs), reqsMean: reqs.reduce((a, b) => a + b, 0) / reqs.length, bytesMedian: median(bytes), bytesMean: bytes.reduce((a, b) => a + b, 0) / bytes.length, ratioMedian: median(ratio), model };
		row.byGap.push(r);
		console.log(`${String(gap).padStart(11)}   ${String(r.reqsMedian).padStart(12)} (${r.reqsMean.toFixed(1)})   ${(r.bytesMedian / MB).toFixed(2).padStart(12)} (${(r.bytesMean / MB).toFixed(2)})   ${r.ratioMedian.toFixed(2).padStart(14)}${MODELS.map((m) => String(model[m.name].toFixed(0)).padStart(10 + m.name.length)).join("")}`);
	}
	out.layouts.push(row);
}
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify(out) + "\n");
console.log(`\nwrote ${path.relative(root, OUT)}`);
