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
 *
 * Layouts: each `--pack` (its real directory), plus virtual ones computed from the slabs table: `key` (cell order,
 * what packIndex writes), `tree` (make_pack.mjs --order tree) and `tree-split` (a layout the pack format cannot
 * express today: within each tree group, all scan prefixes first, then all rerank tiers + payloads).
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

/** Merged phase-1 reads of one query in one layout. */
function plan(at, cells, gap) {
	const items = cells.map((c) => ({ off: at.get(c).off, len: scanLen(c) })).sort((a, b) => a.off - b.off);
	let reqs = 0;
	let bytes = 0;
	let end = -Infinity;
	let start = 0;
	for (const it of items) {
		if (it.off - end <= gap) end = Math.max(end, it.off + it.len);
		else {
			if (reqs) bytes += end - start;
			reqs++;
			start = it.off;
			end = it.off + it.len;
		}
	}
	bytes += end - start;
	return { reqs, bytes };
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
	console.log("        gap   requests med (mean)   MB read med (mean)   x prefix bytes");
	for (const gap of GAPS) {
		const ps = queries.map((q) => plan(L.at, q.cells, gap));
		const reqs = ps.map((p) => p.reqs);
		const bytes = ps.map((p) => p.bytes);
		const ratio = ps.map((p, i) => p.bytes / prefixBytes[i]);
		const r = { gap, reqsMedian: median(reqs), reqsMean: reqs.reduce((a, b) => a + b, 0) / reqs.length, bytesMedian: median(bytes), bytesMean: bytes.reduce((a, b) => a + b, 0) / bytes.length, ratioMedian: median(ratio) };
		row.byGap.push(r);
		console.log(`${String(gap).padStart(11)}   ${String(r.reqsMedian).padStart(12)} (${r.reqsMean.toFixed(1)})   ${(r.bytesMedian / MB).toFixed(2).padStart(12)} (${(r.bytesMean / MB).toFixed(2)})   ${r.ratioMedian.toFixed(2).padStart(14)}`);
	}
	out.layouts.push(row);
}
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify(out) + "\n");
console.log(`\nwrote ${path.relative(root, OUT)}`);
