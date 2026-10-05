#!/usr/bin/env node
/**
 * Builds a warm pack (sdk/ts/src/warm.ts: the scan prefixes of the most-probed slabs, one object) for the revision the
 * proxy pins, from the probe counts hot_cells.mjs wrote and the proxy's disk cache, WITHOUT downloading slabs: like
 * make_pack.mjs, a slab the cache does not hold becomes a hole of its exact prefix length in a sparse file, so the
 * pack's size, directory and transfer time are real, and every prefix a query of the warmed set (0..29) reads is
 * real. A query outside that set that probes a hole cell reads zeros and fails on the slab header.
 *
 *   node bench/cold/make_warm_pack.mjs [--hot data/cold/open/hot-cells.json] [--n 256 | --budget-mb 32]
 *        [--rank count|density] [--cache-dir data/cold/objects] [--proxy http://127.0.0.1:9575]
 *        [--out data/cold/open-overlay/warm/g4-hot256.threadpack]
 *
 * `--rank count` takes the N most-probed cells; `--rank density` ranks by probes per prefix byte (more probes covered
 * per MB of pack; pair it with `--budget-mb`). Serve the pack with `fake_s3.mjs --overlay data/cold/open-overlay` and
 * open with `warmPack: "warm/g4-hot256.threadpack"` (cold_bench.mjs `--warm-pack`).
 *
 * In production the counts would come from query logs instead of a query sample: every query result carries the
 * probed cells (`stats.cells`), a Worker can aggregate them per generation, and `publish` / `publish-rev` would
 * write `idx/{gen}/warm.{rev}.threadpack` with `writeWarmPack(store, rev, topCells, sink)` right after the slabs of
 * that revision (docs/cold-start.md, "Warm packs").
 */
import fs from "node:fs";
import path from "node:path";
import { parseArgs, root } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const HOT = path.resolve(root, args.hot ?? "data/cold/open/hot-cells.json");
const RANK = args.rank ?? "count";
if (!["count", "density"].includes(RANK)) throw new Error("--rank must be count or density");
const N = args.n !== undefined ? Number(args.n) : undefined;
const BUDGET = args["budget-mb"] !== undefined ? Number(args["budget-mb"]) * 1048576 : undefined;
if ((N === undefined) === (BUDGET === undefined)) throw new Error("pass exactly one of --n and --budget-mb");
const CACHE = path.resolve(root, args["cache-dir"] ?? "data/cold/objects");
const PROXY = args.proxy ?? "http://127.0.0.1:9575";
const name = `g4-${RANK === "count" ? "hot" : "dense"}${N ?? `${args["budget-mb"]}mb`}`;
const OUT = path.resolve(root, args.out ?? `data/cold/open-overlay/warm/${name}.threadpack`);
const dist = path.join(root, "sdk/ts/dist");
const sdk = await import(path.join(dist, "index.js"));
const { FileBlobStore } = await import(path.join(dist, "node.js"));

const hot = JSON.parse(fs.readFileSync(HOT, "utf8"));
const proxy = new sdk.S3BlobStore({ endpoint: PROXY, bucket: "vib-bench", region: "auto", accessKeyId: "fake", secretAccessKey: "fake" });
const t = await sdk.resolveTarget(proxy);
const rev = await sdk.loadRevState(proxy, t.gen, t.rev);
if (hot.gen !== t.gen || hot.rev !== t.rev) throw new Error(`${HOT} counts ${hot.gen} rev ${hot.rev}, the proxy pins ${t.gen} rev ${t.rev}`);
const count = new Map(hot.ranking.map((c, i) => [c, hot.counts[i]]));
const scanLen = (c) => rev.slabs[c][0];
let ranking = hot.ranking;
if (RANK === "density") ranking = [...ranking].sort((a, b) => count.get(b) / scanLen(b) - count.get(a) / scanLen(a) || a - b);
let cells;
if (N !== undefined) cells = ranking.slice(0, N);
else {
	cells = [];
	let bytes = 0;
	for (const c of ranking) {
		const b = Math.ceil(scanLen(c) / 8) * 8;
		if (bytes + b > BUDGET) break;
		bytes += b;
		cells.push(c);
	}
}

// Prefixes from the proxy's cache; a slab it does not hold is a hole of the prefix's length.
const cache = new FileBlobStore(CACHE);
const holes = new WeakSet();
const holeKeys = [];
const src = {
	async get(key, range) {
		const b = await cache.get(key, range);
		if (b) return b;
		const z = new Uint8Array(range.length);
		holes.add(z);
		holeKeys.push(key);
		return z;
	},
	head: (k) => cache.head(k),
};
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.rmSync(OUT, { force: true });
const fd = fs.openSync(OUT, "w");
let pos = 0;
const sink = {
	write(b) {
		if (!holes.has(b)) fs.writeSync(fd, b, 0, b.byteLength, pos);
		pos += b.byteLength;
	},
};
const sample = hot.sample;
const res = await sdk.writeWarmPack(src, rev, cells, sink, { concurrency: 1, meta: { source: `query sample ${sample.from}..${sample.from + sample.count - 1}, nprobe ${hot.nprobe}`, rank: RANK } });
fs.ftruncateSync(fd, res.bytes);
fs.closeSync(fd);

// Check: it parses as a warm pack, and real prefixes read back identical to the slabs' first bytes.
const parsed = sdk.parseWarmPack(new Uint8Array(fs.readFileSync(OUT)));
const holeSet = new Set(holeKeys);
let checked = 0;
for (const e of parsed.entries) {
	if (holeSet.has(e.key)) continue;
	const want = await cache.get(e.key, { offset: 0, length: e.bytes.byteLength });
	if (Buffer.compare(Buffer.from(want), Buffer.from(e.bytes)) !== 0) throw new Error(`${e.key} reads back differently`);
	checked++;
}
const probes = cells.reduce((a, c) => a + count.get(c), 0);
const st = fs.statSync(OUT);
const info = {
	pack: path.relative(root, OUT),
	key: path.relative(path.resolve(root, "data/cold/open-overlay"), OUT),
	gen: t.gen,
	rev: t.rev,
	rank: RANK,
	cells: cells.length,
	bytes: res.bytes,
	prefixBytes: res.prefixBytes,
	onDiskBytes: st.blocks * 512,
	sampleCoverage: probes / sample.probes,
	realPrefixes: checked,
	holePrefixes: holeKeys.length,
};
console.log(JSON.stringify(info, null, 1));
fs.writeFileSync(OUT.replace(/\.threadpack$/, ".json"), JSON.stringify({ ...info, cellList: cells }) + "\n");
