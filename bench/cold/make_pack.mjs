#!/usr/bin/env node
/**
 * Builds a pack of the revision fake_s3.mjs's disk cache pinned, for `cold_bench --store pack`, WITHOUT downloading
 * the whole index: the pack has the exact layout `packIndex` writes for that revision (`idx/current.json`, the base
 * objects a cold open reads with `meta.base` = their range, then the rest of `snapshotKeys` in key order, 8-byte
 * aligned, the JSON directory, the footer; `--layout keys`: the older layout, everything after current.json in key
 * order, no base range), but the slabs the cache does not hold
 * are left as holes of the right length in a sparse file. So the pack's size, directory, every offset and the open
 * cost (head + tail + directory reads) are those of the full pack, and every slab the warmed query set touches is
 * real. A query outside that set reads zeros and fails on the slab header: run warm_proxy.mjs for the set first.
 *
 *   node bench/cold/make_pack.mjs [--cache-dir data/cold/objects] [--out data/cold/overlay/packs/g4-cold.threadpack]
 *        [--proxy http://127.0.0.1:9555] [--overlay DIR] [--layout base|keys]
 *
 * `--overlay DIR` is read before the cache (e.g. data/cold/base-overlay, whose current.json names the revision with
 * the routing objects of make_routing.mjs).
 *
 * Objects other than slabs that are missing from the cache (e.g. the centroid tree, which the SDKs never read) are
 * fetched through the proxy (read-only, it caches them). Serve the pack with `fake_s3.mjs --overlay data/cold/overlay`.
 */
import fs from "node:fs";
import path from "node:path";
import { parseArgs, root } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const CACHE = path.resolve(root, args["cache-dir"] ?? "data/cold/objects");
const OUT = path.resolve(root, args.out ?? "data/cold/overlay/packs/g4-cold.threadpack");
const PROXY = args.proxy ?? "http://127.0.0.1:9555";
const dist = path.join(root, "sdk/ts/dist");
const sdk = await import(path.join(dist, "index.js"));
const { FileBlobStore } = await import(path.join(dist, "node.js"));

const LAYOUT = args.layout ?? "base";
const cacheDir = new FileBlobStore(CACHE);
const overlay = args.overlay ? new FileBlobStore(path.resolve(root, args.overlay)) : null;
/** The overlay (when given), then the proxy's cache. */
const cache = overlay
	? { get: async (k, r) => (await overlay.get(k, r)) ?? cacheDir.get(k, r), head: async (k) => (await overlay.head(k)) ?? cacheDir.head(k) }
	: cacheDir;
const t = await sdk.resolveTarget(cache);
const rev = await sdk.loadRevState(cache, t.gen, t.rev);
const all = sdk.snapshotKeys(rev);
const inSnapshot = new Set(all);
const base = LAYOUT === "base" ? sdk.baseKeys(rev, "base", "auto").filter((k) => inSnapshot.has(k)) : [];
const baseSet = new Set(base);
const keys = [...base, ...all.filter((k) => !baseSet.has(k))];
const slabSize = new Map();
for (let cell = 0; cell < rev.slabs.length; cell++) slabSize.set(sdk.keys.slab(rev.gen, cell, sdk.slabRevOf(rev.slabs[cell])), rev.slabs[cell][3]);

const proxy = new sdk.S3BlobStore({ endpoint: PROXY, bucket: "vib-bench", accessKeyId: "fake", secretAccessKey: "fake" });
const holes = new WeakSet();
let real = 0;
let realBytes = 0;
let holeBytes = 0;
const holeKeys = [];

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
const w = new sdk.PackWriter(sink, { gen: t.gen, rev: t.rev });
await w.add("idx/current.json", new TextEncoder().encode(JSON.stringify({ gen: t.gen, rev: t.rev })));
for (const k of keys) {
	let b = await cache.get(k);
	if (!b && slabSize.has(k)) {
		b = new Uint8Array(slabSize.get(k));
		holes.add(b);
		holeBytes += b.byteLength;
		holeKeys.push(k);
	} else if (!b) {
		b = await proxy.get(k);
		if (!b) throw new Error(`${k} is missing from the cache and the proxy`);
	}
	if (!holes.has(b)) {
		real++;
		realBytes += b.byteLength;
	}
	await w.add(k, b);
	if (base.length && k === base[base.length - 1]) w.setMeta("base", [sdk.PACK_HEADER_BYTES, w.written - sdk.PACK_HEADER_BYTES]);
}
const { bytes, count } = await w.finish();
fs.ftruncateSync(fd, bytes);
fs.closeSync(fd);

// Check: the pack opens, its directory is the one packIndex would write, and real entries read back identical.
const ps = await sdk.PackedBlobStore.open(new FileBlobStore(path.dirname(OUT)), path.basename(OUT));
const order = [...ps.directory.entries.keys()];
if (order.join("\n") !== ["idx/current.json", ...keys].join("\n")) throw new Error("pack entry order differs from packIndex's");
for (const k of [rev.files.fine, rev.files.slabs, ...keys.filter((k) => slabSize.has(k) && !holeKeys.includes(k)).slice(0, 5)]) {
	const a = await ps.get(k);
	const b = await cache.get(k);
	if (Buffer.compare(Buffer.from(a), Buffer.from(b)) !== 0) throw new Error(`${k} reads back differently from the pack`);
}
// The directory starts right after the last entry (no padding) and ends at the footer.
const dirLen = bytes - sdk.PACK_FOOTER_BYTES - [...ps.directory.entries.values()].reduce((m, e) => Math.max(m, e.offset + e.length), 0);
const st = fs.statSync(OUT);
console.log(
	JSON.stringify(
		{
			pack: path.relative(root, OUT),
			gen: t.gen,
			rev: t.rev,
			entries: count,
			bytes,
			onDiskBytes: st.blocks * 512,
			directoryBytes: dirLen,
			directoryInTail: dirLen + 32 <= sdk.PACK_TAIL_READ,
			realObjects: real,
			realBytes,
			holeSlabs: holeKeys.length,
			holeBytes,
			layout: LAYOUT,
			base: ps.directory.meta.base ?? null,
			firstEntries: order.slice(0, 8),
			lastEntries: order.slice(-3),
		},
		null,
		1,
	),
);
