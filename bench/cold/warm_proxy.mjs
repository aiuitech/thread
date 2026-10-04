#!/usr/bin/env node
/**
 * Fills fake_s3.mjs's disk cache with every object a fixed query set touches, so the measured runs never wait on R2:
 * opens the index through the proxy (TypeScript SDK, one process) and runs queries [from, from + count) at the
 * bench's nprobe / rerank. The proxy fetches each object whole on first touch, so one range read is enough.
 *
 *   node bench/cold/warm_proxy.mjs [--proxy http://127.0.0.1:9555] [--from 0] [--count 30] [--nprobe N] [--rerank R]
 *        [--keys-out data/cold/warm-keys.json]
 *
 * Run it twice: the second run must report 0 upstream fetches. `--keys-out` writes the sorted slab keys the set
 * touches (make_pack.mjs reads the proxy's cache directly, this list is for the record).
 */
import fs from "node:fs";
import path from "node:path";
import { loadDataset, parseArgs, root } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const PROXY = args.proxy ?? "http://127.0.0.1:9555";
const FROM = Number(args.from ?? 0);
const COUNT = Number(args.count ?? 30);
const dist = path.join(root, "sdk/ts/dist");
const sdk = await import(path.join(dist, "index.js"));
const thread = await sdk.Thread.init({ wasmModule: new WebAssembly.Module(fs.readFileSync(path.join(dist, "wasm/thread_bg.wasm"))) });
const ds = loadDataset();
const config = await (await fetch(`${PROXY}/__config`)).json();
// Warm at full speed, then restore the profile.
await fetch(`${PROXY}/__config`, { method: "POST", body: JSON.stringify({ rttMs: 0, mbps: 0, jitterMs: 0, connMs: 0, linkMbps: 0 }) });
try {
	await fetch(`${PROXY}/__reset`, { method: "POST" });
	const store = new sdk.S3BlobStore({ endpoint: PROXY, bucket: "vib-bench", accessKeyId: "fake", secretAccessKey: "fake" });
	const index = await thread.open({ store });
	const opts = { topK: 10, ...(args.nprobe ? { nprobe: Number(args.nprobe) } : {}), ...(args.rerank ? { rerank: Number(args.rerank) } : {}) };
	const slabKeys = new Set();
	for (let i = FROM; i < FROM + COUNT; i++) {
		const r = await index.query(ds.vector(i), opts);
		for (const c of r.stats.cells) slabKeys.add(sdk.keys.slab(index.gen, c, index.revision.slabs ? sdk.slabRevOf(index.revision.slabs[c]) : 0));
	}
	const st = await (await fetch(`${PROXY}/__reset`, { method: "POST" })).json();
	console.log(`warmed queries ${FROM}..${FROM + COUNT - 1} of ${index.gen} rev ${index.rev}: ${slabKeys.size} distinct slabs, ${st.keys.length} objects touched, ${st.requests} requests; upstream fetches ${st.upstreamFetches} (${(st.upstreamBytes / 1048576).toFixed(1)} MB)`);
	if (args["keys-out"]) {
		fs.mkdirSync(path.dirname(path.resolve(root, args["keys-out"])), { recursive: true });
		fs.writeFileSync(path.resolve(root, args["keys-out"]), JSON.stringify({ gen: index.gen, rev: index.rev, from: FROM, count: COUNT, objects: st.keys.map((k) => k.key), slabs: [...slabKeys].sort() }, null, 1));
	}
	index.close();
} finally {
	await fetch(`${PROXY}/__config`, { method: "POST", body: JSON.stringify(config) });
}
