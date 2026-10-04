#!/usr/bin/env node
/**
 * What a revision switch costs the query that notices it (item "prefetch on refresh" of docs/cold-start.md).
 * Needs a fake_s3.mjs proxy with `--overlay DIR` where DIR holds `idx/current.json` and a second revision (e.g.
 * `idx/g4/manifest.180.json` naming a new slabs table and a tombstone set; bench/cold/make_refresh_overlay.py writes
 * one). Each trial is a fresh process: open at rev A (refreshInterval 1 s), query, point DIR/idx/current.json at rev
 * B, wait 1.1 s, then
 *   blocking  (the old behaviour): await index.refresh() then query -> the latency the triggering query used to see
 *   background (now): query (it starts the refresh in the background) -> its latency; then the time until the index
 *             has switched and the first query on the new revision
 * Both SDKs: --sdk ts|py.  node bench/cold/refresh_bench.mjs --proxy http://127.0.0.1:9656 --overlay DIR --n 10
 */
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { loadDataset, parseArgs, root } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const PROXY = args.proxy ?? "http://127.0.0.1:9656";
const OVERLAY = path.resolve(root, args.overlay ?? "data/cold-store/overlay-refresh");
const N = Number(args.n ?? 10);
const SDK = args.sdk ?? "ts";
const [A, B] = (args.revs ?? "179,180").split(",").map(Number);
const cur = path.join(OVERLAY, "idx/current.json");
const point = (rev) => fs.writeFileSync(cur, JSON.stringify({ gen: "g4", rev }));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function childTs(mode, trial) {
	const sdk = await import(path.join(root, "sdk/ts/dist/index.js"));
	const thread = await sdk.Thread.init({ wasmModule: new WebAssembly.Module(fs.readFileSync(path.join(root, "sdk/ts/dist/wasm/thread_bg.wasm"))) });
	const ds = loadDataset();
	const s3 = new sdk.S3BlobStore({ endpoint: PROXY, bucket: "vib-bench", accessKeyId: "f", secretAccessKey: "f" });
	const ix = await thread.open({ store: s3, refreshInterval: 1000 });
	const q = (i) => ds.vector(20 + 2 * trial + i);
	await ix.query(q(0));
	await ix.store.drain?.();
	point(B);
	await sleep(1100);
	const out = { mode, trial };
	let t = performance.now();
	if (mode === "blocking") {
		await ix.refresh();
		const r = await ix.query(q(0));
		out.trigger = performance.now() - t;
		out.triggerRev = r.stats.rev;
	} else {
		const r = await ix.query(q(0));
		out.trigger = performance.now() - t;
		out.triggerRev = r.stats.rev;
		while (ix.rev !== B) await sleep(1);
		out.switchMs = performance.now() - t;
	}
	t = performance.now();
	const r2 = await ix.query(q(0));
	out.after = performance.now() - t;
	out.afterRev = r2.stats.rev;
	ix.close();
	return out;
}

if (args.child === "true") {
	const r = await childTs(args.mode, Number(args.trial));
	console.log("@@ROW " + JSON.stringify(r));
	process.exit(0);
}

const rows = [];
for (let t = 0; t < N; t++) {
	for (const mode of ["blocking", "background"]) {
		point(A);
		const cmd = SDK === "ts" ? [process.execPath, [new URL(import.meta.url).pathname, ...process.argv.slice(2), "--child", "--mode", mode, "--trial", String(t)]] : [path.join(root, "sdk/python/.venv/bin/python"), [path.join(root, "bench/cold/refresh_bench.py"), "--proxy", PROXY, "--overlay", OVERLAY, "--mode", mode, "--trial", String(t), "--revs", `${A},${B}`]];
		const p = spawnSync(cmd[0], cmd[1], { encoding: "utf8" });
		if (p.status !== 0) throw new Error(`${SDK} ${mode} trial ${t}: ${p.stderr}`);
		const row = JSON.parse(p.stdout.split("\n").find((l) => l.startsWith("@@ROW ")).slice(6));
		rows.push(row);
		console.log(JSON.stringify(row));
	}
}
point(A);
const med = (xs) => {
	const s = xs.filter((x) => x !== undefined).sort((a, b) => a - b);
	return s.length ? Math.round(s[Math.floor((s.length - 1) / 2)] * 10) / 10 : null;
};
const p90 = (xs) => {
	const s = xs.filter((x) => x !== undefined).sort((a, b) => a - b);
	return s.length ? Math.round(s[Math.min(s.length - 1, Math.ceil(0.9 * s.length) - 1)] * 10) / 10 : null;
};
for (const mode of ["blocking", "background"]) {
	const rs = rows.filter((r) => r.mode === mode);
	console.log(`${SDK} ${mode}: triggering query med ${med(rs.map((r) => r.trigger))} p90 ${p90(rs.map((r) => r.trigger))} ms (rev ${rs[0].triggerRev}); switch after ${med(rs.map((r) => r.switchMs))} ms; next query med ${med(rs.map((r) => r.after))} p90 ${p90(rs.map((r) => r.after))} ms (rev ${rs[0].afterRev})`);
}
