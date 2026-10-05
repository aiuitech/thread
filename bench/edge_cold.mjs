#!/usr/bin/env node
/**
 * First-query latency against a cold edge cache: one query at a time, spaced apart, each with a query vector the
 * caller has not sent recently (pick indices the warm benchmarks do not use: query_bench.mjs sends 0..n-1).
 *   node bench/edge_cold.mjs --gen g4 --queries 4100,4200,4300 [--gap 20] [--json FILE]
 * Prints per query: client ms, server total / phase1 / phase2, slab cache hits and misses (a miss reads the whole slab
 * from R2), bytes read (whole slabs for misses + cached prefixes), whether the isolate had to load the index first
 * (timing.load - timing.gen, the time loading the base and revision before routing, above 20 ms: a cold isolate)
 * and the edge colo; then the medians.
 * Queries only; manifest defaults unless --nprobe / --rerank are given.
 */
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const args = {};
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, "")] = process.argv[i + 1];
const BASE = process.env.THREAD_URL ?? "https://vib-bench.aiuitech.workers.dev";
const TOKEN = fs.readFileSync(path.join(root, "examples/cloudflare-worker/.env.secrets"), "utf8").trim().split("=").slice(1).join("=");
const dim = JSON.parse(fs.readFileSync(path.join(root, "data/prep/manifest.json"), "utf8")).dim;
const qbuf = fs.readFileSync(path.join(root, "data/prep/queries.f32"));
const gtbuf = fs.readFileSync(path.join(root, "data/prep/groundtruth.u32"));
const total = qbuf.byteLength / (dim * 4);
const gtk = gtbuf.byteLength / (total * 4);
const list = (args.queries ?? "4100,4200,4300,4400,4500").split(",").map(Number);
const gap = Number(args.gap ?? 20) * 1000;
const qs = ["k=10", "cache=edge", args.gen ? `gen=${args.gen}` : null, args.nprobe ? `nprobe=${args.nprobe}` : null, args.rerank ? `rerank=${args.rerank}` : null].filter(Boolean).join("&");

const rows = [];
for (const [n, qi] of list.entries()) {
	if (n > 0) await new Promise((r) => setTimeout(r, gap));
	const t0 = performance.now();
	const res = await fetch(`${BASE}/query?${qs}`, { method: "POST", headers: { authorization: `Bearer ${TOKEN}`, "content-type": "application/octet-stream" }, body: qbuf.subarray(qi * dim * 4, (qi + 1) * dim * 4) });
	const ms = performance.now() - t0;
	if (!res.ok) throw new Error(`query ${qi}: HTTP ${res.status} ${(await res.text()).slice(0, 200)}`);
	const d = await res.json();
	const gt = new Set(Array.from({ length: 10 }, (_, j) => gtbuf.readUInt32LE((qi * gtk + j) * 4)));
	const row = {
		at: new Date().toISOString(), query: qi, gen: d.gen, rev: d.rev, ms: +ms.toFixed(1), srvTotal: d.timing.total, phase1: d.timing.phase1 ?? null, phase2: d.timing.phase2 ?? null,
		hits: d.stats.cacheHits, misses: d.stats.cacheMisses, readMB: +(d.stats.slabBytes / 1048576).toFixed(2), scanMB: +((d.stats.scanBytes ?? 0) / 1048576).toFixed(2),
		load: d.timing.load, indexLoadMs: d.timing.indexLoadMs ?? null, coldIsolate: d.timing.load - (d.timing.gen ?? 0) > 20, recall: d.ids.filter((id) => gt.has(id)).length / 10, colo: d.stats.colo,
	};
	rows.push(row);
	console.log(`q${qi} ${row.ms} ms | srv ${row.srvTotal} (ph1 ${row.phase1}, ph2 ${row.phase2}) | slabs ${row.hits} hit / ${row.misses} miss | read ${row.readMB} MB | load ${row.load} ms (isolate ${row.coldIsolate ? "cold" : "warm"}) | recall ${row.recall} | ${row.colo}`);
	if (args.json) fs.appendFileSync(args.json, JSON.stringify(row) + "\n");
}
const med = (f) => {
	const v = rows.map(f).sort((a, b) => a - b);
	return v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2;
};
console.log(`median of ${rows.length}: client ${med((r) => r.ms).toFixed(1)} ms, server ${med((r) => r.srvTotal)} ms, misses ${med((r) => r.misses)} of ${med((r) => r.hits + r.misses)}, read ${med((r) => r.readMB).toFixed(1)} MB; cold isolates ${rows.filter((r) => r.coldIsolate).length}/${rows.length}`);
