#!/usr/bin/env node
/**
 * Verifies the deployed SDK example against the reference Worker and the dataset ground truth.
 *   node sdk/scripts/verify.mjs [--n 50] [--nprobe 34] [--rerank 150] [--k 10] [--store r2|s3] [--example URL] [--main URL]
 * For each of the first n queries of data/prep/queries.f32 it POSTs the vector to the example's /search (JSON) and
 * to the main Worker's /query (octet-stream) with the same nprobe/rerank, then compares the top-k id sets, the
 * recall@k of each against data/prep/groundtruth.u32 and the client latencies. Tokens: worker/.env.secrets
 * (BUILD_TOKEN, main Worker) and sdk/examples/basic/.env.secrets (AUTH_TOKEN, example).
 */
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");
const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) => (a.startsWith("--") ? [a.slice(2), all[i + 1] && !all[i + 1].startsWith("--") ? all[i + 1] : "true"] : null)).filter(Boolean));
const N = Number(args.n ?? 50);
const k = Number(args.k ?? 10);
const nprobe = Number(args.nprobe ?? 34);
const rerank = Number(args.rerank ?? 150);
const store = args.store ?? "r2";
const EXAMPLE = args.example ?? process.env.VIB_SDK_URL ?? "https://vib-sdk-example.aiuitech.workers.dev";
const MAIN = args.main ?? process.env.VIB_URL ?? "https://vib-bench.aiuitech.workers.dev";
const envOf = (f) => Object.fromEntries(fs.readFileSync(f, "utf8").split("\n").filter((l) => l.includes("=")).map((l) => [l.slice(0, l.indexOf("=")).trim(), l.slice(l.indexOf("=") + 1).trim()]));
const mainToken = envOf(path.join(root, "worker/.env.secrets")).BUILD_TOKEN;
const exampleToken = envOf(path.join(root, "sdk/examples/basic/.env.secrets")).AUTH_TOKEN;
const prep = JSON.parse(fs.readFileSync(path.join(root, "data/prep/manifest.json"), "utf8"));
const dim = prep.dim;
const qbuf = fs.readFileSync(path.join(root, "data/prep/queries.f32"));
const gtbuf = fs.readFileSync(path.join(root, "data/prep/groundtruth.u32"));
const total = qbuf.byteLength / (dim * 4);
const gtk = gtbuf.byteLength / (total * 4);
const truth = (i) => new Set(Array.from({ length: Math.min(k, gtk) }, (_, j) => gtbuf.readUInt32LE(i * gtk * 4 + j * 4)));

async function timed(fn) {
	const t0 = performance.now();
	const r = await fn();
	return { ...r, ms: performance.now() - t0 };
}
async function sdkQuery(i, vec) {
	const res = await fetch(`${EXAMPLE}/search`, { method: "POST", headers: { authorization: `Bearer ${exampleToken}`, "content-type": "application/json" }, body: JSON.stringify({ vector: Array.from(new Float32Array(vec.buffer, vec.byteOffset, dim)), topK: k, nprobe, rerank, store }) });
	if (!res.ok) throw new Error(`example HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
	const j = await res.json();
	return { ids: j.ids, scores: j.scores, server: j.timing, stats: j.stats, gen: j.gen, rev: j.rev };
}
async function mainQuery(i, vec) {
	const res = await fetch(`${MAIN}/query?k=${k}&nprobe=${nprobe}&rerank=${rerank}`, { method: "POST", headers: { authorization: `Bearer ${mainToken}`, "content-type": "application/octet-stream" }, body: vec });
	if (!res.ok) throw new Error(`main HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
	const j = await res.json();
	return { ids: j.ids, scores: j.scores, server: j.timing, stats: j.stats, gen: j.gen, rev: j.rev };
}

const rows = [];
let warm = false;
for (let pass = 0; pass < 2; pass++) {
	// pass 0 warms both Workers' caches on the same queries (slabs are then in the edge cache for both); pass 1 measures.
	for (let i = 0; i < N; i++) {
		const vec = qbuf.subarray(i * dim * 4, (i + 1) * dim * 4);
		const [a, b] = await Promise.all([timed(() => sdkQuery(i, vec)), timed(() => mainQuery(i, vec))]);
		if (pass === 0) continue;
		const gt = truth(i);
		const same = a.ids.length === b.ids.length && a.ids.every((id, j) => id === b.ids[j]);
		const setA = new Set(a.ids);
		const overlap = b.ids.filter((id) => setA.has(id)).length;
		rows.push({ i, sameOrder: same, overlap, recallSdk: a.ids.filter((id) => gt.has(id)).length / gt.size, recallMain: b.ids.filter((id) => gt.has(id)).length / gt.size, msSdk: a.ms, msMain: b.ms, srvSdk: a.server?.total, srvMain: b.server?.total, sdk: a, main: b });
	}
	warm = true;
}
const pct = (xs, q) => xs.slice().sort((x, y) => x - y)[Math.min(xs.length - 1, Math.floor(q * xs.length))];
const mean = (xs) => xs.reduce((x, y) => x + y, 0) / xs.length;
const fmt = (x) => x.toFixed(1);
const r = rows;
const first = r[0];
console.log(`store=${store} n=${r.length} k=${k} nprobe=${nprobe} rerank=${rerank}  example ${EXAMPLE} (gen ${first.sdk.gen} rev ${first.sdk.rev})  main ${MAIN} (gen ${first.main.gen} rev ${first.main.rev})`);
console.log(`identical top-${k} (same ids, same order): ${r.filter((x) => x.sameOrder).length}/${r.length};  mean overlap ${fmt(mean(r.map((x) => x.overlap)))}/${k};  queries with overlap < ${k}: ${r.filter((x) => x.overlap < k).map((x) => `#${x.i}(${x.overlap})`).join(" ") || "none"}`);
console.log(`recall@${k} vs ground truth: SDK ${(100 * mean(r.map((x) => x.recallSdk))).toFixed(2)}%  main ${(100 * mean(r.map((x) => x.recallMain))).toFixed(2)}%`);
console.log(`client latency ms (warm, sequential, 2 in flight): SDK p50 ${fmt(pct(r.map((x) => x.msSdk), 0.5))} p90 ${fmt(pct(r.map((x) => x.msSdk), 0.9))} mean ${fmt(mean(r.map((x) => x.msSdk)))} | main p50 ${fmt(pct(r.map((x) => x.msMain), 0.5))} p90 ${fmt(pct(r.map((x) => x.msMain), 0.9))} mean ${fmt(mean(r.map((x) => x.msMain)))}`);
console.log(`server-reported total ms: SDK mean ${fmt(mean(r.map((x) => x.srvSdk ?? 0)))} | main mean ${fmt(mean(r.map((x) => x.srvMain ?? 0)))}`);
const s = r.map((x) => x.sdk.stats);
console.log(`SDK stats (mean): scanBytes ${(mean(s.map((x) => x.scanBytes)) / 1048576).toFixed(2)} MB  rerankReads ${fmt(mean(s.map((x) => x.rerankReads)))}  rerankSlabs ${fmt(mean(s.map((x) => x.rerankSlabs)))}  specUsed ${fmt(mean(s.map((x) => x.specUsed)))}  cache hits ${fmt(mean(s.map((x) => x.io.hits)))} misses ${fmt(mean(s.map((x) => x.io.misses)))} origin reads ${fmt(mean(s.map((x) => x.io.reads)))}  r206 ${fmt(mean(s.map((x) => x.io.r206)))} r200 ${fmt(mean(s.map((x) => x.io.r200)))}  deltas ${fmt(mean(s.map((x) => x.deltas)))}  attempt>1: ${s.filter((x) => x.attempt > 1).length}`);
for (const x of r.filter((x) => !x.sameOrder).slice(0, 5)) {
	console.log(`  #${x.i}: sdk ${JSON.stringify(x.sdk.ids)} ${x.sdk.scores.map((v) => v.toFixed(4)).join(",")}`);
	console.log(`  #${x.i}: main ${JSON.stringify(x.main.ids)} ${x.main.scores.map((v) => v.toFixed(4)).join(",")}`);
}
if (args.json) fs.writeFileSync(args.json, JSON.stringify(rows.map(({ sdk, main, ...rest }) => rest), null, 1));
