#!/usr/bin/env node
/**
 * Query benchmark against the deployed Worker.
 *   node bench/query_bench.mjs --n 1000 --concurrency 50 --k 10 [--nprobe N] [--pc 0] [--rerank N] [--gen g3] [--rev R] [--warm 1] [--cache none|mem|edge]
 * --rev R reads manifest revision R (default: the revision idx/current.json points at, or the head of --gen).
 * --nprobe/--rerank/--pc are sent only when given; otherwise the Worker uses the generation's
 * manifest defaults (g3+: profile defaults, older generations: nprobe 1, rerank 32, pc 0). The
 * values the server actually used are printed from stats.params.
 * --rerank N: int8 re-scoring of the scan tier's top max(k, N) (format-2 generations); 0 = scan-tier scores only.
 * --filter '{"attrs":{"$any":1}}' [--maxNprobe N]: an attribute filter (field conditions under manifest.attrs, or the
 * raw `attrs` word without a schema) applied inside every scan; the Worker widens its probes past nprobe up to
 * maxNprobe (default 4 x nprobe) until the rerank set is full. The summary adds the cells scanned per query.
 * --route flat|beam, --beam N: fine-cell routing (beam = only the cells of the centroid tree's best N groups; needs
 * manifest.tree); sent only when given (default: manifest.defaults.route, else flat; beam: manifest.tree.beam).
 * Interleaved routing comparison: --extras "route=flat|route=beam&beam=8|route=beam&beam=16".
 * Reports latency percentiles (client-observed), throughput and recall@k
 * against the dataset's exact top-10 ground truth, plus the server-reported
 * slab cache hit rate (cache=mem: in-isolate Map, edge: Cache API, none: direct R2).
 * --extra "mode=fetchonly&scanRepeat=3": raw query-string debug switches passed to /query (see DebugOpts in
 * examples/cloudflare-worker/src/index.ts). --json FILE appends one JSON summary line per measured pass.
 * --extras "a|b|c": interleaved A/B mode. Request i runs query vector floor(i/m) with config (i+floor(i/m))%m
 * (m configs, order rotated per vector), so
 * every config sees the same queries and the same network conditions over time; a per-config summary follows
 * the pass (one JSON line per config with --json). Use "-" for "no extra switches".
 * Workers that return a trace envelope (`trace`, docs/observability.md) get a per-span breakdown after the summary
 * (`--trace 0` hides it); the --json row carries it as `trace`.
 */
import fs from "node:fs";
import path from "node:path";
import { printSummary, summarize } from "./lib/trace.mjs";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const args = {};
for (let i = 2; i < process.argv.length; i++) {
	const m = process.argv[i].match(/^--([^=\s]+)(?:=(.*))?$/);
	if (!m) continue;
	if (m[2] !== undefined) args[m[1]] = m[2];
	else if (process.argv[i + 1] && !process.argv[i + 1].startsWith("--")) args[m[1]] = process.argv[++i];
	else args[m[1]] = "true";
}
const BASE = process.env.THREAD_URL ?? "https://vib-bench.aiuitech.workers.dev";
const TOKEN = fs.readFileSync(path.join(root, "examples/cloudflare-worker/.env.secrets"), "utf8").trim().split("=").slice(1).join("=");
const manifest = JSON.parse(fs.readFileSync(path.join(root, "data/prep/manifest.json"), "utf8"));
const dim = manifest.dim;
const N = Number(args.n ?? 1000);
const C = Number(args.concurrency ?? 50);
const k = Number(args.k ?? 10);
// Omitted unless given, so the Worker applies the manifest defaults.
const opt = (name) => (args[name] !== undefined ? Number(args[name]) : null);
const nprobe = opt("nprobe");
const pc = opt("pc"); // 0 = flat routing
const warm = args.warm !== "0";
const cache = ({ "1": "mem", "0": "none" })[args.cache] ?? args.cache ?? "edge";
if (!["none", "mem", "edge"].includes(cache)) throw new Error(`--cache must be none|mem|edge, got ${cache}`);
const gen = (args.gen ? `&gen=${args.gen}` : "") + (args.rev !== undefined ? `&rev=${args.rev}` : "");
const rerank = opt("rerank");
const beam = opt("beam");
const route = args.route ?? null;
if (route !== null && !["flat", "beam"].includes(route)) throw new Error(`--route must be flat|beam, got ${route}`);
const maxNprobe = opt("maxNprobe");
const filter = args.filter ? encodeURIComponent(JSON.stringify(JSON.parse(args.filter))) : null;
const qs = [`k=${k}`, `cache=${cache}`, ...Object.entries({ nprobe, pc, rerank, route, beam, maxNprobe, filter }).filter(([, v]) => v !== null).map(([n, v]) => `${n}=${v}`)].join("&");
const show = (v) => (v === null ? "default" : v);
const normExtra = (x) => (x && x !== "-" ? `&${x.replace(/^[?&]/, "")}` : "");
const extra = normExtra(args.extra);
const extras = args.extras ? args.extras.split("|").map(normExtra) : null;

const qbuf = fs.readFileSync(path.join(root, "data/prep/queries.f32"));
const gtbuf = fs.readFileSync(path.join(root, "data/prep/groundtruth.u32"));
const total = qbuf.byteLength / (dim * 4);
const gtk = gtbuf.byteLength / (total * 4);
const M = extras ? extras.length : 1;
// In interleaved mode request i uses query vector floor(i / M) under config i % M.
const queries = Array.from({ length: Math.min(N, total * M) }, (_, i) => qbuf.subarray(Math.floor(i / M) * dim * 4, (Math.floor(i / M) + 1) * dim * 4));
const qIndex = (i) => Math.floor(i / M);
const truth = (i) => {
	const dv = new DataView(gtbuf.buffer, gtbuf.byteOffset + i * gtk * 4, gtk * 4);
	return new Set(Array.from({ length: Math.min(k, gtk) }, (_, j) => dv.getUint32(j * 4, true)));
};

async function query(i) {
	// Rotate the config order per query vector (Latin square) so no config always touches the slabs first.
	const cfg = extras ? (i + qIndex(i)) % M : 0;
	const t0 = performance.now();
	const res = await fetch(`${BASE}/query?${qs}${gen}${extras ? extras[cfg] : extra}`, {
		method: "POST",
		headers: { authorization: `Bearer ${TOKEN}`, "content-type": "application/octet-stream" },
		body: queries[i],
	});
	const ms = performance.now() - t0;
	if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
	const data = await res.json();
	const gt = truth(qIndex(i));
	const hit = data.ids.filter((id) => gt.has(id)).length;
	return { ms, cfg, recall: hit / gt.size, timing: data.timing, stats: data.stats, gen: data.gen, rev: data.rev, cells: data.cells, trace: data.trace ?? null };
}

async function run(label) {
	const results = new Array(queries.length);
	let next = 0;
	const t0 = performance.now();
	await Promise.all(Array.from({ length: Math.min(C, queries.length) }, async () => {
		while (true) {
			const i = next++;
			if (i >= queries.length) return;
			try {
				results[i] = await query(i);
			} catch (e) {
				results[i] = { error: String(e.message) };
			}
		}
	}));
	const wall = (performance.now() - t0) / 1000;
	const ok = results.filter((r) => !r.error);
	const errs = results.length - ok.length;
	const lat = ok.map((r) => r.ms).sort((a, b) => a - b);
	const p = (q) => lat[Math.min(lat.length - 1, Math.floor(q * lat.length))].toFixed(1);
	const avg = (f) => (ok.reduce((a, r) => a + f(r), 0) / ok.length).toFixed(1);
	console.log(`\n[${label}] n=${ok.length} errors=${errs} concurrency=${C} k=${k} nprobe=${show(nprobe)} pc=${show(pc)} cache=${cache} rerank=${show(rerank)}${extra ? ` extra=${extra.slice(1)}` : ""}`);
	// What the server used (query string, manifest defaults or legacy constants).
	const used = [
		...new Set(
			ok.map((r) => {
				const pp = r.stats.params;
				if (!pp) return "n/a (old Worker)";
				const rt = pp.route ? ` route=${pp.route} (${pp.source.route})${pp.beam !== null && pp.beam !== undefined ? ` beam=${pp.beam} (${pp.source.beam})` : ""}` : "";
				return `nprobe=${pp.nprobe} (${pp.source.nprobe}) rerank=${pp.rerank} (${pp.source.rerank}) pc=${pp.pc} (${pp.source.pc})${rt}`;
			}),
		),
	];
	const gens = [...new Set(ok.map((r) => `${r.gen ?? "?"}${r.rev !== undefined ? ` rev ${r.rev}` : ""}${r.stats.profile ? ` profile ${r.stats.profile}` : ""}`))];
	console.log(`  server params: ${used.join(" | ")}  gen ${gens.join(",")}`);
	console.log(`  recall@${k}: ${(100 * ok.reduce((a, r) => a + r.recall, 0) / ok.length).toFixed(2)}%`);
	console.log(`  latency ms: p50 ${p(0.5)}  p75 ${p(0.75)}  p90 ${p(0.9)}  p95 ${p(0.95)}  p99 ${p(0.99)}  mean ${avg((r) => r.ms)}`);
	console.log(`  throughput: ${(ok.length / wall).toFixed(1)} rps over ${wall.toFixed(1)}s`);
	console.log(`  server: gen ${avg((r) => r.timing.gen ?? 0)}ms  load ${avg((r) => r.timing.load)}ms  fetch ${avg((r) => r.timing.fetch)}ms  total ${avg((r) => r.timing.total)}ms  scanned ${avg((r) => r.stats.scanned)} vecs  slabBytes ${(avg((r) => r.stats.slabBytes) / 1024).toFixed(0)} KB`);
	const fmts = [...new Set(ok.map((r) => r.stats.format ?? 1))].join(",");
	console.log(
		`  format ${fmts}: scanBytes ${(avg((r) => r.stats.scanBytes ?? r.stats.slabBytes) / 1024).toFixed(0)} KB  rerank T ${avg((r) => r.stats.rerank ?? 0)}  rerankReads ${avg((r) => r.stats.rerankReads ?? 0)}  rerankSlabs ${avg((r) => r.stats.rerankSlabs ?? 0)}  multipart ${avg((r) => r.stats.rerankMultipart ?? 0)}  rerankBytes ${(avg((r) => r.stats.rerankBytes ?? 0) / 1024).toFixed(1)} KB  rerankLocal ${avg((r) => r.stats.rerankLocal ?? 0)}`,
	);
	console.log(`  phases: phase1 ${avg((r) => r.timing.phase1 ?? 0)}ms (fetch ${avg((r) => r.timing.fetch)}ms)  phase2 ${avg((r) => r.timing.phase2 ?? 0)}ms`);
	if (filter) {
		console.log(`  filter ${args.filter}: cells scanned ${avg((r) => r.cells?.length ?? 0)} (maxNprobe ${show(maxNprobe)}, server ${avg((r) => r.stats.maxNprobe ?? 0)})  filterDropped ${avg((r) => r.stats.filterDropped ?? 0)}  candidates ${avg((r) => r.stats.candidates ?? 0)}`);
	}
	if (ok.some((r) => r.stats.deltas || r.stats.tombstones)) {
		console.log(
			`  update path: deltas ${avg((r) => r.stats.deltas ?? 0)}  deltaBytes ${(avg((r) => r.stats.deltaBytes ?? 0) / 1024).toFixed(0)} KB  deltaScanned ${avg((r) => r.stats.deltaScanned ?? 0)}  deltaHits ${avg((r) => r.stats.deltaHits ?? 0)} (dropped ${avg((r) => r.stats.deltaDropped ?? 0)}, in top-k ${avg((r) => r.stats.deltaTop ?? 0)})  tombstones ${avg((r) => r.stats.tombstones ?? 0)} in ${avg((r) => r.stats.tombSets ?? 0)} sets (slab hits dropped ${avg((r) => r.stats.tombDropped ?? 0)})  delta ms ${avg((r) => r.timing.delta ?? 0)}  wasmMem max ${Math.max(...ok.map((r) => r.stats.wasmMemMB ?? 0))} MB`,
		);
	}
	const rng = ok.reduce((a, r) => ({ r206: a.r206 + (r.stats.range?.r206 ?? 0), r200: a.r200 + (r.stats.range?.r200 ?? 0), miss: a.miss + (r.stats.range?.miss ?? 0) }), { r206: 0, r200: 0, miss: 0 });
	const retried = ok.filter((r) => (r.stats.attempt ?? 1) > 1).length;
	console.log(`  cache range requests: 206 ${rng.r206}  200(full) ${rng.r200}  miss ${rng.miss}  retried-after-wasm-reset ${retried}`);
	const hits = ok.reduce((a, r) => a + (r.stats.cacheHits ?? 0), 0);
	const misses = ok.reduce((a, r) => a + (r.stats.cacheMisses ?? 0), 0);
	const colos = [...new Set(ok.map((r) => r.stats.colo))].join(",");
	const traced = ok.filter((r) => r.trace);
	const traceSum = traced.length ? summarize(traced.map((r) => r.trace)) : null;
	if (traceSum && args.trace !== "0") printSummary(traceSum, { indent: "  trace " });
	console.log(`  cache: ${cache} slab hit rate ${hits + misses ? ((100 * hits) / (hits + misses)).toFixed(1) : "n/a"}% (${hits} hits / ${misses} misses)  colo ${colos}`);
	if (errs) {
		const kinds = {};
		for (const r of results) if (r.error) kinds[r.error.slice(0, 120)] = (kinds[r.error.slice(0, 120)] ?? 0) + 1;
		console.log("  first error:", results.find((r) => r.error).error);
		console.log("  error kinds:", JSON.stringify(kinds));
	}
	if (extras) {
		console.log(`  per config (interleaved; client ms | server ms | stats):`);
		for (let c = 0; c < M; c++) {
			const g = ok.filter((r) => r.cfg === c);
			if (!g.length) continue;
			const l = g.map((r) => r.ms).sort((a, b) => a - b);
			const q = (x) => l[Math.min(l.length - 1, Math.floor(x * l.length))];
			const m = (f) => g.reduce((a, r) => a + f(r), 0) / g.length;
			const row = {
				at: new Date().toISOString(), label, n: g.length, errors: results.filter((r, i) => r.error && (i + qIndex(i)) % M === c).length, c: C, nprobe: show(nprobe), rerank: show(rerank), extra: extras[c].slice(1) || "-",
				gen: g[0]?.gen, rev: g[0]?.rev, routing: g[0]?.stats.routing, beam: g[0]?.stats.beam ?? null, wall: +wall.toFixed(2),
				recall: +(100 * m((r) => r.recall)).toFixed(2), p50: +q(0.5).toFixed(1), p90: +q(0.9).toFixed(1), p95: +q(0.95).toFixed(1), p99: +q(0.99).toFixed(1), mean: +m((r) => r.ms).toFixed(1),
				srvTotal: +m((r) => r.timing.total).toFixed(1), fetch: +m((r) => r.timing.fetch ?? 0).toFixed(1), phase2: +m((r) => r.timing.phase2 ?? 0).toFixed(1),
				rerankReads: +m((r) => r.stats.rerankReads ?? 0).toFixed(2), rerankSlabs: +m((r) => r.stats.rerankSlabs ?? 0).toFixed(2), rerankKB: +m((r) => (r.stats.rerankBytes ?? 0) / 1024).toFixed(1),
				specUsed: +m((r) => r.stats.specUsed ?? 0).toFixed(2), specReads: +m((r) => r.stats.specReads ?? 0).toFixed(2), retries: g.reduce((a, r) => a + (r.stats.limiterRetries ?? 0), 0), wasmMemMB: Math.max(...g.map((r) => r.stats.wasmMemMB ?? 0)),
			};
			console.log(`  [${row.extra}] n=${row.n} err=${row.errors} recall ${row.recall}% | p50 ${row.p50} p90 ${row.p90} p95 ${row.p95} p99 ${row.p99} mean ${row.mean} | srv total ${row.srvTotal} fetch ${row.fetch} ph2 ${row.phase2} | rr ${row.rerankReads}/${row.rerankSlabs} slabs ${row.rerankKB} KB spec ${row.specReads} used ${row.specUsed} retries ${row.retries} wasmMem ${row.wasmMemMB} MB | routing ${row.routing}${row.beam ? ` beam ${row.beam}` : ""}`);
			if (args.json && label === "measured") fs.appendFileSync(args.json, JSON.stringify(row) + "\n");
		}
	}
	if (args.json && label === "measured" && !extras) {
		const num = (f) => Number(avg(f));
		fs.appendFileSync(args.json, JSON.stringify({
			at: new Date().toISOString(), n: ok.length, errors: errs, c: C, nprobe: show(nprobe), rerank: show(rerank), route: show(route), beam: show(beam), extra: extra.slice(1), gen: ok[0]?.gen, rev: ok[0]?.rev,
			routing: ok[0]?.stats.routing,
			deltas: num((r) => r.stats.deltas ?? 0), deltaKB: num((r) => (r.stats.deltaBytes ?? 0) / 1024), tombstones: num((r) => r.stats.tombstones ?? 0),
			recall: Number((100 * ok.reduce((a, r) => a + r.recall, 0) / ok.length).toFixed(2)),
			p50: Number(p(0.5)), p75: Number(p(0.75)), p90: Number(p(0.9)), p95: Number(p(0.95)), p99: Number(p(0.99)), mean: num((r) => r.ms),
			rps: Number((ok.length / wall).toFixed(1)),
			srvTotal: num((r) => r.timing.total), fetch: num((r) => r.timing.fetch), phase1: num((r) => r.timing.phase1 ?? 0), phase2: num((r) => r.timing.phase2 ?? 0),
			scanMB: num((r) => (r.stats.scanBytes ?? 0) / 1048576), rerankReads: num((r) => r.stats.rerankReads ?? 0), rerankSlabs: num((r) => r.stats.rerankSlabs ?? 0),
			rerankKB: num((r) => (r.stats.rerankBytes ?? 0) / 1024), misses: ok.reduce((a, r) => a + (r.stats.cacheMisses ?? 0), 0),
			...(traceSum ? { trace: traceSum } : {}),
		}) + "\n");
	}
	return { ok, lat };
}

if (warm) await run("warmup");
await run("measured");
