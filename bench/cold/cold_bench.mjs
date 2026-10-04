#!/usr/bin/env node
/**
 * Cold-start bench of the TypeScript SDK (Node) against S3: real R2 S3, the fake_s3.mjs proxy, or a pack served
 * by the proxy. bench/cold/cold_bench.py is the same bench for the Python SDK (same CLI, same JSONL schema).
 *
 *   node bench/cold/cold_bench.mjs --store r2s3|proxy|pack [--n 10] [--queries 0] [--nprobe N] [--rerank R]
 *        [--proxy http://127.0.0.1:9555] [--pack-key packs/g4-cold.threadpack] [--label TEXT]
 *        [--out data/cold/NAME.jsonl] [--env sdk/ts/examples/basic/.env.secrets]
 *        [--warm-pack KEY [--warm-wait slab|none]] [--coalesce-across-slabs GAP_BYTES]
 *
 * Each trial t spawns a fresh child process (fresh TLS/connection pool, fresh wasm, fresh index, empty caches) that
 * runs three scenarios, each on query vectors nobody used before in the run:
 *   process-cold  query (offset + 2t): import the SDK, compile + init the wasm, open the index, one query
 *   slab-cold     query (offset + 2t + 1), same process: index open, base loaded, slabs not read yet
 *   warm          query (offset + 2t + 1) again: whatever the SDK caches is now warm
 * Every query's recall@10 is checked against data/prep/groundtruth.u32. With --store proxy|pack the child resets
 * the proxy's counters before opening and reads them after each scenario, so `proxy` holds the exact requests and
 * bytes each scenario sent (the SDK's own counters are in `reqs` / `bytes`).
 *
 * `--warm-pack KEY` opens with `warmPack: KEY` (warm.ts; with `--store pack` the key is read from the S3 store next to
 * the pack); `--warm-wait slab` (the default) waits for the warm pack to be seeded before the slab-cold scenario, so
 * slab-cold and warm measure a loaded warm pack (process-cold never waits: it gets whatever has arrived). Each row
 * records the warm pack's state when its query started (`warm`). `--coalesce-across-slabs GAP` passes
 * `coalesceAcrossSlabs: GAP` to every query (`--store pack` only: phase-1 prefix reads of slabs within GAP bytes
 * of each other in the pack become one range read).
 *
 * Output: a summary table (median / p90 per scenario) on stdout and, with --out, JSONL: one `kind: "meta"` row
 * then one `kind: "trial"` row per (trial, scenario). `node bench/cold/summarize.mjs a.jsonl b.jsonl` compares runs.
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { execSync } from "node:child_process";
import { loadDataset, loadEnv, parseArgs, recallAt, root, SCHEMA, summary } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const STORE = args.store ?? "proxy";
if (!["r2s3", "proxy", "pack"].includes(STORE)) throw new Error(`--store must be r2s3, proxy or pack, not ${STORE}`);
const N = Number(args.n ?? 10);
const OFFSET = Number(args.queries ?? 0);
const PROXY = args.proxy ?? "http://127.0.0.1:9555";
const PACK_KEY = args["pack-key"] ?? "packs/g4-cold.threadpack";
const ENV_FILE = args.env ?? "sdk/ts/examples/basic/.env.secrets";
const nprobe = args.nprobe !== undefined ? Number(args.nprobe) : undefined;
const rerank = args.rerank !== undefined ? Number(args.rerank) : undefined;
const WARM_PACK = args["warm-pack"] ?? null;
const WARM_WAIT = args["warm-wait"] ?? "slab";
if (!["slab", "none"].includes(WARM_WAIT)) throw new Error("--warm-wait must be slab or none");
const COALESCE = args["coalesce-across-slabs"] !== undefined ? Number(args["coalesce-across-slabs"]) : undefined;
const dist = path.join(root, "sdk/ts/dist");

const nowMs = () => performance.now();
const r1 = (x) => (x === null || x === undefined ? null : Math.round(x * 10) / 10);

/**
 * Admin calls to the proxy over node:http (imported on first use), never fetch: Node loads its fetch/Response globals
 * lazily (13-18 ms in a cold process), wasm-bindgen's init touches `Response`, and whoever touches them first pays.
 * The child makes its first admin call only after `Thread.init`, so that cost stays in `wasmInit` for every store,
 * as in a real process.
 */
async function proxyCall(method, p) {
	const http = await import("node:http");
	return new Promise((resolve, reject) => {
		const req = http.request(PROXY + p, { method }, (res) => {
			let body = "";
			res.setEncoding("utf8");
			res.on("data", (d) => (body += d));
			res.on("end", () => (res.statusCode === 200 ? resolve(JSON.parse(body)) : reject(new Error(`proxy ${method} ${p}: HTTP ${res.statusCode}`))));
		});
		req.on("error", reject);
		req.end();
	});
}
const proxyStats = (s) => (s ? { requests: s.requests, bytes: s.bytes, newConnections: s.newConnections, upstreamFetches: s.upstreamFetches, byStatus: s.byStatus } : null);

// ---------- child: one trial ----------

async function child() {
	const tEntry = Date.now();
	const spawnAt = Number(args["spawn-at"]);
	const trial = Number(args.trial);
	const useProxy = STORE !== "r2s3";
	const ms = { boot: tEntry - spawnAt };
	let t = nowMs();
	const sdk = await import(path.join(dist, "index.js"));
	ms.import = nowMs() - t;
	const env = loadEnv(ENV_FILE);
	const ds = loadDataset();
	t = nowMs();
	const wasmModule = new WebAssembly.Module(fs.readFileSync(path.join(dist, "wasm/thread_bg.wasm")));
	ms.wasmCompile = nowMs() - t;
	const thread = await sdk.Thread.init({ wasmModule });
	ms.wasmInit = thread.initMs;
	if (useProxy) await proxyCall("POST", "/__reset"); // nothing above talks to the store
	const s3 = useProxy
		? new sdk.S3BlobStore({ endpoint: PROXY, bucket: env.S3_BUCKET ?? "vib-bench", region: "auto", accessKeyId: "fake", secretAccessKey: "fake" })
		: new sdk.S3BlobStore({ endpoint: env.S3_ENDPOINT, bucket: env.S3_BUCKET, region: "auto", accessKeyId: env.S3_ACCESS_KEY_ID, secretAccessKey: env.S3_SECRET_ACCESS_KEY });
	const store = STORE === "pack" ? await sdk.PackedBlobStore.open(s3, PACK_KEY) : s3;
	// No `cache` option: what a Node user gets by default (the Cache API does not exist in Node, so: none).
	const index = await thread.open({ store, ...(WARM_PACK ? { warmPack: WARM_PACK } : {}) });
	const os = index.openStats;
	const qopts = { topK: 10, ...(nprobe !== undefined ? { nprobe } : {}), ...(rerank !== undefined ? { rerank } : {}), ...(COALESCE !== undefined ? { coalesceAcrossSlabs: COALESCE } : {}) };
	const emit = (row) => process.stdout.write("@@ROW " + JSON.stringify(row) + "\n");
	const run = async (scenario, qi, withOpen) => {
		const w = index.warmStats;
		const warm = w ? { state: w.state, entries: w.entries, seeded: w.seeded, seededBytes: w.seededBytes, readyMs: w.readyMs, fetchMs: w.fetch?.ms ?? null, fetchBytes: w.fetch?.bytes ?? null } : null;
		const t0 = nowMs();
		const res = await index.query(ds.vector(qi), qopts);
		const wall = nowMs() - t0;
		const e2e = withOpen ? Date.now() - spawnAt : null;
		const proxy = useProxy ? proxyStats(await proxyCall("POST", "/__reset")) : null;
		const st = res.stats;
		const tl = st.timeline;
		const ids = res.hits.map((h) => h.id);
		const sum = (f, ...ps) => ps.reduce((a, p) => a + (p ? p[f] : 0), 0);
		const openPhases = withOpen ? [os.pack, os.current, os.manifest, os.slabs, os.base] : [];
		emit({
			kind: "trial",
			schema: SCHEMA,
			sdk: "ts",
			store: STORE,
			label: args.label ?? null,
			trial,
			scenario,
			query: qi,
			gen: st.gen,
			rev: st.rev,
			nprobe: res.params.nprobe,
			rerank: res.params.rerank,
			recall: recallAt(ids, ds.truth(qi)),
			ids,
			ms: {
				boot: withOpen ? ms.boot : null,
				import: withOpen ? r1(ms.import) : null,
				wasmCompile: withOpen ? r1(ms.wasmCompile) : null,
				wasmInit: withOpen ? r1(ms.wasmInit) : null,
				pack: withOpen && os.pack ? r1(os.pack.ms) : null,
				current: withOpen ? r1(os.current.ms) : null,
				manifest: withOpen ? r1(os.manifest.ms) : null,
				slabs: withOpen ? r1(os.slabs.ms) : null,
				base: withOpen ? r1(os.base.ms) : null,
				baseFetch: withOpen ? r1(os.base.fetch) : null,
				baseBuild: withOpen ? r1(os.base.build) : null,
				open: withOpen ? r1(os.total + (os.pack ? os.pack.ms : 0)) : null,
				route: r1(tl.route.ms),
				phase1: r1(tl.phase1.ms),
				phase2: r1(tl.phase2.ms),
				query: r1(wall),
				e2e,
			},
			reqs: {
				pack: withOpen && os.pack ? os.pack.reads : null,
				current: withOpen ? os.current.reads : null,
				manifest: withOpen ? os.manifest.reads : null,
				slabs: withOpen ? os.slabs.reads : null,
				base: withOpen ? os.base.reads : null,
				open: withOpen ? sum("reads", ...openPhases) : null,
				route: tl.route.reads,
				phase1: tl.phase1.reads,
				phase2: tl.phase2.reads,
				query: st.io.reads,
			},
			bytes: {
				pack: withOpen && os.pack ? os.pack.bytes : null,
				current: withOpen ? os.current.bytes : null,
				manifest: withOpen ? os.manifest.bytes : null,
				slabs: withOpen ? os.slabs.bytes : null,
				base: withOpen ? os.base.bytes : null,
				open: withOpen ? sum("bytes", ...openPhases) : null,
				route: tl.route.bytes,
				phase1: tl.phase1.bytes,
				phase2: tl.phase2.bytes,
				query: st.io.bytes,
			},
			warm,
			stats: { scanned: st.scanned, scanBytes: st.scanBytes, slabBytes: st.slabBytes, rerankSlabs: st.rerankSlabs, rerankReads: st.rerankReads, rerankBytes: st.rerankBytes, rerankLocal: st.rerankLocal, specRecords: st.specRecords, specUsed: st.specUsed, cacheHits: st.io.hits, cacheMisses: st.io.misses, coalescedReads: st.coalesced?.reads ?? null, coalescedSlabs: st.coalesced?.slabs ?? null, coalescedBytes: st.coalesced?.bytes ?? null, baseFiles: withOpen ? os.base.files : null },
			proxy,
		});
	};
	const qa = OFFSET + 2 * trial;
	const qb = qa + 1;
	await run("process-cold", qa, true);
	if (WARM_PACK && WARM_WAIT === "slab") {
		await index.warmReady;
		if (useProxy) await proxyCall("POST", "/__reset"); // the rest of the warm pack's GET is not slab-cold's
	}
	await run("slab-cold", qb, false);
	await run("warm", qb, false);
	index.close();
}

// ---------- parent ----------

function runChild(trial) {
	return new Promise((resolve, reject) => {
		const argv = [...process.argv.slice(2).filter((a, i, all) => !(a === "--out" || all[i - 1] === "--out")), "--child", "--trial", String(trial), "--spawn-at", String(Date.now())];
		const p = spawn(process.execPath, [fileURLToPathSafe(import.meta.url), ...argv], { stdio: ["ignore", "pipe", "inherit"] });
		let out = "";
		p.stdout.on("data", (d) => (out += d));
		p.on("error", reject);
		p.on("close", (code) => {
			if (code !== 0) return reject(new Error(`trial ${trial}: child exited with ${code}`));
			resolve(out.split("\n").filter((l) => l.startsWith("@@ROW ")).map((l) => JSON.parse(l.slice(6))));
		});
	});
}
function fileURLToPathSafe(u) {
	return new URL(u).pathname;
}

async function signMicrobench(sdk) {
	const n = 300;
	const t0 = nowMs();
	for (let i = 0; i < n; i++) await sdk.signV4({ method: "GET", path: "/vib-bench/idx/g4/slab/01234.179", headers: { range: `bytes=0-${1000 + i}` }, payloadHash: sdk.EMPTY_SHA256, host: "x.r2.cloudflarestorage.com", region: "auto", accessKeyId: "AKIDEXAMPLE", secretAccessKey: "secret", amzDate: sdk.amzDateOf(Date.now()) });
	return Math.round(((nowMs() - t0) / n) * 1000);
}

async function parent() {
	const sdk = await import(path.join(dist, "s3.js"));
	let proxyConfig = null;
	if (STORE !== "r2s3") proxyConfig = await proxyCall("GET", "/__config");
	const env = loadEnv(ENV_FILE);
	let git = null;
	try {
		git = execSync("git rev-parse --short HEAD", { cwd: root }).toString().trim();
	} catch {}
	const meta = {
		kind: "meta",
		schema: SCHEMA,
		sdk: "ts",
		runtime: `node ${process.version}`,
		store: STORE,
		endpoint: STORE === "r2s3" ? new URL(env.S3_ENDPOINT).host : PROXY,
		packKey: STORE === "pack" ? PACK_KEY : null,
		warmPack: WARM_PACK,
		warmWait: WARM_PACK ? WARM_WAIT : null,
		coalesceAcrossSlabs: COALESCE ?? null,
		proxyConfig,
		cache: "none (Node has no Cache API; thread.open puts no cache in front of the store)",
		n: N,
		queries: { offset: OFFSET, used: `${OFFSET}..${OFFSET + 2 * N - 1}` },
		nprobe: nprobe ?? "manifest",
		rerank: rerank ?? "manifest",
		signUsPerRequest: await signMicrobench(sdk),
		label: args.label ?? null,
		git,
		date: new Date().toISOString(),
	};
	const rows = [meta];
	const out = args.out ? path.resolve(root, args.out) : null;
	if (out) {
		fs.mkdirSync(path.dirname(out), { recursive: true });
		fs.writeFileSync(out, JSON.stringify(meta) + "\n");
	}
	console.log(`cold_bench ts store=${STORE} ${meta.endpoint}${proxyConfig ? ` (rtt ${proxyConfig.rttMs} ms, ${proxyConfig.mbps} Mbps, conn +${proxyConfig.connMs} ms, jitter ${proxyConfig.jitterMs} ms${proxyConfig.linkMbps ? `, link ${proxyConfig.linkMbps} Mbps shared` : ""})` : ""} n=${N} queries ${meta.queries.used}  signV4 ${meta.signUsPerRequest} us/request`);
	for (let t = 0; t < N; t++) {
		const got = await runChild(t);
		for (const r of got) {
			rows.push(r);
			if (out) fs.appendFileSync(out, JSON.stringify(r) + "\n");
		}
		const [pc, sc, w] = got;
		console.log(`  trial ${t}: process-cold e2e ${pc.ms.e2e} ms (open ${pc.ms.open}, query ${pc.ms.query}, ${pc.proxy?.requests ?? pc.reqs.open + pc.reqs.query} req)  slab-cold ${sc.ms.query} ms  warm ${w.ms.query} ms  recall ${pc.recall}/${sc.recall}${got.some((r) => r.proxy?.upstreamFetches) ? "  WARNING: proxy fetched from upstream (not warmed up)" : ""}`);
	}
	console.log("\n" + summary(rows));
	if (out) console.log(`\nwrote ${path.relative(root, out)}`);
}

if (args.child === "true") await child();
else await parent();
