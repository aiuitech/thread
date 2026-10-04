#!/usr/bin/env node
/**
 * Cold-start bench of the TypeScript SDK (Node) against S3: real R2 S3, the fake_s3.mjs proxy, or a pack served
 * by the proxy. bench/cold/cold_bench.py is the same bench for the Python SDK (same CLI, same JSONL schema).
 *
 *   node bench/cold/cold_bench.mjs --store r2s3|proxy|pack [--n 10] [--queries 0] [--nprobe N] [--rerank R]
 *        [--proxy http://127.0.0.1:9555] [--pack-key packs/g4-cold.threadpack] [--label TEXT]
 *        [--out data/cold/NAME.jsonl] [--env sdk/ts/examples/basic/.env.secrets]
 *        [--routing f32|f16|i8|auto] [--cold-routing tree] [--cold-beam N] [--base-pack 0|1] [--gen G --rev R]
 *
 * The open options are thread.open's: `--routing` (routing table), `--cold-routing tree` (+ `--cold-beam`: route the
 * first queries over the tree's groups while the full table loads), `--base-pack 0` (no base pack, no prefetch: the
 * reads of the open one after another, the behaviour before base packs), `--gen/--rev` (pin a revision).
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
const dist = path.join(root, "sdk/ts/dist");
/** thread.open options from the command line. */
const openOpts = () => ({
	...(args.routing ? { routing: args.routing } : {}),
	...(args["cold-routing"] ? { coldRouting: args["cold-routing"] } : {}),
	...(args["cold-beam"] ? { coldBeam: Number(args["cold-beam"]) } : {}),
	...(args["base-pack"] !== undefined ? { basePack: args["base-pack"] !== "0" } : {}),
	...(args.gen ? { gen: args.gen } : {}),
	...(args.rev !== undefined ? { rev: Number(args.rev) } : {}),
});

const nowMs = () => performance.now();
const r1 = (x) => (x === null || x === undefined ? null : Math.round(x * 10) / 10);

async function proxyCall(method, p) {
	const r = await fetch(PROXY + p, { method });
	if (!r.ok) throw new Error(`proxy ${method} ${p}: HTTP ${r.status}`);
	return r.json();
}
const proxyStats = (s) => (s ? { requests: s.requests, bytes: s.bytes, newConnections: s.newConnections, upstreamFetches: s.upstreamFetches, byStatus: s.byStatus } : null);

// ---------- child: one trial ----------

async function child() {
	const tEntry = Date.now();
	const spawnAt = Number(args["spawn-at"]);
	const trial = Number(args.trial);
	const useProxy = STORE !== "r2s3";
	if (useProxy) await proxyCall("POST", "/__reset");
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
	const s3 = useProxy
		? new sdk.S3BlobStore({ endpoint: PROXY, bucket: env.S3_BUCKET ?? "vib-bench", region: "auto", accessKeyId: "fake", secretAccessKey: "fake" })
		: new sdk.S3BlobStore({ endpoint: env.S3_ENDPOINT, bucket: env.S3_BUCKET, region: "auto", accessKeyId: env.S3_ACCESS_KEY_ID, secretAccessKey: env.S3_SECRET_ACCESS_KEY });
	const store = STORE === "pack" ? await sdk.PackedBlobStore.open(s3, PACK_KEY) : s3;
	// No `cache` option: what a Node user gets by default (the Cache API does not exist in Node, so: none).
	const index = await thread.open({ store, ...openOpts() });
	const os = index.openStats;
	const qopts = { topK: 10, ...(nprobe !== undefined ? { nprobe } : {}), ...(rerank !== undefined ? { rerank } : {}) };
	const emit = (row) => process.stdout.write("@@ROW " + JSON.stringify(row) + "\n");
	const run = async (scenario, qi, withOpen) => {
		const t0 = nowMs();
		const res = await index.query(ds.vector(qi), qopts);
		const wall = nowMs() - t0;
		const e2e = withOpen ? Date.now() - spawnAt : null;
		const proxy = useProxy ? proxyStats(await proxyCall("POST", "/__reset")) : null;
		const st = res.stats;
		const tl = st.timeline;
		const ids = res.hits.map((h) => h.id);
		const sum = (f, ...ps) => ps.reduce((a, p) => a + (p ? p[f] : 0), 0);
		const openPhases = withOpen ? [os.pack, os.prefetch, os.current, os.manifest, os.slabs, os.base] : [];
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
			routing: st.routing,
			routingTable: st.routingTable,
			cells: st.cells,
			cold: st.cold ?? null,
			prefetchSource: withOpen ? os.prefetch?.source ?? null : null,
			ms: {
				boot: withOpen ? ms.boot : null,
				import: withOpen ? r1(ms.import) : null,
				wasmCompile: withOpen ? r1(ms.wasmCompile) : null,
				wasmInit: withOpen ? r1(ms.wasmInit) : null,
				pack: withOpen && os.pack ? r1(os.pack.ms) : null,
				prefetch: withOpen && os.prefetch ? r1(os.prefetch.ms) : null,
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
				prefetch: withOpen && os.prefetch ? os.prefetch.reads : null,
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
				prefetch: withOpen && os.prefetch ? os.prefetch.bytes : null,
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
			stats: { scanned: st.scanned, scanBytes: st.scanBytes, slabBytes: st.slabBytes, rerankSlabs: st.rerankSlabs, rerankReads: st.rerankReads, rerankBytes: st.rerankBytes, rerankLocal: st.rerankLocal, specRecords: st.specRecords, specUsed: st.specUsed, cacheHits: st.io.hits, cacheMisses: st.io.misses, baseFiles: withOpen ? os.base.files : null },
			proxy,
		});
	};
	const qa = OFFSET + 2 * trial;
	const qb = qa + 1;
	await run("process-cold", qa, true);
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
		proxyConfig,
		cache: "none (Node has no Cache API; thread.open puts no cache in front of the store)",
		n: N,
		queries: { offset: OFFSET, used: `${OFFSET}..${OFFSET + 2 * N - 1}` },
		nprobe: nprobe ?? "manifest",
		rerank: rerank ?? "manifest",
		signUsPerRequest: await signMicrobench(sdk),
		label: args.label ?? null,
		open: openOpts(),
		git,
		date: new Date().toISOString(),
	};
	const rows = [meta];
	const out = args.out ? path.resolve(root, args.out) : null;
	if (out) {
		fs.mkdirSync(path.dirname(out), { recursive: true });
		fs.writeFileSync(out, JSON.stringify(meta) + "\n");
	}
	console.log(`cold_bench ts store=${STORE} ${meta.endpoint}${proxyConfig ? ` (rtt ${proxyConfig.rttMs} ms, ${proxyConfig.mbps} Mbps, conn +${proxyConfig.connMs} ms, jitter ${proxyConfig.jitterMs} ms)` : ""} n=${N} queries ${meta.queries.used}  signV4 ${meta.signUsPerRequest} us/request`);
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
