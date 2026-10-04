#!/usr/bin/env node
/**
 * Cold-start bench of the TypeScript SDK (Node) against S3: real R2 S3, the fake_s3.mjs proxy, or a pack served
 * by the proxy. bench/cold/cold_bench.py is the same bench for the Python SDK (same CLI, same JSONL schema).
 *
 *   node bench/cold/cold_bench.mjs --store r2s3|proxy|pack [--n 10] [--queries 0] [--nprobe N] [--rerank R]
 *        [--proxy http://127.0.0.1:9555] [--pack-key packs/g4-cold.threadpack] [--label TEXT]
 *        [--out data/cold/NAME.jsonl] [--env sdk/ts/examples/basic/.env.secrets]
 *        store / cache options (absent = the SDK's defaults):
 *        [--legacy] (the pre-cold-store defaults: no cache, inflight 12, no warm-up)
 *        [--cache none|memory|disk] [--disk-dir data/cold/dc-{t}] [--miss-policy whole|range|range+warm]
 *        [--inflight N] [--coalesce BYTES] [--span BYTES] [--warmup N|false] [--cold-nprobe N] [--cold-rerank R]
 *        [--keepalive MS] [--h2] [--drain] (wait for background cache warms before the `warm` scenario)
 *        [--early-warmup] (create the store with `warmup` before the wasm compiles)
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
const num = (k) => (args[k] !== undefined ? Number(args[k]) : undefined);
const LEGACY = args.legacy === "true";
/** The store / cache / query configuration under test (recorded in every row). */
const config = {
	legacy: LEGACY,
	cache: args.cache ?? (LEGACY ? "none" : "default"),
	diskDir: args["disk-dir"] ?? null,
	missPolicy: args["miss-policy"] ?? null,
	inflight: num("inflight") ?? (LEGACY ? 12 : null),
	coalesceBytes: num("coalesce") ?? null,
	spanBytes: num("span") ?? null,
	warmup: args.warmup === "false" ? 0 : (num("warmup") ?? (LEGACY ? 0 : null)),
	coldPolicy: args["cold-nprobe"] || args["cold-rerank"] ? { ...(num("cold-nprobe") !== undefined ? { nprobe: num("cold-nprobe") } : {}), ...(num("cold-rerank") !== undefined ? { rerank: num("cold-rerank") } : {}) } : null,
	keepAliveMs: num("keepalive") ?? null,
	h2: args.h2 === "true",
	drain: args.drain === "true",
	earlyWarmup: args["early-warmup"] === "true",
};

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
	const node = await import(path.join(dist, "node.js"));
	const makeStore = async () => {
		const fetchFn = config.keepAliveMs !== null || config.h2 ? await node.nodeFetch({ keepAliveTimeout: config.keepAliveMs ?? 4000, allowH2: config.h2 }) : undefined;
		const early = config.earlyWarmup ? { warmup: config.inflight ?? 32 } : {};
		const s3opts = { ...(fetchFn ? { fetch: fetchFn } : {}), ...(config.inflight !== null ? { inflight: config.inflight } : {}), ...(config.coalesceBytes !== null ? { coalesceBytes: config.coalesceBytes } : {}), ...(config.spanBytes !== null ? { spanBytes: config.spanBytes } : {}), ...early };
		if (useProxy) await proxyCall("POST", "/__reset"); // nothing before this talks to the store
		return useProxy
			? new sdk.S3BlobStore({ endpoint: PROXY, bucket: env.S3_BUCKET ?? "vib-bench", region: "auto", accessKeyId: "fake", secretAccessKey: "fake", ...s3opts })
			: new sdk.S3BlobStore({ endpoint: env.S3_ENDPOINT, bucket: env.S3_BUCKET, region: "auto", accessKeyId: env.S3_ACCESS_KEY_ID, secretAccessKey: env.S3_SECRET_ACCESS_KEY, ...s3opts });
	};
	// --early-warmup: the store (and its connection warm-up) is created before the wasm compiles, so the handshakes
	// and Node's lazy fetch globals overlap the wasm compile + init.
	const earlyStore = config.earlyWarmup ? await makeStore() : null;
	t = nowMs();
	const wasmModule = new WebAssembly.Module(fs.readFileSync(path.join(dist, "wasm/thread_bg.wasm")));
	ms.wasmCompile = nowMs() - t;
	const thread = await sdk.Thread.init({ wasmModule });
	ms.wasmInit = thread.initMs;
	const s3 = earlyStore ?? (await makeStore());
	const store = STORE === "pack" ? await sdk.PackedBlobStore.open(s3, PACK_KEY) : s3;
	// No options: what a Node user gets by default (a MemoryCache with "range+warm" in front of a remote store).
	const cache = config.cache === "none" ? false : config.cache === "memory" ? new sdk.MemoryCache() : config.cache === "disk" ? new node.DiskCache(path.resolve(root, (config.diskDir ?? "data/cold-store/dc-{t}").replace("{t}", String(trial)))) : undefined;
	const index = await thread.open({
		store,
		...(cache !== undefined ? { cache } : {}),
		...(config.missPolicy ? { missPolicy: config.missPolicy } : {}),
		...(config.warmup !== null ? { warmup: config.warmup } : {}),
		...(config.coldPolicy ? { coldPolicy: config.coldPolicy } : {}),
	});
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
			config,
			stats: { wholeSlabs: st.wholeSlabs, warms: st.io.warms ?? 0, inflight: st.inflight, coalesceBytes: st.coalesceBytes, spanBytes: st.spanBytes, cold: st.cold ?? null, scanned: st.scanned, scanBytes: st.scanBytes, slabBytes: st.slabBytes, rerankSlabs: st.rerankSlabs, rerankReads: st.rerankReads, rerankBytes: st.rerankBytes, rerankLocal: st.rerankLocal, specRecords: st.specRecords, specUsed: st.specUsed, cacheHits: st.io.hits, cacheMisses: st.io.misses, baseFiles: withOpen ? os.base.files : null },
			proxy,
		});
	};
	const qa = OFFSET + 2 * trial;
	const qb = qa + 1;
	await run("process-cold", qa, true);
	await run("slab-cold", qb, false);
	if (config.drain && index.store.drain) {
		await index.store.drain();
		if (useProxy) await proxyCall("POST", "/__reset"); // background warm traffic is not the warm query's
	}
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

/** SigV4 cost per request (us), deriving the signing key every time or with the store's `SigningKeyCache`. */
async function signMicrobench(sdk, cached = false) {
	const n = 300;
	const keys = cached && sdk.SigningKeyCache ? new sdk.SigningKeyCache("secret") : undefined;
	const t0 = nowMs();
	for (let i = 0; i < n; i++) await sdk.signV4({ method: "GET", path: "/vib-bench/idx/g4/slab/01234.179", headers: { range: `bytes=0-${1000 + i}` }, payloadHash: sdk.EMPTY_SHA256, host: "x.r2.cloudflarestorage.com", region: "auto", accessKeyId: "AKIDEXAMPLE", secretAccessKey: "secret", amzDate: sdk.amzDateOf(Date.now()) }, keys);
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
		cache: config.cache,
		config,
		n: N,
		queries: { offset: OFFSET, used: `${OFFSET}..${OFFSET + 2 * N - 1}` },
		nprobe: nprobe ?? "manifest",
		rerank: rerank ?? "manifest",
		signUsPerRequest: await signMicrobench(sdk),
		signUsPerRequestCached: await signMicrobench(sdk, true),
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
	console.log(`cold_bench ts store=${STORE} ${meta.endpoint}${proxyConfig ? ` (rtt ${proxyConfig.rttMs} ms, ${proxyConfig.mbps} Mbps, conn +${proxyConfig.connMs} ms, jitter ${proxyConfig.jitterMs} ms${proxyConfig.linkMbps ? `, link ${proxyConfig.linkMbps} Mbps shared` : ""})` : ""} n=${N} queries ${meta.queries.used}  signV4 ${meta.signUsPerRequest} us/request (${meta.signUsPerRequestCached} with the key cache)  config ${JSON.stringify(config)}`);
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
