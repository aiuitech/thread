#!/usr/bin/env node
/**
 * Workload scenarios over the TypeScript SDK in this process (no Worker, no deployment): each scenario runs one
 * kind of operation many times and summarizes its trace envelopes (docs/observability.md): wall time percentiles,
 * per-span breakdown, store work, kernel CPU and counters. Use it to find where a use case spends its time and to
 * compare a change before and after (`--json` rows).
 *
 *   (cd sdk/ts && npm run build)                  # the scenarios import sdk/ts/dist
 *   node bench/scenarios.mjs                      # a synthetic index (20k docs, dim 256), every scenario
 *   node bench/scenarios.mjs --only dense,hybrid --n 500 --docs 50000 --dim 768
 *   node bench/scenarios.mjs --index data/idx-g4  # an existing index on disk (e.g. a local copy of a bucket)
 *   node bench/scenarios.mjs --latency 15         # every store read waits 15 ms (an I/O-bound picture)
 *   node bench/scenarios.mjs --events --only dense --n 20   # per-read events, printed for the slowest query
 *
 * Options: --docs N, --dim D, --seed S (synthetic index); --store DIR (build the synthetic index into DIR and reuse
 * it on the next run); --n N per scenario (default 200); --concurrency C (default 1: no queueing in the picture);
 * --topK K; --nprobe/--rerank (dense; default: the manifest's); --json FILE (one row per scenario).
 *
 * Scenarios: dense, dense-filtered, lexical, lexical-filtered, hybrid, hybrid-docs, get, listKeys, open, upsert
 * (writes 50 documents per call; runs last). Scenarios the index cannot serve (no lexical side, no keys, no
 * attrs schema) are skipped with a note.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { printSummary, summarize } from "./lib/trace.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "sdk/ts/dist");
if (!fs.existsSync(path.join(dist, "index.js"))) {
	console.error("sdk/ts/dist is missing: cd sdk/ts && npm run build");
	process.exit(1);
}
const { Thread, MemoryBlobStore } = await import(path.join(dist, "index.js"));
const { FileBlobStore } = await import(path.join(dist, "node.js"));

const args = {};
for (let i = 2; i < process.argv.length; i++) {
	const m = process.argv[i].match(/^--([^=\s]+)(?:=(.*))?$/);
	if (!m) continue;
	if (m[2] !== undefined) args[m[1]] = m[2];
	else if (process.argv[i + 1] && !process.argv[i + 1].startsWith("--")) args[m[1]] = process.argv[++i];
	else args[m[1]] = "true";
}
const N = Number(args.n ?? 200);
const C = Number(args.concurrency ?? 1);
const TOPK = Number(args.topK ?? 10);
const DOCS = Number(args.docs ?? 20_000);
const DIM = Number(args.dim ?? 256);
const SEED = Number(args.seed ?? 1);
const LATENCY = Number(args.latency ?? 0);
const only = args.only ? new Set(args.only.split(",")) : null;
const dense = { ...(args.nprobe ? { nprobe: Number(args.nprobe) } : {}), ...(args.rerank ? { rerank: Number(args.rerank) } : {}) };

const wasmModule = new WebAssembly.Module(fs.readFileSync(path.join(dist, "wasm/thread_bg.wasm")));
const thread = await Thread.init({ wasmModule });

// ---- synthetic data ----

function rng(seed) {
	let s = seed >>> 0;
	return () => {
		s = (s + 0x6d2b79f5) >>> 0;
		let t = s;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
const r = rng(SEED);
const gauss = () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
const CENTERS = Array.from({ length: 64 }, () => unit(Float32Array.from({ length: DIM }, gauss)));
function unit(v) {
	let ss = 0;
	for (const x of v) ss += x * x;
	const inv = 1 / Math.sqrt(ss);
	for (let i = 0; i < v.length; i++) v[i] *= inv;
	return v;
}
/** A unit vector near one of the cluster centres (what an embedding corpus looks like to the router). */
const vector = () => {
	const c = CENTERS[Math.floor(r() * CENTERS.length)];
	return unit(Float32Array.from(c, (x) => x + 0.35 * gauss() / Math.sqrt(DIM) * 4));
};
// A Zipf-ish vocabulary: a few very common words, a long tail.
const VOCAB = Array.from({ length: 4000 }, (_, i) => `w${i.toString(36)}${["deploy", "budget", "error", "host", "retry", "index", "query", "cache"][i % 8]}`);
const word = () => VOCAB[Math.min(VOCAB.length - 1, Math.floor(Math.pow(r(), 2.2) * VOCAB.length))];
const text = (n) => Array.from({ length: n }, word).join(" ");
const TENANTS = 16;

class LatencyStore {
	constructor(inner, ms) {
		this.inner = inner;
		this.ms = ms;
		this.remote = true;
		this.inflight = 32;
	}
	wait() {
		return new Promise((res) => setTimeout(res, this.ms));
	}
	async get(k, range, o) {
		await this.wait();
		return this.inner.get(k, range, o);
	}
	async getRanges(k, ranges, o) {
		await this.wait();
		return this.inner.getRanges ? this.inner.getRanges(k, ranges, o) : null;
	}
	async head(k) {
		await this.wait();
		return this.inner.head(k);
	}
	put(k, d, o) {
		return this.inner.put(k, d, o);
	}
	list(p) {
		return this.inner.list(p);
	}
	listObjects(p) {
		return this.inner.listObjects?.(p);
	}
	delete(k) {
		return this.inner.delete?.(k);
	}
}

async function buildSynthetic(store) {
	const t0 = performance.now();
	await thread.createGeneration({ store, gen: "s", dim: DIM, attrs: { tenant: { bits: 5 } }, lex: { text: false } });
	const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
	const batch = 2000;
	for (let a = 0; a < DOCS; a += batch) {
		const n = Math.min(batch, DOCS - a);
		const ids = Array.from({ length: n }, (_, i) => a + i);
		await ix.upsert({
			keys: ids.map((i) => `doc/${String(i).padStart(7, "0")}`),
			vectors: ids.map(() => vector()),
			texts: ids.map(() => text(20 + Math.floor(r() * 40))),
			metas: ids.map((i) => JSON.stringify({ n: i })),
			attrs: ids.map((i) => ({ tenant: i % TENANTS })),
		});
		// Fold the deltas and split the cells as automatic compaction would.
		// maxBytes: this process is not a 128 MB isolate; a wide index needs more than the 48 MiB default per run.
		for (let runs = 0; runs < 20; runs++) {
			const c = await ix.compact({ force: runs === 0, maxBytes: 1 << 30 });
			if (!c.more) break;
		}
		process.stdout.write(`\r  building: ${a + n}/${DOCS} docs, ${ix.manifest.k1 * ix.manifest.k2} cells`);
	}
	await ix.compactSegments();
	process.stdout.write(`\n  built in ${((performance.now() - t0) / 1000).toFixed(1)} s\n`);
	if (ix.manifest.deltas.length) console.log(`  note: ${ix.manifest.deltas.length} delta segments left unfolded (compaction stopped early); dense queries scan them whole`);
	ix.close();
}

// ---- the index ----

let base;
if (args.index) base = new FileBlobStore(path.resolve(args.index));
else if (args.store) {
	base = new FileBlobStore(path.resolve(args.store));
	if (!(await base.head("idx/current.json"))) await buildSynthetic(base);
} else {
	base = new MemoryBlobStore();
	await buildSynthetic(base);
}
const store = LATENCY > 0 ? new LatencyStore(base, LATENCY) : base;
const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
const m = ix.manifest;
const hasLex = m.lex != null;
const hasKeys = hasLex && (await ix.listKeys({ limit: 1 })).entries.length > 0;
const hasAttrs = Boolean(m.attrs?.fields?.length);
const qvec = () => (m.dim === DIM ? vector() : unit(Float32Array.from({ length: m.dim }, gauss)));
const keyOf = (i) => `doc/${String(i % Math.max(1, DOCS)).padStart(7, "0")}`;
const filterOf = () => (hasAttrs ? { [m.attrs.fields[0].name]: Math.floor(r() * TENANTS) } : undefined);
console.log(`index: gen ${ix.gen} rev ${ix.rev}, ${m.vectors ?? "?"} vectors x ${m.dim}, ${m.k1 * m.k2} cells, format ${m.format}, ${m.deltas?.length ?? 0} deltas, ${m.segments?.length ?? 0} segments${hasLex ? "" : ", no lexical side"}${LATENCY ? `, +${LATENCY} ms per read` : ""}`);

const SCENARIOS = [
	{ name: "dense", need: () => ix.hasDense, run: () => ix.query(qvec(), { topK: TOPK, ...dense, traceEvents: args.events === "true" }).then((x) => x.stats.trace) },
	{ name: "dense-filtered", need: () => ix.hasDense && hasAttrs, run: () => ix.query(qvec(), { topK: TOPK, ...dense, filter: filterOf() }).then((x) => x.stats.trace) },
	{ name: "lexical", need: () => hasLex, run: () => ix.lexicalSearch(`${word()} ${word()} ${word()}`, TOPK).then((x) => x.stats.trace) },
	{ name: "lexical-filtered", need: () => hasLex && hasAttrs, run: () => ix.lexicalSearch(`${word()} ${word()}`, TOPK, { filter: ix.compileFilter(filterOf()) }).then((x) => x.stats.trace) },
	{ name: "hybrid", need: () => hasLex && ix.hasDense, run: () => ix.search({ text: `${word()} ${word()}`, vector: qvec(), topK: TOPK, dense }).then((x) => x.stats.trace) },
	{ name: "hybrid-docs", need: () => hasLex && ix.hasDense, run: () => ix.search({ text: `${word()} ${word()}`, vector: qvec(), topK: TOPK, dense, docs: true }).then((x) => x.stats.trace) },
	{ name: "get", need: () => hasKeys, run: () => ix.get([keyOf(Math.floor(r() * DOCS)), keyOf(Math.floor(r() * DOCS)), keyOf(Math.floor(r() * DOCS))]).then((x) => x.trace) },
	{ name: "listKeys", need: () => hasKeys, run: () => ix.listKeys({ prefix: `doc/00${Math.floor(r() * 10)}`, limit: 50 }).then((x) => x.trace) },
	{
		name: "open",
		need: () => true,
		run: async () => {
			const o = await thread.open({ store, cache: false, refreshInterval: 0 });
			const t = o.openTrace;
			o.close();
			return t;
		},
	},
	{
		name: "upsert",
		need: () => hasKeys && ix.hasDense && m.dim === DIM,
		run: () => {
			const pick = new Set();
			while (pick.size < Math.min(50, DOCS)) pick.add(Math.floor(r() * DOCS));
			const ids = [...pick];
			return ix.upsert({ keys: ids.map(keyOf), vectors: ids.map(() => vector()), texts: ids.map(() => text(30)), attrs: ids.map((i) => ({ tenant: i % TENANTS })) }).then((x) => x.trace);
		},
	},
];

async function runScenario(s) {
	const n = s.name === "upsert" || s.name === "open" ? Math.min(N, 50) : N;
	for (let i = 0; i < Math.min(5, n); i++) await s.run(); // warm: JIT, demand-paged lexical pages, the first base load
	const out = new Array(n);
	let next = 0;
	const t0 = performance.now();
	await Promise.all(
		Array.from({ length: Math.min(C, n) }, async () => {
			for (let i = next++; i < n; i = next++) out[i] = await s.run();
		}),
	);
	const wall = (performance.now() - t0) / 1000;
	return { out, wall, n };
}

for (const s of SCENARIOS) {
	if (only && !only.has(s.name)) continue;
	if (!s.need()) {
		console.log(`\n[${s.name}] skipped: the index cannot serve it`);
		continue;
	}
	const { out, wall, n } = await runScenario(s);
	const sum = summarize(out);
	console.log(`\n[${s.name}] ${n} ops at concurrency ${C}, ${(n / wall).toFixed(0)} ops/s`);
	printSummary(sum);
	if (args.events === "true") {
		const slow = [...out].sort((a, b) => b.ms - a.ms)[0];
		if (slow?.events) {
			console.log(`  slowest op (${slow.ms} ms), its reads:`);
			for (const e of slow.events) console.log(`    @${e.start.toFixed(2).padStart(8)} ${e.ms.toFixed(2).padStart(7)} ms  ${e.kind.padEnd(8)} ${String(e.bytes).padStart(8)} B  ${e.ranges} range(s)  ${e.key}`);
		}
	}
	if (args.json) fs.appendFileSync(args.json, JSON.stringify({ at: new Date().toISOString(), scenario: s.name, n, concurrency: C, latency: LATENCY, gen: ix.gen, rev: ix.rev, opsPerSec: +(n / wall).toFixed(1), ...sum }) + "\n");
}
ix.close();
