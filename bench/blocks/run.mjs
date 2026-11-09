#!/usr/bin/env node
/**
 * Cold and warm searches through the bench Workers (bench/blocks/README.md), one line per search and a median
 * table per configuration.
 *
 *   BENCH_TOKEN=... node bench/blocks/run.mjs --url old=https://thread-blockbench-old.<sub>.workers.dev \
 *       --url new=https://thread-blockbench-new.<sub>.workers.dev --index hybrid8 --blocks 262144,1048576 --reps 3
 *
 * Per repetition and configuration: a fresh cache origin (cold), the same search again after the background block
 * puts (warm, twice), and a search for other terms on the same origin (partly warm). `--mode dense` searches by
 * vector only, `lexical` by text only, `hybrid` (default) by both. Queries are made from the local copy of the index
 * (--data, default data/blockbench/<index>).
 */
import fs from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import { Thread } from "../../sdk/ts/dist/index.js";
import { FileBlobStore } from "../../sdk/ts/dist/node.js";

const { values: a } = parseArgs({ options: { url: { type: "string", multiple: true }, index: { type: "string", default: "hybrid8" }, data: { type: "string" }, blocks: { type: "string", default: "" }, reps: { type: "string", default: "3" }, mode: { type: "string", default: "hybrid" }, docs: { type: "string", default: "text" }, pause: { type: "string", default: "4000" }, out: { type: "string" } } });
const token = process.env.BENCH_TOKEN;
const urls = (a.url ?? []).map((u) => u.split("=", 2));
const here = path.dirname(fileURLToPath(import.meta.url));
const data = a.data ?? path.join(here, "../../data/blockbench", a.index);

// Queries: two rare-ish words of one stored document and its key; a vector near a cell mean.
const thread = await Thread.init({ wasmModule: new WebAssembly.Module(fs.readFileSync(path.join(here, "../../sdk/ts/src/wasm/thread_bg.wasm"))) });
const local = await thread.open({ store: new FileBlobStore(data), cache: false });
const query = async (n) => {
	const key = `doc/${(n * 7919) % 2000}`;
	const doc = (await local.get([key], { text: true })).records.get(key);
	const words = doc.text.split(" ").slice(1);
	const text = `${words[3]} ${words[11]} ${words[40]}`;
	const v = await local.embed?.([text]).catch(() => null);
	let s = n + 1;
	const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0), s / 4294967296 - 0.5);
	const vector = v?.[0] ? [...v[0]] : Array.from({ length: local.dim }, rnd);
	return a.mode === "dense" ? { vector } : a.mode === "lexical" ? { text } : { text, vector };
};

const call = async (url, body) => {
	const r = await fetch(`${url}/search`, { method: "POST", headers: { authorization: `Bearer ${token}`, "content-type": "application/json" }, body: JSON.stringify(body) });
	const j = await r.json();
	if (!r.ok) throw new Error(JSON.stringify(j));
	return j;
};

const row = (label, run, r) => {
	const sp = Object.fromEntries(r.spans.map((s) => [s.name, s]));
	const f = (n) => (sp[n] ? { ms: sp[n].ms, ...sp[n].io } : null);
	return { label, run, open: { ms: r.open.ms, ...r.open.io, r2: r.open.r2.gets }, dense: f("dense"), lexical: f("lexical"), docs: f("docs"), search: { ms: r.search.ms, ...r.search.io }, r2: r.r2, blocks: r.blocks, hits: r.hits };
};

const results = [];
const configs = [];
for (const [label, url] of urls) {
	const bs = label === "old" || !a.blocks ? [undefined] : a.blocks.split(",").map(Number);
	for (const b of bs) configs.push({ label: b === undefined ? label : `${label}/B=${b >> 10}K`, url, b });
}
const stamp = Date.now().toString(36);
for (let rep = 0; rep < Number(a.reps); rep++) {
	for (const c of configs) {
		const origin = `https://bb-${stamp}-${rep}-${c.label.replace(/[^a-z0-9]/gi, "")}.cache/`;
		const base = { prefix: `tmp/blockbench/${a.index}/`, cacheOrigin: origin, topK: 20, recallK: 100, docs: a.docs === "none" ? false : a.docs, ...(c.b !== undefined ? { blockBytes: c.b } : {}) };
		const q0 = await query(rep);
		const q1 = await query(rep + 100);
		for (const [run, q] of [["cold", q0], ["warm", q0], ["warm2", q0], ["newq", q1]]) {
			const r = await call(c.url, { ...base, ...q });
			const x = row(c.label, run, r);
			results.push(x);
			console.log(JSON.stringify(x));
			await new Promise((res) => setTimeout(res, Number(a.pause)));
		}
	}
}
if (a.out) fs.writeFileSync(a.out, JSON.stringify(results, null, 1));

// Medians per (config, run).
const med = (xs) => {
	const s = xs.filter((x) => typeof x === "number").sort((p, q) => p - q);
	return s.length ? s[Math.floor((s.length - 1) / 2)] : "";
};
console.log(`\n${a.index} (${a.mode}), medians of ${a.reps}:`);
console.log("config\trun\topen ms\tdense ms\tdense hit/miss/fill\tlexical ms\tlexical hit/miss/fill\tdocs ms\tdocs hit/miss/fill\tsearch ms\tR2 gets\tR2 MB");
for (const c of configs) {
	for (const run of ["cold", "warm", "warm2", "newq"]) {
		const xs = results.filter((x) => x.label === c.label && x.run === run);
		const p = (k, f) => med(xs.map((x) => x[k]?.[f]));
		const hmf = (k) => (xs[0]?.[k] ? `${p(k, "hits")}/${p(k, "misses")}/${p(k, "fills")}` : "");
		console.log([c.label, run, p("open", "ms"), p("dense", "ms"), hmf("dense"), p("lexical", "ms"), hmf("lexical"), p("docs", "ms"), hmf("docs"), p("search", "ms"), med(xs.map((x) => x.r2.gets)), (med(xs.map((x) => x.r2.bytes)) / 1e6).toFixed(2)].join("\t"));
	}
}
local.close();
