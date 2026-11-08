#!/usr/bin/env node
/**
 * Measured peak memory of one SDK dense compaction against the planner's estimate (docs/update-path.md "Memory of a
 * call"). Needs the built SDK (`cd sdk/ts && npm run build`).
 *
 *   node bench/compact_memory.mjs prep <dir> <entries> [dim 1024] [batch 200]   # a one-cell generation c1 with deltas
 *   node --expose-gc bench/compact_memory.mjs measure <dir> fold|split [gen c1] [maxMiB 2048]
 *   node bench/compact_memory.mjs all <dir> [dim 1024]                         # prep + measure 1000/3000/6000, fold and split
 *
 * `measure` compacts once with `force` (fold: maxCellEntries 0; split: 2), reading the directory and writing to memory
 * (the directory is left as it was), and samples the JS ArrayBuffer bytes and the wasm memory after every wasm call,
 * store read and write, and every millisecond. Run it in a fresh process: the wasm memory never shrinks, so its growth
 * over the call is exact only from a small start. Any generation directory works (`gen`), e.g. a copy of an R2 index.
 */
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const SDK = path.join(here, "../sdk/ts/dist");
const T = await import(path.join(SDK, "index.js"));
const N = await import(path.join(SDK, "node.js"));
const thread = await T.Thread.init({ wasmModule: new WebAssembly.Module(fs.readFileSync(path.join(SDK, "wasm/thread_bg.wasm"))) });
const MB = (b) => +(b / 1048576).toFixed(1);

const probe = {
	peak: 0,
	base: 0,
	sample() {
		const t = process.memoryUsage().arrayBuffers + thread.memoryBytes();
		if (t > this.peak) this.peak = t;
	},
};

/** The wasm bindings with every function and job method sampled on return. */
function probedWasm(W) {
	const out = {};
	for (const [name, v] of Object.entries(W)) {
		if (typeof v !== "function") out[name] = v;
		else if (/^[A-Z]/.test(name)) {
			for (const m of Object.getOwnPropertyNames(v.prototype)) {
				const f = Object.getOwnPropertyDescriptor(v.prototype, m)?.value;
				if (typeof f !== "function" || m === "constructor") continue;
				v.prototype[m] = function (...a) {
					const r = f.apply(this, a);
					probe.sample();
					return r;
				};
			}
			out[name] = v;
		} else
			out[name] = (...a) => {
				const r = v(...a);
				probe.sample();
				return r;
			};
	}
	return out;
}

/** Reads a directory, writes to memory. */
class OverlayStore {
	constructor(root) {
		this.file = new N.FileBlobStore(root);
		this.mem = new T.MemoryBlobStore();
	}
	async get(key, range) {
		const r = this.mem.objects.has(key) ? await this.mem.get(key, range) : await this.file.get(key, range).catch((e) => (String(e).includes("ENOENT") ? null : Promise.reject(e)));
		probe.sample();
		return r;
	}
	async head(key) {
		return (await this.mem.head(key)) ?? (await this.file.head(key));
	}
	async put(key, data, opts = {}) {
		if (opts.ifAbsent && (await this.head(key))) return null;
		const r = await this.mem.put(key, data, { contentType: opts.contentType });
		probe.sample();
		return r;
	}
	async list(prefix) {
		return [...new Set([...(await this.file.list(prefix)), ...(await this.mem.list(prefix))])].sort();
	}
	async delete(key) {
		await this.mem.delete(key);
	}
}

async function prep(dir, entries, dim = 1024, batch = 200) {
	const store = new N.FileBlobStore(dir);
	await thread.createGeneration({ store, gen: "c1", dim, lex: null, compaction: { maxDeltas: 1_000_000, maxCellEntries: 0 } });
	const ix = await thread.open({ store, gen: "c1", cache: false, refreshInterval: 0 });
	let s = 7;
	const rnd = () => (s = (s * 1103515245 + 12345) >>> 0) / 4294967296 - 0.5;
	const centers = Array.from({ length: 16 }, () => Float32Array.from({ length: dim }, rnd));
	for (let a = 0; a < entries; a += batch) {
		const n = Math.min(batch, entries - a);
		const v = new Float32Array(n * dim);
		for (let i = 0; i < n; i++) {
			const c = centers[Math.floor((rnd() + 0.5) * centers.length)];
			let ss = 0;
			for (let d = 0; d < dim; d++) ss += (v[i * dim + d] = c[d] + 0.6 * rnd()) ** 2;
			for (let d = 0; d < dim; d++) v[i * dim + d] /= Math.sqrt(ss);
		}
		await ix.upsert({ ids: Array.from({ length: n }, (_, i) => a + i), vectors: v });
	}
	ix.close();
}

async function measure(dir, mode, gen = "c1", maxMiB = 2048) {
	global.gc?.();
	probe.base = process.memoryUsage().arrayBuffers + thread.memoryBytes();
	probe.peak = probe.base;
	const timer = setInterval(() => probe.sample(), 1);
	const store = new OverlayStore(dir);
	const ctx = { thread, w: probedWasm(thread.wasm), store, gen, borrowPq: () => null, resolveKeys: async () => new Map(), listKeys: async () => [] };
	const r = await T.compactDeltas(ctx, { force: true, maxBytes: maxMiB * 1048576, ...(mode === "split" ? { maxCellEntries: 2 } : mode === "fold" ? { maxCellEntries: 0 } : {}), concurrency: 1, record: false, lock: false });
	clearInterval(timer);
	return { mode, outcome: r.outcome, folded: r.folded, splits: r.splits.map((x) => [x.entries, x.a, x.b]), writtenMB: MB(r.writtenBytes), estimatedMB: MB(r.estimatedPeakBytes), measuredMB: MB(probe.peak - probe.base), wasmMB: MB(thread.memoryBytes()) };
}

const [cmd, dir, ...rest] = process.argv.slice(2);
if (cmd === "prep") await prep(dir, Number(rest[0]), Number(rest[1] ?? 1024), Number(rest[2] ?? 200));
else if (cmd === "measure") console.log(JSON.stringify(await measure(dir, rest[0], rest[1], rest[2] ? Number(rest[2]) : undefined)));
else if (cmd === "all") {
	for (const n of [1000, 3000, 6000]) {
		const d = path.join(dir, String(n));
		if (!fs.existsSync(d)) await prep(d, n, Number(rest[0] ?? 1024));
		for (const mode of ["fold", "split"]) {
			const p = spawnSync(process.execPath, ["--expose-gc", fileURLToPath(import.meta.url), "measure", d, mode], { encoding: "utf8" });
			if (p.status !== 0) throw new Error(p.stderr);
			const m = JSON.parse(p.stdout.trim().split("\n").pop());
			console.log(`${n} entries, ${mode}: estimated ${m.estimatedMB} MB, measured ${m.measuredMB} MB${m.measuredMB > m.estimatedMB ? "  OVER" : ""}`);
		}
	}
} else {
	console.error("usage: compact_memory.mjs prep <dir> <entries> [dim] [batch] | measure <dir> fold|split [gen] [maxMiB] | all <dir> [dim]");
	process.exit(2);
}
