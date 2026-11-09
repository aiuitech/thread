#!/usr/bin/env node
/**
 * Builds the bench index into a local directory, shaped like an inseam host after its segment merges: a dense side
 * (`--docs` vectors of `--dim`, `--cells` cells, compacted into slabs) and `--segments` lexical/docs segments of
 * about `--docs / --segments` documents each, texts stored (about 4 MiB per segment with the defaults).
 *
 *   node bench/blocks/build_index.mjs --out data/blockbench/hybrid8 [--docs 20000 --segments 8 --dim 256 --cells 16]
 *
 * Uses the built SDK (sdk/ts/dist: cd sdk/ts && npm run build). Upload with upload.mjs.
 */
import fs from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import { Thread } from "../../sdk/ts/dist/index.js";
import { FileBlobStore } from "../../sdk/ts/dist/node.js";

const { values: a } = parseArgs({ options: { out: { type: "string" }, docs: { type: "string", default: "20000" }, segments: { type: "string", default: "8" }, dim: { type: "string", default: "256" }, cells: { type: "string", default: "16" }, words: { type: "string", default: "200" } } });
if (!a.out) throw new Error("--out is required");
const N = Number(a.docs), S = Number(a.segments), DIM = Number(a.dim), K = Number(a.cells), WORDS = Number(a.words);
fs.rmSync(a.out, { recursive: true, force: true });
fs.mkdirSync(a.out, { recursive: true });

const here = path.dirname(fileURLToPath(import.meta.url));
const wasm = new WebAssembly.Module(fs.readFileSync(path.join(here, "../../sdk/ts/src/wasm/thread_bg.wasm")));
const thread = await Thread.init({ wasmModule: wasm });

let s = 12345;
const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0), s / 4294967296);
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
const unit = (v) => {
	let n = 0;
	for (const x of v) n += x * x;
	n = Math.sqrt(n);
	for (let i = 0; i < v.length; i++) v[i] /= n;
	return v;
};
const means = new Float32Array(K * DIM);
for (let c = 0; c < K; c++) means.set(unit(Float32Array.from({ length: DIM }, gauss)), c * DIM);
// A Zipf-ish vocabulary: word r is drawn with probability ~ 1 / (r + 10).
const vocab = Array.from({ length: 30000 }, (_, i) => `w${i.toString(36)}x${Math.floor(rnd() * 46656).toString(36)}`);
const word = () => vocab[Math.min(vocab.length - 1, Math.floor(10 * Math.pow(vocab.length / 10 + 1, rnd()) - 10))];

const store = new FileBlobStore(a.out);
await thread.createGeneration({ store, gen: "b1", dim: DIM, means, lex: { docs: true, text: true }, compaction: null });
const ix = await thread.open({ store, cache: false, refreshInterval: 0 });
const per = Math.ceil(N / S);
for (let b = 0; b < S; b++) {
	const n = Math.min(per, N - b * per);
	const vectors = new Float32Array(n * DIM);
	for (let i = 0; i < n; i++) {
		const c = Math.floor(rnd() * K);
		const v = Float32Array.from({ length: DIM }, (_, j) => means[c * DIM + j] + 0.35 * gauss() / Math.sqrt(DIM));
		vectors.set(unit(v), i * DIM);
	}
	const keys = Array.from({ length: n }, (_, i) => `doc/${b * per + i}`);
	const texts = keys.map((k) => `${k} ` + Array.from({ length: WORDS }, word).join(" "));
	const metas = keys.map((k) => JSON.stringify({ address: k, title: `title of ${k}` }));
	const r = await ix.upsert({ keys, vectors, texts, metas });
	console.log(`batch ${b}: ${n} docs, rev ${r.rev}`);
}
for (let i = 0; ; i++) {
	const r = await ix.compact({ force: true, maxBytes: 512 * 1024 * 1024 });
	console.log(`compact ${i}: ${r.outcome}${r.more ? " (more)" : ""}`);
	if (!r.more || r.outcome === "noop") break;
}
await ix.refresh?.();
console.log(JSON.stringify({ gen: ix.gen, rev: ix.rev, segments: ix.segments.map((x) => x.bytes), deltas: ix.manifest.deltas?.length ?? 0, cells: ix.manifest.k2 }));
ix.close();
