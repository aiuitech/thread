// The TypeScript half of tests/test_interop.py, over the built SDK (sdk/ts/dist) and a directory store:
//   node ts_side.mjs build <dir>   builds a format-3 generation with deltas, lexical segments and tombstones written
//                                  by the TS SDK, and records queries and their TS results in <dir>/ts-results.json
//   node ts_side.mjs run <dir> [store]   answers the requests in <dir>/requests.json (written by Python) against the
//                                  index at <store> (a directory or a pack file; default <dir>) and prints them
//   node ts_side.mjs pack <dir> <file>    packs the current revision of the index in <dir> into <file>
//   node ts_side.mjs basepack <dir> <file> <kind>   writes the base pack (kind base | cold) of the current revision
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.join(here, "../../../ts/dist");
const { Thread, keys, buildBasePack, loadRevState, resolveTarget } = await import(path.join(dist, "index.js"));
const { FileBlobStore, packIndexToFile, storeAt } = await import(path.join(dist, "node.js"));
const wasmModule = new WebAssembly.Module(fs.readFileSync(path.join(dist, "wasm/thread_bg.wasm")));
const thread = await Thread.init({ wasmModule });
const w = thread.wasm;

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
const gaussian = (r) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
function unitRows(n, dim, r, centers, spread = 0.5) {
	const out = new Float32Array(n * dim);
	for (let i = 0; i < n; i++) {
		const c = centers ? Math.floor(r() * (centers.length / dim)) : -1;
		let ss = 0;
		for (let d = 0; d < dim; d++) {
			const v = gaussian(r) * (c >= 0 ? spread : 1) + (c >= 0 ? centers[c * dim + d] : 0);
			out[i * dim + d] = v;
			ss += v * v;
		}
		for (let d = 0; d < dim; d++) out[i * dim + d] /= Math.sqrt(ss);
	}
	return out;
}
const f32bytes = (f) => new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
const dot = (a, b, ao, bo, dim) => {
	let s = 0;
	for (let d = 0; d < dim; d++) s += a[ao + d] * b[bo + d];
	return s;
};

const DIM = 32;
const K2 = 16;
const N = 1500;

/** Builds a format-3 generation like the Worker's publish (tests/helpers buildGen, larger). */
async function buildGen(store, gen) {
	const r = rng(42);
	const fine = unitRows(K2, DIM, r);
	const vectors = unitRows(N, DIM, r, fine, 0.6);
	const coarse = new Float32Array(DIM).fill(1 / Math.sqrt(DIM));
	const job = new w.AssignJob(DIM, 1, K2, coarse, fine, 1, 2);
	const assign = new Uint32Array(N);
	for (let i = 0; i < N; i++) {
		let best = -1;
		let bs = -Infinity;
		for (let c = 0; c < K2; c++) {
			const s = dot(vectors, fine, i * DIM, c * DIM, DIM);
			if (s > bs) (bs = s), (best = c);
		}
		assign[i] = best;
	}
	const pqM = 8;
	const codebook = w.pq_init_codebook(w.residuals_for(vectors, fine, assign, DIM), DIM, pqM, 7);
	job.set_codebook(codebook);
	job.add_batch(vectors, 0);
	const part = job.take_part(0);
	job.free();
	const dir = w.part_directory(part);
	const rows = [];
	for (let cell = 0; cell < K2; cell++) {
		const [count, off, len] = [dir[2 + cell * 3], dir[3 + cell * 3], dir[4 + cell * 3]];
		const sj = new w.SlabJob(DIM, cell, fine.subarray(cell * DIM, (cell + 1) * DIM), 3);
		if (len && sj.add_records(part.subarray(off, off + len)) !== count) throw new Error("records");
		const bytes = sj.finish();
		const info = w.slab_info(bytes);
		rows.push([info[5], info[6], info[1], bytes.byteLength]);
		await store.put(keys.slab(gen, cell), bytes);
	}
	await store.put(`idx/${gen}/coarse.f32`, f32bytes(coarse));
	await store.put(`idx/${gen}/fine.f32`, f32bytes(fine));
	await store.put(`idx/${gen}/pq.bin`, codebook);
	await store.put(`idx/${gen}/slabs.json`, JSON.stringify(rows));
	const manifest = { gen, k1: 1, k2: K2, dim: DIM, m: 2, probeCoarse: 1, vectors: N, format: 3, tiers: `pq${pqM}+sq8`, pq: { m: pqM }, defaults: { nprobe: 4, rerank: 40, pc: 0 } };
	await store.put(keys.manifest(gen, 0), JSON.stringify(manifest));
	await store.put(keys.current(), JSON.stringify({ gen, rev: 0 }));
	return vectors;
}

const WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima"];
const textOf = (i) => `${WORDS[i % 12]} ${WORDS[(i * 7) % 12]} ${WORDS[(i * 5 + 3) % 12]} item${i % 40} deployments ${i % 3 ? "budget" : "review"}`;

/** Runs one request against an index; the shape Python compares. */
async function answer(index, req) {
	if (req.kind === "query") {
		const r = await index.query(Float32Array.from(req.vector), { topK: req.topK, nprobe: req.nprobe, rerank: req.rerank, returnPayload: true });
		return r.hits.map((h) => [h.id, h.score, h.attrs, h.source, new TextDecoder().decode(h.payload)]);
	}
	if (req.kind === "search") {
		const r = await index.search({ text: req.text ?? undefined, vector: req.vector ? Float32Array.from(req.vector) : undefined, topK: req.topK, dense: { nprobe: req.nprobe }, docs: "text" });
		return { mode: r.mode, hits: r.hits.map((h) => [h.id, h.score, h.doc ? [h.doc.meta, h.doc.text ?? null] : null]) };
	}
	if (req.kind === "lexical") {
		const r = await index.lexicalSearch(req.text, req.topK);
		return { tokens: r.tokens, hits: r.hits.map((h) => [h.id, h.score]), numDocs: r.stats.numDocs };
	}
	throw new Error(`unknown request ${req.kind}`);
}

const [cmd, dir, extra] = process.argv.slice(2);
const store = new FileBlobStore(dir);
if (cmd === "build") {
	const vectors = await buildGen(store, "gi");
	const index = await thread.open({ store, cache: false });
	await index.configureLexical({ text: true, bucketTerms: 8, defaults: { rrfK: 30 } });
	const r = rng(9);
	const fresh = unitRows(60, DIM, r);
	const ids = Array.from({ length: 60 }, (_, i) => 10000 + i);
	await index.upsert({ ids: ids.slice(0, 40), vectors: fresh.subarray(0, 40 * DIM), texts: ids.slice(0, 40).map(textOf), metas: ids.slice(0, 40).map((i) => `m${i}`), attrs: ids.slice(0, 40).map((i) => i % 7), payloads: ids.slice(0, 40).map((i) => `p${i}`) });
	await index.upsert({ ids: ids.slice(40), vectors: fresh.subarray(40 * DIM), texts: ids.slice(40).map(textOf), metas: ids.slice(40).map((i) => `m${i}`) });
	await index.upsert({ ids: [20000, 20001, 20002], texts: ["galaxy budget review", "galaxy galaxy", "deployments failed"], metas: ["g0", "g1", "g2"] });
	await index.upsert({ ids: [5, 10001], vectors: fresh.subarray(50 * DIM, 52 * DIM), texts: ["replaced five", "replaced 10001 galaxy"], metas: ["r5", "r10001"] });
	await index.delete([7, 10002, 20001]);
	const requests = [];
	const qr = rng(11);
	for (let i = 0; i < 12; i++) requests.push({ kind: "query", vector: Array.from(unitRows(1, DIM, qr)), topK: 10, nprobe: 4 + (i % 5), rerank: 40 });
	for (const i of [0, 5, 7, 300, 1499]) requests.push({ kind: "query", vector: Array.from(vectors.subarray(i * DIM, (i + 1) * DIM)), topK: 5, nprobe: 16, rerank: 60 });
	for (const i of [0, 3, 41, 50]) requests.push({ kind: "query", vector: Array.from(fresh.subarray(i * DIM, (i + 1) * DIM)), topK: 5, nprobe: 4, rerank: 40 });
	for (const t of ["galaxy budget", "deployment review", "alpha kilo item3", "replaced", "nothing-here"]) requests.push({ kind: "lexical", text: t, topK: 10 });
	for (const [t, i] of [["alpha bravo", 0], ["galaxy", 51], ["budget item7", 9]]) requests.push({ kind: "search", text: t, vector: Array.from(fresh.subarray(i * DIM, (i + 1) * DIM)), topK: 8, nprobe: 6 });
	requests.push({ kind: "search", text: "deployments failed", vector: null, topK: 5, nprobe: 4 });
	const results = [];
	for (const req of requests) results.push(await answer(index, req));
	fs.writeFileSync(path.join(dir, "ts-results.json"), JSON.stringify({ gen: index.gen, rev: index.rev, requests, results }));
	index.close();
} else if (cmd === "run") {
	const { requests } = JSON.parse(fs.readFileSync(path.join(dir, "requests.json"), "utf8"));
	const index = await thread.open({ store: extra ? await storeAt(extra) : store, cache: false });
	const results = [];
	for (const req of requests) results.push(await answer(index, req));
	process.stdout.write(JSON.stringify({ gen: index.gen, rev: index.rev, results }));
	index.close();
} else if (cmd === "pack") {
	const r = await packIndexToFile(store, extra);
	process.stdout.write(JSON.stringify(r));
} else if (cmd === "basepack") {
	const t = await resolveTarget(store);
	const r = await buildBasePack(store, await loadRevState(store, t.gen, t.rev), process.argv[5] ?? "base");
	fs.writeFileSync(extra, r.bytes);
	process.stdout.write(JSON.stringify({ key: r.key, keys: r.keys, routing: r.routing }));
} else throw new Error(`usage: ts_side.mjs build <dir> | run <dir> [store] | pack <dir> <file> | basepack <dir> <file> <kind>`);
