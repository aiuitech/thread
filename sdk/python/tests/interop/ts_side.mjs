// The TypeScript half of tests/test_interop.py, over the built SDK (sdk/ts/dist) and a directory store:
//   node ts_side.mjs build <dir>   builds a format-3 generation with deltas, lexical segments and tombstones written
//                                  by the TS SDK, and records queries and their TS results in <dir>/ts-results.json
//   node ts_side.mjs run <dir> [store]   answers the requests in <dir>/requests.json (written by Python) against the
//                                  index at <store> (a directory or a pack file; default <dir>) and prints them
//   node ts_side.mjs pack <dir> <file>    packs the current revision of the index in <dir> into <file>
//   node ts_side.mjs basepack <dir> <file> <kind>   writes the base pack (kind base | cold) of the current revision
//   node ts_side.mjs merged <dir>  many small writes, then compactions that merge the tombstone sets; records the
//                                  requests and their TS results at the head and before the first compaction
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
// The trace envelope's shape (docs/observability.md), compared across SDKs: span names (search: top-level sorted, its
// lists run concurrently), the counters that depend on neither caching nor timing, and the kernel and attrs keys.
const DETERMINISTIC = {
	query: ["hits", "cells", "scanned", "scanBytes", "slabBytes", "pushed", "candidates", "filterDropped", "tombDropped", "rerankCandidates", "rerankSlabs", "deltas", "deltaBytes", "deltaScanned", "deltaHits", "deltaDropped", "deltaTop", "tombstones", "tombSets"],
	lexical: ["hits", "segments", "numDocs", "tokens", "distinct", "found", "depth", "filterDropped", "passes"],
	listKeys: ["entries", "more"],
};
function traceShape(t) {
	const names = (spans) => spans.map((s) => (s.children ? [s.name, names(s.children)] : s.name));
	let spans = names(t.spans);
	if (t.op === "search") spans = spans.map((s) => JSON.stringify(s)).sort().map((s) => JSON.parse(s));
	const pick = (op, prefix) => Object.fromEntries(DETERMINISTIC[op].filter((k) => prefix + k in t.counters).map((k) => [prefix + k, t.counters[prefix + k]]));
	const counters = t.op === "search" ? { hits: t.counters.hits, ...pick("query", "dense."), ...pick("lexical", "lexical.") } : pick(t.op, "");
	return { v: t.v, op: t.op, spans, counters, kernels: Object.keys(t.kernels).sort(), attrs: Object.keys(t.attrs).sort() };
}

async function answer(index, req) {
	if (req.kind === "query") {
		const r = await index.query(Float32Array.from(req.vector), { topK: req.topK, nprobe: req.nprobe, rerank: req.rerank, returnPayload: true, filter: req.filter ?? undefined, maxNprobe: req.maxNprobe ?? undefined });
		return { hits: r.hits.map((h) => [h.id, h.score, h.attrs, h.source, new TextDecoder().decode(h.payload)]), cells: r.stats.cells, trace: traceShape(r.stats.trace) };
	}
	if (req.kind === "search") {
		const r = await index.search({ text: req.text ?? undefined, vector: req.vector ? Float32Array.from(req.vector) : undefined, topK: req.topK, dense: { nprobe: req.nprobe }, docs: "text", filter: req.filter ?? undefined });
		return { mode: r.mode, hits: r.hits.map((h) => [h.id, h.score, h.doc ? [h.doc.meta, h.doc.text ?? null, h.doc.attrs] : null]), trace: traceShape(r.stats.trace) };
	}
	if (req.kind === "lexical") {
		const r = await index.lexicalSearch(req.text, req.topK, req.filter ? { filter: index.compileFilter(req.filter) } : {});
		return { tokens: r.tokens, hits: r.hits.map((h) => [h.id, h.score]), numDocs: r.stats.numDocs, trace: traceShape(r.stats.trace) };
	}
	if (req.kind === "compile") {
		return Array.from(index.compileFilter(req.filter));
	}
	if (req.kind === "get") {
		const r = await index.get(req.keys, { text: true });
		return req.keys.map((k) => (r.records.has(k) ? [k, r.records.get(k).id, r.records.get(k).meta, r.records.get(k).text ?? null, r.records.get(k).attrs] : null));
	}
	if (req.kind === "listKeys") {
		const r = await index.listKeys({ prefix: req.prefix, after: req.after ?? undefined, limit: req.limit });
		return { entries: r.entries.map((e) => [e.key, e.id]), more: r.more, next: r.next ?? null, trace: traceShape(r.trace) };
	}
	throw new Error(`unknown request ${req.kind}`);
}

const [cmd, dir, extra] = process.argv.slice(2);
const store = new FileBlobStore(dir);
if (cmd === "build") {
	const vectors = await buildGen(store, "gi");
	const index = await thread.open({ store, cache: false });
	await index.configureLexical({ text: true, bucketTerms: 8, defaults: { rrfK: 30 } });
	// an attribute schema; ids < 10040 get packed field values, the next batch raw words
	await index.configureAttrs({ kind: { values: ["doc", "note", "ticket"] }, public: "bool", team: { bits: 6 } });
	const r = rng(9);
	const fresh = unitRows(60, DIM, r);
	const ids = Array.from({ length: 60 }, (_, i) => 10000 + i);
	const attrsOf = (i) => ({ kind: ["doc", "note", "ticket"][i % 3], public: i % 2 === 0, team: i % 5 });
	await index.upsert({ ids: ids.slice(0, 40), vectors: fresh.subarray(0, 40 * DIM), texts: ids.slice(0, 40).map(textOf), metas: ids.slice(0, 40).map((i) => `m${i}`), attrs: ids.slice(0, 40).map(attrsOf), payloads: ids.slice(0, 40).map((i) => `p${i}`) });
	await index.upsert({ ids: ids.slice(40), vectors: fresh.subarray(40 * DIM), texts: ids.slice(40).map(textOf), metas: ids.slice(40).map((i) => `m${i}`), attrs: ids.slice(40).map((i) => i % 7) });
	await index.upsert({ ids: [20000, 20001, 20002], texts: ["galaxy budget review", "galaxy galaxy", "deployments failed"], metas: ["g0", "g1", "g2"], attrs: [{ kind: "note", public: true }, { kind: "ticket" }, { team: 3 }] });
	await index.upsert({ ids: [5, 10001], vectors: fresh.subarray(50 * DIM, 52 * DIM), texts: ["replaced five", "replaced 10001 galaxy"], metas: ["r5", "r10001"] });
	await index.delete([7, 10002, 20001]);
	// keyed documents: ids allocated from manifest.nextId (past the explicit ids above), a re-put keeps its id, a
	// delete by key
	const B = 20003;
	const keyed = await index.upsert({ keys: ["k/a", "k/b", "k/c", "k/d"], texts: ["keyed alpha galaxy", "keyed bravo", "keyed charlie budget", "keyed delta"], metas: ["ka", "kb", "kc", "kd"], attrs: [{ kind: "note" }, { team: 2 }, {}, { public: true }] });
	if (keyed.ids.join() !== `${B},${B + 1},${B + 2},${B + 3}` || keyed.allocated !== 4) throw new Error(`keyed ids ${keyed.ids}`);
	const re = await index.upsert({ keys: ["k/b", "k/e"], vectors: fresh.subarray(52 * DIM, 54 * DIM), texts: ["keyed bravo again galaxy", "keyed echo"], metas: ["kb2", "ke"] });
	if (re.ids.join() !== `${B + 1},${B + 4}` || re.allocated !== 1) throw new Error(`re-put ids ${re.ids}`);
	const dk = await index.deleteKeys(["k/c", "k/none"]);
	if (dk.count !== 1 || dk.missing.join() !== "k/none") throw new Error(`deleteKeys ${JSON.stringify(dk)}`);
	const requests = [];
	const qr = rng(11);
	for (let i = 0; i < 12; i++) requests.push({ kind: "query", vector: Array.from(unitRows(1, DIM, qr)), topK: 10, nprobe: 4 + (i % 5), rerank: 40 });
	for (const i of [0, 5, 7, 300, 1499]) requests.push({ kind: "query", vector: Array.from(vectors.subarray(i * DIM, (i + 1) * DIM)), topK: 5, nprobe: 16, rerank: 60 });
	for (const i of [0, 3, 41, 50]) requests.push({ kind: "query", vector: Array.from(fresh.subarray(i * DIM, (i + 1) * DIM)), topK: 5, nprobe: 4, rerank: 40 });
	for (const t of ["galaxy budget", "deployment review", "alpha kilo item3", "replaced", "nothing-here"]) requests.push({ kind: "lexical", text: t, topK: 10 });
	for (const [t, i] of [["alpha bravo", 0], ["galaxy", 51], ["budget item7", 9]]) requests.push({ kind: "search", text: t, vector: Array.from(fresh.subarray(i * DIM, (i + 1) * DIM)), topK: 8, nprobe: 6 });
	requests.push({ kind: "search", text: "deployments failed", vector: null, topK: 5, nprobe: 4 });
	requests.push({ kind: "get", keys: ["k/a", "k/b", "k/c", "k/e", "k/none"] });
	requests.push({ kind: "listKeys", prefix: "k/", limit: 10, after: null });
	requests.push({ kind: "listKeys", prefix: "k/", limit: 2, after: null });
	requests.push({ kind: "listKeys", prefix: "k/", limit: 2, after: "k/b" });
	requests.push({ kind: "search", text: "keyed galaxy", vector: null, topK: 5, nprobe: 4 });
	requests.push({ kind: "query", vector: Array.from(fresh.subarray(52 * DIM, 53 * DIM)), topK: 3, nprobe: 4, rerank: 40 });
	// filtered: field conditions on the schema (dense, with widening; lexical; hybrid) and the compiled words themselves
	const filters = [{ kind: "doc" }, { public: true, team: { $gte: 2 } }, { kind: { $in: ["note", "ticket"] }, public: false }, { team: { $gt: 1, $lt: 4 } }, { kind: { $ne: "doc" } }];
	for (const [i, f] of filters.entries()) {
		requests.push({ kind: "query", vector: Array.from(fresh.subarray(i * DIM, (i + 1) * DIM)), topK: 6, nprobe: 2, rerank: 40, filter: f, maxNprobe: 2 + i * 3 });
		requests.push({ kind: "lexical", text: ["alpha", "galaxy budget", "deployments", "item3 review", "bravo"][i], topK: 6, filter: f });
		requests.push({ kind: "search", text: ["alpha bravo", "galaxy", "deployments", "budget", "echo"][i], vector: Array.from(fresh.subarray((i + 20) * DIM, (i + 21) * DIM)), topK: 6, nprobe: 4, filter: f });
		requests.push({ kind: "compile", filter: f });
	}
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
} else if (cmd === "compact") {
	// Dense compaction in the TS SDK (folds every delta, splits cells over 120 entries), then the requests on the result.
	const { requests } = JSON.parse(fs.readFileSync(path.join(dir, "requests.json"), "utf8"));
	const index = await thread.open({ store, cache: false, refreshInterval: 0 });
	const runs = [];
	for (let i = 0; i < 20; i++) {
		const r = await index.compact({ force: true, maxCellEntries: 120 });
		runs.push({ outcome: r.outcome, folded: r.folded, splits: r.splits.length, k: r.k });
		if (!r.more) break;
	}
	const results = [];
	for (const req of requests) results.push(await answer(index, req));
	process.stdout.write(JSON.stringify({ gen: index.gen, rev: index.rev, k: index.manifest.k1 * index.manifest.k2, deltas: index.manifest.deltas.length, runs, results }));
	index.close();
} else if (cmd === "scoped") {
	// Scoped upserts from TS on a fresh dense generation (vectors from scoped-input.json), then the requests.
	const { vectors } = JSON.parse(fs.readFileSync(path.join(dir, "scoped-input.json"), "utf8"));
	await thread.createGeneration({ store, gen: "sc", dim: vectors[0].length, lex: { text: true }, attrs: { tag: { bits: 4 } }, compaction: null });
	const index = await thread.open({ store, cache: false, refreshInterval: 0 });
	await index.upsert({ keys: ["t/1", "t/2", "t/3"], texts: ["first text only", "second has a vector", "third will be renamed"], metas: ["m1", "m2", "m3"], vectors: [vectors[0], vectors[1], vectors[2]], attrs: [{ tag: 1 }, { tag: 2 }, { tag: 3 }] });
	await index.upsert({ keys: ["t/2"], vectors: [vectors[3]], attrs: [{ tag: 2 }], scope: "dense" });
	await index.upsert({ keys: ["t/3"], texts: ["third renamed by typescript"], metas: ["m3b"], scope: "lexical" });
	const { requests } = JSON.parse(fs.readFileSync(path.join(dir, "requests.json"), "utf8"));
	const results = [];
	for (const req of requests) results.push(await answer(index, req));
	process.stdout.write(JSON.stringify({ gen: index.gen, rev: index.rev, results }));
	index.close();
} else if (cmd === "merged") {
	// Many small writes from TS on a fresh generation of many cells (vector upserts, dense- and lexical-scoped re-puts,
	// keyed upserts with texts, deletes, re-inserts of deleted ids), then dense compactions that merge the tombstone
	// sets (with a segment compaction between them); records the requests and their TS results at the head (merged
	// sets) and at the revision before the first compaction (the original sets) in <dir>/ts-results.json.
	const dim = 16;
	await thread.createGeneration({ store, gen: "mg", dim, lex: { text: true }, compaction: { maxDeltas: 8, maxCellEntries: 12, maxTombstones: 1_000_000 } });
	const index = await thread.open({ store, cache: false, refreshInterval: 0 });
	const r = rng(23);
	const centers = unitRows(8, dim, r);
	const rows = (n) => unitRows(n, dim, r, centers, 0.3);
	const row = (x, i) => x.subarray(i * dim, (i + 1) * dim);
	const base = rows(300);
	await index.upsert({ keys: Array.from({ length: 300 }, (_, i) => `doc/${i}`), texts: Array.from({ length: 300 }, (_, i) => `document ${i} word${i % 7}`), vectors: base });
	while ((await index.compact({ force: true })).more);
	const [v, w2, w3, nv] = [rows(80), rows(80), rows(80), rows(80)];
	const deleted = [];
	for (let i = 0; i < 80; i++) {
		await index.upsert({ ids: [1000 + i], vectors: row(v, i) });
		if (i % 2 === 1) await index.upsert({ ids: [1000 + i - 1], vectors: row(w2, i - 1), scope: "dense" });
		else await index.upsert({ ids: [i % 300], texts: [`retitled ${i} word${i % 7}`], scope: "lexical" });
		if (i % 10 === 3) await index.upsert({ keys: [`new/${i}`], texts: [`fresh ${i} word${i % 7}`], vectors: row(nv, i) });
		if (i % 5 === 4) await index.delete([1000 + i - 2]), deleted.push(1000 + i - 2);
		if (i % 15 === 14) await index.upsert({ ids: [deleted.shift()], vectors: row(w3, i) });
		for (let j = 0; j < 3; j++) await index.delete([90000 + 3 * i + j]);
	}
	const beforeRev = index.rev;
	const written = index.manifest.tombstones.length;
	const requests = [];
	const qr = rng(5);
	for (let i = 0; i < 6; i++) requests.push({ kind: "query", vector: Array.from(unitRows(1, dim, qr)), topK: 10, nprobe: 64, rerank: 80 });
	for (const i of [0, 3, 10, 41, 77]) requests.push({ kind: "query", vector: Array.from(row(v, i)), topK: 3, nprobe: 64, rerank: 40 });
	for (const i of [14, 29, 44, 59, 74]) requests.push({ kind: "query", vector: Array.from(row(w3, i)), topK: 3, nprobe: 64, rerank: 40 });
	for (const i of [0, 10, 60]) requests.push({ kind: "query", vector: Array.from(row(w2, i)), topK: 3, nprobe: 64, rerank: 40 });
	for (const t of ["word3", "retitled", "document 7", "fresh", "word5 retitled"]) requests.push({ kind: "lexical", text: t, topK: 50 });
	requests.push({ kind: "search", text: "fresh word3", vector: Array.from(row(nv, 13)), topK: 8, nprobe: 64 });
	requests.push({ kind: "get", keys: ["doc/0", "doc/1", "doc/2", "doc/150", "new/3", "new/13", "new/none"] });
	requests.push({ kind: "listKeys", prefix: "new/", limit: 20, after: null });
	requests.push({ kind: "listKeys", prefix: "doc/", limit: 40, after: "doc/1" });
	const answerAll = async (ix) => {
		const out = [];
		for (const req of requests) out.push(await answer(ix, req));
		return out;
	};
	const beforeResults = await answerAll(index);
	const runs = [];
	const go = async (o) => {
		const c = await index.compact(o);
		runs.push({ outcome: c.outcome, folded: c.folded, merged: c.mergedTombstoneSets, sets: c.tombSets });
		return c;
	};
	while (index.compactionDue().due) await go({ deadlineMs: 1 });
	await index.compactSegments();
	while ((await go({ force: true, purge: false })).more);
	const results = await answerAll(index);
	fs.writeFileSync(path.join(dir, "ts-results.json"), JSON.stringify({ gen: index.gen, rev: index.rev, beforeRev, written, runs, requests, results, beforeResults }));
	index.close();
} else if (cmd === "pack") {
	const r = await packIndexToFile(store, extra);
	process.stdout.write(JSON.stringify(r));
} else if (cmd === "basepack") {
	const t = await resolveTarget(store);
	const r = await buildBasePack(store, await loadRevState(store, t.gen, t.rev), process.argv[5] ?? "base");
	fs.writeFileSync(extra, r.bytes);
	process.stdout.write(JSON.stringify({ key: r.key, keys: r.keys, routing: r.routing }));
} else throw new Error(`usage: ts_side.mjs build <dir> | run <dir> [store] | compact <dir> | scoped <dir> | merged <dir> | pack <dir> <file> | basepack <dir> <file> <kind>`);
