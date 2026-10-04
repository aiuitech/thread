#!/usr/bin/env node
/**
 * Cell-split benchmark: splits the largest cells of a generation with a cheap compaction of only those cells and
 * compares recall, latency and the query isolates' wasm memory before and after.
 *
 *   node bench/split_bench.mjs [--gen g4] [--maxCellEntries 2600] [--top 64] [--nprobe 34] [--rerank 150]
 *                              [--c 50] [--n 1000] [--out data/bench-split.txt] [--dryRun 1]
 *
 * 1. GET /index/status (cell sizes, the `--top` largest cells) and a baseline: recall@10 over the 5000 test
 *    queries (k = 10, ground truth of data/prep) and a latency pass (warmup + measured, c, n) at the head rev.
 * 2. compact --cells <the cells over --maxCellEntries> --maxCellEntries <it> (bench/compact.mjs: plan in slot admin,
 *    cells over build slots, commit in slot admin): each such cell is 2-means split and re-encoded, k grows by one per
 *    split, the commit writes fine.{rev}.f32 and a rebuilt centroid tree (recall measured on sample vectors).
 * 3. The same measurements at the new rev (the first queries of every query isolate load the new base: the
 *    Searcher over k + splits means), the new cell sizes, and the decision: keep if recall@10 is within 0.2 points.
 * Writes a summary to --out and one JSON line per measurement to the same path with .jsonl.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { compact } from "./compact.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
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
const H = { authorization: `Bearer ${TOKEN}` };
const prep = JSON.parse(fs.readFileSync(path.join(root, "data/prep/manifest.json"), "utf8"));
const dim = prep.dim;
const GEN = args.gen ?? "g4";
const CAP = Number(args.maxCellEntries ?? 2600);
const TOP = Number(args.top ?? 64);
const NPROBE = Number(args.nprobe ?? 34);
const RERANK = Number(args.rerank ?? 150);
const C = Number(args.c ?? 50);
const N = Number(args.n ?? 1000);
const OUT = path.resolve(root, args.out ?? "data/bench-split.txt");
const JSONL = OUT.replace(/\.txt$/, "") + ".jsonl";

const qbuf = fs.readFileSync(path.join(root, "data/prep/queries.f32"));
const gtbuf = fs.readFileSync(path.join(root, "data/prep/groundtruth.u32"));
const NQ = qbuf.byteLength / (dim * 4);
const gtk = gtbuf.byteLength / (NQ * 4);
const qvec = (i) => qbuf.subarray(i * dim * 4, (i + 1) * dim * 4);
const truth = (i) => new Set(Array.from({ length: 10 }, (_, j) => gtbuf.readUInt32LE((i * gtk + j) * 4)));

const lines = [];
const out = (s = "") => {
	console.log(s);
	lines.push(s);
	fs.writeFileSync(OUT, lines.join("\n") + "\n");
};
const record = (o) => fs.appendFileSync(JSONL, JSON.stringify({ at: new Date().toISOString(), ...o }) + "\n");
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const pct = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
const r1 = (x) => +x.toFixed(1);

async function getJson(p) {
	const res = await fetch(`${BASE}${p}`, { headers: H });
	const d = await res.json();
	if (!res.ok || d.error) throw new Error(`GET ${p}: HTTP ${res.status} ${d.error ?? ""}`);
	return d;
}

async function query(i, rev, extra = "") {
	const t0 = performance.now();
	const res = await fetch(`${BASE}/query?gen=${GEN}&rev=${rev}&k=10&nprobe=${NPROBE}&rerank=${RERANK}${extra}`, { method: "POST", headers: { ...H, "content-type": "application/octet-stream" }, body: qvec(i) });
	const ms = performance.now() - t0;
	if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
	const d = await res.json();
	return { i, ms, ids: d.ids, stats: d.stats, timing: d.timing };
}

async function run(idxs, rev, c, extra = "") {
	const res = new Array(idxs.length);
	let next = 0;
	const t0 = performance.now();
	await Promise.all(
		Array.from({ length: Math.min(c, idxs.length) }, async () => {
			while (next < idxs.length) {
				const j = next++;
				try {
					res[j] = await query(idxs[j], rev, extra);
				} catch (e) {
					res[j] = { error: e.message };
				}
			}
		}),
	);
	const ok = res.filter((r) => !r.error);
	if (ok.length !== res.length) log(`  ${res.length - ok.length} query errors, first: ${res.find((r) => r.error).error}`);
	return { ok, errors: res.length - ok.length, wall: (performance.now() - t0) / 1000 };
}

async function measure(label, rev) {
	const all = Array.from({ length: NQ }, (_, i) => i);
	// The first pass loads the revision's base in the query isolates (and is the recall pass).
	const q = await run(all, rev, C);
	const recall = (100 * q.ok.reduce((a, r) => a + r.ids.filter((id) => truth(r.i).has(id)).length / 10, 0)) / q.ok.length;
	const beam = await run(all, rev, C, "&route=beam&beam=32");
	const recallBeam = (100 * beam.ok.reduce((a, r) => a + r.ids.filter((id) => truth(r.i).has(id)).length / 10, 0)) / beam.ok.length;
	const idxs = Array.from({ length: N }, (_, i) => i);
	await run(idxs, rev, C);
	const l = await run(idxs, rev, C);
	const lat = l.ok.map((r) => r.ms).sort((a, b) => a - b);
	const mem = [...q.ok, ...l.ok].map((r) => r.stats.wasmMemMB ?? 0);
	const loads = q.ok.map((r) => r.timing.indexLoadMs ?? 0).filter((x) => x > 0);
	const row = {
		label,
		rev,
		n: q.ok.length,
		errors: q.errors + beam.errors + l.errors,
		recall: +recall.toFixed(2),
		recallBeam32: +recallBeam.toFixed(2),
		scanned: Math.round(q.ok.reduce((a, r) => a + r.stats.scanned, 0) / q.ok.length),
		p50: r1(pct(lat, 0.5)),
		p95: r1(pct(lat, 0.95)),
		rps: r1(l.ok.length / l.wall),
		srvTotal: r1(l.ok.reduce((a, r) => a + r.timing.total, 0) / l.ok.length),
		wasmMemMBmax: Math.max(...mem),
		wasmMemMBmedian: mem.sort((a, b) => a - b)[Math.floor(mem.length / 2)],
		indexLoadMsMedian: loads.length ? loads.sort((a, b) => a - b)[Math.floor(loads.length / 2)] : null,
	};
	record({ kind: "measure", ...row });
	log(`${label}: ${JSON.stringify(row)}`);
	return row;
}

const T = (rows, cols) => {
	out(`| ${cols.map((c) => c[0]).join(" | ")} |`);
	out(`|${cols.map((c) => (c[2] === "l" ? "---" : "---:")).join("|")}|`);
	for (const r of rows) out(`| ${cols.map((c) => (typeof c[1] === "function" ? c[1](r) : r[c[1]])).join(" | ")} |`);
	out();
};

out(`Cell-split benchmark: ${GEN}, split the cells over ${CAP} entries, nprobe ${NPROBE} / rerank ${RERANK}, recall over ${NQ} queries (k=10), latency c=${C} n=${N}, ${new Date().toISOString()}`);
out(`Client bench/split_bench.mjs; queries to ${BASE}; compaction over build slots (bench/compact.mjs). Raw JSON lines: ${path.relative(root, JSONL)}.`);
out();
const st0 = await getJson(`/index/status?gen=${GEN}&top=${TOP}`);
if (st0.deltas || st0.tombSets) throw new Error(`${GEN} rev ${st0.rev} has ${st0.deltas} deltas and ${st0.tombSets} tombstone sets; compact first`);
const cells = st0.cells.largest.filter(([, n]) => n > CAP).map(([c]) => c);
out(`Before: rev ${st0.rev}, k ${st0.k}, ${st0.cells.entries} entries (mean ${st0.cells.mean}/cell), largest ${JSON.stringify(st0.cells.largest.slice(0, 12))}`);
out(`Cells over ${CAP}: ${cells.length} ${JSON.stringify(cells)}`);
if (!cells.length) throw new Error(`no cell over ${CAP} among the top ${TOP}`);
if (cells.length >= TOP) throw new Error(`every one of the top ${TOP} cells is over ${CAP}: raise --top`);
if (args.dryRun) process.exit(0);
const before = await measure("before", st0.rev);
const c = await compact({ gen: GEN, cells, maxCellEntries: CAP, chunk: 4 });
record({ kind: "compact", ...c });
const st1 = await getJson(`/index/status?gen=${GEN}&top=${TOP}`);
const after = await measure("after split", c.rev);
const delta = after.recall - before.recall;
const keep = Math.abs(delta) <= 0.2 || delta > 0;
out(`Compaction ${c.baseRev} -> ${c.rev}: ${c.cells} cells rewritten (${cells.length} split -> ${c.cells - cells.length} new), ${c.readMB} MB read, ${c.writtenMB} MB written, ${(c.wallMs / 1000).toFixed(1)} s (plan ${(c.planMs / 1000).toFixed(1)}, cells ${(c.cellsMs / 1000).toFixed(1)}, commit ${(c.commitMs / 1000).toFixed(1)} s incl. fine.{rev}.f32 + tree rebuild).`);
out(`Rebuilt tree: ${JSON.stringify(c.tree)}`);
out();
out("Splits (entries before -> cell A / new cell B):");
T(c.splits, [["cell", "cell"], ["new cell", "newCell"], ["entries", "entries"], ["A", "a"], ["B", "b"]]);
out(`After: rev ${st1.rev}, k ${st1.k}, ${st1.cells.entries} entries, largest ${JSON.stringify(st1.cells.largest.slice(0, 12))}`);
out(`Max cell: ${st0.cells.maxCell.count} (cell ${st0.cells.maxCell.cell}) -> ${st1.cells.maxCell.count} (cell ${st1.cells.maxCell.cell}).`);
out();
T([before, after], [["state", "label", "l"], ["rev", "rev"], ["recall@10 % (flat)", "recall"], ["recall@10 % (beam 32)", "recallBeam32"], ["scanned/query", "scanned"], ["p50 ms", "p50"], ["p95 ms", "p95"], ["RPS", "rps"], ["srv ms", "srvTotal"], ["wasm MB max", "wasmMemMBmax"], ["wasm MB median", "wasmMemMBmedian"], ["base load ms (median of loads)", (r) => r.indexLoadMsMedian ?? "-"], ["errors", "errors"]]);
out(`Recall delta ${delta >= 0 ? "+" : ""}${delta.toFixed(2)} points -> ${keep ? "KEEP the split (within 0.2 points)" : "recall dropped by more than 0.2 points"}.`);
