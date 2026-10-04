#!/usr/bin/env node
/**
 * Update-path benchmark (upsert / delete / compaction) against the deployed Worker.
 *
 *   node bench/update_bench.mjs [--gen g4] [--nprobe 34] [--rerank 150] [--c 50] [--n 1000] [--out data/bench-updates.txt]
 *
 * The 5000 test query vectors are inserted as new vectors with ids 1_000_000 + i (they are not in the index), so
 * query i must find id 1_000_000 + i at rank 1 ("self-retrieval"). 894 test queries have an identical twin among
 * the other test queries (cos >= 0.999, data/prep/query-twins.json; none has one in the base set): when both are
 * inserted they tie at ~1.0, so rank 1 is counted as a hit if it is the query's own id or a twin's id ("rank-1,
 * twins ok"); the strict rank-1 rate and "own id in the top 10" are reported too. Recall@10 of the original ground
 * truth is computed over the hits that remain after removing ids >= 1_000_000 (queries ask for k = 20 there).
 *
 *  1. baseline: query latency (c, n) and recall/self-retrieval over 5000 queries with no deltas
 *  2. insert all 5000 in 20 batches of 250 (upsert latency, bytes); query latency with 1, 5 and 20 deltas
 *  3. self-retrieval + recall with 20 deltas
 *  4. delete ids 1_000_000..1_000_999 (one call): latency; the deleted ids must vanish, the rest stay
 *  5. compaction (bench/compact.mjs over build slots): wall time, cells, bytes; self-retrieval and latency after
 *  6. upsert latency for batches of 100 / 500 / 2000 (re-upserts of live ids = replace), then self-retrieval
 *  7. cleanup: delete every inserted id, compact with purge (delta cells + cells of the first compaction), check the manifest has no deltas or
 *     tombstones and recall equals the baseline
 * Writes a summary to --out and one JSON line per measurement to the same path with .jsonl (or --jsonl).
 *
 * Trigger mode `--only trigger [--batch 250] [--bg 10] [--cleanup 0]` (needs manifest.compaction): baseline latency,
 * then upserts of `--batch` new vectors until a response reports a started automatic compaction (more than
 * maxDeltas deltas), with `--bg` concurrent background queries on the current pointer the whole time; latency at
 * the trigger rev while the compaction runs; polls GET /index/status until the run is recorded (trigger -> new rev);
 * keeps the load 40 s more; latency and self-retrieval/recall at the new rev; background p50/p95/errors per phase.
 * Cleanup (unless --cleanup 0): delete the inserted ids, compact --purge, check the head is clean at baseline recall.
 *
 * Focused mode `--only upserts [--allowDirty 1] [--fold 1] [--cleanup 1] [--label X]`: only step 6's upsert plan
 * (100 x3, 500 x3, 2000 x2 of the test vectors [1000, 5000) as ids 1_000_000+i, client and server ms, encode ms,
 * the primary-cell search the Worker used: tree groups/beam or exhaustive), then self-retrieval of those ids;
 * `--fold 1` compacts (folds the segments into slabs) and checks self-retrieval again (the vectors now live in the
 * primary cells the encoder picked); `--cleanup 1` deletes all 5000 ids, compacts with purge and checks that the
 * head is clean with baseline recall. `--allowDirty 1` runs on a head that has deltas (e.g. a second pass).
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
// Writes run in the `writer` build slot (a dynamic Worker of the same deployment that serves no queries): an
// upsert's DeltaJob needs ~50 MB of transient wasm memory, which a query isolate would keep (wasm memory never
// shrinks). The Worker forwards /upsert and /delete there by default; THREAD_WRITE_SLOT=local runs them in the query
// isolates instead (A/B).
const WRITE_SLOT = process.env.THREAD_WRITE_SLOT ?? "writer";
const WRITE = `${BASE} (slot ${WRITE_SLOT})`;
const TOKEN = fs.readFileSync(path.join(root, "examples/cloudflare-worker/.env.secrets"), "utf8").trim().split("=").slice(1).join("=");
const prep = JSON.parse(fs.readFileSync(path.join(root, "data/prep/manifest.json"), "utf8"));
const dim = prep.dim;
const GEN = args.gen ?? "g4";
const NPROBE = Number(args.nprobe ?? 34);
const RERANK = Number(args.rerank ?? 150);
const C = Number(args.c ?? 50);
const N = Number(args.n ?? 1000);
const OUT = path.resolve(root, args.out ?? "data/bench-updates.txt");
const JSONL = args.jsonl ? path.resolve(root, args.jsonl) : OUT.replace(/\.txt$/, "") + ".jsonl";
const ID0 = 1_000_000;

const qbuf = fs.readFileSync(path.join(root, "data/prep/queries.f32"));
const gtbuf = fs.readFileSync(path.join(root, "data/prep/groundtruth.u32"));
const NQ = qbuf.byteLength / (dim * 4);
const gtk = gtbuf.byteLength / (NQ * 4);
const qvec = (i) => qbuf.subarray(i * dim * 4, (i + 1) * dim * 4);
const truth = (i) => new Set(Array.from({ length: 10 }, (_, j) => gtbuf.readUInt32LE((i * gtk + j) * 4)));
const H = { authorization: `Bearer ${TOKEN}` };
// Identical test queries (cos >= 0.999), computed offline from queries.f32.
const TWINS = JSON.parse(fs.readFileSync(path.join(root, "data/prep/query-twins.json"), "utf8")).twins;
const twinIds = (i) => new Set((TWINS[i] ?? []).map((j) => ID0 + j));

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

async function req(url, init) {
	for (let a = 1; ; a++) {
		const t0 = performance.now();
		try {
			const res = await fetch(url, init);
			const text = await res.text();
			const data = JSON.parse(text);
			if (!res.ok || data.error) throw Object.assign(new Error(`HTTP ${res.status}: ${data.error ?? text.slice(0, 200)}`), { status: res.status });
			return { ...data, _ms: performance.now() - t0 };
		} catch (e) {
			if (a >= 3 || (e.status && e.status < 500)) throw e;
			log(`  retry ${url.replace(BASE, "")} after ${e.message.slice(0, 160)}`);
			await new Promise((r) => setTimeout(r, 1000 * a));
		}
	}
}

async function manifest() {
	return (await req(`${BASE}/manifest?gen=${GEN}`, { headers: H })).manifest;
}

/** Upserts query vectors [from, from + n) as ids ID0 + from ... (binary body, ?firstId). */
async function upsert(from, n) {
	const body = new Uint8Array(n * dim * 4);
	for (let i = 0; i < n; i++) body.set(qvec(from + i), i * dim * 4);
	const r = await req(`${BASE}/upsert?gen=${GEN}&firstId=${ID0 + from}&slot=${WRITE_SLOT}`, { method: "POST", headers: { ...H, "content-type": "application/octet-stream" }, body });
	const bytes = r.bytes.delta + r.bytes.tombstones + r.bytes.manifest;
	record({ kind: "upsert", label: args.label ?? null, n, from, ms: r._ms, serverMs: r.ms, rev: r.rev, seq: r.seq, bytes: r.bytes, conflicts: r.conflicts, deltas: r.deltas, deltaBytes: r.deltaBytes, distinctCells: r.distinctCells, tree: r.tree ?? null, timing: r.timing, compaction: r.compaction ?? null });
	return { ...r, clientMs: r._ms, written: bytes };
}

async function del(ids) {
	const r = await req(`${BASE}/delete?slot=${WRITE_SLOT}`, { method: "POST", headers: { ...H, "content-type": "application/json" }, body: JSON.stringify({ gen: GEN, ids }) });
	record({ kind: "delete", n: ids.length, ms: r._ms, serverMs: r.ms, rev: r.rev, bytes: r.bytes, conflicts: r.conflicts });
	return { ...r, clientMs: r._ms };
}

async function query(i, rev, k) {
	const t0 = performance.now();
	const res = await fetch(`${BASE}/query?gen=${GEN}&rev=${rev}&k=${k}&nprobe=${NPROBE}&rerank=${RERANK}`, { method: "POST", headers: { ...H, "content-type": "application/octet-stream" }, body: qvec(i) });
	const ms = performance.now() - t0;
	if (!res.ok) {
		const text = await res.text();
		const title = text.match(/<title>([^<]*)<\/title>/i)?.[1];
		throw new Error(`HTTP ${res.status}: ${title ? `[html] ${title.trim()}` : text.slice(0, 200)}`);
	}
	const d = await res.json();
	return { i, ms, ids: d.ids, stats: d.stats, timing: d.timing, rev: d.rev };
}

async function run(idxs, rev, k, c = C) {
	const res = new Array(idxs.length);
	let next = 0;
	const t0 = performance.now();
	await Promise.all(
		Array.from({ length: Math.min(c, idxs.length) }, async () => {
			while (next < idxs.length) {
				const j = next++;
				try {
					res[j] = await query(idxs[j], rev, k);
				} catch (e) {
					res[j] = { error: e.message };
				}
			}
		}),
	);
	const wall = (performance.now() - t0) / 1000;
	const ok = res.filter((r) => !r.error);
	if (ok.length !== res.length) log(`  ${res.length - ok.length} query errors, first: ${res.find((r) => r.error).error}`);
	return { ok, errors: res.length - ok.length, wall };
}

const avg = (rs, f) => rs.reduce((a, r) => a + f(r), 0) / rs.length;

/** Latency pass: warmup + measured over queries [0, N) at concurrency C, k = 10. */
async function latency(label, rev) {
	const idxs = Array.from({ length: N }, (_, i) => i);
	await run(idxs, rev, 10);
	const { ok, errors, wall } = await run(idxs, rev, 10);
	const lat = ok.map((r) => r.ms).sort((a, b) => a - b);
	const row = {
		label,
		rev,
		n: ok.length,
		errors,
		c: C,
		p50: r1(pct(lat, 0.5)),
		p95: r1(pct(lat, 0.95)),
		p99: r1(pct(lat, 0.99)),
		mean: r1(avg(ok, (r) => r.ms)),
		rps: r1(ok.length / wall),
		srvTotal: r1(avg(ok, (r) => r.timing.total)),
		phase1: r1(avg(ok, (r) => r.timing.phase1)),
		phase2: r1(avg(ok, (r) => r.timing.phase2)),
		deltaMs: r1(avg(ok, (r) => r.timing.delta ?? 0)),
		deltas: ok[0]?.stats.deltas ?? 0,
		deltaKB: r1((ok[0]?.stats.deltaBytes ?? 0) / 1024),
		deltaScanned: ok[0]?.stats.deltaScanned ?? 0,
		tombstones: ok[0]?.stats.tombstones ?? 0,
		cacheMisses: ok.reduce((a, r) => a + (r.stats.cacheMisses ?? 0), 0),
		wasmMemMB: Math.max(...ok.map((r) => r.stats.wasmMemMB ?? 0)),
	};
	record({ kind: "latency", ...row });
	log(`latency ${label}: ${JSON.stringify(row)}`);
	return row;
}

/**
 * Self-retrieval / recall pass over `idxs` (k = 20). `inserted(i)`: whether id ID0 + i is expected live.
 * Returns the self-retrieval rate over live ids, recall@10 over base hits, how many results contain a dead
 * inserted id, and result lists with duplicate ids.
 */
async function quality(label, rev, idxs, live, deadIds = () => false) {
	const { ok, errors } = await run(idxs, rev, 20);
	let selfN = 0;
	let selfHit = 0;
	let selfTwin = 0;
	let selfTop10 = 0;
	let rec = 0;
	let dead = 0;
	let dups = 0;
	for (const r of ok) {
		if (live(r.i)) {
			selfN++;
			if (r.ids[0] === ID0 + r.i) selfHit++;
			if (r.ids[0] === ID0 + r.i || twinIds(r.i).has(r.ids[0])) selfTwin++;
			if (r.ids.slice(0, 10).includes(ID0 + r.i)) selfTop10++;
		}
		const gt = truth(r.i);
		const baseHits = r.ids.filter((id) => id < ID0).slice(0, 10);
		rec += baseHits.filter((id) => gt.has(id)).length / 10;
		if (r.ids.some((id) => id >= ID0 && deadIds(id))) dead++;
		if (new Set(r.ids).size !== r.ids.length) dups++;
	}
	const rate = (x) => (selfN ? +((100 * x) / selfN).toFixed(2) : null);
	const row = { label, rev, n: ok.length, errors, selfRate: rate(selfTwin), selfStrict: rate(selfHit), selfTop10: rate(selfTop10), selfN, recall: +((100 * rec) / ok.length).toFixed(2), deadIdResults: dead, dupResults: dups, deltas: ok[0]?.stats.deltas ?? 0, tombstones: ok[0]?.stats.tombstones ?? 0, routing: ok[0]?.stats.routing ?? null };
	record({ kind: "quality", ...row });
	log(`quality ${label}: ${JSON.stringify(row)}`);
	return row;
}

const all = Array.from({ length: NQ }, (_, i) => i);
const summary = { upserts: [], deletes: [], latency: [], quality: [], compactions: [] };
const T0 = performance.now();

out(`Update-path benchmark: ${GEN}, nprobe ${NPROBE} / rerank ${RERANK}, query passes c=${C} n=${N} (warmup + measured), ${new Date().toISOString()}`);
out(`Client bench/update_bench.mjs; writes to ${WRITE}, queries to ${BASE}; compaction over build slots (bench/compact.mjs).`);
out(`New vectors: the ${NQ} test queries as ids ${ID0}+i. Raw JSON lines: ${path.relative(root, JSONL)}.`);
out();

// ---- focused mode: the step-6 upsert plan only (see the header)
if (args.only === "upserts") {
	const T = (rows, cols) => {
		out(`| ${cols.map((c) => c[0]).join(" | ")} |`);
		out(`|${cols.map((c) => (c[2] === "l" ? "---" : "---:")).join("|")}|`);
		for (const r of rows) out(`| ${cols.map((c) => (typeof c[1] === "function" ? c[1](r) : r[c[1]])).join(" | ")} |`);
		out();
	};
	const label = args.label ?? "upserts";
	let mf = await manifest();
	if (!args.allowDirty && ((mf.deltas ?? []).length || (mf.tombstones ?? []).length)) throw new Error(`${GEN} rev ${mf.rev} is not clean; pass --allowDirty 1`);
	out(`### ${label}: ${GEN} rev ${mf.rev}, tree ${mf.tree ? `${mf.tree.groups} groups, beam ${mf.tree.beam} (${mf.tree.key})` : "none (exhaustive primary-cell search)"}, ${(mf.deltas ?? []).length} deltas before`);
	const plan = [
		[100, 1000],
		[100, 1100],
		[100, 1200],
		[500, 1300],
		[500, 1800],
		[500, 2300],
		[2000, 1000],
		[2000, 3000],
	];
	const ups = [];
	let rev = mf.rev;
	for (const [n, from] of plan) {
		const u = await upsert(from, n);
		ups.push({ n, ms: u.clientMs, serverMs: u.ms, encode: u.timing?.encode ?? null, load: u.timing?.load ?? null, put: u.timing?.put ?? null, delta: u.bytes.delta, cells: u.distinctCells, tree: u.tree ?? null, rev: u.rev });
		rev = u.rev;
		log(`upsert ${n} (ids ${ID0 + from}..): rev ${u.rev} client ${u.clientMs.toFixed(0)} ms, server ${u.ms} ms (load ${u.timing?.load} encode ${u.timing?.encode} put ${u.timing?.put}), tree ${JSON.stringify(u.tree ?? null)}`);
	}
	const med = (xs) => {
		const s = xs.filter((x) => x !== null).sort((a, b) => a - b);
		return s.length ? s[Math.floor((s.length - 1) / 2)] : null;
	};
	const rows = [100, 500, 2000].map((n) => {
		const g = ups.filter((u) => u.n === n);
		return { n, calls: g.length, client: g.map((u) => Math.round(u.ms)).join(", "), clientMed: Math.round(med(g.map((u) => u.ms))), server: g.map((u) => u.serverMs).join(", "), serverMed: med(g.map((u) => u.serverMs)), encode: g.map((u) => u.encode).join(", "), encodeMed: med(g.map((u) => u.encode)), encPerVec: +(med(g.map((u) => u.encode)) / n).toFixed(3), cells: Math.round(avg(g, (u) => u.cells)), search: g[0].tree ? `tree ${g[0].tree.groups}/${g[0].tree.beam}` : "exhaustive" };
	});
	T(rows, [["batch", "n"], ["calls", "calls"], ["client ms", "client", "l"], ["client median", "clientMed"], ["server ms", "server", "l"], ["server median", "serverMed"], ["encode ms", "encode", "l"], ["encode median", "encodeMed"], ["encode ms/vector", "encPerVec"], ["distinct cells", "cells"], ["primary-cell search", "search", "l"]]);
	const live = (i) => i >= 1000 && i < 5000;
	const qs = [await quality(`${label}: after the upserts`, rev, all, live)];
	if (args.fold) {
		const c = await compact({ gen: GEN });
		record({ kind: "compact", label, ...c });
		rev = c.rev;
		qs.push(await quality(`${label}: after compaction (folded)`, rev, all, live));
		out(`Compaction (fold): ${c.baseRev} -> ${c.rev}, ${c.cells} cells, ${(c.wallMs / 1000).toFixed(1)} s.`);
	}
	let cleanRow = null;
	if (args.cleanup) {
		const d = await del(Array.from({ length: NQ }, (_, i) => ID0 + i));
		qs.push(await quality(`${label}: after delete all ${NQ}`, d.rev, all, () => false, (id) => id >= ID0));
		const c = await compact({ gen: GEN, purge: true });
		record({ kind: "compact", label: `${label} cleanup`, ...c });
		mf = await manifest();
		cleanRow = await quality(`${label}: final (clean)`, c.rev, all, () => false, (id) => id >= ID0);
		qs.push(cleanRow);
		const clean = mf.rev === c.rev && !(mf.deltas ?? []).length && !(mf.tombstones ?? []).length;
		out(`Cleanup: delete ${NQ} ids (${d.clientMs.toFixed(0)} ms), compact --purge ${c.baseRev} -> ${c.rev} (${c.cells} cells, ${(c.wallMs / 1000).toFixed(1)} s, ${c.droppedTombstoneSets} tombstone sets dropped); head rev ${mf.rev}: ${(mf.deltas ?? []).length} deltas, ${(mf.tombstones ?? []).length} tombstone sets -> ${clean ? "clean" : "NOT CLEAN"}, tree ${mf.tree ? "kept" : "none"}.`);
	}
	T(qs, [["state", "label", "l"], ["rev", "rev"], ["deltas", "deltas"], ["tombstones", "tombstones"], ["self rank-1 % (twins ok)", (r) => r.selfRate ?? "-"], ["own id in top 10 %", (r) => r.selfTop10 ?? "-"], ["live ids", "selfN"], ["recall@10 %", "recall"], ["results with a deleted id", "deadIdResults"], ["duplicate ids", "dupResults"]]);
	out(`Total ${((performance.now() - T0) / 60000).toFixed(1)} min.`);
	process.exit(0);
}

// ---- trigger mode: automatic compaction (see the header)
if (args.only === "trigger") {
	const T = (rows, cols) => {
		out(`| ${cols.map((c) => c[0]).join(" | ")} |`);
		out(`|${cols.map((c) => (c[2] === "l" ? "---" : "---:")).join("|")}|`);
		for (const r of rows) out(`| ${cols.map((c) => (typeof c[1] === "function" ? c[1](r) : r[c[1]])).join(" | ")} |`);
		out();
	};
	const status = async () => req(`${BASE}/index/status?gen=${GEN}`, { headers: H });
	const BATCH = Number(args.batch ?? 250);
	const MAXB = Number(args.maxBatches ?? Math.floor(NQ / BATCH));
	const BG = Number(args.bg ?? 10);
	let mf = await manifest();
	if ((mf.deltas ?? []).length || (mf.tombstones ?? []).length) throw new Error(`${GEN} rev ${mf.rev} is not clean`);
	if (!mf.compaction) throw new Error(`${GEN} rev ${mf.rev} has no compaction policy (driver.mjs --publish-rev --compaction profile)`);
	const st0 = await status();
	if (st0.compacting) throw new Error(`a compaction holds the lock: ${JSON.stringify(st0.compacting)}`);
	const rev0 = mf.rev;
	out(`### Automatic compaction trigger: ${GEN} rev ${rev0}, policy ${JSON.stringify(mf.compaction)}`);
	out(`Upserts of ${BATCH} new vectors (ids ${ID0}+i) to ${WRITE} until a response reports a started compaction; background load: ${BG} concurrent queries on the current revision (no ?rev=, so isolates follow idx/current.json within 30 s), counted per phase.`);
	out();
	const baseLat = await latency("0 deltas (baseline)", rev0);
	// background load on the current pointer
	const bg = [];
	let bgStop = false;
	const bgLoop = async (w) => {
		for (let j = w; !bgStop; j += BG) {
			const i = j % NQ;
			const t = performance.now();
			try {
				const res = await fetch(`${BASE}/query?gen=${GEN}&k=10&nprobe=${NPROBE}&rerank=${RERANK}`, { method: "POST", headers: { ...H, "content-type": "application/octet-stream" }, body: qvec(i) });
				const ms = performance.now() - t;
				if (!res.ok) bg.push({ t, ms, error: `HTTP ${res.status}: ${(await res.text()).slice(0, 120)}` });
				else {
					const d = await res.json();
					bg.push({ t, ms, rev: d.rev, deltas: d.stats.deltas });
				}
			} catch (e) {
				bg.push({ t, ms: performance.now() - t, error: e.message });
			}
		}
	};
	const bgDone = Promise.all(Array.from({ length: BG }, (_, w) => bgLoop(w)));
	const tBg0 = performance.now();
	let trig = null;
	let rev = rev0;
	const ups = [];
	for (let b = 0; b < MAXB && !trig; b++) {
		const u = await upsert(b * BATCH, BATCH);
		rev = u.rev;
		ups.push({ b: b + 1, rev: u.rev, ms: u.clientMs, deltas: u.deltas, deltaKB: Math.round(u.deltaBytes / 1024), compaction: u.compaction ?? null });
		log(`upsert ${b + 1}: rev ${u.rev}, ${u.deltas} deltas (${(u.deltaBytes / 1048576).toFixed(1)} MB), ${u.clientMs.toFixed(0)} ms, compaction ${JSON.stringify(u.compaction ?? null)}`);
		if (u.compaction?.started) trig = { t: performance.now(), rev: u.rev, batches: b + 1, reasons: u.compaction.reasons };
	}
	if (!trig) throw new Error(`no compaction started after ${MAXB} batches`);
	const inserted = trig.batches * BATCH;
	// the rev the trigger produced, scanned while the compaction runs
	const during = await latency(`${trig.batches} deltas (trigger rev, during the compaction)`, trig.rev);
	let run = null;
	let seenLock = false;
	for (let w = 0; w < 240; w++) {
		const st = await status();
		if (st.compacting) seenLock = true;
		const last = st.lastCompaction;
		if (last && last.trigger?.rev === trig.rev && last.finishedAt) {
			run = { ...last, seenAt: performance.now(), headRev: st.rev };
			break;
		}
		await new Promise((r) => setTimeout(r, 500));
	}
	if (!run) throw new Error("the triggered compaction did not finish within 120 s");
	log(`compaction run: ${JSON.stringify(run)}`);
	record({ kind: "auto-compaction", ...run, clientTriggerToSeenMs: run.seenAt - trig.t });
	const newRev = run.rev ?? rev;
	// keep the load running until every isolate serves the new rev (current.json TTL 30 s) for a while
	const tCommit = run.seenAt;
	while (performance.now() - tCommit < 40_000) await new Promise((r) => setTimeout(r, 1000));
	bgStop = true;
	await bgDone;
	const afterLat = await latency(`${(await manifest()).deltas.length} deltas (after the automatic compaction)`, newRev);
	const afterQ = await quality("after the automatic compaction", newRev, all, (i) => i < inserted);
	const phase = (name, f) => {
		const rs = bg.filter(f);
		const ok = rs.filter((r) => !r.error).map((r) => r.ms).sort((a, b) => a - b);
		return { name, n: rs.length, errors: rs.length - ok.length, p50: ok.length ? r1(pct(ok, 0.5)) : "-", p95: ok.length ? r1(pct(ok, 0.95)) : "-", deltas: [...new Set(rs.filter((r) => !r.error).map((r) => r.deltas))].sort((a, b) => a - b).join(",") };
	};
	const phases = [
		phase("inserting (before the trigger)", (r) => r.t < trig.t),
		phase("trigger -> compaction committed", (r) => r.t >= trig.t && r.t < tCommit),
		phase("committed -> +40 s (isolates switch within 30 s)", (r) => r.t >= tCommit),
		phase("answered from the new rev", (r) => r.rev !== undefined && r.rev >= newRev),
	];
	const firstErr = bg.find((r) => r.error);
	T(ups, [["batch", "b"], ["rev", "rev"], ["deltas", "deltas"], ["delta KB", "deltaKB"], ["upsert ms", (r) => r1(r.ms)], ["compaction", (r) => (r.compaction ? `${r.compaction.started ? "started" : "not started"}: ${r.compaction.reasons.join("; ")}` : "-"), "l"]]);
	out(`Trigger: upsert ${trig.batches} (rev ${trig.rev}), ${trig.reasons.join("; ")}. Lock seen by the status poll: ${seenLock}.`);
	out(`Run (server, idx/${GEN}/compact.last.json): outcome ${run.outcome}, rev ${run.baseRev} -> ${run.rev}, ${run.cells} cells on ${run.slots ?? 1} slot(s)${run.slotJobs ? ` in ${run.slotJobs} jobs` : ""}${run.slotMs ? ` (slot wall ms ${run.slotMs.join(" / ")})` : ""}, ${run.folded} deltas folded, ${run.splits?.length ?? 0} splits, ${run.readMB} MB read / ${run.writtenMB} MB written, trigger -> new rev ${run.ms} ms (lock ${run.phases?.lock}, plan ${run.phases?.plan}, cells ${run.phases?.cells}, commit ${run.phases?.commit} ms); client: upsert response -> status shows it ${(run.seenAt - trig.t).toFixed(0)} ms (500 ms poll).${run.error ? ` Error: ${run.error}` : ""}`);
	out();
	out("Background load (current pointer):");
	T(phases, [["phase", "name", "l"], ["queries", "n"], ["errors", "errors"], ["p50 ms", "p50"], ["p95 ms", "p95"], ["deltas seen", "deltas", "l"]]);
	if (firstErr) out(`First background error: ${firstErr.error}`);
	out("Measured passes (explicit rev, warmup + measured, c=" + C + ", n=" + N + "):");
	T([baseLat, during, afterLat], [["state", "label", "l"], ["rev", "rev"], ["deltas", "deltas"], ["delta KB", "deltaKB"], ["p50", "p50"], ["p95", "p95"], ["mean", "mean"], ["RPS", "rps"], ["errors", "errors"], ["wasm MB", "wasmMemMB"]]);
	T([afterQ], [["state", "label", "l"], ["rev", "rev"], ["deltas", "deltas"], ["self rank-1 % (twins ok)", (r) => r.selfRate ?? "-"], ["own id in top 10 %", (r) => r.selfTop10 ?? "-"], ["live ids", "selfN"], ["recall@10 %", "recall"], ["duplicate ids", "dupResults"]]);
	if (args.cleanup !== "0") {
		for (let w = 0; w < 120 && (await status()).compacting; w++) await new Promise((r) => setTimeout(r, 1000));
		const d = await del(Array.from({ length: inserted }, (_, i) => ID0 + i));
		if (d.compaction?.started) {
			log(`the cleanup delete started a compaction (${d.compaction.reasons}); waiting for it`);
			for (let w = 0; w < 120; w++) {
				const st = await status();
				if (!st.compacting && st.lastCompaction?.trigger?.rev === d.rev && st.lastCompaction.finishedAt) break;
				await new Promise((r) => setTimeout(r, 1000));
			}
		}
		const c = await compact({ gen: GEN, purge: true });
		record({ kind: "compact", label: "trigger cleanup", ...c });
		mf = await manifest();
		const finalQ = await quality("final (clean)", c.rev, all, () => false, (id) => id >= ID0);
		const clean = mf.rev === c.rev && !(mf.deltas ?? []).length && !(mf.tombstones ?? []).length;
		const stF = await status();
		out(`Cleanup: delete ${inserted} ids (${d.clientMs.toFixed(0)} ms), compact --purge ${c.baseRev} -> ${c.rev} (${c.cells} cells, ${c.splits?.length ?? 0} splits, ${(c.wallMs / 1000).toFixed(1)} s, ${c.droppedTombstoneSets} tombstone sets dropped); head rev ${mf.rev}: ${(mf.deltas ?? []).length} deltas, ${(mf.tombstones ?? []).length} tombstone sets -> ${clean ? "clean" : "NOT CLEAN"}; ${stF.cells.entries} entries in ${stF.k} cells; recall@10 ${finalQ.recall}% (dead ids in results: ${finalQ.deadIdResults}).`);
	}
	out(`Total ${((performance.now() - T0) / 60000).toFixed(1)} min.`);
	process.exit(0);
}

// ---- 0/1: baseline
let m = await manifest();
if ((m.deltas ?? []).length || (m.tombstones ?? []).length) throw new Error(`${GEN} rev ${m.rev} is not clean (${m.deltas.length} deltas, ${m.tombstones.length} tombstone sets); compact first`);
const rev0 = m.rev;
log(`baseline ${GEN} rev ${rev0}`);
const baseQ = await quality("baseline", rev0, all, () => false);
summary.quality.push(baseQ);
summary.latency.push(await latency("0 deltas (baseline)", rev0));

// ---- 2: 20 batches of 250, latency at 1, 5, 20 deltas
let rev = rev0;
for (let b = 0; b < 20; b++) {
	const u = await upsert(b * 250, 250);
	summary.upserts.push({ phase: "insert", n: 250, ms: u.clientMs, written: u.written, delta: u.bytes.delta, rev: u.rev, cells: u.distinctCells, conflicts: u.conflicts });
	rev = u.rev;
	log(`upsert batch ${b + 1}/20: rev ${u.rev} seq ${u.seq} ${u.clientMs.toFixed(0)} ms, delta ${u.bytes.delta} B, ${u.distinctCells} cells`);
	if (b + 1 === 1 || b + 1 === 5 || b + 1 === 20) summary.latency.push(await latency(`${b + 1} deltas (${(b + 1) * 250} vectors)`, rev));
}

// ---- 3: self-retrieval at 20 deltas
summary.quality.push(await quality("20 deltas", rev, all, () => true));

// ---- 4: delete 1000
const delIds = Array.from({ length: 1000 }, (_, i) => ID0 + i);
const d1 = await del(delIds);
summary.deletes.push({ n: 1000, ms: d1.clientMs, rev: d1.rev, written: d1.bytes.tombstones + d1.bytes.manifest });
rev = d1.rev;
log(`delete 1000: rev ${rev} ${d1.clientMs.toFixed(0)} ms`);
const isDeleted = (id) => id >= ID0 && id < ID0 + 1000;
summary.quality.push(await quality("after delete 1000 (20 deltas)", rev, all, (i) => i >= 1000, isDeleted));

// ---- 5: compaction
const c1 = await compact({ gen: GEN });
summary.compactions.push({ label: "fold 20 deltas + 1 delete", ...c1 });
record({ kind: "compact", ...c1 });
rev = c1.rev;
summary.quality.push(await quality("after compaction", rev, all, (i) => i >= 1000, isDeleted));
summary.latency.push(await latency("after compaction (0 deltas)", rev));

// ---- 6: upsert latency for 100 / 500 / 2000 (re-upserts of live ids: replace)
const plan = [
	[100, 1000],
	[100, 1100],
	[100, 1200],
	[500, 1300],
	[500, 1800],
	[500, 2300],
	[2000, 1000],
	[2000, 3000],
];
for (const [n, from] of plan) {
	const u = await upsert(from, n);
	summary.upserts.push({ phase: "replace", n, ms: u.clientMs, written: u.written, delta: u.bytes.delta, rev: u.rev, cells: u.distinctCells, conflicts: u.conflicts });
	rev = u.rev;
	log(`upsert ${n} (replace ids ${ID0 + from}..): rev ${u.rev} ${u.clientMs.toFixed(0)} ms, delta ${u.bytes.delta} B`);
}
summary.quality.push(await quality(`after ${plan.length} replace upserts`, rev, all, (i) => i >= 1000, isDeleted));
summary.latency.push(await latency(`${plan.length} deltas (${plan.reduce((a, [n]) => a + n, 0)} vectors, replaces)`, rev));

// ---- 7: cleanup
const d2 = await del(Array.from({ length: NQ }, (_, i) => ID0 + i));
summary.deletes.push({ n: NQ, ms: d2.clientMs, rev: d2.rev, written: d2.bytes.tombstones + d2.bytes.manifest });
rev = d2.rev;
summary.quality.push(await quality(`after delete all ${NQ}`, rev, all, () => false, (id) => id >= ID0));
// purge: also rewrite the cells the first compaction wrote, so every tombstone set can be dropped.
const c2 = await compact({ gen: GEN, purge: true });
summary.compactions.push({ label: `fold ${plan.length} deltas + delete all`, ...c2 });
record({ kind: "compact", ...c2 });
rev = c2.rev;
m = await manifest();
const finalQ = await quality("final (clean)", rev, all, () => false, (id) => id >= ID0);
summary.quality.push(finalQ);
summary.latency.push(await latency("final (clean)", rev));
const clean = m.rev === rev && !(m.deltas ?? []).length && !(m.tombstones ?? []).length;

// ---- report
const T = (rows, cols) => {
	out(`| ${cols.map((c) => c[0]).join(" | ")} |`);
	out(`|${cols.map((c) => (c[2] === "l" ? "---" : "---:")).join("|")}|`);
	for (const r of rows) out(`| ${cols.map((c) => (typeof c[1] === "function" ? c[1](r) : r[c[1]])).join(" | ")} |`);
	out();
};
const upStats = (rows) => {
	const ms = rows.map((r) => r.ms).sort((a, b) => a - b);
	return { n: rows[0].n, count: rows.length, p50: r1(pct(ms, 0.5)), min: r1(ms[0]), max: r1(ms.at(-1)), mean: r1(ms.reduce((a, b) => a + b, 0) / ms.length), deltaKB: r1(avg(rows, (r) => r.delta) / 1024), writtenKB: r1(avg(rows, (r) => r.written) / 1024), perVecB: Math.round(avg(rows, (r) => r.delta / r.n)), cells: Math.round(avg(rows, (r) => r.cells)), conflicts: rows.reduce((a, r) => a + r.conflicts, 0) };
};
out("## 1. Upsert latency (client-observed, one request = one delta segment + tombstone set + manifest rev + current.json CAS)");
const ups = [100, 250, 500, 2000].map((n) => summary.upserts.filter((u) => u.n === n)).filter((g) => g.length).map(upStats);
T(ups, [["batch", "n"], ["calls", "count"], ["p50 ms", "p50"], ["min", "min"], ["max", "max"], ["mean", "mean"], ["delta KB", "deltaKB"], ["B/vector", "perVecB"], ["written KB/call", "writtenKB"], ["distinct cells", "cells"], ["conflicts", "conflicts"]]);
out("Batches of 250 inserted new ids; 100/500/2000 re-upserted live ids (replace: their tombstone set hides the compacted copy).");
out();
out("## 2/3/4/6. Self-retrieval and recall (5000 queries, k=20; recall@10 over hits with id < 1,000,000)");
T(summary.quality, [["state", "label", "l"], ["rev", "rev"], ["deltas", "deltas"], ["tombstones", "tombstones"], ["self rank-1 % (twins ok)", (r) => r.selfRate ?? "-"], ["strict rank-1 %", (r) => r.selfStrict ?? "-"], ["own id in top 10 %", (r) => r.selfTop10 ?? "-"], ["live ids", "selfN"], ["recall@10 %", "recall"], ["Δ vs baseline", (r) => (r.recall - baseQ.recall).toFixed(2)], ["results with a deleted id", "deadIdResults"], ["duplicate ids", "dupResults"]]);
out("## 3. Query latency vs delta segments present (c=" + C + ", n=" + N + ", k=10, measured pass after a warmup pass)");
T(summary.latency, [["state", "label", "l"], ["rev", "rev"], ["deltas", "deltas"], ["delta KB", "deltaKB"], ["entries", "deltaScanned"], ["tombstones", "tombstones"], ["p50", "p50"], ["p95", "p95"], ["p99", "p99"], ["mean", "mean"], ["RPS", "rps"], ["srv total", "srvTotal"], ["phase1", "phase1"], ["phase2", "phase2"], ["misses", "cacheMisses"], ["wasm MB", "wasmMemMB"]]);
out("## 4. Deletes");
T(summary.deletes, [["ids", "n"], ["ms", (r) => r1(r.ms)], ["rev", "rev"], ["bytes written", "written"]]);
out("## 5/7. Compaction");
T(summary.compactions, [["run", "label", "l"], ["baseRev -> rev", (r) => `${r.baseRev} -> ${r.rev}`], ["cells", "cells"], ["delta cells", "deltaCells"], ["tomb cells", "tombCells"], ["delta entries folded", "deltaEntries"], ["entries before -> after", (r) => `${r.oldEntries} -> ${r.newEntries}`], ["read MB", "readMB"], ["written MB", "writtenMB"], ["jobs", "jobs"], ["job p50/max ms", (r) => `${r.jobMsP50}/${r.jobMsMax}`], ["plan/cells/commit s", (r) => `${(r.planMs / 1000).toFixed(1)}/${(r.cellsMs / 1000).toFixed(1)}/${(r.commitMs / 1000).toFixed(1)}`], ["wall s", (r) => (r.wallMs / 1000).toFixed(1)], ["tomb sets dropped", "droppedTombstoneSets"]]);
out(`Final state: ${GEN} rev ${m.rev}, deltas ${(m.deltas ?? []).length}, tombstone sets ${(m.tombstones ?? []).length} -> ${clean ? "clean" : "NOT CLEAN"}; final recall ${finalQ.recall}% vs baseline ${baseQ.recall}%.`);
out(`Total ${((performance.now() - T0) / 60000).toFixed(1)} min.`);
