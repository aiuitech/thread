#!/usr/bin/env node
/**
 * Compaction driver: folds delta segments and tombstones into new slab revisions.
 *
 *   node bench/compact.mjs [--gen g4] [--cells all|1,2,3] [--purge 1] [--maxCellEntries N] [--chunk 48] [--concurrency 8] [--cellConcurrency 3] [--gc 0|1]
 *
 * Every stage runs in a build slot of the orchestrator (THREAD_ORCH, default https://vib-bench.aiuitech.workers.dev):
 * POST /build/<stage>?slot=<id>, one dynamic Worker isolate per slot id (see worker/src/main.ts).
 *
 * 1. compact-plan   (slot admin): head rev = baseRev, the cells to rewrite: cells of every delta segment, plus,
 *                    with --purge, the cells an earlier compaction rewrote at a rev <= the newest tombstone set
 *                    (they may hold ids a set hides; rewriting them lets the commit drop the sets).
 * 2. compact-cells  (slots cmp-0..cmp-{N-1}, N = --concurrency / THREAD_SLOTS (8), one job per slot at a time, `--chunk`
 *                    cells per job, `--cellConcurrency` cells in flight per job; the slot ids differ from the
 *                    driver's 0..N-1 so a compaction never shares an isolate with a running build job):
 *                    old slab records minus tombstones + the cell's records from the folded deltas
 *                    -> slab/{cell:05}.{baseRev+1}.
 *                    Cells whose entry count would exceed maxCellEntries (manifest.compaction, or --maxCellEntries;
 *                    0 = no splits) are split in two (plan.splits: [cell, newCell], new cells numbered from k):
 *                    2-means over the decoded entries, every entry re-encoded against its new mean.
 * 3. compact-commit (slot admin): slabs.{rev}.json, manifest rev + 1 without the folded deltas (plan.folded: those
 *                    whose cells were all rewritten; a partial --cells run keeps the rest) and without the
 *                    tombstone sets nothing references any more, idx/current.json. With splits also
 *                    fine.{rev}.f32 (k + splits means), k2 and a rebuilt centroid tree (recall measured).
 * Automatic compactions (the Worker's own, after writes) hold idx/{gen}/compact.lock; this driver does not take it,
 * so a commit that races one fails with 409 (nothing is lost; rerun).
 * `--gc 1` then runs the gc stage (dry run unless `--gcDelete 1`).
 *
 * Also importable: `compact({gen, cells, ...})` returns the summary.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const TOKEN = fs.readFileSync(path.join(root, "worker/.env.secrets"), "utf8").trim().split("=").slice(1).join("=");
const ORCH = process.env.THREAD_ORCH ?? process.env.THREAD_URL ?? "https://vib-bench.aiuitech.workers.dev";
const POOL = Array.from({ length: Number(process.env.THREAD_SLOTS ?? 8) }, (_, i) => `cmp-${i}`);
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

/** POST {ORCH}/build/{stage}?slot={slot}. */
export async function post(slot, stage, body, attempts = 4) {
	const p = `/build/${stage}?slot=${slot}`;
	for (let a = 1; ; a++) {
		const t0 = performance.now();
		try {
			const res = await fetch(`${ORCH}${p}`, { method: "POST", headers: { authorization: `Bearer ${TOKEN}`, "content-type": "application/json" }, body: JSON.stringify(body) });
			const text = await res.text();
			let data;
			try {
				data = JSON.parse(text);
			} catch {
				throw Object.assign(new Error(`HTTP ${res.status} non-JSON: ${text.slice(0, 300)}`), { status: res.status });
			}
			if (!res.ok || data.error) throw Object.assign(new Error(`HTTP ${res.status}: ${data.error ?? text.slice(0, 300)}`), { status: res.status });
			return { ...(data.result ?? data), _ms: performance.now() - t0 };
		} catch (e) {
			// 409 = a semantic conflict (another compaction committed): never retried.
			if (a >= attempts || e.status === 409 || e.status === 400) throw e;
			log(`  retry ${p} (${a}) after: ${e.message.replace(/\s+/g, " ").slice(0, 200)}`);
			await new Promise((r) => setTimeout(r, 1500 * a));
		}
	}
}

export async function compact({ gen, cells, purge = false, maxCellEntries, chunk = 48, concurrency = POOL.length, cellConcurrency = 3, quiet = false } = {}) {
	const T0 = performance.now();
	const plan = await post("admin", "compact-plan", { gen, purge, ...(cells ? { cells } : {}), ...(maxCellEntries !== undefined ? { maxCellEntries } : {}) });
	if (!quiet) log(`plan: ${plan.gen} baseRev ${plan.baseRev} -> slabRev ${plan.slabRev}: ${plan.cells.length} cells (delta cells ${plan.deltaCells}, tombstone cells ${plan.tombCells}), ${plan.deltas.length} deltas (${plan.deltaEntries} entries, ${plan.folded.length} folded), ${plan.tombstoneSets} tombstone sets (${plan.tombstones} ids), splits ${JSON.stringify(plan.splitCandidates ?? [])} (cap ${plan.maxCellEntries}${plan.splitBlocked?.length ? `, blocked ${JSON.stringify(plan.splitBlocked)}` : ""}), ${plan._ms.toFixed(0)} ms`);
	const splits = plan.splits ?? [];
	const jobs = [];
	for (let i = 0; i < plan.cells.length; i += chunk) jobs.push(plan.cells.slice(i, i + chunk));
	const free = POOL.slice(0, Math.max(1, Math.min(concurrency, POOL.length)));
	const waiters = [];
	const acquire = () => (free.length ? Promise.resolve(free.pop()) : new Promise((r) => waiters.push(r)));
	const release = (w) => (waiters.length ? waiters.shift()(w) : free.push(w));
	const results = new Array(jobs.length);
	let next = 0;
	let done = 0;
	const tCells = performance.now();
	await Promise.all(
		Array.from({ length: Math.min(concurrency, jobs.length) }, async () => {
			while (next < jobs.length) {
				const j = next++;
				const w = await acquire();
				try {
					const mine = new Set(jobs[j]);
					results[j] = await post(w, "compact-cells", { gen: plan.gen, baseRev: plan.baseRev, cells: jobs[j], concurrency: cellConcurrency, folded: plan.folded, splits: splits.filter(([c]) => mine.has(c)) });
				} finally {
					release(w);
				}
				done++;
				if (!quiet && (done % Math.max(1, Math.floor(jobs.length / 8)) === 0 || done === jobs.length)) log(`  ${done}/${jobs.length} cell jobs, ${((performance.now() - tCells) / 1000).toFixed(1)}s`);
			}
		}),
	);
	const cellsMs = performance.now() - tCells;
	const rows = results.flatMap((r) => r.rows);
	const sum = (f) => results.reduce((a, r) => a + f(r), 0);
	// A commit with splits builds fine.{rev}.f32 and a centroid tree (+ its recall check: ~75 MB): like every stage it
	// runs in a slot, never in an isolate that serves queries.
	const commit = await post("admin", "compact-commit", { gen: plan.gen, baseRev: plan.baseRev, rows, folded: plan.folded, splits });
	const splitsDone = results.flatMap((r) => r.splits ?? []);
	const wallMs = performance.now() - T0;
	const jobMs = results.map((r) => r._ms).sort((a, b) => a - b);
	const summary = {
		gen: plan.gen,
		baseRev: plan.baseRev,
		rev: commit.rev,
		slabRev: plan.slabRev,
		cells: rows.length,
		deltaCells: plan.deltaCells,
		tombCells: plan.tombCells,
		foldedDeltas: commit.folded,
		deltaEntries: sum((r) => r.deltaEntries),
		oldEntries: sum((r) => r.oldEntries),
		newEntries: sum((r) => r.newEntries),
		readMB: +(sum((r) => r.readBytes) / 1e6).toFixed(1),
		writtenMB: +(sum((r) => r.writtenBytes) / 1e6).toFixed(1),
		slabsTableKB: +(commit.slabsTableBytes / 1024).toFixed(1),
		droppedTombstoneSets: commit.droppedTombstoneSets.length,
		remainingDeltas: commit.deltas,
		remainingTombstoneSets: commit.tombstoneSets,
		k: commit.k,
		splits: splitsDone,
		tree: commit.tree,
		jobs: jobs.length,
		jobMsP50: Math.round(jobMs[Math.floor(jobMs.length / 2)] ?? 0),
		jobMsMax: Math.round(jobMs.at(-1) ?? 0),
		planMs: Math.round(plan._ms),
		cellsMs: Math.round(cellsMs),
		commitMs: Math.round(commit._ms),
		wallMs: Math.round(wallMs),
	};
	if (!quiet) log("compacted", JSON.stringify(summary));
	return summary;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
	const args = {};
	for (let i = 2; i < process.argv.length; i++) {
		const m = process.argv[i].match(/^--([^=\s]+)(?:=(.*))?$/);
		if (!m) continue;
		if (m[2] !== undefined) args[m[1]] = m[2];
		else if (process.argv[i + 1] && !process.argv[i + 1].startsWith("--")) args[m[1]] = process.argv[++i];
		else args[m[1]] = "true";
	}
	const cells = args.cells === "all" ? "all" : args.cells ? args.cells.split(",").map(Number) : undefined;
	const s = await compact({ gen: args.gen, cells, purge: args.purge === "1" || args.purge === "true", maxCellEntries: args.maxCellEntries !== undefined ? Number(args.maxCellEntries) : undefined, chunk: Number(args.chunk ?? 48), concurrency: Number(args.concurrency ?? POOL.length), cellConcurrency: Number(args.cellConcurrency ?? 3) });
	if (args.gc === "1") {
		const g = await post("admin", "gc", { gen: s.gen, dryRun: args.gcDelete !== "1", graceSec: Number(args.graceSec ?? 300) });
		log("gc", JSON.stringify(g));
	}
}
