#!/usr/bin/env node
/**
 * Tables for the store-option matrix (bench/cold/run_store.sh), markdown: one row per config, per profile and SDK,
 * with the baseline run of docs/cold-start.md first (data/cold/baseline-{sdk}-proxy-{profile}.jsonl, or -r2s3).
 * Columns: process-cold e2e and query, slab-cold query (median / p90) with its phase 1 / phase 2 split, the slab-cold
 * query's requests (phase 1 / phase 2, the SDK's foreground counters), its MB (SDK / what the proxy saw, which
 * includes background warms), the warm query, and recall@10 over the cold queries.
 *
 *   node bench/cold/store_table.mjs [--dir data/cold-store] [--profiles rtt25,rtt25link500,rtt60,r2s3] [--sdks ts,py]
 */
import fs from "node:fs";
import path from "node:path";
import { median, parseArgs, pct, root } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const dir = path.resolve(root, args.dir ?? "data/cold-store");
const profiles = (args.profiles ?? "rtt25,rtt25link500,rtt60,r2s3").split(",");
const sdks = (args.sdks ?? "ts,py").split(",");
const MB = 1048576;
const f = (x, d = 0) => (x === null || x === undefined || Number.isNaN(x) ? "-" : x.toFixed(d));
const order = fs.readFileSync(path.join(root, "bench/cold/run_store.sh"), "utf8").match(/^([a-z0-9-]+)\|/gm).map((s) => s.slice(0, -1));
const load = (file) => (fs.existsSync(file) ? fs.readFileSync(file, "utf8").trim().split("\n").map((l) => JSON.parse(l)).filter((r) => r.kind === "trial") : null);

function row(name, recs) {
	const sc = (s) => recs.filter((r) => r.scenario === s);
	const m = (s, g) => median(sc(s).map(g).filter((v) => v !== null && v !== undefined));
	const p = (s, g) => pct(sc(s).map(g).filter((v) => v !== null && v !== undefined), 0.9);
	const cold = [...sc("process-cold"), ...sc("slab-cold")];
	const recall = cold.reduce((a, r) => a + r.recall, 0) / cold.length;
	return [
		name,
		f(m("process-cold", (r) => r.ms.e2e)),
		f(m("process-cold", (r) => r.ms.query)),
		`${f(m("slab-cold", (r) => r.ms.query))} / ${f(p("slab-cold", (r) => r.ms.query))}`,
		`${f(m("slab-cold", (r) => r.ms.phase1))} / ${f(m("slab-cold", (r) => r.ms.phase2))}`,
		`${f(m("slab-cold", (r) => r.reqs.phase1))} / ${f(m("slab-cold", (r) => r.reqs.phase2))}`,
		`${f(m("slab-cold", (r) => r.bytes.query / MB), 1)} / ${f(m("slab-cold", (r) => (r.proxy ? r.proxy.bytes / MB : null)), 1)}`,
		f(m("warm", (r) => r.ms.query), 1),
		`${(100 * recall).toFixed(1)}%`,
	];
}

if (!args.tag) for (const profile of profiles)
	for (const sdk of sdks) {
		const rows = [];
		const base = load(path.join(root, "data/cold", profile === "r2s3" ? `baseline-${sdk}-r2s3.jsonl` : `baseline-${sdk}-proxy-${profile}.jsonl`));
		if (base) rows.push(row("baseline (docs)", base));
		for (const name of order) {
			const recs = load(path.join(dir, `${profile}-${sdk}-${name}.jsonl`));
			if (recs) rows.push(row(name, recs));
		}
		if (rows.length < 2) continue;
		const header = ["config", "process-cold e2e", "process-cold query", "slab-cold query med / p90", "slab-cold phase1 / phase2 ms", "slab-cold req phase1 / phase2", "slab-cold MB SDK / proxy", "warm query", "recall@10"];
		console.log(`\n#### ${profile}, ${sdk === "ts" ? "TypeScript (Node)" : "Python"}\n`);
		console.log(`| ${header.join(" | ")} |`);
		console.log(`|${header.map((_, i) => (i ? "---:" : "---")).join("|")}|`);
		for (const r of rows) console.log(`| ${r.join(" | ")} |`);
	}

// --tag TAG: the run_baseline.sh runs of TAG against the baseline, side by side (median, scenario by scenario).
if (args.tag) {
	const pair = (a, b, d = 0) => `${f(a, d)} -> ${f(b, d)}`;
	const header = ["profile", "store", "sdk", "process-cold e2e", "process-cold query", "slab-cold query", "warm query", "slab-cold requests", "slab-cold proxy MB", "process-cold proxy conns", "recall@10"];
	console.log(`\n#### baseline -> ${args.tag} (run_baseline.sh, medians)\n`);
	console.log(`| ${header.join(" | ")} |`);
	console.log(`|${header.map((_, i) => (i > 2 ? "---:" : "---")).join("|")}|`);
	for (const profile of ["rtt25", "rtt25link500", "rtt60", "r2s3"])
		for (const store of profile === "r2s3" ? ["r2s3"] : ["proxy", "pack"])
			for (const sdk of sdks) {
				const name = (tag) => path.join(root, "data/cold", profile === "r2s3" ? `${tag}-${sdk}-r2s3.jsonl` : `${tag}-${sdk}-${store}-${profile}.jsonl`);
				const a = load(name("baseline"));
				const b = load(name(args.tag));
				if (!a || !b) continue;
				const m = (recs, s, g) => median(recs.filter((r) => r.scenario === s).map(g).filter((v) => v !== null && v !== undefined));
				const rc = (recs) => {
					const c = recs.filter((r) => r.scenario !== "warm");
					return (100 * c.reduce((x, r) => x + r.recall, 0)) / c.length;
				};
				const g = (s, fn, d) => pair(m(a, s, fn), m(b, s, fn), d);
				console.log(`| ${[profile, store, sdk, g("process-cold", (r) => r.ms.e2e), g("process-cold", (r) => r.ms.query), g("slab-cold", (r) => r.ms.query), g("warm", (r) => r.ms.query), g("slab-cold", (r) => r.reqs.query), g("slab-cold", (r) => (r.proxy ? r.proxy.bytes / MB : null), 1), g("process-cold", (r) => r.proxy?.newConnections), pair(rc(a), rc(b), 1) + "%"].join(" | ")} |`);
			}
}
