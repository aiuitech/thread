#!/usr/bin/env node
/**
 * Tables for the store-option matrix (bench/cold/run_store.sh): one row per config, per profile and SDK, with the
 * cold (process-cold and slab-cold) and warm query latency, the query's origin requests and bytes (the SDK's own
 * counters: foreground reads only) and what the proxy saw (including background warms), and recall@10.
 *
 *   node bench/cold/store_table.mjs [--dir data/cold-store] [--profiles rtt25,rtt60,r2s3] [--sdks ts,py] [--md]
 */
import fs from "node:fs";
import path from "node:path";
import { median, parseArgs, pct, root } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const dir = path.resolve(root, args.dir ?? "data/cold-store");
const profiles = (args.profiles ?? "rtt25,rtt60,r2s3").split(",");
const sdks = (args.sdks ?? "ts,py").split(",");
const MB = 1048576;
const f1 = (x) => (x === null || x === undefined || Number.isNaN(x) ? "-" : x >= 100 ? x.toFixed(0) : x.toFixed(1));
const order = fs.readFileSync(path.join(root, "bench/cold/run_store.sh"), "utf8").match(/^([a-z0-9-]+)\|/gm).map((s) => s.slice(0, -1));

for (const profile of profiles)
	for (const sdk of sdks) {
		const rows = [];
		for (const name of order) {
			const file = path.join(dir, `${profile}-${sdk}-${name}.jsonl`);
			if (!fs.existsSync(file)) continue;
			const recs = fs.readFileSync(file, "utf8").trim().split("\n").map((l) => JSON.parse(l)).filter((r) => r.kind === "trial");
			const sc = (s) => recs.filter((r) => r.scenario === s);
			const m = (s, f) => median(sc(s).map(f).filter((v) => v !== null && v !== undefined));
			const p = (s, f) => pct(sc(s).map(f).filter((v) => v !== null && v !== undefined), 0.9);
			const cold = [...sc("process-cold"), ...sc("slab-cold")];
			const recall = cold.reduce((a, r) => a + r.recall, 0) / cold.length;
			rows.push([
				name,
				`${f1(m("process-cold", (r) => r.ms.query))} / ${f1(p("process-cold", (r) => r.ms.query))}`,
				`${f1(m("process-cold", (r) => r.ms.e2e))}`,
				`${f1(m("slab-cold", (r) => r.ms.query))} / ${f1(p("slab-cold", (r) => r.ms.query))}`,
				`${f1(m("warm", (r) => r.ms.query))} / ${f1(p("warm", (r) => r.ms.query))}`,
				`${f1(m("slab-cold", (r) => r.reqs.query))}`,
				`${f1(m("slab-cold", (r) => r.bytes.query / MB))}`,
				`${f1(m("slab-cold", (r) => r.proxy?.requests))}`,
				`${f1(m("slab-cold", (r) => (r.proxy ? r.proxy.bytes / MB : null)))}`,
				`${(100 * recall).toFixed(1)}%`,
			]);
		}
		if (!rows.length) continue;
		const header = ["config", "process-cold query med/p90", "e2e med", "slab-cold med/p90", "warm med/p90", "slab-cold req", "slab-cold MB", "proxy req", "proxy MB", "recall@10 (cold)"];
		console.log(`\n### ${profile}, ${sdk === "ts" ? "TypeScript (Node)" : "Python"}\n`);
		console.log(`| ${header.join(" | ")} |`);
		console.log(`|${header.map((_, i) => (i ? "---:" : "---")).join("|")}|`);
		for (const r of rows) console.log(`| ${r.join(" | ")} |`);
	}
