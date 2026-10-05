#!/usr/bin/env node
/**
 * The summary table of docs/cold-start.md: the baseline against the merged code's runs of run_baseline.sh, per
 * profile, store and SDK, medians over the n trials of each scenario:
 *
 *   baseline      data/cold/baseline-*.jsonl (rev 179, the code before any cold-start work)
 *   final         TAG=final: the merged defaults (rev 180 overlay with an inline manifest in current.json)
 *   final-i8      TAG=final-i8: + routing "i8" (base pack, or a pack's base range)
 *   final-cold16  TAG=final-cold16: + coldRouting "tree", beam 16 (cold base pack)
 *
 * Columns: process-cold e2e, the open (incl. a pack's own open), the process-cold / slab-cold / warm query, the
 * process-cold reads and MB (open + query, the SDK's own counters: requests that reached the store, bytes delivered),
 * the slab-cold reads and MB, recall@10 over the cold queries (process-cold + slab-cold).
 *
 *   node bench/cold/final_table.mjs [--tags baseline,final,final-i8,final-cold16] [--sdks ts,py]
 */
import fs from "node:fs";
import path from "node:path";
import { median, parseArgs, root } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const tags = (args.tags ?? "baseline,final,final-i8,final-cold16").split(",");
const sdks = (args.sdks ?? "ts,py").split(",");
const label = { baseline: "baseline", final: "final defaults", "final-i8": "i8 + base pack", "final-cold16": "cold tree, beam 16" };
const MB = 1048576;
const f = (x, d = 0) => (x === null || x === undefined || Number.isNaN(x) ? "-" : x.toFixed(d));
const load = (file) => (fs.existsSync(file) ? fs.readFileSync(file, "utf8").trim().split("\n").map((l) => JSON.parse(l)).filter((r) => r.kind === "trial") : null);

function row(name, recs) {
	const sc = (s) => recs.filter((r) => r.scenario === s);
	const m = (s, g) => median(sc(s).map(g).filter((v) => v !== null && v !== undefined));
	const cold = [...sc("process-cold"), ...sc("slab-cold")];
	const recall = cold.reduce((a, r) => a + r.recall, 0) / cold.length;
	return [
		name,
		f(m("process-cold", (r) => r.ms.e2e)),
		f(m("process-cold", (r) => r.ms.open)),
		f(m("process-cold", (r) => r.ms.query)),
		f(m("slab-cold", (r) => r.ms.query)),
		f(m("warm", (r) => r.ms.query)),
		`${f(m("process-cold", (r) => r.reqs.open + r.reqs.query))} / ${f(m("process-cold", (r) => (r.bytes.open + r.bytes.query) / MB), 1)}`,
		`${f(m("slab-cold", (r) => r.reqs.query))} / ${f(m("slab-cold", (r) => r.bytes.query / MB), 1)}`,
		`${(100 * recall).toFixed(1)}%`,
	];
}

const header = ["mode", "process-cold e2e ms", "open ms", "process-cold query ms", "slab-cold query ms", "warm query ms", "process-cold req / MB", "slab-cold req / MB", "recall@10"];
for (const sdk of sdks)
	for (const profile of ["rtt25", "rtt60", "rtt25link500", "r2s3"])
		for (const store of profile === "r2s3" ? ["r2s3"] : ["proxy", "pack"]) {
			const rows = [];
			for (const tag of tags) {
				const recs = load(path.join(root, "data/cold", profile === "r2s3" ? `${tag}-${sdk}-r2s3.jsonl` : `${tag}-${sdk}-${store}-${profile}.jsonl`));
				if (recs?.length) rows.push(row(label[tag] ?? tag, recs));
			}
			if (rows.length < 2) continue;
			console.log(`\n#### ${sdk === "ts" ? "TypeScript (Node)" : "Python"}, ${profile === "r2s3" ? "real R2 (S3 API)" : `${profile}, ${store === "pack" ? "pack" : "per-object store"}`}\n`);
			console.log(`| ${header.join(" | ")} |`);
			console.log(`|${header.map((_, i) => (i ? "---:" : "---")).join("|")}|`);
			for (const r of rows) console.log(`| ${r.join(" | ")} |`);
		}
