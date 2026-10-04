#!/usr/bin/env node
/**
 * The table of docs/query-path.md "Cold opens" from run_base.sh output: per open mode, the process-cold medians of
 * requests / MB before the first slab read, open ms, first-query ms, process e2e and recall, for one profile.
 *
 *   node bench/cold/base_table.mjs [--tag base] [--profile rtt25-200-link0] [--modes f32-seq,f32,...]
 */
import fs from "node:fs";
import path from "node:path";
import { median, parseArgs, readJsonl, root } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const TAG = args.tag ?? "base";
const PROFILE = args.profile ?? "rtt25-200-link0";
const MODES = (args.modes ?? "f32-seq,f32,f16,i8-seq,i8,cold8,cold16,cold32,cold16-seq").split(",");
const MB = 1048576;
const rows = [];
for (const m of MODES) {
	const f = path.join(root, `data/cold/${TAG}-ts-${m}-${PROFILE}.jsonl`);
	if (!fs.existsSync(f)) continue;
	const rs = readJsonl(f).filter((r) => r.kind === "trial");
	const pc = rs.filter((r) => r.scenario === "process-cold");
	const sc = rs.filter((r) => r.scenario === "slab-cold");
	const med = (xs) => median(xs.filter((x) => x !== null && x !== undefined));
	rows.push([
		m,
		String(med(pc.map((r) => r.reqs.open + r.reqs.route))),
		(med(pc.map((r) => r.bytes.open + r.bytes.route)) / MB).toFixed(2),
		med(pc.map((r) => r.ms.open)).toFixed(0),
		med(pc.map((r) => r.ms.route)).toFixed(0),
		med(pc.map((r) => r.ms.query)).toFixed(0),
		med(pc.map((r) => r.ms.e2e)).toFixed(0),
		med(sc.map((r) => r.ms.query)).toFixed(0),
		`${pc.filter((r) => r.routing === "cold").length}/${pc.length}`,
		((pc.reduce((a, r) => a + r.recall, 0) / pc.length) * 100).toFixed(0) + "%",
	]);
}
const head = ["mode", "req before 1st slab read", "MB before 1st slab read", "open ms", "route ms", "1st query ms", "process e2e ms", "2nd query ms", "cold-routed", "recall@10"];
console.log(`| ${head.join(" | ")} |\n|${head.map((_, i) => (i ? "---:" : "---")).join("|")}|\n` + rows.map((r) => `| ${r.join(" | ")} |`).join("\n"));
