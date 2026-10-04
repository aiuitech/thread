#!/usr/bin/env node
/**
 * Summary tables of cold_bench runs (TypeScript or Python, same schema), and their medians side by side.
 *
 *   node bench/cold/summarize.mjs data/cold/baseline-ts-proxy25.jsonl [more.jsonl ...] [--only-compare]
 *
 * One file: its summary table (what cold_bench printed). Several: each summary, then per scenario the medians of
 * every metric with one column per file (before/after comparisons of an optimization). `--markdown`: one markdown
 * table per scenario, a column per file, cells "median / p90" (what docs/cold-start.md shows).
 */
import path from "node:path";
import { compare, markdown, parseArgs, readJsonl, SCENARIOS, summary } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2), ["markdown", "only-compare"]);
const files = args._ ?? [];
if (!files.length) {
	console.error("usage: summarize.mjs RUN.jsonl [RUN2.jsonl ...] [--only-compare]");
	process.exit(2);
}
const runs = files.map((f) => {
	const rows = readJsonl(f);
	const meta = rows.find((r) => r.kind === "meta") ?? {};
	return { name: path.basename(f, ".jsonl"), rows, meta };
});
if (args.markdown === "true") {
	console.log(markdown(runs, args.scenarios ? args.scenarios.split(",") : SCENARIOS));
	process.exit(0);
}
if (args["only-compare"] !== "true" || runs.length === 1)
	for (const r of runs) {
		const m = r.meta;
		const net = m.proxyConfig ? ` rtt ${m.proxyConfig.rttMs} ms, ${m.proxyConfig.mbps} Mbps, conn +${m.proxyConfig.connMs} ms${m.proxyConfig.linkMbps ? `, link ${m.proxyConfig.linkMbps} Mbps shared` : ""}` : "";
		console.log(`# ${r.name}: ${m.sdk} ${m.runtime ?? ""} store=${m.store} ${m.endpoint ?? ""}${net} n=${m.n} queries ${m.queries?.used} nprobe ${m.nprobe} rerank ${m.rerank} git ${m.git ?? "?"}\n`);
		console.log(summary(r.rows) + "\n");
	}
if (runs.length > 1) console.log(compare(runs));
