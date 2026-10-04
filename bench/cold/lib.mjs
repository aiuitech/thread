/**
 * Shared by cold_bench.mjs, warm_proxy.mjs, make_pack.mjs and summarize.mjs: arguments, the dataset's queries and
 * ground truth, recall, and the summary table (cold_bench.py prints the same table, see `summary` there).
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
export const SCHEMA = "thread-cold/1";
export const SCENARIOS = ["process-cold", "slab-cold", "warm"];

/** `--k v` pairs, `--flag` alone (or listed in `booleans`) = "true", everything else in `_`. */
export function parseArgs(argv, booleans = []) {
	const out = {};
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (!a.startsWith("--")) {
			(out._ ??= []).push(a);
			continue;
		}
		const v = argv[i + 1];
		if (v === undefined || v.startsWith("--") || booleans.includes(a.slice(2))) out[a.slice(2)] = "true";
		else (out[a.slice(2)] = v), i++;
	}
	return out;
}

export function loadEnv(file) {
	return Object.fromEntries(
		fs
			.readFileSync(path.resolve(root, file), "utf8")
			.split("\n")
			.filter((l) => l.includes("=") && !l.trim().startsWith("#"))
			.map((l) => [l.slice(0, l.indexOf("=")).trim(), l.slice(l.indexOf("=") + 1).trim()]),
	);
}

/** Query vectors and top-k ground truth from data/prep (see bench/prep/prep_dataset.py). */
export function loadDataset() {
	const prep = JSON.parse(fs.readFileSync(path.join(root, "data/prep/manifest.json"), "utf8"));
	const q = fs.readFileSync(path.join(root, "data/prep/queries.f32"));
	const gt = fs.readFileSync(path.join(root, "data/prep/groundtruth.u32"));
	const dim = prep.dim;
	const total = q.byteLength / (dim * 4);
	const gtk = gt.byteLength / (total * 4);
	return {
		dim,
		total,
		gtk,
		vector: (i) => new Float32Array(q.buffer.slice(q.byteOffset + i * dim * 4, q.byteOffset + (i + 1) * dim * 4)),
		truth: (i, k = 10) => Array.from({ length: Math.min(k, gtk) }, (_, j) => gt.readUInt32LE((i * gtk + j) * 4)),
	};
}

export const recallAt = (ids, truth) => {
	const t = new Set(truth);
	return ids.slice(0, truth.length).filter((id) => t.has(id)).length / truth.length;
};

// ---------- summary ----------

/** Median (mean of the middle two for an even count). */
export function median(xs) {
	const s = xs.slice().sort((a, b) => a - b);
	const n = s.length;
	if (!n) return null;
	return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
/** Nearest-rank percentile: the ceil(q * n)-th smallest. */
export function pct(xs, q) {
	const s = xs.slice().sort((a, b) => a - b);
	if (!s.length) return null;
	return s[Math.min(s.length - 1, Math.max(0, Math.ceil(q * s.length) - 1))];
}

const MB = 1048576;
/** [label, row => value, kind]: kind "ms" (1 decimal), "n" (count), "mb" (bytes shown as MB), "pct". */
export const METRICS = [
	["ms boot (spawn -> main)", (r) => r.ms.boot, "ms"],
	["ms import SDK", (r) => r.ms.import, "ms"],
	["ms wasm compile", (r) => r.ms.wasmCompile, "ms"],
	["ms wasm init", (r) => r.ms.wasmInit, "ms"],
	["ms open.pack", (r) => r.ms.pack, "ms"],
	["ms open.current", (r) => r.ms.current, "ms"],
	["ms open.manifest", (r) => r.ms.manifest, "ms"],
	["ms open.slabs", (r) => r.ms.slabs, "ms"],
	["ms open.base", (r) => r.ms.base, "ms"],
	["ms   base fetch", (r) => r.ms.baseFetch, "ms"],
	["ms   base build", (r) => r.ms.baseBuild, "ms"],
	["ms open total", (r) => r.ms.open, "ms"],
	["ms route", (r) => r.ms.route, "ms"],
	["ms phase1", (r) => r.ms.phase1, "ms"],
	["ms phase2", (r) => r.ms.phase2, "ms"],
	["ms query total", (r) => r.ms.query, "ms"],
	["ms process e2e (spawn -> hits)", (r) => r.ms.e2e, "ms"],
	["req open (pack+cur+man+slabs+base)", (r) => r.reqs.open, "n"],
	["req phase1", (r) => r.reqs.phase1, "n"],
	["req phase2", (r) => r.reqs.phase2, "n"],
	["req query", (r) => r.reqs.query, "n"],
	["MB open", (r) => r.bytes.open, "mb"],
	["MB phase1", (r) => r.bytes.phase1, "mb"],
	["MB phase2", (r) => r.bytes.phase2, "mb"],
	["MB query", (r) => r.bytes.query, "mb"],
	["cache hits (query)", (r) => r.stats.cacheHits, "n"],
	["warm pack ready at query start (1/0)", (r) => (r.warm ? (r.warm.state === "ready" ? 1 : 0) : undefined), "n"],
	["ms warm pack seeded (since open start)", (r) => r.warm?.readyMs, "ms"],
	["scan prefix MB", (r) => r.stats.scanBytes, "mb"],
	["slab MB transferred", (r) => r.stats.slabBytes, "mb"],
	["rerank reads", (r) => r.stats.rerankReads, "n"],
	["rerank local / spec used", (r) => (r.stats.rerankLocal ?? 0) + (r.stats.specUsed ?? 0), "n"],
	["proxy requests", (r) => r.proxy?.requests, "n"],
	["proxy MB", (r) => r.proxy?.bytes, "mb"],
	["proxy new connections", (r) => r.proxy?.newConnections, "n"],
	["proxy upstream fetches (want 0)", (r) => r.proxy?.upstreamFetches, "n"],
	["recall@10", (r) => r.recall, "pct"],
];

export function fmtVal(v, kind) {
	if (v === null || v === undefined || Number.isNaN(v)) return "-";
	if (kind === "ms") return v.toFixed(1);
	if (kind === "mb") return (v / MB).toFixed(2);
	if (kind === "pct") return (100 * v).toFixed(1) + "%";
	return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

function table(header, rows) {
	const w = header.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i].length)));
	const line = (cells) => cells.map((c, i) => (i === 0 ? c.padEnd(w[i]) : c.padStart(w[i]))).join("  ");
	return [line(header), line(w.map((n) => "-".repeat(n))), ...rows.map(line)].join("\n");
}

/** The summary table of one run: median and p90 of every metric per scenario (recall: mean and min). */
export function summary(rows) {
	const trials = rows.filter((r) => r.kind === "trial");
	const by = Object.fromEntries(SCENARIOS.map((s) => [s, trials.filter((r) => r.scenario === s)]));
	const out = [];
	for (const [label, get, kind] of METRICS) {
		const cells = [];
		let any = false;
		for (const s of SCENARIOS) {
			const xs = by[s].map(get).filter((v) => v !== null && v !== undefined);
			if (xs.length) any = true;
			if (kind === "pct") cells.push(fmtVal(xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null, kind), fmtVal(xs.length ? Math.min(...xs) : null, kind));
			else cells.push(fmtVal(median(xs), kind), fmtVal(pct(xs, 0.9), kind));
		}
		if (any) out.push([label, ...cells]);
	}
	const n = SCENARIOS.map((s) => by[s].length).join("/");
	return table(["metric (n=" + n + ")", ...SCENARIOS.flatMap((s) => [`${s} med`, "p90"])], out) + "\n(recall@10 columns: mean, min)";
}

/** Medians of several runs side by side, one scenario at a time (for before/after comparisons). */
export function compare(runs) {
	const out = [];
	for (const s of SCENARIOS) {
		const rows = [];
		for (const [label, get, kind] of METRICS) {
			const cells = runs.map(({ rows: rs }) => {
				const xs = rs.filter((r) => r.kind === "trial" && r.scenario === s).map(get).filter((v) => v !== null && v !== undefined);
				return kind === "pct" ? fmtVal(xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null, kind) : fmtVal(median(xs), kind);
			});
			if (cells.some((c) => c !== "-")) rows.push([label, ...cells]);
		}
		out.push(`== ${s} (medians; recall@10 mean)\n` + table(["metric", ...runs.map((r) => r.name)], rows));
	}
	return out.join("\n\n");
}

/** Markdown tables, one per scenario: a row per metric, a column per run, cells "median / p90" (recall: "mean / min"). */
export function markdown(runs, scenarios = SCENARIOS) {
	const out = [];
	for (const s of scenarios) {
		const rows = [];
		for (const [label, get, kind] of METRICS) {
			const cells = runs.map(({ rows: rs }) => {
				const xs = rs.filter((r) => r.kind === "trial" && r.scenario === s).map(get).filter((v) => v !== null && v !== undefined);
				if (!xs.length) return "-";
				if (kind === "pct") return `${fmtVal(xs.reduce((a, b) => a + b, 0) / xs.length, kind)} / ${fmtVal(Math.min(...xs), kind)}`;
				const a = fmtVal(median(xs), kind);
				const b = fmtVal(pct(xs, 0.9), kind);
				return a === b ? a : `${a} / ${b}`;
			});
			if (cells.some((c) => c !== "-")) rows.push(`| ${label.trim()} | ${cells.join(" | ")} |`);
		}
		out.push(`**${s}** (median / p90; recall@10: mean / min)\n\n| metric | ${runs.map((r) => r.name).join(" | ")} |\n|---|${runs.map(() => "---:").join("|")}|\n${rows.join("\n")}`);
	}
	return out.join("\n\n");
}

export function readJsonl(file) {
	return fs
		.readFileSync(file, "utf8")
		.split("\n")
		.filter((l) => l.trim())
		.map((l) => JSON.parse(l));
}
