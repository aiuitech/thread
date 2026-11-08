/**
 * Summaries of trace envelopes (docs/observability.md), the same for the SDKs' and the Worker's: per span path
 * (`phase1/fetch`) percentiles of wall time and mean store work, mean counters, mean kernel CPU.
 *
 *   import { summarize, printSummary } from "./lib/trace.mjs";
 *   printSummary(summarize(envelopes));
 */

const pct = (sorted, q) => (sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] : 0);
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const round = (x, d = 2) => Math.round(x * 10 ** d) / 10 ** d;

function* walk(spans, prefix = "") {
	for (const s of spans ?? []) {
		const path = prefix ? `${prefix}/${s.name}` : s.name;
		yield [path, s];
		if (s.children) yield* walk(s.children, path);
	}
}

/** Envelopes of one kind (same op) -> a plain summary object (JSON-friendly, for `--json` rows). */
export function summarize(envelopes) {
	const ts = envelopes.filter(Boolean);
	const ms = ts.map((t) => t.ms).sort((a, b) => a - b);
	const spans = new Map();
	const order = [];
	for (const t of ts) {
		for (const [path, s] of walk(t.spans)) {
			let e = spans.get(path);
			if (!e) {
				spans.set(path, (e = { ms: [], bytes: [], reads: [], misses: [], starts: [] }));
				order.push(path);
			}
			e.ms.push(s.ms);
			if (s.start !== undefined) e.starts.push(s.start);
			if (s.io) {
				e.bytes.push(s.io.bytes);
				e.reads.push(s.io.reads);
				e.misses.push(s.io.misses);
			}
		}
	}
	const counters = {};
	for (const k of new Set(ts.flatMap((t) => Object.keys(t.counters ?? {})))) counters[k] = round(mean(ts.map((t) => t.counters[k] ?? 0)));
	const kernels = {};
	for (const k of new Set(ts.flatMap((t) => Object.keys(t.kernels ?? {})))) {
		const per = ts.map((t) => t.kernels[k] ?? { calls: 0, ms: 0 });
		const kms = per.map((x) => x.ms).sort((a, b) => a - b);
		kernels[k] = { calls: round(mean(per.map((x) => x.calls))), ms: round(mean(kms), 3), p95: round(pct(kms, 0.95), 3) };
	}
	return {
		n: ts.length,
		op: ts[0]?.op ?? null,
		sdk: [...new Set(ts.map((t) => t.sdk))].join(","),
		clock: [...new Set(ts.map((t) => t.clock))].join(","),
		ms: { p50: round(pct(ms, 0.5)), p95: round(pct(ms, 0.95)), p99: round(pct(ms, 0.99)), mean: round(mean(ms)) },
		io: { reads: round(mean(ts.map((t) => t.io.reads))), misses: round(mean(ts.map((t) => t.io.misses))), kb: round(mean(ts.map((t) => t.io.bytes)) / 1024, 1) },
		spans: order.map((path) => {
			const e = spans.get(path);
			const s = [...e.ms].sort((a, b) => a - b);
			return { path, n: e.ms.length, start: round(mean(e.starts)), p50: round(pct(s, 0.5)), p95: round(pct(s, 0.95)), mean: round(mean(s)), kb: round(mean(e.bytes) / 1024, 1), reads: round(mean(e.reads)), misses: round(mean(e.misses)) };
		}),
		counters,
		kernels,
	};
}

/** Prints a summary as aligned text lines (indented by `indent`). */
export function printSummary(sum, { indent = "  ", counters = true } = {}) {
	const out = [];
	out.push(`${indent}${sum.op ?? "?"} n=${sum.n} (${sum.sdk}, clock ${sum.clock}): p50 ${sum.ms.p50} ms  p95 ${sum.ms.p95}  p99 ${sum.ms.p99}  mean ${sum.ms.mean}  | ${sum.io.reads} reads ${sum.io.kb} KB ${sum.io.misses} misses`);
	out.push(`${indent}  ${"span".padEnd(26)} ${"start".padStart(7)} ${"p50".padStart(8)} ${"p95".padStart(8)} ${"mean".padStart(8)} ${"KB".padStart(8)} ${"reads".padStart(6)}`);
	for (const s of sum.spans) {
		const depth = s.path.split("/").length - 1;
		const name = `${"  ".repeat(depth)}${s.path.split("/").at(-1)}${s.n < sum.n ? ` (${s.n})` : ""}`;
		out.push(`${indent}  ${name.padEnd(26)} ${s.start.toFixed(2).padStart(7)} ${s.p50.toFixed(2).padStart(8)} ${s.p95.toFixed(2).padStart(8)} ${s.mean.toFixed(2).padStart(8)} ${s.kb.toFixed(1).padStart(8)} ${s.reads.toFixed(1).padStart(6)}`);
	}
	const ks = Object.entries(sum.kernels);
	if (ks.length) {
		const total = ks.reduce((a, [, k]) => a + k.ms, 0);
		out.push(`${indent}  kernels (CPU ms per op, mean): ${ks.map(([n, k]) => `${n} ${k.ms} (${k.calls} calls, p95 ${k.p95})`).join("  ")}  = ${round(total, 3)} of ${sum.ms.mean} ms`);
	}
	if (counters) {
		const cs = Object.entries(sum.counters).filter(([, v]) => v !== 0);
		if (cs.length) out.push(`${indent}  counters (mean): ${cs.map(([k, v]) => `${k} ${v}`).join("  ")}`);
	}
	console.log(out.join("\n"));
}
