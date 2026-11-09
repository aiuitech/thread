/**
 * Trace envelopes: one JSON object per SDK operation (query, search, lexical, open, get, listKeys, upsert, delete),
 * the same shape in the TypeScript SDK, the Python SDK and the reference Worker, so a bench script, a log query or a
 * dashboard reads every host the same way. docs/observability.md has the schema and how to use it.
 *
 * - `spans`: the operation's steps, each with its start (ms from the operation's start), wall time and the store
 *   work done in it; children for nested steps (a search's dense list holds the dense query's phases).
 * - `counters`: work done (entries scanned, bytes, candidates, records read, ...), camelCase in every SDK.
 * - `kernels`: CPU in the wasm / native kernels, `{calls, ms}` per kernel. Real on Node and Python; in Cloudflare
 *   Workers the clock only advances across I/O (`clock: "io-only"`), so kernel ms and CPU-only spans read 0 there:
 *   use Workers Logs' `cpuTimeMs` or a CPU profile for Worker CPU.
 * - `events` (opt-in, `traceEvents: true`): one entry per store read of a dense query.
 *
 * Building an envelope costs a few small objects per operation and nothing inside the scan loops.
 */
import type { BlobTrace, PhaseStat } from "./store.js";
import { inWorkers, nowMs, roundMs } from "./store.js";

export const TRACE_VERSION = 1;

export type TraceOp = "query" | "search" | "lexical" | "open" | "get" | "listKeys" | "upsert" | "delete";

/**
 * Store work: origin/cache reads, cache hits and misses, 206/200 answers to range requests, bytes transferred, and
 * block fills (blocks read from the origin into the Cache API's block cache; each is also a read and a miss).
 */
export interface TraceIo {
	reads: number;
	hits: number;
	misses: number;
	r206: number;
	r200: number;
	bytes: number;
	fills: number;
}

export interface TraceSpan {
	name: string;
	/** ms from the operation's start (absent when the step's start is not known, e.g. a pack opened earlier). */
	start?: number;
	ms: number;
	io?: TraceIo;
	children?: TraceSpan[];
}

export interface KernelStat {
	calls: number;
	/** CPU ms (wall time around the calls; 0 under `clock: "io-only"`). */
	ms: number;
}

/** One store read of a dense query (`traceEvents: true`). */
export interface TraceEvent {
	/** `prefix` (phase-1 scan prefix), `slab` (a v1 slab, whole), `merged` (cross-slab pack range), `records` (phase-2 rerank records), `spec` (speculative records), `payload`. */
	kind: string;
	key: string;
	/** ms from the operation's start. */
	start: number;
	ms: number;
	/** Ranges requested in this read (a multi-range request has several). */
	ranges: number;
	bytes: number;
	hits: number;
	misses: number;
}

export interface TraceEnvelope {
	v: typeof TRACE_VERSION;
	op: TraceOp;
	/** Which implementation produced it. */
	sdk: "ts" | "py" | "worker";
	/** Start, ms since the epoch. */
	at: number;
	/** Wall time of the whole operation. */
	ms: number;
	/** `io-only` inside Cloudflare Workers: timers advance only across I/O, so CPU-only steps read 0. */
	clock: "real" | "io-only";
	/** Dimensions to group by: gen, rev, mode, routing, the query parameters in effect, ... */
	attrs: Record<string, string | number | boolean>;
	/** Store work of the whole operation. */
	io: TraceIo;
	spans: TraceSpan[];
	counters: Record<string, number>;
	kernels: Record<string, KernelStat>;
	events?: TraceEvent[];
}

/** Where envelopes go (`open({ onTrace, traceSample })`): called once per operation, sampled, never awaited. */
export interface TraceSink {
	onTrace?: (trace: TraceEnvelope) => void;
	/** Fraction of operations passed to `onTrace` (default 1). Envelopes are built and returned either way. */
	traceSample?: number;
}

export const clockKind = (): TraceEnvelope["clock"] => (inWorkers() ? "io-only" : "real");

export const zeroIo = (): TraceIo => ({ reads: 0, hits: 0, misses: 0, r206: 0, r200: 0, bytes: 0, fills: 0 });

/** The `TraceIo` part of a store trace or a phase stat. */
export const ioOf = (t: BlobTrace | PhaseStat): TraceIo => ({ reads: t.reads, hits: t.hits, misses: t.misses, r206: t.r206, r200: t.r200, bytes: t.bytes, fills: t.fills ?? 0 });

/** `a + b`, field by field. */
export function addIo(a: TraceIo, b: TraceIo | undefined): TraceIo {
	if (!b) return a;
	return { reads: a.reads + b.reads, hits: a.hits + b.hits, misses: a.misses + b.misses, r206: a.r206 + b.r206, r200: a.r200 + b.r200, bytes: a.bytes + b.bytes, fills: a.fills + (b.fills ?? 0) };
}

/** Adds a store trace's counters into another (a per-read trace into the query's). */
export function addTrace(into: BlobTrace, t: BlobTrace): void {
	into.reads += t.reads;
	into.hits += t.hits;
	into.misses += t.misses;
	into.r206 += t.r206;
	into.r200 += t.r200;
	into.bytes += t.bytes;
	if (t.warms) into.warms = (into.warms ?? 0) + t.warms;
	if (t.fills) into.fills = (into.fills ?? 0) + t.fills;
}

/** A span from a phase stat (`timeline.*`, `openStats.*`). */
export const phaseSpan = (name: string, start: number | undefined, p: PhaseStat, children?: TraceSpan[]): TraceSpan => ({
	name,
	...(start !== undefined ? { start: roundMs(start) } : {}),
	ms: p.ms,
	io: ioOf(p),
	...(children?.length ? { children } : {}),
});

/** The same spans `by` ms later (a nested operation placed inside its parent). */
export function shiftSpans(spans: TraceSpan[], by: number): TraceSpan[] {
	return spans.map((s) => ({ ...s, ...(s.start !== undefined ? { start: roundMs(s.start + by) } : {}), ...(s.children ? { children: shiftSpans(s.children, by) } : {}) }));
}

/** Kernel timers: `time("scan", () => w.scan_pq(...))` adds one call and its wall time. */
export class Kernels {
	readonly stats: Record<string, KernelStat> = {};
	time<T>(name: string, fn: () => T): T {
		const t = nowMs();
		try {
			return fn();
		} finally {
			this.add(name, 1, nowMs() - t);
		}
	}
	add(name: string, calls: number, ms: number): void {
		const k = this.stats[name] ?? (this.stats[name] = { calls: 0, ms: 0 });
		k.calls += calls;
		k.ms += ms;
	}
	/** The stats with ms rounded to µs. */
	done(): Record<string, KernelStat> {
		const out: Record<string, KernelStat> = {};
		for (const [k, v] of Object.entries(this.stats)) out[k] = { calls: v.calls, ms: roundMs(v.ms) };
		return out;
	}
}

/** Kernel stats of several operations, summed by name (a search's dense and lexical lists). */
export function mergeKernels(...all: (Record<string, KernelStat> | undefined)[]): Record<string, KernelStat> {
	const out: Record<string, KernelStat> = {};
	for (const ks of all) {
		if (!ks) continue;
		for (const [k, v] of Object.entries(ks)) {
			const o = out[k] ?? (out[k] = { calls: 0, ms: 0 });
			o.calls += v.calls;
			o.ms = roundMs(o.ms + v.ms);
		}
	}
	return out;
}

/** Counters of a nested operation under a prefix (`dense.scanned`, `lexical.rounds`). */
export function prefixCounters(prefix: string, c: Record<string, number> | undefined): Record<string, number> {
	const out: Record<string, number> = {};
	if (c) for (const [k, v] of Object.entries(c)) out[`${prefix}.${k}`] = v;
	return out;
}

/** Every finite number among an object's own top-level fields (a write result's counts and bytes). */
export function numericFields(o: object, skip: readonly string[] = []): Record<string, number> {
	const out: Record<string, number> = {};
	for (const [k, v] of Object.entries(o)) if (typeof v === "number" && Number.isFinite(v) && !skip.includes(k)) out[k] = v;
	return out;
}

export interface EnvelopeInit {
	op: TraceOp;
	/** Which implementation produced it (default `ts`; the reference Worker passes `worker`). */
	sdk?: TraceEnvelope["sdk"];
	/** `nowMs()` at the start (for `ms`) and the epoch ms at the start (for `at`). */
	t0: number;
	at: number;
	attrs?: TraceEnvelope["attrs"];
	io?: TraceIo;
	spans?: TraceSpan[];
	counters?: Record<string, number>;
	kernels?: Record<string, KernelStat>;
	events?: TraceEvent[];
	/** Override the wall time (default `nowMs() - t0`). */
	ms?: number;
}

/** A TS SDK envelope; `io` defaults to the sum of the top-level spans' io. */
export function envelope(e: EnvelopeInit): TraceEnvelope {
	const spans = e.spans ?? [];
	return {
		v: TRACE_VERSION,
		op: e.op,
		sdk: e.sdk ?? "ts",
		at: e.at,
		ms: roundMs(e.ms ?? nowMs() - e.t0),
		clock: clockKind(),
		attrs: e.attrs ?? {},
		io: e.io ?? spans.reduce((a, s) => addIo(a, s.io), zeroIo()),
		spans,
		counters: e.counters ?? {},
		kernels: e.kernels ?? {},
		...(e.events ? { events: e.events } : {}),
	};
}

/** Hands an envelope to the sink, sampled; a sink that throws never fails the operation. */
export function emitTrace(sink: TraceSink | undefined, t: TraceEnvelope | undefined): void {
	if (!sink?.onTrace || !t) return;
	const rate = sink.traceSample ?? 1;
	if (rate < 1 && Math.random() >= rate) return;
	try {
		sink.onTrace(t);
	} catch {
		/* observability must not break a query */
	}
}
