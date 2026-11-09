/**
 * The lexical side of an index: its segments (`manifest.segments`, each a `THREADIX` container object with a BM25
 * index over a batch of documents and/or their `docs` entries) read on demand through the wasm `SegmentReader`.
 *
 * Every wasm call either succeeds or throws `MissingRanges` naming every blob-absolute range its current round
 * still needs; `LexSegment.run` fetches them from the store (one multi-range request where the store can, else one
 * range per request), feeds them and calls again. The resident parts (container table, `meta`, section headers,
 * the terms and docs directories) are fetched once per segment per isolate; after that a search is two rounds per
 * segment (the query terms' buckets, then their posting lists), run for every segment concurrently, a docs
 * lookup one or two (buckets, then texts) and a key lookup one (the keys' buckets, newest segment first). Scores
 * use corpus-wide statistics (N, average length, per-term df summed over the segments), so the segments together
 * score exactly like one index over all documents.
 */
import type { BlobRange, BlobStore, BlobTrace } from "./store.js";
import { newTrace, nowMs, phaseStat, snapTrace } from "./store.js";
import { envelope, ioOf, Kernels, mergeKernels, phaseSpan, shiftSpans, type TraceEnvelope, type TraceSpan } from "./trace.js";
import type * as W from "./wasm/thread.js";
import { filterMatches } from "./attrs.js";
import { parseHits, tombForSegment, tombHides, ThreadError, type ResolvedLexConfig, type SegmentRef, type TombState } from "./wire.js";

export interface Range {
	offset: number;
	len: number;
}

/** The error the wasm reader throws when bytes are not held yet (`name === "MissingRanges"`). */
export interface MissingRangesError extends Error {
	ranges: Range[];
}

export function isMissingRanges(e: unknown): e is MissingRangesError {
	return e instanceof Error && e.name === "MissingRanges" && Array.isArray((e as MissingRangesError).ranges);
}

/** Segments up to this size are cached whole on a miss (`LexSegmentOptions.fillBytes`). */
export const LEX_FILL_BYTES = 2 * 1024 * 1024;

/** Ranges closer than this are fetched as one (the core's `PLAN_GAP`). */
export const PLAN_GAP = 4096;

/** Sorts and merges ranges whose gap is at most `maxGap` bytes. */
export function coalesce(ranges: readonly Range[], maxGap = PLAN_GAP): Range[] {
	const sorted = ranges
		.filter((r) => r.len > 0)
		.map((r) => ({ offset: r.offset, len: r.len }))
		.sort((a, b) => a.offset - b.offset);
	const out: Range[] = [];
	for (const r of sorted) {
		const last = out[out.length - 1];
		if (last && r.offset <= last.offset + last.len + maxGap) {
			const end = Math.max(last.offset + last.len, r.offset + r.len);
			last.len = end - last.offset;
		} else out.push(r);
	}
	return out;
}

/** What a lexical call cost (rounds and bytes over every segment). */
export interface LexIo {
	rounds: number;
	ranges: number;
	bytes: number;
}

export interface LexSegmentInfo {
	numDocs: number;
	totalLen: number;
	numTerms: number;
	numPostings: number;
	k1: number;
	b: number;
	lengths: "exact" | "byte" | null;
	bucketTerms: number;
	termsDirectoryBytes: number;
	termsBytes: number;
	postingsBytes: number;
	docs: { n: number; hasText: boolean; bucketDocs: number; directoryBytes: number; textBytes: number; nbytes: number } | null;
	keys: { n: number; bucketKeys: number; directoryBytes: number; nbytes: number } | null;
}

export interface LexSegmentOptions {
	/** Transient bytes (buckets, posting lists, texts) a reader may hold before they are evicted (default 8 MB). */
	transientBudget?: number;
	/** Fetch rounds per call before giving up (default 8). */
	maxRounds?: number;
	/**
	 * Segments up to this size are read whole on a cache miss (`fill`) and cached whole; larger ones are read by range
	 * with their size, which a Cache API store serves from cached blocks (`CachedBlobStoreOptions.blockBytes`)
	 * (default `LEX_FILL_BYTES`, 2 MiB).
	 */
	fillBytes?: number;
}

/**
 * One segment object under demand paging. `open` fetches the resident parts; `run` executes a wasm call through the
 * `MissingRanges` loop. One per segment key per isolate; wasm objects belong to the instance epoch they were
 * created in (dropped without `free` after a reset).
 */
export class LexSegment {
	readonly reader: W.SegmentReader;
	info: LexSegmentInfo | null = null;
	hasLex = false;
	hasDocs = false;
	hasKeys = false;
	/** The tokenizer config recorded in the segment's `meta` (JSON) or null. */
	tokenizer: string | null = null;
	readonly transientBudget: number;
	readonly maxRounds: number;
	readonly fillBytes: number;
	private transient: Range[] = [];
	private transientBytes = 0;
	private resident: Range[] = [];
	private opened = false;

	constructor(
		readonly w: typeof W,
		readonly store: BlobStore,
		readonly ref: SegmentRef,
		readonly epoch: number,
		readonly size: number,
		opts: LexSegmentOptions = {},
	) {
		this.reader = new w.SegmentReader(size);
		this.transientBudget = opts.transientBudget ?? 8 * 1024 * 1024;
		this.maxRounds = opts.maxRounds ?? 8;
		this.fillBytes = opts.fillBytes ?? LEX_FILL_BYTES;
	}

	/** Opens the segment: container table + meta, section headers, directories (three fetches cold, once). */
	static async open(w: typeof W, store: BlobStore, ref: SegmentRef, epoch: number, io: LexIo, trace?: BlobTrace, opts: LexSegmentOptions = {}): Promise<LexSegment> {
		let size = ref.bytes;
		if (!size) {
			const h = await store.head(ref.key);
			if (!h) throw new ThreadError(`missing segment ${ref.key}`, 404);
			size = h.size;
		}
		const seg = new LexSegment(w, store, ref, epoch, size, opts);
		try {
			await seg.fetch(seg.reader.plan_open() as Range[], io, trace, true);
			const o = (await seg.run(() => seg.reader.open(), io, trace, true)) as { hasLex: boolean; hasDocs: boolean; hasKeys: boolean; tokenizer: Record<string, unknown> | null };
			seg.hasLex = o.hasLex;
			seg.hasDocs = o.hasDocs;
			seg.hasKeys = o.hasKeys;
			seg.tokenizer = o.tokenizer ? JSON.stringify(o.tokenizer) : null;
			await seg.fetch(seg.reader.plan_headers() as Range[], io, trace, true);
			seg.info = (await seg.run(() => seg.reader.info(), io, trace, true)) as LexSegmentInfo;
			await seg.fetch((await seg.run(() => seg.reader.plan_directories(), io, trace, true)) as Range[], io, trace, true);
			seg.opened = true;
			return seg;
		} catch (e) {
			seg.free();
			throw e;
		}
	}

	get stats(): { numDocs: number; totalLen: number } {
		return { numDocs: this.info?.numDocs ?? 0, totalLen: this.info?.totalLen ?? 0 };
	}

	/** Fetches `ranges` (coalesced) and feeds them; `resident` ranges are never evicted. */
	private async fetch(ranges: Range[], io: LexIo, trace: BlobTrace | undefined, resident: boolean): Promise<void> {
		const wanted = coalesce(ranges).filter((r) => r.offset < this.size).map((r) => ({ offset: r.offset, len: Math.min(r.len, this.size - r.offset) }));
		const missing = this.reader.missing(new Float64Array(wanted.flatMap((r) => [r.offset, r.len]))) as Range[];
		if (missing.length === 0) return;
		const blobRanges: BlobRange[] = missing.map((r) => ({ offset: r.offset, length: r.len }));
		// A small segment is cached whole on its first miss; a larger one is read by range with its size, so a cache
		// with blocks serves it from them (a block miss reads and caches that block only, never the whole segment).
		const opts = { trace, fill: this.size <= this.fillBytes, size: this.size };
		let parts: Uint8Array[] | null = null;
		if (blobRanges.length > 1 && this.store.getRanges) parts = await this.store.getRanges(this.ref.key, blobRanges, opts);
		if (!parts) {
			parts = await Promise.all(
				blobRanges.map(async (r) => {
					const b = await this.store.get(this.ref.key, r, opts);
					if (!b) throw new ThreadError(`missing segment ${this.ref.key}`, 404);
					return b;
				}),
			);
		}
		for (let i = 0; i < missing.length; i++) {
			this.reader.feed(missing[i].offset, parts[i]);
			io.bytes += parts[i].byteLength;
			if (resident) this.resident.push(missing[i]);
			else {
				this.transient.push(missing[i]);
				this.transientBytes += missing[i].len;
			}
		}
		io.ranges += missing.length;
		io.rounds += 1;
	}

	/** Calls `fn` until it stops throwing `MissingRanges`, fetching what it asks for. */
	async run<T>(fn: () => T, io: LexIo, trace?: BlobTrace, resident = false): Promise<T> {
		for (let round = 0; ; round++) {
			try {
				const v = fn();
				this.trim();
				return v;
			} catch (e) {
				if (!isMissingRanges(e)) throw e;
				if (round >= this.maxRounds) throw new ThreadError(`${this.ref.key}: still missing ${e.ranges.length} range(s) after ${this.maxRounds} rounds`);
				await this.fetch(e.ranges, io, trace, resident);
			}
		}
	}

	/** Evicts the transient bytes once they exceed the budget (the resident parts stay). */
	private trim(): void {
		if (this.transientBytes <= this.transientBudget || !this.transient.length) return;
		const flat = new Float64Array(this.transient.flatMap((r) => [r.offset, r.len]));
		this.reader.evict_ranges(flat);
		this.transient = [];
		this.transientBytes = 0;
		// Re-feed nothing: resident ranges were never in the transient list, but a coalesced transient fetch may
		// have overlapped one (same object, adjacent sections); the next call simply re-fetches that sliver.
	}

	/** Bytes held in wasm memory for this segment. */
	heldBytes(): number {
		return this.reader.held_bytes();
	}

	free(): void {
		try {
			this.reader.free();
		} catch {
			/* poisoned instance */
		}
	}
}

/** One result of a lexical search. */
export interface LexHit {
	id: number;
	/** BM25 score (float32), corpus-wide statistics. */
	score: number;
}

export interface LexSearchOptions {
	trace?: BlobTrace;
	/** Also return the query's tokens and how many segments had each. */
	explain?: boolean;
	/**
	 * A compiled attribute filter (`compileFilter`): the BM25 list is taken deeper than `k`, each candidate's
	 * attrs word is resolved through the `docs` store and non-matching hits are dropped, deepening until `k` match
	 * or the lists are exhausted (at most `maxDepth`).
	 */
	filter?: Uint32Array;
	/** First depth of a filtered search (default `max(4 k, 32)`) and the deepest it goes (default 1 << 16). */
	filterDepth?: number;
	maxDepth?: number;
}

export interface LexSearchResult {
	hits: LexHit[];
	/** The query's tokens (after the index's tokenizer). */
	tokens: string[];
	/** Tokens found in at least one segment. */
	found: number;
	stats: {
		segments: number;
		/** Corpus-wide counts used for scoring. */
		numDocs: number;
		totalLen: number;
		/** Per distinct token: its corpus-wide df. */
		df: Record<string, number>;
		io: LexIo;
		ms: number;
		/** With a filter: the list depth used and the candidates the filter dropped. */
		depth?: number;
		filterDropped?: number;
		/**
		 * The search's trace envelope (docs/observability.md): `lookup` (term lookups in every segment) and `score`
		 * (BM25 over the postings) spans, one `pass` + `attrs` pair per depth of a filtered search. Kernel `calls`
		 * count attempts: a reader that needs pages it does not hold yet is run again after they are fetched.
		 */
		trace: TraceEnvelope;
	};
}

/** One page of a key listing. */
export interface KeyPage {
	/** `{key, id}` in key order. */
	entries: { key: string; id: number }[];
	/** Whether more keys with the prefix may follow: continue with `after: next`. */
	more: boolean;
	/** The cursor for the next page (the last key examined; set when `more`). */
	next?: string;
}

export interface ListKeysOptions {
	/** Keys starting with this (default: every key). */
	prefix?: string;
	/** Only keys after this one (the last key of the previous page). */
	after?: string;
	/** Page size (default 100, at most 10,000). */
	limit?: number;
}

export const MAX_LIST_KEYS = 10_000;

/** A resolved document. */
export interface DocRecord {
	id: number;
	meta: string;
	/** The document's attribute word (0 when none was stored). */
	attrs: number;
	/** The text, when stored and requested. */
	text?: string;
}

/**
 * The lexical index of one revision: its segments, tokenizer and tombstones. Built by `ThreadIndex` per revision;
 * segment handles are shared across revisions by key.
 */
export class LexIndex {
	readonly tokenizer: W.LexTokenizer;
	constructor(
		readonly w: typeof W,
		readonly config: ResolvedLexConfig,
		/** Oldest first (a compaction's base first). */
		readonly segments: { ref: SegmentRef; seg: LexSegment }[],
		readonly tomb: TombState | null,
	) {
		this.tokenizer = new w.LexTokenizer(JSON.stringify(config.tokenizer ?? {}));
	}

	get hasLex(): boolean {
		return this.segments.some((s) => s.seg.hasLex);
	}
	get hasDocs(): boolean {
		return this.segments.some((s) => s.seg.hasDocs);
	}
	get hasKeys(): boolean {
		return this.segments.some((s) => s.seg.hasKeys);
	}

	/** Corpus-wide statistics over the lexical segments. */
	stats(): { numDocs: number; totalLen: number } {
		let numDocs = 0;
		let totalLen = 0;
		for (const { seg } of this.segments) {
			if (!seg.hasLex) continue;
			numDocs += seg.stats.numDocs;
			totalLen += seg.stats.totalLen;
		}
		return { numDocs, totalLen };
	}

	tokenize(text: string): string[] {
		return this.tokenizer.tokenize(text);
	}

	/**
	 * BM25 top `k` over every segment: tokenize, look the distinct tokens up in every lexical segment (one round
	 * each, concurrently), sum their df, score every segment with the corpus-wide statistics (one round each),
	 * merge. Tombstoned ids never take a slot.
	 */
	async search(text: string, k: number, opts: LexSearchOptions = {}): Promise<LexSearchResult> {
		if (!opts.filter?.length) return this.searchUnfiltered(text, k, opts);
		// A filtered search: take the list deeper, resolve the candidates' attrs through `docs`, keep the matches.
		const t0 = Date.now();
		const tt0 = nowMs();
		const trace = opts.trace ?? newTrace();
		opts = { ...opts, trace };
		const spans: TraceSpan[] = [];
		const passKernels: TraceEnvelope["kernels"][] = [];
		const filter = opts.filter!;
		const attrsOf = new Map<number, number>();
		let depth = Math.max(opts.filterDepth ?? Math.max(4 * k, 32), k);
		const maxDepth = Math.max(opts.maxDepth ?? 1 << 16, depth);
		let dropped = 0;
		let rounds = 0;
		let ranges = 0;
		for (;;) {
			const tPass = nowMs() - tt0;
			const raw = await this.searchUnfiltered(text, depth, opts);
			const pt = raw.stats.trace;
			spans.push({ name: "pass", start: tPass, ms: pt.ms, io: pt.io, children: shiftSpans(pt.spans, tPass) });
			passKernels.push(pt.kernels);
			const unknown = raw.hits.filter((h) => !attrsOf.has(h.id)).map((h) => h.id);
			if (unknown.length) {
				const tA = nowMs();
				const trA = snapTrace(trace);
				const found = await this.attrs(unknown, trace, raw.stats.io);
				for (const id of unknown) attrsOf.set(id, found.get(id) ?? 0);
				spans.push(phaseSpan("attrs", tA - tt0, phaseStat(nowMs() - tA, trace, trA)));
			}
			rounds += raw.stats.io.rounds;
			ranges += raw.stats.io.ranges;
			const hits = raw.hits.filter((h) => filterMatches(filter, attrsOf.get(h.id)!));
			dropped = raw.hits.length - hits.length;
			if (hits.length >= k || raw.hits.length < depth || depth >= maxDepth) {
				const out = hits.slice(0, k);
				const env = envelope({
					op: "lexical",
					t0: tt0,
					at: t0,
					attrs: { ...pt.attrs, filtered: true },
					io: ioOf(trace),
					spans,
					counters: { ...pt.counters, hits: out.length, rounds, ranges, depth, filterDropped: dropped, passes: passKernels.length },
					kernels: mergeKernels(...passKernels),
				});
				return { ...raw, hits: out, stats: { ...raw.stats, depth, filterDropped: dropped, ms: Date.now() - t0, trace: env } };
			}
			depth = Math.min(maxDepth, depth * 4);
		}
	}

	/** The attrs words of `ids` through the `docs` sections (newest live copy); ids without an entry are absent. */
	async attrs(ids: readonly number[], trace?: BlobTrace, io?: LexIo): Promise<Map<number, number>> {
		const out = new Map<number, number>();
		const r = await this.docs(ids, { trace });
		if (io) {
			io.rounds += r.io.rounds;
			io.ranges += r.io.ranges;
			io.bytes += r.io.bytes;
		}
		for (const [id, d] of r.docs) out.set(id, d.attrs);
		return out;
	}

	private async searchUnfiltered(text: string, k: number, opts: LexSearchOptions = {}): Promise<LexSearchResult> {
		const t0 = Date.now();
		const tt0 = nowMs();
		const trace = opts.trace ?? newTrace();
		const kern = new Kernels();
		const spans: TraceSpan[] = [];
		const io: LexIo = { rounds: 0, ranges: 0, bytes: 0 };
		const tokens = kern.time("tokenize", () => this.tokenize(text));
		const weights = new Map<string, number>();
		for (const t of tokens) weights.set(t, (weights.get(t) ?? 0) + 1);
		const distinct = [...weights.keys()];
		const lexSegs = this.segments.filter((s) => s.seg.hasLex);
		const { numDocs, totalLen } = this.stats();
		const traced = (hits: LexHit[], found: number): TraceEnvelope =>
			envelope({
				op: "lexical",
				t0: tt0,
				at: t0,
				attrs: { topK: k, filtered: false },
				io: ioOf(trace),
				spans,
				counters: { hits: hits.length, segments: lexSegs.length, numDocs, tokens: tokens.length, distinct: distinct.length, found, rounds: io.rounds, ranges: io.ranges },
				kernels: kern.done(),
			});
		const empty = (df: Record<string, number> = {}): LexSearchResult => ({ hits: [], tokens, found: 0, stats: { segments: lexSegs.length, numDocs, totalLen, df, io, ms: Date.now() - t0, trace: traced([], 0) } });
		if (!distinct.length || !lexSegs.length || k < 1) return empty();
		const tL = nowMs();
		const trL = snapTrace(trace);
		const infos = await Promise.all(lexSegs.map(({ seg }) => seg.run(() => kern.time("bm25Lookup", () => seg.reader.lookup(distinct)), io, trace)));
		spans.push(phaseSpan("lookup", tL - tt0, phaseStat(nowMs() - tL, trace, trL)));
		const dfs = new Float64Array(distinct.length);
		for (const inf of infos) for (let i = 0; i < distinct.length; i++) if (inf[3 * i] >= 0) dfs[i] += inf[3 * i];
		const dfRecord: Record<string, number> = {};
		distinct.forEach((t, i) => (dfRecord[t] = dfs[i]));
		const found = distinct.filter((_, i) => dfs[i] > 0).length;
		if (!found) return empty(dfRecord);
		const w = Float32Array.from(distinct, (t) => weights.get(t)!);
		const tS = nowMs();
		const trS = snapTrace(trace);
		const packed = await Promise.all(
			lexSegs.map(({ ref, seg }, i) => {
				const inf = infos[i];
				if (!inf.some((v, j) => j % 3 === 0 && v >= 0)) return Promise.resolve(null);
				const tomb = this.tomb ? tombForSegment(this.tomb, ref.addedRev) : new Uint8Array(0);
				return seg.run(() => kern.time("bm25Score", () => seg.reader.search(inf, dfs, w, numDocs, totalLen, this.config.k1, this.config.b, k, tomb)), io, trace);
			}),
		);
		spans.push(phaseSpan("score", tS - tt0, phaseStat(nowMs() - tS, trace, trS)));
		const all: LexHit[] = [];
		for (const p of packed) if (p) for (const h of parseHits(p)) all.push({ id: h.id, score: h.score });
		// Ids are unique across live segments (older copies of a replaced id are tombstoned); keep the best anyway.
		const best = new Map<number, number>();
		for (const h of all) if ((best.get(h.id) ?? -Infinity) < h.score) best.set(h.id, h.score);
		const hits = [...best].map(([id, score]) => ({ id, score })).sort((a, b) => b.score - a.score || a.id - b.id).slice(0, k);
		return { hits, tokens, found, stats: { segments: lexSegs.length, numDocs, totalLen, df: dfRecord, io, ms: Date.now() - t0, trace: traced(hits, found) } };
	}

	/**
	 * Resolves ids through the `docs` sections, newest segment first (the live copy of a replaced id is the newest
	 * one not hidden by a later tombstone set). Ids with no live entry are absent from the result.
	 */
	async docs(ids: readonly number[], opts: { text?: boolean; trace?: BlobTrace } = {}): Promise<{ docs: Map<number, DocRecord>; io: LexIo }> {
		const io: LexIo = { rounds: 0, ranges: 0, bytes: 0 };
		const out = new Map<number, DocRecord>();
		let pending = [...new Set(ids)];
		for (let i = this.segments.length - 1; i >= 0 && pending.length; i--) {
			const { ref, seg } = this.segments[i];
			if (!seg.hasDocs) continue;
			const want = pending.filter((id) => !tombHides(this.tomb, id, ref.addedRev));
			if (!want.length) continue;
			const got = (await seg.run(() => seg.reader.docs_get(Uint32Array.from(want), opts.text ?? false), io, opts.trace)) as ({ id: number; meta: string; text: string | null; attrs: number } | null)[];
			const resolved = new Set<number>();
			got.forEach((g, j) => {
				if (!g) return;
				resolved.add(want[j]);
				out.set(want[j], { id: want[j], meta: g.meta, attrs: g.attrs, ...(g.text !== null && opts.text ? { text: g.text } : {}) });
			});
			pending = pending.filter((id) => !resolved.has(id));
		}
		return { docs: out, io };
	}

	/**
	 * Resolves string keys to ids through the `keys` sections, newest segment first (a key's live entry is the
	 * newest one whose id a later tombstone set does not hide). Keys with no live entry are absent.
	 */
	async resolveKeys(keys: readonly string[], opts: { trace?: BlobTrace } = {}): Promise<{ ids: Map<string, number>; io: LexIo }> {
		const io: LexIo = { rounds: 0, ranges: 0, bytes: 0 };
		const out = new Map<string, number>();
		let pending = [...new Set(keys)];
		for (let i = this.segments.length - 1; i >= 0 && pending.length; i--) {
			const { ref, seg } = this.segments[i];
			if (!seg.hasKeys) continue;
			const got = (await seg.run(() => seg.reader.keys_get(pending), io, opts.trace)) as Float64Array;
			const resolved = new Set<string>();
			pending.forEach((key, j) => {
				const id = got[j];
				if (id < 0) return;
				// The newest copy decides: a hidden one means the key was deleted (or its id replaced under another key).
				resolved.add(key);
				if (!tombHides(this.tomb, id, ref.addedRev)) out.set(key, id);
			});
			pending = pending.filter((k) => !resolved.has(k));
		}
		return { ids: out, io };
	}

	/**
	 * Keys with a prefix, in key order, over every segment (the newest copy of a key decides whether it is live),
	 * paged with `after` / `more`.
	 */
	async listKeys(opts: ListKeysOptions & { trace?: BlobTrace } = {}): Promise<KeyPage & { io: LexIo }> {
		const io: LexIo = { rounds: 0, ranges: 0, bytes: 0 };
		const prefix = opts.prefix ?? "";
		const limit = Math.min(MAX_LIST_KEYS, Math.max(0, Math.floor(opts.limit ?? 100)));
		const after = opts.after;
		const segs = this.segments.filter((s) => s.seg.hasKeys);
		if (!limit || !segs.length) return { entries: [], more: false, io };
		const pages = await Promise.all(segs.map(({ seg }) => seg.run(() => seg.reader.keys_list(prefix, after, limit), io, opts.trace) as Promise<{ entries: [string, number][]; more: boolean }>));
		// Newest segment first: the first copy of a key decides whether it is live. A segment page that stops early
		// bounds what is known: keys past its last entry may have a newer copy there, so the merged page stops at the
		// smallest such cursor and the caller continues from it.
		const decided = new Set<string>();
		const live = new Map<string, number>();
		let cursor: string | null = null;
		for (let i = segs.length - 1; i >= 0; i--) {
			const page = pages[i];
			if (page.more && page.entries.length) {
				const last = page.entries[page.entries.length - 1][0];
				if (cursor === null || last < cursor) cursor = last;
			}
			for (const [key, id] of page.entries) {
				if (decided.has(key)) continue;
				decided.add(key);
				if (!tombHides(this.tomb, id, segs[i].ref.addedRev)) live.set(key, id);
			}
		}
		let entries = [...live].map(([key, id]) => ({ key, id })).sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
		let more = false;
		let next: string | undefined;
		if (cursor !== null) {
			entries = entries.filter((e) => e.key <= cursor!);
			more = true;
			next = cursor;
		}
		if (entries.length > limit) {
			entries.length = limit;
			more = true;
			next = entries[limit - 1].key;
		}
		return { entries, more, ...(more ? { next } : {}), io };
	}

	free(): void {
		try {
			this.tokenizer.free();
		} catch {
			/* poisoned instance */
		}
	}
}
