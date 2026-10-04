/**
 * The lexical side of an index: its segments (`manifest.segments`, each a `THREADIX` container object with a BM25
 * index over a batch of documents and/or their `docs` entries) read on demand through the wasm `SegmentReader`.
 *
 * Every wasm call either succeeds or throws `MissingRanges` naming every blob-absolute range its current round
 * still needs; `LexSegment.run` fetches them from the store (one multi-range request where the store can, else one
 * range per request), feeds them and calls again. The resident parts (container table, `meta`, section headers,
 * the terms and docs directories) are fetched once per segment per isolate; after that a search is two rounds per
 * segment (the query terms' buckets, then their posting lists), run for every segment concurrently, and a docs
 * lookup one or two (buckets, then texts). Scores use corpus-wide statistics (N, average length, per-term df
 * summed over the segments), so the segments together score exactly like one index over all documents.
 */
import type { BlobRange, BlobStore, BlobTrace } from "./store.js";
import type * as W from "./wasm/thread.js";
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
}

export interface LexSegmentOptions {
	/** Transient bytes (buckets, posting lists, texts) a reader may hold before they are evicted (default 8 MB). */
	transientBudget?: number;
	/** Fetch rounds per call before giving up (default 8). */
	maxRounds?: number;
	/** Objects up to this size are read whole on a cache miss (`fill`), larger ones by range (default 2 MB). */
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
		this.fillBytes = opts.fillBytes ?? 2 * 1024 * 1024;
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
			const o = (await seg.run(() => seg.reader.open(), io, trace, true)) as { hasLex: boolean; hasDocs: boolean; tokenizer: Record<string, unknown> | null };
			seg.hasLex = o.hasLex;
			seg.hasDocs = o.hasDocs;
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
		const fill = this.size <= this.fillBytes;
		let parts: Uint8Array[] | null = null;
		if (blobRanges.length > 1 && this.store.getRanges) parts = await this.store.getRanges(this.ref.key, blobRanges, { trace, fill });
		if (!parts) {
			parts = await Promise.all(
				blobRanges.map(async (r) => {
					const b = await this.store.get(this.ref.key, r, { trace, fill });
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
	};
}

/** A resolved document. */
export interface DocRecord {
	id: number;
	meta: string;
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
		const t0 = Date.now();
		const io: LexIo = { rounds: 0, ranges: 0, bytes: 0 };
		const tokens = this.tokenize(text);
		const weights = new Map<string, number>();
		for (const t of tokens) weights.set(t, (weights.get(t) ?? 0) + 1);
		const distinct = [...weights.keys()];
		const lexSegs = this.segments.filter((s) => s.seg.hasLex);
		const { numDocs, totalLen } = this.stats();
		const empty = (df: Record<string, number> = {}): LexSearchResult => ({ hits: [], tokens, found: 0, stats: { segments: lexSegs.length, numDocs, totalLen, df, io, ms: Date.now() - t0 } });
		if (!distinct.length || !lexSegs.length || k < 1) return empty();
		const infos = await Promise.all(lexSegs.map(({ seg }) => seg.run(() => seg.reader.lookup(distinct), io, opts.trace)));
		const dfs = new Float64Array(distinct.length);
		for (const inf of infos) for (let i = 0; i < distinct.length; i++) if (inf[3 * i] >= 0) dfs[i] += inf[3 * i];
		const dfRecord: Record<string, number> = {};
		distinct.forEach((t, i) => (dfRecord[t] = dfs[i]));
		const found = distinct.filter((_, i) => dfs[i] > 0).length;
		if (!found) return empty(dfRecord);
		const w = Float32Array.from(distinct, (t) => weights.get(t)!);
		const packed = await Promise.all(
			lexSegs.map(({ ref, seg }, i) => {
				const inf = infos[i];
				if (!inf.some((v, j) => j % 3 === 0 && v >= 0)) return Promise.resolve(null);
				const tomb = this.tomb ? tombForSegment(this.tomb, ref.addedRev) : new Uint8Array(0);
				return seg.run(() => seg.reader.search(inf, dfs, w, numDocs, totalLen, this.config.k1, this.config.b, k, tomb), io, opts.trace);
			}),
		);
		const all: LexHit[] = [];
		for (const p of packed) if (p) for (const h of parseHits(p)) all.push({ id: h.id, score: h.score });
		// Ids are unique across live segments (older copies of a replaced id are tombstoned); keep the best anyway.
		const best = new Map<number, number>();
		for (const h of all) if ((best.get(h.id) ?? -Infinity) < h.score) best.set(h.id, h.score);
		const hits = [...best].map(([id, score]) => ({ id, score })).sort((a, b) => b.score - a.score || a.id - b.id).slice(0, k);
		return { hits, tokens, found, stats: { segments: lexSegs.length, numDocs, totalLen, df: dfRecord, io, ms: Date.now() - t0 } };
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
			const got = (await seg.run(() => seg.reader.docs_get(Uint32Array.from(want), opts.text ?? false), io, opts.trace)) as ({ id: number; meta: string; text: string | null } | null)[];
			const resolved = new Set<number>();
			got.forEach((g, j) => {
				if (!g) return;
				resolved.add(want[j]);
				out.set(want[j], { id: want[j], meta: g.meta, ...(g.text !== null && opts.text ? { text: g.text } : {}) });
			});
			pending = pending.filter((id) => !resolved.has(id));
		}
		return { docs: out, io };
	}

	free(): void {
		try {
			this.tokenizer.free();
		} catch {
			/* poisoned instance */
		}
	}
}
