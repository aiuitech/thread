/**
 * Warm packs: the scan prefixes (bytes `[0, scanLen)`) of an index's most-probed slabs in ONE object, fetched with one
 * GET when an index opens (`open({ warmPack })`, in the background) and used to seed an in-process layer, so the
 * first queries of a cold process read those prefixes from memory instead of one range GET each. Probe frequency is
 * skewed (some cells are probed by many more queries than others), so a few hundred cells cover a useful share of
 * all probes; `bench/cold/hot_cells.mjs` measures the curve and `bench/cold/make_warm_pack.mjs` builds one.
 *
 * Format: a pack (`pack.ts`) whose directory records `{"kind": "warm", "gen", "rev", ...}` and whose entries are the
 * slabs' own keys (`idx/{gen}/slab/{cell}.{slabRev}`, so a prefix is only ever used for the immutable object it was
 * cut from), each marked `{"partial": true, "size": <whole slab bytes>}`. A seeded prefix serves ranges inside it;
 * anything else (the rerank records phase 2 reads, a whole-object read) goes to the store as before.
 */
import { MemorySink, PACK_FOOTER_BYTES, PACK_HEADER_BYTES, PackWriter, isPackHeader, parsePackDirectory, parsePackFooter, type PackSink } from "./pack.js";
import type { RevState } from "./query.js";
import { CachedBlobStore, newTrace, nowMs, phaseStat, type BlobHead, type BlobRange, type BlobStore, type GetOptions, type PhaseStat, type PutOptions } from "./store.js";
import { keys, slabRevOf, ThreadError } from "./wire.js";

export const WARM_PACK_KIND = "warm";

/** An entry of a warm pack: the first `bytes.byteLength` bytes of the object `key` (`size` bytes in all). */
export interface WarmEntry {
	key: string;
	bytes: Uint8Array;
	size: number;
}

/** Parses a warm pack held in memory: its metadata and its prefixes (views into `bytes`). */
export function parseWarmPack(bytes: Uint8Array): { meta: Record<string, unknown>; entries: WarmEntry[] } {
	if (bytes.byteLength < PACK_HEADER_BYTES + PACK_FOOTER_BYTES || !isPackHeader(bytes)) throw new ThreadError("not a thread pack (bad header)");
	const { dirOff, dirLen, count } = parsePackFooter(bytes.subarray(bytes.byteLength - PACK_FOOTER_BYTES));
	if (dirOff < PACK_HEADER_BYTES || dirOff + dirLen > bytes.byteLength - PACK_FOOTER_BYTES) throw new ThreadError(`pack directory [${dirOff}, +${dirLen}) is outside the pack`);
	const d = parsePackDirectory(bytes.subarray(dirOff, dirOff + dirLen), dirOff, count);
	if (d.meta.kind !== WARM_PACK_KIND) throw new ThreadError(`not a warm pack (kind ${JSON.stringify(d.meta.kind ?? null)})`);
	const entries: WarmEntry[] = [];
	for (const [key, e] of d.entries) entries.push({ key, bytes: bytes.subarray(e.offset, e.offset + e.length), size: e.size ?? e.length });
	return { meta: d.meta, entries };
}

export interface WarmPackOptions {
	/** Prefixes read concurrently from the source store (default 8). */
	concurrency?: number;
	/** Extra directory metadata (e.g. how the cells were chosen); `kind`, `gen`, `rev` and `cells` are set here. */
	meta?: Record<string, unknown>;
}

export interface WarmPackResult {
	gen: string;
	rev: number;
	/** The slab keys in the pack, in the order given (hottest first, by convention). */
	keys: string[];
	/** Pack size and the prefix bytes in it. */
	bytes: number;
	prefixBytes: number;
}

/**
 * Writes the warm pack of `cells` (format 2/3 slabs: their scan prefixes at the revision's slab revisions) to `sink`,
 * entries in the order given. Reads only the prefixes from `store`.
 */
export async function writeWarmPack(store: BlobStore, rev: RevState, cells: readonly number[], sink: PackSink, opts: WarmPackOptions = {}): Promise<WarmPackResult> {
	if (!rev.slabs) throw new ThreadError(`${rev.gen} rev ${rev.rev} has no slabs table (format 1 slabs are read whole: no prefixes to warm)`, 400);
	const slabs = rev.slabs;
	const items = cells.map((cell) => {
		const meta = slabs[cell];
		if (!meta) throw new ThreadError(`${rev.files.slabs} has no entry for cell ${cell}`, 400);
		return { key: keys.slab(rev.gen, cell, slabRevOf(meta)), scanLen: meta[0], size: meta[3] };
	});
	if (new Set(items.map((i) => i.key)).size !== items.length) throw new ThreadError("warm pack: a cell is listed twice", 400);
	const w = new PackWriter(sink, { ...opts.meta, kind: WARM_PACK_KIND, gen: rev.gen, rev: rev.rev, cells: items.length });
	const window = Math.max(1, opts.concurrency ?? 8);
	const pending: Promise<Uint8Array>[] = [];
	const read = (i: number) =>
		store.get(items[i].key, { offset: 0, length: items[i].scanLen }).then((b) => {
			if (!b) throw new ThreadError(`warm pack needs ${items[i].key}, which is missing`, 404);
			return b;
		});
	let prefixBytes = 0;
	for (let i = 0; i < items.length; i++) {
		while (pending.length < window && i + pending.length < items.length) pending.push(read(i + pending.length));
		const b = await pending.shift()!;
		prefixBytes += b.byteLength;
		await w.add(items[i].key, b, { partialOf: items[i].size });
	}
	const { bytes } = await w.finish();
	return { gen: rev.gen, rev: rev.rev, keys: items.map((i) => i.key), bytes, prefixBytes };
}

/** `writeWarmPack` into memory. */
export async function buildWarmPack(store: BlobStore, rev: RevState, cells: readonly number[], opts: WarmPackOptions = {}): Promise<{ bytes: Uint8Array; result: WarmPackResult }> {
	const sink = new MemorySink();
	const result = await writeWarmPack(store, rev, cells, sink, opts);
	return { bytes: sink.bytes(), result };
}

// ---------- seeding ----------

/**
 * Where a warm pack's prefixes go: an in-process layer that can serve ranges inside an object's first bytes and
 * sends everything else to the store. `WarmPrefixStore` is the one `open` uses; an in-process cache can implement
 * this to take the prefixes instead (it must keep them prefix-only: a read past the prefix goes to the store).
 */
export interface PrefixSeed {
	seedPrefix(key: string, bytes: Uint8Array): void;
}

export interface WarmPrefixStoreOptions {
	/** Keeps background fills alive (`ctx.waitUntil`). */
	waitUntil?: (p: Promise<unknown>) => void;
	/** Whole-object fills behind a cache running at once (default 2; 0 disables them). */
	fillConcurrency?: number;
}

/**
 * A store in front of another one that holds object PREFIXES in memory: a read of a range inside a held prefix is
 * served from memory (counted as a cache hit in the trace), every other read goes to the inner store unchanged
 * (without an extra await, so pass-through reads keep their timing). Prefixes are views into the warm pack they came
 * from (no copies).
 *
 * When the inner store is a cache (`CachedBlobStore`) and a served read asked for `fill` (a slab's scan prefix, which
 * on a cache miss would have fetched the whole slab into the cache), the whole object is fetched into the cache in the
 * background instead, so later queries find the slab's rerank records there: the warm pack takes the first touch off
 * the critical path without keeping the cache cold.
 */
export class WarmPrefixStore implements BlobStore, PrefixSeed {
	private readonly prefixes = new Map<string, Uint8Array>();
	/** Reads served from a prefix, and their bytes. */
	hits = 0;
	hitBytes = 0;
	/** Background whole-object fills behind a cache: queued, done, failed. */
	readonly fills = { queued: 0, done: 0, failed: 0 };
	private readonly fillQ: string[] = [];
	private readonly filling = new Set<string>();
	private fillActive = 0;
	private readonly fillConcurrency: number;
	private readonly waitUntil: ((p: Promise<unknown>) => void) | undefined;

	constructor(
		public readonly inner: BlobStore,
		opts: WarmPrefixStoreOptions = {},
	) {
		this.fillConcurrency = opts.fillConcurrency ?? 2;
		this.waitUntil = opts.waitUntil;
	}

	seedPrefix(key: string, bytes: Uint8Array): void {
		const cur = this.prefixes.get(key);
		if (!cur || cur.byteLength < bytes.byteLength) this.prefixes.set(key, bytes);
	}

	/** Keeps only the prefixes `keep` accepts (after a switch to a revision with other slab revisions). */
	retain(keep: (key: string) => boolean): void {
		for (const k of [...this.prefixes.keys()]) if (!keep(k)) this.prefixes.delete(k);
	}

	/** Whether `range` of `key` is held (no I/O). */
	holds(key: string, range: BlobRange): boolean {
		const p = this.prefixes.get(key);
		return !!p && range.offset >= 0 && range.length >= 0 && range.offset + range.length <= p.byteLength;
	}

	/** Prefixes held and their bytes. */
	get size(): { prefixes: number; bytes: number } {
		let bytes = 0;
		for (const b of this.prefixes.values()) bytes += b.byteLength;
		return { prefixes: this.prefixes.size, bytes };
	}

	private serve(key: string, range: BlobRange, opts?: GetOptions): Uint8Array {
		const b = this.prefixes.get(key)!.subarray(range.offset, range.offset + range.length);
		this.hits++;
		this.hitBytes += b.byteLength;
		if (opts?.trace) {
			opts.trace.hits++;
			opts.trace.bytes += b.byteLength;
		}
		if (opts?.fill) this.fillBehind(key);
		return b;
	}

	/** Fetches `key` whole into the cache behind this layer, in the background (once per key, bounded). */
	private fillBehind(key: string): void {
		if (!(this.inner instanceof CachedBlobStore) || this.fillConcurrency <= 0 || this.filling.has(key)) return;
		this.filling.add(key);
		this.fillQ.push(key);
		this.fills.queued++;
		this.pumpFills();
	}

	private pumpFills(): void {
		while (this.fillActive < this.fillConcurrency && this.fillQ.length) {
			const key = this.fillQ.shift()!;
			this.fillActive++;
			const p = this.inner
				.get(key) // a whole-object read through the cache: a miss fills it
				.then(
					() => void this.fills.done++,
					(e: unknown) => {
						this.fills.failed++;
						this.filling.delete(key); // may be retried by a later hit
						console.error(`thread-sdk: background fill of ${key} failed: ${e instanceof Error ? e.message : String(e)}`);
					},
				)
				.finally(() => {
					this.fillActive--;
					this.pumpFills();
				});
			this.waitUntil?.(p);
		}
	}

	get(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		if (range && this.holds(key, range)) return Promise.resolve(this.serve(key, range, opts));
		return this.inner.get(key, range, opts);
	}

	/**
	 * Present only when the inner store has `getRanges` (a store without it must not gain one that answers null at
	 * once: the query path would take and release a read slot for nothing). Phase 2 never reads inside a prefix.
	 */
	get getRanges(): ((key: string, ranges: BlobRange[], opts?: GetOptions) => Promise<Uint8Array[] | null>) | undefined {
		const inner = this.inner;
		if (!inner.getRanges) return undefined;
		return (key, ranges, opts) => {
			if (ranges.length && ranges.every((r) => this.holds(key, r))) {
				const p = this.prefixes.get(key)!;
				const out = ranges.map((r) => p.subarray(r.offset, r.offset + r.length));
				const n = out.reduce((a, b) => a + b.byteLength, 0);
				this.hits++;
				this.hitBytes += n;
				if (opts?.trace) {
					opts.trace.hits++;
					opts.trace.bytes += n;
				}
				return Promise.resolve(out);
			}
			return inner.getRanges!(key, ranges, opts);
		};
	}

	head(key: string): Promise<BlobHead | null> {
		return this.inner.head(key);
	}

	get put(): ((key: string, data: Uint8Array | string, opts?: PutOptions) => Promise<BlobHead | null>) | undefined {
		const inner = this.inner;
		return inner.put ? (key, data, opts) => inner.put!(key, data, opts) : undefined;
	}

	get list(): ((prefix: string) => Promise<string[]>) | undefined {
		const inner = this.inner;
		return inner.list ? (prefix) => inner.list!(prefix) : undefined;
	}
}

/** What loading a warm pack did (`index.warmStats`). */
export interface WarmStats {
	key: string;
	state: "loading" | "ready" | "failed";
	/** The GET (`ms` from the start of `open`, reads, bytes) once it finished. */
	fetch: PhaseStat | null;
	/** Entries in the pack; prefixes seeded; entries skipped (not a slab of the open revision); prefix bytes seeded. */
	entries: number;
	seeded: number;
	skipped: number;
	seededBytes: number;
	/** ms from the start of `open` until the prefixes were seeded. */
	readyMs: number | null;
	error?: string;
}

/** Starts the ONE GET of a warm pack (at `t0`, the start of `open`); resolves with its bytes and the read's stats. */
export function fetchWarmPack(source: BlobStore, key: string, t0 = nowMs()): Promise<{ bytes: Uint8Array; stat: PhaseStat }> {
	const trace = newTrace();
	return source.get(key, undefined, { trace }).then((b) => {
		if (!b) throw new ThreadError(`no warm pack at ${key}`, 404);
		return { bytes: b, stat: phaseStat(nowMs() - t0, trace) };
	});
}

/** Seeds `target` with the prefixes of a warm pack that `accept` takes (default: all). */
export function seedWarmPack(bytes: Uint8Array, target: PrefixSeed, accept: (key: string) => boolean = () => true): { entries: number; seeded: number; skipped: number; seededBytes: number; meta: Record<string, unknown> } {
	const { meta, entries } = parseWarmPack(bytes);
	let seeded = 0;
	let seededBytes = 0;
	for (const e of entries) {
		if (!accept(e.key)) continue;
		target.seedPrefix(e.key, e.bytes);
		seeded++;
		seededBytes += e.bytes.byteLength;
	}
	return { entries: entries.length, seeded, skipped: entries.length - seeded, seededBytes, meta };
}

/** The slab keys a revision reads (what a warm pack's entries must match to be used). */
export function revisionSlabKeys(rev: RevState): Set<string> {
	const out = new Set<string>();
	if (!rev.slabs) return out;
	rev.slabs.forEach((meta, cell) => out.add(keys.slab(rev.gen, cell, slabRevOf(meta))));
	return out;
}
