/**
 * Packs: one immutable file holding a whole index snapshot, so an index can be moved as a single object and
 * queried anywhere a `BlobStore` can range-read it (a directory, R2, S3, memory). `packIndex` writes the objects
 * one revision reads (`snapshotKeys`) plus an `idx/current.json` naming it; `PackedBlobStore` serves them back
 * under their original keys by translating reads into ranges of the pack. A pack is read-only: to write on top,
 * `copyObjects` it into a directory or a bucket.
 *
 * Layout (all integers little-endian; `docs/portability.md` has the specification):
 *
 *   [0, 16)      header: magic "THRDPACK", version u16 (1), flags u16 (0), reserved u32 (0)
 *   [16, D)      the objects, each at an 8-byte-aligned offset, zero padding between
 *   [D, D + L)   directory: UTF-8 JSON `{"entries":[[key, offset, length], ...], ...meta}`
 *   [end - 32)   footer: directory offset u64, directory length u64, entry count u32, version u16, flags u16,
 *                magic "THRDPACK"
 *
 * The footer comes last so a pack streams out without seeking back; a reader does one `head` and one or two range
 * reads to open it.
 */
import { newTrace, type BlobTrace, nowMs, phaseStat, type BlobHead, type BlobRange, type BlobStore, type GetOptions, type PhaseStat, type PutOptions, MemoryBlobStore } from "./store.js";
import { baseKeys, loadRevState, resolveTarget, snapshotKeys } from "./revision.js";
import type { RevState } from "./query.js";
import { chooseRouting, concatBytes, decodeJson, hasDense, keys as indexKeys, ThreadError, type RoutingOption } from "./wire.js";

export const PACK_MAGIC = "THRDPACK";
export const PACK_VERSION = 1;
export const PACK_HEADER_BYTES = 16;
export const PACK_FOOTER_BYTES = 32;
/** Bytes a reader fetches from the end of a pack to find the footer (the directory often fits in them). */
export const PACK_TAIL_READ = 64 * 1024;
const ALIGN = 8;

const ascii = new TextEncoder();
const magicBytes = ascii.encode(PACK_MAGIC);

export interface PackEntry {
	offset: number;
	length: number;
}

/** The parsed directory of a pack: entries by key plus whatever metadata the writer recorded (`gen`, `rev`). */
export interface PackDirectory {
	entries: Map<string, PackEntry>;
	meta: Record<string, unknown>;
}

const readU64 = (dv: DataView, o: number, what: string): number => {
	const v = dv.getBigUint64(o, true);
	if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new ThreadError(`pack ${what} does not fit in a number`);
	return Number(v);
};

const hasMagic = (b: Uint8Array, at: number): boolean => magicBytes.every((c, i) => b[at + i] === c);

/** Whether `b` starts with a pack header (the first 16 bytes of a file are enough). */
export function isPackHeader(b: Uint8Array): boolean {
	return b.byteLength >= PACK_HEADER_BYTES && hasMagic(b, 0) && new DataView(b.buffer, b.byteOffset, b.byteLength).getUint16(8, true) === PACK_VERSION;
}

export function packHeader(): Uint8Array {
	const h = new Uint8Array(PACK_HEADER_BYTES);
	h.set(magicBytes, 0);
	new DataView(h.buffer).setUint16(8, PACK_VERSION, true);
	return h;
}

export function packFooter(dirOff: number, dirLen: number, count: number): Uint8Array {
	const f = new Uint8Array(PACK_FOOTER_BYTES);
	const dv = new DataView(f.buffer);
	dv.setBigUint64(0, BigInt(dirOff), true);
	dv.setBigUint64(8, BigInt(dirLen), true);
	dv.setUint32(16, count, true);
	dv.setUint16(20, PACK_VERSION, true);
	f.set(magicBytes, 24);
	return f;
}

/** Parses the last 32 bytes of a pack. */
export function parsePackFooter(f: Uint8Array): { dirOff: number; dirLen: number; count: number } {
	if (f.byteLength !== PACK_FOOTER_BYTES) throw new ThreadError(`pack footer needs ${PACK_FOOTER_BYTES} bytes, have ${f.byteLength}`);
	if (!hasMagic(f, 24)) throw new ThreadError("not a thread pack (bad footer magic)");
	const dv = new DataView(f.buffer, f.byteOffset, f.byteLength);
	const version = dv.getUint16(20, true);
	if (version !== PACK_VERSION) throw new ThreadError(`unknown pack version ${version}`);
	return { dirOff: readU64(dv, 0, "directory offset"), dirLen: readU64(dv, 8, "directory length"), count: dv.getUint32(16, true) };
}

/** Parses a pack directory and checks every entry lies inside `[PACK_HEADER_BYTES, dirOff)`. */
export function parsePackDirectory(b: Uint8Array, dirOff: number, count: number): PackDirectory {
	const j = decodeJson<{ entries?: unknown } & Record<string, unknown>>(b);
	if (!j || typeof j !== "object" || !Array.isArray(j.entries)) throw new ThreadError("pack directory has no entries");
	if (j.entries.length !== count) throw new ThreadError(`pack directory lists ${j.entries.length} entries, footer says ${count}`);
	const entries = new Map<string, PackEntry>();
	for (const e of j.entries as unknown[]) {
		if (!Array.isArray(e) || e.length < 3 || typeof e[0] !== "string" || !Number.isInteger(e[1]) || !Number.isInteger(e[2])) throw new ThreadError("pack directory: bad entry");
		const [key, offset, length] = e as [string, number, number];
		if (offset < PACK_HEADER_BYTES || length < 0 || offset + length > dirOff) throw new ThreadError(`pack entry ${key} [${offset}, +${length}) is outside the data area`);
		if (entries.has(key)) throw new ThreadError(`pack directory lists ${key} twice`);
		entries.set(key, { offset, length });
	}
	const { entries: _e, ...meta } = j;
	return { entries, meta };
}

/** The objects of a pack held in memory, by key (views into `bytes`). */
export function packEntries(bytes: Uint8Array): { entries: Map<string, Uint8Array>; meta: Record<string, unknown> } {
	if (bytes.byteLength < PACK_HEADER_BYTES + PACK_FOOTER_BYTES || !isPackHeader(bytes)) throw new ThreadError("not a thread pack (bad header)");
	const { dirOff, dirLen, count } = parsePackFooter(bytes.subarray(bytes.byteLength - PACK_FOOTER_BYTES));
	if (dirOff < PACK_HEADER_BYTES || dirOff + dirLen > bytes.byteLength - PACK_FOOTER_BYTES) throw new ThreadError(`pack directory [${dirOff}, +${dirLen}) is outside the pack`);
	const d = parsePackDirectory(bytes.subarray(dirOff, dirOff + dirLen), dirOff, count);
	const entries = new Map<string, Uint8Array>();
	for (const [k, e] of d.entries) entries.set(k, bytes.subarray(e.offset, e.offset + e.length));
	return { entries, meta: d.meta };
}

// ---------- reading ----------

/**
 * The objects of a pack under their original keys, read as ranges of one object `key` of `inner`. Read-only.
 * Put a cache in front of this store (`open` does), never behind it: a cache behind it would fetch the whole pack
 * on a miss, which is why `fill` is never forwarded.
 */
export class PackedBlobStore implements BlobStore {
	private constructor(
		public readonly inner: BlobStore,
		public readonly key: string,
		/** Size and etag of the pack object. */
		public readonly pack: BlobHead,
		public readonly directory: PackDirectory,
		/**
		 * What `open` did: the `head` (counted as a read of 0 bytes), the tail read and, when the directory does not
		 * fit in the tail, the directory read. `thread.open` reports it as `openStats.pack`.
		 */
		public readonly openStats: PhaseStat,
	) {}

	/** Reads the footer and directory of the pack at `key` in `inner`. */
	static async open(inner: BlobStore, key: string): Promise<PackedBlobStore> {
		const t0 = nowMs();
		const trace = newTrace();
		const h = await inner.head(key);
		trace.reads++;
		if (!h) throw new ThreadError(`no pack at ${key}`, 404);
		if (h.size < PACK_HEADER_BYTES + PACK_FOOTER_BYTES) throw new ThreadError(`${key} is too small to be a pack (${h.size} bytes)`);
		const tailLen = Math.min(h.size, PACK_TAIL_READ);
		const tail = await inner.get(key, { offset: h.size - tailLen, length: tailLen }, { trace });
		if (!tail) throw new ThreadError(`no pack at ${key}`, 404);
		const { dirOff, dirLen, count } = parsePackFooter(tail.subarray(tail.byteLength - PACK_FOOTER_BYTES));
		if (dirOff < PACK_HEADER_BYTES || dirOff + dirLen > h.size - PACK_FOOTER_BYTES) throw new ThreadError(`${key}: directory [${dirOff}, +${dirLen}) is outside the pack`);
		const tailStart = h.size - tailLen;
		let dir: Uint8Array;
		if (dirOff >= tailStart) dir = tail.subarray(dirOff - tailStart, dirOff - tailStart + dirLen);
		else {
			const d = await inner.get(key, { offset: dirOff, length: dirLen }, { trace });
			if (!d) throw new ThreadError(`no pack at ${key}`, 404);
			dir = d;
		}
		const directory = parsePackDirectory(dir, dirOff, count);
		return new PackedBlobStore(inner, key, h, directory, phaseStat(nowMs() - t0, trace));
	}

	/** A pack held in memory (tests, small indexes shipped inside a bundle). */
	static fromBytes(bytes: Uint8Array, key = "index.threadpack"): Promise<PackedBlobStore> {
		const m = new MemoryBlobStore();
		m.objects.set(key, { bytes, etag: "pack" });
		return PackedBlobStore.open(m, key);
	}

	/** `gen` / `rev` the pack records (absent for packs written with other metadata). */
	get gen(): string | undefined {
		return typeof this.directory.meta.gen === "string" ? this.directory.meta.gen : undefined;
	}
	get rev(): number | undefined {
		return typeof this.directory.meta.rev === "number" ? this.directory.meta.rev : undefined;
	}

	/** The `base: [offset, length]` range `packIndex` records: the objects a cold open reads, contiguous. */
	get baseRange(): BlobRange | null {
		const b = this.directory.meta.base;
		if (!Array.isArray(b) || b.length !== 2 || !b.every((x) => Number.isInteger(x) && x >= 0)) return null;
		return { offset: b[0] as number, length: b[1] as number };
	}

	/**
	 * Reads the pack's base range with ONE range read and returns the objects that lie inside it (views into the
	 * read), for a cold open (`thread.open` does this for a `PackedBlobStore`). Null when the pack records no range.
	 */
	async readBase(trace?: BlobTrace): Promise<Map<string, Uint8Array> | null> {
		const r = this.baseRange;
		if (!r || r.length === 0) return null;
		const b = await this.inner.get(this.key, r, trace ? { trace } : undefined);
		if (!b) throw new ThreadError(`pack ${this.key} disappeared`, 404);
		const out = new Map<string, Uint8Array>();
		for (const [k, e] of this.directory.entries) if (e.offset >= r.offset && e.offset + e.length <= r.offset + r.length) out.set(k, b.subarray(e.offset - r.offset, e.offset - r.offset + e.length));
		return out;
	}

	private locate(key: string, range: BlobRange | undefined): BlobRange | null {
		const e = this.directory.entries.get(key);
		if (!e) return null;
		if (!range) return { offset: e.offset, length: e.length };
		if (range.offset < 0 || range.length < 0 || range.offset + range.length > e.length) throw new ThreadError(`${key} [${range.offset}, +${range.length}) is outside the object (${e.length} bytes)`);
		return { offset: e.offset + range.offset, length: range.length };
	}

	async get(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		const r = this.locate(key, range);
		if (!r) return null;
		if (r.length === 0) return new Uint8Array(0);
		const b = await this.inner.get(this.key, r, opts?.trace ? { trace: opts.trace } : undefined);
		if (!b) throw new ThreadError(`pack ${this.key} disappeared`, 404);
		return b;
	}

	async getRanges(key: string, ranges: BlobRange[], opts?: GetOptions): Promise<Uint8Array[] | null> {
		if (!this.inner.getRanges) return null;
		const e = this.directory.entries.get(key);
		if (!e) return null;
		const rs = ranges.map((r) => this.locate(key, r)!);
		return this.inner.getRanges(this.key, rs, opts?.trace ? { trace: opts.trace } : undefined);
	}

	async head(key: string): Promise<BlobHead | null> {
		const e = this.directory.entries.get(key);
		return e ? { size: e.length, etag: `${this.pack.etag}@${e.offset}` } : null;
	}

	async list(prefix: string): Promise<string[]> {
		return [...this.directory.entries.keys()].filter((k) => k.startsWith(prefix)).sort();
	}

	// `put` is deliberately absent: writers check `store.put` and report the store as read-only.
}

// ---------- writing ----------

/** Where a `PackWriter` streams to: a file, a growing buffer, an upload. */
export interface PackSink {
	write(bytes: Uint8Array): Promise<void> | void;
}

/** Collects a pack in memory. */
export class MemorySink implements PackSink {
	readonly chunks: Uint8Array[] = [];
	write(bytes: Uint8Array): void {
		this.chunks.push(bytes);
	}
	bytes(): Uint8Array {
		return concatBytes(this.chunks);
	}
}

/**
 * Streams a pack: `add` each object (any order, keys unique), then `finish`. Entries are 8-byte aligned; the
 * directory lists them in the order added and carries `meta` (`packIndex` records `{gen, rev}`).
 */
export class PackWriter {
	private readonly entries: [string, number, number][] = [];
	private readonly seen = new Set<string>();
	private pos = 0;
	private finished = false;

	constructor(
		private readonly sink: PackSink,
		private readonly meta: Record<string, unknown> = {},
	) {}

	/** Bytes written so far. */
	get written(): number {
		return this.pos;
	}

	/** Sets a metadata field of the directory (any time before `finish`; `packIndex` records the base range). */
	setMeta(key: string, value: unknown): void {
		if (key === "entries") throw new ThreadError("pack: meta cannot be named entries", 400);
		this.meta[key] = value;
	}

	private async emit(b: Uint8Array): Promise<void> {
		await this.sink.write(b);
		this.pos += b.byteLength;
	}

	async add(key: string, bytes: Uint8Array): Promise<void> {
		if (this.finished) throw new ThreadError("pack writer is finished", 400);
		if (!key) throw new ThreadError("pack: empty key", 400);
		if (this.seen.has(key)) throw new ThreadError(`pack: ${key} added twice`, 400);
		if (this.pos === 0) await this.emit(packHeader());
		const pad = (ALIGN - (this.pos % ALIGN)) % ALIGN;
		if (pad) await this.emit(new Uint8Array(pad));
		this.seen.add(key);
		this.entries.push([key, this.pos, bytes.byteLength]);
		await this.emit(bytes);
	}

	/** Writes the directory and footer; returns the pack size and entry count. */
	async finish(): Promise<{ bytes: number; count: number }> {
		if (this.finished) throw new ThreadError("pack writer is finished", 400);
		this.finished = true;
		if (this.pos === 0) await this.emit(packHeader());
		const dir = new TextEncoder().encode(JSON.stringify({ entries: this.entries, ...this.meta }));
		const dirOff = this.pos;
		await this.emit(dir);
		await this.emit(packFooter(dirOff, dir.byteLength, this.entries.length));
		return { bytes: this.pos, count: this.entries.length };
	}
}

export interface PackIndexOptions {
	/** The generation / revision to snapshot (default: what `idx/current.json` names, see `resolveTarget`). */
	gen?: string;
	rev?: number;
	/** Objects read concurrently from the source store (default 8). */
	concurrency?: number;
}

export interface PackIndexResult {
	gen: string;
	rev: number;
	/** Objects in the pack, including the synthesized `idx/current.json`. */
	count: number;
	bytes: number;
	keys: string[];
}

/**
 * Writes a snapshot of one revision to `sink`: `idx/current.json` pointing at it, then the objects a cold open reads
 * (`baseKeys` with the smallest routing table), contiguous, then every other object the revision reads
 * (`snapshotKeys`) in key order. The directory records `{gen, rev, base: [offset, length]}`: the range from
 * `idx/current.json` to the end of the base objects, so a reader can load a cold base with one range read
 * (`PackedBlobStore.readBase`). Two SDKs packing the same revision produce the same bytes.
 */
export async function packIndex(store: BlobStore, sink: PackSink, opts: PackIndexOptions = {}): Promise<PackIndexResult> {
	const t = await resolveTarget(store, opts.gen, opts.rev);
	const rev = await loadRevState(store, t.gen, t.rev);
	const all = snapshotKeys(rev);
	const inSnapshot = new Set(all);
	const base = baseKeys(rev, "base", "auto").filter((k) => inSnapshot.has(k));
	const baseSet = new Set(base);
	const keys = [...base, ...all.filter((k) => !baseSet.has(k))];
	const w = new PackWriter(sink, { gen: t.gen, rev: t.rev });
	await w.add("idx/current.json", new TextEncoder().encode(JSON.stringify({ gen: t.gen, rev: t.rev })));
	const trace = newTrace();
	// Read ahead `concurrency` objects, write them in order.
	const window = Math.max(1, opts.concurrency ?? 8);
	const pending: Promise<Uint8Array>[] = [];
	const fetch = (k: string) =>
		store.get(k, undefined, { trace }).then((b) => {
			if (!b) throw new ThreadError(`snapshot of ${t.gen} rev ${t.rev} needs ${k}, which is missing`, 404);
			return b;
		});
	for (let i = 0; i < keys.length; i++) {
		while (pending.length < window && i + pending.length < keys.length) pending.push(fetch(keys[i + pending.length]));
		await w.add(keys[i], await pending.shift()!);
		if (i === base.length - 1) w.setMeta("base", [PACK_HEADER_BYTES, w.written - PACK_HEADER_BYTES]);
	}
	const { bytes, count } = await w.finish();
	return { gen: t.gen, rev: t.rev, count, bytes, keys: ["idx/current.json", ...keys] };
}

/** `packIndex` into memory. */
export async function packToBytes(store: BlobStore, opts: PackIndexOptions = {}): Promise<{ bytes: Uint8Array; result: PackIndexResult }> {
	const sink = new MemorySink();
	const result = await packIndex(store, sink, opts);
	return { bytes: sink.bytes(), result };
}

export interface CopyOptions {
	/** Objects copied concurrently (default 8). */
	concurrency?: number;
	/** Skip keys that already exist in `dst` with the same size (default false: overwrite). */
	skipExisting?: boolean;
	putOptions?: PutOptions;
}

/**
 * Copies objects from one store to another: a pack into a directory (to write on top of it), a bucket prefix to a
 * laptop, a laptop index to a bucket. `keys` defaults to everything `src` lists; pass `snapshotKeys(rev)` plus
 * `idx/current.json` to copy one consistent revision out of a live index.
 */
export async function copyObjects(src: BlobStore, dst: BlobStore, keys?: readonly string[], opts: CopyOptions = {}): Promise<{ count: number; bytes: number; skipped: number }> {
	if (!dst.put) throw new ThreadError("the destination store is read-only (no put)", 400);
	const put = dst.put.bind(dst);
	if (!keys) {
		if (!src.list) throw new ThreadError("the source store cannot list objects: pass the keys to copy", 400);
		keys = await src.list("");
	}
	let count = 0;
	let bytes = 0;
	let skipped = 0;
	const todo = [...keys];
	const worker = async () => {
		for (let k = todo.shift(); k !== undefined; k = todo.shift()) {
			if (opts.skipExisting) {
				const [sh, dh] = await Promise.all([src.head(k), dst.head(k)]);
				if (sh && dh && sh.size === dh.size) {
					skipped++;
					continue;
				}
			}
			const b = await src.get(k);
			if (!b) throw new ThreadError(`missing object ${k}`, 404);
			if ((await put(k, b, opts.putOptions)) === null) throw new ThreadError(`put of ${k} was refused by the destination`, 409);
			count++;
			bytes += b.byteLength;
		}
	};
	await Promise.all(Array.from({ length: Math.max(1, Math.min(opts.concurrency ?? 8, todo.length)) }, worker));
	return { count, bytes, skipped };
}

// ---------- base packs ----------

export interface BasePackResult {
	key: string;
	bytes: Uint8Array;
	/** The objects in it, in order. */
	keys: string[];
	/** The routing table it holds (`kind: "base"`). */
	routing: string | null;
}

/**
 * The base pack of a revision (`idx/{gen}/base.{rev}.threadpack`): every object a cold open reads (`baseKeys`), so a
 * reader that knows (gen, rev) from `idx/current.json` loads the revision and its routing table with ONE GET. A cold
 * base pack (`kind: "cold"`, `idx/{gen}/base.cold.{rev}.threadpack`) leaves the routing table out, for
 * `coldRouting: "tree"`. The directory records `{gen, rev, kind: "base" | "base-cold", routing}`. Publish writes them
 * (the Worker's publish / publish-rev); `bench/cold/make_base_pack.mjs` writes them offline.
 */
export async function buildBasePack(store: BlobStore, rev: RevState, kind: "base" | "cold" = "base", routing: RoutingOption = "auto"): Promise<BasePackResult> {
	const ks = baseKeys(rev, kind, routing);
	const elem = kind === "base" && hasDense(rev.manifest) ? chooseRouting(rev.manifest, rev.files, routing).elem : null;
	const sink = new MemorySink();
	const w = new PackWriter(sink, { gen: rev.gen, rev: rev.rev, kind: kind === "cold" ? "base-cold" : "base", routing: elem });
	const got = await Promise.all(
		ks.map(async (k) => {
			const b = await store.get(k);
			if (!b) throw new ThreadError(`base pack of ${rev.gen} rev ${rev.rev} needs ${k}, which is missing`, 404);
			return b;
		}),
	);
	for (let i = 0; i < ks.length; i++) await w.add(ks[i], got[i]);
	await w.finish();
	return { key: kind === "cold" ? indexKeys.baseColdPack(rev.gen, rev.rev) : indexKeys.basePack(rev.gen, rev.rev), bytes: sink.bytes(), keys: ks, routing: elem };
}

/**
 * A read-through view of a store with some objects already in memory (a base pack's, or reads started early):
 * `get` / `getRanges` / `head` of such a key are served from memory (byte-identical: the same object), everything
 * else goes to the inner store. A read started early (`prefetch`) keeps its own trace, which the first `get` of the
 * key adds to the caller's, so per-step accounting (`openStats`) still sees the read where its bytes are used. Used by
 * `thread.open` for the reads of the open only; read-only.
 */
export class PrefetchStore implements BlobStore {
	private readonly entries = new Map<string, { bytes: Promise<Uint8Array | null>; trace: BlobTrace | null }>();
	constructor(readonly inner: BlobStore) {}

	/** Serves `key` from `bytes` (accounted elsewhere, e.g. as part of a pack). */
	set(key: string, bytes: Uint8Array | Promise<Uint8Array | null>): void {
		const p = Promise.resolve(bytes);
		p.catch(() => {});
		this.entries.set(key, { bytes: p, trace: null });
	}

	/** Starts reading `key` now (once); later `get`s of it wait for this read. */
	prefetch(key: string): void {
		if (this.entries.has(key)) return;
		const trace = newTrace();
		const p = this.inner.get(key, undefined, { trace });
		p.catch(() => {});
		this.entries.set(key, { bytes: p, trace });
	}

	/** Serves every object of a pack held in memory. */
	addPack(pack: Uint8Array): Record<string, unknown> {
		const { entries, meta } = packEntries(pack);
		for (const [k, b] of entries) this.set(k, b);
		return meta;
	}

	has(key: string): boolean {
		return this.entries.has(key);
	}

	/** The bytes of a prefetched key without crediting its read to anyone (null when absent or not prefetched). */
	async peek(key: string): Promise<Uint8Array | null> {
		const e = this.entries.get(key);
		return e ? e.bytes : null;
	}

	private async take(key: string, trace: BlobTrace | undefined): Promise<Uint8Array | null> {
		const e = this.entries.get(key)!;
		const b = await e.bytes;
		if (e.trace && trace) {
			for (const f of ["reads", "hits", "misses", "r206", "r200", "bytes"] as const) trace[f] += e.trace[f];
			e.trace = null; // credited once
		}
		return b;
	}

	async get(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		if (!this.entries.has(key)) return this.inner.get(key, range, opts);
		const b = await this.take(key, opts?.trace);
		if (!b) return null;
		if (!range) return b;
		if (range.offset < 0 || range.length < 0 || range.offset + range.length > b.byteLength) throw new ThreadError(`${key} [${range.offset}, +${range.length}) is outside the object (${b.byteLength} bytes)`);
		return b.subarray(range.offset, range.offset + range.length);
	}

	async getRanges(key: string, ranges: BlobRange[], opts?: GetOptions): Promise<Uint8Array[] | null> {
		if (!this.entries.has(key)) return this.inner.getRanges ? this.inner.getRanges(key, ranges, opts) : null;
		const out: Uint8Array[] = [];
		for (const r of ranges) {
			const b = await this.get(key, r, opts);
			if (!b) return null;
			out.push(b);
		}
		return out;
	}

	async head(key: string): Promise<BlobHead | null> {
		if (!this.entries.has(key)) return this.inner.head(key);
		const b = await this.entries.get(key)!.bytes;
		return b ? { size: b.byteLength, etag: "prefetched" } : null;
	}

	get list(): ((prefix: string) => Promise<string[]>) | undefined {
		const inner = this.inner;
		return inner.list ? (prefix: string) => inner.list!(prefix) : undefined;
	}
}
