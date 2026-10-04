/**
 * Wire formats shared with the thread Worker: packed hits, slab and delta-segment headers, the manifest
 * (revisions, deltas, tombstone sets), the slabs table and the R2 key layout. Everything here is plain
 * JS over `Uint8Array`s (no wasm), so headers can be parsed without copying a slab into wasm memory.
 */

export class ThreadError extends Error {
	constructor(
		message: string,
		/** HTTP-like status: 400 bad input, 404 missing object, 409 write conflict, 500 corrupt index. */
		public readonly status = 500,
	) {
		super(message);
		this.name = "ThreadError";
	}
}

// ---------- packed hits ----------

/** One hit of `scan`/`scan_pq`/`scan_delta`/`merge`: `idx` is the entry index in its slab or delta segment. */
export interface PackedHit {
	id: number;
	score: number;
	idx: number;
}

/** Parses packed hits `[count u32][id u32, score f32, idx u32]*` (little-endian). */
export function parseHits(packed: Uint8Array): PackedHit[] {
	if (packed.byteLength < 4) return [];
	const dv = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
	const n = dv.getUint32(0, true);
	if (packed.byteLength < 4 + n * 12) throw new ThreadError(`packed hits: ${n} hits need ${4 + n * 12} bytes, have ${packed.byteLength}`);
	const out: PackedHit[] = new Array(n);
	for (let i = 0; i < n; i++) {
		const o = 4 + i * 12;
		out[i] = { id: dv.getUint32(o, true), score: dv.getFloat32(o + 4, true), idx: dv.getUint32(o + 8, true) };
	}
	return out;
}

/** Number of hits in a packed hit buffer. */
export const packedCount = (b: Uint8Array): number => (b.byteLength < 4 ? 0 : new DataView(b.buffer, b.byteOffset, 4).getUint32(0, true));

export function concatBytes(parts: Uint8Array[]): Uint8Array {
	const out = new Uint8Array(parts.reduce((a, b) => a + b.byteLength, 0));
	let o = 0;
	for (const p of parts) {
		out.set(p, o);
		o += p.byteLength;
	}
	return out;
}

// ---------- slab header (64 bytes, "VIBS") ----------

export const SLAB_MAGIC = 0x53424956; // "VIBS" LE
export const SLAB_ENTRY_BYTES = 16;

export interface SlabHeader {
	/** 1 = int8 only (read whole, no phase 2); 2 = 4-bit scan + int8 rerank; 3 = PQ scan + int8 rerank. */
	version: number;
	dim: number;
	count: number;
	cell: number;
	centroidOff: number;
	/** Entry table: `count` x 16 B `[id u32, payloadOff u32, payloadLen u32, attrs u32]`. */
	entriesOff: number;
	/** Bytes to range-read for a scan (`[0, scanLen)`); for v1 the payload offset (the slab is read whole anyway). */
	scanLen: number;
	/** v2/v3: rerank record `i` is `[rerankOff + i * recLen, +recLen)`; 0 for v1. */
	rerankOff: number;
	/** `4 + dim` for v2/v3, 0 for v1. */
	recLen: number;
	payloadOff: number;
	payloadLen: number;
	/** v3: PQ subspaces of the scan tier (0 otherwise). */
	pqM: number;
}

export function parseSlabHeader(b: Uint8Array): SlabHeader {
	if (b.byteLength < 64) throw new ThreadError(`slab header needs 64 bytes, have ${b.byteLength}`);
	const dv = new DataView(b.buffer, b.byteOffset, 64);
	if (dv.getUint32(0, true) !== SLAB_MAGIC) throw new ThreadError("bad slab magic");
	const version = dv.getUint16(4, true);
	const dim = dv.getUint32(8, true);
	const count = dv.getUint32(12, true);
	const cell = dv.getUint32(16, true);
	const centroidOff = dv.getUint32(20, true);
	const entriesOff = dv.getUint32(24, true);
	if (version === 2 || version === 3) {
		// v2 and v3 share every field read here (v3 stores pq_m at 32, where v2 has scales4_off).
		return { version, dim, count, cell, centroidOff, entriesOff, rerankOff: dv.getUint32(36, true), payloadOff: dv.getUint32(40, true), payloadLen: dv.getUint32(44, true), scanLen: dv.getUint32(48, true), recLen: 4 + dim, pqM: version === 3 ? dv.getUint32(32, true) : 0 };
	}
	if (version === 1) {
		const payloadOff = dv.getUint32(36, true);
		return { version, dim, count, cell, centroidOff, entriesOff, rerankOff: 0, payloadOff, payloadLen: dv.getUint32(40, true), scanLen: payloadOff, recLen: 0, pqM: 0 };
	}
	throw new ThreadError(`unknown slab version ${version}`);
}

/** The cell mean stored in a slab prefix (copied out, so it outlives the prefix). */
export function slabCentroid(b: Uint8Array, h: SlabHeader): Float32Array {
	const end = h.centroidOff + h.dim * 4;
	if (b.byteLength < end) throw new ThreadError(`slab prefix too short for the centroid (${b.byteLength} < ${end})`);
	return new Float32Array(b.slice(h.centroidOff, end).buffer);
}

export interface SlabEntry {
	id: number;
	payloadOff: number;
	payloadLen: number;
	attrs: number;
}

/** Entry `i` of the slab entry table (inside the scan prefix). */
export function slabEntry(b: Uint8Array, h: SlabHeader, i: number): SlabEntry {
	const o = h.entriesOff + i * SLAB_ENTRY_BYTES;
	if (i >= h.count || b.byteLength < o + SLAB_ENTRY_BYTES) throw new ThreadError(`slab entry ${i} out of range`);
	const dv = new DataView(b.buffer, b.byteOffset + o, SLAB_ENTRY_BYTES);
	return { id: dv.getUint32(0, true), payloadOff: dv.getUint32(4, true), payloadLen: dv.getUint32(8, true), attrs: dv.getUint32(12, true) };
}

// ---------- delta segment header (64 bytes, "VIBD") ----------

export const DELTA_MAGIC = 0x44424956; // "VIBD" LE
export const DELTA_ENTRY_BYTES = 24;

export interface DeltaHeader {
	version: number;
	dim: number;
	count: number;
	pqM: number;
	seq: number;
	/** Entry table: `count` x 24 B `[id, cell, attrs, payloadOff, payloadLen, reserved]` (u32 each). */
	entriesOff: number;
	rerankOff: number;
	payloadOff: number;
	payloadLen: number;
}

export function parseDeltaHeader(b: Uint8Array): DeltaHeader {
	if (b.byteLength < 64) throw new ThreadError(`delta header needs 64 bytes, have ${b.byteLength}`);
	const dv = new DataView(b.buffer, b.byteOffset, 64);
	if (dv.getUint32(0, true) !== DELTA_MAGIC) throw new ThreadError("bad delta magic");
	const version = dv.getUint16(4, true);
	if (version !== 1) throw new ThreadError(`unknown delta version ${version}`);
	return { version, dim: dv.getUint32(8, true), count: dv.getUint32(12, true), pqM: dv.getUint32(16, true), seq: dv.getUint32(20, true), entriesOff: dv.getUint32(24, true), rerankOff: dv.getUint32(28, true), payloadOff: dv.getUint32(44, true), payloadLen: dv.getUint32(48, true) };
}

export interface DeltaEntry extends SlabEntry {
	cell: number;
}

export function deltaEntry(b: Uint8Array, h: DeltaHeader, i: number): DeltaEntry {
	const o = h.entriesOff + i * DELTA_ENTRY_BYTES;
	if (i >= h.count || b.byteLength < o + DELTA_ENTRY_BYTES) throw new ThreadError(`delta entry ${i} out of range`);
	const dv = new DataView(b.buffer, b.byteOffset + o, DELTA_ENTRY_BYTES);
	return { id: dv.getUint32(0, true), cell: dv.getUint32(4, true), attrs: dv.getUint32(8, true), payloadOff: dv.getUint32(12, true), payloadLen: dv.getUint32(16, true) };
}

// ---------- manifest, slabs table, current pointer ----------

/** Query defaults recorded in a manifest; an absent query parameter takes its value from here. */
export interface QueryDefaults {
	nprobe: number;
	rerank: number;
	pc: number;
}

/** Defaults of manifests without `defaults` (generations before g3). */
export const LEGACY_DEFAULTS: QueryDefaults = { nprobe: 1, rerank: 32, pc: 0 };

/**
 * Row of a slabs table (array indexed by cell): `[scanLen, rerankOff, count, bytes, slabRev?]`. `slabRev`
 * (absent = 0) names the slab object: 0 = `slab/{cell:05}`, r > 0 = `slab/{cell:05}.{r}` (written by a
 * compaction that read manifest rev r - 1).
 */
export type SlabRow = [number, number, number, number] | [number, number, number, number, number];
export const slabRevOf = (row: SlabRow | undefined): number => (row && row.length > 4 ? (row[4] as number) : 0);

/** A delta segment listed in a manifest. */
export interface DeltaRef {
	seq: number;
	key: string;
	count: number;
	bytes: number;
	/** The manifest rev that first listed this segment: tombstone sets with rev > addedRev hide its entries. */
	addedRev: number;
	cells?: number;
}

/**
 * A lexical/docs segment listed in a manifest (`idx/{gen}/seg/{seq}.thread`, or the base `seg/base.{rev}.thread` a
 * compaction wrote): a `THREADIX` container with `meta`, optionally `terms` + `postings` (BM25 over its documents)
 * and `docs` (id -> metadata, optionally the text). Filtered like a delta: sets with rev > addedRev hide its entries.
 */
export interface SegmentRef {
	/** The delta seq of the upsert that wrote it (0 for a compaction's base segment). */
	seq: number;
	key: string;
	/** The manifest rev that first listed the segment. */
	addedRev: number;
	/** Documents in the segment. */
	count: number;
	bytes: number;
	/** Which sections it holds. */
	lex: boolean;
	docs: boolean;
	text: boolean;
	/** Written by a compaction (holds every live document of the segments it replaced). */
	base?: boolean;
}

/**
 * The lexical/docs configuration of a generation (`manifest.lex`): how upserts build segments and how queries
 * tokenize. Set with `index.configureLexical(...)` (a manifest revision); every segment of a generation shares it.
 */
export interface LexConfig {
	/** The `tokenizer` object (bm25s-compatible keys: lowercase, token_pattern, stopwords, stemmer, segmenter, identifiers). */
	tokenizer?: Record<string, unknown>;
	/** BM25 parameters (default 1.2 / 0.75). */
	k1?: number;
	b?: number;
	/** How postings carry document lengths: `exact` (default) or `byte` (one SmallFloat byte, smaller, scores within a few %). */
	lengths?: "exact" | "byte";
	/** Terms per `terms` bucket (default 256): fewer = smaller reads per query term, more = smaller resident directory. */
	bucketTerms?: number;
	/** Write `docs` entries (default true) and store the texts in them (default false). */
	docs?: boolean;
	text?: boolean;
	/** Documents per `docs` bucket (default 128). */
	bucketDocs?: number;
	/** Query defaults for `search`: RRF constant (60), per-list depth (max(topK, 50)) and list weights (1 / 1). */
	defaults?: { rrfK?: number; recallK?: number; weights?: { dense?: number; lexical?: number } };
}

export interface ResolvedLexConfig {
	tokenizer: Record<string, unknown> | null;
	k1: number;
	b: number;
	lengths: "exact" | "byte";
	bucketTerms: number;
	docs: boolean;
	text: boolean;
	bucketDocs: number;
	rrfK: number;
	recallK: number | null;
	weights: { dense: number; lexical: number };
}

/** `manifest.lex` with its defaults filled in (`null` when the generation has no lexical configuration). */
export function resolveLexConfig(m: Manifest): ResolvedLexConfig | null {
	const c = m.lex;
	if (!c) return null;
	return {
		tokenizer: c.tokenizer ?? null,
		k1: c.k1 ?? 1.2,
		b: c.b ?? 0.75,
		lengths: c.lengths ?? "exact",
		bucketTerms: c.bucketTerms ?? 256,
		docs: c.docs ?? true,
		text: c.text ?? false,
		bucketDocs: c.bucketDocs ?? 128,
		rrfK: c.defaults?.rrfK ?? 60,
		recallK: c.defaults?.recallK ?? null,
		weights: { dense: c.defaults?.weights?.dense ?? 1, lexical: c.defaults?.weights?.lexical ?? 1 },
	};
}

/** A tombstone set: ids deleted (or replaced by an upsert) at manifest rev `rev`. */
export interface TombRef {
	rev: number;
	key: string;
	count: number;
}

/** Objects a revision reads. Manifests without `files` (rev 0) use the generation's fixed names. */
export interface IndexFiles {
	coarse: string;
	fine: string;
	pq: string | null;
	slabs: string | null;
	/**
	 * The newest base pack of the generation (`idx/{gen}/base.{rev}.threadpack`, written by publish / publish-rev
	 * for that rev; later revisions inherit the name). For discoverability: a reader fetches the base pack of the
	 * revision it opens by its conventional key (`keys.basePack`), and never needs this field.
	 */
	base?: string;
	/** The newest cold base pack (`idx/{gen}/base.cold.{rev}.threadpack`), likewise. */
	baseCold?: string;
}

/** Element type of a routing table: the f32 means, IEEE half floats, or int8 with one f32 scale per row. */
export type RoutingElem = "f32" | "f16" | "i8";

/** The routing table a reader loads: an element type, or `"auto"` for the smallest one the revision has. */
export type RoutingOption = RoutingElem | "auto";

/** Bytes of one routing table row. */
export const routingRowBytes = (dim: number, elem: RoutingElem): number => (elem === "f32" ? 4 * dim : elem === "f16" ? 2 * dim : 4 + dim);

/** Bytes of a whole routing table of `k` rows (i8: the f32 centre the rows are residuals from, then the rows). */
export const routingTableBytes = (dim: number, k: number, elem: RoutingElem): number => (elem === "i8" ? 4 * dim : 0) + k * routingRowBytes(dim, elem);

export interface RoutingTableRef {
	key: string;
	bytes: number;
}

/**
 * The fine means in centroid-tree group order (`fine.grouped.{rev}.{elem}`, "VIBG"): group g's rows are
 * `[dataOff + offsets[g] * rowBytes, +count * rowBytes)` with the tree's CSR `offsets`, so a reader holding the tree
 * range-reads just the groups a query needs.
 */
export interface GroupedRoutingRef {
	key: string;
	/** The centroid tree whose group order the rows follow (a reader checks it is `manifest.tree.key`). */
	tree: string;
	elem: RoutingElem;
	groups: number;
	rows: number;
	dataOff: number;
	rowBytes: number;
	bytes: number;
}

/**
 * Compact copies of the fine means (`manifest.routing`). They apply only while `files.fine` is the table they were
 * made from (`fine`): a compaction that splits cells writes new means, and a reader then falls back to them.
 */
export interface RoutingTables {
	fine: string;
	f16?: RoutingTableRef;
	i8?: RoutingTableRef;
	grouped?: GroupedRoutingRef;
}

/** Centroid tree over the fine means (`manifest.tree`, written by the Worker's publish / publish-rev). */
export interface TreeRef {
	key: string;
	groups: number;
	beam: number;
	bytes?: number;
	[extra: string]: unknown;
}

export interface Manifest {
	gen: string;
	k1: number;
	k2: number;
	dim: number;
	m: number;
	probeCoarse?: number;
	vectors: number;
	/** 3 = PQ scan + int8 rerank tiers; 2 = 4-bit scan + int8 rerank; absent/1 = int8-only slabs. */
	format?: number;
	profile?: string;
	tiers?: string;
	pq?: { m: number };
	nNeigh?: number;
	defaults?: QueryDefaults;
	createdAt?: string;
	notes?: unknown;
	/** Revision; absent = 0 (`idx/{gen}/manifest.json`). */
	rev?: number;
	parentRev?: number;
	updatedAt?: string;
	change?: Record<string, unknown>;
	files?: IndexFiles;
	deltas?: DeltaRef[];
	tombstones?: TombRef[];
	/** Next delta sequence number. */
	nextSeq?: number;
	/** Lexical / docs configuration (absent: the generation has no lexical index and no docs store). */
	lex?: LexConfig;
	/** Lexical / docs segments, oldest first (a compaction's base first). */
	segments?: SegmentRef[];
	/** Compact routing tables (fine.f16, fine.i8, tree-grouped means). */
	routing?: RoutingTables;
	/** Centroid tree over the fine means. */
	tree?: TreeRef;
}

/** `idx/current.json`: the generation and revision queries use by default. */
export interface CurrentPointer {
	gen: string;
	rev: number;
}

export const pad5 = (n: number): string => n.toString().padStart(5, "0");

/** Object keys of the index layout, relative to the store prefix. */
export const keys = {
	current: (): string => "idx/current.json",
	manifest: (gen: string, rev: number): string => (rev === 0 ? `idx/${gen}/manifest.json` : `idx/${gen}/manifest.${rev}.json`),
	slabsTable: (gen: string, rev: number): string => (rev === 0 ? `idx/${gen}/slabs.json` : `idx/${gen}/slabs.${rev}.json`),
	slab: (gen: string, cell: number, slabRev = 0): string => (slabRev === 0 ? `idx/${gen}/slab/${pad5(cell)}` : `idx/${gen}/slab/${pad5(cell)}.${slabRev}`),
	tombstones: (gen: string, rev: number): string => `idx/${gen}/tombstones.${rev}.u32`,
	delta: (gen: string, seq: number): string => `idx/${gen}/delta/${seq}.vibd`,
	segment: (gen: string, seq: number): string => `idx/${gen}/seg/${seq}.thread`,
	segmentBase: (gen: string, rev: number): string => `idx/${gen}/seg/base.${rev}.thread`,
	/** The base pack of a revision: every object a cold open reads, in one file (`docs/query-path.md`). */
	basePack: (gen: string, rev: number): string => `idx/${gen}/base.${rev}.threadpack`,
	/** The cold base pack: the base pack without the full routing table (for `coldRouting: "tree"`). */
	baseColdPack: (gen: string, rev: number): string => `idx/${gen}/base.cold.${rev}.threadpack`,
	/** Compact routing tables derived from the f32 means at `fine` (`idx/g4/fine.105.f32` -> `idx/g4/fine.105.f16`). */
	routingTable: (fine: string, elem: RoutingElem): string => (elem === "f32" ? fine : fine.replace(/\.f32$/, "") + `.${elem}`),
	groupedTable: (gen: string, rev: number, elem: RoutingElem): string => `idx/${gen}/fine.grouped.${rev}.${elem}`,
};

/** The usable compact routing tables of a revision: those made from the means `files.fine` names. */
export function routingTablesOf(m: Manifest, files: IndexFiles = manifestFiles(m)): RoutingTables | null {
	const r = m.routing;
	if (!r || r.fine !== files.fine) return null;
	return r;
}

/** The routing table to load for `opt`: falls back to f32 when the revision has no such table. */
export function chooseRouting(m: Manifest, files: IndexFiles, opt: RoutingOption = "f32"): { elem: RoutingElem; key: string } {
	const r = routingTablesOf(m, files);
	const pick = (e: RoutingElem): { elem: RoutingElem; key: string } | null => {
		if (e === "f32") return { elem: "f32", key: files.fine };
		const t = r?.[e];
		return t ? { elem: e, key: t.key } : null;
	};
	if (opt === "auto") return pick("i8") ?? pick("f16") ?? pick("f32")!;
	return pick(opt) ?? pick("f32")!;
}

/** The tree-grouped table of a revision, if it has one that follows its current tree and means. */
export function groupedTableOf(m: Manifest, files: IndexFiles = manifestFiles(m)): GroupedRoutingRef | null {
	const g = routingTablesOf(m, files)?.grouped;
	if (!g || !m.tree || g.tree !== m.tree.key || m.k1 !== 1) return null;
	return g;
}

export function manifestFiles(m: Manifest): IndexFiles {
	if (m.files) return m.files;
	const g = m.gen;
	return { coarse: `idx/${g}/coarse.f32`, fine: `idx/${g}/fine.f32`, pq: (m.format ?? 1) === 3 ? `idx/${g}/pq.bin` : null, slabs: `idx/${g}/slabs.json` };
}

const utf8 = new TextDecoder();
export const decodeJson = <T>(b: Uint8Array): T => JSON.parse(utf8.decode(b)) as T;

/** Parses `idx/current.json`; a pointer without `rev` (older publishes) means rev 0. */
export function parseCurrent(b: Uint8Array): CurrentPointer {
	const j = decodeJson<{ gen?: unknown; rev?: unknown }>(b);
	if (typeof j.gen !== "string" || !j.gen) throw new ThreadError("current.json has no gen");
	const rev = j.rev === undefined ? 0 : Number(j.rev);
	if (!Number.isInteger(rev) || rev < 0) throw new ThreadError(`current.json has a bad rev ${String(j.rev)}`);
	return { gen: j.gen, rev };
}

/** Whether a generation has a dense index (a lexical-only generation is published with `k2: 0, dim: 0`). */
export const hasDense = (m: Manifest): boolean => m.k2 > 0 && m.dim > 0;

/** Parses a manifest and checks it is for `gen`; fills `rev` (absent in rev-0 manifests). */
export function parseManifest(b: Uint8Array, gen: string, rev: number): Manifest {
	const m = decodeJson<Manifest>(b);
	if (m.gen !== gen) throw new ThreadError(`manifest for ${gen} rev ${rev} is for generation ${m.gen}`);
	for (const f of ["k1", "k2", "dim", "vectors"] as const) if (!Number.isInteger(m[f])) throw new ThreadError(`manifest ${gen} rev ${rev}: missing ${f}`);
	if ((m.rev ?? 0) !== rev) throw new ThreadError(`manifest ${gen} rev ${rev} says rev ${m.rev}`);
	return { ...m, rev };
}

export function parseSlabsTable(b: Uint8Array, cells: number, key: string): SlabRow[] {
	const t = decodeJson<unknown>(b);
	if (!Array.isArray(t) || t.length !== cells) throw new ThreadError(`${key} has ${Array.isArray(t) ? t.length : "no"} rows, expected ${cells}`);
	return t as SlabRow[];
}

// ---------- tombstones ----------

/** Sorted unique union of id sets as a tombstone buffer (`u32` LE), the format `filter_hits` takes. */
export function idUnion(sets: ArrayLike<number>[]): Uint8Array {
	let n = 0;
	for (const s of sets) n += s.length;
	const all = new Uint32Array(n);
	let o = 0;
	for (const s of sets) {
		all.set(s, o);
		o += s.length;
	}
	all.sort();
	let w = 0;
	for (let i = 0; i < all.length; i++) if (i === 0 || all[i] !== all[i - 1]) all[w++] = all[i];
	return new Uint8Array(all.buffer, 0, w * 4);
}

/** The tombstone sets of a revision, parsed once; `unions` caches the per-slabRev filter buffers. */
export interface TombState {
	sets: { rev: number; ids: Uint32Array }[];
	/** Ids over all sets (with repeats). */
	total: number;
	/** id -> highest rev of a set containing it (the delta rule). */
	maxRev: Map<number, number>;
	unions: Map<number, Uint8Array>;
	/** Union of the sets with rev > a (the filter for delta segments added at rev a), by a. */
	deltaUnions: Map<number, Uint8Array>;
}

export function buildTombState(sets: { rev: number; ids: Uint32Array }[]): TombState {
	const maxRev = new Map<number, number>();
	let total = 0;
	for (const s of sets) {
		total += s.ids.length;
		for (const id of s.ids) if ((maxRev.get(id) ?? -1) < s.rev) maxRev.set(id, s.rev);
	}
	return { sets, total, maxRev, unions: new Map(), deltaUnions: new Map() };
}

/**
 * Filter buffer for hits of a slab with revision `slabRev`: the union of the sets with rev >= slabRev. A slab
 * written by a compaction that read manifest rev R has slabRev R + 1 and already excludes the sets with
 * rev <= R; rev-0 slabs are hidden by every set.
 */
export function tombFor(t: TombState, slabRev: number): Uint8Array {
	let u = t.unions.get(slabRev);
	if (!u) {
		u = idUnion(t.sets.filter((s) => s.rev >= slabRev).map((s) => s.ids));
		t.unions.set(slabRev, u);
	}
	return u;
}

/**
 * Filter buffer for the entries of a delta segment added at manifest rev `addedRev`: the union of the sets with
 * rev > addedRev (the upsert's own set has rev = addedRev and hides older copies, not the new one).
 */
export function tombForDelta(t: TombState, addedRev: number): Uint8Array {
	let u = t.deltaUnions.get(addedRev);
	if (!u) {
		u = idUnion(t.sets.filter((s) => s.rev > addedRev).map((s) => s.ids));
		t.deltaUnions.set(addedRev, u);
	}
	return u;
}

/** Filter buffer for the entries of a lexical/docs segment added at rev `addedRev`: the delta rule. */
export const tombForSegment = tombForDelta;

/** Whether `id` is hidden in a segment added at `addedRev` (a set with rev > addedRev holds it). */
export function tombHides(t: TombState | null, id: number, addedRev: number): boolean {
	if (!t) return false;
	const r = t.maxRev.get(id);
	return r !== undefined && r > addedRev;
}

export const NO_TOMBSTONES: Uint8Array = new Uint8Array(0);
