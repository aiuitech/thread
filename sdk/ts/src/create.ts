/**
 * `createGeneration`: a dense generation from nothing, for an embedding model or a vector size. Kept apart from
 * write.ts so the Worker (which has its own wasm bindings) can import it without the SDK's wasm glue.
 */
import type { BlobStore } from "./store.js";
import { defineAttrs, validateSchema, type AttrFieldSpec, type AttrSchema } from "./attrs.js";
import { validateEmbeddingConfig, type EmbeddingConfig } from "./embed.js";
import { COMPACTION_PROFILES, currentBody, keys, parseCurrent, ThreadError, type CompactionPolicy, type CurrentPointer, type LexConfig, type Manifest, type QueryDefaults, type SlabRow } from "./wire.js";

function writable(store: BlobStore): Required<Pick<BlobStore, "put">> & BlobStore {
	if (!store.put) throw new ThreadError("this store is read-only (no put)", 400);
	return store as Required<Pick<BlobStore, "put">> & BlobStore;
}

async function putIfAbsent(store: BlobStore, key: string, data: Uint8Array | string, contentType?: string): Promise<boolean> {
	return (await writable(store).put(key, data, { ifAbsent: true, contentType })) !== null;
}

async function readCurrent(store: BlobStore): Promise<{ cur: CurrentPointer; etag: string } | null> {
	const h = await store.head(keys.current());
	if (!h) return null;
	const b = await store.get(keys.current());
	if (!b) return null;
	return { cur: parseCurrent(b), etag: h.etag };
}

export interface CreateGenerationOptions {
	store: BlobStore;
	gen: string;
	/**
	 * The embedding model the generation is for (`manifest.embedding`): its `dim` sizes the index, and texts
	 * upserted or searched without vectors are embedded through it. Without one, pass `dim` and bring vectors.
	 */
	embedding?: EmbeddingConfig;
	/** Vector size; required without `embedding` (and must equal `embedding.dim` with one). */
	dim?: number;
	/**
	 * Lexical / docs configuration (default `{}`: BM25 over texts, metas stored, texts not); needed for keys. `null`
	 * for a dense-only generation (ids only).
	 */
	lex?: LexConfig | null;
	/** Attribute schema (`defineAttrs` or a field spec map), optional. */
	attrs?: AttrSchema | Record<string, AttrFieldSpec>;
	/**
	 * The cells' means (`k x dim` floats, e.g. from k-means over a sample; `bulkBuild` passes them): rev 0 then has
	 * `k` empty cells (`k2 = k`) instead of one, and upserts route to the nearest. Default: one cell with an arbitrary
	 * unit mean, which compaction splits as the data grows.
	 */
	means?: Float32Array;
	/** Compaction profile: `edge` (split cells at 3000 entries; default) or `local` (1000). */
	profile?: "edge" | "local";
	/** Query defaults (default nprobe 32, rerank 150, pc 0; nprobe is clamped to the cell count). */
	defaults?: Partial<QueryDefaults>;
	/** Automatic compaction policy (default: the profile's); `null` for none (no automatic compaction, no splits). */
	compaction?: Partial<CompactionPolicy> | null;
	/** Point `idx/current.json` at the new generation (default true). */
	setCurrent?: boolean;
	notes?: unknown;
}

export interface CreateGenerationResult {
	gen: string;
	rev: number;
	manifest: Manifest;
	/** The objects written (manifest, means, the empty slab, the slabs table). */
	keys: string[];
	currentUpdated: boolean;
}

/** The wasm exports `createGeneration` needs (the SDK's bindings or the Worker's own). */
export interface SlabWasm {
	SlabJob: new (dim: number, cell: number, centroid_mean: Float32Array, format?: number) => { finish(): Uint8Array; free(): void };
	slab_info: (slab: Uint8Array) => Uint32Array;
}

/** `defaults` of a generation `createGeneration` publishes. */
export const CREATE_DEFAULTS: QueryDefaults = { nprobe: 32, rerank: 150, pc: 0 };

/**
 * Publishes a dense generation from nothing, for an embedding model (`embedding`) or a vector size (`dim`): rev 0
 * with ONE cell (an empty format-2 slab, `fine.f32` and `coarse.f32` of one unit row, a slabs table) plus the lexical
 * configuration (default `{}`) and the compaction policy of `profile`. Vectors (or texts, with `embedding` and an
 * embedder) are then added with ordinary upserts: they go to delta segments, the Worker's automatic compaction
 * folds them into the cell and splits it in two (2-means) when it passes `maxCellEntries`, so the cell count grows
 * with the data. No k-means build, no PQ codebook: the scan tier is 4-bit (`sq4+sq8`). Needs the wasm `SlabJob`
 * (`thread.wasm`, or the Worker's bindings).
 */
export async function createGeneration(w: SlabWasm, opts: CreateGenerationOptions): Promise<CreateGenerationResult> {
	const { store, gen } = opts;
	writable(store);
	if (!/^[A-Za-z0-9_.-]+$/.test(gen)) throw new ThreadError(`bad generation name ${JSON.stringify(gen)}`, 400);
	const embedding = opts.embedding !== undefined ? validateEmbeddingConfig(opts.embedding, opts.dim) : undefined;
	const dim = embedding ? embedding.dim : opts.dim;
	if (dim === undefined) throw new ThreadError("createGeneration needs an embedding model (embedding) or a vector size (dim)", 400);
	if (!Number.isInteger(dim) || dim < 1 || dim > 65_536) throw new ThreadError("dim must be an integer in [1, 65536]", 400);
	const profile = opts.profile ?? "edge";
	if (profile !== "edge" && profile !== "local") throw new ThreadError(`profile must be "edge" or "local"`, 400);
	const defaults: QueryDefaults = { ...CREATE_DEFAULTS, ...(opts.defaults ?? {}) };
	for (const [name, v, min] of [["nprobe", defaults.nprobe, 1], ["rerank", defaults.rerank, 0], ["pc", defaults.pc, 0]] as const) if (!Number.isInteger(v) || v < min) throw new ThreadError(`defaults.${name} must be an integer >= ${min}`, 400);
	const compaction: CompactionPolicy | null = opts.compaction === null ? null : { ...COMPACTION_PROFILES[profile], ...(opts.compaction ?? {}) };
	const lex: LexConfig | null = opts.lex === undefined ? {} : opts.lex;
	const attrs = opts.attrs === undefined ? undefined : "fields" in opts.attrs && Array.isArray((opts.attrs as AttrSchema).fields) ? validateSchema(opts.attrs as AttrSchema) : defineAttrs(opts.attrs as Record<string, AttrFieldSpec>);
	// One cell whose mean is an arbitrary unit vector (the data decides the means once compaction splits the cell),
	// or one empty cell per given mean.
	const mean = new Float32Array(dim).fill(1 / Math.sqrt(dim));
	const means = opts.means ?? mean;
	if (!(means instanceof Float32Array) || means.length === 0 || means.length % dim !== 0) throw new ThreadError(`means must be a Float32Array of k x ${dim} floats`, 400);
	if (!means.every(Number.isFinite)) throw new ThreadError("means must be finite", 400);
	const k = means.length / dim;
	const slabs: Uint8Array[] = [];
	const rows: SlabRow[] = [];
	for (let cell = 0; cell < k; cell++) {
		const slab = new w.SlabJob(dim, cell, means.subarray(cell * dim, (cell + 1) * dim), 2).finish();
		const info = w.slab_info(slab);
		slabs.push(slab);
		rows.push([info[5], info[6], 0, slab.byteLength]);
	}
	const meanBytes = new Uint8Array(mean.buffer, mean.byteOffset, mean.byteLength);
	const fineBytes = new Uint8Array(means.buffer, means.byteOffset, means.byteLength);
	const manifest: Manifest = {
		gen,
		k1: 1,
		k2: k,
		dim,
		m: 1,
		probeCoarse: 1,
		vectors: 0,
		format: 2,
		tiers: "sq4+sq8",
		profile,
		defaults,
		createdAt: new Date().toISOString(),
		...(opts.notes !== undefined ? { notes: opts.notes } : {}),
		...(lex ? { lex } : {}),
		...(attrs ? { attrs } : {}),
		...(embedding ? { embedding } : {}),
		...(compaction ? { compaction } : {}),
		deltas: [],
		tombstones: [],
		segments: [],
		nextSeq: 1,
		nextId: 0,
	};
	const written = [keys.manifest(gen, 0), `idx/${gen}/coarse.f32`, `idx/${gen}/fine.f32`, ...rows.map((_, cell) => keys.slab(gen, cell)), keys.slabsTable(gen, 0)];
	// The manifest is the create-only lock: written first, so two creators of the same name cannot both succeed.
	if (!(await putIfAbsent(store, keys.manifest(gen, 0), JSON.stringify(manifest), "application/json"))) throw new ThreadError(`generation ${gen} already exists`, 409);
	const w2 = writable(store);
	await Promise.all([w2.put(`idx/${gen}/coarse.f32`, meanBytes), w2.put(`idx/${gen}/fine.f32`, fineBytes), w2.put(keys.slabsTable(gen, 0), JSON.stringify(rows), { contentType: "application/json" })]);
	for (let i = 0; i < slabs.length; i += 16) await Promise.all(slabs.slice(i, i + 16).map((slab, j) => w2.put(keys.slab(gen, i + j), slab)));
	let currentUpdated = false;
	if (opts.setCurrent !== false) {
		const cur = await readCurrent(store);
		const body = currentBody(gen, 0, manifest);
		if (!cur) currentUpdated = await putIfAbsent(store, keys.current(), body, "application/json");
		else currentUpdated = (await w2.put(keys.current(), body, { ifMatch: cur.etag, contentType: "application/json" })) !== null;
	}
	return { gen, rev: 0, manifest, keys: written, currentUpdated };
}
