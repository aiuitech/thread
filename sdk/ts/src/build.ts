/**
 * @experimental Bulk build of a dense generation in bounded, resumable steps (`thread.bulkBuild`): a large set of
 * documents with vectors becomes a generation with a multi-cell layout from the start, instead of one cell grown by
 * thousands of small upserts and splits.
 *
 * 1. **Stage** (`add`, any number of calls): each call writes its documents and vectors to `build/{gen}/stage/`
 *    (a JSON object and an f32 object), outside the generation, and nothing else.
 * 2. **Seal** (`seal`): no more adds; the cell count `k` and the sample size follow from the vector count and
 *    `maxBytes`.
 * 3. **Steps** (`step`, one per Durable Object alarm, each bounded by `maxBytes` and `deadlineMs`):
 *    `sample` (an evenly spaced sample of the staged vectors, one stage object at a time), `kmeans` (Lloyd
 *    iterations over the sample in wasm, empty cells reseeded, the means saved after each iteration), `create` (the
 *    generation, rev 0 with `k` empty cells: `createGeneration({ means })`), `load` (each stage replayed in order as
 *    an ordinary upsert, routed to the nearest mean, with the budget-bounded `compact()` whenever it is due), and
 *    `compact` (the remaining deltas folded). Then `done`: the caller switches `idx/current.json` (`setCurrent`) and
 *    `cleanup()` deletes the staging objects.
 *
 * State lives in `build/{gen}/state.json`, written with a compare-and-swap, so a step that crashed is simply run
 * again: a sample chunk, a means iteration and a stage replay are idempotent (an upsert under the same keys or ids
 * replaces what an earlier attempt wrote). One step at a time: a second concurrent step fails with 409.
 */
import { compactionLimits, type CompactionLimits } from "./compact.js";
import type { EmbeddingConfig } from "./embed.js";
import type { Thread, ThreadIndex } from "./index.js";
import type { BlobStore } from "./store.js";
import type { AttrFieldSpec, AttrSchema, AttrValues } from "./attrs.js";
import { COMPACTION_PROFILES, decodeJson, keys, ThreadError, type CompactionPolicy, type LexConfig, type Manifest, type QueryDefaults } from "./wire.js";
import { checkStrings, MAX_UPSERT } from "./write.js";

export interface BulkBuildOptions {
	store: BlobStore;
	/** The generation to build; it must not exist yet (the build creates it in its `create` phase). */
	gen: string;
	/** As `createGeneration`: the embedding model (or `dim`), lexical configuration, attrs, profile, policy, defaults. */
	embedding?: EmbeddingConfig;
	dim?: number;
	lex?: LexConfig | null;
	attrs?: AttrSchema | Record<string, AttrFieldSpec>;
	profile?: "edge" | "local";
	compaction?: Partial<CompactionPolicy>;
	defaults?: Partial<QueryDefaults>;
	notes?: unknown;
	/**
	 * Memory budget of every call (bytes, default 32 MiB), and the `maxBytes` of the compactions the build runs. It
	 * sizes the cells (half of what that budget splits at, `CompactionLimits.cellEntries`), the sample (a quarter of
	 * it) and the largest `add` (an eighth). Use the same budget for the generation's later compactions.
	 */
	maxBytes?: number;
	/** Target entries per cell (default: half of `compactionLimits(...).cellEntries`, so cells can grow before splitting). */
	cellEntries?: number;
	/** Sample size for k-means (default: 256 per cell, at most a quarter of `maxBytes` of f32 vectors). */
	sampleSize?: number;
	/** k-means iterations (default 10). */
	iterations?: number;
	/** Seed of the k-means initialisation (default 7). */
	seed?: number;
	/** Owner of the compaction lock (a Durable Object's id), as `compact({ owner })`. */
	owner?: string;
}

/** One `add`: documents as `upsert` takes them; a document whose vector is null is staged text-only. */
export interface BulkDocs {
	keys?: readonly string[];
	ids?: ArrayLike<number>;
	metas?: (string | null)[];
	texts?: (string | null)[];
	attrs?: ArrayLike<AttrValues>;
	/** One row per document (null: no vector), or every document's row in one flat array. */
	vectors: Float32Array | (Float32Array | number[] | null)[];
}

export type BulkPhase = "stage" | "sample" | "kmeans" | "create" | "load" | "compact" | "done";

export interface BulkBuildState {
	v: 1;
	gen: string;
	phase: BulkPhase;
	dim: number;
	maxBytes: number;
	/** Stage objects written (their seq numbers are 0 .. staged - 1). */
	staged: number;
	docs: number;
	vectors: number;
	/** Fixed at `seal`. */
	k?: number;
	cellEntries?: number;
	sampleSize?: number;
	iterations?: number;
	/** Sample progress: the next stage to read, vector rows seen so far, sample rows written, the chunk objects. */
	sample?: { pos: number; row: number; rows: number; chunks: string[] };
	/** Means after `iter` iterations (`iter` 0: the initial means). */
	kmeans?: { iter: number; means: string };
	/** Next stage to replay. */
	load?: { pos: number };
	/** The generation's head when the build finished. */
	rev?: number;
	/** The options the generation is created with (kept for the `create` phase). */
	create: { embedding?: EmbeddingConfig; lex?: LexConfig | null; attrs?: AttrSchema | Record<string, AttrFieldSpec>; profile?: "edge" | "local"; compaction?: Partial<CompactionPolicy>; defaults?: Partial<QueryDefaults>; notes?: unknown };
	seed: number;
	createdAt: string;
	updatedAt: string;
}

export interface BulkStepResult {
	gen: string;
	/** The phase after the call. */
	phase: BulkPhase;
	done: boolean;
	/** The phases the call worked in. */
	worked: BulkPhase[];
	/** The planner's estimate of the call's peak memory (bytes), at most `maxBytes`. */
	estimatedPeakBytes: number;
	progress: { staged: number; vectors: number; k: number | null; sampleRows: number; sampleSize: number | null; iteration: number; iterations: number | null; loaded: number; deltas: number | null; rev: number | null };
	ms: number;
}

const stateKey = (gen: string) => `build/${gen}/state.json`;
const stageKey = (gen: string, seq: number, ext: "json" | "f32") => `build/${gen}/stage/${String(seq).padStart(6, "0")}.${ext}`;
const sampleKey = (gen: string, pos: number) => `build/${gen}/sample/${String(pos).padStart(6, "0")}.f32`;
const meansKey = (gen: string, iter: number) => `build/${gen}/means.${iter}.f32`;
const DEFAULT_MAX_BYTES = 32 * 1024 * 1024;
const DEFAULT_DEADLINE_MS = 10_000;
/** Time kept for a compaction after the last replay of a `load` step (ms). */
const LOAD_COMPACT_RESERVE_MS = 4_000;
/** Rows a k-means job takes per call (its copy in wasm). */
const KMEANS_CHUNK = 512;
/** Accumulator header of a `FlatKmeansJob.partial()` (thread-core kmeans.rs: magic, k, dim, 0). */
const ACCUM_HEADER = 16;

interface StageDoc {
	n: number;
	keys?: string[];
	ids?: number[];
	metas?: (string | null)[];
	texts?: (string | null)[];
	attrs?: AttrValues[];
	/** Row of each document in the stage's f32 object, -1 without a vector. */
	vec: number[];
}

function writable(store: BlobStore): Required<Pick<BlobStore, "put">> & BlobStore {
	if (!store.put) throw new ThreadError("this store is read-only (no put)", 400);
	return store as Required<Pick<BlobStore, "put">> & BlobStore;
}

const f32Of = (b: Uint8Array): Float32Array => (b.byteOffset % 4 === 0 ? new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2) : new Float32Array(b.slice().buffer));
const bytesOf = (f: Float32Array): Uint8Array => new Uint8Array(f.buffer, f.byteOffset, f.byteLength);

/** A manifest-shaped description of the generation to be (for `compactionLimits` before it exists). */
function plannedManifest(s: BulkBuildState, k: number): Manifest {
	const policy = { ...COMPACTION_PROFILES[s.create.profile ?? "edge"], ...(s.create.compaction ?? {}) };
	return { gen: s.gen, k1: 1, k2: k, dim: s.dim, m: 1, vectors: 0, format: 2, compaction: policy } as Manifest;
}

/** The sizes `seal` fixes (exported for the docs and tests: what a build of n vectors does with a budget). */
export function bulkBuildPlan(n: number, dim: number, opts: { maxBytes?: number; profile?: "edge" | "local"; compaction?: Partial<CompactionPolicy>; cellEntries?: number; sampleSize?: number } = {}): { k: number; cellEntries: number; sampleSize: number; limits: CompactionLimits; maxCells: number } {
	const maxBytes = opts.maxBytes ?? DEFAULT_MAX_BYTES;
	const policy = { ...COMPACTION_PROFILES[opts.profile ?? "edge"], ...(opts.compaction ?? {}) };
	const m = { gen: "_", k1: 1, k2: 1, dim, m: 1, vectors: 0, format: 2, compaction: policy } as Manifest;
	const limits = compactionLimits(m, { maxBytes });
	if (limits.cellEntries < 2) throw new ThreadError(`maxBytes ${maxBytes} cannot split cells of dim ${dim} (compactionLimits): raise it`, 413);
	const cellEntries = Math.max(1, Math.floor(opts.cellEntries ?? limits.cellEntries / 2));
	if (cellEntries > limits.cellEntries) throw new ThreadError(`cellEntries ${cellEntries} is more than this budget splits at (${limits.cellEntries})`, 400);
	const k = Math.max(1, Math.min(n, Math.ceil(n / cellEntries)));
	// k-means holds the sample plus about six copies of the means (theirs, the job's, its sums, the partial, the
	// reducer's, the new ones); the split commits later hold three. A quarter of the budget for the sample.
	const sampleCap = Math.floor(maxBytes / 4 / (dim * 4));
	const maxCells = Math.min(limits.splitCommitCells, Math.floor((maxBytes - sampleCap * dim * 4 - (2 << 20)) / (dim * 4 * 6)));
	const sampleSize = Math.min(n, opts.sampleSize ?? Math.min(256 * k, sampleCap));
	return { k, cellEntries, sampleSize, limits, maxCells };
}

/** Opens a bulk build: resumes the one in `build/{gen}/state.json`, or starts one (`thread.bulkBuild`). */
export async function openBulkBuild(thread: Thread, opts: BulkBuildOptions): Promise<BulkBuild> {
	const { store, gen } = opts;
	writable(store);
	if (!/^[A-Za-z0-9_.-]+$/.test(gen)) throw new ThreadError(`bad generation name ${JSON.stringify(gen)}`, 400);
	const cur = await readState(store, gen);
	if (cur) return new BulkBuild(thread, store, cur.state, cur.etag, opts.owner);
	if (await store.head(keys.manifest(gen, 0))) throw new ThreadError(`generation ${gen} already exists`, 409);
	const dim = opts.embedding?.dim ?? opts.dim;
	if (dim === undefined || !Number.isInteger(dim) || dim < 2 || dim % 2 !== 0) throw new ThreadError("a bulk build needs an even vector size (embedding.dim or dim)", 400);
	if (opts.embedding && opts.dim !== undefined && opts.dim !== opts.embedding.dim) throw new ThreadError(`dim ${opts.dim} != embedding.dim ${opts.embedding.dim}`, 400);
	const maxBytes = opts.maxBytes ?? DEFAULT_MAX_BYTES;
	if (!(maxBytes > 0)) throw new ThreadError("maxBytes must be positive", 400);
	const now = new Date().toISOString();
	const state: BulkBuildState = {
		v: 1,
		gen,
		phase: "stage",
		dim,
		maxBytes,
		staged: 0,
		docs: 0,
		vectors: 0,
		...(opts.cellEntries !== undefined ? { cellEntries: opts.cellEntries } : {}),
		...(opts.sampleSize !== undefined ? { sampleSize: opts.sampleSize } : {}),
		iterations: opts.iterations ?? 10,
		create: { ...(opts.embedding ? { embedding: opts.embedding } : {}), ...(opts.lex !== undefined ? { lex: opts.lex } : {}), ...(opts.attrs ? { attrs: opts.attrs } : {}), ...(opts.profile ? { profile: opts.profile } : {}), ...(opts.compaction ? { compaction: opts.compaction } : {}), ...(opts.defaults ? { defaults: opts.defaults } : {}), ...(opts.notes !== undefined ? { notes: opts.notes } : {}) },
		seed: opts.seed ?? 7,
		createdAt: now,
		updatedAt: now,
	};
	bulkBuildPlan(1, dim, { maxBytes, profile: opts.profile, compaction: opts.compaction }); // the budget can split cells at all
	const h = await writable(store).put(stateKey(gen), JSON.stringify(state), { ifAbsent: true, contentType: "application/json" });
	if (!h) {
		const again = await readState(store, gen);
		if (!again) throw new ThreadError(`build/${gen}/state.json: lost a create race`, 409);
		return new BulkBuild(thread, store, again.state, again.etag, opts.owner);
	}
	return new BulkBuild(thread, store, state, h.etag, opts.owner);
}

async function readState(store: BlobStore, gen: string): Promise<{ state: BulkBuildState; etag: string } | null> {
	const h = await store.head(stateKey(gen));
	if (!h) return null;
	const b = await store.get(stateKey(gen));
	if (!b) return null;
	const state = decodeJson<BulkBuildState>(b);
	if (state.v !== 1 || state.gen !== gen) throw new ThreadError(`build/${gen}/state.json is not a bulk build of ${gen}`, 500);
	return { state, etag: h.etag };
}

export class BulkBuild {
	private ix: ThreadIndex | null = null;

	constructor(
		private readonly thread: Thread,
		private readonly store: BlobStore,
		private st: BulkBuildState,
		private etag: string,
		private readonly owner?: string,
	) {}

	get gen(): string {
		return this.st.gen;
	}

	/** The build's state as last written. */
	get state(): Readonly<BulkBuildState> {
		return this.st;
	}

	/** Writes the state with a compare-and-swap on the one this handle last read or wrote (409 if another changed it). */
	private async save(next: BulkBuildState): Promise<void> {
		next.updatedAt = new Date().toISOString();
		const h = await writable(this.store).put(stateKey(this.st.gen), JSON.stringify(next), { ifMatch: this.etag, contentType: "application/json" });
		if (!h) throw new ThreadError(`build/${this.st.gen}/state.json changed under this build (another step or add ran): reopen it`, 409);
		this.st = next;
		this.etag = h.etag;
	}

	/**
	 * Stages documents (phase `stage` only): one JSON object and one f32 object under `build/{gen}/stage/`. At most
	 * `MAX_UPSERT` documents, `CompactionLimits.upsertVectors` vectors and an eighth of `maxBytes` per call (413
	 * otherwise: split the call), so that every replay folds within the budget. Later adds of the same keys or ids
	 * replace earlier ones, as upserts do.
	 */
	async add(docs: BulkDocs): Promise<{ seq: number; docs: number; vectors: number }> {
		const s = this.st;
		if (s.phase !== "stage") throw new ThreadError(`bulk build ${s.gen} is sealed (phase ${s.phase}): no more adds`, 409);
		const n = docs.keys?.length ?? (docs.ids ? docs.ids.length : 0);
		if (n === 0) throw new ThreadError("an add needs keys or ids", 400);
		if (n > MAX_UPSERT) throw new ThreadError(`at most ${MAX_UPSERT} documents per add`, 400);
		if (docs.keys && docs.ids && docs.ids.length !== n) throw new ThreadError(`${docs.ids.length} ids for ${n} keys`, 400);
		for (const [name, a] of [["metas", docs.metas], ["texts", docs.texts], ["attrs", docs.attrs]] as const) if (a && a.length !== n) throw new ThreadError(`${a.length} ${name} for ${n} documents`, 400);
		checkStrings("metas", docs.metas);
		checkStrings("texts", docs.texts);
		const dim = s.dim;
		const rows: (Float32Array | number[] | null)[] = docs.vectors instanceof Float32Array ? Array.from({ length: docs.vectors.length / dim }, (_, i) => (docs.vectors as Float32Array).subarray(i * dim, (i + 1) * dim)) : docs.vectors;
		if (docs.vectors instanceof Float32Array && docs.vectors.length !== n * dim) throw new ThreadError(`vectors hold ${docs.vectors.length} floats, expected ${n} x ${dim}`, 400);
		if (rows.length !== n) throw new ThreadError(`${rows.length} vectors for ${n} documents`, 400);
		const vec: number[] = [];
		let m = 0;
		for (const r of rows) {
			if (r === null) vec.push(-1);
			else {
				if (r.length !== dim) throw new ThreadError(`a vector has ${r.length} floats, expected ${dim}`, 400);
				vec.push(m++);
			}
		}
		const limits = compactionLimits(plannedManifest(s, 1), { maxBytes: s.maxBytes });
		if (m > limits.upsertVectors) throw new ThreadError(`an add of ${m} vectors is more than one compaction with maxBytes ${s.maxBytes} always folds (${limits.upsertVectors}): split it`, 413);
		const f = new Float32Array(m * dim);
		let j = 0;
		for (const r of rows) if (r !== null) f.set(r, j++ * dim);
		if (!f.every(Number.isFinite)) throw new ThreadError("vectors must be finite", 400);
		const doc: StageDoc = { n, ...(docs.keys ? { keys: [...docs.keys] } : {}), ...(docs.ids ? { ids: Array.from(docs.ids) } : {}), ...(docs.metas ? { metas: docs.metas } : {}), ...(docs.texts ? { texts: docs.texts } : {}), ...(docs.attrs ? { attrs: Array.from(docs.attrs) } : {}), vec };
		const json = JSON.stringify(doc);
		const bytes = json.length * 2 + f.byteLength;
		if (bytes > s.maxBytes / 8) throw new ThreadError(`an add of about ${bytes} bytes is more than an eighth of maxBytes (${Math.floor(s.maxBytes / 8)}), which a replay holds about 5 times: split it`, 413);
		const seq = s.staged;
		// Plain puts: an object for a seq the state does not list yet belongs to an add that never committed.
		const w = writable(this.store);
		await w.put(stageKey(s.gen, seq, "json"), json, { contentType: "application/json" });
		if (m) await w.put(stageKey(s.gen, seq, "f32"), bytesOf(f));
		await this.save({ ...s, staged: seq + 1, docs: s.docs + n, vectors: s.vectors + m });
		return { seq, docs: n, vectors: m };
	}

	/** Ends the stage phase: fixes `k`, the cell size and the sample size from the staged vector count. */
	async seal(): Promise<BulkBuildState> {
		const s = this.st;
		if (s.phase !== "stage") return s;
		if (!s.vectors) throw new ThreadError("nothing staged with a vector: a bulk build needs vectors", 400);
		const plan = bulkBuildPlan(s.vectors, s.dim, { maxBytes: s.maxBytes, profile: s.create.profile, compaction: s.create.compaction, cellEntries: s.cellEntries, sampleSize: s.sampleSize });
		if (plan.k > plan.maxCells) throw new ThreadError(`${s.vectors} vectors at ${plan.cellEntries} per cell need ${plan.k} cells, more than one isolate with maxBytes ${s.maxBytes} can cluster and later split (${plan.maxCells}): raise maxBytes, or build it in the Worker's build slots (bench/driver.mjs)`, 413);
		if (plan.sampleSize < plan.k) throw new ThreadError(`a sample of ${plan.sampleSize} vectors cannot seed ${plan.k} cells: raise maxBytes or sampleSize`, 413);
		await this.save({ ...s, phase: "sample", k: plan.k, cellEntries: plan.cellEntries, sampleSize: plan.sampleSize, sample: { pos: 0, row: 0, rows: 0, chunks: [] } });
		return this.st;
	}

	/**
	 * Advances the build within `deadlineMs` (default 10 s) and `maxBytes`: as many phases and pieces of them as fit.
	 * Call again until `done`. Each piece commits its own progress, so a step cut short (or crashed) loses at most the
	 * piece in flight.
	 */
	async step(opts: { deadlineMs?: number } = {}): Promise<BulkStepResult> {
		const t0 = Date.now();
		const deadline = t0 + (opts.deadlineMs ?? DEFAULT_DEADLINE_MS);
		const worked: BulkPhase[] = [];
		let peak = 0;
		if (this.st.phase === "stage") throw new ThreadError(`bulk build ${this.st.gen} is not sealed`, 409);
		while (this.st.phase !== "done" && Date.now() < deadline) {
			const phase: BulkPhase = this.st.phase;
			if (!worked.includes(phase)) worked.push(phase);
			if (phase === "sample") peak = Math.max(peak, await this.sampleStep(deadline));
			else if (phase === "kmeans") peak = Math.max(peak, await this.kmeansStep(deadline));
			else if (phase === "create") peak = Math.max(peak, await this.createStep());
			else if (phase === "load") peak = Math.max(peak, await this.loadStep(deadline));
			else if (phase === "compact") peak = Math.max(peak, await this.compactStep(deadline));
			if (this.st.phase === phase && Date.now() >= deadline) break;
		}
		if (this.st.phase === "done") this.closeIndex();
		const s = this.st;
		return {
			gen: s.gen,
			phase: s.phase,
			done: s.phase === "done",
			worked,
			estimatedPeakBytes: peak,
			progress: { staged: s.staged, vectors: s.vectors, k: s.k ?? null, sampleRows: s.sample?.rows ?? 0, sampleSize: s.sampleSize ?? null, iteration: s.kmeans?.iter ?? 0, iterations: s.iterations ?? null, loaded: s.load?.pos ?? 0, deltas: this.ix ? (this.ix.manifest.deltas?.length ?? 0) : null, rev: s.rev ?? null },
			ms: Date.now() - t0,
		};
	}

	/** Deletes the staging objects and the state (`build/{gen}/`); the generation stays. */
	async cleanup(): Promise<number> {
		this.closeIndex();
		const store = this.store;
		if (!store.list || !store.delete) throw new ThreadError("cleanup needs a store with list and delete", 400);
		const all = await store.list(`build/${this.st.gen}/`);
		for (let i = 0; i < all.length; i += 16) await Promise.all(all.slice(i, i + 16).map((k) => store.delete!(k)));
		return all.length;
	}

	private closeIndex(): void {
		this.ix?.close();
		this.ix = null;
	}

	// ---------- phases ----------

	/** Evenly spaced rows: row g is in the sample when floor((g + 1) S / N) > floor(g S / N), S rows in all. */
	private async sampleStep(deadline: number): Promise<number> {
		const s = this.st;
		const S = s.sampleSize!;
		const N = s.vectors;
		const dim = s.dim;
		const sm = { ...s.sample! };
		const out = new Float32Array((S - sm.rows) * dim);
		let got = 0;
		let largest = 0;
		let pos = sm.pos;
		let row = sm.row;
		while (pos < s.staged && Date.now() < deadline) {
			const b = await this.store.get(stageKey(s.gen, pos, "f32"));
			if (b) {
				largest = Math.max(largest, b.byteLength);
				const f = f32Of(b);
				const n = f.length / dim;
				for (let i = 0; i < n; i++, row++) if (Math.floor(((row + 1) * S) / N) > Math.floor((row * S) / N)) out.set(f.subarray(i * dim, (i + 1) * dim), got++ * dim);
			}
			pos++;
		}
		const chunks = [...sm.chunks];
		if (got) {
			const key = sampleKey(s.gen, sm.pos);
			await writable(this.store).put(key, bytesOf(out.subarray(0, got * dim)));
			chunks.push(key);
		}
		const done = pos >= s.staged;
		await this.save({ ...s, ...(done ? { phase: "kmeans" as const } : {}), sample: { pos, row, rows: sm.rows + got, chunks } });
		return (S - sm.rows) * dim * 4 + 2 * largest + (1 << 20);
	}

	private async loadSample(): Promise<Float32Array> {
		const s = this.st;
		const out = new Float32Array(s.sample!.rows * s.dim);
		let at = 0;
		for (const key of s.sample!.chunks) {
			const b = await this.store.get(key);
			if (!b) throw new ThreadError(`missing ${key}`, 404);
			out.set(f32Of(b), at);
			at += b.byteLength >> 2;
		}
		if (at !== out.length) throw new ThreadError(`the sample holds ${at / s.dim} rows, expected ${s.sample!.rows}`, 500);
		return out;
	}

	private async kmeansStep(deadline: number): Promise<number> {
		const s = this.st;
		const { dim } = s;
		const k = s.k!;
		const W = this.thread.wasm;
		const sample = await this.loadSample();
		const S = sample.length / dim;
		let means: Float32Array;
		let iter: number;
		if (!s.kmeans) {
			// k distinct sample rows, picked by a seeded shuffle.
			let x = s.seed >>> 0 || 1;
			const order = Array.from({ length: S }, (_, i) => i);
			for (let i = S - 1; i > 0; i--) {
				x ^= x << 13;
				x ^= x >>> 17;
				x ^= x << 5;
				const j = (x >>> 0) % (i + 1);
				[order[i], order[j]] = [order[j], order[i]];
			}
			means = new Float32Array(k * dim);
			for (let c = 0; c < k; c++) means.set(sample.subarray(order[c] * dim, (order[c] + 1) * dim), c * dim);
			iter = 0;
			await writable(this.store).put(meansKey(s.gen, 0), bytesOf(means));
			await this.save({ ...s, kmeans: { iter: 0, means: meansKey(s.gen, 0) } });
		} else {
			const b = await this.store.get(s.kmeans.means);
			if (!b) throw new ThreadError(`missing ${s.kmeans.means}`, 404);
			means = f32Of(b).slice();
			iter = s.kmeans.iter;
		}
		const iterations = s.iterations!;
		while (iter < iterations && Date.now() < deadline) {
			const job = new W.FlatKmeansJob(dim, k, means);
			let partial: Uint8Array;
			try {
				for (let i = 0; i < S; i += KMEANS_CHUNK) job.add_batch(sample.subarray(i * dim, Math.min(S, i + KMEANS_CHUNK) * dim));
				partial = job.partial();
			} finally {
				job.free();
			}
			const red = new W.Reducer(k, dim);
			try {
				red.add_records(partial.subarray(ACCUM_HEADER));
				means = red.means_reseed(means, 1, s.seed + iter + 1);
			} finally {
				red.free();
			}
			iter++;
			const key = meansKey(s.gen, iter);
			await writable(this.store).put(key, bytesOf(means));
			await this.save({ ...this.st, ...(iter >= iterations ? { phase: "create" as const } : {}), kmeans: { iter, means: key } });
		}
		if (iter >= iterations && this.st.phase === "kmeans") await this.save({ ...this.st, phase: "create" });
		return S * dim * 4 + KMEANS_CHUNK * dim * 4 + k * dim * 4 * 6 + (1 << 20);
	}

	private async createStep(): Promise<number> {
		const s = this.st;
		const b = await this.store.get(s.kmeans!.means);
		if (!b) throw new ThreadError(`missing ${s.kmeans!.means}`, 404);
		const means = f32Of(b).slice();
		const c = s.create;
		try {
			await this.thread.createGeneration({ store: this.store, gen: s.gen, ...(c.embedding ? { embedding: c.embedding } : { dim: s.dim }), ...(c.lex !== undefined ? { lex: c.lex } : {}), ...(c.attrs ? { attrs: c.attrs } : {}), ...(c.profile ? { profile: c.profile } : {}), ...(c.compaction ? { compaction: c.compaction } : {}), ...(c.defaults ? { defaults: c.defaults } : {}), ...(c.notes !== undefined ? { notes: c.notes } : {}), means, setCurrent: false });
		} catch (e) {
			// A crash after the create: the generation is ours when it has our cells and nothing else yet.
			if (!(e instanceof ThreadError && e.status === 409)) throw e;
			const m = await this.store.get(keys.manifest(s.gen, 0));
			const man = m ? decodeJson<Manifest>(m) : null;
			if (!man || man.k2 !== s.k || man.dim !== s.dim) throw new ThreadError(`generation ${s.gen} exists and is not this build's`, 409);
		}
		await this.save({ ...s, phase: "load", load: { pos: 0 } });
		return means.byteLength * 3 + (1 << 20);
	}

	private async index(): Promise<ThreadIndex> {
		if (!this.ix) this.ix = await this.thread.open({ store: this.store, gen: this.st.gen, cache: false, refreshInterval: 0, deltaCacheBytes: 0, prefetchBytes: 0 });
		return this.ix;
	}

	private async compactIfDue(ix: ThreadIndex, deadline: number, force: boolean): Promise<{ more: boolean; locked: boolean; progressed: boolean; peak: number }> {
		if (!force && !ix.compactionDue().due) return { more: false, locked: false, progressed: false, peak: 0 };
		const r = await ix.compact({ force, maxBytes: this.st.maxBytes, deadlineMs: Math.max(1_000, deadline - Date.now()), ...(this.owner ? { owner: this.owner } : {}) });
		return { more: r.more, locked: r.outcome === "locked", progressed: r.outcome === "committed" || r.outcome === "partial", peak: r.estimatedPeakBytes };
	}

	private async loadStep(deadline: number): Promise<number> {
		const ix = await this.index();
		let peak = 0;
		while (this.st.phase === "load" && Date.now() < deadline - LOAD_COMPACT_RESERVE_MS) {
			// Compaction first: a replay only while the head is below the thresholds, so the deltas stay few.
			if (ix.compactionDue().due) {
				const c = await this.compactIfDue(ix, deadline, false);
				peak = Math.max(peak, c.peak);
				if (!c.progressed) break; // locked, or the deadline: the next step continues
				continue;
			}
			const s = this.st;
			const pos = s.load!.pos;
			if (pos >= s.staged) {
				await this.save({ ...s, phase: "compact" });
				break;
			}
			const jb = await this.store.get(stageKey(s.gen, pos, "json"));
			if (!jb) throw new ThreadError(`missing ${stageKey(s.gen, pos, "json")}`, 404);
			const doc = decodeJson<StageDoc>(jb);
			const fb = doc.vec.some((r) => r >= 0) ? await this.store.get(stageKey(s.gen, pos, "f32")) : null;
			const f = fb ? f32Of(fb) : new Float32Array(0);
			peak = Math.max(peak, 5 * (jb.byteLength * 2 + f.byteLength) + (1 << 20));
			const pick = (want: boolean) => doc.vec.map((r, i) => ((r >= 0) === want ? i : -1)).filter((i) => i >= 0);
			for (const want of [false, true]) {
				const at = pick(want);
				if (!at.length) continue;
				const sel = <T>(a: T[] | undefined) => (a ? at.map((i) => a[i]) : undefined);
				const vectors = want ? at.map((i) => f.subarray(doc.vec[i] * s.dim, (doc.vec[i] + 1) * s.dim)) : undefined;
				await ix.upsert({ ...(doc.keys ? { keys: sel(doc.keys) } : {}), ...(doc.ids ? { ids: sel(doc.ids) } : {}), ...(doc.metas ? { metas: sel(doc.metas) } : {}), ...(doc.texts ? { texts: sel(doc.texts) } : {}), ...(doc.attrs ? { attrs: sel(doc.attrs) } : {}), ...(vectors ? { vectors } : { embed: false }) });
			}
			await this.save({ ...s, load: { pos: pos + 1 } });
		}
		return peak;
	}

	private async compactStep(deadline: number): Promise<number> {
		const ix = await this.index();
		let peak = 0;
		while (Date.now() < deadline) {
			const c = await this.compactIfDue(ix, deadline, true);
			peak = Math.max(peak, c.peak);
			if (!c.progressed) break; // locked, or the deadline: the next step continues
			if (!c.more && !(ix.manifest.deltas?.length ?? 0)) {
				await this.save({ ...this.st, phase: "done", rev: ix.rev });
				break;
			}
			if (!c.more) break;
		}
		return peak;
	}
}
