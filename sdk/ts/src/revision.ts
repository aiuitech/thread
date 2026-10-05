/**
 * Resolving and loading a revision from a store: `idx/current.json` (or a generation's head) to (gen, rev), then
 * the manifest, slabs table and tombstone sets that revision reads. No wasm here, so the pack writer and other
 * tools can use it without an index.
 */
import { newTrace, nowMs, phaseStat, type BlobStore, type BlobTrace, type PhaseStat } from "./store.js";
import type { RevState } from "./query.js";
import { buildTombState, chooseRouting, groupedTableOf, hasDense, keys, manifestFiles, parseCurrent, parseManifest, parseSlabsTable, routingTablesOf, slabRevOf, ThreadError, type CurrentPointer, type Manifest, type RoutingOption } from "./wire.js";

export async function getRequired(store: BlobStore, key: string, trace?: BlobTrace): Promise<Uint8Array> {
	const b = await store.get(key, undefined, trace ? { trace } : undefined);
	if (!b) throw new ThreadError(`missing object ${key}`, 404);
	return b;
}

/**
 * Wall time and store work of the steps of opening a revision (`index.openStats`): `current` (current.json, or
 * listing the generation's manifests), `manifest`, `slabs` (the slabs table and the tombstone sets, read
 * concurrently). Filled in by `resolveTarget` / `loadRevState` when passed.
 */
export interface RevTimeline {
	current?: PhaseStat;
	manifest?: PhaseStat;
	slabs?: PhaseStat;
}

/** Runs `fn` with a fresh trace and records its wall time and store work as `timeline[name]` (when a timeline is given). */
async function timed<T>(timeline: RevTimeline | undefined, name: keyof RevTimeline, fn: (trace: BlobTrace | undefined) => Promise<T>): Promise<T> {
	if (!timeline) return fn(undefined);
	const t0 = nowMs();
	const trace = newTrace();
	try {
		return await fn(trace);
	} finally {
		timeline[name] = phaseStat(nowMs() - t0, trace);
	}
}

/** The newest manifest revision of `gen` present in the store (needs `list`), else 0 (legacy `manifest.json`). */
export async function headRevision(store: BlobStore, gen: string, trace?: BlobTrace): Promise<number> {
	if (!store.list) return 0;
	const prefix = `idx/${gen}/manifest.`;
	let head = -1;
	const listed = await store.list(prefix);
	if (trace) trace.reads++; // a listing (its size is not accounted)
	for (const key of listed) {
		const m = key.slice(prefix.length).match(/^(?:(\d+)\.)?json$/);
		if (m) head = Math.max(head, m[1] ? Number(m[1]) : 0);
	}
	if (head < 0) throw new ThreadError(`no manifest for generation ${gen}`, 404);
	return head;
}

/**
 * The (gen, rev) to read: both given; `gen` alone at its head (current.json's rev when it names that
 * generation, else the newest manifest in the store); else `idx/current.json` (rev absent = 0, the legacy
 * `idx/{gen}/manifest.json`).
 */
export function resolveTarget(store: BlobStore, gen?: string, rev?: number, timeline?: RevTimeline): Promise<CurrentPointer> {
	return timed(timeline, "current", async (trace) => {
		if (gen !== undefined && rev !== undefined) return { gen, rev };
		const cur = await store.get(keys.current(), undefined, trace ? { trace } : undefined);
		if (!cur) {
			if (gen === undefined) throw new ThreadError("no published index (idx/current.json missing)", 404);
			return { gen, rev: await headRevision(store, gen, trace) };
		}
		const c = parseCurrent(cur);
		if (gen === undefined || gen === c.gen) return rev !== undefined && rev !== c.rev ? { gen: c.gen, rev } : c;
		return { gen, rev: await headRevision(store, gen, trace) };
	});
}

/**
 * Loads a revision: manifest (falling back to nothing else: rev 0 is `manifest.json`), slabs table, tombstone sets.
 * `inline`: the manifest when `idx/current.json` carried it (`CurrentPointer.manifest`); used instead of a GET when it
 * is for (gen, rev).
 */
export async function loadRevState(store: BlobStore, gen: string, rev: number, timeline?: RevTimeline, inline?: Manifest): Promise<RevState> {
	const manifest = await timed(timeline, "manifest", async (trace) => {
		if (inline && inline.gen === gen && (inline.rev ?? 0) === rev) return inline;
		const mb = await store.get(keys.manifest(gen, rev), undefined, trace ? { trace } : undefined);
		if (!mb) throw new ThreadError(`no manifest ${keys.manifest(gen, rev)}`, 404);
		return parseManifest(mb, gen, rev);
	});
	const files = manifestFiles(manifest);
	const format = manifest.format ?? 1;
	const cells = manifest.k1 * manifest.k2;
	const [slabs, sets] = await timed(timeline, "slabs", (trace) => Promise.all([
		(async () => {
			if (!files.slabs || !hasDense(manifest)) return null;
			const b = await store.get(files.slabs, undefined, trace ? { trace } : undefined);
			// v1 generations may predate slabs tables: their slabs are read whole.
			if (!b && format < 2 && !manifest.files) return null;
			if (!b) throw new ThreadError(`missing ${files.slabs}`, 404);
			return parseSlabsTable(b, cells, files.slabs);
		})(),
		Promise.all(
			(manifest.tombstones ?? []).map(async (t) => {
				const b = await getRequired(store, t.key, trace);
				if (b.byteLength % 4 !== 0) throw new ThreadError(`${t.key} is not u32-aligned`);
				return { rev: t.rev, ids: new Uint32Array(b.slice().buffer) };
			}),
		),
	]));
	if (format >= 2 && !slabs && hasDense(manifest)) throw new ThreadError(`${files.slabs} missing for format ${format}`);
	return { gen, rev, manifest, files, format, slabs, deltas: manifest.deltas ?? [], segments: manifest.segments ?? [], tomb: sets.length ? buildTombState(sets) : null, baseKey: `${gen}|${files.coarse}|${files.fine}|${files.pq ?? "-"}` };
}

/**
 * Every object a revision reads, sorted: its manifest, the base files (`coarse`, `fine`, `pq`, `slabs` and any other
 * file the manifest lists), one slab per cell at the revision the slabs table names, the delta segments, tombstone
 * sets and lexical/docs segments. `idx/current.json` is not included: a snapshot writes its own pointer.
 */
export function snapshotKeys(rev: RevState): string[] {
	const { manifest, files } = rev;
	const out = new Set<string>([keys.manifest(rev.gen, rev.rev)]);
	if (hasDense(manifest)) {
		for (const [name, v] of Object.entries(files)) {
			if (name === "slabs" && !rev.slabs) continue; // v1 generations without a slabs table
			if (name === "base" || name === "baseCold") continue; // base packs repeat objects listed here
			if (typeof v === "string" && v) out.add(v);
		}
		// Compact routing tables made from the current means (fine.f16, fine.i8, the tree-grouped table).
		const r = routingTablesOf(manifest, files);
		for (const t of [r?.f16, r?.i8, groupedTableOf(manifest, files)]) if (t) out.add(t.key);
		const cells = manifest.k1 * manifest.k2;
		for (let cell = 0; cell < cells; cell++) out.add(keys.slab(rev.gen, cell, rev.slabs ? slabRevOf(rev.slabs[cell]) : 0));
	}
	// A centroid tree (written by the Worker's publish-rev --tree) is read by the Worker's beam routing.
	const tree = manifest.tree as { key?: unknown } | undefined;
	if (tree && typeof tree.key === "string") out.add(tree.key);
	for (const d of rev.deltas) out.add(d.key);
	for (const t of manifest.tombstones ?? []) out.add(t.key);
	for (const s of rev.segments) out.add(s.key);
	return [...out].sort();
}

/**
 * The objects a cold open of `rev` reads, in the order a base pack stores them: the manifest, the slabs table, the
 * tombstone sets, `coarse.f32`, the PQ codebook, the centroid tree and, for `kind: "base"`, the routing table
 * `routing` picks (`"auto"`: the smallest the revision has). A cold base pack (`kind: "cold"`, for
 * `coldRouting: "tree"`) leaves the routing table out: cold queries range-read the tree-grouped table instead.
 */
export function baseKeys(rev: RevState, kind: "base" | "cold" = "base", routing: RoutingOption = "auto"): string[] {
	return baseKeysOf(rev.manifest, rev.gen, rev.rev, kind, routing, rev.slabs !== null);
}

/** `baseKeys` from a manifest alone (`withSlabs`: whether the revision has a slabs table, v1 ones may not). */
export function baseKeysOf(manifest: Manifest, gen: string, rev: number, kind: "base" | "cold" = "base", routing: RoutingOption = "auto", withSlabs = true): string[] {
	const files = manifestFiles(manifest);
	const out = [keys.manifest(gen, rev)];
	if (hasDense(manifest) && files.slabs && withSlabs) out.push(files.slabs);
	for (const t of manifest.tombstones ?? []) out.push(t.key);
	if (hasDense(manifest)) {
		out.push(files.coarse);
		if (files.pq) out.push(files.pq);
		if (manifest.tree?.key) out.push(manifest.tree.key);
		if (kind === "base") out.push(chooseRouting(manifest, files, routing).key);
	}
	return [...new Set(out)];
}

/** A view of `b` as f32 (copying when the view is not 4-byte aligned). */
export function asF32(b: Uint8Array, what: string): Float32Array {
	if (b.byteLength % 4 !== 0) throw new ThreadError(`${what} is not f32-aligned`);
	return b.byteOffset % 4 === 0 ? new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4) : new Float32Array(b.slice().buffer);
}
