/**
 * Resolving and loading a revision from a store: `idx/current.json` (or a generation's head) to (gen, rev), then
 * the manifest, slabs table and tombstone sets that revision reads. No wasm here, so the pack writer and other
 * tools can use it without an index.
 */
import type { BlobStore } from "./store.js";
import type { RevState } from "./query.js";
import { buildTombState, hasDense, keys, manifestFiles, parseCurrent, parseManifest, parseSlabsTable, slabRevOf, ThreadError, type CurrentPointer, type Manifest } from "./wire.js";

export async function getRequired(store: BlobStore, key: string): Promise<Uint8Array> {
	const b = await store.get(key);
	if (!b) throw new ThreadError(`missing object ${key}`, 404);
	return b;
}

/** The newest manifest revision of `gen` present in the store (needs `list`), else 0 (legacy `manifest.json`). */
export async function headRevision(store: BlobStore, gen: string): Promise<number> {
	if (!store.list) return 0;
	const prefix = `idx/${gen}/manifest.`;
	let head = -1;
	for (const key of await store.list(prefix)) {
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
export async function resolveTarget(store: BlobStore, gen?: string, rev?: number): Promise<CurrentPointer> {
	if (gen !== undefined && rev !== undefined) return { gen, rev };
	const cur = await store.get(keys.current());
	if (!cur) {
		if (gen === undefined) throw new ThreadError("no published index (idx/current.json missing)", 404);
		return { gen, rev: await headRevision(store, gen) };
	}
	const c = parseCurrent(cur);
	if (gen === undefined || gen === c.gen) return rev !== undefined ? { gen: c.gen, rev } : c;
	return { gen, rev: await headRevision(store, gen) };
}

/** Loads a revision: manifest (falling back to nothing else: rev 0 is `manifest.json`), slabs table, tombstone sets. */
export async function loadRevState(store: BlobStore, gen: string, rev: number): Promise<RevState> {
	const mb = await store.get(keys.manifest(gen, rev));
	if (!mb) throw new ThreadError(`no manifest ${keys.manifest(gen, rev)}`, 404);
	const manifest = parseManifest(mb, gen, rev);
	const files = manifestFiles(manifest);
	const format = manifest.format ?? 1;
	const cells = manifest.k1 * manifest.k2;
	const [slabs, sets] = await Promise.all([
		(async () => {
			if (!files.slabs || !hasDense(manifest)) return null;
			const b = await store.get(files.slabs);
			// v1 generations may predate slabs tables: their slabs are read whole.
			if (!b && format < 2 && !manifest.files) return null;
			if (!b) throw new ThreadError(`missing ${files.slabs}`, 404);
			return parseSlabsTable(b, cells, files.slabs);
		})(),
		Promise.all(
			(manifest.tombstones ?? []).map(async (t) => {
				const b = await getRequired(store, t.key);
				if (b.byteLength % 4 !== 0) throw new ThreadError(`${t.key} is not u32-aligned`);
				return { rev: t.rev, ids: new Uint32Array(b.slice().buffer) };
			}),
		),
	]);
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
			if (typeof v === "string" && v) out.add(v);
		}
		const cells = manifest.k1 * manifest.k2;
		for (let cell = 0; cell < cells; cell++) out.add(keys.slab(rev.gen, cell, rev.slabs ? slabRevOf(rev.slabs[cell]) : 0));
	}
	// A centroid tree (written by the Worker's publish-rev --tree) is read by the Worker's beam routing.
	const tree = (manifest as Manifest & { tree?: { key?: unknown } }).tree;
	if (tree && typeof tree.key === "string") out.add(tree.key);
	for (const d of rev.deltas) out.add(d.key);
	for (const t of manifest.tombstones ?? []) out.add(t.key);
	for (const s of rev.segments) out.add(s.key);
	return [...out].sort();
}

/** A view of `b` as f32 (copying when the view is not 4-byte aligned). */
export function asF32(b: Uint8Array, what: string): Float32Array {
	if (b.byteLength % 4 !== 0) throw new ThreadError(`${what} is not f32-aligned`);
	return b.byteOffset % 4 === 0 ? new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4) : new Float32Array(b.slice().buffer);
}
