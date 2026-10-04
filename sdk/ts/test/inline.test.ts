/**
 * An inline manifest in `idx/current.json`: the writers add a copy of the manifest the pointer names (when small),
 * readers use it instead of the manifest GET at open and on refresh, and fall back to the GET when it is absent or
 * does not check out.
 */
import { describe, expect, it } from "vitest";
import { currentBody, INLINE_MANIFEST_MAX, keys, MemoryBlobStore, parseCurrent, type BlobRange, type BlobStore, type GetOptions, type Manifest, type PutOptions } from "../src/index.js";
import { buildGen, rng, thread, unitRows } from "./helpers.js";

const enc = new TextEncoder();

/** Records the keys read. */
class Counting implements BlobStore {
	reads: string[] = [];
	constructor(readonly inner: MemoryBlobStore) {}
	get(key: string, range?: BlobRange, opts?: GetOptions) {
		this.reads.push(key);
		return this.inner.get(key, range, opts);
	}
	getRanges(key: string, ranges: BlobRange[], opts?: GetOptions) {
		this.reads.push(key);
		return this.inner.getRanges(key, ranges, opts);
	}
	head(key: string) {
		return this.inner.head(key);
	}
	put(key: string, data: Uint8Array | string, opts?: PutOptions) {
		return this.inner.put(key, data, opts);
	}
	list(prefix: string) {
		return this.inner.list(prefix);
	}
}

describe("inline manifest in current.json", () => {
	it("currentBody inlines small manifests only; parseCurrent takes a copy that checks out and ignores others", () => {
		const m = { gen: "g", rev: 3, k1: 1, k2: 4, dim: 8, vectors: 10 } as Manifest;
		const body = currentBody("g", 3, m);
		expect(JSON.parse(body)).toEqual({ gen: "g", rev: 3, manifest: m });
		expect(parseCurrent(enc.encode(body))).toEqual({ gen: "g", rev: 3, manifest: m });
		const big = { ...m, notes: "x".repeat(INLINE_MANIFEST_MAX) } as Manifest;
		expect(JSON.parse(currentBody("g", 3, big))).toEqual({ gen: "g", rev: 3 });
		expect(currentBody("g", 3)).toBe('{"gen":"g","rev":3}');
		// a copy for another revision or generation, or a broken one, is ignored
		expect(parseCurrent(enc.encode(JSON.stringify({ gen: "g", rev: 3, manifest: { ...m, rev: 2 } })))).toEqual({ gen: "g", rev: 3 });
		expect(parseCurrent(enc.encode(JSON.stringify({ gen: "g", rev: 3, manifest: { ...m, gen: "h" } })))).toEqual({ gen: "g", rev: 3 });
		expect(parseCurrent(enc.encode(JSON.stringify({ gen: "g", rev: 3, manifest: { gen: "g", rev: 3 } })))).toEqual({ gen: "g", rev: 3 });
		expect(parseCurrent(enc.encode(JSON.stringify({ gen: "g", rev: 3, manifest: 7 })))).toEqual({ gen: "g", rev: 3 });
		// rev 0 manifests have no rev field
		const m0 = { gen: "g", k1: 1, k2: 4, dim: 8, vectors: 10 } as Manifest;
		expect(parseCurrent(enc.encode(currentBody("g", 0, m0))).manifest).toEqual({ ...m0, rev: 0 });
	});

	it("an open uses the inline copy (no manifest GET); without one it reads the manifest", async () => {
		const built = await buildGen("gi", 3, { k2: 8, n: 300 });
		const manifest = JSON.parse(new TextDecoder().decode((await built.store.get(keys.manifest("gi", 0)))!));
		const counting = new Counting(built.store);
		const plain = await thread.open({ store: counting, cache: false });
		expect(counting.reads).toContain(keys.manifest("gi", 0));
		expect(plain.openStats.manifest.reads).toBe(1);
		await built.store.put(keys.current(), currentBody("gi", 0, manifest));
		counting.reads = [];
		const inl = await thread.open({ store: counting, cache: false });
		expect(counting.reads).not.toContain(keys.manifest("gi", 0));
		expect(inl.openStats.manifest.reads).toBe(0);
		expect(inl.manifest).toEqual(plain.manifest);
		const q = unitRows(1, built.dim, rng(2));
		expect((await inl.query(q)).hits.map((h) => h.id)).toEqual((await plain.query(q)).hits.map((h) => h.id));
	});

	it("writers inline the manifest; refresh on another handle switches without the manifest GET", async () => {
		const built = await buildGen("gw", 3, { k2: 8, n: 300 });
		const writer = await thread.open({ store: built.store, cache: false });
		const counting = new Counting(built.store);
		const reader = await thread.open({ store: counting, cache: false, refreshInterval: 0 });
		const fresh = unitRows(2, built.dim, rng(4));
		const up = await writer.upsert({ ids: [9000, 9001], vectors: fresh });
		const cur = parseCurrent((await built.store.get(keys.current()))!);
		expect(cur.rev).toBe(up.rev);
		expect(cur.manifest).toEqual(writer.manifest);
		counting.reads = [];
		expect(await reader.refresh()).toBe(true);
		expect(reader.rev).toBe(up.rev);
		expect(counting.reads).toEqual([keys.current(), `idx/gw/slabs.json`, ...writer.manifest.tombstones!.map((t) => t.key)]);
		const r = await reader.query(fresh.subarray(0, built.dim), { topK: 1 });
		expect(r.hits[0].id).toBe(9000);
		// a pointer written by an older writer (no copy) still works
		await built.store.put(keys.current(), JSON.stringify({ gen: "gw", rev: up.rev }));
		const old = await thread.open({ store: counting, cache: false });
		expect(old.openStats.manifest.reads).toBe(1);
		expect(old.manifest).toEqual(writer.manifest);
	});
});
