/**
 * The cold path: miss policies (whole / range / range+warm) over the Cache API and in-process caches, per-store
 * in-flight and coalescing defaults, the cold-path query policy, the SigV4 signing-key cache, connection warm-up,
 * the Node disk cache and prefetch on refresh.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { CachedBlobStore, coalesceRuns, copyObjects, MemoryBlobStore, MemoryCache, newTrace, PackedBlobStore, packToBytes, S3BlobStore, signV4, SigningKeyCache, snapshotKeys, EMPTY_SHA256, type BlobRange, type GetOptions, type MissPolicy } from "../src/index.js";
import { DiskCache, FileBlobStore } from "../src/node.js";
import { buildGen, FakeCache, rng, thread, unitRows } from "./helpers.js";

/** A memory store that says it is remote (like S3) and records every read. */
class RemoteStore extends MemoryBlobStore {
	readonly remote = true;
	reads: { key: string; range?: BlobRange }[] = [];
	heads = 0;
	delayMs = 0;
	constructor(from?: MemoryBlobStore) {
		super();
		if (from) (this as { objects: MemoryBlobStore["objects"] }).objects = from.objects; // shared: writes through `from` are visible
	}
	override async get(key: string, range?: BlobRange, opts?: GetOptions) {
		this.reads.push({ key, range });
		if (this.delayMs) await new Promise((r) => setTimeout(r, this.delayMs));
		return super.get(key, range, opts);
	}
	override async head(key: string) {
		this.heads++;
		return super.head(key);
	}
	async warmup(n: number) {
		await Promise.all(Array.from({ length: n }, () => this.head("idx/current.json")));
	}
}
// S3 has no multi-range GET: the remote store in these tests has none either.
(RemoteStore.prototype as { getRanges?: unknown }).getRanges = undefined;

const body = Uint8Array.from({ length: 5000 }, (_, i) => (i * 7) & 0xff);

describe("miss policies", () => {
	const setup = async (policy?: MissPolicy) => {
		const inner = new RemoteStore();
		await inner.put("idx/g/slab/00001", body);
		const cache = new MemoryCache();
		const s = new CachedBlobStore(inner, cache, policy ? { missPolicy: policy } : {});
		return { inner, cache, s };
	};

	it("defaults: whole for the Cache API, range+warm for an in-process cache", async () => {
		const inner = new MemoryBlobStore();
		expect(new CachedBlobStore(inner, new FakeCache()).missPolicy).toBe("whole");
		expect(new CachedBlobStore(inner, new MemoryCache()).missPolicy).toBe("range+warm");
		expect(new CachedBlobStore(inner, new FakeCache()).inflight).toBe(12);
		expect(() => new CachedBlobStore(inner, new MemoryCache(), { missPolicy: "bogus" as MissPolicy })).toThrow(/missPolicy/);
	});

	it("whole: a fill miss reads the object once, caches it and returns a view into it", async () => {
		const { inner, cache, s } = await setup("whole");
		const t = newTrace();
		const v = (await s.get("idx/g/slab/00001", { offset: 100, length: 50 }, { trace: t, fill: true }))!;
		expect(v).toEqual(body.subarray(100, 150));
		expect(v.buffer.byteLength).toBe(5000);
		expect(inner.reads).toEqual([{ key: "idx/g/slab/00001", range: undefined }]);
		expect(cache.has("idx/g/slab/00001")).toBe(true);
		expect(t.misses).toBe(1);
		const v2 = (await s.get("idx/g/slab/00001", { offset: 4990, length: 10 }, { trace: t }))!;
		expect(v2).toEqual(body.subarray(4990));
		expect([t.hits, t.r206, inner.reads.length]).toEqual([1, 1, 1]);
	});

	it("range: reads exactly the range and caches nothing", async () => {
		const { inner, cache, s } = await setup("range");
		const t = newTrace();
		expect(await s.get("idx/g/slab/00001", { offset: 100, length: 50 }, { trace: t, fill: true })).toEqual(body.subarray(100, 150));
		expect(await s.get("idx/g/slab/00001", undefined, { trace: t })).toEqual(body);
		await s.drain();
		expect(inner.reads).toEqual([
			{ key: "idx/g/slab/00001", range: { offset: 100, length: 50 } },
			{ key: "idx/g/slab/00001", range: undefined },
		]);
		expect(cache.has("idx/g/slab/00001")).toBe(false);
		expect(t.warms).toBe(0);
	});

	it("range+warm: serves the range now, warms the whole object once in the background", async () => {
		const { inner, cache, s } = await setup("range+warm");
		inner.delayMs = 5;
		const t = newTrace();
		const got = await Promise.all([1, 2, 3].map((i) => s.get("idx/g/slab/00001", { offset: i, length: 10 }, { trace: t, fill: true })));
		expect(got.map((b) => b!.byteLength)).toEqual([10, 10, 10]);
		expect(got[0]!.buffer.byteLength).toBe(10); // just the range, not a view into the object
		expect(t.warms).toBe(1); // deduplicated
		expect(t.reads).toBe(3);
		await s.drain();
		expect(cache.has("idx/g/slab/00001")).toBe(true);
		expect(inner.reads.filter((r) => !r.range).length).toBe(1);
		expect(s.warmStats).toMatchObject({ queued: 1, fetched: 1, bytes: 5000, failed: 0 });
		const t2 = newTrace();
		expect(await s.get("idx/g/slab/00001", { offset: 0, length: 4 }, { trace: t2, fill: true })).toEqual(body.subarray(0, 4));
		expect([t2.hits, t2.reads, t2.warms]).toEqual([1, 0, 0]);
	});

	it("range+warm: warms wait for foreground reads (one at a time while busy) and are bounded", async () => {
		const inner = new RemoteStore();
		for (let i = 0; i < 6; i++) await inner.put(`idx/g/slab/0000${i}`, body);
		inner.delayMs = 10;
		const s = new CachedBlobStore(inner, new MemoryCache(), { warmConcurrency: 3, warmQueue: 1 });
		let maxWhole = 0;
		let whole = 0;
		const orig = inner.get.bind(inner);
		inner.get = async (k, r, o) => {
			if (!r) maxWhole = Math.max(maxWhole, ++whole);
			try {
				return await orig(k, r, o);
			} finally {
				if (!r) whole--;
			}
		};
		await Promise.all(Array.from({ length: 6 }, (_, i) => s.get(`idx/g/slab/0000${i}`, { offset: 0, length: 8 }, { fill: true })));
		await s.drain();
		expect(s.warmStats.queued + s.warmStats.dropped).toBe(6);
		expect(s.warmStats.dropped).toBeGreaterThan(0);
		expect(maxWhole).toBeLessThanOrEqual(3);
		// while foreground reads are in flight at most one warm fetch runs
		const s2 = new CachedBlobStore(inner, new MemoryCache(), { warmConcurrency: 3 });
		maxWhole = 0;
		const slow = Array.from({ length: 6 }, (_, i) => s2.get(`idx/g/slab/0000${i}`, { offset: 0, length: 8 }, { fill: true }));
		const busy = (async () => {
			for (let i = 0; i < 20; i++) await inner.get("idx/g/slab/00000", { offset: 0, length: 1 }).then(() => s2.get("idx/x", { offset: 0, length: 1 }).catch(() => null));
		})();
		await Promise.all([...slow, busy]);
		await s2.drain();
		expect(s2.warmStats.fetched).toBe(6);
	});

	it("over a pack: whole and range+warm fetch the object's range of the pack, never the pack", async () => {
		const b = await buildGen("gpk", 3);
		const { bytes: pack } = await packToBytes(b.store);
		for (const policy of ["whole", "range+warm"] as const) {
			const inner = new RemoteStore();
			await inner.put("packs/g.threadpack", pack);
			const ps = await PackedBlobStore.open(inner, "packs/g.threadpack");
			inner.reads = [];
			const cache = new MemoryCache();
			const store = new CachedBlobStore(ps, cache, { missPolicy: policy });
			const ix = await thread.open({ store, cache: false });
			const r1 = await ix.query(b.vectors.subarray(0, b.dim), { nprobe: 8 });
			await store.drain();
			expect(inner.reads.every((r) => r.key === "packs/g.threadpack" && r.range && r.range.length < pack.byteLength)).toBe(true);
			expect([...(cache as unknown as { items: Map<string, unknown> }).items.keys()].every((k) => k.startsWith("idx/gpk/"))).toBe(true);
			expect(cache.has("idx/gpk/slab/00000")).toBe(true);
			const r2 = await ix.query(b.vectors.subarray(0, b.dim), { nprobe: 8 });
			expect(r2.hits).toEqual(r1.hits);
			expect(r2.stats.io.reads).toBe(0);
			ix.close();
		}
	});
});

describe("queries under each miss policy", () => {
	it("same answers; range policies read prefixes and rerank records from the origin, whole reads slabs whole", async () => {
		const b = await buildGen("gq", 3, { n: 800 });
		const plain = await thread.open({ store: b.store, cache: false });
		const qs = [0, 5, 9, 33].map((i) => b.vectors.subarray(i * b.dim, (i + 1) * b.dim));
		const want = await Promise.all(qs.map((q) => plain.query(q, { nprobe: 8, rerank: 40 })));
		for (const policy of ["whole", "range", "range+warm"] as const) {
			const remote = new RemoteStore(b.store);
			const ix = await thread.open({ store: remote, missPolicy: policy, warmup: false });
			expect(ix.store).toBeInstanceOf(CachedBlobStore); // Node: a MemoryCache in front of a remote store by default
			expect((ix.store as CachedBlobStore).missPolicy).toBe(policy);
			const r1 = await ix.query(qs[0], { nprobe: 8, rerank: 40 });
			expect(r1.hits).toEqual(want[0].hits);
			if (policy === "whole") {
				expect(r1.stats.wholeSlabs).toBe(8);
				expect(r1.stats.rerankLocal).toBe(r1.stats.rerankCandidates);
				expect(r1.stats.rerankReads).toBe(0);
			} else {
				expect(r1.stats.wholeSlabs).toBe(0);
				expect(r1.stats.rerankLocal).toBe(0);
				expect(r1.stats.rerankReads).toBeGreaterThan(0);
				expect(r1.stats.scanBytes).toBe(r1.stats.slabBytes);
				expect(r1.stats.io.warms ?? 0).toBe(policy === "range+warm" ? 8 : 0);
			}
			await (ix.store as CachedBlobStore).drain();
			const rest = await Promise.all(qs.map((q) => ix.query(q, { nprobe: 8, rerank: 40 })));
			rest.forEach((r, i) => expect(r.hits).toEqual(want[i].hits));
			if (policy !== "range") {
				expect(rest[0].stats.io.reads).toBe(0);
				expect(rest[0].stats.wholeSlabs).toBe(8); // memory-cache hits are views into the whole slab
			} else expect(rest[0].stats.io.reads).toBeGreaterThan(0);
			ix.close();
		}
		plain.close();
	});
});

describe("in flight and coalescing", () => {
	it("store defaults: S3 32 in Node, Cache API 12, explicit options win; stats report them", async () => {
		const s3 = new S3BlobStore({ endpoint: "https://x.example", bucket: "b", accessKeyId: "a", secretAccessKey: "s" });
		expect([s3.remote, s3.inflight]).toEqual([true, 32]);
		expect(new S3BlobStore({ endpoint: "https://x.example", bucket: "b", accessKeyId: "a", secretAccessKey: "s", inflight: 6, coalesceBytes: 65536 }).coalesceBytes).toBe(65536);
		const nav = Object.getOwnPropertyDescriptor(globalThis, "navigator");
		Object.defineProperty(globalThis, "navigator", { value: { userAgent: "Cloudflare-Workers" }, configurable: true });
		try {
			expect(new S3BlobStore({ endpoint: "https://x.example", bucket: "b", accessKeyId: "a", secretAccessKey: "s" }).inflight).toBe(6);
		} finally {
			if (nav) Object.defineProperty(globalThis, "navigator", nav);
		}
		const b = await buildGen("gi", 3);
		const remote = Object.assign(new RemoteStore(b.store), { inflight: 32, coalesceBytes: 1 << 20 });
		const ix = await thread.open({ store: remote, warmup: false });
		const r = await ix.query(b.vectors.subarray(0, b.dim), { nprobe: 8 });
		expect([r.stats.inflight, r.stats.coalesceBytes, r.stats.spanBytes]).toEqual([32, 1 << 20, 0]);
		const r2 = await ix.query(b.vectors.subarray(0, b.dim), { nprobe: 8, inflight: 3, coalesceBytes: 4096, spanBytes: 9999 });
		expect([r2.stats.inflight, r2.stats.coalesceBytes, r2.stats.spanBytes]).toEqual([3, 4096, 9999]);
		ix.close();
	});

	it("coalesceRuns: greedy span cap, or one range for the whole span", () => {
		expect(coalesceRuns([1, 2, 5, 9], 100, 400).map((r) => [r.a, r.b])).toEqual([[1, 2], [5, 5], [9, 9]]);
		expect(coalesceRuns([1, 2, 5, 9], 100, 500).map((r) => [r.a, r.b])).toEqual([[1, 5], [9, 9]]);
		expect(coalesceRuns([1, 2, 5, 9], 100, 100, 900).map((r) => [r.a, r.b, r.members.length])).toEqual([[1, 9, 4]]);
		expect(coalesceRuns([1, 2, 5, 9], 100, 100, 800).length).toBe(4);
		expect(coalesceRuns([], 100, 100, 800)).toEqual([]);
	});

	it("larger coalescing or a span read: fewer rerank requests, same answers", async () => {
		const b = await buildGen("gco", 3, { n: 1200 });
		const qs = [3, 17, 40].map((i) => b.vectors.subarray(i * b.dim, (i + 1) * b.dim));
		const runs: Record<string, { reads: number; bytes: number; ids: number[][] }> = {};
		for (const [name, o] of Object.entries({ c4k: {}, c128k: { coalesceBytes: 128 << 10 }, span: { spanBytes: 1 << 20 } })) {
			const ix = await thread.open({ store: new RemoteStore(b.store), cache: false, warmup: false });
			const res = await Promise.all(qs.map((q) => ix.query(q, { nprobe: 8, rerank: 60, speculate: false, ...o })));
			runs[name] = { reads: res.reduce((a, r) => a + r.stats.rerankReads, 0), bytes: res.reduce((a, r) => a + r.stats.rerankBytes, 0), ids: res.map((r) => r.hits.map((h) => h.id)) };
			ix.close();
		}
		expect(runs.c128k.ids).toEqual(runs.c4k.ids);
		expect(runs.span.ids).toEqual(runs.c4k.ids);
		expect(runs.c128k.reads).toBeLessThan(runs.c4k.reads);
		expect(runs.span.reads).toBeLessThanOrEqual(runs.c128k.reads);
		expect(runs.span.bytes).toBeGreaterThanOrEqual(runs.c4k.bytes);
	});
});

describe("cold-path policy", () => {
	it("applies only when most probed slabs are not cached", async () => {
		const b = await buildGen("gcp", 3, { n: 800 });
		const ix = await thread.open({ store: new RemoteStore(b.store), warmup: false, coldPolicy: { nprobe: 3, rerank: 10 } });
		const q = b.vectors.subarray(0, b.dim);
		const cold = await ix.query(q, { nprobe: 8, rerank: 40 });
		expect(cold.stats.cold).toEqual({ misses: 8, probed: 8, applied: true });
		expect(cold.stats.cells.length).toBe(3);
		expect(cold.params).toMatchObject({ nprobe: 3, rerank: 10 });
		expect(cold.stats.rerank).toBe(10);
		// explicit opt-out per query
		const off = await ix.query(q, { nprobe: 8, rerank: 40, coldPolicy: false });
		expect(off.stats.cold).toBeUndefined();
		expect(off.stats.cells.length).toBe(8);
		await (ix.store as CachedBlobStore).drain();
		const warm = await ix.query(q, { nprobe: 8, rerank: 40 });
		expect(warm.stats.cold).toEqual({ misses: 0, probed: 8, applied: false });
		expect(warm.stats.cells.length).toBe(8);
		expect(warm.params).toMatchObject({ nprobe: 8, rerank: 40 });
		// the threshold is a fraction of the probed slabs
		const half = await ix.query(b.vectors.subarray(b.dim * 5, b.dim * 6), { nprobe: 8, coldPolicy: { nprobe: 2, threshold: 1 } });
		expect(half.stats.cold?.applied).toBe(false);
		ix.close();
		// a local store without a cache counts as resident
		const local = await thread.open({ store: b.store, cache: false, coldPolicy: { nprobe: 2 } });
		expect((await local.query(q, { nprobe: 8 })).stats.cold).toEqual({ misses: 0, probed: 8, applied: false });
		local.close();
	});
});

describe("SigV4 signing-key cache", () => {
	const CREDS = { accessKeyId: "AKIAIOSFODNN7EXAMPLE", secretAccessKey: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" };
	it("reproduces the AWS example with a cached key, deriving it once per scope", async () => {
		const keys = new SigningKeyCache(CREDS.secretAccessKey);
		for (let i = 0; i < 3; i++) {
			const h = await signV4({ method: "GET", path: "/test.txt", headers: { range: "bytes=0-9" }, payloadHash: EMPTY_SHA256, host: "examplebucket.s3.amazonaws.com", region: "us-east-1", ...CREDS, amzDate: "20130524T000000Z" }, keys);
			expect(h.authorization).toContain("Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41");
		}
		const l = await signV4({ method: "GET", path: "/", query: { "max-keys": "2", prefix: "J" }, headers: {}, payloadHash: EMPTY_SHA256, host: "examplebucket.s3.amazonaws.com", region: "us-east-1", ...CREDS, amzDate: "20130524T000000Z" }, keys);
		expect(l.authorization).toContain("Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7");
		expect(keys.derived).toBe(1);
		await signV4({ method: "GET", path: "/x", headers: {}, payloadHash: EMPTY_SHA256, host: "h", region: "us-east-1", ...CREDS, amzDate: "20130525T000000Z" }, keys);
		expect(keys.derived).toBe(2);
	});
	it("S3BlobStore signs every request with its cache", async () => {
		const seen: string[] = [];
		const s3 = new S3BlobStore({ endpoint: "https://x.example", bucket: "b", accessKeyId: "a", secretAccessKey: "s", now: () => Date.UTC(2026, 9, 4), fetch: async (_u, init) => (seen.push(new Headers(init?.headers).get("authorization")!), new Response("hi", { status: 200 })) });
		await Promise.all([s3.get("k1"), s3.get("k2"), s3.head("k3")]);
		await s3.warmup(4);
		expect(seen.length).toBe(7);
		expect(s3.signingKeys.derived).toBe(1);
	});
});

describe("connection warm-up", () => {
	it("open warms remote stores (default: the store's inflight), not when disabled", async () => {
		const b = await buildGen("gw", 3);
		const r1 = Object.assign(new RemoteStore(b.store), { inflight: 5 });
		(await thread.open({ store: r1 })).close();
		expect(r1.heads).toBe(5);
		const r2 = new RemoteStore(b.store);
		(await thread.open({ store: r2, warmup: false })).close();
		expect(r2.heads).toBe(0);
		const r3 = new RemoteStore(b.store);
		(await thread.open({ store: r3, warmup: 2 })).close();
		expect(r3.heads).toBe(2);
		const local = new MemoryBlobStore();
		for (const [k, v] of b.store.objects) local.objects.set(k, v);
		(await thread.open({ store: local, warmup: true, cache: false })).close(); // no warmup method: nothing to do
	});
});

describe("DiskCache (Node)", () => {
	const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "thread-dc-"));
	it("the bucket layout, LRU-bounded, survives a restart, readable as a FileBlobStore", async () => {
		const dir = tmp();
		const dc = new DiskCache(dir, { maxBytes: 12000 });
		await dc.put("idx/g/slab/00001", body);
		expect(fs.existsSync(path.join(dir, "idx/g/slab/00001"))).toBe(true);
		expect(dc.has("idx/g/slab/00001")).toBe(true);
		expect(await dc.getRanges("idx/g/slab/00001", [{ offset: 10, length: 5 }])).toEqual([body.subarray(10, 15)]);
		await dc.put("idx/g/slab/00002", body);
		await dc.get("idx/g/slab/00001"); // 00001 is now the most recently used
		await dc.put("idx/g/slab/00003", body);
		await new Promise((r) => setTimeout(r, 20));
		expect(dc.held).toBeLessThanOrEqual(12000);
		expect(dc.has("idx/g/slab/00002")).toBe(false);
		expect(fs.existsSync(path.join(dir, "idx/g/slab/00002"))).toBe(false);
		const again = new DiskCache(dir, { maxBytes: 12000 });
		expect(again.held).toBe(dc.held);
		expect(again.has("idx/g/slab/00001") && again.has("idx/g/slab/00003")).toBe(true);
		expect(new Uint8Array((await new FileBlobStore(dir).get("idx/g/slab/00003"))!)).toEqual(body);
		// a file another process evicted is a miss
		fs.unlinkSync(path.join(dir, "idx/g/slab/00003"));
		expect(await again.get("idx/g/slab/00003")).toBeNull();
		expect(again.has("idx/g/slab/00003")).toBe(false);
	});

	it("in front of a remote store: a restart is not a cold start; copyObjects pre-warms it", async () => {
		const b = await buildGen("gdc", 3, { n: 800 });
		const dir = tmp();
		const remote = new RemoteStore(b.store);
		const q = b.vectors.subarray(0, b.dim);
		const ix1 = await thread.open({ store: remote, cache: new DiskCache(dir), warmup: false });
		const r1 = await ix1.query(q, { nprobe: 8 });
		await (ix1.store as CachedBlobStore).drain();
		ix1.close();
		const ix2 = await thread.open({ store: remote, cache: new DiskCache(dir), warmup: false });
		const r2 = await ix2.query(q, { nprobe: 8 });
		expect(r2.hits).toEqual(r1.hits);
		expect(r2.stats.io.reads).toBe(0);
		expect(r2.stats.io.hits).toBeGreaterThan(0);
		// pre-warming a fresh directory with one revision: every query object comes from disk
		const dir2 = tmp();
		const dc = new DiskCache(dir2);
		const n = await copyObjects(remote, dc, snapshotKeys(ix2.revision));
		expect(n.count).toBe(snapshotKeys(ix2.revision).length);
		ix2.close();
		const before = remote.reads.length;
		const ix3 = await thread.open({ store: remote, cache: new DiskCache(dir2), warmup: false });
		const r3 = await ix3.query(q, { nprobe: 8 });
		expect(r3.hits).toEqual(r1.hits);
		expect(remote.reads.length - before).toBe(1); // current.json only
		// and the cache directory is a directory store (pinned: it holds no current.json)
		const fromDir = await thread.open({ store: new FileBlobStore(dir2), cache: false, gen: "gdc", rev: 0 });
		expect((await fromDir.query(q, { nprobe: 8 })).hits).toEqual(r1.hits);
		fromDir.close();
		ix3.close();
	});
});

describe("prefetch on refresh", () => {
	it("a background refresh switches only once the new revision is loaded; warm() fills the cache", async () => {
		const b = await buildGen("grf", 3, { n: 800 });
		const writer = await thread.open({ store: b.store, cache: false, warmup: false });
		const remote = new RemoteStore(b.store);
		const ix = await thread.open({ store: remote, refreshInterval: 1, warmup: false });
		const q = b.vectors.subarray(0, b.dim);
		await ix.query(q);
		const fresh = unitRows(2, b.dim, rng(5));
		const up = await writer.upsert({ ids: [7000, 7001], vectors: fresh });
		await new Promise((r) => setTimeout(r, 5));
		const reads0 = remote.reads.length;
		const during = await ix.query(fresh.subarray(0, b.dim), { topK: 1 }); // triggers the check, does not wait for it
		expect(during.stats.rev).toBe(0);
		for (let i = 0; i < 50 && ix.rev !== up.rev; i++) await new Promise((r) => setTimeout(r, 5));
		expect(ix.rev).toBe(up.rev);
		const newReads = remote.reads.slice(reads0).map((r) => r.key);
		expect(newReads).toContain(up.keys.delta); // the new delta segment was prefetched before the switch
		expect(newReads.some((k) => k.endsWith("coarse.f32"))).toBe(false); // same base files: not reloaded
		await (ix.store as CachedBlobStore).drain();
		const reads1 = remote.reads.length;
		const after = await ix.query(fresh.subarray(0, b.dim), { topK: 1 });
		expect(after.hits[0].id).toBe(7000);
		expect(remote.reads.slice(reads1).map((r) => r.key).filter((k) => k !== "idx/current.json")).toEqual([]);
		// warm(): the snapshot's objects through the cache, within a budget
		const w = await ix.warm(undefined, { budget: 1 << 30 });
		expect(w.objects + w.skipped).toBe(snapshotKeys(ix.revision).length);
		await (ix.store as CachedBlobStore).drain();
		const cached = new Set(snapshotKeys(ix.revision).filter((k) => (ix.store as CachedBlobStore).contains(k)));
		expect(cached.size).toBe(snapshotKeys(ix.revision).length);
		ix.close();
		writer.close();
	});
});
