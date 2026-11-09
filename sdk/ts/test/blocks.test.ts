/**
 * The block cache of `CachedBlobStore` over the Cache API: range reads that pass the object's `size` (lexical
 * segments over 2 MiB) are served from cached aligned blocks, a block miss reads and caches that block only, fills
 * are shared and bounded, objects cached whole stay served from their entry, and scopes never share blocks.
 */
import { describe, expect, it } from "vitest";
import { CachedBlobStore, createLexicalGeneration, MemoryBlobStore, MemoryCache, newTrace, type BlobRange, type GetOptions } from "../src/index.js";
import { FakeCache, rng, thread } from "./helpers.js";

const B = 64 * 1024;
const bytes = (n: number, seed = 0) => Uint8Array.from({ length: n }, (_, i) => (i * 31 + seed + (i >> 8)) & 0xff);
const KEY = "idx/g/seg/7.thread";

/** A memory store that records every origin read and can hold reads until a gate opens. */
class CountingStore extends MemoryBlobStore {
	reads: { key: string; range?: BlobRange }[] = [];
	inFlight = 0;
	maxInFlight = 0;
	gate: Promise<void> | null = null;
	override async get(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		this.reads.push({ key, range });
		this.inFlight++;
		this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
		try {
			if (this.gate) await this.gate;
			return await super.get(key, range, opts);
		} finally {
			this.inFlight--;
		}
	}
}

/** A FakeCache whose puts can be held until a gate opens. */
class GatedCache extends FakeCache {
	gate: Promise<void> | null = null;
	override async put(req: Request, res: Response): Promise<void> {
		if (this.gate) await this.gate;
		return super.put(req, res);
	}
}

const gate = () => {
	let open!: () => void;
	const p = new Promise<void>((r) => (open = r));
	return { p, open };
};

async function setup(size: number, opts: { cache?: FakeCache; scope?: string; blockBytes?: number; blockBuffers?: number; inner?: CountingStore } = {}) {
	const inner = opts.inner ?? new CountingStore();
	const obj = bytes(size, size & 7);
	await inner.put(KEY, obj);
	const cache = opts.cache ?? new FakeCache();
	const pending: Promise<unknown>[] = [];
	const s = new CachedBlobStore(inner, cache, { origin: "https://t.cache", scope: opts.scope, waitUntil: (p) => pending.push(p), blockBytes: opts.blockBytes ?? B, blockBuffers: opts.blockBuffers });
	return { inner, cache, s, obj, settle: () => Promise.all(pending), size };
}

const blockUrl = (i: number, scope = "") => `https://t.cache/${scope}${KEY}/~block/${B}/${i}`;

describe("CachedBlobStore blocks", () => {
	it("a range inside one block: the miss reads and caches that block only, the next read hits it", async () => {
		const { inner, cache, s, obj, settle } = await setup(5 * B + 100);
		const trace = newTrace();
		const r = await s.get(KEY, { offset: 2 * B + 10, length: 500 }, { trace, size: 5 * B + 100 });
		expect(r).toEqual(obj.slice(2 * B + 10, 2 * B + 510));
		expect(inner.reads).toEqual([{ key: KEY, range: { offset: 2 * B, length: B } }]);
		expect(trace).toMatchObject({ misses: 1, hits: 0, reads: 1, fills: 1, bytes: B });
		await settle();
		expect([...cache.objects.keys()]).toEqual([blockUrl(2)]);
		expect(cache.objects.get(blockUrl(2))).toEqual(obj.slice(2 * B, 3 * B));
		const t2 = newTrace();
		const again = await s.get(KEY, { offset: 2 * B + 1000, length: 64 }, { trace: t2, size: 5 * B + 100 });
		expect(again).toEqual(obj.slice(2 * B + 1000, 2 * B + 1064));
		expect(inner.reads.length).toBe(1);
		// one cache request with the piece as its range: a 206 of just those bytes
		expect(t2).toMatchObject({ hits: 1, misses: 0, reads: 0, fills: 0, r206: 1, bytes: 64 });
		expect(s.blockStats).toEqual({ fills: 1, bytes: B, failed: 0 });
	});

	it("a range across a block boundary reads both blocks and is reassembled", async () => {
		const size = 4 * B;
		const { inner, s, obj, settle } = await setup(size);
		const r = await s.get(KEY, { offset: B - 100, length: 300 }, { size });
		expect(r).toEqual(obj.slice(B - 100, B + 200));
		expect(inner.reads.map((x) => x.range)).toEqual([
			{ offset: 0, length: B },
			{ offset: B, length: B },
		]);
		await settle();
		const trace = newTrace();
		expect(await s.get(KEY, { offset: B - 1, length: 2 * B + 2 }, { trace, size })).toEqual(obj.slice(B - 1, 3 * B + 1));
		// blocks 0 and 1 hit; blocks 2 and 3 are filled; block 2 is read whole (the piece covers it)
		expect(trace).toMatchObject({ hits: 2, misses: 2, fills: 2 });
		expect(inner.reads.length).toBe(4);
	});

	it("the last block is shorter: its origin read and its entry stop at the object's end", async () => {
		const size = 3 * B + 1234;
		const { inner, cache, s, obj, settle } = await setup(size);
		expect(await s.get(KEY, { offset: size - 34, length: 34 }, { size })).toEqual(obj.slice(size - 34));
		expect(inner.reads).toEqual([{ key: KEY, range: { offset: 3 * B, length: 1234 } }]);
		await settle();
		expect(cache.objects.get(blockUrl(3))!.byteLength).toBe(1234);
		const trace = newTrace();
		expect(await s.get(KEY, { offset: 3 * B, length: 1234 }, { trace, size })).toEqual(obj.slice(3 * B));
		expect(trace).toMatchObject({ hits: 1, misses: 0, reads: 0 });
		// a range past the end is an error, not an origin read
		await expect(s.get(KEY, { offset: size - 10, length: 11 }, { size })).rejects.toThrow(/outside the object/);
		expect(inner.reads.length).toBe(1);
	});

	it("many ranges in one getRanges: grouped by block, each block looked up and filled once", async () => {
		const size = 6 * B;
		const { inner, cache, s, obj, settle } = await setup(size);
		const ranges = [
			{ offset: 10, length: 20 },
			{ offset: 5000, length: 100 },
			{ offset: 2 * B + 7, length: 9 },
			{ offset: 2 * B + 40_000, length: 1000 },
			{ offset: 3 * B - 50, length: 100 }, // crosses into block 3
			{ offset: 5 * B + 3, length: 0 },
			{ offset: 30, length: 10 }, // overlaps nothing, same block as the first
		];
		const trace = newTrace();
		const got = await s.getRanges(KEY, ranges, { trace, size });
		expect(got!.map((b) => b.byteLength)).toEqual(ranges.map((r) => r.length));
		got!.forEach((b, i) => expect(b).toEqual(obj.slice(ranges[i].offset, ranges[i].offset + ranges[i].length)));
		expect(inner.reads.map((x) => x.range!.offset).sort((a, b) => a - b)).toEqual([0, 2 * B, 3 * B]);
		expect(trace).toMatchObject({ misses: 3, fills: 3, reads: 3 });
		expect(cache.matches).toBe(4); // three blocks, then one look at the whole object for the three that missed
		await settle();
		const t2 = newTrace();
		const m0 = cache.matches;
		const warm = await s.getRanges(KEY, ranges, { trace: t2, size });
		warm!.forEach((b, i) => expect(b).toEqual(obj.slice(ranges[i].offset, ranges[i].offset + ranges[i].length)));
		expect(cache.matches - m0).toBe(3); // one request per block, several ranges in it (multipart)
		expect(t2).toMatchObject({ hits: 3, misses: 0, reads: 0, fills: 0, r206: 3 });
		expect(inner.reads.length).toBe(3);
	});

	it("concurrent readers of one block share one origin read", async () => {
		const size = 3 * B;
		const inner = new CountingStore();
		const g = gate();
		const { s, obj, settle } = await setup(size, { inner });
		inner.gate = g.p;
		const a = s.get(KEY, { offset: B + 1, length: 10 }, { size });
		const b = s.getRanges(KEY, [{ offset: B + 100, length: 5 }, { offset: 2 * B - 5, length: 5 }], { size });
		const c = s.get(KEY, { offset: B + 1, length: 10 }, { size });
		await new Promise((r) => setTimeout(r, 5));
		g.open();
		expect(await a).toEqual(obj.slice(B + 1, B + 11));
		expect(await b).toEqual([obj.slice(B + 100, B + 105), obj.slice(2 * B - 5, 2 * B)]);
		expect(await c).toEqual(obj.slice(B + 1, B + 11));
		expect(inner.reads).toEqual([{ key: KEY, range: { offset: B, length: B } }]);
		await settle();
		await s.drain();
		expect(s.blockStats.fills).toBe(1);
	});

	it("bounds the blocks buffered (until their put settles) and the origin reads in flight", async () => {
		const size = 8 * B;
		const cache = new GatedCache();
		const puts = gate();
		cache.gate = puts.p;
		const { inner, s, obj } = await setup(size, { cache, blockBuffers: 2 });
		const reads = Array.from({ length: 6 }, (_, i) => s.get(KEY, { offset: i * B, length: 10 }, { size }));
		await new Promise((r) => setTimeout(r, 10));
		// two blocks read and waiting for their puts; the other four wait for a buffer
		expect(inner.reads.length).toBe(2);
		expect(await reads[0]).toEqual(obj.slice(0, 10));
		puts.open();
		const all = await Promise.all(reads);
		all.forEach((b, i) => expect(b).toEqual(obj.slice(i * B, i * B + 10)));
		expect(inner.reads.length).toBe(6);
		expect(inner.maxInFlight).toBeLessThanOrEqual(2);
		await s.drain();
	});

	it("an object cached whole is served from that entry, not from blocks", async () => {
		const size = 4 * B;
		const { inner, cache, s, obj } = await setup(size);
		cache.objects.set(`https://t.cache/${KEY}`, obj); // cached whole before 0.2.5 (or by a whole read)
		const trace = newTrace();
		expect(await s.getRanges(KEY, [{ offset: 5, length: 5 }, { offset: 3 * B, length: 100 }], { trace, size })).toEqual([obj.slice(5, 10), obj.slice(3 * B, 3 * B + 100)]);
		expect(inner.reads.length).toBe(0);
		expect(cache.puts).toBe(0);
		// the blocks missed (not counted again), the whole entry answered
		expect(trace).toMatchObject({ misses: 2, hits: 1, reads: 0, fills: 0 });
		// once seen, the whole entry is asked first: one cache request
		const m0 = cache.matches;
		const t2 = newTrace();
		expect(await s.get(KEY, { offset: 2 * B - 3, length: 6 }, { trace: t2, size })).toEqual(obj.slice(2 * B - 3, 2 * B + 3));
		expect(cache.matches - m0).toBe(1);
		expect(t2).toMatchObject({ hits: 1, misses: 0 });
	});

	it("scopes never share blocks", async () => {
		const cache = new FakeCache();
		const size = 2 * B;
		const a = await setup(size, { cache, scope: "tenants/a/" });
		const innerB = new CountingStore();
		const objB = bytes(size, 99);
		await innerB.put(KEY, objB);
		const sb = new CachedBlobStore(innerB, cache, { origin: "https://t.cache", scope: "tenants/b/", blockBytes: B });
		expect(await a.s.get(KEY, { offset: 100, length: 50 }, { size })).toEqual(a.obj.slice(100, 150));
		await a.settle();
		expect([...cache.objects.keys()]).toEqual([blockUrl(0, "tenants/a/")]);
		expect(await sb.get(KEY, { offset: 100, length: 50 }, { size })).toEqual(objB.slice(100, 150));
		expect(innerB.reads.length).toBe(1);
		await sb.drain();
		expect(cache.objects.has(blockUrl(0, "tenants/b/"))).toBe(true);
	});

	it("reads with fill, without a size, with blocks off or over an in-process cache are unchanged", async () => {
		const size = 3 * B;
		// fill: the whole object, cached whole
		const f = await setup(size);
		await f.s.get(KEY, { offset: 0, length: 10 }, { fill: true, size });
		expect(f.inner.reads).toEqual([{ key: KEY, range: undefined }]);
		await f.settle();
		expect([...f.cache.objects.keys()]).toEqual([`https://t.cache/${KEY}`]);
		// no size: just the range from the origin, nothing cached
		const n = await setup(size);
		await n.s.get(KEY, { offset: 5, length: 10 });
		expect(n.inner.reads).toEqual([{ key: KEY, range: { offset: 5, length: 10 } }]);
		expect(n.cache.puts).toBe(0);
		// blockBytes 0
		const off = await setup(size, { blockBytes: 0 });
		await off.s.get(KEY, { offset: 5, length: 10 }, { size });
		expect(off.inner.reads).toEqual([{ key: KEY, range: { offset: 5, length: 10 } }]);
		expect(off.s.blockBytes).toBe(0);
		// an in-process cache has no blocks
		const mem = new CachedBlobStore(new MemoryBlobStore(), new MemoryCache());
		expect(mem.blockBytes).toBe(0);
		expect(() => new CachedBlobStore(new MemoryBlobStore(), new FakeCache(), { blockBytes: 1000 })).toThrow(/blockBytes/);
	});

	it("a lexical search over a segment larger than 2 MiB is served from blocks once they are cached", async () => {
		const store = new MemoryBlobStore();
		await createLexicalGeneration({ store, gen: "big", lex: { docs: true, text: true } });
		const writer = await thread.open({ store, cache: false, refreshInterval: 0 });
		const r = rng(5);
		const vocab = Array.from({ length: 4000 }, (_, i) => `w${i.toString(36)}${Math.floor(r() * 1e6).toString(36)}`);
		const texts = Array.from({ length: 3000 }, (_, d) => Array.from({ length: 120 }, () => vocab[Math.floor(r() ** 2 * vocab.length)]).join(" ") + ` doc${d}`);
		await writer.upsert({ keys: texts.map((_, i) => `k/${i}`), texts, metas: texts.map((_, i) => `m${i}`) });
		const seg = writer.segments[0];
		expect(seg.bytes).toBeGreaterThan(2 * 1024 * 1024);
		writer.close();
		const want = await (await thread.open({ store, cache: false })).search({ text: `${vocab[3]} ${vocab[1500]} doc77`, topK: 10, docs: "text", embed: false });

		const cache = new FakeCache();
		const pending: Promise<unknown>[] = [];
		const run = async () => {
			const ix = await thread.open({ store, cache, waitUntil: (p) => pending.push(p), blockBytes: 256 * 1024 });
			try {
				return await ix.search({ text: `${vocab[3]} ${vocab[1500]} doc77`, topK: 10, docs: "text", embed: false });
			} finally {
				ix.close();
			}
		};
		const cold = await run();
		await Promise.all(pending);
		const warm = await run();
		expect(cold.hits).toEqual(want.hits);
		expect(warm.hits).toEqual(want.hits);
		const io = (res: typeof cold) => res.stats.trace.spans.filter((x) => x.name === "lexical" || x.name === "docs").map((x) => x.io!);
		const coldIo = io(cold);
		const warmIo = io(warm);
		expect(coldIo.reduce((a, x) => a + x.fills, 0)).toBeGreaterThan(0);
		expect(warmIo.every((x) => x.reads === 0 && x.misses === 0 && x.fills === 0)).toBe(true);
		expect(warmIo.reduce((a, x) => a + x.hits, 0)).toBeGreaterThan(0);
		// blocks only: the segment is never cached (or read) whole
		const urls = [...cache.objects.keys()].filter((u) => u.includes("/seg/"));
		expect(urls.length).toBeGreaterThan(0);
		expect(urls.every((u) => u.includes("/~block/262144/"))).toBe(true);
		expect(warm.stats.trace.io.fills).toBe(0);
	});
});
