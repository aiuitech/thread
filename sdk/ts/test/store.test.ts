import { describe, expect, it } from "vitest";
import { CachedBlobStore, MemoryBlobStore, newTrace, parseMultipart, PrefixedBlobStore, rangeHeader, responseParts, sliceRanges } from "../src/store.js";
import { FakeCache } from "./helpers.js";

const bytes = (n: number, seed = 0) => Uint8Array.from({ length: n }, (_, i) => (i * 31 + seed) & 0xff);
const tick = () => new Promise((r) => setTimeout(r, 0));

describe("MemoryBlobStore", () => {
	it("get/head/put/list with conditions", async () => {
		const s = new MemoryBlobStore();
		expect(await s.get("a")).toBeNull();
		const h1 = await s.put("a", "hello", { ifAbsent: true });
		expect(h1?.size).toBe(5);
		expect(await s.put("a", "again", { ifAbsent: true })).toBeNull();
		expect(await s.put("a", "x", { ifMatch: "nope" })).toBeNull();
		const h2 = await s.put("a", "world!", { ifMatch: h1!.etag });
		expect(h2).not.toBeNull();
		expect((await s.head("a"))?.etag).toBe(h2!.etag);
		expect(new TextDecoder().decode((await s.get("a", { offset: 1, length: 3 }))!)).toBe("orl");
		await s.put("b/1", bytes(3));
		expect(await s.list("b/")).toEqual(["b/1"]);
	});
});

describe("multipart/byteranges parsing", () => {
	it("splits parts and slices requested ranges", async () => {
		const body = bytes(5000);
		const cache = new FakeCache();
		cache.objects.set("https://x/k", body);
		const ranges = [
			{ offset: 10, length: 20 },
			{ offset: 1000, length: 5 },
			{ offset: 4990, length: 10 },
		];
		const res = await cache.match(new Request("https://x/k", { headers: { range: rangeHeader(ranges) } }));
		const ct = res!.headers.get("content-type")!;
		const parts = parseMultipart(new Uint8Array(await res!.arrayBuffer()), ct.match(/boundary=(.*)/)![1]);
		expect(parts.map((p) => [p.start, p.end])).toEqual([[10, 29], [1000, 1004], [4990, 4999]]);
		const out = sliceRanges(parts, ranges, "k");
		expect(out[0]).toEqual(body.slice(10, 30));
		expect(out[2]).toEqual(body.slice(4990, 5000));
		expect(() => sliceRanges(parts, [{ offset: 0, length: 1 }], "k")).toThrow(/missing from the response/);
	});
	it("handles single-part 206 and 200 answers", () => {
		const body = bytes(100);
		const single = responseParts(206, null, "bytes 10-19/100", body.slice(10, 20));
		expect(single.kind).toBe("single");
		expect(sliceRanges(single.parts, [{ offset: 12, length: 2 }], "k")[0]).toEqual(body.slice(12, 14));
		const full = responseParts(200, "application/octet-stream", null, body);
		expect(full.kind).toBe("full");
		expect(sliceRanges(full.parts, [{ offset: 90, length: 10 }], "k")[0]).toEqual(body.slice(90));
		expect(() => responseParts(206, "multipart/byteranges", null, body)).toThrow(/boundary/);
	});
});

describe("CachedBlobStore", () => {
	it("range read: miss fills the whole object (view into it), hit answers 206", async () => {
		const inner = new MemoryBlobStore();
		const obj = bytes(10_000, 3);
		await inner.put("idx/g/slab/00001", obj);
		const cache = new FakeCache();
		const pending: Promise<unknown>[] = [];
		const s = new CachedBlobStore(inner, cache, { origin: "https://t.cache", waitUntil: (p) => pending.push(p) });
		const trace = newTrace();
		const miss = await s.get("idx/g/slab/00001", { offset: 0, length: 1000 }, { trace, fill: true });
		expect(miss!.byteLength).toBe(1000);
		expect(miss!.byteOffset).toBe(0);
		expect(miss!.buffer.byteLength).toBe(10_000); // the whole object is recoverable from the view
		expect(trace).toMatchObject({ misses: 1, hits: 0, reads: 1, bytes: 10_000 });
		await Promise.all(pending);
		expect(cache.puts).toBe(1);
		expect(cache.objects.get("https://t.cache/idx/g/slab/00001")).toEqual(obj);
		const hit = await s.get("idx/g/slab/00001", { offset: 500, length: 100 }, { trace });
		expect(hit).toEqual(obj.slice(500, 600));
		expect(hit!.buffer.byteLength).toBe(100);
		expect(trace).toMatchObject({ hits: 1, r206: 1 });
		// without fill, a miss reads only the range from the origin and caches nothing
		await inner.put("idx/g/slab/00002", obj);
		const r = await s.get("idx/g/slab/00002", { offset: 20, length: 5 }, { trace });
		expect(r).toEqual(obj.slice(20, 25));
		expect(cache.puts).toBe(1);
		expect(await s.get("idx/g/none")).toBeNull();
	});
	it("multi-range hit, 200 fallback and miss delegation", async () => {
		const inner = new MemoryBlobStore();
		const obj = bytes(8000, 9);
		await inner.put("idx/g/slab/00007", obj);
		const cache = new FakeCache();
		const s = new CachedBlobStore(inner, cache, { origin: "https://t.cache/" });
		const ranges = [
			{ offset: 100, length: 10 },
			{ offset: 7000, length: 1000 },
		];
		const trace = newTrace();
		expect((await s.getRanges("idx/g/slab/00007", ranges, { trace }))![1]).toEqual(obj.slice(7000)); // miss: the inner store serves the ranges
		expect(trace.misses).toBe(1);
		expect(cache.puts).toBe(0);
		const filled = await s.getRanges("idx/g/slab/00007", ranges, { trace, fill: true });
		expect(filled![1]).toEqual(obj.slice(7000));
		await tick();
		const got = await s.getRanges("idx/g/slab/00007", ranges, { trace });
		expect(got![0]).toEqual(obj.slice(100, 110));
		expect(got![1]).toEqual(obj.slice(7000, 8000));
		expect(trace.r206).toBe(1);
		cache.ignoreRange = true;
		const full = await s.getRanges("idx/g/slab/00007", ranges, { trace });
		expect(full![0]).toEqual(obj.slice(100, 110));
		expect(trace.r200).toBe(1);
		const one = await s.get("idx/g/slab/00007", { offset: 7990, length: 10 }, { trace });
		expect(one).toEqual(obj.slice(7990));
		expect(trace.r200).toBe(2);
	});
	it("never caches current.json; puts and lists go to the inner store", async () => {
		const inner = new MemoryBlobStore();
		await inner.put("idx/current.json", '{"gen":"g1"}');
		const cache = new FakeCache();
		const s = new CachedBlobStore(inner, cache);
		await s.get("idx/current.json");
		await tick();
		expect(cache.matches).toBe(0);
		expect(cache.puts).toBe(0);
		await s.put("idx/x", "1", { ifAbsent: true });
		expect(await s.list("idx/")).toEqual(["idx/current.json", "idx/x"]);
	});
	it("PrefixedBlobStore prepends the prefix", async () => {
		const inner = new MemoryBlobStore();
		const s = new PrefixedBlobStore(inner, "tenants/a/");
		await s.put("idx/current.json", "{}");
		expect(await inner.get("tenants/a/idx/current.json")).not.toBeNull();
		expect(await s.list("idx/")).toEqual(["idx/current.json"]);
	});
});
