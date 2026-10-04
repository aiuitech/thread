import { describe, expect, it } from "vitest";
import { amzDateOf, awsEncode, EMPTY_SHA256, S3BlobStore, signV4 } from "../src/s3.js";

// AWS's published SigV4 example (S3 docs, "Example: GET Object" with a Range header).
const CREDS = { accessKeyId: "AKIAIOSFODNN7EXAMPLE", secretAccessKey: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" };

describe("SigV4", () => {
	it("reproduces the AWS GET Object example signature", async () => {
		const h = await signV4({ method: "GET", path: "/test.txt", headers: { range: "bytes=0-9" }, payloadHash: EMPTY_SHA256, host: "examplebucket.s3.amazonaws.com", region: "us-east-1", ...CREDS, amzDate: "20130524T000000Z" });
		expect(h.authorization).toBe("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41");
		expect(h["x-amz-date"]).toBe("20130524T000000Z");
		expect(h.host).toBeUndefined();
	});
	it("reproduces the AWS GET Bucket (list) example with query parameters", async () => {
		const h = await signV4({ method: "GET", path: "/", query: { "max-keys": "2", prefix: "J" }, headers: {}, payloadHash: EMPTY_SHA256, host: "examplebucket.s3.amazonaws.com", region: "us-east-1", ...CREDS, amzDate: "20130524T000000Z" });
		expect(h.authorization).toContain("Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7");
	});
	it("encodes keys AWS-style and formats dates", () => {
		expect(awsEncode("idx/g4/slab/00001", true)).toBe("idx/g4/slab/00001");
		expect(awsEncode("a b+c*~/d", false)).toBe("a%20b%2Bc%2A~%2Fd");
		expect(amzDateOf(Date.UTC(2013, 4, 24, 0, 0, 0))).toBe("20130524T000000Z");
	});
});

/** A minimal S3 server over `fetch`: checks the signature shape, serves ranges, conditional puts and ListObjectsV2. */
function fakeS3(objects: Map<string, Uint8Array>) {
	const calls: { method: string; url: string; headers: Record<string, string> }[] = [];
	const fetchFn: typeof fetch = async (input, init) => {
		const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url);
		const headers = Object.fromEntries(new Headers(init?.headers).entries());
		const method = init?.method ?? "GET";
		calls.push({ method, url: url.href, headers });
		if (!/^AWS4-HMAC-SHA256 Credential=AK\/\d{8}\/[a-z0-9-]+\/s3\/aws4_request, SignedHeaders=.*host.*x-amz-date.*, Signature=[0-9a-f]{64}$/.test(headers.authorization ?? "")) return new Response("bad auth", { status: 403 });
		const key = decodeURIComponent(url.pathname.replace(/^\/(bucket\/?)?/, ""));
		if (method === "GET" && url.searchParams.get("list-type") === "2") {
			const prefix = url.searchParams.get("prefix") ?? "";
			const keys = [...objects.keys()].filter((k) => k.startsWith(prefix)).sort();
			return new Response(`<ListBucketResult><IsTruncated>false</IsTruncated>${keys.map((k) => `<Contents><Key>${k.replace(/&/g, "&amp;")}</Key></Contents>`).join("")}</ListBucketResult>`, { status: 200 });
		}
		const cur = objects.get(key);
		const etag = (b: Uint8Array) => `"${b.byteLength}-${b[0] ?? 0}"`;
		if (method === "PUT") {
			if (headers["if-none-match"] === "*" && cur) return new Response("", { status: 412 });
			if (headers["if-match"] && (!cur || headers["if-match"] !== etag(cur))) return new Response("", { status: 412 });
			const body = new Uint8Array(await new Response(init!.body as BodyInit).arrayBuffer());
			objects.set(key, body);
			return new Response("", { status: 200, headers: { etag: etag(body) } });
		}
		if (!cur) return new Response("<Error><Code>NoSuchKey</Code></Error>", { status: 404 });
		if (method === "HEAD") return new Response(null, { status: 200, headers: { "content-length": String(cur.byteLength), etag: etag(cur) } });
		const suffix = headers.range?.match(/^bytes=-(\d+)$/);
		if (suffix) {
			const a = Math.max(0, cur.byteLength - Number(suffix[1]));
			return new Response(cur.slice(a), { status: 206, headers: { "content-range": `bytes ${a}-${cur.byteLength - 1}/${cur.byteLength}`, etag: etag(cur) } });
		}
		const range = headers.range?.match(/^bytes=(\d+)-(\d+)$/);
		if (range) {
			const a = Number(range[1]);
			const b = Math.min(Number(range[2]), cur.byteLength - 1);
			return new Response(cur.slice(a, b + 1), { status: 206, headers: { "content-range": `bytes ${a}-${b}/${cur.byteLength}` } });
		}
		return new Response(cur.slice(), { status: 200, headers: { etag: etag(cur) } });
	};
	return { fetchFn, calls };
}

describe("S3BlobStore", () => {
	const objects = new Map<string, Uint8Array>();
	const { fetchFn, calls } = fakeS3(objects);
	const store = new S3BlobStore({ endpoint: "https://acct.r2.cloudflarestorage.com", bucket: "bucket", accessKeyId: "AK", secretAccessKey: "SK", fetch: fetchFn });
	it("put (conditional), head, get, range get, list", async () => {
		const h1 = await store.put("idx/current.json", '{"gen":"g4","rev":1}', { ifAbsent: true, contentType: "application/json" });
		expect(h1).not.toBeNull();
		expect(h1!.etag).toMatch(/^\d+-\d+$/); // quotes stripped
		expect(await store.put("idx/current.json", "x", { ifAbsent: true })).toBeNull();
		expect(await store.put("idx/current.json", "x", { ifMatch: "wrong" })).toBeNull();
		expect(await store.put("idx/current.json", '{"gen":"g4","rev":2}', { ifMatch: h1!.etag })).not.toBeNull();
		expect((await store.head("idx/current.json"))?.size).toBe(20);
		expect(await store.head("idx/nope")).toBeNull();
		expect(await store.get("idx/nope")).toBeNull();
		const data = Uint8Array.from({ length: 300 }, (_, i) => i & 0xff);
		await store.put("idx/g4/slab/00001", data);
		expect(await store.get("idx/g4/slab/00001")).toEqual(data);
		expect(await store.get("idx/g4/slab/00001", { offset: 100, length: 50 })).toEqual(data.slice(100, 150));
		expect(await store.list("idx/g4/")).toEqual(["idx/g4/slab/00001"]);
		expect((store as unknown as { getRanges?: unknown }).getRanges).toBeUndefined(); // S3 has no multi-range GET: the SDK falls back to single ranges
		const put = calls.find((c) => c.method === "PUT")!;
		expect(put.url).toBe("https://acct.r2.cloudflarestorage.com/bucket/idx/current.json");
		expect(put.headers["if-none-match"]).toBe("*");
		expect(put.headers["x-amz-content-sha256"]).toMatch(/^[0-9a-f]{64}$/);
		const get = calls.find((c) => c.method === "GET" && c.headers.range)!;
		expect(get.headers.authorization).toContain("SignedHeaders=host;range;x-amz-content-sha256;x-amz-date");
	});
	it("whole-object GETs of large objects are split into parallel range GETs", async () => {
		const chunked = new S3BlobStore({ endpoint: "https://acct.r2.cloudflarestorage.com", bucket: "bucket", accessKeyId: "AK", secretAccessKey: "SK", fetch: fetchFn, chunkBytes: 100, chunkConcurrency: 2 });
		const data = Uint8Array.from({ length: 350 }, (_, i) => (i * 3) & 0xff);
		await chunked.put("idx/g4/fine.f32", data);
		const n0 = calls.length;
		const trace = { reads: 0, hits: 0, misses: 0, r206: 0, r200: 0, bytes: 0 };
		expect(await chunked.get("idx/g4/fine.f32", undefined, { trace })).toEqual(data);
		const gets = calls.slice(n0).map((c) => c.headers.range);
		expect(gets.sort()).toEqual(["bytes=0-99", "bytes=100-199", "bytes=200-299", "bytes=300-349"]);
		expect(trace).toMatchObject({ reads: 4, bytes: 350 });
		const n1 = calls.length;
		expect(await chunked.get("idx/g4/fine.f32", { offset: 20, length: 300 })).toEqual(data.slice(20, 320)); // > 2 chunks: split too
		expect(calls.slice(n1).map((c) => c.headers.range).sort()).toEqual(["bytes=120-219", "bytes=20-119", "bytes=220-319"]);
		expect(await chunked.get("idx/g4/fine.f32", { offset: 20, length: 150 })).toEqual(data.slice(20, 170)); // one GET
		await chunked.put("idx/small", new Uint8Array([1, 2, 3]));
		expect(await chunked.get("idx/small")).toEqual(new Uint8Array([1, 2, 3]));
		expect(await chunked.get("idx/none")).toBeNull();
		// a suffix read is never chunked: one request, the size from Content-Range
		const n2 = calls.length;
		const tail = await chunked.getTail!("idx/g4/fine.f32", 250);
		expect(tail!.size).toBe(350);
		expect(tail!.bytes).toEqual(data.slice(100));
		expect(calls.slice(n2).map((c) => c.headers.range)).toEqual(["bytes=-250"]);
	});
	it("getTail: one suffix-range GET with the object size", async () => {
		const data = Uint8Array.from({ length: 1000 }, (_, i) => (i * 7) & 0xff);
		await store.put("tail/obj", data);
		const t = await store.getTail!("tail/obj", 100);
		expect(t!.size).toBe(1000);
		expect(t!.bytes).toEqual(data.slice(900));
		expect(calls.at(-1)!.headers.range).toBe("bytes=-100");
		expect((await store.getTail!("tail/obj", 5000))!.bytes).toEqual(data); // longer than the object: all of it
		expect(await store.getTail!("tail/nope", 10)).toBeNull();
	});
	it("virtual-hosted style and session tokens", async () => {
		const vh = new S3BlobStore({ endpoint: "https://s3.us-east-1.amazonaws.com", bucket: "bucket", region: "us-east-1", accessKeyId: "AK", secretAccessKey: "SK", sessionToken: "tok", pathStyle: false, fetch: fetchFn });
		expect(await vh.get("idx/current.json")).not.toBeNull();
		const c = calls.at(-1)!;
		expect(c.url).toBe("https://bucket.s3.us-east-1.amazonaws.com/idx/current.json");
		expect(c.headers["x-amz-security-token"]).toBe("tok");
	});
});
