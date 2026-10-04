/**
 * The Node side: FileBlobStore over a temp directory (conditions, ranges, listing, key checks), writing a pack to a
 * file and opening either a directory or a pack from a path with storeAt.
 */
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { copyObjects, PackedBlobStore, snapshotKeys, ThreadError } from "../src/index.js";
import { FileBlobStore, packIndexToFile, storeAt } from "../src/node.js";
import { buildGen, rng, thread, unitRows } from "./helpers.js";

const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "thread-node-test-"));
afterAll(() => fs.rm(tmp, { recursive: true, force: true }));
const bytes = (n: number, seed = 0) => Uint8Array.from({ length: n }, (_, i) => (i * 31 + seed) & 0xff);
const text = (b: Uint8Array | null) => new TextDecoder().decode(b!);

describe("FileBlobStore", () => {
	it("get/head/put/list with conditions, ranges, hidden temp files", async () => {
		const s = new FileBlobStore(path.join(tmp, "store"));
		expect(await s.get("a")).toBeNull();
		expect(await s.head("a")).toBeNull();
		const h1 = await s.put("a", "hello", { ifAbsent: true });
		expect(h1?.size).toBe(5);
		expect(await s.put("a", "again", { ifAbsent: true })).toBeNull();
		expect(await s.put("a", "x", { ifMatch: "nope" })).toBeNull();
		const h2 = await s.put("a", "world!", { ifMatch: h1!.etag });
		expect(h2).not.toBeNull();
		expect(h2!.etag).not.toBe(h1!.etag);
		expect((await s.head("a"))?.etag).toBe(h2!.etag);
		expect(text(await s.get("a", { offset: 1, length: 3 }))).toBe("orl");
		expect((await s.getRanges("a", [{ offset: 0, length: 1 }, { offset: 5, length: 1 }]))!.map(text)).toEqual(["w", "!"]);
		await expect(s.get("a", { offset: 4, length: 10 })).rejects.toThrow(/outside/);
		expect(await s.getRanges("missing", [{ offset: 0, length: 1 }])).toBeNull();
		await s.put("b/1", bytes(3));
		await s.put("b/c/2", bytes(3));
		await s.put("bb", bytes(1));
		expect(await s.list("b/")).toEqual(["b/1", "b/c/2"]);
		expect(await s.list("b")).toEqual(["b/1", "b/c/2", "bb"]);
		expect(await s.list("b/c/")).toEqual(["b/c/2"]);
		expect(await s.list("")).toEqual(["a", "b/1", "b/c/2", "bb"]);
		expect(await s.list("zzz/")).toEqual([]);
		// temp files and the lock directory are not objects
		await fs.writeFile(path.join(s.root, ".tmp-leftover"), "x");
		await fs.mkdir(path.join(s.root, ".thread-lock.d")).catch(() => {});
		expect(await s.list("")).toEqual(["a", "b/1", "b/c/2", "bb"]);
		await fs.rmdir(path.join(s.root, ".thread-lock.d"));
		// the whole object has its own buffer (no pooled Buffer views)
		const whole = (await s.get("a"))!;
		expect(whole.byteOffset).toBe(0);
		expect(whole.buffer.byteLength).toBe(whole.byteLength);
		// a stale lock directory is taken over
		await fs.mkdir(path.join(s.root, ".thread-lock.d"));
		const old = new Date(Date.now() - 120_000);
		await fs.utimes(path.join(s.root, ".thread-lock.d"), old, old);
		expect(await s.put("a", "cas", { ifMatch: h2!.etag })).not.toBeNull();
		expect(text(await s.get("a"))).toBe("cas");
	});

	it("rejects keys that escape the root", async () => {
		const s = new FileBlobStore(path.join(tmp, "esc"));
		for (const bad of ["../x", "/etc/passwd", "a/../../x", "", "a//b", "a/./b", "a/.."]) await expect(s.get(bad)).rejects.toThrow(/bad object key/);
	});

	it("concurrent create-only puts: exactly one wins", async () => {
		const s = new FileBlobStore(path.join(tmp, "race"));
		const results = await Promise.all(Array.from({ length: 16 }, (_, i) => s.put("idx/g/manifest.1.json", `w${i}`, { ifAbsent: true })));
		expect(results.filter((r) => r !== null).length).toBe(1);
		const casers = await Promise.all(Array.from({ length: 8 }, (_, i) => s.put("idx/g/manifest.1.json", `c${i}`, { ifMatch: results.find((r) => r)!.etag })));
		expect(casers.filter((r) => r !== null).length).toBe(1);
	});
});

describe("packs on disk", () => {
	it("an index in a directory, packed to a file, opened from either path", async () => {
		const b = await buildGen("gd", 3, { n: 300 });
		const dir = path.join(tmp, "idx");
		const fsStore = new FileBlobStore(dir);
		await copyObjects(b.store, fsStore);
		const onDisk = await thread.open({ store: await storeAt(dir), cache: false });
		expect(onDisk.store).toBeInstanceOf(FileBlobStore);
		await onDisk.configureLexical({ text: true });
		const fresh = unitRows(10, b.dim, rng(8));
		const ids = Array.from({ length: 10 }, (_, i) => 800 + i);
		await onDisk.upsert({ ids, vectors: fresh, texts: ids.map((i) => `disk doc ${i}`), metas: ids.map((i) => `d${i}`) });
		await onDisk.delete([0, 801]);
		const file = path.join(tmp, "packs", "gd.threadpack");
		const r = await packIndexToFile(fsStore, file);
		expect(r).toMatchObject({ gen: "gd", rev: onDisk.rev });
		expect((await fs.stat(file)).size).toBe(r.bytes);
		expect((await fs.readdir(path.dirname(file))).filter((n) => n.startsWith(".tmp-"))).toEqual([]);
		expect(r.keys.sort()).toEqual(["idx/current.json", ...snapshotKeys(onDisk.revision)].sort());

		const ps = await storeAt(file);
		expect(ps).toBeInstanceOf(PackedBlobStore);
		const fromPack = await thread.open({ store: ps, cache: false });
		expect([fromPack.gen, fromPack.rev]).toEqual(["gd", onDisk.rev]);
		const qr = rng(31);
		for (let i = 0; i < 5; i++) {
			const q = unitRows(1, b.dim, qr);
			expect((await fromPack.query(q, { nprobe: 6 })).hits).toEqual((await onDisk.query(q, { nprobe: 6 })).hits);
			const [a, c] = await Promise.all([onDisk.search({ text: "disk doc", vector: q, topK: 6, docs: true }), fromPack.search({ text: "disk doc", vector: q, topK: 6, docs: true })]);
			expect(c.hits).toEqual(a.hits);
		}
		expect((await fromPack.search({ text: "disk", topK: 3, docs: true })).hits.map((h) => h.doc?.meta)).toEqual((await onDisk.search({ text: "disk", topK: 3, docs: true })).hits.map((h) => h.doc?.meta));
		await expect(fromPack.upsert({ ids: [1], vectors: unitRows(1, b.dim, qr) })).rejects.toThrow(/read-only/);
		onDisk.close();
		fromPack.close();
		// not a pack, not there
		await fs.writeFile(path.join(tmp, "plain.bin"), bytes(100));
		await expect(storeAt(path.join(tmp, "plain.bin"))).rejects.toThrow(/not a thread pack/);
		await expect(storeAt(path.join(tmp, "nothing"))).rejects.toThrow(ThreadError);
	});
});
