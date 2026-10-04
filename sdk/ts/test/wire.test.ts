import { describe, expect, it } from "vitest";
import { buildTombState, idUnion, keys, manifestFiles, parseCurrent, parseDeltaHeader, parseHits, parseManifest, parseSlabHeader, parseSlabsTable, slabCentroid, slabEntry, slabRevOf, tombFor, tombForDelta, ThreadError } from "../src/wire.js";

function packHits(hits: { id: number; score: number; idx: number }[]): Uint8Array {
	const b = new Uint8Array(4 + hits.length * 12);
	const dv = new DataView(b.buffer);
	dv.setUint32(0, hits.length, true);
	hits.forEach((h, i) => {
		dv.setUint32(4 + i * 12, h.id, true);
		dv.setFloat32(8 + i * 12, h.score, true);
		dv.setUint32(12 + i * 12, h.idx, true);
	});
	return b;
}

/** A synthetic slab header + centroid + entry table (enough for the JS parsers). */
export function fakeSlab(version: 1 | 2 | 3, dim: number, entries: { id: number; attrs: number; payload: Uint8Array }[], pqM = 0): Uint8Array {
	const n = entries.length;
	const align8 = (x: number) => (x + 7) & ~7;
	const centroidOff = 64;
	const entriesOff = align8(centroidOff + dim * 4);
	let scanLen: number;
	let rerankOff = 0;
	let payloadOff: number;
	if (version === 1) {
		const codesOff = align8(entriesOff + n * 16);
		const scalesOff = align8(codesOff + n * dim);
		payloadOff = align8(scalesOff + n * 4);
		scanLen = payloadOff;
	} else {
		const codesOff = align8(entriesOff + n * 16);
		const scalesOff = version === 2 ? align8(codesOff + n * (dim / 2)) : codesOff + n * pqM;
		rerankOff = align8(version === 2 ? scalesOff + n * 4 : scalesOff);
		scanLen = rerankOff;
		payloadOff = align8(rerankOff + n * (4 + dim));
	}
	const payloadLen = entries.reduce((a, e) => a + e.payload.byteLength, 0);
	const b = new Uint8Array(payloadOff + payloadLen);
	const dv = new DataView(b.buffer);
	dv.setUint32(0, 0x53424956, true);
	dv.setUint16(4, version, true);
	dv.setUint32(8, dim, true);
	dv.setUint32(12, n, true);
	dv.setUint32(16, 7, true);
	dv.setUint32(20, centroidOff, true);
	dv.setUint32(24, entriesOff, true);
	if (version === 1) {
		dv.setUint32(36, payloadOff, true);
		dv.setUint32(40, payloadLen, true);
	} else {
		if (version === 3) dv.setUint32(32, pqM, true);
		dv.setUint32(36, rerankOff, true);
		dv.setUint32(40, payloadOff, true);
		dv.setUint32(44, payloadLen, true);
		dv.setUint32(48, scanLen, true);
	}
	for (let d = 0; d < dim; d++) dv.setFloat32(centroidOff + d * 4, d + 0.5, true);
	let po = 0;
	entries.forEach((e, i) => {
		const o = entriesOff + i * 16;
		dv.setUint32(o, e.id, true);
		dv.setUint32(o + 4, po, true);
		dv.setUint32(o + 8, e.payload.byteLength, true);
		dv.setUint32(o + 12, e.attrs, true);
		b.set(e.payload, payloadOff + po);
		po += e.payload.byteLength;
	});
	return b;
}

describe("packed hits", () => {
	it("round-trips", () => {
		const hits = [
			{ id: 7, score: 0.5, idx: 1 },
			{ id: 4294967295, score: -1.25, idx: 99 },
		];
		expect(parseHits(packHits(hits))).toEqual(hits);
		expect(parseHits(packHits([]))).toEqual([]);
		expect(parseHits(new Uint8Array(0))).toEqual([]);
	});
	it("rejects truncated buffers", () => {
		const b = packHits([{ id: 1, score: 1, idx: 0 }]).subarray(0, 10);
		expect(() => parseHits(b)).toThrow(ThreadError);
	});
});

describe("slab header", () => {
	const entries = [
		{ id: 10, attrs: 3, payload: new TextEncoder().encode("hello") },
		{ id: 20, attrs: 0, payload: new Uint8Array(0) },
	];
	it("parses v2", () => {
		const b = fakeSlab(2, 8, entries);
		const h = parseSlabHeader(b);
		expect(h.version).toBe(2);
		expect(h.dim).toBe(8);
		expect(h.count).toBe(2);
		expect(h.cell).toBe(7);
		expect(h.recLen).toBe(12);
		expect(h.scanLen).toBe(h.rerankOff);
		expect(h.pqM).toBe(0);
		expect(h.payloadLen).toBe(5);
		expect(Array.from(slabCentroid(b, h))).toEqual([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]);
		expect(slabEntry(b, h, 0)).toEqual({ id: 10, payloadOff: 0, payloadLen: 5, attrs: 3 });
		expect(slabEntry(b, h, 1)).toEqual({ id: 20, payloadOff: 5, payloadLen: 0, attrs: 0 });
		expect(new TextDecoder().decode(b.subarray(h.payloadOff, h.payloadOff + 5))).toBe("hello");
		// the scan prefix alone is enough for header, centroid and entries
		expect(slabEntry(b.subarray(0, h.scanLen), h, 1).id).toBe(20);
	});
	it("parses v3 with pq_m", () => {
		const h = parseSlabHeader(fakeSlab(3, 8, entries, 4));
		expect(h.version).toBe(3);
		expect(h.pqM).toBe(4);
		expect(h.recLen).toBe(12);
		expect(h.rerankOff).toBeGreaterThan(h.entriesOff);
	});
	it("parses v1 (no rerank tier)", () => {
		const h = parseSlabHeader(fakeSlab(1, 8, entries));
		expect(h.version).toBe(1);
		expect(h.rerankOff).toBe(0);
		expect(h.recLen).toBe(0);
		expect(h.scanLen).toBe(h.payloadOff);
	});
	it("rejects bad magic, short buffers and unknown versions", () => {
		expect(() => parseSlabHeader(new Uint8Array(10))).toThrow(/64 bytes/);
		const b = fakeSlab(2, 8, []);
		b[0] = 0;
		expect(() => parseSlabHeader(b)).toThrow(/magic/);
		const v = fakeSlab(2, 8, []);
		new DataView(v.buffer).setUint16(4, 9, true);
		expect(() => parseSlabHeader(v)).toThrow(/version 9/);
	});
});

describe("delta header", () => {
	it("parses", () => {
		const b = new Uint8Array(64);
		const dv = new DataView(b.buffer);
		dv.setUint32(0, 0x44424956, true);
		dv.setUint16(4, 1, true);
		dv.setUint32(8, 1536, true);
		dv.setUint32(12, 250, true);
		dv.setUint32(16, 192, true);
		dv.setUint32(20, 3, true);
		dv.setUint32(24, 64, true);
		dv.setUint32(28, 6064, true);
		dv.setUint32(44, 500000, true);
		dv.setUint32(48, 12, true);
		expect(parseDeltaHeader(b)).toEqual({ version: 1, dim: 1536, count: 250, pqM: 192, seq: 3, entriesOff: 64, rerankOff: 6064, payloadOff: 500000, payloadLen: 12 });
		b[0] = 1;
		expect(() => parseDeltaHeader(b)).toThrow(/magic/);
	});
});

describe("manifest, current pointer, keys", () => {
	const enc = (o: unknown) => new TextEncoder().encode(JSON.stringify(o));
	it("current.json without rev is rev 0 (legacy manifest.json)", () => {
		expect(parseCurrent(enc({ gen: "g1" }))).toEqual({ gen: "g1", rev: 0 });
		expect(parseCurrent(enc({ gen: "g4", rev: 12 }))).toEqual({ gen: "g4", rev: 12 });
		expect(() => parseCurrent(enc({ rev: 1 }))).toThrow(/gen/);
		expect(keys.manifest("g4", 0)).toBe("idx/g4/manifest.json");
		expect(keys.manifest("g4", 12)).toBe("idx/g4/manifest.12.json");
		expect(keys.slab("g4", 42)).toBe("idx/g4/slab/00042");
		expect(keys.slab("g4", 42, 80)).toBe("idx/g4/slab/00042.80");
		expect(keys.slabsTable("g4", 79)).toBe("idx/g4/slabs.79.json");
		expect(keys.tombstones("g4", 5)).toBe("idx/g4/tombstones.5.u32");
		expect(keys.delta("g4", 3)).toBe("idx/g4/delta/3.vibd");
	});
	it("rev-0 manifests get fixed file names; rev >= 1 use `files`", () => {
		const m0 = parseManifest(enc({ gen: "g4", k1: 1, k2: 4096, dim: 1536, m: 3, vectors: 975000, format: 3 }), "g4", 0);
		expect(m0.rev).toBe(0);
		expect(manifestFiles(m0)).toEqual({ coarse: "idx/g4/coarse.f32", fine: "idx/g4/fine.f32", pq: "idx/g4/pq.bin", slabs: "idx/g4/slabs.json" });
		const m1 = parseManifest(enc({ gen: "g2", k1: 1, k2: 4096, dim: 1536, m: 3, vectors: 1, format: 2 }), "g2", 0);
		expect(manifestFiles(m1).pq).toBeNull();
		const files = { coarse: "idx/g4/coarse.f32", fine: "idx/g4/fine.f32", pq: "idx/g4/pq.bin", slabs: "idx/g4/slabs.11.json" };
		const m12 = parseManifest(enc({ gen: "g4", rev: 12, k1: 1, k2: 4096, dim: 1536, m: 3, vectors: 975000, format: 3, files }), "g4", 12);
		expect(manifestFiles(m12)).toEqual(files);
		expect(() => parseManifest(enc({ gen: "g3", k1: 1, k2: 1, dim: 1, vectors: 1 }), "g4", 0)).toThrow(/generation g3/);
		expect(() => parseManifest(enc({ gen: "g4", rev: 3, k1: 1, k2: 1, dim: 1, vectors: 1 }), "g4", 4)).toThrow(/says rev 3/);
	});
	it("slabs table rows and slab revisions", () => {
		const t = parseSlabsTable(enc([[100, 200, 3, 400], [100, 200, 3, 400, 80]]), 2, "idx/g4/slabs.json");
		expect(slabRevOf(t[0])).toBe(0);
		expect(slabRevOf(t[1])).toBe(80);
		expect(slabRevOf(undefined)).toBe(0);
		expect(() => parseSlabsTable(enc([[1, 2, 3, 4]]), 2, "k")).toThrow(/1 rows/);
	});
});

describe("tombstones", () => {
	it("idUnion sorts and dedupes", () => {
		const u = new Uint32Array(idUnion([Uint32Array.of(5, 1, 9), Uint32Array.of(1, 7)]).slice().buffer);
		expect(Array.from(u)).toEqual([1, 5, 7, 9]);
		expect(idUnion([]).byteLength).toBe(0);
	});
	it("applies the (id, rev) rule for slabs and deltas", () => {
		const t = buildTombState([
			{ rev: 3, ids: Uint32Array.of(1, 2) },
			{ rev: 5, ids: Uint32Array.of(2, 3) },
		]);
		const ids = (b: Uint8Array) => Array.from(new Uint32Array(b.slice().buffer));
		expect(t.total).toBe(4);
		expect(t.maxRev.get(2)).toBe(5);
		expect(ids(tombFor(t, 0))).toEqual([1, 2, 3]); // rev-0 slabs: every set
		expect(ids(tombFor(t, 4))).toEqual([2, 3]); // compaction that read rev 3 applied the rev-3 set
		expect(ids(tombFor(t, 6))).toEqual([]);
		expect(ids(tombForDelta(t, 3))).toEqual([2, 3]); // a delta added at rev 3 is hidden only by newer sets
		expect(ids(tombForDelta(t, 5))).toEqual([]);
		expect(tombFor(t, 4)).toBe(tombFor(t, 4)); // cached
	});
});
