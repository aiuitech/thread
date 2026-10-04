#!/usr/bin/env node
/**
 * End-to-end recall@10 of the routing modes (fine.f32 / fine.f16 / fine.i8 flat routing, cold tree routing at several
 * beams) with the TypeScript SDK over the real R2 S3 endpoint (read-only GETs), on the queries of data/prep.
 *
 *   node bench/cold/routing_recall.mjs [--queries 2000] [--concurrency 16] [--modes f32,f16,i8,cold8,cold16,cold32]
 *        [--gen g4] [--rev 180] [--overlay data/cold/base-overlay] [--base data/cold/g4-base]
 *        [--cache data/cold/rangecache.sqlite] [--json data/cold/routing-recall.json]
 *
 * Objects found in the overlay / base directories (the routing objects and the manifest make_routing.mjs wrote, the
 * downloaded base files) are read from disk; everything else (slab scan prefixes, rerank records) is read from R2
 * by range and kept in a local SQLite range cache, so slabs are never downloaded whole and a second mode re-reads
 * only what its own probes add. Cold modes hold the full routing table back, so every query routes cold (each loads
 * the groups it needs, as the first queries after a cold open do). Reports recall@10 against the ground truth, the
 * probe-set overlap with f32 flat routing and the share of queries whose top 10 equals f32's.
 */
import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { loadDataset, loadEnv, parseArgs, recallAt, root } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const NQ = Number(args.queries ?? 2000);
const CONC = Number(args.concurrency ?? 16);
const MODES = (args.modes ?? "f32,f16,i8,cold8,cold16,cold32").split(",");
const GEN = args.gen ?? "g4";
const REV = Number(args.rev ?? 180);
const DIRS = [args.overlay ?? "data/cold/base-overlay", args.base ?? "data/cold/g4-base"].map((d) => path.resolve(root, d));
const dist = path.join(root, "sdk/ts/dist");
const sdk = await import(path.join(dist, "index.js"));
const thread = await sdk.Thread.init({ wasmModule: new WebAssembly.Module(fs.readFileSync(path.join(dist, "wasm/thread_bg.wasm"))) });
const env = loadEnv(args.env ?? "sdk/ts/examples/basic/.env.secrets");
const s3 = new sdk.S3BlobStore({ endpoint: env.S3_ENDPOINT, bucket: env.S3_BUCKET, region: "auto", accessKeyId: env.S3_ACCESS_KEY_ID, secretAccessKey: env.S3_SECRET_ACCESS_KEY, retries: 3 });
const db = new DatabaseSync(path.resolve(root, args.cache ?? "data/cold/rangecache.sqlite"));
db.exec("CREATE TABLE IF NOT EXISTS r (key TEXT, off INTEGER, len INTEGER, data BLOB, PRIMARY KEY (key, off, len))");
const qGet = db.prepare("SELECT data FROM r WHERE key = ? AND off = ? AND len = ?");
const qPut = db.prepare("INSERT OR REPLACE INTO r VALUES (?, ?, ?, ?)");
const io = { local: 0, cached: 0, remote: 0, remoteBytes: 0 };

const localFile = (key) => DIRS.map((d) => path.join(d, key)).find((f) => fs.existsSync(f));
/** Read-only: local overlay files, else the range cache, else R2 (cached). `hold` keys never resolve. */
function cachedStore(hold = new Set()) {
	const never = new Promise(() => {});
	return {
		async get(key, range) {
			if (hold.has(key)) return never;
			const f = localFile(key);
			if (f) {
				io.local++;
				const b = new Uint8Array(fs.readFileSync(f));
				return range ? b.subarray(range.offset, range.offset + range.length) : b;
			}
			const off = range?.offset ?? -1;
			const len = range?.length ?? -1;
			const row = qGet.get(key, off, len);
			if (row) {
				io.cached++;
				return new Uint8Array(row.data);
			}
			const b = await s3.get(key, range);
			if (!b) return null;
			io.remote++;
			io.remoteBytes += b.byteLength;
			qPut.run(key, off, len, b);
			return b;
		},
		async head(key) {
			const f = localFile(key);
			return f ? { size: fs.statSync(f).size, etag: "local" } : s3.head(key);
		},
	};
}

const ds = loadDataset();
const out = { gen: GEN, rev: REV, queries: NQ, modes: [] };
let ref = null;
for (const mode of MODES) {
	const m = /^cold(\d+)$/.exec(mode);
	const manifest = JSON.parse(fs.readFileSync(localFile(sdk.keys.manifest(GEN, REV)), "utf8"));
	const r = manifest.routing;
	// cold: hold every full routing table back so each query routes cold
	const hold = new Set(m ? [manifest.files.fine, r?.f16?.key, r?.i8?.key].filter(Boolean) : []);
	const index = await thread.open({ store: cachedStore(hold), cache: false, gen: GEN, rev: REV, basePack: false, ...(m ? { coldRouting: "tree", coldBeam: Number(m[1]), routing: "i8" } : { routing: mode }) });
	const t0 = performance.now();
	const res = new Array(NQ);
	let next = 0;
	await Promise.all(
		Array.from({ length: CONC }, async () => {
			for (let i = next++; i < NQ; i = next++) {
				const q = await index.query(ds.vector(i), { topK: 10 });
				res[i] = { ids: q.hits.map((h) => h.id), cells: q.stats.cells, routing: q.stats.routing, table: q.stats.routingTable };
			}
		}),
	);
	const recall = res.reduce((a, x, i) => a + recallAt(x.ids, ds.truth(i)), 0) / NQ;
	const row = { mode, routing: res[0].routing, table: res[0].table, recall, secs: (performance.now() - t0) / 1000 };
	if (!ref) ref = res;
	else {
		row.overlap = res.reduce((a, x, i) => a + x.cells.filter((c) => ref[i].cells.includes(c)).length / ref[i].cells.length, 0) / NQ;
		row.sameTop10 = res.filter((x, i) => x.ids.length === ref[i].ids.length && x.ids.every((id, j) => id === ref[i].ids[j])).length / NQ;
		row.recallDelta = recall - out.modes[0].recall;
	}
	out.modes.push(row);
	console.log(`${mode.padEnd(8)} routing ${row.routing}/${row.table}  recall@10 ${(recall * 100).toFixed(2)}%` + (row.overlap !== undefined ? `  (${row.recallDelta >= 0 ? "+" : ""}${(row.recallDelta * 100).toFixed(2)} pts)  probe overlap ${(row.overlap * 100).toFixed(2)}%  same top-10 ${(row.sameTop10 * 100).toFixed(1)}%` : "") + `  ${row.secs.toFixed(0)} s  io ${JSON.stringify(io)}`);
	if (!m) index.close();
}
if (args.json) fs.writeFileSync(path.resolve(root, args.json), JSON.stringify(out, null, 1));
process.exit(0);
