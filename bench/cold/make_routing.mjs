#!/usr/bin/env node
/**
 * Builds the routing objects `publish-rev {routing}` writes, OFFLINE, for an existing generation whose base files were
 * copied to a local directory, and writes them plus a new manifest revision into an overlay directory (for
 * fake_s3.mjs --overlay or a local file store). Never writes to R2.
 *
 *   node bench/cold/make_routing.mjs [--base data/cold/g4-base] [--out data/cold/base-overlay] [--gen g4]
 *        [--rev 179] [--grouped i8] [--extra-grouped f16,f32] [--current 1] [--packs 1]
 *
 * Reads `{base}/idx/{gen}/manifest.{rev}.json` and the files it names (fine, tree), writes under `{out}`:
 *   idx/{gen}/fine{.r}.f16, fine{.r}.i8          compact routing tables (sdk/ts/src/routing.ts, the Worker's builder)
 *   idx/{gen}/fine.grouped.{rev+1}.{elem}        tree-grouped table(s): `--grouped` is named by the manifest,
 *                                                `--extra-grouped` ones are written for measurements only
 *   idx/{gen}/manifest.{rev+1}.json              rev + 1 = rev + routing + files.base/baseCold (--packs 1)
 *                                                (+ change {op: "publish-rev"})
 *   idx/current.json                             {gen, rev: rev + 1} (with --current 1, default)
 * The base pack(s) of item 3 are written by make_base_pack.mjs.
 */
import fs from "node:fs";
import path from "node:path";
import { parseArgs, root } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const BASE = path.resolve(root, args.base ?? "data/cold/g4-base");
const OUT = path.resolve(root, args.out ?? "data/cold/base-overlay");
const GEN = args.gen ?? "g4";
const dist = path.join(root, "sdk/ts/dist");
const sdk = await import(path.join(dist, "index.js"));
const thread = await sdk.Thread.init({ wasmModule: new WebAssembly.Module(fs.readFileSync(path.join(dist, "wasm/thread_bg.wasm"))) });
const w = thread.wasm;

const read = (key) => fs.readFileSync(path.join(BASE, key));
const write = (key, bytes) => {
	const f = path.join(OUT, key);
	fs.mkdirSync(path.dirname(f), { recursive: true });
	fs.writeFileSync(f, bytes);
	console.log(`wrote ${key} (${typeof bytes === "string" ? bytes.length : bytes.byteLength} bytes)`);
};

const rev = Number(args.rev ?? JSON.parse(read("idx/current.json")).rev);
const m = JSON.parse(read(sdk.keys.manifest(GEN, rev)));
const files = sdk.manifestFiles(m);
const fineBytes = read(files.fine);
const fine = new Float32Array(fineBytes.buffer, fineBytes.byteOffset, fineBytes.byteLength / 4);
if (fine.length !== m.k1 * m.k2 * m.dim) throw new Error(`${files.fine} has ${fine.length / m.dim} rows, expected ${m.k1 * m.k2}`);
const next = rev + 1;
const tree = m.tree ? { key: m.tree.key, bytes: new Uint8Array(read(m.tree.key)) } : null;
const elem = args.grouped ?? "i8";
const t0 = performance.now();
const built = sdk.buildRoutingObjects(w, { fineKey: files.fine, fine, dim: m.dim, grouped: tree ? { gen: GEN, rev: next, treeKey: tree.key, tree: tree.bytes, elem } : undefined });
console.log(`built ${built.objects.length} objects in ${(performance.now() - t0).toFixed(0)} ms`);
for (const [k, b] of built.objects) write(k, b);
for (const extra of (args["extra-grouped"] ?? "").split(",").filter(Boolean)) {
	if (!tree || extra === elem) continue;
	const b = w.grouped_table_encode(fine, m.dim, tree.bytes, extra);
	write(sdk.keys.groupedTable(GEN, next, extra), b);
}
const { notes: _n, change: _c, ...rest } = m;
// publish-rev names the base packs it writes after the commit (make_base_pack.mjs writes them here).
const packFiles = { base: sdk.keys.basePack(GEN, next), ...(built.routing.grouped ? { baseCold: sdk.keys.baseColdPack(GEN, next) } : {}) };
const { base: _b, baseCold: _bc, ...plainFiles } = files;
const manifest = { ...rest, rev: next, parentRev: rev, updatedAt: new Date().toISOString(), files: { ...plainFiles, ...((args.packs ?? "1") !== "0" ? packFiles : {}) }, routing: built.routing, change: { op: "publish-rev", routing: { f16: built.routing.f16?.key ?? null, i8: built.routing.i8?.key ?? null, grouped: built.routing.grouped?.key ?? null }, offline: true } };
write(sdk.keys.manifest(GEN, next), JSON.stringify(manifest));
if ((args.current ?? "1") !== "0") write(sdk.keys.current(), JSON.stringify({ gen: GEN, rev: next }));
console.log(JSON.stringify(built.routing, null, 1));
