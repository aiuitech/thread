#!/usr/bin/env node
/**
 * Writes the base packs of a revision OFFLINE, the way the Worker's publish / publish-rev does after its commit
 * (`buildBasePack` of sdk/ts/src/pack.ts): `idx/{gen}/base.{rev}.threadpack` (manifest, slabs table, tombstone sets,
 * coarse, codebook, tree, the `--routing` table) and, when the revision has a tree-grouped table,
 * `idx/{gen}/base.cold.{rev}.threadpack` (the same without the routing table). Objects are read from the overlay
 * directory, then the base directory; the packs are written into the overlay. Never writes to R2.
 *
 *   node bench/cold/make_base_pack.mjs [--base data/cold/g4-base] [--out data/cold/base-overlay] [--gen g4] [--rev 180]
 *        [--routing auto]
 */
import fs from "node:fs";
import path from "node:path";
import { parseArgs, root } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
const BASE = path.resolve(root, args.base ?? "data/cold/g4-base");
const OUT = path.resolve(root, args.out ?? "data/cold/base-overlay");
const GEN = args.gen ?? "g4";
const sdk = await import(path.join(root, "sdk/ts/dist/index.js"));
const find = (key) => [OUT, BASE].map((d) => path.join(d, key)).find((f) => fs.existsSync(f));
/** Read-only union of the overlay and the base directory. */
const store = {
	async get(key, range) {
		const f = find(key);
		if (!f) return null;
		const b = new Uint8Array(fs.readFileSync(f));
		return range ? b.subarray(range.offset, range.offset + range.length) : b;
	},
	async head(key) {
		const f = find(key);
		return f ? { size: fs.statSync(f).size, etag: "local" } : null;
	},
};
const rev = Number(args.rev ?? JSON.parse(new TextDecoder().decode(await store.get("idx/current.json"))).rev);
const rs = await sdk.loadRevState(store, GEN, rev);
const kinds = rs.files.baseCold ? ["base", "cold"] : ["base"];
for (const kind of kinds) {
	const p = await sdk.buildBasePack(store, rs, kind, args.routing ?? "auto");
	const f = path.join(OUT, p.key);
	fs.mkdirSync(path.dirname(f), { recursive: true });
	fs.writeFileSync(f, p.bytes);
	console.log(`wrote ${p.key} (${p.bytes.byteLength} bytes, routing ${p.routing}): ${p.keys.join(", ")}`);
}
