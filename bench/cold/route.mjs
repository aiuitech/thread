/**
 * Routing without a query: the SDK's own routing over the real routing table (what `executeQuery` does before any
 * slab read: normalize, `cell_dots`, `route_from_dots`, flat as the manifest defaults say), for analyses that must not
 * read slabs (hot_cells.mjs, pack_adjacency.mjs). Reads only the open's objects (current, manifest, slabs table,
 * coarse, fine, pq) from the proxy.
 */
import fs from "node:fs";
import path from "node:path";
import { root } from "./lib.mjs";

export async function openRouting(proxy) {
	const dist = path.join(root, "sdk/ts/dist");
	const sdk = await import(path.join(dist, "index.js"));
	const store = new sdk.S3BlobStore({ endpoint: proxy, bucket: "vib-bench", region: "auto", accessKeyId: "fake", secretAccessKey: "fake" });
	const wasmModule = new WebAssembly.Module(fs.readFileSync(path.join(dist, "wasm/thread_bg.wasm")));
	const thread = await sdk.Thread.init({ wasmModule });
	const W = thread.wasm;
	const t = await sdk.resolveTarget(store);
	const rev = await sdk.loadRevState(store, t.gen, t.rev);
	const base = await sdk.loadBase(store, rev, thread.epoch);
	const defaultNprobe = rev.manifest.defaults?.nprobe ?? 1;
	/** The cells a query probes, in routing order (best first). */
	const route = (vector, nprobe = defaultNprobe) => Array.from(base.searcher.route_from_dots(base.searcher.cell_dots(W.normalize_vector(vector)), nprobe));
	return { sdk, store, thread, rev, gen: t.gen, revNo: t.rev, route, defaultNprobe };
}
