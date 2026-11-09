#!/usr/bin/env node
/**
 * Uploads a local index directory to `tmp/blockbench/<name>/` of the vib-bench bucket through the bench Worker.
 *
 *   BENCH_URL=https://thread-blockbench-new.<subdomain>.workers.dev BENCH_TOKEN=... \
 *     node bench/blocks/upload.mjs data/blockbench/hybrid8 hybrid8
 */
import fs from "node:fs";
import path from "node:path";

const [dir, name] = process.argv.slice(2);
if (!dir || !name) throw new Error("usage: upload.mjs <dir> <name>");
const { BENCH_URL, BENCH_TOKEN } = process.env;
const files = [];
const walk = (d) => {
	for (const e of fs.readdirSync(d, { withFileTypes: true })) {
		const p = path.join(d, e.name);
		if (e.isDirectory()) walk(p);
		else if (!e.name.startsWith(".")) files.push(p);
	}
};
walk(dir);
let bytes = 0;
const worker = async () => {
	for (let f = files.shift(); f; f = files.shift()) {
		const key = `tmp/blockbench/${name}/${path.relative(dir, f).split(path.sep).join("/")}`;
		const body = fs.readFileSync(f);
		for (let attempt = 0; ; attempt++) {
			try {
				const r = await fetch(`${BENCH_URL}/put?key=${encodeURIComponent(key)}`, { method: "PUT", body, headers: { authorization: `Bearer ${BENCH_TOKEN}` } });
				if (!r.ok) throw new Error(`${key}: ${r.status} ${await r.text()}`);
				break;
			} catch (e) {
				if (attempt >= 3) throw e;
				await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
			}
		}
		bytes += body.byteLength;
	}
};
await Promise.all(Array.from({ length: Number(process.env.UPLOAD_CONCURRENCY ?? 4) }, worker));
console.log(`uploaded ${bytes} bytes to tmp/blockbench/${name}/`);
