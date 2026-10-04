#!/usr/bin/env node
/**
 * Experiments against vib-loader-spike. Usage:
 *   node test.mjs mem   [--slots 8] [--mb 90] [--kind js|wasm] [--holdMs 4000] [--same 0] [--gen <id>]
 *        --same 1 sends every request to slot 0 (one dynamic Worker id) instead of slot i
 *   node test.mjs local [--n 8] [--mb 90]          control: the same allocation inside the orchestrator
 *   node test.mjs r2    [--key idx/current.json]
 *   node test.mjs build [--slots 4] [--jobs 4] [--shards 10] [--gen <id>] [--cpuMs N]
 *        runs `jobs` kmeans-tree-map jobs over g4 inputs, at most `slots` at a time, one per slot
 *   node test.mjs pool  [--n 1] [--shards 10]       same job on vib-bench-0..n-1 (pool baseline; not run in the spike,
 *                                                   to avoid colliding with other users of the pool)
 *   node test.mjs cold  [--n 3]                     cold vs warm load cost of the build bundle
 *   node test.mjs cleanup                           deletes build/spike/ objects (orchestrator /cleanup)
 */
import fs from "node:fs";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const URL_BASE = process.env.SPIKE_URL ?? "https://vib-loader-spike.aiuitech.workers.dev";
const TOKEN = fs.readFileSync(path.join(here, ".env.secrets"), "utf8").trim().split("=").slice(1).join("=");
const [cmd = "mem", ...rest] = process.argv.slice(2);
const args = {};
for (let i = 0; i < rest.length; i += 2) args[rest[i].replace(/^--/, "")] = rest[i + 1];
const num = (k, d) => (args[k] !== undefined ? Number(args[k]) : d);
const gen = args.gen ?? `t${Date.now().toString(36)}`;
const H = { authorization: `Bearer ${TOKEN}`, "content-type": "application/json" };

async function post(p, body) {
	const t0 = Date.now();
	try {
		const res = await fetch(`${URL_BASE}${p}`, { method: "POST", headers: H, body: JSON.stringify(body) });
		const text = await res.text();
		let data;
		try {
			data = JSON.parse(text);
		} catch {
			data = { raw: text.slice(0, 300) };
		}
		const hdr = Object.fromEntries([...res.headers].filter(([k]) => k.startsWith("x-")));
		return { status: res.status, ms: Date.now() - t0, hdr, data };
	} catch (e) {
		return { status: "net", ms: Date.now() - t0, data: { error: String(e) } };
	}
}

if (cmd === "mem") {
	const slots = num("slots", 8);
	const same = num("same", 0);
	const body = { mb: num("mb", 90), holdMs: num("holdMs", 4000), kind: args.kind ?? "js" };
	console.log(`mem: ${slots} concurrent, ${body.mb} MB ${body.kind}, hold ${body.holdMs} ms, ${same ? "all on slot 0" : "one slot each"}, gen ${gen}`);
	const t0 = Date.now();
	const rs = await Promise.all(Array.from({ length: slots }, (_, i) => post(`/run?slot=${same ? 0 : i}&gen=${gen}`, body)));
	rs.forEach((r, i) =>
		console.log(
			`  req ${i}: ${r.status} ${r.ms} ms cold=${r.hdr?.["x-orch-cold"]} orch=${r.hdr?.["x-orch-iso"]} ` +
				(r.data.ok ? `iso=${r.data.iso} inflight=${r.data.inflightAtStart} max=${r.data.maxInflight} served=${r.data.served} mb=${r.data.mbAllocated}` : `ERROR ${JSON.stringify(r.data).slice(0, 300)}`),
		),
	);
	const ok = rs.filter((r) => r.data.ok).length;
	const isos = new Set(rs.filter((r) => r.data.ok).map((r) => r.data.iso));
	console.log(`  => ${ok}/${slots} ok, ${isos.size} distinct dynamic isolates, wall ${Date.now() - t0} ms`);
} else if (cmd === "local") {
	const n = num("n", 8);
	const body = { mb: num("mb", 90), holdMs: num("holdMs", 4000) };
	const rs = await Promise.all(Array.from({ length: n }, () => post(`/run-local`, body)));
	rs.forEach((r, i) => console.log(`  req ${i}: ${r.status} ${r.ms} ms ${JSON.stringify(r.data).slice(0, 200)}`));
	console.log(`  => ${rs.filter((r) => r.data.ok).length}/${n} ok, ${new Set(rs.map((r) => r.data.orchIso)).size} orchestrator isolates`);
} else if (cmd === "r2") {
	const res = await fetch(`${URL_BASE}/r2test?slot=0&gen=${gen}&key=${encodeURIComponent(args.key ?? "idx/current.json")}`, { headers: H });
	console.log(res.status, JSON.stringify(await res.json(), null, 2));
} else if (cmd === "build") {
	const slots = num("slots", 4);
	const jobs = num("jobs", slots);
	const shardsPerJob = num("shards", 10);
	const free = Array.from({ length: slots }, (_, i) => i);
	const waiters = [];
	const acquire = () => (free.length ? Promise.resolve(free.shift()) : new Promise((r) => waiters.push(r)));
	const release = (s) => (waiters.length ? waiters.shift()(s) : free.push(s));
	const q = args.cpuMs ? `&cpuMs=${args.cpuMs}` : "";
	console.log(`build: ${jobs} kmeans-tree-map jobs x ${shardsPerJob} shards over ${slots} slots, gen ${gen}`);
	const t0 = Date.now();
	const rs = await Promise.all(
		Array.from({ length: jobs }, async (_, j) => {
			const slot = await acquire();
			try {
				const shards = Array.from({ length: shardsPerJob }, (_, s) => {
					const index = j * shardsPerJob + s;
					return { key: `ingest/${String(index).padStart(5, "0")}.f32`, index };
				});
				const r = await post(`/build/kmeans-tree-map?slot=${slot}&gen=${gen}${q}`, {
					gen: "g4",
					coarseKey: "build/g4/coarse/0.f32",
					fineKey: "build/g4/fine/12.f32",
					shards,
					k1: 1,
					k2: 4096,
					dim: 1536,
					probeCoarse: 1,
					outKey: `build/spike/partial-${j}.acc`,
				});
				return { j, slot, ...r };
			} finally {
				release(slot);
			}
		}),
	);
	for (const r of rs)
		console.log(
			`  job ${r.j} slot ${r.slot}: ${r.status} client ${r.ms} ms, stage ${r.data.ms} ms, cold=${r.hdr?.["x-orch-cold"]} orch=${r.hdr?.["x-orch-iso"]}/${r.hdr?.["x-orch-inflight"]} dynIso=${r.hdr?.["x-dyn-iso"]} inflight=${r.hdr?.["x-dyn-inflight"]} ` +
				(r.data.ok ? `vectors=${r.data.result.vectors} bytes=${r.data.result.bytes}` : `ERROR ${JSON.stringify(r.data).slice(0, 400)}`),
		);
	const ok = rs.filter((r) => r.data.ok);
	console.log(`  => ${ok.length}/${jobs} ok, wall ${Date.now() - t0} ms, mean stage ${Math.round(ok.reduce((a, r) => a + r.data.ms, 0) / Math.max(1, ok.length))} ms`);
} else if (cmd === "pool") {
	// Same job on one pool Worker (vib-bench-N), for a timing baseline.
	const shardsPerJob = num("shards", 10);
	const n = num("n", 1);
	const t0 = Date.now();
	const rs = await Promise.all(
		Array.from({ length: n }, async (_, j) => {
			const shards = Array.from({ length: shardsPerJob }, (_, s) => ({ key: `ingest/${String(j * shardsPerJob + s).padStart(5, "0")}.f32`, index: j * shardsPerJob + s }));
			const tj = Date.now();
			const res = await fetch(`https://vib-bench-${j}.aiuitech.workers.dev/build/kmeans-tree-map`, {
				method: "POST",
				headers: H,
				body: JSON.stringify({ gen: "g4", coarseKey: "build/g4/coarse/0.f32", fineKey: "build/g4/fine/12.f32", shards, k1: 1, k2: 4096, dim: 1536, probeCoarse: 1, outKey: `build/spike/pool-partial-${j}.acc` }),
			});
			const d = await res.json().catch(() => ({}));
			return { j, status: res.status, ms: Date.now() - tj, d };
		}),
	);
	for (const r of rs) console.log(`  pool job ${r.j}: ${r.status} client ${r.ms} ms stage ${r.d.ms} ms ${r.d.ok ? `vectors=${r.d.result.vectors}` : JSON.stringify(r.d).slice(0, 300)}`);
	console.log(`  => wall ${Date.now() - t0} ms`);
} else if (cmd === "cold") {
	// Cold vs warm load of the build bundle: a cheap stage (delete-prefix dry run = one R2 list) on fresh ids.
	const n = num("n", 3);
	for (let i = 0; i < n; i++) {
		const g = `${gen}-${i}`;
		for (const label of ["cold", "warm", "warm"]) {
			const r = await post(`/build/delete-prefix?slot=0&gen=${g}`, { prefix: "build/smoke/" });
			console.log(`  ${g} ${label}: ${r.status} client ${r.ms} ms, stage ${r.data.ms} ms, orch ${r.hdr?.["x-orch-ms"]} ms, cold=${r.hdr?.["x-orch-cold"]} dynIso=${r.hdr?.["x-dyn-iso"]}`);
		}
	}
} else if (cmd === "cleanup") {
	console.log(await post(`/cleanup`, {}));
}
