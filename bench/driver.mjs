#!/usr/bin/env node
/**
 * Drives the index build through the deployed Worker's stage endpoints.
 *
 *   node bench/driver.mjs --gen g4 [--profile edge|local] [--start=<stage>] [--k2 4096 --m 3 ...] [--gcBuild 0]
 *
 * Every stage call goes to the orchestrator (THREAD_ORCH, default https://vib-bench.aiuitech.workers.dev) as
 * POST /build/<stage>?slot=<id>; the Worker runs it in the dynamic Worker of that slot (Worker Loader: one isolate
 * with its own 128 MB per slot id). The driver keeps a free list of THREAD_SLOTS (default 8) slot ids 0..N-1 and sends
 * at most one job per slot at a time: concurrent jobs in one slot would share its 128 MB.
 * After a successful publish the intermediate outputs nothing reads any more (build/{gen}/partials/, parts/,
 * pq/partials/) are deleted (`--gcBuild 0` keeps them, e.g. to rerun finalize).
 *
 * A profile sets every layout knob and the query defaults recorded in the manifest;
 * explicit flags override it (see PROFILES below and README.md). The default profile is
 * `edge`: flat 4096 cells, m=3, format 3 (PQ-192 scan + sq8 rerank tiers), defaults
 * nprobe 34 / rerank 150 (g4's 95% point; g4 itself was built with 26 and moved to 34 by publish-rev). `--format 2` builds the g2/g3 layout (sq4 scan tier, no PQ stage).
 *
 * Fine k-means: iteration 1 is exhaustive (kmeans-tree-map, which also writes every
 * vector's cell); iterations 2..fineIters are incremental (kmeans-inc-map: each vector
 * is compared only with the nNeigh neighbours of its previous cell, from a neighbour
 * table recomputed by cell-neighbors after every iteration). `--exactLast` makes the
 * last iteration exhaustive too. assign then picks each vector's top m cells among the
 * neighbours of its final k-means cell (AssignJob.with_candidates).
 *
 * Format 3 adds the `pq` stage after fine k-means: residuals of the first pqSampleObjects
 * sample groups against their final cell (pq-residuals), an initial codebook from the first
 * two residual objects (pq-init), then pqIters Lloyd iterations of pq-train-map (pqMapJobs
 * jobs) + pq-train-reduce. assign then encodes every record's residual with the final
 * codebook too, finalize writes v3 slabs and publish copies the codebook to idx/{gen}/pq.bin.
 *
 * Every stage is idempotent given the same --gen, so a failed run can be resumed with
 * --start=<stage> (and --fineStart <it> to resume k-means at iteration it). Stages:
 *   sample, init-coarse, coarse, init-fine, fine, pq, assign, finalize, publish
 * (--start pq --pqStart <it> resumes PQ training at iteration it; needs pq/codebook/{it-1}.)
 * Centroid tree: `--tree groups,beam` (edge profile: 128,8; `--tree 0` = none) makes publish build a tree over
 * the final fine means (idx/{gen}/tree.0.bin, manifest.tree) for beam routing (`/query?route=beam`) and the
 * pruned primary-cell search of upserts; publish records its recall on ~2000 sample vectors. `--defRoute
 * flat|beam` sets manifest.defaults.route and `--defBeam N` the beam of route=beam queries that give none (default:
 * the tree's beam, which is also the upsert encoder's). To add a tree (or change defaults) on an existing generation:
 *   node bench/driver.mjs --gen g4 --publish-rev [--tree [128,8]] [--treeRecallBeams 4,16] [--defRoute beam]
 * (one publish-rev in a build slot: a new manifest revision; the slabs stay as they are). With --publish-rev a
 * tree is built only when --tree is given (bare `--tree` = the profile's 128,8; `--tree 0` removes it); otherwise
 * the revision keeps the head's tree.
 *
 * Compaction policy (manifest.compaction; the Worker compacts automatically after a write that leaves the head over a
 * threshold, and splits rewritten cells over maxCellEntries): profile defaults (edge: 16 deltas / 32 MB of deltas /
 * 50k tombstones, cells over 3000 entries split, 256 cells per automatic run, 27 s deadline), overridden by `--compactMaxDeltas N
 * --compactMaxDeltaMB N --compactMaxTombstones N --compactMaxCellEntries N (0 = never split) --compactCellsPerRun N
 * --compactDeadlineMs N --compactSplitCellsPerRun N` (automatic runs split only in slices of at most
 * that many cells, default 160) `--compactSlots N` (build slots an automatic run spreads its cells over, default 4;
 * 1 = the writer rewrites them itself, then compactCellsPerRun is the budget) `--compactCellsPerSlot N` (default
 * 160: a run takes up to slots x cellsPerSlot cells); `--compaction 0` = no policy. publish writes it into rev 0; on an existing generation:
 *   node bench/driver.mjs --gen g4 --publish-rev --compaction profile [--compactMaxDeltas 8 ...]
 * (with --publish-rev the policy changes only when --compaction or a --compact* flag is given; flags without
 * --compaction are merged over the head's policy).
 *
 * A g1-style two-level build (exhaustive assignment) is e.g.
 *   --k1 64 --k2 64 --pc 2 --m 3 --perShard 42 --sampleGroups 10 --coarseIters 8
 *   --fineIters 5 --minCount 0 --assignShards 4 --incremental 0 --candidates 0
 */
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const args = {};
for (let i = 2; i < process.argv.length; i++) {
	const m = process.argv[i].match(/^--([^=\s]+)(?:=(.*))?$/);
	if (!m) continue;
	if (m[2] !== undefined) args[m[1]] = m[2];
	else if (process.argv[i + 1] && !process.argv[i + 1].startsWith("--")) args[m[1]] = process.argv[++i];
	else args[m[1]] = "true";
}
// Build slots: each slot id is one dynamic Worker isolate (own 128 MB), so one heavy job per slot keeps every job
// inside the memory limit while running THREAD_SLOTS wide.
const ORCH = process.env.THREAD_ORCH ?? "https://vib-bench.aiuitech.workers.dev";
const POOL = Array.from({ length: Number(process.env.THREAD_SLOTS ?? 8) }, (_, i) => String(i));
const freeSlots = [...POOL];
const waiters = [];
async function acquireSlot() {
	if (freeSlots.length) return freeSlots.pop();
	return new Promise((r) => waiters.push(r));
}
function releaseSlot(s) {
	const next = waiters.shift();
	if (next) next(s);
	else freeSlots.push(s);
}
const TOKEN = fs.readFileSync(path.join(root, "worker/.env.secrets"), "utf8").trim().split("=").slice(1).join("=");
const manifest = JSON.parse(fs.readFileSync(path.join(root, "data/prep/manifest.json"), "utf8"));

/**
 * Index-time profiles. Every layout knob is a build parameter recorded in the manifest;
 * `defaults` are the query parameters the Worker uses when a request omits them.
 *   edge:  m=3, PQ-192 scan tier (208 B/entry) + int8 rerank of the top 150: ~3.7x fewer scan
 *          bytes than the sq4 tier at the g2 operating point (nprobe 26).
 *   local: m=1 (no duplication, smallest index), more probes; for hosts without a connection cap.
 */
const PQ = { pqM: 192, pqIters: 12, pqSampleObjects: 30 };
// Automatic compaction thresholds and the cell split cap (manifest.compaction, see worker COMPACTION_PROFILES).
const COMPACTION = { maxDeltas: 16, maxDeltaBytes: 32 * 1024 * 1024, maxTombstones: 50_000, cellsPerRun: 256, deadlineMs: 27_000 };
const PROFILES = {
	edge: { k1: 1, k2: 4096, m: 3, format: 3, tiers: "pq192+sq8", ...PQ, nNeigh: 64, tree: { groups: 128, beam: 8 }, defaults: { nprobe: 34, rerank: 150, pc: 0 }, compaction: { ...COMPACTION, maxCellEntries: 3000 } },
	local: { k1: 1, k2: 4096, m: 1, format: 3, tiers: "pq192+sq8", ...PQ, nNeigh: 64, tree: null, defaults: { nprobe: 32, rerank: 150, pc: 0 }, compaction: { ...COMPACTION, maxCellEntries: 1000 } },
};
const profileName = args.profile ?? "edge";
const profile = PROFILES[profileName];
if (!profile) throw new Error(`unknown --profile ${profileName} (${Object.keys(PROFILES).join("|")})`);
const num = (name, def) => (args[name] !== undefined ? Number(args[name]) : def);
const bool = (name, def) => (args[name] !== undefined ? !["0", "false", "no"].includes(args[name]) : def);

// --tree groups,beam (or 0/none): centroid tree built at publish / publish-rev.
function parseTree() {
	const raw = args.tree;
	if (raw === undefined || raw === "true") return profile.tree ? { ...profile.tree } : raw === "true" ? { groups: 128, beam: 8 } : null;
	if (["0", "none", "false", "no"].includes(raw)) return null;
	const [groups, beam] = raw.split(",").map(Number);
	if (!Number.isInteger(groups) || groups < 1 || !Number.isInteger(beam ?? 8) || (beam ?? 8) < 1) throw new Error(`--tree must be groups,beam (e.g. 128,8) or 0, got ${raw}`);
	return { groups, beam: beam ?? 8 };
}
const treeOpt = parseTree();
if (treeOpt) {
	treeOpt.iters = num("treeIters", 10);
	treeOpt.seed = num("treeSeed", 7);
	if (args.treeRecallBeams) treeOpt.recallBeams = args.treeRecallBeams.split(",").map(Number);
}
// --compaction 0|profile and --compact* flags (see the header).
const COMPACT_FLAGS = [
	["compactMaxDeltas", "maxDeltas", 1],
	["compactMaxDeltaMB", "maxDeltaBytes", 1024 * 1024],
	["compactMaxTombstones", "maxTombstones", 1],
	["compactMaxCellEntries", "maxCellEntries", 1],
	["compactCellsPerRun", "cellsPerRun", 1],
	["compactDeadlineMs", "deadlineMs", 1],
	["compactSplitCellsPerRun", "splitCellsPerRun", 1],
	["compactSlots", "slots", 1],
	["compactCellsPerSlot", "cellsPerSlot", 1],
];
const compactOverrides = {};
for (const [flag, name, mul] of COMPACT_FLAGS) if (args[flag] !== undefined) compactOverrides[name] = Math.round(Number(args[flag]) * mul);
const compactionOff = ["0", "none", "false", "no"].includes(args.compaction);
/** Policy for publish (rev 0): the profile's with the flags applied, or null. */
const compactionCfg = compactionOff ? null : { ...profile.compaction, ...compactOverrides };
/** publish-rev body field: undefined = unchanged, null = remove, "profile", or the overrides (merged by the Worker). */
const compactionRev = compactionOff ? null : args.compaction === "profile" || args.compaction === "true" ? (Object.keys(compactOverrides).length ? { ...profile.compaction, ...compactOverrides } : "profile") : Object.keys(compactOverrides).length ? compactOverrides : undefined;
const defRoute = args.defRoute;
if (defRoute !== undefined && !["flat", "beam"].includes(defRoute)) throw new Error(`--defRoute must be flat|beam, got ${defRoute}`);
if (defRoute === "beam" && !treeOpt) throw new Error("--defRoute beam needs a tree (--tree groups,beam)");

// format <-> tiers: 3 = PQ-{pqM} scan tier + sq8 rerank tier, 2 = sq4 scan tier + sq8 rerank tier, 1 = sq8 only.
let format = num("format", undefined);
let tiers = args.tiers;
let pqM = num("pqM", undefined);
if (tiers !== undefined) {
	const pq = tiers.match(/^pq(\d+)\+sq8$/);
	const tf = pq ? 3 : { "sq8": 1, "sq4+sq8": 2 }[tiers];
	if (tf === undefined) throw new Error(`unknown --tiers ${tiers} (pq<M>+sq8 | sq4+sq8 | sq8)`);
	format ??= tf;
	if (pq) pqM ??= Number(pq[1]);
}
format ??= profile.format;
pqM ??= profile.pqM;
const tiersFor = (f) => ({ 1: "sq8", 2: "sq4+sq8", 3: `pq${pqM}+sq8` })[f];
if (!tiersFor(format)) throw new Error(`--format must be 1, 2 or 3, got ${format}`);
tiers ??= tiersFor(format);
if (tiersFor(format) !== tiers) throw new Error(`--tiers ${tiers} does not match --format ${format} (expected ${tiersFor(format)})`);
if (format === 3 && manifest.dim % pqM !== 0) throw new Error(`dim ${manifest.dim} is not a multiple of --pqM ${pqM}`);

const cfg = {
	gen: args.gen ?? `g${Date.now().toString(36)}`,
	profile: profileName,
	dim: manifest.dim,
	shards: num("shards", manifest.shards),
	shardSize: manifest.shard,
	vectors: manifest.vectors,
	k1: num("k1", profile.k1),
	k2: num("k2", profile.k2),
	m: num("m", profile.m),
	format,
	tiers,
	// PQ scan tier (format 3): subspaces, Lloyd iterations, training sample (sample objects of ~2000 rows).
	pqM: format === 3 ? pqM : 0,
	pqIters: num("pqIters", profile.pqIters),
	pqSampleObjects: num("pqSampleObjects", profile.pqSampleObjects),
	pqMapJobs: num("pqMapJobs", 8),
	pqStart: num("pqStart", 1),
	nNeigh: num("nNeigh", profile.nNeigh),
	defaults: {
		nprobe: num("defNprobe", profile.defaults.nprobe),
		// A rerank of 150 exists for the coarse PQ scan scores; the sq4 tier needs ~32 (data/quant-experiment.md).
		rerank: num("defRerank", format === 3 ? profile.defaults.rerank : 32),
		pc: num("defPc", profile.defaults.pc),
		...(defRoute ? { route: defRoute } : {}),
		...(args.defBeam !== undefined ? { beam: Number(args.defBeam) } : {}),
	},
	tree: treeOpt,
	compaction: compactionCfg,
	// Build-time coarse groups searched per vector; with k1=1 this is flat anyway.
	probeCoarse: num("pc", 2),
	perShard: num("perShard", 336), // 477 * 336 = 160k sample vectors
	sampleGroups: num("sampleGroups", 80), // ~2000 vectors (12 MB) per sample object
	coarseIters: num("coarseIters", 0), // with k1=1 kmeans-flat is skipped
	fineIters: num("fineIters", 12),
	fineStart: num("fineStart", 1),
	// Iterations 2..fineIters compare each vector with the neighbours of its previous cell only.
	incremental: bool("incremental", true),
	// Make the last fine iteration exhaustive as well.
	exactLast: bool("exactLast", false),
	// assign picks the top m cells among the neighbours of each vector's final k-means cell.
	candidates: bool("candidates", true),
	// Reseed cells with fewer than minCount members on fine iterations 1..fineIters-2 (0 = off).
	minCount: num("minCount", 20),
	reduceCellsPerJob: num("reduceCells", 64),
	mapShardsPerJob: num("mapShards", 10),
	incShardsPerJob: num("incShards", 10),
	neighborJobs: num("neighborJobs", 8),
	// Parts are ~2.3 KB (+ pqM B with a codebook) x m per vector: m=8 one shard (~38 MB) per job, m<=4 two shards.
	assignShardsPerJob: num("assignShards", num("m", profile.m) <= 4 ? 2 : 1),
	// Split each assign shard into this many batches/parts to bound the part buffer (1 = off).
	assignSplit: num("assignSplit", 1),
	finalizeCellsPerJob: num("finalizeCells", 8),
	// Record bytes a finalize job holds in SlabJobs at once (the Worker splits bigger chunks).
	finalizeBudgetMB: num("finalizeBudgetMB", 48),
	concurrency: num("concurrency", POOL.length),
	setCurrent: bool("setCurrent", true),
	// Delete build/{gen}/partials/, parts/ and pq/partials/ after a successful publish.
	gcBuild: bool("gcBuild", true),
	// finalize and publish refuse to rewrite rev 0 of an existing generation (cached as immutable) unless asked to.
	overwrite: bool("overwrite", false),
	start: args.start ?? "sample",
};
// Clamped to [1, k1]: the build jobs (TreeKmeansJob, AssignJob) panic with probe_coarse 0; 0 means "all groups" = k1.
cfg.probeCoarse = cfg.probeCoarse >= 1 ? Math.min(cfg.probeCoarse, cfg.k1) : cfg.k1;
if ((cfg.incremental || cfg.candidates) && cfg.k1 !== 1) throw new Error("incremental k-means / candidate assignment need k1 = 1 (use --incremental 0 --candidates 0)");
if (cfg.candidates && cfg.nNeigh < cfg.m) throw new Error(`--nNeigh ${cfg.nNeigh} must be >= --m ${cfg.m}`);
const totalCells = cfg.k1 * cfg.k2;
const shardRefs = Array.from({ length: cfg.shards }, (_, i) => ({ key: `ingest/${String(i).padStart(5, "0")}.f32`, index: i }));
const G = cfg.gen;
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

/** Errors seen by call(), by status (to spot isolate memory limits: 503 / "exceeded memory"). */
const errorCounts = {};
async function call(stage, body, attempt = 1) {
	const t0 = Date.now();
	const slot = await acquireSlot();
	try {
		const res = await fetch(`${ORCH}/build/${stage}?slot=${slot}`, {
			method: "POST",
			headers: { authorization: `Bearer ${TOKEN}`, "content-type": "application/json" },
			body: JSON.stringify(body),
		});
		const text = await res.text();
		let data;
		try {
			data = JSON.parse(text);
		} catch {
			throw Object.assign(new Error(`HTTP ${res.status} non-JSON: ${text.slice(0, 300)}`), { status: res.status });
		}
		if (!res.ok || data.error) throw Object.assign(new Error(`HTTP ${res.status}: ${data.error ?? text.slice(0, 300)}`), { status: res.status });
		return { ...data.result, _ms: Date.now() - t0, _serverMs: data.ms, _slot: slot };
	} catch (e) {
		const k = `${stage} ${e.status ?? "net"}`;
		errorCounts[k] = (errorCounts[k] ?? 0) + 1;
		if (attempt >= 4) throw e;
		log(`  retry ${stage} (${attempt}) on slot ${slot} after error: ${e.message.replace(/\s+/g, " ").slice(0, 200)}`);
		await new Promise((r) => setTimeout(r, 2000 * attempt));
		return call(stage, body, attempt + 1);
	} finally {
		releaseSlot(slot);
	}
}

async function pmap(items, limit, fn, quiet = false) {
	const out = new Array(items.length);
	let next = 0;
	let done = 0;
	const t0 = Date.now();
	await Promise.all(
		Array.from({ length: Math.min(limit, items.length) }, async () => {
			while (true) {
				const i = next++;
				if (i >= items.length) return;
				out[i] = await fn(items[i], i);
				done++;
				if (!quiet && (done % Math.max(1, Math.floor(items.length / 8)) === 0 || done === items.length)) {
					log(`  ${done}/${items.length} jobs, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
				}
			}
		}),
	);
	return out;
}

const chunk = (arr, n) => Array.from({ length: Math.ceil(arr.length / n) }, (_, i) => arr.slice(i * n, (i + 1) * n));
const pct = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
const order = ["sample", "init-coarse", "coarse", "init-fine", "fine", "pq", "assign", "finalize", "publish", "gc-build"];
if (!order.includes(cfg.start)) throw new Error(`--start must be one of ${order.join(", ")}`);
const shouldRun = (stage) => order.indexOf(stage) >= order.indexOf(cfg.start);

const sampleKeys = Array.from({ length: cfg.sampleGroups }, (_, i) => `build/${G}/sample/${i}.f32`);
const coarseKey = (it) => `build/${G}/coarse/${it}.f32`;
const fineKey = (it) => `build/${G}/fine/${it}.f32`;
const assignPrefix = (it) => `build/${G}/assign/${it}`;
const neighborsKey = (it) => `build/${G}/neighbors/${it}.u32`;
// The sample stage's shard groups (sample object i = the first perShard vectors of each shard of group i).
const sampleGroups = chunk(shardRefs, Math.ceil(shardRefs.length / cfg.sampleGroups));
const pqResidualKey = (i) => `build/${G}/pq/residuals/${i}.f32`;
const pqCodebookKey = (it) => `build/${G}/pq/codebook/${it}.bin`;
const finalCodebookKey = cfg.format === 3 ? pqCodebookKey(cfg.pqIters) : null;
const assignJobs = chunk(shardRefs, cfg.assignShardsPerJob).map((refs, j) => {
	const name = `j${String(j).padStart(4, "0")}`;
	const nParts = cfg.assignSplit > 1 ? refs.length * cfg.assignSplit : 1;
	return {
		name,
		shards: refs.map(({ key, index }) => ({ key, index, firstId: index * cfg.shardSize })),
		parts: cfg.assignSplit > 1 ? Array.from({ length: nParts }, (_, s) => `${name}.${s}`) : [name],
	};
});

const fineStats = [];
let slabs = null;
log("config", JSON.stringify(cfg));
log(`orchestrator ${ORCH}, ${POOL.length} build slots`);
const T0 = Date.now();
const timings = {};
const timed = async (name, fn) => {
	if (!shouldRun(name)) return log(`skip ${name}`);
	log(`== ${name}`);
	const t = Date.now();
	const before = JSON.stringify(errorCounts);
	const r = await fn();
	timings[name] = Date.now() - t;
	log(`== ${name} done in ${(timings[name] / 1000).toFixed(1)}s${JSON.stringify(errorCounts) !== before ? `  (errors so far: ${JSON.stringify(errorCounts)})` : ""}`);
	return r;
};

/** Neighbour table of fine/{it}: sharded cell-neighbors jobs, then a concat to neighbors/{it}.u32. */
async function buildNeighbors(it) {
	const t = Date.now();
	const per = Math.ceil(totalCells / cfg.neighborJobs);
	const ranges = Array.from({ length: Math.ceil(totalCells / per) }, (_, r) => ({ r, first: r * per, count: Math.min(per, totalCells - r * per) }));
	const keys = ranges.map(({ r }) => `build/${G}/neighbors/${it}/${r}.u32`);
	const res = await pmap(
		ranges,
		cfg.concurrency,
		({ first, count }, i) => call("cell-neighbors", { gen: G, fineKey: fineKey(it), k: totalCells, dim: cfg.dim, first, count, n: cfg.nNeigh, outKey: keys[i] }),
		true,
	);
	const cat = await call("concat", { keys, outKey: neighborsKey(it) });
	if (cat.bytes !== totalCells * cfg.nNeigh * 4) throw new Error(`neighbour table ${neighborsKey(it)} is ${cat.bytes} bytes, expected ${totalCells * cfg.nNeigh * 4}`);
	const ms = Date.now() - t;
	log(`  neighbours of fine/${it}: ${ranges.length} jobs (max ${Math.max(...res.map((x) => x._ms))}ms) + concat, ${(ms / 1000).toFixed(1)}s`);
	return ms;
}

// --publish-rev: one new manifest revision of an existing generation (tree and/or defaults), then exit.
if (args["publish-rev"] !== undefined) {
	const defaults = {};
	for (const [flag, name] of [["defNprobe", "nprobe"], ["defRerank", "rerank"], ["defPc", "pc"], ["defBeam", "beam"]]) if (args[flag] !== undefined) defaults[name] = Number(args[flag]);
	if (defRoute) defaults.route = defRoute;
	const body = { gen: G, ...(Object.keys(defaults).length ? { defaults } : {}), ...(args.tree !== undefined ? { tree: treeOpt } : {}), ...(compactionRev !== undefined ? { compaction: compactionRev } : {}) };
	log("publish-rev", JSON.stringify(body));
	const r = await call("publish-rev", body);
	log("  ->", JSON.stringify({ ...r, _slot: undefined }));
	process.exit(0);
}

await timed("sample", async () => {
	const r = await pmap(sampleGroups, cfg.concurrency, (refs, i) =>
		call("sample", { gen: G, shards: refs.map((s) => s.key), perShard: cfg.perShard, dim: cfg.dim, out: String(i) }),
	);
	log("  sample vectors:", r.reduce((a, b) => a + b.vectors, 0));
});

await timed("init-coarse", () => call("init-coarse", { gen: G, sampleKeys, k: cfg.k1, dim: cfg.dim }));

await timed("coarse", async () => {
	if (cfg.coarseIters === 0) log(`  coarseIters=0: using ${coarseKey(0)} as the coarse table (k1=${cfg.k1})`);
	for (let it = 1; it <= cfg.coarseIters; it++) {
		const r = await call("kmeans-flat", {
			gen: G,
			centroidsKey: coarseKey(it - 1),
			inputKeys: sampleKeys,
			k: cfg.k1,
			dim: cfg.dim,
			outKey: coarseKey(it),
		});
		log(`  coarse iter ${it}: ${r.vectors} vectors, ${r._ms}ms`);
	}
});

await timed("init-fine", () =>
	call("init-fine", {
		gen: G,
		coarseKey: coarseKey(cfg.coarseIters),
		sampleKeys,
		k1: cfg.k1,
		k2: cfg.k2,
		dim: cfg.dim,
		seed: 12345,
		outKey: fineKey(0),
	}),
);

/** Iteration `it` is exhaustive (all cells) or incremental (neighbours of the previous cell). */
const isExhaustive = (it) => !cfg.incremental || it === 1 || (cfg.exactLast && it === cfg.fineIters);
// Whether this run built neighbors/{fineIters} (assign needs it with n = nNeigh).
let finalNeighborsBuilt = false;

await timed("fine", async () => {
	const fineStart = shouldRun("fine") && cfg.start === "fine" ? cfg.fineStart : 1;
	for (let it = fineStart; it <= cfg.fineIters; it++) {
		const tIt = Date.now();
		const exhaustive = isExhaustive(it);
		const jobs = chunk(shardRefs, exhaustive ? cfg.mapShardsPerJob : cfg.incShardsPerJob);
		const partialKeys = jobs.map((_, j) => `build/${G}/partials/fine${it}/${j}.acc`);
		// The neighbour table of the means this iteration starts from (normally built at the end of it-1).
		let neighMs = 0;
		if (!exhaustive && it === fineStart) neighMs += await buildNeighbors(it - 1);
		const map = await pmap(jobs, cfg.concurrency, (shards, j) =>
			exhaustive
				? call("kmeans-tree-map", {
						gen: G,
						coarseKey: coarseKey(cfg.coarseIters),
						fineKey: fineKey(it - 1),
						shards,
						k1: cfg.k1,
						k2: cfg.k2,
						dim: cfg.dim,
						probeCoarse: cfg.probeCoarse,
						outKey: partialKeys[j],
						...(cfg.incremental || cfg.candidates ? { assignOutPrefix: assignPrefix(it) } : {}),
					})
				: call("kmeans-inc-map", {
						gen: G,
						fineKey: fineKey(it - 1),
						neighborsKey: neighborsKey(it - 1),
						k: totalCells,
						dim: cfg.dim,
						nNeigh: cfg.nNeigh,
						shards,
						prevAssignPrefix: assignPrefix(it - 1),
						assignOutPrefix: assignPrefix(it),
						outKey: partialKeys[j],
					}),
		);
		const mapMs = Date.now() - tIt - neighMs;
		const changed = exhaustive ? null : map.reduce((a, r) => a + r.changed, 0);
		// Fixed cell chunks over all k1*k2 cells, independent of k1.
		const tRed = Date.now();
		const reduceJobs = Array.from({ length: Math.ceil(totalCells / cfg.reduceCellsPerJob) }, (_, c) => ({
			c,
			firstCell: c * cfg.reduceCellsPerJob,
			nCells: Math.min(cfg.reduceCellsPerJob, totalCells - c * cfg.reduceCellsPerJob),
		}));
		const reseed = cfg.minCount > 0 && it <= cfg.fineIters - 2;
		const red = await pmap(
			reduceJobs,
			cfg.concurrency,
			({ c, firstCell, nCells }) =>
				call("kmeans-reduce", {
					gen: G,
					partialKeys,
					prevFineKey: fineKey(it - 1),
					dim: cfg.dim,
					firstCell,
					nCells,
					outKey: `build/${G}/fine/${it}/${c}.f32`,
					...(reseed ? { reseed: { minCount: cfg.minCount, seed: (it * 100003 + c * 7919) >>> 0 } } : {}),
				}),
			true,
		);
		await call("concat", { keys: reduceJobs.map(({ c }) => `build/${G}/fine/${it}/${c}.f32`), outKey: fineKey(it) });
		const reduceMs = Date.now() - tRed;
		// Neighbour table of the new means: for the next incremental iteration, or for assign after the last one.
		if ((it < cfg.fineIters && !isExhaustive(it + 1)) || (it === cfg.fineIters && cfg.candidates)) {
			neighMs += await buildNeighbors(it);
			if (it === cfg.fineIters) finalNeighborsBuilt = true;
		}
		const empty = red.reduce((a, r) => a + r.empty, 0);
		const reseeded = red.reduce((a, r) => a + (r.reseeded ?? 0), 0);
		const counts = red.flatMap((r) => r.counts ?? []).sort((a, b) => a - b);
		const total = red.reduce((a, r) => a + r.total, 0);
		const q = (p) => pct(counts, p);
		const mean = total / counts.length;
		const cv = Math.sqrt(counts.reduce((a, x) => a + (x - mean) ** 2, 0) / counts.length) / mean;
		const small = cfg.minCount > 0 ? counts.filter((x) => x < cfg.minCount).length : 0;
		const itMs = Date.now() - tIt;
		fineStats.push({
			it,
			mode: exhaustive ? "exhaustive" : "incremental",
			min: counts[0],
			p10: q(0.1),
			p50: q(0.5),
			p90: q(0.9),
			p99: q(0.99),
			max: counts.at(-1),
			cv: +cv.toFixed(3),
			empty,
			small,
			reseeded,
			changed,
			ms: itMs,
			mapMs,
			reduceMs,
			neighMs,
		});
		log(
			`  fine iter ${it} (${exhaustive ? "exhaustive" : "incremental"}): ${total} assigned${changed !== null ? `, ${changed} changed cell (${((100 * changed) / total).toFixed(2)}%)` : ""}, cell size min ${counts[0]} p10 ${q(0.1)} p50 ${q(0.5)} p90 ${q(0.9)} p99 ${q(0.99)} max ${counts.at(-1)} cv ${cv.toFixed(3)}, empty ${empty}, <${cfg.minCount}: ${small}, reseeded ${reseed ? reseeded : "off"}, ${(itMs / 1000).toFixed(0)}s (map ${(mapMs / 1000).toFixed(0)}s, reduce ${(reduceMs / 1000).toFixed(0)}s, neighbours ${(neighMs / 1000).toFixed(1)}s)`,
		);
	}
});

const pqStats = [];
await timed("pq", async () => {
	if (cfg.format !== 3) return log(`  format ${cfg.format}: no PQ codebook`);
	if (cfg.pqSampleObjects > sampleGroups.length) throw new Error(`--pqSampleObjects ${cfg.pqSampleObjects} > ${sampleGroups.length} sample groups`);
	if (cfg.k1 !== 1) throw new Error("the pq stage needs k1 = 1 (flat k-means assignments)");
	const residualKeys = Array.from({ length: cfg.pqSampleObjects }, (_, i) => pqResidualKey(i));
	const pqStart = cfg.start === "pq" ? cfg.pqStart : 1;
	let energy = null;
	if (pqStart === 1) {
		const t = Date.now();
		const r = await pmap(
			sampleGroups.slice(0, cfg.pqSampleObjects),
			cfg.concurrency,
			(shards, i) =>
				call("pq-residuals", {
					gen: G,
					shards,
					perShard: cfg.perShard,
					assignPrefix: assignPrefix(cfg.fineIters),
					fineKey: fineKey(cfg.fineIters),
					dim: cfg.dim,
					k: totalCells,
					outKey: residualKeys[i],
				}),
			true,
		);
		const rows = r.reduce((a, x) => a + x.rows, 0);
		energy = r.reduce((a, x) => a + x.residualEnergy * x.rows, 0) / rows;
		const ms = r.map((x) => x._ms).sort((a, b) => a - b);
		log(`  pq-residuals: ${r.length} objects, ${rows} rows (${((rows * cfg.dim * 4) / 1e6).toFixed(0)} MB), residual energy (mean |r|^2) ${energy.toFixed(5)}, job ms p50 ${pct(ms, 0.5)} max ${ms.at(-1)}, ${((Date.now() - t) / 1000).toFixed(1)}s`);
		pqStats.push({ stage: "residuals", rows, energy, ms: Date.now() - t });
		const init = await call("pq-init", { residualKeys: residualKeys.slice(0, 2), dim: cfg.dim, m: cfg.pqM, seed: 4242, outKey: pqCodebookKey(0) });
		log(`  pq-init: ${init.rows} rows -> codebook ${init.bytes} B [dim, m, sub] = ${JSON.stringify(init.info)}, ${init._ms}ms`);
		pqStats.push({ stage: "init", rows: init.rows, ms: init._ms });
	}
	const mapGroups = chunk(residualKeys, Math.ceil(residualKeys.length / cfg.pqMapJobs));
	for (let it = pqStart; it <= cfg.pqIters; it++) {
		const t = Date.now();
		const partialKeys = mapGroups.map((_, j) => `build/${G}/pq/partials/${it}/${j}.acc`);
		const map = await pmap(
			mapGroups,
			cfg.concurrency,
			(keys, j) => call("pq-train-map", { codebookKey: pqCodebookKey(it - 1), residualKeys: keys, dim: cfg.dim, m: cfg.pqM, outKey: partialKeys[j] }),
			true,
		);
		const mapMs = Date.now() - t;
		const red = await call("pq-train-reduce", { partialKeys, prevCodebookKey: pqCodebookKey(it - 1), dim: cfg.dim, m: cfg.pqM, outKey: pqCodebookKey(it) });
		const ms = Date.now() - t;
		// mean_sq_err is the error of the codebook this iteration started from (pq/codebook/{it-1}).
		const rel = energy ? ` (${((100 * red.mean_sq_err) / energy).toFixed(1)}% of residual energy)` : "";
		log(
			`  pq iter ${it}: mean_sq_err ${red.mean_sq_err.toFixed(6)}${rel} under codebook ${it - 1}, ${red.rows} rows, empty sub-centroids ${red.empty_centroids}/${cfg.pqM * 256}, ${(ms / 1000).toFixed(1)}s (map ${(mapMs / 1000).toFixed(1)}s max job ${Math.max(...map.map((x) => x._ms))}ms, reduce ${red._ms}ms)`,
		);
		pqStats.push({ it, mean_sq_err: red.mean_sq_err, rel: energy ? red.mean_sq_err / energy : null, empty: red.empty_centroids, rows: red.rows, ms, mapMs, reduceMs: red._ms });
	}
});

await timed("assign", async () => {
	if (cfg.candidates && !finalNeighborsBuilt) await buildNeighbors(cfg.fineIters);
	const r = await pmap(assignJobs, cfg.concurrency, (job) =>
		call("assign", {
			gen: G,
			coarseKey: coarseKey(cfg.coarseIters),
			fineKey: fineKey(cfg.fineIters),
			shards: job.shards,
			k1: cfg.k1,
			k2: cfg.k2,
			dim: cfg.dim,
			probeCoarse: cfg.probeCoarse,
			m: cfg.m,
			job: job.name,
			...(cfg.assignSplit > 1 ? { split: cfg.assignSplit } : {}),
			...(cfg.candidates ? { neighborsKey: neighborsKey(cfg.fineIters), nNeigh: cfg.nNeigh, prevAssignPrefix: assignPrefix(cfg.fineIters) } : {}),
			...(finalCodebookKey ? { codebookKey: finalCodebookKey } : {}),
		}),
	);
	if (finalCodebookKey && r.some((x) => x.pqM !== cfg.pqM)) throw new Error(`assign jobs report pq_m ${[...new Set(r.map((x) => x.pqM))]} != ${cfg.pqM}`);
	const records = r.reduce((a, b) => a + b.records, 0);
	const partBytes = r.reduce((a, b) => a + b.partBytes, 0);
	const ms = r.map((x) => x._ms).sort((a, b) => a - b);
	log(
		`  assigned ${records} records (${(records / cfg.vectors).toFixed(2)}x, ${r[0]?.mode}), ${r.reduce((a, b) => a + (b.parts ?? 1), 0)} parts ${(partBytes / 1e9).toFixed(2)} GB, job ms p50 ${pct(ms, 0.5)} p90 ${pct(ms, 0.9)} max ${ms.at(-1)}`,
	);
});

await timed("finalize", async () => {
	const jobs = [];
	for (let g = 0; g < cfg.k1; g++) {
		for (const cells of chunk(Array.from({ length: cfg.k2 }, (_, j) => j), cfg.finalizeCellsPerJob)) jobs.push({ g, cells });
	}
	const r = await pmap(jobs, cfg.concurrency, ({ g, cells }) =>
		call("finalize", {
			gen: G,
			g,
			k2: cfg.k2,
			dim: cfg.dim,
			partKeys: assignJobs.flatMap((j) => j.parts.map((p) => `build/${G}/parts/${g}/${p}.part`)),
			fineKey: fineKey(cfg.fineIters),
			cells,
			format: cfg.format,
			budgetBytes: cfg.finalizeBudgetMB * 1024 * 1024,
			overwrite: cfg.overwrite,
		}),
	);
	slabs = r.flatMap((x) => x.results);
	const bytes = slabs.reduce((a, s) => a + s.bytes, 0);
	const scanBytes = slabs.reduce((a, s) => a + (s.scanLen || s.bytes), 0);
	const counts = slabs.map((s) => s.count).sort((a, b) => a - b);
	const entries = counts.reduce((a, b) => a + b, 0);
	const ms = r.map((x) => x._ms).sort((a, b) => a - b);
	log(
		`  ${slabs.length} slabs (format ${cfg.format}), ${entries} entries, ${(bytes / 1e9).toFixed(2)} GB total (scan tier ${(scanBytes / 1e9).toFixed(2)} GB, ${(scanBytes / entries).toFixed(0)} B/entry; total ${(bytes / entries).toFixed(0)} B/entry), entries/slab min ${counts[0]} p10 ${pct(counts, 0.1)} p50 ${pct(counts, 0.5)} p90 ${pct(counts, 0.9)} p99 ${pct(counts, 0.99)} max ${counts.at(-1)}`,
	);
	log(
		`  finalize jobs: ms p50 ${pct(ms, 0.5)} p90 ${pct(ms, 0.9)} max ${ms.at(-1)}; span reads/job ${(r.reduce((a, x) => a + x.spanReads, 0) / r.length).toFixed(0)}, dir ms avg ${(r.reduce((a, x) => a + x.dirMs, 0) / r.length).toFixed(0)}, span ms avg ${(r.reduce((a, x) => a + x.spanMs, 0) / r.length).toFixed(0)}, jobs split into batches ${r.filter((x) => x.batches > 1).length}`,
	);
	fs.writeFileSync(path.join(root, `data/slabs-${G}.json`), JSON.stringify(slabs));
});

if (!slabs && shouldRun("publish")) {
	const f = path.join(root, `data/slabs-${G}.json`);
	if (fs.existsSync(f)) {
		slabs = JSON.parse(fs.readFileSync(f, "utf8"));
		log(`  loaded ${slabs.length} finalize results from ${f}`);
	}
}
if (shouldRun("publish") && (!slabs || slabs.length !== totalCells)) throw new Error(`publish needs ${totalCells} finalize results, have ${slabs?.length ?? 0}`);

await timed("publish", () =>
	call("publish", {
		gen: G,
		coarseKey: coarseKey(cfg.coarseIters),
		fineKey: fineKey(cfg.fineIters),
		profile: cfg.profile,
		k1: cfg.k1,
		k2: cfg.k2,
		dim: cfg.dim,
		m: cfg.m,
		format: cfg.format,
		tiers: cfg.tiers,
		nNeigh: cfg.nNeigh,
		defaults: cfg.defaults,
		probeCoarse: cfg.probeCoarse,
		vectors: cfg.vectors,
		setCurrent: cfg.setCurrent,
		...(cfg.tree ? { tree: cfg.tree } : {}),
		...(cfg.compaction ? { compaction: cfg.compaction } : {}),
		overwrite: cfg.overwrite,
		slabs: slabs.map(({ cell, count, bytes, scanLen, rerankOff, pqM }) => ({ cell, count, bytes, scanLen, rerankOff, pqM })),
		...(finalCodebookKey ? { codebookKey: finalCodebookKey } : {}),
		notes: { cfg, timings, fineStats, pqStats },
	}).then((m) => log("  manifest", JSON.stringify({ ...m, notes: undefined }))),
);

await timed("gc-build", async () => {
	if (!cfg.gcBuild) return log("  --gcBuild 0: keeping build intermediates");
	for (const prefix of [`build/${G}/partials/`, `build/${G}/parts/`, `build/${G}/pq/partials/`]) {
		const r = await call("delete-prefix", { prefix, dryRun: false });
		log(`  ${prefix}: deleted ${r.deleted} objects (${(r.bytes / 1e9).toFixed(2)} GB), ${r.remaining} remain`);
	}
});

log(`all done in ${((Date.now() - T0) / 1000).toFixed(0)}s`, JSON.stringify(timings), Object.keys(errorCounts).length ? `errors ${JSON.stringify(errorCounts)}` : "no errors");
