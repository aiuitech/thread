#!/usr/bin/env node
/**
 * A fake S3 endpoint for deterministic cold-start measurement: a read-only, caching read-through proxy in front of
 * the real R2 S3 API, with an injected network model and exact request/byte accounting.
 *
 *   node bench/cold/fake_s3.mjs [--port 9555] [--rtt-ms 25] [--mbps 200] [--jitter-ms 0] [--conn-ms 2*rtt]
 *        [--link-mbps 0] [--log data/cold/proxy.jsonl] [--cache-dir data/cold/objects] [--overlay DIR]
 *        [--offline] [--env sdk/ts/examples/basic/.env.secrets]
 *
 * S3 subset (what thread-sdk's S3BlobStore / S3BlobStore.py send), path style `/{bucket}/{key}`:
 *   GET with no Range or ONE `bytes=a-b` / `bytes=a-` / `bytes=-n` range (206 + Content-Range; a range past the end
 *   is clamped; one starting past the end, or `bytes=-0`, is a 416 InvalidRange; several ranges or bad syntax are a
 *   400 InvalidArgument and `bytes=b-a` with b > a a 400 "range must be positive", byte for byte what R2 answers;
 *   AWS S3 instead ignores a multi-range and sends the whole object), HEAD, ListObjectsV2
 *   (`?list-type=2&prefix=&max-keys=&continuation-token=`), 404 NoSuchKey XML (R2's error bodies verbatim). Any bucket name is accepted and maps to the upstream bucket. Any Authorization header is
 *   accepted without verification (requests without one are served too; the log records `signed`).
 *   PUT / POST / DELETE are refused with 403: the proxy never writes anywhere but its own disk cache.
 *
 * Objects: `--overlay DIR/{key}` first (a place to drop new object formats without touching R2), then the disk
 * cache `--cache-dir/{key}`, else the WHOLE object is fetched once from the upstream R2 S3 endpoint (signed with
 * sdk/ts's S3BlobStore from dist/, GET/HEAD/List only) and stored in the cache. `idx/current.json` is cached like
 * everything else, so the proxy pins the revision it first saw (delete the cached file to re-pin). `--offline`
 * never contacts the upstream (missing objects are 404).
 *
 * Network model, per request (admin endpoints are exempt): the response starts after `rtt-ms` (+ uniform
 * [0, jitter-ms]) plus `conn-ms` when it is the first request on a new TCP connection (TCP + TLS handshakes; the
 * proxy speaks plain HTTP, so this is where the handshake cost of HTTPS to S3 is modelled; default 2 x rtt); the
 * body then streams at `mbps` per response, and, with `--link-mbps`, all bodies share one link of that capacity.
 *
 * Accounting: every request is appended to `--log` as JSONL `{t, method, key, range, status, bytes, ms, conn, src,
 * signed}` (`bytes` = body bytes sent, `conn` = new connection, `src` = overlay | cache | upstream | -).
 *   GET  /__stats   {requests, bytes, newConnections, upstreamFetches, upstreamBytes, byMethod, byStatus,
 *                    keys: [{key, requests, bytes}], since, config}   (since the last reset)
 *   POST /__reset   zero the counters (returns the stats before the reset)
 *   GET  /__config  the network model;  POST /__config {rttMs, mbps, jitterMs, connMs, linkMbps} changes it live
 */
import fs from "node:fs";
import fsp from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");

function parseArgs(argv) {
	const out = {};
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (!a.startsWith("--")) continue;
		const k = a.slice(2);
		const v = argv[i + 1];
		if (v === undefined || v.startsWith("--")) out[k] = "true";
		else (out[k] = v), i++;
	}
	return out;
}

const args = parseArgs(process.argv.slice(2));
const num = (v, d) => (v === undefined ? d : Number(v));
const config = {
	rttMs: num(args["rtt-ms"], 25),
	mbps: num(args.mbps, 200),
	jitterMs: num(args["jitter-ms"], 0),
	connMs: undefined,
	linkMbps: num(args["link-mbps"], 0),
};
config.connMs = num(args["conn-ms"], 2 * config.rttMs);
const PORT = num(args.port, 9555);
const HOST = args.host ?? "127.0.0.1";
const CACHE_DIR = path.resolve(root, args["cache-dir"] ?? "data/cold/objects");
const OVERLAY = args.overlay ? path.resolve(args.overlay) : null;
const OFFLINE = args.offline === "true";
const LOG = args.log ? path.resolve(args.log) : null;
const CHUNK = 64 * 1024;

// ---------- upstream (read-only) ----------

let upstream = null;
if (!OFFLINE) {
	const envFile = path.resolve(root, args.env ?? "sdk/ts/examples/basic/.env.secrets");
	const env = Object.fromEntries(
		fs
			.readFileSync(envFile, "utf8")
			.split("\n")
			.filter((l) => l.includes("=") && !l.trim().startsWith("#"))
			.map((l) => [l.slice(0, l.indexOf("=")).trim(), l.slice(l.indexOf("=") + 1).trim()]),
	);
	const { S3BlobStore } = await import(path.join(root, "sdk/ts/dist/s3.js"));
	const s3 = new S3BlobStore({ endpoint: env.S3_ENDPOINT, bucket: env.S3_BUCKET, region: env.S3_REGION ?? "auto", accessKeyId: env.S3_ACCESS_KEY_ID, secretAccessKey: env.S3_SECRET_ACCESS_KEY, retries: 2 });
	// Only these three are ever called: the proxy must never write to the bucket.
	upstream = { get: (k) => s3.get(k), head: (k) => s3.head(k), list: (p) => s3.list(p), bucket: env.S3_BUCKET };
}

// ---------- objects ----------

function keyPath(dir, key) {
	const parts = key.split("/");
	if (!key || key.startsWith("/") || parts.some((p) => p === "" || p === "." || p === "..")) return null;
	return path.join(dir, ...parts);
}

async function statFile(p) {
	try {
		const st = await fsp.stat(p);
		return st.isFile() ? st : null;
	} catch {
		return null;
	}
}

const fetching = new Map();
const counters = newCounters();

/** The local file holding `key` (overlay, then cache, then a whole-object fetch from upstream), or null (404). */
async function locate(key) {
	if (OVERLAY) {
		const p = keyPath(OVERLAY, key);
		const st = p && (await statFile(p));
		if (st) return { file: p, st, src: "overlay" };
	}
	const p = keyPath(CACHE_DIR, key);
	if (!p) return null;
	let st = await statFile(p);
	if (st) return { file: p, st, src: "cache" };
	if (!upstream) return null;
	let pending = fetching.get(key);
	if (!pending) {
		pending = (async () => {
			const b = await upstream.get(key);
			if (!b) return false;
			counters.upstreamFetches++;
			counters.upstreamBytes += b.byteLength;
			await fsp.mkdir(path.dirname(p), { recursive: true });
			const tmp = `${p}.tmp-${process.pid}-${Math.random().toString(36).slice(2)}`;
			await fsp.writeFile(tmp, b);
			await fsp.rename(tmp, p);
			return true;
		})();
		fetching.set(key, pending);
		pending.finally(() => fetching.delete(key)).catch(() => {});
	}
	if (!(await pending)) return null;
	st = await statFile(p);
	return st ? { file: p, st, src: "upstream" } : null;
}

async function walk(dir, prefix) {
	const out = [];
	const head = prefix.includes("/") ? prefix.slice(0, prefix.lastIndexOf("/")) : "";
	const start = head ? keyPath(dir, head) : dir;
	if (!start) return out;
	const rec = async (d, rel) => {
		let ents;
		try {
			ents = await fsp.readdir(d, { withFileTypes: true });
		} catch {
			return;
		}
		for (const e of ents) {
			const k = rel ? `${rel}/${e.name}` : e.name;
			if (e.isDirectory()) {
				if ((k + "/").startsWith(prefix) || prefix.startsWith(k + "/")) await rec(path.join(d, e.name), k);
			} else if (k.startsWith(prefix) && !e.name.includes(".tmp-")) out.push(k);
		}
	};
	await rec(start, head);
	return out;
}

const listings = new Map();
/** Every key under `prefix`: upstream (cached per proxy run) or the local cache when offline, plus the overlay. */
async function listKeys(prefix) {
	const keys = new Set();
	if (upstream) {
		let p = listings.get(prefix);
		if (!p) listings.set(prefix, (p = upstream.list(prefix)));
		for (const k of await p) keys.add(k);
	} else for (const k of await walk(CACHE_DIR, prefix)) keys.add(k);
	if (OVERLAY) for (const k of await walk(OVERLAY, prefix)) keys.add(k);
	return [...keys].sort();
}

// ---------- network model ----------

const sleep = (ms) => (ms > 0 ? new Promise((r) => setTimeout(r, ms)) : Promise.resolve());
const now = () => performance.now();
let linkFreeAt = 0;

/** Streams `[start, end]` of `file` to `res`, paced at `mbps` per response and `linkMbps` across responses. */
async function sendBody(res, file, start, end) {
	const total = end - start + 1;
	if (total <= 0) return 0;
	const fh = await fsp.open(file, "r");
	let sent = 0;
	try {
		const t0 = now();
		const perMs = config.mbps > 0 ? (config.mbps * 1e6) / 8 / 1000 : Infinity; // bytes per ms
		const linkPerMs = config.linkMbps > 0 ? (config.linkMbps * 1e6) / 8 / 1000 : Infinity;
		while (sent < total) {
			const n = Math.min(CHUNK, total - sent);
			const buf = Buffer.allocUnsafe(n);
			const { bytesRead } = await fh.read(buf, 0, n, start + sent);
			if (bytesRead !== n) throw new Error(`short read of ${file}`);
			let due = t0 + (sent + n) / perMs;
			if (linkPerMs !== Infinity) {
				const begin = Math.max(now(), linkFreeAt);
				linkFreeAt = begin + n / linkPerMs;
				due = Math.max(due, linkFreeAt);
			}
			await sleep(due - now());
			if (!res.write(buf)) await new Promise((r) => res.once("drain", r));
			sent += n;
		}
	} finally {
		await fh.close();
	}
	return sent;
}

// ---------- S3 responses ----------

const xmlEsc = (s) => String(s).replace(/[<>&'"]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", "'": "&apos;", '"': "&quot;" })[c]);
const errorXml = (code, msg) => `<?xml version="1.0" encoding="UTF-8"?><Error><Code>${code}</Code><Message>${xmlEsc(msg)}</Message></Error>`;
const BAD_RANGE_FORMAT = "range must be in format 'bytes=start-end', 'bytes=start-' or 'bytes=-suffix'.'";
const etagOf = (st) => `"${st.size.toString(16)}-${Math.floor(st.mtimeMs).toString(16)}"`;

/**
 * `{start, end}` (inclusive) for one satisfiable range, null without a Range header, else `{status, code, msg}`:
 * the error R2 answers (checked by verify_proxy.mjs).
 */
function parseRange(h, size) {
	if (h === undefined) return null;
	const m = /^bytes=(\d*)-(\d*)$/.exec(h.trim());
	if (!m || (m[1] === "" && m[2] === "")) return { status: 400, code: "InvalidArgument", msg: BAD_RANGE_FORMAT };
	if (m[1] === "") {
		const n = Number(m[2]);
		if (n === 0) return { status: 416, code: "InvalidRange", msg: "The requested range is not satisfiable" };
		return { start: Math.max(0, size - n), end: size - 1 };
	}
	const start = Number(m[1]);
	if (m[2] !== "" && Number(m[2]) < start) return { status: 400, code: "InvalidArgument", msg: "range must be positive." };
	if (start >= size) return { status: 416, code: "InvalidRange", msg: "The requested range is not satisfiable" };
	return { start, end: m[2] === "" ? size - 1 : Math.min(Number(m[2]), size - 1) };
}

function newCounters() {
	return { requests: 0, bytes: 0, newConnections: 0, upstreamFetches: 0, upstreamBytes: 0, byMethod: {}, byStatus: {}, keys: new Map(), since: new Date().toISOString() };
}
function statsJson() {
	return {
		requests: counters.requests,
		bytes: counters.bytes,
		newConnections: counters.newConnections,
		upstreamFetches: counters.upstreamFetches,
		upstreamBytes: counters.upstreamBytes,
		byMethod: counters.byMethod,
		byStatus: counters.byStatus,
		keys: [...counters.keys].map(([key, v]) => ({ key, ...v })).sort((a, b) => (a.key < b.key ? -1 : 1)),
		since: counters.since,
		config,
	};
}

const logStream = LOG ? (fs.mkdirSync(path.dirname(LOG), { recursive: true }), fs.createWriteStream(LOG, { flags: "a" })) : null;
const tStart = now();

function account(entry) {
	counters.requests++;
	counters.bytes += entry.bytes;
	if (entry.conn) counters.newConnections++;
	counters.byMethod[entry.method] = (counters.byMethod[entry.method] ?? 0) + 1;
	counters.byStatus[entry.status] = (counters.byStatus[entry.status] ?? 0) + 1;
	if (entry.key) {
		const k = counters.keys.get(entry.key) ?? { requests: 0, bytes: 0 };
		k.requests++;
		k.bytes += entry.bytes;
		counters.keys.set(entry.key, k);
	}
	logStream?.write(JSON.stringify(entry) + "\n");
}

async function readBody(req) {
	const chunks = [];
	for await (const c of req) chunks.push(c);
	return Buffer.concat(chunks).toString("utf8");
}

async function admin(req, res, url) {
	const send = (code, obj) => {
		res.writeHead(code, { "content-type": "application/json" });
		res.end(JSON.stringify(obj));
	};
	if (url.pathname === "/__stats" && req.method === "GET") return send(200, statsJson());
	if (url.pathname === "/__reset" && req.method === "POST") {
		const before = statsJson();
		Object.assign(counters, newCounters());
		return send(200, before);
	}
	if (url.pathname === "/__config") {
		if (req.method === "POST") {
			const body = JSON.parse((await readBody(req)) || "{}");
			for (const k of ["rttMs", "mbps", "jitterMs", "connMs", "linkMbps"]) if (body[k] !== undefined) config[k] = Number(body[k]);
		}
		return send(200, config);
	}
	return send(404, { error: "unknown admin endpoint" });
}

async function handle(req, res) {
	const url = new URL(req.url, "http://proxy");
	if (url.pathname.startsWith("/__")) return admin(req, res, url);
	const isNewConn = req.socket.__fresh === true;
	req.socket.__fresh = false;
	const t0 = now();
	const segs = url.pathname.split("/").slice(1);
	const bucket = decodeURIComponent(segs[0] ?? "");
	const key = segs.slice(1).map(decodeURIComponent).join("/");
	const entry = { t: Math.round(now() - tStart), method: req.method, bucket, key, range: req.headers.range ?? null, status: 0, bytes: 0, ms: 0, conn: isNewConn, src: "-", signed: /^AWS4-HMAC-SHA256 /.test(req.headers.authorization ?? "") };
	const delay = config.rttMs + (config.jitterMs > 0 ? Math.random() * config.jitterMs : 0) + (isNewConn ? config.connMs : 0);
	const reqId = `fake${counters.requests.toString(16)}`;
	const finish = () => {
		entry.ms = Math.round((now() - t0) * 10) / 10;
		account(entry);
	};
	const fail = async (status, code, msg) => {
		await sleep(delay - (now() - t0));
		const body = req.method === "HEAD" ? "" : errorXml(code, msg);
		res.writeHead(status, { "content-type": "application/xml", "content-length": Buffer.byteLength(body), "x-amz-request-id": reqId });
		res.end(body);
		entry.status = status;
		entry.bytes = Buffer.byteLength(body);
		finish();
	};
	try {
		if (req.method !== "GET" && req.method !== "HEAD") {
			await readBody(req);
			return fail(403, "AccessDenied", "fake_s3 is read-only (it never writes to the upstream bucket)");
		}
		if (!bucket) return fail(400, "InvalidRequest", "path-style /{bucket}/{key} expected");
		if (!key) {
			if (req.method !== "GET") return fail(405, "MethodNotAllowed", "HEAD of a bucket is not supported");
			// ListObjectsV2
			const prefix = url.searchParams.get("prefix") ?? "";
			const maxKeys = Math.max(1, Math.min(1000, Number(url.searchParams.get("max-keys") ?? 1000)));
			const token = url.searchParams.get("continuation-token");
			const after = token ? Buffer.from(token, "base64url").toString("utf8") : (url.searchParams.get("start-after") ?? "");
			const all = (await listKeys(prefix)).filter((k) => k > after);
			const page = all.slice(0, maxKeys);
			const truncated = all.length > maxKeys;
			const sizes = await Promise.all(page.map(async (k) => (await statFile(keyPath(OVERLAY ?? CACHE_DIR, k) ?? "")) ?? (await statFile(keyPath(CACHE_DIR, k) ?? ""))));
			const body =
				`<?xml version="1.0" encoding="UTF-8"?>\n<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>${xmlEsc(bucket)}</Name><Prefix>${xmlEsc(prefix)}</Prefix><KeyCount>${page.length}</KeyCount><MaxKeys>${maxKeys}</MaxKeys><IsTruncated>${truncated}</IsTruncated>` +
				(token ? `<ContinuationToken>${xmlEsc(token)}</ContinuationToken>` : "") +
				(truncated ? `<NextContinuationToken>${Buffer.from(page[page.length - 1]).toString("base64url")}</NextContinuationToken>` : "") +
				page.map((k, i) => `<Contents><Key>${xmlEsc(k)}</Key><Size>${sizes[i]?.size ?? 0}</Size></Contents>`).join("") +
				`</ListBucketResult>`;
			await sleep(delay - (now() - t0));
			res.writeHead(200, { "content-type": "application/xml", "content-length": Buffer.byteLength(body), "x-amz-request-id": reqId });
			res.end(body);
			entry.status = 200;
			entry.bytes = Buffer.byteLength(body);
			entry.src = "list";
			return finish();
		}
		const loc = await locate(key);
		if (!loc) return fail(404, "NoSuchKey", "The specified key does not exist.");
		entry.src = loc.src;
		const size = loc.st.size;
		const base = { "content-type": "application/octet-stream", "accept-ranges": "bytes", etag: etagOf(loc.st), "last-modified": loc.st.mtime.toUTCString(), "x-amz-request-id": reqId };
		const r = req.method === "GET" ? parseRange(req.headers.range, size) : null;
		if (r && r.status) return fail(r.status, r.code, r.msg);
		await sleep(delay - (now() - t0));
		if (req.method === "HEAD") {
			res.writeHead(200, { ...base, "content-length": size });
			res.end();
			entry.status = 200;
			return finish();
		}
		const start = r ? r.start : 0;
		const end = r ? r.end : size - 1;
		res.writeHead(r ? 206 : 200, { ...base, "content-length": end - start + 1, ...(r ? { "content-range": `bytes ${start}-${end}/${size}` } : {}) });
		entry.status = r ? 206 : 200;
		entry.bytes = await sendBody(res, loc.file, start, end);
		res.end();
		return finish();
	} catch (e) {
		console.error(`fake_s3: ${req.method} ${key}: ${e instanceof Error ? (e.stack ?? e.message) : String(e)}`);
		if (!res.headersSent) return fail(500, "InternalError", e instanceof Error ? e.message : String(e));
		res.destroy();
		entry.status = 500;
		finish();
	}
}

const server = http.createServer((req, res) => {
	handle(req, res).catch((e) => {
		console.error(`fake_s3: ${e instanceof Error ? (e.stack ?? e.message) : String(e)}`);
		res.destroy();
	});
});
server.keepAliveTimeout = 60_000;
server.headersTimeout = 65_000;
server.on("connection", (sock) => {
	sock.__fresh = true;
	sock.setNoDelay(true);
});
server.listen(PORT, HOST, () => {
	console.log(`fake_s3 listening on http://${HOST}:${PORT}  rtt ${config.rttMs} ms  jitter ${config.jitterMs} ms  conn +${config.connMs} ms  ${config.mbps} Mbps/response  link ${config.linkMbps || "unlimited"} Mbps  cache ${CACHE_DIR}${OVERLAY ? `  overlay ${OVERLAY}` : ""}${OFFLINE ? "  OFFLINE" : `  upstream ${upstream.bucket}`}${LOG ? `  log ${LOG}` : ""}`);
});
const stop = () => {
	server.close();
	logStream?.end();
	process.exit(0);
};
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
