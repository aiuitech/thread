#!/usr/bin/env node
/**
 * Checks that fake_s3.mjs answers like the real R2 S3 endpoint: the same status, Content-Range, Content-Length and
 * body bytes (sha256) for whole-object GETs, single ranges (closed, open-ended, suffix, past the end, starting past
 * the end), a multi-range request, HEAD, a missing key and a listing. Read-only (GET/HEAD/List).
 *
 *   node bench/cold/verify_proxy.mjs [--proxy http://127.0.0.1:9555] [--bucket vib-bench] [--env sdk/ts/examples/basic/.env.secrets]
 *
 * Exit code 1 when a case differs.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) => (a.startsWith("--") ? [a.slice(2), all[i + 1] && !all[i + 1].startsWith("--") ? all[i + 1] : "true"] : null)).filter(Boolean));
const env = Object.fromEntries(
	fs
		.readFileSync(path.resolve(root, args.env ?? "sdk/ts/examples/basic/.env.secrets"), "utf8")
		.split("\n")
		.filter((l) => l.includes("="))
		.map((l) => [l.slice(0, l.indexOf("=")).trim(), l.slice(l.indexOf("=") + 1).trim()]),
);
const { signV4, amzDateOf, awsEncode, EMPTY_SHA256 } = await import(path.join(root, "sdk/ts/dist/s3.js"));
const PROXY = args.proxy ?? "http://127.0.0.1:9555";
const BUCKET = args.bucket ?? env.S3_BUCKET;

async function raw(endpoint, bucket, method, key, headers = {}, query) {
	const u = new URL(endpoint);
	const p = `${u.pathname.replace(/\/$/, "")}/${awsEncode(bucket, false)}${key ? "/" + awsEncode(key, true) : ""}`;
	const signed = await signV4({ method, path: p, query, headers, payloadHash: EMPTY_SHA256, host: u.host, region: "auto", accessKeyId: env.S3_ACCESS_KEY_ID, secretAccessKey: env.S3_SECRET_ACCESS_KEY, amzDate: amzDateOf(Date.now()) });
	const qs = query ? "?" + Object.entries(query).map(([k, v]) => `${awsEncode(k, false)}=${awsEncode(v, false)}`).join("&") : "";
	const r = await fetch(`${u.protocol}//${u.host}${p}${qs}`, { method, headers: signed });
	const body = new Uint8Array(await r.arrayBuffer());
	return { status: r.status, contentRange: r.headers.get("content-range"), contentLength: r.headers.get("content-length"), contentType: r.headers.get("content-type"), body, sha: crypto.createHash("sha256").update(body).digest("hex").slice(0, 16) };
}

const r2 = (m, k, h, q) => raw(env.S3_ENDPOINT, env.S3_BUCKET, m, k, h, q);
const px = (m, k, h, q) => raw(PROXY, BUCKET, m, k, h, q);

const cur = JSON.parse(new TextDecoder().decode((await r2("GET", "idx/current.json")).body));
const manifestKey = `idx/${cur.gen}/manifest.${cur.rev}.json`;
const manifest = JSON.parse(new TextDecoder().decode((await r2("GET", manifestKey)).body));
const slabsTable = JSON.parse(new TextDecoder().decode((await r2("GET", manifest.files.slabs)).body));
const cell = 7;
const row = slabsTable[cell];
const slabKey = `idx/${cur.gen}/slab/${String(cell).padStart(5, "0")}${row[4] ? "." + row[4] : ""}`;
const slabSize = row[3];

const cases = [];
for (const [key, size] of [
	["idx/current.json", null],
	[manifestKey, null],
	[manifest.files.pq, null],
	[slabKey, slabSize],
]) {
	const s = size ?? (await r2("HEAD", key)).contentLength * 1;
	cases.push([key, "GET whole", {}]);
	cases.push([key, "GET bytes=0-9", { range: "bytes=0-9" }]);
	cases.push([key, `GET bytes=${Math.floor(s / 2)}- (open-ended)`, { range: `bytes=${Math.floor(s / 2)}-` }]);
	cases.push([key, "GET bytes=-7 (suffix)", { range: "bytes=-7" }]);
	cases.push([key, "GET end past size (clamped)", { range: `bytes=${s - 5}-${s + 100}` }]);
	cases.push([key, "GET start past size (416)", { range: `bytes=${s + 10}-${s + 20}` }]);
	if (size) cases.push([key, `GET scan prefix bytes=0-${row[0] - 1}`, { range: `bytes=0-${row[0] - 1}` }]);
	if (size) cases.push([key, `GET one rerank record`, { range: `bytes=${row[1] + 3 * 1540}-${row[1] + 4 * 1540 - 1}` }]);
	cases.push([key, "HEAD", null]);
}
cases.push(["idx/does-not-exist", "GET missing (404)", {}]);
cases.push([slabKey, "GET two ranges (400)", { range: "bytes=0-9,20-29" }]);
cases.push([slabKey, "GET bytes=20-10 (400)", { range: "bytes=20-10" }]);
cases.push([slabKey, "GET bytes=-0 (416)", { range: "bytes=-0" }]);

let bad = 0;
const rows = [];
for (const [key, label, headers] of cases) {
	const method = headers === null ? "HEAD" : "GET";
	const [a, b] = await Promise.all([r2(method, key, headers ?? {}), px(method, key, headers ?? {})]);
	const informational = label.includes("informational");
	const sameStatus = a.status === b.status;
	const sameRange = (a.contentRange ?? "") === (b.contentRange ?? "");
	// R2 gzips a 404 body (chunked, no Content-Length): compare the decoded body instead of the header there.
	const sameLen = a.contentLength === b.contentLength || (a.contentLength === null && a.body.byteLength === b.body.byteLength);
	const sameBody = a.sha === b.sha;
	const ok = sameStatus && sameRange && sameLen && sameBody;
	if (!ok && !informational) bad++;
	rows.push({ key: key.length > 34 ? "..." + key.slice(-31) : key, case: label, r2: `${a.status} ${a.contentRange ?? "-"} len ${a.contentLength} ${a.status < 400 ? a.sha : ""}`, proxy: `${b.status} ${b.contentRange ?? "-"} len ${b.contentLength} ${b.status < 400 ? b.sha : ""}`, match: ok ? "yes" : informational ? "differs (info)" : "NO" });
}
const lr2 = await r2("GET", "", {}, { "list-type": "2", prefix: `idx/${cur.gen}/manifest.17`, "max-keys": "1000" });
const lpx = await px("GET", "", {}, { "list-type": "2", prefix: `idx/${cur.gen}/manifest.17`, "max-keys": "1000" });
const keysOf = (r) => [...new TextDecoder().decode(r.body).matchAll(/<Key>([^<]*)<\/Key>/g)].map((m) => m[1]).join(",");
const listOk = keysOf(lr2) === keysOf(lpx);
if (!listOk) bad++;
rows.push({ key: `idx/${cur.gen}/manifest.17*`, case: "ListObjectsV2", r2: `${lr2.status} ${keysOf(lr2).split(",").length} keys`, proxy: `${lpx.status} ${keysOf(lpx).split(",").length} keys`, match: listOk ? "yes" : "NO" });
console.table(rows);
console.log(bad ? `${bad} case(s) differ` : "proxy matches R2 on every must-match case");
process.exit(bad ? 1 : 0);
