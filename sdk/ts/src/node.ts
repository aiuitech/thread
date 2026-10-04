/**
 * Node-only stores and helpers (`import ... from "thread-sdk/node"`): an index in a local directory
 * (`FileBlobStore`, the layout Python's `FileBlobStore` reads and writes), opening a directory or a pack file from
 * a path (`storeAt`), and writing a pack to a file (`packIndexToFile`).
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { isPackHeader, PACK_HEADER_BYTES, packIndex, PackedBlobStore, type PackIndexOptions, type PackIndexResult, type PackSink } from "./pack.js";
import type { BlobHead, BlobRange, BlobStore, GetOptions, PutOptions } from "./store.js";
import { ThreadError } from "./wire.js";

const LOCK_DIR = ".thread-lock.d";
const LOCK_STALE_MS = 30_000;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const errno = (e: unknown): string | undefined => (e as { code?: string })?.code;

/** A Uint8Array with its own buffer (Node's pooled Buffers share one). */
const own = (b: Uint8Array): Uint8Array => (b.byteOffset === 0 && b.buffer.byteLength === b.byteLength ? b : new Uint8Array(b.slice()));

/**
 * An index in a local directory (`{root}/idx/current.json`, ...): a synced copy of a bucket, a pack unpacked with
 * `copyObjects`, or a writable local index. Puts are atomic (temp file + rename); create-only puts use a hard link,
 * compare-and-swap puts are serialized across Node processes with a lock directory `{root}/.thread-lock.d`
 * (Python's FileBlobStore uses an advisory lock, so TypeScript and Python writers on the same directory are not
 * serialized against each other for that one step; the manifest create-only put is atomic for both).
 */
export class FileBlobStore implements BlobStore {
	readonly root: string;

	constructor(root: string) {
		this.root = path.resolve(root);
	}

	path(key: string): string {
		const parts = key.split("/");
		if (!key || key.startsWith("/") || parts.slice(0, -1).some((p) => p === "" || p === "." || p === "..") || parts[parts.length - 1] === "." || parts[parts.length - 1] === "..") throw new ThreadError(`bad object key ${JSON.stringify(key)}`, 400);
		return path.join(this.root, ...parts);
	}

	private static etag(st: { ino: bigint; mtimeNs: bigint; size: bigint }): string {
		// Every put renames a fresh temp file into place, so the inode changes with each write.
		return `${st.ino.toString(16)}-${st.mtimeNs.toString(16)}-${st.size.toString(16)}`;
	}

	async get(key: string, range?: BlobRange, opts?: GetOptions): Promise<Uint8Array | null> {
		const file = this.path(key);
		let b: Uint8Array;
		try {
			if (!range) b = own(await fs.readFile(file));
			else {
				const fh = await fs.open(file, "r");
				try {
					const size = (await fh.stat()).size;
					if (range.offset < 0 || range.length < 0 || range.offset + range.length > size) throw new ThreadError(`${key} [${range.offset}, +${range.length}) is outside the object (${size} bytes)`);
					b = await readAt(fh, range);
				} finally {
					await fh.close();
				}
			}
		} catch (e) {
			if (errno(e) === "ENOENT") return null;
			throw e;
		}
		if (opts?.trace) {
			opts.trace.reads++;
			opts.trace.bytes += b.byteLength;
		}
		return b;
	}

	async getRanges(key: string, ranges: BlobRange[], opts?: GetOptions): Promise<Uint8Array[] | null> {
		let fh: fs.FileHandle;
		try {
			fh = await fs.open(this.path(key), "r");
		} catch (e) {
			if (errno(e) === "ENOENT") return null;
			throw e;
		}
		try {
			const size = (await fh.stat()).size;
			const out: Uint8Array[] = [];
			for (const r of ranges) {
				if (r.offset < 0 || r.length < 0 || r.offset + r.length > size) throw new ThreadError(`${key} [${r.offset}, +${r.length}) is outside the object (${size} bytes)`);
				out.push(await readAt(fh, r));
			}
			if (opts?.trace) {
				opts.trace.reads++;
				opts.trace.bytes += out.reduce((a, b) => a + b.byteLength, 0);
			}
			return out;
		} finally {
			await fh.close();
		}
	}

	async head(key: string): Promise<BlobHead | null> {
		try {
			const st = await fs.stat(this.path(key), { bigint: true });
			return { size: Number(st.size), etag: FileBlobStore.etag(st) };
		} catch (e) {
			if (errno(e) === "ENOENT") return null;
			throw e;
		}
	}

	async put(key: string, data: Uint8Array | string, opts: PutOptions = {}): Promise<BlobHead | null> {
		const file = this.path(key);
		const dir = path.dirname(file);
		await fs.mkdir(dir, { recursive: true });
		const tmp = path.join(dir, `.tmp-${process.pid}-${Math.random().toString(36).slice(2, 10)}`);
		await fs.writeFile(tmp, typeof data === "string" ? data : data);
		let keepTmp = true;
		try {
			if (opts.ifAbsent) {
				try {
					await fs.link(tmp, file); // atomic create-only
				} catch (e) {
					const c = errno(e);
					if (c === "EEXIST") return null;
					if (c !== "EPERM" && c !== "ENOTSUP" && c !== "EXDEV" && c !== "EOPNOTSUPP") throw e; // no hard links here: fall back to the lock
					await this.locked(async () => {
						if (await this.head(key)) return false;
						await fs.rename(tmp, file);
						keepTmp = false;
						return true;
					});
					if (keepTmp) return null;
				}
			} else if (opts.ifMatch !== undefined) {
				const ok = await this.locked(async () => {
					const cur = await this.head(key);
					if (!cur || cur.etag !== opts.ifMatch) return false;
					await fs.rename(tmp, file);
					keepTmp = false;
					return true;
				});
				if (!ok) return null;
			} else {
				await fs.rename(tmp, file);
				keepTmp = false;
			}
		} finally {
			if (keepTmp) await fs.unlink(tmp).catch(() => {});
		}
		const h = await this.head(key);
		if (!h) throw new ThreadError(`${key} vanished after put`);
		return h;
	}

	/** Runs `fn` holding `{root}/.thread-lock.d` (mkdir is atomic); a lock older than 30 s is taken over. */
	private async locked<T>(fn: () => Promise<T>): Promise<T> {
		const lock = path.join(this.root, LOCK_DIR);
		for (let i = 0; ; i++) {
			try {
				await fs.mkdir(lock);
				break;
			} catch (e) {
				if (errno(e) !== "EEXIST") throw e;
				const st = await fs.stat(lock).catch(() => null);
				if (st && Date.now() - st.mtimeMs > LOCK_STALE_MS) {
					await fs.rmdir(lock).catch(() => {});
					continue;
				}
				if (i > 2000) throw new ThreadError(`could not take ${lock} in time`, 409);
				await sleep(2 + Math.random() * 8);
			}
		}
		try {
			return await fn();
		} finally {
			await fs.rmdir(lock).catch(() => {});
		}
	}

	async list(prefix: string): Promise<string[]> {
		// Walk the deepest directory the prefix names and filter by the full prefix.
		const cut = prefix.lastIndexOf("/");
		const head = cut >= 0 ? prefix.slice(0, cut) : "";
		const base = head ? path.join(this.root, ...head.split("/")) : this.root;
		const out: string[] = [];
		const walk = async (dir: string, rel: string): Promise<void> => {
			let ents: import("node:fs").Dirent[];
			try {
				ents = await fs.readdir(dir, { withFileTypes: true });
			} catch (e) {
				if (errno(e) === "ENOENT" || errno(e) === "ENOTDIR") return;
				throw e;
			}
			for (const e of ents) {
				const k = rel + e.name;
				if (e.isDirectory()) {
					if (e.name === LOCK_DIR && !rel) continue;
					if ((k + "/").startsWith(prefix) || prefix.startsWith(k + "/")) await walk(path.join(dir, e.name), k + "/");
				} else if (k.startsWith(prefix) && !e.name.startsWith(".tmp-") && k !== ".thread-lock") out.push(k);
			}
		};
		await walk(base, head ? head + "/" : "");
		return out.sort();
	}
}

async function readAt(fh: fs.FileHandle, r: BlobRange): Promise<Uint8Array> {
	const out = new Uint8Array(r.length);
	let got = 0;
	while (got < r.length) {
		const { bytesRead } = await fh.read(out, got, r.length - got, r.offset + got);
		if (bytesRead === 0) throw new ThreadError(`short read at ${r.offset + got}`);
		got += bytesRead;
	}
	return out;
}

/**
 * The store at a path: a directory holding an index layout (`FileBlobStore`), or a pack file (`PackedBlobStore`
 * over the file's directory). `thread.open({ store: await storeAt("/data/idx.threadpack") })`.
 */
export async function storeAt(p: string): Promise<BlobStore> {
	const st = await fs.stat(p).catch((e: unknown) => {
		if (errno(e) === "ENOENT") throw new ThreadError(`no such file or directory: ${p}`, 404);
		throw e;
	});
	if (st.isDirectory()) return new FileBlobStore(p);
	if (!st.isFile()) throw new ThreadError(`${p} is neither a directory nor a file`, 400);
	const fh = await fs.open(p, "r");
	let head: Uint8Array;
	try {
		head = await readAt(fh, { offset: 0, length: Math.min(PACK_HEADER_BYTES, st.size) });
	} finally {
		await fh.close();
	}
	if (!isPackHeader(head)) throw new ThreadError(`${p} is not a thread pack`, 400);
	return PackedBlobStore.open(new FileBlobStore(path.dirname(path.resolve(p))), path.basename(p));
}

/** A `PackSink` appending to an open file. */
export class FileSink implements PackSink {
	constructor(private readonly fh: fs.FileHandle) {}
	async write(bytes: Uint8Array): Promise<void> {
		let off = 0;
		while (off < bytes.byteLength) off += (await this.fh.write(bytes, off, bytes.byteLength - off)).bytesWritten;
	}
}

/** `packIndex` to a file, written to a temp sibling and renamed into place (the file is never seen half-written). */
export async function packIndexToFile(store: BlobStore, file: string, opts: PackIndexOptions = {}): Promise<PackIndexResult> {
	const dir = path.dirname(path.resolve(file));
	await fs.mkdir(dir, { recursive: true });
	const tmp = path.join(dir, `.tmp-${path.basename(file)}-${process.pid}-${Math.random().toString(36).slice(2, 10)}`);
	const fh = await fs.open(tmp, "w");
	try {
		const r = await packIndex(store, new FileSink(fh), opts);
		await fh.sync().catch(() => {});
		await fh.close();
		await fs.rename(tmp, file);
		return r;
	} catch (e) {
		await fh.close().catch(() => {});
		await fs.unlink(tmp).catch(() => {});
		throw e;
	}
}
