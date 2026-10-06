/**
 * Flat word-window chunking of a text before it is embedded and upserted: `chunkText` splits a text into windows
 * of `words` words (default 200) that share `overlap` words with the previous window (default 0), each a verbatim
 * slice of the original text. Chunking is a client-side transform (the index never sees the parent text), so the
 * Python SDK (`chunk_text`) and the CLI (`thread chunk`) implement the same rule; the three are checked against
 * test/fixtures/chunks.json.
 *
 * A word is a maximal run of characters that are not Unicode White_Space (what Rust's `char::is_whitespace` tests:
 * `\t \n \v \f \r`, space, U+0085, U+00A0, U+1680, U+2000..U+200A, U+2028, U+2029, U+202F, U+205F, U+3000; not
 * U+FEFF, which JS's `\s` would include). Chunk `i` holds words `[i * (words - overlap), i * (words - overlap) +
 * words)`; the last chunk is whatever remains and no chunk starts once the previous one reached the end. A text
 * with no words is one empty chunk, so a keyed document always writes at least one chunk. `start` / `end` are
 * code-point offsets of the chunk in the text (not UTF-16 indices; whitespace inside a chunk is kept, whitespace
 * around it is not).
 *
 * A chunked upsert (`index.upsert({ ..., chunk: true })`) stores chunk `i` of the document keyed `k` under the key
 * `chunkKey(k, i)` = `${k}#${i}`, with the parent's meta and attrs.
 */

export const DEFAULT_CHUNK_WORDS = 200;
export const DEFAULT_CHUNK_OVERLAP = 0;
/** Between a document's key and its chunk index in a chunk's key: `${key}#${i}`. */
export const CHUNK_KEY_SEPARATOR = "#";

/** Flat chunking parameters: `words` per chunk (>= 1) and `overlap` words shared with the previous chunk (0 <= overlap < words). */
export interface ChunkOptions {
	words?: number;
	overlap?: number;
}

/** One chunk: its position among the text's chunks, its text, its code-point span in the text and its word count. */
export interface Chunk {
	index: number;
	text: string;
	start: number;
	end: number;
	words: number;
}

/** Unicode White_Space (spelled out so that every implementation agrees). */
function isSpace(cp: number): boolean {
	if (cp <= 0x20) return cp === 0x20 || (cp >= 0x09 && cp <= 0x0d);
	if (cp < 0x85) return false;
	return cp === 0x85 || cp === 0xa0 || cp === 0x1680 || (cp >= 0x2000 && cp <= 0x200a) || cp === 0x2028 || cp === 0x2029 || cp === 0x202f || cp === 0x205f || cp === 0x3000;
}

/** `true` / `undefined` -> the defaults; a `{ words, overlap }` with the defaults filled in; validated. */
export function resolveChunkOptions(opts?: ChunkOptions | boolean): Required<ChunkOptions> {
	if (opts === undefined || opts === true) return { words: DEFAULT_CHUNK_WORDS, overlap: DEFAULT_CHUNK_OVERLAP };
	if (opts === false || typeof opts !== "object" || opts === null) throw new RangeError("chunk must be true or { words, overlap }");
	const words = opts.words ?? DEFAULT_CHUNK_WORDS;
	const overlap = opts.overlap ?? DEFAULT_CHUNK_OVERLAP;
	if (!Number.isInteger(words)) throw new RangeError(`chunk words must be an integer, got ${words}`);
	if (!Number.isInteger(overlap)) throw new RangeError(`chunk overlap must be an integer, got ${overlap}`);
	if (words < 1) throw new RangeError(`chunk words must be >= 1, got ${words}`);
	if (overlap < 0 || overlap >= words) throw new RangeError(`chunk overlap must be in [0, words), got ${overlap} for ${words} words`);
	return { words, overlap };
}

/**
 * Splits `text` into flat windows of `words` words sharing `overlap` words with the previous one (see the module
 * docs). Always at least one chunk.
 */
export function chunkText(text: string, opts?: ChunkOptions): Chunk[] {
	const { words, overlap } = resolveChunkOptions(opts);
	// Word spans as [utf16 start, utf16 end, code-point start, code-point end].
	const u16s: number[] = [];
	const u16e: number[] = [];
	const cps: number[] = [];
	const cpe: number[] = [];
	let cp = 0;
	let inWord = false;
	for (let i = 0; i < text.length; ) {
		const c = text.codePointAt(i)!;
		const w = c > 0xffff ? 2 : 1;
		if (isSpace(c)) {
			if (inWord) {
				u16e.push(i);
				cpe.push(cp);
				inWord = false;
			}
		} else if (!inWord) {
			u16s.push(i);
			cps.push(cp);
			inWord = true;
		}
		i += w;
		cp++;
	}
	if (inWord) {
		u16e.push(text.length);
		cpe.push(cp);
	}
	const n = u16s.length;
	if (n === 0) return [{ index: 0, text: "", start: 0, end: 0, words: 0 }];
	const stride = words - overlap;
	const out: Chunk[] = [];
	for (let first = 0; ; first += stride) {
		const last = Math.min(first + words, n);
		out.push({ index: out.length, text: text.slice(u16s[first], u16e[last - 1]), start: cps[first], end: cpe[last - 1], words: last - first });
		if (last >= n) return out;
	}
}

/** The key chunk `index` of the document keyed `key` is stored under. */
export function chunkKey(key: string, index: number): string {
	return `${key}${CHUNK_KEY_SEPARATOR}${index}`;
}

/** The chunk index when `key` is `chunkKey(parent, i)` for some `i`, else null. */
export function chunkIndexOf(key: string, parent: string): number | null {
	const head = parent + CHUNK_KEY_SEPARATOR;
	if (!key.startsWith(head)) return null;
	const tail = key.slice(head.length);
	return /^[0-9]+$/.test(tail) ? Number(tail) : null;
}
