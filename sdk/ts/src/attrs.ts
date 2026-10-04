/**
 * Attributes and filters. Every document carries one 32-bit `attrs` word, stored in its dense entry (slabs and
 * delta segments, inside the scan prefix every query reads anyway) and in its `docs` entry. A generation may
 * declare a schema (`manifest.attrs`) that packs named fields into that word; `packAttrs` turns `{tenant: 5,
 * lang: "en"}` into the word and `compileFilter` turns `{tenant: 5, lang: {$in: ["en", "fr"]}}` into the clause
 * list the kernels apply inside every scan, before top-k insertion (so a filtered-out entry never takes a
 * candidate slot). Without a schema the whole word is one numeric field named `attrs`.
 *
 * Compiled form (`u32` words, `thread-core/src/filter.rs`): `[n, (mask, op, a, b, nValues, values...) x n]`, the
 * clauses ANDed; every clause tests `x = attrs & mask`. The Python SDK compiles the same expressions to the same
 * words (`thread_sdk/attrs.py`); `sdk/ts/test/fixtures/filters.json` holds the shared vectors.
 */
import { ThreadError } from "./wire.js";

/** A field of the attribute schema, as stored in `manifest.attrs.fields`. */
export interface AttrField {
	name: string;
	/** `number`: an unsigned integer of `bits` bits (biased by `min`); `enum`: one of `values`; `bool`: one bit. */
	type: "number" | "enum" | "bool";
	/** Bit offset of the field in the word (assigned by `defineAttrs`). */
	offset: number;
	/** Width in bits (1..32). */
	bits: number;
	/** `number`: the value stored as 0 (default 0); e.g. `min: 1970` for a year. */
	min?: number;
	/** `enum`: the values, stored as 1..n (0 = unset). */
	values?: string[];
}

export interface AttrSchema {
	fields: AttrField[];
}

/** What `defineAttrs` / `index.configureAttrs` take: fields in order, each a width, an enum or a bool. */
export type AttrFieldSpec = { bits: number; min?: number } | { values: string[] } | "bool" | { bool: true };

/** An attribute value per field (unset fields are 0); a plain number is the raw word. */
export type AttrValues = Record<string, number | string | boolean | null | undefined> | number;

/** One field's condition: a bare value means `$eq`. */
export type AttrCondition =
	| number
	| string
	| boolean
	| {
			$eq?: number | string | boolean;
			$ne?: number | string | boolean;
			$in?: (number | string | boolean)[];
			$nin?: (number | string | boolean)[];
			$lt?: number;
			$lte?: number;
			$gt?: number;
			$gte?: number;
			/** Bit tests on a number field (or the raw `attrs` word): some / every / no bit of the mask set. */
			$any?: number;
			$all?: number;
			$none?: number;
	  };

/** A filter: conditions on fields, ANDed. Without a schema the only field is `attrs` (the raw word). */
export type AttrFilterExpr = Record<string, AttrCondition>;

export const OP_EQ = 0;
export const OP_NE = 1;
export const OP_IN = 2;
export const OP_NIN = 3;
export const OP_RANGE = 4;
export const OP_BITS_ANY = 5;
export const OP_BITS_ALL = 6;
export const OP_BITS_NONE = 7;

/** The implicit field of a generation without a schema: the whole word. */
const RAW_FIELD: AttrField = { name: "attrs", type: "number", offset: 0, bits: 32 };

const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);
const u32mask = (bits: number): number => (bits >= 32 ? 0xffffffff : (2 ** bits - 1) >>> 0);

/**
 * Lays out named fields in declaration order into the 32-bit word. With `prev` (the schema in force), fields that
 * already exist keep their offset and width (an enum may append values that fit its width; a number keeps its
 * `bits` and `min`), fields `prev` has that `spec` omits are kept, and new fields go after them: stored words stay
 * valid.
 */
export function defineAttrs(spec: Record<string, AttrFieldSpec>, prev: AttrSchema | null = null): AttrSchema {
	const fields: AttrField[] = prev ? prev.fields.map((f) => ({ ...f, ...(f.values ? { values: [...f.values] } : {}) })) : [];
	let offset = fields.reduce((o, f) => Math.max(o, f.offset + f.bits), 0);
	for (const [name, s] of Object.entries(spec)) {
		if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new ThreadError(`attrs: bad field name ${JSON.stringify(name)}`, 400);
		let f: AttrField;
		if (s === "bool" || (typeof s === "object" && "bool" in s)) f = { name, type: "bool", offset, bits: 1 };
		else if ("values" in s) {
			const values = s.values;
			if (!Array.isArray(values) || !values.length || values.some((v) => typeof v !== "string") || new Set(values).size !== values.length) throw new ThreadError(`attrs: field ${name}: values must be distinct strings`, 400);
			f = { name, type: "enum", offset, bits: Math.max(1, Math.ceil(Math.log2(values.length + 1))), values: [...values] };
		} else {
			if (!isInt(s.bits) || s.bits < 1 || s.bits > 32) throw new ThreadError(`attrs: field ${name}: bits must be 1..32`, 400);
			if (s.min !== undefined && !isInt(s.min)) throw new ThreadError(`attrs: field ${name}: min must be an integer`, 400);
			f = { name, type: "number", offset, bits: s.bits, ...(s.min ? { min: s.min } : {}) };
		}
		const at = fields.findIndex((p) => p.name === name);
		if (at >= 0) {
			// an existing field keeps its layout
			const p = fields[at];
			const same = p.type === f.type && (p.type !== "number" || (p.bits === f.bits && (p.min ?? 0) === (f.min ?? 0)));
			const enumOk = p.type !== "enum" || ((p.values ?? []).every((v, i) => f.values![i] === v) && f.values!.length <= u32mask(p.bits));
			if (!same || !enumOk) throw new ThreadError(`attrs: field ${name} exists with another layout (a schema only adds fields, or enum values that fit the field's ${p.bits} bits)`, 400);
			fields[at] = { ...p, ...(f.values ? { values: f.values } : {}) };
			continue;
		}
		offset += f.bits;
		fields.push(f);
	}
	if (offset > 32) throw new ThreadError(`attrs: the fields need ${offset} bits, the word has 32`, 400);
	return validateSchema({ fields });
}

/** Checks a schema read from a manifest (fields disjoint and within the word, names unique). */
export function validateSchema(schema: AttrSchema): AttrSchema {
	if (!schema || !Array.isArray(schema.fields)) throw new ThreadError("attrs: schema needs a fields array", 400);
	const names = new Set<string>();
	let used = 0;
	for (const f of schema.fields) {
		if (typeof f.name !== "string" || !f.name || names.has(f.name)) throw new ThreadError(`attrs: bad or duplicate field name ${JSON.stringify(f.name)}`, 400);
		names.add(f.name);
		if (!["number", "enum", "bool"].includes(f.type)) throw new ThreadError(`attrs: field ${f.name}: unknown type ${String(f.type)}`, 400);
		if (!isInt(f.offset) || !isInt(f.bits) || f.offset < 0 || f.bits < 1 || f.offset + f.bits > 32) throw new ThreadError(`attrs: field ${f.name}: bad offset/bits`, 400);
		if (f.type === "bool" && f.bits !== 1) throw new ThreadError(`attrs: field ${f.name}: a bool is one bit`, 400);
		if (f.type === "enum" && (!Array.isArray(f.values) || f.values.length > u32mask(f.bits))) throw new ThreadError(`attrs: field ${f.name}: enum values do not fit ${f.bits} bits`, 400);
		const m = (u32mask(f.bits) * 2 ** f.offset) >>> 0;
		if (used & m) throw new ThreadError(`attrs: field ${f.name} overlaps another field`, 400);
		used = (used | m) >>> 0;
	}
	return schema;
}

/** Whether `next` keeps every field of `prev` as it is (an enum may append values): the rule for changing a schema. */
export function schemaExtends(prev: AttrSchema, next: AttrSchema): boolean {
	for (const p of prev.fields) {
		const n = next.fields.find((f) => f.name === p.name);
		if (!n || n.type !== p.type || n.offset !== p.offset || n.bits !== p.bits || (n.min ?? 0) !== (p.min ?? 0)) return false;
		if (p.type === "enum" && (p.values ?? []).some((v, i) => n.values?.[i] !== v)) return false;
	}
	return true;
}

function fieldOf(schema: AttrSchema | null, name: string): AttrField {
	const f = schema?.fields.find((x) => x.name === name);
	if (f) return f;
	if (!schema?.fields.length && name === RAW_FIELD.name) return RAW_FIELD;
	throw new ThreadError(schema ? `attrs: unknown field ${JSON.stringify(name)}` : `attrs: this index has no attribute schema; the only field is "attrs"`, 400);
}

/** A field value as the unsigned integer it is stored as, or null when it cannot be represented. */
function encode(f: AttrField, v: number | string | boolean): number | null {
	if (f.type === "bool") return v === true || v === 1 ? 1 : v === false || v === 0 ? 0 : null;
	if (f.type === "enum") {
		if (typeof v !== "string") return null;
		const i = f.values!.indexOf(v);
		return i < 0 ? null : i + 1;
	}
	if (!isInt(v)) return null;
	const x = v - (f.min ?? 0);
	return x >= 0 && x <= u32mask(f.bits) ? x : null;
}

/** Packs field values into the word (unset fields are 0). A number is taken as the raw word. */
export function packAttrs(schema: AttrSchema | null, values: AttrValues): number {
	if (typeof values === "number") {
		if (!isInt(values) || values < 0 || values > 0xffffffff) throw new ThreadError("attrs: a raw word must be a u32", 400);
		return values;
	}
	if (!values || typeof values !== "object") throw new ThreadError("attrs: expected an object of field values or a u32", 400);
	let word = 0;
	for (const [name, v] of Object.entries(values)) {
		if (v === undefined || v === null) continue;
		const f = fieldOf(schema, name);
		const x = encode(f, v);
		if (x === null) throw new ThreadError(`attrs: ${name}: ${JSON.stringify(v)} is not a valid ${f.type}${f.type === "number" ? ` (${f.bits} bits from ${f.min ?? 0})` : ""}`, 400);
		word = (word | (x * 2 ** f.offset)) >>> 0;
	}
	return word;
}

/** The field values of a word under the schema (a raw `attrs` entry without one). */
export function unpackAttrs(schema: AttrSchema | null, word: number): Record<string, number | string | boolean> {
	if (!schema?.fields.length) return { attrs: word >>> 0 };
	const out: Record<string, number | string | boolean> = {};
	for (const f of schema.fields) {
		const x = Math.floor((word >>> 0) / 2 ** f.offset) & u32mask(f.bits);
		if (f.type === "bool") out[f.name] = x === 1;
		else if (f.type === "enum") out[f.name] = x === 0 ? "" : (f.values![x - 1] ?? "");
		else out[f.name] = x + (f.min ?? 0);
	}
	return out;
}

interface Clause {
	mask: number;
	op: number;
	a: number;
	b: number;
	values: number[];
}

/** A clause no entry matches. */
const NOTHING: Clause = { mask: 0, op: OP_RANGE, a: 1, b: 0, values: [] };

/**
 * Compiles a filter expression against the schema. Conditions on different fields are ANDed; within a field
 * each operator adds a clause (`$gt`/`$gte`/`$lt`/`$lte` combine into one range). A value a field cannot hold
 * makes `$eq` match nothing and `$ne` match everything; such values are dropped from `$in` / `$nin`. Returns the
 * kernel's words (empty for an empty expression).
 */
export function compileFilter(schema: AttrSchema | null, expr: AttrFilterExpr | null | undefined): Uint32Array {
	if (!expr) return new Uint32Array(0);
	if (typeof expr !== "object" || Array.isArray(expr)) throw new ThreadError("filter: expected an object of field conditions", 400);
	const clauses: Clause[] = [];
	for (const [name, cond] of Object.entries(expr)) {
		if (name.startsWith("$")) throw new ThreadError(`filter: unsupported operator ${name} at the top level (conditions are ANDed; nest operators under a field)`, 400);
		const f = fieldOf(schema, name);
		const shift = 2 ** f.offset;
		const mask = (u32mask(f.bits) * shift) >>> 0;
		const c = cond !== null && typeof cond === "object" && !Array.isArray(cond) ? cond : { $eq: cond as number | string | boolean };
		const enc = (v: unknown, op: string): number | null => {
			if (v === null || v === undefined || (typeof v !== "number" && typeof v !== "string" && typeof v !== "boolean")) throw new ThreadError(`filter: ${name} ${op}: bad value ${JSON.stringify(v)}`, 400);
			const x = encode(f, v);
			return x === null ? null : (x * shift) >>> 0;
		};
		let lo = 0;
		let hi = u32mask(f.bits);
		let ranged = false;
		for (const [op, v] of Object.entries(c)) {
			if (v === undefined) continue;
			switch (op) {
				case "$eq": {
					const x = enc(v, op);
					clauses.push(x === null ? NOTHING : { mask, op: OP_EQ, a: x, b: 0, values: [] });
					break;
				}
				case "$ne": {
					const x = enc(v, op);
					if (x !== null) clauses.push({ mask, op: OP_NE, a: x, b: 0, values: [] });
					break;
				}
				case "$in":
				case "$nin": {
					if (!Array.isArray(v)) throw new ThreadError(`filter: ${name} ${op}: expected an array`, 400);
					const xs = [...new Set(v.map((e) => enc(e, op)).filter((x): x is number => x !== null))].sort((p, q) => p - q);
					if (op === "$in") clauses.push(xs.length ? { mask, op: OP_IN, a: 0, b: 0, values: xs } : NOTHING);
					else if (xs.length) clauses.push({ mask, op: OP_NIN, a: 0, b: 0, values: xs });
					break;
				}
				case "$lt":
				case "$lte":
				case "$gt":
				case "$gte": {
					if (f.type !== "number" || !isInt(v)) throw new ThreadError(`filter: ${name} ${op}: needs an integer on a number field`, 400);
					const x = v - (f.min ?? 0); // may be out of the field's range: clamp below
					ranged = true;
					if (op === "$lt") hi = Math.min(hi, x - 1);
					else if (op === "$lte") hi = Math.min(hi, x);
					else if (op === "$gt") lo = Math.max(lo, x + 1);
					else lo = Math.max(lo, x);
					break;
				}
				case "$any":
				case "$all":
				case "$none": {
					if (f.type !== "number" || !isInt(v) || v < 0 || v > 0xffffffff) throw new ThreadError(`filter: ${name} ${op}: needs a u32 bit mask on a number field`, 400);
					const m = ((v & u32mask(f.bits)) * shift) >>> 0;
					if (m === 0) {
						// an empty mask: $any matches nothing, $all and $none everything
						if (op === "$any") clauses.push(NOTHING);
						break;
					}
					clauses.push({ mask: m, op: op === "$any" ? OP_BITS_ANY : op === "$all" ? OP_BITS_ALL : OP_BITS_NONE, a: 0, b: 0, values: [] });
					break;
				}
				default:
					throw new ThreadError(`filter: ${name}: unknown operator ${op}`, 400);
			}
		}
		if (ranged) {
			lo = Math.max(lo, 0);
			hi = Math.min(hi, u32mask(f.bits));
			if (lo > hi) clauses.push(NOTHING);
			else if (lo > 0 || hi < u32mask(f.bits)) clauses.push({ mask, op: OP_RANGE, a: (lo * shift) >>> 0, b: (hi * shift) >>> 0, values: [] });
		}
	}
	return clausesToWords(clauses);
}

export function clausesToWords(clauses: Clause[]): Uint32Array {
	if (!clauses.length) return new Uint32Array(0);
	const out: number[] = [clauses.length];
	for (const c of clauses) out.push(c.mask >>> 0, c.op, c.a >>> 0, c.b >>> 0, c.values.length, ...c.values.map((v) => v >>> 0));
	return Uint32Array.from(out);
}

/** Evaluates compiled words in JS (the same rule as the kernel's `AttrFilter::matches`). */
export function filterMatches(words: Uint32Array | null | undefined, attrs: number): boolean {
	if (!words || !words.length) return true;
	const n = words[0];
	let pos = 1;
	for (let i = 0; i < n; i++) {
		const [mask, op, a, b, nv] = [words[pos], words[pos + 1], words[pos + 2], words[pos + 3], words[pos + 4]];
		pos += 5;
		const values = words.subarray(pos, pos + nv);
		pos += nv;
		const x = (attrs & mask) >>> 0;
		let ok: boolean;
		switch (op) {
			case OP_EQ:
				ok = x === a;
				break;
			case OP_NE:
				ok = x !== a;
				break;
			case OP_IN:
				ok = values.includes(x);
				break;
			case OP_NIN:
				ok = !values.includes(x);
				break;
			case OP_RANGE:
				ok = a <= x && x <= b;
				break;
			case OP_BITS_ANY:
				ok = x !== 0;
				break;
			case OP_BITS_ALL:
				ok = x === mask;
				break;
			case OP_BITS_NONE:
				ok = x === 0;
				break;
			default:
				throw new ThreadError(`filter: unknown op ${op}`, 400);
		}
		if (!ok) return false;
	}
	return true;
}
