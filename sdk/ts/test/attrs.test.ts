/** The attribute schema and filter compiler against the fixtures shared with the Python SDK, and the kernel's evaluation of the compiled words. */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { compileFilter, defineAttrs, filterMatches, packAttrs, schemaExtends, unpackAttrs, validateSchema, type AttrFilterExpr, type AttrSchema } from "../src/attrs.js";
import { w } from "./helpers.js";

const fx = JSON.parse(fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), "fixtures/filters.json"), "utf8")) as {
	spec: Parameters<typeof defineAttrs>[0];
	schema: AttrSchema;
	pack: { values: Record<string, unknown> | number; word: number }[];
	packErrors: Record<string, unknown>[];
	filters: { expr: AttrFilterExpr; words: number[] }[];
	filtersRaw: { expr: AttrFilterExpr; words: number[] }[];
	filterErrors: AttrFilterExpr[];
	matches: { expr: AttrFilterExpr; words: number[]; pass: boolean[] }[];
};

describe("attrs schema", () => {
	it("lays fields out in order and validates", () => {
		expect(defineAttrs(fx.spec)).toEqual(fx.schema);
		expect(validateSchema(fx.schema)).toEqual(fx.schema);
		expect(() => defineAttrs({ a: { bits: 20 }, b: { bits: 13 } })).toThrow(/33 bits/);
		expect(() => defineAttrs({ a: { bits: 0 } })).toThrow(/bits/);
		expect(() => defineAttrs({ a: { values: ["x", "x"] } })).toThrow(/distinct/);
		expect(() => defineAttrs({ "bad name": "bool" })).toThrow(/field name/);
		expect(() => validateSchema({ fields: [{ name: "a", type: "number", offset: 0, bits: 8 }, { name: "b", type: "number", offset: 4, bits: 8 }] })).toThrow(/overlaps/);
		expect(() => validateSchema({ fields: [{ name: "a", type: "enum", offset: 0, bits: 1, values: ["x", "y"] }] })).toThrow(/fit/);
		// an enum of 3 values takes 2 bits (0 = unset); one bit holds a single value
		expect(defineAttrs({ e: { values: ["only"] } }).fields[0].bits).toBe(1);
		// growth against the schema in force: existing fields keep their layout (an enum may append values that fit
		// its width), omitted ones are kept, new ones go after them
		const small = defineAttrs({ lang: { values: ["en", "fr"] }, tenant: { bits: 8 } });
		const grown = defineAttrs({ lang: { values: ["en", "fr", "de"] }, extra: "bool" }, small);
		expect(grown.fields.map((f) => [f.name, f.offset, f.bits])).toEqual([
			["lang", 0, 2],
			["tenant", 2, 8],
			["extra", 10, 1],
		]);
		expect(schemaExtends(small, grown)).toBe(true);
		expect(() => defineAttrs({ lang: { values: ["en", "fr", "de", "it"] } }, small)).toThrow(/fit/);
		expect(() => defineAttrs({ lang: { values: ["fr", "en"] } }, small)).toThrow(/layout/);
		expect(() => defineAttrs({ tenant: { bits: 9 } }, small)).toThrow(/layout/);
		expect(() => defineAttrs({ tenant: "bool" }, small)).toThrow(/layout/);
		expect(schemaExtends(fx.schema, defineAttrs({ ...fx.spec, lang: { values: ["fr", "en", "de"] } }))).toBe(false);
		expect(schemaExtends(fx.schema, defineAttrs({ tenant: { bits: 11 } }))).toBe(false);
	});

	it("packs and unpacks values as the fixtures say", () => {
		for (const c of fx.pack) expect(packAttrs(fx.schema, c.values as never)).toBe(c.word);
		for (const bad of fx.packErrors) expect(() => packAttrs(fx.schema, bad as never)).toThrow();
		expect(unpackAttrs(fx.schema, 1040389)).toEqual({ tenant: 5, lang: "fr", public: true, year: 2001 });
		expect(unpackAttrs(fx.schema, 0)).toEqual({ tenant: 0, lang: "", public: false, year: 1970 });
		expect(unpackAttrs(null, 77)).toEqual({ attrs: 77 });
		expect(packAttrs(null, 77)).toBe(77);
		expect(() => packAttrs(null, { tenant: 1 })).toThrow(/no attribute schema/);
		expect(packAttrs(null, { attrs: 9 })).toBe(9);
	});
});

describe("filter compiler", () => {
	it("compiles the shared fixtures to the same words as Python", () => {
		for (const c of fx.filters) expect(Array.from(compileFilter(fx.schema, c.expr)), JSON.stringify(c.expr)).toEqual(c.words);
		for (const c of fx.filtersRaw) expect(Array.from(compileFilter(null, c.expr)), JSON.stringify(c.expr)).toEqual(c.words);
		for (const bad of fx.filterErrors) expect(() => compileFilter(fx.schema, bad), JSON.stringify(bad)).toThrow();
		expect(compileFilter(fx.schema, null).length).toBe(0);
	});

	it("evaluates like the kernel", () => {
		for (const c of fx.matches) {
			const words = compileFilter(fx.schema, c.expr);
			expect(w.filter_check(words)).toBe(words[0]);
			expect(c.words.map((a) => filterMatches(words, a)), JSON.stringify(c.expr)).toEqual(c.pass);
			expect(Array.from(w.filter_matches(words, Uint32Array.from(c.words))).map(Boolean), JSON.stringify(c.expr)).toEqual(c.pass);
		}
		expect(filterMatches(new Uint32Array(0), 123)).toBe(true);
		expect(Array.from(w.filter_matches(new Uint32Array(0), Uint32Array.from([1, 2])))).toEqual([1, 1]);
		expect(() => w.filter_check(Uint32Array.from([1, 0, 9, 0, 0, 0]))).toThrow(/unknown op/);
	});
});
