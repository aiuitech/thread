/**
 * Builders of the per-generation routing objects a publish writes next to `fine.f32`: the compact routing tables
 * (`fine.f16`, `fine.i8`) and the means in centroid-tree group order (`fine.grouped.{rev}.{elem}`). Pure functions over the wasm kernels (passed in, so the Worker can call them with its own
 * wasm instance) that return the objects to write and the `manifest.routing` entry naming them; the caller writes the
 * objects before the manifest that references them.
 */
import { keys, type RoutingElem, type RoutingTables } from "./wire.js";

/** The wasm functions the builders use (the SDK's `thread.wasm`, or the Worker's own wasm-bindgen module). */
export interface RoutingWasm {
	routing_table_encode(means: Float32Array, dim: number, elem: string): Uint8Array;
	grouped_table_encode(means: Float32Array, dim: number, tree_bytes: Uint8Array, elem: string): Uint8Array;
	grouped_table_info(bytes: Uint8Array): Uint32Array;
}

export interface RoutingBuildOptions {
	/** Key of the f32 means the tables are made from (`files.fine`). */
	fineKey: string;
	/** The means (`k * dim` f32). */
	fine: Float32Array;
	dim: number;
	/** Compact tables to write (default both). */
	tables?: ("f16" | "i8")[];
	/**
	 * The centroid tree (`manifest.tree`) to write the tree-grouped table for (cold routing), with the generation and
	 * the revision that will name it (`fine.grouped.{rev}.{elem}`). Absent: no grouped table.
	 */
	grouped?: { gen: string; rev: number; treeKey: string; tree: Uint8Array; elem?: RoutingElem };
}

export interface RoutingObjects {
	/** Objects to write, in order. */
	objects: [key: string, bytes: Uint8Array][];
	/** The `manifest.routing` entry naming them. */
	routing: RoutingTables;
}

/** Encodes `fine.f16` / `fine.i8` from the f32 means at `fineKey`. */
export function buildRoutingObjects(w: RoutingWasm, o: RoutingBuildOptions): RoutingObjects {
	if (o.dim <= 0 || o.fine.length % o.dim !== 0) throw new Error(`${o.fineKey}: ${o.fine.length} floats is not a whole number of rows of dim ${o.dim}`);
	const objects: [string, Uint8Array][] = [];
	const routing: RoutingTables = { fine: o.fineKey };
	for (const elem of o.tables ?? ["f16", "i8"]) {
		const bytes = w.routing_table_encode(o.fine, o.dim, elem);
		const key = keys.routingTable(o.fineKey, elem);
		objects.push([key, bytes]);
		routing[elem] = { key, bytes: bytes.byteLength };
	}
	if (o.grouped) {
		const g = o.grouped;
		const elem = g.elem ?? "i8";
		const bytes = w.grouped_table_encode(o.fine, o.dim, g.tree, elem);
		const [, , , groups, rows, dataOff, rowBytes] = w.grouped_table_info(bytes);
		const key = keys.groupedTable(g.gen, g.rev, elem);
		objects.push([key, bytes]);
		routing.grouped = { key, tree: g.treeKey, elem, groups, rows, dataOff, rowBytes, bytes: bytes.byteLength };
	}
	return { objects, routing };
}
