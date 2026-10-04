/**
 * Cold routing (`open({coldRouting: "tree"})`): until the full routing table has loaded, a query routes with the
 * centroid tree and the means of just the groups it needs, range-read from the tree-grouped table
 * (`fine.grouped.{rev}.{elem}`, "VIBG"). Group g's rows are `[dataOff + offsets[g] * rowBytes, +n * rowBytes)` with
 * the tree's CSR offsets, so one round trip after the tree (0.8 MB at 4096 cells / 128 groups) brings the beam's
 * groups (beam x 32 cells); for i8 rows the first such read also brings the table's head (the directory and the centre
 * the rows are residuals from, 23 KB at 4096 x 1536). Loaded groups stay loaded for later queries; concurrent queries share in-flight reads.
 * Delta segments need `q·mean` for the cells they hold entries in: their rows are read the same way (by row, from
 * the cell's row in the grouped table), so delta scores match the full table's.
 */
import type { BlobRange, BlobStore, BlobTrace } from "./store.js";
import type * as W from "./wasm/thread.js";
import { ThreadError, type GroupedRoutingRef } from "./wire.js";

/** What a cold query had to read (`stats.coldRouting`). */
export interface ColdStat {
	/** Groups ranked (the beam). */
	beam: number;
	/** Groups and delta cells whose means this query read (0 when earlier queries had loaded them). */
	groups: number;
	cells: number;
	/** Requests and bytes of those reads. */
	reads: number;
	bytes: number;
	/** Groups / cells loaded so far (after this query). */
	loadedGroups: number;
	loadedCells: number;
}

interface Run {
	first: number;
	n: number;
}

/** Merges ascending row runs that touch. */
function coalesce(runs: Run[]): Run[] {
	runs.sort((a, b) => a.first - b.first);
	const out: Run[] = [];
	for (const r of runs) {
		const last = out.at(-1);
		if (last && last.first + last.n === r.first) last.n += r.n;
		else out.push({ ...r });
	}
	return out;
}

/** The cold router of one revision's base and the reads that feed it. */
export class ColdRouting {
	/** Cells of the generation (rows of the full table). */
	readonly k: number;
	/** In-flight or finished loads by `g{group}` / `c{cell}`. */
	private readonly pending = new Map<string, Promise<void>>();
	private readonly deltaCells = new Map<string, Uint32Array>();

	constructor(
		private readonly w: typeof W,
		private readonly store: BlobStore,
		readonly router: W.ColdRouter,
		readonly ref: GroupedRoutingRef,
		/** Default beam (groups ranked per query). */
		readonly beam: number,
	) {
		this.k = router.k();
	}

	private async readRuns(runs: Run[], trace: BlobTrace, stat: ColdStat): Promise<void> {
		const rb = this.ref.rowBytes;
		const ranges: BlobRange[] = runs.map((r) => ({ offset: this.ref.dataOff + r.first * rb, length: r.n * rb }));
		// i8 rows are residuals from the centre in the table's head: the first read brings it along.
		const head = this.router.needs_head();
		if (head) ranges.unshift({ offset: 0, length: this.ref.dataOff });
		let got: Uint8Array[] | null = null;
		if (ranges.length > 1 && this.store.getRanges) {
			got = await this.store.getRanges(this.ref.key, ranges, { trace });
			if (got) stat.reads++;
		}
		if (!got) {
			got = await Promise.all(
				ranges.map(async (r) => {
					const b = await this.store.get(this.ref.key, r, { trace });
					if (!b) throw new ThreadError(`missing ${this.ref.key}`, 404);
					stat.reads++;
					return b;
				}),
			);
		}
		if (head) {
			const h = got.shift()!;
			stat.bytes += h.byteLength;
			if (this.router.needs_head()) this.router.set_head(h);
		}
		runs.forEach((r, i) => {
			if (got![i].byteLength !== r.n * rb) throw new ThreadError(`${this.ref.key}: rows ${r.first}..${r.first + r.n} came back as ${got![i].byteLength} bytes`);
			this.router.add_rows(r.first, got![i]);
			stat.bytes += got![i].byteLength;
		});
	}

	/** Starts (or joins) the loads of `ids` (`g`/`c` prefixed), reading the new ones as `runs`. */
	private async load(ids: string[], runsOf: (fresh: string[]) => Run[], trace: BlobTrace, stat: ColdStat): Promise<number> {
		const waits: Promise<void>[] = [];
		const fresh: string[] = [];
		for (const id of ids) {
			const p = this.pending.get(id);
			if (p) waits.push(p);
			else fresh.push(id);
		}
		if (fresh.length) {
			const p = this.readRuns(coalesce(runsOf(fresh)), trace, stat);
			for (const id of fresh) this.pending.set(id, p);
			p.catch(() => {
				for (const id of fresh) if (this.pending.get(id) === p) this.pending.delete(id);
			});
			waits.push(p);
		}
		await Promise.all(waits);
		return fresh.length;
	}

	/** Loads the groups among the `beam` best for unit query `q` that are not loaded yet. */
	async ensureGroups(q: Float32Array, beam: number, trace: BlobTrace, stat: ColdStat): Promise<void> {
		const missing = Array.from(this.router.missing_groups(q, beam));
		if (missing.length)
			stat.groups += await this.load(
				missing.map((g) => `g${g}`),
				(fresh) =>
					fresh.map((id) => {
						const [first, n] = this.router.group_rows(Number(id.slice(1)));
						return { first, n };
					}),
				trace,
				stat,
			);
	}

	/** Loads the means of `cells` that are not loaded yet (by row of the grouped table). */
	async ensureCells(cells: Uint32Array, trace: BlobTrace, stat: ColdStat): Promise<void> {
		const missing = [...new Set(this.router.missing_cells(cells))];
		if (!missing.length) return;
		stat.cells += await this.load(
			missing.map((c) => `c${c}`),
			(fresh) =>
				fresh.map((id) => {
					const row = this.router.cell_row(Number(id.slice(1)));
					if (row < 0) throw new ThreadError(`${this.ref.key}: cell ${id.slice(1)} is in no group of the tree`);
					return { first: row, n: 1 };
				}),
			trace,
			stat,
		);
	}

	/** The cells a delta segment holds entries in (cached by key). */
	cellsOf(key: string, delta: Uint8Array): Uint32Array {
		let cells = this.deltaCells.get(key);
		if (!cells) {
			const pairs = this.w.delta_cells(delta);
			cells = new Uint32Array(pairs.length / 2);
			for (let i = 0; i < cells.length; i++) cells[i] = pairs[2 * i];
			this.deltaCells.set(key, cells);
		}
		return cells;
	}

	/** The `qc` table `scan_delta` reads for a delta segment: `q·mean` at its cells (loading their rows first). */
	async deltaDots(key: string, delta: Uint8Array, q: Float32Array, trace: BlobTrace, stat: ColdStat): Promise<Float32Array> {
		const cells = this.cellsOf(key, delta);
		await this.ensureCells(cells, trace, stat);
		const dots = this.router.cell_dots_for(q, cells);
		const qc = new Float32Array(this.k);
		cells.forEach((c, i) => (qc[c] = dots[i]));
		return qc;
	}

	newStat(beam: number): ColdStat {
		return { beam, groups: 0, cells: 0, reads: 0, bytes: 0, loadedGroups: 0, loadedCells: 0 };
	}

	finishStat(stat: ColdStat): ColdStat {
		stat.loadedGroups = this.router.loaded_groups();
		stat.loadedCells = this.router.loaded_cells();
		return stat;
	}
}
