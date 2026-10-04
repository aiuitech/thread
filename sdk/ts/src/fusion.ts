/**
 * Reciprocal Rank Fusion in TypeScript, the same rule as the core (`thread_core::fusion::rrf`, exposed by the
 * wasm as `rrf_fuse`): `score(d) = Σ weight / (k + rank)` with rank from 1, accumulated list by list in float64,
 * ranked by score descending then id ascending, truncated to `topN`. Lists are id arrays best first; their own
 * scores do not matter, only the ranks, which is what makes a cosine list and a BM25 list comparable.
 */

export interface FusedHits {
	ids: Uint32Array;
	scores: Float64Array;
}

export const DEFAULT_RRF_K = 60;

/** Fuses ranked lists (`[weight, ids]`, ids best first) into the top `topN`. */
export function rrf(lists: ReadonlyArray<readonly [number, ArrayLike<number>]>, k: number, topN: number): FusedHits {
	// Insertion order keeps the float64 sums in the same order as the Rust core (list order, then rank order).
	const scores = new Map<number, number>();
	for (const [weight, ids] of lists) {
		for (let i = 0; i < ids.length; i++) {
			const id = ids[i];
			scores.set(id, (scores.get(id) ?? 0) + weight / (k + i + 1));
		}
	}
	const ranked = Array.from(scores, ([id, score]) => ({ id, score }));
	ranked.sort((a, b) => b.score - a.score || a.id - b.id);
	const n = Math.min(topN, ranked.length);
	const ids = new Uint32Array(n);
	const out = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		ids[i] = ranked[i].id;
		out[i] = ranked[i].score;
	}
	return { ids, scores: out };
}
