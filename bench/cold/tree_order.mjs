/**
 * Slab order by centroid-tree group, for packs whose probed slabs should sit close together (make_pack.mjs
 * `--order tree`, pack_adjacency.mjs). The tree (`idx/{gen}/tree.{rev}.bin`, "VIBC" v1, see crates/thread-core
 * src/route.rs) clusters the cell means into groups; the wasm exposes no "group of cell" helper, so the CSR member
 * lists are read here. Groups are chained greedily by group-mean similarity, and within a group the cells are
 * chained by nearest neighbour of their means (starting from the cell nearest the previous group's last cell), so
 * consecutive slabs in the order are close in vector space, hence often probed by the same query.
 */

/** Parses "VIBC" v1: `{dim, k, groups, groupMeans (groups x dim, unit), offsets (groups + 1), members}`. */
export function parseTree(b) {
	const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
	if (String.fromCharCode(b[0], b[1], b[2], b[3]) !== "VIBC") throw new Error("not a centroid tree (magic)");
	const version = dv.getUint32(4, true);
	if (version !== 1) throw new Error(`centroid tree version ${version}`);
	const [dim, k, groups, nMembers] = [8, 12, 16, 20].map((o) => dv.getUint32(o, true));
	let o = 32;
	const f32 = (n) => {
		const a = new Float32Array(n);
		for (let i = 0; i < n; i++) a[i] = dv.getFloat32(o + 4 * i, true);
		o += 4 * n;
		return a;
	};
	const u32 = (n) => {
		const a = new Uint32Array(n);
		for (let i = 0; i < n; i++) a[i] = dv.getUint32(o + 4 * i, true);
		o += 4 * n;
		return a;
	};
	const groupMeans = f32(groups * dim);
	f32(groups); // inverse norms
	const offsets = u32(groups + 1);
	const members = u32(nMembers);
	if (o !== b.byteLength) throw new Error(`centroid tree is ${b.byteLength} bytes, parsed ${o}`);
	return { dim, k, groups, groupMeans, offsets, members };
}

const unitRows = (m, dim) => {
	const out = new Float32Array(m.length);
	for (let r = 0; r < m.length / dim; r++) {
		let s = 0;
		for (let j = 0; j < dim; j++) s += m[r * dim + j] ** 2;
		const inv = s > 0 ? 1 / Math.sqrt(s) : 0;
		for (let j = 0; j < dim; j++) out[r * dim + j] = m[r * dim + j] * inv;
	}
	return out;
};
const dotRows = (a, i, b, j, dim) => {
	let s = 0;
	for (let d = 0; d < dim; d++) s += a[i * dim + d] * b[j * dim + d];
	return s;
};

/**
 * Cell group per cell (`Int32Array(k)`): the tree's member lists; a cell the tree does not list (a split after the
 * tree was built) joins the group whose mean is nearest its own.
 */
export function cellGroups(tree, fineMeans, k) {
	const { dim, groups, groupMeans, offsets, members } = tree;
	const g = new Int32Array(k).fill(-1);
	for (let gi = 0; gi < groups; gi++) for (let m = offsets[gi]; m < offsets[gi + 1]; m++) if (members[m] < k && g[members[m]] < 0) g[members[m]] = gi;
	const unit = unitRows(fineMeans, dim);
	for (let c = 0; c < k; c++) {
		if (g[c] >= 0) continue;
		let best = 0;
		let bestS = -Infinity;
		for (let gi = 0; gi < groups; gi++) {
			const s = dotRows(unit, c, groupMeans, gi, dim);
			if (s > bestS) (bestS = s), (best = gi);
		}
		g[c] = best;
	}
	return g;
}

/**
 * Every cell `0..k-1` in tree order: groups chained greedily by group-mean cosine (from group 0), cells inside a
 * group chained by nearest neighbour of their unit means (from the cell nearest the previous group's last cell).
 * Returns `{order, group}` (group per cell).
 */
export function treeOrder(tree, fineMeans, k) {
	const { dim, groups, groupMeans } = tree;
	const group = cellGroups(tree, fineMeans, k);
	const unit = unitRows(fineMeans, dim);
	const byGroup = Array.from({ length: groups }, () => []);
	for (let c = 0; c < k; c++) byGroup[group[c]].push(c);
	// Group chain.
	const gOrder = [0];
	const gUsed = new Uint8Array(groups);
	gUsed[0] = 1;
	while (gOrder.length < groups) {
		const last = gOrder.at(-1);
		let best = -1;
		let bestS = -Infinity;
		for (let gi = 0; gi < groups; gi++) {
			if (gUsed[gi]) continue;
			const s = dotRows(groupMeans, last, groupMeans, gi, dim);
			if (s > bestS) (bestS = s), (best = gi);
		}
		gUsed[best] = 1;
		gOrder.push(best);
	}
	// Cell chain inside each group.
	const order = [];
	for (const gi of gOrder) {
		const left = new Set(byGroup[gi]);
		if (!left.size) continue;
		let cur;
		if (order.length) {
			const prev = order.at(-1);
			let bestS = -Infinity;
			for (const c of left) {
				const s = dotRows(unit, prev, unit, c, dim);
				if (s > bestS) (bestS = s), (cur = c);
			}
		} else cur = Math.min(...left);
		while (cur !== undefined) {
			order.push(cur);
			left.delete(cur);
			let next;
			let bestS = -Infinity;
			for (const c of left) {
				const s = dotRows(unit, cur, unit, c, dim);
				if (s > bestS) (bestS = s), (next = c);
			}
			cur = next;
		}
	}
	if (order.length !== k) throw new Error(`tree order lists ${order.length} of ${k} cells`);
	return { order, group };
}
