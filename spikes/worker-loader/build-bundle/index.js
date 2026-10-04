import wasmModule from "./assets/vib_bg-Ddd4raaf.wasm";
//#region pkg/vib.js
/**
* Assign job: multi-assigns each vector to its `m` nearest fine cells, encodes
* the residual against each cell's mean in both tiers (4-bit MSE-clipped scan
* code + int8 rerank code), and buffers records per coarse group. After
* `set_codebook`, records also carry the residual's PQ codes (needed to
* build format 3 slabs); without it their `pq_m` is 0.
*/
var AssignJob = class AssignJob {
	static __wrap(ptr) {
		const obj = Object.create(AssignJob.prototype);
		obj.__wbg_ptr = ptr;
		Object.defineProperty(obj, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		AssignJobFinalization.register(obj, {
			ptr,
			instance: __wbg_instance_id
		}, obj);
		return obj;
	}
	__destroy_into_raw() {
		const ptr = this.__wbg_ptr;
		this.__wbg_ptr = 0;
		AssignJobFinalization.unregister(this);
		return ptr;
	}
	free() {
		const ptr = this.__destroy_into_raw();
		__wbg_call_guard();
		wasm.__wbg_assignjob_free(ptr, 0);
	}
	/**
	* Vectors have ids `first_id, first_id + 1, ...`. Payloads are empty.
	* @param {Float32Array} vectors
	* @param {number} first_id
	* @returns {number}
	*/
	add_batch(vectors, first_id) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.assignjob_add_batch(this.__wbg_ptr, ptr0, len0, first_id);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		return ret[0] >>> 0;
	}
	/**
	* `with_candidates` jobs only: like `add_batch`, but vector `i`'s top `m`
	* cells are chosen among the neighbours of `prev[i]` (its cell from the
	* last k-means iteration). Returns vectors seen.
	* @param {Float32Array} vectors
	* @param {number} first_id
	* @param {Uint32Array} prev
	* @returns {number}
	*/
	add_batch_prev(vectors, first_id, prev) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		const ptr1 = passArray32ToWasm0(prev, wasm.__wbindgen_malloc);
		const len1 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.assignjob_add_batch_prev(this.__wbg_ptr, ptr0, len0, first_id, ptr1, len1);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		return ret[0] >>> 0;
	}
	/**
	* @returns {number}
	*/
	assigned_records() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.assignjob_assigned_records(this.__wbg_ptr);
		return ret;
	}
	/**
	* @param {number} dim
	* @param {number} k1
	* @param {number} k2
	* @param {Float32Array} coarse_means
	* @param {Float32Array} fine_means
	* @param {number} probe_coarse
	* @param {number} m
	*/
	constructor(dim, k1, k2, coarse_means, fine_means, probe_coarse, m) {
		const ptr0 = passArrayF32ToWasm0(coarse_means, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		const ptr1 = passArrayF32ToWasm0(fine_means, wasm.__wbindgen_malloc);
		const len1 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.assignjob_new(dim, k1, k2, ptr0, len0, ptr1, len1, probe_coarse, m);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		this.__wbg_ptr = ret[0];
		Object.defineProperty(this, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		AssignJobFinalization.register(this, {
			ptr: ret[0],
			instance: __wbg_instance_id
		}, this);
		return this;
	}
	/**
	* Current serialized size of part `g`.
	* @param {number} g
	* @returns {number}
	*/
	part_len(g) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.assignjob_part_len(this.__wbg_ptr, g);
		return ret >>> 0;
	}
	/**
	* PQ codes per record (0 without a codebook).
	* @returns {number}
	*/
	pq_m() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.assignjob_pq_m(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* Makes every following record carry the PQ codes of its residual under
	* this codebook (`pq_init_codebook` / `pq_reduce` bytes; its dim must be the
	* job's dim). Works for jobs from either constructor; call before adding.
	* @param {Uint8Array} codebook_bytes
	*/
	set_codebook(codebook_bytes) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArray8ToWasm0(codebook_bytes, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.assignjob_set_codebook(this.__wbg_ptr, ptr0, len0);
		if (ret[1]) throw takeFromExternrefTable0(ret[0]);
	}
	/**
	* Makes `add_batch` pick each vector's top `m` cells with a pruned search:
	* `tree_bytes` (`build_centroid_tree` over all `k1 * k2` fine means, global
	* cell ids) scores its group means and ranks only the cells of the best
	* `beam` groups; `probe_coarse` is then ignored. With `beam >= groups` the
	* parts equal those of the exhaustive flat search (`probe_coarse = k1`, or
	* `k1 = 1`). `add_batch_prev` is unaffected. Call before adding.
	* @param {Uint8Array} tree_bytes
	* @param {number} beam
	*/
	set_tree(tree_bytes, beam) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArray8ToWasm0(tree_bytes, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.assignjob_set_tree(this.__wbg_ptr, ptr0, len0, beam);
		if (ret[1]) throw takeFromExternrefTable0(ret[0]);
	}
	/**
	* Serializes part `g` and resets its buffer.
	* @param {number} g
	* @returns {Uint8Array}
	*/
	take_part(g) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.assignjob_take_part(this.__wbg_ptr, g);
		var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
		return v1;
	}
	/**
	* Flat (`k1 = 1`: one part group, `k2 = k`) assign job whose
	* `add_batch_prev` picks each vector's top `m` cells among the `n_neigh`
	* neighbours (`neighbors`, a `k x n_neigh` table from `cell_neighbors`)
	* of its previous cell. Requires `n_neigh >= m`. `add_batch` also works
	* (exhaustive flat top-m).
	* @param {number} dim
	* @param {number} k
	* @param {Float32Array} means
	* @param {Uint32Array} neighbors
	* @param {number} n_neigh
	* @param {number} m
	* @returns {AssignJob}
	*/
	static with_candidates(dim, k, means, neighbors, n_neigh, m) {
		const ptr0 = passArrayF32ToWasm0(means, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		const ptr1 = passArray32ToWasm0(neighbors, wasm.__wbindgen_malloc);
		const len1 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.assignjob_with_candidates(dim, k, ptr0, len0, ptr1, len1, n_neigh, m);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		return AssignJob.__wrap(ret[0]);
	}
};
if (Symbol.dispose) AssignJob.prototype[Symbol.dispose] = AssignJob.prototype.free;
/**
* Update job: encodes new vectors into one delta segment ("VIBD"). Holds the
* `k * dim` cell means (+ inverse norms, + the PQ codebook after
* `set_codebook`). Each vector goes to its primary cell (highest cosine over
* all `k` cells) with residual codes against that cell's mean, as an
* `AssignJob` part record (4-bit + int8, + PQ codes with a codebook), so
* compaction moves it into the cell's slab without re-encoding. Vectors
* must be unit-normalized, as for `AssignJob`.
*/
var DeltaJob = class {
	__destroy_into_raw() {
		const ptr = this.__wbg_ptr;
		this.__wbg_ptr = 0;
		DeltaJobFinalization.unregister(this);
		return ptr;
	}
	free() {
		const ptr = this.__destroy_into_raw();
		__wbg_call_guard();
		wasm.__wbg_deltajob_free(ptr, 0);
	}
	/**
	* Adds `n` vectors (`n * dim`) with ids `ids` (`n`) and attrs (`n`, or
	* empty for all 0), no payloads. Returns each vector's primary cell.
	* Nothing is added if the arguments are inconsistent.
	* @param {Float32Array} vectors
	* @param {Uint32Array} ids
	* @param {Uint32Array} attrs
	* @returns {Uint32Array}
	*/
	add(vectors, ids, attrs) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		const ptr1 = passArray32ToWasm0(ids, wasm.__wbindgen_malloc);
		const len1 = WASM_VECTOR_LEN;
		const ptr2 = passArray32ToWasm0(attrs, wasm.__wbindgen_malloc);
		const len2 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.deltajob_add(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
		if (ret[3]) throw takeFromExternrefTable0(ret[2]);
		var v4 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v4;
	}
	/**
	* Like `add`, with payloads: `payloads` is the concatenation of every
	* vector's payload, `payload_lens[i]` the length of vector `i`'s.
	* @param {Float32Array} vectors
	* @param {Uint32Array} ids
	* @param {Uint32Array} attrs
	* @param {Uint8Array} payloads
	* @param {Uint32Array} payload_lens
	* @returns {Uint32Array}
	*/
	add_with_payloads(vectors, ids, attrs, payloads, payload_lens) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		const ptr1 = passArray32ToWasm0(ids, wasm.__wbindgen_malloc);
		const len1 = WASM_VECTOR_LEN;
		const ptr2 = passArray32ToWasm0(attrs, wasm.__wbindgen_malloc);
		const len2 = WASM_VECTOR_LEN;
		const ptr3 = passArray8ToWasm0(payloads, wasm.__wbindgen_malloc);
		const len3 = WASM_VECTOR_LEN;
		const ptr4 = passArray32ToWasm0(payload_lens, wasm.__wbindgen_malloc);
		const len4 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.deltajob_add_with_payloads(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4);
		if (ret[3]) throw takeFromExternrefTable0(ret[2]);
		var v6 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v6;
	}
	/**
	* Serialized size of the segment so far.
	* @returns {number}
	*/
	byte_len() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.deltajob_byte_len(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* The serialized delta segment.
	* @returns {Uint8Array}
	*/
	finish() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr = this.__destroy_into_raw();
		let ret;
		__wbg_call_guard();
		ret = wasm.deltajob_finish(ptr);
		var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
		return v1;
	}
	/**
	* Entries added so far.
	* @returns {number}
	*/
	len() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.deltajob_len(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* `means`: the generation's `k * dim` fine means (`fine.f32`); `seq`: the
	* segment's sequence number (stored in its header).
	* @param {number} dim
	* @param {number} k
	* @param {Float32Array} means
	* @param {number} seq
	*/
	constructor(dim, k, means, seq) {
		const ptr0 = passArrayF32ToWasm0(means, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.deltajob_new(dim, k, ptr0, len0, seq);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		this.__wbg_ptr = ret[0];
		Object.defineProperty(this, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		DeltaJobFinalization.register(this, {
			ptr: ret[0],
			instance: __wbg_instance_id
		}, this);
		return this;
	}
	/**
	* PQ codes per entry (0 without a codebook).
	* @returns {number}
	*/
	pq_m() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.deltajob_pq_m(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* Entries also carry PQ codes under this codebook (required to compact
	* into format 3 slabs). Call before adding vectors.
	* @param {Uint8Array} codebook_bytes
	*/
	set_codebook(codebook_bytes) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArray8ToWasm0(codebook_bytes, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.deltajob_set_codebook(this.__wbg_ptr, ptr0, len0);
		if (ret[1]) throw takeFromExternrefTable0(ret[0]);
	}
	/**
	* Same as `set_codebook`, from a parsed `PqCodebook` (copied).
	* @param {PqCodebook} codebook
	*/
	set_pq_codebook(codebook) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		_assertClass(codebook, PqCodebook);
		if (codebook.__wbg_inst !== void 0 && codebook.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.deltajob_set_pq_codebook(this.__wbg_ptr, codebook.__wbg_ptr);
		if (ret[1]) throw takeFromExternrefTable0(ret[0]);
	}
	/**
	* Picks each added vector's primary cell with a pruned search: the tree
	* (`build_centroid_tree` over these `k` means) scores its group means and
	* only the cells of the best `beam` groups (`beam >= groups`: identical to
	* the exhaustive search). Applies to vectors added afterwards.
	* @param {Uint8Array} tree_bytes
	* @param {number} beam
	*/
	set_tree(tree_bytes, beam) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArray8ToWasm0(tree_bytes, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.deltajob_set_tree(this.__wbg_ptr, ptr0, len0, beam);
		if (ret[1]) throw takeFromExternrefTable0(ret[0]);
	}
};
if (Symbol.dispose) DeltaJob.prototype[Symbol.dispose] = DeltaJob.prototype.free;
/**
* Picks initial fine centroids: a reservoir of up to `k2` sample vectors per
* coarse group, streamed.
*/
var FineInitJob = class {
	__destroy_into_raw() {
		const ptr = this.__wbg_ptr;
		this.__wbg_ptr = 0;
		FineInitJobFinalization.unregister(this);
		return ptr;
	}
	free() {
		const ptr = this.__destroy_into_raw();
		__wbg_call_guard();
		wasm.__wbg_fineinitjob_free(ptr, 0);
	}
	/**
	* @param {Float32Array} vectors
	* @returns {number}
	*/
	add_batch(vectors) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.fineinitjob_add_batch(this.__wbg_ptr, ptr0, len0);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		return ret[0] >>> 0;
	}
	/**
	* `k1 * k2 * dim` initial fine means, grouped by coarse group.
	* @returns {Float32Array}
	*/
	finish() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr = this.__destroy_into_raw();
		let ret;
		__wbg_call_guard();
		ret = wasm.fineinitjob_finish(ptr);
		var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v1;
	}
	/**
	* @param {number} dim
	* @param {number} k1
	* @param {number} k2
	* @param {Float32Array} coarse_means
	* @param {number} seed
	*/
	constructor(dim, k1, k2, coarse_means, seed) {
		const ptr0 = passArrayF32ToWasm0(coarse_means, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.fineinitjob_new(dim, k1, k2, ptr0, len0, seed);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		this.__wbg_ptr = ret[0];
		Object.defineProperty(this, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		FineInitJobFinalization.register(this, {
			ptr: ret[0],
			instance: __wbg_instance_id
		}, this);
		return this;
	}
};
if (Symbol.dispose) FineInitJob.prototype[Symbol.dispose] = FineInitJob.prototype.free;
/**
* Flat k-means map job over `k` centroids (used for the coarse level).
*/
var FlatKmeansJob = class {
	__destroy_into_raw() {
		const ptr = this.__wbg_ptr;
		this.__wbg_ptr = 0;
		FlatKmeansJobFinalization.unregister(this);
		return ptr;
	}
	free() {
		const ptr = this.__destroy_into_raw();
		__wbg_call_guard();
		wasm.__wbg_flatkmeansjob_free(ptr, 0);
	}
	/**
	* Assigns each vector in the batch and accumulates. Returns vectors seen.
	* @param {Float32Array} vectors
	* @returns {number}
	*/
	add_batch(vectors) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.flatkmeansjob_add_batch(this.__wbg_ptr, ptr0, len0);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		return ret[0] >>> 0;
	}
	/**
	* Convenience for single-job k-means: new means from this job's accumulator alone.
	* @param {Float32Array} prev
	* @returns {Float32Array}
	*/
	means(prev) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(prev, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.flatkmeansjob_means(this.__wbg_ptr, ptr0, len0);
		var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v2;
	}
	/**
	* @param {number} dim
	* @param {number} k
	* @param {Float32Array} centroid_means
	*/
	constructor(dim, k, centroid_means) {
		const ptr0 = passArrayF32ToWasm0(centroid_means, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.flatkmeansjob_new(dim, k, ptr0, len0);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		this.__wbg_ptr = ret[0];
		Object.defineProperty(this, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		FlatKmeansJobFinalization.register(this, {
			ptr: ret[0],
			instance: __wbg_instance_id
		}, this);
		return this;
	}
	/**
	* Serialized partial accumulator.
	* @returns {Uint8Array}
	*/
	partial() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.flatkmeansjob_partial(this.__wbg_ptr);
		var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
		return v1;
	}
};
if (Symbol.dispose) FlatKmeansJob.prototype[Symbol.dispose] = FlatKmeansJob.prototype.free;
/**
* Incremental flat k-means map job: each vector is compared only against the
* `n_neigh` neighbour cells of its previous cell (`neighbors`, a `k x n_neigh`
* table from `cell_neighbors`), not all `k` centroids. Accumulates like
* `TreeKmeansJob`; `partial()` has the same format, so `Reducer` is unchanged.
*/
var IncKmeansJob = class {
	__destroy_into_raw() {
		const ptr = this.__wbg_ptr;
		this.__wbg_ptr = 0;
		IncKmeansJobFinalization.unregister(this);
		return ptr;
	}
	free() {
		const ptr = this.__destroy_into_raw();
		__wbg_call_guard();
		wasm.__wbg_inckmeansjob_free(ptr, 0);
	}
	/**
	* Assigns vector `i` to its best cell among the neighbours of `prev[i]`,
	* accumulates, and returns the new cell per vector. Nothing is
	* accumulated if the arguments are invalid.
	* @param {Float32Array} vectors
	* @param {Uint32Array} prev
	* @returns {Uint32Array}
	*/
	add_batch(vectors, prev) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		const ptr1 = passArray32ToWasm0(prev, wasm.__wbindgen_malloc);
		const len1 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.inckmeansjob_add_batch(this.__wbg_ptr, ptr0, len0, ptr1, len1);
		if (ret[3]) throw takeFromExternrefTable0(ret[2]);
		var v3 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v3;
	}
	/**
	* @param {number} dim
	* @param {number} k
	* @param {Float32Array} means
	* @param {Uint32Array} neighbors
	* @param {number} n_neigh
	*/
	constructor(dim, k, means, neighbors, n_neigh) {
		const ptr0 = passArrayF32ToWasm0(means, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		const ptr1 = passArray32ToWasm0(neighbors, wasm.__wbindgen_malloc);
		const len1 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.inckmeansjob_new(dim, k, ptr0, len0, ptr1, len1, n_neigh);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		this.__wbg_ptr = ret[0];
		Object.defineProperty(this, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		IncKmeansJobFinalization.register(this, {
			ptr: ret[0],
			instance: __wbg_instance_id
		}, this);
		return this;
	}
	/**
	* @returns {Uint8Array}
	*/
	partial() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.inckmeansjob_partial(this.__wbg_ptr);
		var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
		return v1;
	}
};
if (Symbol.dispose) IncKmeansJob.prototype[Symbol.dispose] = IncKmeansJob.prototype.free;
/**
* A parsed codebook to keep across queries (parsing ~1.5 MB per query is
* avoidable): build each query's table with `PqTable.from_codebook`.
*/
var PqCodebook = class {
	__destroy_into_raw() {
		const ptr = this.__wbg_ptr;
		this.__wbg_ptr = 0;
		PqCodebookFinalization.unregister(this);
		return ptr;
	}
	free() {
		const ptr = this.__destroy_into_raw();
		__wbg_call_guard();
		wasm.__wbg_pqcodebook_free(ptr, 0);
	}
	/**
	* @returns {number}
	*/
	dim() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.pqcodebook_dim(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* @returns {number}
	*/
	m() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.pqcodebook_m(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* @param {Uint8Array} codebook_bytes
	*/
	constructor(codebook_bytes) {
		const ptr0 = passArray8ToWasm0(codebook_bytes, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.pqcodebook_new(ptr0, len0);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		this.__wbg_ptr = ret[0];
		Object.defineProperty(this, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		PqCodebookFinalization.register(this, {
			ptr: ret[0],
			instance: __wbg_instance_id
		}, this);
		return this;
	}
};
if (Symbol.dispose) PqCodebook.prototype[Symbol.dispose] = PqCodebook.prototype.free;
/**
* Streaming reduce of PQ training partials: `add_partial` each partial
* (so only one is held at a time), then `finish(prev_codebook)` returns the
* next codebook. Empty sub-centroids keep their previous value.
*/
var PqReducer = class {
	__destroy_into_raw() {
		const ptr = this.__wbg_ptr;
		this.__wbg_ptr = 0;
		PqReducerFinalization.unregister(this);
		return ptr;
	}
	free() {
		const ptr = this.__destroy_into_raw();
		__wbg_call_guard();
		wasm.__wbg_pqreducer_free(ptr, 0);
	}
	/**
	* @param {Uint8Array} partial
	*/
	add_partial(partial) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArray8ToWasm0(partial, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.pqreducer_add_partial(this.__wbg_ptr, ptr0, len0);
		if (ret[1]) throw takeFromExternrefTable0(ret[0]);
	}
	/**
	* Sub-centroids (out of `m * 256`) that received no residual.
	* @returns {number}
	*/
	empty_centroids() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.pqreducer_empty_centroids(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* Next serialized codebook.
	* @param {Uint8Array} prev_codebook
	* @returns {Uint8Array}
	*/
	finish(prev_codebook) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArray8ToWasm0(prev_codebook, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.pqreducer_finish(this.__wbg_ptr, ptr0, len0);
		if (ret[3]) throw takeFromExternrefTable0(ret[2]);
		var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
		return v2;
	}
	/**
	* Mean squared quantization error over all added partials (under the
	* codebook the map jobs used): a convergence signal per iteration.
	* @returns {number}
	*/
	mean_sq_err() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.pqreducer_mean_sq_err(this.__wbg_ptr);
		return ret;
	}
	/**
	* @param {number} dim
	* @param {number} m
	*/
	constructor(dim, m) {
		let ret;
		__wbg_call_guard();
		ret = wasm.pqreducer_new(dim, m);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		this.__wbg_ptr = ret[0];
		Object.defineProperty(this, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		PqReducerFinalization.register(this, {
			ptr: ret[0],
			instance: __wbg_instance_id
		}, this);
		return this;
	}
	/**
	* Residuals over all added partials.
	* @returns {number}
	*/
	rows() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.pqreducer_rows(this.__wbg_ptr);
		return ret;
	}
};
if (Symbol.dispose) PqReducer.prototype[Symbol.dispose] = PqReducer.prototype.free;
/**
* Per-query asymmetric-distance table (`m x 256` f32: `q_sub · sub-centroid`).
* Build once per query, then `scan_pq` every probed v3 slab with it.
*/
var PqTable = class PqTable {
	static __wrap(ptr) {
		const obj = Object.create(PqTable.prototype);
		obj.__wbg_ptr = ptr;
		Object.defineProperty(obj, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		PqTableFinalization.register(obj, {
			ptr,
			instance: __wbg_instance_id
		}, obj);
		return obj;
	}
	__destroy_into_raw() {
		const ptr = this.__wbg_ptr;
		this.__wbg_ptr = 0;
		PqTableFinalization.unregister(this);
		return ptr;
	}
	free() {
		const ptr = this.__destroy_into_raw();
		__wbg_call_guard();
		wasm.__wbg_pqtable_free(ptr, 0);
	}
	/**
	* Same as the constructor, from a cached `PqCodebook`.
	* @param {Float32Array} q
	* @param {PqCodebook} codebook
	* @returns {PqTable}
	*/
	static from_codebook(q, codebook) {
		const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		_assertClass(codebook, PqCodebook);
		if (codebook.__wbg_inst !== void 0 && codebook.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.pqtable_from_codebook(ptr0, len0, codebook.__wbg_ptr);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		return PqTable.__wrap(ret[0]);
	}
	/**
	* @returns {number}
	*/
	m() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.pqtable_m(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* `q`: the unit query; `codebook_bytes`: the index's serialized codebook.
	* @param {Float32Array} q
	* @param {Uint8Array} codebook_bytes
	*/
	constructor(q, codebook_bytes) {
		const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		const ptr1 = passArray8ToWasm0(codebook_bytes, wasm.__wbindgen_malloc);
		const len1 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.pqtable_new(ptr0, len0, ptr1, len1);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		this.__wbg_ptr = ret[0];
		Object.defineProperty(this, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		PqTableFinalization.register(this, {
			ptr: ret[0],
			instance: __wbg_instance_id
		}, this);
		return this;
	}
};
if (Symbol.dispose) PqTable.prototype[Symbol.dispose] = PqTable.prototype.free;
/**
* PQ training map job (one Lloyd iteration): assigns each residual's
* subvectors to the nearest sub-centroids of the current codebook and
* accumulates sums and counts. `partial()` is ~1.8 MB at dim 1536, m 192.
*/
var PqTrainJob = class {
	__destroy_into_raw() {
		const ptr = this.__wbg_ptr;
		this.__wbg_ptr = 0;
		PqTrainJobFinalization.unregister(this);
		return ptr;
	}
	free() {
		const ptr = this.__destroy_into_raw();
		__wbg_call_guard();
		wasm.__wbg_pqtrainjob_free(ptr, 0);
	}
	/**
	* Adds `n * dim` residuals (see `residuals_for`). Returns `n`.
	* @param {Float32Array} residuals
	* @returns {number}
	*/
	add_residuals(residuals) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(residuals, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.pqtrainjob_add_residuals(this.__wbg_ptr, ptr0, len0);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		return ret[0] >>> 0;
	}
	/**
	* Mean squared quantization error of the added residuals under the
	* job's (input) codebook.
	* @returns {number}
	*/
	mean_sq_err() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.pqtrainjob_mean_sq_err(this.__wbg_ptr);
		return ret;
	}
	/**
	* @param {number} dim
	* @param {number} m
	* @param {Uint8Array} codebook_bytes
	*/
	constructor(dim, m, codebook_bytes) {
		const ptr0 = passArray8ToWasm0(codebook_bytes, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.pqtrainjob_new(dim, m, ptr0, len0);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		this.__wbg_ptr = ret[0];
		Object.defineProperty(this, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		PqTrainJobFinalization.register(this, {
			ptr: ret[0],
			instance: __wbg_instance_id
		}, this);
		return this;
	}
	/**
	* Serialized partial accumulator ("VIBT"), for `pq_reduce` / `PqReducer`.
	* @returns {Uint8Array}
	*/
	partial() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.pqtrainjob_partial(this.__wbg_ptr);
		var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
		return v1;
	}
	/**
	* Residuals added so far.
	* @returns {number}
	*/
	rows() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.pqtrainjob_rows(this.__wbg_ptr);
		return ret;
	}
};
if (Symbol.dispose) PqTrainJob.prototype[Symbol.dispose] = PqTrainJob.prototype.free;
/**
* Reduce job over a cell range: sums partial records, then emits new means.
*/
var Reducer = class {
	__destroy_into_raw() {
		const ptr = this.__wbg_ptr;
		this.__wbg_ptr = 0;
		ReducerFinalization.unregister(this);
		return ptr;
	}
	free() {
		const ptr = this.__destroy_into_raw();
		__wbg_call_guard();
		wasm.__wbg_reducer_free(ptr, 0);
	}
	/**
	* Adds the raw records for this reducer's cell range from one partial.
	* @param {Uint8Array} records
	*/
	add_records(records) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArray8ToWasm0(records, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.reducer_add_records(this.__wbg_ptr, ptr0, len0);
		if (ret[1]) throw takeFromExternrefTable0(ret[0]);
	}
	/**
	* Per-cell counts after reduction (useful to spot empty cells).
	* @returns {Uint32Array}
	*/
	counts() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.reducer_counts(this.__wbg_ptr);
		var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v1;
	}
	/**
	* Cells reseeded by the last `means_reseed` call.
	* @returns {number}
	*/
	last_reseeded() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.reducer_last_reseeded(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* New means; `prev` is the previous means for the same cell range.
	* @param {Float32Array} prev
	* @returns {Float32Array}
	*/
	means(prev) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(prev, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.reducer_means(this.__wbg_ptr, ptr0, len0);
		if (ret[3]) throw takeFromExternrefTable0(ret[2]);
		var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v2;
	}
	/**
	* Like `means`, but cells with fewer than `min_count` members are reseeded
	* next to the densest cells of this range (deterministic jitter from `seed`).
	* The number of reseeded cells is available from `last_reseeded()`.
	* @param {Float32Array} prev
	* @param {number} min_count
	* @param {number} seed
	* @returns {Float32Array}
	*/
	means_reseed(prev, min_count, seed) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(prev, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.reducer_means_reseed(this.__wbg_ptr, ptr0, len0, min_count, seed);
		if (ret[3]) throw takeFromExternrefTable0(ret[2]);
		var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v2;
	}
	/**
	* @param {number} cells
	* @param {number} dim
	*/
	constructor(cells, dim) {
		let ret;
		__wbg_call_guard();
		ret = wasm.reducer_new(cells, dim);
		this.__wbg_ptr = ret;
		Object.defineProperty(this, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		ReducerFinalization.register(this, {
			ptr: ret,
			instance: __wbg_instance_id
		}, this);
		return this;
	}
};
if (Symbol.dispose) Reducer.prototype[Symbol.dispose] = Reducer.prototype.free;
/**
* Query-side: routing table plus slab scanning.
*/
var Searcher = class {
	__destroy_into_raw() {
		const ptr = this.__wbg_ptr;
		this.__wbg_ptr = 0;
		SearcherFinalization.unregister(this);
		return ptr;
	}
	free() {
		const ptr = this.__destroy_into_raw();
		__wbg_call_guard();
		wasm.__wbg_searcher_free(ptr, 0);
	}
	/**
	* `q·mean[c]` (unnormalized means) for all `k1 * k2` fine cells, by global
	* cell id: the `qc` argument of `scan_delta`. Pass the result to
	* `route_from_dots` to route without computing the dots twice.
	* @param {Float32Array} q
	* @returns {Float32Array}
	*/
	cell_dots(q) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.searcher_cell_dots(this.__wbg_ptr, ptr0, len0);
		if (ret[3]) throw takeFromExternrefTable0(ret[2]);
		var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v2;
	}
	/**
	* `q·mean[c]` for the listed global cell ids only, in order (the same
	* values as `cell_dots(q)[c]`): the delta-scan base scores of just the
	* cells the caller needs.
	* @param {Float32Array} q
	* @param {Uint32Array} cells
	* @returns {Float32Array}
	*/
	cell_dots_for(q, cells) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		const ptr1 = passArray32ToWasm0(cells, wasm.__wbindgen_malloc);
		const len1 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.searcher_cell_dots_for(this.__wbg_ptr, ptr0, len0, ptr1, len1);
		if (ret[3]) throw takeFromExternrefTable0(ret[2]);
		var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v3;
	}
	/**
	* @returns {number}
	*/
	dim() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.searcher_dim(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* @param {number} dim
	* @param {number} k1
	* @param {number} k2
	* @param {Float32Array} coarse_means
	* @param {Float32Array} fine_means
	*/
	constructor(dim, k1, k2, coarse_means, fine_means) {
		const ptr0 = passArrayF32ToWasm0(coarse_means, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		const ptr1 = passArrayF32ToWasm0(fine_means, wasm.__wbindgen_malloc);
		const len1 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.searcher_new(dim, k1, k2, ptr0, len0, ptr1, len1);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		this.__wbg_ptr = ret[0];
		Object.defineProperty(this, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		SearcherFinalization.register(this, {
			ptr: ret[0],
			instance: __wbg_instance_id
		}, this);
		return this;
	}
	/**
	* Global ids of the `nprobe` fine cells to scan for unit query `q`.
	* `probe_coarse == 0` (or `>= k1`) ranks all fine cells directly (flat routing);
	* otherwise only the fine cells of the `probe_coarse` best coarse groups are ranked.
	* @param {Float32Array} q
	* @param {number} probe_coarse
	* @param {number} nprobe
	* @returns {Uint32Array}
	*/
	route(q, probe_coarse, nprobe) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.searcher_route(this.__wbg_ptr, ptr0, len0, probe_coarse, nprobe);
		if (ret[3]) throw takeFromExternrefTable0(ret[2]);
		var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v2;
	}
	/**
	* Pruned flat routing: global ids of the `nprobe` best fine cells (by
	* cosine, descending) among the members of the tree's best `beam` groups.
	* `beam >= groups` equals `route_flat(q, nprobe)`. Error without `set_tree`.
	* @param {Float32Array} q
	* @param {number} beam
	* @param {number} nprobe
	* @returns {Uint32Array}
	*/
	route_beam(q, beam, nprobe) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.searcher_route_beam(this.__wbg_ptr, ptr0, len0, beam, nprobe);
		if (ret[3]) throw takeFromExternrefTable0(ret[2]);
		var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v2;
	}
	/**
	* Flat routing: global ids of the `nprobe` best fine cells over all `k1 * k2`.
	* @param {Float32Array} q
	* @param {number} nprobe
	* @returns {Uint32Array}
	*/
	route_flat(q, nprobe) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.searcher_route_flat(this.__wbg_ptr, ptr0, len0, nprobe);
		if (ret[3]) throw takeFromExternrefTable0(ret[2]);
		var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v2;
	}
	/**
	* Flat routing from `cell_dots(q)`: same result as `route_flat(q, nprobe)`.
	* @param {Float32Array} dots
	* @param {number} nprobe
	* @returns {Uint32Array}
	*/
	route_from_dots(dots, nprobe) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(dots, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.searcher_route_from_dots(this.__wbg_ptr, ptr0, len0, nprobe);
		if (ret[3]) throw takeFromExternrefTable0(ret[2]);
		var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v2;
	}
	/**
	* Installs a centroid tree (`build_centroid_tree` over this searcher's
	* `k1 * k2` fine means) for `route_beam`. Replaces any previous tree.
	* @param {Uint8Array} tree_bytes
	*/
	set_tree(tree_bytes) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArray8ToWasm0(tree_bytes, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.searcher_set_tree(this.__wbg_ptr, ptr0, len0);
		if (ret[1]) throw takeFromExternrefTable0(ret[0]);
	}
	/**
	* Coarse group ids ranked for `q` (for lazy fine-centroid loading).
	* @param {Float32Array} q
	* @param {number} n
	* @returns {Uint32Array}
	*/
	top_coarse(q, n) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.searcher_top_coarse(this.__wbg_ptr, ptr0, len0, n);
		var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v2;
	}
	/**
	* Groups of the installed tree (0 = none).
	* @returns {number}
	*/
	tree_groups() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.searcher_tree_groups(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* Fraction of `sample` vectors (`n * dim`) whose exact nearest fine cell
	* the tree finds with this `beam` (for logging; costs an exhaustive
	* search per vector). Error without `set_tree`.
	* @param {Float32Array} sample
	* @param {number} beam
	* @returns {number}
	*/
	tree_recall(sample, beam) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(sample, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.searcher_tree_recall(this.__wbg_ptr, ptr0, len0, beam);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		return ret[0];
	}
	/**
	* Mean fraction of each sample vector's exact top `m` fine cells that the
	* tree's top `m` with this `beam` contains (`m = 1`: `tree_recall`).
	* @param {Float32Array} sample
	* @param {number} beam
	* @param {number} m
	* @returns {number}
	*/
	tree_recall_top(sample, beam, m) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(sample, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.searcher_tree_recall_top(this.__wbg_ptr, ptr0, len0, beam, m);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		return ret[0];
	}
};
if (Symbol.dispose) Searcher.prototype[Symbol.dispose] = Searcher.prototype.free;
/**
* Finalize job for one fine cell: concatenates part record ranges into a slab.
*/
var SlabJob = class {
	__destroy_into_raw() {
		const ptr = this.__wbg_ptr;
		this.__wbg_ptr = 0;
		SlabJobFinalization.unregister(this);
		return ptr;
	}
	free() {
		const ptr = this.__destroy_into_raw();
		__wbg_call_guard();
		wasm.__wbg_slabjob_free(ptr, 0);
	}
	/**
	* Adds every record in a records byte range (one cell's directory range,
	* or any run of whole records, e.g. a cell sliced out of a `part_span`).
	* @param {Uint8Array} bytes
	* @returns {number}
	*/
	add_records(bytes) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.slabjob_add_records(this.__wbg_ptr, ptr0, len0);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		return ret[0] >>> 0;
	}
	/**
	* Adds the records of one cell from a fetched multi-cell span without
	* copying it in JS: `span` holds part bytes starting at part offset
	* `span_offset`; the cell's directory entry is `[cell_off, cell_len)`
	* (part offsets). Adds `span[cell_off - span_offset, +cell_len)`.
	* @param {Uint8Array} span
	* @param {number} span_offset
	* @param {number} cell_off
	* @param {number} cell_len
	* @returns {number}
	*/
	add_records_range(span, span_offset, cell_off, cell_len) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArray8ToWasm0(span, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.slabjob_add_records_range(this.__wbg_ptr, ptr0, len0, span_offset, cell_off, cell_len);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		return ret[0] >>> 0;
	}
	/**
	* @returns {number}
	*/
	byte_len() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.slabjob_byte_len(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* @returns {Uint8Array}
	*/
	finish() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr = this.__destroy_into_raw();
		let ret;
		__wbg_call_guard();
		ret = wasm.slabjob_finish(ptr);
		var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
		return v1;
	}
	/**
	* Slab format version this job writes (1, 2 or 3).
	* @returns {number}
	*/
	format() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.slabjob_format(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* @returns {number}
	*/
	len() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.slabjob_len(this.__wbg_ptr);
		return ret >>> 0;
	}
	/**
	* `format`: 2 (default when omitted) = v2 slab (4-bit scan tier + int8
	* rerank tier); 1 = v1 int8-only slab (from the records' int8 codes);
	* 3 = v3 slab (the records' PQ codes as scan tier + int8 rerank tier; the
	* records must come from an `AssignJob` with `set_codebook`, otherwise
	* `add_records*` errors).
	* @param {number} dim
	* @param {number} cell
	* @param {Float32Array} centroid_mean
	* @param {number | null} [format]
	*/
	constructor(dim, cell, centroid_mean, format) {
		const ptr0 = passArrayF32ToWasm0(centroid_mean, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.slabjob_new(dim, cell, ptr0, len0, isLikeNone(format) ? 16777215 : format);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		this.__wbg_ptr = ret[0];
		Object.defineProperty(this, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		SlabJobFinalization.register(this, {
			ptr: ret[0],
			instance: __wbg_instance_id
		}, this);
		return this;
	}
};
if (Symbol.dispose) SlabJob.prototype[Symbol.dispose] = SlabJob.prototype.free;
/**
* Two-level k-means map job: assigns each vector to its best fine cell among
* the `probe_coarse` nearest coarse groups and accumulates per fine cell.
*/
var TreeKmeansJob = class {
	__destroy_into_raw() {
		const ptr = this.__wbg_ptr;
		this.__wbg_ptr = 0;
		TreeKmeansJobFinalization.unregister(this);
		return ptr;
	}
	free() {
		const ptr = this.__destroy_into_raw();
		__wbg_call_guard();
		wasm.__wbg_treekmeansjob_free(ptr, 0);
	}
	/**
	* Like `add_batch`, but returns the chosen global fine cell of every
	* vector (with `k1 = 1` a flat assignment), e.g. to seed `IncKmeansJob`.
	* @param {Float32Array} vectors
	* @returns {Uint32Array}
	*/
	add_batch_assign(vectors) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.treekmeansjob_add_batch_assign(this.__wbg_ptr, ptr0, len0);
		if (ret[3]) throw takeFromExternrefTable0(ret[2]);
		var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
		return v2;
	}
	/**
	* @param {Float32Array} vectors
	* @returns {number}
	*/
	add_batch(vectors) {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.treekmeansjob_add_batch(this.__wbg_ptr, ptr0, len0);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		return ret[0] >>> 0;
	}
	/**
	* @param {number} dim
	* @param {number} k1
	* @param {number} k2
	* @param {Float32Array} coarse_means
	* @param {Float32Array} fine_means
	* @param {number} probe_coarse
	*/
	constructor(dim, k1, k2, coarse_means, fine_means, probe_coarse) {
		const ptr0 = passArrayF32ToWasm0(coarse_means, wasm.__wbindgen_malloc);
		const len0 = WASM_VECTOR_LEN;
		const ptr1 = passArrayF32ToWasm0(fine_means, wasm.__wbindgen_malloc);
		const len1 = WASM_VECTOR_LEN;
		let ret;
		__wbg_call_guard();
		ret = wasm.treekmeansjob_new(dim, k1, k2, ptr0, len0, ptr1, len1, probe_coarse);
		if (ret[2]) throw takeFromExternrefTable0(ret[1]);
		this.__wbg_ptr = ret[0];
		Object.defineProperty(this, "__wbg_inst", {
			value: __wbg_instance_id,
			writable: true
		});
		TreeKmeansJobFinalization.register(this, {
			ptr: ret[0],
			instance: __wbg_instance_id
		}, this);
		return this;
	}
	/**
	* @returns {Uint8Array}
	*/
	partial() {
		if (this.__wbg_inst !== void 0 && this.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
		let ret;
		__wbg_call_guard();
		ret = wasm.treekmeansjob_partial(this.__wbg_ptr);
		var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
		wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
		return v1;
	}
};
if (Symbol.dispose) TreeKmeansJob.prototype[Symbol.dispose] = TreeKmeansJob.prototype.free;
function __wbg_reset_state() {
	__wbg_instance_id++;
	cachedFloat32ArrayMemory0 = null;
	cachedUint32ArrayMemory0 = null;
	cachedUint8ArrayMemory0 = null;
	if (typeof numBytesDecoded !== "undefined") numBytesDecoded = 0;
	if (typeof WASM_VECTOR_LEN !== "undefined") WASM_VECTOR_LEN = 0;
	__wbg_reinit_scheduled = false;
	wasmInstance = new WebAssembly.Instance(wasmModule$1, __wbg_get_imports());
	wasm = wasmInstance.exports;
	wasm.__wbindgen_start();
}
/**
* Byte range `[offset, length]` inside a serialized partial accumulator that
* covers cells `[first, first + n)`.
* @param {number} dim
* @param {number} first
* @param {number} n
* @returns {Uint32Array}
*/
function accum_record_range(dim, first, n) {
	let ret;
	__wbg_call_guard();
	ret = wasm.accum_record_range(dim, first, n);
	var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
	return v1;
}
/**
* Builds a centroid tree ("VIBC" bytes) over `k = means.len() / dim` cell
* means (raw, e.g. `fine.f32`): spherical k-means of the means into
* `groups` groups (clamped to `1..=k`), k-means++ seeding, up to `iters`
* Lloyd iterations, deterministic for `seed`. Lookups (`Searcher.set_tree`,
* `DeltaJob.set_tree`, `AssignJob.set_tree`) score the group means, then only
* the cells of the best `beam` groups. Cost: about `(iters + 2) * groups * k * dim`
* multiply-adds (k = 4096, dim = 1536, groups = 64, iters = 10: ~5e9).
* @param {Float32Array} means
* @param {number} dim
* @param {number} groups
* @param {number} iters
* @param {number} seed
* @returns {Uint8Array}
*/
function build_centroid_tree(means, dim, groups, iters, seed) {
	const ptr0 = passArrayF32ToWasm0(means, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.build_centroid_tree(ptr0, len0, dim, groups, iters, seed);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
	return v2;
}
/**
* Cell neighbour table rows for cells `[first, first + count)`: the `n`
* cells with the highest cosine between means, each row starting with the
* cell itself, flattened row-major (`count * n` ids). `means` is the full
* `k * dim` table; shard the work over `first/count` ranges.
* @param {Float32Array} means
* @param {number} dim
* @param {number} first
* @param {number} count
* @param {number} n
* @returns {Uint32Array}
*/
function cell_neighbors(means, dim, first, count, n) {
	const ptr0 = passArrayF32ToWasm0(means, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.cell_neighbors(ptr0, len0, dim, first, count, n);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
	return v2;
}
/**
* Header summary of centroid tree bytes: `[dim, k, groups, n_members, byte_len]`.
* @param {Uint8Array} tree_bytes
* @returns {Uint32Array}
*/
function centroid_tree_info(tree_bytes) {
	const ptr0 = passArray8ToWasm0(tree_bytes, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.centroid_tree_info(ptr0, len0);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
	return v2;
}
/**
* Distinct cells of a delta segment with their entry counts, ascending:
* `[cell0, count0, cell1, count1, ...]`.
* @param {Uint8Array} delta_bytes
* @returns {Uint32Array}
*/
function delta_cells(delta_bytes) {
	const ptr0 = passArray8ToWasm0(delta_bytes, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.delta_cells(ptr0, len0);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
	return v2;
}
/**
* Part records (for `SlabJob.add_records`) of every entry of `cell` in a delta
* segment whose id is not in `tombstones` (`u32` LE ids, may be empty).
* @param {Uint8Array} delta_bytes
* @param {number} cell
* @param {Uint8Array} tombstones
* @returns {Uint8Array}
*/
function delta_records_for_cell(delta_bytes, cell, tombstones) {
	const ptr0 = passArray8ToWasm0(delta_bytes, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	const ptr1 = passArray8ToWasm0(tombstones, wasm.__wbindgen_malloc);
	const len1 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.delta_records_for_cell(ptr0, len0, cell, ptr1, len1);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
	return v3;
}
/**
* Installs a panic hook that logs the panic message (and location) via
* `console.error`, so panics show up in Workers logs before the wasm traps.
* Idempotent; call once after instantiating the module. After a trap
* (`WebAssembly.RuntimeError`) the instance is poisoned: call
* `__wbg_reset_state()` and re-init before using it again.
*/
function init_panic_hook() {
	__wbg_call_guard();
	wasm.init_panic_hook();
}
/**
* Merges concatenated packed hit buffers, dedupes by id (keeping the best
* score and that copy's `idx`), keeps the best `k`.
* @param {Uint8Array} packed
* @param {number} k
* @returns {Uint8Array}
*/
function merge(packed, k) {
	const ptr0 = passArray8ToWasm0(packed, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.merge(ptr0, len0, k);
	var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
	return v2;
}
/**
* Returns a unit-normalized copy of `q`.
* @param {Float32Array} q
* @returns {Float32Array}
*/
function normalize_vector(q) {
	const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.normalize_vector(ptr0, len0);
	var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
	return v2;
}
/**
* Parses a part directory: returns `[k2, dim, count0, off0, len0, count1, ...]`.
* @param {Uint8Array} bytes
* @returns {Uint32Array}
*/
function part_directory(bytes) {
	const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.part_directory(ptr0, len0);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
	return v2;
}
/**
* @param {number} k2
* @returns {number}
*/
function part_directory_len(k2) {
	let ret;
	__wbg_call_guard();
	ret = wasm.part_directory_len(k2);
	return ret >>> 0;
}
/**
* Byte span `[offset, length]` of one part covering cells
* `[first_cell, first_cell + count)`, from `dir` = the output of
* `part_directory`. Empty cells at the edges are skipped; no records gives
* `[0, 0]`. Slice each cell out of the fetched span with its directory entry
* (`off - offset`), or use `SlabJob.add_records_range`.
* @param {Uint32Array} dir
* @param {number} first_cell
* @param {number} count
* @returns {Uint32Array}
*/
function part_span(dir, first_cell, count) {
	const ptr0 = passArray32ToWasm0(dir, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.part_span(ptr0, len0, first_cell, count);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
	return v2;
}
/**
* `[dim, m, sub]` of a serialized codebook.
* @param {Uint8Array} codebook_bytes
* @returns {Uint32Array}
*/
function pq_codebook_info(codebook_bytes) {
	const ptr0 = passArray8ToWasm0(codebook_bytes, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.pq_codebook_info(ptr0, len0);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
	return v2;
}
/**
* Initial PQ codebook (serialized, "VIBQ"): per subspace, the subvectors of
* 256 random rows of `residuals` (`n * dim`; pass ~2-4k rows). Requires
* `dim % m == 0`.
* @param {Float32Array} residuals
* @param {number} dim
* @param {number} m
* @param {number} seed
* @returns {Uint8Array}
*/
function pq_init_codebook(residuals, dim, m, seed) {
	const ptr0 = passArrayF32ToWasm0(residuals, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.pq_init_codebook(ptr0, len0, dim, m, seed);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
	return v2;
}
/**
* Scores `n` consecutive v2/v3 rerank records (`[scale8 f32][code8 i8 * dim]`,
* `dim = q.length`) from one slab against `q`: `q·centroid + scale8 * (q·code8)`.
* @param {Float32Array} q
* @param {Float32Array} centroid
* @param {Uint8Array} records
* @param {number} n
* @returns {Float32Array}
*/
function rerank(q, centroid, records, n) {
	const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	const ptr1 = passArrayF32ToWasm0(centroid, wasm.__wbindgen_malloc);
	const len1 = WASM_VECTOR_LEN;
	const ptr2 = passArray8ToWasm0(records, wasm.__wbindgen_malloc);
	const len2 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.rerank(ptr0, len0, ptr1, len1, ptr2, len2, n);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
	return v4;
}
/**
* `vectors[i] - means[assign[i]]` for each row (`n * dim` floats): turns a
* sample batch and its cell assignment into PQ training residuals.
* @param {Float32Array} vectors
* @param {Float32Array} means
* @param {Uint32Array} assign
* @param {number} dim
* @returns {Float32Array}
*/
function residuals_for(vectors, means, assign, dim) {
	const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	const ptr1 = passArrayF32ToWasm0(means, wasm.__wbindgen_malloc);
	const len1 = WASM_VECTOR_LEN;
	const ptr2 = passArray32ToWasm0(assign, wasm.__wbindgen_malloc);
	const len2 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.residuals_for(ptr0, len0, ptr1, len1, ptr2, len2, dim);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
	return v4;
}
/**
* Scans one slab for unit query `q`; returns packed top-k hits
* (`[count u32][id u32, score f32, idx u32]*`, `idx` = entry index in the slab).
* For a v2 slab, pass the `[0, scan_len)` prefix (the full slab also works);
* scores are 4-bit scan-tier scores, to be refined with `rerank`. v1 slabs
* must be passed whole up to their payload and give int8 scores. v3 slabs are
* an error: scan them with `scan_pq`.
*
* `tombstones` (`u32` LE ids, sorted unique as `tombstones_merge` writes them;
* empty = none) are skipped inside the scan, before top-k insertion, so the
* result holds the best `k` live entries. `stats` is an out-parameter
* (`Uint32Array`, copied back after the call): `stats[0]` = entries scanned,
* `stats[1]` = tombstoned entries skipped (only those that scored high enough
* to enter the top-k when reached). Pass `new Uint32Array(0)` to ignore them.
* @param {Uint8Array} slab_scan_bytes
* @param {Float32Array} q
* @param {number} k
* @param {Uint8Array} tombstones
* @param {Uint32Array} stats
* @returns {Uint8Array}
*/
function scan(slab_scan_bytes, q, k, tombstones, stats) {
	const ptr0 = passArray8ToWasm0(slab_scan_bytes, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	const ptr1 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
	const len1 = WASM_VECTOR_LEN;
	const ptr2 = passArray8ToWasm0(tombstones, wasm.__wbindgen_malloc);
	const len2 = WASM_VECTOR_LEN;
	var ptr3 = passArray32ToWasm0(stats, wasm.__wbindgen_malloc);
	var len3 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.scan(ptr0, len0, ptr1, len1, k, ptr2, len2, ptr3, len3, stats);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v5 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
	return v5;
}
/**
* Scans a whole delta segment for unit query `q` with its int8 records:
* score = `qc[cell] + scale8 * (q·code8)`, where `qc = Searcher.cell_dots(q)`
* (one dot per fine cell, unnormalized means). Returns the top `k` as packed
* hits in the `scan` wire format, `idx` = entry index in the segment. Scores
* are on the reranked (int8) scale, so they merge directly with `rerank`
* results; no second phase is needed. `tombstones` and `stats` as in `scan`:
* pass the ids to hide in this segment (for a segment added at rev `a`, the
* union of the sets with rev > `a`); `stats` receives `[scanned, skipped]`.
* @param {Uint8Array} delta_bytes
* @param {Float32Array} q
* @param {Float32Array} qc
* @param {number} k
* @param {Uint8Array} tombstones
* @param {Uint32Array} stats
* @returns {Uint8Array}
*/
function scan_delta(delta_bytes, q, qc, k, tombstones, stats) {
	const ptr0 = passArray8ToWasm0(delta_bytes, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	const ptr1 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
	const len1 = WASM_VECTOR_LEN;
	const ptr2 = passArrayF32ToWasm0(qc, wasm.__wbindgen_malloc);
	const len2 = WASM_VECTOR_LEN;
	const ptr3 = passArray8ToWasm0(tombstones, wasm.__wbindgen_malloc);
	const len3 = WASM_VECTOR_LEN;
	var ptr4 = passArray32ToWasm0(stats, wasm.__wbindgen_malloc);
	var len4 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.scan_delta(ptr0, len0, ptr1, len1, ptr2, len2, k, ptr3, len3, ptr4, len4, stats);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v6 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
	return v6;
}
/**
* Scans one v3 slab (its `[0, scan_len)` prefix, or the whole slab) for unit
* query `q` with PQ scores `q·centroid + sum_j table[j][code_j]`; returns
* packed top-k hits in the `scan` wire format (`idx` = entry index, locating
* the int8 rerank record for `rerank`). `table` must be built from the same
* `q`. Errors on v1/v2 slabs and on a table whose `m` differs from the slab's.
* `tombstones` and `stats` as in `scan`: tombstoned ids are skipped before
* top-k insertion; `stats` receives `[scanned, skipped]`.
* @param {Uint8Array} slab_scan_bytes
* @param {PqTable} table
* @param {Float32Array} q
* @param {number} k
* @param {Uint8Array} tombstones
* @param {Uint32Array} stats
* @returns {Uint8Array}
*/
function scan_pq(slab_scan_bytes, table, q, k, tombstones, stats) {
	const ptr0 = passArray8ToWasm0(slab_scan_bytes, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	_assertClass(table, PqTable);
	if (table.__wbg_inst !== void 0 && table.__wbg_inst !== __wbg_instance_id) throw new Error("Invalid stale object from previous Wasm instance");
	const ptr1 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
	const len1 = WASM_VECTOR_LEN;
	const ptr2 = passArray8ToWasm0(tombstones, wasm.__wbindgen_malloc);
	const len2 = WASM_VECTOR_LEN;
	var ptr3 = passArray32ToWasm0(stats, wasm.__wbindgen_malloc);
	var len3 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.scan_pq(ptr0, len0, table.__wbg_ptr, ptr1, len1, k, ptr2, len2, ptr3, len3, stats);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v5 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
	return v5;
}
/**
* The cell mean stored in a slab (or v2 scan prefix).
* @param {Uint8Array} slab_scan_bytes
* @returns {Float32Array}
*/
function slab_centroid(slab_scan_bytes) {
	const ptr0 = passArray8ToWasm0(slab_scan_bytes, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.slab_centroid(ptr0, len0);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
	return v2;
}
/**
* Header summary of a slab (or v2/v3 scan prefix):
* `[dim, count, cell, byte_len, version, scan_len, rerank_off, rerank_record_len, pq_m]`.
* `byte_len` is the full object size from the header (equals `slab.length`
* for a whole slab). For v1, `scan_len` is the payload offset and
* `rerank_off`/`rerank_record_len` are 0. `pq_m` is 0 except for v3.
* @param {Uint8Array} slab
* @returns {Uint32Array}
*/
function slab_info(slab) {
	const ptr0 = passArray8ToWasm0(slab, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.slab_info(ptr0, len0);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
	return v2;
}
/**
* Part records (for `SlabJob.add_records`) of every entry of a whole slab
* (not a scan prefix) whose id is not in `tombstones` (may be empty). A slab
* rebuilt from them with `SlabJob(dim, cell, mean, sameFormat)` is byte-identical.
* v3 and v1 slabs have no 4-bit codes: their records are flagged and a
* format 2 `SlabJob` rejects them (`add_records` errors); v2 records carry no
* PQ codes, so a format 3 `SlabJob` rejects them.
* @param {Uint8Array} slab_bytes
* @param {Uint8Array} tombstones
* @returns {Uint8Array}
*/
function slab_to_records(slab_bytes, tombstones) {
	const ptr0 = passArray8ToWasm0(slab_bytes, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	const ptr1 = passArray8ToWasm0(tombstones, wasm.__wbindgen_malloc);
	const len1 = WASM_VECTOR_LEN;
	let ret;
	__wbg_call_guard();
	ret = wasm.slab_to_records(ptr0, len0, ptr1, len1);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
	return v3;
}
function __wbg_get_imports() {
	return {
		__proto__: null,
		"./vib_bg.js": {
			__proto__: null,
			__wbg___wbindgen_copy_to_typed_array_88899a52af046901: function(arg0, arg1, arg2) {
				new Uint8Array(arg2.buffer, arg2.byteOffset, arg2.byteLength).set(getArrayU8FromWasm0(arg0, arg1));
			},
			__wbg___wbindgen_throw_41e9ee4f547fc59a: function(arg0, arg1) {
				throw new Error(getStringFromWasm0(arg0, arg1));
			},
			__wbg_error_9cf5d0367d002490: function(arg0, arg1) {
				console.error(getStringFromWasm0(arg0, arg1));
			},
			__wbindgen_generic_0000000000000001: function(arg0, arg1) {
				return getStringFromWasm0(arg0, arg1);
			},
			__wbindgen_init_externref_table: function() {
				const table = wasm.__wbindgen_externrefs;
				const offset = table.grow(4);
				table.set(0, void 0);
				table.set(offset + 0, void 0);
				table.set(offset + 1, null);
				table.set(offset + 2, true);
				table.set(offset + 3, false);
			}
		}
	};
}
function __wbg_call_guard() {
	if (__wbg_reinit_scheduled) {
		__wbg_reset_state();
		return;
	}
}
var __wbg_instance_id = 0;
var AssignJobFinalization = typeof FinalizationRegistry === "undefined" ? {
	register: () => {},
	unregister: () => {}
} : new FinalizationRegistry(({ ptr, instance }) => {
	if (instance === __wbg_instance_id) wasm.__wbg_assignjob_free(ptr, 1);
});
var DeltaJobFinalization = typeof FinalizationRegistry === "undefined" ? {
	register: () => {},
	unregister: () => {}
} : new FinalizationRegistry(({ ptr, instance }) => {
	if (instance === __wbg_instance_id) wasm.__wbg_deltajob_free(ptr, 1);
});
var FineInitJobFinalization = typeof FinalizationRegistry === "undefined" ? {
	register: () => {},
	unregister: () => {}
} : new FinalizationRegistry(({ ptr, instance }) => {
	if (instance === __wbg_instance_id) wasm.__wbg_fineinitjob_free(ptr, 1);
});
var FlatKmeansJobFinalization = typeof FinalizationRegistry === "undefined" ? {
	register: () => {},
	unregister: () => {}
} : new FinalizationRegistry(({ ptr, instance }) => {
	if (instance === __wbg_instance_id) wasm.__wbg_flatkmeansjob_free(ptr, 1);
});
var IncKmeansJobFinalization = typeof FinalizationRegistry === "undefined" ? {
	register: () => {},
	unregister: () => {}
} : new FinalizationRegistry(({ ptr, instance }) => {
	if (instance === __wbg_instance_id) wasm.__wbg_inckmeansjob_free(ptr, 1);
});
var PqCodebookFinalization = typeof FinalizationRegistry === "undefined" ? {
	register: () => {},
	unregister: () => {}
} : new FinalizationRegistry(({ ptr, instance }) => {
	if (instance === __wbg_instance_id) wasm.__wbg_pqcodebook_free(ptr, 1);
});
var PqReducerFinalization = typeof FinalizationRegistry === "undefined" ? {
	register: () => {},
	unregister: () => {}
} : new FinalizationRegistry(({ ptr, instance }) => {
	if (instance === __wbg_instance_id) wasm.__wbg_pqreducer_free(ptr, 1);
});
var PqTableFinalization = typeof FinalizationRegistry === "undefined" ? {
	register: () => {},
	unregister: () => {}
} : new FinalizationRegistry(({ ptr, instance }) => {
	if (instance === __wbg_instance_id) wasm.__wbg_pqtable_free(ptr, 1);
});
var PqTrainJobFinalization = typeof FinalizationRegistry === "undefined" ? {
	register: () => {},
	unregister: () => {}
} : new FinalizationRegistry(({ ptr, instance }) => {
	if (instance === __wbg_instance_id) wasm.__wbg_pqtrainjob_free(ptr, 1);
});
var ReducerFinalization = typeof FinalizationRegistry === "undefined" ? {
	register: () => {},
	unregister: () => {}
} : new FinalizationRegistry(({ ptr, instance }) => {
	if (instance === __wbg_instance_id) wasm.__wbg_reducer_free(ptr, 1);
});
var SearcherFinalization = typeof FinalizationRegistry === "undefined" ? {
	register: () => {},
	unregister: () => {}
} : new FinalizationRegistry(({ ptr, instance }) => {
	if (instance === __wbg_instance_id) wasm.__wbg_searcher_free(ptr, 1);
});
var SlabJobFinalization = typeof FinalizationRegistry === "undefined" ? {
	register: () => {},
	unregister: () => {}
} : new FinalizationRegistry(({ ptr, instance }) => {
	if (instance === __wbg_instance_id) wasm.__wbg_slabjob_free(ptr, 1);
});
var TreeKmeansJobFinalization = typeof FinalizationRegistry === "undefined" ? {
	register: () => {},
	unregister: () => {}
} : new FinalizationRegistry(({ ptr, instance }) => {
	if (instance === __wbg_instance_id) wasm.__wbg_treekmeansjob_free(ptr, 1);
});
function _assertClass(instance, klass) {
	if (!(instance instanceof klass)) throw new Error(`expected instance of ${klass.name}`);
}
function getArrayF32FromWasm0(ptr, len) {
	ptr = ptr >>> 0;
	return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}
function getArrayU32FromWasm0(ptr, len) {
	ptr = ptr >>> 0;
	return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}
function getArrayU8FromWasm0(ptr, len) {
	ptr = ptr >>> 0;
	return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}
var cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
	if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
	return cachedFloat32ArrayMemory0;
}
function getStringFromWasm0(ptr, len) {
	return decodeText(ptr >>> 0, len);
}
var cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
	if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
	return cachedUint32ArrayMemory0;
}
var cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
	if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
	return cachedUint8ArrayMemory0;
}
function isLikeNone(x) {
	return x === void 0 || x === null;
}
function passArray32ToWasm0(arg, malloc) {
	const ptr = malloc(arg.length * 4, 4) >>> 0;
	getUint32ArrayMemory0().set(arg, ptr / 4);
	WASM_VECTOR_LEN = arg.length;
	return ptr;
}
function passArray8ToWasm0(arg, malloc) {
	const ptr = malloc(arg.length * 1, 1) >>> 0;
	getUint8ArrayMemory0().set(arg, ptr / 1);
	WASM_VECTOR_LEN = arg.length;
	return ptr;
}
function passArrayF32ToWasm0(arg, malloc) {
	const ptr = malloc(arg.length * 4, 4) >>> 0;
	getFloat32ArrayMemory0().set(arg, ptr / 4);
	WASM_VECTOR_LEN = arg.length;
	return ptr;
}
var __wbg_reinit_scheduled = false;
function takeFromExternrefTable0(idx) {
	const value = wasm.__wbindgen_externrefs.get(idx);
	wasm.__externref_table_dealloc(idx);
	return value;
}
var cachedTextDecoder = new TextDecoder("utf-8", {
	ignoreBOM: true,
	fatal: true
});
cachedTextDecoder.decode();
var MAX_SAFARI_DECODE_BYTES = 2146435072;
var numBytesDecoded = 0;
function decodeText(ptr, len) {
	numBytesDecoded += len;
	if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
		cachedTextDecoder = new TextDecoder("utf-8", {
			ignoreBOM: true,
			fatal: true
		});
		cachedTextDecoder.decode();
		numBytesDecoded = len;
	}
	return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}
var WASM_VECTOR_LEN = 0;
var wasmModule$1;
var wasmInstance;
var wasm;
function __wbg_finalize_init(instance, module) {
	wasmInstance = instance;
	wasm = instance.exports;
	wasmModule$1 = module;
	cachedFloat32ArrayMemory0 = null;
	cachedUint32ArrayMemory0 = null;
	cachedUint8ArrayMemory0 = null;
	wasm.__wbindgen_start();
	return wasm;
}
async function __wbg_load(module, imports) {
	if (typeof Response === "function" && module instanceof Response) {
		if (!module.ok) throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
		if (typeof WebAssembly.instantiateStreaming === "function") try {
			return await WebAssembly.instantiateStreaming(module, imports);
		} catch (e) {
			if (expectedResponseType(module.type) && module.headers.get("Content-Type") !== "application/wasm") console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);
			else throw e;
		}
		const bytes = await module.arrayBuffer();
		return await WebAssembly.instantiate(bytes, imports);
	} else {
		const instance = await WebAssembly.instantiate(module, imports);
		if (instance instanceof WebAssembly.Instance) return {
			instance,
			module
		};
		else return instance;
	}
	function expectedResponseType(type) {
		switch (type) {
			case "basic":
			case "cors":
			case "default": return true;
		}
		return false;
	}
}
async function __wbg_init(module_or_path) {
	if (wasm !== void 0) return wasm;
	if (module_or_path !== void 0) {
		if (Object.getPrototypeOf(module_or_path) === Object.prototype) ({module_or_path} = module_or_path);
		else console.warn("using deprecated parameters for the initialization function; pass a single object instead");
	}
	if (module_or_path === void 0) module_or_path = new URL("vib_bg.wasm", import.meta.url);
	const imports = __wbg_get_imports();
	if (typeof module_or_path === "string" || typeof Request === "function" && module_or_path instanceof Request || typeof URL === "function" && module_or_path instanceof URL) module_or_path = fetch(module_or_path);
	const { instance, module } = await __wbg_load(await module_or_path, imports);
	return __wbg_finalize_init(instance, module);
}
//#endregion
//#region src/index.ts
/**
* vib-bench Worker: thin I/O shell around the Rust/wasm core.
*
* Build stages are stateless HTTP endpoints driven by a local script, so each
* invocation stays inside the Worker memory/CPU budget and the orchestration
* (fan-out, retries) lives outside. Query serving caches the routing table per
* isolate and reads one slab object per probed cell.
*
* R2 key layout:
*   ingest/{shard:05}.f32                raw unit vectors, f32 LE
*   build/{gen}/sample/{i}.f32           strided sample for coarse k-means
*   build/{gen}/coarse/{iter}.f32        k1*dim means
*   build/{gen}/fine/{iter}.f32          k1*k2*dim means (grouped by coarse)
*   build/{gen}/fine/{iter}/{g}.f32      per-group slice written by reduce jobs
*   build/{gen}/partials/{tag}/{job}.acc partial k-means accumulators
*   build/{gen}/assign/{iter}/{shard:05}.u32  k-means cell per vector after iteration iter (u32 LE)
*   build/{gen}/neighbors/{iter}.u32     k x nNeigh cell neighbour table of fine/{iter} (u32 LE)
*   build/{gen}/neighbors/{iter}/{r}.u32 row ranges written by cell-neighbors jobs
*   build/{gen}/pq/residuals/{i}.f32     PQ training residuals (sample vectors minus their final cell mean)
*   build/{gen}/pq/codebook/{it}.bin     PQ codebook after iteration it (0 = pq-init)
*   build/{gen}/pq/partials/{it}/{j}.acc PQ training map partials of iteration it
*   build/{gen}/parts/{g}/{job}.part     assign-job output per coarse group
*   idx/{gen}/manifest.json, coarse.f32, fine.f32, slab/{cell:05}   rev 0 (publish / finalize)
*   idx/{gen}/slabs.json                 per cell [scanLen, rerankOff, count, bytes] (rerankOff 0 for format 1)
*   idx/{gen}/pq.bin                     format 3: the PQ codebook the scan tier was encoded with
*   idx/{gen}/manifest.{rev}.json        revisions >= 1 (publish-rev, upsert, delete, compact); immutable
*   idx/{gen}/slabs.{rev}.json           slabs table written by a compaction; rows gain a 5th column, slabRev
*   idx/{gen}/slab/{cell:05}.{slabRev}   slab rewritten by a compaction that read manifest rev slabRev - 1
*   idx/{gen}/delta/{seq}.vibd           delta segment of one upsert
*   idx/{gen}/tombstones.{rev}.u32       ids deleted/replaced at rev (sorted u32 LE)
*   idx/{gen}/tree.{rev}.bin             centroid tree over the fine means (publish / publish-rev with `tree`)
*   idx/current.json                     {gen, rev}
*
* The manifest records the build profile (profile, m, k1, k2, format, tiers, pq, nNeigh), the query defaults
* (defaults: {nprobe, rerank, pc, route?}) that /query uses for absent parameters, the centroid tree (tree: {key,
* groups, beam, ...}) and, from rev 1 on, its files, deltas and tombstone sets. See docs/update-path.md.
*/
var wasmReady = null;
/**
* Bumped on every reset; work started under an older epoch holds stale wasm objects. Rule: every wasm
* object is freed only in the epoch it was created in (safeFree(o, epoch) skips free() otherwise).
*/
var wasmEpoch = 0;
var wasmResets = 0;
/** Raw wasm exports (for the wasm memory size reported in query stats). */
var wasmRaw = null;
function initWasm() {
	return __wbg_init({ module_or_path: wasmModule }).then((w) => {
		wasmRaw = w;
		init_panic_hook();
	});
}
function ensureWasm() {
	if (!wasmReady) {
		const p = initWasm();
		wasmReady = p;
		p.catch(() => {
			if (wasmReady === p) wasmReady = null;
		});
	}
	return wasmReady;
}
/**
* After a trap (WebAssembly.RuntimeError) the instance is poisoned: re-instantiate it,
* reinstall the panic hook and drop every cached wasm object (the Searcher).
*/
function resetWasm(reason) {
	wasmResets++;
	wasmEpoch++;
	console.error(`wasm trap, resetting instance (reset #${wasmResets}): ${reason instanceof Error ? `${reason.message}\n${reason.stack ?? ""}` : String(reason)}`);
	__wbg_reset_state();
	wasmReady = initWasm();
	for (const slot of baseCache.values()) slot.evicted = true;
	baseCache.clear();
	slabCache.clear();
	slabCacheBytes = 0;
}
/**
* Runs wasm-using work; on a wasm trap resets the instance and retries once. Errors thrown
* because another request reset the instance mid-flight (stale objects) are retried too.
*/
async function withWasmRecovery(label, fn) {
	const epoch = wasmEpoch;
	try {
		return await fn(1);
	} catch (e) {
		const trap = e instanceof WebAssembly.RuntimeError;
		if (!trap && epoch === wasmEpoch) throw e;
		if (trap && epoch === wasmEpoch) resetWasm(e);
		else console.error(`${label}: retrying after concurrent wasm reset: ${e instanceof Error ? e.message : String(e)}`);
		await ensureWasm();
		return await fn(2);
	}
}
var HttpError = class extends Error {
	status;
	constructor(status, message) {
		super(message);
		this.status = status;
	}
};
function json(data, status = 200, headers = {}) {
	return new Response(JSON.stringify(data), {
		status,
		headers: {
			"content-type": "application/json",
			...headers
		}
	});
}
async function pmap(items, limit, fn) {
	const out = new Array(items.length);
	let next = 0;
	const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
		while (true) {
			const i = next++;
			if (i >= items.length) return;
			out[i] = await fn(items[i], i);
		}
	});
	await Promise.all(workers);
	return out;
}
/** Runs fn over keys sequentially while prefetching `ahead` objects. */
async function streamObjects(bucket, keys, ahead, fn, range) {
	let bytes = 0;
	const pending = [];
	const start = (i) => getBuffer(bucket, keys[i], range?.(i));
	for (let i = 0; i < Math.min(ahead, keys.length); i++) pending.push(start(i));
	for (let i = 0; i < keys.length; i++) {
		const buf = await pending.shift();
		const nextIdx = i + ahead;
		if (nextIdx < keys.length) pending.push(start(nextIdx));
		bytes += buf.byteLength;
		await fn(buf, i);
	}
	return bytes;
}
async function getBuffer(bucket, key, range) {
	const obj = await bucket.get(key, range ? { range } : void 0);
	if (!obj) throw new HttpError(404, `missing object ${key}`);
	return await obj.arrayBuffer();
}
async function getF32(bucket, key, range) {
	const buf = await getBuffer(bucket, key, range);
	if (buf.byteLength % 4 !== 0) throw new HttpError(500, `object ${key} is not f32-aligned`);
	return new Float32Array(buf);
}
async function putBytes(bucket, key, data) {
	const body = typeof data === "string" ? data : data.buffer ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : data;
	await bucket.put(key, body);
}
function pad5(n) {
	return n.toString().padStart(5, "0");
}
/**
* Frees a wasm object created in `epoch` (capture `wasmEpoch` right after `await ensureWasm()`).
* Every wasm object is freed only in the epoch it was created in: after a reset its pointer means
* nothing in the new instance, and free() there could release memory another request now owns.
* Ignores errors (after a trap the instance is poisoned and free() throws too).
*/
function safeFree(o, epoch) {
	if (epoch !== wasmEpoch) return;
	try {
		o.free();
	} catch {}
}
async function readJson(request) {
	try {
		return await request.json();
	} catch {
		throw new HttpError(400, "invalid JSON body");
	}
}
/** Reads the first `perShard` vectors of each shard and concatenates them. */
async function stageSample(env, body) {
	const { gen, shards, perShard, dim, out } = body;
	const rowBytes = dim * 4;
	const buf = new Float32Array(shards.length * perShard * dim);
	let n = 0;
	await streamObjects(env.BUCKET, shards, 6, (b) => {
		const f = new Float32Array(b);
		buf.set(f, n * dim);
		n += f.length / dim;
	}, () => ({
		offset: 0,
		length: perShard * rowBytes
	}));
	const key = `build/${gen}/sample/${out}.f32`;
	await putBytes(env.BUCKET, key, buf.subarray(0, n * dim));
	return {
		key,
		vectors: n
	};
}
/** Initial coarse centroids: the first k sample vectors (sample is strided over shards). */
async function stageInitCoarse(env, body) {
	const { gen, sampleKeys, k, dim } = body;
	const out = new Float32Array(k * dim);
	let n = 0;
	for (const key of sampleKeys) {
		if (n >= k) break;
		const f = await getF32(env.BUCKET, key);
		const take = Math.min(f.length / dim, k - n);
		out.set(f.subarray(0, take * dim), n * dim);
		n += take;
	}
	if (n < k) throw new HttpError(400, `sample has ${n} vectors, need ${k}`);
	const key = `build/${gen}/coarse/0.f32`;
	await putBytes(env.BUCKET, key, out);
	return {
		key,
		k,
		dim
	};
}
/** One full iteration of flat k-means over a small input set (coarse level). */
async function stageKmeansFlat(env, body) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { centroidsKey, inputKeys, k, dim, outKey } = body;
	const prev = await getF32(env.BUCKET, centroidsKey);
	const job = new FlatKmeansJob(dim, k, prev);
	let vectors = 0;
	const bytes = await streamObjects(env.BUCKET, inputKeys, 2, (b) => {
		vectors += job.add_batch(new Float32Array(b));
	});
	const means = job.means(prev);
	safeFree(job, epoch);
	await putBytes(env.BUCKET, outKey, means);
	return {
		outKey,
		vectors,
		bytes
	};
}
async function stageInitFine(env, body) {
	await ensureWasm();
	const { coarseKey, sampleKeys, k1, k2, dim, seed, outKey } = body;
	const job = new FineInitJob(dim, k1, k2, await getF32(env.BUCKET, coarseKey), seed >>> 0);
	let vectors = 0;
	await streamObjects(env.BUCKET, sampleKeys, 1, (b) => {
		vectors += job.add_batch(new Float32Array(b));
	});
	const fine = job.finish();
	await putBytes(env.BUCKET, outKey, fine);
	return {
		outKey,
		vectors,
		cells: k1 * k2
	};
}
/** Accepts `{key, index}` or a bare ingest key (index parsed from `ingest/{index:05}.f32`). */
function shardRefs(shards) {
	return shards.map((s) => {
		if (typeof s !== "string") {
			if (!Number.isInteger(s.index) || s.index < 0) throw new HttpError(400, `shard ${s.key}: bad index ${s.index}`);
			return s;
		}
		const m = s.match(/(\d+)\.f32$/);
		if (!m) throw new HttpError(400, `cannot parse a shard index from ${s}`);
		return {
			key: s,
			index: Number(m[1])
		};
	});
}
/** Per-shard assignment object written by the k-means map stages: one u32 LE cell id per vector. */
var assignKey = (prefix, index) => `${prefix}/${pad5(index)}.u32`;
async function getU32(bucket, key) {
	const buf = await getBuffer(bucket, key);
	if (buf.byteLength % 4 !== 0) throw new HttpError(500, `object ${key} is not u32-aligned`);
	return new Uint32Array(buf);
}
/**
* Build-time coarse groups searched per vector, clamped to [1, k1]. The build jobs (TreeKmeansJob, AssignJob)
* panic with probe_coarse 0 (no candidate cells), unlike Searcher.route where 0 means flat; so 0 (or absent)
* is mapped to k1, which is the exhaustive flat search.
*/
function buildProbeCoarse(pc, k1) {
	if (!Number.isInteger(k1) || k1 < 1) throw new HttpError(400, "k1 must be an integer >= 1");
	if (pc === void 0 || pc === null || !Number.isFinite(pc) || pc < 1) return k1;
	return Math.min(Math.floor(pc), k1);
}
async function stageKmeansTreeMap(env, body) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { coarseKey, fineKey, k1, k2, dim, outKey, assignOutPrefix } = body;
	const probeCoarse = buildProbeCoarse(body.probeCoarse, k1);
	const shards = shardRefs(body.shards);
	const job = await (async () => {
		const [coarse, fine] = await Promise.all([getF32(env.BUCKET, coarseKey), getF32(env.BUCKET, fineKey)]);
		return new TreeKmeansJob(dim, k1, k2, coarse, fine, probeCoarse);
	})();
	let vectors = 0;
	const writes = [];
	try {
		const bytes = await streamObjects(env.BUCKET, shards.map((s) => s.key), 1, (b, i) => {
			const v = new Float32Array(b);
			if (assignOutPrefix) {
				const a = job.add_batch_assign(v);
				vectors += a.length;
				writes.push(putBytes(env.BUCKET, assignKey(assignOutPrefix, shards[i].index), a));
			} else vectors += job.add_batch(v);
		});
		await Promise.all(writes);
		const partial = job.partial();
		await putBytes(env.BUCKET, outKey, partial);
		return {
			outKey,
			vectors,
			bytes,
			partialBytes: partial.byteLength,
			assignments: assignOutPrefix ? shards.length : 0
		};
	} finally {
		safeFree(job, epoch);
	}
}
/**
* Incremental k-means map: each vector is compared only against the nNeigh neighbours of its
* previous cell. Writes the new assignment per shard and the partial accumulator (same format
* as kmeans-tree-map, so kmeans-reduce is unchanged).
*/
async function stageKmeansIncMap(env, body) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { fineKey, neighborsKey, k, dim, nNeigh, prevAssignPrefix, assignOutPrefix, outKey } = body;
	const shards = shardRefs(body.shards);
	const job = await (async () => {
		const [means, neighbors] = await Promise.all([getF32(env.BUCKET, fineKey), getU32(env.BUCKET, neighborsKey)]);
		if (means.length !== k * dim) throw new HttpError(400, `${fineKey} has ${means.length / dim} means, expected ${k}`);
		if (neighbors.length !== k * nNeigh) throw new HttpError(400, `${neighborsKey} has ${neighbors.length} ids, expected ${k}x${nNeigh}`);
		return new IncKmeansJob(dim, k, means, neighbors, nNeigh);
	})();
	let vectors = 0;
	let changed = 0;
	let bytes = 0;
	const writes = [];
	try {
		const fetchShard = (s) => Promise.all([getBuffer(env.BUCKET, s.key), getU32(env.BUCKET, assignKey(prevAssignPrefix, s.index))]);
		let next = shards.length ? fetchShard(shards[0]) : null;
		for (let i = 0; i < shards.length; i++) {
			const [buf, prev] = await next;
			next = i + 1 < shards.length ? fetchShard(shards[i + 1]) : null;
			bytes += buf.byteLength;
			const v = new Float32Array(buf);
			if (prev.length * dim !== v.length) throw new HttpError(400, `shard ${shards[i].index}: ${v.length / dim} vectors but ${prev.length} previous assignments`);
			const a = job.add_batch(v, prev);
			if (a.length !== prev.length) throw new HttpError(500, `IncKmeansJob returned ${a.length} assignments for ${prev.length} vectors`);
			for (let x = 0; x < a.length; x++) if (a[x] !== prev[x]) changed++;
			vectors += a.length;
			writes.push(putBytes(env.BUCKET, assignKey(assignOutPrefix, shards[i].index), a));
		}
		await Promise.all(writes);
		const partial = job.partial();
		await putBytes(env.BUCKET, outKey, partial);
		return {
			outKey,
			vectors,
			changed,
			bytes,
			partialBytes: partial.byteLength
		};
	} finally {
		safeFree(job, epoch);
	}
}
/** Rows [first, first+count) of the cell neighbour table of the means in fineKey, as u32 LE (count x n). */
async function stageCellNeighbors(env, body) {
	await ensureWasm();
	const { fineKey, k, dim, first, count, n, outKey } = body;
	const rows = await (async () => {
		const means = await getF32(env.BUCKET, fineKey);
		if (means.length !== k * dim) throw new HttpError(400, `${fineKey} has ${means.length / dim} means, expected ${k}`);
		return cell_neighbors(means, dim, first, count, n);
	})();
	await putBytes(env.BUCKET, outKey, rows);
	return {
		outKey,
		first,
		count,
		n,
		bytes: rows.byteLength
	};
}
/** Reduces one cell range across all partials and writes the new means for it. */
async function stageKmeansReduce(env, body) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { partialKeys, prevFineKey, dim, firstCell, nCells, outKey } = body;
	const [off, len] = accum_record_range(dim, firstCell, nCells);
	const reducer = new Reducer(nCells, dim);
	const prevPromise = getF32(env.BUCKET, prevFineKey, {
		offset: firstCell * dim * 4,
		length: nCells * dim * 4
	});
	const bytes = await streamObjects(env.BUCKET, partialKeys, 6, (b) => reducer.add_records(new Uint8Array(b)), () => ({
		offset: off,
		length: len
	}));
	const prev = await prevPromise;
	const reseed = body.reseed;
	const means = reseed ? reducer.means_reseed(prev, reseed.minCount >>> 0, reseed.seed >>> 0) : reducer.means(prev);
	const reseeded = reseed ? reducer.last_reseeded() : 0;
	const counts = Array.from(reducer.counts());
	safeFree(reducer, epoch);
	await putBytes(env.BUCKET, outKey, means);
	return {
		outKey,
		bytes,
		cells: nCells,
		empty: counts.filter((c) => c === 0).length,
		reseeded,
		small: reseed ? counts.filter((c) => c < reseed.minCount).length : void 0,
		min: Math.min(...counts),
		max: Math.max(...counts),
		total: counts.reduce((a, b) => a + b, 0),
		counts
	};
}
async function stageConcat(env, body) {
	const parts = await pmap(body.keys, 6, (k) => getBuffer(env.BUCKET, k));
	const total = parts.reduce((a, b) => a + b.byteLength, 0);
	const out = new Uint8Array(total);
	let o = 0;
	for (const p of parts) {
		out.set(new Uint8Array(p), o);
		o += p.byteLength;
	}
	await putBytes(env.BUCKET, body.outKey, out);
	return {
		outKey: body.outKey,
		bytes: total
	};
}
/**
* PQ training residuals: for each shard, the first `perShard` vectors minus the mean of their final
* k-means cell (`vib.residuals_for`), concatenated (f32 LE) into outKey. Also reports the residual
* energy (mean squared norm), the reference for the PQ mean_sq_err.
*/
async function stagePqResiduals(env, body) {
	await ensureWasm();
	const { perShard, assignPrefix, fineKey, dim, k, outKey } = body;
	const shards = shardRefs(body.shards);
	const means = await getF32(env.BUCKET, fineKey);
	if (means.length !== k * dim) throw new HttpError(400, `${fineKey} has ${means.length / dim} means, expected ${k}`);
	const out = new Float32Array(shards.length * perShard * dim);
	let rows = 0;
	let sumSq = 0;
	let bytes = 0;
	const fetchShard = (s) => Promise.all([getBuffer(env.BUCKET, s.key, {
		offset: 0,
		length: perShard * dim * 4
	}), getBuffer(env.BUCKET, assignKey(assignPrefix, s.index), {
		offset: 0,
		length: perShard * 4
	})]);
	let next = shards.length ? fetchShard(shards[0]) : null;
	for (let i = 0; i < shards.length; i++) {
		const [vbuf, abuf] = await next;
		next = i + 1 < shards.length ? fetchShard(shards[i + 1]) : null;
		bytes += vbuf.byteLength + abuf.byteLength;
		const v = new Float32Array(vbuf);
		const a = new Uint32Array(abuf);
		const n = Math.min(v.length / dim, a.length);
		for (let x = 0; x < n; x++) if (a[x] >= k) throw new HttpError(500, `shard ${shards[i].index}: cell ${a[x]} >= k ${k}`);
		const r = residuals_for(v.subarray(0, n * dim), means, a.subarray(0, n), dim);
		for (let x = 0; x < r.length; x++) sumSq += r[x] * r[x];
		out.set(r, rows * dim);
		rows += n;
	}
	await putBytes(env.BUCKET, outKey, out.subarray(0, rows * dim));
	return {
		outKey,
		rows,
		bytes,
		residualEnergy: rows ? sumSq / rows : 0
	};
}
/** Initial codebook from the concatenated residual objects (~4000 rows). */
async function stagePqInit(env, body) {
	await ensureWasm();
	const { residualKeys, dim, m, seed, outKey } = body;
	if (dim % m !== 0) throw new HttpError(400, `dim ${dim} is not a multiple of m ${m}`);
	const parts = await pmap(residualKeys, SUBREQUEST_CONCURRENCY, (k) => getBuffer(env.BUCKET, k));
	const res = new Float32Array(parts.reduce((a, b) => a + b.byteLength, 0) / 4);
	let o = 0;
	for (const p of parts) {
		res.set(new Float32Array(p), o);
		o += p.byteLength / 4;
	}
	parts.length = 0;
	const codebook = pq_init_codebook(res, dim, m, seed >>> 0);
	const info = pq_codebook_info(codebook);
	await putBytes(env.BUCKET, outKey, codebook);
	return {
		outKey,
		rows: res.length / dim,
		bytes: codebook.byteLength,
		info: Array.from(info)
	};
}
/** One PQ Lloyd map job: streams residual objects through a PqTrainJob and writes its partial. */
async function stagePqTrainMap(env, body) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { codebookKey, residualKeys, dim, m, outKey } = body;
	const job = new PqTrainJob(dim, m, new Uint8Array(await getBuffer(env.BUCKET, codebookKey)));
	try {
		const bytes = await streamObjects(env.BUCKET, residualKeys, 1, (b) => {
			job.add_residuals(new Float32Array(b));
		});
		const partial = job.partial();
		await putBytes(env.BUCKET, outKey, partial);
		return {
			outKey,
			rows: job.rows(),
			mean_sq_err: job.mean_sq_err(),
			bytes,
			partialBytes: partial.byteLength
		};
	} finally {
		safeFree(job, epoch);
	}
}
/** Streams the map partials through a PqReducer and writes the next codebook. */
async function stagePqTrainReduce(env, body) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { partialKeys, prevCodebookKey, dim, m, outKey } = body;
	const reducer = new PqReducer(dim, m);
	try {
		const prevPromise = getBuffer(env.BUCKET, prevCodebookKey);
		const bytes = await streamObjects(env.BUCKET, partialKeys, 4, (b) => reducer.add_partial(new Uint8Array(b)));
		const rows = reducer.rows();
		const mean_sq_err = reducer.mean_sq_err();
		const empty_centroids = reducer.empty_centroids();
		const next = reducer.finish(new Uint8Array(await prevPromise));
		await putBytes(env.BUCKET, outKey, next);
		return {
			outKey,
			rows,
			mean_sq_err,
			empty_centroids,
			bytes,
			codebookBytes: next.byteLength
		};
	} finally {
		safeFree(reducer, epoch);
	}
}
async function stageAssign(env, body) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { gen, coarseKey, fineKey, k1, k2, dim, m, job: jobName, neighborsKey, nNeigh, prevAssignPrefix, codebookKey } = body;
	const probeCoarse = buildProbeCoarse(body.probeCoarse, k1);
	const split = body.split ?? 1;
	const candidates = neighborsKey !== void 0 || nNeigh !== void 0 || prevAssignPrefix !== void 0;
	if (candidates && (!neighborsKey || !nNeigh || !prevAssignPrefix)) throw new HttpError(400, "candidate mode needs neighborsKey, nNeigh and prevAssignPrefix");
	if (candidates && k1 !== 1) throw new HttpError(400, "candidate mode requires k1 = 1");
	if (!Number.isInteger(split) || split < 1) throw new HttpError(400, "split must be an integer >= 1");
	const shards = body.shards.map((s) => ({
		...s,
		index: s.index ?? shardRefs([s.key])[0].index
	}));
	const job = await (async () => {
		if (candidates) {
			const [fine, neighbors] = await Promise.all([getF32(env.BUCKET, fineKey), getU32(env.BUCKET, neighborsKey)]);
			if (neighbors.length !== k2 * nNeigh) throw new HttpError(400, `${neighborsKey} has ${neighbors.length} ids, expected ${k2}x${nNeigh}`);
			return AssignJob.with_candidates(dim, k2, fine, neighbors, nNeigh, m);
		}
		const [coarse, fine] = await Promise.all([getF32(env.BUCKET, coarseKey), getF32(env.BUCKET, fineKey)]);
		return new AssignJob(dim, k1, k2, coarse, fine, probeCoarse, m);
	})();
	if (codebookKey) try {
		job.set_codebook(new Uint8Array(await getBuffer(env.BUCKET, codebookKey)));
	} catch (e) {
		safeFree(job, epoch);
		throw e;
	}
	const pqM = job.pq_m();
	let vectors = 0;
	let partBytes = 0;
	let parts = 0;
	const groups = Array.from({ length: k1 }, (_, g) => g);
	const flush = async (suffix) => {
		await pmap(groups, 4, async (g) => {
			const part = job.take_part(g);
			partBytes += part.byteLength;
			parts++;
			await putBytes(env.BUCKET, `build/${gen}/parts/${g}/${jobName}${suffix}.part`, part);
		});
	};
	try {
		const fetchShard = (s) => Promise.all([getBuffer(env.BUCKET, s.key), candidates ? getU32(env.BUCKET, assignKey(prevAssignPrefix, s.index)) : Promise.resolve(null)]);
		let bytes = 0;
		let next = shards.length ? fetchShard(shards[0]) : null;
		for (let i = 0; i < shards.length; i++) {
			const [buf, prev] = await next;
			next = i + 1 < shards.length ? fetchShard(shards[i + 1]) : null;
			bytes += buf.byteLength;
			const v = new Float32Array(buf);
			const n = v.length / dim;
			if (prev && prev.length !== n) throw new HttpError(400, `shard ${shards[i].index}: ${n} vectors but ${prev.length} previous assignments`);
			const per = Math.ceil(n / split);
			for (let s = 0; s < split; s++) {
				const a = Math.min(n, s * per);
				const b = Math.min(n, a + per);
				if (b > a) {
					const sub = v.subarray(a * dim, b * dim);
					const firstId = shards[i].firstId + a;
					vectors += prev ? job.add_batch_prev(sub, firstId, prev.subarray(a, b)) : job.add_batch(sub, firstId);
				}
				if (split > 1) await flush(`.${i * split + s}`);
			}
		}
		const records = job.assigned_records();
		if (split === 1) await flush("");
		return {
			vectors,
			bytes,
			records,
			partBytes,
			parts,
			pqM,
			mode: candidates ? "candidates" : "exhaustive"
		};
	} finally {
		safeFree(job, epoch);
	}
}
/**
* Builds the slabs of a contiguous chunk of fine cells of coarse group g. Per part it reads the
* directory, then ONE contiguous span covering all of the chunk's cells (`part_span`), and feeds
* each cell's records to its SlabJob with `add_records_range` (no per-cell range reads).
*/
async function stageFinalize(env, body) {
	await ensureWasm();
	const { gen, g, k2, dim, partKeys, fineKey } = body;
	if (!body.overwrite) {
		const revs = await listRevs(env, gen);
		if (revs.length) throw new HttpError(409, `${gen} already has manifest revs ${revs.map((r) => r.rev).join(",")}; finalize would overwrite its published slabs (overwrite: true for a rebuild of the same gen)`);
	}
	const format = body.format ?? 2;
	if (format !== 1 && format !== 2 && format !== 3) throw new HttpError(400, "format must be 1, 2 or 3");
	const conc = body.concurrency ?? SUBREQUEST_CONCURRENCY;
	const budget = body.budgetBytes ?? 50331648;
	const js = body.cells ?? Array.from({ length: k2 }, (_, j) => j);
	if (js.length === 0) throw new HttpError(400, "no cells");
	if (js.some((j, x) => !Number.isInteger(j) || j < 0 || j >= k2 || x > 0 && j !== js[x - 1] + 1)) throw new HttpError(400, "cells must be a contiguous ascending range in [0, k2)");
	const first = js[0];
	const count = js.length;
	const dirLen = part_directory_len(k2);
	const tDir = Date.now();
	const [dirs, means] = await Promise.all([pmap(partKeys, conc, async (key) => {
		const d = part_directory(new Uint8Array(await getBuffer(env.BUCKET, key, {
			offset: 0,
			length: dirLen
		})));
		if (d[0] !== k2 || d[1] !== dim) throw new HttpError(500, `part ${key}: k2/dim ${d[0]}/${d[1]} != ${k2}/${dim}`);
		const [spanOff, spanLen] = part_span(d, first, count);
		const ent = new Uint32Array(count * 2);
		for (let x = 0; x < count; x++) {
			ent[x * 2] = d[2 + (first + x) * 3 + 1];
			ent[x * 2 + 1] = d[2 + (first + x) * 3 + 2];
		}
		return {
			key,
			spanOff,
			spanLen,
			ent
		};
	}), getF32(env.BUCKET, fineKey, {
		offset: (g * k2 + first) * dim * 4,
		length: count * dim * 4
	})]);
	const dirMs = Date.now() - tDir;
	const cellBytes = new Array(count).fill(0);
	for (const p of dirs) for (let x = 0; x < count; x++) cellBytes[x] += p.ent[x * 2 + 1];
	const batches = [];
	for (let a = 0; a < count;) {
		let b = a + 1;
		let sum = cellBytes[a];
		while (b < count && sum + cellBytes[b] <= budget) sum += cellBytes[b++];
		batches.push([a, b]);
		a = b;
	}
	const results = [];
	let spanReads = 0;
	let spanBytes = 0;
	const tSpan = Date.now();
	for (const [a, b] of batches) {
		const epoch = wasmEpoch;
		const slabs = Array.from({ length: b - a }, (_, i) => new SlabJob(dim, g * k2 + first + a + i, means.subarray((a + i) * dim, (a + i + 1) * dim), format));
		const live = new Set(slabs);
		try {
			if (slabs[0].format() !== format) throw new HttpError(500, `SlabJob writes format ${slabs[0].format()}, asked for ${format}`);
			await pmap(dirs, conc, async (p) => {
				let off = p.spanOff;
				let len = p.spanLen;
				if (batches.length > 1) {
					let lo = -1;
					let hi = 0;
					for (let x = a; x < b; x++) {
						const [o, l] = [p.ent[x * 2], p.ent[x * 2 + 1]];
						if (l === 0) continue;
						if (lo < 0) lo = o;
						hi = o + l;
					}
					[off, len] = lo < 0 ? [0, 0] : [lo, hi - lo];
				}
				if (len === 0) return;
				const span = new Uint8Array(await getBuffer(env.BUCKET, p.key, {
					offset: off,
					length: len
				}));
				spanReads++;
				spanBytes += len;
				for (let x = a; x < b; x++) {
					const l = p.ent[x * 2 + 1];
					if (l > 0) slabs[x - a].add_records_range(span, off, p.ent[x * 2], l);
				}
			});
			for (let i = 0; i < slabs.length; i++) {
				const s = slabs[i];
				const cell = g * k2 + first + a + i;
				const n = s.len();
				live.delete(s);
				const bytes = s.finish();
				const info = slab_info(bytes);
				await putBytes(env.BUCKET, `idx/${gen}/slab/${pad5(cell)}`, bytes);
				results.push({
					cell,
					count: n,
					bytes: bytes.byteLength,
					scanLen: info[5],
					rerankOff: info[6],
					...format === 3 ? { pqM: info[8] } : {}
				});
			}
		} finally {
			for (const s of live) safeFree(s, epoch);
		}
	}
	return {
		g,
		format,
		slabs: results.length,
		results,
		parts: partKeys.length,
		batches: batches.length,
		spanReads,
		spanBytes,
		dirMs,
		spanMs: Date.now() - tSpan
	};
}
var ROUTE_MODES = ["flat", "beam"];
/** Defaults of manifests without `defaults` (generations before g3). */
var LEGACY_DEFAULTS = {
	nprobe: 1,
	rerank: 32,
	pc: 0
};
var treeKey = (gen, rev) => `idx/${gen}/tree.${rev}.bin`;
/** Default sample objects for the tree recall figures: up to 8 of the generation's k-means sample objects. */
function defaultTreeSampleKeys(gen, sampleGroups = 80) {
	const n = Math.min(8, sampleGroups);
	return Array.from({ length: n }, (_, i) => `build/${gen}/sample/${Math.floor(i * sampleGroups / n)}.f32`);
}
function checkTreeReq(t, k) {
	const groups = t.groups ?? 128;
	const iters = t.iters ?? 10;
	const seed = t.seed ?? 7;
	const beam = t.beam ?? 8;
	if (!Number.isInteger(groups) || groups < 1 || groups > k) throw new HttpError(400, `tree.groups must be an integer in 1..${k}`);
	if (!Number.isInteger(iters) || iters < 0 || iters > 100) throw new HttpError(400, "tree.iters must be an integer in 0..100");
	if (!Number.isInteger(seed) || seed < 0) throw new HttpError(400, "tree.seed must be a non-negative integer");
	if (!Number.isInteger(beam) || beam < 1 || beam > groups) throw new HttpError(400, `tree.beam must be an integer in 1..${groups}`);
	return {
		groups,
		iters,
		seed,
		beam
	};
}
/**
* Builds the centroid tree over `fineKey` (k1 * k2 means) and measures its recall on sample vectors with a Searcher
* over the same means. Returns the bytes (not yet written) and the manifest entry without its key.
*/
async function buildTree(env, gen, m, coarseKey, fineKey, req, sampleGroups) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const k = m.k1 * m.k2;
	const { groups, iters, seed, beam } = checkTreeReq(req, k);
	let bytes;
	{
		const fine = await getF32(env.BUCKET, fineKey);
		if (fine.length !== k * m.dim) throw new HttpError(500, `${fineKey} has ${fine.length / m.dim} means, expected ${k}`);
		bytes = build_centroid_tree(fine, m.dim, groups, iters, seed);
	}
	const [tDim, tK, tGroups] = centroid_tree_info(bytes);
	if (tDim !== m.dim || tK !== k) throw new HttpError(500, `tree is dim ${tDim} k ${tK}, expected ${m.dim} / ${k}`);
	const ref = {
		groups: tGroups,
		beam,
		iters,
		seed,
		bytes: bytes.byteLength
	};
	const rows = req.sampleRows ?? 250;
	if (rows > 0) {
		const keys = req.sampleKeys ?? defaultTreeSampleKeys(gen, sampleGroups);
		const rowBytes = m.dim * 4;
		const parts = await Promise.all(keys.map(async (key) => {
			const o = await env.BUCKET.get(key, { range: {
				offset: 0,
				length: rows * rowBytes
			} });
			return o ? new Float32Array(await o.arrayBuffer()) : null;
		}));
		const got = [];
		for (const p of parts) if (p && p.length >= m.dim) got.push(p);
		const n = got.reduce((a, p) => a + Math.floor(p.length / m.dim), 0);
		if (n > 0) {
			const sample = new Float32Array(n * m.dim);
			let o = 0;
			for (const p of got) {
				const len = Math.floor(p.length / m.dim) * m.dim;
				sample.set(p.subarray(0, len), o);
				o += len;
			}
			const [coarse, fine] = await Promise.all([getF32(env.BUCKET, coarseKey), getF32(env.BUCKET, fineKey)]);
			const searcher = new Searcher(m.dim, m.k1, m.k2, coarse, fine);
			try {
				searcher.set_tree(bytes);
				ref.nearestRecall = +searcher.tree_recall(sample, beam).toFixed(5);
				ref.top3Recall = +searcher.tree_recall_top(sample, beam, 3).toFixed(5);
				const extra = [...new Set(req.recallBeams ?? [])].filter((b) => Number.isInteger(b) && b >= 1 && b !== beam).sort((a, b) => a - b);
				if (extra.length) {
					ref.recallByBeam = {};
					for (const b of extra) ref.recallByBeam[b] = {
						nearest: +searcher.tree_recall(sample, b).toFixed(5),
						top3: +searcher.tree_recall_top(sample, b, 3).toFixed(5)
					};
				}
			} finally {
				safeFree(searcher, epoch);
			}
			ref.sampleRows = n;
		}
	}
	return {
		bytes,
		ref
	};
}
async function stagePublish(env, body) {
	const { gen, coarseKey, fineKey, slabs, setCurrent, codebookKey, overwrite, tree: treeReq, ...rest } = body;
	if (!overwrite) {
		const revs = await listRevs(env, gen);
		if (revs.length) throw new HttpError(409, `${gen} already has manifest revs ${revs.map((r) => r.rev).join(",")}; use publish-rev to change defaults (or overwrite: true for a rebuild of the same gen)`);
	}
	const format = body.format ?? 2;
	if (format !== 1 && format !== 2 && format !== 3) throw new HttpError(400, "format must be 1, 2 or 3");
	if (format === 3 !== !!codebookKey) throw new HttpError(400, format === 3 ? "format 3 needs codebookKey" : `codebookKey given for format ${format}`);
	let codebook = null;
	let pqM = 0;
	if (codebookKey) {
		await ensureWasm();
		codebook = await getBuffer(env.BUCKET, codebookKey);
		const [cbDim, cbM] = pq_codebook_info(new Uint8Array(codebook));
		if (cbDim !== body.dim) throw new HttpError(400, `codebook dim ${cbDim} != ${body.dim}`);
		pqM = cbM;
	}
	const expectTiers = format === 3 ? `pq${pqM}+sq8` : format === 2 ? "sq4+sq8" : "sq8";
	const tiers = body.tiers ?? expectTiers;
	if (tiers !== expectTiers) throw new HttpError(400, `tiers ${tiers} does not match format ${format} (expected ${expectTiers})`);
	if (body.defaults) checkDefaults(body.defaults, treeReq ? {} : null);
	let slabTable = null;
	if (slabs) {
		const cells = body.k1 * body.k2;
		const table = new Array(cells).fill(null);
		for (const s of slabs) {
			if (!Number.isInteger(s.cell) || s.cell < 0 || s.cell >= cells) throw new HttpError(400, `slab cell ${s.cell} out of range`);
			if (!(s.scanLen > 0)) throw new HttpError(400, `slab ${s.cell} has no scanLen`);
			if (format >= 2 && !(s.rerankOff > 0)) throw new HttpError(400, `slab ${s.cell} has no rerankOff (v1 slab in a format ${format} generation?)`);
			if (format === 3 && s.count > 0 && s.pqM !== void 0 && s.pqM !== pqM) throw new HttpError(400, `slab ${s.cell} has pq_m ${s.pqM}, codebook has ${pqM}`);
			if (format === 1 && s.rerankOff !== 0) throw new HttpError(400, `slab ${s.cell} has a rerank tier in a format 1 generation`);
			table[s.cell] = [
				s.scanLen,
				s.rerankOff,
				s.count,
				s.bytes
			];
		}
		const missing = table.findIndex((t) => t === null);
		if (missing >= 0) throw new HttpError(400, `slabs is missing cell ${missing}`);
		slabTable = table;
	} else if (format >= 2) throw new HttpError(400, `format ${format} needs slabs (finalize results)`);
	const [coarse, fine] = await Promise.all([getBuffer(env.BUCKET, coarseKey), getBuffer(env.BUCKET, fineKey)]);
	await Promise.all([
		putBytes(env.BUCKET, `idx/${gen}/coarse.f32`, coarse),
		putBytes(env.BUCKET, `idx/${gen}/fine.f32`, fine),
		slabTable ? putBytes(env.BUCKET, `idx/${gen}/slabs.json`, JSON.stringify(slabTable)) : Promise.resolve(),
		codebook ? putBytes(env.BUCKET, `idx/${gen}/pq.bin`, codebook) : Promise.resolve()
	]);
	let tree;
	if (treeReq) {
		const notes = body.notes;
		const t = await buildTree(env, gen, body, coarseKey, fineKey, treeReq, notes?.cfg?.sampleGroups);
		tree = {
			key: treeKey(gen, 0),
			...t.ref
		};
		await putBytes(env.BUCKET, tree.key, t.bytes);
	}
	const manifest = {
		gen,
		...rest,
		format,
		tiers,
		...format === 3 ? { pq: { m: pqM } } : {},
		...tree ? { tree } : {},
		rev: 0,
		createdAt: (/* @__PURE__ */ new Date()).toISOString()
	};
	await putBytes(env.BUCKET, manifestKey(gen, 0), JSON.stringify(manifest));
	if (setCurrent !== false) await putBytes(env.BUCKET, CURRENT_KEY, JSON.stringify({
		gen,
		rev: 0
	}));
	return {
		...manifest,
		current: setCurrent !== false
	};
}
var slabRevOf = (row) => row && row.length > 4 ? row[4] : 0;
var CURRENT_KEY = "idx/current.json";
var manifestKey = (gen, rev) => rev === 0 ? `idx/${gen}/manifest.json` : `idx/${gen}/manifest.${rev}.json`;
var slabsTableKey = (gen, rev) => rev === 0 ? `idx/${gen}/slabs.json` : `idx/${gen}/slabs.${rev}.json`;
var slabKey = (gen, cell, slabRev = 0) => slabRev === 0 ? `idx/${gen}/slab/${pad5(cell)}` : `idx/${gen}/slab/${pad5(cell)}.${slabRev}`;
var tombKey = (gen, rev) => `idx/${gen}/tombstones.${rev}.u32`;
var deltaKey = (gen, seq) => `idx/${gen}/delta/${seq}.vibd`;
function manifestFiles(m) {
	if (m.files) return m.files;
	const g = m.gen;
	return {
		coarse: `idx/${g}/coarse.f32`,
		fine: `idx/${g}/fine.f32`,
		pq: (m.format ?? 1) === 3 ? `idx/${g}/pq.bin` : null,
		slabs: `idx/${g}/slabs.json`
	};
}
var decodeJson = (buf) => JSON.parse(new TextDecoder().decode(buf));
/** Every object under `prefix` (paginated R2 list). */
async function listAll(env, prefix) {
	const out = [];
	let cursor;
	do {
		const r = await env.BUCKET.list({
			prefix,
			cursor,
			limit: 1e3
		});
		out.push(...r.objects);
		cursor = r.truncated ? r.cursor : void 0;
	} while (cursor);
	return out;
}
/** Manifest revisions of `gen` that exist in R2, ascending, with their upload times. */
async function listRevs(env, gen) {
	const prefix = `idx/${gen}/manifest.`;
	const revs = [];
	for (const o of await listAll(env, prefix)) {
		const m = o.key.slice(prefix.length).match(/^(?:(\d+)\.)?json$/);
		if (m) revs.push({
			rev: m[1] ? Number(m[1]) : 0,
			uploaded: o.uploaded
		});
	}
	return revs.sort((a, b) => a.rev - b.rev);
}
/** The newest manifest revision of `gen` (the head writers build on). */
async function latestRev(env, gen) {
	const revs = await listRevs(env, gen);
	if (!revs.length) throw new HttpError(404, `no manifest for generation ${gen}`);
	return revs.at(-1).rev;
}
/** A manifest read straight from R2 (writers; legacy rev-0 manifests may have been republished, so not through the cache). */
async function readManifestR2(env, gen, rev) {
	const m = decodeJson(await getBuffer(env.BUCKET, manifestKey(gen, rev)));
	return {
		...m,
		rev: m.rev ?? rev
	};
}
async function readCurrent(env) {
	const o = await env.BUCKET.get(CURRENT_KEY);
	if (!o) return null;
	const j = await o.json();
	return {
		cur: {
			gen: j.gen,
			rev: j.rev ?? 0
		},
		etag: o.etag
	};
}
var currentPtr = null;
var CURRENT_TTL_MS = 3e4;
/** Head revision of generations other than the current one (`/query?gen=` without `rev`), same TTL. */
var genHeads = /* @__PURE__ */ new Map();
async function getCurrent(env) {
	if (currentPtr && Date.now() - currentPtr.at < CURRENT_TTL_MS) return currentPtr.cur;
	const r = await readCurrent(env);
	if (!r) throw new HttpError(404, "no published index");
	currentPtr = {
		cur: r.cur,
		at: Date.now()
	};
	return r.cur;
}
/**
* The (gen, rev) a query reads: `?gen=&rev=` explicitly, `?rev=` of the current generation, `?gen=` at its head
* (the current rev if it is the current generation), else idx/current.json.
*/
async function resolveTarget(env, genParam, revParam) {
	if (genParam && revParam !== null) return {
		gen: genParam,
		rev: revParam
	};
	const cur = await getCurrent(env);
	if (!genParam || genParam === cur.gen) return revParam !== null ? {
		gen: cur.gen,
		rev: revParam
	} : cur;
	const h = genHeads.get(genParam);
	if (h && Date.now() - h.at < CURRENT_TTL_MS) return {
		gen: genParam,
		rev: h.rev
	};
	const rev = await latestRev(env, genParam);
	genHeads.set(genParam, {
		rev,
		at: Date.now()
	});
	return {
		gen: genParam,
		rev
	};
}
var baseCache = /* @__PURE__ */ new Map();
var BASE_CACHE_SIZE = 1;
/** Parsed revisions per isolate (immutable, keyed `${gen}@${rev}`), most recently used last. */
var revCache = /* @__PURE__ */ new Map();
var REV_CACHE_SIZE = 8;
/** Parsed slabs tables by key (shared by the revisions between two compactions). */
var slabsTableCache = /* @__PURE__ */ new Map();
var SLABS_TABLE_CACHE_SIZE = 3;
var slabCache = /* @__PURE__ */ new Map();
var slabCacheBytes = 0;
var SLAB_CACHE_LIMIT = 16777216;
/**
* Delta segments by key (immutable). Queries share one copy per isolate; a segment beyond the budget is still
* shared by the queries that read it concurrently, then dropped.
*/
var deltaCache = /* @__PURE__ */ new Map();
var deltaCacheSizes = /* @__PURE__ */ new Map();
var deltaCacheBytes = 0;
var DELTA_CACHE_LIMIT = 41943040;
/** Distinct cells of each delta segment (from vib.delta_cells), by key: what beam-routed queries compute dots for. */
var deltaCellsCache = /* @__PURE__ */ new Map();
var DELTA_CELLS_CACHE_SIZE = 256;
/** Distinct cells (ascending) holding entries of the delta segment `key`, cached per isolate. */
function deltaCellsOf(key, bytes) {
	let cells = deltaCellsCache.get(key);
	if (cells) {
		lruTouch(deltaCellsCache, key, cells);
		return cells;
	}
	const pairs = delta_cells(bytes);
	cells = new Uint32Array(pairs.length / 2);
	for (let i = 0; i < cells.length; i++) cells[i] = pairs[2 * i];
	deltaCellsCache.set(key, cells);
	while (deltaCellsCache.size > DELTA_CELLS_CACHE_SIZE) deltaCellsCache.delete(deltaCellsCache.keys().next().value);
	return cells;
}
function lruTouch(m, k, v) {
	m.delete(k);
	m.set(k, v);
}
var CACHE_ORIGIN = "https://vib-bench.aiuitech.workers.dev/_cache/";
var CACHE_CONTROL = "public, max-age=31536000, immutable";
/** rr=runs: rerank winners of one slab whose records lie within this span are read with one range request. */
var COALESCE_SPAN = 65536;
/** rr=multi (default): winners within this span share one range of the multi-range request (i.e. adjacent records). */
var MULTI_COALESCE = 4096;
function cacheRequest(key, range) {
	const headers = {};
	if (range) headers.range = `bytes=${range.offset}-${range.offset + range.length - 1}`;
	return new Request(CACHE_ORIGIN + key.replace(/^idx\//, ""), {
		method: "GET",
		headers
	});
}
/** Stores a full immutable object in caches.default in the background. */
function putCache(ctx, key, buf) {
	const resp = new Response(buf.slice(0), { headers: {
		"content-type": "application/octet-stream",
		"content-length": String(buf.byteLength),
		"cache-control": CACHE_CONTROL,
		"accept-ranges": "bytes"
	} });
	ctx.waitUntil(caches.default.put(cacheRequest(key), resp).catch((e) => console.error(`cache.put ${key} failed: ${e}`)));
}
/**
* Reads an immutable index object (`idx/{gen}/...`) through caches.default.
* On a miss it reads R2 and stores a copy in the background via ctx.waitUntil.
*/
async function getCached(env, ctx, key) {
	const hit = await caches.default.match(cacheRequest(key));
	if (hit) return {
		buf: await hit.arrayBuffer(),
		outcome: "hit"
	};
	const buf = await getBuffer(env.BUCKET, key);
	putCache(ctx, key, buf);
	return {
		buf,
		outcome: "miss"
	};
}
async function getCachedOptional(env, ctx, key) {
	try {
		return await getCached(env, ctx, key);
	} catch (e) {
		if (e instanceof HttpError && e.status === 404) return null;
		throw e;
	}
}
/**
* Bounds a request's in-flight subrequests (cache.match + body read, R2 GETs). Workers keep at
* most 6 connections open per invocation and cancel stalled responses beyond that
* ("Response closed due to connection limit"), so reads are queued here instead.
*/
var Limiter = class {
	max;
	active = 0;
	queue = [];
	/** Low-priority waiters (speculative reads): started only when no normal read is waiting. */
	low = [];
	/** Reads retried after "connection limit" cancellations. */
	retries = 0;
	constructor(max) {
		this.max = max;
	}
	async run(fn, low = false) {
		if (this.active >= this.max) await new Promise((r) => (low ? this.low : this.queue).push(r));
		this.active++;
		try {
			return await fn();
		} finally {
			this.active--;
			(this.queue.shift() ?? this.low.shift())?.();
		}
	}
};
var SUBREQUEST_CONCURRENCY = 6;
/**
* In-flight Cache API reads per query. Measured on g2 (data/bench-latency.txt): Cache API reads are not held
* to the 6-connection limit (26 in flight: no "connection limit" cancellations), and 8-12 in flight cut
* phase 1 by ~10 ms at c=1..50 against 6. R2 reads (cache misses) still retry once on a cancellation.
*/
var QUERY_INFLIGHT = 12;
/** Runs one read; retries once if the runtime closed it for the connection limit, the connection dropped or R2 reported an internal error. */
async function readWithRetry(lim, fn, low = false) {
	try {
		return await lim.run(fn, low);
	} catch (e) {
		if (!(e instanceof Error && [
			"connection limit",
			"(10001)",
			"Network connection lost"
		].some((m) => e.message.includes(m)))) throw e;
		lim.retries++;
		return await lim.run(fn, low);
	}
}
/**
* Range read of a cached object. Returns null when the object is not cached. If the cache
* ignores the Range header (status 200) the full body is returned as `full` as well.
*/
async function cacheMatchRange(key, offset, length, rs) {
	const r = await caches.default.match(cacheRequest(key, {
		offset,
		length
	}));
	if (!r) {
		rs.miss++;
		return null;
	}
	const body = new Uint8Array(await r.arrayBuffer());
	if (r.status === 206) {
		if (body.byteLength !== length) throw new HttpError(500, `cache range ${key} [${offset}, +${length}) returned ${body.byteLength} bytes`);
		rs.r206++;
		return {
			bytes: body,
			full: null
		};
	}
	rs.r200++;
	if (body.byteLength < offset + length) throw new HttpError(500, `cached ${key} is ${body.byteLength} bytes, need ${offset + length}`);
	return {
		bytes: body.subarray(offset, offset + length),
		full: body
	};
}
/** Frees a loaded base's wasm objects, unless they belong to an instance that was reset since. */
function freeLoadedBase(b) {
	if (b.epoch !== wasmEpoch) return;
	safeFree(b.searcher, b.epoch);
	if (b.pq) safeFree(b.pq, b.epoch);
}
/** Removes a slot from the cache; its base is freed now, or by the last query still using it. */
function dropBaseSlot(slot) {
	if (slot.evicted) return;
	slot.evicted = true;
	if (baseCache.get(slot.key) === slot) baseCache.delete(slot.key);
	if (slot.users === 0) slot.promise.then(freeLoadedBase, () => {});
}
var baseKeyOf = (gen, f) => `${gen}|${f.coarse}|${f.fine}|${f.pq ?? "-"}`;
/** Centroid tree bytes by key (immutable, ~1 MB each), fetched through the cache layer before a query needs them. */
var treeBytes = /* @__PURE__ */ new Map();
var TREE_BYTES_CACHE_SIZE = 2;
var treeBytesReady = /* @__PURE__ */ new Map();
function fetchTreeBytes(env, ctx, key) {
	let p = treeBytes.get(key);
	if (p) {
		lruTouch(treeBytes, key, p);
		return p.then((r) => ({
			buf: r.buf,
			outcome: "mem"
		}));
	}
	p = getCached(env, ctx, key).then((r) => {
		const buf = new Uint8Array(r.buf);
		treeBytesReady.set(key, buf);
		return {
			buf,
			outcome: r.outcome
		};
	});
	treeBytes.set(key, p);
	const mine = p;
	p.catch(() => treeBytes.get(key) === mine && treeBytes.delete(key));
	while (treeBytes.size > TREE_BYTES_CACHE_SIZE) {
		const old = treeBytes.keys().next().value;
		treeBytes.delete(old);
		treeBytesReady.delete(old);
	}
	return p;
}
/**
* Makes `base.searcher` hold `tree` (synchronous; the bytes must have been fetched with fetchTreeBytes). Called
* right before routing, with no await in between, so concurrent queries of revisions with different trees cannot
* route through the wrong one.
*/
function useTree(base, tree) {
	if (base.tree?.key === tree.key) return;
	const bytes = treeBytesReady.get(tree.key);
	if (!bytes) throw new HttpError(500, `tree ${tree.key} not loaded`);
	base.searcher.set_tree(bytes);
	if (base.searcher.tree_groups() !== tree.groups) throw new HttpError(500, `${tree.key} has ${base.searcher.tree_groups()} groups, manifest says ${tree.groups}`);
	base.tree = tree;
}
/** Parses a PQ codebook and checks it against the manifest. */
function parseCodebook(buf, m, key) {
	const epoch = wasmEpoch;
	const pq = new PqCodebook(new Uint8Array(buf));
	if (pq.dim() !== m.dim || m.pq && pq.m() !== m.pq.m) {
		const msg = `${key} is dim ${pq.dim()} m ${pq.m()}, manifest dim ${m.dim} pq ${JSON.stringify(m.pq)}`;
		safeFree(pq, epoch);
		throw new HttpError(500, msg);
	}
	return pq;
}
async function buildBase(env, ctx, rs) {
	basesLoaded++;
	writerMeans = null;
	await ensureWasm();
	const epoch = wasmEpoch;
	const t0 = Date.now();
	const { manifest, files } = rs;
	const tree = manifest.tree ?? null;
	const [c, f, pqObj, treeObj] = await Promise.all([
		getCached(env, ctx, files.coarse),
		getCached(env, ctx, files.fine),
		files.pq ? getCached(env, ctx, files.pq) : Promise.resolve(null),
		tree ? fetchTreeBytes(env, ctx, tree.key) : Promise.resolve(null)
	]);
	for (const [name, r] of [[files.coarse, c], [files.fine, f]]) if (r.buf.byteLength % 4 !== 0) throw new HttpError(500, `object ${name} is not f32-aligned`);
	if ((manifest.format ?? 1) === 3 && !pqObj) throw new HttpError(500, `format 3 generation ${rs.gen} has no PQ codebook`);
	const pq = pqObj ? parseCodebook(pqObj.buf, manifest, files.pq) : null;
	let searcher = null;
	try {
		searcher = new Searcher(manifest.dim, manifest.k1, manifest.k2, new Float32Array(c.buf), new Float32Array(f.buf));
		if (tree && treeObj) {
			searcher.set_tree(treeObj.buf);
			if (searcher.tree_groups() !== tree.groups) throw new HttpError(500, `${tree.key} has ${searcher.tree_groups()} groups, manifest says ${tree.groups}`);
		}
	} catch (e) {
		if (searcher) safeFree(searcher, epoch);
		if (pq) safeFree(pq, epoch);
		throw e;
	}
	return {
		key: rs.baseKey,
		gen: rs.gen,
		searcher,
		pq,
		tree,
		epoch,
		loadMs: Date.now() - t0,
		loadCache: {
			coarse: c.outcome,
			fine: f.outcome,
			...pqObj ? { pq: pqObj.outcome } : {},
			...treeObj ? { tree: treeObj.outcome } : {}
		}
	};
}
/** The loaded base of a revision, held until `release()` (call it once the query is done). */
async function acquireBase(env, ctx, rs) {
	const key = rs.baseKey;
	let slot = baseCache.get(key);
	if (slot) lruTouch(baseCache, key, slot);
	else {
		const promise = buildBase(env, ctx, rs);
		const fresh = {
			key,
			promise,
			users: 0,
			evicted: false
		};
		slot = fresh;
		baseCache.set(key, fresh);
		promise.catch(() => {
			if (baseCache.get(key) === fresh) baseCache.delete(key);
		});
		while (baseCache.size > BASE_CACHE_SIZE) dropBaseSlot(baseCache.values().next().value);
	}
	const held = slot;
	held.users++;
	let released = false;
	const release = () => {
		if (released) return;
		released = true;
		held.users--;
		if (held.evicted && held.users === 0) held.promise.then(freeLoadedBase, () => {});
	};
	try {
		return {
			base: await held.promise,
			release
		};
	} catch (e) {
		release();
		throw e;
	}
}
async function loadSlabsTable(env, ctx, key, optional) {
	const hit = slabsTableCache.get(key);
	if (hit) {
		lruTouch(slabsTableCache, key, hit);
		return {
			table: await hit,
			outcome: "mem"
		};
	}
	let outcome = "none";
	const p = (optional ? getCachedOptional(env, ctx, key) : getCached(env, ctx, key)).then((r) => {
		if (!r) return null;
		outcome = r.outcome;
		return decodeJson(r.buf);
	});
	slabsTableCache.set(key, p);
	p.catch(() => slabsTableCache.get(key) === p && slabsTableCache.delete(key));
	while (slabsTableCache.size > SLABS_TABLE_CACHE_SIZE) slabsTableCache.delete(slabsTableCache.keys().next().value);
	return {
		table: await p,
		outcome
	};
}
/** Sorted unique union of id sets as a tombstone buffer (`u32` LE). */
function idUnion(sets) {
	const all = new Uint32Array(sets.reduce((a, s) => a + s.length, 0));
	let o = 0;
	for (const s of sets) {
		all.set(s, o);
		o += s.length;
	}
	all.sort();
	let n = 0;
	for (let i = 0; i < all.length; i++) if (i === 0 || all[i] !== all[i - 1]) all[n++] = all[i];
	return new Uint8Array(all.buffer, 0, n * 4);
}
function buildTombState(sets) {
	return {
		sets,
		total: sets.reduce((a, s) => a + s.ids.length, 0),
		unions: /* @__PURE__ */ new Map()
	};
}
/**
* Tombstone buffer for slab hits of a slab with revision `slabRev`: the union of the sets with rev >= slabRev.
* A slab written by a compaction that read manifest rev R has slabRev R + 1 and already excludes every id of
* the sets with rev <= R; rev-0 slabs are hidden by every set.
*/
function tombFor(t, slabRev) {
	let u = t.unions.get(slabRev);
	if (!u) {
		u = idUnion(t.sets.filter((s) => s.rev >= slabRev).map((s) => s.ids));
		t.unions.set(slabRev, u);
	}
	return u;
}
async function loadRevState(env, ctx, gen, rev) {
	const t0 = Date.now();
	const m = rev === 0 ? await env.BUCKET.get(manifestKey(gen, 0)).then(async (o) => o ? {
		buf: await o.arrayBuffer(),
		outcome: "none"
	} : null) : await getCachedOptional(env, ctx, manifestKey(gen, rev));
	if (!m) throw new HttpError(404, `no manifest ${manifestKey(gen, rev)}`);
	const manifest = decodeJson(m.buf);
	if (manifest.gen !== gen) throw new HttpError(500, `${manifestKey(gen, rev)} is for generation ${manifest.gen}`);
	const files = manifestFiles(manifest);
	const format = manifest.format ?? 1;
	const tombRefs = manifest.tombstones ?? [];
	const [s, tombs] = await Promise.all([files.slabs ? loadSlabsTable(env, ctx, files.slabs, format < 2 && !manifest.files) : Promise.resolve({
		table: null,
		outcome: "none"
	}), Promise.all(tombRefs.map(async (t) => {
		const r = await getCached(env, ctx, t.key);
		if (r.buf.byteLength % 4 !== 0) throw new HttpError(500, `${t.key} is not u32-aligned`);
		return {
			rev: t.rev,
			ids: new Uint32Array(r.buf)
		};
	}))]);
	if (format >= 2 && !s.table) throw new HttpError(500, `${files.slabs} missing for format ${format}`);
	if (s.table && s.table.length !== manifest.k1 * manifest.k2) throw new HttpError(500, `${files.slabs} has ${s.table.length} cells, expected ${manifest.k1 * manifest.k2}`);
	return {
		gen,
		rev,
		manifest,
		files,
		baseKey: baseKeyOf(gen, files),
		slabs: s.table,
		deltas: manifest.deltas ?? [],
		tomb: tombs.length ? buildTombState(tombs) : null,
		loadMs: Date.now() - t0,
		loadCache: {
			manifest: m.outcome,
			slabs: s.outcome
		}
	};
}
/** The parsed revision (gen, rev), cached per isolate (revisions are immutable). */
function getRevState(env, ctx, gen, rev) {
	const key = `${gen}@${rev}`;
	const hit = revCache.get(key);
	if (hit) {
		lruTouch(revCache, key, hit);
		return hit;
	}
	const p = loadRevState(env, ctx, gen, rev);
	revCache.set(key, p);
	p.catch(() => revCache.get(key) === p && revCache.delete(key));
	while (revCache.size > REV_CACHE_SIZE) revCache.delete(revCache.keys().next().value);
	return p;
}
/**
* A delta segment through the isolate cache and the Cache API. Concurrent queries share one in-flight read, so
* the read is started at once and NOT queued in the first query's Limiter: a request awaiting I/O that another
* request has not started yet is cancelled by the runtime as hung (seen with 5 MB segments at c=50).
*/
async function getDelta(env, ctx, ref) {
	const hit = deltaCache.get(ref.key);
	if (hit) {
		lruTouch(deltaCache, ref.key, hit);
		return {
			bytes: await hit,
			outcome: "mem"
		};
	}
	let outcome = "none";
	const p = getCached(env, ctx, ref.key).then((r) => {
		outcome = r.outcome;
		return new Uint8Array(r.buf);
	});
	deltaCache.set(ref.key, p);
	deltaCacheSizes.set(ref.key, ref.bytes);
	deltaCacheBytes += ref.bytes;
	const drop = (k) => {
		deltaCache.delete(k);
		deltaCacheBytes -= deltaCacheSizes.get(k) ?? 0;
		deltaCacheSizes.delete(k);
	};
	p.catch(() => deltaCache.get(ref.key) === p && drop(ref.key));
	for (const k of deltaCache.keys()) {
		if (deltaCacheBytes <= DELTA_CACHE_LIMIT || k === ref.key) break;
		drop(k);
	}
	return {
		bytes: await p,
		outcome
	};
}
function memCachePut(key, buf) {
	if (buf.byteLength >= SLAB_CACHE_LIMIT / 4) return;
	if (slabCacheBytes + buf.byteLength > SLAB_CACHE_LIMIT) {
		slabCache.clear();
		slabCacheBytes = 0;
	}
	slabCache.set(key, buf);
	slabCacheBytes += buf.byteLength;
}
/** v1: the whole slab object. */
async function fetchSlab(env, ctx, key, mode) {
	if (mode === "edge") {
		const r = await getCached(env, ctx, key);
		return {
			buf: new Uint8Array(r.buf),
			outcome: r.outcome
		};
	}
	if (mode === "mem") {
		const hit = slabCache.get(key);
		if (hit) return {
			buf: hit,
			outcome: "mem"
		};
	}
	const buf = new Uint8Array(await getBuffer(env.BUCKET, key));
	if (mode === "mem") memCachePut(key, buf);
	return {
		buf,
		outcome: mode === "mem" ? "miss" : "none"
	};
}
var DEFAULT_DEBUG = {
	mode: "",
	scanRepeat: 1,
	copyRepeat: 0,
	copy: 0,
	inflight: QUERY_INFLIGHT,
	rr: "multi",
	coalesce: COALESCE_SPAN,
	spanMax: 262144,
	spec: 1,
	specAt: 6,
	specMargin: 100
};
function enumParam(url, name, allowed, def) {
	const raw = url.searchParams.get(name);
	if (raw === null || raw === "") return def;
	if (!allowed.includes(raw)) throw new HttpError(400, `${name} must be one of ${allowed.join("|")}`);
	return raw;
}
function parseDebug(url) {
	const d = DEFAULT_DEBUG;
	return {
		mode: enumParam(url, "mode", [
			"",
			"fetchonly",
			"noop"
		], d.mode),
		scanRepeat: Math.min(intParam(url, "scanRepeat", d.scanRepeat, 1), 16),
		copyRepeat: Math.min(intParam(url, "copyRepeat", d.copyRepeat, 0), 16),
		copy: Math.min(intParam(url, "copy", d.copy, 0), 1),
		inflight: Math.min(intParam(url, "inflight", d.inflight, 1), 32),
		rr: enumParam(url, "rr", [
			"runs",
			"span",
			"multi"
		], d.rr),
		coalesce: intParam(url, "coalesce", url.searchParams.get("rr") === "runs" || url.searchParams.get("rr") === "span" ? d.coalesce : MULTI_COALESCE, 0),
		spanMax: intParam(url, "spanMax", d.spanMax, 0),
		spec: Math.min(intParam(url, "spec", d.spec, 0), 1),
		specAt: intParam(url, "specAt", d.specAt, 0),
		specMargin: Math.min(intParam(url, "specMargin", d.specMargin, 100), 400)
	};
}
function parseSlabHeader(b) {
	if (b.byteLength < 64) throw new HttpError(500, `slab header needs 64 bytes, have ${b.byteLength}`);
	const dv = new DataView(b.buffer, b.byteOffset, 64);
	if (dv.getUint32(0, true) !== 1396853078) throw new HttpError(500, "bad slab magic");
	const version = dv.getUint16(4, true);
	const dim = dv.getUint32(8, true);
	const count = dv.getUint32(12, true);
	const centroidOff = dv.getUint32(20, true);
	if (version === 2 || version === 3) return {
		version,
		dim,
		count,
		centroidOff,
		scanLen: dv.getUint32(48, true),
		rerankOff: dv.getUint32(36, true),
		recLen: 4 + dim,
		pqM: version === 3 ? dv.getUint32(32, true) : 0
	};
	if (version === 1) return {
		version,
		dim,
		count,
		centroidOff,
		scanLen: dv.getUint32(36, true),
		rerankOff: 0,
		recLen: 0,
		pqM: 0
	};
	throw new HttpError(500, `unknown slab version ${version}`);
}
function slabHdrFromWasm(prefix) {
	const a = slab_info(prefix);
	return {
		version: a[4],
		dim: a[0],
		count: a[1],
		centroidOff: 64,
		scanLen: a[5],
		rerankOff: a[6],
		recLen: a[7],
		pqM: a[8]
	};
}
/** The cell mean at `centroidOff` (copied out, so it outlives the prefix). */
function centroidFromBytes(b, h) {
	const s = b.slice(h.centroidOff, h.centroidOff + h.dim * 4);
	return new Float32Array(s.buffer, 0, h.dim);
}
/**
* Phase 1 read. v2 + edge: Range request for the scan prefix against the cached full object;
* on a miss GET the whole object from R2, cache it in the background, and keep it in memory.
*/
async function fetchScanSource(env, ctx, rv, cell, mode, rs, lim) {
	return readWithRetry(lim, () => fetchScanSourceOnce(env, ctx, rv, cell, mode, rs));
}
async function fetchScanSourceOnce(env, ctx, rv, cell, mode, rs) {
	if (!rv.slabs) {
		const { buf, outcome } = await fetchSlab(env, ctx, slabKey(rv.gen, cell), mode);
		return {
			prefix: buf,
			full: null,
			outcome,
			bytes: buf.byteLength
		};
	}
	const meta = rv.slabs[cell];
	if (!meta) throw new HttpError(500, `${rv.files.slabs} has no entry for cell ${cell}`);
	const scanLen = meta[0];
	const key = slabKey(rv.gen, cell, slabRevOf(meta));
	if (mode === "edge") {
		const hit = await cacheMatchRange(key, 0, scanLen, rs);
		if (hit) return {
			prefix: hit.bytes,
			full: hit.full,
			outcome: "hit",
			bytes: (hit.full ?? hit.bytes).byteLength
		};
		const buf = await getBuffer(env.BUCKET, key);
		putCache(ctx, key, buf);
		const full = new Uint8Array(buf);
		return {
			prefix: full.subarray(0, scanLen),
			full,
			outcome: "miss",
			bytes: full.byteLength
		};
	}
	const memKey = `${key}#scan`;
	if (mode === "mem") {
		const hit = slabCache.get(memKey);
		if (hit) return {
			prefix: hit,
			full: null,
			outcome: "mem",
			bytes: 0
		};
	}
	const prefix = new Uint8Array(await getBuffer(env.BUCKET, key, {
		offset: 0,
		length: scanLen
	}));
	if (mode === "mem") memCachePut(memKey, prefix);
	return {
		prefix,
		full: null,
		outcome: mode === "mem" ? "miss" : "none",
		bytes: prefix.byteLength
	};
}
var latin1 = new TextDecoder("latin1");
/** Splits a multipart/byteranges body into its parts (RFC 9110 §14.6). */
function parseMultipart(body, boundary) {
	const parts = [];
	const delim = new TextEncoder().encode(`--${boundary}`);
	const find = (pat, from) => {
		outer: for (let i = from; i + pat.length <= body.length; i++) {
			for (let j = 0; j < pat.length; j++) if (body[i + j] !== pat[j]) continue outer;
			return i;
		}
		return -1;
	};
	const crlf2 = new Uint8Array([
		13,
		10,
		13,
		10
	]);
	let pos = find(delim, 0);
	while (pos >= 0) {
		const after = pos + delim.length;
		if (body[after] === 45 && body[after + 1] === 45) break;
		const hdrEnd = find(crlf2, after);
		if (hdrEnd < 0) throw new HttpError(500, "multipart: unterminated part headers");
		const headers = latin1.decode(body.subarray(after, hdrEnd));
		const m = headers.match(/content-range:\s*bytes\s+(\d+)-(\d+)\//i);
		if (!m) throw new HttpError(500, `multipart: part without Content-Range: ${headers.slice(0, 200)}`);
		const start = Number(m[1]);
		const end = Number(m[2]);
		const dataStart = hdrEnd + 4;
		const dataEnd = dataStart + (end - start + 1);
		if (dataEnd > body.length) throw new HttpError(500, "multipart: part runs past the body");
		parts.push({
			start,
			end,
			bytes: body.subarray(dataStart, dataEnd)
		});
		pos = find(delim, dataEnd);
	}
	return parts;
}
/**
* Several ranges of one cached object in ONE Cache API request (`Range: bytes=a-b, c-d`).
* Handles a multipart/byteranges 206, a single-part 206 (ranges merged by the server) and a 200
* (Range ignored). Returns one Uint8Array per requested range, or null when the object is not cached.
*/
async function cacheMatchMulti(key, ranges, rs) {
	const hdr = "bytes=" + ranges.map((r) => `${r.offset}-${r.offset + r.length - 1}`).join(", ");
	const r = await caches.default.match(new Request(cacheRequest(key).url, {
		method: "GET",
		headers: { range: hdr }
	}));
	if (!r) {
		rs.miss++;
		return null;
	}
	const body = new Uint8Array(await r.arrayBuffer());
	let parts;
	let multipart = false;
	const ct = r.headers.get("content-type") ?? "";
	if (r.status === 206 && /multipart\/byteranges/i.test(ct)) {
		const b = ct.match(/boundary=("?)([^";]+)\1/i);
		if (!b) throw new HttpError(500, `multipart without boundary: ${ct}`);
		parts = parseMultipart(body, b[2]);
		multipart = true;
		rs.r206++;
	} else if (r.status === 206) {
		const m = (r.headers.get("content-range") ?? "").match(/bytes\s+(\d+)-(\d+)\//);
		if (!m) throw new HttpError(500, `206 without Content-Range for ${key}`);
		parts = [{
			start: Number(m[1]),
			end: Number(m[2]),
			bytes: body
		}];
		rs.r206++;
	} else {
		parts = [{
			start: 0,
			end: body.length - 1,
			bytes: body
		}];
		rs.r200++;
	}
	return {
		out: ranges.map((rg) => {
			const p = parts.find((p) => p.start <= rg.offset && rg.offset + rg.length - 1 <= p.end);
			if (!p) throw new HttpError(500, `range ${rg.offset}+${rg.length} of ${key} missing from the response (status ${r.status}, ${parts.length} parts)`);
			return p.bytes.subarray(rg.offset - p.start, rg.offset - p.start + rg.length);
		}),
		multipart,
		wireBytes: body.byteLength
	};
}
/** Parses packed hits `[count u32][id u32, score f32, idx u32]*`. */
function parseHits(packed) {
	const dv = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
	const n = dv.getUint32(0, true);
	const out = new Array(n);
	for (let i = 0; i < n; i++) {
		const o = 4 + i * 12;
		out[i] = {
			id: dv.getUint32(o, true),
			score: dv.getFloat32(o + 4, true),
			idx: dv.getUint32(o + 8, true)
		};
	}
	return out;
}
/** Query limits: k and rerank size wasm heaps per probed slab; a query body holds at most 65536 f32 values. */
var MAX_K = 1e3;
var MAX_RERANK = 4096;
var MAX_QUERY_BYTES = 262144;
function intParam(url, name, def, min = 0) {
	return optIntParam(url, name, min) ?? def;
}
/** An integer query parameter, or null when absent (or empty). */
function optIntParam(url, name, min = 0) {
	const raw = url.searchParams.get(name);
	if (raw === null || raw === "") return null;
	const v = Number(raw);
	if (!Number.isInteger(v) || v < min) throw new HttpError(400, `${name} must be an integer >= ${min}`);
	return v;
}
async function handleQuery(request, env, ctx, url) {
	const tStart = Date.now();
	const k = intParam(url, "k", 10, 1);
	const nprobe = optIntParam(url, "nprobe", 1);
	const probeCoarse = optIntParam(url, "pc", 0);
	const rerank = optIntParam(url, "rerank", 0);
	const routeRaw = url.searchParams.get("route");
	if (routeRaw !== null && routeRaw !== "" && !ROUTE_MODES.includes(routeRaw)) throw new HttpError(400, `route must be ${ROUTE_MODES.join("|")}`);
	const route = routeRaw ? routeRaw : null;
	const beam = optIntParam(url, "beam", 1);
	if (k > MAX_K) throw new HttpError(400, `k must be <= ${MAX_K}`);
	if (rerank !== null && rerank > MAX_RERANK) throw new HttpError(400, `rerank must be <= ${MAX_RERANK}`);
	const cacheParam = url.searchParams.get("cache") ?? "edge";
	if (cacheParam !== "none" && cacheParam !== "mem" && cacheParam !== "edge") throw new HttpError(400, "cache must be none|mem|edge");
	const dbg = parseDebug(url);
	const { gen, rev } = await resolveTarget(env, url.searchParams.get("gen"), optIntParam(url, "rev", 0));
	const tGen = Date.now();
	let q;
	if ((request.headers.get("content-type") ?? "").includes("octet-stream")) {
		const buf = await request.arrayBuffer();
		if (buf.byteLength > MAX_QUERY_BYTES) throw new HttpError(413, `query body must be <= ${MAX_QUERY_BYTES} bytes`);
		if (buf.byteLength % 4) throw new HttpError(400, "query body must be little-endian f32 values");
		q = new Float32Array(buf);
	} else {
		const body = await readJson(request);
		if (!Array.isArray(body?.vector) || body.vector.length > MAX_QUERY_BYTES / 4 || !body.vector.every((x) => typeof x === "number")) throw new HttpError(400, "vector must be an array of numbers");
		q = Float32Array.from(body.vector);
	}
	let norm2 = 0;
	for (const x of q) {
		if (!Number.isFinite(x)) throw new HttpError(400, "query vector must be finite");
		norm2 += x * x;
	}
	if (!(norm2 > 0)) throw new HttpError(400, "query vector must have nonzero norm");
	const p = {
		k,
		nprobe,
		probeCoarse,
		route,
		beam,
		mode: cacheParam,
		rerank,
		gen,
		rev,
		tStart,
		tGen,
		dbg
	};
	return withWasmRecovery("query", (attempt) => runQuery(request, env, ctx, q, p, attempt));
}
async function runQuery(request, env, ctx, qRaw, p, attempt) {
	const rv = await getRevState(env, ctx, p.gen, p.rev);
	const { base, release } = await acquireBase(env, ctx, rv);
	const scratch = {
		table: null,
		epoch: wasmEpoch
	};
	try {
		if (rv.manifest.tree && base.tree?.key !== rv.manifest.tree.key) await fetchTreeBytes(env, ctx, rv.manifest.tree.key);
		return await runQueryOn(rv, base, request, env, ctx, qRaw, p, attempt, scratch);
	} finally {
		if (scratch.table) safeFree(scratch.table, scratch.epoch);
		release();
	}
}
async function runQueryOn(rv, base, request, env, ctx, qRaw, p, attempt, scratch) {
	const { k, mode, gen, tStart, tGen, dbg } = p;
	const manifest = rv.manifest;
	if (qRaw.length !== manifest.dim) throw new HttpError(400, `query dim ${qRaw.length} != ${manifest.dim}`);
	const defaults = manifest.defaults ?? LEGACY_DEFAULTS;
	const nprobe = p.nprobe ?? defaults.nprobe;
	const probeCoarse = p.probeCoarse ?? defaults.pc;
	const rerankParam = p.rerank ?? defaults.rerank;
	const source = (v) => v !== null ? "query" : manifest.defaults ? "manifest" : "legacy";
	if (nprobe > manifest.k1 * manifest.k2) throw new HttpError(400, `nprobe must be <= ${manifest.k1 * manifest.k2} (k1*k2)`);
	if (probeCoarse > manifest.k1) throw new HttpError(400, `pc must be <= ${manifest.k1} (k1)`);
	const q = normalize_vector(qRaw);
	const dim = manifest.dim;
	const format = manifest.format ?? (rv.slabs ? 2 : 1);
	const doRerank = format >= 2 && rv.slabs !== null && rerankParam > 0 && dbg.mode !== "fetchonly";
	const T = doRerank ? Math.max(k, rerankParam) : k;
	const flat = probeCoarse === 0 || probeCoarse >= manifest.k1;
	const routeMode = p.route ?? defaults.route ?? "flat";
	const useBeam = flat && routeMode === "beam";
	const tree = manifest.tree ?? null;
	if (routeMode === "beam" && !tree) throw new HttpError(400, `route=beam needs a centroid tree (${gen} rev ${rv.rev} has none)`);
	const beam = useBeam ? p.beam ?? defaults.beam ?? tree.beam : null;
	if (beam !== null && beam > tree.groups) throw new HttpError(400, `beam must be <= ${tree.groups} (tree groups)`);
	const deltas = dbg.mode === "noop" ? [] : rv.deltas;
	const tomb = rv.tomb;
	const lim = new Limiter(dbg.inflight);
	const deltaReads = deltas.map((d) => getDelta(env, ctx, d));
	const tRoute = Date.now();
	const cellDots = flat && !useBeam ? base.searcher.cell_dots(q) : null;
	if (useBeam) useTree(base, tree);
	const cells = Array.from(useBeam ? base.searcher.route_beam(q, beam, nprobe) : flat ? base.searcher.route_from_dots(cellDots, nprobe) : base.searcher.route(q, probeCoarse, nprobe));
	const nCells = manifest.k1 * manifest.k2;
	let qcPartial = null;
	let dotsComputed = cellDots ? nCells : 0;
	/** scan_delta's `qc` for one segment: the full table, or the partial one with this segment's cells filled in. */
	const deltaQc = (d, bytes) => {
		if (cellDots) return cellDots;
		qcPartial ??= {
			qc: new Float32Array(nCells),
			filled: new Uint8Array(nCells)
		};
		const { qc, filled } = qcPartial;
		const missing = deltaCellsOf(d.key, bytes).filter((c) => !filled[c]);
		if (missing.length) {
			const dots = base.searcher.cell_dots_for(q, missing);
			missing.forEach((c, i) => {
				qc[c] = dots[i];
				filled[c] = 1;
			});
			dotsComputed += missing.length;
		}
		return qc;
	};
	const table = base.pq && dbg.mode !== "noop" ? scratch.table = PqTable.from_codebook(q, base.pq) : null;
	const tFetch = Date.now();
	if (dbg.mode === "noop") return json({
		gen,
		rev: rv.rev,
		ids: [],
		scores: [],
		cells,
		stats: {
			scanned: 0,
			slabBytes: 0,
			cacheHits: 0,
			cacheMisses: 0,
			debug: dbg
		},
		timing: {
			total: Date.now() - tStart,
			gen: tGen - tStart,
			load: tRoute - tStart,
			fetch: 0
		}
	});
	const rs = {
		r206: 0,
		r200: 0,
		miss: 0
	};
	const packed = [];
	/** Best scan-tier copy per id (same rule as vib.merge), remembering its cell. */
	const cand = /* @__PURE__ */ new Map();
	/** Rerank records of local winners from slabs that were read whole: `${cell}:${idx}`. */
	const local = /* @__PURE__ */ new Map();
	const slabInfo = /* @__PURE__ */ new Map();
	const versions = /* @__PURE__ */ new Set();
	let scanned = 0;
	let slabBytes = 0;
	let scanBytes = 0;
	let cacheHits = 0;
	let cacheMisses = 0;
	let tLastSlab = tFetch;
	let rerankReads = 0;
	let rerankBytes = 0;
	let rerankMultipart = 0;
	let specReadN = 0;
	let tombDropped = 0;
	const scanStats = /* @__PURE__ */ new Uint32Array(2);
	const noTomb = /* @__PURE__ */ new Uint8Array(0);
	/**
	* Reads the rerank records `idxs` (ascending) of one slab: one multi-range request (rr=multi), else
	* range reads of runs within `coalesce` bytes (or one span within `spanMax`, rr=span).
	*/
	const fetchRecords = async (cell, idxs, low) => {
		const si = slabInfo.get(cell);
		const recLen = si.recLen;
		const out = /* @__PURE__ */ new Map();
		if (!idxs.length) return out;
		const runs = [];
		const spanBytes = (idxs.at(-1) - idxs[0] + 1) * recLen;
		const limit = dbg.rr === "span" && spanBytes <= dbg.spanMax ? spanBytes : dbg.coalesce;
		for (const i of idxs) {
			const run = runs.at(-1);
			if (run && (i - run.a + 1) * recLen <= limit) {
				run.b = i;
				run.members.push(i);
			} else runs.push({
				a: i,
				b: i,
				members: [i]
			});
		}
		const key = si.key;
		const place = (run, bytes) => {
			for (const i of run.members) out.set(i, bytes.subarray((i - run.a) * recLen, (i - run.a + 1) * recLen));
		};
		const rangeOf = (run) => ({
			offset: si.rerankOff + run.a * recLen,
			length: (run.b - run.a + 1) * recLen
		});
		const readRun = async (run) => {
			const { offset, length } = rangeOf(run);
			const bytes = await readWithRetry(lim, async () => {
				return (mode === "edge" ? await cacheMatchRange(key, offset, length, rs) : null)?.bytes ?? new Uint8Array(await getBuffer(env.BUCKET, key, {
					offset,
					length
				}));
			}, low);
			rerankReads++;
			if (low) specReadN++;
			rerankBytes += length;
			place(run, bytes);
		};
		if (dbg.rr === "multi" && runs.length > 1 && mode === "edge") {
			const got = await readWithRetry(lim, () => cacheMatchMulti(key, runs.map(rangeOf), rs), low);
			if (got) {
				rerankReads++;
				if (low) specReadN++;
				if (got.multipart) rerankMultipart++;
				rerankBytes += got.wireBytes;
				runs.forEach((run, i) => place(run, got.out[i]));
				return out;
			}
		}
		await Promise.all(runs.map(readRun));
		return out;
	};
	let slabsDone = 0;
	let specFired = false;
	/** Speculative reads per cell (records by entry index). Reads for cells that end up without winners are not awaited. */
	const specReads = /* @__PURE__ */ new Map();
	let specRecords = 0;
	const speculate = () => {
		const top = [...cand.values()].sort((a, b) => b.score - a.score).slice(0, Math.ceil(T * dbg.specMargin / 100));
		const byCell = /* @__PURE__ */ new Map();
		for (const c of top) {
			if (local.has(`${c.cell}:${c.idx}`)) continue;
			let l = byCell.get(c.cell);
			if (!l) byCell.set(c.cell, l = []);
			l.push(c.idx);
		}
		for (const [cell, idxs] of byCell) {
			idxs.sort((a, b) => a - b);
			specRecords += idxs.length;
			specReads.set(cell, fetchRecords(cell, idxs, true).catch(() => null));
		}
	};
	const slabPhase = Promise.all(cells.map(async (cell) => {
		const src = await fetchScanSource(env, ctx, rv, cell, mode, rs, lim);
		tLastSlab = Date.now();
		if (src.outcome === "hit" || src.outcome === "mem") cacheHits++;
		else if (src.outcome === "miss") cacheMisses++;
		slabBytes += src.bytes;
		if (dbg.mode === "fetchonly") {
			scanBytes += src.prefix.byteLength;
			return;
		}
		const prefix = src.prefix;
		scanBytes += prefix.byteLength;
		const info = dbg.copy ? slabHdrFromWasm(prefix) : parseSlabHeader(prefix);
		for (let r = 0; r < dbg.copyRepeat; r++) slab_info(prefix);
		const slabRev = rv.slabs ? slabRevOf(rv.slabs[cell]) : 0;
		const skip = tomb ? tombFor(tomb, slabRev) : noTomb;
		let hits;
		if (info.version === 3) {
			if (!table) throw new HttpError(500, `slab ${cell} is v3 but ${gen} has no PQ codebook`);
			hits = scan_pq(prefix, table, q, T, skip, scanStats);
			for (let r = 1; r < dbg.scanRepeat; r++) hits = scan_pq(prefix, table, q, T, skip, scanStats);
		} else {
			hits = scan(prefix, q, T, skip, scanStats);
			for (let r = 1; r < dbg.scanRepeat; r++) hits = scan(prefix, q, T, skip, scanStats);
		}
		tombDropped += scanStats[1];
		const centroid = doRerank && info.version >= 2 ? dbg.copy ? slab_centroid(prefix) : centroidFromBytes(prefix, info) : null;
		versions.add(info.version);
		scanned += info.count;
		packed.push(hits);
		if (!doRerank || info.version < 2) return;
		const { rerankOff, recLen } = info;
		if (recLen !== 4 + dim) throw new HttpError(500, `slab ${cell}: rerank record length ${recLen} != ${4 + dim}`);
		slabInfo.set(cell, {
			centroid,
			rerankOff,
			recLen,
			key: slabKey(gen, cell, slabRev)
		});
		for (const h of parseHits(hits)) {
			const prev = cand.get(h.id);
			if (!prev || h.score > prev.score) cand.set(h.id, {
				cell,
				idx: h.idx,
				score: h.score
			});
			if (src.full) {
				const o = rerankOff + h.idx * recLen;
				local.set(`${cell}:${h.idx}`, src.full.slice(o, o + recLen));
			}
		}
		slabsDone++;
		if (dbg.spec && !specFired && cells.length - slabsDone <= dbg.specAt) {
			specFired = true;
			speculate();
		}
	}));
	const deltaHits = [];
	let deltaBytes = 0;
	let deltaScanned = 0;
	let deltaDropped = 0;
	let deltaMem = 0;
	let tDeltaEnd = tFetch;
	const deltaPhase = Promise.all(deltas.map(async (d, i) => {
		const { bytes, outcome } = await deltaReads[i];
		if (outcome === "mem") deltaMem++;
		deltaBytes += bytes.byteLength;
		deltaScanned += d.count;
		const hits = scan_delta(bytes, q, deltaQc(d, bytes), T, tomb ? tombFor(tomb, d.addedRev + 1) : noTomb, scanStats);
		deltaDropped += scanStats[1];
		for (const h of parseHits(hits)) deltaHits.push({
			id: h.id,
			score: h.score,
			seq: d.seq
		});
		tDeltaEnd = Date.now();
	}));
	await Promise.all([slabPhase, deltaPhase]);
	const all = new Uint8Array(packed.reduce((a, b) => a + b.byteLength, 0));
	let o = 0;
	for (const b of packed) {
		all.set(b, o);
		o += b.byteLength;
	}
	packed.length = 0;
	const merged = all.byteLength === 0 ? [] : parseHits(merge(all, T));
	const tPhase1 = Date.now();
	let slabResults;
	let rerankLocal = 0;
	let rerankSlabs = 0;
	let specUsed = 0;
	const reranked = doRerank && (versions.has(2) || versions.has(3));
	if (reranked) {
		const byCell = /* @__PURE__ */ new Map();
		for (const h of merged) {
			const c = cand.get(h.id);
			if (!c) throw new HttpError(500, `merge returned unknown id ${h.id}`);
			let list = byCell.get(c.cell);
			if (!list) byCell.set(c.cell, list = []);
			list.push({
				id: h.id,
				idx: c.idx
			});
		}
		rerankSlabs = byCell.size;
		const rescored = [];
		await Promise.all([...byCell].map(async ([cell, ws]) => {
			const si = slabInfo.get(cell);
			const recLen = si.recLen;
			ws.sort((a, b) => a.idx - b.idx);
			const records = new Uint8Array(ws.length * recLen);
			const spec = await specReads.get(cell) ?? null;
			const remote = [];
			ws.forEach((w, pos) => {
				const l = local.get(`${cell}:${w.idx}`);
				if (l) {
					records.set(l, pos * recLen);
					rerankLocal++;
					return;
				}
				const sp = spec?.get(w.idx);
				if (sp) {
					records.set(sp, pos * recLen);
					specUsed++;
					return;
				}
				remote.push(pos);
			});
			const got = await fetchRecords(cell, remote.map((pos) => ws[pos].idx), false);
			for (const pos of remote) records.set(got.get(ws[pos].idx), pos * recLen);
			const scores = rerank(q, si.centroid, records, ws.length);
			ws.forEach((w, i) => rescored.push({
				id: w.id,
				score: scores[i]
			}));
		}));
		slabResults = rescored;
	} else slabResults = merged.map((h) => ({
		id: h.id,
		score: h.score
	}));
	const best = /* @__PURE__ */ new Map();
	for (const r of slabResults) {
		const prev = best.get(r.id);
		if (!prev || r.score > prev.score) best.set(r.id, {
			id: r.id,
			score: r.score,
			delta: false
		});
	}
	for (const h of deltaHits) {
		const prev = best.get(h.id);
		if (!prev || h.score > prev.score) best.set(h.id, {
			id: h.id,
			score: h.score,
			delta: true
		});
	}
	const results = [...best.values()].sort((a, b) => b.score - a.score).slice(0, k);
	const tEnd = Date.now();
	return json({
		gen,
		rev: rv.rev,
		ids: results.map((r) => r.id),
		scores: results.map((r) => r.score),
		cells,
		stats: {
			scanned,
			slabBytes,
			scanBytes,
			k,
			nprobe,
			probeCoarse,
			routing: useBeam ? "beam" : flat ? "flat" : "tree",
			beam,
			treeGroups: tree?.groups ?? 0,
			dotsComputed,
			format,
			tiers: manifest.tiers ?? null,
			pqM: base.pq ? base.pq.m() : 0,
			profile: manifest.profile ?? null,
			params: {
				nprobe,
				rerank: rerankParam,
				pc: probeCoarse,
				route: routeMode,
				beam,
				source: {
					nprobe: source(p.nprobe),
					rerank: source(p.rerank),
					pc: source(p.probeCoarse),
					route: p.route !== null ? "query" : manifest.defaults?.route ? "manifest" : "legacy",
					beam: beam === null ? null : p.beam !== null ? "query" : defaults.beam !== void 0 ? "manifest" : "tree"
				}
			},
			rerank: reranked ? T : 0,
			rerankCandidates: reranked ? merged.length : 0,
			rerankReads,
			rerankBytes,
			rerankLocal,
			rerankSlabs,
			rerankMultipart,
			specRecords,
			specUsed,
			specReads: specReadN,
			deltas: deltas.length,
			deltaBytes,
			deltaScanned,
			deltaHits: deltaHits.length,
			deltaDropped,
			deltaTop: results.filter((r) => r.delta).length,
			deltaMem,
			tombstones: tomb?.total ?? 0,
			tombSets: tomb?.sets.length ?? 0,
			tombDropped,
			limiterRetries: lim.retries,
			cache: mode,
			cacheHits,
			cacheMisses,
			range: rs,
			indexCache: {
				...base.loadCache,
				...rv.loadCache
			},
			attempt,
			wasmResets,
			wasmMemMB: wasmRaw ? Math.round(wasmRaw.memory.buffer.byteLength / 1048576) : null,
			debug: dbg,
			colo: request.cf?.colo
		},
		timing: {
			total: tEnd - tStart,
			gen: tGen - tStart,
			load: tRoute - tStart,
			fetch: tLastSlab - tFetch,
			delta: tDeltaEnd - tFetch,
			phase1: tPhase1 - tFetch,
			phase2: tEnd - tPhase1,
			scanAndMerge: tPhase1 - tLastSlab,
			indexLoadMs: base.loadMs,
			revLoadMs: rv.loadMs
		}
	});
}
/**
* Debug: how caches.default answers a Range request for a cached object.
* `offset`/`length` = one range; `ranges=o:l,o:l,...` = one multi-range request (each part is
* checked against an R2 range read).
*/
async function handleRangeProbe(env, ctx, url) {
	const key = url.searchParams.get("key") ?? "";
	if (!key.startsWith("idx/")) throw new HttpError(400, "key must be under idx/");
	const offset = intParam(url, "offset", 0);
	const length = intParam(url, "length", 64, 1);
	const prime = url.searchParams.get("prime") === "1";
	let primed = false;
	if (prime && !await caches.default.match(cacheRequest(key))) {
		const buf = await getBuffer(env.BUCKET, key);
		const resp = new Response(buf, { headers: {
			"content-type": "application/octet-stream",
			"content-length": String(buf.byteLength),
			"cache-control": CACHE_CONTROL,
			"accept-ranges": "bytes"
		} });
		await caches.default.put(cacheRequest(key), resp);
		primed = true;
	}
	const rangesParam = url.searchParams.get("ranges");
	if (rangesParam) {
		const ranges = rangesParam.split(",").map((s) => {
			const [o, l] = s.split(":").map(Number);
			if (!Number.isInteger(o) || !Number.isInteger(l) || o < 0 || l < 1) throw new HttpError(400, "ranges must be offset:length,...");
			return {
				offset: o,
				length: l
			};
		});
		const hdr = "bytes=" + ranges.map((r) => `${r.offset}-${r.offset + r.length - 1}`).join(", ");
		const r = await caches.default.match(new Request(cacheRequest(key).url, {
			method: "GET",
			headers: { range: hdr }
		}));
		if (!r) return json({
			key,
			cached: false,
			primed,
			rangeHeader: hdr
		});
		const body = new Uint8Array(await r.arrayBuffer());
		const head = {
			status: r.status,
			contentType: r.headers.get("content-type"),
			contentRange: r.headers.get("content-range"),
			contentLength: r.headers.get("content-length"),
			bodyBytes: body.byteLength
		};
		let parts = null;
		let parseError = null;
		try {
			const got = await cacheMatchMulti(key, ranges, {
				r206: 0,
				r200: 0,
				miss: 0
			});
			if (got) parts = await Promise.all(ranges.map(async (rg, i) => {
				const ref = new Uint8Array(await getBuffer(env.BUCKET, key, rg));
				const b = got.out[i];
				return {
					...rg,
					matchesR2: ref.byteLength === b.byteLength && ref.every((v, j) => v === b[j])
				};
			}));
		} catch (e) {
			parseError = e instanceof Error ? e.message : String(e);
		}
		return json({
			key,
			cached: true,
			primed,
			rangeHeader: hdr,
			...head,
			bodyHead: latin1.decode(body.subarray(0, 300)),
			parts,
			parseError
		});
	}
	const r = await caches.default.match(cacheRequest(key, {
		offset,
		length
	}));
	if (!r) return json({
		key,
		cached: false,
		primed
	});
	const body = new Uint8Array(await r.arrayBuffer());
	let matchesR2 = null;
	if (r.status === 206) {
		const ref = new Uint8Array(await getBuffer(env.BUCKET, key, {
			offset,
			length
		}));
		matchesR2 = ref.byteLength === body.byteLength && ref.every((v, i) => v === body[i]);
	}
	return json({
		key,
		cached: true,
		primed,
		status: r.status,
		contentRange: r.headers.get("content-range"),
		contentLength: r.headers.get("content-length"),
		bodyBytes: body.byteLength,
		matchesR2
	});
}
/** Body for a PUT: a string or the exact bytes of a view. */
function putBody(data) {
	return typeof data === "string" || data instanceof ArrayBuffer ? data : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}
/** PUT only if the key does not exist yet. Returns false if it does. */
async function putIfAbsent(env, key, data) {
	return await env.BUCKET.put(key, putBody(data), { onlyIf: { etagDoesNotMatch: "*" } }) !== null;
}
/**
* Writes an immutable object at `key`, or, if that key exists (a concurrent or crashed writer used the same
* rev/seq), at a unique sibling `{stem}.{nonce}{ext}`. Returns the key written; manifests store it verbatim.
*/
async function putUnique(env, key, data) {
	if (await putIfAbsent(env, key, data)) return key;
	const dot = key.lastIndexOf(".");
	for (let i = 0; i < 4; i++) {
		const alt = `${key.slice(0, dot)}.${crypto.randomUUID().slice(0, 8)}${key.slice(dot)}`;
		if (await putIfAbsent(env, alt, data)) return alt;
	}
	throw new HttpError(500, `could not write a unique key for ${key}`);
}
var sleep = (ms) => new Promise((r) => setTimeout(r, ms));
/** Moves idx/current.json forward to (gen, rev) with a CAS loop; leaves it alone if it points at another generation or a newer rev. */
async function advanceCurrent(env, gen, rev, setIfOtherGen = false, hint = null) {
	let conflicts = 0;
	for (let attempt = 0; attempt < 10; attempt++) {
		const r = attempt === 0 && hint ? hint : await readCurrent(env);
		if (!r) {
			if (await putIfAbsent(env, CURRENT_KEY, JSON.stringify({
				gen,
				rev
			}))) return {
				updated: true,
				conflicts,
				current: {
					gen,
					rev
				}
			};
			conflicts++;
			continue;
		}
		if (r.cur.gen !== gen && !setIfOtherGen || r.cur.gen === gen && r.cur.rev >= rev) return {
			updated: false,
			conflicts,
			current: r.cur
		};
		if (await env.BUCKET.put(CURRENT_KEY, JSON.stringify({
			gen,
			rev
		}), { onlyIf: { etagMatches: r.etag } })) {
			currentPtr = {
				cur: {
					gen,
					rev
				},
				at: Date.now()
			};
			return {
				updated: true,
				conflicts,
				current: {
					gen,
					rev
				}
			};
		}
		conflicts++;
		await sleep(20 + Math.random() * 50);
	}
	throw new HttpError(409, `idx/current.json: gave up after ${conflicts} CAS conflicts`);
}
var COMMIT_ATTEMPTS = 12;
/**
* Commits a new revision of `gen` (null = the current generation): `make` receives the head (rev, manifest) and
* returns the next manifest (it may write the objects that manifest references, keyed by rev + 1, via putUnique).
* The head is taken from idx/current.json when it points at `gen` (one GET instead of a LIST); a lost race (or a
* crashed writer's manifest beyond current.json) makes the next attempt list the manifest revisions.
*/
async function commitRev(env, genArg, make) {
	let conflicts = 0;
	let list = false;
	for (let attempt = 1; attempt <= COMMIT_ATTEMPTS; attempt++) {
		const cur = await readCurrent(env);
		const gen = genArg ?? cur?.cur.gen;
		if (!gen) throw new HttpError(404, "no published index");
		const head = !list && cur && cur.cur.gen === gen ? cur.cur.rev : await latestRev(env, gen);
		const next = await make({
			gen,
			rev: head,
			manifest: await readManifestR2(env, gen, head),
			attempt
		});
		const rev = head + 1;
		const manifest = {
			...next,
			gen,
			rev,
			parentRev: head,
			updatedAt: (/* @__PURE__ */ new Date()).toISOString()
		};
		const body = JSON.stringify(manifest);
		if (await putIfAbsent(env, manifestKey(gen, rev), body)) {
			genHeads.set(gen, {
				rev,
				at: Date.now()
			});
			const c = await advanceCurrent(env, gen, rev, false, cur);
			return {
				gen,
				rev,
				parentRev: head,
				manifest,
				conflicts,
				casConflicts: c.conflicts,
				currentUpdated: c.updated,
				manifestBytes: body.length
			};
		}
		conflicts++;
		list = true;
		await sleep(10 + Math.random() * 40 * attempt);
	}
	throw new HttpError(409, `${genArg ?? "current generation"}: lost the manifest race ${conflicts} times`);
}
/** The manifest fields a new revision inherits (build notes stay in rev 0). */
function inherit(m) {
	const { notes: _notes, change: _change, rev: _rev, parentRev: _parent, updatedAt: _u, ...rest } = m;
	return {
		...rest,
		files: manifestFiles(m),
		deltas: m.deltas ?? [],
		tombstones: m.tombstones ?? [],
		nextSeq: m.nextSeq ?? 1
	};
}
function checkDefaults(d, tree) {
	for (const name of [
		"nprobe",
		"rerank",
		"pc"
	]) {
		const v = d[name];
		if (!Number.isInteger(v) || v < (name === "nprobe" ? 1 : 0)) throw new HttpError(400, `defaults.${name} must be a non-negative integer${name === "nprobe" ? " >= 1" : ""}`);
	}
	if (d.route !== void 0 && !ROUTE_MODES.includes(d.route)) throw new HttpError(400, `defaults.route must be ${ROUTE_MODES.join("|")}`);
	if (d.route === "beam" && !tree) throw new HttpError(400, "defaults.route beam needs a centroid tree (pass tree)");
	if (d.beam !== void 0 && (!Number.isInteger(d.beam) || d.beam < 1 || tree?.groups !== void 0 && d.beam > tree.groups)) throw new HttpError(400, `defaults.beam must be an integer in 1..${tree?.groups ?? "tree groups"}`);
}
/** Manifest rev + 1 with changed query defaults and/or a new centroid tree (no slab, delta or tombstone changes). */
async function stagePublishRev(env, body) {
	const { gen } = body;
	if (!gen) throw new HttpError(400, "gen required");
	let built = null;
	let treeWritten = null;
	const c = await commitRev(env, gen, async ({ rev, manifest }) => {
		const files = manifestFiles(manifest);
		let tree = manifest.tree;
		if (body.tree === null) tree = void 0;
		else if (body.tree) {
			if (!built || built.fine !== files.fine) {
				const notes = rev === 0 ? manifest.notes : void 0;
				built = {
					fine: files.fine,
					...await buildTree(env, gen, manifest, files.coarse, files.fine, body.tree, notes?.cfg?.sampleGroups)
				};
				treeWritten = null;
			}
			treeWritten ??= {
				key: await putUnique(env, treeKey(gen, rev + 1), built.bytes),
				...built.ref
			};
			tree = treeWritten;
		}
		const prev = manifest.defaults ?? LEGACY_DEFAULTS;
		const defaults = {
			...prev,
			...body.defaults ?? {}
		};
		if (!tree && defaults.route === "beam" && body.defaults?.route === void 0) delete defaults.route;
		if (!tree && body.defaults?.beam === void 0) delete defaults.beam;
		checkDefaults(defaults, tree);
		const next = {
			...inherit(manifest),
			defaults,
			...body.profile ? { profile: body.profile } : {}
		};
		if (tree) next.tree = tree;
		else delete next.tree;
		const treeChange = body.tree === void 0 ? {} : { tree: tree ? {
			key: tree.key,
			groups: tree.groups,
			beam: tree.beam
		} : null };
		return {
			...next,
			change: {
				op: "publish-rev",
				defaults: {
					from: prev,
					to: defaults
				},
				...treeChange
			}
		};
	});
	const cur = body.setCurrent ? await advanceCurrent(env, gen, c.rev, true) : null;
	return {
		gen,
		rev: c.rev,
		parentRev: c.parentRev,
		defaults: c.manifest.defaults,
		tree: c.manifest.tree ?? null,
		conflicts: c.conflicts,
		currentUpdated: c.currentUpdated || !!cur?.updated
	};
}
/** Sorted unique ids as a tombstone buffer. */
var tombBytes = (ids) => idUnion([ids]);
var MAX_UPSERT = 4096;
function b64ToBytes(s) {
	const bin = atob(s);
	const out = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
	return out;
}
function parseIdList(raw) {
	return u32Array(raw.split(",").filter((x) => x.trim() !== "").map(Number), "ids");
}
function u32Array(a, name) {
	if (!Array.isArray(a)) throw new HttpError(400, `${name} must be an array`);
	for (const v of a) if (!Number.isInteger(v) || v < 0 || v > 4294967295) throw new HttpError(400, `${name} must hold u32 integers`);
	return Uint32Array.from(a);
}
/**
* Upsert body: JSON {gen?, ids, vectors (flat number[] or base64 f32 LE), attrs?, payloads? (strings, or
* {b64}), replace?}, or a binary f32 LE body (`content-type: application/octet-stream`) with the ids in the
* `x-ids` header or `?ids=` (comma-separated) or as `?firstId=` (ids firstId, firstId+1, ...), and `?gen=`,
* `?replace=0`.
*/
async function parseUpsert(request, url) {
	const ct = request.headers.get("content-type") ?? "";
	let gen = url.searchParams.get("gen");
	let replace = url.searchParams.get("replace") !== "0";
	let ids = null;
	let firstId = null;
	let vectors;
	let attrs = null;
	let payloads = null;
	if (ct.includes("octet-stream")) {
		const buf = await request.arrayBuffer();
		if (buf.byteLength % 4 !== 0) throw new HttpError(400, "binary body must be f32 LE");
		vectors = new Float32Array(buf);
		const hdr = request.headers.get("x-ids") ?? url.searchParams.get("ids");
		if (hdr) ids = parseIdList(hdr);
		firstId = optIntParam(url, "firstId", 0);
		if (!ids && firstId === null) throw new HttpError(400, "binary upsert needs ids (x-ids header, ?ids=) or ?firstId=");
	} else {
		const body = await readJson(request);
		gen = body.gen ?? gen;
		if (body.replace === false) replace = false;
		ids = u32Array(body.ids, "ids");
		if (typeof body.vectors === "string") {
			const b = b64ToBytes(body.vectors);
			if (b.byteLength % 4 !== 0) throw new HttpError(400, "base64 vectors must be f32 LE");
			vectors = new Float32Array(b.buffer, 0, b.byteLength / 4);
		} else if (Array.isArray(body.vectors)) vectors = Float32Array.from(body.vectors);
		else throw new HttpError(400, "vectors must be a flat number[] or base64 f32");
		if (body.attrs) attrs = u32Array(body.attrs, "attrs");
		if (body.payloads) {
			const enc = new TextEncoder();
			payloads = body.payloads.map((p) => typeof p === "string" ? enc.encode(p) : p && typeof p === "object" && typeof p.b64 === "string" ? b64ToBytes(p.b64) : enc.encode(JSON.stringify(p)));
		}
	}
	return {
		gen,
		ids,
		firstId,
		vectors,
		attrs,
		payloads,
		replace
	};
}
/** Validates the batch against the manifest and returns the final ids and the vectors, unit-normalized in place. */
function prepareBatch(input, dim) {
	const { vectors } = input;
	if (vectors.length % dim !== 0) throw new HttpError(400, `vectors has ${vectors.length} floats, not a multiple of dim ${dim}`);
	const n = vectors.length / dim;
	const first = input.firstId;
	const ids = input.ids ?? Uint32Array.from({ length: n }, (_, i) => first + i);
	if (first !== null && first + n - 1 > 4294967295) throw new HttpError(400, "firstId + n exceeds u32");
	if (n === 0) throw new HttpError(400, "empty batch");
	if (n > MAX_UPSERT) throw new HttpError(413, `at most ${MAX_UPSERT} vectors per upsert`);
	if (ids.length !== n) throw new HttpError(400, `${ids.length} ids for ${n} vectors`);
	if (new Set(ids).size !== n) throw new HttpError(400, "duplicate ids in the batch");
	if (input.attrs && input.attrs.length !== n) throw new HttpError(400, `${input.attrs.length} attrs for ${n} vectors`);
	if (input.payloads && input.payloads.length !== n) throw new HttpError(400, `${input.payloads.length} payloads for ${n} vectors`);
	for (let i = 0; i < n; i++) {
		let ss = 0;
		for (let j = i * dim; j < (i + 1) * dim; j++) ss += vectors[j] * vectors[j];
		if (!(ss > 0) || !Number.isFinite(ss)) throw new HttpError(400, `vector ${i} (id ${ids[i]}) has zero or non-finite norm`);
		const inv = 1 / Math.sqrt(ss);
		for (let j = i * dim; j < (i + 1) * dim; j++) vectors[j] *= inv;
	}
	return {
		ids,
		vectors
	};
}
/** The first free delta seq >= `from` (HEAD probes, so a crashed writer's segment is skipped). */
async function allocSeq(env, gen, from) {
	let seq = Math.max(1, from);
	for (let i = 0; i < 64 && await env.BUCKET.head(deltaKey(gen, seq)); i++) seq++;
	return seq;
}
/** Bases this isolate has loaded (0 = a write-only isolate). */
var basesLoaded = 0;
/** Fine means kept by write-only isolates (no base ever loaded), saving a 25 MB cache read per upsert. */
var writerMeans = null;
/** The last centroid tree a write used (~1 MB; kept by every isolate, trees are small). */
var writerTree = null;
/** What a delta segment's encoding depends on: a retry re-encodes only if one of these changed under it. */
var deltaFilesKey = (files, m) => `${files.fine}|${files.pq}|${m.tree?.key ?? "-"}|${m.tree?.beam ?? 0}`;
/** Vectors passed to DeltaJob.add per call (bounds the batch copy in wasm memory). */
var DELTA_ADD_CHUNK = 256;
/**
* Encodes one delta segment (primary cell + residual codes against the generation's fine means; PQ codes with
* the generation's codebook) and writes it. Writes never load the Searcher. DeltaJob holds a copy of the means
* (~25 MB at 4096 x 1536, ~50 MB while it is passed in) and wasm memory never shrinks, so in an isolate that
* also serves queries an idle loaded base is dropped first (its memory is reused; the next query reloads it);
* a base that queries are using lends its PqCodebook instead. Writes are best sent to a Worker deployment that
* does not serve queries.
*/
async function encodeDelta(env, ctx, m, files, ids, vectors, input) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const tLoad = Date.now();
	const gen = m.gen;
	const format = m.format ?? 1;
	let slot = baseCache.get(baseKeyOf(gen, files)) ?? null;
	if (slot && slot.users === 0) {
		dropBaseSlot(slot);
		slot = null;
	}
	for (const other of [...baseCache.values()]) if (other.users === 0) dropBaseSlot(other);
	if (format === 3 && !files.pq) throw new HttpError(500, `format 3 generation ${gen} has no PQ codebook`);
	const loads = await Promise.all([
		allocSeq(env, gen, m.nextSeq ?? 1),
		slot ? slot.promise.catch(() => null) : Promise.resolve(null),
		format === 3 && !slot ? getCached(env, ctx, files.pq) : Promise.resolve(null),
		writerMeans?.key === files.fine ? Promise.resolve({ buf: writerMeans.buf }) : getCached(env, ctx, files.fine),
		!m.tree ? Promise.resolve(null) : writerTree?.key === m.tree.key ? Promise.resolve({ buf: writerTree.buf }) : getCached(env, ctx, m.tree.key)
	]);
	const [seq, loaded] = loads;
	if (basesLoaded === 0) writerMeans = {
		key: files.fine,
		buf: loads[3].buf
	};
	const treeBuf = loads[4];
	if (m.tree && treeBuf) writerTree = {
		key: m.tree.key,
		buf: treeBuf.buf
	};
	let pqBuf = loads[2];
	let meansBuf = loads[3];
	loads.length = 0;
	let pq = loaded && loaded.epoch === wasmEpoch && !slot.evicted ? loaded.pq : null;
	let ownPq = null;
	if (format === 3 && !pq) pq = ownPq = parseCodebook(pqBuf?.buf ?? (await getCached(env, ctx, files.pq)).buf, m, files.pq);
	pqBuf = null;
	if (slot) slot.users++;
	try {
		const job = (() => {
			const means = new Float32Array(meansBuf.buf);
			meansBuf = null;
			if (means.length !== m.k1 * m.k2 * m.dim) throw new HttpError(500, `${files.fine} has ${means.length / m.dim} means, expected ${m.k1 * m.k2}`);
			return new DeltaJob(m.dim, m.k1 * m.k2, means, seq);
		})();
		const loadMs = Date.now() - tLoad;
		let finished = false;
		let bytes;
		let cells;
		const tEnc = Date.now();
		try {
			if (format === 3) job.set_pq_codebook(pq);
			if (m.tree && treeBuf) job.set_tree(new Uint8Array(treeBuf.buf), m.tree.beam);
			const dim = m.dim;
			cells = new Uint32Array(ids.length);
			for (let a = 0; a < ids.length; a += DELTA_ADD_CHUNK) {
				const b = Math.min(ids.length, a + DELTA_ADD_CHUNK);
				const v = vectors.subarray(a * dim, b * dim);
				const attrs = input.attrs ? input.attrs.subarray(a, b) : /* @__PURE__ */ new Uint32Array(0);
				let c;
				if (input.payloads) {
					const ps = input.payloads.slice(a, b);
					const lens = Uint32Array.from(ps, (p) => p.byteLength);
					const cat = new Uint8Array(lens.reduce((x, y) => x + y, 0));
					let o = 0;
					for (const p of ps) {
						cat.set(p, o);
						o += p.byteLength;
					}
					c = job.add_with_payloads(v, ids.subarray(a, b), attrs, cat, lens);
				} else c = job.add(v, ids.subarray(a, b), attrs);
				if (c.length !== b - a) throw new HttpError(500, `DeltaJob added ${c.length} of ${b - a} vectors`);
				cells.set(c, a);
			}
			if (cells.length !== ids.length) throw new HttpError(500, `DeltaJob added ${cells.length} of ${ids.length} vectors`);
			finished = true;
			bytes = job.finish();
		} finally {
			if (!finished) safeFree(job, epoch);
		}
		const encodeMs = Date.now() - tEnc;
		const nCells = delta_cells(bytes).length / 2;
		const tPut = Date.now();
		return {
			seq,
			key: await putUnique(env, deltaKey(gen, seq), bytes),
			count: ids.length,
			bytes: bytes.byteLength,
			cells,
			nCells,
			filesKey: deltaFilesKey(files, m),
			tree: m.tree ? {
				groups: m.tree.groups,
				beam: m.tree.beam
			} : null,
			encodeMs,
			loadMs,
			putMs: Date.now() - tPut
		};
	} finally {
		if (ownPq) safeFree(ownPq, epoch);
		if (slot) {
			slot.users--;
			if (slot.evicted && slot.users === 0) slot.promise.then(freeLoadedBase, () => {});
		}
	}
}
async function targetGen(env, gen) {
	if (gen) return gen;
	const r = await readCurrent(env);
	if (!r) throw new HttpError(404, "no published index");
	return r.cur.gen;
}
/** POST /upsert: one delta segment + (by default) a tombstone set for its ids, in one new manifest revision. */
async function handleUpsert(request, env, ctx, url) {
	const t0 = Date.now();
	const input = await parseUpsert(request, url);
	return withWasmRecovery("upsert", async () => {
		const st = {
			delta: null,
			batch: null,
			tombKey: null,
			tombBytes: 0
		};
		const c = await commitRev(env, input.gen, async ({ gen, rev, manifest }) => {
			const files = manifestFiles(manifest);
			const batch = st.batch ??= prepareBatch(input, manifest.dim);
			const next = inherit(manifest);
			const newRev = rev + 1;
			const tb = input.replace ? tombBytes(batch.ids) : null;
			const tombPut = tb ? putUnique(env, tombKey(gen, newRev), tb) : Promise.resolve(null);
			try {
				if (!st.delta || st.delta.filesKey !== deltaFilesKey(files, manifest)) st.delta = await encodeDelta(env, ctx, manifest, files, batch.ids, batch.vectors, input);
			} finally {
				st.tombKey = await tombPut.catch(() => null);
			}
			const delta = st.delta;
			if (tb) {
				if (!st.tombKey) st.tombKey = await putUnique(env, tombKey(gen, newRev), tb);
				st.tombBytes = tb.byteLength;
				next.tombstones.push({
					rev: newRev,
					key: st.tombKey,
					count: tb.byteLength / 4
				});
			}
			next.deltas.push({
				seq: delta.seq,
				key: delta.key,
				count: delta.count,
				bytes: delta.bytes,
				addedRev: newRev,
				cells: delta.nCells
			});
			next.nextSeq = Math.max(next.nextSeq ?? 1, delta.seq + 1);
			return {
				...next,
				change: {
					op: "upsert",
					seq: delta.seq,
					count: delta.count,
					replace: input.replace
				}
			};
		});
		const d = st.delta;
		const m = c.manifest;
		return json({
			gen: c.gen,
			rev: c.rev,
			seq: d.seq,
			count: d.count,
			cells: Array.from(d.cells),
			distinctCells: d.nCells,
			ms: Date.now() - t0,
			bytes: {
				delta: d.bytes,
				tombstones: st.tombBytes,
				manifest: c.manifestBytes
			},
			keys: {
				delta: d.key,
				tombstones: st.tombKey
			},
			conflicts: c.conflicts,
			casConflicts: c.casConflicts,
			currentUpdated: c.currentUpdated,
			deltas: m.deltas.length,
			deltaBytes: m.deltas.reduce((a, x) => a + x.bytes, 0),
			tombstoneSets: m.tombstones.length,
			tree: d.tree,
			timing: {
				load: d.loadMs,
				encode: d.encodeMs,
				put: d.putMs
			}
		});
	});
}
/** POST /delete {gen?, ids}: one tombstone set in a new manifest revision. */
async function handleDelete(request, env, url) {
	const t0 = Date.now();
	const body = await readJson(request);
	const ids = u32Array(body.ids, "ids");
	if (!ids.length) throw new HttpError(400, "no ids");
	const tb = tombBytes(ids);
	let key = "";
	const c = await commitRev(env, body.gen ?? url.searchParams.get("gen"), async ({ gen, rev, manifest }) => {
		const next = inherit(manifest);
		key = await putUnique(env, tombKey(gen, rev + 1), tb);
		next.tombstones.push({
			rev: rev + 1,
			key,
			count: tb.byteLength / 4
		});
		return {
			...next,
			change: {
				op: "delete",
				count: tb.byteLength / 4
			}
		};
	});
	return json({
		gen: c.gen,
		rev: c.rev,
		count: tb.byteLength / 4,
		key,
		ms: Date.now() - t0,
		bytes: {
			tombstones: tb.byteLength,
			manifest: c.manifestBytes
		},
		conflicts: c.conflicts,
		casConflicts: c.casConflicts,
		currentUpdated: c.currentUpdated,
		tombstoneSets: c.manifest.tombstones.length
	});
}
async function loadCompactBase(env, gen, baseRev) {
	const manifest = await readManifestR2(env, gen, baseRev);
	const files = manifestFiles(manifest);
	if (!files.slabs) throw new HttpError(400, `${gen} has no slabs table`);
	if ((manifest.format ?? 1) < 2) throw new HttpError(400, "compaction needs a format 2 or 3 generation");
	const [slabs, tombs] = await Promise.all([getBuffer(env.BUCKET, files.slabs).then((b) => decodeJson(b)), pmap(manifest.tombstones ?? [], SUBREQUEST_CONCURRENCY, async (t) => ({
		rev: t.rev,
		key: t.key,
		ids: new Uint32Array(await getBuffer(env.BUCKET, t.key))
	}))]);
	return {
		gen,
		baseRev,
		manifest,
		files,
		slabs,
		deltas: manifest.deltas ?? [],
		tombs
	};
}
async function stageCompactPlan(env, body) {
	await ensureWasm();
	const gen = await targetGen(env, body.gen ?? null);
	const baseRev = await latestRev(env, gen);
	const b = await loadCompactBase(env, gen, baseRev);
	const nCells = b.manifest.k1 * b.manifest.k2;
	const deltaCells = /* @__PURE__ */ new Set();
	const cellsOf = /* @__PURE__ */ new Map();
	await pmap(b.deltas, SUBREQUEST_CONCURRENCY, async (d) => {
		const dc = delta_cells(new Uint8Array(await getBuffer(env.BUCKET, d.key)));
		const own = [];
		for (let i = 0; i < dc.length; i += 2) if (dc[i + 1] > 0) own.push(dc[i]);
		for (const c of own) deltaCells.add(c);
		cellsOf.set(d.key, own);
	});
	const maxTomb = b.tombs.reduce((a, t) => Math.max(a, t.rev), 0);
	const tombCells = /* @__PURE__ */ new Set();
	if (b.tombs.length && body.purge) b.slabs.forEach((row, cell) => slabRevOf(row) > 0 && slabRevOf(row) <= maxTomb && tombCells.add(cell));
	let cells;
	if (body.cells === "all") cells = Array.from({ length: nCells }, (_, i) => i);
	else if (body.cells) {
		cells = [...new Set(body.cells)].sort((x, y) => x - y);
		if (cells.some((c) => !Number.isInteger(c) || c < 0 || c >= nCells)) throw new HttpError(400, "cells out of range");
	} else cells = [.../* @__PURE__ */ new Set([...deltaCells, ...tombCells])].sort((x, y) => x - y);
	const planned = new Set(cells);
	const folded = b.deltas.filter((d) => cellsOf.get(d.key).every((c) => planned.has(c))).map((d) => d.key);
	return {
		gen,
		baseRev,
		slabRev: baseRev + 1,
		cells,
		deltaCells: deltaCells.size,
		tombCells: tombCells.size,
		deltas: b.deltas.map((d) => d.key),
		/** Deltas whose cells are all in `cells`: fold these into compact-cells and compact-commit. */
		folded,
		deltaEntries: b.deltas.reduce((a, d) => a + d.count, 0),
		tombstoneSets: b.tombs.length,
		tombstones: b.tombs.reduce((a, t) => a + t.ids.length, 0)
	};
}
async function compactCells(env, b, cells, concurrency, folded) {
	await ensureWasm();
	const epoch = wasmEpoch;
	const { gen, manifest, slabs } = b;
	const slabRev = b.baseRev + 1;
	const format = manifest.format ?? 1;
	const nCells = manifest.k1 * manifest.k2;
	const foldSet = folded ? new Set(folded) : null;
	const deltas = await pmap(foldSet ? b.deltas.filter((d) => foldSet.has(d.key)) : b.deltas, SUBREQUEST_CONCURRENCY, async (d) => {
		const bytes = new Uint8Array(await getBuffer(env.BUCKET, d.key));
		const dc = delta_cells(bytes);
		const counts = /* @__PURE__ */ new Map();
		for (let i = 0; i < dc.length; i += 2) counts.set(dc[i], dc[i + 1]);
		return {
			ref: d,
			bytes,
			counts
		};
	});
	const sets = b.tombs.map((t) => ({
		rev: t.rev,
		ids: t.ids
	}));
	const unionCache = /* @__PURE__ */ new Map();
	const union = (pred, tag) => {
		let u = unionCache.get(tag);
		if (!u) unionCache.set(tag, u = idUnion(sets.filter((s) => pred(s.rev)).map((s) => s.ids)));
		return u;
	};
	const slabTomb = (r) => union((rev) => rev >= r, `s${r}`);
	const deltaTomb = (addedRev) => union((rev) => rev > addedRev, `d${addedRev}`);
	const rows = [];
	let readBytes = 0;
	let writtenBytes = 0;
	let oldEntries = 0;
	let newEntries = 0;
	let deltaEntries = 0;
	await pmap(cells, concurrency, async (cell) => {
		if (!Number.isInteger(cell) || cell < 0 || cell >= nCells) throw new HttpError(400, `cell ${cell} out of range`);
		const row = slabs[cell];
		const oldRev = slabRevOf(row);
		if (oldRev >= slabRev) throw new HttpError(409, `cell ${cell} already has slabRev ${oldRev} >= ${slabRev}`);
		const old = new Uint8Array(await getBuffer(env.BUCKET, slabKey(gen, cell, oldRev)));
		readBytes += old.byteLength;
		const info = slab_info(old);
		if (info[2] !== cell) throw new HttpError(500, `slab ${slabKey(gen, cell, oldRev)} is for cell ${info[2]}`);
		oldEntries += info[1];
		const job = new SlabJob(manifest.dim, cell, slab_centroid(old), format);
		let finished = false;
		try {
			job.add_records(slab_to_records(old, slabTomb(oldRev)));
			for (const d of deltas) {
				if (!d.counts.has(cell)) continue;
				deltaEntries += job.add_records(delta_records_for_cell(d.bytes, cell, deltaTomb(d.ref.addedRev)));
			}
			const n = job.len();
			finished = true;
			const out = job.finish();
			const oi = slab_info(out);
			await putBytes(env.BUCKET, slabKey(gen, cell, slabRev), out);
			writtenBytes += out.byteLength;
			newEntries += n;
			rows.push([cell, [
				oi[5],
				oi[6],
				n,
				out.byteLength,
				slabRev
			]]);
		} finally {
			if (!finished) safeFree(job, epoch);
		}
	});
	rows.sort((x, y) => x[0] - y[0]);
	return {
		gen,
		baseRev: b.baseRev,
		slabRev,
		rows,
		cells: rows.length,
		readBytes,
		writtenBytes,
		oldEntries,
		deltaEntries,
		newEntries
	};
}
async function stageCompactCells(env, body) {
	return compactCells(env, await loadCompactBase(env, body.gen, body.baseRev), body.cells, Math.max(1, Math.min(body.concurrency ?? 3, 6)), body.folded);
}
/**
* Whether a tombstone set can be dropped from the manifest: no remaining delta predates it, no slab written at
* a rev in (0, set.rev] remains (such a slab may hold an id the set hides), and none of its ids is a build id
* (rev-0 slabs hold ids [0, manifest.vectors), the driver's id space).
*/
function tombDroppable(t, deltas, slabs, baseIds) {
	if (deltas.some((d) => d.addedRev < t.rev)) return false;
	if (slabs.some((row) => slabRevOf(row) > 0 && slabRevOf(row) <= t.rev)) return false;
	for (const id of t.ids) if (id < baseIds) return false;
	return true;
}
async function compactCommit(env, body) {
	const { gen, baseRev } = body;
	const slabRev = baseRev + 1;
	const baseM = await readManifestR2(env, gen, baseRev);
	const folded = new Set(body.folded ?? (baseM.deltas ?? []).map((d) => d.key));
	await ensureWasm();
	const rowCells = new Set(body.rows.map(([cell]) => cell));
	await pmap([...folded], SUBREQUEST_CONCURRENCY, async (key) => {
		const dc = delta_cells(new Uint8Array(await getBuffer(env.BUCKET, key)));
		for (let i = 0; i < dc.length; i += 2) if (dc[i + 1] > 0 && !rowCells.has(dc[i])) throw new HttpError(409, `folded delta ${key} has entries in cell ${dc[i]}, which is not among the rewritten rows`);
	});
	let dropped = [];
	let slabsKeyWritten = "";
	let tableBytes = 0;
	const c = await commitRev(env, gen, async ({ rev, manifest }) => {
		const files = manifestFiles(manifest);
		if (files.slabs !== manifestFiles(baseM).slabs) throw new HttpError(409, `slabs table changed since rev ${baseRev} (another compaction committed)`);
		const deltas = manifest.deltas ?? [];
		for (const k of folded) if (!deltas.some((d) => d.key === k)) throw new HttpError(409, `folded delta ${k} is no longer in rev ${rev}`);
		const table = decodeJson(await getBuffer(env.BUCKET, files.slabs));
		for (const [cell, row] of body.rows) {
			if (!Number.isInteger(cell) || cell < 0 || cell >= table.length) throw new HttpError(400, `row cell ${cell} out of range`);
			if (slabRevOf(row) !== slabRev) throw new HttpError(400, `row for cell ${cell} has slabRev ${slabRevOf(row)}, expected ${slabRev}`);
			table[cell] = row;
		}
		const remaining = deltas.filter((d) => !folded.has(d.key));
		const tombs = await pmap(manifest.tombstones ?? [], SUBREQUEST_CONCURRENCY, async (t) => ({
			...t,
			ids: new Uint32Array(await getBuffer(env.BUCKET, t.key))
		}));
		const keep = tombs.filter((t) => !tombDroppable(t, remaining, table, manifest.vectors));
		dropped = tombs.filter((t) => !keep.includes(t)).map((t) => t.rev);
		const tj = JSON.stringify(table);
		tableBytes = tj.length;
		slabsKeyWritten = await putUnique(env, slabsTableKey(gen, rev + 1), tj);
		return {
			...inherit(manifest),
			files: {
				...files,
				slabs: slabsKeyWritten
			},
			deltas: remaining,
			tombstones: keep.map(({ rev: r, key, count }) => ({
				rev: r,
				key,
				count
			})),
			change: {
				op: "compact",
				baseRev,
				slabRev,
				cells: body.rows.length,
				foldedDeltas: folded.size,
				droppedTombstoneSets: dropped.length
			}
		};
	});
	return {
		gen,
		rev: c.rev,
		baseRev,
		slabRev,
		cells: body.rows.length,
		folded: folded.size,
		droppedTombstoneSets: dropped,
		slabsKey: slabsKeyWritten,
		slabsTableBytes: tableBytes,
		deltas: c.manifest.deltas.length,
		tombstoneSets: c.manifest.tombstones.length,
		conflicts: c.conflicts,
		currentUpdated: c.currentUpdated
	};
}
/** Plan + rewrite + commit in one invocation (fine for a few hundred cells; bench/compact.mjs splits larger ones). */
async function stageCompact(env, body) {
	const t0 = Date.now();
	const plan = await stageCompactPlan(env, body);
	if (!plan.cells.length && !plan.deltas.length && !plan.tombstoneSets) return {
		...plan,
		noop: true
	};
	const b = await loadCompactBase(env, plan.gen, plan.baseRev);
	const tCells = Date.now();
	const r = await compactCells(env, b, plan.cells, Math.max(1, Math.min(body.concurrency ?? 3, 6)), plan.folded);
	const cellsMs = Date.now() - tCells;
	const commit = await compactCommit(env, {
		gen: plan.gen,
		baseRev: plan.baseRev,
		rows: r.rows,
		folded: plan.folded
	});
	return {
		plan: {
			...plan,
			cells: plan.cells.length
		},
		...commit,
		readBytes: r.readBytes,
		writtenBytes: r.writtenBytes,
		oldEntries: r.oldEntries,
		deltaEntries: r.deltaEntries,
		newEntries: r.newEntries,
		cellsMs,
		ms: Date.now() - t0
	};
}
/**
* Deletes objects under idx/{gen}/ that none of the kept revisions references: old slab revisions, folded
* deltas, dropped tombstone sets, superseded slabs tables and the leftovers of writers that lost a race.
* Manifests are kept (small; they are the history). Kept revisions: the newest `keep`, plus any revision
* superseded less than graceSec ago (queries may still hold it for CURRENT_TTL_MS).
*/
async function stageGc(env, body) {
	const { gen } = body;
	if (!gen) throw new HttpError(400, "gen required");
	const keepN = Math.max(1, body.keep ?? 2);
	const graceMs = (body.graceSec ?? 300) * 1e3;
	const dryRun = body.dryRun !== false;
	const now = Date.now();
	const revs = await listRevs(env, gen);
	if (!revs.length) throw new HttpError(404, `no manifest for ${gen}`);
	const kept = revs.filter((r, i) => i >= revs.length - keepN || i + 1 < revs.length && now - revs[i + 1].uploaded.getTime() < graceMs).map((r) => r.rev);
	const referenced = /* @__PURE__ */ new Set();
	const tables = /* @__PURE__ */ new Set();
	await pmap(kept, SUBREQUEST_CONCURRENCY, async (rev) => {
		const m = await readManifestR2(env, gen, rev);
		const f = manifestFiles(m);
		for (const k of [
			f.coarse,
			f.fine,
			f.pq,
			f.slabs
		]) if (k) referenced.add(k);
		if (f.slabs) tables.add(f.slabs);
		for (const d of m.deltas ?? []) referenced.add(d.key);
		for (const t of m.tombstones ?? []) referenced.add(t.key);
		if (m.tree) referenced.add(m.tree.key);
	});
	await pmap([...tables], SUBREQUEST_CONCURRENCY, async (key) => {
		decodeJson(await getBuffer(env.BUCKET, key)).forEach((row, cell) => referenced.add(slabKey(gen, cell, slabRevOf(row))));
	});
	const all = await listAll(env, `idx/${gen}/`);
	const kindOf = (key) => {
		const rest = key.slice(`idx/${gen}/`.length);
		return rest.startsWith("slab/") ? "slab" : rest.startsWith("delta/") ? "delta" : rest.startsWith("tombstones.") ? "tombstones" : rest.startsWith("slabs.") ? "slabs-table" : null;
	};
	const garbage = all.filter((o) => kindOf(o.key) !== null && !referenced.has(o.key) && now - o.uploaded.getTime() >= graceMs);
	const kinds = {};
	for (const o of garbage) {
		const kind = kindOf(o.key);
		kinds[kind] ??= {
			n: 0,
			bytes: 0
		};
		kinds[kind].n++;
		kinds[kind].bytes += o.size;
	}
	let deleted = 0;
	if (!dryRun) for (let i = 0; i < garbage.length; i += 1e3) {
		const keys = garbage.slice(i, i + 1e3).map((o) => o.key);
		await env.BUCKET.delete(keys);
		deleted += keys.length;
	}
	return {
		gen,
		dryRun,
		keptRevs: kept,
		objects: all.length,
		referenced: referenced.size,
		garbage: garbage.length,
		kinds,
		deleted,
		sample: garbage.slice(0, 10).map((o) => o.key)
	};
}
/** Deletes every object under one of the allowed scratch prefixes (housekeeping of test generations). */
var SCRATCH_PREFIXES = [
	"build/smoke/",
	"build/g3smoke/",
	"idx/g3smoke/",
	"build/pqtest/",
	"idx/pqtest/",
	"build/g1/partials/test/"
];
async function stageDeletePrefix(env, body) {
	if (!SCRATCH_PREFIXES.includes(body.prefix)) throw new HttpError(400, `prefix must be one of ${SCRATCH_PREFIXES.join(", ")}`);
	const objs = await listAll(env, body.prefix);
	const bytes = objs.reduce((a, o) => a + o.size, 0);
	let deleted = 0;
	if (body.dryRun === false) for (let i = 0; i < objs.length; i += 1e3) {
		const keys = objs.slice(i, i + 1e3).map((o) => o.key);
		await env.BUCKET.delete(keys);
		deleted += keys.length;
	}
	return {
		prefix: body.prefix,
		objects: objs.length,
		bytes,
		deleted,
		remaining: body.dryRun === false ? (await listAll(env, body.prefix)).length : objs.length
	};
}
var stages = {
	sample: stageSample,
	"init-coarse": stageInitCoarse,
	"kmeans-flat": stageKmeansFlat,
	"init-fine": stageInitFine,
	"kmeans-tree-map": stageKmeansTreeMap,
	"kmeans-inc-map": stageKmeansIncMap,
	"cell-neighbors": stageCellNeighbors,
	"kmeans-reduce": stageKmeansReduce,
	concat: stageConcat,
	"pq-residuals": stagePqResiduals,
	"pq-init": stagePqInit,
	"pq-train-map": stagePqTrainMap,
	"pq-train-reduce": stagePqTrainReduce,
	assign: stageAssign,
	finalize: stageFinalize,
	publish: stagePublish,
	"publish-rev": stagePublishRev,
	compact: stageCompact,
	"compact-plan": stageCompactPlan,
	"compact-cells": stageCompactCells,
	"compact-commit": compactCommit,
	gc: stageGc,
	"delete-prefix": stageDeletePrefix
};
function authorized(request, env) {
	const token = env.BUILD_TOKEN;
	if (typeof token !== "string" || token === "") return false;
	const enc = new TextEncoder();
	const got = enc.encode(request.headers.get("authorization") ?? "");
	const want = enc.encode(`Bearer ${token}`);
	return got.byteLength === want.byteLength && crypto.subtle.timingSafeEqual(got, want);
}
//#endregion
//#region \0virtual:cloudflare/worker-entry
var worker_entry_default = { async fetch(request, env, ctx) {
	const url = new URL(request.url);
	const t0 = Date.now();
	try {
		if (url.pathname === "/health") {
			await ensureWasm();
			return json({
				ok: true,
				wasm: true,
				wasmResets
			});
		}
		if (!authorized(request, env)) throw new HttpError(401, "unauthorized");
		if (url.pathname === "/query" && request.method === "POST") return await handleQuery(request, env, ctx, url);
		if (url.pathname.startsWith("/build/") && request.method === "POST") {
			const stage = stages[url.pathname.slice(7)];
			if (!stage) throw new HttpError(404, "unknown stage");
			const body = await readJson(request);
			const result = await withWasmRecovery(`stage ${url.pathname}`, () => stage(env, body, ctx));
			return json({
				ok: true,
				ms: Date.now() - t0,
				result
			});
		}
		if (url.pathname === "/debug/range" && request.method === "GET") return await handleRangeProbe(env, ctx, url);
		if (url.pathname === "/upsert" && request.method === "POST") return await handleUpsert(request, env, ctx, url);
		if (url.pathname === "/delete" && request.method === "POST") return await handleDelete(request, env, url);
		if (url.pathname === "/manifest" && request.method === "GET") {
			const current = (await readCurrent(env))?.cur ?? null;
			const gen = url.searchParams.get("gen") ?? current?.gen;
			if (!gen) throw new HttpError(404, "no published index");
			const head = await latestRev(env, gen);
			const rev = optIntParam(url, "rev", 0) ?? head;
			const m = await readManifestR2(env, gen, rev);
			return json({
				gen,
				rev,
				head,
				current,
				manifest: {
					...m,
					notes: m.notes ? "(omitted)" : void 0
				}
			});
		}
		if (url.pathname === "/admin/refresh" && request.method === "POST") {
			currentPtr = null;
			genHeads.clear();
			return json({ ok: true });
		}
		if (url.pathname === "/admin/cas-check" && request.method === "POST") {
			const key = `tmp/cas-check/${crypto.randomUUID()}`;
			const first = await putIfAbsent(env, key, "a");
			const second = await putIfAbsent(env, key, "b");
			const head = await env.BUCKET.head(key);
			const casWrong = await env.BUCKET.put(key, "c", { onlyIf: { etagMatches: "0".repeat(32) } }) !== null;
			const casRight = await env.BUCKET.put(key, "d", { onlyIf: { etagMatches: head.etag } }) !== null;
			const final = await (await env.BUCKET.get(key)).text();
			await env.BUCKET.delete(key);
			const ok = first && !second && !casWrong && casRight && final === "d";
			return json({
				ok,
				createFirst: first,
				createSecond: second,
				casWrongEtag: casWrong,
				casRightEtag: casRight,
				final
			}, ok ? 200 : 500);
		}
		if (url.pathname === "/stat" && request.method === "GET") {
			const key = url.searchParams.get("key") ?? "";
			const head = await env.BUCKET.head(key);
			return json(head ? {
				key,
				size: head.size,
				etag: head.etag
			} : {
				key,
				missing: true
			}, head ? 200 : 404);
		}
		throw new HttpError(404, "not found");
	} catch (e) {
		if (e instanceof HttpError) return json({ error: e.message }, e.status);
		return json({
			error: e instanceof Error ? `${e.name}: ${e.message}\n${e.stack ?? ""}` : String(e),
			ms: Date.now() - t0
		}, 500);
	}
} };
//#endregion
export { worker_entry_default as default };
