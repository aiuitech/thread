/* @ts-self-types="./thread.d.ts" */

/**
 * Assign job: multi-assigns each vector to its `m` nearest fine cells (within
 * the `probe_coarse` best coarse groups; `0` or `>= k1` = all groups), encodes
 * the residual against each cell's mean in both tiers (4-bit MSE-clipped scan
 * code + int8 rerank code), and buffers records per coarse group. After
 * `set_codebook`, records also carry the residual's PQ codes (needed to
 * build format 3 slabs); without it their `pq_m` is 0.
 */
export class AssignJob {
    static __wrap(ptr) {
        const obj = Object.create(AssignJob.prototype);
        obj.__wbg_ptr = ptr;
        Object.defineProperty(obj, '__wbg_inst', { value: __wbg_instance_id, writable: true });

        AssignJobFinalization.register(obj, { ptr, instance: __wbg_instance_id }, obj);
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.assignjob_add_batch(this.__wbg_ptr, ptr0, len0, first_id);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(prev, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.assignjob_add_batch_prev(this.__wbg_ptr, ptr0, len0, first_id, ptr1, len1);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * @returns {number}
     */
    assigned_records() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        Object.defineProperty(this, '__wbg_inst', { value: __wbg_instance_id, writable: true });
        AssignJobFinalization.register(this, { ptr: ret[0], instance: __wbg_instance_id }, this);
        return this;
    }
    /**
     * Current serialized size of part `g`.
     * @param {number} g
     * @returns {number}
     */
    part_len(g) {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArray8ToWasm0(codebook_bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.assignjob_set_codebook(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArray8ToWasm0(tree_bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.assignjob_set_tree(this.__wbg_ptr, ptr0, len0, beam);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Serializes part `g` and resets its buffer.
     * @param {number} g
     * @returns {Uint8Array}
     */
    take_part(g) {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return AssignJob.__wrap(ret[0]);
    }
}
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
export class DeltaJob {
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(ids, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray32ToWasm0(attrs, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.deltajob_add(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v6 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v6;
    }
    /**
     * Serialized size of the segment so far.
     * @returns {number}
     */
    byte_len() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        Object.defineProperty(this, '__wbg_inst', { value: __wbg_instance_id, writable: true });
        DeltaJobFinalization.register(this, { ptr: ret[0], instance: __wbg_instance_id }, this);
        return this;
    }
    /**
     * PQ codes per entry (0 without a codebook).
     * @returns {number}
     */
    pq_m() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArray8ToWasm0(codebook_bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.deltajob_set_codebook(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Same as `set_codebook`, from a parsed `PqCodebook` (copied).
     * @param {PqCodebook} codebook
     */
    set_pq_codebook(codebook) {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        _assertClass(codebook, PqCodebook);
        if ((codebook).__wbg_inst !== undefined && (codebook).__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        let ret;
        __wbg_call_guard();
        ret = wasm.deltajob_set_pq_codebook(this.__wbg_ptr, codebook.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArray8ToWasm0(tree_bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.deltajob_set_tree(this.__wbg_ptr, ptr0, len0, beam);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) DeltaJob.prototype[Symbol.dispose] = DeltaJob.prototype.free;

/**
 * Picks initial fine centroids: a reservoir of up to `k2` sample vectors per
 * coarse group, streamed.
 */
export class FineInitJob {
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.fineinitjob_add_batch(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * `k1 * k2 * dim` initial fine means, grouped by coarse group.
     * @returns {Float32Array}
     */
    finish() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        Object.defineProperty(this, '__wbg_inst', { value: __wbg_instance_id, writable: true });
        FineInitJobFinalization.register(this, { ptr: ret[0], instance: __wbg_instance_id }, this);
        return this;
    }
}
if (Symbol.dispose) FineInitJob.prototype[Symbol.dispose] = FineInitJob.prototype.free;

/**
 * Flat k-means map job over `k` centroids (used for the coarse level).
 */
export class FlatKmeansJob {
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.flatkmeansjob_add_batch(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Convenience for single-job k-means: new means from this job's accumulator alone.
     * @param {Float32Array} prev
     * @returns {Float32Array}
     */
    means(prev) {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        Object.defineProperty(this, '__wbg_inst', { value: __wbg_instance_id, writable: true });
        FlatKmeansJobFinalization.register(this, { ptr: ret[0], instance: __wbg_instance_id }, this);
        return this;
    }
    /**
     * Serialized partial accumulator.
     * @returns {Uint8Array}
     */
    partial() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        let ret;
        __wbg_call_guard();
        ret = wasm.flatkmeansjob_partial(this.__wbg_ptr);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
}
if (Symbol.dispose) FlatKmeansJob.prototype[Symbol.dispose] = FlatKmeansJob.prototype.free;

/**
 * Incremental flat k-means map job: each vector is compared only against the
 * `n_neigh` neighbour cells of its previous cell (`neighbors`, a `k x n_neigh`
 * table from `cell_neighbors`), not all `k` centroids. Accumulates like
 * `TreeKmeansJob`; `partial()` has the same format, so `Reducer` is unchanged.
 */
export class IncKmeansJob {
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(prev, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.inckmeansjob_add_batch(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
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
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        Object.defineProperty(this, '__wbg_inst', { value: __wbg_instance_id, writable: true });
        IncKmeansJobFinalization.register(this, { ptr: ret[0], instance: __wbg_instance_id }, this);
        return this;
    }
    /**
     * @returns {Uint8Array}
     */
    partial() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        let ret;
        __wbg_call_guard();
        ret = wasm.inckmeansjob_partial(this.__wbg_ptr);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
}
if (Symbol.dispose) IncKmeansJob.prototype[Symbol.dispose] = IncKmeansJob.prototype.free;

/**
 * A parsed codebook to keep across queries (parsing ~1.5 MB per query is
 * avoidable): build each query's table with `PqTable.from_codebook`.
 */
export class PqCodebook {
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        let ret;
        __wbg_call_guard();
        ret = wasm.pqcodebook_dim(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    m() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        Object.defineProperty(this, '__wbg_inst', { value: __wbg_instance_id, writable: true });
        PqCodebookFinalization.register(this, { ptr: ret[0], instance: __wbg_instance_id }, this);
        return this;
    }
}
if (Symbol.dispose) PqCodebook.prototype[Symbol.dispose] = PqCodebook.prototype.free;

/**
 * Streaming reduce of PQ training partials: `add_partial` each partial
 * (so only one is held at a time), then `finish(prev_codebook)` returns the
 * next codebook. Empty sub-centroids keep their previous value.
 */
export class PqReducer {
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArray8ToWasm0(partial, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.pqreducer_add_partial(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Sub-centroids (out of `m * 256`) that received no residual.
     * @returns {number}
     */
    empty_centroids() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArray8ToWasm0(prev_codebook, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.pqreducer_finish(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        Object.defineProperty(this, '__wbg_inst', { value: __wbg_instance_id, writable: true });
        PqReducerFinalization.register(this, { ptr: ret[0], instance: __wbg_instance_id }, this);
        return this;
    }
    /**
     * Residuals over all added partials.
     * @returns {number}
     */
    rows() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        let ret;
        __wbg_call_guard();
        ret = wasm.pqreducer_rows(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) PqReducer.prototype[Symbol.dispose] = PqReducer.prototype.free;

/**
 * Per-query asymmetric-distance table (`m x 256` f32: `q_sub · sub-centroid`).
 * Build once per query, then `scan_pq` every probed v3 slab with it.
 */
export class PqTable {
    static __wrap(ptr) {
        const obj = Object.create(PqTable.prototype);
        obj.__wbg_ptr = ptr;
        Object.defineProperty(obj, '__wbg_inst', { value: __wbg_instance_id, writable: true });

        PqTableFinalization.register(obj, { ptr, instance: __wbg_instance_id }, obj);
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
        if ((codebook).__wbg_inst !== undefined && (codebook).__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        let ret;
        __wbg_call_guard();
        ret = wasm.pqtable_from_codebook(ptr0, len0, codebook.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return PqTable.__wrap(ret[0]);
    }
    /**
     * @returns {number}
     */
    m() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        Object.defineProperty(this, '__wbg_inst', { value: __wbg_instance_id, writable: true });
        PqTableFinalization.register(this, { ptr: ret[0], instance: __wbg_instance_id }, this);
        return this;
    }
}
if (Symbol.dispose) PqTable.prototype[Symbol.dispose] = PqTable.prototype.free;

/**
 * PQ training map job (one Lloyd iteration): assigns each residual's
 * subvectors to the nearest sub-centroids of the current codebook and
 * accumulates sums and counts. `partial()` is ~1.8 MB at dim 1536, m 192.
 */
export class PqTrainJob {
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(residuals, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.pqtrainjob_add_residuals(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Mean squared quantization error of the added residuals under the
     * job's (input) codebook.
     * @returns {number}
     */
    mean_sq_err() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        Object.defineProperty(this, '__wbg_inst', { value: __wbg_instance_id, writable: true });
        PqTrainJobFinalization.register(this, { ptr: ret[0], instance: __wbg_instance_id }, this);
        return this;
    }
    /**
     * Serialized partial accumulator ("VIBT"), for `pq_reduce` / `PqReducer`.
     * @returns {Uint8Array}
     */
    partial() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        let ret;
        __wbg_call_guard();
        ret = wasm.pqtrainjob_rows(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) PqTrainJob.prototype[Symbol.dispose] = PqTrainJob.prototype.free;

/**
 * Reduce job over a cell range: sums partial records, then emits new means.
 */
export class Reducer {
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArray8ToWasm0(records, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.reducer_add_records(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Per-cell counts after reduction (useful to spot empty cells).
     * @returns {Uint32Array}
     */
    counts() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(prev, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.reducer_means(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(prev, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.reducer_means_reseed(this.__wbg_ptr, ptr0, len0, min_count, seed);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
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
        Object.defineProperty(this, '__wbg_inst', { value: __wbg_instance_id, writable: true });
        ReducerFinalization.register(this, { ptr: ret, instance: __wbg_instance_id }, this);
        return this;
    }
}
if (Symbol.dispose) Reducer.prototype[Symbol.dispose] = Reducer.prototype.free;

/**
 * Query-side: routing table plus slab scanning.
 */
export class Searcher {
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.searcher_cell_dots(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(cells, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.searcher_cell_dots_for(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * @returns {number}
     */
    dim() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        Object.defineProperty(this, '__wbg_inst', { value: __wbg_instance_id, writable: true });
        SearcherFinalization.register(this, { ptr: ret[0], instance: __wbg_instance_id }, this);
        return this;
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.searcher_route_beam(this.__wbg_ptr, ptr0, len0, beam, nprobe);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.searcher_route(this.__wbg_ptr, ptr0, len0, probe_coarse, nprobe);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.searcher_route_flat(this.__wbg_ptr, ptr0, len0, nprobe);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(dots, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.searcher_route_from_dots(this.__wbg_ptr, ptr0, len0, nprobe);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArray8ToWasm0(tree_bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.searcher_set_tree(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Coarse group ids ranked for `q` (for lazy fine-centroid loading).
     * @param {Float32Array} q
     * @param {number} n
     * @returns {Uint32Array}
     */
    top_coarse(q, n) {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(sample, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.searcher_tree_recall(this.__wbg_ptr, ptr0, len0, beam);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(sample, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.searcher_tree_recall_top(this.__wbg_ptr, ptr0, len0, beam, m);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
}
if (Symbol.dispose) Searcher.prototype[Symbol.dispose] = Searcher.prototype.free;

/**
 * Finalize job for one fine cell: concatenates part record ranges into a slab.
 */
export class SlabJob {
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.slabjob_add_records(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArray8ToWasm0(span, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.slabjob_add_records_range(this.__wbg_ptr, ptr0, len0, span_offset, cell_off, cell_len);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * @returns {number}
     */
    byte_len() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        let ret;
        __wbg_call_guard();
        ret = wasm.slabjob_byte_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Uint8Array}
     */
    finish() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        let ret;
        __wbg_call_guard();
        ret = wasm.slabjob_format(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    len() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
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
        ret = wasm.slabjob_new(dim, cell, ptr0, len0, isLikeNone(format) ? 0xFFFFFF : format);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        Object.defineProperty(this, '__wbg_inst', { value: __wbg_instance_id, writable: true });
        SlabJobFinalization.register(this, { ptr: ret[0], instance: __wbg_instance_id }, this);
        return this;
    }
}
if (Symbol.dispose) SlabJob.prototype[Symbol.dispose] = SlabJob.prototype.free;

/**
 * Two-level k-means map job: assigns each vector to its best fine cell among
 * the `probe_coarse` nearest coarse groups (`0` or `>= k1` = all groups, as
 * `Searcher.route`) and accumulates per fine cell.
 */
export class TreeKmeansJob {
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
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.treekmeansjob_add_batch_assign(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * @param {Float32Array} vectors
     * @returns {number}
     */
    add_batch(vectors) {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ret;
        __wbg_call_guard();
        ret = wasm.treekmeansjob_add_batch(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
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
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        Object.defineProperty(this, '__wbg_inst', { value: __wbg_instance_id, writable: true });
        TreeKmeansJobFinalization.register(this, { ptr: ret[0], instance: __wbg_instance_id }, this);
        return this;
    }
    /**
     * @returns {Uint8Array}
     */
    partial() {
        if (this.__wbg_inst !== undefined && this.__wbg_inst !== __wbg_instance_id) {
            throw new Error('Invalid stale object from previous Wasm instance');
        }
        let ret;
        __wbg_call_guard();
        ret = wasm.treekmeansjob_partial(this.__wbg_ptr);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
}
if (Symbol.dispose) TreeKmeansJob.prototype[Symbol.dispose] = TreeKmeansJob.prototype.free;

export function __wbg_reset_state () {
    __wbg_instance_id++;
    cachedFloat32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    if (typeof numBytesDecoded !== 'undefined') numBytesDecoded = 0;
    if (typeof WASM_VECTOR_LEN !== 'undefined') WASM_VECTOR_LEN = 0;
    __wbg_reinit_scheduled = false;
    wasmInstance = new WebAssembly.Instance(wasmModule, __wbg_get_imports());
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
export function accum_record_range(dim, first, n) {
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
export function build_centroid_tree(means, dim, groups, iters, seed) {
    const ptr0 = passArrayF32ToWasm0(means, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.build_centroid_tree(ptr0, len0, dim, groups, iters, seed);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
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
export function cell_neighbors(means, dim, first, count, n) {
    const ptr0 = passArrayF32ToWasm0(means, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.cell_neighbors(ptr0, len0, dim, first, count, n);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Header summary of centroid tree bytes: `[dim, k, groups, n_members, byte_len]`.
 * @param {Uint8Array} tree_bytes
 * @returns {Uint32Array}
 */
export function centroid_tree_info(tree_bytes) {
    const ptr0 = passArray8ToWasm0(tree_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.centroid_tree_info(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
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
export function delta_cells(delta_bytes) {
    const ptr0 = passArray8ToWasm0(delta_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.delta_cells(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Header summary of a delta segment: `[dim, count, pq_m, seq, byte_len]`.
 * @param {Uint8Array} delta_bytes
 * @returns {Uint32Array}
 */
export function delta_info(delta_bytes) {
    const ptr0 = passArray8ToWasm0(delta_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.delta_info(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Payload of delta entry `idx` (a `scan_delta` hit's `idx`).
 * @param {Uint8Array} delta_bytes
 * @param {number} idx
 * @returns {Uint8Array}
 */
export function delta_payload(delta_bytes, idx) {
    const ptr0 = passArray8ToWasm0(delta_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.delta_payload(ptr0, len0, idx);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
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
export function delta_records_for_cell(delta_bytes, cell, tombstones) {
    const ptr0 = passArray8ToWasm0(delta_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(tombstones, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.delta_records_for_cell(ptr0, len0, cell, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Drops hits whose id is tombstoned from packed hits (one or more
 * concatenated `scan`/`merge` buffers); returns one packed buffer, order kept.
 * Applied to a finished top-k it can leave fewer than k hits: queries pass
 * the tombstones to `scan`/`scan_pq`/`scan_delta` instead.
 * @param {Uint8Array} packed
 * @param {Uint8Array} tombstones
 * @returns {Uint8Array}
 */
export function filter_hits(packed, tombstones) {
    const ptr0 = passArray8ToWasm0(packed, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(tombstones, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.filter_hits(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
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
export function init_panic_hook() {
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
export function merge(packed, k) {
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
export function normalize_vector(q) {
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
export function part_directory(bytes) {
    const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.part_directory(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * @param {number} k2
 * @returns {number}
 */
export function part_directory_len(k2) {
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
export function part_span(dir, first_cell, count) {
    const ptr0 = passArray32ToWasm0(dir, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.part_span(ptr0, len0, first_cell, count);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * `[dim, m, sub]` of a serialized codebook.
 * @param {Uint8Array} codebook_bytes
 * @returns {Uint32Array}
 */
export function pq_codebook_info(codebook_bytes) {
    const ptr0 = passArray8ToWasm0(codebook_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.pq_codebook_info(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
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
export function pq_init_codebook(residuals, dim, m, seed) {
    const ptr0 = passArrayF32ToWasm0(residuals, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.pq_init_codebook(ptr0, len0, dim, m, seed);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * One-shot reduce: `partials_concat` is `n_partials` `PqTrainJob.partial()`
 * buffers back to back (all the same size). Returns the next codebook.
 * Prefer `PqReducer` when the partials do not fit in memory together.
 * @param {Uint8Array} partials_concat
 * @param {number} n_partials
 * @param {Uint8Array} prev_codebook
 * @returns {Uint8Array}
 */
export function pq_reduce(partials_concat, n_partials, prev_codebook) {
    const ptr0 = passArray8ToWasm0(partials_concat, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(prev_codebook, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.pq_reduce(ptr0, len0, n_partials, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Re-encodes part records (decoded from their int8 tier against `old_mean`)
 * against `new_mean`: 4-bit, int8 and, with a codebook (empty bytes = none),
 * PQ codes. Ids, attrs and payloads are kept.
 * @param {Uint8Array} records
 * @param {Float32Array} old_mean
 * @param {Float32Array} new_mean
 * @param {Uint8Array} codebook_bytes
 * @returns {Uint8Array}
 */
export function reencode_records(records, old_mean, new_mean, codebook_bytes) {
    const ptr0 = passArray8ToWasm0(records, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(old_mean, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF32ToWasm0(new_mean, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(codebook_bytes, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.reencode_records(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v5;
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
export function rerank(q, centroid, records, n) {
    const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(centroid, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray8ToWasm0(records, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.rerank(ptr0, len0, ptr1, len1, ptr2, len2, n);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
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
export function residuals_for(vectors, means, assign, dim) {
    const ptr0 = passArrayF32ToWasm0(vectors, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(means, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray32ToWasm0(assign, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.residuals_for(ptr0, len0, ptr1, len1, ptr2, len2, dim);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
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
export function scan(slab_scan_bytes, q, k, tombstones, stats) {
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
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
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
export function scan_delta(delta_bytes, q, qc, k, tombstones, stats) {
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
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
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
export function scan_pq(slab_scan_bytes, table, q, k, tombstones, stats) {
    const ptr0 = passArray8ToWasm0(slab_scan_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    _assertClass(table, PqTable);
    if ((table).__wbg_inst !== undefined && (table).__wbg_inst !== __wbg_instance_id) {
        throw new Error('Invalid stale object from previous Wasm instance');
    }
    const ptr1 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray8ToWasm0(tombstones, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    var ptr3 = passArray32ToWasm0(stats, wasm.__wbindgen_malloc);
    var len3 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.scan_pq(ptr0, len0, table.__wbg_ptr, ptr1, len1, k, ptr2, len2, ptr3, len3, stats);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v5;
}

/**
 * The cell mean stored in a slab (or v2 scan prefix).
 * @param {Uint8Array} slab_scan_bytes
 * @returns {Float32Array}
 */
export function slab_centroid(slab_scan_bytes) {
    const ptr0 = passArray8ToWasm0(slab_scan_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.slab_centroid(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
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
export function slab_info(slab) {
    const ptr0 = passArray8ToWasm0(slab, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.slab_info(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Bytes to range-read for a scan, from the first 64 header bytes.
 * @param {Uint8Array} header_bytes
 * @returns {number}
 */
export function slab_scan_len(header_bytes) {
    const ptr0 = passArray8ToWasm0(header_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.slab_scan_len(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
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
export function slab_to_records(slab_bytes, tombstones) {
    const ptr0 = passArray8ToWasm0(slab_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(tombstones, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.slab_to_records(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * The same split for part records (e.g. `delta_records_for_cell` of the split
 * cell, or `slab_to_records`): decoded against `old_mean` (the cell's mean the
 * records were encoded with; its length is the dim). Same output layout as
 * `split_slab_records`.
 * @param {Uint8Array} records
 * @param {Float32Array} old_mean
 * @param {Float32Array} means2
 * @param {Uint8Array} codebook_bytes
 * @returns {Uint8Array}
 */
export function split_part_records(records, old_mean, means2, codebook_bytes) {
    const ptr0 = passArray8ToWasm0(records, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(old_mean, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF32ToWasm0(means2, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(codebook_bytes, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.split_part_records(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v5;
}

/**
 * Proposes a split of an overfull cell: decodes every entry of a whole slab
 * (any format; v2/v3 need the rerank section) as `mean + scale8 * code8`,
 * runs spherical 2-means (`kmeans::split_cell`, seeded with `seed`) and
 * returns the two new means (`2 * dim` floats). `mean` must be the slab's
 * cell mean; pass an empty array to use the mean stored in the slab.
 * Memory: `count * dim * 4` bytes of decoded vectors.
 * @param {Uint8Array} slab_bytes
 * @param {Float32Array} mean
 * @param {number} seed
 * @returns {Float32Array}
 */
export function split_slab(slab_bytes, mean, seed) {
    const ptr0 = passArray8ToWasm0(slab_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(mean, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.split_slab(ptr0, len0, ptr1, len1, seed);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v3;
}

/**
 * Executes a cell split: every entry of a whole slab (`slab_bytes`, not a scan
 * prefix) whose id is not in `tombstones` (`u32` LE, may be empty) is decoded
 * from its int8 tier against the slab's mean, assigned to the nearer (cosine)
 * of the two new means `means2` (`2 * dim` floats, e.g. from `split_slab`) and
 * re-encoded against it as a part record (4-bit, int8, and PQ codes when
 * `codebook_bytes` is a codebook; pass empty bytes for none, which only
 * format 1/2 slabs accept). Returns ONE buffer: a 4-byte `u32` LE prefix
 * holding the byte length `la` of records_a, then records_a (`[4, 4 + la)`),
 * then records_b (`[4 + la, end)`). Feed them to `SlabJob(dim, cell, meanA,
 * format)` and `SlabJob(dim, newCell, meanB, format)`.
 * @param {Uint8Array} slab_bytes
 * @param {Uint8Array} tombstones
 * @param {Float32Array} means2
 * @param {Uint8Array} codebook_bytes
 * @returns {Uint8Array}
 */
export function split_slab_records(slab_bytes, tombstones, means2, codebook_bytes) {
    const ptr0 = passArray8ToWasm0(slab_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(tombstones, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF32ToWasm0(means2, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray8ToWasm0(codebook_bytes, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.split_slab_records(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v5;
}

/**
 * The union of a tombstone buffer (`u32` LE ids, sorted unique; may be
 * empty) and `add`, serialized the same way.
 * @param {Uint8Array} existing
 * @param {Uint32Array} add
 * @returns {Uint8Array}
 */
export function tombstones_merge(existing, add) {
    const ptr0 = passArray8ToWasm0(existing, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray32ToWasm0(add, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    let ret;
    __wbg_call_guard();
    ret = wasm.tombstones_merge(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_copy_to_typed_array_88899a52af046901: function(arg0, arg1, arg2) {
            new Uint8Array(arg2.buffer, arg2.byteOffset, arg2.byteLength).set(getArrayU8FromWasm0(arg0, arg1));
        },
        __wbg___wbindgen_throw_41e9ee4f547fc59a: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_error_7b401e87950b0819: function(arg0, arg1) {
            console.error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_generic_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./thread_bg.js": import0,
    };
}

function __wbg_call_guard() {
    if (__wbg_reinit_scheduled) {
        __wbg_reset_state();
        return;
    }
}


let __wbg_instance_id = 0;
const AssignJobFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(({ ptr, instance }) => {
    if (instance === __wbg_instance_id) wasm.__wbg_assignjob_free(ptr, 1);
});
const DeltaJobFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(({ ptr, instance }) => {
    if (instance === __wbg_instance_id) wasm.__wbg_deltajob_free(ptr, 1);
});
const FineInitJobFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(({ ptr, instance }) => {
    if (instance === __wbg_instance_id) wasm.__wbg_fineinitjob_free(ptr, 1);
});
const FlatKmeansJobFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(({ ptr, instance }) => {
    if (instance === __wbg_instance_id) wasm.__wbg_flatkmeansjob_free(ptr, 1);
});
const IncKmeansJobFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(({ ptr, instance }) => {
    if (instance === __wbg_instance_id) wasm.__wbg_inckmeansjob_free(ptr, 1);
});
const PqCodebookFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(({ ptr, instance }) => {
    if (instance === __wbg_instance_id) wasm.__wbg_pqcodebook_free(ptr, 1);
});
const PqReducerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(({ ptr, instance }) => {
    if (instance === __wbg_instance_id) wasm.__wbg_pqreducer_free(ptr, 1);
});
const PqTableFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(({ ptr, instance }) => {
    if (instance === __wbg_instance_id) wasm.__wbg_pqtable_free(ptr, 1);
});
const PqTrainJobFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(({ ptr, instance }) => {
    if (instance === __wbg_instance_id) wasm.__wbg_pqtrainjob_free(ptr, 1);
});
const ReducerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(({ ptr, instance }) => {
    if (instance === __wbg_instance_id) wasm.__wbg_reducer_free(ptr, 1);
});
const SearcherFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(({ ptr, instance }) => {
    if (instance === __wbg_instance_id) wasm.__wbg_searcher_free(ptr, 1);
});
const SlabJobFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(({ ptr, instance }) => {
    if (instance === __wbg_instance_id) wasm.__wbg_slabjob_free(ptr, 1);
});
const TreeKmeansJobFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(({ ptr, instance }) => {
    if (instance === __wbg_instance_id) wasm.__wbg_treekmeansjob_free(ptr, 1);
});

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
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

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function isLikeNone(x) {
    return x === undefined || x === null;
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

let __wbg_reinit_scheduled = false;

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('thread_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
