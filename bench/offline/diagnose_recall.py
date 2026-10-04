#!/usr/bin/env python3
"""Offline recall diagnosis for a published index generation (default g1).

Replicates the build-time routing/assignment rule against the published
centroids (data/idx-g1/{coarse,fine}.f32) and measures, without touching R2:

  1. routing recall upper bound (fraction of true top-10 neighbours with at
     least one assigned cell among the probed cells), for several m / pc / nprobe,
     plus expected entries scanned per query (from replicated slab sizes);
  2. quantization-only recall@10 (residual SQ8) over the exact top-50;
  3. cell balance of the primary fine cell + near-duplicate fine means;
  4. (--retrain) a counterfactual: flat spherical k-means (k=4096) trained
     offline on a larger strided sample with more iterations and tiny-cell
     re-seeding, evaluated with the same routing metrics, to see whether better
     clustering would move the recall/bytes curve.

One streaming pass over vectors.npy (chunks of CHUNK rows) does the assignment
of all vectors (top-MMAX cells, coarse pc=2) and the brute-force top-50 for the
quantization queries. Peak memory is well under 1 GB.

Usage: .venv/bin/python bench/offline/diagnose_recall.py [--q2 300] [--chunk 4096] [--retrain 1]
Results: printed, and data/idx-g1/diagnosis.json. Intermediate passes are
cached as .npz under data/idx-g1/ (delete them to recompute).
"""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--q2", type=int, default=300, help="queries for quantization study")
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--mmax", type=int, default=8)
ap.add_argument("--build-pc", type=int, default=2)
ap.add_argument("--idx", default="data/idx-g1")
ap.add_argument("--cache", default="data/idx-g1/assign_top8.npz")
ap.add_argument("--retrain", type=int, default=1)
ap.add_argument("--rt-sample", type=int, default=160000)
ap.add_argument("--rt-iters", type=int, default=20)
ap.add_argument("--target", type=float, default=0.95)
args = ap.parse_args()

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
D, K1, K2 = 1536, 64, 64
NC = K1 * K2
MMAX = args.mmax
t0 = time.time()
def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

V = np.load("data/vectors.npy", mmap_mode="r")
N = V.shape[0]
Q = np.fromfile("data/prep/queries.f32", dtype="<f4").reshape(-1, D)
GT = np.fromfile("data/prep/groundtruth.u32", dtype="<u4").reshape(Q.shape[0], -1).astype(np.int64)
NQ = Q.shape[0]
C = np.fromfile(os.path.join(args.idx, "coarse.f32"), dtype="<f4").reshape(K1, D)
F = np.fromfile(os.path.join(args.idx, "fine.f32"), dtype="<f4").reshape(NC, D)
Cn = (C / np.linalg.norm(C, axis=1, keepdims=True)).astype(np.float32)
Fnorm = np.linalg.norm(F, axis=1)
Fn = (F / Fnorm[:, None]).astype(np.float32)
CHILD = np.arange(K2)


def route(X, pc, top):
    """Top-`top` fine cells (ranked) among children of the top-`pc` coarse groups.
    Cosine against a mean = dot / ||mean||. Returns (ids, scores, full fine scores)."""
    cs = X @ Cn.T
    g = np.argpartition(-cs, pc - 1, axis=1)[:, :pc] if pc < K1 else np.tile(np.arange(K1), (len(X), 1))
    cand = (g[:, :, None] * K2 + CHILD).reshape(len(X), -1)
    fs = X @ Fn.T
    cs2 = np.take_along_axis(fs, cand, axis=1)
    top = min(top, cand.shape[1])
    part = np.argpartition(-cs2, top - 1, axis=1)[:, :top]
    ps = np.take_along_axis(cs2, part, axis=1)
    order = np.argsort(-ps, axis=1)
    part = np.take_along_axis(part, order, axis=1)
    return np.take_along_axis(cand, part, axis=1), np.take_along_axis(ps, order, axis=1), fs


# ---------------------------------------------------------------- streaming pass
Q2 = np.ascontiguousarray(Q[: args.q2])
K50 = 50
if os.path.exists(args.cache):
    z = np.load(args.cache)
    A, prim_cos, flat_best_in, bf_s, bf_i = z["A"], z["prim_cos"], z["flat_best_in"], z["bf_s"], z["bf_i"]
    log("loaded cached pass", args.cache)
else:
    A = np.empty((N, MMAX), dtype=np.uint16)          # ranked top-MMAX cells per vector
    prim_cos = np.empty(N, dtype=np.float32)          # cosine to primary fine mean
    flat_best_in = np.empty(N, dtype=bool)            # global best fine cell reachable via top-2 coarse?
    bf_s = np.full((len(Q2), K50), -np.inf, dtype=np.float32)
    bf_i = np.full((len(Q2), K50), -1, dtype=np.int64)
    norms_dev = 0.0
    for s in range(0, N, args.chunk):
        X = np.asarray(V[s:s + args.chunk], dtype=np.float32)
        nr = np.linalg.norm(X, axis=1)
        norms_dev = max(norms_dev, float(np.abs(nr - 1).max()))
        X = X / nr[:, None]
        ids, sc, fs = route(X, args.build_pc, MMAX)
        A[s:s + len(X)] = ids
        prim_cos[s:s + len(X)] = sc[:, 0]
        flat_best_in[s:s + len(X)] = fs.argmax(1) == ids[:, 0]
        del fs
        S = Q2 @ X.T
        allS = np.concatenate([bf_s, S], axis=1)
        allI = np.concatenate([bf_i, np.broadcast_to(np.arange(s, s + len(X)), S.shape)], axis=1)
        p = np.argpartition(-allS, K50 - 1, axis=1)[:, :K50]
        bf_s = np.take_along_axis(allS, p, 1)
        bf_i = np.take_along_axis(allI, p, 1)
        if (s // args.chunk) % 40 == 0:
            log(f"pass {s}/{N}")
    o = np.argsort(-bf_s, axis=1)
    bf_s = np.take_along_axis(bf_s, o, 1); bf_i = np.take_along_axis(bf_i, o, 1)
    log(f"pass done; max | ||x|| - 1 | = {norms_dev:.2e}")
    np.savez(args.cache, A=A, prim_cos=prim_cos, flat_best_in=flat_best_in, bf_s=bf_s, bf_i=bf_i)

out = {}

# ------------------------------------------------- validation vs published slabs
slabs = json.load(open("data/slabs-g1.json"))
pub = np.zeros(NC, dtype=np.int64)
for e in slabs:
    pub[e["cell"]] = e["count"]
bytes_per_entry = sum(e["bytes"] for e in slabs) / pub.sum()
sizes = {m: np.bincount(A[:, :m].ravel().astype(np.int64), minlength=NC) for m in (1, 2, 3, 4, 6, 8) if m <= MMAX}
match3 = float((sizes[3] == pub).mean())
l1 = int(np.abs(sizes[3] - pub).sum())
log(f"replicated m=3 slab sizes: exact match on {match3*100:.2f}% of cells, L1 diff {l1} of {pub.sum()} entries; "
    f"bytes/entry {bytes_per_entry:.0f}")
out["validation"] = dict(match_frac=match3, l1=l1, bytes_per_entry=bytes_per_entry)
gt_ok = float(np.mean([len(set(GT[i]) & set(bf_i[i, :10])) / 10 for i in range(len(Q2))]))
log(f"ground truth vs brute force top-10 overlap on {len(Q2)} queries: {gt_ok*100:.2f}%")

# ---------------------------------------------------- 1. routing recall
PCS = [2, 4, 8, 64]
NPROBES = [1, 2, 4, 8, 16, 32, 64]
MS = [1, 3, 6, 8]
rank = {}
for pc in PCS:
    pos = np.full((NQ, NC), 32767, dtype=np.int16)
    probes = np.empty((NQ, min(256, pc * K2)), dtype=np.int64)
    for s in range(0, NQ, 1000):
        ids, _, _ = route(Q[s:s + 1000], pc, 256)
        probes[s:s + 1000] = ids
        np.put_along_axis(pos[s:s + 1000], ids, np.arange(ids.shape[1], dtype=np.int16)[None, :].repeat(len(ids), 0), 1)
    rank[pc] = (pos, probes)

nb_cells = A[GT].astype(np.int64)  # (NQ, 10, MMAX)
def routing(pc, m, nprobe, qs=slice(None)):
    pos, _ = rank[pc]
    pq = pos[qs]
    best = np.take_along_axis(pq, nb_cells[qs, :, :m].reshape(pq.shape[0], -1), 1).reshape(pq.shape[0], 10, m).min(2)
    return float((best < nprobe).mean())

def scanned(pc, m, nprobe, qs=slice(None)):
    _, probes = rank[pc]
    return float(sizes[m][probes[qs, :nprobe]].sum(1).mean())

tab1 = []
for m in MS:
    for pc in PCS:
        for npb in NPROBES:
            if npb > pc * K2:
                continue
            tab1.append(dict(m=m, pc=pc, nprobe=npb, recall_5000=routing(pc, m, npb),
                             recall_first500=routing(pc, m, npb, slice(0, 500)),
                             scanned=scanned(pc, m, npb)))
out["routing"] = tab1


def min_nprobe_table(rank_by_pc, A_, sizes_, ms, target, label):
    """Smallest nprobe reaching `target` routing recall, per (m, pc), and its cost."""
    nbc = A_[GT].astype(np.int64)
    rows = []
    for m in ms:
        for pc, (pos, probes) in rank_by_pc.items():
            best = np.take_along_axis(pos, nbc[:, :, :m].reshape(NQ, -1), 1).reshape(NQ, 10, m).min(2)
            curve = np.array([(best < n).mean() for n in range(1, probes.shape[1] + 1)])
            hit = np.nonzero(curve >= target)[0]
            if len(hit):
                n = int(hit[0]) + 1
                sc = float(sizes_[m][probes[:, :n]].sum(1).mean())
                rows.append(dict(clustering=label, m=m, pc=pc, nprobe=n, recall=float(curve[n - 1]), scanned=sc,
                                 mb=sc * bytes_per_entry / 1e6, storage_gb=m * N * bytes_per_entry / 1e9))
            else:
                rows.append(dict(clustering=label, m=m, pc=pc, nprobe=None, recall=float(curve[-1]), scanned=None,
                                 mb=None, storage_gb=m * N * bytes_per_entry / 1e9))
    return rows

out["min_nprobe"] = min_nprobe_table(rank, A, sizes, MS, args.target, "g1")

# rank (among probed cells) needed to cover all 10 neighbours, per query
def needed_rank(pc, m):
    pos, _ = rank[pc]
    best = np.take_along_axis(pos, nb_cells[:, :, :m].reshape(NQ, -1), 1).reshape(NQ, 10, m).min(2)
    return best  # (NQ,10) rank of first probed cell holding each neighbour

# ---------------------------------------------------- 2. quantization-only
ids50 = bf_i
uniq = np.unique(ids50)
Xc = np.empty((len(uniq), D), dtype=np.float32)
for s in range(0, len(uniq), 4096):
    blk = uniq[s:s + 4096]
    Xc[s:s + 4096] = V[blk]
Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
where = np.searchsorted(uniq, ids50)

def sq8(X, cells):
    M = F[cells]
    R = X - M
    scale = np.abs(R).max(1) / 127.0
    code = np.clip(np.rint(R / scale[:, None]), -127, 127).astype(np.int8)
    return M, scale, code

qres = {}
for label, col in (("primary", 0), ("3rd cell (worst of m=3)", 2), ("8th cell", 7)):
    cells = A[uniq, col].astype(np.int64)
    M, scale, code = sq8(Xc, cells)
    rec, err, err_rel, gap = [], [], [], []
    for i in range(len(Q2)):
        q = Q2[i]
        w = where[i]
        exact = Xc[w] @ q
        approx = M[w] @ q + scale[w] * (code[w].astype(np.float32) @ q)
        ex_top = set(np.argsort(-exact)[:10]); ap_top = set(np.argsort(-approx)[:10])
        rec.append(len(ex_top & ap_top) / 10)
        err.append(np.abs(approx - exact))
        se = np.sort(exact)[::-1]
        gap.append(se[9] - se[10])
    err = np.concatenate(err)
    qres[label] = dict(recall10=float(np.mean(rec)), mae=float(err.mean()), p99=float(np.quantile(err, 0.99)),
                       max=float(err.max()), mean_scale=float(scale.mean()))
gaps = np.array(gap)
out["quant"] = qres
out["quant_context"] = dict(top1=float(bf_s[:, 0].mean()), top10=float(bf_s[:, 9].mean()), top50=float(bf_s[:, 49].mean()),
                            gap10_11_median=float(np.median(gaps)), gap10_11_p10=float(np.quantile(gaps, 0.1)))

# ---------------------------------------------------- 3. cell balance
p1 = sizes[1]
qs_ = [0, 10, 50, 90, 99, 100]
bal = {f"p{q}": float(np.percentile(p1, q)) for q in qs_}
bal["empty"] = int((p1 == 0).sum())
bal["frac_vectors_in_cells_gt2000"] = float(p1[p1 > 2000].sum() / N)
bal["cells_gt2000"] = int((p1 > 2000).sum())
bal["ideal"] = N / NC
bal["cv"] = float(p1.std() / p1.mean())
bal["vectors_in_top10pct_cells"] = float(np.sort(p1)[::-1][: NC // 10].sum() / N)
s3 = sizes[3]
bal["m3"] = {f"p{q}": float(np.percentile(s3, q)) for q in qs_}
coarse_sz = np.bincount(A[:, 0].astype(np.int64) // K2, minlength=K1)
bal["coarse_primary"] = dict(min=int(coarse_sz.min()), p50=float(np.median(coarse_sz)), max=int(coarse_sz.max()), ideal=N / K1)
pg = p1.reshape(K1, K2)
bal["fine_size_var_explained_by_coarse_group"] = float(((pg.mean(1) - p1.mean()) ** 2).mean() / p1.var())
bal["cells_lt50_in_5_smallest_groups"] = int((pg[np.argsort(coarse_sz)[:5]] < 50).sum())
bal["cells_lt50_in_5_largest_groups"] = int((pg[np.argsort(coarse_sz)[-5:]] < 50).sum())
bal["flat_best_reachable"] = float(flat_best_in.mean())
bal["mean_cos_primary"] = float(prim_cos.mean())
bal["fine_norm"] = dict(min=float(Fnorm.min()), p50=float(np.median(Fnorm)), max=float(Fnorm.max()))
# near duplicate fine means within a coarse group
dup_pairs, dup_cells, maxcos = 0, set(), []
for g in range(K1):
    B = Fn[g * K2:(g + 1) * K2]
    G = B @ B.T
    np.fill_diagonal(G, -1)
    iu = np.triu_indices(K2, 1)
    vals = G[iu]
    maxcos.append(G.max())
    hit = vals > 0.99
    dup_pairs += int(hit.sum())
    for a, b in zip(iu[0][hit], iu[1][hit]):
        dup_cells.add(g * K2 + a); dup_cells.add(g * K2 + b)
bal["dup_pairs_gt0.99"] = dup_pairs
bal["dup_cells_gt0.99"] = len(dup_cells)
allpair = []
for g in range(K1):
    B = Fn[g * K2:(g + 1) * K2]; G = B @ B.T
    allpair.append(G[np.triu_indices(K2, 1)])
allpair = np.concatenate(allpair)
bal["pair_cos_within_group"] = {"p50": float(np.median(allpair)), "p99": float(np.quantile(allpair, 0.99)),
                                "max": float(allpair.max()), "gt0.95": int((allpair > 0.95).sum()),
                                "gt0.9": int((allpair > 0.9).sum())}
# exact duplicate means (identical rows)
_, cnt = np.unique(F.round(6), axis=0, return_counts=True)
bal["identical_fine_means_extra"] = int((cnt - 1).sum())
bal["tiny_cells_lt50"] = int((p1 < 50).sum())
out["balance"] = bal

# ---------------------------------------------------- print
def pct(x): return f"{100*x:5.1f}%"
print("\n=== 1. routing recall upper bound (5000 queries; measured-500 subset in brackets) ===")
for m in MS:
    print(f"m={m}")
    for pc in PCS:
        rows = [r for r in tab1 if r["m"] == m and r["pc"] == pc]
        print(f"  pc={pc:2d}: " + "  ".join(f"np{r['nprobe']}:{pct(r['recall_5000'])}[{pct(r['recall_first500'])}]/{r['scanned']:.0f}" for r in rows))
print("\n=== 2. quantization only (top-50 exact candidates, %d queries) ===" % len(Q2))
for k, v in qres.items(): print(f"  {k}: {v}")
print("  context:", out["quant_context"])
print("\n=== 3. balance ===")
print(json.dumps(bal, indent=1))
def print_min(rows):
    for r in rows:
        if r["nprobe"] is None:
            print(f"  {r['clustering']:>10} m={r['m']} pc={r['pc']:>4}: not reached (max {pct(r['recall'])})")
        else:
            print(f"  {r['clustering']:>10} m={r['m']} pc={r['pc']:>4}: nprobe={r['nprobe']:4d} recall {pct(r['recall'])} "
                  f"scanned {r['scanned']:8.0f}  read {r['mb']:6.1f} MB/query  storage {r['storage_gb']:.1f} GB")
print(f"\n=== min nprobe for routing recall >= {args.target:.0%} (5000 queries) ===")
print_min(out["min_nprobe"])

# ---------------------------------------------------- 4. counterfactual clustering
if args.retrain:
    rt_cache = os.path.join(args.idx, "retrain_flat.npz")
    if os.path.exists(rt_cache):
        z = np.load(rt_cache); Fr, Ar = z["F"], z["A"]
        log("loaded cached retrain", rt_cache)
    else:
        stride = N // args.rt_sample
        idx = np.arange(0, N, stride)[: args.rt_sample]
        S = np.empty((len(idx), D), dtype=np.float32)
        for s in range(0, len(idx), 8192):
            S[s:s + 8192] = V[idx[s:s + 8192]]
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        rng = np.random.default_rng(1)
        Fr = S[rng.choice(len(S), NC, replace=False)].copy()
        log(f"retrain: sample {len(S)}, k={NC}, iters {args.rt_iters}")
        for it in range(args.rt_iters):
            sums = np.zeros((NC, D), dtype=np.float64)
            cnt = np.zeros(NC, dtype=np.int64)
            obj = 0.0
            for s in range(0, len(S), 16384):
                X = S[s:s + 16384]
                sc = X @ Fr.T
                lab = sc.argmax(1)
                obj += float(sc[np.arange(len(X)), lab].sum())
                o = np.argsort(lab, kind="stable")
                ul, st = np.unique(lab[o], return_index=True)
                sums[ul] += np.add.reduceat(X[o], st, axis=0)
                cnt += np.bincount(lab, minlength=NC)
            Fr = (sums / np.maximum(cnt, 1)[:, None]).astype(np.float32)
            Fr /= np.linalg.norm(Fr, axis=1, keepdims=True)  # spherical k-means
            # re-seed empty / tiny cells by splitting the largest ones
            tiny = np.nonzero(cnt < 5)[0]
            if len(tiny) and it < args.rt_iters - 3:
                big = np.argsort(-cnt)[: len(tiny)]
                for t, b in zip(tiny, big):
                    jit = rng.standard_normal(D).astype(np.float32) * 1e-3
                    Fr[t] = Fr[b] + jit; Fr[b] = Fr[b] - jit
                Fr /= np.linalg.norm(Fr, axis=1, keepdims=True)
            log(f"  iter {it+1}: mean cos {obj/len(S):.4f}, min cell {cnt.min()}, max {cnt.max()}, reseeded {len(tiny)}")
        del S
        Ar = np.empty((N, MMAX), dtype=np.uint16)
        for s in range(0, N, args.chunk):
            X = np.asarray(V[s:s + args.chunk], dtype=np.float32)
            sc = X @ Fr.T
            p = np.argpartition(-sc, MMAX - 1, axis=1)[:, :MMAX]
            o = np.argsort(-np.take_along_axis(sc, p, 1), axis=1)
            Ar[s:s + len(X)] = np.take_along_axis(p, o, 1)
        np.savez(rt_cache, F=Fr, A=Ar)
        log("retrain: assigned all vectors")
    sizes_r = {m: np.bincount(Ar[:, :m].ravel().astype(np.int64), minlength=NC) for m in (1, 3, 6, 8) if m <= MMAX}
    sc = Q @ Fr.T
    probes = np.argsort(-sc, axis=1)[:, :256]
    pos = np.full((NQ, NC), 32767, dtype=np.int16)
    np.put_along_axis(pos, probes, np.arange(256, dtype=np.int16)[None, :].repeat(NQ, 0), 1)
    rank_r = {"flat": (pos, probes)}
    nbr = Ar[GT].astype(np.int64)
    rt_tab = []
    for m in MS:
        row = []
        for npb in NPROBES:
            best = np.take_along_axis(pos, nbr[:, :, :m].reshape(NQ, -1), 1).reshape(NQ, 10, m).min(2)
            row.append(dict(m=m, nprobe=npb, recall=float((best < npb).mean()),
                            scanned=float(sizes_r[m][probes[:, :npb]].sum(1).mean())))
        rt_tab += row
    p1r = sizes_r[1]
    out["retrain"] = dict(
        routing=rt_tab,
        min_nprobe=min_nprobe_table(rank_r, Ar, sizes_r, MS, args.target, "retrained"),
        balance={**{f"p{q}": float(np.percentile(p1r, q)) for q in qs_},
                 "frac_vectors_in_cells_gt2000": float(p1r[p1r > 2000].sum() / N), "cv": float(p1r.std() / p1r.mean()),
                 "tiny_cells_lt50": int((p1r < 50).sum())},
    )
    print("\n=== 4. retrained flat k-means (k=4096) routing recall / scanned ===")
    for m in MS:
        print(f"  m={m}: " + "  ".join(f"np{r['nprobe']}:{pct(r['recall'])}/{r['scanned']:.0f}" for r in rt_tab if r["m"] == m))
    print("  balance:", out["retrain"]["balance"])
    print(f"  min nprobe for >= {args.target:.0%}:")
    print_min(out["retrain"]["min_nprobe"])

json.dump(out, open(os.path.join(args.idx, "diagnosis.json"), "w"), indent=1)
log("wrote", os.path.join(args.idx, "diagnosis.json"))
