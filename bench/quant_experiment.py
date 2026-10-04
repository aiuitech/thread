#!/usr/bin/env python3
"""Residual-encoding experiment for the slab scan tier (index generation g1).

Part 1 (candidate level). For NQ queries and their exact top-K candidates (f32
cosine, brute force), each candidate's residual against its PRIMARY fine cell
mean is encoded with several schemes. We report recall@10 of each scheme's
ranking (over the top-50 and over the top-K pool) vs the exact top-10, absolute
score error, and bytes per slab entry, plus a two-tier variant (scheme top-T,
reranked with int8 residual scores).

Part 2 (end to end). A second streaming pass encodes ALL vectors (against their
primary cell) with a few selected schemes and ranks, per query, every vector
reachable with flat routing over the 4096 fine means at nprobe P on the g1 m=3
slabs (the ~28k-entry scan set of the 95% configuration). This gives
end-to-end recall@10 (routing + quantization), and two-tier recall where the
scan tier's top-T is drawn from the full scan set, not just from the exact top-K.

Scores are q . (mean + decoded residual), which is what the code-domain kernels
(int dot with a per-query table, ADC tables for PQ) compute up to f32 rounding.

Schemes
  a   int8 residual, per-vector scale max|r|/127 (baseline)
  b   4-bit residual, levels -7..7, per-vector scale max|r|/7
  b'  4-bit, per-vector scale chosen to minimise reconstruction MSE (clipping)
  b'' 4-bit, random orthogonal rotation of the residual, then as b'
  c   4-bit, residual normalised per dim by the cell's member bias/std
      (stats from the cell's primary members, 2 x 1536 f32 per slab header), then as b
  c'  as c with the clipped scale of b'
  d   6-bit, levels -31..31 (max scale); d' clipped
  5b  5-bit clipped (960 B)
  e   PQ on residuals, one codebook for all cells, 256 centroids per subspace,
      M=192 x 8 dims (192 B) and M=96 x 16 dims (96 B); k-means on a 100k sample

Bytes/entry = code bytes + 16 (id / entry) + 4 (per-vector scale; none for PQ).

Memory: vectors are read with pread-style np.fromfile (not a memmap), so the
6 GB file lands in page cache rather than this process's RSS.

Usage: .venv/bin/python bench/quant_experiment.py [--nq 500] [--k 100] [--nprobe 22]
Results: printed, and data/idx-g1/quant_experiment.json. Passes are cached in
data/idx-g1/quant_pass_q{nq}_k{k}.npz, quant_pq.npz, quant_e2e_q{nq}_np{P}.npz.
"""
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--nq", type=int, default=500)
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--pq-sample", type=int, default=100000)
ap.add_argument("--pq-iters", type=int, default=15)
ap.add_argument("--nprobe", type=int, default=22)
ap.add_argument("--m", type=int, default=3)
ap.add_argument("--e2e", type=int, default=1, help="run the full end-to-end pass (part 2)")
ap.add_argument("--idx", default="data/idx-g1")
args = ap.parse_args()

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
D, NC = 1536, 4096
t0 = time.time()
def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

VPATH = "data/vectors.npy"
_vm = np.load(VPATH, mmap_mode="r")
N, VOFF = _vm.shape[0], _vm.offset
assert _vm.dtype.str == "<f4" and _vm.shape[1] == D
del _vm

def read_rows(s, e):
    """Rows [s, e) of vectors.npy, normalised, without mapping the file."""
    X = np.fromfile(VPATH, dtype="<f4", count=(e - s) * D, offset=VOFF + s * D * 4).reshape(-1, D)
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def gather_rows(ids):
    out = np.empty((len(ids), D), dtype=np.float32)
    with open(VPATH, "rb") as f:
        fd = f.fileno()
        for j, i in enumerate(ids):
            out[j] = np.frombuffer(os.pread(fd, D * 4, VOFF + int(i) * D * 4), dtype="<f4")
    return out / np.linalg.norm(out, axis=1, keepdims=True)

Q = np.fromfile("data/prep/queries.f32", dtype="<f4").reshape(-1, D)[: args.nq].copy()
GT = np.fromfile("data/prep/groundtruth.u32", dtype="<u4").reshape(5000, -1)[: args.nq].astype(np.int64)
F = np.fromfile(os.path.join(args.idx, "fine.f32"), dtype="<f4").reshape(NC, D)
Fn = (F / np.linalg.norm(F, axis=1, keepdims=True)).astype(np.float32)
A = np.load(os.path.join(args.idx, "assign_top8.npz"))["A"]       # build-rule (pc=2) top-8 cells per vector
P = A[:, 0].astype(np.int64)                                       # primary cell
K = args.k
NQ = args.nq
sample_idx = np.linspace(0, N - 1, args.pq_sample).astype(np.int64)

# ------------------------------------------------------------------ pass 1
pass_cache = os.path.join(args.idx, f"quant_pass_q{NQ}_k{K}.npz")
pq_cache = os.path.join(args.idx, "quant_pq.npz")
need_sample = not os.path.exists(pq_cache)
Rs = None
if os.path.exists(pass_cache) and not need_sample:
    z = np.load(pass_cache)
    bf_s, bf_i, csum, csq, ccnt = z["bf_s"], z["bf_i"], z["csum"], z["csq"], z["ccnt"]
    log("loaded cached pass", pass_cache)
else:
    bf_s = np.full((NQ, K), -np.inf, dtype=np.float32)
    bf_i = np.full((NQ, K), -1, dtype=np.int64)
    csum = np.zeros((NC, D), dtype=np.float64)
    csq = np.zeros((NC, D), dtype=np.float64)
    ccnt = np.bincount(P, minlength=NC)
    if need_sample:
        Rs = np.empty((len(sample_idx), D), dtype=np.float32)
    for s in range(0, N, args.chunk):
        X = read_rows(s, min(N, s + args.chunk))
        n = len(X)
        S = Q @ X.T
        allS = np.concatenate([bf_s, S], axis=1)
        allI = np.concatenate([bf_i, np.broadcast_to(np.arange(s, s + n), S.shape)], axis=1)
        p = np.argpartition(-allS, K - 1, axis=1)[:, :K]
        bf_s = np.take_along_axis(allS, p, 1); bf_i = np.take_along_axis(allI, p, 1)
        lab = P[s:s + n]
        R = X - F[lab]
        o = np.argsort(lab, kind="stable")
        ul, st = np.unique(lab[o], return_index=True)
        Ro = R[o]
        csum[ul] += np.add.reduceat(Ro, st, axis=0)
        csq[ul] += np.add.reduceat(Ro.astype(np.float64) ** 2, st, axis=0)
        if Rs is not None:
            lo, hi = np.searchsorted(sample_idx, [s, s + n])
            Rs[lo:hi] = R[sample_idx[lo:hi] - s]
        del X, R, Ro, S, allS, allI
        if (s // args.chunk) % 20 == 0:
            log(f"pass 1: {s}/{N}")
    o = np.argsort(-bf_s, axis=1)
    bf_s = np.take_along_axis(bf_s, o, 1); bf_i = np.take_along_axis(bf_i, o, 1)
    np.savez(pass_cache, bf_s=bf_s, bf_i=bf_i, csum=csum.astype(np.float32), csq=csq.astype(np.float32), ccnt=ccnt)
    log("pass 1 done")

gt_ok = float(np.mean([len(set(GT[i]) & set(bf_i[i, :10])) / 10 for i in range(NQ)]))
log(f"ground truth vs brute-force top-10 overlap: {gt_ok*100:.2f}%")

# per-cell residual bias / std from primary members; cells with < 16 members use the global per-dim std
ccnt = ccnt.astype(np.float64)
csum = csum.astype(np.float64); csq = csq.astype(np.float64)
cmean = csum / np.maximum(ccnt, 1)[:, None]
cvar = csq / np.maximum(ccnt, 1)[:, None] - cmean ** 2
gstd = np.sqrt(csq.sum(0) / ccnt.sum() - (csum.sum(0) / ccnt.sum()) ** 2)
small = ccnt < 16
cstd = np.sqrt(np.maximum(cvar, 0))
cstd[small] = gstd
cmean[small] = 0
cstd = np.maximum(cstd, 1e-3 * gstd.mean()).astype(np.float32)
cmean = cmean.astype(np.float32)
del csum, csq, cvar
log(f"cell stats: residual std per dim, mean over dims {gstd.mean():.4f}; cells < 16 members: {int(small.sum())}")

# ------------------------------------------------------------------ PQ
def kmeans_sub(X, k, iters, seed):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        lab = np.empty(len(X), dtype=np.int64)
        cn = (C * C).sum(1)
        for s in range(0, len(X), 25000):
            lab[s:s + 25000] = (cn[None, :] - 2 * (X[s:s + 25000] @ C.T)).argmin(1)
        cnt = np.bincount(lab, minlength=k)
        newC = np.stack([np.bincount(lab, weights=X[:, j], minlength=k) for j in range(X.shape[1])], 1)
        nz = cnt > 0
        C[nz] = (newC[nz] / cnt[nz, None]).astype(np.float32)
        if (~nz).any():
            C[~nz] = X[rng.choice(len(X), int((~nz).sum()), replace=False)]
    return C

def pq_encode(R, books):
    M, k, ds = books.shape
    cn = (books * books).sum(2)                                    # (M, k)
    codes = np.empty((len(R), M), dtype=np.uint8)
    for s in range(0, len(R), 64):                                 # 64 x M x 256 f32 temporaries (~12 MB); runs in 16 threads
        Xs = R[s:s + 64].reshape(-1, M, ds)
        d = cn[None] - 2 * np.einsum("nmd,mkd->nmk", Xs, books, optimize=True)
        codes[s:s + 64] = d.argmin(2)
    return codes

def pq_decode(codes, books):
    M, k, ds = books.shape
    return books[np.arange(M)[None, :], codes].reshape(len(codes), M * ds)

if os.path.exists(pq_cache):
    z = np.load(pq_cache)
    books = {192: z["b192"], 96: z["b96"]}
    log("loaded PQ codebooks", pq_cache)
else:
    books = {}
    for M in (192, 96):
        ds = D // M
        tt = time.time()
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(lambda m: kmeans_sub(np.ascontiguousarray(Rs[:, m * ds:(m + 1) * ds]), 256, args.pq_iters, m),
                              range(M)))
        books[M] = np.stack(res)
        log(f"PQ M={M} trained in {time.time() - tt:.0f}s")
    np.savez(pq_cache, b192=books[192], b96=books[96])
    del Rs
_Rt = gather_rows(sample_idx[:20000:2])
_Rt -= F[P[sample_idx[:20000:2]]]
pq_stats = {}
for M in (192, 96):
    err = ((_Rt - pq_decode(pq_encode(_Rt, books[M]), books[M])) ** 2).sum(1).mean()
    pq_stats[M] = dict(residual_energy=float((_Rt ** 2).sum(1).mean()), distortion=float(err))
    log(f"PQ M={M}: mean residual energy {pq_stats[M]['residual_energy']:.4f}, mean distortion {err:.4f}")
del _Rt

# ------------------------------------------------------------------ encoders (residual -> decoded residual)
def quant_uniform(R, L, clip="max"):
    """Symmetric uniform quantizer, levels -L..L, per-vector scale."""
    amax = np.maximum(np.abs(R).max(1), 1e-12)
    if clip == "max":
        s = amax / L
        return np.clip(np.rint(R / s[:, None]), -L, L) * s[:, None]
    best, bestE = None, None
    for f in np.linspace(0.3, 1.0, 15):                            # per-vector MSE-optimal clip of the scale
        s = amax * f / L
        Rh = np.clip(np.rint(R / s[:, None]), -L, L) * s[:, None]
        E = ((R - Rh) ** 2).sum(1)
        if best is None:
            best, bestE = Rh, E
        else:
            upd = E < bestE
            best[upd] = Rh[upd]
            bestE = np.minimum(bestE, E)
    return best

rng = np.random.default_rng(7)
ROT = np.linalg.qr(rng.standard_normal((D, D)))[0].astype(np.float32)

def cellnorm(L, clip):
    def f(R, c):
        Z = (R - cmean[c]) / cstd[c]
        return cmean[c] + quant_uniform(Z, L, clip) * cstd[c]
    return f

POOL = ThreadPoolExecutor(16)
def par(dec):
    """Run a row-wise decoder over row blocks in threads (numpy ufuncs release the GIL)."""
    def f(R, c):
        if len(R) < 1024:
            return dec(R, c)
        parts = list(range(0, len(R), (len(R) + 15) // 16))
        outs = list(POOL.map(lambda s: dec(R[s:s + (len(R) + 15) // 16], c[s:s + (len(R) + 15) // 16]), parts))
        return np.concatenate(outs)
    return f

EH = 16
SCHEMES = {   # name -> (decoder(R, cells) -> R_hat, bytes per entry)
    "a int8": (lambda R, c: quant_uniform(R, 127), 1536 + EH + 4),
    "b 4-bit max-scale": (lambda R, c: quant_uniform(R, 7), 768 + EH + 4),
    "b' 4-bit clipped": (lambda R, c: quant_uniform(R, 7, "mse"), 768 + EH + 4),
    "b'' 4-bit rotated+clipped": (lambda R, c: quant_uniform(R @ ROT, 7, "mse") @ ROT.T, 768 + EH + 4),
    "c 4-bit cell-norm": (cellnorm(7, "max"), 768 + EH + 4),
    "c' 4-bit cell-norm+clipped": (cellnorm(7, "mse"), 768 + EH + 4),
    "5-bit clipped": (lambda R, c: quant_uniform(R, 15, "mse"), 960 + EH + 4),
    "d 6-bit max-scale": (lambda R, c: quant_uniform(R, 31), 1152 + EH + 4),
    "d' 6-bit clipped": (lambda R, c: quant_uniform(R, 31, "mse"), 1152 + EH + 4),
    "e PQ 192x8d": (lambda R, c: pq_decode(pq_encode(R, books[192]), books[192]), 192 + EH),
    "e PQ 96x16d": (lambda R, c: pq_decode(pq_encode(R, books[96]), books[96]), 96 + EH),
}
SCHEMES = {k: (par(d), b) for k, (d, b) in SCHEMES.items()}

# ------------------------------------------------------------------ part 1: candidate level
uniq = np.unique(bf_i)
Xc = gather_rows(uniq)
where = np.searchsorted(uniq, bf_i)                                # (NQ, K) -> row of Xc
log(f"part 1: {len(uniq)} unique candidates")
qi = np.repeat(np.arange(NQ), K)
ui = where.ravel()
order = np.argsort(ui, kind="stable")
qi_s, ui_s = qi[order], ui[order]
bounds = np.searchsorted(ui_s, np.arange(0, len(uniq) + 4096, 4096))

def pair_scores(cell_of, decoders):
    """Scores (NQ, K) for each decoder; candidates encoded against cell_of[row]."""
    out = {k: np.empty(NQ * K, dtype=np.float32) for k in decoders}
    for b, s in enumerate(range(0, len(uniq), 4096)):
        X = Xc[s:s + 4096]
        c = cell_of[s:s + 4096]
        Mb = F[c]
        R = X - Mb
        lo, hi = bounds[b], bounds[b + 1]
        rows, qs = ui_s[lo:hi] - s, qi_s[lo:hi]
        for k, dec in decoders.items():
            Xh = Mb + dec(R, c)
            out[k][order[lo:hi]] = np.einsum("ij,ij->i", Xh[rows], Q[qs])
    return {k: v.reshape(NQ, K) for k, v in out.items()}

exact = np.stack([Xc[where[i]] @ Q[i] for i in range(NQ)])
assert np.allclose(exact, bf_s, atol=1e-4)
gt10 = np.argsort(-exact, axis=1)[:, :10]                          # positions in the top-K list
sexact = np.sort(exact, axis=1)[:, ::-1]
ctx = dict(top1=float(sexact[:, 0].mean()), top10=float(sexact[:, 9].mean()), top30=float(sexact[:, 29].mean()),
           top50=float(sexact[:, 49].mean()), topK=float(sexact[:, -1].mean()),
           gap10_11_median=float(np.median(sexact[:, 9] - sexact[:, 10])),
           gap10_11_p10=float(np.quantile(sexact[:, 9] - sexact[:, 10], 0.1)),
           gap10_K_p1=float(np.quantile(sexact[:, 9] - sexact[:, -1], 0.01)))
log("context:", ctx)

probes = np.argsort(-(Q @ Fn.T), axis=1)[:, : args.nprobe]
PB = np.zeros((NQ, NC), dtype=bool)
np.put_along_axis(PB, probes, True, 1)
reach = PB[np.arange(NQ)[:, None, None], A[bf_i, : args.m].astype(np.int64)].any(2)   # (NQ, K)
routing_ub = float(np.take_along_axis(reach, gt10, 1).mean())
log(f"flat nprobe={args.nprobe} m={args.m}: routing UB {routing_ub*100:.2f}% ({NQ} queries)")

def recall_pos(top):                                               # top: positions in top-K list
    return float(np.mean([len(set(top[i]) & set(gt10[i])) / 10 for i in range(len(top))]))

def recall(score, cut=None, mask=None):
    s = score.copy()
    if cut is not None:
        s[:, cut:] = -np.inf
    if mask is not None:
        s[~mask] = -np.inf
    return recall_pos(np.argsort(-s, axis=1)[:, :10])

def two_tier(sc, s8, T, mask=None):
    s = sc.copy()
    if mask is not None:
        s[~mask] = -np.inf
    sel = np.argsort(-s, axis=1)[:, :T]
    rr = np.take_along_axis(s8, sel, 1)
    if mask is not None:
        rr[~np.take_along_axis(mask, sel, 1)] = -np.inf
    return recall_pos(np.take_along_axis(sel, np.argsort(-rr, axis=1)[:, :10], 1))

scores = pair_scores(P[uniq], {k: v[0] for k, v in SCHEMES.items()})
log("part 1: encoded and scored all schemes")
s8 = scores["a int8"]
res = {}
for name, sc in scores.items():
    err = np.abs(sc - exact)
    r = dict(bytes=SCHEMES[name][1], recall_top50=recall(sc, cut=50), recall_topK=recall(sc),
             mae=float(err.mean()), p99=float(np.quantile(err, 0.99)), max=float(err.max()),
             e2e_topK=recall(sc, mask=reach))
    if name != "a int8":
        r["two_tier_topK"] = {T: two_tier(sc, s8, T) for T in (10, 20, 30, 50)}
    res[name] = r
# far-cell encoding (slab members whose 3rd assigned cell is the slab), for the clipped 4-bit scheme
sc3 = pair_scores(A[uniq, 2].astype(np.int64), {"x": SCHEMES["b' 4-bit clipped"][0]})["x"]
err = np.abs(sc3 - exact)
res["b' 4-bit clipped, vs 3rd cell"] = dict(bytes=788, recall_top50=recall(sc3, cut=50), recall_topK=recall(sc3),
                                            mae=float(err.mean()), p99=float(np.quantile(err, 0.99)),
                                            max=float(err.max()), e2e_topK=recall(sc3, mask=reach),
                                            two_tier_topK={T: two_tier(sc3, s8, T) for T in (10, 20, 30, 50)})
del Xc, scores

print(f"\n=== part 1: {NQ} queries, exact top-{K} candidates, encoded against the primary cell ===")
print(f"exact top-10 vs groundtruth.u32 overlap {gt_ok*100:.1f}%; flat routing m={args.m} nprobe={args.nprobe} "
      f"UB {routing_ub*100:.2f}%")
print(f"{'scheme':32s} {'B/entry':>7s} {'R@10/50':>8s} {'R@10/' + str(K):>8s} {'mae':>9s} {'p99':>9s} {'max':>9s} "
      f"{'e2e*':>7s}   two-tier T=10/20/30/50 (int8 rerank)")
for name, r in res.items():
    tt = "  ".join(f"{v*100:5.1f}" for v in r.get("two_tier_topK", {}).values())
    print(f"{name:32s} {r['bytes']:7d} {r['recall_top50']*100:7.2f}% {r['recall_topK']*100:7.2f}% "
          f"{r['mae']:9.2e} {r['p99']:9.2e} {r['max']:9.2e} {r['e2e_topK']*100:6.2f}%   {tt}")
print("(e2e* = ranking restricted to reachable members of the exact top-K)")

# ------------------------------------------------------------------ part 2: full end-to-end pass
E2E = ["a int8", "b' 4-bit clipped", "c' 4-bit cell-norm+clipped", "5-bit clipped", "d 6-bit max-scale", "e PQ 192x8d"]
TK = {"a int8": 50, "e PQ 192x8d": 1000}                         # per-query top list kept per scheme (default 100)
e2e_out = {}
if args.e2e:
    e2e_cache = os.path.join(args.idx, f"quant_e2e_q{NQ}_np{args.nprobe}_m{args.m}.npz")
    if os.path.exists(e2e_cache):
        z = np.load(e2e_cache)
        tops = {k: (z[f"s_{i}"], z[f"i_{i}"], z[f"r_{i}"]) for i, k in enumerate(E2E)}
        ex_s, ex_i, nscan = z["ex_s"], z["ex_i"], z["nscan"]
        log("loaded cached e2e pass", e2e_cache)
    else:
        tops = {k: (np.full((NQ, TK.get(k, 100)), -np.inf, np.float32), np.full((NQ, TK.get(k, 100)), -1, np.int64),
                    np.full((NQ, TK.get(k, 100)), -np.inf, np.float32)) for k in E2E}
        ex_s = np.full((NQ, 10), -np.inf, np.float32); ex_i = np.full((NQ, 10), -1, np.int64)
        nscan = np.zeros(NQ, np.int64)
        tdec = {k: 0.0 for k in E2E}
        def merge(cur_s, cur_i, cur_r, S, ids, Rr):
            k = cur_s.shape[1]
            allS = np.concatenate([cur_s, S], 1)
            allI = np.concatenate([cur_i, np.broadcast_to(ids, S.shape)], 1)
            allR = np.concatenate([cur_r, Rr], 1) if Rr is not None else None
            p = np.argpartition(-allS, k - 1, axis=1)[:, :k]
            return (np.take_along_axis(allS, p, 1), np.take_along_axis(allI, p, 1),
                    np.take_along_axis(allR, p, 1) if allR is not None else None)
        for s in range(0, N, args.chunk):
            X = read_rows(s, min(N, s + args.chunk))
            n = len(X)
            ids = np.arange(s, s + n)
            c = P[s:s + n]
            Mb = F[c]
            R = X - Mb
            mask = PB[:, A[s:s + n, : args.m].astype(np.int64)].any(2)       # (NQ, n) reachable
            nscan += mask.sum(1)
            Se = Q @ X.T
            Se[~mask] = -np.inf
            ex_s, ex_i, _ = merge(ex_s, ex_i, None, Se, ids, None)
            S8 = None
            for k in E2E:
                tt = time.time()
                Xh = Mb + SCHEMES[k][0](R, c)
                tdec[k] += time.time() - tt
                Sk = Q @ Xh.T
                Sk[~mask] = -np.inf
                if k == "a int8":
                    S8 = Sk
                cs, ci, cr = tops[k]
                tops[k] = merge(cs, ci, cr, Sk, ids, S8)
            if (s // args.chunk) % 10 == 0:
                log(f"pass 2: {s}/{N}  decode s so far: " + ", ".join(f"{k}={v:.0f}" for k, v in tdec.items()))
        np.savez(e2e_cache, ex_s=ex_s, ex_i=ex_i, nscan=nscan,
                 **{f"{p}_{i}": tops[k][j] for i, k in enumerate(E2E) for j, p in enumerate("sir")})
        log("pass 2 done")
    gtset = [set(GT[i]) for i in range(NQ)]
    def rec_ids(top_ids):
        return float(np.mean([len(gtset[i] & set(top_ids[i])) / 10 for i in range(NQ)]))
    e2e_out["unique_reachable_vectors_mean"] = float(nscan.mean())
    e2e_out["exact"] = rec_ids(ex_i)
    print(f"\n=== part 2: end to end, flat routing m={args.m} nprobe={args.nprobe}, all reachable vectors "
          f"({nscan.mean():.0f} unique per query on average), primary-cell encoding ===")
    print(f"exact f32 scoring (routing upper bound): {e2e_out['exact']*100:.2f}%")
    for k in E2E:
        cs, ci, cr = tops[k]
        o = np.argsort(-cs, axis=1)
        cs, ci, cr = (np.take_along_axis(a, o, 1) for a in (cs, ci, cr))
        r = dict(bytes=SCHEMES[k][1], recall=rec_ids(ci[:, :10]))
        if k != "a int8":
            r["two_tier"] = {}
            for T in (10, 20, 30, 50, 100, 200, 500, 1000):
                if T > cs.shape[1]:
                    continue
                o2 = np.argsort(-cr[:, :T], axis=1)[:, :10]
                r["two_tier"][T] = rec_ids(np.take_along_axis(ci[:, :T], o2, 1))
        e2e_out[k] = r
        tt = "  ".join(f"T={T}: {v*100:5.2f}%" for T, v in r.get("two_tier", {}).items())
        print(f"  {k:28s} {r['bytes']:5d} B  recall@10 {r['recall']*100:6.2f}%   int8 rerank of top-T: {tt}")

out = dict(nq=NQ, k=K, nprobe=args.nprobe, m=args.m, routing_ub_topK=routing_ub, gt_overlap=gt_ok, context=ctx,
           pq=pq_stats, part1=res, part2=e2e_out)
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    return o
json.dump(clean(out), open(os.path.join(args.idx, "quant_experiment.json"), "w"), indent=1)
log("wrote", os.path.join(args.idx, "quant_experiment.json"))
