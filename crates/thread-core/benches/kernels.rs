//! Criterion benchmarks of the query kernels, the CPU half of a query (`cargo bench -p thread-core`).
//!
//! By default every input is synthetic at g4's shape: 1536 dims, a 4096-cell routing table, slabs of 714 entries
//! (975,000 vectors x 3 copies / 4096 cells), PQ-192 scan codes, rerank T = 150. Environment variables change it:
//!
//! - `THREAD_BENCH_DIM`, `THREAD_BENCH_CELLS`, `THREAD_BENCH_SLAB_N`, `THREAD_BENCH_PQ_M`: the synthetic shape.
//! - `THREAD_BENCH_SLAB=<path>` and `THREAD_BENCH_PQ=<path>`: a real v3 slab object (`idx/{gen}/slabs/...`) and the
//!   generation's `pq.bin`, scanned as they are (e.g. copied from the bucket with `cf r2 object get`). The query is
//!   the slab's centroid plus noise, so the scan does the heap work a real probe does.
//!
//! These are native timings. The dot products (`math.rs`: `dot`, `dot_f32_i8`, `dot_f32_i4`) have a SIMD path only
//! on wasm32 + simd128; native builds (this bench, the CLI, the Python extension) run their scalar fallbacks, so a
//! native figure is not a wasm figure: measured on an M2 Pro, the 4-bit scan costs 0.44 us/entry here and
//! 0.19 us/entry as wasm in Node. For wasm costs use the trace envelope's `kernels` from Node (`bench/scenarios.mjs`);
//! use this bench to compare kernel changes against each other and for flamegraphs:
//! `cargo bench -p thread-core --profile profiling --bench kernels -- --profile-time 10 scan/pq` under
//! `samply record` or `perf record` (docs/observability.md).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use thread_core::math::{normalize, Rng};
use thread_core::pq::{Codebook, PqAccum, QueryTable};
use thread_core::quant::{encode_residual_sq4, encode_residual_sq8};
use thread_core::route::Router;
use thread_core::search::{rerank_records, scan_slab, scan_slab_pq};
use thread_core::slab::{SlabBuilder, SlabBuilderV3, SlabView};
use thread_core::tokenize::{Tokenizer, TokenizerConfig};
use thread_core::topk::{merge_dedup, Hit, TopK};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Shape {
    dim: usize,
    cells: usize,
    slab_n: usize,
    pq_m: usize,
    /// Rerank T (the manifest default) and top k.
    t: usize,
    k: usize,
    nprobe: usize,
}

fn shape() -> Shape {
    Shape {
        dim: env_usize("THREAD_BENCH_DIM", 1536),
        cells: env_usize("THREAD_BENCH_CELLS", 4096),
        slab_n: env_usize("THREAD_BENCH_SLAB_N", 714),
        pq_m: env_usize("THREAD_BENCH_PQ_M", 192),
        t: 150,
        k: 10,
        nprobe: 34,
    }
}

fn rand_unit(rng: &mut Rng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim)
        .map(|_| rng.below(20001) as f32 / 10000.0 - 1.0)
        .collect();
    normalize(&mut v);
    v
}

/// `n` unit vectors around unit centroid `c` (spread 0.3, as the CLI's `bench-scan`).
fn around(rng: &mut Rng, c: &[f32], n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| {
            let mut x: Vec<f32> = c
                .iter()
                .map(|v| v + (rng.below(20001) as f32 / 10000.0 - 1.0) * 0.3)
                .collect();
            normalize(&mut x);
            x
        })
        .collect()
}

/// A trained codebook for the residuals of `xs` around `c` (init + 2 Lloyd iterations).
fn train_codebook(xs: &[Vec<f32>], c: &[f32], m: usize) -> Codebook {
    let dim = c.len();
    let res: Vec<f32> = xs
        .iter()
        .flat_map(|x| x.iter().zip(c).map(|(a, b)| a - b))
        .collect();
    let mut cb = Codebook::init_from_samples(&res, dim, m, 7).expect("codebook init");
    for _ in 0..2 {
        let mut acc = PqAccum::new(dim, m).expect("accum");
        for r in res.chunks_exact(dim) {
            acc.add(r, &cb);
        }
        cb = acc.means(&cb).expect("means");
    }
    cb
}

/// One v3 slab (PQ scan tier + int8 rerank tier) of `xs` around `c`, and its codebook.
fn v3_slab(xs: &[Vec<f32>], c: &[f32], cb: &Codebook) -> Vec<u8> {
    let dim = c.len();
    let mut b = SlabBuilderV3::new(dim, 0, c, cb.m);
    let (mut code, mut code8) = (vec![0u8; cb.m], vec![0i8; dim]);
    for (i, x) in xs.iter().enumerate() {
        let r: Vec<f32> = x.iter().zip(c).map(|(a, b)| a - b).collect();
        thread_core::pq::encode(&r, cb, &mut code);
        let s8 = encode_residual_sq8(x, c, &mut code8);
        b.push(i as u32, &code, &code8, s8, &[], 0).expect("push");
    }
    b.finish()
}

/// A query near the slab's centroid (the cells a query probes are the ones it is close to).
fn query_near(rng: &mut Rng, c: &[f32]) -> Vec<f32> {
    around(rng, c, 1).pop().unwrap()
}

fn bench_route(cr: &mut Criterion) {
    let s = shape();
    let mut rng = Rng::new(3);
    let fine: Vec<f32> = (0..s.cells)
        .flat_map(|_| rand_unit(&mut rng, s.dim))
        .collect();
    let coarse = rand_unit(&mut rng, s.dim);
    let router = Router::new(s.dim, 1, s.cells, &coarse, fine);
    let q = rand_unit(&mut rng, s.dim);
    let mut g = cr.benchmark_group("route");
    g.throughput(Throughput::Elements((s.cells * s.dim) as u64));
    g.bench_function(
        BenchmarkId::new("flat", format!("{}x{}", s.cells, s.dim)),
        |b| b.iter(|| black_box(router.route_flat(black_box(&q), s.nprobe))),
    );
    g.bench_function(
        BenchmarkId::new("cell_dots", format!("{}x{}", s.cells, s.dim)),
        |b| b.iter(|| black_box(router.cell_dots(black_box(&q)))),
    );
    g.finish();
}

fn bench_scan(cr: &mut Criterion) {
    let s = shape();
    let mut rng = Rng::new(1);
    let mut g = cr.benchmark_group("scan");

    // Real fixtures, when given: the bytes a production probe scans.
    if let (Ok(slab_path), Ok(pq_path)) = (
        std::env::var("THREAD_BENCH_SLAB"),
        std::env::var("THREAD_BENCH_PQ"),
    ) {
        let bytes = std::fs::read(&slab_path).expect("read THREAD_BENCH_SLAB");
        let cb = Codebook::from_bytes(&std::fs::read(&pq_path).expect("read THREAD_BENCH_PQ"))
            .expect("parse pq.bin");
        let full = SlabView::parse(&bytes).expect("parse slab");
        let scan = SlabView::parse(&bytes[..full.scan_len()]).expect("parse scan prefix");
        let q = query_near(&mut rng, &scan.centroid());
        let table = QueryTable::new(&q, &cb).expect("table");
        g.throughput(Throughput::Elements(scan.count as u64));
        g.bench_function(
            BenchmarkId::new("pq_fixture", format!("{} entries", scan.count)),
            |b| {
                b.iter(|| {
                    let mut top = TopK::new(s.t);
                    black_box(scan_slab_pq(&scan, &table, &q, &mut top, None, None).unwrap());
                    black_box(top.into_sorted())
                })
            },
        );
    }

    let c = rand_unit(&mut rng, s.dim);
    let xs = around(&mut rng, &c, s.slab_n);
    let q = query_near(&mut rng, &c);
    if s.pq_m > 0 && s.dim.is_multiple_of(s.pq_m) {
        let cb = train_codebook(&xs, &c, s.pq_m);
        let bytes = v3_slab(&xs, &c, &cb);
        let full = SlabView::parse(&bytes).unwrap();
        let scan = SlabView::parse(&bytes[..full.scan_len()]).unwrap();
        let table = QueryTable::new(&q, &cb).unwrap();
        g.throughput(Throughput::Elements(s.slab_n as u64));
        g.bench_function(
            BenchmarkId::new("pq", format!("{}x{} m{}", s.slab_n, s.dim, s.pq_m)),
            |b| {
                b.iter(|| {
                    let mut top = TopK::new(s.t);
                    black_box(scan_slab_pq(&scan, &table, &q, &mut top, None, None).unwrap());
                    black_box(top.into_sorted())
                })
            },
        );
        g.bench_function(
            BenchmarkId::new("pq_table", format!("{} m{}", s.dim, s.pq_m)),
            |b| b.iter(|| black_box(QueryTable::new(black_box(&q), &cb).unwrap())),
        );
    }

    // v2: 4-bit scan tier (generations created from nothing, `sq4+sq8`).
    let mut b2 = SlabBuilder::new(s.dim, 0, &c);
    let (mut code8, mut code4, mut scratch) =
        (vec![0i8; s.dim], vec![0u8; s.dim / 2], vec![0f32; s.dim]);
    for (i, x) in xs.iter().enumerate() {
        let s8 = encode_residual_sq8(x, &c, &mut code8);
        let s4 = encode_residual_sq4(x, &c, &mut code4, &mut scratch);
        b2.push(i as u32, &code4, s4, &code8, s8, &[], 0);
    }
    let bytes2 = b2.finish();
    let full2 = SlabView::parse(&bytes2).unwrap();
    let scan2 = SlabView::parse(&bytes2[..full2.scan_len()]).unwrap();
    g.throughput(Throughput::Elements(s.slab_n as u64));
    g.bench_function(
        BenchmarkId::new("sq4", format!("{}x{}", s.slab_n, s.dim)),
        |b| {
            b.iter(|| {
                let mut top = TopK::new(s.t);
                black_box(scan_slab(&scan2, &q, &mut top, None, None));
                black_box(top.into_sorted())
            })
        },
    );
    g.finish();

    // Phase 2: rerank T records of one slab (a query reranks 150 across ~10 slabs; per-record cost is what counts).
    let mut g = cr.benchmark_group("rerank");
    let mut recs = Vec::new();
    for i in 0..s.t.min(s.slab_n) {
        recs.extend_from_slice(full2.rerank_record(i));
    }
    let n = s.t.min(s.slab_n);
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function(BenchmarkId::new("int8", format!("{n}x{}", s.dim)), |b| {
        b.iter(|| black_box(rerank_records(&q, &c, &recs, n).unwrap()))
    });
    g.finish();
}

fn bench_merge(cr: &mut Criterion) {
    let s = shape();
    let mut rng = Rng::new(5);
    // nprobe slabs x T hits each, ids overlapping (multi-assignment puts each vector in m = 3 slabs).
    let hits: Vec<Hit> = (0..s.nprobe * s.t)
        .map(|i| Hit {
            id: rng.below((s.nprobe * s.t / 3) as u64) as u32,
            score: rng.below(1_000_000) as f32 / 1e6,
            idx: i as u32,
        })
        .collect();
    let mut g = cr.benchmark_group("merge");
    g.throughput(Throughput::Elements(hits.len() as u64));
    g.bench_function(
        BenchmarkId::new("dedup", format!("{} hits -> {}", hits.len(), s.t)),
        |b| b.iter(|| black_box(merge_dedup(hits.clone(), s.t))),
    );
    g.finish();
    let _ = s.k;
}

fn bench_tokenize(cr: &mut Criterion) {
    let text = "Deploy failed: the worker thread-wasm returned HTTP 503 while compacting idx/g4/slabs/0042.3 on 2026-10-05; \
                retrying with backoff. Rerank T=150, nprobe 34, recall@10 95.1%. ";
    let doc = text.repeat(20);
    let mut tok = Tokenizer::new(&TokenizerConfig::bm25s_english()).expect("tokenizer");
    let mut g = cr.benchmark_group("tokenize");
    g.throughput(Throughput::Bytes(doc.len() as u64));
    g.bench_function(
        BenchmarkId::new("english", format!("{} bytes", doc.len())),
        |b| b.iter(|| black_box(tok.tokenize(black_box(&doc)))),
    );
    g.finish();
}

criterion_group!(
    benches,
    bench_route,
    bench_scan,
    bench_merge,
    bench_tokenize
);
criterion_main!(benches);
