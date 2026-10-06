// Dense-side file written before rustfmt was adopted; kept as is (`cargo fmt` skips it).
#![cfg_attr(rustfmt, rustfmt::skip)]
//! Native CLI for thread.
//!
//!   thread seg build|search|inspect|merge ...  lexical/docs segments on local files (see lex_cli.rs)
//!   thread tokenize [--tokenizer json] "text"   the index's tokenizer
//!   thread chunk [--words 200] [--overlap 0] "text"   flat word-window chunks (see chunk.rs)
//!   thread bench-scan [n_vectors] [dim] [pq_m]   time v1 (int8), v2 (4-bit + int8 rerank) and v3 (PQ + int8 rerank) scans of a synthetic slab
//!   thread bench-route [k1] [k2] [dim] [nprobe]  time flat vs two-level routing
//!   thread bench-neighbors [k] [dim] [count] [n]  time one cell_neighbors range (cells [0, count))
//!   thread inspect <slab-file>            print slab header info

mod chunk;
mod lex_cli;

use std::time::Instant;
use thread_core::math::{dot, normalize, Rng};
use thread_core::pq::{self, Codebook, PqAccum, QueryTable};
use thread_core::quant::{encode_residual_sq4, encode_residual_sq8};
use thread_core::route::Router;
use thread_core::search::{rerank_records, scan_slab, scan_slab_pq};
use thread_core::slab::{SlabBuilder, SlabBuilderV1, SlabBuilderV3, SlabView};
use thread_core::topk::{Hit, TopK};

fn rand_unit(rng: &mut Rng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.below(20001) as f32 / 10000.0 - 1.0).collect();
    normalize(&mut v);
    v
}

/// Mean milliseconds per call of `f` over enough iterations to take ~0.3 s.
fn time_ms(mut f: impl FnMut() -> usize) -> f64 {
    let mut sink = f(); // warm
    let t = Instant::now();
    let mut iters = 0usize;
    while iters < 5 || (t.elapsed().as_secs_f64() < 0.3 && iters < 10_000) {
        sink = sink.wrapping_add(f());
        iters += 1;
    }
    std::hint::black_box(sink);
    t.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn bench_scan(n: usize, dim: usize, pq_m: usize) {
    const K: usize = 10;
    const T: usize = 30; // 4-bit candidates reranked with int8
    let mut rng = Rng::new(1);
    let c = rand_unit(&mut rng, dim);
    let mut b1 = SlabBuilderV1::new(dim, 0, &c);
    let mut b2 = SlabBuilder::new(dim, 0, &c);
    let (mut code8, mut code4, mut scratch) = (vec![0i8; dim], vec![0u8; dim / 2], vec![0f32; dim]);
    let mut xs = Vec::with_capacity(n);
    for i in 0..n {
        let mut x: Vec<f32> = c.iter().map(|v| v + (rng.below(20001) as f32 / 10000.0 - 1.0) * 0.3).collect();
        normalize(&mut x);
        let s8 = encode_residual_sq8(&x, &c, &mut code8);
        let s4 = encode_residual_sq4(&x, &c, &mut code4, &mut scratch);
        b1.push(i as u32, &code8, s8, &[], 0);
        b2.push(i as u32, &code4, s4, &code8, s8, &[], 0);
        xs.push(x);
    }
    let bytes1 = b1.finish();
    let bytes2 = b2.finish();
    let v1 = SlabView::parse(&bytes1).unwrap();
    let full2 = SlabView::parse(&bytes2).unwrap();
    let scan_len = full2.scan_len();
    let v2 = SlabView::parse(&bytes2[..scan_len]).unwrap();
    let h2 = full2.header;
    let q = xs[n / 2].clone();

    let mut exact: Vec<(usize, f32)> = xs.iter().enumerate().map(|(i, x)| (i, dot(&q, x))).collect();
    exact.sort_by(|a, b| b.1.total_cmp(&a.1));
    let overlap = |ids: &[u32]| ids.iter().take(K).filter(|id| exact[..K].iter().any(|(i, _)| *i as u32 == **id)).count();

    let scan = |v: &SlabView<'_>, k: usize| {
        let mut top = TopK::new(k);
        scan_slab(v, &q, &mut top, None, None);
        top.into_sorted()
    };
    // 4-bit scan of the prefix, then one rerank record per candidate (as the host would range-read them)
    let rerank = |cands: &[Hit]| -> Vec<u32> {
        let mut recs = Vec::with_capacity(cands.len() * h2.rerank_record_len);
        for h in cands {
            recs.extend_from_slice(full2.rerank_record(h.idx as usize));
        }
        let centroid = v2.centroid();
        let scores = rerank_records(&q, &centroid, &recs, cands.len()).unwrap();
        let mut rr: Vec<(u32, f32)> = cands.iter().zip(scores).map(|(h, s)| (h.id, s)).collect();
        rr.sort_by(|a, b| b.1.total_cmp(&a.1));
        rr.into_iter().take(K).map(|p| p.0).collect()
    };

    let t1 = time_ms(|| scan(&v1, K).len());
    let t2 = time_ms(|| scan(&v2, T).len());
    let t2r = time_ms(|| rerank(&scan(&v2, T)).len());
    let ids1: Vec<u32> = scan(&v1, K).iter().map(|h| h.id).collect();
    let ids4: Vec<u32> = scan(&v2, K).iter().map(|h| h.id).collect();
    let ids2r = rerank(&scan(&v2, T));
    let kb = |b: usize| b as f64 / 1024.0;
    println!("slab: {n} vectors x {dim} dims");
    println!(
        "  v1 int8 : read {:>8.1} KB, scan {t1:.3} ms ({:.0} vec/ms), top{K} overlap vs exact f32: {}/{K}",
        kb(v1.scan_len()),
        n as f64 / t1,
        overlap(&ids1)
    );
    println!(
        "  v2 4-bit: read {:>8.1} KB, scan {t2:.3} ms ({:.0} vec/ms), top{K} overlap (4-bit only): {}/{K}",
        kb(scan_len),
        n as f64 / t2,
        overlap(&ids4)
    );
    println!(
        "  v2 4-bit + int8 rerank of top{T}: +{:.1} KB in {T} records, {t2r:.3} ms, top{K} overlap: {}/{K}   (slab total v1 {:.1} KB, v2 {:.1} KB)",
        kb(T * h2.rerank_record_len),
        overlap(&ids2r),
        kb(bytes1.len()),
        kb(bytes2.len())
    );

    // v3: PQ scan tier. Codebook: init from the slab's own residuals + 4 Lloyd
    // iterations (synthetic residuals are isotropic noise, the worst case for PQ).
    if pq_m == 0 || dim % pq_m != 0 {
        println!("  v3 skipped: pq_m {pq_m} does not divide dim {dim}");
        return;
    }
    const T3: [usize; 3] = [50, 100, 200];
    let res: Vec<f32> = xs.iter().flat_map(|x| x.iter().zip(&c).map(|(a, b)| a - b)).collect();
    let t_train = Instant::now();
    let mut cb = Codebook::init_from_samples(&res, dim, pq_m, 7).unwrap();
    for _ in 0..4 {
        let mut acc = PqAccum::new(dim, pq_m).unwrap();
        for r in res.chunks_exact(dim) {
            acc.add(r, &cb);
        }
        cb = acc.means(&cb).unwrap();
    }
    let train_s = t_train.elapsed().as_secs_f64();
    let mut code = vec![0u8; pq_m];
    let t_enc = time_ms(|| pq::encode(&res[..dim], &cb, &mut code) as usize);
    let mut b3 = SlabBuilderV3::new(dim, 0, &c, pq_m);
    let mut sq_err = 0.0f64;
    for (i, x) in xs.iter().enumerate() {
        sq_err += pq::encode(&res[i * dim..(i + 1) * dim], &cb, &mut code) as f64;
        let s8 = encode_residual_sq8(x, &c, &mut code8);
        b3.push(i as u32, &code, &code8, s8, &[], 0).unwrap();
    }
    let res_energy: f64 = res.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let bytes3 = b3.finish();
    let full3 = SlabView::parse(&bytes3).unwrap();
    let scan_len3 = full3.scan_len();
    let v3 = SlabView::parse(&bytes3[..scan_len3]).unwrap();
    let t_table = time_ms(|| QueryTable::new(&q, &cb).unwrap().m);
    let table = QueryTable::new(&q, &cb).unwrap();
    let scan3 = |k: usize| {
        let mut top = TopK::new(k);
        scan_slab_pq(&v3, &table, &q, &mut top, None, None).unwrap();
        top.into_sorted()
    };
    let t3 = time_ms(|| scan3(K).len());
    let t3_200 = time_ms(|| scan3(200).len());
    let rerank3 = |cands: &[Hit]| -> Vec<u32> {
        let mut recs = Vec::with_capacity(cands.len() * full3.header.rerank_record_len);
        for h in cands {
            recs.extend_from_slice(full3.rerank_record(h.idx as usize));
        }
        let scores = rerank_records(&q, &c, &recs, cands.len()).unwrap();
        let mut rr: Vec<(u32, f32)> = cands.iter().zip(scores).map(|(h, s)| (h.id, s)).collect();
        rr.sort_by(|a, b| b.1.total_cmp(&a.1));
        rr.into_iter().take(K).map(|p| p.0).collect()
    };
    let ids3: Vec<u32> = scan3(K).iter().map(|h| h.id).collect();
    println!(
        "  v3 PQ m={pq_m}: read {:>8.1} KB ({:.2}x less than v2, {:.0} B/entry), table {t_table:.3} ms, scan {t3:.3} ms ({:.0} vec/ms; top-200 heap {t3_200:.3} ms), top{K} overlap (PQ only): {}/{K}",
        kb(scan_len3),
        scan_len as f64 / scan_len3 as f64,
        scan_len3 as f64 / n as f64,
        n as f64 / t3,
        overlap(&ids3)
    );
    for t in T3 {
        let cands = scan3(t.min(n));
        println!(
            "  v3 PQ + int8 rerank of top{t}: +{:.1} KB in {t} records, top{K} overlap: {}/{K}",
            kb(t.min(n) * full3.header.rerank_record_len),
            overlap(&rerank3(&cands))
        );
    }
    println!(
        "  v3 codebook: {:.1} KB, train (init + 4 iters on {n} residuals) {train_s:.2} s, encode {:.1} us/residual, residual energy left {:.1}%   (slab total v3 {:.1} KB)",
        kb(cb.byte_len()),
        t_enc * 1000.0,
        100.0 * sq_err / res_energy,
        kb(bytes3.len())
    );
}

fn bench_route(k1: usize, k2: usize, dim: usize, nprobe: usize) {
    let mut rng = Rng::new(3);
    let mut rows = |n: usize| -> Vec<f32> { (0..n).flat_map(|_| rand_unit(&mut rng, dim)).collect() };
    let coarse = rows(k1);
    let fine = rows(k1 * k2);
    let queries = rows(64);
    let router = Router::new(dim, k1, k2, &coarse, fine);
    let qs: Vec<&[f32]> = queries.chunks_exact(dim).collect();
    let time = |label: &str, f: &dyn Fn(&[f32]) -> usize| {
        let mut sink = 0;
        for q in &qs {
            sink += f(q); // warm
        }
        let iters = 4;
        let t = Instant::now();
        for _ in 0..iters {
            for q in &qs {
                sink += f(q);
            }
        }
        let per = t.elapsed().as_secs_f64() * 1000.0 / (iters * qs.len()) as f64;
        println!("{label}: {per:.3} ms/query (sink {sink})");
        per
    };
    let flat = time(&format!("route_flat {}x{dim} nprobe {nprobe}", k1 * k2), &|q| router.route_flat(q, nprobe).len());
    let macs = (k1 * k2 * dim) as f64;
    println!("  {:.2} GMAC/s", macs / (flat * 1e6));
    for pc in [2, 4, 8] {
        time(&format!("route pc={pc} ({} cells) nprobe {nprobe}", k1 + pc * k2), &|q| router.route(q, pc, nprobe).len());
    }
}

fn bench_neighbors(k: usize, dim: usize, count: usize, n: usize) {
    let mut rng = Rng::new(9);
    let means: Vec<f32> = (0..k).flat_map(|_| rand_unit(&mut rng, dim)).collect();
    let _ = thread_core::kmeans::cell_neighbors(&means, dim, 0, 16.min(count), n); // warm
    let t = Instant::now();
    let nb = thread_core::kmeans::cell_neighbors(&means, dim, 0, count, n);
    let s = t.elapsed().as_secs_f64();
    let macs = (count * k * dim) as f64;
    println!(
        "cell_neighbors k={k} dim={dim} range of {count} cells n={n}: {:.3} s ({:.2} GMAC/s), {} ids; full table ~{:.1} s",
        s,
        macs / s / 1e9,
        nb.len(),
        s * k as f64 / count as f64
    );
}

fn inspect(path: &str) {
    let bytes = std::fs::read(path).expect("read slab");
    let v = SlabView::parse(&bytes).expect("parse slab");
    let h = v.header;
    println!(
        "cell {} dim {} count {} bytes {} version {} scan_len {} rerank_off {} payload {}+{}",
        v.cell, v.dim, v.count, bytes.len(), h.version, h.scan_len, h.rerank_off, h.payload_off, h.payload_len
    );
    println!("first ids: {:?}", (0..v.count.min(8)).map(|i| v.id(i)).collect::<Vec<_>>());
    let range = |f: &dyn Fn(usize) -> f32| {
        (0..v.count).map(f).fold((f32::MAX, f32::MIN), |(lo, hi), s| (lo.min(s), hi.max(s)))
    };
    if v.is_v2() {
        let (lo, hi) = range(&|i| v.scale4(i));
        println!("scale4: min {lo:.5} max {hi:.5}");
    }
    if v.is_v3() {
        println!("pq_m {} ({} B/entry scan tier)", v.pq_m(), v.pq_m() + 16);
    }
    if !v.has_rerank_tier() || v.has_rerank() {
        let (lo, hi) = range(&|i| v.scale8(i));
        println!("scale8: min {lo:.5} max {hi:.5}");
    }
}

fn usage() -> String {
    format!(
        "thread: a blob-native search index (dense + lexical). Commands:\n{}\n  thread bench-scan [n] [dim] [pq_m]            time v1/v2/v3 scans of a synthetic slab\n  thread bench-route [k1] [k2] [dim] [nprobe]   time flat vs two-level routing\n  thread bench-neighbors [k] [dim] [count] [n]  time one cell_neighbors range\n  thread inspect <slab-file>                    print a dense slab's header",
        lex_cli::USAGE
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some(cmd @ ("seg" | "tokenize" | "chunk")) => match lex_cli::run(cmd, &args[2..]) {
            Ok(lex_cli::Output::Json(v)) => println!("{}", serde_json::to_string(&v).unwrap()),
            Ok(lex_cli::Output::Help(h)) => print!("{h}"),
            Err(e) => {
                eprintln!("thread: error: {e}");
                std::process::exit(if e.contains("required") || e.contains("needs") || e.contains("unknown") { 1 } else { 2 });
            }
        },
        Some("bench-scan") => {
            let n = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(750);
            let dim = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1536);
            let pq_m = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(192);
            bench_scan(n, dim, pq_m);
        }
        Some("bench-route") => {
            let a = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
            bench_route(a(2, 64), a(3, 64), a(4, 1536), a(5, 32));
        }
        Some("bench-neighbors") => {
            let a = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
            bench_neighbors(a(2, 4096), a(3, 1536), a(4, 512), a(5, 64));
        }
        Some("inspect") => inspect(args.get(2).expect("slab path")),
        Some("--help" | "-h" | "help") => println!("{}", usage()),
        _ => {
            eprintln!("{}", usage());
            std::process::exit(1);
        }
    }
}
