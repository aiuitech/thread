//! Recall/latency of `route::CentroidTree` against exhaustive search on real
//! centroids. Usage:
//! `cargo run --release -p vib-core --example tree_bench -- <means.f32> <sample.f32> <dim> [groups,..] [iters]`
//! (`means.f32`: k x dim f32 LE raw cell means; `sample.f32`: n x dim unit vectors.)

use std::time::Instant;
use vib_core::error::f32s_from_le;
use vib_core::math::{argmax_dot_scaled, inv_norms, top_n_dot_scaled};
use vib_core::route::CentroidTree;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let means = f32s_from_le(&std::fs::read(&a[1]).expect("means"));
    let sample = f32s_from_le(&std::fs::read(&a[2]).expect("sample"));
    let dim: usize = a[3].parse().unwrap();
    let groups: Vec<usize> = a.get(4).map(|s| s.split(',').map(|g| g.parse().unwrap()).collect()).unwrap_or(vec![64]);
    let iters: usize = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(10);
    let k = means.len() / dim;
    let n = sample.len() / dim;
    let inv = inv_norms(&means, dim);
    println!("k={k} dim={dim} sample={n}");

    let t = Instant::now();
    let mut sink = 0usize;
    for x in sample.chunks_exact(dim) {
        sink += argmax_dot_scaled(x, &means, &inv, dim).0;
    }
    let exh_nearest = t.elapsed().as_secs_f64() * 1e6 / n as f64;
    let t = Instant::now();
    for x in sample.chunks_exact(dim) {
        sink += top_n_dot_scaled(x, &means, &inv, dim, 3)[0].0;
    }
    let exh_top3 = t.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!("exhaustive: nearest {exh_nearest:.0} us, top-3 {exh_top3:.0} us, {k} dots");

    for &g in &groups {
        let t = Instant::now();
        let tree = CentroidTree::build(&means, dim, k, g, iters, 1);
        let build_s = t.elapsed().as_secs_f64();
        let sizes: Vec<usize> = (0..tree.groups()).map(|i| tree.members(i).len()).collect();
        println!(
            "\ngroups={g} iters={iters}: build {build_s:.2} s, {} bytes, group sizes min {} max {} (mean {:.0})",
            tree.to_bytes().len(),
            sizes.iter().min().unwrap(),
            sizes.iter().max().unwrap(),
            k as f64 / tree.groups() as f64
        );
        println!("beam | nearest recall | top-3 recall | avg dots | nearest us | speedup | top-3 us");
        for beam in [1usize, 2, 4, 8, 16, 32] {
            if beam > tree.groups() {
                continue;
            }
            let r1 = tree.recall_check(&sample, &means, &inv, beam);
            let r3 = tree.recall_check_top_m(&sample, &means, &inv, beam, 3);
            let cost = sample.chunks_exact(dim).map(|x| tree.probe_cost(x, beam)).sum::<usize>() as f64 / n as f64;
            let t = Instant::now();
            for x in sample.chunks_exact(dim) {
                sink += tree.nearest(x, &means, &inv, beam).0;
            }
            let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
            let t = Instant::now();
            for x in sample.chunks_exact(dim) {
                sink += tree.top_m(x, &means, &inv, beam, 3)[0].0;
            }
            let us3 = t.elapsed().as_secs_f64() * 1e6 / n as f64;
            println!(
                "{beam:4} | {r1:14.4} | {r3:12.4} | {cost:8.0} | {us:10.0} | {:6.1}x | {us3:8.0}",
                exh_nearest / us
            );
        }
    }
    std::hint::black_box(sink);
}
