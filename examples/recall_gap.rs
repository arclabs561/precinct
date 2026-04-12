//! Recall-gap benchmark: center-based ANN vs exhaustive region search.
//!
//! Generates synthetic box embeddings with configurable width distributions,
//! builds a RegionIndex, and measures recall@k at various overretrieve factors.
//!
//! Run: cargo run --release --example recall_gap

use precinct::{AxisBox, Region, RegionIndex, SearchParams};
use rand::Rng;
use std::time::Instant;

/// Configuration for one benchmark run.
struct Config {
    /// Number of boxes to index.
    n: usize,
    /// Embedding dimension.
    dim: usize,
    /// Number of queries.
    n_queries: usize,
    /// k for recall@k.
    k: usize,
    /// Half-width distribution: (mean, stddev).
    /// Larger widths = broader boxes = bigger recall gap expected.
    width_mean: f32,
    width_std: f32,
    /// Label for this configuration.
    label: &'static str,
}

fn generate_boxes(cfg: &Config, rng: &mut impl Rng) -> Vec<AxisBox> {
    (0..cfg.n)
        .map(|_| {
            let center: Vec<f32> = (0..cfg.dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let half_widths: Vec<f32> = (0..cfg.dim)
                .map(|_| (cfg.width_mean + rng.random_range(-1.0..1.0) * cfg.width_std).max(0.01))
                .collect();
            AxisBox::from_center_offset(center, half_widths)
        })
        .collect()
}

fn generate_queries(cfg: &Config, rng: &mut impl Rng) -> Vec<Vec<f32>> {
    (0..cfg.n_queries)
        .map(|_| (0..cfg.dim).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect()
}

fn recall_at_k(exact: &[(u32, f32)], approx: &[(u32, f32)], k: usize) -> f32 {
    let exact_ids: std::collections::HashSet<u32> =
        exact.iter().take(k).map(|(id, _)| *id).collect();
    let approx_ids: std::collections::HashSet<u32> =
        approx.iter().take(k).map(|(id, _)| *id).collect();
    let hits = exact_ids.intersection(&approx_ids).count();
    hits as f32 / k as f32
}

fn run_benchmark(cfg: &Config) {
    let mut rng = rand::rng();

    // Generate data
    let boxes = generate_boxes(cfg, &mut rng);
    let queries = generate_queries(cfg, &mut rng);

    // Build index
    let t = Instant::now();
    let mut index = RegionIndex::new(cfg.dim, Default::default()).unwrap();
    for (i, b) in boxes.iter().enumerate() {
        index.add(i as u32, b.clone()).unwrap();
    }
    index.build().unwrap();
    let build_ms = t.elapsed().as_millis();

    // Compute exhaustive ground truth
    let t = Instant::now();
    let ground_truth: Vec<Vec<(u32, f32)>> = queries
        .iter()
        .map(|q| index.search_exhaustive(q, cfg.k))
        .collect();
    let exhaustive_ms = t.elapsed().as_millis();

    println!(
        "\n=== {} (n={}, dim={}, width_mean={:.2}) ===",
        cfg.label, cfg.n, cfg.dim, cfg.width_mean
    );
    println!("Build: {}ms | Exhaustive: {}ms", build_ms, exhaustive_ms);

    // Sweep overretrieve factors
    for overretrieve in [1, 2, 5, 10, 20, 50] {
        let t = Instant::now();
        let mut total_recall = 0.0f32;

        for (qi, q) in queries.iter().enumerate() {
            let params = SearchParams {
                ef: 200,
                overretrieve,
            };
            let approx = index.search(q, cfg.k, params).unwrap();
            total_recall += recall_at_k(&ground_truth[qi], &approx, cfg.k);
        }

        let mean_recall = total_recall / cfg.n_queries as f32;
        let search_ms = t.elapsed().as_millis();

        println!(
            "  overretrieve={:>3}x  recall@{}={:.4}  search={}ms ({:.1} qps)",
            overretrieve,
            cfg.k,
            mean_recall,
            search_ms,
            cfg.n_queries as f64 / (search_ms as f64 / 1000.0)
        );
    }

    // Also measure: how many ground-truth results are inside a box?
    let mut inside_count = 0usize;
    let mut total_count = 0usize;
    for (qi, q) in queries.iter().enumerate() {
        for (id, _) in ground_truth[qi].iter().take(cfg.k) {
            let b = &boxes[*id as usize];
            if b.contains(q) {
                inside_count += 1;
            }
            total_count += 1;
        }
    }
    println!(
        "  (ground truth: {:.1}% of top-{} results contain the query point)",
        100.0 * inside_count as f64 / total_count as f64,
        cfg.k
    );
}

fn main() {
    let configs = vec![
        // Narrow boxes: center is a good proxy
        Config {
            n: 10_000,
            dim: 128,
            n_queries: 200,
            k: 10,
            width_mean: 0.01,
            width_std: 0.005,
            label: "narrow (d=128, w=0.01)",
        },
        // Medium boxes
        Config {
            n: 10_000,
            dim: 128,
            n_queries: 200,
            k: 10,
            width_mean: 0.1,
            width_std: 0.05,
            label: "medium (d=128, w=0.1)",
        },
        // Wide boxes: center is a poor proxy
        Config {
            n: 10_000,
            dim: 128,
            n_queries: 200,
            k: 10,
            width_mean: 0.5,
            width_std: 0.2,
            label: "wide (d=128, w=0.5)",
        },
        // Mixed hierarchy: some narrow, some wide (simulates ontology)
        Config {
            n: 10_000,
            dim: 128,
            n_queries: 200,
            k: 10,
            width_mean: 0.2,
            width_std: 0.3,
            label: "mixed hierarchy (d=128, w=0.2+/-0.3)",
        },
        // Higher dimension
        Config {
            n: 10_000,
            dim: 400,
            n_queries: 100,
            k: 10,
            width_mean: 0.1,
            width_std: 0.05,
            label: "medium (d=400, w=0.1)",
        },
        // Scale test
        Config {
            n: 50_000,
            dim: 128,
            n_queries: 100,
            k: 10,
            width_mean: 0.1,
            width_std: 0.05,
            label: "scale 50K (d=128, w=0.1)",
        },
    ];

    for cfg in &configs {
        run_benchmark(cfg);
    }
}
