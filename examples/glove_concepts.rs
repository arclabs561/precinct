//! Recall benchmark on real high-dimensional concept regions built from GloVe.
//!
//! A "concept region" is the axis-aligned bounding box of a cluster of related
//! word vectors: a box over `{king, queen, prince, ...}` is the region those
//! words occupy. This is the box-embedding motivation (a region = a set), but
//! grounded in a real word-vector distribution instead of uniform-random boxes.
//!
//! We cluster the top-N GloVe vectors with k-means (via `clump`), build one
//! box per cluster, then measure precinct's recall@k against an exhaustive
//! point-to-region scan, the property that matters: does the HNSW-over-centers
//! plus region-distance rerank recover the true nearest concept regions on real
//! high-dim data?
//!
//! Data (gitignored): run `scripts/fetch_glove.sh` first. Without it this
//! example prints instructions and exits 0.
//!
//! Run: cargo run --release --example glove_concepts

use std::time::Instant;

use clump::Kmeans;
use precinct::{box_to_point_l2, AxisBox, IndexParams, RegionIndex, SearchParams};

const GLOVE: &str = "data/glove.6B.50d.txt";
const N_WORDS: usize = 50_000; // top-N most frequent (GloVe is frequency-sorted)
const K_CLUSTERS: usize = 5_000;
const N_QUERIES: usize = 2_000;
const K: usize = 10;

fn main() {
    let path = std::env::var("GLOVE").unwrap_or_else(|_| GLOVE.to_string());
    let Some((dim, vectors)) = load_glove(&path, N_WORDS) else {
        eprintln!("GloVe vectors not found at {path}.");
        eprintln!("Fetch with: scripts/fetch_glove.sh");
        return; // data-gated: a clean no-op when the dataset is absent.
    };
    println!("Loaded {} GloVe vectors (dim {dim}).", vectors.len());

    // Cluster into concept regions.
    let t = Instant::now();
    let fit = Kmeans::new(K_CLUSTERS)
        .with_seed(42)
        .with_max_iter(25)
        .fit(&vectors)
        .expect("kmeans fit");
    println!(
        "Clustered into {} concepts in {:.1}s.",
        fit.centroids.len(),
        t.elapsed().as_secs_f64()
    );

    // Each non-empty cluster becomes the axis-aligned bounding box of its members.
    let boxes = cluster_boxes(&vectors, &fit.labels, dim);
    let mean_side = boxes
        .iter()
        .map(|b| {
            b.min()
                .iter()
                .zip(b.max())
                .map(|(lo, hi)| hi - lo)
                .sum::<f32>()
                / dim as f32
        })
        .sum::<f32>()
        / boxes.len() as f32;
    println!(
        "Built {} concept boxes (mean side {mean_side:.3}).",
        boxes.len()
    );

    // Index the regions.
    let mut idx = RegionIndex::<AxisBox>::new(dim, IndexParams::default()).expect("index");
    for (i, b) in boxes.iter().enumerate() {
        idx.add(i as u32, b.clone()).expect("add");
    }
    idx.build().expect("build");

    // Query with a deterministic sample of the corpus points.
    let queries: Vec<&Vec<f32>> = vectors
        .iter()
        .step_by(vectors.len() / N_QUERIES)
        .take(N_QUERIES)
        .collect();

    for over in [10usize, 50] {
        let t = Instant::now();
        let mut hit = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let truth = exhaustive_top_k(&boxes, q, K);
            let params = SearchParams {
                ef: 100,
                overretrieve: over,
            };
            let got = idx.search(q, K, params).expect("search");
            let got_ids: std::collections::HashSet<u32> = got.iter().map(|(id, _)| *id).collect();
            for (id, _) in &truth {
                if got_ids.contains(id) {
                    hit += 1;
                }
                total += 1;
            }
        }
        let recall = hit as f64 / total as f64;
        let qps = queries.len() as f64 / t.elapsed().as_secs_f64();
        println!(
            "recall@{K} ({over}x over-retrieve): {:.1}%   ({qps:.0} q/s)",
            recall * 100.0
        );
    }
}

/// Parse the first `n` GloVe lines (`word v0 v1 ...`) into vectors. Returns the
/// dimension and the vectors, or `None` if the file is missing.
fn load_glove(path: &str, n: usize) -> Option<(usize, Vec<Vec<f32>>)> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path).ok()?;
    let mut dim = 0;
    let mut vectors = Vec::with_capacity(n);
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if vectors.len() >= n {
            break;
        }
        let v: Vec<f32> = line
            .split_whitespace()
            .skip(1) // the word token
            .filter_map(|t| t.parse().ok())
            .collect();
        if v.is_empty() {
            continue;
        }
        dim = v.len();
        vectors.push(v);
    }
    if vectors.is_empty() {
        return None;
    }
    Some((dim, vectors))
}

/// Per-cluster axis-aligned bounding box over the cluster's member vectors.
fn cluster_boxes(vectors: &[Vec<f32>], labels: &[usize], dim: usize) -> Vec<AxisBox> {
    let k = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut lo = vec![vec![f32::INFINITY; dim]; k];
    let mut hi = vec![vec![f32::NEG_INFINITY; dim]; k];
    let mut count = vec![0usize; k];
    for (v, &c) in vectors.iter().zip(labels) {
        count[c] += 1;
        for d in 0..dim {
            lo[c][d] = lo[c][d].min(v[d]);
            hi[c][d] = hi[c][d].max(v[d]);
        }
    }
    (0..k)
        .filter(|&c| count[c] > 0)
        .map(|c| AxisBox::new(lo[c].clone(), hi[c].clone()))
        .collect()
}

/// Exhaustive top-k regions by point-to-region surface distance (ground truth).
fn exhaustive_top_k(boxes: &[AxisBox], q: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| (i as u32, box_to_point_l2(b.min(), b.max(), q)))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    all.truncate(k);
    all
}
