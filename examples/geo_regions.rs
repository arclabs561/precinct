//! Nearest-region search over real geographic boxes (Natural Earth countries).
//!
//! Each country's axis-aligned bounding box (min/max longitude and latitude) is
//! a region; a query is a `[lon, lat]` point. This is the most literal form of
//! region search, and it makes precinct's distinguishing property visible: the
//! nearest region by *surface* distance (point-to-box) is often not the nearest
//! by *center* distance. A point in the South Pacific is near Chile's long
//! coastline (its box edge) yet far from Chile's centroid; a plain point-ANN
//! over centers would miss it.
//!
//! Data (gitignored): run `scripts/fetch_natural_earth.sh` first. Without it
//! this example prints instructions and exits 0.
//!
//! Run: cargo run --release --example geo_regions

use precinct::{box_to_point_l2, AxisBox, IndexParams, RegionIndex, SearchParams};
use serde_json::Value;

const COUNTRIES: &str = "data/ne_countries.json";

fn main() {
    let path = std::env::var("NE_COUNTRIES").unwrap_or_else(|_| COUNTRIES.to_string());
    let Some(regions) = load_countries(&path) else {
        eprintln!("Natural Earth countries not found at {path}.");
        eprintln!("Fetch with: scripts/fetch_natural_earth.sh");
        return; // data-gated: a clean no-op when the dataset is absent.
    };
    println!("Loaded {} country regions.", regions.len());

    let mut idx = RegionIndex::<AxisBox>::new(2, IndexParams::default()).expect("index");
    for (i, (_, b)) in regions.iter().enumerate() {
        idx.add(i as u32, b.clone()).expect("add");
    }
    idx.build().expect("build");

    // A few illustrative `[lon, lat]` queries. (Bounding boxes are coarse: a
    // country with far-flung territories, e.g. France, has a box spanning them,
    // so an open-ocean point can fall inside it. The surface-distance rerank is
    // still exact for the boxes as given.)
    let queries = [
        ("South Pacific (off Chile)", [-90.0, -35.0]),
        ("central Europe", [10.0, 50.0]),
        ("Bay of Bengal", [88.0, 15.0]),
        ("Sahara interior", [12.0, 23.0]),
    ];

    let params = || SearchParams {
        ef: 64,
        overretrieve: 20,
    };
    for (label, q) in queries {
        let got = idx.search(&q, 3, params()).expect("search");
        let names: Vec<String> = got
            .iter()
            .map(|(id, d)| format!("{} ({:.1})", regions[*id as usize].0, d))
            .collect();
        println!("  {label:<28} -> {}", names.join(", "));

        // Contrast: the nearest region by box-center distance can differ.
        let by_center = nearest_by_center(&regions, &q);
        if by_center != regions[got[0].0 as usize].0 {
            println!(
                "      (nearest by center would be {by_center}, not {})",
                regions[got[0].0 as usize].0
            );
        }
    }

    // Sanity: recall@3 vs exhaustive surface-distance scan over a grid of points.
    let grid: Vec<[f32; 2]> = (-170..170)
        .step_by(20)
        .flat_map(|lon| {
            (-80..80)
                .step_by(20)
                .map(move |lat| [lon as f32, lat as f32])
        })
        .collect();
    let (mut hit, mut total) = (0usize, 0usize);
    for q in &grid {
        let truth = exhaustive_top_k(&regions, q, 3);
        let got = idx.search(q, 3, params()).expect("search");
        let ids: std::collections::HashSet<u32> = got.iter().map(|(id, _)| *id).collect();
        for (id, _) in &truth {
            hit += usize::from(ids.contains(id));
            total += 1;
        }
    }
    println!(
        "recall@3 over a {}-point world grid: {:.1}%",
        grid.len(),
        hit as f64 / total as f64 * 100.0
    );
}

/// Parse a Natural Earth `FeatureCollection` into `(name, bounding-box)` regions.
fn load_countries(path: &str) -> Option<Vec<(String, AxisBox)>> {
    let text = std::fs::read_to_string(path).ok()?;
    let doc: Value = serde_json::from_str(&text).ok()?;
    let features = doc.get("features")?.as_array()?;
    let mut out = Vec::with_capacity(features.len());
    for f in features {
        let name = f
            .pointer("/properties/NAME")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();
        let coords = f.pointer("/geometry/coordinates")?;
        let mut lo = [f32::INFINITY; 2];
        let mut hi = [f32::NEG_INFINITY; 2];
        collect_bbox(coords, &mut lo, &mut hi);
        if lo[0].is_finite() {
            out.push((name, AxisBox::new(lo.to_vec(), hi.to_vec())));
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Walk nested GeoJSON coordinate arrays, expanding the running bbox at each
/// `[lon, lat]` leaf.
fn collect_bbox(v: &Value, lo: &mut [f32; 2], hi: &mut [f32; 2]) {
    if let Value::Array(a) = v {
        if let (Some(lon), Some(lat)) = (
            a.first().and_then(Value::as_f64),
            a.get(1).and_then(Value::as_f64),
        ) {
            if a.len() == 2 {
                lo[0] = lo[0].min(lon as f32);
                hi[0] = hi[0].max(lon as f32);
                lo[1] = lo[1].min(lat as f32);
                hi[1] = hi[1].max(lat as f32);
                return;
            }
        }
        for e in a {
            collect_bbox(e, lo, hi);
        }
    }
}

/// Name of the region whose box center is nearest the query (the point-ANN proxy).
fn nearest_by_center(regions: &[(String, AxisBox)], q: &[f32; 2]) -> String {
    regions
        .iter()
        .min_by(|a, b| center_dist(&a.1, q).total_cmp(&center_dist(&b.1, q)))
        .map(|(n, _)| n.clone())
        .unwrap_or_default()
}

fn center_dist(b: &AxisBox, q: &[f32; 2]) -> f32 {
    let cx = (b.min()[0] + b.max()[0]) / 2.0;
    let cy = (b.min()[1] + b.max()[1]) / 2.0;
    ((cx - q[0]).powi(2) + (cy - q[1]).powi(2)).sqrt()
}

/// Exhaustive top-k regions by point-to-region surface distance (ground truth).
fn exhaustive_top_k(regions: &[(String, AxisBox)], q: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = regions
        .iter()
        .enumerate()
        .map(|(i, (_, b))| (i as u32, box_to_point_l2(b.min(), b.max(), q)))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    all.truncate(k);
    all
}
