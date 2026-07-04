//! Measure sidecar-first reopen vs rebuild for the segstore-backed region store.
//!
//! Run:
//! `cargo run --release --features store --example store_reopen_diagnostics`

use std::sync::Arc;
use std::time::{Duration, Instant};

use durability::{Directory, MemoryDirectory};
use precinct::{store::UpdatableIndex, AxisBox, IndexParams, SearchParams};

const N: usize = 1_000;
const DIM: usize = 16;
const FLUSH: usize = 200;

type DynError = Box<dyn std::error::Error>;
type StoreDir = Arc<dyn Directory>;
type SearchHits = Vec<(u32, f32)>;

fn main() -> Result<(), DynError> {
    let (load_dir, query) = build_checkpointed_dir()?;
    let load_sidecars = sidecar_count(&load_dir)?;
    let (load_elapsed, load_hits) = first_search(load_dir.clone(), &query)?;

    let (rebuild_dir, rebuild_query) = build_checkpointed_dir()?;
    let sidecars_before_delete = sidecar_count(&rebuild_dir)?;
    delete_sidecars(&rebuild_dir)?;
    let sidecars_after_delete = sidecar_count(&rebuild_dir)?;
    let (rebuild_elapsed, rebuild_hits) = first_search(rebuild_dir, &rebuild_query)?;

    assert!(!load_hits.is_empty());
    assert!(!rebuild_hits.is_empty());

    println!("regions: {N}, dim: {DIM}, flush threshold: {FLUSH}");
    println!("sidecars loaded path: {load_sidecars}");
    println!(
        "sidecars rebuild path before/after delete: {sidecars_before_delete}/{sidecars_after_delete}"
    );
    println!("first search with sidecars: {}", micros(load_elapsed));
    println!(
        "first search after deleting sidecars: {}",
        micros(rebuild_elapsed)
    );
    println!("top hit with sidecars: {:?}", load_hits.first());
    println!("top hit after rebuild: {:?}", rebuild_hits.first());

    Ok(())
}

fn build_checkpointed_dir() -> Result<(StoreDir, Vec<f32>), DynError> {
    let dir: StoreDir = MemoryDirectory::arc();
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut index = UpdatableIndex::open(dir.clone(), FLUSH, DIM, IndexParams::default())?;
    for id in 0..N {
        index.add(id as u32, region(&mut state))?;
    }
    index.checkpoint()?;
    Ok((dir, point(&mut state)))
}

fn first_search(dir: StoreDir, query: &[f32]) -> Result<(Duration, SearchHits), DynError> {
    let index = UpdatableIndex::open(dir, FLUSH, DIM, IndexParams::default())?;
    let params = SearchParams {
        ef: 100,
        overretrieve: 100,
    };
    let start = Instant::now();
    let hits = index.search(query, 10, params);
    Ok((start.elapsed(), hits))
}

fn delete_sidecars(dir: &StoreDir) -> Result<(), DynError> {
    for name in dir.list_dir("")? {
        if name.starts_with("segstore.idx.") {
            dir.delete(&name)?;
        }
    }
    Ok(())
}

fn sidecar_count(dir: &StoreDir) -> Result<usize, DynError> {
    Ok(dir
        .list_dir("")?
        .into_iter()
        .filter(|name| name.starts_with("segstore.idx."))
        .count())
}

fn region(state: &mut u64) -> AxisBox {
    let center: Vec<f32> = (0..DIM)
        .map(|_| (xorshift(state) % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    let half: Vec<f32> = (0..DIM)
        .map(|_| 0.1 + (xorshift(state) % 100) as f32 / 1000.0)
        .collect();
    AxisBox::from_center_offset(center, half)
}

fn point(state: &mut u64) -> Vec<f32> {
    (0..DIM)
        .map(|_| (xorshift(state) % 2000) as f32 / 1000.0 - 1.0)
        .collect()
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn micros(duration: Duration) -> String {
    format!("{} us", duration.as_micros())
}
