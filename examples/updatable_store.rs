//! Updatable region store: add, checkpoint, delete, reopen, and query.
//!
//! Run: cargo run --features store --example updatable_store

use durability::MemoryDirectory;
use precinct::{store::UpdatableIndex, AxisBox, IndexParams, SearchParams};

fn b(lo: f32, hi: f32) -> AxisBox {
    AxisBox::new(vec![lo, lo], vec![hi, hi])
}

fn params() -> SearchParams {
    SearchParams {
        ef: 100,
        overretrieve: 100,
    }
}

fn ids(results: &[(u32, f32)]) -> Vec<u32> {
    results.iter().map(|(id, _)| *id).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = MemoryDirectory::arc();
    let inner = AxisBox::new(vec![1.25, 1.25], vec![1.75, 1.75]);
    let overlap = AxisBox::new(vec![9.0, 9.0], vec![11.0, 11.0]);

    {
        let mut index = UpdatableIndex::open(dir.clone(), 2, 2, IndexParams::default())?;
        index.extend([
            (0, b(0.0, 10.0)),
            (1, b(1.0, 2.0)),
            (2, b(8.0, 12.0)),
            (3, b(20.0, 21.0)),
            (4, b(4.0, 6.0)),
        ])?;
        index.checkpoint()?;

        let containing = index.containing(&[5.0, 5.0], params());
        let nearest_origin = index.search(&[0.5, 0.5], 3, params());

        println!("before delete:");
        println!("  containing [5,5]: {containing:?}");
        println!("  nearest [0.5,0.5]: {:?}", ids(&nearest_origin));

        assert_eq!(containing, vec![0, 4]);
        assert_eq!(nearest_origin.first().map(|(id, _)| *id), Some(0));

        index.delete(0)?;
        index.checkpoint()?;
    }

    let recovered = UpdatableIndex::open(dir, 2, 2, IndexParams::default())?;
    let containing = recovered.containing(&[5.0, 5.0], params());
    let subsumers = recovered.subsumers(&inner, params());
    let overlapping = recovered.overlapping(&overlap, params());
    let nearest_region = recovered.nearest_region(&overlap, 1, params());
    let nearest_origin = recovered.search(&[0.5, 0.5], 3, params());

    println!("after reopen:");
    println!("  containing [5,5]: {containing:?}");
    println!("  subsumers [1.25,1.75]: {subsumers:?}");
    println!("  overlapping [9,11]: {overlapping:?}");
    println!("  nearest region [9,11]: {:?}", ids(&nearest_region));
    println!("  nearest [0.5,0.5]: {:?}", ids(&nearest_origin));

    assert_eq!(containing, vec![4]);
    assert_eq!(subsumers, vec![1]);
    assert_eq!(overlapping, vec![2]);
    assert_eq!(nearest_region.first().map(|(id, _)| *id), Some(2));
    assert!(!ids(&nearest_origin).contains(&0));

    Ok(())
}
