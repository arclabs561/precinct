//! `RegionIndex::search_with_distance` passes the closure insertion-order ids.

use precinct::{AxisBox, Region};

fn rng(s: &mut u64) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    ((*s >> 40) as f32) / (1u64 << 24) as f32
}

#[test]
fn search_with_distance_closure_ids_are_insertion_order() {
    use precinct::RegionIndex;
    let mut s = 0x1234_5678u64;
    let mut regions = Vec::new();
    let mut idx = RegionIndex::new(4, Default::default()).unwrap();
    for i in 0..300u32 {
        let c: Vec<f32> = (0..4).map(|_| rng(&mut s) * 20.0).collect();
        let h: Vec<f32> = (0..4).map(|_| 0.1 + rng(&mut s)).collect();
        let b = AxisBox::from_center_offset(c, h);
        regions.push(b.clone());
        idx.add(i, b).unwrap();
    }
    idx.build().unwrap();
    let mut mismatched = 0;
    for _ in 0..20 {
        let q: Vec<f32> = (0..4).map(|_| rng(&mut s) * 20.0).collect();
        let got = idx
            .search_with_distance(&q, 5, 300, &|qq, id| {
                regions[id as usize].distance_to_point(qq)
            })
            .unwrap();
        // Every returned (doc_id, d) must carry that doc's true distance.
        for (doc, d) in got {
            if (regions[doc as usize].distance_to_point(&q) - d).abs() > 1e-5 {
                mismatched += 1;
            }
        }
    }
    assert_eq!(
        mismatched, 0,
        "{mismatched}/100 results scored against a different region"
    );
}
