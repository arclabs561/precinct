//! Integration test exercising the public `RegionIndex` API end to end.
//!
//! Invariant (correct nearest-region retrieval): for a query point, a built
//! index returns the region whose *surface* (point-to-region L2) distance is
//! smallest, and that result agrees with the exhaustive ground truth.
//!
//! The case is constructed so the nearest region by surface distance is NOT the
//! nearest by center distance. Since the HNSW graph indexes centers, getting
//! this right proves the rerank step consults true region geometry rather than
//! the proxy center -- the property that distinguishes precinct from a plain
//! point-ANN index.

use precinct::{AxisBox, Region, RegionIndex, SearchParams};

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[test]
fn nearest_region_uses_surface_distance_not_center() {
    let mut idx = RegionIndex::new(2, Default::default()).unwrap();

    // Region A (id 0): a wide, flat slab. Its center is far from the query
    // (y = 1.75), but its lower edge sits at y = 0.5, so its *surface* is the
    // closest thing to q = [0, 0].
    let a = AxisBox::new(vec![-5.0, 0.5], vec![5.0, 3.0]);
    // Region B (id 1): a tiny box whose center (y = 1.0) is CLOSER to the query
    // than A's center, but whose surface (y = 0.95) is FARTHER than A's surface.
    let b = AxisBox::new(vec![-0.05, 0.95], vec![0.05, 1.05]);

    idx.add(0, a.clone()).unwrap();
    idx.add(1, b.clone()).unwrap();

    // Decoy regions placed far away: they cannot win the query, and they keep
    // the HNSW graph from being a degenerate handful of nodes.
    for i in 2..16u32 {
        let o = 50.0 + (i as f32) * 5.0;
        idx.add(i, AxisBox::new(vec![o, o], vec![o + 1.0, o + 1.0]))
            .unwrap();
    }
    idx.build().unwrap();

    let q = [0.0_f32, 0.0];

    // Geometry preconditions, independent of the index: A's surface is nearer
    // than B's, while B's center is nearer than A's. This is what makes the
    // center-vs-surface distinction load-bearing.
    assert!(
        a.distance_to_point(&q) < b.distance_to_point(&q),
        "setup: A surface ({}) must be nearer than B surface ({})",
        a.distance_to_point(&q),
        b.distance_to_point(&q)
    );
    assert!(
        l2(b.center(), &q) < l2(a.center(), &q),
        "setup: B center ({}) must be nearer than A center ({})",
        l2(b.center(), &q),
        l2(a.center(), &q)
    );

    // overretrieve = 16 with 16 indexed regions fetches the whole candidate set,
    // so the rerank stage sees both A and B regardless of HNSW recall: the only
    // way to return A is to score by true surface distance.
    let params = SearchParams {
        ef: 200,
        overretrieve: 16,
    };
    let approx = idx.search(&q, 1, params).unwrap();
    assert_eq!(approx.len(), 1);
    assert_eq!(
        approx[0].0, 0,
        "nearest region must be the wide slab (id 0), not the close-centered box (id 1)"
    );
    assert!(
        (approx[0].1 - a.distance_to_point(&q)).abs() < 1e-6,
        "reported distance {} must equal A's true surface distance {}",
        approx[0].1,
        a.distance_to_point(&q)
    );

    // The approximate search must agree with exhaustive ground truth.
    let exact = idx.search_exhaustive(&q, 1);
    assert_eq!(exact[0].0, 0);
    assert_eq!(approx[0].0, exact[0].0);
}
