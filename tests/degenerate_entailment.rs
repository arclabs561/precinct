//! Probe: degenerate-region containment contract.
//!
//! `Region::entailment_prob` documents `P(self ⊒ other) = vol(self ∩ other) /
//! vol(other)`, `1.0` when `self` contains `other`. When either region has zero
//! volume the ratio is `0/0` and must be resolved by the containment limit, not
//! by accidental f32 arithmetic or an interpolating formula.

use precinct::{AxisBox, Ball, Region, RegionIndex, SearchParams};

#[test]
fn contained_zero_width_box_entails_one() {
    let outer = AxisBox::new(vec![0.0, 0.0], vec![10.0, 10.0]);
    let point = AxisBox::new(vec![5.0, 5.0], vec![5.0, 5.0]);
    let flat = AxisBox::new(vec![5.0, 4.0], vec![5.0, 6.0]);

    let p_point = outer.entailment_prob(&point);
    let p_flat = outer.entailment_prob(&flat);
    println!("outer.entailment_prob(point) = {p_point}");
    println!("outer.entailment_prob(flat)  = {p_flat}");
    assert_eq!(p_point, 1.0);
    assert_eq!(p_flat, 1.0);
}

#[test]
fn zero_radius_other_entails_by_containment() {
    let outer = Ball::new(vec![0.0, 0.0], 5.0);
    let inside = outer.entailment_prob(&Ball::new(vec![1.0, 0.0], 0.0));
    let outside = outer.entailment_prob(&Ball::new(vec![20.0, 0.0], 0.0));
    let boundary = outer.entailment_prob(&Ball::new(vec![5.0, 0.0], 0.0));
    println!("outer.entailment_prob(inside r=0)   = {inside}");
    println!("outer.entailment_prob(outside r=0)  = {outside}");
    println!("outer.entailment_prob(boundary r=0) = {boundary}");
    assert_eq!(inside, 1.0);
    assert_eq!(outside, 0.0);
    assert_eq!(boundary, 1.0, "boundary points are contained (closed ball)");
}

#[test]
fn zero_radius_self_entails_only_an_equal_point() {
    // A zero-radius ball has zero volume, so vol(self ∩ other) = 0 for any
    // other with positive volume: P(point ⊒ ball) = 0. The reported bug was
    // 0.25 here (self r=0, other r=1, centers 0.5 apart).
    let point = Ball::new(vec![0.0, 0.0], 0.0);
    let at_half = point.entailment_prob(&Ball::new(vec![0.5, 0.0], 1.0));
    let at_zero = point.entailment_prob(&Ball::new(vec![0.0, 0.0], 1.0));
    println!("point.entailment_prob(ball r=1, cd=0.5) = {at_half}");
    println!("point.entailment_prob(ball r=1, cd=0.0) = {at_zero}");
    assert_eq!(at_half, 0.0);
    assert_eq!(at_zero, 0.0);

    // 0/0 between two points is the containment limit.
    let same = point.entailment_prob(&Ball::new(vec![0.0, 0.0], 0.0));
    let different = point.entailment_prob(&Ball::new(vec![1.0, 0.0], 0.0));
    println!("point.entailment_prob(point, same)      = {same}");
    println!("point.entailment_prob(point, other)     = {different}");
    assert_eq!(same, 1.0);
    assert_eq!(different, 0.0);
}

#[test]
fn zero_radius_box_self_entails_only_points_on_it() {
    // The AxisBox mirror: vol(flat ∩ solid) = 0, so a flat self entails no
    // positive-volume box, but it does contain a point lying on it.
    let flat = AxisBox::new(vec![5.0, 0.0], vec![5.0, 10.0]);
    let solid = flat.entailment_prob(&AxisBox::new(vec![0.0, 0.0], vec![10.0, 10.0]));
    let on_flat = flat.entailment_prob(&AxisBox::new(vec![5.0, 5.0], vec![5.0, 5.0]));
    let off_flat = flat.entailment_prob(&AxisBox::new(vec![4.0, 5.0], vec![4.0, 5.0]));
    println!("flat.entailment_prob(solid)  = {solid}");
    println!("flat.entailment_prob(on)     = {on_flat}");
    println!("flat.entailment_prob(off)    = {off_flat}");
    assert_eq!(solid, 0.0);
    assert_eq!(on_flat, 1.0);
    assert_eq!(off_flat, 0.0);
}

#[test]
fn soft_subsumers_find_containers_of_a_point_box() {
    // The adapter symptom: a point-embedded entity (zero-width query box) must
    // find the stored regions that contain it at probability 1.
    let mut idx = RegionIndex::new(2, Default::default()).unwrap();
    idx.add(0, AxisBox::new(vec![-1.0, -1.0], vec![1.0, 1.0]))
        .unwrap();
    idx.add(1, AxisBox::new(vec![5.0, 5.0], vec![6.0, 6.0]))
        .unwrap();
    idx.build().unwrap();

    let point = AxisBox::new(vec![0.0, 0.0], vec![0.0, 0.0]);
    let got = idx
        .subsumers_soft(&point, 1.0, SearchParams::default())
        .unwrap();
    println!("soft subsumers of point box: {got:?}");
    assert_eq!(got, vec![(0, 1.0)]);
}
