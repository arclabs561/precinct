/// A geometric region in d-dimensional space.
///
/// Regions have a center point (used for ANN candidate retrieval), a distance
/// function from an external point to the region surface, and a containment
/// predicate.
pub trait Region {
    /// Dimensionality of the embedding space.
    fn dim(&self) -> usize;

    /// Center of the region. Used as the proxy point for ANN indexing.
    fn center(&self) -> &[f32];

    /// Minimum L2 distance from `point` to the region surface.
    /// Returns 0.0 if the point is inside the region.
    fn distance_to_point(&self, point: &[f32]) -> f32;

    /// Whether `point` lies inside (or on the boundary of) this region.
    fn contains(&self, point: &[f32]) -> bool;

    /// A ball `(center, radius)` that encloses this region.
    ///
    /// Used as a conservative proxy for the power-distance lift in
    /// [`RegionIndex`](crate::RegionIndex): `self ⊆ bounding_ball`, so any point
    /// inside `self` is inside the bounding ball. The tighter the ball, the less
    /// over-retrieval the rerank pays for. For a `Ball` this is exact.
    fn bounding_ball(&self) -> (Vec<f32>, f32);

    /// Whether this region fully contains `other` (`self ⊇ other`).
    ///
    /// The region-to-region subsumption predicate: in a trained ontology,
    /// `self.contains_region(other)` means `self` is a more general concept than
    /// `other`.
    fn contains_region(&self, other: &Self) -> bool
    where
        Self: Sized;

    /// Whether this region intersects `other` (`self ∩ other ≠ ∅`).
    ///
    /// The overlap predicate: the conjunction primitive for region queries (two
    /// concepts share members). Symmetric: `a.overlaps_region(b) ==
    /// b.overlaps_region(a)`.
    fn overlaps_region(&self, other: &Self) -> bool
    where
        Self: Sized;

    /// Natural log of the region's volume.
    ///
    /// A measure of generality: a larger region is a more general concept. Log
    /// space because volume underflows to zero in high dimensions.
    fn log_volume(&self) -> f32;

    /// The probability that `self` subsumes `other`, `P(self ⊒ other) =
    /// vol(self ∩ other) / vol(other)`.
    ///
    /// The soft form of [`contains_region`](Self::contains_region): `1.0` when
    /// `self` fully contains `other`, `0.0` when they are disjoint, in between
    /// for partial overlap. This is the box-lattice conditional probability
    /// (Vilnis et al. 2018); exact for boxes, an approximation for balls (the
    /// exact lens volume needs the regularized incomplete beta function).
    fn entailment_prob(&self, other: &Self) -> f32
    where
        Self: Sized;
}

// ─── AxisBox ─────────────────────────────────────────────────────────────────

/// Axis-aligned hyperrectangle defined by min/max corners.
///
/// The standard representation for box embeddings (Query2Box, Box2EL,
/// BoxTaxo). Each dimension `i` spans `[min[i], max[i]]`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AxisBox {
    min: Vec<f32>,
    max: Vec<f32>,
    center: Vec<f32>,
}

impl AxisBox {
    /// Create a box from min and max corners.
    ///
    /// # Panics
    ///
    /// Panics if `min` and `max` have different lengths, or if any
    /// `min[i] > max[i]`.
    pub fn new(min: Vec<f32>, max: Vec<f32>) -> Self {
        assert_eq!(min.len(), max.len(), "min/max dimension mismatch");
        debug_assert!(
            min.iter().zip(max.iter()).all(|(lo, hi)| lo <= hi),
            "min must be <= max in every dimension"
        );
        let center: Vec<f32> = min
            .iter()
            .zip(max.iter())
            .map(|(lo, hi)| (lo + hi) * 0.5)
            .collect();
        Self { min, max, center }
    }

    /// Create a box from center and half-widths (offset representation).
    ///
    /// This matches the `center/offset` parameterization used by EL++ trainers
    /// (Box2EL, TransBox): `min = center - offset`, `max = center + offset`.
    pub fn from_center_offset(center: Vec<f32>, offset: Vec<f32>) -> Self {
        assert_eq!(center.len(), offset.len());
        let min: Vec<f32> = center
            .iter()
            .zip(offset.iter())
            .map(|(c, o)| c - o.abs())
            .collect();
        let max: Vec<f32> = center
            .iter()
            .zip(offset.iter())
            .map(|(c, o)| c + o.abs())
            .collect();
        Self { min, max, center }
    }

    /// Create a box from the `mu/delta` (log-width) parameterization.
    ///
    /// This matches `TrainableBox` in subsume:
    /// `min = mu - exp(delta)/2`, `max = mu + exp(delta)/2`.
    pub fn from_mu_delta(mu: Vec<f32>, delta: Vec<f32>) -> Self {
        assert_eq!(mu.len(), delta.len());
        let min: Vec<f32> = mu
            .iter()
            .zip(delta.iter())
            .map(|(m, d)| m - d.exp() * 0.5)
            .collect();
        let max: Vec<f32> = mu
            .iter()
            .zip(delta.iter())
            .map(|(m, d)| m + d.exp() * 0.5)
            .collect();
        let center = mu;
        Self { min, max, center }
    }

    pub fn min(&self) -> &[f32] {
        &self.min
    }

    pub fn max(&self) -> &[f32] {
        &self.max
    }

    /// Per-dimension half-widths.
    pub fn half_widths(&self) -> Vec<f32> {
        self.min
            .iter()
            .zip(self.max.iter())
            .map(|(lo, hi)| (hi - lo) * 0.5)
            .collect()
    }

    /// Log-volume of the box (sum of log side-lengths).
    ///
    /// Returns `f32::NEG_INFINITY` if any dimension has zero width.
    pub fn log_volume(&self) -> f32 {
        self.min
            .iter()
            .zip(self.max.iter())
            .map(|(lo, hi)| (hi - lo).ln())
            .sum()
    }
}

impl Region for AxisBox {
    fn dim(&self) -> usize {
        self.min.len()
    }

    fn center(&self) -> &[f32] {
        &self.center
    }

    fn distance_to_point(&self, point: &[f32]) -> f32 {
        debug_assert_eq!(point.len(), self.min.len(), "point dimension mismatch");
        crate::distance::box_to_point_l2(&self.min, &self.max, point)
    }

    fn contains(&self, point: &[f32]) -> bool {
        debug_assert_eq!(point.len(), self.min.len(), "point dimension mismatch");
        point
            .iter()
            .zip(self.min.iter())
            .zip(self.max.iter())
            .all(|((p, lo), hi)| *p >= *lo && *p <= *hi)
    }

    fn bounding_ball(&self) -> (Vec<f32>, f32) {
        // The sphere through the box corners: center, radius = ||half-widths||_2.
        let radius = self
            .min
            .iter()
            .zip(self.max.iter())
            .map(|(lo, hi)| {
                let h = (hi - lo) * 0.5;
                h * h
            })
            .sum::<f32>()
            .sqrt();
        (self.center.clone(), radius)
    }

    fn contains_region(&self, other: &Self) -> bool {
        // self ⊇ other iff self.min <= other.min and self.max >= other.max,
        // componentwise.
        self.min.iter().zip(other.min.iter()).all(|(s, o)| *s <= *o)
            && self.max.iter().zip(other.max.iter()).all(|(s, o)| *s >= *o)
    }

    fn overlaps_region(&self, other: &Self) -> bool {
        // Boxes intersect iff their intervals overlap in every dimension.
        self.min
            .iter()
            .zip(self.max.iter())
            .zip(other.min.iter().zip(other.max.iter()))
            .all(|((s_lo, s_hi), (o_lo, o_hi))| *s_lo <= *o_hi && *o_lo <= *s_hi)
    }

    fn log_volume(&self) -> f32 {
        // Sum of log side-lengths; the inherent method does the same.
        AxisBox::log_volume(self)
    }

    fn entailment_prob(&self, other: &Self) -> f32 {
        // vol(self ∩ other) / vol(other), in log space then exponentiated.
        // The intersection box is [max(lo), min(hi)] per dimension; if empty in
        // any dimension the regions are disjoint and the probability is 0.
        let mut inter_log_vol = 0.0f32;
        for (((s_lo, s_hi), o_lo), o_hi) in self
            .min
            .iter()
            .zip(self.max.iter())
            .zip(other.min.iter())
            .zip(other.max.iter())
        {
            let lo = s_lo.max(*o_lo);
            let hi = s_hi.min(*o_hi);
            if hi <= lo {
                return 0.0; // disjoint in this dimension
            }
            inter_log_vol += (hi - lo).ln();
        }
        (inter_log_vol - Region::log_volume(other)).exp().min(1.0)
    }
}

// ─── Ball ────────────────────────────────────────────────────────────────────

/// Hypersphere defined by center and radius.
///
/// Used by ball embedding models (subsume's `Ball` type, RegD embeddings).
#[derive(Debug, Clone)]
pub struct Ball {
    center: Vec<f32>,
    radius: f32,
}

impl Ball {
    pub fn new(center: Vec<f32>, radius: f32) -> Self {
        assert!(radius >= 0.0, "radius must be non-negative");
        Self { center, radius }
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }
}

impl Region for Ball {
    fn dim(&self) -> usize {
        self.center.len()
    }

    fn center(&self) -> &[f32] {
        &self.center
    }

    fn distance_to_point(&self, point: &[f32]) -> f32 {
        debug_assert_eq!(point.len(), self.center.len(), "point dimension mismatch");
        crate::distance::ball_to_point_l2(&self.center, self.radius, point)
    }

    fn contains(&self, point: &[f32]) -> bool {
        debug_assert_eq!(point.len(), self.center.len(), "point dimension mismatch");
        let dist_sq: f32 = self
            .center
            .iter()
            .zip(point.iter())
            .map(|(c, p)| (c - p).powi(2))
            .sum();
        dist_sq <= self.radius * self.radius
    }

    fn bounding_ball(&self) -> (Vec<f32>, f32) {
        // A ball is its own bounding ball.
        (self.center.clone(), self.radius)
    }

    fn contains_region(&self, other: &Self) -> bool {
        // self ⊇ other iff ||c_self - c_other|| + r_other <= r_self.
        center_dist(&self.center, &other.center) + other.radius <= self.radius
    }

    fn overlaps_region(&self, other: &Self) -> bool {
        // Balls intersect iff their centers are within the sum of radii.
        center_dist(&self.center, &other.center) <= self.radius + other.radius
    }

    fn log_volume(&self) -> f32 {
        // ln vol of a d-ball: (d/2) ln(pi) + d ln(r) - ln(Gamma(d/2 + 1)).
        let d = self.center.len() as f64;
        let r = self.radius as f64;
        let lv = 0.5 * d * std::f64::consts::PI.ln() + d * r.ln() - lgamma(0.5 * d + 1.0);
        lv as f32
    }

    fn entailment_prob(&self, other: &Self) -> f32 {
        // Approximate (exact lens volume needs the regularized incomplete beta):
        // 1 if self contains other, 0 if disjoint, else a monotone interpolation.
        let cd = center_dist(&self.center, &other.center);
        if other.radius == 0.0 {
            return if cd <= self.radius { 1.0 } else { 0.0 };
        }
        if cd + other.radius <= self.radius {
            1.0
        } else if cd >= self.radius + other.radius {
            0.0
        } else {
            ((self.radius + other.radius - cd) / (2.0 * other.radius)).clamp(0.0, 1.0)
        }
    }
}

/// L2 distance between two region centers.
fn center_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// `ln(Gamma(x))` for `x >= 0.5` via the Lanczos approximation (g = 7).
fn lgamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let g = 7.0;
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + g + 0.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

// ─── Ellipsoid ─────────────────────────────────────────────────────────────

/// Axis-aligned ellipsoid: center `c` with per-axis semi-axes `a_i > 0`.
///
/// A point `p` is inside iff `sum_i ((p_i - c_i) / a_i)^2 <= 1`. The anisotropic
/// region type: an ellipsoid is a Gaussian/covariance concept whose extent
/// differs per dimension (a `Ball` is the special case `a_i = r`).
///
/// Point queries (`contains`, `distance_to_point`, `log_volume`) are exact;
/// the region-to-region predicates use the axis-aligned bounding box `[c - a,
/// c + a]` as a tractable approximation (exact ellipsoid-ellipsoid containment
/// and intersection volume have no closed form).
#[derive(Debug, Clone)]
pub struct Ellipsoid {
    center: Vec<f32>,
    semi_axes: Vec<f32>,
}

impl Ellipsoid {
    /// Create an ellipsoid from a center and per-axis semi-axes (all `> 0`).
    pub fn new(center: Vec<f32>, semi_axes: Vec<f32>) -> Self {
        assert_eq!(center.len(), semi_axes.len(), "center/semi-axes mismatch");
        assert!(
            semi_axes.iter().all(|&a| a > 0.0),
            "semi-axes must be positive"
        );
        Self { center, semi_axes }
    }

    pub fn semi_axes(&self) -> &[f32] {
        &self.semi_axes
    }

    /// The axis-aligned bounding box `[c - a, c + a]` (used by the region-to-region
    /// approximations).
    fn bbox(&self) -> AxisBox {
        let min = self
            .center
            .iter()
            .zip(self.semi_axes.iter())
            .map(|(c, a)| c - a)
            .collect();
        let max = self
            .center
            .iter()
            .zip(self.semi_axes.iter())
            .map(|(c, a)| c + a)
            .collect();
        AxisBox::new(min, max)
    }
}

impl Region for Ellipsoid {
    fn dim(&self) -> usize {
        self.center.len()
    }

    fn center(&self) -> &[f32] {
        &self.center
    }

    fn distance_to_point(&self, point: &[f32]) -> f32 {
        debug_assert_eq!(point.len(), self.center.len(), "point dimension mismatch");
        let d = self.center.len();
        let q: Vec<f64> = (0..d).map(|i| (point[i] - self.center[i]) as f64).collect();
        let a: Vec<f64> = self.semi_axes.iter().map(|&x| x as f64).collect();
        // Inside (or on the surface)?
        if (0..d).map(|i| (q[i] / a[i]).powi(2)).sum::<f64>() <= 1.0 {
            return 0.0;
        }
        // Nearest surface point solves x_i = a_i^2 q_i / (a_i^2 + lambda) with
        // lambda > 0 the root of f(lambda) = sum (a_i q_i / (a_i^2 + lambda))^2 - 1.
        // f is decreasing on lambda > 0 with f(0) > 0, so Newton from 0 converges up.
        let mut lambda = 0.0f64;
        for _ in 0..64 {
            let mut f = -1.0f64;
            let mut fp = 0.0f64;
            for i in 0..d {
                let denom = a[i] * a[i] + lambda;
                let t = a[i] * q[i] / denom;
                f += t * t;
                fp += -2.0 * a[i] * a[i] * q[i] * q[i] / denom.powi(3);
            }
            if f.abs() < 1e-9 || fp == 0.0 {
                break;
            }
            lambda -= f / fp;
            if lambda < 0.0 {
                lambda = 0.0;
            }
        }
        let dist_sq: f64 = (0..d)
            .map(|i| {
                let x = a[i] * a[i] * q[i] / (a[i] * a[i] + lambda);
                (q[i] - x).powi(2)
            })
            .sum();
        dist_sq.sqrt() as f32
    }

    fn contains(&self, point: &[f32]) -> bool {
        debug_assert_eq!(point.len(), self.center.len(), "point dimension mismatch");
        point
            .iter()
            .zip(self.center.iter())
            .zip(self.semi_axes.iter())
            .map(|((p, c), a)| ((p - c) / a).powi(2))
            .sum::<f32>()
            <= 1.0
    }

    fn bounding_ball(&self) -> (Vec<f32>, f32) {
        // The ellipsoid fits in a ball of its largest semi-axis.
        let radius = self.semi_axes.iter().copied().fold(0.0f32, f32::max);
        (self.center.clone(), radius)
    }

    fn contains_region(&self, other: &Self) -> bool {
        // Bounding-box approximation: self's box contains other's box.
        self.bbox().contains_region(&other.bbox())
    }

    fn overlaps_region(&self, other: &Self) -> bool {
        // Bounding-box approximation: the boxes intersect.
        self.bbox().overlaps_region(&other.bbox())
    }

    fn log_volume(&self) -> f32 {
        // ln vol = ln(unit d-ball vol) + sum ln(a_i).
        let d = self.semi_axes.len() as f64;
        let unit = 0.5 * d * std::f64::consts::PI.ln() - lgamma(0.5 * d + 1.0);
        let sum_ln_a: f64 = self.semi_axes.iter().map(|&a| (a as f64).ln()).sum();
        (unit + sum_ln_a) as f32
    }

    fn entailment_prob(&self, other: &Self) -> f32 {
        // Bounding-box approximation of vol(self ∩ other) / vol(other).
        self.bbox().entailment_prob(&other.bbox())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_overlap_predicate() {
        let a = AxisBox::new(vec![0.0, 0.0], vec![2.0, 2.0]);
        let touching = AxisBox::new(vec![1.0, 1.0], vec![3.0, 3.0]);
        let disjoint = AxisBox::new(vec![5.0, 5.0], vec![6.0, 6.0]);
        assert!(a.overlaps_region(&touching));
        assert!(touching.overlaps_region(&a)); // symmetric
        assert!(!a.overlaps_region(&disjoint));
        // Containment implies overlap.
        let inner = AxisBox::new(vec![0.5, 0.5], vec![1.0, 1.0]);
        assert!(a.overlaps_region(&inner));
    }

    #[test]
    fn box_log_volume_and_entailment() {
        let outer = AxisBox::new(vec![0.0, 0.0], vec![4.0, 4.0]); // area 16
        let inner = AxisBox::new(vec![1.0, 1.0], vec![3.0, 3.0]); // area 4
        assert!((Region::log_volume(&outer) - 16.0_f32.ln()).abs() < 1e-5);
        // outer fully contains inner -> P(outer ⊒ inner) = 1.
        assert!((outer.entailment_prob(&inner) - 1.0).abs() < 1e-5);
        // inner subsumes outer only fractionally: vol(inner∩outer)/vol(outer) = 4/16.
        assert!((inner.entailment_prob(&outer) - 0.25).abs() < 1e-5);
        // disjoint -> 0.
        let far = AxisBox::new(vec![10.0, 10.0], vec![11.0, 11.0]);
        assert_eq!(outer.entailment_prob(&far), 0.0);
    }

    #[test]
    fn ball_overlap_and_entailment() {
        let outer = Ball::new(vec![0.0, 0.0], 5.0);
        let inner = Ball::new(vec![1.0, 0.0], 1.0);
        let disjoint = Ball::new(vec![20.0, 0.0], 1.0);
        assert!(outer.overlaps_region(&inner));
        assert!(!outer.overlaps_region(&disjoint));
        assert_eq!(outer.entailment_prob(&inner), 1.0); // outer contains inner
        assert_eq!(outer.entailment_prob(&disjoint), 0.0); // disjoint
                                                           // Larger ball has larger log-volume.
        assert!(Region::log_volume(&outer) > Region::log_volume(&inner));
    }

    #[test]
    fn ellipsoid_contains_distance_volume() {
        // 2-d ellipsoid, semi-axes (2, 1) at the origin.
        let e = Ellipsoid::new(vec![0.0, 0.0], vec![2.0, 1.0]);
        assert!(e.contains(&[0.0, 0.0]));
        assert!(e.contains(&[1.9, 0.0]));
        assert!(!e.contains(&[2.1, 0.0]));
        assert_eq!(e.distance_to_point(&[0.5, 0.5]), 0.0); // inside
                                                           // Point (3, 0): nearest surface point is (2, 0), distance 1.
        assert!((e.distance_to_point(&[3.0, 0.0]) - 1.0).abs() < 1e-4);
        // Point (0, 3): nearest surface point is (0, 1), distance 2.
        assert!((e.distance_to_point(&[0.0, 3.0]) - 2.0).abs() < 1e-4);
        // Area = pi * a1 * a2 = 2*pi.
        assert!((Region::log_volume(&e) - (2.0 * std::f32::consts::PI).ln()).abs() < 1e-4);
        // A bigger ellipsoid subsumes a smaller concentric one (bbox approx).
        let big = Ellipsoid::new(vec![0.0, 0.0], vec![4.0, 4.0]);
        let small = Ellipsoid::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        assert!(big.contains_region(&small));
        assert!(!small.contains_region(&big));
    }

    #[test]
    fn ellipsoid_in_region_index() {
        use crate::{IndexParams, RegionIndex, SearchParams};
        let mut idx: RegionIndex<Ellipsoid> = RegionIndex::new(2, IndexParams::default()).unwrap();
        for i in 0..20u32 {
            let o = i as f32 * 3.0;
            idx.add(i, Ellipsoid::new(vec![o, o], vec![1.5, 0.8]))
                .unwrap();
        }
        idx.build().unwrap();
        // Nearest ellipsoid to a point inside ellipsoid 4 ([12,12]).
        let got = idx
            .search(&[12.0, 12.0], 1, SearchParams::default())
            .unwrap();
        assert_eq!(got[0].0, 4);
        assert_eq!(got[0].1, 0.0);
    }

    #[test]
    fn box_contains_interior_point() {
        let b = AxisBox::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        assert!(b.contains(&[0.5, 0.5]));
        assert!(b.contains(&[0.0, 0.0])); // boundary
        assert!(b.contains(&[1.0, 1.0])); // boundary
        assert!(!b.contains(&[1.5, 0.5]));
        assert!(!b.contains(&[-0.1, 0.5]));
    }

    #[test]
    fn box_distance_inside_is_zero() {
        let b = AxisBox::new(vec![0.0, 0.0], vec![2.0, 2.0]);
        assert_eq!(b.distance_to_point(&[1.0, 1.0]), 0.0);
    }

    #[test]
    fn box_distance_outside() {
        let b = AxisBox::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        // Point at (2, 0.5) -- distance is 1.0 (only x-axis contributes)
        let d = b.distance_to_point(&[2.0, 0.5]);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn box_distance_corner() {
        let b = AxisBox::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        // Point at (2, 2) -- distance is sqrt(2)
        let d = b.distance_to_point(&[2.0, 2.0]);
        assert!((d - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn box_from_center_offset() {
        let b = AxisBox::from_center_offset(vec![1.0, 1.0], vec![0.5, 0.5]);
        assert!((b.min()[0] - 0.5).abs() < 1e-6);
        assert!((b.max()[0] - 1.5).abs() < 1e-6);
        assert!(b.contains(&[1.0, 1.0]));
        assert!(!b.contains(&[2.0, 1.0]));
    }

    #[test]
    fn box_from_mu_delta() {
        // delta=0 => width = exp(0) = 1, so half-width = 0.5
        let b = AxisBox::from_mu_delta(vec![0.0, 0.0], vec![0.0, 0.0]);
        assert!((b.min()[0] - (-0.5)).abs() < 1e-6);
        assert!((b.max()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ball_contains_and_distance() {
        let ball = Ball::new(vec![0.0, 0.0, 0.0], 1.0);
        assert!(ball.contains(&[0.5, 0.5, 0.5])); // inside
        assert!(!ball.contains(&[1.0, 1.0, 0.0])); // outside (dist = sqrt(2))

        assert_eq!(ball.distance_to_point(&[0.0, 0.0, 0.0]), 0.0);
        // Point at (2, 0, 0): dist to surface = 2 - 1 = 1
        let d = ball.distance_to_point(&[2.0, 0.0, 0.0]);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ball_boundary_is_contained() {
        let ball = Ball::new(vec![0.0, 0.0], 1.0);
        assert!(ball.contains(&[1.0, 0.0]));
        assert!(ball.contains(&[0.0, 1.0]));
    }
}
