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
}

// ─── AxisBox ─────────────────────────────────────────────────────────────────

/// Axis-aligned hyperrectangle defined by min/max corners.
///
/// The standard representation for box embeddings (Query2Box, Box2EL,
/// BoxTaxo). Each dimension `i` spans `[min[i], max[i]]`.
#[derive(Debug, Clone)]
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
