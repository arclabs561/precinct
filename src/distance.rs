/// L2 distance from a point to an axis-aligned box.
///
/// Returns 0.0 if the point is inside the box. Otherwise, returns the
/// Euclidean distance from the point to the nearest face/edge/corner.
///
/// Per-dimension: `max(0, lo - p) + max(0, p - hi)` gives the signed
/// overshoot. The L2 norm of these overshoots is the distance.
#[inline]
pub fn box_to_point_l2(min: &[f32], max: &[f32], point: &[f32]) -> f32 {
    let dist_sq: f32 = min
        .iter()
        .zip(max.iter())
        .zip(point.iter())
        .map(|((lo, hi), p)| {
            let below = lo - p;
            let above = p - hi;
            let gap = below.max(above).max(0.0);
            gap * gap
        })
        .sum();
    dist_sq.sqrt()
}

/// L2 distance from a point to a ball (hypersphere) surface.
///
/// Returns 0.0 if the point is inside the ball.
#[inline]
pub fn ball_to_point_l2(center: &[f32], radius: f32, point: &[f32]) -> f32 {
    let dist_sq: f32 = center
        .iter()
        .zip(point.iter())
        .map(|(c, p)| (c - p).powi(2))
        .sum();
    let dist = dist_sq.sqrt();
    (dist - radius).max(0.0)
}

/// Query2Box-style distance from a point to an axis-aligned box.
///
/// Unlike [`box_to_point_l2`] which is zero inside, this function also
/// scores points *inside* the box -- entities closer to the center score
/// lower (better). This is the distance function from Ren et al. (ICLR 2020):
///
/// ```text
/// d(e, q) = ||dist_outside||_1 + alpha * ||dist_inside||_1
/// ```
///
/// `alpha` is typically 0.02 -- inside-box entities are strongly preferred
/// but still rank-ordered by center proximity.
#[inline]
pub fn query2box_distance(min: &[f32], max: &[f32], point: &[f32], alpha: f32) -> f32 {
    let mut dist_outside = 0.0f32;
    let mut dist_inside = 0.0f32;

    let center_iter = min.iter().zip(max.iter()).map(|(lo, hi)| (lo + hi) * 0.5);

    for (((lo, hi), p), c) in min
        .iter()
        .zip(max.iter())
        .zip(point.iter())
        .zip(center_iter)
    {
        let below = lo - p;
        let above = p - hi;
        if below > 0.0 || above > 0.0 {
            dist_outside += below.max(above).max(0.0);
        } else {
            dist_inside += (p - c).abs();
        }
    }

    dist_outside + alpha * dist_inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_l2_inside() {
        assert_eq!(box_to_point_l2(&[0.0, 0.0], &[1.0, 1.0], &[0.5, 0.5]), 0.0);
    }

    #[test]
    fn box_l2_one_axis() {
        let d = box_to_point_l2(&[0.0, 0.0], &[1.0, 1.0], &[3.0, 0.5]);
        assert!((d - 2.0).abs() < 1e-6);
    }

    #[test]
    fn ball_l2_inside() {
        assert_eq!(ball_to_point_l2(&[0.0, 0.0], 1.0, &[0.5, 0.0]), 0.0);
    }

    #[test]
    fn ball_l2_outside() {
        let d = ball_to_point_l2(&[0.0, 0.0], 1.0, &[3.0, 0.0]);
        assert!((d - 2.0).abs() < 1e-6);
    }

    #[test]
    fn query2box_inside_nonzero() {
        // Point inside the box should have nonzero distance (center penalty)
        let d = query2box_distance(&[0.0, 0.0], &[2.0, 2.0], &[0.5, 0.5], 0.02);
        assert!(d > 0.0); // inside but not at center
        assert!(d < 0.1); // but very small due to alpha=0.02
    }

    #[test]
    fn query2box_center_is_zero() {
        let d = query2box_distance(&[0.0, 0.0], &[2.0, 2.0], &[1.0, 1.0], 0.02);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn query2box_outside_dominates() {
        let d = query2box_distance(&[0.0, 0.0], &[1.0, 1.0], &[3.0, 3.0], 0.02);
        // dist_outside = (3-1) + (3-1) = 4.0, dist_inside = 0
        assert!((d - 4.0).abs() < 1e-6);
    }
}
