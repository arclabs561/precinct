//! Approximate nearest-neighbor search over region embeddings (boxes, balls).
//!
//! Point-based ANN indices (HNSW, IVF, Vamana) assume queries and database
//! entries are single vectors. Region embeddings -- axis-aligned boxes, balls,
//! cones -- represent concepts as *volumes* in embedding space. precinct bridges
//! this gap.
//!
//! # Core abstractions
//!
//! - [`Region`] -- trait for geometric regions with center, point-to-region
//!   distance, and containment.
//! - [`AxisBox`], [`Ball`] -- concrete region types.
//! - [`RegionIndex`] -- ANN index over regions. Builds a point index over
//!   region centers, retrieves candidates, reranks with true region distance.
//!
//! # Usage
//!
//! ```
//! use precinct::{AxisBox, Region};
//!
//! let b = AxisBox::new(vec![0.0, 0.0], vec![1.0, 1.0]);
//! assert!(b.contains(&[0.5, 0.5]));
//! assert!(!b.contains(&[1.5, 0.5]));
//! assert_eq!(b.distance_to_point(&[0.5, 0.5]), 0.0);
//! ```

mod region;
pub use region::{AxisBox, Ball, Ellipsoid, Region};

mod distance;
pub use distance::{ball_to_point_l2, box_to_point_l2, query2box_distance};

pub mod io;

#[cfg(feature = "index")]
pub mod index;
#[cfg(feature = "index")]
pub use index::{Error, IndexParams, RegionIndex, SearchParams};

/// Updatable, durable multi-segment region-ANN index via segstore (`store` feature).
#[cfg(feature = "store")]
pub mod store;
