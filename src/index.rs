use std::collections::HashMap;

use crate::Region;
use vicinity::hnsw::HNSWIndex;

/// Error type for region index operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("index must be built before search")]
    NotBuilt,
    #[error("vicinity: {0}")]
    Vicinity(#[from] vicinity::RetrieveError),
}

/// ANN index over region embeddings.
///
/// Indexes region centers in an HNSW graph and reranks candidates using the
/// true point-to-region distance. This is the "flatten to center + rerank"
/// strategy: fast approximate recall via point ANN, exact region scoring on
/// the candidate set.
///
/// # Type parameter
///
/// `R` is the region type ([`AxisBox`](crate::AxisBox), [`Ball`](crate::Ball),
/// or any custom [`Region`] implementation).
///
/// # Example
///
/// ```no_run
/// use precinct::{AxisBox, RegionIndex};
///
/// let mut idx = RegionIndex::new(2, Default::default()).unwrap();
/// idx.add(0, AxisBox::new(vec![0.0, 0.0], vec![1.0, 1.0]));
/// idx.add(1, AxisBox::new(vec![5.0, 5.0], vec![6.0, 6.0]));
/// idx.build().unwrap();
///
/// let results = idx.search(&[0.5, 0.5], 1, Default::default()).unwrap();
/// assert_eq!(results[0].0, 0);
/// ```
pub struct RegionIndex<R: Region> {
    hnsw: HNSWIndex,
    regions: Vec<R>,
    /// Maps external doc_id -> index into `regions`.
    id_to_pos: HashMap<u32, usize>,
    built: bool,
}

/// Parameters for building the region index.
pub struct IndexParams {
    /// HNSW `m` parameter (number of bi-directional links per node).
    pub m: usize,
    /// HNSW `m_max` parameter (max links on non-zero layers).
    pub m_max: usize,
    /// HNSW `ef_construction` parameter.
    pub ef_construction: usize,
}

impl Default for IndexParams {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: 32,
            ef_construction: 200,
        }
    }
}

/// Parameters for search.
pub struct SearchParams {
    /// HNSW `ef` search parameter (beam width).
    pub ef: usize,
    /// Over-retrieval factor. Retrieves `k * overretrieve` candidates from
    /// the point index, then reranks with true region distance and returns
    /// the top `k`.
    pub overretrieve: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            ef: 200,
            overretrieve: 10,
        }
    }
}

/// Search result: (region_id, distance).
pub type SearchResult = (u32, f32);

impl<R: Region> RegionIndex<R> {
    /// Create a new region index for the given dimensionality.
    pub fn new(dim: usize, params: IndexParams) -> Result<Self, Error> {
        let hnsw = HNSWIndex::builder(dim)
            .m(params.m)
            .m_max(params.m_max)
            .ef_construction(params.ef_construction)
            .metric(vicinity::DistanceMetric::L2)
            .build()?;

        Ok(Self {
            hnsw,
            regions: Vec::new(),
            id_to_pos: HashMap::new(),
            built: false,
        })
    }

    /// Add a region to the index.
    ///
    /// The region's center is indexed for ANN retrieval. The full region
    /// geometry is stored for reranking.
    pub fn add(&mut self, id: u32, region: R) {
        let center = region.center().to_vec();
        self.hnsw
            .add(id, center)
            .expect("failed to add center to HNSW");
        let pos = self.regions.len();
        self.regions.push(region);
        self.id_to_pos.insert(id, pos);
        self.built = false;
    }

    /// Build the underlying HNSW graph. Must be called before search.
    pub fn build(&mut self) -> Result<(), Error> {
        self.hnsw.build()?;
        self.built = true;
        Ok(())
    }

    /// Search for the `k` nearest regions to `query`.
    ///
    /// Retrieves `k * overretrieve` candidates from the center-based HNSW
    /// index, then reranks each candidate using the true point-to-region
    /// distance from [`Region::distance_to_point`].
    #[must_use = "search results are not used"]
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        params: SearchParams,
    ) -> Result<Vec<SearchResult>, Error> {
        if !self.built {
            return Err(Error::NotBuilt);
        }

        let fetch_k = k.saturating_mul(params.overretrieve).max(k);

        // Phase 1: center-based ANN retrieval
        let candidates = self.hnsw.search(query, fetch_k, params.ef)?;

        // Phase 2: rerank with true region distance
        let mut reranked: Vec<SearchResult> = candidates
            .into_iter()
            .map(|(doc_id, _center_dist)| {
                let region = &self.regions[self.id_to_pos[&doc_id]];
                let true_dist = region.distance_to_point(query);
                (doc_id, true_dist)
            })
            .collect();

        reranked
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);
        Ok(reranked)
    }

    /// Find all regions that contain `point`.
    ///
    /// This is a stabbing query. In high dimensions there is no known
    /// efficient index structure, so this retrieves a large candidate set
    /// from the center-based index and filters by containment.
    ///
    /// For exhaustive containment (guaranteed recall), use
    /// [`containing_exhaustive`](Self::containing_exhaustive).
    pub fn containing(&self, point: &[f32], params: SearchParams) -> Result<Vec<u32>, Error> {
        if !self.built {
            return Err(Error::NotBuilt);
        }

        let fetch_k = self.regions.len().min(params.ef * params.overretrieve);

        let candidates = self.hnsw.search(point, fetch_k, params.ef)?;

        let result: Vec<u32> = candidates
            .into_iter()
            .filter(|(doc_id, _)| self.regions[self.id_to_pos[doc_id]].contains(point))
            .map(|(doc_id, _)| doc_id)
            .collect();

        Ok(result)
    }

    /// Exhaustive containment query -- checks every region.
    ///
    /// Guaranteed recall but O(n) in the number of regions.
    pub fn containing_exhaustive(&self, point: &[f32]) -> Vec<u32> {
        self.id_to_pos
            .iter()
            .filter(|(_, &pos)| self.regions[pos].contains(point))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Exhaustive nearest-region search. O(n) but guaranteed correct ranking.
    ///
    /// Useful as ground truth for measuring recall of the ANN-based search.
    pub fn search_exhaustive(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self
            .id_to_pos
            .iter()
            .map(|(&id, &pos)| (id, self.regions[pos].distance_to_point(query)))
            .collect();
        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Search using a custom distance function during graph traversal.
    ///
    /// The closure `dist_fn(query, internal_id)` is called for every distance
    /// computation during beam search. The `internal_id` is the zero-based
    /// insertion order (i.e., the Nth region added has internal_id N).
    ///
    /// Note: for region distance (box-to-point), this typically produces
    /// *worse* recall than [`search`](Self::search) because the graph was
    /// built for center-to-center L2, not for region distance. Use this
    /// for custom metrics that are monotonically related to L2 (e.g.,
    /// quantized distance approximations).
    pub fn search_with_distance(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        dist_fn: &dyn Fn(&[f32], u32) -> f32,
    ) -> Result<Vec<SearchResult>, Error> {
        if !self.built {
            return Err(Error::NotBuilt);
        }

        Ok(self.hnsw.search_with_distance(query, k, ef, dist_fn)?)
    }

    /// Get a region by its external ID.
    pub fn get(&self, id: u32) -> Option<&R> {
        self.id_to_pos.get(&id).map(|&pos| &self.regions[pos])
    }

    /// Number of indexed regions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AxisBox;

    /// Build a test index with enough nodes for stable HNSW behavior.
    /// 20 boxes along the diagonal in 3d, each 1x1x1.
    fn build_test_index() -> RegionIndex<AxisBox> {
        let mut idx = RegionIndex::new(3, Default::default()).unwrap();
        for i in 0..20 {
            let o = i as f32 * 2.0; // spacing > box width avoids overlap
            idx.add(
                i,
                AxisBox::new(vec![o, o, o], vec![o + 1.0, o + 1.0, o + 1.0]),
            );
        }
        idx.build().unwrap();
        idx
    }

    #[test]
    fn search_finds_nearest_box() {
        let idx = build_test_index();
        // Query inside box 0 ([0,0,0]-[1,1,1])
        let results = idx.search(&[0.5, 0.5, 0.5], 1, Default::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[0].1, 0.0);
    }

    #[test]
    fn search_reranks_correctly() {
        let idx = build_test_index();
        // Query inside box 5 ([10,10,10]-[11,11,11])
        let query = [10.5, 10.5, 10.5];
        let results = idx
            .search(
                &query,
                5,
                SearchParams {
                    ef: 100,
                    overretrieve: 10,
                },
            )
            .unwrap();

        assert_eq!(results[0].0, 5);
        assert_eq!(results[0].1, 0.0);

        // Results must be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted: {:?}", results);
        }
    }

    #[test]
    fn search_with_custom_distance() {
        let idx = build_test_index();

        // Use the same L2 metric that the graph was built with --
        // monotonically related distances give reliable graph traversal.
        let dist_fn = |q: &[f32], internal_id: u32| -> f32 {
            let o = internal_id as f32 * 2.0;
            let center = [o + 0.5, o + 0.5, o + 0.5];
            center
                .iter()
                .zip(q)
                .map(|(c, p)| (c - p).powi(2))
                .sum::<f32>()
                .sqrt()
        };

        // Query at box 3's center: custom distance should return sorted results
        let results = idx
            .search_with_distance(&[6.5, 6.5, 6.5], 3, 200, &dist_fn)
            .unwrap();
        assert_eq!(results.len(), 3);
        // Results must be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted: {:?}", results);
        }
        // Nearest result should have distance < 2.0 (within 1 box width of query)
        assert!(
            results[0].1 < 2.0,
            "closest result too far: {}",
            results[0].1
        );
    }

    #[test]
    fn containing_exhaustive_finds_enclosing_boxes() {
        // Separate index for containment: needs overlapping boxes
        let mut idx = RegionIndex::new(2, Default::default()).unwrap();
        idx.add(0, AxisBox::new(vec![0.0, 0.0], vec![10.0, 10.0])); // big
        idx.add(1, AxisBox::new(vec![4.0, 4.0], vec![6.0, 6.0])); // small, inside big
        idx.add(2, AxisBox::new(vec![20.0, 20.0], vec![21.0, 21.0])); // far
                                                                      // Pad to avoid degenerate 3-node HNSW
        for i in 3..15 {
            let o = (i as f32) * 3.0;
            idx.add(i, AxisBox::new(vec![o, o], vec![o + 0.5, o + 0.5]));
        }
        idx.build().unwrap();

        let result = idx.containing_exhaustive(&[5.0, 5.0]);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
        assert!(!result.contains(&2));
    }

    #[test]
    fn get_returns_region() {
        let mut idx = RegionIndex::new(2, Default::default()).unwrap();
        idx.add(42, AxisBox::new(vec![0.0, 0.0], vec![1.0, 1.0]));
        idx.build().unwrap();

        assert!(idx.get(42).is_some());
        assert!(idx.get(99).is_none());
    }

    #[test]
    fn error_on_search_before_build() {
        let idx: RegionIndex<AxisBox> = RegionIndex::new(2, Default::default()).unwrap();
        assert!(idx.search(&[0.0, 0.0], 1, Default::default()).is_err());
    }
}
