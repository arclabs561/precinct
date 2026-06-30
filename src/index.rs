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
    #[cfg(feature = "store")]
    #[error("encode: {0}")]
    Encode(String),
    #[cfg(feature = "store")]
    #[error("decode: {0}")]
    Decode(String),
}

/// ANN index over region embeddings.
///
/// Each region is lifted to a *power-distance* vector and indexed in an HNSW
/// graph, so the candidate set is ranked by a metric that respects region
/// *extent*, not just center proximity; candidates are then reranked by the true
/// point-to-region distance. The lift (see the design note) is what lets a large
/// general concept be retrieved for a query it encloses even when its center is
/// far away, the case a plain center-ANN misses.
///
/// The index answers the region query algebra:
/// - similarity: [`search`](Self::search) (k nearest regions to a point) and
///   [`nearest_region`](Self::nearest_region) (k regions nearest a query region),
/// - membership: [`containing`](Self::containing) (regions enclosing a point),
/// - subsumption: [`subsumers`](Self::subsumers) / [`subsumees`](Self::subsumees)
///   (regions that contain / are contained by a query region),
/// - overlap: [`overlapping`](Self::overlapping) (regions intersecting a query
///   region, the conjunction primitive).
///
/// Retrieved regions can be scored with [`Region::log_volume`] (generality) and
/// [`Region::entailment_prob`] (soft subsumption probability).
///
/// # Type parameter
///
/// `R` is the region type ([`AxisBox`](crate::AxisBox), [`Ball`](crate::Ball),
/// or any custom [`Region`] implementation in the Euclidean family).
///
/// # Example
///
/// ```no_run
/// use precinct::{AxisBox, RegionIndex};
///
/// let mut idx = RegionIndex::new(2, Default::default()).unwrap();
/// idx.add(0, AxisBox::new(vec![0.0, 0.0], vec![1.0, 1.0])).unwrap();
/// idx.add(1, AxisBox::new(vec![5.0, 5.0], vec![6.0, 6.0])).unwrap();
/// idx.build().unwrap();
///
/// let results = idx.search(&[0.5, 0.5], 1, Default::default()).unwrap();
/// assert_eq!(results[0].0, 0);
/// ```
pub struct RegionIndex<R: Region> {
    /// HNSW over region centers (`dim`). Serves the nearest-region query, where
    /// center proximity is the right candidate signal.
    center: HNSWIndex,
    /// HNSW over the lifted `dim + 2` power-distance vectors. Serves the
    /// membership / subsumption queries, where extent (not center) decides
    /// enclosure.
    lift: HNSWIndex,
    /// Embedding dimension of the regions.
    dim: usize,
    regions: Vec<R>,
    /// Insertion-order external ids; `ids[pos]` is the id of `regions[pos]`.
    ids: Vec<u32>,
    /// Maps external doc_id -> index into `regions`.
    id_to_pos: HashMap<u32, usize>,
    built: bool,
}

#[cfg(feature = "store")]
#[derive(serde::Serialize, serde::Deserialize)]
struct RegionIndexSnapshot<R> {
    dim: usize,
    center: Vec<u8>,
    lift: Vec<u8>,
    regions: Vec<R>,
    ids: Vec<u32>,
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
    /// Over-retrieval factor. Retrieves `k * overretrieve` candidates from the
    /// lifted index, then reranks with true region distance and returns the top
    /// `k`.
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
    /// Create a new region index for the given embedding dimensionality.
    pub fn new(dim: usize, params: IndexParams) -> Result<Self, Error> {
        let builder = |d: usize| {
            HNSWIndex::builder(d)
                .m(params.m)
                .m_max(params.m_max)
                .ef_construction(params.ef_construction)
                .metric(vicinity::DistanceMetric::L2)
                .build()
        };
        // Center index over `dim`; lift index over `dim + 2` (d+1 power-distance
        // MIPS form, +1 for the MIPS->L2 reduction).
        let center = builder(dim)?;
        let lift = builder(dim + 2)?;

        Ok(Self {
            center,
            lift,
            dim,
            regions: Vec::new(),
            ids: Vec::new(),
            id_to_pos: HashMap::new(),
            built: false,
        })
    }

    /// Add a region to the index.
    ///
    /// The region is buffered; the power-distance lift and HNSW insertion happen
    /// in [`build`](Self::build), because the lift's normalization constant needs
    /// every region's bounding ball first.
    pub fn add(&mut self, id: u32, region: R) -> Result<(), Error> {
        let pos = self.regions.len();
        self.regions.push(region);
        self.ids.push(id);
        self.id_to_pos.insert(id, pos);
        self.built = false;
        Ok(())
    }

    /// Build the underlying HNSW graph. Must be called before any query.
    ///
    /// Lifts every region to its `(d + 2)` power-distance vector and inserts it,
    /// then builds the graph.
    pub fn build(&mut self) -> Result<(), Error> {
        // Power-distance lift of each bounding ball: u = (2c, r^2 - ||c||^2).
        let lifted: Vec<Vec<f32>> = self
            .regions
            .iter()
            .map(|r| {
                let (c, radius) = r.bounding_ball();
                lift_region(&c, radius)
            })
            .collect();

        // MIPS->L2 reduction needs M = max ||u||, so ||u'||^2 = M^2 is constant.
        let m_sq = lifted
            .iter()
            .map(|u| u.iter().map(|x| x * x).sum::<f32>())
            .fold(0.0f32, f32::max);

        for (pos, region) in self.regions.iter().enumerate() {
            self.center.add(self.ids[pos], region.center().to_vec())?;

            let u = &lifted[pos];
            let norm_sq: f32 = u.iter().map(|x| x * x).sum();
            let aug = (m_sq - norm_sq).max(0.0).sqrt();
            let mut v = u.clone();
            v.push(aug);
            self.lift.add(self.ids[pos], v)?;
        }

        self.center.build()?;
        self.lift.build()?;
        self.built = true;
        Ok(())
    }

    #[cfg(feature = "store")]
    pub(crate) fn to_postcard(&self) -> Result<Vec<u8>, Error>
    where
        R: serde::Serialize + Clone,
    {
        let snapshot = RegionIndexSnapshot {
            dim: self.dim,
            center: self.center.to_postcard()?,
            lift: self.lift.to_postcard()?,
            regions: self.regions.clone(),
            ids: self.ids.clone(),
        };
        postcard::to_allocvec(&snapshot).map_err(|e| Error::Encode(e.to_string()))
    }

    #[cfg(feature = "store")]
    pub(crate) fn from_postcard(bytes: &[u8]) -> Result<Self, Error>
    where
        R: serde::de::DeserializeOwned,
    {
        let snapshot: RegionIndexSnapshot<R> =
            postcard::from_bytes(bytes).map_err(|e| Error::Decode(e.to_string()))?;
        if snapshot.regions.len() != snapshot.ids.len() {
            return Err(Error::Decode(
                "region and id counts differ in region-index snapshot".into(),
            ));
        }
        let mut id_to_pos = HashMap::with_capacity(snapshot.ids.len());
        for (pos, id) in snapshot.ids.iter().copied().enumerate() {
            if id_to_pos.insert(id, pos).is_some() {
                return Err(Error::Decode(format!(
                    "duplicate region id {id} in region-index snapshot"
                )));
            }
        }
        Ok(Self {
            center: HNSWIndex::from_postcard(&snapshot.center)?,
            lift: HNSWIndex::from_postcard(&snapshot.lift)?,
            dim: snapshot.dim,
            regions: snapshot.regions,
            ids: snapshot.ids,
            id_to_pos,
            built: true,
        })
    }

    #[cfg(feature = "store")]
    pub(crate) fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// The lifted `(d + 2)` query vector `(p, 1, 0)`: the power-distance MIPS
    /// query, with a zero in the MIPS->L2 reduction coordinate.
    fn lift_query(&self, point: &[f32]) -> Vec<f32> {
        let mut q = Vec::with_capacity(self.dim + 2);
        q.extend_from_slice(point);
        q.push(1.0);
        q.push(0.0);
        q
    }

    /// Search for the `k` nearest regions to `query` by point-to-region distance.
    ///
    /// Retrieves `k * overretrieve` candidates from the lifted power-distance
    /// index, then reranks each by the true [`Region::distance_to_point`].
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
        // Nearest-region uses the center index: center proximity is the right
        // candidate signal, and the rerank fixes the surface-vs-center gap.
        let candidates = self.center.search(query, fetch_k, params.ef)?;

        let mut reranked: Vec<SearchResult> = candidates
            .into_iter()
            .map(|(doc_id, _)| {
                let region = &self.regions[self.id_to_pos[&doc_id]];
                (doc_id, region.distance_to_point(query))
            })
            .collect();

        reranked
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);
        Ok(reranked)
    }

    /// Find all indexed regions that contain `point` (the membership query).
    ///
    /// Generates candidates from the lifted index, whose power-distance ranking
    /// surfaces an enclosing region even when its center is far from `point`,
    /// then filters by [`Region::contains`]. Recall is bounded by `overretrieve`;
    /// for a guarantee use [`containing_exhaustive`](Self::containing_exhaustive).
    pub fn containing(&self, point: &[f32], params: SearchParams) -> Result<Vec<u32>, Error> {
        if !self.built {
            return Err(Error::NotBuilt);
        }

        let fetch_k = self
            .regions
            .len()
            .min(params.ef.saturating_mul(params.overretrieve).max(1));
        let lifted = self.lift_query(point);
        let candidates = self.lift.search(&lifted, fetch_k, params.ef)?;

        Ok(candidates
            .into_iter()
            .filter(|(doc_id, _)| self.regions[self.id_to_pos[doc_id]].contains(point))
            .map(|(doc_id, _)| doc_id)
            .collect())
    }

    /// Regions that subsume (fully contain) `query` (`S ⊇ query`).
    ///
    /// A subsumer must contain `query`'s center, so candidates come from
    /// `containing(query.center())`, filtered by the region-to-region
    /// [`Region::contains_region`] predicate.
    pub fn subsumers(&self, query: &R, params: SearchParams) -> Result<Vec<u32>, Error> {
        let candidates = self.containing(query.center(), params)?;
        Ok(candidates
            .into_iter()
            .filter(|id| self.regions[self.id_to_pos[id]].contains_region(query))
            .collect())
    }

    /// Regions that *softly* subsume `query`: `entailment_prob(query) >= min_prob`,
    /// returned with their probability, highest first.
    ///
    /// Trained region embeddings (Gumbel boxes, etc.) nest *softly* -- a child's
    /// center falls inside its parent but the child's full region pokes outside --
    /// so strict [`subsumers`](Self::subsumers) misses real is-a ancestors that
    /// the soft form recovers (`min_prob = 1.0` reduces to strict containment).
    /// Candidates come from `containing(query.center())`; the score is the
    /// box-lattice conditional `vol(S ∩ query) / vol(query)`.
    pub fn subsumers_soft(
        &self,
        query: &R,
        min_prob: f32,
        params: SearchParams,
    ) -> Result<Vec<SearchResult>, Error> {
        let candidates = self.containing(query.center(), params)?;
        let mut out: Vec<SearchResult> = candidates
            .into_iter()
            .filter_map(|id| {
                let p = self.regions[self.id_to_pos[&id]].entailment_prob(query);
                (p >= min_prob).then_some((id, p))
            })
            .collect();
        out.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out)
    }

    /// Regions subsumed by (fully contained in) `query` (`T ⊆ query`).
    ///
    /// This is the "region-centers inside `query`" direction, which has no clean
    /// lift, so it is exhaustive (`O(n)`): every region is tested with
    /// `query.contains_region`.
    pub fn subsumees(&self, query: &R) -> Vec<u32> {
        self.ids
            .iter()
            .enumerate()
            .filter(|(pos, _)| query.contains_region(&self.regions[*pos]))
            .map(|(_, &id)| id)
            .collect()
    }

    /// Regions that intersect `query` (`S ∩ query ≠ ∅`), the overlap query.
    ///
    /// Candidates come from the regions nearest `query`'s center plus those that
    /// enclose it, filtered by [`Region::overlaps_region`]. Approximate (recall
    /// bounded by `overretrieve`); for a guarantee use
    /// [`overlapping_exhaustive`](Self::overlapping_exhaustive).
    pub fn overlapping(&self, query: &R, params: SearchParams) -> Result<Vec<u32>, Error> {
        if !self.built {
            return Err(Error::NotBuilt);
        }
        let SearchParams { ef, overretrieve } = params;
        let fetch_k = self
            .regions
            .len()
            .min(ef.saturating_mul(overretrieve).max(1));
        let mut cand: std::collections::HashSet<u32> = self
            .center
            .search(query.center(), fetch_k, ef)?
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        // Large regions that overlap may not be center-near; catch the ones that
        // enclose the query's center via the lift.
        cand.extend(self.containing(query.center(), SearchParams { ef, overretrieve })?);
        Ok(cand
            .into_iter()
            .filter(|id| self.regions[self.id_to_pos[id]].overlaps_region(query))
            .collect())
    }

    /// Exhaustive overlap query -- checks every region. `O(n)`, guaranteed.
    pub fn overlapping_exhaustive(&self, query: &R) -> Vec<u32> {
        self.ids
            .iter()
            .enumerate()
            .filter(|(pos, _)| query.overlaps_region(&self.regions[*pos]))
            .map(|(_, &id)| id)
            .collect()
    }

    /// The `k` regions most similar to `query`, by center distance.
    ///
    /// Region-to-region nearest neighbor (a concept's nearest concepts), via the
    /// center index reranked by center L2. Returns `(id, center_distance)`.
    pub fn nearest_region(
        &self,
        query: &R,
        k: usize,
        params: SearchParams,
    ) -> Result<Vec<SearchResult>, Error> {
        if !self.built {
            return Err(Error::NotBuilt);
        }
        let fetch_k = k.saturating_mul(params.overretrieve).max(k);
        let candidates = self.center.search(query.center(), fetch_k, params.ef)?;
        let mut reranked: Vec<SearchResult> = candidates
            .into_iter()
            .map(|(id, _)| {
                let r = &self.regions[self.id_to_pos[&id]];
                (id, l2(query.center(), r.center()))
            })
            .collect();
        reranked
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);
        Ok(reranked)
    }

    /// Exhaustive containment query -- checks every region. `O(n)`, guaranteed.
    pub fn containing_exhaustive(&self, point: &[f32]) -> Vec<u32> {
        self.id_to_pos
            .iter()
            .filter(|(_, &pos)| self.regions[pos].contains(point))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Exhaustive subsumer query -- checks every region. `O(n)`, guaranteed.
    pub fn subsumers_exhaustive(&self, query: &R) -> Vec<u32> {
        self.ids
            .iter()
            .enumerate()
            .filter(|(pos, _)| self.regions[*pos].contains_region(query))
            .map(|(_, &id)| id)
            .collect()
    }

    /// Exhaustive nearest-region search. `O(n)` but guaranteed-correct ranking.
    ///
    /// Useful as ground truth for measuring recall of [`search`](Self::search).
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
    /// computation during beam search over the center graph. `internal_id` is the
    /// zero-based insertion order (the Nth region added has `internal_id` N).
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
        Ok(self
            .center
            .search_with_distance(query, k, ef, &|q, id| dist_fn(q, id))?)
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

/// Power-distance lift of a ball `(c, r)`: `u = (2c, r^2 - ||c||^2)` in `R^(d+1)`,
/// so `u · (p, 1) = 2 p·c + r^2 - ||c||^2` and `argmax_u` is the min power
/// distance.
/// L2 distance between two equal-length vectors.
fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn lift_region(center: &[f32], radius: f32) -> Vec<f32> {
    let mut u = Vec::with_capacity(center.len() + 1);
    let mut norm_c_sq = 0.0f32;
    for &ci in center {
        u.push(2.0 * ci);
        norm_c_sq += ci * ci;
    }
    u.push(radius * radius - norm_c_sq);
    u
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AxisBox, Ball};

    /// Build a test index with enough nodes for stable HNSW behavior.
    /// 20 boxes along the diagonal in 3d, each 1x1x1.
    fn build_test_index() -> RegionIndex<AxisBox> {
        let mut idx = RegionIndex::new(3, Default::default()).unwrap();
        for i in 0..20 {
            let o = i as f32 * 2.0; // spacing > box width avoids overlap
            idx.add(
                i,
                AxisBox::new(vec![o, o, o], vec![o + 1.0, o + 1.0, o + 1.0]),
            )
            .unwrap();
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

        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted: {:?}", results);
        }
    }

    #[test]
    fn search_with_custom_distance() {
        let idx = build_test_index();

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

        let results = idx
            .search_with_distance(&[6.5, 6.5, 6.5], 3, 200, &dist_fn)
            .unwrap();
        assert_eq!(results.len(), 3);
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted: {:?}", results);
        }
        assert!(
            results[0].1 < 2.0,
            "closest result too far: {}",
            results[0].1
        );
    }

    #[test]
    fn containing_finds_far_centered_enclosing_box() {
        // The far-centered-enclosure case the lift is built to fix: a huge box
        // whose center is far from the query but which still encloses it. A
        // center-ANN candidate generator would not surface box 0; the
        // power-distance lift must.
        let mut idx = RegionIndex::new(2, Default::default()).unwrap();
        // Box 0: huge, center at (50, 0.5), encloses the query at (1, 0.5).
        idx.add(0, AxisBox::new(vec![0.0, 0.0], vec![100.0, 1.0]))
            .unwrap();
        // Many small decoy boxes with centers near the query, none enclosing it.
        for i in 1..60u32 {
            let x = i as f32 * 0.05;
            idx.add(i, AxisBox::new(vec![x, 2.0], vec![x + 0.1, 2.1]))
                .unwrap();
        }
        idx.build().unwrap();

        let params = SearchParams {
            ef: 64,
            overretrieve: 4,
        };
        let got = idx.containing(&[1.0, 0.5], params).unwrap();
        assert!(
            got.contains(&0),
            "lift must surface the far-centered enclosing box; got {got:?}"
        );
        // And it agrees with the exhaustive ground truth.
        assert!(idx.containing_exhaustive(&[1.0, 0.5]).contains(&0));
    }

    #[test]
    fn subsumers_and_subsumees_nested_boxes() {
        // Nested boxes: 0 ⊇ 1 ⊇ 2, plus a disjoint box 3.
        let mut idx = RegionIndex::new(2, Default::default()).unwrap();
        idx.add(0, AxisBox::new(vec![0.0, 0.0], vec![10.0, 10.0]))
            .unwrap();
        idx.add(1, AxisBox::new(vec![2.0, 2.0], vec![8.0, 8.0]))
            .unwrap();
        idx.add(2, AxisBox::new(vec![4.0, 4.0], vec![6.0, 6.0]))
            .unwrap();
        idx.add(3, AxisBox::new(vec![20.0, 20.0], vec![21.0, 21.0]))
            .unwrap();
        for i in 4..20u32 {
            let o = i as f32 * 3.0;
            idx.add(i, AxisBox::new(vec![o, o], vec![o + 0.5, o + 0.5]))
                .unwrap();
        }
        idx.build().unwrap();

        // Subsumers of box 1 ([2,2]-[8,8]) = box 0 (and box 1 itself, which
        // contains itself); box 2 and 3 do not.
        let middle = AxisBox::new(vec![2.0, 2.0], vec![8.0, 8.0]);
        let mut subs = idx
            .subsumers(
                &middle,
                SearchParams {
                    ef: 64,
                    overretrieve: 8,
                },
            )
            .unwrap();
        subs.sort_unstable();
        assert!(subs.contains(&0), "box 0 subsumes box 1; got {subs:?}");
        assert!(!subs.contains(&2), "box 2 does not subsume box 1");
        assert!(!subs.contains(&3), "box 3 is disjoint");
        // Matches exhaustive ground truth.
        let mut exh = idx.subsumers_exhaustive(&middle);
        exh.sort_unstable();
        assert_eq!(subs, exh, "indexed subsumers must match exhaustive");

        // Subsumees of box 1 = box 2 (and box 1 itself); box 0 and 3 are not.
        let mut down = idx.subsumees(&middle);
        down.sort_unstable();
        assert!(
            down.contains(&2),
            "box 2 is subsumed by box 1; got {down:?}"
        );
        assert!(!down.contains(&0), "box 0 is larger, not subsumed");
        assert!(!down.contains(&3), "box 3 is disjoint");
    }

    #[test]
    fn containment_recall_on_realistic_hierarchy() {
        // Quantify the lift: on a hierarchy of big general concepts (varied,
        // scattered centers) and small specific ones nested inside them, the
        // indexed `containing` must recover the enclosing regions -- including
        // big boxes whose center is far from the query -- at high recall vs the
        // exhaustive ground truth.
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let dim = 16;
        let mut rng = StdRng::seed_from_u64(7);
        let mut idx = RegionIndex::new(dim, IndexParams::default()).unwrap();
        let mut id = 0u32;

        let mut bigs = Vec::new();
        for _ in 0..10 {
            let c: Vec<f32> = (0..dim).map(|_| rng.random_range(-5.0..5.0)).collect();
            let hw: Vec<f32> = (0..dim).map(|_| rng.random_range(3.0..6.0)).collect();
            idx.add(id, AxisBox::from_center_offset(c.clone(), hw))
                .unwrap();
            bigs.push(c);
            id += 1;
        }
        let mut queries = Vec::new();
        for _ in 0..300 {
            let bc = &bigs[rng.random_range(0..bigs.len())];
            let c: Vec<f32> = bc.iter().map(|x| x + rng.random_range(-1.5..1.5)).collect();
            let hw: Vec<f32> = (0..dim).map(|_| rng.random_range(0.05..0.2)).collect();
            idx.add(id, AxisBox::from_center_offset(c.clone(), hw))
                .unwrap();
            queries.push(c);
            id += 1;
        }
        idx.build().unwrap();

        let (mut hit, mut total) = (0usize, 0usize);
        for q in &queries {
            let truth: std::collections::HashSet<u32> =
                idx.containing_exhaustive(q).into_iter().collect();
            let got: std::collections::HashSet<u32> = idx
                .containing(
                    q,
                    SearchParams {
                        ef: 100,
                        overretrieve: 10,
                    },
                )
                .unwrap()
                .into_iter()
                .collect();
            for t in &truth {
                hit += usize::from(got.contains(t));
                total += 1;
            }
        }
        let recall = hit as f64 / total as f64;
        assert!(recall > 0.95, "containment recall {recall:.3} below 0.95");
    }

    #[test]
    fn overlapping_and_nearest_region() {
        let mut idx = RegionIndex::new(2, Default::default()).unwrap();
        // A cluster of overlapping boxes near the origin, and far decoys.
        idx.add(0, AxisBox::new(vec![0.0, 0.0], vec![4.0, 4.0]))
            .unwrap();
        idx.add(1, AxisBox::new(vec![3.0, 3.0], vec![7.0, 7.0]))
            .unwrap(); // overlaps 0
        idx.add(2, AxisBox::new(vec![6.0, 6.0], vec![8.0, 8.0]))
            .unwrap(); // overlaps 1, not 0
        idx.add(3, AxisBox::new(vec![50.0, 50.0], vec![51.0, 51.0]))
            .unwrap(); // far
        for i in 4..18u32 {
            let o = 60.0 + i as f32 * 3.0;
            idx.add(i, AxisBox::new(vec![o, o], vec![o + 0.5, o + 0.5]))
                .unwrap();
        }
        idx.build().unwrap();

        let probe = AxisBox::new(vec![2.0, 2.0], vec![3.5, 3.5]); // overlaps 0 and 1
        let params = || SearchParams {
            ef: 64,
            overretrieve: 8,
        };

        let mut ov = idx.overlapping(&probe, params()).unwrap();
        ov.sort_unstable();
        assert!(
            ov.contains(&0) && ov.contains(&1),
            "probe overlaps 0 and 1; got {ov:?}"
        );
        assert!(!ov.contains(&3), "box 3 is far");
        // Matches exhaustive.
        let mut exh = idx.overlapping_exhaustive(&probe);
        exh.sort_unstable();
        assert_eq!(ov, exh, "indexed overlap must match exhaustive");

        // nearest_region to a box near the origin cluster returns 0/1/2 first.
        let near = idx
            .nearest_region(&AxisBox::new(vec![1.0, 1.0], vec![2.0, 2.0]), 3, params())
            .unwrap();
        let ids: std::collections::HashSet<u32> = near.iter().map(|(id, _)| *id).collect();
        assert!(
            ids.contains(&0),
            "nearest region to origin cluster includes box 0"
        );
        assert!(!ids.contains(&3), "far box 3 is not among the 3 nearest");
        for w in near.windows(2) {
            assert!(w[0].1 <= w[1].1, "nearest_region not sorted");
        }
    }

    #[test]
    fn subsumers_soft_recovers_partial_overlap() {
        // The trained-embedding case: a query whose center is inside a region but
        // whose box pokes out. Strict subsumers misses that region; soft recovers
        // it, thresholded by how much of the query it covers.
        let mut idx = RegionIndex::new(2, Default::default()).unwrap();
        idx.add(0, AxisBox::new(vec![0.0, 0.0], vec![10.0, 10.0]))
            .unwrap(); // contains query
        idx.add(1, AxisBox::new(vec![2.0, 2.0], vec![8.0, 8.0]))
            .unwrap(); // partial
        for i in 2..16u32 {
            let o = i as f32 * 4.0;
            idx.add(i, AxisBox::new(vec![o, o], vec![o + 0.5, o + 0.5]))
                .unwrap();
        }
        idx.build().unwrap();

        let query = AxisBox::new(vec![7.0, 7.0], vec![9.0, 9.0]); // center [8,8]
        let params = || SearchParams {
            ef: 64,
            overretrieve: 8,
        };

        // Strict: only box 0 fully contains the query.
        let strict = idx.subsumers(&query, params()).unwrap();
        assert!(
            strict.contains(&0) && !strict.contains(&1),
            "strict: {strict:?}"
        );

        // Soft: box 1 covers 1/4 of the query (vol([7,7]-[8,8]) / vol([7,7]-[9,9])).
        let soft_lo: Vec<u32> = idx
            .subsumers_soft(&query, 0.2, params())
            .unwrap()
            .iter()
            .map(|(id, _)| *id)
            .collect();
        assert!(
            soft_lo.contains(&0) && soft_lo.contains(&1),
            "soft@0.2: {soft_lo:?}"
        );
        let soft_hi: Vec<u32> = idx
            .subsumers_soft(&query, 0.5, params())
            .unwrap()
            .iter()
            .map(|(id, _)| *id)
            .collect();
        assert!(
            soft_hi.contains(&0) && !soft_hi.contains(&1),
            "soft@0.5: {soft_hi:?}"
        );

        // Ranked by probability, the full container (prob 1) first.
        let ranked = idx.subsumers_soft(&query, 0.0, params()).unwrap();
        assert_eq!(ranked[0].0, 0);
    }

    #[test]
    fn ball_subsumption() {
        let outer = Ball::new(vec![0.0, 0.0], 5.0);
        let inner = Ball::new(vec![1.0, 0.0], 1.0);
        let disjoint = Ball::new(vec![20.0, 0.0], 1.0);
        assert!(outer.contains_region(&inner));
        assert!(!inner.contains_region(&outer));
        assert!(!outer.contains_region(&disjoint));
    }

    #[test]
    fn containing_exhaustive_finds_enclosing_boxes() {
        let mut idx = RegionIndex::new(2, Default::default()).unwrap();
        idx.add(0, AxisBox::new(vec![0.0, 0.0], vec![10.0, 10.0]))
            .unwrap();
        idx.add(1, AxisBox::new(vec![4.0, 4.0], vec![6.0, 6.0]))
            .unwrap();
        idx.add(2, AxisBox::new(vec![20.0, 20.0], vec![21.0, 21.0]))
            .unwrap();
        for i in 3..15 {
            let o = (i as f32) * 3.0;
            idx.add(i, AxisBox::new(vec![o, o], vec![o + 0.5, o + 0.5]))
                .unwrap();
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
        idx.add(42, AxisBox::new(vec![0.0, 0.0], vec![1.0, 1.0]))
            .unwrap();
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
