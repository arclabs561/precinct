//! Updatable, durable multi-segment region-ANN index via `segstore`.
//!
//! Enabled by the optional `store` feature. The base [`RegionIndex`] is
//! build-once; this wraps a corpus of axis-aligned boxes in a segstore
//! `SegmentedStore` so regions can be added and deleted incrementally with a
//! write-ahead log + checkpoint + compaction, and the index survives a restart.
//!
//! Like vicinity's store, this is *multi-segment*: a per-segment `RegionIndex`
//! is built from the live regions of each segment and searched, then the
//! per-segment top-k are merged (the merge is exact). It is the deliberate
//! alternative to a single evolving graph; precinct builds on vicinity's HNSW, so
//! it inherits that family's choice.
//!
//! Specialized to [`AxisBox`]; a generic-over-[`crate::Region`] form is possible
//! once a second region type needs it.

use std::cmp::Ordering;
use std::sync::Arc;

use durability::{Directory, PersistenceResult};
use segstore::{SegmentedStore, Store};

use crate::{AxisBox, IndexParams, RegionIndex, SearchParams};

/// segstore payload: items are axis-aligned box regions, a segment is a batch of
/// source boxes (a per-segment `RegionIndex` is built from the live ones per query).
struct BoxBacking;

impl Store for BoxBacking {
    type Id = u32;
    type Item = AxisBox;
    type Segment = Vec<(u32, AxisBox)>;

    fn build_segment(&self, batch: &[(u32, AxisBox)]) -> Vec<(u32, AxisBox)> {
        batch.to_vec()
    }

    fn merge_segments(
        &self,
        segs: &[Vec<(u32, AxisBox)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, AxisBox)> {
        segs.iter()
            .flatten()
            .filter(|(id, _)| live(id))
            .cloned()
            .collect()
    }
}

/// An updatable, durable multi-segment region-ANN index over axis-aligned boxes.
pub struct UpdatableIndex {
    inner: SegmentedStore<BoxBacking>,
    dim: usize,
    m: usize,
    m_max: usize,
    ef_construction: usize,
}

impl UpdatableIndex {
    /// Open (or recover) an index under `dir` for `dim`-dimensional regions, using
    /// `params` to build each segment's index. Up to `flush_threshold` regions are
    /// buffered before a new immutable segment is sealed.
    pub fn open(
        dir: Arc<dyn Directory>,
        flush_threshold: usize,
        dim: usize,
        params: IndexParams,
    ) -> PersistenceResult<Self> {
        Ok(Self {
            inner: SegmentedStore::open(dir, BoxBacking, flush_threshold)?,
            dim,
            m: params.m,
            m_max: params.m_max,
            ef_construction: params.ef_construction,
        })
    }

    /// Add (or re-add) a box region by id.
    pub fn add(&mut self, id: u32, region: AxisBox) -> PersistenceResult<()> {
        self.inner.add(id, region)
    }

    /// Tombstone a region.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)
    }

    /// Merge segments (dropping tombstoned regions) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()
    }

    /// Persist a checkpoint without merging.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()
    }

    /// The `k` nearest regions to the query point, by point-to-region distance,
    /// over the live corpus. Returns `(region_id, distance)`.
    pub fn search(&self, query: &[f32], k: usize, params: SearchParams) -> Vec<(u32, f32)> {
        let (ef, overretrieve) = (params.ef, params.overretrieve);
        let mut cand: Vec<(u32, f32)> = Vec::new();
        for seg in self.inner.segments() {
            cand.extend(self.search_batch(seg, query, k, ef, overretrieve));
        }
        let buffered = self.inner.buffer().to_vec();
        cand.extend(self.search_batch(&buffered, query, k, ef, overretrieve));
        // Lower point-to-region distance is nearer.
        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        cand.truncate(k);
        cand
    }

    fn search_batch(
        &self,
        batch: &[(u32, AxisBox)],
        query: &[f32],
        k: usize,
        ef: usize,
        overretrieve: usize,
    ) -> Vec<(u32, f32)> {
        let params = IndexParams {
            m: self.m,
            m_max: self.m_max,
            ef_construction: self.ef_construction,
        };
        let mut idx = match RegionIndex::<AxisBox>::new(self.dim, params) {
            Ok(i) => i,
            Err(_) => return Vec::new(),
        };
        let mut any = false;
        for (id, region) in batch {
            if self.inner.is_live(id) && idx.add(*id, region.clone()).is_ok() {
                any = true;
            }
        }
        if !any || idx.build().is_err() {
            return Vec::new();
        }
        idx.search(query, k, SearchParams { ef, overretrieve })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use durability::MemoryDirectory;

    fn b(lo: f32, hi: f32) -> AxisBox {
        AxisBox::new(vec![lo, lo], vec![hi, hi])
    }

    #[test]
    fn add_delete_compact_recover_through_real_region_index() {
        let dir = MemoryDirectory::arc();
        {
            let mut store =
                UpdatableIndex::open(dir.clone(), 2, 2, IndexParams::default()).unwrap();
            store.add(0, b(0.0, 1.0)).unwrap(); // near the origin
            store.add(1, b(5.0, 6.0)).unwrap(); // flush
            store.add(2, b(10.0, 11.0)).unwrap(); // far; buffered

            // A point inside box 0 is nearest to region 0.
            let top: Vec<u32> = store
                .search(&[0.5, 0.5], 1, SearchParams::default())
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert_eq!(top, vec![0], "point inside box 0 retrieves region 0");

            store.delete(0).unwrap();
            let top: Vec<u32> = store
                .search(&[0.5, 0.5], 1, SearchParams::default())
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert_eq!(top, vec![1], "after deleting 0, nearest region is 1");

            store.compact().unwrap();
            assert_eq!(
                store
                    .search(&[0.5, 0.5], 1, SearchParams::default())
                    .first()
                    .map(|(id, _)| *id),
                Some(1)
            );
        }
        let store = UpdatableIndex::open(dir, 2, 2, IndexParams::default()).unwrap();
        let top: Vec<u32> = store
            .search(&[0.5, 0.5], 1, SearchParams::default())
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(top, vec![1], "recovery preserves the search");
    }
}
