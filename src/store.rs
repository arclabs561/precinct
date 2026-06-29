//! Updatable, durable multi-segment region-ANN index via `segstore`.
//!
//! Enabled by the optional `store` feature. The base [`RegionIndex`] is
//! build-once; this wraps a corpus of axis-aligned boxes in a segstore
//! `SegmentedStore` so regions can be added and deleted incrementally with a
//! write-ahead log + checkpoint + compaction, and the index survives a restart.
//!
//! Like vicinity's store, this is *multi-segment*: a per-segment `RegionIndex`
//! is built from the live regions of each segment and searched, then the
//! per-segment top-k are merged. The cross-segment merge is exact *given* exact
//! per-segment top-k, but the underlying HNSW search is itself approximate, so
//! the merged result is approximate (as any HNSW result is). It is the deliberate
//! alternative to a single evolving graph; precinct builds on vicinity's HNSW, so
//! it inherits that family's choice.
//!
//! Each per-segment index is built over that segment's *live* regions and
//! **cached**, rebuilt only when the index is mutated (an add that seals a
//! segment, a delete, or a compaction), not on every query. The small unflushed
//! buffer is built per query.
//!
//! Specialized to [`AxisBox`]; a generic-over-[`crate::Region`] form is possible
//! once a second region type needs it.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use durability::{Directory, PersistenceError, PersistenceResult};
use segstore::{SegmentedStore, Store};

use crate::{AxisBox, IndexParams, Region, RegionIndex, SearchParams};

/// segstore payload: items are axis-aligned box regions, a segment is a batch of
/// source boxes (a per-segment `RegionIndex` is built + cached from the live ones).
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
        segs: &[&Vec<(u32, AxisBox)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, AxisBox)> {
        segs.iter()
            .flat_map(|s| s.iter())
            .filter(|(id, _)| live(id))
            .cloned()
            .collect()
    }

    fn segment_len(&self, seg: &Vec<(u32, AxisBox)>) -> usize {
        seg.len()
    }

    fn live_len(&self, seg: &Vec<(u32, AxisBox)>, live: &dyn Fn(&u32) -> bool) -> Option<usize> {
        Some(seg.iter().filter(|(id, _)| live(id)).count())
    }
}

/// Per-segment region indexes keyed by the segment's stable `Arc` identity. Because
/// segstore keeps an unchanged segment's `Arc` across mutations, a sealed add only
/// builds the one new segment's index (the rest are reused) instead of rebuilding
/// the whole corpus -- the dominant cost for an interactive add-then-search loop.
struct Cache {
    by_ptr: HashMap<usize, Option<RegionIndex<AxisBox>>>,
}

/// An updatable, durable multi-segment region-ANN index over axis-aligned boxes.
pub struct UpdatableIndex {
    inner: SegmentedStore<BoxBacking>,
    dim: usize,
    m: usize,
    m_max: usize,
    ef_construction: usize,
    cache: RefCell<Cache>,
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
            cache: RefCell::new(Cache {
                by_ptr: HashMap::new(),
            }),
        })
    }

    /// Add (or re-add) a box region by id. Returns an error if the region's
    /// dimension does not match the index, rather than silently dropping it from
    /// every per-segment rebuild.
    pub fn add(&mut self, id: u32, region: AxisBox) -> PersistenceResult<()> {
        if region.dim() != self.dim {
            return Err(PersistenceError::InvalidConfig(format!(
                "region dimension {} does not match index dimension {}",
                region.dim(),
                self.dim
            )));
        }
        // A sealed add introduces a new segment (a new Arc identity); existing
        // segments keep theirs, so the cache reuses them and builds only the new one.
        self.inner.add(id, region)?;
        Ok(())
    }

    /// Add (or re-add) many regions, syncing the write-ahead log once for the whole
    /// batch instead of once per region. This is the bulk-ingest path (the
    /// corpus-load phase): per-item WAL sync is the dominant cost on a real disk, so
    /// one sync per batch is several times faster than a loop of [`Self::add`].
    /// Every region's dimension is validated before any is ingested.
    pub fn extend(
        &mut self,
        regions: impl IntoIterator<Item = (u32, AxisBox)>,
    ) -> PersistenceResult<()> {
        let dim = self.dim;
        let validated: Result<Vec<(u32, AxisBox)>, PersistenceError> = regions
            .into_iter()
            .map(|(id, region)| {
                if region.dim() != dim {
                    Err(PersistenceError::InvalidConfig(format!(
                        "region dimension {} does not match index dimension {}",
                        region.dim(),
                        dim
                    )))
                } else {
                    Ok((id, region))
                }
            })
            .collect();
        self.inner.extend(validated?)?;
        Ok(())
    }

    /// Tombstone a region.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)?;
        // A tombstone only changes the live-set of the segment that holds `id`, so
        // invalidate just that segment's cached index -- not the whole cache.
        let mut cache = self.cache.borrow_mut();
        for seg in self.inner.segments() {
            if seg.iter().any(|(sid, _)| *sid == id) {
                cache.by_ptr.remove(&(Arc::as_ptr(seg) as usize));
            }
        }
        Ok(())
    }

    /// Merge segments (dropping tombstoned regions) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()?;
        Ok(())
    }

    /// Persist a checkpoint without merging.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()
    }

    /// Run one round of size-tiered compaction, merging similarly-sized segments
    /// so the segment count stays bounded without a full [`compact`](Self::compact).
    pub fn compact_tiers(&mut self) -> PersistenceResult<()> {
        self.inner.compact_tiers()?;
        Ok(())
    }

    /// Merge only the segments whose live ratio is below `min_live_ratio`,
    /// reclaiming tombstoned regions -- the cheap alternative to a full
    /// [`compact`](Self::compact) when a few segments are delete-heavy.
    pub fn reclaim(&mut self, min_live_ratio: f64) -> PersistenceResult<()> {
        self.inner.reclaim_tombstones(min_live_ratio)?;
        Ok(())
    }

    /// Storage amplification: stored regions divided by live regions (`1.0` with
    /// no tombstones, higher as deletes accumulate).
    pub fn space_amplification(&self) -> Option<f64> {
        self.inner.space_amplification()
    }

    /// The `k` nearest regions to the query point, by point-to-region distance,
    /// over the live corpus. Returns `(region_id, distance)`.
    pub fn search(&self, query: &[f32], k: usize, params: SearchParams) -> Vec<(u32, f32)> {
        let SearchParams { ef, overretrieve } = params;
        let sp = || SearchParams { ef, overretrieve };
        let mut cand: Vec<(u32, f32)> = Vec::new();
        {
            let segs = self.inner.segments();
            let mut cache = self.cache.borrow_mut();
            // Drop cached indexes for segments no longer present (post-compaction).
            let current: std::collections::HashSet<usize> =
                segs.iter().map(|a| Arc::as_ptr(a) as usize).collect();
            cache.by_ptr.retain(|key, _| current.contains(key));
            // Build only segments not already cached (i.e. new ones).
            for seg in segs {
                let key = Arc::as_ptr(seg) as usize;
                cache
                    .by_ptr
                    .entry(key)
                    .or_insert_with(|| self.build_live_index(&seg[..]));
            }
            for idx in cache.by_ptr.values().flatten() {
                cand.extend(idx.search(query, k, sp()).unwrap_or_default());
            }
        }
        let buffered = self.inner.buffer().to_vec();
        if let Some(idx) = self.build_live_index(&buffered) {
            cand.extend(idx.search(query, k, sp()).unwrap_or_default());
        }
        // Lower point-to-region distance is nearer.
        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        cand.truncate(k);
        cand
    }

    /// Build a per-segment `RegionIndex` over the live regions of `batch` (None if
    /// empty or the build fails).
    fn build_live_index(&self, batch: &[(u32, AxisBox)]) -> Option<RegionIndex<AxisBox>> {
        let params = IndexParams {
            m: self.m,
            m_max: self.m_max,
            ef_construction: self.ef_construction,
        };
        let mut idx = match RegionIndex::<AxisBox>::new(self.dim, params) {
            Ok(i) => i,
            Err(_) => return None,
        };
        let mut any = false;
        for (id, region) in batch {
            if self.inner.is_live(id) && idx.add(*id, region.clone()).is_ok() {
                any = true;
            }
        }
        if !any || idx.build().is_err() {
            return None;
        }
        Some(idx)
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
            // Second query (no mutation) must use the cache and stay correct.
            let again: Vec<u32> = store
                .search(&[0.5, 0.5], 1, SearchParams::default())
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert_eq!(again, vec![0], "cached query is stable");

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
