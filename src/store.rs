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
use std::collections::{HashMap, HashSet};
use std::io::Read;
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

/// The `kind` tag for a persisted per-segment region-index sidecar.
const INDEX_KIND: &str = "region-hnsw";
const SIDECAR_MAGIC: &[u8; 8] = b"PRECIDX1";
const SIDECAR_VERSION: u32 = 1;

/// An updatable, durable multi-segment region-ANN index over axis-aligned boxes.
pub struct UpdatableIndex {
    inner: SegmentedStore<BoxBacking>,
    dim: usize,
    m: usize,
    m_max: usize,
    ef_construction: usize,
    sidecar_recipe: String,
    cache: RefCell<Cache>,
    /// Segment ids whose on-disk region-index sidecar was validated or written
    /// in this process, so checkpoint persistence stays O(new segments).
    persisted: RefCell<HashSet<u64>>,
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
            sidecar_recipe: Self::make_sidecar_recipe(dim, params),
            cache: RefCell::new(Cache {
                by_ptr: HashMap::new(),
            }),
            persisted: RefCell::new(HashSet::new()),
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
        let ids = self.inner.segment_ids();
        for (seg_idx, seg) in self.inner.segments().iter().enumerate() {
            if seg.iter().any(|(sid, _)| *sid == id) {
                cache.by_ptr.remove(&(Arc::as_ptr(seg) as usize));
                let seg_id = ids[seg_idx];
                self.persisted.borrow_mut().remove(&seg_id);
                let _ = self
                    .inner
                    .dir()
                    .delete(&self.inner.index_name(seg_id, INDEX_KIND));
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
        self.inner.checkpoint()?;
        self.persist_new_segments();
        Ok(())
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
            // Build only segments not already cached, loading a persisted sidecar
            // first when one matches the current recipe and live id set.
            let ids = self.inner.segment_ids();
            for (i, seg) in segs.iter().enumerate() {
                let key = Arc::as_ptr(seg) as usize;
                let seg_id = ids[i];
                cache
                    .by_ptr
                    .entry(key)
                    .or_insert_with(|| self.build_or_load(&seg[..], seg_id));
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

    /// Load segment `seg_id`'s persisted region index from its sidecar, or build
    /// it over the segment's live regions and persist it for the next restart.
    fn build_or_load(&self, seg: &[(u32, AxisBox)], seg_id: u64) -> Option<RegionIndex<AxisBox>> {
        if let Some(idx) = self.load_sidecar(seg, seg_id) {
            self.persisted.borrow_mut().insert(seg_id);
            return Some(idx);
        }
        let idx = self.build_live_index(seg)?;
        self.persist_sidecar(&idx, seg_id);
        Some(idx)
    }

    /// Load a sidecar only if its recipe matches and its ids match the segment's
    /// current live ids. A stale sidecar can never serve a tombstoned region.
    fn load_sidecar(&self, seg: &[(u32, AxisBox)], seg_id: u64) -> Option<RegionIndex<AxisBox>> {
        let name = self.inner.index_name(seg_id, INDEX_KIND);
        if !self.inner.dir().exists(&name) {
            return None;
        }
        let mut bytes = Vec::new();
        self.inner
            .dir()
            .open_file(&name)
            .ok()?
            .read_to_end(&mut bytes)
            .ok()?;
        let index_bytes = self.decode_sidecar(&bytes)?;
        let idx = RegionIndex::from_postcard(index_bytes).ok()?;
        let mut live = HashSet::with_capacity(seg.len());
        for (id, _) in seg {
            if self.inner.is_live(id) {
                live.insert(*id);
            }
        }
        if idx.ids().len() == live.len() && idx.ids().iter().all(|id| live.contains(id)) {
            Some(idx)
        } else {
            None
        }
    }

    /// Persist a built per-segment region index as its sidecar. Best-effort: a
    /// failed write leaves the in-memory index usable and simply rebuilds later.
    fn persist_sidecar(&self, idx: &RegionIndex<AxisBox>, seg_id: u64) {
        if let Ok(index) = idx.to_postcard() {
            let Some(bytes) = self.encode_sidecar(&index) else {
                return;
            };
            if self
                .inner
                .dir()
                .atomic_write(&self.inner.index_name(seg_id, INDEX_KIND), &bytes)
                .is_ok()
            {
                self.persisted.borrow_mut().insert(seg_id);
            }
        }
    }

    fn make_sidecar_recipe(dim: usize, params: IndexParams) -> String {
        format!(
            "precinct-store-region-hnsw-v1;\
             region=axis-box;dim={};m={};m_max={};ef_construction={};\
             center=vicinity-hnsw-l2;lift=power-distance-mips-l2;\
             codec=postcard-region-index-v1",
            dim, params.m, params.m_max, params.ef_construction
        )
    }

    fn encode_sidecar(&self, index: &[u8]) -> Option<Vec<u8>> {
        let recipe = self.sidecar_recipe.as_bytes();
        let recipe_len = u32::try_from(recipe.len()).ok()?;
        let mut bytes = Vec::with_capacity(16 + recipe.len() + index.len());
        bytes.extend_from_slice(SIDECAR_MAGIC);
        bytes.extend_from_slice(&SIDECAR_VERSION.to_le_bytes());
        bytes.extend_from_slice(&recipe_len.to_le_bytes());
        bytes.extend_from_slice(recipe);
        bytes.extend_from_slice(index);
        Some(bytes)
    }

    fn decode_sidecar<'a>(&self, bytes: &'a [u8]) -> Option<&'a [u8]> {
        if bytes.len() < 16 {
            return None;
        }
        if &bytes[..8] != SIDECAR_MAGIC {
            return None;
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
        if version != SIDECAR_VERSION {
            return None;
        }
        let recipe_len = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
        let recipe_start = 16usize;
        let recipe_end = recipe_start.checked_add(recipe_len)?;
        if bytes.len() < recipe_end {
            return None;
        }
        if &bytes[recipe_start..recipe_end] != self.sidecar_recipe.as_bytes() {
            return None;
        }
        Some(&bytes[recipe_end..])
    }

    /// Persist sidecars for sealed segments that lack a current one. This is
    /// incremental: already validated/written segment ids are skipped.
    fn persist_new_segments(&self) {
        let ids = self.inner.segment_ids();
        let id_set: HashSet<u64> = ids.iter().copied().collect();
        self.persisted.borrow_mut().retain(|id| id_set.contains(id));
        for (i, seg) in self.inner.segments().iter().enumerate() {
            let seg_id = ids[i];
            if self.persisted.borrow().contains(&seg_id) {
                continue;
            }
            if self.load_sidecar(&seg[..], seg_id).is_some() {
                self.persisted.borrow_mut().insert(seg_id);
                continue;
            }
            if let Some(idx) = self.build_live_index(&seg[..]) {
                self.persist_sidecar(&idx, seg_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use durability::MemoryDirectory;

    fn b(lo: f32, hi: f32) -> AxisBox {
        AxisBox::new(vec![lo, lo], vec![hi, hi])
    }

    fn read_file(dir: &Arc<dyn Directory>, name: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        dir.open_file(name)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }

    fn checkpointed_store(dir: Arc<dyn Directory>, params: IndexParams) -> (String, Vec<u8>) {
        let mut store = UpdatableIndex::open(dir, 4, 2, params).unwrap();
        for i in 0..12u32 {
            let lo = i as f32 * 0.25;
            store.add(i, b(lo, lo + 0.5)).unwrap();
        }
        store.checkpoint().unwrap();
        let seg_id = store.inner.segment_ids()[0];
        let name = store.inner.index_name(seg_id, INDEX_KIND);
        let bytes = read_file(store.inner.dir(), &name);
        (name, bytes)
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

    #[test]
    fn checkpoint_persists_sidecars_and_reopen_loads_them() {
        let dir = MemoryDirectory::arc();
        {
            let mut store =
                UpdatableIndex::open(dir.clone(), 4, 2, IndexParams::default()).unwrap();
            for i in 0..12u32 {
                let lo = i as f32 * 0.25;
                store.add(i, b(lo, lo + 0.5)).unwrap();
            }
            store.checkpoint().unwrap();

            let ids: Vec<u64> = store.inner.segment_ids().to_vec();
            assert!(
                !ids.is_empty(),
                "12 adds at flush 4 seal at least one segment"
            );
            for id in &ids {
                assert!(
                    store
                        .inner
                        .dir()
                        .exists(&store.inner.index_name(*id, INDEX_KIND)),
                    "segment {id} must have a persisted sidecar after checkpoint"
                );
            }
        }

        let store = UpdatableIndex::open(dir, 4, 2, IndexParams::default()).unwrap();
        assert!(
            !store
                .search(&[0.3, 0.3], 1, SearchParams::default())
                .is_empty(),
            "search over loaded sidecars returns results"
        );
    }

    #[test]
    fn region_sidecar_recipe_mismatch_rebuilds() {
        let dir = MemoryDirectory::arc();
        let original = IndexParams::default();
        let (name, before) = checkpointed_store(dir.clone(), original);
        assert_eq!(
            &before[..SIDECAR_MAGIC.len()],
            SIDECAR_MAGIC,
            "new sidecars carry the precinct region-index envelope"
        );

        let changed = IndexParams {
            m: 8,
            m_max: 16,
            ..IndexParams::default()
        };
        let store = UpdatableIndex::open(dir.clone(), 4, 2, changed).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "sidecar built with default HNSW params must not load under changed params"
        );
        assert!(
            !store
                .search(&[0.3, 0.3], 1, SearchParams::default())
                .is_empty(),
            "mismatched sidecar falls back to rebuild"
        );

        let after = read_file(store.inner.dir(), &name);
        assert_ne!(before, after, "rebuild overwrites the stale-recipe sidecar");
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_some(),
            "rebuilt sidecar now matches the current recipe"
        );
    }

    #[test]
    fn region_sidecar_envelope_rejects_corrupt_headers() {
        let store =
            UpdatableIndex::open(MemoryDirectory::arc(), 4, 2, IndexParams::default()).unwrap();
        let index = b"index-bytes";
        let bytes = store.encode_sidecar(index).unwrap();
        assert_eq!(store.decode_sidecar(&bytes), Some(index.as_slice()));

        assert!(store.decode_sidecar(&bytes[..8]).is_none());

        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 0xFF;
        assert!(store.decode_sidecar(&bad_magic).is_none());

        let mut bad_version = bytes.clone();
        bad_version[8..12].copy_from_slice(&(SIDECAR_VERSION + 1).to_le_bytes());
        assert!(store.decode_sidecar(&bad_version).is_none());

        let mut bad_recipe_len = bytes.clone();
        bad_recipe_len[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(store.decode_sidecar(&bad_recipe_len).is_none());

        let mut bad_recipe = bytes.clone();
        bad_recipe[16] ^= 0x01;
        assert!(store.decode_sidecar(&bad_recipe).is_none());
    }

    #[test]
    fn region_sidecar_invalid_payload_rebuilds() {
        let dir = MemoryDirectory::arc();
        let (name, _) = checkpointed_store(dir.clone(), IndexParams::default());
        {
            let store = UpdatableIndex::open(dir.clone(), 4, 2, IndexParams::default()).unwrap();
            let corrupt = store
                .encode_sidecar(b"not-a-postcard-region-index")
                .unwrap();
            store.inner.dir().atomic_write(&name, &corrupt).unwrap();
        }

        let store = UpdatableIndex::open(dir.clone(), 4, 2, IndexParams::default()).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "valid envelope with invalid region-index bytes is rejected"
        );
        assert!(
            !store
                .search(&[0.3, 0.3], 1, SearchParams::default())
                .is_empty(),
            "invalid payload falls back to rebuild"
        );
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_some(),
            "rebuilt sidecar loads after the fallback"
        );
    }

    #[test]
    fn deleted_id_does_not_resurface_through_a_sidecar() {
        let dir = MemoryDirectory::arc();
        {
            let mut store =
                UpdatableIndex::open(dir.clone(), 2, 2, IndexParams::default()).unwrap();
            store.add(0, b(0.0, 1.0)).unwrap();
            store.add(1, b(0.2, 1.2)).unwrap();
            store.add(2, b(5.0, 6.0)).unwrap();
            store.checkpoint().unwrap();
            store.delete(0).unwrap();
            store.checkpoint().unwrap();
        }

        let store = UpdatableIndex::open(dir, 2, 2, IndexParams::default()).unwrap();
        let top: Vec<u32> = store
            .search(&[0.5, 0.5], 3, SearchParams::default())
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(
            !top.contains(&0),
            "deleted id 0 must not resurface from a persisted sidecar"
        );
        assert!(
            top.contains(&1),
            "nearest live box should remain searchable"
        );
    }

    #[test]
    fn checkpoint_after_replayed_delete_rewrites_stale_sidecar() {
        let dir = MemoryDirectory::arc();
        let (name, stale_bytes) = {
            let mut store =
                UpdatableIndex::open(dir.clone(), 2, 2, IndexParams::default()).unwrap();
            store.add(0, b(0.0, 1.0)).unwrap();
            store.add(1, b(0.2, 1.2)).unwrap();
            store.add(2, b(5.0, 6.0)).unwrap();
            store.checkpoint().unwrap();

            let seg_id = store.inner.segment_ids()[0];
            let name = store.inner.index_name(seg_id, INDEX_KIND);
            let bytes = read_file(store.inner.dir(), &name);

            // Simulate a crash after the delete is durably logged but before
            // `UpdatableIndex::delete` removes the now-stale sidecar.
            store.inner.delete(0).unwrap();
            (name, bytes)
        };

        let mut store = UpdatableIndex::open(dir.clone(), 2, 2, IndexParams::default()).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "replayed tombstone must make the old sidecar stale"
        );

        store.checkpoint().unwrap();

        let rewritten = read_file(&dir, &name);
        assert_ne!(
            rewritten, stale_bytes,
            "checkpoint should rewrite stale sidecars even before search"
        );
        let idx = store
            .load_sidecar(&store.inner.segments()[0][..], seg_id)
            .expect("rewritten sidecar should be valid");
        assert!(
            !idx.ids().contains(&0),
            "rewritten sidecar must exclude the replayed delete"
        );
        assert!(
            idx.ids().contains(&1),
            "rewritten sidecar should keep live ids from the segment"
        );
    }
}
