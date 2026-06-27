# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The 0.x series is
unstable: minor bumps may break the public API.

## [0.8.2] - 2026-06-27

### Added

- `store::UpdatableIndex::compact_tiers()`: one round of size-tiered compaction
  (merge similarly-sized segments), keeping segment count bounded without a full
  `compact()`.

## [0.8.1] - 2026-06-27

### Added

- `store::UpdatableIndex::reclaim(min_live_ratio)` and `space_amplification()`
  (via the new `Store::live_len`): cheap tombstone reclamation, merging only the
  delete-heavy segments instead of a full compaction.

## [0.8.0] - 2026-06-27

### Added

- `RegionIndex::subsumers_soft(query, min_prob)`: soft subsumption, returning
  regions with `entailment_prob(query) >= min_prob` ranked by probability.
  Trained region embeddings (Gumbel boxes) nest *softly* -- a child's center
  falls inside its parent but its full box pokes out -- so strict `subsumers` is
  0% on real trained WordNet boxes while the soft form (and membership) recover
  the is-a ancestor 98%. `min_prob = 1.0` reduces to strict containment.
- `examples/wordnet_boxes`: in-domain validation on subsume's trained WordNet box
  checkpoint, reporting membership / soft / strict ancestor recall (98/98/0%).

## [0.7.0] - 2026-06-27

### Added

- `Ellipsoid` region type (axis-aligned, per-axis semi-axes): the anisotropic
  member of the region family. Exact `contains`, `log_volume`, and a
  Newton-iterated exact point-to-surface `distance_to_point`; region-to-region
  predicates use the bounding box as a documented approximation. Implements
  `Region`, so it works in `RegionIndex` like `AxisBox` / `Ball`.
- A containment-recall regression test on a realistic far-centered hierarchy
  (big general boxes + small nested boxes): the lift recovers enclosing regions
  at >95% recall vs the exhaustive ground truth.

## [0.6.0] - 2026-06-27

Completes the region query algebra: beyond containment, the serving interface
now covers overlap, region-to-region similarity, and the box scoring primitives.

### Added

- `RegionIndex::overlapping(region)` / `overlapping_exhaustive` -- the overlap
  query (regions intersecting a query region, the conjunction primitive).
- `RegionIndex::nearest_region(region, k)` -- region-to-region nearest neighbor
  (a concept's nearest concepts), by center distance.
- `Region::overlaps_region` (intersection predicate), `Region::log_volume`
  (generality), and `Region::entailment_prob` (soft subsumption probability
  `vol(self ∩ other) / vol(other)`, the box-lattice conditional; exact for boxes,
  approximate for balls). Implemented for `AxisBox` and `Ball`.

### Breaking

- `Region` requires three more methods: `overlaps_region`, `log_volume`,
  `entailment_prob`. Custom `Region` implementations must add them; `AxisBox` and
  `Ball` are covered.

## [0.5.0] - 2026-06-27

precinct becomes a high-dimensional index for regions-as-objects (the serving
layer for trained region embeddings), answering three query families instead of
one. See `docs/design/region-index.md`.

### Added

- `RegionIndex::containing(point)` -- membership query, now backed by a
  *power-distance lift*: each region is indexed by a vector whose ranking
  respects extent, so an enclosing region is found even when its center is far
  from the query (the case a center-only index misses).
- `RegionIndex::subsumers(region)` / `subsumees(region)` -- region-to-region
  subsumption retrieval (a concept's hypernyms / hyponyms), with
  `subsumers_exhaustive` for ground truth.
- `Region::bounding_ball()` and `Region::contains_region()` on the trait, with
  `AxisBox` and `Ball` implementations.

### Changed

- `RegionIndex` now keeps two HNSW graphs: a center graph for `search` (nearest)
  and a lift graph for `containing` / `subsumers`. Nearest recall is unchanged
  (GloVe recall@10 92.1% at 10x); routing nearest through the lift would have
  cost ~4 points, so the queries use separate indexes (~2x graph memory).

### Breaking

- `Region` requires two new methods, `bounding_ball` and `contains_region`. Any
  custom `Region` implementation must add them; `AxisBox` and `Ball` are covered.

## [0.4.0] - 2026-06-27

### Added

- `examples/glove_concepts`: a recall benchmark on real high-dimensional concept
  regions (top-50K GloVe-6B-50d vectors clustered into 5,000 bounding boxes via
  `clump`). recall@10 92.1% (10x over-retrieve), 99.3% (50x). Fetch with
  `scripts/fetch_glove.sh`.
- `examples/geo_regions`: nearest-region search over real geographic boxes (177
  Natural Earth country bounding boxes). Shows the surface-vs-center distinction
  (a South Pacific point resolves to Chile). Fetch with
  `scripts/fetch_natural_earth.sh`.

### Changed

- `store::UpdatableIndex` now caches each segment's `RegionIndex` by the
  segment's stable `Arc` identity (via segstore 0.2), so a mutation rebuilds only
  the new or changed segments instead of the whole corpus on the next query.
- Requires `segstore` 0.2 (only affects the optional `store` feature; the on-disk
  store format changed, so a `store` index written by 0.3.x is not read by 0.4.0).

## [0.3.1] - 2026-06-26

### Fixed

- `store::UpdatableIndex` caches the per-segment `RegionIndex` and rebuilds it
  only on mutation (add/delete/compact), instead of rebuilding every segment on
  every query.
- `store::UpdatableIndex::add` now returns an error for a region whose dimension
  does not match the index, rather than silently dropping it from every rebuild.

### Changed

- `store` docs no longer claim an "exact" cross-segment merge. The underlying
  HNSW search is approximate, so the merged result is approximate.

## [0.3.0] - 2026-06-26

### Added

- Optional `serde` feature: `Serialize`/`Deserialize` derives on `AxisBox`.
- Optional `store` feature: `store::UpdatableIndex`, an updatable, durable
  multi-segment region-ANN index over axis-aligned boxes, backed by
  [`segstore`](https://crates.io/crates/segstore). A per-segment `RegionIndex` is
  built from the live regions of each segment and searched, then the per-segment
  top-k are merged (exact). Incremental add/delete plus write-ahead log,
  checkpoint, compaction, and crash recovery. Opt-in; the default build does not
  depend on segstore.

## [0.2.0]

Initial documented release.
