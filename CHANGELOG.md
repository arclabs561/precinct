# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The 0.x series is
unstable: minor bumps may break the public API.

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
