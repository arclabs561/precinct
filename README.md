# precinct

Approximate nearest-neighbor search over region embeddings.

`precinct` indexes boxes, balls, ellipsoids, or any custom `Region` in high
dimensions. It answers these query families over a region corpus:

- **nearest** -- the `k` regions closest to a point, by true point-to-region
  distance (center index + rerank).
- **membership** (`containing`) -- the regions that enclose a point. Candidates
  come from a *power-distance lift* that ranks regions by extent, so a large
  general concept is found even when its center is far from the point -- the case
  a center-only index misses.
- **subsumption** (`subsumers` / `subsumees`) -- the regions that contain, or are
  contained by, a query region (a concept's hypernyms / hyponyms).
- **overlap** (`overlapping`) -- the regions that intersect a query region (the
  conjunction primitive: concepts sharing members).
- **region similarity** (`nearest_region`) -- the regions nearest a query region.

Retrieved regions carry their own scoring: `Region::log_volume` (generality) and
`Region::entailment_prob` (the soft subsumption probability
`vol(self ∩ other) / vol(other)`, the box-lattice conditional).

## Install

```toml
[dependencies]
precinct = "0.9"
```

or `cargo add precinct`.

## Usage

```rust
use precinct::{AxisBox, RegionIndex, SearchParams};

// Build an index of 2-d boxes
let mut idx = RegionIndex::new(2, Default::default()).unwrap();
idx.add(0, AxisBox::new(vec![0.0, 0.0], vec![10.0, 10.0])); // general concept
idx.add(1, AxisBox::new(vec![4.0, 4.0], vec![6.0, 6.0]));   // specific concept
idx.add(2, AxisBox::new(vec![20.0, 20.0], vec![21.0, 21.0]));
idx.build().unwrap();

// nearest region to a point inside only the general concept
let nearest = idx.search(&[1.0, 1.0], 1, Default::default()).unwrap();
assert_eq!(nearest[0].0, 0);

// membership: regions enclosing [5, 5]  -> {0, 1}
let enclosing = idx.containing(&[5.0, 5.0], Default::default()).unwrap();

// subsumption: regions that contain a small probe box -> the more general concepts
let probe = AxisBox::new(vec![4.5, 4.5], vec![5.5, 5.5]);
let subsumers = idx.subsumers(&probe, Default::default()).unwrap();
```

`SearchParams::overretrieve` controls the over-retrieval factor (default 10x).
Increasing it trades query latency for recall.

## Updatable index (`store` feature)

`store::UpdatableIndex` wraps the region index in a durable, segmented store
([`segstore`](https://crates.io/crates/segstore)): incremental add/delete, a
write-ahead log, checkpoint, compaction, and crash recovery. Per-segment
`RegionIndex`es are cached by stable segment identity and persisted as sidecars,
so a mutation or restart rebuilds only the new or changed segments, not the whole
corpus. For read-only serving, `store::SnapshotIndex` opens the checkpoint
manifest and queries sidecars first, decoding a source-region segment only when
its sidecar is missing or must be rebuilt. Segments are searched and merged, and
like the underlying HNSW the merged result is approximate.
The store exposes the same query families as `RegionIndex`: nearest,
membership, subsumption, overlap, region similarity, and exhaustive scans.
Opt-in; the default build does not depend on segstore.

```bash
cargo run --features store --example updatable_store
```

For measurement, `cargo run --release --features store --example store_reopen_diagnostics`
prints the first snapshot-search cost with persisted region-index sidecars
present versus after deleting those sidecars and forcing source-segment
rebuilds.

## Recall

The examples measure recall@k against an exhaustive point-to-region scan. Run
`examples/recall_gap` for synthetic box datasets, `examples/glove_concepts` for
clustered GloVe vectors, or `examples/geo_regions` for geographic bounding
boxes. See [examples/README.md](examples/README.md) for commands, captured
output, and data requirements.

The point ANN backend is [vicinity](https://github.com/arclabs561/vicinity) (HNSW).

## Examples

See [examples/README.md](examples/README.md) for the full set of runnable
examples.

## License

MIT OR Apache-2.0
