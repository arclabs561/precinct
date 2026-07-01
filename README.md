# precinct

Approximate nearest-neighbor search over region embeddings (boxes, balls, ellipsoids).

Point-ANN indices (HNSW, FAISS) index points; R-trees index regions but collapse
above ~10 dimensions. Region embeddings -- axis-aligned boxes, balls, ellipsoids,
or any custom `Region` -- represent
concepts as volumes, and trained ones live in 64-200 dimensions, so neither tool
fits. precinct is the high-dimensional index for regions-as-objects: it answers
three queries over a region corpus.

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
precinct = "0.8"
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
corpus; segments are searched and merged, and like the underlying HNSW the
merged result is approximate.
Opt-in; the default build does not depend on segstore.

## Recall

Recall@k against an exhaustive point-to-region scan (the correctness oracle),
reported next to the realistic baseline you would use without precinct: plain
point-ANN over the region *centers*, which ignores extent.

Real data, `examples/glove_concepts` (50K GloVe-6B-50d vectors clustered into
5,000 concept boxes, the bounding box of each cluster of related words):

| Over-retrieve | precinct (region-aware) | naive point-ANN on centers |
|---|---|---|
| 10x | 92.1% | 46.7% |
| 50x | 99.3% | 46.7% |

The region-distance rerank roughly doubles recall over ranking by center
distance; over-retrieve does not help the baseline because its ranking is wrong,
not just truncated.

Real data, `examples/geo_regions` (177 Natural Earth country boxes, `[lon, lat]`
point queries): recall@3 92.9% over a world grid, and the nearest region by
surface distance correctly diverges from the nearest by center (a South Pacific
point resolves to Chile, far from any centroid). Fetch either dataset with the
matching `scripts/fetch_*.sh`.

Synthetic box datasets (uniform-random centers, varied widths, `examples/recall_gap`):

| Scenario | Recall@10 (10x) | Recall@10 (50x) |
|---|---|---|
| Narrow (w=0.01, d=128) | 96.3% | 99.4% |
| Medium (w=0.1, d=128) | 97.1% | 99.9% |
| Wide (w=0.5, d=128) | 93.7% | 99.6% |
| Mixed hierarchy (d=128) | 93.6% | 99.4% |
| Medium (d=400) | 88.3% | 97.5% |
| 50K scale (d=128) | 78.7% | 91.8% |

The point ANN backend is [vicinity](https://github.com/arclabs561/vicinity) (HNSW).

## Examples

See [examples/README.md](examples/README.md) for runnable examples with
captured output and data requirements.

## License

MIT OR Apache-2.0
