# precinct

Approximate nearest-neighbor search over region embeddings (boxes, balls).

Point-based ANN indices assume queries and database entries are single vectors.
Region embeddings -- axis-aligned boxes, balls -- represent concepts as volumes
in embedding space. precinct bridges this gap by indexing region centers in an
HNSW graph and reranking candidates with the true point-to-region distance.

## Install

```toml
[dependencies]
precinct = "0.3"
```

or `cargo add precinct`.

## Usage

```rust
use precinct::{AxisBox, RegionIndex, SearchParams};

// Build an index of 2-d boxes
let mut idx = RegionIndex::new(2, Default::default()).unwrap();
idx.add(0, AxisBox::new(vec![0.0, 0.0], vec![1.0, 1.0]));
idx.add(1, AxisBox::new(vec![5.0, 5.0], vec![6.0, 6.0]));
idx.add(2, AxisBox::new(vec![10.0, 10.0], vec![11.0, 11.0]));
idx.build().unwrap();

// Search: retrieve 1 nearest region to a query point
let results = idx.search(&[0.5, 0.5], 1, Default::default()).unwrap();
assert_eq!(results[0].0, 0);   // region id
assert_eq!(results[0].1, 0.0); // distance (inside the box)
```

`SearchParams::overretrieve` controls the over-retrieval factor (default 10x).
Increasing it trades query latency for recall.

## Updatable index (`store` feature)

`store::UpdatableIndex` wraps the region index in a durable, segmented store
([`segstore`](https://crates.io/crates/segstore)): incremental add/delete, a
write-ahead log, checkpoint, compaction, and crash recovery. Per-segment
`RegionIndex`es are cached by stable segment identity, so a mutation rebuilds
only the new or changed segments, not the whole corpus; segments are searched
and merged, and like the underlying HNSW the merged result is approximate.
Opt-in; the default build does not depend on segstore.

## Recall

Recall@k measured against an exhaustive point-to-region scan.

Real data, `examples/glove_concepts` (50K GloVe-6B-50d vectors clustered into
5,000 concept boxes, the bounding box of each cluster of related words):

| Over-retrieve | Recall@10 |
|---|---|
| 10x | 92.1% |
| 50x | 99.3% |

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

## License

MIT OR Apache-2.0
