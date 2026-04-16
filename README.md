# precinct

Approximate nearest-neighbor search over region embeddings (boxes, balls).

Point-based ANN indices assume queries and database entries are single vectors.
Region embeddings -- axis-aligned boxes, balls -- represent concepts as volumes
in embedding space. precinct bridges this gap by indexing region centers in an
HNSW graph and reranking candidates with the true point-to-region distance.

## Install

```toml
[dependencies]
precinct = "0.2"
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

## Recall

Recall@10 measured against exhaustive ground truth on synthetic box datasets.

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
