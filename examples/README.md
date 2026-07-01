# precinct examples

Examples are runnable from the repo root. Output excerpts below are real,
captured from release runs except for `glove_concepts`, which is the heavier
GloVe clustering benchmark.

## Which example should I run?

| I want to... | Example | Notes |
|---|---|---|
| Search real geographic bounding boxes | `geo_regions` | Data-gated |
| Query trained WordNet concept boxes | `wordnet_boxes` | Data-gated |
| Measure synthetic center-ANN recall gaps | `recall_gap` | Always runnable, slower |
| Build concept boxes from GloVe clusters | `glove_concepts` | Data-gated, heavier |

## Real Regions

### `geo_regions`: when does surface distance differ from center distance?

Loads Natural Earth country polygons, converts each country to a longitude/latitude
bounding box, and queries nearest regions by point-to-box distance.

```bash
scripts/fetch_natural_earth.sh
cargo run --release --example geo_regions
```

```text
Loaded 177 country regions.
  South Pacific (off Chile)    -> Chile (14.4), Brazil (16.1), Argentina (16.6)
  central Europe               -> Germany (0.0), Russia (0.0), Austria (1.0)
  Bay of Bengal                -> India (0.0), China (3.2), Myanmar (4.3)
  Sahara interior              -> Niger (0.0), Libya (0.0), Algeria (0.0)
      (nearest by center would be Libya, not Niger)
recall@3 over a 136-point world grid: 92.9%
```

If the dataset is absent, the example exits 0 and prints the fetch command.

### `wordnet_boxes`: do trained boxes recover hypernym ancestors?

Loads trained WordNet boxes, then compares membership, soft subsumption, and
strict box containment for known child-parent edges.

```bash
scripts/fetch_wordnet_boxes.sh
cargo run --release --example wordnet_boxes
```

```text
Loaded 47 trained WordNet concept boxes (dim 16).
hypernym-ancestor recall over 46 edges:
  membership  (parent encloses child center):     98%
  soft subsumers (entailment_prob >= 0.3):       98%
  strict subsumers (parent box ⊇ child box):      0%
The trained Gumbel boxes nest *softly*: a child's center lands inside its
parent, but the child's full box pokes outside. So membership and the soft
subsumption query recover the is-a ancestor while strict box-containment
does not -- the index is sound; the trained embedding is soft.
```

If the checkpoint is absent, the example exits 0 and prints the fetch command.

## Recall Benchmarks

### `recall_gap`: how much over-retrieval does region reranking need?

Generates synthetic high-dimensional boxes, builds a region index, and compares
approximate search with exhaustive point-to-region ground truth.

```bash
cargo run --release --example recall_gap
```

```text
=== wide (d=128, w=0.5) (n=10000, dim=128, width_mean=0.50) ===
Build: 5517ms | Exhaustive: 190ms
  overretrieve=  1x  recall@10=0.4870  search=28ms (7142.9 qps)
  overretrieve= 10x  recall@10=0.9300  search=28ms (7142.9 qps)
  overretrieve= 50x  recall@10=0.9940  search=66ms (3030.3 qps)
  (ground truth: 0.0% of top-10 results contain the query point)

=== scale 50K (d=128, w=0.1) (n=50000, dim=128, width_mean=0.10) ===
Build: 41514ms | Exhaustive: 1333ms
  overretrieve=  1x  recall@10=0.7490  search=31ms (3225.8 qps)
  overretrieve= 50x  recall@10=0.9410  search=90ms (1111.1 qps)
```

This is a benchmark-style example. Timings vary by machine.

### `glove_concepts`: do region boxes help on real word-vector concepts?

Clusters the top 50K GloVe vectors into 5K concept boxes, then compares
precinct's region-aware search with a naive point-ANN over box centers.

```bash
scripts/fetch_glove.sh
cargo run --release --example glove_concepts
```

This is the heaviest example in the crate. The top-level README's recall table
records the current captured result for this benchmark.
