//! End-to-end: trained subsume boxes -> heyting complex queries -> precinct
//! candidate pruning.
//!
//! The integration-spine proof: the same trained WordNet box checkpoint that
//! `wordnet_boxes` validates is here queried with multi-hop logic. heyting
//! composes the query (AND over hypernym hops in the Godel algebra); the
//! membership degrees come from the box-lattice conditional
//! (`Region::entailment_prob`, `P(child ⊑ ancestor)`); and precinct's
//! `subsumers_soft` proposes each hop's candidates through heyting's
//! `CandidateSource` seam, so the pruned path scores a handful of regions
//! instead of every entity. The example asserts pruned == dense on every
//! query it prints: the roadmap's Phase 1 gate.
//!
//! Data (gitignored): run `scripts/fetch_wordnet_boxes.sh`. Without it this
//! example prints instructions and exits 0.
//!
//! Run: cargo run --release --example heyting_queries

use heyting::{
    answer_query_topk, answer_query_topk_pruned, AtomicScorer, CandidateSource, Godel, Query,
    QueryConfig,
};
use precinct::{AxisBox, IndexParams, Region, RegionIndex, SearchParams};
use serde_json::Value;

const CHECKPOINT: &str = "data/wordnet_boxes.json";
const HYPERNYM: usize = 0;

/// Same hypernym subset the checkpoint was trained on (see wordnet_boxes).
const EDGES: &str = include_str!("wordnet_edges.txt");

/// Dense scorer over trained boxes: the degree of `e` as an answer to
/// `(anchor, hypernym, ?)` is the box-lattice conditional
/// `P(anchor ⊑ e) = vol(e ∩ anchor) / vol(anchor)`. The anchor itself is
/// excluded (everything trivially entails itself).
struct BoxKg {
    boxes: Vec<AxisBox>,
}

impl BoxKg {
    fn degree(&self, anchor: usize, e: usize) -> f32 {
        if e == anchor {
            return 0.0;
        }
        self.boxes[e].entailment_prob(&self.boxes[anchor])
    }
}

impl AtomicScorer for BoxKg {
    fn num_entities(&self) -> usize {
        self.boxes.len()
    }

    fn project(&self, anchor: usize, relation: usize) -> Vec<f32> {
        let n = self.num_entities();
        if relation != HYPERNYM || anchor >= n {
            return vec![0.0; n];
        }
        (0..n).map(|e| self.degree(anchor, e)).collect()
    }

    fn project_subset(&self, anchor: usize, relation: usize, candidates: &[usize]) -> Vec<f32> {
        if relation != HYPERNYM || anchor >= self.num_entities() {
            return vec![0.0; candidates.len()];
        }
        candidates
            .iter()
            .map(|&e| {
                if e < self.num_entities() {
                    self.degree(anchor, e)
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// precinct as the candidate source: a hop's candidates are the regions that
/// softly subsume the anchor's box. `min_prob` trades candidate-set size for
/// recall, exactly like ANN over-retrieval.
struct SubsumerCandidates<'a> {
    index: &'a RegionIndex<AxisBox>,
    boxes: &'a [AxisBox],
    min_prob: f32,
}

impl CandidateSource for SubsumerCandidates<'_> {
    fn candidates(&self, anchor: usize, relation: usize) -> Option<Vec<usize>> {
        if relation != HYPERNYM || anchor >= self.boxes.len() {
            return Some(vec![]);
        }
        let params = SearchParams {
            ef: 64,
            overretrieve: 16,
        };
        let found = self
            .index
            .subsumers_soft(&self.boxes[anchor], self.min_prob, params)
            .unwrap_or_default();
        Some(
            found
                .into_iter()
                .map(|(id, _)| id as usize)
                .filter(|&id| id != anchor)
                .collect(),
        )
    }
}

fn main() {
    let path = std::env::var("WORDNET_BOXES").unwrap_or_else(|_| CHECKPOINT.to_string());
    let Some(boxes_by_idx) = load_checkpoint(&path) else {
        eprintln!("Trained WordNet boxes not found at {path}.");
        eprintln!("Build with: scripts/fetch_wordnet_boxes.sh");
        return; // data-gated: a clean no-op when the dataset is absent.
    };

    // Reproduce subsume's entity interning (first appearance, head then tail).
    let mut id_of: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut name_of: Vec<String> = Vec::new();
    for line in EDGES.lines() {
        for tok in line.split_whitespace().take(2) {
            if !id_of.contains_key(tok) {
                id_of.insert(tok.to_string(), name_of.len());
                name_of.push(tok.to_string());
            }
        }
    }

    let Some(boxes): Option<Vec<AxisBox>> = (0..name_of.len())
        .map(|i| boxes_by_idx.get(&(i as u32)).cloned())
        .collect()
    else {
        eprintln!("Checkpoint is missing boxes for some interned entities; re-fetch.");
        return;
    };

    let dim = boxes[0].dim();
    let mut index = RegionIndex::<AxisBox>::new(dim, IndexParams::default()).expect("index");
    for (i, b) in boxes.iter().enumerate() {
        index.add(i as u32, b.clone()).expect("add");
    }
    index.build().expect("build");
    println!(
        "Loaded {} trained WordNet concept boxes (dim {dim}).\n",
        boxes.len()
    );

    let kg = BoxKg { boxes };
    let source = SubsumerCandidates {
        index: &index,
        boxes: &kg.boxes,
        min_prob: 0.01,
    };
    let config = QueryConfig::default();
    let id = |name: &str| *id_of.get(name).expect("known concept");

    let queries: Vec<(String, Query)> = vec![
        (
            "1p  dog.n.01 is_a ?".into(),
            Query::anchor(id("dog.n.01"), HYPERNYM),
        ),
        (
            "2p  dog.n.01 is_a ∘ is_a ?".into(),
            Query::anchor(id("dog.n.01"), HYPERNYM).then(HYPERNYM),
        ),
        (
            "2i  (dog is_a ?) AND (cat is_a ?)".into(),
            Query::intersection(vec![
                Query::anchor(id("dog.n.01"), HYPERNYM),
                Query::anchor(id("cat.n.01"), HYPERNYM),
            ]),
        ),
        (
            "2i  (dog is_a ?) AND (eagle is_a ?)".into(),
            Query::intersection(vec![
                Query::anchor(id("dog.n.01"), HYPERNYM),
                Query::anchor(id("eagle.n.01"), HYPERNYM),
            ]),
        ),
    ];

    let k = 3;
    let mut disagreements = 0usize;
    for (label, q) in &queries {
        let dense = answer_query_topk::<Godel>(&kg, q, &config, k);
        let pruned = answer_query_topk_pruned::<Godel>(&kg, &source, q, &config, k);

        println!("{label}");
        for (rank, (e, d)) in dense.iter().enumerate() {
            let mark = match pruned.get(rank) {
                Some((_, pd)) if (pd - d).abs() < 1e-6 => ' ',
                _ => '!',
            };
            println!("  {}{} {:<20} {:.3}", mark, rank + 1, name_of[*e], d);
        }
        // Equivalence under ties: Godel min saturates whole cohorts to the
        // same degree (a 2p hop is bottlenecked by hop 1), and dense and
        // pruned may break those ties differently. The gate is therefore:
        // (a) the two top-k degree profiles match, and (b) each pruned
        // answer's degree is its true dense degree, i.e. nothing better was
        // dropped and nothing was invented.
        let dense_all = heyting::answer_query::<Godel>(&kg, q, &config);
        let profile_matches = dense.len() == pruned.len()
            && dense
                .iter()
                .zip(pruned.iter())
                .all(|((_, dd), (_, pd))| (dd - pd).abs() < 1e-6);
        let degrees_genuine = pruned
            .iter()
            .all(|(e, pd)| (dense_all[*e] - pd).abs() < 1e-6);
        if !(profile_matches && degrees_genuine) {
            disagreements += 1;
            println!(
                "  pruned diverges: {:?}",
                pruned
                    .iter()
                    .map(|(e, d)| (name_of[*e].as_str(), *d))
                    .collect::<Vec<_>>()
            );
        } else if dense
            .iter()
            .map(|(e, _)| e)
            .ne(pruned.iter().map(|(e, _)| e))
        {
            println!("  (tie-order differs; degree profiles identical)");
        }
        println!();
    }

    println!(
        "pruned-vs-dense top-{k} agreement: {}/{} queries",
        queries.len() - disagreements,
        queries.len()
    );
    println!(
        "(dense scores all {} entities per hop; pruned scores only precinct's\n\
         soft-subsumer candidates, and the Godel min ranks general ancestors\n\
         high -- generality weighting is the consumer's choice via log_volume.)",
        kg.num_entities()
    );
    // The Phase 1 gate: index-served candidates must not change the answers.
    assert_eq!(disagreements, 0, "pruned top-{k} diverged from dense");
}

/// Parse a subsume box checkpoint (`{boxes: {idx: {mu, delta}}, dim}`).
fn load_checkpoint(path: &str) -> Option<std::collections::BTreeMap<u32, AxisBox>> {
    let text = std::fs::read_to_string(path).ok()?;
    let doc: Value = serde_json::from_str(&text).ok()?;
    let boxes = doc.get("boxes")?.as_object()?;
    let mut out = std::collections::BTreeMap::new();
    for (k, v) in boxes {
        let id: u32 = k.parse().ok()?;
        let mu = as_f32_vec(v.get("mu")?)?;
        let delta = as_f32_vec(v.get("delta")?)?;
        out.insert(id, AxisBox::from_mu_delta(mu, delta));
    }
    (!out.is_empty()).then_some(out)
}

fn as_f32_vec(v: &Value) -> Option<Vec<f32>> {
    v.as_array()?
        .iter()
        .map(|x| x.as_f64().map(|f| f as f32))
        .collect()
}
