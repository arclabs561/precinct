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
        (
            // A numeric-literal constraint (heyting 0.6 Query::given): the
            // attribute is taxonomy depth derived from the edge list, and the
            // constraint "depth <= 3" keeps only general concepts. The
            // planner also uses the constraint's small support to restrict
            // the relation hop's scoring.
            "lit (dog is_a ?) AND depth(?) <= 3".into(),
            Query::intersection(vec![
                Query::anchor(id("dog.n.01"), HYPERNYM),
                Query::given(
                    depth_from_root(&id_of)
                        .iter()
                        .map(|d| match d {
                            Some(d) if *d <= 3 => 1.0,
                            _ => 0.0,
                        })
                        .collect(),
                ),
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

    conformal_section(&kg, &id_of, &config);
}

/// Conformal answer sets over the same trained boxes: calibrate "child is_a
/// -> parent" nonconformities on part of the edge list, then check on the
/// held-out edges that the guaranteed sets actually contain the true parent.
fn conformal_section(
    kg: &BoxKg,
    id_of: &std::collections::HashMap<String, usize>,
    config: &QueryConfig,
) {
    let pairs: Vec<(Query, usize)> = EDGES
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let (child, parent) = (it.next()?, it.next()?);
            Some((Query::anchor(id_of[child], HYPERNYM), id_of[parent]))
        })
        .collect();

    // Deterministic interleaved split: every 3rd edge held out for testing.
    let (calibration, test): (Vec<_>, Vec<_>) =
        pairs.into_iter().enumerate().partition(|(i, _)| i % 3 != 0);
    let calibration: Vec<(Query, usize)> = calibration.into_iter().map(|(_, p)| p).collect();
    let test: Vec<(Query, usize)> = test.into_iter().map(|(_, p)| p).collect();

    let alpha = 0.2;
    let threshold =
        heyting::calibrate::<Godel>(kg, &calibration, config, alpha).expect("calibrate");
    let coverage = heyting::empirical_coverage::<Godel>(kg, &test, config, &threshold);
    let mean_set_size = test
        .iter()
        .map(|(q, _)| heyting::answer_set::<Godel>(kg, q, config, &threshold).len())
        .sum::<usize>() as f32
        / test.len() as f32;

    println!(
        "\nconformal answer sets (alpha = {alpha}): calibrated qhat = {:.3} on {} edges;\n\
         held-out coverage {:.0}% over {} edges (nominal {:.0}%), mean set size {:.1} of {}\n\
         entities. The guarantee is marginal and assumes exchangeable queries; hypernym\n\
         edges at different taxonomy depths only approximate that, so read the coverage\n\
         as an empirical check, not a certificate.",
        threshold.qhat,
        threshold.n_calibration,
        coverage * 100.0,
        test.len(),
        (1.0 - alpha) * 100.0,
        mean_set_size,
        kg.num_entities(),
    );
    // Phase 2 gate: held-out coverage at (or above) the nominal level, with
    // slack for the small test split's binomial noise.
    assert!(
        coverage >= 1.0 - alpha - 0.15,
        "coverage {coverage} far below nominal {}",
        1.0 - alpha
    );
}

/// Hop distance from each concept to its taxonomy root (a node with no
/// outgoing hypernym edge), following the edge list; `None` for concepts
/// with a parent missing from the subset (does not occur in this data).
fn depth_from_root(id_of: &std::collections::HashMap<String, usize>) -> Vec<Option<usize>> {
    let mut parent: Vec<Option<usize>> = vec![None; id_of.len()];
    for line in EDGES.lines() {
        let mut it = line.split_whitespace();
        if let (Some(c), Some(p)) = (it.next(), it.next()) {
            parent[id_of[c]] = Some(id_of[p]);
        }
    }
    (0..id_of.len())
        .map(|mut e| {
            let mut depth = 0usize;
            while let Some(p) = parent[e] {
                depth += 1;
                e = p;
                if depth > id_of.len() {
                    return None; // cycle guard; hypernym edges should be acyclic.
                }
            }
            Some(depth)
        })
        .collect()
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
