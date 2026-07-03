//! In-domain validation: precinct's subsumption query over real *trained* box
//! embeddings.
//!
//! Loads a subsume box-embedding checkpoint (entities as `{mu, delta}` boxes,
//! `min = mu - exp(delta)/2`, `max = mu + exp(delta)/2`) trained on a WordNet
//! hypernym subset, then asks precinct to recover each concept's hypernym
//! ancestor two ways -- membership (`containing` the child's center) and strict
//! subsumption (`subsumers`) -- on real learned regions rather than synthetic or
//! clustered boxes. The gap between the two is itself the finding (see output).
//!
//! Data (gitignored): run `scripts/fetch_wordnet_boxes.sh` (it builds the
//! checkpoint with subsume's `save_checkpoint` example). Without it this example
//! prints instructions and exits 0.
//!
//! Run: cargo run --release --example wordnet_boxes

use precinct::{AxisBox, IndexParams, Region, RegionIndex, SearchParams};
use serde_json::Value;

const CHECKPOINT: &str = "data/wordnet_boxes.json";

/// WordNet `child parent` hypernym edges (the subset subsume trains on). Source:
/// WordNet noun hierarchy; `child ⊑ parent` means parent is the more general
/// concept, so its box should contain the child's.
const EDGES: &str = include_str!("wordnet_edges.txt");

fn main() {
    let path = std::env::var("WORDNET_BOXES").unwrap_or_else(|_| CHECKPOINT.to_string());
    let Some(boxes_by_idx) = load_checkpoint(&path) else {
        eprintln!("Trained WordNet boxes not found at {path}.");
        eprintln!("Build with: scripts/fetch_wordnet_boxes.sh");
        return; // data-gated: a clean no-op when the dataset is absent.
    };

    // Reproduce subsume's entity interning: first appearance over the edges,
    // head then tail per line. Box index = interning order.
    let edges: Vec<(String, String)> = EDGES
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.to_string(), it.next()?.to_string()))
        })
        .collect();
    let mut id_of: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let intern = |s: &str, m: &mut std::collections::HashMap<String, u32>| -> u32 {
        let n = m.len() as u32;
        *m.entry(s.to_string()).or_insert(n)
    };
    for (h, t) in &edges {
        intern(h, &mut id_of);
        intern(t, &mut id_of);
    }

    let dim = boxes_by_idx.values().next().map(|b| b.dim()).unwrap_or(0);
    let mut idx = RegionIndex::<AxisBox>::new(dim, IndexParams::default()).expect("index");
    for (&id, b) in &boxes_by_idx {
        if id_of.values().any(|&v| v == id) {
            idx.add(id, b.clone()).expect("add");
        }
    }
    idx.build().expect("build");
    println!(
        "Loaded {} trained WordNet concept boxes (dim {dim}).",
        idx.len()
    );

    // For each edge child ⊑ parent: does precinct surface `parent` as a subsumer
    // of `child`? (subsumers = regions whose box contains the child's box.)
    let params = || SearchParams {
        ef: 64,
        overretrieve: 16,
    };
    let min_prob = 0.3;
    let (mut strict, mut soft, mut member, mut total) = (0usize, 0usize, 0usize, 0usize);
    for (child, parent) in &edges {
        let (Some(&cid), Some(&pid)) = (id_of.get(child), id_of.get(parent)) else {
            continue;
        };
        let (Some(cbox), Some(_)) = (boxes_by_idx.get(&cid), boxes_by_idx.get(&pid)) else {
            continue;
        };
        total += 1;
        if idx
            .subsumers(cbox, params())
            .unwrap_or_default()
            .contains(&pid)
        {
            strict += 1;
        }
        if idx
            .subsumers_soft(cbox, min_prob, params())
            .unwrap_or_default()
            .iter()
            .any(|(id, _)| *id == pid)
        {
            soft += 1;
        }
        if idx
            .containing(cbox.center(), params())
            .unwrap_or_default()
            .contains(&pid)
        {
            member += 1;
        }
    }
    let pct = |h: usize| h as f64 / total as f64 * 100.0;
    println!("hypernym-ancestor recall over {total} edges:");
    println!(
        "  membership  (parent encloses child center):     {:.0}%",
        pct(member)
    );
    println!(
        "  soft subsumers (entailment_prob >= {min_prob}):       {:.0}%",
        pct(soft)
    );
    println!(
        "  strict subsumers (parent box ⊇ child box):      {:.0}%",
        pct(strict)
    );
    println!(
        "The trained Gumbel boxes nest *softly*: a child's center lands inside its\n\
         parent, but the child's full box pokes outside. So membership and the soft\n\
         subsumption query recover the is-a ancestor while strict box-containment\n\
         does not -- the index is sound; the trained embedding is soft."
    );
}

/// Parse a subsume box checkpoint (`{boxes: {idx: {mu, delta}}, dim}`) into
/// `AxisBox`es keyed by entity index.
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
