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
const EDGES: &str = "\
dog.n.01 canine.n.02
canine.n.02 carnivore.n.01
carnivore.n.01 placental.n.01
placental.n.01 mammal.n.01
mammal.n.01 vertebrate.n.01
vertebrate.n.01 chordate.n.01
chordate.n.01 animal.n.01
animal.n.01 organism.n.01
organism.n.01 entity.n.01
cat.n.01 feline.n.01
feline.n.01 carnivore.n.01
wolf.n.01 canine.n.02
fox.n.01 canine.n.02
lion.n.01 feline.n.01
tiger.n.01 feline.n.01
horse.n.01 equine.n.01
equine.n.01 placental.n.01
eagle.n.01 bird_of_prey.n.01
bird_of_prey.n.01 bird.n.01
bird.n.01 vertebrate.n.01
sparrow.n.01 passerine.n.01
passerine.n.01 bird.n.01
salmon.n.01 fish.n.01
fish.n.01 vertebrate.n.01
trout.n.01 fish.n.01
oak.n.01 tree.n.01
tree.n.01 plant.n.02
plant.n.02 organism.n.01
pine.n.01 tree.n.01
rose.n.01 flower.n.01
flower.n.01 plant.n.02
tulip.n.01 flower.n.01
car.n.01 vehicle.n.01
vehicle.n.01 artifact.n.01
artifact.n.01 entity.n.01
truck.n.01 vehicle.n.01
bicycle.n.01 vehicle.n.01
whale.n.01 placental.n.01
dolphin.n.01 placental.n.01
snake.n.01 reptile.n.01
reptile.n.01 vertebrate.n.01
lizard.n.01 reptile.n.01
penguin.n.01 bird.n.01
bat.n.01 placental.n.01
spider.n.01 arthropod.n.01
arthropod.n.01 animal.n.01";

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
    let (mut subsumer_hits, mut center_hits, mut total) = (0usize, 0usize, 0usize);
    for (child, parent) in &edges {
        let (Some(&cid), Some(&pid)) = (id_of.get(child), id_of.get(parent)) else {
            continue;
        };
        let (Some(cbox), Some(_)) = (boxes_by_idx.get(&cid), boxes_by_idx.get(&pid)) else {
            continue;
        };
        total += 1;
        let subs = idx.subsumers(cbox, params()).unwrap_or_default();
        if subs.contains(&pid) {
            subsumer_hits += 1;
        }
        // Also: is the parent among the regions enclosing the child's center?
        let enc = idx.containing(cbox.center(), params()).unwrap_or_default();
        if enc.contains(&pid) {
            center_hits += 1;
        }
    }
    println!(
        "hypernym-ancestor recall over {total} edges:\n  membership  (parent box encloses child center): {:.0}%\n  strict containment (parent box ⊇ child box):     {:.0}%",
        center_hits as f64 / total as f64 * 100.0,
        subsumer_hits as f64 / total as f64 * 100.0,
    );
    println!(
        "The trained Gumbel boxes nest *softly* -- a child's center lands inside its\n\
         parent, but the child's full box pokes outside -- so membership recovers the\n\
         is-a ancestor while strict box-containment does not. For trained (soft)\n\
         embeddings, `containing` is the right ancestor query, not `subsumers`."
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
