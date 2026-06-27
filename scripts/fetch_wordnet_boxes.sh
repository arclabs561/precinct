#!/usr/bin/env bash
# Build the trained WordNet box checkpoint for examples/wordnet_boxes.
# Uses subsume's `save_checkpoint` example (a sibling crate). Writes to data/
# (gitignored). No-op if the checkpoint is already present.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

OUT="data/wordnet_boxes.json"
if [ -f "$OUT" ]; then
  echo "Already present: $OUT"
  exit 0
fi

SUBSUME="${SUBSUME_DIR:-../subsume}"
if [ ! -d "$SUBSUME" ]; then
  echo "subsume crate not found at $SUBSUME (set SUBSUME_DIR)."
  echo "It trains the box checkpoint this example reads."
  exit 1
fi

echo "Training the WordNet box checkpoint via subsume ..."
( cd "$SUBSUME" && cargo run -p subsume --example save_checkpoint --release )
cp "$SUBSUME/pretrained/wordnet_subset.json" "$OUT"
echo "Done: $OUT"
