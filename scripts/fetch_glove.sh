#!/usr/bin/env bash
# Fetch GloVe 6B word vectors for examples/glove_concepts.
# Downloads to data/ (gitignored). Idempotent: skips if the 50d file exists.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

OUT="data/glove.6B.50d.txt"
if [ -f "$OUT" ]; then
  echo "Already present: $OUT"
  exit 0
fi

ZIP="data/glove.6B.zip"
URL="https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
echo "Downloading GloVe 6B (~862 MB) from $URL ..."
curl -fSL "$URL" -o "$ZIP"
echo "Extracting 50d vectors ..."
unzip -o "$ZIP" glove.6B.50d.txt -d data/
rm -f "$ZIP"
echo "Done: $OUT"
