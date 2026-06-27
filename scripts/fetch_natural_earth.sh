#!/usr/bin/env bash
# Fetch Natural Earth country polygons for examples/geo_regions.
# Downloads to data/ (gitignored). Public domain (Natural Earth, 110m admin-0).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

OUT="data/ne_countries.json"
if [ -f "$OUT" ]; then
  echo "Already present: $OUT"
  exit 0
fi

URL="https://raw.githubusercontent.com/martynafford/natural-earth-geojson/master/110m/cultural/ne_110m_admin_0_countries.json"
echo "Downloading Natural Earth countries from $URL ..."
curl -fSL "$URL" -o "$OUT"
echo "Done: $OUT ($(wc -c <"$OUT") bytes)"
