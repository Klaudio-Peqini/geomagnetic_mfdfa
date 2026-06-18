#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after copying scripts/analyze_short_geophysical_series.py
# into the repo's scripts/ folder.

python scripts/analyze_short_geophysical_series.py \
  --mode radon \
  --input data/radon/PE22111010052_LogData.txt \
  --out results_short/radon \
  --qmin -3 --qmax 3 --qstep 0.5 \
  --smin 8 \
  --min-segments 10 \
  --poly 1 \
  --series recommended \
  --n-surrogates 100

python scripts/analyze_short_geophysical_series.py \
  --mode geomag \
  --input data/geomagnetic/PADM2M.xlsx \
  --out results_short/PADM2M \
  --age-unit kyr \
  --qmin -4 --qmax 4 --qstep 0.5 \
  --smin 8 \
  --min-segments 10 \
  --poly 1 \
  --series recommended \
  --n-surrogates 100
