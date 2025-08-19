#!/usr/bin/env bash
set -euo pipefail
DATA="${1:-data/processed/mnq_1min_rth.csv}"
CLUSTERS_JSON="${2:-artifacts/patterns/clusters.json}"

# conservative / medium / aggressive
python -m src.patterns.similar_scan \
  --data "$DATA" \
  --clusters-json "$CLUSTERS_JSON" \
  --out artifacts/patterns/simscan_conservative \
  --metric cosine --sim-thresh 0.45 \
  --sessions "09:35-16:00" --cooldown-min 5 \
  --target-ticks 10 --stop-ticks 50 --tick-size 0.25 \
  --dollars-per-point 2.0 --contracts 2 \
  --commission-per-fill 1.20 --slippage-ticks 1

python -m src.patterns.similar_scan \
  --data "$DATA" \
  --clusters-json "$CLUSTERS_JSON" \
  --out artifacts/patterns/simscan_medium \
  --metric cosine --sim-thresh 0.35 \
  --sessions "09:35-16:00" --cooldown-min 2 \
  --target-ticks 10 --stop-ticks 50 --tick-size 0.25 \
  --dollars-per-point 2.0 --contracts 2 \
  --commission-per-fill 1.20 --slippage-ticks 1

python -m src.patterns.similar_scan \
  --data "$DATA" \
  --clusters-json "$CLUSTERS_JSON" \
  --out artifacts/patterns/simscan_aggressive \
  --metric cosine --sim-thresh 0.25 \
  --sessions "09:35-16:00" --cooldown-min 0 \
  --target-ticks 10 --stop-ticks 50 --tick-size 0.25 \
  --dollars-per-point 2.0 --contracts 2 \
  --commission-per-fill 1.20 --slippage-ticks 1
