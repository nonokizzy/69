#!/usr/bin/env bash
set -euo pipefail
DATA="${1:-data/processed/mnq_1min_rth.csv}"
CLUSTERS_JSON="${2:-artifacts/patterns/clusters.json}"
OUT="artifacts/patterns/leaderboards"
python -m src.patterns.cluster_leaderboard \
  --data "$DATA" \
  --clusters-json "$CLUSTERS_JSON" \
  --out "$OUT" \
  --target-ticks 10 \
  --stop-ticks 50 \
  --tick-size 0.25 \
  --dollars-per-point 2.0 \
  --contracts 2 \
  --commission-per-fill 1.20 \
  --slippage-ticks 1 \
  --sessions "09:35-16:00" \
  --top 50
