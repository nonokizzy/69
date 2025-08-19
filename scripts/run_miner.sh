#!/usr/bin/env bash
set -euo pipefail
DATA="${1:-data/processed/mnq_1min_rth.csv}"
OUT="artifacts/patterns"
python -m src.patterns.miner \
  --data "$DATA" \
  --out "$OUT" \
  --window 30 \
  --stride 1 \
  --clusters 20
