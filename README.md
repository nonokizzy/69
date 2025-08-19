# LOFTAI — Pattern Mining & Backtesting

Tools:
- `patterns.miner` — cluster repeating motifs from 1-minute OHLCV.
- `patterns.cluster_leaderboard` — backtest each cluster (long/short) with simple 10-tick target / 50-tick stop and produce a leaderboard.
- `patterns.similar_scan` — stream over the series and trade when a window is similar to any cluster prototype (cosine or euclidean).

> **Data:** Put your cleaned minute data at `data/processed/mnq_1min_rth.csv` (DatetimeIndex, columns: open, high, low, close[, volume]).

## Quickstart

```bash
python -m patterns.miner \
  --data data/processed/mnq_1min_rth.csv \
  --out artifacts/patterns \
  --window 30 --stride 1 --clusters 20

python -m patterns.cluster_leaderboard \
  --data data/processed/mnq_1min_rth.csv \
  --clusters-json artifacts/patterns/clusters.json \
  --out artifacts/patterns/leaderboards \
  --target-ticks 10 --stop-ticks 50 \
  --tick-size 0.25 --dollars-per-point 2.0 \
  --contracts 2 --commission-per-fill 1.20 --slippage-ticks 1 \
  --sessions "09:35-16:00" --top 30

python -m patterns.similar_scan \
  --data data/processed/mnq_1min_rth.csv \
  --clusters-json artifacts/patterns/clusters.json \
  --out artifacts/patterns/simscan_aggressive \
  --metric cosine --sim-thresh 0.20 \
  --sessions "09:35-16:00" --cooldown-min 0 --both-sides --debug \
  --target-ticks 10 --stop-ticks 50 \
  --tick-size 0.25 --dollars-per-point 2.0 \
  --contracts 2 --commission-per-fill 1.20 --slippage-ticks 1


---

### 4) Package init
```bash
cat > src/patterns/__init__.py << 'EOF'
# loftai-patterns package
