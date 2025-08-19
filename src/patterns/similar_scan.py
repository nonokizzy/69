from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from .utils import load_ohlcv_csv, ensure_dt_index, to_returns, z_norm, sliding_windows, parse_sessions, in_any_session

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    da = np.linalg.norm(a); db = np.linalg.norm(b)
    if da < 1e-12 or db < 1e-12: return 0.0
    return float(np.dot(a,b)/(da*db))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--clusters-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim-thresh", type=float, default=0.40, help="min cosine similarity to trigger")
    ap.add_argument("--sessions", type=str, default="09:35-16:00")
    ap.add_argument("--cooldown-min", type=int, default=2)
    ap.add_argument("--target-ticks", type=int, default=10)
    ap.add_argument("--stop-ticks",   type=int, default=50)
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--dollars-per-point", type=float, default=2.0)
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--commission-per-fill", type=float, default=1.20)
    ap.add_argument("--slippage-ticks", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df = ensure_dt_index(load_ohlcv_csv(Path(args.data)))

    js = json.loads(Path(args.clusters_json).read_text())
    centroids = js.get("centroids", {})
    W = int(js["meta"]["window"])

    close = df["close"].values.astype(float)
    r = to_returns(close.reshape(-1,1))
    X, starts = sliding_windows(r, W, 1)
    if len(starts)==0:
        (out_dir/"summary.txt").write_text("No trades. Empty windows.\n")
        return
    Xz = np.stack([z_norm(x) for x in X], axis=0)

    sessions = parse_sessions(args.sessions)
    slippage_pts = args.slippage_ticks * args.tick_size

    trades=[]
    last_entry_ts_by_day = {}

    # try both long/short per match; you can later restrict by centroid sign if you want
    for cid_str, cvec in centroids.items():
        cid = int(cid_str)
        c = np.asarray(cvec, dtype=float)
        cz = z_norm(c)
        for win_idx, st in enumerate(starts):
            et = st + W
            if et+1 >= len(df): 
                continue
            # time filters
            entry_ts = df.index[et+1]
            if sessions:
                day_df = df.loc[str(entry_ts.date())]
                if day_df.empty: 
                    continue
                mask = in_any_session(day_df.index, sessions)
                if entry_ts not in day_df[mask].index: 
                    continue
                # cooldown per day
                last_ts = last_entry_ts_by_day.get(entry_ts.date())
                if last_ts is not None:
                    delta = (entry_ts - last_ts).total_seconds()/60.0
                    if delta < args.cooldown_min:
                        continue

            sim = cosine(Xz[win_idx], cz)
            if sim < args.sim_thresh:
                continue

            # build two trades: long and short
            row = df.loc[entry_ts]
            base_entry = float(row["close"])

            for side in ("long","short"):
                entry = base_entry + (slippage_pts if side=="long" else -slippage_pts)
                if side=="long":
                    target = entry + args.target_ticks*args.tick_size
                    stop   = entry - args.stop_ticks*args.tick_size
                else:
                    target = entry - args.target_ticks*args.tick_size
                    stop   = entry + args.stop_ticks*args.tick_size

                outcome=None; exit_price=None; exit_time=None
                after = df.loc[entry_ts:].iloc[1:]
                for ts, r2 in after.iterrows():
                    hi, lo = float(r2["high"]), float(r2["low"])
                    if side=="long":
                        stop_hit=(lo <= stop); targ_hit=(hi >= target)
                        if stop_hit and targ_hit: outcome, exit_price="AMBIGUOUS_SAME_BAR", stop
                        elif stop_hit:            outcome, exit_price="LOSS_STOP", stop
                        elif targ_hit:            outcome, exit_price="WIN_TARGET", target
                    else:
                        stop_hit=(hi >= stop); targ_hit=(lo <= target)
                        if stop_hit and targ_hit: outcome, exit_price="AMBIGUOUS_SAME_BAR", stop
                        elif stop_hit:            outcome, exit_price="LOSS_STOP", stop
                        elif targ_hit:            outcome, exit_price="WIN_TARGET", target
                    if outcome:
                        exit_time = ts
                        break
                if not outcome:
                    continue

                pnl_points = (exit_price - entry) if side=="long" else (entry - exit_price)
                pnl_cash = pnl_points * args.dollars_per_point * args.contracts - (2 * args.commission_per_fill * args.contracts)
                trades.append(dict(cluster_id=cid, similarity=sim, side=side, entry_time=entry_ts, exit_time=exit_time, entry=entry, exit_price=exit_price, outcome=outcome, pnl=pnl_cash))
                last_entry_ts_by_day[entry_ts.date()] = entry_ts  # cooldown marker

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        (out_dir/"summary.txt").write_text("No trades. Wrote empty summary.\n")
        return

    tdf.sort_values("entry_time").to_csv(out_dir/"similar_trades.csv", index=False)

    wins  = (tdf["outcome"]=="WIN_TARGET").sum()
    losses= (tdf["outcome"]=="LOSS_STOP").sum()
    wr = wins / max(1,(wins+losses))
    net = tdf["pnl"].sum()
    dd  = (tdf["pnl"].cumsum().cummax() - tdf["pnl"].cumsum()).max()
    days= pd.to_datetime(tdf["entry_time"]).dt.date.nunique()
    summary = f"Trades: {len(tdf)} | Wins: {wins} | Losses: {losses} | Winrate: {wr:.1%} | Net: {net:.2f} | MaxDD: {dd:.2f} | Days: {days} | Avg/day: {len(tdf)/max(1,days):.2f}"
    (out_dir/"summary.txt").write_text(summary + "\n")

    print(summary)
    print(f"Wrote {out_dir/'similar_trades.csv'} and {out_dir/'summary.txt'}")

if __name__ == "__main__":
    main()
