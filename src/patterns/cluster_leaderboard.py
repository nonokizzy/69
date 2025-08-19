from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from .utils import load_ohlcv_csv, ensure_dt_index, parse_sessions, in_any_session

def simulate_trades(df: pd.DataFrame,
                    windows: list[dict],
                    side: str,
                    target_ticks: int,
                    stop_ticks: int,
                    tick_size: float,
                    dollars_per_point: float,
                    contracts: int,
                    commission_per_fill: float,
                    slippage_ticks: int,
                    sessions_spec: str) -> pd.DataFrame:
    """Enter at next bar after end_idx, exit at target/stop, no lookahead."""
    sessions = parse_sessions(sessions_spec)
    slippage_pts = slippage_ticks * tick_size
    rec=[]
    for w in windows:
        end_idx = int(w["end_idx"])
        if end_idx+1 >= len(df): 
            continue
        entry_ts = df.index[end_idx+1]
        day_df = df.loc[str(entry_ts.date())]
        if day_df.empty: 
            continue
        if sessions:
            mask = in_any_session(day_df.index, sessions)
            day_df = day_df[mask]
            if day_df.empty or entry_ts not in day_df.index:
                continue

        row = df.loc[entry_ts]
        entry = float(row["close"])
        # apply 1-tick slippage in adverse direction
        if side == "long":
            entry += slippage_pts
            target = entry + target_ticks * tick_size
            stop   = entry - stop_ticks   * tick_size
        else:
            entry -= slippage_pts
            target = entry - target_ticks * tick_size
            stop   = entry + stop_ticks   * tick_size

        outcome = None; exit_price=None; exit_time=None
        # walk forward bars AFTER entry
        after = df.loc[entry_ts:].iloc[1:]
        for ts, r in after.iterrows():
            hi, lo = float(r["high"]), float(r["low"])
            if side=="long":
                stop_hit = (lo <= stop); targ_hit=(hi >= target)
                if stop_hit and targ_hit: outcome, exit_price="AMBIGUOUS_SAME_BAR", stop
                elif stop_hit:            outcome, exit_price="LOSS_STOP", stop
                elif targ_hit:            outcome, exit_price="WIN_TARGET", target
            else:
                stop_hit = (hi >= stop); targ_hit=(lo <= target)
                if stop_hit and targ_hit: outcome, exit_price="AMBIGUOUS_SAME_BAR", stop
                elif stop_hit:            outcome, exit_price="LOSS_STOP", stop
                elif targ_hit:            outcome, exit_price="WIN_TARGET", target

            if outcome:
                exit_time = ts
                break
        if not outcome:
            continue

        pnl_points = (exit_price - entry) if side=="long" else (entry - exit_price)
        pnl_cash = pnl_points * dollars_per_point * contracts - (2 * commission_per_fill * contracts)

        rec.append(dict(
            cluster_id=int(w["cluster_id"]),
            side=side, entry_time=entry_ts, exit_time=exit_time,
            entry=entry, exit_price=exit_price, outcome=outcome, pnl=pnl_cash
        ))
    return pd.DataFrame(rec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--clusters-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-ticks", type=int, default=10)
    ap.add_argument("--stop-ticks",   type=int, default=50)
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--dollars-per-point", type=float, default=2.0)
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--commission-per-fill", type=float, default=1.20)
    ap.add_argument("--slippage-ticks", type=int, default=1)
    ap.add_argument("--sessions", type=str, default="09:35-16:00")
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df = ensure_dt_index(load_ohlcv_csv(Path(args.data)))

    js = json.loads(Path(args.clusters_json).read_text())
    hits_dict = js["hits"] if "hits" in js else {}
    windows=[]
    for cid, arr in hits_dict.items():
        for h in arr:
            windows.append(dict(cluster_id=int(cid), **h))
    if not windows:
        raise SystemExit("No windows parsed from clusters.json (hits empty).")

    # per-cluster simulate both sides
    trades_all=[]
    for cid in sorted(set(w["cluster_id"] for w in windows)):
        subset = [dict(cluster_id=cid, **{k:v for k,v in w.items() if k!="cluster_id"}) for w in windows if w["cluster_id"]==cid]
        t_long  = simulate_trades(df, subset, "long",
                                  args.target_ticks, args.stop_ticks, args.tick_size,
                                  args.dollars_per_point, args.contracts, args.commission_per_fill,
                                  args.slippage_ticks, args.sessions)
        t_short = simulate_trades(df, subset, "short",
                                  args.target_ticks, args.stop_ticks, args.tick_size,
                                  args.dollars_per_point, args.contracts, args.commission_per_fill,
                                  args.slippage_ticks, args.sessions)
        trades_all.append(t_long); trades_all.append(t_short)

    trades = pd.concat([t for t in trades_all if not t.empty], ignore_index=True) if trades_all else pd.DataFrame(columns=["cluster_id","side"])
    trades.to_csv(out_dir/"trades.csv", index=False)

    # leaderboard
    if trades.empty:
        print("No trades produced.")
        return

    def agg(g: pd.DataFrame):
        wins  = (g["outcome"]=="WIN_TARGET").sum()
        losses= (g["outcome"]=="LOSS_STOP").sum()
        amb   = (g["outcome"]=="AMBIGUOUS_SAME_BAR").sum()
        wr = wins / max(1,(wins+losses))
        net = g["pnl"].sum()
        dd = (g["pnl"].cumsum().cummax() - g["pnl"].cumsum()).max()
        days = g["entry_time"].dt.date.nunique()
        tpd = len(g)/max(1,days)
        return pd.Series(dict(trades=len(g), wins=wins, losses=losses, ambiguous=amb, winrate=wr, avg_pnl=g["pnl"].mean(), total_pnl=net, trades_per_day=tpd, max_dd=dd))

    lb = trades.groupby(["cluster_id","side"], as_index=False).apply(agg).reset_index(drop=True)
    lb_sorted = lb.sort_values(["total_pnl","winrate","trades"], ascending=[False,False,False]).head(args.top)
    lb_sorted.to_csv(out_dir/"leaderboard.csv", index=False)

    print("\n=== Leaderboard (top {} shown) ===".format(args.top))
    print(lb_sorted.to_string(index=False))
    print(f"\nSaved leaderboard to {out_dir/'leaderboard.csv'}")

if __name__ == "__main__":
    main()
