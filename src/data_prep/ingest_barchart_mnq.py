from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

# Expected final columns: open, high, low, close, volume, (optional) open_interest, contract

# ---- helpers ----
def normalize_barchart_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map various Barchart header variants to our canonical names (case-insensitive).
    Handles intraday files where 'Close' may be 'Last' or 'Price'.
    """
    original_cols = list(df.columns)
    # Lowercase & strip the names for detection, but preserve df values
    lower_map = {c: c.strip().lower() for c in df.columns}

    # Candidate timestamp headers Barchart uses
    ts_candidates = ["timestamp", "date", "datetime", "time", "trade time", "trading day", "bar time"]

    # Start building a rename dict
    rename = {}

    # Timestamp
    for c, lc in lower_map.items():
        if lc in ts_candidates:
            rename[c] = "timestamp"
            break
    if "timestamp" not in rename.values():
        # Heuristic: first column if it looks like a time column
        c0 = df.columns[0]
        rename[c0] = "timestamp"

    # O/H/L/C-ish
    oh = { "open": ["open","opening price"],
           "high": ["high","high price"],
           "low" : ["low","low price"],
           "close": ["close","settlement","settle","last","price","close price"] }

    for target, aliases in oh.items():
        found = None
        for c, lc in lower_map.items():
            if lc in aliases:
                found = c; break
        if found:
            rename[found] = target

    # Volume
    vol_candidates = ["volume","total volume","vol"]
    for c, lc in lower_map.items():
        if lc in vol_candidates:
            rename[c] = "volume"
            break

    # Open interest (optional)
    oi_candidates = ["open interest","open_interest","oi"]
    for c, lc in lower_map.items():
        if lc in oi_candidates:
            rename[c] = "open_interest"
            break

    # Apply rename
    df = df.rename(columns=rename)

    # Basic checks & minimal fill-ins
    must_have = ["timestamp","open","high","low","close"]
    missing = [m for m in must_have if m not in df.columns]
    if missing:
        raise ValueError(f"After normalization missing required columns {missing}. "
                         f"Original columns={original_cols} | rename={rename}")

    if "volume" not in df.columns: df["volume"] = 0
    # keep only the columns we care about (plus any extra left over)
    return df

def parse_barchart_file(path: Path, source_tz: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_barchart_columns(df)

    # timestamp to UTC-naive then convert to wall clock (we'll localize later)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # numeric coercion
    for c in ["open","high","low","close","volume","open_interest"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = df["volume"].fillna(0).astype(float)

    # attach a 'contract' label from filename stem (best-effort)
    df["contract"] = path.stem.lower().split("_")[0]
    return df[["timestamp","open","high","low","close","volume","contract"] + ([ "open_interest"] if "open_interest" in df.columns else [])]

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    if "open_interest" in df.columns:
        agg["open_interest"] = "last"
    return df.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["open","high","low","close"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, type=Path)
    ap.add_argument("--glob", default="*.csv")
    ap.add_argument("--source-tz", default="UTC")  # kept for CLI symmetry (we parse as UTC anyway)
    ap.add_argument("--output-tz", default="America/New_York")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--make-5m", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Read all CSVs
    parts = []
    for p in sorted(args.src_dir.glob(args.glob)):
        try:
            df = parse_barchart_file(p, args.source_tz)
            print(f"[ok] {p.name} -> {len(df):,} rows, contract={df['contract'].iat[0]}")
            parts.append(df)
        except Exception as e:
            print(f"[skip] {p.name}: {e}")

    if not parts:
        raise SystemExit("No files parsed successfully. Check your --src-dir/--glob and headers.")

    full = pd.concat(parts, ignore_index=True)
    # de-dup and sort
    full = full.drop_duplicates(subset=["timestamp","contract"]).sort_values(["timestamp","contract"])
    print(f"Concatenated: {len(full):,} -> {len(full):,} unique rows")

    # set index; convert to output tz (we keep naive wall-clock in output tz)
    full = full.set_index("timestamp")
    # We already have UTC-naive; convert to desired wall clock by localizing to UTC then tz-convert, then drop tz
    full = full.tz_localize("UTC").tz_convert(args.output_tz).tz_localize(None)

    # Save continuous 1m (all sessions)
    out_cont = args.out_dir / "mnq_1min_continuous.csv"
    full.sort_index().to_csv(out_cont)
    print(f"Saved {out_cont}")

    # RTH filter 09:30–16:00 (left-inclusive, exclude 16:00 bar)
    rth = full.between_time("09:30","16:00", inclusive="left").sort_index()
    out_rth = args.out_dir / "mnq_1min_rth.csv"
    rth.to_csv(out_rth)
    print(f"Saved {out_rth} (rows: {len(rth):,})")

    if args.make_5m:
        rth5 = resample_ohlcv(rth, "5min")
        out_5m = args.out_dir / "mnq_5min_rth.csv"
        rth5.to_csv(out_5m)
        print(f"Saved {out_5m} (rows: {len(rth5):,})")

if __name__ == "__main__":
    main()
