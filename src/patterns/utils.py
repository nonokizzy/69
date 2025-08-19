from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
    df.columns = [c.lower() for c in df.columns]
    for c in ("open","high","low","close"):
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    return df

def ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")
    return df

def to_returns(x: np.ndarray) -> np.ndarray:
    # simple pct change of closes, fallback when flat
    if x.ndim == 1:
        x = x.reshape(-1,1)
    closes = x[:,0] if x.shape[1] == 1 else x[:,3]  # close if OHLC passed
    r = np.diff(closes) / np.where(closes[:-1]==0, 1.0, closes[:-1])
    return r

def z_norm(a: np.ndarray) -> np.ndarray:
    mu = a.mean()
    sd = a.std()
    if sd < 1e-12:
        return np.zeros_like(a)
    return (a - mu) / sd

def sliding_windows(series: np.ndarray, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Return array of windows [n,window] and start indices."""
    n = len(series)
    idxs = list(range(0, n - window + 1, stride))
    out = np.stack([series[i:i+window] for i in idxs], axis=0) if idxs else np.zeros((0,window))
    return out, np.array(idxs, dtype=int)

def parse_sessions(spec: str):
    # "HH:MM-HH:MM,HH:MM-HH:MM"
    out=[]
    for seg in (spec or "").split(","):
        seg=seg.strip()
        if not seg: continue
        a,b=seg.split("-")
        out.append((a.strip(), b.strip()))
    return out

def in_any_session(idx: pd.DatetimeIndex, sessions):
    mask = pd.Series(False, index=idx)
    for a,b in sessions:
        # mark those timestamps that fall between a..b (left-inclusive)
        take = mask.index.indexer_between_time(a, b, include_start=True, include_end=False)
        mask.iloc[take] = True
    return mask
