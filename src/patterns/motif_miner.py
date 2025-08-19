from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .utils import load_ohlcv_csv, ensure_dt_index, to_returns, z_norm, sliding_windows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with OHLCV, DatetimeIndex in col0")
    ap.add_argument("--out", required=True, help="output dir for artifacts")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-support", type=int, default=20, help="min occurrences to keep cluster")
    ap.add_argument("--clusters", type=int, default=12)
    ap.add_argument("--max-samples", type=int, default=20000, help="cap windows to cluster speed")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df = ensure_dt_index(load_ohlcv_csv(Path(args.data)))
    close = df["close"].values.astype(float)

    # windows of returns (z-normalized)
    r = to_returns(close.reshape(-1,1))
    W = args.window
    X, starts = sliding_windows(r, W, args.stride)
    if len(starts)==0:
        raise SystemExit("No windows available. Reduce --window or ensure data length.")

    Xz = np.stack([z_norm(x) for x in X], axis=0)

    # sample if too many
    if len(Xz) > args.max_samples:
        sel = np.random.RandomState(0).choice(len(Xz), size=args.max_samples, replace=False)
        Xz_s, starts_s = Xz[sel], starts[sel]
    else:
        Xz_s, starts_s = Xz, starts

    # cluster
    km = KMeans(n_clusters=args.clusters, n_init=10, random_state=0)
    labels = km.fit_predict(Xz_s)
    sil = silhouette_score(Xz_s, labels) if len(Xz_s) > args.clusters else np.nan

    # assign centroids -> collect members
    clusters = {i: [] for i in range(args.clusters)}
    for lab, st in zip(labels, starts_s):
        clusters[int(lab)].append(int(st))

    # prune by support
    clusters = {k:v for k,v in clusters.items() if len(v) >= args.min_support}

    # write clusters.json in unified format:
    # { "meta": {...}, "centroids": {cid: [floats...]}, "hits": {cid: [{start_idx,end_idx,start_time,end_time}]}}
    meta = dict(window=W, stride=args.stride, clusters=len(clusters), silhouette=float(sil))
    centroids = {int(i): km.cluster_centers_[i].tolist() for i in clusters.keys()}

    hits = {}
    for cid, starts_list in clusters.items():
        arr = []
        for st in starts_list:
            et = st + W
            st_ts = df.index[st+1] if st+1 < len(df.index) else df.index[-1]        # +1 because returns start at 1
            et_ts = df.index[min(et+1, len(df.index)-1)]
            arr.append(dict(start_idx=int(st), end_idx=int(et), start_time=str(st_ts), end_time=str(et_ts)))
        hits[int(cid)] = arr

    out_json = dict(meta=meta, centroids=centroids, hits=hits)
    (out_dir/"clusters.json").write_text(json.dumps(out_json, indent=2))
    # convenience CSV of occurrences
    rows=[]
    for cid, arr in hits.items():
        for h in arr:
            rows.append(dict(cluster_id=cid, **h))
    pd.DataFrame(rows).to_csv(out_dir/"occurrences.csv", index=False)

    print(f"[motif_miner] windows={len(X)} (sampled {len(Xz_s)}) | kept clusters={len(clusters)} | silhouette={sil:.3f}")
    print(f"wrote: {out_dir/'clusters.json'}")
    print(f"wrote: {out_dir/'occurrences.csv'}")

if __name__ == "__main__":
    main()
