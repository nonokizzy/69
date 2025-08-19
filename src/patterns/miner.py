from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import stumpy

from .utils import sliding_windows, to_returns, z_norm, ensure_datetime_index

def choose_candidates(mp: np.ndarray, top_k: int, stride: int) -> list[int]:
    # select indices with lowest matrix profile (most motif-like), de-duplicated with stride
    idx = np.argsort(mp)
    chosen=[]
    last=-1e9
    for i in idx:
        if not np.isfinite(mp[i]): 
            continue
        if chosen and (i - chosen[-1]) < stride:
            continue
        chosen.append(int(i))
        if len(chosen) >= top_k: break
    return chosen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with DatetimeIndex and columns open,high,low,close")
    ap.add_argument("--out", required=True, help="output folder, e.g., artifacts/patterns")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--clusters", type=int, default=20)
    ap.add_argument("--samples-per-cluster", type=int, default=20, help="target windows/cluster for thumbnails")
    args = ap.parse_args()

    out_base = Path(args.out); out_base.mkdir(parents=True, exist_ok=True)
    run_dir = out_base / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=[0], index_col=0).sort_index()
    df.columns = [c.lower() for c in df.columns]
    ensure_datetime_index(df)
    closes = df["close"].values.astype(float)

    # Matrix profile on closes
    mp_full = stumpy.stump(closes, m=args.window)[:,0]  # shape n-m+1
    print(f"Matrix profile computed: shape=({mp_full.shape[0]},)")

    # pick up to clusters * samples_per_cluster windows for clustering
    top_k = args.clusters * args.samples_per_cluster * 2
    cand_idx = choose_candidates(mp_full, top_k=top_k, stride=max(1, args.stride))
    if not cand_idx:
        raise SystemExit("No candidate windows selected.")
    print(f"Windows selected for clustering: {len(cand_idx)}")

    # Build windowed features: returns (W-1), z-normalized
    W = args.window
    feats = []
    for s in cand_idx:
        if s + W > len(closes): continue
        r = to_returns(closes[s:s+W])
        feats.append(z_norm(r))
    X = np.vstack(feats)

    # KMeans
    k = args.clusters
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X)

    # Map cluster -> start indices
    clusters = {}
    for lab, s in zip(labels, cand_idx[:len(labels)]):
        clusters.setdefault(int(lab), []).append(int(s))

    # Save CSV of occurrences
    rows=[]
    for cid, starts in clusters.items():
        for s in starts:
            e = s + W - 1
            rows.append(dict(
                cluster_id=cid,
                start_idx=s,
                end_idx=e,
                start_time=str(df.index[s]),
                end_time=str(df.index[e]),
                mp_value=float(mp_full[s]) if s < len(mp_full) else np.nan
            ))
    occ_path = run_dir / "occurrences.csv"
    pd.DataFrame(rows).to_csv(occ_path, index=False)

    # Save clusters.json (miner format)
    clusters_json = dict(k=k, window=W, stride=args.stride, clusters={str(cid): v for cid,v in clusters.items()})
    (out_base / "clusters.json").write_text(json.dumps(clusters_json, indent=2))
    (run_dir / "clusters.json").write_text(json.dumps(clusters_json, indent=2))

    # Quick thumbnails per cluster
    for cid, starts in clusters.items():
        take = starts[:min(len(starts), args.samples_per_cluster)]
        plt.figure(figsize=(6,3))
        for s in take:
            seg = closes[s:s+W]
            if len(seg) < W: continue
            seg = (seg - seg[0])  # normalize to start at 0
            plt.plot(seg, alpha=0.5)
        plt.title(f"Cluster {cid} (n={len(starts)})")
        plt.tight_layout()
        plt.savefig(run_dir / f"thumb_cluster_{cid}.png", dpi=130)
        plt.close()

    # Optional UMAP plot if available
    try:
        import umap
        reducer = umap.UMAP(random_state=42)
        emb = reducer.fit_transform(X)
        plt.figure(figsize=(5,5))
        sc = plt.scatter(emb[:,0], emb[:,1], c=labels, s=6, cmap="tab20")
        plt.title("UMAP of motif windows")
        plt.tight_layout()
        plt.savefig(run_dir / "umap_2d.png", dpi=130)
        plt.close()
        print("Saved UMAP plot.")
    except Exception:
        pass

    # Summary
    print("\n[mined] Artifacts:")
    print(" -", occ_path)
    print(" -", run_dir / "clusters.json")
    print(" -", out_base / "clusters.json")
    print(" -", run_dir / "thumb_cluster_*.png")
    print(" -", run_dir / "umap_2d.png (if UMAP installed)")
    print("Done.")
if __name__ == "__main__":
    main()
