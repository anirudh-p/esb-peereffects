from __future__ import annotations

from pathlib import Path
import argparse
from typing import List

import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import BallTree
except Exception as e:
    raise ImportError("scikit-learn is required for DynamicPeers.py (BallTree). Install with pip/conda.") from e


EARTH_R_KM = 6371.0088


def _to_radians(lat_series: pd.Series, lon_series: pd.Series) -> np.ndarray:
    lat = np.deg2rad(lat_series.values.astype(float))
    lon = np.deg2rad(lon_series.values.astype(float))
    return np.vstack([lat, lon]).T


def build_neighbors(peer_vars_path: Path, k: int, radii: List[float], lat_col: str = "1s. Latitude", lon_col: str = "1t. Longitude"):
    pv = pd.read_csv(peer_vars_path)
    pv.columns = [c.strip() if isinstance(c, str) else c for c in pv.columns]
    if lat_col not in pv.columns or lon_col not in pv.columns:
        raise KeyError(f"Latitude/Longitude columns not found in {peer_vars_path}: expected {lat_col}, {lon_col}")
    # Keep LEA IDs as strings to preserve any non-numeric IDs
    ids = pv["1c. LEA ID"].astype(str).values
    coords = _to_radians(pv[lat_col], pv[lon_col])
    tree = BallTree(coords, metric="haversine")

    # KNN (exclude self)
    knn_k = max(1, k)
    dist, ind = tree.query(coords, k=knn_k + 1)
    # drop self (first neighbor with zero distance)
    neigh_ind = [list(n[1:knn_k+1]) for n in ind]
    knn_neighbors = {ids[i]: [ids[j] for j in neigh_ind[i]] for i in range(len(ids))}

    # radii neighbors (within radius r km) -> convert to radians
    radius_neighbors = {}
    for r in radii:
        r_rad = r / EARTH_R_KM
        ind_list = tree.query_radius(coords, r_rad)
        # exclude self
    radius_neighbors[r] = {ids[i]: [ids[j] for j in ind_list[i] if j != i] for i in range(len(ids))}

    return knn_neighbors, radius_neighbors


def compute_dynamic_peers(panel_path: Path,
                          peer_vars_path: Path,
                          k: int = 5,
                          radii: List[float] = [50.0, 100.0],
                          lea_col: str = "1c. LEA ID",
                          out_path: Path = Path("lea_quarter_panel_with_peers.csv"),
                          lag_quarters: int = 1) -> Path:
    # Build neighbors index
    knn_map, rad_map = build_neighbors(peer_vars_path, k, radii)

    # Load panel and keep minimal columns
    panel = pd.read_csv(panel_path)
    panel.columns = [c.strip() if isinstance(c, str) else c for c in panel.columns]
    if lea_col not in panel.columns:
        raise KeyError(f"LEA id column {lea_col} not found in panel")
    # Normalize LEA id to string to match neighbor mapping keys
    panel[lea_col] = panel[lea_col].astype(str)
    if "quarter" not in panel.columns:
        raise KeyError("panel must contain 'quarter' column in YYYYQ# format")
    if "operating" not in panel.columns:
        raise KeyError("panel must contain binary 'operating' column")

    # Parse quarter to Period for sorting and lagging
    panel["_q_per"] = panel["quarter"].apply(lambda x: pd.Period(str(x), freq="Q"))
    panel.sort_values([lea_col, "_q_per"], inplace=True)

    # Keep a small table for neighbor lookup (neighbor_id, lea, quarter, operating)
    neighbor_status = panel[[lea_col, "_q_per", "operating"]].copy()
    neighbor_status.rename(columns={lea_col: "neighbor_id", "_q_per": "_q_per", "operating": "neighbor_operating"}, inplace=True)

    # Function to aggregate given neighbor mapping name
    def _aggregate_for_map(name: str, mapping):
        # Build exploded neighbor map DataFrame: original_lea, neighbor_id
        rows = []
        for lea, neighbors in mapping.items():
            if not neighbors:
                continue
            for nb in neighbors:
                rows.append((str(lea), str(nb)))
        if not rows:
            return pd.DataFrame(columns=[lea_col, "_q_per", f"{name}_peer_count", f"{name}_peer_operating_count", f"{name}_peer_operating_share"])
        neighbor_pairs = pd.DataFrame(rows, columns=[lea_col, "neighbor_id"]) if rows else pd.DataFrame(columns=[lea_col, "neighbor_id"])
        quarters = panel["_q_per"].drop_duplicates().tolist()
        parts = []
        # process quarter-by-quarter to avoid a huge cross join
        for q in quarters:
            ns_q = neighbor_status[neighbor_status["_q_per"] == q][["neighbor_id", "neighbor_operating"]]
            # merge neighbor pairs (all LEA->neighbor rows) with neighbor status for this quarter
            merged = neighbor_pairs.merge(ns_q, left_on="neighbor_id", right_on="neighbor_id", how="left")
            agg_q = merged.groupby([lea_col], dropna=False).agg(
                **{
                    f"{name}_peer_count": ("neighbor_id", "count"),
                    f"{name}_peer_operating_count": ("neighbor_operating", lambda s: int(s.fillna(0).sum())),
                }
            ).reset_index()
            agg_q["_q_per"] = q
            agg_q[f"{name}_peer_operating_share"] = agg_q[f"{name}_peer_operating_count"] / agg_q[f"{name}_peer_count"].replace({0: pd.NA})
            parts.append(agg_q)
        if parts:
            agg = pd.concat(parts, axis=0, ignore_index=True)
        else:
            agg = pd.DataFrame(columns=[lea_col, "_q_per", f"{name}_peer_count", f"{name}_peer_operating_count", f"{name}_peer_operating_share"])
        return agg

    # Compute KNN aggregates
    knn_agg = _aggregate_for_map(f"knn{k}", knn_map)

    # Compute radius aggregates for each radius
    rad_aggs = []
    for r in rad_map:
        rad_aggs.append(_aggregate_for_map(f"within_{int(r)}km", rad_map[r]))

    # Merge aggregates into panel
    out = panel.copy()
    # merge knn
    if not knn_agg.empty:
        out = out.merge(knn_agg, left_on=[lea_col, "_q_per"], right_on=[lea_col, "_q_per"], how="left")
    # merge radii
    for agg in rad_aggs:
        if agg.empty:
            continue
        out = out.merge(agg, left_on=[lea_col, "_q_per"], right_on=[lea_col, "_q_per"], how="left")

    # Fill zeros for counts where NaN
    for c in out.columns:
        if c.endswith("_peer_count") or c.endswith("_peer_operating_count"):
            out[c] = out[c].fillna(0).astype(int)

    # Create lagged columns (previous quarter) for key measures
    peer_cols = [c for c in out.columns if c.endswith("_peer_operating_share")]
    # Sort and create shift key
    out.sort_values([lea_col, "_q_per"], inplace=True)
    for c in peer_cols:
        out[f"{c}_lag{lag_quarters}"] = out.groupby(lea_col)[c].shift(lag_quarters)

    # Drop helper Period column and restore quarter string
    out.drop(columns=["_q_per"], inplace=True)

    out.to_csv(out_path, index=False)
    print(f"Wrote dynamic peers panel: {len(out):,} rows -> {out_path.resolve()}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Compute time-varying peer measures for each LEA-quarter.")
    ap.add_argument("--panel", type=Path, default=Path("lea_quarter_panel.csv"))
    ap.add_argument("--peer-vars", type=Path, default=Path("lea_peer_vars.csv"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--radii", type=float, nargs="*", default=[50.0, 100.0])
    ap.add_argument("--out", type=Path, default=Path("lea_quarter_panel_with_peers.csv"))
    ap.add_argument("--lag", type=int, default=1)
    args = ap.parse_args()

    compute_dynamic_peers(args.panel, args.peer_vars, args.k, args.radii, out_path=args.out, lag_quarters=args.lag)


if __name__ == "__main__":
    main()
