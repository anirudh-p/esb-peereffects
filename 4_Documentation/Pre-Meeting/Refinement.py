from __future__ import annotations

from pathlib import Path
import argparse
import sys
import json
import math
from typing import Tuple

import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def load_data(workbook: Path,
              bus_sheet: str = "2. Bus-level data",
              district_sheet: str = "1. District-level data",
              lea_col: str = "1c. LEA ID",
              lat_col: str = "1s. Latitude",
              lon_col: str = "1t. Longitude") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load bus and district sheets and return DataFrames."""
    if not workbook.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook}")
    bus = pd.read_excel(workbook, sheet_name=bus_sheet, engine="openpyxl")
    district = pd.read_excel(workbook, sheet_name=district_sheet, engine="openpyxl")
    # normalize column names (trim whitespace)
    bus.columns = [c.strip() if isinstance(c, str) else c for c in bus.columns]
    district.columns = [c.strip() if isinstance(c, str) else c for c in district.columns]
    # basic sanity
    for col in (lea_col,):
        if col not in bus.columns:
            raise KeyError(f"Missing '{col}' in bus sheet")
        if col not in district.columns:
            raise KeyError(f"Missing '{col}' in district sheet")
    for col in (lat_col, lon_col):
        if col not in district.columns:
            raise KeyError(f"Missing '{col}' in district sheet")
    return bus, district


def build_lea_level(bus: pd.DataFrame,
                    district: pd.DataFrame,
                    lea_col: str = "1c. LEA ID",
                    lat_col: str = "1s. Latitude",
                    lon_col: str = "1t. Longitude") -> pd.DataFrame:
    """Create LEA-level table with coordinates and adoption indicators.

    - adoption_count: number of bus rows for the LEA
    - adopted: 1 if adoption_count > 0 else 0
    """
    # Aggregate adoption at LEA level using bus rows as adoption evidence
    adopt_count = bus.groupby(lea_col, dropna=False).size().rename("adoption_count").to_frame()

    lea = district[[lea_col, lat_col, lon_col]].copy()
    # Keep a few labels if present (optional)
    for c in ("1a. State", "1b. Local Education Agency (LEA) or entity name", "1l. LEA type (name)"):
        if c in district.columns:
            lea[c] = district[c]

    lea = lea.merge(adopt_count, left_on=lea_col, right_index=True, how="left")
    lea["adoption_count"] = lea["adoption_count"].fillna(0).astype(int)
    lea["adopted"] = (lea["adoption_count"] > 0).astype(int)
    # drop rows without coords
    missing_coords = lea[[lat_col, lon_col]].isna().any(axis=1).sum()
    if missing_coords:
        print(f"Info: dropping {missing_coords} LEAs due to missing coordinates", file=sys.stderr)
    lea = lea.dropna(subset=[lat_col, lon_col]).copy()
    return lea


def compute_knn_metrics(lea: pd.DataFrame,
                        lea_col: str,
                        lat_col: str,
                        lon_col: str,
                        k: int = 5,
                        epsilon_km: float = 1e-3) -> pd.DataFrame:
    """Compute k-NN neighbors using BallTree (haversine) and derive peer metrics.

    Adds:
    - knn{k}_peer_adopt_share
    - knn{k}_peer_adopt_share_idw (inverse-distance weighted)
    - knn{k}_peer_adopt_count
    """
    try:
        from sklearn.neighbors import BallTree
        import numpy as np
    except Exception as e:
        raise ImportError("scikit-learn and numpy are required for k-NN metrics. Please install them.") from e

    coords_rad = np.radians(lea[[lat_col, lon_col]].to_numpy(dtype=float))
    tree = BallTree(coords_rad, metric="haversine")
    # query k+1 to include self; we'll drop self later
    dists_rad, idx = tree.query(coords_rad, k=min(k + 1, len(lea)))

    # Convert to km
    dists_km = dists_rad * EARTH_RADIUS_KM

    # drop self neighbor (distance close to 0)
    # assume first neighbor is self
    if idx.shape[1] > 0:
        idx = idx[:, 1:]
        dists_km = dists_km[:, 1:]

    adopted = lea["adopted"].to_numpy()
    # peer adoption count and share
    peer_counts = adopted[idx].sum(axis=1)
    peer_share = peer_counts / idx.shape[1]

    # inverse-distance weighted share (avoid divide by zero with epsilon)
    weights = 1.0 / (dists_km + epsilon_km)
    idw_share = (weights * adopted[idx]).sum(axis=1) / weights.sum(axis=1)

    lea[f"knn{k}_peer_adopt_count"] = peer_counts
    lea[f"knn{k}_peer_adopt_share"] = peer_share
    lea[f"knn{k}_peer_adopt_share_idw"] = idw_share
    return lea


def compute_radius_metrics(lea: pd.DataFrame,
                           lea_col: str,
                           lat_col: str,
                           lon_col: str,
                           radii_km=(50, 100),
                           epsilon_km: float = 1e-3) -> pd.DataFrame:
    """Compute within-radius neighbor counts and adoption shares using BallTree."""
    try:
        from sklearn.neighbors import BallTree
        import numpy as np
    except Exception as e:
        raise ImportError("scikit-learn and numpy are required for radius metrics. Please install them.") from e

    coords_rad = np.radians(lea[[lat_col, lon_col]].to_numpy(dtype=float))
    tree = BallTree(coords_rad, metric="haversine")
    adopted = lea["adopted"].to_numpy()

    for rkm in radii_km:
        rad = rkm / EARTH_RADIUS_KM
        ind_list = tree.query_radius(coords_rad, r=rad, return_distance=False)
        within_counts = []
        within_adopt_counts = []
        within_shares = []
        for i, inds in enumerate(ind_list):
            # remove self (where index equals i) if present
            inds = [j for j in inds if j != i]
            cnt = len(inds)
            within_counts.append(cnt)
            if cnt == 0:
                within_adopt_counts.append(0)
                within_shares.append(float("nan"))
            else:
                ac = int(adopted[inds].sum())
                within_adopt_counts.append(ac)
                within_shares.append(ac / cnt)
        lea[f"within_{rkm}km_peer_count"] = within_counts
        lea[f"within_{rkm}km_peer_adopt_count"] = within_adopt_counts
        lea[f"within_{rkm}km_peer_adopt_share"] = within_shares
    return lea


def summarize_and_save(lea: pd.DataFrame,
                       out_csv: Path,
                       lea_col: str,
                       lat_col: str,
                       lon_col: str) -> None:
    lea.to_csv(out_csv, index=False)
    summary = {
        "rows": int(len(lea)),
        "unique_lea_ids": int(lea[lea_col].nunique(dropna=True)),
        "missing_coords_rows": int(lea[[lat_col, lon_col]].isna().any(axis=1).sum()),
        "columns_added": [c for c in lea.columns if c.startswith("knn") or c.startswith("within_")],
    }
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved LEA-level peer variables to: {out_csv.resolve()}")


def main():
    ap = argparse.ArgumentParser(description="Compute LEA-level peer variables (k-NN and within-radius) using Haversine distances.")
    ap.add_argument("--file", type=Path, default=Path(r"c:\BC PhD\Research\Peer-Effects and Adoption\ESB_adoption_dataset_v9_update_june_2025.xlsx"), help="Path to workbook")
    ap.add_argument("--bus-sheet", default="2. Bus-level data", help="Bus-level sheet name")
    ap.add_argument("--district-sheet", default="1. District-level data", help="District-level sheet name")
    ap.add_argument("--lea-col", default="1c. LEA ID", help="LEA identifier column name")
    ap.add_argument("--lat-col", default="1s. Latitude", help="Latitude column in district sheet")
    ap.add_argument("--lon-col", default="1t. Longitude", help="Longitude column in district sheet")
    ap.add_argument("--k", type=int, default=5, help="k for k-NN peer metrics")
    ap.add_argument("--radii", nargs="*", type=int, default=[50, 100], help="Radii in km for within-radius metrics (space-separated)")
    ap.add_argument("--out", type=Path, default=Path("lea_peer_vars.csv"), help="Output CSV path")
    args = ap.parse_args()

    bus, district = load_data(args.file, args.bus_sheet, args.district_sheet, args.lea_col, args.lat_col, args.lon_col)
    lea = build_lea_level(bus, district, args.lea_col, args.lat_col, args.lon_col)
    lea = compute_knn_metrics(lea, args.lea_col, args.lat_col, args.lon_col, k=args.k)
    lea = compute_radius_metrics(lea, args.lea_col, args.lat_col, args.lon_col, radii_km=tuple(args.radii))
    summarize_and_save(lea, args.out, args.lea_col, args.lat_col, args.lon_col)


if __name__ == "__main__":
    main()
