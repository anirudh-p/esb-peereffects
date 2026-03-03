"""
02_build_spatial_weights.py
============================
Builds K-nearest-neighbour spatial weight matrices from the NCES EDGE
school-district shapefile and attaches spatial lags of all key variables
to the analysis dataset.

Key choices documented inline.

Outputs
-------
  1_Data/Cleaned/analysis_dataset_spatial.csv  — analysis_dataset + uniform + inv-dist lags
  1_Data/Cleaned/knn_weights_k{K}.npz          — sparse CSR matrix (uniform 1/K)
  3_Output/Logs/spatial_weights_audit.txt

Column naming:
  w{K}_{var}  — row-standardised uniform 1/K weights (primary specs)
  wd{K}_{var} — inverse-distance row-normalised weights (robustness)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.sparse import save_npz, csr_matrix
from libpysal.weights import KNN
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SHP_FILE, ANALYSIS_DATASET, CLEAN, LOGS_DIR, ensure_dirs,
)
warnings.filterwarnings("ignore")
ensure_dirs()

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


# ── Configuration ────────────────────────────────────────────────────────────
K_VALUES = [6, 10, 15]   # run multiple K for robustness; k=6 is the main spec


# ══════════════════════════════════════════════════════════════════════════════
# 1  Load analysis dataset
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("1  Analysis dataset")
log("=" * 70)
df = pd.read_csv(ANALYSIS_DATASET, low_memory=False, dtype={"nces_id": str})
df["nces_id"] = df["nces_id"].str.zfill(7)
log(f"  Rows: {len(df):,}  |  NCES IDs: {df['nces_id'].nunique():,}")


# ══════════════════════════════════════════════════════════════════════════════
# 2  Load shapefile and compute centroids
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("2  Shapefile (NCES EDGE SY2021)")
log("=" * 70)

shp = gpd.read_file(SHP_FILE)
log(f"  Shapefile rows: {len(shp):,}")
log(f"  CRS: {shp.crs}")

# Identify NCES ID column: EDGE SY2021 shapefile uses 'GEOID' (7-char FIPS-based LEAID)
id_candidates = [c for c in shp.columns if "GEOID" in c.upper() or "LEAID" in c.upper()]
log(f"  ID column candidates: {id_candidates}")

# Use GEOID (= 7-digit NCES LEAID in EDGE files)
id_col = id_candidates[0] if id_candidates else "GEOID"
shp["nces_id"] = shp[id_col].astype(str).str.zfill(7)
log(f"  Using ID column: '{id_col}'")
log(f"  Unique NCES IDs in shapefile: {shp['nces_id'].nunique():,}")
log(f"  Sample: {shp['nces_id'].head(5).tolist()}")

# Project to Albers Equal Area (EPSG:5070) for accurate KNN centroids in the CONUS
# This projection uses metres, so KNN distances are in metres.
shp_proj = shp.to_crs("EPSG:5070")
shp_proj["cx"] = shp_proj.geometry.centroid.x
shp_proj["cy"] = shp_proj.geometry.centroid.y

# Merge with analysis dataset to keep only matched districts
shp_m = shp_proj[["nces_id", "cx", "cy"]].drop_duplicates("nces_id")
shp_m = shp_m.merge(df[["nces_id"]], on="nces_id", how="inner")
log(f"\n  Matched districts (shapefile ∩ analysis dataset): {len(shp_m):,}")
unmatched_in_df = len(set(df["nces_id"]) - set(shp_m["nces_id"]))
log(f"  Districts in analysis dataset with no shapefile entry: {unmatched_in_df:,}")

shp_m = shp_m.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3  Build KNN weight matrices and compute spatial lags
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("3  KNN weight matrices & spatial lags")
log("=" * 70)

# Variables for which we compute spatial lags
LAG_VARS = [
    "IV_Z_R1",        # R1 lottery win (instrument — main)
    "IV_Z_R3",        # R3 lottery win
    "IV_Z",           # pooled lottery win
    "IS_R2_GRANTEE",  # R2 grant awardee
    "IS_R1_LOSER",    # R1 lottery loser (application density control)
    "IS_R3_LOSER",    # R3 lottery loser
    "IS_LOSER_pooled",# pooled lottery loser
    "IS_ADOPTER",     # own CSBP lottery adoption (outcome for Est. 1)
    "IS_ADOPTER_CSBP_ANY",  # CSBP incl R2
    "Y_R3_apply",     # applied to R3 (outcome for Est. 2)
    "wri_any_2023",   # all-source 2023 (outcome for Est. 3)
    "wri_any_2024",   # all-source 2024 (outcome for Est. 3)
    "wri_any_2023_24",# all-source 2023-24 (outcome for Est. 3)
    "is_pre_r1_adopter",  # prior adopter (control)
    "r1_early_delivery",  # R1 early deployment (instrument for Est. 4)
    "r1_late_or_unknown", # R1 late/unknown deployment  (instrument for Est. 4)
    # controls (for within-W summaries)
    "median_income", "poverty_rate", "enrollment", "pm25", "pct_white",
]

# Align: match df to shp_m index
df_r = df.merge(shp_m[["nces_id"]], on="nces_id", how="inner").reset_index(drop=True)
log(f"  Districts entering W construction: {len(df_r):,}")

coords = shp_m[["cx", "cy"]].values   # shape (N, 2)

result_dfs = {}   # k → merged dataframe

for K in K_VALUES:
    log(f"\n  ── K = {K} ──")
    w = KNN(coords, k=K)
    w.transform = "r"   # row-standardise → each row sums to 1

    # Build sparse weight matrix (N x N)
    # w.neighbors: {i: [j1, j2, ...]}  w.weights: {i: [w1, w2, ...]}
    N = len(shp_m)
    rows, cols, vals = [], [], []
    for i in range(N):
        for j, vij in zip(w.neighbors[i], w.weights[i]):
            rows.append(i)
            cols.append(j)
            vals.append(vij)
    W = csr_matrix((vals, (rows, cols)), shape=(N, N))

    # Save sparse matrix
    npz_path = CLEAN / f"knn_weights_k{K}.npz"
    save_npz(str(npz_path), W)
    log(f"    Sparse W saved: {npz_path.name}")

    # Compute spatial lags (W @ X for each variable)
    lag_dict = {"nces_id": shp_m["nces_id"].values}
    for var in LAG_VARS:
        if var in df_r.columns:
            x = pd.to_numeric(df_r[var], errors="coerce").fillna(0).values
            lag = W.dot(x)
            lag_dict[f"w{K}_{var}"] = lag
        else:
            log(f"    WARNING: '{var}' not in dataframe — skipping")

    lag_df = pd.DataFrame(lag_dict)
    log(f"    Spatial lag columns created: {len(lag_df.columns)-1}")
    result_dfs[K] = lag_df

    # Quick sanity: mean spatial lag of IV_Z_R1
    wvar = f"w{K}_IV_Z_R1"
    mean_lag = lag_df[wvar].mean()
    nonzero = (lag_df[wvar] > 0).sum()
    log(f"    mean w_IV_Z_R1 = {mean_lag:.5f}  |  districts with ≥1 R1 winner neighbor: {nonzero:,}")

    # ── Inverse-distance spatial lags ─────────────────────────────────────
    # Use sklearn to get the same K neighbors plus their actual distances
    # (EPSG:5070 metres). Weights = (1/d) / sum(1/d), row-normalised.
    # Column prefix  wd{K}_  to distinguish from uniform  w{K}_  lags.
    nbrs = NearestNeighbors(n_neighbors=K + 1, algorithm="ball_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    # Column 0 is the point itself (d≈0); skip it.
    dist_K = distances[:, 1:]   # (N, K)
    idx_K  = indices[:, 1:]     # (N, K)

    # Clip distances to 1 m minimum to guard against coincident centroids
    inv_d = 1.0 / np.maximum(dist_K, 1.0)
    inv_d_norm = inv_d / inv_d.sum(axis=1, keepdims=True)  # row-normalise

    N = len(shp_m)
    rows_d, cols_d, vals_d = [], [], []
    for i in range(N):
        for j_idx, wij in zip(idx_K[i], inv_d_norm[i]):
            rows_d.append(i)
            cols_d.append(int(j_idx))
            vals_d.append(wij)
    W_d = csr_matrix((vals_d, (rows_d, cols_d)), shape=(N, N))

    for var in LAG_VARS:
        if var in df_r.columns:
            x = pd.to_numeric(df_r[var], errors="coerce").fillna(0).values
            lag_df[f"wd{K}_{var}"] = W_d.dot(x)

    log(f"    Inverse-dist lag columns added (wd{K}_*): {sum(c.startswith(f'wd{K}_') for c in lag_df.columns)}")
    # Sanity
    mean_d = lag_df[f"wd{K}_IV_Z_R1"].mean()
    log(f"    mean wd_IV_Z_R1 = {mean_d:.5f}  (cf. uniform {mean_lag:.5f})")

log("\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4  Merge spatial lags back to full dataset and save
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("4  Merge and save enriched dataset")
log("=" * 70)

out = df.copy()
for K, lag_df in result_dfs.items():
    out = out.merge(lag_df, on="nces_id", how="left")

log(f"  Final shape: {out.shape}")
log(f"  Columns: {list(out.columns)[:12]} ...")

# Estimation-sample coverage check (all controls + at least one lag variable)
required = ["enrollment", "median_income", "poverty_rate", "pct_white",
            "pm25", "state", "pct_dem_2020", f"w6_IV_Z_R1"]
est = out.dropna(subset=required)
log(f"\n  Estimation sample (all controls + K=6 lag): {len(est):,} / {len(out):,}")
log(f"    R1 winners in sample : {est['IV_Z_R1'].sum():,}")
log(f"    R3 winners in sample : {est['IV_Z_R3'].sum():,}")
log(f"    R1 losers in sample  : {est['IS_R1_LOSER'].sum():,}")

# Save
out_path = CLEAN / "analysis_dataset_spatial.csv"
out.to_csv(out_path, index=False)
log(f"\n  Saved: {out_path}")

log_text = "\n".join(log_lines)
with open(LOGS_DIR / "spatial_weights_audit.txt", "w", encoding="utf-8") as f:
    f.write(log_text)
log(f"  Saved: {LOGS_DIR / 'spatial_weights_audit.txt'}")
