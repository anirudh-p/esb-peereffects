"""
03_build_supplementary.py
=========================
Post-Meeting branch: builds supplementary variables not in the core pipeline.

Adds to analysis_dataset_spatial.csv:
  (A) New outcome margins
      - Y_R2_apply      : applied to R2 grant (winner or rejected)
      - Y_any_post_r1   : applied to R2 OR R3 after R1 results (broadest CSBP entry)

  (B) Vendor network
      - vendor_flag     : district has at least one R1-winner-sharing dealer neighbor
        (constructed from WRI 3x. Dealer column)
      - w6_shares_dealer_with_r1 : share of K=6 neighbors sharing a dealer with an R1 winner

  (C) Lat/lon centroids (WGS84) for Conley HAC
      - lat, lon        : from NCES EDGE shapefile INTPTLAT / INTPTLON

  (D) PM2.5 and poverty quartiles for priority decomposition
      - pm25_q, poverty_q : quartile labels (Q1-Q4)

Output
------
  1_Data/Cleaned/analysis_dataset_supp.csv
  3_Output/Logs/supplementary_audit.txt
"""

import sys, warnings, re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SPATIAL_DATASET, SHP_FILE, CSB_APPLICANTS, CSB_GRANTS,
    WRI_BUS_EXCEL, CLEAN, LOGS_DIR, ensure_dirs,
)
warnings.filterwarnings("ignore")
ensure_dirs()

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

def clean_nces(s):
    return s.astype(str).str.split(".").str[0].str.strip().str.zfill(7)

# =============================================================================
# Load base spatial dataset
# =============================================================================
df = pd.read_csv(SPATIAL_DATASET, low_memory=False, dtype={"nces_id": str})
log(f"Base spatial dataset: {len(df):,} rows")

# =============================================================================
# A — New outcome margins
# =============================================================================
log("\n" + "=" * 60)
log("A — New outcome margins")
log("=" * 60)

# ── R2 applicants (rejected from competitive grant) ───────────────────────────
app = pd.read_excel(CSB_APPLICANTS)
app["nces_id"] = clean_nces(app["NCES District ID"])
r2_rows = app[app["Round"].str.contains("R2", na=False)].copy()
r2_app_nces = set(r2_rows["nces_id"].unique())
log(f"  R2 applicant file rows: {len(r2_rows)}  unique districts: {len(r2_app_nces)}")

# R2 grant winners
grn = pd.read_excel(CSB_GRANTS)
grn_active = grn[grn["Project Status"].str.strip() == "Funds Awarded"]
r2_win_nces = set(clean_nces(grn_active["NCES District ID"]).unique())
log(f"  R2 grant winners: {len(r2_win_nces)}")

# Y_R2_apply = applied to R2 (rejected applicant OR winner)
r2_all_nces = r2_app_nces | r2_win_nces
log(f"  Total R2 applicants (rejected + winners): {len(r2_all_nces)}")

df["Y_R2_apply"] = df["nces_id"].isin(r2_all_nces).astype(int)

# R3 applicants already in dataset as Y_R3_apply
# Y_any_post_r1 = applied to R2 OR R3 (any post-R1 CSBP engagement)
df["Y_any_post_r1"] = ((df["Y_R2_apply"] == 1) | (df["Y_R3_apply"] == 1)).astype(int)

log(f"\n  Y_R2_apply (all districts) : {df['Y_R2_apply'].sum():,} ({100*df['Y_R2_apply'].mean():.2f}%)")
log(f"  Y_R3_apply (all districts) : {df['Y_R3_apply'].sum():,} ({100*df['Y_R3_apply'].mean():.2f}%)")
log(f"  Y_any_post_r1              : {df['Y_any_post_r1'].sum():,} ({100*df['Y_any_post_r1'].mean():.2f}%)")
log(f"  Overlap (R2 AND R3)        : {((df['Y_R2_apply']==1) & (df['Y_R3_apply']==1)).sum():,}")

# Rates in estimation sample (non-R1-winners)
est = df[df["IV_Z_R1"] == 0].dropna(
    subset=["enrollment","median_income","poverty_rate","pct_white","pm25","state","pct_dem_2020"]
).copy()
log(f"\n  Rates in non-R1-winner estimation sample (N={len(est):,}):")
for col in ["Y_R2_apply","Y_R3_apply","Y_any_post_r1"]:
    log(f"    {col:<25}: {est[col].sum():4,} ({100*est[col].mean():.2f}%)")


# =============================================================================
# B — Vendor network
# =============================================================================
log("\n" + "=" * 60)
log("B — Vendor network")
log("=" * 60)

buses = pd.read_excel(WRI_BUS_EXCEL, sheet_name="2. Bus-level data",
                      usecols=["1c. LEA ID", "3p. Quarter awarded", "3x. Dealer"])
buses.columns = ["nces_id","award_quarter","dealer"]
buses["nces_id"] = clean_nces(buses["nces_id"])

# Extract award year
buses["award_year"] = buses["award_quarter"].astype(str).str.extract(r"(\d{4})")[0]
buses["award_year"] = pd.to_numeric(buses["award_year"], errors="coerce")

# R1 winners' buses (award year 2022)
r1_winner_nces = set(df[df["IV_Z_R1"] == 1]["nces_id"])
r1_buses = buses[(buses["award_year"] == 2022) & buses["nces_id"].isin(r1_winner_nces)].copy()
r1_buses = r1_buses.dropna(subset=["dealer"])
log(f"  R1 winner buses with dealer info: {len(r1_buses):,}")
log(f"  Unique dealers serving R1 winners: {r1_buses['dealer'].nunique():,}")

# Map R1 winners to their dealers
r1_dealer_map = (
    r1_buses.groupby("nces_id")["dealer"]
    .apply(lambda s: set(s.dropna()))
    .reset_index()
    .rename(columns={"dealer": "r1_dealers"})
)
log(f"  R1 winner districts with dealer data: {len(r1_dealer_map):,}")

# For each non-R1-winner district: does it share a dealer with ANY R1 winner?
# Use district-level dealer from their own buses (any year)
own_dealers = (
    buses.dropna(subset=["dealer"])
    .groupby("nces_id")["dealer"]
    .apply(lambda s: set(s))
    .reset_index()
    .rename(columns={"dealer": "own_dealers"})
)

# Build dealer -> R1 winner set lookup
from collections import defaultdict
dealer_to_r1 = defaultdict(set)
for _, row in r1_dealer_map.iterrows():
    for d in row["r1_dealers"]:
        dealer_to_r1[d].add(row["nces_id"])

# Flag: does this district's dealer serve any R1 winner?
def shares_dealer_with_r1(own_set):
    if not isinstance(own_set, set):
        return 0
    return int(any(dealer_to_r1.get(d) for d in own_set))

own_dealers["shares_dealer_with_r1"] = own_dealers["own_dealers"].apply(shares_dealer_with_r1)
log(f"\n  Districts with own dealer data: {len(own_dealers):,}")
log(f"  Districts sharing a dealer with R1 winner: {own_dealers['shares_dealer_with_r1'].sum():,}")
log(f"  Coverage: {100*own_dealers['shares_dealer_with_r1'].mean():.1f}% of districts with dealer info")

df = df.merge(own_dealers[["nces_id","shares_dealer_with_r1"]], on="nces_id", how="left")
df["shares_dealer_with_r1"] = df["shares_dealer_with_r1"].fillna(0).astype(int)
log(f"\n  shares_dealer_with_r1 in full dataset: {df['shares_dealer_with_r1'].sum():,} ({100*df['shares_dealer_with_r1'].mean():.2f}%)")
log(f"  Note: 0 can mean no dealer data (31% coverage) OR no shared dealer")


# =============================================================================
# C — Lat/lon for Conley HAC
# =============================================================================
log("\n" + "=" * 60)
log("C — Lat/lon centroids (WGS84) from EDGE shapefile")
log("=" * 60)

shp = gpd.read_file(SHP_FILE)
shp["nces_id"] = shp["GEOID"].astype(str).str.zfill(7)

# INTPTLAT / INTPTLON are internal point coordinates in the shapefile
latlon = shp[["nces_id","INTPTLAT","INTPTLON"]].copy()
latlon["lat"] = pd.to_numeric(latlon["INTPTLAT"], errors="coerce")
latlon["lon"] = pd.to_numeric(latlon["INTPTLON"], errors="coerce")
latlon = latlon[["nces_id","lat","lon"]].drop_duplicates("nces_id")

log(f"  Shapefile districts with lat/lon: {latlon['lat'].notna().sum():,}")
df = df.merge(latlon, on="nces_id", how="left")
log(f"  Matched in dataset: {df['lat'].notna().sum():,} / {len(df):,}")


# =============================================================================
# D — Quartile variables for priority decomposition
# =============================================================================
log("\n" + "=" * 60)
log("D — PM2.5 and poverty quartiles")
log("=" * 60)

for var, qcol in [("pm25","pm25_q"), ("poverty_rate","poverty_q")]:
    df[qcol] = pd.qcut(df[var], 4, labels=["Q1","Q2","Q3","Q4"])
    df[qcol] = df[qcol].astype(str).replace("nan","Unknown")
    log(f"  {qcol} distribution:")
    log(f"    {df[qcol].value_counts().sort_index().to_dict()}")


# =============================================================================
# Save
# =============================================================================
out_path = CLEAN / "analysis_dataset_supp.csv"
df.to_csv(out_path, index=False)
log(f"\n  Saved: {out_path}")
log(f"  Shape: {df.shape}")
log(f"  New columns: Y_R2_apply, Y_any_post_r1, shares_dealer_with_r1, lat, lon, pm25_q, poverty_q")

with open(LOGS_DIR / "supplementary_audit.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"  Saved: {LOGS_DIR / 'supplementary_audit.txt'}")
