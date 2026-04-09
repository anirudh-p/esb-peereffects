"""
map_03_california_zoom.py
==========================
Single combined California map showing the delivery-timing mechanism:
  - R3 applicant districts (amber, non-R1) — outcome
  - R1 winner districts colored by delivery timing, overlaid — treatment:
      green  = pre_r3  (delivered ≤ 2023 Q3, visible before R3 deadline)
      red    = post_r3 (delivered ≥ 2023 Q4, invisible at R3 decision time)
      grey   = unknown delivery date

Seeing both layers on one map lets the viewer directly assess whether R3
applicants cluster near pre-deadline R1 deployments (the mechanism story).

Output: 3_Output/Figures/map_03_california_zoom.png  (150 dpi, poster-ready)
"""
import sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SPATIAL_DATASET, SHP_FILE, FIGURES_DIR, CSB_REBATES, WRI_BUS_EXCEL


# ── Colors ───────────────────────────────────────────────────────────────────
C_PRE_R3  = "#1a7d40"   # green  — visible before R3 deadline
C_POST_R3 = "#e74c3c"   # red    — arrived after R3 deadline
C_UNKNOWN = "#95a5a6"   # grey   — delivery date unknown
C_R3_APP  = "#f39c12"   # amber  — R3 applicant (non-R1)
C_NONE    = "#ececec"   # light grey — no activity


def build_delivery_categories():
    """Reconstruct the sharp delivery cutoff from WRI bus-level data."""
    reb = pd.read_excel(CSB_REBATES, dtype=str)
    reb.columns = reb.columns.str.strip().str.lower().str.replace(" ", "_")
    r1_ids = set(
        reb.loc[reb["funding_year"] == "2022", "nces_district_id"]
           .str.strip().str.zfill(7)
    )

    buses = pd.read_excel(WRI_BUS_EXCEL, sheet_name="2. Bus-level data", dtype=str)
    buses["nces_id"] = buses["1c. LEA ID"].str.strip().str.zfill(7)

    def parse_qtr(s):
        try:
            parts = str(s).strip().split()
            return int(parts[0]), int(parts[1].replace("Q", ""))
        except Exception:
            return np.nan, np.nan

    buses[["del_year", "del_q"]] = buses["3r. Quarter delivered"].apply(
        lambda x: pd.Series(parse_qtr(x))
    )

    def delivery_cat(row):
        y, q = row["del_year"], row["del_q"]
        if pd.isna(y):
            return "unknown"
        y, q = int(y), int(q)
        if y < 2023 or (y == 2023 and q <= 3):
            return "pre_r3"
        return "post_r3"

    buses["del_cat"] = buses.apply(delivery_cat, axis=1)

    r1_buses = buses[buses["nces_id"].isin(r1_ids)].copy()
    cat_order = {"pre_r3": 0, "post_r3": 1, "unknown": 2}
    r1_buses["cat_rank"] = r1_buses["del_cat"].map(cat_order)

    r1_del = (r1_buses.sort_values("cat_rank")
                      .groupby("nces_id")
                      .first()[["del_cat"]]
                      .rename(columns={"del_cat": "r1_del_cat"})
                      .reset_index())
    return r1_del


def main():
    print("Building delivery categories...")
    r1_del = build_delivery_categories()

    print("Loading spatial dataset...")
    sp = pd.read_csv(SPATIAL_DATASET, low_memory=False, dtype={"nces_id": str})
    sp["nces_id"] = sp["nces_id"].str.zfill(7)

    print("Loading shapefile...")
    shp = gpd.read_file(SHP_FILE)
    shp = shp.rename(columns={"GEOID": "nces_id"})
    shp["nces_id"] = shp["nces_id"].str.zfill(7)
    # Filter to CA BEFORE reprojecting — avoids memory spike on full shapefile
    shp_ca = shp[shp["nces_id"].str.startswith("06")].copy()
    shp_ca = shp_ca.to_crs(epsg=5070)
    sp_ca  = sp[sp["nces_id"].str.startswith("06")].copy()

    df = sp_ca[["nces_id", "IV_Z_R1", "Y_R3_apply"]].merge(
        r1_del, on="nces_id", how="left"
    )

    def classify(row):
        if row["IV_Z_R1"] == 1:
            cat = row.get("r1_del_cat", np.nan)
            if cat == "pre_r3":  return "pre_r3"
            if cat == "post_r3": return "post_r3"
            return "r1_unknown"
        if row["Y_R3_apply"] == 1:
            return "r3_applicant"
        return "none"

    df["display_cat"] = df.apply(classify, axis=1)

    gdf = shp_ca[["nces_id", "geometry"]].merge(df, on="nces_id", how="left")
    gdf["display_cat"] = gdf["display_cat"].fillna("none")

    n_pre     = (gdf["display_cat"] == "pre_r3").sum()
    n_post    = (gdf["display_cat"] == "post_r3").sum()
    n_unk     = (gdf["display_cat"] == "r1_unknown").sum()
    n_r3      = (gdf["display_cat"] == "r3_applicant").sum()
    print(f"CA districts:        {len(gdf):,}")
    print(f"  pre_r3 R1:         {n_pre}")
    print(f"  post_r3 R1:        {n_post}")
    print(f"  R1 unknown timing: {n_unk}")
    print(f"  R3 applicants:     {n_r3}")

    # ── single combined map ───────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=150, facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.12)

    ax.set_facecolor("#dde8f0")
    ax.axis("off")

    # 1. Grey base
    gdf.plot(ax=ax, color=C_NONE, linewidth=0.12, edgecolor="#bbbbbb", zorder=1)

    # 2. R3 applicants (outcome) — amber, drawn below R1 timing layers
    r3 = gdf[gdf["display_cat"] == "r3_applicant"]
    if len(r3):
        r3.plot(ax=ax, color=C_R3_APP, alpha=0.9, linewidth=0.3,
                edgecolor="white", zorder=2)

    # 3. R1 winners by delivery timing — on top so they're clearly visible
    for cat, color, zord in [
        ("r1_unknown", C_UNKNOWN, 3),
        ("post_r3",    C_POST_R3, 4),
        ("pre_r3",     C_PRE_R3,  5),
    ]:
        sub = gdf[gdf["display_cat"] == cat]
        if len(sub):
            sub.plot(ax=ax, color=color, alpha=0.95, linewidth=0.4,
                     edgecolor="white", zorder=zord)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_entries = [
        mpatches.Patch(color=C_PRE_R3,
                       label=f"R1 winner — pre-deadline delivery ($\\leq$2023 Q3) [N={n_pre}]"),
        mpatches.Patch(color=C_POST_R3,
                       label=f"R1 winner — post-deadline delivery ($\\geq$2023 Q4) [N={n_post}]"),
        mpatches.Patch(color=C_UNKNOWN,
                       label=f"R1 winner — delivery date unknown [N={n_unk}]"),
        mpatches.Patch(color=C_R3_APP,
                       label=f"R3 applicant, non-R1 winner [N={n_r3}]"),
        mpatches.Patch(color=C_NONE,
                       label="No program activity"),
    ]
    ax.legend(handles=legend_entries, loc="lower left", fontsize=9,
              frameon=True, framealpha=0.92, edgecolor="#cccccc",
              bbox_to_anchor=(0.01, 0.01))

    ax.set_title(
        "California: R1 Delivery Timing and R3 Application Activity\n"
        "Do R3 applicants (amber) cluster near pre-deadline R1 deployments (green)?",
        fontsize=13, fontweight="bold", pad=10
    )

    out = FIGURES_DIR / "map_03_california_zoom.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
