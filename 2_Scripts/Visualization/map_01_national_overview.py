"""
map_01_national_overview.py
============================
Single combined national choropleth:
  - R1 lottery winner districts (green, 2022) — treatment
  - R3 applicant districts (orange, 2023) — outcome
  Both shown on the same map; by design R3 applicants exclude R1 winners,
  so the two colours are essentially non-overlapping (rare both→navy).

Output: 3_Output/Figures/map_01_national_overview.png  (150 dpi, poster-ready)
"""
import sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANALYSIS_DATASET, SHP_FILE, FIGURES_DIR

# ── colors ───────────────────────────────────────────────────────────────────
C_R1   = "#1a7d40"   # deep green   — R1 winners (treatment)
C_R3   = "#e67e22"   # burnt orange — R3 applicants (outcome)
C_BOTH = "#1a3a5c"   # dark navy    — R1 winner + R3 applicant
C_NONE = "#d8d8d8"   # light grey   — no activity
C_BG   = "#f0f4f8"   # panel background

EXCLUDE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
CONUS_XLIM   = (-2.45e6, 2.35e6)
CONUS_YLIM   = (1.5e5,   3.30e6)


def load_data():
    shp = gpd.read_file(SHP_FILE)
    shp = shp.rename(columns={"GEOID": "nces_id"})
    shp["nces_id"] = shp["nces_id"].str.zfill(7)
    shp["fips2_pre"] = shp["nces_id"].str[:2]
    shp = shp[~shp["fips2_pre"].isin(EXCLUDE_FIPS)].copy()
    shp = shp.to_crs(epsg=5070)
    shp["geometry"] = shp.geometry.simplify(500)

    df = pd.read_csv(ANALYSIS_DATASET, low_memory=False, dtype={"nces_id": str})
    df["nces_id"] = df["nces_id"].str.zfill(7)

    def classify(row):
        r1, r3 = row["IV_Z_R1"], row["Y_R3_apply"]
        if r1 == 1 and r3 == 1: return "both"
        if r1 == 1:             return "r1_only"
        if r3 == 1:             return "r3_only"
        return "none"

    df["category"] = df.apply(classify, axis=1)

    gdf = shp[["nces_id", "geometry"]].merge(
        df[["nces_id", "IV_Z_R1", "Y_R3_apply", "category"]],
        on="nces_id", how="left"
    )
    gdf["category"] = gdf["category"].fillna("none")
    gdf["fips2"] = gdf["nces_id"].str[:2]
    gdf_conus = gdf[~gdf["fips2"].isin(EXCLUDE_FIPS)].copy()

    states = gdf_conus.dissolve(by="fips2").reset_index()[["fips2", "geometry"]]
    return gdf_conus, states, df


def main():
    print("Loading data...")
    gdf_conus, states, df = load_data()

    n_r1   = int((df["IV_Z_R1"] == 1).sum())
    n_r3   = int((df["Y_R3_apply"] == 1).sum())
    n_both = int(((df["IV_Z_R1"] == 1) & (df["Y_R3_apply"] == 1)).sum())
    print(f"  CONUS districts: {len(gdf_conus):,}")
    print(f"  R1 winners:      {n_r1}")
    print(f"  R3 applicants:   {n_r3}  (non-R1)")
    print(f"  Both:            {n_both}")

    fig, ax = plt.subplots(1, 1, figsize=(14, 9), dpi=150, facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.10)

    ax.set_facecolor(C_BG)
    ax.set_xlim(*CONUS_XLIM)
    ax.set_ylim(*CONUS_YLIM)
    ax.axis("off")

    # 1. Grey base — all districts
    gdf_conus.plot(ax=ax, color=C_NONE, linewidth=0.0, edgecolor="none", zorder=1)

    # 2. State outlines
    states.boundary.plot(ax=ax, linewidth=0.55, edgecolor="#888888", zorder=2)

    # 3. R3 applicants (outcome) — drawn first so R1 winners appear on top
    r3 = gdf_conus[gdf_conus["category"] == "r3_only"]
    if len(r3):
        r3.plot(ax=ax, color=C_R3, linewidth=0.5, edgecolor="white", zorder=3)

    # 4. R1 winners (treatment)
    r1 = gdf_conus[gdf_conus["category"] == "r1_only"]
    if len(r1):
        r1.plot(ax=ax, color=C_R1, linewidth=0.5, edgecolor="white", zorder=4)

    # 5. Rare both — top layer
    both = gdf_conus[gdf_conus["category"] == "both"]
    if len(both):
        both.plot(ax=ax, color=C_BOTH, linewidth=0.5, edgecolor="white", zorder=5)

    # ── legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=C_R1,   label=f"R1 lottery winner — treatment (N\u202f=\u202f{n_r1})"),
        mpatches.Patch(color=C_R3,   label=f"R3 applicant, non-R1 — outcome (N\u202f=\u202f{n_r3})"),
        mpatches.Patch(color=C_BOTH, label=f"R1 winner + R3 applicant (N\u202f=\u202f{n_both})"),
        mpatches.Patch(color=C_NONE, label="No program activity"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=10,
              frameon=True, framealpha=0.9, edgecolor="#cccccc",
              bbox_to_anchor=(0.01, 0.02))

    ax.set_title(
        "R1 Lottery Winners (Treatment) and R3 Applicants (Outcome)\n"
        "Clean School Bus Program — CONUS Districts",
        fontsize=14, fontweight="bold", pad=10
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "map_01_national_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
