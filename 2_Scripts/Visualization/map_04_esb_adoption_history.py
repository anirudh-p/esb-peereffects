"""
map_04_esb_adoption_history.py
================================
Motivation map: all US school districts that have ever received an electric
school bus (all funding sources), colored by their FIRST year of adoption.

Uses WRI bus-level data (ESB_adoption_dataset_v9), `3p. Quarter awarded`
to determine award year per district.  Only districts with at least one
WRI-tracked bus are shown; all others are light grey.

Output: 3_Output/Figures/map_04_esb_adoption_history.png  (300 dpi)
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
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WRI_BUS_EXCEL, SHP_FILE, FIGURES_DIR

# ── CONUS display bounds (corrected EPSG:5070) ───────────────────────────────
EXCLUDE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
CONUS_XLIM   = (-2.45e6, 2.35e6)
CONUS_YLIM   = (1.5e5,   3.30e6)

C_NONE = "#dedede"   # districts with no WRI bus
C_BG   = "#eef2f7"   # panel background


def load_wri_first_year():
    """Return dict nces_id → first award year from WRI bus-level data."""
    print("  Reading WRI bus-level data...")
    buses = pd.read_excel(WRI_BUS_EXCEL, sheet_name="2. Bus-level data", dtype=str)
    buses["nces_id"] = buses["1c. LEA ID"].str.strip().str.zfill(7)

    # Award year from "3p. Quarter awarded" (format "YYYY QN")
    def parse_year(s):
        try:
            return int(str(s).strip().split()[0])
        except Exception:
            return np.nan

    buses["award_year"] = buses["3p. Quarter awarded"].apply(parse_year)
    buses = buses.dropna(subset=["award_year"])
    buses["award_year"] = buses["award_year"].astype(int)

    first_year = (buses.groupby("nces_id")["award_year"]
                       .min()
                       .reset_index()
                       .rename(columns={"award_year": "first_year"}))

    print(f"  Districts with WRI bus: {len(first_year):,}")
    print("  First-year distribution:")
    print(first_year["first_year"].value_counts().sort_index().to_string())
    return first_year


def main():
    print("Building ESB adoption history map...")

    first_year = load_wri_first_year()

    print("  Loading shapefile...")
    shp = gpd.read_file(SHP_FILE)
    shp = shp.rename(columns={"GEOID": "nces_id"})
    shp["nces_id"] = shp["nces_id"].str.zfill(7)
    # Filter to CONUS BEFORE reprojecting to reduce memory footprint
    shp["fips2"] = shp["nces_id"].str[:2]
    shp_conus = shp[~shp["fips2"].isin(EXCLUDE_FIPS)].copy()
    shp_conus = shp_conus.to_crs(epsg=5070)
    shp_conus["geometry"] = shp_conus.geometry.simplify(500)

    # State outlines
    states = shp_conus.dissolve(by="fips2").reset_index()[["fips2", "geometry"]]

    # Merge
    gdf = shp_conus[["nces_id", "geometry"]].merge(first_year, on="nces_id", how="left")

    # ── color scheme: one color per year cohort ───────────────────────────────
    years = sorted(gdf["first_year"].dropna().unique().astype(int))
    print(f"\n  Adoption years in data: {years}")

    # Group pre-2019 into one "early" bucket for visual clarity
    gdf["year_display"] = gdf["first_year"].where(
        gdf["first_year"] >= 2016, other=np.nan
    ).where(
        gdf["first_year"].notna(), other=np.nan
    )

    # Use a sequential colormap for years 2016–2024
    year_min, year_max = 2016, 2024
    cmap = plt.get_cmap("plasma_r", year_max - year_min + 1)
    norm = BoundaryNorm(
        boundaries=list(range(year_min, year_max + 2)),
        ncolors=cmap.N
    )

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(18, 10), dpi=150, facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.88, top=0.90, bottom=0.06)

    ax.set_facecolor(C_BG)
    ax.set_xlim(*CONUS_XLIM)
    ax.set_ylim(*CONUS_YLIM)
    ax.axis("off")

    # 1. Grey background for all districts
    gdf.plot(ax=ax, color=C_NONE, linewidth=0.0, edgecolor="none", zorder=1)

    # 2. State outlines
    states.boundary.plot(ax=ax, linewidth=0.55, edgecolor="#777777", zorder=2)

    # 3. Adopter districts colored by first year (most recent year on top)
    adopters = gdf[gdf["year_display"].notna()].copy()
    adopters["year_int"] = adopters["year_display"].astype(int)

    for yr in range(year_min, year_max + 1):
        sub = adopters[adopters["year_int"] == yr]
        if len(sub) == 0:
            continue
        color = cmap(norm(yr))
        sub.plot(ax=ax, color=color, linewidth=0.4, edgecolor="white", zorder=3)

    # 4. Pre-2016 adopters in darkest color
    early = gdf[gdf["first_year"].notna() & (gdf["first_year"] < year_min)]
    if len(early):
        early.plot(ax=ax, color=cmap(0), linewidth=0.4, edgecolor="white", zorder=3)

    # ── colorbar ──────────────────────────────────────────────────────────────
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.90, 0.20, 0.025, 0.55])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("First year of ESB adoption\n(any funding source)", fontsize=11, labelpad=10)
    tick_locs = [y + 0.5 for y in range(year_min, year_max + 1)]
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([str(y) for y in range(year_min, year_max + 1)])
    cbar.ax.tick_params(labelsize=10)

    # ── legend for non-adopters ───────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=cmap(norm(year_min)),   label=f"Early adopters (\u2264{year_min})"),
        mpatches.Patch(facecolor=cmap(norm(year_max)),   label=f"Recent adopters ({year_max})"),
        mpatches.Patch(facecolor=C_NONE,                 label="No WRI-tracked ESB"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=10,
              frameon=True, framealpha=0.9, edgecolor="#cccccc",
              bbox_to_anchor=(0.01, 0.02))

    n_adopters = gdf["first_year"].notna().sum()
    n_total    = len(gdf)
    ax.set_title(
        f"Electric School Bus Adoption by District — All Funding Sources\n"
        f"WRI ESB Dataset (as of June 2025)   |   "
        f"{n_adopters:,} of {n_total:,} districts have at least one ESB",
        fontsize=14, fontweight="bold", pad=10
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "map_04_esb_adoption_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
