"""
map_02_peer_exposure.py
========================
National choropleth of the KEY INSTRUMENT: share of K=6 nearest neighbors
who won the R1 CSBP lottery (w6_IV_Z_R1).

Also draws a second panel with the reduced-form outcome: Y_R3_apply rate
aggregated to the state level to show geographic concentration of R3 activity.

Output: 3_Output/Figures/map_02_peer_exposure.png  (300 dpi, poster-ready)
"""
import sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SPATIAL_DATASET, SHP_FILE, FIGURES_DIR


CONUS_XLIM = (-2.45e6, 2.35e6)
CONUS_YLIM = (1.5e5,   3.30e6)
C_BG = "#f4f4f0"

EXCLUDE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}


def load_data():
    shp = gpd.read_file(SHP_FILE)
    shp = shp.rename(columns={"GEOID": "nces_id"})
    shp["nces_id"] = shp["nces_id"].str.zfill(7)
    # Filter to CONUS BEFORE reprojecting to reduce memory footprint
    shp["fips2_pre"] = shp["nces_id"].str[:2]
    shp = shp[~shp["fips2_pre"].isin(EXCLUDE_FIPS)].copy()
    shp = shp.to_crs(epsg=5070)
    shp["geometry"] = shp.geometry.simplify(500)   # 500 m — fine for poster display

    sp = pd.read_csv(SPATIAL_DATASET, low_memory=False, dtype={"nces_id": str})
    sp["nces_id"] = sp["nces_id"].str.zfill(7)

    gdf = shp[["nces_id", "geometry"]].merge(
        sp[["nces_id", "w6_IV_Z_R1", "Y_R3_apply", "IV_Z_R1", "state"]],
        on="nces_id", how="left"
    )
    gdf["fips2"] = gdf["nces_id"].str[:2]
    gdf = gdf[~gdf["fips2"].isin(EXCLUDE_FIPS)].copy()

    # State outlines for overlay
    states = gdf.dissolve(by="fips2").reset_index()[["fips2", "geometry"]]
    return gdf, states


def choropleth_continuous(ax, gdf, states, col, cmap_name, title,
                           vmin=None, vmax=None, label="",
                           missing_color="#cccccc"):
    """Continuous color choropleth for a scalar variable."""
    ax.set_facecolor(C_BG)
    ax.set_xlim(*CONUS_XLIM)
    ax.set_ylim(*CONUS_YLIM)
    ax.axis("off")

    vals = gdf[col]
    if vmin is None: vmin = vals.quantile(0.02)
    if vmax is None: vmax = vals.quantile(0.98)

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # 1. Plot districts colored by value (no edge lines — too cluttered at district scale)
    has_data = gdf[col].notna()
    gdf[has_data].plot(
        ax=ax, column=col, cmap=cmap_name,
        vmin=vmin, vmax=vmax,
        linewidth=0.0, edgecolor="none", zorder=1,
        missing_kwds={"color": missing_color, "linewidth": 0.0}
    )
    gdf[~has_data].plot(
        ax=ax, color=missing_color, linewidth=0.0, edgecolor="none", zorder=1
    )

    # 2. State outlines on top for geographic context
    states.boundary.plot(ax=ax, linewidth=0.7, edgecolor="#555555", zorder=2)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.03, pad=0.02, aspect=35, shrink=0.6)
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)


def state_aggregate_panel(ax, gdf, col, cmap_name, title, label=""):
    """Aggregate col by state polygon, then draw a state-level choropleth."""
    state_rates = (gdf.groupby("state")[col]
                      .mean()
                      .reset_index()
                      .rename(columns={col: "rate"}))

    gdf2 = gdf.copy()
    gdf2["state"] = gdf2["state"].fillna("UNKNOWN")
    state_gdf = gdf2.dissolve(by="state", aggfunc="first").reset_index()
    state_gdf = state_gdf[["state", "geometry"]].merge(state_rates, on="state", how="left")

    ax.set_facecolor(C_BG)
    ax.set_xlim(*CONUS_XLIM)
    ax.set_ylim(*CONUS_YLIM)
    ax.axis("off")

    vmin = state_gdf["rate"].quantile(0.05)
    vmax = state_gdf["rate"].quantile(0.95)
    state_gdf.plot(ax=ax, column="rate", cmap=cmap_name,
                   vmin=vmin, vmax=vmax,
                   linewidth=0.6, edgecolor="white", zorder=1,
                   missing_kwds={"color": "#cccccc"})

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.03, pad=0.02, aspect=35, shrink=0.6)
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)


def main():
    print("Loading data...")
    gdf, states = load_data()

    # Exclude R1 winners from estimation sample (same as analysis)
    est = gdf[(gdf["IV_Z_R1"] == 0) | gdf["IV_Z_R1"].isna()].copy()
    print(f"  CONUS districts (non-R1): {len(est):,}")
    print(f"  w6_IV_Z_R1 mean: {est['w6_IV_Z_R1'].mean():.4f}")
    print(f"  Y_R3_apply mean: {est['Y_R3_apply'].mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(24, 9), dpi=150, facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.06, wspace=0.06)

    # Panel A: Instrument — neighbor R1 winner share (continuous)
    choropleth_continuous(
        axes[0], est, states, col="w6_IV_Z_R1",
        cmap_name="YlGn",
        title="Panel A: Neighbor R1 Winner Exposure\n(w6_IV_Z_R1, K=6 KNN, non-R1 districts)",
        label="Share of 6 nearest neighbors who won R1 lottery",
        vmin=0, vmax=0.5
    )

    # Panel B: Outcome — R3 application rate by state
    state_aggregate_panel(
        axes[1], est, col="Y_R3_apply",
        cmap_name="YlOrRd",
        title="Panel B: R3 Application Rate by State\n(non-R1 districts)",
        label="State average R3 application rate"
    )

    fig.suptitle(
        "Spatial Distribution of Instrument (R1 Neighbor Exposure) and Outcome (R3 Application)\n"
        "Non-R1-Winner Districts — CONUS",
        fontsize=15, fontweight="bold", y=0.97
    )

    out = FIGURES_DIR / "map_02_peer_exposure.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
