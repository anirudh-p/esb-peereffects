"""
Version 2: Number of ESBs Map
Shows school districts colored by number of electric school buses with choropleth styling.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# --- CONFIGURATION ---
HOME_PATH = r"C:\BC PhD\Research\Peer-Effects and Adoption"

SHAPEFILE_PATH = os.path.join(HOME_PATH, 'EDGE_GEOCODE_PUBLICSCH_2223', 'Shapefiles_SCH', 'EDGE_GEOCODE_PUBLICSCH_2223.shp')
ADOPTER_DATA_PATH = os.path.join(HOME_PATH, 'merged.csv')

SHAPEFILE_ID_COL = 'LEAID'
ADOPTER_ID_COL = 'LEAID'

OUTPUT_PATH = os.path.join(HOME_PATH, 'plots', 'map_v2_esb_count.png')

# --- LOAD DATA ---
print("Loading shapefile...")
shp_path = Path(SHAPEFILE_PATH)
if not shp_path.exists():
    print(f"ERROR: Shapefile not found at {shp_path}")
    sys.exit(1)

try:
    districts_gdf = gpd.read_file(str(shp_path), columns=[SHAPEFILE_ID_COL])
except TypeError:
    districts_gdf = gpd.read_file(str(shp_path))

print("Loading adopter data...")
adopter_path = Path(ADOPTER_DATA_PATH)
if not adopter_path.exists():
    print(f"ERROR: Adopter CSV not found at {adopter_path}")
    sys.exit(1)

adopters_df = pd.read_csv(str(adopter_path), low_memory=False)

# Auto-detect adopter ID column
if ADOPTER_ID_COL not in adopters_df.columns:
    found = None
    for col in adopters_df.columns:
        lc = col.lower().replace('_', ' ')
        if 'lea' in lc and 'id' in lc:
            found = col
            break
    if found:
        print(f"Using detected adopter ID column '{found}' from CSV")
        ADOPTER_ID_COL = found
    else:
        print(f"ERROR: Could not find adopter ID column")
        sys.exit(1)

# Aggregate ESB counts by district
print("Aggregating ESB counts by district...")
esb_col = '3b. Number of delivered or operating ESBs'
if esb_col not in adopters_df.columns:
    print(f"ERROR: ESB count column '{esb_col}' not found")
    sys.exit(1)

district_esb = adopters_df.groupby(ADOPTER_ID_COL)[esb_col].sum().reset_index()
district_esb.columns = [ADOPTER_ID_COL, 'ESB_Count']

# --- MERGE DATA ---
print("Merging datasets...")
districts_gdf[SHAPEFILE_ID_COL] = districts_gdf[SHAPEFILE_ID_COL].astype(str)
district_esb[ADOPTER_ID_COL] = district_esb[ADOPTER_ID_COL].astype(str)

merged_gdf = districts_gdf.merge(
    district_esb,
    left_on=SHAPEFILE_ID_COL,
    right_on=ADOPTER_ID_COL,
    how='left'
)

merged_gdf['ESB_Count'] = merged_gdf['ESB_Count'].fillna(0)

# Reproject to Web Mercator
print("Reprojecting to Web Mercator (EPSG:3857)...")
merged_gdf = merged_gdf.to_crs(epsg=3857)

# Simplify geometries
print("Simplifying geometries...")
merged_gdf['geometry'] = merged_gdf['geometry'].simplify(tolerance=1000, preserve_topology=True)

# --- CREATE MAP ---
print("Generating map...")
fig, ax = plt.subplots(figsize=(20, 14))

# Create custom colormap (white to dark blue)
colors = ['#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('esb_cmap', colors, N=n_bins)

# Plot with graduated colors
merged_gdf.plot(
    ax=ax,
    column='ESB_Count',
    cmap=cmap,
    edgecolor='#999999',
    linewidth=0.1,
    alpha=0.85,
    legend=True,
    legend_kwds={
        'label': 'Number of Electric School Buses',
        'orientation': 'horizontal',
        'shrink': 0.5,
        'pad': 0.05
    }
)

# Add basemap (commented out for faster generation - uncomment if desired)
print("Skipping basemap for faster generation...")
# try:
#     ctx.add_basemap(
#         ax,
#         crs=merged_gdf.crs.to_string(),
#         source=ctx.providers.CartoDB.Positron,
#         alpha=0.4
#     )
# except Exception as e:
#     print(f"Warning: Could not add basemap: {e}")

# Styling
ax.set_title(
    'Electric School Bus Fleet Size by District (2017-2023)',
    fontsize=24,
    fontweight='bold',
    pad=20
)
ax.axis('off')

# Add statistics text
total_esb = merged_gdf['ESB_Count'].sum()
districts_with_esb = len(merged_gdf[merged_gdf['ESB_Count'] > 0])
avg_esb = merged_gdf[merged_gdf['ESB_Count'] > 0]['ESB_Count'].mean()

stats_text = f'Total ESBs: {int(total_esb):,}\n'
stats_text += f'Districts with ESBs: {districts_with_esb:,}\n'
stats_text += f'Average ESBs per adopting district: {avg_esb:.1f}'

ax.text(
    0.02, 0.98,
    stats_text,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
)

# Add attribution
fig.text(
    0.99, 0.01,
    'Data: EPA Clean School Bus Program | Basemap: CartoDB',
    ha='right',
    fontsize=10,
    style='italic',
    color='gray'
)

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Map saved to {OUTPUT_PATH}")

plt.close()
