"""
Version 1: Adopters vs Non-Adopters Map
Shows school districts colored by adoption status with professional styling and basemap.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

# Use non-interactive backend to avoid display issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import Patch

# --- CONFIGURATION ---
HOME_PATH = r"C:\BC PhD\Research\Peer-Effects and Adoption"

SHAPEFILE_PATH = os.path.join(HOME_PATH, 'EDGE_GEOCODE_PUBLICSCH_2223', 'Shapefiles_SCH', 'EDGE_GEOCODE_PUBLICSCH_2223.shp')
ADOPTER_DATA_PATH = os.path.join(HOME_PATH, 'merged.csv')

SHAPEFILE_ID_COL = 'LEAID'
ADOPTER_ID_COL = 'LEAID'

OUTPUT_PATH = os.path.join(HOME_PATH, 'plots', 'map_v1_adopters_vs_non_adopters.png')

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
if 'Adopted' not in adopters_df.columns:
    adopters_df['Adopted'] = 1

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

# Keep only essential columns
adopters_df = adopters_df[[ADOPTER_ID_COL, 'Adopted']].drop_duplicates(subset=ADOPTER_ID_COL)

# --- MERGE DATA ---
print("Merging datasets...")
districts_gdf[SHAPEFILE_ID_COL] = districts_gdf[SHAPEFILE_ID_COL].astype(str)
adopters_df[ADOPTER_ID_COL] = adopters_df[ADOPTER_ID_COL].astype(str)

merged_gdf = districts_gdf.merge(
    adopters_df,
    left_on=SHAPEFILE_ID_COL,
    right_on=ADOPTER_ID_COL,
    how='left'
)

merged_gdf['Adopted'] = merged_gdf['Adopted'].fillna(0)

# Reproject to Web Mercator for basemap compatibility
print("Reprojecting to Web Mercator (EPSG:3857)...")
merged_gdf = merged_gdf.to_crs(epsg=3857)

# Simplify geometries for faster rendering
print("Simplifying geometries...")
merged_gdf['geometry'] = merged_gdf['geometry'].simplify(tolerance=1000, preserve_topology=True)

# --- CREATE MAP ---
print("Generating map...")
fig, ax = plt.subplots(figsize=(20, 14))

# Plot non-adopters in light gray
non_adopters = merged_gdf[merged_gdf['Adopted'] == 0]
non_adopters.plot(
    ax=ax,
    color='#e8e8e8',
    edgecolor='#d0d0d0',
    linewidth=0.1,
    alpha=0.7
)

# Plot adopters in blue
adopters = merged_gdf[merged_gdf['Adopted'] == 1]
adopters.plot(
    ax=ax,
    color='#1f77b4',
    edgecolor='#0d5a99',
    linewidth=0.3,
    alpha=0.8
)

# Add basemap (commented out for faster generation - uncomment if desired)
print("Skipping basemap for faster generation...")
# try:
#     ctx.add_basemap(
#         ax,
#         crs=merged_gdf.crs.to_string(),
#         source=ctx.providers.CartoDB.Positron,
#         alpha=0.5
#     )
# except Exception as e:
#     print(f"Warning: Could not add basemap: {e}")

# Styling
ax.set_title(
    'Electric School Bus Adoption by School District (2017-2023)',
    fontsize=24,
    fontweight='bold',
    pad=20
)
ax.axis('off')

# Add legend
legend_elements = [
    Patch(facecolor='#1f77b4', edgecolor='#0d5a99', label=f'Adopters (n={len(adopters):,})'),
    Patch(facecolor='#e8e8e8', edgecolor='#d0d0d0', label=f'Non-Adopters (n={len(non_adopters):,})')
]
ax.legend(
    handles=legend_elements,
    loc='lower right',
    fontsize=14,
    frameon=True,
    fancybox=True,
    shadow=True
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
