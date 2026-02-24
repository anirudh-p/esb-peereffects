import os
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
# Point this to your unzipped shapefile

HOME_PATH = r"C:\BC PhD\Research\Peer-Effects and Adoption"

# Build absolute paths relative to HOME_PATH so the script works on Windows
SHAPEFILE_PATH = os.path.join(HOME_PATH, 'EDGE_GEOCODE_PUBLICSCH_2223', 'Shapefiles_SCH', 'EDGE_GEOCODE_PUBLICSCH_2223.shp')

# Point this to your CSV of adopters (relative to HOME_PATH)
ADOPTER_DATA_PATH = os.path.join(HOME_PATH, 'merged.csv')
# What is the LEA ID column called in each file?
SHAPEFILE_ID_COL = 'LEAID' # Or 'LEAID' - check this by opening the .dbf file in Excel
ADOPTER_ID_COL = 'LEAID'

# --- 2. LOAD DATA ---
print("Loading shapefile...")
# Load the US school district boundaries
# This might take a moment
shp_path = Path(SHAPEFILE_PATH)
if not shp_path.exists():
    print(f"ERROR: Shapefile not found at {shp_path}")
    print(f"Expected location: {shp_path}\nMake sure the 'EDGE_GEOCODE_PUBLICSCH_2223' folder is present under HOME_PATH or update SHAPEFILE_PATH.")
    sys.exit(1)

try:
    # Load only the ID and geometry columns to speed up reading large shapefiles.
    districts_gdf = gpd.read_file(str(shp_path), columns=[SHAPEFILE_ID_COL])
except TypeError:
    # Older geopandas/pyogrio versions may not support the 'columns' argument; fall
    # back to reading the entire shapefile in that case.
    districts_gdf = gpd.read_file(str(shp_path))

print("Loading adopter data...")
# Load your CSV (adopters list)
adopter_path = Path(ADOPTER_DATA_PATH)
if not adopter_path.exists():
    print(f"ERROR: Adopter CSV not found at {adopter_path}")
    print("Expected file: merged.csv in the project root. Update ADOPTER_DATA_PATH if needed.")
    sys.exit(1)

adopters_df = pd.read_csv(str(adopter_path), low_memory=False)

# If adopters_df doesn't explicitly have an 'Adopted' column, assume any row in the
# adopters CSV represents an adopter and set Adopted=1. This makes the merge robust.
if 'Adopted' not in adopters_df.columns:
    adopters_df['Adopted'] = 1

# If the expected adopter ID column isn't present, try to auto-detect a column that
# looks like an LEA/ID column (e.g., '1c. LEA ID', 'LEAID', 'LEA ID'). This makes the
# script more flexible with various CSV formats.
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
        print(f"ERROR: Could not find adopter ID column '{ADOPTER_ID_COL}' in CSV columns.")
        print("CSV columns available:", list(adopters_df.columns))
        sys.exit(1)

# Keep only the ID and Adopted columns from the adopters CSV to avoid merging in
# dozens of unnecessary columns (the raw CSV contains many survey fields).
adopters_df = adopters_df[[ADOPTER_ID_COL, 'Adopted']].drop_duplicates(subset=ADOPTER_ID_COL)

# --- 3. PREPARE & MERGE DATA ---
# Ensure the IDs are the same data type for merging (e.g., string)
districts_gdf[SHAPEFILE_ID_COL] = districts_gdf[SHAPEFILE_ID_COL].astype(str)
adopters_df[ADOPTER_ID_COL] = adopters_df[ADOPTER_ID_COL].astype(str)

# Perform the "join"
# This merges your adopter data onto the geographic data
print("Merging datasets...")
merged_gdf = districts_gdf.merge(
    adopters_df,
    left_on=SHAPEFILE_ID_COL,
    right_on=ADOPTER_ID_COL,
    how='left'
)

# Fill in non-adopters
# The 'left' merge will create 'NaN' for districts not in your adopter list.
# We'll fill those NaNs with '0' (non-adopter)
merged_gdf['Adopted'] = merged_gdf['Adopted'].fillna(0)

# --- 4. CREATE THE MAP ---
print("Generating map...")
fig, ax = plt.subplots(figsize=(15, 10))

# We'll create two layers:
# 1. A base layer of all districts in light gray
# Simplify geometries for faster plotting if the dataset is large. This will
# speed up rendering and reduce memory/cpu usage when saving the figure.
plot_gdf = merged_gdf.copy()
row_count = len(plot_gdf)
if row_count > 2000:
    tol = 0.01
else:
    tol = 0.001
try:
    plot_gdf['geometry'] = plot_gdf['geometry'].simplify(tolerance=tol, preserve_topology=True)
except Exception:
    # If simplification fails for any geometry, continue with original geometries.
    pass

plot_gdf.plot(
    ax=ax,
    color='#cccccc', # Light gray for non-adopters
    edgecolor='none'
)

# 2. An overlay of *only* the adopters in blue
adopters_map = merged_gdf[merged_gdf['Adopted'] == 1]
adopters_map.plot(
    ax=ax,
    color='#0073e6', # A strong blue for adopters
    edgecolor='none'
)

# --- 5. STYLE AND SAVE ---
# Clean up the map
ax.set_title('Electric School Bus Adoptions by District (2022-2023)', fontsize=18)
ax.axis('off') # Remove the lat/lon axes

# Optional: Zoom into a specific region
# ax.set_xlim(-80, -72) # Example: Longitude for East Coast
# ax.set_ylim(38, 43)   # Example: Latitude for NY/MD/PA

# Save the figure
output_path = 'district_cluster_map.png'
# Reduce DPI to speed up rendering, and use the simplified GeoDataFrame when
# saving.
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Map saved to {output_path}")

# Pro-tip: To add a basemap (like Google Maps), explore the 'contextily' library
# import contextily as ctx
# ctx.add_basemap(ax, crs=merged_gdf.crs.to_string())