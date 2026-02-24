"""
Version 4: Temporal Snapshots - Adoption Over Time
Creates multiple maps showing adoption patterns at different time periods (by year).
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
from matplotlib.patches import Patch

# --- CONFIGURATION ---
HOME_PATH = r"C:\BC PhD\Research\Peer-Effects and Adoption"

SHAPEFILE_PATH = os.path.join(HOME_PATH, 'EDGE_GEOCODE_PUBLICSCH_2223', 'Shapefiles_SCH', 'EDGE_GEOCODE_PUBLICSCH_2223.shp')
ADOPTER_DATA_PATH = os.path.join(HOME_PATH, 'merged.csv')

SHAPEFILE_ID_COL = 'LEAID'
ADOPTER_ID_COL = 'LEAID'

OUTPUT_DIR = os.path.join(HOME_PATH, 'plots', 'temporal_snapshots')

# Years to create snapshots for
SNAPSHOT_YEARS = [2019, 2020, 2021, 2022, 2023]

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

# Extract year from quarter awarded column
quarter_col = '3p. Quarter awarded'
if quarter_col not in adopters_df.columns:
    print(f"ERROR: Quarter column '{quarter_col}' not found")
    sys.exit(1)

# Parse year from quarter string (e.g., "2020 Q1" -> 2020)
def extract_year(quarter_str):
    if pd.isna(quarter_str):
        return None
    try:
        return int(str(quarter_str).split()[0])
    except:
        return None

adopters_df['Award_Year'] = adopters_df[quarter_col].apply(extract_year)

# Prepare base shapefile
print("Preparing base shapefile...")
districts_gdf[SHAPEFILE_ID_COL] = districts_gdf[SHAPEFILE_ID_COL].astype(str)
districts_gdf = districts_gdf.to_crs(epsg=3857)
districts_gdf['geometry'] = districts_gdf['geometry'].simplify(tolerance=1000, preserve_topology=True)

# --- CREATE TEMPORAL MAPS ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

cumulative_adopters = set()

for year in SNAPSHOT_YEARS:
    print(f"\nGenerating map for {year}...")
    
    # Get adopters up to and including this year (cumulative)
    year_adopters = adopters_df[
        (adopters_df['Award_Year'].notna()) & 
        (adopters_df['Award_Year'] <= year)
    ][ADOPTER_ID_COL].astype(str).unique()
    
    cumulative_adopters.update(year_adopters)
    
    # Create adoption indicator
    districts_year = districts_gdf.copy()
    districts_year['Adopted'] = districts_year[SHAPEFILE_ID_COL].isin(cumulative_adopters).astype(int)
    
    # Create map
    fig, ax = plt.subplots(figsize=(20, 14))
    
    # Plot non-adopters
    non_adopters = districts_year[districts_year['Adopted'] == 0]
    non_adopters.plot(
        ax=ax,
        color='#e8e8e8',
        edgecolor='#d0d0d0',
        linewidth=0.1,
        alpha=0.7
    )
    
    # Plot adopters
    adopters = districts_year[districts_year['Adopted'] == 1]
    adopters.plot(
        ax=ax,
        color='#1f77b4',
        edgecolor='#0d5a99',
        linewidth=0.3,
        alpha=0.8
    )
    
    # Add basemap (commented out for faster generation - uncomment if desired)
    # try:
    #     ctx.add_basemap(
    #         ax,
    #         crs=districts_year.crs.to_string(),
    #         source=ctx.providers.CartoDB.Positron,
    #         alpha=0.5
    #     )
    # except Exception as e:
    #     print(f"Warning: Could not add basemap for {year}: {e}")
    
    # Styling
    ax.set_title(
        f'Electric School Bus Adoption Status (Through {year})',
        fontsize=24,
        fontweight='bold',
        pad=20
    )
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='#0d5a99', 
              label=f'Adopters (n={len(adopters):,})'),
        Patch(facecolor='#e8e8e8', edgecolor='#d0d0d0', 
              label=f'Non-Adopters (n={len(non_adopters):,})')
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    # Add year-specific statistics
    year_new = len(adopters_df[adopters_df['Award_Year'] == year][ADOPTER_ID_COL].astype(str).unique())
    
    stats_text = f'Cumulative Adopters: {len(cumulative_adopters):,}\n'
    stats_text += f'New in {year}: {year_new:,}'
    
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
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
    output_path = os.path.join(OUTPUT_DIR, f'adoption_snapshot_{year}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    plt.close()

print(f"\nAll temporal snapshots saved to: {OUTPUT_DIR}")
