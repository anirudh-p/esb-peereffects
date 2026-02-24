"""
Version 3: Adopters with District Size (Student Population)
Shows gray district boundaries with blue circles at centroids sized by student population.
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
import numpy as np

# --- CONFIGURATION ---
HOME_PATH = r"C:\BC PhD\Research\Peer-Effects and Adoption"

SHAPEFILE_PATH = os.path.join(HOME_PATH, 'EDGE_GEOCODE_PUBLICSCH_2223', 'Shapefiles_SCH', 'EDGE_GEOCODE_PUBLICSCH_2223.shp')
ADOPTER_DATA_PATH = os.path.join(HOME_PATH, 'merged.csv')

SHAPEFILE_ID_COL = 'LEAID'
ADOPTER_ID_COL = 'LEAID'

OUTPUT_PATH = os.path.join(HOME_PATH, 'plots', 'map_v3_district_size.png')

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

# Aggregate by district to get student population and adoption status
print("Aggregating district data...")
student_col = '4b. Number of students in district'
esb_col = '3b. Number of delivered or operating ESBs'

if student_col not in adopters_df.columns:
    print(f"ERROR: Student population column '{student_col}' not found")
    sys.exit(1)

# Group by district and get max student population and sum of ESBs
district_data = adopters_df.groupby(ADOPTER_ID_COL).agg({
    student_col: 'max',  # Student population is same for all rows in a district
    esb_col: 'sum'  # Sum ESBs across all records
}).reset_index()

district_data.columns = [ADOPTER_ID_COL, 'Student_Population', 'ESB_Count']
district_data['Adopted'] = (district_data['ESB_Count'] > 0).astype(int)

# --- MERGE DATA ---
print("Merging datasets...")
districts_gdf[SHAPEFILE_ID_COL] = districts_gdf[SHAPEFILE_ID_COL].astype(str)
district_data[ADOPTER_ID_COL] = district_data[ADOPTER_ID_COL].astype(str)

merged_gdf = districts_gdf.merge(
    district_data,
    left_on=SHAPEFILE_ID_COL,
    right_on=ADOPTER_ID_COL,
    how='left'
)

merged_gdf['Adopted'] = merged_gdf['Adopted'].fillna(0)
merged_gdf['Student_Population'] = merged_gdf['Student_Population'].fillna(0)

# Reproject to Web Mercator
print("Reprojecting to Web Mercator (EPSG:3857)...")
merged_gdf = merged_gdf.to_crs(epsg=3857)

# Simplify geometries for base layer
print("Simplifying geometries...")
merged_gdf['geometry'] = merged_gdf['geometry'].simplify(tolerance=1000, preserve_topology=True)

# Calculate centroids for adopting districts
print("Calculating centroids...")
adopters = merged_gdf[merged_gdf['Adopted'] == 1].copy()
adopters['centroid'] = adopters.geometry.centroid

# --- CREATE MAP ---
print("Generating map...")
fig, ax = plt.subplots(figsize=(20, 14))

# Plot base layer: all districts in light gray
merged_gdf.plot(
    ax=ax,
    color='#e8e8e8',
    edgecolor='#c0c0c0',
    linewidth=0.2,
    alpha=0.6
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

# Plot circles at centroids, sized by student population
if len(adopters) > 0:
    # Filter out missing or zero student populations
    adopters_with_pop = adopters[adopters['Student_Population'] > 0].copy()
    
    if len(adopters_with_pop) > 0:
        # Scale circle sizes (use sqrt to make area proportional to population)
        # Normalize to a reasonable size range
        min_size = 10
        max_size = 500
        
        populations = adopters_with_pop['Student_Population'].values
        # Use log scale for better visualization of wide range
        log_pop = np.log10(populations + 1)
        normalized = (log_pop - log_pop.min()) / (log_pop.max() - log_pop.min() + 0.001)
        sizes = min_size + normalized * (max_size - min_size)
        
        # Extract centroid coordinates
        x_coords = adopters_with_pop['centroid'].x
        y_coords = adopters_with_pop['centroid'].y
        
        # Plot circles
        scatter = ax.scatter(
            x_coords,
            y_coords,
            s=sizes,
            c='#1f77b4',
            alpha=0.7,
            edgecolors='#0d5a99',
            linewidths=1.5,
            zorder=5
        )
        
        # Create size legend
        # Show representative sizes
        pop_values = [1000, 10000, 50000, 100000]
        legend_sizes = []
        for pop in pop_values:
            if pop <= populations.max():
                log_val = np.log10(pop + 1)
                norm_val = (log_val - log_pop.min()) / (log_pop.max() - log_pop.min() + 0.001)
                size = min_size + norm_val * (max_size - min_size)
                legend_sizes.append(size)
        
        if legend_sizes:
            # Create size legend
            for i, (pop, size) in enumerate(zip([p for p in pop_values if p <= populations.max()], legend_sizes)):
                ax.scatter(
                    [], [],
                    s=size,
                    c='#1f77b4',
                    alpha=0.7,
                    edgecolors='#0d5a99',
                    linewidths=1.5,
                    label=f'{pop:,} students'
                )
            
            ax.legend(
                title='District Size',
                loc='lower right',
                fontsize=12,
                title_fontsize=14,
                frameon=True,
                fancybox=True,
                shadow=True
            )

# Styling
ax.set_title(
    'Electric School Bus Adoption: District Size by Student Population',
    fontsize=24,
    fontweight='bold',
    pad=20
)
ax.axis('off')

# Add statistics text
if len(adopters) > 0 and len(adopters_with_pop) > 0:
    total_students = adopters_with_pop['Student_Population'].sum()
    avg_students = adopters_with_pop['Student_Population'].mean()
    median_students = adopters_with_pop['Student_Population'].median()
    
    stats_text = f'Adopting Districts: {len(adopters):,}\n'
    stats_text += f'Total Students Served: {int(total_students):,}\n'
    stats_text += f'Avg. District Size: {int(avg_students):,} students\n'
    stats_text += f'Median District Size: {int(median_students):,} students'
    
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    )

# Add attribution
fig.text(
    0.99, 0.01,
    'Data: EPA Clean School Bus Program | Circle size represents student population | Basemap: CartoDB',
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
