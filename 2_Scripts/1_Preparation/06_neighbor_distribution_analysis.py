#!/usr/bin/env python3
"""
06_neighbor_distribution_analysis.py
======================================
Analyzes spatial neighbor patterns for ESB robustness study.

Part 1: Distance bands - distribution of neighbor counts by radius
Part 2: KNN - distribution of distances by k

Inputs:
    - Cleaned/esb_full_analysis_dataset.csv (from script 04)
    - Raw/Spatial/EDGE_SCHOOLDISTRICT_TL21_SY2021/schooldistrict_sy2021_tl21.shp

Outputs:
    - Tables/neighbor_distribution_results.txt

Dependencies: 04_esb_analysis_prep.py

Note: This is a supplementary analysis script, not required for main pipeline.

Author: [Your Name]
Last Updated: November 30, 2025
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from libpysal.weights import KNN, DistanceBand
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, TABLES_DIR, ensure_dirs_exist

# Ensure output directories exist
ensure_dirs_exist()

# Output path
OUTPUT_FILE = TABLES_DIR / 'neighbor_distribution_results.txt'

# Load the rebuilt dataset (now includes charters)
print("Loading dataset...")
df = pd.read_csv(str(ESB_FULL_ANALYSIS))

print(f"Sample size: {len(df)}")
if 'is_charter' in df.columns:
    print(f"Charters: {df['is_charter'].sum()}")
else:
    print("Warning: No charter flag in dataset")
    df['is_charter'] = 0

# Load shapefile and merge
print("Loading shapefile...")
gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
shp_id_col = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]

gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.lstrip('0').astype(int)
df['nces_id'] = df['nces_id'].astype(int)

gdf_reg = gdf[[shp_id_col, 'geometry']].merge(df, left_on=shp_id_col, right_on='nces_id', how='inner')
gdf_reg = gpd.GeoDataFrame(gdf_reg, geometry='geometry')
gdf_reg = gdf_reg.to_crs(epsg=5070)  # Albers Equal Area
gdf_reg['centroid'] = gdf_reg.geometry.centroid

print(f"Spatial sample size: {len(gdf_reg)}")
print(f"  Charters: {gdf_reg['is_charter'].sum()}")
print(f"  Traditional: {(gdf_reg['is_charter']==0).sum()}")

# ============================================================================
# PART 1: DISTANCE-BASED NEIGHBOR ANALYSIS
# ============================================================================

with open(str(OUTPUT_FILE), 'w') as f:
    f.write("="*80 + "\n")
    f.write("NEIGHBOR DISTRIBUTION ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Sample Size: {len(gdf_reg)} districts\n")
    f.write(f"Charter Schools: {gdf_reg['is_charter'].sum()} ({100*gdf_reg['is_charter'].sum()/len(gdf_reg):.2f}%)\n")
    f.write(f"Traditional Districts: {(gdf_reg['is_charter']==0).sum()}\n\n")
    
    # --- DISTANCE BANDS ---
    f.write("\n" + "="*80 + "\n")
    f.write("PART 1: DISTANCE BANDS - Number of Neighbors by Radius\n")
    f.write("="*80 + "\n\n")
    
    distance_km = [25, 50, 75, 100, 150, 200]
    
    for d in distance_km:
        print(f"  Analyzing {d}km radius...")
        f.write(f"\n{'='*70}\n")
        f.write(f"Radius: {d} km\n")
        f.write(f"{'='*70}\n\n")
        
        threshold_m = d * 1000
        
        # Create distance band
        w_dist = DistanceBand.from_dataframe(gdf_reg, threshold=threshold_m, geom_col='centroid', binary=False)
        
        # Analyze neighbor counts
        neighbor_counts_dict = w_dist.cardinalities
        neighbor_counts = np.array([neighbor_counts_dict[i] for i in range(len(gdf_reg))])
        
        # Overall statistics
        f.write("Overall Neighbor Statistics:\n")
        f.write(f"  Mean neighbors: {neighbor_counts.mean():.2f}\n")
        f.write(f"  Median neighbors: {np.median(neighbor_counts):.2f}\n")
        f.write(f"  Std Dev: {neighbor_counts.std():.2f}\n")
        f.write(f"  Min neighbors: {neighbor_counts.min()}\n")
        f.write(f"  Max neighbors: {neighbor_counts.max()}\n")
        f.write(f"  Islands (0 neighbors): {(neighbor_counts == 0).sum()} ({100*(neighbor_counts==0).sum()/len(neighbor_counts):.1f}%)\n")
        f.write(f"  Districts with neighbors: {(neighbor_counts > 0).sum()}\n")
        
        # Distribution by quantiles
        f.write("\nNeighbor Count Distribution:\n")
        quantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        for q in quantiles:
            val = np.quantile(neighbor_counts, q)
            f.write(f"  {int(q*100):3d}th percentile: {val:.1f} neighbors\n")
        
        # Breakdown by charter status
        gdf_reg['neighbor_count'] = neighbor_counts
        
        f.write("\nBy District Type:\n")
        for is_charter, label in [(0, 'Traditional Public'), (1, 'Charter')]:
            subset = gdf_reg[gdf_reg['is_charter'] == is_charter]['neighbor_count']
            if len(subset) > 0:
                f.write(f"  {label} (n={len(subset)}):\n")
                f.write(f"    Mean: {subset.mean():.2f}\n")
                f.write(f"    Median: {subset.median():.2f}\n")
                f.write(f"    Islands: {(subset == 0).sum()} ({100*(subset==0).sum()/len(subset):.1f}%)\n")
        
        # Analyze NEIGHBOR composition (charter vs traditional)
        f.write("\nNeighbor Composition Analysis:\n")
        
        total_charter_neighbors = 0
        total_traditional_neighbors = 0
        districts_with_neighbors = 0
        
        charter_has_charter_neighbor = 0
        traditional_has_charter_neighbor = 0
        
        for idx in range(len(gdf_reg)):
            neighbors_idx = w_dist.neighbors[idx]
            if len(neighbors_idx) > 0:
                districts_with_neighbors += 1
                neighbor_charters = gdf_reg.iloc[neighbors_idx]['is_charter'].sum()
                neighbor_traditional = len(neighbors_idx) - neighbor_charters
                total_charter_neighbors += neighbor_charters
                total_traditional_neighbors += neighbor_traditional
                
                # Track if charters/traditional have charter neighbors
                if gdf_reg.iloc[idx]['is_charter'] == 1 and neighbor_charters > 0:
                    charter_has_charter_neighbor += 1
                if gdf_reg.iloc[idx]['is_charter'] == 0 and neighbor_charters > 0:
                    traditional_has_charter_neighbor += 1
        
        total_neighbors = total_charter_neighbors + total_traditional_neighbors
        f.write(f"  Total neighbor connections: {total_neighbors}\n")
        if total_neighbors > 0:
            f.write(f"  Charter neighbors: {total_charter_neighbors} ({100*total_charter_neighbors/total_neighbors:.2f}%)\n")
            f.write(f"  Traditional neighbors: {total_traditional_neighbors} ({100*total_traditional_neighbors/total_neighbors:.2f}%)\n")
        
        n_charters_in_sample = gdf_reg['is_charter'].sum()
        n_traditional = len(gdf_reg) - n_charters_in_sample
        
        if n_charters_in_sample > 0:
            f.write(f"\n  Charter districts with >=1 charter neighbor: {charter_has_charter_neighbor}/{n_charters_in_sample} ({100*charter_has_charter_neighbor/n_charters_in_sample:.1f}%)\n")
        if n_traditional > 0:
            f.write(f"  Traditional districts with >=1 charter neighbor: {traditional_has_charter_neighbor}/{n_traditional} ({100*traditional_has_charter_neighbor/n_traditional:.1f}%)\n")
    
    # --- KNN ANALYSIS ---
    f.write("\n\n" + "="*80 + "\n")
    f.write("PART 2: KNN - Distance Distribution by K\n")
    f.write("="*80 + "\n\n")
    
    knn_values = [4, 6, 8, 10, 12, 15, 20]
    
    for k in knn_values:
        print(f"  Analyzing KNN k={k}...")
        f.write(f"\n{'='*70}\n")
        f.write(f"K = {k} nearest neighbors\n")
        f.write(f"{'='*70}\n\n")
        
        # Create KNN weights
        w_knn = KNN.from_dataframe(gdf_reg, k=k, geom_col='centroid')
        
        # Calculate distances to neighbors
        all_distances = []
        max_distances = []
        charter_neighbor_counts = []
        
        for idx in range(len(gdf_reg)):
            # Get neighbors for this district
            neighbors_idx = w_knn.neighbors[idx]
            
            # Calculate distances
            origin_point = gdf_reg.iloc[idx]['centroid']
            distances_this_district = []
            for neighbor_idx in neighbors_idx:
                neighbor_point = gdf_reg.iloc[neighbor_idx]['centroid']
                distance_m = origin_point.distance(neighbor_point)
                distance_km = distance_m / 1000
                all_distances.append(distance_km)
                distances_this_district.append(distance_km)
            
            # Track max distance for this district
            if len(distances_this_district) > 0:
                max_distances.append(max(distances_this_district))
                
                # Count charter neighbors
                n_charter_neighbors = gdf_reg.iloc[neighbors_idx]['is_charter'].sum()
                charter_neighbor_counts.append(n_charter_neighbors)
        
        # Overall distance statistics
        f.write("Distance Statistics (km):\n")
        f.write(f"  Mean distance to neighbor: {np.mean(all_distances):.2f} km\n")
        f.write(f"  Median distance: {np.median(all_distances):.2f} km\n")
        f.write(f"  Std Dev: {np.std(all_distances):.2f} km\n")
        f.write(f"  Min distance: {np.min(all_distances):.2f} km\n")
        f.write(f"  Max distance: {np.max(all_distances):.2f} km\n")
        
        f.write("\nMaximum Distance to Kth Neighbor (per district):\n")
        f.write(f"  Mean max distance: {np.mean(max_distances):.2f} km\n")
        f.write(f"  Median max distance: {np.median(max_distances):.2f} km\n")
        f.write(f"  90th percentile: {np.quantile(max_distances, 0.9):.2f} km\n")
        f.write(f"  95th percentile: {np.quantile(max_distances, 0.95):.2f} km\n")
        f.write(f"  99th percentile: {np.quantile(max_distances, 0.99):.2f} km\n")
        
        # Distance distribution
        f.write("\nDistance Distribution (all neighbor pairs):\n")
        quantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        for q in quantiles:
            val = np.quantile(all_distances, q)
            f.write(f"  {int(q*100):3d}th percentile: {val:.2f} km\n")
        
        # Neighbor composition
        f.write("\nNeighbor Composition:\n")
        total_neighbor_pairs = len(all_distances)
        total_charter_neighbors = sum(charter_neighbor_counts)
        f.write(f"  Total neighbor pairs: {total_neighbor_pairs}\n")
        f.write(f"  Total charter neighbor connections: {total_charter_neighbors} ({100*total_charter_neighbors/total_neighbor_pairs:.2f}%)\n")
        f.write(f"  Districts with >=1 charter neighbor: {sum(1 for c in charter_neighbor_counts if c > 0)}/{len(charter_neighbor_counts)} ({100*sum(1 for c in charter_neighbor_counts if c > 0)/len(charter_neighbor_counts):.1f}%)\n")
        f.write(f"  Mean charter neighbors per district: {np.mean(charter_neighbor_counts):.2f}\n")
        f.write(f"  Median charter neighbors: {np.median(charter_neighbor_counts):.0f}\n")
        f.write(f"  Max charter neighbors for one district: {np.max(charter_neighbor_counts):.0f}\n")

print(f"\nAnalysis complete! Results saved to {OUTPUT_FILE}")
