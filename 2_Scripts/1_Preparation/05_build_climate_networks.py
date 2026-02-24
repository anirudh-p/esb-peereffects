"""
05_build_climate_networks.py
==============================
Build Climate-Based Neighbor Networks
Constructs spatial weights matrices based on climate similarity rather than 
geographic distance. Uses PRISM climate normals (30-year averages) for:
  - Precipitation (annual average)
  - Minimum temperature (annual average)

Strategy:
1. Extract climate values at district centroids from PRISM rasters
2. Compute climate distance matrix (Euclidean distance in standardized climate space)
3. Construct KNN-style climate neighbors (k most similar climates)
4. Save climate weights matrices for use in peer effect regressions

Inputs:
    - Cleaned/esb_full_analysis_dataset.csv (from script 04)
    - Raw/Spatial/EDGE_SCHOOLDISTRICT_TL21_SY2021/schooldistrict_sy2021_tl21.shp
    - Raw/Climate/PRISM_ppt_30yr_normal_4kmM4_annual_bil.bil
    - Raw/Climate/PRISM_tmin_30yr_normal_4kmM4_annual_bil.bil

Outputs:
    - Cleaned/district_climate_normals.csv
    - Weights/climate_weights/climate_knn_k4.pkl (and k=6,8,10,12,15,20)

Dependencies: 04_esb_analysis_prep.py

Author: [Your Name]
Last Updated: November 30, 2025
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from scipy.spatial.distance import cdist
from scipy.stats import zscore
import libpysal
from libpysal.weights import W
import os
import pickle
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, PRISM_PRECIP_FILE, PRISM_TMIN_FILE,
    CLIMATE_SIMILARITY_DATA, CLIMATE_WEIGHTS_DIR, CLIMATE_KNN_VALUES,
    ensure_dirs_exist, get_climate_weight_file
)

# Ensure output directories exist
ensure_dirs_exist()

print("="*80)
print("CLIMATE-BASED NEIGHBOR NETWORK CONSTRUCTION")
print("="*80)

# --- 1. LOAD DATA ---
print("\nLoading dataset...")
df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
print(f"  Sample size: {len(df)}")
print(f"  Charters: {df.get('is_charter', pd.Series([0])).sum()}")

# Load shapefile and merge with data
print("\nLoading shapefile...")
gdf_shape = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
gdf_shape['ncessch'] = gdf_shape['GEOID'].str.zfill(7)

# Merge spatial and attribute data
# Note: esb = IS_ADOPTER, z = IV_Z in dataset
gdf = gdf_shape[['ncessch', 'geometry']].merge(
    df[['nces_id', 'is_charter', 'IS_ADOPTER', 'IV_Z']], 
    left_on='ncessch', 
    right_on='nces_id', 
    how='inner'
)
gdf.rename(columns={'IS_ADOPTER': 'esb', 'IV_Z': 'z'}, inplace=True)

# Project to Albers Equal Area for accurate distances
gdf = gdf.to_crs('ESRI:102003')
print(f"  Spatial sample size: {len(gdf)}")

# Compute centroids for climate extraction
print("\nComputing district centroids...")
gdf['centroid'] = gdf.geometry.centroid
gdf_centroids = gdf.copy()
gdf_centroids['geometry'] = gdf_centroids['centroid']

# Reproject centroids to WGS84 (PRISM rasters are in lat/lon)
gdf_centroids_wgs84 = gdf_centroids.to_crs('EPSG:4326')

# --- 2. EXTRACT CLIMATE DATA ---
print("\n" + "="*80)
print("EXTRACTING CLIMATE VALUES FROM PRISM RASTERS")
print("="*80)

def extract_raster_values(gdf_points, raster_path, variable_name):
    """
    Extract raster values at point locations.
    
    Parameters:
    -----------
    gdf_points : GeoDataFrame
        Points in WGS84 (EPSG:4326) coordinate system
    raster_path : str
        Path to raster file
    variable_name : str
        Name for the extracted variable
    
    Returns:
    --------
    numpy array of extracted values
    """
    print(f"\nExtracting {variable_name}...")
    print(f"  Raster: {raster_path}")
    
    with rasterio.open(raster_path) as src:
        print(f"  Raster CRS: {src.crs}")
        print(f"  Raster bounds: {src.bounds}")
        print(f"  Raster shape: {src.shape}")
        
        # Extract coordinates
        coords = [(point.x, point.y) for point in gdf_points.geometry]
        
        # Sample raster at point locations
        values = [val[0] for val in src.sample(coords)]
        
        # Check for nodata values
        nodata = src.nodata
        values = np.array(values)
        
        if nodata is not None:
            n_nodata = np.sum(values == nodata)
            if n_nodata > 0:
                print(f"  WARNING: {n_nodata} points have nodata values")
                values[values == nodata] = np.nan
        
        n_valid = np.sum(~np.isnan(values))
        print(f"  Extracted values: {n_valid} valid, {len(values) - n_valid} missing")
        print(f"  Value range: {np.nanmin(values):.2f} to {np.nanmax(values):.2f}")
        print(f"  Mean: {np.nanmean(values):.2f}, Std: {np.nanstd(values):.2f}")
        
    return values

# Extract precipitation
precip_values = extract_raster_values(gdf_centroids_wgs84, str(PRISM_PRECIP_FILE), "Precipitation (mm/year)")
gdf['precip_mm'] = precip_values

# Extract minimum temperature
tmin_values = extract_raster_values(gdf_centroids_wgs84, str(PRISM_TMIN_FILE), "Min Temperature (°C)")
gdf['tmin_c'] = tmin_values

# --- 3. HANDLE MISSING VALUES ---
print("\n" + "="*80)
print("HANDLING MISSING CLIMATE DATA")
print("="*80)

print(f"\nMissing values:")
print(f"  Precipitation: {gdf['precip_mm'].isna().sum()} / {len(gdf)}")
print(f"  Min Temperature: {gdf['tmin_c'].isna().sum()} / {len(gdf)}")

# Strategy: Drop districts with missing climate data (likely Alaska/Hawaii/territories)
gdf_climate = gdf[gdf['precip_mm'].notna() & gdf['tmin_c'].notna()].copy()
print(f"\nAfter removing missing climate data: {len(gdf_climate)} districts")
print(f"  Charters: {gdf_climate['is_charter'].sum()}")

# Save climate data
print(f"\nSaving climate data to {CLIMATE_SIMILARITY_DATA}...")
climate_export = gdf_climate[['ncessch', 'precip_mm', 'tmin_c', 'is_charter']].copy()
climate_export.to_csv(str(CLIMATE_SIMILARITY_DATA), index=False)
print(f"  Saved {len(climate_export)} districts")

# --- 4. COMPUTE CLIMATE SIMILARITY ---
print("\n" + "="*80)
print("COMPUTING CLIMATE SIMILARITY MATRIX")
print("="*80)

# Standardize climate variables (z-scores)
print("\nStandardizing climate variables...")
climate_matrix = np.column_stack([
    zscore(gdf_climate['precip_mm'].values),
    zscore(gdf_climate['tmin_c'].values)
])

print(f"  Climate matrix shape: {climate_matrix.shape}")
print(f"  Precipitation z-scores: mean={climate_matrix[:,0].mean():.3f}, std={climate_matrix[:,0].std():.3f}")
print(f"  Temperature z-scores: mean={climate_matrix[:,1].mean():.3f}, std={climate_matrix[:,1].std():.3f}")

# Compute pairwise Euclidean distances in climate space
print("\nComputing pairwise climate distances...")
climate_distance_matrix = cdist(climate_matrix, climate_matrix, metric='euclidean')
print(f"  Distance matrix shape: {climate_distance_matrix.shape}")
print(f"  Mean climate distance: {climate_distance_matrix[np.triu_indices_from(climate_distance_matrix, k=1)].mean():.3f}")

# --- 5. BUILD CLIMATE-BASED KNN WEIGHTS ---
print("\n" + "="*80)
print("CONSTRUCTING CLIMATE-BASED KNN WEIGHTS")
print("="*80)

# Directory already created by ensure_dirs_exist()

# Store weights for each k value
climate_weights = {}

for k in CLIMATE_KNN_VALUES:
    print(f"\n--- K = {k} ---")
    
    # For each district, find k nearest climate neighbors
    neighbors = {}
    weights_dict = {}
    
    for i in range(len(gdf_climate)):
        # Get distances to all other districts
        distances = climate_distance_matrix[i, :]
        
        # Sort by distance (excluding self at distance 0)
        sorted_indices = np.argsort(distances)
        
        # Take k nearest neighbors (excluding self)
        neighbor_indices = sorted_indices[1:k+1]  # Skip index 0 (self)
        neighbor_distances = distances[neighbor_indices]
        
        # Store neighbors and inverse-distance weights
        neighbors[i] = neighbor_indices.tolist()
        weights_dict[i] = (1.0 / (neighbor_distances + 1e-6)).tolist()  # Add small constant to avoid division by zero
    
    # Create libpysal W object
    w_climate = W(neighbors, weights_dict)
    
    # Row-standardize weights
    w_climate.transform = 'r'
    
    # Save weights object
    weights_file = get_climate_weight_file(k)
    with open(str(weights_file), 'wb') as f:
        pickle.dump(w_climate, f)
    
    print(f"  Created weights matrix: {w_climate.n} districts, {w_climate.n * k} connections")
    print(f"  Mean neighbors: {w_climate.mean_neighbors:.2f}")
    print(f"  Saved to: {weights_file}")
    
    # Store in dictionary
    climate_weights[k] = w_climate

# --- 6. SUMMARY STATISTICS ---
print("\n" + "="*80)
print("CLIMATE NEIGHBOR SUMMARY")
print("="*80)

print(f"\nClimate data extracted for {len(gdf_climate)} districts")
print(f"  Precipitation range: {gdf_climate['precip_mm'].min():.0f} - {gdf_climate['precip_mm'].max():.0f} mm/year")
print(f"  Min temp range: {gdf_climate['tmin_c'].min():.1f} - {gdf_climate['tmin_c'].max():.1f} °C")

print(f"\nCreated {len(CLIMATE_KNN_VALUES)} climate-based KNN weights matrices:")
for k in CLIMATE_KNN_VALUES:
    weight_file = get_climate_weight_file(k)
    print(f"  K={k}: {weight_file}")

print("\n" + "="*80)
print("CLIMATE NETWORK CONSTRUCTION COMPLETE")
print("="*80)
