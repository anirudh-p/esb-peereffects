"""
12_orthogonalized_climate_measures.py
=====================================
Creates climate measures that are orthogonal to geography to enable
clean decomposition of geographic vs climate peer effects.

PROBLEM:
--------
Climate (precipitation, temperature) is highly correlated with geography:
- Districts in same state have similar climate
- State FE absorbs most climate variation
- Climate-based "peer effects" may just reflect regional confounding

SOLUTION:
---------
Create orthogonalized climate measures by:
1. Residualizing climate on geographic predictors (state, lat, lon)
2. Using within-state climate deviations
3. Creating operationally-relevant indices (HDD, CDD, elevation)
4. Building peer networks based on residual climate similarity

OUTPUT:
-------
- District-level orthogonalized climate measures
- Climate peer weight matrices (based on residual similarity)
- Diagnostic plots showing orthogonalization quality
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, CLEANED_DIR,
    CLIMATE_WEIGHTS_DIR, FIGURES_DIR, TABLES_DIR,
    ensure_dirs_exist
)

import statsmodels.api as sm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

ensure_dirs_exist()

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("=" * 80)
print("ORTHOGONALIZED CLIMATE MEASURES FOR DECOMPOSITION")
print("=" * 80)

# Load main dataset
df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
print(f"\nLoaded {len(df):,} districts")

# Load climate data
climate_path = CLEANED_DIR / "district_climate_normals.csv"
if climate_path.exists():
    climate_df = pd.read_csv(climate_path)
    # Rename columns to standard names
    climate_df = climate_df.rename(columns={
        'ncessch': 'nces_id',
        'precip_mm': 'precip',
        'tmin_c': 'tmin'
    })
    climate_df['nces_id'] = climate_df['nces_id'].astype(str).str.zfill(7)
    print(f"Loaded climate data: {len(climate_df)} districts")
    
    # Merge
    df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
    df = df.merge(climate_df[['nces_id', 'precip', 'tmin']], on='nces_id', how='left')
    print(f"After merge: {df['precip'].notna().sum()} with climate data")
else:
    print(f"WARNING: Climate file not found at {climate_path}")
    # Create placeholders
    df['precip'] = np.nan
    df['tmin'] = np.nan

# Get district centroids for geographic controls
print("\nLoading shapefile for centroid coordinates...")
try:
    gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
    shp_id_col = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]
    gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.zfill(7)
    
    # Get centroids in lat/lon
    gdf_wgs = gdf.to_crs(epsg=4326)
    gdf_wgs['centroid_lat'] = gdf_wgs.geometry.centroid.y
    gdf_wgs['centroid_lon'] = gdf_wgs.geometry.centroid.x
    
    coords = gdf_wgs[[shp_id_col, 'centroid_lat', 'centroid_lon']].rename(
        columns={shp_id_col: 'nces_id'}
    )
    df = df.merge(coords, on='nces_id', how='left')
    print(f"  Added coordinates: {df['centroid_lat'].notna().sum()} districts")
except Exception as e:
    print(f"  WARNING: Could not load shapefile: {e}")
    df['centroid_lat'] = np.nan
    df['centroid_lon'] = np.nan

# ==============================================================================
# MEASURE 1: STATE-RESIDUALIZED CLIMATE
# ==============================================================================

print("\n" + "=" * 80)
print("MEASURE 1: STATE-RESIDUALIZED CLIMATE")
print("=" * 80)
print("Climate_resid = Climate - E[Climate | State]")

for var in ['precip', 'tmin']:
    if var not in df.columns or df[var].isna().all():
        print(f"  {var}: No data available")
        continue
    
    # Within-state deviation
    state_means = df.groupby('state')[var].transform('mean')
    df[f'{var}_state_resid'] = df[var] - state_means
    
    # Report variation explained
    total_var = df[var].var()
    resid_var = df[f'{var}_state_resid'].var()
    r2 = 1 - resid_var / total_var if total_var > 0 else 0
    
    print(f"\n{var}:")
    print(f"  Total variance: {total_var:.2f}")
    print(f"  Within-state variance: {resid_var:.2f}")
    print(f"  R² (state explains): {r2:.3f}")

# ==============================================================================
# MEASURE 2: GEO-RESIDUALIZED CLIMATE (State + Lat + Lon)
# ==============================================================================

print("\n" + "=" * 80)
print("MEASURE 2: GEOGRAPHICALLY-RESIDUALIZED CLIMATE")
print("=" * 80)
print("Climate_resid = Climate - E[Climate | State, Lat, Lon]")

for var in ['precip', 'tmin']:
    if var not in df.columns or df[var].isna().all():
        continue
    
    # Subset with all required data
    subset = df[['nces_id', var, 'state', 'centroid_lat', 'centroid_lon']].dropna().copy()
    
    if len(subset) < 100:
        print(f"  {var}: Insufficient data for geo-residualization")
        continue
    
    # Ensure numeric types
    subset[var] = subset[var].astype(float)
    subset['centroid_lat'] = subset['centroid_lat'].astype(float)
    subset['centroid_lon'] = subset['centroid_lon'].astype(float)
    
    # Add state dummies
    state_dummies = pd.get_dummies(subset['state'], prefix='st', drop_first=True).astype(float)
    
    # Regression: climate ~ state + lat + lon + lat² + lon²
    geo_vars = pd.DataFrame({
        'centroid_lat': subset['centroid_lat'].values,
        'centroid_lon': subset['centroid_lon'].values,
        'lat_sq': (subset['centroid_lat']**2).values,
        'lon_sq': (subset['centroid_lon']**2).values,
        'lat_lon': (subset['centroid_lat'] * subset['centroid_lon']).values
    }, index=subset.index)
    
    X = pd.concat([state_dummies.reset_index(drop=True), geo_vars.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = subset[var].reset_index(drop=True)
    
    model = sm.OLS(y, X).fit()
    
    # Residuals
    subset = subset.reset_index(drop=True)
    subset[f'{var}_geo_resid'] = model.resid
    
    # Merge back
    df = df.merge(
        subset[['nces_id', f'{var}_geo_resid']],
        on='nces_id',
        how='left'
    )
    
    print(f"\n{var}:")
    print(f"  Original variance: {y.var():.2f}")
    print(f"  Residual variance: {model.resid.var():.2f}")
    print(f"  R² (geo explains): {model.rsquared:.3f}")

# ==============================================================================
# MEASURE 3: HEATING/COOLING DEGREE DAYS (if available)
# ==============================================================================

print("\n" + "=" * 80)
print("MEASURE 3: OPERATIONAL CLIMATE INDICES")
print("=" * 80)

# Approximate HDD/CDD from annual tmin
# HDD ≈ max(0, 18 - tmin) * 365 (simplified)
# CDD ≈ max(0, tmin - 18) * 365 (simplified)
# These are rough approximations; ideally use NOAA climate normals

if 'tmin' in df.columns and df['tmin'].notna().any():
    # Base temperature for ESB operations (18°C ≈ 65°F)
    base_temp = 18
    
    # Very rough HDD proxy (using annual min temp as indicator)
    # True HDD would sum (base - daily_temp) for all days where daily_temp < base
    df['hdd_proxy'] = np.maximum(0, base_temp - df['tmin']) * 180  # ~half year below base
    df['cdd_proxy'] = np.maximum(0, df['tmin'] - base_temp) * 90   # ~quarter year above base
    
    print("\nCreated proxy HDD/CDD from minimum temperature:")
    print(f"  HDD proxy: mean={df['hdd_proxy'].mean():.0f}, range=[{df['hdd_proxy'].min():.0f}, {df['hdd_proxy'].max():.0f}]")
    print(f"  CDD proxy: mean={df['cdd_proxy'].mean():.0f}, range=[{df['cdd_proxy'].min():.0f}, {df['cdd_proxy'].max():.0f}]")
    
    # Also create state-residualized versions
    for var in ['hdd_proxy', 'cdd_proxy']:
        state_means = df.groupby('state')[var].transform('mean')
        df[f'{var}_state_resid'] = df[var] - state_means
    
    print("\n  Created state-residualized HDD/CDD proxies")
else:
    print("  No temperature data available for HDD/CDD calculation")

# ==============================================================================
# MEASURE 4: CLIMATE ANOMALY INDEX
# ==============================================================================

print("\n" + "=" * 80)
print("MEASURE 4: CLIMATE ANOMALY INDEX")
print("=" * 80)
print("How unusual is this district's climate within its state?")

if 'precip_state_resid' in df.columns and 'tmin_state_resid' in df.columns:
    # Standardize within-state residuals
    for var in ['precip_state_resid', 'tmin_state_resid']:
        state_sds = df.groupby('state')[var].transform('std')
        df[f'{var}_z'] = df[var] / state_sds.replace(0, np.nan)
    
    # Climate anomaly = sqrt(z_precip² + z_tmin²)
    df['climate_anomaly'] = np.sqrt(
        df['precip_state_resid_z']**2 + 
        df['tmin_state_resid_z']**2
    )
    
    print(f"\nClimate anomaly distribution:")
    print(f"  Mean: {df['climate_anomaly'].mean():.2f}")
    print(f"  Median: {df['climate_anomaly'].median():.2f}")
    print(f"  95th percentile: {df['climate_anomaly'].quantile(0.95):.2f}")
    
    # Identify climate outliers within states
    outliers = df[df['climate_anomaly'] > df['climate_anomaly'].quantile(0.95)]
    print(f"\n  Climate outliers (top 5%): {len(outliers)} districts")
    print(f"  These have unusual climate for their state (mountains, coasts, etc.)")
else:
    print("  Insufficient data for climate anomaly calculation")

# ==============================================================================
# BUILD ORTHOGONALIZED CLIMATE PEER NETWORKS
# ==============================================================================

print("\n" + "=" * 80)
print("BUILDING ORTHOGONALIZED CLIMATE PEER NETWORKS")
print("=" * 80)

def build_knn_weights(df, climate_vars, k=6, output_name=""):
    """Build KNN weights based on climate similarity."""
    
    # Get complete cases
    subset = df[['nces_id'] + climate_vars].dropna()
    print(f"\n  {output_name}: {len(subset)} districts with complete data")
    
    if len(subset) < k + 1:
        print(f"    Too few observations for K={k}")
        return None
    
    # Standardize
    X = subset[climate_vars].values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Compute pairwise distances
    distances = cdist(X_std, X_std, metric='euclidean')
    
    # For each district, find K nearest neighbors
    neighbors = {}
    weights = {}
    
    for i in range(len(subset)):
        # Sort by distance
        sorted_idx = np.argsort(distances[i])
        # Skip self (index 0), take next K
        knn_idx = sorted_idx[1:k+1]
        knn_dist = distances[i][knn_idx]
        
        # Convert to inverse-distance weights
        knn_weights = 1 / (knn_dist + 0.001)
        knn_weights = knn_weights / knn_weights.sum()  # Normalize
        
        district_id = subset.iloc[i]['nces_id']
        neighbor_ids = subset.iloc[knn_idx]['nces_id'].tolist()
        
        neighbors[district_id] = neighbor_ids
        weights[district_id] = knn_weights.tolist()
    
    return neighbors, weights, subset['nces_id'].tolist()

# Original climate peers
if 'precip' in df.columns and 'tmin' in df.columns:
    result = build_knn_weights(df, ['precip', 'tmin'], k=6, output_name="Original climate (precip + tmin)")

# Orthogonalized climate peers (state-residualized)
if 'precip_state_resid' in df.columns and 'tmin_state_resid' in df.columns:
    result_ortho = build_knn_weights(
        df, 
        ['precip_state_resid', 'tmin_state_resid'], 
        k=6, 
        output_name="Orthogonalized climate (state-residualized)"
    )

# Geo-orthogonalized climate peers
if 'precip_geo_resid' in df.columns and 'tmin_geo_resid' in df.columns:
    result_geo_ortho = build_knn_weights(
        df, 
        ['precip_geo_resid', 'tmin_geo_resid'], 
        k=6, 
        output_name="Geo-orthogonalized climate"
    )

# ==============================================================================
# CORRELATION DIAGNOSTICS
# ==============================================================================

print("\n" + "=" * 80)
print("CORRELATION DIAGNOSTICS")
print("=" * 80)
print("How much do orthogonalized measures reduce geographic correlation?")

# Create geographic region indicators
if 'state' in df.columns:
    # Simple region mapping
    northeast = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA']
    midwest = ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD']
    south = ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'DC', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX']
    west = ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']
    
    def assign_region(state):
        if state in northeast:
            return 'Northeast'
        elif state in midwest:
            return 'Midwest'
        elif state in south:
            return 'South'
        elif state in west:
            return 'West'
        else:
            return 'Other'
    
    df['region'] = df['state'].apply(assign_region)
    
    # ANOVA: How much variance explained by region?
    print("\nVariance explained by region (R²):")
    
    climate_vars = ['precip', 'tmin', 'precip_state_resid', 'tmin_state_resid', 
                    'precip_geo_resid', 'tmin_geo_resid']
    
    for var in climate_vars:
        if var not in df.columns or df[var].isna().all():
            continue
        
        subset = df[[var, 'region']].dropna()
        if len(subset) < 100:
            continue
        
        # R² from regression on region dummies
        region_dummies = pd.get_dummies(subset['region'], drop_first=True)
        X = sm.add_constant(region_dummies)
        y = subset[var]
        
        model = sm.OLS(y, X).fit()
        
        print(f"  {var:25s}: R² = {model.rsquared:.3f}")

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================

print("\n" + "=" * 80)
print("SAVING OUTPUTS")
print("=" * 80)

# Save orthogonalized climate measures
output_cols = ['nces_id', 'state']
for col in df.columns:
    if 'resid' in col or 'anomaly' in col or 'proxy' in col:
        output_cols.append(col)

output_df = df[output_cols].copy()
output_path = CLEANED_DIR / "orthogonalized_climate_measures.csv"
output_df.to_csv(output_path, index=False)
print(f"\nSaved orthogonalized measures to: {output_path}")
print(f"  Columns: {output_cols}")

# ==============================================================================
# SUMMARY & RECOMMENDATIONS
# ==============================================================================

print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS FOR DECOMPOSITION")
print("=" * 80)

print("""
ORTHOGONALIZED CLIMATE MEASURES CREATED:
----------------------------------------
1. precip_state_resid, tmin_state_resid: 
   Climate minus state mean. Captures within-state variation only.
   
2. precip_geo_resid, tmin_geo_resid:
   Climate minus E[climate | state, lat, lon, lat², lon²].
   Removes systematic geographic patterns.

3. hdd_proxy_state_resid, cdd_proxy_state_resid:
   Operationally-relevant heating/cooling load, state-residualized.

4. climate_anomaly:
   How unusual is this district's climate within its state?
   Useful for identifying microclimates (mountains, coasts).

RECOMMENDED DECOMPOSITION APPROACH:
-----------------------------------
1. BASELINE: Geographic KNN-6 + State FE
   - This is your main specification
   - Geographic peer effect: β_geo ≈ 0.06

2. ADD ORTHOGONALIZED CLIMATE:
   - Build climate peers from RESIDUALIZED climate (not raw)
   - Include both w_geo_adoption and w_clim_ortho_adoption
   - If w_clim_ortho is significant: climate matters beyond geography
   - If w_clim_ortho is NOT significant: climate was just proxying for region

3. INTERPRETATION:
   - If ONLY geographic matters: Peer effects are about proximity/information
   - If BOTH matter: Distinct mechanisms (social + technical similarity)
   - If orthogonalized climate is NEGATIVE: Climate similarity → competition?

CAVEAT:
-------
With state FE absorbing ~60-80% of climate variance, orthogonalized climate
variation is quite limited. May lack power to detect true climate effects.
Consider this a "ruling out" test rather than a confirmation test.
""")

print("\nDone.")
