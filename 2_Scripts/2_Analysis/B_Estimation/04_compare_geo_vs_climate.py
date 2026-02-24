"""
Compare Geographic vs Climate-Based Peer Effects
=================================================
Tests robustness of ESB adoption peer effects by comparing:
  1. Geographic neighbors (spatial proximity, KNN)
  2. Climate neighbors (climate similarity, KNN)

Research Question: Do peer effects persist when we define neighbors 
by climate similarity rather than geographic proximity?

Hypothesis: If peer effects are driven by information spillovers and 
common environmental challenges, climate neighbors might show similar 
or stronger peer effects than geographic neighbors.

Alternative: If peer effects are driven by social networks and local 
interactions, geographic neighbors should dominate.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import libpysal
from libpysal.weights import KNN
from linearmodels.iv import IV2SLS
import pickle
import os
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, CLIMATE_WEIGHTS_DIR,
    CLIMATE_SIMILARITY_DATA, CLIMATE_KNN_VALUES, TABLES_DIR,
    get_climate_weight_file, ensure_dirs_exist
)

# Ensure output directories exist
ensure_dirs_exist()

print("="*80)
print("GEOGRAPHIC VS CLIMATE PEER EFFECTS COMPARISON")
print("="*80)

# --- CONFIGURATION ---
OUTPUT_FILE = TABLES_DIR / 'geo_vs_climate_peer_effects.csv'

# KNN values to compare
K_VALUES = CLIMATE_KNN_VALUES

# Control variables
CONTROLS = [
    'log_median_income', 'log_enrollment', 'poverty_rate',
    'pct_hispanic', 'pct_white', 'pct_black', 'pm25'
]

# --- 1. LOAD DATA ---
print("\nLoading dataset...")
df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})

# Create log transformations
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Rename variables for consistency
df['esb'] = df['IS_ADOPTER']
df['z'] = df['IV_Z']

print(f"  Sample size: {len(df)}")
print(f"  Adopters: {df['esb'].sum()} ({100*df['esb'].mean():.1f}%)")

# Load shapefile
print("\nLoading shapefile...")
gdf_shape = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
gdf_shape['ncessch'] = gdf_shape['GEOID'].str.zfill(7)

# Merge spatial and attribute data (include state for clustering)
gdf = gdf_shape[['ncessch', 'geometry']].merge(
    df[['nces_id', 'state'] + ['esb', 'z'] + CONTROLS],
    left_on='ncessch',
    right_on='nces_id',
    how='inner'
)

# Project to Albers Equal Area
gdf = gdf.to_crs('ESRI:102003')
print(f"  Spatial sample size: {len(gdf)}")

# Drop missing values
print("\nHandling missing values...")
df_clean = gdf.dropna(subset=['esb', 'z', 'state'] + CONTROLS)
print(f"  Clean sample: {len(df_clean)}")

# --- 2. HELPER FUNCTION FOR 2SLS ---
def run_2sls_spec(df_data, w_geo, w_clim, spec_name, k_val):
    """
    Run 2SLS with BOTH geographic and climate spatial lags as endogenous.
    
    Two endogenous variables: w_geo_adoption, w_clim_adoption
    Two instruments: w_geo_z, w_clim_z
    """
    # Create spatial lags
    w_geo_adoption = libpysal.weights.lag_spatial(w_geo, df_data['esb'].values)
    w_clim_adoption = libpysal.weights.lag_spatial(w_clim, df_data['esb'].values)
    w_geo_z = libpysal.weights.lag_spatial(w_geo, df_data['z'].values)
    w_clim_z = libpysal.weights.lag_spatial(w_clim, df_data['z'].values)
    
    # Prepare data
    df_reg = df_data.copy()
    df_reg['w_geo_adoption'] = w_geo_adoption
    df_reg['w_clim_adoption'] = w_clim_adoption
    df_reg['w_geo_z'] = w_geo_z
    df_reg['w_clim_z'] = w_clim_z
    
    # Drop any remaining NaN
    df_reg = df_reg.dropna(subset=['w_geo_adoption', 'w_clim_adoption', 'w_geo_z', 'w_clim_z'])
    
    # Named DataFrames for reliable coefficient extraction
    dependent = df_reg['esb']
    endog = df_reg[['w_geo_adoption', 'w_clim_adoption']]
    instruments = df_reg[['w_geo_z', 'w_clim_z']]
    import statsmodels.api as sm_api
    exog = sm_api.add_constant(df_reg[CONTROLS])

    # Run 2SLS with state-clustered standard errors
    model = IV2SLS(dependent, exog, endog, instruments)
    results = model.fit(cov_type='clustered', clusters=df_reg['state'])

    # Extract coefficients by name (not fragile positional indexing)
    coef_geo = results.params['w_geo_adoption']
    se_geo = results.std_errors['w_geo_adoption']
    pval_geo = results.pvalues['w_geo_adoption']

    coef_clim = results.params['w_clim_adoption']
    se_clim = results.std_errors['w_clim_adoption']
    pval_clim = results.pvalues['w_clim_adoption']

    # First stage F-statistics
    try:
        fstat_geo = results.first_stage.diagnostics['f.stat']['w_geo_adoption']
        fstat_clim = results.first_stage.diagnostics['f.stat']['w_clim_adoption']
    except Exception:
        try:
            fstat_geo = results.first_stage.diagnostics.iloc[0]['f.stat']
            fstat_clim = results.first_stage.diagnostics.iloc[1]['f.stat']
        except Exception:
            fstat_geo = np.nan
            fstat_clim = np.nan
    
    return {
        'specification': spec_name,
        'k': k_val,
        'n': len(df_reg),
        'coef_geo': coef_geo,
        'se_geo': se_geo,
        'pval_geo': pval_geo,
        'coef_clim': coef_clim,
        'se_clim': se_clim,
        'pval_clim': pval_clim,
        'fstat_geo': fstat_geo,
        'fstat_clim': fstat_clim,
        'rsquared': results.rsquared,
        'rsquared_adj': results.rsquared_adj
    }

# --- 3. RUN COMPARISONS ---
print("\n" + "="*80)
print("RUNNING PEER EFFECT COMPARISONS")
print("="*80)

results_list = []

for k in K_VALUES:
    print(f"\n{'='*60}")
    print(f"K = {k} NEAREST NEIGHBORS")
    print(f"{'='*60}")
    
    # --- Geographic KNN ---
    print(f"\nConstructing geographic KNN (k={k})...")
    w_geo = KNN.from_dataframe(df_clean, k=k)
    w_geo.transform = 'r'
    print(f"  Geographic neighbors: {w_geo.n} districts, {w_geo.n * k} connections")
    
    # --- Load Climate KNN ---
    climate_weights_file = get_climate_weight_file(k)
    print(f"\nLoading climate KNN (k={k})...")
    with open(climate_weights_file, 'rb') as f:
        w_clim_full = pickle.load(f)
    
    print(f"  Climate neighbors (full): {w_clim_full.n} districts")
    
    # --- Subset climate weights to match df_clean ---
    # OPTIMIZED: Use vectorized mapping instead of nested loops
    print(f"\nAligning climate weights with clean sample...")
    
    # Get ncessch from df_clean
    ncessch_clean = df_clean['ncessch'].values
    
    # Load climate data to get mapping
    df_clim = pd.read_csv(str(Path(__file__).parent.parent.parent.parent / '1_Data' / 'Cleaned' / 'district_climate_normals.csv'), dtype={'ncessch': str})
    
    # Create FAST lookup: ncessch -> climate index
    ncessch_to_clim_idx = {ncessch: idx for idx, ncessch in enumerate(df_clim['ncessch'].values)}
    
    # Create FAST reverse lookup: climate index -> clean sample index
    clim_idx_to_clean_idx = {}
    for clean_idx, ncessch in enumerate(ncessch_clean):
        if ncessch in ncessch_to_clim_idx:
            clim_idx = ncessch_to_clim_idx[ncessch]
            clim_idx_to_clean_idx[clim_idx] = clean_idx
    
    print(f"  Matched {len(clim_idx_to_clean_idx)} districts to climate data")
    
    # **CRITICAL**: Subset df_clean to only districts with climate data
    districts_with_climate = sorted(set(clim_idx_to_clean_idx.values()))
    df_clean_subset = df_clean.iloc[districts_with_climate].reset_index(drop=True)
    
    print(f"  Subsetting to {len(df_clean_subset)} districts with climate data")
    
    # Reconstruct geographic weights for subsetted sample
    print(f"  Reconstructing geographic KNN for subset...")
    w_geo_subset = KNN.from_dataframe(df_clean_subset, k=k)
    w_geo_subset.transform = 'r'
    
    # Remap climate weights to new indices (0 to len-1)
    old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(districts_with_climate)}
    
    # Subset climate weights using fast lookups
    neighbors_subset = {}
    weights_subset = {}
    
    for clim_idx in sorted(clim_idx_to_clean_idx.keys()):
        old_clean_idx = clim_idx_to_clean_idx[clim_idx]
        new_clean_idx = old_to_new_idx[old_clean_idx]
        
        # Get neighbors in climate space
        clim_neighbors = w_clim_full.neighbors[clim_idx]
        clim_weights_orig = w_clim_full.weights[clim_idx]
        
        # Map back to NEW clean sample indices (fast lookup)
        new_neighbors = []
        new_weights = []
        
        for j, clim_j in enumerate(clim_neighbors):
            if clim_j in clim_idx_to_clean_idx:
                old_neighbor_idx = clim_idx_to_clean_idx[clim_j]
                new_neighbor_idx = old_to_new_idx[old_neighbor_idx]
                new_neighbors.append(new_neighbor_idx)
                new_weights.append(clim_weights_orig[j])
        
        if new_neighbors:
            neighbors_subset[new_clean_idx] = new_neighbors
            weights_subset[new_clean_idx] = new_weights
    
    # Create new weights object
    w_clim = libpysal.weights.W(neighbors_subset, weights_subset)
    w_clim.transform = 'r'
    
    print(f"  Climate neighbors (subset): {w_clim.n} districts")
    print(f"  Mean climate neighbors per district: {w_clim.mean_neighbors:.2f}")
    
    # --- Run 2SLS with BOTH networks on SUBSETTED data ---
    print(f"\nRunning 2SLS (k={k}, n={len(df_clean_subset)})...")
    spec_name = f"KNN-{k}"
    result = run_2sls_spec(df_clean_subset, w_geo_subset, w_clim, spec_name, k)
    results_list.append(result)
    
    print(f"\nResults for k={k}:")
    print(f"  Geographic peer effect: {result['coef_geo']:.4f} (SE={result['se_geo']:.4f}, p={result['pval_geo']:.4f})")
    print(f"  Climate peer effect:    {result['coef_clim']:.4f} (SE={result['se_clim']:.4f}, p={result['pval_clim']:.4f})")
    print(f"  First-stage F-stats: Geo={result['fstat_geo']:.2f}, Clim={result['fstat_clim']:.2f}")
    print(f"  R-squared: {result['rsquared']:.4f}")
    print(f"  Sample size: {result['n']}")

# --- 4. SAVE RESULTS ---
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

df_results = pd.DataFrame(results_list)
df_results.to_csv(OUTPUT_FILE, index=False)
print(f"\nResults saved to: {OUTPUT_FILE}")
print(f"Total specifications: {len(df_results)}")

# --- 5. SUMMARY STATISTICS ---
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)

print("\nGeographic Peer Effects:")
print(df_results[['k', 'coef_geo', 'se_geo', 'pval_geo']].to_string(index=False))

print("\nClimate Peer Effects:")
print(df_results[['k', 'coef_clim', 'se_clim', 'pval_clim']].to_string(index=False))

print("\nFirst-Stage Diagnostics:")
print(df_results[['k', 'fstat_geo', 'fstat_clim']].to_string(index=False))

# Calculate average effects
print("\nAverage Effects Across Specifications:")
print(f"  Geographic: {df_results['coef_geo'].mean():.4f} (SD={df_results['coef_geo'].std():.4f})")
print(f"  Climate:    {df_results['coef_clim'].mean():.4f} (SD={df_results['coef_clim'].std():.4f})")

# Significance counts
geo_sig = (df_results['pval_geo'] < 0.05).sum()
clim_sig = (df_results['pval_clim'] < 0.05).sum()
print(f"\nSignificant at 5% level:")
print(f"  Geographic: {geo_sig}/{len(df_results)} specifications")
print(f"  Climate:    {clim_sig}/{len(df_results)} specifications")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
