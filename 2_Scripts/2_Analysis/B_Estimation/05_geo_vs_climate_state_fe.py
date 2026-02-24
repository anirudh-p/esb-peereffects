"""
Detailed Geographic vs Climate Peer Effects with State FE
==========================================================
Re-runs K=10 specification with:
  1. State fixed effects
  2. Full coefficient table for all covariates
  3. Detailed inference statistics
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import libpysal
from libpysal.weights import KNN
from linearmodels.iv import IV2SLS
import pickle
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, CLIMATE_WEIGHTS_DIR,
    CLIMATE_SIMILARITY_DATA, TABLES_DIR, get_climate_weight_file,
    ensure_dirs_exist
)

# Ensure output directories exist
ensure_dirs_exist()

print("="*80)
print("DETAILED GEO VS CLIMATE COMPARISON (K=10) WITH STATE FE")
print("="*80)

# --- CONFIGURATION ---
K = 10
OUTPUT_FILE = TABLES_DIR / 'geo_vs_climate_detailed_k10_with_state_fe.csv'
OUTPUT_SUMMARY = TABLES_DIR / 'geo_vs_climate_summary_k10_state_fe.csv'

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

# Rename variables
df['esb'] = df['IS_ADOPTER']
df['z'] = df['IV_Z']

print(f"  Sample size: {len(df)}")
print(f"  States: {df['state'].nunique()}")

# Load shapefile
print("\nLoading shapefile...")
gdf_shape = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
gdf_shape['ncessch'] = gdf_shape['GEOID'].str.zfill(7)

# Merge spatial and attribute data
gdf = gdf_shape[['ncessch', 'geometry']].merge(
    df[['nces_id', 'state'] + ['esb', 'z'] + CONTROLS], 
    left_on='ncessch', 
    right_on='nces_id', 
    how='inner'
)

# Project to Albers
gdf = gdf.to_crs('ESRI:102003')

# Drop missing values
df_clean = gdf.dropna(subset=['esb', 'z', 'state'] + CONTROLS)
print(f"  Clean sample: {len(df_clean)}")

# --- 2. CONSTRUCT WEIGHTS ---
print(f"\nConstructing geographic KNN (k={K})...")
w_geo = KNN.from_dataframe(df_clean, k=K)
w_geo.transform = 'r'

print(f"\nLoading climate KNN (k={K})...")
climate_weights_file = get_climate_weight_file(K)
with open(climate_weights_file, 'rb') as f:
    w_clim_full = pickle.load(f)

# Subset climate weights (same logic as before)
print(f"\nAligning climate weights...")
ncessch_clean = df_clean['ncessch'].values
df_clim = pd.read_csv(str(Path(__file__).parent.parent.parent.parent / '1_Data' / 'Cleaned' / 'district_climate_normals.csv'), dtype={'ncessch': str})

ncessch_to_clim_idx = {ncessch: idx for idx, ncessch in enumerate(df_clim['ncessch'].values)}
clim_idx_to_clean_idx = {}
for clean_idx, ncessch in enumerate(ncessch_clean):
    if ncessch in ncessch_to_clim_idx:
        clim_idx = ncessch_to_clim_idx[ncessch]
        clim_idx_to_clean_idx[clim_idx] = clean_idx

districts_with_climate = sorted(set(clim_idx_to_clean_idx.values()))
df_clean_subset = df_clean.iloc[districts_with_climate].reset_index(drop=True)

print(f"  Subset to {len(df_clean_subset)} districts with climate data")

# Reconstruct geographic weights for subset
w_geo_subset = KNN.from_dataframe(df_clean_subset, k=K)
w_geo_subset.transform = 'r'

# Remap climate weights
old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(districts_with_climate)}

neighbors_subset = {}
weights_subset = {}

for clim_idx in sorted(clim_idx_to_clean_idx.keys()):
    old_clean_idx = clim_idx_to_clean_idx[clim_idx]
    new_clean_idx = old_to_new_idx[old_clean_idx]
    
    clim_neighbors = w_clim_full.neighbors[clim_idx]
    clim_weights_orig = w_clim_full.weights[clim_idx]
    
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

w_clim = libpysal.weights.W(neighbors_subset, weights_subset)
w_clim.transform = 'r'

print(f"  Climate neighbors: {w_clim.n} districts")

# --- 3. PREPARE DATA WITH STATE FE ---
print(f"\nPreparing data with state fixed effects...")

# Create spatial lags
w_geo_adoption = libpysal.weights.lag_spatial(w_geo_subset, df_clean_subset['esb'].values)
w_clim_adoption = libpysal.weights.lag_spatial(w_clim, df_clean_subset['esb'].values)
w_geo_z = libpysal.weights.lag_spatial(w_geo_subset, df_clean_subset['z'].values)
w_clim_z = libpysal.weights.lag_spatial(w_clim, df_clean_subset['z'].values)

df_reg = df_clean_subset.copy()
df_reg['w_geo_adoption'] = w_geo_adoption
df_reg['w_clim_adoption'] = w_clim_adoption
df_reg['w_geo_z'] = w_geo_z
df_reg['w_clim_z'] = w_clim_z

# Drop any remaining NaN
df_reg = df_reg.dropna(subset=['w_geo_adoption', 'w_clim_adoption', 'w_geo_z', 'w_clim_z'])

print(f"  Final sample: {len(df_reg)}")
print(f"  States: {df_reg['state'].nunique()}")

# Create state dummies (drop first for reference category)
state_dummies = pd.get_dummies(df_reg['state'], prefix='state', drop_first=True)
print(f"  State FE variables: {len(state_dummies.columns)}")

# --- 4. RUN 2SLS WITH STATE FE ---
print(f"\nRunning 2SLS with state fixed effects...")

# Dependent variable
y = df_reg['esb'].values

# Endogenous variables
endog = df_reg[['w_geo_adoption', 'w_clim_adoption']].values

# Instruments
instruments = df_reg[['w_geo_z', 'w_clim_z']].values

# Exogenous controls: covariates + state FE
exog_controls = df_reg[CONTROLS].values
exog_state_fe = state_dummies.values
exog = np.column_stack([np.ones(len(exog_controls)), exog_controls, exog_state_fe])

print(f"  Total exogenous variables: {exog.shape[1]} (1 intercept + {len(CONTROLS)} controls + {state_dummies.shape[1]} state FE)")

# Fit model with state-clustered standard errors
model = IV2SLS(y, exog, endog, instruments)
results = model.fit(cov_type='clustered', clusters=df_reg['state'])

# --- 5. EXTRACT AND DISPLAY RESULTS ---
print("\n" + "="*80)
print("FULL REGRESSION RESULTS (K=10, WITH STATE FE)")
print("="*80)

# Create variable names
var_names = ['Intercept'] + CONTROLS + list(state_dummies.columns) + ['w_geo_adoption', 'w_clim_adoption']

# Build results table
results_df = pd.DataFrame({
    'Variable': var_names,
    'Coefficient': results.params.values,
    'Std_Error': results.std_errors.values,
    'T_Stat': results.tstats.values,
    'P_Value': results.pvalues.values
})

# Add significance stars
def add_stars(p):
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.10:
        return '*'
    else:
        return ''

results_df['Sig'] = results_df['P_Value'].apply(add_stars)

# Display main coefficients (exclude state FE for readability)
print("\n--- MAIN COEFFICIENTS (COVARIATES + PEER EFFECTS) ---")
main_vars = ['Intercept'] + CONTROLS + ['w_geo_adoption', 'w_clim_adoption']
main_results = results_df[results_df['Variable'].isin(main_vars)]
print(main_results.to_string(index=False))

# Count significant covariates
n_sig_5pct = (main_results['P_Value'] < 0.05).sum() - 1  # Exclude intercept
print(f"\nSignificant at 5% level: {n_sig_5pct}/{len(main_vars)-1} variables (excluding intercept)")

# Display state FE summary
print("\n--- STATE FIXED EFFECTS SUMMARY ---")
state_fe_results = results_df[results_df['Variable'].str.startswith('state_')]
n_states_sig = (state_fe_results['P_Value'] < 0.05).sum()
print(f"Number of state FE: {len(state_fe_results)}")
print(f"Significant at 5%: {n_states_sig}/{len(state_fe_results)}")
print(f"\nState FE range: [{state_fe_results['Coefficient'].min():.4f}, {state_fe_results['Coefficient'].max():.4f}]")

# Model diagnostics
print("\n--- MODEL DIAGNOSTICS ---")
print(f"N: {len(df_reg)}")
print(f"R-squared: {results.rsquared:.4f}")
print(f"Adjusted R-squared: {results.rsquared_adj:.4f}")

print("\nFirst-stage F-statistics:")
try:
    print(f"  w_geo_adoption: {results.first_stage.diagnostics.iloc[0]['f.stat']:.2f}")
    print(f"  w_clim_adoption: {results.first_stage.diagnostics.iloc[1]['f.stat']:.2f}")
except:
    print("  (F-stats not available)")

# Save detailed results
results_df.to_csv(str(OUTPUT_FILE), index=False)
print(f"\nDetailed results saved to: {OUTPUT_FILE}")

# Save summary statistics
summary_stats = {
    'specification': 'KNN-10 with State FE',
    'n': len(df_reg),
    'n_states': df_reg['state'].nunique(),
    'coef_geo': results.params.iloc[-2],
    'se_geo': results.std_errors.iloc[-2],
    'pval_geo': results.pvalues.iloc[-2],
    'coef_clim': results.params.iloc[-1],
    'se_clim': results.std_errors.iloc[-1],
    'pval_clim': results.pvalues.iloc[-1],
    'rsquared': results.rsquared,
    'rsquared_adj': results.rsquared_adj,
    'n_covariates_sig_5pct': n_sig_5pct,
    'n_state_fe_sig_5pct': n_states_sig
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv(str(OUTPUT_SUMMARY), index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
