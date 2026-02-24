"""
ESB Spatial Robustness - With Error Recovery
Saves results incrementally to survive forrtl crashes
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import libpysal
from libpysal.weights import KNN, DistanceBand
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
import warnings
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, CHARTER_FLAGS,
    TABLES_DIR, ensure_dirs_exist
)

# Ensure output directories exist
ensure_dirs_exist()

# Configuration
OUTPUT_CSV = TABLES_DIR / 'esb_robustness_results_with_charters.csv'

# Controls
x_names = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pct_white', 'pm25']

print("="*80)
print("ESB ROBUSTNESS WITH INCREMENTAL SAVING")
print("="*80)

# Load data
print("\n[1/3] Loading Data...")
df = pd.read_csv(str(ESB_FULL_ANALYSIS))

if 'log_median_income' not in df.columns:
    df['log_median_income'] = np.log(df['median_income'] + 1)
if 'log_enrollment' not in df.columns:
    df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Add charter flag if not already present
if 'is_charter' not in df.columns:
    print(f"  Adding charter flags from {CHARTER_FLAGS}...")
    charter_flags = pd.read_csv(str(CHARTER_FLAGS))
    charter_flags['nces_id'] = charter_flags['nces_id'].astype(int)
    df['nces_id'] = df['nces_id'].astype(int)
    df = df.merge(charter_flags, on='nces_id', how='left')
    df['is_charter'] = df['is_charter'].fillna(0).astype(int)
else:
    print("  Charter flag already in dataset")
    df['is_charter'] = df['is_charter'].astype(int)

print(f"  Districts: {len(df)}, Charters: {df['is_charter'].sum()}")

# Spatial merge
print("\n[2/3] Spatial Merge...")
gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
shp_id_col = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]
gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.lstrip('0').astype(int)

reg_df = gdf[[shp_id_col, 'geometry']].merge(df, left_on=shp_id_col, right_on='nces_id', how='inner')
reg_df = gpd.GeoDataFrame(reg_df, geometry='geometry')  # Ensure GeoDataFrame
reg_df = reg_df.to_crs(epsg=5070)
reg_df['centroid'] = reg_df.geometry.centroid

# Verify charter flag retained
print(f"  Spatial sample: {len(reg_df)}, Charters: {reg_df['is_charter'].sum()}")
print(f"  Charter columns check: {'is_charter' in reg_df.columns}")

# Regression function
def run_spec(geo_df, spec_name, sample_name):
    required_vars = x_names + ['IS_ADOPTER', 'w_geo_adoption', 'w_clim_adoption', 'w_geo_z', 'w_clim_z', 'IV_Z', 'state']
    geo_df_clean = geo_df.dropna(subset=required_vars).copy()
    
    y = geo_df_clean['IS_ADOPTER'].values
    endog = geo_df_clean[['w_geo_adoption', 'w_clim_adoption']].values
    exog = sm.add_constant(geo_df_clean[x_names].values)
    instruments = geo_df_clean[['w_geo_z', 'w_clim_z']].values
    
    model = IV2SLS(y, exog, endog, instruments)
    res = model.fit(cov_type='clustered', clusters=geo_df_clean['state'])
    
    # Get w_geo_adoption coefficient (first endog variable)
    w_geo_coef = res.params.iloc[len(x_names)+1]  # After constant and x_names, first endog
    w_geo_se = res.std_errors.iloc[len(x_names)+1]
    w_geo_pval = res.pvalues.iloc[len(x_names)+1]
    
    # First stage F-stat for w_geo_adoption
    f_stat = res.first_stage.diagnostics['f.stat'][0]  # First endogenous variable
    
    return {
        'specification': spec_name,
        'sample': sample_name,
        'n_obs': len(geo_df_clean),
        'w_geo_coef': w_geo_coef,
        'w_geo_se': w_geo_se,
        'w_geo_pval': w_geo_pval,
        'f_stat_geo': f_stat
    }

#  Load existing results if any
if os.path.exists(OUTPUT_CSV):
    results_df = pd.read_csv(OUTPUT_CSV)
    results_list = results_df.to_dict('records')
    completed_specs = set(results_df['specification'] + '_' + results_df['sample'])
    print(f"\nResuming: {len(results_list)} specs already completed")
else:
    results_list = []
    completed_specs = set()

print("\n[3/3] Running Specifications...")

# KNN specs
for k in [4, 6, 8, 10, 12, 15, 20]:
    for sample_type in ['Full Sample', 'No Charters']:
        spec_id = f'KNN-{k}_{sample_type}'
        
        if spec_id in completed_specs:
            print(f"  Skipping {spec_id} (already done)")
            continue
        
        try:
            print(f"  {spec_id}...")
            
            df_working = reg_df if sample_type == 'Full Sample' else reg_df[reg_df['is_charter'] == 0].copy()
            print(f"    Sample size: {len(df_working)}")
            
            w_geo = KNN.from_dataframe(df_working, k=k, geom_col='centroid')
            w_geo.transform = 'r'
            w_clim = KNN.from_dataframe(df_working, k=k*2, geom_col='centroid')
            w_clim.transform = 'r'
            
            # Create spatial lags (matching original script)
            df_working['w_geo_adoption'] = libpysal.weights.lag_spatial(w_geo, df_working['IS_ADOPTER'].values)
            df_working['w_clim_adoption'] = libpysal.weights.lag_spatial(w_clim, df_working['IS_ADOPTER'].values)
            df_working['w_geo_z'] = libpysal.weights.lag_spatial(w_geo, df_working['IV_Z'].values)
            df_working['w_clim_z'] = libpysal.weights.lag_spatial(w_clim, df_working['IV_Z'].values)
            
            result = run_spec(df_working, f'KNN-{k}', sample_type)
            results_list.append(result)
            
            # Save incrementally
            pd.DataFrame(results_list).to_csv(str(OUTPUT_CSV), index=False)
            print(f"    ✓ Saved (total: {len(results_list)})")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

# Distance bands
for d in [25, 50, 75, 100]:
    for sample_type in ['Full Sample', 'No Charters']:
        spec_id = f'Distance-{d}km_{sample_type}'
        
        if spec_id in completed_specs:
            print(f"  Skipping {spec_id} (already done)")
            continue
        
        try:
            print(f"  {spec_id}...")
            
            df_working = reg_df if sample_type == 'Full Sample' else reg_df[reg_df['is_charter'] == 0].copy()
            print(f"    Sample size: {len(df_working)}")
            
            threshold = d * 1000
            w_geo = DistanceBand.from_dataframe(df_working, threshold=threshold, geom_col='centroid', binary=False)
            w_geo.transform = 'r'
            w_clim = DistanceBand.from_dataframe(df_working, threshold=threshold*2, geom_col='centroid', binary=False)
            w_clim.transform = 'r'
            
            # Create spatial lags (matching original script)
            df_working['w_geo_adoption'] = libpysal.weights.lag_spatial(w_geo, df_working['IS_ADOPTER'].values)
            df_working['w_clim_adoption'] = libpysal.weights.lag_spatial(w_clim, df_working['IS_ADOPTER'].values)
            df_working['w_geo_z'] = libpysal.weights.lag_spatial(w_geo, df_working['IV_Z'].values)
            df_working['w_clim_z'] = libpysal.weights.lag_spatial(w_clim, df_working['IV_Z'].values)
            
            result = run_spec(df_working, f'Distance-{d}km', sample_type)
            results_list.append(result)
            
            # Save incrementally
            pd.DataFrame(results_list).to_csv(str(OUTPUT_CSV), index=False)
            print(f"    ✓ Saved (total: {len(results_list)})")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

print(f"\n✓ COMPLETE: {len(results_list)} specifications")
print(f"Saved to: {OUTPUT_CSV}")
print("="*80)
