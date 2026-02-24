"""
ESB Spatial Robustness Analysis - Simplified Version
Uses pre-processed CSV to avoid Excel file loading issues
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from libpysal.weights import KNN, DistanceBand
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Fix for Intel MKL threading issues
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configuration
DATASET_FILE = 'esb_full_analysis_dataset.csv'
SHAPEFILE_PATH = 'EDGE_SCHOOLDISTRICT_TL21_SY2021/schooldistrict_sy2021_tl21.shp'
WRI_FILE = 'ESB_adoption_dataset_v9_update_june_2025.xlsx'
OUTPUT_CSV = 'esb_robustness_results_with_charters.csv'
OUTPUT_TXT = 'esb_spatial_results_with_charters.txt'

# Regression specification
x_names_A = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pct_white', 'pm25']
x_names_B = x_names_A

print("="*80)
print("ESB ADOPTION PEER EFFECTS: ROBUSTNESS ANALYSIS WITH CHARTERS")
print("="*80)

# ============================================================================
# PART 1: LOAD DATA
# ============================================================================

print("\n[1/5] Loading Data...")
df = pd.read_csv(DATASET_FILE)
print(f"  Loaded {len(df)} districts from CSV")

# Create log transformations if not present
if 'log_median_income' not in df.columns and 'median_income' in df.columns:
    df['log_median_income'] = np.log(df['median_income'] + 1)
if 'log_enrollment' not in df.columns and 'enrollment' in df.columns:
    df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Load charter identification from pre-extracted CSV (avoids Excel file issues)
print("  Loading charter school flags from charter_flags.csv...")
try:
    charter_flags = pd.read_csv('charter_flags.csv')
    print(f"    Charter flags file: {len(charter_flags)} rows")
    print(f"    Sample charter flag nces_ids: {charter_flags['nces_id'].head(3).tolist()}")
    print(f"    Sample main df nces_ids: {df['nces_id'].head(3).tolist()}")
    
    # Ensure matching dtypes for merge
    charter_flags['nces_id'] = charter_flags['nces_id'].astype(int)
    df['nces_id'] = df['nces_id'].astype(int)
    
    df = df.merge(charter_flags, on='nces_id', how='left')
    df['is_charter'] = df['is_charter'].fillna(0).astype(int)
    
    print(f"  Charter schools: {df['is_charter'].sum()} ({100*df['is_charter'].mean():.1f}%)")
except Exception as e:
    print(f"  Warning: Could not load charter flags: {e}")
    import traceback
    traceback.print_exc()
    df['is_charter'] = 0

# ============================================================================
# PART 2: MERGE WITH SPATIAL DATA
# ============================================================================

print("\n[2/5] Loading Spatial Data...")
gdf = gpd.read_file(SHAPEFILE_PATH)
shp_id_col = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]
gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.lstrip('0').astype(int)

gdf_reg = gdf[[shp_id_col, 'geometry']].merge(df, left_on=shp_id_col, right_on='nces_id', how='inner')
gdf_reg = gdf_reg.to_crs(epsg=5070)
gdf_reg['centroid'] = gdf_reg.geometry.centroid

print(f"  Spatial sample: {len(gdf_reg)} districts")
print(f"  Charters in spatial sample: {gdf_reg['is_charter'].sum()}")

# Store clean dataset
reg_df = gdf_reg.copy()

# ============================================================================
# PART 3: RUN ROBUSTNESS SPECIFICATIONS
# ============================================================================

def run_2sls_spec(geo_df, spec_name, sample_name):
    """Run 2SLS specification"""
    required_vars = x_names_A + ['IS_ADOPTER', 'w_geo', 'w_clim', 'IV_Z', 'state']
    geo_df_clean = geo_df.dropna(subset=required_vars).copy()
    
    y = geo_df_clean['IS_ADOPTER'].values
    endog = geo_df_clean['w_geo'].values.reshape(-1, 1)
    exog_B = sm.add_constant(geo_df_clean[x_names_B].values)
    instruments = geo_df_clean[['w_clim', 'IV_Z']].values
    
    # Estimate
    model = IV2SLS(y, exog_B, endog, instruments)
    res = model.fit(cov_type='clustered', clusters=geo_df_clean['state'])
    
    # Extract coefficient (last parameter is the endogenous variable)
    w_geo_coef = res.params.iloc[-1]
    w_geo_se = res.std_errors.iloc[-1]
    w_geo_pval = res.pvalues.iloc[-1]
    
    # First stage F-stat
    first_stage = IV2SLS(endog.flatten(), sm.add_constant(geo_df_clean[x_names_A].values), 
                        np.empty((len(endog), 0)), instruments)
    fs_res = first_stage.fit()
    f_stat = fs_res.f_statistic.stat
    
    return {
        'specification': spec_name,
        'sample': sample_name,
        'n_obs': len(geo_df_clean),
        'w_geo_coef': w_geo_coef,
        'w_geo_se': w_geo_se,
        'w_geo_pval': w_geo_pval,
        'f_stat_geo': f_stat
    }

print("\n[3/5] Running Robustness Specifications...")
results_list = []

# KNN specifications
knn_values = [4, 6, 8, 10, 12, 15, 20]
for k in knn_values:
    print(f"  Running KNN k={k}...")
    
    # Create KNN weight matrix
    w_geo = KNN.from_dataframe(reg_df, k=k, geom_col='centroid')
    w_geo.transform = 'r'
    
    # Create climate network (k*2)
    w_clim = KNN.from_dataframe(reg_df, k=k*2, geom_col='centroid')
    w_clim.transform = 'r'
    
    # Spatial lags
    reg_df['w_geo'] = w_geo.sparse.dot(reg_df['IS_ADOPTER'].values)
    
    w_clim_only = w_clim.sparse - w_geo.sparse
    w_clim_only[w_clim_only < 0] = 0
    row_sums = w_clim_only.sum(axis=1).A1
    row_sums[row_sums == 0] = 1
    w_clim_only = w_clim_only.multiply(1/row_sums[:, np.newaxis])
    reg_df['w_clim'] = w_clim_only.dot(reg_df['IS_ADOPTER'].values)
    
    # Full sample
    res_full = run_2sls_spec(reg_df, f'KNN-{k}', 'Full Sample')
    results_list.append(res_full)
    
    # No charters
    gdf_no_charter = reg_df[reg_df['is_charter'] == 0].copy()
    w_geo_nc = KNN.from_dataframe(gdf_no_charter, k=k, geom_col='centroid')
    w_geo_nc.transform = 'r'
    w_clim_nc = KNN.from_dataframe(gdf_no_charter, k=k*2, geom_col='centroid')
    w_clim_nc.transform = 'r'
    
    gdf_no_charter['w_geo'] = w_geo_nc.sparse.dot(gdf_no_charter['IS_ADOPTER'].values)
    w_clim_only_nc = w_clim_nc.sparse - w_geo_nc.sparse
    w_clim_only_nc[w_clim_only_nc < 0] = 0
    row_sums_nc = w_clim_only_nc.sum(axis=1).A1
    row_sums_nc[row_sums_nc == 0] = 1
    w_clim_only_nc = w_clim_only_nc.multiply(1/row_sums_nc[:, np.newaxis])
    gdf_no_charter['w_clim'] = w_clim_only_nc.dot(gdf_no_charter['IS_ADOPTER'].values)
    
    res_no_charter = run_2sls_spec(gdf_no_charter, f'KNN-{k}', 'No Charters')
    results_list.append(res_no_charter)

# Distance band specifications
distance_bands = [25, 50, 75, 100]
for d in distance_bands:
    print(f"  Running Distance {d}km...")
    
    threshold_m = d * 1000
    
    w_geo = DistanceBand.from_dataframe(reg_df, threshold=threshold_m, geom_col='centroid', binary=False)
    w_geo.transform = 'r'
    
    w_clim = DistanceBand.from_dataframe(reg_df, threshold=threshold_m*2, geom_col='centroid', binary=False)
    w_clim.transform = 'r'
    
    reg_df['w_geo'] = w_geo.sparse.dot(reg_df['IS_ADOPTER'].values)
    w_clim_only = w_clim.sparse - w_geo.sparse
    w_clim_only[w_clim_only < 0] = 0
    row_sums = w_clim_only.sum(axis=1).A1
    row_sums[row_sums == 0] = 1
    w_clim_only = w_clim_only.multiply(1/row_sums[:, np.newaxis])
    reg_df['w_clim'] = w_clim_only.dot(reg_df['IS_ADOPTER'].values)
    
    res_full = run_2sls_spec(reg_df, f'Distance-{d}km', 'Full Sample')
    results_list.append(res_full)
    
    # No charters
    gdf_no_charter = reg_df[reg_df['is_charter'] == 0].copy()
    w_geo_nc = DistanceBand.from_dataframe(gdf_no_charter, threshold=threshold_m, geom_col='centroid', binary=False)
    w_geo_nc.transform = 'r'
    w_clim_nc = DistanceBand.from_dataframe(gdf_no_charter, threshold=threshold_m*2, geom_col='centroid', binary=False)
    w_clim_nc.transform = 'r'
    
    gdf_no_charter['w_geo'] = w_geo_nc.sparse.dot(gdf_no_charter['IS_ADOPTER'].values)
    w_clim_only_nc = w_clim_nc.sparse - w_geo_nc.sparse
    w_clim_only_nc[w_clim_only_nc < 0] = 0
    row_sums_nc = w_clim_only_nc.sum(axis=1).A1
    row_sums_nc[row_sums_nc == 0] = 1
    w_clim_only_nc = w_clim_only_nc.multiply(1/row_sums_nc[:, np.newaxis])
    gdf_no_charter['w_clim'] = w_clim_only_nc.dot(gdf_no_charter['IS_ADOPTER'].values)
    
    res_no_charter = run_2sls_spec(gdf_no_charter, f'Distance-{d}km', 'No Charters')
    results_list.append(res_no_charter)

# ============================================================================
# PART 4: EXPORT RESULTS
# ============================================================================

print("\n[4/5] Exporting Results...")
results_df = pd.DataFrame(results_list)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"  Saved to {OUTPUT_CSV}")

# Create formatted table
with open(OUTPUT_TXT, 'w') as f:
    f.write("="*80 + "\n")
    f.write("ESB ADOPTION PEER EFFECTS: ROBUSTNESS RESULTS WITH CHARTERS\n")
    f.write("="*80 + "\n\n")
    
    for _, row in results_df.iterrows():
        f.write(f"{row['specification']:15s} {row['sample']:15s} ")
        f.write(f"n={row['n_obs']:5d} ")
        f.write(f"β={row['w_geo_coef']:7.4f} ")
        f.write(f"SE={row['w_geo_se']:6.4f} ")
        f.write(f"p={row['w_geo_pval']:5.3f} ")
        f.write(f"F={row['f_stat_geo']:8.1f}\n")

print(f"  Saved to {OUTPUT_TXT}")
print("\n[5/5] Complete!")
print("="*80)
