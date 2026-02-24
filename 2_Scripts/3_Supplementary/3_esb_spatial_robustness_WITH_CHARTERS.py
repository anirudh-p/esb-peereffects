#!/usr/bin/env python3
"""
ESB Adoption Peer Effects: Comprehensive Robustness Analysis (WITH CHARTERS)
Author: Anirudh Narla
Purpose: Test robustness of geographic peer effects to alternative neighbor definitions
         Includes KNN (k=4,6,8,10,12,15,20) and distance bands (25,50,75,100km)
         Compares Full Sample vs No Charters specifications
         
CRITICAL: This script REBUILDS esb_full_analysis_dataset.csv from scratch,
          KEEPING charter schools (type 7) instead of dropping them like 1_esb_analysis_prep.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from libpysal.weights import KNN, DistanceBand
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Fix for Intel MKL threading issues causing forrtl errors
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# --- CONFIGURATION ---
# Input files
WRI_FILE = 'ESB_adoption_dataset_v9_update_june_2025.xlsx'
MASTER_REGRESSION_FILE = 'esb_master_data_for_regression.csv'
APPLICANT_FILE = 'CSBP Applicants waitlisted and rejected_11.18.25.xlsx'
SHAPEFILE_PATH = 'EDGE_SCHOOLDISTRICT_TL21_SY2021/schooldistrict_sy2021_tl21.shp'
CHARGING_STATION_FILE = 'alt_fuel_stations_historical_day (Jan 1 2020).csv'
ZIP_COUNTY_FILE = 'ZIP_COUNTY_122020.xlsx'
POLITICAL_DATA_FILE = 'dataverse_files/countypres_2000-2024.csv'

# Output files
OUTPUT_CSV = 'esb_robustness_results_with_charters.csv'
OUTPUT_TXT = 'esb_spatial_results_with_charters.txt'

print("="*80)
print("ESB ADOPTION PEER EFFECTS: COMPREHENSIVE ROBUSTNESS ANALYSIS")
print("WITH CHARTER SCHOOLS INCLUDED")
print("="*80)

# ============================================================================
# PART 1: REBUILD ANALYSIS DATASET (INCLUDING CHARTERS)
# ============================================================================

print("\n[1/8] Rebuilding Analysis Dataset from Source Files...")

# --- 1a. Load Event Data (Round Status) ---
print("  Loading event data (R1/R2/R3 status)...")
df_events = pd.read_csv(MASTER_REGRESSION_FILE, dtype={'NCES District ID': str})

# --- 1b. Load Applicant Priority Data ---
print("  Loading applicant priority data...")
df_app_raw = pd.read_excel(APPLICANT_FILE)
df_app_raw['NCES District ID'] = df_app_raw['NCES District ID'].astype(str).str.split('.').str[0].str.zfill(7)
df_app_raw.sort_values('School District Prioritized', ascending=False, inplace=True)
df_priority = df_app_raw[['NCES District ID', 'School District Prioritized']].drop_duplicates(subset='NCES District ID')

# --- 1c. Load WRI Agentic Data (ALL LEA TYPES - NO FILTERING) ---
print("  Loading WRI district-level data (ALL types including charters)...")
# Use try/except and simpler loading to avoid forrtl errors
try:
    df_wri = pd.read_excel(WRI_FILE, sheet_name='1. District-level data')
    print(f"    Successfully loaded District data")
except Exception as e:
    print(f"    Error loading Excel file: {e}")
    print(f"    Attempting with different engine...")
    df_wri = pd.read_excel(WRI_FILE, sheet_name='1. District-level data', engine=None)

n_raw = len(df_wri)
print(f"    Raw universe size: {n_raw}")

# Load utilities sheet separately
try:
    df_util_raw = pd.read_excel(WRI_FILE, sheet_name='4. Utilities')
    print(f"    Successfully loaded Utilities data")
except Exception as e:
    print(f"    Warning: Could not load Utilities sheet: {e}")
    df_util_raw = None

# CRITICAL DIFFERENCE: We do NOT filter by LEA type here (no exclusion of type 7)
# Original 1_esb_analysis_prep.py had: df_wri = df_wri[df_wri['1k. LEA type (number)'].isin([1, 2])]
# We keep ALL types to preserve charter schools

# Map and rename columns
column_map = {
    '1c. LEA ID': 'nces_id',
    '1b. Local Education Agency (LEA) or entity name': 'district_name',
    '1g. State': 'state',
    '1p. Locale broad type (name)': 'urbanicity',
    '2a. Total number of buses': 'fleet_size',
    '3a. Number of ESBs committed': 'total_esbs_committed',
    '4b. Number of students in district': 'enrollment',
    '4f. Median household income': 'median_income',
    '4g. Percent of population below the poverty level': 'poverty_rate',
    '4u. Percent Hispanic or Latino (of any race) ': 'pct_hispanic',
    '4i. Percent race alone or multiracial: White': 'pct_white',
    '4k. Percent race alone or multiracial: Black or African American': 'pct_black',
    '5f. PM2.5 concentration': 'pm25',
}

available_cols = [c for c in column_map.keys() if c in df_wri.columns]
df_wri_clean = df_wri[available_cols + ['1k. LEA type (number)', '1l. LEA type (name)']].rename(columns=column_map)

# Clean NCES ID - filter non-numeric, convert to int
df_wri_clean['nces_id_str'] = df_wri_clean['nces_id'].astype(str).str.replace('.0', '', regex=False)
df_wri_clean = df_wri_clean[df_wri_clean['nces_id_str'].str.isdigit()].copy()
df_wri_clean['nces_id'] = df_wri_clean['nces_id_str'].astype(int)
df_wri_clean = df_wri_clean.drop(columns=['nces_id_str'])

# --- 1d. Create Charter School Flag ---
print("  Identifying charter schools...")
df_wri_clean['is_charter'] = (
    (df_wri_clean['1k. LEA type (number)'] == 7) | 
    (df_wri_clean['1l. LEA type (name)'].str.contains('Charter', case=False, na=False))
).astype(int)

n_charters = df_wri_clean['is_charter'].sum()
print(f"    Charter schools identified: {n_charters} ({100*n_charters/len(df_wri_clean):.1f}%)")
print(f"    Traditional districts: {len(df_wri_clean) - n_charters}")

# Drop the LEA type columns (no longer needed)
df_wri_clean = df_wri_clean.drop(columns=['1k. LEA type (number)', '1l. LEA type (name)'])

# --- 1e. Merge Datasets ---
print("  Merging with event and priority data...")

# Convert event file IDs to int for consistency
df_events['nces_id'] = df_events['NCES District ID'].astype(str).str.split('.').str[0]
df_events = df_events[df_events['nces_id'].str.isdigit()].copy()
df_events['nces_id'] = df_events['nces_id'].astype(int)

df_priority['nces_id'] = df_priority['NCES District ID'].astype(str).str.split('.').str[0]
df_priority = df_priority[df_priority['nces_id'].str.isdigit()].copy()
df_priority['nces_id'] = df_priority['nces_id'].astype(int)

df = pd.merge(df_wri_clean, df_events, on='nces_id', how='left')
df = pd.merge(df, df_priority, on='nces_id', how='left')

# Fill NaNs
fill_zeros = ['IS_ADOPTER', 'IS_APPLICANT', 'NUM_ADOPTED', 'IV_Z']
for col in fill_zeros:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

# Backfill applicant status (adopters must have applied)
if 'IS_ADOPTER' in df.columns and 'IS_APPLICANT' in df.columns:
    df.loc[df['IS_ADOPTER'] == 1, 'IS_APPLICANT'] = 1

if 'Funding Mechanism' in df.columns:
    df['Funding Mechanism'] = df['Funding Mechanism'].fillna('None')

# Create binary priority variable
def normalize_priority(x):
    if pd.isna(x): return 0
    s = str(x).upper()
    return 1 if s in ['YES', 'TRUE', '1'] else 0

if 'School District Prioritized' in df.columns:
    df['is_priority'] = df['School District Prioritized'].apply(normalize_priority)
else:
    df['is_priority'] = 0

# Define lottery winner/loser
if 'IV_Z' in df.columns:
    df['IS_LOTTERY_WINNER'] = df['IV_Z']
    df['IS_LOTTERY_LOSER'] = ((df['IS_APPLICANT'] == 1) & (df['IV_Z'] == 0)).astype(int)
else:
    df['IS_LOTTERY_WINNER'] = 0
    df['IS_LOTTERY_LOSER'] = 0

# Engineered features
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

n_before_covariates = len(df)
print(f"    Sample size before covariate filter: {n_before_covariates}")
print(f"    Charters before covariate filter: {df['is_charter'].sum()}")

# ============================================================================
# PART 2: ADD ADDITIONAL COVARIATES
# ============================================================================

print("\n[2/8] Loading Additional Covariates...")

# --- 2a. LOAD UTILITY DATA ---
print("  Loading utility ownership data...")
# Data already loaded in Part 1 as df_util_raw
if df_util_raw is not None:
    util_df = df_util_raw.copy()
    
    # Clean LEA ID
    util_df['lea_id_str'] = util_df['1c. LEA ID'].astype(str).str.replace('.0', '', regex=False)
    util_df = util_df[util_df['lea_id_str'].str.isdigit()].copy()
    util_df['nces_id'] = util_df['lea_id_str'].astype(int)

    # Extract utility ownership type
    ownership_cols = {
        'coop': '9b. Cooperative ownership',
        'federal': '9c. Federal ownership',
        'investor': '9d. Investor ownership',
        'municipal': '9e. Municipal ownership',
        'muni_marketing': '9f. Municipal marketing authority',
        'political_subdiv': '9g. Political subdivision',
        'state': '9h. State ownership',
        'wholesale': '9i. Wholesale'
    }

    for short, col in ownership_cols.items():
        if col in util_df.columns:
            util_df[short] = util_df[col].fillna(0)

    util_df['utility_type'] = util_df[[col for col in ownership_cols.keys() if col in util_df.columns]].idxmax(axis=1)
    util_df['utility_type'] = util_df['utility_type'].fillna('unknown')

    util_agg = util_df.groupby('nces_id')['utility_type'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown').reset_index()

    print(f"    Loaded utility data for {len(util_agg)} districts")

    # Merge utility data
    df = df.merge(util_agg, on='nces_id', how='left')
    df['utility_type'] = df['utility_type'].fillna('unknown')
else:
    print(f"    Skipping utility data (not available)")
    df['utility_type'] = 'unknown'

# --- 2b. LOAD POLITICAL DATA (COUNTY-LEVEL) ---
print("  Loading county-level political data...")
pres_df = pd.read_csv(POLITICAL_DATA_FILE)

# Filter to 2020 presidential election
pres_2020 = pres_df[pres_df['year'] == 2020].copy()

# Calculate Democratic vote share by county
county_results = pres_2020.groupby(['state_po', 'county_name']).apply(
    lambda x: pd.Series({
        'total_votes': x['candidatevotes'].sum(),
        'dem_votes': x[x['party'] == 'DEMOCRAT']['candidatevotes'].sum()
    })
).reset_index()

county_results['pct_dem_2020'] = county_results['dem_votes'] / county_results['total_votes']
county_results['county_fips'] = county_results['state_po'] + '_' + county_results['county_name'].str.upper()

# For now, we'll skip the county merge (requires geocoding districts to counties)
# Just create placeholder
df['pct_dem_2020'] = 0.5  # Neutral default

# --- 2c. LOAD EV CHARGING INFRASTRUCTURE ---
print("  Loading EV charging station data...")
stations = pd.read_csv(CHARGING_STATION_FILE)

# Filter to electric stations
stations = stations[stations['Fuel Type Code'] == 'ELEC'].copy()

# Clean ZIP codes (remove non-numeric)
stations['ZIP'] = stations['ZIP'].astype(str).str[:5]
stations = stations[stations['ZIP'].str.isdigit()].copy()

print(f"    Loaded {len(stations)} electric charging stations")

# Load ZIP-County crosswalk
zip_county = pd.read_excel(ZIP_COUNTY_FILE)
zip_county['ZIP'] = zip_county['ZIP'].astype(str).str.zfill(5)
zip_county = zip_county[zip_county['ZIP'].str.isdigit()].copy()

# Merge stations with county
stations_county = stations.merge(zip_county[['ZIP', 'COUNTY']], on='ZIP', how='left')
stations_county = stations_county.dropna(subset=['COUNTY'])

# Count chargers per county
chargers_per_county = stations_county.groupby('COUNTY').size().reset_index(name='ev_chargers_county')

# For now, skip the complex county->LEA aggregation
# Create placeholder
df['ev_chargers_baseline'] = 0  # Will need proper county-level merge

print(f"    EV infrastructure data loaded (county-level aggregation pending)")

# ============================================================================
# PART 3: CREATE URBANICITY DUMMIES
# ============================================================================

print("\n[3/8] Creating Control Variables...")

# Urbanicity dummies
if 'urbanicity' in df.columns:
    print(f"    Original urbanicity values: {df['urbanicity'].value_counts().to_dict()}")
    
    urbanicity_map = {
        'City': 'urban',
        'Suburb': 'suburban', 
        'Town': 'town',
        'Rural': 'rural'
    }
    
    df['urbanicity_clean'] = df['urbanicity'].map(urbanicity_map).fillna('rural')
    
    for cat in ['urban', 'suburban', 'town']:
        df[f'is_{cat}'] = (df['urbanicity_clean'] == cat).astype(int)
    
    print(f"    Urban: {df['is_urban'].sum()}, Suburban: {df['is_suburban'].sum()}, Town: {df['is_town'].sum()}, Rural: {(df['urbanicity_clean']=='rural').sum()}")
else:
    print("    WARNING: urbanicity column not found, creating dummy variables")
    df['is_urban'] = 0
    df['is_suburban'] = 0
    df['is_town'] = 0

print(f"    Created urbanicity dummies")

# ============================================================================
# PART 4: HANDLE MISSING DATA (NO FILTERING - FILL WITH DEFAULTS)
# ============================================================================

print("\n[4/8] Handling Missing Data...")

n_before = len(df)
n_charters_before = df['is_charter'].sum()

# Instead of dropping, fill missing values with reasonable defaults
print("    Filling missing numeric variables with median/mean...")
for col in ['median_income', 'poverty_rate', 'enrollment', 'pm25', 'pct_white', 'pct_black', 'pct_hispanic']:
    if col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df[col].fillna(median_val)
                print(f"      {col}: filled {n_missing} missing values with median {median_val:.2f}")

# Recalculate log variables after filling
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Fill categorical/binary variables
for col in ['is_urban', 'is_suburban', 'is_town']:
    if col in df.columns:
        df[col] = df[col].fillna(0)

n_after = len(df)
n_charters_after = df['is_charter'].sum()

print(f"    Sample size retained: {n_after} districts (no filtering)")
print(f"    Charter schools in sample: {n_charters_after} ({100*n_charters_after/n_after:.1f}%)")
print(f"    Traditional districts: {n_after - n_charters_after}")

# Check for constant/near-constant variables
print("\n    Checking variable variance...")
check_vars = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pct_white', 'pct_black', 'pm25', 'is_urban', 'is_suburban']
for var in check_vars:
    if var in df.columns:
        var_std = df[var].std()
        var_mean = df[var].mean()
        var_min = df[var].min()
        var_max = df[var].max()
        if var_std < 0.001:
            print(f"      WARNING: {var} has very low variance (std={var_std:.6f})")
        print(f"      {var}: mean={var_mean:.3f}, std={var_std:.3f}, range=[{var_min:.3f}, {var_max:.3f}]")

# Ensure nces_id is integer for spatial merge
df['nces_id'] = df['nces_id'].astype(int)

# ============================================================================
# PART 5: SPATIAL MATRIX CONSTRUCTION
# ============================================================================

print("\n[5/8] Constructing Spatial Weight Matrices...")

gdf = gpd.read_file(SHAPEFILE_PATH)
if 'GEOID' not in gdf.columns:
    shp_id_col = [c for c in gdf.columns if 'ID' in c][0]
else:
    shp_id_col = 'GEOID'

# Convert shapefile IDs to integer (they may be stored as strings with leading zeros)
gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.lstrip('0').astype(int)

# Merge and project
gdf_reg = gdf[[shp_id_col, 'geometry']].merge(df, left_on=shp_id_col, right_on='nces_id', how='inner')
gdf_reg = gdf_reg.to_crs(epsg=5070)  # Albers Equal Area
gdf_reg['centroid'] = gdf_reg.geometry.centroid

n_spatial_sample = len(gdf_reg)
n_charters_spatial = gdf_reg['is_charter'].sum()

print(f"    Spatial sample size: {n_spatial_sample} districts")
print(f"    Charters in spatial sample: {n_charters_spatial}")
print(f"    Lost {n_after - n_spatial_sample} districts in spatial merge")

# Store the clean dataset for regression
reg_df = gdf_reg.copy()

# ============================================================================
# PART 6: DEFINE REGRESSION SPECIFICATION
# ============================================================================

print("\n[6/8] Preparing Regression Specification...")

# Control variables - EXCLUDE urbanicity dummies (they have zero variance)
x_names_A = [
    'log_median_income', 'poverty_rate', 'log_enrollment',
    'pct_white', 'pm25'
]

x_names_B = x_names_A  # No state FE to avoid multicollinearity

print(f"    First stage controls: {len(x_names_A)} variables: {x_names_A}")
print(f"    Second stage controls: {len(x_names_B)} variables")
print(f"    Note: Urbanicity dummies excluded due to zero variance")

def run_2sls_spec(geo_df, spec_name, sample_name):
    """Run 2SLS specification with spatial lag instruments"""
    
    # Check for missing values in required variables
    required_vars = x_names_A + ['IS_ADOPTER', 'w_geo', 'w_clim', 'IV_Z', 'state']
    geo_df_clean = geo_df.dropna(subset=required_vars).copy()
    
    if len(geo_df_clean) < len(geo_df):
        print(f"      Warning: Dropped {len(geo_df) - len(geo_df_clean)} observations with missing values")
    
    y = geo_df_clean['IS_ADOPTER'].values
    endog = geo_df_clean['w_geo'].values.reshape(-1, 1)
    exog_B = geo_df_clean[x_names_B].values
    instruments = geo_df_clean[['w_clim', 'IV_Z']].values
    
    # Add constant to exog only (not instruments)
    exog_B = sm.add_constant(exog_B)
    
    # Check for multicollinearity
    exog_endog = np.column_stack([exog_B, endog])
    rank = np.linalg.matrix_rank(exog_endog)
    expected_rank = exog_endog.shape[1]
    
    if rank < expected_rank:
        print(f"      ERROR: Rank deficiency detected!")
        print(f"        Matrix shape: {exog_endog.shape}, Rank: {rank}, Expected: {expected_rank}")
        
        # Check which variables cause issues
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        exog_scaled = scaler.fit_transform(exog_B)
        corr_matrix = np.corrcoef(exog_scaled.T)
        
        # Find highly correlated pairs
        high_corr = np.where(np.abs(corr_matrix) > 0.95)
        for i, j in zip(high_corr[0], high_corr[1]):
            if i < j:
                var_names = ['const'] + x_names_B
                print(f"        High correlation between {var_names[i]} and {var_names[j]}: {corr_matrix[i,j]:.3f}")
        
        # Try without constant
        print(f"      Attempting without constant term...")
        exog_B_no_const = geo_df_clean[x_names_B].values
        exog_endog_no_const = np.column_stack([exog_B_no_const, endog])
        rank_no_const = np.linalg.matrix_rank(exog_endog_no_const)
        print(f"        Without constant - Rank: {rank_no_const}/{exog_endog_no_const.shape[1]}")
        
        if rank_no_const == exog_endog_no_const.shape[1]:
            print(f"      Running without constant term...")
            exog_B = exog_B_no_const
            exog_A = geo_df_clean[x_names_A].values
        else:
            raise ValueError("Cannot resolve rank deficiency")
    else:
        exog_A = sm.add_constant(geo_df_clean[x_names_A].values)
    
    # Estimate 2SLS
    model = IV2SLS(y, exog_B, endog, instruments)
    res = model.fit(cov_type='clustered', clusters=geo_df_clean['state'])
    
    # Extract results - the endogenous variable is named 'w_geo'
    # If not found by name, it's the last parameter (after exog_B variables)
    if 'w_geo' in res.params.index:
        w_geo_coef = res.params['w_geo']
        w_geo_se = res.std_errors['w_geo']
        w_geo_pval = res.pvalues['w_geo']
    else:
        # Endogenous variable is the last parameter
        w_geo_coef = res.params.iloc[-1]
        w_geo_se = res.std_errors.iloc[-1]
        w_geo_pval = res.pvalues.iloc[-1]
    
    # First stage F-stat
    first_stage = IV2SLS(endog.flatten(), exog_A, np.empty((len(endog), 0)), instruments)
    fs_res = first_stage.fit()
    f_stat = fs_res.f_statistic.stat
    
    return {
        'specification': spec_name,
        'sample': sample_name,
        'w_geo': w_geo_coef,
        'se': w_geo_se,
        'pval': w_geo_pval,
        'sig': '***' if w_geo_pval < 0.01 else '**' if w_geo_pval < 0.05 else '*' if w_geo_pval < 0.10 else '',
        'f_stat': f_stat,
        'n': len(geo_df_clean)
    }

# ============================================================================
# PART 7: RUN ROBUSTNESS ANALYSIS
# ============================================================================

print("\n[7/8] Running Robustness Specifications...")

results_list = []

# --- KNN SPECIFICATIONS ---
print("\n  KNN Specifications:")
knn_values = [4, 6, 8, 10, 12, 15, 20]

for k in knn_values:
    print(f"    Running KNN k={k}...")
    
    # Create geographic peer network (KNN-k)
    w_geo = KNN.from_dataframe(gdf_reg, k=k, geom_col='centroid')
    w_geo.transform = 'r'  # Row-standardize
    
    # Create regional climate network (KNN larger radius for climate effects)
    w_clim = KNN.from_dataframe(gdf_reg, k=k+10, geom_col='centroid')
    w_clim.transform = 'r'
    
    # Create spatial lags on FULL sample
    gdf_reg['w_geo'] = w_geo.sparse.dot(gdf_reg['IS_ADOPTER'].values)
    
    # For climate instrument: neighbors in w_clim but NOT in w_geo (donut)
    w_clim_only = w_clim.sparse - w_geo.sparse
    w_clim_only[w_clim_only < 0] = 0
    row_sums = w_clim_only.sum(axis=1).A1
    row_sums[row_sums == 0] = 1
    w_clim_only = w_clim_only.multiply(1/row_sums[:, np.newaxis])
    gdf_reg['w_clim'] = w_clim_only.dot(gdf_reg['IS_ADOPTER'].values)
    
    # Run on FULL sample
    res_full = run_2sls_spec(gdf_reg, f'KNN-{k}', 'Full Sample')
    results_list.append(res_full)
    
    # Run on NO CHARTERS sample
    gdf_no_charter = gdf_reg[gdf_reg['is_charter'] == 0].copy()
    
    # Rebuild spatial weights for no-charter sample
    w_geo_nc = KNN.from_dataframe(gdf_no_charter, k=k, geom_col='centroid')
    w_geo_nc.transform = 'r'
    w_clim_nc = KNN.from_dataframe(gdf_no_charter, k=k+10, geom_col='centroid')
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

# --- DISTANCE BAND SPECIFICATIONS ---
print("\n  Distance Band Specifications:")
distance_km = [25, 50, 75, 100]

for d in distance_km:
    print(f"    Running Distance Band {d}km...")
    
    threshold_m = d * 1000
    
    # Create geographic peer network
    w_geo = DistanceBand.from_dataframe(gdf_reg, threshold=threshold_m, geom_col='centroid', binary=False)
    w_geo.transform = 'r'
    
    # Create regional network (2x distance)
    w_clim = DistanceBand.from_dataframe(gdf_reg, threshold=threshold_m*2, geom_col='centroid', binary=False)
    w_clim.transform = 'r'
    
    # Check for islands
    cardinalities = w_geo.cardinalities
    if isinstance(cardinalities, (list, np.ndarray)):
        n_islands = (np.array(cardinalities) == 0).sum()
    else:
        n_islands = 0  # No islands if cardinalities is scalar
    
    if n_islands > 0:
        print(f"      WARNING: {n_islands} islands detected at {d}km")
    
    # Create spatial lags
    gdf_reg['w_geo'] = w_geo.sparse.dot(gdf_reg['IS_ADOPTER'].values)
    w_clim_only = w_clim.sparse - w_geo.sparse
    w_clim_only[w_clim_only < 0] = 0
    row_sums = w_clim_only.sum(axis=1).A1
    row_sums[row_sums == 0] = 1
    w_clim_only = w_clim_only.multiply(1/row_sums[:, np.newaxis])
    gdf_reg['w_clim'] = w_clim_only.dot(gdf_reg['IS_ADOPTER'].values)
    
    # Run on FULL sample
    res_full = run_2sls_spec(gdf_reg, f'{d}km', 'Full Sample')
    results_list.append(res_full)
    
    # Run on NO CHARTERS sample
    gdf_no_charter = gdf_reg[gdf_reg['is_charter'] == 0].copy()
    
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
    
    res_no_charter = run_2sls_spec(gdf_no_charter, f'{d}km', 'No Charters')
    results_list.append(res_no_charter)

# ============================================================================
# PART 8: EXPORT RESULTS
# ============================================================================

print("\n[8/8] Exporting Results...")

# Convert to DataFrame
results_df = pd.DataFrame(results_list)

# Save CSV
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"    Saved results to {OUTPUT_CSV}")

# Create formatted table
with open(OUTPUT_TXT, 'w') as f:
    f.write("="*80 + "\n")
    f.write("ESB ADOPTION PEER EFFECTS: ROBUSTNESS TO NEIGHBOR DEFINITION\n")
    f.write("WITH CHARTER SCHOOLS INCLUDED\n")
    f.write("="*80 + "\n\n")
    
    f.write("Specification".ljust(20))
    f.write("Sample".ljust(15))
    f.write("w_geo".rjust(10))
    f.write("SE".rjust(10))
    f.write("p-val".rjust(10))
    f.write("Sig".rjust(5))
    f.write("F-stat".rjust(12))
    f.write("N".rjust(8))
    f.write("\n" + "-"*80 + "\n")
    
    for _, row in results_df.iterrows():
        f.write(f"{row['specification']:20s}")
        f.write(f"{row['sample']:15s}")
        f.write(f"{row['w_geo']:10.4f}")
        f.write(f"{row['se']:10.4f}")
        f.write(f"{row['pval']:10.4f}")
        f.write(f"{row['sig']:>5s}")
        f.write(f"{row['f_stat']:12.2f}")
        f.write(f"{row['n']:8d}")
        f.write("\n")

print(f"    Saved formatted table to {OUTPUT_TXT}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print(f"Charter schools in final sample: {n_charters_spatial}")
print("="*80)
