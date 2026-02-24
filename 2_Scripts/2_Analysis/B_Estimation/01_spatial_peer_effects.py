import pandas as pd
import geopandas as gpd
import numpy as np
import libpysal 
from libpysal.weights import KNN, W
from linearmodels.iv import IV2SLS
import statsmodels.api as sm          # Standard API for add_constant
import statsmodels.formula.api as smf # Formula API for OLS
from scipy import stats
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, POLITICAL_COUNTY_PRES_FILE,
    WRI_EXCEL_FILE, TABLES_DIR, ensure_dirs_exist
)

# Ensure output directories exist
ensure_dirs_exist()

# --- CONFIGURATION ---
OUTPUT_RESULTS_FILE = TABLES_DIR / 'esb_spatial_results_final.txt'

# --- 1. DATA PREP ---
print("Loading Analysis Data...")
df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})

# Engineered features
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Load and merge political data (county-level)
print("Loading political data...")

# Load presidential election data
pres_df = pd.read_csv(str(POLITICAL_COUNTY_PRES_FILE))
pres_2020 = pres_df[(pres_df['year'] == 2020) & (pres_df['office'] == 'US PRESIDENT')].copy()

# Calculate county-level Democratic vote share
county_totals = pres_2020.groupby('county_fips')['candidatevotes'].sum().reset_index()
county_totals.columns = ['county_fips', 'total_votes']

county_dem = pres_2020[pres_2020['party'] == 'DEMOCRAT'].groupby('county_fips')['candidatevotes'].sum().reset_index()
county_dem.columns = ['county_fips', 'dem_votes']

county_political = county_totals.merge(county_dem, on='county_fips', how='left')
county_political['pct_dem_2020'] = county_political['dem_votes'] / county_political['total_votes']
county_political = county_political[['county_fips', 'pct_dem_2020']]
county_political['county_fips'] = county_political['county_fips'].astype(int)

print(f"  Loaded political data for {len(county_political)} counties")
print(f"  pct_dem_2020 range: {county_political['pct_dem_2020'].min():.3f} to {county_political['pct_dem_2020'].max():.3f}")

# Load LEA-to-county mapping from WRI data
lea_county = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='5. Counties')
lea_county = lea_county[['1c. LEA ID', '10b. County FIPS Code']].copy()
lea_county.columns = ['nces_id', 'county_fips']
lea_county['nces_id'] = lea_county['nces_id'].astype(str).str.split('.').str[0].str.zfill(7)

# Merge political data to LEA-county mapping
lea_county = lea_county.merge(county_political, on='county_fips', how='left')

# For LEAs spanning multiple counties, take population-weighted average (approximate with simple mean)
lea_political = lea_county.groupby('nces_id')['pct_dem_2020'].mean().reset_index()

print(f"  Mapped political data to {len(lea_political)} school districts")

# Merge political data into main dataframe
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
df = df.merge(lea_political, on='nces_id', how='left')
print(f"  Successfully merged political data: {df['pct_dem_2020'].notna().sum()} / {len(df)} districts have data")

# Create urbanicity dummy variables (control for rural/urban differences)
print("Creating urbanicity controls...")
if 'urbanicity' in df.columns:
    urbanicity_dummies = pd.get_dummies(df['urbanicity'], prefix='locale', drop_first=True)
    df = pd.concat([df, urbanicity_dummies], axis=1)
    locale_vars = [col for col in df.columns if col.startswith('locale_')]
    print(f"  Urbanicity categories: {df['urbanicity'].unique()}")
    print(f"  Created dummy variables: {locale_vars}")
else:
    print("  Warning: urbanicity variable not found")
    locale_vars = []

y_name = 'IS_ADOPTER'

# Version A: Without own lottery status (pure peer effects)
x_names_A = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pm25', 'pct_white', 'pct_dem_2020', 'is_priority'] + locale_vars

# Version B: With own lottery status (controlling for direct effect)
x_names_B = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pm25', 'pct_white', 'pct_dem_2020', 'is_priority', 'IV_Z'] + locale_vars

# KEEP 'state' for clustering standard errors later
print("Cleaning data for regression...")
all_vars_needed = list(set(x_names_A + x_names_B))  # Union of both variable sets
reg_df = df.dropna(subset=[y_name] + all_vars_needed + ['nces_id', 'state']).copy()
print(f"Regression Sample Size: {len(reg_df)}")

# --- 2. FAST SPATIAL MATRIX CONSTRUCTION ---
print("\nLoading Shapefile...")
gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
# Standardize IDs
if 'GEOID' not in gdf.columns:
    possible_cols = [c for c in gdf.columns if 'ID' in c]
    shp_id_col = possible_cols[0]
else:
    shp_id_col = 'GEOID'
gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.zfill(7)
reg_df['nces_id'] = reg_df['nces_id'].astype(str).str.zfill(7)

# MEMORY FIX: Merge first to reduce dataset before expensive CRS transformation
print("Merging to regression sample (reduces memory for transformation)...")
gdf = gdf.merge(reg_df[['nces_id']], left_on=shp_id_col, right_on='nces_id', how='inner')
print(f"  Reduced shapefile from full to {len(gdf)} matched districts")

# Project to Albers Equal Area for accurate distance calc
print("Projecting to EPSG:5070 (Albers Equal Area)...")
gdf = gdf.to_crs(epsg=5070)

# Merge with full regression data & Get Centroids
print("Merging full covariates and calculating centroids...")
gdf = gdf.drop(columns=['nces_id'])  # Drop merge key to avoid duplicate
geo_df = gdf.merge(reg_df, left_on=shp_id_col, right_on='nces_id', how='inner')
geo_df = geo_df.reset_index(drop=True) # CRITICAL for PySAL

# Use Centroids for KNN (Super Fast)
geo_df['centroid'] = geo_df.geometry.centroid
# Set geometry to centroid temporarily for PySAL
geo_df_points = geo_df.set_geometry('centroid')

print(f"Matched Sample: {len(geo_df)}")

# --- 2.1 W_geo (Emulation): Immediate Neighbors ---
print("Building W_geo (KNN-6)...")
w_geo = KNN.from_dataframe(geo_df_points, k=6)
w_geo.transform = 'r'

# --- 2.2 W_clim (Learning): Regional 'Donut' Peers ---
print("Building W_clim (KNN-20 minus KNN-6)...")
# Strategy: Build KNN-20, then set weights for the first 6 to 0
w_clim_base = KNN.from_dataframe(geo_df_points, k=20)

neighbors_clean = {}
weights_clean = {}

for idx in range(len(geo_df)):
    all_neighbors = w_clim_base.neighbors[idx]
    regional_peers = all_neighbors[6:] # Slice: Keep only neighbors 7-20
    neighbors_clean[idx] = regional_peers
    weights_clean[idx] = [1] * len(regional_peers)

w_clim = W(neighbors_clean, weights_clean)
w_clim.transform = 'r'
print(f"  > W_clim constructed. Mean neighbors: {w_clim.mean_neighbors:.2f}")


# --- 3. NAIVE OLS (Biased Baseline) ---
print("\n" + "="*80)
print("NAIVE OLS ESTIMATION")
print("="*80)
geo_df['w_geo_adoption'] = libpysal.weights.lag_spatial(w_geo, geo_df[y_name])
geo_df['w_clim_adoption'] = libpysal.weights.lag_spatial(w_clim, geo_df[y_name])

print("\nVersion A: Without Own Lottery Status Control")
print("-" * 60)
naive_formula_A = f"{y_name} ~ w_geo_adoption + w_clim_adoption + {' + '.join(x_names_A)} + C(state)"
naive_model_A = smf.ols(naive_formula_A, data=geo_df).fit()
print(naive_model_A.summary().tables[1])

print("\nVersion B: With Own Lottery Status Control")
print("-" * 60)
naive_formula_B = f"{y_name} ~ w_geo_adoption + w_clim_adoption + {' + '.join(x_names_B)} + C(state)"
naive_model_B = smf.ols(naive_formula_B, data=geo_df).fit()
print(naive_model_B.summary().tables[1])


# --- 4. PROPER 2SLS ESTIMATION ---
print("\n" + "="*80)
print("2SLS ESTIMATION (Correct Standard Errors)")
print("="*80)

# Construct Instruments (Spatial Lags of Z)
geo_df['w_geo_z'] = libpysal.weights.lag_spatial(w_geo, geo_df['IV_Z'])
geo_df['w_clim_z'] = libpysal.weights.lag_spatial(w_clim, geo_df['IV_Z'])

# Dependent Variable (Y)
dependent = geo_df[y_name]

# Endogenous Regressors (Peer Adoption)
endog = geo_df[['w_geo_adoption', 'w_clim_adoption']]

# Instruments (Peer Lottery Wins)
instruments = geo_df[['w_geo_z', 'w_clim_z']]

# Clusters (State level)
clusters = geo_df['state']

# Create state fixed effects dummies for 2SLS
print("Creating state fixed effects...")
state_dummies = pd.get_dummies(geo_df['state'], prefix='state', drop_first=True)
print(f"  Added {state_dummies.shape[1]} state fixed effects")

# VERSION A: Without Own Lottery Status
print("\nVersion A: Without Own Lottery Status Control (with State FE)")
print("-" * 60)
exog_A = pd.concat([geo_df[x_names_A], state_dummies], axis=1)
exog_A = sm.add_constant(exog_A)
model_iv_A = IV2SLS(dependent, exog_A, endog, instruments).fit(cov_type='clustered', clusters=clusters)
print(model_iv_A.summary)

# VERSION B: With Own Lottery Status
print("\nVersion B: With Own Lottery Status Control (with State FE)")
print("-" * 60)
exog_B = pd.concat([geo_df[x_names_B], state_dummies], axis=1)
exog_B = sm.add_constant(exog_B)
model_iv_B = IV2SLS(dependent, exog_B, endog, instruments).fit(cov_type='clustered', clusters=clusters)
print(model_iv_B.summary)


# --- 5. SAVE RESULTS ---
with open(str(OUTPUT_RESULTS_FILE), 'w') as f:
    f.write("ESB SPATIAL ANALYSIS RESULTS (FINAL)\n")
    f.write("====================================\n\n")
    
    f.write("="*80 + "\n")
    f.write("VERSION A: WITHOUT OWN LOTTERY STATUS CONTROL\n")
    f.write("="*80 + "\n\n")
    
    f.write("1A. NAIVE OLS (BIASED) - WITH STATE FE\n")
    f.write("-" * 30 + "\n")
    f.write(str(naive_model_A.summary()))
    f.write("\n\n")
    
    f.write("2A. 2SLS ESTIMATION (CORRECT INFERENCE) - WITH STATE FE & COUNTY POLITICAL CONTROLS\n")
    f.write("-" * 30 + "\n")
    f.write("Standard Errors: Clustered by State; State Fixed Effects + County Political Controls\n")
    f.write("Standard Errors: Clustered by State; State Fixed Effects Included\n")
    f.write(str(model_iv_A.summary))
    f.write("\n\n")
    
    f.write("DIAGNOSTICS (Version A)\n")
    f.write("-" * 30 + "\n")
    f.write(f"First Stage F-stat: {model_iv_A.first_stage.diagnostics['f.stat']['w_geo_adoption']:.2f} (Geo), ")
    f.write(f"{model_iv_A.first_stage.diagnostics['f.stat']['w_clim_adoption']:.2f} (Clim)\n")
    f.write("\n\n")
    
    f.write("="*80 + "\n")
    f.write("VERSION B: WITH OWN LOTTERY STATUS CONTROL\n")
    f.write("="*80 + "\n\n")
    
    f.write("1B. NAIVE OLS (BIASED) - WITH STATE FE\n")
    f.write("-" * 30 + "\n")
    f.write(str(naive_model_B.summary()))
    f.write("\n\n")
    
    f.write("2B. 2SLS ESTIMATION (CORRECT INFERENCE) - WITH STATE FE & COUNTY POLITICAL CONTROLS\n")
    f.write("-" * 30 + "\n")
    f.write("Standard Errors: Clustered by State; State Fixed Effects + County Political Controls\n")
    f.write(str(model_iv_B.summary))
    f.write("\n\n")
    
    f.write("DIAGNOSTICS (Version B)\n")
    f.write("-" * 30 + "\n")
    f.write(f"First Stage F-stat: {model_iv_B.first_stage.diagnostics['f.stat']['w_geo_adoption']:.2f} (Geo), ")
    f.write(f"{model_iv_B.first_stage.diagnostics['f.stat']['w_clim_adoption']:.2f} (Clim)\n")

print(f"\nResults saved to {OUTPUT_RESULTS_FILE}")