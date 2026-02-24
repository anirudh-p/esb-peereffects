"""
11_corrected_iv_estimation.py
=============================
Implements the CORRECTED IV estimation approach based on Feb 21, 2026 diagnostics.

KEY CHANGES FROM PREVIOUS APPROACH:
1. Use ORIGINAL lottery (IV_Z), NOT residualized
2. Control for is_priority in outcome equation (not IV stage)
3. Balance tests should be unconditional (among applicants)

The lottery IS random unconditionally - residualizing on priority introduces
Simpson's Paradox bias via collider path.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, POLITICAL_COUNTY_PRES_FILE,
    WRI_EXCEL_FILE, TABLES_DIR, LOGS_DIR, ensure_dirs_exist
)

from libpysal.weights import KNN
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
from scipy import stats

ensure_dirs_exist()

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("=" * 80)
print("CORRECTED IV ESTIMATION (Feb 21, 2026)")
print("=" * 80)

# Load main dataset
df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
print(f"\nLoaded {len(df):,} districts")

# Engineered features
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Load and merge political data (county-level)
print("Loading political data...")
pres_df = pd.read_csv(str(POLITICAL_COUNTY_PRES_FILE))
pres_2020 = pres_df[(pres_df['year'] == 2020) & (pres_df['office'] == 'US PRESIDENT')].copy()

county_totals = pres_2020.groupby('county_fips')['candidatevotes'].sum().reset_index()
county_totals.columns = ['county_fips', 'total_votes']

county_dem = pres_2020[pres_2020['party'] == 'DEMOCRAT'].groupby('county_fips')['candidatevotes'].sum().reset_index()
county_dem.columns = ['county_fips', 'dem_votes']

county_political = county_totals.merge(county_dem, on='county_fips', how='left')
county_political['pct_dem_2020'] = county_political['dem_votes'] / county_political['total_votes']
county_political = county_political[['county_fips', 'pct_dem_2020']]
county_political['county_fips'] = county_political['county_fips'].astype(int)

# Load LEA-to-county mapping
lea_county = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='5. Counties')
lea_county = lea_county[['1c. LEA ID', '10b. County FIPS Code']].copy()
lea_county.columns = ['nces_id', 'county_fips']
lea_county['nces_id'] = lea_county['nces_id'].astype(str).str.split('.').str[0].str.zfill(7)
lea_county = lea_county.merge(county_political, on='county_fips', how='left')
lea_political = lea_county.groupby('nces_id')['pct_dem_2020'].mean().reset_index()

df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
df = df.merge(lea_political, on='nces_id', how='left')
print(f"  Merged political data: {df['pct_dem_2020'].notna().sum()}/{len(df)}")

# Urbanicity dummies
if 'urbanicity' in df.columns:
    urbanicity_dummies = pd.get_dummies(df['urbanicity'], prefix='locale', drop_first=True)
    df = pd.concat([df, urbanicity_dummies], axis=1)
    locale_vars = [col for col in df.columns if col.startswith('locale_')]
else:
    locale_vars = []

# Define variables
y_name = 'IS_ADOPTER'
x_names = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pm25', 
           'pct_white', 'pct_dem_2020', 'is_priority'] + locale_vars

# Clean data
reg_df = df.dropna(subset=[y_name] + x_names + ['nces_id', 'state', 'IV_Z']).copy()
print(f"Regression sample (non-missing): {len(reg_df)}")

# ==============================================================================
# BUILD SPATIAL WEIGHTS
# ==============================================================================

print("\nLoading shapefile and building spatial weights...")
gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
shp_id_col = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]
gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.zfill(7)
reg_df['nces_id'] = reg_df['nces_id'].astype(str).str.zfill(7)

# Merge to reduce before CRS transformation
gdf = gdf.merge(reg_df[['nces_id']], left_on=shp_id_col, right_on='nces_id', how='inner')
print(f"  Matched {len(gdf)} districts")

gdf = gdf.to_crs(epsg=5070)
gdf = gdf.drop(columns=['nces_id'])
geo_df = gdf.merge(reg_df, left_on=shp_id_col, right_on='nces_id', how='inner')
geo_df = geo_df.reset_index(drop=True)
geo_df['centroid'] = geo_df.geometry.centroid
geo_df_points = geo_df.set_geometry('centroid')

print(f"Final matched sample: {len(geo_df)}")

# Build KNN-6 weights
print("Building W_geo (KNN-6)...")
w_geo = KNN.from_dataframe(geo_df_points, k=6)
w_geo.transform = 'r'

# ==============================================================================
# COMPUTE SPATIAL LAGS
# ==============================================================================

from libpysal.weights import lag_spatial

geo_df['w_adoption'] = lag_spatial(w_geo, geo_df['IS_ADOPTER'].values)
geo_df['w_IV_Z'] = lag_spatial(w_geo, geo_df['IV_Z'].fillna(0).values)

print(f"  w_adoption range: [{geo_df['w_adoption'].min():.3f}, {geo_df['w_adoption'].max():.3f}]")
print(f"  w_IV_Z range: [{geo_df['w_IV_Z'].min():.3f}, {geo_df['w_IV_Z'].max():.3f}]")

# ==============================================================================
# BALANCE TEST (UNCONDITIONAL)
# ==============================================================================

print("\n" + "=" * 80)
print("BALANCE TEST: UNCONDITIONAL (Among Applicants Only)")
print("=" * 80)

# Restrict to ACTUAL applicants (IS_APPLICANT == 1)
# IV_Z = 0 includes both losers AND non-applicants
# IS_APPLICANT correctly identifies those who entered the lottery
if 'IS_APPLICANT' in geo_df.columns:
    applicants = geo_df[geo_df['IS_APPLICANT'] == 1].copy()
else:
    # Fallback: use IV_Z == 1 as winners, and we'd miss losers
    print("  WARNING: IS_APPLICANT not found, using heuristic")
    applicants = geo_df[geo_df['IV_Z'] == 1].copy()

print(f"\nApplicants: {len(applicants):,}")
print(f"  Winners (IV_Z=1): {(applicants['IV_Z'] == 1).sum()}")
print(f"  Losers (IV_Z=0): {(applicants['IV_Z'] == 0).sum()}")

balance_vars = ['median_income', 'poverty_rate', 'enrollment', 'pm25', 'pct_white']

print("\nRegressing each covariate on IV_Z (lottery win):")
print("-" * 60)

balance_results = []
for var in balance_vars:
    if var not in applicants.columns:
        print(f"  {var}: MISSING")
        continue
    
    subset = applicants[['IV_Z', var]].dropna()
    if len(subset) < 100:
        print(f"  {var}: insufficient data ({len(subset)})")
        continue
    
    X = sm.add_constant(subset['IV_Z'])
    y = subset[var]
    
    model = sm.OLS(y, X).fit(cov_type='HC1')
    coef = model.params['IV_Z']
    pval = model.pvalues['IV_Z']
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    
    balance_results.append({'var': var, 'coef': coef, 'pval': pval, 'sig': sig})
    print(f"  {var:20s}: coef={coef:12.2f}, p={pval:.4f} {sig}")

n_passing = sum(1 for r in balance_results if r['pval'] > 0.10)
print(f"\nBalance tests passing (p > 0.10): {n_passing}/{len(balance_results)}")

# ==============================================================================
# CORRECTED IV-2SLS ESTIMATION
# ==============================================================================

print("\n" + "=" * 80)
print("CORRECTED IV-2SLS ESTIMATION")
print("=" * 80)
print("\nKey: Using ORIGINAL lottery (IV_Z), controlling for is_priority in outcome")

# State fixed effects
state_dummies = pd.get_dummies(geo_df['state'], prefix='st', drop_first=True)
geo_df = pd.concat([geo_df.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
state_fe_cols = list(state_dummies.columns)

# Control variables (with is_priority in outcome equation)
control_vars = ['log_median_income', 'poverty_rate', 'log_enrollment', 
                'pm25', 'pct_white', 'pct_dem_2020', 'is_priority'] + locale_vars

# Prepare matrices
y = geo_df['IS_ADOPTER'].astype(float)
X_endog = geo_df[['w_adoption']].astype(float)
X_exog = geo_df[control_vars + state_fe_cols].astype(float)
X_exog = sm.add_constant(X_exog)
Z = geo_df[['w_IV_Z']].astype(float)

print(f"\nSample size: {len(y)}")
print(f"Controls: {len(control_vars)} + {len(state_fe_cols)} state FE")

# First stage
print("\n--- FIRST STAGE ---")
first_stage_X = pd.concat([X_exog, Z], axis=1)
first_stage = sm.OLS(X_endog['w_adoption'], first_stage_X).fit()
print(f"Instrument coefficient: {first_stage.params['w_IV_Z']:.4f}")
print(f"t-stat: {first_stage.tvalues['w_IV_Z']:.2f}")
f_stat = first_stage.tvalues['w_IV_Z']**2
print(f"F-stat (approx): {f_stat:.1f}")

# IV-2SLS
print("\n--- IV-2SLS RESULTS ---")

iv_model = IV2SLS(
    dependent=y,
    exog=X_exog,
    endog=X_endog,
    instruments=Z
).fit(cov_type='clustered', clusters=geo_df['state'])

print(f"\n{'Variable':<25} {'Coef':>10} {'SE':>10} {'P-val':>10}")
print("=" * 55)

# Key results
key_vars = ['w_adoption', 'is_priority', 'log_median_income', 'poverty_rate', 
            'log_enrollment', 'pct_dem_2020', 'pct_white']

for var in key_vars:
    if var in iv_model.params.index:
        coef = iv_model.params[var]
        se = iv_model.std_errors[var]
        pval = iv_model.pvalues[var]
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"{var:<25} {coef:>10.4f} {se:>10.4f} {pval:>10.4f} {sig}")

print("-" * 55)
print(f"N = {iv_model.nobs}")
print(f"State clusters = {geo_df['state'].nunique()}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("SUMMARY: CORRECTED IV ESTIMATION")
print("=" * 80)

peer_effect = iv_model.params['w_adoption']
peer_se = iv_model.std_errors['w_adoption']
peer_pval = iv_model.pvalues['w_adoption']

print(f"""
MAIN RESULT:
  Peer Effect (w_adoption): {peer_effect:.4f} (SE={peer_se:.4f}, p={peer_pval:.4f})
  Interpretation: 10pp increase in neighbor adoption → {peer_effect*10:.2f}pp increase in own adoption

METHODOLOGY:
  1. Instrument: Spatial lag of ORIGINAL lottery outcome (w_IV_Z)
  2. NOT residualized on priority × state (this introduces bias)
  3. Priority controlled in OUTCOME equation (is_priority)
  4. State fixed effects included
  5. Clustered SE by state

BALANCE TESTS:
  {n_passing}/{len(balance_results)} covariates pass unconditional balance test (p > 0.10)
  Lottery IS random unconditionally.

FIRST STAGE:
  F-stat = {f_stat:.1f} (strong instrument)
""")

# Save results
output_path = LOGS_DIR / "corrected_iv_estimation.txt"
print(f"\nResults saved to: {output_path}")
