"""
16_enhanced_iv_state_controls.py
=================================
Add state-level controls from new datasets:
1. EV charging station density
2. Diesel prices
3. State ESB incentive count

Test whether peer effects remain robust with these controls.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, POLITICAL_COUNTY_PRES_FILE,
    WRI_EXCEL_FILE, TABLES_DIR, LOGS_DIR, ensure_dirs_exist
)

from libpysal.weights import KNN, lag_spatial
from linearmodels.iv import IV2SLS
import statsmodels.api as sm

ensure_dirs_exist()

DATA_DIR = Path(__file__).parent.parent.parent.parent / "1_Data"
NEW_DATA = DATA_DIR / "New Datasets"

print("=" * 80)
print("ENHANCED IV WITH STATE-LEVEL CONTROLS")
print("=" * 80)

# ==============================================================================
# 1. PROCESS EV CHARGING STATIONS (State-level)
# ==============================================================================

print("\n--- Processing EV Charging Station Data ---")

xls = pd.ExcelFile(NEW_DATA / "historical-station-counts.xlsx")
df_2023 = pd.read_excel(xls, sheet_name='2023', header=None, skiprows=3)

# Structure: Row 0=Alabama, Row 1=Alabama electric breakdown, Row 2=Alaska, etc.
# Keep only state rows (even indices in 0-indexed after skip)
df_2023.columns = ['State', 'Biodiesel', 'CNG', 'E85', 'Electric', 
                   'Hydrogen', 'LNG', 'Propane', 'Renewable_Diesel', 'Total']

# Every other row has state name (rows 0, 2, 4, ...)
ev_stations = df_2023[df_2023['State'].notna()].copy()

# Parse Electric column: format is "stations | outlets" like "424 | 1,096"
def parse_stations(x):
    if pd.isna(x):
        return np.nan, np.nan
    parts = str(x).split('|')
    if len(parts) >= 2:
        stations = parts[0].replace(',', '').strip()
        outlets = parts[1].replace(',', '').strip()
        try:
            return float(stations), float(outlets)
        except:
            return np.nan, np.nan
    return np.nan, np.nan

ev_stations[['ev_stations', 'ev_outlets']] = ev_stations['Electric'].apply(
    lambda x: pd.Series(parse_stations(x))
)

ev_stations = ev_stations[['State', 'ev_stations', 'ev_outlets']].copy()

# State abbreviation mapping
state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}

ev_stations['state'] = ev_stations['State'].map(state_abbrev)
ev_stations = ev_stations.dropna(subset=['state'])
# ev_stations and ev_outlets already created from parsing
ev_stations = ev_stations[['state', 'ev_stations', 'ev_outlets']]

print(f"  EV stations by state: {len(ev_stations)} states")
print(f"  Range: {ev_stations['ev_stations'].min():.0f} - {ev_stations['ev_stations'].max():.0f}")

# ==============================================================================
# 2. PROCESS DIESEL PRICES (State-level)
# ==============================================================================

print("\n--- Processing Diesel Price Data ---")

fuel = pd.read_csv(NEW_DATA / "pr_all.csv")

# Find distillate fuel (diesel) - transportation sector
# DFTCD = Distillate Fuel Transportation price
diesel = fuel[fuel['MSN'] == 'DFTCD'][['State', '2022', '2023']].copy()
diesel.columns = ['state', 'diesel_price_2022', 'diesel_price_2023']
diesel = diesel[diesel['state'].notna() & (diesel['state'] != 'US')]
diesel['diesel_price_2022'] = pd.to_numeric(diesel['diesel_price_2022'], errors='coerce')
diesel['diesel_price_2023'] = pd.to_numeric(diesel['diesel_price_2023'], errors='coerce')

print(f"  Diesel prices by state: {len(diesel)} states")
print(f"  2022 range: ${diesel['diesel_price_2022'].min():.2f} - ${diesel['diesel_price_2022'].max():.2f}")

# ==============================================================================
# 3. PROCESS STATE INCENTIVE COUNTS
# ==============================================================================

print("\n--- Processing State Incentive Data ---")

clear = pd.read_excel(NEW_DATA / "Clearinghouse of Electric School Bus Funding and Financing Opportunities (January 2024) (1).xlsx",
                      sheet_name='Clearinghouse Official')

# Count programs by state
incentives = clear.groupby('Geography').size().reset_index(name='n_incentives')
incentives.columns = ['state', 'n_incentives']

# Filter to US states (2-letter codes)
incentives = incentives[incentives['state'].str.len() == 2]

print(f"  States with incentives: {len(incentives)}")
print(f"  Incentive range: {incentives['n_incentives'].min()} - {incentives['n_incentives'].max()}")

# ==============================================================================
# 4. MERGE STATE CONTROLS
# ==============================================================================

print("\n--- Merging State Controls ---")

# Combine all state-level controls
state_controls = ev_stations.merge(diesel, on='state', how='outer')
state_controls = state_controls.merge(incentives, on='state', how='outer')

# Fill missing incentives with 0 (state has no specific programs)
state_controls['n_incentives'] = state_controls['n_incentives'].fillna(0)

print(f"  Combined state controls: {len(state_controls)} states")
print(state_controls.head())

# ==============================================================================
# 5. LOAD MAIN DATA AND MERGE
# ==============================================================================

print("\n--- Loading Main Dataset ---")

df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
print(f"Loaded {len(df):,} districts")

# Engineered features
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Merge state controls
df = df.merge(state_controls, on='state', how='left')
print(f"  Merged state controls: {df['ev_stations'].notna().sum()}/{len(df)}")

# Normalize state controls (for coefficient interpretability)
df['ev_stations_log'] = np.log(df['ev_stations'] + 1)
df['diesel_price'] = df['diesel_price_2022']  # Use 2022 as it predates most adoption
df['log_incentives'] = np.log(df['n_incentives'] + 1)

# Load political data
print("\nLoading political data...")
pres_df = pd.read_csv(str(POLITICAL_COUNTY_PRES_FILE))
pres_2020 = pres_df[(pres_df['year'] == 2020) & (pres_df['office'] == 'US PRESIDENT')].copy()
county_totals = pres_2020.groupby('county_fips')['candidatevotes'].sum().reset_index()
county_totals.columns = ['county_fips', 'total_votes']
county_dem = pres_2020[pres_2020['party'] == 'DEMOCRAT'].groupby('county_fips')['candidatevotes'].sum().reset_index()
county_dem.columns = ['county_fips', 'dem_votes']
county_political = county_totals.merge(county_dem, on='county_fips', how='left')
county_political['pct_dem_2020'] = county_political['dem_votes'] / county_political['total_votes']

lea_county = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='5. Counties')
lea_county = lea_county[['1c. LEA ID', '10b. County FIPS Code']].copy()
lea_county.columns = ['nces_id', 'county_fips']
lea_county['nces_id'] = lea_county['nces_id'].astype(str).str.split('.').str[0].str.zfill(7)
lea_county['county_fips'] = lea_county['county_fips'].astype(int)
lea_county = lea_county.merge(county_political[['county_fips', 'pct_dem_2020']], on='county_fips', how='left')
lea_political = lea_county.groupby('nces_id')['pct_dem_2020'].mean().reset_index()
df = df.merge(lea_political, on='nces_id', how='left')

# Urbanicity dummies
if 'urbanicity' in df.columns:
    urbanicity_dummies = pd.get_dummies(df['urbanicity'], prefix='locale', drop_first=True)
    df = pd.concat([df, urbanicity_dummies], axis=1)
    locale_vars = [col for col in df.columns if col.startswith('locale_')]
else:
    locale_vars = []

# ==============================================================================
# 6. BUILD REGRESSION SAMPLE
# ==============================================================================

# Define variables - now including new state controls
y_name = 'IS_ADOPTER'
base_controls = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pm25', 
                 'pct_white', 'pct_dem_2020', 'is_priority'] + locale_vars
new_controls = ['ev_stations_log', 'diesel_price', 'log_incentives']

# Regression sample
all_vars = [y_name] + base_controls + new_controls + ['nces_id', 'state', 'IV_Z']
reg_df = df.dropna(subset=[v for v in all_vars if v in df.columns]).copy()
print(f"\nRegression sample: {len(reg_df):,}")

# ==============================================================================
# 7. BUILD SPATIAL WEIGHTS
# ==============================================================================

print("\nBuilding spatial weights...")
gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
shp_id_col = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]
gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.zfill(7)

gdf = gdf.merge(reg_df[['nces_id']], left_on=shp_id_col, right_on='nces_id', how='inner')
gdf = gdf.to_crs(epsg=5070)
gdf = gdf.drop(columns=['nces_id'])
geo_df = gdf.merge(reg_df, left_on=shp_id_col, right_on='nces_id', how='inner')
geo_df = geo_df.reset_index(drop=True)
geo_df['centroid'] = geo_df.geometry.centroid
geo_df_points = geo_df.set_geometry('centroid')

print(f"  Matched sample: {len(geo_df):,}")

w_geo = KNN.from_dataframe(geo_df_points, k=6)
w_geo.transform = 'r'

geo_df['w_adoption'] = lag_spatial(w_geo, geo_df['IS_ADOPTER'].values)
geo_df['w_IV_Z'] = lag_spatial(w_geo, geo_df['IV_Z'].fillna(0).values)

# ==============================================================================
# 8. IV-2SLS ESTIMATION: BASELINE vs ENHANCED
# ==============================================================================

print("\n" + "=" * 80)
print("IV-2SLS: BASELINE vs ENHANCED (with state controls)")
print("=" * 80)

# State FE
state_dummies = pd.get_dummies(geo_df['state'], prefix='st', drop_first=True)
geo_df = pd.concat([geo_df.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
state_fe_cols = list(state_dummies.columns)

# Prepare matrices
y = geo_df['IS_ADOPTER'].astype(float)
X_endog = geo_df[['w_adoption']].astype(float)
Z = geo_df[['w_IV_Z']].astype(float)

# Model 1: Baseline (without new state controls)
print("\n--- Model 1: Baseline (original controls + state FE) ---")
X_base = geo_df[base_controls + state_fe_cols].astype(float)
X_base = sm.add_constant(X_base)

iv_base = IV2SLS(
    dependent=y,
    exog=X_base,
    endog=X_endog,
    instruments=Z
).fit(cov_type='clustered', clusters=geo_df['state'])

print(f"  Peer Effect: {iv_base.params['w_adoption']:.4f} (SE={iv_base.std_errors['w_adoption']:.4f}, p={iv_base.pvalues['w_adoption']:.4f})")

# Model 2: Enhanced (with new state controls, drop state FE to avoid collinearity)
print("\n--- Model 2: Enhanced (+ EV stations, diesel price, incentives) ---")
X_enhanced = geo_df[base_controls + new_controls].astype(float)
X_enhanced = sm.add_constant(X_enhanced)

iv_enhanced_no_fe = IV2SLS(
    dependent=y,
    exog=X_enhanced,
    endog=X_endog,
    instruments=Z
).fit(cov_type='clustered', clusters=geo_df['state'])

print(f"  Peer Effect: {iv_enhanced_no_fe.params['w_adoption']:.4f} (SE={iv_enhanced_no_fe.std_errors['w_adoption']:.4f}, p={iv_enhanced_no_fe.pvalues['w_adoption']:.4f})")

# Check new control coefficients
print("\n  New control coefficients:")
for var in new_controls:
    coef = iv_enhanced_no_fe.params[var]
    se = iv_enhanced_no_fe.std_errors[var]
    pval = iv_enhanced_no_fe.pvalues[var]
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    print(f"    {var:20s}: {coef:8.4f} (SE={se:.4f}) {sig}")

# Model 3: Enhanced WITH state FE (most conservative)
# Note: State-level controls + State FE can cause collinearity
print("\n--- Model 3: Enhanced + State FE (most conservative) ---")
try:
    X_full = geo_df[base_controls + new_controls + state_fe_cols].astype(float)
    X_full = sm.add_constant(X_full)

    iv_full = IV2SLS(
        dependent=y,
        exog=X_full,
        endog=X_endog,
        instruments=Z
    ).fit(cov_type='clustered', clusters=geo_df['state'])

    print(f"  Peer Effect: {iv_full.params['w_adoption']:.4f} (SE={iv_full.std_errors['w_adoption']:.4f}, p={iv_full.pvalues['w_adoption']:.4f})")
    full_peer = iv_full.params['w_adoption']
    full_se = iv_full.std_errors['w_adoption']
    full_pval = iv_full.pvalues['w_adoption']
except ValueError as e:
    print(f"  Collinearity detected (state FE + state controls)")
    print(f"  This is expected: state-level vars are absorbed by state FE")
    full_peer = np.nan
    full_se = np.nan
    full_pval = np.nan

# ==============================================================================
# 9. SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("SUMMARY: STATE CONTROL ROBUSTNESS")
print("=" * 80)

results = pd.DataFrame({
    'Model': ['Baseline (State FE)', 'Enhanced (No FE)', 'Enhanced + State FE'],
    'Peer_Effect': [iv_base.params['w_adoption'], 
                    iv_enhanced_no_fe.params['w_adoption'],
                    full_peer],
    'SE': [iv_base.std_errors['w_adoption'],
           iv_enhanced_no_fe.std_errors['w_adoption'],
           full_se],
    'P_value': [iv_base.pvalues['w_adoption'],
                iv_enhanced_no_fe.pvalues['w_adoption'],
                full_pval],
    'N': [len(geo_df)] * 3
})

print(results.to_string(index=False))

# Save results
results.to_csv(TABLES_DIR / "iv_state_controls_robustness.csv", index=False)
print(f"\nResults saved to: {TABLES_DIR / 'iv_state_controls_robustness.csv'}")

# Interpretation
print("\n" + "-" * 60)
print("INTERPRETATION:")
base_pe = iv_base.params['w_adoption']
enhanced_pe = iv_enhanced_no_fe.params['w_adoption']
diff = abs(base_pe - enhanced_pe)
print(f"  Baseline (State FE): {base_pe:.4f}")
print(f"  Enhanced (Observable State Controls): {enhanced_pe:.4f}")
print(f"  Difference: {diff:.4f}")

if diff < 0.05:
    print("\n  ✓ Peer effect is ROBUST to state-level controls")
    print("    Replacing state FE with EV infrastructure, diesel prices, and incentives")
    print("    does not substantially change the estimate.")
else:
    print("\n  ⚠ Peer effect CHANGES with state controls")
    print("    Enhanced model shows different peer effect magnitude.")
    
print("\n  Note: State-level controls explain what state FE absorbs,")
print("        making the model more interpretable for policy analysis.")
