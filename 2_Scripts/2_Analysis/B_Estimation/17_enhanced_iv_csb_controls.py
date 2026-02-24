"""
17_enhanced_iv_csb_controls.py
==============================
Add CSB (Clean School Bus) application controls from waitlist/rejected data.

Key insight: Districts that applied to competitive CSB grants showed "ESB interest"
even if they didn't win lottery. This creates better control group.

Controls:
1. applied_csb_competitive: District applied to competitive (non-lottery) round
2. csb_rejected: District was explicitly rejected
3. csb_waitlisted: District is on waitlist
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
RAW_WRI = DATA_DIR / "Raw" / "WRI"

print("=" * 80)
print("ENHANCED IV WITH CSB APPLICATION CONTROLS")
print("=" * 80)

# ==============================================================================
# 1. PROCESS CSB WAITLIST/REJECTED DATA
# ==============================================================================

print("\n--- Processing CSB Waitlist/Rejected Data ---")

csb_wait = pd.read_excel(RAW_WRI / "CSBP Applicants waitlisted and rejected_11.18.25.xlsx")
print(f"  Total CSB records: {len(csb_wait)}")

# Clean NCES ID
csb_wait['nces_id'] = csb_wait['NCES District ID'].astype(str).str.split('.').str[0].str.zfill(7)

# Create indicators
csb_wait['csb_rejected'] = (csb_wait['Project Status'] == 'Rejected').astype(int)
csb_wait['csb_waitlisted'] = (csb_wait['Project Status'] == 'Waitlist').astype(int)

# Round breakdown
print(f"  Rounds: {csb_wait['Round'].value_counts().to_dict()}")
print(f"  Status: {csb_wait['Project Status'].value_counts().to_dict()}")

# Convert buses to numeric (may have non-numeric values)
csb_wait['Total Number of Buses'] = pd.to_numeric(csb_wait['Total Number of Buses'], errors='coerce')

# Aggregate to district level (some districts appear multiple times)
csb_agg = csb_wait.groupby('nces_id').agg({
    'csb_rejected': 'max',
    'csb_waitlisted': 'max',
    'Total Number of Buses': 'sum'
}).reset_index()
csb_agg.columns = ['nces_id', 'csb_rejected', 'csb_waitlisted', 'csb_buses_requested']
csb_agg['applied_csb_competitive'] = 1

print(f"  Unique districts in CSB data: {len(csb_agg)}")

# ==============================================================================
# 2. PROCESS CSB GRANTS (Successful competitive)
# ==============================================================================

print("\n--- Processing CSB Grants (Successful Competitive) ---")

csb_grants = pd.read_excel(RAW_WRI / "CSB_Grants.xlsx")
print(f"  Total grant records: {len(csb_grants)}")

# This data has different structure - check columns
print(f"  Columns: {csb_grants.columns.tolist()[:10]}")

# Look for NCES ID or district name
if 'Grants School District Name' in csb_grants.columns:
    print(f"  Note: CSB Grants uses district names, not NCES IDs")
    print(f"  Will need fuzzy matching or skip for now")
    csb_winners = pd.DataFrame()  # Skip for now
else:
    csb_winners = pd.DataFrame()

# ==============================================================================
# 3. LOAD MAIN DATA AND MERGE
# ==============================================================================

print("\n--- Loading Main Dataset ---")

df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
print(f"Loaded {len(df):,} districts")

# Merge CSB application status
df = df.merge(csb_agg, on='nces_id', how='left')

# Fill missing = didn't apply to competitive
df['applied_csb_competitive'] = df['applied_csb_competitive'].fillna(0)
df['csb_rejected'] = df['csb_rejected'].fillna(0)
df['csb_waitlisted'] = df['csb_waitlisted'].fillna(0)

print(f"  Districts that applied to competitive CSB: {df['applied_csb_competitive'].sum():.0f}")
print(f"  Districts rejected: {df['csb_rejected'].sum():.0f}")
print(f"  Districts waitlisted: {df['csb_waitlisted'].sum():.0f}")

# Engineered features
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

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
# 4. BUILD REGRESSION SAMPLE
# ==============================================================================

y_name = 'IS_ADOPTER'
base_controls = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pm25', 
                 'pct_white', 'pct_dem_2020', 'is_priority'] + locale_vars
csb_controls = ['applied_csb_competitive']  # or ['csb_rejected', 'csb_waitlisted']

all_vars = [y_name] + base_controls + csb_controls + ['nces_id', 'state', 'IV_Z']
reg_df = df.dropna(subset=[v for v in all_vars if v in df.columns]).copy()
print(f"\nRegression sample: {len(reg_df):,}")

# ==============================================================================
# 5. BUILD SPATIAL WEIGHTS
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
# 6. IV-2SLS: BASELINE vs CSB-ENHANCED
# ==============================================================================

print("\n" + "=" * 80)
print("IV-2SLS: BASELINE vs CSB-ENHANCED")
print("=" * 80)

# State FE
state_dummies = pd.get_dummies(geo_df['state'], prefix='st', drop_first=True)
geo_df = pd.concat([geo_df.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
state_fe_cols = list(state_dummies.columns)

# Prepare matrices
y = geo_df['IS_ADOPTER'].astype(float)
X_endog = geo_df[['w_adoption']].astype(float)
Z = geo_df[['w_IV_Z']].astype(float)

# Model 1: Baseline
print("\n--- Model 1: Baseline (without CSB controls) ---")
X_base = geo_df[base_controls + state_fe_cols].astype(float)
X_base = sm.add_constant(X_base)

iv_base = IV2SLS(
    dependent=y,
    exog=X_base,
    endog=X_endog,
    instruments=Z
).fit(cov_type='clustered', clusters=geo_df['state'])

print(f"  Peer Effect: {iv_base.params['w_adoption']:.4f} (SE={iv_base.std_errors['w_adoption']:.4f}, p={iv_base.pvalues['w_adoption']:.4f})")

# Model 2: With applied_csb_competitive control
print("\n--- Model 2: + Applied to Competitive CSB ---")
X_csb = geo_df[base_controls + csb_controls + state_fe_cols].astype(float)
X_csb = sm.add_constant(X_csb)

iv_csb = IV2SLS(
    dependent=y,
    exog=X_csb,
    endog=X_endog,
    instruments=Z
).fit(cov_type='clustered', clusters=geo_df['state'])

print(f"  Peer Effect: {iv_csb.params['w_adoption']:.4f} (SE={iv_csb.std_errors['w_adoption']:.4f}, p={iv_csb.pvalues['w_adoption']:.4f})")
print(f"  applied_csb_competitive: {iv_csb.params['applied_csb_competitive']:.4f} (SE={iv_csb.std_errors['applied_csb_competitive']:.4f}, p={iv_csb.pvalues['applied_csb_competitive']:.4f})")

# Model 3: With rejected/waitlisted separately
print("\n--- Model 3: + Rejected/Waitlisted Separately ---")
X_detail = geo_df[base_controls + ['csb_rejected', 'csb_waitlisted'] + state_fe_cols].astype(float)
X_detail = sm.add_constant(X_detail)

iv_detail = IV2SLS(
    dependent=y,
    exog=X_detail,
    endog=X_endog,
    instruments=Z
).fit(cov_type='clustered', clusters=geo_df['state'])

print(f"  Peer Effect: {iv_detail.params['w_adoption']:.4f} (SE={iv_detail.std_errors['w_adoption']:.4f}, p={iv_detail.pvalues['w_adoption']:.4f})")
print(f"  csb_rejected: {iv_detail.params['csb_rejected']:.4f} (p={iv_detail.pvalues['csb_rejected']:.4f})")
print(f"  csb_waitlisted: {iv_detail.params['csb_waitlisted']:.4f} (p={iv_detail.pvalues['csb_waitlisted']:.4f})")

# ==============================================================================
# 7. DESCRIPTIVE: WHO APPLIES TO COMPETITIVE CSB?
# ==============================================================================

print("\n" + "=" * 80)
print("DESCRIPTIVE: CSB APPLICANT CHARACTERISTICS")
print("=" * 80)

applicants = geo_df[geo_df['applied_csb_competitive'] == 1]
non_applicants = geo_df[geo_df['applied_csb_competitive'] == 0]

print(f"\nCompetitive CSB Applicants: {len(applicants):,}")
print(f"Non-Applicants: {len(non_applicants):,}")

compare_vars = ['IS_ADOPTER', 'IV_Z', 'median_income', 'enrollment', 'poverty_rate', 'pct_white']
print(f"\n{'Variable':<20} {'Applicants':>12} {'Non-App':>12} {'Diff':>12}")
print("-" * 60)
for var in compare_vars:
    app_mean = applicants[var].mean()
    non_mean = non_applicants[var].mean()
    diff = app_mean - non_mean
    print(f"{var:<20} {app_mean:12.4f} {non_mean:12.4f} {diff:12.4f}")

# ==============================================================================
# 8. SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("SUMMARY: CSB CONTROL ROBUSTNESS")
print("=" * 80)

results = pd.DataFrame({
    'Model': ['Baseline', 'With CSB Applied', 'With Rejected/Waitlisted'],
    'Peer_Effect': [iv_base.params['w_adoption'], 
                    iv_csb.params['w_adoption'],
                    iv_detail.params['w_adoption']],
    'SE': [iv_base.std_errors['w_adoption'],
           iv_csb.std_errors['w_adoption'],
           iv_detail.std_errors['w_adoption']],
    'P_value': [iv_base.pvalues['w_adoption'],
                iv_csb.pvalues['w_adoption'],
                iv_detail.pvalues['w_adoption']],
    'N': [len(geo_df)] * 3
})

print(results.to_string(index=False))

# Save results
results.to_csv(TABLES_DIR / "iv_csb_controls_robustness.csv", index=False)
print(f"\nResults saved to: {TABLES_DIR / 'iv_csb_controls_robustness.csv'}")

# Interpretation
print("\n" + "-" * 60)
print("INTERPRETATION:")
base_pe = iv_base.params['w_adoption']
csb_pe = iv_csb.params['w_adoption']
diff = abs(base_pe - csb_pe)
print(f"  Baseline peer effect: {base_pe:.4f}")
print(f"  With CSB control: {csb_pe:.4f}")
print(f"  Difference: {diff:.4f}")

csb_coef = iv_csb.params['applied_csb_competitive']
csb_pval = iv_csb.pvalues['applied_csb_competitive']
print(f"\n  CSB application effect: {csb_coef:.4f} (p={csb_pval:.4f})")

if csb_pval < 0.05:
    print("  → Districts that applied to competitive CSB have DIFFERENT adoption rates")
    print("    This controls for 'ESB interest' that lottery might not capture")
else:
    print("  → CSB application doesn't independently predict adoption")
    print("    Lottery applicants may be similar to competitive applicants")
