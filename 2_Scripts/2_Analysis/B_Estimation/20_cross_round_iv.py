"""
20_cross_round_iv.py
=====================
Cross-round IV estimation: R1 lottery wins as instrument for R3 adoption.

MOTIVATION:
  The baseline design (Scripts 11, 18) pools R1 and R3 lottery wins into a
  single cross-sectional IV. This conflates:
    (a) Cross-round effects: neighbor won R1 → buses operational → I apply R3
    (b) Within-round co-application: neighbor and I both applied R1 in same draw
  A genuine temporal instrument must be causally PRIOR to the outcome.

THIS SCRIPT:
  Outcome:    IS_R3_ADOPTER  = 1 if district adopted via R3 rebates (2023),
                               conditional on NOT having adopted in R1 (2022).
  Instrument: w_IV_Z_R1      = fraction of K nearest neighbors who won R1 lottery.
  Logic:  A neighbor's 2022 lottery win is fully predetermined relative to a
          district's 2023 adoption decision. The temporal gap (~12-18 months)
          allows buses to be delivered and observed before the R3 application.

DATA NOTES:
  - IS_ADOPTER in the main dataset is CSBP-federal only (CSB_Rebates + CSB_Grants).
    It does NOT capture ESBs bought via VW Settlement, CA HVIP, state programs, etc.
    (~33% of all WRI-tracked buses are non-federal). The LATE is program-specific.
  - Rebates 2022 (R1):  369 winner records (lottery)
  - Grants  2023 (R2):  241 winner records (competitive, no lottery)
  - Rebates 2023 (R3):  458 winner records (lottery)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

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
print("CROSS-ROUND IV: R1 WINS → R3 ADOPTION")
print("=" * 80)

# ==============================================================================
# 1. CONSTRUCT ROUND-SPECIFIC VARIABLES FROM RAW DATA
# ==============================================================================

print("\n--- Constructing round-specific adoption indicators ---")

# Load rebates (R1 = 2022, R3 = 2023)
rebates = pd.read_excel(RAW_WRI / "CSB_Rebates.xlsx")
rebates['nces_id'] = rebates['NCES District ID'].astype(str).str.split('.').str[0].str.zfill(7)
excluded = ['WITHDRAWN', 'CANCELLED', 'NOT SELECTED', 'INELIGIBLE', 'DENIED']
rebates = rebates[~rebates['Project Status'].str.upper().isin(excluded)]

r1_winners = rebates[rebates['Funding Year'] == 2022][['nces_id']].drop_duplicates()
r1_winners['IV_Z_R1'] = 1
r3_winners = rebates[rebates['Funding Year'] == 2023][['nces_id']].drop_duplicates()
r3_winners['IS_R3_ADOPTER'] = 1

print(f"R1 lottery winners: {len(r1_winners):,}")
print(f"R3 lottery winners: {len(r3_winners):,}")
print(f"R1∩R3 (adopted both rounds): {len(set(r1_winners.nces_id) & set(r3_winners.nces_id)):,}")

# ==============================================================================
# 2. LOAD MAIN DATASET AND CONSTRUCT CROSS-ROUND SAMPLE
# ==============================================================================

df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Merge round-specific outcomes
df = df.merge(r1_winners, on='nces_id', how='left')
df = df.merge(r3_winners, on='nces_id', how='left')
df['IV_Z_R1'] = df['IV_Z_R1'].fillna(0).astype(int)
df['IS_R3_ADOPTER'] = df['IS_R3_ADOPTER'].fillna(0).astype(int)

# Exclude R1 adopters from the outcome sample
# (they already have buses; R3 adoption would be additional/replacement, different decision)
df_r3 = df[df['IV_Z_R1'] == 0].copy()
print(f"\nSample excluding R1 adopters: {len(df_r3):,}")
print(f"  R3 adopters among them: {df_r3['IS_R3_ADOPTER'].sum():,}")
print(f"  R3 adoption rate: {df_r3['IS_R3_ADOPTER'].mean():.4f}")

# Load political data
pres_df   = pd.read_csv(str(POLITICAL_COUNTY_PRES_FILE))
pres_2020 = pres_df[(pres_df['year'] == 2020) & (pres_df['office'] == 'US PRESIDENT')].copy()
county_totals = pres_2020.groupby('county_fips')['candidatevotes'].sum().reset_index()
county_totals.columns = ['county_fips', 'total_votes']
county_dem = pres_2020[pres_2020['party'] == 'DEMOCRAT'].groupby('county_fips')['candidatevotes'].sum().reset_index()
county_dem.columns = ['county_fips', 'dem_votes']
county_pol = county_totals.merge(county_dem, on='county_fips', how='left')
county_pol['pct_dem_2020'] = county_pol['dem_votes'] / county_pol['total_votes']
county_pol['county_fips'] = county_pol['county_fips'].astype(int)
lea_county = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='5. Counties')
lea_county = lea_county[['1c. LEA ID', '10b. County FIPS Code']].copy()
lea_county.columns = ['nces_id', 'county_fips']
lea_county['nces_id'] = lea_county['nces_id'].astype(str).str.split('.').str[0].str.zfill(7)
lea_county = lea_county.merge(county_pol[['county_fips', 'pct_dem_2020']], on='county_fips', how='left')
lea_political = lea_county.groupby('nces_id')['pct_dem_2020'].mean().reset_index()
df_r3 = df_r3.merge(lea_political, on='nces_id', how='left')

if 'urbanicity' in df_r3.columns:
    locale_dummies = pd.get_dummies(df_r3['urbanicity'], prefix='locale', drop_first=True)
    df_r3 = pd.concat([df_r3.reset_index(drop=True), locale_dummies.reset_index(drop=True)], axis=1)
    locale_vars = [c for c in df_r3.columns if c.startswith('locale_')]
else:
    locale_vars = []

base_controls = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pm25',
                 'pct_white', 'pct_dem_2020', 'is_priority'] + locale_vars

all_req = ['IS_R3_ADOPTER'] + base_controls + ['nces_id', 'state']
reg_df = df_r3.dropna(subset=[v for v in all_req if v in df_r3.columns]).copy()
print(f"\nRegression sample (non-missing controls): {len(reg_df):,}")

# ==============================================================================
# 3. BUILD SPATIAL WEIGHTS ON FULL SPATIAL SAMPLE
# ==============================================================================

print("\nLoading shapefile...")
gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
shp_id = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]
gdf[shp_id] = gdf[shp_id].astype(str).str.zfill(7)

# Include ALL districts in the spatial weights (full network)
# but merge round-specific outcomes/instruments
full_df = df.merge(lea_political, on='nces_id', how='left')
if 'urbanicity' in full_df.columns:
    locale_dummies_f = pd.get_dummies(full_df['urbanicity'], prefix='locale', drop_first=True)
    full_df = pd.concat([full_df.reset_index(drop=True), locale_dummies_f.reset_index(drop=True)], axis=1)

full_df_clean = full_df.dropna(subset=[v for v in base_controls if v in full_df.columns]).copy()
full_df_clean['nces_id'] = full_df_clean['nces_id'].astype(str).str.zfill(7)

gdf_m = gdf.merge(full_df_clean[['nces_id']], left_on=shp_id, right_on='nces_id', how='inner')
gdf_m = gdf_m.to_crs(epsg=5070).drop(columns=['nces_id'])
geo_all = gdf_m.merge(full_df_clean, left_on=shp_id, right_on='nces_id', how='inner').reset_index(drop=True)
geo_all['centroid'] = geo_all.geometry.centroid
geo_pts = geo_all.set_geometry('centroid')
print(f"Full spatial sample for weight construction: {len(geo_all):,}")

# Build KNN-6 on full network
w6 = KNN.from_dataframe(geo_pts, k=6)
w6.transform = 'r'

# R1 winner spatial lag (the instrument)
geo_all['w_IV_Z_R1'] = lag_spatial(w6, geo_all['IV_Z_R1'].fillna(0).values)
# R3 adoption spatial lag (peer variable)
geo_all['w_R3_adoption'] = lag_spatial(w6, geo_all['IS_R3_ADOPTER'].fillna(0).values)

print(f"w_IV_Z_R1 range:    [{geo_all['w_IV_Z_R1'].min():.3f}, {geo_all['w_IV_Z_R1'].max():.3f}]")
print(f"w_R3_adoption range:[{geo_all['w_R3_adoption'].min():.3f}, {geo_all['w_R3_adoption'].max():.3f}]")

# Restrict to estimation sample (non-R1-adopters with non-missing controls)
est = geo_all[geo_all['IV_Z_R1'] == 0].copy().reset_index(drop=True)
est = est.dropna(subset=[v for v in base_controls if v in est.columns])
print(f"\nEstimation sample (non-R1 districts): {len(est):,}")

state_dummies = pd.get_dummies(est['state'], prefix='st', drop_first=True)
est = pd.concat([est.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
state_fe_cols = list(state_dummies.columns)

# ==============================================================================
# 4. FIRST STAGE
# ==============================================================================

print("\n" + "=" * 80)
print("FIRST STAGE: w_IV_Z_R1 → w_R3_adoption")
print("=" * 80)

y    = est['IS_R3_ADOPTER'].astype(float)
Xend = est[['w_R3_adoption']].astype(float)
Z    = est[['w_IV_Z_R1']].astype(float)

ctrl_cols = [c for c in base_controls if c in est.columns]
Xexo = est[ctrl_cols + state_fe_cols].astype(float)
Xexo = sm.add_constant(Xexo)
Xexo = Xexo.loc[:, Xexo.var(axis=0) > 0]

fs_X = pd.concat([Xexo, Z], axis=1)
fs = sm.OLS(Xend['w_R3_adoption'], fs_X).fit()
f_stat = fs.tvalues['w_IV_Z_R1'] ** 2
print(f"Coef on w_IV_Z_R1: {fs.params['w_IV_Z_R1']:.4f}")
print(f"t-stat:            {fs.tvalues['w_IV_Z_R1']:.2f}")
print(f"F-stat (approx):   {f_stat:.1f}")

if f_stat < 10:
    print("  WARNING: Weak instrument (F < 10). Interpret 2SLS with caution.")
elif f_stat < 100:
    print("  NOTE: Moderate instrument strength.")
else:
    print("  Strong instrument.")

# ==============================================================================
# 5. IV-2SLS
# ==============================================================================

print("\n" + "=" * 80)
print("IV-2SLS: CROSS-ROUND PEER EFFECT")
print("=" * 80)

try:
    iv = IV2SLS(y, Xexo, Xend, Z).fit(cov_type='clustered', clusters=est['state'])
    b   = iv.params['w_R3_adoption']
    se  = iv.std_errors['w_R3_adoption']
    pv  = iv.pvalues['w_R3_adoption']
    sig = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.1 else "(ns)"
    print(f"\nPeer Effect (w_R3_adoption): {b:.4f} (SE={se:.4f}, p={pv:.4f}) {sig}")
    print(f"N = {iv.nobs:,}, state clusters = {est['state'].nunique()}")
    print(f"\nInterpretation: A 10pp increase in share of R1-winning neighbors")
    print(f"raises own R3 adoption probability by {b*10:.2f}pp.")
except Exception as e:
    print(f"Error: {e}")
    b, se, pv = np.nan, np.nan, np.nan

# ==============================================================================
# 6. K SENSITIVITY ON CROSS-ROUND SPEC
# ==============================================================================

print("\n" + "=" * 80)
print("K SENSITIVITY — CROSS-ROUND SPEC")
print("=" * 80)

k_results = []
for k in [4, 6, 8, 10]:
    wk = KNN.from_dataframe(geo_pts, k=k)
    wk.transform = 'r'
    geo_all[f'w_Z_R1_k{k}']  = lag_spatial(wk, geo_all['IV_Z_R1'].fillna(0).values)
    geo_all[f'w_R3_k{k}']    = lag_spatial(wk, geo_all['IS_R3_ADOPTER'].fillna(0).values)

    est_k = geo_all[geo_all['IV_Z_R1'] == 0].copy().reset_index(drop=True)
    est_k = est_k.dropna(subset=[v for v in base_controls if v in est_k.columns])
    sd_k = pd.get_dummies(est_k['state'], prefix='st', drop_first=True)
    est_k = pd.concat([est_k.reset_index(drop=True), sd_k.reset_index(drop=True)], axis=1)
    sfe_k = list(sd_k.columns)

    Xend_k = est_k[[f'w_R3_k{k}']].astype(float)
    Z_k    = est_k[[f'w_Z_R1_k{k}']].astype(float)
    ctrl_k = [c for c in base_controls if c in est_k.columns]
    Xexo_k = est_k[ctrl_k + sfe_k].astype(float)
    Xexo_k = sm.add_constant(Xexo_k)
    Xexo_k = Xexo_k.loc[:, Xexo_k.var(axis=0) > 0]
    y_k    = est_k['IS_R3_ADOPTER'].astype(float)

    # First stage F
    fs_k = sm.OLS(Xend_k[f'w_R3_k{k}'], pd.concat([Xexo_k, Z_k], axis=1)).fit()
    f_k  = fs_k.tvalues[f'w_Z_R1_k{k}'] ** 2

    try:
        iv_k  = IV2SLS(y_k, Xexo_k, Xend_k, Z_k).fit(cov_type='clustered', clusters=est_k['state'])
        b_k   = iv_k.params[f'w_R3_k{k}']
        se_k_ = iv_k.std_errors[f'w_R3_k{k}']
        p_k   = iv_k.pvalues[f'w_R3_k{k}']
        sig_k = "***" if p_k < 0.01 else "**" if p_k < 0.05 else "*" if p_k < 0.1 else "(ns)"
        print(f"  K={k}: {b_k:.4f} (SE={se_k_:.4f}, p={p_k:.4f}) {sig_k} | F_1st={f_k:.0f}")
        k_results.append({'K': k, 'Peer_Effect': b_k, 'SE': se_k_, 'P_value': p_k, 'F_first': f_k})
    except Exception as e:
        print(f"  K={k}: Error: {e}")

# ==============================================================================
# 7. SUMMARY AND COMPARISON TABLE
# ==============================================================================

print("\n" + "=" * 80)
print("COMPARISON: CONTEMPORANEOUS vs. CROSS-ROUND IV")
print("=" * 80)

print(f"""
  Outcome: IS_R3_ADOPTER (Funding Year 2023 rebates) among non-R1 districts.
  Instrument: w_IV_Z_R1 (fraction of K=6 neighbors who won R1, 2022).

  ┌────────────────────────────────────┬──────────┬────────┬─────────┬────────┐
  │ Specification                      │ Peer Eff │   SE   │ p-value │ F_1st  │
  ├────────────────────────────────────┼──────────┼────────┼─────────┼────────┤
  │ Contemporaneous IV, full (Scr. 11) │ +0.1921  │ 0.0337 │ <0.001  │ 54,428 │
  │ Cross-round IV (R1→R3), K=6        │ {b:+.4f}  │ {se:.4f} │  {pv:.4f}  │ {f_stat:>6.0f} │
  └────────────────────────────────────┴──────────┴────────┴─────────┴────────┘

  The cross-round F-stat being lower than 54,428 would confirm that some of
  the instrument strength in the contemporaneous spec came from within-round
  co-application patterns, not pure temporal demonstration effects.
""")

# Save
out = pd.DataFrame(k_results)
out['Outcome'] = 'IS_R3_ADOPTER'
out['Instrument'] = 'w_IV_Z_R1'
out['Contemporaneous_ref'] = 0.1921
out.to_csv(TABLES_DIR / 'cross_round_iv_results.csv', index=False)
print(f"Results saved to: {TABLES_DIR / 'cross_round_iv_results.csv'}")
