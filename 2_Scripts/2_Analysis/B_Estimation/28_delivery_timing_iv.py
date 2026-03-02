"""
28_delivery_timing_iv.py
=========================
Test whether peer effects differ by R1 deployment timing.

HYPOTHESIS:
  - Early-delivery R1 neighbors signal "ESB adoption works" → positive peer effect
  - Late-delivery R1 neighbors signal "ESB is risky/difficult" → negative peer effect

DESIGN:
  1. Split R1 winners into early vs late delivery (relative to R3 deadline)
  2. Use w_R1_early and w_R1_late as separate instruments
  3. Estimate IV with each instrument separately and jointly

Outputs:
  - 3_Output/Tables/delivery_timing_iv.csv
  - 3_Output/Logs/delivery_timing_iv.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, WRI_EXCEL_FILE,
    TABLES_DIR, LOGS_DIR, ensure_dirs_exist, RAW_WRI_DIR
)

from libpysal.weights import KNN, lag_spatial
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
import geopandas as gpd

ensure_dirs_exist()

log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(msg)

def clean_lea_id(series):
    return series.astype(str).str.split('.').str[0].str.zfill(7)

def extract_quarter_date(series):
    def parse_q(val):
        if pd.isna(val):
            return pd.NaT
        s = str(val)
        m = re.search(r'Q(\d)\s*(\d{4})', s)
        if m:
            q, y = int(m.group(1)), int(m.group(2))
            month = {1: 1, 2: 4, 3: 7, 4: 10}.get(q, 1)
            return pd.Timestamp(year=y, month=month, day=1)
        m2 = re.search(r'(\d{4})', s)
        if m2:
            return pd.Timestamp(year=int(m2.group(1)), month=7, day=1)
        return pd.NaT
    return series.apply(parse_q)

def significance_stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

log('=' * 80)
log('DELIVERY TIMING IV: EARLY vs LATE R1 DEPLOYMENT')
log('Testing whether peer effect sign depends on deployment success signal')
log('=' * 80)

# ==============================================================================
# 1. LOAD DATA AND CONSTRUCT TIMING VARIABLES
# ==============================================================================

log('\n--- Loading WRI bus-level data ---')
df_bus = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='2. Bus-level data')
df_bus['nces_id'] = clean_lea_id(df_bus['1c. LEA ID'])
df_bus['award_date'] = extract_quarter_date(df_bus['3p. Quarter awarded'])
df_bus['delivery_date'] = extract_quarter_date(df_bus['3r. Quarter delivered'])

# R1 CSBP buses (awarded 2022)
r1_buses = df_bus[
    (df_bus['award_date'].dt.year == 2022) & 
    (df_bus['3z. Funding source 1'].str.contains('CLEAN SCHOOL BUS', na=False, case=False))
].copy()

log(f'R1 CSBP buses: {len(r1_buses):,}')

# R3 deadline = October 2023
r3_deadline = pd.Timestamp('2023-10-01')

# District-level first delivery
r1_district = r1_buses.groupby('nces_id').agg({
    'delivery_date': 'min'
}).reset_index()
r1_district.columns = ['nces_id', 'first_delivery']

# Early = delivered before R3 deadline
r1_district['early_delivery'] = (r1_district['first_delivery'] < r3_deadline).fillna(False).astype(int)
# Late = delivery date missing OR after R3 deadline
r1_district['late_delivery'] = (~(r1_district['first_delivery'] < r3_deadline)).astype(int)

n_early = r1_district['early_delivery'].sum()
n_late = r1_district['late_delivery'].sum()
log(f'R1 districts with early delivery (before Oct 2023): {n_early}')
log(f'R1 districts with late/missing delivery: {n_late}')

# ==============================================================================
# 2. LOAD REBATES AND MERGE
# ==============================================================================

log('\n--- Loading rebate data ---')
rebates = pd.read_excel(RAW_WRI_DIR / "CSB_Rebates.xlsx")
rebates['nces_id'] = clean_lea_id(rebates['NCES District ID'])
excluded = ['WITHDRAWN', 'CANCELLED', 'NOT SELECTED', 'INELIGIBLE', 'DENIED']
rebates = rebates[~rebates['Project Status'].str.upper().isin(excluded)]

r1_winners = rebates[rebates['Funding Year'] == 2022][['nces_id']].drop_duplicates()
r1_winners['IV_Z_R1'] = 1
r3_winners = rebates[rebates['Funding Year'] == 2023][['nces_id']].drop_duplicates()
r3_winners['IS_R3_WINNER'] = 1

log(f'R1 lottery winners: {len(r1_winners):,}')
log(f'R3 lottery winners: {len(r3_winners):,}')

# Merge delivery timing into R1 winners
r1_winners = r1_winners.merge(r1_district[['nces_id', 'early_delivery', 'late_delivery']], 
                               on='nces_id', how='left')
r1_winners['early_delivery'] = r1_winners['early_delivery'].fillna(0).astype(int)
r1_winners['late_delivery'] = r1_winners['late_delivery'].fillna(1).astype(int)  # Missing = late

# Create separate treatment indicators
r1_winners['R1_early'] = r1_winners['IV_Z_R1'] * r1_winners['early_delivery']
r1_winners['R1_late'] = r1_winners['IV_Z_R1'] * r1_winners['late_delivery']

log(f'R1 winners with early delivery: {r1_winners["R1_early"].sum()}')
log(f'R1 winners with late delivery: {r1_winners["R1_late"].sum()}')

# ==============================================================================
# 3. LOAD MAIN DATASET AND BUILD SPATIAL WEIGHTS
# ==============================================================================

log('\n--- Loading main dataset ---')
df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Merge R1/R3 status
df = df.merge(r1_winners[['nces_id', 'IV_Z_R1', 'R1_early', 'R1_late']], on='nces_id', how='left')
df = df.merge(r3_winners, on='nces_id', how='left')
for c in ['IV_Z_R1', 'R1_early', 'R1_late', 'IS_R3_WINNER']:
    df[c] = df[c].fillna(0).astype(int)

log(f'Total districts: {len(df):,}')

# Build spatial weights
log('\n--- Building spatial weights ---')
gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
shp_id = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]
gdf[shp_id] = gdf[shp_id].astype(str).str.zfill(7)

gdf_m = gdf.merge(df[['nces_id']].drop_duplicates(), left_on=shp_id, right_on='nces_id', how='inner')
gdf_m = gdf_m.to_crs(epsg=5070).drop(columns=['nces_id'])
geo_all = gdf_m.merge(df, left_on=shp_id, right_on='nces_id', how='inner').reset_index(drop=True)
geo_all['centroid'] = geo_all.geometry.centroid
geo_pts = geo_all.set_geometry('centroid')

log(f'Spatial sample: {len(geo_all):,}')

results = []

for K in [6]:  # Focus on K=6
    log(f'\n{"="*80}')
    log(f'K = {K}')
    log(f'{"="*80}')
    
    w = KNN.from_dataframe(geo_pts, k=K)
    w.transform = 'r'
    
    # Spatial lags
    geo_all['w_IV_Z_R1'] = lag_spatial(w, geo_all['IV_Z_R1'].values)
    geo_all['w_R1_early'] = lag_spatial(w, geo_all['R1_early'].values)
    geo_all['w_R1_late'] = lag_spatial(w, geo_all['R1_late'].values)
    geo_all['w_R3_WINNER'] = lag_spatial(w, geo_all['IS_R3_WINNER'].values)
    
    # Estimation sample: non-R1 winners
    df_est = geo_all[geo_all['IV_Z_R1'] == 0].copy().reset_index(drop=True)
    
    # Controls
    locale_vars = []
    if 'urbanicity' in df_est.columns:
        locale_dummies = pd.get_dummies(df_est['urbanicity'], prefix='locale', drop_first=True)
        df_est = pd.concat([df_est.reset_index(drop=True), locale_dummies.reset_index(drop=True)], axis=1)
        locale_vars = list(locale_dummies.columns)
    
    base_controls = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pm25',
                     'pct_white', 'is_priority', 'IS_APPLICANT'] + locale_vars
    
    state_dummies = pd.get_dummies(df_est['state'], prefix='st', drop_first=True)
    df_est = pd.concat([df_est.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
    state_fe_cols = list(state_dummies.columns)
    
    ctrl_cols = [c for c in base_controls if c in df_est.columns]
    df_est = df_est.dropna(subset=ctrl_cols + ['state']).copy()
    
    log(f'\nEstimation sample: {len(df_est):,}')
    log(f'Mean(w_R1_early): {df_est["w_R1_early"].mean():.5f}')
    log(f'Mean(w_R1_late): {df_est["w_R1_late"].mean():.5f}')
    log(f'R3 winner rate: {df_est["IS_R3_WINNER"].mean():.4f}')
    
    # -------------------------------------------------------------------------
    # TEST 1: Reduced Form - Early vs Late separately
    # -------------------------------------------------------------------------
    log('\n--- REDUCED FORM: R1 neighbor delivery timing → R3 adoption ---')
    
    y = df_est['IS_R3_WINNER'].astype(float)
    
    # Early only
    X_early = sm.add_constant(df_est[ctrl_cols + state_fe_cols + ['w_R1_early']].astype(float))
    model_early = sm.OLS(y, X_early).fit(cov_type='HC1')
    log(f'\n  Early-delivery R1 neighbors only:')
    log(f'    coef = {model_early.params["w_R1_early"]:.4f} '
        f'(SE = {model_early.bse["w_R1_early"]:.4f}) '
        f'p = {model_early.pvalues["w_R1_early"]:.4f} '
        f'{significance_stars(model_early.pvalues["w_R1_early"])}')
    
    results.append({
        'test': 'RF: w_R1_early → R3_win',
        'K': K, 'coef': model_early.params["w_R1_early"],
        'se': model_early.bse["w_R1_early"],
        'pval': model_early.pvalues["w_R1_early"],
        'stars': significance_stars(model_early.pvalues["w_R1_early"]),
        'N': len(df_est)
    })
    
    # Late only
    X_late = sm.add_constant(df_est[ctrl_cols + state_fe_cols + ['w_R1_late']].astype(float))
    model_late = sm.OLS(y, X_late).fit(cov_type='HC1')
    log(f'\n  Late-delivery R1 neighbors only:')
    log(f'    coef = {model_late.params["w_R1_late"]:.4f} '
        f'(SE = {model_late.bse["w_R1_late"]:.4f}) '
        f'p = {model_late.pvalues["w_R1_late"]:.4f} '
        f'{significance_stars(model_late.pvalues["w_R1_late"])}')
    
    results.append({
        'test': 'RF: w_R1_late → R3_win',
        'K': K, 'coef': model_late.params["w_R1_late"],
        'se': model_late.bse["w_R1_late"],
        'pval': model_late.pvalues["w_R1_late"],
        'stars': significance_stars(model_late.pvalues["w_R1_late"]),
        'N': len(df_est)
    })
    
    # Both in same regression (horse race)
    X_both = sm.add_constant(df_est[ctrl_cols + state_fe_cols + ['w_R1_early', 'w_R1_late']].astype(float))
    model_both = sm.OLS(y, X_both).fit(cov_type='HC1')
    log(f'\n  Both in same regression (horse race):')
    log(f'    w_R1_early: coef = {model_both.params["w_R1_early"]:.4f} '
        f'(SE = {model_both.bse["w_R1_early"]:.4f}) '
        f'p = {model_both.pvalues["w_R1_early"]:.4f} '
        f'{significance_stars(model_both.pvalues["w_R1_early"])}')
    log(f'    w_R1_late:  coef = {model_both.params["w_R1_late"]:.4f} '
        f'(SE = {model_both.bse["w_R1_late"]:.4f}) '
        f'p = {model_both.pvalues["w_R1_late"]:.4f} '
        f'{significance_stars(model_both.pvalues["w_R1_late"])}')
    
    # Test for equality
    diff = model_both.params["w_R1_early"] - model_both.params["w_R1_late"]
    # Approximate SE of difference (assume independence for simplicity)
    se_diff = np.sqrt(model_both.bse["w_R1_early"]**2 + model_both.bse["w_R1_late"]**2)
    z_diff = diff / se_diff
    from scipy import stats
    p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
    log(f'\n    Difference (early - late): {diff:.4f} (SE ≈ {se_diff:.4f}) p ≈ {p_diff:.4f}')
    
    results.append({
        'test': 'RF: w_R1_early (joint)',
        'K': K, 'coef': model_both.params["w_R1_early"],
        'se': model_both.bse["w_R1_early"],
        'pval': model_both.pvalues["w_R1_early"],
        'stars': significance_stars(model_both.pvalues["w_R1_early"]),
        'N': len(df_est)
    })
    results.append({
        'test': 'RF: w_R1_late (joint)',
        'K': K, 'coef': model_both.params["w_R1_late"],
        'se': model_both.bse["w_R1_late"],
        'pval': model_both.pvalues["w_R1_late"],
        'stars': significance_stars(model_both.pvalues["w_R1_late"]),
        'N': len(df_est)
    })
    
    # -------------------------------------------------------------------------
    # TEST 2: IV with early-only instrument
    # -------------------------------------------------------------------------
    log('\n--- IV: Early-delivery R1 neighbors as instrument ---')
    
    # First stage: w_R1_early → w_R3_WINNER
    X_fs = sm.add_constant(df_est[ctrl_cols + state_fe_cols + ['w_R1_early']].astype(float))
    y_fs = df_est['w_R3_WINNER'].astype(float)
    fs_model = sm.OLS(y_fs, X_fs).fit(cov_type='HC1')
    f_stat = (fs_model.params["w_R1_early"] / fs_model.bse["w_R1_early"])**2
    log(f'\n  First stage (w_R1_early → w_R3_WINNER):')
    log(f'    coef = {fs_model.params["w_R1_early"]:.4f} '
        f'(SE = {fs_model.bse["w_R1_early"]:.4f}) F = {f_stat:.1f}')
    
    # IV estimation
    try:
        dep_var = df_est['IS_R3_WINNER'].astype(float)
        endog_var = df_est[['w_R3_WINNER']].astype(float)
        exog_ctrl = sm.add_constant(df_est[ctrl_cols + state_fe_cols].astype(float))
        instruments = df_est[['w_R1_early']].astype(float)
        
        iv_model = IV2SLS(dep_var, exog_ctrl, endog_var, instruments).fit(cov_type='robust')
        
        log(f'\n  IV estimate (2SLS):')
        log(f'    Peer effect (w_R3_WINNER): {iv_model.params["w_R3_WINNER"]:.4f} '
            f'(SE = {iv_model.std_errors["w_R3_WINNER"]:.4f}) '
            f'p = {iv_model.pvalues["w_R3_WINNER"]:.4f} '
            f'{significance_stars(iv_model.pvalues["w_R3_WINNER"])}')
        
        results.append({
            'test': 'IV: w_R1_early instrument',
            'K': K, 'coef': iv_model.params["w_R3_WINNER"],
            'se': iv_model.std_errors["w_R3_WINNER"],
            'pval': iv_model.pvalues["w_R3_WINNER"],
            'stars': significance_stars(iv_model.pvalues["w_R3_WINNER"]),
            'F_first': f_stat, 'N': len(df_est)
        })
    except Exception as e:
        log(f'  IV estimation failed: {e}')
    
    # -------------------------------------------------------------------------
    # TEST 3: IV with late-only instrument
    # -------------------------------------------------------------------------
    log('\n--- IV: Late-delivery R1 neighbors as instrument ---')
    
    X_fs_late = sm.add_constant(df_est[ctrl_cols + state_fe_cols + ['w_R1_late']].astype(float))
    fs_model_late = sm.OLS(y_fs, X_fs_late).fit(cov_type='HC1')
    f_stat_late = (fs_model_late.params["w_R1_late"] / fs_model_late.bse["w_R1_late"])**2
    log(f'\n  First stage (w_R1_late → w_R3_WINNER):')
    log(f'    coef = {fs_model_late.params["w_R1_late"]:.4f} '
        f'(SE = {fs_model_late.bse["w_R1_late"]:.4f}) F = {f_stat_late:.1f}')
    
    try:
        instruments_late = df_est[['w_R1_late']].astype(float)
        iv_model_late = IV2SLS(dep_var, exog_ctrl, endog_var, instruments_late).fit(cov_type='robust')
        
        log(f'\n  IV estimate (2SLS):')
        log(f'    Peer effect (w_R3_WINNER): {iv_model_late.params["w_R3_WINNER"]:.4f} '
            f'(SE = {iv_model_late.std_errors["w_R3_WINNER"]:.4f}) '
            f'p = {iv_model_late.pvalues["w_R3_WINNER"]:.4f} '
            f'{significance_stars(iv_model_late.pvalues["w_R3_WINNER"])}')
        
        results.append({
            'test': 'IV: w_R1_late instrument',
            'K': K, 'coef': iv_model_late.params["w_R3_WINNER"],
            'se': iv_model_late.std_errors["w_R3_WINNER"],
            'pval': iv_model_late.pvalues["w_R3_WINNER"],
            'stars': significance_stars(iv_model_late.pvalues["w_R3_WINNER"]),
            'F_first': f_stat_late, 'N': len(df_est)
        })
    except Exception as e:
        log(f'  IV estimation failed: {e}')

# ==============================================================================
# SUMMARY
# ==============================================================================

log('\n' + '=' * 80)
log('SUMMARY: DELIVERY TIMING AND PEER EFFECTS')
log('=' * 80)

log('\nHYPOTHESIS:')
log('  Early delivery → "ESB works" signal → POSITIVE peer effect')
log('  Late delivery → "ESB is difficult" signal → NEGATIVE peer effect')

log('\nKEY FINDINGS:')
log('  Compare reduced form coefficients:')
log('    - If early > 0 and late < 0: hypothesis supported')
log('    - If both negative but |late| > |early|: partial support')
log('    - If both similar: hypothesis rejected')

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(TABLES_DIR / 'delivery_timing_iv.csv', index=False)
log(f'\nSaved: {TABLES_DIR / "delivery_timing_iv.csv"}')

with open(LOGS_DIR / 'delivery_timing_iv.txt', 'w') as f:
    f.write('\n'.join(log_lines))
log(f'Saved: {LOGS_DIR / "delivery_timing_iv.txt"}')

print('\n' + '=' * 80)
print('ANALYSIS COMPLETE')
print('=' * 80)
