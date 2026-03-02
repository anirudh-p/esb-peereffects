"""
27b_early_delivery_peer_effects.py
===================================
Test whether peer effects are POSITIVE when R1 neighbors received buses on time
(signal: "ESB adoption works") vs NEGATIVE when buses were delayed
(signal: "ESB adoption is risky").

HYPOTHESIS:
  - Early delivery neighbors → Positive peer effect (success signal)
  - Late delivery neighbors → Negative peer effect (difficulty signal)

The temporal null may mask heterogeneous effects: the average combines
positive spillovers from successful deployments with negative spillovers
from problematic ones.

Outputs:
  - 3_Output/Tables/early_delivery_peer_effects.csv
  - 3_Output/Logs/early_delivery_peer_effects.txt
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
    """Convert 'Q1 2023' format to approximate date."""
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
log('EARLY vs LATE DELIVERY PEER EFFECTS')
log('Testing signal quality hypothesis')
log('=' * 80)

# ==============================================================================
# 1. LOAD AND PROCESS WRI BUS DATA FOR DELIVERY TIMING
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

# R3 deadline for visibility
r3_deadline = pd.Timestamp('2023-10-01')

# Classify delivery timing
r1_buses['delivered_early'] = r1_buses['delivery_date'] < r3_deadline
r1_buses['delivered_late'] = (r1_buses['delivery_date'] >= r3_deadline) | r1_buses['delivery_date'].isna()

log(f'R1 buses delivered before R3 deadline: {r1_buses["delivered_early"].sum():,}')
log(f'R1 buses delivered late/unknown: {r1_buses["delivered_late"].sum():,}')

# District-level delivery classification
# A district is "early delivery" if their FIRST bus was delivered before R3 deadline
r1_district = r1_buses.groupby('nces_id').agg({
    'delivery_date': 'min',  # First delivery
    'delivered_early': 'any'  # Any bus delivered early
}).reset_index()
r1_district.columns = ['nces_id', 'first_delivery', 'any_early_delivery']
r1_district['district_early'] = r1_district['first_delivery'] < r3_deadline

# Also create "all late" indicator
r1_all_late = r1_buses.groupby('nces_id')['delivered_early'].max().reset_index()
r1_all_late.columns = ['nces_id', 'has_any_early']
r1_all_late['all_late'] = ~r1_all_late['has_any_early']

r1_district = r1_district.merge(r1_all_late[['nces_id', 'all_late']], on='nces_id', how='left')

log(f'\nR1 winner districts: {len(r1_district):,}')
log(f'  Early delivery (first bus before R3): {r1_district["district_early"].sum():,}')
log(f'  Late delivery (first bus after R3 or missing): {(~r1_district["district_early"]).sum():,}')

# ==============================================================================
# 2. LOAD MAIN DATA AND CONSTRUCT OUTCOMES
# ==============================================================================

log('\n--- Loading main dataset and constructing outcomes ---')

# Load rebates for R1/R3 winner status
rebates = pd.read_excel(RAW_WRI_DIR / "CSB_Rebates.xlsx")
rebates['nces_id'] = clean_lea_id(rebates['NCES District ID'])
excluded = ['WITHDRAWN', 'CANCELLED', 'NOT SELECTED', 'INELIGIBLE', 'DENIED']
rebates = rebates[~rebates['Project Status'].str.upper().isin(excluded)]

r1_winners = rebates[rebates['Funding Year'] == 2022][['nces_id']].drop_duplicates()
r1_winners['IV_Z_R1'] = 1
r3_winners = rebates[rebates['Funding Year'] == 2023][['nces_id']].drop_duplicates()
r3_winners['IS_R3_WINNER'] = 1

# Merge delivery timing into R1 winners
r1_winners = r1_winners.merge(
    r1_district[['nces_id', 'district_early', 'all_late']], 
    on='nces_id', how='left'
)
r1_winners['R1_early'] = r1_winners['district_early'].fillna(False).astype(int)
r1_winners['R1_late'] = (~r1_winners['district_early'].fillna(True)).astype(int)

log(f'R1 winners with timing data: {r1_winners["R1_early"].sum() + r1_winners["R1_late"].sum():,}')
log(f'  Early delivery: {r1_winners["R1_early"].sum():,}')
log(f'  Late delivery: {r1_winners["R1_late"].sum():,}')

# All-source outcomes from WRI
df_bus_all = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='2. Bus-level data')
df_bus_all['nces_id'] = clean_lea_id(df_bus_all['1c. LEA ID'])
df_bus_all['award_year'] = extract_quarter_date(df_bus_all['3p. Quarter awarded']).dt.year

# R3 window outcomes (2023-2024)
r3_bus = df_bus_all[df_bus_all['award_year'].between(2023, 2024)]
r3_adopters = r3_bus['nces_id'].drop_duplicates()

# Main dataset
df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Merge outcomes and instruments
df = df.merge(r1_winners[['nces_id', 'IV_Z_R1', 'R1_early', 'R1_late']], on='nces_id', how='left')
df = df.merge(r3_winners, on='nces_id', how='left')
df['IV_Z_R1'] = df['IV_Z_R1'].fillna(0).astype(int)
df['R1_early'] = df['R1_early'].fillna(0).astype(int)
df['R1_late'] = df['R1_late'].fillna(0).astype(int)
df['IS_R3_WINNER'] = df['IS_R3_WINNER'].fillna(0).astype(int)
df['adopted_r3_window'] = df['nces_id'].isin(r3_adopters).astype(int)

log(f'\nTotal districts: {len(df):,}')
log(f'R1 winners: {df["IV_Z_R1"].sum():,}')
log(f'  Early: {df["R1_early"].sum():,}')
log(f'  Late: {df["R1_late"].sum():,}')

# ==============================================================================
# 3. BUILD SPATIAL WEIGHTS AND LAGS
# ==============================================================================

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

for K in [4, 6, 8]:
    log(f'\n{"=" * 80}')
    log(f'K = {K}')
    log(f'{"=" * 80}')
    
    w = KNN.from_dataframe(geo_pts, k=K)
    w.transform = 'r'
    
    # Create spatial lags for different neighbor types
    geo_all['w_R1_all'] = lag_spatial(w, geo_all['IV_Z_R1'].values)
    geo_all['w_R1_early'] = lag_spatial(w, geo_all['R1_early'].values)
    geo_all['w_R1_late'] = lag_spatial(w, geo_all['R1_late'].values)
    geo_all['w_R3_WIN'] = lag_spatial(w, geo_all['IS_R3_WINNER'].values)
    geo_all['w_r3_any'] = lag_spatial(w, geo_all['adopted_r3_window'].values)
    
    # Estimation sample: exclude R1 winners
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
    
    log(f'Estimation sample: {len(df_est):,}')
    log(f'Mean w_R1_early: {df_est["w_R1_early"].mean():.5f}')
    log(f'Mean w_R1_late: {df_est["w_R1_late"].mean():.5f}')
    
    # -------------------------------------------------------------------------
    # TEST 1: Reduced form - separate early vs late effects
    # -------------------------------------------------------------------------
    log('\n--- TEST 1: Reduced Form (Early vs Late R1 Neighbors) ---')
    
    for outcome_name, y_col in [('R3_CSBP_win', 'IS_R3_WINNER'), ('R3_any_adoption', 'adopted_r3_window')]:
        y = df_est[y_col].astype(float)
        X = sm.add_constant(df_est[ctrl_cols + state_fe_cols + ['w_R1_early', 'w_R1_late']].astype(float))
        
        try:
            model = sm.OLS(y, X).fit(cov_type='HC1')
            
            coef_early = model.params['w_R1_early']
            se_early = model.bse['w_R1_early']
            pval_early = model.pvalues['w_R1_early']
            
            coef_late = model.params['w_R1_late']
            se_late = model.bse['w_R1_late']
            pval_late = model.pvalues['w_R1_late']
            
            log(f'\n  {outcome_name}:')
            log(f'    w_R1_EARLY: coef={coef_early:7.4f} (SE={se_early:.4f}) p={pval_early:.4f} {significance_stars(pval_early)}')
            log(f'    w_R1_LATE:  coef={coef_late:7.4f} (SE={se_late:.4f}) p={pval_late:.4f} {significance_stars(pval_late)}')
            
            # Test for difference
            diff = coef_early - coef_late
            # Approximate SE of difference (assumes independence - conservative)
            se_diff = np.sqrt(se_early**2 + se_late**2)
            z_diff = diff / se_diff
            p_diff = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(z_diff)))
            log(f'    DIFFERENCE (early - late): {diff:7.4f} (SE={se_diff:.4f}) p={p_diff:.4f}')
            
            results.append({
                'K': K, 'outcome': outcome_name, 'neighbor_type': 'R1_early',
                'coef': coef_early, 'se': se_early, 'pval': pval_early,
                'stars': significance_stars(pval_early), 'N': len(df_est)
            })
            results.append({
                'K': K, 'outcome': outcome_name, 'neighbor_type': 'R1_late',
                'coef': coef_late, 'se': se_late, 'pval': pval_late,
                'stars': significance_stars(pval_late), 'N': len(df_est)
            })
            results.append({
                'K': K, 'outcome': outcome_name, 'neighbor_type': 'DIFFERENCE',
                'coef': diff, 'se': se_diff, 'pval': p_diff,
                'stars': significance_stars(p_diff), 'N': len(df_est)
            })
            
        except Exception as e:
            log(f'  {outcome_name}: Error - {e}')
    
    # -------------------------------------------------------------------------
    # TEST 2: IV using ONLY early-delivery R1 as instrument
    # -------------------------------------------------------------------------
    log('\n--- TEST 2: IV with Early-Delivery R1 Neighbors Only ---')
    
    for outcome_name, y_col, endog_col in [
        ('R3_CSBP_win', 'IS_R3_WINNER', 'w_R3_WIN'),
        ('R3_any_adoption', 'adopted_r3_window', 'w_r3_any')
    ]:
        try:
            y = df_est[y_col].astype(float)
            X_exog = df_est[ctrl_cols + state_fe_cols].astype(float)
            X_endog = df_est[[endog_col]].astype(float)
            Z = df_est[['w_R1_early']].astype(float)
            
            # Check first stage strength
            first_stage = sm.OLS(
                X_endog.values.flatten(), 
                sm.add_constant(pd.concat([X_exog, Z], axis=1))
            ).fit()
            f_stat = first_stage.f_pvalue
            
            model = IV2SLS(
                dependent=y,
                exog=sm.add_constant(X_exog),
                endog=X_endog,
                instruments=Z
            ).fit(cov_type='robust')
            
            coef = model.params[endog_col]
            se = model.std_errors[endog_col]
            pval = model.pvalues[endog_col]
            
            # First stage F
            first_y = X_endog.values.flatten()
            first_X = sm.add_constant(pd.concat([X_exog, Z], axis=1))
            first_model = sm.OLS(first_y, first_X).fit()
            f_stat = first_model.f_test(f'w_R1_early = 0').fvalue[0][0]
            
            log(f'\n  {outcome_name} (IV: w_R1_early only):')
            log(f'    Peer effect: coef={coef:7.4f} (SE={se:.4f}) p={pval:.4f} {significance_stars(pval)}')
            log(f'    First-stage F: {f_stat:.1f}')
            
            results.append({
                'K': K, 'outcome': f'{outcome_name}_IV_early_only', 'neighbor_type': 'peer_effect',
                'coef': coef, 'se': se, 'pval': pval,
                'stars': significance_stars(pval), 'N': len(df_est), 'F': f_stat
            })
            
        except Exception as e:
            log(f'  {outcome_name}: Error - {e}')
    
    # -------------------------------------------------------------------------
    # TEST 3: IV using ONLY late-delivery R1 as instrument (comparison)
    # -------------------------------------------------------------------------
    log('\n--- TEST 3: IV with Late-Delivery R1 Neighbors Only ---')
    
    for outcome_name, y_col, endog_col in [
        ('R3_CSBP_win', 'IS_R3_WINNER', 'w_R3_WIN'),
        ('R3_any_adoption', 'adopted_r3_window', 'w_r3_any')
    ]:
        try:
            y = df_est[y_col].astype(float)
            X_exog = df_est[ctrl_cols + state_fe_cols].astype(float)
            X_endog = df_est[[endog_col]].astype(float)
            Z = df_est[['w_R1_late']].astype(float)
            
            model = IV2SLS(
                dependent=y,
                exog=sm.add_constant(X_exog),
                endog=X_endog,
                instruments=Z
            ).fit(cov_type='robust')
            
            coef = model.params[endog_col]
            se = model.std_errors[endog_col]
            pval = model.pvalues[endog_col]
            
            # First stage F
            first_y = X_endog.values.flatten()
            first_X = sm.add_constant(pd.concat([X_exog, Z], axis=1))
            first_model = sm.OLS(first_y, first_X).fit()
            f_stat = first_model.f_test(f'w_R1_late = 0').fvalue[0][0]
            
            log(f'\n  {outcome_name} (IV: w_R1_late only):')
            log(f'    Peer effect: coef={coef:7.4f} (SE={se:.4f}) p={pval:.4f} {significance_stars(pval)}')
            log(f'    First-stage F: {f_stat:.1f}')
            
            results.append({
                'K': K, 'outcome': f'{outcome_name}_IV_late_only', 'neighbor_type': 'peer_effect',
                'coef': coef, 'se': se, 'pval': pval,
                'stars': significance_stars(pval), 'N': len(df_est), 'F': f_stat
            })
            
        except Exception as e:
            log(f'  {outcome_name}: Error - {e}')

# ==============================================================================
# SUMMARY
# ==============================================================================

log('\n' + '=' * 80)
log('SUMMARY: SIGNAL QUALITY HYPOTHESIS')
log('=' * 80)

log('\nHYPOTHESIS:')
log('  - Early delivery = "ESB adoption works" → Positive peer effect')
log('  - Late delivery = "ESB adoption is risky" → Negative peer effect')

log('\nKEY RESULTS:')
results_df = pd.DataFrame(results)

# Summarize K=6 results (middle ground)
k6 = results_df[results_df['K'] == 6]
for outcome in ['R3_CSBP_win', 'R3_any_adoption']:
    subset = k6[k6['outcome'] == outcome]
    if len(subset) >= 2:
        early = subset[subset['neighbor_type'] == 'R1_early'].iloc[0] if len(subset[subset['neighbor_type'] == 'R1_early']) > 0 else None
        late = subset[subset['neighbor_type'] == 'R1_late'].iloc[0] if len(subset[subset['neighbor_type'] == 'R1_late']) > 0 else None
        diff = subset[subset['neighbor_type'] == 'DIFFERENCE'].iloc[0] if len(subset[subset['neighbor_type'] == 'DIFFERENCE']) > 0 else None
        
        if early is not None and late is not None:
            log(f'\n  {outcome}:')
            log(f'    Early R1 neighbor: {early["coef"]:.4f} {early["stars"]}')
            log(f'    Late R1 neighbor:  {late["coef"]:.4f} {late["stars"]}')
            if diff is not None:
                log(f'    Difference:        {diff["coef"]:.4f} {diff["stars"]}')

# Save results
results_df.to_csv(TABLES_DIR / 'early_delivery_peer_effects.csv', index=False)
log(f'\nSaved: {TABLES_DIR / "early_delivery_peer_effects.csv"}')

with open(LOGS_DIR / 'early_delivery_peer_effects.txt', 'w') as f:
    f.write('\n'.join(log_lines))
log(f'Saved: {LOGS_DIR / "early_delivery_peer_effects.txt"}')

print('\n' + '=' * 80)
print('ANALYSIS COMPLETE')
print('=' * 80)
