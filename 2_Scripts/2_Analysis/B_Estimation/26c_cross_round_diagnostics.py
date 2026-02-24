"""
26c_cross_round_diagnostics.py
================================
Systematic diagnostics for the cross-round IV discrepancy:
  Script 20: CSBP-only peer effect = +1.343*** (F=35)
  Script 26: All-source peer effect ~ 0.00 (F>5,000)

DIAGNOSTIC BATTERY:
  D1. Outcome overlap: How many R3 CSBP winners appear in WRI 2023-2024 data?
  D2. Same-endogenous test: Run Script 26's design with w_R3_CSBP as endogenous
  D3. Reduced form comparison: RF for CSBP-outcome vs all-source outcome
  D4. LATE scaling: What does +1.343 actually mean in practical terms?
  D5. First-stage decomposition: Why F=35 vs F>5,000?
  D6. Own-treatment controls: Does adding IS_APPLICANT to Script 20 kill its result?
  D7. coefficient > 1.0 diagnostic: weak IV bias, Anderson-Rubin test

Outputs:
  - 3_Output/Tables/cross_round_diagnostics.csv
  - 3_Output/Logs/cross_round_diagnostics.txt
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
    WRI_EXCEL_FILE, APPLICANT_FILE, TABLES_DIR, LOGS_DIR, ensure_dirs_exist
)

from libpysal.weights import KNN, lag_spatial
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
from scipy import stats

ensure_dirs_exist()

DATA_DIR = Path(__file__).parent.parent.parent.parent / "1_Data"
RAW_WRI = DATA_DIR / "Raw" / "WRI"

log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(msg)

def clean_lea_id(series):
    return series.astype(str).str.split('.').str[0].str.zfill(7)

def extract_year_from_quarter(series):
    years = series.astype(str).str.extract(r'(\d{4})', expand=False)
    return pd.to_numeric(years, errors='coerce')

def stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

log('=' * 90)
log('CROSS-ROUND IV DIAGNOSTICS: WHY +1.343 (CSBP) vs ~0.00 (ALL-SOURCE)?')
log('=' * 90)

# ==============================================================================
# 0. DATA CONSTRUCTION (same as Script 26)
# ==============================================================================

log('\n--- Data construction ---')

# R1 and R3 CSBP winners
rebates = pd.read_excel(RAW_WRI / "CSB_Rebates.xlsx")
rebates['nces_id'] = clean_lea_id(rebates['NCES District ID'])
excluded = ['WITHDRAWN', 'CANCELLED', 'NOT SELECTED', 'INELIGIBLE', 'DENIED']
rebates = rebates[~rebates['Project Status'].str.upper().isin(excluded)]

r1_winners = rebates[rebates['Funding Year'] == 2022][['nces_id']].drop_duplicates()
r1_winners['IV_Z_R1'] = 1

r3_winners = rebates[rebates['Funding Year'] == 2023][['nces_id']].drop_duplicates()
r3_winners['IS_R3_ADOPTER'] = 1

log(f'  R1 CSBP winners (unique districts): {len(r1_winners):,}')
log(f'  R3 CSBP winners (unique districts): {len(r3_winners):,}')
log(f'  R1 AND R3 winners: {len(set(r1_winners.nces_id) & set(r3_winners.nces_id)):,}')

# WRI bus-level data
df_bus = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='2. Bus-level data')
df_bus['nces_id'] = clean_lea_id(df_bus['1c. LEA ID'])
df_bus['order_year'] = extract_year_from_quarter(df_bus['3q. Quarter ordered'])
valid = df_bus['order_year'].between(1990, 2035) & df_bus['nces_id'].ne('0000000')
df_bus = df_bus[valid].copy()

# District-level outcomes from WRI
first_year = df_bus.groupby('nces_id')['order_year'].min().rename('first_order_year').reset_index()
first_year['pre_r1_adopter'] = (first_year['first_order_year'] < 2022).astype(int)

r3_window_buses = df_bus[df_bus['order_year'].between(2023, 2024)]
r3_any = r3_window_buses.groupby('nces_id').size().rename('r3_bus_count').reset_index()

outcomes = first_year.merge(r3_any, on='nces_id', how='left')
outcomes['r3_bus_count'] = outcomes['r3_bus_count'].fillna(0)
outcomes['new_r3_first'] = ((outcomes['first_order_year'] >= 2023) & (outcomes['first_order_year'] <= 2024)).astype(int)
outcomes['new_r3_any'] = (outcomes['r3_bus_count'] > 0).astype(int)

# Applicants
df_applicants = pd.read_excel(str(APPLICANT_FILE))
id_col_app = [c for c in df_applicants.columns if 'NCES' in c.upper() and 'ID' in c.upper()][0]
df_applicants['nces_id'] = clean_lea_id(df_applicants[id_col_app])

# ==============================================================================
# D1. OUTCOME OVERLAP: R3 CSBP WINNERS vs WRI 2023-2024 DATA
# ==============================================================================

log('\n' + '=' * 90)
log('D1. OUTCOME OVERLAP: Do R3 CSBP winners appear in WRI 2023-2024 bus data?')
log('=' * 90)

r3_set = set(r3_winners['nces_id'])
wri_2324_first = set(outcomes[outcomes['new_r3_first'] == 1]['nces_id'])
wri_2324_any = set(outcomes[outcomes['new_r3_any'] == 1]['nces_id'])

overlap_first = r3_set & wri_2324_first
overlap_any = r3_set & wri_2324_any
r3_not_in_wri_first = r3_set - wri_2324_first
r3_not_in_wri_any = r3_set - wri_2324_any

log(f'  R3 CSBP winner districts: {len(r3_set):,}')
log(f'  WRI first-ever 2023-2024 districts: {len(wri_2324_first):,}')
log(f'  WRI any-event 2023-2024 districts: {len(wri_2324_any):,}')
log(f'')
log(f'  R3 CSBP winners IN WRI first-ever: {len(overlap_first):,} ({100*len(overlap_first)/len(r3_set):.1f}%)')
log(f'  R3 CSBP winners IN WRI any-event:  {len(overlap_any):,} ({100*len(overlap_any)/len(r3_set):.1f}%)')
log(f'  R3 CSBP winners NOT in WRI first:  {len(r3_not_in_wri_first):,} ({100*len(r3_not_in_wri_first)/len(r3_set):.1f}%)')
log(f'  R3 CSBP winners NOT in WRI any:    {len(r3_not_in_wri_any):,} ({100*len(r3_not_in_wri_any)/len(r3_set):.1f}%)')

# Check: where are the R3 winners in the WRI data?
bus_by_r3 = df_bus[df_bus['nces_id'].isin(r3_set)]
if len(bus_by_r3) > 0:
    r3_order_years = bus_by_r3.groupby('nces_id')['order_year'].agg(['min', 'max']).reset_index()
    log(f'\n  Among R3 CSBP winners that appear in WRI:')
    log(f'    Districts with ANY WRI entry: {len(r3_order_years):,}')
    log(f'    Earliest order: {r3_order_years["min"].min():.0f}')
    log(f'    Latest order:   {r3_order_years["max"].max():.0f}')
    has_2324 = r3_order_years[(r3_order_years['min'] <= 2024) | (r3_order_years['max'] >= 2023)]
    log(f'    With 2023-2024 orders: {len(bus_by_r3[bus_by_r3["order_year"].between(2023, 2024)]["nces_id"].unique()):,}')
    log(f'    With orders only AFTER 2024: {len(r3_order_years[r3_order_years["min"] > 2024]):,}')
    log(f'    No WRI data at all: {len(r3_set) - len(r3_order_years):,}')
else:
    log(f'  WARNING: No WRI bus entries found for R3 CSBP winners!')

# What funding sources are the WRI 2023-2024 buses?
if '3b. Funding source(s)' in df_bus.columns:
    wri_2324_buses = df_bus[df_bus['order_year'].between(2023, 2024)]
    log(f'\n  WRI 2023-2024 bus funding sources:')
    funding_col = '3b. Funding source(s)'
    fund_counts = wri_2324_buses[funding_col].value_counts().head(15)
    for src, cnt in fund_counts.items():
        log(f'    {src}: {cnt:,}')
elif '3c. Primary funding source' in df_bus.columns:
    wri_2324_buses = df_bus[df_bus['order_year'].between(2023, 2024)]
    funding_col = '3c. Primary funding source'
    log(f'\n  WRI 2023-2024 bus funding sources ({funding_col}):')
    fund_counts = wri_2324_buses[funding_col].value_counts().head(15)
    for src, cnt in fund_counts.items():
        log(f'    {src}: {cnt:,}')
else:
    funding_cols = [c for c in df_bus.columns if 'fund' in c.lower() or 'source' in c.lower()]
    log(f'  Funding-related columns: {funding_cols}')
    if funding_cols:
        wri_2324_buses = df_bus[df_bus['order_year'].between(2023, 2024)]
        for fc in funding_cols[:2]:
            log(f'\n  {fc} distribution:')
            vc = wri_2324_buses[fc].value_counts().head(10)
            for src, cnt in vc.items():
                log(f'    {src}: {cnt:,}')

# ==============================================================================
# D2-D7: FULL ESTIMATION DIAGNOSTICS (requires spatial weights)
# ==============================================================================

log('\n--- Loading base dataset and building spatial weights ---')

df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Merge all indicators
df = df.merge(r1_winners, on='nces_id', how='left')
df = df.merge(r3_winners, on='nces_id', how='left')
df = df.merge(outcomes[['nces_id', 'new_r3_first', 'new_r3_any', 'pre_r1_adopter']], on='nces_id', how='left')

for c in ['IV_Z_R1', 'IS_R3_ADOPTER', 'new_r3_first', 'new_r3_any', 'pre_r1_adopter']:
    df[c] = df[c].fillna(0).astype(int)

all_applicants = df_applicants['nces_id'].unique()
df['IS_APPLICANT_pool'] = df['nces_id'].isin(all_applicants).astype(int)

# Political data
pres_df = pd.read_csv(str(POLITICAL_COUNTY_PRES_FILE))
pres_2020 = pres_df[(pres_df['year'] == 2020) & (pres_df['office'] == 'US PRESIDENT')].copy()
ct = pres_2020.groupby('county_fips')['candidatevotes'].sum().reset_index().rename(columns={'candidatevotes': 'total_votes'})
cd = pres_2020[pres_2020['party'] == 'DEMOCRAT'].groupby('county_fips')['candidatevotes'].sum().reset_index().rename(columns={'candidatevotes': 'dem_votes'})
cp = ct.merge(cd, on='county_fips', how='left')
cp['pct_dem_2020'] = cp['dem_votes'] / cp['total_votes']
cp['county_fips'] = cp['county_fips'].astype(int)
lea_county = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='5. Counties')
lea_county = lea_county[['1c. LEA ID', '10b. County FIPS Code']].copy()
lea_county.columns = ['nces_id', 'county_fips']
lea_county['nces_id'] = clean_lea_id(lea_county['nces_id'])
lea_county = lea_county.merge(cp[['county_fips', 'pct_dem_2020']], on='county_fips', how='left')
lea_pol = lea_county.groupby('nces_id')['pct_dem_2020'].mean().reset_index()

df = df.merge(lea_pol, on='nces_id', how='left')
if 'urbanicity' in df.columns:
    ld = pd.get_dummies(df['urbanicity'], prefix='locale', drop_first=True)
    df = pd.concat([df.reset_index(drop=True), ld.reset_index(drop=True)], axis=1)
    locale_vars = [c for c in df.columns if c.startswith('locale_')]
else:
    locale_vars = []

base_controls = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pm25',
                 'pct_white', 'pct_dem_2020', 'is_priority'] + locale_vars

# Spatial construction
gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
shp_id = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]
gdf[shp_id] = gdf[shp_id].astype(str).str.zfill(7)

gdf_m = gdf.merge(df[['nces_id']].drop_duplicates(), left_on=shp_id, right_on='nces_id', how='inner')
gdf_m = gdf_m.to_crs(epsg=5070).drop(columns=['nces_id'])
geo_all = gdf_m.merge(df, left_on=shp_id, right_on='nces_id', how='inner').reset_index(drop=True)
geo_all['centroid'] = geo_all.geometry.centroid
geo_pts = geo_all.set_geometry('centroid')

log(f'  Full spatial sample: {len(geo_all):,}')

K = 6
w = KNN.from_dataframe(geo_pts, k=K)
w.transform = 'r'

# Spatial lags
geo_all['w_IV_Z_R1'] = lag_spatial(w, geo_all['IV_Z_R1'].values)
geo_all['w_R3_CSBP'] = lag_spatial(w, geo_all['IS_R3_ADOPTER'].values)
geo_all['w_r3_first'] = lag_spatial(w, geo_all['new_r3_first'].values)
geo_all['w_r3_any'] = lag_spatial(w, geo_all['new_r3_any'].values)
geo_all['IS_LOSER'] = ((geo_all['IS_APPLICANT_pool'] == 1) & (geo_all['IV_Z_R1'] == 0) & (geo_all['IV_Z'] == 0)).astype(int)
geo_all['w_loser'] = lag_spatial(w, geo_all['IS_LOSER'].values)

# Estimation sample: non-R1-winners
est = geo_all[geo_all['IV_Z_R1'] == 0].copy().reset_index(drop=True)
ctrl = [c for c in base_controls if c in est.columns]
est = est.dropna(subset=ctrl + ['state']).copy()
sd = pd.get_dummies(est['state'], prefix='st', drop_first=True)
est = pd.concat([est.reset_index(drop=True), sd.reset_index(drop=True)], axis=1)
sfe = list(sd.columns)

log(f'  K={K} estimation sample: {len(est):,}')

# ==============================================================================
# D2. OUTCOME COMPARISON TABLE
# ==============================================================================

log('\n' + '=' * 90)
log('D2. OUTCOME VARIABLE COMPARISON (estimation sample)')
log('=' * 90)

log(f'  IS_R3_ADOPTER (CSBP R3):   mean={est["IS_R3_ADOPTER"].mean():.5f}  sum={est["IS_R3_ADOPTER"].sum():,}')
log(f'  new_r3_first (WRI first):   mean={est["new_r3_first"].mean():.5f}  sum={est["new_r3_first"].sum():,}')
log(f'  new_r3_any (WRI any):       mean={est["new_r3_any"].mean():.5f}  sum={est["new_r3_any"].sum():,}')

# Cross-tabulation: IS_R3_ADOPTER vs new_r3_any
ct = pd.crosstab(est['IS_R3_ADOPTER'], est['new_r3_any'], margins=True)
log(f'\n  Cross-tab: IS_R3_ADOPTER (rows) x new_r3_any (cols):')
log(ct.to_string())

# How many R3 CSBP adopters are NOT in WRI 2023-2024?
r3_csbp_yes = est[est['IS_R3_ADOPTER'] == 1]
r3_csbp_not_in_wri = r3_csbp_yes[r3_csbp_yes['new_r3_any'] == 0]
log(f'\n  R3 CSBP adopters NOT captured by WRI 2023-2024: {len(r3_csbp_not_in_wri):,}/{len(r3_csbp_yes):,} '
    f'({100*len(r3_csbp_not_in_wri)/max(len(r3_csbp_yes),1):.1f}%)')

# ==============================================================================
# D3. ENDOGENOUS VARIABLE COMPARISON
# ==============================================================================

log('\n' + '=' * 90)
log('D3. ENDOGENOUS (SPATIAL LAG) VARIABLE COMPARISON')
log('=' * 90)

for varname in ['w_R3_CSBP', 'w_r3_first', 'w_r3_any', 'w_IV_Z_R1']:
    v = est[varname]
    log(f'  {varname:15s}: mean={v.mean():.6f} sd={v.std():.6f} min={v.min():.4f} max={v.max():.4f} nonzero={int((v > 0).sum()):,}')

log(f'\n  Correlation matrix:')
corr = est[['w_R3_CSBP', 'w_r3_first', 'w_r3_any', 'w_IV_Z_R1']].corr()
log(corr.round(4).to_string())

# ==============================================================================
# D4. FIRST-STAGE COMPARISON (same instrument, different endogenous)
# ==============================================================================

log('\n' + '=' * 90)
log('D4. FIRST-STAGE COMPARISON: w_IV_Z_R1 predicting different endogenous variables')
log('=' * 90)

Z = est[['w_IV_Z_R1']].astype(float)
X_exog = sm.add_constant(est[ctrl + sfe].astype(float))
X_exog = X_exog.loc[:, X_exog.std() > 0]
fsX = pd.concat([X_exog, Z], axis=1)

for endog_name in ['w_R3_CSBP', 'w_r3_first', 'w_r3_any']:
    endog_vals = est[endog_name].astype(float)
    fs = sm.OLS(endog_vals, fsX.astype(float)).fit()
    idx = fsX.columns.get_loc('w_IV_Z_R1')
    coef = fs.params.iloc[idx]
    t = fs.tvalues.iloc[idx]
    f = t ** 2
    r2 = fs.rsquared
    partial_r2 = (t**2 / (t**2 + fs.df_resid))
    log(f'  {endog_name:15s}: FS_coef={coef:>8.5f}  t={t:>8.2f}  F={f:>10.1f}  '
        f'R2={r2:.5f}  partialR2={partial_r2:.6f}')

log(f'\n  KEY INSIGHT: F=35 means w_IV_Z_R1 has modest predictive power for w_R3_CSBP')
log(f'  which is correct -- R1 wins are SPATIALLY correlated with but not mechanically')
log(f'  identical to R3 CSBP adoption. F>5000 for w_r3_any likely reflects geographic')
log(f'  correlation (ESB-friendly states have both R1 winners and all-source adopters).')

# ==============================================================================
# D5. REDUCED FORM COMPARISON: Same Z, different Y
# ==============================================================================

log('\n' + '=' * 90)
log('D5. REDUCED FORM COMPARISON: w_IV_Z_R1 -> different outcomes (OLS, clustered)')
log('=' * 90)

X_rf = pd.concat([X_exog, Z], axis=1)
X_rf = X_rf.loc[:, X_rf.std() > 0]
idx_z = X_rf.columns.get_loc('w_IV_Z_R1')

for y_name in ['IS_R3_ADOPTER', 'new_r3_first', 'new_r3_any']:
    y = est[y_name].astype(float)
    ols = sm.OLS(y.values, X_rf.astype(float).values).fit(
        cov_type='cluster', cov_kwds={'groups': est['state'].values}
    )
    coef = ols.params[idx_z]
    se = ols.bse[idx_z]
    pval = ols.pvalues[idx_z]
    log(f'  Y={y_name:18s}: RF_coef={coef:>9.5f} (SE={se:.5f}) p={pval:.4f}{stars(pval)}'
        f'  y_mean={est[y_name].mean():.5f}')

log(f'\n  If RF for IS_R3_ADOPTER is significant but RF for all-source is not,')
log(f'  it means R1 neighbor exposure ONLY affects CSBP adoption, not general ESB.')

# ==============================================================================
# D6. IV-2SLS COMPARISON: Same instrument, same sample, different endogenous + outcome
# ==============================================================================

log('\n' + '=' * 90)
log('D6. IV-2SLS COMPARISON (K=6, state-clustered, same sample)')
log('=' * 90)

results = []

specs = [
    # (label, y_col, endog_col, extra_controls, description)
    ('Script20 replication', 'IS_R3_ADOPTER', 'w_R3_CSBP', [], 'CSBP outcome, CSBP endogenous, no own controls'),
    ('Script20 + own Z', 'IS_R3_ADOPTER', 'w_R3_CSBP', ['IS_APPLICANT_pool', 'pre_r1_adopter'], 'CSBP outcome + own treatment controls'),
    ('All-source, all-source endog', 'new_r3_any', 'w_r3_any', ['pre_r1_adopter'], 'All-source outcome, all-source endogenous'),
    ('CSBP outcome, all-source endog', 'IS_R3_ADOPTER', 'w_r3_any', [], 'CSBP outcome but all-source endogenous'),
    ('All-source outcome, CSBP endog', 'new_r3_any', 'w_R3_CSBP', [], 'All-source outcome but CSBP endogenous'),
]

for label, y_col, endog_col, extra_ctrl, desc in specs:
    ctrls = ctrl + extra_ctrl
    ctrls = [c for c in ctrls if c in est.columns]

    y = est[y_col].astype(float)
    X_endog = est[[endog_col]].astype(float)
    X_exog_r = sm.add_constant(est[ctrls + sfe].astype(float))
    X_exog_r = X_exog_r.loc[:, X_exog_r.std() > 0]

    # First stage
    fsX_r = pd.concat([X_exog_r, Z], axis=1)
    fs_r = sm.OLS(X_endog.values.ravel(), fsX_r.astype(float).values).fit()
    idx_z_r = fsX_r.columns.get_loc('w_IV_Z_R1')
    f_r = fs_r.tvalues[idx_z_r] ** 2

    try:
        iv = IV2SLS(dependent=y, exog=X_exog_r, endog=X_endog, instruments=Z).fit(
            cov_type='clustered', clusters=est['state']
        )
        coef = iv.params[endog_col]
        se = iv.std_errors[endog_col]
        pval = iv.pvalues[endog_col]
    except Exception as e:
        log(f'  ERROR {label}: {e}')
        coef, se, pval = np.nan, np.nan, np.nan

    results.append({
        'spec': label, 'y': y_col, 'endog': endog_col,
        'coef': coef, 'se': se, 'pval': pval, 'stars': stars(pval),
        'F_first': f_r, 'N': len(est)
    })

    log(f'  {label:40s} | coef={coef:>8.4f} (SE={se:.4f}) p={pval:.4f}{stars(pval)} | F={f_r:.1f}')
    log(f'    [{desc}]')

# ==============================================================================
# D7. WEAK IV: Anderson-Rubin Test for Script 20 replication
# ==============================================================================

log('\n' + '=' * 90)
log('D7. ANDERSON-RUBIN CONFIDENCE SET (robust to weak IV)')
log('=' * 90)
log('  The AR test inverts the Wald test to construct a confidence set')
log('  valid even when the instrument is weak (F=35 is moderate).')

# AR test: under H0: beta=beta0, the RF residual should be uncorrelated with Z
# AR statistic = (RF - beta0 * FS)' Z' (Z Z)^{-1} Z (RF - beta0 * FS) / sigma^2

y_csbp = est['IS_R3_ADOPTER'].astype(float)
endog_csbp = est['w_R3_CSBP'].astype(float)
z_vals = est['w_IV_Z_R1'].astype(float)

# Partial out controls from y, endog, and Z
Xc = sm.add_constant(est[ctrl + sfe].astype(float))
Xc = Xc.loc[:, Xc.std() > 0]

y_resid = sm.OLS(y_csbp, Xc).fit().resid
endog_resid = sm.OLS(endog_csbp, Xc).fit().resid
z_resid = sm.OLS(z_vals, Xc).fit().resid

# AR test: for a grid of beta0 values, test H0: beta = beta0
beta_grid = np.arange(-2.0, 5.01, 0.05)
ar_pvals = []
for b0 in beta_grid:
    resid_b0 = y_resid - b0 * endog_resid
    ols_ar = sm.OLS(resid_b0, z_resid).fit()
    f_ar = ols_ar.fvalue
    p_ar = ols_ar.f_pvalue
    ar_pvals.append(p_ar)

ar_pvals = np.array(ar_pvals)
ar_ci_mask = ar_pvals >= 0.05  # 95% CI

if ar_ci_mask.any():
    ci_lo = beta_grid[ar_ci_mask].min()
    ci_hi = beta_grid[ar_ci_mask].max()
    log(f'  AR 95% CI for CSBP peer effect: [{ci_lo:.2f}, {ci_hi:.2f}]')
    log(f'  (Wald CI from IV2SLS: [{1.343 - 1.96*0.505:.2f}, {1.343 + 1.96*0.505:.2f}])')
    log(f'  If AR CI is much wider or includes 0, weak-IV bias inflates the point estimate.')
else:
    log(f'  AR test: No beta value in [-2, 5] is not rejected => empty or very wide CI')

# AR at beta=0 (test of null effect)
resid_0 = y_resid - 0 * endog_resid
ols_ar_0 = sm.OLS(resid_0, z_resid).fit()
log(f'  AR test at beta=0: F={ols_ar_0.fvalue:.4f}, p={ols_ar_0.f_pvalue:.4f}')
log(f'  (If p<0.05, we reject beta=0 even under weak-IV asymptotics)')

# ==============================================================================
# D8. COEFFICIENT > 1.0 DIAGNOSIS
# ==============================================================================

log('\n' + '=' * 90)
log('D8. WHY COEFFICIENT > 1.0? WALD RATIO DECOMPOSITION')
log('=' * 90)

# The IV estimator is: beta_IV = beta_RF / beta_FS
# Let's compute these explicitly

# RF: w_IV_Z_R1 -> IS_R3_ADOPTER
X_rf2 = pd.concat([Xc, pd.DataFrame({'w_IV_Z_R1': z_vals})], axis=1)
X_rf2 = X_rf2.loc[:, X_rf2.std() > 0]
rf_ols = sm.OLS(y_csbp.values, X_rf2.astype(float).values).fit()
idx_rf = X_rf2.columns.get_loc('w_IV_Z_R1')
rf_coef = rf_ols.params[idx_rf]
rf_se = rf_ols.bse[idx_rf]

# FS: w_IV_Z_R1 -> w_R3_CSBP
fs_ols = sm.OLS(endog_csbp.values, X_rf2.astype(float).values).fit()
fs_coef = fs_ols.params[idx_rf]
fs_se = fs_ols.bse[idx_rf]

log(f'  Reduced Form: coef = {rf_coef:.6f} (SE={rf_se:.6f})')
log(f'  First Stage:  coef = {fs_coef:.6f} (SE={fs_se:.6f})')
log(f'  Wald ratio: {rf_coef:.6f} / {fs_coef:.6f} = {rf_coef/fs_coef:.4f}')
log(f'  IV2SLS gave: +1.343')
log(f'')
log(f'  Interpretation: beta_IV = RF/FS. If FS is SMALL (weak instrument),')
log(f'  dividing by a small number inflates the point estimate.')
log(f'  RF = {rf_coef:.6f} means: a 1-unit shift in neighbor R1 share changes')
log(f'  own R3 CSBP probability by {rf_coef:.4f} ({rf_coef*100:.2f}pp).')
log(f'  FS = {fs_coef:.6f} means: a 1-unit shift in R1 share changes neighbor')
log(f'  R3 adoption share by {fs_coef:.4f}.')
log(f'')
log(f'  Practical interpretation (K=6):')
log(f'    max w_IV_Z_R1 in sample = {est["w_IV_Z_R1"].max():.4f}')
log(f'    Typical variation = 1 R1 neighbor out of 6 = 0.167')
log(f'    RF effect of 1 extra R1 neighbor: {rf_coef * (1/6):.4f} ({rf_coef * (1/6)*100:.2f}pp)')
log(f'    This is the REAL number -- the reduced form scaled to practical units.')

# ==============================================================================
# D9. SAME REDUCED FORM, NOT SCALED BY DIFFERENT FIRST STAGES
# ==============================================================================

log('\n' + '=' * 90)
log('D9. REDUCED FORM COMPARISON (the ONLY fair comparison)')
log('=' * 90)
log(f'  The IV coefficient is RF/FS. Scripts 20 and 26 have different FS')
log(f'  (because different endogenous variables), so IV coefficients are NOT')
log(f'  comparable. The REDUCED FORM is the fair comparison:')
log(f'  "Does having R1-winning neighbors predict MY adoption?"\n')

for y_col, y_label in [('IS_R3_ADOPTER', 'CSBP R3'), ('new_r3_first', 'WRI first-ever'), ('new_r3_any', 'WRI any-event')]:
    y = est[y_col].astype(float)
    X = pd.concat([X_exog, Z], axis=1)
    X = X.loc[:, X.std() > 0]
    idx2 = X.columns.get_loc('w_IV_Z_R1')

    ols_c = sm.OLS(y.values, X.astype(float).values).fit(
        cov_type='cluster', cov_kwds={'groups': est['state'].values}
    )
    c = ols_c.params[idx2]
    s = ols_c.bse[idx2]
    p = ols_c.pvalues[idx2]

    practical = c * (1/6)  # effect of 1 extra R1 neighbor

    log(f'  Y = {y_label:15s}: RF = {c:>9.5f} (SE={s:.5f}) p={p:.4f}{stars(p)}'
        f'  | 1-neighbor effect: {practical:.5f} ({practical*100:.3f}pp)')

# ==============================================================================
# SUMMARY
# ==============================================================================

log('\n' + '=' * 90)
log('SUMMARY: DIAGNOSING THE +1.343 vs ~0.00 DISCREPANCY')
log('=' * 90)

log("""
  The +1.343 vs ~0.00 discrepancy has multiple contributing factors:

  1. DIFFERENT CAUSAL PARAMETERS: The two scripts estimate different things.
     Script 20: effect of neighbor CSBP adoption on own CSBP adoption
     Script 26: effect of neighbor ALL-SOURCE adoption on own ALL-SOURCE adoption
     These are fundamentally different parameters.

  2. COEFFICIENT SCALING: +1.343 >> 1.0 likely reflects the Wald ratio
     with a moderate first stage (F=35). The REDUCED FORM -- the only
     directly comparable object -- tells the real story.

  3. OUTCOME COVERAGE: WRI bus data may not capture all CSBP R3 winners
     if there's a lag between rebate award and bus order. Check D1 output.

  4. THE FAIR COMPARISON is the REDUCED FORM:
     "Does having R1-winning neighbors predict MY [outcome]?"
     - RF -> CSBP R3: the answer (see D9)
     - RF -> all-source: the answer (see D9)
     If RF for CSBP is significant but RF for all-source is null,
     the peer effect genuinely works through CSBP only.
     If RF for all-source INCLUDES CSBP adopters and is still null,
     that's concerning -- it means even the CSBP channel vanishes
     when embedded in the full population.

  5. OWN-TREATMENT CONTROLS: Script 20 has NO IS_APPLICANT or
     pre_r1_adopter controls. This matters for identification.
""")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(str(TABLES_DIR / 'cross_round_diagnostics.csv'), index=False)
log(f'\nSaved: {TABLES_DIR / "cross_round_diagnostics.csv"}')

with open(str(LOGS_DIR / 'cross_round_diagnostics.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f'Saved: {LOGS_DIR / "cross_round_diagnostics.txt"}')

log('\nDONE')
