"""
26b_bad_control_test.py
========================
Test whether IS_APPLICANT is a "bad control" in the cross-round design.

THE BAD CONTROL PROBLEM:
  In the cross-round design, the causal chain is:
    Neighbor R1 win (2022) -> deploy ESBs -> I observe -> I learn about CSBP ->
    I APPLY for R3 -> I win R3 (random) -> I adopt

  IS_APPLICANT is pooled across ALL rounds (R1+R2+R3).
  If a district applied for R3 BECAUSE their neighbor won R1, then
  IS_APPLICANT captures the MEDIATOR of the peer effect.
  Controlling for it blocks the very channel we're trying to identify.

  This script runs three specifications:
    (A) Original:        IS_APPLICANT + pre_r1_adopter  (reproduces null)
    (B) No app control:  pre_r1_adopter only            (removes bad control)
    (C) Predetermined:   IS_R1_APPLICANT + pre_r1_adopter (safe control)

  IS_R1_APPLICANT = 1 if district applied in Round 1 (predetermined relative to
  R1 lottery outcome). R3 application is post-treatment, so excluded.

Outputs:
  - 3_Output/Tables/bad_control_test.csv
  - 3_Output/Logs/bad_control_test.txt
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

def significance_stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

log('=' * 80)
log('BAD CONTROL TEST: IS_APPLICANT AS POST-TREATMENT MEDIATOR')
log('=' * 80)

# ==============================================================================
# 1. CONSTRUCT R1 INSTRUMENT FROM RAW CSBP DATA
# ==============================================================================

log('\n--- Constructing R1 lottery instrument ---')

rebates = pd.read_excel(RAW_WRI / "CSB_Rebates.xlsx")
rebates['nces_id'] = clean_lea_id(rebates['NCES District ID'])
excluded = ['WITHDRAWN', 'CANCELLED', 'NOT SELECTED', 'INELIGIBLE', 'DENIED']
rebates = rebates[~rebates['Project Status'].str.upper().isin(excluded)]

r1_winners = rebates[rebates['Funding Year'] == 2022][['nces_id']].drop_duplicates()
r1_winners['IV_Z_R1'] = 1

log(f'  R1 lottery winners: {len(r1_winners):,}')

# ==============================================================================
# 2. CONSTRUCT ROUND-SPECIFIC APPLICANT INDICATORS
# ==============================================================================

log('\n--- Building round-specific applicant indicators ---')

df_applicants = pd.read_excel(str(APPLICANT_FILE))

# Identify the LEA ID column
id_candidates = [c for c in df_applicants.columns if 'LEA' in c.upper() or 'NCES' in c.upper() or 'ID' in c.upper()]
log(f'  Applicant file columns: {list(df_applicants.columns)}')
log(f'  ID candidate columns: {id_candidates}')

# Find round column
round_candidates = [c for c in df_applicants.columns if 'round' in c.lower()]
log(f'  Round candidate columns: {round_candidates}')

# Use the standard ID column name from prep script
id_col_app = None
for c in df_applicants.columns:
    if 'NCES' in c.upper() and 'ID' in c.upper():
        id_col_app = c
        break
if id_col_app is None:
    for c in df_applicants.columns:
        if 'LEA' in c.upper() and 'ID' in c.upper():
            id_col_app = c
            break
if id_col_app is None:
    id_col_app = df_applicants.columns[0]
    log(f'  WARNING: Could not find ID column, using first column: {id_col_app}')

log(f'  Using ID column: {id_col_app}')

df_applicants['nces_id'] = clean_lea_id(df_applicants[id_col_app])
df_applicants['Round'] = df_applicants['Round'].astype(str).str.strip()

log(f'  Total applicant rows: {len(df_applicants):,}')
log(f'  Unique applicant districts: {df_applicants["nces_id"].nunique():,}')
log(f'  Round values: {df_applicants["Round"].value_counts().to_dict()}')

# Round-specific indicators
# IS_R1_APPLICANT: applied in Round 1 - PREDETERMINED (safe control)
# IS_R3_APPLICANT: applied in Round 3 - POST-TREATMENT MEDIATOR (bad control)
r1_applicants = df_applicants[df_applicants['Round'].str.contains('1', na=False)]['nces_id'].unique()
r3_applicants = df_applicants[df_applicants['Round'].str.contains('3', na=False)]['nces_id'].unique()
all_applicants = df_applicants['nces_id'].unique()

log(f'  R1 applicants: {len(r1_applicants):,}')
log(f'  R3 applicants: {len(r3_applicants):,}')
log(f'  All applicants: {len(all_applicants):,}')

# Districts that applied in R3 but NOT R1 -- these are the "potentially treated" by peer exposure
r3_only = set(r3_applicants) - set(r1_applicants)
log(f'  R3-only applicants (new entrants, potential peer effect channel): {len(r3_only):,}')

# ==============================================================================
# 3. CONSTRUCT ALL-SOURCE OUTCOMES FROM WRI BUS-LEVEL DATA
# ==============================================================================

log('\n--- Building all-source outcomes from WRI bus-level data ---')

df_bus = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='2. Bus-level data')
df_bus['nces_id'] = clean_lea_id(df_bus['1c. LEA ID'])
df_bus['order_year'] = extract_year_from_quarter(df_bus['3q. Quarter ordered'])

valid = df_bus['order_year'].between(1990, 2035) & df_bus['nces_id'].ne('0000000')
df_bus = df_bus[valid].copy()

log(f'  Bus-level rows with valid year+LEA: {len(df_bus):,}')

# District-level earliest order year
first_year = df_bus.groupby('nces_id')['order_year'].min().rename('first_order_year').reset_index()
first_year['pre_r1_adopter'] = (first_year['first_order_year'] < 2022).astype(int)

# R3-window outcomes (2023-2024)
r3_window = df_bus[df_bus['order_year'].between(2023, 2024)].copy()
r3_any = r3_window.groupby('nces_id').size().rename('r3_bus_count').reset_index()

outcomes = first_year.merge(r3_any, on='nces_id', how='left')
outcomes['r3_bus_count'] = outcomes['r3_bus_count'].fillna(0)

outcomes['new_r3_first'] = (
    (outcomes['first_order_year'] >= 2023) &
    (outcomes['first_order_year'] <= 2024)
).astype(int)
outcomes['new_r3_any'] = (outcomes['r3_bus_count'] > 0).astype(int)

log(f'  Districts with first-ever ESB in 2023-2024: {outcomes["new_r3_first"].sum():,}')
log(f'  Districts with any ESB in 2023-2024: {outcomes["new_r3_any"].sum():,}')
log(f'  Districts with pre-R1 adoption: {outcomes["pre_r1_adopter"].sum():,}')

# ==============================================================================
# 4. LOAD MAIN DATASET AND MERGE
# ==============================================================================

log('\n--- Loading base analysis dataset ---')

df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Merge R1 status
df = df.merge(r1_winners, on='nces_id', how='left')
df['IV_Z_R1'] = df['IV_Z_R1'].fillna(0).astype(int)

# Merge all-source outcomes
df = df.merge(
    outcomes[['nces_id', 'new_r3_first', 'new_r3_any', 'pre_r1_adopter']],
    on='nces_id', how='left'
)
for c in ['new_r3_first', 'new_r3_any', 'pre_r1_adopter']:
    df[c] = df[c].fillna(0).astype(int)

# Merge round-specific applicant indicators
df['IS_R1_APPLICANT'] = df['nces_id'].isin(r1_applicants).astype(int)
df['IS_R3_APPLICANT'] = df['nces_id'].isin(r3_applicants).astype(int)
# Pooled IS_APPLICANT for comparison
df['IS_APPLICANT_pool'] = df['nces_id'].isin(all_applicants).astype(int)

log(f'  Total districts: {len(df):,}')
log(f'  R1 winners: {df["IV_Z_R1"].sum():,}')
log(f'  IS_APPLICANT (pooled): {df["IS_APPLICANT_pool"].sum():,}')
log(f'  IS_R1_APPLICANT: {df["IS_R1_APPLICANT"].sum():,}')
log(f'  IS_R3_APPLICANT: {df["IS_R3_APPLICANT"].sum():,}')

# Cross-tab: IS_R3_APPLICANT among non-R1-winners by neighbor R1 exposure
df_nonr1 = df[df['IV_Z_R1'] == 0].copy()
log(f'\n  --- R3 application as OUTCOME (should respond to peer exposure) ---')
log(f'  Among non-R1-winners:')
log(f'    IS_R3_APPLICANT rate: {df_nonr1["IS_R3_APPLICANT"].mean():.4f}')

# ==============================================================================
# 5. PREPARE ESTIMATION SAMPLE
# ==============================================================================

# Exclude R1 winners
df_est = df[df['IV_Z_R1'] == 0].copy()

log(f'\n--- Estimation sample (non-R1-winners): {len(df_est):,} ---')
log(f'  R3-window first-ever rate: {df_est["new_r3_first"].mean():.4f}')
log(f'  R3-window any-event rate:  {df_est["new_r3_any"].mean():.4f}')

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
lea_county['nces_id'] = clean_lea_id(lea_county['nces_id'])
lea_county = lea_county.merge(county_pol[['county_fips', 'pct_dem_2020']], on='county_fips', how='left')
lea_political = lea_county.groupby('nces_id')['pct_dem_2020'].mean().reset_index()

df_est = df_est.merge(lea_political, on='nces_id', how='left')

if 'urbanicity' in df_est.columns:
    locale_dummies = pd.get_dummies(df_est['urbanicity'], prefix='locale', drop_first=True)
    df_est = pd.concat([df_est.reset_index(drop=True), locale_dummies.reset_index(drop=True)], axis=1)
    locale_vars = [c for c in df_est.columns if c.startswith('locale_')]
else:
    locale_vars = []

# Standard controls (NO own-treatment controls yet -- those vary by spec)
standard_controls = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pm25',
                     'pct_white', 'pct_dem_2020', 'is_priority'] + locale_vars

# ==============================================================================
# 6. SPATIAL WEIGHTS + ESTIMATION
# ==============================================================================

log('\n--- Building spatial weights on full network ---')

# Full dataset for weight construction (including R1 winners as neighbors)
df_full = df.merge(lea_political, on='nces_id', how='left')
if 'urbanicity' in df_full.columns:
    ld_f = pd.get_dummies(df_full['urbanicity'], prefix='locale', drop_first=True)
    df_full = pd.concat([df_full.reset_index(drop=True), ld_f.reset_index(drop=True)], axis=1)

gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
shp_id = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]
gdf[shp_id] = gdf[shp_id].astype(str).str.zfill(7)

gdf_m = gdf.merge(df_full[['nces_id']].drop_duplicates(), left_on=shp_id, right_on='nces_id', how='inner')
gdf_m = gdf_m.to_crs(epsg=5070).drop(columns=['nces_id'])
geo_all = gdf_m.merge(df_full, left_on=shp_id, right_on='nces_id', how='inner').reset_index(drop=True)
geo_all['centroid'] = geo_all.geometry.centroid
geo_pts = geo_all.set_geometry('centroid')

log(f'  Full spatial sample: {len(geo_all):,}')

results = []

# Focus on K=6 (main spec) and K=4,8 for robustness
for K in [4, 6, 8, 10]:
    log(f'\n{"=" * 80}')
    log(f'K = {K}')
    log(f'{"=" * 80}')

    w = KNN.from_dataframe(geo_pts, k=K)
    w.transform = 'r'

    # Spatial lags on full network
    geo_all['w_IV_Z_R1'] = lag_spatial(w, geo_all['IV_Z_R1'].fillna(0).values)
    geo_all['w_r3_first'] = lag_spatial(w, geo_all['new_r3_first'].values)
    geo_all['w_r3_any'] = lag_spatial(w, geo_all['new_r3_any'].values)

    # Loser density suppressor
    geo_all['IS_LOSER'] = ((geo_all['IS_APPLICANT_pool'] == 1) & (geo_all['IV_Z_R1'] == 0) & (geo_all['IV_Z'] == 0)).astype(int)
    geo_all['w_loser'] = lag_spatial(w, geo_all['IS_LOSER'].values)

    # Also: spatial lag of R3 application (for testing R3 app as outcome)
    geo_all['w_R3_app'] = lag_spatial(w, geo_all['IS_R3_APPLICANT'].values)

    # Restriction to estimation sample
    est = geo_all[geo_all['IV_Z_R1'] == 0].copy().reset_index(drop=True)
    ctrl_avail = [c for c in standard_controls if c in est.columns]
    est = est.dropna(subset=ctrl_avail + ['state']).copy()

    state_dummies = pd.get_dummies(est['state'], prefix='st', drop_first=True)
    est = pd.concat([est.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
    state_fe_cols = list(state_dummies.columns)

    log(f'  Estimation sample: {len(est):,}')
    log(f'  w_IV_Z_R1 range: [{est["w_IV_Z_R1"].min():.4f}, {est["w_IV_Z_R1"].max():.4f}]')

    # -------------------------------------------------------------------------
    # 6a. IS_R3_APPLICANT as OUTCOME (stage 0 test)
    # -------------------------------------------------------------------------
    if K == 6:
        log(f'\n  --- R3 APPLICATION AS OUTCOME (peer-induced application channel) ---')
        for ctrl_label, extra_ctrl in [('no app control', []), ('+ IS_R1_APP', ['IS_R1_APPLICANT'])]:
            controls = ctrl_avail + ['pre_r1_adopter'] + extra_ctrl
            X = sm.add_constant(
                pd.concat([est[controls + state_fe_cols], est[['w_IV_Z_R1']]], axis=1).astype(float)
            )
            X = X.loc[:, X.std() > 0]
            idx_z = X.columns.get_loc('w_IV_Z_R1')

            ols = sm.OLS(est['IS_R3_APPLICANT'].astype(float).values, X.values).fit(
                cov_type='cluster', cov_kwds={'groups': est['state'].values}
            )
            coef = ols.params[idx_z]
            se = ols.bse[idx_z]
            pval = ols.pvalues[idx_z]

            log(f'  Y=IS_R3_APP {ctrl_label:25s} | coef={coef:>8.4f} (SE={se:.4f}) p={pval:.4f}{significance_stars(pval)}'
                f' | N={len(est)}')

    # -------------------------------------------------------------------------
    # 6b. MAIN COMPARISON: Three control specifications
    # -------------------------------------------------------------------------
    spec_configs = [
        ('(A) IS_APPLICANT+pre_r1', ['IS_APPLICANT_pool', 'pre_r1_adopter']),
        ('(B) pre_r1 only',         ['pre_r1_adopter']),
        ('(C) IS_R1_APP+pre_r1',    ['IS_R1_APPLICANT', 'pre_r1_adopter']),
    ]

    for outcome_label, y_col, endog_col in [
        ('First-ever R3', 'new_r3_first', 'w_r3_first'),
        ('Any-event R3', 'new_r3_any', 'w_r3_any'),
    ]:
        for ctrl_label, extra_ctrl in [('baseline', []), ('+ w_loser', ['w_loser'])]:
            for spec_name, own_ctrl in spec_configs:
                controls = ctrl_avail + own_ctrl + extra_ctrl
                controls = [c for c in controls if c in est.columns]
                tag = f'K={K} {outcome_label} {ctrl_label} {spec_name}'

                y = est[y_col].astype(float)
                X_endog = est[[endog_col]].astype(float)
                Z = est[['w_IV_Z_R1']].astype(float)

                X_exog_raw = sm.add_constant(est[controls + state_fe_cols].astype(float))
                X_exog = X_exog_raw.loc[:, X_exog_raw.std() > 0]

                # First stage
                fs_X = pd.concat([X_exog, Z], axis=1)
                fs = sm.OLS(X_endog.values.ravel(), fs_X.astype(float).values).fit()
                idx_z = fs_X.columns.get_loc('w_IV_Z_R1')
                f_stat = fs.tvalues[idx_z] ** 2

                # IV-2SLS
                try:
                    iv = IV2SLS(dependent=y, exog=X_exog, endog=X_endog, instruments=Z).fit(
                        cov_type='clustered', clusters=est['state']
                    )
                    coef = iv.params[endog_col]
                    se = iv.std_errors[endog_col]
                    pval = iv.pvalues[endog_col]
                except Exception as e:
                    log(f'  ERROR {tag}: {e}')
                    coef, se, pval = np.nan, np.nan, np.nan

                results.append({
                    'K': K, 'outcome': outcome_label, 'w_loser': ctrl_label,
                    'spec': spec_name, 'coef': coef, 'se': se, 'pval': pval,
                    'stars': significance_stars(pval),
                    'F_first': f_stat, 'N': len(est),
                    'n_states': est['state'].nunique(),
                    'y_mean': est[y_col].mean(),
                })

                log(f'  {tag:60s} | coef={coef:>8.4f} (SE={se:.4f}) p={pval:.4f}{significance_stars(pval)}'
                    f' | F={f_stat:.1f}')

    # Reduced forms for K=6
    if K == 6:
        log(f'\n  --- REDUCED FORM (ITT) K=6 ---')
        for outcome_label, y_col in [('RF First-ever', 'new_r3_first'), ('RF Any-event', 'new_r3_any')]:
            for spec_name, own_ctrl in spec_configs:
                controls = ctrl_avail + own_ctrl
                controls = [c for c in controls if c in est.columns]
                tag = f'{outcome_label} {spec_name}'

                X = sm.add_constant(
                    pd.concat([est[controls + state_fe_cols], est[['w_IV_Z_R1']]], axis=1).astype(float)
                )
                X = X.loc[:, X.std() > 0]
                idx_z = X.columns.get_loc('w_IV_Z_R1')

                ols = sm.OLS(est[y_col].astype(float).values, X.values).fit(
                    cov_type='cluster', cov_kwds={'groups': est['state'].values}
                )
                coef = ols.params[idx_z]
                se = ols.bse[idx_z]
                pval = ols.pvalues[idx_z]

                results.append({
                    'K': K, 'outcome': outcome_label, 'w_loser': 'baseline',
                    'spec': spec_name, 'coef': coef, 'se': se, 'pval': pval,
                    'stars': significance_stars(pval),
                    'F_first': np.nan, 'N': len(est),
                    'n_states': est['state'].nunique(),
                    'y_mean': est[y_col].mean(),
                })

                log(f'  {tag:60s} | coef={coef:>8.4f} (SE={se:.4f}) p={pval:.4f}{significance_stars(pval)}')

# ==============================================================================
# 7. SUMMARY
# ==============================================================================

log('\n' + '=' * 80)
log('SUMMARY: BAD CONTROL DIAGNOSIS')
log('=' * 80)

log("""
KEY QUESTION: Does removing IS_APPLICANT restore the peer effect?

If (A) shows null but (B) and (C) show significant effects, this confirms
IS_APPLICANT was blocking the peer effect channel:

  Neighbor R1 win -> I learn about CSBP -> I APPLY for R3 (= IS_APPLICANT=1)
  -> I win R3 (random) -> I adopt through CSBP

Spec (C) is the correct specification: IS_R1_APPLICANT is predetermined
(applied before R1 lottery was drawn), so it controls for baseline ESB
enthusiasm without blocking the peer-induced R3 application channel.

COMPARISON WITH SCRIPT 20:
  Script 20 (CSBP-only, NO IS_APPLICANT): +1.343*** (F=35)
  If the all-source result also becomes significant without IS_APPLICANT,
  this confirms the effect works through the CSBP application channel.
""")

# ==============================================================================
# 8. SAVE
# ==============================================================================

results_df = pd.DataFrame(results)
results_df.to_csv(str(TABLES_DIR / 'bad_control_test.csv'), index=False)
log(f'\nSaved: {TABLES_DIR / "bad_control_test.csv"}')

with open(str(LOGS_DIR / 'bad_control_test.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f'Saved: {LOGS_DIR / "bad_control_test.txt"}')

log('\n' + '=' * 80)
log('DONE')
log('=' * 80)
