"""
26_cross_round_all_source.py
============================
Cross-round IV with ALL-SOURCE ESB outcomes (not just CSBP).

IDENTIFICATION:
  The cleanest peer-effect identification uses temporal ordering:
    - Neighbor won R1 lottery (2022) -> deployed ESBs -> I observe them ->
      I adopt ESBs in R3 window (2023-2024) through ANY funding channel.

  Instrument: w_IV_Z_R1 = share of K nearest neighbors who won R1 lottery (2022).
  Outcome:    Whether district i adopted ESBs from ANY source in 2023-2024.
  Sample:     All districts EXCEPT R1 winners (who received direct treatment).

  Own-treatment controls (exclusion restriction):
    - IS_APPLICANT: own CSBP application status (unobserved ESB enthusiasm)
    - pre_r1_adopter: had ESB orders before 2022 (baseline propensity)

  Why this is the identified peer effect:
    1. R1 lottery is exogenous (random assignment conditional on application)
    2. Temporal ordering: R1 deployments (2022) precede R3-window adoption
    3. Own R1 winners excluded -> no own-treatment contamination
    4. IS_APPLICANT absorbs geographic application clustering
    5. All-source outcome captures full behavioral response, not just CSBP

Outputs:
  - 3_Output/Tables/cross_round_all_source.csv
  - 3_Output/Tables/cross_round_all_source_detail.csv
  - 3_Output/Logs/cross_round_all_source.txt
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
log('CROSS-ROUND IV: R1 WINS -> ALL-SOURCE ESB ADOPTION (2023-2024)')
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
# 2. CONSTRUCT ALL-SOURCE OUTCOMES FROM WRI BUS-LEVEL DATA
# ==============================================================================

log('\n--- Building all-source outcomes from WRI bus-level data ---')

df_bus = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='2. Bus-level data')
df_bus['nces_id'] = clean_lea_id(df_bus['1c. LEA ID'])
# Use award date (3p) not order date (3q) -- 3q is 55% NaN; award date is 99.95% complete
# R3 CSBP buses were awarded Q2 2024; using order_year missed 92% of them
df_bus['award_year'] = extract_year_from_quarter(df_bus['3p. Quarter awarded'])

valid = df_bus['award_year'].between(1990, 2035) & df_bus['nces_id'].ne('0000000')
df_bus = df_bus[valid].copy()

log(f'  Bus-level rows with valid year+LEA: {len(df_bus):,}')

# District-level earliest award year
first_year = df_bus.groupby('nces_id')['award_year'].min().rename('first_award_year').reset_index()

# Pre-R1 adoption indicator (awarded buses before 2022)
first_year['pre_r1_adopter'] = (first_year['first_award_year'] < 2022).astype(int)

# R3-window outcomes (2023-2024): post-R1 deployment observation window
r3_window = df_bus[df_bus['award_year'].between(2023, 2024)].copy()

# First-ever in R3 window: district had NEVER ordered before 2023
r3_any = r3_window.groupby('nces_id').size().rename('r3_bus_count').reset_index()

outcomes = first_year.merge(r3_any, on='nces_id', how='left')
outcomes['r3_bus_count'] = outcomes['r3_bus_count'].fillna(0)

# Two outcomes:
# (a) new_r3_first: first-ever ESB order falls in 2023-2024
outcomes['new_r3_first'] = (
    (outcomes['first_award_year'] >= 2023) &
    (outcomes['first_award_year'] <= 2024)
).astype(int)

# (b) new_r3_any: any ESB order in 2023-2024 (includes repeat adopters)
outcomes['new_r3_any'] = (outcomes['r3_bus_count'] > 0).astype(int)

log(f'  Districts with first-ever ESB in 2023-2024: {outcomes["new_r3_first"].sum():,}')
log(f'  Districts with any ESB in 2023-2024: {outcomes["new_r3_any"].sum():,}')
log(f'  Districts with pre-R1 adoption: {outcomes["pre_r1_adopter"].sum():,}')

# ==============================================================================
# 3. LOAD MAIN DATASET AND MERGE
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

log(f'  Total districts: {len(df):,}')
log(f'  R1 winners: {df["IV_Z_R1"].sum():,}')
log(f'  Districts with R3-window first-ever: {df["new_r3_first"].sum():,}')
log(f'  Districts with R3-window any event: {df["new_r3_any"].sum():,}')

# ==============================================================================
# 4. EXCLUDE R1 WINNERS FROM OUTCOME SAMPLE
# ==============================================================================

# R1 winners received direct treatment -- cannot be in outcome sample
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

# Own-treatment controls:
# - IS_APPLICANT: CSBP application signals ESB enthusiasm
# - pre_r1_adopter: prior ESB experience from any source
own_controls = ['IS_APPLICANT', 'pre_r1_adopter']
base_controls = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pm25',
                 'pct_white', 'pct_dem_2020', 'is_priority'] + locale_vars + own_controls

# ==============================================================================
# 5. BUILD SPATIAL WEIGHTS ON FULL NETWORK (including R1 winners)
# ==============================================================================

log('\n--- Building spatial weights on full network ---')

# Need full dataset (including R1 winners) for weight construction
# so that R1 winners appear as neighbors
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

for K in [4, 6, 8, 10]:
    log(f'\n{"=" * 80}')
    log(f'K = {K}')
    log(f'{"=" * 80}')

    w = KNN.from_dataframe(geo_pts, k=K)
    w.transform = 'r'

    # Spatial lags computed on full network
    geo_all['w_IV_Z_R1'] = lag_spatial(w, geo_all['IV_Z_R1'].fillna(0).values)
    geo_all['w_r3_first'] = lag_spatial(w, geo_all['new_r3_first'].values)
    geo_all['w_r3_any'] = lag_spatial(w, geo_all['new_r3_any'].values)

    # Loser density for suppressor control
    geo_all['IS_LOSER'] = ((geo_all['IS_APPLICANT'] == 1) & (geo_all['IV_Z_R1'] == 0) & (geo_all['IV_Z'] == 0)).astype(int)
    geo_all['w_loser'] = lag_spatial(w, geo_all['IS_LOSER'].values)

    # Restrict to estimation sample (non-R1 winners with non-missing controls)
    est = geo_all[geo_all['IV_Z_R1'] == 0].copy().reset_index(drop=True)
    ctrl_available = [c for c in base_controls if c in est.columns]
    est = est.dropna(subset=ctrl_available + ['state']).copy()

    state_dummies = pd.get_dummies(est['state'], prefix='st', drop_first=True)
    est = pd.concat([est.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
    state_fe_cols = list(state_dummies.columns)

    log(f'  Estimation sample: {len(est):,}')
    log(f'  w_IV_Z_R1 range: [{est["w_IV_Z_R1"].min():.4f}, {est["w_IV_Z_R1"].max():.4f}]')
    log(f'  Mean(w_IV_Z_R1): {est["w_IV_Z_R1"].mean():.5f}')

    for outcome_label, y_col, endog_col in [
        ('First-ever R3', 'new_r3_first', 'w_r3_first'),
        ('Any-event R3', 'new_r3_any', 'w_r3_any'),
    ]:
        for ctrl_label, extra_ctrl in [('baseline', []), ('+ w_loser', ['w_loser'])]:
            controls = ctrl_available + extra_ctrl
            spec_label = f'K={K} {outcome_label} {ctrl_label}'

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
                log(f'  ERROR {spec_label}: {e}')
                coef, se, pval = np.nan, np.nan, np.nan

            results.append({
                'K': K, 'outcome': outcome_label, 'controls': ctrl_label,
                'coef': coef, 'se': se, 'pval': pval,
                'stars': significance_stars(pval),
                'F_first': f_stat, 'N': len(est),
                'n_states': est['state'].nunique(),
                'y_mean': est[y_col].mean(),
            })

            log(f'  {spec_label:45s} | coef={coef:>8.4f} (SE={se:.4f}) p={pval:.4f}{significance_stars(pval)}'
                f' | F={f_stat:.1f} | N={len(est)}')

    # Reduced form for K=6 only
    if K == 6:
        log(f'\n  --- Reduced Form (ITT) K=6 ---')
        for outcome_label, y_col in [('RF First-ever', 'new_r3_first'), ('RF Any-event', 'new_r3_any')]:
            for ctrl_label, extra_ctrl in [('baseline', []), ('+ w_loser', ['w_loser'])]:
                controls = ctrl_available + extra_ctrl
                spec_label = f'{outcome_label} {ctrl_label}'

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
                    'K': K, 'outcome': outcome_label, 'controls': ctrl_label,
                    'coef': coef, 'se': se, 'pval': pval,
                    'stars': significance_stars(pval),
                    'F_first': np.nan, 'N': len(est),
                    'n_states': est['state'].nunique(),
                    'y_mean': est[y_col].mean(),
                })

                log(f'  {spec_label:45s} | coef={coef:>8.4f} (SE={se:.4f}) p={pval:.4f}{significance_stars(pval)}'
                    f' | N={len(est)}')

# ==============================================================================
# 6. COMPARISON TABLE
# ==============================================================================

log('\n' + '=' * 80)
log('COMPARISON: CROSS-SECTION vs CROSS-ROUND, CSBP-ONLY vs ALL-SOURCE')
log('=' * 80)

log("""
  Instrument: w_IV_Z_R1 = share of K=6 neighbors who won R1 (2022) lottery.
  Sample: ALL districts excluding R1 winners.
  Own controls: IS_APPLICANT, pre_r1_adopter (+ full controls, state FE).

  Cross-section (Script 25):  instrument = pooled R1+R3 neighbor lottery status
  Cross-round (this script):  instrument = R1-only neighbor lottery status

  The cross-round design is the IDENTIFIED peer effect: temporal ordering
  ensures neighbor's R1 deployment preceded district i's adoption decision.
""")

# ==============================================================================
# 7. SAVE
# ==============================================================================

results_df = pd.DataFrame(results)
results_df.to_csv(str(TABLES_DIR / 'cross_round_all_source.csv'), index=False)
log(f'\nSaved: {TABLES_DIR / "cross_round_all_source.csv"}')

with open(str(LOGS_DIR / 'cross_round_all_source.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f'Saved: {LOGS_DIR / "cross_round_all_source.txt"}')

log('\n' + '=' * 80)
log('DONE')
log('=' * 80)
