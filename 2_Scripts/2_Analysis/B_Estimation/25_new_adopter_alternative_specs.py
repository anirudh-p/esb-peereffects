"""
25_new_adopter_alternative_specs.py
===================================
Alternative specifications using WRI bus-level timing and all funding sources.

User-selected construction choices:
- Merge key: 1c. LEA ID
- Timing column: 3p. Quarter awarded  (NOTE: 3q. Quarter ordered has 55% NaN; award date is 99.95% complete)
- Scope: all funding sources in WRI bus-level data
- Two dependent variables for robustness:
    (1) new_first_window: district's first-ever ESB order falls in window
    (2) new_any_window: district has any ESB order in window

Window used in this script:
- 2022 to 2024 inclusive (aligned to current CSBP study horizon)

Outputs:
- 3_Output/Tables/new_adopter_alternative_specs.csv
- 3_Output/Tables/new_adopter_source_mix_window.csv
- 3_Output/Logs/new_adopter_alternative_specs.txt
"""

import pandas as pd
import numpy as np
import warnings
import re
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from _estimation_utils import (
    load_analysis_data, load_shapefile_and_merge,
    build_geo_knn, FULL_CONTROLS, significance_stars
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import WRI_EXCEL_FILE, TABLES_DIR, LOGS_DIR, ensure_dirs_exist

import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from libpysal.weights import lag_spatial

ensure_dirs_exist()

OUTPUT_CSV = TABLES_DIR / 'new_adopter_alternative_specs.csv'
OUTPUT_SOURCE_CSV = TABLES_DIR / 'new_adopter_source_mix_window.csv'
OUTPUT_LOG = LOGS_DIR / 'new_adopter_alternative_specs.txt'

WINDOW_START = 2022
WINDOW_END = 2024
K = 6

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(msg)


def clean_lea_id(series):
    return series.astype(str).str.split('.').str[0].str.zfill(7)


def extract_year_from_quarter(series):
    years = series.astype(str).str.extract(r'(\d{4})', expand=False)
    return pd.to_numeric(years, errors='coerce')


def categorize_source(text):
    txt = (str(text) if pd.notna(text) else '').strip().lower()

    if txt == '' or txt == 'nan':
        return 'unknown'

    if ('clean school bus' in txt) or ('csbp' in txt) or ('epa rebate' in txt) or ('epa grant' in txt):
        return 'federal_csbp'

    if ('volkswagen' in txt) or ('vw settlement' in txt) or ('vw trust' in txt):
        return 'vw_settlement'

    if ('hvip' in txt) or ('carl moyer' in txt):
        return 'state_program'

    if ('state' in txt) or ('department of education' in txt) or ('dept. of education' in txt):
        return 'state_program'

    if ('federal' in txt) or ('usda' in txt) or ('dot' in txt) or ('doe' in txt):
        return 'federal_other'

    if ('utility' in txt) or ('local' in txt) or ('district' in txt) or ('county' in txt) or ('city' in txt):
        return 'utility_local'

    return 'other_or_mixed'


log('=' * 80)
log('ALTERNATIVE SPEC: NEW ADOPTERS (ALL-SOURCE WRI BUS-LEVEL)')
log('=' * 80)

# ============================================================================
# 1) BUILD ALTERNATIVE OUTCOMES FROM WRI BUS-LEVEL DATA
# ============================================================================

log('\nLoading WRI bus-level sheet...')
df_bus = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='2. Bus-level data')

required_cols = [
    '1c. LEA ID',
    '3p. Quarter awarded',
    '3z. Funding source 1', '3aa. Agency administering funds 1',
    '3z. Funding source 2', '3aa. Agency administering funds 2',
    '3z. Funding source 3', '3aa. Agency administering funds 3',
    '3z. Funding source 4', '3aa. Agency administering funds 4',
]
missing_cols = [c for c in required_cols if c not in df_bus.columns]
if missing_cols:
    raise ValueError(f'Missing expected columns in bus-level sheet: {missing_cols}')

df_bus = df_bus[required_cols].copy()
df_bus['nces_id'] = clean_lea_id(df_bus['1c. LEA ID'])
# Use award date (3p) not order date (3q) -- 3q is 55% NaN; 3p is 99.95% complete
df_bus['award_year'] = extract_year_from_quarter(df_bus['3p. Quarter awarded'])

# Construct a text field combining up to 4 funding source + agency pairs
source_parts = []
for idx in [1, 2, 3, 4]:
    source_col = f'3z. Funding source {idx}'
    agency_col = f'3aa. Agency administering funds {idx}'
    source_parts.append(df_bus[source_col].fillna('').astype(str))
    source_parts.append(df_bus[agency_col].fillna('').astype(str))

combined_source_text = source_parts[0]
for s in source_parts[1:]:
    combined_source_text = combined_source_text + ' | ' + s

df_bus['source_category'] = combined_source_text.apply(categorize_source)

valid_year = df_bus['award_year'].between(1990, 2035, inclusive='both')
df_bus_valid = df_bus[valid_year & df_bus['nces_id'].ne('0000000')].copy()

log(f'  Bus-level rows total: {len(df_bus):,}')
log(f'  Bus-level rows with valid award year + LEA ID: {len(df_bus_valid):,}')

# District-level first-ever year and any-event indicator in window
first_year = df_bus_valid.groupby('nces_id')['award_year'].min().rename('first_award_year').reset_index()

window_rows = df_bus_valid[
    (df_bus_valid['award_year'] >= WINDOW_START) &
    (df_bus_valid['award_year'] <= WINDOW_END)
].copy()

any_window = window_rows.groupby('nces_id').size().rename('any_event_count_window').reset_index()

outcomes = first_year.merge(any_window, on='nces_id', how='left')
outcomes['any_event_count_window'] = outcomes['any_event_count_window'].fillna(0)

outcomes['new_first_window'] = (
    (outcomes['first_award_year'] >= WINDOW_START) &
    (outcomes['first_award_year'] <= WINDOW_END)
).astype(int)

outcomes['new_any_window'] = (outcomes['any_event_count_window'] > 0).astype(int)
outcomes['repeat_event_window'] = (outcomes['any_event_count_window'] > 1).astype(int)
outcomes['pre_window_adopter'] = (outcomes['first_award_year'] < WINDOW_START).astype(int)

log('\nConstructed district-level outcomes:')
log(f"  Districts with first-ever order in {WINDOW_START}-{WINDOW_END}: {int(outcomes['new_first_window'].sum()):,}")
log(f"  Districts with any order in {WINDOW_START}-{WINDOW_END}: {int(outcomes['new_any_window'].sum()):,}")
log(f"  Districts with multiple orders in window: {int(outcomes['repeat_event_window'].sum()):,}")
log(f"  Districts with pre-window ESB adoption:    {int(outcomes['pre_window_adopter'].sum()):,}")

# Source mix among districts with activity in window
source_window = window_rows[['nces_id', 'source_category']].drop_duplicates()
source_mix = (
    source_window
    .groupby('source_category')['nces_id']
    .nunique()
    .rename('district_count')
    .reset_index()
    .sort_values('district_count', ascending=False)
)
source_mix['share_of_active_districts'] = source_mix['district_count'] / source_mix['district_count'].sum()
source_mix.to_csv(str(OUTPUT_SOURCE_CSV), index=False)
log(f"\nSaved source mix table: {OUTPUT_SOURCE_CSV}")

# ============================================================================
# 2) LOAD BASE ANALYSIS DATA AND MERGE OUTCOMES
# ============================================================================

df_base, locale_vars = load_analysis_data()
df_base['nces_id'] = clean_lea_id(df_base['nces_id'])

df_analysis = df_base.merge(
    outcomes[['nces_id', 'new_first_window', 'new_any_window', 'repeat_event_window', 'pre_window_adopter']],
    on='nces_id',
    how='left'
)

for c in ['new_first_window', 'new_any_window', 'repeat_event_window', 'pre_window_adopter']:
    df_analysis[c] = df_analysis[c].fillna(0).astype(int)

log(f"\nBase analysis rows: {len(df_analysis):,}")
log(f"  Mean(new_first_window):   {df_analysis['new_first_window'].mean():.4f}")
log(f"  Mean(new_any_window):     {df_analysis['new_any_window'].mean():.4f}")
log(f"  Mean(pre_window_adopter): {df_analysis['pre_window_adopter'].mean():.4f}")
log(f"  Own CSBP winners (IV_Z=1): {int(df_analysis['IV_Z'].sum()):,}")
log(f"  Own CSBP applicants:       {int(df_analysis['IS_APPLICANT'].sum()):,}")

geo_df = load_shapefile_and_merge(df_analysis)
log(f"  Spatial merged rows: {len(geo_df):,}")

# ============================================================================
# 3) BUILD SPATIAL LAGS / IV INPUTS
# ============================================================================

w = build_geo_knn(geo_df, K)

geo_df['w_IV_Z'] = lag_spatial(w, geo_df['IV_Z'].fillna(0).values)
geo_df['w_new_first'] = lag_spatial(w, geo_df['new_first_window'].values)
geo_df['w_new_any'] = lag_spatial(w, geo_df['new_any_window'].values)

geo_df['IS_LOSER'] = ((geo_df['IS_APPLICANT'] == 1) & (geo_df['IV_Z'] == 0)).astype(int)
geo_df['w_loser'] = lag_spatial(w, geo_df['IS_LOSER'].values)

state_dummies = pd.get_dummies(geo_df['state'], prefix='st', drop_first=True)
geo_df = pd.concat([geo_df.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
state_fe_cols = list(state_dummies.columns)

# Own-district controls to isolate the peer channel:
# - IV_Z: own CSBP lottery winner status (direct adoption channel)
# - IS_APPLICANT: own CSBP application (unobserved ESB enthusiasm)
# - pre_window_adopter: had ESB orders before study window
own_controls = ['IV_Z', 'IS_APPLICANT', 'pre_window_adopter']
base_controls = FULL_CONTROLS + locale_vars + own_controls

results = []


def run_iv(label, y_col, endog_col, extra_controls=None):
    controls = list(base_controls) + (extra_controls or [])
    needed = [y_col, endog_col, 'w_IV_Z', 'state'] + controls + state_fe_cols
    df_r = geo_df.dropna(subset=list(set(needed))).copy()

    y = df_r[y_col].astype(float)
    X_endog = df_r[[endog_col]].astype(float)
    X_exog_raw = sm.add_constant(df_r[controls + state_fe_cols].astype(float))
    X_exog = X_exog_raw.loc[:, X_exog_raw.std() > 0]
    Z = df_r[['w_IV_Z']].astype(float)

    fs_X = pd.concat([X_exog, Z], axis=1)
    fs = sm.OLS(X_endog.values.ravel(), fs_X.astype(float).values).fit()
    idx_z = fs_X.columns.get_loc('w_IV_Z')
    f_stat = fs.tvalues[idx_z] ** 2

    iv = IV2SLS(dependent=y, exog=X_exog, endog=X_endog, instruments=Z).fit(
        cov_type='clustered', clusters=df_r['state']
    )

    coef = iv.params[endog_col]
    se = iv.std_errors[endog_col]
    pval = iv.pvalues[endog_col]

    results.append({
        'spec': label,
        'outcome': y_col,
        'endogenous_peer_var': endog_col,
        'controls_added': ','.join(extra_controls or []),
        'coef': coef,
        'se': se,
        'pval': pval,
        'stars': significance_stars(pval),
        'F_first': f_stat,
        'N': len(df_r),
        'n_states': df_r['state'].nunique(),
        'outcome_mean': df_r[y_col].mean(),
    })

    log(
        f"  {label:40s} | coef={coef:>8.4f} (SE={se:.4f}) p={pval:.4f}{significance_stars(pval)}"
        f" | F={f_stat:.1f} | N={len(df_r)}"
    )


def run_rf(label, y_col, extra_controls=None):
    controls = list(base_controls) + (extra_controls or [])
    needed = [y_col, 'w_IV_Z', 'state'] + controls + state_fe_cols
    df_r = geo_df.dropna(subset=list(set(needed))).copy()

    X = sm.add_constant(
        pd.concat([df_r[controls + state_fe_cols], df_r[['w_IV_Z']]], axis=1).astype(float)
    )

    idx_z = X.columns.get_loc('w_IV_Z')
    ols = sm.OLS(df_r[y_col].astype(float).values, X.values).fit(
        cov_type='cluster', cov_kwds={'groups': df_r['state'].values}
    )

    coef = ols.params[idx_z]
    se = ols.bse[idx_z]
    pval = ols.pvalues[idx_z]

    results.append({
        'spec': label,
        'outcome': y_col,
        'endogenous_peer_var': 'RF',
        'controls_added': ','.join(extra_controls or []),
        'coef': coef,
        'se': se,
        'pval': pval,
        'stars': significance_stars(pval),
        'F_first': np.nan,
        'N': len(df_r),
        'n_states': df_r['state'].nunique(),
        'outcome_mean': df_r[y_col].mean(),
    })

    log(
        f"  {label:40s} | coef={coef:>8.4f} (SE={se:.4f}) p={pval:.4f}{significance_stars(pval)}"
        f" | N={len(df_r)}"
    )


log('\n' + '=' * 80)
log('IV RESULTS: NEW-ADOPTER OUTCOMES')
log('=' * 80)

# Outcome 1: first-ever in window
run_iv('A1. First-ever window baseline', 'new_first_window', 'w_new_first', extra_controls=None)
run_iv('A2. First-ever + w_loser', 'new_first_window', 'w_new_first', extra_controls=['w_loser'])

# Outcome 2: any event in window
run_iv('B1. Any-event window baseline', 'new_any_window', 'w_new_any', extra_controls=None)
run_iv('B2. Any-event + w_loser', 'new_any_window', 'w_new_any', extra_controls=['w_loser'])

log('\n' + '=' * 80)
log('REDUCED FORM (ITT) FOR NEW-ADOPTER OUTCOMES')
log('=' * 80)

run_rf('C1. RF first-ever baseline', 'new_first_window', extra_controls=None)
run_rf('C2. RF first-ever + w_loser', 'new_first_window', extra_controls=['w_loser'])
run_rf('D1. RF any-event baseline', 'new_any_window', extra_controls=None)
run_rf('D2. RF any-event + w_loser', 'new_any_window', extra_controls=['w_loser'])

# ============================================================================
# 4) SAVE
# ============================================================================

results_df = pd.DataFrame(results)
results_df.to_csv(str(OUTPUT_CSV), index=False)
log(f"\nSaved estimation results: {OUTPUT_CSV}")

with open(str(OUTPUT_LOG), 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"Saved log: {OUTPUT_LOG}")

log('\n' + '=' * 80)
log('DONE')
log('=' * 80)
