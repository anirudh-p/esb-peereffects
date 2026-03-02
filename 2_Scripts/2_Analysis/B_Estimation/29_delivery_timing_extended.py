"""
29_delivery_timing_extended.py
==============================
Extended analysis of delivery timing peer effects:
  1. All-source adoption outcomes (not just CSBP)
  2. Geographic cluster analysis of negative late-delivery effect

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
import statsmodels.api as sm
import geopandas as gpd

ensure_dirs_exist()

log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

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
log('EXTENDED DELIVERY TIMING ANALYSIS')
log('1. All-source adoption outcomes')
log('2. Geographic cluster heterogeneity')
log('=' * 80)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

log('\n--- Loading WRI bus-level data ---')
df_bus = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='2. Bus-level data')
df_bus['nces_id'] = clean_lea_id(df_bus['1c. LEA ID'])
df_bus['award_date'] = extract_quarter_date(df_bus['3p. Quarter awarded'])
df_bus['delivery_date'] = extract_quarter_date(df_bus['3r. Quarter delivered'])
df_bus['award_year'] = df_bus['award_date'].dt.year

# R1 CSBP buses
r1_buses = df_bus[
    (df_bus['award_year'] == 2022) & 
    (df_bus['3z. Funding source 1'].str.contains('CLEAN SCHOOL BUS', na=False, case=False))
].copy()

log(f'R1 CSBP buses: {len(r1_buses):,}')

r3_deadline = pd.Timestamp('2023-10-01')

# District-level delivery timing
r1_district = r1_buses.groupby('nces_id').agg({
    'delivery_date': 'min'
}).reset_index()
r1_district.columns = ['nces_id', 'first_delivery']
r1_district['early_delivery'] = (r1_district['first_delivery'] < r3_deadline).fillna(False).astype(int)
r1_district['late_delivery'] = (~(r1_district['first_delivery'] < r3_deadline)).astype(int)

# R1 winners from rebates
rebates = pd.read_excel(RAW_WRI_DIR / "CSB_Rebates.xlsx")
rebates['nces_id'] = clean_lea_id(rebates['NCES District ID'])
excluded = ['WITHDRAWN', 'CANCELLED', 'NOT SELECTED', 'INELIGIBLE', 'DENIED']
rebates = rebates[~rebates['Project Status'].str.upper().isin(excluded)]

r1_winners = rebates[rebates['Funding Year'] == 2022][['nces_id']].drop_duplicates()
r1_winners['IV_Z_R1'] = 1
r1_winners = r1_winners.merge(r1_district[['nces_id', 'early_delivery', 'late_delivery']], 
                               on='nces_id', how='left')
r1_winners['early_delivery'] = r1_winners['early_delivery'].fillna(0).astype(int)
r1_winners['late_delivery'] = r1_winners['late_delivery'].fillna(1).astype(int)
r1_winners['R1_early'] = r1_winners['IV_Z_R1'] * r1_winners['early_delivery']
r1_winners['R1_late'] = r1_winners['IV_Z_R1'] * r1_winners['late_delivery']

log(f'R1 early-delivery: {r1_winners["R1_early"].sum()}')
log(f'R1 late-delivery: {r1_winners["R1_late"].sum()}')

# ==============================================================================
# 2. BUILD ALL-SOURCE OUTCOMES (from WRI bus data)
# ==============================================================================

log('\n--- Building all-source outcomes ---')

# Any bus awarded in 2023-2024 window
r3_window_bus = df_bus[df_bus['award_year'].between(2023, 2024)].copy()
first_year = df_bus.groupby('nces_id')['award_year'].min().rename('first_award_year').reset_index()

# Outcome: first-ever ESB in R3 window (2023-2024)
outcomes = first_year.copy()
outcomes['new_r3_first'] = ((outcomes['first_award_year'] >= 2023) & 
                            (outcomes['first_award_year'] <= 2024)).astype(int)

# R3 window adoption by any source
r3_any = r3_window_bus.groupby('nces_id').size().rename('r3_bus_count').reset_index()
outcomes = outcomes.merge(r3_any, on='nces_id', how='left')
outcomes['r3_bus_count'] = outcomes['r3_bus_count'].fillna(0)
outcomes['new_r3_any'] = (outcomes['r3_bus_count'] > 0).astype(int)

log(f'Districts with first-ever ESB in 2023-24: {outcomes["new_r3_first"].sum():,}')
log(f'Districts with any ESB in 2023-24: {outcomes["new_r3_any"].sum():,}')

# ==============================================================================
# 3. LOAD MAIN DATASET AND MERGE
# ==============================================================================

log('\n--- Loading main dataset ---')
df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Merge R1 status
df = df.merge(r1_winners[['nces_id', 'IV_Z_R1', 'R1_early', 'R1_late']], on='nces_id', how='left')
for c in ['IV_Z_R1', 'R1_early', 'R1_late']:
    df[c] = df[c].fillna(0).astype(int)

# Merge all-source outcomes
df = df.merge(outcomes[['nces_id', 'new_r3_first', 'new_r3_any']], on='nces_id', how='left')
df['new_r3_first'] = df['new_r3_first'].fillna(0).astype(int)
df['new_r3_any'] = df['new_r3_any'].fillna(0).astype(int)

log(f'Total districts: {len(df):,}')

# ==============================================================================
# 4. BUILD SPATIAL WEIGHTS AND SAMPLE
# ==============================================================================

log('\n--- Building spatial weights ---')
gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
shp_id = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]
gdf[shp_id] = gdf[shp_id].astype(str).str.zfill(7)

# Keep only needed columns to reduce memory
gdf = gdf[[shp_id, 'geometry']].copy()

gdf_m = gdf.merge(df[['nces_id']].drop_duplicates(), left_on=shp_id, right_on='nces_id', how='inner')
gdf_m = gdf_m.drop(columns=['nces_id'])

# Simplify geometries to reduce memory
gdf_m['geometry'] = gdf_m.geometry.simplify(tolerance=100)
gdf_m = gdf_m.to_crs(epsg=5070)

geo_all = gdf_m.merge(df, left_on=shp_id, right_on='nces_id', how='inner').reset_index(drop=True)
geo_all['centroid'] = geo_all.geometry.centroid
geo_pts = geo_all.set_geometry('centroid')

K = 6
w = KNN.from_dataframe(geo_pts, k=K)
w.transform = 'r'

# Spatial lags
geo_all['w_R1_early'] = lag_spatial(w, geo_all['R1_early'].values)
geo_all['w_R1_late'] = lag_spatial(w, geo_all['R1_late'].values)
geo_all['w_r3_first'] = lag_spatial(w, geo_all['new_r3_first'].values)
geo_all['w_r3_any'] = lag_spatial(w, geo_all['new_r3_any'].values)

# Estimation sample
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

results = []

# ==============================================================================
# PART 1: ALL-SOURCE ADOPTION OUTCOMES
# ==============================================================================

log('\n' + '=' * 80)
log('PART 1: ALL-SOURCE ADOPTION OUTCOMES')
log('=' * 80)

for outcome_name, y_col in [('First-ever R3', 'new_r3_first'), ('Any R3', 'new_r3_any')]:
    log(f'\n--- Outcome: {outcome_name} ---')
    
    y = df_est[y_col].astype(float)
    
    # Both early and late
    X = sm.add_constant(df_est[ctrl_cols + state_fe_cols + ['w_R1_early', 'w_R1_late']].astype(float))
    model = sm.OLS(y, X).fit(cov_type='HC1')
    
    log(f'  w_R1_early: coef = {model.params["w_R1_early"]:.4f} '
        f'(SE = {model.bse["w_R1_early"]:.4f}) '
        f'p = {model.pvalues["w_R1_early"]:.4f} {significance_stars(model.pvalues["w_R1_early"])}')
    log(f'  w_R1_late:  coef = {model.params["w_R1_late"]:.4f} '
        f'(SE = {model.bse["w_R1_late"]:.4f}) '
        f'p = {model.pvalues["w_R1_late"]:.4f} {significance_stars(model.pvalues["w_R1_late"])}')
    
    results.append({
        'analysis': 'All-source',
        'outcome': outcome_name,
        'variable': 'w_R1_early',
        'coef': model.params["w_R1_early"],
        'se': model.bse["w_R1_early"],
        'pval': model.pvalues["w_R1_early"],
        'stars': significance_stars(model.pvalues["w_R1_early"]),
        'N': len(df_est)
    })
    results.append({
        'analysis': 'All-source',
        'outcome': outcome_name,
        'variable': 'w_R1_late',
        'coef': model.params["w_R1_late"],
        'se': model.bse["w_R1_late"],
        'pval': model.pvalues["w_R1_late"],
        'stars': significance_stars(model.pvalues["w_R1_late"]),
        'N': len(df_est)
    })

# ==============================================================================
# PART 2: GEOGRAPHIC CLUSTER ANALYSIS
# ==============================================================================

log('\n' + '=' * 80)
log('PART 2: GEOGRAPHIC CLUSTER HETEROGENEITY')
log('=' * 80)

# Census division mapping
state_to_division = {
    'CT': 'New England', 'ME': 'New England', 'MA': 'New England', 
    'NH': 'New England', 'RI': 'New England', 'VT': 'New England',
    'NJ': 'Mid-Atlantic', 'NY': 'Mid-Atlantic', 'PA': 'Mid-Atlantic',
    'IL': 'East North Central', 'IN': 'East North Central', 
    'MI': 'East North Central', 'OH': 'East North Central', 'WI': 'East North Central',
    'IA': 'West North Central', 'KS': 'West North Central', 
    'MN': 'West North Central', 'MO': 'West North Central',
    'NE': 'West North Central', 'ND': 'West North Central', 'SD': 'West North Central',
    'DE': 'South Atlantic', 'FL': 'South Atlantic', 'GA': 'South Atlantic',
    'MD': 'South Atlantic', 'NC': 'South Atlantic', 'SC': 'South Atlantic',
    'VA': 'South Atlantic', 'WV': 'South Atlantic', 'DC': 'South Atlantic',
    'AL': 'East South Central', 'KY': 'East South Central',
    'MS': 'East South Central', 'TN': 'East South Central',
    'AR': 'West South Central', 'LA': 'West South Central',
    'OK': 'West South Central', 'TX': 'West South Central',
    'AZ': 'Mountain', 'CO': 'Mountain', 'ID': 'Mountain', 
    'MT': 'Mountain', 'NV': 'Mountain', 'NM': 'Mountain', 'UT': 'Mountain', 'WY': 'Mountain',
    'AK': 'Pacific', 'CA': 'Pacific', 'HI': 'Pacific', 'OR': 'Pacific', 'WA': 'Pacific'
}

df_est['division'] = df_est['state'].map(state_to_division)

log('\n--- R1 late-delivery effect by Census Division ---')
log('(Testing whether negative effect is geographically concentrated)\n')

# CSBP outcome for comparison (use IS_ADOPTER as proxy)
# First get R3 winner status
r3_winners = rebates[rebates['Funding Year'] == 2023][['nces_id']].drop_duplicates()
r3_winners['IS_R3_WINNER'] = 1
df_est = df_est.merge(r3_winners, on='nces_id', how='left')
df_est['IS_R3_WINNER'] = df_est['IS_R3_WINNER'].fillna(0).astype(int)

division_results = []

for div in df_est['division'].dropna().unique():
    div_data = df_est[df_est['division'] == div].copy()
    
    if len(div_data) < 100 or div_data['IS_R3_WINNER'].sum() < 5:
        continue
    
    # Simpler controls for subsamples
    simple_ctrl = ['log_median_income', 'log_enrollment', 'IS_APPLICANT']
    simple_ctrl = [c for c in simple_ctrl if c in div_data.columns]
    
    try:
        y = div_data['IS_R3_WINNER'].astype(float)
        X = sm.add_constant(div_data[simple_ctrl + ['w_R1_early', 'w_R1_late']].astype(float))
        model = sm.OLS(y, X).fit(cov_type='HC1')
        
        division_results.append({
            'division': div,
            'N': len(div_data),
            'n_r3': div_data['IS_R3_WINNER'].sum(),
            'n_late_neighbor': (div_data['w_R1_late'] > 0).sum(),
            'coef_early': model.params.get("w_R1_early", np.nan),
            'pval_early': model.pvalues.get("w_R1_early", np.nan),
            'coef_late': model.params.get("w_R1_late", np.nan),
            'pval_late': model.pvalues.get("w_R1_late", np.nan),
        })
    except:
        pass

div_df = pd.DataFrame(division_results).sort_values('coef_late')

log(f'{"Division":<22} {"N":>6} {"R3":>5} {"Late Nbr":>8} {"Early coef":>10} {"Late coef":>10} {"Late p":>8}')
log('-' * 80)

for _, row in div_df.iterrows():
    stars = significance_stars(row['pval_late'])
    log(f'{row["division"]:<22} {row["N"]:>6} {row["n_r3"]:>5} {row["n_late_neighbor"]:>8} '
        f'{row["coef_early"]:>10.4f} {row["coef_late"]:>10.4f}{stars:<3} {row["pval_late"]:>8.4f}')
    
    results.append({
        'analysis': 'Geographic',
        'outcome': 'R3_CSBP',
        'variable': f'{row["division"]}_late',
        'coef': row['coef_late'],
        'se': np.nan,
        'pval': row['pval_late'],
        'stars': significance_stars(row['pval_late']),
        'N': row['N']
    })

# ==============================================================================
# PART 3: STATE-LEVEL HETEROGENEITY (top 10 states by R1 winner count)
# ==============================================================================

log('\n--- R1 late-delivery effect by State (top states) ---')

# Count R1 winners by state
r1_by_state = df_est.groupby('state').agg({
    'w_R1_late': lambda x: (x > 0).sum(),
    'IS_R3_WINNER': 'sum'
}).reset_index()
r1_by_state.columns = ['state', 'n_late_neighbor', 'n_r3']
r1_by_state = r1_by_state.sort_values('n_late_neighbor', ascending=False)

top_states = r1_by_state[r1_by_state['n_late_neighbor'] >= 20]['state'].tolist()[:15]

log(f'\n{"State":<8} {"N":>6} {"R3":>5} {"Late Nbr":>8} {"Late coef":>10} {"Late p":>8}')
log('-' * 60)

state_results = []

for st in top_states:
    st_data = df_est[df_est['state'] == st].copy()
    
    if len(st_data) < 50 or st_data['IS_R3_WINNER'].sum() < 3:
        continue
    
    simple_ctrl = ['log_median_income', 'log_enrollment', 'IS_APPLICANT']
    simple_ctrl = [c for c in simple_ctrl if c in st_data.columns]
    
    try:
        y = st_data['IS_R3_WINNER'].astype(float)
        X = sm.add_constant(st_data[simple_ctrl + ['w_R1_late']].astype(float))
        model = sm.OLS(y, X).fit(cov_type='HC1')
        
        coef = model.params.get("w_R1_late", np.nan)
        pval = model.pvalues.get("w_R1_late", np.nan)
        stars = significance_stars(pval)
        
        n_late = (st_data['w_R1_late'] > 0).sum()
        n_r3 = st_data['IS_R3_WINNER'].sum()
        
        log(f'{st:<8} {len(st_data):>6} {n_r3:>5} {n_late:>8} {coef:>10.4f}{stars:<3} {pval:>8.4f}')
        
        state_results.append({
            'state': st,
            'N': len(st_data),
            'n_r3': n_r3,
            'coef_late': coef,
            'pval_late': pval
        })
    except:
        pass

# ==============================================================================
# SUMMARY
# ==============================================================================

log('\n' + '=' * 80)
log('SUMMARY')
log('=' * 80)

log('\n1. ALL-SOURCE ADOPTION:')
log('   Does the early/late pattern hold for non-CSBP adoption?')
all_source = [r for r in results if r['analysis'] == 'All-source']
for r in all_source:
    log(f'   {r["outcome"]} | {r["variable"]}: {r["coef"]:.4f} {r["stars"]} (p={r["pval"]:.4f})')

log('\n2. GEOGRAPHIC CONCENTRATION:')
log('   Is the negative late-delivery effect concentrated in specific regions?')
neg_divs = [r for r in div_df.to_dict('records') if r['coef_late'] < -0.05 and r['pval_late'] < 0.2]
if neg_divs:
    log(f'   Strongest negative effects in: {", ".join([d["division"] for d in neg_divs])}')
else:
    log('   No strong geographic concentration detected')

pos_divs = [r for r in div_df.to_dict('records') if r['coef_late'] > 0.05]
if pos_divs:
    log(f'   Positive effects in: {", ".join([d["division"] for d in pos_divs])}')

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(TABLES_DIR / 'delivery_timing_extended.csv', index=False)
log(f'\nSaved: {TABLES_DIR / "delivery_timing_extended.csv"}')

with open(LOGS_DIR / 'delivery_timing_extended.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f'Saved: {LOGS_DIR / "delivery_timing_extended.txt"}')

print('\n' + '=' * 80)
print('ANALYSIS COMPLETE')
print('=' * 80)
