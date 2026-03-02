"""
30_california_deep_dive.py
==========================
Deep dive into California's strong negative late-delivery peer effect.

California context:
- Early ESB adopter (HVIP program since 2015)
- Dense school district network (935 districts)
- Strong negative late-delivery effect (-0.40, p<0.0001)

Key questions:
1. Which R1 districts had late deliveries? Any publicized problems?
2. Is the effect driven by specific counties/regions within CA?
3. Does CA's existing HVIP infrastructure moderate the effect?
4. Is there substitution to state programs when CSBP signals are bad?
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
log('CALIFORNIA DEEP DIVE: LATE-DELIVERY PEER EFFECTS')
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

# Load LEA info for names
df_lea = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='1. District-level data')
df_lea['nces_id'] = clean_lea_id(df_lea['1c. LEA ID'])
lea_names = df_lea[['nces_id', '1b. Local Education Agency (LEA) or entity name', '1g. State']].drop_duplicates()
lea_names.columns = ['nces_id', 'lea_name', 'state']

# Focus on California
ca_lea = lea_names[lea_names['state'] == 'CA'].copy()
log(f'California LEAs in WRI: {len(ca_lea):,}')

# ==============================================================================
# 2. R1 CSBP BUSES IN CALIFORNIA
# ==============================================================================

log('\n--- R1 CSBP buses in California ---')

r1_buses = df_bus[
    (df_bus['award_year'] == 2022) & 
    (df_bus['3z. Funding source 1'].str.contains('CLEAN SCHOOL BUS', na=False, case=False))
].copy()

r1_ca = r1_buses[r1_buses['nces_id'].isin(ca_lea['nces_id'])].copy()
log(f'R1 CSBP buses in CA: {len(r1_ca):,}')

r3_deadline = pd.Timestamp('2023-10-01')
r1_ca['early_delivery'] = r1_ca['delivery_date'] < r3_deadline
r1_ca['late_delivery'] = ~(r1_ca['delivery_date'] < r3_deadline)

log(f'  Early delivery (before Oct 2023): {r1_ca["early_delivery"].sum():,}')
log(f'  Late/missing delivery: {r1_ca["late_delivery"].sum():,}')

# District-level summary
r1_ca_dist = r1_ca.groupby('nces_id').agg({
    'delivery_date': 'min',
    'early_delivery': 'any',
    'late_delivery': 'all'  # All buses late = district had problems
}).reset_index()
r1_ca_dist.columns = ['nces_id', 'first_delivery', 'has_early', 'all_late']
r1_ca_dist = r1_ca_dist.merge(lea_names, on='nces_id', how='left')

log(f'\nR1 CA districts: {len(r1_ca_dist):,}')
log(f'  With at least one early delivery: {r1_ca_dist["has_early"].sum()}')
log(f'  All buses late/missing: {r1_ca_dist["all_late"].sum()}')

# ==============================================================================
# 3. WHICH CA DISTRICTS HAD LATE DELIVERIES?
# ==============================================================================

log('\n--- CA R1 districts with late/missing deliveries ---')

late_districts = r1_ca_dist[r1_ca_dist['all_late']].copy()
late_districts = late_districts.sort_values('first_delivery', ascending=False)

log(f'\nDistricts with ALL buses late/missing (potential bad signals):')
log(f'{"LEA Name":<40} {"First Delivery":<15} {"NCES ID"}')
log('-' * 70)

for _, row in late_districts.head(20).iterrows():
    fd = row['first_delivery'].strftime('%Y-%m') if pd.notna(row['first_delivery']) else 'Missing'
    log(f'{row["lea_name"][:40]:<40} {fd:<15} {row["nces_id"]}')

# ==============================================================================
# 4. CA ESB ADOPTION BY FUNDING SOURCE
# ==============================================================================

log('\n--- CA ESB adoption by funding source (2023-2024) ---')

ca_buses_all = df_bus[df_bus['nces_id'].isin(ca_lea['nces_id'])].copy()
ca_r3_window = ca_buses_all[ca_buses_all['award_year'].between(2023, 2024)].copy()

def categorize_funding(source):
    if pd.isna(source):
        return 'Unknown'
    s = str(source).upper()
    if 'CLEAN SCHOOL BUS' in s:
        return 'EPA_CSBP'
    if 'HVIP' in s or 'HYBRID AND ZERO' in s:
        return 'CA_HVIP'
    if 'MOYER' in s:
        return 'Carl_Moyer'
    if 'COMMUNITY AIR' in s:
        return 'CAPP'
    if 'VOLKSWAGEN' in s or 'VW' in s:
        return 'VW_Settlement'
    return 'Other_State'

ca_r3_window['funding_cat'] = ca_r3_window['3z. Funding source 1'].apply(categorize_funding)

log('\nCA buses in 2023-2024 by funding source:')
funding_dist = ca_r3_window.groupby('funding_cat').size().sort_values(ascending=False)
for cat, cnt in funding_dist.items():
    log(f'  {cat}: {cnt:,} buses')

# District-level adoption by source
ca_adoption = {}
for cat in funding_dist.index:
    districts = ca_r3_window[ca_r3_window['funding_cat'] == cat]['nces_id'].unique()
    ca_adoption[cat] = set(districts)
    log(f'  {cat} district count: {len(districts)}')

# ==============================================================================
# 5. LOAD MAIN DATASET AND TEST SUBSTITUTION
# ==============================================================================

log('\n--- Testing substitution hypothesis ---')
log('Do late-delivery R1 neighbors push CA districts toward state programs?')

df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Filter to CA
df_ca = df[df['state'] == 'CA'].copy()
log(f'\nCA districts in analysis dataset: {len(df_ca):,}')

# R1 winner status with timing
rebates = pd.read_excel(RAW_WRI_DIR / "CSB_Rebates.xlsx")
rebates['nces_id'] = clean_lea_id(rebates['NCES District ID'])
excluded = ['WITHDRAWN', 'CANCELLED', 'NOT SELECTED', 'INELIGIBLE', 'DENIED']
rebates = rebates[~rebates['Project Status'].str.upper().isin(excluded)]

r1_winners = rebates[rebates['Funding Year'] == 2022][['nces_id']].drop_duplicates()
r1_winners['IV_Z_R1'] = 1

# Merge delivery timing
r1_district = r1_ca_dist[['nces_id', 'has_early', 'all_late']].copy()
r1_district['R1_early'] = r1_district['has_early'].astype(int)
r1_district['R1_late'] = r1_district['all_late'].astype(int)
r1_winners = r1_winners.merge(r1_district[['nces_id', 'R1_early', 'R1_late']], on='nces_id', how='left')
r1_winners['R1_early'] = r1_winners['R1_early'].fillna(0).astype(int)
r1_winners['R1_late'] = r1_winners['R1_late'].fillna(0).astype(int)

df_ca = df_ca.merge(r1_winners[['nces_id', 'IV_Z_R1', 'R1_early', 'R1_late']], on='nces_id', how='left')
for c in ['IV_Z_R1', 'R1_early', 'R1_late']:
    df_ca[c] = df_ca[c].fillna(0).astype(int)

# Create outcome variables
r3_winners = rebates[rebates['Funding Year'] == 2023][['nces_id']].drop_duplicates()
r3_winners['IS_R3_CSBP'] = 1
df_ca = df_ca.merge(r3_winners, on='nces_id', how='left')
df_ca['IS_R3_CSBP'] = df_ca['IS_R3_CSBP'].fillna(0).astype(int)

# State program adoption
df_ca['adopted_HVIP'] = df_ca['nces_id'].isin(ca_adoption.get('CA_HVIP', set())).astype(int)
df_ca['adopted_state'] = df_ca['nces_id'].isin(
    ca_adoption.get('CA_HVIP', set()) | 
    ca_adoption.get('Carl_Moyer', set()) | 
    ca_adoption.get('CAPP', set())
).astype(int)

log(f'CA R3 CSBP adopters: {df_ca["IS_R3_CSBP"].sum()}')
log(f'CA HVIP adopters (2023-24): {df_ca["adopted_HVIP"].sum()}')
log(f'CA state program adopters: {df_ca["adopted_state"].sum()}')

# ==============================================================================
# 6. BUILD SPATIAL WEIGHTS FOR CA ONLY
# ==============================================================================

log('\n--- Building CA-only spatial weights ---')

gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
shp_id = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]
gdf[shp_id] = gdf[shp_id].astype(str).str.zfill(7)
gdf = gdf[[shp_id, 'geometry']].copy()

# Filter to CA
gdf_ca = gdf[gdf[shp_id].isin(df_ca['nces_id'])].copy()
gdf_ca['geometry'] = gdf_ca.geometry.simplify(tolerance=100)
gdf_ca = gdf_ca.to_crs(epsg=5070)

geo_ca = gdf_ca.merge(df_ca, left_on=shp_id, right_on='nces_id', how='inner').reset_index(drop=True)
geo_ca['centroid'] = geo_ca.geometry.centroid
geo_pts = geo_ca.set_geometry('centroid')

log(f'CA spatial sample: {len(geo_ca):,}')

K = 6
w = KNN.from_dataframe(geo_pts, k=K)
w.transform = 'r'

geo_ca['w_R1_early'] = lag_spatial(w, geo_ca['R1_early'].values)
geo_ca['w_R1_late'] = lag_spatial(w, geo_ca['R1_late'].values)

# Estimation sample (non-R1 winners)
df_est = geo_ca[geo_ca['IV_Z_R1'] == 0].copy().reset_index(drop=True)

base_controls = ['log_median_income', 'log_enrollment', 'IS_APPLICANT']
ctrl_cols = [c for c in base_controls if c in df_est.columns]
df_est = df_est.dropna(subset=ctrl_cols).copy()

log(f'CA estimation sample: {len(df_est):,}')
log(f'Mean(w_R1_early): {df_est["w_R1_early"].mean():.4f}')
log(f'Mean(w_R1_late): {df_est["w_R1_late"].mean():.4f}')

# ==============================================================================
# 7. REGRESSION ANALYSIS
# ==============================================================================

log('\n--- Reduced form regressions (CA only) ---')

results = []

for outcome_name, y_col in [
    ('CSBP R3', 'IS_R3_CSBP'),
    ('HVIP', 'adopted_HVIP'),
    ('Any State Program', 'adopted_state')
]:
    if df_est[y_col].sum() < 5:
        log(f'  {outcome_name}: too few observations')
        continue
    
    y = df_est[y_col].astype(float)
    X = sm.add_constant(df_est[ctrl_cols + ['w_R1_early', 'w_R1_late']].astype(float))
    
    try:
        model = sm.OLS(y, X).fit(cov_type='HC1')
        
        log(f'\n  Outcome: {outcome_name} (N adopters = {df_est[y_col].sum()})')
        log(f'    w_R1_early: coef = {model.params["w_R1_early"]:.4f} '
            f'(SE = {model.bse["w_R1_early"]:.4f}) '
            f'p = {model.pvalues["w_R1_early"]:.4f} {significance_stars(model.pvalues["w_R1_early"])}')
        log(f'    w_R1_late:  coef = {model.params["w_R1_late"]:.4f} '
            f'(SE = {model.bse["w_R1_late"]:.4f}) '
            f'p = {model.pvalues["w_R1_late"]:.4f} {significance_stars(model.pvalues["w_R1_late"])}')
        
        results.append({
            'outcome': outcome_name,
            'variable': 'w_R1_early',
            'coef': model.params["w_R1_early"],
            'se': model.bse["w_R1_early"],
            'pval': model.pvalues["w_R1_early"],
            'stars': significance_stars(model.pvalues["w_R1_early"]),
            'N': len(df_est),
            'n_adopt': df_est[y_col].sum()
        })
        results.append({
            'outcome': outcome_name,
            'variable': 'w_R1_late',
            'coef': model.params["w_R1_late"],
            'se': model.bse["w_R1_late"],
            'pval': model.pvalues["w_R1_late"],
            'stars': significance_stars(model.pvalues["w_R1_late"]),
            'N': len(df_est),
            'n_adopt': df_est[y_col].sum()
        })
    except Exception as e:
        log(f'  {outcome_name}: error - {e}')

# ==============================================================================
# 8. COUNTY-LEVEL ANALYSIS WITHIN CA
# ==============================================================================

log('\n--- County-level heterogeneity within CA ---')

# Load county info
try:
    lea_county = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='5. Counties')
    lea_county = lea_county[['1c. LEA ID', '10a. County Name']].copy()
    lea_county.columns = ['nces_id', 'county']
    lea_county['nces_id'] = clean_lea_id(lea_county['nces_id'])
    
    df_est = df_est.merge(lea_county[['nces_id', 'county']].drop_duplicates(), on='nces_id', how='left')
    
    # Counties with most R1 late-delivery exposure
    county_exposure = df_est.groupby('county').agg({
        'w_R1_late': 'mean',
        'IS_R3_CSBP': 'sum',
        'nces_id': 'count'
    }).reset_index()
    county_exposure.columns = ['county', 'mean_late_exposure', 'n_r3', 'n_districts']
    county_exposure = county_exposure[county_exposure['n_districts'] >= 10].sort_values('mean_late_exposure', ascending=False)
    
    log(f'\nCounties with highest late-delivery exposure:')
    log(f'{"County":<25} {"N Dist":>8} {"Late Exp":>10} {"R3 CSBP":>8}')
    log('-' * 55)
    for _, row in county_exposure.head(15).iterrows():
        log(f'{str(row["county"])[:25]:<25} {row["n_districts"]:>8} {row["mean_late_exposure"]:>10.4f} {row["n_r3"]:>8}')
    
except Exception as e:
    log(f'County analysis failed: {e}')

# ==============================================================================
# 9. SUMMARY
# ==============================================================================

log('\n' + '=' * 80)
log('CALIFORNIA DEEP DIVE SUMMARY')
log('=' * 80)

log('\n1. R1 DELIVERY TIMING IN CA:')
log(f'   - R1 CSBP buses: {len(r1_ca):,}')
log(f'   - Districts with late/missing delivery: {r1_ca_dist["all_late"].sum()}')

log('\n2. SUBSTITUTION HYPOTHESIS:')
log('   If late CSBP delivery signals "avoid federal program", do districts')
log('   substitute to state programs (HVIP, Carl Moyer, CAPP)?')

csbp_results = [r for r in results if r['outcome'] == 'CSBP R3' and r['variable'] == 'w_R1_late']
state_results = [r for r in results if r['outcome'] == 'Any State Program' and r['variable'] == 'w_R1_late']

if csbp_results:
    r = csbp_results[0]
    log(f'\n   CSBP R3 adoption: late effect = {r["coef"]:.4f} {r["stars"]} (p={r["pval"]:.4f})')
if state_results:
    r = state_results[0]
    log(f'   State program adoption: late effect = {r["coef"]:.4f} {r["stars"]} (p={r["pval"]:.4f})')

log('\n3. INTERPRETATION:')
log('   - If CSBP effect is negative and state effect is positive: SUBSTITUTION')
log('   - If both negative: GENERAL DETERRENCE')
log('   - If CSBP negative, state null: PROGRAM-SPECIFIC BAD SIGNAL')

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(TABLES_DIR / 'california_deep_dive.csv', index=False)
log(f'\nSaved: {TABLES_DIR / "california_deep_dive.csv"}')

with open(LOGS_DIR / 'california_deep_dive.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f'Saved: {LOGS_DIR / "california_deep_dive.txt"}')

print('\n' + '=' * 80)
print('ANALYSIS COMPLETE')
print('=' * 80)
