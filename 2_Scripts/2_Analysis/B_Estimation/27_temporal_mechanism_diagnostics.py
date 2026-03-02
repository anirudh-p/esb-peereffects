"""
27_temporal_mechanism_diagnostics.py
=====================================
Diagnostic analysis to understand why cross-round temporal IV yields null 
results for all-source adoption but positive results for CSBP-only.

KEY HYPOTHESES:
  1. Deployment lag: R1 buses weren't visible by R3 decision time
  2. Information channel: The peer effect is program awareness, not bus observation
  3. Thin LATE: Non-CSBP adoption channel is inherently near-zero

DIAGNOSTICS:
  A. R1 Deployment Timing Analysis
     - When were R1 buses actually delivered vs awarded?
     - What % were operational by R3 application deadline (fall 2023)?
  
  B. Mechanism Tests
     - Does R1 neighbor win predict R3 APPLICATION (not just win)?
     - Does effect differ by R1 deployment speed (early vs late delivery)?
  
  C. Decomposition by Adoption Channel
     - CSBP-only vs VW/state programs vs other
     - Which channels show peer effects?

Outputs:
  - 3_Output/Tables/temporal_mechanism_diagnostics.csv
  - 3_Output/Logs/temporal_mechanism_diagnostics.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
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
        import re
        m = re.search(r'Q(\d)\s*(\d{4})', s)
        if m:
            q, y = int(m.group(1)), int(m.group(2))
            # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
            month = {1: 1, 2: 4, 3: 7, 4: 10}.get(q, 1)
            return pd.Timestamp(year=y, month=month, day=1)
        # Try just year
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
log('TEMPORAL MECHANISM DIAGNOSTICS')
log('Why is cross-round IV null for all-source but positive for CSBP?')
log('=' * 80)

# ==============================================================================
# PART A: R1 DEPLOYMENT TIMING ANALYSIS
# ==============================================================================

log('\n' + '=' * 80)
log('PART A: R1 DEPLOYMENT TIMING ANALYSIS')
log('=' * 80)

log('\n--- Loading WRI bus-level data ---')
df_bus = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='2. Bus-level data')
df_bus['nces_id'] = clean_lea_id(df_bus['1c. LEA ID'])

# Key date columns in WRI:
# 3p. Quarter awarded - when funding was awarded
# 3q. Quarter ordered - when bus was ordered
# 3r. Quarter delivered - when bus arrived
# 3s. Quarter first operating - when bus entered operation

date_cols = ['3p. Quarter awarded', '3q. Quarter ordered', 
             '3r. Quarter delivered', '3s. Quarter first operating']

for col in date_cols:
    df_bus[col + '_date'] = extract_quarter_date(df_bus[col])

log(f'\nBus-level data shape: {df_bus.shape}')
log(f'\nDate column coverage:')
for col in date_cols:
    pct = df_bus[col + '_date'].notna().mean() * 100
    log(f'  {col}: {pct:.1f}% non-missing')

# Focus on R1 buses (awarded 2022, CSBP funding)
# Filter to rebate program, 2022 funding
r1_buses = df_bus[
    (df_bus['3p. Quarter awarded_date'].dt.year == 2022) & 
    (df_bus['3z. Funding source 1'].str.contains('CLEAN SCHOOL BUS', na=False, case=False))
].copy()

log(f'\n--- R1 CSBP buses (awarded 2022): {len(r1_buses):,} ---')

# Timeline analysis
r1_buses['award_date'] = r1_buses['3p. Quarter awarded_date']
r1_buses['delivery_date'] = r1_buses['3r. Quarter delivered_date']
r1_buses['service_date'] = r1_buses['3s. Quarter first operating_date']

# R3 application deadline was approximately October 2023
r3_deadline = pd.Timestamp('2023-10-01')

log(f'\nR3 application deadline (approx): {r3_deadline.strftime("%Y-%m")}')

# What % of R1 buses were delivered/operational by R3 deadline?
r1_buses['delivered_by_r3'] = r1_buses['delivery_date'] < r3_deadline
r1_buses['operational_by_r3'] = r1_buses['service_date'] < r3_deadline

with_delivery = r1_buses['delivery_date'].notna()
with_service = r1_buses['service_date'].notna()

log(f'\nR1 buses with delivery date: {with_delivery.sum():,} ({with_delivery.mean()*100:.1f}%)')
log(f'R1 buses with service date: {with_service.sum():,} ({with_service.mean()*100:.1f}%)')

if with_delivery.sum() > 0:
    pct_delivered = r1_buses.loc[with_delivery, 'delivered_by_r3'].mean() * 100
    log(f'\nOf those with delivery date:')
    log(f'  Delivered before R3 deadline: {pct_delivered:.1f}%')

if with_service.sum() > 0:
    pct_operational = r1_buses.loc[with_service, 'operational_by_r3'].mean() * 100
    log(f'  Operational before R3 deadline: {pct_operational:.1f}%')

# Distribution of delivery dates
log(f'\nR1 bus delivery date distribution:')
if with_delivery.sum() > 0:
    delivery_dist = r1_buses.loc[with_delivery, 'delivery_date'].dt.to_period('Q').value_counts().sort_index()
    for period, count in delivery_dist.head(10).items():
        log(f'  {period}: {count:,} buses')

# District-level: when did R1 winner districts get their first bus operational?
r1_district_first = r1_buses.groupby('nces_id').agg({
    'delivery_date': 'min',
    'service_date': 'min'
}).reset_index()
r1_district_first.columns = ['nces_id', 'first_delivery', 'first_service']

log(f'\n--- R1 winner districts: {len(r1_district_first):,} ---')
r1_district_first['vis_by_r3'] = r1_district_first['first_delivery'] < r3_deadline
pct_vis = r1_district_first['vis_by_r3'].mean() * 100
log(f'Districts with first delivery before R3 deadline: {pct_vis:.1f}%')

# ==============================================================================
# PART B: MECHANISM TESTS - APPLICATION vs OBSERVATION
# ==============================================================================

log('\n' + '=' * 80)
log('PART B: MECHANISM TESTS')
log('=' * 80)

# Load applicant data to test information channel
applicant_file = RAW_WRI_DIR / "CSBP Applicants waitlisted and rejected_11.18.25.xlsx"
rebates_file = RAW_WRI_DIR / "CSB_Rebates.xlsx"

log('\n--- Loading applicant/rebate data ---')

rebates = pd.read_excel(rebates_file)
rebates['nces_id'] = clean_lea_id(rebates['NCES District ID'])
excluded = ['WITHDRAWN', 'CANCELLED', 'NOT SELECTED', 'INELIGIBLE', 'DENIED']
rebates = rebates[~rebates['Project Status'].str.upper().isin(excluded)]

# R1 and R3 winners
r1_winners = rebates[rebates['Funding Year'] == 2022][['nces_id']].drop_duplicates()
r1_winners['IV_Z_R1'] = 1
r3_winners = rebates[rebates['Funding Year'] == 2023][['nces_id']].drop_duplicates()
r3_winners['IS_R3_WINNER'] = 1

log(f'R1 winners: {len(r1_winners):,}')
log(f'R3 winners: {len(r3_winners):,}')

# Try to identify R3 applicants (losers)
# The applicant file has waitlisted/rejected applicants
try:
    applicants = pd.read_excel(applicant_file)
    applicants['nces_id'] = clean_lea_id(applicants['NCES District ID'])
    
    # R3 = Funding Year 2023 in rebates
    r3_applicants = applicants[applicants['Funding Year'] == 2023][['nces_id']].drop_duplicates()
    r3_applicants['IS_R3_APPLICANT'] = 1
    
    # Also R3 winners are applicants
    r3_all_applicants = pd.concat([
        r3_applicants[['nces_id']],
        r3_winners[['nces_id']]
    ]).drop_duplicates()
    r3_all_applicants['IS_R3_APPLICANT'] = 1
    
    log(f'R3 applicants (losers only): {len(r3_applicants):,}')
    log(f'R3 applicants (all): {len(r3_all_applicants):,}')
    has_r3_applicants = True
except Exception as e:
    log(f'Could not load applicant data: {e}')
    has_r3_applicants = False

# ==============================================================================
# PART C: IV ESTIMATION WITH DEPLOYMENT HETEROGENEITY
# ==============================================================================

log('\n' + '=' * 80)
log('PART C: IV ESTIMATION WITH MECHANISM TESTS')
log('=' * 80)

# Load main dataset
df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Merge R1/R3 status
df = df.merge(r1_winners, on='nces_id', how='left')
df = df.merge(r3_winners, on='nces_id', how='left')
df['IV_Z_R1'] = df['IV_Z_R1'].fillna(0).astype(int)
df['IS_R3_WINNER'] = df['IS_R3_WINNER'].fillna(0).astype(int)

if has_r3_applicants:
    df = df.merge(r3_all_applicants, on='nces_id', how='left')
    df['IS_R3_APPLICANT'] = df['IS_R3_APPLICANT'].fillna(0).astype(int)

# Merge R1 deployment timing
df = df.merge(r1_district_first, on='nces_id', how='left')
df['r1_early_delivery'] = (df['first_delivery'] < r3_deadline).fillna(False).astype(int)

log(f'\nTotal districts: {len(df):,}')
log(f'R1 winners: {df["IV_Z_R1"].sum():,}')
log(f'R3 winners: {df["IS_R3_WINNER"].sum():,}')
if has_r3_applicants:
    log(f'R3 applicants: {df["IS_R3_APPLICANT"].sum():,}')
log(f'R1 winners with early delivery: {df["r1_early_delivery"].sum():,}')

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

K = 6
w = KNN.from_dataframe(geo_pts, k=K)
w.transform = 'r'

# Spatial lags
geo_all['w_IV_Z_R1'] = lag_spatial(w, geo_all['IV_Z_R1'].values)
geo_all['w_R3_WINNER'] = lag_spatial(w, geo_all['IS_R3_WINNER'].values)
geo_all['w_r1_early'] = lag_spatial(w, geo_all['r1_early_delivery'].values)

if has_r3_applicants:
    geo_all['w_R3_APPLICANT'] = lag_spatial(w, geo_all['IS_R3_APPLICANT'].values)

log(f'Spatial sample: {len(geo_all):,}')

# Create estimation sample (non-R1 winners)
df_est = geo_all[geo_all['IV_Z_R1'] == 0].copy().reset_index(drop=True)

# Controls
locale_vars = []
if 'urbanicity' in df_est.columns:
    locale_dummies = pd.get_dummies(df_est['urbanicity'], prefix='locale', drop_first=True)
    df_est = pd.concat([df_est.reset_index(drop=True), locale_dummies.reset_index(drop=True)], axis=1)
    locale_vars = list(locale_dummies.columns)

base_controls = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pm25',
                 'pct_white', 'is_priority', 'IS_APPLICANT'] + locale_vars

# State FE
state_dummies = pd.get_dummies(df_est['state'], prefix='st', drop_first=True)
df_est = pd.concat([df_est.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
state_fe_cols = list(state_dummies.columns)

ctrl_cols = [c for c in base_controls if c in df_est.columns]
df_est = df_est.dropna(subset=ctrl_cols + ['state']).copy()

log(f'\nEstimation sample: {len(df_est):,}')

results = []

# --------------------------------------------------------------------------
# TEST 1: R1 neighbor wins -> R3 APPLICATION (information channel)
# --------------------------------------------------------------------------
if has_r3_applicants:
    log('\n--- TEST 1: R1 neighbor wins → R3 APPLICATION (information channel) ---')
    
    y = df_est['IS_R3_APPLICANT'].astype(float)
    X = sm.add_constant(df_est[ctrl_cols + state_fe_cols].astype(float))
    Z = df_est['w_IV_Z_R1'].astype(float)
    
    X['w_IV_Z_R1'] = Z
    model = sm.OLS(y, X).fit(cov_type='HC1')
    coef = model.params['w_IV_Z_R1']
    se = model.bse['w_IV_Z_R1']
    pval = model.pvalues['w_IV_Z_R1']
    
    log(f'  OLS: R1 neighbor wins → R3 application')
    log(f'  coef = {coef:.4f} (SE = {se:.4f}) p = {pval:.4f} {significance_stars(pval)}')
    log(f'  Interpretation: 10ppt increase in R1 neighbor share → {coef*0.1:.3f} change in R3 application prob')
    
    results.append({
        'test': 'R1_neighbor → R3_application',
        'method': 'OLS',
        'coef': coef, 'se': se, 'pval': pval,
        'stars': significance_stars(pval), 'N': len(df_est)
    })

# --------------------------------------------------------------------------
# TEST 2: Compare effect by R1 neighbor deployment timing
# --------------------------------------------------------------------------
log('\n--- TEST 2: Heterogeneity by R1 neighbor deployment timing ---')

# Split R1 neighbors into early vs late delivery
# w_r1_early = share of neighbors who are R1 winners AND had early delivery
# w_r1_late = share of R1 neighbors - early = late/missing

# We need to recalculate this correctly
# First, create indicator for R1-with-early-delivery in full sample
geo_all['R1_early'] = ((geo_all['IV_Z_R1'] == 1) & (geo_all['r1_early_delivery'] == 1)).astype(int)
geo_all['R1_late'] = ((geo_all['IV_Z_R1'] == 1) & (geo_all['r1_early_delivery'] == 0)).astype(int)

geo_all['w_R1_early'] = lag_spatial(w, geo_all['R1_early'].values)
geo_all['w_R1_late'] = lag_spatial(w, geo_all['R1_late'].values)

# Update estimation sample
df_est = geo_all[geo_all['IV_Z_R1'] == 0].copy().reset_index(drop=True)
if 'urbanicity' in df_est.columns:
    locale_dummies = pd.get_dummies(df_est['urbanicity'], prefix='locale', drop_first=True)
    df_est = pd.concat([df_est.reset_index(drop=True), locale_dummies.reset_index(drop=True)], axis=1)
state_dummies = pd.get_dummies(df_est['state'], prefix='st', drop_first=True)
df_est = pd.concat([df_est.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
state_fe_cols = list(state_dummies.columns)
df_est = df_est.dropna(subset=ctrl_cols + ['state']).copy()

# OLS with both early and late
y = df_est['IS_R3_WINNER'].astype(float)
X_vars = ctrl_cols + state_fe_cols + ['w_R1_early', 'w_R1_late']
X = sm.add_constant(df_est[[c for c in X_vars if c in df_est.columns]].astype(float))

try:
    model = sm.OLS(y, X).fit(cov_type='HC1')
    
    if 'w_R1_early' in model.params.index:
        log(f'\n  OLS: Separate effects by R1 neighbor deployment timing')
        log(f'  w_R1_early (visible by R3): coef = {model.params["w_R1_early"]:.4f} '
            f'(SE = {model.bse["w_R1_early"]:.4f}) p = {model.pvalues["w_R1_early"]:.4f}')
        log(f'  w_R1_late (not visible yet): coef = {model.params["w_R1_late"]:.4f} '
            f'(SE = {model.bse["w_R1_late"]:.4f}) p = {model.pvalues["w_R1_late"]:.4f}')
        
        results.append({
            'test': 'R1_early_neighbor → R3_win',
            'method': 'OLS',
            'coef': model.params["w_R1_early"], 
            'se': model.bse["w_R1_early"], 
            'pval': model.pvalues["w_R1_early"],
            'stars': significance_stars(model.pvalues["w_R1_early"]), 
            'N': len(df_est)
        })
        results.append({
            'test': 'R1_late_neighbor → R3_win',
            'method': 'OLS',
            'coef': model.params["w_R1_late"], 
            'se': model.bse["w_R1_late"], 
            'pval': model.pvalues["w_R1_late"],
            'stars': significance_stars(model.pvalues["w_R1_late"]), 
            'N': len(df_est)
        })
except Exception as e:
    log(f'  Could not estimate timing heterogeneity: {e}')

# --------------------------------------------------------------------------
# TEST 3: Decompose by funding source
# --------------------------------------------------------------------------
log('\n--- TEST 3: Decomposition by funding source ---')

# Load bus-level data to identify funding sources
df_bus = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='2. Bus-level data')
df_bus['nces_id'] = clean_lea_id(df_bus['1c. LEA ID'])
df_bus['award_year'] = extract_quarter_date(df_bus['3p. Quarter awarded']).dt.year

# R3 window = 2023-2024
r3_bus = df_bus[df_bus['award_year'].between(2023, 2024)].copy()

# Funding source categories
def categorize_funding(source):
    if pd.isna(source):
        return 'Unknown'
    s = str(source).upper()
    if 'EPA' in s or 'CLEAN SCHOOL BUS' in s:
        return 'CSBP'
    if 'VOLKSWAGEN' in s or 'VW' in s:
        return 'VW_Settlement'
    if 'STATE' in s or any(x in s for x in ['CALIFORNIA', 'HVIP', 'NYSERDA', 'TEXAS']):
        return 'State_Program'
    return 'Other'

r3_bus['funding_cat'] = r3_bus['3z. Funding source 1'].apply(categorize_funding)

log(f'\nR3-window bus count by funding source:')
funding_dist = r3_bus.groupby('funding_cat').size()
for cat, cnt in funding_dist.items():
    log(f'  {cat}: {cnt:,} buses')

# District-level: first adoption in R3 window by source
for cat in ['CSBP', 'VW_Settlement', 'State_Program', 'Other']:
    cat_bus = r3_bus[r3_bus['funding_cat'] == cat]
    districts = cat_bus['nces_id'].drop_duplicates()
    df_est[f'adopted_{cat}'] = df_est['nces_id'].isin(districts).astype(int)
    log(f'  Districts with {cat} adoption in R3 window: {df_est[f"adopted_{cat}"].sum():,}')

# Run reduced form for each funding channel
log('\n  Reduced form by funding channel:')
for cat in ['CSBP', 'VW_Settlement', 'State_Program', 'Other']:
    y_col = f'adopted_{cat}'
    if df_est[y_col].sum() < 10:
        log(f'  {cat}: too few observations')
        continue
        
    y = df_est[y_col].astype(float)
    X = sm.add_constant(df_est[ctrl_cols + state_fe_cols + ['w_IV_Z_R1']].astype(float))
    
    try:
        model = sm.OLS(y, X).fit(cov_type='HC1')
        coef = model.params['w_IV_Z_R1']
        se = model.bse['w_IV_Z_R1']
        pval = model.pvalues['w_IV_Z_R1']
        
        log(f'  {cat}: coef = {coef:.4f} (SE = {se:.4f}) p = {pval:.4f} {significance_stars(pval)}')
        
        results.append({
            'test': f'R1_neighbor → {cat}_adoption',
            'method': 'OLS_RF',
            'coef': coef, 'se': se, 'pval': pval,
            'stars': significance_stars(pval), 'N': len(df_est)
        })
    except Exception as e:
        log(f'  {cat}: error - {e}')

# ==============================================================================
# SUMMARY AND INTERPRETATION
# ==============================================================================

log('\n' + '=' * 80)
log('SUMMARY AND INTERPRETATION')
log('=' * 80)

log('\nKEY FINDINGS:')
log('\n1. R1 DEPLOYMENT TIMING:')
try:
    if with_delivery.sum() > 0:
        log(f'   - Only {pct_delivered:.0f}% of R1 buses were delivered before R3 deadline')
        log(f'   - This means most R1 buses were NOT visible when R3 decisions were made')
except:
    log('   - Could not determine deployment timing')

log('\n2. MECHANISM DIAGNOSIS:')
log('   - If R1 neighbor → R3 APPLICATION is significant:')
log('     → Information channel (heard about CSBP from neighbor)')
log('   - If R1 neighbor → R3 WIN is significant only for early-delivery:')
log('     → Bus observation channel (saw neighbor\'s buses)')

log('\n3. IMPLICATION FOR NULL RESULT:')
log('   - The temporal null for all-source adoption is likely because:')
log('     a) R1 buses weren\'t visible yet (deployment lag)')
log('     b) The peer effect is CSBP-specific (program awareness, not general ESB adoption)')
log('     c) Non-CSBP adoption channels operate independently')

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(TABLES_DIR / 'temporal_mechanism_diagnostics.csv', index=False)
log(f'\nSaved: {TABLES_DIR / "temporal_mechanism_diagnostics.csv"}')

with open(LOGS_DIR / 'temporal_mechanism_diagnostics.txt', 'w') as f:
    f.write('\n'.join(log_lines))
log(f'Saved: {LOGS_DIR / "temporal_mechanism_diagnostics.txt"}')

print('\n' + '=' * 80)
print('DIAGNOSTICS COMPLETE')
print('=' * 80)
