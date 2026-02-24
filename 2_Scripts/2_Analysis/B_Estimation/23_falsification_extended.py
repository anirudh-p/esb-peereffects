"""
23_falsification_extended.py
==============================
Extended falsification tests targeting the exclusion restriction directly.

Three new tests (complementing Script 09):

Test 1 — PLACEBO OUTCOMES:
  Regress pre-determined variables (pct_dem_2020, poverty_rate, pm25) on 
  w_IV_Z (instrumented via w_adoption). If the IV strategy is valid, 
  neighbor lottery wins should NOT predict own pre-determined characteristics.

Test 2 — DONUT NETWORK:
  Drop the 2 closest neighbors (where vendor/charging spillovers are 
  strongest). Use neighbors 3–8 only. If peer effects persist, the channel 
  is informational diffusion (travels farther), not local infrastructure.

Test 3 — SECOND-RING PLACEBO:
  Use neighbors 7–12 as a placebo exposure (far ring). Peer effects should 
  attenuate with distance. Strong attenuation supports geographic specificity 
  of the peer channel.

Inputs:
    - Cleaned/esb_full_analysis_dataset.csv
    - Raw/Spatial shapefile

Outputs:
    - Tables/falsification_extended.csv
    - Logs/falsification_extended.txt
"""

import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from _estimation_utils import (
    load_analysis_data, load_shapefile_and_merge,
    build_geo_knn, build_geo_donut,
    FULL_CONTROLS, significance_stars
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TABLES_DIR, LOGS_DIR, ensure_dirs_exist

import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from libpysal.weights import lag_spatial

ensure_dirs_exist()

OUTPUT_CSV = TABLES_DIR / 'falsification_extended.csv'
OUTPUT_LOG = LOGS_DIR / 'falsification_extended.txt'

K = 6

log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(msg)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

log("=" * 80)
log("EXTENDED FALSIFICATION TESTS")
log("=" * 80)

df, locale_vars = load_analysis_data()
geo_df = load_shapefile_and_merge(df)

log(f"\nFull spatial sample: {len(geo_df)}")

# Build primary weights
w_geo = build_geo_knn(geo_df, K)

geo_df['w_adoption'] = lag_spatial(w_geo, geo_df['IS_ADOPTER'].values)
geo_df['w_IV_Z'] = lag_spatial(w_geo, geo_df['IV_Z'].fillna(0).values)
geo_df['IS_LOSER'] = ((geo_df['IS_APPLICANT'] == 1) & (geo_df['IV_Z'] == 0)).astype(int)
geo_df['w_loser'] = lag_spatial(w_geo, geo_df['IS_LOSER'].values)

# State FE
state_dummies = pd.get_dummies(geo_df['state'], prefix='st', drop_first=True)
geo_df = pd.concat([geo_df.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
state_fe_cols = list(state_dummies.columns)

base_controls = FULL_CONTROLS + locale_vars

results = []

# ==============================================================================
# TEST 1: PLACEBO OUTCOMES
# ==============================================================================

log("\n" + "=" * 80)
log("TEST 1: PLACEBO OUTCOMES")
log("Regress pre-determined variables on instrumented neighbor adoption.")
log("If IV is valid, w_adoption (instrumented) should NOT predict these.")
log("Then re-estimate adding w_loser control (neighbor loser density).")
log("=" * 80)

# Placebo outcomes — all pre-determined (before the lottery)
placebo_outcomes = {
    'pct_dem_2020': 'County Dem vote share 2020',
    'poverty_rate': 'District poverty rate',
    'pm25': 'PM2.5 concentration',
    'pct_white': 'Pct white population',
    'median_income': 'Median household income',
    'enrollment': 'Student enrollment',
}

for outcome_var, description in placebo_outcomes.items():
    # Remove the placebo outcome from controls if it's there
    controls_this = [c for c in base_controls if c != outcome_var 
                     and c != f'log_{outcome_var}']
    
    needed = [outcome_var, 'w_adoption', 'w_IV_Z', 'state'] + controls_this + state_fe_cols
    df_r = geo_df.dropna(subset=list(set(needed))).copy()
    
    y_plac = df_r[outcome_var].astype(float)
    X_endog = df_r[['w_adoption']].astype(float)
    X_exog_raw = sm.add_constant(df_r[controls_this + state_fe_cols].astype(float))
    X_exog = X_exog_raw.loc[:, X_exog_raw.std() > 0]
    Z = df_r[['w_IV_Z']].astype(float)
    
    try:
        iv_plac = IV2SLS(dependent=y_plac, exog=X_exog, endog=X_endog, instruments=Z
                         ).fit(cov_type='clustered', clusters=df_r['state'])
        
        coef = iv_plac.params['w_adoption']
        se = iv_plac.std_errors['w_adoption']
        pval = iv_plac.pvalues['w_adoption']
        sig = significance_stars(pval)
        
        results.append({
            'test': 'Placebo outcome',
            'spec': f'{outcome_var}',
            'description': description,
            'peer_effect': coef, 'se': se, 'pval': pval, 'stars': sig,
            'N': len(df_r), 'expected': 'Null',
        })
        
        verdict = "PASS ✓" if pval > 0.10 else "CONCERN ✗"
        log(f"  {outcome_var:20s} [base]: β={coef:>10.4f} (SE={se:.4f}) p={pval:.4f}{sig}  {verdict}")

        # Placebo with neighbor loser-density control (w_loser)
        needed_l = [outcome_var, 'w_adoption', 'w_IV_Z', 'w_loser', 'state'] + controls_this + state_fe_cols
        df_l = geo_df.dropna(subset=list(set(needed_l))).copy()

        y_l = df_l[outcome_var].astype(float)
        X_endog_l = df_l[['w_adoption']].astype(float)
        X_exog_l_raw = sm.add_constant(df_l[controls_this + ['w_loser'] + state_fe_cols].astype(float))
        X_exog_l = X_exog_l_raw.loc[:, X_exog_l_raw.std() > 0]
        Z_l = df_l[['w_IV_Z']].astype(float)

        iv_plac_loser = IV2SLS(
            dependent=y_l,
            exog=X_exog_l,
            endog=X_endog_l,
            instruments=Z_l,
        ).fit(cov_type='clustered', clusters=df_l['state'])

        coef_l = iv_plac_loser.params['w_adoption']
        se_l = iv_plac_loser.std_errors['w_adoption']
        pval_l = iv_plac_loser.pvalues['w_adoption']
        sig_l = significance_stars(pval_l)

        results.append({
            'test': 'Placebo outcome (+w_loser)',
            'spec': f'{outcome_var}',
            'description': f'{description} (controls include w_loser)',
            'peer_effect': coef_l, 'se': se_l, 'pval': pval_l, 'stars': sig_l,
            'N': len(df_l), 'expected': 'Null',
        })

        verdict_l = "PASS ✓" if pval_l > 0.10 else "CONCERN ✗"
        log(f"  {outcome_var:20s} [+w_loser]: β={coef_l:>10.4f} (SE={se_l:.4f}) p={pval_l:.4f}{sig_l}  {verdict_l}")
    
    except Exception as e:
        log(f"  {outcome_var:20s}: FAILED — {str(e)[:60]}")

n_placebo_pass = sum(1 for r in results if r['test'] == 'Placebo outcome' and r['pval'] > 0.10)
n_placebo_total = sum(1 for r in results if r['test'] == 'Placebo outcome')
log(f"\n  Placebo outcomes passing (p > 0.10): {n_placebo_pass}/{n_placebo_total}")

n_placebo_l_pass = sum(1 for r in results if r['test'] == 'Placebo outcome (+w_loser)' and r['pval'] > 0.10)
n_placebo_l_total = sum(1 for r in results if r['test'] == 'Placebo outcome (+w_loser)')
log(f"  Placebo outcomes passing +w_loser (p > 0.10): {n_placebo_l_pass}/{n_placebo_l_total}")

# ==============================================================================
# TEST 2: DONUT NETWORK
# ==============================================================================

log("\n" + "=" * 80)
log("TEST 2: DONUT NETWORK")
log("Drop 2 closest neighbors, use neighbors 3–8 only.")
log("If effects persist → informational diffusion, not local infrastructure.")
log("=" * 80)

# Build donut weights: neighbors 3 through 8 (skip 1-2)
log("  Building donut weights (k_inner=2, k_outer=8)...")
w_donut = build_geo_donut(geo_df, k_inner=2, k_outer=8)

geo_df['w_donut_adoption'] = lag_spatial(w_donut, geo_df['IS_ADOPTER'].values)
geo_df['w_donut_IV_Z'] = lag_spatial(w_donut, geo_df['IV_Z'].fillna(0).values)

log(f"  w_donut_adoption range: [{geo_df['w_donut_adoption'].min():.3f}, {geo_df['w_donut_adoption'].max():.3f}]")
log(f"  w_donut_IV_Z range:     [{geo_df['w_donut_IV_Z'].min():.3f}, {geo_df['w_donut_IV_Z'].max():.3f}]")

# Run IV with donut weights
needed = ['IS_ADOPTER', 'w_donut_adoption', 'w_donut_IV_Z', 'state'] + base_controls + state_fe_cols
df_donut = geo_df.dropna(subset=list(set(needed))).copy()

y_d = df_donut['IS_ADOPTER'].astype(float)
X_endog_d = df_donut[['w_donut_adoption']].astype(float)
X_exog_d_raw = sm.add_constant(df_donut[base_controls + state_fe_cols].astype(float))
X_exog_d = X_exog_d_raw.loc[:, X_exog_d_raw.std() > 0]
Z_d = df_donut[['w_donut_IV_Z']].astype(float)

# First stage
fs_X_d = pd.concat([X_exog_d, Z_d], axis=1).astype(float)
fs_d = sm.OLS(X_endog_d.values.ravel(), fs_X_d.values).fit()
idx_zd = fs_X_d.columns.get_loc('w_donut_IV_Z')
f_donut = fs_d.tvalues[idx_zd] ** 2

iv_donut = IV2SLS(dependent=y_d, exog=X_exog_d, endog=X_endog_d, instruments=Z_d
                  ).fit(cov_type='clustered', clusters=df_donut['state'])

peer_d = iv_donut.params['w_donut_adoption']
se_d = iv_donut.std_errors['w_donut_adoption']
pval_d = iv_donut.pvalues['w_donut_adoption']
sig_d = significance_stars(pval_d)

results.append({
    'test': 'Donut network (3-8)',
    'spec': 'IV-2SLS, neighbors 3-8',
    'description': 'Excludes 2 closest neighbors',
    'peer_effect': peer_d, 'se': se_d, 'pval': pval_d, 'stars': sig_d,
    'N': len(df_donut), 'expected': 'Positive (if informational)',
    'F_first': f_donut,
})
log(f"\n  Donut (3-8): β={peer_d:.4f} (SE={se_d:.4f}) p={pval_d:.4f}{sig_d} | F={f_donut:.1f}")

# Also run baseline for comparison
needed_b = ['IS_ADOPTER', 'w_adoption', 'w_IV_Z', 'state'] + base_controls + state_fe_cols
df_base = geo_df.dropna(subset=list(set(needed_b))).copy()

_base_exog_raw = sm.add_constant(df_base[base_controls + state_fe_cols].astype(float))
_base_exog = _base_exog_raw.loc[:, _base_exog_raw.std() > 0]
iv_base = IV2SLS(
    dependent=df_base['IS_ADOPTER'].astype(float),
    exog=_base_exog,
    endog=df_base[['w_adoption']].astype(float),
    instruments=df_base[['w_IV_Z']].astype(float)
).fit(cov_type='clustered', clusters=df_base['state'])

peer_b = iv_base.params['w_adoption']
se_b = iv_base.std_errors['w_adoption']
pval_b = iv_base.pvalues['w_adoption']

results.append({
    'test': 'Baseline (1-6)',
    'spec': 'IV-2SLS, KNN-6 (baseline)',
    'description': 'Standard KNN-6 for comparison',
    'peer_effect': peer_b, 'se': se_b, 'pval': pval_b, 'stars': significance_stars(pval_b),
    'N': len(df_base), 'expected': 'Positive',
})
log(f"  Baseline (1-6): β={peer_b:.4f} (SE={se_b:.4f}) p={pval_b:.4f}{significance_stars(pval_b)}")

# ==============================================================================
# TEST 3: SECOND-RING PLACEBO (far neighbors)
# ==============================================================================

log("\n" + "=" * 80)
log("TEST 3: SECOND-RING PLACEBO (NEIGHBORS 7-12)")
log("Peer effects should attenuate with distance.")
log("=" * 80)

log("  Building far-ring weights (k_inner=6, k_outer=12)...")
w_far = build_geo_donut(geo_df, k_inner=6, k_outer=12)

geo_df['w_far_adoption'] = lag_spatial(w_far, geo_df['IS_ADOPTER'].values)
geo_df['w_far_IV_Z'] = lag_spatial(w_far, geo_df['IV_Z'].fillna(0).values)

log(f"  w_far_adoption range: [{geo_df['w_far_adoption'].min():.3f}, {geo_df['w_far_adoption'].max():.3f}]")
log(f"  w_far_IV_Z range:     [{geo_df['w_far_IV_Z'].min():.3f}, {geo_df['w_far_IV_Z'].max():.3f}]")

# IV with far-ring weights
needed_f = ['IS_ADOPTER', 'w_far_adoption', 'w_far_IV_Z', 'state'] + base_controls + state_fe_cols
df_far = geo_df.dropna(subset=list(set(needed_f))).copy()

y_f = df_far['IS_ADOPTER'].astype(float)
X_endog_f = df_far[['w_far_adoption']].astype(float)
X_exog_f_raw = sm.add_constant(df_far[base_controls + state_fe_cols].astype(float))
X_exog_f = X_exog_f_raw.loc[:, X_exog_f_raw.std() > 0]
Z_f = df_far[['w_far_IV_Z']].astype(float)

# First stage
fs_X_f = pd.concat([X_exog_f, Z_f], axis=1).astype(float)
fs_f = sm.OLS(X_endog_f.values.ravel(), fs_X_f.values).fit()
idx_zf = fs_X_f.columns.get_loc('w_far_IV_Z')
f_far = fs_f.tvalues[idx_zf] ** 2

iv_far = IV2SLS(dependent=y_f, exog=X_exog_f, endog=X_endog_f, instruments=Z_f
                ).fit(cov_type='clustered', clusters=df_far['state'])

peer_f = iv_far.params['w_far_adoption']
se_f = iv_far.std_errors['w_far_adoption']
pval_f = iv_far.pvalues['w_far_adoption']
sig_f = significance_stars(pval_f)

results.append({
    'test': 'Far ring (7-12)',
    'spec': 'IV-2SLS, neighbors 7-12',
    'description': 'Second-ring neighbors (placebo)',
    'peer_effect': peer_f, 'se': se_f, 'pval': pval_f, 'stars': sig_f,
    'N': len(df_far), 'expected': 'Attenuated / null',
    'F_first': f_far,
})
log(f"\n  Far ring (7-12): β={peer_f:.4f} (SE={se_f:.4f}) p={pval_f:.4f}{sig_f} | F={f_far:.1f}")

# Also do a "medium ring" for gradient
log("\n  Building medium-ring weights (k_inner=3, k_outer=9)...")
w_med = build_geo_donut(geo_df, k_inner=3, k_outer=9)

geo_df['w_med_adoption'] = lag_spatial(w_med, geo_df['IS_ADOPTER'].values)
geo_df['w_med_IV_Z'] = lag_spatial(w_med, geo_df['IV_Z'].fillna(0).values)

needed_m = ['IS_ADOPTER', 'w_med_adoption', 'w_med_IV_Z', 'state'] + base_controls + state_fe_cols
df_med = geo_df.dropna(subset=list(set(needed_m))).copy()

y_m = df_med['IS_ADOPTER'].astype(float)
X_endog_m = df_med[['w_med_adoption']].astype(float)
X_exog_m_raw = sm.add_constant(df_med[base_controls + state_fe_cols].astype(float))
X_exog_m = X_exog_m_raw.loc[:, X_exog_m_raw.std() > 0]
Z_m = df_med[['w_med_IV_Z']].astype(float)

fs_X_m = pd.concat([X_exog_m, Z_m], axis=1).astype(float)
fs_m = sm.OLS(X_endog_m.values.ravel(), fs_X_m.values).fit()
idx_zm = fs_X_m.columns.get_loc('w_med_IV_Z')
f_med = fs_m.tvalues[idx_zm] ** 2

iv_med = IV2SLS(dependent=y_m, exog=X_exog_m, endog=X_endog_m, instruments=Z_m
                ).fit(cov_type='clustered', clusters=df_med['state'])

peer_m = iv_med.params['w_med_adoption']
se_m = iv_med.std_errors['w_med_adoption']
pval_m = iv_med.pvalues['w_med_adoption']
sig_m = significance_stars(pval_m)

results.append({
    'test': 'Medium ring (4-9)',
    'spec': 'IV-2SLS, neighbors 4-9',
    'description': 'Medium-ring neighbors',
    'peer_effect': peer_m, 'se': se_m, 'pval': pval_m, 'stars': sig_m,
    'N': len(df_med), 'expected': 'Intermediate',
    'F_first': f_med,
})
log(f"  Medium ring (4-9): β={peer_m:.4f} (SE={se_m:.4f}) p={pval_m:.4f}{sig_m} | F={f_med:.1f}")

# ==============================================================================
# DISTANCE GRADIENT SUMMARY
# ==============================================================================

log("\n" + "=" * 80)
log("DISTANCE GRADIENT SUMMARY")
log("=" * 80)
log(f"\n  {'Ring':15s} | {'Peer Effect':>12s} | {'SE':>8s} | {'p':>8s} | {'F_1st':>8s}")
log("  " + "-" * 65)
log(f"  {'Close (1-6)':15s} | {peer_b:>12.4f} | {se_b:>8.4f} | {pval_b:>8.4f} | {'~54k':>8s}")
log(f"  {'Donut (3-8)':15s} | {peer_d:>12.4f} | {se_d:>8.4f} | {pval_d:>8.4f} | {f_donut:>8.1f}")
log(f"  {'Medium (4-9)':15s} | {peer_m:>12.4f} | {se_m:>8.4f} | {pval_m:>8.4f} | {f_med:>8.1f}")
log(f"  {'Far (7-12)':15s} | {peer_f:>12.4f} | {se_f:>8.4f} | {pval_f:>8.4f} | {f_far:>8.1f}")

# ==============================================================================
# SAVE
# ==============================================================================

results_df = pd.DataFrame(results)
results_df.to_csv(str(OUTPUT_CSV), index=False)
log(f"\nResults saved to {OUTPUT_CSV}")

with open(str(OUTPUT_LOG), 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"Log saved to {OUTPUT_LOG}")

log("""
TAKEAWAY:
- Placebo outcomes: If w_IV_Z (instrumented through neighbor adoption) does
  NOT predict pre-determined covariates, the exclusion restriction holds for
  characteristics we can observe.
  
- Donut network: If peer effects persist when dropping the 2 closest neighbors,
  the channel operates beyond immediate adjacency (informational > infrastructure).
  
- Distance gradient: Effects should monotonically decline from close → medium → 
  far ring. A flat or increasing gradient would suggest spurious spatial 
  correlation rather than true peer effects.
""")
