"""
Propensity-Score Residualized Instrument
==========================================
Fixes the balance failure by residualizing the lottery instrument
on stratification variables before computing spatial lags.

The EPA CSB Program uses a stratified lottery:
  - Priority applicants (high-poverty, rural, tribal) drawn first
  - One-per-state guarantee in first selection step
  - 10% state funding cap

This creates differential win rates by (priority x state), making
IV_Z correlated with district characteristics. The spatial lag
w_geo_z inherits this confounding.

Fix: Abdulkadiroglu et al. (2011) / Angrist et al. (2013) approach:
  1. Compute Pr(winning | priority, state) within applicants
  2. Residualize: z_resid = IV_Z - Pr(winning | stratum)
  3. Recompute spatial lag on residualized instrument
  4. Additionally control for spatial lags of application and priority

Outputs:
  - Tables/resid_iv_balance_test.csv
  - Tables/resid_iv_estimation_results.csv
  - Tables/resid_iv_summary.txt
"""

import pandas as pd
import numpy as np
import warnings
import os
import sys
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import libpysal
from libpysal.weights import KNN, W

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from _estimation_utils import (
    load_analysis_data, load_shapefile_and_merge,
    build_geo_knn, create_spatial_lags,
    FULL_CONTROLS, significance_stars
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TABLES_DIR, RANDOM_SEED, ensure_dirs_exist

ensure_dirs_exist()

OUTPUT_BALANCE = TABLES_DIR / 'resid_iv_balance_test.csv'
OUTPUT_ESTIMATION = TABLES_DIR / 'resid_iv_estimation_results.csv'
OUTPUT_SUMMARY = TABLES_DIR / 'resid_iv_summary.txt'

K_VALUES = [6, 10]

print("=" * 80)
print("PROPENSITY-SCORE RESIDUALIZED INSTRUMENT")
print("=" * 80)

# =============================================================================
# 1. Load Data
# =============================================================================
df, locale_vars = load_analysis_data()
geo_df = load_shapefile_and_merge(df)

print(f"\nFull sample: {len(geo_df)} districts")
print(f"  Applicants:       {(geo_df['IS_APPLICANT'] == 1).sum()}")
print(f"  Lottery winners:  {(geo_df['IV_Z'] == 1).sum()}")
print(f"  Priority:         {(geo_df['is_priority'] == 1).sum()}")

# =============================================================================
# 2. Compute Stratum-Level Propensity Scores
# =============================================================================
print(f"\n{'='*80}")
print("STEP 1: PROPENSITY SCORE ESTIMATION")
print(f"{'='*80}")

# For applicants: compute Pr(IV_Z = 1 | is_priority, state) as cell means
applicants = geo_df[geo_df['IS_APPLICANT'] == 1].copy()
print(f"\nApplicants: {len(applicants)}")
print(f"  Winners:     {(applicants['IV_Z'] == 1).sum()}")
print(f"  Losers:      {(applicants['IV_Z'] == 0).sum()}")
print(f"  Priority:    {(applicants['is_priority'] == 1).sum()}")
print(f"  Non-priority: {(applicants['is_priority'] == 0).sum()}")

# Win rates by priority
for p_label, p_val in [('Priority', 1), ('Non-priority', 0)]:
    sub = applicants[applicants['is_priority'] == p_val]
    if len(sub) > 0:
        rate = sub['IV_Z'].mean()
        print(f"  Win rate ({p_label}): {rate:.4f} ({sub['IV_Z'].sum()}/{len(sub)})")

# Cell means: (is_priority x state)
# This is the most transparent propensity score estimator
cell_means = applicants.groupby(['is_priority', 'state'])['IV_Z'].agg(
    ['mean', 'count', 'sum']
).reset_index()
cell_means.columns = ['is_priority', 'state', 'propensity', 'n_applicants', 'n_winners']

print(f"\nPropensity score cells (is_priority x state):")
print(f"  Total cells: {len(cell_means)}")
print(f"  Min cell size: {cell_means['n_applicants'].min()}")
print(f"  Max cell size: {cell_means['n_applicants'].max()}")
print(f"  Median cell size: {cell_means['n_applicants'].median():.0f}")
print(f"  Propensity range: [{cell_means['propensity'].min():.4f}, {cell_means['propensity'].max():.4f}]")

# Flag small cells (n < 5) — these may need collapsing
small_cells = cell_means[cell_means['n_applicants'] < 5]
if len(small_cells) > 0:
    print(f"\n  WARNING: {len(small_cells)} cells with < 5 applicants")
    print(f"  These will use state-level propensity (collapsing across priority)")

# For small cells, use state-level propensity instead
state_means = applicants.groupby('state')['IV_Z'].mean().to_dict()

# Build propensity lookup
propensity_lookup = {}
for _, row in cell_means.iterrows():
    key = (int(row['is_priority']), row['state'])
    if row['n_applicants'] >= 5:
        propensity_lookup[key] = row['propensity']
    else:
        # Collapse to state level
        propensity_lookup[key] = state_means[row['state']]

# =============================================================================
# 3. Residualize IV_Z
# =============================================================================
print(f"\n{'='*80}")
print("STEP 2: RESIDUALIZE IV_Z")
print(f"{'='*80}")

# Assign propensity scores to all districts
propensity = np.zeros(len(geo_df))
for i, row in geo_df.iterrows():
    if row['IS_APPLICANT'] == 1:
        key = (int(row['is_priority']), row['state'])
        if key in propensity_lookup:
            propensity[i] = propensity_lookup[key]
        else:
            # Applicant in a state with no cell — use overall rate
            propensity[i] = applicants['IV_Z'].mean()
    else:
        # Non-applicant: zero probability of winning
        propensity[i] = 0.0

geo_df['propensity'] = propensity
geo_df['z_resid'] = geo_df['IV_Z'].values - propensity

print(f"\nResidual statistics:")
print(f"  Non-applicants (z_resid = 0):  {(geo_df['z_resid'] == 0).sum()}")
print(f"  Winners (z_resid > 0):         {(geo_df['z_resid'] > 0).sum()}")
print(f"  Losers (z_resid < 0):          {(geo_df['z_resid'] < 0).sum()}")
print(f"  Mean z_resid (all):            {geo_df['z_resid'].mean():.6f}")
print(f"  Mean z_resid (applicants):     {geo_df.loc[geo_df['IS_APPLICANT']==1, 'z_resid'].mean():.6f}")
print(f"  SD z_resid:                    {geo_df['z_resid'].std():.6f}")

# =============================================================================
# 4. Build Spatial Lags of Residualized Instrument + Controls
# =============================================================================
all_results = []
all_balance = []

for k in K_VALUES:
    print(f"\n{'='*80}")
    print(f"K = {k}: ESTIMATION AND BALANCE TEST")
    print(f"{'='*80}")

    w_geo = build_geo_knn(geo_df, k)

    # --- Original spatial lags ---
    geo_df['w_geo_adoption'] = libpysal.weights.lag_spatial(w_geo, geo_df['IS_ADOPTER'].values)
    geo_df['w_geo_z'] = libpysal.weights.lag_spatial(w_geo, geo_df['IV_Z'].values)

    # --- Residualized instrument spatial lag ---
    geo_df['w_geo_z_resid'] = libpysal.weights.lag_spatial(w_geo, geo_df['z_resid'].values)

    # --- Additional spatial controls ---
    geo_df['w_geo_applicant'] = libpysal.weights.lag_spatial(w_geo, geo_df['IS_APPLICANT'].values)
    geo_df['w_geo_priority'] = libpysal.weights.lag_spatial(w_geo, geo_df['is_priority'].values)

    print(f"\nSpatial lag statistics (K={k}):")
    for col in ['w_geo_z', 'w_geo_z_resid', 'w_geo_applicant', 'w_geo_priority']:
        print(f"  {col}: mean={geo_df[col].mean():.4f}, sd={geo_df[col].std():.4f}")

    # Correlation between original and residualized
    corr = np.corrcoef(geo_df['w_geo_z'].values, geo_df['w_geo_z_resid'].values)[0, 1]
    print(f"  Corr(w_geo_z, w_geo_z_resid): {corr:.4f}")

    # =========================================================================
    # 4a. Balance Test — ORIGINAL instrument
    # =========================================================================
    print(f"\n--- Balance Test: ORIGINAL w_geo_z (K={k}) ---")

    covariates_to_test = [
        'median_income', 'poverty_rate', 'enrollment', 'pm25',
        'pct_white', 'pct_black', 'pct_hispanic', 'is_priority'
    ]

    needed = covariates_to_test + ['w_geo_z', 'w_geo_z_resid',
             'w_geo_applicant', 'w_geo_priority', 'state']
    df_bal = geo_df.dropna(subset=list(set(needed))).copy()

    state_dum = pd.get_dummies(df_bal['state'], prefix='state', drop_first=True)
    n_tests = len(covariates_to_test)

    for cov in covariates_to_test:
        X = sm.add_constant(pd.concat([state_dum, df_bal[['w_geo_z']]], axis=1))
        idx_bal = X.columns.get_loc('w_geo_z')
        ols = sm.OLS(df_bal[cov].values, X.astype(float).values).fit(
            cov_type='cluster', cov_kwds={'groups': df_bal['state'].values}
        )
        pval = ols.pvalues[idx_bal]
        pval_bonf = min(pval * n_tests, 1.0)

        all_balance.append({
            'k': k, 'instrument': 'Original (w_geo_z)',
            'controls': 'State FE only',
            'covariate': cov,
            'coef': ols.params[idx_bal],
            'se': ols.bse[idx_bal],
            'pval': pval,
            'pval_bonferroni': pval_bonf,
            'significant': 'Yes' if pval_bonf < 0.05 else 'No',
            'n': len(df_bal)
        })

        star = significance_stars(pval_bonf)
        print(f"  {cov:<20s}: p_bonf={pval_bonf:.4f}{star}")

    # =========================================================================
    # 4b. Balance Test — RESIDUALIZED instrument
    # =========================================================================
    print(f"\n--- Balance Test: RESIDUALIZED w_geo_z_resid (K={k}) ---")

    for cov in covariates_to_test:
        X = sm.add_constant(pd.concat([state_dum, df_bal[['w_geo_z_resid']]], axis=1))
        idx_bal = X.columns.get_loc('w_geo_z_resid')
        ols = sm.OLS(df_bal[cov].values, X.astype(float).values).fit(
            cov_type='cluster', cov_kwds={'groups': df_bal['state'].values}
        )
        pval = ols.pvalues[idx_bal]
        pval_bonf = min(pval * n_tests, 1.0)

        all_balance.append({
            'k': k, 'instrument': 'Residualized (w_geo_z_resid)',
            'controls': 'State FE only',
            'covariate': cov,
            'coef': ols.params[idx_bal],
            'se': ols.bse[idx_bal],
            'pval': pval,
            'pval_bonferroni': pval_bonf,
            'significant': 'Yes' if pval_bonf < 0.05 else 'No',
            'n': len(df_bal)
        })

        star = significance_stars(pval_bonf)
        print(f"  {cov:<20s}: p_bonf={pval_bonf:.4f}{star}")

    # =========================================================================
    # 4c. Balance Test — RESIDUALIZED + spatial controls
    # =========================================================================
    print(f"\n--- Balance Test: RESIDUALIZED + Spatial Controls (K={k}) ---")

    for cov in covariates_to_test:
        X = sm.add_constant(pd.concat([
            state_dum,
            df_bal[['w_geo_z_resid', 'w_geo_applicant', 'w_geo_priority']]
        ], axis=1))
        idx_bal = X.columns.get_loc('w_geo_z_resid')
        ols = sm.OLS(df_bal[cov].values, X.astype(float).values).fit(
            cov_type='cluster', cov_kwds={'groups': df_bal['state'].values}
        )
        pval = ols.pvalues[idx_bal]
        pval_bonf = min(pval * n_tests, 1.0)

        all_balance.append({
            'k': k, 'instrument': 'Resid + Spatial Controls',
            'controls': 'State FE + w_geo_applicant + w_geo_priority',
            'covariate': cov,
            'coef': ols.params[idx_bal],
            'se': ols.bse[idx_bal],
            'pval': pval,
            'pval_bonferroni': pval_bonf,
            'significant': 'Yes' if pval_bonf < 0.05 else 'No',
            'n': len(df_bal)
        })

        star = significance_stars(pval_bonf)
        print(f"  {cov:<20s}: p_bonf={pval_bonf:.4f}{star}")

    # =========================================================================
    # 5. IV-2SLS Estimation: Compare Original vs Residualized
    # =========================================================================
    print(f"\n--- IV-2SLS Estimation (K={k}) ---")

    control_cols = FULL_CONTROLS + locale_vars
    needed_est = ['IS_ADOPTER', 'w_geo_adoption', 'w_geo_z', 'w_geo_z_resid',
                  'w_geo_applicant', 'w_geo_priority', 'IV_Z', 'state'] + control_cols
    df_reg = geo_df.dropna(subset=list(set(needed_est))).copy()

    state_dum_est = pd.get_dummies(df_reg['state'], prefix='state', drop_first=True)

    # --- Spec A: Original instrument (replicating Script 07) ---
    print(f"\n  Spec A: Original w_geo_z")
    exog_A = sm.add_constant(pd.concat([
        df_reg[control_cols + ['IV_Z']],
        state_dum_est
    ], axis=1))

    model_A = IV2SLS(
        dependent=df_reg['IS_ADOPTER'],
        exog=exog_A,
        endog=df_reg[['w_geo_adoption']],
        instruments=df_reg[['w_geo_z']]
    )
    res_A = model_A.fit(cov_type='clustered', clusters=df_reg['state'])

    coef_A = res_A.params['w_geo_adoption']
    se_A = res_A.std_errors['w_geo_adoption']
    pval_A = res_A.pvalues['w_geo_adoption']
    try:
        fstat_A = res_A.first_stage.diagnostics['f.stat']['w_geo_adoption']
    except Exception:
        fstat_A = np.nan

    print(f"    Peer effect: {coef_A:.4f} (SE={se_A:.4f}, p={pval_A:.4f})")
    print(f"    First-stage F: {fstat_A:.2f}")

    all_results.append({
        'k': k, 'specification': 'A: Original IV',
        'instrument': 'w_geo_z',
        'spatial_controls': 'None',
        'peer_effect': coef_A, 'se': se_A, 'pval': pval_A,
        'f_stat': fstat_A, 'n': len(df_reg)
    })

    # --- Spec B: Residualized instrument ---
    print(f"\n  Spec B: Residualized w_geo_z_resid")
    exog_B = sm.add_constant(pd.concat([
        df_reg[control_cols + ['IV_Z']],
        state_dum_est
    ], axis=1))

    model_B = IV2SLS(
        dependent=df_reg['IS_ADOPTER'],
        exog=exog_B,
        endog=df_reg[['w_geo_adoption']],
        instruments=df_reg[['w_geo_z_resid']]
    )
    res_B = model_B.fit(cov_type='clustered', clusters=df_reg['state'])

    coef_B = res_B.params['w_geo_adoption']
    se_B = res_B.std_errors['w_geo_adoption']
    pval_B = res_B.pvalues['w_geo_adoption']
    try:
        fstat_B = res_B.first_stage.diagnostics['f.stat']['w_geo_adoption']
    except Exception:
        fstat_B = np.nan

    print(f"    Peer effect: {coef_B:.4f} (SE={se_B:.4f}, p={pval_B:.4f})")
    print(f"    First-stage F: {fstat_B:.2f}")

    all_results.append({
        'k': k, 'specification': 'B: Residualized IV',
        'instrument': 'w_geo_z_resid',
        'spatial_controls': 'None',
        'peer_effect': coef_B, 'se': se_B, 'pval': pval_B,
        'f_stat': fstat_B, 'n': len(df_reg)
    })

    # --- Spec C: Residualized + spatial controls ---
    print(f"\n  Spec C: Residualized + spatial controls")
    exog_C = sm.add_constant(pd.concat([
        df_reg[control_cols + ['IV_Z', 'w_geo_applicant', 'w_geo_priority']],
        state_dum_est
    ], axis=1))

    model_C = IV2SLS(
        dependent=df_reg['IS_ADOPTER'],
        exog=exog_C,
        endog=df_reg[['w_geo_adoption']],
        instruments=df_reg[['w_geo_z_resid']]
    )
    res_C = model_C.fit(cov_type='clustered', clusters=df_reg['state'])

    coef_C = res_C.params['w_geo_adoption']
    se_C = res_C.std_errors['w_geo_adoption']
    pval_C = res_C.pvalues['w_geo_adoption']
    try:
        fstat_C = res_C.first_stage.diagnostics['f.stat']['w_geo_adoption']
    except Exception:
        fstat_C = np.nan

    print(f"    Peer effect: {coef_C:.4f} (SE={se_C:.4f}, p={pval_C:.4f})")
    print(f"    First-stage F: {fstat_C:.2f}")

    all_results.append({
        'k': k, 'specification': 'C: Resid IV + Spatial Controls',
        'instrument': 'w_geo_z_resid',
        'spatial_controls': 'w_geo_applicant + w_geo_priority',
        'peer_effect': coef_C, 'se': se_C, 'pval': pval_C,
        'f_stat': fstat_C, 'n': len(df_reg)
    })

    # --- Spec D: Original IV + spatial controls (to isolate the effect of residualization vs controls) ---
    print(f"\n  Spec D: Original IV + spatial controls (disentangle residualization from controls)")
    exog_D = sm.add_constant(pd.concat([
        df_reg[control_cols + ['IV_Z', 'w_geo_applicant', 'w_geo_priority']],
        state_dum_est
    ], axis=1))

    model_D = IV2SLS(
        dependent=df_reg['IS_ADOPTER'],
        exog=exog_D,
        endog=df_reg[['w_geo_adoption']],
        instruments=df_reg[['w_geo_z']]
    )
    res_D = model_D.fit(cov_type='clustered', clusters=df_reg['state'])

    coef_D = res_D.params['w_geo_adoption']
    se_D = res_D.std_errors['w_geo_adoption']
    pval_D = res_D.pvalues['w_geo_adoption']
    try:
        fstat_D = res_D.first_stage.diagnostics['f.stat']['w_geo_adoption']
    except Exception:
        fstat_D = np.nan

    print(f"    Peer effect: {coef_D:.4f} (SE={se_D:.4f}, p={pval_D:.4f})")
    print(f"    First-stage F: {fstat_D:.2f}")

    all_results.append({
        'k': k, 'specification': 'D: Original IV + Spatial Controls',
        'instrument': 'w_geo_z',
        'spatial_controls': 'w_geo_applicant + w_geo_priority',
        'peer_effect': coef_D, 'se': se_D, 'pval': pval_D,
        'f_stat': fstat_D, 'n': len(df_reg)
    })

# =============================================================================
# 6. Save Results
# =============================================================================
print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

balance_df = pd.DataFrame(all_balance)
balance_df.to_csv(str(OUTPUT_BALANCE), index=False)
print(f"Balance test: {OUTPUT_BALANCE}")

results_df = pd.DataFrame(all_results)
results_df.to_csv(str(OUTPUT_ESTIMATION), index=False)
print(f"Estimation:   {OUTPUT_ESTIMATION}")

# =============================================================================
# 7. Summary Report
# =============================================================================
with open(str(OUTPUT_SUMMARY), 'w') as f:
    f.write("PROPENSITY-SCORE RESIDUALIZED IV: SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("MOTIVATION\n")
    f.write("-" * 40 + "\n")
    f.write("The EPA CSB Program lottery is stratified by priority status\n")
    f.write("(high-poverty, rural, tribal) and state. Priority applicants\n")
    f.write("had ~12% win rate vs ~43% for non-priority. The original\n")
    f.write("spatial IV (w_geo_z) inherits this confounding, causing it to\n")
    f.write("predict district characteristics (5/8 balance test failures).\n\n")

    f.write("FIX: Residualize IV_Z on (is_priority x state) cell means,\n")
    f.write("then recompute spatial lag. Also add spatial controls for\n")
    f.write("neighbor application rates and priority status.\n\n")

    f.write("PROPENSITY SCORE DISTRIBUTION\n")
    f.write("-" * 40 + "\n")
    app = geo_df[geo_df['IS_APPLICANT'] == 1]
    f.write(f"Applicants: {len(app)}\n")
    f.write(f"Propensity range: [{app['propensity'].min():.4f}, {app['propensity'].max():.4f}]\n")
    f.write(f"Propensity mean:  {app['propensity'].mean():.4f}\n")
    f.write(f"z_resid mean (applicants): {app['z_resid'].mean():.6f}\n")
    f.write(f"z_resid SD:   {app['z_resid'].std():.4f}\n\n")

    # Balance test summary by instrument type
    f.write("BALANCE TEST RESULTS\n")
    f.write("-" * 40 + "\n")
    for k in K_VALUES:
        f.write(f"\nK = {k}:\n")
        for inst_type in balance_df['instrument'].unique():
            sub = balance_df[(balance_df['k'] == k) & (balance_df['instrument'] == inst_type)]
            n_sig = (sub['significant'] == 'Yes').sum()
            n_total = len(sub)
            f.write(f"  {inst_type}: {n_sig}/{n_total} significant (Bonferroni 5%)\n")

    # Estimation comparison
    f.write(f"\n\nIV-2SLS ESTIMATION RESULTS\n")
    f.write("-" * 40 + "\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")

    f.write("INTERPRETATION\n")
    f.write("-" * 40 + "\n")
    f.write("Compare balance failures across instrument types.\n")
    f.write("If residualized IV passes balance but original fails,\n")
    f.write("the stratified lottery was the source of confounding.\n")
    f.write("If the peer effect estimate changes substantially,\n")
    f.write("the original estimate was biased by confounding.\n")
    f.write("If F-stat drops below 10, residualization weakened\n")
    f.write("the instrument too much.\n")

print(f"Summary:      {OUTPUT_SUMMARY}")

print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print(f"{'='*80}")

for k in K_VALUES:
    print(f"\n--- K={k} ---")
    for inst_type in balance_df['instrument'].unique():
        sub = balance_df[(balance_df['k'] == k) & (balance_df['instrument'] == inst_type)]
        n_sig = (sub['significant'] == 'Yes').sum()
        n_total = len(sub)
        print(f"  Balance ({inst_type}): {n_sig}/{n_total} sig (Bonferroni)")

    print()
    sub_res = results_df[results_df['k'] == k]
    for _, row in sub_res.iterrows():
        star = significance_stars(row['pval'])
        print(f"  {row['specification']}: "
              f"coef={row['peer_effect']:.4f}, SE={row['se']:.4f}, "
              f"p={row['pval']:.4f}{star}, F={row['f_stat']:.1f}")

print("\nDONE.")
