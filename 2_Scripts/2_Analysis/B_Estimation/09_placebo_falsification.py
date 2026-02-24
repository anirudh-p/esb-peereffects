"""
Placebo and Falsification Tests
=================================
Three tests to validate the identification strategy:

Test 1: Non-Neighbor ("Distant Stranger") Instrument
  - Use lottery outcomes of non-neighbors as a placebo instrument
  - Should produce null effect if IV strategy is valid

Test 2: Permutation Inference
  - Randomly shuffle IV_Z 1000 times, build null distribution
  - Compute exact p-value for the true estimate

Test 3: Balance on Spatial IV
  - Test whether w_geo_z predicts pre-determined covariates
  - Should be insignificant (with Bonferroni correction)
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

OUTPUT_PLACEBO = TABLES_DIR / 'placebo_non_neighbor.csv'
OUTPUT_PERMUTATION = TABLES_DIR / 'placebo_permutation.csv'
OUTPUT_BALANCE = TABLES_DIR / 'placebo_balance_spatial_iv.csv'
OUTPUT_SUMMARY = TABLES_DIR / 'placebo_summary.txt'

N_PERM = 1000
K_VALUES = [6, 10]

print("=" * 80)
print("PLACEBO AND FALSIFICATION TESTS")
print("=" * 80)

# =============================================================================
# 1. Load Data
# =============================================================================
df, locale_vars = load_analysis_data()
geo_df = load_shapefile_and_merge(df)

print(f"\nSample: {len(geo_df)} districts")

# =============================================================================
# TEST 1: Non-Neighbor Placebo Instrument
# =============================================================================
print(f"\n{'='*80}")
print("TEST 1: NON-NEIGHBOR PLACEBO INSTRUMENT")
print(f"{'='*80}")

rng = np.random.default_rng(RANDOM_SEED)
placebo_results = []

for k in K_VALUES:
    print(f"\n--- K={k} ---")

    # Build true geographic KNN
    w_geo = build_geo_knn(geo_df, k)

    # Also build KNN-20 to define "non-neighbors" (anyone beyond k=20)
    w_geo_20 = build_geo_knn(geo_df, 20)

    # Create true spatial lags
    geo_lags = create_spatial_lags(geo_df, w_geo, w_clim=None)

    # --- Build placebo weight matrix ---
    # For each district: randomly select k districts from the same state
    # that are NOT among its KNN-20 neighbors
    print("  Building placebo (non-neighbor) weight matrix...")
    states = geo_lags['state'].values
    iv_z = geo_lags['IV_Z'].values
    n = len(geo_lags)

    neighbors_placebo = {}
    weights_placebo = {}

    state_indices = {}
    for i, s in enumerate(states):
        if s not in state_indices:
            state_indices[s] = []
        state_indices[s].append(i)

    for i in range(n):
        # Get KNN-20 neighbors to exclude
        knn20_neighbors = set(w_geo_20.neighbors[i])
        knn20_neighbors.add(i)  # Also exclude self

        # Same-state districts that are NOT KNN-20 neighbors
        same_state = state_indices[states[i]]
        candidates = [j for j in same_state if j not in knn20_neighbors]

        if len(candidates) >= k:
            selected = rng.choice(candidates, size=k, replace=False).tolist()
        elif len(candidates) > 0:
            selected = rng.choice(candidates, size=min(k, len(candidates)), replace=False).tolist()
        else:
            # Fallback: use random districts from other states
            all_others = [j for j in range(n) if j not in knn20_neighbors]
            selected = rng.choice(all_others, size=min(k, len(all_others)), replace=False).tolist()

        neighbors_placebo[i] = selected
        weights_placebo[i] = [1.0] * len(selected)

    w_placebo = W(neighbors_placebo, weights_placebo)
    w_placebo.transform = 'r'

    # Create placebo spatial lag of IV_Z
    w_distant_z = libpysal.weights.lag_spatial(w_placebo, iv_z)
    geo_lags_p = geo_lags.copy()
    geo_lags_p['w_distant_z'] = w_distant_z

    # --- Reduced form with placebo instrument ---
    print("  Running reduced form with placebo instrument...")
    control_cols = FULL_CONTROLS + locale_vars + ['IV_Z']
    needed = ['IS_ADOPTER', 'w_distant_z', 'state'] + control_cols
    df_reg = geo_lags_p.dropna(subset=list(set(needed))).copy()

    state_dum = pd.get_dummies(df_reg['state'], prefix='state', drop_first=True)
    X_placebo = sm.add_constant(pd.concat([
        df_reg[control_cols],
        state_dum,
        df_reg[['w_distant_z']]
    ], axis=1))

    ols_placebo = sm.OLS(df_reg['IS_ADOPTER'].values, X_placebo.astype(float).values).fit(
        cov_type='cluster', cov_kwds={'groups': df_reg['state'].values}
    )

    # Get coefficient by position
    idx_placebo = X_placebo.columns.get_loc('w_distant_z')
    coef_placebo = ols_placebo.params[idx_placebo]
    se_placebo = ols_placebo.bse[idx_placebo]
    pval_placebo = ols_placebo.pvalues[idx_placebo]

    # --- Reduced form with TRUE instrument (for comparison) ---
    X_true = sm.add_constant(pd.concat([
        df_reg[control_cols],
        state_dum,
        df_reg[['w_geo_z']]
    ], axis=1))

    ols_true = sm.OLS(df_reg['IS_ADOPTER'].values, X_true.astype(float).values).fit(
        cov_type='cluster', cov_kwds={'groups': df_reg['state'].values}
    )

    # Get coefficient by position
    idx_true = X_true.columns.get_loc('w_geo_z')
    coef_true = ols_true.params[idx_true]
    se_true = ols_true.bse[idx_true]
    pval_true = ols_true.pvalues[idx_true]

    print(f"  TRUE instrument:    coef={coef_true:.4f}, SE={se_true:.4f}, "
          f"p={pval_true:.4f}{significance_stars(pval_true)}")
    print(f"  PLACEBO instrument: coef={coef_placebo:.4f}, SE={se_placebo:.4f}, "
          f"p={pval_placebo:.4f}{significance_stars(pval_placebo)}")

    placebo_results.append({
        'k': k, 'type': 'True', 'coef': coef_true,
        'se': se_true, 'pval': pval_true, 'n': len(df_reg)
    })
    placebo_results.append({
        'k': k, 'type': 'Placebo (Non-Neighbor)', 'coef': coef_placebo,
        'se': se_placebo, 'pval': pval_placebo, 'n': len(df_reg)
    })

placebo_df = pd.DataFrame(placebo_results)
placebo_df.to_csv(str(OUTPUT_PLACEBO), index=False)
print(f"\nPlacebo results saved to {OUTPUT_PLACEBO}")


# =============================================================================
# TEST 2: Permutation Inference
# =============================================================================
print(f"\n{'='*80}")
print(f"TEST 2: PERMUTATION INFERENCE ({N_PERM} permutations)")
print(f"{'='*80}")

perm_results = []

for k in K_VALUES:
    print(f"\n--- K={k} ---")

    w_geo = build_geo_knn(geo_df, k)
    geo_lags = create_spatial_lags(geo_df, w_geo, w_clim=None)

    # Prepare regression data once
    control_cols = FULL_CONTROLS + locale_vars + ['IV_Z']
    needed = ['IS_ADOPTER', 'w_geo_z', 'state'] + control_cols
    df_reg = geo_lags.dropna(subset=list(set(needed))).copy()

    # Track which rows of geo_df survived into df_reg
    reg_indices = df_reg.index.values

    state_dum = pd.get_dummies(df_reg['state'], prefix='state', drop_first=True)
    X_base = sm.add_constant(pd.concat([df_reg[control_cols], state_dum], axis=1))

    # True estimate (reduced form)
    X_true = X_base.copy()
    X_true['w_geo_z'] = df_reg['w_geo_z'].values
    idx_wgz = X_true.columns.get_loc('w_geo_z')
    ols_true = sm.OLS(df_reg['IS_ADOPTER'].values, X_true.astype(float).values).fit()
    beta_true = ols_true.params[idx_wgz]
    print(f"  True reduced-form coefficient: {beta_true:.6f}")

    # Permutation loop
    # IMPORTANT: Permute IV_Z on FULL geo_df (matching w_geo dimensions),
    # then compute spatial lag on full sample, then subset to df_reg.
    print(f"  Running {N_PERM} permutations...")
    iv_z_full = geo_df['IV_Z'].values.copy()
    beta_perms = np.zeros(N_PERM)

    for i in range(N_PERM):
        if (i + 1) % 200 == 0:
            print(f"    Permutation {i+1}/{N_PERM}...")

        # Randomly shuffle IV_Z across FULL sample
        iv_z_perm = rng.permutation(iv_z_full)

        # Recompute spatial lag on full sample (matches w_geo dimensions)
        w_geo_z_perm_full = libpysal.weights.lag_spatial(w_geo, iv_z_perm)

        # Subset to regression sample
        w_geo_z_perm = w_geo_z_perm_full[reg_indices]

        # Reduced form OLS (fast)
        X_perm = X_base.copy()
        X_perm['w_geo_z'] = w_geo_z_perm
        ols_perm = sm.OLS(df_reg['IS_ADOPTER'].values, X_perm.astype(float).values).fit()
        beta_perms[i] = ols_perm.params[idx_wgz]

    # Compute exact p-value (two-sided)
    pval_exact = (np.sum(np.abs(beta_perms) >= np.abs(beta_true)) + 1) / (N_PERM + 1)

    # Null distribution statistics
    perm_mean = beta_perms.mean()
    perm_std = beta_perms.std()
    perm_2_5 = np.percentile(beta_perms, 2.5)
    perm_97_5 = np.percentile(beta_perms, 97.5)

    print(f"  Permutation results:")
    print(f"    True coefficient:  {beta_true:.6f}")
    print(f"    Null mean:         {perm_mean:.6f}")
    print(f"    Null SD:           {perm_std:.6f}")
    print(f"    Null 95% CI:       [{perm_2_5:.6f}, {perm_97_5:.6f}]")
    print(f"    Exact p-value:     {pval_exact:.4f}{significance_stars(pval_exact)}")

    perm_results.append({
        'k': k, 'beta_true': beta_true, 'pval_exact': pval_exact,
        'null_mean': perm_mean, 'null_sd': perm_std,
        'null_ci_low': perm_2_5, 'null_ci_high': perm_97_5,
        'n_perms': N_PERM, 'n': len(df_reg)
    })

perm_df = pd.DataFrame(perm_results)
perm_df.to_csv(str(OUTPUT_PERMUTATION), index=False)
print(f"\nPermutation results saved to {OUTPUT_PERMUTATION}")


# =============================================================================
# TEST 3: Balance on Spatial IV
# =============================================================================
print(f"\n{'='*80}")
print("TEST 3: BALANCE ON SPATIAL IV (w_geo_z)")
print(f"{'='*80}")

balance_results = []

# Use K=6 (primary specification)
k = 6
print(f"\nUsing K={k}")

w_geo = build_geo_knn(geo_df, k)
geo_lags = create_spatial_lags(geo_df, w_geo, w_clim=None)

# Pre-determined covariates to test
covariates_to_test = [
    'median_income', 'poverty_rate', 'enrollment', 'pm25',
    'pct_white', 'pct_black', 'pct_hispanic', 'is_priority'
]

needed = covariates_to_test + ['w_geo_z', 'state']
df_bal = geo_lags.dropna(subset=list(set(needed))).copy()

state_dum = pd.get_dummies(df_bal['state'], prefix='state', drop_first=True)

n_tests = len(covariates_to_test)
print(f"Testing {n_tests} covariates (Bonferroni threshold: {0.05/n_tests:.4f})")

for cov in covariates_to_test:
    X = sm.add_constant(pd.concat([
        state_dum,
        df_bal[['w_geo_z']]
    ], axis=1))

    idx_bal = X.columns.get_loc('w_geo_z')
    ols = sm.OLS(df_bal[cov].values, X.astype(float).values).fit(
        cov_type='cluster', cov_kwds={'groups': df_bal['state'].values}
    )

    coef = ols.params[idx_bal]
    se = ols.bse[idx_bal]
    pval = ols.pvalues[idx_bal]
    pval_bonf = min(pval * n_tests, 1.0)
    sig = pval_bonf < 0.05

    balance_results.append({
        'covariate': cov,
        'w_geo_z_coef': coef,
        'w_geo_z_se': se,
        'w_geo_z_pval': pval,
        'pval_bonferroni': pval_bonf,
        'significant_5pct': 'Yes' if sig else 'No',
        'n': len(df_bal)
    })

    star = significance_stars(pval)
    bonf_star = significance_stars(pval_bonf)
    print(f"  {cov:<20s}: coef={coef:>10.4f}, SE={se:>8.4f}, "
          f"p={pval:.4f}{star}, p_bonf={pval_bonf:.4f}{bonf_star}")

balance_df = pd.DataFrame(balance_results)
balance_df.to_csv(str(OUTPUT_BALANCE), index=False)

n_sig_raw = sum(1 for r in balance_results if r['w_geo_z_pval'] < 0.05)
n_sig_bonf = sum(1 for r in balance_results if r['pval_bonferroni'] < 0.05)

print(f"\n  Significant at 5% (raw):       {n_sig_raw}/{n_tests}")
print(f"  Significant at 5% (Bonferroni): {n_sig_bonf}/{n_tests}")

if n_sig_bonf == 0:
    print("  PASS: Spatial IV is balanced on pre-determined covariates")
else:
    print("  WARNING: Some covariates predict the spatial IV")


# =============================================================================
# Summary
# =============================================================================
with open(str(OUTPUT_SUMMARY), 'w') as f:
    f.write("PLACEBO AND FALSIFICATION TESTS SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    f.write("TEST 1: NON-NEIGHBOR PLACEBO INSTRUMENT\n")
    f.write("-" * 40 + "\n")
    f.write(placebo_df.to_string(index=False))
    f.write("\n\nExpected: Placebo coefficient insignificant, true coefficient significant.\n")
    f.write("If placebo is significant, spatial proximity may not be necessary\n")
    f.write("for the instrument channel.\n\n")

    f.write("TEST 2: PERMUTATION INFERENCE\n")
    f.write("-" * 40 + "\n")
    f.write(perm_df.to_string(index=False))
    f.write("\n\nExpected: exact p-value < 0.05 (true estimate in tails of null).\n")
    f.write("If exact p-value is large, the instrument may not be stronger\n")
    f.write("than random assignment would produce.\n\n")

    f.write("TEST 3: BALANCE ON SPATIAL IV (w_geo_z)\n")
    f.write("-" * 40 + "\n")
    f.write(balance_df.to_string(index=False))
    f.write(f"\n\nSignificant (raw 5%): {n_sig_raw}/{n_tests}\n")
    f.write(f"Significant (Bonferroni 5%): {n_sig_bonf}/{n_tests}\n")
    f.write("Expected: All insignificant after Bonferroni correction.\n")
    f.write("If significant, the spatial IV may predict own characteristics,\n")
    f.write("threatening the exclusion restriction.\n")

print(f"\nSummary saved to {OUTPUT_SUMMARY}")
print("\nALL FALSIFICATION TESTS COMPLETE.")
