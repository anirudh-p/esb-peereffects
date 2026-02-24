"""
Adoption Intensity Models
==========================
Tests whether peer effects influence the INTENSITY of ESB adoption
(number of buses), not just the binary adoption decision.

Models:
  1. Linear 2SLS on log(NUM_ADOPTED + 1)
  2. Linear 2SLS on NUM_ADOPTED (raw count)
  3. Poisson IV via Control Function (2-stage residual inclusion)
  4. Negative Binomial IV via Control Function

All use the canonical specification (full controls + state FE + state clustering).
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

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from _estimation_utils import (
    load_analysis_data, load_shapefile_and_merge,
    build_geo_knn, load_and_align_climate_weights,
    create_spatial_lags, FULL_CONTROLS, significance_stars
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TABLES_DIR, ensure_dirs_exist

ensure_dirs_exist()

OUTPUT_FILE = TABLES_DIR / 'intensity_results.csv'
OUTPUT_COMPARISON = TABLES_DIR / 'intensity_comparison.csv'

K_VALUES = [6, 10]

print("=" * 80)
print("ADOPTION INTENSITY MODELS")
print("=" * 80)

# =============================================================================
# 1. Load Data
# =============================================================================
df, locale_vars = load_analysis_data()
geo_df = load_shapefile_and_merge(df)

# Check NUM_ADOPTED availability
if 'NUM_ADOPTED' not in geo_df.columns:
    # Try total_esbs_committed
    if 'total_esbs_committed' in geo_df.columns:
        geo_df['NUM_ADOPTED'] = geo_df['total_esbs_committed'].fillna(0)
        print("  Using total_esbs_committed as NUM_ADOPTED")
    else:
        print("  ERROR: Neither NUM_ADOPTED nor total_esbs_committed found")
        sys.exit(1)

geo_df['NUM_ADOPTED'] = geo_df['NUM_ADOPTED'].fillna(0).astype(float)
geo_df['log_num_adopted'] = np.log(geo_df['NUM_ADOPTED'] + 1)

print(f"\nSample: {len(geo_df)} districts")
print(f"  NUM_ADOPTED > 0: {(geo_df['NUM_ADOPTED'] > 0).sum()} "
      f"({100*(geo_df['NUM_ADOPTED'] > 0).mean():.1f}%)")
print(f"  NUM_ADOPTED mean: {geo_df['NUM_ADOPTED'].mean():.3f}")
print(f"  NUM_ADOPTED max: {geo_df['NUM_ADOPTED'].max():.0f}")
print(f"  NUM_ADOPTED distribution: "
      f"0={int((geo_df['NUM_ADOPTED']==0).sum())}, "
      f"1-5={int(((geo_df['NUM_ADOPTED']>=1) & (geo_df['NUM_ADOPTED']<=5)).sum())}, "
      f"6+={int((geo_df['NUM_ADOPTED']>5).sum())}")


# =============================================================================
# Helper: Poisson Control Function IV
# =============================================================================

def poisson_control_function(df_reg, endog_name, instrument_name, control_names,
                             locale_vars, state_fe=True, n_bootstrap=500,
                             seed=42):
    """
    Poisson IV via Control Function (2-Stage Residual Inclusion).

    Stage 1: OLS of endog on instrument + controls -> residuals
    Stage 2: Poisson of Y on endog + controls + stage1_residuals
    Bootstrap SEs (cluster by state).

    Returns dict with coefficient, bootstrap SE, bootstrap p-value.
    """
    rng = np.random.default_rng(seed)

    # Prepare variables
    all_control_cols = list(control_names) + list(locale_vars)
    if 'IV_Z' in df_reg.columns and 'IV_Z' not in all_control_cols:
        all_control_cols.append('IV_Z')

    needed = ['NUM_ADOPTED', endog_name, instrument_name, 'state'] + all_control_cols
    df_clean = df_reg.dropna(subset=list(set(needed))).copy()

    # Build control matrix
    controls_df = df_clean[all_control_cols]
    if state_fe:
        state_dummies = pd.get_dummies(df_clean['state'], prefix='state', drop_first=True)
        controls_df = pd.concat([controls_df, state_dummies], axis=1)

    X_controls = sm.add_constant(controls_df)

    #Stage 1: OLS for residuals
    X_stage1 = X_controls.copy()
    X_stage1[instrument_name] = df_clean[instrument_name].values
    stage1 = sm.OLS(df_clean[endog_name].values, X_stage1.astype(float).values).fit()
    residuals = stage1.resid

    # Stage 2: Poisson with control function
    X_stage2 = X_controls.copy()
    X_stage2[endog_name] = df_clean[endog_name].values
    X_stage2['cf_resid'] = residuals

    try:
        poisson_model = sm.GLM(
            df_clean['NUM_ADOPTED'].values,
            X_stage2.astype(float).values,
            family=sm.families.Poisson()
        ).fit(maxiter=100)
        coef_point = poisson_model.params[X_stage2.columns.get_loc(endog_name)]
    except Exception as e:
        print(f"    Poisson estimation failed: {e}")
        return {'coef': np.nan, 'se': np.nan, 'pval': np.nan, 'n': len(df_clean)}

    # Bootstrap SEs (cluster by state)
    states = df_clean['state'].unique()
    boot_coefs = []

    for b in range(n_bootstrap):
        # Resample states with replacement
        boot_states = rng.choice(states, size=len(states), replace=True)

        # Build bootstrap sample
        boot_dfs = []
        for s in boot_states:
            boot_dfs.append(df_clean[df_clean['state'] == s])
        boot_df = pd.concat(boot_dfs, ignore_index=True)

        # Rebuild controls for bootstrap sample
        boot_controls = boot_df[all_control_cols]
        if state_fe:
            boot_state_dum = pd.get_dummies(boot_df['state'], prefix='state', drop_first=True)
            # Align columns with original
            for col in state_dummies.columns:
                if col not in boot_state_dum.columns:
                    boot_state_dum[col] = 0
            boot_state_dum = boot_state_dum[state_dummies.columns]
            boot_controls = pd.concat([boot_controls, boot_state_dum], axis=1)

        boot_X_controls = sm.add_constant(boot_controls)

        try:
            # Stage 1 on bootstrap sample
            boot_X1 = boot_X_controls.copy()
            boot_X1[instrument_name] = boot_df[instrument_name].values
            boot_s1 = sm.OLS(boot_df[endog_name].values, boot_X1.astype(float).values).fit()

            # Stage 2 on bootstrap sample
            boot_X2 = boot_X_controls.copy()
            boot_X2[endog_name] = boot_df[endog_name].values
            boot_X2['cf_resid'] = boot_s1.resid

            boot_poisson = sm.GLM(
                boot_df['NUM_ADOPTED'].values,
                boot_X2.astype(float).values,
                family=sm.families.Poisson()
            ).fit(maxiter=100)
            boot_coefs.append(boot_poisson.params[boot_X2.columns.get_loc(endog_name)])
        except Exception:
            continue

    boot_coefs = np.array(boot_coefs)
    if len(boot_coefs) > 10:
        boot_se = boot_coefs.std()
        boot_pval = 2 * min(
            (boot_coefs >= 0).mean(),
            (boot_coefs <= 0).mean()
        )
    else:
        boot_se = np.nan
        boot_pval = np.nan

    return {
        'coef': coef_point,
        'se': boot_se,
        'pval': boot_pval,
        'n': len(df_clean),
        'n_bootstrap_success': len(boot_coefs)
    }


# =============================================================================
# 2. Run Models
# =============================================================================

all_results = []

for k in K_VALUES:
    print(f"\n{'='*80}")
    print(f"K = {k}")
    print(f"{'='*80}")

    # Build weights
    w_geo = build_geo_knn(geo_df, k)
    geo_lags = create_spatial_lags(geo_df, w_geo, w_clim=None)

    # Also create spatial lags of NUM_ADOPTED for the count models
    geo_lags['w_geo_num_adopted'] = libpysal.weights.lag_spatial(
        w_geo, geo_lags['NUM_ADOPTED'].values
    )

    # -----------------------------------------------------------------
    # Model 1: Linear 2SLS on log(NUM_ADOPTED + 1)
    # -----------------------------------------------------------------
    print(f"\n--- Model 1: Linear 2SLS on log(NUM_ADOPTED + 1) ---")

    control_cols = FULL_CONTROLS + locale_vars + ['IV_Z']
    needed = ['log_num_adopted', 'w_geo_adoption', 'w_geo_z', 'state'] + control_cols
    df_reg = geo_lags.dropna(subset=list(set(needed))).copy()

    exog_parts = [df_reg[control_cols]]
    state_dum = pd.get_dummies(df_reg['state'], prefix='state', drop_first=True)
    exog_parts.append(state_dum)
    exog = sm.add_constant(pd.concat(exog_parts, axis=1))

    model1 = IV2SLS(
        df_reg['log_num_adopted'],
        exog,
        df_reg[['w_geo_adoption']],
        df_reg[['w_geo_z']]
    ).fit(cov_type='clustered', clusters=df_reg['state'])

    coef1 = model1.params['w_geo_adoption']
    se1 = model1.std_errors['w_geo_adoption']
    pval1 = model1.pvalues['w_geo_adoption']
    print(f"  N={len(df_reg)}: coef={coef1:.4f}, SE={se1:.4f}, p={pval1:.4f}{significance_stars(pval1)}")

    all_results.append({
        'model': 'Linear 2SLS on log(count+1)', 'k': k, 'n': len(df_reg),
        'coef': coef1, 'se': se1, 'pval': pval1, 'dv': 'log(NUM_ADOPTED+1)'
    })

    # -----------------------------------------------------------------
    # Model 2: Linear 2SLS on NUM_ADOPTED (raw count)
    # -----------------------------------------------------------------
    print(f"\n--- Model 2: Linear 2SLS on NUM_ADOPTED (raw count) ---")

    model2 = IV2SLS(
        df_reg['NUM_ADOPTED'],
        exog,
        df_reg[['w_geo_adoption']],
        df_reg[['w_geo_z']]
    ).fit(cov_type='clustered', clusters=df_reg['state'])

    coef2 = model2.params['w_geo_adoption']
    se2 = model2.std_errors['w_geo_adoption']
    pval2 = model2.pvalues['w_geo_adoption']
    print(f"  N={len(df_reg)}: coef={coef2:.4f}, SE={se2:.4f}, p={pval2:.4f}{significance_stars(pval2)}")

    all_results.append({
        'model': 'Linear 2SLS on raw count', 'k': k, 'n': len(df_reg),
        'coef': coef2, 'se': se2, 'pval': pval2, 'dv': 'NUM_ADOPTED'
    })

    # -----------------------------------------------------------------
    # Model 3: Poisson IV via Control Function
    # -----------------------------------------------------------------
    print(f"\n--- Model 3: Poisson IV (Control Function, {500} bootstrap reps) ---")

    res_poisson = poisson_control_function(
        geo_lags, endog_name='w_geo_adoption', instrument_name='w_geo_z',
        control_names=FULL_CONTROLS, locale_vars=locale_vars,
        state_fe=True, n_bootstrap=50, seed=42
    )
    print(f"  N={res_poisson['n']}: coef={res_poisson['coef']:.4f}, "
          f"boot_SE={res_poisson['se']:.4f}, boot_p={res_poisson['pval']:.4f}"
          f"{significance_stars(res_poisson['pval']) if not np.isnan(res_poisson['pval']) else ''}")
    print(f"  Bootstrap successes: {res_poisson['n_bootstrap_success']}/500")

    all_results.append({
        'model': 'Poisson IV (Control Function)', 'k': k, 'n': res_poisson['n'],
        'coef': res_poisson['coef'], 'se': res_poisson['se'],
        'pval': res_poisson['pval'], 'dv': 'NUM_ADOPTED (count)'
    })

    # -----------------------------------------------------------------
    # Model 4: Negative Binomial IV via Control Function
    # -----------------------------------------------------------------
    print(f"\n--- Model 4: Negative Binomial IV (Control Function) ---")

    # Only point estimate (bootstrap would be very slow for NB)
    all_control_cols = FULL_CONTROLS + locale_vars + ['IV_Z']
    needed_nb = ['NUM_ADOPTED', 'w_geo_adoption', 'w_geo_z', 'state'] + all_control_cols
    df_nb = geo_lags.dropna(subset=list(set(needed_nb))).copy()

    controls_nb = df_nb[all_control_cols]
    state_dum_nb = pd.get_dummies(df_nb['state'], prefix='state', drop_first=True)
    controls_nb = pd.concat([controls_nb, state_dum_nb], axis=1)
    X_nb = sm.add_constant(controls_nb)

    # Stage 1
    X_s1 = X_nb.copy()
    X_s1['w_geo_z'] = df_nb['w_geo_z'].values
    stage1_nb = sm.OLS(df_nb['w_geo_adoption'].values, X_s1.astype(float).values).fit()

    # Stage 2: NegBin
    X_s2 = X_nb.copy()
    X_s2['w_geo_adoption'] = df_nb['w_geo_adoption'].values
    X_s2['cf_resid'] = stage1_nb.resid

    try:
        nb_model = sm.GLM(
            df_nb['NUM_ADOPTED'].values,
            X_s2.astype(float).values,
            family=sm.families.NegativeBinomial()
        ).fit(maxiter=100)
        coef_nb = nb_model.params[X_s2.columns.get_loc('w_geo_adoption')]
        se_nb = nb_model.bse[X_s2.columns.get_loc('w_geo_adoption')]
        pval_nb = nb_model.pvalues[X_s2.columns.get_loc('w_geo_adoption')]
        print(f"  N={len(df_nb)}: coef={coef_nb:.4f}, SE={se_nb:.4f}, "
              f"p={pval_nb:.4f}{significance_stars(pval_nb)}")
        print(f"  (Note: SEs are from single-stage, not bootstrapped)")
    except Exception as e:
        print(f"  NegBin estimation failed: {e}")
        coef_nb, se_nb, pval_nb = np.nan, np.nan, np.nan

    all_results.append({
        'model': 'NegBin IV (Control Function)', 'k': k, 'n': len(df_nb),
        'coef': coef_nb, 'se': se_nb, 'pval': pval_nb,
        'dv': 'NUM_ADOPTED (count)'
    })

    # -----------------------------------------------------------------
    # Binary comparison (for side-by-side table)
    # -----------------------------------------------------------------
    print(f"\n--- Binary benchmark (IS_ADOPTER, for comparison) ---")

    model_bin = IV2SLS(
        df_reg['IS_ADOPTER'],
        exog,
        df_reg[['w_geo_adoption']],
        df_reg[['w_geo_z']]
    ).fit(cov_type='clustered', clusters=df_reg['state'])

    coef_bin = model_bin.params['w_geo_adoption']
    se_bin = model_bin.std_errors['w_geo_adoption']
    pval_bin = model_bin.pvalues['w_geo_adoption']
    print(f"  N={len(df_reg)}: coef={coef_bin:.4f}, SE={se_bin:.4f}, "
          f"p={pval_bin:.4f}{significance_stars(pval_bin)}")

    all_results.append({
        'model': 'Binary 2SLS (benchmark)', 'k': k, 'n': len(df_reg),
        'coef': coef_bin, 'se': se_bin, 'pval': pval_bin, 'dv': 'IS_ADOPTER'
    })

# =============================================================================
# 3. Save Results
# =============================================================================
results_df = pd.DataFrame(all_results)
results_df.to_csv(str(OUTPUT_FILE), index=False)
print(f"\nResults saved to {OUTPUT_FILE}")

# Comparison table
print(f"\n{'='*80}")
print("SIDE-BY-SIDE COMPARISON")
print(f"{'='*80}")
print(results_df[['model', 'k', 'dv', 'n', 'coef', 'se', 'pval']].to_string(index=False))

print("\nDONE.")
