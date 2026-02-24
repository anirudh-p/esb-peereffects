"""
21_application_density_control.py
==================================
Address the exclusion restriction concern that w_IV_Z proxies for 
local ESB "interest intensity" rather than pure lottery luck.

Two approaches:
  A) Add w_A_i (neighbor application density) as an exogenous control, 
     so w_IV_Z isolates luck AMONG applicants, not applicant presence.
  B) Construct a conditional win-rate instrument: among neighbor-applicants,
     what fraction won? (Undefined when no neighbors applied — those obs 
     are dropped.)

Both target the concern: "variation in w_IV_Z partly reflects how many 
nearby districts were in the game."

Inputs:
    - Cleaned/esb_full_analysis_dataset.csv
    - Raw/Spatial shapefile
    - Raw/Political county presidential data

Outputs:
    - Tables/application_density_robustness.csv
    - Logs/application_density_robustness.txt
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
    build_geo_knn, FULL_CONTROLS, significance_stars
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TABLES_DIR, LOGS_DIR, ensure_dirs_exist

import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from libpysal.weights import lag_spatial

ensure_dirs_exist()

OUTPUT_CSV = TABLES_DIR / 'application_density_robustness.csv'
OUTPUT_LOG = LOGS_DIR / 'application_density_robustness.txt'

K = 6  # primary spec

log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(msg)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

log("=" * 80)
log("APPLICATION DENSITY CONTROL & CONDITIONAL WIN-RATE INSTRUMENT")
log("=" * 80)

df, locale_vars = load_analysis_data()
geo_df = load_shapefile_and_merge(df)

log(f"\nFull spatial sample: {len(geo_df)}")
log(f"  IS_APPLICANT==1: {(geo_df['IS_APPLICANT']==1).sum()}")
log(f"  IV_Z==1 (winners): {(geo_df['IV_Z']==1).sum()}")

# ==============================================================================
# 2. BUILD SPATIAL LAGS
# ==============================================================================

log(f"\nBuilding KNN-{K} weights...")
w = build_geo_knn(geo_df, K)

# Standard lags
geo_df['w_adoption'] = lag_spatial(w, geo_df['IS_ADOPTER'].values)
geo_df['w_IV_Z'] = lag_spatial(w, geo_df['IV_Z'].fillna(0).values)

# NEW: Neighbor application density
geo_df['w_applicant'] = lag_spatial(w, geo_df['IS_APPLICANT'].values)

# CRITICAL: Also compute loser-only density
# IS_APPLICANT includes both winners and losers. Since IS_ADOPTER ⊂ IS_APPLICANT,
# controlling for w_applicant creates a bad-control problem (it contains w_adoption).
# The clean control is w_loser = spatial lag of lottery losers only.
geo_df['IS_LOSER'] = ((geo_df['IS_APPLICANT'] == 1) & (geo_df['IV_Z'] == 0)).astype(int)
geo_df['w_loser'] = lag_spatial(w, geo_df['IS_LOSER'].values)

log(f"  w_adoption range:  [{geo_df['w_adoption'].min():.3f}, {geo_df['w_adoption'].max():.3f}]")
log(f"  w_IV_Z range:      [{geo_df['w_IV_Z'].min():.3f}, {geo_df['w_IV_Z'].max():.3f}]")
log(f"  w_applicant range: [{geo_df['w_applicant'].min():.3f}, {geo_df['w_applicant'].max():.3f}]")
log(f"  w_applicant mean:  {geo_df['w_applicant'].mean():.4f}")
log(f"  w_loser range:     [{geo_df['w_loser'].min():.3f}, {geo_df['w_loser'].max():.3f}]")
log(f"  w_loser mean:      {geo_df['w_loser'].mean():.4f}")
log(f"  IS_LOSER count:    {geo_df['IS_LOSER'].sum()}")

# ==============================================================================
# 3. CONDITIONAL WIN-RATE INSTRUMENT
# ==============================================================================

# For each district, compute: (# neighbor winners) / (# neighbor applicants)
# This is undefined (NaN) when no neighbors applied.
# w_IV_Z = sum(Z_j) / K  and  w_applicant = sum(A_j) / K
# So conditional win rate = w_IV_Z / w_applicant (when w_applicant > 0)

geo_df['w_IV_Z_cond'] = np.where(
    geo_df['w_applicant'] > 0,
    geo_df['w_IV_Z'] / geo_df['w_applicant'],
    np.nan
)

n_with_applicant_neighbors = (geo_df['w_applicant'] > 0).sum()
n_without = (geo_df['w_applicant'] == 0).sum()
log(f"\n  Districts with >=1 applicant neighbor: {n_with_applicant_neighbors}")
log(f"  Districts with 0 applicant neighbors:  {n_without}")
log(f"  w_IV_Z_cond range (when defined): [{geo_df['w_IV_Z_cond'].min():.3f}, {geo_df['w_IV_Z_cond'].max():.3f}]")

# ==============================================================================
# 4. ESTIMATION SETUP
# ==============================================================================

# State FE
state_dummies = pd.get_dummies(geo_df['state'], prefix='st', drop_first=True)
geo_df = pd.concat([geo_df.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
state_fe_cols = list(state_dummies.columns)

base_controls = FULL_CONTROLS + locale_vars

results = []

def run_spec(label, df_est, y_col, endog_col, instr_col, extra_controls=None):
    """Run a single IV-2SLS specification and collect results."""
    controls = list(base_controls) + (extra_controls or [])
    
    needed = [y_col, endog_col, instr_col, 'state'] + controls + state_fe_cols
    df_r = df_est.dropna(subset=list(set(needed))).copy()
    
    y = df_r[y_col].astype(float)
    X_endog = df_r[[endog_col]].astype(float)
    X_exog_raw = sm.add_constant(df_r[controls + state_fe_cols].astype(float))
    # Drop zero-variance columns to avoid rank deficiency
    X_exog = X_exog_raw.loc[:, X_exog_raw.std() > 0]
    Z = df_r[[instr_col]].astype(float)
    
    # First stage
    fs_X = pd.concat([X_exog, Z], axis=1)
    fs = sm.OLS(X_endog.values.ravel(), fs_X.astype(float).values).fit()
    idx_z = fs_X.columns.get_loc(instr_col)
    f_stat = fs.tvalues[idx_z] ** 2
    fs_coef = fs.params[idx_z]
    
    # IV-2SLS
    iv = IV2SLS(dependent=y, exog=X_exog, endog=X_endog, instruments=Z
                ).fit(cov_type='clustered', clusters=df_r['state'])
    
    peer = iv.params[endog_col]
    se = iv.std_errors[endog_col]
    pval = iv.pvalues[endog_col]
    
    row = {
        'spec': label,
        'N': len(df_r),
        'peer_effect': peer,
        'se': se,
        'pval': pval,
        'stars': significance_stars(pval),
        'F_first': f_stat,
        'fs_coef': fs_coef,
        'n_states': df_r['state'].nunique(),
    }
    results.append(row)
    
    sig = significance_stars(pval)
    log(f"  {label:50s} | β={peer:>8.4f} (SE={se:.4f}) p={pval:.4f}{sig} | F={f_stat:.1f} | N={len(df_r)}")
    return iv

# ==============================================================================
# 5. ESTIMATION
# ==============================================================================

log("\n" + "=" * 80)
log("ESTIMATION RESULTS")
log("=" * 80)

# --- A. Baseline (no application density control) ---
log("\n--- A. Baseline (replicating Script 11) ---")
run_spec("A1. Baseline (no w_applicant)", geo_df, 'IS_ADOPTER', 'w_adoption', 'w_IV_Z')

# --- B. Add w_applicant as exogenous control ---
log("\n--- B. Add neighbor application density (w_applicant) control ---")
log("  NOTE: w_applicant includes IS_ADOPTER (bad control concern).")
run_spec("B1. + w_applicant control (BAD CTRL)", geo_df, 'IS_ADOPTER', 'w_adoption', 'w_IV_Z',
         extra_controls=['w_applicant'])

# --- B2. Add w_loser as exogenous control (CORRECT) ---
log("\n--- B2. Add neighbor LOSER density (w_loser) control ---")
log("  w_loser = fraction of neighbors who applied but LOST.")
log("  This captures local ESB interest WITHOUT containing adoption.")
run_spec("B2. + w_loser control (CORRECT)", geo_df, 'IS_ADOPTER', 'w_adoption', 'w_IV_Z',
         extra_controls=['w_loser'])

# --- C. Conditional win-rate instrument ---
log("\n--- C. Conditional win-rate instrument (w_IV_Z_cond) ---")

# C1: on full sample with applicant neighbors
df_cond = geo_df[geo_df['w_applicant'] > 0].copy()
log(f"  Conditional sample (≥1 applicant neighbor): {len(df_cond)}")

# Rebuild spatial lags for conditional adoption among neighbors
# w_adoption is already computed on full network, which is correct — 
# the point is just to change the instrument
run_spec("C1. Conditional instrument (w_IV_Z_cond)", df_cond, 'IS_ADOPTER', 'w_adoption', 'w_IV_Z_cond')

# C2: conditional instrument + w_applicant control
run_spec("C2. Conditional instr + w_applicant", df_cond, 'IS_ADOPTER', 'w_adoption', 'w_IV_Z_cond',
         extra_controls=['w_applicant'])

# --- D. K sensitivity with w_loser control (correct) ---
log("\n--- D. K sensitivity with w_loser control ---")
for k in [4, 6, 8, 10]:
    w_k = build_geo_knn(geo_df, k)
    geo_df[f'w_adoption_k{k}'] = lag_spatial(w_k, geo_df['IS_ADOPTER'].values)
    geo_df[f'w_IV_Z_k{k}'] = lag_spatial(w_k, geo_df['IV_Z'].fillna(0).values)
    geo_df[f'w_loser_k{k}'] = lag_spatial(w_k, geo_df['IS_LOSER'].values)
    
    run_spec(f"D. K={k} + w_loser", geo_df, 'IS_ADOPTER',
             f'w_adoption_k{k}', f'w_IV_Z_k{k}',
             extra_controls=[f'w_loser_k{k}'])

# --- E. Reduced form: w_IV_Z -> IS_ADOPTER directly ---
log("\n--- E. Reduced Form (ITT: w_IV_Z -> IS_ADOPTER) ---")

for label_suffix, extra in [("baseline", []), ("+ w_loser", ['w_loser']), ("+ w_applicant", ['w_applicant'])]:
    controls_rf = base_controls + extra + state_fe_cols
    needed_rf = ['IS_ADOPTER', 'w_IV_Z', 'state'] + controls_rf
    df_rf = geo_df.dropna(subset=list(set(needed_rf))).copy()
    
    X_rf = sm.add_constant(pd.concat([
        df_rf[base_controls + extra + state_fe_cols],
        df_rf[['w_IV_Z']]
    ], axis=1).astype(float))
    
    idx_z = X_rf.columns.get_loc('w_IV_Z')
    ols_rf = sm.OLS(df_rf['IS_ADOPTER'].astype(float).values, X_rf.values).fit(
        cov_type='cluster', cov_kwds={'groups': df_rf['state'].values}
    )
    
    coef_rf = ols_rf.params[idx_z]
    se_rf = ols_rf.bse[idx_z]
    pval_rf = ols_rf.pvalues[idx_z]
    sig_rf = significance_stars(pval_rf)
    
    results.append({
        'spec': f"E. Reduced form ({label_suffix})",
        'N': len(df_rf),
        'peer_effect': coef_rf,
        'se': se_rf,
        'pval': pval_rf,
        'stars': sig_rf,
        'F_first': np.nan,
        'fs_coef': np.nan,
        'n_states': df_rf['state'].nunique(),
    })
    
    log(f"  RF ({label_suffix:15s}): coef={coef_rf:.4f} (SE={se_rf:.4f}) p={pval_rf:.4f}{sig_rf} | N={len(df_rf)}")

# ==============================================================================
# 6. SAVE
# ==============================================================================

results_df = pd.DataFrame(results)
results_df.to_csv(str(OUTPUT_CSV), index=False)
log(f"\nResults saved to {OUTPUT_CSV}")

with open(str(OUTPUT_LOG), 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"Log saved to {OUTPUT_LOG}")

# ==============================================================================
# TAKEAWAY
# ==============================================================================

log("\n" + "=" * 80)
log("TAKEAWAY")
log("=" * 80)
log("""
The key concern is that w_IV_Z (spatial lag of lottery wins) also captures
"how many nearby districts applied" — confounding lottery luck with local 
ESB interest. Two defenses:

1. Adding w_applicant (neighbor application density) as a control absorbs
   the "interest intensity" channel. If the peer effect survives, w_IV_Z 
   is identifying luck conditional on local participation.

2. The conditional win-rate instrument (w_IV_Z / w_applicant) directly 
   measures luck among those who applied, purging the extensivemargin.
   Sample is restricted to districts with ≥1 applicant neighbor.

The reduced-form (ITT) estimates show the causal effect of w_IV_Z on 
adoption directly — this is the "intention-to-treat" analog that requires
only lottery randomness, not the full exclusion restriction.
""")
