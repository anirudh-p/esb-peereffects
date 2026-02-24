"""
24_count_based_instrument.py
============================
Cleaner alternative to the ratio-based conditional instrument test.

Instead of instrumenting with win-rate among applicants (w_IV_Z / w_applicant),
this script uses count-based exposure:

    Instrument: # winning neighbors (n_win_neighbors)
    Controls:   # applicant neighbors OR # loser neighbors

This keeps the full sample and avoids ratio-instability when applicant density is low.

Inputs:
    - Cleaned/esb_full_analysis_dataset.csv
    - Raw/Spatial shapefile

Outputs:
    - Tables/count_based_instrument_robustness.csv
    - Logs/count_based_instrument_robustness.txt
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

OUTPUT_CSV = TABLES_DIR / 'count_based_instrument_robustness.csv'
OUTPUT_LOG = LOGS_DIR / 'count_based_instrument_robustness.txt'

K = 6

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(msg)


log("=" * 80)
log("COUNT-BASED INSTRUMENT ROBUSTNESS")
log("Instrument = # winning neighbors, controls = # applicant/#loser neighbors")
log("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

df, locale_vars = load_analysis_data()
geo_df = load_shapefile_and_merge(df)
log(f"\nFull spatial sample: {len(geo_df)}")

# ============================================================================
# 2. BUILD SPATIAL LAGS + COUNT VARIABLES
# ============================================================================

w = build_geo_knn(geo_df, K)

geo_df['w_adoption'] = lag_spatial(w, geo_df['IS_ADOPTER'].values)
geo_df['w_IV_Z'] = lag_spatial(w, geo_df['IV_Z'].fillna(0).values)
geo_df['w_applicant'] = lag_spatial(w, geo_df['IS_APPLICANT'].values)
geo_df['IS_LOSER'] = ((geo_df['IS_APPLICANT'] == 1) & (geo_df['IV_Z'] == 0)).astype(int)
geo_df['w_loser'] = lag_spatial(w, geo_df['IS_LOSER'].values)

# Count versions (K is fixed and row-standardized weights are 1/K)
geo_df['n_win_neighbors'] = geo_df['w_IV_Z'] * K
geo_df['n_app_neighbors'] = geo_df['w_applicant'] * K
geo_df['n_loser_neighbors'] = geo_df['w_loser'] * K

log("\nNeighbor count distributions (K=6):")
for col in ['n_win_neighbors', 'n_app_neighbors', 'n_loser_neighbors']:
    vals = geo_df[col]
    log(f"  {col:18s}: mean={vals.mean():.3f}, p50={vals.median():.3f}, max={vals.max():.1f}")

# ============================================================================
# 3. ESTIMATION SETUP
# ============================================================================

state_dummies = pd.get_dummies(geo_df['state'], prefix='st', drop_first=True)
geo_df = pd.concat([geo_df.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
state_fe_cols = list(state_dummies.columns)

base_controls = FULL_CONTROLS + locale_vars
results = []


def run_iv(label, instr_col, extra_controls=None):
    controls = list(base_controls) + (extra_controls or [])

    needed = ['IS_ADOPTER', 'w_adoption', instr_col, 'state'] + controls + state_fe_cols
    df_r = geo_df.dropna(subset=list(set(needed))).copy()

    y = df_r['IS_ADOPTER'].astype(float)
    X_endog = df_r[['w_adoption']].astype(float)
    X_exog_raw = sm.add_constant(df_r[controls + state_fe_cols].astype(float))
    X_exog = X_exog_raw.loc[:, X_exog_raw.std() > 0]
    Z = df_r[[instr_col]].astype(float)

    # First stage diagnostics
    fs_X = pd.concat([X_exog, Z], axis=1)
    fs = sm.OLS(X_endog.values.ravel(), fs_X.astype(float).values).fit()
    idx_z = fs_X.columns.get_loc(instr_col)
    f_stat = fs.tvalues[idx_z] ** 2
    fs_coef = fs.params[idx_z]

    iv = IV2SLS(
        dependent=y,
        exog=X_exog,
        endog=X_endog,
        instruments=Z,
    ).fit(cov_type='clustered', clusters=df_r['state'])

    peer = iv.params['w_adoption']
    se = iv.std_errors['w_adoption']
    pval = iv.pvalues['w_adoption']

    results.append({
        'spec': label,
        'instrument': instr_col,
        'controls_added': ','.join(extra_controls or []),
        'peer_effect': peer,
        'se': se,
        'pval': pval,
        'stars': significance_stars(pval),
        'F_first': f_stat,
        'fs_coef': fs_coef,
        'N': len(df_r),
        'n_states': df_r['state'].nunique(),
    })

    log(
        f"  {label:45s} | beta={peer:>8.4f} (SE={se:.4f}) p={pval:.4f}{significance_stars(pval)}"
        f" | F={f_stat:.1f} | N={len(df_r)}"
    )


def run_rf(label, instr_col, extra_controls=None):
    controls = list(base_controls) + (extra_controls or [])
    needed = ['IS_ADOPTER', instr_col, 'state'] + controls + state_fe_cols
    df_r = geo_df.dropna(subset=list(set(needed))).copy()

    X = sm.add_constant(
        pd.concat([
            df_r[controls + state_fe_cols],
            df_r[[instr_col]],
        ], axis=1).astype(float)
    )

    idx_z = X.columns.get_loc(instr_col)
    ols = sm.OLS(
        df_r['IS_ADOPTER'].astype(float).values,
        X.values,
    ).fit(cov_type='cluster', cov_kwds={'groups': df_r['state'].values})

    coef = ols.params[idx_z]
    se = ols.bse[idx_z]
    pval = ols.pvalues[idx_z]

    results.append({
        'spec': label,
        'instrument': instr_col,
        'controls_added': ','.join(extra_controls or []),
        'peer_effect': coef,
        'se': se,
        'pval': pval,
        'stars': significance_stars(pval),
        'F_first': np.nan,
        'fs_coef': np.nan,
        'N': len(df_r),
        'n_states': df_r['state'].nunique(),
    })

    log(
        f"  {label:45s} | coef={coef:>8.4f} (SE={se:.4f}) p={pval:.4f}{significance_stars(pval)}"
        f" | N={len(df_r)}"
    )


# ============================================================================
# 4. ESTIMATION
# ============================================================================

log("\n" + "=" * 80)
log("2SLS SPECIFICATIONS")
log("=" * 80)

# Anchor baseline (share instrument)
run_iv(
    label='A1. Baseline share instrument (w_IV_Z)',
    instr_col='w_IV_Z',
    extra_controls=None,
)

# Count instrument without added count controls
run_iv(
    label='A2. Count instrument (# winners)',
    instr_col='n_win_neighbors',
    extra_controls=None,
)

# Count instrument + applicant count control
run_iv(
    label='B1. # winners + # applicants control',
    instr_col='n_win_neighbors',
    extra_controls=['n_app_neighbors'],
)

# Count instrument + loser count control
run_iv(
    label='B2. # winners + # losers control',
    instr_col='n_win_neighbors',
    extra_controls=['n_loser_neighbors'],
)

# Note: do NOT include both #applicants and #losers together with #winners instrument.
# Mechanical identity within K-neighbor sets: n_app_neighbors = n_win_neighbors + n_loser_neighbors,
# which creates perfect collinearity in the IV system.

log("\n" + "=" * 80)
log("REDUCED FORM (ITT) WITH COUNT INSTRUMENT")
log("=" * 80)

run_rf(
    label='C1. RF with # winners (no count controls)',
    instr_col='n_win_neighbors',
    extra_controls=None,
)
run_rf(
    label='C2. RF with # winners + # applicants',
    instr_col='n_win_neighbors',
    extra_controls=['n_app_neighbors'],
)
run_rf(
    label='C3. RF with # winners + # losers',
    instr_col='n_win_neighbors',
    extra_controls=['n_loser_neighbors'],
)

# ============================================================================
# 5. SAVE
# ============================================================================

results_df = pd.DataFrame(results)
results_df.to_csv(str(OUTPUT_CSV), index=False)
log(f"\nResults saved to {OUTPUT_CSV}")

with open(str(OUTPUT_LOG), 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"Log saved to {OUTPUT_LOG}")

log("\n" + "=" * 80)
log("TAKEAWAY")
log("=" * 80)
log(
    """
Count-based instrumentation avoids the ratio-instrument issue by using
absolute local winner exposure (# winners) while directly controlling for
local participation intensity (# applicants or # losers).

Interpretation focus:
- Baseline share instrument and count instrument should align directionally.
- If #winner effects persist after controlling #losers, peer exposure is robust
  to local applicant clustering.
"""
)
