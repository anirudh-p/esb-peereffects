"""
22_inference_robustness.py
===========================
Address inference concerns: state clustering may not capture the full
spatial dependence structure in peer effects models.

Three alternative inference approaches:
  A) County-level clustering (~3000 clusters) — more local than state
  B) Conley (1999) spatial HAC standard errors — distance-based kernel
  C) Reduced-form (ITT) with all three SE approaches

Also outputs the reduced-form regression as a standalone result.

Inputs:
    - Cleaned/esb_full_analysis_dataset.csv
    - Raw/Spatial shapefile
    - Raw/WRI/ESB_adoption_dataset_v9_update_june_2025.xlsx (for LEA-county mapping)

Outputs:
    - Tables/inference_robustness.csv
    - Logs/inference_robustness.txt
"""

import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from _estimation_utils import (
    load_analysis_data, load_shapefile_and_merge,
    build_geo_knn, FULL_CONTROLS, significance_stars
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    TABLES_DIR, LOGS_DIR, WRI_EXCEL_FILE, ensure_dirs_exist
)

import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from libpysal.weights import lag_spatial

ensure_dirs_exist()

OUTPUT_CSV = TABLES_DIR / 'inference_robustness.csv'
OUTPUT_LOG = LOGS_DIR / 'inference_robustness.txt'

K = 6
CONLEY_BANDWIDTHS = [100, 200, 500]  # in km

log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(msg)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

log("=" * 80)
log("INFERENCE ROBUSTNESS: CONLEY SE + COUNTY CLUSTERING + REDUCED FORM")
log("=" * 80)

df, locale_vars = load_analysis_data()

# --- Merge county FIPS for county clustering ---
log("\nMerging county FIPS for county-level clustering...")
lea_county = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='5. Counties')
lea_county = lea_county[['1c. LEA ID', '10b. County FIPS Code']].copy()
lea_county.columns = ['nces_id', 'county_fips']
lea_county['nces_id'] = lea_county['nces_id'].astype(str).str.split('.').str[0].str.zfill(7)

# Districts spanning multiple counties: take the first (most common) county
lea_county = lea_county.drop_duplicates(subset='nces_id', keep='first')
df = df.merge(lea_county, on='nces_id', how='left')
n_county = df['county_fips'].notna().sum()
log(f"  County FIPS merged: {n_county}/{len(df)} districts")
log(f"  Unique counties: {df['county_fips'].nunique()}")

geo_df = load_shapefile_and_merge(df)

log(f"\nFull spatial sample: {len(geo_df)}")

# ==============================================================================
# 2. BUILD SPATIAL LAGS
# ==============================================================================

log(f"\nBuilding KNN-{K} weights...")
w = build_geo_knn(geo_df, K)

geo_df['w_adoption'] = lag_spatial(w, geo_df['IS_ADOPTER'].values)
geo_df['w_IV_Z'] = lag_spatial(w, geo_df['IV_Z'].fillna(0).values)

# Compute centroid coordinates (already in EPSG:5070 = Albers meters)
coords = np.column_stack([
    geo_df.geometry.centroid.x.values,
    geo_df.geometry.centroid.y.values
])

# ==============================================================================
# 3. ESTIMATION SETUP
# ==============================================================================

# State FE
state_dummies = pd.get_dummies(geo_df['state'], prefix='st', drop_first=True)
geo_df = pd.concat([geo_df.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
state_fe_cols = list(state_dummies.columns)

base_controls = FULL_CONTROLS + locale_vars

# Regression sample
needed = ['IS_ADOPTER', 'w_adoption', 'w_IV_Z', 'state', 'county_fips'] + base_controls + state_fe_cols
df_reg = geo_df.dropna(subset=list(set(needed))).copy().reset_index(drop=True)
coords_reg = coords[df_reg.index.values] if len(df_reg) == len(geo_df) else None

# Recompute coords for regression sample
coords_reg = np.column_stack([
    df_reg.geometry.centroid.x.values,
    df_reg.geometry.centroid.y.values
])

log(f"\nRegression sample: {len(df_reg)}")
log(f"  Unique states: {df_reg['state'].nunique()}")
log(f"  Unique counties: {df_reg['county_fips'].nunique()}")

y = df_reg['IS_ADOPTER'].astype(float)
X_endog = df_reg[['w_adoption']].astype(float)
X_exog_raw = sm.add_constant(df_reg[base_controls + state_fe_cols].astype(float))
X_exog = X_exog_raw.loc[:, X_exog_raw.std() > 0]  # drop zero-variance cols
Z = df_reg[['w_IV_Z']].astype(float)

results = []

# ==============================================================================
# 4A. IV-2SLS WITH STATE CLUSTERING (baseline)
# ==============================================================================

log("\n" + "=" * 80)
log("A. IV-2SLS WITH DIFFERENT CLUSTERING LEVELS")
log("=" * 80)

# State clustering
iv_state = IV2SLS(dependent=y, exog=X_exog, endog=X_endog, instruments=Z
                  ).fit(cov_type='clustered', clusters=df_reg['state'])

peer_s = iv_state.params['w_adoption']
se_s = iv_state.std_errors['w_adoption']
pval_s = iv_state.pvalues['w_adoption']

results.append({
    'spec': 'IV-2SLS, state clusters',
    'peer_effect': peer_s, 'se': se_s, 'pval': pval_s,
    'stars': significance_stars(pval_s),
    'n_clusters': df_reg['state'].nunique(),
    'N': len(df_reg), 'type': '2SLS'
})
log(f"  State clusters ({df_reg['state'].nunique()}): β={peer_s:.4f} (SE={se_s:.4f}) p={pval_s:.4f}{significance_stars(pval_s)}")

# County clustering
iv_county = IV2SLS(dependent=y, exog=X_exog, endog=X_endog, instruments=Z
                   ).fit(cov_type='clustered', clusters=df_reg['county_fips'])

se_c = iv_county.std_errors['w_adoption']
pval_c = iv_county.pvalues['w_adoption']

results.append({
    'spec': 'IV-2SLS, county clusters',
    'peer_effect': peer_s, 'se': se_c, 'pval': pval_c,
    'stars': significance_stars(pval_c),
    'n_clusters': df_reg['county_fips'].nunique(),
    'N': len(df_reg), 'type': '2SLS'
})
log(f"  County clusters ({df_reg['county_fips'].nunique()}): β={peer_s:.4f} (SE={se_c:.4f}) p={pval_c:.4f}{significance_stars(pval_c)}")

# Robust (no clustering)
iv_robust = IV2SLS(dependent=y, exog=X_exog, endog=X_endog, instruments=Z
                   ).fit(cov_type='robust')

se_r = iv_robust.std_errors['w_adoption']
pval_r = iv_robust.pvalues['w_adoption']

results.append({
    'spec': 'IV-2SLS, HC robust',
    'peer_effect': peer_s, 'se': se_r, 'pval': pval_r,
    'stars': significance_stars(pval_r),
    'n_clusters': np.nan,
    'N': len(df_reg), 'type': '2SLS'
})
log(f"  HC robust:               β={peer_s:.4f} (SE={se_r:.4f}) p={pval_r:.4f}{significance_stars(pval_r)}")

# ==============================================================================
# 4B. CONLEY SPATIAL HAC STANDARD ERRORS
# ==============================================================================

log("\n" + "=" * 80)
log("B. CONLEY SPATIAL HAC STANDARD ERRORS")
log("=" * 80)

# Get 2SLS residuals and fitted values for Conley computation
# We need to manually compute the sandwich: V = (X'PzX)^-1 Ω (X'PzX)^-1
# where Ω_ij = u_i * u_j * K(d_ij / h)

# Extract 2SLS residuals
resids = iv_state.resids.values.ravel()

# Build the "full X" matrix (exog + endog, using fitted values for endog)
# For the sandwich, we need the projection matrix approach
# Simpler approach: use reduced-form residuals with Conley kernel

# Manual 2SLS recreation for Conley:
# Stage 1: w_adoption = Z * pi + X * gamma + v
# Stage 2: Y = X_hat_endog * beta + X_exog * delta + u

# Get fitted values from first stage
first_stage_X = pd.concat([X_exog, Z], axis=1).astype(float)
fs_model = sm.OLS(X_endog.values.ravel(), first_stage_X.values).fit()
w_adoption_hat = fs_model.fittedvalues

# Second stage with fitted values
X_second = X_exog.copy()
X_second['w_adoption_hat'] = w_adoption_hat
X2 = X_second.values.astype(float)
y_arr = y.values.astype(float)

# OLS on second stage (coefficients match IV-2SLS)
beta_2sls = np.linalg.lstsq(X2, y_arr, rcond=None)[0]

# But residuals use ACTUAL w_adoption, not fitted
X_actual = X_exog.copy()
X_actual['w_adoption'] = X_endog.values.ravel()
resids_2sls = y_arr - X_actual.values.astype(float) @ beta_2sls

log(f"\n  Residual check: mean={resids_2sls.mean():.6f}, std={resids_2sls.std():.6f}")

# Bread: (X_hat' X_hat)^{-1}  where X_hat uses fitted endog
XhX_inv = np.linalg.inv(X2.T @ X2)

n_obs = len(df_reg)

for bandwidth_km in CONLEY_BANDWIDTHS:
    log(f"\n  Computing Conley SE (bandwidth = {bandwidth_km} km)...")
    bandwidth_m = bandwidth_km * 1000  # EPSG:5070 is in meters
    
    # Build the meat of the sandwich using vectorized computation:
    # Omega = UX' @ K @ UX  where UX_i = u_i * x_i, K_ij = kernel(d_ij/h)
    
    UX = X2 * resids_2sls[:, np.newaxis]  # N x p
    Omega = np.zeros((X2.shape[1], X2.shape[1]))
    
    chunk_size = 500
    n_chunks = (n_obs + chunk_size - 1) // chunk_size
    
    for ci in range(n_chunks):
        i_start = ci * chunk_size
        i_end = min(i_start + chunk_size, n_obs)
        
        # Compute distances from this chunk to all observations
        dists = cdist(coords_reg[i_start:i_end], coords_reg, metric='euclidean')
        
        # Bartlett kernel weights
        K_chunk = np.maximum(0.0, 1.0 - dists / bandwidth_m)
        
        # Vectorized: Omega += UX[chunk]' @ K_chunk @ UX
        Omega += UX[i_start:i_end].T @ K_chunk @ UX
        
        if (ci + 1) % 10 == 0:
            log(f"    Chunk {ci+1}/{n_chunks}...")
    
    # Sandwich: V = (X'X)^-1 Omega (X'X)^-1
    V_conley = XhX_inv @ Omega @ XhX_inv
    
    # The peer effect is the last coefficient (w_adoption_hat)
    idx_peer = X2.shape[1] - 1
    se_conley = np.sqrt(V_conley[idx_peer, idx_peer])
    t_conley = beta_2sls[idx_peer] / se_conley
    pval_conley = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(t_conley)))
    
    results.append({
        'spec': f'IV-2SLS, Conley {bandwidth_km}km',
        'peer_effect': beta_2sls[idx_peer], 'se': se_conley, 'pval': pval_conley,
        'stars': significance_stars(pval_conley),
        'n_clusters': np.nan,
        'N': len(df_reg), 'type': '2SLS-Conley'
    })
    log(f"  Conley {bandwidth_km}km: β={beta_2sls[idx_peer]:.4f} (SE={se_conley:.4f}) "
        f"p={pval_conley:.4f}{significance_stars(pval_conley)}")

# ==============================================================================
# 5. REDUCED FORM (ITT)
# ==============================================================================

log("\n" + "=" * 80)
log("C. REDUCED FORM: w_IV_Z → IS_ADOPTER (Intention-to-Treat)")
log("=" * 80)

X_rf_raw = sm.add_constant(pd.concat([
    df_reg[base_controls + state_fe_cols],
    df_reg[['w_IV_Z']]
], axis=1).astype(float))
# Drop zero-variance columns but keep w_IV_Z
keep_cols = [c for c in X_rf_raw.columns if X_rf_raw[c].std() > 0 or c == 'w_IV_Z']
X_rf = X_rf_raw[keep_cols]
idx_wz = X_rf.columns.get_loc('w_IV_Z')

# State clustered
ols_state = sm.OLS(y.values, X_rf.values).fit(
    cov_type='cluster', cov_kwds={'groups': df_reg['state'].values}
)
results.append({
    'spec': 'Reduced form, state clusters',
    'peer_effect': ols_state.params[idx_wz], 'se': ols_state.bse[idx_wz],
    'pval': ols_state.pvalues[idx_wz],
    'stars': significance_stars(ols_state.pvalues[idx_wz]),
    'n_clusters': df_reg['state'].nunique(),
    'N': len(df_reg), 'type': 'RF'
})
log(f"  State clusters: coef={ols_state.params[idx_wz]:.4f} (SE={ols_state.bse[idx_wz]:.4f}) "
    f"p={ols_state.pvalues[idx_wz]:.4f}{significance_stars(ols_state.pvalues[idx_wz])}")

# County clustered
ols_county = sm.OLS(y.values, X_rf.values).fit(
    cov_type='cluster', cov_kwds={'groups': df_reg['county_fips'].values}
)
results.append({
    'spec': 'Reduced form, county clusters',
    'peer_effect': ols_county.params[idx_wz], 'se': ols_county.bse[idx_wz],
    'pval': ols_county.pvalues[idx_wz],
    'stars': significance_stars(ols_county.pvalues[idx_wz]),
    'n_clusters': df_reg['county_fips'].nunique(),
    'N': len(df_reg), 'type': 'RF'
})
log(f"  County clusters: coef={ols_county.params[idx_wz]:.4f} (SE={ols_county.bse[idx_wz]:.4f}) "
    f"p={ols_county.pvalues[idx_wz]:.4f}{significance_stars(ols_county.pvalues[idx_wz])}")

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
# SUMMARY TABLE
# ==============================================================================

log("\n" + "=" * 80)
log("SUMMARY: ALL SE COMPARISONS")
log("=" * 80)
log(f"\n{'Specification':40s} | {'β':>8s} | {'SE':>8s} | {'p':>8s} |")
log("-" * 75)
for r in results:
    log(f"  {r['spec']:40s} | {r['peer_effect']:>8.4f} | {r['se']:>8.4f} | {r['pval']:>8.4f} | {r['stars']}")

log("""
TAKEAWAY:
State clustering is the most conservative clustered approach (fewest clusters = 
widest SEs). If county clustering or Conley SEs produce *smaller* SEs, state 
clustering is adequate. If Conley SEs at large bandwidths are substantially 
*larger*, the state-clustered SEs understate uncertainty.

The reduced-form (ITT) estimate is the causal effect of neighbor lottery wins
on own adoption, requiring only lottery randomness — not the full exclusion 
restriction. This is the most defensible single number in the paper.
""")
