"""
03_estimand1_itt.py
====================
Estimand 1: Cross-sectional Intention-to-Treat (ITT)

Question: Does having lottery-winning neighbours increase own CSBP adoption?

Design:
  Outcome          : IS_ADOPTER (R1+R3 CSBP lottery win = 1)
  Treatment proxy  : w{K}_IV_Z  (share of K nearest neighbours with IV_Z=1)
  Application ctrl : w{K}_IS_LOSER_pooled  (local application density)
  Instrument check : lottery is random → w_IV_Z IS the ITT instrument
  Estimator        : Linear Probability Model; reduced form = ITT
                     2SLS: IS_ADOPTER ~ w_IS_ADOPTER | w_IV_Z (for completeness)
  Sample           : All estimation-sample districts (12,721)
  Controls         : log_enroll, log_income, poverty, pct_white, pm25,
                     priority_r1, pct_dem_2020, state FE

Key identification note: within-state, conditional on neighbour application
density (w_loser), variation in w_IV_Z is driven by lottery luck, not
application propensity. The ITT δ is the headline estimand.

Outputs
-------
  3_Output/Logs/estimand1_itt.txt
  3_Output/Tables/estimand1_itt.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SPATIAL_DATASET, LOGS_DIR, TABLES_DIR, ensure_dirs

warnings.filterwarnings("ignore")
ensure_dirs()

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(SPATIAL_DATASET, low_memory=False, dtype={"nces_id": str})

log("=" * 70)
log("ESTIMAND 1 — Cross-sectional ITT")
log("=" * 70)
log(f"  Full dataset rows: {len(df):,}")

# ── Controls ─────────────────────────────────────────────────────────────────
df["log_enroll"] = np.log(df["enrollment"].clip(lower=1))
df["log_income"] = np.log(df["median_income"].clip(lower=1))

required = ["enrollment", "median_income", "poverty_rate", "pct_white",
            "pm25", "state", "pct_dem_2020", "w6_IV_Z"]
d = df.dropna(subset=required).copy().reset_index(drop=True)
log(f"  Estimation sample: {len(d):,}")

# State fixed effects (drop one reference state to avoid collinearity)
state_dummies = pd.get_dummies(d["state"], prefix="st", drop_first=True, dtype=float)

# ── Common controls vector ────────────────────────────────────────────────────
def get_controls(df_in, include_loser=True, include_state_fe=True, k=6):
    ctrl = pd.DataFrame(index=df_in.index)
    ctrl["log_enroll"] = df_in["log_enroll"]
    ctrl["log_income"]  = df_in["log_income"]
    ctrl["poverty_rate"]= df_in["poverty_rate"]
    ctrl["pct_white"]   = df_in["pct_white"]
    ctrl["pm25"]        = df_in["pm25"]
    ctrl["priority_r1"] = df_in["priority_r1"].fillna(0).astype(float)
    ctrl["pct_dem_2020"]= df_in["pct_dem_2020"]
    if include_loser:
        wcol = f"w{k}_IS_LOSER_pooled"
        ctrl["w_loser"] = df_in[wcol].fillna(0)
    if include_state_fe:
        sdums = pd.get_dummies(df_in["state"], prefix="st", drop_first=True, dtype=float)
        ctrl = pd.concat([ctrl, sdums], axis=1)
    return ctrl


# ══════════════════════════════════════════════════════════════════════════════
# Specification loop — K ∈ {6, 10, 15}, loser control on/off
# ══════════════════════════════════════════════════════════════════════════════
results_rows = []
K_values = [6, 10, 15]

for K in K_values:
    wcol_iv   = f"w{K}_IV_Z"
    wcol_end  = f"w{K}_IS_ADOPTER"
    if wcol_iv not in d.columns or wcol_end not in d.columns:
        log(f"  Skipping K={K}: column not found")
        continue

    d_k = d.dropna(subset=[wcol_iv, wcol_end]).copy()

    for loser_ctrl in [False, True]:
        label = f"K={K}, loser={'Yes' if loser_ctrl else 'No ':3}"
        ctrl = get_controls(d_k, include_loser=loser_ctrl, include_state_fe=True, k=K)
        Y = d_k["IS_ADOPTER"].astype(float)
        # w_IV_Z = treatment variable in reduced form
        W_iv = d_k[wcol_iv].fillna(0)

        # ── Reduced form (OLS) ──
        X_rf = sm.add_constant(pd.concat([W_iv.rename("w_IV_Z"), ctrl], axis=1)
                               , has_constant="add")
        X_rf = X_rf.dropna(axis=1)
        ols_rf = sm.OLS(Y, X_rf).fit(cov_type="cluster",
                                      cov_kwds={"groups": d_k["state"]})

        coef_rf  = ols_rf.params.get("w_IV_Z", np.nan)
        se_rf    = ols_rf.bse.get("w_IV_Z", np.nan)
        pval_rf  = ols_rf.pvalues.get("w_IV_Z", np.nan)
        nobs_rf  = int(ols_rf.nobs)
        ymean    = Y.mean()

        stars = "***" if pval_rf < 0.01 else "**" if pval_rf < 0.05 else "*" if pval_rf < 0.1 else ""
        log(f"\n  [{label}]  RF coef={coef_rf:+.5f}  SE={se_rf:.5f}  "
            f"p={pval_rf:.4f}{stars}  N={nobs_rf:,}  Ymean={ymean:.4f}")

        # ── 2SLS: IS_ADOPTER ~ w_IS_ADOPTER | w_IV_Z ──
        W_end = d_k[wcol_end].fillna(0)
        X_2s  = pd.concat([W_end.rename("w_adopter"), ctrl], axis=1).dropna(axis=1)
        Z_     = W_iv.rename("w_IV_Z")
        try:
            iv_res = IV2SLS(
                dependent = Y,
                exog      = sm.add_constant(ctrl.dropna(axis=1), has_constant="add"),
                endog     = W_end,
                instruments = Z_
            ).fit(cov_type="clustered", clusters=d_k["state"])

            coef_iv = float(iv_res.params.get("w_IS_ADOPTER" if wcol_end.replace(f"w{K}_","w_") in iv_res.params.index else list(iv_res.params.index)[1], np.nan))
            # Get endogenous variable coefficient
            endog_name = [c for c in iv_res.params.index if "adopter" in c.lower() or c == wcol_end]
            coef_iv = float(iv_res.params.iloc[1]) if endog_name else np.nan
            se_iv   = float(iv_res.std_errors.iloc[1]) if endog_name else np.nan
            pval_iv = float(iv_res.pvalues.iloc[1]) if endog_name else np.nan
            fstat   = float(iv_res.first_stage.diagnostics["f.stat"].iloc[0]) if hasattr(iv_res, "first_stage") else np.nan
            log(f"           2SLS coef={coef_iv:+.4f}  SE={se_iv:.4f}  "
                f"p={pval_iv:.4f}  FS F-stat={fstat:.1f}")
        except Exception as e:
            coef_iv = se_iv = pval_iv = fstat = np.nan
            log(f"           2SLS failed: {e}")

        results_rows.append({
            "estimand": "ITT",
            "K"       : K,
            "loser_ctrl": loser_ctrl,
            "spec"    : label,
            "n_obs"   : nobs_rf,
            "y_mean"  : round(ymean, 5),
            "rf_coef" : round(coef_rf, 6),
            "rf_se"   : round(se_rf, 6),
            "rf_pval" : round(pval_rf, 4),
            "rf_stars": stars,
            "iv_coef" : round(coef_iv, 4) if not np.isnan(coef_iv) else np.nan,
            "iv_se"   : round(se_iv,   4) if not np.isnan(se_iv) else np.nan,
            "iv_pval" : round(pval_iv, 4) if not np.isnan(pval_iv) else np.nan,
            "fs_fstat": round(fstat, 1)   if not np.isnan(fstat) else np.nan,
        })


# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("SUMMARY TABLE — Estimand 1 ITT reduced forms")
log("=" * 70)
res = pd.DataFrame(results_rows)

log(f"\n{'Spec':<30} {'N':>7} {'RF coef':>9} {'SE':>8} {'p-val':>7} {'Stars':>5}")
log("-" * 65)
for _, r in res.iterrows():
    log(f"  {r['spec']:<28} {r['n_obs']:>7,} {r['rf_coef']:>+9.5f} "
        f"{r['rf_se']:>8.5f} {r['rf_pval']:>7.4f} {r['rf_stars']:>5}")

log("\nInterpretation note:")
log("  RF coef = δ = marginal effect of a 1/K increase in the share of lottery-winning")
log("  neighbours on own CSBP adoption probability (ITT, LPM), state-clustered SE.")
log("  The 'loser=Yes' spec controls for neighbour application density (w_loser_pooled).")
log("  A positive δ that survives the loser control is attributable to lottery LUCK,")
log("  not merely geographic clustering of application propensity.")

# Save
res.to_csv(TABLES_DIR / "estimand1_itt.csv", index=False)
log(f"\n  Saved: {TABLES_DIR / 'estimand1_itt.csv'}")

log_text = "\n".join(log_lines)
with open(LOGS_DIR / "estimand1_itt.txt", "w", encoding="utf-8") as f:
    f.write(log_text)
log(f"  Saved: {LOGS_DIR / 'estimand1_itt.txt'}")
