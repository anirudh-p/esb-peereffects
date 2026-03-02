"""
02_estimand2_application.py
============================
Estimand 2: Application Extensive Margin (Clean Causal Design)

Question: Does having R1-winning neighbours increase your probability of
          applying to R3?

Why this is cleaner than Estimand 1:
  - R1 lottery (2022) strictly precedes R3 application window (Oct 2023)
  - Sample restricted to non-R1-winners → no contamination of own-win status
  - Application is the direct BEHAVIORAL result of belief updating, detached
    from the subsequent random lottery draw in R3

Design:
  Outcome    : Y_R3_apply  (applied to R3, win OR waitlist = 1, else 0)
  Instrument : w{K}_IV_Z_R1  (share of R1 lottery winners among K neighbours)
  Loser ctrl : w{K}_IS_R1_LOSER  (share of R1 non-winners who applied → absorbs
                local application propensity in R1)
  Sample     : non-R1-winners who are in the estimation sample
  Estimator  : Linear Probability Model, state-clustered SE
  Headline   : Reduced form (ITT δ). 2SLS is w_adopter_R1 → w_IV_Z_R1 but
               since R1-win ≈ R1-adoption, first stage F is near-perfect
               (this is fine here — we're intentionally using the RF).

Outputs
-------
  3_Output/Tables/estimand2_application.csv
  3_Output/Logs/estimand2_application.txt
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
# Load and prepare
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(SPATIAL_DATASET, low_memory=False, dtype={"nces_id": str})
df["log_enroll"] = np.log(df["enrollment"].clip(lower=1))
df["log_income"] = np.log(df["median_income"].clip(lower=1))

log("=" * 70)
log("ESTIMAND 2 — Application Extensive Margin")
log("=" * 70)
log(f"  Full dataset: {len(df):,}")

# Sample: non-R1-winners with all controls
required = ["enrollment", "median_income", "poverty_rate", "pct_white",
            "pm25", "state", "pct_dem_2020", "w6_IV_Z_R1"]
d = df[df["IV_Z_R1"] == 0].dropna(subset=required).copy().reset_index(drop=True)
log(f"  Non-R1-winner estimation sample: {len(d):,}")
log(f"  Of which applied to R3 (Y=1): {d['Y_R3_apply'].sum():,}  "
    f"({100*d['Y_R3_apply'].mean():.2f}%)")


def get_controls(df_in, include_r1_loser=True, include_state_fe=True, k=6):
    ctrl = pd.DataFrame(index=df_in.index)
    ctrl["log_enroll"]  = df_in["log_enroll"]
    ctrl["log_income"]  = df_in["log_income"]
    ctrl["poverty_rate"]= df_in["poverty_rate"]
    ctrl["pct_white"]   = df_in["pct_white"]
    ctrl["pm25"]        = df_in["pm25"]
    ctrl["priority_r1"] = df_in["priority_r1"].fillna(0).astype(float)
    ctrl["priority_r23"]= df_in["priority_r23"].fillna(0).astype(float)
    ctrl["pct_dem_2020"]= df_in["pct_dem_2020"]
    if include_r1_loser:
        ctrl["w_r1_loser"] = df_in[f"w{k}_IS_R1_LOSER"].fillna(0)
    if include_state_fe:
        sdums = pd.get_dummies(df_in["state"], prefix="st", drop_first=True, dtype=float)
        ctrl = pd.concat([ctrl, sdums], axis=1)
    return ctrl


# ══════════════════════════════════════════════════════════════════════════════
# Specification loop
# ══════════════════════════════════════════════════════════════════════════════
results_rows = []
K_values = [6, 10, 15]

for K in K_values:
    wcol = f"w{K}_IV_Z_R1"
    if wcol not in d.columns:
        log(f"  Skipping K={K}: {wcol} not found")
        continue

    d_k = d.dropna(subset=[wcol]).copy()

    for loser_ctrl in [False, True]:
        label = f"K={K}, R1-loser={'Yes' if loser_ctrl else 'No ':3}"
        ctrl  = get_controls(d_k, include_r1_loser=loser_ctrl, include_state_fe=True, k=K)
        Y     = d_k["Y_R3_apply"].astype(float)
        W_iv  = d_k[wcol].fillna(0)

        X_rf = sm.add_constant(pd.concat([W_iv.rename("w_IV_Z_R1"), ctrl], axis=1),
                               has_constant="add").dropna(axis=1)
        ols_rf = sm.OLS(Y, X_rf).fit(cov_type="cluster",
                                      cov_kwds={"groups": d_k["state"]})

        coef_rf = ols_rf.params.get("w_IV_Z_R1", np.nan)
        se_rf   = ols_rf.bse.get("w_IV_Z_R1",   np.nan)
        pval_rf = ols_rf.pvalues.get("w_IV_Z_R1", np.nan)
        nobs_rf = int(ols_rf.nobs)
        ymean   = Y.mean()
        stars   = "***" if pval_rf < 0.01 else "**" if pval_rf < 0.05 else "*" if pval_rf < 0.1 else ""

        log(f"\n  [{label}]  RF coef={coef_rf:+.5f}  SE={se_rf:.5f}  "
            f"p={pval_rf:.4f}{stars}  N={nobs_rf:,}  Ymean={ymean:.4f}")

        results_rows.append({
            "estimand"   : "App_Margin",
            "K"          : K,
            "loser_ctrl" : loser_ctrl,
            "spec"       : label,
            "n_obs"      : nobs_rf,
            "y_mean"     : round(ymean, 5),
            "rf_coef"    : round(coef_rf, 6),
            "rf_se"      : round(se_rf, 6),
            "rf_pval"    : round(pval_rf, 4),
            "rf_stars"   : stars,
        })

# ── Priority-district subsample ────────────────────────────────────────────
log("\n" + "─" * 60)
log("  Subsample: Priority-district applicants only (K=6)")
log("─" * 60)
d_pri = d[d["priority_r1"] == 1].dropna(subset=["w6_IV_Z_R1"]).copy()
log(f"  Priority-district non-R1-winners: {len(d_pri):,}  "
    f"(R3-apply rate: {100*d_pri['Y_R3_apply'].mean():.2f}%)")

for loser_ctrl in [False, True]:
    ctrl  = get_controls(d_pri, include_r1_loser=loser_ctrl, include_state_fe=True, k=6)
    Y     = d_pri["Y_R3_apply"].astype(float)
    W_iv  = d_pri["w6_IV_Z_R1"].fillna(0)
    X_rf  = sm.add_constant(pd.concat([W_iv.rename("w_IV_Z_R1"), ctrl], axis=1),
                            has_constant="add").dropna(axis=1)
    ols   = sm.OLS(Y, X_rf).fit(cov_type="cluster",
                                 cov_kwds={"groups": d_pri["state"]})
    coef  = ols.params.get("w_IV_Z_R1", np.nan)
    se    = ols.bse.get("w_IV_Z_R1", np.nan)
    pval  = ols.pvalues.get("w_IV_Z_R1", np.nan)
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    label = f"  Priority subset, loser={'Yes' if loser_ctrl else 'No ':3}"
    log(f"{label:<45}  coef={coef:+.5f}  SE={se:.5f}  p={pval:.4f}{stars}  N={len(d_pri):,}")
    results_rows.append({
        "estimand"  : "App_Margin_Priority",
        "K": 6, "loser_ctrl": loser_ctrl, "spec": label,
        "n_obs": len(d_pri), "y_mean": round(Y.mean(), 5),
        "rf_coef": round(coef, 6), "rf_se": round(se, 6),
        "rf_pval": round(pval, 4), "rf_stars": stars,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("SUMMARY — Estimand 2")
log("=" * 70)
res = pd.DataFrame(results_rows)
log(f"\n{'Spec':<42} {'N':>7} {'RF coef':>9} {'SE':>8} {'p-val':>7} {'Stars':>5}")
log("-" * 78)
for _, r in res.iterrows():
    log(f"  {r['spec']:<40} {r['n_obs']:>7,} {r['rf_coef']:>+9.5f} "
        f"{r['rf_se']:>8.5f} {r['rf_pval']:>7.4f} {r['rf_stars']:>5}")

log("\nInterpretation:")
log("  RF coef = share of K=6 neighbours with R1 lottery wins by which R3 application")
log("  probability rises — measuring the application margin of the peer effect.")
log("  Restricted to non-R1-winners so own R1 outcome does not contaminate.")

res.to_csv(TABLES_DIR / "estimand2_application.csv", index=False)
log(f"\n  Saved: {TABLES_DIR / 'estimand2_application.csv'}")

with open(LOGS_DIR / "estimand2_application.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"  Saved: {LOGS_DIR / 'estimand2_application.txt'}")
