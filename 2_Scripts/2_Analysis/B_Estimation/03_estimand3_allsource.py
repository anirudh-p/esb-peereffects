"""
03_estimand3_allsource.py
==========================
Estimands 3 and 4: Longer-horizon all-source ESB adoption

Estimand 3: Does R1 neighbour lottery luck predict all-source ESB adoption
            through 2023-24? (WRI outcome — broader than CSBP lottery)

Estimand 4: Heterogeneity by R1 deployment visibility. R1 buses delivered
            early (<2024) vs late/unknown. Tests whether demonstration effect
            (operational neighbours) drives the result more than lottery-win
            awareness alone. Nested within Estimand 3 — treat as mechanism check.

Design (Estimand 3):
  Outcomes   : wri_any_2023_24 (primary), wri_any_2024 (pure post-R1 window)
  Instrument : w{K}_IV_Z_R1 (share of R1 lottery winners among K neighbours)
  Loser ctrl : w{K}_IS_R1_LOSER
  Sample     : non-R1-winners in estimation set
  Estimator  : LPM, state-clustered SE

Design (Estimand 4, mechanism check):
  Same, but replace w_IV_Z_R1 with two instruments:
    w_r1_early_delivery — R1 winners with bus delivered before 2024
    w_r1_late_delivery  — R1 winners with bus delivered in 2024 or unknown
  Tests: H0: early == late  (chi-sq / Wald across coefficients)
  Note: 93.5% of R1 buses have delivery dates → reasonable coverage.

Outputs
-------
  3_Output/Tables/estimand3_allsource.csv
  3_Output/Logs/estimand3_allsource.txt
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

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
log("ESTIMAND 3 — All-source ESB adoption (2023-24 window)")
log("=" * 70)
log(f"  Full dataset: {len(df):,}")

# Sample: non-R1-winners, all controls
required = ["enrollment", "median_income", "poverty_rate", "pct_white",
            "pm25", "state", "pct_dem_2020", "w6_IV_Z_R1"]
d = df[df["IV_Z_R1"] == 0].dropna(subset=required).copy().reset_index(drop=True)
log(f"  Non-R1-winner estimation sample: {len(d):,}")
for oc in ["wri_any_2023", "wri_any_2024", "wri_any_2023_24"]:
    log(f"    {oc}: {d[oc].sum():,} adopters ({100*d[oc].mean():.2f}%)")


def get_controls(df_in, include_r1_loser=True, include_pre_r1=True,
                 include_state_fe=True, k=6):
    ctrl = pd.DataFrame(index=df_in.index)
    ctrl["log_enroll"]   = df_in["log_enroll"]
    ctrl["log_income"]   = df_in["log_income"]
    ctrl["poverty_rate"] = df_in["poverty_rate"]
    ctrl["pct_white"]    = df_in["pct_white"]
    ctrl["pm25"]         = df_in["pm25"]
    ctrl["priority_r1"]  = df_in["priority_r1"].fillna(0).astype(float)
    ctrl["priority_r23"] = df_in["priority_r23"].fillna(0).astype(float)
    ctrl["pct_dem_2020"] = df_in["pct_dem_2020"]
    if include_pre_r1:
        # Pre-R1 ESB adopters: could be early movers with different peer networks
        ctrl["pre_r1_adopter"] = df_in["is_pre_r1_adopter"].fillna(0).astype(float)
    if include_r1_loser:
        ctrl["w_r1_loser"] = df_in[f"w{k}_IS_R1_LOSER"].fillna(0)
    if include_state_fe:
        sdums = pd.get_dummies(df_in["state"], prefix="st", drop_first=True, dtype=float)
        ctrl = pd.concat([ctrl, sdums], axis=1)
    return ctrl


# ══════════════════════════════════════════════════════════════════════════════
# Estimand 3 — main loop
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "─" * 60)
log("3A — Estimand 3 specifications")
log("─" * 60)
results_rows = []
K_values = [6, 10, 15]

for outcome in ["wri_any_2023_24", "wri_any_2024"]:
    log(f"\n  Outcome: {outcome}")
    for K in K_values:
        wcol = f"w{K}_IV_Z_R1"
        if wcol not in d.columns:
            continue
        d_k = d.dropna(subset=[wcol]).copy()

        for loser_ctrl in [False, True]:
            label = f"Y={outcome}, K={K}, loser={'Y' if loser_ctrl else 'N'}"
            ctrl  = get_controls(d_k, include_r1_loser=loser_ctrl,
                                 include_pre_r1=True, include_state_fe=True, k=K)
            Y     = d_k[outcome].astype(float)
            W_iv  = d_k[wcol].fillna(0)
            X_rf  = sm.add_constant(
                pd.concat([W_iv.rename("w_IV_Z_R1"), ctrl], axis=1),
                has_constant="add").dropna(axis=1)
            ols   = sm.OLS(Y, X_rf).fit(cov_type="cluster",
                                         cov_kwds={"groups": d_k["state"]})
            coef  = ols.params.get("w_IV_Z_R1", np.nan)
            se    = ols.bse.get("w_IV_Z_R1",   np.nan)
            pval  = ols.pvalues.get("w_IV_Z_R1", np.nan)
            nobs  = int(ols.nobs)
            ymean = Y.mean()
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            log(f"    [{label:<40}]  coef={coef:+.5f}  SE={se:.5f}  "
                f"p={pval:.4f}{stars}  N={nobs:,}  Ymean={ymean:.4f}")
            results_rows.append({
                "estimand"   : "AllSource",
                "outcome"    : outcome,
                "K"          : K,
                "loser_ctrl" : loser_ctrl,
                "n_obs"      : nobs,
                "y_mean"     : round(ymean, 5),
                "rf_coef"    : round(coef, 6),
                "rf_se"      : round(se, 6),
                "rf_pval"    : round(pval, 4),
                "rf_stars"   : stars,
            })


# ══════════════════════════════════════════════════════════════════════════════
# Estimand 4 — Delivery heterogeneity (mechanism check)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "─" * 60)
log("3B — Estimand 4: Delivery-timing heterogeneity (K=6, loser=Yes)")
log("─" * 60)
log("  H₀: effect of early-delivery neighbours = effect of late-delivery neighbours")
log("  Instruments: w6_r1_early_delivery  vs  w6_r1_late_or_unknown")
log()

early_col = "w6_r1_early_delivery"
late_col  = "w6_r1_late_or_unknown"

if early_col not in d.columns or late_col not in d.columns:
    log(f"  WARNING: delivery lag columns not found ({early_col}, {late_col})")
else:
    d_k = d.dropna(subset=["w6_IV_Z_R1"]).fillna({early_col: 0, late_col: 0}).copy()
    for outcome in ["wri_any_2023_24", "wri_any_2024"]:
        log(f"\n  Outcome: {outcome}")
        ctrl = get_controls(d_k, include_r1_loser=True, include_state_fe=True, k=6)
        Y    = d_k[outcome].astype(float)
        W_e  = d_k[early_col].fillna(0)
        W_l  = d_k[late_col].fillna(0)
        X_dt = sm.add_constant(
            pd.concat([W_e.rename("w_early"), W_l.rename("w_late"), ctrl], axis=1),
            has_constant="add").dropna(axis=1)
        ols_dt = sm.OLS(Y, X_dt).fit(cov_type="cluster",
                                      cov_kwds={"groups": d_k["state"]})
        c_e  = ols_dt.params.get("w_early", np.nan)
        c_l  = ols_dt.params.get("w_late",  np.nan)
        se_e = ols_dt.bse.get("w_early",   np.nan)
        se_l = ols_dt.bse.get("w_late",    np.nan)
        p_e  = ols_dt.pvalues.get("w_early", np.nan)
        p_l  = ols_dt.pvalues.get("w_late", np.nan)
        nobs = int(ols_dt.nobs)
        stars_e = "***" if p_e < 0.01 else "**" if p_e < 0.05 else "*" if p_e < 0.1 else ""
        stars_l = "***" if p_l < 0.01 else "**" if p_l < 0.05 else "*" if p_l < 0.1 else ""
        log(f"    Early delivery neighbours: coef={c_e:+.5f}  SE={se_e:.5f}  "
            f"p={p_e:.4f}{stars_e}")
        log(f"    Late/unknown delivery    : coef={c_l:+.5f}  SE={se_l:.5f}  "
            f"p={p_l:.4f}{stars_l}")

        # Wald test: H₀ c_e == c_l
        # Using R = [1, -1] on (c_e, c_l)  →  R'βhat = c_e - c_l
        if not (np.isnan(c_e) or np.isnan(c_l)):
            # Get the subset of the covariance matrix
            idx_e = list(X_dt.columns).index("w_early")
            idx_l = list(X_dt.columns).index("w_late")
            vcov  = ols_dt.cov_params()
            R = np.zeros(len(ols_dt.params))
            R[idx_e] =  1.0
            R[idx_l] = -1.0
            diff     = c_e - c_l
            var_diff = R @ vcov.values @ R
            wald_stat = diff**2 / var_diff
            wald_p    = 1 - chi2.cdf(wald_stat, df=1)
            log(f"    Wald test (early vs late): Δ={diff:+.5f}  χ²(1)={wald_stat:.2f}  "
                f"p={wald_p:.4f}")
        else:
            log("    Wald test unavailable (NaN coefficients)")

        results_rows.append({
            "estimand"   : "Delivery_split_early",
            "outcome"    : outcome,
            "K": 6, "loser_ctrl": True,
            "n_obs": nobs, "y_mean": round(Y.mean(), 5),
            "rf_coef": round(c_e, 6), "rf_se": round(se_e, 6),
            "rf_pval": round(p_e, 4), "rf_stars": stars_e,
        })
        results_rows.append({
            "estimand"   : "Delivery_split_late",
            "outcome"    : outcome,
            "K": 6, "loser_ctrl": True,
            "n_obs": nobs, "y_mean": round(Y.mean(), 5),
            "rf_coef": round(c_l, 6), "rf_se": round(se_l, 6),
            "rf_pval": round(p_l, 4), "rf_stars": stars_l,
        })


# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
res = pd.DataFrame(results_rows)
res.to_csv(TABLES_DIR / "estimand3_allsource.csv", index=False)
log(f"\n  Saved: {TABLES_DIR / 'estimand3_allsource.csv'}")

with open(LOGS_DIR / "estimand3_allsource.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"  Saved: {LOGS_DIR / 'estimand3_allsource.txt'}")
