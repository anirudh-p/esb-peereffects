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
    ctrl["priority_r23"] = df_in["priority_r23"].fillna(0).astype(float)
    ctrl["pct_dem_2020"]  = df_in["pct_dem_2020"]
    if include_r1_loser:
        ctrl["w_r1_loser"] = df_in[f"w{k}_IS_R1_LOSER"].fillna(0)
    if include_state_fe:
        # priority_r1 × state interaction FE (R1 lottery strata)
        pri = df_in["priority_r1"].fillna(0).astype(int).astype(str)
        ps_dums = pd.get_dummies(
            pri + "_" + df_in["state"].astype(str),
            prefix="ps", drop_first=True, dtype=float
        )
        ctrl = pd.concat([ctrl, ps_dums], axis=1)
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
# R2 Grant Neighbour Analysis
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("R2 GRANT NEIGHBOUR ANALYSIS")
log("=" * 70)
log("  R2 grantees are ~2023 adopters whose grants were awarded BEFORE R3 opened.")
log("  Districts that observe a R2-grantee neighbour may be more encouraged to")
log("  apply to R3 independently of R1 lottery variation.  Two robustness checks:")

# ── (a) Heterogeneity split by R2-neighbour presence ──────────────────────
log("\n── (a) Heterogeneity split: R2-neighbour present vs absent (K=6, loser=Yes) ──")
r2_col = "w6_IS_R2_GRANTEE"
if r2_col in d.columns:
    d_r2_yes = d[d[r2_col] > 0].dropna(subset=["w6_IV_Z_R1"]).copy()
    d_r2_no  = d[d[r2_col] == 0].dropna(subset=["w6_IV_Z_R1"]).copy()
    log(f"  R2-neighbour present: N={len(d_r2_yes):,}  "
        f"R3-apply rate={100*d_r2_yes['Y_R3_apply'].mean():.2f}%")
    log(f"  R2-neighbour absent : N={len(d_r2_no):,}  "
        f"R3-apply rate={100*d_r2_no['Y_R3_apply'].mean():.2f}%")
    for label_a, d_sub in [("R2-nbr present", d_r2_yes), ("R2-nbr absent ", d_r2_no)]:
        ctrl_a = get_controls(d_sub, include_r1_loser=True, include_state_fe=True, k=6)
        Y_a    = d_sub["Y_R3_apply"].astype(float)
        W_a    = d_sub["w6_IV_Z_R1"].fillna(0)
        X_a    = sm.add_constant(pd.concat([W_a.rename("w_IV_Z_R1"), ctrl_a], axis=1),
                                 has_constant="add").dropna(axis=1)
        ols_a  = sm.OLS(Y_a, X_a).fit(cov_type="cluster",
                                       cov_kwds={"groups": d_sub["state"]})
        c_a    = ols_a.params.get("w_IV_Z_R1", np.nan)
        s_a    = ols_a.bse.get("w_IV_Z_R1", np.nan)
        p_a    = ols_a.pvalues.get("w_IV_Z_R1", np.nan)
        st_a   = "***" if p_a < 0.01 else "**" if p_a < 0.05 else "*" if p_a < 0.1 else ""
        log(f"    {label_a}: coef={c_a:+.5f}  SE={s_a:.5f}  p={p_a:.4f}{st_a}  N={len(d_sub):,}")
        results_rows.append({
            "estimand": f"R2_split_{label_a.strip()}", "K": 6, "loser_ctrl": True,
            "spec": f"R2 hetero ({label_a})", "n_obs": len(d_sub),
            "y_mean": round(Y_a.mean(), 5), "rf_coef": round(c_a, 6),
            "rf_se": round(s_a, 6), "rf_pval": round(p_a, 4), "rf_stars": st_a,
        })
else:
    log(f"  SKIP (a): column '{r2_col}' not found in dataset.")

# ── (b) Additive R2-neighbour control ─────────────────────────────────────
log("\n── (b) Additive R2-neighbour control — does R1 coefficient change? (K=6, loser=Yes) ──")
if r2_col in d.columns:
    d_b = d.dropna(subset=["w6_IV_Z_R1"]).copy()
    # Without R2 control (baseline already in results_rows, re-run for direct comparison)
    ctrl_b0  = get_controls(d_b, include_r1_loser=True, include_state_fe=True, k=6)
    # With R2 control
    ctrl_b1  = ctrl_b0.copy()
    ctrl_b1["w_r2_grantee"] = d_b[r2_col].fillna(0).values
    Y_b  = d_b["Y_R3_apply"].astype(float)
    W_b  = d_b["w6_IV_Z_R1"].fillna(0)
    for tag, ctrl_use in [("without R2 ctrl", ctrl_b0), ("with R2 ctrl   ", ctrl_b1)]:
        X_b   = sm.add_constant(pd.concat([W_b.rename("w_IV_Z_R1"), ctrl_use], axis=1),
                                has_constant="add").dropna(axis=1)
        ols_b = sm.OLS(Y_b, X_b).fit(cov_type="cluster",
                                      cov_kwds={"groups": d_b["state"]})
        c_b   = ols_b.params.get("w_IV_Z_R1", np.nan)
        s_b   = ols_b.bse.get("w_IV_Z_R1", np.nan)
        p_b   = ols_b.pvalues.get("w_IV_Z_R1", np.nan)
        st_b  = "***" if p_b < 0.01 else "**" if p_b < 0.05 else "*" if p_b < 0.1 else ""
        log(f"    K=6 loser=Yes {tag}: coef={c_b:+.5f}  SE={s_b:.5f}  p={p_b:.4f}{st_b}")
        results_rows.append({
            "estimand": f"R2_additive_{tag.strip()}", "K": 6, "loser_ctrl": True,
            "spec": f"R2 additive ({tag})", "n_obs": int(ols_b.nobs),
            "y_mean": round(Y_b.mean(), 5), "rf_coef": round(c_b, 6),
            "rf_se": round(s_b, 6), "rf_pval": round(p_b, 4), "rf_stars": st_b,
        })
    log("  If R1 coef stable after adding R2 control → R1 peer effect is not proxying")
    log("  for local propensity to adopt driven by nearby R2 grants.")
else:
    log(f"  SKIP (b): column '{r2_col}' not found in dataset.")


# ══════════════════════════════════════════════════════════════════════════════
# Applicant-Exposed Sample  (Difference Design / Discouraged-Loser Test)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("APPLICANT-EXPOSED SAMPLE  (K=6, loser=Yes)")
log("=" * 70)
log("  Restrict to districts that had at least one R1-applicant neighbour,")
log("  i.e. w6_IV_Z_R1 > 0  OR  w6_IS_R1_LOSER > 0.")
log("  This sharpens the comparison to winner-neighbour vs loser-neighbour")
log("  districts, removing the 'never-exposed' remote districts from the")
log("  control group.  The coefficient on w_IS_R1_LOSER tests discouragement:")
log("  a negative sign means observing a lottery loser next door reduces own")
log("  R3 application propensity conditioning on winner share.")

app_mask = (d["w6_IV_Z_R1"].fillna(0) > 0) | (d["w6_IS_R1_LOSER"].fillna(0) > 0)
d_app = d[app_mask].dropna(subset=["w6_IV_Z_R1"]).copy()
log(f"\n  Applicant-exposed N={len(d_app):,}  "
    f"(vs full sample N={len(d.dropna(subset=['w6_IV_Z_R1'])):,})")
log(f"  R3-apply rate in exposed sample: {100*d_app['Y_R3_apply'].mean():.2f}%")

ctrl_app = get_controls(d_app, include_r1_loser=True, include_state_fe=True, k=6)
Y_app    = d_app["Y_R3_apply"].astype(float)
W_app    = d_app["w6_IV_Z_R1"].fillna(0)
X_app    = sm.add_constant(
    pd.concat([W_app.rename("w_IV_Z_R1"), ctrl_app], axis=1),
    has_constant="add"
).dropna(axis=1)
ols_app  = sm.OLS(Y_app, X_app).fit(cov_type="cluster",
                                     cov_kwds={"groups": d_app["state"]})

c_app  = ols_app.params.get("w_IV_Z_R1",  np.nan)
s_app  = ols_app.bse.get("w_IV_Z_R1",    np.nan)
p_app  = ols_app.pvalues.get("w_IV_Z_R1", np.nan)
c_los  = ols_app.params.get("w_r1_loser",  np.nan)
s_los  = ols_app.bse.get("w_r1_loser",    np.nan)
p_los  = ols_app.pvalues.get("w_r1_loser", np.nan)
st_app = "***" if p_app < 0.01 else "**" if p_app < 0.05 else "*" if p_app < 0.1 else ""
st_los = "***" if p_los < 0.01 else "**" if p_los < 0.05 else "*" if p_los < 0.1 else ""

log(f"\n  w_IV_Z_R1  (winner share):  coef={c_app:+.5f}  SE={s_app:.5f}  "
    f"p={p_app:.4f}{st_app}  N={int(ols_app.nobs):,}")
log(f"  w_r1_loser (loser share):   coef={c_los:+.5f}  SE={s_los:.5f}  "
    f"p={p_los:.4f}{st_los}")
log("  Negative w_r1_loser coef would support a discouragement mechanism.")

results_rows.append({
    "estimand": "App_Exposed_winner", "K": 6, "loser_ctrl": True,
    "spec": "Applicant-exposed sample (winner coef)", "n_obs": int(ols_app.nobs),
    "y_mean": round(Y_app.mean(), 5), "rf_coef": round(c_app, 6),
    "rf_se": round(s_app, 6), "rf_pval": round(p_app, 4), "rf_stars": st_app,
})
results_rows.append({
    "estimand": "App_Exposed_loser", "K": 6, "loser_ctrl": True,
    "spec": "Applicant-exposed sample (loser coef)", "n_obs": int(ols_app.nobs),
    "y_mean": round(Y_app.mean(), 5), "rf_coef": round(c_los, 6),
    "rf_se": round(s_los, 6), "rf_pval": round(p_los, 4), "rf_stars": st_los,
})


# ══════════════════════════════════════════════════════════════════════════════
# Distance-Decay Robustness  (K=6, loser=Yes)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("DISTANCE-DECAY ROBUSTNESS  (K=6, loser=Yes)")
log("=" * 70)
log("  Replace uniform 1/K weights with inverse-distance normalised weights")
log("  (wd6_IV_Z_R1 = Σ (1/d_ij / Σ 1/d_ij) × Z_j for K=6 neighbours).")
log("  If the peer effect is driven by proximity rather than a flat network,")
log("  the distance-decay coefficient should be larger in magnitude.")

wd_col = "wd6_IV_Z_R1"
if wd_col in d.columns:
    d_wd = d.dropna(subset=[wd_col]).copy()
    for tag, wcol_use in [("Uniform 1/K (w6) ", "w6_IV_Z_R1"),
                           ("Inv-dist  (wd6)", wd_col)]:
        ctrl_wd = get_controls(d_wd, include_r1_loser=True, include_state_fe=True, k=6)
        # For distance-decay: also swap the loser column if available
        if tag.startswith("Inv") and "wd6_IS_R1_LOSER" in d_wd.columns:
            ctrl_wd["w_r1_loser"] = d_wd["wd6_IS_R1_LOSER"].fillna(0).values
        Y_wd  = d_wd["Y_R3_apply"].astype(float)
        W_wd  = d_wd[wcol_use].fillna(0)
        X_wd  = sm.add_constant(
            pd.concat([W_wd.rename("w_IV_Z_R1"), ctrl_wd], axis=1),
            has_constant="add"
        ).dropna(axis=1)
        ols_wd = sm.OLS(Y_wd, X_wd).fit(cov_type="cluster",
                                          cov_kwds={"groups": d_wd["state"]})
        c_wd   = ols_wd.params.get("w_IV_Z_R1", np.nan)
        s_wd   = ols_wd.bse.get("w_IV_Z_R1",   np.nan)
        p_wd   = ols_wd.pvalues.get("w_IV_Z_R1", np.nan)
        st_wd  = "***" if p_wd < 0.01 else "**" if p_wd < 0.05 else "*" if p_wd < 0.1 else ""
        log(f"    {tag}: coef={c_wd:+.5f}  SE={s_wd:.5f}  p={p_wd:.4f}{st_wd}  N={int(ols_wd.nobs):,}")
        results_rows.append({
            "estimand": f"DistDecay_{tag.strip()}", "K": 6, "loser_ctrl": True,
            "spec": f"Distance-decay ({tag})", "n_obs": int(ols_wd.nobs),
            "y_mean": round(Y_wd.mean(), 5), "rf_coef": round(c_wd, 6),
            "rf_se": round(s_wd, 6), "rf_pval": round(p_wd, 4), "rf_stars": st_wd,
        })
else:
    log(f"  SKIP: column '{wd_col}' not found — re-run 02_build_spatial_weights.py first.")


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
