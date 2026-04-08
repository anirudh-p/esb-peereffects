"""
05_main_iv.py
=============
Post-Meeting branch: single clean IV analysis.

Research question
-----------------
Does ESB adoption by neighbors affect own adoption probability?

Design
------
  Y_i = alpha + beta * (neighbor ESBs) + gamma'X_i + epsilon_i

  Instrument  Z : w{K}_IV_Z_R1   — share of K neighbors who won R1 lottery
  Loser ctrl  L : w{K}_IS_R1_LOSER — share who applied but lost R1
  Primary  outcome : wri_post_r1_cum  — any WRI-tracked ESB awarded by end
                     of 2024, excluding pre-R1 adopters (cumulative post-R1)
  Secondary outcome: Y_R3_apply       — applied to R3 (retained for comparison)
  Strata FE        : priority x state x fuel_group interaction dummies
  SE               : state-clustered

Sections
--------
  1. Core reduced-form table  (K=6/10/15, loser on/off, full + priority)
  2. Outcome comparison       (cumulative vs. narrow windows side-by-side)
  3. Urbanicity heterogeneity (Urban / Suburban / Town / Rural)
  4. Timing/visibility split  (pre-R3 vs post-R3 delivery on cum. outcome)
  5. K sensitivity            (K=6/10/15 for cum. outcome)
  6. Placebo/falsification    (R3 instrument -> pre-program outcomes)
  7. Applicant-exposed robustness

Outputs
-------
  3_Output/Logs/main_iv.txt
  3_Output/Tables/main_iv.csv
"""

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SPATIAL_DATASET, LOGS_DIR, TABLES_DIR, ensure_dirs

warnings.filterwarnings("ignore")
ensure_dirs()

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

results_rows = []

# =============================================================================
# Load data
# =============================================================================
df = pd.read_csv(SPATIAL_DATASET, low_memory=False, dtype={"nces_id": str})
df["log_enroll"] = np.log(df["enrollment"].clip(lower=1))
df["log_income"]  = np.log(df["median_income"].clip(lower=1))

required = ["enrollment", "median_income", "poverty_rate", "pct_white",
            "pm25", "state", "pct_dem_2020", "w6_IV_Z_R1"]

d = df[df["IV_Z_R1"] == 0].dropna(subset=required).copy().reset_index(drop=True)
d_pri = d[d["priority_r1"] == 1].copy()

log("=" * 70)
log("POST-MEETING BRANCH — Main IV Analysis")
log("=" * 70)
log(f"  Full estimation sample (non-R1 winners): {len(d):,}")
log(f"  Priority subsample                     : {len(d_pri):,}")
log(f"\n  Outcome rates (full sample):")
for col in ["Y_R3_apply", "wri_any_2023_24", "wri_post_r1_cum", "wri_any_2024"]:
    if col in d.columns:
        log(f"    {col:<25}: {d[col].sum():5,}  ({100*d[col].mean():.2f}%)")
log(f"\n  Outcome rates (priority subsample):")
for col in ["Y_R3_apply", "wri_any_2023_24", "wri_post_r1_cum"]:
    if col in d_pri.columns:
        log(f"    {col:<25}: {d_pri[col].sum():5,}  ({100*d_pri[col].mean():.2f}%)")


# =============================================================================
# Helpers
# =============================================================================
def get_ctrl(df_in, k=6, loser=True, extra_cols=None):
    pri  = df_in["priority_r1"].fillna(0).astype(int).astype(str)
    fuel = df_in["r1_fuel_group"].fillna("nonapp").astype(str)
    ps   = pd.get_dummies(
        pri + "_" + df_in["state"].astype(str) + "_" + fuel,
        prefix="psf", drop_first=True, dtype=float
    )
    ctrl = pd.DataFrame({
        "log_enroll"       : df_in["log_enroll"],
        "log_income"       : df_in["log_income"],
        "poverty_rate"     : df_in["poverty_rate"],
        "pct_white"        : df_in["pct_white"],
        "pm25"             : df_in["pm25"],
        "pct_dem_2020"     : df_in["pct_dem_2020"],
        "priority_r23"     : df_in["priority_r23"].fillna(0).astype(float),
        "is_pre_r1_adopter": df_in["is_pre_r1_adopter"].fillna(0).astype(float),
    }, index=df_in.index)
    if loser:
        ctrl[f"w_loser"] = df_in[f"w{k}_IS_R1_LOSER"].fillna(0)
    if extra_cols:
        for c, col in extra_cols.items():
            ctrl[c] = df_in[col].fillna(0)
    return pd.concat([ctrl, ps], axis=1)


def fit_ols(Y, W, ctrl, clusters, wcol_name="treat"):
    X = sm.add_constant(
        pd.concat([W.rename(wcol_name), ctrl], axis=1),
        has_constant="add"
    ).dropna(axis=1)
    res = sm.OLS(Y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": clusters},
        use_t=True,
    )
    c  = res.params.get(wcol_name, np.nan)
    s  = res.bse.get(wcol_name, np.nan)
    p  = res.pvalues.get(wcol_name, np.nan)
    n  = int(res.nobs)
    st = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    return c, s, p, n, st, res


def report(label, c, s, p, n, st, ymean, section=""):
    log(f"  {label:<55} coef={c:+.5f}  SE={s:.5f}  p={p:.4f}{st:<3}  "
        f"N={n:,}  Ymean={ymean:.4f}")
    results_rows.append({
        "section": section, "label": label,
        "coef": round(c, 6), "se": round(s, 6),
        "p": round(p, 4), "stars": st, "n": n, "ymean": round(ymean, 4),
    })


# =============================================================================
# SECTION 1 — Core reduced-form table
# =============================================================================
log("\n" + "=" * 70)
log("SECTION 1 — Core reduced-form: wri_post_r1_cum + Y_R3_apply")
log("Primary outcome: wri_post_r1_cum | Secondary: Y_R3_apply")
log("=" * 70)

for outcome, outlabel in [
    ("wri_post_r1_cum", "CUM"),
    ("Y_R3_apply",      "APP"),
]:
    log(f"\n  [{outlabel}] Outcome = {outcome}")
    for K in [6, 10, 15]:
        wcol = f"w{K}_IV_Z_R1"
        if wcol not in d.columns:
            continue
        d_k = d.dropna(subset=[wcol]).copy()
        for loser in [False, True]:
            ctrl = get_ctrl(d_k, k=K, loser=loser)
            Y    = d_k[outcome].astype(float)
            W    = d_k[wcol].fillna(0)
            c, s, p, n, st, _ = fit_ols(Y, W, ctrl, d_k["state"])
            label = f"  K={K}, loser={'Yes' if loser else 'No ':3}, {outlabel}"
            report(label, c, s, p, n, st, float(Y.mean()), "S1_full")

    # Priority subsample K=6
    log(f"\n  [{outlabel}] Priority subsample, K=6")
    d_pk = d_pri.dropna(subset=["w6_IV_Z_R1"]).copy()
    for loser in [False, True]:
        ctrl = get_ctrl(d_pk, k=6, loser=loser)
        Y    = d_pk[outcome].astype(float)
        W    = d_pk["w6_IV_Z_R1"].fillna(0)
        c, s, p, n, st, _ = fit_ols(Y, W, ctrl, d_pk["state"])
        label = f"  Priority K=6, loser={'Yes' if loser else 'No ':3}, {outlabel}"
        report(label, c, s, p, n, st, float(Y.mean()), "S1_priority")


# =============================================================================
# SECTION 2 — Outcome comparison table
# =============================================================================
log("\n" + "=" * 70)
log("SECTION 2 — Outcome comparison (K=6, loser=Yes)")
log("=" * 70)

outcomes = [
    ("Y_R3_apply",      "R3 application       [prior primary]"),
    ("wri_any_2023_24", "WRI any 2023-24       [prior null]"),
    ("wri_any_2024",    "WRI any 2024 only     [prior null]"),
    ("wri_post_r1_cum", "WRI cum post-R1       [new primary]"),
]

for sample_label, d_s in [("Full sample", d), ("Priority subsample", d_pri)]:
    log(f"\n  {sample_label}")
    d_s_k = d_s.dropna(subset=["w6_IV_Z_R1"]).copy()
    ctrl = get_ctrl(d_s_k, k=6, loser=True)
    for outcome, desc in outcomes:
        if outcome not in d_s_k.columns:
            continue
        Y = d_s_k[outcome].astype(float)
        W = d_s_k["w6_IV_Z_R1"].fillna(0)
        c, s, p, n, st, _ = fit_ols(Y, W, ctrl, d_s_k["state"])
        label = f"  {sample_label[:4]}, {desc}"
        report(label, c, s, p, n, st, float(Y.mean()), "S2_comparison")


# =============================================================================
# SECTION 3 — Urbanicity heterogeneity
# =============================================================================
log("\n" + "=" * 70)
log("SECTION 3 — Urbanicity heterogeneity (K=6, loser=Yes)")
log("=" * 70)

for urb in ["Urban", "Suburban", "Town", "Rural"]:
    d_u = d[d["urbanicity"] == urb].dropna(subset=["w6_IV_Z_R1"]).copy()
    if len(d_u) < 100:
        continue
    log(f"\n  [{urb}]  N={len(d_u):,}  "
        f"cum_rate={100*d_u['wri_post_r1_cum'].mean():.2f}%  "
        f"apply_rate={100*d_u['Y_R3_apply'].mean():.2f}%")
    ctrl = get_ctrl(d_u, k=6, loser=True)
    for outcome in ["wri_post_r1_cum", "Y_R3_apply"]:
        Y = d_u[outcome].astype(float)
        W = d_u["w6_IV_Z_R1"].fillna(0)
        c, s, p, n, st, _ = fit_ols(Y, W, ctrl, d_u["state"])
        label = f"  {urb:<10} {outcome}"
        report(label, c, s, p, n, st, float(Y.mean()), "S3_urbanicity")


# =============================================================================
# SECTION 4 — Timing / visibility split on wri_post_r1_cum
# =============================================================================
log("\n" + "=" * 70)
log("SECTION 4 — Timing / visibility split (wri_post_r1_cum, K=6, loser=Yes)")
log("Pre-R3 deadline = delivered or operating by 2023 Q3")
log("Post-R3 deadline = delivered 2023 Q4 or later")
log("=" * 70)

pre_col  = "w6_r1_pre_r3_delivery"
post_col = "w6_r1_post_r3_delivery"
unk_col  = "w6_r1_delivery_unknown"

for sample_label, d_s in [("Full sample", d), ("Priority subsample", d_pri)]:
    d_s_k = d_s.dropna(subset=["w6_IV_Z_R1", pre_col, post_col]).copy()
    log(f"\n  {sample_label}  N={len(d_s_k):,}")

    for outcome in ["wri_post_r1_cum", "Y_R3_apply"]:
        log(f"\n    Outcome: {outcome}")
        ctrl = get_ctrl(d_s_k, k=6, loser=True)

        # Split model: separate pre/post coefficients
        W_pre  = d_s_k[pre_col].fillna(0)
        W_post = d_s_k[post_col].fillna(0)
        W_unk  = d_s_k[unk_col].fillna(0)
        Y = d_s_k[outcome].astype(float)

        X_split = sm.add_constant(
            pd.concat([
                W_pre.rename("pre_r3"),
                W_post.rename("post_r3"),
                W_unk.rename("del_unknown"),
                ctrl,
            ], axis=1),
            has_constant="add"
        ).dropna(axis=1)
        res = sm.OLS(Y, X_split).fit(
            cov_type="cluster",
            cov_kwds={"groups": d_s_k["state"]},
            use_t=True,
        )
        for cname, tag in [("pre_r3", "pre_r3 (visible) "),
                            ("post_r3", "post_r3 (invisible)"),
                            ("del_unknown", "unknown timing  ")]:
            if cname not in res.params:
                continue
            cv = res.params[cname]
            sv = res.bse[cname]
            pv = res.pvalues[cname]
            st = "***" if pv<0.01 else "**" if pv<0.05 else "*" if pv<0.1 else ""
            log(f"      {tag}: coef={cv:+.5f}  SE={sv:.5f}  p={pv:.4f}{st}")
            results_rows.append({
                "section": "S4_timing", "label": f"{sample_label}|{outcome}|{tag.strip()}",
                "coef": round(cv,6), "se": round(sv,6),
                "p": round(pv,4), "stars": st, "n": int(res.nobs),
                "ymean": round(float(Y.mean()),4),
            })

        # Wald test: pre == post
        if "pre_r3" in res.params.index and "post_r3" in res.params.index:
            idx_pre  = list(res.params.index).index("pre_r3")
            idx_post = list(res.params.index).index("post_r3")
            R = np.zeros((1, len(res.params)))
            R[0, idx_pre]  =  1
            R[0, idx_post] = -1
            wald = res.wald_test(R, use_f=False)
            wald_p = float(np.squeeze(wald.pvalue))
            delta  = float(res.params["pre_r3"] - res.params["post_r3"])
            log(f"      Wald (pre=post): delta={delta:+.5f}  chi2(1)p={wald_p:.4f}")


# =============================================================================
# SECTION 5 — K sensitivity for wri_post_r1_cum
# =============================================================================
log("\n" + "=" * 70)
log("SECTION 5 — K sensitivity (wri_post_r1_cum, loser=Yes)")
log("=" * 70)

for sample_label, d_s in [("Full sample", d), ("Priority subsample", d_pri)]:
    log(f"\n  {sample_label}")
    for K in [6, 10, 15]:
        wcol = f"w{K}_IV_Z_R1"
        if wcol not in d_s.columns:
            continue
        d_k = d_s.dropna(subset=[wcol]).copy()
        ctrl = get_ctrl(d_k, k=K, loser=True)
        Y = d_k["wri_post_r1_cum"].astype(float)
        W = d_k[wcol].fillna(0)
        c, s, p, n, st, _ = fit_ols(Y, W, ctrl, d_k["state"])
        label = f"  {sample_label[:4]}, K={K}, loser=Yes, wri_post_r1_cum"
        report(label, c, s, p, n, st, float(Y.mean()), "S5_K_sensitivity")

    # Inverse-distance for K=6
    wd_col = "wd6_IV_Z_R1"
    if wd_col in d_s.columns:
        d_k = d_s.dropna(subset=[wd_col]).copy()
        ctrl = get_ctrl(d_k, k=6, loser=True)
        if "w_loser" in ctrl.columns:
            ctrl["w_loser"] = d_k["wd6_IS_R1_LOSER"].fillna(0).values
        Y = d_k["wri_post_r1_cum"].astype(float)
        W = d_k[wd_col].fillna(0)
        c, s, p, n, st, _ = fit_ols(Y, W, ctrl, d_k["state"])
        label = f"  {sample_label[:4]}, K=6 inv-dist, loser=Yes, wri_post_r1_cum"
        report(label, c, s, p, n, st, float(Y.mean()), "S5_K_sensitivity")


# =============================================================================
# SECTION 6 — Placebo / falsification
# =============================================================================
log("\n" + "=" * 70)
log("SECTION 6 — Placebo / Falsification")
log("R3 instrument (w6_IV_Z_R3) -> pre-program outcomes (should be null)")
log("R1 instrument -> pre-R1 adoption (should be null by construction)")
log("=" * 70)

d_placebo = df[df["IV_Z_R1"] == 0].dropna(subset=required).copy().reset_index(drop=True)

# Placebo 1: R3 lottery wins predicting pre-R1 adoption
# R3 is 2023; pre-R1 adoption happened before 2022 — no causal link possible
if "w6_IV_Z_R3" in d_placebo.columns:
    log("\n  Placebo 1: Z = w6_IV_Z_R3, Y = is_pre_r1_adopter (should be null)")
    d_pl = d_placebo.dropna(subset=["w6_IV_Z_R3"]).copy()
    ctrl = get_ctrl(d_pl, k=6, loser=False)
    Y = d_pl["is_pre_r1_adopter"].astype(float)
    W = d_pl["w6_IV_Z_R3"].fillna(0)
    c, s, p, n, st, _ = fit_ols(Y, W, ctrl, d_pl["state"])
    label = "  Placebo: Z=R3_wins, Y=pre_r1_adoption"
    report(label, c, s, p, n, st, float(Y.mean()), "S6_placebo")
else:
    log("  SKIP: w6_IV_Z_R3 not found in spatial dataset")

# Placebo 2: R1 wins predicting pre-R1 adoption (own pre-R1 already excluded
# from wri_post_r1_cum, but neighbor R1 wins should not predict own pre-R1)
log("\n  Placebo 2: Z = w6_IV_Z_R1, Y = is_pre_r1_adopter (should be null)")
d_pl2 = d.dropna(subset=["w6_IV_Z_R1"]).copy()
ctrl2 = get_ctrl(d_pl2, k=6, loser=True)
Y2 = d_pl2["is_pre_r1_adopter"].astype(float)
W2 = d_pl2["w6_IV_Z_R1"].fillna(0)
c2, s2, p2, n2, st2, _ = fit_ols(Y2, W2, ctrl2, d_pl2["state"])
label2 = "  Placebo: Z=R1_wins, Y=pre_r1_adoption"
report(label2, c2, s2, p2, n2, st2, float(Y2.mean()), "S6_placebo")

# Placebo 3: R3 instrument predicting R3 application but in pre-treatment sample
# Use w6_IV_Z_R3 -> Y_R3_apply in full sample (including R1 winners) to test
# whether R3 lottery own-win is spuriously detected
if "w6_IV_Z_R3" in df.columns:
    log("\n  Placebo 3: Z = w6_IV_Z_R1, Y = wri_any_2023 (2023-only, narrow)")
    log("  (If instrument is valid, R1 wins should not strongly predict same-year adoption)")
    d_pl3 = d.dropna(subset=["w6_IV_Z_R1"]).copy()
    ctrl3 = get_ctrl(d_pl3, k=6, loser=True)
    Y3 = d_pl3["wri_any_2023"].astype(float)
    W3 = d_pl3["w6_IV_Z_R1"].fillna(0)
    c3, s3, p3, n3, st3, _ = fit_ols(Y3, W3, ctrl3, d_pl3["state"])
    label3 = "  Timing check: Z=R1_wins, Y=wri_any_2023 (before most deliveries)"
    report(label3, c3, s3, p3, n3, st3, float(Y3.mean()), "S6_placebo")


# =============================================================================
# SECTION 7 — Applicant-exposed robustness
# =============================================================================
log("\n" + "=" * 70)
log("SECTION 7 — Applicant-exposed subsample robustness (K=6, loser=Yes)")
log("Restrict to districts with at least one R1 applicant neighbor")
log("=" * 70)

app_mask = (d["w6_IV_Z_R1"].fillna(0) > 0) | (d["w6_IS_R1_LOSER"].fillna(0) > 0)
d_app = d[app_mask].dropna(subset=["w6_IV_Z_R1"]).copy()
log(f"\n  Applicant-exposed N={len(d_app):,}  "
    f"(full E2 sample: {len(d):,})")
log(f"  cum_rate={100*d_app['wri_post_r1_cum'].mean():.2f}%  "
    f"apply_rate={100*d_app['Y_R3_apply'].mean():.2f}%")

ctrl_app = get_ctrl(d_app, k=6, loser=True)
for outcome in ["wri_post_r1_cum", "Y_R3_apply"]:
    Y = d_app[outcome].astype(float)
    W = d_app["w6_IV_Z_R1"].fillna(0)
    c, s, p, n, st, _ = fit_ols(Y, W, ctrl_app, d_app["state"])
    label = f"  Applicant-exposed: {outcome}"
    report(label, c, s, p, n, st, float(Y.mean()), "S7_appexposed")


# =============================================================================
# Save
# =============================================================================
log("\n" + "=" * 70)
log("SUMMARY — Main IV results")
log("=" * 70)
res_df = pd.DataFrame(results_rows)
log(f"\n{'Section':<15} {'Label':<55} {'N':>7} {'Coef':>9} {'SE':>8} {'p':>7} {'':>3}")
log("-" * 105)
for _, r in res_df.iterrows():
    log(f"  {r['section']:<13} {r['label']:<55} {r['n']:>7,} "
        f"{r['coef']:>+9.5f} {r['se']:>8.5f} {r['p']:>7.4f} {r['stars']:>3}")

res_df.to_csv(TABLES_DIR / "main_iv.csv", index=False)
log(f"\n  Saved: {TABLES_DIR / 'main_iv.csv'}")

with open(LOGS_DIR / "main_iv.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"  Saved: {LOGS_DIR / 'main_iv.txt'}")
