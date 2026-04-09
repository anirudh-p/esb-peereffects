"""
06_extended_iv.py
=================
Post-Meeting branch: extended IV analysis covering checklist items 3-7
plus additional outcome margins.

Sections
--------
  1.  Additional outcome margins
        Y_R2_apply    : applied to R2 competitive grant
        Y_any_post_r1 : applied to R2 OR R3 (broadest CSBP entry)
  2.  Conley spatial HAC  (K=6, primary outcomes)
  3.  Donut test  (K=1-2 dropped, remaining K=3-6)
  4.  Vendor network channel
  5.  Priority decomposition  (PM2.5 quartile x winning neighbor)
  6.  Climatic peers  (PRISM temperature quintile peer groups)

Outputs
-------
  3_Output/Logs/extended_iv.txt
  3_Output/Tables/extended_iv.csv
"""

import sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import geopandas as gpd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import LOGS_DIR, TABLES_DIR, CLEAN, SHP_FILE, ensure_dirs

warnings.filterwarnings("ignore")
ensure_dirs()

SUPP = CLEAN / "analysis_dataset_supp.csv"

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

results_rows = []

# =============================================================================
# Load data
# =============================================================================
df = pd.read_csv(SUPP, low_memory=False, dtype={"nces_id": str})
df["log_enroll"] = np.log(df["enrollment"].clip(lower=1))
df["log_income"]  = np.log(df["median_income"].clip(lower=1))

required = ["enrollment","median_income","poverty_rate","pct_white",
            "pm25","state","pct_dem_2020","w6_IV_Z_R1"]

d = df[df["IV_Z_R1"] == 0].dropna(subset=required).copy().reset_index(drop=True)
d_pri = d[d["priority_r1"] == 1].copy()

log("=" * 70)
log("POST-MEETING — Extended IV Analysis")
log("=" * 70)
log(f"  Full estimation sample : {len(d):,}")
log(f"  Priority subsample     : {len(d_pri):,}")


# =============================================================================
# Helpers
# =============================================================================
def get_ctrl(df_in, k=6, loser=True, extra=None):
    pri  = df_in["priority_r1"].fillna(0).astype(int).astype(str)
    fuel = df_in["r1_fuel_group"].fillna("nonapp").astype(str)
    ps   = pd.get_dummies(pri+"_"+df_in["state"].astype(str)+"_"+fuel,
                          prefix="psf", drop_first=True, dtype=float)
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
        ctrl["w_loser"] = df_in[f"w{k}_IS_R1_LOSER"].fillna(0)
    if extra:
        for cname, col in extra.items():
            ctrl[cname] = df_in[col].fillna(0) if col in df_in.columns else 0
    return pd.concat([ctrl, ps], axis=1)


def fit_ols(Y, W, ctrl, clusters, wname="treat"):
    X = sm.add_constant(pd.concat([W.rename(wname), ctrl], axis=1),
                        has_constant="add").dropna(axis=1)
    res = sm.OLS(Y, X).fit(cov_type="cluster",
                           cov_kwds={"groups": clusters}, use_t=True)
    c  = res.params.get(wname, np.nan)
    s  = res.bse.get(wname, np.nan)
    p  = res.pvalues.get(wname, np.nan)
    st = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
    return c, s, p, int(res.nobs), st, res


def report(label, c, s, p, n, st, ymean, section):
    log(f"  {label:<58} coef={c:+.5f}  SE={s:.5f}  p={p:.4f}{st:<3}  "
        f"N={n:,}  Ymean={ymean:.4f}")
    results_rows.append({"section":section,"label":label,
                         "coef":round(c,6),"se":round(s,6),
                         "p":round(p,4),"stars":st,"n":n,"ymean":round(ymean,4)})


# =============================================================================
# SECTION 1 — Additional outcome margins
# =============================================================================
log("\n" + "=" * 70)
log("SECTION 1 — Additional outcome margins (K=6, loser=Yes)")
log("Instrument: w6_IV_Z_R1  |  SE: state-clustered")
log("=" * 70)

new_outcomes = [
    ("Y_R2_apply",      "R2 grant application  (competitive, pre-R3)"),
    ("Y_any_post_r1",   "Any post-R1 CSBP entry (R2 OR R3)"),
    ("Y_R3_apply",      "R3 rebate application [reference]"),
    ("wri_post_r1_cum", "WRI cumulative post-R1 [reference]"),
]

for sample_label, d_s in [("Full sample", d), ("Priority subsample", d_pri)]:
    log(f"\n  {sample_label}")
    d_sk = d_s.dropna(subset=["w6_IV_Z_R1"]).copy()
    ctrl = get_ctrl(d_sk, k=6, loser=True)
    for outcome, desc in new_outcomes:
        if outcome not in d_sk.columns:
            log(f"    SKIP {outcome}: not in dataset")
            continue
        Y = d_sk[outcome].astype(float)
        W = d_sk["w6_IV_Z_R1"].fillna(0)
        c, s, p, n, st, _ = fit_ols(Y, W, ctrl, d_sk["state"])
        label = f"  {sample_label[:4]}, {desc}"
        report(label, c, s, p, n, st, float(Y.mean()), "S1_outcomes")


# =============================================================================
# SECTION 2 — Conley spatial HAC
# =============================================================================
log("\n" + "=" * 70)
log("SECTION 2 — Conley spatial HAC (K=6, loser=Yes, primary outcomes)")
log("Cutoff distances: 100km, 200km, 300km")
log("=" * 70)

def conley_se(res, lats, lons, cutoff_km):
    """
    Compute Conley (1999) spatially-clustered SE (vectorized).
    Uses a uniform kernel within cutoff_km.
    Builds a sparse binary neighbor matrix and computes the meat of the
    sandwich as (Xe)' A (Xe) where A[i,j]=1 iff dist(i,j) <= cutoff.
    Returns array of SEs matching res.params.
    """
    import scipy.sparse as sp_sparse
    n, k = res.model.exog.shape
    X = res.model.exog           # (n, k)
    e = res.resid.values         # (n,)
    Xe = X * e[:, None]          # (n, k)  — score contributions

    # Convert lat/lon to 3D unit-sphere coords for cKDTree distance query
    R_km = 6371.0
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    xyz = np.column_stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r)
    ])                           # (n, 3)

    # Chord distance corresponding to arc cutoff_km
    chord = 2.0 * np.sin(cutoff_km / (2.0 * R_km))

    tree = cKDTree(xyz)
    # query_ball_tree returns list of neighbor lists (includes self)
    pairs = tree.query_ball_tree(tree, r=chord)   # list of length n

    # Build sparse (n x n) adjacency matrix (uniform kernel, includes diagonal)
    rows_sp, cols_sp = [], []
    for i, nbrs in enumerate(pairs):
        rows_sp.extend([i] * len(nbrs))
        cols_sp.extend(nbrs)
    data_sp = np.ones(len(rows_sp))
    A = sp_sparse.csr_matrix((data_sp, (rows_sp, cols_sp)), shape=(n, n))

    # Meat = Xe' A Xe   (vectorized, no Python loop over obs)
    meat = Xe.T @ (A @ Xe)      # (k, k)  — fast sparse matmul

    bread = np.linalg.inv(X.T @ X)
    V = (n / (n - k)) * bread @ meat @ bread
    return np.sqrt(np.diag(V))


for outcome, outlabel in [("wri_post_r1_cum","CUM"), ("Y_R3_apply","APP"),
                           ("Y_any_post_r1","ANY")]:
    log(f"\n  Outcome = {outcome}")
    d_k = d.dropna(subset=["w6_IV_Z_R1","lat","lon"]).copy()
    ctrl = get_ctrl(d_k, k=6, loser=True)
    Y = d_k[outcome].astype(float)
    W = d_k["w6_IV_Z_R1"].fillna(0)

    # Base state-clustered result
    c, s, p, n, st, res = fit_ols(Y, W, ctrl, d_k["state"])
    log(f"    State-clustered:  coef={c:+.5f}  SE={s:.5f}  p={p:.4f}{st}  N={n:,}")
    results_rows.append({"section":"S2_conley","label":f"State-clustered|{outcome}",
                         "coef":round(c,6),"se":round(s,6),"p":round(p,4),
                         "stars":st,"n":n,"ymean":round(float(Y.mean()),4)})

    # Conley SE at different cutoffs
    lats = d_k["lat"].values.astype(float)
    lons = d_k["lon"].values.astype(float)
    X_full = sm.add_constant(pd.concat([W.rename("treat"), ctrl], axis=1),
                              has_constant="add").dropna(axis=1)
    res_ols = sm.OLS(Y, X_full).fit()

    for cutoff in [100, 200, 300]:
        try:
            se_conley = conley_se(res_ols, lats, lons, cutoff)
            idx = list(X_full.columns).index("treat")
            se_c = se_conley[idx]
            t_c  = c / se_c
            p_c  = 2 * (1 - float(pd.Series([abs(t_c)]).map(
                lambda t: __import__('scipy').stats.t.cdf(t, df=n-len(X_full.columns))).iloc[0]))
            st_c = "***" if p_c<0.01 else "**" if p_c<0.05 else "*" if p_c<0.1 else ""
            log(f"    Conley {cutoff:>3}km:       coef={c:+.5f}  SE={se_c:.5f}  p={p_c:.4f}{st_c}")
            results_rows.append({"section":"S2_conley",
                                  "label":f"Conley_{cutoff}km|{outcome}",
                                  "coef":round(c,6),"se":round(se_c,6),"p":round(p_c,4),
                                  "stars":st_c,"n":n,"ymean":round(float(Y.mean()),4)})
        except Exception as e:
            log(f"    Conley {cutoff}km: FAILED — {e}")


# =============================================================================
# SECTION 3 — Donut test
# =============================================================================
log("\n" + "=" * 70)
log("SECTION 3 — Donut test")
log("Compare: full K=6 vs donut (drop nearest 1-2 neighbors)")
log("Donut instrument built by subtracting K=1,2 lags from K=6 lags")
log("=" * 70)

# We don't have K=1 or K=2 lags stored. Build them from the KNN weight matrix.
# Alternative: approximate with w3_IV_Z_R1 (K=3 lags exist? check)
k3_exists = "w3_IV_Z_R1" in df.columns

if k3_exists:
    log("  K=3 lags found — using w3_IV_Z_R1 as donut (neighbors 3-6 average)")
else:
    # Load the KNN weight matrix and compute K=1,2 lags directly
    import scipy.sparse as sp
    npz_path = CLEAN / "knn_weights_k6.npz"
    if npz_path.exists():
        log(f"  Loading K=6 weight matrix from {npz_path}")
        from scipy.sparse import load_npz
        W6 = load_npz(str(npz_path))

        # Need to recompute K=1 and K=2 weight matrices from the shapefile
        shp = gpd.read_file(SHP_FILE)
        shp["nces_id"] = shp["GEOID"].astype(str).str.zfill(7)
        shp_proj = shp.to_crs("EPSG:5070")
        shp_proj["cx"] = shp_proj.geometry.centroid.x
        shp_proj["cy"] = shp_proj.geometry.centroid.y
        shp_m = shp_proj[["nces_id","cx","cy"]].drop_duplicates("nces_id")
        shp_m = shp_m.merge(df[["nces_id"]], on="nces_id", how="inner").reset_index(drop=True)

        from sklearn.neighbors import NearestNeighbors
        coords = shp_m[["cx","cy"]].values
        nn6 = NearestNeighbors(n_neighbors=7, algorithm="ball_tree").fit(coords)
        distances, indices = nn6.kneighbors(coords)
        # indices[:,0] is self; indices[:,1:3] are the 2 nearest neighbors
        n_dist = len(shp_m)

        for K_near, label_k in [(1,"k1"), (2,"k2")]:
            rows, cols, vals = [], [], []
            for i in range(n_dist):
                nbrs = indices[i, 1:K_near+1]
                for j in nbrs:
                    rows.append(i)
                    cols.append(j)
                    vals.append(1.0 / K_near)
            W_k = sp.csr_matrix((vals, (rows, cols)), shape=(n_dist, n_dist))
            # Compute lags for the main variables
            shp_m[f"nces_id_{label_k}"] = shp_m["nces_id"]

        # Merge shp_m index back to df
        idx_map = shp_m.reset_index()[["nces_id","index"]].rename(columns={"index":"shp_idx"})
        df2 = df.merge(idx_map, on="nces_id", how="left")

        for var in ["IV_Z_R1","IS_R1_LOSER"]:
            if var not in df.columns:
                continue
            var_vals = np.zeros(n_dist)
            merged = shp_m.merge(df[["nces_id",var]], on="nces_id", how="left")
            var_vals = merged[var].fillna(0).values

            for K_near, label_k in [(1,"k1"), (2,"k2")]:
                rows2, cols2, vals2 = [], [], []
                for i in range(n_dist):
                    nbrs = indices[i, 1:K_near+1]
                    for j in nbrs:
                        rows2.append(i); cols2.append(j); vals2.append(1.0/K_near)
                W_kn = sp.csr_matrix((vals2, (rows2, cols2)), shape=(n_dist, n_dist))
                lag_vals = np.array(W_kn.dot(var_vals)).flatten()
                lag_df = pd.DataFrame({"nces_id": shp_m["nces_id"].values,
                                       f"w{K_near}_{var}": lag_vals})
                df = df.merge(lag_df, on="nces_id", how="left")

        # Donut = K=6 lag minus K=2 lag, rescaled (approximation)
        for var in ["IV_Z_R1","IS_R1_LOSER"]:
            if f"w2_{var}" in df.columns and f"w6_{var}" in df.columns:
                # Donut: average of neighbors 3-6 only
                # w6 = (w2 * 2 + donut_4 * 4) / 6
                # => donut_4 = (w6 * 6 - w2 * 2) / 4
                df[f"wdonut_{var}"] = (df[f"w6_{var}"] * 6 - df[f"w2_{var}"] * 2) / 4

        df["wdonut_IV_Z_R1"]   = df.get("wdonut_IV_Z_R1", np.nan)
        df["wdonut_IS_R1_LOSER"] = df.get("wdonut_IS_R1_LOSER", np.nan)
        log("  Donut lags constructed (neighbors 3-6)")
    else:
        log(f"  KNN weight matrix not found at {npz_path} — skipping donut")

# Run donut test
donut_col = "wdonut_IV_Z_R1"
if donut_col in df.columns and df[donut_col].notna().sum() > 100:
    d_donut = df[df["IV_Z_R1"]==0].dropna(subset=required+[donut_col]).copy().reset_index(drop=True)
    d_donut["log_enroll"] = np.log(d_donut["enrollment"].clip(lower=1))
    d_donut["log_income"]  = np.log(d_donut["median_income"].clip(lower=1))
    log(f"\n  Donut sample N={len(d_donut):,}")
    for outcome in ["wri_post_r1_cum","Y_any_post_r1","Y_R3_apply"]:
        # Full K=6
        ctrl_k6 = get_ctrl(d_donut, k=6, loser=True)
        Y = d_donut[outcome].astype(float)
        W_k6 = d_donut["w6_IV_Z_R1"].fillna(0)
        c6, s6, p6, n6, st6, _ = fit_ols(Y, W_k6, ctrl_k6, d_donut["state"])
        log(f"\n    {outcome} — full K=6:    coef={c6:+.5f}  SE={s6:.5f}  p={p6:.4f}{st6}")
        # Donut K=3-6
        loser_donut = "wdonut_IS_R1_LOSER" if "wdonut_IS_R1_LOSER" in d_donut.columns else None
        ctrl_dn = get_ctrl(d_donut, k=6, loser=False)
        if loser_donut and loser_donut in d_donut.columns:
            ctrl_dn["w_loser"] = d_donut[loser_donut].fillna(0)
        W_dn = d_donut[donut_col].fillna(0)
        cd, sd, pd_, nd, std, _ = fit_ols(Y, W_dn, ctrl_dn, d_donut["state"])
        log(f"    {outcome} — donut K=3-6: coef={cd:+.5f}  SE={sd:.5f}  p={pd_:.4f}{std}")
        for row in [(f"K6|{outcome}", c6, s6, p6, n6, st6),
                    (f"Donut|{outcome}", cd, sd, pd_, nd, std)]:
            results_rows.append({"section":"S3_donut","label":row[0],
                                  "coef":round(row[1],6),"se":round(row[2],6),
                                  "p":round(row[3],4),"stars":row[5],"n":row[4],
                                  "ymean":round(float(Y.mean()),4)})
else:
    log("\n  Donut lags not available — skipping donut test")
    log("  (Run with K=1,2 weight matrices pre-computed to enable)")


# =============================================================================
# SECTION 4 — Vendor network channel
# =============================================================================
log("\n" + "=" * 70)
log("SECTION 4 — Vendor network channel")
log("shares_dealer_with_r1: district has a bus from a dealer who also served R1 winners")
log("Caveat: only 31% of districts have dealer data; 0 means missing OR no shared dealer")
log("=" * 70)

# 4a: Does controlling for vendor network attenuate the geographic peer effect?
d_vend = d[d["shares_dealer_with_r1"].notna()].dropna(subset=["w6_IV_Z_R1"]).copy()
log(f"\n  Full sample N={len(d_vend):,}  (shares_dealer coverage: "
    f"{100*d_vend['shares_dealer_with_r1'].notna().mean():.0f}%)")
log(f"  Districts sharing dealer with R1 winner: {d_vend['shares_dealer_with_r1'].sum():,}")

for outcome in ["wri_post_r1_cum","Y_any_post_r1","Y_R3_apply"]:
    log(f"\n  Outcome: {outcome}")
    Y = d_vend[outcome].astype(float)
    W = d_vend["w6_IV_Z_R1"].fillna(0)

    # Without vendor control
    ctrl_base = get_ctrl(d_vend, k=6, loser=True)
    c0, s0, p0, n0, st0, _ = fit_ols(Y, W, ctrl_base, d_vend["state"])
    log(f"    Without vendor ctrl: coef={c0:+.5f}  SE={s0:.5f}  p={p0:.4f}{st0}")

    # With vendor control
    ctrl_v = get_ctrl(d_vend, k=6, loser=True,
                      extra={"shares_dealer": "shares_dealer_with_r1"})
    cv, sv, pv, nv, stv, resv = fit_ols(Y, W, ctrl_v, d_vend["state"])
    vend_coef = resv.params.get("shares_dealer", np.nan)
    vend_p    = resv.pvalues.get("shares_dealer", np.nan)
    log(f"    With vendor ctrl   : coef={cv:+.5f}  SE={sv:.5f}  p={pv:.4f}{stv}  "
        f"[vendor_coef={vend_coef:+.5f} p={vend_p:.3f}]")
    results_rows.append({"section":"S4_vendor","label":f"No_vendor|{outcome}",
                         "coef":round(c0,6),"se":round(s0,6),"p":round(p0,4),
                         "stars":st0,"n":n0,"ymean":round(float(Y.mean()),4)})
    results_rows.append({"section":"S4_vendor","label":f"With_vendor|{outcome}",
                         "coef":round(cv,6),"se":round(sv,6),"p":round(pv,4),
                         "stars":stv,"n":nv,"ymean":round(float(Y.mean()),4)})

# 4b: Heterogeneity split — dealer-connected vs not
log(f"\n  Heterogeneity: dealer-connected vs not (wri_post_r1_cum)")
for grp, mask in [("Dealer-connected", d_vend["shares_dealer_with_r1"]==1),
                  ("No shared dealer ", d_vend["shares_dealer_with_r1"]==0)]:
    d_g = d_vend[mask].dropna(subset=["w6_IV_Z_R1"]).copy()
    if len(d_g) < 200:
        log(f"    {grp}: N={len(d_g):,} — too small, skip")
        continue
    ctrl_g = get_ctrl(d_g, k=6, loser=True)
    Y = d_g["wri_post_r1_cum"].astype(float)
    W = d_g["w6_IV_Z_R1"].fillna(0)
    c, s, p, n, st, _ = fit_ols(Y, W, ctrl_g, d_g["state"])
    log(f"    {grp}: N={n:,}  coef={c:+.5f}  SE={s:.5f}  p={p:.4f}{st}")


# =============================================================================
# SECTION 5 — Priority subsample decomposition
# =============================================================================
log("\n" + "=" * 70)
log("SECTION 5 — Priority decomposition (K=6, loser=Yes, wri_post_r1_cum)")
log("Which dimension of priority is driving the stronger effect?")
log("=" * 70)

# PM2.5 quartile interaction
log("\n  5a. PM2.5 quartile heterogeneity (full sample)")
d_q = d.dropna(subset=["w6_IV_Z_R1","pm25_q"]).copy()
d_q = d_q[d_q["pm25_q"] != "Unknown"]
for q in ["Q1","Q2","Q3","Q4"]:
    d_qk = d_q[d_q["pm25_q"] == q].copy()
    if len(d_qk) < 100:
        continue
    ctrl = get_ctrl(d_qk, k=6, loser=True)
    Y  = d_qk["wri_post_r1_cum"].astype(float)
    W  = d_qk["w6_IV_Z_R1"].fillna(0)
    c, s, p, n, st, _ = fit_ols(Y, W, ctrl, d_qk["state"])
    log(f"    PM2.5 {q}: N={n:,}  mean_pm25={d_qk['pm25'].mean():.2f}  "
        f"coef={c:+.5f}  SE={s:.5f}  p={p:.4f}{st}")
    results_rows.append({"section":"S5_priority","label":f"PM25_{q}|wri_post_r1_cum",
                         "coef":round(c,6),"se":round(s,6),"p":round(p,4),
                         "stars":st,"n":n,"ymean":round(float(Y.mean()),4)})

# Poverty quartile interaction
log("\n  5b. Poverty quartile heterogeneity (full sample)")
d_pv = d.dropna(subset=["w6_IV_Z_R1","poverty_q"]).copy()
d_pv = d_pv[d_pv["poverty_q"] != "Unknown"]
for q in ["Q1","Q2","Q3","Q4"]:
    d_qk = d_pv[d_pv["poverty_q"] == q].copy()
    if len(d_qk) < 100:
        continue
    ctrl = get_ctrl(d_qk, k=6, loser=True)
    Y  = d_qk["wri_post_r1_cum"].astype(float)
    W  = d_qk["w6_IV_Z_R1"].fillna(0)
    c, s, p, nobs, st, _ = fit_ols(Y, W, ctrl, d_qk["state"])
    log(f"    Poverty {q}: N={nobs:,}  mean_pov={d_qk['poverty_rate'].mean():.3f}  "
        f"coef={c:+.5f}  SE={s:.5f}  p={p:.4f}{st}")
    results_rows.append({"section":"S5_priority","label":f"Poverty_{q}|wri_post_r1_cum",
                         "coef":round(c,6),"se":round(s,6),"p":round(p,4),
                         "stars":st,"n":nobs,"ymean":round(float(Y.mean()),4)})

# Priority subsample: which dimension drives it?
log("\n  5c. Within priority subsample: PM2.5 high vs low")
d_pri_q = d_pri.dropna(subset=["w6_IV_Z_R1","pm25_q"]).copy()
d_pri_q = d_pri_q[d_pri_q["pm25_q"] != "Unknown"]
pm25_med = d_pri_q["pm25"].median()
log(f"    Priority median PM2.5: {pm25_med:.2f}")
for grp, mask in [
    ("High PM2.5 (above median)", d_pri_q["pm25"] >= pm25_med),
    ("Low PM2.5  (below median)", d_pri_q["pm25"] <  pm25_med),
]:
    d_g = d_pri_q[mask].copy()
    ctrl = get_ctrl(d_g, k=6, loser=True)
    Y = d_g["wri_post_r1_cum"].astype(float)
    W = d_g["w6_IV_Z_R1"].fillna(0)
    c, s, p, n, st, _ = fit_ols(Y, W, ctrl, d_g["state"])
    log(f"    {grp}: N={n:,}  coef={c:+.5f}  SE={s:.5f}  p={p:.4f}{st}")
    results_rows.append({"section":"S5_priority","label":f"{grp}|wri_post_r1_cum",
                         "coef":round(c,6),"se":round(s,6),"p":round(p,4),
                         "stars":st,"n":n,"ymean":round(float(Y.mean()),4)})

log("\n  5d. Within priority subsample: poverty high vs low")
pov_med = d_pri["poverty_rate"].median()
for grp, mask in [
    ("High poverty (above median)", d_pri["poverty_rate"] >= pov_med),
    ("Low poverty  (below median)", d_pri["poverty_rate"] <  pov_med),
]:
    d_g = d_pri[mask].dropna(subset=["w6_IV_Z_R1"]).copy()
    ctrl = get_ctrl(d_g, k=6, loser=True)
    Y = d_g["wri_post_r1_cum"].astype(float)
    W = d_g["w6_IV_Z_R1"].fillna(0)
    c, s, p, n, st, _ = fit_ols(Y, W, ctrl, d_g["state"])
    log(f"    {grp}: N={n:,}  coef={c:+.5f}  SE={s:.5f}  p={p:.4f}{st}")
    results_rows.append({"section":"S5_priority","label":f"{grp}|wri_post_r1_cum",
                         "coef":round(c,6),"se":round(s,6),"p":round(p,4),
                         "stars":st,"n":n,"ymean":round(float(Y.mean()),4)})


# =============================================================================
# SECTION 6 — Climatic peers
# =============================================================================
log("\n" + "=" * 70)
log("SECTION 6 — Climatic peers")
log("Peer group = districts in same PRISM temperature quintile (same-climate peers)")
log("Within-quintile, identify districts with winning neighbors in same climate band")
log("=" * 70)

# Use PM2.5 as climate proxy (already available) as PRISM data needs separate load
# Better: use the PRISM climate data if available
import os
climate_dir = "1_Data/Raw/Climate"
climate_files = os.listdir(climate_dir) if os.path.exists(climate_dir) else []
log(f"  Climate raw files: {climate_files}")

# Check if PRISM temp/precip are already in the dataset
prism_cols = [c for c in df.columns if "temp" in c.lower() or "precip" in c.lower()
              or "prism" in c.lower() or "tmin" in c.lower() or "tmax" in c.lower()]
log(f"  PRISM columns in dataset: {prism_cols}")

if prism_cols:
    # Use first available temperature column
    temp_col = [c for c in prism_cols if "temp" in c.lower() or "tmax" in c.lower()][0] \
        if any("temp" in c.lower() or "tmax" in c.lower() for c in prism_cols) else prism_cols[0]
    log(f"  Using climate variable: {temp_col}")
    d_clim = d.dropna(subset=["w6_IV_Z_R1", temp_col]).copy()
    d_clim["climate_q"] = pd.qcut(d_clim[temp_col], 5, labels=["C1","C2","C3","C4","C5"])

    # For each climate quintile, run the main spec
    log(f"\n  Climate quintile results (wri_post_r1_cum, K=6, loser=Yes):")
    for q in ["C1","C2","C3","C4","C5"]:
        d_cq = d_clim[d_clim["climate_q"] == q].copy()
        if len(d_cq) < 100:
            continue
        ctrl = get_ctrl(d_cq, k=6, loser=True)
        Y = d_cq["wri_post_r1_cum"].astype(float)
        W = d_cq["w6_IV_Z_R1"].fillna(0)
        c, s, p, n, st, _ = fit_ols(Y, W, ctrl, d_cq["state"])
        log(f"    {q}: N={n:,}  mean_{temp_col}={d_cq[temp_col].mean():.1f}  "
            f"coef={c:+.5f}  SE={s:.5f}  p={p:.4f}{st}")
        results_rows.append({"section":"S6_climate","label":f"Climate_{q}|wri_post_r1_cum",
                             "coef":round(c,6),"se":round(s,6),"p":round(p,4),
                             "stars":st,"n":n,"ymean":round(float(Y.mean()),4)})
else:
    log("  No PRISM columns found in dataset. Checking raw climate files...")
    if climate_files:
        log(f"  Climate files present but not yet merged: {climate_files[:5]}")
        log("  ACTION NEEDED: merge PRISM data into analysis dataset to enable Section 6")
    else:
        log("  No climate raw files found. Section 6 requires PRISM data.")


# =============================================================================
# Save
# =============================================================================
res_df = pd.DataFrame(results_rows)
res_df.to_csv(TABLES_DIR / "extended_iv.csv", index=False)
log(f"\n  Saved: {TABLES_DIR / 'extended_iv.csv'}")

with open(LOGS_DIR / "extended_iv.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"  Saved: {LOGS_DIR / 'extended_iv.txt'}")
