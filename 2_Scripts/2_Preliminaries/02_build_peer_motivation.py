from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANALYSIS_DATASET, SHP_FILE, TABLES_DIR, LOGS_DIR, ensure_dirs

warnings.filterwarnings("ignore")
ensure_dirs()

OUTDIR = TABLES_DIR / "Preliminaries"
LOGDIR = LOGS_DIR / "Preliminaries"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)

K = 6
TARGET_OUTCOMES = ["Y_R3_apply", "wri_any_2023_24", "wri_post_r1_cum"]

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def clean_nces(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.split(".").str[0].str.strip()
    out = out.str.extract(r"(\d+)", expand=False)
    return out.str.zfill(7)


def load_geo_analysis() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS_DATASET, low_memory=False, dtype={"nces_id": str})
    df["nces_id"] = df["nces_id"].str.zfill(7)

    shp = gpd.read_file(SHP_FILE)
    shp["nces_id"] = clean_nces(shp["GEOID"])
    shp = shp.to_crs("EPSG:5070")
    shp["cx"] = shp.geometry.centroid.x
    shp["cy"] = shp.geometry.centroid.y
    geo = shp[["nces_id", "cx", "cy"]].drop_duplicates("nces_id")

    out = df.merge(geo, on="nces_id", how="inner")
    log(f"Matched districts with geometry: {len(out):,} / {len(df):,}")
    return out


def build_knn_indices(df: pd.DataFrame, k: int) -> np.ndarray:
    coords = df[["cx", "cy"]].to_numpy()
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)
    return indices[:, 1:]


def neighbor_share(values: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    return values[neighbor_idx].mean(axis=1)


def morans_i(values: np.ndarray, neighbor_idx: np.ndarray) -> float:
    x = values.astype(float)
    if np.isclose(x.var(), 0):
        return np.nan
    xc = x - x.mean()
    numerator = (xc * xc[neighbor_idx].sum(axis=1)).sum()
    denominator = (xc ** 2).sum()
    n = len(x)
    s0 = n * neighbor_idx.shape[1]
    return float((n / s0) * (numerator / denominator))


def build_clustering_summary(df: pd.DataFrame, neighbor_idx: np.ndarray) -> pd.DataFrame:
    rows = []
    for col in ["IV_Z_R1", "IS_R1_LOSER", "Y_R3_apply", "wri_any_2023_24", "wri_post_r1_cum"]:
        own = df[col].astype(float).to_numpy()
        peer = neighbor_share(own, neighbor_idx)
        corr = float(np.corrcoef(own, peer)[0, 1])
        rows.append({
            "variable": col,
            "k": neighbor_idx.shape[1],
            "mean": float(own.mean()),
            "mean_neighbor_share": float(peer.mean()),
            "own_neighbor_correlation": corr,
            "morans_i": morans_i(own, neighbor_idx),
        })
    return pd.DataFrame(rows)


def fit_peer_regression(df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, list[str]]:
    sample = df[df["IV_Z_R1"] == 0].copy()
    sample["peer_r1_winner_share"] = sample["peer_r1_winner_share"].astype(float)
    sample["peer_r1_loser_share"] = sample["peer_r1_loser_share"].astype(float)

    for col in ["enrollment", "median_income", "poverty_rate", "pct_white", "pm25", "pct_dem_2020", "priority_r1"]:
        sample[col] = pd.to_numeric(sample[col], errors="coerce")

    sample["log_enrollment"] = np.log1p(sample["enrollment"])
    state_dummies = pd.get_dummies(sample["state"], prefix="state", drop_first=True, dtype=float)
    X = pd.concat([
        sample[["peer_r1_winner_share", "peer_r1_loser_share", "log_enrollment", "median_income", "poverty_rate", "pct_white", "pm25", "pct_dem_2020", "priority_r1"]],
        state_dummies,
    ], axis=1)
    y = sample[outcome].astype(float)

    valid = y.notna() & X.notna().all(axis=1)
    X = X.loc[valid]
    y = y.loc[valid]
    if X.empty:
        return pd.DataFrame(), []

    X = X[[c for c in X.columns if X[c].nunique(dropna=False) > 1]]
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X).fit(cov_type="HC1")

    terms = ["peer_r1_winner_share", "peer_r1_loser_share", "log_enrollment", "median_income", "poverty_rate", "pct_white", "pm25", "pct_dem_2020", "priority_r1"]
    rows = []
    for term in terms:
        if term not in model.params.index:
            continue
        rows.append({
            "outcome": outcome,
            "term": term,
            "coef": model.params[term],
            "se": model.bse[term],
            "pvalue": model.pvalues[term],
            "nobs": int(model.nobs),
            "r_squared": model.rsquared,
        })
    summary = [
        f"## {outcome}",
        f"nobs={int(model.nobs)}",
        f"r_squared={model.rsquared:.4f}",
    ]
    for term in ["peer_r1_winner_share", "peer_r1_loser_share"]:
        if term in model.params.index:
            summary.append(f"{term}: coef={model.params[term]:.6f}, se={model.bse[term]:.6f}, pvalue={model.pvalues[term]:.6f}")
    return pd.DataFrame(rows), summary


def main() -> None:
    df = load_geo_analysis()
    neighbor_idx = build_knn_indices(df, K)

    winner_neighbor_count = df["IV_Z_R1"].astype(float).to_numpy()[neighbor_idx].sum(axis=1)
    loser_neighbor_count = df["IS_R1_LOSER"].astype(float).to_numpy()[neighbor_idx].sum(axis=1)
    df["peer_r1_winner_share"] = neighbor_share(df["IV_Z_R1"].astype(float).to_numpy(), neighbor_idx)
    df["peer_r1_loser_share"] = neighbor_share(df["IS_R1_LOSER"].astype(float).to_numpy(), neighbor_idx)
    df["winner_neighbor_count"] = winner_neighbor_count.astype(int)
    df["loser_neighbor_count"] = loser_neighbor_count.astype(int)

    clustering = build_clustering_summary(df, neighbor_idx)

    exposure_rows = []
    sample = df[df["IV_Z_R1"] == 0].copy()
    sample = sample[sample["peer_r1_winner_share"].notna()].copy()
    sample["winner_exposure_bin"] = np.where(
        sample["winner_neighbor_count"] >= 2,
        "2+",
        sample["winner_neighbor_count"].astype(int).astype(str),
    )
    sample["winner_exposure_order"] = sample["winner_exposure_bin"].map({"0": 0, "1": 1, "2+": 2})
    for outcome in TARGET_OUTCOMES:
        temp = (
            sample.groupby(["winner_exposure_bin", "winner_exposure_order"], as_index=False)
            .agg(
                districts_n=("nces_id", "size"),
                winner_neighbor_count=("winner_neighbor_count", "mean"),
                peer_r1_winner_share=("peer_r1_winner_share", "mean"),
                outcome_mean=(outcome, "mean"),
            )
            .sort_values("winner_exposure_order")
        )
        temp["outcome"] = outcome
        exposure_rows.append(temp.drop(columns=["winner_exposure_order"]))
    exposure_bins = pd.concat(exposure_rows, ignore_index=True)

    reg_rows = []
    summaries = []
    for outcome in TARGET_OUTCOMES:
        reg_df, summary = fit_peer_regression(df, outcome)
        if not reg_df.empty:
            reg_rows.append(reg_df)
            summaries.extend(summary + [""])
    regressions = pd.concat(reg_rows, ignore_index=True) if reg_rows else pd.DataFrame()

    clustering.to_csv(OUTDIR / "peer_motivation_clustering_summary.csv", index=False)
    exposure_bins.to_csv(OUTDIR / "peer_motivation_exposure_bins.csv", index=False)
    regressions.to_csv(OUTDIR / "peer_motivation_regressions.csv", index=False)
    (LOGDIR / "peer_motivation_summary.txt").write_text("\n".join(summaries), encoding="utf-8")

    log("Saved peer-motivation outputs:")
    for name in [
        "peer_motivation_clustering_summary.csv",
        "peer_motivation_exposure_bins.csv",
        "peer_motivation_regressions.csv",
    ]:
        log(f"  {OUTDIR / name}")
    (LOGDIR / "peer_motivation_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
