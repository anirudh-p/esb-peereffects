from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
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

ADOPTION_COL = "has_wri_esb"
K = 6
PERMUTATIONS = 499
SEED = 12345

log_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    log_lines.append(str(msg))


def clean_nces(series: pd.Series) -> pd.Series:
    out = pd.Series(series, copy=False)
    out = out.where(out.notna(), np.nan)
    out = out.astype(str).str.split(".").str[0].str.strip()
    out = out.str.extract(r"(\d+)", expand=False)
    out = out.where(out.notna() & out.ne(""), np.nan)
    return out.str.zfill(7)


def load_geo_adoption_dataset() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS_DATASET, low_memory=False, dtype={"nces_id": str})
    df["nces_id"] = df["nces_id"].str.zfill(7)
    df[ADOPTION_COL] = df["wri_awarded_by_2024"].fillna(0).astype(int)

    shp = gpd.read_file(SHP_FILE)
    shp["nces_id"] = clean_nces(shp["GEOID"])
    shp = shp.to_crs("EPSG:5070")
    shp["cx"] = shp.geometry.centroid.x
    shp["cy"] = shp.geometry.centroid.y
    geo = shp[["nces_id", "cx", "cy"]].drop_duplicates("nces_id")

    out = df.merge(geo, on="nces_id", how="inner")
    log(f"Matched districts with geometry: {len(out):,} / {len(df):,}")
    return out


def remove_legacy_outputs() -> None:
    for name in [
        "peer_motivation_clustering_summary.csv",
        "peer_motivation_exposure_bins.csv",
        "peer_motivation_regressions.csv",
    ]:
        path = OUTDIR / name
        if path.exists():
            path.unlink()


def build_knn_indices(df: pd.DataFrame, k: int) -> np.ndarray:
    coords = df[["cx", "cy"]].to_numpy()
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)
    return indices[:, 1:]


def neighbor_share(values: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    return values[neighbor_idx].mean(axis=1)


def neighbor_count(values: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    return values[neighbor_idx].sum(axis=1)


def morans_i(values: np.ndarray, neighbor_idx: np.ndarray) -> float:
    x = values.astype(float)
    if np.isclose(x.var(), 0):
        return np.nan
    xc = x - x.mean()
    numerator = (xc * xc[neighbor_idx].sum(axis=1)).sum()
    denominator = (xc**2).sum()
    n = len(x)
    s0 = n * neighbor_idx.shape[1]
    return float((n / s0) * (numerator / denominator))


def morans_i_permutation(values: np.ndarray, neighbor_idx: np.ndarray) -> tuple[float, float, float]:
    observed = morans_i(values, neighbor_idx)
    if np.isnan(observed):
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(SEED)
    draws = np.empty(PERMUTATIONS)
    x = values.astype(float)
    for i in range(PERMUTATIONS):
        draws[i] = morans_i(rng.permutation(x), neighbor_idx)

    pvalue = (np.sum(np.abs(draws) >= abs(observed)) + 1) / (PERMUTATIONS + 1)
    return observed, float(draws.mean()), float(pvalue)


def build_clustering_summary(df: pd.DataFrame, neighbor_idx: np.ndarray) -> pd.DataFrame:
    own = df[ADOPTION_COL].astype(float).to_numpy()
    peer_share = neighbor_share(own, neighbor_idx)
    peer_count = neighbor_count(own, neighbor_idx)
    moran_value, moran_null_mean, moran_pvalue = morans_i_permutation(own, neighbor_idx)

    adopter_mask = own == 1
    non_adopter_mask = own == 0

    summary = {
        "districts_with_geometry": int(len(df)),
        "adoption_rate": float(own.mean()),
        "k_neighbors": int(neighbor_idx.shape[1]),
        "mean_neighbor_adoption_share": float(peer_share.mean()),
        "own_neighbor_correlation": float(np.corrcoef(own, peer_share)[0, 1]),
        "morans_i": moran_value,
        "morans_i_null_mean": moran_null_mean,
        "morans_i_two_sided_pvalue": moran_pvalue,
        "mean_adopted_neighbors": float(peer_count.mean()),
        "mean_neighbor_share_if_adopter": float(peer_share[adopter_mask].mean()),
        "mean_neighbor_share_if_non_adopter": float(peer_share[non_adopter_mask].mean()),
        "neighbor_share_gap": float(peer_share[adopter_mask].mean() - peer_share[non_adopter_mask].mean()),
    }
    return pd.DataFrame([summary])


def build_neighbor_gradient(df: pd.DataFrame, neighbor_idx: np.ndarray) -> pd.DataFrame:
    own = df[ADOPTION_COL].astype(float).to_numpy()
    peer_count = neighbor_count(own, neighbor_idx).astype(int)
    peer_share = neighbor_share(own, neighbor_idx)

    temp = df[["nces_id"]].copy()
    temp["adopted_neighbor_count"] = peer_count
    temp["peer_adoption_share"] = peer_share
    temp[ADOPTION_COL] = own
    temp["adopted_neighbor_bin"] = np.where(
        temp["adopted_neighbor_count"] >= 2,
        "2+",
        temp["adopted_neighbor_count"].astype(str),
    )
    temp["bin_order"] = temp["adopted_neighbor_bin"].map({"0": 0, "1": 1, "2+": 2})

    return (
        temp.groupby(["adopted_neighbor_bin", "bin_order"], as_index=False)
        .agg(
            districts_n=("nces_id", "size"),
            mean_adopted_neighbor_count=("adopted_neighbor_count", "mean"),
            mean_peer_adoption_share=("peer_adoption_share", "mean"),
            own_adoption_rate=(ADOPTION_COL, "mean"),
        )
        .sort_values("bin_order")
        .drop(columns=["bin_order"])
        .reset_index(drop=True)
    )


def main() -> None:
    df = load_geo_adoption_dataset()
    neighbor_idx = build_knn_indices(df, K)
    remove_legacy_outputs()

    clustering = build_clustering_summary(df, neighbor_idx)
    neighbor_gradient = build_neighbor_gradient(df, neighbor_idx)

    clustering.to_csv(OUTDIR / "adoption_clustering_summary.csv", index=False)
    neighbor_gradient.to_csv(OUTDIR / "adoption_neighbor_gradient.csv", index=False)

    log("Saved adoption clustering outputs:")
    for name in [
        "adoption_clustering_summary.csv",
        "adoption_neighbor_gradient.csv",
    ]:
        log(f"  {OUTDIR / name}")

    (LOGDIR / "adoption_clustering_log.txt").write_text(
        "\n".join(log_lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
