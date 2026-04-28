"""Build spatial and design-based exposure variables for the hazard panel.

The script is deliberately explicit about geometry coverage. It creates:

- EDGE-only coverage diagnostics, to show which LEAs would be dropped if we used
  only school-district boundary files;
- a hybrid coordinate universe, using EDGE internal points when available and
  WRI latitude/longitude as a fallback for unmatched LEAs;
- KNN spatial exposures and R1 priority-BH/recentered design exposure variables;
- a panel-level spatial file ready for Stata estimation.

The main `w{K}_*` variables use the hybrid coordinate universe. The `edge_w6_*`
variables provide an EDGE-only sensitivity surface for the primary K=6 graph.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from scipy.sparse import csr_matrix, save_npz
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPT_DIR))

from config import AUDIT_DIR, CLEANED_DIR, RAW_DIR  # noqa: E402

BASE_FILE = CLEANED_DIR / "analysis_district_base.csv"
PANEL_FILE = CLEANED_DIR / "analysis_hazard_panel.csv"
PANEL_SPATIAL_CSV = CLEANED_DIR / "analysis_hazard_panel_spatial.csv"
PANEL_SPATIAL_DTA = CLEANED_DIR / "analysis_hazard_panel_spatial.dta"
SPATIAL_ORDER_FILE = CLEANED_DIR / "spatial_neighbor_order_hybrid.csv"
EDGE_ORDER_FILE = CLEANED_DIR / "spatial_neighbor_order_edge_only.csv"

SHP_FILE = RAW_DIR / "Spatial" / "EDGE_SCHOOLDISTRICT_TL21_SY2021" / "schooldistrict_sy2021_tl21.shp"
CSB_REBATES_XLSX = RAW_DIR / "WRI" / "CSB_Rebates.xlsx"
CSBP_APPLICANTS_XLSX = RAW_DIR / "WRI" / "CSBP Applicants waitlisted and rejected_11.18.25.xlsx"

K_VALUES = [4, 6, 8, 10]
PRIMARY_K = 6
METERS_PER_MILE = 1609.344
NON_CONTIGUOUS_STATES = {"AK", "HI", "AS", "GU", "MP", "PR", "VI"}
REGULAR_PUBLIC_TYPES = {
    "Regular public school district that is not a component of a supervisory union",
    "Regular public school district that is a component of a supervisory union",
}
CHARTER_TYPES = {"Independent charter district"}
SPECIALIZED_PUBLIC_TYPES = {"Specialized public school district"}
AGENCY_LIKE_TYPES = {
    "Service agency",
    "Supervisory union",
    "Other local education agency",
    "State operated agency",
    "Federal operated agency",
}

STATIC_EXPOSURE_VARS = [
    "r1_rebate_winner",
    "r3_rebate_winner",
    "r1_lottery_applicant",
    "r3_lottery_applicant",
    "any_lottery_applicant",
    "any_rebate_winner",
    "csb_grant_awardee",
    "wri_pre_2022_adopter",
    "wri_has_committed_esbs",
    "survey_2023_response",
    "survey_2024_response",
    "priority_2022_csbp",
    "priority_2023_csbp",
]

EXPOSURE_SHORT_NAMES = {
    "r1_rebate_winner": "r1win",
    "r3_rebate_winner": "r3win",
    "r1_lottery_applicant": "r1app",
    "r3_lottery_applicant": "r3app",
    "any_lottery_applicant": "anyapp",
    "any_rebate_winner": "anywin",
    "csb_grant_awardee": "grant",
    "wri_pre_2022_adopter": "pre22",
    "wri_has_committed_esbs": "commesb",
    "survey_2023_response": "surv23",
    "survey_2024_response": "surv24",
    "priority_2022_csbp": "pri22",
    "priority_2023_csbp": "pri23",
    "pi_r1_bh_priority": "r1pibh",
    "z_r1_recenter_bh": "r1rcbh",
    "r1_design_priority": "r1dpri",
    "r1_design_app_rows": "r1dapps",
}

DYNAMIC_EVENT_COLS = {
    "award": "first_award_year",
    "delivery": "first_delivery_year",
    "operating": "first_operating_year",
    "lottery_apply": "first_lottery_apply_year",
}

INTERESTING_FLAGS = [
    "r1_lottery_applicant",
    "r3_lottery_applicant",
    "any_rebate_winner",
    "csb_grant_awardee",
    "wri_has_committed_esbs",
    "survey_2023_response",
    "survey_2024_response",
]


def clean_nces(values: Iterable[object]) -> pd.Series:
    text = pd.Series(values, copy=False).astype("string").str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    text = text.str.extract(r"(\d+)", expand=False)
    text = text.where(text.notna() & text.ne(""), pd.NA)
    text = text.str.zfill(7)
    return text.where(text.ne("0000000"), pd.NA)


def to_numeric(values: Iterable[object]) -> pd.Series:
    text = pd.Series(values, copy=False).astype("string").str.strip()
    text = text.str.replace(r"[$,%]", "", regex=True)
    text = text.str.replace(",", "", regex=False)
    text = text.where(text.ne(""), pd.NA)
    return pd.to_numeric(text, errors="coerce")


def yes_priority(values: Iterable[object]) -> pd.Series:
    text = pd.Series(values, copy=False).astype("string").str.strip().str.lower()
    return text.str.startswith("yes").fillna(False).astype(int)


def first_year_from_flags(df: pd.DataFrame, flag_year_pairs: list[tuple[str, int]]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for flag, year in flag_year_pairs:
        if flag not in df.columns:
            continue
        hit = df[flag].fillna(0).astype(int).eq(1) & out.isna()
        out.loc[hit] = year
    return out


def load_edge_points() -> pd.DataFrame:
    if not SHP_FILE.exists():
        raise FileNotFoundError(f"Missing EDGE shapefile: {SHP_FILE}")
    edge = gpd.read_file(
        SHP_FILE,
        columns=["GEOID", "NAME", "INTPTLAT", "INTPTLON"],
        ignore_geometry=True,
    )
    edge = edge.rename(columns={"NAME": "edge_name"})
    edge["nces_id"] = clean_nces(edge["GEOID"])
    edge["edge_latitude"] = to_numeric(edge["INTPTLAT"])
    edge["edge_longitude"] = to_numeric(edge["INTPTLON"])
    edge = edge.dropna(subset=["nces_id", "edge_latitude", "edge_longitude"])
    edge = edge.sort_values("nces_id").drop_duplicates("nces_id", keep="first")
    return edge[["nces_id", "edge_name", "edge_latitude", "edge_longitude"]]


def build_r1_design_probabilities(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_excel(CSB_REBATES_XLSX)
    selected = selected[selected["Funding Year"] == 2022].copy()
    selected_std = pd.DataFrame(
        {
            "nces_id": clean_nces(selected["NCES District ID"]),
            "state": selected["School District State"].astype("string").str.strip().str.upper(),
            "priority": yes_priority(selected["Prioritization"]),
            "selected_actual": 1,
            "source": "selected",
        }
    )

    not_selected = pd.read_excel(CSBP_APPLICANTS_XLSX)
    not_selected = not_selected[
        (not_selected["Funding Year"] == 2022)
        & not_selected["Round"].astype("string").str.contains("R1", na=False)
    ].copy()
    not_selected_std = pd.DataFrame(
        {
            "nces_id": clean_nces(not_selected["NCES District ID"]),
            "state": not_selected["School District State"].astype("string").str.strip().str.upper(),
            "priority": yes_priority(not_selected["School District Prioritized"]),
            "selected_actual": 0,
            "source": "waitlist_or_rejected",
        }
    )

    apps = pd.concat([selected_std, not_selected_std], ignore_index=True)
    apps = apps.dropna(subset=["nces_id"]).copy()
    apps["state"] = apps["state"].str.slice(0, 2)

    district_apps = (
        apps.groupby("nces_id", as_index=False)
        .agg(
            r1_design_app_rows=("nces_id", "size"),
            r1_design_priority=("priority", "max"),
            r1_design_selected=("selected_actual", "max"),
            r1_design_sources=("source", lambda s: "; ".join(sorted(set(map(str, s))))),
        )
    )

    rate_table = (
        district_apps.groupby("r1_design_priority", as_index=False)
        .agg(
            applicants=("nces_id", "size"),
            selected=("r1_design_selected", "sum"),
        )
    )
    rate_table["pi_r1_bh_priority"] = rate_table["selected"] / rate_table["applicants"]
    rate_map = dict(zip(rate_table["r1_design_priority"], rate_table["pi_r1_bh_priority"]))

    district_apps["pi_r1_bh_priority"] = district_apps["r1_design_priority"].map(rate_map).astype(float)
    district_apps["z_r1_recenter_bh"] = district_apps["r1_design_selected"] - district_apps["pi_r1_bh_priority"]

    design = base[["nces_id"]].merge(district_apps, on="nces_id", how="left")
    design["r1_design_app_rows"] = design["r1_design_app_rows"].fillna(0).astype(int)
    design["r1_design_priority"] = design["r1_design_priority"].fillna(0).astype(int)
    design["r1_design_selected"] = design["r1_design_selected"].fillna(0).astype(int)
    design["pi_r1_bh_priority"] = design["pi_r1_bh_priority"].fillna(0.0)
    design["z_r1_recenter_bh"] = design["z_r1_recenter_bh"].fillna(0.0)

    return design, rate_table


def build_geometry_coverage(base: pd.DataFrame, edge: pd.DataFrame) -> pd.DataFrame:
    geo = base.merge(edge, on="nces_id", how="left")
    geo["has_edge_shape"] = geo["edge_latitude"].notna() & geo["edge_longitude"].notna()
    geo["has_wri_point"] = geo["latitude"].notna() & geo["longitude"].notna()
    geo["geometry_source"] = np.select(
        [geo["has_edge_shape"], geo["has_wri_point"]],
        ["edge_shapefile", "wri_point_fallback"],
        default="no_geometry",
    )
    geo["spatial_eligible_edge"] = geo["has_edge_shape"].astype(int)
    geo["spatial_eligible_hybrid"] = geo["geometry_source"].ne("no_geometry").astype(int)
    geo["spatial_latitude"] = geo["edge_latitude"].combine_first(geo["latitude"])
    geo["spatial_longitude"] = geo["edge_longitude"].combine_first(geo["longitude"])
    geo["interesting_for_spatial_drop"] = geo[INTERESTING_FLAGS].fillna(0).sum(axis=1).gt(0).astype(int)
    geo = add_sample_flags(geo)
    return geo


def add_sample_flags(geo: pd.DataFrame) -> pd.DataFrame:
    lea_type = geo["lea_type_name"].fillna("").astype("string").str.strip()
    state = geo["state"].fillna("").astype("string").str.upper().str.strip()

    regular_public = lea_type.isin(REGULAR_PUBLIC_TYPES)
    charter = lea_type.isin(CHARTER_TYPES)
    specialized_public = lea_type.isin(SPECIALIZED_PUBLIC_TYPES)
    agency_like = lea_type.isin(AGENCY_LIKE_TYPES)
    nonlea = lea_type.str.startswith("Non-LEA", na=False)
    contiguous = state.ne("") & ~state.isin(NON_CONTIGUOUS_STATES)

    geo["focal_regular_public"] = regular_public.astype(int)
    geo["focal_charter"] = charter.astype(int)
    geo["focal_specialized_public"] = specialized_public.astype(int)
    geo["focal_public_district_like"] = (regular_public | charter).astype(int)
    geo["focal_agency_like"] = agency_like.astype(int)
    geo["focal_nonlea"] = nonlea.astype(int)
    geo["contiguous_us"] = contiguous.astype(int)
    geo["focal_unit_class"] = np.select(
        [regular_public, charter, specialized_public, agency_like, nonlea],
        ["regular_public", "charter", "specialized_public", "agency_like", "nonlea"],
        default="missing_or_other",
    )
    geo["main_estimation_sample"] = (
        regular_public
        & geo["spatial_eligible_edge"].eq(1)
        & geo["has_core_controls"].eq(1)
        & contiguous
    ).astype(int)
    return geo


def project_points(df: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x, y = transformer.transform(df["spatial_longitude"].to_numpy(), df["spatial_latitude"].to_numpy())
    out = df.copy()
    out["spatial_x"] = x
    out["spatial_y"] = y
    return out


def build_knn_matrix(coords: np.ndarray, k: int) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    neighbor_indices = indices[:, 1 : k + 1]
    neighbor_distances = distances[:, 1 : k + 1]

    n = coords.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = neighbor_indices.reshape(-1)
    vals = np.ones(n * k, dtype=float)
    matrix = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return matrix, neighbor_indices, neighbor_distances


def add_static_exposures(
    exposure: pd.DataFrame,
    ordered: pd.DataFrame,
    W: csr_matrix,
    k: int,
    prefix: str,
    variables: list[str],
) -> pd.DataFrame:
    out = exposure.copy()
    for var in variables:
        if var not in ordered.columns:
            continue
        short = EXPOSURE_SHORT_NAMES.get(var, var[:12])
        x = to_numeric(ordered[var]).fillna(0).to_numpy(dtype=float)
        lag = np.asarray(W.dot(x)).reshape(-1)
        out[f"{prefix}{k}_{short}_n"] = lag
        out[f"{prefix}{k}_{short}_s"] = lag / float(k)
    return out


def build_dynamic_exposures(
    ordered: pd.DataFrame,
    W_by_k: dict[int, csr_matrix],
    years: list[int],
    prefix: str,
) -> pd.DataFrame:
    rows = []
    event_years = {name: to_numeric(ordered[col]).to_numpy(dtype=float) for name, col in DYNAMIC_EVENT_COLS.items()}
    r1_win = to_numeric(ordered["r1_rebate_winner"]).fillna(0).to_numpy(dtype=float)
    pi_bh = to_numeric(ordered["pi_r1_bh_priority"]).fillna(0).to_numpy(dtype=float)
    z_rc = to_numeric(ordered["z_r1_recenter_bh"]).fillna(0).to_numpy(dtype=float)

    for year in years:
        block = pd.DataFrame({"nces_id": ordered["nces_id"].to_numpy(), "year": year})
        for k, W in W_by_k.items():
            for name, event_year in event_years.items():
                adopted = np.nan_to_num(event_year, nan=9999.0) <= (year - 1)
                short = {"award": "award", "delivery": "deliv", "operating": "oper", "lottery_apply": "apply"}[name]
                block[f"{prefix}{k}_{short}_tm1_n"] = np.asarray(W.dot(adopted.astype(float))).reshape(-1)
            available = float(year >= 2023)
            block[f"{prefix}{k}_r1win_tm1_n"] = np.asarray(W.dot(r1_win)).reshape(-1) * available
            block[f"{prefix}{k}_r1expbh_tm1_n"] = np.asarray(W.dot(pi_bh)).reshape(-1) * available
            block[f"{prefix}{k}_r1rcbh_tm1_n"] = np.asarray(W.dot(z_rc)).reshape(-1) * available
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def write_dta(df: pd.DataFrame, path: Path) -> None:
    core_cols = [
        "nces_id",
        "district_name",
        "state",
        "lea_type_name",
        "focal_unit_class",
        "geometry_source",
        "district_panel_id",
        "year",
        "year_index",
        "year_since_r1",
        "year_since_r3",
        "post_r1",
        "post_r3",
        "y_first_award",
        "y_first_delivery",
        "y_first_operating",
        "y_first_lottery_apply",
        "y_first_rebate_win",
        "y_first_grant_award",
        "risk_first_award",
        "risk_first_delivery",
        "risk_first_operating",
        "risk_lottery_apply",
        "risk_rebate_win",
        "risk_grant_award",
        "risk_lottery_apply_window",
        "pre_panel_adopter",
        "pre_r1_adopter",
        "not_pre_r1_adopter",
        "own_r1_win_post",
        "own_r3_win_post",
        "own_any_rebate_win_post",
        "exclude_own_r1_winner",
        "exclude_own_rebate_winner",
        "spatial_eligible_edge",
        "spatial_eligible_hybrid",
        "interesting_for_spatial_drop",
        "focal_regular_public",
        "focal_charter",
        "focal_specialized_public",
        "focal_public_district_like",
        "focal_agency_like",
        "focal_nonlea",
        "contiguous_us",
        "main_estimation_sample",
        "hybrid_w6_nearest_mi",
        "hybrid_w6_max_mi",
        "hybrid_isolated_near50",
        "hybrid_isolated_k6_50",
        "edge_w6_nearest_mi",
        "edge_w6_max_mi",
        "edge_isolated_near50",
        "edge_isolated_k6_50",
        "main_noisol_edge_k6_50",
        "has_core_controls",
        "has_core_controls_geo",
        "has_state",
        "students",
        "schools",
        "total_buses",
        "median_income",
        "poverty_rate",
        "pct_white_alone",
        "low_income_200pct",
        "pm25",
        "pct_dem_2020",
        "priority_2022_csbp",
        "priority_2023_csbp",
        "r1_rebate_winner",
        "r3_rebate_winner",
        "r1_lottery_applicant",
        "r3_lottery_applicant",
        "any_lottery_applicant",
        "any_rebate_winner",
        "csb_grant_awardee",
        "first_award_year",
        "first_delivery_year",
        "first_operating_year",
        "first_lottery_apply_year",
        "first_rebate_win_year",
        "first_grant_award_year",
        "r1_design_app_rows",
        "r1_design_priority",
        "r1_design_selected",
        "pi_r1_bh_priority",
        "z_r1_recenter_bh",
    ]
    exposure_cols = [
        col
        for col in df.columns
        if (col.startswith("w6_") or col.startswith("edge_w6_")) and not col.endswith("_s")
    ]
    keep_cols = []
    for col in core_cols + exposure_cols:
        if col in df.columns and len(col) <= 32 and col not in keep_cols:
            keep_cols.append(col)
    dta = df[keep_cols].copy()
    for col in dta.select_dtypes(include=["object", "string"]).columns:
        dta[col] = dta[col].astype("string").fillna("").str.slice(0, 80)
    dta.to_stata(path, write_index=False, version=118)


def summarize_geometry(geo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        geo.groupby("geometry_source", dropna=False)
        .agg(
            districts=("nces_id", "size"),
            r1_lottery_applicants=("r1_lottery_applicant", "sum"),
            r3_lottery_applicants=("r3_lottery_applicant", "sum"),
            rebate_winners=("any_rebate_winner", "sum"),
            grant_awardees=("csb_grant_awardee", "sum"),
            wri_committed_esb=("wri_has_committed_esbs", "sum"),
            interesting=("interesting_for_spatial_drop", "sum"),
            has_core_controls=("has_core_controls", "sum"),
        )
        .reset_index()
    )
    summary["share_of_base_pct"] = 100 * summary["districts"] / len(geo)

    unmatched = geo[geo["geometry_source"].ne("edge_shapefile")].copy()
    type_summary = (
        unmatched.groupby(["geometry_source", "lea_type_name"], dropna=False)
        .agg(
            districts=("nces_id", "size"),
            interesting=("interesting_for_spatial_drop", "sum"),
            rebate_winners=("any_rebate_winner", "sum"),
            lottery_applicants=("any_lottery_applicant", "sum"),
            wri_committed_esb=("wri_has_committed_esbs", "sum"),
        )
        .reset_index()
        .sort_values(["geometry_source", "districts"], ascending=[True, False])
    )

    interesting_unmatched = unmatched[unmatched["interesting_for_spatial_drop"].eq(1)].copy()
    keep = [
        "nces_id",
        "district_name",
        "state",
        "lea_type_name",
        "geometry_source",
        "latitude",
        "longitude",
        "r1_lottery_applicant",
        "r3_lottery_applicant",
        "any_rebate_winner",
        "csb_grant_awardee",
        "wri_has_committed_esbs",
        "first_award_year",
        "first_oem",
    ]
    return summary, type_summary, interesting_unmatched[keep]


def summarize_sample_flags(geo: pd.DataFrame) -> pd.DataFrame:
    return (
        geo.groupby(["geometry_source", "focal_unit_class"], dropna=False)
        .agg(
            districts=("nces_id", "size"),
            interesting=("interesting_for_spatial_drop", "sum"),
            r1_lottery_applicants=("r1_lottery_applicant", "sum"),
            r3_lottery_applicants=("r3_lottery_applicant", "sum"),
            rebate_winners=("any_rebate_winner", "sum"),
            grant_awardees=("csb_grant_awardee", "sum"),
            wri_committed_esb=("wri_has_committed_esbs", "sum"),
            has_core_controls=("has_core_controls", "sum"),
            main_estimation_sample=("main_estimation_sample", "sum"),
        )
        .reset_index()
        .sort_values(["geometry_source", "districts"], ascending=[True, False])
    )


def build_distance_flags(ordered: pd.DataFrame, neighbor_distances: np.ndarray, prefix: str) -> pd.DataFrame:
    distances_mi = neighbor_distances / METERS_PER_MILE
    out = ordered[["nces_id"]].copy()
    out[f"{prefix}_w{PRIMARY_K}_nearest_mi"] = distances_mi[:, 0]
    out[f"{prefix}_w{PRIMARY_K}_max_mi"] = distances_mi[:, -1]
    out[f"{prefix}_isolated_near50"] = out[f"{prefix}_w{PRIMARY_K}_nearest_mi"].gt(50).astype(int)
    out[f"{prefix}_isolated_k{PRIMARY_K}_50"] = out[f"{prefix}_w{PRIMARY_K}_max_mi"].gt(50).astype(int)
    return out


def write_neighbor_audit(
    ordered: pd.DataFrame,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    path: Path,
) -> None:
    focal_ids = ordered["nces_id"].to_numpy()
    neighbor_ids = focal_ids[neighbor_indices]
    rows = []
    for rank in range(neighbor_indices.shape[1]):
        rows.append(
            pd.DataFrame(
                {
                    "nces_id": focal_ids,
                    "neighbor_rank": rank + 1,
                    "neighbor_nces_id": neighbor_ids[:, rank],
                    "distance_miles": neighbor_distances[:, rank] / METERS_PER_MILE,
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


def main() -> None:
    if not BASE_FILE.exists():
        raise FileNotFoundError(f"Missing {BASE_FILE}. Run 01_build_analysis_dataset.py first.")
    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Missing {PANEL_FILE}. Run 02_build_panel_dataset.py first.")

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(BASE_FILE, dtype={"nces_id": "string"}, low_memory=False)
    panel = pd.read_csv(PANEL_FILE, dtype={"nces_id": "string"}, low_memory=False)
    edge = load_edge_points()
    design, design_rates = build_r1_design_probabilities(base)

    base = base.merge(design, on="nces_id", how="left")
    base["first_lottery_apply_year"] = first_year_from_flags(
        base,
        [("r1_lottery_applicant", 2022), ("r3_lottery_applicant", 2023)],
    )
    base["has_core_controls"] = base[
        ["students", "median_income", "poverty_rate", "pct_white_alone", "pm25", "pct_dem_2020"]
    ].notna().all(axis=1).astype(int)

    geo = build_geometry_coverage(base, edge)
    geometry_summary, unmatched_type_summary, interesting_unmatched = summarize_geometry(geo)
    geometry_summary.to_csv(AUDIT_DIR / "spatial_geometry_source_summary.csv", index=False)
    unmatched_type_summary.to_csv(AUDIT_DIR / "spatial_unmatched_lea_type_summary.csv", index=False)
    interesting_unmatched.to_csv(AUDIT_DIR / "spatial_unmatched_interesting_leas.csv", index=False)
    summarize_sample_flags(geo).to_csv(AUDIT_DIR / "spatial_unit_flag_summary.csv", index=False)
    design_rates.to_csv(AUDIT_DIR / "r1_design_bh_priority_rates.csv", index=False)

    geo_keep = [
        "nces_id",
        "district_name",
        "state",
        "lea_type_name",
        "focal_unit_class",
        "geometry_source",
        "spatial_eligible_edge",
        "spatial_eligible_hybrid",
        "focal_regular_public",
        "focal_charter",
        "focal_specialized_public",
        "focal_public_district_like",
        "focal_agency_like",
        "focal_nonlea",
        "contiguous_us",
        "main_estimation_sample",
        "spatial_latitude",
        "spatial_longitude",
        "edge_name",
        "interesting_for_spatial_drop",
        "has_core_controls",
        "r1_design_app_rows",
        "r1_design_priority",
        "r1_design_selected",
        "pi_r1_bh_priority",
        "z_r1_recenter_bh",
    ]
    geo[geo_keep].to_csv(AUDIT_DIR / "spatial_geometry_coverage.csv", index=False)

    hybrid = project_points(geo[geo["spatial_eligible_hybrid"].eq(1)].copy())
    hybrid = hybrid.sort_values("nces_id").reset_index(drop=True)
    coords = hybrid[["spatial_x", "spatial_y"]].to_numpy(dtype=float)
    hybrid.to_csv(SPATIAL_ORDER_FILE, index=False)

    exposure = hybrid[["nces_id"]].copy()
    W_by_k: dict[int, csr_matrix] = {}
    primary_neighbors = None
    primary_distances = None
    hybrid_distance_flags = pd.DataFrame({"nces_id": hybrid["nces_id"]})
    for k in K_VALUES:
        W, neighbors, distances = build_knn_matrix(coords, k)
        W_by_k[k] = W
        save_npz(CLEANED_DIR / f"knn_weights_hybrid_k{k}.npz", W)
        exposure = add_static_exposures(exposure, hybrid, W, k, "w", STATIC_EXPOSURE_VARS)
        exposure = add_static_exposures(
            exposure,
            hybrid,
            W,
            k,
            "w",
            ["pi_r1_bh_priority", "z_r1_recenter_bh", "r1_design_priority", "r1_design_app_rows"],
        )
        if k == PRIMARY_K:
            primary_neighbors = neighbors
            primary_distances = distances
    if primary_neighbors is not None and primary_distances is not None:
        write_neighbor_audit(hybrid, primary_neighbors, primary_distances, AUDIT_DIR / "spatial_knn_neighbors_hybrid_k6.csv")
        hybrid_distance_flags = build_distance_flags(hybrid, primary_distances, "hybrid")

    years = sorted(panel["year"].unique())
    dynamic = build_dynamic_exposures(hybrid, W_by_k, years, "w")

    edge_only = project_points(geo[geo["spatial_eligible_edge"].eq(1)].copy())
    edge_only = edge_only.sort_values("nces_id").reset_index(drop=True)
    edge_only.to_csv(EDGE_ORDER_FILE, index=False)
    edge_coords = edge_only[["spatial_x", "spatial_y"]].to_numpy(dtype=float)
    W_edge, edge_neighbors, edge_distances = build_knn_matrix(edge_coords, PRIMARY_K)
    save_npz(CLEANED_DIR / f"knn_weights_edge_k{PRIMARY_K}.npz", W_edge)
    edge_exposure = edge_only[["nces_id"]].copy()
    edge_exposure = add_static_exposures(edge_exposure, edge_only, W_edge, PRIMARY_K, "edge_w", STATIC_EXPOSURE_VARS)
    edge_exposure = add_static_exposures(
        edge_exposure,
        edge_only,
        W_edge,
        PRIMARY_K,
        "edge_w",
        ["pi_r1_bh_priority", "z_r1_recenter_bh", "r1_design_priority", "r1_design_app_rows"],
    )
    edge_dynamic = build_dynamic_exposures(edge_only, {PRIMARY_K: W_edge}, years, "edge_w")
    write_neighbor_audit(edge_only, edge_neighbors, edge_distances, AUDIT_DIR / "spatial_knn_neighbors_edge_k6.csv")
    edge_distance_flags = build_distance_flags(edge_only, edge_distances, "edge")

    panel_out = panel.merge(
        geo[geo_keep],
        on="nces_id",
        how="left",
        suffixes=("", "_geo"),
    )
    panel_out = panel_out.merge(exposure, on="nces_id", how="left", suffixes=("", "_hybrid"))
    panel_out = panel_out.merge(dynamic, on=["nces_id", "year"], how="left")
    panel_out = panel_out.merge(edge_exposure, on="nces_id", how="left")
    panel_out = panel_out.merge(edge_dynamic, on=["nces_id", "year"], how="left")
    panel_out = panel_out.merge(hybrid_distance_flags, on="nces_id", how="left")
    panel_out = panel_out.merge(edge_distance_flags, on="nces_id", how="left")
    panel_out["main_noisol_edge_k6_50"] = (
        panel_out["main_estimation_sample"].eq(1) & panel_out["edge_isolated_k6_50"].fillna(1).eq(0)
    ).astype(int)

    if "w6_r1win_n" in panel_out.columns:
        panel_out["w6_r1win_tm1_chk"] = panel_out["w6_r1win_n"] * panel_out["year"].ge(2023).astype(int)

    panel_out.to_csv(PANEL_SPATIAL_CSV, index=False)
    write_dta(panel_out, PANEL_SPATIAL_DTA)

    panel_stage_rows = []
    for label, mask in [
        ("all_panel_rows", pd.Series(True, index=panel_out.index)),
        ("edge_shapefile_rows", panel_out["spatial_eligible_edge"].eq(1)),
        ("hybrid_geometry_rows", panel_out["spatial_eligible_hybrid"].eq(1)),
        ("no_geometry_rows", panel_out["spatial_eligible_hybrid"].ne(1)),
        ("hybrid_with_core_controls", panel_out["spatial_eligible_hybrid"].eq(1) & panel_out["has_core_controls_geo"].eq(1) if "has_core_controls_geo" in panel_out.columns else panel_out["spatial_eligible_hybrid"].eq(1) & panel_out["has_core_controls"].eq(1)),
        ("edge_with_core_controls", panel_out["spatial_eligible_edge"].eq(1) & panel_out["has_core_controls_geo"].eq(1) if "has_core_controls_geo" in panel_out.columns else panel_out["spatial_eligible_edge"].eq(1) & panel_out["has_core_controls"].eq(1)),
        ("main_regular_edge_controls_contig", panel_out["main_estimation_sample"].eq(1)),
        ("main_drop_edge_k6_gt50mi", panel_out["main_noisol_edge_k6_50"].eq(1)),
    ]:
        sub = panel_out.loc[mask]
        panel_stage_rows.append(
            {
                "stage": label,
                "district_year_rows": len(sub),
                "districts": sub["nces_id"].nunique(),
                "first_award_events": int(sub["y_first_award"].sum()),
                "r1_lottery_applicant_districts": int(sub.groupby("nces_id")["r1_lottery_applicant"].max().sum()),
                "r3_lottery_applicant_districts": int(sub.groupby("nces_id")["r3_lottery_applicant"].max().sum()),
            }
        )
    pd.DataFrame(panel_stage_rows).to_csv(AUDIT_DIR / "spatial_panel_sample_stages.csv", index=False)

    print(f"Saved {PANEL_SPATIAL_CSV} with {len(panel_out):,} district-years")
    print(f"Hybrid spatial districts: {len(hybrid):,}; EDGE-only districts: {len(edge_only):,}; no geometry: {int(geo['spatial_eligible_hybrid'].eq(0).sum()):,}")
    print(f"Saved spatial diagnostics in {AUDIT_DIR}")


if __name__ == "__main__":
    main()



