"""Build the annual hazard-panel dataset.

This script consumes `analysis_district_base.csv` from step 01 and expands it to
an annual district-year panel. It defines event timing and at-risk indicators for
first ESB award, delivery, operation, lottery application, and rebate wins.

Spatial and design-based neighbor exposure variables are intentionally left to a
later preparation step; this script creates the panel surface they will merge on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPT_DIR))

from config import AUDIT_DIR, CLEANED_DIR  # noqa: E402

PANEL_START_YEAR = 2018
PANEL_END_YEAR = 2024
BASE_FILE = CLEANED_DIR / "analysis_district_base.csv"
PANEL_CSV = CLEANED_DIR / "analysis_hazard_panel.csv"
PANEL_DTA = CLEANED_DIR / "analysis_hazard_panel.dta"
DIAGNOSTICS_FILE = AUDIT_DIR / "hazard_panel_build_diagnostics.csv"
YEAR_SUMMARY_FILE = AUDIT_DIR / "hazard_panel_year_summary.csv"

DROP_FROM_PANEL = {
    "rebate_applicants",
    "rebate_statuses",
    "grant_grantees",
    "grant_statuses",
    "waitlist_reject_statuses",
    "waitlist_reject_applicants",
    "oem_variants",
    "dealer_variants",
}


def to_numeric(values: Iterable[object]) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def first_year_from_flags(df: pd.DataFrame, flag_year_pairs: list[tuple[str, int]]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for flag, year in flag_year_pairs:
        if flag not in df.columns:
            continue
        hit = df[flag].fillna(0).astype(int).eq(1) & out.isna()
        out.loc[hit] = year
    return out


def hazard_risk(panel: pd.DataFrame, event_year_col: str) -> pd.Series:
    event_year = to_numeric(panel[event_year_col])
    return (event_year.isna() | (panel["year"] <= event_year)).astype(int)


def event_in_year(panel: pd.DataFrame, event_year_col: str) -> pd.Series:
    event_year = to_numeric(panel[event_year_col])
    return (event_year.eq(panel["year"])).fillna(False).astype(int)


def write_dta(df: pd.DataFrame, path: Path) -> None:
    dta = df.copy()
    for col in dta.select_dtypes(include=["object", "string"]).columns:
        dta[col] = dta[col].astype("string").fillna("")
        too_long = dta[col].str.len() > 244
        if too_long.any():
            dta[col] = dta[col].str.slice(0, 244)
    dta.to_stata(path, write_index=False, version=118)


def build_panel(base: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    base = base.copy()
    base["district_panel_id"] = np.arange(1, len(base) + 1)

    base["first_lottery_apply_year"] = first_year_from_flags(
        base,
        [("r1_lottery_applicant", 2022), ("r3_lottery_applicant", 2023)],
    )
    base["first_rebate_win_year"] = first_year_from_flags(
        base,
        [("r1_rebate_winner", 2022), ("r3_rebate_winner", 2023)],
    )

    if "csb_grant_first_year" in base.columns:
        base["first_grant_award_year"] = to_numeric(base["csb_grant_first_year"])
    else:
        base["first_grant_award_year"] = np.nan

    keep_cols = [col for col in base.columns if col not in DROP_FROM_PANEL]
    base = base[keep_cols].copy()

    panel_parts = []
    for year in range(start_year, end_year + 1):
        tmp = base.copy()
        tmp["year"] = year
        panel_parts.append(tmp)
    panel = pd.concat(panel_parts, ignore_index=True)

    panel["panel_start_year"] = start_year
    panel["panel_end_year"] = end_year
    panel["year_index"] = panel["year"] - start_year
    panel["year_since_r1"] = panel["year"] - 2022
    panel["year_since_r3"] = panel["year"] - 2023
    panel["post_r1"] = (panel["year"] >= 2023).astype(int)
    panel["post_r3"] = (panel["year"] >= 2024).astype(int)

    panel["y_first_award"] = event_in_year(panel, "first_award_year")
    panel["y_first_delivery"] = event_in_year(panel, "first_delivery_year")
    panel["y_first_operating"] = event_in_year(panel, "first_operating_year")
    panel["y_first_lottery_apply"] = event_in_year(panel, "first_lottery_apply_year")
    panel["y_first_rebate_win"] = event_in_year(panel, "first_rebate_win_year")
    panel["y_first_grant_award"] = event_in_year(panel, "first_grant_award_year")

    panel["risk_first_award"] = hazard_risk(panel, "first_award_year")
    panel["risk_first_delivery"] = hazard_risk(panel, "first_delivery_year")
    panel["risk_first_operating"] = hazard_risk(panel, "first_operating_year")
    panel["risk_lottery_apply"] = hazard_risk(panel, "first_lottery_apply_year")
    panel["risk_rebate_win"] = hazard_risk(panel, "first_rebate_win_year")
    panel["risk_grant_award"] = hazard_risk(panel, "first_grant_award_year")

    panel["lottery_application_window"] = panel["year"].isin([2022, 2023]).astype(int)
    panel["risk_lottery_apply_window"] = (
        panel["risk_lottery_apply"].eq(1) & panel["lottery_application_window"].eq(1)
    ).astype(int)

    panel["pre_panel_adopter"] = (to_numeric(panel["first_award_year"]) < start_year).fillna(False).astype(int)
    panel["pre_r1_adopter"] = (to_numeric(panel["first_award_year"]) < 2022).fillna(False).astype(int)
    panel["not_pre_r1_adopter"] = (panel["pre_r1_adopter"] == 0).astype(int)

    panel["own_r1_win_post"] = (
        panel.get("r1_rebate_winner", 0).fillna(0).astype(int).eq(1) & panel["year"].ge(2022)
    ).astype(int)
    panel["own_r3_win_post"] = (
        panel.get("r3_rebate_winner", 0).fillna(0).astype(int).eq(1) & panel["year"].ge(2023)
    ).astype(int)
    panel["own_any_rebate_win_post"] = (
        panel["own_r1_win_post"].eq(1) | panel["own_r3_win_post"].eq(1)
    ).astype(int)

    panel["exclude_own_r1_winner"] = panel.get("r1_rebate_winner", 0).fillna(0).astype(int)
    panel["exclude_own_rebate_winner"] = panel.get("any_rebate_winner", 0).fillna(0).astype(int)

    panel["has_coordinates"] = (panel["latitude"].notna() & panel["longitude"].notna()).astype(int)
    control_cols = ["students", "median_income", "poverty_rate", "pct_white_alone", "pm25", "pct_dem_2020"]
    existing_controls = [col for col in control_cols if col in panel.columns]
    panel["has_core_controls"] = panel[existing_controls].notna().all(axis=1).astype(int)
    panel["has_state"] = panel["state"].notna().astype(int)

    panel = panel.sort_values(["nces_id", "year"]).reset_index(drop=True)
    return panel


def write_diagnostics(panel: pd.DataFrame, base: pd.DataFrame, start_year: int, end_year: int) -> None:
    metrics = [
        ("panel_start_year", start_year),
        ("panel_end_year", end_year),
        ("districts_in_base", len(base)),
        ("districts_in_panel", panel["nces_id"].nunique()),
        ("district_year_rows", len(panel)),
        ("pre_panel_adopters", int(base["first_award_year"].lt(start_year).fillna(False).sum())),
        ("pre_r1_adopters", int(base["first_award_year"].lt(2022).fillna(False).sum())),
        ("first_award_events_in_panel", int(panel["y_first_award"].sum())),
        ("first_delivery_events_in_panel", int(panel["y_first_delivery"].sum())),
        ("first_operating_events_in_panel", int(panel["y_first_operating"].sum())),
        ("first_lottery_apply_events", int(panel["y_first_lottery_apply"].sum())),
        ("first_rebate_win_events", int(panel["y_first_rebate_win"].sum())),
        ("first_grant_award_events", int(panel["y_first_grant_award"].sum())),
        ("risk_first_award_rows", int(panel["risk_first_award"].sum())),
        ("risk_lottery_apply_window_rows", int(panel["risk_lottery_apply_window"].sum())),
        ("districts_with_coordinates", int(base["latitude"].notna().fillna(False).sum())),
    ]
    pd.DataFrame(metrics, columns=["metric", "value"]).to_csv(DIAGNOSTICS_FILE, index=False)

    year_summary = (
        panel.groupby("year", as_index=False)
        .agg(
            district_years=("nces_id", "size"),
            risk_first_award_rows=("risk_first_award", "sum"),
            y_first_award=("y_first_award", "sum"),
            y_first_delivery=("y_first_delivery", "sum"),
            y_first_operating=("y_first_operating", "sum"),
            risk_lottery_apply_window_rows=("risk_lottery_apply_window", "sum"),
            y_first_lottery_apply=("y_first_lottery_apply", "sum"),
            y_first_rebate_win=("y_first_rebate_win", "sum"),
            y_first_grant_award=("y_first_grant_award", "sum"),
        )
    )
    year_summary.to_csv(YEAR_SUMMARY_FILE, index=False)


def main() -> None:
    if not BASE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {BASE_FILE}. Run 2_Scripts/1_Preparation/01_build_analysis_dataset.py first."
        )

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(BASE_FILE, dtype={"nces_id": "string"}, low_memory=False)
    base = base.sort_values("nces_id").drop_duplicates("nces_id", keep="first")

    panel = build_panel(base, PANEL_START_YEAR, PANEL_END_YEAR)
    panel.to_csv(PANEL_CSV, index=False)
    write_dta(panel, PANEL_DTA)
    write_diagnostics(panel, base, PANEL_START_YEAR, PANEL_END_YEAR)

    print(f"Saved {PANEL_CSV} with {len(panel):,} district-years")
    print(f"Saved {PANEL_DTA}")
    print(f"Saved diagnostics to {DIAGNOSTICS_FILE} and {YEAR_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
