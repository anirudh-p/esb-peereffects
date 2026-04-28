"""Build the district-level base analysis dataset.

This first preparation step does not build the hazard panel. It creates the
foundation that every later specification needs:

- raw file and raw table inventories;
- a long source-record file with standardized NCES IDs;
- a one-row-per-district identifier crosswalk;
- a one-row-per-district base analysis file with core controls, CSBP flags,
  survey coverage, and first ESB timing/vendor fields.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPT_DIR))

from config import AUDIT_DIR, CLEANED_DIR, RAW_DIR  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated*",
    category=FutureWarning,
)

WRI_DIR = RAW_DIR / "WRI"
POLITICAL_FILE = RAW_DIR / "Political" / "dataverse_files" / "countypres_2000-2024.csv"
WRI_DISTRICT_CSV = WRI_DIR / "wri_data.csv"
WRI_ADOPTION_XLSX = WRI_DIR / "ESB_adoption_dataset_v9_update_june_2025.xlsx"
CSB_REBATES_XLSX = WRI_DIR / "CSB_Rebates.xlsx"
CSB_GRANTS_XLSX = WRI_DIR / "CSB_Grants.xlsx"
CSBP_APPLICANTS_XLSX = WRI_DIR / "CSBP Applicants waitlisted and rejected_11.18.25.xlsx"
SURVEY_2023_XLSX = WRI_DIR / "2023 WRI School District Study Final Data.xlsx"
SURVEY_2024_XLSX = WRI_DIR / "2024 WRI School District Study Final Data.xlsx"

SOURCE_PRIORITY = {
    "wri_district_csv": 1,
    "wri_adoption_district_sheet": 2,
    "csb_rebates": 3,
    "csb_grants": 4,
    "csbp_waitlist_rejected": 5,
    "wri_bus_sheet": 6,
    "survey_2024": 7,
    "survey_2023": 8,
}


def clean_nces(values: Iterable[object]) -> pd.Series:
    """Return 7-digit NCES district IDs, preserving missing/invalid values."""
    text = pd.Series(values, copy=False).astype("string").str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    text = text.str.extract(r"(\d+)", expand=False)
    text = text.where(text.notna() & text.ne(""), pd.NA)
    text = text.str.zfill(7)
    return text.where(text.ne("0000000"), pd.NA)


def clean_text(values: Iterable[object]) -> pd.Series:
    out = pd.Series(values, copy=False).astype("string").str.strip()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out.where(out.ne(""), pd.NA)


def normalize_name(values: Iterable[object]) -> pd.Series:
    out = clean_text(values).str.upper()
    out = out.str.replace(r"[^A-Z0-9]+", " ", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out.where(out.ne(""), pd.NA)


def to_numeric(values: Iterable[object]) -> pd.Series:
    text = pd.Series(values, copy=False).astype("string").str.strip()
    text = text.str.replace(r"[$,%]", "", regex=True)
    text = text.str.replace(",", "", regex=False)
    text = text.where(text.ne(""), pd.NA)
    return pd.to_numeric(text, errors="coerce")


def yes_no_flag(values: Iterable[object]) -> pd.Series:
    text = pd.Series(values, copy=False).astype("string").str.strip().str.lower()
    return text.map({"yes": 1, "no": 0, "y": 1, "n": 0}).astype("Int64")


def parse_year(values: Iterable[object]) -> pd.Series:
    text = pd.Series(values, copy=False).astype("string")
    year = text.str.extract(r"(19\d{2}|20\d{2})", expand=False)
    return pd.to_numeric(year, errors="coerce")


def parse_quarter_index(values: Iterable[object]) -> pd.Series:
    text = pd.Series(values, copy=False).astype("string").str.upper()
    year = parse_year(text)
    quarter = pd.to_numeric(text.str.extract(r"Q([1-4])", expand=False), errors="coerce")
    return year * 4 + quarter


def first_nonmissing(values: Iterable[object]):
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return pd.NA


def join_unique(values: Iterable[object], limit: int = 12) -> str:
    seen: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        item = str(value).strip()
        if item and item not in seen:
            seen.append(item)
    if len(seen) > limit:
        return "; ".join(seen[:limit]) + f"; ... (+{len(seen) - limit})"
    return "; ".join(seen)


def safe_get(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(pd.NA, index=df.index)


def add_source_records(
    records: list[pd.DataFrame],
    coverage: list[dict[str, object]],
    df: pd.DataFrame,
    *,
    source: str,
    raw_file: Path,
    raw_sheet: str,
    nces_col: str,
    name_col: str,
    state_col: str | None = None,
    city_col: str | None = None,
) -> None:
    out = pd.DataFrame(
        {
            "source": source,
            "raw_file": raw_file.name,
            "raw_sheet": raw_sheet,
            "raw_row": np.arange(1, len(df) + 1),
            "nces_id": clean_nces(safe_get(df, nces_col)),
            "district_name": clean_text(safe_get(df, name_col)),
            "state": clean_text(safe_get(df, state_col)) if state_col else pd.Series(pd.NA, index=df.index),
            "city": clean_text(safe_get(df, city_col)) if city_col else pd.Series(pd.NA, index=df.index),
        }
    )
    out["district_name_key"] = normalize_name(out["district_name"])
    records.append(out)

    valid = out["nces_id"].notna()
    coverage.append(
        {
            "source": source,
            "raw_file": raw_file.name,
            "raw_sheet": raw_sheet,
            "rows": len(out),
            "valid_nces_rows": int(valid.sum()),
            "missing_nces_rows": int((~valid).sum()),
            "unique_nces": int(out.loc[valid, "nces_id"].nunique()),
            "duplicate_nces_rows": int(out.loc[valid, "nces_id"].duplicated().sum()),
        }
    )


def inventory_raw_files() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "source_group": path.relative_to(RAW_DIR).parts[0],
                    "relative_path": path.relative_to(RAW_DIR).as_posix(),
                    "extension": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                    "modified_time": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
                }
            )
    return pd.DataFrame(rows)


def inventory_raw_tables() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        rel = path.relative_to(RAW_DIR).as_posix()
        try:
            if suffix == ".csv":
                df = pd.read_csv(path, nrows=0, low_memory=False)
                rows.append(
                    {
                        "relative_path": rel,
                        "sheet": "",
                        "n_columns": len(df.columns),
                        "columns": " | ".join(map(str, df.columns)),
                    }
                )
            elif suffix in {".xlsx", ".xls"}:
                workbook = pd.ExcelFile(path)
                for sheet in workbook.sheet_names:
                    df = pd.read_excel(path, sheet_name=sheet, nrows=0)
                    rows.append(
                        {
                            "relative_path": rel,
                            "sheet": sheet,
                            "n_columns": len(df.columns),
                            "columns": " | ".join(map(str, df.columns)),
                        }
                    )
        except Exception as exc:  # keep inventory robust to odd workbooks
            rows.append(
                {
                    "relative_path": rel,
                    "sheet": "",
                    "n_columns": pd.NA,
                    "columns": f"READ_ERROR: {exc}",
                }
            )
    return pd.DataFrame(rows)


def build_source_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []

    wri = pd.read_csv(WRI_DISTRICT_CSV, low_memory=False)
    add_source_records(
        records,
        coverage,
        wri,
        source="wri_district_csv",
        raw_file=WRI_DISTRICT_CSV,
        raw_sheet="",
        nces_col="1c. LEA ID",
        name_col="1b. Local Education Agency (LEA) or entity name",
        state_col="1g. State",
        city_col="1f. City",
    )

    district_sheet = pd.read_excel(WRI_ADOPTION_XLSX, sheet_name="1. District-level data")
    add_source_records(
        records,
        coverage,
        district_sheet,
        source="wri_adoption_district_sheet",
        raw_file=WRI_ADOPTION_XLSX,
        raw_sheet="1. District-level data",
        nces_col="1c. LEA ID",
        name_col="1b. Local Education Agency (LEA) or entity name",
        state_col="1g. State",
        city_col="1f. City",
    )

    buses = pd.read_excel(WRI_ADOPTION_XLSX, sheet_name="2. Bus-level data")
    add_source_records(
        records,
        coverage,
        buses,
        source="wri_bus_sheet",
        raw_file=WRI_ADOPTION_XLSX,
        raw_sheet="2. Bus-level data",
        nces_col="1c. LEA ID",
        name_col="1b. LEA or entity name",
        state_col="1a. State",
        city_col="1f. City",
    )

    rebates = pd.read_excel(CSB_REBATES_XLSX)
    add_source_records(
        records,
        coverage,
        rebates,
        source="csb_rebates",
        raw_file=CSB_REBATES_XLSX,
        raw_sheet="Sheet1",
        nces_col="NCES District ID",
        name_col="School District Name",
        state_col="School District State",
    )

    grants = pd.read_excel(CSB_GRANTS_XLSX)
    add_source_records(
        records,
        coverage,
        grants,
        source="csb_grants",
        raw_file=CSB_GRANTS_XLSX,
        raw_sheet="Sheet1",
        nces_col="NCES District ID",
        name_col="Grants School District Name",
        state_col="School District State",
    )

    applicants = pd.read_excel(CSBP_APPLICANTS_XLSX)
    add_source_records(
        records,
        coverage,
        applicants,
        source="csbp_waitlist_rejected",
        raw_file=CSBP_APPLICANTS_XLSX,
        raw_sheet="Sheet 1",
        nces_col="NCES District ID",
        name_col="School District Name",
        state_col="School District State",
    )

    survey_2023 = pd.read_excel(SURVEY_2023_XLSX, sheet_name="SchoolDistrictStudyWODuplicates")
    add_source_records(
        records,
        coverage,
        survey_2023,
        source="survey_2023",
        raw_file=SURVEY_2023_XLSX,
        raw_sheet="SchoolDistrictStudyWODuplicates",
        nces_col="LEAID",
        name_col="School District:",
        state_col="State Abb",
        city_col="City:",
    )

    survey_2024 = pd.read_excel(SURVEY_2024_XLSX, sheet_name="2024WRISchoolDistrictAnalysis")
    add_source_records(
        records,
        coverage,
        survey_2024,
        source="survey_2024",
        raw_file=SURVEY_2024_XLSX,
        raw_sheet="2024WRISchoolDistrictAnalysis",
        nces_col="LEAID (as values)",
        name_col="SchoolDistrict_OE",
        state_col="State_Clean",
        city_col="City_OE",
    )

    return pd.concat(records, ignore_index=True), pd.DataFrame(coverage)


def build_crosswalk(source_records: pd.DataFrame) -> pd.DataFrame:
    valid = source_records[source_records["nces_id"].notna()].copy()
    valid["source_priority"] = valid["source"].map(SOURCE_PRIORITY).fillna(99)
    valid = valid.sort_values(["nces_id", "source_priority", "raw_file", "raw_row"])

    canonical = (
        valid.groupby("nces_id", as_index=False)
        .agg(
            district_name=("district_name", first_nonmissing),
            state=("state", first_nonmissing),
            city=("city", first_nonmissing),
            district_name_variants=("district_name", join_unique),
            state_variants=("state", join_unique),
            sources=("source", join_unique),
            source_record_count=("source", "size"),
            source_count=("source", "nunique"),
        )
    )

    flags = pd.crosstab(valid["nces_id"], valid["source"])
    flags = (flags > 0).astype(int).reset_index()
    flags = flags.rename(columns={col: f"in_{col}" for col in flags.columns if col != "nces_id"})

    crosswalk = canonical.merge(flags, on="nces_id", how="left")
    for col in [c for c in crosswalk.columns if c.startswith("in_")]:
        crosswalk[col] = crosswalk[col].fillna(0).astype(int)
    return crosswalk.sort_values("nces_id")


def build_wri_base() -> pd.DataFrame:
    df = pd.read_csv(WRI_DISTRICT_CSV, low_memory=False)
    base = pd.DataFrame(
        {
            "nces_id": clean_nces(df["1c. LEA ID"]),
            "district_name": clean_text(df["1b. Local Education Agency (LEA) or entity name"]),
            "state_name": clean_text(df["1a. State"]),
            "state": clean_text(df["1g. State"]),
            "city": clean_text(df["1f. City"]),
            "zip_code": clean_text(df["1h. ZIP code"]),
            "lea_type_name": clean_text(df["1l. LEA type (name)"]),
            "locale_broad": clean_text(df["1p. Locale broad type (name)"]),
            "census_region": clean_text(df["1q. Census Region"]),
            "census_division": clean_text(df["1r. Census Division"]),
            "latitude": to_numeric(df["1s. Latitude"]),
            "longitude": to_numeric(df["1t. Longitude "]),
            "total_buses": to_numeric(df["2a. Total number of buses"]),
            "wri_has_committed_esbs": yes_no_flag(df["0a. Has committed ESBs?"]),
            "n_esbs_committed": to_numeric(df["3a. Number of ESBs committed "]),
            "n_esbs_deliv_oper": to_numeric(df["3b. Number of delivered or operating ESBs"]),
            "n_esbs_awarded": to_numeric(df["3c. Number of ESBs awarded"]),
            "n_esbs_ordered": to_numeric(df["3d. Number of ESBs ordered"]),
            "n_esbs_delivered": to_numeric(df["3e. Number of ESBs delivered"]),
            "n_esbs_operating": to_numeric(df["3f. Number of ESBs operating"]),
            "pct_fleet_electric": to_numeric(df["3i. Percent of fleet that is electric"]),
            "students": to_numeric(df["4b. Number of students in district"]),
            "schools": to_numeric(df["4c. Number of schools in district"]),
            "title1_share": to_numeric(df["4d. Percentage of schools in district that are Title I schoolwide eligible"]),
            "frpl_share": to_numeric(df["4e. Percentage of students in district eligible for free or reduced price lunch"]),
            "median_income": to_numeric(df["4f. Median household income"]),
            "poverty_rate": to_numeric(df["4g. Percent of population below the poverty level"]),
            "pct_white_alone": to_numeric(df["4h. Percent one race: White "]),
            "low_income_200pct": to_numeric(df["5d. Percent low-income (200% of federal poverty level)"]),
            "pm25": to_numeric(df["5f. PM2.5 concentration"]),
            "priority_2022_csbp": yes_no_flag(df["5o. EPA 2022 Clean School Bus Rebate Program prioritized school district?"]),
            "priority_2023_csbp": yes_no_flag(df["5p. EPA 2023 Clean School Bus Grant & Rebate Programs prioritized school district?"]),
            "wri_priority_outreach": yes_no_flag(df["5q. WRI Priority Outreach District (POD)?"]),
        }
    )
    base = base.dropna(subset=["nces_id"]).copy()
    base = base.sort_values("nces_id").drop_duplicates("nces_id", keep="first")
    return base


def build_csbp_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rebates = pd.read_excel(CSB_REBATES_XLSX)
    rebates["nces_id"] = clean_nces(rebates["NCES District ID"])
    rebates = rebates.dropna(subset=["nces_id"]).copy()
    rebates["funding_year"] = to_numeric(rebates["Funding Year"])
    for col in ["Number of Electric Buses", "Total Number of Buses", "Total Awarded"]:
        rebates[col] = to_numeric(rebates[col])
    rebates["r1_rebate_winner"] = (rebates["funding_year"] == 2022).astype(int)
    rebates["r3_rebate_winner"] = (rebates["funding_year"] == 2023).astype(int)
    rebate_features = (
        rebates.groupby("nces_id", as_index=False)
        .agg(
            csb_rebate_records=("nces_id", "size"),
            csb_rebate_first_year=("funding_year", "min"),
            csb_rebate_last_year=("funding_year", "max"),
            r1_rebate_winner=("r1_rebate_winner", "max"),
            r3_rebate_winner=("r3_rebate_winner", "max"),
            rebate_electric_buses=("Number of Electric Buses", "sum"),
            rebate_total_buses=("Total Number of Buses", "sum"),
            rebate_total_awarded=("Total Awarded", "sum"),
            rebate_applicants=("Applicant Organization Name", join_unique),
            rebate_statuses=("Project Status", join_unique),
        )
    )

    grants = pd.read_excel(CSB_GRANTS_XLSX)
    grants["nces_id"] = clean_nces(grants["NCES District ID"])
    grants = grants.dropna(subset=["nces_id"]).copy()
    grants["funding_year"] = to_numeric(grants["Funding Year"])
    for col in ["Electric Buses", "Total Buses"]:
        grants[col] = to_numeric(grants[col])
    grant_features = (
        grants.groupby("nces_id", as_index=False)
        .agg(
            csb_grant_records=("nces_id", "size"),
            csb_grant_first_year=("funding_year", "min"),
            grant_electric_buses=("Electric Buses", "sum"),
            grant_total_buses=("Total Buses", "sum"),
            grant_grantees=("Grantee Name", join_unique),
            grant_statuses=("Project Status", join_unique),
        )
    )
    grant_features["csb_grant_awardee"] = 1

    applicants = pd.read_excel(CSBP_APPLICANTS_XLSX)
    applicants["nces_id"] = clean_nces(applicants["NCES District ID"])
    applicants = applicants.dropna(subset=["nces_id"]).copy()
    applicants["round_text"] = applicants["Round"].astype("string")
    applicants["r1_waitlist_reject"] = applicants["round_text"].str.contains("R1", na=False).astype(int)
    applicants["r2_grant_reject"] = applicants["round_text"].str.contains("R2", na=False).astype(int)
    applicants["r3_waitlist_reject"] = applicants["round_text"].str.contains("R3", na=False).astype(int)
    for col in ["Number of Electric Buses", "Total Number of Buses"]:
        applicants[col] = to_numeric(applicants[col])
    applicant_features = (
        applicants.groupby("nces_id", as_index=False)
        .agg(
            csbp_waitlist_reject_records=("nces_id", "size"),
            r1_waitlist_reject=("r1_waitlist_reject", "max"),
            r2_grant_reject=("r2_grant_reject", "max"),
            r3_waitlist_reject=("r3_waitlist_reject", "max"),
            waitlist_reject_electric_buses=("Number of Electric Buses", "sum"),
            waitlist_reject_total_buses=("Total Number of Buses", "sum"),
            waitlist_reject_statuses=("Project Status", join_unique),
            waitlist_reject_applicants=("Applicant Organization Name", join_unique),
        )
    )

    return rebate_features, grant_features, applicant_features


def build_bus_features() -> pd.DataFrame:
    buses = pd.read_excel(WRI_ADOPTION_XLSX, sheet_name="2. Bus-level data")
    buses["nces_id"] = clean_nces(buses["1c. LEA ID"])
    buses = buses.dropna(subset=["nces_id"]).copy()
    buses["award_year"] = parse_year(buses["3p. Quarter awarded"])
    buses["award_q_index"] = parse_quarter_index(buses["3p. Quarter awarded"])
    buses["delivery_year"] = parse_year(buses["3r. Quarter delivered"])
    buses["delivery_q_index"] = parse_quarter_index(buses["3r. Quarter delivered"])
    buses["operating_year"] = parse_year(buses["3s. Quarter first operating"])
    buses["operating_q_index"] = parse_quarter_index(buses["3s. Quarter first operating"])
    buses = buses.sort_values(["nces_id", "award_q_index", "award_year"])

    features = (
        buses.groupby("nces_id", as_index=False)
        .agg(
            wri_bus_records=("nces_id", "size"),
            first_award_year=("award_year", "min"),
            first_award_q_index=("award_q_index", "min"),
            first_delivery_year=("delivery_year", "min"),
            first_delivery_q_index=("delivery_q_index", "min"),
            first_operating_year=("operating_year", "min"),
            first_operating_q_index=("operating_q_index", "min"),
            first_oem=("3t. Bus OEM", first_nonmissing),
            first_dealer=("3x. Dealer", first_nonmissing),
            first_charger_company=("3ac. Charging company", first_nonmissing),
            oem_variants=("3t. Bus OEM", join_unique),
            dealer_variants=("3x. Dealer", join_unique),
        )
    )
    features["wri_pre_2022_adopter"] = (features["first_award_year"] < 2022).astype("Int64")
    features["wri_awarded_2023_24"] = features["first_award_year"].between(2023, 2024).astype("Int64")
    return features


def build_survey_features() -> pd.DataFrame:
    s23 = pd.read_excel(SURVEY_2023_XLSX, sheet_name="SchoolDistrictStudyWODuplicates")
    s23["nces_id"] = clean_nces(s23["LEAID"])
    f23 = s23.dropna(subset=["nces_id"])[["nces_id"]].drop_duplicates()
    f23["survey_2023_response"] = 1

    s24 = pd.read_excel(SURVEY_2024_XLSX, sheet_name="2024WRISchoolDistrictAnalysis")
    s24["nces_id"] = clean_nces(s24["LEAID (as values)"])
    f24 = s24.dropna(subset=["nces_id"])[["nces_id"]].drop_duplicates()
    f24["survey_2024_response"] = 1

    return f23.merge(f24, on="nces_id", how="outer")


def build_political_features() -> pd.DataFrame:
    if not POLITICAL_FILE.exists():
        return pd.DataFrame(columns=["nces_id", "pct_dem_2020"])

    pol = pd.read_csv(POLITICAL_FILE, low_memory=False)
    pol20 = pol[(pol["year"] == 2020) & (pol["office"] == "US PRESIDENT")].copy()
    total = pol20.groupby("county_fips", as_index=False)["candidatevotes"].sum()
    total = total.rename(columns={"candidatevotes": "total_votes"})
    dem = pol20[pol20["party"] == "DEMOCRAT"].groupby("county_fips", as_index=False)["candidatevotes"].sum()
    dem = dem.rename(columns={"candidatevotes": "dem_votes"})
    county = total.merge(dem, on="county_fips", how="left")
    county["pct_dem_2020"] = county["dem_votes"] / county["total_votes"]

    xwalk = pd.read_excel(WRI_ADOPTION_XLSX, sheet_name="5. Counties")
    xwalk["nces_id"] = clean_nces(xwalk["1c. LEA ID"])
    xwalk["county_fips"] = to_numeric(xwalk["10b. County FIPS Code"])
    xwalk = xwalk.dropna(subset=["nces_id", "county_fips"])
    xwalk["county_fips"] = xwalk["county_fips"].astype(int)
    xwalk = xwalk.merge(county[["county_fips", "pct_dem_2020"]], on="county_fips", how="left")
    return xwalk.groupby("nces_id", as_index=False)["pct_dem_2020"].mean()


def merge_features(base: pd.DataFrame, feature_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    for frame in feature_frames:
        out = out.merge(frame, on="nces_id", how="left")

    zero_cols = [
        "r1_rebate_winner",
        "r3_rebate_winner",
        "csb_grant_awardee",
        "r1_waitlist_reject",
        "r2_grant_reject",
        "r3_waitlist_reject",
        "survey_2023_response",
        "survey_2024_response",
        "wri_pre_2022_adopter",
        "wri_awarded_2023_24",
    ]
    for col in zero_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)

    count_cols = [
        "csb_rebate_records",
        "csb_grant_records",
        "csbp_waitlist_reject_records",
        "wri_bus_records",
    ]
    for col in count_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)

    if {"r1_rebate_winner", "r1_waitlist_reject"}.issubset(out.columns):
        out["r1_lottery_applicant"] = ((out["r1_rebate_winner"] == 1) | (out["r1_waitlist_reject"] == 1)).astype(int)
    if {"r3_rebate_winner", "r3_waitlist_reject"}.issubset(out.columns):
        out["r3_lottery_applicant"] = ((out["r3_rebate_winner"] == 1) | (out["r3_waitlist_reject"] == 1)).astype(int)
    if {"r1_lottery_applicant", "r3_lottery_applicant"}.issubset(out.columns):
        out["any_lottery_applicant"] = ((out["r1_lottery_applicant"] == 1) | (out["r3_lottery_applicant"] == 1)).astype(int)
    if {"r1_rebate_winner", "r3_rebate_winner"}.issubset(out.columns):
        out["any_rebate_winner"] = ((out["r1_rebate_winner"] == 1) | (out["r3_rebate_winner"] == 1)).astype(int)

    return out.sort_values("nces_id")


def write_dta(df: pd.DataFrame, path: Path) -> None:
    dta = df.copy()
    for col in dta.select_dtypes(include=["object", "string"]).columns:
        dta[col] = dta[col].astype("string").fillna("")
        too_long = dta[col].str.len() > 244
        if too_long.any():
            dta[col] = dta[col].str.slice(0, 244)
    dta.to_stata(path, write_index=False, version=118)


def write_audit(base: pd.DataFrame, source_coverage: pd.DataFrame) -> None:
    summary_rows = [
        {"metric": "district_base_rows", "value": len(base)},
        {"metric": "district_base_unique_nces", "value": base["nces_id"].nunique()},
        {"metric": "wri_has_committed_esbs", "value": int(base["wri_has_committed_esbs"].fillna(0).sum())},
        {"metric": "r1_rebate_winners", "value": int(base.get("r1_rebate_winner", pd.Series(0, index=base.index)).sum())},
        {"metric": "r3_rebate_winners", "value": int(base.get("r3_rebate_winner", pd.Series(0, index=base.index)).sum())},
        {"metric": "csb_grant_awardees", "value": int(base.get("csb_grant_awardee", pd.Series(0, index=base.index)).sum())},
        {"metric": "r1_lottery_applicants", "value": int(base.get("r1_lottery_applicant", pd.Series(0, index=base.index)).sum())},
        {"metric": "r3_lottery_applicants", "value": int(base.get("r3_lottery_applicant", pd.Series(0, index=base.index)).sum())},
        {"metric": "survey_2023_responses", "value": int(base.get("survey_2023_response", pd.Series(0, index=base.index)).sum())},
        {"metric": "survey_2024_responses", "value": int(base.get("survey_2024_response", pd.Series(0, index=base.index)).sum())},
    ]
    pd.DataFrame(summary_rows).to_csv(AUDIT_DIR / "analysis_dataset_build_diagnostics.csv", index=False)
    source_coverage.to_csv(AUDIT_DIR / "district_source_coverage.csv", index=False)


def main() -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    print("Inventorying raw files")
    raw_files = inventory_raw_files()
    raw_files.to_csv(AUDIT_DIR / "raw_file_inventory.csv", index=False)

    print("Inventorying raw tables")
    raw_tables = inventory_raw_tables()
    raw_tables.to_csv(AUDIT_DIR / "raw_table_inventory.csv", index=False)

    print("Building source records and identifier crosswalk")
    source_records, source_coverage = build_source_records()
    source_records.to_csv(CLEANED_DIR / "district_source_records.csv", index=False)

    crosswalk = build_crosswalk(source_records)
    crosswalk.to_csv(CLEANED_DIR / "district_identifier_crosswalk.csv", index=False)

    print("Building district base features")
    wri_base = build_wri_base()
    base = crosswalk[["nces_id", "district_name", "state", "city"]].copy()
    base = base.merge(wri_base, on="nces_id", how="left", suffixes=("_xwalk", ""))
    for col in ["district_name", "state", "city"]:
        base[col] = base[col].combine_first(base[f"{col}_xwalk"])
        base = base.drop(columns=[f"{col}_xwalk"])
    base["state_name"] = base["state_name"].combine_first(base["state"])

    rebate_features, grant_features, applicant_features = build_csbp_features()
    bus_features = build_bus_features()
    survey_features = build_survey_features()
    political_features = build_political_features()

    source_flags = crosswalk[["nces_id"] + [c for c in crosswalk.columns if c.startswith("in_")]]
    district_base = merge_features(
        base,
        [source_flags, rebate_features, grant_features, applicant_features, bus_features, survey_features, political_features],
    )

    if district_base["nces_id"].duplicated().any():
        raise ValueError("analysis_district_base is not one row per NCES district ID")

    district_base.to_csv(CLEANED_DIR / "analysis_district_base.csv", index=False)
    write_dta(district_base, CLEANED_DIR / "analysis_district_base.dta")
    write_audit(district_base, source_coverage)

    print(f"Saved {CLEANED_DIR / 'analysis_district_base.csv'} with {len(district_base):,} districts")
    print(f"Saved {CLEANED_DIR / 'district_identifier_crosswalk.csv'} with {len(crosswalk):,} unique NCES IDs")
    print(f"Saved audit files in {AUDIT_DIR}")


if __name__ == "__main__":
    main()

