"""
01_build_analysis_dataset.py
==============================
Constructs the master analysis dataset from raw sources.
All variable construction decisions are documented inline.
Prints a full data-quality audit before saving.

Output
------
  1_Data/Cleaned/analysis_dataset.csv    — one row per school district
  3_Output/Logs/data_quality_audit.txt
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    WRI_DISTRICT, WRI_BUS_EXCEL, CSB_REBATES, CSB_GRANTS, CSB_APPLICANTS,
    POL_FILE, ANALYSIS_DATASET, LOGS_DIR, TABLES_DIR, ensure_dirs,
)
warnings.filterwarnings("ignore")
ensure_dirs()

log_lines = []
def log(msg=""):
    raw_msg = str(msg)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_msg = raw_msg.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_msg)
    log_lines.append(raw_msg)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def clean_nces(series):
    """Standardise to 7-char zero-padded string and preserve missing values."""
    out = pd.Series(series, copy=False)
    out = out.where(out.notna(), np.nan)
    out = out.astype(str).str.split(".").str[0].str.strip()
    out = out.str.extract(r"(\d+)", expand=False)
    out = out.where(out.notna() & out.ne(""), np.nan)
    return out.str.zfill(7)


def parse_award_year(series):
    """Extract 4-digit year from 'Q3 2023', '2022', etc. Returns float."""
    years = series.astype(str).str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(years, errors="coerce")


def parse_quarter_index(series):
    """Map '2023 Q3' style strings to a sortable numeric quarter index."""
    text = series.astype(str).str.upper()
    years = pd.to_numeric(text.str.extract(r"(\d{4})", expand=False), errors="coerce")
    quarters = pd.to_numeric(text.str.extract(r"Q([1-4])", expand=False), errors="coerce")
    return years * 4 + quarters


def classify_fuel_group(df_in, electric_col, propane_col, cng_col):
    """
    Collapse requested bus mix into coarse lottery strata:
      - ZE_only: electric only
      - clean_nonZE_only: propane/CNG only
      - mixed: electric plus propane/CNG
      - unknown_zero: no positive bus counts observed
    """
    tmp = df_in[[electric_col, propane_col, cng_col]].copy()
    for col in tmp.columns:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0)

    has_e = tmp[electric_col] > 0
    has_p = tmp[propane_col] > 0
    has_c = tmp[cng_col] > 0
    has_clean_nonze = has_p | has_c

    out = pd.Series("unknown_zero", index=df_in.index, dtype="object")
    out.loc[has_e & ~has_clean_nonze] = "ZE_only"
    out.loc[~has_e & has_clean_nonze] = "clean_nonZE_only"
    out.loc[has_e & has_clean_nonze] = "mixed"
    return out


def summarize_fuel_group(df_in, out_col):
    """Reduce district-level fuel group to one row per NCES ID using the modal value."""
    tmp = df_in[["nces_id", "fuel_group"]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=["nces_id", out_col])
    return (
        tmp.groupby("nces_id")["fuel_group"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
        .rename(columns={"fuel_group": out_col})
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — District universe from WRI district-level data
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("STEP 1  —  District universe (wri_data.csv)")
log("=" * 70)

# NOTE: wri_data.csv = WRI v9 Sheet "1. District-level data"  (19,517 rows)
# Covers more LEAs than the traditional school-district universe (~13k) because
# WRI also tracks charter networks and city entities. Controls (income, poverty,
# pct_white, pm25) are populated only for ~13k traditional districts.
df = pd.read_csv(WRI_DISTRICT, low_memory=False)
df["nces_id"] = clean_nces(df["1c. LEA ID"])

log(f"  Rows loaded     : {len(df):,}")
log(f"  Unique NCES IDs : {df['nces_id'].nunique():,}")
log(f"  Sample IDs      : {df['nces_id'].head(5).tolist()}")

# Rename key columns to clean names
col_map = {
    "1a. State"                                          : "state",
    "1p. Locale broad type (name)"                       : "urbanicity",
    "4b. Number of students in district"                 : "enrollment",
    "4f. Median household income"                        : "median_income",
    "4g. Percent of population below the poverty level"  : "poverty_rate",
    "4h. Percent one race: White "                       : "pct_white",
    "5f. PM2.5 concentration"                            : "pm25",
    "5o. EPA 2022 Clean School Bus Rebate Program prioritized school district?" : "priority_r1",
    "5p. EPA 2023 Clean School Bus Grant & Rebate Programs prioritized school district?" : "priority_r23",
}
df = df.rename(columns=col_map)

# Keep only the columns we need as controls
keep_cols = ["nces_id", "state", "urbanicity", "enrollment",
             "median_income", "poverty_rate", "pct_white", "pm25",
             "priority_r1", "priority_r23"]
base = df[keep_cols].copy()
invalid_base_ids = base["nces_id"].isna().sum()
if invalid_base_ids:
    log(f"  Invalid or missing NCES IDs dropped from district universe: {invalid_base_ids:,}")
base = base.dropna(subset=["nces_id"]).copy()
dup_base_ids = int(base["nces_id"].duplicated().sum())
if dup_base_ids:
    log(f"  Duplicate NCES IDs in district universe: {dup_base_ids:,} rows; keeping first occurrence")
    base = base.drop_duplicates("nces_id").copy()

# WRI stores poverty_rate and pct_white as fractions (0–1) already; no rescaling.
# (confirmed: mean pct_white ~0.82, mean poverty_rate ~0.12 makes sense)

# Normalize priority columns: "yes" / "Yes" → 1, "No" → 0
for pcol in ["priority_r1", "priority_r23"]:
    base[pcol] = base[pcol].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0}).astype("Int64")

log(f"\n  Control availability (of {len(base):,} districts):")
for c in ["enrollment", "median_income", "poverty_rate", "pct_white", "pm25",
          "urbanicity", "priority_r1", "priority_r23", "state"]:
    nn  = base[c].notna().sum()
    pct = 100 * nn / len(base)
    log(f"    {c:<20}: {nn:>6,}  ({pct:5.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Lottery winners from CSB_Rebates (R1 and R3 only)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("STEP 2  —  CSBP lottery winners (CSB_Rebates.xlsx)")
log("=" * 70)

# DESIGN NOTE: R2 (CSB_Grants.xlsx) is a COMPETITIVE grant, not a lottery.
# It is excluded from all IV and loser-density construction.
# R2 awardees count as CSBP adopters for the all-source outcome only.

EXCL = ["WITHDRAWN", "CANCELLED", "NOT SELECTED", "INELIGIBLE", "DENIED"]

reb = pd.read_excel(CSB_REBATES)
reb["nces_id"] = clean_nces(reb["NCES District ID"])
reb = reb.dropna(subset=["nces_id"]).copy()
reb_active = reb[~reb["Project Status"].str.upper().isin(EXCL)].copy()
reb_active["fuel_group"] = classify_fuel_group(
    reb_active,
    "Number of Electric Buses",
    "Number of Propane Buses",
    "Number of CNG Buses",
)

log(f"  CSB_Rebates rows          : {len(reb):,}")
log(f"  After excluding {EXCL[:3]}... : {len(reb_active):,}")
log(f"\n  Funding Year distribution :")
log(reb_active["Project Status"].value_counts().to_string())
log()
log(reb_active["Funding Year"].value_counts().sort_index().to_string())

r1_winners = (reb_active[reb_active["Funding Year"] == 2022]["nces_id"]
              .drop_duplicates().to_frame())
r1_winners["IV_Z_R1"] = 1
r1_winner_fuel = summarize_fuel_group(
    reb_active[reb_active["Funding Year"] == 2022].copy(),
    "r1_fuel_group_winner",
)

r3_winners = (reb_active[reb_active["Funding Year"] == 2023]["nces_id"]
              .drop_duplicates().to_frame())
r3_winners["IV_Z_R3"] = 1

log(f"\n  Unique R1 lottery winners : {len(r1_winners):,}")
log(f"  Unique R3 lottery winners : {len(r3_winners):,}")
log(f"  Won both R1 and R3        : {len(set(r1_winners.nces_id) & set(r3_winners.nces_id)):,}")
log("\n  R1 winner fuel-group mix:")
log(reb_active.loc[reb_active["Funding Year"] == 2022, "fuel_group"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — R2 grant awardees (adopted via competitive grant, NOT lottery)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("STEP 3  —  R2 grant awardees (CSB_Grants.xlsx)  — EXCLUDED from IV")
log("=" * 70)

grn = pd.read_excel(CSB_GRANTS)
grn["nces_id"] = clean_nces(grn["NCES District ID"])
grn = grn.dropna(subset=["nces_id"]).copy()
grn_active = grn[grn["Project Status"].str.strip() == "Funds Awarded"].copy()
r2_grantees = grn_active["nces_id"].drop_duplicates().to_frame()
r2_grantees["IS_R2_GRANTEE"] = 1

log(f"  Unique R2 grantees  : {len(r2_grantees):,}  (all Funding Year 2023)")
log(f"  Overlap with R1 win : {len(set(r2_grantees.nces_id) & set(r1_winners.nces_id)):,}")
log(f"  Overlap with R3 win : {len(set(r2_grantees.nces_id) & set(r3_winners.nces_id)):,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Lottery applicants (waitlisted / rejected)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("STEP 4  —  CSBP applicant file (waitlisted & rejected)")
log("=" * 70)

app = pd.read_excel(CSB_APPLICANTS)
app["nces_id"] = clean_nces(app["NCES District ID"])
app = app.dropna(subset=["nces_id"]).copy()
app["fuel_group"] = classify_fuel_group(
    app,
    "Number of Electric Buses",
    "Number of Propane Buses",
    "Number of CNG Buses",
)

log(f"  Total rows : {len(app):,}")
log(f"\n  Round breakdown:")
log(app["Round"].value_counts().to_string())
log(f"\n  Status breakdown:")
log(app["Project Status"].value_counts().to_string())

# ── Exclude R2 grant applicants from all lottery-related variables ──────────
# R2 is competitive (no lottery). Including R2 as "losers" in the loser-density
# control would conflate competitive selection with lottery outcomes.
app_lottery = app[~app["Round"].str.contains("R2", na=False)].copy()
log(f"\n  After excluding R2 rows : {len(app_lottery):,}")
log("\n  Lottery-applicant fuel-group mix:")
log(app_lottery["fuel_group"].value_counts().to_string())

# ── R1 losers: in applicant file, Round='R1', not in R1 winners ─────────────
r1_app_rows = app_lottery[app_lottery["Round"].str.contains("R1", na=False)].copy()
r1_loser_raw = r1_app_rows["nces_id"].drop_duplicates()
overlap_r1 = len(set(r1_loser_raw) & set(r1_winners["nces_id"]))
log(f"\n  R1 waitlisted/rejected (unique districts) : {len(r1_loser_raw):,}")
log(f"  Overlap with R1 winners                   : {overlap_r1:,}  "
    f"← multi-application districts: treating as winners")
r1_losers_clean = r1_loser_raw[~r1_loser_raw.isin(r1_winners["nces_id"])].to_frame()
r1_losers_clean["IS_R1_LOSER"] = 1
r1_loser_fuel = summarize_fuel_group(
    r1_app_rows[~r1_app_rows["nces_id"].isin(r1_winners["nces_id"])].copy(),
    "r1_fuel_group_loser",
)
log(f"  R1 losers after removing winner overlap    : {len(r1_losers_clean):,}")

# ── R3 losers: in applicant file, Round='R3', not in R3 winners ─────────────
r3_loser_raw = (app_lottery[app_lottery["Round"].str.contains("R3", na=False)]
                ["nces_id"].drop_duplicates())
overlap_r3 = len(set(r3_loser_raw) & set(r3_winners["nces_id"]))
log(f"\n  R3 waitlisted/rejected (unique districts) : {len(r3_loser_raw):,}")
log(f"  Overlap with R3 winners                   : {overlap_r3:,}  "
    f"← multi-application districts: treating as winners")
r3_losers_clean = r3_loser_raw[~r3_loser_raw.isin(r3_winners["nces_id"])].to_frame()
r3_losers_clean["IS_R3_LOSER"] = 1
log(f"  R3 losers after removing winner overlap    : {len(r3_losers_clean):,}")

# Summary: R3 "applicant" universe = winners + waitlisted/rejected
r3_applicants = set(r3_winners["nces_id"]) | set(r3_losers_clean["nces_id"])
log(f"\n  Full R3 applicant universe (win ∪ waitlist): {len(r3_applicants):,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — WRI all-source adoption outcomes (bus-level data)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("STEP 5  —  WRI all-source adoption outcomes (bus-level data)")
log("=" * 70)

buses = pd.read_excel(WRI_BUS_EXCEL, sheet_name="2. Bus-level data")
buses["nces_id"]    = clean_nces(buses["1c. LEA ID"])
buses = buses.dropna(subset=["nces_id"]).copy()
buses["award_year"] = parse_award_year(buses["3p. Quarter awarded"])
buses["award_q_index"] = parse_quarter_index(buses["3p. Quarter awarded"])
buses["delivery_year"] = parse_award_year(buses["3r. Quarter delivered"])
buses["delivery_q_index"] = parse_quarter_index(buses["3r. Quarter delivered"])
buses["operating_year"] = parse_award_year(buses["3s. Quarter first operating"])
buses["operating_q_index"] = parse_quarter_index(buses["3s. Quarter first operating"])
buses_valid = buses[buses["award_year"].between(1990, 2035)
                    & buses["nces_id"].notna()
                    & buses["nces_id"].ne("0000000")].copy()

Q3_2022 = 2022 * 4 + 3   # R1 announced ~August 2022; Q3 2022 = Jul-Sep
Q2_2022 = 2022 * 4 + 2   # Q2 2022 = Apr-Jun (last pre-announcement quarter)
Q3_2023 = 2023 * 4 + 3
Q4_2023 = 2023 * 4 + 4
Q4_2024 = 2024 * 4 + 4


def build_indicator_frame(df_in, mask, out_col):
    out = df_in.loc[mask, ["nces_id"]].drop_duplicates().copy()
    out[out_col] = 1
    return out

log(f"  Bus-level rows total         : {len(buses):,}")
log(f"  Rows with valid year + NCES  : {len(buses_valid):,}")
log(f"\n  Award year distribution:")
log(buses_valid["award_year"].value_counts().sort_index().to_string())

# Pre-R1 adopters: awarded anything before 2022 (not a peer effect target)
pre_r1 = (buses_valid[buses_valid["award_year"] < 2022]
          .groupby("nces_id").size().rename("pre_r1_bus_count").reset_index())
pre_r1["is_pre_r1_adopter"] = 1

# All-source outcome windows (for Estimands 3 & 4)
for label, yr_min, yr_max in [("2023",    2023, 2023),
                               ("2024",    2024, 2024),
                               ("2023_24", 2023, 2024)]:
    window = buses_valid[buses_valid["award_year"].between(yr_min, yr_max)]
    dist_in_window = window.groupby("nces_id").size().rename(f"bus_count_{label}").reset_index()
    dist_in_window[f"wri_any_{label}"] = 1
    log(f"\n  Districts with ≥1 bus awarded in {label}: {len(dist_in_window):,}")
    # First-ever: only districts NOT seen before the window
    not_pre = ~dist_in_window["nces_id"].isin(pre_r1["nces_id"])
    log(f"    of which first-ever adopters          : {not_pre.sum():,}")
    if label == "2023":
        wri_2023 = dist_in_window
    elif label == "2024":
        wri_2024 = dist_in_window
    else:
        wri_2324 = dist_in_window

# Cumulative all-source stock / visibility measures used for structural checks
wri_awarded_by_2023 = build_indicator_frame(
    buses_valid,
    buses_valid["award_q_index"].notna() & (buses_valid["award_q_index"] <= Q4_2023),
    "wri_awarded_by_2023",
)
wri_awarded_by_2024 = build_indicator_frame(
    buses_valid,
    buses_valid["award_q_index"].notna() & (buses_valid["award_q_index"] <= Q4_2024),
    "wri_awarded_by_2024",
)
wri_delivered_by_2023q3 = build_indicator_frame(
    buses_valid,
    buses_valid["delivery_q_index"].notna() & (buses_valid["delivery_q_index"] <= Q3_2023),
    "wri_delivered_by_2023q3",
)
wri_operating_by_2023q3 = build_indicator_frame(
    buses_valid,
    buses_valid["operating_q_index"].notna() & (buses_valid["operating_q_index"] <= Q3_2023),
    "wri_operating_by_2023q3",
)
wri_visible_by_2023q3 = pd.concat(
    [
        wri_delivered_by_2023q3[["nces_id"]],
        wri_operating_by_2023q3[["nces_id"]],
    ],
    ignore_index=True,
).drop_duplicates()
wri_visible_by_2023q3["wri_visible_by_2023q3"] = 1

# Post-announcement 2022 adopters: FIRST WRI bus awarded in Q3 or Q4 2022.
# R1 lottery results announced ~August 2022 (Q3 2022). Districts whose first
# bus arrived in Q3-Q4 2022 could plausibly be responding to R1 peer signals
# (announcement visible, even if buses not yet delivered). Districts with a
# bus in Q1-Q2 2022 (pre-announcement) are excluded.
_first_award_q = (buses_valid
                  .groupby("nces_id")["award_q_index"]
                  .min()
                  .reset_index()
                  .rename(columns={"award_q_index": "first_award_q"}))
_post_ann_2022 = _first_award_q[
    _first_award_q["first_award_q"].between(Q3_2022, Q2_2022 + 2)  # Q3-Q4 2022
].copy()
_post_ann_2022["wri_post_ann_2022"] = 1
log(f"\n  Districts with first bus in Q3-Q4 2022 (post-announcement): {len(_post_ann_2022):,}")

log(f"\n  Districts with >=1 bus awarded by end-2023 : {len(wri_awarded_by_2023):,}")
log(f"  Districts with >=1 bus delivered by 2023 Q3: {len(wri_delivered_by_2023q3):,}")
log(f"  Districts with >=1 bus operating by 2023 Q3: {len(wri_operating_by_2023q3):,}")
log(f"  Districts visibly deployed by 2023 Q3      : {len(wri_visible_by_2023q3):,}")

# ── Delivery timing for R1 winners (Estimand 4) ─────────────────────────────
log(f"\n  R1 bus delivery timing:")
r1_buses = buses_valid[
    (buses_valid["award_year"] == 2022) &
    buses_valid["nces_id"].isin(r1_winners["nces_id"])
].copy()
log(f"    R1 buses total            : {len(r1_buses):,}")
log(f"    Delivery date available   : {r1_buses['delivery_year'].notna().sum():,} "
    f"({100 * r1_buses['delivery_year'].notna().mean():.1f}%)")
log(f"    Delivery year distribution:")
log(r1_buses["delivery_year"].value_counts(dropna=False).sort_index().to_string())
r1_delivery = (
    r1_buses.groupby("nces_id")
    .agg(
        r1_first_delivery_year=("delivery_year", "min"),
        r1_first_delivery_q_index=("delivery_q_index", "min"),
        r1_first_operating_year=("operating_year", "min"),
        r1_first_operating_q_index=("operating_q_index", "min"),
    )
    .reset_index()
)
r1_delivery = r1_winners[["nces_id"]].merge(r1_delivery, on="nces_id", how="left")

# Coarse delivery split kept for backwards compatibility with older notes.
r1_delivery["r1_early_delivery"] = (
    r1_delivery["r1_first_delivery_year"].notna()
    & (r1_delivery["r1_first_delivery_year"] < 2024)
).astype(int)
r1_delivery["r1_late_or_unknown"] = (
    r1_delivery["r1_first_delivery_year"].isna()
    | (r1_delivery["r1_first_delivery_year"] >= 2024)
).astype(int)

# Sharp delivery / operating timing used in post-poster diagnostics.
r1_delivery["r1_pre_r3_delivery"] = (
    r1_delivery["r1_first_delivery_q_index"].notna()
    & (r1_delivery["r1_first_delivery_q_index"] <= Q3_2023)
).astype(int)
r1_delivery["r1_post_r3_delivery"] = (
    r1_delivery["r1_first_delivery_q_index"].notna()
    & (r1_delivery["r1_first_delivery_q_index"] > Q3_2023)
).astype(int)
r1_delivery["r1_delivery_unknown"] = r1_delivery["r1_first_delivery_q_index"].isna().astype(int)

r1_delivery["r1_pre_r3_operating"] = (
    r1_delivery["r1_first_operating_q_index"].notna()
    & (r1_delivery["r1_first_operating_q_index"] <= Q3_2023)
).astype(int)
r1_delivery["r1_post_r3_operating"] = (
    r1_delivery["r1_first_operating_q_index"].notna()
    & (r1_delivery["r1_first_operating_q_index"] > Q3_2023)
).astype(int)
r1_delivery["r1_operating_unknown"] = r1_delivery["r1_first_operating_q_index"].isna().astype(int)

e = int(r1_delivery["r1_early_delivery"].sum())
l = int(r1_delivery["r1_late_or_unknown"].sum())
log(f"\n    R1 districts with early delivery (<2024) : {e:,}")
log(f"    R1 districts with late/unknown delivery  : {l:,}")
log(f"    R1 districts delivered by 2023 Q3        : {int(r1_delivery['r1_pre_r3_delivery'].sum()):,}")
log(f"    R1 districts operating by 2023 Q3        : {int(r1_delivery['r1_pre_r3_operating'].sum()):,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Political control (county-level Dem vote share 2020)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("STEP 6  —  Political control (county presidential 2020)")
log("=" * 70)

pol = pd.read_csv(POL_FILE, low_memory=False)
pol20 = pol[(pol["year"] == 2020) & (pol["office"] == "US PRESIDENT")].copy()
tot = pol20.groupby("county_fips")["candidatevotes"].sum().rename("total_votes").reset_index()
dem = (pol20[pol20["party"] == "DEMOCRAT"]
       .groupby("county_fips")["candidatevotes"].sum()
       .rename("dem_votes").reset_index())
county_pol = tot.merge(dem, on="county_fips", how="left")
county_pol["pct_dem_2020"] = county_pol["dem_votes"] / county_pol["total_votes"]
county_pol["county_fips"] = county_pol["county_fips"].astype(int)
log(f"  2020 counties with vote data : {len(county_pol):,}")

# LEA → county crosswalk from WRI v9 sheet "5. Counties"
xwalk = pd.read_excel(WRI_BUS_EXCEL, sheet_name="5. Counties")
xwalk["nces_id"]    = clean_nces(xwalk["1c. LEA ID"])
xwalk = xwalk.dropna(subset=["nces_id"]).copy()
xwalk = xwalk.rename(columns={"10b. County FIPS Code": "county_fips"})
xwalk["county_fips"] = pd.to_numeric(xwalk["county_fips"], errors="coerce")
xwalk = xwalk.dropna(subset=["county_fips"])
xwalk["county_fips"] = xwalk["county_fips"].astype(int)
xwalk = xwalk.merge(county_pol[["county_fips", "pct_dem_2020"]], on="county_fips", how="left")
# Multi-county LEAs → average pct_dem_2020 across counties
lea_pol = (xwalk.groupby("nces_id")["pct_dem_2020"].mean()
           .rename("pct_dem_2020").reset_index())
log(f"  LEA-county crosswalk rows  : {len(xwalk):,}")
log(f"  Unique LEAs with political : {len(lea_pol):,}")
log(f"  pct_dem_2020 coverage      : {lea_pol['pct_dem_2020'].notna().sum():,} "
    f"({100 * lea_pol['pct_dem_2020'].notna().mean():.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Merge everything onto district universe
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("STEP 7  —  Merge onto district universe")
log("=" * 70)

d = base.copy()

# R1 applicant fuel-group mapping used for post-poster lottery-strata controls.
d = d.merge(r1_winner_fuel, on="nces_id", how="left")
d = d.merge(r1_loser_fuel, on="nces_id", how="left")
d["r1_fuel_group"] = (
    d["r1_fuel_group_winner"]
    .combine_first(d["r1_fuel_group_loser"])
    .fillna("nonapp")
)
d = d.drop(columns=["r1_fuel_group_winner", "r1_fuel_group_loser"])

# Lottery & applicant indicators
for frame, col in [(r1_winners,     "IV_Z_R1"),
                   (r3_winners,     "IV_Z_R3"),
                   (r2_grantees,    "IS_R2_GRANTEE"),
                   (r1_losers_clean,"IS_R1_LOSER"),
                   (r3_losers_clean,"IS_R3_LOSER")]:
    d = d.merge(frame, on="nces_id", how="left")

for col in ["IV_Z_R1", "IV_Z_R3", "IS_R2_GRANTEE", "IS_R1_LOSER", "IS_R3_LOSER"]:
    d[col] = d[col].fillna(0).astype(int)

# Derived lottery variables
d["IV_Z"]       = ((d["IV_Z_R1"] == 1) | (d["IV_Z_R3"] == 1)).astype(int)  # pooled
d["IS_ADOPTER"] = d["IV_Z"]                                                  # CSBP lottery only
d["IS_ADOPTER_CSBP_ANY"] = ((d["IV_Z"] == 1) | (d["IS_R2_GRANTEE"] == 1)).astype(int)

# Loser density variables (LOTTERY rounds only, R2 excluded by construction)
d["IS_LOSER_R1_only"] = d["IS_R1_LOSER"]
d["IS_LOSER_R3_only"] = d["IS_R3_LOSER"]
d["IS_LOSER_pooled"]  = ((d["IS_R1_LOSER"] == 1) | (d["IS_R3_LOSER"] == 1)).astype(int)

# R3 application outcome (for Estimand 2)
d["Y_R3_apply"] = ((d["IV_Z_R3"] == 1) | (d["IS_R3_LOSER"] == 1)).astype(int)

# WRI all-source outcomes
for frame, col in [(wri_2023,  "wri_any_2023"),
                   (wri_2024,  "wri_any_2024"),
                   (wri_2324,  "wri_any_2023_24")]:
    d = d.merge(frame[["nces_id", col]], on="nces_id", how="left")
    d[col] = d[col].fillna(0).astype(int)

for frame, col in [
    (wri_awarded_by_2023, "wri_awarded_by_2023"),
    (wri_awarded_by_2024, "wri_awarded_by_2024"),
    (wri_delivered_by_2023q3, "wri_delivered_by_2023q3"),
    (wri_operating_by_2023q3, "wri_operating_by_2023q3"),
    (wri_visible_by_2023q3, "wri_visible_by_2023q3"),
    (_post_ann_2022, "wri_post_ann_2022"),
]:
    d = d.merge(frame[["nces_id", col]], on="nces_id", how="left")
    d[col] = d[col].fillna(0).astype(int)

# Pre-R1 adopter flag
d = d.merge(pre_r1[["nces_id", "is_pre_r1_adopter"]], on="nces_id", how="left")
d["is_pre_r1_adopter"] = d["is_pre_r1_adopter"].fillna(0).astype(int)

# Cumulative post-R1 WRI adoption (Fix A — Post-Meeting branch)
# Definition: any WRI-tracked ESB awarded in calendar years 2023-2024, EXCLUDING
# pre-R1 adopters (districts with a WRI bus awarded before 2022).
#
# Why not include 2022? Q4 2022 is when R1 CSBP buses were awarded to the R1
# winners themselves (~368 districts). Including Q3-Q4 2022 would put the R1
# winners' own CSBP adoption into the peer-effect outcome variable, directly
# confounding the IV design. Q1-Q2 2022 adoptions predate the R1 announcement
# (~August 2022) and are likewise excluded. Using 2023-2024 gives a clean
# post-delivery window where spillovers from observed R1 buses can operate.
d["wri_post_ann_2022"] = d["wri_post_ann_2022"].fillna(0).astype(int)
d["wri_post_r1_cum"] = (
    (d["wri_any_2023_24"] == 1) & (d["is_pre_r1_adopter"] == 0)
).astype(int)

# R1 delivery timing
d = d.merge(
    r1_delivery[
        [
            "nces_id",
            "r1_first_delivery_year",
            "r1_first_delivery_q_index",
            "r1_first_operating_year",
            "r1_first_operating_q_index",
            "r1_early_delivery",
            "r1_late_or_unknown",
            "r1_pre_r3_delivery",
            "r1_post_r3_delivery",
            "r1_delivery_unknown",
            "r1_pre_r3_operating",
            "r1_post_r3_operating",
            "r1_operating_unknown",
        ]
    ],
    on="nces_id",
    how="left",
)
# Non-R1 winner districts: delivery vars not applicable (leave as NaN)

# Political
d = d.merge(lea_pol, on="nces_id", how="left")

log(f"  Total districts in dataset : {len(d):,}")
if d["nces_id"].isna().any():
    raise ValueError("analysis dataset contains missing NCES IDs")
if d["nces_id"].duplicated().any():
    dup_n = int(d["nces_id"].duplicated().sum())
    raise ValueError(f"analysis dataset is not one row per district; found {dup_n} duplicate NCES IDs")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Data quality audit
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("STEP 8  —  Data quality audit")
log("=" * 70)

# ── 8a. Outcome and instrument distributions ─────────────────────────────────
log("\n━━ 8a. Key binary variable rates ━━")
for col, label in [
        ("IV_Z_R1",          "R1 lottery winner"),
        ("IV_Z_R3",          "R3 lottery winner"),
        ("IV_Z",             "Any lottery winner (R1+R3)"),
        ("IS_R2_GRANTEE",    "R2 grant awardee"),
        ("IS_ADOPTER",       "IS_ADOPTER (csbp lottery only)"),
        ("IS_ADOPTER_CSBP_ANY", "IS_ADOPTER_CSBP_ANY (incl. R2)"),
        ("IS_R1_LOSER",      "R1 lottery loser"),
        ("IS_R3_LOSER",      "R3 lottery loser"),
        ("IS_LOSER_pooled",  "Pooled lottery loser"),
        ("Y_R3_apply",       "Applied to R3 (Estimand 2 outcome)"),
        ("wri_any_2023",     "WRI any adoption 2023"),
        ("wri_any_2024",     "WRI any adoption 2024"),
        ("wri_any_2023_24",  "WRI any adoption 2023-24"),
        ("wri_post_r1_cum",  "WRI cumulative post-R1 (new primary outcome)"),
        ("wri_awarded_by_2023", "WRI awarded by end-2023"),
        ("wri_delivered_by_2023q3", "WRI delivered by 2023 Q3"),
        ("wri_operating_by_2023q3", "WRI operating by 2023 Q3"),
        ("wri_visible_by_2023q3", "WRI visible by 2023 Q3"),
        ("is_pre_r1_adopter","Pre-R1 ESB adopter"),
]:
    n  = d[col].sum()
    pct = 100 * d[col].mean()
    log(f"  {label:<45}: {n:>5,}  ({pct:5.2f}%)")

# ── 8b. Control variable coverage ────────────────────────────────────────────
log("\n━━ 8b. Control variable missing-data audit ━━")
ctrl_cols = ["enrollment", "median_income", "poverty_rate", "pct_white",
             "pm25", "urbanicity", "priority_r1", "priority_r23",
             "state", "pct_dem_2020"]
for c in ctrl_cols:
    nn  = d[c].notna().sum()
    mis = len(d) - nn
    log(f"  {c:<25}: {nn:>6,} non-null  |  {mis:>5,} missing ({100*mis/len(d):.1f}%)")

# Estimation sample: districts with all controls
required = ["enrollment", "median_income", "poverty_rate", "pct_white",
            "pm25", "state", "pct_dem_2020"]
est = d.dropna(subset=required)
log(f"\n  Estimation sample (all controls present): {len(est):,} / {len(d):,} "
    f"({100*len(est)/len(d):.1f}%)")

# Are dropped districts systematically different on outcomes?
log("\n  Does sample restriction create systematic bias?")
dropped = d[~d.index.isin(est.index)]
for col in ["IV_Z_R1", "IV_Z_R3", "IS_ADOPTER", "Y_R3_apply",
            "wri_any_2023_24"]:
    full_rate = d[col].mean() * 100
    est_rate  = est[col].mean() * 100
    drop_rate = dropped[col].mean() * 100 if len(dropped) else np.nan
    log(f"    {col:<30}: full={full_rate:.2f}%  "
        f"est={est_rate:.2f}%  dropped={drop_rate:.2f}%")

# ── 8c. Cross-tabulations ─────────────────────────────────────────────────────
log("\n━━ 8c. Lottery winner / loser / non-applicant breakdown ━━")
d["lottery_group"] = "non-applicant"
d.loc[d["IS_R1_LOSER"] == 1,  "lottery_group"] = "R1 loser"
d.loc[d["IV_Z_R1"]     == 1,  "lottery_group"] = "R1 winner"
d.loc[d["IS_R3_LOSER"] == 1,  "lottery_group"] = "R3 loser"
d.loc[d["IV_Z_R3"]     == 1,  "lottery_group"] = "R3 winner"
d.loc[(d["IV_Z_R1"] == 1) & (d["IV_Z_R3"] == 1), "lottery_group"] = "R1+R3 winner"
log(d["lottery_group"].value_counts().to_string())

# ── 8d. Priority variable distributions ──────────────────────────────────────
log("\n━━ 8d. Priority status ━━")
for v in ["priority_r1", "priority_r23"]:
    log(f"\n  {v}:")
    vc = d[v].value_counts(dropna=False)
    log(vc.to_string())

log("\n  R1 fuel-group status:")
log(d["r1_fuel_group"].value_counts(dropna=False).to_string())

# ── 8e. Control variable distributions ───────────────────────────────────────
log("\n━━ 8e. Control variable summary statistics ━━")
for c in ["enrollment", "median_income", "poverty_rate", "pct_white", "pm25"]:
    s = est[c].describe()
    log(f"\n  {c}:  mean={s['mean']:.3g}  sd={s['std']:.3g}  "
        f"p10={s['10%'] if '10%' in s else est[c].quantile(0.1):.3g}  "
        f"p90={s['90%'] if '90%' in s else est[c].quantile(0.9):.3g}  "
        f"missing={est[c].isna().sum()}")

log("\n  Urbanicity distribution (estimation sample):")
log(est["urbanicity"].value_counts(dropna=False).to_string())

# ── 8f. Balance test: do R1 winners differ from R1 losers on pre-treatment vars?
log("\n━━ 8f. Lottery balance: R1 winners vs R1 losers on pre-treatment controls ━━")
r1_pool = est[(est["IV_Z_R1"] == 1) | (est["IS_R1_LOSER"] == 1)].copy()
log(f"  R1 applicants in estimation sample: {len(r1_pool):,} "
    f"({r1_pool['IV_Z_R1'].sum():,} winners, {r1_pool['IS_R1_LOSER'].sum():,} losers)")
for c in ["median_income", "poverty_rate", "enrollment", "pm25", "pct_white"]:
    ols = sm.OLS(r1_pool[c].fillna(r1_pool[c].median()),
                 sm.add_constant(r1_pool["IV_Z_R1"].astype(float))
                ).fit(cov_type="HC1")
    coef = ols.params["IV_Z_R1"]
    pval = ols.pvalues["IV_Z_R1"]
    flag = "  ***" if pval < 0.01 else "  **" if pval < 0.05 else "  *" if pval < 0.1 else ""
    log(f"  {c:<25}: coef={coef:+.4g}  p={pval:.4f}{flag}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Save
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("STEP 9  —  Save")
log("=" * 70)

drop_tmp = ["lottery_group"]
d_save = d.drop(columns=[c for c in drop_tmp if c in d.columns])
d_save.to_csv(ANALYSIS_DATASET, index=False)
log(f"\n  Saved: {ANALYSIS_DATASET}")
log(f"  Final dataset shape: {d_save.shape}")

log_text = "\n".join(log_lines)
audit_path = LOGS_DIR / "data_quality_audit.txt"
with open(audit_path, "w", encoding="utf-8") as f:
    f.write(log_text)
log(f"  Saved: {audit_path}")
