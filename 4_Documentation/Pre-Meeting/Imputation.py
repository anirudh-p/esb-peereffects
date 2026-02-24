from __future__ import annotations

from pathlib import Path
import argparse
import re
from typing import Optional, Tuple

import pandas as pd


QUARTER_RE = re.compile(r"^(?P<year>\d{4})\s*[Qq](?P<q>[1-4])$")


def parse_quarter(qstr: Optional[str]) -> Optional[pd.Period]:
    if not isinstance(qstr, str) or not qstr.strip():
        return None
    s = qstr.strip()
    m = QUARTER_RE.match(s)
    if not m:
        return None
    year = int(m.group("year"))
    q = int(m.group("q"))
    try:
        return pd.Period(f"{year}Q{q}", freq="Q")
    except Exception:
        return None


def period_to_str(p: Optional[pd.Period]) -> Optional[str]:
    if p is None:
        return None
    return f"{p.year}Q{p.quarter}"


def load_workbook(workbook: Path,
                  bus_sheet: str = "2. Bus-level data",
                  district_sheet: str = "1. District-level data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    bus = pd.read_excel(workbook, sheet_name=bus_sheet, engine="openpyxl")
    district = pd.read_excel(workbook, sheet_name=district_sheet, engine="openpyxl")
    # trim column names
    bus.columns = [c.strip() if isinstance(c, str) else c for c in bus.columns]
    district.columns = [c.strip() if isinstance(c, str) else c for c in district.columns]
    return bus, district


def earliest_quarters_by_lea(bus: pd.DataFrame, lea_col: str) -> pd.DataFrame:
    # Map quarter columns to parsed periods and take min per LEA
    quarter_cols = {
        "award_q": "3p. Quarter awarded",
        "order_q": "3q. Quarter ordered",
        "delivered_q": "3r. Quarter delivered",
        "operating_q": "3s. Quarter first operating",
    }
    out = bus[[lea_col]].copy()
    for key, col in quarter_cols.items():
        if col in bus.columns:
            out[key] = bus[col].apply(parse_quarter)
        else:
            out[key] = None
    grp = out.groupby(lea_col, dropna=False)
    agg = grp.agg({k: lambda s: min([x for x in s if isinstance(x, pd.Period)], default=None) for k in quarter_cols})
    agg = agg.reset_index()
    # convert to strings for output stability
    for k in quarter_cols:
        agg[k] = agg[k].apply(period_to_str)
    return agg


def infer_program_application_quarter(row: pd.Series, cal: pd.DataFrame) -> Tuple[Optional[str], str]:
    """Infer an application quarter and method using waitlist proxy or fallbacks.

    Returns (applied_q_str, method)
    """
    # Use waitlist cues to infer program & year
    wl_cols = {
        "ARP": "6b. ARP 2021 waitlist position",
        "DERA2020": "6c. DERA school bus rebates 2020 waitlist position",
        "DERA2021": "6d. DERA school bus rebates 2021 waitlist position",
    }
    program = None
    year = None
    # ARP 2021
    if wl_cols["ARP"] in row.index and pd.notna(row[wl_cols["ARP"]]):
        program, year = "ARP", 2021
    # DERA 2020/2021
    elif wl_cols["DERA2020"] in row.index and pd.notna(row[wl_cols["DERA2020"]]):
        program, year = "DERA", 2020
    elif wl_cols["DERA2021"] in row.index and pd.notna(row[wl_cols["DERA2021"]]):
        program, year = "DERA", 2021

    if program and year:
        m = cal[(cal["program"] == program) & (cal["year"] == year)]
        if not m.empty:
            q = m.iloc[0]["application_open_q"]
            if isinstance(q, str) and q.startswith("Q"):
                # store as YEARQ#: choose open quarter
                return f"{year}{q}", "program_calendar"
        # fallback if mapping not found
        return f"{year}Q2", "year_mid_proxy"
    # no signal found
    return None, "unknown"


def impute_interest_applied_awarded(district: pd.DataFrame,
                                    lea_cols: dict,
                                    bus_quarters: pd.DataFrame,
                                    calendar_path: Path) -> pd.DataFrame:
    lea_id = lea_cols["id"]
    df = district.copy()
    # bring in earliest award/order/delivered/operating from bus
    df = df.merge(bus_quarters, on=lea_id, how="left")

    # load calendar
    cal = pd.read_csv(calendar_path)

    # Columns existence checks
    interest_col = "6a. Has any expression of interest in ESBs?"
    applied_not_awarded_col = "6e. Applied for ESB funding but not awarded"

    # Initialize outputs
    df["awarded_q"] = df.get("award_q")
    df["awarded_method"] = df["awarded_q"].apply(lambda x: "exact" if pd.notna(x) else "unknown")

    # Applied (imputed from waitlists/calendar or from award gap)
    applied_q_list = []
    applied_method_list = []
    for _, row in df.iterrows():
        applied_q, method = infer_program_application_quarter(row, cal)
        # If still unknown but award exists, backcast by 2 quarters as a coarse proxy
        if applied_q is None and pd.notna(row.get("awarded_q")):
            try:
                p = parse_quarter(row.get("awarded_q"))
                if p is not None:
                    applied_q = period_to_str(p - 2)
                    method = "backcast_award_minus_2q"
            except Exception:
                pass
        # If still unknown and applied_not_awarded is True, use year-mid proxy if we can parse any year from context (not available here)
        if applied_q is None and applied_not_awarded_col in df.columns:
            if pd.notna(row.get(applied_not_awarded_col)) and str(row.get(applied_not_awarded_col)).strip().lower() in {"yes", "true", "1"}:
                applied_q = None  # keep None due to missing year; user can refine later
                method = "applied_noawards_unknown_year"
        applied_q_list.append(applied_q)
        applied_method_list.append(method)
    df["applied_q"] = applied_q_list
    df["applied_method"] = applied_method_list

    # Interest timing
    interest_q_list = []
    interest_method_list = []
    for _, row in df.iterrows():
        interest_q = None
        method = "unknown"
        # derive from applied if present
        if pd.notna(row.get("applied_q")):
            p = parse_quarter(row.get("applied_q"))
            if p is not None:
                interest_q = period_to_str(p - 1)
                method = "from_applied_minus_1q"
        elif pd.notna(row.get("awarded_q")):
            p = parse_quarter(row.get("awarded_q"))
            if p is not None:
                interest_q = period_to_str(p - 3)
                method = "backcast_award_minus_3q"
        else:
            # if 6a is yes but no timing, set to earliest observed year if any (not available); default to None
            pass
        interest_q_list.append(interest_q)
        interest_method_list.append(method)
    df["interest_q"] = interest_q_list
    df["interest_method"] = interest_method_list

    # Keep order/delivered/operating from bus_quarters
    # Already present as strings: order_q, delivered_q, operating_q

    # Durations (quarters) where endpoints exist
    for a, b, name in [
        ("awarded_q", "order_q", "award_to_order_q"),
        ("order_q", "delivered_q", "order_to_delivered_q"),
        ("delivered_q", "operating_q", "delivered_to_operating_q"),
        ("awarded_q", "operating_q", "award_to_operating_q"),
    ]:
        vals = []
        for _, row in df.iterrows():
            pa = parse_quarter(row.get(a))
            pb = parse_quarter(row.get(b))
            if pa is not None and pb is not None:
                dq = (pb.year - pa.year) * 4 + (pb.quarter - pa.quarter)
                vals.append(int(dq))
            else:
                vals.append(None)
        df[name] = vals

    # Select final columns for output
    keep_cols = [
        lea_id, "awarded_q", "awarded_method", "applied_q", "applied_method", "interest_q", "interest_method",
        "order_q", "delivered_q", "operating_q",
        "award_to_order_q", "order_to_delivered_q", "delivered_to_operating_q", "award_to_operating_q",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def main():
    ap = argparse.ArgumentParser(description="Impute timing for interest, application, award, and derive durations.")
    ap.add_argument("--file", type=Path, default=Path("ESB_adoption_dataset_v9_update_june_2025.xlsx"))
    ap.add_argument("--bus-sheet", default="2. Bus-level data")
    ap.add_argument("--district-sheet", default="1. District-level data")
    ap.add_argument("--lea-col", default="1c. LEA ID")
    ap.add_argument("--calendar", type=Path, default=Path("program_calendar.csv"))
    ap.add_argument("--out", type=Path, default=Path("lea_stage_imputation.csv"))
    args = ap.parse_args()

    bus, district = load_workbook(args.file, args.bus_sheet, args.district_sheet)
    bus_quarters = earliest_quarters_by_lea(bus, args.lea_col)
    out = impute_interest_applied_awarded(district, {"id": args.lea_col}, bus_quarters, args.calendar)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out.resolve()}")


if __name__ == "__main__":
    main()
