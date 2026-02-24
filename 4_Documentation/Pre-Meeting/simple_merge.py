from pathlib import Path
import sys
import argparse
import pandas as pd


def find_lea_column(df: pd.DataFrame, candidates=None):
    if candidates is None:
        candidates = ["LEA ID", "LEA_ID", "LEAId", "leaid", "LEA", "LEA Code", "LEACode", "LEAcode"]
    cols = [c for c in df.columns if isinstance(c, str)]
    lowered = {c: c.lower() for c in cols}
    for cand in candidates:
        for orig, low in lowered.items():
            if cand.lower() == low or cand.lower() in low:
                return orig
    return None


def main():
    p = argparse.ArgumentParser(description="Simple merge of bus (sheet 2) and district (sheet 1) using LEA id")
    p.add_argument("--file", type=Path, default=Path(r"c:\BC PhD\Research\Peer-Effects and Adoption\ESB_adoption_dataset_v9_update_june_2025.xlsx"), help="Path to workbook")
    p.add_argument("--lea-col", default=None, help="Explicit LEA id column name to use for both sheets")
    p.add_argument("--bus-sheet", default="2. Bus-level data", help="Bus-level sheet name (default: '2. Bus-level data')")
    p.add_argument("--district-sheet", default="1. District-level data", help="District-level sheet name (default: '1. District-level data')")
    p.add_argument("--out", default="merged.csv", help="Output CSV path")
    args = p.parse_args()

    fp: Path = args.file
    if not fp.exists():
        print(f"Workbook not found: {fp}")
        sys.exit(1)

    # load bus-level and district-level sheets (use sheet names by default)
    bus = pd.read_excel(fp, sheet_name=args.bus_sheet, engine="openpyxl")
    district = pd.read_excel(fp, sheet_name=args.district_sheet, engine="openpyxl")

    if args.lea_col:
        bus_col = args.lea_col
        district_col = args.lea_col
        if bus_col not in bus.columns:
            print(f"LEA column '{bus_col}' not found in bus sheet. Available columns: {list(bus.columns)}")
            sys.exit(1)
        if district_col not in district.columns:
            print(f"LEA column '{district_col}' not found in district sheet. Available columns: {list(district.columns)}")
            sys.exit(1)
    else:
        # attempt to auto-detect a sensible common LEA id column
        bus_col = find_lea_column(bus)
        district_col = find_lea_column(district)
        if bus_col is None or district_col is None:
            print("Could not auto-detect LEA id column in one of the sheets.")
            print("Bus columns:", list(bus.columns))
            print("District columns:", list(district.columns))
            print("Rerun with --lea-col '<column-name>' to specify the LEA id column explicitly.")
            sys.exit(1)

    print(f"Merging on bus column '{bus_col}' and district column '{district_col}'")
    merged = pd.merge(bus, district, left_on=bus_col, right_on=district_col, how="left", validate="m:1", suffixes=("", "_district"))

    outp = Path(args.out)
    merged.to_csv(outp, index=False)
    print(f"Wrote merged CSV to {outp.resolve()}")
    print('\nFirst 10 rows of merged DataFrame:')
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
