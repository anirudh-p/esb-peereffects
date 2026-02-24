"""Cleaning utilities for the ESB adoption dataset.

This module provides small helper functions to:
- load the bus-level sheet (sheet 2 by default) from an Excel workbook
- load the district-level sheet (sheet 1 by default)
- perform a 1:m merge between bus-level and district-level data on an LEA identifier

The functions are defensive: they try to auto-detect likely LEA ID column names
and provide helpful error messages when the file or expected columns aren't found.

Usage (example):
	from pathlib import Path
	import Cleaning as cl

	bus = cl.load_bus_level(Path("ESB_adoption_dataset_v9_updated_june_2025.xlsx"))
	district = cl.load_district_level(Path("ESB_adoption_dataset_v9_updated_june_2025.xlsx"))
	merged = cl.merge_bus_district(bus, district)

This file intentionally has no external dependencies beyond pandas (and openpyxl
for modern Excel reading). If running as a script it accepts command-line
arguments to customize sheet indexes/names and output.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _find_lea_column(df: pd.DataFrame) -> Optional[str]:
	"""Try to locate a column that looks like an LEA identifier.

	Returns the column name if found, otherwise None.
	The search is case-insensitive and looks for substrings like 'lea', 'lea id',
	'lea_id', 'leaid'.
	"""
	if df is None or df.columns.empty:
		return None
	candidates = [c for c in df.columns if isinstance(c, str)]
	lowered = {c: c.lower() for c in candidates}
	# prioritized patterns
	patterns = ["lea id", "lea_id", "leaid", "lea", "lea_code", "lea_codeid"]
	for p in patterns:
		for orig, low in lowered.items():
			if p in low:
				return orig
	# fallback: exact 'id' only if there's a single id-like column
	id_like = [c for c, low in lowered.items() if low == "id"]
	if len(id_like) == 1:
		return id_like[0]
	return None


def load_sheet(filepath: Path | str, sheet_name=0) -> pd.DataFrame:
	"""Load an Excel sheet into a pandas DataFrame.

	Parameters
	- filepath: path to the Excel workbook
	- sheet_name: sheet name or zero-based index (default 0)
	"""
	fp = Path(filepath)
	if not fp.exists():
		raise FileNotFoundError(f"Excel file not found: {fp}")
	try:
		# use openpyxl for .xlsx support; pandas will autodetect engines where possible
		df = pd.read_excel(fp, sheet_name=sheet_name, engine="openpyxl")
		logger.info("Loaded sheet '%s' from %s with %d rows, %d cols", sheet_name, fp, len(df), len(df.columns))
		return df
	except Exception as exc:
		logger.exception("Failed to read sheet %s from %s: %s", sheet_name, fp, exc)
		raise


def load_bus_level(filepath: Path | str, sheet_index: int = 1) -> pd.DataFrame:
	"""Load the Bus-Level data.

	By convention this project expects the bus-level data to be on sheet 2 of the
	workbook; pandas uses zero-based sheet indexing so the default is 1.
	You can specify a sheet name (string) or a zero-based index (int).
	"""
	return load_sheet(filepath, sheet_name=sheet_index)


def load_district_level(filepath: Path | str, sheet_index: int = 0) -> pd.DataFrame:
	"""Load the District-Level data (default: sheet 1 / index 0)."""
	return load_sheet(filepath, sheet_name=sheet_index)


def merge_bus_district(
	bus_df: pd.DataFrame,
	district_df: pd.DataFrame,
	bus_lea_col: Optional[str] = None,
	district_lea_col: Optional[str] = None,
	how: str = "left",
) -> pd.DataFrame:
	"""Merge bus-level rows to corresponding district-level rows on LEA identifier.

	If column names are not supplied the function will attempt to auto-detect
	reasonable LEA column names in both DataFrames.

	Returns the merged DataFrame (bus-level as the left table by default).
	"""
	if bus_lea_col is None:
		bus_lea_col = _find_lea_column(bus_df)
	if district_lea_col is None:
		district_lea_col = _find_lea_column(district_df)

	if bus_lea_col is None or district_lea_col is None:
		raise ValueError(
			"Could not auto-detect LEA id columns in bus or district data."
			" Pass explicit column names via bus_lea_col and district_lea_col."
		)

	logger.info("Merging on columns: bus='%s' district='%s' (how=%s)", bus_lea_col, district_lea_col, how)

	# perform merge: many bus rows to one district row expected
	merged = pd.merge(
		bus_df,
		district_df,
		left_on=bus_lea_col,
		right_on=district_lea_col,
		how=how,
		suffixes=("", "_district"),
		validate="m:1",
	)
	logger.info("Merge result: %d rows, %d columns", len(merged), len(merged.columns))
	return merged


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Load and merge bus-level and district-level sheets from an Excel workbook.")
	p.add_argument("file", type=Path, help="Path to the Excel workbook")
	p.add_argument("--bus-sheet", default=1, help="Bus-level sheet index (0-based) or name. Default: 1 (sheet 2).")
	p.add_argument("--district-sheet", default=0, help="District-level sheet index (0-based) or name. Default: 0 (sheet 1).")
	p.add_argument("--bus-lea", default=None, help="Column name in bus sheet to use as LEA id (auto-detected if omitted)")
	p.add_argument("--district-lea", default=None, help="Column name in district sheet to use as LEA id (auto-detected if omitted)")
	p.add_argument("--out", default=None, help="Optional CSV output file to write the merged result")
	return p.parse_args()


def main() -> None:
	args = _parse_args()
	fp: Path = args.file
	try:
		bus = load_bus_level(fp, sheet_index=int(args.bus_sheet) if str(args.bus_sheet).isdigit() else args.bus_sheet)
		district = load_district_level(fp, sheet_index=int(args.district_sheet) if str(args.district_sheet).isdigit() else args.district_sheet)
		merged = merge_bus_district(bus, district, bus_lea_col=args.bus_lea, district_lea_col=args.district_lea)
		if args.out:
			outp = Path(args.out)
			merged.to_csv(outp, index=False)
			logger.info("Wrote merged output to %s", outp)
		else:
			# print shape as a minimal confirmation when CLI used without --out
			logger.info("Merged DataFrame shape: %s", merged.shape)
	except Exception as exc:
		logger.exception("Error during processing: %s", exc)
		raise


if __name__ == "__main__":
	main()