from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

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


def load_adoption_dataset() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS_DATASET, low_memory=False, dtype={"nces_id": str})
    df["nces_id"] = df["nces_id"].str.zfill(7)
    df[ADOPTION_COL] = df["wri_awarded_by_2024"].fillna(0).astype(int)
    return df


def load_geometry_ids() -> set[str]:
    shp = gpd.read_file(SHP_FILE)
    shp["nces_id"] = clean_nces(shp["GEOID"])
    return set(shp["nces_id"].dropna().unique().tolist())


def remove_legacy_outputs() -> None:
    for name in [
        "descriptive_sample_overview.csv",
        "descriptive_control_summary.csv",
        "descriptive_group_summary.csv",
        "descriptive_state_summary.csv",
        "descriptive_urbanicity_summary.csv",
    ]:
        path = OUTDIR / name
        if path.exists():
            path.unlink()


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "all_districts": pd.Series(True, index=df.index),
        "adopters": df[ADOPTION_COL] == 1,
        "non_adopters": df[ADOPTION_COL] == 0,
    }

    rows = []
    total = int(df["nces_id"].nunique())
    for name, mask in groups.items():
        sub = df.loc[mask].copy()
        rows.append(
            {
                "group": name,
                "districts_n": int(sub["nces_id"].nunique()),
                "share_of_total": float(sub["nces_id"].nunique() / total),
                "adoption_rate": float(sub[ADOPTION_COL].mean()),
                "mean_enrollment": float(pd.to_numeric(sub["enrollment"], errors="coerce").mean()),
                "median_enrollment": float(pd.to_numeric(sub["enrollment"], errors="coerce").median()),
                "mean_income": float(pd.to_numeric(sub["median_income"], errors="coerce").mean()),
                "mean_poverty_rate": float(pd.to_numeric(sub["poverty_rate"], errors="coerce").mean()),
                "mean_pct_white": float(pd.to_numeric(sub["pct_white"], errors="coerce").mean()),
                "mean_pm25": float(pd.to_numeric(sub["pm25"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_adoption_dataset()
    geometry_ids = load_geometry_ids()
    df["has_geometry"] = df["nces_id"].isin(geometry_ids).astype(int)
    remove_legacy_outputs()

    overview = pd.DataFrame(
        [
            {"metric": "districts_total", "value": int(df["nces_id"].nunique())},
            {"metric": "districts_with_any_esb", "value": int(df[ADOPTION_COL].sum())},
            {"metric": "share_with_any_esb", "value": float(df[ADOPTION_COL].mean())},
            {"metric": "districts_without_esb", "value": int((df[ADOPTION_COL] == 0).sum())},
            {"metric": "districts_with_geometry", "value": int(df["has_geometry"].sum())},
            {"metric": "geometry_coverage_rate", "value": float(df["has_geometry"].mean())},
            {
                "metric": "adopters_with_geometry",
                "value": int(df.loc[df[ADOPTION_COL] == 1, "has_geometry"].sum()),
            },
            {
                "metric": "adopter_geometry_coverage_rate",
                "value": float(df.loc[df[ADOPTION_COL] == 1, "has_geometry"].mean()),
            },
        ]
    )

    group_summary = build_group_summary(df)

    overview.to_csv(OUTDIR / "adoption_summary_overview.csv", index=False)
    group_summary.to_csv(OUTDIR / "adoption_group_summary.csv", index=False)

    log("Saved adoption descriptive outputs:")
    for name in [
        "adoption_summary_overview.csv",
        "adoption_group_summary.csv",
    ]:
        log(f"  {OUTDIR / name}")

    (LOGDIR / "adoption_descriptive_statistics_log.txt").write_text(
        "\n".join(log_lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
