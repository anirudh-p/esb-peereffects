from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANALYSIS_DATASET, TABLES_DIR, LOGS_DIR, ensure_dirs

warnings.filterwarnings("ignore")
ensure_dirs()

OUTDIR = TABLES_DIR / "Preliminaries"
LOGDIR = LOGS_DIR / "Preliminaries"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def summarize_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "variable": col,
            "nonmissing_n": int(s.notna().sum()),
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "median": float(s.median()) if s.notna().any() else np.nan,
            "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
            "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
            "p90": float(s.quantile(0.90)) if s.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(ANALYSIS_DATASET, low_memory=False, dtype={"nces_id": str})
    df["nces_id"] = df["nces_id"].str.zfill(7)

    sample_overview = pd.DataFrame([
        {"metric": "districts_total", "value": int(df["nces_id"].nunique())},
        {"metric": "r1_winners", "value": int(df["IV_Z_R1"].sum())},
        {"metric": "r3_winners", "value": int(df["IV_Z_R3"].sum())},
        {"metric": "any_lottery_winner", "value": int(df["IV_Z"].sum())},
        {"metric": "r2_grantees", "value": int(df["IS_R2_GRANTEE"].sum())},
        {"metric": "r1_losers", "value": int(df["IS_R1_LOSER"].sum())},
        {"metric": "r3_losers", "value": int(df["IS_R3_LOSER"].sum())},
        {"metric": "pooled_lottery_losers", "value": int(df["IS_LOSER_pooled"].sum())},
        {"metric": "r3_applicants", "value": int(df["Y_R3_apply"].sum())},
        {"metric": "wri_any_2023", "value": int(df["wri_any_2023"].sum())},
        {"metric": "wri_any_2024", "value": int(df["wri_any_2024"].sum())},
        {"metric": "wri_any_2023_24", "value": int(df["wri_any_2023_24"].sum())},
        {"metric": "wri_post_r1_cum", "value": int(df["wri_post_r1_cum"].sum())},
        {"metric": "pre_r1_adopters", "value": int(df["is_pre_r1_adopter"].sum())},
    ])

    control_summary = summarize_numeric(
        df,
        ["enrollment", "median_income", "poverty_rate", "pct_white", "pm25", "pct_dem_2020"],
    )

    groups = {
        "all_districts": pd.Series(True, index=df.index),
        "r1_winners": df["IV_Z_R1"] == 1,
        "r1_losers": df["IS_R1_LOSER"] == 1,
        "r3_winners": df["IV_Z_R3"] == 1,
        "r3_losers": df["IS_R3_LOSER"] == 1,
        "r2_grantees": df["IS_R2_GRANTEE"] == 1,
        "non_applicants": (df["IV_Z"] == 0) & (df["IS_LOSER_pooled"] == 0) & (df["IS_R2_GRANTEE"] == 0),
    }
    group_rows = []
    for name, mask in groups.items():
        sub = df.loc[mask].copy()
        if sub.empty:
            continue
        group_rows.append({
            "group": name,
            "districts_n": int(sub["nces_id"].nunique()),
            "mean_enrollment": float(pd.to_numeric(sub["enrollment"], errors="coerce").mean()),
            "mean_income": float(pd.to_numeric(sub["median_income"], errors="coerce").mean()),
            "mean_poverty_rate": float(pd.to_numeric(sub["poverty_rate"], errors="coerce").mean()),
            "mean_pm25": float(pd.to_numeric(sub["pm25"], errors="coerce").mean()),
            "share_priority_r1": float(pd.to_numeric(sub["priority_r1"], errors="coerce").mean()),
            "share_r3_apply": float(sub["Y_R3_apply"].mean()),
            "share_wri_any_2023_24": float(sub["wri_any_2023_24"].mean()),
            "share_wri_post_r1_cum": float(sub["wri_post_r1_cum"].mean()),
        })
    group_summary = pd.DataFrame(group_rows)

    state_summary = (
        df.groupby("state", as_index=False)
        .agg(
            districts_n=("nces_id", "nunique"),
            r1_winners=("IV_Z_R1", "sum"),
            r3_winners=("IV_Z_R3", "sum"),
            r3_apply=("Y_R3_apply", "sum"),
            wri_post_r1_cum=("wri_post_r1_cum", "sum"),
        )
        .sort_values("districts_n", ascending=False)
        .reset_index(drop=True)
    )

    urbanicity_summary = (
        df["urbanicity"]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis("urbanicity")
        .reset_index(name="districts_n")
    )

    sample_overview.to_csv(OUTDIR / "descriptive_sample_overview.csv", index=False)
    control_summary.to_csv(OUTDIR / "descriptive_control_summary.csv", index=False)
    group_summary.to_csv(OUTDIR / "descriptive_group_summary.csv", index=False)
    state_summary.to_csv(OUTDIR / "descriptive_state_summary.csv", index=False)
    urbanicity_summary.to_csv(OUTDIR / "descriptive_urbanicity_summary.csv", index=False)

    log("Saved descriptive outputs:")
    for name in [
        "descriptive_sample_overview.csv",
        "descriptive_control_summary.csv",
        "descriptive_group_summary.csv",
        "descriptive_state_summary.csv",
        "descriptive_urbanicity_summary.csv",
    ]:
        log(f"  {OUTDIR / name}")

    (LOGDIR / "descriptive_statistics_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
