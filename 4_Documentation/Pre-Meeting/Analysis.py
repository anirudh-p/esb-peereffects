from __future__ import annotations

from pathlib import Path
import argparse
import os

import pandas as pd


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def load_lea_vars(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize column names (strip)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


def make_plots(df: pd.DataFrame, outdir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")

    # Histogram of peer adoption share
    plt.figure(figsize=(7, 4))
    sns.histplot(df["knn5_peer_adopt_share"].dropna(), bins=30, kde=True)
    plt.title("Distribution of kNN (k=5) Peer Adoption Share")
    plt.xlabel("Peer Adoption Share (k=5)")
    plt.tight_layout()
    plt.savefig(outdir / "hist_knn5_peer_adopt_share.png", dpi=150)
    plt.close()

    # Boxplot of peer share by adopted status
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df["adopted"].astype(int), y=df["knn5_peer_adopt_share"], showfliers=False)
    plt.title("Peer Adoption Share by Adoption Status")
    plt.xlabel("Adopted (0/1)")
    plt.ylabel("Peer Adoption Share (k=5)")
    plt.tight_layout()
    plt.savefig(outdir / "box_knn5_share_by_adopted.png", dpi=150)
    plt.close()

    # Scatter: adoption_count vs peer share (clip extreme counts for readability)
    plt.figure(figsize=(6, 4))
    tmp = df.copy()
    tmp["adoption_count_clipped"] = tmp["adoption_count"].clip(upper=100)
    sns.scatterplot(x="knn5_peer_adopt_share", y="adoption_count_clipped", data=tmp, s=12, alpha=0.5)
    plt.title("Adoption Count (clipped at 100) vs Peer Adoption Share")
    plt.xlabel("Peer Adoption Share (k=5)")
    plt.ylabel("Adoption Count (clipped)")
    plt.tight_layout()
    plt.savefig(outdir / "scatter_count_vs_knn5_share.png", dpi=150)
    plt.close()


def run_baseline_ols(df: pd.DataFrame, outdir: Path) -> None:
    import statsmodels.formula.api as smf

    # Minimal baseline: adopted ~ peer share (k=5) + within_50km share
    # Add optional state fixed effects if state column present
    has_state = "1a. State" in df.columns
    formula = "adopted ~ knn5_peer_adopt_share + within_50km_peer_adopt_share"
    if has_state:
        # Use patsy quoting for special characters in column name
        formula += ' + C(Q("1a. State"))'

    # Drop rows with missing regressors
    reg_df = df[["adopted", "knn5_peer_adopt_share", "within_50km_peer_adopt_share"] + (["1a. State"] if has_state else [])].dropna()

    model = smf.ols(formula=formula, data=reg_df).fit(cov_type="HC1")
    out_txt = outdir / "ols_baseline.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
    print(f"Saved OLS baseline results to {out_txt}")


def main():
    ap = argparse.ArgumentParser(description="Exploratory plots and a baseline OLS for LEA-level peer variables.")
    ap.add_argument("--csv", type=Path, default=Path("lea_peer_vars.csv"), help="Path to lea_peer_vars.csv")
    ap.add_argument("--plots-dir", type=Path, default=Path("plots"), help="Directory for saved figures")
    ap.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory for regression outputs")
    args = ap.parse_args()

    ensure_dirs(args.plots_dir, args.results_dir)
    df = load_lea_vars(args.csv)
    make_plots(df, args.plots_dir)
    run_baseline_ols(df, args.results_dir)


if __name__ == "__main__":
    main()
