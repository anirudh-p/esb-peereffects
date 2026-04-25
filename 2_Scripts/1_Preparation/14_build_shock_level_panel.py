"""
14_build_shock_level_panel.py
==============================
Builds the Borusyak-Hull (2023) shock-level panel for the R1 lottery spillover analysis.

Unit of observation: R1 applicant j x year t.
Outcome: weighted-average adoption rate of j's bystander neighbors (non-R1 applicants).

The key methodological fix over a naive implementation is a HAZARD-CONSISTENT outcome:
the denominator is the time-varying at-risk bystander pool (those not yet adopted as of
t-1), not a static count of all bystanders. This aligns with the discrete-time hazard
design used in scripts 10-12. Including already-adopted bystanders in the denominator
mechanically deflates the outcome in later years and dilutes the post-lottery signal.

Four outcomes are computed:
  Y_bys_cum_awd   — cumulative adoption / static denom  (award date; reference only)
  Y_bys_haz_awd   — flow adoption / at-risk denom       (award date; main outcome)
  Y_bys_cum_opr   — cumulative adoption / static denom  (operating date)
  Y_bys_haz_opr   — flow adoption / at-risk denom       (operating date; mechanism check)

Award date = year_first_awarded (bureaucratic event; consistent with scripts 10-12).
Operating date = year_first_operating (physical delivery; correct timing for peer effects
  working through visible/operational buses or activated vendor networks).

Clean R1 loser pool: IS_R1_LOSER==1 AND did NOT win R2 or R3.
Bystanders: districts NOT in the R1 applicant pool.
Weight matrix: K=6 nearest neighbors (knn_weights_k6.npz).

Output: 1_Data/Cleaned/shock_level_panel_r1.dta
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz


def compute_bystander_outcomes(W, is_bystander, yfa, years, applicant_idx):
    """
    For each applicant j and year t, compute:
      cum:  weighted fraction of ALL bystanders who have adopted by t (stock)
      haz:  weighted fraction of AT-RISK bystanders who adopt in t (flow/hazard)

    Returns arrays of shape (len(applicant_idx), len(years)) for each outcome.
    """
    n_app = len(applicant_idx)
    n_yr  = len(years)

    # Static denominator: total bystander weight for each applicant (used for cum)
    denom_static = np.array(W.T.dot(is_bystander)).flatten()[applicant_idx]

    cum_out = np.full((n_app, n_yr), np.nan)
    haz_out = np.full((n_app, n_yr), np.nan)

    for t_idx, t in enumerate(years):
        # --- Cumulative: adopted any time up to and including t ---
        adopted_by_t = np.where(~np.isnan(yfa) & (yfa <= t), 1.0, 0.0)
        num_cum = np.array(W.T.dot(is_bystander * adopted_by_t)).flatten()[applicant_idx]
        cum_out[:, t_idx] = np.where(denom_static > 0, num_cum / denom_static, np.nan)

        # --- Hazard-consistent: adopts in t / at-risk (not yet adopted as of t-1) ---
        # At-risk in year t = yfa >= t OR never adopted (yfa is NaN)
        at_risk_t  = np.where(np.isnan(yfa) | (yfa >= t), 1.0, 0.0)
        adopted_t  = np.where(~np.isnan(yfa) & (yfa == t),  1.0, 0.0)

        bys_at_risk  = is_bystander * at_risk_t
        bys_adopted  = is_bystander * adopted_t

        denom_haz = np.array(W.T.dot(bys_at_risk)).flatten()[applicant_idx]
        num_haz   = np.array(W.T.dot(bys_adopted)).flatten()[applicant_idx]

        haz_out[:, t_idx] = np.where(denom_haz > 0, num_haz / denom_haz, np.nan)

    return cum_out, haz_out, denom_static


def main():
    base_dir = Path("C:/BC PhD/Research/Peer-Effects and Adoption")

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    W = load_npz(base_dir / "1_Data/Cleaned/knn_weights_k6.npz")

    df_order = pd.read_csv(base_dir / "1_Data/Cleaned/spatial_nces_order.csv",
                           dtype={'nces_id': str})
    nces_order = df_order['nces_id'].values
    N = len(nces_order)
    print(f"W matrix: {N} districts")

    df_base = pd.read_csv(base_dir / "1_Data/Cleaned/analysis_dataset.csv",
                          dtype={'nces_id': str}).set_index('nces_id')

    # ------------------------------------------------------------------
    # Classify districts in W-matrix order
    # ------------------------------------------------------------------
    def get_col(col, default=0):
        return np.array(
            [df_base.loc[n, col]
             if n in df_base.index and pd.notna(df_base.loc[n, col])
             else default
             for n in nces_order], dtype=float)

    is_r1_winner   = get_col('IV_Z_R1')
    is_r1_loser    = get_col('IS_R1_LOSER')
    is_r3_winner   = get_col('IV_Z_R3')
    is_r2_grantee  = get_col('IS_R2_GRANTEE')

    # Clean R1 losers: lost R1, did not subsequently win R2 or R3
    is_clean_r1_loser = ((is_r1_loser == 1) &
                         (is_r3_winner == 0) &
                         (is_r2_grantee == 0)).astype(float)

    is_r1_applicant = np.clip(is_r1_winner + is_clean_r1_loser, 0, 1)
    is_bystander    = (1 - is_r1_applicant).astype(float)

    print(f"R1 winners:      {int(is_r1_winner.sum())}")
    print(f"Clean R1 losers: {int(is_clean_r1_loser.sum())}")
    print(f"Bystanders:      {int(is_bystander.sum())}")

    # Adoption timing vectors (NaN = never adopted within panel)
    def get_year_col(col):
        return np.array(
            [df_base.loc[n, col]
             if n in df_base.index and pd.notna(df_base.loc[n, col])
             else np.nan
             for n in nces_order], dtype=float)

    yfa = get_year_col('year_first_awarded')    # award date
    yfo = get_year_col('year_first_operating')  # operating date

    applicant_idx  = np.where(is_r1_applicant == 1)[0]
    applicant_nces = nces_order[applicant_idx]
    n_app          = len(applicant_idx)
    years          = list(range(2018, 2025))

    print(f"\nComputing outcomes for {n_app} applicants x {len(years)} years...")

    # ------------------------------------------------------------------
    # Compute outcomes: award-date and operating-date versions
    # ------------------------------------------------------------------
    cum_awd, haz_awd, denom_static = compute_bystander_outcomes(
        W, is_bystander, yfa, years, applicant_idx)

    cum_opr, haz_opr, _ = compute_bystander_outcomes(
        W, is_bystander, yfo, years, applicant_idx)

    print("Done computing outcomes.")

    # ------------------------------------------------------------------
    # Assemble panel DataFrame
    # ------------------------------------------------------------------
    records = []
    K = 6
    bystander_n = np.round(denom_static * K).astype(int)

    for t_idx, t in enumerate(years):
        for k in range(n_app):
            j = applicant_idx[k]
            records.append({
                'nces_id':         applicant_nces[k],
                'year':            t,
                'Y_bys_cum_awd':   cum_awd[k, t_idx],
                'Y_bys_haz_awd':   haz_awd[k, t_idx],
                'Y_bys_cum_opr':   cum_opr[k, t_idx],
                'Y_bys_haz_opr':   haz_opr[k, t_idx],
                'won_r1':          int(is_r1_winner[j]),
                'bystander_n':     bystander_n[k],
            })

    df_shock = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Merge applicant-level covariates
    # ------------------------------------------------------------------
    covars = ['priority_r1', 'is_r1_third_party', 'is_r1_loser_third_party',
              'locale_code', 'state']
    avail  = [c for c in covars if c in df_base.columns]
    df_shock = df_shock.merge(
        df_base[avail].reset_index(), on='nces_id', how='left')

    df_shock['applicant_id'] = pd.factorize(df_shock['nces_id'])[0] + 1

    # ------------------------------------------------------------------
    # Drop applicants with no bystander neighbors (denom_static == 0)
    # ------------------------------------------------------------------
    n_before = df_shock['nces_id'].nunique()
    df_shock  = df_shock[df_shock['bystander_n'] > 0]
    n_after   = df_shock['nces_id'].nunique()
    print(f"\nApplicants after dropping isolated nodes: {n_after} (dropped {n_before - n_after})")
    print(f"Panel obs: {len(df_shock)}")
    print(f"  R1 winners: {int((df_shock['won_r1']==1).sum() / len(years))}, "
          f"R1 losers: {int((df_shock['won_r1']==0).sum() / len(years))}")

    # ------------------------------------------------------------------
    # Summary: mean hazard-consistent outcome by year and won_r1
    # ------------------------------------------------------------------
    print("\nMean Y_bys_haz_awd (hazard-consistent, award) by year and R1 win status:")
    s = df_shock.groupby(['year','won_r1'])['Y_bys_haz_awd'].mean().unstack()
    print(s.round(4).to_string())

    print("\nMean Y_bys_haz_opr (hazard-consistent, operating) by year and R1 win status:")
    s2 = df_shock.groupby(['year','won_r1'])['Y_bys_haz_opr'].mean().unstack()
    print(s2.round(4).to_string())

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    out_path = base_dir / "1_Data/Cleaned/shock_level_panel_r1.dta"
    df_shock.to_stata(str(out_path), write_index=False, version=118)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
