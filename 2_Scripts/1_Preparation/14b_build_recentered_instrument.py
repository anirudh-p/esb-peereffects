"""
14b_build_recentered_instrument.py
====================================
Constructs the BH-recentered instrument for the district-level IV (scripts 10-11).

The standard instrument w6_Z_t_minus_1 = W @ is_r1_winner is non-randomly assigned
because geographic neighbors are endogenous: districts with more R1-winning neighbors
tend to be in high-adoption areas (the pre-trend problem in script 12).

BH fix: recenter by subtracting each district's EXPECTED exposure given its applicant
neighborhood composition and stratum-specific win probabilities:

    Z_recentered_i = w6_Z_i - E[w6_Z_i | applicant pool]
                   = w6_Z_i - (p_priority * w6_Z_app_priority_i
                               + p_nonpriority * w6_Z_app_nonpriority_i)

Where:
    p_priority    = 349 / 1227 = 0.2844  (R1 priority stratum win rate)
    p_nonpriority =  16 /  660 = 0.0242  (R1 non-priority stratum win rate)

Under this construction Z_recentered is orthogonal to expected exposure by design,
regardless of how applicants sort geographically — the BH guarantee.

New columns added to the panel dataset:
    w6_Z_app_priority     — W @ is_r1_priority_applicant  (static)
    w6_Z_app_nonpriority  — W @ is_r1_nonpriority_applicant  (static)

The recentered instrument itself is constructed in Stata (16_bh_recentered_iv.do)
as a simple linear combination of these lags, so the exact win-rate constants are
transparent and auditable in the do file.

Output: overwrites 1_Data/Cleaned/analysis_panel_dataset_spatial.dta with two
additional columns (all other columns unchanged).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz


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
    # Stratum membership for each district in W-matrix order
    # ------------------------------------------------------------------
    def get_flag(col, default=0):
        return np.array(
            [df_base.loc[n, col]
             if n in df_base.index and pd.notna(df_base.loc[n, col])
             else default
             for n in nces_order], dtype=float)

    is_r1_winner  = get_flag('IV_Z_R1')
    is_r1_loser   = get_flag('IS_R1_LOSER')
    priority_r1   = get_flag('priority_r1')   # 1 = priority, 0 = non-priority, NaN for non-applicants

    is_r1_applicant     = np.clip(is_r1_winner + is_r1_loser, 0, 1)
    is_r1_priority      = (is_r1_applicant * (priority_r1 == 1)).astype(float)
    is_r1_nonpriority   = (is_r1_applicant * (priority_r1 == 0)).astype(float)

    print(f"Priority applicants:     {int(is_r1_priority.sum())} "
          f"(expected ~1227 in spatial sample)")
    print(f"Non-priority applicants: {int(is_r1_nonpriority.sum())} "
          f"(expected ~660 in spatial sample)")

    # Verify win rates match base-dataset counts
    winners_in_priority    = get_flag('IV_Z_R1') * (priority_r1 == 1)
    winners_in_nonpriority = get_flag('IV_Z_R1') * (priority_r1 == 0)
    p1 = winners_in_priority.sum() / is_r1_priority.sum()
    p0 = winners_in_nonpriority.sum() / is_r1_nonpriority.sum()
    print(f"\nWin rates in spatial sample:")
    print(f"  Priority:     {winners_in_priority.sum():.0f} / {is_r1_priority.sum():.0f} = {p1:.4f}")
    print(f"  Non-priority: {winners_in_nonpriority.sum():.0f} / {is_r1_nonpriority.sum():.0f} = {p0:.4f}")
    print(f"\nThese constants go into Stata as:")
    print(f"  p_priority    = {p1:.6f}")
    print(f"  p_nonpriority = {p0:.6f}")

    # ------------------------------------------------------------------
    # Compute stratum-specific spatial lags (static — no year loop needed)
    # ------------------------------------------------------------------
    w6_Z_app_priority    = np.array(W.dot(is_r1_priority)).flatten()
    w6_Z_app_nonpriority = np.array(W.dot(is_r1_nonpriority)).flatten()

    # Sanity check: sum should equal w6_Z_app_R1 (total applicant lag)
    w6_Z_app_total = w6_Z_app_priority + w6_Z_app_nonpriority
    is_r1_applicant_all = np.clip(is_r1_winner + is_r1_loser, 0, 1)
    w6_Z_app_check = np.array(W.dot(is_r1_applicant_all)).flatten()
    max_diff = np.abs(w6_Z_app_total - w6_Z_app_check).max()
    print(f"\nSanity check — max diff (priority + nonpriority) vs total: {max_diff:.2e}")

    # Build a cross-section DataFrame (one row per spatial district)
    df_lags = pd.DataFrame({
        'nces_id':             nces_order,
        'w6_Z_app_priority':   w6_Z_app_priority,
        'w6_Z_app_nonpriority': w6_Z_app_nonpriority,
    })

    print(f"\nw6_Z_app_priority    — mean: {w6_Z_app_priority.mean():.3f}, "
          f"max: {w6_Z_app_priority.max():.1f}")
    print(f"w6_Z_app_nonpriority — mean: {w6_Z_app_nonpriority.mean():.3f}, "
          f"max: {w6_Z_app_nonpriority.max():.1f}")

    # ------------------------------------------------------------------
    # Merge into panel dataset and export
    # ------------------------------------------------------------------
    panel_path = base_dir / "1_Data/Cleaned/analysis_panel_dataset_spatial.dta"
    print(f"\nLoading panel from {panel_path} ...")
    df_panel = pd.read_stata(str(panel_path))
    df_panel['nces_id'] = df_panel['nces_id'].astype(str).str.zfill(7)

    # Drop if already present (re-run safety)
    for col in ['w6_Z_app_priority', 'w6_Z_app_nonpriority']:
        if col in df_panel.columns:
            df_panel = df_panel.drop(columns=[col])

    df_panel = df_panel.merge(df_lags, on='nces_id', how='left')

    n_matched = df_panel['w6_Z_app_priority'].notna().sum()
    print(f"Panel rows with new lags matched: {n_matched} / {len(df_panel)}")

    df_panel.to_stata(str(panel_path), write_index=False, version=118)
    print(f"Saved updated panel to {panel_path}")
    print(f"New columns: w6_Z_app_priority, w6_Z_app_nonpriority")


if __name__ == "__main__":
    main()
