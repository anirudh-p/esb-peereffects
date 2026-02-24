"""
Contradiction Decomposition: Script 01 vs Script 04
=====================================================
Systematically decomposes WHY Script 01 finds geographic peer effects
while Script 04 finds climate peer effects, by progressively changing
one element at a time.

Key discovery: Script 01's "w_clim" is a geographic donut (KNN-20 minus
KNN-6, ranked by physical distance). Script 04's "w_clim" is true
climate-similarity KNN from PRISM precipitation/temperature data.
They measure fundamentally different things.

Specification matrix (for each K):
  S1: Simple controls, no FE, no clustering, climate-restricted, true climate KNN  [= Script 04]
  S2: Full controls, no FE, no clustering, climate-restricted, true climate KNN
  S3: Full controls, state FE, no clustering, climate-restricted, true climate KNN
  S4: Full controls, state FE, state clustering, climate-restricted, true climate KNN  [KEY]
  S5: Full controls, state FE, state clustering, FULL sample, geo donut              [= Script 01]
  S6: Full controls, state FE, state clustering, climate-restricted, geo donut

Interpretation:
  S1->S2: Effect of richer controls (is_priority, pct_dem_2020, urbanicity)
  S2->S3: Effect of state fixed effects
  S3->S4: Effect of clustering (coefficients unchanged, SEs change)
  S4 vs S6: Effect of w_clim definition (true climate vs geo donut), same sample
  S5 vs S6: Effect of sample restriction (full vs climate-restricted)
"""

import pandas as pd
import numpy as np
import warnings
import os
import sys
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from _estimation_utils import (
    load_analysis_data, load_shapefile_and_merge,
    build_geo_knn, build_geo_donut, load_and_align_climate_weights,
    create_spatial_lags, run_iv2sls, format_result_row,
    save_results_csv, print_results_table, significance_stars,
    FULL_CONTROLS, SIMPLE_CONTROLS
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TABLES_DIR, CLIMATE_KNN_VALUES, ensure_dirs_exist

ensure_dirs_exist()

OUTPUT_FILE = TABLES_DIR / 'contradiction_decomposition.csv'

# K values to test
K_VALUES = [6, 10]

print("=" * 80)
print("CONTRADICTION DECOMPOSITION: SCRIPT 01 vs SCRIPT 04")
print("=" * 80)

# =============================================================================
# 1. Load Data
# =============================================================================
df, locale_vars = load_analysis_data()
geo_df = load_shapefile_and_merge(df)

print(f"\nFull sample: {len(geo_df)} districts")
print(f"Adopters: {geo_df['IS_ADOPTER'].sum()} ({100*geo_df['IS_ADOPTER'].mean():.1f}%)")

results_list = []

for k in K_VALUES:
    print(f"\n{'='*80}")
    print(f"K = {k}")
    print(f"{'='*80}")

    # -----------------------------------------------------------------
    # Build weight matrices
    # -----------------------------------------------------------------

    # Geographic KNN-k on full sample
    print(f"\nBuilding geographic KNN-{k} (full sample)...")
    w_geo_full = build_geo_knn(geo_df, k)

    # Geographic donut (Script 01 approach): KNN-(k+14) minus KNN-k
    # At k=6 this gives neighbors 7-20, matching Script 01 exactly
    k_outer = k + 14
    print(f"Building geographic donut KNN-{k_outer} minus KNN-{k} (full sample)...")
    w_donut_full = build_geo_donut(geo_df, k_inner=k, k_outer=k_outer)

    # True climate KNN-k (restricts sample)
    print(f"Loading climate KNN-{k} and aligning sample...")
    geo_df_clim, w_geo_clim, w_clim_true = load_and_align_climate_weights(geo_df, k)

    # Geo donut on climate-restricted sample
    print(f"Building geographic donut on climate-restricted sample...")
    w_donut_clim = build_geo_donut(geo_df_clim, k_inner=k, k_outer=k_outer)

    print(f"\nFull sample: {len(geo_df)} | Climate-restricted: {len(geo_df_clim)}")

    # -----------------------------------------------------------------
    # Create spatial lags for each combination
    # -----------------------------------------------------------------

    # Full sample with geo donut
    geo_df_full_lags = create_spatial_lags(geo_df, w_geo_full, w_donut_full)

    # Climate-restricted with true climate
    geo_df_clim_true = create_spatial_lags(geo_df_clim, w_geo_clim, w_clim_true)

    # Climate-restricted with geo donut
    geo_df_clim_donut = create_spatial_lags(geo_df_clim, w_geo_clim, w_donut_clim)

    # -----------------------------------------------------------------
    # S1: Simple controls, no FE, no clustering, climate-restricted, true climate
    # -----------------------------------------------------------------
    print(f"\n--- S1: Simple controls, no FE, no clustering (replicates Script 04) ---")
    res_s1 = run_iv2sls(
        geo_df_clim_true, y_name='IS_ADOPTER',
        endog_names=['w_geo_adoption', 'w_clim_adoption'],
        instrument_names=['w_geo_z', 'w_clim_z'],
        control_names=SIMPLE_CONTROLS, locale_vars=[],
        state_fe=False, cluster_var=None, include_own_z=False
    )
    row = format_result_row(f'S1_K{k}', 'Climate-Restricted', res_s1)
    row.update({'spec_id': 'S1', 'k': k, 'controls': 'Simple', 'state_fe': False,
                'clustering': 'None', 'w_clim_type': 'true_climate'})
    results_list.append(row)
    print(f"  Geo: {res_s1['params']['w_geo_adoption']:.4f} "
          f"(p={res_s1['pvalues']['w_geo_adoption']:.4f})")
    print(f"  Clim: {res_s1['params']['w_clim_adoption']:.4f} "
          f"(p={res_s1['pvalues']['w_clim_adoption']:.4f})")

    # -----------------------------------------------------------------
    # S2: Full controls, no FE, no clustering, climate-restricted, true climate
    # -----------------------------------------------------------------
    print(f"\n--- S2: Full controls, no FE, no clustering ---")
    res_s2 = run_iv2sls(
        geo_df_clim_true, y_name='IS_ADOPTER',
        endog_names=['w_geo_adoption', 'w_clim_adoption'],
        instrument_names=['w_geo_z', 'w_clim_z'],
        control_names=FULL_CONTROLS, locale_vars=locale_vars,
        state_fe=False, cluster_var=None, include_own_z=False
    )
    row = format_result_row(f'S2_K{k}', 'Climate-Restricted', res_s2)
    row.update({'spec_id': 'S2', 'k': k, 'controls': 'Full', 'state_fe': False,
                'clustering': 'None', 'w_clim_type': 'true_climate'})
    results_list.append(row)
    print(f"  Geo: {res_s2['params']['w_geo_adoption']:.4f} "
          f"(p={res_s2['pvalues']['w_geo_adoption']:.4f})")
    print(f"  Clim: {res_s2['params']['w_clim_adoption']:.4f} "
          f"(p={res_s2['pvalues']['w_clim_adoption']:.4f})")

    # -----------------------------------------------------------------
    # S3: Full controls, state FE, no clustering, climate-restricted, true climate
    # -----------------------------------------------------------------
    print(f"\n--- S3: Full controls, state FE, no clustering ---")
    res_s3 = run_iv2sls(
        geo_df_clim_true, y_name='IS_ADOPTER',
        endog_names=['w_geo_adoption', 'w_clim_adoption'],
        instrument_names=['w_geo_z', 'w_clim_z'],
        control_names=FULL_CONTROLS, locale_vars=locale_vars,
        state_fe=True, cluster_var=None, include_own_z=False
    )
    row = format_result_row(f'S3_K{k}', 'Climate-Restricted', res_s3)
    row.update({'spec_id': 'S3', 'k': k, 'controls': 'Full', 'state_fe': True,
                'clustering': 'None', 'w_clim_type': 'true_climate'})
    results_list.append(row)
    print(f"  Geo: {res_s3['params']['w_geo_adoption']:.4f} "
          f"(p={res_s3['pvalues']['w_geo_adoption']:.4f})")
    print(f"  Clim: {res_s3['params']['w_clim_adoption']:.4f} "
          f"(p={res_s3['pvalues']['w_clim_adoption']:.4f})")

    # -----------------------------------------------------------------
    # S4: Full controls, state FE, state clustering, climate-restricted, true climate
    # -----------------------------------------------------------------
    print(f"\n--- S4: Full controls, state FE, state clustering [KEY TEST] ---")
    res_s4 = run_iv2sls(
        geo_df_clim_true, y_name='IS_ADOPTER',
        endog_names=['w_geo_adoption', 'w_clim_adoption'],
        instrument_names=['w_geo_z', 'w_clim_z'],
        control_names=FULL_CONTROLS, locale_vars=locale_vars,
        state_fe=True, cluster_var='state', include_own_z=False
    )
    row = format_result_row(f'S4_K{k}', 'Climate-Restricted', res_s4)
    row.update({'spec_id': 'S4', 'k': k, 'controls': 'Full', 'state_fe': True,
                'clustering': 'State', 'w_clim_type': 'true_climate'})
    results_list.append(row)
    print(f"  Geo: {res_s4['params']['w_geo_adoption']:.4f} "
          f"(p={res_s4['pvalues']['w_geo_adoption']:.4f})")
    print(f"  Clim: {res_s4['params']['w_clim_adoption']:.4f} "
          f"(p={res_s4['pvalues']['w_clim_adoption']:.4f})")

    # -----------------------------------------------------------------
    # S5: Full controls, state FE, state clustering, FULL sample, geo donut
    # -----------------------------------------------------------------
    print(f"\n--- S5: Full controls, state FE, clustering, FULL sample, geo donut [~Script 01] ---")
    res_s5 = run_iv2sls(
        geo_df_full_lags, y_name='IS_ADOPTER',
        endog_names=['w_geo_adoption', 'w_clim_adoption'],
        instrument_names=['w_geo_z', 'w_clim_z'],
        control_names=FULL_CONTROLS, locale_vars=locale_vars,
        state_fe=True, cluster_var='state', include_own_z=False
    )
    row = format_result_row(f'S5_K{k}', 'Full', res_s5)
    row.update({'spec_id': 'S5', 'k': k, 'controls': 'Full', 'state_fe': True,
                'clustering': 'State', 'w_clim_type': 'geo_donut'})
    results_list.append(row)
    print(f"  Geo: {res_s5['params']['w_geo_adoption']:.4f} "
          f"(p={res_s5['pvalues']['w_geo_adoption']:.4f})")
    print(f"  Clim(donut): {res_s5['params']['w_clim_adoption']:.4f} "
          f"(p={res_s5['pvalues']['w_clim_adoption']:.4f})")

    # -----------------------------------------------------------------
    # S6: Full controls, state FE, state clustering, climate-restricted, geo donut
    # -----------------------------------------------------------------
    print(f"\n--- S6: Full controls, state FE, clustering, climate-restricted, geo donut ---")
    res_s6 = run_iv2sls(
        geo_df_clim_donut, y_name='IS_ADOPTER',
        endog_names=['w_geo_adoption', 'w_clim_adoption'],
        instrument_names=['w_geo_z', 'w_clim_z'],
        control_names=FULL_CONTROLS, locale_vars=locale_vars,
        state_fe=True, cluster_var='state', include_own_z=False
    )
    row = format_result_row(f'S6_K{k}', 'Climate-Restricted', res_s6)
    row.update({'spec_id': 'S6', 'k': k, 'controls': 'Full', 'state_fe': True,
                'clustering': 'State', 'w_clim_type': 'geo_donut'})
    results_list.append(row)
    print(f"  Geo: {res_s6['params']['w_geo_adoption']:.4f} "
          f"(p={res_s6['pvalues']['w_geo_adoption']:.4f})")
    print(f"  Clim(donut): {res_s6['params']['w_clim_adoption']:.4f} "
          f"(p={res_s6['pvalues']['w_clim_adoption']:.4f})")

    # -----------------------------------------------------------------
    # Summary for this K
    # -----------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"DECOMPOSITION SUMMARY (K={k})")
    print(f"{'='*80}")
    print(f"{'Spec':<6} {'Controls':<8} {'FE':<4} {'Clust':<6} {'Sample':<18} {'W_clim':<14} "
          f"{'Geo_coef':>10} {'Geo_p':>8} {'Clim_coef':>10} {'Clim_p':>8}")
    print("-" * 100)

    specs = [
        ('S1', 'Simple', 'No', 'No', 'Clim-Restr', 'Climate', res_s1),
        ('S2', 'Full', 'No', 'No', 'Clim-Restr', 'Climate', res_s2),
        ('S3', 'Full', 'Yes', 'No', 'Clim-Restr', 'Climate', res_s3),
        ('S4', 'Full', 'Yes', 'State', 'Clim-Restr', 'Climate', res_s4),
        ('S5', 'Full', 'Yes', 'State', 'Full', 'Geo-Donut', res_s5),
        ('S6', 'Full', 'Yes', 'State', 'Clim-Restr', 'Geo-Donut', res_s6),
    ]

    for sid, ctrl, fe, clust, samp, wtype, res in specs:
        geo_c = res['params']['w_geo_adoption']
        geo_p = res['pvalues']['w_geo_adoption']
        clim_c = res['params']['w_clim_adoption']
        clim_p = res['pvalues']['w_clim_adoption']
        geo_star = significance_stars(geo_p)
        clim_star = significance_stars(clim_p)
        print(f"{sid:<6} {ctrl:<8} {fe:<4} {clust:<6} {samp:<18} {wtype:<14} "
              f"{geo_c:>9.4f}{geo_star:<1} {geo_p:>8.4f} "
              f"{clim_c:>9.4f}{clim_star:<1} {clim_p:>8.4f}")

# =============================================================================
# Save All Results
# =============================================================================
results_df = save_results_csv(results_list, OUTPUT_FILE)

print(f"\n{'='*80}")
print("INTERPRETATION GUIDE")
print(f"{'='*80}")
print("""
S1 -> S2: Do richer controls (is_priority, pct_dem_2020, urbanicity) change results?
S2 -> S3: Does adding state FE absorb the climate peer effect?
S3 -> S4: Does clustering inflate SEs enough to change significance?
S4 vs S5: True climate vs geo-donut (different definition + different sample)
S4 vs S6: True climate vs geo-donut (same sample — isolates definition effect)
S5 vs S6: Full vs climate-restricted sample (same definition — isolates sample effect)

If S2->S3 kills climate effect: "climate peers" proxy for state-level policies
If S4 vs S6 shows big difference: the definition of w_clim matters fundamentally
If S5 vs S6 shows small difference: sample restriction is not driving results
""")

print("\nDONE.")
