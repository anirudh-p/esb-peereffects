"""
Standardized Estimation: Canonical Results
============================================
Unified estimation script that supersedes Scripts 01-05 for
producing final paper-quality results.

All specifications use:
  - Full controls (log_income, poverty, log_enrollment, pm25, pct_white,
    pct_dem_2020, is_priority) + urbanicity dummies + own IV_Z
  - State fixed effects
  - Standard errors clustered by state

Panels:
  A: Geographic peers only (single endogenous, full + climate-restricted samples)
  B: True climate peers only (single endogenous, climate-restricted sample)
  C: Horse race: geographic + climate (two endogenous, climate-restricted sample)
  D: Distance bands (robustness, full sample)
  E: No charters (robustness, repeat A and C)
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
    build_geo_knn, load_and_align_climate_weights,
    create_spatial_lags, run_iv2sls, format_result_row,
    save_results_csv, print_results_table,
    FULL_CONTROLS
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    TABLES_DIR, CLIMATE_KNN_VALUES, ensure_dirs_exist,
    CHARTER_FLAGS
)
from libpysal.weights import DistanceBand

ensure_dirs_exist()

# Output files
OUTPUT_GEO = TABLES_DIR / 'standardized_geo_results.csv'
OUTPUT_CLIM = TABLES_DIR / 'standardized_climate_results.csv'
OUTPUT_HORSE = TABLES_DIR / 'standardized_horserace_results.csv'
OUTPUT_SUMMARY = TABLES_DIR / 'standardized_estimation_summary.txt'

K_VALUES = CLIMATE_KNN_VALUES  # [4, 6, 8, 10, 12, 15, 20]

print("=" * 80)
print("STANDARDIZED ESTIMATION: CANONICAL RESULTS")
print("=" * 80)

# =============================================================================
# 1. Load Data
# =============================================================================
df, locale_vars = load_analysis_data()

# Add charter flag if not present
if 'is_charter' not in df.columns:
    try:
        charter_flags = pd.read_csv(str(CHARTER_FLAGS))
        charter_flags['nces_id'] = charter_flags['nces_id'].astype(str).str.zfill(7)
        df = df.merge(charter_flags[['nces_id', 'is_charter']], on='nces_id', how='left')
        df['is_charter'] = df['is_charter'].fillna(0).astype(int)
    except FileNotFoundError:
        print("  WARNING: Charter flags file not found, setting all to 0")
        df['is_charter'] = 0

geo_df = load_shapefile_and_merge(df)

print(f"\nFull sample: {len(geo_df)} districts")
print(f"  Adopters: {geo_df['IS_ADOPTER'].sum()} ({100*geo_df['IS_ADOPTER'].mean():.1f}%)")
print(f"  Charters: {geo_df['is_charter'].sum()}")

# =============================================================================
# Panel A: Geographic Peers Only
# =============================================================================
print(f"\n{'='*80}")
print("PANEL A: GEOGRAPHIC PEERS ONLY")
print(f"{'='*80}")

results_geo = []

for k in K_VALUES:
    print(f"\n--- KNN-{k} ---")

    # Full sample
    w_geo = build_geo_knn(geo_df, k)
    geo_lags = create_spatial_lags(geo_df, w_geo, w_clim=None)

    res = run_iv2sls(
        geo_lags, y_name='IS_ADOPTER',
        endog_names=['w_geo_adoption'],
        instrument_names=['w_geo_z'],
        control_names=FULL_CONTROLS, locale_vars=locale_vars,
        state_fe=True, cluster_var='state'
    )
    row = format_result_row(f'KNN-{k}', 'Full', res)
    row['k'] = k
    results_geo.append(row)
    print(f"  Full (N={res['n']}): geo={res['params']['w_geo_adoption']:.4f} "
          f"(p={res['pvalues']['w_geo_adoption']:.4f}), "
          f"F={res['f_stats']['w_geo_adoption']:.0f}")

    # Climate-restricted sample
    geo_clim, w_geo_clim, w_clim = load_and_align_climate_weights(geo_df, k)
    geo_clim_lags = create_spatial_lags(geo_clim, w_geo_clim, w_clim=None)

    res_cr = run_iv2sls(
        geo_clim_lags, y_name='IS_ADOPTER',
        endog_names=['w_geo_adoption'],
        instrument_names=['w_geo_z'],
        control_names=FULL_CONTROLS, locale_vars=locale_vars,
        state_fe=True, cluster_var='state'
    )
    row_cr = format_result_row(f'KNN-{k}', 'Climate-Restricted', res_cr)
    row_cr['k'] = k
    results_geo.append(row_cr)
    print(f"  Clim-Restr (N={res_cr['n']}): geo={res_cr['params']['w_geo_adoption']:.4f} "
          f"(p={res_cr['pvalues']['w_geo_adoption']:.4f})")

save_results_csv(results_geo, OUTPUT_GEO)

# =============================================================================
# Panel B: True Climate Peers Only
# =============================================================================
print(f"\n{'='*80}")
print("PANEL B: TRUE CLIMATE PEERS ONLY")
print(f"{'='*80}")

results_clim = []

for k in K_VALUES:
    print(f"\n--- KNN-{k} ---")

    geo_clim, w_geo_clim, w_clim = load_and_align_climate_weights(geo_df, k)
    # Use climate weights for both adoption lag and instrument
    geo_clim_lags = create_spatial_lags(geo_clim, w_clim, w_clim=None)
    # Rename columns: w_geo_* -> w_clim_* for clarity
    geo_clim_lags = geo_clim_lags.rename(columns={
        'w_geo_adoption': 'w_clim_adoption',
        'w_geo_z': 'w_clim_z'
    })

    res = run_iv2sls(
        geo_clim_lags, y_name='IS_ADOPTER',
        endog_names=['w_clim_adoption'],
        instrument_names=['w_clim_z'],
        control_names=FULL_CONTROLS, locale_vars=locale_vars,
        state_fe=True, cluster_var='state'
    )
    row = format_result_row(f'KNN-{k}', 'Climate-Restricted', res)
    row['k'] = k
    results_clim.append(row)
    print(f"  N={res['n']}: clim={res['params']['w_clim_adoption']:.4f} "
          f"(p={res['pvalues']['w_clim_adoption']:.4f}), "
          f"F={res['f_stats']['w_clim_adoption']:.0f}")

save_results_csv(results_clim, OUTPUT_CLIM)

# =============================================================================
# Panel C: Horse Race (Geographic + Climate)
# =============================================================================
print(f"\n{'='*80}")
print("PANEL C: HORSE RACE (GEOGRAPHIC + CLIMATE)")
print(f"{'='*80}")

results_horse = []

for k in K_VALUES:
    print(f"\n--- KNN-{k} ---")

    geo_clim, w_geo_clim, w_clim = load_and_align_climate_weights(geo_df, k)
    geo_clim_lags = create_spatial_lags(geo_clim, w_geo_clim, w_clim)

    res = run_iv2sls(
        geo_clim_lags, y_name='IS_ADOPTER',
        endog_names=['w_geo_adoption', 'w_clim_adoption'],
        instrument_names=['w_geo_z', 'w_clim_z'],
        control_names=FULL_CONTROLS, locale_vars=locale_vars,
        state_fe=True, cluster_var='state'
    )
    row = format_result_row(f'KNN-{k}', 'Climate-Restricted', res)
    row['k'] = k
    results_horse.append(row)

    geo_c = res['params']['w_geo_adoption']
    geo_p = res['pvalues']['w_geo_adoption']
    clim_c = res['params']['w_clim_adoption']
    clim_p = res['pvalues']['w_clim_adoption']
    print(f"  N={res['n']}: geo={geo_c:.4f} (p={geo_p:.4f}), "
          f"clim={clim_c:.4f} (p={clim_p:.4f})")

save_results_csv(results_horse, TABLES_DIR / 'standardized_horserace_results.csv')

# =============================================================================
# Panel D: Distance Bands (Robustness)
# =============================================================================
print(f"\n{'='*80}")
print("PANEL D: DISTANCE BANDS")
print(f"{'='*80}")

results_dist = []
geo_points = geo_df.set_geometry('centroid')

for d_km in [25, 50, 75, 100]:
    print(f"\n--- {d_km}km ---")
    threshold_m = d_km * 1000

    try:
        w_dist = DistanceBand.from_dataframe(geo_points, threshold=threshold_m, binary=False)
        w_dist.transform = 'r'

        geo_dist_lags = create_spatial_lags(geo_df, w_dist, w_clim=None)

        res = run_iv2sls(
            geo_dist_lags, y_name='IS_ADOPTER',
            endog_names=['w_geo_adoption'],
            instrument_names=['w_geo_z'],
            control_names=FULL_CONTROLS, locale_vars=locale_vars,
            state_fe=True, cluster_var='state'
        )
        row = format_result_row(f'Dist-{d_km}km', 'Full', res)
        row['distance_km'] = d_km
        results_dist.append(row)
        print(f"  N={res['n']}: geo={res['params']['w_geo_adoption']:.4f} "
              f"(p={res['pvalues']['w_geo_adoption']:.4f})")
    except Exception as e:
        print(f"  ERROR: {e}")

# =============================================================================
# Panel E: No Charters (Robustness)
# =============================================================================
print(f"\n{'='*80}")
print("PANEL E: NO CHARTERS")
print(f"{'='*80}")

results_nocharter = []
geo_df_nc = geo_df[geo_df['is_charter'] == 0].reset_index(drop=True)
print(f"No-charter sample: {len(geo_df_nc)} districts")

for k in K_VALUES:
    print(f"\n--- KNN-{k} (No Charters) ---")

    # Geographic only
    w_geo_nc = build_geo_knn(geo_df_nc, k)
    geo_nc_lags = create_spatial_lags(geo_df_nc, w_geo_nc, w_clim=None)

    res = run_iv2sls(
        geo_nc_lags, y_name='IS_ADOPTER',
        endog_names=['w_geo_adoption'],
        instrument_names=['w_geo_z'],
        control_names=FULL_CONTROLS, locale_vars=locale_vars,
        state_fe=True, cluster_var='state'
    )
    row = format_result_row(f'KNN-{k}', 'No Charters (Geo Only)', res)
    row['k'] = k
    results_nocharter.append(row)
    print(f"  Geo only (N={res['n']}): geo={res['params']['w_geo_adoption']:.4f} "
          f"(p={res['pvalues']['w_geo_adoption']:.4f})")

    # Horse race (climate-restricted, no charters)
    try:
        geo_nc_clim, w_geo_nc_clim, w_clim_nc = load_and_align_climate_weights(geo_df_nc, k)
        geo_nc_clim_lags = create_spatial_lags(geo_nc_clim, w_geo_nc_clim, w_clim_nc)

        res_hr = run_iv2sls(
            geo_nc_clim_lags, y_name='IS_ADOPTER',
            endog_names=['w_geo_adoption', 'w_clim_adoption'],
            instrument_names=['w_geo_z', 'w_clim_z'],
            control_names=FULL_CONTROLS, locale_vars=locale_vars,
            state_fe=True, cluster_var='state'
        )
        row_hr = format_result_row(f'KNN-{k}', 'No Charters (Horse Race)', res_hr)
        row_hr['k'] = k
        results_nocharter.append(row_hr)

        geo_c = res_hr['params']['w_geo_adoption']
        clim_c = res_hr['params']['w_clim_adoption']
        print(f"  Horse race (N={res_hr['n']}): geo={geo_c:.4f}, clim={clim_c:.4f}")
    except Exception as e:
        print(f"  Horse race ERROR: {e}")

# =============================================================================
# Save All Results + Summary
# =============================================================================
all_results = results_geo + results_clim + results_horse + results_dist + results_nocharter
all_df = pd.DataFrame(all_results)
all_df.to_csv(str(TABLES_DIR / 'standardized_all_results.csv'), index=False)

# Write formatted summary
with open(str(OUTPUT_SUMMARY), 'w') as f:
    f.write("STANDARDIZED ESTIMATION RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write("All specifications use:\n")
    f.write("  Controls: log_income, poverty, log_enrollment, pm25, pct_white,\n")
    f.write("            pct_dem_2020, is_priority + urbanicity dummies + own IV_Z\n")
    f.write("  State FE: Yes\n")
    f.write("  Clustering: By state\n\n")

    f.write("=" * 80 + "\n")
    f.write("PANEL A: GEOGRAPHIC PEERS ONLY\n")
    f.write("=" * 80 + "\n")
    f.write(pd.DataFrame(results_geo).to_string(index=False))
    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("PANEL B: TRUE CLIMATE PEERS ONLY\n")
    f.write("=" * 80 + "\n")
    f.write(pd.DataFrame(results_clim).to_string(index=False))
    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("PANEL C: HORSE RACE (GEOGRAPHIC + CLIMATE)\n")
    f.write("=" * 80 + "\n")
    f.write(pd.DataFrame(results_horse).to_string(index=False))
    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("PANEL D: DISTANCE BANDS\n")
    f.write("=" * 80 + "\n")
    if results_dist:
        f.write(pd.DataFrame(results_dist).to_string(index=False))
    else:
        f.write("No distance band results.\n")
    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("PANEL E: NO CHARTERS\n")
    f.write("=" * 80 + "\n")
    f.write(pd.DataFrame(results_nocharter).to_string(index=False))
    f.write("\n")

print(f"\nSummary saved to {OUTPUT_SUMMARY}")
print(f"All results saved to {TABLES_DIR / 'standardized_all_results.csv'}")
print("\nDONE.")
