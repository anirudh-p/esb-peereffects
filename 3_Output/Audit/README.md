# Audit

Diagnostics, merge checks, sample counts, lottery/risk-set validation, and exposure-construction checks.

Important current diagnostics:

- `raw_file_inventory.csv` and `raw_table_inventory.csv`: raw source inventory.
- `analysis_dataset_build_diagnostics.csv`: district-base build counts.
- `hazard_panel_build_diagnostics.csv` and `hazard_panel_year_summary.csv`: annual panel counts.
- `spatial_geometry_source_summary.csv`: EDGE shapefile versus WRI point fallback versus no-geometry counts.
- `spatial_unmatched_lea_type_summary.csv`: type composition of LEAs not found in EDGE shapefiles.
- `spatial_unmatched_interesting_leas.csv`: unmatched LEAs with applications, awards, ESBs, grants, or survey responses.
- `spatial_panel_sample_stages.csv`: panel sample size under all, EDGE-only, hybrid, and core-control restrictions.
- `r1_design_bh_priority_rates.csv`: priority-group selection rates used for the initial design-BH expected exposure.
