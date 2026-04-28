# Cleaned Data

Script-created datasets belong here.

## Current Outputs

From `2_Scripts/1_Preparation/01_build_analysis_dataset.py`:

- `analysis_district_base.csv`: one row per standardized NCES district ID in the source crosswalk.
- `analysis_district_base.dta`: Stata-ready mirror of the base district file.
- `district_identifier_crosswalk.csv`: one row per NCES district ID, with canonical district name/state/city and source-presence flags.
- `district_source_records.csv`: long source-record file used to audit identifiers across WRI, CSBP, bus-level, and survey sources.

From `2_Scripts/1_Preparation/02_build_panel_dataset.py`:

- `analysis_hazard_panel.csv`: annual district-year hazard panel, currently 2018-2024.
- `analysis_hazard_panel.dta`: Stata-ready mirror of the hazard panel.

From `2_Scripts/1_Preparation/03_build_spatial_design_exposure.py`:

- `analysis_hazard_panel_spatial.csv`: full annual panel with hybrid KNN spatial exposures, EDGE-only K=6 sensitivity exposures, and design-BH expected/recentered R1 exposure.
- `analysis_hazard_panel_spatial.dta`: lean Stata-ready version focused on the primary K=6 estimation surface.
- `knn_weights_hybrid_k*.npz`: hybrid KNN matrices using EDGE centroids plus WRI point-coordinate fallback.
- `knn_weights_edge_k6.npz`: EDGE-only K=6 matrix for sample-drop sensitivity checks.
- `spatial_neighbor_order_*.csv`: row orders for the saved KNN matrices.

This folder is ignored by Git except for this README. Rebuild generated files from scripts rather than editing them manually.
