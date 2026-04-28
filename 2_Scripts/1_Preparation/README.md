# 1_Preparation

Raw-to-cleaned data construction.

## Included Scripts

- `01_build_analysis_dataset.py`: inventories raw files/tables, standardizes NCES district IDs across WRI/CSBP/survey sources, writes a district identifier crosswalk, and creates the district-level base analysis file.
- `02_build_panel_dataset.py`: expands the base district data into the annual hazard-panel structure and constructs first-event outcomes, risk sets, treatment timing, own-winner exclusion flags, and sample-coverage flags.
- `03_build_spatial_design_exposure.py`: builds geometry coverage diagnostics, focal-unit/sample flags, hybrid and EDGE-only KNN graphs, spatial exposure variables, and priority-BH expected/recentered R1 design exposure.
- `04_export_stata_inputs.py`: validates that Stata-ready `.dta` inputs exist and writes an inventory.

## Current Build Outputs

`01_build_analysis_dataset.py` writes district-base and identifier-crosswalk files to `1_Data/Cleaned` and raw-source audit files to `3_Output/Audit`.

`02_build_panel_dataset.py` writes annual hazard-panel files to `1_Data/Cleaned` and panel diagnostics to `3_Output/Audit`.

`03_build_spatial_design_exposure.py` writes spatial/design panel files to `1_Data/Cleaned` and geometry/sample-drop diagnostics to `3_Output/Audit`.

Generated data/output files are intentionally ignored by Git.
