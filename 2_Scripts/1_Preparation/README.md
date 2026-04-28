# 1_Preparation

Raw-to-cleaned data construction.

## Included Scripts

- `01_build_analysis_dataset.py`: inventories raw files/tables, standardizes NCES district IDs across WRI/CSBP/survey sources, writes a district identifier crosswalk, and creates the district-level base analysis file.
- `02_build_panel_dataset.py`: expands the base district data into the hazard-panel structure and constructs outcomes, risk sets, and timing variables.
- `03_export_stata_inputs.py`: validates cleaned files and exports Stata-ready `.dta` inputs for estimation.

## Current Build Outputs

`01_build_analysis_dataset.py` writes generated files to `1_Data/Cleaned` and audit files to `3_Output/Audit`. Generated data/output files are intentionally ignored by Git.
