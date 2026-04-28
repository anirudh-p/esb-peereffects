# 1_Preparation

Raw-to-cleaned data construction.

## Included Scripts

- `01_build_analysis_dataset.py`: reads raw WRI, EPA/program, and auxiliary sources; harmonizes district identifiers; writes district-level base data and diagnostics.
- `02_build_panel_dataset.py`: expands the base district data into the hazard-panel structure and constructs outcomes, risk sets, and timing variables.
- `03_export_stata_inputs.py`: validates cleaned files and exports Stata-ready `.dta` inputs for estimation.
