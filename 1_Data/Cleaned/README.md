# Cleaned Data

Script-created datasets belong here. Current expected outputs from `2_Scripts/1_Preparation/01_build_analysis_dataset.py`:

- `analysis_district_base.csv`: one row per standardized NCES district ID in the source crosswalk.
- `analysis_district_base.dta`: Stata-ready mirror of the base district file.
- `district_identifier_crosswalk.csv`: one row per NCES district ID, with canonical district name/state/city and source-presence flags.
- `district_source_records.csv`: long source-record file used to audit identifiers across WRI, CSBP, bus-level, and survey sources.

This folder is ignored by Git except for this README. Rebuild generated files from scripts rather than editing them manually.
