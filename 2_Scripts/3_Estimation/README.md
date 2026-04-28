# 3_Estimation

Stata estimation spine for the Brown pitch.

## Included Scripts

- `00_globals.do`: shared paths, output folders, and estimation settings.
- `00_run_all.do`: runs the main estimation sequence.
- `01_specA_hazard_panel_ols.do`: descriptive hazard-panel OLS; establishes the spatial fact but is not causal.
- `02_specB_hazard_panel_raw_iv.do`: raw hazard-panel IV bridge specification.
- `03_specC_hazard_panel_design_bh_iv.do`: main design-based Borusyak-Hull-style IV specification.
- `04_specD_hazard_panel_design_bh_cf_iv.do`: design-based IV with controls/fixed effects/channel restrictions.
- `05_event_study_timing.do`: timing and pre-trend evidence around neighbor exposure.
- `06_mechanism_vendor_capacity.do`: exploratory mechanism tests for vendor, third-party, and capacity channels.
