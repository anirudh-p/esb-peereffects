# 3_Estimation

Stata estimation spine for the Brown pitch.

## Included Scripts

- `00_globals.do`: shared paths, output folders, labeled analysis variables, section banners, and common estimation helper programs.
- `00_run_all.do`: runs the currently implemented table sequence: sample selection, the Spec C shock diagnostic, Spec A, Spec B, Spec C, Spec D, the Spec C non-applicant restriction, and the Spec C radius robustness table. Timing and mechanism scripts are intentionally left out until implemented.
- `01_specA_hazard_panel_ols.do`: descriptive first-award hazard-panel OLS using lagged K6 neighbor-award exposure; establishes the spatial fact but is not causal.
- `02_specB_hazard_panel_raw_iv.do`: raw hazard-panel IV bridge specification using neighbor R1 rebate wins as an instrument for lagged neighbor awards; excludes focal R1 winners and leaves focal R3 timing out of the baseline.
- `03_specC_hazard_panel_design_bh_iv.do`: main design-based Borusyak-Hull-style IV specification using simulated selection-rule recentered neighbor R1 exposure and simulated expected-exposure controls.
- `04_specD_hazard_panel_design_bh_cf_iv.do`: Spec C sensitivity that recenters R1 wins under an all-districts-applied counterfactual using 2022 priority-status win rates.
- `05_event_study_timing.do`: timing and pre-trend evidence around neighbor exposure.
- `06_mechanism_vendor_capacity.do`: exploratory mechanism tests for vendor, third-party, and capacity channels.
- `07_specC_nonapplicant_hazard_iv.do`: Spec C robustness that restricts the focal sample to districts that did not apply in R1.
- `08_specC_radius_robustness.do`: Spec C robustness that replaces the EDGE K6 graph with EDGE radius graphs at 15, 30, and 60 miles.
