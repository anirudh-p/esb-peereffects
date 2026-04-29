* Run the current Brown pitch preliminaries and main estimation sequence.

do "2_Scripts/3_Estimation/00_globals.do"

section_header, title("Run All") ///
    detail("Refresh preliminaries, diagnostics, and the current implemented estimation stack.")

* ------------------------------------------------------------------------------
* Preliminaries
* ------------------------------------------------------------------------------
do "2_Scripts/2_Preliminaries/01_descriptive_statistics.do"
do "2_Scripts/2_Preliminaries/04_specC_shock_diagnostics.do"

* ------------------------------------------------------------------------------
* Main estimation objects
* ------------------------------------------------------------------------------
do "2_Scripts/3_Estimation/01_specA_hazard_panel_ols.do"
do "2_Scripts/3_Estimation/02_specB_hazard_panel_raw_iv.do"
do "2_Scripts/3_Estimation/03_specC_hazard_panel_design_bh_iv.do"
do "2_Scripts/3_Estimation/04_specD_hazard_panel_design_bh_cf_iv.do"
do "2_Scripts/3_Estimation/07_specC_nonapplicant_hazard_iv.do"
do "2_Scripts/3_Estimation/08_specC_radius_robustness.do"

* ------------------------------------------------------------------------------
* Placeholders
* These remain manual until they are fully implemented.
* ------------------------------------------------------------------------------
section_header, title("Placeholders") ///
    detail("Event-study, mechanism, balance, and figure scripts remain explicit placeholders.")
