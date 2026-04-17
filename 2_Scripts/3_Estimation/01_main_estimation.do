/*
================================================================================
  01_main_estimation.do
  
  Executes the First Stage and 2SLS IV estimation on the discrete-time annual panel.
  This script replaces multiple python files for a more standard Economics workflow.

  Panel covers 2018-2024.
  Main DV: Y_first_adopted
  Mechanism DV: Y_first_operating
  
  Treatment (endogenous): w6_P_operating_t_minus_1 (share neighbors adopting by end of t-1)
  Instrument: w6_IV_Z_R1 (share neighbors winning R1 by end of t-1)
  Baseline control: w6_P_operating_preR1 (start of 2022 condition)
  
  Fixed Effects: nces_id (District), year
  Clustering: nces_id (District)
================================================================================
*/

capture log close
clear all
set more off
set matsize 11000 // For high dimensional FEs

// Set working directory to the project root (Update if needed)
cd "C:\BC PhD\Research\Peer-Effects and Adoption"

// Start Log
cap log close
log using "3_Output\Logs\01_main_estimation_log.txt", replace text

// 1. Data Import
// ==============================================================================
disp "Loading Data..."
use "1_Data\Cleaned\analysis_panel_dataset_spatial.dta", clear

// Destring IDs if necessary, generate numeric panel ID
destring nces_id, replace force
egen panel_id = group(nces_id)
xtset panel_id year

// 2. Variable Construction & Cleaning
// ==============================================================================
// We need to enforce the hazard survival model.
// Districts drop out of the "at-risk" estimating sample in the years following their first award.

// First, make sure our dependent variables exist and are 0/1 indicator format
// (They should be generated in prep scripts, but we handle structural generation here if needed)
// hazard_keep_awarded == 1 keeps observations up to and including the year of first award.

// Generate year dummies for xtivreg2
quietly tab year, gen(y_dum)
local year_dummies "y_dum2 y_dum3 y_dum4 y_dum5 y_dum6 y_dum7"

// 3. Main IV Estimation (Awarded)
// ==============================================================================
disp "Running Main IV Specification (DV: Y_first_adopted)..."

// Restrict sample based on hazard condition and removing own winners
preserve
keep if hazard_keep_awarded == 1
keep if IV_Z_R1 == 0 // Exclude focal districts that actually won the exogenous shock

// First Stage (for F-stat and diagnostics)
xtreg w6_P_t_minus_1 w6_Z_t_minus_1 i.year, fe vce(cluster panel_id)
est store first_stage

// Reduced Form (ITT)
xtreg Y_first_adopted w6_Z_t_minus_1 i.year, fe vce(cluster panel_id)
est store reduced_form

// 2SLS IV (Using xtivreg2 for exact sample consistency and K-P stats as a fallback)
xtivreg2 Y_first_adopted (w6_P_t_minus_1 = w6_Z_t_minus_1) `year_dummies', fe cluster(panel_id)
est store second_stage_awarded

restore

// 4. Mechanism Estimation (Operating/Physical Salience)
// ==============================================================================
disp "Running Mechanism IV Specification (DV: Y_first_adopted on Endogenous Peer Operating)..."

preserve
keep if hazard_keep_awarded == 1
keep if IV_Z_R1 == 0 

// 2SLS IV - Mechanism (Physical Salience)
xtivreg2 Y_first_adopted (w6_P_operating_t_minus_1 = w6_Z_t_minus_1) `year_dummies', fe cluster(panel_id)
est store second_stage_operating

restore

// 5. Exclusion Restrictions (Third Party Decomposition)
// ==============================================================================
disp "Decomposing Spillovers: Effect of Neighbors on Self-Filed vs Third-Party-Filed Applications"

// Load cross-section because R3 application is a static outcome.
import delimited "1_Data\Cleaned\analysis_dataset_spatial.csv", clear
destring nces_id, replace force

// Generate Decomposition DVs
gen Y_R3_apply_self = (y_r3_apply == 1 & is_r3_third_party == 0)
gen Y_R3_apply_third = (y_r3_apply == 1 & is_r3_third_party == 1)

// RF: Z's effect on Focal Self-filed Application
areg Y_R3_apply_self w6_iv_z_r1, absorb(state) vce(cluster state)
est store rf_self_filed

// RF: Z's effect on Focal Third-Party Application
areg Y_R3_apply_third w6_iv_z_r1, absorb(state) vce(cluster state)
est store rf_third_party

// 6. Output Standardization
// ==============================================================================
esttab first_stage reduced_form second_stage_awarded second_stage_operating rf_self_filed rf_third_party ///
    using "3_Output\Tables\main_iv_results.rtf", replace ///
    title("Table: Peer Effects on School Bus Adoption (IV Estimates and Robustness)") ///
    mtitle("First Stage" "Reduced Form" "2SLS (Award)" "2SLS (Oper)" "RF: Self-File" "RF: 3rd-Party") ///
    b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    scalars("widstat F-Stat (K-P)" "N Observations") ///
    drop(_cons) ///
    addnotes("Cluster-robust SEs in parentheses." "Panel models include District & Year FEs; Cross-sec models use State FEs.")

log close
exit
