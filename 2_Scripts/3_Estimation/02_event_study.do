/*
================================================================================
  02_event_study.do
  
  Reduced Form Event Study: Dynamic treatment effects of neighbor R1 announcement
  using lead/lag interactions with w6_Z_t_minus_1.
  
  Panel covers 2018-2024.
  DV: Y_first_adopted
  Treatment variable: w6_Z_t_minus_1 (Share of neighbors winning R1 by t-1)
  Base/Omitted Year: 2022 (Because R1 outcomes announced late Oct 2022, 
                     early 2022 behavior was locked in prior to realization)
  
  Fixed Effects: nces_id (District)
  Clustering: nces_id (District)
================================================================================
*/

capture log close
clear all
set more off
set matsize 11000

// Start Log
log using "3_Output\Logs\02_event_study_log.txt", replace text

// 1. Data Import
// ==============================================================================
disp "Loading Data..."
use "1_Data\Cleaned\analysis_panel_dataset_spatial.dta", clear

destring nces_id, replace force
egen panel_id = group(nces_id)
xtset panel_id year

// 2. Sample Restrictions
// ==============================================================================
disp "Applying sample restrictions..."

// Keep until district's first award
keep if hazard_keep_awarded == 1

// Exclude focal districts that won R1 themselves to isolate peer effects
keep if IV_Z_R1 == 0

// 3. Generate Event Study Interactions
// ==============================================================================
disp "Generating event study interactions..."

// Create year dummies manually or use i.year notation in regression.
// For the treatment, we interact the time-invariant instrument (which takes its post-2022 value)
// with year dummies. 
// Note: w6_Z_t_minus_1 in the dataset is ALREADY 0 before 2023 and fractional after.
// To run a standard event study, we need the *static* cross-sectional treatment intensity for each district.
// We proxy this by taking the max instrument exposure for each district over the panel.

bysort panel_id: egen treatment_intensity = max(w6_Z_t_minus_1)

// Create interactions (excluding 2022 as the base year)
gen treat_X_2018 = treatment_intensity * (year == 2018)
gen treat_X_2019 = treatment_intensity * (year == 2019)
gen treat_X_2020 = treatment_intensity * (year == 2020)
gen treat_X_2021 = treatment_intensity * (year == 2021)
// 2022 IS OMITTED BASE YEAR
gen treat_X_2023 = treatment_intensity * (year == 2023)
gen treat_X_2024 = treatment_intensity * (year == 2024)

// 4. Fixed Effects Estimation
// ==============================================================================
disp "Running Event Study with Fixed Effects..."

// Run xtreg with district fixed effects and clustered SEs
xtreg Y_first_adopted ///
    treat_X_2018 treat_X_2019 treat_X_2020 treat_X_2021 ///
    treat_X_2023 treat_X_2024 ///
    i.year enrollment pm25, ///
    fe vce(cluster panel_id)

est store event_study

// Output to RTF
esttab event_study using "3_Output\Tables\event_study_results.rtf", replace ///
    title("Event Study: Dynamic Peer Effects of R1 Winners on Focal Adoption") ///
    b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    drop(_cons *year enrollment pm25) ///
    order(treat_X_2018 treat_X_2019 treat_X_2020 treat_X_2021 treat_X_2023 treat_X_2024) ///
    addnotes("Omitted Base Year is 2022. Models include district and year fixed effects.")

log close
exit
