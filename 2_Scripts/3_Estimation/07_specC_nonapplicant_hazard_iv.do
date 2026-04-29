* Spec C Non-Applicant Robustness: Hazard Panel Design-BH IV
* Purpose: restrict focal districts to districts that did not apply in R1.

* ------------------------------------------------------------------------------
* Preamble
* ------------------------------------------------------------------------------
do "2_Scripts/3_Estimation/00_globals.do"

capture log close
log using "${LOGS}/07_specC_nonapplicant_hazard_iv_log.txt", replace text

section_header, title("Spec C Non-Applicant Robustness") ///
    detail("Repeat Spec C while restricting focal districts to R1 non-applicants.")

* ------------------------------------------------------------------------------
* Setup
* ------------------------------------------------------------------------------
prepare_spatial_panel

local outcome_first_award "y_first_award"
local peer_award_k6_lag "edge_w6_award_tm1_n"
local instrument_rcsim_k6 "edge_w6_r1rcsim_tm1_n"
local control_expected_rcsim_k6 "edge_w6_r1expsim_tm1_n"
local controls_expected "c.`control_expected_rcsim_k6'"
local sample_main_risk "main_estimation_sample == 1 & risk_first_award == 1 & exclude_own_r1_winner == 0 & r1_lottery_applicant == 0"
local sample_noisol_risk "main_noisol_edge_k6_50 == 1 & risk_first_award == 1 & exclude_own_r1_winner == 0 & r1_lottery_applicant == 0"

label variable `peer_award_k6_lag' "Lagged EDGE K6 peer first-award count"
label variable `instrument_rcsim_k6' "Lagged EDGE K6 recentered simulated R1 shock"
label variable `control_expected_rcsim_k6' "Lagged EDGE K6 simulated expected R1 wins"

tempfile spec_results
tempname spec_post
postfile `spec_post' str34 model str28 sample str24 dependent ///
    str34 coefficient_on double coef se p double first_stage_f ///
    long obs districts double outcome_mean ///
    using `spec_results', replace

* ------------------------------------------------------------------------------
* Specification
* ------------------------------------------------------------------------------
section_header, title("Specification") ///
    detail("Estimate first-stage, reduced-form, and 2SLS models on the non-applicant focal sample.")

quietly areg `peer_award_k6_lag' c.`instrument_rcsim_k6' `controls_expected' i.year ///
    if `sample_main_risk', absorb(district_panel_id) vce(cluster district_panel_id)
collect_estimation_stats, outcome(`peer_award_k6_lag') coefvar(`instrument_rcsim_k6')
local first_stage_coef_main = r(coef)
local first_stage_se_main = r(se)
local first_stage_p_main = r(p)
local first_stage_obs_main = r(obs)
local first_stage_districts_main = r(districts)
local first_stage_mean_main = r(outcome_mean)
test `instrument_rcsim_k6'
local first_stage_f_main = r(F)
post `spec_post' ("First stage") ("Non-applicant risk set") ("Peer awards by t-1") ///
    ("Simulated design R1 shock") (`first_stage_coef_main') (`first_stage_se_main') ///
    (`first_stage_p_main') (`first_stage_f_main') ///
    (`first_stage_obs_main') (`first_stage_districts_main') (`first_stage_mean_main')
di as res "First stage, non-applicant sample: coef = " %9.4f `first_stage_coef_main' ///
    ", se = " %9.4f `first_stage_se_main' ", F = " %9.2f `first_stage_f_main'

quietly areg `outcome_first_award' c.`instrument_rcsim_k6' `controls_expected' i.year ///
    if `sample_main_risk', absorb(district_panel_id) vce(cluster district_panel_id)
collect_estimation_stats, outcome(`outcome_first_award') coefvar(`instrument_rcsim_k6')
post `spec_post' ("Reduced form") ("Non-applicant risk set") ("First award hazard") ///
    ("Simulated design R1 shock") (r(coef)) (r(se)) (r(p)) (.) ///
    (r(obs)) (r(districts)) (r(outcome_mean))
di as res "Reduced form, non-applicant sample: coef = " %9.4f r(coef) ///
    ", se = " %9.4f r(se) ", p = " %9.4f r(p)

run_fwl_iv, sample(`sample_main_risk') outcome(`outcome_first_award') ///
    endogenous(`peer_award_k6_lag') instrument(`instrument_rcsim_k6') ///
    controls(`controls_expected')
post `spec_post' ("2SLS") ("Non-applicant risk set") ("First award hazard") ///
    ("Peer awards by t-1") (r(coef)) (r(se)) (r(p)) (`first_stage_f_main') ///
    (r(obs)) (r(districts)) (r(outcome_mean))
di as res "2SLS, non-applicant sample: coef = " %9.4f r(coef) ///
    ", se = " %9.4f r(se) ", p = " %9.4f r(p)

quietly areg `peer_award_k6_lag' c.`instrument_rcsim_k6' `controls_expected' i.year ///
    if `sample_noisol_risk', absorb(district_panel_id) vce(cluster district_panel_id)
test `instrument_rcsim_k6'
local first_stage_f_noisol = r(F)

run_fwl_iv, sample(`sample_noisol_risk') outcome(`outcome_first_award') ///
    endogenous(`peer_award_k6_lag') instrument(`instrument_rcsim_k6') ///
    controls(`controls_expected')
post `spec_post' ("2SLS") ("Non-applicant no isolated") ("First award hazard") ///
    ("Peer awards by t-1") (r(coef)) (r(se)) (r(p)) (`first_stage_f_noisol') ///
    (r(obs)) (r(districts)) (r(outcome_mean))
di as res "2SLS, non-applicant no-isolate sample: coef = " %9.4f r(coef) ///
    ", se = " %9.4f r(se) ", p = " %9.4f r(p)

postclose `spec_post'

* ------------------------------------------------------------------------------
* Export
* ------------------------------------------------------------------------------
section_header, title("Export") ///
    detail("Write compact CSV and LaTeX outputs for the non-applicant robustness.")

use `spec_results', clear
export delimited using "${TABLES}/07_specC_nonapplicant_hazard_iv.csv", replace

forvalues j = 1/`=_N' {
    local b`j' : display %9.4f coef[`j']
    local b`j' = strtrim("`b`j''")
    if p[`j'] < 0.01 local b`j' "`b`j''***"
    else if p[`j'] < 0.05 local b`j' "`b`j''**"
    else if p[`j'] < 0.10 local b`j' "`b`j''*"

    local se`j' : display %9.4f se[`j']
    local se`j' = "(" + strtrim("`se`j''") + ")"

    local obs`j' : display %12.0fc obs[`j']
    local obs`j' = strtrim("`obs`j''")

    local districts`j' : display %12.0fc districts[`j']
    local districts`j' = strtrim("`districts`j''")

    local mean`j' : display %9.4f outcome_mean[`j']
    local mean`j' = strtrim("`mean`j''")

    local fsf`j' ""
    if first_stage_f[`j'] < . {
        local fsf`j' : display %9.2f first_stage_f[`j']
        local fsf`j' = strtrim("`fsf`j''")
    }
}

file open spec_tex using "${TABLES}/07_specC_nonapplicant_hazard_iv.tex", write replace text
file write spec_tex "\begin{table}[!htbp]\centering" _n
file write spec_tex "\caption{Spec C: Non-Applicant Hazard Panel Design-BH IV}" _n
file write spec_tex "\begin{tabular}{lcccc}" _n
file write spec_tex "\hline\hline" _n
file write spec_tex " & (1) & (2) & (3) & (4) \\" _n
file write spec_tex " & First stage & Reduced form & 2SLS & 2SLS no isolated \\" _n
file write spec_tex "\hline" _n
file write spec_tex "Lagged EDGE K6 recentered R1 shock & `b1' & `b2' &  &  \\" _n
file write spec_tex " & `se1' & `se2' &  &  \\" _n
file write spec_tex "Lagged EDGE K6 peer awards &  &  & `b3' & `b4' \\" _n
file write spec_tex " &  &  & `se3' & `se4' \\" _n
file write spec_tex "\hline" _n
file write spec_tex "First-stage F & `fsf1' &  & `fsf3' & `fsf4' \\" _n
file write spec_tex "Observations & `obs1' & `obs2' & `obs3' & `obs4' \\" _n
file write spec_tex "Districts & `districts1' & `districts2' & `districts3' & `districts4' \\" _n
file write spec_tex "Mean dependent variable & `mean1' & `mean2' & `mean3' & `mean4' \\" _n
file write spec_tex "District FE & Yes & Yes & Yes & Yes \\" _n
file write spec_tex "Year FE & Yes & Yes & Yes & Yes \\" _n
file write spec_tex "Expected R1 exposure control & Yes & Yes & Yes & Yes \\" _n
file write spec_tex "Focal R1 applicants excluded & Yes & Yes & Yes & Yes \\" _n
file write spec_tex "\hline\hline" _n
file write spec_tex "\end{tabular}" _n
file write spec_tex "\begin{flushleft}\footnotesize Notes: Outcome in columns 2--4 is first ESB award in district-year t, restricted to districts still at risk of first award, excluding focal districts that won R1 rebates, and further restricting the focal sample to districts that did not apply in R1. The endogenous peer variable is the number of six EDGE-nearest neighboring districts with first awards by t-1. The instrument is recentered K6 neighbor R1 exposure: realized neighbor R1 wins minus simulated design-expected neighbor R1 wins, switched on for post-R1 years. All columns control for simulated design-expected K6 neighbor R1 exposure. Column 4 drops districts whose sixth EDGE neighbor is more than 50 miles away. Standard errors are clustered by district.\end{flushleft}" _n
file write spec_tex "\end{table}" _n
file close spec_tex

list, noobs abbreviate(32)

log close
