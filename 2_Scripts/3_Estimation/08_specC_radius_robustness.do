* Spec C Radius Robustness: Hazard Panel Design-BH IV
* Purpose: compare simulated design-BH IV estimates across EDGE radius graphs.

* ------------------------------------------------------------------------------
* Preamble
* ------------------------------------------------------------------------------
do "2_Scripts/3_Estimation/00_globals.do"

capture log close
log using "${LOGS}/08_specC_radius_robustness_log.txt", replace text

section_header, title("Spec C Radius Robustness") ///
    detail("Compare the design-BH IV using 15, 30, and 60 mile EDGE radius graphs.")

* ------------------------------------------------------------------------------
* Setup
* ------------------------------------------------------------------------------
prepare_spatial_panel

local outcome_first_award "y_first_award"
local sample_main_base "main_estimation_sample == 1 & risk_first_award == 1 & exclude_own_r1_winner == 0"
local radius_list "15 30 60"

tempfile radius_results
tempname radius_post
postfile `radius_post' int radius double iv_coef iv_se iv_p ///
    double first_stage_coef first_stage_se first_stage_p first_stage_f mean_neighbors ///
    long obs districts double outcome_mean ///
    using `radius_results', replace

* ------------------------------------------------------------------------------
* Specification
* Loop across radius definitions and estimate the matching first-stage and 2SLS.
* ------------------------------------------------------------------------------
section_header, title("Specification") ///
    detail("Estimate radius-specific design-BH IV models after dropping radius isolates.")

foreach radius_miles of local radius_list {
    local peer_award_radius_lag "edge_r`radius_miles'_award_tm1_n"
    local instrument_radius "edge_r`radius_miles'_r1rcsim_tm1_n"
    local control_expected_radius "edge_r`radius_miles'_r1expsim_tm1_n"
    local radius_degree_count "edge_r`radius_miles'_degree_n"
    local sample_radius "`sample_main_base' & `radius_degree_count' > 0"

    label variable `peer_award_radius_lag' "Lagged EDGE radius-`radius_miles' peer first-award count"
    label variable `instrument_radius' "Lagged EDGE radius-`radius_miles' recentered simulated R1 shock"
    label variable `control_expected_radius' "Lagged EDGE radius-`radius_miles' simulated expected R1 wins"

    quietly areg `peer_award_radius_lag' c.`instrument_radius' c.`control_expected_radius' i.year ///
        if `sample_radius', absorb(district_panel_id) vce(cluster district_panel_id)
    local first_stage_coef = _b[`instrument_radius']
    local first_stage_se = _se[`instrument_radius']
    local first_stage_p = 2 * normal(-abs(_b[`instrument_radius'] / _se[`instrument_radius']))
    test `instrument_radius'
    local first_stage_f = r(F)

    quietly summarize `radius_degree_count' if `sample_radius', meanonly
    local mean_neighbors = r(mean)

    run_fwl_iv, sample(`sample_radius') outcome(`outcome_first_award') ///
        endogenous(`peer_award_radius_lag') instrument(`instrument_radius') ///
        controls(c.`control_expected_radius')
    post `radius_post' (`radius_miles') (r(coef)) (r(se)) (r(p)) ///
        (`first_stage_coef') (`first_stage_se') (`first_stage_p') (`first_stage_f') ///
        (`mean_neighbors') (r(obs)) (r(districts)) (r(outcome_mean))

    di as res "Radius `radius_miles' miles: 2SLS coef = " %9.4f r(coef) ///
        ", se = " %9.4f r(se) ", F = " %9.2f `first_stage_f'
}

postclose `radius_post'

* ------------------------------------------------------------------------------
* Export
* ------------------------------------------------------------------------------
section_header, title("Export") ///
    detail("Write compact CSV and LaTeX outputs for the radius robustness table.")

use `radius_results', clear
export delimited using "${TABLES}/08_specC_radius_robustness.csv", replace

sort radius
forvalues j = 1/`=_N' {
    local b`j' : display %9.4f iv_coef[`j']
    local b`j' = strtrim("`b`j''")
    if iv_p[`j'] < 0.01 local b`j' "`b`j''***"
    else if iv_p[`j'] < 0.05 local b`j' "`b`j''**"
    else if iv_p[`j'] < 0.10 local b`j' "`b`j''*"

    local se`j' : display %9.4f iv_se[`j']
    local se`j' = "(" + strtrim("`se`j''") + ")"

    local fsb`j' : display %9.4f first_stage_coef[`j']
    local fsb`j' = strtrim("`fsb`j''")

    local fss`j' : display %9.4f first_stage_se[`j']
    local fss`j' = "(" + strtrim("`fss`j''") + ")"

    local fsf`j' : display %9.2f first_stage_f[`j']
    local fsf`j' = strtrim("`fsf`j''")

    local deg`j' : display %9.2f mean_neighbors[`j']
    local deg`j' = strtrim("`deg`j''")

    local obs`j' : display %12.0fc obs[`j']
    local obs`j' = strtrim("`obs`j''")

    local districts`j' : display %12.0fc districts[`j']
    local districts`j' = strtrim("`districts`j''")

    local mean`j' : display %9.4f outcome_mean[`j']
    local mean`j' = strtrim("`mean`j''")
}

file open radius_tex using "${TABLES}/08_specC_radius_robustness.tex", write replace text
file write radius_tex "\begin{table}[!htbp]\centering" _n
file write radius_tex "\caption{Spec C Radius Robustness}" _n
file write radius_tex "\begin{tabular}{lccc}" _n
file write radius_tex "\hline\hline" _n
file write radius_tex " & (1) & (2) & (3) \\" _n
file write radius_tex " & 15 miles & 30 miles & 60 miles \\" _n
file write radius_tex "\hline" _n
file write radius_tex "Lagged EDGE radius peer awards & `b1' & `b2' & `b3' \\" _n
file write radius_tex " & `se1' & `se2' & `se3' \\" _n
file write radius_tex "First-stage coefficient & `fsb1' & `fsb2' & `fsb3' \\" _n
file write radius_tex " & `fss1' & `fss2' & `fss3' \\" _n
file write radius_tex "\hline" _n
file write radius_tex "First-stage F & `fsf1' & `fsf2' & `fsf3' \\" _n
file write radius_tex "Mean neighbors & `deg1' & `deg2' & `deg3' \\" _n
file write radius_tex "Observations & `obs1' & `obs2' & `obs3' \\" _n
file write radius_tex "Districts & `districts1' & `districts2' & `districts3' \\" _n
file write radius_tex "Mean dependent variable & `mean1' & `mean2' & `mean3' \\" _n
file write radius_tex "District FE & Yes & Yes & Yes \\" _n
file write radius_tex "Year FE & Yes & Yes & Yes \\" _n
file write radius_tex "Expected R1 exposure control & Yes & Yes & Yes \\" _n
file write radius_tex "Radius isolates dropped & Yes & Yes & Yes \\" _n
file write radius_tex "\hline\hline" _n
file write radius_tex "\end{tabular}" _n
file write radius_tex "\begin{flushleft}\footnotesize Notes: Each column estimates the Spec C simulated design-BH IV on the main first-award hazard sample using an EDGE radius graph rather than EDGE K6 nearest neighbors. The endogenous peer variable is the number of neighboring districts within the stated radius with first awards by t-1. The instrument is recentered neighbor R1 exposure within the same radius: realized neighbor R1 wins minus simulated design-expected neighbor R1 wins, with the expected-exposure control included directly. Districts with zero neighbors inside the stated radius are dropped from that column's sample. Standard errors are clustered by district.\end{flushleft}" _n
file write radius_tex "\end{table}" _n
file close radius_tex

list, noobs abbreviate(32)

log close
