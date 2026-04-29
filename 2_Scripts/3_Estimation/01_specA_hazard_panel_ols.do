* Spec A: Hazard Panel OLS
* Purpose: descriptive spatial fact, not a causal estimate.

* ------------------------------------------------------------------------------
* Preamble
* Load common globals, helper programs, and the labeled spatial panel.
* ------------------------------------------------------------------------------
do "2_Scripts/3_Estimation/00_globals.do"

capture log close
log using "${LOGS}/01_specA_hazard_panel_ols_log.txt", replace text

section_header, title("Spec A: Hazard Panel OLS") ///
    detail("Descriptive benchmark relating lagged neighbor awards to the first-award hazard.")

* ------------------------------------------------------------------------------
* Setup
* Define labeled analysis variables and estimation samples.
* ------------------------------------------------------------------------------
prepare_spatial_panel

local outcome_first_award "y_first_award"
local peer_award_k6_lag "edge_w6_award_tm1_n"
local sample_main_risk "main_estimation_sample == 1 & risk_first_award == 1"
local sample_noisol_risk "main_noisol_edge_k6_50 == 1 & risk_first_award == 1"

label variable `outcome_first_award' "First ESB award in district-year"
label variable `peer_award_k6_lag' "Lagged EDGE K6 peer first-award count"

* ------------------------------------------------------------------------------
* Construction
* Collect a compact set of post-estimation sample statistics for each model.
* ------------------------------------------------------------------------------
capture program drop collect_speca_stats
program define collect_speca_stats, rclass
    version 16
    syntax , Outcome(name) Exposure(name)

    collect_estimation_stats, outcome(`outcome') coefvar(`exposure')
    return scalar obs = r(obs)
    return scalar districts = r(districts)
    return scalar outcome_mean = r(outcome_mean)
    return scalar coef = r(coef)
    return scalar se = r(se)
    return scalar p = r(p)
end

tempfile speca_results
tempname speca_post
postfile `speca_post' str44 model str26 sample str24 fixed_effects ///
    byte own_rebate_controls double coef se p long obs districts double outcome_mean ///
    using `speca_results', replace

* ------------------------------------------------------------------------------
* Specification
* Estimate the descriptive OLS variants while suppressing long FE coefficient dumps.
* ------------------------------------------------------------------------------
section_header, title("Specification") ///
    detail("Estimate state-FE, district-FE, own-win, state-year, and no-isolate variants.")

quietly regress `outcome_first_award' c.`peer_award_k6_lag' ${CORE_CONTROLS} ///
    i.state_id i.year if `sample_main_risk', vce(cluster district_panel_id)
collect_speca_stats, outcome(`outcome_first_award') exposure(`peer_award_k6_lag')
post `speca_post' ("State FE plus controls") ("Main risk set") ("State and year") ///
    (0) (r(coef)) (r(se)) (r(p)) (r(obs)) (r(districts)) (r(outcome_mean))
di as res "State FE plus controls: coef = " %9.4f r(coef) ", se = " %9.4f r(se) ", p = " %9.4f r(p)

quietly xtreg `outcome_first_award' c.`peer_award_k6_lag' i.year ///
    if `sample_main_risk', fe vce(cluster district_panel_id)
collect_speca_stats, outcome(`outcome_first_award') exposure(`peer_award_k6_lag')
post `speca_post' ("District FE") ("Main risk set") ("District and year") ///
    (0) (r(coef)) (r(se)) (r(p)) (r(obs)) (r(districts)) (r(outcome_mean))
di as res "District FE: coef = " %9.4f r(coef) ", se = " %9.4f r(se) ", p = " %9.4f r(p)

quietly xtreg `outcome_first_award' c.`peer_award_k6_lag' ///
    own_r1_win_post own_r3_win_post i.year ///
    if `sample_main_risk', fe vce(cluster district_panel_id)
collect_speca_stats, outcome(`outcome_first_award') exposure(`peer_award_k6_lag')
post `speca_post' ("District FE plus own wins") ("Main risk set") ("District and year") ///
    (1) (r(coef)) (r(se)) (r(p)) (r(obs)) (r(districts)) (r(outcome_mean))
di as res "District FE plus own wins: coef = " %9.4f r(coef) ", se = " %9.4f r(se) ", p = " %9.4f r(p)

quietly xtreg `outcome_first_award' c.`peer_award_k6_lag' ///
    own_r1_win_post own_r3_win_post i.state_id#i.year ///
    if `sample_main_risk', fe vce(cluster district_panel_id)
collect_speca_stats, outcome(`outcome_first_award') exposure(`peer_award_k6_lag')
post `speca_post' ("State-year FE") ("Main risk set") ("District and state-year") ///
    (1) (r(coef)) (r(se)) (r(p)) (r(obs)) (r(districts)) (r(outcome_mean))
di as res "State-year FE: coef = " %9.4f r(coef) ", se = " %9.4f r(se) ", p = " %9.4f r(p)

quietly xtreg `outcome_first_award' c.`peer_award_k6_lag' ///
    own_r1_win_post own_r3_win_post i.year ///
    if `sample_noisol_risk', fe vce(cluster district_panel_id)
collect_speca_stats, outcome(`outcome_first_award') exposure(`peer_award_k6_lag')
post `speca_post' ("No isolated K6") ("No isolated risk set") ("District and year") ///
    (1) (r(coef)) (r(se)) (r(p)) (r(obs)) (r(districts)) (r(outcome_mean))
di as res "No isolated K6: coef = " %9.4f r(coef) ", se = " %9.4f r(se) ", p = " %9.4f r(p)

postclose `speca_post'

* ------------------------------------------------------------------------------
* Export
* Write compact CSV and LaTeX tables with FE summarized as short indicator rows.
* ------------------------------------------------------------------------------
section_header, title("Export") ///
    detail("Write concise table outputs and print the compact results listing.")

use `speca_results', clear
export delimited using "${TABLES}/01_specA_hazard_panel_ols.csv", replace

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
}

file open speca_tex using "${TABLES}/01_specA_hazard_panel_ols.tex", write replace text
file write speca_tex "\begin{table}[!htbp]\centering" _n
file write speca_tex "\caption{Spec A: Hazard Panel OLS}" _n
file write speca_tex "\begin{tabular}{lccccc}" _n
file write speca_tex "\hline\hline" _n
file write speca_tex " & (1) & (2) & (3) & (4) & (5) \\" _n
file write speca_tex " & State FE & District FE & Own wins & State-year FE & No isolated \\" _n
file write speca_tex "\hline" _n
file write speca_tex "Lagged EDGE K6 peer awards & `b1' & `b2' & `b3' & `b4' & `b5' \\" _n
file write speca_tex " & `se1' & `se2' & `se3' & `se4' & `se5' \\" _n
file write speca_tex "\hline" _n
file write speca_tex "Observations & `obs1' & `obs2' & `obs3' & `obs4' & `obs5' \\" _n
file write speca_tex "Districts & `districts1' & `districts2' & `districts3' & `districts4' & `districts5' \\" _n
file write speca_tex "Mean dependent variable & `mean1' & `mean2' & `mean3' & `mean4' & `mean5' \\" _n
file write speca_tex "District FE & No & Yes & Yes & Yes & Yes \\" _n
file write speca_tex "Year FE & Yes & Yes & Yes & No & Yes \\" _n
file write speca_tex "State-year FE & No & No & No & Yes & No \\" _n
file write speca_tex "Own rebate-win controls & No & No & Yes & Yes & Yes \\" _n
file write speca_tex "\hline\hline" _n
file write speca_tex "\end{tabular}" _n
file write speca_tex "\begin{flushleft}\footnotesize Notes: Outcome is first ESB award in district-year t, restricted to districts still at risk of first award. Peer exposure is the number of six EDGE-nearest neighboring districts with a first award by t-1. Column 5 drops districts whose sixth EDGE neighbor is more than 50 miles away. Standard errors are clustered by district. This is descriptive OLS, not the design-based causal specification.\end{flushleft}" _n
file write speca_tex "\end{table}" _n
file close speca_tex

list, noobs abbreviate(32)

log close
