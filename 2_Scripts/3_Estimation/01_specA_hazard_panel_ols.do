* Spec A: Hazard Panel OLS.
* Purpose: descriptive spatial fact, not a causal estimate.
do "2_Scripts/3_Estimation/00_globals.do"

capture log close
log using "${LOGS}/01_specA_hazard_panel_ols_log.txt", replace text

use "${SPATIAL_PANEL}", clear

encode state, gen(state_id)
xtset district_panel_id year

gen ln_students = ln(students + 1) if students >= 0
gen ln_income = ln(median_income + 1) if median_income >= 0

local y "y_first_award"
local x_edge "edge_w6_award_tm1_n"
local main_sample "main_estimation_sample == 1 & risk_first_award == 1"
local noisol_sample "main_noisol_edge_k6_50 == 1 & risk_first_award == 1"

capture program drop collect_speca_stats
program define collect_speca_stats, rclass
    version 16
    syntax , Outcome(name) Exposure(name)

    tempvar esample district_tag
    gen byte `esample' = e(sample)
    egen byte `district_tag' = tag(district_panel_id) if `esample'

    quietly count if `esample'
    return scalar obs = r(N)

    quietly count if `district_tag' == 1
    return scalar districts = r(N)

    quietly summarize `outcome' if `esample', meanonly
    return scalar outcome_mean = r(mean)

    return scalar coef = _b[`exposure']
    return scalar se = _se[`exposure']
    return scalar p = 2 * ttail(e(df_r), abs(_b[`exposure'] / _se[`exposure']))
end

tempfile speca_results
tempname speca_post
postfile `speca_post' str44 model str26 sample str24 fixed_effects ///
    byte own_rebate_controls double coef se p long obs districts double outcome_mean ///
    using `speca_results', replace

regress `y' c.`x_edge' ${CORE_CONTROLS} i.state_id i.year ///
    if `main_sample', vce(cluster district_panel_id)
collect_speca_stats, outcome(`y') exposure(`x_edge')
post `speca_post' ("State FE plus controls") ("Main risk set") ("State and year") ///
    (0) (r(coef)) (r(se)) (r(p)) (r(obs)) (r(districts)) (r(outcome_mean))

xtreg `y' c.`x_edge' i.year ///
    if `main_sample', fe vce(cluster district_panel_id)
collect_speca_stats, outcome(`y') exposure(`x_edge')
post `speca_post' ("District FE") ("Main risk set") ("District and year") ///
    (0) (r(coef)) (r(se)) (r(p)) (r(obs)) (r(districts)) (r(outcome_mean))

xtreg `y' c.`x_edge' own_r1_win_post own_r3_win_post i.year ///
    if `main_sample', fe vce(cluster district_panel_id)
collect_speca_stats, outcome(`y') exposure(`x_edge')
post `speca_post' ("District FE plus own wins") ("Main risk set") ("District and year") ///
    (1) (r(coef)) (r(se)) (r(p)) (r(obs)) (r(districts)) (r(outcome_mean))

xtreg `y' c.`x_edge' own_r1_win_post own_r3_win_post i.state_id#i.year ///
    if `main_sample', fe vce(cluster district_panel_id)
collect_speca_stats, outcome(`y') exposure(`x_edge')
post `speca_post' ("State-year FE") ("Main risk set") ("District and state-year") ///
    (1) (r(coef)) (r(se)) (r(p)) (r(obs)) (r(districts)) (r(outcome_mean))

xtreg `y' c.`x_edge' own_r1_win_post own_r3_win_post i.year ///
    if `noisol_sample', fe vce(cluster district_panel_id)
collect_speca_stats, outcome(`y') exposure(`x_edge')
post `speca_post' ("No isolated K6") ("No isolated risk set") ("District and year") ///
    (1) (r(coef)) (r(se)) (r(p)) (r(obs)) (r(districts)) (r(outcome_mean))

postclose `speca_post'

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
file write speca_tex "Peer awards by t-1, K6 & `b1' & `b2' & `b3' & `b4' & `b5' \\" _n
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
