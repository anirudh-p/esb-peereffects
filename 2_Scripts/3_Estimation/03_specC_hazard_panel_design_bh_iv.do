* Spec C: Hazard Panel Design-BH IV.
* Purpose: main design-based causal specification for the Brown pitch.
do "2_Scripts/3_Estimation/00_globals.do"

capture log close
log using "${LOGS}/03_specC_hazard_panel_design_bh_iv_log.txt", replace text

use "${SPATIAL_PANEL}", clear

local y "y_first_award"
local p "edge_w6_award_tm1_n"
local z "edge_w6_r1rcsim_tm1_n"
local expected "edge_w6_r1expsim_tm1_n"
local controls "c.`expected'"
local main_sample "main_estimation_sample == 1 & risk_first_award == 1 & exclude_own_r1_winner == 0"
local noisol_sample "main_noisol_edge_k6_50 == 1 & risk_first_award == 1 & exclude_own_r1_winner == 0"

capture program drop collect_reg_stats
program define collect_reg_stats, rclass
    version 16
    syntax , Outcome(name) Coefvar(name)

    tempvar esample district_tag
    gen byte `esample' = e(sample)
    egen byte `district_tag' = tag(district_panel_id) if `esample'

    quietly count if `esample'
    return scalar obs = r(N)

    quietly count if `district_tag' == 1
    return scalar districts = r(N)

    quietly summarize `outcome' if `esample', meanonly
    return scalar outcome_mean = r(mean)

    return scalar coef = _b[`coefvar']
    return scalar se = _se[`coefvar']
    return scalar p = 2 * normal(-abs(_b[`coefvar'] / _se[`coefvar']))
end

capture program drop run_fwl_iv
program define run_fwl_iv, rclass
    version 16
    syntax , Sample(string asis) Outcome(name) Endog(name) Instrument(name) Controls(string asis)

    tempvar y_resid p_resid z_resid esample district_tag

    quietly areg `outcome' `controls' i.year if `sample', absorb(district_panel_id)
    predict double `y_resid' if e(sample), resid

    quietly areg `endog' `controls' i.year if `sample', absorb(district_panel_id)
    predict double `p_resid' if e(sample), resid

    quietly areg `instrument' `controls' i.year if `sample', absorb(district_panel_id)
    predict double `z_resid' if e(sample), resid

    ivregress 2sls `y_resid' (`p_resid' = `z_resid') if `sample', nocons vce(cluster district_panel_id)

    gen byte `esample' = e(sample)
    egen byte `district_tag' = tag(district_panel_id) if `esample'

    quietly count if `esample'
    return scalar obs = r(N)

    quietly count if `district_tag' == 1
    return scalar districts = r(N)

    quietly summarize `outcome' if `esample', meanonly
    return scalar outcome_mean = r(mean)

    return scalar coef = _b[`p_resid']
    return scalar se = _se[`p_resid']
    return scalar p = 2 * normal(-abs(_b[`p_resid'] / _se[`p_resid']))
end

tempfile specc_results
tempname specc_post
postfile `specc_post' str34 model str24 sample str24 dependent ///
    str34 coefficient_on double coef se p double first_stage_f ///
    long obs districts double outcome_mean ///
    using `specc_results', replace

* Main sample: first stage.
areg `p' c.`z' `controls' i.year if `main_sample', absorb(district_panel_id) vce(cluster district_panel_id)
collect_reg_stats, outcome(`p') coefvar(`z')
local fs_coef_main = r(coef)
local fs_se_main = r(se)
local fs_p_main = r(p)
local fs_obs_main = r(obs)
local fs_districts_main = r(districts)
local fs_mean_main = r(outcome_mean)
test `z'
local fs_f_main = r(F)
post `specc_post' ("First stage") ("Main risk set") ("Peer awards by t-1") ///
    ("Simulated design R1 shock") (`fs_coef_main') (`fs_se_main') (`fs_p_main') (`fs_f_main') ///
    (`fs_obs_main') (`fs_districts_main') (`fs_mean_main')

* Main sample: reduced form.
areg `y' c.`z' `controls' i.year if `main_sample', absorb(district_panel_id) vce(cluster district_panel_id)
collect_reg_stats, outcome(`y') coefvar(`z')
post `specc_post' ("Reduced form") ("Main risk set") ("First award hazard") ///
    ("Simulated design R1 shock") (r(coef)) (r(se)) (r(p)) (.) ///
    (r(obs)) (r(districts)) (r(outcome_mean))

* Main sample: design-BH IV via FWL residualization.
run_fwl_iv, sample(`main_sample') outcome(`y') endog(`p') instrument(`z') controls(`controls')
post `specc_post' ("2SLS") ("Main risk set") ("First award hazard") ///
    ("Peer awards by t-1") (r(coef)) (r(se)) (r(p)) (`fs_f_main') ///
    (r(obs)) (r(districts)) (r(outcome_mean))

* No-isolated K6 robustness: first-stage F and design-BH IV via FWL residualization.
areg `p' c.`z' `controls' i.year if `noisol_sample', absorb(district_panel_id) vce(cluster district_panel_id)
test `z'
local fs_f_noisol = r(F)

run_fwl_iv, sample(`noisol_sample') outcome(`y') endog(`p') instrument(`z') controls(`controls')
post `specc_post' ("2SLS") ("No isolated risk set") ("First award hazard") ///
    ("Peer awards by t-1") (r(coef)) (r(se)) (r(p)) (`fs_f_noisol') ///
    (r(obs)) (r(districts)) (r(outcome_mean))

postclose `specc_post'

use `specc_results', clear
export delimited using "${TABLES}/03_specC_hazard_panel_design_bh_iv.csv", replace

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

file open specc_tex using "${TABLES}/03_specC_hazard_panel_design_bh_iv.tex", write replace text
file write specc_tex "\begin{table}[!htbp]\centering" _n
file write specc_tex "\caption{Spec C: Hazard Panel Design-BH IV}" _n
file write specc_tex "\begin{tabular}{lcccc}" _n
file write specc_tex "\hline\hline" _n
file write specc_tex " & (1) & (2) & (3) & (4) \\" _n
file write specc_tex " & First stage & Reduced form & 2SLS & 2SLS no isolated \\" _n
file write specc_tex "\hline" _n
file write specc_tex "Simulated design R1 shock & `b1' & `b2' &  &  \\" _n
file write specc_tex " & `se1' & `se2' &  &  \\" _n
file write specc_tex "Peer awards by t-1, K6 &  &  & `b3' & `b4' \\" _n
file write specc_tex " &  &  & `se3' & `se4' \\" _n
file write specc_tex "\hline" _n
file write specc_tex "First-stage F & `fsf1' &  & `fsf3' & `fsf4' \\" _n
file write specc_tex "Observations & `obs1' & `obs2' & `obs3' & `obs4' \\" _n
file write specc_tex "Districts & `districts1' & `districts2' & `districts3' & `districts4' \\" _n
file write specc_tex "Mean dependent variable & `mean1' & `mean2' & `mean3' & `mean4' \\" _n
file write specc_tex "District FE & Yes & Yes & Yes & Yes \\" _n
file write specc_tex "Year FE & Yes & Yes & Yes & Yes \\" _n
file write specc_tex "Expected R1 exposure control & Yes & Yes & Yes & Yes \\" _n
file write specc_tex "Focal R1 winners excluded & Yes & Yes & Yes & Yes \\" _n
file write specc_tex "Focal R3 controls & No & No & No & No \\" _n
file write specc_tex "\hline\hline" _n
file write specc_tex "\end{tabular}" _n
file write specc_tex "\begin{flushleft}\footnotesize Notes: Outcome in columns 2--4 is first ESB award in district-year t, restricted to districts still at risk of first award and excluding focal districts that won R1 rebates. The endogenous peer variable is the number of six EDGE-nearest neighboring districts with first awards by t-1. The instrument is recentered K6 neighbor R1 exposure: realized neighbor R1 wins minus simulated design-expected neighbor R1 wins, switched on for post-R1 years. The simulation reconstructs the 2022 R1 selection rule using the observed applicant universe, priority status, fuel-pool requests, funding amounts, state/territory first-selection steps, pool budgets calibrated to observed 2022 R1 awards, and the 10 percent state cap. All columns control for simulated design-expected K6 neighbor R1 exposure. Focal R3 timing is not controlled in the baseline because it may be a downstream response to nearby R1 exposure. The 2SLS columns are computed after Frisch-Waugh-Lovell residualization of district fixed effects, year fixed effects, and the expected-exposure control. Column 4 drops districts whose sixth EDGE neighbor is more than 50 miles away. Standard errors are clustered by district.\end{flushleft}" _n
file write specc_tex "\end{table}" _n
file close specc_tex

list, noobs abbreviate(32)

log close
