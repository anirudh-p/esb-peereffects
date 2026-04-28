* Spec B: Hazard Panel Raw IV 2SLS.
* Purpose: bridge between descriptive OLS and design-based IV.
do "2_Scripts/3_Estimation/00_globals.do"

capture log close
log using "${LOGS}/02_specB_hazard_panel_raw_iv_log.txt", replace text

use "${SPATIAL_PANEL}", clear

local y "y_first_award"
local p "edge_w6_award_tm1_n"
local z "edge_w6_r1win_tm1_n"
local controls "own_r1_win_post own_r3_win_post"
local main_sample "main_estimation_sample == 1 & risk_first_award == 1"
local noisol_sample "main_noisol_edge_k6_50 == 1 & risk_first_award == 1"

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

tempfile specb_results
tempname specb_post
postfile `specb_post' str34 model str24 sample str24 dependent ///
    str28 coefficient_on double coef se p double first_stage_f ///
    long obs districts double outcome_mean ///
    using `specb_results', replace

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
post `specb_post' ("First stage") ("Main risk set") ("Peer awards by t-1") ///
    ("Raw neighbor R1 wins") (`fs_coef_main') (`fs_se_main') (`fs_p_main') (`fs_f_main') ///
    (`fs_obs_main') (`fs_districts_main') (`fs_mean_main')

* Main sample: reduced form.
areg `y' c.`z' `controls' i.year if `main_sample', absorb(district_panel_id) vce(cluster district_panel_id)
collect_reg_stats, outcome(`y') coefvar(`z')
post `specb_post' ("Reduced form") ("Main risk set") ("First award hazard") ///
    ("Raw neighbor R1 wins") (r(coef)) (r(se)) (r(p)) (.) ///
    (r(obs)) (r(districts)) (r(outcome_mean))

* Main sample: raw IV via FWL residualization.
run_fwl_iv, sample(`main_sample') outcome(`y') endog(`p') instrument(`z') controls(`controls')
post `specb_post' ("2SLS") ("Main risk set") ("First award hazard") ///
    ("Peer awards by t-1") (r(coef)) (r(se)) (r(p)) (`fs_f_main') ///
    (r(obs)) (r(districts)) (r(outcome_mean))

* No-isolated K6 robustness: first stage F and raw IV via FWL residualization.
areg `p' c.`z' `controls' i.year if `noisol_sample', absorb(district_panel_id) vce(cluster district_panel_id)
test `z'
local fs_f_noisol = r(F)

run_fwl_iv, sample(`noisol_sample') outcome(`y') endog(`p') instrument(`z') controls(`controls')
post `specb_post' ("2SLS") ("No isolated risk set") ("First award hazard") ///
    ("Peer awards by t-1") (r(coef)) (r(se)) (r(p)) (`fs_f_noisol') ///
    (r(obs)) (r(districts)) (r(outcome_mean))

postclose `specb_post'

use `specb_results', clear
export delimited using "${TABLES}/02_specB_hazard_panel_raw_iv.csv", replace

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

file open specb_tex using "${TABLES}/02_specB_hazard_panel_raw_iv.tex", write replace text
file write specb_tex "\begin{table}[!htbp]\centering" _n
file write specb_tex "\caption{Spec B: Hazard Panel Raw IV}" _n
file write specb_tex "\begin{tabular}{lcccc}" _n
file write specb_tex "\hline\hline" _n
file write specb_tex " & (1) & (2) & (3) & (4) \\" _n
file write specb_tex " & First stage & Reduced form & 2SLS & 2SLS no isolated \\" _n
file write specb_tex "\hline" _n
file write specb_tex "Raw neighbor R1 wins & `b1' & `b2' &  &  \\" _n
file write specb_tex " & `se1' & `se2' &  &  \\" _n
file write specb_tex "Peer awards by t-1, K6 &  &  & `b3' & `b4' \\" _n
file write specb_tex " &  &  & `se3' & `se4' \\" _n
file write specb_tex "\hline" _n
file write specb_tex "First-stage F & `fsf1' &  & `fsf3' & `fsf4' \\" _n
file write specb_tex "Observations & `obs1' & `obs2' & `obs3' & `obs4' \\" _n
file write specb_tex "Districts & `districts1' & `districts2' & `districts3' & `districts4' \\" _n
file write specb_tex "Mean dependent variable & `mean1' & `mean2' & `mean3' & `mean4' \\" _n
file write specb_tex "District FE & Yes & Yes & Yes & Yes \\" _n
file write specb_tex "Year FE & Yes & Yes & Yes & Yes \\" _n
file write specb_tex "Own rebate-win controls & Yes & Yes & Yes & Yes \\" _n
file write specb_tex "\hline\hline" _n
file write specb_tex "\end{tabular}" _n
file write specb_tex "\begin{flushleft}\footnotesize Notes: Outcome in columns 2--4 is first ESB award in district-year t, restricted to districts still at risk of first award. The endogenous peer variable is the number of six EDGE-nearest neighboring districts with first awards by t-1. The raw instrument is the number of six EDGE-nearest neighboring districts that won R1 rebates, switched on for post-R1 years. The 2SLS columns are computed after Frisch-Waugh-Lovell residualization of district fixed effects, year fixed effects, and own rebate-win timing controls. Column 4 drops districts whose sixth EDGE neighbor is more than 50 miles away. Standard errors are clustered by district. This is a raw-IV bridge specification; the design-BH IV is the intended causal specification.\end{flushleft}" _n
file write specb_tex "\end{table}" _n
file close specb_tex

list, noobs abbreviate(32)

log close
