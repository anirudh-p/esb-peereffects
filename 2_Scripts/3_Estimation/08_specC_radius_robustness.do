* Spec C Radius Robustness: Hazard Panel Design-BH IV.
* Purpose: compare simulated design-BH IV using EDGE radius graphs.
do "2_Scripts/3_Estimation/00_globals.do"

capture log close
log using "${LOGS}/08_specC_radius_robustness_log.txt", replace text

use "${SPATIAL_PANEL}", clear

local y "y_first_award"
local main_base "main_estimation_sample == 1 & risk_first_award == 1 & exclude_own_r1_winner == 0"
local radii "15 30 60"

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

tempfile radius_results
tempname radius_post
postfile `radius_post' int radius double iv_coef iv_se iv_p ///
    double fs_coef fs_se fs_p fs_f mean_neighbors ///
    long obs districts double outcome_mean ///
    using `radius_results', replace

foreach R of local radii {
    local p "edge_r`R'_award_tm1_n"
    local z "edge_r`R'_r1rcsim_tm1_n"
    local expected "edge_r`R'_r1expsim_tm1_n"
    local degree "edge_r`R'_degree_n"
    local sample "`main_base' & `degree' > 0"

    areg `p' c.`z' c.`expected' i.year if `sample', absorb(district_panel_id) vce(cluster district_panel_id)
    local fs_coef = _b[`z']
    local fs_se = _se[`z']
    local fs_p = 2 * normal(-abs(_b[`z'] / _se[`z']))
    test `z'
    local fs_f = r(F)

    quietly summarize `degree' if `sample', meanonly
    local mean_neighbors = r(mean)

    run_fwl_iv, sample(`sample') outcome(`y') endog(`p') instrument(`z') controls(c.`expected')
    post `radius_post' (`R') (r(coef)) (r(se)) (r(p)) ///
        (`fs_coef') (`fs_se') (`fs_p') (`fs_f') (`mean_neighbors') ///
        (r(obs)) (r(districts)) (r(outcome_mean))
}

postclose `radius_post'

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

    local fsb`j' : display %9.4f fs_coef[`j']
    local fsb`j' = strtrim("`fsb`j''")

    local fss`j' : display %9.4f fs_se[`j']
    local fss`j' = "(" + strtrim("`fss`j''") + ")"

    local fsf`j' : display %9.2f fs_f[`j']
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
file write radius_tex "Peer awards by t-1 & `b1' & `b2' & `b3' \\" _n
file write radius_tex " & `se1' & `se2' & `se3' \\" _n
file write radius_tex "First-stage coeff. & `fsb1' & `fsb2' & `fsb3' \\" _n
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
