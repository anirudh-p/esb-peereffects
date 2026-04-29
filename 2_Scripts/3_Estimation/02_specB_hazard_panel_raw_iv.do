* Spec B: Hazard Panel Raw IV 2SLS
* Purpose: bridge between descriptive OLS and the design-based IV.

* ------------------------------------------------------------------------------
* Preamble
* ------------------------------------------------------------------------------
do "2_Scripts/3_Estimation/00_globals.do"

capture log close
log using "${LOGS}/02_specB_hazard_panel_raw_iv_log.txt", replace text

section_header, title("Spec B: Hazard Panel Raw IV") ///
    detail("Bridge specification using realized neighbor R1 wins as the excluded instrument.")

* ------------------------------------------------------------------------------
* Setup
* ------------------------------------------------------------------------------
prepare_spatial_panel

local outcome_first_award "y_first_award"
local peer_award_k6_lag "edge_w6_award_tm1_n"
local instrument_r1win_k6 "edge_w6_r1win_tm1_n"
local sample_main_risk "main_estimation_sample == 1 & risk_first_award == 1 & exclude_own_r1_winner == 0"
local sample_noisol_risk "main_noisol_edge_k6_50 == 1 & risk_first_award == 1 & exclude_own_r1_winner == 0"

label variable `peer_award_k6_lag' "Lagged EDGE K6 peer first-award count"
label variable `instrument_r1win_k6' "Lagged EDGE K6 realized neighbor R1 win count"

tempfile specb_results
tempname specb_post
postfile `specb_post' str34 model str24 sample str24 dependent ///
    str28 coefficient_on double coef se p double first_stage_f ///
    long obs districts double outcome_mean ///
    using `specb_results', replace

* ------------------------------------------------------------------------------
* Specification
* Run first-stage, reduced-form, and FWL 2SLS estimates with short log output.
* ------------------------------------------------------------------------------
section_header, title("Specification") ///
    detail("Estimate main-sample and no-isolate raw IV variants.")

quietly areg `peer_award_k6_lag' c.`instrument_r1win_k6' i.year ///
    if `sample_main_risk', absorb(district_panel_id) vce(cluster district_panel_id)
collect_estimation_stats, outcome(`peer_award_k6_lag') coefvar(`instrument_r1win_k6')
local first_stage_coef_main = r(coef)
local first_stage_se_main = r(se)
local first_stage_p_main = r(p)
local first_stage_obs_main = r(obs)
local first_stage_districts_main = r(districts)
local first_stage_mean_main = r(outcome_mean)
test `instrument_r1win_k6'
local first_stage_f_main = r(F)
post `specb_post' ("First stage") ("Main risk set") ("Peer awards by t-1") ///
    ("Raw neighbor R1 wins") (`first_stage_coef_main') (`first_stage_se_main') ///
    (`first_stage_p_main') (`first_stage_f_main') ///
    (`first_stage_obs_main') (`first_stage_districts_main') (`first_stage_mean_main')
di as res "First stage, main sample: coef = " %9.4f `first_stage_coef_main' ///
    ", se = " %9.4f `first_stage_se_main' ", F = " %9.2f `first_stage_f_main'

quietly areg `outcome_first_award' c.`instrument_r1win_k6' i.year ///
    if `sample_main_risk', absorb(district_panel_id) vce(cluster district_panel_id)
collect_estimation_stats, outcome(`outcome_first_award') coefvar(`instrument_r1win_k6')
post `specb_post' ("Reduced form") ("Main risk set") ("First award hazard") ///
    ("Raw neighbor R1 wins") (r(coef)) (r(se)) (r(p)) (.) ///
    (r(obs)) (r(districts)) (r(outcome_mean))
di as res "Reduced form, main sample: coef = " %9.4f r(coef) ///
    ", se = " %9.4f r(se) ", p = " %9.4f r(p)

run_fwl_iv, sample(`sample_main_risk') outcome(`outcome_first_award') ///
    endogenous(`peer_award_k6_lag') instrument(`instrument_r1win_k6')
post `specb_post' ("2SLS") ("Main risk set") ("First award hazard") ///
    ("Peer awards by t-1") (r(coef)) (r(se)) (r(p)) (`first_stage_f_main') ///
    (r(obs)) (r(districts)) (r(outcome_mean))
di as res "2SLS, main sample: coef = " %9.4f r(coef) ///
    ", se = " %9.4f r(se) ", p = " %9.4f r(p)

quietly areg `peer_award_k6_lag' c.`instrument_r1win_k6' i.year ///
    if `sample_noisol_risk', absorb(district_panel_id) vce(cluster district_panel_id)
test `instrument_r1win_k6'
local first_stage_f_noisol = r(F)

run_fwl_iv, sample(`sample_noisol_risk') outcome(`outcome_first_award') ///
    endogenous(`peer_award_k6_lag') instrument(`instrument_r1win_k6')
post `specb_post' ("2SLS") ("No isolated risk set") ("First award hazard") ///
    ("Peer awards by t-1") (r(coef)) (r(se)) (r(p)) (`first_stage_f_noisol') ///
    (r(obs)) (r(districts)) (r(outcome_mean))
di as res "2SLS, no-isolate sample: coef = " %9.4f r(coef) ///
    ", se = " %9.4f r(se) ", p = " %9.4f r(p)

postclose `specb_post'

* ------------------------------------------------------------------------------
* Export
* ------------------------------------------------------------------------------
section_header, title("Export") ///
    detail("Write compact CSV and LaTeX outputs for the raw-IV bridge specification.")

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
file write specb_tex "Lagged EDGE K6 neighbor R1 wins & `b1' & `b2' &  &  \\" _n
file write specb_tex " & `se1' & `se2' &  &  \\" _n
file write specb_tex "Lagged EDGE K6 peer awards &  &  & `b3' & `b4' \\" _n
file write specb_tex " &  &  & `se3' & `se4' \\" _n
file write specb_tex "\hline" _n
file write specb_tex "First-stage F & `fsf1' &  & `fsf3' & `fsf4' \\" _n
file write specb_tex "Observations & `obs1' & `obs2' & `obs3' & `obs4' \\" _n
file write specb_tex "Districts & `districts1' & `districts2' & `districts3' & `districts4' \\" _n
file write specb_tex "Mean dependent variable & `mean1' & `mean2' & `mean3' & `mean4' \\" _n
file write specb_tex "District FE & Yes & Yes & Yes & Yes \\" _n
file write specb_tex "Year FE & Yes & Yes & Yes & Yes \\" _n
file write specb_tex "Focal R1 winners excluded & Yes & Yes & Yes & Yes \\" _n
file write specb_tex "Focal R3 controls & No & No & No & No \\" _n
file write specb_tex "\hline\hline" _n
file write specb_tex "\end{tabular}" _n
file write specb_tex "\begin{flushleft}\footnotesize Notes: Outcome in columns 2--4 is first ESB award in district-year t, restricted to districts still at risk of first award and excluding focal districts that won R1 rebates. The endogenous peer variable is the number of six EDGE-nearest neighboring districts with first awards by t-1. The raw instrument is the number of six EDGE-nearest neighboring districts that won R1 rebates, switched on for post-R1 years. Focal R3 timing is not controlled in the baseline because it may be a downstream response to nearby R1 exposure. The 2SLS columns are computed after Frisch-Waugh-Lovell residualization of district and year fixed effects. Column 4 drops districts whose sixth EDGE neighbor is more than 50 miles away. Standard errors are clustered by district. This is a raw-IV bridge specification; the design-BH IV is the intended causal specification.\end{flushleft}" _n
file write specb_tex "\end{table}" _n
file close specb_tex

list, noobs abbreviate(32)

log close
