* Sample-selection table and first descriptive counts for the Brown pitch.

* ------------------------------------------------------------------------------
* Preamble
* ------------------------------------------------------------------------------
do "2_Scripts/3_Estimation/00_globals.do"

capture log close
log using "${LOGS}/01_descriptive_statistics_log.txt", replace text

section_header, title("Descriptive Statistics") ///
    detail("Build the sample-selection table and first-award risk-set counts.")

* ------------------------------------------------------------------------------
* Setup
* ------------------------------------------------------------------------------
prepare_spatial_panel

* ------------------------------------------------------------------------------
* Construction
* Tabulate the staged sample restrictions used in the Brown hazard-panel analyses.
* ------------------------------------------------------------------------------
capture program drop sample_stage_counts
program define sample_stage_counts, rclass
    version 16
    syntax , Condition(string asis)

    preserve
    keep if `condition'

    quietly count
    return scalar district_year_rows = r(N)

    quietly egen byte district_tag = tag(nces_id)
    quietly count if district_tag == 1
    return scalar districts = r(N)

    quietly count if y_first_award == 1
    return scalar first_award_events = r(N)

    quietly count if y_first_lottery_apply == 1
    return scalar first_lottery_apply_events = r(N)

    quietly count if risk_first_award == 1
    return scalar award_risk_rows = r(N)
    restore
end

tempfile sample_selection
tempname sample_post
postfile `sample_post' str52 stage ///
    long district_year_rows districts first_award_events ///
    long first_lottery_apply_events award_risk_rows ///
    using `sample_selection', replace

sample_stage_counts, condition(1)
post `sample_post' ("All district-years") ///
    (r(district_year_rows)) (r(districts)) (r(first_award_events)) ///
    (r(first_lottery_apply_events)) (r(award_risk_rows))

sample_stage_counts, condition(spatial_eligible_hybrid == 1)
post `sample_post' ("Hybrid geometry") ///
    (r(district_year_rows)) (r(districts)) (r(first_award_events)) ///
    (r(first_lottery_apply_events)) (r(award_risk_rows))

sample_stage_counts, condition(spatial_eligible_edge == 1)
post `sample_post' ("EDGE geometry") ///
    (r(district_year_rows)) (r(districts)) (r(first_award_events)) ///
    (r(first_lottery_apply_events)) (r(award_risk_rows))

sample_stage_counts, condition(spatial_eligible_edge == 1 & has_core_controls == 1)
post `sample_post' ("EDGE plus core controls") ///
    (r(district_year_rows)) (r(districts)) (r(first_award_events)) ///
    (r(first_lottery_apply_events)) (r(award_risk_rows))

sample_stage_counts, condition(focal_regular_public == 1 & spatial_eligible_edge == 1 & has_core_controls == 1)
post `sample_post' ("Regular public EDGE plus controls") ///
    (r(district_year_rows)) (r(districts)) (r(first_award_events)) ///
    (r(first_lottery_apply_events)) (r(award_risk_rows))

sample_stage_counts, condition(main_estimation_sample == 1)
post `sample_post' ("Main sample") ///
    (r(district_year_rows)) (r(districts)) (r(first_award_events)) ///
    (r(first_lottery_apply_events)) (r(award_risk_rows))

sample_stage_counts, condition(main_estimation_sample == 1 & risk_first_award == 1)
post `sample_post' ("Main first-award risk set") ///
    (r(district_year_rows)) (r(districts)) (r(first_award_events)) ///
    (r(first_lottery_apply_events)) (r(award_risk_rows))

sample_stage_counts, condition(main_noisol_edge_k6_50 == 1 & risk_first_award == 1)
post `sample_post' ("No isolated K6 risk set") ///
    (r(district_year_rows)) (r(districts)) (r(first_award_events)) ///
    (r(first_lottery_apply_events)) (r(award_risk_rows))

postclose `sample_post'

* ------------------------------------------------------------------------------
* Export
* ------------------------------------------------------------------------------
section_header, title("Export") ///
    detail("Write compact CSV and LaTeX sample-selection tables.")

use `sample_selection', clear
export delimited using "${TABLES}/00_sample_selection.csv", replace

file open sample_tex using "${TABLES}/00_sample_selection.tex", write replace text
file write sample_tex "\begin{table}[!htbp]\centering" _n
file write sample_tex "\caption{Sample Selection for Hazard-Panel Estimation}" _n
file write sample_tex "\begin{tabular}{lrrrrr}" _n
file write sample_tex "\hline\hline" _n
file write sample_tex "Stage & District-years & Districts & First awards & First applications & Award-risk rows \\" _n
file write sample_tex "\hline" _n

forvalues i = 1/`=_N' {
    local stage = stage[`i']
    local rows = district_year_rows[`i']
    local districts = districts[`i']
    local awards = first_award_events[`i']
    local apps = first_lottery_apply_events[`i']
    local risk = award_risk_rows[`i']
    file write sample_tex "`stage' & " %12.0fc (`rows') " & " %12.0fc (`districts') " & " ///
        %12.0fc (`awards') " & " %12.0fc (`apps') " & " %12.0fc (`risk') " \\" _n
}

file write sample_tex "\hline\hline" _n
file write sample_tex "\end{tabular}" _n
file write sample_tex "\begin{flushleft}\footnotesize Notes: Main sample is regular public districts with EDGE geometry, core controls, and contiguous-U.S. geography. The final row drops districts whose sixth EDGE KNN neighbor is more than 50 miles away.\end{flushleft}" _n
file write sample_tex "\end{table}" _n
file close sample_tex

list, noobs abbreviate(32)

log close
