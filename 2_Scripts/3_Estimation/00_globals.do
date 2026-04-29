clear all
set more off
set linesize 180
version 16

local cwd "`c(pwd)'"
global PROJECT_ROOT "`cwd'"
global CLEANED "${PROJECT_ROOT}/1_Data/Cleaned"
global OUTPUT "${PROJECT_ROOT}/3_Output"
global AUDIT "${OUTPUT}/Audit"
global FIGURES "${OUTPUT}/Figures"
global LOGS "${OUTPUT}/Logs"
global TABLES "${OUTPUT}/Tables"
global SPATIAL_PANEL "${CLEANED}/analysis_hazard_panel_spatial.dta"
global CORE_CONTROLS "ln_students ln_income poverty_rate pct_white_alone pm25 pct_dem_2020"

cap mkdir "${OUTPUT}"
cap mkdir "${AUDIT}"
cap mkdir "${FIGURES}"
cap mkdir "${LOGS}"
cap mkdir "${TABLES}"

capture program drop section_header
program define section_header
    version 16
    syntax , Title(string) [Detail(string asis)]

    di as txt _newline "{hline 78}"
    di as txt upper("`title'")
    local detail_text `"`detail'"'
    local detail_text : subinstr local detail_text `"""' "", all
    if `"`detail_text'"' != "" di as txt `"`detail_text'"'
    di as txt "{hline 78}"
end

capture program drop label_spatial_panel_vars
program define label_spatial_panel_vars
    version 16

    capture label variable district_panel_id "Panel district identifier"
    capture label variable year "Calendar year"
    capture label variable state "State abbreviation"
    capture label variable state_id "Encoded state identifier"
    capture label variable locale_broad "Broad NCES locale category"

    capture label variable students "District enrollment"
    capture label variable ln_students "Log enrollment plus one"
    capture label variable median_income "Median household income"
    capture label variable ln_income "Log median household income plus one"
    capture label variable poverty_rate "Poverty rate"
    capture label variable pct_white_alone "Share White alone"
    capture label variable pm25 "Average PM2.5 exposure"
    capture label variable pct_dem_2020 "County Democratic vote share, 2020"

    capture label variable y_first_award "First ESB award occurs in district-year"
    capture label variable y_first_lottery_apply "First lottery application occurs in district-year"
    capture label variable risk_first_award "District remains at risk of first ESB award"
    capture label variable exclude_own_r1_winner "Focal district won an R1 rebate"
    capture label variable r1_lottery_applicant "District applied in R1"
    capture label variable own_r1_win_post "Own R1 rebate winner indicator, post-R1"
    capture label variable own_r3_win_post "Own R3 rebate winner indicator, post-R3"

    capture label variable main_estimation_sample "Main Brown estimation sample flag"
    capture label variable main_noisol_edge_k6_50 "Main sample excluding EDGE K6 isolates beyond 50 miles"
    capture label variable focal_regular_public "Regular public district focal-unit flag"

    capture label variable edge_w6_award_tm1_n "EDGE K6 neighbors with first ESB award by t-1"
    capture label variable edge_w6_r1win_tm1_n "EDGE K6 neighbors with realized R1 rebate wins"
    capture label variable edge_w6_r1expsim_tm1_n "EDGE K6 neighbors with simulated expected R1 wins"
    capture label variable edge_w6_r1rcsim_tm1_n "EDGE K6 recentered simulated R1 neighbor shock"
    capture label variable edge_w6_r1expall_tm1_n "EDGE K6 neighbors with all-apply expected R1 wins"
    capture label variable edge_w6_r1rcall_tm1_n "EDGE K6 recentered all-apply R1 neighbor shock"
    capture label variable edge_w6_nearest_mi "Distance to nearest EDGE K6 neighbor, miles"
    capture label variable edge_w6_max_mi "Distance to sixth EDGE neighbor, miles"

    capture label variable edge_r15_degree_n "EDGE neighbors within 15 miles"
    capture label variable edge_r30_degree_n "EDGE neighbors within 30 miles"
    capture label variable edge_r60_degree_n "EDGE neighbors within 60 miles"
    capture label variable edge_r15_award_tm1_n "EDGE radius-15 neighbors with first award by t-1"
    capture label variable edge_r30_award_tm1_n "EDGE radius-30 neighbors with first award by t-1"
    capture label variable edge_r60_award_tm1_n "EDGE radius-60 neighbors with first award by t-1"
    capture label variable edge_r15_r1expsim_tm1_n "EDGE radius-15 simulated expected R1 wins"
    capture label variable edge_r30_r1expsim_tm1_n "EDGE radius-30 simulated expected R1 wins"
    capture label variable edge_r60_r1expsim_tm1_n "EDGE radius-60 simulated expected R1 wins"
    capture label variable edge_r15_r1rcsim_tm1_n "EDGE radius-15 recentered simulated R1 shock"
    capture label variable edge_r30_r1rcsim_tm1_n "EDGE radius-30 recentered simulated R1 shock"
    capture label variable edge_r60_r1rcsim_tm1_n "EDGE radius-60 recentered simulated R1 shock"
end

capture program drop prepare_spatial_panel
program define prepare_spatial_panel
    version 16

    use "${SPATIAL_PANEL}", clear

    capture confirm variable state_id
    if _rc {
        capture encode state, gen(state_id)
    }

    capture confirm variable ln_students
    if _rc {
        gen double ln_students = ln(students + 1) if students >= 0
    }

    capture confirm variable ln_income
    if _rc {
        gen double ln_income = ln(median_income + 1) if median_income >= 0
    }

    xtset district_panel_id year
    label_spatial_panel_vars
end

capture program drop collect_estimation_stats
program define collect_estimation_stats, rclass
    version 16
    syntax , Outcome(name) Coefvar(name)

    tempvar estimation_row district_tag
    gen byte `estimation_row' = e(sample)
    egen byte `district_tag' = tag(district_panel_id) if `estimation_row'

    quietly count if `estimation_row'
    return scalar obs = r(N)

    quietly count if `district_tag' == 1
    return scalar districts = r(N)

    quietly summarize `outcome' if `estimation_row', meanonly
    return scalar outcome_mean = r(mean)

    return scalar coef = _b[`coefvar']
    return scalar se = _se[`coefvar']
    return scalar p = 2 * normal(-abs(_b[`coefvar'] / _se[`coefvar']))
end

capture program drop run_fwl_iv
program define run_fwl_iv, rclass
    version 16
    syntax , Sample(string asis) Outcome(name) Endogenous(name) Instrument(name) [Controls(string asis)]

    tempvar outcome_residual peer_residual instrument_residual estimation_row district_tag

    quietly areg `outcome' `controls' i.year if `sample', absorb(district_panel_id)
    predict double `outcome_residual' if e(sample), resid

    quietly areg `endogenous' `controls' i.year if `sample', absorb(district_panel_id)
    predict double `peer_residual' if e(sample), resid

    quietly areg `instrument' `controls' i.year if `sample', absorb(district_panel_id)
    predict double `instrument_residual' if e(sample), resid

    quietly ivregress 2sls `outcome_residual' ///
        (`peer_residual' = `instrument_residual') ///
        if `sample', nocons vce(cluster district_panel_id)

    gen byte `estimation_row' = e(sample)
    egen byte `district_tag' = tag(district_panel_id) if `estimation_row'

    quietly count if `estimation_row'
    return scalar obs = r(N)

    quietly count if `district_tag' == 1
    return scalar districts = r(N)

    quietly summarize `outcome' if `estimation_row', meanonly
    return scalar outcome_mean = r(mean)

    return scalar coef = _b[`peer_residual']
    return scalar se = _se[`peer_residual']
    return scalar p = 2 * normal(-abs(_b[`peer_residual'] / _se[`peer_residual']))
end

* Preferred table output is LaTeX for Beamer/Overleaf. RTF can be added later.
