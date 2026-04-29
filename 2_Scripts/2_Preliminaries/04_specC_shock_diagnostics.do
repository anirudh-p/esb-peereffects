* Diagnostic table for Spec C sample composition and recentered-neighbor shock.

* ------------------------------------------------------------------------------
* Preamble
* ------------------------------------------------------------------------------
do "2_Scripts/3_Estimation/00_globals.do"

capture log close
log using "${LOGS}/04_specC_shock_diagnostics_log.txt", replace text

section_header, title("Spec C Shock Diagnostics") ///
    detail("Audit whether the Spec C onset sample is mechanically tilted toward positive or negative recentered neighbor shock.")

* ------------------------------------------------------------------------------
* Setup
* Load the labeled spatial panel and merge in locale for the 2023 onset cross-section.
* ------------------------------------------------------------------------------
prepare_spatial_panel

tempfile base_locale
preserve
use "${CLEANED}/analysis_district_base.dta", clear
keep nces_id locale_broad
duplicates drop nces_id, force
save `base_locale', replace
restore
merge m:1 nces_id using `base_locale', nogen keep(master match)

keep if year == 2023 & main_estimation_sample == 1

gen byte specc_included = risk_first_award == 1 & exclude_own_r1_winner == 0
label variable specc_included "Included in Spec C 2023 risk set"

gen str28 sample_group = ""
replace sample_group = "Included Spec C" if specc_included == 1
replace sample_group = "Excluded: own R1 winner" if specc_included == 0 & exclude_own_r1_winner == 1
replace sample_group = "Excluded: pre-2023 adopter" if specc_included == 0 & exclude_own_r1_winner == 0 & risk_first_award == 0
replace sample_group = "Excluded: other" if sample_group == ""

xtile expected_tercile = edge_w6_r1expsim_tm1_n if specc_included == 1, nq(3)
gen str18 expected_group = ""
replace expected_group = "Low expected" if expected_tercile == 1
replace expected_group = "Mid expected" if expected_tercile == 2
replace expected_group = "High expected" if expected_tercile == 3

gen str20 locale_group = locale_broad
replace locale_group = "Missing" if trim(locale_group) == ""

label variable edge_w6_r1rcsim_tm1_n "Lagged EDGE K6 recentered simulated R1 shock"
label variable edge_w6_r1expsim_tm1_n "Lagged EDGE K6 simulated expected R1 wins"
label variable edge_w6_r1win_tm1_n "Lagged EDGE K6 realized neighbor R1 wins"

* ------------------------------------------------------------------------------
* Construction
* Summarize shock composition by sample status, expected-exposure tercile, and locale.
* ------------------------------------------------------------------------------
capture program drop collect_diag_stats
program define collect_diag_stats, rclass
    version 16
    syntax , Condition(string asis)

    quietly count if `condition'
    local districts = r(N)
    return scalar districts = `districts'

    quietly summarize edge_w6_r1rcsim_tm1_n if `condition', meanonly
    return scalar mean_shock = r(mean)

    quietly summarize edge_w6_r1expsim_tm1_n if `condition', meanonly
    return scalar mean_expected = r(mean)

    quietly summarize edge_w6_r1win_tm1_n if `condition', meanonly
    return scalar mean_realized = r(mean)

    quietly summarize y_first_award if `condition', meanonly
    return scalar mean_adopt_2023 = r(mean)

    quietly count if `condition' & edge_w6_r1rcsim_tm1_n > 0
    return scalar share_pos = r(N) / `districts'

    quietly count if `condition' & edge_w6_r1rcsim_tm1_n < 0
    return scalar share_neg = r(N) / `districts'
end

tempfile diagnostics
tempname diag_post
postfile `diag_post' str26 panel str28 group ///
    long districts double mean_shock share_pos share_neg ///
    double mean_expected mean_realized mean_adopt_2023 ///
    using `diagnostics', replace

collect_diag_stats, condition(1)
post `diag_post' ("A. Sample status") ("All main-sample districts") ///
    (r(districts)) (r(mean_shock)) (r(share_pos)) (r(share_neg)) ///
    (r(mean_expected)) (r(mean_realized)) (r(mean_adopt_2023))

collect_diag_stats, condition(sample_group == "Included Spec C")
post `diag_post' ("A. Sample status") ("Included Spec C") ///
    (r(districts)) (r(mean_shock)) (r(share_pos)) (r(share_neg)) ///
    (r(mean_expected)) (r(mean_realized)) (r(mean_adopt_2023))

collect_diag_stats, condition(sample_group == "Excluded: own R1 winner")
post `diag_post' ("A. Sample status") ("Excluded: own R1 winner") ///
    (r(districts)) (r(mean_shock)) (r(share_pos)) (r(share_neg)) ///
    (r(mean_expected)) (r(mean_realized)) (r(mean_adopt_2023))

collect_diag_stats, condition(sample_group == "Excluded: pre-2023 adopter")
post `diag_post' ("A. Sample status") ("Excluded: pre-2023 adopter") ///
    (r(districts)) (r(mean_shock)) (r(share_pos)) (r(share_neg)) ///
    (r(mean_expected)) (r(mean_realized)) (r(mean_adopt_2023))

foreach grp in "Low expected" "Mid expected" "High expected" {
    collect_diag_stats, condition(specc_included == 1 & expected_group == "`grp'")
    post `diag_post' ("B. Expected tercile") ("`grp'") ///
        (r(districts)) (r(mean_shock)) (r(share_pos)) (r(share_neg)) ///
        (r(mean_expected)) (r(mean_realized)) (r(mean_adopt_2023))
}

levelsof locale_group if specc_included == 1, local(locale_levels)
foreach grp of local locale_levels {
    collect_diag_stats, condition(specc_included == 1 & locale_group == "`grp'")
    post `diag_post' ("C. Locale") ("`grp'") ///
        (r(districts)) (r(mean_shock)) (r(share_pos)) (r(share_neg)) ///
        (r(mean_expected)) (r(mean_realized)) (r(mean_adopt_2023))
}

quietly corr edge_w6_r1rcsim_tm1_n edge_w6_r1expsim_tm1_n if specc_included == 1
local corr_included = r(rho)

postclose `diag_post'

* ------------------------------------------------------------------------------
* Export
* ------------------------------------------------------------------------------
section_header, title("Export") ///
    detail("Write compact CSV and LaTeX diagnostics and print the key correlation.")

use `diagnostics', clear
export delimited using "${TABLES}/prelim_specC_shock_diagnostics.csv", replace

preserve
keep if panel == "A. Sample status"
list, noobs abbreviate(32)
restore

preserve
keep if panel == "B. Expected tercile"
list, noobs abbreviate(32)
restore

preserve
keep if panel == "C. Locale"
list, noobs abbreviate(32)
restore

file open diag_tex using "${TABLES}/prelim_specC_shock_diagnostics.tex", write replace text
file write diag_tex "\begin{table}[!htbp]\centering" _n
file write diag_tex "\caption{Spec C Shock Diagnostics at the 2023 Onset}" _n
file write diag_tex "\begin{tabular}{lrrrrrr}" _n
file write diag_tex "\hline\hline" _n
file write diag_tex "Group & Districts & Mean shock & Share $>$ 0 & Share $<$ 0 & Mean expected & Mean realized \\" _n
file write diag_tex "\hline" _n

local current_panel ""
forvalues i = 1/`=_N' {
    local next_panel = panel[`i']
    if "`next_panel'" != "`current_panel'" {
        if "`current_panel'" != "" file write diag_tex "\hline" _n
        file write diag_tex "\multicolumn{7}{l}{\textit{`next_panel'}} \\" _n
        local current_panel "`next_panel'"
    }
    local row_group = group[`i']
    file write diag_tex "`row_group'" " & " %12.0fc (districts[`i']) " & " ///
        %9.4f (mean_shock[`i']) " & " %9.3f (share_pos[`i']) " & " ///
        %9.3f (share_neg[`i']) " & " %9.4f (mean_expected[`i']) " & " ///
        %9.4f (mean_realized[`i']) " \\" _n
}

file write diag_tex "\hline\hline" _n
file write diag_tex "\end{tabular}" _n
file write diag_tex "\begin{flushleft}\footnotesize Notes: Diagnostics use the 2023 cross-section for main-estimation-sample districts so the R1 recentered neighbor shock is measured at the onset of the post-R1 period. Mean shock is the simulated design recentered K6 EDGE neighbor R1 shock, mean expected is simulated design-expected K6 EDGE neighbor R1 wins, and mean realized is realized K6 EDGE neighbor R1 wins. Locale rows are restricted to districts included in the Spec C 2023 risk set, with locale merged from the cleaned district base file.\end{flushleft}" _n
file write diag_tex "\end{table}" _n
file close diag_tex

di as res "Correlation between recentered shock and expected exposure in included sample: " %9.4f `corr_included'

log close
