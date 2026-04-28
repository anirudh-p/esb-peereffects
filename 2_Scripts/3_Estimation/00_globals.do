clear all
set more off
version 16

local cwd "`c(pwd)'"
global PROJECT_ROOT "`cwd'"
global CLEANED "${PROJECT_ROOT}/1_Data/Cleaned"
global OUTPUT "${PROJECT_ROOT}/3_Output"
global AUDIT "${OUTPUT}/Audit"
global FIGURES "${OUTPUT}/Figures"
global LOGS "${OUTPUT}/Logs"
global TABLES "${OUTPUT}/Tables"

cap mkdir "${OUTPUT}"
cap mkdir "${AUDIT}"
cap mkdir "${FIGURES}"
cap mkdir "${LOGS}"
cap mkdir "${TABLES}"

* Preferred table output is LaTeX for Beamer/Overleaf. RTF can be added for Word drafts.
