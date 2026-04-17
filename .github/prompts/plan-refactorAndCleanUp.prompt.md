## Plan: Refactor and Analysis Migration to Cleaned_Branch

**TL;DR** We will branch from `Fresh-Start`, preserve the Python logic for complex spatial/data wrangling (but flatten the over-engineered functions), and wholesale replace the `3_Estimation` Python scripts with Stata (`.do`) or R (`.R`) files. We will output the final estimation data as `.dta` files specifically formatted for Stata/R consumption. 

**Steps**
1. **Branch Initialization**
   - Create `Cleaned_Branch` from `Fresh-Start` to keep recent spatial weight updates (like the $K$, Radius iterations).
2. **Consolidate Data Preparation (Python)**
   - Keep `1_Preparation/` and `2_Preliminaries/` in Python (Pandas/Geopandas are excellent for nearest-neighbor logic).
   - *Refactor:* Strip out unnecessary functions and OOP wrappers. Convert them to flat, highly commented, top-to-bottom execution scripts.
   - *Handoff:* Add a final export step to write the cleaned panel data as `analysis_panel_dataset_spatial.dta` for seamless consumption by Stata/R.
3. **Migrate Estimation (Stata)** *(depends on 2)*
   - *Delete:* Remove all Python IV scripts (`01_main_iv_panel.py`, etc.).
   - *Consolidate:* Create a single `01_main_estimation.do` that handles both the main IV (with baseline $0.0063$ expectation) and the mechanism tests using command-line arguments or simple toggle variables. Section and comment the code heavily to explain the logic and ensure future maintainability. Organize the code into clear sections:
     - *Data Import:* Load the `.dta` file exported from Python.
     - *Main IV Estimation:* Run the core spatial IV regression, ensuring to replicate the exact specifications (e.g., fixed effects, clustering).
     - *Mechanism Tests:* Sequentially run the mechanism tests (e.g., `Y_first_operating`, `Y_ever_operating`) with clear comments on the logic and expected outcomes.
     - *Robustness Checks:* Include any additional robustness checks (e.g., alternative spatial weights, different clustering levels) in the same script or a separate `02_robustness_checks.do` for clarity.
   - *Event Studies:* Create `02_event_study.do` using standard event-study packages. 
4. **Identify and Document Econometric Inconsistencies** *(parallel with 3)*
   - When rewriting to Stata/R, strictly check for "singleton drops" (Stata's `reghdfe` drops singletons by default; Python's `linearmodels` often does not). This is the biggest source of F-stat and SE discrepancies (e.g., explaining any deviation from the Python F-stat of 592).
5. **Output Standardization**
   - Use `estout` / `esttab` (Stata) or `modelsummary` (R) to pipe beautifully formatted tables directly into `3_Output/Tables/` and `Logs/`.

**Relevant files**
- `2_Scripts/1_Preparation/*.py` — Flatten logic, improve comments, export to `.dta` format.
- `2_Scripts/3_Estimation/` — Total restructure. Drop Python files; initialize streamlined `.do`/`.R` files.
- `3_Output/` — Standardize exports entirely.

**Verification**
1. Verify the Stata/R IV first-stage F-stat perfectly matches or explicitly explains the deviation from the Python output (F=592.61).
2. Try running the code to see the code works (and iteratively fix any bugs that arise).  
3. Verify point estimates for the core spatial weights ($K=6$) remain identical/robust.
4. Validate that the script count in `3_Estimation` is significantly reduced (ideally just 2-3 core files).
5. Ensure all outputs (tables, logs) are cleanly formatted and stored in the appropriate directories.
6. Document any discrepancies or changes in the estimation results a summary in the `CHANGES.md` for future reference.

**Decisions**
- **Branch:** Starting from `Fresh-Start`.
- **Handoff:** Python will export a `.dta` file to seamlessly connect the data prep to Stata/R estimation.
- **Language/Execution:** We are keeping Python for data prep, but fully migrating `3_Estimation` to Stata. Stata executes natively using `"C:\Stata16\StataMP-64.exe"`.

**Further Considerations**
1. **Event Study Baseline:** We established earlier there was confusion about the excluded year (2021 vs 2022). As we refactor, we should hardcode and comment prominently the logic for the 2022 baseline to prevent future confusion.
2. **Mechanism Variables:** The `Y_first_operating` mechanism variable needs to be explicitly merged and checked in the Python-to-DTA handoff so we can cleanly run that missing specification in Stata/R.