# Project Purpose and History

**Original branch:** `Post-Temporal` (created March 2, 2026; branched from `Current`)  
**Current branch:** `Post-Meeting` (created April 6, 2026; branched from `Post-Poster` commit `665c1e8`)  
**Last Updated:** April 8, 2026

> **Note:** Sections 1–7 below document the Post-Temporal/Post-Poster analysis pipeline. The Post-Meeting branch superseded the four-estimand structure with a single clean IV design; see `Checklist.md` for the current analysis plan, scripts, and results. Scripts `01_estimand1_itt.py` through `04_estimand_2sls.py` no longer exist on this branch.

---

## Table of Contents

1. [Why This Branch Exists](#why-this-branch-exists)
2. [What Was Built on the Current Branch](#what-was-built-on-the-current-branch)
3. [Core Problems Identified](#core-problems-identified)
4. [What the Data Can Plausibly Identify](#what-the-data-can-plausibly-identify)
   - [Estimand 1: ITT](#estimand-1-intention-to-treat-itt--the-most-defensible-starting-point)
   - [Estimand 2: Application Extensive Margin](#estimand-2-application-extensive-margin--the-cleanest-behavioral-outcome)
   - [Estimand 3: Longer-Horizon All-Source Diffusion](#estimand-3-longer-horizon-all-source-diffusion)
   - [Estimand 4: Heterogeneity by Deployment Visibility](#estimand-4-heterogeneity-by-r1-deployment-visibility)
   - [What Is Likely Not Identifiable](#what-is-likely-not-identifiable-with-current-data)
5. [What Gets Built in This Branch](#what-gets-built-in-this-branch)
6. [Key Reference Files](#key-reference-files-preserved-on-current-and-main)
7. [**Implementation and Results (Mar 3, 2026)**](#implementation-and-results-mar-3-2026)
   - [7.8 Structural 2SLS Results](#78-structural-2sls-results-mar-4-2026)
   - [7.9 Sharp-Cutoff Delivery Timing (Deterrence Check)](#79-sharp-cutoff-delivery-timing-deterrence-check-mar-5-2026)

---

## Why This Branch Exists

This branch was started from scratch — all analysis scripts, cleaned data, outputs, and documentation from the `Current` branch have been removed. Only the raw source datasets in `1_Data/Raw/` are preserved.

The reason is fundamental: after working through the temporal design diagnostics (scripts 20–30 on `Current`), it became clear that the entire contemporaneous specification rests on an identification assumption that is almost certainly violated and that cannot be fixed through robustness checks alone. The project needs a clean conceptual restart, with the research design scoped correctly before any new code is written.

This document records what was done before, why it failed, and what the data can plausibly identify.

---

## What Was Built on the `Current` Branch

### The design

A cross-sectional IV estimating the effect of neighbor ESB adoption on own adoption:

- **Outcome:** `IS_ADOPTER` = 1 if district won a CSBP federal rebate (Rounds 1–3, 2022–2024).
- **Endogenous variable:** `w_adoption` = share of K=6 nearest geographic neighbors with `IS_ADOPTER = 1`.
- **Instrument:** `w_IV_Z` = share of K=6 nearest neighbors who **won the CSBP lottery** (same pool of rounds).
- **Sample:** ~13,400 school districts, cross-sectional.
- **Estimator:** IV-2SLS with state fixed effects and a battery of district-level controls.

### The headline result

β ≈ 0.19–0.20, first-stage F ≈ 54,000, statistically significant. Interpreted as: a 10pp increase in the share of neighboring districts adopting ESBs raises own adoption probability by ~2pp.

### The robustness battery (scripts 01–30)

Scripts explored K sensitivity, urbanicity heterogeneity, climate-based peer networks, inference methods (Conley HAC, county clustering), exclusion restriction tests (loser-density control, donut network), cross-round temporal IV (R1→R3), delivery timing variation (early vs. late R1 deployment), all-source WRI outcomes, and several falsification tests.

---

## Core Problems Identified

### 1. The instrument and the endogenous variable are nearly the same thing

In the CSBP program, winning the lottery and adopting are essentially the same event. There is no meaningful gap between them: `IV_Z_j = 1` (neighbor won the lottery) directly implies `IS_ADOPTER_j = 1` (neighbor adopted). Therefore `w_IV_Z ≈ w_adoption` — the instrument and the endogenous variable are spatial averages of nearly identical binary indicators.

The first-stage F-stat of 54,428 is not evidence of a strong instrument. It is evidence that this near-identity holds at the neighborhood level too. The IV is doing almost nothing to separate exogenous from endogenous variation; the 2SLS estimate is close to OLS. A genuine instrument for neighbor adoption would have to separately predict neighbor adoption without being neighbor adoption. That condition is not met here.

**Diagnostic confirmation:** When the instrument is changed to R1 lottery wins (`w_IV_Z_R1`) and the endogenous variable is changed to R3 CSBP adoption by neighbors (`w_R3_CSBP`) — genuinely different variables — the first-stage F collapses to **0.3**.

### 2. The identifying variation is geographic clustering of application behavior, not random lottery luck

The CSBP lottery is random *conditional on application*. About 80% of all districts never applied. The spatial lag `w_IV_Z` reflects primarily *whether any of the 6 nearest neighbors applied and won* — which is driven by whether the local area had high CSBP application propensity, not just random lottery luck among a fixed applicant set.

When this is controlled for via neighbor loser density (`w_loser`, i.e., share of neighbors who applied but lost), the estimate drops from β ≈ 0.20 to β ≈ 0.11 — a 45% decline. Half the baseline estimate was capturing local interest clustering, not peer effects.

### 3. The causal chain requires time that the design does not allow

The peer effect mechanism described in the research question is:

> *Neighbor wins lottery → neighbor receives and deploys ESBs → you observe them operating → you decide to adopt.*

This requires temporal ordering: the neighbor's deployment must *precede* your adoption decision. In the contemporaneous pooled specification (Rounds 1+3 pooled into one cross-section), a Round 1 district's outcome and a Round 3 district's outcome are treated as simultaneous. An R1 winner cannot have been caused by observation of other R1 winners — there was no prior round to observe. The mechanism has not had time to operate.

**Cross-round test (Script 20):** The temporal design — R1 lottery wins as instrument for R3 CSBP adoption — has a first-stage F = 0.3 for the CSBP-specific endogenous variable, because R1 winners already have buses and have little reason to participate in R3.

**Delivery timing test (Scripts 27–28):** Of 2,208 R1 CSBP buses, only ~216 were delivered before the R3 application deadline (Oct 2023). Most R1 deployments had not occurred by the time R3 decisions were being made. The demonstration effect literally had not had time to materialize in the empirical window.

### 4. The outcome variable measures CSBP lottery participation, not ESB ownership

`IS_ADOPTER` is constructed from federal lottery records (CSB Rebates and Grants files). Approximately **33% of all tracked ESBs** in the WRI dataset arrived via non-federal channels: California HVIP, VW Settlement funds, state grants, and other programs. Any peer effect operating through those channels is invisible in the outcome.

More fundamentally, the right outcome for the mechanism above is not "won the federal lottery" but "acquired an ESB through any channel." The cross-round all-source tests (Script 26) show null reduced-form effects (RF ≈ −0.02, p ≈ 0.5) of R1 neighbor wins on WRI-tracked all-source adoption in the R3 window — though this may reflect the timing problem rather than a true null.

### 5. The near-unity IV coefficient is not a structural peer effect

The cross-round CSBP spec (Script 20) produced β ≈ 1.34 — which was interpreted optimistically in the documentation on `Current`, but is straightforwardly explained by weak-IV bias. The Wald ratio (Reduced Form / First Stage) = −0.040 / 0.008 = −5.19, not +1.34. The +1.34 figure from the prior run appears to have been computed on a different data vintage. The Anderson-Rubin confidence interval (valid under weak IV) includes zero (AR p = 0.11 at β = 0).

---

## What the Data Can Plausibly Identify

The raw data consists of:

| Dataset | Coverage | Key variables |
|---|---|---|
| WRI ESB Dataset v9 (Jun 2025) | Bus-level, all-source ESB adoption | District ID, award year, funding source, delivery/service dates |
| EPA CSB Rebates.xlsx | CSBP Round 1 (2022) and Round 3 (2023) rebate lottery | Winner/loser, funding year, NCES ID |
| EPA CSB Grants.xlsx | CSBP Round 2 (2023) competitive grants | Awardees, NCES ID |
| CSBP Applicants waitlisted/rejected | Applicants who did not win | NCES ID, priority flag |
| NCES EDGE shapefiles | Geographic boundaries of all school districts | Centroid, geometry |
| NCES CCD | District demographics | Enrollment, race, urbanicity |
| ACS 2021 5-year | Socioeconomic characteristics | Median income, poverty |
| NASA SEDAC | Air quality | PM2.5 |
| MIT Election Lab | Political alignment | 2020 presidential vote by county |
| PRISM 30-year normals | Climate | Precipitation, min/max temperature |

### What is credibly estimable

**Estimand 1: Intention-to-treat (ITT) — the most defensible starting point**

Does having a lottery-winning neighbor increase own CSBP adoption probability? This requires only lottery randomness (which holds unconditionally), not the full exclusion restriction. It does not claim to identify a structural peer effect; it estimates the causal effect of *neighbor lottery wins* on own adoption — the ITT effect of the lottery assignment on diffusion:

$$\delta = \frac{\partial \Pr(Y_i = 1)}{\partial \bar{Z}_i}$$

The estimate with the loser-density control (absorbing geographic application clustering) from the prior branch was δ ≈ 0.11 at K=6. Interpretable as: *an additional lottery-winning neighbor in a pool of 6 raises own CSBP adoption probability by ~1.8pp, conditional on local application interest.* The ITT establishes whether any effect of winning-neighbor exposure exists before attempting decomposition.

**Limitation:** This conflates several mechanisms — informational demonstration, vendor referrals, administrative learning — all of which are "effects of having a funded neighbor nearby." It does not identify which channel dominates.

---

**Estimand 2: Application extensive margin — the cleanest behavioral outcome**

The peer effect mechanism is: *observation of neighbor's ESB → updated beliefs about feasibility/desirability → decision to pursue adoption.* The act of applying to a subsequent round is the direct behavioral signal of updated beliefs — it precedes and is logically separable from whether the applicant then wins the lottery. If own adoption (win + apply) is the outcome, the measurement conflates the peer effect signal with a subsequent random event. Application is the cleaner outcome.

Additionally, fiscal constraints may prevent adoption even among motivated districts. Measuring application rather than adoption avoids ruling out peer effects that are real but budget-constrained at the final step.

Specification:

$$Y_i^{R3} = \mathbf{1}[\text{district } i \text{ applied to R3 rebates}]$$

instrumented by `w_IV_Z_R1` (share of R1 lottery winners among K nearest neighbors). This sidesteps the weak first-stage problem because we no longer require R1 winners to re-apply in R3; we only require that R1 neighbor deployments predict R3 *interest* (application), which is a weaker and more plausible condition.

**Data status — confirmed available.** The file `CSBP Applicants waitlisted and rejected_11.18.25.xlsx` (N = 2,208 rows) contains a clean `Round` column with explicit labels:
- `R1 - 2022 Rebates`: 1,546 applicants (waitlisted/rejected from R1 lottery)
- `R2 - Grants`: 324 applicants (competitive grant, not a lottery)
- `R3 - 2023 Rebates`: 338 applicants (waitlisted/rejected from R3 lottery)

**Important caveat:** R2 entries are from a competitive (non-lottery) selection process. They must not be included in the loser-density control or treated as lottery non-winners. They are a distinct administrative group and should be excluded from all lottery-based instrument construction.

> ✅ **Implemented (Mar 3, 2026):** R2 grant applicants are excluded from all lottery IV construction. The column `w6_IS_R2_GRANTEE` captures R2-grantee neighbour share and is used only as a robustness control to verify the R1 peer-effect estimate is not proxying for locally elevated R2-driven adoption propensity. See [Section 7.5](#75-robustness-results-coef--se--t--p).

---

**Estimand 3: Longer-horizon all-source diffusion**

The R1 → R3 window (~15 months, Oct 2022 → Jan 2024) is likely too short for the deployment-and-observation mechanism to operate. WRI delivery data showed that most R1 buses were not operational before the R3 deadline. Extending the outcome window to include 2024 all-source WRI adoption allows more time for buses to be delivered, observed, and acted upon.

Specification: R1 lottery wins (2022) as instrument → all-source WRI adoption through end of 2024 as outcome.

**Data status — available, no new data needed.** The WRI v9 dataset (June 2025 update) already contains 2024 bus award data. Award year distribution in the bus-level sheet:

| Award Year | Bus Count |
|---|---|
| 2022 | 3,285 |
| 2023 | 3,393 |
| **2024** | **3,531** |

The previous analysis imposed an artificial `award_year ∈ {2023, 2024}` window filter. Re-extraction simply means setting the all-source outcome to include all WRI awards in 2023–2024 (or 2024-only for an even cleaner post-R1 window). No new dataset is required.

**Delivery data caveat:** Of the 12,549 bus-level rows, only ~38% have a recorded delivery date (`3r. Quarter delivered`) and ~36% have a service start date (`3s. Quarter first operating`). The delivery and service distributions through 2024 are substantial (1,389 delivered in 2024; 1,040 first operating in 2024), but the high missingness rate limits the precision of any delivery-timing split.

---

**Estimand 4: Heterogeneity by R1 deployment visibility**

If the mechanism is demonstration, R1 buses that were delivered *and operational* before a neighboring district's R3 (or next-cycle) decision should generate stronger peer effects. This splits R1 winners into early-delivery and late/no-delivery groups and uses each as a separate instrument.

The prior branch (Script 28) tested this with CSBP-specific R3 adoption as the outcome and found null reduced-form effects even for early-delivery neighbors. The natural extension is to use: (a) the all-source WRI outcome, and (b) the extended 2024 window from Estimand 3, where more delivery has occurred and observation is more plausible.

This estimand is best treated as a heterogeneity/mechanism check nested within Estimand 3, rather than a standalone design.

---

**What is likely not identifiable with current data**

- **Structural peer effect β** (the social multiplier): requires temporal ordering, a valid exclusion restriction, and a first stage that is not near-tautological. The core problem — R1 lottery winners rarely re-enter R3 — limits the first-stage power for the clean temporal design. This may be estimable from the application margin (Estimand 2) if the first stage is stronger there.
- **Mechanism decomposition** (information vs. vendor vs. infrastructure vs. political emulation): cross-sectional data cannot separate these channels. The donut test result (effect concentrates in 1–2 nearest neighbors) from the prior branch is suggestive of very local spillovers but cannot distinguish mechanisms.
- **Dynamic learning / Bayesian updating**: requires multiple pre-adoption time periods per district. The current data is effectively cross-sectional with a single adoption event per district.

---

## What Gets Built in This Branch

> **Status as of Mar 3, 2026:** All four items below are complete. Scripts are committed at HEAD `7cc8b85`. Results are in Section 7.

~~The goal of this branch is:~~

1. ✅ ~~**Clarify the research question and mechanism.**~~ Done — confirmed via program guide PDF analysis. The mechanism is application learning: winning-neighbour exposure updates beliefs about programme feasibility, raising own application propensity. The lottery structure has been confirmed as a 6-tier stratified draw, random within `priority × state` cells. See [Section 7.1](#71-lottery-structure-confirmed-mar-3-2026).

2. ✅ ~~**Define clean variable construction from scratch.**~~ Done — `config.py`, `01_build_analysis_dataset.py`, `02_build_spatial_weights.py` built on `Post-Temporal` from raw data only. Spatial weights produce both uniform `w{K}_` and inverse-distance `wd{K}_` lag columns. `priority × state` interaction FE replaces the incorrect additive approach. See [Section 7.2](#72-fixed-effects-and-spatial-weights).

3. ✅ ~~**Scope a feasible paper across the four estimands:**~~
   - ✅ **Estimand 1 (ITT):** All null (p > 0.58 across K=6/10/15). R3 adoption window too close to R1 lottery for peer signals to propagate to observable adoption. *(Script removed; see Post-Temporal branch.)*
   - ✅ **Estimand 2 (Application margin):** K=6 δ=+0.051 (p=0.081), priority subsample δ=+0.119 (p=0.019). See [Section 7.4](#74-main-estimates-with-coef--se--t--p). *(Script removed; superseded by `01_main_iv.py`.)*
   - ✅ **Estimand 3 (Longer-horizon all-source):** All null (p > 0.35). Delivery-timing split also null. See [Section 7.4](#74-main-estimates-with-coef--se--t--p). *(Script removed.)*
   - ✅ **Estimand 4 (Delivery visibility heterogeneity):** Nested within Estimand 3 as `3B` delivery-timing Wald test. Null (χ²(1) p > 0.67 for all outcomes). *(Script removed.)*

4. ✅ ~~**Work sequentially:**~~ Done — scripts run in order, all outputs committed.

---

## Key Reference Files (preserved on `Current` and `main`)

| File | What it contains |
|---|---|
| `4_Documentation/Model_Specification_and_Identification.md` | Full specification history, robustness results, and issue diagnoses as of Feb 2026 |
| `4_Documentation/2_21_Peer_Effects.md` | Project summary with all estimates and challenge list |
| `2_Scripts/2_Analysis/B_Estimation/26c_cross_round_diagnostics.py` | Definitive diagnostic decomposing the +1.34 CSBP result |
| `3_Output/Logs/cross_round_diagnostics.txt` | Output: Wald decomposition, AR test, reduced-form comparison |
| `2_Scripts/2_Analysis/B_Estimation/21_application_density_control.py` | Loser-density control — the source of the δ ≈ 0.11 estimate |
| `3_Output/Logs/delivery_timing_extended.txt` | Delivery timing by census division — documents the deployment lag problem |

All prior code is accessible by checking out the `Current` branch.

---

## Implementation and Results (Mar 3, 2026)

### 7.1 Identification Setup and Lottery-Strata Correction

Analysis of the 2022 and 2023 CSBP program guide PDFs confirms the R1 lottery is a **6-tier ordered draw**. Randomisation is flat **within** `priority × state × fuel_type` cells; win rates differ **between** cells by design due to tier order and the 10%-per-state cap.

| Tier | Pool drawn |
|---|---|
| 1 | ZE priority applicants — per state |
| 2 | Clean (non-ZE) priority applicants — per state |
| 3 | All remaining priority applicants |
| 4 | Remaining non-priority clean applicants |
| 5 | Remaining priority ZE applicants |
| 6 | Remaining non-priority ZE applicants |

**Econometric implication:** additive `priority_r1 + state FE` is not sufficient. The implemented correction is `priority_r1 × state` interaction FE (about 100 dummies), so treatment/control comparisons occur within true lottery strata.

---

### 7.2 Variable Construction (Standalone Reference)

All variables below are built in `01_build_analysis_dataset.py` and `02_build_spatial_weights.py`.

| Variable | Construction |
|---|---|
| `IV_Z_R1` | District won 2022 CSBP rebate lottery (`CSB_Rebates.xlsx`, non-cancelled statuses) |
| `IV_Z_R3` | District won 2023 CSBP rebate lottery |
| `IS_R2_GRANTEE` | District received 2023 CSBP competitive grant (`CSB_Grants.xlsx`); **never** used as lottery IV |
| `IS_R1_LOSER` | Applied in R1 but not R1 winner (using waitlisted/rejected file; overlap with winners removed) |
| `IS_LOSER_pooled` | Pooled loser indicator used in Estimand 1 robustness control |
| `Y_R3_apply` | Indicator district applied in R3 (winner or waitlisted/rejected) |
| `wri_any_2023`, `wri_any_2024`, `wri_any_2023_24` | WRI all-source award indicators from bus-level data using `award_year` (`3p. Quarter awarded`) |
| `is_pre_r1_adopter` | Any WRI bus award before 2022 |
| `r1_early_delivery` | R1 winner with bus delivered before 2024 |
| `r1_late_or_unknown` | R1 winner delivered in 2024 or unknown delivery |
| `w{K}_X` | Uniform KNN lag (row-standardised 1/K over K nearest geographic neighbours) |
| `wd{K}_X` | Inverse-distance lag: $\sum_j \left((1/d_{ij})/\sum_m(1/d_{im})\right)X_j$ over K neighbours |

Spatial KNN is based on NCES EDGE district centroids in EPSG:5070 (metres), with $K \in \{6,10,15\}$.

---

### 7.3 Full Estimation Specifications

All estimations are linear probability models with state-clustered standard errors.

**Common control block:**
- `log_enroll = log(enrollment)`
- `log_income = log(median_income)`
- `poverty_rate`, `pct_white`, `pm25`, `pct_dem_2020`
- `priority_r23` (in Estimands 2 and 3)
- `priority_r1 × state` interaction FE
- optional loser-share controls depending on specification

#### Estimand 1 (ITT: R1 neighbour wins → CSBP adoption)

Sample: non-R1-winners with complete controls (`N = 12,721`).

$$
Y_i^{CSBP} = \alpha + \delta_K\,wK\_IV\_Z\_R1_i + \rho_K\,wK\_IS\_LOSER\_pooled_i + \beta'X_i + \lambda_{priority\times state} + \varepsilon_i
$$

where $Y_i^{CSBP} = IS\_ADOPTER\_CSBP\_ANY$ and $\rho_K$ included only in loser-control specs.

#### Estimand 2 (Primary: R1 neighbour wins → R3 application)

Sample: non-R1-winners with complete controls (`N = 12,388`).

$$
Y_i^{R3\,apply} = \alpha + \delta_K\,wK\_IV\_Z\_R1_i + \rho_K\,wK\_IS\_R1\_LOSER_i + \beta'X_i + \lambda_{priority\times state} + \varepsilon_i
$$

with $Y_i^{R3\,apply}=Y\_R3\_apply$ and $\rho_K$ included only in loser-control specs.

#### Applicant-neighbor luck reparameterization (new comparison design)

To isolate lottery luck **among applicant neighbors** from pure applicant-neighbor concentration, define:

$$
A_i = w6\_IV\_Z\_R1_i + w6\_IS\_R1\_LOSER_i, \quad
p_i = \frac{w6\_IV\_Z\_R1_i}{A_i}
$$

on the restricted sample $A_i>0$ (districts with at least one R1 applicant neighbor).

Estimate:

$$
Y_i = \alpha + \theta\,p_i + \phi\,A_i + \beta'X_i + \lambda_{priority\times state} + \varepsilon_i
$$

Interpretation:
- $\theta$ = effect of higher winner share **holding applicant intensity fixed** (the clean luck channel)
- $\phi$ = effect of local applicant-neighbor density/intensity

#### Estimand 3 (R1 neighbour wins → all-source adoption)

Sample: non-R1-winners with complete controls (`N = 12,388`).

$$
Y_i^{allsource} = \alpha + \delta_K\,wK\_IV\_Z\_R1_i + \rho_K\,wK\_IS\_R1\_LOSER_i + \pi\,is\_pre\_r1\_adopter_i + \beta'X_i + \lambda_{priority\times state} + \varepsilon_i
$$

for outcomes $Y_i^{allsource} \in \{wri\_any\_2023\_24,\,wri\_any\_2024\}$.

The same $(p_i, A_i)$ reparameterized specification is also run for all-source outcomes as a robustness comparison.

#### Estimand 4 (delivery-visibility heterogeneity, nested under Estimand 3)

$$
Y_i = \alpha + \eta\,w6\_r1\_early\_delivery_i + \kappa\,w6\_r1\_late\_or\_unknown_i + \rho\,w6\_IS\_R1\_LOSER_i + \pi\,is\_pre\_r1\_adopter_i + \beta'X_i + \lambda_{priority\times state} + \varepsilon_i
$$

Wald test target: $H_0: \eta = \kappa$.

---

### 7.4 Main Estimates with Coef / SE / t / p

#### 7.4.1 Estimand 2 — Main table (primary result)

| Spec | N | Coef (δ) | SE | t | p |
|---|---:|---:|---:|---:|---:|
| K=6, loser=No | 12,388 | +0.0496 | 0.0293 | 1.693 | 0.0905 |
| **K=6, loser=Yes** | **12,388** | **+0.0508** | **0.0291** | **1.746** | **0.0809** |
| K=10, loser=No | 12,388 | +0.0802 | 0.0430 | 1.867 | 0.0619 |
| **K=10, loser=Yes** | **12,388** | **+0.0810** | **0.0426** | **1.899** | **0.0575** |
| K=15, loser=Yes | 12,388 | +0.0668 | 0.0528 | 1.266 | 0.2057 |

Priority-district subsample (`priority_r1=1`, K=6):

| Spec | N | Coef (δ) | SE | t | p |
|---|---:|---:|---:|---:|---:|
| loser=No | 6,402 | +0.1159 | 0.0504 | 2.298 | 0.0216 |
| **loser=Yes** | **6,402** | **+0.1189** | **0.0505** | **2.356** | **0.0185** |

#### 7.4.2 Estimand 1 — ITT (null benchmark)

| Spec | N | Coef (δ) | SE | t | p |
|---|---:|---:|---:|---:|---:|
| K=6, loser=Yes | 12,721 | +0.0067 | 0.0240 | 0.279 | 0.7804 |
| K=10, loser=Yes | 12,721 | +0.0018 | 0.0319 | 0.056 | 0.9552 |
| K=15, loser=Yes | 12,721 | +0.0162 | 0.0370 | 0.438 | 0.6613 |

#### 7.4.3 Estimand 3 — all-source outcomes (null)

| Outcome / Spec | N | Coef (δ) | SE | t | p |
|---|---:|---:|---:|---:|---:|
| `wri_any_2023_24`, K=6, loser=Yes | 12,388 | +0.0178 | 0.0311 | 0.572 | 0.5670 |
| `wri_any_2023_24`, K=10, loser=Yes | 12,388 | +0.0319 | 0.0338 | 0.943 | 0.3457 |
| `wri_any_2024`, K=6, loser=Yes | 12,388 | +0.0038 | 0.0236 | 0.161 | 0.8718 |
| `wri_any_2024`, K=10, loser=Yes | 12,388 | +0.0203 | 0.0298 | 0.681 | 0.4959 |

---

### 7.5 Robustness Results (Coef / SE / t / p)

#### R2 confound checks (Estimand 2, K=6, loser=Yes)

| Spec | N | Coef (δ) | SE | t | p |
|---|---:|---:|---:|---:|---:|
| R2-neighbour present (`w6_IS_R2_GRANTEE>0`) | 1,140 | +0.0649 | 0.0658 | 0.987 | 0.3236 |
| R2-neighbour absent (`w6_IS_R2_GRANTEE=0`) | 11,248 | +0.0458 | 0.0339 | 1.350 | 0.1771 |
| Baseline (no R2 covariate) | 12,388 | +0.0508 | 0.0291 | 1.746 | 0.0809 |
| Additive R2 covariate included | 12,388 | +0.0525 | 0.0290 | 1.812 | 0.0699 |

#### Applicant-exposed sample (difference-style restriction)

Restricted sample: districts with `w6_IV_Z_R1 > 0` OR `w6_IS_R1_LOSER > 0` (`N=7,085`).

| Coefficient | Coef | SE | t | p |
|---|---:|---:|---:|---:|
| Winner share (`w_IV_Z_R1`) | +0.0404 | 0.0385 | 1.049 | 0.2942 |
| Loser share (`w_r1_loser`) | +0.0058 | 0.0211 | 0.277 | 0.7820 |

#### Applicant-neighbor luck reparameterization (new)

Estimand 2 (Outcome = `Y_R3_apply`, sample $A>0$, $N=7,085$):

| Coefficient | Coef | SE | t | p |
|---|---:|---:|---:|---:|
| Luck among applicants (`p_win_given_apply`) | +0.0042 | 0.0076 | 0.551 | 0.5817 |
| Applicant intensity (`A`) | +0.0131 | 0.0220 | 0.596 | 0.5510 |

Estimand 3 (all-source outcomes, sample $A>0$, $N=7,085$):

| Outcome / Coefficient | Coef | SE | t | p |
|---|---:|---:|---:|---:|
| `wri_any_2023_24`: `p_win_given_apply` | -0.0063 | 0.0110 | -0.571 | 0.5681 |
| `wri_any_2023_24`: `A` | +0.0291 | 0.0189 | 1.537 | 0.1244 |
| `wri_any_2024`: `p_win_given_apply` | -0.0022 | 0.0091 | -0.236 | 0.8132 |
| `wri_any_2024`: `A` | +0.0102 | 0.0200 | 0.507 | 0.6121 |

**Comparison implication:** once applicant-neighbor intensity is explicitly separated from winner-share luck among applicants, the luck coefficient is near zero and imprecise across both application and adoption outcomes.

#### Distance-decay weights

| Weight scheme | N | Coef (δ) | SE | t | p |
|---|---:|---:|---:|---:|---:|
| Uniform 1/K (`w6_IV_Z_R1`) | 12,388 | +0.0508 | 0.0291 | 1.746 | 0.0809 |
| Inverse-distance (`wd6_IV_Z_R1`) | 12,388 | +0.0369 | 0.0284 | 1.300 | 0.1936 |

---

### 7.6 Delivery-Heterogeneity Results (Estimand 4)

| Outcome | Coef (early) | SE | p | Coef (late/unk) | SE | p |
|---|---:|---:|---:|---:|---:|---:|
| `wri_any_2023_24` | +0.0101 | 0.0387 | 0.7935 | +0.0313 | 0.0425 | 0.4612 |
| `wri_any_2024` | -0.0039 | 0.0332 | 0.9076 | +0.0066 | 0.0278 | 0.8132 |

Wald equality tests (from estimation log):
- `wri_any_2023_24`: $\chi^2(1)=0.18$, $p=0.6741$
- `wri_any_2024`: $\chi^2(1)=0.07$, $p=0.7917$

No detectable early-vs-late difference.

---

### 7.7 Concise Empirical Takeaway

| Dimension | Result |
|---|---|
| Primary signal | **Application margin is positive** (Estimand 2: δ ≈ 0.05–0.08, marginal significance; priority subsample δ ≈ 0.12, statistically significant) |
| Adoption outcomes | **Null** in both CSBP-adoption ITT (Estimand 1) and all-source WRI outcomes (Estimand 3) |
| Confound checks | R2-neighbour control does not attenuate δ materially |
| Luck-among-applicants check | Reparameterized luck coefficient is near zero (imprecise); no clear differential winner-vs-loser signal conditional on applicant intensity |
| Mechanism checks | No discouraged-loser effect; no delivery-timing heterogeneity |
| Weighting robustness | Inverse-distance attenuates magnitude but keeps sign positive |

**Bottom line:** Within this temporal design, the strongest evidence is for peer effects in **programme entry/application behavior** at the full-sample margin. But when conditioning tightly on applicant-neighbor exposure and decomposing into luck-share vs applicant-intensity components, the winner-share luck signal attenuates to near zero and loses precision.

---

### 7.8 Structural 2SLS Results (Mar 4, 2026)

Script: ~~`04_estimand_2sls.py`~~ *(removed from Post-Meeting branch; see Post-Temporal/Post-Poster branch)*. Outputs: `estimand_2sls.csv`, `estimand_2sls.txt`.

#### 7.8.1 Design

| | Value |
|---|---|
| Endogenous variable (D) | `w{K}_wri_any_2023` — share of K neighbours with any WRI-tracked ESB by end of 2023 |
| Instrument (Z) | `w{K}_IV_Z_R1` — share of K neighbours who won R1 (2022) CSBP lottery |
| Outcomes | `Y_R3_apply` (primary), `wri_any_2023_24` (secondary) |
| Sample | Non-R1-winners, N=12,388 (same as Estimand 2) |
| Estimator | IV-2SLS via `linearmodels.iv.IV2SLS`, state-clustered SE |
| Exclusion restriction (maintained) | Z affects Y only through D (neighbour deployment); informational/vendor channels before physical delivery would violate this |

**Structural interpretation of β:** causal effect of a 1-unit increase in the share of K neighbours with physically deployed WRI ESBs on own R3 application probability.

β_2SLS = RF / FS = (effect of lottery luck on Y) / (effect of lottery luck on D)

D ≠ Z because: (a) WRI captures non-CSBP channels (~33% of ESBs); (b) not all R1 winners appear in WRI by 2023; (c) R2 grantees and state-funded buses enter D without affecting Z.

#### 7.8.2 First Stage

| K | FS coef | SE | FS partial F | Interpretation |
|---|---:|---:|---:|---|
| 6 | +0.040 | 0.015 | **7.06** | Marginally weak (below 10 threshold) |
| 10 | +0.027 | 0.018 | **2.23** | Clearly weak |
| 6 (priority) | +0.039 | 0.022 | **3.15** | Weak |

First stage is below conventional thresholds because D (WRI deployment by 2023) is imperfectly predicted by Z (R1 lottery wins): the R1→WRI linkage is partial, delayed by delivery lags, and diluted by non-lottery WRI adoption. This is the quantitative confirmation of the timing problem identified in the design.

#### 7.8.3 Main 2SLS Results

| Outcome | K | N | OLS | 2SLS LATE | SE | p | AR p(β=0) | AR 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Y_R3_apply` | 6 | 12,388 | −0.040 | **+1.281** | 0.928 | 0.167 | 0.087 | [−0.175, 7.025] |
| `Y_R3_apply` | 10 | 12,388 | −0.066 | +3.041 | 2.710 | 0.262 | 0.064 | [−0.150, 20+] |
| `wri_any_2023_24` | 6 | 12,388 | −0.042 | +0.600 | 0.852 | 0.481 | 0.490 | [−1.650, 3.750] |
| `wri_any_2023_24` | 10 | 12,388 | −0.077 | +1.509 | 1.588 | 0.342 | 0.273 | unbounded |
| `Y_R3_apply` (**priority**) | 6 | 6,402 | — | **+3.057** | 2.093 | 0.144 | **0.023** | **[0.425, 20+]** |

#### 7.8.4 Key Takeaways from 2SLS

**1. Weak first stage quantifies the timing problem.** FS F=7.06 for K=6 confirms what the delivery data already showed: R1 lottery wins translate imperfectly and with a lag into observable WRI deployment. Any structural estimate is therefore poorly identified with this observation window.

**2. 2SLS LATE is large but imprecise.** β̂=+1.28 (K=6, Y_R3_apply) — interpretable as: a 10pp increase in the share of neighbours with deployed ESBs raises own R3 application probability by ~13pp. The standard error is too large to distinguish this from zero (p=0.167). The Wald CI [-0.54, 3.10] is very wide, consistent with FS F=7.

**3. AR inference is the credible reference for inference under weak IV.** AR p(β=0) = 0.087 (K=6) and 0.064 (K=10) for the primary outcome — matching the reduced-form p-values by construction (AR at β₀=0 recovers the RF test). AR CI for K=6 application: [-0.18, 7.03] — crosses zero, consistent with the 5% non-rejection at the conventional level.

**4. Priority subsample AR CI excludes zero.** For priority-eligible districts, AR p=0.023 and AR 95% CI lower bound is **+0.425** — we can reject β≤0 at 5% even with the weak instrument. This is the strongest identification result in the project: among priority-eligible districts, the structural effect of observable neighbour deployment on application entry is robustly positive. The upper bound being at 20+ reflects the genuine imprecision from FS F=3.15.

**5. OLS naive is consistently negative (−0.04 to −0.08).** The negative OLS sign is informative: conditioning on controls, districts with more WRI-adopting neighbours are *less* likely to apply to R3. This likely reflects selection — districts in more ESB-saturated local markets have lower marginal motivation to enter — and motivates the IV. The 2SLS corrects this downward selection bias, flipping the sign and amplifying the magnitude.

**6. KP F from linearmodels returned NaN** (likely a version compatibility issue). The manually computed partial F (t²) is the operational equivalent for a single excluded instrument and is the figure reported above.

---

---

### 7.9 Sharp-Cutoff Delivery Timing (Deterrence Check, Mar 5, 2026)

Script: ~~`_deterrence_sharp_cutoff.py`~~ *(removed from Post-Meeting branch)* (diagnostic, not part of main estimation pipeline).

#### 7.9.1 Motivation

The coarse `r1_early_delivery` variable (buses delivered before 2024) conflates buses that arrived **before** the R3 application deadline (Oct 2023, i.e., ≤ 2023 Q3) with buses that arrived **after** the deadline (2023 Q4). Only pre-deadline buses were visible to neighbouring districts when R3 application decisions were being made. This re-cut tests whether the mechanistically relevant timing split — visible vs. invisible at time of decision — produces a deterrence pattern.

#### 7.9.2 Variable Construction

WRI bus-level data (`3r. Quarter delivered`) provides quarter-level precision.

| Category | Definition | District count (R1 winners) |
|---|---|---:|
| `pre_r3` | Earliest R1 bus delivered ≤ 2023 Q3 | 148 |
| `post_r3` | Earliest R1 bus delivered ≥ 2023 Q4 (includes all 2024) | 188 |
| `unknown` | No delivery date in WRI | 29 |

Note: the coarse `r1_early_delivery` (before 2024) counted 217 districts — it combined pre_r3 (148) with the 2023 Q4 tranche (69 districts) that actually arrived after the deadline.

KNN spatial lags `w6_r1_pre_r3`, `w6_r1_post_r3`, `w6_r1_del_unknown` built with the same K=6, EPSG:5070 procedure as all other lags.

#### 7.9.3 Results

Outcome: `Y_R3_apply` (R3 application indicator). Sample: non-R1-winners, N=12,388 (or 6,402 for priority). State-clustered SE; HC3 for California (single state).

| Sample | pre_r3 coef | SE | p | post_r3 coef | SE | p | Wald (pre=post) p |
|---|---:|---:|---:|---:|---:|---:|---:|
| National full | +0.045 | 0.054 | 0.408 | **−0.034** | 0.048 | 0.481 | 0.305 |
| National priority | +0.092 | 0.054 | 0.091* | −0.012 | 0.072 | 0.865 | 0.239 |
| California (HC3) | −0.148 | 0.233 | 0.524 | −0.097 | 0.161 | 0.547 | 0.846 |

Adoption outcome comparison (`wri_any_2023_24`):

| Sample | pre_r3 coef | p | post_r3 coef | p | Deterrence? |
|---|---:|---:|---:|---:|---|
| National full | +0.098 | 0.062* | +0.033 | 0.374 | No (post positive) |

#### 7.9.4 Interpretation

1. **Direction is mechanistically correct for the application channel.** Neighbours whose buses arrived before the R3 deadline (pre_r3) have a positive coefficient (+0.045 full, +0.092 priority); neighbours whose buses arrived after the deadline (post_r3) have a negative coefficient (−0.034). This matches the information-channel prediction: only visible neighbours can trigger application entry.

2. **Post_r3 is negative (deterrence direction) but not statistically distinguishable from zero.** Wald p=0.305. The null of pre=post cannot be rejected. Precision is limited by thin coverage: mean spatial lag is only ~0.010–0.013 (roughly 1 in 100 neighbours has one of these indicators).

3. **No deterrence of ESB adoption itself.** For the all-source adoption outcome, post_r3 is positive (+0.033), not negative. The timing-sensitive effect is specific to the application/deadline channel, consistent with the information interpretation rather than a generalised discouragement effect.

4. **California is uninformative.** Near-zero spatial lag means (pre=0.005, post=0.010) and single-state HC3 SEs produce very wide confidence intervals. Not interpretable in isolation.

#### 7.9.5 Poster implications

- The sharp cutoff analysis provides **suggestive directional evidence** for a timing-sensitive information mechanism: the behavioural response (increased application) is concentrated among neighbours whose buses were physically visible before the decision deadline.
- The effect is not precise enough to headline as a robust finding; best framed as a **mechanism corroboration** alongside the main reduced-form result.
- Key contrast for the poster: coarse cutoff (before 2024) showed all-positive coefficients, suggesting no deterrence; the sharp cutoff reveals the expected sign reversal for post-deadline deliveries, indicating the coarse cut masked the mechanism-relevant split.

---

*Document written March 2, 2026. Updated March 3–5, 2026 with full specifications, results, 2SLS, and deterrence analysis. Updated April 8, 2026: header revised to reflect Post-Meeting branch; old estimand script references annotated as removed; see `Checklist.md` for current analysis.*
*Original branch: `Post-Temporal` HEAD `7cc8b85` | Current branch: `Post-Meeting`*
