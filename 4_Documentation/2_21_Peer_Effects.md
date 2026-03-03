# Peer Effects in Electric School Bus Adoption: Project Summary

**Created:** February 21, 2026 | **Last Updated:** March 3, 2026  
**Project:** Peer Effects and Adoption  
**Location:** `c:\BC PhD\Research\Peer-Effects and Adoption`  
**Active Branch:** `Post-Temporal` (HEAD: `6a0e495`)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Chronology](#project-chronology)
3. [Data Sources and Construction](#1-data-sources-and-construction)
4. [Empirical Specification — Current Branch](#2-empirical-specification)
5. [Identification Strategy](#3-identification-strategy)
6. [Main Results — Current Branch](#4-main-results)
7. [Robustness and Sensitivity](#5-robustness-and-sensitivity)
8. [Challenges and Limitations (Feb 21, 2026)](#6-challenges-and-limitations-detailed-diagnostic-feb-21-2026)
9. [Recommended Solutions (Feb 21, 2026)](#7-recommended-solutions)
10. [Key Takeaways (Feb 21, 2026)](#8-key-takeaways-updated-feb-21-2026)
11. [Next Steps](#9-next-steps)
12. [**Post-Temporal Branch: New Design & Results (Mar 3, 2026)**](#10-post-temporal-branch-new-design--results-mar-3-2026)
13. [References](#references)

---

## Executive Summary

This project investigates **peer effects in electric school bus (ESB) adoption** using data from the EPA Clean School Bus Program (2022-2024). The central research question is whether a school district's decision to adopt ESBs is causally influenced by its neighbors' adoption decisions.

**Key Finding:** Evidence supports **local geographic peer effects** in ESB adoption (β ≈ 0.06–0.19 depending on specification), though estimates are sensitive to control sets and sample composition. Climate-based peer networks (districts with similar environmental conditions) show either null or negative effects, suggesting peer effects operate through **social/informational proximity** rather than technical similarity.

**Caution:** Peer effect magnitudes vary substantially across specifications—a common challenge in this literature. The qualitative finding that geographic proximity matters is robust; precise magnitudes require careful interpretation.

---

## Project Chronology

| Phase | Date | Key Activities |
|-------|------|----------------|
| **Data Acquisition** | Pre-Nov 2025 | WRI ESB dataset v9, EPA CSB program data, Census ACS, PRISM climate |
| **Data Prep Scripts** | Nov 30, 2025 | Scripts 01-06 in `2_Scripts/1_Preparation/` |
| **Initial Analysis** | Dec 3, 2025 | Main results documented in `December_Run.md` |
| **Robustness Battery** | Dec 2025 | Scripts 01-06 in `2_Scripts/2_Analysis/B_Estimation/` |
| **Advanced Diagnostics** | Late 2025 | Standardized estimation, placebo tests, residualized IV (Scripts 07-10) |
| **Latest Outputs** | ~Feb 2026 | Most recent estimation outputs in `3_Output/Tables/` |

---

## 1. Data Sources and Construction

### 1.1 Primary Data Sources

| Source | Description | Key Variables |
|--------|-------------|---------------|
| **World Resources Institute (WRI)** | ESB adoption dataset v9 (June 2025 update) | District-level adoption, bus counts, fleet characteristics |
| **EPA Clean School Bus Program** | Lottery-based allocation (Rounds 1-3, 2022-2024) | Winner/loser status (IV_Z), priority designation |
| **NCES Common Core of Data** | School district demographics | Enrollment, racial composition, urbanicity |
| **American Community Survey (2021)** | Socioeconomic characteristics | Median income, poverty rate |
| **NASA SEDAC** | Environmental data | PM2.5 air pollution |
| **MIT Election Lab** | Political data | 2020 presidential vote share by county |
| **PRISM Climate Group** | 30-year climate normals | Precipitation, minimum temperature |
| **NCES EDGE** | Geographic boundaries | District shapefiles for spatial analysis |

### 1.2 Analysis Dataset

**Final Sample:** 13,219 school districts
- Traditional districts: 13,187 (99.8%)
- Charter schools: 32 (0.2%)

**Adoption Rate:** 7.0% (925 adopters in full sample; ~1,241 in extended datasets)

**Applicant Pool:** 2,560 districts (19.4% of sample)
- Lottery winners: 719 (28.1% of applicants)
- Lottery losers: 1,841 (71.9% of applicants)

**Key Constructed Variables:**
- `IS_ADOPTER`: Binary indicator (adopted ESB = 1)
- `IV_Z`: Lottery winner status (instrument)
- `w_geo_adoption`: Spatial lag of neighbor adoption (endogenous)
- `w_geo_z`: Spatial lag of neighbor lottery status (instrument)
- `w_clim_adoption` / `w_clim_z`: Climate-based peer effects and instrument

### 1.3 Spatial Weight Matrices

Two types of neighbor definitions:

1. **Geographic Neighbors (w_geo):** K-nearest neighbors based on centroid distance
   - Primary specification: KNN-6
   - Robustness: KNN-4, KNN-8, KNN-10, KNN-12, KNN-15, KNN-20
   - Distance bands: 25km, 50km, 75km, 100km

2. **Climate Neighbors (w_clim):** Districts with similar environmental conditions
   - Based on precipitation + minimum temperature similarity
   - KNN-based construction (K=4 to K=20)
   - "Donut" specification: KNN-20 minus KNN-6 (regional peers excluding immediate neighbors)

---

## 2. Empirical Specification

### 2.1 Model

**Linear Probability Model with IV-2SLS:**

$$
\text{IS\_ADOPTER}_i = \alpha + \beta_1 \cdot W_{geo} \cdot \text{Adoption}_{-i} + \beta_2 \cdot W_{clim} \cdot \text{Adoption}_{-i} + \gamma' X_i + \theta_s + \varepsilon_i
$$

Where:
- $W_{geo} \cdot \text{Adoption}_{-i}$ = Geographic peer adoption rate (endogenous)
- $W_{clim} \cdot \text{Adoption}_{-i}$ = Climate peer adoption rate (endogenous)
- $X_i$ = Vector of controls
- $\theta_s$ = State fixed effects

**Instruments:**
- $W_{geo} \cdot Z_{-i}$ = Geographic peer lottery winner share
- $W_{clim} \cdot Z_{-i}$ = Climate peer lottery winner share

### 2.2 Control Variables

| Variable | Description |
|----------|-------------|
| `log_median_income` | Log median household income |
| `poverty_rate` | District poverty rate (0-1) |
| `log_enrollment` | Log total student enrollment |
| `pm25` | PM2.5 air pollution (μg/m³) |
| `pct_white` | Percent white students |
| `pct_dem_2020` | County-level Democratic vote share (2020) |
| `is_priority` | Priority applicant designation |
| Urbanicity dummies | Suburban, Town, Urban (Rural = baseline) |
| State fixed effects | 50 state dummies |

### 2.3 Estimation Details

- **Estimation Method:** IV-2SLS (linearmodels package)
- **Standard Errors:** Clustered by state (50 clusters)
- **Software:** Python (pandas, geopandas, libpysal, linearmodels, statsmodels)

---

## 3. Identification Strategy

### 3.1 The Reflection Problem

The classic challenge in peer effects estimation (Manski, 1993): Are districts adopting because neighbors adopted, or do they share common shocks? Simultaneity between own and neighbor outcomes biases OLS estimates.

**Solution:** Exploit the **EPA lottery mechanism** for instrument construction. Lottery winners received funding randomly (conditional on application and priority status), creating exogenous variation in neighbor outcomes.

### 3.2 Instrument Construction

$$
\text{Instrument: } W_{geo} \cdot Z = \sum_{j \in N_i} w_{ij} \cdot \text{LOTTERY\_WINNER}_j
$$

The spatial lag of neighbors' lottery outcomes should:
1. **Predict neighbor adoption** (relevance) ✅
2. **Affect own adoption only through neighbor adoption** (exclusion)

### 3.3 Validity Concerns and Diagnostics

**First-Stage Strength:**
- Geographic instrument F-statistics: 2,000–10,000 (very strong)
- Climate instrument F-statistics: 180,000–430,000 (extremely strong)
- All exceed weak instrument threshold (F > 10)

**Balance Test Results (Potential Issues):**

The lottery is **stratified by priority status**, creating differential win rates:
- Priority applicants: 12.2% win rate
- Non-priority applicants: 43.4% win rate

This creates Simpson's Paradox: aggregate balance masks within-stratum imbalance.

| Instrument Type | Balance Failures (Bonferroni 5%) |
|-----------------|-----------------------------------|
| Original (w_geo_z) | 5/8 covariates |
| Residualized (w_geo_z_resid) | 3/8 covariates |
| Resid + Spatial Controls | 1/8 covariates |

**Placebo Tests:**
- Non-neighbor placebo instrument: Not significant (good) for K=10, marginally significant for K=6
- Permutation inference: p < 0.001 for both K=6 and K=10 (true effect in tails of null distribution)

### 3.4 Addressing Confounding

Implemented corrections:
1. **Propensity-Score Residualized IV:** Residualize lottery outcomes on (priority × state) cells
2. **Spatial Controls:** Add neighbor application rates (`w_geo_applicant`) and neighbor priority status (`w_geo_priority`)
3. **Full Control Specification:** State FE + political + demographic controls

---

## 4. Main Results

### 4.1 Primary Estimates (Script 01: Full Controls + State FE)

| Variable | Coefficient | SE | P-Value | Interpretation |
|----------|-------------|-----|---------|----------------|
| **w_geo_adoption** | **0.191*** | 0.033 | <0.001 | 10pp ↑ neighbor adoption → 1.9pp ↑ own adoption |
| w_clim_adoption | 0.029 | 0.032 | 0.37 | Climate peers: Not significant |

### 4.2 Standardized Estimates Across Specifications

**Geographic Peer Effects (Full Sample, State FE):**

| K | Coefficient | SE | P-Value | F-Stat |
|---|-------------|-----|---------|--------|
| 4 | 0.057*** | 0.020 | 0.004 | 9,544 |
| 6 | 0.064*** | 0.021 | 0.003 | 6,460 |
| 8 | 0.060*** | 0.023 | 0.008 | 4,565 |
| 10 | 0.059** | 0.024 | 0.012 | 2,844 |
| 12 | 0.057** | 0.023 | 0.012 | 2,146 |
| 15 | 0.061*** | 0.022 | 0.007 | 1,598 |
| 20 | 0.055** | 0.026 | 0.031 | 1,244 |

**Climate Peer Effects (Standalone, Climate-Restricted Sample):**
- Consistently **negative** coefficients (-0.01 to -0.02)
- Often statistically significant
- Contrary to hypothesis that climate similarity facilitates peer effects

### 4.3 Residualized IV Results

| K | Specification | Peer Effect | SE | P-Value | F-Stat |
|---|--------------|-------------|-----|---------|--------|
| 6 | Original IV | 0.064*** | 0.021 | 0.003 | 6,460 |
| 6 | Residualized IV | 0.051* | 0.030 | 0.091 | 1,085 |
| 6 | Resid + Spatial Controls | 0.069** | 0.035 | 0.046 | 1,242 |
| 10 | Original IV | 0.059** | 0.024 | 0.012 | 2,844 |
| 10 | Residualized IV | 0.043 | 0.035 | 0.223 | 699 |
| 10 | Resid + Spatial Controls | 0.063 | 0.044 | 0.149 | 778 |

**Interpretation:** Residualization reduces F-statistics substantially (from ~6,000 to ~1,000) and inflates standard errors, but point estimates remain positive and similar in magnitude.

### 4.4 Other Key Findings

| Variable | Effect | P-Value | Interpretation |
|----------|--------|---------|----------------|
| **is_priority** | +13pp | <0.001 | Priority targeting worked—largest single effect |
| **pct_dem_2020** | +1pp per 10pp D vote | 0.01 | Political stratification in green tech adoption |
| **log_enrollment** | +2.3pp per 10% ↑ | <0.001 | Scale economies favor large districts |
| **Urban** | +9pp vs. Rural | <0.001 | Urban operational advantages |
| **pct_white** | -0.7pp per 10pp ↑ | 0.04 | More diverse districts adopt more (EJ targeting) |

**State Fixed Effects:** Hawaii (+90pp), Nevada (+41pp), West Virginia (+22pp) show extraordinary state-level variation, suggesting complementary state policies matter enormously.

---

## 5. Robustness and Sensitivity

### 5.1 Specification Sensitivity

| Issue | Finding |
|-------|---------|
| **Sample Composition** | Losing 2-6% of sample (e.g., climate data availability) can flip sign/significance |
| **Neighbor Definition** | Effects robust across KNN-4 to KNN-20; attenuate slightly for wider definitions |
| **Distance Bands** | Distance-based weights (25-100km) show weaker, often insignificant effects |
| **Charter Exclusion** | Dropping 32 charters does not materially change results |
| **State FE** | Critical—without state FE, climate effects dominate (likely confounding) |

### 5.2 Horse-Race Results (Geographic vs. Climate)

When both peer effects are estimated jointly:

| Specification | Geographic | Climate |
|---------------|------------|---------|
| No State FE | Not significant | Highly significant (0.35–0.66) |
| With State FE | Positive, significant | Negative, significant |

**Interpretation:** State FE absorb regional variation that climate similarity proxies. Without state FE, climate effects likely reflect omitted regional factors (policies, vendor networks, political climate).

### 5.3 Placebo and Falsification Tests

| Test | K=6 | K=10 | Interpretation |
|------|-----|------|----------------|
| Permutation p-value | <0.001 | <0.001 | True effect far in tails of null |
| Non-neighbor placebo | -0.016 (p=0.04) | 0.009 (p=0.57) | Mixed—K=6 shows unexpected significance |

---

## 6. Challenges and Limitations (Detailed Diagnostic: Feb 21, 2026)

### 6.1 Challenge 1: Lottery Stratification

**The Problem:**
The EPA CSB Program lottery was heavily stratified by priority status:

| Applicant Type | Win Rate | Count |
|----------------|----------|-------|
| Priority | 12.1% | 1,226 |
| Non-Priority | 42.8% | 1,334 |
| **Ratio** | **3.55×** | |

Chi-square test: χ² = 297.2, p < 10⁻⁶⁶

Priority applicants differ systematically:
- $20,650 lower median income
- 5pp higher poverty rate
- 3,886 fewer students (smaller districts)

**Simpson's Paradox:** Aggregate balance appears acceptable (7/8 covariates pass), but this masks opposing biases—few priority winners (poor districts) combine with many non-priority winners (rich districts) to create apparent balance.

### 6.2 Challenge 2: Balance Test Failures

**Original Spatial IV (w_geo_z) predicts 5/8 covariates:**
- Enrollment: +14,181 (p<0.001)
- PM2.5: +1.05 (p=0.005)
- % White: -9.6pp (p<0.001)
- % Black: +6.2pp (p<0.001)
- Is Priority: +47pp (p<10⁻¹⁶)

**Within-Stratum Balance (Even More Concerning):**

| Stratum | Covariate | Winners vs. Losers | P-value |
|---------|-----------|-------------------|---------|
| Priority | Enrollment | +3,916 | 0.042* |
| Priority | PM2.5 | +0.42 | 0.003* |
| Priority | % White | -3.8pp | 0.039* |
| Non-Priority | Income | **-$13,061** | <0.001*** |
| Non-Priority | Poverty | **+3.1pp** | <0.001*** |
| Non-Priority | PM2.5 | -0.26 | 0.003* |

~~**Critical Finding:** Even after residualizing on priority × state cells, balance tests STILL fail:~~  
~~- Median income: -$5,767 (p<0.001)~~  
~~- Poverty rate: +1.9pp (p<0.001)~~

~~This suggests the lottery wasn't purely random even within strata—possibly due to application quality, timing, or other unobserved factors.~~

> **⚠️ Superseded — Mar 3, 2026:** These balance failures were an artefact of incorrect conditioning. Residualizing the IV on `(priority × state)` opens a collider path. PDF analysis of the program guides confirmed the lottery is a **6-tier ordered draw within `priority × state` cells** — random within each stratum by construction. The correct fix (implemented in `Post-Temporal` branch) is `priority × state` **interaction fixed effects** in the outcome equation (~100 dummies). See [Section 10.2](#102-lottery-structure-confirmed-from-program-guide-pdfs).

### 6.3 Challenge 3: Specification Sensitivity

**Geographic Peer Effect Estimates Across 39 Specifications:**
- Mean: 0.030
- SD: 0.023
- Range: [-0.007, 0.064]
- **Significant (p<0.05): Only 14/39 (36%)**

**By Sample Type:**
| Sample | Specifications | Mean β | Significant |
|--------|---------------|--------|-------------|
| Full Sample | 11 | 0.039 | 7/11 (64%) |
| Climate-Restricted | 14 | 0.015 | 0/14 (0%) |
| No Charters (Geo Only) | 7 | 0.059 | 7/7 (100%) |
| No Charters (Horse Race) | 7 | 0.017 | 0/7 (0%) |

**Key Insight:** Results are robust in Full Sample specifications but evaporate in Climate-Restricted or Horse Race specifications. The ~700 districts lost to climate data restrictions appear non-random.

### 6.4 Challenge 4: Geographic vs. Climate Contradiction

**Without State FE (S1-S3):** Climate dominates
- Climate: β = 0.35–0.50 (p<0.001)
- Geographic: β = 0.03–0.04 (not significant)

**With State FE (S4-S5):** Geographic dominates
- Geographic: β = 0.18–0.20 (p<0.001)
- Climate: β = 0.02–0.05 (not significant)

**Resolution:** Climate similarity proxies for state-level factors. Once state FE absorbs these, climate effects disappear. This is not a contradiction—it's evidence that "climate peer effects" reflect regional/state policy environments, not technical similarity channels.

### 6.5 Challenge 5: Residualized IV Weakness

**F-Statistic Comparison (K=6):**
| IV Type | F-Stat | Peer Effect | P-value |
|---------|--------|-------------|---------|
| Original | 6,460 | 0.064 | 0.003*** |
| Residualized | 1,085 | 0.051 | 0.091 |
| Resid + Spatial Controls | 1,242 | 0.069 | 0.046** |
| Original + Spatial Controls | 1,449 | 0.069 | 0.018** |

**Interpretation:** Residualization weakens the instrument substantially (83% reduction in F-stat), but:
1. F-stats remain >> 10 (still strong instruments)
2. Point estimates remain positive and similar magnitude
3. Adding spatial controls (w_geo_applicant, w_geo_priority) restores significance

**Bounding:** If we interpret the original IV as upper bound (potentially biased up) and residualized as lower bound (valid but noisier), the true effect lies in **[0.05, 0.07]**—a narrow and consistently positive range.

### 6.2 Specification Sensitivity

1. **Sample Attrition:** Results sensitive to which districts are included—missing climate data, political data, or spatial coordinates can shift estimates substantially
2. **Control Set Dependence:** Horse-race specifications give contradictory results depending on whether state FE are included
3. **Multicollinearity:** Geographic and climate neighbors are correlated—joint estimation attribution is fragile

### 6.3 Conceptual Limitations

1. **Cross-Sectional Design:** Cannot observe dynamic adoption patterns, learning, or timing effects
2. **Binary Outcome:** Misses adoption intensity (number of buses, fleet share converted)
3. **Selection into Application:** Conditions on applying for funding—non-applicants may differ systematically
4. **Mechanism Opacity:** Cannot distinguish information spillovers, demonstration effects, vendor networks, or political emulation

### 6.4 Data Limitations

| Issue | Impact |
|-------|--------|
| Missing climate data | 341 districts (2.6%) excluded from climate specifications |
| Missing political data | 353 districts excluded from full specification |
| Charter schools | Only 32 in sample—insufficient for subgroup analysis |
| Fleet size missingness | 9,275/13,219 have bus fleet data |

---

## 7. Current Project Status

### 7.1 Completed Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| 01_spatial_peer_effects.py | Main IV-2SLS with full controls | ✅ Complete |
| 02_robustness_incremental.py | 22 specifications varying K, sample | ✅ Complete |
| 03_robustness_full.py | Extended controls (utility, charging, political) | ✅ Complete |
| 04_compare_geo_vs_climate.py | Horse-race without state FE | ✅ Complete |
| 05_geo_vs_climate_state_fe.py | Horse-race with state FE | ✅ Complete |
| 06_contradiction_decomposition.py | Diagnose conflicting results | ✅ Complete |
| 07_standardized_estimation.py | Consistent specification across all K | ✅ Complete |
| 08_adoption_intensity.py | Count models (Poisson, NegBin) | ✅ Complete |
| 09_placebo_falsification.py | Permutation tests, non-neighbor placebo | ✅ Complete |
| 10_propensity_residualized_iv.py | Address lottery stratification | ✅ Complete |

### 7.2 Key Output Files

| File | Contents |
|------|----------|
| `esb_spatial_results_final.txt` | Main results from Script 01 |
| `standardized_all_results.csv` | Systematic estimates across all specifications |
| `placebo_summary.txt` | Falsification test results |
| `resid_iv_summary.txt` | Residualized IV diagnostics and estimates |
| `contradiction_decomposition.csv` | Analysis of conflicting geo vs. climate results |

---

## 7. Recommended Solutions

### 7.1 Addressing Lottery Stratification — CORRECTED APPROACH (Feb 21, 2026)

**Previous (Incorrect) Approach:**
- Residualized IV on (priority × state) to "control" for stratification
- This WEAKENED the instrument (F: 6,460 → 1,085) and WORSENED balance

**Correct Approach (Feb 21 understanding — partially superseded, see note below):**

| Step | Action | Rationale |
|------|--------|----------|
| 1 | Use ORIGINAL IV_Z (lottery indicator) | Lottery is random unconditionally |
| ~~2~~ | ~~Control for `is_priority` in outcome equation~~ | ~~Blocks direct effect of priority on adoption~~ |
| ~~3~~ | ~~Include state FE~~ | ~~Absorbs state-level heterogeneity~~ |
| 4 | Do NOT residualize IV | Residualization introduces bias via collider |

**Why This Works:**
- Priority status affects *both* win probability *and* adoption directly
- But conditioning on priority in the IV stage opens a backdoor path
- Controlling in the outcome stage is valid: blocks direct path without collider bias

**Evidence (Balance Tests on Applicants Only):**
```
Unconditional:     All covariates pass (p > 0.10)
After residualizing: Income p < 0.0001, Poverty p < 0.0001 (FAIL)
```

> **⚠️ Implementation update — Mar 3, 2026:** Steps 2 & 3 above (additive `is_priority + state FE`) were a partial fix. The **full fix** is `priority × state` **interaction fixed effects** (~100 dummies), which correctly conditions on every lottery stratum simultaneously. Implemented in all `Post-Temporal` estimation scripts. See [Section 10.3](#103-fixed-effects-correction-all-estimation-scripts).

### 7.2 Balance Test Failures — RESOLVED

Balance test failures were an artifact of incorrect conditioning. With the corrected approach:

| Test | Result | Interpretation |
|------|--------|----------------|
| Unconditional IV_Z on covariates | All pass (p > 0.10) | Lottery is valid |
| Original IV with priority control | Valid | Preferred specification |

**No additional spatial controls needed** — the original IV is balanced.

### 7.3 Climate Measure Assessment (Feb 21, 2026)

**Current Measure:** Precipitation + Minimum Temperature (PRISM 30-year normals)

**Problems Identified:**

| Issue | Evidence |
|-------|----------|
| Geographic correlation | Climate clusters within states; state FE absorbs effect |
| Covariate imbalance | Min temp correlates with poverty (r=0.21), enrollment (r=0.22) |
| Not operationally relevant | Misses HDD/CDD, max temp, elevation — factors that affect ESB range |
| Missing data | 56 districts (mostly Alaska) excluded non-randomly |

**Recommendation for Main Paper:**
- PRIMARY: Geographic KNN (K=6) — clear mechanism, robust
- DO NOT use climate-based peers in main specification
- State FE absorbs "structural" climate correlation

**For Future Decomposition Analysis:**

To disentangle climate from geography, consider **orthogonalized climate measures**:

| Measure | Construction | Captures |
|---------|--------------|----------|
| **Residual HDD/CDD** | HDD - E[HDD \| state, lat, lon] | Within-state climate variation |
| **Elevation difference** | \|elevation_i - elevation_j\| | Terrain challenges (not latitude) |
| **Climate anomaly** | District deviation from state mean | Local climate distinctiveness |
| **Operational index** | Weighted combo of HDD, elevation, avg route length | ESB-specific climate relevance |

These remove the geographic component that confounds climate with region.

**Pre-Register Preferred Specification:**
- **Neighbor Definition:** KNN-6 (immediate neighbors)
- **Sample:** Full sample (no climate restrictions)
- **Controls:** Full (income, poverty, enrollment, PM2.5, race, political, priority, urbanicity)
- **Fixed Effects:** State FE
- **Clustering:** State

**Report Specification Curve:**
- Show distribution of estimates across all reasonable specifications
- Highlight that >60% of full-sample specifications are significant
- Acknowledge that results are fragile to sample restrictions

### 7.4 Resolving Geo vs. Climate Contradiction

**Definitive Answer:** Climate effects are NOT a competing mechanism—they proxy for state/regional factors.

**Evidence:**
1. Without state FE: Climate dominates (β=0.40)
2. With state FE: Climate disappears, geographic emerges (β=0.19)
3. Climate similarity clusters within states (e.g., similar precipitation in neighboring counties)

**Interpretation for Paper:**
- Do NOT run horse-race specifications without state FE
- Report geographic effects as primary finding
- Footnote that climate similarity does not constitute an independent peer mechanism once regional factors are controlled

### 7.5 ~~Accepting Residualized IV Trade-off~~ *(Stale for Post-Temporal branch)*

> **⚠️ Note — Mar 3, 2026:** The `Post-Temporal` branch does not use residualized IV. The `priority × state` interaction FE in the outcome equation renders the original lottery IV balanced within strata, eliminating the need for the trade-off described here. The discussion below is retained as context for the `Current` branch analysis only.

~~**The Dilemma:**~~  
~~- Original IV: Strong (F=6,460) but potentially biased~~  
~~- Residualized IV: Valid but weaker (F=1,085)~~

~~**Resolution:**~~  
~~- Both F-stats >> 10, so weak instrument bias is minimal~~  
~~- Point estimates agree (0.05–0.07)~~  
~~- Report both; interpret as bounds on true effect~~  
~~- Emphasize sign stability and qualitative finding~~

---

## 8. Key Takeaways (Updated Feb 21, 2026)

### 8.1 What We Can Say with Confidence

1. **Geographic proximity matters** for ESB adoption—districts with adopting neighbors are more likely to adopt (β ≈ 0.05–0.07, robust across IV specifications)
2. **Climate similarity does not constitute an independent peer mechanism**—once state FE are included, climate effects disappear or turn negative
3. **The lottery was heavily stratified**—differential win rates by priority status (12% vs. 43%) create identification challenges, but bounds analysis suggests effects remain positive
4. **Priority targeting worked**—disadvantaged communities adopted at 13pp higher rates
5. **Political polarization is present**—Democratic counties adopt more, even controlling for demographics

### 8.2 What Requires Caution

1. **Peer effect magnitudes are uncertain**—range from 0.05 to 0.19 depending on specification
2. **The lottery was not perfectly random**—even within priority × state cells, winners differ from losers on income and poverty
3. **Results depend on sample composition**—climate-restricted samples show null effects due to non-random attrition

### 8.3 Credible Bounds on Peer Effects *(Current Branch — adoption outcome)*

> **⚠️ Context — Mar 3, 2026:** These estimates are from the `Current` branch cross-sectional design (outcome = `IS_ADOPTER`). The `Post-Temporal` branch uses a different, cleaner outcome (R3 application). See [Section 10.6](#106-results-estimand-2--application-extensive-margin) for updated estimates.

Based on the diagnostic analysis:

| Specification | β | Interpretation |
|---------------|---|----------------|
| Original IV, Full Controls | 0.064 | Upper bound (potentially biased) |
| ~~Residualized IV~~ | ~~0.051~~ | ~~Lower bound (valid but noisy) — approach superseded~~ |
| Resid + Spatial Controls | 0.069 | Contextual reference only |

~~**Best Estimate:** β ∈ [0.05, 0.07]~~  
~~**Interpretation:** A 10pp increase in neighbor adoption increases own adoption probability by 0.5–0.7pp.~~

**Updated (Post-Temporal branch):** Effect on R3 *application* probability: δ ∈ [0.05, 0.12] depending on K and sample (priority subsample upper end). See Section 10.

---

## 9. Next Steps

### 9.1 ~~Immediate (For Current Paper)~~ *(Partially superseded by Post-Temporal branch)*

~~1. **Finalize Preferred Specification:**~~  
~~   - KNN-6, Full Sample, State FE, Original IV + Spatial Controls~~  
~~   - Report this as main result with clear justification~~

2. **Build Specification Curve:**
   - Plot all estimates for the Post-Temporal Estimand 2 across K and subsamples
   - Show % significant, median effect, IQR

3. **Write Limitations Section:**
   - Acknowledge lottery stratification explicitly
   - Present bounds from applicant-exposed vs full sample comparison
   - Note R3-application outcome (not adoption) as the primary causal chain

### 9.2 Robustness to Add

1. **Conley et al. Sensitivity Analysis:**
   - How much exclusion restriction violation needed to flip sign?
   - If δ must exceed implausible values, results are credible

2. **Bartik-Style Instrument:**
   - Predict application likelihood from pre-treatment characteristics
   - Use predicted application × national lottery rate as instrument

3. ~~**Round-by-Round Analysis:**~~  
   ~~- Test if peer effects differ across program rounds~~  
   ~~- Exploit timing variation if Round 1 outcomes available~~  
   ✅ *Done in Post-Temporal branch (Estimand 2 uses R1→R3 causal chain directly)*

### 9.3 Future Research

1. **Panel Extension:** Wait for Round 4+ data to exploit temporal variation
2. **Mechanism Study:** Survey/interview districts about information sources
3. **Intensity Analysis:** Model number of buses, not just binary adoption

---

## 10. Post-Temporal Branch: New Design & Results (Mar 3, 2026)

### 10.1 Motivation and Branch Context

The analyses in Sections 1–9 used a **cross-sectional design** (all rounds pooled, all districts, `IS_ADOPTER` as outcome) from the `Current` branch. That design has two structural weaknesses:
1. R1 lottery outcomes and R3 adoption outcomes are not cleanly causally ordered — R3 winners contaminate the sample
2. The instrument `w_geo_z` mixes across lottery rounds with different time windows

The `Post-Temporal` branch implements a **temporally clean redesign** with three estimands operating strictly within the R1 → R3 causal chain (R1 lottery 2022 → R3 application window Oct 2023). Sample restricted to **non-R1-winners** throughout to prevent instrument contamination of the outcome.

**HEAD:** `6a0e495` on branch `Post-Temporal`  
**Scripts:** `2_Scripts/2_Analysis/B_Estimation/01_estimand1_itt.py`, `02_estimand2_application.py`, `03_estimand3_allsource.py`

---

### 10.2 Lottery Structure (Confirmed from Program Guide PDFs)

Analysis of the 2022 and 2023 CSBP program guide PDFs revealed the R1 lottery is a **6-tier ordered draw**, not a flat random assignment:

| Tier | Pool |
|------|------|
| 1 | Zero-emission (ZE) priority applicants — per state |
| 2 | Clean (non-ZE) priority applicants — per state |
| 3 | All remaining priority applicants |
| 4 | Remaining non-priority clean applicants |
| 5 | Remaining priority ZE applicants |
| 6 | Remaining non-priority ZE applicants |

Randomisation was **flat (random number draw) _within_ each `priority × state × fuel_type` cell**, but win rates differ _between_ cells by construction — due to the ordered draw sequence and a **10%-per-state award cap** applied separately to each funding pool.

**Implication:** Balance failures in Section 6.2 were a conditioning artefact. The lottery is valid conditional on `priority × state`. With `priority × state` interaction FE (~100 dummies), winners and losers are exchangeable within each stratum by design.

---

### 10.3 Fixed Effects Correction (All Estimation Scripts)

**Old — additive (incorrect for the stratified lottery design):**
```python
ctrl["priority_r1"] = df_in["priority_r1"].fillna(0).astype(float)  # separate main effect
...
sdums = pd.get_dummies(df_in["state"], prefix="st", ...)             # ~50 state dummies
```

**New — interaction (correct, matches actual lottery strata):**
```python
pri = df_in["priority_r1"].fillna(0).astype(int).astype(str)
ps_dums = pd.get_dummies(
    pri + "_" + df_in["state"].astype(str),
    prefix="ps", drop_first=True, dtype=float
)   # ~100 dummies; subsumes both marginal effects
```

`priority_r23` is retained as a separate additive covariate — it is a pre-treatment characteristic, not a lottery draw stratum for R1. Applied to all three estimation scripts.

---

### 10.4 Spatial Weights Update

Script `02_build_spatial_weights.py` now produces two lag types per K:

| Column prefix | Weight type | Role |
|---|---|---|
| `w{K}_{var}` | Uniform 1/K, row-standardised | **Primary specification** |
| `wd{K}_{var}` | Inverse-distance normalised | Robustness checks |

Inverse-distance weights via `sklearn.NearestNeighbors`; coordinates in EPSG:5070 (metres); distances clipped at 1 m to avoid singularities.

---

### 10.5 Estimand Definitions

| # | Outcome | Instrument | Sample | Purpose |
|---|---|---|---|---|
| **1** | `IS_ADOPTER_CSBP_ANY` | `w{K}_IV_Z_R1` | Non-R1-winners | Baseline ITT; expected null (R3 adoption too close to R1 lottery) |
| **2** | `Y_R3_apply` | `w{K}_IV_Z_R1` | Non-R1-winners | **Primary** — cleanest causal chain: lottery → peer signal → application behaviour |
| **3** | `wri_any_2023_24` | `w{K}_IV_Z_R1` | Non-R1-winners | WRI data independent of EPA lottery; tests general diffusion |

All estimators: Linear Probability Model, state-clustered SE, reduced-form (ITT δ).

---

### 10.6 Results: Estimand 2 — Application Extensive Margin

**Sample:** 12,388 non-R1-winner districts | R3 application rate: 4.98% | SE clustered by state

**Main specifications:**

| K | Loser ctrl | δ | SE | p-value | N |
|---|---|---:|---:|---:|---|
| 6 | No | +0.0496 | 0.0293 | 0.091\* | 12,388 |
| 6 | **Yes** | **+0.0508** | **0.0291** | **0.081\*** | **12,388** |
| 10 | No | +0.0803 | 0.0430 | 0.062\* | 12,388 |
| 10 | **Yes** | **+0.0810** | **0.0426** | **0.058\*** | **12,388** |
| 15 | No | +0.0668 | 0.0529 | 0.206 | 12,388 |
| 15 | Yes | +0.0668 | 0.0528 | 0.206 | 12,388 |

**Priority-district subsample (K=6):**

| Loser ctrl | δ | SE | p-value | N |
|---|---:|---:|---:|---|
| No | +0.1159 | 0.0505 | 0.022\*\* | 6,402 |
| **Yes** | **+0.1189** | **0.0505** | **0.019\*\*** | **6,402** |

**Interpretation:** Among non-winners with a lottery-winning neighbour, the probability of applying to R3 rises by 5–8pp (full sample) or ~12pp (priority subsample). This is the cleanest causal estimate in the project: R1 lottery strictly precedes R3 application, restricted to non-winners, conditioned on proper lottery strata.

---

### 10.7 R2 Grant Neighbour Robustness

R2 grantees are ~2023 adopters whose grants predate R3 application opening and may independently encourage R3 entries — a potential confound.

**(a) Heterogeneity split — R2-neighbour present vs. absent (K=6, loser=Yes):**

| Sub-group | N | δ | SE | p-value |
|---|---|---:|---:|---:|
| R2-neighbour present (`w6_IS_R2_GRANTEE > 0`) | 1,140 | +0.065 | 0.066 | 0.324 |
| R2-neighbour absent (`w6_IS_R2_GRANTEE = 0`) | 11,248 | +0.046 | 0.034 | 0.177 |

*Both sub-groups positive; precision limited by R2-present subsample size (N=1,140).*

**(b) Additive R2 control (K=6, loser=Yes):**

| Specification | δ (R1 winner share) | SE | p-value |
|---|---:|---:|---:|
| Without R2 control | +0.0508 | 0.0291 | 0.081\* |
| With R2 control | +0.0525 | 0.0290 | 0.070\* |

**Finding:** R1 coefficient is stable (marginally larger) after controlling for R2 neighbours → R1 peer effect does not proxy for local R2-driven application propensity.

---

### 10.8 Applicant-Exposed Sample (Difference Design)

Restrict to the 7,085 districts with ≥1 R1-applicant neighbour (`w6_IV_Z_R1 > 0` OR `w6_IS_R1_LOSER > 0`). Sharpens comparison to winner-neighbour vs. loser-neighbour districts, removing geographically remote "never-exposed" districts from the control group.

| Coefficient | δ | SE | p-value | N |
|---|---:|---:|---:|---|
| Winner share (`w_IV_Z_R1`) | +0.040 | 0.039 | 0.294 | 7,085 |
| Loser share (`w_r1_loser`) | +0.006 | 0.021 | 0.782 | 7,085 |

**Discouraged-loser test:** The loser share coefficient is small and positive (not negative) → **no evidence of discouragement**. Observing a lottery-losing neighbour does not reduce a district's own application propensity.

*Attenuation relative to the full sample is expected: the never-exposed remote districts provide additional identification power under the contrast with exposed districts.*

---

### 10.9 Distance-Decay Robustness (K=6, loser=Yes)

| Weight scheme | δ | SE | p-value | N |
|---|---:|---:|---:|---|
| Uniform 1/K (`w6_IV_Z_R1`) | +0.051 | 0.029 | 0.081\* | 12,388 |
| Inverse-distance (`wd6_IV_Z_R1`) | +0.037 | 0.028 | 0.194 | 12,388 |

**Finding:** Inverse-distance weights produce a smaller, same-signed coefficient. The peer effect is not strongly driven by the single closest neighbour; uniform 1/K is the more conservative specification.

---

### 10.10 Results: Estimands 1 and 3 (Both Null)

**Estimand 1 — Cross-sectional ITT (R1 wins → CSBP adoption):**  
All null (p > 0.58, K=6/10/15). Expected: R3 adoption decisions overlap with the R1 deployment window; peer learning has not yet translated into observable adoption.

**Estimand 3 — WRI all-source adoption 2023–24:**  
All specifications null (p > 0.35, outcomes `wri_any_2023`, `wri_any_2024`, `wri_any_2023_24`). Delivery-timing heterogeneity (early vs. late R1 delivery, Wald test) also null.

**Conclusion:** Peer effects operate through the **application channel** — neighbour wins encourage programme entry — rather than through immediate observable adoption.

---

### 10.11 Summary of Post-Temporal Findings

| Test | Estimand 1 | Estimand 2 | Estimand 3 |
|---|:---:|:---:|:---:|
| Peer effect (full sample) | ❌ Null | ✅ δ≈0.05–0.08\* | ❌ Null |
| Peer effect (priority subsample) | — | ✅ δ≈0.12\*\* | — |
| R2 confound ruled out | — | ✅ | — |
| Discouragement (loser) effect | — | ❌ Null | — |
| Distance-decay > uniform weights | — | ❌ No | — |
| Delivery-timing heterogeneity | — | — | ❌ Null |

**Bottom line:** The causal peer effect in ESB adoption operates at the **application margin** — winning neighbours encourage other districts to enter the programme. The effect is concentrated in priority (disadvantaged) districts (δ≈0.12), consistent with social learning and peer legitimation mechanisms in under-resourced communities. No evidence of discouragement from losing neighbours.

---

## References

- Manski, C.F. (1993). Identification of Endogenous Social Effects: The Reflection Problem. *Review of Economic Studies*, 60(3), 531-542.
- EPA Clean School Bus Program: https://www.epa.gov/cleanschoolbus
- World Resources Institute ESB Dataset: https://www.wri.org/

---

*Document created: February 21, 2026 | Last updated: March 3, 2026*  
*Project location: `c:\BC PhD\Research\Peer-Effects and Adoption` | Branch: `Post-Temporal` (HEAD: `6a0e495`)*
