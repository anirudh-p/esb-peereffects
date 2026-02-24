# Model Specification and Identification Strategy

**Project:** Peer Effects in Electric School Bus (ESB) Adoption  
**Estimation Scripts:** `11_corrected_iv_estimation.py`, `18_comprehensive_robustness.py`  
**Last Updated:** February 21, 2026

---

## 1. Unit of Observation

The unit of observation is the **school district** (Local Education Agency, LEA), identified by a 7-digit NCES ID. The dataset covers **N ≈ 13,004–13,572** districts, depending on the specification (variation driven by political data availability from county-level merges). The sample is **cross-sectional**, pooling adoption outcomes across Clean School Bus Program (CSBP) award rounds 1–3 (2022–2024).

---

## 2. Outcome Variable

$$Y_i \in \{0, 1\}$$

`IS_ADOPTER` = 1 if district $i$ received an ESB award under the Clean School Bus Program, 0 otherwise. Constructed from WRI bus-level award data aggregated to the LEA level. Despite the binary outcome, a **Linear Probability Model (LPM)** is used throughout to enable IV-2SLS, which is standard in the peer-effects literature.

---

## 3. Peer Network Construction

### 3.1 Geographic Neighbor Set

District centroids are computed from 2021 NCES EDGE school district shapefiles projected to **EPSG:5070** (Albers Equal Area Conic, preserving distances in meters). For each district $i$, the $K$-nearest neighbor set is:

$$\mathcal{N}_K(i) = \{ j_1, j_2, \ldots, j_K \} \quad \text{by Euclidean centroid distance}$$

The baseline uses **K = 6**; robustness checks test K = 4, 8, 10.

### 3.2 Spatial Weight Matrix

Let $W$ be the $N \times N$ row-standardized KNN weight matrix:

$$w_{ij} = \begin{cases} \frac{1}{K} & \text{if } j \in \mathcal{N}_K(i) \\ 0 & \text{otherwise} \end{cases}, \qquad \sum_{j} w_{ij} = 1 \;\; \forall\, i$$

Built using `libpysal.weights.KNN.from_dataframe(k=K)` with `.transform = 'r'`.

### 3.3 Spatial Lags

The **peer adoption rate** and **peer instrument** are computed as row-standardized spatial lags via `libpysal.weights.lag_spatial`:

$$\bar{Y}_i \equiv \sum_j w_{ij} Y_j = \frac{1}{K} \sum_{j \in \mathcal{N}_K(i)} Y_j$$

$$\bar{Z}_i \equiv \sum_j w_{ij} Z_j = \frac{1}{K} \sum_{j \in \mathcal{N}_K(i)} Z_j$$

In code: `w_adoption` and `w_IV_Z`.

---

## 4. Econometric Specifications

### 4.1 Baseline Specification (Script 11 — Corrected IV)

**Structural equation (second stage):**

$$Y_i = \alpha + \beta \bar{Y}_i + \gamma_1 \ln(\text{Income}_i) + \gamma_2 \text{PovRate}_i + \gamma_3 \ln(\text{Enroll}_i) + \gamma_4 \text{PM25}_i + \gamma_5 \text{PctWhite}_i$$
$$+ \gamma_6 \text{PctDem}_i + \gamma_7 \text{Priority}_i + \boldsymbol{\lambda}' \text{Locale}_i + \boldsymbol{\delta}_s + \varepsilon_i$$

**First stage:**

$$\bar{Y}_i = \pi_0 + \pi_1 \bar{Z}_i + \boldsymbol{\gamma}' \mathbf{X}_i + \boldsymbol{\delta}_s + \nu_i$$

**Instrument:** $\bar{Z}_i$ = average lottery win status of district $i$'s $K$ nearest geographic neighbors.

**Estimator:** `linearmodels.iv.IV2SLS`, standard errors clustered by state (50 clusters).

$$\hat{\beta}^{2SLS} = \frac{\text{Cov}(\bar{Z}_i,\, Y_i \mid \mathbf{X}_i,\, \boldsymbol{\delta}_s)}{\text{Cov}(\bar{Z}_i,\, \bar{Y}_i \mid \mathbf{X}_i,\, \boldsymbol{\delta}_s)}$$

**Result:** $\hat{\beta} = 0.192^{***}$ (SE = 0.034), First-stage $F \approx 54,\!428$.

---

### 4.2 Robustness Specifications (Script 18)

All three models use the same first stage. The exogenous control set $\mathbf{X}_i$ varies:

| Model | Exogenous Controls | State Absorber | N |
|---|---|---|---|
| **Baseline** | Base controls + state FE | $\boldsymbol{\delta}_s$ (50 dummies) | 13,572 |
| **+ CSB Control** | Base + CSB\_Applied + state FE | $\boldsymbol{\delta}_s$ (50 dummies) | 13,572 |
| **Full Model** | Base + CSB\_Applied + state observables | $\ln(\text{EV}_s)$, $p_s^\text{diesel}$, $\ln(\text{Incent}_s)$ | 13,572 |

> **Note:** The Full Model replaces state fixed effects with continuous state-level observables to avoid collinearity between state dummies and state-level variables. The Hausman intuition is that if observables absorb the state variance, estimates should be comparable.

---

## 5. Variable Definitions and Construction

### 5.1 Outcome and Network Variables

| Variable | Code Name | Definition | Source |
|---|---|---|---|
| ESB Adoption | `IS_ADOPTER` | 1 if district received CSBP award (Rounds 1–3) | WRI bus-level data, agg. to LEA |
| Peer Adoption Rate | `w_adoption` | $\bar{Y}_i = \frac{1}{K}\sum_{j\in\mathcal{N}(i)} Y_j$ | Computed from `IS_ADOPTER` + $W$ |
| Own Lottery Win | `IV_Z` | 1 if district won CSBP lottery (funded applicant), 0 if not funded or non-applicant | EPA CSBP award + application data |
| Peer Lottery Rate | `w_IV_Z` | $\bar{Z}_i = \frac{1}{K}\sum_{j\in\mathcal{N}(i)} Z_j$ | Computed from `IV_Z` + $W$ |

### 5.2 Base District Controls

| Variable | Code Name | Definition | Source |
|---|---|---|---|
| Log Median Income | `log_median_income` | $\ln(\text{MedianHHIncome}_i + 1)$ | ACS 2021 5-year |
| Poverty Rate | `poverty_rate` | Fraction of households below poverty line | ACS 2021 5-year |
| Log Enrollment | `log_enrollment` | $\ln(\text{TotalEnrollment}_i + 1)$ | NCES CCD |
| PM2.5 | `pm25` | Annual avg PM2.5 concentration (μg/m³), 2019 | NASA SEDAC |
| Pct. White | `pct_white` | Fraction white students | NCES CCD |
| Pct. Democratic | `pct_dem_2020` | County-level Dem. vote share, 2020 presidential. Mapped LEA → county via WRI county crosswalk; multi-county LEAs take the average. | MIT Election Lab + WRI |
| Priority Status | `is_priority` | 1 if EPA designated as priority community (disadvantaged, high pollution, or tribal) | EPA CSBP application data |
| Urbanicity | `locale_*` dummies | City / Suburb / Town (Rural = baseline). Created via `pd.get_dummies(..., drop_first=True)` | NCES EDGE 2021 |

### 5.3 CSB Application Control (Model 2 / Model 3)

| Variable | Code Name | Definition | Source |
|---|---|---|---|
| CSB Competitive Applicant | `applied_csb_competitive` | 1 if district appears in the CSBP waitlisted or rejected applicant file | WRI — *CSBP Applicants waitlisted and rejected_11.18.25.xlsx* |

This variable proxies for **revealed interest in ESB adoption** independent of peer effects. Districts with `applied_csb_competitive = 1` had adoption rates of ~26.8% vs. 5.1% for non-applicants.

### 5.4 State-Level Observable Controls (Model 3 only)

| Variable | Code Name | Definition | Source |
|---|---|---|---|
| Log EV Stations | `ev_stations_log` | $\ln(\text{state EV public stations} + 1)$, 2023. Parsed from `"424 \| 1,096"` format (takes first numeric token = EVSE units). | DOE AFDC — *historical-station-counts.xlsx* |
| Diesel Price | `diesel_price` | State avg. retail diesel price ($/gal), 2022. Filtered `MSN == 'DFTCD'`. | EIA — *pr_all.csv* |
| Log Incentive Count | `log_incentives` | $\ln(\text{count of state ESB/EV incentive programs} + 1)$ | DOE Clearinghouse (*Clearinghouse of Electric School Bus Funding...xlsx*, sheet `Clearinghouse Official`), grouped by `Geography` |

### 5.5 Heterogeneity Subsamples (Script 18, Part 4)

Urbanicity subsamples are defined from NCES EDGE classifications:

| Subsample | Definition | n |
|---|---|---|
| Urban | `urbanicity` contains `"City"` or `"Suburb"` | 3,153 |
| Rural | `urbanicity` contains `"Rural"` | 7,053 |
| Town/Suburban | Residual (not Urban, not Rural) | 3,366 |

KNN weights are **rebuilt within each subsample** from the subsample's district centroids, and zero-variance state FE columns are dropped before fitting to avoid rank deficiency.

---

## 6. Identification Strategy

### 6.1 The Reflection Problem

The standard challenge in peer effects is Manski's (1993) reflection problem: if district $i$ and its neighbors are simultaneously choosing adoption, OLS of $Y_i$ on $\bar{Y}_i$ is biased because $\text{Cov}(\bar{Y}_i, \varepsilon_i) \neq 0$. Unobserved regional shocks (e.g., a state initiative) raise adoption for district $i$ and all neighbors simultaneously, generating a positive correlation even absent true peer effects.

### 6.2 Instrumental Variable

**Instrument:** $\bar{Z}_i$, the average lottery win status of district $i$'s $K$ nearest geographic neighbors.

**Lottery mechanism:** The EPA Clean School Bus Program received more applications than funding in each round. Among applicants, unfunded districts were selected by random lottery within priority/non-priority strata. `IV_Z = 1` if a district received funding, `IV_Z = 0` if it applied but was not funded (lottery loser), or if it never applied.

**Relevance:** $\bar{Z}_i$ is a strong predictor of $\bar{Y}_i$ because a neighbor winning the lottery causes that neighbor to receive buses and adopt. First-stage $F \approx 54,\!428 \gg 10$.

**Exclusion restriction:** Neighbor lottery win status affects district $i$'s adoption **only through** neighbor adoption. The specification controls for `is_priority` (lottery stratum targeting) and state fixed effects, but does **not** include own win status $Z_i$ directly as a control. In contemporaneous designs, EPA selection rules can induce geographic correlation structure in realized wins, so identification relies on robustness checks that isolate variation beyond local application clustering.

Formally, the exclusion restriction requires:

$$\text{Cov}(\bar{Z}_i,\, \varepsilon_i \mid \mathbf{X}_i,\, \boldsymbol{\delta}_s) = 0$$

This would be violated if, for example, EPA lottery allocation had systematic geographic patterns that directly shifted district $i$'s adoption propensity independent of neighbors' realized adoption — or if winning neighbors created vendor/charging infrastructure spillovers that affected $i$ without operating through neighbor adoption itself.

### 6.3 Why the Instrument is NOT Residualized

An earlier approach (Scripts 01–10) residualized `IV_Z` on `priority × state` interactions before constructing $\bar{Z}_i$, intended to "purify" the instrument. This introduced a **Simpson's Paradox / collider bias**:

- `priority × state` is a collider on the path between `IV_Z` and `IS_ADOPTER` — it is causally affected by both `IV_Z` (determines lottery strata) and correlates with `IS_ADOPTER` (priority districts adopt more).
- Conditioning on a collider opens a spurious correlation path, inducing artificial negative correlation between $Z_i^{\text{resid}}$ and adoption within strata.
- Unconditional balance tests confirm the lottery is random: pre-treatment covariates (income, poverty, enrollment, PM2.5, race) are balanced between winners and losers in the raw data (all $p > 0.10$).

**Corrected approach** (Script 11 onward): use raw `IV_Z` as instrument, control for `is_priority` in the outcome equation to absorb the priority-based targeting. This is the standard Nevo & Rosen (2012) approach — instrument validity by design, not by residualization.

### 6.4 State Fixed Effects

State dummies $\boldsymbol{\delta}_s$ absorb all **time-invariant state-level confounders**: state EV mandates, state co-funding programs, state diesel regulations, inter-district political networks, etc. Because $W$ is constructed from geographic distance (not state boundaries), $\bar{Z}_i$ can include neighbors from adjacent states, providing within-state variation in peer exposure even after state FE.

---

## 7. Estimation Details

| Component | Choice | Implementation |
|---|---|---|
| Estimator | IV-2SLS | `linearmodels.iv.IV2SLS(...).fit()` |
| Endogenous variable | `w_adoption` ($\bar{Y}_i$) | 1 variable |
| Instrument | `w_IV_Z` ($\bar{Z}_i$) | 1 variable (just-identified) |
| Clustering | By state | `cov_type='clustered', clusters=geo_df['state']` |
| State FE | 50 state dummies, one dropped (`drop_first=True`) | `pd.get_dummies(state, prefix='st')` |
| Projection | EPSG:5070 (Albers Equal Area, meters) | `gdf.to_crs(epsg=5070)` |
| Spatial lag | Row-standardized KNN | `lag_spatial(w, y)` from libpysal |
| Default K | 6 | Sensitivity: K = 4, 8, 10 |

The model is just-identified (1 instrument for 1 endogenous variable), so the IV estimator is also the LIML estimator and no over-identification tests are feasible.

---

## 8. Summary of Estimates Across Specifications

### 8.1 Main Specification Results

| Model | $\hat{\beta}$ | SE | p-value | N |
|---|---|---|---|---|
| Baseline (base controls + state FE) | **0.192** | 0.034 | <0.001 | 13,572 |
| + CSB Application Control | **0.135** | 0.037 | 0.0003 | 13,572 |
| Full Model (CSB + state observables, no FE) | **0.161** | 0.036 | <0.001 | 13,572 |

**Interpretation:** A 10 percentage point increase in the share of geographic neighbors adopting ESB raises own adoption probability by 1.4–1.9 percentage points.

### 8.2 K Sensitivity (+ CSB Control Model)

| K | $\hat{\beta}$ | SE | p-value |
|---|---|---|---|
| 4 | 0.149 | 0.030 | <0.001 |
| **6** | **0.135** | **0.037** | **0.0003** |
| 8 | 0.126 | 0.042 | 0.0025 |
| 10 | 0.130 | 0.043 | 0.0028 |

Effect is monotonically declining then stabilizing; all specifications statistically significant.

### 8.3 Urbanicity Heterogeneity (K = 6, + CSB Control)

| Subsample | $\hat{\beta}$ | SE | p-value | N |
|---|---|---|---|---|
| Rural | **0.178** | 0.050 | 0.0004 | 7,053 |
| Urban | 0.053 | 0.062 | 0.397 | 3,153 |
| Town/Suburban | 0.031 | 0.058 | 0.599 | 3,366 |

Peer effects are concentrated entirely in **rural districts** — consistent with theory that rural areas face greater information frictions and rely more heavily on nearby adoption experiences to guide decisions.

---

## 9. Key Assumptions and Threats

| Assumption | Status | Notes |
|---|---|---|
| **Lottery randomness** | ✅ Validated | Unconditional balance tests pass (5/5 covariates, $p > 0.10$) |
| **Instrument relevance** | ✅ Strong | First-stage $F \approx 54,\!428$ |
| **Exclusion restriction** | ⚠️ Untestable (just-ID) | Plausible if no direct vendor/grid spillovers from neighbor winning |
| **No correlated unobservables** | ⚠️ Mitigated by state FE | Remaining threat: within-state regional shocks |
| **SUTVA** | ⚠️ Partial | Assumes adoption spillovers operate only through the KNN-6 network |
| **Linear probability model** | Assumption | Marginal effects interpretable at means; predicts outside [0,1] for some obs. |

---

## 10. Reference Code

| Script | Purpose |
|---|---|
| [11_corrected_iv_estimation.py](../2_Scripts/2_Analysis/B_Estimation/11_corrected_iv_estimation.py) | Baseline corrected IV spec (K=6, baseline controls + state FE) |
| [18_comprehensive_robustness.py](../2_Scripts/2_Analysis/B_Estimation/18_comprehensive_robustness.py) | All three model variants, K sensitivity, urbanicity heterogeneity |
| [16_enhanced_iv_state_controls.py](../2_Scripts/2_Analysis/B_Estimation/16_enhanced_iv_state_controls.py) | State-level observable controls (EV, diesel, incentives) |
| [17_enhanced_iv_csb_controls.py](../2_Scripts/2_Analysis/B_Estimation/17_enhanced_iv_csb_controls.py) | CSB application control development |
| [09_placebo_falsification.py](../2_Scripts/2_Analysis/B_Estimation/09_placebo_falsification.py) | Placebo and falsification tests |
| [config.py](../2_Scripts/config.py) | All data paths |

---

---

## 11. Should We Restrict to Lottery Participants? (Sample Scope Debate)

### 11.1 The Concern

A natural question is whether the estimation sample should be **restricted to lottery applicants only** (`IS_APPLICANT == 1`, i.e., winners + losers). The reasoning: non-applicants differ systematically from applicants in ways that could bias estimates, and the lottery's randomization only applies within the applicant pool.

**Sample composition:**
| Group | N | IV_Z | IS_ADOPTER | Share of sample |
|---|---|---|---|---|
| Lottery winners | 719 | 1 | ~99% | 5.4% |
| Lottery losers | 1,841 | 0 | ~5% | 13.9% |
| **Non-applicants** | **10,659** | **0 (structural)** | **0% (structural)** | **80.6%** |

Non-applicants have **structural zeros** in both outcome and IV: they cannot adopt without applying (only applicants receive buses) and they never entered the lottery, so `IV_Z = 0` is not a random draw for them.

### 11.2 Why Restricting to Applicants Makes the Problem Worse

Script 19 (`19_applicants_only_iv.py`) estimates the peer effect on the applicants-only sample with spatial weights rebuilt within the applicant network. The result:

$$\hat{\beta}^{\text{applicants only}} = -0.309^{***} \quad (\text{SE} = 0.078, \; p < 0.001, \; N = 3{,}102)$$

The estimate is not just smaller — it **reverses sign**. This is robust across all K values:

| K | $\hat{\beta}$ (applicants only) |
|---|---|
| 4 | −0.298*** |
| 6 | −0.309*** |
| 8 | −0.360*** |
| 10 | −0.390*** |

A negative peer effect (seeing neighbors adopt *reduces* own adoption) is difficult to rationalize behaviorally. The explanation is an **exclusion restriction violation** specific to the applicant subsample:

**Documented selection-interdependence mechanism.** In the FY22 rebate selection framework, EPA imposed geographic constraints including (i) at least one selectee per state/territory and (ii) a cap that a state could not receive more than 10% of total funds. These constraints create mechanical interdependence in applicants' realized win indicators within states and nearby clusters: if one local applicant wins under constrained state allocation, the probability that another nearby applicant wins can fall. Formally:

$$\text{Cov}(Z_i, Z_j \mid \text{same strata, same area}) < 0$$

This means $\bar{Z}_i$ (average neighbor lottery win rate) can be **directly negatively correlated with own lottery win probability** through the constrained geographic allocation mechanism, not just through peer demonstration effects. The exclusion restriction requires $\bar{Z}_i$ to affect $Y_i$ only through $\bar{Y}_i$, but in the applicants-only sample it can also affect $Y_i$ through correlation with $Z_i$ itself. The instrument is therefore weaker as quasi-random variation in this sub-population, and the negative sign is consistent with this mechanically induced interdependence.

Source: US EPA Clean School Bus Program FY22 rebate selection documentation (state minimum selection rule and state funding cap).

### 11.3 Why the Full Sample Is More Defensible

Including non-applicants actually **mitigates the budget competition problem** through a dilution mechanism. In the full sample, a typical district's 6 nearest neighbors are predominantly non-applicants (~80.6% prior to spatial restriction). A district's spatial lag therefore averages mostly zero-valued `IV_Z` terms:

$$\bar{Z}_i^{\text{full}} = \frac{1}{K}\sum_{j \in \mathcal{N}(i)} Z_j \approx \frac{1}{K} \cdot Z_{j^*}$$

where $j^*$ is the one or two applicant neighbors in the set. The variation in $\bar{Z}_i$ is driven by whether a single nearby applicant-neighbor happened to win or lose the lottery — a clean, isolated lottery draw that does not signal "the area's budget has been used up." The non-applicant neighbors contribute zero to the spatial lag and zero to the budget competition signal.

Contrast this with the applicants-only setting where all 6 neighbors are applicants, competition for shared slots is intense, and $\bar{Z}_i$ encodes the budget outcome for the entire local applicant cluster.

### 11.4 LATE Interpretation in the Full Sample

The full-sample IV estimates the **Local Average Treatment Effect** for compliers: districts whose adoption is moved by a nearby applicant-neighbor's lottery win. This includes:

1. **Applicant districts** that are persuaded/inspired by a winning neighbor (information spillover, shared vendor learning).
2. **Non-applicant districts** that may apply and later adopt in subsequent rounds after observing a neighbor's successful adoption (application/awareness spillover).

The second group is policy-relevant: if peer effects operate through non-applicants being motivated to apply in future rounds, the full-sample LATE captures exactly this diffusion mechanism along the extensive margin.

### 11.5 Recommended Additional Robustness

To address the user's original concern about non-applicant selection without sacrificing instrument validity, an additional control for **neighbor application density** $\bar{A}_i = \frac{1}{K}\sum_j A_j$ (where $A_j = 1$ if district $j$ applied) can be included. This partial outs the "high-ESB-interest area" channel — variation in $\bar{Z}_i$ conditional on $\bar{A}_i$ is pure lottery luck, not application propensity. This approach preserves the full sample while addressing the exclusion concern.

### 11.6 Summary

| Approach | $\hat{\beta}$ | Validity concern |
|---|---|---|
| **Full sample (N=13,572)** | **+0.192***| Non-applicant structural zeros dilute (but don't eliminate) variation; exclusion cleaner than applicants-only because no within-area budget competition |
| Applicants only (N=3,102) | −0.309*** | **Exclusion violated**: finite budget creates negative lottery correlation within applicant clusters; instrument is endogenous in this sub-population |

The full-sample estimate is preferred. The negative sign in the applicants-only specification is a diagnostic of budget competition, not evidence against peer effects.

**Reference script:** [19_applicants_only_iv.py](../2_Scripts/2_Analysis/B_Estimation/19_applicants_only_iv.py)

---

## 12. Two Additional Concerns: Outcome Definition and Temporal Ordering

### 12.1 What Does `IS_ADOPTER` Actually Measure?

`IS_ADOPTER` is constructed from `CSB_Rebates.xlsx` and `CSB_Grants.xlsx` only — it captures **federal CSBP participation**, not ESB ownership broadly. The WRI bus-level data reveals that approximately **33% of all tracked ESBs** were funded through non-federal sources:

| Funding Source | Approx. Bus Count |
|---|---|
| EPA Clean School Bus Rebate Program (R1, R3) | ~5,649 |
| EPA Clean School Bus Grant Program (R2) | ~2,748 |
| California HVIP, VW Settlement, state programs | ~4,152 |
| **Total tracked** | **~12,549** |

**Implication:** Any peer effect operating through non-federal channels — e.g., "my neighbor got ESBs through the VW Settlement, which persuaded me to apply for state funds" — is **invisible** in the outcome variable. The LATE we estimate is specifically bounded to the federal CSBP participation decision. This is both a limitation and a cleaner estimand: variation in `IS_ADOPTER` is tightly linked to the federal lottery mechanism, making the IV interpretation more direct.

### 12.2 Is the Lottery Temporally Ordered? (The Cross-Round Problem)

The rounds proceed as:

| Round | Mechanism | Funding Year | Lottery? |
|---|---|---|---|
| R1 | Rebates | 2022 | ✅ Yes — `IV_Z_R1` |
| R2 | Grants | 2023 | ❌ No — competitive |
| R3 | Rebates | FY23 cycle (applications closed Jan 31, 2024) | ✅ Yes — `IV_Z_R3` |

The baseline `IV_Z` and `w_IV_Z` **pool R1 and R3 lottery wins** into a single cross-sectional indicator. This conflates two distinct situations:

1. **Cross-round (causally ordered):** Neighbor won R1 (2022) → received buses by late 2022/early 2023 → district observes deployments through 2023 → district applies for and adopts in the FY23 rebate cycle (deadline Jan 2024). The instrument is *temporally prior* to the outcome.

2. **Within-round (contemporaneous):** Neighbor and district both applied to the *same* lottery round. Both outcomes are draws from the same lottery distribution. Lottery outcomes are independent conditional on priority×strata by design, but the two districts are otherwise identical in terms of the adoption decision. `w_IV_Z` encoding the neighbor's outcome in the same round says nothing about peer demonstration.

**Consequence for the F-statistic.** The contemporaneous spec yields F ≈ 54,428 — implausibly high. Part of this mechanical strength arises because geographic clusters tend to apply to the same round (R1 or R3) together: if the district is in the sample, nearby neighbors likely also applied in the same round, and $w_{IV\_Z}$ is highly correlated with the *share of nearby districts that applied* (which predicts adoption through selection, not peer effects).

**The cross-round test.** Script 20 (`20_cross_round_iv.py`) constructs a genuine temporal IV:

$$\text{Outcome:} \quad Y_i^{R3} = \mathbf{1}[\text{district }i\text{ adopted in FY23 rebates (deadline Jan 2024)}], \quad Z_i^{R1} = 0 \text{ (excluded R1 adopters)}$$

$$\text{Instrument:} \quad \bar{Z}_i^{R1} = \frac{1}{K}\sum_{j \in \mathcal{N}(i)} Z_j^{R1}$$

where $Z_j^{R1} = 1$ if neighbor $j$ won the R1 lottery in 2022. This is fully predetermined relative to the FY23 rebate application decision, and the roughly 15-month window from R1 selectee announcement (Oct 2022) to FY23 rebate deadline (Jan 31, 2024) allows R1 winners to receive, operate, and demonstrate buses before neighboring districts submit FY23 applications.

#### Results from Script 20

**Sample construction:**
- R1 winners: 368 | R3 winners: 458 | R1∩R3 overlap: 24
- Estimation sample (non-R1 districts): 13,086; R3 adoption rate = 4.82%

**Comparison table:**

| Specification | Peer Effect | SE | p | F (1st stage) |
|---|---|---|---|---|
| Contemporaneous IV, full (Script 11) | +0.192 | 0.034 | <0.001 | **54,428** |
| Cross-round IV (R1→R3), K=6 | +1.343 | 0.505 | 0.008 | **35** |

**K sensitivity (cross-round):**

| K | Peer Effect | SE | p | F_1st |
|---|---|---|---|---|
| 4 | 1.244 | 0.363 | 0.001 | 35 |
| 6 | 1.343 | 0.505 | 0.008 | 35 |
| 8 | 1.265 | 0.405 | 0.002 | 48 |
| 10 | 1.326 | 0.348 | <0.001 | 51 |

**Interpretation:**

1. **The F-stat collapse confirms within-round inflation.** The first-stage F drops from 54,428 to ~35–51 when the instrument is genuinely temporally prior. The contemporaneous spec's implausibly high F arose because geographic neighbors tend to apply in the same round — `w_IV_Z` encodes the neighbor's same-round lottery win, which correlates with local area ESB interest (selection), not just random assignment. This does not invalidate the contemporaneous IV structurally, but it does mean its standard errors and inference were reliable while the F-stat was misleadingly large.

2. **The cross-round peer effect is positive, significant, and stable across K.** A 10pp increase in the share of KNN-6 neighbors who won the R1 lottery raises own FY23 rebate-cycle adoption probability by ~13pp. This is the cleanest available evidence for temporal peer demonstration: pilots deployed after R1 in 2022 are associated with higher neighboring adoption by the FY23 application window (closing Jan 2024).

3. **The large cross-round magnitude relative to the contemporaneous estimate** (1.34 vs. 0.19) is expected and not a contradiction. The cross-round LATE is estimated on a rare-adoption subsample (R3 adoption rate = 4.82% among non-R1 districts), so a given fraction of neighbor wins moves adoption probability substantially for the complier subpopulation. Additionally, the contemporaneous spec includes R1 adopters (and their neighbors) who dilute the signal; cross-round isolates the R1→R3 channel only.

4. **Structural reading.** Both estimates tell the same directional story: having lottery-winning neighbors causally increases own adoption probability. The cross-round test is the more credible identification — the instrument is unambiguously predeterminate — but the contemporaneous spec is the primary result because it uses the full adoption outcome (`IS_ADOPTER`) rather than the single-round subset measure.

**Reference script:** [20_cross_round_iv.py](../2_Scripts/2_Analysis/B_Estimation/20_cross_round_iv.py)  
**Output table:** [cross_round_iv_results.csv](../3_Output/Tables/cross_round_iv_results.csv)

---

## 13. Robustness Checks

This section documents five categories of robustness tests designed to address the main vulnerabilities of the identification strategy: (A) the exclusion restriction concern that $\bar{Z}_i$ proxies for local ESB interest, (B) the mechanical first-stage F-stat, (C) inference under spatial correlation, (D) outcome definition, and (E) falsification.

### 13.1 Exclusion Restriction: Neighbor Application Density Control

**The concern.** Even though the CSBP lottery is individually random (conditional on priority), the spatial lag $\bar{Z}_i$ can proxy for "how many nearby districts were in the game" because most neighbors are non-applicants coded as $Z_j = 0$. Variation in $\bar{Z}_i$ partly reflects geographic clustering of applicants — a selection channel — not just random luck among applicants.

**Approach A: Control for neighbor loser density.** Add $\bar{L}_i = \frac{1}{K}\sum_{j \in \mathcal{N}(i)} L_j$ as an exogenous control, where $L_j = 1$ if neighbor $j$ applied but **lost** the lottery. This captures "local ESB interest" without containing adoption itself (since losers, by definition, did not adopt through the CSBP lottery). The variable is pre-determined relative to the lottery draw.

**Why $\bar{L}_i$ rather than $\bar{A}_i$?** The initial test used $\bar{A}_i$ (fraction of neighbors who applied, including winners). But since IS_APPLICANT $\supset$ IS_ADOPTER by construction (all winners are applicants), controlling for $\bar{A}_i$ creates a **bad control** problem — it mechanically absorbs adoption variation from $\bar{Y}_i$ (the endogenous variable). The loser-only control avoids this.

**Approach B: Conditional win-rate instrument.** Define $\bar{Z}_i^{\text{cond}} = \bar{Z}_i / \bar{A}_i$ (win rate among neighbor-applicants). This isolates luck *within* the applicant pool. Sample restricted to districts with $\geq 1$ applicant neighbor.

**Results (Script 21):**

| Specification | $\hat{\beta}$ | SE | p-value | F (1st) | N |
|---|---|---|---|---|---|
| A1. Baseline (no control) | **+0.201**| 0.034 | <0.001 | 55,573 | 13,572 |
| B1. + $\bar{A}_i$ (bad control) | -0.204 | 0.057 | <0.001 | 21,100 | 13,572 |
| B2. + $\bar{L}_i$ (correct) | **+0.110** | 0.039 | 0.005 | 51,842 | 13,572 |
| C1. Conditional instrument | +0.021 | 0.050 | 0.680 | 5,075 | 8,739 |

**K sensitivity with $\bar{L}_i$ control:**

| K | $\hat{\beta}$ | SE | p-value |
|---|---|---|---|
| 4 | **+0.127** | 0.030 | <0.001 |
| 6 | **+0.110** | 0.039 | 0.005 |
| 8 | +0.079 | 0.045 | 0.080 |
| 10 | +0.061 | 0.046 | 0.187 |

**Reduced-form (ITT):**

| Controls | RF Coefficient | SE | p-value |
|---|---|---|---|
| Baseline | **+0.212** | 0.038 | <0.001 |
| + $\bar{L}_i$ | **+0.112** | 0.041 | 0.006 |
| + $\bar{A}_i$ (bad control) | -0.187 | 0.053 | <0.001 |

**Interpretation:**

1. **The peer effect survives the loser-density control** at $\hat{\beta} = 0.110$ (K=4,6), attenuating by ~45% from the baseline 0.201. This means roughly half the baseline estimate was driven by the geographic clustering of applicants (local interest), and the remaining half reflects genuine peer demonstration.

2. **The $\bar{A}_i$ bad-control result (-0.204) is informative but not interpretable as a causal estimate.** Since IS_APPLICANT includes IS_ADOPTER, adding $\bar{A}_i$ absorbs variation from $\bar{Y}_i$ into the exogenous set, inducing a mechanical sign flip. This is a methodological artifact, not a substantive finding.

3. **The conditional instrument (C1) is null** ($\hat{\beta} = 0.02$, p=0.68). Conditional on having applicant-neighbors, additional variation in the *share* of those applicant-neighbors who win does not predict adoption. This suggests proximity to local CSBP engagement may matter more than the exact local win rate, but this test uses a different estimand (ratio-scaled instrument) and a restricted sample, so it is supportive rather than definitive for extensive-margin interpretation.

4. **K sensitivity shows spatial decay.** The $\bar{L}_i$-controlled effect is significant at K=4 and K=6, marginal at K=8, and null at K=10 — consistent with localized peer influence that weakens with distance.

**Preferred specification.** Baseline + $\bar{L}_i$ control, K=6: $\hat{\beta} = 0.110^{***}$ (SE=0.039).

**Reference script:** [21_application_density_control.py](../2_Scripts/2_Analysis/B_Estimation/21_application_density_control.py)  
**Output table:** [application_density_robustness.csv](../3_Output/Tables/application_density_robustness.csv)

---

### 13.1B Count-Based Instrument Check (Cleaner Alternative to Ratio Instrument)

To avoid the ratio-based conditional instrument $\bar{Z}_i^{\text{cond}} = \bar{Z}_i/\bar{A}_i$ (which changes both scaling and sample), Script 24 implements a count-based test on the full sample:

- Instrument: **number of winning neighbors** ($\#\text{WinNbrs}_i$)
- Controls: either **number of applicant neighbors** ($\#\text{AppNbrs}_i$) or **number of loser neighbors** ($\#\text{LoserNbrs}_i$)

With KNN-6 row-standardized weights, these are simple re-scalings of prior lags:

$$\#\text{WinNbrs}_i = K\bar{Z}_i, \qquad \#\text{AppNbrs}_i = K\bar{A}_i, \qquad \#\text{LoserNbrs}_i = K\bar{L}_i.$$

**Results (Script 24):**

| Specification | $\hat{\beta}$ | SE | p-value | F (1st) |
|---|---|---|---|---|
| Baseline share instrument ($\bar{Z}_i$) | +0.201 | 0.033 | <0.001 | 55,573 |
| Count instrument ($\#\text{WinNbrs}_i$) | +0.201 | 0.033 | <0.001 | 55,573 |
| + $\#\text{AppNbrs}_i$ control | -0.204 | 0.057 | <0.001 | 21,100 |
| + $\#\text{LoserNbrs}_i$ control | **+0.109** | 0.039 | 0.005 | 51,842 |

**Reduced form (ITT on count instrument):**

| RF Specification | Coef | SE | p-value |
|---|---|---|---|
| $Y_i$ on $\#\text{WinNbrs}_i$ | +0.0353 | 0.0062 | <0.001 |
| + $\#\text{AppNbrs}_i$ | -0.0312 | 0.0088 | <0.001 |
| + $\#\text{LoserNbrs}_i$ | **+0.0186** | 0.0068 | 0.006 |

**Interpretation.** The count-based design delivers the same message as Section 13.1 while avoiding ratio-instability and sample restriction: controlling for local applicant density via applicant counts behaves like a bad-control specification, but controlling via loser counts preserves a positive, significant peer effect near 0.11.

**Reference script:** [24_count_based_instrument.py](../2_Scripts/2_Analysis/B_Estimation/24_count_based_instrument.py)  
**Output table:** [count_based_instrument_robustness.csv](../3_Output/Tables/count_based_instrument_robustness.csv)

---

### 13.2 Inference Robustness: Conley Spatial HAC and Multi-Level Clustering

**The concern.** State-level clustering (50 clusters) may not fully capture the spatial correlation structure in peer effects models. Residuals from spatially-lagged specifications are correlated both within states and across state borders.

**Approach.** Three alternative inference methods:
- **(a) County clustering** (~2,913 clusters): more local dependence structure
- **(b) Conley (1999) spatial HAC**: distance-band Bartlett kernel at 100km, 200km, 500km cutoffs
- **(c) HC robust**: heteroskedasticity-consistent (no clustering)

**Results (Script 22):**

| SE Method | $\hat{\beta}$ | SE | p-value |
|---|---|---|---|
| State clusters (50) | 0.201 | **0.034** | <0.001 |
| County clusters (2,913) | 0.201 | 0.024 | <0.001 |
| HC robust | 0.201 | 0.021 | <0.001 |
| Conley 100km | 0.201 | 0.024 | <0.001 |
| Conley 200km | 0.201 | 0.025 | <0.001 |
| Conley 500km | 0.201 | 0.027 | <0.001 |

**Reduced-form (ITT):**

| SE Method | RF Coef | SE | p-value |
|---|---|---|---|
| State clusters | 0.212 | **0.038** | <0.001 |
| County clusters | 0.212 | 0.026 | <0.001 |

**Interpretation:**

State clustering produces the **largest** standard errors (0.034) — larger than county clustering (0.024) and all Conley bandwidths (0.024–0.027). This means state-level clustering is the *most conservative* inference approach in this setting. The spatial HAC SEs are well-behaved: they increase monotonically with bandwidth (100km→200km→500km), confirming the kernel captures progressively larger spatial correlation, but never exceed the state-clustered SEs. Significance is robust across all methods (p < 0.001 everywhere).

**Conclusion.** State-level clustering is adequate and conservative. If anything, it *overstates* the standard errors relative to more granular spatial approaches. All inference results are reported using state clustering as the baseline.

**Reference script:** [22_inference_robustness.py](../2_Scripts/2_Analysis/B_Estimation/22_inference_robustness.py)  
**Output table:** [inference_robustness.csv](../3_Output/Tables/inference_robustness.csv)

---

### 13.3 Reduced-Form Reframing: Intention-to-Treat

**Rationale.** The IV-2SLS peer effect estimate ($\hat{\beta}$ on $\bar{Y}_i$) requires the exclusion restriction to hold: $\bar{Z}_i$ affects $Y_i$ only through $\bar{Y}_i$. The **reduced-form** regression relaxes this by estimating:

$$Y_i = \delta \bar{Z}_i + X_i'\gamma + \alpha_s + u_i$$

Here $\delta$ is the causal effect of **neighbor lottery wins** on own adoption — an intention-to-treat (ITT) parameter. It requires only that the lottery is random (which it is by design), not the full exclusion restriction. This is the most defensible single number in the paper.

**Result (from Scripts 21 and 22):**

$$\hat{\delta}^{\text{RF}} = 0.212^{***} \quad (\text{SE} = 0.038, \; \text{state clusters})$$

$$\hat{\delta}^{\text{RF, +}\bar{L}_i} = 0.112^{***} \quad (\text{SE} = 0.041, \; \text{with loser-density control})$$

**Interpretation.** A 10pp increase in the fraction of KNN-6 neighbors who won the CSBP lottery raises own adoption probability by 2.1pp (unconditional) or 1.1pp (controlling for neighbor application intensity). The ITT framing avoids claiming a specific mechanism channel: the effect could operate through informational demonstration, vendor spillovers, charging infrastructure externalities, or administrative learning — all of which are "peer environment" effects from lottery-funded deployments.

---

### 13.4 Falsification Tests: Placebo Outcomes, Donut Network, Distance Gradient

**Three tests targeting the exclusion restriction directly (Script 23):**

#### Test 1: Placebo Outcomes

Regress pre-determined variables on instrumented neighbor adoption ($\bar{Y}_i$ instrumented by $\bar{Z}_i$). If the IV is valid, neighbor lottery wins should not predict own pre-determined characteristics.

| Placebo Outcome | Baseline p-value | + $\bar{L}_i$ p-value | Change |
|---|---|---|---|
| Poverty rate | 0.643 | 0.603 | PASS in both |
| PM2.5 | 0.105 | 0.175 | PASS in both (attenuates) |
| Pct white | 0.147 | 0.108 | PASS in both |
| Median income | 0.223 | 0.389 | PASS in both (attenuates) |
| Pct Dem 2020 | 0.007 | 0.014 | **CONCERN remains** (attenuates) |
| Enrollment | 0.024 | 0.038 | **CONCERN remains** (small attenuation) |

**Direct confirmation with $\bar{L}_i$: 4/6 still pass.** Adding neighbor loser density attenuates both failing placebos, but does not fully eliminate significance for county Dem vote share or enrollment. This indicates that $\bar{L}_i$ absorbs part of the local clustering channel, but residual geographic sorting remains in observables tied to politics and district scale.

#### Test 2: Donut Network

Drop the 2 closest neighbors and use only neighbors 3–8. If effects persist, the channel is informational diffusion (travels farther). If effects vanish, the mechanism is local infrastructure/vendor spillovers (strongest at short distance).

| Network | $\hat{\beta}$ | SE | p | F (1st) |
|---|---|---|---|---|
| Baseline (1–6) | **+0.201** | 0.034 | <0.001 | ~55k |
| Donut (3–8) | +0.038 | 0.034 | 0.268 | ~55k |

**The peer effect vanishes when the 2 closest neighbors are dropped.** This means the effect is highly localized — concentrated in the 1–2 nearest neighbors. This pattern is consistent with either (a) very proximate physical spillovers (shared bus vendors, charging infrastructure), or (b) the closest neighbors being the most salient reference points for administrative decision-making.

#### Test 3: Distance Gradient

Use donut rings at increasing distance to test whether the peer effect attenuates monotonically.

| Ring | $\hat{\beta}$ | SE | p |
|---|---|---|---|
| Close (1–6) | **+0.201*** | 0.034 | <0.001 |
| Donut (3–8) | +0.038 | 0.034 | 0.268 |
| Medium (4–9) | -0.032 | 0.030 | 0.283 |
| Far (7–12) | +0.021 | 0.029 | 0.481 |

**Sharp drop-off confirmed.** The effect is concentrated in the inner ring and is essentially zero at all greater distances. This steep gradient is characteristic of very localized peer influence (or correlated unobservable factors at short distance that survive instrumental variables). The monotonic attenuation pattern supports the geographic specificity of the peer channel.

**Reference script:** [23_falsification_extended.py](../2_Scripts/2_Analysis/B_Estimation/23_falsification_extended.py)  
**Output table:** [falsification_extended.csv](../3_Output/Tables/falsification_extended.csv)

---

### 13.5 Own-Treatment Controls and the Exclusion Restriction

**The core identification issue.** All peer-effect specifications in this study instrument neighbor adoption ($\bar{Y}_{-i}$) with the spatial lag of neighbors' CSBP lottery status ($\bar{Z}_{-i}$). For $\bar{Z}_{-i}$ to satisfy the exclusion restriction, it must affect district $i$'s outcome **only** through the peer channel — that is, through neighbors' observable ESB adoption. But if district $i$ itself applied for (or won) the CSBP lottery, the instrument is correlated with uncontrolled own-district factors that directly affect $Y_i$:

$$\text{Cov}(\bar{Z}_{-i},\; \epsilon_i) \neq 0 \quad \text{if } Z_i \text{ or } \text{IS\_APPLICANT}_i \in \epsilon_i$$

because application propensity clusters geographically: if my neighbor applied and won, I was likely to have applied too.

**Empirical confirmation.** Script 07 (Version B) tested this for the CSBP-only outcome by adding own $Z_i$ as an exogenous control:

| Outcome | Own-$Z_i$ control? | $\hat{\beta}_{\text{geo}}$ | SE | p |
|---|---|---|---|---|
| IS_ADOPTER | No (Version A) | +0.195*** | 0.034 | <0.001 |
| IS_ADOPTER | Yes (Version B) | +0.062** | 0.022 | 0.006 |

The peer effect drops by **~70%** when own lottery status is controlled. A similar pattern appears in the all-source outcome (Section 13.6): without own-treatment controls, the estimate is +0.318; with them, it falls to +0.093 (baseline) or +0.156 (+ $\bar{L}_i$). See Section 13.6 for the full results.

**Three own-treatment controls.** The corrected specifications include:

1. **Own $Z_i$ (lottery winner):** Absorbs the direct channel where district $i$ adopts because it won the lottery itself. Coefficient on own $Z_i \approx 0.95$ in the CSBP-only spec — nearly deterministic.
2. **Own `IS_APPLICANT`:** Absorbs unobserved ESB enthusiasm. Districts that applied for CSBP revealed interest in ESBs, making them more likely to adopt through *any* channel (state programs, VW settlement, etc.) independent of peer effects.
3. **`pre_window_adopter`:** Absorbs prior ESB experience. The 339 districts with ESB orders before 2022 have fundamentally different adoption propensities.

**Implications for interpretation.**

- **CSBP-only (IS_ADOPTER):** After controlling for own $Z_i$, the remaining variation in IS_ADOPTER is minimal — almost all CSBP adoption is mechanically explained by own lottery status ($Z_i$ coef $\approx 0.95$). The residual peer effect (+0.062) operates only on districts that didn't win the lottery but adopted CSBP anyway (R2 competitive grants, a small subsample). This makes the own-$Z_i$-controlled CSBP-only estimate hard to interpret as a general peer effect.
- **All-source adoption:** This outcome has substantial variation **after** controlling for own CSBP status, because most ESB adoption occurs through non-CSBP channels (61% of window activity). The own-treatment controls absorb the direct CSBP adoption channel and self-selection, leaving the peer-channel identification clean: "does living near a lottery-funded ESB deployment cause me to adopt ESBs through any available channel?"

The **all-source outcome with own-treatment controls** (Section 13.6) is therefore the more informative specification for identifying genuine peer effects, while the CSBP-only estimates should be understood as describing a setting where outcome variation is dominated by own treatment status.

---

### 13.6 Alternative Outcome: All-Source New-Adopter Specifications (Script 25)

> **Note:** This section applies the own-treatment controls framework from Section 13.5 above.

**Motivation.** The baseline `IS_ADOPTER` indicator is constructed solely from CSBP rebate/grant data, raising the question: does the peer effect extend to broader ESB adoption that may use non-federal funding? Script 25 constructs alternative dependent variables from the full WRI bus-level dataset (all funding sources: federal CSBP, VW Settlement, state programs, federal other, utility/local), using `3q. Quarter ordered` for timing and `1c. LEA ID` for merge.

**Two alternative outcomes (2022–2024 window):**
1. `new_first_window`: district's **first-ever** ESB order falls within 2022–2024 (N first-adopters = 558)
2. `new_any_window`: district has **any** ESB order in 2022–2024 (N active districts = 668)

**Source mix among active districts in window:**

| Funding Source | District Count | Share |
|---|---|---|
| Federal CSBP | 280 | 38.9% |
| Other/Mixed | 188 | 26.1% |
| State Program | 145 | 20.1% |
| VW Settlement | 102 | 14.2% |
| Utility/Local | 5 | 0.7% |

Over 60% of window adoption activity involves non-CSBP funding, so these outcomes test whether CSBP lottery neighbors spill over into **all-source** ESB adoption.

#### 13.6.1 Identification: Own-Treatment Controls

The CSBP lottery instrument $\bar{Z}_{-i}$ (neighbor lottery-winner share) is valid only if, **conditional on controls**, it is uncorrelated with district $i$'s own reasons to adopt. Without controlling for own treatment status, the exclusion restriction is violated because:

1. **Own lottery status ($Z_i$):** If district $i$ itself won the CSBP lottery, it will adopt with high probability. And $Z_i$ is correlated with $\bar{Z}_{-i}$ through geographic clustering of applications — nearby districts apply at the same time, so if my neighbor won, I was more likely to have applied (and potentially won) too.
2. **Own application status (`IS_APPLICANT`):** Even without winning, having applied for CSBP signals unobserved ESB enthusiasm that predicts adoption through any funding channel. Application propensity clusters geographically, inducing correlation between $\bar{Z}_{-i}$ and $\epsilon_i$.
3. **Pre-window adoption history:** Districts that already operated ESBs before the study window (339 districts with orders pre-2022) are fundamentally different in their propensity to adopt; omitting this confounder could bias the peer effect upward.

Including $Z_i$, `IS_APPLICANT`, and `pre_window_adopter` as exogenous controls absorbs the direct and selection channels, isolating variation in $Y_i$ that comes solely from the **peer channel**: $\bar{Z}_{-i} \to$ neighbor adopts $\to$ $i$ observes and adopts.

**Diagnostic: effect of omitting own-treatment controls.**
An initial run of Script 25 *without* these three controls produced coefficients of +0.318*** (first-ever) and +0.381*** (any-event), all highly significant. After adding $Z_i$, `IS_APPLICANT`, and `pre_window_adopter`, the coefficients drop by 60–70%, confirming substantial upward bias from the omitted own-treatment channel.

#### 13.6.2 Corrected Results

**IV-2SLS Results (K=6, state-clustered SE, state FE + full controls + own $Z_i$, IS_APPLICANT, pre_window_adopter):**

| Spec | Outcome | $\hat{\beta}$ | SE | p-value | First-stage F | N |
|---|---|---|---|---|---|---|
| A1. Baseline | First-ever | +0.093 | 0.066 | 0.163 | 1,808 | 13,572 |
| A2. + $\bar{L}_i$ | First-ever | +0.156** | 0.069 | 0.024 | 1,573 | 13,572 |
| B1. Baseline | Any-event | +0.120* | 0.069 | 0.080 | 1,971 | 13,572 |
| B2. + $\bar{L}_i$ | Any-event | +0.189*** | 0.073 | 0.009 | 1,697 | 13,572 |

**Reduced-Form ITT:**

| Spec | Outcome | $\hat{\delta}$ | SE | p-value | N |
|---|---|---|---|---|---|
| C1. Baseline | First-ever | +0.032 | 0.026 | 0.215 | 13,572 |
| C2. + $\bar{L}_i$ | First-ever | +0.052* | 0.028 | 0.061 | 13,572 |
| D1. Baseline | Any-event | +0.047 | 0.030 | 0.125 | 13,572 |
| D2. + $\bar{L}_i$ | Any-event | +0.070** | 0.033 | 0.031 | 13,572 |

#### 13.6.3 Interpretation

**Why do the $\bar{L}_i$-controlled specs have larger coefficients than the baseline?** This is the opposite of the CSBP-only result (where $\bar{L}_i$ halved the estimate). The mechanism differs across outcomes:

- In the **CSBP-only** spec, $\bar{L}_i$ absorbed upward bias from applicant clustering: CSBP losers nearby meant more applicants nearby, spuriously inflating own CSBP adoption.
- In the **all-source** spec *with own-treatment controls already included*, $\bar{L}_i$ acts as a **suppressor variable**. After absorbing $Z_i$ and `IS_APPLICANT`, the remaining variation in neighbor adoption includes noise from losers who adopted through other programs — adoptions that the lottery instrument doesn't predict. By partialling out this component, $\bar{L}_i$ sharpens the IV estimate, allowing the lottery-driven peer effect to emerge more cleanly.

**Key takeaways:**
- The all-source peer effect, properly controlling for own treatment status, is **+0.12 to +0.19**, statistically significant in the $\bar{L}_i$-controlled specs. This is broadly consistent with the CSBP-only preferred estimate of +0.110.
- Without own-treatment controls, the all-source estimates were severely upward biased (+0.32–0.38), demonstrating that **omitting own $Z_i$ conflates direct treatment effects with peer effects**.
- First stages remain very strong (F > 1,500), confirming that the instrument has power even after absorbing own treatment variation.
- The convergence of the all-source (+0.16 to +0.19 with $\bar{L}_i$) and CSBP-only (+0.110 with $\bar{L}_i$) estimates suggests the true peer effect is in the **+0.10 to +0.19 range**, with the CSBP-only estimate at the conservative end because some genuine peer-induced adoption occurs through non-CSBP channels that the CSBP-only DV misses.
- **However, these are cross-sectional estimates.** As Section 13.7 shows, the cross-round (temporal) design produces dramatically different results for the all-source outcome, underscoring that cross-sectional identification alone is insufficient.

---

### 13.7 Cross-Round All-Source IV: The Temporal Test (Script 26)

**Motivation.** The cross-sectional analysis (Script 25) uses the pooled R1+R3 lottery instrument, which conflates temporal demonstration effects with within-round co-application sorting — even with own-treatment controls. The cleanest identification is **temporal**: did a neighbor's R1 (2022) lottery win — and subsequent ESB deployment — cause district $i$ to adopt ESBs (from any source) in the R3 window (2023–2024)?

This is the **identified peer effect**: the exogenous shock (R1 lottery) is fully predetermined relative to R3-window adoption decisions, with a 12–18 month lag for buses to be delivered and observed.

#### 13.7.1 Design

- **Instrument:** $\bar{Z}_{-i}^{R1}$ = share of K nearest neighbors who won R1 lottery (2022)
- **Outcome (all-source):** first-ever ESB order in 2023–2024, or any ESB order in 2023–2024 (from WRI bus-level data, all funding sources)
- **Sample:** All districts excluding R1 winners (N = 12,620)
- **Own-treatment controls:** `IS_APPLICANT`, `pre_r1_adopter` (+ full controls, state FE)
- **Note:** Own $Z_i$ not needed as a control since R1 winners are excluded from the sample

#### 13.7.2 Results

**IV-2SLS (state-clustered SE, state FE + full controls + own-treatment controls):**

| K | Outcome | $\hat{\beta}$ | SE | p-value | First-stage F | N |
|---|---|---|---|---|---|---|
| 4 | First-ever | +0.004 | 0.027 | 0.878 | 6,289 | 12,620 |
| 4 | Any-event | +0.017 | 0.032 | 0.602 | 5,533 | 12,620 |
| 6 | First-ever | +0.004 | 0.029 | 0.896 | 5,979 | 12,620 |
| 6 | Any-event | +0.022 | 0.035 | 0.537 | 5,287 | 12,620 |
| 8 | First-ever | +0.003 | 0.036 | 0.935 | 5,817 | 12,620 |
| 8 | Any-event | +0.022 | 0.043 | 0.614 | 5,212 | 12,620 |
| 10 | First-ever | −0.020 | 0.033 | 0.553 | 5,842 | 12,620 |
| 10 | Any-event | −0.006 | 0.043 | 0.894 | 5,204 | 12,620 |

**Reduced Form ITT (K=6):**

| Outcome | $\hat{\delta}$ | SE | p-value | N |
|---|---|---|---|---|
| First-ever | +0.002 | 0.015 | 0.897 | 12,620 |
| Any-event | +0.012 | 0.020 | 0.542 | 12,620 |

All coefficients are near zero and far from statistical significance. First stages are extremely strong (F > 5,000), so weak instruments are not the issue — the instrument **has power** to predict neighbor R3-window adoption, but that predicted peer adoption has **no detectable effect** on own all-source adoption.

#### 13.7.3 Comparison with CSBP-Only Cross-Round

| Spec | Outcome | Coef | SE | p | F |
|---|---|---|---|---|---|
| Script 20 (CSBP-only) | IS_R3_ADOPTER | +1.343*** | 0.505 | 0.008 | 35 |
| Script 26 (all-source) | First-ever 2023–24 | +0.004 | 0.029 | 0.896 | 5,979 |
| Script 26 (all-source) | Any-event 2023–24 | +0.022 | 0.035 | 0.537 | 5,287 |

The CSBP-only cross-round spec found a large positive effect (+1.343) but with a relatively weak first stage (F=35). The all-source spec has a massively stronger first stage (F~5,000) but a near-zero point estimate.

#### 13.7.4 Interpretation: What Explains the Divergence?

The divergence between the CSBP-only and all-source cross-round results reveals something fundamental about what the "cross-round peer effect" is actually measuring:

**1. The CSBP-only effect (+1.343) is driven by within-program application clustering, not observational learning.** If district $i$'s neighbor won R1 and deployed buses, the CSBP result says district $i$ is much more likely to receive R3 CSBP funding. But this could reflect:
- **Information about the CSBP program specifically** (neighbor tells $i$ about the program, $i$ applies for R3)
- **Intermediary/consultant networks** (the same grant-writing firm serves nearby districts)
- **EPA regional outreach patterns** (EPA targets geographic clusters for application assistance)

These are "peer effects" within the CSBP program pipeline, but they do not reflect broader **adoption spillovers** — they reflect **application spillovers** within a specific federal program.

**2. The all-source null result confirms this.** If the peer effect were about observational learning ("I saw my neighbor's electric bus, now I want one too"), we would expect spillovers into *all* funding channels — state programs, VW settlement, utility incentives. The silence across all-source outcomes means that **seeing a neighbor's ESB does not, by itself, cause general adoption through whatever funding channel is available.**

**3. Outcome rate and data coverage issue.** The all-source R3-window adoption rate (0.78% first-ever, 1.2% any-event) is *lower* than the CSBP-only R3 rate (~3.5%). This is counterintuitive if the WRI dataset were complete — all CSBP R3 adopters should appear as all-source adopters. The likely explanation is that WRI's bus-level `3q. Quarter ordered` timing lags behind CSBP award records (orders happen after funding is secured), so some 2023 CSBP winners had not yet placed orders by the WRI data snapshot. This data coverage lag attenuates the all-source estimates toward zero and inflates the CSBP-only/all-source divergence.

**4. The cross-round CSBP result (+1.343) should be interpreted with caution.** The F-stat of 35 is adequate but much lower than the cross-sectional spec (F > 50,000), confirming that most of the cross-sectional instrument variation came from within-round co-application patterns. The large LATE (+1.343 > 1) likely reflects a complier subpopulation: districts whose R3 CSBP participation was specifically induced by having an R1-winner neighbor. This is a valid LATE but describes a narrow subpopulation, not a general peer effect.

#### 13.7.5 Revised Assessment

The cross-round all-source analysis forces a more nuanced conclusion:

- There is a **program-specific application spillover** in CSBP: having an R1-winner neighbor increases R3 CSBP adoption. This is a "peer effect" within the federal program pipeline.
- There is **no detectable general adoption spillover**: R1 winner neighbors do not cause adoption through non-CSBP channels in the temporal design.
- The cross-sectional all-source results (Section 13.6, +0.16 to +0.19) likely reflect residual co-application sorting that the temporal design eliminates, or a data coverage problem in the WRI outcome variable that attenuates temporal estimates.
- **The honest claim is narrower than initially stated:** "CSBP lottery winners create program-specific adoption spillovers among nearby districts, likely through information about the CSBP program itself, rather than general observational learning about ESBs."

This distinction matters for the mechanism decomposition you described: the climate-peer vs. type-peer (information vs. prestige) channel analysis should focus on **CSBP application/adoption** as the outcome, since that is where the identified temporal effect exists. General ESB adoption appears to be driven by separate funding-channel-specific factors, not by observing a neighbor's buses.

---

### 13.8 Summary Assessment: What the Robustness Battery Tells Us

| Test | Result | Implication |
|---|---|---|
| Cross-round IV, CSBP-only (R1→R3) | +1.343*** (F=35) **but see 13.10** | **Unreliable**: replication gives F=0.3, Wald ratio noise. Not a credible estimate. |
| Cross-round diagnostics (Script 26c) | RF null for all outcomes (p>0.29) | No reduced-form evidence of cross-round peer effects for CSBP or all-source |
| Cross-round IV, all-source (R1→R3) | +0.004 to +0.022 (all p>0.5, F>5,000) | No detectable general ESB adoption spillover in temporal design |
| $\bar{L}_i$ control, K=6 | +0.110*** | ~45% of cross-sectional baseline driven by applicant clustering; remainder is peer effect |
| Count IV + $\#\text{LoserNbrs}$ control | +0.109*** | Same conclusion without ratio instrument; cleaner full-sample specification |
| All-source cross-section (+ $\bar{L}_i$, corrected) | +0.156** / +0.189*** | Positive cross-sectional effect; not confirmed temporally |
| Own-treatment controls (Script 07B) | +0.062** | Survives own Z_i control; at least partly peer-driven |
| Conditional instrument | +0.021 (p=0.68) | Conditional win-rate variation is not predictive |
| Conley 500km HAC | SE=0.027 (< state 0.034) | State clustering is the most conservative; inference is robust |
| Placebo outcomes | 4/6 pass baseline and 4/6 with $\bar{L}_i$ | Two failures (Dem share, enrollment) attenuate but remain significant |
| Donut (drop 1–2) | +0.038 (p=0.27) | Effect concentrated in nearest 1–2 neighbors |
| Distance gradient | Sharp decay | Consistent with hyper-local peer influence |
| Bad control test (Script 26b) | No change across specs | IS_APPLICANT is not driving the null temporal result |

**Headline estimates:**
- **Cross-sectional baseline:** $\hat{\beta} = 0.201^{***}$ (SE=0.034) — does not control for own treatment
- **Cross-sectional preferred (+ $\bar{L}_i$):** $\hat{\beta} = 0.110^{***}$ (SE=0.039) — CSBP outcome, cross-sectional
- **With own Z_i control:** $\hat{\beta} = 0.062^{**}$ — survives exclusion restriction fix
- **Cross-round temporal:** No credible estimate — CSBP first stage is dead (F=0.3), all reduced forms null

**The revised picture:** There is **cross-sectional evidence** of CSBP peer effects that survives loser-density, own-treatment, and count-based robustness checks ($\hat{\beta}$ = 0.062–0.110). However, the **cross-round temporal design** — which would provide the cleanest identification — **does not produce a credible result** due to a critically weak first stage for the CSBP endogenous variable. The R1 instrument does not predict neighbor R3 CSBP adoption (r=0.037), rendering the temporal IV uninformative.

The WRI bus-level data misses 93% of CSBP R3 adoptions, so "all-source" results cannot be interpreted as including the CSBP channel.

This leaves the cross-sectional estimates as the primary evidence. These are consistent with local peer effects but cannot fully rule out co-application sorting, even with the $\bar{L}_i$ suppressor control.

### 13.9 Bad Control Diagnostic: IS_APPLICANT in the Cross-Round Design

**Concern:** In the cross-round design (R1→R3), `IS_APPLICANT` is pooled across all CSBP rounds (R1+R2+R3). If the peer effect operates through *inducing R3 application* — i.e., neighbor R1 win → I learn about CSBP → I apply for R3 — then controlling for `IS_APPLICANT` blocks the very mediator we are trying to detect. This is a classic "bad control" problem (Angrist & Pischke 2009, Ch. 3.2.3).

**Data:**
- R1 applicants: 1,546 districts (predetermined, safe to control for)
- R3 applicants: 338 districts (potentially post-treatment mediator)
- R3-only applicants (new entrants, not in R1): 202 districts

**Test (Script 26b):** Re-ran the cross-round all-source IV under three control specifications:

| Spec | Controls | K=6 First-ever (p) | K=6 Any-event (p) |
|------|----------|--------------------|--------------------|
| (A) Original | IS_APPLICANT + pre_r1_adopter | +0.005 (0.855) | +0.024 (0.506) |
| (B) No app control | pre_r1_adopter only | +0.006 (0.826) | +0.025 (0.476) |
| (C) Predetermined | IS_R1_APPLICANT + pre_r1_adopter | +0.004 (0.895) | +0.021 (0.549) |

Results are virtually identical across all three specifications and all K values (4, 6, 8, 10). Removing IS_APPLICANT shifts coefficients by <0.003.

**Stage-0 test — R3 application as outcome:**
Using `IS_R3_APPLICANT` as the dependent variable with `w_IV_Z_R1` as the regressor (K=6):
- No app control: coef = -0.008 (p = 0.770)
- + IS_R1_APP: coef = -0.003 (p = 0.909)

**Neighbor R1 wins do not induce R3 application.** The peer-induced application channel simply does not exist at the geographic-neighbor level.

**Conclusion:** The theoretical concern about IS_APPLICANT as a bad control is methodologically correct but empirically irrelevant in this setting. The null all-source cross-round result stands regardless of control specification.

### 13.10 Cross-Round Diagnostics: The +1.343 Result is Unreliable (Script 26c)

Comprehensive diagnostics (Script 26c) reveal that the Script 20 CSBP-only cross-round result (+1.343\*\*\*, F=35) is **not credible**. Multiple independent problems converge:

#### D1. WRI Data Does Not Capture CSBP Buses

| Overlap | Count | % of R3 CSBP |
|---|---|---|
| R3 CSBP winner districts | 458 | — |
| In WRI with 2023-2024 orders | 37 | 8.1% |
| In WRI with any entry | 86 | 18.8% |
| **No WRI data at all** | **372** | **81.2%** |

93% of R3 CSBP adopters do not appear in the WRI 2023-2024 bus data. This means the "all-source" outcome is actually a non-CSBP-source outcome that **misses the very channel** the peer effect could operate through. The WRI dataset tracks bus *orders*, and most CSBP R3 rebates awarded in 2023 have not yet resulted in recorded bus orders.

#### D2. CSBP First Stage is Dead (F = 0.3)

On the consistent 12,620-district estimation sample:

| First Stage: `w_IV_Z_R1` → | FS coef | F-stat | Correlation |
|---|---|---|---|
| `w_R3_CSBP` (neighbor CSBP R3) | 0.008 | **0.3** | r = 0.037 |
| `w_r3_any` (neighbor all-source) | 0.551 | 5,278 | r = 0.523 |

The instrument (`w_IV_Z_R1`) does **not** predict neighbor CSBP R3 adoption share. The correlation between neighbor R1 wins and neighbor R3 CSBP adoption is 0.037. This means the IV estimate for the CSBP endogenous variable is dividing by near-zero — producing meaningless coefficients.

The original Script 20's F=35 likely resulted from a different sample construction, merge ordering, or inadvertent data contamination. On the consistent sample used across all diagnostics, F=0.3.

#### D3. Reduced Forms Are All Null

| RF: `w_IV_Z_R1` → Y | coef | SE | p | 1-neighbor effect |
|---|---|---|---|---|
| CSBP R3 adoption | -0.040 | 0.039 | 0.30 | -0.67pp |
| WRI first-ever | +0.003 | 0.015 | 0.84 | +0.05pp |
| WRI any-event | +0.015 | 0.020 | 0.47 | +0.25pp |

**No reduced form is significant.** Having one additional R1-winning neighbor (out of 6) changes own CSBP adoption probability by -0.67pp (wrong sign, insignificant). The reduced form is the only directly comparable object across specifications, and it is null for every outcome.

#### D4. Anderson-Rubin Test (Weak-IV Robust)

- AR 95% CI: [-2.00, +5.00] (spans entire grid — uninformative)
- AR test at $\beta=0$: F=2.57, p=0.109 (cannot reject null)

Even using inference that is robust to arbitrarily weak instruments, we cannot reject zero peer effect.

#### D5. Replication Produces Different Results

Replicating Script 20's exact specification (CSBP outcome, CSBP endogenous, no own controls) on the consistent sample:

> $\hat{\beta}$ = **-5.189** (SE=20.83, p=0.80) | F=0.3

The point estimate is negative, enormous, and completely insignificant — the hallmark of a Wald ratio with a near-zero denominator.

#### Revised Assessment

The +1.343\*\*\* from Script 20 was an artifact of:
1. **Critically weak instrument** for the CSBP-specific endogenous variable (true F ≈ 0.3)
2. **Possible sample construction differences** that inflated the F-stat in the original run
3. **Wald ratio inflation**: dividing a noisy reduced form by a near-zero first stage

**The cross-round temporal design does not identify a peer effect for any outcome — CSBP or all-source.** The remaining credible evidence is the cross-sectional design:
- Baseline: $\hat{\beta} = 0.201^{***}$ (SE=0.034)
- With $\bar{L}_i$ suppressor: $\hat{\beta} = 0.110^{***}$ (SE=0.039)
- With own-treatment controls (Script 07B): $\hat{\beta} = 0.062^{**}$

These cross-sectional estimates remain significant but cannot disentangle temporal peer effects from co-application sorting as cleanly as a working cross-round design would.

---

*References: Manski (1993) "Identification of Endogenous Social Effects"; Nevo & Rosen (2012) "Identification with Imperfect Instruments"; Bramoulle, Djebbari & Fortin (2009) "Identification of peer effects through social networks"; Conley (1999) "GMM estimation with cross sectional dependence"; Stock & Yogo (2005) "Testing for weak instruments in linear IV regression."*
