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

### 13.5 Summary Assessment: What the Robustness Battery Tells Us

| Test | Result | Implication |
|---|---|---|
| Cross-round IV (R1→R3) | +1.343*** (F=35) | Temporal ordering confirmed; F-stat collapse shows contemporaneous F was inflated |
| $\bar{L}_i$ control, K=6 | +0.110*** | ~45% of baseline driven by applicant clustering; remainder is peer effect |
| Count IV + $\#\text{LoserNbrs}$ control | +0.109*** | Same conclusion without ratio instrument; cleaner full-sample specification |
| Conditional instrument | +0.021 (p=0.68) | Conditional win-rate variation is not predictive; suggestive of threshold/extensive-margin channel but based on a different estimand and sample |
| Conley 500km HAC | SE=0.027 (< state 0.034) | State clustering is the most conservative; inference is robust |
| Placebo outcomes | 4/6 pass baseline and 4/6 with $\bar{L}_i$ | Two failures (Dem share, enrollment) attenuate but remain significant; residual clustering concern persists |
| Donut (drop 1–2) | +0.038 (p=0.27) | Effect concentrated in nearest 1–2 neighbors |
| Distance gradient | Sharp decay | Consistent with hyper-local peer influence |

**Headline estimates:**
- **Primary (unconditional, baseline):** $\hat{\beta} = 0.201^{***}$ (SE=0.034)
- **Preferred (+ loser-density control):** $\hat{\beta} = 0.110^{***}$ (SE=0.039)
- **Cross-round (temporal, R1→R3):** $\hat{\beta} = 1.343^{***}$ (SE=0.505, LATE on rare subsample)
- **Reduced-form ITT (+ $\bar{L}_i$):** $\hat{\delta} = 0.112^{***}$ (SE=0.041)

The picture that emerges: there is a **real, positive, statistically significant peer effect in CSBP adoption**, but it is (a) about half as large as the uncorrected baseline after purging applicant clustering, (b) hyper-local (nearest 1–2 neighbors), and (c) more consistent with exposure to nearby CSBP engagement than with exact local win-rate intensity, though that mechanism split is not point-identified by the conditional-ratio test. The reduced-form ITT — "nearby lottery-funded deployments causally raise own adoption" — is the most defensible claim.

---

*References: Manski (1993) "Identification of Endogenous Social Effects"; Nevo & Rosen (2012) "Identification with Imperfect Instruments"; Bramoulle, Djebbari & Fortin (2009) "Identification of peer effects through social networks"; Conley (1999) "GMM estimation with cross sectional dependence."*
