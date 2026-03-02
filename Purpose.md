# Purpose of the Post-Temporal Branch

**Created:** March 2, 2026  
**Branched from:** `Current` (commit `6c9d823`)  
**Previous branch:** `Current` (scripts 01–30 + full documentation)

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

**Estimand 1: Intention-to-treat (ITT) — the most defensible number**

Does having a lottery-winning neighbor *in the same pool of applicants* increase own adoption probability? This requires only lottery randomness (which holds unconditionally), not the full exclusion restriction. It does not claim to identify a structural peer effect; it estimates the causal effect of *neighbor lottery wins* on own adoption:

$$\delta = \frac{\partial \Pr(Y_i = 1)}{\partial \bar{Z}_i}$$

The estimate with the loser-density control (absorbing local application clustering) is δ ≈ 0.11. This is interpretable as: *an additional lottery-winning neighbor in a pool of 6 raises own adoption probability by ~1.8pp, conditional on local application interest.* This is the ITT effect of the lottery assignment on diffusion.

**Limitation:** This conflates several mechanisms — informational demonstration, vendor referrals, administrative learning, shared budget — all of which are "effects of having a funded neighbor nearby." It does not identify which channel dominates.

---

**Estimand 2: Application extensive margin (requires round-specific applicant data)**

If the application roster can be split by round (R1 applicants, R3 applicants — separate lists), then the outcome can be reframed as:

$$Y_i^{R3} = \mathbf{1}[\text{district } i \text{ applied to R3}]$$

instrumented by `w_IV_Z_R1` (share of R1 lottery winners among K nearest neighbors). This directly tests whether R1 neighbor deployments caused R3 application — the extensive margin of program interest. It sidesteps the weak first-stage problem because we no longer require R1 winners to also re-apply in R3; we only require that R1 neighbor wins predict R3 *interest* (application), which is a weaker and more plausible condition.

**Data requirement:** Clean separation of R3 applicants from R1 applicants. The existing waitlisted/rejected file (`CSBP Applicants waitlisted and rejected.xlsx`) contains applicants without explicit round separation — this needs to be verified.

---

**Estimand 3: Longer-horizon all-source diffusion**

The R1 → R3 window (~15 months, Oct 2022 → Jan 2024) may be too short for deployment-and-observation to occur. A longer window — R1 wins (2022) → all-source WRI adoption 2024–2025 — allows more time for buses to be delivered (delivery data in WRI field `3r. Quarter delivered`), operate, and influence neighbor decisions.

**Data requirement:** Extend the WRI adoption window beyond the current 2023–2024 cutoff. The WRI v9 dataset (June 2025) should contain 2025 entries. This requires re-extracting the WRI bus-level data without the current year filter.

---

**Estimand 4: Heterogeneity by deployment visibility**

Among R1 winners, buses delivered *before* the R3 deadline (early delivery) should generate stronger peer effects than those delivered after (late delivery). This is a more targeted test of the demonstration mechanism. Script 28 explored this but found null reduced-form effects even for early-delivery neighbors (RF coef ≈ +0.009, p = 0.80 for WRI first-ever R3 adoption). However, this used the CSBP-specific outcome; re-running with all-source adoption and a longer window is a natural extension.

---

**What is likely not identifiable with current data**

- **Structural peer effect β** (the social multiplier): requires temporal ordering, a valid exclusion restriction, and a first stage that isn't near-tautological. None of these are cleanly available.
- **Mechanism decomposition** (information vs. vendor vs. infrastructure): cross-sectional data cannot separate these channels. The donut test (effect concentrates in 1–2 nearest neighbors) is suggestive of very local spillovers but cannot distinguish mechanisms.
- **Dynamic learning / Bayesian updating**: requires at least two pre-adoption time periods per district. The current data is effectively cross-sectional.

---

## What Gets Built in This Branch

The goal of this branch is:

1. **Clarify the research question and mechanism.** Write down precisely what peer effect is being claimed (informational demonstration, application learning, social pressure?), what the ideal data would look like to identify it, and how far the existing data can go.

2. **Define clean variable construction from scratch.** Decide on the outcome, the endogenous variable, the instrument, and the sample *before* writing any estimation code. Apply the right temporal structure from the start.

3. **Scope a feasible paper.** The most defensible result is the ITT (δ ≈ 0.11 after loser control). The paper can be framed around the ITT as the headline, with the structural β as an additional result that requires the exclusion restriction.

4. **Potentially extend the WRI time window** to capture longer-horizon diffusion (2024–2025 all-source adoption), which is the most natural fix for the timing problem.

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

*This document was written on March 2, 2026 after a full diagnostic review of the identification strategy. The branch is a clean slate.*
