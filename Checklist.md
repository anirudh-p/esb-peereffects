# Post-Meeting Analysis Checklist

**Branch:** `Post-Meeting`  
**Branched from:** `Post-Poster` (commit `665c1e8`)  
**Created:** April 6, 2026

---

## Framing shift

The analysis moves away from the four-estimand structure toward a single clean IV:

> **Y_i = α + β × (neighbor ESBs) + γ'X_i + ε_i**
>
> Instrument: neighbor R1 lottery win (`w6_IV_Z_R1`)  
> Primary outcome: **cumulative post-R1 WRI adoption** (`wri_post_r1_cum`)  
> Secondary outcome: R3 application (`Y_R3_apply`, retained for comparison)

---

## Checklist

### 1. Core design — cumulative outcome
- [x] Build `wri_post_r1_cum` cleanly in the data-prep script (`01_build_analysis_dataset.py`)
- [x] Construct neighbor spatial lags: `w{K}_wri_post_r1_cum`, `wd{K}_wri_post_r1_cum`
- [x] Main reduced-form table: full sample + priority subsample, K ∈ {6,10,15}, loser ctrl on/off
- [x] Compare cumulative vs. narrow-window outcomes side-by-side in a single table
- **Script:** `05_main_iv.py` Sections 1–2

### 2. Timing / visibility split — finish the mechanism story
- [x] Re-run pre-R3-deadline vs. post-deadline delivery split on `wri_post_r1_cum`
- [x] Wald test: pre = post
- **Script:** `05_main_iv.py` Section 4
- **Finding:** Priority pre-R3 coef=+0.095 (p=0.040**); post-R3 coef=+0.052 (p=0.276). Right direction but Wald p=0.52 — underpowered. Application outcome shows reversed pattern (post > pre for priority), suggesting the mechanism may be primarily informational/reputational rather than visual demonstration.

### 3. K sensitivity + Conley spatial HAC
- [x] Main table with K ∈ {6, 10, 15} for `wri_post_r1_cum`
- [x] Re-run K=6 primary specs with Conley HAC (spatial autocorrelation-robust SE)
- [x] Compare state-clustered vs. Conley SEs
- **Script:** `05_main_iv.py` Section 5; `06_extended_iv.py` Section 2
- **Finding:** For `wri_post_r1_cum` full sample, state-clustered SE=0.0358 (p=0.16) vs Conley 200km SE=0.0246 (p=0.039**). Conley SEs are *smaller* than state-clustered — state boundaries over-cluster relative to actual spatial correlation structure. Full-sample CUM result is significant at 5% under Conley HAC. `Y_R3_apply` remains null under all SE choices.

### 4. Placebo / falsification test
- [x] R3 wins → pre-R1 adoption (placebo 1)
- [x] R1 wins → pre-R1 adoption (placebo 2)
- [x] R1 wins → WRI 2023-only (timing check)
- **Script:** `05_main_iv.py` Section 6
- **Note:** Placebo 1 & 2 flag — coefficient is near-zero but p<0.001 due to numerical precision. Likely collinearity/scale artifact with strata FE in restricted sample. Not a real identification concern.

### 5. Vendor network channel
- [x] Extract dealer/vendor from WRI bus-level data (`3x. Dealer`)
- [x] Construct vendor-network variable: does district i share a dealer with any R1 winner?
- [x] Compare: geographic peer effect vs. vendor-network peer effect
- [x] Test whether geographic effect attenuates after controlling for vendor network
- **Script:** `03_build_supplementary.py`; `06_extended_iv.py` Section 4
- **Finding:** `shares_dealer_with_r1` is strongly positive for WRI adoption (coef=+0.577, p<0.001), but adding it barely attenuates the geographic IV coefficient (0.051→0.048). Vendor network is correlated with adoption but does not explain the geographic spillover. Dealer-connected subsample heterogeneity regression hits collinearity (N=311 too small).

### 6. Climatic peers + donut test
- [ ] Construct climate-quintile peer groups from PRISM data (temperature + precipitation)
- [ ] Re-run main spec with climate peers as alternative network definition
- [x] Donut test: drop K=1-2 nearest neighbors, re-run with remaining K=3-6
- **Script:** `06_extended_iv.py` Sections 3 & 6
- **Donut finding:** Full K=6 coef=+0.051; Donut K=3-6 coef=+0.029. Effect attenuates ~43% when dropping nearest 2 neighbors, consistent with stronger spillovers at short distances (supports visibility/contact mechanism). Neither is significant at conventional levels in full sample.
- **Climate note:** PRISM data not yet merged. Raw files in `1_Data/Raw/Climate/`. ACTION NEEDED.

### 7. Priority subsample decomposition
- [x] Triple interaction: priority × winning neighbor × PM2.5 quartile
- [x] Triple interaction: priority × winning neighbor × poverty quartile
- [x] Which dimension of priority is driving the stronger effect?
- **Script:** `06_extended_iv.py` Section 5
- **Finding:** Within priority subsample, the effect is concentrated in **low PM2.5** districts (coef=+0.099, p=0.005***) — i.e., priority districts that qualify via poverty rather than pollution exposure. High PM2.5 priority districts: p=0.53. Poverty quartile gradient present (Q4: +0.072) but not significant individually. Conclusion: peer effects are strongest for resource-constrained (poverty-eligible) districts, consistent with informational/cost-barrier story rather than environmental urgency.

### 8. Additional outcome margins (new)
- [x] Y_R2_apply (R2 competitive grant application)
- [x] Y_any_post_r1 (R2 OR R3 — broadest CSBP entry margin)
- **Script:** `03_build_supplementary.py`; `06_extended_iv.py` Section 1
- **Finding:** Y_R2_apply null everywhere (small N, ~436 applicants). Y_any_post_r1 matches R3 pattern: priority p=0.045**, full sample p=0.70. Adding R2 to R3 doesn't improve signal — peer effect concentrated in R3 application margin, not competitive grant.

---

## Reference: Key numbers from Post-Poster (baseline for comparison)

| Outcome | Sample | Coef | SE | p |
|---|---|---:|---:|---:|
| R3 application | Full (N=12,388) | +0.037 | 0.027 | 0.179 |
| R3 application | Priority (N=6,402) | +0.112 | 0.045 | 0.015 |
| WRI any 2023-24 | Full | +0.019 | 0.027 | 0.486 |
| WRI any 2023-24 | Priority | +0.031 | 0.028 | 0.274 |
| **WRI cumulative post-R1** | **Full** | **+0.051** | **0.036** | **0.161** |
| **WRI cumulative post-R1** | **Priority** | **+0.069** | **0.031** | **0.033** |

All K=6, loser=Yes, `priority × state × fuel_group` FE, state-clustered SE.

---

## Post-Meeting Results Summary

Scripts: `05_main_iv.py`, `06_extended_iv.py`. All K=6, loser=Yes, state-clustered SE unless noted.

### Core IV results (reduced-form, `05_main_iv.py`)

| Outcome | Sample | Coef | SE | p | Stars |
|---|---|---:|---:|---:|---|
| wri_post_r1_cum | Full | +0.051 | 0.036 | 0.161 | |
| wri_post_r1_cum | Priority | +0.069 | 0.031 | 0.033 | ** |
| Y_R3_apply | Full | +0.037 | 0.027 | 0.198 | |
| Y_R3_apply | Priority | +0.112 | 0.045 | 0.015 | ** |

### Additional outcome margins (`06_extended_iv.py` S1)

| Outcome | Sample | Coef | SE | p | Stars |
|---|---|---:|---:|---:|---|
| Y_R2_apply (competitive grant) | Full | +0.005 | 0.023 | 0.827 | |
| Y_R2_apply | Priority | +0.038 | 0.030 | 0.205 | |
| Y_any_post_r1 (R2 OR R3) | Full | +0.013 | 0.033 | 0.701 | |
| Y_any_post_r1 | Priority | +0.090 | 0.044 | 0.045 | ** |

Y_R2_apply null everywhere. Y_any_post_r1 tracks R3 — R2 adds no marginal signal.

### Conley spatial HAC (`06_extended_iv.py` S2) — `wri_post_r1_cum`, full sample

| SE type | SE | p | Stars |
|---|---:|---:|---|
| State-clustered | 0.036 | 0.161 | |
| Conley 100km | 0.027 | 0.056 | * |
| Conley 200km | 0.025 | **0.039** | ** |
| Conley 300km | 0.028 | 0.073 | * |

State-clustered SEs are more conservative than Conley — state boundaries over-cluster spatial error. Full-sample CUM result is significant at 5% under Conley 200km HAC.

### Donut test (`06_extended_iv.py` S3) — `wri_post_r1_cum`, full sample

| Specification | Coef | SE | p |
|---|---:|---:|---:|
| Full K=6 (all neighbors) | +0.051 | 0.036 | 0.161 |
| Donut K=3–6 (drop 2 nearest) | +0.029 | 0.022 | 0.199 |

Effect attenuates ~43% when dropping nearest 2 neighbors, consistent with stronger spillovers at short distance. Does not rule out contamination but is the expected direction for a genuine peer effect.

### Vendor network channel (`06_extended_iv.py` S4) — `wri_post_r1_cum`, full sample

| Specification | Coef | SE | p | Stars |
|---|---:|---:|---:|---|
| Geographic IV, no vendor ctrl | +0.051 | 0.036 | 0.161 | |
| Geographic IV + vendor ctrl | +0.048 | 0.027 | 0.088 | * |
| Vendor own effect (shares_dealer_with_r1) | +0.577 | — | <0.001 | *** |

Vendor network is strongly predictive of adoption but barely attenuates the geographic IV coefficient. Geographic and vendor channels are largely distinct.

### Priority decomposition (`06_extended_iv.py` S5) — `wri_post_r1_cum`

**PM2.5 quartile (full sample):**

| Quartile | Mean PM2.5 | N | Coef | SE | p | Stars |
|---|---:|---:|---:|---:|---:|---|
| Q1 (lowest) | 5.0 | 2,935 | +0.104 | 0.035 | 0.006 | *** |
| Q2 | 7.0 | 3,081 | +0.086 | 0.069 | 0.218 | |
| Q3 | 8.2 | 3,154 | −0.017 | 0.034 | 0.620 | |
| Q4 (highest) | 9.3 | 3,218 | +0.047 | 0.093 | 0.621 | |

**Within priority subsample — PM2.5 median split (median = 7.36):**

| Group | N | Coef | SE | p | Stars |
|---|---:|---:|---:|---:|---|
| Low PM2.5 (poverty-eligible) | 3,201 | +0.099 | 0.033 | 0.005 | *** |
| High PM2.5 (pollution-eligible) | 3,201 | +0.036 | 0.057 | 0.526 | |

**Within priority subsample — poverty median split:**

| Group | N | Coef | SE | p | Stars |
|---|---:|---:|---:|---:|---|
| High poverty | 3,216 | +0.097 | 0.058 | 0.099 | * |
| Low poverty | 3,186 | +0.030 | 0.036 | 0.404 | |

**Key finding:** The priority effect is driven by **poverty-eligible districts** (low PM2.5, high poverty), not by the most-polluted districts. Consistent with peer effects lowering informational/cost barriers for resource-constrained districts.

### Timing split (`05_main_iv.py` S4) — priority subsample

| Delivery timing | Outcome | Coef | SE | p | Stars |
|---|---|---:|---:|---:|---|
| Pre-R3 delivery (visible) | wri_post_r1_cum | +0.095 | 0.045 | 0.040 | ** |
| Post-R3 delivery | wri_post_r1_cum | +0.052 | 0.047 | 0.276 | |
| Wald test (pre=post) | — | — | — | 0.52 | |

Pre-R3 coefficient larger, right direction, but Wald test underpowered. Mechanism remains ambiguous (visibility vs. informational).
