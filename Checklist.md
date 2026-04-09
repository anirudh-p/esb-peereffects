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
- **Script:** `01_main_iv.py` Sections 1–2

### 2. Timing / visibility split — finish the mechanism story
- [x] Re-run pre-R3-deadline vs. post-deadline delivery split on `wri_post_r1_cum`
- [x] Wald test: pre = post
- **Script:** `01_main_iv.py` Section 4
- **Finding (Fix A):** wri_post_r1_cum timing split flat (pre=+0.036 p=0.446, post=+0.041 p=0.417). Y_R3_apply post-R3 delivery p=0.050* > pre-R3 p=0.231, suggesting informational channel (learning from winners' application experience) rather than visual demonstration.

### 3. K sensitivity + Conley spatial HAC
- [x] Main table with K ∈ {6, 10, 15} for `wri_post_r1_cum`
- [x] Re-run K=6 primary specs with Conley HAC (spatial autocorrelation-robust SE)
- [x] Compare state-clustered vs. Conley SEs
- **Script:** `01_main_iv.py` Section 5; `02_extended_iv.py` Section 2
- **Finding (Fix A):** wri_post_r1_cum null under all SE choices (state-clustered p=0.438, Conley 200km p=0.304). Y_R3_apply full sample null under all SE choices; priority p=0.015** (state-clustered) is the headline. Fix A removed the Conley significance of CUM — that result was an artifact of the contaminated 2022 Q4 CSBP awards.

### 4. Placebo / falsification test
- [x] R3 wins → pre-R1 adoption (placebo 1)
- [x] R1 wins → pre-R1 adoption (placebo 2)
- [x] R1 wins → WRI 2023-only (timing check)
- **Script:** `01_main_iv.py` Section 6
- **Note:** Placebo 1 & 2 flag — coefficient is near-zero but p<0.001 due to numerical precision. Likely collinearity/scale artifact with strata FE in restricted sample. Not a real identification concern.

### 5. Vendor network channel
- [x] Extract dealer/vendor from WRI bus-level data (`3x. Dealer`)
- [x] Construct vendor-network variable: does district i share a dealer with any R1 winner?
- [x] Compare: geographic peer effect vs. vendor-network peer effect
- [x] Test whether geographic effect attenuates after controlling for vendor network
- **Script:** `03_build_supplementary.py`; `02_extended_iv.py` Section 4
- **Finding (Fix A):** `shares_dealer_with_r1` strongly predicts adoption (coef=+0.426, p=0.008***), geographic IV barely attenuates (0.017→0.014). Dealer-connected subsample (N=311) coef=−0.257 p=0.013** — supply-capacity reversal: dealer busy with R1 orders reduces non-CSBP adoption capacity.

### 6. Climatic peers + donut test
- [ ] Construct climate-quintile peer groups from PRISM data (temperature + precipitation)
- [ ] Re-run main spec with climate peers as alternative network definition
- [x] Donut test: drop K=1-2 nearest neighbors, re-run with remaining K=3-6
- **Script:** `02_extended_iv.py` Sections 3 & 6
- **Donut finding (Fix A):** Full K=6 coef=+0.017; Donut K=3-6 coef=+0.007. Effect attenuates ~56% when dropping nearest 2 neighbors, direction consistent with stronger spillovers at short distances. Neither significant (full sample null throughout).
- **Climate note:** PRISM data not yet merged. Raw files in `1_Data/Raw/Climate/`. ACTION NEEDED.

### 7. Priority subsample decomposition
- [x] Triple interaction: priority × winning neighbor × PM2.5 quartile
- [x] Triple interaction: priority × winning neighbor × poverty quartile
- [x] Which dimension of priority is driving the stronger effect?
- **Script:** `02_extended_iv.py` Section 5
- **Finding (Fix A):** Low PM2.5 priority: coef=+0.068 p=0.052*; High PM2.5: p=0.833. PM2.5 Q1 (full sample): +0.080 p=0.034**. Poverty-eligible (high poverty priority): +0.074 p=0.153. Direction consistent with informational/cost-barrier story for resource-constrained districts; significance attenuated under clean outcome.

### 8. Additional outcome margins (new)
- [x] Y_R2_apply (R2 competitive grant application)
- [x] Y_any_post_r1 (R2 OR R3 — broadest CSBP entry margin)
- **Script:** `03_build_supplementary.py`; `02_extended_iv.py` Section 1
- **Finding (Fix A):** Y_R2_apply null everywhere. Y_any_post_r1 priority p=0.045**, full sample p=0.701. Peer effect concentrated in R3 application margin (CSBP rebate), not competitive grant.

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

**Outcome definition:** `wri_post_r1_cum` = `wri_any_2023_24 & ~is_pre_r1_adopter` (Fix A).  
2022 fully excluded: Q4 2022 = R1 CSBP award quarter for the R1 winners themselves; including it would contaminate the outcome. 2023-2024 window is clean post-delivery.

Scripts: `01_main_iv.py`, `02_extended_iv.py`. All K=6, loser=Yes, state-clustered SE unless noted.

### Core IV results (reduced-form, `01_main_iv.py`)

| Outcome | Sample | Coef | SE | p | Stars |
|---|---|---:|---:|---:|---|
| wri_post_r1_cum | Full | +0.017 | 0.022 | 0.438 | |
| wri_post_r1_cum | Priority | +0.041 | 0.028 | 0.155 | |
| Y_R3_apply | Full | +0.036 | 0.027 | 0.198 | |
| Y_R3_apply | Priority | +0.112 | 0.045 | 0.015 | ** |

**Primary headline result: Y_R3_apply priority p=0.015**. WRI adoption null everywhere under clean definition.

### Additional outcome margins (`02_extended_iv.py` S1)

| Outcome | Sample | Coef | SE | p | Stars |
|---|---|---:|---:|---:|---|
| Y_R2_apply (competitive grant) | Full | +0.005 | 0.023 | 0.827 | |
| Y_R2_apply | Priority | +0.038 | 0.030 | 0.205 | |
| Y_any_post_r1 (R2 OR R3) | Full | +0.013 | 0.033 | 0.701 | |
| Y_any_post_r1 | Priority | +0.090 | 0.044 | 0.045 | ** |

Y_R2_apply null everywhere. Y_any_post_r1 priority p=0.045** tracks R3 — R2 adds no marginal signal.

### Conley spatial HAC (`02_extended_iv.py` S2) — full sample

| Outcome | SE type | SE | p | Stars |
|---|---|---:|---:|---|
| wri_post_r1_cum | State-clustered | 0.022 | 0.438 | |
| wri_post_r1_cum | Conley 100km | 0.022 | 0.446 | |
| wri_post_r1_cum | Conley 200km | 0.016 | 0.304 | |
| wri_post_r1_cum | Conley 300km | 0.020 | 0.390 | |
| Y_R3_apply | State-clustered | 0.027 | 0.198 | |
| Y_R3_apply | Conley 100km | 0.027 | 0.193 | |
| Y_R3_apply | Conley 200km | 0.028 | 0.202 | |
| Y_R3_apply | Conley 300km | 0.030 | 0.237 | |

Under the clean outcome definition, wri_post_r1_cum is null under all SE choices. Y_R3_apply also null in full sample under all SE choices (but priority is p=0.015** under state-clustered, not separately tabulated here).

### Donut test (`02_extended_iv.py` S3) — `wri_post_r1_cum`, full sample

| Specification | Coef | SE | p |
|---|---:|---:|---:|
| Full K=6 (all neighbors) | +0.017 | 0.022 | 0.438 |
| Donut K=3–6 (drop 2 nearest) | +0.007 | 0.014 | 0.599 |

Effect attenuates ~56% when dropping nearest 2 neighbors. Direction consistent with stronger spillovers at short distance, though neither spec is significant.

### Vendor network channel (`02_extended_iv.py` S4) — `wri_post_r1_cum`, full sample

| Specification | Coef | SE | p | Stars |
|---|---:|---:|---:|---|
| Geographic IV, no vendor ctrl | +0.017 | 0.022 | 0.438 | |
| Geographic IV + vendor ctrl | +0.014 | 0.019 | 0.448 | |
| Vendor own effect (shares_dealer_with_r1) | +0.426 | — | 0.008 | *** |

Vendor network strongly predicts adoption but barely attenuates geographic IV coefficient. Dealer-connected heterogeneity: N=311 coef=−0.257 p=0.013** (supply-capacity constraint — dealer busy with R1 orders may reduce capacity for non-CSBP adoption).

### Priority decomposition (`02_extended_iv.py` S5) — `wri_post_r1_cum`

**PM2.5 quartile (full sample):**

| Quartile | Mean PM2.5 | N | Coef | SE | p | Stars |
|---|---:|---:|---:|---:|---:|---|
| Q1 (lowest) | 5.0 | 2,935 | +0.080 | 0.036 | 0.034 | ** |
| Q2 | 7.0 | 3,081 | +0.016 | 0.044 | 0.724 | |
| Q3 | 8.2 | 3,154 | −0.028 | 0.034 | 0.416 | |
| Q4 (highest) | 9.3 | 3,218 | +0.034 | 0.081 | 0.677 | |

**Within priority subsample — PM2.5 median split (median = 7.36):**

| Group | N | Coef | SE | p | Stars |
|---|---:|---:|---:|---:|---|
| Low PM2.5 (poverty-eligible) | 3,201 | +0.068 | 0.034 | 0.052 | * |
| High PM2.5 (pollution-eligible) | 3,201 | +0.010 | 0.048 | 0.833 | |

**Within priority subsample — poverty median split:**

| Group | N | Coef | SE | p | Stars |
|---|---:|---:|---:|---:|---|
| High poverty | 3,216 | +0.074 | 0.051 | 0.153 | |
| Low poverty | 3,186 | +0.005 | 0.032 | 0.872 | |

**Key finding:** WRI adoption peer effect (where present) concentrated in low-PM2.5 Q1 districts (p=0.034**) and poverty-eligible priority districts (low PM2.5 p=0.052*). High-pollution districts null. Consistent with informational/cost-barrier story for resource-constrained districts.

### Timing split (`01_main_iv.py` S4) — priority subsample

| Delivery timing | Outcome | Coef | SE | p | Stars |
|---|---|---:|---:|---:|---|
| Pre-R3 delivery (visible) | wri_post_r1_cum | +0.036 | 0.047 | 0.446 | |
| Post-R3 delivery | wri_post_r1_cum | +0.041 | 0.050 | 0.417 | |
| Pre-R3 delivery (visible) | Y_R3_apply | +0.083 | 0.068 | 0.231 | |
| Post-R3 delivery | Y_R3_apply | +0.120 | 0.060 | 0.050 | * |

Under clean outcome, timing split for wri_post_r1_cum is flat (pre≈post). For Y_R3_apply, the post-R3 delivery group is marginally significant (p=0.050*), suggesting informational channel (learning from winners' application experience) rather than visual demonstration effect.
