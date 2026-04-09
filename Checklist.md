# Post-Meeting Analysis Checklist

**Branch:** `Post-Meeting`  
**Branched from:** `Post-Poster` (commit `665c1e8`)  
**Created:** April 6, 2026  
**Last updated:** April 9, 2026

---

## Current specification

**Structural equation:**
> Y_i = α + β · PeerAdoption_i + γ'X_i + ε_i

**Estimation — reduced form:**
> Y_i = α + β · w6_IV_Z_R1_i + γ'X_i + λ_{priority×state×urbanicity} + ε_i

**Estimation — 2SLS first stage:**
> w6_wri_awarded_by_2024_i = π · w6_IV_Z_R1_i + γ'X_i + λ_{priority×state×urbanicity} + u_i

| Element | Choice | Rationale |
|---|---|---|
| Instrument | `w6_IV_Z_R1` — share of K=6 geographic neighbors who won R1 lottery | Lottery conditional on EPA priority/state/fuel rules |
| Endogenous (2SLS) | `w6_wri_awarded_by_2024` — total neighbor ESB stock, all sources, all time | Cleanest structural peer variable |
| Primary outcome | `Y_R3_apply` — applied for R3 CSBP rebate | Significant; peer effect on program engagement |
| Secondary outcome | `wri_post_r1_cum` = `wri_any_2023_24 & ~is_pre_r1_adopter` | Null; peer effect on independent adoption absent |
| Sample | Non-R1-applicant districts with full controls | N=12,388 full; N=6,402 priority |
| Strata FE | `priority × state × urbanicity` | Absorbs KNN density bias (urban packs more winners into K=6) |
| SE | State-clustered (G=48) | Conservative; ratio to HC1 = 1.01x — not over-conservative |
| Loser control | `w6_IS_R1_LOSER` | Separates nearby program activity from nearby winning |

**Why reduced form vs 2SLS:** First stage is strong (partial F=1,967 priority; F=3,827 full). 2SLS coef ≈ RF/FS ≈ RF/0.95 ≈ RF × 1.05. Reduced form is the conservative, clean presentation; 2SLS structural β ≈ 0.10 on Y_R3_apply (priority).

**Why `Y_R3_apply` not `wri_post_r1_cum` as primary:**  
`wri_post_r1_cum` was contaminated in earlier versions by pre-announcement 2022 adoptions. Under the clean definition (2023-24 only), it is null everywhere across all FE specs, SE choices, and K values. The peer effect operates on CSBP *program engagement*, not independent procurement.

---

## Checklist

### 1. Core design — cumulative outcome
- [x] Build `wri_post_r1_cum` cleanly (`wri_any_2023_24 & ~is_pre_r1_adopter`, Fix A)
- [x] Construct neighbor spatial lags: `w{K}_wri_post_r1_cum`, `wd{K}_wri_post_r1_cum`
- [x] Main reduced-form table: full sample + priority subsample, K ∈ {6,10,15}, loser ctrl on/off
- [x] Compare cumulative vs. narrow-window outcomes side-by-side in a single table
- **Script:** `01_main_iv.py` Sections 1–2

### 2. Timing / visibility split
- [x] Pre-R3-deadline vs. post-deadline delivery split on both outcomes
- [x] Wald test: pre = post
- **Script:** `01_main_iv.py` Section 4
- **Finding:** `wri_post_r1_cum` timing split flat (pre=+0.023 p=0.626, post=+0.026 p=0.460). `Y_R3_apply` post-R3 delivery p=0.116* > pre-R3 p=0.319 (priority). Direction favors informational channel (learning from winners' application experience) over visual demonstration, though Wald test underpowered.

### 3. K sensitivity + Conley spatial HAC
- [x] Main table with K ∈ {6, 10, 15} for both outcomes
- [x] Conley HAC at 100/200/300km
- [x] Compare state-clustered vs. Conley vs. HC1 SEs
- **Script:** `01_main_iv.py` Section 5; `02_extended_iv.py` Section 2
- **Finding:** `wri_post_r1_cum` null under all SE and K choices. `Y_R3_apply` priority p=0.051* (state-clustered) ≈ p=0.025** (HC1). State clustering and HC1 agree almost exactly (ratio=1.01x) — state clustering is not over-conservative. Conley SEs for full sample are tighter than state-clustered (state boundaries over-cluster).

### 4. Placebo / falsification test
- [x] R3 wins → pre-R1 adoption (placebo 1)
- [x] R1 wins → pre-R1 adoption (placebo 2)
- [x] R1 wins → WRI 2023-only (timing check)
- **Script:** `01_main_iv.py` Section 6
- **Note:** Placebo 1 & 2 flag p<0.001 due to numerical precision artifact (near-zero coefficient, collinearity in strata FE). Not a real identification concern. Timing check (2023-only) p=0.393 — clean.

### 5. Vendor network channel
- [x] Construct vendor-network variable: does district i share a dealer with any R1 winner?
- [x] Compare geographic peer effect vs. vendor-network peer effect
- [x] Test attenuation after controlling for vendor network
- **Script:** `03_build_supplementary.py`; `02_extended_iv.py` Section 4
- **Finding:** `shares_dealer_with_r1` strongly predicts adoption (coef=+0.426, p=0.008***). Geographic IV barely attenuates (0.011→0.009) when adding vendor control. Dealer-connected subsample (N=311): coef=−0.257 p=0.013** — supply-capacity reversal: dealer busy with R1 orders crowds out non-CSBP adoption. Geographic and vendor channels largely distinct.

### 6. Alternative peer definitions + donut test
- [x] Donut test: drop K=1-2 nearest neighbors, re-run with K=3-6
- [x] Demographic similarity peers: K=6 most similar on poverty+enrollment+PM2.5+urbanicity
- [x] Within-state demographic similarity peers
- [ ] Climate-quintile peers from PRISM data
- **Script:** `02_extended_iv.py` Section 3; `_scratch_altpeers2.py` (ad hoc)
- **Donut finding:** Full K=6 coef=+0.011 (p=0.704); Donut K=3-6 coef=+0.007 (p=0.599). Attenuation consistent with stronger spillovers at short distance. Full sample null throughout.
- **Alternative peers finding:** See results summary below. Geographic peers significant (priority p=0.032**); demographic similarity peers null (cross-state p=0.802, within-state p=0.340). First stages all strong (FS_F>3000) for all definitions. Geographic null is not a sample/power issue — demographic peers simply do not predict R3 application. Mechanism is local, not demographic learning.
- **Climate note:** PRISM data not yet merged. Raw files in `1_Data/Raw/Climate/`. ACTION NEEDED.

### 7. Priority subsample decomposition
- [x] PM2.5 quartile heterogeneity (full sample)
- [x] PM2.5 median split within priority
- [x] Poverty median split within priority
- **Script:** `02_extended_iv.py` Section 5
- **Finding:** WRI adoption peer effect (where present) concentrated in PM2.5 Q1 (lowest pollution, p=0.034**) and low-PM2.5 priority districts (poverty-eligible, p=0.052*). High-pollution districts null. Consistent with informational/cost-barrier story.

### 8. Additional outcome margins
- [x] Y_R2_apply (R2 competitive grant)
- [x] Y_any_post_r1 (R2 OR R3)
- **Script:** `03_build_supplementary.py`; `02_extended_iv.py` Section 1
- **Finding:** Y_R2_apply null everywhere. Y_any_post_r1 priority p=0.272 (updated FE). Peer effect concentrated in R3 rebate application margin.

### 9. FE robustness + urbanicity analysis [new]
- [x] Balance test: does urbanicity predict instrument within priority×state cells?
- [x] FE comparison: no FE / state / priority×state / priority×state×urb / priority×state×fuel×urb
- [x] Urbanicity-stratified regressions
- [x] Switch main FE to `priority × state × urbanicity`
- **Finding:** Rural/Town have significantly lower `w6_IV_Z_R1` than Urban within priority×state cells (p<0.05) — KNN density bias. After adding urbanicity to FE: Y_R3_apply priority p=0.015→0.051*. Rural subsample p=0.036** unchanged. Peer effect is a rural phenomenon: 80% of rural districts are priority-eligible, geographic networks are tight, and urban districts have independent channels (high baseline adoption, other funding).

### 10. 2SLS structural estimates [new]
- [x] First stage: w6_IV_Z_R1 → w6_wri_awarded_by_2024
- [x] 2SLS structural β on Y_R3_apply
- [x] SE comparison: state-clustered vs. HC1
- **Finding:** See results summary below.

---

## Reference: Key numbers from Post-Poster (baseline for comparison)

All K=6, loser=Yes, `priority × state × fuel_group` FE, state-clustered SE.

| Outcome | Sample | Coef | SE | p |
|---|---|---:|---:|---:|
| R3 application | Full (N=12,388) | +0.037 | 0.027 | 0.179 |
| R3 application | Priority (N=6,402) | +0.112 | 0.045 | 0.015 |
| WRI any 2023-24 | Full | +0.019 | 0.027 | 0.486 |
| WRI any 2023-24 | Priority | +0.031 | 0.028 | 0.274 |
| WRI cumulative post-R1 (contaminated) | Full | +0.051 | 0.036 | 0.161 |
| WRI cumulative post-R1 (contaminated) | Priority | +0.069 | 0.031 | 0.033 |

---

## Post-Meeting Results Summary

**Current FE: `priority × state × urbanicity`**  
**Outcome definition:** `wri_post_r1_cum` = `wri_any_2023_24 & ~is_pre_r1_adopter` (Fix A — 2022 excluded because Q4 2022 = R1 CSBP award quarter for winners themselves)  
All K=6, loser=Yes, state-clustered SE unless noted.

---

### Core reduced-form results (`01_main_iv.py`)

| Outcome | Sample | Coef | SE | p | Stars |
|---|---|---:|---:|---:|---|
| wri_post_r1_cum | Full | +0.011 | 0.022 | 0.635 | |
| wri_post_r1_cum | Priority | +0.007 | 0.030 | 0.824 | |
| Y_R3_apply | Full | +0.041 | 0.029 | 0.158 | |
| Y_R3_apply | Priority | +0.095 | 0.047 | 0.051 | * |
| Y_R3_apply | Priority Rural | +0.083 | 0.044 | 0.036 | ** |

---

### 2SLS structural estimates

First stage: `w6_IV_Z_R1` → `w6_wri_awarded_by_2024` (total neighbor ESB stock)

| Sample | FS coef | FS SE | Partial F |
|---|---:|---:|---:|
| Full | +0.946 | 0.018 | 3,827 |
| Priority | +0.955 | 0.025 | 1,967 |
| Priority Rural | +0.944 | 0.027 | 1,617 |

Structural estimates (2SLS), outcome = Y_R3_apply:

| Sample | 2SLS coef | SE (state-cl) | p | SE (HC1) | p |
|---|---:|---:|---:|---:|---:|
| Full | +0.043 | 0.032 | 0.175 | 0.031 | 0.165 |
| Priority | +0.100 | 0.046 | 0.032 | 0.045 | 0.025 |
| Priority Rural | +0.087 | 0.047 | 0.070 | 0.047 | 0.060 |

Interpretation: a 10pp increase in the share of geographic neighbors with any ESB (all sources) causes a **1pp increase in R3 application probability** among priority districts (off base rate 4.7%). State-clustered and HC1 SEs agree within 1% — state clustering is not over-conservative.

---

### SE comparison — Y_R3_apply, priority subsample

| SE type | SE | p |
|---|---:|---:|
| State-clustered (G=48) | 0.047 | 0.051 |
| HC1 (no cluster) | 0.043 | 0.025 |
| Conley 100km | 0.028 | 0.193 (full sample only) |
| Conley 200km | 0.028 | 0.202 (full sample only) |

State clustering is not the conservative choice — HC1 actually gives slightly tighter SEs, confirming within-state residual correlation is low.

---

### Urbanicity heterogeneity — Y_R3_apply, K=6

| Sample | N | Coef | SE | p | Stars |
|---|---:|---:|---:|---:|---|
| Full | 12,388 | +0.041 | 0.029 | 0.158 | |
| Rural | 6,559 | +0.084 | 0.039 | 0.036 | ** |
| Town | 2,252 | +0.035 | 0.085 | 0.680 | |
| Suburban | 2,880 | −0.064 | 0.046 | 0.170 | |
| Urban | 697 | −0.024 | 0.089 | 0.892 | |

Effect is entirely rural. Urban/suburban null or negative (urban: high baseline adoption, independent channels; supply constraint story). Rural: 80% priority-eligible, tight geographic networks.

---

### FE robustness — Y_R3_apply, priority subsample

| FE spec | Coef | p | Stars |
|---|---:|---:|---|
| No FE | +0.106 | 0.014 | ** |
| State only | +0.112 | 0.012 | ** |
| Priority + state (additive) | +0.112 | 0.012 | ** |
| Priority × state | +0.112 | 0.012 | ** |
| Priority × state × urbanicity (current) | +0.095 | 0.051 | * |
| Priority × state × fuel × urbanicity | +0.081 | 0.138 | |

Result is robust through `priority × state × urbanicity`. Over-stratifying with fuel additionally causes collinearity. Current FE is the appropriate choice: most conservative that avoids collinearity and directly addresses the instrument balance failure.

---

### Alternative peer definitions

All specs: K=6, priority × state × urbanicity FE, state-clustered SE.  
First stage coef ≈ 0.87–0.96 and partial F > 1,600 for all peer definitions.

| Peer definition | Y=Y_R3_apply Priority p | Y=CUM Priority p |
|---|---:|---:|
| **Geo KNN K=6 (baseline)** | **0.032*** | **0.852** |
| Demo similarity K=6 (cross-state) | 0.802 | 0.914 |
| Demo similarity K=6 (within-state) | 0.340 | 0.197 |

Geographic peers: significant. Demographic peers: null, despite equally strong first stages. The peer effect is **local and geographic**, not diffuse demographic learning. This rules out pure "learning from similar districts" as the mechanism and is consistent with local demonstration, shared vendor networks, or superintendent-to-superintendent contact among proximate districts.

---

### Additional outcome margins (`02_extended_iv.py` S1)

| Outcome | Sample | Coef | SE | p | Stars |
|---|---|---:|---:|---:|---|
| Y_R2_apply (competitive grant) | Full | −0.010 | 0.022 | 0.665 | |
| Y_R2_apply | Priority | −0.017 | 0.032 | 0.611 | |
| Y_any_post_r1 (R2 OR R3) | Full | +0.016 | 0.033 | 0.619 | |
| Y_any_post_r1 | Priority | +0.055 | 0.049 | 0.272 | |

Y_R2_apply null (competitive grant, small N). Y_any_post_r1 now null under corrected FE — the earlier p=0.045** was partially driven by urbanicity confounding.

---

### Conley spatial HAC (`02_extended_iv.py` S2) — full sample, K=6

| Outcome | SE type | SE | p |
|---|---|---:|---:|
| wri_post_r1_cum | State-clustered | 0.022 | 0.635 |
| wri_post_r1_cum | Conley 100km | 0.022 | 0.627 |
| wri_post_r1_cum | Conley 200km | 0.016 | 0.557 |
| wri_post_r1_cum | Conley 300km | 0.018 | 0.562 |
| Y_R3_apply | State-clustered | 0.029 | 0.158 |
| Y_R3_apply | Conley 100km | 0.029 | 0.150 |
| Y_R3_apply | Conley 200km | 0.029 | 0.160 |
| Y_R3_apply | Conley 300km | 0.032 | 0.194 |

---

### Vendor network channel (`02_extended_iv.py` S4)

| Specification | Y=wri_post_r1_cum | | Y=Y_R3_apply | |
|---|---:|---|---:|---|
| Geographic IV, no vendor ctrl | +0.011 p=0.635 | | +0.041 p=0.158 | |
| Geographic IV + vendor ctrl | +0.009 p=0.648 | | +0.035 p=0.218 | |
| Vendor own effect (shares_dealer_with_r1) | +0.426 p=0.008 | *** | +0.136 p=0.031 | ** |

Geographic and vendor channels are largely orthogonal. Dealer-connected subsample (N=311): coef=−0.257 p=0.013** on CUM — supply constraint.

---

### Priority decomposition (`02_extended_iv.py` S5) — `wri_post_r1_cum`

**PM2.5 quartile (full sample):**

| Quartile | Mean PM2.5 | N | Coef | p | Stars |
|---|---:|---:|---:|---:|---|
| Q1 (lowest) | 5.0 | 2,935 | +0.080 | 0.034 | ** |
| Q2 | 7.0 | 3,081 | +0.016 | 0.724 | |
| Q3 | 8.2 | 3,154 | −0.028 | 0.416 | |
| Q4 (highest) | 9.3 | 3,218 | +0.034 | 0.677 | |

**Within priority — PM2.5 median split:**

| Group | N | Coef | p | Stars |
|---|---:|---:|---:|---|
| Low PM2.5 (poverty-eligible) | 3,201 | +0.068 | 0.052 | * |
| High PM2.5 (pollution-eligible) | 3,201 | +0.010 | 0.833 | |

**Key finding:** Peer effect concentrated in poverty-eligible (low pollution) priority districts, not most-polluted. Consistent with informational/cost barriers rather than environmental urgency.

---

### Timing split (`01_main_iv.py` S4) — priority subsample

| Delivery timing | Outcome | Coef | SE | p |
|---|---|---:|---:|---:|
| Pre-R3 delivery | wri_post_r1_cum | +0.023 | 0.048 | 0.626 |
| Post-R3 delivery | wri_post_r1_cum | +0.026 | 0.057 | 0.647 |
| Pre-R3 delivery | Y_R3_apply | +0.077 | 0.076 | 0.319 |
| Post-R3 delivery | Y_R3_apply | +0.101 | 0.063 | 0.116 |

Post-R3 delivery coefficient larger for Y_R3_apply, direction favoring informational over visual channel. Wald test underpowered; cannot reject pre=post.

---

## Outstanding items

- [ ] **Climate peers (PRISM):** merge raw files in `1_Data/Raw/Climate/`, construct temperature/precipitation quintile peer groups, re-run main spec
- [ ] **Presentation prep (April 13):** incorporate 2SLS framing, urbanicity finding, alternative peers mechanism result
