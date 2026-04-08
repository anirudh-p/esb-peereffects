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
- [ ] Re-run K=6 primary specs with Conley HAC (spatial autocorrelation-robust SE)
- [ ] Compare state-clustered vs. Conley SEs
- **Script:** `05_main_iv.py` Section 5 (K sensitivity done); Conley HAC pending

### 4. Placebo / falsification test
- [x] R3 wins → pre-R1 adoption (placebo 1)
- [x] R1 wins → pre-R1 adoption (placebo 2)
- [x] R1 wins → WRI 2023-only (timing check)
- **Script:** `05_main_iv.py` Section 6
- **Note:** Placebo 1 & 2 flag — coefficient is near-zero but p<0.001 due to numerical precision. Needs investigation (likely a collinearity/scale issue with the strata FE).

### 5. Vendor network channel
- [ ] Extract dealer/vendor from WRI bus-level data (`3x. Dealer`)
- [ ] Construct vendor-network variable: does district i share a dealer with any R1 winner?
- [ ] Compare: geographic peer effect vs. vendor-network peer effect
- [ ] Test whether geographic effect attenuates after controlling for vendor network

### 6. Climatic peers + donut test
- [ ] Construct climate-quintile peer groups from PRISM data (temperature + precipitation)
- [ ] Re-run main spec with climate peers as alternative network definition
- [ ] Donut test: drop K=1-2 nearest neighbors, re-run with remaining K=3-6

### 7. Priority subsample decomposition
- [ ] Triple interaction: priority × winning neighbor × PM2.5 quartile
- [ ] Triple interaction: priority × winning neighbor × poverty quartile
- [ ] Which dimension of priority is driving the stronger effect?

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
