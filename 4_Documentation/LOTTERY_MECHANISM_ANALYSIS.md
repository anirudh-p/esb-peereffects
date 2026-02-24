# Analysis of Clean School Bus Lottery Mechanism

## Executive Summary

The lottery mechanism shows **strong evidence of stratification by priority status**, which explains the apparent paradox you observed: balance exists in the aggregate sample but not within priority strata.

---

## Key Findings

### 1. **Differential Win Rates by Priority Status**

| Group | Applicants | Winners | Win Rate |
|-------|-----------|---------|----------|
| **Priority** | 1,265 | 154 | **12.2%** |
| **Non-Priority** | 1,372 | 595 | **43.4%** |
| **Overall** | 2,637 | 749 | **28.4%** |

**Statistical Test**: χ² = 313.39, p < 0.0001
- **Conclusion**: Priority status is **highly significantly** associated with lottery outcome
- This is the opposite of what "priority" intuitively suggests!

---

## Explanation: Why Balance Appears in Aggregate But Not Within Strata

### Simpson's Paradox in Action

The phenomenon you observed is a classic example of **Simpson's Paradox**:

1. **Priority Applicants** (48% of pool):
   - Systematically WORSE on covariates (lower income, higher poverty, etc.)
   - Much LOWER probability of winning (12%)
   
2. **Non-Priority Applicants** (52% of pool):
   - Systematically BETTER on covariates (higher income, lower poverty, etc.)
   - Much HIGHER probability of winning (43%)

3. **Aggregate Balance**:
   - When pooling both groups together, the opposing biases partially cancel out
   - Winners appear similar to losers on average because:
     - Few priority winners (poor districts) + many non-priority winners (rich districts) 
     - Many priority losers (poor districts) + fewer non-priority losers (rich districts)
   - The weighted average makes it look balanced!

4. **Stratified Imbalance**:
   - **Within priority stratum**: Only 12% won → small winner sample, potential selection issues
   - **Within non-priority stratum**: 43% won → large winner sample, but may still have imbalance

---

## Possible Explanations for the Lottery Design

### Theory 1: **Separate Lottery Pools by Priority Status**
- The EPA may have run **two separate lotteries**:
  - One pool for priority applicants (with fewer slots or lower probability)
  - One pool for non-priority applicants (with more slots or higher probability)
- This would be consistent with wanting to ensure geographic/demographic diversity

### Theory 2: **Sequential Selection Process**
- Non-priority applicants were selected first, then priority applicants filled remaining slots
- Or vice versa, but with a quota system

### Theory 3: **Application Quality/Completeness**
- "Priority" status might reflect a different characteristic than advantageous selection
- Perhaps priority applicants had weaker applications or missed eligibility criteria

---

## Implications for Your Analysis

### ✅ **What This Means for IV Validity**

**GOOD NEWS**: The lottery mechanism is still valid for causal inference, but you must:

1. **Always Control for Priority Status**
   - Include `is_priority` as a covariate in all regressions
   - Or run separate analyses by priority stratum

2. **Use Stratified IV Estimation**
   - Your current approach of testing balance within strata is CORRECT
   - This accounts for the differential selection probabilities

3. **Interpret Lottery as "Conditional Randomization"**
   - The lottery is random **within priority strata**
   - The instrument (IV_Z) is valid conditional on priority status
   - This is similar to how RCTs often stratify by baseline characteristics

### ⚠️ **What Would Be Problematic**

- Ignoring priority status and pooling all applicants (naive approach)
- This would introduce omitted variable bias because priority status predicts both:
  - Treatment (winning lottery)
  - Covariates (demographics, poverty, etc.)

---

## Recommended Approach

### Primary Specification:
```python
# Include priority status as control
model = "IS_ADOPTER ~ w_geo_adoption + w_clim_adoption + is_priority + [other_controls]"

# Instruments should also condition on priority
IV_spec = "w_geo_z + w_clim_z | is_priority + [other_controls]"
```

### Robustness Checks:
1. **Separate analysis by priority status** to check for heterogeneous effects
2. **Interaction terms**: `IV_Z × is_priority` to test if treatment effects differ
3. **Report stratified F-statistics** in first stage (should be strong within both strata)

---

## Additional Balance Test Enhancement

The updated script now includes:

1. ✅ **Chi-square test** for independence of priority status and lottery outcome
2. ✅ **Balance tests for BOTH priority AND non-priority strata** (previously only non-priority)
3. ✅ **Diagnostic visualization** showing:
   - Win rate disparities
   - Sample composition
   - Priority distribution among winners/losers
   - Statistical test interpretation

---

## Bottom Line

**Your observation was correct and important!** The aggregate balance was masking a fundamental feature of the lottery design: stratification by priority status. This doesn't invalidate your IV approach—in fact, your instinct to test balance within strata shows good empirical practice. Just ensure all analyses control for or stratify by priority status.
