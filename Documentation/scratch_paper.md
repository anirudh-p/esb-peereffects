# Scratch Paper

## 2026-04-28 Brown Pitch Setup

### Empirical Object

Top-level question: do nearby ESB lottery shocks, awards, or adoptions affect a focal district's later ESB applications, awards, or adoption?

The central empirical fact is spatial spillovers. The mechanisms are interpretations of that fact.

### Conceptual Hierarchy

1. Spatial spillovers: nearby treatment/adoption predicts focal outcomes.
2. Peer effects or peer learning: nearby districts reveal information about costs, procurement, charging, maintenance, feasibility, and political acceptability.
3. Capacity constraints or competition: nearby wins may crowd out focal districts through limited vendor capacity, grant-writing support, buses, chargers, staff time, or EPA funding queues.
4. Supply, vendor, or OEM push: early local adoption may attract vendors, installers, consultants, demonstrations, or OEM attention.
5. Policy or administrative diffusion: districts copy procurement templates, board language, consultants, or application strategies.
6. Homophily and common shocks: nearby districts may be similar or exposed to the same politics, infrastructure, state programs, vendors, and environmental priorities. This is the main identification threat.

### Identification Ladder

Level 0: spatial fact. Hazard Panel OLS establishes whether nearby ESB activity predicts focal behavior. It is not causal because of homophily and common shocks.

Level 1: lottery reduced form. Nearby randomized wins or subsidy shocks predict focal behavior. This can identify the effect of nearby subsidy shocks, not the precise mechanism.

Level 2: design-based exposure. Recentered or expected exposure under the lottery design isolates plausibly random variation in local exposure after conditioning on risk-set and timing structure.

Level 3: IV for neighbor adoption. Nearby lottery shocks instrument realized nearby adoption. This is appealing but has a stronger exclusion restriction because the shock may operate through learning, vendors, consultants, media salience, or capacity constraints.

Level 4: mechanisms. Channel tests can show whether patterns are more consistent with peer learning, supply expansion, administrative diffusion, or capacity competition. They should not be oversold as complete mechanism separation.

### Brown Spine

1. Descriptive spatial clustering.
2. Lottery reduced form.
3. Design-based or recentered IV.
4. Event-study timing.
5. One mechanism section: peer learning versus vendor/capacity channels.

### Four Main Specifications From Spatial_Spillovers

Spec A: Hazard Panel OLS. Use as descriptive baseline and motivation.

Spec B: Hazard Panel Raw IV 2SLS. Use as a bridge specification. It is less elegant than the design-based version and should not be the star.

Spec C: Hazard Panel Design-BH IV. Main causal specification for the Brown pitch.

Spec D: Hazard Panel Design-BH IV (C/F). Preferred robustness or final preferred version if controls, fixed effects, and channel restrictions are stable and pre-specified.

### Vendor Direction Question

Observed vendor-district matches are equilibrium outcomes. Same-vendor clustering may mean vendors targeted districts, districts selected vendors, consultants bundled applications, or all actors responded to the same EPA opportunity.

Possible routes:

1. Timing: vendor presence before focal application/adoption supports vendor diffusion; focal action before vendor exposure weakens that story.
2. First local entry: ask whether randomized nearby wins create subsequent same-vendor adoption nearby.
3. Same-vendor spillovers: compare exposure from neighbors using the same vendor or brand versus different vendors or brands.
4. Application versus award timing: effects on applications suggest learning or grant-writing support; effects only after awards or deliveries suggest visibility, implementation learning, or supply capacity.
5. Capacity constraint test: if neighbor wins reduce focal success conditional on application, or delay adoption, that supports congestion rather than positive learning.
6. Third-party filer proxies: shared filers or consultants can reveal administrative networks, but direction is still hard without sharp timing.

Working conclusion: the data probably cannot fully separate vendors choosing districts from districts choosing vendors without contact, proposal, or timestamp data. The attainable goal is to classify the spillover as more consistent with information diffusion, supply expansion, or capacity competition.

### Immediate Build Order

Start with the analysis dataset, not estimation. The first concrete task is a raw-data inventory and district identifier crosswalk that can support all four specs. Then build the hazard panel with outcomes, risk sets, spatial exposure, lottery exposure, and design-based/recentered exposure.

## 2026-04-28 Build Note: District Base

Implemented `01_build_analysis_dataset.py` as the first preparation step. The script now:

1. inventories raw files and raw tables;
2. standardizes NCES district IDs across WRI district data, WRI bus-level data, CSBP rebates, CSBP grants, CSBP waitlisted/rejected applicants, and 2023/2024 survey files;
3. writes a long source-record file and a one-row-per-NCES identifier crosswalk;
4. builds `analysis_district_base.csv` and `.dta` with WRI controls, CSBP applicant/winner flags, survey coverage flags, first ESB timing, first OEM/dealer/charger fields, and county-averaged 2020 Democratic vote share;
5. keeps the base universe as the full source crosswalk rather than only `wri_data.csv`, preserving 21 CSBP/applicant-source IDs absent from the WRI district CSV.

Current diagnostic counts: 19,516 unique NCES IDs in the crosswalk/base; 368 R1 rebate winners; 458 R3 rebate winners; 234 CSB grant awardees; 1,902 R1 lottery applicants; 755 R3 lottery applicants; 271 2023 survey responses; 255 2024 survey responses.

## 2026-04-28 Build Note: Hazard Panel

Implemented `02_build_panel_dataset.py` as the second preparation step. The script now:

1. consumes `analysis_district_base.csv`;
2. expands the source crosswalk universe to an annual 2018-2024 district-year panel;
3. constructs first-event outcomes for WRI award, delivery, operation, lottery application, rebate win, and grant award;
4. constructs at-risk indicators for each first-event outcome;
5. adds timing variables around R1/R3, own-winner post flags, own-winner exclusion flags, and basic sample-coverage flags;
6. intentionally leaves spatial/design-based neighbor exposure variables for the next preparation step.

Current diagnostic counts: 136,612 district-years; 19,516 districts; 1,492 first-award events in-panel; 2,308 first lottery-application events; 802 first rebate-win events; 234 grant-award events; 37,130 district-year rows in the 2022-2023 lottery application risk window.

## 2026-04-28 Build Note: Spatial and Design Exposure

Implemented `03_build_spatial_design_exposure.py` as the spatial/design layer. The script builds two coordinate universes:

1. EDGE-only: districts matched to the NCES EDGE school-district shapefile.
2. Hybrid: EDGE internal points where available, with WRI latitude/longitude as fallback for unmatched LEAs.

This directly addresses the earlier sample-drop problem. Current geometry coverage:

- EDGE shapefile: 13,083 districts; 1,815 R1 lottery applicants; 665 R3 applicants; 726 rebate winners; 207 grant awardees; 1,366 WRI committed ESB districts.
- WRI point fallback: 6,420 districts; 79 R1 applicants; 86 R3 applicants; 70 rebate winners; 27 grant awardees; 172 WRI committed ESB districts.
- No geometry: 13 districts; 8 R1 applicants; 4 R3 applicants; 6 rebate winners; 0 first-award events in the current WRI timing panel.

Interpretation: EDGE-only drops many LEAs that are not conventional geographic districts, especially charter districts, service agencies, supervisory unions, and non-LEA/private fleet entities. The hybrid graph preserves almost all substantively interesting rows for descriptive and lower-control specifications. However, once core WRI controls are required, the hybrid and EDGE samples are currently identical: 12,721 districts and 89,047 district-years. The fallback LEAs mostly lack the WRI control block, so retaining them requires either lighter controls, separate missing-control handling, or a deliberate non-geographic/point-location interpretation.

The first design-BH exposure is priority-cell based: among R1 2022 district applicants, non-priority selection probability is 2/620 = 0.0032 and priority selection probability is 366/1,282 = 0.2855. The script constructs `pi_r1_bh_priority`, `z_r1_recenter_bh`, and KNN exposure variables such as `w6_r1expbh_tm1_n` and `w6_r1rcbh_tm1_n`.

## Appendix A: Spatial-Peers Construction

### Unit Classes and Focal Sample

The spatial peer object should be conservative about the focal decision-maker. The cleanest focal unit is a regular public school district: it is a geographically meaningful district, appears in the EDGE school-district boundary file, and has a plausible district-level procurement/adoption decision. Independent charter districts are also plausibly school districts, but they are institutionally different enough that they should be a sensitivity or extension rather than the baseline. Specialized public districts, service agencies, supervisory unions, state/federal agencies, other LEAs, and non-LEA/private entities are not clean focal peer units for the main causal estimand.

The preparation layer now carries explicit flags:

- `focal_regular_public`: regular public district, including component and non-component districts.
- `focal_charter`: independent charter district.
- `focal_specialized_public`: specialized public school district.
- `focal_public_district_like`: regular public or independent charter district.
- `focal_agency_like`: service agency, supervisory union, other local education agency, state operated agency, or federal operated agency.
- `focal_nonlea`: non-LEA entity.
- `focal_unit_class`: compact string class for auditing.
- `contiguous_us`: lower-48 plus DC sample flag, excluding AK, HI, AS, GU, MP, PR, and VI.
- `main_estimation_sample`: regular public, EDGE geometry, full core controls, and contiguous U.S.

This distinction matters because the WRI point fallback adds many entities that are not district-like focal units. Of the 6,420 point-fallback entities, the largest district-like group is independent charter districts, followed by a much smaller set of regular public districts. Service agencies, supervisory unions, state/federal agencies, private fleets, private schools, nonprofits, and municipalities should not anchor the main peer-effect interpretation. They may still be useful for descriptive coverage or mechanism checks about vendor/supply-side activity.

### Isolation and KNN Interpretation

The KNN graph always assigns neighbors. For geographically remote districts, this can create artificial peer exposure over very long distances. The current diagnostics show that the severe isolation cases are concentrated in Alaska, Hawaii, Puerto Rico, American Samoa, other territories, and a few large western districts. This is a graph-design issue, not a missing-data issue.

The preparation layer now adds distance/isolation flags for both K6 graphs:

- `hybrid_w6_nearest_mi` and `hybrid_w6_max_mi`.
- `hybrid_isolated_near50` and `hybrid_isolated_k6_50`.
- `edge_w6_nearest_mi` and `edge_w6_max_mi`.
- `edge_isolated_near50` and `edge_isolated_k6_50`.
- `main_noisol_edge_k6_50`: main sample after dropping districts whose sixth EDGE neighbor is more than 50 miles away.

Baseline recommendation: estimate the main tables on `main_estimation_sample == 1`, use EDGE-only K6 exposure as the cleanest geographic peer graph, and report robustness using hybrid K6, all-U.S. geography, and the no-isolated-K6 sample. The point fallback layer should be kept for diagnostics and robustness, but the main causal interpretation should not rely on non-geographic or non-district focal units.

Current flag audit after adding these fields: the WRI point-fallback universe has 4,225 charters, 547 regular public districts, 442 specialized public districts, 1,184 agency-like entities, and 22 non-LEA entities. The EDGE universe is overwhelmingly regular public districts: 12,986 of 13,083 districts. The recommended main sample has 12,683 districts and 88,781 district-years, with 1,303 first-award events. Dropping districts whose sixth EDGE neighbor is more than 50 miles away leaves 12,433 districts and 87,031 district-years, with 1,276 first-award events.

## 2026-04-28 Build Note: Sample Selection and Spec A

Implemented the first Stata outputs:

1. `2_Scripts/2_Preliminaries/01_descriptive_statistics.do` writes `00_sample_selection.tex/.csv`.
2. `2_Scripts/3_Estimation/01_specA_hazard_panel_ols.do` writes `01_specA_hazard_panel_ols.tex/.csv`.
3. `2_Scripts/3_Estimation/00_run_all.do` now runs only the implemented sequence: sample selection and Spec A.

The main first-award hazard risk set has 85,972 district-years, 12,642 districts, and 1,303 first-award events. The no-isolated K6 risk-set robustness has 84,285 district-years, 12,392 districts, and 1,276 first-award events.

Spec A uses the first-award hazard as the outcome and the number of six EDGE-nearest neighboring districts with first awards by `t-1` as the peer-exposure regressor. The coefficient is positive across all descriptive OLS variants:

- State FE plus controls: 0.0046, SE 0.0017.
- District FE plus year FE: 0.0162, SE 0.0020.
- District FE plus year FE plus own rebate-win timing controls: 0.0128, SE 0.0016.
- District FE plus state-year FE plus own rebate-win timing controls: 0.0055, SE 0.0017.
- No-isolated-K6 robustness: 0.0128, SE 0.0016.

Interpretation: this is a strong spatial fact, not causal identification. The coefficient remains positive after district fixed effects and after state-by-year shocks, but the variation is still endogenous to local time-varying demand, vendor targeting, charger/infrastructure constraints, and policy diffusion. This table should motivate the design-IV specs rather than serve as the main claim.
