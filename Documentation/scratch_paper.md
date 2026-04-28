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
