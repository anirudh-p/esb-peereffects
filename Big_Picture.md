The right way to do this is to **separate the causal chain into layers**. Right now, the phrase “peer effects” is doing too much work. You want to distinguish:

1. **The treatment of interest**: neighbors adopt ESBs
2. **The reduced-form causal effect**: my district becomes more likely to adopt
3. **The mechanism/channel**: *why* neighbors’ adoption changes my behavior

That distinction will help you decide what the main estimating equations are, what is just supporting evidence, and what assumptions each piece needs.

---

# 1. Start with the descriptive causal chain

A clean version is:

## Core causal chain

**Neighbor receives exogenous encouragement to adopt**
→ **Neighbor successfully secures funding (awarded) and/or begins operating ESBs**
→ this changes the **information environment** facing nearby districts
→ focal district updates beliefs / lowers perceived fixed costs / sees feasibility / learns operationally
→ focal district is more likely to **apply / adopt / operate**

That already suggests an important distinction:

* **Peer effect** = effect of neighbors’ adoption/operation on my own adoption
* **Mechanism/channel** = the intermediate margin through which that happens

So your paper’s logic can be framed as:

> I first establish that neighbors’ ESB adoption causally affects a focal district’s own adoption behavior. I then probe mechanisms by testing which kinds of neighbor exposure matter most and which downstream margins in the focal district respond.

---

# 2. What exactly is a “channel”?

This is where people often get fuzzy.

A **channel** is not just “another outcome.”
A channel is a **specific intermediate pathway** linking peer adoption to own adoption.

For your setting, I think the cleanest candidate channels are:

## A. Information / administrative learning

Neighbors help districts understand:

* grant application process
* procurement steps
* vendor selection
* charging/infrastructure coordination
* maintenance/training requirements
* total cost and timelines

This is a **knowledge transfer** channel.

### Observable implications

If this is the channel, peer exposure should matter more for:

* districts with less prior capacity
* early-stage margins like **application**
* exposure to **nearby peers who successfully navigated funding and paperwork** (e.g., lottery winners), even before buses arrive
* maybe same-state or same-administrative environments

---

## B. Operational learning / demonstration

Seeing buses actually in service reduces uncertainty about:

* route feasibility
* charging logistics
* winter performance
* maintenance
* driver acceptance
* school board/community concerns

This is more “learning from observed operation” than from paperwork.

### Observable implications

If this is the channel, effects should be stronger for:

* **operating** neighbors, not just awarded neighbors
* more recent visible deployments
* geographically proximate neighbors
* climates / route conditions that are similar

---

## C. Social imitation / legitimacy / political cover

A neighbor’s adoption may make adoption feel legitimate or expected:

* “districts like us are doing this”
* school boards feel safer approving
* administrators face less perceived career risk

This is a softer emulation channel.

### Observable implications

If this is the channel, effects may arise even when:

* there is limited deep information sharing
* nearby adoption is merely visible/publicized
* “peer group similarity” matters a lot

---

## D. Market-building / ecosystem channel

Neighbor adoption may induce local complementarity:

* dealers/vendors pay more attention to the region
* infrastructure providers become more active
* service/maintenance support improves
* financing/procurement templates diffuse

This is not purely informational; it is a **local equilibrium / supply ecosystem** channel.

### Observable implications

Effects may be stronger:

* when local adoption density crosses thresholds
* over slightly broader geography than immediate peers
* in later periods after ecosystem buildout

---

# 3. The main empirical distinction you need

You should distinguish three levels of empirical claims:

## Claim 1: Localized spillover / Peer effect exists

Exogenous shocks to neighbors' adoption resources causally increase focal adoption.

This is your **main causal identification result**.

## Claim 2: Which exposure matters?

Is it:

* winning neighbors?
* adopting neighbors?
* operating neighbors?
* nearby visible peers?
* peers in similar conditions?

This is **mechanism discrimination**, but still not full mediation analysis.

## Claim 3: What focal margin responds first?

Do peers affect:

* application?
* adoption?
* operation?
* fleet share?

This helps map the channel.

This is important because in your setting, “mechanism tests” probably will not be a formal mediation decomposition. They will be **mechanism-consistent patterns**.

That is okay. In many peer effects papers, that is exactly what mechanism evidence looks like.

---

# 4. A useful conceptual decomposition

I would define the objects like this.

## Treatment object

Let ($A_{jt}$) be whether district ($j$) adopts ESBs by time ($t$).

## Peer exposure

Let
$$
P_{it} = \sum_{j \neq i} w_{ij} A_{jt}
$$

where ($w_{ij}$) is your geographic or other peer weight.

This is the endogenous peer exposure of real interest.

## Channel-specific exposure

Now decompose peer exposure into types:

$$
P_{it}^{\text{award}},\quad P_{it}^{\text{operate}},\quad P_{it}^{\text{same-state}},\quad P_{it}^{\text{similar-climate}},\quad P_{it}^{\text{priority-group}}, \ldots
$$

These are not all separate “peer effects”; rather they are alternative versions of **which peer exposure carries the effect**.

That distinction matters.

---

# 5. What should the main set of estimating equations be?

I would strongly recommend organizing them into **three blocks**.

---

## Block I. Main peer-effect equation

This should answer the headline question:

> Does neighboring adoption causally affect own adoption?

A clean structural reduced-form target is:

$$
Y_{it} = \alpha + \beta P_{it-1} + \Gamma' X_{it} + \lambda_t + \mu_i + \varepsilon_{it}
$$

where:

* $Y_{it}: $ focal district outcome
* $P_{it-1}:$ lagged peer adoption exposure
* $\lambda_t:$ time FE
* $\mu_i:$ district fixed effects (crucial for a setting with spatial spillovers)

### Preferred outcomes

Use one main outcome only for the headline:

* **Adoption by end of period**
  or
* **Adoption hazard / first adoption in year (t)**

Given your setting, I think the cleanest main outcome is probably:

$$
Y_{it} = \mathbf{1}\{ \text{district } i \text{ applies / gets awarded in } t \}
$$

because it aligns with diffusion timing.

But if timing is coarse/noisy, a simpler cumulative outcome is fine:

* adopted by end-2024
* first adoption by (t)

### Endogeneity problem and Manski's Reflection

The structural OLS equation above is heavily endogenous and suffers from Manski's (1993) reflection problem. We must distinguish between:
1. **Endogenous Peer Effects (The Target):** A neighbor's adoption *causes* the focal district to adopt. This implies a **Social Multiplier**—if policymakers subsidize one district, neighbors adopt organically, maximizing program cost-effectiveness. 
2. **Correlated Effects (The Threat):** Neighbors adopt simultaneously because they share similar unobserved environments (e.g., regional "green-friendliness", a new local vendor, or a shared air-quality ruling). This implies **No Multiplier**—subsidizing one district does not affect its neighbors; the policy would have to target each district independently.
3. **Contextual Effects:** A district adopts because its neighbors are wealthy or large, irrespective of their actual ESB adoption.

This is why **Two-Way Fixed Effects (District + Year)** ($\mu_i$, $\lambda_t$) are vital—district FEs absorb static correlated spatial and contextual effects. However, since unobserved correlated shocks can still occur dynamically, $P_{it-1}$ remains endogenous.

### Defining Peer Exposure

Constructing $P_{it-1}$ requires careful definition along two sets of dimensions:
1. **Who is a peer?** Geographic proximity (KNN or radius matrices) is the defensible baseline for physical buses routing and local vendors. However, alternative boundaries (same-state, similar capacity strata) help test administrative channels.
2. **Status of Adoption (Vintages):** "Awarded", "Ordered", "Delivered", and "Operating" are simply distinct stages—often separated by years—of the exact same core adoption choice. Thus, lagged peer exposure measures become highly collinear over time. 
3. **Lagged vs Contemporaneous Timing:** Using the lag, $P_{it-1}$, mechanically breaks the worst part of Manski's reflection problem (I adopt today because you adopt today, and vice-versa). However, it **does not solve** the presence of correlated effects. Even if a local vendor shocked your neighbor to adopt at $t-1$, and then caused you to adopt at $t$, your lag variable will show a strong spurious correlation. 

Because unobserved spatial shocks are highly serially correlated across time, the OLS lag remains hopelessly endogenous. So this is where your IV comes in.

---

## Block II. IV strategy for peer exposure

Your first stage should isolate exogenous variation in peer adoption coming from neighbors’ lottery wins.

If $Z_{it}$ is weighted share of neighbors who won the R1 lottery, then:

### First stage

$$
P_{it-1} = \pi_0 + \pi_1 Z_{it} + \Pi' X_{it} + \lambda_t + \mu_i + \nu_{it}
$$

### Second stage

$$
Y_{it} = \alpha + \beta \widehat{P}_{it-1} + \Gamma' X_{it} + \lambda_t + \mu_i + \varepsilon_{it}
$$

This is the core causal design.

### Interpretation

$\beta$ is the causal effect of neighbors’ induced adoption on focal adoption for the compliers generated by lottery exposure.

That is your main peer-effect estimate.

---

# 6. What should count as “mechanism” equations?

Here is the key move:

You do **not** want mechanism to become a grab-bag of every alternative dependent variable.

Instead, define mechanism empirically as one of two things:

## Type A. Exposure-side mechanism tests

Which kind of neighbors matter?

A naive approach would be to run a "horse race" regression:

$$
Y_{it} = \alpha + \beta_1 P_{it}^{\text{operate}} + \beta_2 P_{it}^{\text{award}} + \Gamma' X_{it} + \lambda_t + \mu_i + \varepsilon_{it}
$$

**Do not do this.** You cannot cleanly separate these in a single equation. You only have *one* main instrument (R1 Lottery Wins) that shocks *both* awarded and operating statuses, inherently causing severe underidentification if you try to use it twice simultaneously. Furthermore, because operating is mechanically just a delayed vintage of being awarded, $P^{\text{operate}}$ and $P^{\text{award}}$ are powerfully collinear. 

Instead, separate these mechanisms empirically using:

1. **Reduced Form Timing (Event Studies):** Because R1 is a sharp point-in-time shock, the best way to decouple "award" vs "operate" mechanisms is *time*. If focal districts respond in the identical year ($t$) their neighbor wins the lottery (before any buses are built), the time response is *highly consistent with* administrative learning or early-stage mobilization. If they only respond 2-3 years later when the physical buses arrive, it strongly supports operational visibility.
2. **Separate Structural IV Models:** Run the 2SLS using *only* Awarded, and separately using *only* Operated. The actual strength of the First-Stage F-stats and parameter estimates will mechanically reveal which constraint is empirically binding (i.e. did the lottery induce early physical operations, or primarily just administrative completions?).

---

## Type B. Outcome-side mechanism tests

What focal margin responds first?

Example outcomes:

* application
* adoption
* operation
* fleet share
* charging installation, if observed

Then estimate:

$$
Y^{m}_{it} = \alpha_m + \beta_m \widehat{P}_{it-1} + \Gamma_m' X_{it} + \lambda_t + \mu_i + \varepsilon^m_{it}
$$

for different ($m$).

Interpretation:

* If peer exposure raises **applications**, that points to early-stage information/administrative learning
* If peer exposure mostly affects **operation** or post-award follow-through, that suggests operational feasibility learning or ecosystem support

This is not “proving mediation,” but it is persuasive channel evidence.

---

# 7. A practical taxonomy for your paper

I would recommend the following language.

## Main causal effect

**Peer adoption effect**
Effect of neighboring districts’ ESB adoption on own district adoption.

## Mechanism family 1: Knowledge transfer

Includes:

* application know-how
* procurement know-how
* grant administration
* infrastructure planning
* vendor/process learning

## Mechanism family 2: Operational demonstration

Includes:

* learning from real bus deployment
* route feasibility
* maintenance/charging experience
* climate/terrain performance

## Mechanism family 3: Social legitimacy

Includes:

* political cover
* reduced perceived risk
* imitation of similar districts

## Mechanism family 4: Local ecosystem

Includes:

* vendor presence
* service capacity
* charging support
* procurement templates/network effects

This gives you a disciplined way to say:

> I cannot perfectly observe all channels directly. Instead, I test for patterns consistent with these mechanism families.

That is much stronger than vaguely saying “mechanisms.”

---

# 8. What not to call a mechanism

A few things are better thought of as **measurement choices** or **timing choices**, not mechanisms:

* using operating neighbors instead of awarded neighbors
* using KNN-6 vs KNN-20
* using same-state peers
* using cumulative vs current exposure

These are not mechanisms by themselves. They are **ways to operationalize exposure that may map to different channels**.

That wording matters.

---

# 9. A strong descriptive framework you can use in the paper

You could describe the diffusion process like this:

> Neighboring adoption may affect focal districts through several conceptually distinct pathways. First, neighboring districts may transmit administrative and procurement knowledge, reducing informational and bureaucratic barriers to adoption. Second, nearby operating buses may provide demonstration effects, allowing districts to learn about route feasibility, charging, maintenance, and performance under local conditions. Third, neighboring adoption may generate social legitimacy or political cover, making adoption more acceptable to school boards and communities. Fourth, local adoption may thicken the regional support ecosystem, increasing vendor attention and service capacity. My empirical strategy first identifies the causal effect of peer adoption on own adoption, then uses the type of peer exposure and the focal outcome margin to distinguish among these mechanism families.

That is crisp and disciplined.

---

# 10. How I would organize the empirical section

## Section A. Estimand

Define the causal estimand:

* effect of neighbors’ adoption on own adoption

## Section B. Identification

Explain:

* peer adoption is strictly endogenous—even when lagged—due to spatially and serially correlated shocks.
* use exogenous neighbor lottery wins as instrument ($Z_{it}$), conditioning on applicant density / strata / state / priority structure.

### Note on IV-TWFE and Staggered Treatment DiD 
You should classify your overarching design as an **IV bound within a Two-Way Fixed Effects (TWFE) framework**.
* **Is there a "Bad Controls" TWFE problem here?** The modern DiD literature (Goodman-Bacon, Callaway & Sant'Anna) highlights that TWFE becomes severely biased (with potential negative weights) when treatments are *staggered* across time and have heterogeneous effects. While your endogenous outcome variable ($P_{it-1}$) is staggered, your **instrument variation ($Z_{it}$)** is completely point-in-time—one single shock from the R1 Lottery. Because the causal variation comes exclusively from a single random cohort assignment against a cleanly defined untouched control group, your design effectively avoids the primary negative-weighting concerns of Bacon-decomposition. *(Note: This does not mean interpreting a dynamic network exposure TWFE coefficient is trivial, but it eliminates the most fatal staggered-DiD critiques).*
* **Future Extension Warning:** If you later incorporate the R3 Lottery, your instrument *becomes* fully staggered. Standard TWFE would break immediately under the new literature. You would have to implement modern staggered-IV estimators (e.g., de Chaisemartin & D'Haultfœuille 2020) to handle the multi-cohort instrumental design.

## Section C. Main equations

Main 2SLS:

$$
P_{it-1} \leftarrow Z_{it}
$$
$$
Y_{it} \leftarrow \widehat{P}_{it-1}
$$

## Section D. Mechanism evidence

Two subparts:

### D1. Which peer exposure matters?

* winners vs adopters vs operators
* geographic proximity
* same-state / similar-climate / same-priority group

### D2. Which focal margin responds?

* application
* adoption
* operation

That is enough. Clean and convincing.

---

# 11. What the identifying assumptions look like

For the main peer-effect IV:

## Relevance

Neighbors’ lottery wins shift neighbors’ actual adoption/operation.

## Exclusion (The Ultimate Identification Threat)

Conditional on controls and fixed effects, a neighbor's lottery win affects focal adoption **only** through the neighbor's actual adoption behavior, not through some other direct channel. 

**The Core Threat:** The exclusion restriction is exceptionally fragile if the R1 win generates direct **supply-side effects** or **general salience**. If an R1 award attracts media publicity or causes an ESB vendor to flood the county with aggressive sales representatives, then the focal district adopts because of the vendor/media (a direct effect of the instrument), not because they learned from their peer.

**How to assess and address this (embrace the complexity):**

1. **Define Two Distinct Estimands ("The Motte and Bailey"):**
   * **Estimand A (Reduced-form / ITT):** The effect of exogenous neighbor R1 success on focal adoption. This identifies a **Localized Policy Spillover**. Even if supply-side intermediaries play a role, the fact that a federal grant dropped into a geography causes neighboring districts to adopt is credible, robust, and highly policy-relevant.
   * **Estimand B (2SLS / LATE):** The narrower effect of *neighbor adoption* induced by R1 on focal adoption. This requires strong exclusion. Rather than claiming to "prove" exclusion, we use the following strategies to assess whether broad supply-side responses can explain the results.

2. **Assessing the Vendor Threat (Using Granular Market Data):** You *cannot* control for the focal district's *chosen* vendor post-treatment (a "bad control"), but you can exploit granular supply-side constraints. By utilizing OEM and Dealer footprints available in the WRI bus-level data, you can construct **Baseline/Historical Vendor Territory $\times$ Year FEs**. While this does not rule out hyper-local dealer targeting, it absorbs broad, territory-wide supply-side pushes, allowing you to isolate spillovers occurring *within* a constant vendor environment.

3. **Decomposing the "Shared Intermediary" vs. District Learning:** A major threat to pure peer-learning is that a focal district's application is entirely driven and prepared by a third-party consultant (or vendor) activated by the neighbor's win. Because the raw EPA data tracks the precise `Applicant Organization Name`, you can empirically flag whether a subsequent application was *Self-Filed* (District-led) or *Third-Party Filed* (Intermediary-led). This allows for a powerful decomposition of the localized spillover into vendor-assisted transmission vs. horizontal district-to-district learning.

4. **Public Salience vs. Administrative Capacity:** To defend against the threat that spillovers are merely driven by localized media attention from the neighbor's lottery win, we rely on both timing diagnostics and institutional context. In this setting, public salience alone is unlikely to be sufficient to generate application and adoption responses, because ESB procurement requires substantial administrative and technical capacity. This makes pure media-attention explanations less plausible. However, such evidence cannot by itself distinguish district-to-district peer learning from intermediary-mediated transmission of application know-how.

## SUTVA vs. "Structured Interference"

Strict SUTVA (Stable Unit Treatment Value Assumption) requires that one district's treatment assignment does not affect any other district's potential outcomes. **In a peer effects paper, strict SUTVA is explicitly violated by design.** Your central research question is dedicated to measuring the magnitude of this exact violation. 

Instead of traditional SUTVA, you rely on an assumption of **Structured Interference** (also called *Local SUTVA*):
1. You assume that spillovers only occur *through your strictly defined network* (the spatial $w_{ij}$ matrix). Distant districts on the other side of the country do not interfere with the focal district.
2. The IV Exclusion Restriction becomes your primary defense. It assumes that a neighbor's lottery win ($Z_{jt}$) only affects the focal district's outcome ($Y_{it}$) *through* its effect on the neighbor's actual adoption behavior ($A_{jt}$). The lottery win itself cannot have any direct unobserved spillover effect on the focal district.

## Monotonicity

Neighbor lottery wins do not reduce neighbor adoption for the relevant compliers.

For mechanism evidence, assumptions are usually weaker because those exercises are more interpretive. You should present them as **supporting evidence**, not fully identified mediation.

---

# 12. The big conceptual warning

Do not overclaim:

> “I identify the exact mechanism.”

That is probably too strong.

A better claim is:

> “I identify a causal peer effect and provide evidence distinguishing among plausible mechanisms.”

That is much more defensible.

---

# 13. My recommendation for the actual core equations

If I had to narrow it to a **main set**, I would keep only these:

## Equation 1: First stage

$$
P_{it-1} = \pi_0 + \pi_1 Z_{it} + \pi_2 W^{app}_{it} + \Pi' X_{it} + \lambda_t + \mu_i + \nu_{it}
$$

where:

* $P_{it-1}$: weighted neighbor adoption/operation
* $Z_{it}$: weighted neighbor lottery wins
* $W^{app}_{it}$: weighted neighbor applicants, to hold application density fixed

## Equation 2: Main second stage

$$
Y_{it} = \alpha + \beta \widehat{P}_{it-1} + \Gamma' X_{it} + \lambda_t + \mu_i + \varepsilon_{it}
$$

with $Y_{it}$ as first adoption or adoption by end-period.

## Equation 3: Exposure-type mechanism equation

*(Note: As discussed in Section 6, do not run both variables at once; run as separate tests or Reduced-Form timing models)*

$$
Y_{it} = \alpha + \beta_1 \widehat{P}^{\text{operate}}_{it-1} + \beta_2 \widehat{P}^{\text{award-only}}_{it-1} + \Gamma' X_{it} + \lambda_t + \mu_i + \varepsilon_{it}
$$

Interpretation: operational learning vs mere program visibility.

## Equation 4: Focal-margin mechanism equation

$$
Y^m_{it} = \alpha_m + \beta_m \widehat{P}_{it-1} + \Gamma_m' X_{it} + \lambda_t + \mu_i + \varepsilon^m_{it}
$$

where (Y^m) is one of:

* applied
* adopted
* operating

That is enough for a coherent paper.

---

# 14. Where I think your paper should land substantively

My guess is that your strongest framing is:

## Main localized spillover

Neighbors’ exogenous R1 success (and induced adoption) increases focal adoption.

## Most plausible mechanisms

Not “peer pressure” in a vague sense, but:

* **administrative/procurement learning**
* **operational demonstration**
* possibly **political legitimacy**

I would make those your headline mechanism hypotheses.
The ecosystem story can remain a secondary channel unless you have direct measures for it.

---

# 15. The cleanest hypothesis structure

You could write:

### H1. Local adoption spillover

Exogenous encouragement of neighboring ESB adoption increases a focal district’s probability of applying/adopting.

### H2. Administrative learning

Peer effects are stronger when neighboring districts provide administratively relevant information, and should appear on earlier margins such as application behavior.

### H3. Operational demonstration

Peer effects are stronger for nearby districts with buses in operation, consistent with learning from visible real-world deployment.

### H4. Social legitimacy

Peer effects are stronger among observationally similar or salient peers, consistent with imitation or reduced political risk.

That is much tighter than “mechanism tests.”

---

# 16. My honest view on what you should avoid

I would avoid building the paper around:

* too many alternative outcomes
* too many weight matrices without interpretation
* vague phrases like “channel analysis” unless each channel has a clear empirical implication
* treating every heterogeneity test as a mechanism test

Instead, keep asking:

> What specific intermediate friction is being reduced?

That question usually clarifies the mechanism.

---

# 17. The key design question I’d want answered next

To sharpen the equations, I need to know which of these you observe reliably at the district-year level:

| Variable                              | Do you observe it?          | Variable Name & Data Source                               | Granularity                               | Why it matters                                                         |
| ------------------------------------- | ------------------------- | --------------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------- |
| Application to CSBP or other programs | **Yes**                   | `Y_R3_apply`, `IS_R1_LOSER` (EPA CSBP applicant lists)    | District-Round (or District-Year)         | identifies early-stage admin learning (Did R1 cause a focal R3 app?)   |
| Lottery win/loss                      | **Yes**                   | `is_r1_winner`, `IV_Z_R1` (EPA CSBP award lists)          | District-Year (R1 point-in-time design)   | provides the randomized instrument shock                               |
| Adoption/purchase                     | **Yes**                   | `Y_first_adopted` / `wri_awarded_by_*` (WRI ESB database) | District-Year                             | main outcome (administrative completion of adoption choice)            |
| Delivery / operation date             | **Yes**                   | `Y_first_operating`, `Y_first_delivered` (WRI Database)   | District-Year (and Quarter)               | decisively distinguishes physical/operational learning from admin      |
| Applicant Identity / Intermediary     | **Yes**                   | `Applicant Organization Name` (EPA Raw File)              | District-Round                            | decomposes district-led vs third-party-led application spillovers      |
| Dealer and OEM Footprints             | **Yes**                   | `3x. Dealer`, `3t. Bus OEM` (WRI Bus-level data)          | Bus-level / District-level                | enables Vendor Territory $\times$ Year fixed effects                   |
| Charging infrastructure               | **Maybe (Proxy)**         | Not explicitly split out in main panel yet                | TBD                                       | ecosystem/supporting mechanism evidence                                |
| District covariates                   | **Yes**                   | `nces_id`, `poverty_rate`, `urbanicity`, `pm25` (NCES)    | District-Year (Baseline Fixed)            | heterogeneity and structural control stability                         |
| Route/climate/geography               | **Maybe**                 | TBD                                                       | TBD                                       | operational learning tests                                             |
| State procurement environment         | **Maybe**                 | TBD                                                       | TBD                                       | admin-learning tests                                                   |

Once that is clear, I can help you turn this into:

1. a **one-page causal framework**,
2. the **main estimating equations**, and
3. a **short “what counts as mechanism” section** written in paper language.

