import re

with open('c:/BC PhD/Research/Peer-Effects and Adoption/Running_Project_Draft.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Try to find the event study block using regex or a simpler substring
start = text.find('### 5. Dynamic Reduced Form (Event Study)')
end = text.find('### 6. Robustness to Spatial Bandwidth')

if start != -1 and end != -1:
    print("Found block!")
    
    new_event_study = """### 5. Dynamic Reduced Form (Event Study)
To validate the parallel trends assumption critical for our strategy, we estimate a dynamic reduced form (event study) regressing focal district adoption on the interactions between the cross-sectional neighbor R1 winner count ($Z_i$) and year dummies. Crucially, we now omit **2022** as the excluded reference year. Because the EPA CSB R1 lottery outcomes were not announced until late 2022 (October), any early 2022 focal district adoption behavior was mechanically locked in prior to the realization of the neighbor's treatment status. Anchoring the event study baseline to 2022 properly aligns the estimation with the institutional friction in the timeline.

As shown in Figure 1, the point estimates on the pre-treatment interactions (2018–2021) remain statistically indistinguishable from zero, confirming that districts situated near eventual R1 winners did not exhibit pre-existing differential adoption trends compared to districts near R1 losers. Following the announcement of the R1 winners, the 2023 coefficient remains null ($p \approx 0.916$), reflecting the extreme structural latency of preparing new grant applications. However, we observe a sharp, concentrated, and significant ($p = 0.016$) spike in adjacent-district adoption hazard isolated strictly to 2024. 

![Reduced Form Event Study](3_Output/Figures/reduced_form_event_study.png)
*Figure 1. Dynamic reduced form plot showing the effect of having neighbors win the late-2022 EPA R1 lottery on a focal district's likelihood of adopting their first electric school bus. 2022 serves as the excluded reference year.*

The precise 2024 timing of this divergence cleanly aligns with the bureaucratic processing latency (the 18-month grant cycle) required for neighboring districts to observe the fall 2022 R1 peers, prepare an application for the next round during 2023, and subsequently receive their own awards in early 2024. While we must continue to interrogate the Exclusion Restriction (e.g., confirming EPA rounds 2 and 3 did not systematically target areas adjacent to R1 winners for arbitrary administrative reasons), the sharply localized spatial decay demonstrated in Section 6 strongly corroborates this administrative learning channel.

"""
    text = text[:start] + new_event_study + text[end:]
    
    with open('c:/BC PhD/Research/Peer-Effects and Adoption/Running_Project_Draft.md', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Replaced!")
else:
    print("Could not find the block boundaries.")
