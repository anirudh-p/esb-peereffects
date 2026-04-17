import re

with open('c:/BC PhD/Research/Peer-Effects and Adoption/Running_Project_Draft.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Update TOC
text = text.replace('- [II. Literature Review](#literature-review)', '- [II. Literature Review](#literature-review)\n- [III. Institutional Context](#institutional-context)')
text = text.replace('- [III. Data Sources](#data-sources)', '- [IV. Data Sources](#data-sources)')
text = text.replace('- [IV. Summary Statistics and Preliminary Data Analysis](#summary-statistics-and-preliminary-data-analysis)', '- [V. Summary Statistics and Preliminary Data Analysis](#summary-statistics-and-preliminary-data-analysis)')
text = text.replace('- [V. Identification and Estimation Strategy](#identification-and-estimation-strategy)', '- [VI. Identification and Estimation Strategy](#identification-and-estimation-strategy)')
text = text.replace('- [VI. Results and Discussion](#results-and-discussion)', '- [VII. Results and Discussion](#results-and-discussion)')
text = text.replace('- [VII. Conclusion](#conclusion)', '- [VIII. Conclusion](#conclusion)')

# Replace headers in text
text = text.replace('## III. Data Sources', '## IV. Data Sources')
text = text.replace('## IV. Summary Statistics and Preliminary Data Analysis', '## V. Summary Statistics and Preliminary Data Analysis')
text = text.replace('## V. Identification and Estimation Strategy', '## VI. Identification and Estimation Strategy')
text = text.replace('## VI. Results and Discussion', '## VII. Results and Discussion')
text = text.replace('## VII. Conclusion', '## VIII. Conclusion')

institutional_text = """<a id="institutional-context"></a>
## III. Institutional Context

### 1. School Buses and Local Educational Agency Procurement
The U.S. school bus fleet comprises nearly 500,000 vehicles, making it the largest mass transit fleet in the country. Ownership models across Local Educational Agencies (LEAs) generally fall into two categories: district-owned-and-operated fleets, and contractor-operated fleets (e.g., First Student, Student Transportation of America). In both models, the cost of transitioning from diesel to Electric School Buses (ESBs) represents a massive capital barrier. While a traditional diesel bus costs approximately $100,000, a new ESB often ranges from $350,000 to $400,000. 

Beyond the vehicle premium, the transition imposes severe administrative and infrastructural burdens. LEAs must coordinate complex EV Supply Equipment (EVSE) installations, negotiate municipal grid upgrades with local utilities, manage trenching and construction, and retrain mechanics and drivers. Because school district procurement is highly decentralized, subject to municipal bond limitations, and often managed by under-resourced transportation departments, these administrative frictions dictate the pace of electrification as heavily as budget constraints do.

### 2. The EPA Clean School Bus Program (CSBP)
To overcome these barriers, the 2021 Bipartisan Infrastructure Law (BIL) established the EPA Clean School Bus Program, a historic $5 billion investment allocated over five years (FY2022 to FY2026). The CSBP distributes funds through both Rebates (lotteries) and Grants (competitive scoring) to replace diesel buses with zero-emission alternatives. 

To ensure equitable distribution, the program stratifies applicants into "Priority" and "Non-Priority" tiers based on income, rurality, and tribal status. Priority districts receive larger per-bus subsidy maximums and highly favorable odds in the rebate lotteries.

The timeline of key early CSBP events forms the backbone of the empirical design:
*   **May - August 2022:** The FY2022 CSB Rebate (Round 1) application window opens and closes.
*   **October 2022:** Round 1 Awards ($965 million) are announced to nearly 400 school districts via a conditionally randomized lottery system weighted toward priority districts. *This serves as our exogenous treatment shock.*
*   **April - August 2023:** The FY2023 CSB Grant (Round 2) competitive application window opens and closes.
*   **September 2023 - February 2024:** The FY2023 CSB Rebate (Round 3) application window opens and closes. 
*   **January 2024:** Round 2 Grant Awards ($965 million) are announced.
*   **May 2024:** Round 3 Rebate Awards ($900 million) are announced.

### 3. The Subsidy Lifecycle and Implications for Peer Effects
While the CSBP provides the capital, the procurement timeline is notoriously protracted. A single adoption event moves through several distinct granular stages:
1.  **Application:** The LEA formally submits intent, requiring active SAM.gov registration (a notable administrative hurdle for small districts) and initial utility coordination.
2.  **Selected / Awarded:** The EPA formally selects the district for the rebate/grant. 
3.  **Ordered (Payment Request Phase):** The LEA submits proof of a finalized Purchase Order (PO) for the buses and charging infrastructure. At this stage, adoption is financially committed.
4.  **Delivered / Operating:** The manufacturer physically delivers the buses to the district. Industry supply bottlenecks mean lead times often span 9 to 18 months after the PO.
5.  **Closeout / Scrappage:** The final stage requires districts to physically scrap the engine block of the replaced diesel bus to clear final EPA compliance.

These granular stages create a sharp conceptual divergence for the mechanism of **peer effects**. If peer effects operate primarily through *salience*—where parents and superintendents are inspired by seeing a physical electric bus driving in the neighboring town—then the peer spillover should only trigger **after** the "Delivered" stage. 

However, if peer effects operate through *administrative knowledge spillovers*—where a district learns how to conquer SAM.gov, negotiate with utility monopolies, and structure vendor contracts by talking to a neighboring transportation director—this spillover can trigger immediately after the neighbor completes the "Application" or "Awarded" stages. Given the 12-to-18 month lag between awards and delivery, mapping the response timing exactly against these institutional milestones enables the empirical design to disentangle physical salience from administrative learning.

"""

marker = '<a id="data-sources"></a>'
if marker in text:
    print('Found marker! Injecting text!')
    text = text.replace(marker, institutional_text + marker)
else:
    print('Marker not found.')

with open('c:/BC PhD/Research/Peer-Effects and Adoption/Running_Project_Draft.md', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done!')