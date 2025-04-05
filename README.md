# Public-Sector-Data-Demand-Research-NLP

## Methodology

This project analyzes public sector job postings to identify roles that are either explicitly or implicitly involved in third-party data acquisition — referred to throughout this report as “data buyers.” Using a blend of traditional keyword detection, fuzzy matching, rule-based tagging, and predictive modeling via natural language processing (NLP), this approach surfaces both overt and hidden demand for external data tools and services.

The goal is to provide commercial data vendors with a replicable, high-precision strategy for identifying public sector buyers aligned to specific strategic use cases: fraud detection, sentiment analysis, patient record matching, and ad targeting.

---

### Data and Scope

The dataset includes 5,829 public sector job postings from U.S. government agencies. Each record contains:

- Unstructured text: `JobTitle`, `JobDescription`, `SearchKeywords`, `KeyDuties`
- Categorical features: `Agency`, `Department`, `AgencySize`, `Industry`
- Derived features: `DataBuyerScore`, `FuzzyMatchedPhrase`, `IsLikelyDataBuyer`

Analysis was conducted using two primary notebooks:
- `AA BSD Analysis.ipynb` – data cleaning, keyword labeling, sector tagging
- `AA BSD Update (1).ipynb` – NLP modeling, use case mapping, statistical tests

---

### Labeling Explicit Data Buyers

Initial classification used a **hybrid strategy**:

1. **Keyword detection**: targeting phrases like:
   - “third-party data”
   - “data vendor”
   - “data subscription”
   - “procured dataset”
   - “market intelligence”
2. **Fuzzy matching**: to capture variants such as “third party provider” or “external analytics vendor.”
3. **Manual review**: to remove false positives, e.g., “Pharmacy Technician” or “Dental Assistant,” which sometimes triggered due to ambiguous phrases like “system management” or “data entry.”

This created a labeled set of **explicit data buyers** that became the foundation for NLP model training and pattern detection.

---

### Sector and Use Case Classification

Each job was tagged to one or more of the following sectors:
- **Health & Medical**
- **Finance & Oversight**
- **Communications & Outreach**
- **Policy & Legal**
- **Technology & Security**

In addition, each posting was classified into up to four high-value **use cases**:

| Use Case               | Example Keywords                                                   |
|------------------------|---------------------------------------------------------------------|
| Fraud Detection        | “eligibility verification,” “audit,” “fraud risk,” “screening”     |
| Sentiment Analysis     | “public opinion,” “media analytics,” “feedback loop”              |
| Patient Record Matching| “EHR,” “interoperability,” “data linkage,” “record integration”    |
| Ad Targeting           | “audience segmentation,” “campaign optimization,” “targeting data” |

---

### Natural Language Processing (NLP) Model

To go beyond keyword logic, a **logistic regression NLP model** was trained on the explicitly labeled data buyers. This allowed us to classify additional roles based on how similar their text was to confirmed buyer jobs.

#### Features:
- TF-IDF vectorized text (from title, description, and keywords)
- N-gram tokens (1- to 3-word combinations)
- Structured metadata: `AgencySize`, `Industry`, `DataBuyerScore`, `IsSeniorRole`

#### Outputs:
- A probability score (`PredictedDataBuyer`) for each job
- Token weight coefficients, identifying influential terms

This model allowed us to surface **latent data buyers** — jobs that don't use explicit terms but share linguistic patterns with confirmed buyers.

---

#### Oversight and False Negatives

While the model performed well overall, a notable subset of **false negatives** emerged — roles that should not have been labeled as likely data buyers, yet were mistakenly flagged either through keyword matches or overgeneralized text patterns. These included:

- **Pharmacy Technician**
- **Dental Assistant**
- **Nursing Assistant**
- **Motor Vehicle Operator**
- **Custodian**
- **Food Service Worker**

These roles often contain language about “inventory,” “records,” or “systems,” but these refer to clinical or operational workflows rather than third-party data use.

To improve precision:
- These titles were **manually removed from the buyer class** for all downstream statistical testing, modeling evaluation, and visualizations.
- A refined set of **exclusionary keywords** was created to suppress these roles in future applications.

> **Recommendation**: Continued human-in-the-loop review — especially from personnel with field expertise — is important to guide future tuning. Expert oversight can help identify new use case language, evolving terminology, and overlooked sectors.

Together, the NLP model and manual refinement process enable a reliable, adaptable framework for classifying and understanding public sector demand for external data.

---

### Model Deployment and Flexibility

The entire classification and scoring pipeline has been designed for **scalability**. A **full API-call–ready codebase** is available, enabling:

- Real-time job scoring and ingestion
- Integration into CRM systems or lead qualification dashboards
- Periodic retraining or keyword updates without code rewrites

**Keyword libraries** used for inclusion and exclusion are fully modular and can be **easily adapted to evolving use cases** — such as open location data, AI readiness, or health analytics.

---

### Additional Considerations

- **Posting Seasonality**: Hiring volume shifts over fiscal quarters; model should be rerun periodically.
- **Role Ambiguity**: A “Program Analyst” in CDC is not the same as in GSA — departmental context matters.
- **Proxy vs. Confirmation**: NLP scores and keyword matches reflect *intent*, not confirmed contract data. Still, strong patterns emerge across departments.
- **Evolving Terminology**: Language around “LLMs,” “synthetic data,” or “real-time insight” is starting to appear — the keyword and model pipeline is built to evolve with these trends.

---

## Market Analysis: Public Sector Demand for Third-Party Data

### Overview

While only a fraction of public sector jobs explicitly mention third-party data vendors, many reference tasks, platforms, or responsibilities that imply it. This analysis surfaces which **roles**, **sectors**, and **agencies** are most likely to be engaging in data buying, either directly or through embedded programmatic work.

---

### Use Case Alignment by Agency

| Use Case               | High-Aligned Agencies                             | Buyer Role Share (%)       |
|------------------------|---------------------------------------------------|-----------------------------|
| Fraud Detection        | SSA, GSA, VA                                       | 60–75%                      |
| Sentiment Analysis     | FEMA, HHS, CDC                                     | 40–60%                      |
| Patient Matching       | VA Medical, IHS, DoD Medical                       | 70–80%                      |
| Ad Targeting           | CDC Foundation, HHS Comms, FEMA                    | 35–45%                      |

---

### Sector Trends

| Sector              | Dominant Use Cases         | Representative Agencies            |
|---------------------|----------------------------|------------------------------------|
| Health              | Patient Matching, Fraud    | VA, IHS, NIH                       |
| Finance             | Fraud Detection            | SSA, Treasury, GSA                 |
| Communications      | Sentiment, Ad Targeting    | FEMA, CDC, HHS Comms Office        |
| Policy & Legal      | Sentiment, Compliance      | DOJ, State Department              |
| Tech & Security     |Fraud Detection     | DoD, NSA, DHS                      |

---

### Key Roles and Latent Buyers

Roles flagged by the model as likely data buyers — even without “data” in the title:

- Contract Specialist
- Program Analyst
- Management Analyst
- Grants Officer
- Public Affairs Officer
- Health IT Coordinator
- Communications Strategist
- Budget Officer

These roles frequently involve vendor management, platform evaluation, fraud analysis, campaign tracking, and operational analytics — all indicating third-party data use.

---

### Token Insights and Role Patterns

Text patterns extracted from the NLP model reveal what kinds of language most commonly appear in likely data buyer postings — and what kinds of phrases typically indicate non-buyers.

#### Top Predictive Tokens:
- “procured”
- “external vendor”
- “data subscription”
- “market intelligence”
- “platform integration”
- “analytics tools”

#### Negative Signals:
- “patient bathing”
- “meal preparation”
- “linen services”
- “custodial duties”
- “vehicle operation”

These were common in support or clinical jobs that mention records or systems, but not in the context of analytics or procurement. These jobs were originally responsible for many false positives and were filtered out manually.

---

### Statistical Patterns

Several significant patterns emerged across the classified and scored dataset:

- **Seniority matters**: Jobs flagged as `IsSeniorRole = 1` were significantly more likely to be data buyers (*Welch’s t-test, p < 0.01*), particularly in smaller agencies with limited internal analytics capacity.
- **Agency size effect**: Small and medium-sized agencies were more likely to embed data responsibilities in generalist or administrative roles (e.g., program coordinators, contract officers) instead of standalone analyst positions.
- **Kurtosis in buyer score**: The `DataBuyerScore` exhibited **high kurtosis (>4)** — suggesting that most public jobs are definitively non-buyers, while a small, distinct cluster exhibit strong signals of buyer behavior.
- **False positive clusters were removed**: Operational roles like “Pharmacy Technician” and “Food Service Worker” were initially over-flagged but manually excluded to improve model precision.

---

### Data Buyer Volume and Concentration by Agency and Sector

In addition to use case and title-level signals, we evaluated which **agencies and sectors** are most strongly associated with data buyers.

#### Agencies with the Most Data Buyer Jobs (Total Count)

| Agency                              | Data Buyer Jobs |
|-------------------------------------|------------------|
| Department of Veterans Affairs (VA) | 263              |
| Department of the Army              | 37               |
| Department of the Navy              | 35               |
| Department of the Air Force         | 32               |
| Department of Homeland Security     | 24               |

The VA dominates in both hiring volume and data buyer count, reflecting its operational scale and emphasis on health record systems, fraud controls, and vendor-based analytics platforms.

#### Agencies with Highest % of Jobs as Data Buyers

Agencies with a **smaller total number of postings** but a **high proportion** of data buyer roles include:

- Legislative Branch (e.g., research and communications offices)
- FEMA and CDC (especially in communications, ad targetting, and outreach)
- Department of Justice (in legal compliance and fraud)

These agencies should not be overlooked simply because of lower job counts — they often rely heavily on **external data vendors to fulfill narrow mission-critical functions**.

---

#### Sector-Level Trends by Buyer Concentration

While **Health and Finance** sectors had the **highest volume** of data buyer roles overall, **Communications and Policy** showed the **highest proportion** of buyer roles relative to total hiring.

This suggests:
- Communications teams are becoming major consumers of third-party data (e.g., for media tracking, sentiment analysis, and ad targeting).
- Policy groups engage data for monitoring impact, audience feedback, and legislative decision support.
- Security and IT roles, while large in number, more often use internally generated data or manage infrastructure, not procure third-party sources.

---

### Filtering Strategy: Precision Targeting

**Positive signals to track:**
- “data subscription”
- “vendor-supplied”
- “analytics vendor”
- “platform integration”
- “audience segmentation”
- “procured dataset”

**Negative signals to filter:**
- “meal prep”
- “linen delivery”
- “patient hygiene”
- “custodial support”
- “motor vehicle operation”
- Titles with “Technician” or “Assistant” in non-technical settings

---

### Strategic Recommendations

| Use Case           | Target Agencies                  | Product Opportunity                       |
|--------------------|----------------------------------|--------------------------------------------|
| Fraud Detection    | SSA, VA, GSA                     | Eligibility checks, ID verification        |
| Sentiment Analysis | FEMA, HHS, CDC                   | Public engagement, media analytics         |
| Patient Matching   | VA, IHS, DoD Medical             | Interoperability tools, health ID systems  |
| Ad Targeting       | CDC Foundation, FEMA, HHS Comms  | Segmentation, outreach optimization        |

**Go-to-market guidance**:
- Prioritize hybrid roles and generalists in small or mid-sized agencies.
- Use job scoring outputs for lead segmentation and follow-up prioritization.
- Monitor changes in job language using keyword tracking + token weights.

---

### Conclusion

Data buyers in government are not always called “data analysts.” They are Program Managers, Contract Officers, and Health IT Leads managing real-world platforms and workflows that depend on third-party data. 

This analysis combines keyword matching, machine learning, and contextual classification to help vendors find these roles — and act on them. With an adaptable pipeline and strong textual patterns, this framework is not only accurate today, but built to evolve with the market.

