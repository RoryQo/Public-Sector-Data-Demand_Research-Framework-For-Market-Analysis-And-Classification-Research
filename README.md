# Public-Sector-Data-Demand-Research-NLP


## Methodology

This project analyzes public sector job postings to identify roles that are either explicitly or implicitly involved in third-party data acquisition — referred to throughout this report as “data buyers.” Using a blend of traditional keyword detection, fuzzy matching, rule-based tagging, and predictive modeling via natural language processing (NLP), this approach surfaces both overt and hidden demand for external data tools and services.

The goal is to provide commercial data vendors with a replicable, high-precision strategy for identifying public sector buyers aligned to specific strategic use cases: fraud detection, sentiment analysis, patient record matching, and ad targeting.

---

### Data and Scope

The dataset includes 5,829 public sector job postings from U.S. government agencies, retrieved via an API call to the publicly available USAJobs.gov database. Roughly 1/5 of these postings were flagged as likely third-party data buyers. Each record contains:

- Unstructured text: `JobTitle`, `JobDescription`, `SearchKeywords`, `KeyDuties`
- Categorical features: `Agency`, `Department`, `AgencySize`, `Industry`
- Derived features: `DataBuyerScore`, `FuzzyMatchedPhrase`, `IsLikelyDataBuyer`

Analysis was conducted using two primary notebooks:
- `Data Aquisition and Wrangling.ipynb` – data cleaning, keyword labeling, sector tagging
- `Modeling.ipynb` – NLP modeling, use case mapping, statistical tests

---

### Text Normalization and Feature Engineering

To prepare for classification and modeling, text fields were:

- Lowercased and stripped of punctuation for consistency
- Tokenized using standard NLP pipelines (`nltk`/`scikit-learn`)
- Combined into a single field (`CombinedText`) for unified parsing

From these fields, several features were created:

- `IsSeniorRole`: flagged if title included “chief,” “director,” “senior,” “lead”
- `IsExplicitDataJob`: flagged if title included “data,” “analyst,” “statistics,” or “information”
- `AgencySize`: categorized into small, medium, or large using thresholds developed collaboratively with AI
- `Industry`: assigned using department and title patterns

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

### Why NLP Is Effective for This Project

Natural Language Processing (NLP) enables pattern recognition at scale by understanding **how language is used**, not just what exact words are present. Public sector job titles and descriptions are highly inconsistent — NLP allows us to:

- Classify ambiguous roles (e.g., “Program Analyst”) that describe analytics or vendor management
- Generalize from confirmed buyers to latent ones using learned patterns
- Scale scoring across future datasets or job boards with minimal manual tagging

By training on real data buyers, the model reflects actual hiring language and captures demand that keyword-only methods miss.


A logistic regression NLP model was trained on the explicitly labeled data buyers to go beyond keyword logic. Using TF-IDF features (including unigrams, bigrams, and trigrams) and structured metadata (agency size, role seniority, data buyer scores), we classified additional roles based on linguistic and contextual similarity to known data buyer jobs. Class balancing via SMOTE significantly improved recall, ensuring high sensitivity for identifying emerging data buyer patterns in generalist and hybrid roles.

#### Features:

#### Text Features
- TF-IDF vectorization (unigrams to trigrams)
- Input sources: 
  - `JobTitle`
  - `JobDescription`
  - `KeyDuties`

#### Structured Features
- `AgencySize`
- `Industry` (automatically classified from job language,(same as sector))
- `IsSeniorRole` (inferred from title keywords like "Director", "Manager", etc.)

#### Class Balancing
- SMOTE (Synthetic Minority Oversampling Technique) was used to balance the training set for fair learning between buyers and non-buyers.

#### Outputs:
- A probability score (`PredictedDataBuyer`) for each job
- Token weight coefficients, identifying influential terms

This model allowed us to surface **latent data buyers** — jobs that don't use explicit terms but share linguistic patterns with confirmed buyers.

---


#### Validation:
- Accuracy and precision metrics reviewed
- Compared output against manually verified samples
- Reviewed top 50 false positives and false negatives for insights

> **Key modeling insight**: The NLP model revealed many hidden data buyer roles that used different language than the keywords — especially generalist roles like “Program Analyst” and “Grants Specialist.”


### Model Performance


| Metric         | Class 0 (Not a Buyer) | Class 1 (Likely Buyer) | Macro Avg | Weighted Avg |
|----------------|------------------------|--------------------------|-----------|---------------|
| Precision      | 97.99%                 | 61.42%                   | 79.71%    | 92.27%        |
| Recall         | 89.48%                 | 90.14%                   | 89.81%    | 89.59%        |
| F1 Score       | 93.55%                 | 73.06%                   | 83.30%    | 90.34%        |
| Accuracy       |                        |                          |           | **89.59%**    |


<img src="https://github.com/RoryQo/MQE-BSD-Capstone-Project/blob/main/Rory%20Files/Figures/CF.png?raw=true" alt="Confusion Matrix" width="400"/>



### Interpretation

- The model achieves strong **overall accuracy** (89.6%) across thousands of public sector job listings.
- It is **highly sensitive to actual data buyer roles**, with a 90.1% recall — ensuring very few buyers are missed.
- **Precision is moderate** (61.4%), which makes this model well-suited for **broad lead discovery** and outreach where recall matters more than strict precision.
- The inclusion of `Industry`, `AgencySize`, and `IsSeniorRole` as structured features adds **contextual richness**, helping the model distinguish buyers embedded in generalist, administrative, or hybrid roles.
- By focusing on **real job text** (title, description, and duties) and excluding engineered search terms (search keywords), the model maintains **generalizability and interpretability** across agencies and departments.

---
#### Oversight and False Positives

While the model performed well overall, a notable subset of **false positives** emerged — roles that should not have been labeled as likely data buyers, yet were mistakenly flagged either through keyword matches or overgeneralized text patterns. These included:

- **Pharmacy Technician**
- **Dental Assistant**
- **Nursing Assistant**
- **Motor Vehicle Operator**
- **Custodian**
- **Food Service Worker**

These roles often contain language about “inventory,” “records,” or “systems,” but these refer to clinical or operational workflows rather than third-party data use.

To improve precision:
- A refined set of **exclusionary keywords** was created to suppress these roles in future applications.

> **Recommendation**: Continued field expert oversight is important to guide future tuning. Expert oversight can help identify new use case language, evolving terminology, and overlooked sectors.

Together, the NLP model and manual refinement process enable a reliable, adaptable framework for classifying and understanding public sector demand for external data.

---

### Model Deployment and Flexibility

The entire classification and scoring pipeline has been designed for **scalability**. A **full API-call–ready codebase** is available, enabling:

- Real-time job scoring and ingestion
- Integration into CRM systems or lead qualification dashboards
- Periodic retraining or keyword updates without code rewrites

**Keyword libraries** used for inclusion and exclusion are fully modular and can be **easily adapted to evolving use cases** — such as open location data, AI readiness, or health analytics.

---


### Future Applications

- **Job Board Monitoring**: Real-time scanning of USAJobs or state boards
- **Agency Profiling**: Map departments to use case scores and buyer density
- **Lead Scoring**: Prioritize outreach based on predicted data buyer likelihood
- **Keyword Expansion**: Refine campaign targeting using evolving term patterns

---

### Additional Considerations

- **Posting Seasonality**: Hiring volume shifts over fiscal quarters; model should be rerun periodically.
- **Role Ambiguity**: A “Program Analyst” in CDC is not the same as in GSA — departmental context matters.
- **Proxy vs. Confirmation**: NLP scores and keyword matches reflect *intent*, not confirmed contract data. Still, strong patterns emerge across departments.
- **Evolving Terminology**: Language around “LLMs,” “synthetic data,” or “real-time insight” is starting to appear — the keyword and model pipeline is built to evolve with these trends.

---

### Limitations

- No access to actual procurement data — analysis infers *intent*, not confirmed purchases
- Evolving language — terms like “audience segmentation” or “performance analytics” may shift over time
- Context confusion — some words (e.g., “records,” “system,” “data entry”) appear across very different job types
- Occasional false positives from operational roles (e.g., food service, dental support) remain possible without continued tuning

---

## Market Analysis: Public Sector Demand for Third-Party Data

### Overview

While only a fraction of public sector jobs explicitly mention third-party data vendors, many reference tasks, platforms, or responsibilities that imply it. This analysis surfaces which **roles**, **sectors**, and **agencies** are most likely to be engaging in data buying, either directly or through embedded programmatic work.

---

### Interpretation Note on Statistical Testing
Statistical comparisons — including those related to seniority, agency size, sector, and use case alignment — were tested using the subset of explicitly labeled data buyers to confirm significance. These comparisons ensured that observed differences (e.g., between senior and non-senior roles, or across sectors) were not artifacts of modeling assumptions.

Likewise, the tables and visualizations in the Market Analysis section reflect only these explicitly identified and fuzzy-matched data buyers. No roles classified solely by the NLP model are included in these summaries. This ensures that observed patterns are grounded in direct, interpretable evidence from job language rather than probabilistic inference.

### Use Case Alignment by Agency

| Use Case               | High-Aligned Agencies                             | Buyer Role Share (%)       |
|------------------------|---------------------------------------------------|-----------------------------|
| Fraud Detection        | SSA, GSA, VA                                       | 60–75%                      |
| Sentiment Analysis     | FEMA, HHS, CDC                                     | 40–60%                      |
| Patient Matching       | VA Medical, IHS, DoD Medical                       | 70–80%                      |
| Ad Targeting           | CDC Foundation, HHS Comms, FEMA                    | 35–45%                      |

> **Note:** The "Buyer Role Share (%)" column reflects the **range of proportions** of data buyer jobs within each listed agency that are associated with the given use case. For example, 70–80% for patient matching means that among data buyer roles at VA Medical, IHS, and DoD Medical, between 70% and 80% are tagged as related to patient record matching. This indicates the **functional focus** of those agencies' data buyer roles.

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

| Token             | Model Weight |
|------------------|--------------|
| acquisition       | 1.40         |
| purchase          | 1.32         |
| hiring managers   | 1.29         |
| external partners | 1.21         |
| budget planning   | 1.15         |
| vendors           | 1.09         |
| contracts         | 1.03         |
| procurement       | 0.98         |
| interoperability  | 0.92         |
| commercial data   | 0.90         |
| licensing         | 0.88         |
| data integration  | 0.85         |
| vendor platform   | 0.82         |
| market intelligence | 0.79       |
| external data     | 0.77         |


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

- **Seniority doesnt matter**: Jobs flagged as `IsSeniorRole = 1` were not significantly more likely to be data buyers (*Welch’s t-test, p < 0.33*), however, particularly in smaller agencies with limited internal analytics capacity, seniority becomes slightly significant.
- **Agency size effect**: Small and medium-sized agencies were more likely to embed data responsibilities in generalist or administrative roles (e.g., program coordinators, contract officers) instead of standalone analyst positions.
- **Kurtosis in buyer score**: The `DataBuyerScore` exhibited **high kurtosis (>4)** — suggesting that most public jobs are definitively non-buyers, while a small, distinct cluster exhibit strong signals of buyer behavior.


---

### Data Buyer Volume and Concentration by Agency and Sector

In addition to use case and title-level signals, we evaluated which **agencies and sectors** are most strongly associated with data buyers.

#### Agencies with the Most Data Buyer Jobs (Total Count)

| Agency | Data Buyer Jobs |
|--------|------------------|
| Department of Veterans Affairs | 471 |
| Department of the Navy | 92 |
| Department of the Army | 81 |
| Department of the Air Force | 63 |
| Department of Homeland Security | 56 |
| Department of Justice | 36 |
| Legislative Branch | 26 |
| Department of Transportation | 26 |
| Department of Health and Human Services | 23 |
| Department of Defense | 21 |

The VA dominates in both hiring volume and data buyer count, reflecting its operational scale and emphasis on health record systems, fraud controls, and vendor-based analytics platforms.

#### Agencies with Highest % of Jobs as Data Buyers

Agencies with a **smaller total number of postings** but a **high proportion** of data buyer roles include:

- Legislative Branch (e.g., research and communications offices)
- FEMA and CDC (especially in communications, ad targeting, and outreach)
- Department of Justice (in legal compliance and fraud)

These agencies should not be overlooked simply because of lower job counts — they often rely heavily on **external data vendors to fulfill narrow mission-critical functions**.


<img src="https://github.com/RoryQo/MQE-BSD-Capstone-Project/blob/main/Rory%20Files/Figures/agencypct.png?raw=true" alt="Agency Buyer % Distribution" width=600/>

---

#### Sector-Level Trends by Buyer Concentration

While **Health and Finance** sectors had the **highest volume** of data buyer roles overall, **Communications and Policy** showed the **highest proportion** of buyer roles relative to total hiring.

This suggests:
- Communications teams are becoming major consumers of third-party data (e.g., for media tracking, sentiment analysis, and ad targeting).
- Policy groups engage data for monitoring impact, audience feedback, and legislative decision support.
- Security and IT roles, while large in number, more often use internally generated data or manage infrastructure, not procure third-party sources.



<p align="center">
  <img src="https://github.com/RoryQo/MQE-BSD-Capstone-Project/blob/main/Rory%20Files/Figures/pctsec.png?raw=true" alt="Percent by Sector" width="500"/>
  <img src="https://github.com/RoryQo/MQE-BSD-Capstone-Project/blob/main/Rory%20Files/Figures/sectornum.png?raw=true" alt="Count by Sector" width="500"/>
</p>

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




