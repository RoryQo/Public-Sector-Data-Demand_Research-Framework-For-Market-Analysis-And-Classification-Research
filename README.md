<h1 align="center"> Targeting Data Buyers in the Public Sector:<br> A Predictive Modeling and Market Strategy Guide</h1>

<table align="center">
  <tr>
    <td colspan="2" align="center" style="background-color: white; color: black;"><strong>Table of Contents</strong></td>
  </tr>

  <tr>
    <td style="background-color: white; color: black; padding: 10px;">
      1. <a href="#overview" style="color: black;">Overview</a><br>
    </td>
    <td style="background-color: gray; color: black; padding: 10px;">
      2. <a href="#methodology" style="color: black;">Methodology</a><br>
      &nbsp;&nbsp;&nbsp;- <a href="#data-and-scope" style="color: black;">Data and Scope</a><br>
      &nbsp;&nbsp;&nbsp;- <a href="#text-normalization-and-feature-engineering" style="color: black;">Text Normalization & Feature Engineering</a><br>
    </td>
  </tr>

  <tr>
    <td style="background-color: gray; color: black; padding: 10px;">
      3. <a href="#natural-language-processing-model" style="color: black;">NLP Model</a><br>
      &nbsp;&nbsp;&nbsp;- <a href="#feature-inputs" style="color: black;">Model Specifications</a><br>
      &nbsp;&nbsp;&nbsp;- <a href="#interpretation-of-databuyerscore-distribution" style="color: black;">Score Distribution</a><br>
      &nbsp;&nbsp;&nbsp;- <a href="#model-performance-and-fit" style="color: black;">Model Performance and Fit</a><br>
      &nbsp;&nbsp;&nbsp;- <a href="#model-deployment-and-flexibility" style="color: black;">Model Deployment and Flexibility</a><br>
      &nbsp;&nbsp;&nbsp;- <a href="#limitations" style="color: black;">Limitations</a>
    </td>
    <td style="background-color: white; color: black; padding: 10px;">
      4. <a href="#market-analysis-public-sector-demand-for-third-party-data" style="color: black;">Market Analysis</a><br>
      &nbsp;&nbsp;&nbsp;- <a href="#use-case-alignment-by-agency" style="color: black;">Use Case Alignment</a><br>
      &nbsp;&nbsp;&nbsp;- <a href="#sector-trends" style="color: black;">Sector Trends</a><br>
      &nbsp;&nbsp;&nbsp;- <a href="#key-roles-and-latent-buyers" style="color: black;">Key Roles & Latent Buyers</a><br>
      &nbsp;&nbsp;&nbsp;- <a href="#token-insights-and-role-patterns" style="color: black;">Token Insights</a><br>
      &nbsp;&nbsp;&nbsp;- <a href="#statistical-patterns-in-data-buyer-classification" style="color: black;">Statistical Patterns</a>
    </td>
  </tr>

  <tr>
    <td style="background-color: gray; color: black; padding: 10px;">
      5. <a href="#strategic-recommendations" style="color: black;">Strategic Recommendations</a>
    </td>
    <td style="background-color: white; color: black; padding: 10px;">
      6. <a href="#conclusion" style="color: black;">Conclusion</a>
    </td>
  </tr>

  <!-- Faked-centered row -->
  <tr>
    <td colspan="2" style="background-color: white; color: black; padding: 10px;">
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      <a href="#appendix-automation-and-reproducibility" style="color: black;">7. Fully Automated Data Buyer Lead Generation</a>
    </td>
  </tr>
</table>









# Overview

This project analyzes thousands of U.S. federal government job postings scraped from the USAJOBS public API to identify roles that are likely engaged in purchasing or using third-party data. Many such roles are not explicitly labeled as “data buyer” positions, but contain textual and structural signals indicating involvement with vendor platforms, data subscriptions, analytics integration, or fraud detection tools.

To uncover these latent buyer roles, we trained a logistic regression classification model using TF-IDF features derived from job titles, descriptions, and key duties, alongside structured metadata like agency size, industry classification, and role seniority. Training data was built from a curated set of known buyers identified through keyword and fuzzy matching logic. The model was then applied across the full dataset to assign each job a probability score—used to assess buying likelihood and analyze broader trends.

In parallel, the project includes a market analysis component that investigates broader patterns in public sector data demand. This includes mapping strategic use cases — such as fraud detection, sentiment analysis, patient matching, and ad targeting — to specific agencies and roles. It also explores how data buyer behavior varies by agency size, role seniority, Industry, and the presence of generalist versus specialist job functions. This project goes beyond role-level classification to map how data use cases show up across agencies and industries. Through visualization and scoring, it highlights which parts of government are functionally reliant on third-party data, even when job titles and duties don’t explicitly say so.

This work provides value to vendors seeking to identify government demand for external data, researchers studying digital public infrastructure, and policymakers interested in how data-driven functions are operationalized in the federal workforce.

## Applications

- **Lead Prioritization for Data Vendors**: Identify which agencies, offices, and job roles are most likely to rely on third-party data providers.
- **Labor Market Insights**: Reveal how data-related responsibilities are evolving across federal hiring.
- **Procurement and Policy Research**: Analyze how public sector roles incorporate vendor engagement, analytics tools, and compliance monitoring through language in job postings.
- **Market Alignment for Data Vendors**: Help commercial data providers identify public sector job functions that align with their product offerings based on job language and organizational context.
  



# Methodology

This project analyzes public sector job postings to identify roles that are either explicitly or implicitly involved in third-party data acquisition — referred to throughout this report as “data buyers.” Using a blend of traditional keyword detection, fuzzy matching, rule-based tagging, and predictive modeling via natural language processing (NLP), this approach surfaces both overt and hidden demand for external data tools and services.

The goal is to provide commercial data vendors with a replicable, high-precision strategy for identifying public sector buyers aligned to specific strategic use cases: fraud detection, sentiment analysis, patient record matching, and ad targeting.

Beyond individual lead identification, the project also surfaces **market-level trends** that shape how data purchasing decisions are made. These include patterns in **agency size**, **role seniority**, **Industry**, and whether buyers tend to hold **generalist or specialist titles**. Understanding these structural dynamics enables vendors to tailor their outreach, prioritize agencies more likely to buy, and better anticipate where buyer authority is embedded within government organizations.

---

### Data and Scope

The dataset includes 5,829 public sector job postings from U.S. government agencies, retrieved via an API call to the publicly available USAJobs.gov database. Roughly 1/5 of these postings were flagged as likely third-party data buyers. Each record contains:

- Unstructured text: `JobTitle`, `JobDescription`, `SearchKeywords`, `KeyDuties`
- Categorical features: `Agency`, `Department`, `AgencySize`, `Industry`
- Derived features: `FuzzyMatchedPhrase`, `IsLikelyDataBuyer`, `sector`, `explicit data job`, `generalist`, `search keyword`

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

- `IsSeniorRole`: flagged if title included “chief,” “director,” “senior,” “lead”, "Sr.", "senior"
- `IsExplicitDataJob`: flagged if title included “data,” “analyst,” “statistics,” or “information”
- `AgencySize`: categorized into small, medium, or large using thresholds developed collaboratively with AI
- `Industry`: assigned using department and title patterns
- `IsGeneralistRole`: Flagged if the job title matched common generalist roles such as “Program Analyst,” “Contract Specialist,” or “Budget Officer” — roles that may be involved in procurement, coordination, or administrative tasks related to data, especially in smaller agencies.
- `SearchKeyword`: Created through a keyword-based tagging system designed to surface job postings aligned with public sector data buying positions.

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



### Industry Classification

Created a derived `Industry` column using a **keyword-based classification function**. This method scans multiple fields in each job posting — specifically `JobTitle`, `Department`, and `SearchKeywords` — and assigns a sector based on the presence of domain-specific terms.


Each job was tagged to one or more of the following industries:

| Industry        | Example Keywords                                                       |
|-----------------|------------------------------------------------------------------------|
| Finance         | “finance,” “financial,” “account,” “budget”                            |
| Marketing       | “marketing,” “communications,” “advertising”                           |
| Medical         | “medical,” “pharmacy,” “nurse,” “health,” “clinical”                   |
| Security/Tech   | “cyber,” “security,” “information technology,” “software,” “tech”      |
| Policy          | “policy,” “regulation,” “legislative,” “analyst,” “compliance”         |
| Other           | Default category for postings that do not match the above keywords     |

### Use Case Classification

**Use case indicators** based on keyword detection in the `CombinedText` field (which merges job title, description, and duties). 

These flags reveal the functional needs of each role, helping data vendors assess product fit and agency compatibility.


| Use Case               | Example Keywords                                                   |
|------------------------|---------------------------------------------------------------------|
| Fraud Detection        | “eligibility verification,” “audit,” “fraud risk,” “screening”     |
| Sentiment Analysis     | “public opinion,” “media analytics,” “feedback loop”              |
| Patient Record Matching| “EHR,” “interoperability,” “data linkage,” “record integration”    |
| Ad Targeting           | “audience segmentation,” “campaign optimization,” “targeting data” |

---

### Search Keyword

To explore how job postings aligned with specific public sector data use cases, we implemented a **keyword-based tagging system**. This allowed us to surface jobs relevant to particular themes — such as **data**, **security**, **technology**, or **research** — and analyze how those align with sector and buyer signals.

- **One record per job**: Even if a job matches multiple keywords, it is only counted once in the dataset.
- **All matched keywords stored**: Jobs that trigger multiple matches retain all triggering keywords for analysis (e.g., a posting may match both `data` and `security`).
- This allows us to identify **overlapping themes** and understand which jobs represent **hybrid use cases** or cross-cutting responsibilities.

---

# Natural Language Processing Model

### Why NLP Is Effective for This Project

Natural Language Processing (NLP) enables pattern recognition at scale by understanding **how language is used**, not just what exact words are present. Public sector job titles and descriptions are highly inconsistent — NLP allows us to:

- Generalize from confirmed buyers to latent ones using learned patterns
- Prioritize outreach by assigning a **likelihood score** to each role, indicating its probability of representing a data buyer.
- Scale scoring across future datasets or job boards with minimal manual tagging

By training on real data buyers, the model reflects actual hiring language and captures demand that keyword-only methods miss.


A logistic regression NLP model was trained on the explicitly labeled data buyers to go beyond keyword logic. Using TF-IDF features (including unigrams, bigrams, and trigrams) and structured metadata (agency size, role seniority, and industry), we classified additional roles based on linguistic and contextual similarity to known data buyer jobs. Class balancing via SMOTE significantly improved recall, ensuring high sensitivity for identifying emerging data buyer patterns in generalist and hybrid roles.
    
#### Formal Model Specification

The model used is a **logistic regression**, estimated to classify federal job postings as either likely third-party data buyers or not.

Let:


- $\( Y_i \in \{0, 1\} \)$ be a binary outcome variable indicating whether job $\( i \)$ is classified as a likely data buyer.
- $\( \mathbf{X}_i \)$ be the feature vector for job $\( i \)$, including both structured metadata and TF-IDF vectorized text features.
- $\( p_i = \mathbb{P}(Y_i = 1 \mid \mathbf{X}_i) \)$ be the predicted probability of being a data buyer.


The logistic regression model estimates:

```math
\log\left( \frac{p_i}{1 - p_i} \right) = \beta_0 + \sum_{j=1}^{k} \beta_j X_{ij}
```

Where:
- $\( \beta_0 \)$ is the model intercept
- $\( \beta_j \)$ are the estimated coefficients for features $\( X_{ij} \)$
- The model uses **L2 regularization** and is trained using **maximum likelihood estimation** on a SMOTE-balanced training set

### Feature Inputs

1. **Text Features** (TF-IDF):
   - N-grams (1 to 3 words) from:
     - `JobTitle`
     - `JobDescription`
     - `KeyDuties`
   - Top 5,000 tokens selected by frequency and importance

To identify common language patterns associated with data-buying behavior:
 1. A bag-of-words and TF-IDF (term frequency-inverse document frequency) representation was generated from combined job text.
 2. Keyword importance was assessed by comparing term frequencies between data buyer and non-data buyer job postings.
 3. N-grams (two- and three-word sequences) were extracted to capture meaningful phrases such as “market intelligence” or “patient matching.”


2. **Structured Features**:
   - `AgencySize` (one-hot encoded)
   - `Industry` (derived from rule-based classifier, one-hot encoded)
   - `IsSeniorRole` (derived by keyword matching in job titles, binary)

#### Class Balancing

To address class imbalance between confirmed data buyers and non-buyers, we applied SMOTE (Synthetic Minority Oversampling Technique) during training. Rather than duplicating minority examples, SMOTE generates synthetic samples by interpolating between existing observations. This creates a more diverse and representative set of buyer-like roles.

Class balancing is essential for fair learning. Without it, the model would be biased toward the majority class (non-buyers), suppressing its ability to detect latent buyers. SMOTE ensures that the model learns equally from both classes, improving generalization and recall for identifying underrepresented buyer roles.


#### Outputs:
- `PredictedLabel`: A binary classification (0 or 1) indicating likely buyer status.
- `DataBuyerScore`: The predicted probability (∈ [0,1]) assigned by the model.
- `Token Coefficients`: Feature weights extracted from the trained model to identify high-impact terms.


This model allowed us to surface **latent data buyers** — jobs that don't use explicit terms but share linguistic patterns with confirmed buyers.

---

### Interpretation of DataBuyerScore Distribution

The distribution of `DataBuyerScore` reveals important patterns about the model's confidence and behavior in classifying likely data buyers.

<img src="https://github.com/RoryQo/MQE-BSD-Capstone-Project/blob/main/Rory%20Files/Figures/download%20(8).png?raw=true" alt="DataBuyerScore Distribution" width=600/>


#### Distribution Overview

- The histogram shows a **long right tail**: while most jobs have low to moderate scores, a smaller cluster of jobs receive very high scores (close to 1.0).
- A **significant number of jobs are scored near 0**, indicating high confidence that these are not buyer roles.
- The middle of the distribution (scores between 0.1 and 0.6) is broad, reflecting many cases where the model has moderate or uncertain confidence.

#### Kurtosis: -0.24

- The `DataBuyerScore` has a **slightly negative kurtosis** of **-0.24**, indicating a **flatter-than-normal distribution** (platykurtic).
- This confirms what the histogram suggests: the model does not produce a sharply peaked distribution with extreme outliers.
- Scores are more **evenly spread**, with fewer strong peaks or long tails than a normal distribution.

### Implications

| Insight | Implication |
|--------|-------------|
| Model is not overly confident | Most jobs fall between 0.1 and 0.6 — suggesting nuanced scoring rather than binary separation. |
| Few extremely high scores | High-scoring jobs (e.g., > 0.8) likely represent very clear data buyer roles. |
| Wide mid-range of scores | Many jobs have ambiguous or mixed signals, such as hybrid roles or vague language. |
| Low kurtosis confirms flat shape | There is no strong clustering — scores are widely distributed across the range. |

### Possible Additional Implementations

- **Ranking**: Treat the score as a priority index for outreach or further human review.
- **Thresholding**: Use score bands (e.g., > 0.7 = likely buyer, 0.4–0.7 = possible, < 0.4 = unlikely) for segmentation.

---


#### Validation:
- Accuracy and precision metrics reviewed
- Compared output against manually verified samples
- Reviewed top 50 false positives and false negatives for insights

> **Key modeling insight**: The NLP model revealed many hidden data buyer roles that used different language than the keywords — especially generalist roles like “Program Analyst” and “Grants Specialist.”


## Model Performance and Fit

### Performance

| Metric         | Class 0 (Not a Buyer) | Class 1 (Likely Buyer) | Macro Avg | Weighted Avg |
|----------------|------------------------|--------------------------|-----------|---------------|
| Precision      | 97.99%                 | 61.42%                   | 79.71%    | 92.27%        |
| Recall         | 89.48%                 | 90.14%                   | 89.81%    | 89.59%        |
| F1 Score       | 93.55%                 | 73.06%                   | 83.30%    | 90.34%        |
| Accuracy       |                        |                          |           | **89.59%**    |


<img src="https://github.com/RoryQo/MQE-BSD-Capstone-Project/blob/main/Rory%20Files/Figures/CF.png?raw=true" alt="Confusion Matrix" width="400"/>


#### Interpretation

- The model achieves strong **overall accuracy** (89.6%) across thousands of public sector job listings.
- It is **highly sensitive to actual data buyer roles**, with a 90.1% recall — ensuring very few buyers are missed.
- **Precision is moderate** (61.4%), which makes this model well-suited for **broad lead discovery** and outreach where recall matters more than strict precision.
- The inclusion of `Industry`, `AgencySize`, and `IsSeniorRole` as structured features adds **contextual richness**, helping the model distinguish buyers embedded in generalist, administrative, or hybrid roles.
- By focusing on **real job text** (title, description, and duties) and excluding engineered search terms (search keywords), the model maintains **generalizability and interpretability** across agencies and departments.

#### Log Loss

- **Log Loss Score (Model):** `0.316`

Log loss evaluates how well the predicted probabilities align with actual class labels, penalizing incorrect predictions more harshly when they are overly confident. Our model’s log loss score of **0.316** represents a substantial improvement over the naïve baseline score of **0.690**, which reflects a model that always predicts the average class probability. This improvement confirms that our model generates **well-calibrated, informative probabilities**.

Importantly, this score aligns with our earlier observation that **most predicted probabilities fall between 0.1 and 0.6**, indicating the model is not overly confident in its predictions. This moderate range of scores suggests the model is capturing real-world ambiguity—providing nuanced, probabilistic estimates instead of binary, all-or-nothing outputs. That calibration helps lower log loss and supports more effective prioritization and ranking of potential leads.


#### Threshold Optimization on Performance

To further improve performance, we applied **threshold optimization** by selecting the probability cutoff that maximized the **F1 score**. This process identified an **optimal threshold of 0.57**, which provided a better balance between precision and recall compared to the default threshold of 0.5.

Overall metrics such as **F1 score** and **accuracy** improved at this threshold. However, we also observed that the model missed some true data buyers. Since the primary goal of this model is the **initial identification of potential public sector data buyers**, we consider **recall more critical than precision**.

In other words, it’s preferable to **over-predict** (accept more false positives) than to **under-predict** and miss genuine prospects. As a result, while the threshold of 0.57 is useful for improving general performance, we continue to evaluate thresholds that prioritize **high recall**, even at the expense of precision.

That said, when we optimized directly for recall — for example, targeting a **90% recall rate** — the resulting threshold was **too low**, leading to the generation of **many unqualified or unuseful leads**. In this context, the original **default threshold of 0.5** strikes a practical balance: it **successfully identifies most true data buyers**, while avoiding the overgeneration of low-value predictions. This balance supports a more effective and efficient lead identification strategy in real-world use.

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

### Fit

#### McFadden’s Pseudo R²

**McFadden’s R²** is a commonly used metric to assess model fit in **logistic regression**. It serves as an analog to the traditional R² used in linear regression but is based on **log-likelihood** rather than variance. As a result, its values are typically **much lower** than what is considered “good” in linear models. In **economics**, a pseudo R² of **0.20 or higher** is often considered **good**, as it suggests the model explains a meaningful portion of the variation in the outcome, especially in the context of complex or noisy data like human behavior or public sector trends.

> **Note:** McFadden’s R² values should **not** be directly compared to linear R² values — even models with **excellent fit** may only reach values around 0.3 to 0.4.

Our model achieved a **McFadden's pseudo R² of 0.271**, indicating a **moderate to strong fit**. This confirms that the combination of text and metadata features contributes substantially to the model's ability to distinguish between data buyers and non-buyers.


The formula is:

```math
R^2_{\text{McFadden}} = 1 - \frac{\log L_{\text{model}}}{\log L_{\text{null}}}
```

Where:
- $\( \log L_{\text{model}} \)$ is the log-likelihood of the fitted model
- $\( \log L_{\text{null}} \)$ is the log-likelihood of a null model with no predictors

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

# Market Analysis: Public Sector Demand for Third-Party Data

### Overview

Public sector organizations are increasingly turning to third-party data to support operational efficiency, improve service delivery, and drive evidence-based policy. This trend is evident across job postings that explicitly or implicitly reference the use, procurement, or integration of external data sources. A close analysis of public sector job language reveals demand for data tools and services across a diverse range of functions, not just IT or analytics.

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

**1. Data Buying Roles Appear in Non-Data Job Titles**

A large proportion of roles classified as likely data buyers do not carry traditional data science or analysis titles. In smaller agencies especially, data acquisition responsibilities are often embedded in roles like procurement specialists, operations coordinators, and public health administrators.

Roles flagged by explicit labeling as data buyers — even without “data” in the title or a technical role:

- Contract Specialist
- Program Analyst
- Management Analyst
- Grants Officer
- Public Affairs Officer
- Health IT Coordinator
- Communications Strategist
- Budget Officer

These roles frequently involve vendor management, platform evaluation, fraud analysis, campaign tracking, and operational analytics — all indicating third-party data use- a useful commonality that the nlp might be able to put to contextual use.

---

### Token Insights and Role Patterns

Text patterns extracted from the NLP model reveal what kinds of language most commonly appear in likely data buyer postings — and what kinds of phrases typically indicate non-buyers.


#### Filtering Strategy: Precision Targeting  (Top Token Predictors)

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

#### Importance of N-grams for Context Understanding
These patterns underscore how critical **n-grams** are for understanding context in natural language. For example, the word **“operation”** on its own might suggest data operations or business functions related to analytics. However, when preceded by terms like **“motor vehicle”**, the context clearly shifts away from data buying.

Allowing the model to interpret **multi-word phrases** instead of just individual tokens enables it to:

- Distinguish between similar words used in **very different contexts**
- Reduce misclassification by learning from **preceding and surrounding words**
- Increase precision by correctly ignoring irrelevant but superficially similar terms

---

## Statistical Patterns in Data Buyer Classification

Several statistically observable patterns emerged in the structured and text-based scoring of government job postings using raw observations and the trained NLP model.

### 1. Seniority Doesn’t Always Matter

Across all postings, jobs flagged as `IsSeniorRole = 1` were **not significantly more likely** to be associated with data purchasing (Welch’s t-test, *p* = 0.33). This challenges the assumption that data buying is primarily driven by executive or senior leadership roles.

That said, **agency size introduces meaningful context**:

- In **small agencies**, senior roles were *slightly less likely* than non-senior roles to be data buyers.
- In **large agencies**, the opposite pattern emerged — senior roles were *somewhat more likely* to be buyers.
- Neither result was statistically significant, but the reversal in trend hints at structural differences.


**Interpretation**: Seniority alone isn’t a strong predictor of buyer behavior. But in **larger, more hierarchical agencies**, senior staff may be more involved in procurement decisions. In **smaller agencies**, the distinction between senior and non-senior roles may blur, with non-senior staff taking on broader responsibilities, including vendor coordination.

---

 ### 2. Smaller Agencies Are Statistically More Likely to Contain Buyer Roles

When comparing buyer-likelihood across agency sizes, smaller agencies show a **significantly higher proportion** of roles classified as likely data buyers.

- **Small agencies** had a buyer likelihood close to **19%**
- **Medium agencies** followed at **~17%**
- **Large agencies** trailed at **~15%**
- Welch’s t-test between small and large agencies yielded a **t-statistic of -2.912** and **p-value of 0.0036**, indicating a statistically significant difference.

  <img src="https://github.com/RoryQo/MQE-BSD-Capstone-Project/blob/main/Rory%20Files/Figures/size.png?raw=true" alt="Percent by Sector" width="500"/>

**Interpretation:**  
Agency size appears to be a meaningful predictor of buyer presence since smaller agencies are likelier to include roles that engage in data purchasing. Vendors may benefit from tailoring outreach strategies based on the structural and operational dynamics of minor versus large agencies. This may reflect different procurement structures, leaner staffing models, or greater operational flexibility compared to larger agencies. 


### 3. Agency Size Effect

Smaller and mid-sized agencies are more likely to embed data responsibilities in **generalist or administrative titles**, rather than posting explicitly labeled analyst roles. Examples include:

- Contracting Officers managing third-party data subscriptions
- Program Analysts coordinating vendor platforms
- Grants Officers using external data to evaluate program outcomes

Larger agencies (e.g., VA, DoD, HHS) more frequently post **specialized analyst roles**, making their buyer signals easier to detect via job title alone.

**Implication**: Vendors targeting smaller agencies should pay close attention to generalist or hybrid roles with cross-functional responsibilities.


---

### 3. Data Buyer Volume and Concentration by Agency and Sector

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

#### 4. Sector-Level Trends by Buyer Concentration

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

### Strategic Recommendations

| Use Case           | Target Agencies                  | Product Opportunity                       |
|--------------------|----------------------------------|--------------------------------------------|
| Fraud Detection    | SSA, VA, GSA                     | Eligibility checks, ID verification        |
| Sentiment Analysis | FEMA, HHS, CDC                   | Public engagement, media analytics         |
| Patient Matching   | VA, IHS, DoD Medical             | Interoperability tools, health ID systems  |
| Ad Targeting       | CDC Foundation, FEMA, HHS Comms  | Segmentation, outreach optimization        |

**Go-to-market guidance**:
1. **Broaden Targeting Criteria**: Sales and engagement strategies should include procurement and programmatic roles, not just technical ones.
2. **Segment Leads by Use Case**: Assign agencies to fraud, health, sentiment, or targeting use cases based on job text for tailored outreach.
3. **Prioritize Smaller Agencies for Embedded Data Needs**: These agencies are more likely to seek turnkey or vendor-supplied solutions, even within non-specialist roles.
4. **Use Percent-Based Use Case Scores**: Identify niche or emerging opportunities even in agencies with low total hiring volume.

**Continual Considerations**:
- Monitor changes in job language using keyword tracking + token weights.
- Use job scoring outputs for lead segmentation and follow-up prioritization.
- Prioritize hybrid roles and generalists in small or mid-sized agencies.


---

### Conclusion

Data buyers in government are not always called “data analysts.” They are Program Managers, Contract Officers, and Health IT Leads managing real-world platforms and workflows that depend on third-party data. 

This analysis combines keyword matching, machine learning, and contextual classification to help vendors find these roles — and act on them. With an adaptable pipeline and strong textual patterns, this framework is not only accurate today, but built to evolve with the market.


---

## Appendix: Automation and Reproducibility

This project is fully automated from **data acquisition to model deployment**, using two modular notebooks:

### 1. API Call and Full Automatic Data Generation Label Creation.ipynb
- Connects to the official **USAJobs API** and pulls federal job postings based on configurable search criteria
- Cleans and parses returned data, standardizes fields, and formats for modeling
- Automatically applies a hybrid **labeling strategy**:
  - Keyword detection (e.g., “data subscription,” “vendor-supplied”)
  - Fuzzy phrase matching
  - Role-based flags (`IsSeniorRole`, `IsExplicitDataJob`, etc.)
- Outputs a labeled dataset with buyer and non-buyer tags — ready for training or prediction

**All keyword lists are fully editable**, allowing you to adapt tagging logic to reflect new tools, vendors, and terminology as the public sector data market evolves

### 2. Automated Modeling and Lead Generating.ipynb
- Loads labeled job data and feeds it into a **prebuilt NLP modeling pipeline**
- Applies preprocessing:
  - TF-IDF text vectorization
  - One-hot encoding of structured fields
  - SMOTE-based class balancing
- Generates predictions using a trained **NLP logistic regression model**
- Outputs:
  - `PredictedLabel` (0/1)
  - `DataBuyerScore` (likelihood ∈ [0,1])
- Automatically **exports a ranked lead list** of high-priority public sector buyer roles

###  Reusability
All model and feature pipelines are stored in `.pkl` files (included in this repo). This allows:
- Plug-and-play reruns on new job data
- Scalable deployment on different job boards
- Consistent inference without retraining
- 

---

### Future Work: Automated Contact Scraper for Data Buyer Leads

This project can be extended into a **full data buyer discovery engine** by adding a contact scraping module that connects model-identified roles to real professionals on LinkedIn.

#### Proposed Extension Pipeline:

##### 1. **Ingest Pre-Filtered Buyer Leads**
   - Use the output DataFrame from `Automated Modeling and Lead Generating.ipynb`
   - Extract unique combinations of:
     - `JobTitle`
     - `AgencyName`

##### 2. **Automated LinkedIn Search**
   - For each prospect, create a search string from the `JobTitle` and `AgencyName`, e.g.:
     ```
     "Senior Data Analyst" AND "Department of Education" site:linkedin.com/in
     ```
   - Create code to automatically perform a search for each prospect using this string
   - Use **Selenium** or **Playwright** to:
     - Simulate searching
     - Scroll results
     - Scrape preview data (title, company, link)

##### 3. **Scrape Public Contact Information**
   - For each matching LinkedIn profile, extract:
     - Full name
     - Current job title
     - Agency/employer
     - Location
     - LinkedIn URL
     - (Optional) Public email or shared connections

##### 4. **Build Contact-Level Lead File**
   - Merge scraped contact data with:
     - `DataBuyerScore`
     - `JobTitle`
     - `AgencyName`
   - Output a ranked contact DataFrame:

     | Name         | Title             | Agency               | LinkedIn URL             | Score | Location     |
     |--------------|-------------------|------------------------|---------------------------|--------|--------------|
     | Alex Chen    | Risk Analyst Lead | Dept. of Transportation | linkedin.com/in/alexchen | 0.88   | Boston, MA   |

##### 5. **Export for Outreach**
   - Automatically save as `data_buyer_contacts.csv`
   - Optionally push to Google Sheets or a CRM tool

> **Note:** Always follow LinkedIn’s Terms of Service. For compliant and scalable enrichment, consider APIs such as Clearbit, Apollo, or PeopleDataLabs.








> **Note:** Make sure to download the accompanying pipeline files (`.pkl`), which are required for loading the model and running predictions in `Automated Modeling and Lead Generating.ipynb`.
>  Simply update the file paths and USAJobs API key in the acquisition notebook, and the entire process will run start-to-finish — from pulling raw job data to generating export-ready lead lists.


