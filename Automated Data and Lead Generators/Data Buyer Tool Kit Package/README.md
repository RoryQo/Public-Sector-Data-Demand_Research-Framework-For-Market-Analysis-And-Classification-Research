# Data Buyer Toolkit

A Python package to identify and score potential third-party data buyers from U.S. public sector job postings using Natural Language Processing (NLP) and feature engineering.

## Overview

The Data Buyer Toolkit processes U.S. federal job postings to predict which positions are likely involved in purchasing third-party data.
It automates the process of scraping, feature engineering, and scoring, making it a valuable tool for public sector market intelligence, lead generation, and vendor targeting.

- The core capabilities include:
- Live USAJobs search by keyword
- Automatic job description preprocessing and feature construction
- Scoring jobs using a trained machine learning pipeline
- Batch scoring multiple roles at once

## Installation

Clone the repository and install locally:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -e .

```

## Functions

### `fetch_and_score_job`

**Purpose**
Fetch a job posting from USAJobs by ControlNumber and predict its likelihood of being a third-party data buyer.

**Inputs**

- `job_id` (str): USAJobs ControlNumber
- `api_key` (str): Your USAJobs API key
- `email` (str): Your registered USAJobs email
- `pipeline_path` (str): Path to your trained `.joblib pipeline` (default: "nlp_pipeline_with_smote.joblib")

**Output**

Dictionary containing:

- `data_buyer_score` (float, 0 to 1)
- `title` (str): Job title
- `agency` (str): Hiring agency

**Example**

```python
from data_buyer_toolkit import fetch_and_score_job

result = fetch_and_score_job("820370600", api_key="your-api-key", email="your-email")
print(result)
```

### `search_and_score_keyword_live`

**Purpose**

Search USAJobs live by keyword, fetch matching jobs, preprocess them, and predict their likelihood of buying third-party data.

**Inputs**

- `keyword` (str): Search term (e.g., "data analyst", "procurement specialist")
- `api_key` (str): USAJobs API key
- `email` (str): USAJobs email
- `pipeline_path` (str): Path to your trained pipeline
- `max_results` (int, default=10): How many jobs to fetch
- `delay` (float, default=0.5): Optional delay between API calls

**Output**

- Pandas DataFrame containing:
  - data_buyer_score
  - title
  - agency
  - location
  - job_id

**Example**

```python
from data_buyer_toolkit import search_and_score_keyword_live

jobs = search_and_score_keyword_live(
    keyword="contract specialist",
    api_key="your-api-key",
    email="your-email",
    max_results=5
)

print(jobs)
```

### `batch_fetch_and_score_jobs`

**Purpose**

Given a list of job title keywords, search, fetch, preprocess, and score all matches automatically.

**Inputs**

- `job_titles` (list of str): List of job titles/keywords
- `api_key` (str): USAJobs API key
- `email` (str): USAJobs email
- `pipeline_path` (str): Path to trained pipeline
- `delay` (float, optional): Delay between jobs (default: 1 second)

**Output**

- Pandas DataFrame containing scored jobs

**Example**

```python
from data_buyer_toolkit import batch_fetch_and_score_jobs

titles = ["grants specialist", "data analyst", "procurement officer"]
scored_jobs = batch_fetch_and_score_jobs(titles, api_key="your-api-key", email="your-email")
print(scored_jobs)
```

## Typical Use Case

- Search for jobs by keyword (e.g., "fraud analyst")

- Fetch full job descriptions live from USAJobs

- Preprocess the data automatically using engineered features (title, duties, agency, seniority, use case match)

- Predict the Data Buyer Score

- Rank leads based on scores for outreach targeting



