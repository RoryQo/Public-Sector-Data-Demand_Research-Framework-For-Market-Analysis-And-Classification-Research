# Data Buyer Toolkit — Function Documentation

## Overview

The `data_buyer_toolkit` package is a core component of the broader **Public Sector Data Demand Research Framework**.  
It provides a modular set of tools to **analyze, preprocess, and score federal government job postings** from USAJobs for **third-party data acquisition demand**.

Specifically, this package allows users to:
- Load a trained natural language processing (NLP) model.
- Fetch live job data from the USAJobs API.
- Preprocess and feature-engineer job descriptions for model input.
- Score the likelihood that a government position involves external data purchasing.
- Target specific use cases, such as fraud detection, sentiment analysis, patient record matching, or advertising targeting.

By operationalizing job text analysis, this package helps commercial data vendors, researchers, and policy analysts **identify promising government leads** and **map market demand trends** for external data products.

---

# Function Inputs and Outputs

## `load_pipeline()`

```python
from data_buyer_toolkit.toolkit import load_pipeline
pipeline = load_pipeline()
```

**Purpose**:  
Load the trained NLP pipeline stored inside the package (`nlp_pipeline_with_smote.joblib`).

**Inputs**:
- None

**Outputs**:
- A scikit-learn pipeline object containing:
  - A preprocessing step (`preprocessor`)
  - A classification step (`classifier`)

---

## `preprocess_job_api_response(job_json)`
**Purpose**:  
Preprocess a single USAJobs API job posting into a structured, model-ready pandas DataFrame.

**Inputs**:
- `job_json` (`dict`):  
  A dictionary representing a single job posting JSON, typically from the USAJobs API.  
  Must contain at least:
  - `PositionTitle`
  - `OrganizationName`
  - `UserArea -> Details -> JobSummary`
  - (Optional) `MajorDuties`

**Outputs**:
- `df_processed` (`pd.DataFrame`):  
  A **single-row** DataFrame with all engineered features needed for modeling and scoring.

---

## `fetch_and_score_job(job_id, api_key, email)`

```python
score_result = fetch_and_score_job(job_id="1234567", api_key="YOUR_USAJOBS_API_KEY", email="YOUR_EMAIL@example.com")
print(score_result)
```

**Purpose**:  
Fetch a job posting by its ID from USAJobs, preprocess it, and score its likelihood of being a third-party data buyer using the NLP model.

**Inputs**:
- `job_id` (`str` or `int`):  
  The USAJobs position ID.
- `api_key` (`str`):  
  Your registered [USAJobs API Key](https://developer.usajobs.gov/).
- `email` (`str`):  
  Email address used as a `User-Agent` for the API call (must match your registered account).

**Outputs**:
- `result` (`dict`):  
  A dictionary with:
  - `data_buyer_score` (`float`): The predicted probability (0 to 1) that this job is a data buyer.
  - `title` (`str`): The job's title.
  - `agency` (`str`): The hiring agency.

---

## `search_job_ids_by_title(position_title, api_key, email, max_results=10)`

```python
job_matches = search_job_ids_by_title(position_title="Data Scientist", api_key="YOUR_USAJOBS_API_KEY", email="YOUR_EMAIL@example.com")
```

**Purpose**:  
Search the USAJobs API for job postings by job title keyword.

**Inputs**:
- `position_title` (`str`):  
  Keyword(s) to search job titles.
- `api_key` (`str`):  
  Your USAJobs API key.
- `email` (`str`):  
  Your email address for the API `User-Agent`.
- `max_results` (`int`, default = 10):  
  Maximum number of jobs to return.

**Outputs**:
- `jobs` (`list` of `dict`):  
  A list of jobs where each job is a dictionary with:
  - `job_id` (`str`)
  - `title` (`str`)
  - `agency` (`str`)``

---

##`batch_fetch_and_score_jobs(job_titles, api_key, email)`

```python
titles = ["Data Analyst", "Contract Specialist", "Program Manager"]
batch_scores = batch_fetch_and_score_jobs(titles, api_key="YOUR_USAJOBS_API_KEY", email="YOUR_EMAIL@example.com")
print(batch_scores)
```

**Purpose**:  
Search and score multiple job titles in batch.

**Inputs**:
- `job_titles` (`list` of `str`):  
  A list of job titles or keywords to search and score.
- `api_key` (`str`):  
  USAJobs API key.
- `email` (`str`):  
  USAJobs API registered email address.

**Outputs**:
- `results_df` (`pd.DataFrame`):  
  A DataFrame where each row is:
  - Title
  - Agency
  - Data buyer score

---

## `fetch_and_score_top_by_use_case_auto(api_key, email, use_case="Fraud", top_n=100)`

```python
top_fraud_jobs = fetch_and_score_top_by_use_case_auto(api_key="YOUR_USAJOBS_API_KEY", email="YOUR_EMAIL@example.com", use_case="Fraud", top_n=50)
print(top_fraud_jobs)
```

**Purpose**:  
Automatically search a broad set of keywords, pull all matches, and rank top-scoring jobs for a selected use case (e.g., Fraud, Sentiment).

**Inputs**:
- `api_key` (`str`):  
  USAJobs API key.
- `email` (`str`):  
  USAJobs API email `User-Agent`.
- `use_case` (`str`, default = `"Fraud"`):  
  Which use case column to filter and sort on. Options include:
  - `Fraud`
  - `Sentiment`
  - `PatientMatching`
  - `AdTargeting`
- `top_n` (`int`, default = 100):  
  Number of top jobs to return.

**Outputs**:
- `top_jobs_df` (`pd.DataFrame`):  
  A DataFrame with top job titles, agencies, and their data buyer scores filtered by the selected use case.

---

# Quick Visual Summary

| Function | Input | Output |
|:---------|:------|:-------|
| `load_pipeline()` | None | Scikit-learn pipeline |
| `preprocess_job_api_response()` | `job_json` dict | Preprocessed DataFrame |
| `fetch_and_score_job()` | `job_id`, `api_key`, `email` | Dict: score, title, agency |
| `search_job_ids_by_title()` | `position_title`, `api_key`, `email`, `max_results` | List of job dicts |
| `batch_fetch_and_score_jobs()` | List of titles, `api_key`, `email` | Results DataFrame |
| `fetch_and_score_top_by_use_case_auto()` | `api_key`, `email`, `use_case`, `top_n` | Top jobs DataFrame |

---

# When to Use Each Function

| Situation | Recommended Function |
|:----------|:----------------------|
| You want to **load the trained machine learning model** | `load_pipeline()` |
| You have a **raw USAJobs API job posting** and want to **prepare it for scoring** | `preprocess_job_api_response(job_json)` |
| You know a **specific job ID** and want to **score that job** | `fetch_and_score_job(job_id, api_key, email)` |
| You know a **job title keyword** and want to **find matching job IDs** | `search_job_ids_by_title(position_title, api_key, email)` |
| You have a **list of job titles** and want to **batch search and score them** | `batch_fetch_and_score_jobs(job_titles, api_key, email)` |
| You want to **search broadly across many jobs** and **filter top scoring jobs by a specific use case** (like fraud detection) | `fetch_and_score_top_by_use_case_auto(api_key, email, use_case)` |

---
