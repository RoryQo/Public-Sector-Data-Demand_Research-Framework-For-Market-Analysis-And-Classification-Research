<h1 align="center">Data Buyer Toolkit — Function Documentation</h1>



<p align="center">
  <a href="https://pypi.org/project/data-buyer-toolkit/">
    <img src="https://img.shields.io/static/v1?label=pypi%20package&message=data_buyer_toolkit%20v0.1.0&color=blue" alt="PyPI package">
  </a>
</p>



<table align="center">
  <tr>
    <td colspan="2" align="center" style="background-color: white; color: black; padding: 10px;">
      <strong>Table of Contents</strong>
    </td>
  </tr>

  <tr>
    <td align="center" style="background-color: white; color: black; padding: 10px;">
      1. <a href="#overview" style="color: black;">Overview</a>
    </td>
    <td align="center" style="background-color: gray; color: black; padding: 10px;">
      2. <a href="#folder-structure" style="color: black;">Folder Structure</a>
    </td>
  </tr>

  <tr>
    <td align="center" style="background-color: gray; color: black; padding: 10px;">
      3. <a href="#function-inputs-and-outputs" style="color: black;">Function Inputs and Outputs</a>
    </td>
    <td align="center" style="background-color: white; color: black; padding: 10px;">
      4. <a href="#quick-visual-summary" style="color: black;">Quick Visual Summary</a>
    </td>
  </tr>

  <tr>
    <td align="center" style="background-color: white; color: black; padding: 10px;">
      5. <a href="#when-to-use-each-function" style="color: black;">When to Use Each Function</a>
    </td>
    <td align="center" style="background-color: gray; color: black; padding: 10px;">
      6. <a href="#installation-instructions" style="color: black;">Installation Instructions</a>
    </td>
  </tr>

  <tr>
    <td align="center" style="background-color: gray; color: black; padding: 10px;">
      7. <a href="#license" style="color: black;">License</a>
    </td>
    <td align="center" style="background-color: white; color: black; padding: 10px;">
      8. <a href="#contributions" style="color: black;">Contributions</a>
    </td>
  </tr>

  <tr>
    <td colspan="2" align="center" style="background-color: gray; color: black; padding: 10px;">
      9. <a href="#usage-examples" style="color: black;">Usage Examples Notebook</a>
    </td>
  </tr>
</table>




## Overview

The `data_buyer_toolkit` package is a modular, published Python library for analyzing, preprocessing, and scoring U.S. federal job postings for third-party data acquisition demand. 

It is a core component of the broader **Public Sector Data Demand Research Framework**, but can also be used independently as a lightweight toolkit for real-time job analysis and data buyer detection.

Specifically, this package allows users to:
- Load a trained natural language processing (NLP) model.
- Fetch live job data from the USAJobs API.
- Preprocess and feature-engineer job descriptions for model input.
- Score the likelihood that a government position involves external data purchasing.
- Target specific use cases, such as fraud detection, sentiment analysis, patient record matching, or advertising targeting.

By operationalizing job text analysis, this package helps commercial data vendors, researchers, and policy analysts **identify promising government leads** and **map market demand trends** for external data products.

### Folder Structure

```
Root Directory:
├── setup.py
├── pyproject.toml


Python Package: data_buyer_toolkit/
├── __init__.py
├── toolkit.py
├── nlp_pipeline_with_smote.joblib
├── README.md
├── examples/
│   └── usage_examples.ipynb
```

---

# Function Inputs and Outputs

## `Import the Package`

```python
from data_buyer_toolkit.toolkit import (
    fetch_and_score_job,
    batch_fetch_and_score_jobs,
    search_job_ids_by_title,
    fetch_and_score_top_by_use_case_auto,
    preprocess_job_api_response,
    load_pipeline,
)
```

## `load_pipeline()`

```python
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


# Installation Instructions

## PyPi Install
You can install the package directly from PyPI:

```bash
pip install data-buyer-toolkit
```


Or for local development (editable mode with GitHub clone):
- Follow these steps to fully install and set up the `data_buyer_toolkit` for local development or usage inside Jupyter notebooks.


## Local Dev Install

### 1. Clone the Repository

First, clone the full project to your local machine:

```bash
git clone https://github.com/RoryQo/Public-Sector-Data-Demand_Research-Framework-For-Market-Analysis-And-Classification.git
cd Public-Sector-Data-Demand_Research-Framework-For-Market-Analysis-And-Classification
```

### 2. Create and activate a Virtual Environment

It is strongly recommended to use a virtual environment for this project.


- Create a new conda environment specifying Python version 3.10
- Activate the conda environment

```bash
conda create -n data-buyer-env python=3.10 -y
conda activate data-buyer-env
```

### 3. Install the package in Editable Mode

Inside the project root directory:

- Upgrade `pip`
- Install the package in editable mode (`-e`) so local changes are immediately reflected without reinstalling


```bash
pip install --upgrade pip
pip install -e .
```

### 4. Install Jupyter Kernel

- Install `notebook` and `ipykernel`
- Create a dedicated Jupyter kernel associated with your environment
- Name the kernel something descriptive (e.g., "Data Buyer Toolkit")

```bash
pip install notebook ipykernel
python -m ipykernel install --user --name=data-buyer-env --display-name "Data Buyer Toolkit"
```

---

# License

This project is licensed under the **MIT License**.

You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the following conditions:

- The above copyright notice and this permission notice shall be included in all copies or substantial portions of the software.
- The software is provided "as is", without warranty of any kind, express or implied.

For the full license text, see the [LICENSE](../LICENSE) file included in this repository.

---

# Contributions

Contributions are welcome and encouraged!

If you would like to suggest improvements, add new features, or report bugs, please follow these guidelines:

1. **Fork the repository** to your own GitHub account.
2. **Create a new branch** for your feature or fix.
3. **Write clear, descriptive commit messages**.
4. **Test your changes** thoroughly before submitting.
5. **Submit a pull request** describing what you have changed and why.


---

# Usage Examples

A full Jupyter notebook with hands-on examples is provided to demonstrate the `data_buyer_toolkit` in action.

📂 Access it here: [examples/usage_examples.ipynb](https://github.com/RoryQo/Public-Sector-Data-Demand_Research-Framework-For-Market-Analysis-And-Classification/tree/main/data_buyer_toolkit/examples)



## What the Notebook Covers

- **Load the trained NLP model** and initialize the pipeline.
- **Fetch job postings** live from the USAJobs API.
- **Preprocess job data** into a model-ready format.
- **Score jobs** using the `DataBuyerScore`.
- **Batch search and score** multiple job titles.
- **Filter and rank** jobs by targeted use cases (e.g., fraud detection, sentiment analysis).



## How to Use

1. Open `examples/usage_examples.ipynb` after installing the package.
2. Insert your USAJobs **API Key** and **email address** where indicated.
3. Run the cells to explore example workflows and customize as needed.


The notebook provides a practical guide for integrating the toolkit into custom workflows for real-time scoring, lead generation, and market targeting.





