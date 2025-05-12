#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import requests
import joblib
from rapidfuzz import fuzz, process
import importlib.resources

# ------------------------
# Helper to Load Pipeline
# ------------------------

def load_pipeline():
    """Load the nlp_pipeline_with_smote.joblib from inside the package."""
    with importlib.resources.path('data_buyer_toolkit', 'nlp_pipeline_with_smote.joblib') as model_path:
        pipeline = joblib.load(model_path)
    return pipeline

# ------------------------
# Preprocessing Function
# ------------------------

import pandas as pd
import re
from rapidfuzz import fuzz, process

def preprocess_job_api_response(job_json):
    """
    Preprocess a single job JSON response into a model-ready dataframe.

    Args:
        job_json (dict): A single job's JSON dictionary from USAJobs API.

    Returns:
        pd.DataFrame: A single-row dataframe ready for model prediction.
    """

    # Basic fields
    title = job_json.get('PositionTitle', '')
    agency = job_json.get('OrganizationName', '')

    desc = job_json.get('UserArea', {}).get('Details', {}).get('JobSummary', '')
    duties = job_json.get('UserArea', {}).get('Details', {}).get('MajorDuties', '')

    if isinstance(desc, list):
        desc = ' '.join(desc)
    if isinstance(duties, list):
        duties = ' '.join(duties)

    df = pd.DataFrame([{
        'JobTitle': title,
        'Agency': agency,
        'JobDescription': desc,
        'KeyDuties': duties
    }])

    # Combined text
    df['CombinedText'] = (df['JobDescription'].fillna('') + ' ' + df['KeyDuties'].fillna('')).str.lower()

    # Direct keyword match
    related_phrases = [
        "data acquisition", "data procurement", "procure data", "purchase data",
        "buy data", "acquiring data", "data sourcing", "data licensing", 
        "external data acquisition", "third-party data", "data vendor", 
        "data provider", "data contracts", "contracting data", "data subscriptions",
        "vendor management", "external data", "commercial data"
    ]
    pattern = '|'.join(map(re.escape, related_phrases))
    df['IsDataBuyer'] = df['CombinedText'].str.contains(pattern, case=False, na=False).astype(int)

    # Fuzzy match
    signal_phrases = [
        "data acquisition", "data procurement", "procure data", "purchase data",
        "buy data", "acquiring data", "data sourcing", "data licensing", 
        "external data", "third-party data", "data vendor", "data provider", 
        "data contracts", "contracting data", "data subscriptions", "vendor management",
        "commercial data", "data assets", "data commercialization",
        "procurement of data", "external data sources", "data aggregators",
        "data monetization", "sourcing external data", "partner data", "data purchasing agreements",
        "data ingestion", "subscription data", "data acquisition strategy", "data buying",
        "external datasets", "external partnerships", "data sharing agreements",
        "data acquisition channels", "third-party data sources", "sourcing data providers",
        "managing data vendors", "data reseller", "external data vendors", "contracted data"
    ]

    def fuzzy_match(text, phrases, threshold=80):
        for phrase in phrases:
            if fuzz.partial_ratio(phrase.lower(), text.lower()) >= threshold:
                return phrase
        return None

    df['FuzzyMatchedPhrase'] = df['CombinedText'].apply(lambda x: fuzzy_match(x, signal_phrases))
    df['IsFuzzyMatch'] = df['FuzzyMatchedPhrase'].notnull().astype(int)

    # Likely buyer if either is true
    df['IsLikelyDataBuyer'] = ((df['IsDataBuyer'] == 1) | (df['IsFuzzyMatch'] == 1)).astype(int)

    # Agency size
    large_agencies = [
        "Department of Defense", "Department of Veterans Affairs", "Department of the Treasury",
        "Department of Homeland Security", "Department of Health and Human Services",
        "Department of Justice", "Department of the Army"
    ]
    medium_agencies = [
        "Department of Transportation", "Department of Commerce", "Department of Agriculture",
        "Department of Energy", "Department of the Interior", "National Aeronautics and Space Administration"
    ]

    def classify_agency(agency_name):
        if agency_name in large_agencies:
            return 'Large'
        elif agency_name in medium_agencies:
            return 'Medium'
        else:
            return 'Small'

    df['AgencySize'] = df['Agency'].apply(classify_agency).fillna('Unknown')

    # Industry
    def classify_industry(row):
        text = f"{row['JobTitle']} {row['Agency']}".lower()
        if any(word in text for word in ['finance', 'financial', 'account', 'budget']):
            return 'Finance'
        if any(word in text for word in ['marketing', 'communications', 'advertising']):
            return 'Marketing'
        if any(word in text for word in ['medical', 'health', 'clinical', 'pharmacy', 'nurse']):
            return 'Medical'
        if any(word in text for word in ['cyber', 'security', 'software', 'data scientist', 'tech', 'information technology']):
            return 'Security/Tech'
        if any(word in text for word in ['policy', 'regulation', 'legislative', 'compliance', 'analyst']):
            return 'Policy'
        return 'Other'

    df['Industry'] = df.apply(classify_industry, axis=1)

    # Senior role
    df['IsSeniorRole'] = df['JobTitle'].str.lower().str.contains(r'\bsenior\b|\blead\b|\bchief\b|\bprincipal\b|\bdirector\b|\bhead\b', na=False)

    # Explicitly a "data" job
    data_keywords = ['data', 'analyst', 'scientist', 'analytics', 'statistician', 'intelligence', 'information', 'it']
    df['IsExplicitDataJob'] = df['JobTitle'].str.lower().str.contains('|'.join(data_keywords), na=False).astype(int)

    # Use case detection
    use_case_keywords = {
        'Fraud': ['fraud', 'eligibility', 'verification', 'audit', 'compliance'],
        'Sentiment': ['sentiment', 'public opinion', 'media monitoring', 'engagement', 'communication'],
        'PatientMatching': ['patient match', 'interoperability', 'record linkage', 'ehr', 'health record'],
        'AdTargeting': ['audience segmentation', 'targeting', 'ad performance', 'campaign data']
    }
    for use_case, keywords in use_case_keywords.items():
        pattern = '|'.join(map(re.escape, keywords))
        df[f'UseCase_{use_case}'] = df['CombinedText'].str.contains(pattern, case=False, na=False).astype(int)

    # Generalist role detection
    generalist_titles = [
        'Contract Specialist', 'Grants Officer', 'Grants Specialist', 'Budget Officer',
        'Administrative Officer', 'Operations Coordinator', 'Program Coordinator',
        'Project Coordinator', 'Procurement Specialist', 'Procurement Analyst',
        'Communications Specialist', 'Public Affairs Officer', 'Public Information Officer',
        'Community Outreach Coordinator', 'Health IT Coordinator', 'Program Specialist',
        'Program Manager', 'Business Operations Specialist'
    ]

    def is_generalist(title):
        if not title:
            return False
        match, score, _ = process.extractOne(title, generalist_titles, scorer=fuzz.partial_ratio)
        return score >= 65

    df['IsGeneralistRole'] = df['JobTitle'].apply(lambda x: is_generalist(x))

    # FINAL CLEANUP
    df = df.fillna('')  # In case any column has missing values
    return df

# ------------------------
# Core Functions
# ------------------------

def fetch_and_score_job(job_id, api_key, email):
    headers = {"User-Agent": email, "Authorization-Key": api_key}
    url = f"https://data.usajobs.gov/api/Search?Keyword={job_id}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch job ID {job_id}: {response.status_code}")

    job_data = response.json()['SearchResult']['SearchResultItems'][0]['MatchedObjectDescriptor']
    df_processed = preprocess_job_api_response(job_data)
    pipeline = load_pipeline()
    X = pipeline.named_steps['preprocessor'].transform(df_processed)
    score = pipeline.named_steps['classifier'].predict_proba(X)[0][1]

    return {
        "data_buyer_score": round(score, 4),
        "title": job_data['PositionTitle'],
        "agency": job_data['OrganizationName']
    }

def search_job_ids_by_title(position_title, api_key, email, max_results=10):
    headers = {"User-Agent": email, "Authorization-Key": api_key}
    url = "https://data.usajobs.gov/api/Search"
    params = {"Keyword": position_title, "ResultsPerPage": max_results}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(f"Failed to search: {response.status_code}")

    jobs = response.json()['SearchResult']['SearchResultItems']
    return [{
        "job_id": job['MatchedObjectDescriptor']['PositionID'],
        "title": job['MatchedObjectDescriptor']['PositionTitle'],
        "agency": job['MatchedObjectDescriptor']['OrganizationName']
    } for job in jobs]

def batch_fetch_and_score_jobs(job_titles, api_key, email):
    results = []
    for title in job_titles:
        try:
            search_results = search_job_ids_by_title(title, api_key, email, max_results=1)
            if search_results:
                job_id = search_results[0]['job_id']
                results.append(fetch_and_score_job(job_id, api_key, email))
        except Exception as e:
            print(f"Error processing {title}: {e}")
    return pd.DataFrame(results)

# ------------------------
# USAJobs Live Full Search Functions
# ------------------------

def fetch_and_score_top_by_use_case_auto(api_key, email, use_case="Fraud", top_n=100):
    headers = {"User-Agent": email, "Authorization-Key": api_key}
    keywords = ['data', 'contract', 'analyst', 'machine learning', 'marketing']
    url = "https://data.usajobs.gov/api/Search"
    all_jobs = {}

    for keyword in keywords:
        params = {'Keyword': keyword, 'ResultsPerPage': 500, 'Page': 1}
        while True:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                break
            jobs = response.json().get('SearchResult', {}).get('SearchResultItems', [])
            if not jobs:
                break
            for job in jobs:
                job_id = job.get('MatchedObjectId')
                if job_id and job_id not in all_jobs:
                    descriptor = job['MatchedObjectDescriptor']
                    details = descriptor.get('UserArea', {}).get('Details', {})
                    all_jobs[job_id] = {
                        'JobTitle': descriptor.get('PositionTitle'),
                        'JobDescription': details.get('JobSummary'),
                        'KeyDuties': details.get('MajorDuties', ''),
                        'Agency': descriptor.get('OrganizationName')
                    }
            params['Page'] += 1

    df = pd.DataFrame(all_jobs.values())
    if df.empty:
        raise ValueError("No jobs found.")

    processed = [preprocess_job_api_response({
        'PositionTitle': row['JobTitle'],
        'OrganizationName': row['Agency'],
        'UserArea': {'Details': {'JobSummary': row['JobDescription'], 'MajorDuties': row['KeyDuties']}}
    }) for _, row in df.iterrows()]

    df_processed = pd.concat(processed, ignore_index=True)
    pipeline = load_pipeline()
    X = pipeline.named_steps['preprocessor'].transform(df_processed)
    df_processed['data_buyer_score'] = pipeline.named_steps['classifier'].predict_proba(X)[:, 1]

    use_case_column = f"UseCase_{use_case}"
    if use_case_column not in df_processed.columns:
        raise ValueError(f"Use case '{use_case}' not available.")

    return df_processed[df_processed[use_case_column] == 1].sort_values("data_buyer_score", ascending=False).head(top_n)[['JobTitle', 'Agency', 'data_buyer_score', use_case_column]]

def fetch_and_score_top_by_custom_inputs(api_key, email, search_keywords, use_case, top_n=100):
    if not search_keywords or not isinstance(search_keywords, list):
        raise ValueError("You must provide a list of search keywords.")
    if not isinstance(use_case, str):
        raise ValueError("You must provide a use case string.")

    headers = {"User-Agent": email, "Authorization-Key": api_key}
    url = "https://data.usajobs.gov/api/Search"
    all_jobs = {}

    for keyword in search_keywords:
        params = {'Keyword': keyword, 'ResultsPerPage': 500, 'Page': 1}
        while True:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                break
            jobs = response.json().get('SearchResult', {}).get('SearchResultItems', [])
            if not jobs:
                break
            for job in jobs:
                job_id = job.get('MatchedObjectId')
                if job_id and job_id not in all_jobs:
                    descriptor = job['MatchedObjectDescriptor']
                    details = descriptor.get('UserArea', {}).get('Details', {})
                    all_jobs[job_id] = {
                        'JobTitle': descriptor.get('PositionTitle'),
                        'JobDescription': details.get('JobSummary'),
                        'KeyDuties': details.get('MajorDuties', ''),
                        'Agency': descriptor.get('OrganizationName')
                    }
            params['Page'] += 1

    df = pd.DataFrame(all_jobs.values())
    if df.empty:
        raise ValueError("No jobs found.")

    processed = [preprocess_job_api_response({
        'PositionTitle': row['JobTitle'],
        'OrganizationName': row['Agency'],
        'UserArea': {'Details': {'JobSummary': row['JobDescription'], 'MajorDuties': row['KeyDuties']}}
    }) for _, row in df.iterrows()]

    df_processed = pd.concat(processed, ignore_index=True)
    pipeline = load_pipeline()
    X = pipeline.named_steps['preprocessor'].transform(df_processed)
    df_processed['data_buyer_score'] = pipeline.named_steps['classifier'].predict_proba(X)[:, 1]

    use_case_column = f"UseCase_{use_case}"
    if use_case_column not in df_processed.columns:
        raise ValueError(f"Use case '{use_case}' not available.")

    return df_processed[df_processed[use_case_column] == 1].sort_values("data_buyer_score", ascending=False).head(top_n)[['JobTitle', 'Agency', 'data_buyer_score', use_case_column]]
