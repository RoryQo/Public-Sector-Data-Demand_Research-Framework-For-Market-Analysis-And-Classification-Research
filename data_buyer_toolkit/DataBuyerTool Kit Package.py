#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
from rapidfuzz import fuzz, process

def preprocess_job_api_response(job_json):
    title = job_json['PositionTitle']
    agency = job_json['OrganizationName']

    desc = job_json['UserArea']['Details'].get('JobSummary', '')
    duties = job_json['UserArea']['Details'].get('MajorDuties', '')

  
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

    # IsDataBuyer
    related_phrases = [
        "data acquisition", "data procurement", "procure data", "purchase data",
        "buy data", "acquiring data", "data sourcing", "data licensing", 
        "external data acquisition", "third-party data", "data vendor", 
        "data provider", "data contracts", "contracting data", "data subscriptions",
        "vendor management", "external data", "commercial data"
    ]
    pattern = '|'.join([re.escape(phrase) for phrase in related_phrases])
    df['IsDataBuyer'] = df['CombinedText'].str.contains(pattern, case=False, na=False).astype(int)

    # FuzzyMatchedPhrase and IsFuzzyMatch
    signal_phrases = [
        "data acquisition", "data procurement", "procure data", "purchase data",
        "buy data", "acquiring data", "data sourcing", "data licensing", 
        "external data", "third-party data", "data vendor", 
        "data provider", "data contracts", "contracting data", "data subscriptions",
        "vendor management", "commercial data", "data assets", "data commercialization",
        "procurement of data", "external data sources", "data aggregators",
        "data monetization", "sourcing external data", "partner data", "data purchasing agreements",
        "data ingestion", "subscription data", "data acquisition strategy", "data buying",
        "external datasets", "external partnerships", "data sharing agreements",
        "data acquisition channels", "third-party data sources", "sourcing data providers",
        "managing data vendors", "data reseller", "external data vendors", "contracted data"
    ]

    def fuzzy_match_phrases(text, phrases, threshold=80):
        for phrase in phrases:
            score = fuzz.partial_ratio(phrase.lower(), text.lower())
            if score >= threshold:
                return phrase
        return None

    df['FuzzyMatchedPhrase'] = df['CombinedText'].apply(lambda x: fuzzy_match_phrases(x, signal_phrases))
    df['IsFuzzyMatch'] = df['FuzzyMatchedPhrase'].notnull().astype(int)

    # IsLikelyDataBuyer
    df['IsLikelyDataBuyer'] = ((df['IsDataBuyer'] == 1) | (df['IsFuzzyMatch'] == 1)).astype(int)

    # AgencySize
    large_agencies = [
        "Department of Defense", "Department of Veterans Affairs", "Department of the Treasury",
        "Department of Homeland Security", "Department of Health and Human Services",
        "Department of Justice", "Department of the Army"
    ]
    medium_agencies = [
        "Department of Transportation", "Department of Commerce", "Department of Agriculture",
        "Department of Energy", "Department of the Interior", "National Aeronautics and Space Administration"
    ]

    def classify_agency_size(agency):
        if agency in large_agencies:
            return 'Large'
        elif agency in medium_agencies:
            return 'Medium'
        else:
            return 'Small'

    df['AgencySize'] = df['Agency'].apply(classify_agency_size).fillna('Unknown')

    # Industry classifier
    def classify_industry(row):
        text = f"{row['JobTitle']} {row['Agency']}".lower()
        if any(x in text for x in ['finance', 'financial', 'account', 'budget']):
            return 'Finance'
        elif any(x in text for x in ['marketing', 'communications', 'advertising']):
            return 'Marketing'
        elif any(x in text for x in ['medical', 'pharmacy', 'nurse', 'health', 'clinical']):
            return 'Medical'
        elif any(x in text for x in ['cyber', 'security', 'information technology', 'it', 'data scientist', 'software', 'tech']):
            return 'Security/Tech'
        elif any(x in text for x in ['policy', 'regulation', 'legislative', 'analyst', 'compliance']):
            return 'Policy'
        else:
            return 'Other'

    df['Industry'] = df.apply(classify_industry, axis=1).fillna('Other')

    # IsSeniorRole
    senior_keywords = ['senior', 'lead', 'chief', 'principal', 'director', 'head']
    df['IsSeniorRole'] = df['JobTitle'].str.lower().str.contains('|'.join(senior_keywords), na=False)

    # IsExplicitDataJob
    data_keywords = ['data', 'analyst', 'scientist', 'analytics', 'it', 'information', 'statistician', 'intelligence']
    df['IsExplicitDataJob'] = df['JobTitle'].str.lower().str.contains('|'.join(data_keywords), na=False).astype(int)

    # Use Cases
    use_case_keywords = {
        'Fraud': ['fraud', 'eligibility', 'verification', 'audit', 'compliance'],
        'Sentiment': ['sentiment', 'public opinion', 'media monitoring', 'engagement', 'communication'],
        'PatientMatching': ['patient match', 'interoperability', 'record linkage', 'ehr', 'health record'],
        'AdTargeting': ['audience segmentation', 'targeting', 'ad performance', 'campaign data']
    }
    for use_case, keywords in use_case_keywords.items():
        pattern = '|'.join(keywords)
        df[f'UseCase_{use_case}'] = df['CombinedText'].str.lower().str.contains(pattern, na=False).astype(int)

    # IsGeneralistRole
    generalist_titles = [
        'Contract Specialist', 'Grants Officer', 'Grants Specialist', 'Budget Officer',
        'Administrative Officer', 'Operations Coordinator', 'Program Coordinator',
        'Project Coordinator', 'Procurement Specialist', 'Procurement Analyst',
        'Communications Specialist', 'Public Affairs Officer', 'Public Information Officer',
        'Community Outreach Coordinator', 'Health IT Coordinator', 'Program Specialist',
        'Program Manager', 'Business Operations Specialist'
    ]

    def is_generalist(title, threshold=65):
        match, score, _ = process.extractOne(title, generalist_titles, scorer=fuzz.partial_ratio)
        return score >= threshold

    df['IsGeneralistRole'] = df['JobTitle'].apply(lambda x: is_generalist(str(x)))

    return df


# In[2]:


import joblib
import requests

def fetch_and_score_job(job_id, api_key, email, pipeline_path="nlp_pipeline_with_smote.joblib"):
    headers = {
        "User-Agent": email,
        "Authorization-Key": api_key
    }

    url = f"https://data.usajobs.gov/api/Search?Keyword={job_id}"

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch job ID {job_id}: {response.status_code}")

    job_data = response.json()['SearchResult']['SearchResultItems'][0]['MatchedObjectDescriptor']
    df_processed = preprocess_job_api_response(job_data)

    pipeline = joblib.load(pipeline_path)
    X = pipeline.named_steps['preprocessor'].transform(df_processed)
    score = pipeline.named_steps['classifier'].predict_proba(X)[0][1]

    return {
        "data_buyer_score": round(score, 4),
        "title": job_data['PositionTitle'],
        "agency": job_data['OrganizationName']
        
    }


# In[3]:


import requests

def search_job_ids_by_title(position_title, api_key, email, max_results=10):
    headers = {
        "User-Agent": email,
        "Authorization-Key": api_key
    }

    url = f"https://data.usajobs.gov/api/Search"

    params = {
        "Keyword": position_title,
        "ResultsPerPage": max_results
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(f"Failed to search: {response.status_code}")

    jobs = response.json()['SearchResult']['SearchResultItems']
    results = []
    for job in jobs:
        job_id = job['MatchedObjectDescriptor']['PositionID']
        title = job['MatchedObjectDescriptor']['PositionTitle']
        agency = job['MatchedObjectDescriptor']['OrganizationName']
        results.append({
            "job_id": job_id,
            "title": title,
            "agency": agency
        })

    return results


# In[4]:


def batch_fetch_and_score_jobs(job_titles, api_key, email, pipeline_path="nlp_pipeline_with_smote.joblib"):
    results = []
    for title in job_titles:
        try:
            search_results = search_job_ids_by_title(title, api_key, email, max_results=1)
            if search_results:
                job_id = search_results[0]['job_id']
                scored = fetch_and_score_job(job_id, api_key, email, pipeline_path)
                results.append(scored)
        except Exception as e:
            print(f"Error processing {title}: {e}")

    return pd.DataFrame(results)


# In[17]:


import pandas as pd
import requests
import joblib

def fetch_and_score_top_by_use_case_auto(api_key, email, use_case="Fraud", top_n=100, pipeline_path="nlp_pipeline_with_smote.joblib"):
    """
    Fetches jobs live from USAJobs API using a predefined keyword list,
    scores them, and returns the top N jobs matching a specified use case.

    Args:
        api_key (str): Your USAJobs API Key.
        email (str): Your registered email for USAJobs API.
        use_case (str): Use case category ('Fraud', 'Sentiment', 'PatientMatching', 'AdTargeting').
        top_n (int): Number of top jobs to return.
        pipeline_path (str): Path to your saved modeling pipeline (.joblib).

    Returns:
        pd.DataFrame: Top N scored jobs matching the specified use case.
    """

    headers = {
        "User-Agent": email,
        "Authorization-Key": api_key
    }

    # Define search keywords
    keywords = [
        'data', 'contract', 'analyst', 'machine learning', 'marketing', 'aquisition',
        'finance', 'security', 'tech', 'purchasing', 'statistics', 'math', 'data scientist',
        'research', 'economist'
    ]

    url = "https://data.usajobs.gov/api/Search"
    all_jobs = {}

    # Loop over each keyword
    for keyword in keywords:
        print(f"Searching for keyword: {keyword}")
        params = {
            'Keyword': keyword,
            'ResultsPerPage': 500,
            'Page': 1
        }

        while True:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                break

            data = response.json()
            jobs = data.get('SearchResult', {}).get('SearchResultItems', [])
            if not jobs:
                break

            for job in jobs:
                job_id = job.get('MatchedObjectId')
                descriptor = job.get('MatchedObjectDescriptor', {})
                details = descriptor.get('UserArea', {}).get('Details', {})

                if job_id not in all_jobs:
                    all_jobs[job_id] = {
                        'JobID': job_id,
                        'JobTitle': descriptor.get('PositionTitle'),
                        'JobDescription': details.get('JobSummary'),
                        'KeyDuties': details.get('MajorDuties', 'N/A'),
                        'Department': descriptor.get('OrganizationName'),
                        'Agency': descriptor.get('DepartmentName'),
                        'SearchKeywords': [keyword]
                    }
                else:
                    if keyword not in all_jobs[job_id]['SearchKeywords']:
                        all_jobs[job_id]['SearchKeywords'].append(keyword)

            print(f"Retrieved page {params['Page']} with {len(jobs)} jobs")
            params['Page'] += 1

    # Convert dictionary to DataFrame
    df = pd.DataFrame(all_jobs.values())

    if df.empty:
        raise ValueError("No jobs found across all keywords.")

    # Preprocess all jobs
    processed_jobs = []
    for _, row in df.iterrows():
        job_json = {
            'PositionTitle': row['JobTitle'],
            'OrganizationName': row['Agency'],
            'UserArea': {
                'Details': {
                    'JobSummary': row['JobDescription'],
                    'MajorDuties': row['KeyDuties'],
                    'JobCategory': ', '.join(row['SearchKeywords'])
                }
            }
        }
        processed_job = preprocess_job_api_response(job_json)
        processed_jobs.append(processed_job)

    df_processed = pd.concat(processed_jobs, ignore_index=True)

    # Load pipeline and predict
    pipeline = joblib.load(pipeline_path)
    X = pipeline.named_steps['preprocessor'].transform(df_processed)
    scores = pipeline.named_steps['classifier'].predict_proba(X)[:, 1]
    df_processed['data_buyer_score'] = scores

    # Filter by use case
    use_case_column = f"UseCase_{use_case}"
    if use_case_column not in df_processed.columns:
        raise ValueError(f"Use case '{use_case}' is not available.")

    filtered = df_processed[df_processed[use_case_column] == 1]
    ranked = filtered.sort_values("data_buyer_score", ascending=False).head(top_n)

    return ranked[['JobTitle', 'Agency', 'data_buyer_score', use_case_column]]


# In[18]:


import pandas as pd
import requests
import joblib

def fetch_and_score_top_by_custom_inputs(api_key, email, search_keywords, use_case, top_n=100, pipeline_path="nlp_pipeline_with_smote.joblib"):
    """
    Fetches jobs live from USAJobs API using user-specified search keywords,
    scores them, and returns the top N jobs matching a specified use case.

    Args:
        api_key (str): Your USAJobs API Key.
        email (str): Your registered email for USAJobs API.
        search_keywords (list): List of search keywords to use for the query.
        use_case (str): Use case category ('Fraud', 'Sentiment', 'PatientMatching', 'AdTargeting').
        top_n (int): Number of top jobs to return.
        pipeline_path (str): Path to your saved modeling pipeline (.joblib).

    Returns:
        pd.DataFrame: Top N scored jobs matching the specified use case.
    """

    if not search_keywords or not isinstance(search_keywords, list):
        raise ValueError("You must provide a list of search keywords.")
    if not isinstance(use_case, str):
        raise ValueError("You must provide a use case string.")

    headers = {
        "User-Agent": email,
        "Authorization-Key": api_key
    }

    url = "https://data.usajobs.gov/api/Search"
    all_jobs = {}

    for keyword in search_keywords:
        print(f"Searching for keyword: {keyword}")
        params = {
            'Keyword': keyword,
            'ResultsPerPage': 500,
            'Page': 1
        }

        while True:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                break

            data = response.json()
            jobs = data.get('SearchResult', {}).get('SearchResultItems', [])
            if not jobs:
                break

            for job in jobs:
                job_id = job.get('MatchedObjectId')
                descriptor = job.get('MatchedObjectDescriptor', {})
                details = descriptor.get('UserArea', {}).get('Details', {})

                if job_id not in all_jobs:
                    all_jobs[job_id] = {
                        'JobID': job_id,
                        'JobTitle': descriptor.get('PositionTitle'),
                        'JobDescription': details.get('JobSummary'),
                        'KeyDuties': details.get('MajorDuties', 'N/A'),
                        'Department': descriptor.get('OrganizationName'),
                        'Agency': descriptor.get('DepartmentName'),
                        'SearchKeywords': [keyword]
                    }
                else:
                    if keyword not in all_jobs[job_id]['SearchKeywords']:
                        all_jobs[job_id]['SearchKeywords'].append(keyword)

            print(f"Retrieved page {params['Page']} with {len(jobs)} jobs")
            params['Page'] += 1

    df = pd.DataFrame(all_jobs.values())

    if df.empty:
        raise ValueError("No jobs found across provided keywords.")

    # Preprocess all jobs
    processed_jobs = []
    for _, row in df.iterrows():
        job_json = {
            'PositionTitle': row['JobTitle'],
            'OrganizationName': row['Agency'],
            'UserArea': {
                'Details': {
                    'JobSummary': row['JobDescription'],
                    'MajorDuties': row['KeyDuties'],
                    'JobCategory': ', '.join(row['SearchKeywords'])
                }
            }
        }
        processed_job = preprocess_job_api_response(job_json)
        processed_jobs.append(processed_job)

    df_processed = pd.concat(processed_jobs, ignore_index=True)

    # Load pipeline and predict
    pipeline = joblib.load(pipeline_path)
    X = pipeline.named_steps['preprocessor'].transform(df_processed)
    scores = pipeline.named_steps['classifier'].predict_proba(X)[:, 1]
    df_processed['data_buyer_score'] = scores

    # Filter by use case
    use_case_column = f"UseCase_{use_case}"
    if use_case_column not in df_processed.columns:
        raise ValueError(f"Use case '{use_case}' not available.")

    filtered = df_processed[df_processed[use_case_column] == 1]
    ranked = filtered.sort_values("data_buyer_score", ascending=False).head(top_n)

    return ranked[['JobTitle', 'Agency', 'data_buyer_score', use_case_column]]

