#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


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

