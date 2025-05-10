#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

