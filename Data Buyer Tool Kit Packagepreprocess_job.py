#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
from rapidfuzz import fuzz, process

def preprocess_job_api_response(job_json):
    title = job_json['PositionTitle']
    agency = job_json['OrganizationName']

    desc = job_json['UserArea']['Details'].get('JobSummary', '')
    duties = job_json['UserArea']['Details'].get('MajorDuties', '')

    # FIX: if JobSummary or MajorDuties are lists, join them
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

