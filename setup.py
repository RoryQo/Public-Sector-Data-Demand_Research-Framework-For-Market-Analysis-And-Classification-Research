#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup, find_packages

setup(
    name="data_buyer_toolkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "requests",
        "scikit-learn",
        "rapidfuzz",
        "joblib"
    ],
    author="Rory G. Quinlan",
    author_email="RoryQuinlan@pitt.edu",
    description="A toolkit for identifying and scoring third-party data buyers from public sector job postings (USAJobs API).",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RoryQo/Public-Sector-Data-Demand_Research-Framework-For-Market-Analysis-And-Classification",  # Update later if publishing
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

