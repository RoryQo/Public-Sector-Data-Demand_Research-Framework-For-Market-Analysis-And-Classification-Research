#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup, find_packages

setup(
    name="data_buyer_toolkit",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "requests",
        "nltk",
        "rapidfuzz",
    ],
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    author="Rory G. Quinlan",
    description="Toolkit for identifying third-party data buyers from USAJobs postings.",
)

