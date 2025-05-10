#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup, find_packages

setup(
    name='data_buyer_toolkit',
    version='0.1',
    packages=find_packages(where='Automated Data and Lead Generators/Data Buyer Tool Kit Package'),
    package_dir={'': 'Automated Data and Lead Generators/Data Buyer Tool Kit Package'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'requests',
        'nltk',
        'rapidfuzz'
    ],
    author='Rory G. Quinlan',
    description='Toolkit for scoring third-party data buyer demand in USAJobs postings',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RoryQo/Public-Sector-Data-Demand_Research-Framework-For-Market-Analysis-And-Classification',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.7'
)


