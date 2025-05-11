#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from setuptools import setup, find_packages

setup(
    name='data_buyer_toolkit',
    version='0.1.0',
    author='Rory G. Quinlan',
    description='Toolkit for scoring public sector job postings for third-party data demand.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),  # <--- THIS automatically finds /data_buyer_toolkit/
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'nltk',
        'requests',
        'rapidfuzz',
        'joblib'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)


