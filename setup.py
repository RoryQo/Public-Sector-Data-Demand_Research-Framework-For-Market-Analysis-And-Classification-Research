#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

setup(
    name='data_buyer_toolkit',
    version='0.1.0',
    author='Rory G. Quinlan',
    description='Toolkit for scoring USA Jobs open position postings for third-party data demand.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),  # Finds /data_buyer_toolkit/
    include_package_data=True,  # <--- tell setuptools to look for extra files
    package_data={              # <--- explicitly list the joblib files
        'data_buyer_toolkit': [
            'nlp_model.joblib',
            'nlp_pipeline_with_smote.joblib',
            'vectorizer.joblib'
        ]
    },
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
