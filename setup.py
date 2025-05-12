#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

setup(
    name='data_buyer_toolkit',
    version='0.1.0',
    author='Rory G. Quinlan',
    description='Toolkit for identifying third-party data buyers in U.S. federal job postings from USAJobs.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'data_buyer_toolkit': ['*.joblib'],
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
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
