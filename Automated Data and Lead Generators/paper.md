---
title: 'A Lead Generation Tool for Third-Party Data Demand via U.S. Public Job Postings'
tags:
  - Python 
  - Demand Modeling
  - Market Research
  - Economics
authors:
  - name: Rory G. Quinlan
    orcid: 0009-0006-7483-6769
    equal-contrib: true
    affiliation: "1"
affiliations:
  - name: University of Pittsburgh
    index: 1
    ror: 00hx57361
date: 03 May 2025
bibliography: paper.bib
nocite: |
  @*
---

# Summary

This project is a tool designed to help identify the demand for external data and target potential customers in the public sector using available job posts. The tool analyzes details of U.S. government job postings via USAJobs. First, it creates structured features such as seniority, agency size, and industries. Second, it implements natural language processing (NLP) techniques to analyze the text in the job descriptions to identify positions likely to engage in purchasing external data. Each potential data buyer is assigned a data buyer score, which reflects the likelihood that the position is data-buying; this feature enables easy prioritization of future outreach and targeting. 

The framework is fully automated through two modular scripts and a standalone package. The modular scripts enable automatic and real-time lead generation and scoring, allowing demand assessment and trend analysis. The data_buyer_toolkit package, available on PyPI, was developed to integrate the same framework and enhance flexibility and reusability. This package allows users to score specific postings, identify leads by use case or industry, and map job IDs from titles of interest, providing more control and customization than the end-to-end automated scripts. The package and scripts are easy to reuse, modify to target various industries, and assess demand in real time, or expand to explore demand seasonality.

# Statement of need

Demand for external data in businesses has grown exponentially across
industries. Despite this demand, public procurement records are
often unavailable or do not identify the roles or departments responsible for the
purchasing. This lack of transparency leads to market inefficiencies in which data
vendors cannot reliably connect with or identify clients.

This project addresses this inefficiency by creating a reproducible framework
to estimate this demand through job descriptions in the available posts. Additionally, it outputs a ranked list of job titles and agencies based on their likelihood of being data buyers, enabling researchers and data vendors to identify likely clients and prioritize their outreach based on these scores.


# Acknowledgments

This project was completed during master's degree studies at the University of Pittsburgh. We thank Editage (https://editage.com) for English language editing support and acknowledge the university faculty for their support and instruction throughout the program.

\clearpage

# References
::: {#refs}
:::

