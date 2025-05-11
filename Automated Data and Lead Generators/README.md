# Public Sector Data Buyer Detection: End-to-End Pipeline

This project provides a fully automated pipeline for identifying third-party data demand in U.S. federal job postings.
Integrating data acquisition, weak and supervised labeling, structured feature engineering, and NLP modeling generates a ranked list of roles likely to purchase external data, helping vendors and researchers target relevant public sector opportunities.

The project is fully automated from data acquisition to model deployment, using two modular notebooks,
and is supplemented by a modular Python package for easy integration of scoring and search functions into custom workflows.

This project is fully automated from **data acquisition to model deployment**, using two modular notebooks:

## 0.0 Data Buyer Toolkit (Python Package)

In addition to the main notebooks, this repository includes a standalone python package.

The **Data Buyer Toolkit** provides lightweight, reusable Python functions for:

- **Fetching and scoring a single job by ID** (`fetch_and_score_job`)
- **Searching USAJobs live by keyword and scoring results** (`search_and_score_keyword_live`)
- **Batch processing and scoring multiple job titles** (`batch_fetch_and_score_jobs`)

These functions allow users to **integrate third-party data buyer detection into their own workflows**,  
without needing to manually run the full notebooks.

👉 **See the `data_buyer_toolkit` folder for installation instructions and usage examples.**


## 1. API Call and Full Automatic Data Generation Label Creation.ipynb
- Connects to the official **USAJobs API** and pulls federal job postings based on configurable search criteria
- Cleans and parses returned data, standardizes fields, creates meta-data, and formats for modeling

**⚠️ Setup Instructions Before Running**

**Update file paths:**
Use Ctrl + F to find and replace all instances of
C://Users//...// with your local directory path.

**Insert your API credentials:**

Replace the placeholder Authorization-Key with your own key from developer.usajobs.gov.

Ensure the email address used in the API header matches the one registered with your key.

**All keyword lists are fully editable**, allowing you to adapt tagging logic to reflect new tools, vendors, and terminology as the public sector data market evolves
This unified notebook handles the complete process:

## 2. Automated Modeling and Lead Generating.ipynb
- Loads labeled job data and feeds it into a **prebuilt NLP modeling pipeline**
- Applies preprocessing:
  - TF-IDF text vectorization
  - One-hot encoding of structured fields
  - SMOTE-based class balancing
- Generates predictions using a trained **NLP logistic regression model**
- Outputs:
  - `PredictedLabel` (0/1)
  - `DataBuyerScore` (likelihood ∈ [0,1])
- Automatically **exports a ranked lead list** of high-priority public sector buyer roles
1. Connects to the USAJobs API to fetch real-time job postings.
2. Cleans, labels, and structures the data.
3. Applies a pre-trained NLP model to score third-party data demand.
4. Outputs a ranked list of predicted buyers to CSV.

**⚠️ Setup Instructions Before Running**

**Set File Paths**
Replace all file paths in the notebook with your own local paths.
Use Ctrl + F to search for and replace all instances of the default path (C://Users/.../) with your desired directory.

**Data File Consistency**
Ensure the input CSV file has the same name and format as the one created by the automated data acquisition script.

>If you used the USAJobs fetcher provided in this repository, no changes are needed.

**Pipeline Files**
The notebook depends on pretrained pipeline files (.pkl files for vectorizers, transformers, and the classifier), which are included in this repository.
Be sure they are downloaded and saved in the correct folder as referenced in the notebook.

##  Dependencies

This project requires the following Python packages:

- pandas  
- numpy  
- scikit-learn  
- nltk  
- requests  
- json  
- matplotlib  
- tqdm  
- fuzzywuzzy or rapidfuzz  

All dependencies are listed in `requirements.txt`.



## Reusability & Extension

- All model components are stored in reusable `.joblib` files.
- The pipeline can be rerun on any new job dataset by changing the input file path.
- Easily adaptable for new labeling rules, different job boards, or extended use cases in public sector analytics.


> **Note**: You may need to run `nltk.download('stopwords')` and `nltk.download('punkt')` the first time you use the text preprocessing functions.


## 🔒 License

This project is licensed under the **MIT License**, an OSI-approved permissive license that allows reuse, modification, and distribution with proper attribution.

See the [LICENSE](./LICENSE) file for full terms.


## Future Contributions

Contributions are welcome and encouraged as the public-sector data landscape evolves!

### How to Contribute
1. Fork this repository
2. Create a new branch: `git checkout -b feature/my-feature`
3. Make your changes and commit them
4. Push to your fork: `git push origin feature/my-feature`
5. Open a pull request describing your proposed change

For major changes, please open an issue first to discuss scope and direction.

