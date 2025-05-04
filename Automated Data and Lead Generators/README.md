This project is fully automated from **data acquisition to model deployment**, using two modular notebooks:

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

It is fully automated and will automatically download the wrangled data to your designated file path.
Output is saved in a structured format (CSV).

**All keyword lists are fully editable**, allowing you to adapt tagging logic to reflect new tools, vendors, and terminology as the public sector data market evolves

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

###  Reusability
All model and feature pipelines are stored in `.pkl` files (included in this repo). This allows:
- Plug-and-play reruns on new job data
- Scalable deployment on different job boards
- Consistent inference without retraining

## Dependencies

This project requires the following Python packages:

- `pandas` – for data manipulation  
- `numpy` – for numerical operations  
- `scikit-learn` – for TF-IDF vectorization, SMOTE, and logistic regression modeling  
- `nltk` – for text preprocessing (tokenization, stop word removal)  
- `requests` – for accessing the USAJOBS API  
- `json` – for parsing API responses  
- `matplotlib` – for plotting score distributions and outputs  
- `tqdm` – for progress tracking during data collection  
- `fuzzywuzzy` or `rapidfuzz` – for fuzzy string matching in weak supervision

All dependencies are listed in `requirements.txt`.

> **Note**: You may need to run `nltk.download('stopwords')` and `nltk.download('punkt')` the first time you use the text preprocessing functions.
