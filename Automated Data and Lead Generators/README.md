# Public Sector Data Buyer Detection: End-to-End Pipeline

This project provides a fully automated pipeline for identifying third-party data demand in U.S. federal job postings. By integrating data acquisition, weak labeling, and NLP modeling, it generates a ranked list of roles likely to purchase external data, helping vendors and researchers target relevant public sector opportunities.



## Recommended Entry Point: Full Pipeline Notebook

### `Full Pipeline Combined.ipynb`

This unified notebook handles the complete process:

1. Connects to the USAJobs API to fetch real-time job postings.
2. Cleans, labels, and structures the data.
3. Applies a pre-trained NLP model to score third-party data demand.
4. Outputs a ranked list of predicted buyers to CSV.

**Before Running:**
- Adjust the `DATA_PATH` variable to set your preferred working directory.
- Insert your USAJobs API credentials in the request headers.
- Ensure pretrained `.joblib` files (model, vectorizer, pipeline) are available in the working folder.



## Modular Notebooks (Optional for Customization)

The full pipeline is also available in two modular components for deeper inspection or future adaptation.

### 1. `API Call and Full Automatic Data Generation Label Creation.ipynb`

Handles the data acquisition and labeling phase.

**Functionality:**
- Connects to the USAJobs API with configurable keyword search.
- Parses and cleans job metadata and descriptions.
- Applies labeling logic based on role attributes and keyword presence.
- Outputs a structured, labeled dataset in CSV format.

**Setup:**
- Insert your API credentials (email and Authorization-Key).
- Update the output path using the `DATA_PATH` variable.
- Customize the keyword list to target specific domains (e.g., fraud, analytics, healthcare).


### 2. `Automated Modeling and Lead Generating.ipynb`

Handles the modeling and prediction phase.

**Functionality:**
- Loads the labeled dataset.
- Applies TF-IDF text vectorization, one-hot encoding, and SMOTE class balancing.
- Uses a trained logistic regression classifier to predict data-buying likelihood.
- Outputs binary labels and continuous scores, and saves a filtered lead list.

**Setup:**
- Ensure the input file matches the output from the labeling notebook.
- Update paths using the `DATA_PATH` variable.
- Verify all required `.joblib` pipeline components are present.

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

