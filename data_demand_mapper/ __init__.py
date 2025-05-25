# --- Suppress non-critical warnings globally for this package ---
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Re-export main functions ---
from .toolkit import (
    load_pipeline,
    preprocess_job_api_response,
    fetch_and_score_job,
    search_job_ids_by_title,
    batch_fetch_and_score_jobs,
    fetch_and_score_top_by_use_case_auto,
    fetch_top_data_buyers_by_industry_auto,
    fetch_and_score_top_by_use_case_custom,
    fetch_top_data_buyers_by_industry_custom,
)
