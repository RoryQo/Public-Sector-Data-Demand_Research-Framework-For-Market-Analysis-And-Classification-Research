# --- Suppress non-critical warnings globally for this package ---
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Re-export main functions ---
from .toolkit import (
    fetch_and_score_job,
    batch_fetch_and_score_jobs,
    search_job_ids_by_title,
    fetch_and_score_top_by_use_case_auto,
    preprocess_job_api_response,
    load_pipeline,
)
