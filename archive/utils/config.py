from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

# =============================================================================
# DATA CONFIGURATIONS
# =============================================================================

X_TRAIN_FILE = DATA_PATH / "x_train_Meacfjr.csv"
Y_TRAIN_FILE = DATA_PATH / "y_train_SwJNMSu.csv"
X_TEST_FILE = DATA_PATH / "x_test_jCBBNP2.csv"
JOB_LISTINGS_FILE = DATA_PATH / "job_listings.json"