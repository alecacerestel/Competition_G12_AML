"""
Configuration for the Job Recommendation Pipeline.
Lightweight hyperparameters following Collaborative Filtering approach.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for the pipeline."""
    
    # Paths
    data_dir: Path = Path("Data")
    output_dir: Path = Path("experiments")
    
    # Data files
    x_train_file: str = "x_train_Meacfjr.csv"
    y_train_file: str = "y_train_SwJNMSu.csv"
    x_test_file: str = "x_test_jCBBNP2.csv"
    job_listings_file: str = "job_listings.json"
    
    # Collaborative Filtering parameters
    k_similar_sessions: int = 50       # Number of similar sessions to use
    min_similarity: float = 0.01       # Minimum similarity threshold
    
    # Job Ranking weights (tuned via grid search)
    transition_weight: float = 3.0     # Weight for direct transitions (job A -> job B)
    cooccurrence_weight: float = 3.0   # Weight for session co-occurrence
    cf_weight: float = 2.0             # Weight for collaborative filtering score
    popularity_weight: float = 0.1     # Weight for job popularity
    
    # Action prediction
    action_threshold: float = 0.5      # Threshold for apply vs view
    
    # Evaluation
    val_split: float = 0.15            # Validation split ratio
    top_k: int = 10                    # Top-K jobs to recommend
    random_seed: int = 42
    
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)


# Default configuration
DEFAULT_CONFIG = Config()
