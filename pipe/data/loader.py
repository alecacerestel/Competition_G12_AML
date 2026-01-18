"""
Data loading utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
import json


def parse_sequence(s: str) -> List[int]:
    """Parse a string representation of a list into actual list of ints."""
    if isinstance(s, str):
        s = s.strip('[]')
        if not s:
            return []
        return [int(x.strip()) for x in s.split(',') if x.strip()]
    elif hasattr(s, '__iter__'):
        return list(s)
    return [s] if s else []


def parse_actions(s: str) -> List[str]:
    """Parse action sequence string into list of actions."""
    if isinstance(s, str):
        s = s.strip('[]').replace("'", "").replace('"', '')
        if not s:
            return []
        return [x.strip() for x in s.split(',') if x.strip()]
    elif hasattr(s, '__iter__'):
        return list(s)
    return [s] if s else []


def load_train_data(data_dir: Path, x_file: str, y_file: str) -> pd.DataFrame:
    """Load and merge training data."""
    x_train = pd.read_csv(data_dir / x_file)
    y_train = pd.read_csv(data_dir / y_file)
    
    # Merge on session_id
    train_df = x_train.merge(y_train, on='session_id')
    
    # Parse sequences
    train_df['job_ids'] = train_df['job_ids'].apply(parse_sequence)
    train_df['actions'] = train_df['actions'].apply(parse_actions)
    
    return train_df


def load_test_data(data_dir: Path, x_file: str) -> pd.DataFrame:
    """Load test data."""
    x_test = pd.read_csv(data_dir / x_file)
    
    # Parse sequences
    x_test['job_ids'] = x_test['job_ids'].apply(parse_sequence)
    x_test['actions'] = x_test['actions'].apply(parse_actions)
    
    return x_test


def load_job_listings(data_dir: Path, file: str) -> Dict[int, str]:
    """Load job listings (job_id -> description)."""
    with open(data_dir / file, 'r') as f:
        data = json.load(f)
    
    # Convert to dict with int keys
    return {int(k): v for k, v in data.items()}


def build_job_vocabulary(train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build job vocabulary (job_id <-> index mapping)."""
    all_jobs = set()
    
    # From training data
    for jobs in train_df['job_ids']:
        all_jobs.update(jobs)
    if 'job_id' in train_df.columns:
        all_jobs.update(train_df['job_id'].tolist())
    
    # From test data if provided
    if test_df is not None:
        for jobs in test_df['job_ids']:
            all_jobs.update(jobs)
    
    # Create mappings
    job_to_idx = {job: idx for idx, job in enumerate(sorted(all_jobs))}
    idx_to_job = {idx: job for job, idx in job_to_idx.items()}
    
    return job_to_idx, idx_to_job


def split_train_val(
    train_df: pd.DataFrame, 
    val_ratio: float = 0.15, 
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into train and validation sets."""
    np.random.seed(seed)
    
    val_size = int(len(train_df) * val_ratio)
    val_indices = np.random.choice(len(train_df), val_size, replace=False)
    train_indices = np.array([i for i in range(len(train_df)) if i not in set(val_indices)])
    
    val_df = train_df.iloc[val_indices].reset_index(drop=True)
    train_split = train_df.iloc[train_indices].reset_index(drop=True)
    
    return train_split, val_df


class DataLoader:
    """Central data loading and management class."""
    
    def __init__(self, config):
        self.config = config
        self.train_df = None
        self.test_df = None
        self.job_to_idx = None
        self.idx_to_job = None
        
    def load(self) -> 'DataLoader':
        """Load all data."""
        print("Loading data...")
        
        # Load training data
        self.train_df = load_train_data(
            self.config.data_dir,
            self.config.x_train_file,
            self.config.y_train_file
        )
        print(f"  Training samples: {len(self.train_df)}")
        
        # Load test data
        self.test_df = load_test_data(
            self.config.data_dir,
            self.config.x_test_file
        )
        print(f"  Test samples: {len(self.test_df)}")
        
        # Build vocabulary
        self.job_to_idx, self.idx_to_job = build_job_vocabulary(
            self.train_df, self.test_df
        )
        print(f"  Total unique jobs: {len(self.job_to_idx)}")
        
        return self
    
    def get_train_val_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get train/validation split."""
        return split_train_val(
            self.train_df, 
            self.config.val_split, 
            self.config.random_seed
        )
