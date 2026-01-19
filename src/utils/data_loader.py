"""
Data loading and preparation utilities
"""

import pandas as pd
import json
import ast
from pathlib import Path
from typing import Tuple, Dict


# Data paths
DATA_PATH = Path(__file__).parent.parent.parent / "Data"
X_TRAIN_FILE = DATA_PATH / "x_train_Meacfjr.csv"
Y_TRAIN_FILE = DATA_PATH / "y_train_SwJNMSu.csv"
X_TEST_FILE = DATA_PATH / "x_test_jCBBNP2.csv"
JOB_LISTINGS_FILE = DATA_PATH / "job_listings.json"


def parse_sequence(sequence_str: str):
    """
    Parse string representation of list to actual list
    
    Args:
        sequence_str: String representation of a list
        
    Returns:
        Parsed list or empty list if parsing fails
    """
    try:
        return ast.literal_eval(sequence_str)
    except:
        return []


def load_all_data() -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all data files
    
    Returns:
        Tuple of (jobs_dict, x_train, y_train, x_test)
    """
    # Load job listings
    with open(JOB_LISTINGS_FILE, 'r', encoding='utf-8') as f:
        jobs = json.load(f)
    
    # Load training and test data
    x_train = pd.read_csv(X_TRAIN_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE)
    x_test = pd.read_csv(X_TEST_FILE)
    
    return jobs, x_train, y_train, x_test


def prepare_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse job_ids and actions columns into lists
    
    Args:
        df: DataFrame with 'job_ids' and 'actions' columns
        
    Returns:
        DataFrame with additional 'jobs_list' and 'actions_list' columns
    """
    df = df.copy()
    
    # Parse sequences
    df['jobs_list'] = df['job_ids'].apply(parse_sequence)
    df['actions_list'] = df['actions'].apply(parse_sequence)
    
    # Add sequence length
    df['seq_length'] = df['jobs_list'].apply(len)
    
    return df


def load_and_prepare_data() -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all data and prepare sequences
    
    Returns:
        Tuple of (jobs_dict, x_train, y_train, x_test) with prepared sequences
    """
    jobs, x_train, y_train, x_test = load_all_data()
    
    # Prepare sequences
    x_train = prepare_sequences(x_train)
    x_test = prepare_sequences(x_test)
    
    return jobs, x_train, y_train, x_test
