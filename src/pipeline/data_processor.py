import pandas as pd
import json
import ast
from pathlib import Path
from typing import List, Tuple, Dict

from src.utils.config import (
    X_TRAIN_FILE, Y_TRAIN_FILE, X_TEST_FILE, JOB_LISTINGS_FILE
)

class DataProcessor:
    def __init__(self):
        self.jobs = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        
    def load_data(self) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        with open(JOB_LISTINGS_FILE, 'r', encoding='utf-8') as f:
            self.jobs = json.load(f)
        
        self.x_train = pd.read_csv(X_TRAIN_FILE)
        self.y_train = pd.read_csv(Y_TRAIN_FILE)
        self.x_test = pd.read_csv(X_TEST_FILE)

        return self.jobs, self.x_train, self.y_train, self.x_test
    
    @staticmethod
    def parse_sequence(sequence_str: str) -> List:
        try:
            return ast.literal_eval(sequence_str)
        except Exception:
            return []

    @staticmethod
    def prepare_sequences(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["jobs_list"] = df["job_ids"].apply(DataProcessor.parse_sequence)
        df["actions_list"] = df["actions"].apply(DataProcessor.parse_sequence)
        df["seq_length"] = df["jobs_list"].apply(len)
        return df
