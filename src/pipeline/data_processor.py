import pandas as pd
import json
import ast
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

from src.utils.config import (
    X_TRAIN_FILE, Y_TRAIN_FILE, X_TEST_FILE, JOB_LISTINGS_FILE
)
from src.pipeline.feature_engineer import FeatureEngineer

class DataProcessor:
    def __init__(self):
        self.jobs = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.feature_engineer = FeatureEngineer()
        
    def load_data(self) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        with open(JOB_LISTINGS_FILE, 'r', encoding='utf-8') as f:
            self.jobs = json.load(f)
        
        self.x_train = pd.read_csv(X_TRAIN_FILE)
        self.y_train = pd.read_csv(Y_TRAIN_FILE)
        self.x_test = pd.read_csv(X_TEST_FILE)
        
        # Set jobs dict in feature engineer
        self.feature_engineer.set_jobs_dict(self.jobs)

        return self.jobs, self.x_train, self.y_train, self.x_test
    
    @staticmethod
    def parse_sequence(sequence_str: str) -> List:
        try:
            return ast.literal_eval(sequence_str)
        except (ValueError, SyntaxError):
            return []
    
    def prepare_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['jobs_list'] = df['job_ids'].apply(self.parse_sequence)
        df['actions_list'] = df['actions'].apply(self.parse_sequence)
        df['seq_length'] = df['jobs_list'].apply(len)
        
        return df
    
    def add_features(self, df: pd.DataFrame, feature_types: List[str] = None) -> pd.DataFrame:
        """
        Add engineered features to the dataframe
        
        Args:
            df: DataFrame with sequences prepared
            feature_types: List of feature types to add. Options: 'sequence', 'session', 'text', 'all'
                          If None or 'all', adds all features
        
        Returns:
            DataFrame with additional features
        """
        if feature_types is None or 'all' in feature_types:
            return self.feature_engineer.extract_all_features(df)
        
        df = df.copy()
        
        if 'sequence' in feature_types:
            df = self.feature_engineer.extract_sequence_features(df)
        
        if 'session' in feature_types:
            df = self.feature_engineer.extract_session_features(df)
        
        if 'text' in feature_types:
            df = self.feature_engineer.extract_job_text_features(df)
        
        return df
