import pandas as pd
import json
import ast
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

from utils.config import (
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
    
    def parse_sequence(self, sequence_str: str) -> List:
        try:
            return ast.literal_eval(sequence_str)
        except:
            return []

    def prepare_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['jobs_list'] = df['job_ids'].apply(self.parse_sequence)
        df['actions_list'] = df['actions'].apply(self.parse_sequence)
        df['seq_length'] = df['jobs_list'].apply(len)
        
        return df
    
    def split_train_test(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide x_train e y_train en sets de entrenamiento y testing.
        
        Args:
            test_size: Proporción del conjunto de prueba (por defecto 0.2)
            random_state: Semilla para reproducibilidad
            
        Returns:
            Tupla con (x_train_split, x_test_split, y_train_split, y_test_split)
        """
        if self.x_train is None or self.y_train is None:
            raise ValueError("Primero debe cargar los datos con load_data()")
        
        x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
            self.x_train, 
            self.y_train, 
            test_size=test_size, 
            random_state=random_state
        )
        
        return x_train_split, x_test_split, y_train_split, y_test_split
