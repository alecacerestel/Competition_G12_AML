import pandas as pd
import json
import ast
from pathlib import Path
from typing import List, Tuple, Dict


def load_all_data(data_dir: str = None) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if data_dir is None:
        project_root = Path(__file__).parent.parent
        data_dir = project_root / 'Data'
    else:
        data_dir = Path(data_dir)
    
    with open(data_dir / 'job_listings.json', 'r', encoding='utf-8') as f:
        jobs = json.load(f)
    
    x_train = pd.read_csv(data_dir / 'x_train_Meacfjr.csv')
    y_train = pd.read_csv(data_dir / 'y_train_SwJNMSu.csv')
    x_test = pd.read_csv(data_dir / 'x_test_jCBBNP2.csv')
    
    return jobs, x_train, y_train, x_test


def parse_sequence(sequence_str: str) -> List:
    try:
        return ast.literal_eval(sequence_str)
    except:
        return []


def prepare_sequences(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['jobs_list'] = df['job_ids'].apply(parse_sequence)
    df['actions_list'] = df['actions'].apply(parse_sequence)
    df['seq_length'] = df['jobs_list'].apply(len)
    return df
