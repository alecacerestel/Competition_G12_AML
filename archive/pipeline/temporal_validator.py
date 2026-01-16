"""
Temporal Cross-Validation for Job Recommendation System

This module implements temporal cross-validation strategies that respect the 
time-ordered nature of session data. Unlike random splits, temporal CV ensures
that training data always comes before validation data, simulating real-world
prediction scenarios.

Strategies implemented:
1. TimeSeriesSplit - Standard sklearn-style time series CV
2. ExpandingWindow - Training window expands over time
3. SlidingWindow - Fixed-size sliding training window
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Dict, Any
from dataclasses import dataclass


@dataclass
class CVFold:
    """Represents a single cross-validation fold."""
    fold_number: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_size: int
    val_size: int


class TemporalValidator:
    """
    Temporal cross-validation for time-ordered session data.
    
    Since sessions are ordered by session_id (which corresponds to time),
    we use this ordering to create proper train/validation splits.
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 0):
        """
        Initialize the temporal validator.
        
        Args:
            n_splits: Number of cross-validation folds
            gap: Number of samples to skip between train and validation
                 (useful if there's temporal leakage concerns)
        """
        self.n_splits = n_splits
        self.gap = gap
        
    def expanding_window_split(
        self, 
        X: pd.DataFrame, 
        y: pd.DataFrame = None,
        min_train_size: float = 0.3
    ) -> Generator[CVFold, None, None]:
        """
        Generate expanding window cross-validation splits.
        
        The training window starts at min_train_size and expands to include
        more historical data in each fold, while validation always uses the
        next unseen portion.
        
        Args:
            X: Features DataFrame (assumes sorted by session_id)
            y: Target DataFrame (optional, for info only)
            min_train_size: Minimum training size as fraction of total data
            
        Yields:
            CVFold objects with train/validation indices
        """
        n_samples = len(X)
        min_train_samples = int(n_samples * min_train_size)
        
        # Calculate fold size for validation
        remaining = n_samples - min_train_samples - self.gap
        fold_size = remaining // self.n_splits
        
        for fold in range(self.n_splits):
            # Training: from start to current position
            train_end = min_train_samples + (fold * fold_size)
            val_start = train_end + self.gap
            val_end = val_start + fold_size
            
            if val_end > n_samples:
                val_end = n_samples
            
            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)
            
            if len(val_indices) == 0:
                continue
                
            yield CVFold(
                fold_number=fold + 1,
                train_indices=train_indices,
                val_indices=val_indices,
                train_size=len(train_indices),
                val_size=len(val_indices)
            )
    
    def sliding_window_split(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        train_window_size: float = 0.5,
        val_window_size: float = 0.1
    ) -> Generator[CVFold, None, None]:
        """
        Generate sliding window cross-validation splits.
        
        Uses a fixed-size training window that slides forward in time,
        simulating a production scenario with limited historical data.
        
        Args:
            X: Features DataFrame
            y: Target DataFrame (optional)
            train_window_size: Training window size as fraction of total data
            val_window_size: Validation window size as fraction of total data
            
        Yields:
            CVFold objects with train/validation indices
        """
        n_samples = len(X)
        train_size = int(n_samples * train_window_size)
        val_size = int(n_samples * val_window_size)
        
        # Calculate step size to get n_splits
        total_needed = train_size + self.gap + val_size
        step = (n_samples - total_needed) // (self.n_splits - 1) if self.n_splits > 1 else 0
        
        for fold in range(self.n_splits):
            train_start = fold * step
            train_end = train_start + train_size
            val_start = train_end + self.gap
            val_end = val_start + val_size
            
            if val_end > n_samples:
                val_end = n_samples
                val_start = max(train_end + self.gap, n_samples - val_size)
            
            if train_end >= n_samples or val_start >= n_samples:
                continue
                
            train_indices = np.arange(train_start, train_end)
            val_indices = np.arange(val_start, val_end)
            
            if len(val_indices) == 0:
                continue
                
            yield CVFold(
                fold_number=fold + 1,
                train_indices=train_indices,
                val_indices=val_indices,
                train_size=len(train_indices),
                val_size=len(val_indices)
            )
    
    def time_series_split(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None
    ) -> Generator[CVFold, None, None]:
        """
        Generate time series cross-validation splits (sklearn style).
        
        This is equivalent to expanding window where validation size equals
        the growth in training size between folds.
        
        Args:
            X: Features DataFrame
            y: Target DataFrame (optional)
            
        Yields:
            CVFold objects with train/validation indices
        """
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        
        for fold in range(self.n_splits):
            train_end = (fold + 1) * fold_size
            val_start = train_end + self.gap
            val_end = val_start + fold_size
            
            if val_end > n_samples:
                val_end = n_samples
            
            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)
            
            if len(val_indices) == 0:
                continue
                
            yield CVFold(
                fold_number=fold + 1,
                train_indices=train_indices,
                val_indices=val_indices,
                train_size=len(train_indices),
                val_size=len(val_indices)
            )
    
    def get_splits_info(self, X: pd.DataFrame, method: str = 'expanding') -> pd.DataFrame:
        """
        Get information about all splits for visualization.
        
        Args:
            X: Features DataFrame
            method: Split method ('expanding', 'sliding', 'time_series')
            
        Returns:
            DataFrame with split information
        """
        if method == 'expanding':
            splits = list(self.expanding_window_split(X))
        elif method == 'sliding':
            splits = list(self.sliding_window_split(X))
        else:
            splits = list(self.time_series_split(X))
        
        info = []
        for fold in splits:
            info.append({
                'fold': fold.fold_number,
                'train_start': fold.train_indices[0],
                'train_end': fold.train_indices[-1],
                'train_size': fold.train_size,
                'val_start': fold.val_indices[0],
                'val_end': fold.val_indices[-1],
                'val_size': fold.val_size
            })
        
        return pd.DataFrame(info)


def temporal_cross_validate(
    model_class,
    model_params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.DataFrame,
    evaluator,
    n_splits: int = 5,
    method: str = 'expanding',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform temporal cross-validation on a model.
    
    Args:
        model_class: Class of the model to instantiate
        model_params: Parameters to pass to model constructor
        X: Features DataFrame with 'jobs_list' and 'actions_list'
        y: Targets DataFrame with 'job_id' and 'action'
        evaluator: Evaluator instance
        n_splits: Number of CV folds
        method: CV method ('expanding', 'sliding', 'time_series')
        verbose: Whether to print progress
        
    Returns:
        Dictionary with fold results and aggregated metrics
    """
    validator = TemporalValidator(n_splits=n_splits)
    
    if method == 'expanding':
        splits = validator.expanding_window_split(X)
    elif method == 'sliding':
        splits = validator.sliding_window_split(X)
    else:
        splits = validator.time_series_split(X)
    
    fold_results = []
    
    for fold in splits:
        # Split data
        X_train = X.iloc[fold.train_indices].reset_index(drop=True)
        y_train = y.iloc[fold.train_indices].reset_index(drop=True)
        X_val = X.iloc[fold.val_indices].reset_index(drop=True)
        y_val = y.iloc[fold.val_indices].reset_index(drop=True)
        
        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_top10 = model.predict(X_val, top_k=10)
        y_pred_actions = model.predict_actions(X_val)
        
        # Get true values
        y_true_ids = y_val['job_id'].tolist()
        y_true_actions = [1 if action == 'apply' else 0 for action in y_val['action'].tolist()]
        
        # Calculate metrics
        mrr = evaluator.calculate_mrr(y_true_ids, y_pred_top10)
        accuracy = evaluator.calculate_action_accuracy(y_true_actions, y_pred_actions)
        final_score = evaluator.calculate_final_score(mrr, accuracy)
        
        fold_result = {
            'fold': fold.fold_number,
            'train_size': fold.train_size,
            'val_size': fold.val_size,
            'mrr': mrr,
            'accuracy': accuracy,
            'final_score': final_score
        }
        fold_results.append(fold_result)
        
        if verbose:
            print(f"Fold {fold.fold_number}: MRR={mrr:.4f}, Acc={accuracy:.4f}, Score={final_score:.4f}")
    
    # Aggregate results
    results_df = pd.DataFrame(fold_results)
    
    return {
        'fold_results': results_df,
        'mean_mrr': results_df['mrr'].mean(),
        'std_mrr': results_df['mrr'].std(),
        'mean_accuracy': results_df['accuracy'].mean(),
        'std_accuracy': results_df['accuracy'].std(),
        'mean_final_score': results_df['final_score'].mean(),
        'std_final_score': results_df['final_score'].std()
    }
