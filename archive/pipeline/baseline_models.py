"""
Baseline Models for Job Recommendation System

This module implements simple baseline models for the job recommendation challenge:
1. Most Popular Jobs Baseline - Recommends the most frequently applied/viewed jobs
2. Last Viewed Baseline - Recommends the last viewed jobs in the session
3. Random Baseline - Random job recommendations

These baselines serve as benchmarks to compare against more sophisticated models.
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict


class MostPopularBaseline:
    """
    Baseline model that recommends the most popular jobs globally.
    
    Popularity can be measured by:
    - Number of applications
    - Number of views
    - Combined score
    """
    
    def __init__(self, metric: str = 'apply'):
        """
        Initialize the Most Popular baseline.
        
        Args:
            metric: How to measure popularity ('apply', 'view', 'combined')
        """
        self.metric = metric
        self.popular_jobs = []
        self.job_popularity = {}
        
    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame = None) -> 'MostPopularBaseline':
        """
        Fit the model by computing job popularity from training data.
        
        Args:
            x_train: Training data with 'jobs_list' and 'actions_list' columns
            y_train: Target data with 'job_id' and 'action' columns
            
        Returns:
            self
        """
        job_counts = Counter()
        
        if self.metric == 'apply':
            # Count jobs that were applied to
            for idx, row in x_train.iterrows():
                jobs = row['jobs_list']
                actions = row['actions_list']
                for job, action in zip(jobs, actions):
                    if action == 'apply':
                        job_counts[job] += 1
            
            # Also count from y_train if available (target jobs that were applied)
            if y_train is not None:
                apply_jobs = y_train[y_train['action'] == 'apply']['job_id']
                for job in apply_jobs:
                    job_counts[job] += 1
                    
        elif self.metric == 'view':
            # Count all job views
            for idx, row in x_train.iterrows():
                for job in row['jobs_list']:
                    job_counts[job] += 1
                    
        else:  # combined
            # Weight applies more than views
            for idx, row in x_train.iterrows():
                jobs = row['jobs_list']
                actions = row['actions_list']
                for job, action in zip(jobs, actions):
                    weight = 3 if action == 'apply' else 1
                    job_counts[job] += weight
        
        # Store results sorted by popularity
        self.job_popularity = dict(job_counts)
        self.popular_jobs = [job for job, _ in job_counts.most_common()]
        
        return self
    
    def predict(self, x_test: pd.DataFrame, top_k: int = 10) -> List[List[int]]:
        """
        Predict top-k jobs for each session.
        
        Args:
            x_test: Test data (not used, same recommendation for all)
            top_k: Number of jobs to recommend
            
        Returns:
            List of lists with top-k job recommendations for each session
        """
        top_jobs = self.popular_jobs[:top_k]
        
        # Pad if we don't have enough popular jobs
        while len(top_jobs) < top_k:
            top_jobs.append(top_jobs[-1] if top_jobs else 0)
        
        return [top_jobs.copy() for _ in range(len(x_test))]
    
    def predict_actions(self, x_test: pd.DataFrame, threshold: float = 0.5) -> List[int]:
        """
        Predict whether each session will result in an apply action.
        
        For baseline, we predict based on a fixed threshold.
        
        Args:
            x_test: Test data
            threshold: Probability threshold for action prediction
            
        Returns:
            List of predicted actions (1 = apply, 0 = view)
        """
        # Simple heuristic: if session has any applies, predict apply
        predictions = []
        for idx, row in x_test.iterrows():
            actions = row['actions_list']
            has_apply = any(a == 'apply' for a in actions)
            predictions.append(1 if has_apply else 0)
        
        return predictions


class LastViewedBaseline:
    """
    Baseline model that recommends based on the last viewed jobs in the session.
    
    This is a personalized baseline that uses session context.
    """
    
    def __init__(self, fallback_popular: bool = True):
        """
        Initialize the Last Viewed baseline.
        
        Args:
            fallback_popular: If True, use popular jobs to fill recommendations
        """
        self.fallback_popular = fallback_popular
        self.popular_jobs = []
        
    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame = None) -> 'LastViewedBaseline':
        """
        Fit the model by computing popular jobs for fallback.
        
        Args:
            x_train: Training data with 'jobs_list' column
            y_train: Target data (optional)
            
        Returns:
            self
        """
        if self.fallback_popular:
            # Count job frequencies for fallback
            job_counts = Counter()
            for idx, row in x_train.iterrows():
                for job in row['jobs_list']:
                    job_counts[job] += 1
            self.popular_jobs = [job for job, _ in job_counts.most_common()]
        
        return self
    
    def predict(self, x_test: pd.DataFrame, top_k: int = 10) -> List[List[int]]:
        """
        Predict top-k jobs for each session based on last viewed jobs.
        
        Args:
            x_test: Test data with 'jobs_list' column
            top_k: Number of jobs to recommend
            
        Returns:
            List of lists with top-k job recommendations for each session
        """
        predictions = []
        
        for idx, row in x_test.iterrows():
            jobs_viewed = row['jobs_list']
            
            # Take last viewed jobs in reverse order (most recent first)
            recommendations = list(reversed(jobs_viewed[-top_k:]))
            
            # Fill with popular jobs if needed
            if len(recommendations) < top_k:
                for job in self.popular_jobs:
                    if job not in recommendations:
                        recommendations.append(job)
                    if len(recommendations) >= top_k:
                        break
            
            # Pad if still not enough
            while len(recommendations) < top_k:
                recommendations.append(recommendations[-1] if recommendations else 0)
            
            predictions.append(recommendations[:top_k])
        
        return predictions
    
    def predict_actions(self, x_test: pd.DataFrame, threshold: float = 0.5) -> List[int]:
        """
        Predict whether each session will result in an apply action.
        
        Args:
            x_test: Test data
            threshold: Not used for this baseline
            
        Returns:
            List of predicted actions (1 = apply, 0 = view)
        """
        predictions = []
        for idx, row in x_test.iterrows():
            actions = row['actions_list']
            # Predict apply if the last action was apply
            last_action = actions[-1] if actions else 'view'
            predictions.append(1 if last_action == 'apply' else 0)
        
        return predictions


class RandomBaseline:
    """
    Random baseline that recommends random jobs.
    
    This is the simplest baseline for comparison.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the Random baseline.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.all_jobs = []
        
    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame = None) -> 'RandomBaseline':
        """
        Fit the model by collecting all unique jobs.
        
        Args:
            x_train: Training data with 'jobs_list' column
            y_train: Target data (optional)
            
        Returns:
            self
        """
        all_jobs = set()
        for idx, row in x_train.iterrows():
            all_jobs.update(row['jobs_list'])
        
        if y_train is not None:
            all_jobs.update(y_train['job_id'].unique())
        
        self.all_jobs = list(all_jobs)
        return self
    
    def predict(self, x_test: pd.DataFrame, top_k: int = 10) -> List[List[int]]:
        """
        Predict top-k random jobs for each session.
        
        Args:
            x_test: Test data
            top_k: Number of jobs to recommend
            
        Returns:
            List of lists with top-k random job recommendations
        """
        np.random.seed(self.seed)
        predictions = []
        
        for _ in range(len(x_test)):
            if len(self.all_jobs) >= top_k:
                recommendations = list(np.random.choice(self.all_jobs, top_k, replace=False))
            else:
                recommendations = self.all_jobs.copy()
                while len(recommendations) < top_k:
                    recommendations.append(np.random.choice(self.all_jobs))
            
            predictions.append(recommendations)
        
        return predictions
    
    def predict_actions(self, x_test: pd.DataFrame, threshold: float = 0.5) -> List[int]:
        """
        Predict random actions.
        
        Args:
            x_test: Test data
            threshold: Probability of predicting apply
            
        Returns:
            List of predicted actions (1 = apply, 0 = view)
        """
        np.random.seed(self.seed)
        return [1 if np.random.random() < threshold else 0 for _ in range(len(x_test))]


def evaluate_baseline(
    model,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_val: pd.DataFrame,
    evaluator
) -> Dict[str, float]:
    """
    Evaluate a baseline model on validation data.
    
    Args:
        model: Baseline model with fit() and predict() methods
        x_train: Training features
        y_train: Training targets
        x_val: Validation features
        y_val: Validation targets
        evaluator: Evaluator instance with metric methods
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Fit model
    model.fit(x_train, y_train)
    
    # Predict
    y_pred_top10 = model.predict(x_val, top_k=10)
    y_pred_actions = model.predict_actions(x_val)
    
    # Get true values
    y_true_ids = y_val['job_id'].tolist()
    y_true_actions = [1 if action == 'apply' else 0 for action in y_val['action'].tolist()]
    
    # Calculate metrics
    mrr = evaluator.calculate_mrr(y_true_ids, y_pred_top10)
    accuracy = evaluator.calculate_action_accuracy(y_true_actions, y_pred_actions)
    final_score = evaluator.calculate_final_score(mrr, accuracy)
    
    return {
        'mrr': mrr,
        'accuracy': accuracy,
        'final_score': final_score
    }
