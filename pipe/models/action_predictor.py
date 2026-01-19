"""
Action Prediction Model.

Predicts whether the candidate's next action will be 'apply' or 'view'
based on patterns from similar sessions.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd


class ActionPredictor:
    """
    Predicts the next action (apply/view) based on:
    1. Action patterns in similar sessions
    2. Apply rate for specific jobs
    3. Session behavior patterns
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Probability threshold for predicting 'apply' (1) vs 'view' (0)
        """
        self.threshold = threshold
        
        # Statistics from training
        self.job_apply_rate = {}      # job_id -> P(apply | job)
        self.session_patterns = {}     # Last action pattern -> P(apply)
        self.global_apply_rate = 0.0  # Overall apply rate
        
    def fit(self, train_df: pd.DataFrame) -> 'ActionPredictor':
        """
        Build action prediction model from training data.
        
        Args:
            train_df: Training DataFrame with 'actions' and 'action' columns
        """
        print("Building action predictor...")
        
        # 1. Compute per-job apply rate
        job_apply_counts = defaultdict(int)
        job_total_counts = defaultdict(int)
        
        for _, row in train_df.iterrows():
            jobs = row['job_ids']
            actions = row.get('actions', [])
            
            for i, job in enumerate(jobs):
                if i < len(actions):
                    job_total_counts[job] += 1
                    if actions[i] == 'apply':
                        job_apply_counts[job] += 1
            
            # Target job and action
            if 'job_id' in row and 'action' in row:
                target_job = row['job_id']
                target_action = row['action']
                job_total_counts[target_job] += 1
                if target_action == 'apply':
                    job_apply_counts[target_job] += 1
        
        for job in job_total_counts:
            self.job_apply_rate[job] = job_apply_counts[job] / job_total_counts[job]
        
        print(f"  Jobs with apply rate data: {len(self.job_apply_rate)}")
        
        # 2. Compute global apply rate
        total_applies = sum(job_apply_counts.values())
        total_actions = sum(job_total_counts.values())
        self.global_apply_rate = total_applies / total_actions if total_actions > 0 else 0.5
        
        print(f"  Global apply rate: {self.global_apply_rate:.4f}")
        
        # 3. Session-based patterns: does last action predict next action?
        last_action_counts = defaultdict(lambda: {'apply': 0, 'view': 0, 'total': 0})
        
        for _, row in train_df.iterrows():
            actions = row.get('actions', [])
            target_action = row.get('action', 'view')
            
            if actions:
                last_action = actions[-1]
                last_action_counts[last_action]['total'] += 1
                last_action_counts[last_action][target_action] += 1
        
        self.session_patterns = {}
        for last_action, counts in last_action_counts.items():
            if counts['total'] > 0:
                self.session_patterns[last_action] = counts['apply'] / counts['total']
        
        print(f"  Session patterns: {self.session_patterns}")
        
        return self
    
    def predict_proba(
        self, 
        session_jobs: List[int], 
        session_actions: List[str],
        predicted_jobs: List[int] = None
    ) -> float:
        """
        Predict probability of 'apply' for next action.
        
        Args:
            session_jobs: Jobs in the current session
            session_actions: Actions in the current session
            predicted_jobs: Top predicted jobs (can use their apply rates)
            
        Returns:
            Probability of 'apply'
        """
        scores = []
        weights = []
        
        # 1. Last action pattern
        if session_actions:
            last_action = session_actions[-1]
            if last_action in self.session_patterns:
                scores.append(self.session_patterns[last_action])
                weights.append(2.0)  # Higher weight for session pattern
        
        # 2. Apply rate of session jobs
        if session_jobs:
            session_apply_rates = [
                self.job_apply_rate.get(job, self.global_apply_rate)
                for job in session_jobs
            ]
            scores.append(np.mean(session_apply_rates))
            weights.append(1.0)
        
        # 3. Apply rate of predicted jobs
        if predicted_jobs:
            pred_apply_rates = [
                self.job_apply_rate.get(job, self.global_apply_rate)
                for job in predicted_jobs[:5]  # Top 5 predictions
            ]
            scores.append(np.mean(pred_apply_rates))
            weights.append(1.5)
        
        # Weighted average
        if scores:
            return np.average(scores, weights=weights)
        else:
            return self.global_apply_rate
    
    def predict(
        self, 
        session_jobs: List[int], 
        session_actions: List[str],
        predicted_jobs: List[int] = None
    ) -> int:
        """
        Predict next action (0=view, 1=apply).
        
        Args:
            session_jobs: Jobs in the current session
            session_actions: Actions in the current session
            predicted_jobs: Top predicted jobs
            
        Returns:
            0 for 'view', 1 for 'apply'
        """
        proba = self.predict_proba(session_jobs, session_actions, predicted_jobs)
        return 1 if proba >= self.threshold else 0
