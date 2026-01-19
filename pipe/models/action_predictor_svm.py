"""
Action Prediction Model using SVM.

Predicts whether the candidate's next action will be 'apply' or 'view'
using a Support Vector Machine with RBF kernel based on session behavior features.
"""

import numpy as np
import pandas as pd
from typing import List
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class ActionPredictorSVM:
    """
    Predicts the next action (apply/view) using SVM based on:
    1. Apply ratio in session
    2. View ratio in session
    3. Sequence length
    4. Last action type
    5. Action change frequency
    """
    
    def __init__(self, C: float = 1.0, gamma: str = 'scale'):
        """
        Args:
            C: Regularization parameter for SVM
            gamma: Kernel coefficient ('scale' or 'auto')
        """
        self.model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _extract_features(
        self, 
        session_jobs: List[int],
        session_actions: List[str]
    ) -> np.ndarray:
        """
        Extract 5 behavioral features from session.
        
        Args:
            session_jobs: Jobs viewed in the session
            session_actions: Actions taken in the session
            
        Returns:
            Feature vector [apply_ratio, view_ratio, seq_length, 
                          last_action_is_apply, action_change_ratio]
        """
        if not session_actions:
            return np.zeros(5)
        
        # 1. Apply ratio
        apply_count = session_actions.count('apply')
        apply_ratio = apply_count / len(session_actions)
        
        # 2. View ratio
        view_ratio = 1.0 - apply_ratio
        
        # 3. Sequence length
        seq_length = len(session_jobs)
        
        # 4. Last action is apply
        last_action_is_apply = 1.0 if session_actions[-1] == 'apply' else 0.0
        
        # 5. Action change ratio
        if len(session_actions) > 1:
            changes = sum(
                1 for i in range(len(session_actions) - 1) 
                if session_actions[i] != session_actions[i + 1]
            )
            action_change_ratio = changes / (len(session_actions) - 1)
        else:
            action_change_ratio = 0.0
        
        return np.array([
            apply_ratio,
            view_ratio,
            seq_length,
            last_action_is_apply,
            action_change_ratio
        ])
    
    def fit(self, train_df: pd.DataFrame) -> 'ActionPredictorSVM':
        """
        Train the SVM model from training data.
        
        Args:
            train_df: Training DataFrame with 'job_ids', 'actions', and 'action' columns
        """
        print("Building SVM action predictor...")
        
        X = []
        y = []
        
        for _, row in train_df.iterrows():
            jobs = row['job_ids']
            actions = row.get('actions', [])
            target_action = row.get('action', 'view')
            
            # Skip sessions without history
            if not actions:
                continue
            
            features = self._extract_features(jobs, actions)
            X.append(features)
            y.append(1 if target_action == 'apply' else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"  Training samples: {len(X)}")
        print(f"  Class distribution: {np.sum(y)} applies, {len(y) - np.sum(y)} views")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train SVM
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        print(f"  SVM trained successfully")
        
        return self
    
    def predict(
        self, 
        session_jobs: List[int],
        session_actions: List[str],
        predicted_jobs: List[int] = None  # Kept for compatibility
    ) -> int:
        """
        Predict next action (0=view, 1=apply).
        
        Args:
            session_jobs: Jobs in the current session
            session_actions: Actions in the current session
            predicted_jobs: Top predicted jobs (not used, kept for compatibility)
            
        Returns:
            0 for 'view', 1 for 'apply'
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if not session_actions:
            return 0  # Default: view
        
        features = self._extract_features(session_jobs, session_actions)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        return int(self.model.predict(features_scaled)[0])
    
    def predict_proba(
        self, 
        session_jobs: List[int],
        session_actions: List[str],
        predicted_jobs: List[int] = None
    ) -> float:
        """
        Predict probability of 'apply' action.
        
        Note: SVM with probability=False doesn't support predict_proba.
        This returns the hard decision instead.
        
        Args:
            session_jobs: Jobs in the current session
            session_actions: Actions in the current session
            predicted_jobs: Top predicted jobs (not used)
            
        Returns:
            1.0 if predict 'apply', 0.0 if predict 'view'
        """
        pred = self.predict(session_jobs, session_actions, predicted_jobs)
        return float(pred)
