"""
Evaluation Metrics.

Following the competition:
- MRR (Mean Reciprocal Rank): 70% weight
- Action Accuracy: 30% weight
"""

import numpy as np
from typing import List, Tuple, Dict
import pandas as pd


def compute_mrr(predictions: List[List[int]], targets: List[int], k: int = 10) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    For each prediction:
    - If target is at rank r (1-indexed) in top-k: reciprocal = 1/r
    - If target is not in top-k: reciprocal = 0
    
    Args:
        predictions: List of predicted job ID lists (each of length k)
        targets: List of true target job IDs
        k: Maximum rank to consider
        
    Returns:
        Mean Reciprocal Rank
    """
    reciprocal_ranks = []
    
    for pred_list, target in zip(predictions, targets):
        pred_list = pred_list[:k]  # Only consider top-k
        
        if target in pred_list:
            rank = pred_list.index(target) + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)


def compute_action_accuracy(predictions: List[int], targets: List[int]) -> float:
    """
    Compute accuracy of action predictions (apply/view).
    
    Args:
        predictions: List of predicted actions (0=view, 1=apply)
        targets: List of true actions (0=view, 1=apply)
        
    Returns:
        Accuracy
    """
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(predictions) if predictions else 0.0


def compute_final_score(mrr: float, action_accuracy: float) -> float:
    """
    Compute final competition score.
    
    Final Score = 0.7 * MRR + 0.3 * Action_Accuracy
    
    Args:
        mrr: Mean Reciprocal Rank
        action_accuracy: Action prediction accuracy
        
    Returns:
        Final score
    """
    return 0.7 * mrr + 0.3 * action_accuracy


def evaluate_model(
    job_ranker,
    action_predictor,
    val_df: pd.DataFrame,
    top_k: int = 10
) -> Dict[str, float]:
    """
    Evaluate the complete model on validation data.
    
    Args:
        job_ranker: JobRanker model
        action_predictor: ActionPredictor model
        val_df: Validation DataFrame
        top_k: Number of jobs to predict
        
    Returns:
        Dictionary with 'mrr', 'action_accuracy', 'final_score'
    """
    job_predictions = []
    action_predictions = []
    job_targets = []
    action_targets = []
    
    for _, row in val_df.iterrows():
        session_jobs = row['job_ids']
        session_actions = row.get('actions', [])
        target_job = row['job_id']
        target_action = row.get('action', 'view')
        
        # Convert target action to int
        target_action_int = 1 if target_action == 'apply' else 0
        
        # Predict top-k jobs
        pred_jobs = job_ranker.predict_top_k(session_jobs, k=top_k)
        job_predictions.append(pred_jobs)
        job_targets.append(target_job)
        
        # Predict action
        pred_action = action_predictor.predict(session_jobs, session_actions, pred_jobs)
        action_predictions.append(pred_action)
        action_targets.append(target_action_int)
    
    # Compute metrics
    mrr = compute_mrr(job_predictions, job_targets, k=top_k)
    action_acc = compute_action_accuracy(action_predictions, action_targets)
    final_score = compute_final_score(mrr, action_acc)
    
    return {
        'mrr': mrr,
        'action_accuracy': action_acc,
        'final_score': final_score
    }
