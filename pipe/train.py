#!/usr/bin/env python3
"""
Main Training Script for Job Recommendation.

Follows the Collaborative Filtering approach from the competition benchmark.
Lightweight, no neural networks required.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipe.config import Config, DEFAULT_CONFIG
from pipe.data.loader import DataLoader
from pipe.models.job_ranker import JobRanker
from pipe.models.action_predictor_svm import ActionPredictorSVM
from pipe.evaluation.metrics import evaluate_model

import pickle


def train(config: Config = None):
    """
    Train the complete recommendation pipeline.
    
    Args:
        config: Configuration object (uses default if None)
        
    Returns:
        Trained job_ranker and action_predictor
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    print("=" * 60)
    print("Job Recommendation Training Pipeline")
    print("=" * 60)
    
    # 1. Load data
    data_loader = DataLoader(config)
    data_loader.load()
    
    # 2. Split train/val
    train_df, val_df = data_loader.get_train_val_split()
    print(f"\nTrain/Val split: {len(train_df)}/{len(val_df)}")
    
    # 3. Train Job Ranker
    print("\n" + "-" * 60)
    job_ranker = JobRanker(
        transition_weight=config.transition_weight,
        cooccurrence_weight=config.cooccurrence_weight,
        cf_weight=config.cf_weight,
        popularity_weight=config.popularity_weight,
        k_similar_sessions=config.k_similar_sessions
    )
    job_ranker.fit(train_df, data_loader.job_to_idx)
    
    # 4. Train Action Predictor (SVM)
    print("\n" + "-" * 60)
    action_predictor = ActionPredictorSVM(C=1.0, gamma='scale')
    action_predictor.fit(train_df)
    
    # 5. Evaluate on validation set
    print("\n" + "-" * 60)
    print("Evaluating on validation set...")
    
    metrics = evaluate_model(job_ranker, action_predictor, val_df, top_k=config.top_k)
    
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"  MRR:             {metrics['mrr']:.4f}")
    print(f"  Action Accuracy: {metrics['action_accuracy']:.4f}")
    print(f"  Final Score:     {metrics['final_score']:.4f}")
    print("=" * 60)
    
    # 6. Save models
    print("\nSaving models...")
    config.output_dir.mkdir(exist_ok=True)
    
    with open(config.output_dir / 'job_ranker.pkl', 'wb') as f:
        pickle.dump(job_ranker, f)
    
    with open(config.output_dir / 'action_predictor.pkl', 'wb') as f:
        pickle.dump(action_predictor, f)
    
    with open(config.output_dir / 'data_loader.pkl', 'wb') as f:
        pickle.dump(data_loader, f)
    
    print(f"  Saved to {config.output_dir}/")
    
    return job_ranker, action_predictor, data_loader


def tune_hyperparameters(config: Config = None):
    """
    Simple grid search over key hyperparameters.
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    print("=" * 60)
    print("Hyperparameter Tuning")
    print("=" * 60)
    
    # Load data once
    data_loader = DataLoader(config)
    data_loader.load()
    train_df, val_df = data_loader.get_train_val_split()
    
    best_score = 0
    best_params = {}
    
    # Grid search
    for trans_w in [3.0, 5.0, 7.0]:
        for cooc_w in [1.0, 2.0, 3.0]:
            for cf_w in [0.5, 1.0, 2.0]:
                # Train with these params
                job_ranker = JobRanker(
                    transition_weight=trans_w,
                    cooccurrence_weight=cooc_w,
                    cf_weight=cf_w,
                    popularity_weight=0.1
                )
                job_ranker.fit(train_df, data_loader.job_to_idx)
                
                action_predictor = ActionPredictorSVM()
                action_predictor.fit(train_df)
                
                # Evaluate
                metrics = evaluate_model(job_ranker, action_predictor, val_df)
                
                print(f"trans={trans_w}, cooc={cooc_w}, cf={cf_w} -> "
                      f"MRR={metrics['mrr']:.4f}, Final={metrics['final_score']:.4f}")
                
                if metrics['final_score'] > best_score:
                    best_score = metrics['final_score']
                    best_params = {'trans': trans_w, 'cooc': cooc_w, 'cf': cf_w}
    
    print("\n" + "=" * 60)
    print(f"Best params: {best_params}")
    print(f"Best final score: {best_score:.4f}")
    print("=" * 60)
    
    return best_params


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train job recommendation model')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--trans-weight', type=float, default=5.0, help='Transition weight')
    parser.add_argument('--cooc-weight', type=float, default=2.0, help='Co-occurrence weight')
    parser.add_argument('--cf-weight', type=float, default=1.0, help='Collaborative filter weight')
    
    args = parser.parse_args()
    
    config = Config(
        transition_weight=args.trans_weight,
        cooccurrence_weight=args.cooc_weight,
        cf_weight=args.cf_weight
    )
    
    if args.tune:
        tune_hyperparameters(config)
    else:
        train(config)
