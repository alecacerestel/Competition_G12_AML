#!/usr/bin/env python3
"""
Generate Submission for Job Recommendation Challenge.

Produces predictions in the required format:
- Top 10 job IDs per session
- Action prediction (applies_for: 0 or 1)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipe.config import Config, DEFAULT_CONFIG
from pipe.data.loader import DataLoader
from pipe.models.job_ranker import JobRanker
from pipe.models.action_predictor import ActionPredictor

import pickle
import pandas as pd
from datetime import datetime


def generate_submission(config: Config = None, output_file: str = None):
    """
    Generate submission file for the competition.
    
    Args:
        config: Configuration object
        output_file: Output filename (auto-generated if None)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    print("=" * 60)
    print("Generating Submission")
    print("=" * 60)
    
    # Load saved models or train new ones
    try:
        print("Loading saved models...")
        with open(config.output_dir / 'job_ranker.pkl', 'rb') as f:
            job_ranker = pickle.load(f)
        with open(config.output_dir / 'action_predictor.pkl', 'rb') as f:
            action_predictor = pickle.load(f)
        with open(config.output_dir / 'data_loader.pkl', 'rb') as f:
            data_loader = pickle.load(f)
        print("  Models loaded successfully!")
    except FileNotFoundError:
        print("  No saved models found. Training new models...")
        from pipe.train import train
        job_ranker, action_predictor, data_loader = train(config)
    
    # Load test data
    print("\nLoading test data...")
    test_df = data_loader.test_df
    print(f"  Test samples: {len(test_df)}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []
    
    for _, row in test_df.iterrows():
        session_id = row['session_id']
        session_jobs = row['job_ids']
        session_actions = row.get('actions', [])
        
        # Predict top 10 jobs
        pred_jobs = job_ranker.predict_top_k(session_jobs, k=config.top_k)
        
        # Pad with popular jobs if needed
        if len(pred_jobs) < config.top_k:
            # Get most popular jobs as fallback
            popular = sorted(
                job_ranker.job_popularity.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for job, _ in popular:
                if job not in pred_jobs and job not in session_jobs:
                    pred_jobs.append(job)
                    if len(pred_jobs) >= config.top_k:
                        break
        
        # Predict action
        pred_action = action_predictor.predict(session_jobs, session_actions, pred_jobs)
        
        predictions.append({
            'session_id': session_id,
            'job_ids': pred_jobs[:config.top_k],
            'applies_for': pred_action
        })
    
    # Format for submission
    print("\nFormatting submission...")
    submission_rows = []
    
    for pred in predictions:
        # Format job_ids as string list
        job_ids_str = '[' + ', '.join(map(str, pred['job_ids'])) + ']'
        
        submission_rows.append({
            'session_id': pred['session_id'],
            'job_ids': job_ids_str,
            'applies_for': pred['applies_for']
        })
    
    submission_df = pd.DataFrame(submission_rows)
    
    # Generate output filename
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"submissions/submission_cf_{timestamp}.csv"
    
    # Ensure directory exists
    Path(output_file).parent.mkdir(exist_ok=True)
    
    # Save
    submission_df.to_csv(output_file, index=False)
    print(f"\n  Saved to: {output_file}")
    print(f"  Rows: {len(submission_df)}")
    
    # Show sample
    print("\nSample predictions:")
    print(submission_df.head(3).to_string())
    
    return submission_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate submission')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    
    args = parser.parse_args()
    
    generate_submission(output_file=args.output)
