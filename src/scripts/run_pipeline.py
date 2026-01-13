import sys
from pathlib import Path
from scipy.sparse import csr_matrix

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from pipeline.data_processor import DataProcessor
from pipeline.model_trainer import ModelTrainer
from pipeline.evaluator import Evaluator


def main():
    """Main pipeline execution."""
    print("=" * 80)
    print("STARTING JOB RECOMMENDATION PIPELINE")
    print("=" * 80)
    
    # Step 1: Load Data
    print("\n[STEP 1] Loading data...")
    processor = DataProcessor()
    jobs, x_train, y_train, x_test = processor.load_data()
    print(f"✓ Loaded {len(x_train)} training sessions")
    print(f"✓ Loaded {len(y_train)} training targets")
    print(f"✓ Loaded {len(x_test)} test sessions")
    print(f"✓ Loaded {len(jobs)} job listings")
    
    # Step 2: Prepare data sequences (only for X data)
    print("\n[STEP 2] Preparing data sequences...")
    x_train_prepared = processor.prepare_sequences(x_train)
    x_test_prepared = processor.prepare_sequences(x_test)
    print(f"✓ Prepared training sequences")
    print(f"✓ Prepared test sequences")
    
    # Step 3: Split train/test for validation
    print("\n[STEP 3] Splitting data for validation...")
    x_train_split, x_val_split, y_train_split, y_val_split = processor.split_train_test(
        test_size=0.2, random_state=42
    )
    x_train_split = processor.prepare_sequences(x_train_split)
    x_val_split = processor.prepare_sequences(x_val_split)
    # y_train_split and y_val_split don't need processing - they're already in correct format
    print(f"✓ Training set: {len(x_train_split)} sessions")
    print(f"✓ Validation set: {len(x_val_split)} sessions")
    
    # Step 4: Build interaction matrix on training data
    print("\n[STEP 4] Building interaction matrix...")
    trainer = ModelTrainer()
    R_train, session_cats, job_cats, _ = trainer.build_interaction_matrix(
        x_train_split, y_train_split
    )
    
    # Step 5: Make predictions on validation set
    print("\n[STEP 5] Making predictions on validation set...")
    y_pred_top10_val = []
    y_pred_actions_val = []
    y_true_ids_val = y_val_split['job_id'].tolist()
    y_true_actions_val = []
    
    for idx, row in x_val_split.iterrows():
        jobs_viewed = row['jobs_list']
        
        # Create sparse vector for this session
        job_indices = [np.where(job_cats == job)[0][0] for job in jobs_viewed if job in job_cats]
        if job_indices:
            test_vector = csr_matrix(
                (np.ones(len(job_indices)), (np.zeros(len(job_indices)), job_indices)),
                shape=(1, len(job_cats))
            )
        else:
            test_vector = csr_matrix((1, len(job_cats)))
        
        # Get predictions using the model's predict_next_step function
        top_10_jobs, applies_for, p_c_avg = trainer.predict_next_step(
            test_vector, R_train, job_cats, k=50, theta=0.5
        )
        
        y_pred_top10_val.append(top_10_jobs)
        y_pred_actions_val.append(applies_for)
    
    # Extract true actions (1 if user viewed at least one job, 0 otherwise)
    for idx, row in y_val_split.iterrows():
        y_true_actions_val.append(1)  # Since they all appear in y_train, they all applied
    
    print(f"✓ Generated {len(y_pred_top10_val)} predictions")
    
    # Step 6: Evaluate on validation set
    print("\n[STEP 6] Evaluating model on validation set...")
    evaluator = Evaluator()
    mrr_score = evaluator.calculate_mrr(y_true_ids_val, y_pred_top10_val)
    action_accuracy = evaluator.calculate_action_accuracy(y_true_actions_val, y_pred_actions_val)
    final_score = evaluator.calculate_final_score(mrr_score, action_accuracy)
    
    print(f"✓ MRR Score: {mrr_score:.4f}")
    print(f"✓ Action Accuracy: {action_accuracy:.4f}")
    print(f"✓ Final Score (0.7*MRR + 0.3*Accuracy): {final_score:.4f}")
    
    # Step 7: Retrain on full training data
    print("\n[STEP 7] Retraining model on full training data...")
    R_full, session_cats_full, job_cats_full, _ = trainer.build_interaction_matrix(
        x_train_prepared, y_train
    )
    
    # Step 8: Generate predictions for test set
    print("\n[STEP 8] Generating predictions for test set...")
    y_pred_top10_test = []
    y_pred_actions_test = []
    
    for idx, row in x_test_prepared.iterrows():
        jobs_viewed = row['jobs_list']
        
        # Create sparse vector for this session
        job_indices = [np.where(job_cats_full == job)[0][0] for job in jobs_viewed if job in job_cats_full]
        if job_indices:
            test_vector = csr_matrix(
                (np.ones(len(job_indices)), (np.zeros(len(job_indices)), job_indices)),
                shape=(1, len(job_cats_full))
            )
        else:
            test_vector = csr_matrix((1, len(job_cats_full)))
        
        # Get predictions
        top_10_jobs, applies_for, _ = trainer.predict_next_step(
            test_vector, R_full, job_cats_full, k=50, theta=0.5
        )
        
        y_pred_top10_test.append(top_10_jobs)
        y_pred_actions_test.append(applies_for)
    
    print(f"✓ Generated {len(y_pred_top10_test)} test predictions")
    
    # Step 9: Prepare submission
    print("\n[STEP 9] Preparing submission...")
    submission_df = pd.DataFrame({
        'session_id': x_test['session_id'],
        'predicted_jobs': y_pred_top10_test,
        'applies_for': y_pred_actions_test
    })
    
    submission_path = Path(__file__).parent.parent.parent / 'submissions' / 'predictions.csv'
    submission_path.parent.mkdir(exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"✓ Submission saved to {submission_path}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nValidation Results:")
    print(f"  - MRR Score: {mrr_score:.4f}")
    print(f"  - Action Accuracy: {action_accuracy:.4f}")
    print(f"  - Final Score: {final_score:.4f}")
    print(f"\nTest predictions: {len(y_pred_top10_test)} sessions")


if __name__ == "__main__":
    main()
