import pandas as pd
import numpy as np
import torch
from pathlib import Path
import ast
import sys
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.contrastive_model import SessionDataset, ContrastiveRecommender, ContrastiveTrainer
from torch.utils.data import DataLoader


def load_test_data(data_path: Path):
    x_test_path = data_path / 'x_test_with_features.csv'
    
    if not x_test_path.exists():
        print("Test features not found. Using raw test data...")
        x_test = pd.read_csv(data_path / 'x_test_jCBBNP2.csv')
        
        from pipeline.feature_engineer import FeatureEngineer
        
        x_test['jobs_list'] = x_test['job_ids'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        x_test['actions_list'] = x_test['actions'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        feature_engineer = FeatureEngineer()
        x_test = feature_engineer.extract_all_features(x_test)
        x_test.to_csv(x_test_path, index=False)
    else:
        x_test = pd.read_csv(x_test_path)
    
    return x_test


def get_all_candidate_jobs(data_path: Path):
    x_train = pd.read_csv(data_path / 'x_train_Meacfjr.csv')
    
    all_jobs = []
    for jobs in x_train['job_ids']:
        job_list = ast.literal_eval(jobs) if isinstance(jobs, str) else jobs
        all_jobs.extend(job_list)
    
    return sorted(list(set(all_jobs)))


def load_diversity_weights(data_path: Path):
    balancer_path = data_path / 'job_listing_feature' / 'job_balancer.csv'
    balancer = pd.read_csv(balancer_path)
    return dict(zip(balancer['job_id'], balancer['weight']))


def load_job_skills(data_path: Path):
    skills_path = data_path / 'job_listing_feature' / 'job_skills.csv'
    skills_df = pd.read_csv(skills_path)
    
    job_skills = defaultdict(set)
    for _, row in skills_df.iterrows():
        job_skills[row['job_id']].add(row['skill'])
    
    return dict(job_skills)


def compute_skill_overlap(session_jobs, candidate_job, job_skills_dict):
    session_skills = set()
    for job_id in session_jobs:
        if job_id in job_skills_dict:
            session_skills.update(job_skills_dict[job_id])
    
    if not session_skills:
        return 0.0
    
    candidate_skills = job_skills_dict.get(candidate_job, set())
    if not candidate_skills:
        return 0.0
    
    intersection = len(session_skills & candidate_skills)
    union = len(session_skills | candidate_skills)
    
    return intersection / union if union > 0 else 0.0


def hybrid_rerank(predictions, x_test, diversity_weights, job_skills_dict, 
                  alpha=0.5, beta=0.3, gamma=0.2):
    
    reranked_predictions = []
    
    for i, pred_jobs in enumerate(predictions):
        session_id = x_test.iloc[i]['session_id']
        session_jobs = ast.literal_eval(x_test.iloc[i]['job_ids']) if isinstance(x_test.iloc[i]['job_ids'], str) else x_test.iloc[i]['job_ids']
        
        job_scores = []
        
        for rank, job_id in enumerate(pred_jobs, 1):
            model_score = 1.0 / rank
            
            diversity_score = diversity_weights.get(job_id, 0.2)
            
            skill_score = compute_skill_overlap(session_jobs, job_id, job_skills_dict)
            
            final_score = alpha * model_score + beta * diversity_score + gamma * skill_score
            
            job_scores.append((job_id, final_score))
        
        job_scores.sort(key=lambda x: x[1], reverse=True)
        reranked_predictions.append([job_id for job_id, _ in job_scores[:10]])
    
    return reranked_predictions


def generate_submission(model_path: str, data_path: str, output_path: str, 
                       top_k: int = 10, device: str = 'cuda', use_hybrid: bool = True):
    data_path = Path(data_path)
    
    print("Loading test data...")
    x_test = load_test_data(data_path)
    
    print("Loading candidate jobs...")
    candidate_jobs = get_all_candidate_jobs(data_path)
    
    print(f"Building model with {len(candidate_jobs)} candidate jobs...")
    num_jobs = max(candidate_jobs) + 100
    model = ContrastiveRecommender(num_jobs=num_jobs)
    
    trainer = ContrastiveTrainer(model, device=device)
    
    print(f"Loading model from {model_path}...")
    trainer.load_model(model_path)
    
    print("Creating test dataset...")
    test_dataset = SessionDataset(x_test, all_jobs=candidate_jobs)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print("Generating base predictions...")
    predictions = trainer.predict_batch(test_loader, candidate_jobs=candidate_jobs, top_k=20)
    
    if use_hybrid:
        print("Loading diversity weights...")
        diversity_weights = load_diversity_weights(data_path)
        
        print("Loading job skills...")
        job_skills_dict = load_job_skills(data_path)
        
        print("Applying hybrid re-ranking...")
        predictions = hybrid_rerank(predictions, x_test, diversity_weights, job_skills_dict)
    else:
        predictions = [jobs[:10] for jobs in predictions]
    
    print("Formatting submission...")
    submission_data = []
    for i, pred_jobs in enumerate(predictions):
        session_id = x_test.iloc[i]['session_id']
        
        for rank, job_id in enumerate(pred_jobs, start=1):
            submission_data.append({
                'session_id': session_id,
                'job_id': job_id,
                'rank': rank
            })
    
    df_submission = pd.DataFrame(submission_data)
    
    df_submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Total predictions: {len(df_submission)} rows for {len(predictions)} sessions")
    
    if use_hybrid:
        print("\nDiversity analysis:")
        unique_jobs = df_submission['job_id'].nunique()
        print(f"Unique jobs recommended: {unique_jobs}")
        
        job_freq = df_submission['job_id'].value_counts()
        print(f"Most recommended job appears in {job_freq.iloc[0]} sessions")
        print(f"Top 5 jobs: {job_freq.head(5).index.tolist()}")


def main():
    base_path = Path(__file__).parent.parent.parent
    
    model_path = base_path / 'experiments' / 'contrastive_model.pt'
    data_path = base_path / 'Data'
    output_path = base_path / 'submissions' / 'contrastive_submission.csv'
    
    generate_submission(
        model_path=str(model_path),
        data_path=str(data_path),
        output_path=str(output_path),
        top_k=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


if __name__ == '__main__':
    main()
