import pandas as pd
import numpy as np
import torch
from pathlib import Path
import ast
import sys
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.contrastive_model import SessionDataset, ContrastiveRecommender, ContrastiveTrainer
from torch.utils.data import DataLoader


def load_test_data(data_path: Path):
    x_test_path = data_path / 'x_test_with_features.csv'
    
    if not x_test_path.exists():
        print("Test features not found. Using raw test data...")
        x_test = pd.read_csv(data_path / 'x_test_jCBBNP2.csv')
        
        from pipeline.build_session_features import build_features
        
        x_test['jobs_list'] = x_test['job_ids'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        x_test['actions_list'] = x_test['actions'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        x_test = build_features(x_test)
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
    all_scores = []
    
    for i, pred_jobs in enumerate(predictions):
        session_id = x_test.iloc[i]['session_id']
        session_jobs = ast.literal_eval(x_test.iloc[i]['job_ids']) if isinstance(x_test.iloc[i]['job_ids'], str) else x_test.iloc[i]['job_ids']
        
        job_scores = []
        
        for rank, job_id in enumerate(pred_jobs, 1):
            model_score = 1.0 / rank
            
            diversity_score = diversity_weights.get(job_id, 0.2)
            
            skill_score = compute_skill_overlap(session_jobs, job_id, job_skills_dict)
            
            final_score = alpha * model_score + beta * diversity_score + gamma * skill_score
            
            job_scores.append((job_id, final_score, model_score))
        
        job_scores.sort(key=lambda x: x[1], reverse=True)
        top_jobs = [(job_id, score, model_score) for job_id, score, model_score in job_scores[:10]]
        reranked_predictions.append([job_id for job_id, _, _ in top_jobs])
        all_scores.append(top_jobs)
    
    return reranked_predictions, all_scores


def train_action_classifier(data_path: Path):
    print("Training action classifier...")
    
    x_train = pd.read_csv(data_path / 'x_train_with_features.csv')
    y_train = pd.read_csv(data_path / 'y_train_SwJNMSu.csv')
    
    # Build viewed jobs dictionary
    viewed_jobs_dict = {}
    for _, row in x_train.iterrows():
        session_id = row['session_id']
        jobs = ast.literal_eval(row['job_ids']) if isinstance(row['job_ids'], str) else row['job_ids']
        actions = ast.literal_eval(row['actions']) if isinstance(row['actions'], str) else row['actions']
        viewed = set([j for j, a in zip(jobs, actions) if a == 'view'])
        viewed_jobs_dict[session_id] = viewed
    
    # Create mock predictions with ground truth ranked first
    train_predictions_scores = []
    train_labels = []
    
    for i, row in x_train.iterrows():
        session_id = row['session_id']
        
        gt_rows = y_train[y_train['session_id'] == session_id]
        if len(gt_rows) == 0:
            continue
            
        gt_job = gt_rows.iloc[0]['job_id']
        gt_action = gt_rows.iloc[0]['action']
        
        # Create mock top-10 with ground truth first
        mock_scores = [(gt_job, 1.0, 1.0)]
        for j in range(1, 10):
            mock_scores.append((gt_job + j * 10, 1.0 / (j + 1), 1.0 / (j + 1)))
        
        train_predictions_scores.append(mock_scores)
        train_labels.append(gt_action)
    
    # Extract features for rank 1 where ground truth is
    features_list = []
    for i, (session_scores, label) in enumerate(zip(train_predictions_scores, train_labels)):
        session_row = x_train.iloc[i]
        session_jobs = ast.literal_eval(session_row['job_ids']) if isinstance(session_row['job_ids'], str) else session_row['job_ids']
        viewed_in_session = viewed_jobs_dict.get(session_row['session_id'], set())
        
        job_id, final_score, model_score = session_scores[0]
        top_scores = [s for _, _, s in session_scores[:5]]
        
        feature_dict = {
            'apply_ratio': session_row.get('apply_ratio', 0),
            'view_ratio': session_row.get('view_ratio', 1),
            'seq_length': session_row.get('seq_length', len(session_jobs)),
            'last_action_is_apply': int(session_row.get('last_action_is_apply', 0)),
            'action_change_ratio': session_row.get('action_change_ratio', 0),
            'top1_similarity': top_scores[0] if len(top_scores) > 0 else 0,
            'top2_similarity': top_scores[1] if len(top_scores) > 1 else 0,
            'top3_similarity': top_scores[2] if len(top_scores) > 2 else 0,
            'gap_top1_top2': (top_scores[0] - top_scores[1]) if len(top_scores) > 1 else 0,
            'avg_top5_similarity': np.mean(top_scores),
            'top1_was_viewed': int(job_id in viewed_in_session)
        }
        features_list.append(feature_dict)
    
    X_train = pd.DataFrame(features_list)
    y_train_binary = [1 if action == 'apply' else 0 for action in train_labels]
    
    # Train classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train_scaled, y_train_binary)
    
    print(f"Trained classifier on {len(X_train)} samples")
    
    return clf, scaler


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
    
    predictions_scores = None
    if use_hybrid:
        print("Loading diversity weights...")
        diversity_weights = load_diversity_weights(data_path)
        
        print("Loading job skills...")
        job_skills_dict = load_job_skills(data_path)
        
        print("Applying hybrid re-ranking...")
        predictions, predictions_scores = hybrid_rerank(predictions, x_test, diversity_weights, job_skills_dict)
    else:
        predictions = [jobs[:10] for jobs in predictions]
        predictions_scores = [[(job_id, 1.0/rank, 1.0/rank) for rank, job_id in enumerate(jobs, 1)] for jobs in predictions]
    
    # Train action classifier
    print("Training action classifier...")
    action_clf, action_scaler = train_action_classifier(data_path)
    
    # Build viewed jobs dictionary for test data
    viewed_jobs_dict = {}
    for _, row in x_test.iterrows():
        session_id = row['session_id']
        jobs = ast.literal_eval(row['job_ids']) if isinstance(row['job_ids'], str) else row['job_ids']
        actions = ast.literal_eval(row['actions']) if isinstance(row['actions'], str) else row['actions']
        viewed = set([j for j, a in zip(jobs, actions) if a == 'view'])
        viewed_jobs_dict[session_id] = viewed
    
    print("Formatting submission with action predictions...")
    submission_data = []
    for i, pred_jobs in enumerate(predictions):
        session_id = x_test.iloc[i]['session_id']
        session_row = x_test.iloc[i]
        session_scores = predictions_scores[i]
        
        # Predict action for top-ranked job
        session_jobs = ast.literal_eval(session_row['job_ids']) if isinstance(session_row['job_ids'], str) else session_row['job_ids']
        viewed_in_session = viewed_jobs_dict.get(session_id, set())
        top_scores = [s for _, _, s in session_scores[:5]]
        
        features = {
            'apply_ratio': session_row.get('apply_ratio', 0),
            'view_ratio': session_row.get('view_ratio', 1),
            'seq_length': session_row.get('seq_length', len(session_jobs)),
            'last_action_is_apply': int(session_row.get('last_action_is_apply', 0)),
            'action_change_ratio': session_row.get('action_change_ratio', 0),
            'top1_similarity': top_scores[0] if len(top_scores) > 0 else 0,
            'top2_similarity': top_scores[1] if len(top_scores) > 1 else 0,
            'top3_similarity': top_scores[2] if len(top_scores) > 2 else 0,
            'gap_top1_top2': (top_scores[0] - top_scores[1]) if len(top_scores) > 1 else 0,
            'avg_top5_similarity': np.mean(top_scores),
            'top1_was_viewed': int(pred_jobs[0] in viewed_in_session)
        }
        
        X_pred = pd.DataFrame([features])
        X_pred_scaled = action_scaler.transform(X_pred)
        action_pred = action_clf.predict(X_pred_scaled)[0]
        predicted_action = 'apply' if action_pred == 1 else 'view'
        
        submission_data.append({
            'session_id': session_id,
            'action': predicted_action,
            'job_id': str(pred_jobs)
        })
    
    df_submission = pd.DataFrame(submission_data)
    
    df_submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Total predictions: {len(df_submission)} sessions")
    
    if use_hybrid:
        print("\nDiversity analysis:")
        all_jobs = []
        for job_list_str in df_submission['job_id']:
            jobs = ast.literal_eval(job_list_str)
            all_jobs.extend(jobs)
        
        unique_jobs = len(set(all_jobs))
        print(f"Unique jobs recommended: {unique_jobs}")
        
        from collections import Counter
        job_freq = Counter(all_jobs)
        most_common = job_freq.most_common(1)[0]
        print(f"Most recommended job {most_common[0]} appears in {most_common[1]} sessions")
        print(f"Top 5 jobs: {[job for job, _ in job_freq.most_common(5)]}")


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
