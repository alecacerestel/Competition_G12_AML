import pandas as pd
import numpy as np
from pathlib import Path
import ast
import sys
from collections import defaultdict, Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.cooccurrence_model import CooccurrenceRecommender


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


def hybrid_rerank(predictions, predictions_scores, x_data, diversity_weights, job_skills_dict, 
                  alpha=0.6, beta=0.25, gamma=0.15):
    """Apply diversity and skill-based re-ranking"""
    
    reranked_predictions = []
    all_scores = []
    
    for i, (pred_jobs, pred_scores) in enumerate(zip(predictions, predictions_scores)):
        session_jobs = ast.literal_eval(x_data.iloc[i]['job_ids']) if isinstance(x_data.iloc[i]['job_ids'], str) else x_data.iloc[i]['job_ids']
        
        job_scores = []
        
        for (job_id, model_score) in pred_scores:
            diversity_score = diversity_weights.get(job_id, 0.2)
            skill_score = compute_skill_overlap(session_jobs, job_id, job_skills_dict)
            
            final_score = alpha * model_score + beta * diversity_score + gamma * skill_score
            
            job_scores.append((job_id, final_score, model_score))
        
        job_scores.sort(key=lambda x: x[1], reverse=True)
        top_jobs = job_scores[:10]
        reranked_predictions.append([job_id for job_id, _, _ in top_jobs])
        all_scores.append(top_jobs)
    
    return reranked_predictions, all_scores


def calculate_mrr(predictions, ground_truth):
    """Calculate Mean Reciprocal Rank"""
    reciprocal_ranks = []
    
    for pred_list, gt_job in zip(predictions, ground_truth):
        if gt_job in pred_list:
            rank = pred_list.index(gt_job) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)


def calculate_hit_rate(predictions, ground_truth, k=10):
    """Calculate Hit Rate@K"""
    hits = 0
    
    for pred_list, gt_job in zip(predictions, ground_truth):
        if gt_job in pred_list[:k]:
            hits += 1
    
    return hits / len(predictions)


def train_action_classifier(x_train, y_train):
    """Train action classifier"""
    
    viewed_jobs_dict = {}
    for _, row in x_train.iterrows():
        session_id = row['session_id']
        jobs = ast.literal_eval(row['job_ids']) if isinstance(row['job_ids'], str) else row['job_ids']
        actions = ast.literal_eval(row['actions']) if isinstance(row['actions'], str) else row['actions']
        viewed = set([j for j, a in zip(jobs, actions) if a == 'view'])
        viewed_jobs_dict[session_id] = viewed
    
    train_predictions_scores = []
    train_labels = []
    
    for i, row in x_train.iterrows():
        session_id = row['session_id']
        gt_rows = y_train[y_train['session_id'] == session_id]
        if len(gt_rows) == 0:
            continue
            
        gt_job = gt_rows.iloc[0]['job_id']
        gt_action = gt_rows.iloc[0]['action']
        
        mock_scores = [(gt_job, 1.0, 1.0)]
        for j in range(1, 10):
            mock_scores.append((gt_job + j * 10, 1.0 / (j + 1), 1.0 / (j + 1)))
        
        train_predictions_scores.append(mock_scores)
        train_labels.append(gt_action)
    
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42, probability=True)
    clf.fit(X_train_scaled, y_train_binary)
    
    return clf, scaler


def evaluate_cooccurrence_model(model_path: str, data_path: str):
    data_path = Path(data_path)
    
    print("Loading data...")
    x_train = pd.read_csv(data_path / 'x_train_with_features.csv')
    y_train = pd.read_csv(data_path / 'y_train_SwJNMSu.csv')
    
    # Split for validation
    train_sessions = x_train['session_id'].unique()
    train_idx, val_idx = train_test_split(range(len(train_sessions)), test_size=0.2, random_state=42)
    val_sessions = train_sessions[val_idx]
    
    x_val = x_train[x_train['session_id'].isin(val_sessions)].reset_index(drop=True)
    y_val = y_train[y_train['session_id'].isin(val_sessions)].reset_index(drop=True)
    
    x_train_split = x_train[~x_train['session_id'].isin(val_sessions)].reset_index(drop=True)
    y_train_split = y_train[~y_train['session_id'].isin(val_sessions)].reset_index(drop=True)
    
    print(f"Train: {len(x_train_split)} sessions, Val: {len(x_val)} sessions")
    
    # Load model
    print("Loading co-occurrence model...")
    model = CooccurrenceRecommender()
    model.load(model_path)
    
    # Generate predictions
    print("Generating predictions...")
    val_sessions_list = []
    for _, row in x_val.iterrows():
        jobs = ast.literal_eval(row['job_ids']) if isinstance(row['job_ids'], str) else row['job_ids']
        val_sessions_list.append(jobs)
    
    predictions = []
    predictions_scores = []
    
    for session_jobs in val_sessions_list:
        pred_with_scores = model.predict(session_jobs, top_k=20, return_scores=True)
        predictions.append([job for job, _ in pred_with_scores])
        predictions_scores.append(pred_with_scores)
    
    # Apply hybrid re-ranking
    print("Loading diversity weights and job skills...")
    diversity_weights = load_diversity_weights(data_path)
    job_skills_dict = load_job_skills(data_path)
    
    print("Applying hybrid re-ranking...")
    predictions, predictions_scores = hybrid_rerank(
        predictions, predictions_scores, x_val, 
        diversity_weights, job_skills_dict
    )
    
    # Get ground truth
    gt_jobs = []
    gt_actions = []
    for i, row in x_val.iterrows():
        session_id = row['session_id']
        gt_row = y_val[y_val['session_id'] == session_id]
        if len(gt_row) > 0:
            gt_jobs.append(gt_row.iloc[0]['job_id'])
            gt_actions.append(gt_row.iloc[0]['action'])
        else:
            gt_jobs.append(-1)
            gt_actions.append('view')
    
    # Calculate recommendation metrics
    print("\n=== Recommendation Quality ===")
    mrr = calculate_mrr(predictions, gt_jobs)
    print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
    
    for k in [1, 5, 10]:
        hit_rate = calculate_hit_rate(predictions, gt_jobs, k=k)
        print(f"Hit Rate@{k}: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    
    # Train action classifier and evaluate
    print("\n=== Action Prediction Quality ===")
    print("Training action classifier...")
    action_clf, action_scaler = train_action_classifier(x_train_split, y_train_split)
    
    # Predict actions
    viewed_jobs_dict = {}
    for _, row in x_val.iterrows():
        session_id = row['session_id']
        jobs = ast.literal_eval(row['job_ids']) if isinstance(row['job_ids'], str) else row['job_ids']
        actions = ast.literal_eval(row['actions']) if isinstance(row['actions'], str) else row['actions']
        viewed = set([j for j, a in zip(jobs, actions) if a == 'view'])
        viewed_jobs_dict[session_id] = viewed
    
    predicted_actions = []
    for i, session_scores in enumerate(predictions_scores):
        session_row = x_val.iloc[i]
        session_jobs = ast.literal_eval(session_row['job_ids']) if isinstance(session_row['job_ids'], str) else session_row['job_ids']
        viewed_in_session = viewed_jobs_dict.get(session_row['session_id'], set())
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
            'top1_was_viewed': int(predictions[i][0] in viewed_in_session)
        }
        
        X_pred = pd.DataFrame([features])
        X_pred_scaled = action_scaler.transform(X_pred)
        action_pred = action_clf.predict(X_pred_scaled)[0]
        predicted_actions.append('apply' if action_pred == 1 else 'view')
    
    # Calculate action accuracy
    correct = sum([1 for pred, gt in zip(predicted_actions, gt_actions) if pred == gt])
    accuracy = correct / len(gt_actions)
    print(f"Action Prediction Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Calculate final score
    final_score = 0.7 * mrr + 0.3 * accuracy
    print(f"\n=== Final Challenge Score ===")
    print(f"Score = 0.7 * MRR + 0.3 * Accuracy")
    print(f"Score = 0.7 * {mrr:.4f} + 0.3 * {accuracy:.4f}")
    print(f"Final Score: {final_score:.4f}")
    
    # Show confusion matrix
    confusion = {'view_view': 0, 'view_apply': 0, 'apply_view': 0, 'apply_apply': 0}
    for pred, gt in zip(predicted_actions, gt_actions):
        confusion[f"{pred}_{gt}"] += 1
    
    print("\nConfusion Matrix:")
    print(f"  Predicted View -> Actual View: {confusion['view_view']}")
    print(f"  Predicted View -> Actual Apply: {confusion['view_apply']}")
    print(f"  Predicted Apply -> Actual View: {confusion['apply_view']}")
    print(f"  Predicted Apply -> Actual Apply: {confusion['apply_apply']}")
    
    # Action distribution
    print("\nAction Distribution:")
    pred_counter = Counter(predicted_actions)
    gt_counter = Counter(gt_actions)
    print(f"  Predicted: View={pred_counter['view']}, Apply={pred_counter['apply']}")
    print(f"  Ground Truth: View={gt_counter['view']}, Apply={gt_counter['apply']}")


def main():
    base_path = Path(__file__).parent.parent.parent
    
    model_path = base_path / 'experiments' / 'cooccurrence_model.pkl'
    data_path = base_path / 'Data'
    
    evaluate_cooccurrence_model(
        model_path=str(model_path),
        data_path=str(data_path)
    )


if __name__ == '__main__':
    main()
