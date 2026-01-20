import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import ast


"""
Contrastive Learning Model for Job Recommendation

OBJECTIVE:
Predict which jobs a user will INTERACT with next (view or apply) by learning
to distinguish between relevant jobs (previously interacted) and irrelevant jobs (never seen).

HOW IT CONTRIBUTES TO FINAL GOAL:
- Creates embeddings that capture user exploration patterns
- Ranks all candidate jobs by likelihood of next interaction
- Generates top-10 job recommendations per session for competition submission

KEY INSIGHT:
Uses contrastive learning to push interacted jobs closer to session embeddings
while pushing never-seen jobs farther away in the embedding space.

RUNNING THE CODE:
1. Train model:
   python src/scripts/run_contrastive.py
   
2. Generate predictions:
   python src/scripts/generate_submission.py
   
3. Interactive exploration:
   Open notebooks/04_contrastive_learning.ipynb in VS Code

MODEL ARCHITECTURE:
- SessionEncoder: LSTM + features -> 64-dim session embedding
- JobEncoder: MLP -> 64-dim job embedding
- Loss: Maximize similarity(session, interacted_job) - similarity(session, unseen_job)

FINAL OUTPUT:
submissions/contrastive_submission.csv with top-10 job predictions per session
"""

class SessionDataset(Dataset):
    """
    Prepares training data with positive (interacted) and negative (never-seen) pairs
    """
    def __init__(self, df: pd.DataFrame, max_seq_len: int = 50, all_jobs: List[int] = None):
        self.df = df.reset_index(drop=True)
        self.max_seq_len = max_seq_len
        self.all_jobs = all_jobs if all_jobs is not None else []
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        job_ids = ast.literal_eval(row['job_ids']) if isinstance(row['job_ids'], str) else row['job_ids']
        actions = ast.literal_eval(row['actions']) if isinstance(row['actions'], str) else row['actions']
        
        action_encoded = [1 if a == 'apply' else 0 for a in actions]
        
        seq_len = min(len(job_ids), self.max_seq_len)
        job_seq = job_ids[:seq_len] + [0] * (self.max_seq_len - seq_len)
        action_seq = action_encoded[:seq_len] + [0] * (self.max_seq_len - seq_len)
        
        features = self._extract_row_features(row)
        
        interacted_jobs = list(set(job_ids))
        positive_job = int(interacted_jobs[-1]) if interacted_jobs else 0
        
        if len(self.all_jobs) > 0:
            unseen_jobs = [j for j in self.all_jobs if j not in interacted_jobs]
            negative_job = int(np.random.choice(unseen_jobs)) if unseen_jobs else 0
        else:
            negative_job = 0
        
        return {
            'job_seq': torch.LongTensor(job_seq),
            'action_seq': torch.FloatTensor(action_seq),
            'features': torch.FloatTensor(features),
            'seq_len': seq_len,
            'positive_job': positive_job,
            'negative_job': negative_job
        }
    
    def _extract_row_features(self, row):
        feature_cols = [
            'seq_length', 'unique_jobs_ratio', 'most_frequent_job_ratio',
            'avg_job_position', 'job_transitions_ratio', 'max_consecutive_views',
            'avg_consecutive_views', 'view_ratio', 'apply_ratio',
            'action_change_ratio', 'first_apply_position_ratio'
        ]
        features = []
        for col in feature_cols:
            if col in row:
                val = row[col]
                features.append(val if not pd.isna(val) else 0.0)
            else:
                features.append(0.0)
        return features


class SessionEncoder(nn.Module):
    """
    Encodes user navigation sequence + engineered features into session embedding
    """
    def __init__(self, num_jobs: int, job_emb_dim: int = 64, hidden_dim: int = 128, 
                 feature_dim: int = 11, dropout: float = 0.3):
        super().__init__()
        
        self.job_embedding = nn.Embedding(num_jobs + 1, job_emb_dim, padding_idx=0)
        self.action_linear = nn.Linear(1, 32)
        
        self.lstm = nn.LSTM(
            input_size=job_emb_dim + 32,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64)
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64)
        )
        
    def forward(self, job_seq, action_seq, features):
        job_emb = self.job_embedding(job_seq)
        action_emb = self.action_linear(action_seq.unsqueeze(-1))
        
        seq_input = torch.cat([job_emb, action_emb], dim=-1)
        
        lstm_out, (hidden, _) = self.lstm(seq_input)
        
        session_repr = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        
        feature_repr = self.feature_mlp(features)
        
        combined = torch.cat([session_repr, feature_repr], dim=-1)
        output = self.output_projection(combined)
        
        return F.normalize(output, p=2, dim=1)


class JobEncoder(nn.Module):
    """
    Encodes job ID into job embedding in shared space with sessions
    """
    def __init__(self, num_jobs: int, emb_dim: int = 64, output_dim: int = 64):
        super().__init__()
        
        self.job_embedding = nn.Embedding(num_jobs + 1, emb_dim, padding_idx=0)
        
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, job_ids):
        emb = self.job_embedding(job_ids)
        output = self.mlp(emb)
        return F.normalize(output, p=2, dim=1)


class ContrastiveRecommender(nn.Module):
    """
    Main model: learns to align sessions with interacted jobs, separate from unseen jobs
    """
    def __init__(self, num_jobs: int, temperature: float = 0.4):
        super().__init__()
        
        self.session_encoder = SessionEncoder(num_jobs)
        self.job_encoder = JobEncoder(num_jobs)
        self.temperature = temperature
        
    def forward(self, job_seq, action_seq, features, positive_jobs, negative_jobs):
        session_emb = self.session_encoder(job_seq, action_seq, features)
        
        pos_job_emb = self.job_encoder(positive_jobs)
        neg_job_emb = self.job_encoder(negative_jobs)
        
        pos_sim = (session_emb * pos_job_emb).sum(dim=1) / self.temperature
        neg_sim = (session_emb * neg_job_emb).sum(dim=1) / self.temperature
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss, pos_sim.mean(), neg_sim.mean()
    
    def get_session_embedding(self, job_seq, action_seq, features):
        return self.session_encoder(job_seq, action_seq, features)
    
    def get_job_embeddings(self, job_ids):
        return self.job_encoder(job_ids)
    
    def predict_next_action(self, job_seq, action_seq, features, candidate_jobs):
        session_emb = self.session_encoder(job_seq, action_seq, features)
        
        candidate_jobs_tensor = torch.LongTensor(candidate_jobs).to(session_emb.device)
        job_embs = self.job_encoder(candidate_jobs_tensor)
        
        similarities = torch.matmul(session_emb, job_embs.T)
        
        scores, indices = torch.sort(similarities, dim=1, descending=True)
        
        return indices.cpu().tolist(), scores.cpu().tolist()


class ContrastiveTrainer:
    def __init__(self, model: ContrastiveRecommender, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0
        total_pos_sim = 0
        total_neg_sim = 0
        
        for batch_idx, batch in enumerate(train_loader):
            job_seq = batch['job_seq'].to(self.device)
            action_seq = batch['action_seq'].to(self.device)
            features = batch['features'].to(self.device)
            pos_jobs = torch.LongTensor(batch['positive_job']).to(self.device)
            neg_jobs = torch.LongTensor(batch['negative_job']).to(self.device)
            
            self.optimizer.zero_grad()
            
            loss, pos_sim, neg_sim = self.model(
                job_seq, action_seq, features, pos_jobs, neg_jobs
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_pos_sim += pos_sim.item()
            total_neg_sim += neg_sim.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f} Pos: {pos_sim.item():.4f} Neg: {neg_sim.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        avg_pos_sim = total_pos_sim / len(train_loader)
        avg_neg_sim = total_neg_sim / len(train_loader)
        
        if self.scheduler:
            self.scheduler.step()
        
        return {
            'loss': avg_loss,
            'pos_similarity': avg_pos_sim,
            'neg_similarity': avg_neg_sim
        }
    
    def fit(self, train_loader: DataLoader, num_epochs: int = 10, lr: float = 0.001):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )
        
        history = []
        for epoch in range(num_epochs):
            metrics = self.train_epoch(train_loader, epoch)
            history.append(metrics)
            print(f"\nEpoch {epoch} Summary: Loss={metrics['loss']:.4f} "
                  f"Pos Sim={metrics['pos_similarity']:.4f} Neg Sim={metrics['neg_similarity']:.4f}\n")
        
        return history
    
    def predict_batch(self, test_loader: DataLoader, candidate_jobs: List[int], top_k: int = 10):
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                job_seq = batch['job_seq'].to(self.device)
                action_seq = batch['action_seq'].to(self.device)
                features = batch['features'].to(self.device)
                
                indices, scores = self.model.predict_next_action(
                    job_seq, action_seq, features, candidate_jobs
                )
                
                batch_predictions = []
                for i in range(len(indices)):
                    top_jobs = [candidate_jobs[idx] for idx in indices[i][:top_k]]
                    batch_predictions.append(top_jobs)
                
                all_predictions.extend(batch_predictions)
        
        return all_predictions
    
    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
