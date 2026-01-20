import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
import ast

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.contrastive_model import (
    SessionDataset, ContrastiveRecommender, ContrastiveTrainer
)
from pipeline.build_session_features import build_features
from torch.utils.data import DataLoader


class ContrastivePipeline:
    def __init__(self, data_path: str, device: str = 'cuda'):
        self.data_path = Path(data_path)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.trainer = None
        self.all_jobs = None
        
        print(f"Using device: {self.device}")
    
    def load_data(self):
        print("\nLoading data...")
        
        x_train_path = self.data_path / 'x_train_with_features.csv'
        
        if not x_train_path.exists():
            print("Features file not found. Loading raw data and extracting features...")
            x_train = pd.read_csv(self.data_path / 'x_train_Meacfjr.csv')
            
            x_train['jobs_list'] = x_train['job_ids'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            x_train['actions_list'] = x_train['actions'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            
            x_train = build_features(x_train)
            x_train.to_csv(x_train_path, index=False)
        else:
            x_train = pd.read_csv(x_train_path)
        
        all_jobs_list = []
        for jobs in x_train['job_ids']:
            job_list = ast.literal_eval(jobs) if isinstance(jobs, str) else jobs
            all_jobs_list.extend(job_list)
        
        self.all_jobs = sorted(list(set(all_jobs_list)))
        
        print(f"Loaded {len(x_train)} training sessions")
        print(f"Total unique jobs: {len(self.all_jobs)}")
        
        return x_train
    
    def prepare_dataloaders(self, x_train: pd.DataFrame, batch_size: int = 32, 
                           val_split: float = 0.1):
        print("\nPreparing dataloaders...")
        
        train_size = int(len(x_train) * (1 - val_split))
        
        train_df = x_train.iloc[:train_size]
        val_df = x_train.iloc[train_size:]
        
        train_dataset = SessionDataset(train_df, all_jobs=self.all_jobs)
        val_dataset = SessionDataset(val_df, all_jobs=self.all_jobs)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def build_model(self, num_jobs: int):
        print("\nBuilding contrastive model...")
        
        self.model = ContrastiveRecommender(num_jobs=num_jobs, temperature=0.07)
        self.trainer = ContrastiveTrainer(self.model, device=self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def train(self, train_loader: DataLoader, num_epochs: int = 10, lr: float = 0.001):
        print(f"\nTraining for {num_epochs} epochs...")
        
        history = self.trainer.fit(train_loader, num_epochs=num_epochs, lr=lr)
        
        return history
    
    def predict(self, test_data: pd.DataFrame, top_k: int = 10, batch_size: int = 32):
        print("\nGenerating predictions...")
        
        test_dataset = SessionDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        predictions = self.trainer.predict_batch(
            test_loader, candidate_jobs=self.all_jobs, top_k=top_k
        )
        
        return predictions
    
    def save_predictions(self, predictions: list, output_path: str, session_ids: list):
        print(f"\nSaving predictions to {output_path}...")
        
        results = []
        for session_id, pred_jobs in zip(session_ids, predictions):
            results.append({
                'session_id': session_id,
                'predicted_jobs': pred_jobs
            })
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_path, index=False)
        
        print(f"Saved {len(results)} predictions")
    
    def evaluate_mrr(self, predictions: list, ground_truth: pd.DataFrame):
        print("\nEvaluating MRR...")
        
        mrr_scores = []
        
        for i, pred_jobs in enumerate(predictions):
            session_id = i
            
            true_jobs = ground_truth[ground_truth['session_id'] == session_id]
            
            if len(true_jobs) == 0:
                continue
            
            true_job = true_jobs.iloc[0]['job_id']
            
            if true_job in pred_jobs:
                rank = pred_jobs.index(true_job) + 1
                mrr_scores.append(1.0 / rank)
            else:
                mrr_scores.append(0.0)
        
        avg_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
        
        print(f"MRR: {avg_mrr:.4f}")
        
        return avg_mrr
    
    def run_full_pipeline(self, num_epochs: int = 10, batch_size: int = 32, 
                         lr: float = 0.001):
        x_train = self.load_data()
        
        train_loader, val_loader = self.prepare_dataloaders(
            x_train, batch_size=batch_size, val_split=0.1
        )
        
        self.build_model(num_jobs=max(self.all_jobs) + 100)
        
        history = self.train(train_loader, num_epochs=num_epochs, lr=lr)
        
        self.trainer.save_model('experiments/contrastive_model.pt')
        
        print("\nPipeline completed successfully")
        
        return history


def main():
    data_path = Path(__file__).parent.parent.parent / 'Data'
    
    pipeline = ContrastivePipeline(data_path=str(data_path))
    
    history = pipeline.run_full_pipeline(
        num_epochs=15,
        batch_size=64,
        lr=0.001
    )
    
    print("\nTraining complete. Model saved.")
    print("Final metrics:", history[-1])


if __name__ == '__main__':
    main()
