import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
from pathlib import Path
import ast


class CooccurrenceRecommender:
    """Item-to-item collaborative filtering using job co-occurrence patterns"""
    
    def __init__(self, window_size=3, min_count=2):
        self.window_size = window_size
        self.min_count = min_count
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.job_counts = defaultdict(int)
        self.all_jobs = set()
        
    def fit(self, x_train):
        """Build co-occurrence matrix from training sessions"""
        print(f"Building co-occurrence matrix with window_size={self.window_size}...")
        
        for _, row in x_train.iterrows():
            jobs = ast.literal_eval(row['job_ids']) if isinstance(row['job_ids'], str) else row['job_ids']
            
            # Count job frequencies
            for job in jobs:
                self.job_counts[job] += 1
                self.all_jobs.add(job)
            
            # Count transitions within window
            for i in range(len(jobs)):
                current_job = jobs[i]
                
                # Look ahead within window
                for j in range(i + 1, min(i + self.window_size + 1, len(jobs))):
                    next_job = jobs[j]
                    # Weight by distance: closer jobs get higher weight
                    weight = 1.0 / (j - i)
                    self.transitions[current_job][next_job] += weight
        
        print(f"Found {len(self.transitions)} source jobs with transitions")
        print(f"Total unique jobs: {len(self.all_jobs)}")
        
        return self
    
    def _get_candidates_for_job(self, job_id, top_k=50):
        """Get top-k most likely next jobs given a job"""
        if job_id not in self.transitions:
            return []
        
        candidates = []
        for next_job, count in self.transitions[job_id].items():
            if count >= self.min_count:
                # Score is normalized count
                score = count / self.job_counts[job_id]
                candidates.append((next_job, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def predict(self, session_jobs, top_k=10, return_scores=False):
        """Predict next jobs based on session history"""
        if len(session_jobs) == 0:
            popular = self._get_popular_jobs(top_k)
            if return_scores:
                return [(job, 1.0) for job in popular]
            return popular
        
        # Use last N jobs as context
        context_jobs = session_jobs[-self.window_size:]
        
        # Aggregate scores from all context jobs
        candidate_scores = defaultdict(float)
        
        for idx, job in enumerate(reversed(context_jobs)):
            # Recent jobs get higher weight
            position_weight = (idx + 1) / len(context_jobs)
            
            candidates = self._get_candidates_for_job(job, top_k=50)
            
            for next_job, score in candidates:
                # Don't recommend jobs already in session
                if next_job not in session_jobs:
                    candidate_scores[next_job] += score * position_weight
        
        # Fallback to popular jobs if no candidates
        if len(candidate_scores) == 0:
            popular = self._get_popular_jobs(top_k)
            if return_scores:
                return [(job, 1.0) for job in popular]
            return popular
        
        # Sort by score
        ranked_jobs = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        if return_scores:
            return [(job, score) for job, score in ranked_jobs[:top_k]]
        else:
            return [job for job, score in ranked_jobs[:top_k]]
    
    def _get_popular_jobs(self, top_k=10):
        """Fallback to most popular jobs"""
        popular = sorted(self.job_counts.items(), key=lambda x: x[1], reverse=True)
        return [job for job, count in popular[:top_k]]
    
    def predict_batch(self, sessions, top_k=10):
        """Predict for multiple sessions"""
        predictions = []
        
        for session_jobs in sessions:
            pred = self.predict(session_jobs, top_k=top_k)
            predictions.append(pred)
        
        return predictions
    
    def save(self, path):
        """Save model to disk"""
        model_data = {
            'transitions': dict(self.transitions),
            'job_counts': dict(self.job_counts),
            'all_jobs': self.all_jobs,
            'window_size': self.window_size,
            'min_count': self.min_count
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.transitions = defaultdict(lambda: defaultdict(int), model_data['transitions'])
        self.job_counts = defaultdict(int, model_data['job_counts'])
        self.all_jobs = model_data['all_jobs']
        self.window_size = model_data['window_size']
        self.min_count = model_data['min_count']
        
        print(f"Model loaded from {path}")
        return self
