"""
Job Ranking Model.

Combines multiple signals to rank jobs for recommendation:
1. Transition probability (job A -> job B)
2. Co-occurrence (jobs appearing in same session)
3. Collaborative Filtering scores
4. Job popularity
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd

from .collaborative_filter import CollaborativeFilter


class JobRanker:
    """
    Lightweight job ranking model combining multiple signals.
    
    This follows a simple weighted combination approach that's
    interpretable and doesn't require heavy neural network training.
    """
    
    def __init__(
        self,
        transition_weight: float = 5.0,
        cooccurrence_weight: float = 2.0,
        cf_weight: float = 1.0,
        popularity_weight: float = 0.1,
        k_similar_sessions: int = 50
    ):
        """
        Args:
            transition_weight: Weight for transition probabilities
            cooccurrence_weight: Weight for co-occurrence scores
            cf_weight: Weight for collaborative filtering scores
            popularity_weight: Weight for popularity scores
            k_similar_sessions: Number of similar sessions for CF
        """
        self.transition_weight = transition_weight
        self.cooccurrence_weight = cooccurrence_weight
        self.cf_weight = cf_weight
        self.popularity_weight = popularity_weight
        
        # Models/statistics
        self.transition_counts = None  # job_a -> {job_b: count}
        self.cooccurrence_counts = None  # job_a -> {job_b: count}
        self.job_popularity = None  # job -> count
        self.cf_model = None  # Collaborative filter
        
        self.job_to_idx = None
        self.idx_to_job = None
        
    def fit(self, train_df: pd.DataFrame, job_to_idx: Dict[int, int]) -> 'JobRanker':
        """
        Build all ranking models from training data.
        
        Args:
            train_df: Training DataFrame with 'job_ids' column
            job_to_idx: Job ID to index mapping
        """
        print("Building job ranker...")
        
        self.job_to_idx = job_to_idx
        self.idx_to_job = {v: k for k, v in job_to_idx.items()}
        
        # 1. Build transition probabilities (job_i -> job_{i+1})
        print("  Building transition model...")
        self.transition_counts = {}  # Use regular dict for pickling
        
        for _, row in train_df.iterrows():
            jobs = row['job_ids']
            # Include target as the next job
            if 'job_id' in row:
                jobs = jobs + [row['job_id']]
            
            for i in range(len(jobs) - 1):
                if jobs[i] not in self.transition_counts:
                    self.transition_counts[jobs[i]] = {}
                if jobs[i + 1] not in self.transition_counts[jobs[i]]:
                    self.transition_counts[jobs[i]][jobs[i + 1]] = 0
                self.transition_counts[jobs[i]][jobs[i + 1]] += 1
        
        print(f"    Unique transitions: {sum(len(v) for v in self.transition_counts.values())}")
        
        # 2. Build co-occurrence (jobs in same session)
        print("  Building co-occurrence model...")
        self.cooccurrence_counts = {}  # Use regular dict for pickling
        
        for _, row in train_df.iterrows():
            jobs = row['job_ids']
            for i, job1 in enumerate(jobs):
                for job2 in jobs[i + 1:]:
                    if job1 not in self.cooccurrence_counts:
                        self.cooccurrence_counts[job1] = {}
                    if job2 not in self.cooccurrence_counts[job1]:
                        self.cooccurrence_counts[job1][job2] = 0
                    self.cooccurrence_counts[job1][job2] += 1
                    
                    if job2 not in self.cooccurrence_counts:
                        self.cooccurrence_counts[job2] = {}
                    if job1 not in self.cooccurrence_counts[job2]:
                        self.cooccurrence_counts[job2][job1] = 0
                    self.cooccurrence_counts[job2][job1] += 1
        
        print(f"    Unique co-occurrences: {sum(len(v) for v in self.cooccurrence_counts.values())}")
        
        # 3. Build job popularity
        print("  Building popularity model...")
        self.job_popularity = {}  # Use regular dict for pickling
        
        for _, row in train_df.iterrows():
            for job in row['job_ids']:
                if job not in self.job_popularity:
                    self.job_popularity[job] = 0
                self.job_popularity[job] += 1
            if 'job_id' in row:
                job = row['job_id']
                if job not in self.job_popularity:
                    self.job_popularity[job] = 0
                self.job_popularity[job] += 1
        
        print(f"    Jobs with popularity data: {len(self.job_popularity)}")
        
        # 4. Build collaborative filter
        print("  Building collaborative filter...")
        self.cf_model = CollaborativeFilter(k_neighbors=50)
        self.cf_model.fit(train_df, job_to_idx)
        
        return self
    
    def _get_transition_scores(self, session_jobs: List[int]) -> Dict[int, float]:
        """Get transition probability scores from the last job in session."""
        if not session_jobs:
            return {}
        
        last_job = session_jobs[-1]
        trans = self.transition_counts.get(last_job, {})
        
        if not trans:
            return {}
        
        total = sum(trans.values())
        return {job: count / total for job, count in trans.items()}
    
    def _get_cooccurrence_scores(self, session_jobs: List[int]) -> Dict[int, float]:
        """Get co-occurrence scores based on all jobs in session."""
        scores = defaultdict(float)
        
        for session_job in session_jobs:
            cooc = self.cooccurrence_counts.get(session_job, {})
            total = sum(cooc.values()) if cooc else 1
            
            for job, count in cooc.items():
                scores[job] += count / total
        
        # Normalize by session length
        if session_jobs:
            for job in scores:
                scores[job] /= len(session_jobs)
        
        return dict(scores)
    
    def _get_popularity_scores(self) -> Dict[int, float]:
        """Get normalized popularity scores."""
        if not self.job_popularity:
            return {}
        
        max_pop = max(self.job_popularity.values())
        return {job: count / max_pop for job, count in self.job_popularity.items()}
    
    def rank_jobs(
        self, 
        session_jobs: List[int], 
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Rank all jobs for a given session.
        
        Args:
            session_jobs: Jobs viewed in the current session
            top_k: Number of top jobs to return
            
        Returns:
            List of (job_id, score) tuples sorted by score descending
        """
        exclude_jobs = set(session_jobs)
        
        # Get scores from each model
        transition_scores = self._get_transition_scores(session_jobs)
        cooccurrence_scores = self._get_cooccurrence_scores(session_jobs)
        popularity_scores = self._get_popularity_scores()
        cf_scores = self.cf_model.predict_job_scores(session_jobs, exclude_jobs)
        
        # Combine all scores
        final_scores = defaultdict(float)
        
        # Transition scores (strongest signal for direct prediction)
        for job, score in transition_scores.items():
            if job not in exclude_jobs:
                final_scores[job] += self.transition_weight * score
        
        # Co-occurrence scores
        for job, score in cooccurrence_scores.items():
            if job not in exclude_jobs:
                final_scores[job] += self.cooccurrence_weight * score
        
        # CF scores
        for job, score in cf_scores.items():
            if job not in exclude_jobs:
                final_scores[job] += self.cf_weight * score
        
        # Popularity scores (fallback)
        for job, score in popularity_scores.items():
            if job not in exclude_jobs:
                final_scores[job] += self.popularity_weight * score
        
        # Sort and return top-k
        sorted_jobs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_jobs[:top_k]
    
    def predict_top_k(self, session_jobs: List[int], k: int = 10) -> List[int]:
        """
        Predict top-k job IDs for a session.
        
        Args:
            session_jobs: Jobs in the current session
            k: Number of jobs to predict
            
        Returns:
            List of top-k job IDs
        """
        ranked = self.rank_jobs(session_jobs, top_k=k)
        return [job_id for job_id, score in ranked]
