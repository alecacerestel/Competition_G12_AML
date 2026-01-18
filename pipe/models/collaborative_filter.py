"""
Collaborative Filtering Implementation.

Following the competition benchmark:
1. Build User-Job Interaction Matrix (sessions as users)
2. Find similar sessions using cosine similarity
3. Predict job scores based on similar sessions' interactions
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import pandas as pd


class CollaborativeFilter:
    """
    Session-based Collaborative Filtering.
    
    Treats each session as a "user" and builds an interaction matrix
    where R[session, job] = 1 if the session interacted with that job.
    
    For a new session, finds similar sessions and recommends jobs
    based on what similar sessions interacted with.
    """
    
    def __init__(self, k_neighbors: int = 50, min_similarity: float = 0.01):
        """
        Args:
            k_neighbors: Number of similar sessions to consider
            min_similarity: Minimum similarity threshold
        """
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity
        
        # Will be built during fit
        self.session_ids = []           # List of session IDs
        self.session_to_idx = {}        # session_id -> index
        self.job_to_idx = {}            # job_id -> index
        self.idx_to_job = {}            # index -> job_id
        self.interaction_matrix = None  # Sparse session-job matrix
        self.session_norms = None       # Precomputed norms for similarity
        
    def fit(self, train_df: pd.DataFrame, job_to_idx: Dict[int, int]) -> 'CollaborativeFilter':
        """
        Build the interaction matrix from training data.
        
        Args:
            train_df: Training DataFrame with 'session_id' and 'job_ids' columns
            job_to_idx: Job ID to index mapping
        """
        print("  Building collaborative filter...")
        
        self.job_to_idx = job_to_idx
        self.idx_to_job = {v: k for k, v in job_to_idx.items()}
        n_jobs = len(job_to_idx)
        
        # Get unique sessions
        self.session_ids = train_df['session_id'].unique().tolist()
        self.session_to_idx = {sid: idx for idx, sid in enumerate(self.session_ids)}
        n_sessions = len(self.session_ids)
        
        # Build sparse interaction matrix
        mat = lil_matrix((n_sessions, n_jobs), dtype=np.float32)
        
        for _, row in train_df.iterrows():
            session_idx = self.session_to_idx[row['session_id']]
            for job in row['job_ids']:
                if job in self.job_to_idx:
                    job_idx = self.job_to_idx[job]
                    mat[session_idx, job_idx] = 1.0
            
            # Also include target job
            if 'job_id' in row and row['job_id'] in self.job_to_idx:
                job_idx = self.job_to_idx[row['job_id']]
                mat[session_idx, job_idx] = 1.0
        
        self.interaction_matrix = csr_matrix(mat)
        
        # Precompute norms for cosine similarity
        self.session_norms = np.sqrt(np.array(self.interaction_matrix.power(2).sum(axis=1)).flatten())
        self.session_norms[self.session_norms == 0] = 1.0  # Avoid division by zero
        
        print(f"    Sessions: {n_sessions}, Jobs: {n_jobs}")
        print(f"    Interaction density: {self.interaction_matrix.nnz / (n_sessions * n_jobs) * 100:.4f}%")
        
        return self
    
    def get_session_vector(self, job_ids: List[int]) -> np.ndarray:
        """Convert a list of job IDs to a sparse vector."""
        vec = np.zeros(len(self.job_to_idx), dtype=np.float32)
        for job in job_ids:
            if job in self.job_to_idx:
                vec[self.job_to_idx[job]] = 1.0
        return vec
    
    def find_similar_sessions(self, query_jobs: List[int]) -> List[Tuple[int, float]]:
        """
        Find k most similar sessions to the query session.
        
        Args:
            query_jobs: List of job IDs in the query session
            
        Returns:
            List of (session_idx, similarity_score) tuples
        """
        # Convert query to vector
        query_vec = self.get_session_vector(query_jobs)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        # Compute similarities with all sessions
        # similarity = (query · session) / (||query|| * ||session||)
        dot_products = self.interaction_matrix.dot(query_vec)
        similarities = dot_products / (self.session_norms * query_norm)
        
        # Get top-k similar sessions
        top_k_indices = np.argsort(similarities)[-self.k_neighbors:][::-1]
        
        result = []
        for idx in top_k_indices:
            sim = similarities[idx]
            if sim >= self.min_similarity:
                result.append((idx, sim))
        
        return result
    
    def predict_job_scores(
        self, 
        query_jobs: List[int], 
        exclude_jobs: Set[int] = None
    ) -> Dict[int, float]:
        """
        Predict job scores based on collaborative filtering.
        
        For each job j, the score is:
            P_j = sum(similarity_u * R_uj) / sum(similarity_u)
        
        where similarity_u is the similarity to session u, and R_uj is 1 if
        session u interacted with job j.
        
        Args:
            query_jobs: Jobs in the current session
            exclude_jobs: Jobs to exclude from predictions
            
        Returns:
            Dictionary of job_id -> CF score
        """
        if exclude_jobs is None:
            exclude_jobs = set(query_jobs)
        
        similar_sessions = self.find_similar_sessions(query_jobs)
        
        if not similar_sessions:
            return {}
        
        # Aggregate interactions from similar sessions
        scores = defaultdict(float)
        total_similarity = sum(sim for _, sim in similar_sessions)
        
        if total_similarity == 0:
            return {}
        
        for session_idx, similarity in similar_sessions:
            # Get jobs this session interacted with
            row = self.interaction_matrix.getrow(session_idx)
            job_indices = row.indices
            
            for job_idx in job_indices:
                job_id = self.idx_to_job[job_idx]
                if job_id not in exclude_jobs:
                    scores[job_id] += similarity
        
        # Normalize by total similarity
        for job_id in scores:
            scores[job_id] /= total_similarity
        
        return dict(scores)
