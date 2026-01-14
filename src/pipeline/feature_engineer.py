"""
Feature Engineering Module for Job Listing Classification

This module implements Phase 3 feature engineering:
- Sequence features (order, recency, frequency)
- Job features (text similarity with NLP)
- Session features (behavior patterns)
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FeatureEngineer:
    """Feature engineering for job listing recommendation system"""
    
    def __init__(self, jobs_dict: Dict = None):
        """
        Initialize feature engineer
        
        Args:
            jobs_dict: Dictionary of job listings with job_id as key
        """
        self.jobs_dict = jobs_dict
        self.tfidf_vectorizer = None
        self.job_vectors = None
        
    def set_jobs(self, jobs_dict: Dict):
        """Set job listings dictionary"""
        self.jobs_dict = jobs_dict
        
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from the dataframe
        
        Args:
            df: DataFrame with 'jobs_list' and 'actions_list' columns
            
        Returns:
            DataFrame with all engineered features
        """
        df = df.copy()
        
        # Sequence features
        df = self.add_sequence_features(df)
        
        # Session features
        df = self.add_session_features(df)
        
        # Job features (if jobs dictionary is available)
        if self.jobs_dict is not None:
            df = self.add_job_features(df)
        
        return df
    
    # ==================== SEQUENCE FEATURES ====================
    
    def add_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sequence-based features (order, recency, frequency)
        
        Features:
        - Job frequency in session
        - Position in sequence (first, last, middle)
        - Recency (inverse of position from end)
        - Action patterns (consecutive views/applies)
        """
        df = df.copy()
        
        # Job frequency features
        df['job_frequency'] = df['jobs_list'].apply(self._calculate_job_frequency)
        df['unique_jobs_count'] = df['jobs_list'].apply(lambda x: len(set(x)) if x else 0)
        df['repeat_rate'] = df.apply(
            lambda row: 1 - (row['unique_jobs_count'] / len(row['jobs_list'])) 
            if len(row['jobs_list']) > 0 else 0, 
            axis=1
        )
        
        # Position features
        df['first_job_id'] = df['jobs_list'].apply(lambda x: x[0] if x else None)
        df['last_job_id'] = df['jobs_list'].apply(lambda x: x[-1] if x else None)
        df['second_last_job_id'] = df['jobs_list'].apply(lambda x: x[-2] if len(x) > 1 else None)
        
        # Recency features (weight based on position from end)
        df['recency_scores'] = df['jobs_list'].apply(self._calculate_recency_scores)
        
        # Action sequence patterns
        df['consecutive_views'] = df['actions_list'].apply(self._count_consecutive_views)
        df['consecutive_applies'] = df['actions_list'].apply(self._count_consecutive_applies)
        df['view_to_apply_transitions'] = df['actions_list'].apply(self._count_view_to_apply)
        df['apply_to_view_transitions'] = df['actions_list'].apply(self._count_apply_to_view)
        
        return df
    
    def _calculate_job_frequency(self, jobs_list: List) -> Dict:
        """Calculate frequency of each job in the session"""
        if not jobs_list:
            return {}
        return dict(Counter(jobs_list))
    
    def _calculate_recency_scores(self, jobs_list: List) -> Dict:
        """
        Calculate recency scores for jobs (higher weight for recent jobs)
        Uses exponential decay: weight = exp(-alpha * position_from_end)
        """
        if not jobs_list:
            return {}
        
        n = len(jobs_list)
        alpha = 0.1  # decay factor
        recency_scores = {}
        
        for i, job_id in enumerate(jobs_list):
            position_from_end = n - i - 1
            score = np.exp(-alpha * position_from_end)
            
            if job_id in recency_scores:
                recency_scores[job_id] = max(recency_scores[job_id], score)
            else:
                recency_scores[job_id] = score
                
        return recency_scores
    
    def _count_consecutive_views(self, actions_list: List) -> int:
        """Count maximum consecutive views"""
        if not actions_list:
            return 0
        
        max_count = 0
        current_count = 0
        
        for action in actions_list:
            if action == 'view':
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
                
        return max_count
    
    def _count_consecutive_applies(self, actions_list: List) -> int:
        """Count maximum consecutive applies"""
        if not actions_list:
            return 0
        
        max_count = 0
        current_count = 0
        
        for action in actions_list:
            if action == 'apply':
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
                
        return max_count
    
    def _count_view_to_apply(self, actions_list: List) -> int:
        """Count transitions from view to apply"""
        if not actions_list or len(actions_list) < 2:
            return 0
        
        count = 0
        for i in range(len(actions_list) - 1):
            if actions_list[i] == 'view' and actions_list[i + 1] == 'apply':
                count += 1
        return count
    
    def _count_apply_to_view(self, actions_list: List) -> int:
        """Count transitions from apply to view"""
        if not actions_list or len(actions_list) < 2:
            return 0
        
        count = 0
        for i in range(len(actions_list) - 1):
            if actions_list[i] == 'apply' and actions_list[i + 1] == 'view':
                count += 1
        return count
    
    # ==================== SESSION FEATURES ====================
    
    def add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add session behavior features
        
        Features:
        - Session length
        - View/Apply ratio
        - Session diversity (unique jobs / total jobs)
        - Action type distribution
        """
        df = df.copy()
        
        # Basic session stats
        df['session_length'] = df['jobs_list'].apply(len)
        
        # Action distribution
        df['view_count'] = df['actions_list'].apply(lambda x: x.count('view') if x else 0)
        df['apply_count'] = df['actions_list'].apply(lambda x: x.count('apply') if x else 0)
        df['view_ratio'] = df.apply(
            lambda row: row['view_count'] / row['session_length'] 
            if row['session_length'] > 0 else 0, 
            axis=1
        )
        df['apply_ratio'] = df.apply(
            lambda row: row['apply_count'] / row['session_length'] 
            if row['session_length'] > 0 else 0, 
            axis=1
        )
        
        # Behavior pattern indicators
        df['starts_with_view'] = df['actions_list'].apply(
            lambda x: 1 if x and x[0] == 'view' else 0
        )
        df['ends_with_apply'] = df['actions_list'].apply(
            lambda x: 1 if x and x[-1] == 'apply' else 0
        )
        
        # Session diversity
        df['session_diversity'] = df.apply(
            lambda row: row['unique_jobs_count'] / row['session_length'] 
            if row['session_length'] > 0 else 0, 
            axis=1
        )
        
        # Exploration vs exploitation
        df['exploration_score'] = df['session_diversity']  # High diversity = exploration
        df['exploitation_score'] = df['repeat_rate']  # High repeat = exploitation
        
        return df
    
    # ==================== JOB FEATURES (NLP) ====================
    
    def add_job_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add job-based features using NLP
        
        Features:
        - Text similarity between jobs in session
        - Average job description length
        - Keyword diversity
        - Skill overlap
        """
        df = df.copy()
        
        # Prepare job texts if not already done
        if self.tfidf_vectorizer is None:
            self._prepare_job_vectors()
        
        # Calculate job similarity features
        df['avg_job_similarity'] = df['jobs_list'].apply(self._calculate_avg_similarity)
        df['max_job_similarity'] = df['jobs_list'].apply(self._calculate_max_similarity)
        df['min_job_similarity'] = df['jobs_list'].apply(self._calculate_min_similarity)
        
        # Job text features
        df['avg_job_text_length'] = df['jobs_list'].apply(self._calculate_avg_text_length)
        df['job_title_diversity'] = df['jobs_list'].apply(self._calculate_title_diversity)
        
        # Skill and keyword features
        df['unique_skills_count'] = df['jobs_list'].apply(self._count_unique_skills)
        df['common_keywords_count'] = df['jobs_list'].apply(self._count_common_keywords)
        
        return df
    
    def _prepare_job_vectors(self):
        """Prepare TF-IDF vectors for all jobs"""
        if self.jobs_dict is None:
            return
        
        # Extract all job texts
        job_ids = []
        job_texts = []
        
        for job_id, job_data in self.jobs_dict.items():
            job_ids.append(job_id)
            # Combine relevant text fields
            text = self._extract_job_text(job_data)
            job_texts.append(text)
        
        # Create TF-IDF vectors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.job_vectors = self.tfidf_vectorizer.fit_transform(job_texts)
        self.job_id_to_idx = {job_id: idx for idx, job_id in enumerate(job_ids)}
    
    def _extract_job_text(self, job_data: Dict) -> str:
        """Extract and clean text from job listing"""
        text_parts = []
        
        # Extract title
        if 'TITLE' in job_data:
            text_parts.append(job_data['TITLE'])
        
        # Extract summary
        if 'SUMMARY' in job_data:
            text_parts.append(job_data['SUMMARY'])
        
        # Extract skills
        if 'SKILLS' in job_data:
            text_parts.append(job_data['SKILLS'])
        
        # Combine and clean
        text = ' '.join(text_parts)
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove special characters and unicode
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower().strip()
        
        return text
    
    def _calculate_avg_similarity(self, jobs_list: List) -> float:
        """Calculate average pairwise similarity between jobs"""
        if not jobs_list or len(jobs_list) < 2:
            return 0.0
        
        similarities = []
        unique_jobs = list(set(jobs_list))
        
        for i in range(len(unique_jobs)):
            for j in range(i + 1, len(unique_jobs)):
                sim = self._get_job_similarity(unique_jobs[i], unique_jobs[j])
                if sim is not None:
                    similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_max_similarity(self, jobs_list: List) -> float:
        """Calculate maximum pairwise similarity between jobs"""
        if not jobs_list or len(jobs_list) < 2:
            return 0.0
        
        similarities = []
        unique_jobs = list(set(jobs_list))
        
        for i in range(len(unique_jobs)):
            for j in range(i + 1, len(unique_jobs)):
                sim = self._get_job_similarity(unique_jobs[i], unique_jobs[j])
                if sim is not None:
                    similarities.append(sim)
        
        return np.max(similarities) if similarities else 0.0
    
    def _calculate_min_similarity(self, jobs_list: List) -> float:
        """Calculate minimum pairwise similarity between jobs"""
        if not jobs_list or len(jobs_list) < 2:
            return 0.0
        
        similarities = []
        unique_jobs = list(set(jobs_list))
        
        for i in range(len(unique_jobs)):
            for j in range(i + 1, len(unique_jobs)):
                sim = self._get_job_similarity(unique_jobs[i], unique_jobs[j])
                if sim is not None:
                    similarities.append(sim)
        
        return np.min(similarities) if similarities else 0.0
    
    def _get_job_similarity(self, job_id1: int, job_id2: int) -> float:
        """Get cosine similarity between two jobs"""
        if self.job_vectors is None or job_id1 not in self.job_id_to_idx or job_id2 not in self.job_id_to_idx:
            return None
        
        idx1 = self.job_id_to_idx[job_id1]
        idx2 = self.job_id_to_idx[job_id2]
        
        vec1 = self.job_vectors[idx1]
        vec2 = self.job_vectors[idx2]
        
        similarity = cosine_similarity(vec1, vec2)[0, 0]
        return similarity
    
    def _calculate_avg_text_length(self, jobs_list: List) -> float:
        """Calculate average text length of jobs"""
        if not jobs_list:
            return 0.0
        
        lengths = []
        for job_id in set(jobs_list):
            if str(job_id) in self.jobs_dict:
                text = self._extract_job_text(self.jobs_dict[str(job_id)])
                lengths.append(len(text))
        
        return np.mean(lengths) if lengths else 0.0
    
    def _calculate_title_diversity(self, jobs_list: List) -> float:
        """Calculate diversity of job titles"""
        if not jobs_list:
            return 0.0
        
        titles = set()
        for job_id in jobs_list:
            if str(job_id) in self.jobs_dict and 'TITLE' in self.jobs_dict[str(job_id)]:
                title = self.jobs_dict[str(job_id)]['TITLE']
                titles.add(title.lower().strip())
        
        return len(titles) / len(jobs_list) if jobs_list else 0.0
    
    def _count_unique_skills(self, jobs_list: List) -> int:
        """Count unique skills across all jobs in session"""
        if not jobs_list:
            return 0
        
        skills = set()
        for job_id in jobs_list:
            if str(job_id) in self.jobs_dict and 'SKILLS' in self.jobs_dict[str(job_id)]:
                skill_text = self.jobs_dict[str(job_id)]['SKILLS']
                if skill_text:
                    # Simple split on common delimiters
                    job_skills = re.split(r'[,;|\n]', skill_text.lower())
                    skills.update([s.strip() for s in job_skills if s.strip()])
        
        return len(skills)
    
    def _count_common_keywords(self, jobs_list: List) -> int:
        """Count common keywords appearing in multiple jobs"""
        if not jobs_list or len(jobs_list) < 2:
            return 0
        
        # Extract keywords from each job
        job_keywords = []
        for job_id in set(jobs_list):
            if str(job_id) in self.jobs_dict:
                text = self._extract_job_text(self.jobs_dict[str(job_id)])
                keywords = set(text.split())
                job_keywords.append(keywords)
        
        if len(job_keywords) < 2:
            return 0
        
        # Find intersection
        common = job_keywords[0]
        for keywords in job_keywords[1:]:
            common = common.intersection(keywords)
        
        return len(common)
    
    # ==================== UTILITY METHODS ====================
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        features = [
            # Sequence features
            'unique_jobs_count', 'repeat_rate',
            'consecutive_views', 'consecutive_applies',
            'view_to_apply_transitions', 'apply_to_view_transitions',
            # Session features
            'session_length', 'view_count', 'apply_count',
            'view_ratio', 'apply_ratio',
            'starts_with_view', 'ends_with_apply',
            'session_diversity', 'exploration_score', 'exploitation_score',
        ]
        
        # Add job features if available
        if self.jobs_dict is not None:
            features.extend([
                'avg_job_similarity', 'max_job_similarity', 'min_job_similarity',
                'avg_job_text_length', 'job_title_diversity',
                'unique_skills_count', 'common_keywords_count'
            ])
        
        return features
