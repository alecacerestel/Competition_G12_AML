import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Comprehensive feature engineering for job recommendation system.
    Includes sequence features, job text features, and session behavior patterns.
    """
    
    def __init__(self, jobs_dict: Dict = None):
        self.jobs = jobs_dict
        self.tfidf_vectorizer = None
        self.job_tfidf_matrix = None
        
    def set_jobs_dict(self, jobs_dict: Dict):
        """Set the jobs dictionary for text feature extraction"""
        self.jobs = jobs_dict
        
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features: sequence, job text, and session features
        """
        df = df.copy()
        
        # Extract sequence features
        df = self.extract_sequence_features(df)
        
        # Extract session behavior features
        df = self.extract_session_features(df)
        
        # Extract job-specific features (if jobs dict is available)
        if self.jobs is not None:
            df = self.extract_job_text_features(df)
        
        return df
    
    def extract_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract sequence-based features: order, recency, frequency
        """
        print("Extracting sequence features...")
        
        # Basic sequence length
        df['seq_length'] = df['jobs_list'].apply(len)
        
        # Position-based features (order)
        df['first_job'] = df['jobs_list'].apply(lambda x: x[0] if len(x) > 0 else -1)
        df['last_job'] = df['jobs_list'].apply(lambda x: x[-1] if len(x) > 0 else -1)
        df['second_job'] = df['jobs_list'].apply(lambda x: x[1] if len(x) > 1 else -1)
        df['second_last_job'] = df['jobs_list'].apply(lambda x: x[-2] if len(x) > 1 else -1)
        
        # Job frequency in sequence
        df['unique_jobs_count'] = df['jobs_list'].apply(lambda x: len(set(x)))
        df['unique_jobs_ratio'] = df['unique_jobs_count'] / (df['seq_length'] + 1)
        
        # Most frequent job and its frequency
        df['most_frequent_job'] = df['jobs_list'].apply(
            lambda x: Counter(x).most_common(1)[0][0] if len(x) > 0 else -1
        )
        df['most_frequent_job_count'] = df['jobs_list'].apply(
            lambda x: Counter(x).most_common(1)[0][1] if len(x) > 0 else 0
        )
        df['most_frequent_job_ratio'] = df['most_frequent_job_count'] / (df['seq_length'] + 1)
        
        # Recency features - position of last occurrence of each job
        df['job_last_positions'] = df['jobs_list'].apply(self._get_job_last_positions)
        
        # Average position of jobs (earlier = smaller values)
        df['avg_job_position'] = df['jobs_list'].apply(
            lambda x: np.mean(range(len(x))) if len(x) > 0 else 0
        )
        
        # Job transitions (how many times user switches between different jobs)
        df['job_transitions'] = df['jobs_list'].apply(
            lambda x: sum(1 for i in range(len(x)-1) if x[i] != x[i+1]) if len(x) > 1 else 0
        )
        df['job_transitions_ratio'] = df['job_transitions'] / (df['seq_length'] + 1)
        
        # Consecutive same job views
        df['max_consecutive_views'] = df['jobs_list'].apply(self._max_consecutive)
        df['avg_consecutive_views'] = df['jobs_list'].apply(self._avg_consecutive)
        
        return df
    
    def extract_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract session behavior pattern features
        """
        print("Extracting session behavior features...")
        
        # Action-based features
        if 'actions_list' in df.columns:
            # Count of each action type
            df['view_count'] = df['actions_list'].apply(
                lambda x: sum(1 for a in x if a == 'view')
            )
            df['apply_count'] = df['actions_list'].apply(
                lambda x: sum(1 for a in x if a == 'apply')
            )
            
            # Action ratios
            df['view_ratio'] = df['view_count'] / (df['seq_length'] + 1)
            df['apply_ratio'] = df['apply_count'] / (df['seq_length'] + 1)
            
            # First and last actions
            df['first_action_is_apply'] = df['actions_list'].apply(
                lambda x: 1 if len(x) > 0 and x[0] == 'apply' else 0
            )
            df['last_action_is_apply'] = df['actions_list'].apply(
                lambda x: 1 if len(x) > 0 and x[-1] == 'apply' else 0
            )
            
            # Action transitions (view -> apply or apply -> view)
            df['view_to_apply_count'] = df['actions_list'].apply(
                lambda x: sum(1 for i in range(len(x)-1) if x[i] == 'view' and x[i+1] == 'apply')
            )
            df['apply_to_view_count'] = df['actions_list'].apply(
                lambda x: sum(1 for i in range(len(x)-1) if x[i] == 'apply' and x[i+1] == 'view')
            )
            
            # Action change ratio
            df['action_change_ratio'] = df['actions_list'].apply(
                lambda x: sum(1 for i in range(len(x)-1) if x[i] != x[i+1]) / len(x) if len(x) > 1 else 0
            )
            
            # Positions of applies in sequence
            df['first_apply_position'] = df['actions_list'].apply(
                lambda x: next((i for i, a in enumerate(x) if a == 'apply'), -1)
            )
            df['first_apply_position_ratio'] = df.apply(
                lambda row: row['first_apply_position'] / row['seq_length'] if row['seq_length'] > 0 and row['first_apply_position'] >= 0 else -1,
                axis=1
            )
            
            # Applied job IDs
            df['applied_jobs'] = df.apply(
                lambda row: [job for job, action in zip(row['jobs_list'], row['actions_list']) if action == 'apply'],
                axis=1
            )
            df['applied_jobs_count'] = df['applied_jobs'].apply(len)
            df['unique_applied_jobs_count'] = df['applied_jobs'].apply(lambda x: len(set(x)))
            
            # Viewed but not applied
            df['viewed_jobs'] = df.apply(
                lambda row: [job for job, action in zip(row['jobs_list'], row['actions_list']) if action == 'view'],
                axis=1
            )
            df['viewed_not_applied_count'] = df.apply(
                lambda row: len(set(row['viewed_jobs']) - set(row['applied_jobs'])),
                axis=1
            )
            
        return df
    
    def extract_job_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract NLP-based features from job text similarity
        """
        print("Extracting job text features with NLP...")
        
        if self.jobs is None:
            print("Warning: Jobs dictionary not provided. Skipping text features.")
            return df
        
        # Build TF-IDF matrix if not already built
        if self.tfidf_vectorizer is None:
            self._build_tfidf_matrix()
        
        # Calculate similarity features for each session
        df['avg_job_similarity'] = df['jobs_list'].apply(self._avg_job_similarity)
        df['max_job_similarity'] = df['jobs_list'].apply(self._max_job_similarity)
        df['min_job_similarity'] = df['jobs_list'].apply(self._min_job_similarity)
        
        # Similarity between first and last job
        df['first_last_similarity'] = df.apply(
            lambda row: self._pairwise_job_similarity(row['first_job'], row['last_job'])
            if row['seq_length'] > 1 else 0,
            axis=1
        )
        
        # Job category features (if available in jobs dict)
        if self.jobs:
            df['avg_skills_count'] = df['jobs_list'].apply(self._avg_skills_count)
            df['total_skills_count'] = df['jobs_list'].apply(self._total_unique_skills_count)
            
            df['has_finance_job'] = df['jobs_list'].apply(self._has_keyword, keyword='finance')
            df['has_tech_job'] = df['jobs_list'].apply(self._has_keyword, keyword='tech')
            df['has_management_job'] = df['jobs_list'].apply(self._has_keyword, keyword='management')
        
        return df
    
    def _build_tfidf_matrix(self):
        """Build TF-IDF matrix for all jobs"""
        print("Building TF-IDF matrix for job descriptions...")
        
        # Combine all text fields from job postings
        job_texts = []
        job_ids = []
        
        for job_id, job_data in self.jobs.items():
            # Job data is stored as a formatted string, use it directly
            if isinstance(job_data, str):
                job_texts.append(job_data)
            else:
                # Fallback for dictionary format
                text_parts = []
                
                if 'TITLE' in job_data and job_data['TITLE']:
                    text_parts.append(str(job_data['TITLE']))
                
                if 'SUMMARY' in job_data and job_data['SUMMARY']:
                    text_parts.append(str(job_data['SUMMARY']))
                
                if 'DESCRIPTION' in job_data and job_data['DESCRIPTION']:
                    for desc in job_data['DESCRIPTION']:
                        if isinstance(desc, dict) and 'DESCRIPTION' in desc:
                            text_parts.append(str(desc['DESCRIPTION']))
                
                # Combine all text
                combined_text = ' '.join(text_parts)
                job_texts.append(combined_text)
            
            job_ids.append(job_id)
        
        # Build TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.job_tfidf_matrix = self.tfidf_vectorizer.fit_transform(job_texts)
        self.job_id_to_idx = {job_id: idx for idx, job_id in enumerate(job_ids)}
        
        print(f"TF-IDF matrix built: {self.job_tfidf_matrix.shape}")
    
    def _pairwise_job_similarity(self, job_id1, job_id2) -> float:
        """Calculate cosine similarity between two jobs"""
        if job_id1 == -1 or job_id2 == -1:
            return 0.0
        
        job_id1_str = str(job_id1)
        job_id2_str = str(job_id2)
        
        if job_id1_str not in self.job_id_to_idx or job_id2_str not in self.job_id_to_idx:
            return 0.0
        
        idx1 = self.job_id_to_idx[job_id1_str]
        idx2 = self.job_id_to_idx[job_id2_str]
        
        vec1 = self.job_tfidf_matrix[idx1]
        vec2 = self.job_tfidf_matrix[idx2]
        
        similarity = cosine_similarity(vec1, vec2)[0][0]
        return similarity
    
    def _avg_job_similarity(self, jobs_list: List) -> float:
        """Calculate average pairwise similarity in a job sequence"""
        if len(jobs_list) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(jobs_list) - 1):
            sim = self._pairwise_job_similarity(jobs_list[i], jobs_list[i+1])
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _max_job_similarity(self, jobs_list: List) -> float:
        """Calculate maximum pairwise similarity in a job sequence"""
        if len(jobs_list) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(jobs_list) - 1):
            sim = self._pairwise_job_similarity(jobs_list[i], jobs_list[i+1])
            similarities.append(sim)
        
        return max(similarities) if similarities else 0.0
    
    def _min_job_similarity(self, jobs_list: List) -> float:
        """Calculate minimum pairwise similarity in a job sequence"""
        if len(jobs_list) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(jobs_list) - 1):
            sim = self._pairwise_job_similarity(jobs_list[i], jobs_list[i+1])
            similarities.append(sim)
        
        return min(similarities) if similarities else 0.0
    
    def _avg_skills_count(self, jobs_list: List) -> float:
        """Average number of skills across jobs in sequence"""
        if not self.jobs:
            return 0.0
        
        skill_counts = []
        for job_id in jobs_list:
            job_id_str = str(job_id)
            if job_id_str in self.jobs:
                job_data = self.jobs[job_id_str]
                if isinstance(job_data, str):
                    # Count skills from formatted string
                    skills_section = job_data.split('SKILLS')[1] if 'SKILLS' in job_data else ''
                    # Count lines that start with '-' in skills section (up to next section)
                    next_section = skills_section.split('\n\n')[0] if skills_section else ''
                    skill_lines = [line for line in next_section.split('\n') if line.strip().startswith('-')]
                    if skill_lines:
                        skill_counts.append(len(skill_lines))
                elif isinstance(job_data, dict) and 'SKILLS' in job_data:
                    skills = job_data['SKILLS']
                    if skills:
                        skill_counts.append(len(skills))
        
        return np.mean(skill_counts) if skill_counts else 0.0
    
    def _total_unique_skills_count(self, jobs_list: List) -> int:
        """Total unique skills across all jobs in sequence"""
        if not self.jobs:
            return 0
        
        all_skills = set()
        for job_id in jobs_list:
            job_id_str = str(job_id)
            if job_id_str in self.jobs:
                job_data = self.jobs[job_id_str]
                if isinstance(job_data, str):
                    # Extract skills from formatted string
                    skills_section = job_data.split('SKILLS')[1] if 'SKILLS' in job_data else ''
                    next_section = skills_section.split('\n\n')[0] if skills_section else ''
                    skill_lines = [line for line in next_section.split('\n') if line.strip().startswith('-')]
                    for skill_line in skill_lines:
                        # Extract skill name from line like "- {'name': 'Python', ...}"
                        if "'name':" in skill_line:
                            skill_name = skill_line.split("'name': '")[1].split("'")[0] if "'name': '" in skill_line else ''
                            if skill_name:
                                all_skills.add(skill_name.lower())
                elif isinstance(job_data, dict) and 'SKILLS' in job_data:
                    skills = job_data['SKILLS']
                    if skills:
                        for skill in skills:
                            if isinstance(skill, dict) and 'name' in skill:
                                all_skills.add(skill['name'].lower())
        
        return len(all_skills)
    
    def _has_keyword(self, jobs_list: List, keyword: str) -> int:
        """Check if any job in sequence contains keyword in title/description"""
        if not self.jobs:
            return 0
        
        for job_id in jobs_list:
            job_id_str = str(job_id)
            if job_id_str in self.jobs:
                job_data = self.jobs[job_id_str]
                
                # Job data is a string, search directly in it
                if isinstance(job_data, str):
                    if keyword.lower() in job_data.lower():
                        return 1
                else:
                    # Fallback for dictionary format
                    # Check title
                    if 'TITLE' in job_data and job_data['TITLE']:
                        if keyword.lower() in str(job_data['TITLE']).lower():
                            return 1
                    
                    # Check summary
                    if 'SUMMARY' in job_data and job_data['SUMMARY']:
                        if keyword.lower() in str(job_data['SUMMARY']).lower():
                            return 1
        
        return 0
    
    @staticmethod
    def _get_job_last_positions(jobs_list: List) -> Dict:
        """Get the last position (recency) of each job in sequence"""
        last_positions = {}
        for idx, job_id in enumerate(jobs_list):
            last_positions[job_id] = idx
        return last_positions
    
    @staticmethod
    def _max_consecutive(jobs_list: List) -> int:
        """Maximum consecutive views of the same job"""
        if len(jobs_list) == 0:
            return 0
        
        max_count = 1
        current_count = 1
        
        for i in range(1, len(jobs_list)):
            if jobs_list[i] == jobs_list[i-1]:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 1
        
        return max_count
    
    @staticmethod
    def _avg_consecutive(jobs_list: List) -> float:
        """Average consecutive views of the same job"""
        if len(jobs_list) == 0:
            return 0.0
        
        consecutive_counts = []
        current_count = 1
        
        for i in range(1, len(jobs_list)):
            if jobs_list[i] == jobs_list[i-1]:
                current_count += 1
            else:
                consecutive_counts.append(current_count)
                current_count = 1
        
        consecutive_counts.append(current_count)
        
        return np.mean(consecutive_counts) if consecutive_counts else 0.0
