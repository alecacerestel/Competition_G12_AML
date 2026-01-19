import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class CollaborativeFilteringRecommender:
    """
    Collaborative filtering model trainer for job recommendations.
    
    Uses user-item interaction matrix and cosine similarity
    to predict the top 10 recommended jobs.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        pass
    
    def create_model(self):
        """Create and configure the model."""
        pass
    
    @staticmethod
    def build_interaction_matrix(df_x, df_y):
        """
        Build user-item interaction matrix (session-job)
        with weights based on interaction frequency.
        
        Args:
            df_x: DataFrame with jobs viewed per session
            df_y: DataFrame with target jobs per session
            
        Returns:
            tuple: (R, session_cats, job_cats, interaction_counts)
                - R: sparse interaction matrix
                - session_cats: session categories
                - job_cats: job categories
                - interaction_counts: interaction counts
        """
        # Expand jobs for each session
        exploded_jobs = df_x.explode('jobs_list')[['session_id', 'jobs_list']].copy()
        exploded_jobs.columns = ['session_id', 'job_id']
        
        # Add target jobs
        target_jobs = df_y[['session_id', 'job_id']].copy()
        
        # Combine all interactions
        all_interactions = pd.concat([exploded_jobs, target_jobs], ignore_index=True)
        
        # Count interaction frequency (weighting)
        interaction_counts = all_interactions.groupby(['session_id', 'job_id']).size().reset_index(name='weight')
        
        print(f"Total unique interactions: {len(interaction_counts)}")
        print(f"Frequency range: {interaction_counts['weight'].min()} to {interaction_counts['weight'].max()}")
        
        # Create ordered categories
        session_cats = pd.Categorical(interaction_counts['session_id'])
        job_cats = pd.Categorical(interaction_counts['job_id'])
        
        # Build sparse matrix with weights
        R = csr_matrix(
            (interaction_counts['weight'].values, 
             (session_cats.codes, job_cats.codes)),
            shape=(len(session_cats.categories), len(job_cats.categories))
        )
        
        sparsity = R.nnz / (R.shape[0] * R.shape[1]) * 100
        print(f"Interaction matrix: {R.shape}")
        print(f"Sparsity: {sparsity:.2f}%")
        
        return R, session_cats.categories, job_cats.categories, interaction_counts

    @staticmethod
    def get_similar_users(test_session_vector, R_train, k=50):
        """
        Find k most similar users using cosine similarity.
        
        Args:
            test_session_vector: sparse matrix (1, n_items) with viewed jobs
            R_train: training interaction matrix
            k: number of similar users to return
            
        Returns:
            tuple: (top_k_indices, similarities)
        """
        if not isinstance(test_session_vector, csr_matrix):
            raise ValueError("test_session_vector must be a sparse matrix")
        
        similarities = cosine_similarity(test_session_vector, R_train).flatten()
        
        # Filter positive similarities (users with overlap)
        valid_indices = np.where(similarities > 0)[0]
        
        if len(valid_indices) == 0:
            return np.array([]), np.array([])
        
        # Get top k
        top_k = min(k, len(valid_indices))
        top_k_indices = similarities[valid_indices].argsort()[-top_k:][::-1]
        
        return valid_indices[top_k_indices], similarities[valid_indices[top_k_indices]]

    @staticmethod
    def predict_next_step(test_session_vector, R_train, job_categories, k=50, theta=0.5):
        """
        Predict top 10 recommended jobs and user application likelihood.
        
        Args:
            test_session_vector: sparse matrix (1, n_items) with viewed jobs
            R_train: training interaction matrix
            job_categories: job category indices
            k: number of similar users (default: 50)
            theta: application prediction threshold (default: 0.5)
            
        Returns:
            tuple: (top_10_job_ids, applies_for, p_c_average)
                - top_10_job_ids: list of recommended job IDs
                - applies_for: 1 if user will apply, 0 otherwise
                - p_c_average: average score for top 10 jobs
        """
        # 1. Find similar users
        top_k_idx, sims = CollaborativeFilteringRecommender.get_similar_users(
            test_session_vector, R_train, k
        )
        
        if len(top_k_idx) == 0:
            return CollaborativeFilteringRecommender._get_popular_jobs(R_train, job_categories)
        
        # 2. Calculate Pcj: weighted average of similar users' interactions
        relevant_users_matrix = R_train[top_k_idx]
        weighted_matrix = relevant_users_matrix.multiply(sims.reshape(-1, 1))
        pcj_scores = np.asarray(weighted_matrix.sum(axis=0)).flatten() / sims.sum()
        
        # 3. Get Top 10 Jobs (excluding already viewed)
        top_10_job_ids, p_c_average = CollaborativeFilteringRecommender._get_top_10_jobs(
            pcj_scores, test_session_vector, job_categories
        )
        
        # 4. Calculate application intention
        applies_for = 1 if p_c_average >= theta else 0
        
        return top_10_job_ids, applies_for, p_c_average
    
    @staticmethod
    def _get_popular_jobs(R_train, job_categories):
        """Returns popular jobs when no similar users are found."""
        print("No similar users found, returning popular jobs")
        all_scores = np.asarray(R_train.mean(axis=0)).flatten()
        top_10_indices = all_scores.argsort()[-10:][::-1]
        top_10_job_ids = [
            job_categories[i] for i in top_10_indices 
            if i < len(job_categories)
        ]
        return top_10_job_ids[:10], 0, 0
    
    @staticmethod
    def _get_top_10_jobs(pcj_scores, test_session_vector, job_categories):
        """Extract top 10 jobs excluding already viewed ones."""
        already_seen = test_session_vector.indices
        pcj_scores[already_seen] = -1
        
        valid_indices = np.where(pcj_scores > -1)[0]
        if len(valid_indices) == 0:
            print("All jobs have already been viewed")
            return [], 0
        
        top_10_count = min(10, len(valid_indices))
        top_10_indices = pcj_scores.argsort()[-top_10_count:][::-1]
        
        top_10_job_ids = [
            job_categories[idx] for idx in top_10_indices
            if idx < len(job_categories) and pcj_scores[idx] >= 0
        ]
        
        top_10_scores = pcj_scores[top_10_indices]
        p_c_average = np.mean(top_10_scores)
        
        return top_10_job_ids, p_c_average


class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
        self.tfidf_matrix = None
        self.job_ids = None

    def fit(self, job_df):
        # Combinar title y summary para más contexto
        job_df['combined'] = job_df['title'].fillna('') + ' ' + job_df['summary'].fillna('')
        self.tfidf_matrix = self.vectorizer.fit_transform(job_df['combined'])
        self.job_ids = job_df['job_id'].astype(int).values
        self.job_index = {idx: i for i, idx in enumerate(self.job_ids)}

    def get_recommendations(self, viewed_job_ids, top_n=10):
        # 1. Obtenemos los índices de los trabajos vistos
        indices = [self.job_index[jid] for jid in viewed_job_ids if jid in self.job_index]
        
        if not indices:
            return [], 0.0

        # 2. Creamos el perfil del usuario promediando sus vistas
        user_profile = np.asarray(self.tfidf_matrix[indices].mean(axis=0))
        
        # 3. Calculamos similitud de este perfil contra todos los trabajos
        # linear_kernel es más rápido que cosine_similarity para TF-IDF
        cosine_sim = linear_kernel(user_profile, self.tfidf_matrix).flatten()
        
        # 4. Filtramos los que ya vio
        for idx in indices:
            cosine_sim[idx] = -1
            
        # 5. Top N
        related_indices = cosine_sim.argsort()[-top_n:][::-1]
        return [self.job_ids[i] for i in related_indices], np.mean(cosine_sim[related_indices])


class HybridRecommender:
    def __init__(self, cf_recommender, cb_recommender):
        self.cf = cf_recommender
        self.cb = cb_recommender

    def predict_next_step(self, test_session_vector, R_train, job_categories, viewed_job_ids, k=50, theta=0.5):
        """
        Hybrid prediction: Use CF if similar users exist, otherwise use CB.
        
        Args:
            test_session_vector: sparse matrix (1, n_items) with viewed jobs
            R_train: training interaction matrix
            job_categories: job category indices
            viewed_job_ids: list of job IDs already viewed
            k: number of similar users (default: 50)
            theta: application prediction threshold (default: 0.5)
            
        Returns:
            tuple: (top_10_job_ids, applies_for, p_c_average)
        """
        # Try CF first
        top_k_idx, sims = self.cf.get_similar_users(test_session_vector, R_train, k)
        
        if len(top_k_idx) > 5:
            # Use CF
            return self.cf.predict_next_step(
                test_session_vector, R_train, job_categories, k, theta
            )
        else:
            # print("No similar users found in CF, switching to CB")
            # Use CB
            recommendations, avg_score = self.cb.get_recommendations(viewed_job_ids, top_n=10)
            # For CB, we don't have applies_for prediction, so default to 0 or based on avg_score
            applies_for = 1 if avg_score >= theta else 0
        
        return recommendations, applies_for, avg_score