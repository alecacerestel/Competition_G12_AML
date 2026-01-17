"""
Deep Learning Training Script for Job Recommendation

This script implements a Transformer-based deep learning model for job recommendation
that optimizes MRR (Mean Reciprocal Rank) score.

Key Features:
- Transformer encoder for sequence modeling (better than LSTM for long sequences)
- Multi-head attention to capture job relationships
- Feature fusion from session behavior patterns
- Listwise ranking loss for direct MRR optimization
- Dual-task learning: job prediction + action prediction

Usage:
    python src/scripts/run_training.py

Author: Competition G12 AML
"""

import sys
import ast
import json
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "Data"

# Model hyperparameters
CONFIG = {
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_heads': 8,
    'num_layers': 3,
    'dropout': 0.2,
    'max_seq_len': 50,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'num_epochs': 1000,
    'top_k': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
}

print(f"Using device: {CONFIG['device']}")

# =============================================================================
# JOB CONTENT EMBEDDINGS
# =============================================================================

def load_job_content_embeddings(job_to_idx: dict, embedding_dim: int):
    """
    Load pre-computed job content embeddings from Sentence-BERT.
    
    These embeddings capture semantic information about job titles,
    descriptions, skills, and tasks - allowing the model to understand
    that similar jobs should be recommended together.
    
    Args:
        job_to_idx: Mapping from job_id to model index
        embedding_dim: Target embedding dimension for the model
        
    Returns:
        Tensor of shape (num_jobs + 1, embedding_dim) ready for nn.Embedding
    """
    embeddings_path = DATA_PATH / "job_content_embeddings.pt"
    mapping_path = DATA_PATH / "job_embedding_mapping.json"
    
    if not embeddings_path.exists() or not mapping_path.exists():
        print("  ⚠ Job content embeddings not found. Using random initialization.")
        print("    Run: python src/scripts/create_job_embeddings.py")
        return None
    
    print("  Loading pre-computed job content embeddings...")
    
    # Load embeddings and mapping
    data = torch.load(embeddings_path, weights_only=True)
    with open(mapping_path, 'r') as f:
        content_job_to_idx = json.load(f)
    
    content_embeddings = data['embeddings']  # (num_jobs, 384)
    content_dim = content_embeddings.shape[1]
    
    print(f"    Loaded {len(content_embeddings)} job embeddings (dim={content_dim})")
    
    # Project to target embedding dimension if needed
    if content_dim != embedding_dim:
        print(f"    Projecting from {content_dim} to {embedding_dim} dimensions...")
        # Use PCA-like random projection (fast, preserves distances)
        projection = torch.randn(content_dim, embedding_dim) / np.sqrt(content_dim)
        content_embeddings = content_embeddings @ projection
    
    # Create embedding matrix aligned with job_to_idx
    num_jobs = len(job_to_idx)
    aligned_embeddings = torch.zeros(num_jobs + 1, embedding_dim)  # +1 for padding
    
    matched = 0
    for job_id, model_idx in job_to_idx.items():
        # job_id might be int or string in content mapping
        content_idx = content_job_to_idx.get(str(job_id))
        if content_idx is None:
            content_idx = content_job_to_idx.get(job_id)
        
        if content_idx is not None:
            aligned_embeddings[model_idx] = content_embeddings[content_idx]
            matched += 1
    
    print(f"    Matched {matched}/{num_jobs} jobs ({100*matched/num_jobs:.1f}%)")
    
    # Normalize embeddings
    norms = aligned_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    aligned_embeddings = aligned_embeddings / norms
    
    return aligned_embeddings


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_data():
    """Load all data files"""
    print("Loading data...")
    
    # Load job listings
    with open(DATA_PATH / "job_listings.json", 'r', encoding='utf-8') as f:
        jobs = json.load(f)
    
    # Load training and test data
    x_train = pd.read_csv(DATA_PATH / "x_train_Meacfjr.csv")
    y_train = pd.read_csv(DATA_PATH / "y_train_SwJNMSu.csv")
    x_test = pd.read_csv(DATA_PATH / "x_test_jCBBNP2.csv")
    
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples: {len(x_test)}")
    print(f"  Total jobs: {len(jobs)}")
    
    return jobs, x_train, y_train, x_test


def parse_sequence(val):
    """Parse string representation of list to actual list"""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return ast.literal_eval(val)
    return []


def build_job_vocabulary(x_train, x_test, y_train):
    """Build vocabulary of all job IDs"""
    all_jobs = set()
    
    for df in [x_train, x_test]:
        for _, row in df.iterrows():
            jobs = parse_sequence(row['job_ids'])
            all_jobs.update(jobs)
    
    # Add target jobs
    all_jobs.update(y_train['job_id'].unique())
    
    # Create mappings (0 reserved for padding)
    job_to_idx = {job: idx + 1 for idx, job in enumerate(sorted(all_jobs))}
    idx_to_job = {idx: job for job, idx in job_to_idx.items()}
    
    print(f"  Vocabulary size: {len(job_to_idx)}")
    
    return job_to_idx, idx_to_job


def build_transition_matrix(x_train, y_train, job_to_idx, window_size=3):
    """Build job transition probabilities for hybrid scoring"""
    print("Building transition matrix...")
    
    transitions = defaultdict(lambda: defaultdict(float))
    job_counts = defaultdict(int)
    
    for idx, row in x_train.iterrows():
        jobs = parse_sequence(row['job_ids'])
        target_job = y_train.iloc[idx]['job_id']
        
        # Add target to sequence for learning
        jobs_with_target = jobs + [target_job]
        
        for job in jobs_with_target:
            job_counts[job] += 1
        
        for i in range(len(jobs_with_target)):
            current = jobs_with_target[i]
            for j in range(i + 1, min(i + window_size + 1, len(jobs_with_target))):
                next_job = jobs_with_target[j]
                weight = 1.0 / (j - i)
                transitions[current][next_job] += weight
    
    # Normalize
    for job in transitions:
        total = sum(transitions[job].values())
        for next_job in transitions[job]:
            transitions[job][next_job] /= total
    
    return dict(transitions), dict(job_counts)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_session_features(row, jobs_list, actions_list):
    """Extract comprehensive session features"""
    features = {}
    
    seq_len = len(jobs_list)
    features['seq_length'] = seq_len
    
    # Unique jobs ratio
    unique_jobs = len(set(jobs_list))
    features['unique_jobs_ratio'] = unique_jobs / max(seq_len, 1)
    
    # Most frequent job ratio
    if jobs_list:
        counter = Counter(jobs_list)
        most_common_count = counter.most_common(1)[0][1]
        features['most_frequent_ratio'] = most_common_count / seq_len
    else:
        features['most_frequent_ratio'] = 0
    
    # Job transitions
    if seq_len > 1:
        transitions = sum(1 for i in range(seq_len - 1) if jobs_list[i] != jobs_list[i + 1])
        features['transition_ratio'] = transitions / (seq_len - 1)
    else:
        features['transition_ratio'] = 0
    
    # Consecutive views
    max_consec = 1
    current_consec = 1
    for i in range(1, seq_len):
        if jobs_list[i] == jobs_list[i - 1]:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 1
    features['max_consecutive'] = max_consec
    
    # Action-based features
    apply_count = sum(1 for a in actions_list if a == 'apply')
    view_count = seq_len - apply_count
    
    features['apply_count'] = apply_count
    features['view_count'] = view_count
    features['apply_ratio'] = apply_count / max(seq_len, 1)
    
    # Action transitions
    if seq_len > 1:
        action_changes = sum(1 for i in range(seq_len - 1) if actions_list[i] != actions_list[i + 1])
        features['action_change_ratio'] = action_changes / (seq_len - 1)
    else:
        features['action_change_ratio'] = 0
    
    # First apply position
    first_apply = next((i for i, a in enumerate(actions_list) if a == 'apply'), -1)
    features['first_apply_position'] = first_apply / max(seq_len, 1) if first_apply >= 0 else -1
    
    # Last action encoding
    features['last_action_is_apply'] = 1 if actions_list and actions_list[-1] == 'apply' else 0
    
    # Applied jobs count (unique)
    applied_jobs = set(job for job, action in zip(jobs_list, actions_list) if action == 'apply')
    features['unique_applied_count'] = len(applied_jobs)
    
    return features


# =============================================================================
# FEATURE NORMALIZATION
# =============================================================================

class FeatureNormalizer:
    """
    Robust feature normalization using statistics learned from training data.
    
    Uses Z-score normalization (mean=0, std=1) which works best for neural networks.
    Handles missing values (like -1 for first_apply_position) separately.
    """
    
    FEATURE_NAMES = [
        'seq_length',
        'unique_jobs_ratio', 
        'most_frequent_ratio',
        'transition_ratio',
        'max_consecutive',
        'apply_ratio',
        'action_change_ratio',
        'first_apply_position',
        'last_action_is_apply',
        'unique_applied_count',
    ]
    
    def __init__(self):
        self.means = {}
        self.stds = {}
        self.mins = {}
        self.maxs = {}
        self.fitted = False
        
    def fit(self, x_df: pd.DataFrame):
        """Compute normalization statistics from training data"""
        print("Fitting feature normalizer...")
        
        # Collect all feature values
        all_features = {name: [] for name in self.FEATURE_NAMES}
        
        for _, row in x_df.iterrows():
            jobs_list = parse_sequence(row['job_ids'])
            actions_list = parse_sequence(row['actions'])
            features = extract_session_features(row, jobs_list, actions_list)
            
            for name in self.FEATURE_NAMES:
                val = features.get(name, 0)
                # Skip sentinel values (-1) for statistics
                if name == 'first_apply_position' and val == -1:
                    continue
                all_features[name].append(val)
        
        # Compute statistics
        for name in self.FEATURE_NAMES:
            values = np.array(all_features[name])
            if len(values) > 0:
                self.means[name] = np.mean(values)
                self.stds[name] = np.std(values) + 1e-8  # Avoid division by zero
                self.mins[name] = np.min(values)
                self.maxs[name] = np.max(values)
            else:
                self.means[name] = 0.0
                self.stds[name] = 1.0
                self.mins[name] = 0.0
                self.maxs[name] = 1.0
        
        self.fitted = True
        
        print("  Feature statistics:")
        for name in self.FEATURE_NAMES:
            print(f"    {name}: mean={self.means[name]:.3f}, std={self.stds[name]:.3f}, "
                  f"range=[{self.mins[name]:.2f}, {self.maxs[name]:.2f}]")
        
        return self
    
    def transform(self, features: dict) -> List[float]:
        """Transform features using learned statistics"""
        if not self.fitted:
            raise RuntimeError("FeatureNormalizer must be fitted before transform")
        
        normalized = []
        
        for name in self.FEATURE_NAMES:
            val = features.get(name, 0)
            
            # Handle special cases
            if name == 'first_apply_position':
                if val == -1:
                    # No apply in session - use a distinct normalized value
                    # Use -2 std below mean to indicate "missing"
                    normalized.append(-2.0)
                else:
                    normalized.append((val - self.means[name]) / self.stds[name])
            
            elif name == 'last_action_is_apply':
                # Binary feature - keep as is (0 or 1)
                normalized.append(float(val))
            
            else:
                # Z-score normalization
                normalized.append((val - self.means[name]) / self.stds[name])
        
        return normalized
    
    def save(self, path: str):
        """Save normalizer statistics"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'means': self.means,
                'stds': self.stds,
                'mins': self.mins,
                'maxs': self.maxs,
                'fitted': self.fitted
            }, f)
    
    def load(self, path: str):
        """Load normalizer statistics"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.means = data['means']
            self.stds = data['stds']
            self.mins = data['mins']
            self.maxs = data['maxs']
            self.fitted = data['fitted']
        return self


# =============================================================================
# DATASET CLASS
# =============================================================================

class JobRecommendationDataset(Dataset):
    """
    Dataset for training the job recommendation model.
    
    Each sample contains:
    - Job sequence (padded)
    - Action sequence (padded)
    - Session features (properly normalized)
    - Target job ID
    - Target action (for dual-task learning)
    - Negative samples for contrastive learning
    """
    
    def __init__(self, x_df, y_df, job_to_idx, all_job_indices, 
                 normalizer: FeatureNormalizer,
                 max_seq_len=50, num_negatives=10):
        self.x_df = x_df.reset_index(drop=True)
        self.y_df = y_df.reset_index(drop=True)
        self.job_to_idx = job_to_idx
        self.all_job_indices = list(all_job_indices)
        self.normalizer = normalizer
        self.max_seq_len = max_seq_len
        self.num_negatives = num_negatives
        
    def __len__(self):
        return len(self.x_df)
    
    def __getitem__(self, idx):
        row = self.x_df.iloc[idx]
        target_row = self.y_df.iloc[idx]
        
        # Parse sequences
        jobs_list = parse_sequence(row['job_ids'])
        actions_list = parse_sequence(row['actions'])
        
        # Convert to indices
        job_indices = [self.job_to_idx.get(j, 0) for j in jobs_list]
        action_encoded = [1 if a == 'apply' else 0 for a in actions_list]
        
        # Truncate/pad sequences
        seq_len = min(len(job_indices), self.max_seq_len)
        
        job_seq = job_indices[-self.max_seq_len:] + [0] * (self.max_seq_len - seq_len)
        action_seq = action_encoded[-self.max_seq_len:] + [0] * (self.max_seq_len - seq_len)
        
        # Create attention mask
        attn_mask = [1] * min(len(job_indices), self.max_seq_len) + [0] * (self.max_seq_len - seq_len)
        
        # Extract and normalize session features
        features = extract_session_features(row, jobs_list, actions_list)
        feature_vector = self.normalizer.transform(features)
        
        # Target
        target_job_id = target_row['job_id']
        target_job_idx = self.job_to_idx.get(target_job_id, 0)
        target_action = 1 if target_row['action'] == 'apply' else 0
        
        # Negative sampling (jobs not in session and not target)
        session_jobs_set = set(job_indices)
        session_jobs_set.add(target_job_idx)
        
        available_negatives = [j for j in self.all_job_indices if j not in session_jobs_set]
        if len(available_negatives) >= self.num_negatives:
            negatives = np.random.choice(available_negatives, self.num_negatives, replace=False)
        else:
            negatives = np.random.choice(self.all_job_indices, self.num_negatives, replace=True)
        
        return {
            'job_seq': torch.LongTensor(job_seq),
            'action_seq': torch.FloatTensor(action_seq),
            'attn_mask': torch.FloatTensor(attn_mask),
            'features': torch.FloatTensor(feature_vector),
            'seq_len': seq_len,
            'target_job': target_job_idx,
            'target_action': target_action,
            'negatives': torch.LongTensor(negatives),
            'session_jobs': torch.LongTensor(job_seq),  # Already padded to max_seq_len
        }


class TestDataset(Dataset):
    """Dataset for test/inference"""
    
    def __init__(self, x_df, job_to_idx, normalizer: FeatureNormalizer, max_seq_len=50):
        self.x_df = x_df.reset_index(drop=True)
        self.job_to_idx = job_to_idx
        self.normalizer = normalizer
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.x_df)
    
    def __getitem__(self, idx):
        row = self.x_df.iloc[idx]
        
        jobs_list = parse_sequence(row['job_ids'])
        actions_list = parse_sequence(row['actions'])
        
        job_indices = [self.job_to_idx.get(j, 0) for j in jobs_list]
        action_encoded = [1 if a == 'apply' else 0 for a in actions_list]
        
        seq_len = min(len(job_indices), self.max_seq_len)
        
        job_seq = job_indices[-self.max_seq_len:] + [0] * (self.max_seq_len - seq_len)
        action_seq = action_encoded[-self.max_seq_len:] + [0] * (self.max_seq_len - seq_len)
        attn_mask = [1] * min(len(job_indices), self.max_seq_len) + [0] * (self.max_seq_len - seq_len)
        
        # Extract and normalize session features
        features = extract_session_features(row, jobs_list, actions_list)
        feature_vector = self.normalizer.transform(features)
        
        return {
            'job_seq': torch.LongTensor(job_seq),
            'action_seq': torch.FloatTensor(action_seq),
            'attn_mask': torch.FloatTensor(attn_mask),
            'features': torch.FloatTensor(feature_vector),
            'seq_len': seq_len,
            'session_jobs': torch.LongTensor(job_seq),  # Already padded to max_seq_len
            'session_id': row['session_id'],
        }


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class JobTransformerEncoder(nn.Module):
    """
    Transformer-based encoder for job sequences.
    
    This captures:
    - Sequential patterns in job viewing
    - Attention between different jobs
    - Position-aware encoding
    
    Can be initialized with pre-trained job content embeddings for better
    semantic understanding of job relationships.
    """
    
    def __init__(self, num_jobs, embedding_dim, hidden_dim, num_heads, 
                 num_layers, dropout, max_seq_len, pretrained_embeddings=None):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Job embedding layer (0 is padding)
        self.job_embedding = nn.Embedding(num_jobs + 1, embedding_dim, padding_idx=0)
        
        # Initialize with pre-trained embeddings if provided
        if pretrained_embeddings is not None:
            self.job_embedding.weight.data.copy_(pretrained_embeddings)
            print("    ✓ Encoder initialized with pre-trained job embeddings")
        
        # Action embedding (view/apply)
        self.action_embedding = nn.Linear(1, embedding_dim // 4)
        
        # Combine job + action embeddings
        self.input_projection = nn.Linear(embedding_dim + embedding_dim // 4, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, job_seq, action_seq, attn_mask):
        """
        Args:
            job_seq: (batch, seq_len) - Job IDs
            action_seq: (batch, seq_len) - Action indicators
            attn_mask: (batch, seq_len) - Attention mask (1 for valid, 0 for padding)
        """
        batch_size, seq_len = job_seq.shape
        
        # Get embeddings
        job_emb = self.job_embedding(job_seq)  # (batch, seq, emb_dim)
        action_emb = self.action_embedding(action_seq.unsqueeze(-1))  # (batch, seq, emb_dim/4)
        
        # Combine embeddings
        combined = torch.cat([job_emb, action_emb], dim=-1)
        x = self.input_projection(combined)  # (batch, seq, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create transformer mask (True = ignore)
        key_padding_mask = (attn_mask == 0)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.layer_norm(x)
        
        return x, attn_mask


class FeatureFusion(nn.Module):
    """Fuses transformer output with session-level features"""
    
    def __init__(self, hidden_dim, feature_dim, output_dim):
        super().__init__()
        
        # Process session features
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )
        
        # Attention pooling for sequence
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 64, output_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, seq_output, attn_mask, features):
        """
        Args:
            seq_output: (batch, seq_len, hidden_dim)
            attn_mask: (batch, seq_len)
            features: (batch, feature_dim)
        """
        # Attention-weighted pooling
        attn_weights = self.attention_pool(seq_output)  # (batch, seq, 1)
        attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        pooled = (seq_output * attn_weights).sum(dim=1)  # (batch, hidden_dim)
        
        # Process features
        feat_repr = self.feature_mlp(features)  # (batch, 64)
        
        # Fuse
        combined = torch.cat([pooled, feat_repr], dim=-1)
        output = self.fusion(combined)
        
        return F.normalize(output, p=2, dim=1)


class JobRanker(nn.Module):
    """
    Main model: Ranks candidate jobs for a given session.
    
    Architecture:
    1. JobTransformerEncoder: Encodes job sequence
    2. FeatureFusion: Combines sequence representation with session features
    3. Scoring head: Computes relevance scores for candidate jobs
    4. Action head: Predicts view/apply action
    
    Can leverage pre-trained job content embeddings for better semantic
    understanding of job similarities based on titles, descriptions, skills.
    """
    
    def __init__(self, num_jobs, config, pretrained_embeddings=None):
        super().__init__()
        
        self.num_jobs = num_jobs
        
        # Sequence encoder (with optional pre-trained embeddings)
        self.encoder = JobTransformerEncoder(
            num_jobs=num_jobs,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
            pretrained_embeddings=pretrained_embeddings
        )
        
        # Feature fusion
        self.fusion = FeatureFusion(
            hidden_dim=config['hidden_dim'],
            feature_dim=10,  # Number of engineered features
            output_dim=config['embedding_dim']
        )
        
        # Job embedding for scoring (initialized with same pre-trained embeddings)
        self.job_scorer_embedding = nn.Embedding(num_jobs + 1, config['embedding_dim'], padding_idx=0)
        
        # Initialize scorer embeddings with pre-trained if available
        if pretrained_embeddings is not None:
            self.job_scorer_embedding.weight.data.copy_(pretrained_embeddings)
            print("    ✓ Scorer initialized with pre-trained job embeddings")
        
        # Scoring MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(config['embedding_dim'] * 2, config['embedding_dim']),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['embedding_dim'], 1)
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(config['embedding_dim'], 64),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def get_session_embedding(self, job_seq, action_seq, attn_mask, features):
        """Get session representation"""
        seq_output, mask = self.encoder(job_seq, action_seq, attn_mask)
        session_emb = self.fusion(seq_output, mask, features)
        return session_emb
    
    def score_jobs(self, session_emb, job_ids):
        """
        Score candidate jobs for a session
        
        Args:
            session_emb: (batch, emb_dim) - Session embeddings
            job_ids: (batch, num_candidates) - Candidate job IDs
        
        Returns:
            scores: (batch, num_candidates)
        """
        # Get job embeddings
        job_emb = self.job_scorer_embedding(job_ids)  # (batch, num_cand, emb_dim)
        
        # Expand session embedding
        session_exp = session_emb.unsqueeze(1).expand(-1, job_ids.size(1), -1)
        
        # Concatenate and score
        combined = torch.cat([session_exp, job_emb], dim=-1)
        scores = self.score_mlp(combined).squeeze(-1)
        
        return scores
    
    def forward(self, job_seq, action_seq, attn_mask, features, 
                target_job, negatives):
        """
        Forward pass for training
        
        Returns:
            loss: Combined ranking + action loss
            metrics: Dict with individual loss components
        """
        # Get session embedding
        session_emb = self.get_session_embedding(job_seq, action_seq, attn_mask, features)
        
        # Prepare candidates: [positive, negatives]
        batch_size = job_seq.size(0)
        
        # Score positive
        pos_job = target_job.unsqueeze(1)  # (batch, 1)
        candidates = torch.cat([pos_job, negatives], dim=1)  # (batch, 1 + num_neg)
        
        scores = self.score_jobs(session_emb, candidates)  # (batch, 1 + num_neg)
        
        # Listwise ranking loss (positive should be ranked first)
        # Use listwise softmax cross-entropy
        labels = torch.zeros(batch_size, dtype=torch.long, device=scores.device)  # Positive is index 0
        ranking_loss = F.cross_entropy(scores, labels)
        
        # Action prediction loss
        action_pred = self.action_head(session_emb).squeeze(-1)
        
        return {
            'ranking_loss': ranking_loss,
            'scores': scores,
            'action_pred': action_pred,
            'session_emb': session_emb
        }
    
    def predict(self, job_seq, action_seq, attn_mask, features, candidate_jobs, 
                session_jobs=None, top_k=10):
        """
        Predict top-k jobs for inference
        
        Args:
            candidate_jobs: Tensor of all possible job indices
            session_jobs: Jobs already in session (to exclude from recommendations)
        """
        self.eval()
        
        with torch.no_grad():
            # Get session embedding
            session_emb = self.get_session_embedding(job_seq, action_seq, attn_mask, features)
            
            batch_size = job_seq.size(0)
            num_candidates = len(candidate_jobs)
            
            # Score all candidates
            candidates = candidate_jobs.unsqueeze(0).expand(batch_size, -1)
            scores = self.score_jobs(session_emb, candidates)  # (batch, num_candidates)
            
            # Get top-k per batch
            _, top_indices = torch.topk(scores, min(top_k * 2, num_candidates), dim=1)
            
            # Action prediction
            action_pred = self.action_head(session_emb).squeeze(-1)
            
        return top_indices, action_pred


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def calculate_mrr(y_true_ids, y_pred_top10):
    """Calculate Mean Reciprocal Rank"""
    rr_scores = []
    
    for true_id, pred_list in zip(y_true_ids, y_pred_top10):
        if true_id in pred_list:
            rank = pred_list.index(true_id) + 1
            rr_scores.append(1.0 / rank)
        else:
            rr_scores.append(0.0)
    
    return np.mean(rr_scores)


def calculate_action_accuracy(y_true_actions, y_pred_actions):
    """Calculate action prediction accuracy"""
    correct = sum(1 for t, p in zip(y_true_actions, y_pred_actions) if t == p)
    return correct / len(y_true_actions)


def calculate_final_score(mrr_score, action_accuracy):
    """Calculate competition final score"""
    return 0.7 * mrr_score + 0.3 * action_accuracy


class Trainer:
    """Training manager for the job recommendation model"""
    
    def __init__(self, model, device, idx_to_job, transitions=None, job_counts=None):
        self.model = model.to(device)
        self.device = device
        self.idx_to_job = idx_to_job
        self.transitions = transitions or {}
        self.job_counts = job_counts or {}
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2
        )
        
        self.best_mrr = 0
        self.history = []
        self.metrics_file = None
    
    def get_lr(self):
        """Get current learning rate from optimizer"""
        return self.optimizer.param_groups[0]['lr']
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with detailed progress tracking"""
        self.model.train()
        total_loss = 0
        total_rank_loss = 0
        total_action_loss = 0
        num_batches = len(train_loader)
        
        import time
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()
            
            # Move to device
            job_seq = batch['job_seq'].to(self.device)
            action_seq = batch['action_seq'].to(self.device)
            attn_mask = batch['attn_mask'].to(self.device)
            features = batch['features'].to(self.device)
            target_job = batch['target_job'].to(self.device)
            target_action = batch['target_action'].float().to(self.device)
            negatives = batch['negatives'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(job_seq, action_seq, attn_mask, features, 
                                target_job, negatives)
            
            # Combined loss
            ranking_loss = outputs['ranking_loss']
            action_loss = F.binary_cross_entropy(outputs['action_pred'], target_action)
            
            # Weight losses (focus on ranking for MRR optimization)
            loss = 0.99 * ranking_loss + 0.01 * action_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_rank_loss += ranking_loss.item()
            total_action_loss += action_loss.item()
            
            # Progress bar style output every 20 batches
            if batch_idx % 20 == 0 or batch_idx == num_batches - 1:
                progress = (batch_idx + 1) / num_batches * 100
                elapsed = time.time() - epoch_start
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                
                avg_loss = total_loss / (batch_idx + 1)
                avg_rank = total_rank_loss / (batch_idx + 1)
                avg_action = total_action_loss / (batch_idx + 1)
                
                print(f"\r  [{batch_idx+1:3d}/{num_batches}] {progress:5.1f}% | "
                      f"Loss: {avg_loss:.4f} (Rank: {avg_rank:.4f}, Act: {avg_action:.4f}) | "
                      f"ETA: {eta:.0f}s", end="", flush=True)
        
        print()  # New line after progress
        
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        avg_rank_loss = total_rank_loss / num_batches
        avg_action_loss = total_action_loss / num_batches
        epoch_time = time.time() - epoch_start
        
        return {
            'loss': avg_loss,
            'rank_loss': avg_rank_loss,
            'action_loss': avg_action_loss,
            'time': epoch_time
        }
    
    def _save_metrics_to_file(self):
        """Save training metrics to CSV file after each epoch"""
        if self.metrics_file is None:
            return
        
        import csv
        
        # Define columns
        columns = [
            'epoch', 'learning_rate', 'train_loss', 'train_rank_loss', 
            'train_action_loss', 'val_mrr', 'val_action_accuracy', 
            'val_final_score', 'epoch_time_seconds', 'is_best'
        ]
        
        # Write to CSV
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)
    
    def evaluate(self, val_loader, y_val, all_job_indices, job_to_idx):
        """Evaluate model on validation set"""
        self.model.eval()
        
        all_predictions = []
        all_action_preds = []
        
        # Convert all job indices to tensor
        candidate_jobs = torch.LongTensor(all_job_indices).to(self.device)
        
        with torch.no_grad():
            for batch in val_loader:
                job_seq = batch['job_seq'].to(self.device)
                action_seq = batch['action_seq'].to(self.device)
                attn_mask = batch['attn_mask'].to(self.device)
                features = batch['features'].to(self.device)
                session_jobs = batch['session_jobs']
                
                # Get predictions
                top_indices, action_pred = self.model.predict(
                    job_seq, action_seq, attn_mask, features,
                    candidate_jobs, session_jobs, top_k=CONFIG['top_k']
                )
                
                # Convert indices to job IDs
                for i, indices in enumerate(top_indices):
                    session_job_set = set(session_jobs[i].tolist())
                    
                    top_jobs = []
                    for idx in indices:
                        job_idx = candidate_jobs[idx].item()
                        if job_idx not in session_job_set and job_idx in self.idx_to_job:
                            top_jobs.append(self.idx_to_job[job_idx])
                        if len(top_jobs) >= CONFIG['top_k']:
                            break
                    
                    # Pad if needed
                    while len(top_jobs) < CONFIG['top_k']:
                        top_jobs.append(top_jobs[-1] if top_jobs else 0)
                    
                    all_predictions.append(top_jobs[:CONFIG['top_k']])
                
                # Action predictions
                action_preds = (action_pred > 0.5).int().cpu().tolist()
                all_action_preds.extend(action_preds)
        
        # Calculate metrics
        y_true_jobs = y_val['job_id'].tolist()
        y_true_actions = [1 if a == 'apply' else 0 for a in y_val['action'].tolist()]
        
        mrr = calculate_mrr(y_true_jobs, all_predictions)
        action_acc = calculate_action_accuracy(y_true_actions, all_action_preds)
        final_score = calculate_final_score(mrr, action_acc)
        
        return {
            'mrr': mrr,
            'action_accuracy': action_acc,
            'final_score': final_score,
            'predictions': all_predictions
        }
    
    def fit(self, train_loader, val_loader, y_val, all_job_indices, job_to_idx, 
            num_epochs, save_path=None, metrics_path=None):
        """Full training loop with detailed progress tracking"""
        
        import time
        
        # Set up metrics file
        self.metrics_file = metrics_path
        
        print("\n" + "=" * 80)
        print("Starting Training")
        print("=" * 80)
        print(f"{'Epoch':^7} | {'LR':^10} | {'Train Loss':^12} | {'MRR':^8} | {'Act Acc':^8} | {'Final':^8} | {'Time':^6}")
        print("-" * 80)
        
        total_start = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Get current learning rate
            current_lr = self.get_lr()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            print("  Evaluating...", end="\r")
            val_metrics = self.evaluate(val_loader, y_val, all_job_indices, job_to_idx)
            
            epoch_time = time.time() - epoch_start
            
            # Determine if this is the best model
            is_best = val_metrics['mrr'] > self.best_mrr
            best_marker = " ★" if is_best else ""
            
            # Print epoch summary in table format
            print(f"{epoch+1:^7} | {current_lr:^10.2e} | {train_metrics['loss']:^12.4f} | {val_metrics['mrr']:^8.4f} | "
                  f"{val_metrics['action_accuracy']:^8.4f} | {val_metrics['final_score']:^8.4f} | "
                  f"{epoch_time:^5.0f}s{best_marker}")
            
            self.history.append({
                'epoch': epoch + 1,
                'learning_rate': current_lr,
                'train_loss': train_metrics['loss'],
                'train_rank_loss': train_metrics['rank_loss'],
                'train_action_loss': train_metrics['action_loss'],
                'val_mrr': val_metrics['mrr'],
                'val_action_accuracy': val_metrics['action_accuracy'],
                'val_final_score': val_metrics['final_score'],
                'epoch_time_seconds': epoch_time,
                'is_best': is_best
            })
            
            # Save metrics to CSV after each epoch
            self._save_metrics_to_file()
            
            # Save best model
            if is_best:
                self.best_mrr = val_metrics['mrr']
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'mrr': val_metrics['mrr'],
                        'action_accuracy': val_metrics['action_accuracy'],
                        'final_score': val_metrics['final_score'],
                    }, save_path)
            
            # Early stopping check - print warning if no improvement for 10 epochs
            if epoch > 10:
                recent_mrrs = [h['val_mrr'] for h in self.history[-10:]]
                if max(recent_mrrs) <= self.history[-11]['val_mrr']:
                    print("  ⚠ No improvement in last 10 epochs")
        
        total_time = time.time() - total_start
        
        print("-" * 80)
        print(f"\n✓ Training Complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best MRR: {self.best_mrr:.4f}")
        
        # Find best epoch
        best_epoch = max(self.history, key=lambda x: x['val_mrr'])
        print(f"  Best epoch: {best_epoch['epoch']}")
        print(f"  Best Final Score: {best_epoch['val_final_score']:.4f}")
        
        if self.metrics_file:
            print(f"  Metrics saved to: {self.metrics_file}")
        
        return self.history


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main training pipeline"""
    
    # Set random seeds
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Load data
    jobs, x_train, y_train, x_test = load_data()
    
    # Build vocabulary
    job_to_idx, idx_to_job = build_job_vocabulary(x_train, x_test, y_train)
    all_job_indices = list(job_to_idx.values())
    
    # Build transition matrix
    transitions, job_counts = build_transition_matrix(x_train, y_train, job_to_idx)
    
    # Split training data
    print("\nSplitting data for validation...")
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, 
        test_size=0.15, 
        random_state=CONFIG['seed']
    )
    print(f"  Training: {len(x_tr)}, Validation: {len(x_val)}")
    
    # Fit feature normalizer on training data only (avoid data leakage)
    print("\nFitting feature normalizer...")
    normalizer = FeatureNormalizer()
    normalizer.fit(x_tr)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = JobRecommendationDataset(
        x_tr, y_tr, job_to_idx, all_job_indices,
        normalizer=normalizer,
        max_seq_len=CONFIG['max_seq_len'],
        num_negatives=10
    )
    
    val_dataset = TestDataset(
        x_val, job_to_idx,
        normalizer=normalizer,
        max_seq_len=CONFIG['max_seq_len']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Load pre-trained job content embeddings
    print("\nLoading job content embeddings...")
    pretrained_embeddings = load_job_content_embeddings(job_to_idx, CONFIG['embedding_dim'])
    
    # Initialize model with pre-trained embeddings
    print("\nInitializing model...")
    num_jobs = len(job_to_idx)
    model = JobRanker(num_jobs, CONFIG, pretrained_embeddings=pretrained_embeddings)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = Trainer(
        model, 
        CONFIG['device'], 
        idx_to_job,
        transitions,
        job_counts
    )
    
    # Set up paths for saving
    save_path = PROJECT_ROOT / "experiments" / "best_transformer_model.pt"
    metrics_path = PROJECT_ROOT / "experiments" / "training_metrics.csv"
    
    # Ensure experiments directory exists
    (PROJECT_ROOT / "experiments").mkdir(parents=True, exist_ok=True)
    
    # Train
    history = trainer.fit(
        train_loader, val_loader, y_val,
        all_job_indices, job_to_idx,
        num_epochs=CONFIG['num_epochs'],
        save_path=str(save_path),
        metrics_path=str(metrics_path)
    )
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Training History Summary (Last 10 epochs)")
    print("=" * 80)
    
    for h in history[-10:]:
        print(f"Epoch {h['epoch']:3d}: LR={h['learning_rate']:.2e}, Loss={h['train_loss']:.4f}, "
              f"MRR={h['val_mrr']:.4f}, Action Acc={h['val_action_accuracy']:.4f}, "
              f"Final={h['val_final_score']:.4f}")
    
    # Best results
    best_epoch = max(history, key=lambda x: x['val_mrr'])
    print(f"\n★ Best Epoch: {best_epoch['epoch']}")
    print(f"  MRR: {best_epoch['val_mrr']:.4f}")
    print(f"  Action Accuracy: {best_epoch['val_action_accuracy']:.4f}")
    print(f"  Final Score: {best_epoch['val_final_score']:.4f}")
    
    return model, trainer, history


if __name__ == "__main__":
    model, trainer, history = main()
