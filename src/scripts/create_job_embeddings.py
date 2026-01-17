"""
Create Job Content Embeddings using Sentence-BERT

This script generates semantic embeddings for each job based on:
- Title
- Summary/Description
- Skills
- Tasks

These embeddings capture the semantic meaning of job content,
allowing the model to understand that similar jobs should be recommended together,
even if they haven't been co-viewed in the training data.

Usage:
    python src/scripts/create_job_embeddings.py

Output:
    Data/job_embeddings.pt - PyTorch tensor of job embeddings
    Data/job_embedding_mapping.json - Mapping of job_id to embedding index
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "Data"


def parse_job_text(job_text: str) -> dict:
    """
    Parse the job text into structured components.
    
    Args:
        job_text: Raw text from job_listings.json
        
    Returns:
        Dict with title, summary, skills, tasks
    """
    result = {
        'title': '',
        'summary': '',
        'skills': [],
        'tasks': []
    }
    
    # Extract title
    title_match = re.search(r'TITLE\n(.+?)(?=\n\nSUMMARY|\n\nSECTION|\Z)', job_text, re.DOTALL)
    if title_match:
        result['title'] = title_match.group(1).strip()
    
    # Extract summary
    summary_match = re.search(r'SUMMARY\n(.+?)(?=\n\nSECTION|\n\nSKILLS|\n\nTASKS|\Z)', job_text, re.DOTALL)
    if summary_match:
        result['summary'] = summary_match.group(1).strip()[:1000]  # Limit length
    
    # Extract skills
    skills_match = re.search(r'SKILLS\n(.+?)(?=\n\nCOURSES|\n\nTASKS|\Z)', job_text, re.DOTALL)
    if skills_match:
        skills_text = skills_match.group(1)
        # Parse skill names from the dict-like strings
        skill_names = re.findall(r"'name':\s*'([^']+)'", skills_text)
        result['skills'] = list(set(skill_names))[:20]  # Unique skills, limit to 20
    
    # Extract tasks
    tasks_match = re.search(r'TASKS\n(.+?)(?=\Z)', job_text, re.DOTALL)
    if tasks_match:
        tasks_text = tasks_match.group(1)
        task_names = re.findall(r"'name':\s*'([^']+)'", tasks_text)
        result['tasks'] = list(set(task_names))[:10]  # Unique tasks, limit to 10
    
    return result


def create_job_embedding_text(parsed_job: dict) -> str:
    """
    Create a single text string for embedding from parsed job components.
    
    Uses a structured format that works well with sentence transformers.
    """
    parts = []
    
    if parsed_job['title']:
        parts.append(f"Job Title: {parsed_job['title']}")
    
    if parsed_job['summary']:
        # Take first 500 chars of summary
        summary = parsed_job['summary'][:500]
        parts.append(f"Description: {summary}")
    
    if parsed_job['skills']:
        skills_str = ", ".join(parsed_job['skills'][:15])
        parts.append(f"Required Skills: {skills_str}")
    
    if parsed_job['tasks']:
        tasks_str = "; ".join(parsed_job['tasks'][:8])
        parts.append(f"Main Tasks: {tasks_str}")
    
    return " | ".join(parts) if parts else "Unknown job"


def main():
    print("=" * 60)
    print("Creating Job Content Embeddings")
    print("=" * 60)
    
    # Load job listings
    print("\n1. Loading job listings...")
    with open(DATA_PATH / "job_listings.json", 'r', encoding='utf-8') as f:
        jobs = json.load(f)
    print(f"   Loaded {len(jobs)} jobs")
    
    # Parse jobs and create texts for embedding
    print("\n2. Parsing job content...")
    job_texts = {}
    for job_id, job_text in tqdm(jobs.items(), desc="   Parsing"):
        parsed = parse_job_text(job_text)
        embedding_text = create_job_embedding_text(parsed)
        job_texts[job_id] = embedding_text
    
    # Show sample
    sample_id = list(job_texts.keys())[0]
    print(f"\n   Sample job text for embedding:")
    print(f"   Job ID: {sample_id}")
    print(f"   Text: {job_texts[sample_id][:200]}...")
    
    # Load sentence transformer model
    print("\n3. Loading sentence transformer model...")
    from sentence_transformers import SentenceTransformer
    
    # Using all-MiniLM-L6-v2: fast, good quality, 384 dimensions
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"   Model loaded: all-MiniLM-L6-v2")
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Create embeddings in batches
    print("\n4. Creating embeddings...")
    job_ids = list(job_texts.keys())
    texts = [job_texts[jid] for jid in job_ids]
    
    # Batch encoding for efficiency
    embeddings = model.encode(
        texts, 
        batch_size=64, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"   Created embeddings with shape: {embeddings.shape}")
    
    # Create mapping from job_id to index
    job_id_to_idx = {job_id: idx for idx, job_id in enumerate(job_ids)}
    
    # Save embeddings as PyTorch tensor
    print("\n5. Saving embeddings...")
    embeddings_tensor = torch.from_numpy(embeddings).float()
    torch.save({
        'embeddings': embeddings_tensor,
        'job_ids': job_ids,
        'embedding_dim': embeddings_tensor.shape[1]
    }, DATA_PATH / "job_content_embeddings.pt")
    print(f"   Saved: {DATA_PATH / 'job_content_embeddings.pt'}")
    
    # Save mapping
    with open(DATA_PATH / "job_embedding_mapping.json", 'w') as f:
        json.dump(job_id_to_idx, f)
    print(f"   Saved: {DATA_PATH / 'job_embedding_mapping.json'}")
    
    # Verify with similarity check
    print("\n6. Verification - Similarity Check")
    print("   Finding similar jobs to the first job...")
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    first_job_emb = embeddings[0:1]
    similarities = cosine_similarity(first_job_emb, embeddings)[0]
    top_similar = np.argsort(similarities)[-6:-1][::-1]  # Top 5 (excluding self)
    
    print(f"\n   Job: {job_ids[0]}")
    print(f"   Title: {parse_job_text(jobs[job_ids[0]])['title'][:50]}...")
    print("\n   Most similar jobs:")
    for idx in top_similar:
        sim_job = jobs[job_ids[idx]]
        parsed = parse_job_text(sim_job)
        print(f"   - {job_ids[idx]}: {parsed['title'][:40]}... (sim: {similarities[idx]:.3f})")
    
    print("\n" + "=" * 60)
    print("✓ Job embeddings created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
