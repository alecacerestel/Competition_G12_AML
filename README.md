# Job Recommendation System - Advanced Machine Learning Project

## Project Overview

This project solves a job recommendation challenge from ENS Data Challenge page. The goal is to predict which job a user will interact with next and what type of action they will perform (view or apply).

The system receives a sequence of jobs that a user has already seen and must predict:
1. The next 10 jobs the user is most likely to interact with (ranked by probability)
2. Whether the user will view or apply to the next job

The final score combines both predictions using a weighted formula where job ranking quality counts for 70% and action prediction accuracy counts for 30%.

**Challenge Link:** [ENS Challenge](https://challengedata.ens.fr/participants/challenges/164/)


## Problem Context

### The Business Problem

Job platforms like LinkedIn or JobTeaser need to recommend relevant jobs to users. A good recommendation system should:
- Show jobs that the user is likely to click on
- Predict user intent (just browsing vs ready to apply)
- Adapt to user behavior patterns during the session
- Propose similar jobs based on previous interactions (Please with us an Internship!)

### Why This Is Difficult

1. **Session-Based Behavior**: Users change intent during a session
2. **Large Catalog**: Thousands of jobs to choose from
3. **Implicit Feedback**: We only see clicks, not explicit ratings
4. **Sequential Nature**: The order of interactions matters

### Evaluation Metrics

The challenge uses two metrics:

**Mean Reciprocal Rank (MRR)**: Measures the quality of the ranked job list
- If the correct job is ranked first: score = 1.0
- If the correct job is ranked second: score = 0.5
- If the correct job is ranked tenth: score = 0.1
- If the correct job is not in top 10: score = 0.0

**Accuracy**: Measures how often we correctly predict view vs apply

**Final Score Formula:**
```
Final Score = 0.7 * MRR + 0.3 * Accuracy
```

## Data Description

### Training Data

| File | Description |
|------|-------------|
| `x_train_Meacfjr.csv` | Raw training sessions with job sequences |
| `x_train_with_features.csv` | Processed training data with engineered features |
| `y_train_SwJNMSu.csv` | next job and action for each session |

### Test Data

| File | Description |
|------|-------------|
| `x_test_jCBBNP2.csv` | Raw test sessions |
| `x_test_with_features.csv` | Processed test data with features |

### Supplementary Data

| File | Description |
|------|-------------|
| `job_listings.json` | Full job catalog |
| `job_listing_feature/job_balancer.csv` | Diversity weights for each job |
| `job_listing_feature/job_skills.csv` | Skills associated with each job |

---

## Solution Architecture

Our solution has two independent components that work together:

### Component 1: Job Recommender, Co-occurrence Model

This part predicts which 10 jobs the user will most likely interact next.

**Approach**: Item-to-item collaborative filtering based on transition patterns.

**Intuition**: If many users who viewed job A then viewed job B, and a new user just viewed job A, they will probably view job B next.

### Component 2: Action Classifier, SVM

This component predicts whether the user will "view" or "apply" to the next job.

**Approach**: Supervised classification using session behavior features.

**Intuition**: Users who have been applying to jobs during the session are more likely to apply again. Users who are just browsing tend to continue browsing.

## Mathematical Foundation

### Co-occurrence Model

#### Transition Probability

We model job recommendations as a Markov chain where the probability of the next job depends on recent jobs in the session.

Given a session with jobs `[j_1, j_2, ..., j_n]`, we want to find:

```
P(j_next | j_{n}, j_{n-1}, j_{n-2})
```

We estimate this probability using co-occurrence counts from training data.

#### Building the Transition Matrix

For each pair of jobs `(j_i, j_k)` that appear within a window of size `w=3` in the training data, we count weighted transitions:

```
weight(j_i -> j_k) = 1 / distance(i, k)
```

Where `distance(i, k) = k - i` (how many positions apart).

This gives higher weight to jobs that appear closer together.

#### Transition Count Formula

For job `j_i`, the transition count to job `j_k` is:

```
T(j_i, j_k) = sum over all sessions of: weight(j_i -> j_k)
```

#### Scoring Candidates

Given the last `w=3` jobs in a session `[j_{n-2}, j_{n-1}, j_n]`, we score each candidate job `c` as:

```
score(c) = sum_{i=n-2}^{n} [ position_weight(i) * T(j_i, c) ]
```

Where `position_weight(i)` gives more importance to more recent jobs:
- `j_n` (most recent): weight = 1.0
- `j_{n-1}`: weight = 0.67
- `j_{n-2}`: weight = 0.33

#### Hybrid Re-ranking

The final score combines multiple signals:

```
final_score(c) = 0.60 * model_score(c) + 0.25 * diversity_score(c) + 0.15 * skill_overlap(c)
```

Where:
- `model_score`: Normalized transition probability from co-occurrence matrix
- `diversity_score`: Weight from job_balancer.csv
- `skill_overlap`: Similarity between session skills and candidate skills

#### Skill Overlap, Similarity

```
skill_overlap(session, candidate) = |S_session AND S_candidate| / |S_session OR S_candidate|
```

Where `S_session` is the union of all skills from jobs in the session.

### SVM Action Classifier

#### Problem Formulation

Binary classification problem:
- Class 0: "view"
- Class 1: "apply"

#### Feature Vector

For each session, we create 11 features:

```
x = [apply_ratio, view_ratio, seq_length, last_action_is_apply, action_change_ratio, top1_similarity, top2_similarity, top3_similarity, gap_top1_top2, avg_top5_similarity, top1_was_viewed]
```

#### SVM with RBF Kernel

The SVM finds a decision boundary in a high-dimensional feature space using the RBF (Radial Basis Function) kernel:

```
K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
```

The decision function is:

```
f(x) = sign(sum_{i} alpha_i * y_i * K(x_i, x) + b)
```

Where:
- `alpha_i`: Lagrange multipliers
- `y_i`: Training labels (+1 or -1)
- `b`: Bias term
- `gamma = 1 / (n_features * variance)` (using 'scale' option)

#### Hyperparameters

- `C = 1.0`: Regularization parameter (trade-off between margin and errors)
- `kernel = 'rbf'`: Non-linear kernel for complex decision boundaries
- `gamma = 'scale'`: Automatic gamma based on feature variance

### Evaluation Metrics, MRR

#### Final Score

```
Final Score = 0.7 × MRR + 0.3 × Accuracy
```
## How to Run

### Prerequisites

1. Required packages:

```bash
pip install -r requirements.txt
```
### Instructions

#### Step 1: Prepare the Data

Make sure you have the following files in the `Data/` folder:
- `x_train_Meacfjr.csv`
- `y_train_SwJNMSu.csv`
- `x_test_jCBBNP2.csv`
- `job_listing_feature/job_balancer.csv`
- `job_listing_feature/job_skills.csv`

`x_train_with_features.csv` and `x_test_with_features.csv` do not exist, run the notebooks
This will create the feature files in `Data/`.

#### Step 2: Train the Co-occurrence Model

```bash
python src/scripts/train_cooccurrence.py
```

This will:
- Load training data
- Build the co-occurrence transition matrix
- Save the model to `experiments/cooccurrence_model.pkl`

Expected output:
```
Loading training data...
Training on 15882 sessions
Building co-occurrence matrix with window_size=3...
Found 19204 source jobs with transitions
Total unique jobs: 20505
Model saved to experiments/cooccurrence_model.pkl
```

#### Step 3: Evaluate the model

```bash
python src/scripts/evaluate_cooccurrence.py
```

This will:
- Split training data into train/validation (85%/15%)
- Generate predictions on validation set
- Calculate MRR, Hit Rate, and Accuracy
- Print detailed results

Expected output:
```
MRR (Mean Reciprocal Rank): 0.0345
Hit Rate@10: 10.99%
Action Prediction Accuracy: 83.70%
Final Score: 0.2752
```

#### Step 4: Generate Submission

```bash
python src/scripts/generate_cooccurrence_submission.py
```

This will:
- Load the trained model
- Generate predictions for test data
- Train the action classifier
- Save submission to `submissions/cooccurrence_submission.csv`

Expected output:
```
Generating predictions for 1819 test sessions...
Trained SVM classifier on 15882 samples
Submission saved to submissions/cooccurrence_submission.csv
```

## Results

### Final Performance

| Metric | Value |
|--------|-------|
| MRR | 0.0345 |
| Hit Rate at 1 | 0.98% |
| Hit Rate at 5 | 6.58% |
| Hit Rate at 10 | 10.99% |
| Action Accuracy | 83.70% |
| **Final Score** | **0.2752** |

### Analysis

1. **MRR is low (0.0345)**: This means the correct job is rarely in the top positions. This is expected because:
   - There are 20,000+ possible jobs
   - User behavior is hard to predict exactly
   - Our model uses simple co-occurrence patterns
   - Hundresds of jobs similar in content but not in the name and format

2. **Action accuracy is high (83.70%)**: The SVM classifier works well because:
   - User behavior within a session is consistent
   - Features like `apply_ratio` and `last_action_is_apply` are strong predictors

3. **Diversity**: Our model recommends 2,114 unique jobs across all test sessions, showing good coverage.