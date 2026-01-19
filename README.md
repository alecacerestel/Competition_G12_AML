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
| `y_train_SwJNMSu.csv` | Target job and action for each session |

### Test Data

| File | Description |
|------|-------------|
| `x_test_jCBBNP2.csv` | Raw test sessions |

### Supplementary Data

| File | Description |
|------|-------------|
| `job_listings.json` | Full job catalog |

---

## Solution Architecture

Our solution has two independent components that work together:

### Component 1: Job Recommender - Hybrid Ranking Model

This part predicts which 10 jobs the user will most likely interact next.

**Approach**: Combines four different signals: transition probabilities, co-occurrence patterns, collaborative filtering, and job popularity.

**Intuition**: Different signals capture different aspects of user behavior. Transitions capture sequential patterns, co-occurrence captures related jobs in sessions, collaborative filtering finds similar user preferences, and popularity provides a baseline for new jobs.

### Component 2: Action Classifier, SVM

This component predicts whether the user will "view" or "apply" to the next job.

**Approach**: Supervised classification using session behavior features.

**Intuition**: Users who have been applying to jobs during the session are more likely to apply again. Users who are just browsing tend to continue browsing.

## Mathematical Foundation

### Hybrid Job Ranking Model

#### Problem Formulation

Given a sequence of jobs viewed by a user S = [j1, j2, ..., jn], predict the next 10 most likely jobs the user will interact with.

```
Goal: Find rank(j_next | S) for all possible jobs j_next
```

We rank jobs based on their co-occurrence patterns with jobs in the user session.

#### Building the Transition and Co-occurrence Matrices

**Transition Matrix**: Counts direct job-to-job transitions:
```
T[i][j] = count(job j appears immediately after job i)
```

**Co-occurrence Matrix**: Counts jobs appearing in the same session:
```
C[i][j] = count(sessions where both job i and job j appear)
```

Both matrices capture different patterns: transitions capture sequential behavior, co-occurrence captures related jobs.

#### Transition Probabilities

We normalize the transition counts to get probabilities:

```
P(j_next = j | j_current = i) = T[i][j] / sum(T[i][k] for all k)
```

This gives us the probability of moving from job i to job j.

#### Collaborative Filtering Component

We build a session-job interaction matrix and find similar sessions using cosine similarity:

```
CF_score(j) = sum(similarity(current_session, past_session) * interaction(past_session, j))
```

This captures preferences from users with similar behavior patterns.

#### Final Scoring Formula

The final score combines all four signals with learned weights:

```
final_score(j) = w_transition * transition_score(j) +
                 w_cooccurrence * cooccurrence_score(j) +
                 w_cf * cf_score(j) +
                 w_popularity * popularity_score(j)
```

Weights are tuned via grid search:
- w_transition = 3.0 (sequential patterns)
- w_cooccurrence = 3.0 (related jobs)
- w_cf = 2.0 (similar users)
- w_popularity = 0.1 (baseline)
#### Cold Start Handling

```
if no_transition_or_cooccurrence_data:
    rely_on_popularity_component()
```

For sessions with jobs not seen in training, the popularity component ensures we always have candidate recommendations. Jobs are ranked by their overall frequency in the training data.

### SVM Action Classifier

#### Problem Formulation

Binary classification problem:
- Class 0: "view"
- Class 1: "apply"

#### Feature Vector

For each session, we create 5 features:

```
x = [apply_ratio, view_ratio, seq_length, last_action_is_apply, action_change_ratio]
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

#### Step 2: Train the hybrid model

```bash
python pipe/train.py
```

This will:
- Load training data
- Build transition, co-occurrence, and collaborative filtering models
- Train the SVM action predictor
- Evaluate on validation set
- Save models to `experiments/`

Expected output:
```
============================================================
Job Recommendation Training Pipeline
============================================================
Loading data...
  Training samples: 15882
  Test samples: 1819
  Total unique jobs: 21873

Train/Val split: 13500/2382

------------------------------------------------------------
Building job ranker...
  Building transition model...
  Building co-occurrence model...
  Building popularity model...
  Building collaborative filter...
Building SVM action predictor...
Evaluating on validation set...
MRR: 0.0851
Action Accuracy: 0.8342
Final Score: 0.3098
============================================================
```

#### Step 3: Generate Submission

```bash
python pipe/predict.py
```

This will:
- Load the trained models
- Generate predictions for test data
- Save submission to `submissions/submission_cf_TIMESTAMP.csv`

Expected output:
```
============================================================
Generating Submission
============================================================
Loading saved models...
  Models loaded successfully!

Loading test data...
  Test samples: 1819

Generating predictions...
Formatting submission...

  Saved to: submissions/submission_cf_YYYYMMDD_HHMMSS.csv
  Rows: 1819
```

## Results

### Final Performance

The model was evaluated using grid search over hyperparameters. Best configuration:

| Metric | Value |
|--------|-------|
| MRR | 0.0851 |
| Action Accuracy | 83.42% |
| **Final Score** | **0.3098** |

**Best hyperparameters:**
- Transition weight: 3.0
- Co-occurrence weight: 3.0
- Collaborative filtering weight: 2.0
- Popularity weight: 0.1

### Analysis

1. **MRR of 0.0851**: This means the correct job appears in the top-10 recommendations about 9% of the time, with better positions contributing more. This is reasonable because:
   - There are 20,000+ possible jobs in the catalog
   - User behavior patterns are complex and hard to predict exactly
   - Many jobs have similar characteristics but different IDs
   - Sessions are short, providing limited information

2. **Action prediction works well**: The SVM classifier achieves strong accuracy because:
   - User behavior within a session is consistent
   - Features like `apply_ratio` and `last_action_is_apply` are strong predictors
   - The action pattern tends to be stable during a session

3. **Model strength**: The hybrid approach successfully combines:
   - **Sequential patterns** (transitions) - what users do next
   - **Related jobs** (co-occurrence) - jobs viewed together
   - **Similar users** (collaborative filtering) - preferences from similar sessions
   - **Popularity baseline** - ensures coverage for cold-start cases
