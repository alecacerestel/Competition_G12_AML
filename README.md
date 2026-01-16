# Advanced ML Project

Machine learning challenge for job listing classification

**Data**: Training and test datasets with job listings features

**Challenge**: [ENS Challenge #164](https://challengedata.ens.fr/participants/challenges/164/submissions)

**link drive**: [Google Drive Folder](https://drive.google.com/drive/folders/1YCL4CPkvpp2KI8DE986CrnSjX8JLxSXP?usp=sharing)  

**link Rapport**: [Rapport de projet](https://drive.google.com/drive/u/0/folders/1YCL4CPkvpp2KI8DE986CrnSjX8JLxSXP)



PROJECT ROADMAP
===============

Phase 1: Data Exploration
- Load and clean data
- Exploratory Data Analysis (EDA)
- Understand aspects   

Phase 2: Baseline Models
- Implement MRR and Accuracy metrics
- Create simple baseline (most popular jobs, last viewed)
- Temporal cross-validation

Phase 3: Feature Engineering
- Sequence features (order, recency, frequency) 
- Job features (text similarity with NLP) 
- Session features (behavior patterns)

Phase 4: Modeling
- Collaborative Filtering (item-item, user-item)
- Online Learning (River library, bandit algorithms)
- Sequential models (RNN/LSTM if applicable)
- Ensemble models


Data
===============
**Training Data**
- `x_train.csv`: Input features for training set containing user session sequences and job interactions
- `y_train.csv`: Target labels for training set with the jobs that users actually applied to

**Test Data**
- `x_test.csv`: Input features for test set used for final predictions

**Submission**
- `random_predictions.csv`: Example submission file showing the required format for predictions

**Supplementary Files**
- `job_listings.json`: Complete catalog of job postings with detailed information (title, description, requirements, etc.)

## Implemented Model

### Co-occurrence Model

**Architecture:**

The model has **two independent components**:

#### Component 1: Job Recommender a Co-occurrence model
**Input:**
- User session: list of visited jobs `[job_1, job_2, ..., job_n]`

**Process:**
1. Extracts the last 3 jobs from the session as context
2. For each job in context, looks up in transition matrix which jobs usually follow
3. Aggregates scores giving more weight to recent jobs
4. Applies hybrid re-ranking:
   - 60% model score (transition frequency)
   - 25% diversity (from job_balancer.csv)
   - 15% skill overlap (from job_skills.csv)

**Output:**
- List of 10 ranked job IDs: `[job_a, job_b, ..., job_j]`

**Why it works:**
- Captures direct patterns: "after seeing X, users see Y"

---

#### Component 2: Action Classifier
**Input (11 features):**
1. `apply_ratio`: Proportion of applies in session (0.0 to 1.0)
2. `view_ratio`: Proportion of views in session (0.0 to 1.0)
3. `seq_length`: Total number of jobs in session
4. `last_action_is_apply`: Whether last action was apply (0 or 1)
5. `action_change_ratio`: Frequency of view→apply or apply→view changes
6. `top1_similarity`: Score of ranked job #1
7. `top2_similarity`: Score of ranked job #2
8. `top3_similarity`: Score of ranked job #3
9. `gap_top1_top2`: Score difference between top-1 and top-2
10. `avg_top5_similarity`: Average score of top-5
11. `top1_was_viewed`: Whether job #1 was already viewed in session (0 or 1)

**Process:**
1. Trains RandomForest on training data with ground truth
2. For each test session, extracts the 11 features
3. Predicts probability of apply vs view
4. Assigns action with highest probability

**Output:**
- Predicted action: `"view"` or `"apply"`

**Why it works:**
- Users with high apply_ratio tend to keep applying
- last_action_is_apply captures behavioral momentum
- top1_similarity indicates model confidence in recommendation
- 83.57% accuracy is competitive

---

**Complete Flow:**
```
Session [305, 299, 300] → Co-occurrence → Top-10: [88, 214, 138, ...]
                                              ↓
                        Features: apply_ratio=0.2, seq_length=3, ...
                                              ↓
                        RandomForest → Action: "view"
                                              ↓
                Final Output: session_id=0, action="view", job_id="[88, 214, 138, ...]"
```

---

**Run:**
```bash
# 1. Train model
python src\scripts\train_cooccurrence.py

# 2. Evaluate on validation
python src\scripts\evaluate_cooccurrence.py

# 3. Generate final submission
python src\scripts\generate_cooccurrence_submission.py
```

**Results:**
- MRR: 0.0345
- Hit Rate for the top 10 jobs: 10.99%
- Action Accuracy: 83.57%
- **Final Score: 0.2748**
    
**Generated Files:**
- Model: `experiments/cooccurrence_model.pkl`
- Submission: `submissions/cooccurrence_submission.csv`

---

### How the Final Score is Calculated

**Challenge Formula:**
```
Final Score = 0.7 × MRR + 0.3 × Accuracy
Final Score = 0.7 × 0.0345 + 0.3 × 0.8357
Final Score = 0.0242 + 0.2507
Final Score = 0.2748
```

**Components:**
1. **MRR (70% of the score):** Measures how high the correct job is ranked in the top-10
   - If the correct job is in position 1: MRR = 1.0
   - If it is in position 2: MRR = 0.5
   - If it is in position 10: MRR = 0.1
   - If it is not in the top-10: MRR = 0.0

2. **Accuracy (30% of the score):** Measures if it correctly predicts view vs apply
   - Correct prediction: +1
   - Incorrect prediction: 0

---
