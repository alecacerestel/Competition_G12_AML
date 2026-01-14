# Advanced ML Project

Machine learning challenge for job listing classification

**Data**: Training and test datasets with job listings features

**Challenge**: [ENS Challenge #164](https://challengedata.ens.fr/participants/challenges/164/submissions)

**link drive**: [Google Drive Folder](https://drive.google.com/drive/folders/1YCL4CPkvpp2KI8DE986CrnSjX8JLxSXP?usp=sharing)  

**link Rapport**: [Rapport de projet](https://drive.google.com/drive/u/0/folders/1YCL4CPkvpp2KI8DE986CrnSjX8JLxSXP)

Code Objective
===============
- The top 10 jobs a candidate is most likely to explore next.
- The candidate’s next action. It can be view or apply.


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
