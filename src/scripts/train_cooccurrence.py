import pandas as pd
import ast
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.cooccurrence_model import CooccurrenceRecommender


def main():
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / 'Data'
    
    print("Loading training data...")
    x_train = pd.read_csv(data_path / 'x_train_with_features.csv')
    
    print(f"Training on {len(x_train)} sessions")
    
    # Train model
    model = CooccurrenceRecommender(window_size=3, min_count=2)
    model.fit(x_train)
    
    # Save model
    model_path = base_path / 'experiments' / 'cooccurrence_model.pkl'
    model.save(model_path)
    
    # Test prediction on first session
    print("\nTesting prediction on sample session...")
    sample_session = ast.literal_eval(x_train.iloc[0]['job_ids'])
    print(f"Session jobs: {sample_session[:5]}...")
    
    predictions = model.predict(sample_session, top_k=10, return_scores=True)
    print(f"Top-10 predictions:")
    for rank, (job, score) in enumerate(predictions, 1):
        print(f"  {rank}. Job {job}: {score:.4f}")


if __name__ == '__main__':
    main()
