#!/usr/bin/env python3
"""
Test script for feature engineering pipeline
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from src.pipeline.data_processor import DataProcessor
from src.pipeline.feature_engineer import FeatureEngineer


def test_sequence_features():
    """Test sequence feature extraction"""
    print("Testing sequence features...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'session_id': [0, 1, 2],
        'job_ids': ['[1, 2, 3, 2, 1]', '[5, 5, 5]', '[10, 11, 12, 13]'],
        'actions': ['[view, view, view, view, view]', '[view, view, apply]', '[view, view, view, apply]']
    })
    
    # Initialize and process
    processor = DataProcessor()
    sample_data = processor.prepare_sequences(sample_data)
    sample_data_feat = processor.add_features(sample_data, feature_types=['sequence'])
    
    # Assertions
    assert 'unique_jobs_count' in sample_data_feat.columns
    assert 'most_frequent_job' in sample_data_feat.columns
    assert 'job_transitions' in sample_data_feat.columns
    assert sample_data_feat.loc[0, 'unique_jobs_count'] == 3  # [1,2,3,2,1] has 3 unique
    assert sample_data_feat.loc[1, 'most_frequent_job'] == 5  # [5,5,5] most frequent is 5
    assert sample_data_feat.loc[2, 'job_transitions'] == 3  # [10,11,12,13] has 3 transitions
    
    print("✓ Sequence features test passed!")
    return True


def test_session_features():
    """Test session behavior feature extraction"""
    print("Testing session behavior features...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'session_id': [0, 1, 2],
        'job_ids': ['[1, 2, 3]', '[4, 5, 6]', '[7, 8]'],
        'actions': ['[view, view, apply]', '[view, apply, apply]', '[apply, apply]']
    })
    
    # Initialize and process
    processor = DataProcessor()
    sample_data = processor.prepare_sequences(sample_data)
    sample_data_feat = processor.add_features(sample_data, feature_types=['sequence', 'session'])
    
    # Assertions
    assert 'view_count' in sample_data_feat.columns
    assert 'apply_count' in sample_data_feat.columns
    assert 'view_ratio' in sample_data_feat.columns
    assert sample_data_feat.loc[0, 'view_count'] == 2
    assert sample_data_feat.loc[0, 'apply_count'] == 1
    assert sample_data_feat.loc[1, 'apply_count'] == 2
    assert sample_data_feat.loc[2, 'apply_ratio'] == 1.0  # All applies
    
    print("✓ Session behavior features test passed!")
    return True


def test_text_features():
    """Test text feature extraction (basic check)"""
    print("Testing text features (basic)...")
    
    # Create minimal jobs dict
    jobs = {
        '1': {
            'TITLE': 'Software Engineer',
            'SUMMARY': 'Python developer position',
            'DESCRIPTION': [{'DESCRIPTION': 'Work with Python and ML'}],
            'SKILLS': [{'name': 'Python'}, {'name': 'ML'}]
        },
        '2': {
            'TITLE': 'Data Scientist',
            'SUMMARY': 'ML and AI position',
            'DESCRIPTION': [{'DESCRIPTION': 'Work with machine learning'}],
            'SKILLS': [{'name': 'Python'}, {'name': 'ML'}, {'name': 'AI'}]
        },
        '3': {
            'TITLE': 'Marketing Manager',
            'SUMMARY': 'Marketing role',
            'DESCRIPTION': [{'DESCRIPTION': 'Lead marketing campaigns'}],
            'SKILLS': [{'name': 'Marketing'}, {'name': 'Leadership'}]
        }
    }
    
    # Create sample data
    sample_data = pd.DataFrame({
        'session_id': [0, 1],
        'job_ids': ['[1, 2]', '[1, 3]'],
        'actions': ['[view, view]', '[view, view]']
    })
    
    # Initialize feature engineer with jobs
    fe = FeatureEngineer(jobs_dict=jobs)
    sample_data = fe.extract_sequence_features(
        pd.DataFrame({
            'session_id': sample_data['session_id'],
            'job_ids': sample_data['job_ids'],
            'actions': sample_data['actions'],
            'jobs_list': sample_data['job_ids'].apply(eval),
            'actions_list': sample_data['actions'].apply(eval)
        })
    )
    
    sample_data_feat = fe.extract_job_text_features(sample_data)
    
    # Basic assertions
    assert 'avg_job_similarity' in sample_data_feat.columns
    assert 'total_skills_count' in sample_data_feat.columns
    
    # Jobs 1 and 2 should be more similar than 1 and 3
    sim_0 = sample_data_feat.loc[0, 'avg_job_similarity']
    sim_1 = sample_data_feat.loc[1, 'avg_job_similarity']
    assert sim_0 > sim_1  # Python/ML jobs more similar than Python/Marketing
    
    print("✓ Text features test passed!")
    return True


def test_full_pipeline():
    """Test the full pipeline with real data (first few rows)"""
    print("Testing full pipeline with real data...")
    
    try:
        # Load real data
        processor = DataProcessor()
        jobs, x_train, y_train, x_test = processor.load_data()
        
        # Process small sample
        x_sample = x_train.head(10)
        x_sample = processor.prepare_sequences(x_sample)
        
        # Extract all features
        x_sample_feat = processor.add_features(x_sample, feature_types=['sequence', 'session'])
        
        # Verify features exist
        expected_features = ['unique_jobs_count', 'view_count', 'apply_ratio']
        for feat in expected_features:
            assert feat in x_sample_feat.columns, f"Missing feature: {feat}"
        
        # Verify no NaN in key features
        assert not x_sample_feat['seq_length'].isna().any()
        assert not x_sample_feat['unique_jobs_count'].isna().any()
        
        print("✓ Full pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Full pipeline test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Running Feature Engineering Tests")
    print("="*60)
    print()
    
    tests = [
        test_sequence_features,
        test_session_features,
        test_text_features,
        test_full_pipeline
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with error: {e}")
            results.append(False)
        print()
    
    print("="*60)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print("="*60)
    
    return all(results)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
