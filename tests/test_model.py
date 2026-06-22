import pytest
import json
import os
import sys
import numpy as np

sys.path.insert(0, 'src')
from train import train_model
from predict import predict, load_model

class TestTraining:
    
    def test_model_trains(self):
        accuracy, f1 = train_model()
        assert accuracy > 0
        assert f1 > 0
    
    def test_model_file_exists(self):
        assert os.path.exists('models/model.pkl')
    
    def test_metrics_file_exists(self):
        assert os.path.exists('metrics/metrics.json')
    
    def test_accuracy_above_threshold(self):
        with open('metrics/metrics.json') as f:
            metrics = json.load(f)
        assert metrics['accuracy'] >= 0.85
    
    def test_f1_above_threshold(self):
        with open('metrics/metrics.json') as f:
            metrics = json.load(f)
        assert metrics['f1_score'] >= 0.80

class TestPrediction:
    
    def test_prediction_works(self):
        result = predict(
            [1.2, 0.5, -0.3, 2.1, 0.8,
             -1.2, 0.4, 1.1, -0.7, 0.9]
        )
        assert 'prediction' in result
        assert 'performance_tier' in result
        assert 'confidence' in result
    
    def test_valid_tier(self):
        result = predict(
            [1.2, 0.5, -0.3, 2.1, 0.8,
             -1.2, 0.4, 1.1, -0.7, 0.9]
        )
        assert result['performance_tier'] in [
            "High Performer",
            "Low Performer"
        ]
    
    def test_confidence_range(self):
        result = predict(
            [0.5, -0.3, 1.2, -0.8, 0.6,
              0.9, -0.4, 1.1, 0.3, -0.7]
        )
        assert 0 <= result['confidence'] <= 1
    
    def test_batch_predictions(self):
        model = load_model()
        samples = [
            [1.2, 0.5, -0.3, 2.1, 0.8,
             -1.2, 0.4, 1.1, -0.7, 0.9],
            [-0.5, 1.2, 0.8, -0.3, 1.1,
              0.4, -0.7, 0.9, 2.1, -1.2]
        ]
        for features in samples:
            result = predict(features, model)
            assert result['prediction'] in [0, 1]