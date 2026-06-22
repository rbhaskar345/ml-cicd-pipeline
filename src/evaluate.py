import json
import sys
import os

MIN_ACCURACY = 0.85
MIN_F1_SCORE = 0.80

def evaluate_model():
    metrics_path = 'metrics/metrics.json'
    
    if not os.path.exists(metrics_path):
        print("ERROR: Metrics file not found!")
        sys.exit(1)
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    accuracy = metrics['accuracy']
    f1 = metrics['f1_score']
    
    print("\nModel Evaluation Report:")
    print("=" * 40)
    print(f"Accuracy: {accuracy:.4f} "
          f"(min: {MIN_ACCURACY})")
    print(f"F1 Score: {f1:.4f} "
          f"(min: {MIN_F1_SCORE})")
    print("=" * 40)
    
    passed = True
    
    if accuracy < MIN_ACCURACY:
        print(f"FAIL: Accuracy below threshold!")
        passed = False
    else:
        print(f"PASS: Accuracy meets threshold!")
    
    if f1 < MIN_F1_SCORE:
        print(f"FAIL: F1 below threshold!")
        passed = False
    else:
        print(f"PASS: F1 meets threshold!")
    
    if passed:
        print("\nAll checks PASSED!")
        sys.exit(0)
    else:
        print("\nEvaluation FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    evaluate_model()