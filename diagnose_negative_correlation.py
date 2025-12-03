"""
Diagnose why transfer learning is producing negative correlation
"""

import json
import numpy as np
from pathlib import Path

def diagnose_results():
    """Analyze the results to identify issues"""
    
    results_path = Path("mosei_to_mosi_results.json")
    if not results_path.exists():
        print("ERROR: results file not found!")
        return
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print("DIAGNOSIS: Negative Correlation Issue")
    print("="*80)
    
    print(f"\nTest Results:")
    print(f"  MSE: {data.get('mse', 'N/A')}")
    print(f"  Correlation: {data.get('correlation', 'N/A')}")
    print(f"  MAE: {data.get('mae', 'N/A')}")
    print(f"  Valid samples: {data.get('num_samples', 'N/A')}")
    
    # Check if predictions and labels are saved
    if 'predictions' in data and 'labels' in data:
        predictions = np.array(data['predictions'])
        labels = np.array(data['labels'])
        
        print(f"\nPrediction Statistics:")
        print(f"  Predictions range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
        print(f"  Predictions mean: {np.mean(predictions):.4f}")
        print(f"  Predictions std: {np.std(predictions):.4f}")
        
        print(f"\nLabel Statistics:")
        print(f"  Labels range: [{np.min(labels):.4f}, {np.max(labels):.4f}]")
        print(f"  Labels mean: {np.mean(labels):.4f}")
        print(f"  Labels std: {np.std(labels):.4f}")
        
        # Check for sign issues
        pred_pos = np.sum(predictions > 0)
        pred_neg = np.sum(predictions < 0)
        pred_zero = np.sum(predictions == 0)
        
        label_pos = np.sum(labels > 0)
        label_neg = np.sum(labels < 0)
        label_zero = np.sum(labels == 0)
        
        print(f"\nPrediction Distribution:")
        print(f"  Positive (>0): {pred_pos} ({100*pred_pos/len(predictions):.1f}%)")
        print(f"  Negative (<0): {pred_neg} ({100*pred_neg/len(predictions):.1f}%)")
        print(f"  Zero: {pred_zero} ({100*pred_zero/len(predictions):.1f}%)")
        
        print(f"\nLabel Distribution:")
        print(f"  Positive (>0): {label_pos} ({100*label_pos/len(labels):.1f}%)")
        print(f"  Negative (<0): {label_neg} ({100*label_neg/len(labels):.1f}%)")
        print(f"  Zero: {label_zero} ({100*label_zero/len(labels):.1f}%)")
        
        # Check if signs are inverted
        if (pred_pos > label_neg and pred_neg > label_pos) or (np.mean(predictions) * np.mean(labels) < 0):
            print(f"\n⚠️  WARNING: Signs may be inverted!")
            print(f"   Mean prediction: {np.mean(predictions):.4f}")
            print(f"   Mean label: {np.mean(labels):.4f}")
            print(f"   Product: {np.mean(predictions) * np.mean(labels):.4f}")
        
        # Sample comparisons
        print(f"\nSample Predictions vs Labels (first 10):")
        for i in range(min(10, len(predictions))):
            print(f"  Sample {i}: Pred={predictions[i]:.4f}, Label={labels[i]:.4f}, Error={abs(predictions[i]-labels[i]):.4f}")
    
    print(f"\n{'='*80}")
    print("POSSIBLE ISSUES:")
    print(f"{'='*80}")
    print("1. Label sign inversion (predictions might need negation)")
    print("2. Feature adapter not properly trained (adapters producing wrong features)")
    print("3. Model loaded incorrectly (wrong checkpoint or weights)")
    print("4. Feature extraction mismatch (FaceMesh/BERT/Librosa features incompatible)")
    print("5. Normalization/preprocessing difference between train and test")
    print("6. Adapter dimension mismatch or architecture issue")

if __name__ == '__main__':
    diagnose_results()




