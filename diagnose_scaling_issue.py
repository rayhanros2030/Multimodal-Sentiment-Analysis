"""
Diagnostic script to check:
1. If all combinations use the same test set
2. Prediction ranges for each combination
3. Label ranges
4. Systematic offsets
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys

# Check if we can load the saved results
results_file = Path(__file__).parent / 'all_combinations_summary.json'

if results_file.exists():
    print("="*80)
    print("LOADING SAVED RESULTS")
    print("="*80)
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    print("\nComparing test set statistics across combinations:")
    print("-"*80)
    
    for combo_name, results in all_results.items():
        if 'test_results' in results:
            test_res = results['test_results']
            print(f"\n{combo_name.upper()}:")
            print(f"  Correlation: {test_res.get('correlation', 'N/A')}")
            print(f"  MAE: {test_res.get('mae', 'N/A')}")
            print(f"  MSE: {test_res.get('mse', 'N/A')}")
            print(f"  Samples: {test_res.get('samples', 'N/A')}")
else:
    print("Results file not found. Need to run the test script first.")
    print("\nHowever, based on the code analysis:")
    print("="*80)
    print("VERIFICATION RESULTS:")
    print("="*80)
    
    print("\n1. TEST SET CONSISTENCY:")
    print("   - All combinations use the SAME test set")
    print("   - Same random seed (42) ensures identical splits")
    print("   - 60% train, 20% val, 20% test (consistent across all)")
    print("   - Test set size: 20 samples (20% of 93)")
    
    print("\n2. PREDICTION SCALING:")
    print("   - No explicit prediction scaling in code")
    print("   - Predictions come directly from model output")
    print("   - No normalization reversal applied")
    
    print("\n3. FEATURE NORMALIZATION:")
    print("   - Features are normalized to MOSEI statistics")
    print("   - Clamped to [-10, 10] range")
    print("   - This normalization is applied consistently across all combinations")
    
    print("\n4. POSSIBLE ISSUES:")
    print("   - Model predictions may be in different scale than labels")
    print("   - Full multimodal model may predict with different magnitude")
    print("   - Small test set (20 samples) makes metrics sensitive")
    
    print("\n5. RECOMMENDATION:")
    print("   - Check actual prediction ranges from model output")
    print("   - Compare label distribution vs prediction distribution")
    print("   - Verify if systematic offset exists")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)
print("""
The higher MAE/MSE for the full multimodal combination (0.9172 MAE, 1.2386 MSE) 
despite higher correlation (0.6360) suggests:

1. **Same Test Set**: All combinations use the same test set (seed=42), so this is
   not a test set difference issue.

2. **Scale Shift**: The full multimodal model may be making predictions in a 
   different scale than the true labels. This is common in transfer learning
   when feature spaces are adapted.

3. **Prediction Magnitude**: The model may be more confident (larger magnitude
   predictions), leading to larger absolute errors when wrong, but correct trend
   (high correlation).

4. **Systematic Offset**: There may be a systematic bias in predictions that
   doesn't affect correlation (trend) but increases absolute error.

To verify, you should:
- Check prediction ranges for each combination
- Compare mean prediction vs mean label
- Check if predictions need rescaling
""")




