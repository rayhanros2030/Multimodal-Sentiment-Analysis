# Dataset Splits Used

## Standard Split for All Experiments

### Split Ratio
- **Training Set**: 70% of total samples
- **Validation Set**: 15% of total samples  
- **Test Set**: ~15% of total samples (remaining after train + val)

### Implementation
```python
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)
```

## Applied To

### 1. CMU-MOSEI Only Training (`train_mosei_only.py`)
- Total samples: ~3,292
- Train: ~2,304 samples (70%)
- Validation: ~494 samples (15%)
- Test: ~494 samples (15%)
- **Random split** (no fixed seed specified)

### 2. Combined CMU-MOSEI + IEMOCAP (`train_combined_final.py`)
- Datasets are combined first (ConcatDataset)
- Then split using 70/15/15 ratio
- Total combined: ~13,331 samples
  - Train: ~9,332 samples (70%)
  - Validation: ~1,999 samples (15%)
  - Test: ~2,000 samples (15%)
- **Random split** (no fixed seed specified)

### 3. Modality Combination Tests (`train_mosei_modality_combinations.py`)
- Uses same 70/15/15 split
- **Random split with fixed seed** (seed=42 for reproducibility)
- Ensures fair comparison across different modality combinations

### 4. Optimized 3-Modality Training (`train_mosei_three_modalities_complete.py`)
- Uses same 70/15/15 split
- **Random split with fixed seed** (seed=42)

## Important Details

### Random Split vs Fixed Seed
- **CMU-MOSEI only**: Random split (no seed) - different splits each run
- **Combined training**: Random split (no seed)
- **Modality comparisons**: Fixed seed (42) - ensures same splits for fair comparison
- **Optimized training**: Fixed seed (42) - reproducible results

### Data Leakage Prevention
- **RobustScaler fitted ONLY on training data**
- Applied to validation and test via `TransformedSubset` wrapper
- Ensures no information from val/test leaks into normalization

## Split Strategy Rationale

### Why 70/15/15?
- **70% training**: Standard for deep learning (sufficient data for learning)
- **15% validation**: Enough for reliable validation metrics and early stopping
- **15% test**: Standard holdout set for final evaluation

### Why Random Split?
- Simple and straightforward
- Ensures balanced representation across splits
- Standard practice for regression tasks
- No temporal dependencies to preserve (videos are independent)

### For Modality Comparisons
- **Fixed seed ensures fairness**: Same samples in each split for all combinations
- Allows direct comparison of modality effects
- Eliminates randomness as a confounding factor

## Example Splits (CMU-MOSEI with ~3,292 samples)

| Split | Percentage | Approximate Count |
|-------|-----------|-------------------|
| Train | 70% | ~2,304 samples |
| Validation | 15% | ~494 samples |
| Test | ~15% | ~494 samples |
| **Total** | **100%** | **~3,292 samples** |

## For Regeneron Documentation

You can state:
- "Standard 70/15/15 train/validation/test split applied consistently"
- "Random split with fixed seed (42) for modality ablation studies to ensure fair comparison"
- "RobustScaler normalization fitted only on training data to prevent data leakage"
- "Splits applied to combined dataset when using CMU-MOSEI + IEMOCAP"

