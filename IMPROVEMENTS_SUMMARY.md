# Improvements to Increase Correlation

## Current Results
- Test Correlation: 0.4113
- Test MAE: 0.5984
- Validation Correlation: 0.4799

## Key Improvements Implemented

### 1. Better Sentiment Extraction (IMPORTANT!)
**Current:** Only uses `features[0, 0]` (first segment)
**Improved:** Uses mean of all segments: `np.mean(features[:, 0])`

**Why:** Many videos have multiple segments. Using only the first loses information.
**Expected gain:** +0.02-0.05 correlation

### 2. Enhanced Feature Aggregation
**Current:** Simple mean across time: `np.mean(features, axis=0)`
**Improved:** Statistical aggregation: mean, std, min, max, median

**Why:** Captures more information about temporal dynamics
**Expected gain:** +0.01-0.03 correlation

### 3. Larger Model Capacity
**Current:** hidden_dim=192, embed_dim=96
**Improved:** hidden_dim=256, embed_dim=128

**Why:** More capacity to learn complex patterns
**Expected gain:** +0.01-0.02 correlation

### 4. Residual Connections
**Improved:** Skip connections in fusion layers

**Why:** Helps with gradient flow and learning
**Expected gain:** +0.01-0.02 correlation

### 5. More Correlation-Focused Loss
**Current:** alpha=0.4, beta=0.6 (40% MSE/MAE, 60% correlation)
**Improved:** alpha=0.3, beta=0.7 (30% MSE/MAE, 70% correlation)
**Plus:** Squared correlation loss for stronger gradients

**Why:** Directly optimizes for correlation
**Expected gain:** +0.02-0.04 correlation

### 6. Better Hyperparameters
- Learning rate: 0.0008 (slightly lower for stability)
- Weight decay: 0.03 (less aggressive regularization)
- Dropout: 0.6 (allows more learning while preventing overfit)
- Attention heads: 8 (increased from 4)

**Expected gain:** +0.01-0.02 correlation

### 7. Better Feature Cleaning
**Improved:** Tighter clipping (-500 to 500) for extreme values

**Why:** Prevents extreme outliers from dominating
**Expected gain:** +0.01 correlation

## Expected Total Improvement
Combined, these improvements should increase correlation by:
- **Conservative estimate:** +0.05 to +0.08 (correlation ~0.46-0.49)
- **Optimistic estimate:** +0.08 to +0.12 (correlation ~0.49-0.53)

## Implementation Priority
1. **High Priority (Biggest Impact):**
   - Better sentiment extraction (use mean of segments)
   - More correlation-focused loss (70% weight)

2. **Medium Priority:**
   - Enhanced feature aggregation
   - Larger model capacity

3. **Lower Priority (Nice to Have):**
   - Residual connections
   - Hyperparameter tuning

## Quick Wins (Can implement in 10 minutes)
1. Fix sentiment extraction in existing script
2. Adjust loss weights (alpha=0.3, beta=0.7)
3. Slightly increase model size (hidden_dim=224, embed_dim=112)

Expected improvement: +0.03-0.06 correlation




