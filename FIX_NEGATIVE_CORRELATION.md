# Fixing Negative Correlation Issue

## Problem Analysis

**Results:**
- Correlation: **-0.1020** (NEGATIVE - this is bad!)
- MSE: 0.7680 (reasonable)
- MAE: 0.7081 (reasonable)
- 93 valid samples

**What Negative Correlation Means:**
- Predictions are **inversely** related to labels
- When sentiment is positive, model predicts negative
- When sentiment is negative, model predicts positive
- This suggests a **sign inversion** or **fundamental mismatch**

---

## Possible Causes

### 1. **Sign Inversion (Most Likely)**
- Predictions might need to be negated: `-predictions`
- Model might be learning inverted relationship
- Simple fix: multiply predictions by -1

### 2. **Adapter Training Issue**
- Adapters might be mapping features incorrectly
- Adapters learned wrong direction in feature space
- Need to check adapter training loss

### 3. **Feature Distribution Mismatch**
- Adapted features have wrong distribution
- MOSEI and MOSI features are too different
- Adapters can't bridge the gap

### 4. **Model Loading Issue**
- Wrong checkpoint loaded
- Model trained on wrong data
- Weights corrupted

### 5. **Label Format Issue**
- Labels might be in different format than training
- CMU-MOSI labels might need transformation
- Labels might be inverted

---

## Quick Fixes to Try

### Fix 1: Check Sign Inversion
```python
# In test_on_mosi function, try negating predictions:
pred = model(v_adapted, a_adapted, t_adapted)
predictions.extend((-pred).cpu().numpy().flatten())  # Negate!
```

### Fix 2: Check Predictions vs Labels
```python
# Add this to see if signs are inverted:
print(f"Sample: Pred={predictions[0]:.4f}, Label={labels[0]:.4f}")
# If pred > 0 and label < 0 (or vice versa), signs are inverted
```

### Fix 3: Check Adapter Outputs
```python
# Verify adapters are producing reasonable features:
print(f"Visual adapter output range: [{v_adapted.min():.4f}, {v_adapted.max():.4f}]")
print(f"Audio adapter output range: [{a_adapted.min():.4f}, {a_adapted.max():.4f}]")
print(f"Text adapter output range: [{t_adapted.min():.4f}, {t_adapted.max():.4f}]")
```

---

## Diagnostic Steps

1. **Check if predictions are inverted:**
   - Compare prediction signs with label signs
   - If most predictions have opposite sign to labels → sign inversion

2. **Check adapter training loss:**
   - Visual adapter loss should be decreasing
   - Audio adapter loss should be decreasing
   - Text adapter loss should be decreasing
   - If losses are high (>1000), adapters aren't learning

3. **Check feature statistics:**
   - Compare MOSEI feature stats vs adapted MOSI feature stats
   - Mean, std should be similar
   - If very different, adapters aren't working

4. **Check model predictions on MOSEI:**
   - Test trained model on MOSEI test set
   - Should get positive correlation (~0.44-0.48)
   - If negative, model issue

---

## Most Likely Issue: Adapter Training

Given that:
- MSE and MAE are reasonable (0.77 and 0.71)
- But correlation is negative
- All 93 samples have labels

**Most likely:** Adapters are not properly mapping features to MOSEI space.

### Solution:
1. **Re-train adapters with better settings:**
   - More epochs (30 → 50-100)
   - Better learning rate scheduling
   - K-means target selection instead of random

2. **Check adapter outputs match MOSEI distributions:**
   - Compare statistics (mean, std) of adapted features vs MOSEI features
   - Should be similar

3. **Use different adapter architecture:**
   - Deeper adapters
   - Residual connections
   - Batch normalization

---

## Quick Test: Try Negating Predictions

The easiest test is to see if negating predictions fixes it:

```python
# In test_on_mosi, replace:
predictions.extend(pred.cpu().numpy().flatten())

# With:
predictions.extend((-pred).cpu().numpy().flatten())
```

If correlation becomes positive (e.g., +0.10), then it's a sign inversion issue and you need to figure out where the sign gets flipped.

---

## Expected Fixes

After fixing, you should see:
- Correlation: **0.30-0.45** (positive!)
- MSE: 0.60-1.00 (similar or slightly better)
- MAE: 0.60-0.85 (similar or slightly better)

If you get positive correlation but still low (0.10-0.20), then adapters need improvement.




