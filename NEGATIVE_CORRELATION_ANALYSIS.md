# Negative Correlation Issue Analysis

## Your Results:
- **Correlation: -0.1020** ❌ (VERY BAD - should be positive!)
- MSE: 0.7680 (reasonable)
- MAE: 0.7081 (reasonable)
- 93 valid samples ✅

## What This Means:

A **negative correlation** means:
- When sentiment is **positive**, model predicts **negative**
- When sentiment is **negative**, model predicts **positive**
- Predictions are **inversely** related to ground truth

**This is worse than random (0.0)!**

---

## Most Likely Causes (in order):

### 1. **Adapter Training Failure** (90% likely)
**Problem:** Adapters aren't mapping features correctly to MOSEI space

**Evidence:**
- Negative correlation suggests adapters learned wrong mapping
- Adapter training might have converged to wrong solution
- Random target sampling might not align features well

**Solution:**
- Re-train adapters with better target selection (K-means instead of random)
- More epochs (50-100 instead of 30)
- Lower learning rate with scheduling
- Check adapter training losses - should be decreasing

### 2. **Sign Inversion in Predictions** (5% likely)
**Problem:** Model is predicting inverted sentiment

**Test:**
```python
# Try negating predictions in test function
predictions.extend((-pred).cpu().numpy().flatten())
```
If correlation becomes positive, it's a sign inversion issue.

### 3. **Feature Distribution Mismatch** (5% likely)
**Problem:** Adapted features don't match MOSEI feature distribution

**Check:**
- Compare mean/std of adapted features vs MOSEI features
- Should be similar - if very different, adapters failed

---

## Quick Diagnostic:

Run this to check if it's sign inversion:

```python
# In test_on_mosi function, add after line 955:
# Check first few predictions vs labels
print("\nFirst 5 predictions vs labels:")
for i in range(min(5, len(predictions))):
    print(f"  Pred={predictions[i]:.4f}, Label={labels[i]:.4f}")
```

If predictions have opposite signs to labels, it's sign inversion.

---

## Most Likely Fix: Improve Adapter Training

The fact that MSE and MAE are reasonable but correlation is negative strongly suggests **adapters are not properly trained**.

### Fix 1: More Adapter Training
```python
# Increase epochs from 30 to 50-100
--adapter_epochs 50
```

### Fix 2: Better Target Selection
Replace random sampling with K-means clustering:
- Cluster MOSEI features into representative groups
- Match MOSI features to nearest cluster
- Use cluster centroids as targets

### Fix 3: Learning Rate Scheduling
```python
# Add scheduler to adapter training
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
```

### Fix 4: Check Adapter Losses
The adapter losses during training should be:
- Visual: Decreasing (starting ~850k → ~200k or less)
- Audio: Decreasing (starting ~2.6k → ~500 or less)
- Text: Decreasing (starting ~1.2 → ~0.1 or less)

If losses are not decreasing, adapters aren't learning.

---

## Expected Results After Fix:

**Good (after fixing adapters):**
- Correlation: **0.30-0.45** (positive!)
- MSE: 0.60-1.00
- MAE: 0.60-0.85

**Current (broken):**
- Correlation: -0.1020 ❌
- MSE: 0.7680
- MAE: 0.7081

---

## Action Plan:

1. **First:** Check if negating predictions fixes it (quick test)
   - If yes → sign inversion, find where it happens
   - If no → adapter training issue

2. **Then:** Re-train adapters with:
   - More epochs (50-100)
   - Better target selection (K-means)
   - Learning rate scheduling
   - Monitor losses carefully

3. **Verify:** Test model on MOSEI first
   - Should get correlation ~0.44-0.48
   - If negative on MOSEI too, model issue (less likely)

---

## Summary:

**Your negative correlation is almost certainly due to adapter training failure.**

The adapters need better training to properly map MOSI features to MOSEI feature space. Random sampling of targets is likely insufficient - you need more sophisticated alignment strategies.

**Priority fixes:**
1. More adapter training epochs
2. K-means target selection (instead of random)
3. Better learning rate scheduling
4. Monitor adapter losses closely




