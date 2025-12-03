# Metrics and Loss Function Validation

## Issues Found in Your Paper

### ✅ **CORRECT Parts:**

1. **MAE Description**: ✅ Correct - accurately describes what MAE measures
2. **Pearson Correlation Description**: ✅ Mostly correct, but see minor clarification below
3. **Formula for Pearson Correlation**: ✅ Correct
4. **Formula for MAE**: ✅ Correct
5. **Use of Metrics**: ✅ Correct - all three metrics (MAE, Correlation, Combined Loss) are used

### ❌ **INCORRECT Parts:**

#### 1. **Loss Function Formula - MISSING MAE Component**

**Your Paper Says:**
```
L_Combined = α·L_MSE + β·L_correlation
```

**Actual Implementation (Line 138 in train_mosei_only.py):**
```python
total_loss = self.alpha * (mse_loss + mae_loss) / 2 + self.beta * corr_loss
```

**Correct Formula Should Be:**
```
L_Combined = α·(L_MSE + L_MAE)/2 + β·L_correlation
```

**Issue**: Your paper is missing the MAE component! The actual loss function averages MSE and MAE, then multiplies by α.

#### 2. **Loss Function Weights**

**Your Paper**: Doesn't specify α and β values

**Actual Implementation**: 
- α = 0.3 (weight for MSE/MAE)
- β = 0.7 (weight for correlation)

This prioritizes correlation optimization over absolute accuracy, which is intentional for sentiment analysis.

#### 3. **Correlation Loss Computation**

**Your Paper**: Shows the Pearson correlation formula, but doesn't explain how it's converted to a loss

**Actual Implementation**:
- Correlation is computed using the formula you show ✅
- Loss is: `(1 - correlation)²` - squared loss for stronger gradient signal
- Uses mean-centered values for stability

#### 4. **Minor: Pearson Correlation Description**

**Your Paper Says**: 
> "where a positive value reflects higher predictions, with a negative value reflecting lower targets"

**Better Description**:
> "where a positive value indicates predictions increase with targets (positive correlation), and a negative value indicates predictions decrease as targets increase (negative correlation)"

The current wording "higher predictions with lower targets" is confusing. Correlation measures the direction of the relationship, not absolute values.

---

## Corrected Text for Your Paper

### Section: Loss Function (Correction Needed)

**Current (INCORRECT):**
> To guide learning, I used a combined loss function that jointly optimizes prediction accuracy and rank consistency. By using Mean Squared Error (MSE) and Pearson correlation coefficient, it ensures that the model produces both accurate values and maintains the correct relative ordering of sentiment intensities.
> 
> L_Combined = α·L_MSE + β·L_correlation

**Corrected (SHOULD BE):**
> To guide learning, I used a combined loss function that jointly optimizes prediction accuracy and rank consistency. The loss function combines Mean Squared Error (MSE), Mean Absolute Error (MAE), and Pearson correlation coefficient to ensure that the model produces both accurate values and maintains the correct relative ordering of sentiment intensities. Specifically, we average MSE and MAE to provide balanced absolute error signals, then combine this with correlation loss to emphasize rank consistency:
> 
> L_Combined = α·(L_MSE + L_MAE)/2 + β·L_correlation
> 
> where α = 0.3 and β = 0.7, prioritizing correlation optimization while maintaining reasonable absolute accuracy. The correlation loss is computed as (1 - r)², where r is the Pearson correlation coefficient, providing a stronger gradient signal for correlation improvements compared to linear correlation loss.

---

## Summary

| Component | Your Paper | Implementation | Status |
|-----------|------------|----------------|--------|
| **MAE Formula** | ✅ Correct | ✅ Matches | ✅ Correct |
| **Correlation Formula** | ✅ Correct | ✅ Matches | ✅ Correct |
| **Loss Function Formula** | ❌ Missing MAE | Uses (MSE+MAE)/2 | ❌ Needs Fix |
| **Loss Weights (α, β)** | ❌ Not specified | α=0.3, β=0.7 | ❌ Should Add |
| **Correlation Loss Detail** | ❌ Not explained | (1-r)² | ❌ Should Add |
| **MAE Description** | ✅ Correct | ✅ Matches | ✅ Correct |
| **Correlation Description** | 🟡 Minor issue | ✅ Matches | 🟡 Minor Fix |

---

## Recommended Changes

1. **Fix Loss Function Formula**: Add MAE component
2. **Specify Weights**: Mention α=0.3, β=0.7
3. **Clarify Correlation Description**: Fix the "higher predictions with lower targets" wording
4. **Add Correlation Loss Detail**: Explain that it's (1-r)², not just 1-r




