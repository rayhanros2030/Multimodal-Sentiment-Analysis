# Corrected Section 3.7: Loss Function

## ❌ **CURRENT (WRONG) VERSION:**
Section 3.7 "Multitask Loss Weighting" describes multi-task learning with classification, which doesn't exist in your implementation.

---

## ✅ **CORRECTED VERSION (Paragraph Form):**

### 3.7 Loss Function

The model is optimized using an improved correlation loss function that jointly minimizes regression error and maximizes Pearson correlation. The loss function addresses a key challenge in sentiment analysis: achieving both accurate absolute predictions and correct relative ordering of sentiment intensities. While Mean Squared Error (MSE) and Mean Absolute Error (MAE) measure prediction accuracy in absolute terms, they do not explicitly optimize for correlation, which captures whether the model correctly ranks sentiment intensities even when absolute values are imperfect. To address this, we employ a weighted combination that balances absolute accuracy with rank consistency:

L = α × (MSE + MAE)/2 + β × (1 - r)²

where α = 0.3 (accuracy weight), β = 0.7 (correlation weight), and r represents the Pearson correlation coefficient between predictions and ground truth labels. The squared correlation term (1 - r)² provides stronger gradient signals for correlation improvements compared to linear correlation loss, as small correlation increases near r = 1 produce larger gradient magnitudes that encourage the model to push correlation closer to perfect alignment. This weighting scheme prioritizes correlation optimization (β = 0.7) while maintaining reasonable absolute accuracy (α = 0.3), reflecting the importance of relative sentiment ordering in applications where understanding whether one sample is more positive than another is often more valuable than achieving perfect absolute predictions. The combined loss enables the model to learn both precise sentiment magnitudes and correct sentiment rankings throughout training, leading to improved performance on both correlation and absolute error metrics compared to using MSE or MAE alone.

---

## Alternative Shorter Version (if space is limited):

### 3.7 Loss Function

The model is optimized using a correlation-enhanced loss function that jointly minimizes regression error and maximizes Pearson correlation. The loss combines Mean Squared Error (MSE), Mean Absolute Error (MAE), and correlation optimization:

L = α × (MSE + MAE)/2 + β × (1 - r)²

where α = 0.3 (accuracy weight), β = 0.7 (correlation weight), and r is the Pearson correlation coefficient. This weighting scheme prioritizes correlation optimization while maintaining reasonable absolute accuracy. The squared correlation term (1 - r)² provides stronger gradient signals for correlation improvements, enabling the model to learn both precise sentiment magnitudes and correct sentiment rankings.




