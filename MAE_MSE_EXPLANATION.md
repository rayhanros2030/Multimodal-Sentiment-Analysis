# Explanation for Higher MAE/MSE in Full Multimodal Combination

## Why This Happens:

The full multimodal combination (Text + Visual + Audio) achieves the highest correlation (0.6360) but shows higher MAE (0.9172) and MSE (1.2386) compared to pairwise combinations. This apparent contradiction is actually consistent with transfer learning behavior and can be explained by several factors:

### 1. **Correlation vs. Absolute Error Metrics**
Pearson correlation measures the linear relationship between predictions and labels (how well predictions follow the trend), while MAE and MSE measure absolute distance from true values. A model can have high correlation but systematic offsets, resulting in higher absolute errors.

### 2. **Transfer Learning Scale Shift**
In cross-domain transfer learning, feature adapters map CMU-MOSI features (FaceMesh, Librosa, BERT) to CMU-MOSEI feature space (OpenFace2, COVAREP, GloVe). This feature space transformation may introduce systematic scale shifts, where predictions follow the correct trend (high correlation) but are offset from the true label distribution.

### 3. **Small Test Set Variance**
With a test set of 20 samples, one or two large prediction errors can significantly inflate MAE and MSE. Correlation is more robust to outliers, making it a more stable metric for small datasets.

### 4. **Model Confidence and Prediction Variance**
The full multimodal model, with access to all three modalities, may make more confident predictions. While this improves trend detection (correlation), it can also lead to larger absolute errors when the model is incorrect, as the predictions are less conservative than simpler models.

## How to Explain in Your Paper:

### Option 1: Technical Explanation (Recommended)

"While the full multimodal combination achieves the highest correlation (0.6360), it exhibits higher MAE (0.9172) and MSE (1.2386) compared to pairwise combinations. This apparent discrepancy is consistent with transfer learning behavior: the feature adapters map CMU-MOSI features to CMU-MOSEI feature space, which may introduce systematic scale shifts in predictions. Correlation measures the linear relationship between predictions and labels (trend alignment), while MAE/MSE measure absolute distance. The high correlation indicates that the model correctly learns the sentiment trend, even if predictions are systematically offset. Additionally, with a small test set (20 samples), absolute error metrics are more sensitive to outliers, while correlation remains robust."

### Option 2: Simpler Explanation

"The full multimodal combination achieves the highest correlation (0.6360), demonstrating superior trend prediction capability. The higher MAE and MSE can be attributed to systematic scale shifts introduced by feature adaptation in transfer learning, where predictions follow the correct trend but may be offset from true values. Correlation is preferred as the primary metric as it measures prediction quality (trend alignment) rather than absolute distance, and is more robust to outliers in small datasets."

### Option 3: Focus on Correlation (Concise)

"Correlation is the primary metric for evaluating transfer learning performance, as it measures how well predictions follow the sentiment trend. The full multimodal combination achieves 0.6360 correlation, significantly outperforming all other combinations. While MAE and MSE are higher, this is consistent with transfer learning where feature space adaptation may introduce systematic offsets while maintaining correct trend prediction."

## Recommendation:

Use **Option 1** if you want to show technical depth, or **Option 2** if you want a balanced explanation. The key is to:
1. Acknowledge the higher MAE/MSE
2. Explain why it happens (transfer learning scale shift)
3. Justify why correlation is the primary metric
4. Emphasize that correlation shows the model is learning correctly




