# Metrics and Loss Function - Corrected Paragraph Form

## Corrected Section for Your Paper

To validate that the detection of emotion through the three modalities is truly viable, we employ multiple evaluation metrics: Mean Absolute Error (MAE), Pearson Correlation Coefficient, and a combined loss function that guides model training. The Pearson Correlation Coefficient measures the strength and direction of the linear relationship between predictions and ground truth labels, ranging from -1 to +1. A positive correlation indicates that predictions increase with targets (positive relationship), while a negative correlation indicates that predictions decrease as targets increase (inverse relationship). In the context of sentiment analysis, we compare model predictions with human-annotated sentiment labels across the test set using the formula:

\[r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i-\bar{x})^2(y_i-\bar{y})^2}}\]

The Mean Absolute Error (MAE) measures the average magnitude of prediction errors in the same units as the target variable, providing an interpretable measure of how far predictions deviate from true values. MAE is preferred over other error metrics because it is immediately interpretable, treats errors linearly (outliers do not dominate the metric), and provides a balanced view of typical performance. MAE is complementary to correlation because correlation measures the rank or order similarity between predictions and targets, whereas MAE measures actual error magnitude. Together, they provide a comprehensive picture of model performance.

To guide learning, we use a combined loss function that jointly optimizes prediction accuracy and rank consistency. The loss function combines Mean Squared Error (MSE), Mean Absolute Error (MAE), and Pearson correlation coefficient to ensure that the model produces both accurate values and maintains the correct relative ordering of sentiment intensities. Specifically, we average MSE and MAE to provide balanced absolute error signals, then combine this with correlation loss to emphasize rank consistency:

\[L_{Combined} = \alpha \cdot \frac{L_{MSE} + L_{MAE}}{2} + \beta \cdot L_{correlation}\]

where \(\alpha = 0.3\) and \(\beta = 0.7\), prioritizing correlation optimization while maintaining reasonable absolute accuracy. The correlation loss is computed as \((1 - r)^2\), where \(r\) is the Pearson correlation coefficient, providing a stronger gradient signal for correlation improvements compared to linear correlation loss. This weighting scheme ensures that both absolute accuracy (through MSE and MAE) and relative ordering (through correlation) meaningfully influence parameter updates throughout training, enabling the model to capture both precise sentiment magnitudes and correct sentiment rankings.

---

## Key Changes Made:

1. ✅ **Added MAE to loss function**: Now includes \(\frac{L_{MSE} + L_{MAE}}{2}\)
2. ✅ **Specified weights**: \(\alpha = 0.3\) and \(\beta = 0.7\)
3. ✅ **Clarified correlation description**: Fixed "higher predictions with lower targets" confusion
4. ✅ **Added correlation loss detail**: Explained \((1 - r)^2\) computation
5. ✅ **Fixed grammar**: "modalities are" → "modalities is" (subject-verb agreement)




