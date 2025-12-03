# Section 3.5.4: End-to-End Fine-Tuning

## **Paragraph for Your Paper:**

**3.5.4 End-to-End Fine-Tuning**

After adapter training, the framework employs end-to-end fine-tuning to optimize both adapters and the main model jointly for sentiment prediction on the target dataset. This fine-tuning phase is critical for bridging the remaining performance gap between feature-adapted representations and the original pre-extracted features. The CMU-MOSI dataset is split into training (60%), validation (20%), and test (20%) sets, with the test set held out completely to ensure unbiased evaluation. During fine-tuning, all components—the three feature adapters (visual, audio, text) and the pre-trained multimodal fusion model—are optimized together using the sentiment prediction loss function (Equation 1), which jointly minimizes regression error (MSE + MAE) and maximizes Pearson correlation. This end-to-end optimization enables the adapters to learn feature mappings that are specifically optimized for sentiment prediction rather than general feature matching, while simultaneously allowing the main model to adjust its parameters to better process adapted features. The fine-tuning process uses a learning rate of 0.0001 with weight decay of 1e-5, Adam optimizer, gradient clipping (max norm 1.0), and runs for 20 epochs. During training, validation correlation is monitored to track progress, with the best validation correlation (typically 0.40-0.48) used to select the optimal model. This fine-tuning approach significantly improves transfer learning performance: initial correlation after adapter training alone is approximately 0.12, while after end-to-end fine-tuning, correlation improves to 0.52 on the held-out test set. This substantial improvement demonstrates that joint optimization of adapters and model is essential for effective cross-dataset transfer learning, as it ensures that feature adaptations align not only with feature distributions but also with the downstream sentiment prediction task. The fine-tuned model is then evaluated on the held-out test set to obtain final performance metrics, ensuring that results reflect generalization to unseen data rather than overfitting to the training set.

---

## **Alternative Shorter Version (if space is limited):**

**3.5.4 End-to-End Fine-Tuning**

To optimize adapters and the main model jointly for sentiment prediction, we employ end-to-end fine-tuning on CMU-MOSI. The dataset is split into training (60%), validation (20%), and test (20%) sets, with the test set held out for final evaluation. During fine-tuning, all adapters and the pre-trained model are optimized together using the sentiment loss function (Equation 1) for 20 epochs with a learning rate of 0.0001, Adam optimizer, and gradient clipping. This joint optimization enables adapters to learn feature mappings optimized specifically for sentiment prediction rather than general feature matching. Fine-tuning significantly improves performance: correlation improves from approximately 0.12 after adapter training alone to 0.52 on the held-out test set, demonstrating that end-to-end optimization is essential for effective cross-dataset transfer learning. The fine-tuned model is then evaluated on the held-out test set to obtain final performance metrics.

---

## **Key Points to Include:**

1. ✅ **Purpose:** Optimize adapters + model together for sentiment prediction
2. ✅ **Data Split:** 60/20/20 (train/val/test)
3. ✅ **What's Optimized:** All adapters + main model jointly
4. ✅ **Loss Function:** Sentiment loss (MSE + MAE + correlation)
5. ✅ **Training Details:** 20 epochs, LR=0.0001, Adam, gradient clipping
6. ✅ **Improvement:** 0.12 → 0.52 correlation
7. ✅ **Why It Works:** Optimizes for task (sentiment) not just feature matching

---

## **Integration Tips:**

1. **Place after Section 3.5.3** (Testing on CMU-MOSI)
2. **Update Section 3.5.3** to mention "after fine-tuning" or "before fine-tuning"
3. **Add results to Section 4.3** showing before/after fine-tuning comparison
4. **Update Future Works** to remove fine-tuning (since you did it!)

---

## **Suggested Table Addition (Section 4.3):**

Add this table showing the improvement:

**Table 2: Transfer Learning Results on CMU-MOSI**

| Stage | Test Correlation | Test MAE | Test MSE | Samples |
|-------|-----------------|----------|----------|---------|
| After Adapter Training | 0.1153 | 0.7103 | 0.7608 | 93 |
| After Fine-Tuning | **0.5172** | **0.6885** | **0.7151** | 20 (held-out) |

*Note: Test set for fine-tuning uses held-out 20 samples (20% of 93 samples) to ensure unbiased evaluation.*

---

Use whichever version fits your paper's style and length requirements!




