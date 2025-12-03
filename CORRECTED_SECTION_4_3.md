# Corrected Section 4.3 - Data

## Issues Found:

1. **Table 1 Description**: Missing OpenFace2 for visual features
   - Current: "pre-extracted data using COVAREP, GLoVe, and text"
   - Should be: "pre-extracted features using OpenFace2 (visual), COVAREP (audio), and GloVe (text)"

2. **Table 2 Title**: Says "Train corr" but these are TEST results
   - Current: "The Average Train corr, MAE, and MSE"
   - Should be: "Test Correlation, MAE, and MSE" (these are from held-out test set)

3. **Table 2 Values**: All values are correct ✓
   - Text: -0.0604 ✓
   - Audio + Visual: 0.0214 ✓
   - Audio + Text: -0.0281 ✓
   - Text + Visual: 0.1128 ✓
   - Text + Visual + Audio: 0.6360 ✓

## Corrected Section 4.3:

**Table 1:** The Average Train Loss, Train MAE, Train Corr, Val Loss, Val MAE, and Val Corr for each modality combination. These are the results from training on CMU-MOSEI using pre-extracted features: OpenFace2 (visual), COVAREP (audio), and GloVe (text). These results serve to determine which modality combination has the best results on the source dataset.

[Table 1 values remain the same]

**Table 2:** Test Correlation, MAE, and MSE for each modality combination on CMU-MOSI. These are the results from transfer learning, where the model trained on CMU-MOSEI pre-extracted features was adapted to CMU-MOSI using feature adapters (FaceMesh, Librosa, BERT) and evaluated on a held-out test set (20% of CMU-MOSI samples). These results demonstrate the generalization capability of different modality combinations in the transfer learning setting.

[Table 2 values remain the same]

**Analysis:** The results demonstrate that single modalities and pairwise combinations achieve low or negative correlation when transferred to CMU-MOSI, while the full multimodal combination (Text + Visual + Audio) achieves 0.6360 correlation on the test set. This demonstrates the importance of combining all three modalities for robust transfer learning performance. The significant improvement from pairwise combinations (0.01-0.11) to the full combination (0.6360) highlights the synergistic effect of multimodal fusion in cross-domain sentiment analysis.




