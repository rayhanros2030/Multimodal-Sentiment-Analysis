# Corrected Abstract

Understanding human emotion through multimodal cues is a key challenge in mental-health computing. This project introduces a transfer learning framework for multimodal sentiment analysis that enables cross-dataset generalization through feature adaptation. The system trains on CMU-MOSEI using pre-extracted features (OpenFace2 for visual, COVAREP for audio, GloVe for text) and tests on CMU-MOSI using real-time extracted features (MediaPipe FaceMesh for visual, Librosa for audio, BERT for text). To bridge the gap between different feature extraction paradigms, the framework employs feature adapter networks that learn mappings between feature spaces: FaceMesh visual features (65-dim) are adapted to OpenFace2 space (713-dim), BERT text embeddings (768-dim) are adapted to GloVe space (300-dim), and Librosa audio features (74-dim) are adapted to COVAREP space (74-dim). Critically, all features are aggregated (temporally averaged) into single vector representations before model input, enabling efficient processing without temporal sequence modeling. The extracted features are processed through modality-specific encoders that transform them into a unified 96-dimensional representation. These unified features are then fused through a cross-modal architecture that employs MultiheadAttention with 4 heads to model bidirectional interactions between modalities, followed by concatenation and a multi-layer perceptron (MLP) fusion network. The system predicts continuous sentiment scores in the range [-3, +3] through regression, optimized using a correlation-enhanced loss function that jointly minimizes mean squared error (MSE) and mean absolute error (MAE) while maximizing Pearson correlation. Evaluations demonstrate that the proposed feature adaptation strategy enables effective transfer learning between datasets with different feature extraction pipelines, improving sentiment prediction correlation and reducing mean absolute error compared to baseline approaches. This work contributes toward cross-domain emotion understanding that may support future systems for psychological assessment and human-computer interaction.

---

## Issues Fixed:

1. ❌ "Understanding emotion dynamically" → ✅ Removed "dynamically", clarified feature aggregation
2. ❌ "how emotion evolves" → ✅ Removed temporal evolution claims
3. ❌ "IEMOCAP, CMU-MOSEI, and CMU-MOSI datasets" → ✅ Only MOSEI (training) and MOSI (testing), no IEMOCAP
4. ❌ "temporally aligned" → ✅ Features are aggregated/averaged, no temporal alignment
5. ❌ "pairwise cross-attention and self-attention refinement" → ✅ MultiheadAttention (4 heads) + concatenation + MLP
6. ❌ "Bidirectional LSTM decoder" → ✅ MLP fusion network (no LSTM)
7. ❌ "multi-task loss" / "categorical emotion classes" → ✅ Regression only, correlation-optimized loss
8. ❌ Missing transfer learning → ✅ Added feature adaptation and transfer learning as main contribution




