# Corrected Section 4.2.1: Modality Encoders

## ❌ **CURRENT VERSION:**
The current version is mostly correct but could be more precise about input dimensions and clarify that all encoders share the same architecture pattern.

## ✅ **CORRECTED VERSION (Paragraph Form):**

### 4.2.1 Modality Encoders

Each modality (visual, audio, text) is processed through its respective encoder to transform input features into a unified 96-dimensional embedding space (embed_dim = 96). All three encoders share an identical architecture pattern, enabling consistent feature transformations across modalities while maintaining modality-specific input dimensions. Each encoder consists of three layers: an input layer that projects modality-specific features to an intermediate 192-dimensional hidden space, a hidden layer that refines the intermediate representation, and an output layer that projects to the unified 96-dimensional embedding space.

The input layer transforms modality-specific features to 192 dimensions using a linear transformation, followed by BatchNorm1d for normalization, ReLU activation for non-linearity, and Dropout (0.7) for regularization. The hidden layer maintains the 192-dimensional representation through another linear transformation with the same sequence of operations (BatchNorm1d, ReLU, Dropout 0.7), enabling the encoder to learn refined feature representations. The output layer projects from 192 to 96 dimensions using a linear transformation with BatchNorm1d, producing the final unified embedding representation. The high dropout rate (0.7) across all encoder layers provides strong regularization to prevent overfitting, which is particularly important for multimodal models with limited training data.

The encoders transform features as follows: the visual encoder processes 713-dimensional OpenFace2 features (or adapted FaceMesh features mapped to 713 dimensions) into 96-dimensional embeddings; the audio encoder processes 74-dimensional COVAREP features (or adapted Librosa features mapped to 74 dimensions) into 96-dimensional embeddings; and the text encoder processes 300-dimensional GloVe embeddings (or adapted BERT embeddings mapped to 300 dimensions) into 96-dimensional embeddings. During training on CMU-MOSEI, the encoders receive pre-extracted features (OpenFace2: 713-dim, COVAREP: 74-dim, GloVe: 300-dim). During testing on CMU-MOSI, the encoders receive feature-adapted representations: FaceMesh features (65-dim) are adapted to OpenFace2 space (713-dim) via the visual adapter, Librosa features (74-dim) pass through the audio adapter (74-dim, dimension-preserving), and BERT embeddings (768-dim) are adapted to GloVe space (300-dim) via the text adapter. This design enables the same encoder architecture to process both pre-extracted and adapted features, maintaining consistency across training and testing phases while enabling cross-dataset transfer learning.

The unified 96-dimensional output from all encoders ensures that features from different modalities occupy the same embedding space, facilitating effective cross-modal attention and fusion operations in subsequent layers. This architectural choice enables the model to learn meaningful inter-modal relationships by operating on representations of equal dimensionality, regardless of the original input feature dimensions or extraction methods.

---

## Alternative Shorter Version (if space is limited):

### 4.2.1 Modality Encoders

Each modality (visual, audio, text) is processed through its respective encoder to transform input features into a unified 96-dimensional embedding space (embed_dim = 96). All three encoders share an identical three-layer architecture pattern: Input Layer (Input_dim → 192), Hidden Layer (192 → 192), and Output Layer (192 → 96). Each layer includes BatchNorm1d, ReLU activation, and Dropout (0.7) except the output layer, which only includes BatchNorm1d. The encoders transform: Visual (713-dim OpenFace2 features → 96-dim), Audio (74-dim COVAREP features → 96-dim), and Text (300-dim GloVe embeddings → 96-dim). During testing on CMU-MOSI, adapted features are used: FaceMesh (65-dim) → Visual Adapter → 713-dim, Librosa (74-dim) → Audio Adapter → 74-dim, BERT (768-dim) → Text Adapter → 300-dim, before passing through the respective encoders. This unified architecture enables consistent feature transformations across modalities, facilitating effective cross-modal attention and fusion operations.




