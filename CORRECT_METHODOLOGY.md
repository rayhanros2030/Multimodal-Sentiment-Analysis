# Complete Methodology - Accurate to Implementation

## 3. Methodology

This section describes the complete methodology for our multimodal sentiment analysis framework, including dataset preparation, feature extraction, transfer learning approach, model architecture, and training procedures.

---

## 3.1 Dataset

This study utilizes the CMU-MOSEI and CMU-MOSI datasets to train and evaluate a transfer learning framework for multimodal sentiment analysis. The approach employs a novel feature adaptation strategy where models are trained on CMU-MOSEI using pre-extracted features, then adapted to work with real-time feature extractors on CMU-MOSI through learned feature space mappings, enabling cross-dataset generalization with minimal model modification.

### 3.1.1 CMU-MOSEI Dataset

CMU-MOSEI (Multimodal Opinion Sentiment and Emotion Intensity) is one of the largest multimodal sentiment analysis corpora, comprising over 23,500 annotated video segments drawn from 1,000 distinct speakers discussing more than 250 topics. Each clip captures diverse recording conditions—varying in camera distance, lighting, and background—reflecting natural, in-the-wild expression patterns. The dataset provides continuous sentiment labels ranging from -3 (strongly negative) to +3 (strongly positive), enabling fine-grained analysis of sentiment intensity.

For CMU-MOSEI, we utilize the provided pre-extracted features:
- **Visual features**: OpenFace2 (713-dimensional) - facial action units, pose, gaze, and appearance features
- **Audio features**: COVAREP (74-dimensional) - prosodic, spectral, and cepstral features
- **Text features**: GloVe embeddings (300-dimensional) - word-level semantic representations

We employ a standard 70/15/15 train/validation/test partition for CMU-MOSEI. These pre-extracted features serve as the training data for our base multimodal fusion model and establish the target feature space for our feature adaptation framework.

### 3.1.2 CMU-MOSI Dataset

CMU-MOSI (Multimodal Opinion-Level Sentiment Intensity) is a widely used multimodal dataset for sentiment analysis, consisting of 2,199 video segments from 93 YouTube movie review videos. Each segment is annotated for sentiment intensity, subjectivity, and various audio and visual features. Similar to CMU-MOSEI, CMU-MOSI contains continuous sentiment labels from -3 to +3 and exhibits diverse recording conditions typical of in-the-wild videos.

Unlike CMU-MOSEI, CMU-MOSI requires real-time feature extraction from raw video, audio, and text data. We extract features using:
- **Visual**: MediaPipe FaceMesh (real-time processing)
- **Audio**: Librosa (real-time audio analysis)
- **Text**: BERT (contextual text embeddings)

CMU-MOSI serves as the target domain for our transfer learning evaluation. We apply the same 70/15/15 train/validation/test partition to CMU-MOSI for consistency.

### 3.1.3 Feature Adaptation Mechanism

To bridge the feature space gap between pre-extracted (MOSEI) and real-time (MOSI) features, we employ neural feature adapter networks. Specifically, we train three separate two-layer feedforward networks:

1. **Visual Adapter**: Maps FaceMesh features (65 dimensions) to OpenFace2-compatible representations (713 dimensions)
2. **Audio Adapter**: Maps Librosa features (74 dimensions) to COVAREP-compatible features (74 dimensions)  
3. **Text Adapter**: Maps BERT embeddings (768 dimensions) to GloVe-compatible vectors (300 dimensions)

Each adapter is trained to minimize mean squared error between adapted MOSI features and randomly sampled MOSEI target features, learning to align feature distributions across domains. The adapters enable the pre-trained MOSEI model to process real-time extracted features without requiring retraining of the base model, representing a form of feature space adaptation that preserves sentiment-relevant information while transforming dimensions and aligning distributions.

---

## 3.2 Visual Feature Extraction

For visual feature extraction, we process video frames using MediaPipe FaceMesh to extract 468 3D facial landmarks per frame. To handle variable-length videos efficiently, we process frames sequentially from the start of each video, processing up to the first 100 frames per video (or all frames if the video contains fewer than 100 frames). This frame-based processing strategy ensures consistent feature extraction across videos of different lengths while maintaining computational efficiency.

For each frame, we derive 65-dimensional emotion-focused features through geometric computations, with the face first normalized by face width (computed as the Euclidean distance between landmarks at indices 0 and 16) to handle scale variation across videos and frames.

### 3.2.1 Feature Derivation

We extract 12 explicitly defined emotion-relevant features:

**Mouth characteristics (5 features)**:
- Mouth width (Euclidean distance between landmarks 61 and 291)
- Mouth height (distance between landmarks 13 and 14)
- Left mouth corner Y-coordinate (landmark 61)
- Right mouth corner Y-coordinate (landmark 291)
- Mouth corner angle (computed using arctangent)

**Eye features (3 features)**:
- Left eye width (distance between landmarks 33 and 133)
- Right eye width (distance between landmarks 362 and 263)
- Inter-eye distance (distance between landmarks 33 and 263)

**Eyebrow features (2 features)**:
- Left eyebrow height (average of landmarks 21, 55, and 107)
- Right eyebrow height (average of landmarks 251, 285, and 336)

**Symmetry metrics (2 features)**:
- Eye symmetry (normalized absolute difference between left and right eye widths)
- Mouth symmetry (absolute difference between left and right corner Y-coordinates)

The remaining 53 features are derived from normalized landmark magnitudes: we compute the Euclidean norm (L2 norm) of normalized landmark coordinates at indices 12 through 64 (inclusive), providing additional geometric information about facial structure relative to face scale.

### 3.2.2 Temporal Aggregation and Encoding

This yields exactly 65 features per frame: [1 frame × 65 features]. Features are extracted at the frame level (one 65-dimensional vector per frame) and then temporally averaged across all processed frames (up to 100, or all frames for shorter videos) to obtain a single video-level representation: [1 video × 65 features].

This video-level feature vector is passed through the visual encoder, which consists of two linear transformations:
- **Layer 1**: 65 → 192 dimensions (with BatchNorm1d, ReLU activation, Dropout 0.7)
- **Layer 2**: 192 → 192 dimensions (with BatchNorm1d, ReLU activation, Dropout 0.7)
- **Output Layer**: 192 → 96 dimensions (with BatchNorm1d)

We employ dropout of 0.7 in all encoder and fusion layers to provide strong regularization and prevent overfitting, which is particularly important for multimodal models with limited training data. The encoder produces a 96-dimensional representation (embed_dim = 96), matching the text and audio encoder output dimensions. This encoded representation is then used in cross-modal attention and fusion.

---

## 3.3 Audio Feature Extraction

To represent the vocal dimension of emotion, we extract acoustic prosody features using Librosa, a Python library for audio and music signal processing that provides standardized tools for converting raw speech waveforms into quantitative descriptors of prosody and timbre.

### 3.3.1 Audio Processing

Each audio track is processed using Librosa's default parameters optimized for spectral analysis. We load audio at a sampling rate of 22.05 kHz with a maximum duration of 3.0 seconds to balance computational efficiency with temporal coverage. Librosa uses default Short-Time Fourier Transform (STFT) parameters:
- **Frame size (n_fft)**: 2048 samples (93 ms at 22.05 kHz)
- **Hop length**: 512 samples (23 ms at 22.05 kHz)
- **Window**: Hann window with 75% overlap

Within each 3-second audio segment, we compute frame-level acoustic descriptors and then temporally average across all frames to obtain a single vector representation per audio sample.

### 3.3.2 Feature Extraction

Specifically, we extract the following features:
- **Mel-frequency cepstral coefficients (MFCCs)**: 13 coefficients capturing spectral envelope characteristics that reflect vocal tract shape
- **Chroma features**: 12 coefficients representing pitch class distribution, which implicitly capture fundamental frequency patterns
- **Spectral centroid**: 1 coefficient indicating the "brightness" or spectral center of mass
- **Spectral rolloff**: 1 coefficient marking the frequency below which a specified percentage of spectral energy is contained
- **Zero-crossing rate (ZCR)**: 1 coefficient indicating voice quality and voicing characteristics
- **Tempo**: 1 coefficient capturing rhythmic characteristics

This yields 29 frame-level features per frame. Temporal averaging (mean) is applied across all frames within the 3-second segment, reducing the representation from [29 features × multiple frames] to a fixed-size vector of 29 features per audio sample.

To ensure compatibility with the CMU-MOSEI dataset's COVAREP audio feature format (74 dimensions), the feature vector is zero-padded to 74 dimensions.

### 3.3.3 Audio Encoding

These features capture how speech energy, pitch variation, and spectral shape fluctuate with emotion. For example, higher pitch (reflected in chroma distributions) and spectral centroid values often correspond to excitement or anger, while lower, flatter contours indicate sadness or fatigue.

The 74-dimensional audio features are passed through the audio encoder, which consists of two linear transformations:
- **Layer 1**: 74 → 192 dimensions (with BatchNorm1d, ReLU activation, Dropout 0.7)
- **Layer 2**: 192 → 192 dimensions (with BatchNorm1d, ReLU activation, Dropout 0.7)
- **Output Layer**: 192 → 96 dimensions (with BatchNorm1d)

The encoder produces a 96-dimensional representation (embed_dim = 96) that is used in cross-modal attention and fusion.

---

## 3.4 Text Feature Extraction

To represent the linguistic dimension of emotion, we extract contextualized semantic features from transcripts using BERT-base-uncased, a bidirectional transformer model pretrained on large-scale English corpora.

### 3.4.1 BERT Model Specifications

BERT-base-uncased consists of:
- **12 transformer layers** with 768-dimensional hidden states
- **12 attention heads** per layer
- **Vocabulary size**: 30,522 tokens
- **Maximum sequence length**: 512 tokens
- **Model size**: ~110M parameters

### 3.4.2 Tokenization and Encoding

Transcripts are processed through the BERT-base-uncased tokenizer, which splits text into subword tokens using WordPiece tokenization. Special tokens are automatically added:
- **[CLS]**: Added at the start of each sequence (token ID=101)
- **[SEP]**: Added between sentences or at sequence end (token ID=102)
- **[PAD]**: Added for sequences shorter than 512 tokens (token ID=0)

We set a maximum sequence length of 512 tokens, with shorter sequences padded using [PAD] tokens and longer sequences truncated from the right. The tokenizer produces token ID sequences and an attention mask, where 1 indicates real tokens and 0 indicates padding tokens.

### 3.4.3 Embedding Extraction

We use BERT as a frozen encoder (model.eval() with requires_grad=False for all parameters), leveraging pretrained representations without task-specific fine-tuning to maintain generalization and reduce overfitting. The model processes tokenized input to produce hidden states of shape [batch_size, sequence_length, 768], where each token is represented by a 768-dimensional vector.

To obtain a single fixed-size representation per transcript, we apply mean pooling over the sequence dimension, computing the average of all token embeddings. This yields a 768-dimensional vector capturing the overall semantic content of the transcript.

**Note**: While masked mean pooling (excluding padding tokens) is recommended, our current implementation uses simple mean pooling. Future improvements should incorporate the attention mask to exclude padding tokens from the average.

### 3.4.4 Feature Adaptation for Transfer Learning

For transfer learning compatibility with CMU-MOSEI's GloVe embeddings (300 dimensions), the 768-dimensional BERT embeddings are passed through a neural feature adapter network. The adapter is a two-layer feedforward network with the architecture:
- **Input Layer**: Linear(768 → 384) → BatchNorm1d → ReLU → Dropout(0.3)
- **Hidden Layer**: Linear(384 → 384) → BatchNorm1d → ReLU → Dropout(0.3)
- **Output Layer**: Linear(384 → 300)

The adapter is trained to minimize mean squared error between adapted BERT embeddings and randomly sampled GloVe target features from CMU-MOSEI, learning to map BERT embeddings to GloVe-compatible representations.

### 3.4.5 Text Encoding

The adapted text features (300 dimensions) are then processed through a text encoder consisting of two linear transformations:
- **Layer 1**: 300 → 192 dimensions (with BatchNorm1d, ReLU activation, Dropout 0.7)
- **Layer 2**: 192 → 192 dimensions (with BatchNorm1d, ReLU activation, Dropout 0.7)
- **Output Layer**: 192 → 96 dimensions (with BatchNorm1d)

This encoder produces the final text representation of dimension 96 (embed_dim), matching the visual and audio encoder output dimensions. The encoder learns to project text features into a shared semantic space suitable for cross-modal attention and fusion.

---

## 3.5 Model Architecture

The proposed multimodal sentiment analysis framework processes three input modalities through a hierarchical architecture, transforming extracted features into unified sentiment predictions. The architecture consists of four main components: modality encoders, cross-modal attention, fusion layers, and prediction head.

### 3.5.1 Modality Encoders

Each modality (visual, audio, text) is processed through its respective encoder to transform input features into a unified 96-dimensional embedding space (embed_dim = 96). All encoders share the same architecture pattern:
- **Input Layer**: Input_dim → 192 dimensions (Linear, BatchNorm1d, ReLU, Dropout 0.7)
- **Hidden Layer**: 192 → 192 dimensions (Linear, BatchNorm1d, ReLU, Dropout 0.7)
- **Output Layer**: 192 → 96 dimensions (Linear, BatchNorm1d)

The encoders transform:
- **Visual**: 65-dimensional FaceMesh features → 96-dimensional embeddings
- **Audio**: 74-dimensional Librosa features → 96-dimensional embeddings
- **Text**: 300-dimensional GloVe-compatible features → 96-dimensional embeddings

### 3.5.2 Cross-Modal Attention

Cross-modal attention mechanisms enable each modality to dynamically attend to relevant information from the other modalities. For three modalities (Visual, Audio, Text), we compute bidirectional attention between all pairs using MultiheadAttention with:
- **Number of heads**: 4
- **Embedding dimension**: 96 (embed_dim)
- **Dropout**: 0.8 (min(dropout + 0.1, 0.8) where dropout=0.7)

The encoded features are stacked along a new dimension: [v_enc, a_enc, t_enc] → [batch_size, 3, 96]. Cross-attention is applied as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

where queries (Q), keys (K), and values (V) are derived from the stacked feature representations. This produces attended features that capture inter-modal relationships.

### 3.5.3 Fusion Layers

Following cross-modal attention, the three encoded modalities are concatenated along the feature dimension to form a unified representation:

**Concatenated Features**: [batch_size, 288] (96 × 3 modalities)

This concatenated representation is passed through a three-layer fusion network:
- **Layer 1**: Linear(288 → 192) → BatchNorm1d → ReLU → Dropout(0.7)
- **Layer 2**: Linear(192 → 96) → BatchNorm1d → ReLU → Dropout(0.7)
- **Layer 3**: Linear(96 → 1)

The fusion network compresses the multimodal representation while preserving sentiment-relevant information from all three modalities.

### 3.5.4 Prediction

The final layer produces a single scalar output representing the predicted sentiment score in the range [-3, +3]. The output is used directly for regression-based sentiment prediction without additional activation functions, allowing the model to learn the full range of sentiment intensity.

---

## 3.6 Transfer Learning Framework

The transfer learning framework operates in three sequential phases:

### 3.6.1 Phase 1: Training on CMU-MOSEI

In the first phase, we train the base multimodal fusion model on CMU-MOSEI using pre-extracted features:
- **Visual**: OpenFace2 features (713-dimensional)
- **Audio**: COVAREP features (74-dimensional)
- **Text**: GloVe embeddings (300-dimensional)

The model is trained for 100 epochs with early stopping (patience=25) based on validation correlation. The training uses:
- **Batch size**: 32
- **Learning rate**: 0.0008
- **Weight decay**: 0.04
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau (factor=0.7, patience=7, mode='max')

The best model (based on validation correlation) is saved for use in subsequent phases.

### 3.6.2 Phase 2: Training Feature Adapters

In the second phase, we train three feature adapter networks to map real-time extracted features (from CMU-MOSI) to the pre-extracted feature space (from CMU-MOSEI). Each adapter is trained independently using mean squared error loss:

- **Visual Adapter**: Maps FaceMesh features (65-dim) → OpenFace2 features (713-dim)
  - Architecture: Linear(65 → 512) → BatchNorm → ReLU → Dropout(0.3) → Linear(512 → 512) → BatchNorm → ReLU → Dropout(0.3) → Linear(512 → 713)
  
- **Audio Adapter**: Maps Librosa features (74-dim) → COVAREP features (74-dim)
  - Architecture: Linear(74 → 256) → BatchNorm → ReLU → Dropout(0.3) → Linear(256 → 256) → BatchNorm → ReLU → Dropout(0.3) → Linear(256 → 74)

- **Text Adapter**: Maps BERT embeddings (768-dim) → GloVe embeddings (300-dim)
  - Architecture: Linear(768 → 384) → BatchNorm → ReLU → Dropout(0.3) → Linear(384 → 384) → BatchNorm → ReLU → Dropout(0.3) → Linear(384 → 300)

Training parameters:
- **Batch size**: 16
- **Learning rate**: 0.001
- **Optimizer**: Adam
- **Epochs**: 30
- **Loss**: Mean Squared Error (MSE)

During training, target features are randomly sampled from CMU-MOSEI data (1000 samples) and paired with extracted CMU-MOSI features to minimize the reconstruction error.

### 3.6.3 Phase 3: Testing on CMU-MOSI

In the third phase, we evaluate the transfer learning framework by:
1. Extracting features from CMU-MOSI using real-time extractors (FaceMesh, Librosa, BERT)
2. Adapting features using the trained adapter networks
3. Processing adapted features through the pre-trained MOSEI model
4. Evaluating sentiment prediction performance on CMU-MOSI test set

This approach enables cross-dataset generalization while preserving the benefits of training on large-scale pre-extracted features.

---

## 3.7 Loss Function

We employ a combined loss function that jointly optimizes prediction accuracy and rank consistency, ensuring that the model produces both accurate values and maintains the correct relative ordering of sentiment intensities.

### 3.7.1 Improved Correlation Loss

The loss function combines three components:

**1. Mean Squared Error (MSE)**:
```
L_MSE = (1/n) Σ(pred_i - target_i)²
```

**2. Mean Absolute Error (MAE)**:
```
L_MAE = (1/n) Σ|pred_i - target_i|
```

**3. Pearson Correlation Loss**:
```
L_corr = 1 - correlation(pred, target)
```

where correlation is computed as:
```
correlation = Σ(pred_i - pred_mean)(target_i - target_mean) / 
              √[Σ(pred_i - pred_mean)² Σ(target_i - target_mean)²]
```

**Combined Loss**:
```
L_total = α · (L_MSE + L_MAE)/2 + β · L_corr
```

where:
- **α = 0.3**: Weight for MSE/MAE (emphasizing absolute accuracy)
- **β = 0.7**: Weight for correlation (emphasizing rank consistency)

This weighting scheme prioritizes correlation optimization (which measures how well predictions track the relative ordering of sentiment intensities) while maintaining reasonable absolute accuracy through MSE and MAE components.

### 3.7.2 Training Stability

The correlation loss uses mean-centered predictions and targets for numerical stability:
```
pred_centered = pred - pred.mean()
target_centered = target - target.mean()
```

This prevents division by zero and ensures stable gradient computation during backpropagation.

---

## 3.8 Training Procedure

### 3.8.1 Training Configuration

**Hyperparameters**:
- **Batch size**: 32
- **Learning rate**: 0.0008
- **Weight decay**: 0.04
- **Optimizer**: Adam
- **Epochs**: 100 (with early stopping)
- **Early stopping patience**: 25 epochs
- **Gradient clipping**: 0.5 (max norm)

**Learning Rate Scheduling**:
- **Scheduler**: ReduceLROnPlateau
- **Mode**: 'max' (monitoring validation correlation)
- **Factor**: 0.7 (multiply LR by 0.7 on plateau)
- **Patience**: 7 epochs

### 3.8.2 Regularization

To prevent overfitting and improve generalization:
- **Dropout**: 0.7 in all encoder and fusion layers
- **Batch Normalization**: Applied after every linear layer
- **Weight Decay**: 0.04 (L2 regularization)
- **Gradient Clipping**: Prevents gradient explosion

### 3.8.3 Data Preprocessing

**Feature Cleaning**:
- NaN and Inf values are replaced with 0.0, 1.0, or -1.0 (for posinf and neginf)
- Extreme values are clipped to [-1000, 1000] range
- Features are normalized using RobustScaler (fitted only on training data to prevent data leakage)

**Data Splits**:
- **Train**: 70% of data
- **Validation**: 15% of data
- **Test**: 15% of data
- Random split with fixed seed (42) for reproducibility

### 3.8.4 Evaluation Metrics

We evaluate model performance using three metrics:

**1. Pearson Correlation Coefficient**:
Measures the strength and direction of the linear relationship between predictions and ground truth. Ranges from -1 to +1, where +1 indicates perfect positive correlation.

**2. Mean Absolute Error (MAE)**:
Measures the average magnitude of prediction errors in the same units as the target variable. Provides an interpretable measure of typical prediction accuracy.

**3. Mean Squared Error (MSE)**:
Measures the average squared prediction error. More sensitive to outliers than MAE but provides stronger gradient signals for extreme errors.

---

## 3.9 Implementation Details

### 3.9.1 Software and Libraries

- **PyTorch**: Deep learning framework (version 1.13+)
- **Transformers (Hugging Face)**: BERT model and tokenizer
- **MediaPipe**: FaceMesh for facial landmark detection
- **Librosa**: Audio feature extraction
- **NumPy**: Numerical computations
- **scikit-learn**: RobustScaler for feature normalization
- **scipy**: Pearson correlation computation

### 3.9.2 Hardware

Training is performed on GPU when available (CUDA), with automatic fallback to CPU. The model architecture is designed to be computationally efficient, processing single video samples without requiring batch-level temporal alignment.

### 3.9.3 Computational Efficiency

- **Visual**: Up to 100 frames per video (typically 3-4 seconds at 30 fps)
- **Audio**: Single 3-second window per sample
- **Text**: Full transcript (up to 512 tokens)
- Temporal averaging reduces computational complexity compared to per-frame processing

---

## 3.10 Summary

This methodology presents a complete transfer learning framework for multimodal sentiment analysis that:

1. **Trains on large-scale pre-extracted features** (CMU-MOSEI) to leverage sophisticated feature representations
2. **Adapts to real-time feature extractors** (CMU-MOSI) through learned feature space mappings
3. **Uses hierarchical encoding and cross-modal attention** to capture inter-modal relationships
4. **Optimizes for correlation** while maintaining prediction accuracy through a combined loss function
5. **Achieves cross-dataset generalization** without requiring full model retraining

The approach addresses the practical challenge of deploying models trained on pre-extracted features to real-world scenarios requiring real-time processing, demonstrating successful cross-dataset transfer from CMU-MOSEI to CMU-MOSI through learned feature space adaptation.




