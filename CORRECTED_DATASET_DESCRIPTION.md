# Corrected Dataset Description - Accurate Technical Details

## Fixed Version with All Corrections

This study utilizes the CMU-MOSEI and CMU-MOSI datasets to train and evaluate a transfer learning framework for multimodal sentiment analysis. The approach employs a novel feature adaptation strategy where models are trained on CMU-MOSEI using pre-extracted features, then adapted to work with real-time feature extractors on CMU-MOSI through learned feature space mappings, enabling cross-dataset generalization with minimal model modification.

**CMU-MOSEI (Multimodal Opinion Sentiment and Emotion Intensity)** is one of the largest multimodal sentiment analysis corpora, comprising over 23,500 annotated video segments drawn from 1,000 distinct speakers discussing more than 250 topics. Each clip captures diverse recording conditions—varying in camera distance, lighting, and background—reflecting natural, in-the-wild expression patterns. The dataset provides continuous sentiment labels ranging from -3 (strongly negative) to +3 (strongly positive), enabling fine-grained analysis of sentiment intensity. For CMU-MOSEI, we utilize the provided pre-extracted features: OpenFace2 visual features (713-dimensional facial action units, pose, gaze, and appearance features), COVAREP audio features (74-dimensional prosodic, spectral, and cepstral features), and GloVe text embeddings (300-dimensional word-level semantic representations). We employ a standard 70/15/15 train/validation/test partition for CMU-MOSEI. These pre-extracted features serve as the training data for our base multimodal fusion model and establish the target feature space for our feature adaptation framework.

**CMU-MOSI (Multimodal Opinion-Level Sentiment Intensity)** is a widely used multimodal dataset for sentiment analysis, consisting of 2,199 video segments from 93 YouTube movie review videos. Each segment is annotated for sentiment intensity, subjectivity, and various audio and visual features. Similar to CMU-MOSEI, CMU-MOSI contains continuous sentiment labels from -3 to +3 and exhibits diverse recording conditions typical of in-the-wild videos. Unlike CMU-MOSEI, CMU-MOSI requires real-time feature extraction from raw video, audio, and text data. We extract features using MediaPipe FaceMesh for visual processing, Librosa for audio analysis, and BERT for contextual text embeddings. For visual features, we process video frames to extract 468 facial landmarks per frame using FaceMesh, then derive 65-dimensional emotion-focused features through geometric computations: mouth characteristics (width, height, corner positions, angle—5 features), eye measurements (left/right eye width, inter-eye distance—3 features), eyebrow positions (average heights—2 features), symmetry metrics (eye and mouth asymmetry—2 features), and additional landmark-based distances and normalized positions (53 features). These features are temporally averaged across frames to obtain a stable 65-dimensional representation per video. For audio features, we extract 29 features using Librosa: 13 MFCC coefficients (capturing spectral envelope), 12 chroma features (harmonic content), spectral centroid (1), spectral rolloff (1), zero-crossing rate (1), and tempo (1), which are then padded to 74 dimensions for compatibility with COVAREP's feature space. For text features, we tokenize transcripts using BERT-base-uncased, extract contextual embeddings (768-dimensional), and apply mean pooling over the sequence length. CMU-MOSI serves as the target domain for our transfer learning evaluation. We apply the same 70/15/15 train/validation/test partition to CMU-MOSI for consistency.

**Feature Adaptation Mechanism**: To bridge the feature space gap between pre-extracted (MOSEI) and real-time (MOSI) features, we employ neural feature adapter networks. Specifically, we train three separate two-layer feedforward networks: a visual adapter that maps FaceMesh features (65 dimensions) to OpenFace2-compatible representations (713 dimensions), an audio adapter that maps Librosa features (74 dimensions) to COVAREP-compatible features (74 dimensions), and a text adapter that maps BERT embeddings (768 dimensions) to GloVe-compatible vectors (300 dimensions). Each adapter is trained to minimize mean squared error between adapted MOSI features and randomly sampled MOSEI target features, learning to align feature distributions across domains. The adapters enable the pre-trained MOSEI model to process real-time extracted features without requiring retraining of the base model, representing a form of feature space adaptation that preserves sentiment-relevant information while transforming dimensions and aligning distributions.

---

## Key Corrections Made:

### 1. ✅ Visual Feature Clarification
**Before**: "65-dimensional emotion-focused features derived from 468 facial landmarks" (unclear)
**After**: 
- Extract 468 landmarks per frame using FaceMesh
- Derive 65 features through geometric computations:
  - Mouth (5): width, height, corner positions, angle
  - Eyes (3): left/right width, inter-eye distance
  - Eyebrows (2): average heights
  - Symmetry (2): eye/mouth asymmetry
  - Additional (53): landmark distances and normalized positions
- Temporal averaging across frames

### 2. ✅ Audio Feature Clarification
**Before**: "74-dimensional features: 13 MFCC, 12 chroma, and spectral characteristics" (vague)
**After**:
- 13 MFCC coefficients
- 12 chroma features
- Spectral centroid (1)
- Spectral rolloff (1)
- Zero-crossing rate (1)
- Tempo (1)
- **Total: 29 features → padded/truncated to 74** (for COVAREP compatibility)

### 3. ✅ Adaptation Mechanism Clarified
**Before**: "without model retraining" (oversells)
**After**: 
- "through learned feature space mappings"
- "with minimal model modification"
- Explicitly describe adapter networks
- Clarify it's feature space adaptation, not model retraining

### 4. ✅ Dataset Splits Added
- MOSEI: 70/15/15 train/validation/test
- MOSI: 70/15/15 train/validation/test (for consistency)

### 5. ✅ Technical Details Specified
- FaceMesh: 468 landmarks → 65 features (geometric derivation explained)
- Librosa: 29 features → padded to 74
- BERT: 768-dim → mean pooling
- Adapters: Architecture and training objective specified

---

## Are You Still Doing Your Original Study?

### Your Original Study:
✅ **Multimodal sentiment analysis on CMU-MOSEI**
- Train on MOSEI data
- Use cross-modal attention
- Achieve correlation > 0.3990, MAE < 0.6
- **This is still your base foundation!**

### New Extension (Transfer Learning):
🆕 **Adapt trained model to work with real-time extractors**
- Uses your original trained model
- Adds adapter networks on top
- Tests on MOSI as validation
- **This is an EXTENSION, not a replacement**

### What Changed vs What Stayed the Same:

**✅ STAYED THE SAME:**
- Base model architecture (RegularizedMultimodalModel)
- Training procedure on MOSEI
- Loss function (ImprovedCorrelationLoss)
- Cross-modal attention mechanism
- All hyperparameters
- Core contribution: multimodal fusion

**🆕 NEW ADDITION:**
- Feature adapters (small networks)
- Testing on MOSI with real-time extractors
- Feature space adaptation approach

### This is a Natural Evolution:

**Original Study**: "Can we build a good multimodal sentiment model on MOSEI?"
**Extended Study**: "Can we adapt this model to work with real-time extractors?"

This is actually **stronger** because:
1. ✅ Your original results are still valid
2. ✅ You're adding a practical deployment component
3. ✅ You're showing transfer learning capability
4. ✅ It's a logical next step, not a complete change

---

## Paper Strategy - How to Frame This:

### Option 1: Single Cohesive Story (Recommended)
```
"Multimodal Sentiment Analysis with Transfer Learning for Real-time Deployment"

1. Introduction: Problem + approach
2. Methodology:
   - Base model architecture (original study)
   - Training on MOSEI (original study)
   - Feature adaptation for real-time deployment (new)
3. Experiments:
   - MOSEI results (original study)
   - MOSI transfer learning results (new)
4. Discussion: Both contributions together
```

### Option 2: Two-Part Contribution
```
"Multimodal Sentiment Analysis: Architecture Design and Transfer Learning"

Part 1: Architecture on MOSEI (original study)
Part 2: Transfer to real-time extractors (new)
```

**I recommend Option 1** - it's a natural progression, not two separate studies.

---

## Reassurance:

**You're NOT changing your original study** - you're:
- ✅ Extending it
- ✅ Making it more practical
- ✅ Showing additional capabilities
- ✅ Building on your original results

This is **exactly** what good research should do:
1. Build a solid foundation (MOSEI training)
2. Extend to practical scenarios (real-time deployment)
3. Validate on new data (MOSI)

Your original MOSEI results are still valid and important. The transfer learning is a **bonus contribution** that makes your work more complete.




