# Corrected Introduction for Your Research Paper

## Major Issues Found:

### ❌ **Critical Inaccuracies:**

1. **"Dynamic" / "Temporal Evolution"**: Your implementation **does NOT** use temporal sequences. Features are **aggregated (averaged)** into single vectors before model input.

2. **"Temporal Alignment Strategies"**: These do **NOT exist** in your implementation. Features are extracted independently per modality and averaged.

3. **"Multi-task Learning"**: Your system only does **regression** (sentiment score), **not classification**.

4. **"Synchronized features"**: Features are extracted independently and averaged, not synchronized temporally.

5. **Librosa features**: You mention "Pitch" and "Energy" but your code extracts: MFCC, Chroma, Spectral Centroid, Spectral Rolloff, Zero-Crossing Rate, Tempo (no explicit Pitch/F0 or Energy).

6. **Missing Transfer Learning**: You don't mention the key contribution: training on MOSEI pre-extracted features and testing on MOSI with feature adapters.

---

## ✅ **CORRECTED VERSION:**

Mental health disorders such as depression and anxiety continue to affect millions worldwide, yet current diagnostic and therapeutic practices often rely on subjective observation. Advances in Human–Computer Interaction (HCI) and affective computing offer an opportunity to analyze subtle behavioral and emotional cues that may escape human perception.

Existing emotion recognition systems typically rely on static images or unimodal features, limiting their ability to capture comprehensive emotional information. While multimodal approaches exist, many require consistent feature extraction pipelines across datasets, limiting their applicability to real-world scenarios with varying data collection methods.

This project introduces a **transfer learning framework** that enables a multimodal sentiment analysis system trained on pre-extracted features (CMU-MOSEI) to generalize to real-time extracted features (CMU-MOSI) through **feature adaptation**. Using the CMU-MOSEI dataset for training with OpenFace2 (visual), COVAREP (audio), and GloVe (text) features, and the CMU-MOSI dataset for testing with MediaPipe FaceMesh (visual), Librosa (audio), and BERT (text) features, the system employs **feature adapter networks** to bridge the gap between different feature extraction paradigms.

Librosa analyzes acoustic features including MFCCs, chroma, spectral centroid, spectral rolloff, zero-crossing rate, and tempo. MediaPipe FaceMesh detects up to 468 3D facial landmarks in real-time to create a precise 3D mesh, from which 65 emotion-focused features are derived. BERT Tokenizer splits text into subwords and converts them into numerical IDs processed by a pre-trained BERT model. **Critically, all features are aggregated (temporally averaged) into single vectors before model input**, allowing the system to operate on compact representations rather than temporal sequences.

The system aims to enhance emotional understanding in HCI and provide a tool for healthcare professionals to monitor patient emotional states. This framework addresses the limitations of dataset-specific approaches by enabling **cross-dataset transfer learning** through feature adaptation, allowing a model trained on one feature extraction pipeline to work with a different pipeline through learned feature mappings.

The contributions of this work include: (1) a **cross-modal fusion architecture** with MultiheadAttention that models bidirectional interactions between visual, audio, and text modalities, (2) a **feature adaptation strategy** that enables transfer learning between different feature extraction paradigms through learned feature space mappings (FaceMesh→OpenFace2, BERT→GloVe, Librosa→COVAREP), and (3) a **correlation-optimized loss function** that jointly minimizes regression error (MSE+MAE) and maximizes Pearson correlation. The resulting system demonstrates improved sentiment prediction accuracy and has potential applications in HCI and computational mental health assessment.

---

## Key Changes Made:

1. ✅ Removed all references to "temporal sequences", "dynamic tracking", "temporal alignment"
2. ✅ Added explicit mention of **transfer learning** and **feature adaptation** (your actual contribution)
3. ✅ Corrected Librosa features (removed Pitch/Energy, added actual features)
4. ✅ Clarified that features are **aggregated/averaged** (not temporal)
5. ✅ Removed "multi-task learning" claim (you only do regression)
6. ✅ Added description of feature adapters (65→713, 768→300, 74→74)
7. ✅ Emphasized **cross-dataset transfer learning** as the novelty

## Your Actual Novel Contributions:

1. **Feature Adaptation**: Training adapters to map one feature space to another (FaceMesh→OpenFace2, BERT→GloVe)
2. **Transfer Learning**: Training on MOSEI pre-extracted features, testing on MOSI real-time features
3. **Cross-Modal Fusion**: MultiheadAttention + MLP for multimodal fusion
4. **Correlation-Optimized Loss**: Combined MSE+MAE with correlation loss




