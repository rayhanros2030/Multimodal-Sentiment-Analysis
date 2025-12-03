# Dataset Description - Revised for Accuracy

## Corrected Version (Aligned with Actual Methodology)

This study utilizes the CMU-MOSEI and CMU-MOSI datasets to train and evaluate a transfer learning framework for multimodal sentiment analysis. The approach employs a novel feature adaptation strategy where models are trained on CMU-MOSEI using pre-extracted features, then adapted to work with real-time feature extractors on CMU-MOSI, enabling cross-dataset generalization without model retraining.

**CMU-MOSEI (Multimodal Opinion Sentiment and Emotion Intensity)** is one of the largest multimodal sentiment analysis corpora, comprising over 23,500 annotated video segments drawn from 1,000 distinct speakers discussing more than 250 topics. Each clip captures diverse recording conditions—varying in camera distance, lighting, and background—reflecting natural, in-the-wild expression patterns. The dataset provides continuous sentiment labels ranging from -3 (strongly negative) to +3 (strongly positive), enabling fine-grained analysis of sentiment intensity. For CMU-MOSEI, we utilize the provided pre-extracted features: OpenFace2 visual features (713-dimensional facial action units and pose features), COVAREP audio features (74-dimensional prosodic and spectral features), and GloVe text embeddings (300-dimensional word vectors). These pre-extracted features serve as the training data for our base multimodal fusion model and establish the target feature space for our adaptation framework.

**CMU-MOSI (Multimodal Opinion-Level Sentiment Intensity)** is a widely used multimodal dataset for sentiment analysis, consisting of 2,199 video segments from 93 YouTube movie review videos. Each segment is annotated for sentiment intensity, subjectivity, and various audio and visual features. Similar to CMU-MOSEI, CMU-MOSI contains continuous sentiment labels from -3 to +3 and exhibits diverse recording conditions typical of in-the-wild videos. Unlike CMU-MOSEI, CMU-MOSI requires real-time feature extraction from raw video, audio, and text data. We extract features using MediaPipe FaceMesh for visual processing (65-dimensional emotion-focused features derived from 468 facial landmarks), Librosa for audio analysis (74-dimensional features: 13 MFCC coefficients, 12 chroma features, and spectral characteristics), and BERT for contextual text embeddings (768-dimensional representations). CMU-MOSI serves as the target domain for our transfer learning evaluation, testing whether adapted features from real-time extractors can effectively leverage models trained on pre-extracted features.

For both datasets, we employ a standard 70/15/15 train/validation/test partition. The transfer learning framework operates in three phases: (1) training the base multimodal fusion model on CMU-MOSEI using pre-extracted features, (2) training feature adapters to map real-time extractor outputs (FaceMesh/BERT/Librosa) to pre-extracted feature spaces (OpenFace2/COVAREP/GloVe), and (3) evaluating adapted features on CMU-MOSI using the pre-trained model. This approach addresses the practical challenge of deploying models trained on pre-extracted features to real-world scenarios requiring real-time processing, demonstrating successful cross-dataset transfer from CMU-MOSEI to CMU-MOSI through learned feature space adaptation.

---

## Alternative Version (If You Include IEMOCAP)

If you want to include IEMOCAP in your description, here's a version that accurately reflects a combined approach:

This study utilizes the CMU-MOSEI, CMU-MOSI, and IEMOCAP datasets to train and evaluate the proposed multimodal sentiment analysis framework. These datasets enable a comprehensive evaluation of both transfer learning across feature extraction methods (MOSEI→MOSI with feature adapters) and cross-dataset generalization to diverse expression styles.

**CMU-MOSEI (Multimodal Opinion Sentiment and Emotion Intensity)** is one of the largest multimodal sentiment analysis corpora, comprising over 23,500 annotated video segments drawn from 1,000 distinct speakers discussing more than 250 topics. Each clip captures diverse recording conditions—varying in camera distance, lighting, and background—reflecting natural, in-the-wild expression patterns. The dataset provides continuous sentiment labels ranging from -3 (strongly negative) to +3 (strongly positive), enabling fine-grained analysis of sentiment intensity. For CMU-MOSEI, we utilize the provided pre-extracted features: OpenFace2 visual features (713-dimensional), COVAREP audio features (74-dimensional), and GloVe text embeddings (300-dimensional). These features serve as the training data for our base model and establish the target feature space for feature adaptation.

**CMU-MOSI (Multimodal Opinion-Level Sentiment Intensity)** is a widely used multimodal dataset for sentiment analysis, consisting of 2,199 video segments from 93 YouTube movie review videos. Each segment is annotated for sentiment intensity, subjectivity, and various audio and visual features. Similar to CMU-MOSEI, CMU-MOSI contains continuous sentiment labels from -3 to +3 and exhibits diverse recording conditions typical of in-the-wild videos. We extract features using MediaPipe FaceMesh (65-dimensional emotion features), Librosa (74-dimensional audio features), and BERT (768-dimensional text embeddings). CMU-MOSI serves as the target domain for our transfer learning evaluation, testing whether adapted real-time features can leverage models trained on pre-extracted features.

**IEMOCAP (Interactive Emotional Dyadic Motion Capture Database)** is an acted, multispeaker, and multimodal dataset collected at the SAIL Lab, University of Southern California. It contains approximately 12 hours of audiovisual recordings, including synchronized video, speech, facial motion capture, and text transcripts. Unlike CMU-MOSEI and CMU-MOSI's continuous sentiment scores, IEMOCAP includes both categorical emotion labels (anger, happiness, sadness, neutrality) and dimensional annotations (valence, activation, dominance), providing complementary information for emotion intensity and expressiveness. The controlled recording environment and detailed annotations make IEMOCAP particularly valuable for evaluating model generalization to acted emotional expressions. For IEMOCAP, we extract features using the same real-time pipeline (FaceMesh, Librosa, BERT) and map valence scores to the [-3, +3] sentiment scale for consistency.

For all datasets, we use a standard 70/15/15 train/validation/test partition. The framework addresses two key challenges: (1) **feature space adaptation** - transferring from pre-extracted features (MOSEI) to real-time extractors (MOSI) through learned adapters, and (2) **cross-dataset generalization** - evaluating performance across diverse expression styles from naturalistic (MOSEI, MOSI) to acted emotions (IEMOCAP). Together, these datasets enable comprehensive evaluation of both transfer learning capabilities and model robustness across different annotation schemes and recording conditions.

---

## Key Corrections Made:

1. ✅ **Removed incorrect statement**: "we extract our own features from MOSEI" → Corrected to: "we utilize pre-extracted features from MOSEI"

2. ✅ **Clarified transfer learning**: Changed from "fine-tuning on IEMOCAP" to accurate description: "training on MOSEI (pre-extracted), adapting to MOSI (real-time extractors)"

3. ✅ **Accurate feature descriptions**: 
   - MOSEI: Pre-extracted (OpenFace2, COVAREP, GloVe)
   - MOSI: Real-time extracted (FaceMesh, Librosa, BERT)

4. ✅ **Methodology alignment**: Describes the three-phase approach we actually implemented

5. ✅ **IEMOCAP clarification**: If included, explains how it fits into the evaluation strategy

---

## Recommendations:

1. **Choose your focus**: 
   - If your paper is about MOSEI→MOSI transfer with adapters, use the first version (no IEMOCAP)
   - If you evaluated on IEMOCAP too, use the second version

2. **Be explicit about feature extraction**:
   - MOSEI = pre-extracted (this is key to your contribution)
   - MOSI = real-time extracted (this is what makes transfer learning necessary)

3. **Emphasize the novelty**: The transfer from pre-extracted to real-time is your key contribution, so make this distinction clear in the dataset description.




