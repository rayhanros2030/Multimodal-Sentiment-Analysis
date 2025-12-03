# Revised Future Works and Limitations Sections

## ❌ **CURRENT FUTURE WORKS (Issues Found):**

### Issues:
1. ❌ **Too informal** ("I plan to", "I want to")
2. ❌ **Vague goals** (not specific enough)
3. ❌ **Missing alignment with current work** (doesn't build on transfer learning contribution)
4. ❌ **No mention of improving current limitations**
5. ❌ **IEMOCAP mentioned** (but not used in current work - confusing)
6. ❌ **Lacks technical depth** (doesn't explain HOW to achieve goals)

---

## ✅ **REVISED FUTURE WORKS SECTION:**

### 4.5 Future Works

Several directions for future research emerge from this work. First, extending the feature adaptation framework to additional modalities would strengthen the system's capability to capture comprehensive emotional information. Specifically, incorporating body gesture features using MediaPipe BlazePose or similar pose estimation frameworks could provide complementary signals for sentiment analysis, as body language contributes significantly to emotional expression. The AMIGOS dataset, which provides synchronized video, audio, and text annotations alongside emotion labels, offers an opportunity to evaluate gesture-based feature extraction and adaptation strategies. This extension would require developing an additional feature adapter network to map pose-based features (e.g., joint angles, body posture) to the existing feature space, following the same adaptation paradigm established for visual, audio, and text modalities.

Second, improving the feature adapter training strategy could enhance transfer learning performance. Current adapter training uses random sampling of target features from CMU-MOSEI, which may not optimally align feature distributions. Future work could explore more sophisticated target selection methods, such as K-means clustering to identify representative feature prototypes, or triplet loss functions that learn to pull similar features closer together while pushing dissimilar features apart. Additionally, adversarial training approaches could be employed, where a discriminator network learns to distinguish adapted features from target features, encouraging adapters to produce more indistinguishable feature representations.

Third, expanding the transfer learning evaluation to additional datasets and feature extraction combinations would strengthen generalization claims. Evaluating the framework's performance when adapting between other feature extractor pairs (e.g., OpenFace2 to FACET, COVAREP to OpenSMILE) would demonstrate broader applicability beyond the specific MOSEI→MOSI transfer scenario. Cross-dataset evaluations on additional sentiment analysis datasets, such as IEMOCAP or Meld, would provide further evidence of the approach's robustness across different domains and annotation schemes.

Fourth, incorporating temporal sequence modeling could improve performance for longer video segments. While the current framework aggregates features temporally for computational efficiency, future work could explore lightweight temporal architectures, such as 1D convolutional networks or simplified LSTM layers, that process frame-level features while maintaining the feature adaptation framework. This would enable the model to capture temporal emotion dynamics while preserving the transfer learning capability.

Finally, end-to-end fine-tuning strategies could bridge the remaining performance gap between same-dataset and transfer learning scenarios. After initial adapter training, jointly fine-tuning both adapters and the main model on the target dataset (CMU-MOSI) using a small learning rate could further align feature distributions and improve sentiment prediction accuracy, potentially achieving correlation values comparable to same-dataset baselines.

---

## ✅ **NEW LIMITATIONS SECTION (Created):**

### 4.4 Limitations

Several limitations of the current work should be acknowledged. First, the transfer learning approach incurs a performance penalty compared to same-dataset evaluation, as expected for cross-domain generalization. While feature adapters successfully enable cross-dataset transfer, the adapted features may not perfectly capture all information present in the original pre-extracted features, leading to correlation values (0.30-0.45) that are lower than same-dataset baselines (0.44-0.48). This gap reflects the inherent challenge of bridging different feature extraction paradigms and suggests room for improvement in adapter architectures and training strategies.

Second, the current framework uses temporal feature aggregation (averaging) rather than temporal sequence modeling, which limits the model's ability to capture dynamic emotion evolution within video segments. While this design choice enables efficient processing and maintains compatibility with the fixed-size input requirements of the encoder architecture, it may miss important temporal patterns in emotion expression, such as emotional transitions or cumulative sentiment shifts over time. Future work incorporating lightweight temporal modeling could address this limitation.

Third, the feature adapter training process relies on random sampling of target features from CMU-MOSEI, which may not optimally align feature distributions. More sophisticated target selection methods, such as clustering-based sampling or similarity-based matching, could potentially improve adapter performance but were not explored in this work due to computational constraints and scope limitations.

Fourth, the evaluation is limited to two datasets (CMU-MOSEI and CMU-MOSI) with specific feature extraction combinations (OpenFace2→FaceMesh, COVAREP→Librosa, GloVe→BERT). While these combinations represent realistic deployment scenarios, broader evaluation across additional datasets and feature extractor pairs would strengthen generalization claims and identify potential failure modes of the adaptation approach.

Fifth, the framework requires sufficient training data for both the source dataset (CMU-MOSEI) and feature adapters. In scenarios with limited labeled data in the target domain, adapter training may be insufficient, limiting the effectiveness of the transfer learning approach. Additionally, the current approach assumes the availability of pre-extracted features from the source dataset, which may not always be available in practice.

Finally, computational requirements for real-time feature extraction (particularly FaceMesh video processing) can be demanding, potentially limiting deployment in resource-constrained environments. While the adapters enable deployment flexibility, the underlying feature extraction steps still require sufficient computational resources for real-time processing.

---

## **Alternative Shorter Versions** (if space is limited):

### Short Future Works (1 paragraph):

Several directions for future research include extending the framework to additional modalities (e.g., body gestures using BlazePose and the AMIGOS dataset), improving adapter training through K-means target selection and adversarial training, expanding evaluation to additional datasets and feature extractor combinations, incorporating lightweight temporal sequence modeling, and exploring end-to-end fine-tuning strategies to bridge the performance gap with same-dataset baselines.

### Short Limitations (1 paragraph):

Limitations include the performance gap between transfer learning and same-dataset evaluation (expected for cross-domain generalization), the use of temporal aggregation rather than sequence modeling (limiting dynamic emotion capture), reliance on random sampling for adapter training, limited evaluation to two datasets and specific feature extractor pairs, requirements for sufficient training data in both source and target domains, and computational demands for real-time feature extraction.




