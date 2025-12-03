# Updated Adapter Training and Feature Normalization Sections

## **Section 3.5.2: Training Feature Adapters (UPDATED)**

### **Updated Paragraph:**

**3.5.2 Training Feature Adapters**

In the second phase, we train three feature adapter networks to map real-time extracted features (from CMU-MOSI) to the pre-extracted feature space (from CMU-MOSEI). Each adapter is trained independently using mean squared error loss, but with enhanced target selection and architecture design to improve feature alignment. The three adapters are:

**Visual Adapter:** Maps FaceMesh features (65-dim) → OpenFace2 features (713-dim). For this large dimension expansion (approximately 11x), a deeper 5-layer architecture is employed to provide sufficient model capacity: Linear(65→128)→BatchNorm1d→ReLU→Dropout(0.3)→Linear(128→256)→BatchNorm1d→ReLU→Dropout(0.3)→Linear(256→512)→BatchNorm1d→ReLU→Dropout(0.3)→Linear(512→1024)→BatchNorm1d→ReLU→Dropout(0.3)→Linear(1024→713). This deeper architecture enables the adapter to learn complex non-linear mappings required for the large dimension gap.

**Audio Adapter:** Maps Librosa features (74-dim) → COVAREP features (74-dim). Since the dimensions match, a standard 3-layer architecture suffices: Linear(74→256)→BatchNorm1d→ReLU→Dropout(0.3)→Linear(256→256)→BatchNorm1d→ReLU→Dropout(0.3)→Linear(256→74).

**Text Adapter:** Maps BERT embeddings (768-dim) → GloVe embeddings (300-dim). A 3-layer architecture is used: Linear(768→384)→BatchNorm1d→ReLU→Dropout(0.3)→Linear(384→384)→BatchNorm1d→ReLU→Dropout(0.3)→Linear(384→300).

To improve feature alignment, we employ K-means clustering for target feature selection rather than random sampling. Specifically, 2000 samples are randomly selected from CMU-MOSEI for each modality, and K-means clustering (with K=100 clusters) is applied to identify representative cluster centers that capture the distribution of pre-extracted features. During adapter training, for each input feature from CMU-MOSI, we select the nearest cluster center from the corresponding MOSEI modality as the target, ensuring that adapted features align with representative feature prototypes rather than arbitrary samples. This clustering-based approach provides more stable and meaningful target features compared to random sampling, leading to better feature alignment.

Training parameters are optimized per adapter: the visual adapter uses a learning rate of 0.001 (higher due to the larger dimension gap), while audio and text adapters use 0.0005. All adapters are trained for 75 epochs with Adam optimizer, weight decay of 1e-5, and gradient clipping (max norm 1.0). Learning rate schedulers (ReduceLROnPlateau with factor=0.5, patience=5) are used to adaptively reduce learning rates when loss plateaus, improving convergence. The adapters are trained to minimize mean squared error between adapted MOSI features and their corresponding MOSEI cluster center targets, learning to map real-time extracted features to the pre-extracted feature space while preserving sentiment-relevant information.

---

## **Section 3.5.3: Feature Normalization (NEW SECTION)**

### **New Paragraph:**

**3.5.3 Feature Normalization**

After feature adaptation, the adapted features may exhibit different statistical distributions compared to the original pre-extracted features used during model training on CMU-MOSEI. To ensure the pre-trained model receives features in the expected distribution, we normalize adapted features to match the statistical properties of MOSEI features. Specifically, during adapter training, we compute the mean and standard deviation of each feature dimension across 2000 randomly sampled MOSEI features for each modality (visual, audio, text). These statistics are stored and used during testing to normalize adapted features:

Normalized Feature = (Adapted Feature - MOSEI Mean) / MOSEI Standard Deviation

This normalization ensures that adapted features are centered and scaled to match the distribution the model was trained on, preventing distribution mismatch that could degrade model performance. Additionally, extreme values are clipped to [-10, 10] to prevent outliers from disrupting the model. This feature normalization step is critical for effective transfer learning, as it ensures that adapted features occupy the same statistical space as the original pre-extracted features, enabling the pre-trained model to process them correctly without requiring retraining or fine-tuning of the base model architecture.

---

## **Alternative Shorter Versions:**

### **Adapter Training (Shorter):**

**3.5.2 Training Feature Adapters**

We train three feature adapter networks to map real-time extracted features (CMU-MOSI) to pre-extracted feature spaces (CMU-MOSEI). The visual adapter uses a deeper 5-layer architecture (65→128→256→512→1024→713) to handle the large dimension expansion, while audio (74→74) and text (768→300) adapters use 3-layer architectures. To improve feature alignment, K-means clustering is applied to 2000 MOSEI samples per modality to identify representative cluster centers, which serve as targets during training. Each adapter is trained for 75 epochs with Adam optimizer, learning rate scheduling, gradient clipping, and weight decay. The visual adapter uses LR=0.001, while audio and text adapters use LR=0.0005. Training minimizes mean squared error between adapted features and their nearest MOSEI cluster centers.

### **Feature Normalization (Shorter):**

**3.5.3 Feature Normalization**

After adaptation, features are normalized to match MOSEI feature statistics (mean and standard deviation) computed from 2000 sampled MOSEI features per modality. This normalization ensures adapted features occupy the same statistical space as the original pre-extracted features, preventing distribution mismatch. Extreme values are clipped to [-10, 10] to prevent outliers. This step is critical for enabling the pre-trained model to process adapted features correctly without requiring architecture modifications.

---

## **Integration Points:**

### **Where to Add:**

1. **Section 3.5.2:** Replace the existing adapter training description with the updated version above
2. **Section 3.5.3:** Add the new feature normalization section before "Testing on CMU-MOSI"
3. **Section 3.5.4:** Update "Testing on CMU-MOSI" to mention "after normalization" when describing the testing process

### **Key Changes from Original:**

1. ✅ **Deeper visual adapter** (5 layers instead of 3)
2. ✅ **K-means clustering** (not random sampling)
3. ✅ **2000 samples** (not 1000)
4. ✅ **Different learning rates** (visual: 0.001, others: 0.0005)
5. ✅ **75 epochs** (not 30)
6. ✅ **Learning rate scheduling** (new)
7. ✅ **Gradient clipping** (new)
8. ✅ **Weight decay** (new)
9. ✅ **Feature normalization** (completely new section)

---

## **Suggested Flow:**

**Section 3.5: Transfer Learning**

- 3.5.1 Training on CMU-MOSEI
- 3.5.2 Training Feature Adapters (UPDATED)
- 3.5.3 Feature Normalization (NEW)
- 3.5.4 End-to-End Fine-Tuning (NEW)
- 3.5.5 Testing on CMU-MOSI (UPDATED to mention normalization and fine-tuning)

---

## **Ready to Use!**

Both paragraphs are ready to paste into your paper. Use the full versions for comprehensive detail, or the shorter versions if space is limited. The key improvements are all documented!




