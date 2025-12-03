# Visual Pipeline Clarifications

## 1. What happens if videos have < 100 frames?

**Answer:** If a video has fewer than 100 frames, all available frames are processed. The code processes frames sequentially until either:
- 100 frames are processed, OR
- The video ends (no more frames available)

**Implementation (lines 362-365):**
```python
while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break  # Video ended, exit loop
```

**Result:** If a video has 50 frames, all 50 frames are processed and averaged. If it has 150 frames, only the first 100 are processed.

**For your paper:** 
- "We process up to the first 100 frames per video, processing all available frames for videos shorter than 100 frames."
- OR: "We process up to the first 100 frames per video (or all frames if the video contains fewer than 100 frames)."

---

## 2. Are frames sampled uniformly or taken sequentially from the start?

**Answer:** Frames are taken **sequentially from the start** (not uniformly sampled).

**Implementation (lines 362-380):**
```python
frame_count = 0
while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()  # Sequential frame reading
    # Process frame...
    frame_count += 1
```

**What this means:**
- Frame 0, 1, 2, 3, ... up to frame 99 (or until video ends)
- **NOT** uniformly sampled (e.g., not every Nth frame across the video)
- **NOT** random sampling
- Simply: first 100 frames (or all frames if < 100)

**For your paper:**
- "We process up to the first 100 frames per video sequentially from the start"
- OR: "Frames are extracted sequentially from the start of each video, processing up to 100 frames"

---

## 3. How does this relate to 3-second windows?

**Answer:** The code does **NOT explicitly use 3-second windows**. It uses a **frame-based limit (100 frames)**, not a time-based limit.

**What happens:**
- The code processes up to 100 frames
- At 30 fps (typical video frame rate), 100 frames ≈ 3.33 seconds
- At 25 fps, 100 frames = 4 seconds
- At 60 fps, 100 frames ≈ 1.67 seconds

**So the 3-second reference is approximate/conceptual:**
- At typical 30 fps: 100 frames ≈ 3.3 seconds
- But the implementation is **frame-based**, not time-based

**For your paper:**
- **Option A (Frame-based - more accurate):** "We process up to the first 100 frames per video sequentially from the start"
- **Option B (Time-based approximation):** "We process up to the first 100 frames per video (approximately 3.3 seconds at 30 fps), extracting frames sequentially from the start"
- **Option C (Hybrid):** "We process frames sequentially from the start of each video, limiting processing to the first 100 frames (approximately 3 seconds at typical video frame rates of 30 fps)"

**Recommendation:** Use Option A or Option C. Be clear that it's frame-based, not time-windowed.

---

## 4. Dropout rate confirmation (0.7 = 70%)

**Answer:** Yes, dropout=0.7 (70%) is **correct** and **intentionally high** for strong regularization.

**Implementation confirmation:**
- Line 41: `dropout=0.7` (default parameter)
- Line 78: `nn.Dropout(dropout)` in encoder layers
- Line 87: `nn.Dropout(dropout)` in additional encoder layers
- Line 61: `nn.Dropout(dropout)` in fusion layers

**Why 0.7 is used:**
- **Strong regularization**: Prevents overfitting, especially important with limited/small datasets
- **Common in multimodal models**: Multimodal fusion can be prone to overfitting, higher dropout helps
- **Matches literature**: Some papers use 0.5-0.8 dropout for regularization in sentiment analysis models
- **Your results**: With dropout=0.7, you achieved correlation 0.4113 on CMU-MOSEI, which suggests the high dropout is helping prevent overfitting

**Typical dropout ranges:**
- **Standard**: 0.1-0.5 (common in vision/NLP)
- **High regularization**: 0.5-0.8 (for overfitting prevention)
- **Your model**: 0.7 (high, but justified for regularization)

**For your paper:**
- "We employ dropout of 0.7 in all encoder and fusion layers to provide strong regularization and prevent overfitting, which is particularly important for multimodal models with limited training data."
- OR: "Dropout is set to 0.7 in all encoder and fusion layers to enhance model regularization and reduce overfitting risk in the multimodal fusion architecture."

---

## Revised Visual Pipeline Paragraph with Clarifications

Here's the paragraph with the clarifications incorporated:

```
For visual feature extraction, we process video frames using MediaPipe FaceMesh to extract 468 3D facial landmarks per frame. To handle variable-length videos efficiently, we process frames sequentially from the start of each video, processing up to the first 100 frames (or all frames if the video contains fewer than 100 frames). This frame-based processing strategy ensures consistent feature extraction across videos of different lengths while maintaining computational efficiency. For each frame, we derive 65-dimensional emotion-focused features through geometric computations, with the face first normalized by face width (computed as the Euclidean distance between landmarks at indices 0 and 16) to handle scale variation across videos and frames.

We extract 12 explicitly defined emotion-relevant features: mouth characteristics including mouth width (Euclidean distance between landmarks 61 and 291), mouth height (distance between landmarks 13 and 14), left and right mouth corner Y-coordinates (landmarks 61 and 291), and mouth corner angle computed using arctangent—yielding 5 features. Eye features comprise left eye width (distance between landmarks 33 and 133), right eye width (distance between landmarks 362 and 263), and inter-eye distance (distance between landmarks 33 and 263)—yielding 3 features. Eyebrow features are computed as average heights: left eyebrow using landmarks 21, 55, and 107, and right eyebrow using landmarks 251, 285, and 336—yielding 2 features. Symmetry metrics include eye symmetry (normalized absolute difference between left and right eye widths) and mouth symmetry (absolute difference between left and right corner Y-coordinates)—yielding 2 features, for a total of 12 explicitly defined features.

The remaining 53 features are derived from normalized landmark magnitudes: we compute the Euclidean norm (L2 norm) of normalized landmark coordinates at indices 12 through 64, providing additional geometric information about facial structure relative to face scale. This yields exactly 65 features per frame: [1 frame × 65 features]. Features are extracted at the frame level (one 65-dimensional vector per frame) and then temporally averaged across all processed frames to obtain a single video-level representation: [1 video × 65 features]. This video-level feature vector is passed through the visual encoder, which consists of two linear transformations (65 → 192 → 96 dimensions) with batch normalization, ReLU activation, and dropout (0.7) to provide strong regularization, to obtain the encoded visual representation (embed_dim = 96) used in cross-modal attention and fusion. The temporal averaging strategy provides a stable representation that captures overall facial expression patterns while remaining computationally efficient and compatible with the fixed-size input requirements of the encoder architecture.
```

**Key changes:**
1. ✅ Clarified: "sequentially from the start" and "or all frames if < 100"
2. ✅ Removed ambiguous "3-second window" reference
3. ✅ Explained dropout 0.7 as intentional for regularization

---

## Summary Table

| Question | Answer |
|----------|--------|
| **< 100 frames?** | All available frames processed |
| **Sampling method?** | Sequential from start (first 100 frames) |
| **3-second windows?** | No explicit time windows; frame-based (100 frames ≈ 3.3s at 30fps) |
| **Dropout 0.7 correct?** | Yes, intentionally high for strong regularization |




