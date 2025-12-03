# Visual Pipeline Paragraph - Updated with Clarifications

## Updated Paragraph (With All Clarifications)

```
For visual feature extraction, we process video frames using MediaPipe FaceMesh to extract 468 3D facial landmarks per frame. To handle variable-length videos efficiently, we process frames sequentially from the start of each video, processing up to the first 100 frames (or all frames if the video contains fewer than 100 frames). This frame-based processing strategy ensures consistent feature extraction across videos of different lengths while maintaining computational efficiency. For each frame, we derive 65-dimensional emotion-focused features through geometric computations, with the face first normalized by face width (computed as the Euclidean distance between landmarks at indices 0 and 16) to handle scale variation across videos and frames.

We extract 12 explicitly defined emotion-relevant features: mouth characteristics including mouth width (Euclidean distance between landmarks 61 and 291), mouth height (distance between landmarks 13 and 14), left and right mouth corner Y-coordinates (landmarks 61 and 291), and mouth corner angle computed using arctangent—yielding 5 features. Eye features comprise left eye width (distance between landmarks 33 and 133), right eye width (distance between landmarks 362 and 263), and inter-eye distance (distance between landmarks 33 and 263)—yielding 3 features. Eyebrow features are computed as average heights: left eyebrow using landmarks 21, 55, and 107, and right eyebrow using landmarks 251, 285, and 336—yielding 2 features. Symmetry metrics include eye symmetry (normalized absolute difference between left and right eye widths) and mouth symmetry (absolute difference between left and right corner Y-coordinates)—yielding 2 features, for a total of 12 explicitly defined features.

The remaining 53 features are derived from normalized landmark magnitudes: we compute the Euclidean norm (L2 norm) of normalized landmark coordinates at indices 12 through 64, providing additional geometric information about facial structure relative to face scale. This yields exactly 65 features per frame: [1 frame × 65 features]. Features are extracted at the frame level (one 65-dimensional vector per frame) and then temporally averaged across all processed frames to obtain a single video-level representation: [1 video × 65 features]. This video-level feature vector is passed through the visual encoder, which consists of two linear transformations (65 → 192 → 96 dimensions) with batch normalization, ReLU activation, and dropout (0.7) to provide strong regularization and prevent overfitting, to obtain the encoded visual representation (embed_dim = 96) used in cross-modal attention and fusion. The temporal averaging strategy provides a stable representation that captures overall facial expression patterns while remaining computationally efficient and compatible with the fixed-size input requirements of the encoder architecture.
```

---

## Key Clarifications Addressed

✅ **Frame processing**: "sequentially from the start" - clarifies NOT uniform sampling
✅ **< 100 frames**: "or all frames if the video contains fewer than 100 frames" - handles short videos
✅ **Frame-based**: "This frame-based processing strategy" - clarifies NOT time-windowed
✅ **Dropout 0.7**: "to provide strong regularization and prevent overfitting" - explains why 0.7 is used

---

## Alternative Shorter Version (If Needed)

```
For visual feature extraction, we process video frames using MediaPipe FaceMesh to extract 468 3D facial landmarks per frame. We process frames sequentially from the start of each video, processing up to the first 100 frames (or all frames for shorter videos). For each frame, we derive 65-dimensional emotion-focused features: 12 explicitly defined features (mouth width/height, eye dimensions, eyebrow heights, symmetry metrics) and 53 derived features (Euclidean norms of normalized landmark coordinates at indices 12-64). The face is normalized by face width (Euclidean distance between landmarks 0 and 16) to handle scale variation. Frame-level features (65-dim per frame) are temporally averaged to obtain a single video-level representation (65-dim per video), which is passed through the visual encoder (65 → 192 → 96 dimensions with batch normalization, ReLU activation, and dropout 0.7 for strong regularization) to obtain the encoded visual representation (embed_dim = 96) used in cross-modal attention and fusion.
```




