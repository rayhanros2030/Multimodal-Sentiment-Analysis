# Visual Pipeline - Final Paragraph for Paper

## Recommended Paragraph (Concise & Technical)

```
For visual feature extraction, we process video frames using MediaPipe FaceMesh to extract 468 3D facial landmarks per frame. To handle variable-length videos efficiently, we process up to the first 100 frames per video at the video's native frame rate (typically 30 fps), extracting frame-level features sequentially. For each frame, we derive 65-dimensional emotion-focused features through geometric computations, with the face first normalized by face width (computed as the Euclidean distance between landmarks at indices 0 and 16) to handle scale variation across videos and frames.

We extract 12 explicitly defined emotion-relevant features: mouth characteristics including mouth width (Euclidean distance between landmarks 61 and 291), mouth height (distance between landmarks 13 and 14), left and right mouth corner Y-coordinates (landmarks 61 and 291), and mouth corner angle computed using arctangent—yielding 5 features. Eye features comprise left eye width (distance between landmarks 33 and 133), right eye width (distance between landmarks 362 and 263), and inter-eye distance (distance between landmarks 33 and 263)—yielding 3 features. Eyebrow features are computed as average heights: left eyebrow using landmarks 21, 55, and 107, and right eyebrow using landmarks 251, 285, and 336—yielding 2 features. Symmetry metrics include eye symmetry (normalized absolute difference between left and right eye widths) and mouth symmetry (absolute difference between left and right corner Y-coordinates)—yielding 2 features, for a total of 12 explicitly defined features.

The remaining 53 features are derived from normalized landmark magnitudes: we compute the Euclidean norm (L2 norm) of normalized landmark coordinates at indices 12 through 64, providing additional geometric information about facial structure relative to face scale. This yields exactly 65 features per frame: [1 frame × 65 features]. Features are extracted at the frame level (one 65-dimensional vector per frame) and then temporally averaged across all processed frames (up to 100) to obtain a single video-level representation: [1 video × 65 features]. This video-level feature vector is passed through the visual encoder, which consists of two linear transformations (65 → 192 → 96 dimensions) with batch normalization, ReLU activation, and dropout (0.7), to obtain the encoded visual representation (embed_dim = 96) used in cross-modal attention and fusion. The temporal averaging strategy provides a stable representation that captures overall facial expression patterns while remaining computationally efficient and compatible with the fixed-size input requirements of the encoder architecture.
```

---

## What This Paragraph Addresses

✅ **1. Less FaceMesh internals (80% → minimal)**
- Removed internal FaceMesh architecture details
- Focuses on feature extraction and processing

✅ **2. Feature extraction details**
- ✓ 65-dimensional features from 468 landmarks
- ✓ How 468 → 65 (12 explicit + 53 derived)
- ✓ Specific feature types (mouth, eye, eyebrow, symmetry, landmark magnitudes)
- ✓ Normalization method (face-width normalization)
- ✓ Temporal processing (frame-level → averaged → video-level)

✅ **3. Connection to architecture**
- ✓ Input: Raw video frames → FaceMesh → 468 landmarks
- ✓ Output: 65-dim per frame → 65-dim per video
- ✓ Connection to encoder: 65 → 192 → 96 dimensions
- ✓ Dimensions at each stage clearly specified

✅ **4. Feature derivation explanation**
- ✓ How 468 landmarks → 65 features (explicit computations)
- ✓ What computations (Euclidean distances, angles, means, L2 norms)
- ✓ Why 65 features (emotion-relevant regions + geometric structure)

✅ **5. Temporal processing details**
- ✓ Features computed per frame (up to 100 frames)
- ✓ Averaged to video-level: [1 video × 65 features]
- ✓ Connection to encoder (single vector per video, not broadcast)

✅ **6. Technical specifications**
- ✓ Video processing rate (native fps, typically 30)
- ✓ Frame selection strategy (first 100 frames)
- ✓ Feature dimensions (65 per frame → 65 per video → 96 after encoder)
- ✓ Normalization (face-width normalization)
- ✓ Encoder connection (65 → 192 → 96 with BatchNorm, ReLU, Dropout)

---

## Key Features of This Paragraph

1. **Focus on Feature Extraction**: Minimal mention of FaceMesh internals, maximum detail on what features are extracted and how

2. **Complete Feature Breakdown**: 
   - 12 explicit features with specific landmark indices and computations
   - 53 derived features (landmark L2 norms, indices 12-64)
   - Total: exactly 65 features

3. **Normalization Explained**: Face-width normalization using landmarks 0 and 16

4. **Temporal Processing Clear**: Frame-level → temporal averaging → video-level

5. **Architecture Connection**: Clear dimensions at each stage (65 → 192 → 96)

6. **Technical Specifications**: Frame selection (first 100), processing rate (native fps), dimensions, normalization, encoder architecture

---

## If You Need to Shorten Further

If space is limited, you can condense the explicit feature breakdown:

```
For visual feature extraction, we process video frames using MediaPipe FaceMesh to extract 468 3D facial landmarks per frame. We process up to the first 100 frames per video at the native frame rate, deriving 65-dimensional emotion-focused features per frame through geometric computations. Features include 12 explicitly defined emotion-relevant features (mouth width/height, eye dimensions, eyebrow heights, symmetry metrics) and 53 derived features (Euclidean norms of normalized landmark coordinates at indices 12-64). The face is normalized by face width (Euclidean distance between landmarks 0 and 16) to handle scale variation. Frame-level features (65-dim per frame) are temporally averaged to obtain a single video-level representation (65-dim per video), which is passed through the visual encoder (65 → 192 → 96 dimensions with batch normalization, ReLU activation, and dropout) to obtain the encoded visual representation (embed_dim = 96) used in cross-modal attention and fusion.
```

---

## Dimensions Summary (Quick Reference)

| Stage | Dimensions | Description |
|-------|------------|-------------|
| Input | Variable | Raw video (up to 100 frames) |
| FaceMesh Output | [100 frames, 468 landmarks, 3 coords] | Per-frame landmarks |
| Normalized | [100 frames, 468 landmarks, 3 coords] | Face-width normalized |
| Feature Extraction | [100 frames, 65 features] | Per-frame features |
| Temporal Averaging | [1 video, 65 features] | Video-level features |
| Visual Encoder | 65 → 192 → 96 | Two-layer encoder |
| Encoder Output | [1 video, 96 features] | Encoded visual (embed_dim) |
| Cross-Modal Fusion | [1 video, 288 features] | Concatenated (96×3) |




