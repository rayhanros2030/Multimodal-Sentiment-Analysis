# Visual Pipeline Paragraph (Revised for Paper)

## Recommended Paragraph (Focus on Feature Extraction & Architecture)

```
For visual feature extraction, we process video frames using MediaPipe FaceMesh to extract 468 3D facial landmarks per frame. To handle variable-length videos efficiently, we process up to the first 100 frames per video at the video's native frame rate (typically 30 fps), extracting frame-level features sequentially. For each frame, we derive 65-dimensional emotion-focused features through geometric computations, with the face first normalized by face width (computed as the Euclidean distance between landmarks at indices 0 and 16) to handle scale variation across videos and frames.

We extract 12 explicitly defined emotion-relevant features: mouth characteristics including mouth width (Euclidean distance between landmarks 61 and 291), mouth height (distance between landmarks 13 and 14), left and right mouth corner Y-coordinates (landmarks 61 and 291), and mouth corner angle computed using arctangent—yielding 5 features. Eye features comprise left eye width (distance between landmarks 33 and 133), right eye width (distance between landmarks 362 and 263), and inter-eye distance (distance between landmarks 33 and 263)—yielding 3 features. Eyebrow features are computed as average heights: left eyebrow using landmarks 21, 55, and 107, and right eyebrow using landmarks 251, 285, and 336—yielding 2 features. Symmetry metrics include eye symmetry (normalized absolute difference between left and right eye widths) and mouth symmetry (absolute difference between left and right corner Y-coordinates)—yielding 2 features, for a total of 12 explicitly defined features.

The remaining 53 features are derived from normalized landmark magnitudes: we compute the Euclidean norm (L2 norm) of normalized landmark coordinates at indices 12 through 64, providing additional geometric information about facial structure relative to face scale. This yields exactly 65 features per frame: [1 frame × 65 features]. Features are extracted at the frame level (one 65-dimensional vector per frame) and then temporally averaged across all processed frames (up to 100) to obtain a single video-level representation: [1 video × 65 features]. This video-level feature vector is passed through the visual encoder, which consists of two linear transformations (65 → 192 → 96 dimensions) with batch normalization, ReLU activation, and dropout (0.7), to obtain the encoded visual representation (embed_dim = 96) used in cross-modal attention and fusion. The temporal averaging strategy provides a stable representation that captures overall facial expression patterns while remaining computationally efficient and compatible with the fixed-size input requirements of the encoder architecture.
```

---

## Alternative: More Detailed Version (If Space Permits)

```
For visual feature extraction, we process video frames using MediaPipe FaceMesh to extract 468 3D facial landmarks per frame, where each landmark is represented as (x, y, z) coordinates in normalized screen space (x, y ∈ [0, 1], z relative depth). FaceMesh provides real-time landmark estimation suitable for processing video streams. To handle variable-length videos efficiently, we process up to the first 100 frames per video at the video's native frame rate (typically 30 fps), extracting features sequentially from each frame. This frame selection strategy balances computational efficiency with temporal coverage, capturing the majority of expression dynamics in most video segments.

For each frame, we derive 65-dimensional emotion-focused features through geometric computations, with the face first normalized by face width to handle scale variation. Face width is computed as the Euclidean distance between landmarks at indices 0 and 16 (outer left and right face boundaries), and all 468 landmark coordinates are normalized by dividing by this face width. This normalization ensures that geometric features are scale-invariant across different camera distances and face sizes.

We extract 12 explicitly defined emotion-relevant features based on facial regions known to be important for emotion recognition. Mouth characteristics (5 features) include: mouth width (Euclidean distance between landmarks 61 and 291, corresponding to left and right mouth corners), mouth height (distance between landmarks 13 and 14, corresponding to upper and lower lip midpoints), left and right mouth corner Y-coordinates (landmarks 61 and 291), and mouth corner angle computed using arctangent of the vertical displacement relative to horizontal width. Eye features (3 features) comprise: left eye width (distance between landmarks 33 and 133, corresponding to inner and outer left eye corners), right eye width (distance between landmarks 362 and 263, corresponding to inner and outer right eye corners), and inter-eye distance (distance between landmarks 33 and 263, corresponding to inner eye corners). Eyebrow features (2 features) are computed as average heights: left eyebrow using landmarks 21, 55, and 107 (left eyebrow region), and right eyebrow using landmarks 251, 285, and 336 (right eyebrow region). Symmetry metrics (2 features) include: eye symmetry (normalized absolute difference between left and right eye widths, divided by the maximum eye width), and mouth symmetry (absolute difference between left and right corner Y-coordinates).

The remaining 53 features are derived from normalized landmark magnitudes: we compute the Euclidean norm (L2 norm) of normalized landmark coordinates at indices 12 through 64, providing additional geometric information about facial structure relative to face scale. Specifically, for each landmark index i in [12, 64], we compute ||normalized[i]|| = sqrt(x² + y² + z²), capturing the magnitude of landmark displacement from the origin. This yields exactly 65 features per frame: [1 frame × 65 features].

Features are extracted at the frame level (one 65-dimensional vector per frame) and stored for all processed frames (up to 100 frames per video). These frame-level features are then temporally averaged across all processed frames to obtain a single video-level representation: [1 video × 65 features]. Specifically, we compute the element-wise mean: video_features = (1/N) * Σ(frame_features_i), where N is the number of frames processed (up to 100) and frame_features_i is the 65-dimensional feature vector for frame i.

This video-level feature vector is passed through the visual encoder (Layer 3), which consists of two linear transformations with batch normalization and activation: Linear(65 → 192) → BatchNorm1d(192) → ReLU → Dropout(0.7) → Linear(192 → 192) → BatchNorm1d(192) → ReLU → Dropout(0.7) → Linear(192 → 96) → BatchNorm1d(96). The encoder outputs a 96-dimensional encoded visual representation (embed_dim = 96), which is used in cross-modal attention mechanisms alongside audio and text embeddings of the same dimension. The encoded visual, audio, and text features (each 96-dim) are concatenated to form a 288-dimensional fused representation (96×3) that feeds into the fusion network for final sentiment prediction.

The temporal averaging strategy provides a stable representation that captures overall facial expression patterns while remaining computationally efficient and compatible with the fixed-size input requirements of the encoder architecture. By processing frames sequentially and averaging features temporally, we preserve the dominant expression characteristics while eliminating frame-to-frame noise and reducing computational complexity compared to per-frame processing in the fusion network.
```

---

## Technical Specifications Summary

| Aspect | Specification |
|--------|--------------|
| **Face Detection** | MediaPipe FaceMesh (468 landmarks per frame) |
| **Frame Processing** | Up to 100 frames per video, sequential processing |
| **Frame Rate** | Native video fps (typically 30 fps) |
| **Frame Selection** | First 100 frames (or all if video < 100 frames) |
| **Normalization** | Face-width normalization (landmarks 0-16 distance) |
| **Feature Dimensions** | 65-dimensional per frame → 65-dimensional per video (after averaging) |
| **Feature Types** | 12 explicit (mouth: 5, eye: 3, eyebrow: 2, symmetry: 2) + 53 derived (landmark L2 norms) |
| **Temporal Processing** | Frame-level extraction → Temporal averaging → Video-level vector |
| **Visual Encoder Input** | 65-dimensional (video-level features) |
| **Visual Encoder Architecture** | Linear(65 → 192) → BatchNorm → ReLU → Dropout(0.7) → Linear(192 → 192) → BatchNorm → ReLU → Dropout(0.7) → Linear(192 → 96) → BatchNorm |
| **Visual Encoder Output** | 96-dimensional (embed_dim) |
| **Connection to Fusion** | 96-dim visual + 96-dim audio + 96-dim text → 288-dim fused → Fusion network |

---

## Feature Derivation Breakdown

### Explicit Features (12 total)

**Mouth Features (5)**:
1. Mouth width: `||landmark[61] - landmark[291]||`
2. Mouth height: `||landmark[13] - landmark[14]||`
3. Left corner Y: `landmark[61, 1]`
4. Right corner Y: `landmark[291, 1]`
5. Corner angle: `arctan2((left_y + right_y)/2 - mouth_center_y, mouth_width/2)`

**Eye Features (3)**:
6. Left eye width: `||landmark[33] - landmark[133]||`
7. Right eye width: `||landmark[362] - landmark[263]||`
8. Inter-eye distance: `||landmark[33] - landmark[263]||`

**Eyebrow Features (2)**:
9. Left eyebrow height: `mean([landmark[21, 1], landmark[55, 1], landmark[107, 1]])`
10. Right eyebrow height: `mean([landmark[251, 1], landmark[285, 1], landmark[336, 1]])`

**Symmetry Features (2)**:
11. Eye symmetry: `|left_eye_width - right_eye_width| / max(left_eye_width, right_eye_width)`
12. Mouth symmetry: `|left_corner_y - right_corner_y|`

### Derived Features (53 total)

**Landmark Magnitude Features (53)**:
For landmark indices 12 through 64:
- Feature[i] = `||normalized_landmark[i]||` = `sqrt(x² + y² + z²)`
- This yields 53 features (indices 12-64 inclusive)

**Total: 12 explicit + 53 derived = 65 features**

---

## Architecture Flow Diagram

```
Raw Video
  ↓
Frame Extraction (up to 100 frames at native fps)
  ↓
Per-Frame Processing:
  FaceMesh → 468 landmarks (x, y, z)
  ↓
  Face-Width Normalization (landmarks 0-16)
  ↓
  Feature Extraction → 65-dim per frame
    ├─ 12 explicit (mouth, eye, eyebrow, symmetry)
    └─ 53 derived (landmark L2 norms, indices 12-64)
  ↓
Temporal Averaging (mean over frames)
  ↓
Video-Level Features: [1 video × 65 features]
  ↓
Visual Encoder:
  65 → 192 (Linear + BatchNorm + ReLU + Dropout)
  ↓
  192 → 192 (Linear + BatchNorm + ReLU + Dropout)
  ↓
  192 → 96 (Linear + BatchNorm)
  ↓
Encoded Visual: [1 video × 96 features] (embed_dim)
  ↓
Cross-Modal Attention (visual, audio, text)
  ↓
Fusion Network (288-dim → 1)
```

---

## Key Points Addressed

✅ **Less FaceMesh internals, more feature extraction**: Focuses on what features are extracted and how, not how FaceMesh works internally

✅ **Feature extraction details**: 
- 65-dimensional features from 468 landmarks
- How derived (geometric computations, normalization)
- Specific feature types (mouth, eye, eyebrow, symmetry, landmark magnitudes)
- Normalization (face-width normalization)
- Temporal processing (frame-level → averaged → video-level)

✅ **Connection to architecture**: 
- Input: Raw video frames → FaceMesh
- Output: 65-dim per frame → 65-dim per video (after averaging)
- Connection to encoder: 65 → 192 → 96 dimensions
- Dimensions at each stage clearly specified

✅ **Feature derivation explanation**: 
- How 468 landmarks → 65 features (12 explicit + 53 derived)
- What computations are performed (Euclidean distances, angles, means, L2 norms)
- Why these 65 features are chosen (emotion-relevant regions)

✅ **Temporal processing details**: 
- Features computed per frame (up to 100 frames)
- Averaged to video-level: [1 video × 65 features]
- Used in encoder (not replicated/broadcast, single vector per video)

✅ **Technical specifications**: 
- Video processing rate (native fps, typically 30)
- Frame selection strategy (first 100 frames)
- Feature dimensions at each stage (65 per frame → 65 per video → 96 after encoder)
- Normalization approach (face-width normalization)
- Connection to visual encoder (65 → 192 → 96)




