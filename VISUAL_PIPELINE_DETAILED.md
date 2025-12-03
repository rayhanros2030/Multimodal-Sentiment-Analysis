# Visual Pipeline - Complete Technical Details

## Current Implementation (From Code):

### 1. Temporal Structure

**Actual Structure**: **[1 video × 65 features]**

**Process**:
1. Extract features from each frame: [1 frame × 65 features]
2. Store all frame features: [N frames × 65 features] (N ≤ 100)
3. Temporal averaging: `np.mean(frame_features, axis=0)`
4. Final output: **[1 video × 65 features]**

**NOT**: [30 frames × 65 features] - There is no 30-frame architecture in your code.

**Code evidence**:
```python
frame_features = []  # List to store per-frame features
# ... extract 65 features per frame ...
frame_features.append(features)  # Add each frame's 65 features
# Temporal averaging
return np.mean(frame_features, axis=0).astype(np.float32)  # → [65]
```

### 2. Variable-Length Video Handling

**Current Strategy**: **Simple averaging with frame limit**

**Implementation**:
- Process up to 100 frames (`max_frames = 100`)
- If video has < 100 frames: Average all available frames
- If video has > 100 frames: Sample first 100 frames and average
- If no face detected: Zero vector [65 features]

**Problems with current approach**:
- ❌ No explicit sampling strategy (just takes first 100)
- ❌ No interpolation for very short videos
- ❌ No segmentation for very long videos
- ❌ Doesn't align with "30-frame" mention (if that was intended)

**Better approaches**:
- **Option A**: Uniform sampling (sample 30-100 frames evenly spaced)
- **Option B**: Temporal segmentation (split video into segments, extract from each)
- **Option C**: Fixed window approach (process exactly 30 frames if that was intended)

### 3. The 53 Remaining Features

**Current Implementation** (from code):
```python
# First 12 features explicitly defined:
# - Mouth: 5 features (width, height, corners, angle)
# - Eyes: 3 features (left width, right width, inter-eye)
# - Eyebrows: 2 features (left height, right height)
# - Symmetry: 2 features (eye symmetry, mouth symmetry)
# Total so far: 12 features

# Fill remaining 53 features:
while len(features) < 65:
    idx = len(features)  # Current index (12, 13, 14, ..., 64)
    if idx < len(normalized):  # If idx < 468
        features.append(np.linalg.norm(normalized[idx]))  # Norm of landmark[idx]
    else:  # If idx >= 468 (which won't happen for 53 features)
        i1, i2 = (idx % 468, (idx * 7) % 468)
        features.append(np.linalg.norm(normalized[i1] - normalized[i2]))  # Distance
```

**The 53 features are**:
- **Features 12-467 (if < 468)**: Euclidean norm of normalized landmark coordinates
  - Feature 12 = ||normalized[12]||
  - Feature 13 = ||normalized[13]||
  - ...
  - Feature 64 = ||normalized[64]||
- Since we only need 53 more features (65 - 12 = 53), we get:
  - Features 12-64 = norms of landmarks[12] through landmarks[64]
  - That's exactly 53 features!

**So the 53 features are**:
- Norms of normalized landmark coordinates at indices 12-64
- These are geometric features capturing landmark magnitudes relative to face width

**Issue**: This selection is somewhat arbitrary. Better approach would be:
- Select emotion-relevant landmarks explicitly
- Or use principled dimensionality reduction (PCA)
- Or compute more meaningful geometric features (distances, angles between key landmarks)

### 4. Frame-Level vs Video-Level

**Current Implementation**:
- **Frame-level**: Extract 65 features per frame → [N frames × 65]
- **Video-level**: Average across frames → [1 video × 65]
- **Model input**: [batch_size × 65] (single vector per video)

**What this means**:
- Model receives ONE feature vector per video (not a sequence)
- No temporal modeling in the architecture
- Temporal information is lost in averaging

### 5. Normalization Order

**Current Pipeline** (correct order):
1. Extract 468 landmarks: (x, y, z) coordinates
2. Compute face width: `||landmarks[0] - landmarks[16]||`
3. Normalize landmarks: `normalized = landmarks / face_width`
4. Extract features from normalized landmarks

**This is correct!** Normalization happens before feature extraction.

### 6. Connection to Architecture

**Input to Model**:
- Visual: [batch_size × 65] (single vector per video)
- Audio: [batch_size × 74]
- Text: [batch_size × 300]

**Through Encoder**:
- Visual encoder (Linear 65 → hidden_dim → embed_dim)
- Output: [batch_size × embed_dim] (e.g., 96)

**No temporal dimension** - the model processes single feature vectors, not sequences.

---

## Revised Visual Pipeline Description (Accurate):

For visual feature extraction, we process video frames using MediaPipe FaceMesh to extract 468 3D facial landmarks per frame. FaceMesh provides real-time landmark estimation, estimating landmark coordinates in screen space (X, Y normalized coordinates, Z relative depth). To handle variable-length videos, we process up to the first 100 frames per video for computational efficiency. For each frame, we derive 65-dimensional emotion-focused features through geometric computations. 

First, landmarks are normalized by face width (computed as the Euclidean distance between landmarks at indices 0 and 16) to handle scale variation. We then extract 12 explicitly defined emotion-relevant features: mouth characteristics including mouth width (distance between landmarks 61 and 291), mouth height (distance between landmarks 13 and 14), left and right mouth corner Y-coordinates (landmarks 61 and 291), and mouth corner angle computed using arctangent—yielding 5 features. Eye features comprise left eye width (distance between landmarks 33 and 133), right eye width (distance between landmarks 362 and 263), and inter-eye distance (distance between landmarks 33 and 263)—yielding 3 features. Eyebrow features are computed as average heights: left eyebrow using landmarks 21, 55, and 107, and right eyebrow using landmarks 251, 285, and 336—yielding 2 features. Symmetry metrics include eye symmetry (normalized absolute difference between left and right eye widths) and mouth symmetry (absolute difference between left and right corner Y-coordinates)—yielding 2 features. 

The remaining 53 features are derived from normalized landmark magnitudes: we compute the Euclidean norm of normalized landmark coordinates at indices 12 through 64, providing additional geometric information about facial structure relative to face scale. Features are computed per frame and temporally averaged across all processed frames to obtain a single 65-dimensional representation per video: [1 video × 65 features]. This video-level representation is then passed through the visual encoder (Linear layers: 65 → 192 → 96) to obtain the encoded visual representation used in cross-modal attention and fusion.

---

## Quick Fixes Needed:

### Fix 1: Clarify Temporal Structure ✅
**Added**: "temporally averaged across all processed frames to obtain a single 65-dimensional representation per video: [1 video × 65 features]"

### Fix 2: Specify the 53 Features ✅
**Added**: "Euclidean norm of normalized landmark coordinates at indices 12 through 64"

### Fix 3: Define Frame Processing ✅
**Added**: "process up to the first 100 frames per video for computational efficiency"

### Fix 4: Clarify Normalization Order ✅
**Already correct**: Normalization happens before feature extraction

### Fix 5: Connect to Architecture ✅
**Added**: "passed through the visual encoder (Linear layers: 65 → 192 → 96)"

---

## Potential Improvements:

1. **Better Frame Sampling**: Instead of first 100 frames, use uniform sampling or keyframe selection
2. **More Meaningful 53 Features**: Select emotion-relevant landmarks explicitly (e.g., Action Unit regions)
3. **Temporal Modeling** (if you want): Keep frame-level features and use LSTM/Transformer for temporal modeling
4. **Variable Frame Handling**: Interpolate for very short videos, segment for very long videos

But for now, the description above accurately reflects your current implementation!




