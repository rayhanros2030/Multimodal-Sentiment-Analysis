# Visual Pipeline Analysis: Facemesh vs OpenFace2

## Important Clarification

**Your current pipeline uses OpenFace2 (CMU-MOSEI), NOT Facemesh.**

However, I'll answer your Facemesh questions and explain both systems.

---

## Part 1: What You're Currently Using (OpenFace2)

### CMU-MOSEI Visual Features (OpenFace2):

**Source:** `CMU_MOSEI_VisualOpenFace2.csd`
**Dimension:** 713 features per sample
**Temporal Processing:** Temporal averaging (mean across frames)

### OpenFace2 Features (713 dimensions):

OpenFace2 extracts **high-level facial features**, NOT raw landmarks:

1. **Action Units (AUs)** - ~17 AUs with intensity (0-5 scale)
   - AU01: Inner Brow Raiser
   - AU02: Outer Brow Raiser
   - AU04: Brow Lowerer
   - AU05: Upper Lid Raiser
   - AU06: Cheek Raiser
   - AU07: Lid Tightener
   - AU09: Nose Wrinkler
   - AU10: Upper Lip Raiser
   - AU12: Lip Corner Puller
   - AU14: Dimpler
   - AU15: Lip Corner Depressor
   - AU17: Chin Raiser
   - AU20: Lip Stretcher
   - AU23: Lip Tightener
   - AU25: Lips Part
   - AU26: Jaw Drop
   - AU45: Blink

2. **Pose Features** (6 dimensions)
   - Head rotation: pitch, yaw, roll
   - Head translation: x, y, z

3. **Gaze Features** (4 dimensions)
   - Gaze direction (x, y)
   - Gaze angle (x, y)

4. **Facial Shape Features** (56 dimensions)
   - Facial landmark positions (normalized)

5. **Appearance Features** (600+ dimensions)
   - Texture/appearance representation

**Total:** ~713 dimensions

### Your Temporal Extraction:

```python
def _extract_features(self, data: Dict, target_dim: int) -> np.ndarray:
    features = data['features']  # Shape: [num_frames, 713]
    if len(features.shape) > 1:
        features = np.mean(features, axis=0)  # Temporal averaging
    return features  # Final: [713]
```

**Method:** Temporal averaging (mean across frames)
- Input: `[num_frames, 713]` (frame-level features)
- Output: `[713]` (sample-level, averaged)

**No frame-by-frame preservation** - all temporal information is averaged.

---

## Part 2: Facemesh (MediaPipe) - Answering Your Questions

### 1. What Features from 468 Landmarks?

#### Raw Coordinates:
- **468 landmarks** × **3 coordinates** (x, y, z) = **1404 dimensions**
- Too high-dimensional for direct use
- Usually reduced via feature engineering

#### Distances:
Common distances for emotion recognition:
- **Eye Features:**
  - Left eye width (landmarks 33, 133)
  - Right eye width (landmarks 362, 263)
  - Eye opening (vertical distances)
  - Inter-eye distance (landmarks 33, 263)
  
- **Mouth Features:**
  - Mouth width (landmarks 61, 291)
  - Mouth height (landmarks 13, 14)
  - Lip thickness
  
- **Eyebrow Features:**
  - Eyebrow height (landmarks 107, 336)
  - Eyebrow angle
  
- **Facial Symmetry:**
  - Asymmetry metrics (left vs right distances)

**Typical:** 20-50 distance features

#### Angles:
- **Eye corners angle:** For eye openness
- **Mouth corners angle:** For smile/frown
- **Eyebrow angle:** For frown/surprise
- **Facial tilt:** For head pose

**Typical:** 10-20 angle features

#### Ratios/Proportions:
- Eye width / face width
- Mouth width / face width
- Nose width / face width
- Face symmetry ratios

**Typical:** 5-15 ratio features

#### Geometric Features:
- **Facial Action Coding System (FACS) features:**
  - Computed from landmark relationships
  - Similar to OpenFace2 AUs but derived from landmarks

### 2. Final Vector Dimensions per Window/Frame?

**Per Frame (Window):**
- Raw coordinates: 468 × 3 = 1404 (too large)
- **Recommended feature engineering:**
  - 30-40 distances: ~35 dim
  - 15-20 angles: ~18 dim
  - 10-15 ratios: ~12 dim
  - **Total: ~65-85 dimensions per frame**

**After Temporal Processing:**
- Keep all frames: `[num_frames, 65]` (sequence)
- Temporal averaging: `[65]` (fixed vector)
- Temporal pooling (max/mean): `[65]` (fixed vector)

### 3. Temporal Extraction: Frame-by-Frame?

**Yes, typically frame-by-frame:**

```python
# Pseudocode for Facemesh temporal extraction
for frame in video_frames:
    landmarks = facemesh.process(frame)  # 468 landmarks × 3 coords
    features = extract_features(landmarks)  # [65] per frame
    frame_features.append(features)

# Result: [num_frames, 65]
```

**Then aggregate:**
- **Option 1: Temporal averaging** (what you do for OpenFace2)
  ```python
  features = np.mean(frame_features, axis=0)  # [65]
  ```
- **Option 2: Sequence preservation** (for RNN/LSTM)
  ```python
  features = frame_features  # [num_frames, 65]
  ```
- **Option 3: Temporal pooling** (max/mean)
  ```python
  features = np.concatenate([
      np.mean(frame_features, axis=0),  # [65]
      np.max(frame_features, axis=0),   # [65]
      np.std(frame_features, axis=0)    # [65]
  ])  # [195]
  ```

### 4. Feature Processing After Extraction?

**Typical pipeline:**

1. **Normalization:**
   - Face size normalization (scale by face width)
   - Face position normalization (center face)
   
2. **Feature Engineering:**
   - Compute distances, angles, ratios
   - Remove redundant features
   
3. **Dimension Reduction (optional):**
   - PCA to reduce dimensions
   - Feature selection
   
4. **Temporal Processing:**
   - Temporal averaging
   - Temporal smoothing (moving average)
   - Temporal differentials (velocity/acceleration)

**Example processing:**
```python
def process_facemesh_features(landmarks):
    # 1. Normalize face size
    face_width = distance(landmarks[0], landmarks[16])
    normalized_landmarks = landmarks / face_width
    
    # 2. Extract features
    eye_width_left = distance(normalized_landmarks[33], normalized_landmarks[133])
    eye_width_right = distance(normalized_landmarks[362], normalized_landmarks[263])
    mouth_width = distance(normalized_landmarks[61], normalized_landmarks[291])
    # ... more features
    
    # 3. Compute ratios
    eye_ratio = eye_width_left / eye_width_right
    mouth_face_ratio = mouth_width / face_width
    
    # 4. Combine features
    features = np.array([eye_width_left, eye_width_right, mouth_width, 
                         eye_ratio, mouth_face_ratio, ...])  # [65]
    return features
```

### 5. Temporal Alignment?

**With Audio:**
- Audio: ~43 fps (93 ms frames)
- Video: 30 fps (33 ms frames)
- **Misalignment:** Different frame rates

**Solutions:**
1. **Upsample/Downsample:**
   - Resample to common frame rate (e.g., 30 fps)
   
2. **Temporal Averaging (current approach):**
   - Average both modalities to single vectors
   - Aligns at sample level, not frame level
   
3. **Frame-level Alignment:**
   - Interpolate features to common timestamps
   - Use temporal windows

**With Text:**
- Text: Word-level timestamps (variable)
- Visual: Frame-level (30 fps)
- **Alignment:** Match word timestamps to video frames

### 6. Integration into Multimodal Pipeline?

**Current Pipeline (OpenFace2):**

```
Visual (713) → Encoder → Embedding (96) ──┐
                                          ├─→ Cross-Attention → Fusion → Output
Audio (74)  → Encoder → Embedding (96) ──┤
                                          │
Text (300)  → Encoder → Embedding (96) ──┘
```

**If Using Facemesh (65 dim):**

```
Visual (65) → Encoder → Embedding (96) ──┐
                                         ├─→ Cross-Attention → Fusion → Output
Audio (74) → Encoder → Embedding (96) ──┤
                                         │
Text (300) → Encoder → Embedding (96) ──┘
```

**Changes needed:**
- Update `visual_dim=65` instead of `visual_dim=713`
- Encoder will have different input size
- Rest of architecture remains same

### 7. Most Relevant Landmarks/Features for Emotion?

**Critical Landmarks for Emotion:**

#### Happiness (AU12):
- **Mouth corners** (landmarks 61, 291) - pulled up
- **Lip corners** (landmarks 48, 54) - curvature
- **Eye corners** (landmarks 33, 263) - squinting
- **Cheek regions** (landmarks 116, 345) - raised

**Features:**
- Mouth corner distance (wider = happier)
- Lip corner angle (upward = smile)
- Eye opening (narrower = genuine smile)

#### Sadness (AU15):
- **Mouth corners** (landmarks 61, 291) - pulled down
- **Inner eyebrows** (landmarks 21, 251) - raised inner
- **Upper lip** (landmarks 61-65) - depressed

**Features:**
- Mouth corner distance (narrower)
- Lip corner angle (downward)
- Eyebrow inner height (raised)

#### Anger (AU4, AU7, AU23):
- **Eyebrows** (landmarks 21, 22, 251, 252) - lowered
- **Eyes** (landmarks 33, 133, 362, 263) - tightened
- **Mouth** (landmarks 61, 291) - tightened

**Features:**
- Eyebrow angle (lowered)
- Eye opening (reduced)
- Mouth width (narrower)

#### Surprise (AU1, AU2, AU5, AU26):
- **Eyebrows** (landmarks 21, 22, 251, 252) - raised
- **Eyes** (landmarks 33, 133, 362, 263) - wide open
- **Mouth** (landmarks 61, 291, 13, 14) - open

**Features:**
- Eyebrow height (raised)
- Eye opening (increased)
- Mouth opening (increased)

#### Disgust (AU9, AU10):
- **Nose** (landmarks 1, 2, 5, 4) - wrinkled
- **Upper lip** (landmarks 61-65) - raised
- **Mouth corners** (landmarks 61, 291) - asymmetric

#### Fear (AU1, AU2, AU4, AU5, AU20, AU26):
- **Eyebrows** - raised and pulled together
- **Eyes** - wide open
- **Mouth** - stretched horizontally

**Top Priority Landmarks:**
1. **Mouth region** (61, 291, 13, 14, 48, 54) - most expressive
2. **Eye region** (33, 133, 362, 263, 7, 8, 159, 160) - emotional eyes
3. **Eyebrow region** (21, 22, 23, 24, 251, 252, 253, 254) - brow movements
4. **Cheek region** (116, 117, 345, 346) - smile lines
5. **Nose region** (1, 2, 4, 5) - nose wrinkle

**Recommended Features (Emotion-Focused):**
- **Mouth features** (15-20): width, height, corner positions, angles
- **Eye features** (10-15): opening, width, corner positions
- **Eyebrow features** (8-12): height, angle, inner/outer positions
- **Symmetry features** (5-8): left-right asymmetry
- **Overall pose** (6): head rotation/translation

**Total: ~50-60 emotion-relevant features**

---

## Comparison: OpenFace2 vs Facemesh

| Aspect | OpenFace2 (Current) | Facemesh (Proposed) |
|--------|---------------------|---------------------|
| **Input** | Video frames | Video frames |
| **Output** | 713 high-level features | 468 raw landmarks |
| **Features** | AUs, pose, gaze, appearance | Raw coordinates |
| **Processing** | Pre-processed (already features) | Requires feature engineering |
| **Temporal** | Averaged to [713] | Can preserve [num_frames, 65] |
| **Emotion Relevance** | High (AUs designed for emotion) | High (with proper feature extraction) |
| **Complexity** | Lower (ready to use) | Higher (need to extract features) |

---

## Recommendations

### If You Want to Use Facemesh:

1. **Feature Extraction:**
   - Focus on emotion-relevant landmarks (mouth, eyes, eyebrows)
   - Extract 50-60 features per frame
   - Use distances, angles, ratios

2. **Temporal Processing:**
   - Option A: Temporal averaging (like current OpenFace2)
   - Option B: Preserve temporal dimension for sequence models

3. **Integration:**
   - Replace `visual_dim=713` with `visual_dim=65` (or your chosen size)
   - Rest of pipeline remains same

4. **Advantages:**
   - More control over features
   - Can focus on emotion-specific landmarks
   - Can preserve temporal information if needed

5. **Disadvantages:**
   - More preprocessing required
   - Need to implement feature extraction
   - OpenFace2 AUs are already optimized for emotion

### If You Want to Keep OpenFace2:

**Advantages:**
- Already optimized for emotion (AUs)
- Pre-processed, ready to use
- 713 dimensions capture comprehensive facial information

**You could improve temporal processing:**
- Currently: Temporal averaging (loss of temporal info)
- Alternative: Temporal pooling (mean + max + std) → ~2139 dim
- Alternative: Keep sequence for RNN/LSTM models

---

## Summary

**Your Current System (OpenFace2):**
- 713 dimensions (AUs, pose, gaze, appearance)
- Temporal averaging: `[num_frames, 713] → [713]`
- Aligned with audio/text at sample level (both averaged)

**Facemesh Alternative:**
- 468 landmarks → 50-65 features per frame
- Can preserve temporal: `[num_frames, 65]`
- Requires feature engineering
- Emotion-relevant landmarks: mouth, eyes, eyebrows

**For Emotion Recognition:**
- Both work well
- OpenFace2: Pre-optimized AUs
- Facemesh: More control, need proper feature extraction

