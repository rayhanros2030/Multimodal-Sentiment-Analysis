# Clarification: You're Using OpenFace2, NOT Facemesh!

## Evidence from Your Code:

### 1. File Path (Line 165):
```python
visual_path = self.mosei_dir / 'visuals' / 'CMU_MOSEI_VisualOpenFace2.csd'
```
**Name says "OpenFace2"** ✅

### 2. Data Loading (Line 175):
```python
visual_data = self._load_csd_file(visual_path, 'OpenFace_2') or self._load_csd_file(visual_path, 'Visual')
```
**Looking for 'OpenFace_2' key** ✅

### 3. CMU-MOSEI Dataset:
CMU-MOSEI dataset **pre-processes visual features using OpenFace2**, not Facemesh.

## What You're Actually Using:

**OpenFace2** (from CMU):
- Pre-extracted features from CMU-MOSEI dataset
- 713 dimensions
- Includes: Action Units (AUs), pose, gaze, appearance features
- Already processed - you just load the `.csd` files

## What Facemesh Would Be:

**MediaPipe FaceMesh** (from Google):
- Real-time face detection library
- 468 facial landmarks
- You'd need to process videos yourself
- Extract features from landmarks
- Not in CMU-MOSEI dataset

## The Confusion:

You asked about "Facemesh" and "468 landmarks" - but:
- **CMU-MOSEI doesn't use Facemesh**
- **CMU-MOSEI uses OpenFace2** (different tool, different features)
- OpenFace2 extracts higher-level features (AUs), not raw 468 landmarks

## Your Actual Visual Pipeline:

```
CMU-MOSEI Dataset
    ↓
CMU_MOSEI_VisualOpenFace2.csd file
    ↓
Pre-extracted OpenFace2 features (713 dim)
    ↓
Your code loads them
    ↓
Temporal averaging → [713] vector
    ↓
Your model
```

**No Facemesh involved!**

## If You Want Facemesh:

You'd need to:
1. Download raw videos (not CMU-MOSEI's pre-processed features)
2. Process videos with MediaPipe FaceMesh
3. Extract 468 landmarks
4. Compute features from landmarks
5. Then use those features

But currently, you're using the **ready-made OpenFace2 features** from CMU-MOSEI.

## Summary:

- ✅ **You're using:** OpenFace2 (from CMU-MOSEI)
- ❌ **You're NOT using:** MediaPipe FaceMesh
- 📁 **Your data source:** CMU-MOSEI pre-processed features
- 🔧 **Your involvement:** Just loading and averaging features

The whole time you've been working with **OpenFace2**, not Facemesh!

