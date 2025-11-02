# Preprocessing Comparison: CMU-MOSEI vs IEMOCAP

## Both Datasets Use Preprocessing, But Different Approaches

### CMU-MOSEI Preprocessing

**Feature Source:** Pre-extracted features stored in .csd (HDF5) files
- Visual: OpenFace2 features (713-dim) - already extracted
- Audio: COVAREP features (74-dim) - already extracted  
- Text: GloVe word vectors (300-dim) - already extracted
- Labels: Sentiment scores - already extracted

**Preprocessing Applied:**
1. **Feature Extraction:**
   - Loads from .csd files
   - Temporal averaging (mean across time dimension)
   - Padding/truncation to target dimensions

2. **Feature Cleaning:**
   ```python
   - NaN/Inf replacement: nan=0.0, posinf=1.0, neginf=-1.0
   - Value clipping: [-1000, 1000]
   ```

3. **Sentiment Cleaning:**
   - Handles NaN/Inf
   - Clips to [-3, 3]

4. **Normalization:**
   - RobustScaler fitted on dataset
   - Applied to all features

### IEMOCAP Preprocessing

**Feature Source:** Raw files (need to extract features)
- Audio: .wav files → librosa feature extraction
- Visual: MOCAP .txt files → head motion data extraction
- Text: Transcription .txt files → text statistics extraction
- Labels: EmoEvaluation .txt files → valence score extraction

**Preprocessing Applied:**

1. **Audio Feature Extraction:**
   - Uses librosa to extract from .wav files
   - MFCC (13), Chroma (12), Spectral features (9)
   - Combines to 74-dim vector

2. **Visual Feature Extraction:**
   - Reads MOCAP head motion data (.txt files)
   - Extracts: pitch, roll, yaw, translations (mean, std, min, max)
   - Creates 713-dim feature vector (matches MOSEI dimension)

3. **Text Feature Extraction:**
   - Reads transcription files
   - Extracts: word count, char count, avg word length, sentence count
   - Character frequency features
   - Creates 300-dim feature vector

4. **Sentiment Extraction:**
   - Parses EmoEvaluation files for valence scores
   - Converts from 1-5 scale to -3 to 3 scale
   - Fallback to emotion labels if valence unavailable

5. **Feature Cleaning:**
   - Similar to MOSEI: NaN/Inf handling, clipping
   - But IEMOCAP features are "fresher" - just extracted, less likely to have extreme issues

6. **Normalization:**
   - RobustScaler fitted on dataset
   - Applied to all features (same as MOSEI)

## Key Differences

| Aspect | CMU-MOSEI | IEMOCAP |
|--------|-----------|---------|
| **Feature Source** | Pre-extracted (.csd files) | Raw files (extract on-the-fly) |
| **Extraction** | Just temporal averaging | Full feature extraction (librosa, parsing) |
| **Audio Quality** | Already in features (may have -inf) | Extracted fresh (should be cleaner) |
| **Visual Source** | OpenFace2 features | MOCAP head motion data |
| **Text Source** | GloVe embeddings | Text statistics |
| **Cleaning** | Same approach (NaN/Inf, clipping) | Same approach |
| **Normalization** | RobustScaler | RobustScaler |

## Preprocessing Steps Summary

### CMU-MOSEI:
1. Load pre-extracted features from .csd
2. Average across time dimension
3. Clean (NaN/Inf, clipping)
4. Normalize with RobustScaler

### IEMOCAP:
1. Extract features from raw files:
   - Audio: librosa from .wav
   - Visual: Parse MOCAP .txt
   - Text: Parse transcription .txt
   - Sentiment: Parse EmoEvaluation .txt
2. Clean (NaN/Inf, clipping)
3. Normalize with RobustScaler

## Important Note

**Both datasets get the SAME cleaning and normalization**, but:
- **CMU-MOSEI** features are already extracted (may have pre-existing quality issues)
- **IEMOCAP** features are extracted fresh (should be cleaner quality)

This might explain why:
- IEMOCAP audio might be better quality than CMU-MOSEI audio
- CMU-MOSEI audio has more -inf values
- Different preprocessing needs for different data sources

## For Your Regeneron Study

You can say:
- "Applied consistent preprocessing pipeline to both datasets"
- "CMU-MOSEI uses pre-extracted features; IEMOCAP features extracted from raw data"
- "RobustScaler normalization ensures feature compatibility across datasets"
- "Cleaning procedures handle dataset-specific issues (NaN/Inf values)"

Both datasets ARE preprocessed, just at different stages!

