# What Tools Are You Actually Using?

## Current Tool Usage:

### 1. Audio Processing:

**For CMU-MOSEI (Main Dataset):**
- ❌ **NOT using Librosa**
- ✅ **Using COVAREP** (pre-extracted from CMU-MOSEI dataset)
  - File: `CMU_MOSEI_COVAREP.csd`
  - 74 dimensions
  - Already pre-processed, you just load it

**For IEMOCAP (Combined Dataset):**
- ✅ **Using Librosa** (only for IEMOCAP extraction)
  - Extracts MFCC, Chroma, Spectral features
  - Located in `train_combined_final.py`, line 423-447
  - Only used when combining CMU-MOSEI + IEMOCAP

### 2. Text Processing:

**For CMU-MOSEI:**
- ❌ **NOT using BERT Tokenizer**
- ✅ **Using GloVe word vectors** (pre-extracted from CMU-MOSEI)
  - File: `CMU_MOSEI_TimestampedWordVectors.csd`
  - 300 dimensions (GloVe embeddings)
  - Already pre-processed, you just load it

**For IEMOCAP:**
- ❌ **NOT using BERT Tokenizer**
- ✅ **Using simple text features** (word count, char count, etc.)
  - Located in `train_combined_final.py`, line 518-549

### 3. Visual Processing:

**For CMU-MOSEI:**
- ❌ **NOT using Facemesh**
- ✅ **Using OpenFace2** (pre-extracted from CMU-MOSEI)
  - File: `CMU_MOSEI_VisualOpenFace2.csd`
  - 713 dimensions
  - Already pre-processed, you just load it

**For IEMOCAP:**
- ❌ **NOT using Facemesh**
- ✅ **Using MOCAP head motion data**
  - Extracted from `.txt` files
  - 713 dimensions (padded)

## Summary Table:

| Modality | CMU-MOSEI | IEMOCAP |
|----------|-----------|---------|
| **Audio** | COVAREP (pre-extracted) | Librosa (you extract) |
| **Text** | GloVe (pre-extracted) | Simple features (you extract) |
| **Visual** | OpenFace2 (pre-extracted) | MOCAP (you extract) |

## Can You Switch to Facemesh?

**Short Answer: Yes, but it's complex.**

### Current Situation:
- CMU-MOSEI provides **pre-extracted OpenFace2 features**
- You're loading ready-made features, not processing videos yourself
- No video files needed - just `.csd` files with features

### To Switch to Facemesh, You Would Need:

1. **Raw video files** (CMU-MOSEI videos, not just features)
   - CMU-MOSEI dataset should include videos, but you'd need to download them separately
   
2. **Facemesh processing code:**
   - Install MediaPipe: `pip install mediapipe`
   - Process each video frame-by-frame
   - Extract 468 landmarks
   - Compute features from landmarks
   
3. **Feature extraction:**
   - Use the `FACEMESH_FEATURE_EXTRACTION.py` I provided
   - Extract ~65 features per frame
   - Temporal averaging or pooling
   
4. **Integration:**
   - Replace `visual_dim=713` with `visual_dim=65` (or your chosen size)
   - Update model architecture
   - Retrain models

### Pros/Cons of Switching:

**Pros:**
- More control over features
- Can focus on emotion-specific landmarks
- Real-time processing capability
- Modern framework (MediaPipe)

**Cons:**
- Need raw videos (large download)
- Need to process all videos (time-consuming)
- Need to implement feature extraction
- OpenFace2 AUs are already optimized for emotion
- More code to maintain

### My Recommendation:

**For CMU-MOSEI: Stick with OpenFace2**
- Already optimized for emotion (AUs)
- Pre-processed and ready to use
- Proven to work well
- No extra processing needed

**Consider Facemesh if:**
- You have your own video dataset
- You want real-time processing
- You want more control over features
- You're building a new system from scratch

## Can You Switch to BERT Tokenizer?

**Yes, easier than Facemesh!**

### Current:
- Using GloVe (300-dim) pre-extracted embeddings
- Static word embeddings

### To Switch to BERT:

1. **Install:**
```python
pip install transformers torch
```

2. **Modify text processing:**
```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# For each text sample:
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
outputs = model(**inputs)
text_embeddings = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
```

3. **Update dimensions:**
- Change `text_dim=300` to `text_dim=768` (BERT base)
- Update model architecture
- Retrain

### Pros/Cons:

**Pros:**
- Contextual embeddings (better than static GloVe)
- State-of-the-art for NLP
- Better semantic understanding

**Cons:**
- Need raw text (transcripts), not pre-extracted embeddings
- Larger model (more compute)
- CMU-MOSEI provides GloVe, not raw text for BERT
- Would need to access original transcripts

## Final Answers:

### 1. Are you using Librosa?
- **For CMU-MOSEI:** ❌ No (using COVAREP)
- **For IEMOCAP:** ✅ Yes (for extraction only)

### 2. Are you using BERT Tokenizer?
- **No** - Using GloVe pre-extracted embeddings

### 3. Can you switch to Facemesh?
- **Yes**, but you'd need:
  - Raw video files
  - Processing pipeline
  - Feature extraction code
  - Model retraining
- **Recommendation:** Stick with OpenFace2 for now (already optimized for emotion)




