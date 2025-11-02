# Detailed Librosa Analysis: Answering Your Questions

## 1. Why 93 ms and 23 ms? (Frame and Hop Durations)

### The Values:
- **Frame duration**: 93 ms (2048 samples / 22050 Hz = 0.0929 seconds)
- **Hop duration**: 23 ms (512 samples / 22050 Hz = 0.0232 seconds)

### Why These Values?

**These are librosa's DEFAULT parameters**, not something we chose. Librosa uses these defaults because they're optimized for **music analysis**:

1. **Time-Frequency Trade-off**:
   - 93 ms window provides good frequency resolution (captures spectral detail)
   - 93 ms is short enough to capture temporal changes
   - This is a **balance** - longer windows = better frequency resolution but worse time resolution

2. **Overlap Strategy**:
   - 512 sample hop = 1/4 of window length (2048 / 4 = 512)
   - 75% overlap (1536 samples overlap)
   - High overlap helps capture transient events (like speech onsets)
   - Standard practice: 50-75% overlap is typical

3. **Power-of-2 Optimization**:
   - 2048 and 512 are powers of 2
   - FFT algorithms are most efficient with power-of-2 sizes
   - Faster computation

### Are These Typical?

**For speech processing:**
- Often use **shorter windows** (256-512 samples = ~11-23 ms)
- Speech changes faster than music
- Better temporal resolution needed

**For music processing (librosa's target):**
- 93 ms is **typical and standard**
- Good for capturing musical notes and harmonics
- Standard in audio research

**For your project:**
- 93 ms works fine for sentiment analysis
- Captures prosodic features (tone, stress) which are important for emotion

---

## 2. What Aggregation Method Did We Use?

### Your Code:
```python
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
chroma = librosa.feature.chroma(y=y, sr=sr).mean(axis=1)
spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
```

### Aggregation Method: **TEMPORAL AVERAGING (Mean)**

**What happens:**
1. Librosa extracts features **per frame** (frame-level)
2. Each frame gets feature values (e.g., 13 MFCCs per frame)
3. For 3-second audio: ~129 frames
4. **`.mean(axis=1)`** averages across the **time axis** (across frames)

**Result:**
- Before: `[13, 129]` (13 MFCCs × 129 frames)
- After: `[13]` (13 average MFCC values)

### Other Aggregation Options (NOT used):

- **Standard Deviation**: `np.std()` - would capture variability
- **Percentiles**: `np.percentile()` - would capture distribution
- **Min/Max**: `np.min()`, `np.max()` - would capture extremes
- **Concatenation**: Keep all frames - would preserve temporal info

**Why Mean?**
- Reduces temporal complexity
- Creates fixed-size vector per sample
- Captures overall acoustic characteristics
- Standard for fixed-vector models

---

## 3. Final Feature Vector Size Per Window (Frame)

### Before Temporal Averaging (Per Frame/Window):

| Feature | Per Frame Size | Description |
|---------|---------------|-------------|
| **MFCC** | 13 coefficients | 13 MFCC values per frame |
| **Chroma** | 12 coefficients | 12 chroma bins per frame |
| **Spectral Centroid** | 1 value | Single scalar per frame |
| **Spectral Rolloff** | 1 value | Single scalar per frame |
| **Zero Crossing Rate** | 1 value | Single scalar per frame |
| **Tempo** | 1 value | Single scalar (global, not per frame) |

**Per-frame feature vector size**: 13 + 12 + 1 + 1 + 1 = **28 values per frame**

(Note: Tempo is computed globally, not per frame, but added to the final vector)

### After Temporal Averaging:

**Per-sample feature vector size**: 13 + 12 + 4 = **29 values**

Breakdown:
- 13 MFCCs (averaged)
- 12 Chroma features (averaged)
- 4 additional: spectral_centroid, spectral_rolloff, zero_crossing, tempo

### Final Output:

The code pads/truncates to **74 values** (see section 4).

---

## 4. Why Pad to 74? Alignment with 30-Frame Visual Timeline?

### The Code:
```python
if len(features) < 74:
    padding = np.zeros(74 - len(features))
    features = np.concatenate([features, padding])
else:
    features = features[:74]
```

### Why 74?

**Looking at your codebase:**
- Audio dimension: **74** (used consistently in model definitions)
- Visual dimension: **713** (OpenFace2 features)
- Text dimension: **300** (GloVe vectors)

**The 74 dimension seems ARBITRARY** - it doesn't match the 29 actual features extracted.

**Possible reasons:**
1. **Historical compatibility**: Matches CMU-MOSEI COVAREP audio features (which are 74-dim)
2. **Model architecture**: Model was designed for 74-dim audio input
3. **Standardization**: Ensures consistent input size across all samples

### CMU-MOSEI COVAREP:
- CMU-MOSEI uses **COVAREP** audio features, which are **74-dimensional**
- Your librosa extraction produces **29 features**
- Padding to 74 makes it compatible with existing architecture

### Visual Timeline (30 frames):

**CMU-MOSEI OpenFace2 Visual Features:**
- 713 dimensions total
- Structure: Likely **23-24 features per frame × 30 frames** ≈ 713 dim
- (713 / 30 ≈ 23.77 features per frame)

**Audio-Visual Alignment:**

**Visual**: 30 frames (likely 30 video frames)
**Audio**: ~129 frames (93 ms audio frames)

**They DON'T align frame-by-frame!**

**How they're aligned:**
- **Visual**: Temporal averaging across 30 video frames → 713-dim vector
- **Audio**: Temporal averaging across ~129 audio frames → 29-dim vector (padded to 74)

Both are **averaged to single vectors**, so they align at the **sample level**, not frame level.

### The Misalignment:

1. **Different frame rates**: Video ~30 fps, Audio ~43 fps (1/0.023 seconds)
2. **Different frame durations**: Video ~33 ms, Audio ~93 ms
3. **Both are averaged**: So frame-level misalignment doesn't matter

**Conclusion**: The 74 padding is for **model compatibility**, not temporal alignment. The temporal alignment happens through **aggregation** (averaging), not frame-level matching.

---

## 5. Are RMS, ZCR, and F0 Included?

### What Your Code Actually Extracts:

```python
mfcc = librosa.feature.mfcc(...)                    # ✅ MFCC
chroma = librosa.feature.chroma(...)                # ✅ Chroma
spectral_centroid = librosa.feature.spectral_centroid(...)  # ✅ Spectral
spectral_rolloff = librosa.feature.spectral_rolloff(...)    # ✅ Spectral
zero_crossing = librosa.feature.zero_crossing_rate(...)     # ✅ ZCR
tempo = librosa.beat.beat_track(...)                        # Tempo
```

### Breakdown:

| Feature | Included? | Notes |
|---------|-----------|-------|
| **MFCC** | ✅ Yes | 13 coefficients |
| **Chroma** | ✅ Yes | 12 bins |
| **Spectral Centroid** | ✅ Yes | Frequency centroid |
| **Spectral Rolloff** | ✅ Yes | Frequency rolloff |
| **ZCR** | ✅ Yes | Zero Crossing Rate |
| **RMS** | ❌ No | Not explicitly extracted |
| **F0 (Pitch)** | ⚠️ Partial | Implicit in Chroma, but not explicit F0 |

### F0 (Fundamental Frequency):

**Chroma features implicitly contain pitch information:**
- Chroma maps frequencies to 12 pitch classes
- Captures harmonic content related to F0
- But **not explicit F0 values**

**To get explicit F0:**
```python
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
```

**Your code does NOT extract explicit F0.**

### RMS (Root Mean Square Energy):

**Not explicitly extracted**, but:
- **Spectral centroid** correlates with energy (high energy → higher centroid)
- Could extract with: `librosa.feature.rms(y=y)`
- Not in your current code

### What You Have vs What You Could Add:

**Current:**
- MFCC (timbral)
- Chroma (harmonic/pitch-class)
- Spectral Centroid (brightness)
- Spectral Rolloff (high-frequency content)
- ZCR (noisiness)
- Tempo (rhythm)

**Missing:**
- RMS (explicit energy)
- F0 (explicit pitch)
- Spectral Bandwidth (frequency spread)
- Spectral Contrast (frequency contrast)

---

## Summary Table

| Question | Answer |
|----------|--------|
| **Why 93 ms frame?** | Librosa default, optimized for music analysis (time-frequency trade-off) |
| **Why 23 ms hop?** | Librosa default (1/4 window), 75% overlap for transient capture |
| **Aggregation method?** | **Temporal averaging (mean)** across frames |
| **Feature size per window?** | **28 values per frame** (before averaging) |
| **Final feature size?** | **29 values** (after averaging), **padded to 74** |
| **Why pad to 74?** | Model compatibility (CMU-MOSEI uses 74-dim COVAREP) |
| **30-frame visual alignment?** | Both are averaged, align at sample level, not frame level |
| **RMS included?** | ❌ No |
| **ZCR included?** | ✅ Yes |
| **F0 included?** | ⚠️ Partial (implicit in Chroma, not explicit) |

---

## Recommendations

### If You Want Explicit F0 and RMS:

```python
# Add explicit F0
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
f0_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0.0

# Add explicit RMS
rms = librosa.feature.rms(y=y).mean()

# Add to features
features = np.concatenate([
    mfcc,                    # 13
    chroma,                  # 12
    [spectral_centroid,      # 1
     spectral_rolloff,       # 1
     zero_crossing,          # 1
     tempo,                  # 1
     f0_mean,                # 1 (NEW)
     rms]                    # 1 (NEW)
])
# Total: 13 + 12 + 6 = 31 values (after adding F0 and RMS)
```

### If You Want Frame-Level Features (No Averaging):

```python
# Keep temporal dimension
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Shape: [13, num_frames]
chroma = librosa.feature.chroma(y=y, sr=sr)        # Shape: [12, num_frames]

# Concatenate per frame
frame_features = np.vstack([mfcc, chroma])  # Shape: [25, num_frames]

# Then either:
# 1. Keep all frames (for sequence models)
# 2. Use max pooling instead of mean
# 3. Use attention to weight frames
```

But this would require changes to your model architecture to handle variable-length sequences.

---

## Conclusion

Your current librosa implementation:
1. Uses standard defaults (93 ms frame, 23 ms hop)
2. Uses temporal averaging (mean) for aggregation
3. Extracts 29 features (padded to 74 for compatibility)
4. Includes ZCR but not explicit RMS or F0
5. Aligns with visual at sample level (both averaged), not frame level

The 74-dimension is for **model compatibility** with CMU-MOSEI COVAREP, not for temporal alignment with 30-frame visual timeline.

