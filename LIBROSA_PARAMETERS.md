# Librosa Parameters Used in This Project

## Audio Loading Parameters

**Function:** `librosa.load()`

```python
y, sr = librosa.load(str(wav_file), sr=22050, duration=3.0)
```

**Parameters:**
- `sr=22050`: Sample rate of 22050 Hz (22.05 kHz)
- `duration=3.0`: Maximum duration of 3.0 seconds
- `y`: Audio time series (numpy array)
- `sr`: Sample rate

**Result:**
- Maximum audio length: 3 seconds × 22050 Hz = **66,150 samples**

---

## STFT Parameters (Implicit Defaults)

When using `librosa.feature.*` functions, librosa uses internal STFT with these **default parameters**:

### Default STFT Parameters:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `n_fft` | **2048 samples** | FFT window size (number of samples per FFT) |
| `hop_length` | **512 samples** | Number of samples between consecutive frames (default: `win_length // 4`) |
| `win_length` | **2048 samples** | Window length (defaults to `n_fft` if not specified) |
| `window` | **'hann'** | Window function (Hann window) |
| `center` | **True** | Pad signal so frames are centered |
| `dtype` | **complex64** | Output data type |

### Frame-Level vs Window-Level:

**Librosa operates at the FRAME-LEVEL:**
- Each **frame** is a segment of audio
- Frame length = `win_length` = 2048 samples
- Frames overlap by: `win_length - hop_length` = 2048 - 512 = **1536 samples**
- Overlap percentage: 1536/2048 = **75%**

**Time Resolution:**
- Frame duration: 2048 samples / 22050 Hz = **0.0929 seconds** (~93 ms per frame)
- Hop duration: 512 samples / 22050 Hz = **0.0232 seconds** (~23 ms between frames)
- For 3-second audio: ~3.0 / 0.0232 = **~129 frames**

---

## Feature Extraction Parameters

### 1. MFCC Features

```python
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
```

**Parameters:**
- `n_mfcc=13`: Number of MFCC coefficients to return
- **Underlying STFT**: Uses default parameters (n_fft=2048, hop_length=512, window='hann')

**Output:**
- Shape before `.mean()`: `[13, num_frames]` (13 MFCCs × number of frames)
- Shape after `.mean()`: `[13]` (single vector with 13 values - temporal average)
- **Vector size per window/frame**: 13 coefficients (before averaging)

**STFT → MFCC Pipeline:**
1. STFT: `[n_fft//2 + 1, num_frames]` = `[1025, num_frames]` (frequency bins × frames)
2. Mel spectrogram: `[n_mels, num_frames]` = `[128, num_frames]` (default 128 mel bands)
3. MFCC: `[n_mfcc, num_frames]` = `[13, num_frames]` (13 MFCCs × frames)

### 2. Chroma Features

```python
chroma = librosa.feature.chroma(y=y, sr=sr).mean(axis=1)
```

**Parameters:**
- Uses default parameters
- **Underlying STFT**: Uses default parameters (n_fft=2048, hop_length=512, window='hann')

**Output:**
- Shape before `.mean()`: `[12, num_frames]` (12 chroma bins × number of frames)
- Shape after `.mean()`: `[12]` (single vector with 12 values - temporal average)
- **Vector size per window/frame**: 12 chroma coefficients (before averaging)

### 3. Spectral Centroid

```python
spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
```

**Parameters:**
- Uses default parameters

**Output:**
- Shape: `[1, num_frames]` → averaged to single scalar
- **Vector size per window/frame**: 1 value (before averaging)

### 4. Spectral Rolloff

```python
spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
```

**Output:**
- Shape: `[1, num_frames]` → averaged to single scalar
- **Vector size per window/frame**: 1 value (before averaging)

### 5. Zero Crossing Rate

```python
zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
```

**Output:**
- Shape: `[1, num_frames]` → averaged to single scalar
- **Vector size per window/frame**: 1 value (before averaging)

---

## Final Feature Vector Composition

**Per audio file (after temporal averaging):**

```python
features = np.concatenate([
    mfcc,                    # 13 values
    chroma,                  # 12 values
    [spectral_centroid,      # 1 value
     spectral_rolloff,       # 1 value
     zero_crossing,          # 1 value
     tempo]                  # 1 value
])
```

**Total vector size:** 13 + 12 + 4 = **29 values**

**Final output:** Padded/truncated to **74 values** (as per code)

---

## STFT Output Details

### Frequency Resolution:

**For each frame/window:**
- STFT output: `[n_fft//2 + 1]` = `[2048//2 + 1]` = **1025 frequency bins**
- Frequency range: 0 to `sr/2` = 0 to 11025 Hz
- Frequency resolution: `sr / n_fft` = 22050 / 2048 = **10.77 Hz per bin**

**STFT Output Shape:**
- `[n_fft//2 + 1, num_frames]` = `[1025, num_frames]`
- Each column represents one frame (one window)
- Each row represents one frequency bin

---

## Summary Table

| Aspect | Value | Notes |
|-------|-------|-------|
| **Sample Rate** | 22050 Hz | Fixed by librosa.load() |
| **Max Duration** | 3.0 seconds | Fixed by librosa.load() |
| **STFT n_fft** | 2048 samples | Default librosa parameter |
| **STFT hop_length** | 512 samples | Default librosa parameter |
| **STFT win_length** | 2048 samples | Default (equals n_fft) |
| **STFT window** | 'hann' | Default Hann window |
| **Frame duration** | ~93 ms | 2048 / 22050 seconds |
| **Hop duration** | ~23 ms | 512 / 22050 seconds |
| **Overlap** | 75% | (2048-512)/2048 |
| **STFT output per frame** | 1025 frequency bins | n_fft//2 + 1 |
| **MFCC per frame** | 13 coefficients | Before temporal averaging |
| **Chroma per frame** | 12 coefficients | Before temporal averaging |
| **Final feature vector** | 29 values → 74 values | After concatenation and padding |

---

## Frame-Level vs Window-Level Explanation

**In librosa terminology:**
- **Frame** and **Window** are often used interchangeably
- Each frame IS a windowed segment of audio
- Librosa processes at the **frame-level**: each frame represents a short time segment

**Processing Flow:**
1. **Audio signal** → segmented into overlapping frames
2. **Each frame** → windowed (multiplied by Hann window)
3. **Windowed frame** → FFT → frequency domain representation
4. **Frequency domain** → feature extraction (MFCC, chroma, etc.)

**So to answer your question:**
- **Librosa uses frame-level processing**
- Each frame corresponds to one window (win_length samples)
- The term "window" refers to the time-domain segment
- The term "frame" refers to the same segment in the processing pipeline
- They're essentially the same thing in this context

---

## Example Calculation

**For a 3-second audio clip:**

1. **Total samples**: 3.0 × 22050 = 66,150 samples
2. **Number of frames**: (66150 - 2048) / 512 + 1 ≈ **126 frames**
3. **STFT output shape**: `[1025, 126]` (1025 freq bins × 126 frames)
4. **MFCC output shape**: `[13, 126]` (13 coeffs × 126 frames)
5. **After temporal averaging**: `[13]` (single vector)

---

## Code Reference

The parameters used are in:
- `train_combined_final.py`: Line 423-447 (`_extract_real_audio_features` method)
- Default librosa STFT parameters are implicit (not explicitly set)

If you want to **customize STFT parameters**, you would need to explicitly call `librosa.stft()` first:

```python
# Custom STFT parameters
stft = librosa.stft(y, n_fft=4096, hop_length=256, win_length=4096, window='hann')
# Then extract features from stft
mfcc = librosa.feature.mfcc(S=stft, n_mfcc=13)
```

But currently, the code uses librosa's default STFT parameters implicitly.

