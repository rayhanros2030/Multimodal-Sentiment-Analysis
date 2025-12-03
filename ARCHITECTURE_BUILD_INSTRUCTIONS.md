# How to Build the Architecture Diagram in Draw.io

Since the file isn't opening, here's how to build it manually:

## Step-by-Step Instructions

### 1. Open Draw.io
- Go to https://app.diagrams.net/
- Click "Create New Diagram"
- Choose "Blank Diagram"

### 2. Set Canvas Size
- Right-click canvas → "Edit Style" or go to File → Page Setup
- Set width: 1600px, height: 2400px

---

## Layer by Layer Building:

### LAYER 1: INPUT MODALITIES (Top)

1. **Create 3 boxes horizontally** (use Rectangle shape):
   - Box 1: "Video" (RGB Frames) - Yellow background (#FFF2CC)
   - Box 2: "Audio" (Waveform) - Yellow background  
   - Box 3: "Text" (Transcript) - Yellow background
   - Position: Top row, evenly spaced

2. **Add header**: "LAYER 1: INPUT MODALITIES" (Blue background #E1F5FF)

---

### LAYER 2: FEATURE EXTRACTION

#### CMU-MOSEI Path (Training):
1. **Add label**: "CMU-MOSEI: Pre-extracted Features (Training)" - Purple background (#E1D5E7)

2. **Three boxes below label**:
   - "OpenFace2 - Visual Features [713-dim]" - Green (#D5E8D4)
   - "COVAREP - Audio Features [74-dim]" - Green
   - "GloVe - Text Features [300-dim]" - Green

#### CMU-MOSI Path (Testing):
1. **Add label**: "CMU-MOSI: Real-time Feature Extraction (Testing)" - Red background (#F8CECC)

2. **Visual path** (3 boxes in a row):
   - "FaceMesh - 468 Landmarks → Emotion Features [65-dim]" - Orange (#FFE6CC)
   - "Feature Adapter [65 → 713]" - Blue (#DAE8FC), **BOLD**

3. **Audio path** (2 boxes):
   - "Librosa - MFCC + Chroma + Spectral → [74-dim]" - Orange
   - "No Adapter (Already 74-dim)" - Light Blue

4. **Text path** (2 boxes):
   - "BERT Tokenizer + BERT Model → [768-dim]" - Orange
   - "Feature Adapter [768 → 300]" - Blue, **BOLD**

---

### LAYER 3: MODALITY ENCODERS

1. **Add header**: "LAYER 3: MODALITY ENCODERS (Linear Layers)"

2. **Three encoder boxes** (Gray #F5F5F5):
   - **Visual Encoder**:
     ```
     Linear(713 → 192)
     BatchNorm + ReLU + Dropout(0.7)
     Linear(192 → 96)
     BatchNorm
     Output: [96-dim]
     ```
   
   - **Audio Encoder**:
     ```
     Linear(74 → 192)
     BatchNorm + ReLU + Dropout(0.7)
     Linear(192 → 96)
     BatchNorm
     Output: [96-dim]
     ```
   
   - **Text Encoder**:
     ```
     Linear(300 → 192)
     BatchNorm + ReLU + Dropout(0.7)
     Linear(192 → 96)
     BatchNorm
     Output: [96-dim]
     ```

3. **Connect adapters to encoders** (dashed blue lines from MOSI adapters to encoders)

---

### LAYER 4: CROSS-MODAL FUSION

1. **Add header**: "LAYER 4: CROSS-MODAL FUSION"

2. **MultiheadAttention box** (Purple #E1D5E7):
   ```
   MultiheadAttention
   (4 heads, dropout=0.8)
   Stack features: [batch × 3 × 96]
   ```

3. **Concatenation box** (Blue #DAE8FC):
   ```
   Concatenation
   [96, 96, 96] → [288]
   ```

4. **Fusion MLP box** (Yellow #FFF2CC):
   ```
   Fusion MLP
   Linear(288 → 192)
   BatchNorm + ReLU + Dropout(0.7)
   Linear(192 → 96)
   BatchNorm + ReLU + Dropout(0.7)
   Linear(96 → 1)
   ```

5. **Connect**: Encoders → Attention → Concatenation → MLP

---

### LAYER 5: OUTPUT

1. **Output box** (Green #D5E8D4, **BOLD**):
   ```
   Sentiment Score
   Single Regression Output
   Range: [-3, +3]
   (Continuous sentiment intensity)
   ```

2. **Connect**: Fusion MLP → Output

---

### LOSS FUNCTION (Bottom)

1. **Box** (Yellow #FFF2CC):
   ```
   Improved Correlation Loss
   L = α × (MSE + MAE)/2 + β × (1 - Pearson Corr)²
   α = 0.3 (accuracy weight), β = 0.7 (correlation weight)
   ```

---

## Key Notes Box:

Add a note at the top (Red background #F8CECC):
```
NOTE: Features are aggregated (temporally averaged) before model input.
No temporal sequences - single feature vectors only.
```

---

## Color Scheme:

- **Yellow** (#FFF2CC): Inputs, Fusion, Loss
- **Green** (#D5E8D4): MOSEI features, Output
- **Orange** (#FFE6CC): MOSI real-time extraction
- **Blue** (#DAE8FC): Feature adapters, Concatenation
- **Purple** (#E1D5E7): Attention, MOSEI path label
- **Red** (#F8CECC): MOSI path label, Notes
- **Gray** (#F5F5F5): Encoders

---

## Arrows:

- **Solid lines**: Regular flow
- **Dashed blue lines**: Adapter connections
- **Thick green line**: Final output connection

This manual approach will definitely work and you'll have full control!




