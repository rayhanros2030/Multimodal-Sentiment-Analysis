# Architecture Diagram - Manual Build Guide for Draw.io

Since the .drawio file isn't opening, build it manually using this guide.

---

## Quick Reference: Complete Architecture Text

```
TITLE: Multimodal Sentiment Analysis Architecture (Actual Implementation)

NOTE: Features are aggregated before input. No temporal sequences.

─────────────────────────────────────────────────────────────
LAYER 1: INPUT MODALITIES
─────────────────────────────────────────────────────────────
[Video]          [Audio]          [Text]
RGB Frames       Waveform         Transcript
─────────────────────────────────────────────────────────────

LAYER 2: FEATURE EXTRACTION & AGGREGATION
─────────────────────────────────────────────────────────────

CMU-MOSEI (Training): Pre-extracted Features
[OpenFace2]      [COVAREP]        [GloVe]
713-dim          74-dim           300-dim

CMU-MOSI (Testing): Real-time Extraction + Adapters
[FaceMesh] → [65-dim] → [Adapter 65→713] → [713-dim]
[Librosa] → [74-dim] → (No adapter) → [74-dim]
[BERT] → [768-dim] → [Adapter 768→300] → [300-dim]
─────────────────────────────────────────────────────────────

LAYER 3: MODALITY ENCODERS (Linear Layers)
─────────────────────────────────────────────────────────────
[Visual Encoder]     [Audio Encoder]     [Text Encoder]
713 → 192 → 96      74 → 192 → 96      300 → 192 → 96
BatchNorm+ReLU      BatchNorm+ReLU     BatchNorm+ReLU
Dropout(0.7)        Dropout(0.7)        Dropout(0.7)
─────────────────────────────────────────────────────────────

LAYER 4: CROSS-MODAL FUSION
─────────────────────────────────────────────────────────────
[MultiheadAttention] → [Concatenation] → [Fusion MLP]
(4 heads, 0.8 dropout)  [96,96,96]→[288]  288→192→96→1
─────────────────────────────────────────────────────────────

OUTPUT
─────────────────────────────────────────────────────────────
[Sentiment Score]
Single Regression: [-3 to +3]
─────────────────────────────────────────────────────────────

LOSS FUNCTION
─────────────────────────────────────────────────────────────
L = α × (MSE + MAE)/2 + β × (1 - Pearson Corr)²
α = 0.3, β = 0.7
─────────────────────────────────────────────────────────────
```

---

## Detailed Build Instructions

### Setup:
1. Open https://app.diagrams.net/
2. Create new blank diagram
3. Set canvas: 1600px wide × 2400px tall (File → Page Setup)

---

### Step 1: Title (Top Center)
- **Text**: "Multimodal Sentiment Analysis Architecture (Actual Implementation)"
- **Style**: Large font (22pt), Bold, Centered
- **Position**: Top center

---

### Step 2: Warning Note (Below Title)
- **Text**: "NOTE: Features are aggregated (temporally averaged) before model input. No temporal sequences - single feature vectors only."
- **Background**: Light Red (#F8CECC)
- **Style**: Bold, Italic
- **Position**: Below title

---

### Step 3: Layer 1 Header
- **Text**: "LAYER 1: INPUT MODALITIES"
- **Background**: Light Blue (#E1F5FF)
- **Style**: Bold, 16pt
- **Position**: Below note

---

### Step 4: Input Boxes (3 horizontal)
**Video Box**:
- Text: "Video (RGB Frames)"
- Background: Yellow (#FFF2CC)
- Position: Left

**Audio Box**:
- Text: "Audio (Waveform)"
- Background: Yellow (#FFF2CC)
- Position: Center

**Text Box**:
- Text: "Text (Transcript)"
- Background: Yellow (#FFF2CC)
- Position: Right

---

### Step 5: Layer 2 Header
- **Text**: "LAYER 2: FEATURE EXTRACTION & AGGREGATION"
- **Background**: Light Blue (#E1F5FF)
- **Style**: Bold, 16pt

---

### Step 6: CMU-MOSEI Section
**Label Box**:
- Text: "CMU-MOSEI: Pre-extracted Features (Training)"
- Background: Purple (#E1D5E7)
- Style: Bold, Italic

**Three Feature Boxes** (horizontal):
1. "OpenFace2\nVisual Features\n[713-dim]\n(Pre-extracted)"
   - Background: Green (#D5E8D4)

2. "COVAREP\nAudio Features\n[74-dim]\n(Pre-extracted)"
   - Background: Green (#D5E8D4)

3. "GloVe\nText Features\n[300-dim]\n(Pre-extracted)"
   - Background: Green (#D5E8D4)

**Arrows**: From Input boxes → Feature boxes (green arrows)

---

### Step 7: CMU-MOSI Section
**Label Box**:
- Text: "CMU-MOSI: Real-time Feature Extraction (Testing)"
- Background: Red (#F8CECC)
- Style: Bold, Italic

**Visual Path** (horizontal boxes with arrows):
1. "FaceMesh\n468 Landmarks"
   - Background: Orange (#FFE6CC)

2. "Emotion Features\n[65-dim]"
   - Background: Yellow (#FFF2CC)

3. "Feature Adapter\n[65 → 713]"
   - Background: Blue (#DAE8FC)
   - Style: **BOLD**

**Audio Path**:
1. "Librosa\nMFCC + Chroma + Spectral"
   - Background: Orange (#FFE6CC)

2. "Audio Features\n[74-dim]"
   - Background: Yellow (#FFF2CC)

3. "No Adapter\n(Already 74-dim)"
   - Background: Light Blue (#E1F5FF)
   - Style: Italic

**Text Path**:
1. "BERT Tokenizer\n+ BERT Model"
   - Background: Orange (#FFE6CC)

2. "BERT Embeddings\n[768-dim]"
   - Background: Yellow (#FFF2CC)

3. "Feature Adapter\n[768 → 300]"
   - Background: Blue (#DAE8FC)
   - Style: **BOLD**

**Arrows**: 
- Orange arrows: Input → Extraction → Features
- Blue arrows (thick): Features → Adapters

---

### Step 8: Layer 3 Header
- **Text**: "LAYER 3: MODALITY ENCODERS (Linear Layers with BatchNorm, ReLU, Dropout)"
- **Background**: Light Blue (#E1F5FF)
- **Style**: Bold, 16pt

---

### Step 9: Encoder Boxes (3 horizontal)

**Visual Encoder**:
- Background: Gray (#F5F5F5)
- Text (multi-line):
  ```
  Visual Encoder
  Linear(713 → 192)
  BatchNorm + ReLU + Dropout(0.7)
  Linear(192 → 192) [optional]
  Linear(192 → 96)
  BatchNorm
  Output: [96-dim]
  ```

**Audio Encoder**:
- Background: Gray (#F5F5F5)
- Text:
  ```
  Audio Encoder
  Linear(74 → 192)
  BatchNorm + ReLU + Dropout(0.7)
  Linear(192 → 192) [optional]
  Linear(192 → 96)
  BatchNorm
  Output: [96-dim]
  ```

**Text Encoder**:
- Background: Gray (#F5F5F5)
- Text:
  ```
  Text Encoder
  Linear(300 → 192)
  BatchNorm + ReLU + Dropout(0.7)
  Linear(192 → 192) [optional]
  Linear(192 → 96)
  BatchNorm
  Output: [96-dim]
  ```

**Arrows**:
- Solid arrows: MOSEI features → Encoders
- Dashed blue arrows: MOSI adapters → Encoders

---

### Step 10: Layer 4 Header
- **Text**: "LAYER 4: CROSS-MODAL FUSION (Attention + Concatenation + MLP)"
- **Background**: Light Blue (#E1F5FF)
- **Style**: Bold, 16pt

---

### Step 11: Fusion Components (horizontal flow)

**MultiheadAttention Box**:
- Background: Purple (#E1D5E7)
- Text:
  ```
  MultiheadAttention
  (4 heads, dropout=0.8)
  Stack features: [batch × 3 × 96]
  Attention on stacked features
  ```

**Concatenation Box**:
- Background: Blue (#DAE8FC)
- Text:
  ```
  Concatenation
  [96, 96, 96] → [288]
  ```

**Fusion MLP Box**:
- Background: Yellow (#FFF2CC)
- Text:
  ```
  Fusion MLP
  Linear(288 → 192)
  BatchNorm + ReLU + Dropout(0.7)
  Linear(192 → 96)
  BatchNorm + ReLU + Dropout(0.7)
  Linear(96 → 1)
  ```

**Arrows**: Encoders → Attention → Concatenation → MLP (solid black arrows)

---

### Step 12: Output Box
- **Background**: Green (#D5E8D4)
- **Style**: Bold, Large font (13pt)
- **Text**:
  ```
  Sentiment Score
  Single Regression Output
  Range: [-3, +3]
  (Continuous sentiment intensity)
  ```
- **Arrow**: Thick green arrow from Fusion MLP → Output

---

### Step 13: Loss Function (Bottom)
- **Background**: Yellow (#FFF2CC)
- **Text**:
  ```
  Improved Correlation Loss
  L = α × (MSE + MAE)/2 + β × (1 - Pearson Corr)²
  α = 0.3 (accuracy weight), β = 0.7 (correlation weight)
  ```

---

### Step 14: Differences Note (Optional, Bottom)
- **Background**: Red (#F8CECC)
- **Text**:
  ```
  KEY DIFFERENCES FROM TEMPORAL ARCHITECTURE:
  • NO Temporal Structure: Features aggregated before input
  • Simple Linear Encoders: Not CNN/LSTM
  • Simpler Fusion: Attention + Concatenation + MLP
  • Single Regression Output: No temporal decoder, no classification head
  • Dimensions: 96-dim (not 256), 288-dim concatenated (not 768), 1-dim output
  ```

---

## Color Guide:

| Element | Color Code | Use For |
|---------|-----------|---------|
| #FFF2CC | Yellow | Inputs, Fusion MLP, Loss |
| #D5E8D4 | Green | MOSEI features, Output |
| #FFE6CC | Orange | MOSI real-time extraction |
| #DAE8FC | Blue | Feature adapters, Concatenation |
| #E1D5E7 | Purple | Attention, MOSEI label |
| #F8CECC | Red | MOSI label, Notes |
| #F5F5F5 | Gray | Encoders |
| #E1F5FF | Light Blue | Layer headers |

---

## Arrow Styles:

- **Solid black**: Regular flow
- **Solid green**: MOSEI path
- **Solid orange**: MOSI extraction path
- **Dashed blue**: Adapter connections
- **Thick green**: Final output

---

Build this in draw.io and you'll have an accurate diagram that matches your implementation!




