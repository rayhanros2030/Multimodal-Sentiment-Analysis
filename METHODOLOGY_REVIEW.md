# Methodology Review: Dynamic_Emotion_Project.pdf

## Overall Assessment

The methodology section has **several discrepancies** with the actual implementation. The paper describes a more complex architecture (BiLSTM, temporal alignment to 30 frames, multi-task learning) than what we've implemented (single vector per video, cross-modal attention, regression only).

---

## Key Issues Found

### 1. **Architecture Mismatch**

**Paper Says:**
- Temporal alignment to **30 frames** for all modalities
- **BiLSTM decoder** with 256 hidden units
- **Multi-task learning**: Regression + Classification heads
- Output dimension: **[30×256]** throughout

**Actual Implementation:**
- **Single vector per video** (not 30 frames)
- **No BiLSTM decoder** - uses cross-attention + fusion layers
- **Regression only** (no classification head)
- Visual: [1 video × 65 features] → [1 × 96]
- Audio: [1 × 74] → [1 × 96]
- Text: [1 × 300/768] → [1 × 96]
- Final: [1 × 288] (concatenated) → [1]

**Severity:** 🔴 **CRITICAL** - The entire architecture description doesn't match the code.

---

### 2. **Temporal Alignment Section (3.5) - Does NOT Match Implementation**

**Paper Says:**
- Visual: Set to 30 frames (doesn't change)
- Audio: Adaptive Average Pooling 1D to 30 frames
- Text: Linear interpolation to 30 frames
- All modalities end up at **[30 frames × feature_dim]**

**Actual Implementation:**
- Visual: Up to 100 frames → **temporally averaged** → [1 × 65]
- Audio: Single 3-second window → **temporally averaged** → [1 × 74]
- Text: Full transcript → **mean pooling** → [1 × 768/300]
- **NO temporal alignment to 30 frames**
- **NO adaptive pooling or interpolation**

**Severity:** 🔴 **CRITICAL** - Section 3.5 describes a completely different approach.

---

### 3. **Audio Pipeline (Section 3.2) - Partially Matches**

**Paper Says:**
- 3-second non-overlapping windows
- Frame-level features (93ms frame, 23ms hop)
- 28 frame-level features (but says 29 in summary)
- Temporal averaging → 28 features per window
- **Padded to 74** for compatibility

**Actual Implementation:**
- ✅ 3-second duration (librosa.load with duration=3.0)
- ✅ Frame-level features using Librosa defaults
- ✅ 13 MFCC + 12 chroma + 4 other = **29 features** (matches)
- ✅ Padded to 74 dimensions
- ✅ Temporal averaging (mean)

**Issue:**
- Paper says "28 frame-level features" but code extracts **29** (13+12+4)
- Paper mentions "downsampled to 16 kHz" but code uses **22.05 kHz** (`sr=22050`)

**Severity:** 🟡 **MODERATE** - Minor inconsistencies, but mostly correct.

---

### 4. **Visual Pipeline (Section 3.3) - Mostly Matches**

**Paper Says:**
- Up to 100 frames (or all if < 100)
- 65-dimensional features (12 explicit + 53 derived)
- Face-width normalization
- Temporal averaging → [1 × 65]
- Encoder: 65 → 192 → 96

**Actual Implementation:**
- ✅ Up to 100 frames
- ✅ 65-dimensional features
- ✅ Face-width normalization
- ✅ Temporal averaging → [1 × 65]
- ✅ Encoder: 65 → 192 → 96 (with BatchNorm, ReLU, Dropout 0.7)

**Issue:**
- Paper says "broadcast or replicated as needed during cross-modal fusion" - but in implementation, it's just a single vector, no broadcasting

**Severity:** 🟢 **MINOR** - Very close match.

---

### 5. **Text Pipeline (Section 3.4) - Mostly Matches**

**Paper Says:**
- BERT-base-uncased (12 layers, 768-dim)
- Max 512 tokens, truncation/padding
- Frozen encoder (requires_grad=False)
- **Masked mean pooling** with attention mask
- Adapter: 768 → 300 (for MOSEI compatibility)
- Encoder: 300 → 192 → 96

**Actual Implementation:**
- ✅ BERT-base-uncased
- ✅ Max 512 tokens
- ✅ Frozen encoder
- ⚠️ **Mean pooling** (but attention mask not explicitly used in code)
- ✅ Adapter: 768 → 300
- ✅ Encoder: 300 → 192 → 96

**Issue:**
- Code uses simple mean pooling: `outputs.last_hidden_state.mean(dim=1)`
- Paper says "masked mean pooling" with attention mask exclusion
- This is a discrepancy we identified earlier

**Severity:** 🟡 **MODERATE** - Should use masked pooling as described.

---

### 6. **Cross-Modal Fusion (Section 3.7) - Does NOT Match**

**Paper Says:**
- Bidirectional cross-attention between all pairs (V-A, V-T, A-T)
- 8 attention heads per pair
- Input dimension: [30×256] for each modality
- Residual connections
- Self-attention refinement
- Output: [30×768] → [30×256] after MLP

**Actual Implementation:**
- ✅ Cross-attention (MultiheadAttention)
- ✅ 4 attention heads (not 8)
- ❌ Input: [batch × 96] (not [30×256])
- ✅ Residual-like structure (stacked features)
- ❌ No explicit self-attention refinement
- ❌ Output: [batch × 288] → [batch × 1] (not [30×256])

**Severity:** 🔴 **CRITICAL** - Completely different architecture.

---

### 7. **Temporal Decoder (Section 3.8) - Does NOT Exist**

**Paper Says:**
- 2-layer BiLSTM (256 hidden units)
- Self-attention pooling
- Regression head: 512 → 256 → 1
- Classification head: 512 → 256 → 3
- Multi-task learning with weighted loss

**Actual Implementation:**
- ❌ **NO BiLSTM decoder**
- ❌ **NO self-attention pooling**
- ❌ **NO classification head**
- ✅ Regression: 288 → 192 → 96 → 1 (different architecture)
- ❌ **NO multi-task learning**

**Severity:** 🔴 **CRITICAL** - Entire section doesn't exist in code.

---

### 8. **Loss Function (Section 3.9) - Partially Matches**

**Paper Says:**
- Multi-task: α·L_MSE + β·L_CE (classification)
- Grid search: α=10.0, β=1.0

**Actual Implementation:**
- ✅ Combined loss: α·(MSE+MAE)/2 + β·L_correlation
- ✅ α=0.3, β=0.7 (not α=10, β=1)
- ❌ **NO classification loss**
- ❌ **NO cross-entropy**

**Severity:** 🔴 **CRITICAL** - Different loss formulation entirely.

---

### 9. **Transfer Learning (Section 3.6) - Matches**

**Paper Says:**
- Feature adapters: FaceMesh→OpenFace2, Librosa→COVAREP, BERT→GloVe
- Train on MOSEI, adapt to MOSI
- MSE loss for adapters

**Actual Implementation:**
- ✅ Visual adapter: 65 → 713
- ✅ Audio adapter: 74 → 74
- ✅ Text adapter: 768 → 300
- ✅ MSE loss for adapters
- ✅ Train on MOSEI, test on MOSI

**Severity:** 🟢 **CORRECT** - Matches implementation.

---

## Summary Table

| Section | Paper Description | Implementation | Match? |
|---------|------------------|----------------|--------|
| **3.1 Dataset** | Transfer learning, feature adapters | ✅ Matches | ✅ |
| **3.2 Audio** | 28 features, 16kHz, temporal avg | 29 features, 22.05kHz | 🟡 Mostly |
| **3.3 Visual | 65-dim, temporal avg, encoder | ✅ Matches | ✅ |
| **3.4 Text** | BERT, masked pooling, adapter | BERT, mean pooling, adapter | 🟡 Mostly |
| **3.5 Temporal Alignment** | 30 frames, adaptive pooling | ❌ Single vector | 🔴 No |
| **3.6 Transfer Learning** | Feature adapters | ✅ Matches | ✅ |
| **3.7 Cross-Modal Fusion** | 30×256, 8 heads, BiLSTM | 96-dim, 4 heads, no BiLSTM | 🔴 No |
| **3.8 Temporal Decoder** | BiLSTM, attention pooling, dual-head | ❌ Doesn't exist | 🔴 No |
| **3.9 Loss** | Multi-task: MSE + CE | Correlation loss only | 🔴 No |

---

## Recommendations

### Option 1: Update Paper to Match Implementation (Recommended)

**Update Sections:**
1. **3.5 Temporal Alignment**: Remove entire section or replace with "Temporal averaging is applied per modality to obtain single vector representations."
2. **3.7 Cross-Modal Fusion**: Rewrite to match actual architecture:
   - Input: [batch × 96] for each modality
   - MultiheadAttention with 4 heads
   - Stack features → cross-attention → concatenate → fusion MLP
3. **3.8 Temporal Decoder**: Remove or replace with:
   - "The concatenated features [batch × 288] are passed through a 3-layer MLP (288 → 192 → 96 → 1) with batch normalization, ReLU activation, and dropout to produce final sentiment predictions."
4. **3.9 Loss**: Update to:
   - "We use a combined loss function: L = α·(MSE + MAE)/2 + β·L_correlation, where α=0.3 and β=0.7, emphasizing correlation optimization."
5. **3.2 Audio**: Fix feature count (28→29) and sampling rate (16kHz→22.05kHz)
6. **3.4 Text**: Add note that masked pooling should be used (currently uses simple mean)

### Option 2: Update Implementation to Match Paper (Not Recommended)

This would require significant code changes:
- Add temporal alignment to 30 frames
- Implement BiLSTM decoder
- Add classification head
- Implement multi-task learning
- Update loss function

This is **not recommended** because:
- Current implementation works
- Paper deadline may be soon
- Changes would require retraining

---

## What's Correct

✅ **Dataset description** (Section 3.1) - Accurate
✅ **Transfer learning approach** (Section 3.6) - Matches
✅ **Visual feature extraction** (Section 3.3) - Mostly accurate
✅ **Feature adapter mechanism** - Correct

---

## What Needs Fixing

🔴 **Critical Issues:**
1. Remove or rewrite Section 3.5 (Temporal Alignment) - doesn't match
2. Rewrite Section 3.7 (Cross-Modal Fusion) - wrong architecture
3. Remove or rewrite Section 3.8 (Temporal Decoder) - doesn't exist
4. Update Section 3.9 (Loss) - wrong formulation

🟡 **Moderate Issues:**
1. Section 3.2: Fix feature count (28→29) and sampling rate
2. Section 3.4: Note that masked pooling should be implemented

---

## Conclusion

The methodology section describes a **different architecture** than what's implemented. The paper describes a complex temporal model with BiLSTM and multi-task learning, while the actual code uses a simpler single-vector-per-video approach with cross-attention.

**Recommendation:** Update the paper to match the implementation, focusing on the strengths of the actual system (feature adaptation, cross-modal attention, correlation optimization) rather than describing a non-existent BiLSTM decoder.




