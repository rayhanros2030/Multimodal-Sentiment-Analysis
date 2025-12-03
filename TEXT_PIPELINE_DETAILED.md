# Text Pipeline Technical Specifications (BERT)

## Current Implementation Details

### 1. BERT Model Specifications
- **Model**: BERT-base-uncased (from Hugging Face Transformers)
- **Vocabulary Size**: 30,522 tokens
- **Hidden Size**: 768 dimensions
- **Number of Layers**: 12 transformer layers
- **Number of Attention Heads**: 12
- **Max Sequence Length**: 512 tokens (standard BERT limit)
- **Model Size**: ~110M parameters
- **Pretrained**: Yes (on BookCorpus + English Wikipedia)

### 2. Embedding Dimensions
- **Input**: Variable-length text transcript (raw string)
- **After Tokenization**: Sequence of token IDs (max 512 tokens)
- **BERT Output (Hidden States)**: [batch_size, sequence_length, 768]
  - `last_hidden_state`: 768-dim vector per token
- **After Pooling**: [batch_size, 768] → Mean pooling over sequence dimension
- **Final Output**: 768-dimensional vector per transcript

### 3. Input/Output Dimensions at Each Stage

| Stage | Dimensions | Description |
|-------|------------|-------------|
| Raw Text | Variable string | Transcript file content |
| Tokenization | [1, ≤512] | Token IDs (with padding to 512) |
| BERT Encoding | [1, 512, 768] | Hidden states for each token |
| Mean Pooling | [1, 768] | Averaged over sequence dimension |
| Text Adapter (if used) | [1, 768] → [1, 300] | Maps BERT→GloVe for MOSEI compatibility |
| Text Encoder (Layer 3) | [1, 300] or [1, 768] → [1, 96] | Encodes to embed_dim for fusion |

### 4. Sequence Length Limits
- **Max Input Tokens**: 512 (BERT's maximum)
- **Truncation**: Yes (`truncation=True`) - Longer sequences are truncated
- **Padding**: Yes (`padding=True`) - Shorter sequences padded with [PAD] token (ID=0)
- **Truncation Strategy**: From the right (end of sequence)
- **Padding Side**: Right side (standard BERT)

### 5. Special Tokens
- **[CLS]**: Added automatically at the start (token ID=101)
  - **Role**: Originally for classification tasks, but NOT used in our pipeline
  - **In Our Pipeline**: Included in mean pooling, not extracted separately
- **[SEP]**: Added automatically between sentences (token ID=102)
  - **Role**: Separates different segments/sentences
  - **In Our Pipeline**: Included in mean pooling
- **[PAD]**: Added for sequences shorter than 512 tokens (token ID=0)
  - **Role**: Padding tokens are masked in attention, but mean pooling includes them
  - **Recommendation**: Use attention mask to exclude [PAD] tokens from mean pooling

### 6. Attention Mask
- **Automatic**: Created by tokenizer when `padding=True`
- **Format**: Binary mask [1, 512] where 1=real token, 0=padding
- **Current Usage**: NOT explicitly used in mean pooling (should be fixed)
- **Recommended Fix**: 
  ```python
  attention_mask = inputs['attention_mask']
  embeddings = outputs.last_hidden_state
  masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
  mean_pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
  ```

### 7. Fine-tuning vs Frozen Encoder
- **Current Status**: **Frozen Encoder** (NOT fine-tuned)
  - `model.eval()` sets model to evaluation mode
  - `param.requires_grad = False` prevents gradient updates
- **Rationale**: 
  - Transfer learning approach: Use pretrained BERT embeddings
  - Feature extraction, not task-specific fine-tuning
  - Faster training, less overfitting
- **Alternative**: Fine-tuning could improve performance but requires more data and computation

### 8. Pooling Strategy
- **Current Method**: **Mean Pooling** (average over sequence dimension)
  ```python
  embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch, 768]
  ```
- **Alternatives**:
  - **CLS Token**: `outputs.last_hidden_state[:, 0, :]` - Uses first token embedding
  - **Max Pooling**: `outputs.last_hidden_state.max(dim=1)[0]` - Maximum value per dimension
  - **Attention-weighted**: Weighted average using learned attention
- **Why Mean Pooling**: 
  - Captures overall semantic content
  - Simple and effective for sentiment
  - All tokens contribute equally (good for short transcripts)

### 9. Word-Level Alignment
**Current Implementation**: **No explicit word-level alignment**

- **What Happens Now**: Entire transcript processed as single sequence
- **No Temporal Alignment**: BERT processes full transcript, not aligned to specific video frames
- **No Forced Alignment**: Not using tools like Gentle, Montreal Forced Aligner, or SPPAS
- **No VAD Timestamps**: Not using Voice Activity Detection timestamps
- **No Word Timestamps**: Transcripts don't contain timestamp annotations

**If You Want Word-Level Alignment** (for temporal alignment to 30 frames):

1. **Option A: Force Alignment Tools**
   - Use Gentle (https://github.com/lowerquality/gentle) or Montreal Forced Aligner
   - Align transcript words to audio timestamps
   - Map words to video frames (assuming 30 fps → 30 frames per second)
   - Extract BERT embeddings per word, then aggregate per frame

2. **Option B: Transcript Timestamps** (if available)
   - If CMU-MOSI transcripts have timestamps: `[00:00:00] word1 [00:00:01] word2`
   - Parse timestamps, map to frame numbers
   - Extract BERT embeddings per word segment
   - Aggregate embeddings per frame (average words in same frame)

3. **Option C: Sliding Window**
   - Divide transcript into segments (e.g., one segment per second = 30 frames)
   - Extract BERT embeddings per segment
   - Map to corresponding 30 video frames

4. **Option D: No Temporal Alignment** (Current)
   - Process entire transcript → single 768-dim vector
   - Use same vector for all 30 frames (broadcast)
   - OR: Expand to 30 copies of the same vector
   - Simplest, but loses temporal information

### 10. Connection to Architecture

**Current Pipeline (CMU-MOSEI Training)**:
```
Text Input (GloVe 300-dim) → Text Encoder → [192-dim] → [96-dim embed] → Cross-Attention → Fusion
```

**Transfer Learning Pipeline (CMU-MOSI Testing)**:
```
Raw Transcript → BERT Tokenizer → BERT-base → [768-dim] → Text Adapter → [300-dim] → Text Encoder → [96-dim embed] → Cross-Attention → Fusion
```

**Text Encoder Architecture (Layer 3)**:
- Input: 300-dim (GloVe) or 768-dim (BERT after adapter)
- Layer 1: Linear(300/768 → 192), BatchNorm1d, ReLU, Dropout(0.7)
- Layer 2: Linear(192 → 192), BatchNorm1d, ReLU, Dropout(0.7)
- Output: Linear(192 → 96), BatchNorm1d
- Output Dimension: **96-dim** (embed_dim)

**Connection to Temporal Alignment (30 frames)**:
- **Current**: No temporal alignment - single vector per video
- **If Needed**: Would require word-level alignment → frame-level aggregation → 30 frame vectors → sequence encoder (LSTM/Transformer) → 96-dim per frame

---

## Recommended Text Pipeline Paragraph for Paper

**Concise Version (Recommended)**:

```
For text feature extraction, we employ BERT-base-uncased to extract contextual word embeddings from transcript files. Each transcript is tokenized using the standard BERT tokenizer with a maximum sequence length of 512 tokens, with shorter sequences padded and longer sequences truncated. We use the pretrained BERT model as a frozen encoder (no fine-tuning) to extract 768-dimensional embeddings from the last hidden state. We apply mean pooling over the sequence dimension to obtain a single 768-dimensional vector per transcript, capturing the overall semantic content while remaining computationally efficient. For compatibility with models trained on CMU-MOSEI's GloVe features (300-dim), we employ a learned feature adapter network that maps BERT embeddings (768-dim) to the GloVe feature space (300-dim). The adapted 300-dimensional text features are then passed through the text encoder (300 → 192 → 96 dimensions with batch normalization, ReLU activation, and dropout) to obtain encoded text representations (embed_dim = 96) used in cross-modal attention and fusion. This approach allows us to leverage BERT's rich contextual representations while maintaining compatibility with existing models trained on static word embeddings.
```

**Detailed Version (If Space Permits)**:

```
For text feature extraction, we process transcript files using BERT-base-uncased, a bidirectional transformer model pretrained on large-scale English corpora. The BERT model consists of 12 transformer layers with 768-dimensional hidden states and 12 attention heads, providing rich contextual word representations. Transcripts are tokenized using the standard BERT tokenizer with special tokens [CLS] and [SEP] added automatically. We set a maximum sequence length of 512 tokens, with shorter sequences padded using [PAD] tokens (ID=0) and longer sequences truncated from the right. The tokenizer produces token IDs and an attention mask, where 1 indicates real tokens and 0 indicates padding tokens. 

We use BERT as a frozen encoder (model.eval() with requires_grad=False), leveraging pretrained representations without task-specific fine-tuning to maintain generalization and reduce overfitting. The model processes tokenized input to produce hidden states of shape [batch_size, sequence_length, 768], where each token is represented by a 768-dimensional vector. To obtain a single fixed-size representation per transcript, we apply mean pooling over the sequence dimension, computing the average of all token embeddings weighted by the attention mask to exclude padding tokens. This yields a 768-dimensional vector capturing the overall semantic content of the transcript.

For transfer learning to CMU-MOSI, where transcripts require real-time processing, BERT embeddings (768-dim) must be adapted to match the feature space of CMU-MOSEI's GloVe embeddings (300-dim). We train a feature adapter network with the architecture: Linear(768 → 384) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(384 → 384) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(384 → 300), which learns to map BERT embeddings to GloVe-compatible representations. The adapted 300-dimensional features are then fed into the text encoder (Layer 3), which consists of two linear transformations with batch normalization: 300 → 192 → 96 dimensions, each followed by ReLU activation and dropout (0.7). The final 96-dimensional encoded text representation (embed_dim = 96) is used in cross-modal attention mechanisms and fusion layers alongside visual and audio embeddings of the same dimension.

The text encoder output is concatenated with encoded visual and audio features, producing a 288-dimensional (96×3) fused representation that feeds into three-layer fusion network (288 → 192 → 96 → 1) for final sentiment prediction. This architecture allows the model to leverage BERT's contextual understanding while maintaining compatibility with models trained on static word embeddings through learned feature adaptation.
```

---

## Answers to Specific Questions

### Q1: Which BERT model?
**A:** BERT-base-uncased (standard, case-insensitive, 12 layers, 768-dim hidden)

### Q2: Embedding dimensions?
**A:** 
- BERT output: 768 dimensions per token
- After pooling: 768 dimensions per transcript
- After adapter (for MOSEI compatibility): 300 dimensions
- After text encoder: 96 dimensions (embed_dim)

### Q3: Input/output dimensions at each stage?
**A:** See table in Section 3 above.

### Q4: Sequence length limits?
**A:** Maximum 512 tokens (BERT standard). Truncation and padding are automatic.

### Q5: How word-level alignment is performed?
**A:** **Currently NOT performed**. The implementation processes the entire transcript as a single sequence. See Section 9 for alignment options if needed.

### Q6: Special tokens (CLS, SEP) role?
**A:** 
- Added automatically by tokenizer
- [CLS] at start, [SEP] between sentences
- Included in mean pooling (not extracted separately)
- Could use [CLS] token alone as alternative pooling strategy

### Q7: Attention mask?
**A:** 
- Created automatically by tokenizer
- **Issue**: Currently NOT used in mean pooling (should exclude padding tokens)
- **Fix**: Use masked mean pooling (see Section 6)

### Q8: Fine-tuning vs frozen?
**A:** **Frozen encoder** (no fine-tuning). All BERT parameters have `requires_grad=False`.

### Q9: Pooling strategy?
**A:** **Mean pooling** over sequence dimension. Alternatives: CLS token, max pooling, attention-weighted.

### Q10: Missing connection to architecture?
**A:** 
- BERT (768) → Adapter → (300) → Text Encoder → (96) → Cross-Attention → Fusion → (1)
- No temporal alignment to 30 frames (single vector per video)
- See Section 10 for architecture flow diagram

---

## Issues to Fix in Implementation

1. **Attention Mask Not Used in Pooling**:
   ```python
   # Current (WRONG):
   embeddings = outputs.last_hidden_state.mean(dim=1)
   
   # Should be (CORRECT):
   attention_mask = inputs['attention_mask']
   embeddings = outputs.last_hidden_state
   masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
   mean_pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
   ```

2. **Word-Level Alignment**: Currently missing if you want temporal alignment to 30 frames.

3. **CLS Token Alternative**: Could try using CLS token instead of mean pooling for comparison.




