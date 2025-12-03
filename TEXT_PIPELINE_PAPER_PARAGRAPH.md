# Text Pipeline Paragraph for Research Paper

## Recommended Paragraph (Concise & Technical)

```
For text feature extraction, we employ BERT-base-uncased to extract contextual word embeddings from transcript files. Transcripts are tokenized using the standard BERT tokenizer with a maximum sequence length of 512 tokens, with shorter sequences padded using [PAD] tokens and longer sequences truncated from the right. The tokenizer automatically adds special tokens [CLS] at the start and [SEP] between sentences, and generates an attention mask to distinguish real tokens from padding tokens. We use the pretrained BERT model as a frozen encoder (requires_grad=False) to extract 768-dimensional embeddings from the last hidden state, applying masked mean pooling over the sequence dimension (excluding padding tokens) to obtain a single 768-dimensional vector per transcript. This captures the overall semantic content while remaining computationally efficient.

For compatibility with models trained on CMU-MOSEI's GloVe features (300-dim), we employ a learned feature adapter network (768 → 384 → 300 dimensions with batch normalization, ReLU activation, and dropout) that maps BERT embeddings to the GloVe feature space. The adapted 300-dimensional text features are then passed through the text encoder, which consists of two linear transformations (300 → 192 → 96 dimensions) with batch normalization, ReLU activation, and dropout (0.7). The final 96-dimensional encoded text representation (embed_dim = 96) is concatenated with encoded visual and audio features of the same dimension for cross-modal attention and fusion, producing a 288-dimensional fused representation that feeds into a three-layer fusion network (288 → 192 → 96 → 1) for final sentiment prediction.
```

---

## Alternative: More Detailed Version (If You Have Space)

```
For text feature extraction, we process transcript files using BERT-base-uncased, a bidirectional transformer model pretrained on BookCorpus and English Wikipedia. BERT-base-uncased consists of 12 transformer layers with 768-dimensional hidden states and 12 attention heads, providing rich contextual word representations. Transcripts are tokenized using the standard BERT tokenizer with WordPiece subword tokenization, producing token IDs and an attention mask (binary mask where 1 indicates real tokens and 0 indicates padding tokens). We set a maximum sequence length of 512 tokens (BERT's standard limit), with shorter sequences padded using [PAD] tokens (ID=0) and longer sequences truncated from the right.

We use BERT as a frozen encoder (model.eval() with requires_grad=False for all parameters), leveraging pretrained representations without task-specific fine-tuning to maintain generalization and reduce overfitting. The model processes tokenized input to produce hidden states of shape [batch_size, sequence_length, 768], where each token is represented by a 768-dimensional vector. To obtain a single fixed-size representation per transcript, we apply masked mean pooling over the sequence dimension: we multiply hidden states by the attention mask (broadcast along the embedding dimension) to zero out padding tokens, then compute the sum divided by the number of real tokens. This yields a 768-dimensional vector capturing the overall semantic content of the transcript.

For transfer learning to CMU-MOSI, where transcripts require real-time processing, BERT embeddings (768-dim) must be adapted to match the feature space of CMU-MOSEI's GloVe embeddings (300-dim). We train a feature adapter network with the architecture: Linear(768 → 384) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(384 → 384) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(384 → 300), which learns to map BERT embeddings to GloVe-compatible representations through end-to-end training using mean squared error loss. The adapted 300-dimensional features are then fed into the text encoder (Layer 3), which consists of two linear transformations with batch normalization: 300 → 192 → 96 dimensions, each followed by ReLU activation and dropout (0.7). The final 96-dimensional encoded text representation (embed_dim = 96) is used in cross-modal attention mechanisms alongside visual and audio embeddings of the same dimension.

The three encoded modalities (visual, audio, text) are concatenated to form a 288-dimensional (96×3) fused representation. This fused representation is passed through a three-layer fusion network: Linear(288 → 192) → BatchNorm1d → ReLU → Dropout(0.7) → Linear(192 → 96) → BatchNorm1d → ReLU → Dropout(0.7) → Linear(96 → 1), producing the final sentiment prediction. This architecture allows the model to leverage BERT's contextual understanding while maintaining compatibility with models trained on static word embeddings through learned feature adaptation.
```

---

## Key Technical Specifications Summary

| Aspect | Specification |
|--------|--------------|
| **BERT Model** | BERT-base-uncased (12 layers, 768-dim hidden, 12 attention heads) |
| **Tokenization** | WordPiece subword tokenization, max_length=512 |
| **Special Tokens** | [CLS] (start), [SEP] (sentence separator), [PAD] (padding) |
| **Attention Mask** | Binary mask (1=real token, 0=padding), used in masked mean pooling |
| **Fine-tuning** | Frozen encoder (no fine-tuning, requires_grad=False) |
| **Pooling Strategy** | Masked mean pooling (average over sequence, excluding padding) |
| **BERT Output** | 768-dimensional vector per transcript |
| **Adapter Architecture** | 768 → 384 → 300 (with BatchNorm, ReLU, Dropout) |
| **Text Encoder** | 300 → 192 → 96 (with BatchNorm, ReLU, Dropout 0.7) |
| **Final Dimension** | 96-dim (embed_dim) for cross-modal attention |

---

## Important Note: Word-Level Alignment

**Current Implementation**: No explicit word-level temporal alignment to video frames. The entire transcript is processed as a single sequence, producing one 768-dim vector per video that is used for all frames (broadcast).

**If You Need Word-Level Alignment** (for temporal alignment to 30 frames):
- You would need forced alignment tools (Gentle, Montreal Forced Aligner) to map words to timestamps
- Then map timestamps to video frames (assuming 30 fps → 30 frames per second)
- Extract BERT embeddings per word segment, aggregate per frame
- This would require additional processing not currently implemented

**For Your Paper**: You can either:
1. **Acknowledge the limitation**: "We process transcripts as complete sequences without explicit word-level temporal alignment."
2. **Describe future work**: "Future improvements could include word-level forced alignment to enable frame-specific text embeddings."
3. **Keep it simple**: Focus on the feature extraction pipeline (BERT → adapter → encoder) without discussing temporal alignment, since it's not currently implemented.




