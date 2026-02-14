# S2P — Sentence to Phrase Pipeline

Splits sentence-level parallel text into phrase-level 1:1 aligned pairs.

**Performance**: F1 = 0.8555 (v2.1 Phrase Alignment, 446 sentences)

## Pipeline Overview

```
Input: Sentence pair (source + target)
  │
  ├─ 1. Phrase Alignment Model (v2.1)
  │     BiLSTM encoder + Guided Attention → boundary probabilities
  │     Source: BGE-M3 1024d embeddings
  │     Target: BGE-M3 1024d embeddings
  │
  ├─ 2. Viterbi Decoding
  │     Optimal boundary selection with transition constraints
  │
  ├─ 3. Punctuation Guard
  │     Adjusts boundaries to respect punctuation and bracket pairs
  │
  └─ 4. Integrity Verification
        Ensure 100% character preservation
  │
  ▼
Output: Phrase pairs with sentence/phrase IDs
```

## Files

| File | Description |
|------|-------------|
| `s2p_aligner.py` | Core alignment logic. Runs the Phrase Alignment model, performs Viterbi decoding, and applies BGE-M3 scoring for quality filtering. |
| `punctuation.py` | Punctuation and bracket handling. `IntegrityGuard` class ensures no characters are lost during phrase splitting. |
| `io_manager.py` | I/O operations. `SafeFileProcessor` handles batch processing with integrity guarantees and file format conversions. |
| `main.py` | CLI entry point. Parses arguments, preloads models (BGE + Phrase Alignment), and runs the pipeline. |

## Usage

```bash
# Basic usage
python s2p/main.py <input.csv> <output.xlsx> [--batch-size 32]

# Via universal entry point
python main.py <input_file> <output_file>
```

## Model Architecture (v2.1)

- **Source Encoder**: BiLSTM (input: BGE-M3 1024d, hidden: 512)
- **Target Encoder**: BiLSTM (input: BGE-M3 1024d, hidden: 512)
- **Guided Attention**: Cross-attention between source and target representations
- **Boundary Head**: Linear → sigmoid for per-character boundary probability
- **Decoding**: Viterbi algorithm with transition matrix for optimal segmentation
- **Parameters**: 6.7M
- **Training**: 100 epochs, val boundary F1 = 0.7755

## Version History

| Version | F1 | Key Changes |
|---------|-----|-------------|
| Baseline (DP only) | 0.1563 | Dynamic programming only |
| v2 | 0.6900 | Phrase Alignment model (hidden=256) |
| **v2.1** | **0.8555** | + Source BiLSTM, hidden 512, Guided Attention, 100 epochs |
