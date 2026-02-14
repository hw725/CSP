# P2S — Paragraph to Sentence Pipeline

Splits paragraph-level parallel text (Classical Chinese + Korean translation) into sentence-level aligned pairs.

**Performance**: F1 = 0.9384 (4,934 paragraphs, RunPod H200)

## Pipeline Overview

```
Input: Paragraph pair (source + target)
  │
  ├─ 1. Target Sentence Splitting
  │     Korean translation → sentence boundaries (via Stanza/SuPar)
  │
  ├─ 2. Source Boundary Detection
  │     Classical Chinese → candidate split points
  │     Strategies: SuPar parsing, Boundary Model, Whitespace DP, TopK
  │
  ├─ 3. BGE Refinement (3-pass)
  │     Fine-tune boundaries using BGE-M3 similarity + length-ratio bonus
  │
  └─ 4. Integrity Verification
        Ensure 100% character preservation
  │
  ▼
Output: Sentence pairs with paragraph/sentence IDs
```

## Files

| File | Description |
|------|-------------|
| `processor.py` | Main processing logic. Orchestrates the full pipeline with stage tracing for observability. Contains `process_single_paragraph()` and multi-strategy candidate generation. |
| `aligner.py` | Alignment algorithms. Implements DP-based alignment, BGE refinement with token-boundary candidates, length-ratio bonus, and 현토 ending bonus. |
| `sentence_splitter.py` | Korean sentence boundary detection. Uses SuPar-Kanbun for Classical Chinese and Stanza for Korean. Includes OpenAI fallback wrapper. |
| `main.py` | CLI entry point. Parses arguments, loads models, runs the pipeline on input files. |

## Usage

```bash
# Basic usage
python p2s/main.py <input.csv> <output.xlsx>

# Via universal entry point
python main.py <input_file> <output_file>
```

## Key Algorithms

- **Target-anchored splitting**: Uses Korean sentence boundaries as anchors, then adjusts Chinese source boundaries to match
- **Multi-strategy candidates**: SuPar dependency parsing, trained boundary model, whitespace-based DP, model TopK predictions
- **BGE Refinement**: 3-pass iterative refinement using BGE-M3 cosine similarity with length-ratio bonus to detect 3-6 character boundary shifts
- **Eojeol protection**: Splits only at whitespace boundaries (어절 보호) — never breaks mid-word
