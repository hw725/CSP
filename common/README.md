# Common — Shared Modules

Shared utilities, model loaders, embedders, and tokenizers used by both P2S and S2P pipelines.

## Directory Structure

```
common/
├── embedders/                    # Embedding backends
│   ├── bge.py                    #   BGE-M3 (primary, GPU-accelerated)
│   ├── openai_embedder.py        #   OpenAI text-embedding-ada-002
│   └── gemini_embedder.py        #   Google Gemini text-embedding-004
│
├── tokenizers/                   # Tokenization backends
│   ├── siku_tokenizer.py         #   SikuBERT for Classical Chinese
│   ├── kiwi_tokenizer.py         #   Kiwipiepy for Korean morphology
│   ├── hybrid_korean_tokenizer.py #  Hybrid Korean tokenizer
│   └── roberta_hanja_tokenizer.py #  Hanja-specific RoBERTa tokenizer
│
├── config.py                     # Configuration management (csp_config.json)
├── integrity_verifier.py         # Global text integrity verification
├── text_normalizer.py            # Unicode normalization, whitespace handling
├── progress_manager.py           # Unified progress bar management
├── disk_cache.py                 # Disk-based embedding cache
├── numba_ops.py                  # Numba JIT-accelerated operations
├── korean_particle_matcher.py    # Korean particle/suffix detection
│
├── boundary_model_loader.py      # P2S boundary detection model loader
├── s2p_phrase_alignment_loader.py # S2P v2.1 phrase alignment model loader
├── s2p_crossattn_boundary_loader.py # S2P cross-attention model loader
├── semantic_boundary_loader.py   # Semantic boundary model loader
└── alignment_model_loader.py     # General alignment model loader
```

## Key Components

### Embedders

The primary embedder is **BGE-M3** (`embedders/bge.py`), which provides:
- Dense vectors (1024d) for semantic similarity
- Sparse vectors for keyword matching
- ColBERT vectors for token-level matching
- GPU-accelerated batch encoding with automatic caching

### Tokenizers

- **SikuBERT** (`siku_tokenizer.py`): Tokenizes Classical Chinese text using the SikuBERT model, trained on the Siku Quanshu corpus
- **Kiwipiepy** (`kiwi_tokenizer.py`): Korean morphological analyzer for sentence splitting and eojeol (어절) boundary detection

### Model Loaders

Each loader provides a standardized interface for loading trained PyTorch models:
- `BoundaryModelLoader`: Loads `boundary_multitask.pt` for P2S boundary detection
- `PhraseAlignmentModel`: Loads `s2p_phrase_alignment.pt` for S2P v2.1

### Integrity Verification

`integrity_verifier.py` ensures that all text transformations preserve 100% of original characters (excluding whitespace). This is the core quality guarantee of the CSP system.
