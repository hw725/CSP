# CSP (Corpus Split Parallel)

> **Automated Classical Chinese-Korean parallel text alignment pipeline**
>
> CSP splits and aligns Classical Chinese (漢文) texts with their Korean translations at paragraph, sentence, and phrase levels.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.6%2B-red)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey)](LICENSE)

## Overview

CSP is a two-stage pipeline that processes pre-modern Korean literary corpora:

1. **P2S (Paragraph to Sentence)**: Splits aligned paragraph pairs into sentence pairs
2. **S2P (Sentence to Phrase)**: Splits aligned sentence pairs into phrase pairs (1:1 alignment)

Both stages guarantee **100% text integrity** — no characters are lost or added during splitting.

### Architecture

```
Input XLSX (paragraph-level parallel text)
    │
    ▼
┌─────────────────────────────────────────────┐
│  P2S Pipeline                               │
│  ┌───────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Sentence  │→ │ Boundary │→ │   BGE    │ │
│  │ Splitter  │  │  Model   │  │Refinement│ │
│  │ (Korean)  │  │(Chinese) │  │ (3-pass) │ │
│  └───────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────┘
    │
    ▼
Output XLSX (sentence-level parallel text)
    │
    ▼
┌─────────────────────────────────────────────┐
│  S2P Pipeline                               │
│  ┌───────────┐  ┌──────────┐  ┌──────────┐ │
│  │  Phrase   │→ │  Viterbi │→ │Punctuation│ │
│  │ Alignment │  │ Decoding │  │  Guard    │ │
│  │  Model    │  │          │  │           │ │
│  └───────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────┘
    │
    ▼
Output XLSX (phrase-level parallel text)
```

### Performance (2026-02-10)

| Pipeline | F1 Score | Precision | Recall | Test Size |
|----------|----------|-----------|--------|-----------|
| **P2S** | **0.9384** | 1.0000 | 0.8840 | 4,934 paragraphs |
| **S2P** (v2.1) | **0.8555** | 1.0000 | 0.7475 | 446 sentences |

### Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Embeddings | BGE-M3 (FlagEmbedding) | Semantic similarity scoring |
| Boundary Detection | Cross-Attention BiLSTM | Chinese text segmentation |
| Chinese Parsing | SuPar-Kanbun | Dependency parsing for Classical Chinese |
| Korean Tokenizer | Kiwipiepy | Morphological analysis |
| Chinese Tokenizer | SikuBERT | Classical Chinese subword tokenization |
| Phrase Alignment | BiLSTM + Guided Attention | Source-target phrase boundary prediction |

---

## Quick Start

### Docker (Recommended)

```bash
docker-compose up -d
docker-compose exec csp bash

# Run P2S
python p2s/main.py <input.csv> <output.xlsx>

# Run S2P
python s2p/main.py <input.csv> <output.xlsx> [--batch-size 32]

# Run both pipelines on all 44 books
python batch_44books.py
```

### Universal Entry Point

```bash
# Auto-detects file format (XML/XLSX/TXT/CSV)
python main.py <input_file> <output_file>
```

---

## Project Structure

```
CSP/
├── p2s/                # Paragraph-to-Sentence pipeline
│   ├── processor.py    #   Main processing logic with stage tracing
│   ├── aligner.py      #   Alignment algorithms (DP, BGE refinement)
│   ├── sentence_splitter.py  # Korean sentence boundary detection
│   └── main.py         #   CLI entry point
│
├── s2p/                # Sentence-to-Phrase pipeline
│   ├── s2p_aligner.py  #   Core phrase alignment with Viterbi decoding
│   ├── punctuation.py  #   Punctuation-based segmentation + integrity
│   ├── io_manager.py   #   I/O operations and batch processing
│   └── main.py         #   CLI entry point
│
├── common/             # Shared modules
│   ├── embedders/      #   BGE-M3, OpenAI, Gemini embedding backends
│   ├── tokenizers/     #   SikuBERT (Chinese), Kiwipiepy (Korean)
│   ├── config.py       #   Configuration management
│   ├── integrity_verifier.py  # Text integrity verification
│   ├── boundary_model_loader.py     # P2S boundary model
│   └── s2p_phrase_alignment_loader.py  # S2P v2.1 model
│
├── accuracy/           # Evaluation scripts
│   ├── p2s_evaluator.py  # P2S F1, precision, recall, similarity
│   └── s2p_evaluator.py  # S2P F1, precision, recall, similarity
│
├── scripts/            # Training and analysis scripts
│   ├── train_p2s_boundary.py        # P2S boundary model training
│   ├── train_s2p_phrase_alignment.py # S2P v2.1 model training
│   └── ...             #   Tuning, analysis, diagnostics
│
├── models/             # Trained model weights (.pt files)
├── datasets/           # Gold-standard test data (splits/)
├── docs/               # Technical documentation
├── analytics/          # Monitoring and visualization tools
├── xlsx/               # 44 classical text books (local only)
├── xlsx_pipeline/      # XLSX processing utilities
├── hyeonto/            # Research analysis (현토 marker studies)
│
├── main.py             # Universal entry point
├── batch_44books.py    # Batch processor for all books
├── csp_config.json     # Main configuration file
├── Dockerfile          # GPU-enabled container (PyTorch 2.6 + CUDA 12.4)
└── docker-compose.yml  # Multi-service setup (CSP + MSSQL)
```

---

## Processing Examples

### P2S: Paragraph to Sentence

**Input** (paragraph-level):

| Source (Classical Chinese) | Target (Korean) |
|:--|:--|
| 公子開方事君 不歸視死父. 衛懿公好鶴 不恤死國. 齊桓公得子亂國 | 공자개방이 군주를 섬기며 죽은 아버지를 돌아보지 않았다. 위의공은 학을 좋아하여 나라가 죽는 것을 돌보지 않았다. 제환공은 자란국을 얻었다 |

**Output** (sentence-level):

| Para ID | Sent ID | Source | Target |
|:--|:--|:--|:--|
| 1 | 1 | 公子開方事君 不歸視死父 | 공자개방이 군주를 섬기며 죽은 아버지를 돌아보지 않았다 |
| 1 | 2 | 衛懿公好鶴 不恤死國 | 위의공은 학을 좋아하여 나라가 죽는 것을 돌보지 않았다 |
| 1 | 3 | 齊桓公得子亂國 | 제환공은 자란국을 얻었다 |

### S2P: Sentence to Phrase

**Input** (sentence-level):

| Source | Target |
|:--|:--|
| 公子開方事君 不歸視死父 | 공자개방이 군주를 섬기며 죽은 아버지를 돌아보지 않았다 |

**Output** (phrase-level):

| Sent ID | Phrase ID | Source Phrase | Target Phrase |
|:--|:--|:--|:--|
| 1 | 1 | 公子開方 | 공자개방이 |
| 1 | 2 | 事君 | 군주를 섬기며 |
| 1 | 3 | 不歸視死父 | 죽은 아버지를 돌아보지 않았다 |

---

## Documentation

| Document | Description |
|----------|-------------|
| [P2S Mechanism](docs/P2S_MECHANISM.md) | P2S algorithm: target-anchored splitting with BGE refinement |
| [P2S Code Anatomy](docs/P2S_CODE_ANATOMY.md) | P2S code walkthrough with call hierarchy diagrams |
| [S2P Mechanism](docs/S2P_MECHANISM.md) | S2P algorithm: BiLSTM phrase alignment with Viterbi decoding |
| [S2P Code Anatomy](docs/S2P_CODE_ANATOMY.md) | S2P v2.1 architecture and model internals |
| [Workflow](docs/WORKFLOW.md) | End-to-end pipeline workflow |
| [Data Preparation](docs/DATA_PREPARATION.md) | XML to XLSX data conversion pipeline |
| [Performance](docs/PERFORMANCE.md) | Benchmarks and optimization guide |
| [Cloud GPU Testing](docs/CLOUD_GPU_TESTING.md) | RunPod H200 deployment |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |

---

## Development Environment

### Docker (Recommended for GPU)

```bash
docker-compose up -d
docker-compose exec csp bash
```

- **Base image**: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`
- **Package manager**: `uv` (2-10x faster than pip)
- **Services**: CSP workspace + MSSQL 2022 (dictionary data)

### Cloud GPU (RunPod)

- **Recommended**: H200 SXM ($3.59/hr)
- **Full P2S test** (4,934 paragraphs): ~5.2 hours
- See [Cloud GPU Testing](docs/CLOUD_GPU_TESTING.md) for setup instructions

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## License

This project is licensed under [CC BY 4.0](LICENSE). You are free to share and adapt the material with appropriate attribution.

---

## 한국어 상세 안내

### 프로젝트 소개

CSP는 한문 고전 문헌을 **문단→문장으로 분할(P2S)** 하고 **문장→구로 분할하여 1:1 정렬(S2P)** 하는 작업을 자동화하는 시스템입니다.

### 주요 특징
- **무결성 보장**: 원문 문자 100% 보존 (공백 외 손실 없음)
- **GPU 가속**: CUDA 기반 고속 처리
- **다중 전략**: SuPar 구문분석 + 경계 모델 + DP 정렬 + BGE refinement

### 성능 (2026-02-10 기준)

| 파이프라인 | F1 | Precision | Recall | 원문유사도 | 테스트 규모 |
|----------|-----|-----------|--------|-----------|-----------|
| **P2S** | **0.9384** | 1.0 | 0.8840 | 0.9759 | 4,934문단 |
| **S2P** (v2.1) | **0.8555** | 1.0 | 0.7475 | 0.9362 | 446문장 |

### 모니터링 & 분석

```bash
# 마크다운 대시보드 실행
cd hyeonto && python md_server.py --port 8080
# http://127.0.0.1:8080/dashboard.html
```

- **K=3 클러스터 분석**: 현토 마커 & 서종 분포
- **임베딩 시각화**: 2D/3D UMAP 오버레이
- **Sankey 다이어그램**: P2S ↔ S2P 흐름 분석

### 핵심 기술

- **임베더**: BGE-M3 FlagModel (GPU 가속)
- **경계 모델**: Cross-Attention 기반 Boundary Tagger
- **구문분석**: SuPar-Kanbun (한문) + Stanza (한국어)
- **토크나이저**: SikuBERT (한문) + Kiwipiepy (현토/한국어)
- **구 정렬**: BiLSTM + Guided Attention + Viterbi 디코딩

---

**Last updated**: 2026-02-14
