## 성능 최적화 가이드 (2026-02-10)

### 1. P2S (Paragraph-to-Sentence) 성능

#### 권장 설정
```bash
python p2s/main.py input.xlsx output.xlsx \
    --use-boundary-model --embedder bge \
    --checkpoint-path output_ckpt.csv
```

#### 벤치마크 (4,934문단 전체, RunPod H200 SXM)
| 항목 | 사양 | 결과 |
|------|------|------|
| **F1 (Micro)** | 4,934문단 전체 | **0.9384** |
| **Precision** | - | 1.0 |
| **Recall** | - | 0.8840 |
| **원문유사도** | 평균 | 0.9759 |
| **처리 시간** | RunPod H200 SXM | ~5.2시간 |
| **GPU 메모리** | Max Peak | ~6GB |
| **Seed 안정성** | 5개 seed (0-4) | 결정적 (동일 결과) |

> 100문단 Docker 로컬 테스트: F1=0.9273, ~20분 소요.

---

### 2. S2P (Sentence-to-Phrase) 성능

#### 권장 설정 (v2.1 Phrase Alignment)
```bash
python -u -m s2p.main datasets/sentence/test.csv output.csv \
    --embedder bge --use-phrase-alignment \
    --chunk-size 300 --batch-size 32
```
> **Batch Size 팁**: Embeddings 연산 시 128까지 가능하나, Phrase Alignment 추론 시 GPU OOM 방지를 위해 **32** 권장.

#### 벤치마크 (446문장, Docker RTX 3070 Ti)
| 항목 | 사양 | 결과 |
|------|------|------|
| **F1** | v2.1 Phrase Alignment | **0.8555** |
| **Precision** | - | 1.0 |
| **Recall** | - | 0.7475 |
| **평균 유사도** | BGE-M3 | 0.9362 |
| **처리 시간** | RTX 3070 Ti (8GB) / Docker | ~50분 |
| **모델 파라미터** | PhraseAlignmentModel | 6.7M |

> v2.1 Phrase Alignment (Source BiLSTM + Guided Attention + BGE-M3 1024d + Viterbi)이 baseline DP 대비 **F1 5.5배 향상** (0.1563 → 0.8555).

---

### 3. 디스크 및 캐시 최적화

#### 임베딩 캐시 전략 (BGE)
- **위치**: `~/.cache/huggingface` 및 로컬 `.cache`
- **전략**: **Disk Cache**는 청크 단위로 저장되어 I/O 병목을 줄임.
- **팁**: 첫 실행 이후 재실행 시 **약 3-4배 속도 향상** (임베딩 캐시 Hit).

#### 로그 및 체크포인트
- **로그**: `python -u` (Unbuffered) 옵션을 사용하여 실시간 로그 확인 권장 (Docker logs 지연 방지).
- **P2S 체크포인트**: `--checkpoint-path`로 지정. 기본값은 `{output}_ckpt.csv`. `--checkpoint-every N`으로 저장 주기 설정 (기본 5문단). 중단 후 재실행 시 자동 resume.
- **S2P 체크포인트**: `{output_filename}_checkpoint.csv`가 청크 단위로 자동 생성됨.

---

**최근 업데이트**: 2026년 02월 10일 - P2S F1=0.9384 (4,934문단 전체), S2P F1=0.8555 (v2.1 Phrase Alignment, 446문장) 반영
