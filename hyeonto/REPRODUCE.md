# Hyeonto 연구 재현 가이드 (Reproduction Guide) - v6 + Multi-Resolution

본 문서는 hyeonto 프로젝트의 데이터 가용성 정보와 전체 분석 파이프라인을 처음부터 끝까지 재현하는 방법을 단계별로 설명합니다.

---

## 📂 데이터 가용성 및 안내 (Data Availability)

### 1. 데이터 출처

본 연구에 사용된 현토 데이터는 **동양고전종합DB**(https://db.juntong.or.kr)에서 제공하는 사서삼경 및 기타 유교 경전의 현토본을 기반으로 합니다.

### 2. 저작권 및 재현성 안내

⚠️ **일부 텍스트는 저작권 문제로 공개 접근이 제한될 수 있습니다.**

그러나 공개된 텍스트만으로도 **본 연구의 핵심 발견을 충분히 재현 가능**합니다.

### 3. 데이터 스키마 (CSV)

분석용 데이터셋(`pa_merged_v2.csv`, `sa_merged_v2.csv` 등)의 구조:

| 컬럼명 | 설명 | 예시 |
|:---|:---|:---|
| `문단식별자` | 문단 고유 ID | 1 |
| `문장식별자` | 문장 순번 | 2 |
| `원문` | 현토가 포함된 한문 원문 | 孔子는 名丘요 字仲尼니 其先은 宋人이라 |
| `번역문` | 한국어 번역문 | 공자(孔子)는 이름이 구(丘)요... |
| `book_name` | 도서명 | 논어집주 |

---

## 📋 사전 준비 (Prerequisites)

### 1. 환경 설정

**필요한 소프트웨어**:
- Python 3.9 이상
- CUDA 11.8 이상 (GPU 사용 시)
- Docker (권장 - `csp-workspace` 컨테이너)

### 2. 데이터 배치

- XML 원본: `hyeonto/*.xml`
- 통합 CSV: `hyeonto/datasets/sentence_merged_v2.csv`, `hyeonto/datasets/phrase_merged_v2.csv`

---

## 🔄 전체 파이프라인 실행 (v6 + Multi-Resolution)

### Phase 0: 임베딩 캐시 생성 (최초 1회)

대규모 SA 데이터(약 30만 건)의 임베딩 계산 시간을 절약하기 위해 캐시를 생성합니다.

```bash
# PA 임베딩 캐시 생성 (약 20분)
docker exec csp-workspace python scripts/cluster_pa_boundary_functions.py \
    --input hyeonto/datasets/sentence_merged_v2.csv \
    --out-dir hyeonto/reports/temp \
    --k 4 --use-src --use-tgt \
    --save-embeddings hyeonto/cache/pa_embeddings.npy \
    --device-id 0 --seed 42

# SA 임베딩 캐시 생성 (약 4-5시간, resume 지원)
docker exec csp-workspace python scripts/find_optimal_k.py \
    --csv hyeonto/datasets/phrase_merged_v2.csv \
    --out-dir hyeonto/reports/optimal_k_sa_v3 \
    --k-min 4 --k-max 32 --k-step 2 \
    --save-embeddings hyeonto/cache/sa_embeddings.npy \
    --device-id 0 --seed 42
```

> **참고**: 중단 시 `hyeonto/cache/sa_embeddings_resume.npy`에 중간 결과가 저장되며, 재시작 시 자동으로 이어서 진행됩니다.

---

### Phase 1: 최적 K값 분석 (선택)

```bash
# PA 최적 K 분석
docker exec csp-workspace python scripts/find_optimal_k.py \
    --csv hyeonto/datasets/sentence_merged_v2.csv \
    --out-dir hyeonto/reports/optimal_k_pa \
    --k-min 4 --k-max 32 --k-step 2 \
    --load-embeddings hyeonto/cache/pa_embeddings.npy \
    --device-id 0 --seed 42

# SA 최적 K 분석
docker exec csp-workspace python scripts/find_optimal_k.py \
    --csv hyeonto/datasets/phrase_merged_v2.csv \
    --out-dir hyeonto/reports/optimal_k_sa_v3 \
    --k-min 4 --k-max 32 --k-step 2 \
    --load-embeddings hyeonto/cache/sa_embeddings.npy \
    --device-id 0 --seed 42
```

**결과**: 
- PA 권장: K=4 (거시), K=14 (미시)
- SA 권장: K=4 (거시), K=24 (미시)

---

### Phase 2: PA 클러스터링 (K=4, K=14)

```bash
# PA K=4 (거시적)
docker exec csp-workspace python scripts/cluster_pa_boundary_functions.py \
    --input hyeonto/datasets/sentence_merged_v2.csv \
    --out-dir hyeonto/reports/sentence_boundary_k4_full \
    --k 4 --load-embeddings hyeonto/cache/pa_embeddings.npy \
    --use-src --use-tgt --seed 42 --max-boundaries 500000

# PA K=14 (미시적)
docker exec csp-workspace python scripts/cluster_pa_boundary_functions.py \
    --input hyeonto/datasets/sentence_merged_v2.csv \
    --out-dir hyeonto/reports/sentence_boundary_k14_full \
    --k 14 --load-embeddings hyeonto/cache/pa_embeddings.npy \
    --use-src --use-tgt --seed 42 --max-boundaries 500000
```

---

### Phase 3: SA 클러스터링 (K=4, K=24)

```bash
# SA K=4 (거시적)
docker exec csp-workspace python scripts/cluster_sa_boundary_functions.py \
    --input hyeonto/datasets/phrase_merged_v2.csv \
    --out-dir hyeonto/reports/phrase_boundary_k4_full \
    --k 4 --load-embeddings hyeonto/cache/sa_embeddings.npy \
    --use-src --use-tgt --seed 42 --max-boundaries 500000

# SA K=24 (미시적)
docker exec csp-workspace python scripts/cluster_sa_boundary_functions.py \
    --input hyeonto/datasets/phrase_merged_v2.csv \
    --out-dir hyeonto/reports/phrase_boundary_k24_full \
    --k 24 --load-embeddings hyeonto/cache/sa_embeddings.npy \
    --use-src --use-tgt --seed 42 --max-boundaries 500000
```

---

### Phase 4: 프로파일링

```bash
# PA K=4 프로파일링
docker exec csp-workspace python scripts/profile_boundary_clusters.py \
    --csv hyeonto/reports/sentence_boundary_k4_full/boundary_clusters.csv \
    --out hyeonto/reports/sentence_boundary_k4_full/sentence_cluster_profile.md

# PA K=14 프로파일링
docker exec csp-workspace python scripts/profile_boundary_clusters.py \
    --csv hyeonto/reports/sentence_boundary_k14_full/boundary_clusters.csv \
    --out hyeonto/reports/sentence_boundary_k14_full/sentence_cluster_profile.md

# SA K=4 프로파일링
docker exec csp-workspace python scripts/profile_boundary_clusters.py \
    --csv hyeonto/reports/phrase_boundary_k4_full/sa_boundary_clusters.csv \
    --out hyeonto/reports/phrase_boundary_k4_full/phrase_cluster_profile.md

# SA K=24 프로파일링
docker exec csp-workspace python scripts/profile_boundary_clusters.py \
    --csv hyeonto/reports/phrase_boundary_k24_full/sa_boundary_clusters.csv \
    --out hyeonto/reports/phrase_boundary_k24_full/phrase_cluster_profile.md

# SA K=24 심층 프로파일링 (Lift, Entropy, Syntactic Guess)
docker exec csp-workspace python scripts/profile_deep_sa.py \
    --csv hyeonto/reports/phrase_boundary_k24_full/sa_boundary_clusters.csv \
    --out-dir hyeonto/reports/phrase_boundary_k24_full
```

---

### Phase 5: 시각화

```bash
# PA 고급 시각화 (Convex Hull + 추세선)
docker exec csp-workspace python scripts/visualize_advanced_boundary.py \
    --csv hyeonto/reports/sentence_boundary_k4_full/boundary_clusters.csv \
    --npy hyeonto/cache/pa_embeddings.npy \
    --out-dir hyeonto/reports/sentence_boundary_k4_full/visualization

# SA 고급 시각화
docker exec csp-workspace python scripts/visualize_advanced_boundary.py \
    --csv hyeonto/reports/phrase_boundary_k4_full/sa_boundary_clusters.csv \
    --npy hyeonto/cache/sa_embeddings.npy \
    --out-dir hyeonto/reports/phrase_boundary_k4_full/visualization

# 클러스터 분화 시각화 (PA K=4 → K=14)
docker exec csp-workspace python scripts/visualize_cluster_flow.py \
    --npy hyeonto/cache/pa_embeddings.npy \
    --k-small 4 --k-large 14 \
    --out-dir hyeonto/reports/exploratory/cluster_flow_sentence --seed 42

# 클러스터 분화 시각화 (SA K=4 → K=24)
docker exec csp-workspace python scripts/visualize_cluster_flow.py \
    --npy hyeonto/cache/sa_embeddings.npy \
    --k-small 4 --k-large 24 \
    --out-dir hyeonto/reports/exploratory/cluster_flow_phrase --seed 42
```

---

### Phase 6: PA-SA Sankey 다이어그램

```bash
# PA(K=4) ↔ SA(K=4)
docker exec csp-workspace python scripts/visualize_pa_sa_sankey.py \
    --pa-csv hyeonto/reports/sentence_boundary_k4_full/boundary_clusters.csv \
    --sa-csv hyeonto/reports/phrase_boundary_k4_full/sa_boundary_clusters.csv \
    --pa-k 4 --sa-k 4 --out-dir hyeonto/reports/exploratory/pa_sa_sankey

# PA(K=4) ↔ SA(K=24)
docker exec csp-workspace python scripts/visualize_pa_sa_sankey.py \
    --pa-csv hyeonto/reports/sentence_boundary_k4_full/boundary_clusters.csv \
    --sa-csv hyeonto/reports/phrase_boundary_k24_full/sa_boundary_clusters.csv \
    --pa-k 4 --sa-k 24 --out-dir hyeonto/reports/exploratory/pa_sa_sankey

# PA(K=14) ↔ SA(K=24)
docker exec csp-workspace python scripts/visualize_pa_sa_sankey.py \
    --pa-csv hyeonto/reports/sentence_boundary_k14_full/boundary_clusters.csv \
    --sa-csv hyeonto/reports/phrase_boundary_k24_full/sa_boundary_clusters.csv \
    --pa-k 14 --sa-k 24 --out-dir hyeonto/reports/exploratory/pa_sa_sankey
```

---

## 🧪 결과 검증 (Verification)

### 1. 데이터 수치 확인

| 데이터셋 | 예상 행 수 (header 제외) |
|:---|:---:|
| PA K=4 | 87,943 |
| PA K=14 | 87,943 |
| SA K=4 | 294,889 |
| SA K=24 | 294,889 |

### 2. 주요 지표 확인

- **PA K=4 p1 Canonicity**: 약 13.4%
- **SA K=4 p1 Canonicity**: 약 13.4%
- **SA K=24 p5 Syntactic Function**: Topic (주제/대조) 53%
- **SA K=24 p22 Lift**: 춘추좌씨전 x10.0

---

## ⏱️ 예상 소요 시간

| 단계 | GPU (RTX 3090) | CPU |
|------|:---:|:---:|
| 임베딩 캐시 (SA 30만건) | 약 4-5시간 | 약 20시간+ |
| 클러스터링 (캐시 사용) | 약 2분/K | 약 5분/K |
| 프로파일링 | 약 5분 | 약 10분 |
| 시각화 | 약 5분 | 약 10분 |

---

## 🐛 문제 해결 (Troubleshooting)

- **GPU 메모리 부족**: `--batch 64` (또는 더 낮게) 옵션 추가
- **SA 임베딩 중단**: `hyeonto/cache/sa_embeddings_resume.npy`에서 자동 재개
- **Docker 컨테이너**: `docker start csp-workspace` 명령으로 확인

---

**업데이트 일자**: 2026-01-13
**작성자**: CSP Research Team
