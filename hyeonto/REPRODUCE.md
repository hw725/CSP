# Hyeonto 연구 재현 가이드 (Reproduction Guide) - v6.9.4 Final

본 문서는 hyeonto 프로젝트의 데이터 가용성 정보와 전체 분석 파이프라인을 처음부터 끝까지 재현하는 방법을 단계별로 설명합니다.

> **📌 분석 단위**
> - **Sentence**: 문장 단위 클러스터링 (150,545건)
> - **Phrase**: 구 단위 클러스터링 (366,222건)

---

## 📂 데이터 가용성 및 안내 (Data Availability)

### 1. 데이터 출처

본 연구에 사용된 현토 데이터는 **동양고전종합DB**(https://db.juntong.or.kr)에서 제공하는 사서삼경 및 기타 유교 경전의 현토본을 기반으로 합니다.

### 2. 저작권 및 재현성 안내

⚠️ **일부 텍스트는 저작권 문제로 공개 접근이 제한될 수 있습니다.**

그러나 공개된 텍스트만으로도 **본 연구의 핵심 발견을 충분히 재현 가능**합니다.

### 3. 데이터 스키마 (CSV)

분석용 데이터셋의 구조:

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

**필수 패키지**:
```bash
pip install regex  # Unicode Script Property 지원
```

### 2. 데이터 배치

- XML 원본: `hyeonto/*.xml`
- 통합 CSV: `hyeonto/datasets/sentence_merged_v2.csv`, `hyeonto/datasets/phrase_merged_v2.csv`

---

## 🔄 전체 파이프라인 실행 (v6.9.4)

### Phase 0: Production 파이프라인 (권장)

**v6.9.4에서는 `run_full_pipeline.py`가 공식 파이프라인입니다.**

```bash
# 전체 파이프라인 실행 (from scratch)
docker compose run --rm csp python hyeonto/run_full_pipeline.py
```

이 스크립트는 다음을 수행합니다:
1. XML 원본에서 데이터 추출
2. 171개 마커 정규화 (Zero-Gap)
3. `\p{Hangul}+` Unicode Regex로 옛한글 포착
4. BGE-M3 임베딩 생성 및 캐시
5. K=4 클러스터링 및 프로파일링

---

### Phase 1: 임베딩 캐시 생성 (개별 실행 시)

대규모 Phrase 데이터(약 36만 건)의 임베딩 계산 시간을 절약하기 위해 캐시를 생성합니다.

```bash
# Sentence 임베딩 캐시 생성 (약 30분)
docker exec csp-workspace python scripts/cluster_pa_boundary_functions.py \
    --input hyeonto/datasets/sentence_merged_v2.csv \
    --out-dir hyeonto/reports/temp \
    --k 4 --use-src --use-tgt \
    --save-embeddings hyeonto/cache/sentence_embeddings.npy \
    --device-id 0 --seed 42

# Phrase 임베딩 캐시 생성 (약 5-6시간, resume 지원)
docker exec csp-workspace python scripts/find_optimal_k.py \
    --csv hyeonto/datasets/phrase_merged_v2.csv \
    --out-dir hyeonto/reports/optimal_k_phrase \
    --k-min 4 --k-max 32 --k-step 2 \
    --save-embeddings hyeonto/cache/phrase_embeddings.npy \
    --device-id 0 --seed 42
```

> **참고**: 중단 시 `hyeonto/cache/phrase_embeddings_resume.npy`에 중간 결과가 저장되며, 재시작 시 자동으로 이어서 진행됩니다.

---

### Phase 2: Sentence 클러스터링 (K=4)

```bash
docker exec csp-workspace python scripts/cluster_pa_boundary_functions.py \
    --input hyeonto/datasets/sentence_merged_v2.csv \
    --out-dir hyeonto/reports/sentence_k4_normalized \
    --k 4 --load-embeddings hyeonto/cache/sentence_embeddings.npy \
    --use-src --use-tgt --seed 42 --max-boundaries 500000
```

---

### Phase 3: Phrase 클러스터링 (K=4)

```bash
docker exec csp-workspace python scripts/cluster_sa_boundary_functions.py \
    --input hyeonto/datasets/phrase_merged_v2.csv \
    --out-dir hyeonto/reports/phrase_k4_normalized \
    --k 4 --load-embeddings hyeonto/cache/phrase_embeddings.npy \
    --use-src --use-tgt --seed 42 --max-boundaries 500000
```

---

### Phase 4: 프로파일링

```bash
# Sentence K=4 프로파일링
docker exec csp-workspace python scripts/profile_boundary_clusters.py \
    --csv hyeonto/reports/sentence_k4_normalized/boundary_clusters.csv \
    --out hyeonto/reports/sentence_k4_normalized/sentence_cluster_profile.md

# Phrase K=4 프로파일링
docker exec csp-workspace python scripts/profile_boundary_clusters.py \
    --csv hyeonto/reports/phrase_k4_normalized/sa_boundary_clusters.csv \
    --out hyeonto/reports/phrase_k4_normalized/phrase_cluster_profile.md
```

---

### Phase 5: 시각화

```bash
# UMAP 3D/2D 시각화 (Docker GPU 환경)
docker compose run --rm csp python hyeonto/analyze_embedding_overlay.py

# Sentence-Phrase Sankey 다이어그램 (K=4 ↔ K=4)
docker compose run --rm csp python hyeonto/generate_sankey_diagrams.py
```

---

## 🧪 결과 검증 (Verification)

### 1. 데이터 수치 확인 (v6.9.4)

| 데이터셋 | 예상 행 수 (header 제외) |
|:---|:---:|
| Sentence K=4 | 150,545 |
| Phrase K=4 | 366,222 |

### 2. 주요 지표 확인

- **Sentence K=4 p1 Canonicity**: 약 13.4%
- **Phrase K=4 p5 Canonicity**: 약 16.5%
- **코퍼스 무결성**: `잇고` 359/178건, `잇가` 1,095/963건 (Sentence/Phrase)

### 3. 마커 스키마 검증

```bash
# 정규화 갭 분석 (Zero-Gap 확인)
python hyeonto/analyze_normalization_gaps.py
# 예상 결과: "Zero additional candidates" for 171-entry schema
```

---

## ⏱️ 예상 소요 시간

| 단계 | GPU (RTX 3090) | CPU |
|------|:---:|:---:|
| 임베딩 캐시 (Phrase 36만건) | 약 5-6시간 | 약 24시간+ |
| 클러스터링 (캐시 사용) | 약 2분/K | 약 5분/K |
| 프로파일링 | 약 5분 | 약 10분 |
| 시각화 | 약 5분 | 약 10분 |

---

## 🐛 문제 해결 (Troubleshooting)

- **GPU 메모리 부족**: `--batch 64` (또는 더 낮게) 옵션 추가
- **Phrase 임베딩 중단**: `hyeonto/cache/phrase_embeddings_resume.npy`에서 자동 재개
- **Docker 컨테이너**: `docker start csp-workspace` 명령으로 확인
- **옛한글 누락**: `regex` 라이브러리와 `\p{Hangul}+` 패턴 사용 확인

---

## 📋 기술 표준 (v6.9.4)

### Unicode Regex 표준화

```python
import regex  # 'regex' 라이브러리 (re 아님)

# 한글 (옛한글 포함)
HANGUL_PATTERN = r'\p{Hangul}+'

# 한자
HANJA_PATTERN = r'\p{Han}'
```

**적용 스크립트**:
- `analyze_normalization_gaps.py`
- `analyze_cooccurrence_normalized.py`
- `run_full_pipeline.py`

---

**업데이트 일자**: 2026-01-27
**작성자**: CSP Research Team
