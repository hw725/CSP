# 신규 탐색 분석 실행 가이드

본 문서는 기존 현토 연구를 넘어 새로운 특성을 탐색하기 위한 4가지 분석 스크립트의 실행 방법을 설명합니다.

---

## 분석 스크립트 목록

| 순서 | 스크립트 | 분석 내용 | 주요 산출물 |
|:---:|:---|:---|:---|
| 1 | `detect_outliers_boundary.py` | 클러스터 이상치 탐지 | 이상치 목록, 분석 리포트 |
| 2 | `analyze_ngram_sequences.py` | n-gram 시퀀스 분석 | 빈도표, 장르별 패턴 |
| 3 | `analyze_cooccurrence_network.py` | 한자-현토 공기 네트워크 | PMI 행렬, 네트워크 시각화 |
| 4 | `analyze_phonetic_patterns.py` | 음운 패턴 분석 | 음운 프로파일, 분포 리포트 |

---

## 실행 명령어

### 1. 이상치 탐지

```bash
# PA 이상치 탐지
docker exec csp-workspace python scripts/detect_outliers_boundary.py \
    --input hyeonto/datasets/pa_merged_v2.csv \
    --out-dir hyeonto/reports/exploratory/outliers_pa \
    --analysis-type PA \
    --k 16 \
    --top-n 200 \
    --device-id 0 \
    --batch 128

# SA 이상치 탐지
docker exec csp-workspace python scripts/detect_outliers_boundary.py \
    --input hyeonto/datasets/sa_merged_v2.csv \
    --out-dir hyeonto/reports/exploratory/outliers_sa \
    --analysis-type SA \
    --k 16 \
    --top-n 200 \
    --device-id 0 \
    --batch 128
```

### 2. n-gram 시퀀스 분석

```bash
# SA 데이터로 n-gram 분석 (구 단위가 연쇄 패턴이 더 촘촘함)
docker exec csp-workspace python scripts/analyze_ngram_sequences.py \
    --input hyeonto/reports/sa_boundary_v6_full/sa_boundary_clusters.csv \
    --out-dir hyeonto/reports/exploratory/ngram_sa \
    --analysis-type SA \
    --n-values 2,3
```

### 3. 한자-현토 공기 네트워크

```bash
# SA 데이터로 공기 네트워크 분석 (데이터량 풍부)
docker exec csp-workspace python scripts/analyze_cooccurrence_network.py \
    --input hyeonto/reports/sa_boundary_v6_full/sa_boundary_clusters.csv \
    --out-dir hyeonto/reports/exploratory/cooccurrence_sa \
    --analysis-type SA \
    --top-hanja 100 \
    --top-markers 30 \
    --top-edges 200
```

### 4. 음운 패턴 분석

```bash
# PA + SA 통합 음운 분석
docker exec csp-workspace python scripts/analyze_phonetic_patterns.py \
    --input-pa hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv \
    --input-sa hyeonto/reports/sa_boundary_v6_full/sa_boundary_clusters.csv \
    --out-dir hyeonto/reports/exploratory/phonetic \
    --min-freq 100
```

---

## 예상 산출물

### `hyeonto/reports/exploratory/` 디렉토리 구조

```
exploratory/
├── outliers_pa/
│   ├── outliers_pa.csv
│   └── outlier_analysis_pa.md
├── outliers_sa/
│   ├── outliers_sa.csv
│   └── outlier_analysis_sa.md
├── ngram_sa/
│   ├── 2gram_frequency_sa.csv
│   ├── 3gram_frequency_sa.csv
│   └── ngram_analysis_sa.md
├── cooccurrence_sa/
│   ├── cooccurrence_matrix_sa.csv
│   ├── associations_sa.csv
│   ├── cooccurrence_network_sa.html
│   └── cooccurrence_analysis_sa.md
└── phonetic/
    ├── phonetic_profile.csv
    └── phonetic_analysis_pa+sa.md
```

---

## 분석 목적 및 기대 발견

### 1. 이상치 탐지
- **목적**: 클러스터 분류에서 벗어난 "비정형" 문장 식별
- **기대 발견**: 학자들의 실험적 현토 시도, 필사 오류, 독특한 저자 스타일

### 2. n-gram 시퀀스
- **목적**: 마커 연쇄 패턴의 장르별 특성 분석
- **기대 발견**: 사서에만 존재하는 "시그니처 시퀀스", 다른 문헌의 모방 증거

### 3. 한자-현토 공기 네트워크
- **목적**: 특정 한자와 특정 현토의 강한 결합 패턴 발견
- **기대 발견**: 사서 특화 한자-현토 조합 규칙, "금지된 조합"의 존재

### 4. 음운 패턴
- **목적**: 현토의 음운론적 특성 (음절, 종성, 모음 조화) 분석
- **기대 발견**: 낭송(朗誦)을 위한 운율적 설계 가능성

---

**작성일**: 2026-01-12
