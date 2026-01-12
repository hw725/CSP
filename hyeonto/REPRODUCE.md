# Hyeonto 연구 재현 가이드 (Reproduction Guide) - v6

본 문서는 hyeonto 프로젝트의 데이터 가용성 정보와 전체 분석 파이프라인을 처음부터 끝까지 재현하는 방법을 단계별로 설명합니다.

---

## 📂 데이터 가용성 및 안내 (Data Availability)

### 1. 데이터 출처

본 연구에 사용된 현토 데이터는 다음 DB에서 제공하는 사서삼경 및 기타 유교 경전의 현토본을 기반으로 합니다.

| DB | URL |
|:---|:---|
| **동양고전종합DB** | https://db.juntong.or.kr |
| **동양고전번역용례** | https://db.juntong.or.kr/example |

### 2. 저작권 및 재현성 안내

⚠️ **일부 텍스트는 저작권 문제로 본 저장소에 포함되어 있지 않으며 공개 접근이 제한될 수 있습니다.**

그러나 공개된 텍스트만으로도 **본 연구의 핵심 발견을 충분히 재현 가능**합니다. 가중치 민감도 테스트 결과, 클러스터 구성은 데이터 부분집합에서도 다음과 같은 특성에 대해 안정적이었습니다:
- 사서 클러스터 분리 현상
- 정의형 마커 우세 패턴  
- PA→SA 위계적 흐름

### 3. 데이터 스키마 (CSV)

분석용 데이터셋(`pa_merged_v2.csv`, `sa_merged_v2.csv` 등)의 구조는 다음과 같습니다. 직접 데이터를 수집하여 분석할 경우 이 형식을 따라야 합니다.

| 컬럼명 | 설명 | 예시 |
|:---|:---|:---|
| `src_l` | 좌측 한문 원문 | 子曰學而時習之 |
| `src_r` | 우측 한문 원문 | 不亦說乎 |
| `tgt_l` | 좌측 번역문 | 공자께서 말씀하시길 |
| `tgt_r` | 우측 번역문 | 기쁘지 아니한가 |
| `marker` | 현토 마커 | 하시니 |
| `book` | 도서명 | 논어집주 |

---

## 📋 사전 준비 (Prerequisites)

### 1. 환경 설정

**필요한 소프트웨어**:
- Python 3.9 이상
- CUDA 11.8 이상 (GPU 사용 시)
- Docker (권장 - `csp-workspace` 컨테이너)

**Python 패키지 설치**:
```bash
pip install -r requirements.txt
```

### 2. 데이터 배치

분석을 시작하기 전, 위 DB에서 취득하거나 준비된 데이터를 다음 경로에 배치하십시오.
- XML 원본: `hyeonto/*.xml`
- 통합 CSV: `hyeonto/datasets/pa_merged_v2.csv`, `hyeonto/datasets/sa_merged_v2.csv`

---

## 🔄 전체 파이프라인 실행 (v6)

### Step 1: PA 클러스터링 (번역문 포함)

**목적**: 문장 단위 경계를 한문+번역문 결합 임베딩으로 클러스터링

```bash
# Docker 환경에서 실행 (권장)
docker exec csp-workspace python scripts/cluster_pa_boundary_functions.py \
    --input hyeonto/datasets/pa_merged_v2.csv \
    --out-dir hyeonto/reports/pa_boundary_v6_full \
    --k 16 \
    --max-boundaries 500000 \
    --use-src \
    --use-tgt \
    --device-id 0 \
    --seed 42 \
    --batch 128
```

---

### Step 2: SA 클러스터링 (번역문 포함)

**목적**: 구 단위 경계를 한문+번역문 결합 임베딩으로 클러스터링

```bash
docker exec csp-workspace python scripts/cluster_sa_boundary_functions.py \
    --input hyeonto/datasets/sa_merged_v2.csv \
    --out-dir hyeonto/reports/sa_boundary_v6_full \
    --k 16 \
    --max-boundaries 500000 \
    --use-src \
    --use-tgt \
    --device-id 0 \
    --seed 42
```

---

### Step 3: 클러스터 프로파일링

**목적**: 클러스터별 사서 비율, 마커 분포, 한자 빈도 분석

```bash
# PA 프로파일링
docker exec csp-workspace python scripts/profile_boundary_clusters.py \
    --csv hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv \
    --out hyeonto/reports/pa_boundary_v6_full/pa_cluster_profile.md

# SA 프로파일링
docker exec csp-workspace python scripts/profile_boundary_clusters.py \
    --csv hyeonto/reports/sa_boundary_v6_full/sa_boundary_clusters.csv \
    --out hyeonto/reports/sa_boundary_v6_full/sa_cluster_profile.md
```

---

### Step 4: 시각화 및 라벨링

```bash
# PA 시각화
docker exec csp-workspace python scripts/visualize_clusters_v6.py \
    --csv hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv \
    --out-dir hyeonto/reports/pa_boundary_v6_full/visualization

# 휴리스틱 라벨링
docker exec csp-workspace python scripts/describe_boundary_clusters.py \
    --csv hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv \
    --out hyeonto/reports/pa_boundary_v6_full/boundary_clusters_labeled.md
```

---

## 🧪 결과 검증 (Verification)

### 1. 데이터 수치 확인
- PA 결과 행 수 예상: 87,944 (header 포함)
- SA 결과 행 수 예상: 294,890 (header 포함)

### 2. 주요 지표 확인
시각화 결과물인 `cluster_embedding.html`을 브라우저에서 열어 클러스터 분리 양상을 확인하고, `pa_cluster_profile.md`에서 사서 가중치 및 마커 분포가 연구 보고서와 일치하는지 대조합니다.

---

## ⏱️ 예상 소요 시간

| 단계 | GPU (RTX 3090 기준) | CPU |
|------|-----|-----|
| PA/SA 클러스터링 | 약 6.5시간 | 약 24시간 |
| 분석 및 시각화 | 약 20분 | 약 30분 |

---

## 🐛 문제 해결 (Troubleshooting)

- **GPU 메모리 부족**: 실행 옵션에 `--batch 64` (또는 더 낮게)를 추가하십시오.
- **Docker 컨테이너**: `docker start csp-workspace` 명령으로 컨테이너가 실행 중인지 확인하십시오.
- **컬럼 오류**: 입력 CSV에 `tgt_l`, `tgt_r` (번역문) 컬럼이 반드시 포함되어야 v6 분석이 가능합니다.

---

**업데이트 일자**: 2026-01-12
**작성자**: CSP Research Team
