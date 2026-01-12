# Hyeonto 연구 재현 가이드 (Reproduction Guide) - v6

본 문서는 hyeonto 프로젝트의 전체 분석 파이프라인을 처음부터 끝까지 재현하는 방법을 단계별로 설명합니다.

---

## 📋 사전 준비

### 1. 환경 설정

**필요한 소프트웨어**:
- Python 3.9 이상
- CUDA 11.8 이상 (GPU 사용 시)
- Docker (권장 - csp-workspace 컨테이너)

**Python 패키지**:
```bash
pip install -r requirements.txt
```

주요 패키지:
- pandas >= 1.5.0
- numpy >= 1.24.0
- scikit-learn >= 1.2.0
- umap-learn >= 0.5.0
- plotly >= 5.0.0
- regex
- FlagEmbedding (bge-m3)

### 2. 데이터 확인

```bash
# XML 원본 파일들이 있는지 확인
ls hyeonto/*.xml

# 통합 데이터셋 확인
wc -l hyeonto/datasets/pa_merged_v2.csv
# 예상: 120203 (120,202 + header)

wc -l hyeonto/datasets/sa_merged_v2.csv
# 예상: 415092 (415,091 + header)
```

---

## 🔄 전체 파이프라인 (v6 - 번역문 포함)

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

**핵심 옵션**:
- `--use-src`: 한문 원문 포함
- `--use-tgt`: **번역문 포함 (v6 핵심)**
- `--device-id 0`: GPU 사용

**출력**:
- `hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv` (~45MB)
- `hyeonto/reports/pa_boundary_v6_full/boundary_clusters.md`

**예상 소요 시간**:
- GPU: 1~1.5시간
- CPU: 4~6시간

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

**출력**:
- `hyeonto/reports/sa_boundary_v6_full/sa_boundary_clusters.csv` (~51MB)
- `hyeonto/reports/sa_boundary_v6_full/sa_boundary_clusters.md`

**예상 소요 시간**:
- GPU: 4~5시간 (데이터 약 29.5만 건)
- CPU: 12~18시간

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

**출력**:
- `pa_cluster_profile.md` (~37KB)
- `sa_cluster_profile.md` (~22KB)

---

### Step 4: 시각화

**목적**: 클러스터 2D 매핑 (PCA/t-SNE)

```bash
# PA 시각화
docker exec csp-workspace python scripts/visualize_clusters_v6.py \
    --csv hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv \
    --out-dir hyeonto/reports/pa_boundary_v6_full/visualization

# SA 시각화
docker exec csp-workspace python scripts/visualize_clusters_v6.py \
    --csv hyeonto/reports/sa_boundary_v6_full/sa_boundary_clusters.csv \
    --out-dir hyeonto/reports/sa_boundary_v6_full/visualization
```

**출력**:
- `cluster_embedding.html` (인터랙티브 Plotly)
- `cluster_embedding.csv` (좌표 데이터)
- `config.json` (설정 기록)

---

### Step 5: 휴리스틱 라벨링

**목적**: 클러스터에 인간 가독성 라벨 부여

```bash
docker exec csp-workspace python scripts/describe_boundary_clusters.py \
    --csv hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv \
    --out hyeonto/reports/pa_boundary_v6_full/boundary_clusters_labeled.md
```

---

## 🧪 결과 검증

### 1. 데이터 무결성 확인

```bash
# PA 클러스터링 결과 행 수
wc -l hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv
# 예상: 87944 (87,943 + header)

# SA 클러스터링 결과 행 수
wc -l hyeonto/reports/sa_boundary_v6_full/sa_boundary_clusters.csv
# 예상: 294890 (294,889 + header)

# 클러스터 수 확인
cut -d',' -f1 hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv | \
    sort -u | wc -l
# 예상: 17 (16 클러스터 + header)
```

### 2. 주요 지표 확인

```python
import pandas as pd

# PA 클러스터 크기 분포
df = pd.read_csv("hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv")
print(df["cluster_id"].value_counts().sort_index())

# 사서 비율 확인
CANON = ["논어", "맹자", "대학", "중용"]
df["is_canon"] = df["book_name"].str.contains("|".join(CANON), na=False)
print(df.groupby("cluster_id")["is_canon"].mean().sort_values(ascending=False))
```

### 3. 시각화 확인

브라우저에서 다음 HTML 파일 열기:
```
hyeonto/reports/pa_boundary_v6_full/visualization/cluster_embedding.html
hyeonto/reports/sa_boundary_v6_full/visualization/cluster_embedding.html
```

---

## 📊 예상 산출물 체크리스트 (v6)

### 데이터셋
- [ ] `hyeonto/datasets/pa_merged_v2.csv` (120,202행)
- [ ] `hyeonto/datasets/sa_merged_v2.csv` (415,091행)

### PA 분석 결과
- [ ] `pa_boundary_v6_full/boundary_clusters.csv` (87,943행)
- [ ] `pa_boundary_v6_full/boundary_clusters.md`
- [ ] `pa_boundary_v6_full/boundary_clusters_labeled.md`
- [ ] `pa_boundary_v6_full/pa_cluster_profile.md`
- [ ] `pa_boundary_v6_full/visualization/cluster_embedding.html`

### SA 분석 결과
- [ ] `sa_boundary_v6_full/sa_boundary_clusters.csv` (294,889행)
- [ ] `sa_boundary_v6_full/sa_boundary_clusters.md`
- [ ] `sa_boundary_v6_full/sa_cluster_profile.md`
- [ ] `sa_boundary_v6_full/visualization/cluster_embedding.html`

### 마스터 문서
- [ ] `reports/FINAL_ANALYSIS_REPORT.md` (v6 반영)

---

## ⏱️ 전체 소요 시간 예상 (v6)

| 단계 | GPU | CPU |
|------|-----|-----|
| Step 1: PA 클러스터링 | 1.5시간 | 6시간 |
| Step 2: SA 클러스터링 | 5시간 | 18시간 |
| Step 3: 프로파일링 | 10분 | 10분 |
| Step 4: 시각화 | 5분 | 10분 |
| Step 5: 라벨링 | 5분 | 5분 |
| **총계** | **~7시간** | **~25시간** |

---

## 🐛 문제 해결 (Troubleshooting)

### 문제 1: GPU 메모리 부족

```bash
# 배치 크기 줄이기
--batch 64  # 기본 128에서 줄임
```

### 문제 2: Docker 환경 접속

```bash
# 컨테이너 시작
docker start csp-workspace

# 인터랙티브 접속
docker exec -it csp-workspace bash
```

### 문제 3: 번역문 컬럼 없음

v6 분석을 위해서는 `번역문` 컬럼이 필요합니다:
```python
# 컬럼 확인
df = pd.read_csv("hyeonto/datasets/pa_merged_v2.csv")
print("번역문" in df.columns)  # True여야 함
```

---

## 📞 지원 및 문의

- **문서 업데이트**: 2026-01-11 (v6)
- **작성자**: CSP Research Team

---

**재현 성공 시 다음 단계**:
1. `reports/FINAL_ANALYSIS_REPORT.md`와 결과 비교
2. 주요 지표(Canonicity 48.8% 등)가 일치하는지 확인
3. 시각화 HTML 파일에서 클러스터 분포 확인

**Good Luck!** 🚀
