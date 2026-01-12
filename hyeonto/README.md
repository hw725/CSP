# 현토(懸吐) 분석 프로젝트 (v6 + Multi-Resolution)

> 사서삼경(四書三經) 등 유교 경전의 **현토(懸吐)** 패턴을 클러스터링 및 시각화하여 한문 구문 기능을 분석하는 연구 프로젝트

---

## 📚 프로젝트 개요

### 현토(懸吐)란?

**현토(懸吐)**는 한문 원문에 한글 토씨(조사, 어미 등)를 붙여 한국어로 읽을 수 있도록 하는 전통적인 독법입니다.

예시:
- 원문: `大學之道는 在明明德하며`
- 한자: `大學之道` / 현토: `는`
- 한자: `在明明德` / 현토: `하며`

### 연구 목표

1. **현토 패턴 클러스터링**: 한문 경계(boundary)에서 나타나는 현토의 기능적 역할 분류
2. **PA-SA 위계적 분석**: 문장(PA)과 구(SA) 수준의 이중 레벨 분석
3. **다중 해상도 분석 (Multi-Resolution)**: K=4 (거시적) 및 K=14/K=24 (미시적) 분석으로 다층적 패턴 규명
4. **번역문 통합 임베딩**: 한문 원문과 한국어 번역문을 결합하여 화용론적 뉘앙스 포착 (v6)

---

## 📂 디렉토리 구조

```
hyeonto/
├── datasets/                  # 학습용 데이터셋
│   ├── pa_merged_v2.csv       # PA 통합 데이터 (87,943건)
│   └── sa_merged_v2.csv       # SA 통합 데이터 (294,889건)
├── cache/                     # 임베딩 캐시
│   ├── pa_embeddings.npy      # PA BGE-M3 임베딩 (약 300MB)
│   └── sa_embeddings.npy      # SA BGE-M3 임베딩 (약 1.2GB)
├── reports/                   # 분석 결과
│   ├── FINAL_ANALYSIS_REPORT.md       # ⭐ 통합 마스터 리포트
│   ├── dashboard.html                 # ⭐ 인터랙티브 대시보드
│   ├── pa_boundary_v6_full/           # PA v6 클러스터링 (K=16)
│   ├── sa_boundary_v6_full/           # SA v6 클러스터링 (K=16)
│   ├── pa_boundary_k4_full/           # PA K=4 클러스터링 (거시적)
│   ├── pa_boundary_k14_full/          # PA K=14 클러스터링 (미시적)
│   ├── sa_boundary_k4_full/           # SA K=4 클러스터링 (거시적)
│   ├── sa_boundary_k24_full/          # SA K=24 클러스터링 (미시적) + 심층 프로파일
│   ├── optimal_k_pa/                  # PA 최적 K값 분석
│   ├── optimal_k_sa_v3/               # SA 최적 K값 분석
│   ├── crossmatch_v6/                 # PA-SA 교차 분석 (Sankey)
│   └── exploratory/                   # 탐색적 분석 (Sankey, 비교, 흐름)
├── BIAS_VALIDATION.md         # 편향 검증 보고서
├── DATA_PROVENANCE.md         # 데이터 출처 설명
├── EXPLORATORY_ANALYSIS.md    # 탐색적 분석 가이드
├── KEY_FINDINGS.md            # 핵심 발견 사항
├── REPRODUCE.md               # 재현 가이드
├── VISUALIZATION_GUIDE.md     # 시각화 해석 가이드
└── jti_*.xml                  # 원본 현토 XML 파일들
```

---

## 📖 분석 대상 텍스트

### 사서(四書) - 조선시대 원본 현토
| 코드 | 서명 | 특성 |
|------|------|------|
| 1h0301 | 논어집주(論語集註) | ✅ 진본(眞本) |
| 1h0601 | 맹자집주(孟子集註) | ✅ 진본(眞本) |
| 1h0801 | 대학장구(大學章句) | ✅ 진본(眞本) |
| 1h1001 | 중용장구(中庸章句) | ✅ 진본(眞本) |

### 삼경(三經) 및 기타
| 코드 | 서명 | 특성 |
|------|------|------|
| 1a0201-02 | 주역전의(周易傳義) | ⚠️ 혼합 (원본+재구성) |
| 1b0201-02 | 서경집전(書經集傳) | ✅ 진본 |
| 1c0201-02 | 시경집전(詩經集傳) | ⚠️ 혼합 |
| - | 춘추좌씨전, 자치통감강목 등 | ❌ 현대 재구성 |

---

## 🔬 다중 해상도 분석 (Multi-Resolution Analysis)

### 최적 K값 결정

| 데이터 | 권장 K (거시적) | 기능적 K (미시적) | 근거 |
|:---:|:---:|:---:|:---|
| **PA** | K=4 | K=14 | Silhouette + Calinski-Harabasz 최적화 |
| **SA** | K=4 | K=24 | Davies-Bouldin 최적화 |

### 분석 계층

```
     거시적 분석 (K=4)
     ┌─────────────────────────────────────────┐
     │  장르 대분류: 사서류 / 경전류 / 역사류 / 문집류  │
     └─────────────────────────────────────────┘
                       ↓ 분화
     미시적 분석 (K=14/24)
     ┌─────────────────────────────────────────┐
     │  구문 기능: 조건문 / 정의문 / 서사문 / 나열문 등 │
     └─────────────────────────────────────────┘
```

---

## 🛠 분석 파이프라인

### 1단계: 최적 K값 분석 (선택)
```bash
docker exec csp-workspace python scripts/find_optimal_k.py \
    --csv hyeonto/datasets/pa_merged_v2.csv \
    --out-dir hyeonto/reports/optimal_k_pa \
    --k-min 4 --k-max 32 --k-step 2 \
    --device-id 0 --seed 42
```

### 2단계: PA 클러스터링
```bash
# K=4 (거시적)
docker exec csp-workspace python scripts/cluster_pa_boundary_functions.py \
    --input hyeonto/datasets/pa_merged_v2.csv \
    --out-dir hyeonto/reports/pa_boundary_k4_full \
    --k 4 --load-embeddings hyeonto/cache/pa_embeddings.npy \
    --use-src --use-tgt --seed 42

# K=14 (미시적)
docker exec csp-workspace python scripts/cluster_pa_boundary_functions.py \
    --input hyeonto/datasets/pa_merged_v2.csv \
    --out-dir hyeonto/reports/pa_boundary_k14_full \
    --k 14 --load-embeddings hyeonto/cache/pa_embeddings.npy \
    --use-src --use-tgt --seed 42
```

### 3단계: SA 클러스터링
```bash
# K=4 (거시적)
docker exec csp-workspace python scripts/cluster_sa_boundary_functions.py \
    --input hyeonto/datasets/sa_merged_v2.csv \
    --out-dir hyeonto/reports/sa_boundary_k4_full \
    --k 4 --load-embeddings hyeonto/cache/sa_embeddings.npy \
    --use-src --use-tgt --seed 42

# K=24 (미시적)
docker exec csp-workspace python scripts/cluster_sa_boundary_functions.py \
    --input hyeonto/datasets/sa_merged_v2.csv \
    --out-dir hyeonto/reports/sa_boundary_k24_full \
    --k 24 --load-embeddings hyeonto/cache/sa_embeddings.npy \
    --use-src --use-tgt --seed 42
```

### 4단계: 심층 프로파일링 (K=24 SA)
```bash
docker exec csp-workspace python scripts/profile_deep_sa.py \
    --csv hyeonto/reports/sa_boundary_k24_full/sa_boundary_clusters.csv \
    --out-dir hyeonto/reports/sa_boundary_k24_full
```

### 5단계: PA-SA Sankey 시각화
```bash
docker exec csp-workspace python scripts/visualize_pa_sa_sankey.py \
    --pa-csv hyeonto/reports/pa_boundary_k4_full/boundary_clusters.csv \
    --sa-csv hyeonto/reports/sa_boundary_k4_full/sa_boundary_clusters.csv \
    --pa-k 4 --sa-k 4 \
    --out-dir hyeonto/reports/exploratory/pa_sa_sankey
```

---

## 📊 주요 산출물

### 핵심 시각화

| 시각화 | 경로 | 설명 |
|--------|------|------|
| **대시보드** | `reports/dashboard.html` | ⭐ 모든 분석 결과 통합 탐색 |
| **PA-SA Sankey** | `exploratory/pa_sa_sankey/*.html` | PA↔SA 클러스터 흐름 (K=4, K=14, K=24) |
| **클러스터 분화** | `exploratory/cluster_flow_*/` | K=4 → K=14/24 분화 패턴 |
| **고급 산점도** | `*_full/visualization/advanced_cluster_viz.html` | Convex Hull + 사서 추세선 |

### 주요 발견 사항

1. **사서는 문법적 북극점**: PA K=4에서 p1 사서 집중도 **13.4%**, SA K=4에서 p5 **16.5%**
2. **다중 해상도 일관성**: K=4와 K=24 모두 사서 중심성 유지
3. **SA K=24 심층 분석**: 
   - **p5 (Topic)**: 시경(x3.8), 은/는 53% 우세
   - **p22 (시간/장소)**: 춘추좌씨전(x10.0), 에 집중
   - **p17 (시적 표현)**: 당시삼백수(x9.4), 운율적 생략

---

## 📚 관련 문서

| 문서 | 설명 |
|------|------|
| [FINAL_ANALYSIS_REPORT.md](reports/FINAL_ANALYSIS_REPORT.md) | ⭐ v6 통합 마스터 리포트 |
| [KEY_FINDINGS.md](KEY_FINDINGS.md) | 핵심 발견 사항 (Multi-Resolution 포함) |
| [REPRODUCE.md](REPRODUCE.md) | 재현 가이드 |
| [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) | 시각화 해석 |
| [BIAS_VALIDATION.md](BIAS_VALIDATION.md) | 편향 검증 |
| [DATA_PROVENANCE.md](DATA_PROVENANCE.md) | 데이터 출처 |
| [EXPLORATORY_ANALYSIS.md](EXPLORATORY_ANALYSIS.md) | 탐색적 분석 가이드 |

---

**마지막 업데이트**: 2026-01-13 (v6 + Multi-Resolution)
