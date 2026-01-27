# 현토(懸吐) 분석 프로젝트 (v6 + K=4 핵심 분석)

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
3. **K=4 핵심 분석**: 거시적 4개 클러스터 분석으로 주요 패턴 규명
4. **정규화**: 171개 규칙으로 옛한글 및 이형태 통일 (Zero-Gap 달성)

---

## 📂 디렉토리 구조

```
hyeonto/
├── datasets/                  # 학습용 데이터셋
│   ├── pa_merged_v2.csv       # PA 통합 데이터 (87,943건)
│   └── sa_merged_v2.csv       # SA 통합 데이터 (294,889건)
├── cache/                     # 임베딩 캐시
│   ├── pa_embeddings.npy      # PA BGE-M3 임베딩
│   └── sa_embeddings.npy      # SA BGE-M3 임베딩
├── reports/                   # 분석 결과
│   ├── dashboard.html         # ⭐ 인터랙티브 대시보드
│   ├── md_viewer.html         # 마크다운 뷰어
│   ├── pa_k4_normalized/      # PA K=4 클러스터링 (정규화)
│   ├── sa_k4_normalized/      # SA K=4 클러스터링 (정규화)
│   ├── k4_embedding_overlay_3d.html  # PA/SA 3D UMAP 시각화
│   ├── k4_embedding_overlay_2d.html  # PA/SA 2D UMAP 시각화
│   ├── sankey_diagrams/       # Sankey 다이어그램
│   ├── tam_analysis_v6/       # TAM 분석
│   ├── tense_analysis_v6/     # 시제 분석
│   ├── weight_sensitivity_v6/ # 가중치 민감도 분석
│   ├── exploratory/           # 탐색적 분석
│   └── archive/               # K=14/K=24 세부 분석 (아카이브)
├── hyeonto_normalizer.py      # 현토 정규화 모듈 (171규칙)
├── generate_sankey_diagrams.py # Sankey 다이어그램 생성
├── analyze_embedding_overlay.py # UMAP 시각화 생성
└── rerun_full_analysis.py     # 전체 분석 재실행
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

## 🔬 K=4 핵심 분석

### 분석 개요

| 데이터 | 건수 | K값 | 정규화 |
|:---:|:---:|:---:|:---:|
| **PA** | 87,943 | K=4 | ✅ Zero-Gap |
| **SA** | 294,889 | K=4 | ✅ Zero-Gap |

### 사서 중심성 (Saseo Centrality)

```
사서(四書)는 현토 패턴의 "문법적 북극점" 역할
Cohen's d = 79.5 (극도의 효과 크기)
```

- PA K=4 p1: 사서 집중도 **13.4%** (평균 10.2% 대비)
- SA K=4 p1: 사서 집중도 **16.5%** (평균 10.2% 대비)

---

## 🛠 분석 파이프라인

### Docker 환경 실행
```bash
# 전체 분석 재실행
docker compose run --rm csp python hyeonto/rerun_full_analysis.py --all

# 정규화 후 클러스터링만
docker compose run --rm csp python hyeonto/rerun_full_analysis.py --step clustering

# UMAP 시각화 생성
docker compose run --rm csp python hyeonto/analyze_embedding_overlay.py

# Sankey 다이어그램 생성
python hyeonto/generate_sankey_diagrams.py
```

### 로컬 대시보드 실행
```bash
cd hyeonto/reports
python -m http.server 8080
# 브라우저에서 http://localhost:8080/dashboard.html 접속
```

---

## 📊 주요 산출물

### 핵심 시각화

| 시각화 | 경로 | 설명 |
|--------|------|------|
| **대시보드** | `reports/dashboard.html` | ⭐ 모든 분석 결과 통합 탐색 |
| **3D UMAP** | `reports/k4_embedding_overlay_3d.html` | PA/SA 임베딩 3D 오버레이 |
| **2D UMAP** | `reports/k4_embedding_overlay_2d.html` | PA/SA 임베딩 2D 오버레이 |
| **PA↔SA Sankey** | `reports/sankey_diagrams/sankey_pa4_sa4.html` | PA-SA 클러스터 흐름 |

### K=4 클러스터 프로파일

| 파일 | 설명 |
|------|------|
| `pa_k4_normalized/pa_cluster_profile.md` | PA K=4 정규화 프로파일 |
| `sa_k4_normalized/sa_cluster_profile.md` | SA K=4 정규화 프로파일 |

---

## 📚 관련 문서

| 문서 | 설명 |
|------|------|
| [dashboard.html](reports/dashboard.html) | ⭐ 인터랙티브 대시보드 |
| [FINAL_ANALYSIS_REPORT.md](reports/FINAL_ANALYSIS_REPORT.md) | v6 통합 마스터 리포트 |
| [KEY_FINDINGS.md](KEY_FINDINGS.md) | 핵심 발견 사항 |
| [REPRODUCE.md](REPRODUCE.md) | 재현 가이드 |
| [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) | 시각화 해석 |
| [BIAS_VALIDATION.md](BIAS_VALIDATION.md) | 편향 검증 |

---

**마지막 업데이트**: 2026-01-27 (K=4 핵심 분석 + 정규화 + 시각화 개선)
