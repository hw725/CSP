# 현토(懸吐) 분석 프로젝트 (v6)

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
3. **번역문 통합 임베딩**: 한문 원문과 한국어 번역문을 결합하여 화용론적 뉘앙스 포착 (v6)

---

## 📂 디렉토리 구조

```
hyeonto/
├── datasets/                  # 학습용 데이터셋
│   ├── pa_merged_v2.csv       # PA 통합 데이터 (120,202건)
│   └── sa_merged_v2.csv       # SA 통합 데이터 (415,091건)
├── reports/                   # 분석 결과
│   ├── FINAL_ANALYSIS_REPORT.md    # ⭐ 통합 마스터 리포트
│   ├── pa_boundary_v6_full/        # PA 클러스터링 결과 (v6)
│   ├── sa_boundary_v6_full/        # SA 클러스터링 결과 (v6)
│   └── crossmatch_v6/              # PA-SA 교차 분석 (Sankey)
├── BIAS_VALIDATION.md         # 편향 검증 보고서
├── DATA_PROVENANCE.md         # 데이터 출처 설명
├── KEY_FINDINGS.md            # 핵심 발견 사항
├── NEXT_STEPS.md              # 다음 단계 로드맵
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

## 🛠 분석 파이프라인 (v6)

### 1단계: 데이터셋 구축
```bash
python scripts/hyeonto_build_datasets.py
```

### 2단계: PA 클러스터링 (번역문 포함)
```bash
docker exec csp-workspace python scripts/cluster_pa_boundary_functions.py \
    --input hyeonto/datasets/pa_merged_v2.csv \
    --out-dir hyeonto/reports/pa_boundary_v6_full \
    --k 16 --use-src --use-tgt --device-id 0
```

### 3단계: SA 클러스터링 (번역문 포함)
```bash
docker exec csp-workspace python scripts/cluster_sa_boundary_functions.py \
    --input hyeonto/datasets/sa_merged_v2.csv \
    --out-dir hyeonto/reports/sa_boundary_v6_full \
    --k 16 --use-src --use-tgt --device-id 0
```

### 4단계: PA-SA 교차 분석 (Sankey Diagram)
```bash
docker exec csp-workspace python scripts/visualize_pa_sa_sankey_v6_precision.py \
    --pa-clusters hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv \
    --sa-clusters hyeonto/reports/sa_boundary_v6_full/sa_boundary_clusters.csv \
    --out-dir hyeonto/reports/crossmatch_v6
```

---

## 📊 주요 산출물

### 핵심 시각화 (v6)

| 시각화 | 경로 | 설명 |
|--------|------|------|
| **Sankey Diagram** | `crossmatch_v6/pa_sa_sankey_v6_final.html` | ⭐ PA→SA 클러스터 흐름 |
| **Joint Embedding** | `pa_boundary_v6_full/joint_embedding/*.html` | 마커-클러스터 공동 공간 |
| **Marker Heatmap** | `pa_boundary_v6_full/marker_distribution/*.html` | 현토-클러스터 밀집도 |
| **Entropy Chart** | `pa_boundary_v6_full/marker_distribution/marker_entropy_chart.html` | 범용성 지수 |

### 주요 발견 사항

1. **사서는 문법적 북극점**: PA-12 클러스터에서 사서 집중도 **48.8%**
2. **PA→SA 위계적 흐름**: 문장의 문체적 정체성이 구 단위 문법으로 74% 수렴
3. **번역문 효과**: 한문만으로 구분 어려웠던 화용론적 뉘앙스 포착

---

## 💡 활용 방안

### 1. 한문 교육
- 현토 암기 순서 최적화 (빈도/기능별 분류)
- 유사 현토 그룹화로 학습 효율 증대

### 2. NLP 연구
- 한문 번역 모델 개선 (현토 패턴 feature 활용)
- 경계 감지 모델의 데이터 증강

### 3. 인문학 연구
- 장르별 문법 차이 정량화
- 조선시대 현토 체계의 통계적 규명

---

## 📚 관련 문서

| 문서 | 설명 |
|------|------|
| [FINAL_ANALYSIS_REPORT.md](reports/FINAL_ANALYSIS_REPORT.md) | ⭐ v6 통합 마스터 리포트 |
| [KEY_FINDINGS.md](KEY_FINDINGS.md) | 핵심 발견 사항 |
| [REPRODUCE.md](REPRODUCE.md) | 재현 가이드 |
| [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) | 시각화 해석 |
| [BIAS_VALIDATION.md](BIAS_VALIDATION.md) | 편향 검증 |
| [DATA_PROVENANCE.md](DATA_PROVENANCE.md) | 데이터 출처 |
| [NEXT_STEPS.md](NEXT_STEPS.md) | 다음 단계 로드맵 |

---

**마지막 업데이트**: 2026-01-11 (v6)
