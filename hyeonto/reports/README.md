# 📂 hyeonto/reports/ 디렉토리 구조

본 디렉토리에는 **현토 분석 결과물**이 저장됩니다.

---

## 📜 핵심 파일

| 파일 | 설명 |
|:---|:---|
| **dashboard.html** | ⭐ **통합 분석 대시보드** - 모든 결과 인터랙티브 탐색 |
| **FINAL_ANALYSIS_REPORT.md** | ⭐ **통합 마스터 리포트 (v6)** - Multi-Resolution 분석 포함 |

---

## 📁 Multi-Resolution 분석 결과 (신규)

### `pa_boundary_k4_full/` & `sa_boundary_k4_full/`
- 거시적(Macro) 관점의 4대 장르 대분류 클러스터링 결과
- 주요 산출물: `pa_cluster_profile.md`, `sa_cluster_profile.md`

### `pa_boundary_k14_full/` & `sa_boundary_k24_full/`
- 미시적(Micro) 관점의 기능별 세부 클러스터링 결과
- **[sa_deep_profile.md](sa_boundary_k24_full/sa_deep_profile.md)**: SA K=24 심층 분석 (Lift, Entropy, Syntactic Guess)

### `optimal_k_pa/` & `optimal_k_sa_v3/`
- 최적 K값 탐색 결과 (Elbow Method, Silhouette Score 등)

---

## 📁 Exploratory & Cross-Analysis

### `exploratory/`
탐색적 분석 및 클러스터 분화 시각화
- `pa_sa_sankey/`: PA ↔ SA 다중 해상도 Sankey 다이어그램
- `cluster_flow_pa/`: PA K=4 → K=14 분화 흐름
- `cluster_flow_sa/`: SA K=4 → K=24 분화 흐름
- `outliers_pa/`, `ngram_sa/`: 이상치 및 N-gram 분석 결과

### `crossmatch_v6/`
PA(v6) ↔ SA(v6) 교차 분석
- `pa_sa_sankey_v6_final.html`: 표준 v6 모델 기반 흐름도

### `weight_sensitivity_v6/`
가중치 민감도 분석 및 수치 정당성 검토
- **[WEIGHT_RATIO_JUSTIFICATION.md](weight_sensitivity_v6/WEIGHT_RATIO_JUSTIFICATION.md)**: 7:4:1:1 가중치 도출 상세 과정
- `WEIGHT_SENSITIVITY_REPORT.md`: 민감도 분석 요약 (Strong vs Uniform 등)
- `weight_grid_search.csv`: 122개 가중치 조합 테스트 로우 데이터

---

## 📁 V6 표준 분석 결과 (K=16 Baseline)

### `pa_boundary_v6_full/`
PA(문장 단위) 표준 분석 (87,943건)
- `boundary_clusters.csv`: 원본 데이터
- `visualization/`: 고급 산점도 (Convex Hull)
- `joint_embedding/`: 마커-클러스터 매핑
- `marker_distribution/`: 히트맵 및 엔트로피

### `sa_boundary_v6_full/`
SA(구 단위) 표준 분석 (294,889건)
- `sa_boundary_clusters.csv`: 원본 데이터
- `visualization/`: 고급 산점도
- `sa_cluster_profile.md`: 표준 프로파일

---

## 📦 archive/

이전 버전 분석 결과물 보관 (v5 이하).
- `clusters_raw_data/`
- `pa_joint_embedding/`
- `sa_marker_distribution/`
- 기타 레거시 보고서

---

## 🔄 분석 버전 이력

| 버전 | 날짜 | 주요 변경 |
|:---:|:---:|:---|
| **v6+Multi** | 2026-01-13 | 다중 해상도(K=4/14/24) 분석 및 심층 프로파일링 추가 |
| **v6** | 2026-01-11 | 번역문(tgt) 포함 임베딩으로 전면 재분석 |
| v5 | 2026-01-10 | 마스터 리포트 통합 |
| v4 | - | 의미역 분석 추가 |

---

**최종 업데이트**: 2026-01-13
