# 📂 hyeonto/reports/ 디렉토리 구조

본 디렉토리에는 **현토 분석 결과물**이 저장됩니다.

---

## 📜 핵심 파일

| 파일 | 설명 |
|:---|:---|
| **FINAL_ANALYSIS_REPORT.md** | ⭐ **통합 마스터 리포트 (v6)** - 번역문 포함 재분석 결과 |

---

## 📁 V6 분석 결과 (번역문 포함)

### `pa_boundary_v6_full/`
PA(문장 단위) 경계 클러스터링 결과 (87,943건)

| 파일 | 설명 |
|:---|:---|
| `boundary_clusters.csv` | 클러스터링 원본 데이터 (44MB) |
| `boundary_clusters.md` | 클러스터별 샘플 |
| `boundary_clusters_labeled.md` | 휴리스틱 라벨링 |
| `pa_cluster_profile.md` | 상세 프로파일 (사서 비율, 마커, 한자) |
| `visualization/` | 클러스터 기본 시각화 |
| `joint_embedding/` | ⭐ 마커-클러스터 공동 공간 매핑 |
| `marker_distribution/` | 히트맵 및 엔트로피 분석 |
| `syntactic_analysis/` | 통사 기능 및 다의성 분석 |

### `sa_boundary_v6_full/`
SA(구 단위) 경계 클러스터링 결과 (294,889건)

| 파일 | 설명 |
|:---|:---|
| `sa_boundary_clusters.csv` | 클러스터링 원본 데이터 (51MB) |
| `sa_boundary_clusters.md` | 클러스터별 샘플 |
| `sa_cluster_profile.md` | 상세 프로파일 |
| `visualization/` | 클러스터 기본 시각화 |
| `joint_embedding/` | 마커-클러스터 공동 공간 매핑 |
| `marker_distribution/` | 히트맵 및 엔트로피 분석 |

### `crossmatch_v6/`
PA와 SA의 교차 분석 (위계적 구조)

| 파일 | 설명 |
|:---|:---|
| `pa_sa_sankey_v6_final.html` | ⭐ **핵심 시각화: PA → SA 클러스터 흐름** |
| `pa_sa_flow_stats_v6.csv` | 클러스터 전이 통계 |

---

## 📦 archive/

이전 버전 분석 결과물 보관. v5 이전 데이터가 여기에 있습니다.

- `clusters_raw_data/` - 이전 클러스터 원본 데이터
- `pa_joint_embedding/`, `sa_joint_embedding/` - 이전 시각화
- `pa_marker_distribution/`, `sa_marker_distribution/` - 이전 마커 분포
- `semantic_role_analysis/` - 이전 의미역 분석
- `syntactic_analysis_v2/` - 이전 통사 분석
- `pa_sa_crossmatch/` - 이전 교차 분석
- 각종 개별 보고서 (.md 파일들)

---

## 🔄 분석 버전 이력

| 버전 | 날짜 | 주요 변경 |
|:---:|:---:|:---|
| **v6** | 2026-01-11 | 번역문(tgt) 포함 임베딩으로 전면 재분석 |
| v5 | 2026-01-10 | 마스터 리포트 통합 |
| v4 | - | 의미역 분석 추가 |
| v3 | - | PA/SA 분리 분석 |

---

**최종 업데이트**: 2026-01-11
