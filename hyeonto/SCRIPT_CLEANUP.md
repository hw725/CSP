# Hyeonto 프로젝트 스크립트 정리 (v6)

**작성일**: 2026-01-12
**목적**: 불필요한 임시 코드 삭제 및 필수 코드 커밋 안내

---

## 📁 커밋 필수 스크립트 (KEEP)

아래 스크립트들은 v6 분석 파이프라인의 핵심이며, 반드시 커밋되어야 합니다.

### 핵심 분석 스크립트 (v6)

| 스크립트 | 용도 | 비고 |
|:---|:---|:---|
| `analyze_tam_v6.py` | 현토 TAM(시제-상-서법) 분석 | ✅ 필수 |
| `analyze_tense_from_translation_v6.py` | 번역문 시제 패턴 분석 | ✅ 필수 |
| `analyze_marker_syntactic_function_v6.py` | 현토 통사 기능 분석 | ✅ 필수 |
| `validate_hypothesis_v6.py` | 편향 검증 (영가설/역가중치/대립가설) | ✅ 필수 |
| `validate_tam_bidirectional_v6.py` | TAM 양방향 검증 | ✅ 필수 |
| `validate_hyeonto_patterns_v6.py` | 현토 패턴별 체계적 가설 검증 | ✅ 필수 |
| `rebuild_all_reports_v6.py` | v6 보고서 재생성 | ✅ 필수 |

### 클러스터링 및 시각화

| 스크립트 | 용도 | 비고 |
|:---|:---|:---|
| `visualize_pa_sa_sankey_v6.py` | Sankey 다이어그램 생성 | ✅ 필수 |
| `visualize_pa_sa_sankey_v6_precision.py` | 정밀 Sankey 변형 | ✅ 필수 |
| `visualize_clusters_v6.py` | 클러스터 시각화 | ✅ 필수 |
| `profile_boundary_clusters.py` | 경계 클러스터 프로파일 | ✅ 필수 |
| `profile_clusters.py` | 클러스터 프로파일 | ✅ 필수 |
| `describe_boundary_clusters.py` | 경계 클러스터 기술 | ✅ 필수 |

### 데이터 전처리

| 스크립트 | 용도 | 비고 |
|:---|:---|:---|
| `hyeonto_build_datasets.py` | 데이터셋 구축 | ✅ 필수 |
| `hyeonto_build_xlsx.py` | Excel 변환 | ✅ 필수 |
| `merge_pa_sa.py` | PA/SA 병합 | ✅ 필수 |
| `normalize_book_names.py` | 도서명 정규화 | ✅ 필수 |
| `extract_markers_from_merged.py` | 현토 마커 추출 | ✅ 필수 |

### 검증 및 유효성 확인

| 스크립트 | 용도 | 비고 |
|:---|:---|:---|
| `verify_pa_sa_alignment.py` | PA/SA 정렬 검증 | ✅ 필수 |
| `verify_pa_sa_text_match.py` | 텍스트 일치 검증 | ✅ 필수 |
| `check_hyeonto_markers.py` | 현토 마커 확인 | ✅ 필수 |
| `check_pa_sa_data.py` | PA/SA 데이터 확인 | ✅ 필수 |

---

## 🗑️ 삭제 권장 스크립트 (DELETE)

아래 스크립트들은 임시/디버깅 용도이며, 삭제를 권장합니다.

### temp 폴더 전체

| 경로 | 파일 | 사유 |
|:---|:---|:---|
| `temp/analyze_cluster_entropy.py` | 임시 분석 | 삭제 권장 |
| `temp/check_data_structure.py` | 데이터 구조 확인용 | 삭제 권장 |
| `temp/optimize_weights.py` | 가중치 최적화 실험 | 삭제 권장 |

### 루트 temp 스크립트

| 스크립트 | 사유 |
|:---|:---|
| `temp_merge_datasets.py` | 임시 병합 스크립트 |
| `temp_check_results.py` | 임시 결과 확인 |

### 구버전 (non-v6) 분석 스크립트

| 스크립트 | 사유 | 대체 |
|:---|:---|:---|
| `analyze_marker_syntactic_function.py` | v6로 대체됨 | `analyze_marker_syntactic_function_v6.py` |
| `analyze_tense_morphemes.py` | v6로 대체됨 | `analyze_tense_from_translation_v6.py` |
| `validate_random_labels.py` | v6로 대체됨 | `validate_hypothesis_v6.py` |
| `validate_pa_cluster11_saseo_centrality.py` | v5용 (cluster 11) | `validate_hypothesis_v6.py` (cluster 12) |

### 디버깅 스크립트

| 스크립트 | 사유 |
|:---|:---|
| `debug_pa_sa_mismatch.py` | 디버깅 완료 |
| `debug_pa_boundary_diff.py` | 디버깅 완료 |
| `debug_pa_one_pid_verbose.py` | 디버깅 완료 |
| `debug_pa_tgt_mismatch.py` | 디버깅 완료 |
| `debug_pa_vs_gold_mismatches.py` | 디버깅 완료 |
| `debug_boundary_mismatches_tgt_exact.py` | 디버깅 완료 |

---

## 📋 삭제/유지 요약

| 분류 | 개수 | 조치 |
|:---|:---:|:---|
| **핵심 v6 스크립트** | 7 | ✅ 유지 및 커밋 |
| **시각화/프로파일** | 6 | ✅ 유지 및 커밋 |
| **전처리** | 5 | ✅ 유지 및 커밋 |
| **검증** | 4 | ✅ 유지 및 커밋 |
| **temp 폴더** | 3 | 🗑️ 삭제 권장 |
| **temp 루트** | 2 | 🗑️ 삭제 권장 |
| **구버전** | 4 | 🗑️ 삭제 권장 |
| **디버깅** | 6 | 🗑️ 삭제 권장 |

---

## 🚀 커밋 명령어 예시

```bash
# 1. temp 폴더 삭제
rm -rf scripts/temp/

# 2. 삭제 권장 스크립트 제거
rm scripts/temp_*.py
rm scripts/debug_*.py
rm scripts/analyze_marker_syntactic_function.py
rm scripts/analyze_tense_morphemes.py
rm scripts/validate_random_labels.py
rm scripts/validate_pa_cluster11_saseo_centrality.py

# 3. 커밋
git add scripts/*_v6.py scripts/hyeonto_*.py scripts/merge_*.py scripts/profile_*.py scripts/visualize_*.py
git commit -m "chore: hyeonto v6 스크립트 정리 및 임시 파일 삭제"
```

---

**참고**: 삭제 전에 archive/ 폴더로 백업을 권장합니다.
