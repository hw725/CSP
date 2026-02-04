# Scripts Index

> **경고**: 스크립트를 하위 폴더로 이동하지 마세요. `sys.path` 의존성이 깨집니다.
>
> **최종 업데이트**: 2026-02-04

---

## 🚀 핵심 파이프라인 (스크립트 외)

| 경로 | 설명 |
|------|------|
| `main.py` | CSP 통합 진입점 (XML, TXT, XLSX, CSV 지원) |
| `p2s/main.py` | **P2S 파이프라인** - 문단을 문장으로 분할 |
| `s2p/main.py` | **S2P 파이프라인** - 문장을 구로 분할 및 정렬 |
| `batch_44books.py` | 44권 전체 P2S+S2P 배치 처리 |
| `xlsx_pipeline/xlsx_pipeline_cli.py` | XLSX 파이프라인 CLI |

---

## 🧭 scripts/ 폴더 실제 스크립트

### 🎓 모델 학습 (Training)
| 스크립트 | 설명 |
|---------|------|
| `train_p2s_boundary.py` | P2S 경계 모델 학습 (기본) |
| `train_p2s_crossattn_boundary.py` | P2S Cross-Attention 경계 모델 |
| `train_s2p_alignment_dual_encoder.py` | S2P Dual Encoder 정렬 학습 (출력: `models/dual_encoder_alignment_s2p.pt`) |
| `train_s2p_crossattn_boundary.py` | S2P Cross-Attention 경계 모델 |
| `train_sentence_alignment.py` | 문장 정렬 모델 학습 |

### 📊 평가 및 최적화
| 스크립트 | 설명 |
|---------|------|
| `quick_s2p_eval.py` | S2P 빠른 평가 (샘플 기반) |
| `sweep_p2s_boundary_threshold.py` | P2S 경계값 스윕 (임계값 탐색) |
| `tune_p2s_dp.py` | P2S DP 파라미터 최적화 (Optuna) |
| `optuna_s2p_dp.py` | S2P DP 파라미터 최적화 (Optuna) |
| `aggregate_p2s_drift_summary.py` | P2S 드리프트 요약 집계 |
| `summarize_boundary_delta_patterns.py` | 경계 델타 패턴 요약 |

### 🔍 분석 및 진단
| 스크립트 | 설명 |
|---------|------|
| `analyze_p2s_errors.py` | P2S 오류 분석 (trace 기반) |
| `analyze_failure_patterns.py` | 실패 패턴 분석 |
| `analyze_s2p_mapping.py` | S2P 매핑 분석 |
| `analyze_xlsx_structure.py` | XLSX 구조 분석 |
| `detect_outliers_boundary.py` | 경계 이상치 감지 |
| `diff_p2s_outputs.py` | P2S 출력 비교 |

### 🛠️ 유틸리티 및 데이터 처리
| 스크립트 | 설명 |
|---------|------|
| `build_alignment_dataset.py` | 정렬 데이터셋 구축 (paragraph/sentence/phrase) |
| `merge_parallel_xlsx.py` | 병렬 XLSX 병합 |
| `split_excel.py` | XLSX 분할 |

### 🔐 보안/운영
| 스크립트 | 설명 |
|---------|------|
| `safety_check.ps1` | 취약점 점검 및 자동 수정 |

### 📄 문서
| 문서 | 설명 |
|------|------|
| `CODE_CLEANUP_ANALYSIS.md` | 코드 정리 분석 메모 |
| `CLEANUP_FINAL_REPORT.md` | 정리 최종 보고서 |

---

## 📌 레거시 명칭 (PA/SA)

일부 스크립트/로그/모델 파일명에 **PA/SA** 명칭이 남아 있습니다.
- **PA ≈ P2S**, **SA ≈ S2P**
- 새 파일/설정은 P2S/S2P 명칭을 우선 사용합니다.

---

## 🔗 관련 폴더

- `accuracy/`: 평가 모듈 (`p2s_evaluator.py`, `s2p_evaluator.py` 등)
- `hyeonto/`: 연구/분석 스크립트 및 시각화 파이프라인
- `analytics/`: 모니터링/코퍼스 분석 도구
# Scripts Index

> **경고**: 스크립트를 하위 폴더로 이동하지 마세요. `sys.path` 의존성이 깨집니다.
>
> **최종 업데이트**: 2026-02-03 - 26개 임시/디버그 스크립트 제거 완료 (80개 → 54개)

---

## 🚀 핵심 파이프라인

| 스크립트 | 설명 |
|---------|------|
| `p2s/main.py` | **P2S 파이프라인** - 문단을 문장으로 분할 |
| `s2p/main.py` | **S2P 파이프라인** - 문장을 구로 분할 및 정렬 |
| `main.py` | CSP 통합 진입점 (XML, TXT, XLSX, CSV 지원) |

---

## 🎓 모델 학습 (Training)

| 스크립트 | 설명 |
|---------|------|
| `train_p2s_boundary.py` | P2S 경계 모델 학습 (기본) |
| `train_p2s_crossattn_boundary.py` | P2S Cross-Attention 경계 모델 |
| `train_s2p_alignment_dual_encoder.py` | S2P Dual Encoder 정렬 학습 |
| `train_s2p_crossattn_boundary.py` | S2P Cross-Attention 경계 모델 |
| `train_sentence_alignment.py` | 문장 정렬 모델 학습 |

---

## 📊 평가 및 최적화

| 스크립트 | 설명 |
|---------|------|
| `p2s_multitest_runner.py` | P2S 다중 테스트 실행 (여러 설정 비교) |
| `quick_s2p_eval.py` | S2P 빠른 평가 (샘플 기반) |
| `sweep_p2s_boundary_threshold.py` | P2S 경계값 스윕 (최적 임계값 탐색) |
| `tune_p2s_dp.py` | P2S DP 파라미터 최적화 (Optuna) |
| `optuna_s2p_dp.py` | S2P DP 파라미터 최적화 (Optuna) |

**주의**: `accuracy/` 폴더의 평가 모듈 (`p2s_evaluator.py`, `s2p_evaluator.py`) 사용 권장

---

---

## 🔬 분석 스크립트 (analyze_* / validate_*)

### 언어학적 패턴 분석
| 스크립트 | 설명 |
|---------|------|
| `analyze_tam_v6.py` | TAM(Tense-Aspect-Mood) 분석 |
| `analyze_tense_from_translation_v6.py` | 번역문에서 시제 패턴 분석 |
| `analyze_marker_syntactic_function_v6.py` | 마커 통사 기능 분석 |
| `analyze_hanja_marker_cooccurrence.py` | 한자 마커 공출현 분석 |

### 공출현 및 N-gram 분석
| 스크립트 | 설명 |
|---------|------|
| `analyze_cooccurrence_network.py` | 공출현 네트워크 분석 |
| `analyze_sentence_level_cooccurrence.py` | 문장 수준 공출현 분석 |
| `analyze_ngram_sequences.py` | N-gram 시퀀스 분석 |

### 에러 및 성능 분석
| 스크립트 | 설명 |
|---------|------|
| `analyze_p2s_errors.py` | P2S 오류 분석 (추적 기반) |
| `analyze_failure_patterns.py` | 실패 패턴 분석 |
| `analyze_p2s_s2p_books.py` | P2S/S2P 책별 성능 분석 |
| `analyze_s2p_mapping.py` | S2P 매핑 분석 |
| `analyze_xlsx_structure.py` | XLSX 파일 구조 분석 |

### 가설 검증
| 스크립트 | 설명 |
|---------|------|
| `validate_hyeonto_patterns_v6.py` | Hyeonto 패턴 검증 (통계적) |
| `validate_hypothesis_v6.py` | 일반 가설 검증 |
| `validate_tam_bidirectional_v6.py` | TAM 양방향 검증 |
| `validate_weight_justification.py` | 가중치 정당성 검증 |

---

## 📈 시각화 스크립트 (visualize_*)

| 스크립트 | 설명 |
|---------|------|
| `visualize_clusters_v6.py` | 클러스터 시각화 (v6) |
| `visualize_cluster_flow.py` | 클러스터 플로우 시각화 |
| `visualize_p2s_s2p_sankey.py` | P2S/S2P Sankey 다이어그램 |
| `visualize_marker_parent_overlay.py` | 마커-부모 오버레이 시각화 |
| `visualize_parent_marker_joint_embedding.py` | 부모-마커 공동 임베딩 시각화 |
| `visualize_parent_marker_joint_embedding_ext.py` | 부모-마커 공동 임베딩 (확장) |
| `visualize_parent_situations.py` | 부모 상황 시각화 |

---

## 🔬 클러스터링 및 프로파일링

| 스크립트 | 설명 |
|---------|------|
| `cluster_p2s_boundary_functions.py` | P2S 경계 기능 클러스터링 |
| `cluster_s2p_boundary_functions.py` | S2P 경계 기능 클러스터링 |
| `profile_boundary_clusters.py` | 경계 클러스터 프로파일링 |
| `profile_deep_s2p.py` | S2P 심층 프로파일링 |
| `detect_outliers_boundary.py` | 경계 이상치 감지 |
| `generate_cluster_visualizations.py` | 클러스터 시각화 생성 |

---

## 🛠️ 유틸리티 및 데이터 처리

| 스크립트 | 설명 |
|---------|------|
| `build_alignment_dataset.py` | 정렬 데이터셋 구축 |
| `diff_p2s_outputs.py` | P2S 출력 비교 (여러 버전 간) |
| `prepare_p2s_clusters_for_validation.py` | P2S 클러스터 검증 준비 |
| `prepare_s2p_clusters_for_validation.py` | S2P 클러스터 검증 준비 |
| `classify_syntactic_function.py` | 통사 기능 분류 |

---

## 📊 요약 및 집계

| 스크립트 | 설명 |
|---------|------|
| `aggregate_p2s_drift_summary.py` | P2S 드리프트 요약 집계 |
| `summarize_boundary_delta_patterns.py` | 경계 델타 패턴 요약 |
| `weight_sensitivity_analysis.py` | 가중치 민감도 분석 |

---

## 💾 삭제된 파일들 (2026-02-03)

다음 26개 파일은 일회성 임시 스크립트로 삭제되었습니다:
- `check_*.py`, `debug_*.py`, `verify_*.py` (데이터 확인용 임시)
- `s2p_eval_*.py`, `remap_*.py` (평가 관련 중복 버전)
- `merge_*.py`, `normalize_*.py`, `hyeonto_build_*.py`, `rebuild_*.py` (데이터 처리 일회성)

---

## 📚 관련 모듈

### `accuracy/` 폴더
```
- p2s_evaluator.py      # P2S 평가 (메인)
- s2p_evaluator.py      # S2P 평가 (메인)
- pair_evaluator.py     # 쌍 평가
- compute_thresholds.py # 임계값 계산
```

### `hyeonto/weight_sensitivity/` 폴더
```
- run_validation_analysis.py       # 검증 분석
- recompute_weighted_metrics.py    # Canon3 가중 메트릭 재계산
```

---

## 🔗 참고 문서

- **[SCRIPTS_CLASSIFICATION.md](../SCRIPTS_CLASSIFICATION.md)** - 전체 스크립트 분류 가이드
- **[README.md](../README.md)** - 프로젝트 개요
- **[docs/](../docs/)** - 상세 기술 문서

---

## 🧭 K=3 분석 재현 순서 (권장)

1. **클러스터 시각화**
	- `visualize_clusters_v6.py`
2. **Sankey 다이어그램**
	- `visualize_p2s_s2p_sankey.py`
3. **가중치 민감도**
	- `analyze_weight_sensitivity_v6.py`
4. **서종 분포 요약**
	- `analyze_p2s_s2p_books.py`

> ❗스크립트 파일은 이동하지 마세요. `sys.path` 의존성으로 경로 이동 시 실행이 깨집니다.
