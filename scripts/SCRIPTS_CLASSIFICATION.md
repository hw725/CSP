# 📚 Scripts 폴더 분류 가이드

**마지막 업데이트: 2026-02-03**

> 이 문서는 `scripts/` 폴더의 54개 Python 스크립트를 목적별로 분류합니다.

---

## 🎯 카테고리별 분류

### 1️⃣ **핵심 파이프라인** (3개)
메인 처리 흐름의 진입점

```
- main.py                           # CSP 통합 진입점 (XML, TXT, XLSX, CSV 지원)
- p2s/main.py                       # P2S 파이프라인 (문단→문장)
- s2p/main.py                       # S2P 파이프라인 (문장→구)
```

---

### 2️⃣ **모델 학습** (5개)
경계 감지 및 정렬 모델 학습

```
- train_p2s_boundary.py             # P2S 경계 모델 학습 (기본)
- train_p2s_crossattn_boundary.py   # P2S Cross-Attention 경계 모델
- train_s2p_alignment_dual_encoder.py  # S2P Dual Encoder 정렬 학습
- train_s2p_crossattn_boundary.py   # S2P Cross-Attention 경계 모델
- train_sentence_alignment.py       # 문장 정렬 모델 학습
```

---

### 3️⃣ **평가 및 검증** (4개)
모델 성능 평가 및 데이터 무결성 검증

```
- p2s_multitest_runner.py           # P2S 다중 테스트 실행 (여러 설정 비교)
- quick_s2p_eval.py                 # S2P 빠른 평가 (샘플 기반)
- tune_p2s_dp.py                    # P2S Dynamic Programming 파라미터 최적화 (Optuna)
- optuna_s2p_dp.py                  # S2P DP 파라미터 최적화 (Optuna)

추가: accuracy/ 폴더의 메인 평가 모듈
  - accuracy/p2s_evaluator.py       # P2S 정확도 평가
  - accuracy/s2p_evaluator.py       # S2P 정확도 평가
  - accuracy/pair_evaluator.py      # 쌍 단위 평가
```

---

### 4️⃣ **파라미터 최적화** (2개)
모델 성능 최적화

```
- sweep_p2s_boundary_threshold.py   # P2S 경계값 스윕 (최적 임계값 탐색)
- summarize_boundary_delta_patterns.py  # 경계 변화 패턴 요약
```

---

### 5️⃣ **분석 스크립트** (14개)
언어학적 패턴, 에러, 데이터 분석

#### 5-1. **최신 분석** (v6 버전)
```
- analyze_tam_v6.py                 # TAM(Tense-Aspect-Mood) 분석
- analyze_tense_from_translation_v6.py  # 번역문에서 시제 분석
- analyze_weight_sensitivity_v6.py  # 가중치 민감도 분석
- analyze_marker_syntactic_function_v6.py  # 마커 통사 기능 분석
```

#### 5-2. **패턴 및 공출현 분석**
```
- analyze_cooccurrence_network.py   # 공출현 네트워크 분석
- analyze_sentence_level_cooccurrence.py  # 문장 수준 공출현 분석
- analyze_ngram_sequences.py        # N-gram 시퀀스 분석
- analyze_hanja_marker_cooccurrence.py  # 한자 마커 공출현 분석
```

#### 5-3. **에러 및 실패 분석**
```
- analyze_p2s_errors.py             # P2S 오류 분석 (추적 기반)
- analyze_failure_patterns.py       # 실패 패턴 분석
- analyze_p2s_s2p_books.py          # P2S/S2P 책별 분석
- analyze_s2p_mapping.py            # S2P 매핑 분석
```

#### 5-4. **구조 분석**
```
- analyze_xlsx_structure.py         # XLSX 파일 구조 분석
```

---

### 6️⃣ **검증 스크립트** (4개)
가설 및 패턴 검증

```
- validate_hyeonto_patterns_v6.py   # Hyeonto 패턴 검증 (통계적)
- validate_hypothesis_v6.py         # 일반 가설 검증
- validate_tam_bidirectional_v6.py  # TAM 양방향 검증
- validate_weight_justification.py  # 가중치 정당성 검증
```

---

### 7️⃣ **시각화 스크립트** (7개)
데이터 및 분석 결과 시각화

```
- visualize_clusters_v6.py          # 클러스터 시각화 (v6)
- visualize_cluster_flow.py         # 클러스터 플로우 시각화
- visualize_marker_parent_overlay.py  # 마커-부모 오버레이 시각화
- visualize_p2s_s2p_sankey.py       # P2S/S2P Sankey 다이어그램
- visualize_parent_marker_joint_embedding.py  # 부모-마커 공동 임베딩
- visualize_parent_marker_joint_embedding_ext.py  # 부모-마커 공동 임베딩 (확장)
- visualize_parent_situations.py    # 부모 상황 시각화
```

---

### 8️⃣ **클러스터링 및 프로파일링** (5개)
경계 기능 클러스터링 및 분석

```
- cluster_p2s_boundary_functions.py  # P2S 경계 기능 클러스터링
- cluster_s2p_boundary_functions.py  # S2P 경계 기능 클러스터링
- profile_boundary_clusters.py       # 경계 클러스터 프로파일링
- profile_deep_s2p.py               # S2P 심층 프로파일링
- detect_outliers_boundary.py       # 경계 이상치 감지
```

---

### 9️⃣ **유틸리티 및 도구** (6개)
데이터 처리, 비교, 생성

```
- diff_p2s_outputs.py               # P2S 출력 비교 (여러 버전 간)
- prepare_p2s_clusters_for_validation.py  # P2S 클러스터 검증용 준비
- prepare_s2p_clusters_for_validation.py  # S2P 클러스터 검증용 준비
- generate_cluster_visualizations.py  # 클러스터 시각화 생성
- classify_syntactic_function.py    # 통사 기능 분류
- build_alignment_dataset.py        # 정렬 데이터셋 구축
```

---

### 🔟 **기타 분석** (2개)
특수 목적 분석

```
- aggregate_p2s_drift_summary.py    # P2S 드리프트 요약 집계
- weight_sensitivity_analysis.py    # 가중치 민감도 분석
```

---

### 📋 **다른 폴더의 관련 스크립트**

#### `accuracy/` 폴더
```
- p2s_evaluator.py          # P2S 평가 (메인)
- s2p_evaluator.py          # S2P 평가 (메인)
- pair_evaluator.py         # 쌍 평가
- compute_thresholds.py     # 임계값 계산
- thresholds_config.py      # 임계값 설정
```

#### `hyeonto/` 폴더
```
- run_retrospective_validation.py  # 회상상 가설 검증
- run_validation_analysis.py       # 검증 분석
```

#### `analytics/` 폴더
```
- corpus_analyzer.py        # 코퍼스 분석
- cluster_visualizer.py     # 클러스터 시각화
- book_metadata_extractor.py  # 책 메타데이터 추출
- monitoring_dashboard.py   # 모니터링 대시보드
```

---

## 📊 사용 빈도별 분류

### 자주 사용 (Daily)
```
✨ main.py
✨ p2s/main.py
✨ s2p/main.py
✨ train_p2s_boundary.py
✨ train_s2p_alignment_dual_encoder.py
```

### 정기적 사용 (Weekly)
```
📊 p2s_multitest_runner.py
📊 quick_s2p_eval.py
📊 sweep_p2s_boundary_threshold.py
📊 accuracy/* 평가 스크립트
```

### 특수 목적 (Monthly/Research)
```
🔬 analyze_*_v6.py
🔬 validate_*_v6.py
🔬 visualize_*.py
🔬 cluster_*.py
```

---

## 🗂️ 추천 워크플로우

### 1️⃣ 파이프라인 실행
```bash
# P2S 실행
python scripts/p2s/main.py <input.csv> <output.xlsx>

# S2P 실행
python scripts/s2p/main.py <input.csv> <output.xlsx>
```

### 2️⃣ 모델 학습 & 최적화
```bash
# 경계 모델 학습
python scripts/train_p2s_boundary.py ...

# 파라미터 최적화
python scripts/sweep_p2s_boundary_threshold.py ...
python scripts/tune_p2s_dp.py ...
```

### 3️⃣ 평가 및 검증
```bash
# 빠른 평가
python scripts/quick_s2p_eval.py ...

# 다중 테스트
python scripts/p2s_multitest_runner.py ...

# 상세 분석
python scripts/analyze_p2s_errors.py ...
```

### 4️⃣ 패턴 분석 (연구용)
```bash
# 통계 검증
python scripts/validate_hypothesis_v6.py ...
python scripts/validate_tam_bidirectional_v6.py ...

# 시각화
python scripts/visualize_clusters_v6.py ...
```

---

## 🔍 찾기 팁

### 특정 작업별 스크립트 찾기

| 원하는 작업 | 스크립트 |
|----------|---------|
| P2S 학습 | `train_p2s_boundary.py`, `train_p2s_crossattn_boundary.py` |
| S2P 학습 | `train_s2p_alignment_dual_encoder.py`, `train_s2p_crossattn_boundary.py` |
| 성능 평가 | `p2s_multitest_runner.py`, `quick_s2p_eval.py` |
| 파라미터 최적화 | `tune_p2s_dp.py`, `sweep_p2s_boundary_threshold.py` |
| 에러 분석 | `analyze_p2s_errors.py`, `analyze_failure_patterns.py` |
| 패턴 검증 | `validate_hypothesis_v6.py`, `validate_tam_bidirectional_v6.py` |
| 시각화 | `visualize_clusters_v6.py`, `visualize_p2s_s2p_sankey.py` |

---

## ✅ 정리 현황

### 2026-02-03 대정리
- ✅ 26개 임시/디버그 스크립트 제거
- ✅ 80개 → 54개로 감소 (32.5% 감소)
- ✅ 모든 파일을 목적별 카테고리로 분류

### 제거된 파일들
```
check_mapping_chain.py, check_s2p_input.py, check_sent1_mapping.py,
debug_sent76_v2.py, debug_src_mismatch.py,
verify_gold.py, verify_gold_v2.py, verify_s2p_integrity.py, verify_s2p_integrity_v2.py,
compare_sent76.py, compare_source_data.py, create_correct_split_with_book.py,
s2p_eval_final.py, s2p_eval_final_v3.py, s2p_eval_gold_v2.py, s2p_eval_text_match.py, s2p_eval_v2_limited.py,
merge_original_p2s_s2p.py, merge_p2s_s2p.py, collect_phrase_gold.py, normalize_book_names.py,
hyeonto_build_datasets.py, hyeonto_build_xlsx.py, rebuild_phrase_gold_v2.py, rebuild_all_reports_v6.py,
remap_s2p_output.py
```

---

## 📚 참고 문서

- [README.md](./README.md) - 프로젝트 개요
- [scripts/README.md](./scripts/README.md) - 기존 스크립트 문서
- [docs/](./docs/) - 상세 문서

