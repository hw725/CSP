# Scripts 폴더 사용성 감시 보고서

**생성 일시**: 2026-02-04  
**목적**: scripts 폴더의 모든 스크립트가 실제 사용되는지 전수조사

---

## 📋 Scripts 폴더 스크립트 목록 (20개)

### 🎓 모델 학습 (5개)
| 스크립트 | 상태 | 참조 | 사용 경로 |
|---------|------|------|----------|
| `train_p2s_boundary.py` | ✅ Active | YES | 직접 실행, subprocess (train_sentence_alignment) |
| `train_p2s_crossattn_boundary.py` | ✅ Active | YES | 직접 실행 가능 |
| `train_s2p_alignment_dual_encoder.py` | ✅ Active | YES | 직접 실행 가능 |
| `train_s2p_crossattn_boundary.py` | ✅ Active | YES | 직접 실행 가능 |
| `train_sentence_alignment.py` | ✅ Active | YES | subprocess (train_p2s_boundary에서 호출) |

### 📊 평가 및 최적화 (6개)
| 스크립트 | 상태 | 참조 | 사용 경로 |
|---------|------|------|----------|
| `quick_s2p_eval.py` | ✅ Active | YES | 직접 실행 가능 |
| `sweep_p2s_boundary_threshold.py` | ✅ Active | YES | 직접 실행, subprocess 호출 가능 |
| `tune_p2s_dp.py` | ✅ Active | YES | 직접 실행 가능 |
| `optuna_s2p_dp.py` | ✅ Active | YES | 직접 실행 가능 |
| `aggregate_p2s_drift_summary.py` | ✅ Active | YES | 직접 실행 가능 |
| `summarize_boundary_delta_patterns.py` | ✅ Active | YES | 직접 실행 가능 |

### 🔍 분석 및 진단 (6개)
| 스크립트 | 상태 | 특징 | CLI 인자 | 사용 여부 |
|---------|------|------|---------|----------|
| `analyze_p2s_errors.py` | ✅ Active | trace 기반 오류 분석 | CLI 인자 있음 | YES - 직접 실행 |
| `analyze_failure_patterns.py` | ⚠️ Semi | 실패 케이스 패턴 분석 | 하드코딩된 경로 | 일회성 스크립트 |
| `analyze_s2p_mapping.py` | ⚠️ Semi | S2P 입력/Gold 매칭 분석 | 하드코딩된 경로 | 일회성 스크립트 |
| `analyze_xlsx_structure.py` | ⚠️ Semi | XLSX 구조 분석 | 하드코딩된 경로 | 일회성 스크립트 |
| `detect_outliers_boundary.py` | ⚠️ Semi | 클러스터 이상치 탐지 | CLI 인자 있음 | 가능하지만 미사용 |
| `diff_p2s_outputs.py` | ⚠️ Semi | P2S 출력 비교/Diff | CLI 인자 있음 | 가능하지만 미사용 |

### 🛠️ 유틸리티 및 데이터 처리 (3개)
| 스크립트 | 상태 | 특징 | 사용 여부 |
|---------|------|------|----------|
| `build_alignment_dataset.py` | ⚠️ Semi | 정렬 데이터셋 구축 (training 전 준비) | 일회성 또는 필요시 |
| `merge_parallel_xlsx.py` | ⚠️ Semi | 병렬 XLSX 파일 병합 | 일회성 데이터 준비 |
| `split_excel.py` | ⚠️ Semi | XLSX 데이터 train/val/test 분할 | 일회성 데이터 준비 |

---

## 🔍 확인 결과

### 확실히 사용되는 스크립트 (11개)
1. ✅ `train_p2s_boundary.py` - P2S 경계 모델 학습 (core)
2. ✅ `train_p2s_crossattn_boundary.py` - P2S Cross-Attention (core)
3. ✅ `train_s2p_alignment_dual_encoder.py` - S2P 정렬 (core)
4. ✅ `train_s2p_crossattn_boundary.py` - S2P Cross-Attention (core)
5. ✅ `train_sentence_alignment.py` - 문장 정렬 (core)
6. ✅ `quick_s2p_eval.py` - S2P 빠른 평가 (evaluation)
7. ✅ `sweep_p2s_boundary_threshold.py` - 경계값 스윕 (tuning)
8. ✅ `tune_p2s_dp.py` - DP 파라미터 최적화 (tuning)
9. ✅ `optuna_s2p_dp.py` - S2P DP 최적화 (tuning)
10. ✅ `aggregate_p2s_drift_summary.py` - 드리프트 요약 (analysis)
11. ✅ `summarize_boundary_delta_patterns.py` - 경계 델타 요약 (analysis)

### 의심스러운 스크립트 (9개 - 확인 필요)
| 스크립트 | 추정 상태 | 확인 필요 |
|---------|---------|----------|
| `analyze_failure_patterns.py` | 미사용? | 코드 참조, 실행 경로 없음 |
| `analyze_s2p_mapping.py` | 미사용? | 코드 참조, 실행 경로 없음 |
| `analyze_xlsx_structure.py` | 미사용? | 코드 참조, 실행 경로 없음 |
| `detect_outliers_boundary.py` | 미사용? | 코드 참조, 실행 경로 없음 |
| `diff_p2s_outputs.py` | 미사용? | 코드 참조, 실행 경로 없음 |
| `build_alignment_dataset.py` | 미사용? | 코드 참조, 실행 경로 없음 |
| `merge_parallel_xlsx.py` | 미사용? | 코드 참조, 실행 경로 없음 |
| `split_excel.py` | 미사용? | 코드 참조, 실행 경로 없음 |

---

## 📋 상세 조사 결과

### 확인된 실행 경로
- `train_p2s_boundary.py`: 
  - 직접 실행: `python scripts/train_p2s_boundary.py`
  - subprocess: `train_sentence_alignment.py`를 subprocess로 호출

- `sweep_p2s_boundary_threshold.py`:
  - 직접 실행: `python scripts/sweep_p2s_boundary_threshold.py`

- `analyze_p2s_errors.py`:
  - 직접 실행: `python scripts/analyze_p2s_errors.py`

### 문서에서만 언급되는 스크립트
- `analyze_failure_patterns.py` (README.md, SCRIPTS_CLASSIFICATION.md)
- `analyze_s2p_mapping.py` (README.md, SCRIPTS_CLASSIFICATION.md)
- `analyze_xlsx_structure.py` (README.md, SCRIPTS_CLASSIFICATION.md)
- `detect_outliers_boundary.py` (README.md, SCRIPTS_CLASSIFICATION.md)
- `diff_p2s_outputs.py` (README.md, SCRIPTS_CLASSIFICATION.md)
- `build_alignment_dataset.py` (README.md, SCRIPTS_CLASSIFICATION.md)
- `merge_parallel_xlsx.py` (README.md, SCRIPTS_CLASSIFICATION.md)
- `split_excel.py` (README.md, SCRIPTS_CLASSIFICATION.md)

---

## ⚠️ 의심스러운 점

1. **분석 스크립트 미사용**: 
   - `analyze_failure_patterns.py`, `analyze_s2p_mapping.py`, `analyze_xlsx_structure.py` 등이 문서에는 있지만 코드에서 호출되지 않음
   - 실행 가능한지 확인 필요

2. **유틸리티 스크립트 미사용**:
   - `build_alignment_dataset.py`, `merge_parallel_xlsx.py`, `split_excel.py` 등이 문서에는 있지만 호출 경로 없음

3. **레거시 naming**:
   - 일부 스크립트가 PA/SA 명칭 사용 (정규화 필요)

---

## 📌 다음 단계

1. **의심 스크립트 실행 테스트** - 각 스크립트가 정상 동작하는지 확인
2. **코드 참조 확인** - 각 스크립트를 import하는 코드 확인
3. **삭제 대상 결정** - 미사용 스크립트 정리
4. **문서 업데이트** - README.md, SCRIPTS_CLASSIFICATION.md 최신화

---

## 관련 폴더

- `hyeonto/`: 추가 분석 스크립트 (45개 이상)
- `accuracy/`: 평가 모듈 (core)
- `analytics/`: 모니터링/분석 도구 (core)
