# 코드 정리 분석 보고서
**작성일: 2026-02-03**

## 📊 전체 현황

### Scripts 폴더 분석
- **총 파일 수**: 80개 Python 스크립트 + 1개 PowerShell 스크립트
- **분류 결과**: 
  - ✅ 생산(Production) 코드: ~15개
  - 🧪 테스트/분석 코드: ~35개
  - 🗑️ **임시/디버그 코드: ~30개** ← 삭제 대상

---

## 🗑️ 삭제 대상 코드 (휴지통 이동)

### 1️⃣ **Direct 데이터 확인용 임시 스크립트** (12개)
이들은 특정 데이터 상태를 확인하는 일회성 스크립트입니다.

```
scripts/check_mapping_chain.py          # 매핑 체인 일회성 확인
scripts/check_s2p_input.py              # S2P 입력 데이터 일회성 확인
scripts/check_sent1_mapping.py          # 문장 1 매핑 상세 확인
scripts/debug_sent76_v2.py              # 문장 76 상세 비교
scripts/debug_src_mismatch.py           # 원문 비교 분석 (임시)
scripts/verify_gold.py                  # Gold 검증 v1
scripts/verify_gold_v2.py               # Gold 검증 v2
scripts/verify_s2p_integrity.py         # S2P 무결성 검증 v1
scripts/verify_s2p_integrity_v2.py      # S2P 무결성 검증 v2
scripts/compare_sent76.py               # 문장 76 비교
scripts/compare_source_data.py          # 원본 데이터 비교
scripts/create_correct_split_with_book.py  # 올바른 분할 생성 (완료됨)
```

**이유**: 데이터 검증 목적으로 임시 작성, 현재는 불필요

---

### 2️⃣ **평가 스크립트 - 중복/구 버전** (8개)
S2P, P2S 평가 관련 여러 버전이 존재합니다. 최신 버전만 유지하고 구 버전 삭제.

```
scripts/s2p_eval_final.py               # 최종 평가 (구 버전)
scripts/s2p_eval_final_v3.py            # 최종 평가 v3 (중복 가능성)
scripts/s2p_eval_gold_v2.py             # Gold v2 평가 (구 버전)
scripts/s2p_eval_text_match.py          # 텍스트 매칭 평가 (임시)
scripts/s2p_eval_v2_limited.py          # 제한적 커버리지 평가 (임시)
scripts/remap_s2p_output.py             # S2P 출력 재매핑 (임시)
```

**이유**: 여러 버전 존재로 혼동, 최신 accuracy/ 폴더의 evaluator 사용 권장

---

### 3️⃣ **머지/빌드 스크립트 - 임시** (5개)
데이터 통합 목적의 일회성 스크립트

```
scripts/merge_original_p2s_s2p.py       # 원본 P2S/S2P 머지 (완료됨)
scripts/merge_p2s_s2p.py                # P2S/S2P 머지 (중복)
scripts/remap_s2p_output.py             # S2P 재매핑 (임시)
scripts/collect_phrase_gold.py          # Gold 수집 (완료됨)
scripts/normalize_book_names.py         # 책 이름 정규화 (일회성)
```

**이유**: 일회성 데이터 처리용, 현재 데이터셋은 이미 정규화됨

---

### 4️⃣ **포장 준비용 임시 스크립트** (5개)
hyeonto 특정 패턴 검증용 임시 코드

```
scripts/hyeonto_build_datasets.py       # Hyeonto 데이터셋 구축 (완료됨)
scripts/hyeonto_build_xlsx.py           # Hyeonto XLSX 생성 (완료됨)
scripts/collect_phrase_gold.py          # Gold 수집 (중복)
scripts/rebuild_phrase_gold_v2.py       # Gold v2 재구축 (임시)
scripts/rebuild_all_reports_v6.py       # 모든 리포트 재구축 (임시)
```

**이유**: 한번만 실행된 데이터 처리, 결과물은 datasets/에 저장됨

---

## ⚠️ 재검토 대상 코드

### 분석 스크립트 중 일부 중복 (analyze_*.py)
유사한 목적의 스크립트가 여러 개 있습니다.

```
analyze_tense_from_translation_v6.py    ← v6 버전 유지
analyze_weight_sensitivity_v6.py        ← v6 버전 유지
analyze_tam_v6.py                       ← v6 버전 유지
validate_tamidence_v6.py                ← v6 버전 유지

OLD VERSIONS (should be removed):
- analyze_tense_from_translation.py (if exists)
- analyze_weight_sensitivity.py (if exists)
- analyze_tam.py (if exists)
```

---

## ✅ 유지 대상 코드

### 핵심 파이프라인
```
p2s/main.py                     # P2S 메인 파이프라인
s2p/main.py                     # S2P 메인 파이프라인
main.py                         # 통합 진입점
```

### 학습 스크립트
```
train_p2s_boundary.py           # P2S 경계 학습
train_p2s_crossattn_boundary.py # P2S Cross-Attention
train_s2p_alignment_dual_encoder.py # S2P Alignment
train_s2p_crossattn_boundary.py # S2P Cross-Attention
train_sentence_alignment.py     # 문장 정렬 학습
```

### 평가 및 검증
```
accuracy/p2s_evaluator.py       # P2S 평가 (메인)
accuracy/s2p_evaluator.py       # S2P 평가 (메인)
accuracy/pair_evaluator.py      # 쌍 평가
p2s_multitest_runner.py         # P2S 다중 테스트
quick_s2p_eval.py              # 빠른 S2P 평가
```

### 분석 및 시각화
```
analyze_*_v6.py                 # v6 최신 분석 스크립트
validate_*_v6.py                # v6 최신 검증 스크립트
visualize_*.py                  # 시각화 스크립트 (분석용)
cluster_*.py                    # 클러스터링 스크립트
```

### 유틸리티
```
sweep_p2s_boundary_threshold.py # P2S 임계값 스윕
tune_p2s_dp.py                  # P2S DP 최적화
optuna_s2p_dp.py                # Optuna 기반 S2P 최적화
profile_*.py                    # 프로파일링
detect_outliers_boundary.py     # 이상치 감지
diff_p2s_outputs.py             # 출력 비교
```

---

## 📋 정리 계획

### Phase 1: 확실한 삭제 (30개)
1. **Debug/Check 스크립트** (12개) → 휴지통
2. **평가 중복 버전** (8개) → 휴지통
3. **머지/빌드 임시** (5개) → 휴지통
4. **Hyeonto 데이터 빌드** (5개) → 휴지통

### Phase 2: 재검토 후 정리 (5개)
- 이전 버전 분석/검증 스크립트 확인 후 삭제

### Phase 3: 정리 완료 후
- 남은 ~45개 스크립트 기준으로 README.md 업데이트
- 폴더 구조 재조직 검토
  - `scripts/legacy/` 폴더 생성하여 덜 자주 사용되는 것 이동 (선택사항)

---

## 🎯 예상 효과

- **코드 베이스 명확성**: 80개 → 50개로 50% 감소
- **유지보수 용이성**: 불필요한 중복 제거
- **개발 속도**: 찾아야 할 스크립트 수 대폭 감소
- **학습곡선 개선**: 신규 개발자 온보딩 간소화

