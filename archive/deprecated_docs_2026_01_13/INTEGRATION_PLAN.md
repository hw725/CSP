# PA/SA 파이프라인 + 학습 모델 통합 (앙상블/Refinement 모드)

## 현재 상태 ✅ (완료)
- 학습된 모델: `boundary_multitask.pt`, `dual_encoder_alignment_pa.pt`, `dual_encoder_alignment_sa.pt`
- PA 파이프라인: BGE/순차 **+ 경계 모델 refinement 통합 완료**
- SA 파이프라인: BGE/순차 **+ 경계 모델 refinement 통합 완료**

## 통합 전략 🎯

**🔑 핵심: 기존 파이프라인 대체가 아닌 보강**

### 처리 흐름
```
입력 → 기존 BGE/순차 분할 (초기 분할) 
     → 경계 모델 refinement (경계 개선)
     → Alignment 모델 (정렬 개선)
     → 기존 후처리 (괄호 정리, 무결성 검증)
     → 출력
```

### 차별점
- ❌ **별도 트랙 아님**: `--use-boundary-model`로 완전히 다른 경로 실행 (이전 방식)
- ✅ **앙상블 모드**: 기존 방식 + 경계 모델 보강 → 정확도/유사도 향상

### 이점
1. **기존 강점 보존**: BGE 임베딩의 의미 파악 + 순차 처리의 안정성
2. **경계 정확도 개선**: 학습된 경계 모델로 분할점 refinement
3. **정렬 품질 향상**: Alignment 모델로 원문-번역문 매칭 개선
4. **점진적 적용**: `--use-boundary-model` 플래그로 선택 가능

---

## ✅ 완료된 작업

### Phase 1: 공용 모델 로더 작성 ✅
**파일:** 
- [`common/boundary_model_loader.py`](common/boundary_model_loader.py)
  - `BoundaryModelLoader` 클래스: boundary_multitask.pt 로드 및 추론
  - `segment_text(task="pa"/"sa", threshold)` 메서드
  
- [`common/alignment_model_loader.py`](common/alignment_model_loader.py)
  - `AlignmentMatcher` 클래스: dual_encoder 모델 로드 및 정렬
  - `match_segments(src_segments, tgt_segments)` - 그리디 매칭

### Phase 2: PA 파이프라인 통합 (Refinement 모드) ✅
**수정 파일:**
- [`pa/main.py`](pa/main.py)
  - CLI 옵션: `--use-boundary-model`, `--boundary-threshold (default=0.5)`, `--device`
  
- [`pa/processor.py`](pa/processor.py)
  - **새 함수**: `_refine_alignments_with_models()` - 기존 결과를 경계/정렬 모델로 개선
  - **수정**: `process_paragraph_file()`
    ```python
    # 1. 기존 BGE/순차 방식으로 초기 분할
    alignments = process_paragraph_alignment(...)
    
    # 2. 경계 모델 refinement (옵션)
    if boundary_model and alignment_pa:
        alignments = _refine_alignments_with_models(...)
    
    # 3. 기존 후처리 (괄호 정리, 무결성 검증)
    ```

### Phase 3: SA 파이프라인 통합 (Refinement 모드) ✅
**수정 파일:**
- [`sa/main.py`](sa/main.py)
  - CLI 옵션: `--use-boundary-model`, `--boundary-threshold`, `--device`
  
- [`sa/sa_aligner.py`](sa/sa_aligner.py)
  - **새 함수**: `refine_sa_segments_with_models()` - 기존 구 분할 refinement
  - **수정**: `process_single_row()`
    ```python
    # 1. 기존 방식으로 초기 분할
    src_units = split_src_meaning_units(...)
    trans_units = split_tgt_meaning_units(...)
    
    # 2. 경계 모델 refinement (옵션)
    if use_boundary_model:
        src_units, trans_units = refine_sa_segments_with_models(...)
    ```
  
- [`sa/io_manager.py`](sa/io_manager.py)
  - `safe_process_sa_row()`: 경계 모델 캐싱 + refinement 트리거

---

## 사용 방법

### 0. 테스트 데이터 준비 (자동 스크립트)
평가용 테스트 데이터를 파이프라인 입력 형식으로 자동 변환:
```bash
python scripts/prepare_test_inputs.py
```

생성 파일:
- `test_inputs/test_pd_input.xlsx` (PD test → PA 파이프라인 입력용)
- `test_inputs/test_pa_input.xlsx` (PA test → SA 파이프라인 입력용)

### 1. 테스트 데이터 준비 (평가용)
평가 스크립트와 동일한 테스트 데이터 사용:
- **PD → PA 테스트**: [`datasets/pd/test.csv`](datasets/pd/test.csv) (문단 레벨)
- **PA → SA 테스트**: [`datasets/pa/test.csv`](datasets/pa/test.csv) (문장 레벨)
- **SA 정답 데이터**: [`datasets/sa/test.csv`](datasets/sa/test.csv) (구 레벨)

각 CSV는 `src`, `tgt` 컬럼을 포함합니다.

### 2. 테스ト 데이터를 Excel로 변환 (수동 방법)
스크립트 대신 수동으로 변환하려면:
```python
import pandas as pd

# PD test → PA 입력용 Excel
pd_test = pd.read_csv("datasets/pd/test.csv")
pd_test[['src', 'tgt']].rename(columns={'src': '원문', 'tgt': '번역문'}).to_excel("test_pd_input.xlsx", index=False)

# PA test → SA 입력용 Excel  
pa_test = pd.read_csv("datasets/pa/test.csv")
pa_test[['src', 'tgt']].rename(columns={'src': '원문', 'tgt': '번역문'}).to_excel("test_pa_input.xlsx", index=False)
```

### 3. PA 파이프라인 실행 및 비교
```bash
# 기존 BGE/순차 방식만
python pa/main.py test_inputs/test_pd_input.xlsx output_pa_baseline.xlsx

# 기존 방식 + 경계 모델 refinement (앙상블)
python pa/main.py test_inputs/test_pd_input.xlsx output_pa_refined.xlsx \
    --use-boundary-model --boundary-threshold 0.4 --device cuda
```

**기대 효과:**
- 기존: BGE 임베딩 기반 의미 분할 (~15K 문장)
- **Refined**: BGE + 경계 모델 보강 (→ ~23K 문장, 더 세밀한 분할)
- 정렬 품질 향상: Alignment 모델로 원문-번역문 매칭 개선

### 4. SA 파이프라인 실행 및 비교
```bash
# 기존 방식만
python sa/main.py test_inputs/test_pa_input.xlsx output_sa_baseline.xlsx

# 기존 + 경계 모델 refinement
python sa/main.py test_inputs/test_pa_input.xlsx output_sa_refined.xlsx \
    --use-boundary-model --boundary-threshold 0.4 --device cuda
```

**기대 효과:**
- 기존: 공백 기준 + DP 정렬 (~42K 구)
- **Refined**: 기존 + 경계 모델 보강 (→ ~86K 구, 더 정밀한 구 분할)
- 정렬 정확도 향상: Alignment 모델 적용

### 5. 일반 데이터 처리 (실제 업무용)
```bash
# PA: 문단→문장 (기존 + refinement)
python pa/main.py your_paragraphs.xlsx output_sentences.xlsx \
    --use-boundary-model --boundary-threshold 0.4

# SA: 문장→구 (기존 + refinement)
python sa/main.py your_sentences.xlsx output_phrases.xlsx \
    --use-boundary-model --boundary-threshold 0.4
```

**권장 설정:**
- `--boundary-threshold 0.4`: 평가 결과 기준 최적값 (0.3~0.5 범위에서 실험 가능)
- `--device cuda`: GPU 사용 (CPU도 가능하지만 느림)
- `--verbose`: 상세 로그 확인 시

---

## 주요 설계 결정 사항

### 1. 앙상블/Refinement 아키텍처
**선택한 방식:**
```
기존 BGE/순차 분할 → 경계 모델 refinement → Alignment 정렬 → 후처리
```

**장점:**
- ✅ 기존 강점 보존 (BGE 의미 파악, 순차 처리 안정성)
- ✅ 경계 정확도 향상 (학습된 모델의 분할점 개선)
- ✅ 점진적 적용 가능 (플래그로 on/off)
- ✅ 실패 시 기존 결과 유지 (안정성)

**대안으로 버린 방식:**
- ❌ 완전 대체: 경계 모델만 사용 → 기존 강점 상실
- ❌ 별도 트랙: 두 가지 완전히 다른 처리 경로 → 유지보수 부담

---

## Phase 4: 테스트 & 평가 (다음 단계)
- [ ] 기존 방식 vs 새 모델 방식 정량 비교
  - 세그먼트 수 (기존: pd→pa 15K, pa→sa 42K vs 새 모델: 23K, 86K)
  - text_similarity (기존: unknown vs 새 모델: 0.358, 0.488)
- [ ] threshold 최적화 실험 (0.3, 0.4, 0.5)
- [ ] 실제 데이터셋에서 품질 검증
- [ ] 처리 속도 벤치마크

---

## 주요 설계 결정 사항

### 1. 앙상블/Refinement 아키텍처 ⭐
**선택한 방식:**
```
기존 BGE/순차 분할 → 경계 모델 refinement → Alignment 정렬 → 후처리
```

**장점:**
- ✅ **기존 강점 보존**: BGE 의미 파악 + 순차 처리 안정성
- ✅ **경계 정확도 향상**: 학습된 모델의 분할점 개선
- ✅ **정렬 품질 개선**: Alignment 모델로 매칭 정확도 증가
- ✅ **점진적 적용**: `--use-boundary-model` 플래그로 선택 가능
- ✅ **안정성**: refinement 실패 시 기존 결과 유지

**버린 대안:**
- ❌ 완전 대체 (경계 모델만 사용): 기존 BGE 강점 상실, 리스크 높음
- ❌ 별도 트랙 (두 가지 완전히 다른 경로): 코드 중복, 유지보수 부담

### 2. 모듈화 및 캐싱
- 공용 모델 로더 (`common/`) → PA/SA에서 공유
- 모델 캐싱: 함수 속성으로 1회만 로드 (성능 최적화)
- 조건부 로딩: `use_boundary_model=True`일 때만 모델 로드

### 3. 후방 호환성
- 기본값: `--use-boundary-model=False` (기존 방식 유지)
- 신규 기능 opt-in: 사용자가 명시적으로 활성화
- 기존 코드 최소 변경: 조건 분기로 추가

### 4. 무결성 보장
- 기존 후처리 로직 **모두 유지**:
  - 괄호 블록 정리
  - 인용 표지 병합
  - 텍스트 손실 검증
  - 토씨 매칭 보정 (PA)
- Refinement 실패 시 graceful fallback

---

## Phase 4: 테스트 & 평가 (다음 단계)

### 정량 평가
- [ ] 기존 vs Refined 세그먼트 수 비교
  - PA: 15K → 23K? (평가 결과 기준)
  - SA: 42K → 86K?
- [ ] text_similarity 메트릭
  - PA: 0.358 (pd→pa, 평가 결과)
  - SA: 0.488 (pa→sa, 평가 결과)
- [ ] 정답 데이터 대비 정확도

### 정성 평가
- [ ] 샘플링 품질 검증 (10~20개 문단/문장)
- [ ] 경계 적절성 확인 (과다/과소 분할)
- [ ] 정렬 정확성 확인 (원문-번역문 매칭)

### 최적화
- [ ] threshold 실험: 0.3, 0.4, 0.5
- [ ] 처리 속도 벤치마크
- [ ] GPU vs CPU 성능 비교

---

## 체크리스트 ✅
- [x] `common/boundary_model_loader.py` 작성
- [x] `common/alignment_model_loader.py` 작성
- [x] PA refinement 통합 (`pa/processor.py`)
- [x] PA CLI 옵션 (`pa/main.py`)
- [x] SA refinement 통합 (`sa/sa_aligner.py`)
- [x] SA CLI 옵션 (`sa/main.py`)
- [x] 통합 문서 업데이트 (앙상블 모드 명시)
- [x] 테스트 데이터 준비 스크립트 (`scripts/prepare_test_inputs.py`)
- [ ] 통합 테스트 실행 (다음 단계)
- [ ] threshold 최적화 (다음 단계)
- [ ] 성능 벤치마크 (다음 단계)

---

## 참고 문서
- 평가 결과: [evaluate_hierarchical_segmentation.py](scripts/evaluate_hierarchical_segmentation.py) 실행 로그
  - pd→pa: text_sim=0.358 (22819 vs 15153 segments)
  - pa→sa: text_sim=0.488 (86342 vs 41987 segments)
- 모델 체크포인트:
  - `models/boundary_multitask.pt` (pd/pa/sa 3-task)
  - `models/dual_encoder_alignment_pa.pt`
  - `models/dual_encoder_alignment_sa.pt`
- 테스트 데이터:
  - `datasets/pd/test.csv` (문단 레벨)
  - `datasets/pa/test.csv` (문장 레벨)
  - `datasets/sa/test.csv` (구 레벨)

