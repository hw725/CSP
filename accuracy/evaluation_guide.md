# 정확도 평가 도구 사용 가이드

## 🎯 평가 도구 선택 가이드

### 행별 쌍 평가 (권장) - `row_pair_evaluator.py`
**사용 시기**: PA 시스템의 실제적 성능을 측정하고자 할 때
- ✅ 원문-번역문 쌍의 직접 비교
- ✅ 동적 정렬로 행 수 차이 자동 처리
- ✅ 세그먼트 식별자 오류 문제 회피
- ✅ 가장 현실적인 성능 지표 제공

### 기본 평가 - `accuracy_evaluator.py`
**사용 시기**: 세그먼트 단위 상세 분석이 필요할 때
- ⚠️ 데이터 구조가 완벽히 정렬된 경우에만 유효
- ✅ F1, 정밀도, 재현율 등 상세 지표 제공
- ✅ 세그먼트별 매칭 분석

### 원문 중심 평가 - `source_accuracy_evaluator.py`
**사용 시기**: 데이터 구조 문제 진단이 필요할 때
- ✅ 원문 불일치 문제 상세 분석
- ✅ PA 시스템의 구조적 문제 파악
- ⚠️ 번역문 평가에는 한계

## 📊 평가 지표 설명

### 행별 쌍 평가 지표 (row_pair_evaluator.py)

#### 1. 정렬 성공률 (Alignment Success Rate)
- **정의**: 매칭 가능한 행의 비율
- **계산**: 매칭된 행 수 / min(정답 행 수, 예측 행 수)
- **의미**: 데이터 매칭의 안정성 지표
- **목표**: 95% 이상

#### 2. 텍스트 유사도 (Text Similarity) 
- **정의**: 결합된 "원문|번역문" 텍스트 간 유사도
- **계산**: difflib.SequenceMatcher 사용
- **범위**: 0~1 (1이 완전 일치)
- **의미**: 실제적인 내용 일치 정도

#### 3. 최종 유사도 (Final Similarity)
- **정의**: 정렬 성공률과 텍스트 유사도의 가중평균
- **계산**: (정렬성공률 × 0.3) + (텍스트유사도 × 0.7)
- **의미**: 종합적 PA 성능 지표
- **목표**: 35% 이상

### 기본 평가 지표 (accuracy_evaluator.py)

#### 1. 완전 일치 (Exact Match)
- **정의**: 정답과 예측이 완전히 동일한 경우
- **기준**: 원문, 번역문, 세그먼트 수가 모두 정확히 일치
- **의미**: 가장 엄격한 정확도 기준

#### 2. 부분 일치 (Partial Match)  
- **정의**: 완전 일치는 아니지만 유사성이 있는 경우
- **계산 방식**: 
  - Jaccard 유사도 (교집합/합집합)
  - 텍스트 전체 유사도 (difflib.SequenceMatcher)
  - 세그먼트별 평균 유사도
- **임계값**: 50% 이상 유사하면 부분 일치로 판정

#### 3. 원문 불일치 처리
- **포함 정책**: 원문 불일치 문장도 평가 대상에 포함
- **이유**: 전체 시스템 성능의 완전한 평가를 위함
- **영향**: 원문 불일치 시 해당 문장의 정확도는 0점으로 처리

#### 4. F1 점수
- **정의**: 정밀도(Precision)와 재현율(Recall)의 조화평균
- **용도**: 불균형 데이터에서 균형잡힌 성능 평가

## 🔍 결과 해석 가이드

### 행별 쌍 평가 결과 해석
- **정렬 성공률 99.5%**: 매우 안정적인 데이터 매칭
- **텍스트 유사도 31%**: 실용적 수준의 내용 일치
- **최종 유사도 38.3%**: 양호한 PA 시스템 성능
- **행 수 차이 < 1%**: 데이터 손실 최소화

### 성능 등급 기준
#### 🟢 우수 (Excellent)
- 정렬 성공률: 98% 이상
- 텍스트 유사도: 40% 이상  
- 최종 유사도: 45% 이상

#### 🟡 양호 (Good)
- 정렬 성공률: 95% 이상
- 텍스트 유사도: 30% 이상
- 최종 유사도: 35% 이상

#### 🟠 보통 (Fair)
- 정렬 성공률: 90% 이상
- 텍스트 유사도: 20% 이상
- 최종 유사도: 25% 이상

#### 🔴 개선 필요 (Needs Improvement)
- 정렬 성공률: 90% 미만
- 텍스트 유사도: 20% 미만
- 최종 유사도: 25% 미만

### 기본 평가 결과 해석
- **완전 일치율 30% 이상**: 우수한 성능
- **부분 일치율 80% 이상**: 실용적 수준
- **원문 불일치 1% 미만**: 안정적 데이터 처리

## 📈 사용 방법

### 행별 쌍 평가 실행 (권장)
```python
from row_pair_evaluator import RowPairAccuracyEvaluator

evaluator = RowPairAccuracyEvaluator()
results = evaluator.evaluate_accuracy(
    ground_truth_file="관자3_문장병렬.xlsx",
    prediction_file="output_pa.xlsx"
)
evaluator.print_results(results)
evaluator.save_results(results, "accuracy_results.xlsx")
```

### 기본 정확도 평가
```python
from accuracy_evaluator import AccuracyEvaluator

evaluator = AccuracyEvaluator()
results = evaluator.evaluate_accuracy(
    ground_truth_file="관자1_구병렬.xlsx",
    prediction_file="output01.xlsx"
)
evaluator.print_detailed_results(results)
evaluator.save_results(results, "accuracy_results.xlsx")
```

### 고급 설정
```python
# 행별 쌍 평가 - 사용자 정의 임계값
evaluator = RowPairAccuracyEvaluator(similarity_threshold=0.3)

# 기본 평가 - 사용자 정의 임계값  
evaluator = AccuracyEvaluator(similarity_threshold=0.6)

# 특정 컬럼 지정
results = evaluator.evaluate_accuracy(
    ground_truth_file="ground_truth.xlsx",
    prediction_file="prediction.xlsx",
    gt_source_col="원문",
    gt_target_col="번역문", 
    pred_source_col="원문",
    pred_target_col="번역문"
)
```

## 📝 결과 파일 구성

### 행별 쌍 평가 Excel 출력
1. **Summary**: 전체 요약 통계
   - 정답/예측 행 수, 매칭 통계
   - 정렬 성공률, 텍스트 유사도, 최종 유사도
   
2. **Row_Details**: 행별 상세 결과
   - `row_index`: 행 번호
   - `ground_truth_text`: 정답 결합 텍스트 ("원문|번역문")
   - `prediction_text`: 예측 결합 텍스트
   - `text_similarity`: 텍스트 유사도 점수 (0~1)
   - `is_aligned`: 정렬 성공 여부

### 기본 평가 Excel 출력  
1. **Summary**: 전체 요약 통계
2. **Results**: 문장별 상세 결과
3. **Source_Mismatches**: 원문 불일치 문장 목록
4. **Execution_Log**: 실행 과정 로그

### 주요 지표 설명
#### 행별 쌍 평가
- `alignment_success_rate`: 정렬 성공률 (0~1)
- `text_similarity`: 평균 텍스트 유사도 (0~1)
- `final_similarity`: 최종 유사도 점수 (0~1)
- `matched_rows`: 매칭된 행 수
- `total_ground_truth`: 정답 총 행 수
- `total_prediction`: 예측 총 행 수

#### 기본 평가
- `exact_match`: 완전 일치 여부 (1/0)
- `partial_match`: 부분 일치 점수 (0~1)
- `f1_score`: F1 점수 (0~1)
- `source_text_match`: 원문 텍스트 일치율 (0~1)
- `target_text_match`: 번역문 텍스트 일치율 (0~1)
- `matched_pairs`: 매칭된 세그먼트 쌍 수
- `correct_translation_pairs`: 올바른 번역 쌍 수

## ⚠️ 주의사항 및 문제 해결

### 데이터 준비
#### 행별 쌍 평가
- Excel 파일의 첫 번째 시트 사용
- 필수 컬럼: 원문, 번역문 (문장식별자 선택사항)
- 행 수 차이 자동 처리

#### 기본 평가
- Excel 파일의 첫 번째 시트 사용
- 문장식별자 컬럼 필수 (자동 감지)
- 원문, 번역문 컬럼 필수

### 성능 고려사항
- **대용량 파일**: 메모리 사용량 주의 (1만 행 이상)
- **처리 시간**: 복잡한 유사도 계산으로 시간 소요
- **동적 정렬**: 행 수 차이가 클 경우 처리 시간 증가

### 데이터 구조 문제 해결
#### PA 시스템 식별자 오류
**증상**: 
- 기본 평가에서 97% 번역문 불일치
- source_accuracy_evaluator에서 높은 원문 불일치율

**원인**: 
- PA 시스템의 세그먼트 분할 과정에서 문장 식별자 오류 할당
- 세그먼트화된 데이터와 원본 데이터 간 식별자 불일치

**해결책**:
- 행별 쌍 평가 도구 사용 (`row_pair_evaluator.py`)
- 문장 식별자 기반 매칭 대신 순서 기반 직접 비교

#### 행 수 불일치 문제
**증상**: 정답과 예측 파일의 행 수 차이
**해결**: 동적 정렬 알고리즘 자동 적용
- 최적 매칭 찾기
- 건너뛰기 로직 적용
- 99.5% 이상 정렬 성공률 달성

### 결과 해석 시 주의점
- **행별 쌍 평가**: 가장 현실적이고 신뢰할 수 있는 지표
- **기본 평가**: 데이터 구조가 완벽할 때만 의미 있음
- **텍스트 유사도 30%**: PA 시스템으로는 실용적 수준
- **완전 일치율 낮음**: 자연스러운 현상, 부분 일치로 보완
- **원문 불일치**: 데이터 품질 문제를 나타내는 중요 지표

### 개발자 권장사항
1. **1차 평가**: `row_pair_evaluator.py` 사용하여 전체적 성능 파악
2. **2차 분석**: 필요시 `source_accuracy_evaluator.py`로 구조적 문제 진단  
3. **상세 분석**: 데이터가 완벽할 때만 `accuracy_evaluator.py` 사용
4. **지속적 모니터링**: 정렬 성공률, 텍스트 유사도 추이 관찰
