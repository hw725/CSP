# CSP 정확도 평가 (accuracy)

**중앙 집중식 PA/SA 평가 모듈**

이 디렉토리는 PA(Paragraph Alignment)와 SA(Sentence Alignment) 산출물의 품질을 정량 평가하는 스크립트 모음입니다.

## 파일 구조

```
accuracy/
  ├── pa_evaluator.py          # PA 평가 (문단 병렬, 원문 경계 F1)
  ├── sa_evaluator.py          # SA 평가 (구병렬, 세그먼트 매칭 F1)
  ├── compute_thresholds.py    # 경험적 기준 산출 (P50/P75/P90)
  └── thresholds_config.py     # 프로젝트별 임계값 정의
```

## PA 평가 (pa_evaluator.py)

### 핵심 메트릭
- **micro_f1_tgt_exact**: 번역문이 일치하는 문장들의 원문 경계 F1 (조화평균)
  - 번역문 exact match한 문장들만 추출
  - 그 문장들의 원문 경계 위치(boundary positions)를 비교
  - TP/FP/FN 계산 후 F1 = 2PR/(P+R)
- **mean_similarity**: 번역문 일치 문장에서 원문 유사도 평균

### 사용법

```bash
# Docker (권장)
docker compose run --rm csp python accuracy/pa_evaluator.py \
  --input pa/output_test.xlsx \
  --gold datasets/pa/test.csv

# 직접 실행
python accuracy/pa_evaluator.py \
  --input pa/output_test.xlsx \
  --gold datasets/pa/test.csv

# 하위 호환성 (루트의 integrity_report.py는 wrapper)
python integrity_report.py \
  --input pa/output_test.xlsx \
  --gold datasets/pa/test.csv
```

### 주요 옵션
- `--input`: PA 출력 파일 (.xlsx/.csv)
- `--gold`: 정답 파일 (문장 단위, columns: 문단식별자,문장식별자,원문,번역문,book_name)
- `--keys-from`: 특정 키만 평가 (문단식별자, book_name 포함 파일)
- `--extract`: 정답 부분 추출 모드

## SA 평가 (sa_evaluator.py)

### 핵심 메트릭
- **source_f1_score**: 원문 매칭 F1 (조화평균)
  - Precision = 매칭된 쌍 수 / 예측 세그먼트 수
  - Recall = 매칭된 쌍 수 / 정답 세그먼트 수
  - F1 = 2PR/(P+R)
- **target_f1_score**: 번역문 정확도 F1 (조화평균)
- **f1_score**: 전체 F1 (source_f1과 target_f1의 조화평균)
- **partial_match**: 부분 일치율 (Jaccard + 유사도 평균)
- **target_avg_similarity**: 번역문 평균 유사도

### 사용법

```bash
# Docker (권장)
docker compose run --rm csp python accuracy/sa_evaluator.py \
  accuracy/sa01.xlsx \
  sa/output_test.xlsx \
  -o sa01_eval.xlsx

# 직접 실행
python accuracy/sa_evaluator.py \
  ground_truth.xlsx \
  prediction.xlsx \
  -o evaluation.xlsx
```

### 주요 옵션
- `--project {pa|sa}`: 프로젝트별 임계값 적용
- `--unit {row|sentence}`: 평가 단위 (기본: row)
- `--ignore-space-punct`: 공백/구두점 무시 (관대한 일치)
- `--row-auto-shift`: 행 자동 보정 (인덱스 오프셋 탐지)
- `--csv-dir`: 시트별 CSV 저장 경로
- `-o`: 출력 파일 (.xlsx)

## F1 계산 방식

### PA F1 (Boundary-based)
```
1. 번역문이 GT와 exact match한 문장들만 필터링
2. 원문을 concat하여 하나의 텍스트로 만듦
3. 경계 위치 집합 계산: {누적길이1, 누적길이2, ...}
4. TP = pred ∩ gold, FP = pred - gold, FN = gold - pred
5. F1 = 2 × (TP/(TP+FP)) × (TP/(TP+FN)) / ((TP/(TP+FP)) + (TP/(TP+FN)))
```

### SA F1 (Segment-based)
```
1. 원문 세그먼트 매칭 (유사도 기반)
2. Source Precision = 매칭 쌍 / 예측 세그먼트
3. Source Recall = 매칭 쌍 / 정답 세그먼트
4. Source F1 = 2PR/(P+R)
5. Target F1: 매칭된 쌍에서 번역문 정확도로 동일 계산
6. 전체 F1 = 2 × Source_F1 × Target_F1 / (Source_F1 + Target_F1)
```

**중요**: 모든 F1은 **조화평균**입니다. 산술평균이 아닙니다.

## 임계값 (Thresholds)

### PA (문단 병렬)
| 등급 | partial_match | target_avg_similarity |
|------|---------------|----------------------|
| 최소(P50) | ≥ 0.10 | ≥ 0.10 |
| 권장(P75) | ≥ 0.15 | ≥ 0.19 |
| 상위(P90) | ≥ 0.21 | ≥ 0.26 |

### SA (구병렬)
| 등급 | partial_match | target_avg_similarity |
|------|---------------|----------------------|
| 최소(P50) | ≥ 0.885 | ≥ 0.769 |
| 권장(P75) | ≥ 0.952 | ≥ 0.905 |
| 상위(P90) | ≥ 1.000 | ≥ 1.000 |

## 경험적 기준 산출

```bash
python accuracy/compute_thresholds.py \
  --pa-gt accuracy/pa03.xlsx \
  --pa-pred pa/output_test.xlsx \
  --sa-gt accuracy/sa01.xlsx \
  --sa-pred sa/output_test.xlsx
```

출력: 각 지표의 mean/P50/P75/P90

## 하위 호환성

**루트의 integrity_report.py는 wrapper**입니다:
```python
# 이 두 명령은 동일합니다
python integrity_report.py --input ... --gold ...
python accuracy/pa_evaluator.py --input ... --gold ...
```

## 아키텍처 원칙

1. **중앙 집중식 평가**: 모든 평가 로직은 accuracy/ 폴더에 집중
2. **단일 책임**: PA와 SA 평가자는 각각 독립적
3. **재사용**: Grid Search 등 다른 스크립트는 평가자를 호출만 함
4. **검증**: 평가 로직 중복 구현 금지 (버그 방지)

## 문제 해결

- **ModuleNotFoundError**: accuracy/ 폴더가 sys.path에 있는지 확인
- **F1이 0**: 번역문 일치 문장이 없거나 경계가 모두 불일치
- **하위 호환성 문제**: integrity_report.py wrapper 확인

## 참고

- PA는 **문장 경계 정확도** 중심 평가
- SA는 **세그먼트 매칭 정확도** 중심 평가
- 평가 단위와 목적이 다르므로 별도 평가자 사용
