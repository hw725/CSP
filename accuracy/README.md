# CSP 정확도 평가 가이드 (accuracy)

이 디렉토리는 SA/PA 산출물의 품질을 정량 평가하는 스크립트 모음입니다. Docker 환경 기준으로 설명합니다.
기준 환경: 컨테이너 이름=csp-workspace, 컨테이너 내 루트 경로=/workspace

## 구성
- accuracy_evaluator.py: 세그먼트 기반 정확도(F1, 부분/완전 일치 등)
 
- compute_thresholds.py: 경험적 기준 산출용 보조 스크립트(mean/P50/P75/P90)
- thresholds_config.py: 프로젝트별 임계값 정의(최소/권장/상위)
- sa01.xlsx / pa03.xlsx 등: 예시/최근 결과 파일(있을 수 있음)

## 입력 형식
- 필수 컬럼: 문장식별자, 원문, 번역문
- 선택 컬럼: 구식별자(세그먼트 단위 평가 시 활용)
- 권장 포맷: Excel(xlsx)

예시(정답)
```
문장식별자	구식별자	원문	번역문
1	1	管子舊書는	≪管子≫의 옛 책은
1	2	凡三百八十九篇이니	모두 389편이었는데
```

예시(예측)
```
문장식별자	원문	번역문
1	管子舊書는	≪管子≫의 옛 책은
1	凡三百八十九篇이니	모두 389편이었는데 …
```

## 실행 가이드 (현재 경로 반영)

Docker(권장)
- PA 정확도 평가
```bash
docker exec -it csp-workspace /bin/bash -lc "cd /workspace/accuracy && python accuracy_evaluator.py pa03.xlsx ../pa/output_test.xlsx -o pa03_eval.xlsx"
```
- SA 정확도 평가
```bash
docker exec -it csp-workspace /bin/bash -lc "cd /workspace/accuracy && python accuracy_evaluator.py sa01.xlsx ../sa/output_test.xlsx -o sa01_eval.xlsx"
```
 

고급 옵션 예시(권장 조합)
```bash
# 행 자동 보정(상수 오프셋) + 넓은 탐색 범위 + 임계값 등급화 + 관대한 일치 판단 + CSV 자동 저장
docker exec -it csp-workspace /bin/bash -lc "cd /workspace/accuracy && \
  python accuracy_evaluator.py pa03.xlsx ../pa/output_test.xlsx \
  --project pa --unit row --row-auto-shift --row-auto-shift-range 60 \
  --ignore-space-punct -o pa03_eval.xlsx"
```

Windows(cmd) (경로에 맞게 조정)
```cmd
cd CSP\accuracy
python accuracy_evaluator.py pa03.xlsx ..\pa\output_test.xlsx -o pa03_eval.xlsx
python accuracy_evaluator.py sa01.xlsx ..\sa\output_test.xlsx -o sa01_eval.xlsx
rem 고급 옵션(동일 의미, Windows cmd)
python accuracy_evaluator.py pa03.xlsx ..\pa\output_test.xlsx --project pa --unit row --row-auto-shift --row-auto-shift-range 60 --ignore-space-punct -o pa03_eval.xlsx
```

### 단축 실행 스크립트(.cmd)

아래 스크립트는 csp-workspace 컨테이너 기준 경로로 실행합니다. 필요하면 스크립트 내 파일명만 바꿔 쓰세요.

```cmd
cd CSP\accuracy

rem PA: 행 단위(row) 평가 + CSV 저장
run_pa_row.cmd

rem PA: 문장 단위(sentence) 평가 + CSV 저장
run_pa_sentence.cmd

rem SA: 행 단위(row) 평가 + CSV 저장
run_sa_row.cmd

rem SA: 문장 단위(sentence) 평가 + CSV 저장
run_sa_sentence.cmd
```

파일 경로 커스터마이즈 팁
- pa03.xlsx / sa01.xlsx: 정답 파일명을 바꾸면 스크립트 안의 경로도 같이 수정
- /workspace/pa/output_test.xlsx, /workspace/sa/output.xlsx: 실제 산출물 파일명에 맞춰 수정

추가 안내
- 스크립트는 어떤 경로에서 실행해도 됩니다. run_*.cmd 내부에서 스크립트 폴더로 이동(pushd %~dp0) 후 실행합니다.
- --csv-dir를 생략해도 자동으로 CSV 폴더가 생성됩니다. 예: -o pa03_eval_row.xlsx → pa03_eval_row_csv/ 로 저장
- 생성물(기본): xlsx 1개 + 시트별 CSV(문장별_상세결과, 전체_요약, 원문불일치_상세, 실행_로그)
- 추가 생성물(전역 무결성/문자 분석):
  - 전역_불일치.csv: 원문/번역 전역 텍스트 길이, Δ, 문자 단위 diff(삽입/삭제/치환), 첫 차이 주변 텍스트
  - 전역_문자_빈도_차이_원문.csv, 전역_문자_빈도_차이_번역.csv: 전역 문자별 빈도 차이(Δ), 코드포인트, 공백류 이름 등
  - 메모: 전역 유사도 수치는 로그/요약에서 제외하고, 불일치 관련 항목만 기록됩니다.
  - 행/문장 단위 길이 지표: source_text_len_gt/pred/delta, target_text_len_gt/pred/delta 컬럼이 문장별_상세결과/전체_요약에 포함됩니다.

임계값 기반 등급화 사용법
- 프로젝트 유형 지정으로 임계값을 적용하고 등급을 출력합니다.

Docker 예시
```bash
docker exec -it csp-workspace /bin/bash -lc "cd /workspace/accuracy && python accuracy_evaluator.py pa03.xlsx ../pa/output_test.xlsx --project pa -o pa03_eval.xlsx"
docker exec -it csp-workspace /bin/bash -lc "cd /workspace/accuracy && python accuracy_evaluator.py sa01.xlsx ../sa/output_test.xlsx --project sa -o sa01_eval.xlsx"
```

Windows(cmd) 예시
```cmd
cd CSP\accuracy
python accuracy_evaluator.py pa03.xlsx ..\pa\output_test.xlsx --project pa -o pa03_eval.xlsx
python accuracy_evaluator.py sa01.xlsx ..\sa\output_test.xlsx --project sa -o sa01_eval.xlsx
```

출력 위치
- 콘솔: 프로젝트/단위, 지표별 값과 라벨(min/recommended/top/below), 전체 등급을 요약 표시
- 전체_요약 시트 및 CSV: grade_* 항목과 thresholds가 함께 기록됩니다.

### 전역 문자 빈도 차이 읽는 법(공백 이슈 포함)

- 생성물: `<출력>_csv/전역_문자_빈도_차이_원문.csv`, `<출력>_csv/전역_문자_빈도_차이_번역.csv`
- 컬럼: char, codepoint(U+XXXX), count_gt, count_pred, delta(=GT-Pred), char_name(공백류 표시)
- 해석 팁:
  - SPACE(U+0020)나 IDEOGRAPHIC SPACE(U+3000) 등 공백류의 Δ가 크면, 전역 길이 차이(Δ) 대부분이 공백 규칙 차이에서 비롯된 것일 수 있습니다.
  - 이 경우 텍스트 내용 손실이 아니라 포맷팅(띄어쓰기/줄바꿈/전각 공백) 차이일 가능성이 큽니다.
- 대응 방법:
  - 평가 시 공백/구두점 무시는 `--ignore-space-punct`를 사용하세요. 이는 text_match(원문/번역문 일치 판정)에만 적용되어 공백 기인 불일치 과대집계를 줄입니다.
  - 참고: `--ignore-space-punct`는 길이 Δ나 문자 diff(삽입/삭제/치환) 값 자체를 바꾸지 않습니다. 전역 무결성 수치는 원문 그대로의 텍스트 기준으로 유지됩니다.
  - 필요 시 전처리 단계에서 공백 규칙을 정규화(예: 연속 공백 축소, 전각/half 공백 통일)한 뒤 평가를 수행하세요.
  - SA/PA 파이프라인 산출물에 동일한 공백 규칙을 적용(토크나이저/후처리)하면 전역 Δ와 diff 잡음이 줄어듭니다.

### 평가 단위 선택 (--unit)

- 기본값: `row` (행 단위 1:1 비교)
- 문장 단위로 비교하려면: `sentence`

Docker 예시
```bash
docker exec -it csp-workspace /bin/bash -lc "cd /workspace/accuracy && python accuracy_evaluator.py pa03.xlsx ../pa/output_test.xlsx --unit row -o pa03_eval_row.xlsx"
docker exec -it csp-workspace /bin/bash -lc "cd /workspace/accuracy && python accuracy_evaluator.py sa01.xlsx ../sa/output_test.xlsx --unit sentence -o sa01_eval_sentence.xlsx"
```

Windows(cmd) 예시
```cmd
cd CSP\accuracy
python accuracy_evaluator.py pa03.xlsx ..\pa\output_test.xlsx --unit row -o pa03_eval_row.xlsx
python accuracy_evaluator.py sa01.xlsx ..\sa\output_test.xlsx --unit sentence -o sa01_eval_sentence.xlsx
```

## CLI 옵션 요약(accuracy_evaluator.py)

- --unit {row|sentence}
  - 평가 단위 선택. 기본값=row(행 1:1). sentence는 식별자 그룹 단위 평가.
- --project {pa|sa}
  - 프로젝트별 임계값을 적용해 요약/등급(grade_*)을 기록합니다.
- --brief
  - 콘솔 출력 최소화(핵심 지표만). 파일 저장 내용은 동일.
- --minimal-summary
  - 전체_요약(엑셀/CSV)에 핵심 지표만 저장(파일 용량/가독성 중시).
- --csv-dir <DIR>
  - 시트별 CSV 저장 경로. 미지정 시 -o 기준 자동 생성(<output_basename>_csv/).
- --row-auto-shift
  - 행 단위에서 시스템적 인덱스 오프셋을 자동 탐지·보정합니다(상수 shift). 로그에 best_shift/overlap/avg_sim/개선수 등을 요약합니다.
- --row-auto-shift-range <N>
  - 오프셋 탐지 범위(±N). 기본 50. PA에서 -3, +k 등 전형적 밀림이 있을 경우 60~100 권장.
- --ignore-space-punct
  - 원문/번역문 일치 여부(text_match, source/target_text_match) 판정 시 공백/구두점을 무시한 관대한 비교를 사용합니다. 유사도/부분일치 계산은 기존과 동일합니다.

해석 팁
- 전역 무결성: 길이 Δ와 문자 diff, 첫 차이 스니펫이 원인 파악에 유용합니다. 전역 유사도 수치는 요약/로그에서 제외됩니다.
- 행 자동 보정: constant shift로 해결되지 않는 경우(중간 삽입/삭제로 블록 드리프트)엔 부분적 블록 정렬이나 탐색 범위 확대가 필요할 수 있습니다.

## 지표 상세

세그먼트 평가(accuracy_evaluator.py)
- 완전 일치율 (Exact Match)
  - 정의: 정답 세그먼트 시퀀스와 예측 시퀀스가 길이/순서/내용까지 모두 동일한 경우 1, 아니면 0
  - 산출: 문장 단위의 0/1을 전체 문장에 대해 평균(매크로 평균)
- 세그먼트 수 일치율 (Segment Count Match)
  - 정의: 정답과 예측의 세그먼트 개수가 동일하면 1, 아니면 0
  - 산출: 문장 단위 0/1의 평균
- 텍스트 일치율 (Text Match)
  - 정의: 세그먼트를 결합한 전체 원문/번역문 텍스트가 정확히 동일하면 1, 아니면 0
  - 세부분류: 원문 일치율(Source Text Match), 번역문 일치율(Target Text Match)
  - 산출: 문장 단위 0/1의 평균
- 텍스트 유사도 (Text Similarity)
  - 정의: 전체 결합 텍스트 간 유사도. 기본은 difflib.SequenceMatcher 비율(0~1)
  - 세부분류: 원문 유사도(Source Text Similarity), 번역문 유사도(Target Text Similarity)
  - 산출: 문장 단위 유사도의 평균
- 부분 일치율 (Partial Match)
  - 원문 부분 일치(Source Partial): 아래 3요소의 산술평균
    1) 세트 유사도(Jaccard): 정답 원문 세그먼트 집합과 예측 원문 세그먼트 집합의 교집합/합집합
    2) 전체 텍스트 유사도: 결합 원문 텍스트의 SequenceMatcher 비율
    3) 세그먼트 평균 유사도: 각 정답 원문 세그먼트가 가지는 예측 원문 세그먼트와의 최대 유사도들의 평균
  - 번역문 부분 일치(Target Partial): 원문이 매칭된 쌍에서의 번역문 유사도 평균(0~1)
  - 최종 부분 일치(Partial Match): (Source Partial + Target Partial) / 2
- 원문 기준 매칭 통계
  - 매칭된 쌍 수: 원문 기준 정렬에서 성공적으로 쌍지어진 세그먼트 수
  - 올바른 번역 쌍 수: 매칭된 쌍 중 번역문이 기준 임계에 부합하거나 동일한 경우의 수
  - 번역문 정확도(Target Accuracy): 올바른 번역 쌍 수 / 매칭된 쌍 수
- Precision / Recall / F1
  - 원문 Precision = 매칭된 쌍 수 / 예측 세그먼트 수
  - 원문 Recall = 매칭된 쌍 수 / 정답 세그먼트 수
  - 원문 F1 = 2PR / (P+R) (P+R=0이면 0)
  - 번역문 Precision/Recall/F1은 원문 매칭된 쌍을 기반으로 번역문 일치 여부를 사용해 산출(구현에 따라 정확 정의가 다를 수 있음)
  - 보고값은 문장 단위 점수의 매크로 평균

집계 방식과 규칙
- 평균: 별도 언급이 없으면 문장 단위 매크로 평균을 사용
- 정규화: 공백/개행 제거 및 기본적인 구두점 제거가 적용될 수 있음(유사도 계산/매칭 시)
- 결측 처리: 빈 텍스트 또는 분모가 0인 경우 해당 지표는 0 처리
- 임계값: 번역문 정확 판단 시 임계 사용 시(예: 유사도 ≥ T) 프로젝트 설정을 따름

에지 케이스
- 빈 세그먼트(원문 또는 번역문 없음): 유사도 0, 일치 0으로 간주
- 중복 세그먼트 내용: 세트 기반(Jaccard)에서 중복은 하나로 간주됨
- 다대일/일대다 정렬: 원문 기준 매칭 통계는 가장 유사한 쪽으로만 매칭(중복 매칭 방지)
- 순서 뒤바뀜: Exact Match/Count Match는 불리하게 반영, Partial/유사도 지표는 일부 보완

## 해석 가이드(권장: SA/PA 분리)

임계값 표(행 단위, 현재 데이터 기반)

- SA (권장 지표: partial_match, target_avg_similarity)
  - 최소(P50): partial ≥ 0.885, target_avg_similarity ≥ 0.769
  - 권장(P75): partial ≥ 0.952, target_avg_similarity ≥ 0.905
  - 상위(P90): partial ≥ 1.000, target_avg_similarity ≥ 1.000

- PA (권장 지표: partial_match, target_avg_similarity)
  - 최소(P50): partial ≥ 0.10, target_avg_similarity ≥ 0.10
  - 권장(P75): partial ≥ 0.15, target_avg_similarity ≥ 0.19
  - 상위(P90): partial ≥ 0.21, target_avg_similarity ≥ 0.26

메모
- PA는 구조 차이로 Exact/Count/Text Match가 낮게 나올 수 있어 위 두 지표 중심으로 평가합니다.
- 기준은 현재 파일의 퍼센타일 기반 제안치이며 데이터가 바뀌면 `percentiles_from_csv.py`로 재산출하세요.

## 문제 해결
- 파일 미발견: 컨테이너 내부 경로(`/workspace/accuracy`)에서 실행
- 컬럼명 불일치: 컬럼명을 문장식별자/원문/번역문으로 정규화
- 길이 차이가 비정상적으로 큼: 문장식별자 유무/일치 여부 확인(없으면 인덱스 매칭으로 신뢰도 저하)
 

## 경험적 기준 산출(퍼센타일)
- 스크립트: `accuracy/compute_thresholds.py`
- CSV 직접 산출: `accuracy/percentiles_from_csv.py` (문장별_상세결과.csv에서 바로 요약)
- 기본 대상 파일
  - PA: 정답=`/workspace/accuracy/pa03.xlsx`, 예측=`/workspace/pa/output_test.xlsx`
  - SA: 정답=`/workspace/accuracy/sa01.xlsx`, 예측=`/workspace/sa/output_test.xlsx`
- 실행 예시(Docker)
```bash
docker exec -it csp-workspace /bin/bash -lc "python /workspace/accuracy/compute_thresholds.py"
```
- Windows(cmd)
```cmd
cd CSP\accuracy
python compute_thresholds.py --base .
rem 또는 경로 직접 지정
python compute_thresholds.py --pa-gt CSP/accuracy/pa03.xlsx --pa-pred CSP/pa/output_test.xlsx --sa-gt CSP/accuracy/sa01.xlsx --sa-pred CSP/sa/output_test.xlsx
```
- 출력: 각 지표의 mean/P50/P75/P90. 이 값을 바탕으로 README의 권장 기준을 보정합니다.
 - 비고: 평가/손실 본판이 아닌 보조 도구이며, 파일 경로는 인자로 변경 가능합니다.

## 비고
- Docker 중심. Poetry 관련 내용 제거.
- 입력 파일 인코딩/컬럼 표준화를 권장합니다.
